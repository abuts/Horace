"""Moderator pulse model class (IX_moderator).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_moderator/
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.pdf_table import (
    _parse_rand_shape,
    pdf_table,
)


def ikcarp_param_convert(
    pp: Union[List[float], np.ndarray],
    ei: Union[float, List[float], np.ndarray],
) -> Tuple[Union[float, np.ndarray], float, Union[float, np.ndarray]]:
    """Convert Ikeda-Carpenter energy-dependent parameters to (tauf, taus, R).

    Parameters
    ----------
    pp : sequence of 3 floats
        ``[tauf_grad, taus, E0]`` where:
        - ``tauf = 1166.47 * pp[0] / sqrt(ei)`` (in microseconds, ``ei`` in meV)
        - ``taus = pp[1]`` (in microseconds)
        - ``R = exp(-ei / pp[2])`` (``ei`` and ``pp[2]`` in meV)
    ei : float or array-like
        Incident energy or array of incident energies (meV).

    Returns
    -------
    tauf : float or np.ndarray
        Fast decay time (microseconds).
    taus : float
        Slow decay time (microseconds).
    R : float or np.ndarray
        Weight of storage term (``0 <= R <= 1``).
    """
    pp_arr = np.asarray(pp, dtype=float).ravel()
    if pp_arr.size != 3:
        raise ValueError("HEBERT:IX_moderator:invalid_argument: Parameter array must have 3 elements.")
    if np.any(pp_arr < 0):
        raise ValueError("HEBERT:IX_moderator:invalid_argument: Parameters must be non-negative.")

    ei_is_scalar = np.ndim(ei) == 0 or (isinstance(ei, np.ndarray) and ei.size == 1 and ei.ndim <= 1)
    ei_arr = np.asarray(ei, dtype=float)
    if np.any(ei_arr <= 0):
        raise ValueError(
            "HEBERT:IX_moderator:invalid_argument: "
            "Moderator energy must be greater than zero to compute pulse parameters for pulse model 'ikcarp_param'."
        )

    tauf = 1166.47 * pp_arr[0] / np.sqrt(ei_arr)
    taus = float(pp_arr[1])
    if pp_arr[2] > 0:
        R = np.exp(-ei_arr / pp_arr[2])
    else:
        R = np.zeros_like(ei_arr, dtype=float)

    if ei_is_scalar and np.ndim(ei) == 0:
        return float(tauf), taus, float(R)
    return tauf, taus, R


def ikcarp(
    t: Union[float, np.ndarray],
    tauf: float,
    taus: float,
    R: float,
) -> Union[float, np.ndarray]:
    """Normalized Ikeda-Carpenter function at times ``t`` (microseconds)."""
    is_scalar = np.ndim(t) == 0
    t_arr = np.asarray(t, dtype=float)
    y = np.zeros_like(t_arr, dtype=float)
    pos = t_arr > 0

    if not np.any(pos):
        return float(y) if is_scalar else y

    tp = t_arr[pos]
    if tauf > 0:
        x = tp / tauf
        y_chisqr = (0.5 / tauf) * (x ** 2) * np.exp(-x)
        if R > 0 and taus > 0:
            r = tauf / taus
            if abs(1.0 - r) > 1e-5:
                expx = np.exp(-x)
                exprx = np.exp(-r * x)
                y_conv = (1.0 / (taus * (1.0 - r) ** 3)) * (
                    exprx - expx * (1.0 + (1.0 - r) * x + 0.5 * ((1.0 - r) ** 2) * (x ** 2))
                )
            else:
                eps = 1.0 - r
                expx = np.exp(-x)
                y_conv = (1.0 / (6.0 * taus)) * (x ** 3) * expx * (
                    1.0 - 0.25 * (eps * x) + (1.0 / 20.0) * (eps * x) ** 2 - (1.0 / 120.0) * (eps * x) ** 3
                )
            yp = (1.0 - R) * y_chisqr + R * y_conv
        else:
            yp = y_chisqr
    elif R > 0 and taus > 0:
        yp = (1.0 / taus) * np.exp(-tp / taus)
    else:
        yp = np.zeros_like(tp)

    y[pos] = np.maximum(0.0, yp)
    return float(y) if is_scalar else y


def ikcarp_Area(
    t: Union[float, np.ndarray],
    tauf: float,
    taus: float,
    R: float,
) -> Union[float, np.ndarray]:
    """Integrated Ikeda-Carpenter function from 0 to ``t``."""
    is_scalar = np.ndim(t) == 0
    t_arr = np.asarray(t, dtype=float)
    A = np.zeros_like(t_arr, dtype=float)
    pos = t_arr > 0

    if not np.any(pos):
        return float(A) if is_scalar else A

    tp = t_arr[pos]
    if tauf > 0:
        x = tp / tauf
        expx = np.exp(-x)
        A_chisqr = 1.0 - (1.0 + x + 0.5 * (x ** 2)) * expx
        if R > 0 and taus > 0:
            r = tauf / taus
            if abs(1.0 - r) > 1e-5:
                exprx = np.exp(-r * x)
                A_conv = 1.0 - (
                    exprx - (r * expx) * (
                        (3.0 - 3.0 * r + r ** 2)
                        + (1.0 - r) * (2.0 - r) * x
                        + 0.5 * ((1.0 - r) ** 2) * (x ** 2)
                    )
                ) / ((1.0 - r) ** 3)
            else:
                eps = 1.0 - r
                c0 = 1.0 - 3.0 * eps + 6.0 * (eps ** 2) - 10.0 * (eps ** 3)
                c1 = c0
                c2 = 0.5 * c0
                c3 = (1.0 - 4.0 * eps + 10.0 * (eps ** 2) - 20.0 * (eps ** 3)) / 6.0
                c4 = (-eps + 5.0 * (eps ** 2) - 15.0 * (eps ** 3)) / 24.0
                c5 = ((eps ** 2) - 6.0 * (eps ** 3)) / 120.0
                c6 = -(eps ** 3) / 720.0
                A_conv = 1.0 - (
                    c0 + c1 * x + c2 * (x ** 2) + c3 * (x ** 3) + c4 * (x ** 4) + c5 * (x ** 5) + c6 * (x ** 6)
                ) * expx
            Ap = (1.0 - R) * A_chisqr + R * A_conv
        else:
            Ap = A_chisqr
    elif R > 0 and taus > 0:
        Ap = 1.0 - np.exp(-tp / taus)
    else:
        Ap = np.ones_like(tp)

    A[pos] = np.clip(Ap, 0.0, 1.0)
    return float(A) if is_scalar else A


def ikcarp_Log(t: float, tauf: float, taus: float, R: float) -> Tuple[float, float, float]:
    """Return ``(ln(y), d(ln y)/dt, d^2(ln y)/dt^2)`` of the Ikeda-Carpenter function at ``t > 0``."""
    if t <= 0:
        return -math.inf, math.inf, -math.inf

    if tauf > 0:
        x = t / tauf
        if R > 0 and taus > 0:
            r = tauf / taus
            d = 1.0 - r
            if abs(d) > 1e-4:
                expx = math.exp(-x)
                exprx = math.exp(-r * x)
                # Direct evaluation via y, y', y''
                y_chi = (0.5 / tauf) * (x ** 2) * expx
                dy_chi = (0.5 / (tauf ** 2)) * x * (2.0 - x) * expx
                d2y_chi = (0.5 / (tauf ** 3)) * (2.0 - 4.0 * x + x ** 2) * expx

                pref = 1.0 / (taus * (d ** 3))
                poly = 1.0 + d * x + 0.5 * (d ** 2) * (x ** 2)
                dpoly = d + (d ** 2) * x
                d2poly = d ** 2

                y_conv = pref * (exprx - expx * poly)
                dy_conv = (pref / tauf) * (-r * exprx + expx * poly - expx * dpoly)
                d2y_conv = (pref / (tauf ** 2)) * (
                    (r ** 2) * exprx - expx * poly + 2.0 * expx * dpoly - expx * d2poly
                )

                y = (1.0 - R) * y_chi + R * y_conv
                dy = (1.0 - R) * dy_chi + R * dy_conv
                d2y = (1.0 - R) * d2y_chi + R * d2y_conv
            else:
                # Series expansion around r = 1
                eps = d
                expx = math.exp(-x)
                y_chi = (0.5 / tauf) * (x ** 2) * expx
                dy_chi = (0.5 / (tauf ** 2)) * x * (2.0 - x) * expx
                d2y_chi = (0.5 / (tauf ** 3)) * (2.0 - 4.0 * x + x ** 2) * expx

                poly = (x ** 3) * (1.0 - 0.25 * eps * x + (1.0 / 20.0) * (eps * x) ** 2 - (1.0 / 120.0) * (eps * x) ** 3)
                dpoly = (x ** 2) * (3.0 - eps * x + 0.25 * (eps * x) ** 2 - (1.0 / 20.0) * (eps * x) ** 3)
                d2poly = x * (6.0 - 3.0 * eps * x + (eps * x) ** 2 - 0.25 * (eps * x) ** 3)

                pref = 1.0 / (6.0 * taus)
                y_conv = pref * poly * expx
                dy_conv = (pref / tauf) * (dpoly - poly) * expx
                d2y_conv = (pref / (tauf ** 2)) * (d2poly - 2.0 * dpoly + poly) * expx

                y = (1.0 - R) * y_chi + R * y_conv
                dy = (1.0 - R) * dy_chi + R * dy_conv
                d2y = (1.0 - R) * d2y_chi + R * d2y_conv

            if y <= 0:
                return -math.inf, 0.0, 0.0
            ln_y = math.log(y)
            dln_y = dy / y
            d2ln_y = (d2y * y - dy ** 2) / (y ** 2)
            return ln_y, dln_y, d2ln_y
        else:
            # Pure chi-squared: y = (1/(2*tauf)) * (t/tauf)^2 * exp(-t/tauf)
            ln_y = math.log(0.5 / tauf) + 2.0 * math.log(x) - x
            dln_y = (2.0 / t) - (1.0 / tauf)
            d2ln_y = -2.0 / (t ** 2)
            return ln_y, dln_y, d2ln_y
    elif R > 0 and taus > 0:
        ln_y = -math.log(taus) - t / taus
        dln_y = -1.0 / taus
        d2ln_y = 0.0
        return ln_y, dln_y, d2ln_y
    else:
        return -math.inf, 0.0, 0.0


def ikcarp_peak(tauf: float, taus: float, R: float) -> Tuple[float, float]:
    """Return ``(tmax, ymax)`` at the peak of the Ikeda-Carpenter function."""
    if tauf > 0:
        if R > 0 and taus > 0:
            # Peak lies between 2*tauf and 3*tauf + taus
            t_lo = 2.0 * tauf
            # Find upper bracket where derivative is negative
            t_hi = max(3.0 * tauf, 2.0 * tauf + taus)
            while ikcarp_Log(t_hi, tauf, taus, R)[1] > 0:
                t_hi *= 2.0
            if abs(ikcarp_Log(t_lo, tauf, taus, R)[1]) < 1e-15:
                tmax = t_lo
            else:
                tmax = brentq(lambda t: ikcarp_Log(t, tauf, taus, R)[1], t_lo, t_hi, xtol=1e-13, rtol=1e-13)
            ymax = float(ikcarp(tmax, tauf, taus, R))
            return tmax, ymax
        else:
            tmax = 2.0 * tauf
            ymax = float(ikcarp(tmax, tauf, taus, 0.0))
            return tmax, ymax
    elif R > 0 and taus > 0:
        tmax = 0.0
        ymax = 1.0 / taus
        return tmax, ymax
    else:
        return 0.0, math.inf


def ikcarp_fwhh(
    tauf: float,
    taus: float,
    R: float,
    frac: Union[float, np.ndarray] = 0.5,
) -> Tuple[Union[float, np.ndarray], float, Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Return ``(w, tmax, tlo, thi)`` of the Ikeda-Carpenter pulse at height fraction ``frac``."""
    is_scalar = np.ndim(frac) == 0
    frac_arr = np.asarray(frac, dtype=float)
    if np.any((frac_arr <= 0) | (frac_arr >= 1)):
        raise ValueError("HEBERT:IX_moderator:invalid_argument: Height fraction must satisfy 0 < frac < 1.")

    tmax, ymax = ikcarp_peak(tauf, taus, R)
    if tmax == 0.0 and tauf == 0.0 and R > 0 and taus > 0:
        tlo = np.zeros_like(frac_arr, dtype=float)
        thi = -taus * np.log(frac_arr)
        w = thi - tlo
        if is_scalar:
            return float(w), float(tmax), float(tlo), float(thi)
        return w, float(tmax), tlo, thi
    if ymax == math.inf or (tauf == 0.0 and (R == 0.0 or taus == 0.0)):
        z = np.zeros_like(frac_arr, dtype=float)
        if is_scalar:
            return 0.0, 0.0, 0.0, 0.0
        return z, 0.0, z, z

    ln_ymax = math.log(ymax)
    tlo = np.zeros_like(frac_arr, dtype=float)
    thi = np.zeros_like(frac_arr, dtype=float)

    tau_scale = max(tauf, taus if R > 0 else 0.0)
    for idx in np.ndindex(frac_arr.shape):
        f_val = float(frac_arr[idx])
        target_ln = ln_ymax + math.log(f_val)

        # Lower root in (0, tmax)
        lo_a = 1e-12 * tmax
        while ikcarp_Log(lo_a, tauf, taus, R)[0] > target_ln:
            lo_a *= 0.1
        tlo[idx] = brentq(
            lambda t: ikcarp_Log(t, tauf, taus, R)[0] - target_ln,
            lo_a,
            tmax,
            xtol=1e-13,
            rtol=1e-13,
        )

        # Upper root in (tmax, inf)
        hi_b = tmax + 2.0 * tau_scale
        while ikcarp_Log(hi_b, tauf, taus, R)[0] > target_ln:
            hi_b = tmax + 2.0 * (hi_b - tmax)
        thi[idx] = brentq(
            lambda t: ikcarp_Log(t, tauf, taus, R)[0] - target_ln,
            tmax,
            hi_b,
            xtol=1e-13,
            rtol=1e-13,
        )

    w = thi - tlo
    if is_scalar:
        return float(w), float(tmax), float(tlo), float(thi)
    return w, float(tmax), tlo, thi


def ikcarp_mean(tauf: float, taus: float, R: float) -> Tuple[float, float]:
    """Return ``(dt, t_av)`` (standard deviation and mean) of the Ikeda-Carpenter function."""
    if R > 0:
        t_av = 3.0 * tauf + R * taus
        dt = math.sqrt(3.0 * (tauf ** 2) + R * (2.0 - R) * (taus ** 2))
    else:
        t_av = 3.0 * tauf
        dt = math.sqrt(3.0) * tauf
    return dt, t_av


def _ikcarp_pdf_xvals_private(
    npnt: float, tau: float, t_av: float, m: float, Rmax: float
) -> np.ndarray:
    """Private helper for ``ikcarp_pdf_xvals`` matching ``ikcarp_pdf_xvals_private`` in MATLAB."""
    if Rmax * tau > m * t_av:
        frac = (Rmax + (m ** 2) * (t_av / tau)) / (npnt - 1.0)
        nlo = int(math.floor((t_av / tau) * (m * (m + 1.0)) / frac + 0.5)) + 1
        nhi = max(1, int(math.floor((Rmax - m * (t_av / tau)) / frac + 0.5)))
        t_red = (np.arange(0, nlo, dtype=float) / float(nlo - 1)) * (m / (m + 1.0))
        tlo = t_av * (t_red / (1.0 - t_red))
        i = np.arange(1, nhi + 1, dtype=float)
        thi = ((m * t_av) * (nhi - i) + (Rmax * tau) * i) / float(nhi)
        return np.concatenate([tlo, thi])
    else:
        nlo = max(1, int(math.floor(npnt + 0.5)))
        t_red = (np.arange(0, nlo, dtype=float) / float(nlo - 1)) * (m / (m + 1.0))
        return t_av * (t_red / (1.0 - t_red))


def ikcarp_pdf_xvals(npnt: int, tauf: float, taus: float) -> np.ndarray:
    """Create array of time values suitable for building a lookup table of the Ikeda-Carpenter pdf.

    Exact port of ``@IX_moderator/private/ikcarp_pdf_xvals.m``.
    """
    Rmax = 15.0
    m = 2.0
    t_fast = _ikcarp_pdf_xvals_private(npnt / 2.0, tauf, 3.0 * tauf, m, Rmax)
    t_slow = _ikcarp_pdf_xvals_private(
        npnt / 2.0, max(tauf, taus), 3.0 * tauf + taus, m, Rmax
    )
    return np.unique(np.concatenate([t_fast, t_slow]))


_PULSE_MODELS: List[str] = ["ikcarp", "ikcarp_param", " vividly_unused", "table"]
_VALID_PULSE_MODELS: List[str] = ["ikcarp", "ikcarp_param", "table"]
_NPNT_PDF: int = 200
_NPNT_PLOT: int = 500


@register_serializable
class IX_moderator(hashable):
    """Moderator pulse model class.

    Converted from MATLAB ``IX_moderator``.
    """

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name: str = ""
        self._distance: float = 0.0
        self._angle: float = 0.0
        self._pulse_model: str = "ikcarp"
        self._pp: np.ndarray = np.zeros((1, 3), dtype=float)
        self._temperature: float = 0.0
        self._thickness: float = 0.0
        self._width: float = 0.0
        self._height: float = 0.0
        self._energy: float = 0.0
        self._t: np.ndarray = np.empty((0,), dtype=float)
        self._y: np.ndarray = np.empty((0,), dtype=float)
        self._pdf: pdf_table = pdf_table()
        self._cached_pp_tuple: Optional[Tuple[float, ...]] = None
        self._cached_energy: Optional[float] = None
        self._cached_model: Optional[str] = None

        if not args and not kwargs:
            self._recompute_pdf()
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], IX_moderator):
            other = args[0]
            for f in self.saveable_fields():
                setattr(self, f, getattr(other, f))
            self._recompute_pdf()
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            d = args[0]
            self._do_check_combo_arg = False
            for k, v in d.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._do_check_combo_arg = True
            self.check_combo_arg(True)
            return

        pos_params = [
            "distance",
            "angle",
            "pulse_model",
            "pp",
            "temperature",
            "thickness",
            "width",
            "height",
            "energy",
            "name",
        ]
        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, False, *args, **kwargs
        )
        if remains:
            raise ValueError(f"Unrecognised extra arguments: {remains}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        if val is None:
            self._name = ""
            return
        if not isinstance(val, str):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator name must be a string.")
        self._name = val

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator distance must be >= 0.")
        self._distance = v

    @property
    def angle(self) -> float:
        return self._angle

    @angle.setter
    def angle(self, val: float) -> None:
        v = float(val)
        if not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator angle must be finite.")
        self._angle = v

    @property
    def pulse_model(self) -> str:
        return self._pulse_model

    @pulse_model.setter
    def pulse_model(self, val: str) -> None:
        if not isinstance(val, str) or not val.strip():
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Pulse model must be a non-empty string.")
        val_l = val.strip().lower()
        matches = [m for m in _VALID_PULSE_MODELS if m == val_l]
        if not matches:
            matches = [m for m in _VALID_PULSE_MODELS if m.startswith(val_l)]
        if len(matches) != 1:
            raise ValueError(f"HEBERT:IX_moderator:invalid_argument: Unrecognised pulse model '{val}'.")
        self._pulse_model = matches[0]
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def pp(self) -> np.ndarray:
        return self._pp

    @pp.setter
    def pp(self, val: Any) -> None:
        # Table pulse model can accept [t, y] cell/tuple/list of two arrays
        if isinstance(val, (list, tuple)) and len(val) == 2 and isinstance(val[0], (list, tuple, np.ndarray)) and np.asarray(val[0]).size > 1:
            t_arr = np.asarray(val[0], dtype=float).ravel()
            y_arr = np.asarray(val[1], dtype=float).ravel()
            if t_arr.size != y_arr.size:
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Table t and y arrays must have the same length.")
            self._t = t_arr
            self._y = y_arr
            self._pp = np.empty((0,), dtype=float)
        else:
            arr = np.asarray(val, dtype=float).ravel()
            self._pp = arr
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator temperature must be >= 0.")
        self._temperature = v

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator thickness must be >= 0.")
        self._thickness = v

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator width must be >= 0.")
        self._width = v

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator height must be >= 0.")
        self._height = v

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    def energy(self, val: float) -> None:
        v = float(val)
        if v < 0 or not np.isfinite(v):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Moderator energy must be >= 0.")
        self._energy = v
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def pdf(self) -> pdf_table:
        self._ensure_pdf_current()
        if not self._pdf.filled:
            raise ValueError(
                "HEBERT:IX_moderator:runtime_error: "
                "Moderator pulse distribution function is a delta function or uninitialised."
            )
        return self._pdf

    # ------------------------------------------------------------------
    # Internal validation & PDF table calculation
    # ------------------------------------------------------------------
    def _ensure_pdf_current(self) -> None:
        """Detect in-place mutations of ``self._pp`` and recompute ``_pdf`` if needed."""
        curr_pp = tuple(np.asarray(self._pp, dtype=float).ravel().tolist())
        if (
            self._cached_pp_tuple != curr_pp
            or self._cached_energy != self._energy
            or self._cached_model != self._pulse_model
        ):
            self.check_combo_arg(True)

    def _get_ikcarp_params(self) -> Tuple[float, float, float]:
        """Return ``(tauf, taus, R)`` for ``ikcarp`` or ``ikcarp_param`` models."""
        pp_arr = np.asarray(self._pp, dtype=float).ravel()
        if self._pulse_model == "ikcarp":
            if pp_arr.size != 3:
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Model 'ikcarp' requires 3 parameters [tauf, taus, R].")
            tauf, taus, R = float(pp_arr[0]), float(pp_arr[1]), float(pp_arr[2])
            if tauf < 0 or taus < 0 or R < 0 or R > 1:
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Invalid parameters for 'ikcarp' model.")
            return tauf, taus, R
        elif self._pulse_model == "ikcarp_param":
            if pp_arr.size != 3:
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Model 'ikcarp_param' requires 3 parameters.")
            if np.any(pp_arr < 0):
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Invalid parameters for 'ikcarp_param' model.")
            if self._energy <= 0:
                raise ValueError(
                    "HEBERT:IX_moderator:invalid_argument: "
                    "Moderator energy must be greater than zero to compute pulse parameters for pulse model 'ikcarp_param'."
                )
            tauf, taus, R = ikcarp_param_convert(pp_arr, self._energy)
            return float(tauf), float(taus), float(R)
        else:
            raise ValueError(f"Not an Ikeda-Carpenter model: {self._pulse_model}")

    def _recompute_pdf(self) -> None:
        if self._pulse_model == "table":
            if self._t.size == 0 and self._pp.size > 0 and self._pp.size % 2 == 0:
                half = self._pp.size // 2
                self._t = self._pp[:half].copy()
                self._y = self._pp[half:].copy()
                self._pp = np.empty((0,), dtype=float)
            if self._t.size > 0:
                self._pdf = pdf_table(self._t, self._y)
            else:
                self._pdf = pdf_table()
        elif self._pulse_model == "ikcarp":
            tauf, taus, R = self._get_ikcarp_params()
            if tauf > 0 or (R > 0 and taus > 0):
                t = ikcarp_pdf_xvals(_NPNT_PDF, tauf, taus if R > 0 else 0.0)
                y = ikcarp(t, tauf, taus, R)
                self._pdf = pdf_table(t, y)
            else:
                self._pdf = pdf_table()
        elif self._pulse_model == "ikcarp_param":
            pp_arr = np.asarray(self._pp, dtype=float).ravel()
            if pp_arr.size != 3 or np.any(pp_arr < 0):
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Model 'ikcarp_param' requires 3 non-negative parameters.")
            if self._energy > 0:
                tauf, taus, R = ikcarp_param_convert(pp_arr, self._energy)
                tauf_f, taus_f, R_f = float(tauf), float(taus), float(R)
                if tauf_f > 0 or (R_f > 0 and taus_f > 0):
                    t = ikcarp_pdf_xvals(_NPNT_PDF, tauf_f, taus_f if R_f > 0 else 0.0)
                    y = ikcarp(t, tauf_f, taus_f, R_f)
                    self._pdf = pdf_table(t, y)
                else:
                    self._pdf = pdf_table()
            else:
                self._pdf = pdf_table()

        self._cached_pp_tuple = tuple(np.asarray(self._pp, dtype=float).ravel().tolist())
        self._cached_energy = self._energy
        self._cached_model = self._pulse_model

    # ------------------------------------------------------------------
    # Public calculation methods
    # ------------------------------------------------------------------
    def pulse_width(
        self, ei: Optional[Union[float, List[float], np.ndarray]] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Calculate standard deviation ``dt``, mean time ``t_av``, and FWHH ``fwhh`` (microseconds).

        Parameters
        ----------
        ei : float or array-like, optional
            Incident energy or array of energies (meV). If omitted, uses ``self.energy``.
        """
        self._ensure_pdf_current()
        if ei is None:
            ei_val: Any = self._energy
            ei_given = False
        else:
            ei_val = ei
            ei_given = True

        ei_is_scalar = np.ndim(ei_val) == 0
        ei_arr = np.asarray(ei_val, dtype=float)
        if ei_given and np.any(ei_arr <= 0):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Incident energy must be greater than zero.")

        if self._pulse_model == "ikcarp":
            tauf, taus, R = self._get_ikcarp_params()
            dt_s, t_av_s = ikcarp_mean(tauf, taus, R)
            fwhh_s, _, _, _ = ikcarp_fwhh(tauf, taus, R, 0.5)
            if ei_is_scalar:
                return float(dt_s), float(t_av_s), float(fwhh_s)
            ones = np.ones_like(ei_arr, dtype=float)
            return float(dt_s) * ones, float(t_av_s) * ones, float(fwhh_s) * ones

        elif self._pulse_model == "ikcarp_param":
            if not ei_given and self._energy <= 0:
                raise ValueError(
                    "HEBERT:IX_moderator:invalid_argument: "
                    "Moderator energy must be greater than zero to compute pulse width for pulse model 'ikcarp_param'."
                )
            if ei_is_scalar:
                tauf, taus, R = ikcarp_param_convert(self._pp, float(ei_arr))
                dt_s, t_av_s = ikcarp_mean(float(tauf), float(taus), float(R))
                fwhh_s, _, _, _ = ikcarp_fwhh(float(tauf), float(taus), float(R), 0.5)
                return float(dt_s), float(t_av_s), float(fwhh_s)
            else:
                dt_out = np.zeros_like(ei_arr, dtype=float)
                t_av_out = np.zeros_like(ei_arr, dtype=float)
                fwhh_out = np.zeros_like(ei_arr, dtype=float)
                for idx in np.ndindex(ei_arr.shape):
                    tf_i, ts_i, r_i = ikcarp_param_convert(self._pp, float(ei_arr[idx]))
                    dt_i, tav_i = ikcarp_mean(float(tf_i), float(ts_i), float(r_i))
                    fw_i, _, _, _ = ikcarp_fwhh(float(tf_i), float(ts_i), float(r_i), 0.5)
                    dt_out[idx] = dt_i
                    t_av_out[idx] = tav_i
                    fwhh_out[idx] = fw_i
                return dt_out, t_av_out, fwhh_out

        elif self._pulse_model == "table":
            if not self._pdf.filled:
                dt_s, t_av_s, fwhh_s = 0.0, 0.0, 0.0
            else:
                dtsqr, t_av_s = self._pdf.var()
                dt_s = math.sqrt(max(0.0, dtsqr))
                fwhh_s, _, _, _ = self._pdf.width(0.5)
            if ei_is_scalar:
                return float(dt_s), float(t_av_s), float(fwhh_s)
            ones = np.ones_like(ei_arr, dtype=float)
            return float(dt_s) * ones, float(t_av_s) * ones, float(fwhh_s) * ones

        raise ValueError(f"Unknown pulse model: {self._pulse_model}")

    def pulse_width2(
        self,
        frac: float = 0.5,
        ei: Optional[Union[float, List[float], np.ndarray]] = None,
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Return ``(width, tmax, tlo, thi)`` at height fraction ``frac`` of peak height."""
        self._ensure_pdf_current()
        frac_f = float(frac)
        if not (0.0 < frac_f < 1.0):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Height fraction must satisfy 0 < frac < 1.")

        if ei is None:
            ei_val: Any = self._energy
            ei_given = False
        else:
            ei_val = ei
            ei_given = True

        ei_is_scalar = np.ndim(ei_val) == 0
        ei_arr = np.asarray(ei_val, dtype=float)
        if ei_given and np.any(ei_arr <= 0):
            raise ValueError("HEBERT:IX_moderator:invalid_argument: Incident energy must be greater than zero.")

        if self._pulse_model == "ikcarp":
            tauf, taus, R = self._get_ikcarp_params()
            w, tmax, tlo, thi = ikcarp_fwhh(tauf, taus, R, frac_f)
            if ei_is_scalar:
                return float(w), float(tmax), float(tlo), float(thi)
            ones = np.ones_like(ei_arr, dtype=float)
            return float(w) * ones, float(tmax) * ones, float(tlo) * ones, float(thi) * ones

        elif self._pulse_model == "ikcarp_param":
            if not ei_given and self._energy <= 0:
                raise ValueError(
                    "HEBERT:IX_moderator:invalid_argument: "
                    "Moderator energy must be greater than zero to compute pulse width for pulse model 'ikcarp_param'."
                )
            if ei_is_scalar:
                tauf, taus, R = ikcarp_param_convert(self._pp, float(ei_arr))
                w, tmax, tlo, thi = ikcarp_fwhh(float(tauf), float(taus), float(R), frac_f)
                return float(w), float(tmax), float(tlo), float(thi)
            else:
                w_out = np.zeros_like(ei_arr, dtype=float)
                tmax_out = np.zeros_like(ei_arr, dtype=float)
                tlo_out = np.zeros_like(ei_arr, dtype=float)
                thi_out = np.zeros_like(ei_arr, dtype=float)
                for idx in np.ndindex(ei_arr.shape):
                    tf_i, ts_i, r_i = ikcarp_param_convert(self._pp, float(ei_arr[idx]))
                    w_i, tm_i, tl_i, th_i = ikcarp_fwhh(float(tf_i), float(ts_i), float(r_i), frac_f)
                    w_out[idx] = w_i
                    tmax_out[idx] = tm_i
                    tlo_out[idx] = tl_i
                    thi_out[idx] = th_i
                return w_out, tmax_out, tlo_out, thi_out

        elif self._pulse_model == "table":
            w, tmax, tlo, thi = self._pdf.width(frac_f)
            if ei_is_scalar:
                return float(w), float(tmax), float(tlo), float(thi)
            ones = np.ones_like(ei_arr, dtype=float)
            return float(w) * ones, float(tmax) * ones, float(tlo) * ones, float(thi) * ones

        raise ValueError(f"Unknown pulse model: {self._pulse_model}")

    def pulse_shape(
        self,
        t: Optional[Union[float, List[float], np.ndarray]] = None,
        ei: Optional[float] = None,
    ) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Evaluate normalized moderator pulse shape at time(s) ``t``.

        If ``t`` is None, returns ``(y, t)`` over a suitable range of times.
        Otherwise returns ``y`` with the same shape as ``t``.
        """
        self._ensure_pdf_current()
        if ei is not None:
            ei_f = float(ei)
            if ei_f <= 0:
                raise ValueError("HEBERT:IX_moderator:invalid_argument: Incident energy must be greater than zero.")
        else:
            ei_f = self._energy

        if self._pulse_model in ("ikcarp", "ikcarp_param"):
            if self._pulse_model == "ikcarp":
                tauf, taus, R = self._get_ikcarp_params()
            else:
                if ei_f <= 0:
                    raise ValueError(
                        "HEBERT:IX_moderator:invalid_argument: "
                        "Moderator energy must be greater than zero to compute pulse shape for pulse model 'ikcarp_param'."
                    )
                tf, ts, r = ikcarp_param_convert(self._pp, ei_f)
                tauf, taus, R = float(tf), float(ts), float(r)

            if t is None:
                t_grid = ikcarp_pdf_xvals(_NPNT_PLOT, tauf, taus if R > 0 else 0.0)
                y_grid = ikcarp(t_grid, tauf, taus, R)
                return y_grid, t_grid
            else:
                return ikcarp(t, tauf, taus, R)

        elif self._pulse_model == "table":
            if t is None:
                return self._pdf.f.copy(), self._pdf.x.copy()
            is_scalar = np.ndim(t) == 0
            t_arr = np.asarray(t, dtype=float)
            y_arr = np.interp(
                t_arr,
                self._pdf.x.ravel(),
                self._pdf.f.ravel(),
                left=0.0,
                right=0.0,
            )
            return float(y_arr) if is_scalar else y_arr

        raise ValueError(f"Unknown pulse model: {self._pulse_model}")

    def rand(self, *args: Any, rng: Optional[np.random.Generator] = None) -> Union[float, np.ndarray]:
        """Generate random numbers from the moderator pulse time distribution (microseconds)."""
        self._ensure_pdf_current()
        if not self._pdf.filled:
            if self._pulse_model == "ikcarp_param" and self._energy <= 0:
                raise ValueError(
                    "HEBERT:IX_moderator:runtime_error: "
                    "Moderator energy must be set (> 0) before sampling from pulse model 'ikcarp_param'."
                )
            shape = _parse_rand_shape(args)
            out = np.zeros(shape, dtype=float)
            if out.size == 1 and len(args) == 0:
                return 0.0
            return out
        return self._pdf.rand(*args, rng=rng)

    def retain(self, t: Any, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Retain or reject trial times ``t`` using rejection sampling against the moderator PDF."""
        self._ensure_pdf_current()
        return self._pdf.retain(t, rng=rng)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        fields = [
            "distance",
            "angle",
            "pulse_model",
            "pp",
            "temperature",
            "thickness",
            "width",
            "height",
            "energy",
            "name",
        ]
        if self._pulse_model == "table":
            fields = [
                "distance",
                "angle",
                "pulse_model",
                "t",
                "y",
                "temperature",
                "thickness",
                "width",
                "height",
                "energy",
                "name",
            ]
        return fields

    @property
    def t(self) -> np.ndarray:
        return self._t

    @t.setter
    def t(self, val: Any) -> None:
        self._t = np.asarray(val, dtype=float).ravel()
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, val: Any) -> None:
        self._y = np.asarray(val, dtype=float).ravel()
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    def check_combo_arg(self, recompute_pdf: bool = True) -> None:
        if recompute_pdf:
            self._recompute_pdf()

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_moderator]:
        out = dict(inputs)
        if (ver is None or ver == 1) and "c_to_m" in out:
            out["distance"] = out.pop("c_to_m")
        return out, self
