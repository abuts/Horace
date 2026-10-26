"""Combined moderator, shaping chopper and monochromating chopper class (IX_mod_shape_mono).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_mod_shape_mono/
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import dblquad, quad
from scipy.optimize import fminbound

from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_doubledisk_chopper import IX_doubledisk_chopper
from pyhorace.data.instr_classes.ix_moderator import IX_moderator
from pyhorace.data.instr_classes.pdf_table import (
    _parse_rand_shape,
    pdf_table,
    sampling_statistics,
)


def _triangle_int(u: np.ndarray, tau: float) -> np.ndarray:
    """Indefinite integral of the triangle distribution with half-width ``tau > 0``.

    Matches ``triangle_int(u, tau)`` in ``@IX_mod_shape_mono/private/moments_.m``.
    """
    u_arr = np.asarray(u, dtype=float)
    val = np.zeros_like(u_arr, dtype=float)
    val[u_arr >= tau] = 1.0

    neg = (u_arr > -tau) & (u_arr <= 0.0)
    if np.any(neg):
        x = u_arr[neg] / tau
        val[neg] = 0.5 * ((1.0 + x) ** 2)

    pos = (u_arr > 0.0) & (u_arr < tau)
    if np.any(pos):
        x = u_arr[pos] / tau
        val[pos] = 1.0 - 0.5 * ((1.0 - x) ** 2)

    return val


def _triangle_t_int(u: np.ndarray, tau: float) -> np.ndarray:
    """Indefinite integral of ``u * g(u)`` for the triangle distribution with half-width ``tau > 0``."""
    u_arr = np.asarray(u, dtype=float)
    val = np.zeros_like(u_arr, dtype=float)

    neg = (u_arr > -tau) & (u_arr <= 0.0)
    if np.any(neg):
        x = u_arr[neg] / tau
        val[neg] = -tau * ((1.0 / 6.0) - 0.5 * (x ** 2) - (1.0 / 3.0) * (x ** 3))

    pos = (u_arr > 0.0) & (u_arr < tau)
    if np.any(pos):
        x = u_arr[pos] / tau
        val[pos] = -tau * ((1.0 / 6.0) - 0.5 * (x ** 2) + (1.0 / 3.0) * (x ** 3))

    return val


def _triangle_tsqr_int(u: np.ndarray, tau: float) -> np.ndarray:
    """Indefinite integral of ``u^2 * g(u)`` for the triangle distribution with half-width ``tau > 0``."""
    u_arr = np.asarray(u, dtype=float)
    val = np.zeros_like(u_arr, dtype=float)
    val[u_arr >= tau] = (tau ** 2) / 6.0

    neg = (u_arr > -tau) & (u_arr <= 0.0)
    if np.any(neg):
        x = u_arr[neg] / tau
        val[neg] = (tau ** 2) * ((1.0 / 12.0) + (1.0 / 3.0) * (x ** 3) + 0.25 * (x ** 4))

    pos = (u_arr > 0.0) & (u_arr < tau)
    if np.any(pos):
        x = u_arr[pos] / tau
        val[pos] = (tau ** 2) * ((1.0 / 12.0) + (1.0 / 3.0) * (x ** 3) - 0.25 * (x ** 4))

    return val


@register_serializable
class IX_mod_shape_mono(hashable):
    """Combined moderator, shaping chopper and monochromating chopper class.

    Converted from MATLAB ``IX_mod_shape_mono``.
    """

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._moderator: IX_moderator = IX_moderator()
        self._shaping_chopper: IX_doubledisk_chopper = IX_doubledisk_chopper()
        self._mono_chopper: IX_doubledisk_chopper = IX_doubledisk_chopper()
        self._shaped_mod: bool = False
        self._t_m_offset: np.ndarray = np.zeros(8, dtype=float)
        self._t_chop_av: np.ndarray = np.zeros((2, 8), dtype=float)
        self._t_chop_cov: np.ndarray = np.zeros((2, 2, 8), dtype=float)
        self._component_state_key: Optional[Any] = None

        if not args and not kwargs:
            self._save_state_key()
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], IX_mod_shape_mono):
            other = args[0]
            other._ensure_current()
            self._moderator = IX_moderator(other._moderator)
            self._shaping_chopper = IX_doubledisk_chopper(other._shaping_chopper)
            self._mono_chopper = IX_doubledisk_chopper(other._mono_chopper)
            self._shaped_mod = other._shaped_mod
            self._t_m_offset = other._t_m_offset.copy()
            self._t_chop_av = other._t_chop_av.copy()
            self._t_chop_cov = other._t_chop_cov.copy()
            self._save_state_key()
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
            "moderator",
            "shaping_chopper",
            "mono_chopper",
            "energy",
        ]
        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, False, *args, **kwargs
        )
        if remains:
            raise ValueError(f"Unrecognised extra arguments: {remains}")

    # ------------------------------------------------------------------
    # State tracking for in-place mutations of sub-objects
    # ------------------------------------------------------------------
    def _compute_state_key(self) -> Tuple[Any, ...]:
        mod = self._moderator
        sc = self._shaping_chopper
        mc = self._mono_chopper
        return (
            mod.distance,
            mod.angle,
            mod.pulse_model,
            tuple(np.asarray(mod.pp, dtype=float).ravel().tolist()),
            tuple(np.asarray(mod.t, dtype=float).ravel().tolist()),
            tuple(np.asarray(mod.y, dtype=float).ravel().tolist()),
            mod.energy,
            sc.distance,
            sc.frequency,
            sc.radius,
            sc.slot_width,
            sc.aperture_width,
            sc.aperture_height,
            sc.jitter,
            mc.distance,
            mc.frequency,
            mc.radius,
            mc.slot_width,
            mc.aperture_width,
            mc.aperture_height,
            mc.jitter,
        )

    def _save_state_key(self) -> None:
        self._component_state_key = self._compute_state_key()

    def _ensure_current(self) -> None:
        if self._component_state_key != self._compute_state_key():
            self.check_combo_arg(True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def moderator(self) -> IX_moderator:
        return self._moderator

    @moderator.setter
    def moderator(self, val: IX_moderator) -> None:
        if not isinstance(val, IX_moderator):
            raise ValueError("HEBERT:IX_mod_shape_mono:invalid_argument: Moderator must be an IX_moderator object.")
        self._moderator = val
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def shaping_chopper(self) -> IX_doubledisk_chopper:
        return self._shaping_chopper

    @shaping_chopper.setter
    def shaping_chopper(self, val: IX_doubledisk_chopper) -> None:
        if not isinstance(val, IX_doubledisk_chopper):
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:invalid_argument: "
                "Shaping chopper must be a scalar IX_doubledisk_chopper object."
            )
        self._shaping_chopper = val
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def mono_chopper(self) -> IX_doubledisk_chopper:
        return self._mono_chopper

    @mono_chopper.setter
    def mono_chopper(self, val: IX_doubledisk_chopper) -> None:
        if not isinstance(val, IX_doubledisk_chopper):
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:invalid_argument: "
                "Monochromating chopper must be a scalar IX_doubledisk_chopper object."
            )
        self._mono_chopper = val
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def energy(self) -> float:
        return self._moderator.energy

    @energy.setter
    def energy(self, val: float) -> None:
        self._moderator.energy = float(val)
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def shaped_mod(self) -> bool:
        self._ensure_current()
        return self._shaped_mod

    @property
    def t_m_offset(self) -> np.ndarray:
        self._ensure_current()
        return self._t_m_offset

    @property
    def t_chop_av(self) -> np.ndarray:
        self._ensure_current()
        return self._t_chop_av

    @property
    def t_chop_cov(self) -> np.ndarray:
        self._ensure_current()
        return self._t_chop_cov

    # ------------------------------------------------------------------
    # Internal helper methods (matching private/*.m)
    # ------------------------------------------------------------------
    def valid_components_(self) -> bool:
        """Check whether moderator, shaping chopper and mono chopper distances are ordered."""
        return bool(
            self._moderator.distance > self._shaping_chopper.distance > self._mono_chopper.distance
        )

    def recompute_shaped_mod_(self) -> bool:
        """Determine whether the shaping chopper pulse width is less than the scaled moderator pulse width."""
        try:
            x1 = self._mono_chopper.distance
            x0 = self._moderator.distance - x1
            xa = self._shaping_chopper.distance - x1
            if xa == 0.0:
                return False
            _, _, fwhh_moderator = self._moderator.pulse_width()
            _, fwhh_shaping_chopper = self._shaping_chopper.pulse_width()[:2]
            val = (x0 / xa) * float(fwhh_shaping_chopper)
            if math.isnan(val):
                return False
            return bool(val < float(fwhh_moderator))
        except Exception:
            return False

    def t_m_offset_calibrate_(self) -> np.ndarray:
        """Compute moderator offset times for all 8 combinations of ``mc``.

        Exact port of ``@IX_mod_shape_mono/private/t_m_offset_calibrate_.m``.
        """
        t_m_offset = np.zeros(8, dtype=float)
        _, t_m_av, _ = self._moderator.pulse_width()
        t_m_av = float(t_m_av)

        t_m_offset[4] = t_m_av  # mc = [1, 0, 0]
        t_m_offset[5] = t_m_av  # mc = [1, 0, 1]
        t_m_offset[6] = self._t_m_offset_ms()   # mc = [1, 1, 0]
        t_m_offset[7] = self._t_m_offset_msm()  # mc = [1, 1, 1]
        return t_m_offset

    def _t_m_offset_msm(self) -> float:
        """Calibrate moderator offset time for mc = [1, 1, 1] using discrete convolution."""
        moderator = self._moderator
        shaping_chopper = self._shaping_chopper
        mono_chopper = self._mono_chopper
        energy = self.energy

        x1 = mono_chopper.distance
        x0 = moderator.distance - x1
        xa = shaping_chopper.distance - x1
        const_sh = x0 / xa
        const_ch = (x0 - xa) / xa

        w_mod, _, t_HH_lo, t_HH_hi = moderator.pulse_width2(0.5, energy)
        w_mod = float(w_mod)
        t_HH_lo = float(t_HH_lo)
        t_HH_hi = float(t_HH_hi)

        tlo_sh, thi_sh = shaping_chopper.pulse_range()
        w_shape = 2.0 * max(abs(float(tlo_sh)), abs(float(thi_sh)))
        w_shape_at_mod = w_shape * const_sh

        tlo_ch, thi_ch = mono_chopper.pulse_range()
        w_mono = 2.0 * max(abs(float(tlo_ch)), abs(float(thi_ch)))
        w_mono_at_mod = w_mono * const_ch

        frac = 0.01
        pos_widths = [w for w in (w_mod, w_shape_at_mod, w_mono_at_mod) if w > 0.0]
        if not pos_widths:
            return float(moderator.pulse_width()[1])
        dt = frac * min(pos_widths)

        n_sh = int(math.ceil(0.5 * w_shape_at_mod / dt)) if w_shape_at_mod > 0 else 0
        t_sh = (dt / const_sh) * np.arange(-n_sh, n_sh + 1, dtype=float)
        sh = np.asarray(shaping_chopper.pulse_shape(t_sh), dtype=float).ravel()

        n_ch = int(math.ceil(0.5 * w_mono_at_mod / dt)) if (w_mono_at_mod > 0 and const_ch > 0) else 0
        t_ch = (dt / const_ch) * np.arange(-n_ch, n_ch + 1, dtype=float) if const_ch > 0 else np.array([0.0])
        ch = np.asarray(mono_chopper.pulse_shape(t_ch), dtype=float).ravel()

        chop_fun = np.convolve(sh, ch, mode="full")

        Tlo = t_HH_lo - (w_shape_at_mod + w_mono_at_mod)
        Thi = t_HH_hi + (w_shape_at_mod + w_mono_at_mod)
        n_t = int(math.ceil((Thi - Tlo) / dt))
        t = Tlo + dt * np.arange(0, n_t + 1, dtype=float)
        mod_fun = np.asarray(moderator.pulse_shape(t), dtype=float).ravel()

        intensity = np.convolve(mod_fun, chop_fun, mode="same")
        imax = int(np.argmax(intensity))
        return float(Tlo + dt * imax)

    def _t_m_offset_ms(self) -> float:
        """Calibrate moderator offset time for mc = [1, 1, 0] using discrete convolution."""
        moderator = self._moderator
        shaping_chopper = self._shaping_chopper
        mono_chopper = self._mono_chopper
        energy = self.energy

        x1 = mono_chopper.distance
        x0 = moderator.distance - x1
        xa = shaping_chopper.distance - x1
        const_sh = x0 / xa

        w_mod, _, t_HH_lo, t_HH_hi = moderator.pulse_width2(0.5, energy)
        w_mod = float(w_mod)
        t_HH_lo = float(t_HH_lo)
        t_HH_hi = float(t_HH_hi)

        tlo_sh, thi_sh = shaping_chopper.pulse_range()
        w_shape = 2.0 * max(abs(float(tlo_sh)), abs(float(thi_sh)))
        w_shape_at_mod = w_shape * const_sh

        frac = 0.01
        pos_widths = [w for w in (w_mod, w_shape_at_mod) if w > 0.0]
        if not pos_widths:
            return float(moderator.pulse_width()[1])
        dt = frac * min(pos_widths)

        n_sh = int(math.ceil(0.5 * w_shape_at_mod / dt)) if w_shape_at_mod > 0 else 0
        t_sh = (dt / const_sh) * np.arange(-n_sh, n_sh + 1, dtype=float)
        sh = np.asarray(shaping_chopper.pulse_shape(t_sh), dtype=float).ravel()

        Tlo = t_HH_lo - w_shape_at_mod
        Thi = t_HH_hi + w_shape_at_mod
        n_t = int(math.ceil((Thi - Tlo) / dt))
        t = Tlo + dt * np.arange(0, n_t + 1, dtype=float)
        mod_fun = np.asarray(moderator.pulse_shape(t), dtype=float).ravel()

        intensity = np.convolve(mod_fun, sh, mode="same")
        imax = int(np.argmax(intensity))
        return float(Tlo + dt * imax)

    @staticmethod
    def _composite_gl_nodes_weights(breakpoints: Sequence[float], n_per_panel: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Build composite Gauss-Legendre nodes and weights across sorted breakpoints."""
        pts = np.unique(np.asarray(breakpoints, dtype=float))
        if pts.size < 2:
            return np.array([], dtype=float), np.array([], dtype=float)
        xi, wi = np.polynomial.legendre.leggauss(n_per_panel)
        nodes_list = []
        weights_list = []
        for a, b in zip(pts[:-1], pts[1:]):
            if b > a:
                half = 0.5 * (b - a)
                mid = 0.5 * (a + b)
                nodes_list.append(mid + half * xi)
                weights_list.append(half * wi)
        if not nodes_list:
            return np.array([], dtype=float), np.array([], dtype=float)
        return np.concatenate(nodes_list), np.concatenate(weights_list)

    def _moments_1D(self, alf: float) -> Tuple[float, float]:
        """Port of ``moments_1D`` in ``@IX_mod_shape_mono/private/moments_.m``."""
        moderator = self._moderator
        shaping_chopper = self._shaping_chopper
        t_m_offset = float(self._t_m_offset[6])

        _, t_m_av, _ = moderator.pulse_width()
        t_m_av = float(t_m_av)
        tlo_shape, thi_shape = shaping_chopper.pulse_range()
        tlo_shape = float(tlo_shape)
        thi_shape = float(thi_shape)

        if alf * t_m_offset > abs(tlo_shape):
            tlo = tlo_shape
        else:
            tlo = -alf * t_m_offset

        if thi_shape <= tlo:
            return 0.0, 0.0

        _, sh_breaks = shaping_chopper.pulse_shape()
        waypoints = alf * np.array([-0.5 * t_m_av, 0.0, 0.5 * t_m_av, t_m_av, 2.0 * t_m_av], dtype=float)
        all_breaks = [tlo, thi_shape]
        for p in np.concatenate([np.asarray(sh_breaks, dtype=float).ravel(), waypoints]):
            if tlo < p < thi_shape:
                all_breaks.append(float(p))

        nodes, weights = self._composite_gl_nodes_weights(all_breaks, n_per_panel=48)
        if nodes.size == 0:
            return 0.0, 0.0

        t_m = nodes / alf + t_m_offset
        f = np.asarray(moderator.pulse_shape(t_m), dtype=float).ravel() * np.asarray(
            shaping_chopper.pulse_shape(nodes), dtype=float
        ).ravel()
        wf = weights * f
        area = float(np.sum(wf))
        if area <= 0.0:
            return 0.0, 0.0
        t_av = float(np.sum(wf * nodes) / area)
        t_var = float(np.sum(wf * (nodes ** 2)) / area) - (t_av ** 2)
        return max(0.0, t_var), t_av

    def _moments_2D(self, alf: float) -> Tuple[np.ndarray, np.ndarray]:
        """Port of ``moments_2D`` in ``@IX_mod_shape_mono/private/moments_.m``."""
        moderator = self._moderator
        shaping_chopper = self._shaping_chopper
        mono_chopper = self._mono_chopper
        t_m_offset = float(self._t_m_offset[7])

        _, t_m_av, _ = moderator.pulse_width()
        t_m_av = float(t_m_av)
        tlo_shape, thi_shape = shaping_chopper.pulse_range()
        tlo_shape, thi_shape = float(tlo_shape), float(thi_shape)
        tlo_mono, thi_mono = mono_chopper.pulse_range()
        tlo_mono, thi_mono = float(tlo_mono), float(thi_mono)

        fac = 0.33
        if (alf * t_m_av) > fac * thi_shape and thi_shape > tlo_shape and thi_mono > tlo_mono:
            _, sh_breaks = shaping_chopper.pulse_shape()
            _, ch_breaks = mono_chopper.pulse_shape()
            x_nodes, x_weights = self._composite_gl_nodes_weights(sh_breaks, n_per_panel=32)
            y_nodes, y_weights = self._composite_gl_nodes_weights(ch_breaks, n_per_panel=32)

            # 2D tensor grid: x is t_sh (N_x, 1), y is t_ch (1, N_y)
            X = x_nodes[:, None]
            Y = y_nodes[None, :]
            t_m = X / alf - ((1.0 - alf) / alf) * Y + t_m_offset

            mod_p = np.asarray(moderator.pulse_shape(t_m), dtype=float)
            sh_p = np.asarray(shaping_chopper.pulse_shape(x_nodes), dtype=float).ravel()[:, None]
            ch_p = np.asarray(mono_chopper.pulse_shape(y_nodes), dtype=float).ravel()[None, :]

            W = (x_weights[:, None] * sh_p) * (y_weights[None, :] * ch_p) * mod_p
            area = float(np.sum(W))
            if area <= 0.0:
                return np.zeros((2, 2), dtype=float), np.zeros(2, dtype=float)

            wx = np.sum(W, axis=1)  # marginal over y
            wy = np.sum(W, axis=0)  # marginal over x

            av0 = float(np.sum(wx * x_nodes) / area)
            av1 = float(np.sum(wy * y_nodes) / area)

            c00 = float(np.sum(wx * (x_nodes ** 2)) / area) - av0 ** 2
            c11 = float(np.sum(wy * (y_nodes ** 2)) / area) - av1 ** 2
            c01 = float(np.sum(W * (X * Y)) / area) - av0 * av1

            t_cov = np.array([[c00, c01], [c01, c11]], dtype=float)
            t_av = np.array([av0, av1], dtype=float)
            return t_cov, t_av
        else:
            # Monte Carlo branch when shaping chopper width is large compared to moderator
            rng_mc = np.random.default_rng(0)
            npnt = 200000
            X = self._rand_internal((npnt, 1), [True, True, True], rng_mc)
            t_cov = np.cov(X)
            t_av = np.mean(X, axis=1)
            return t_cov, t_av

    def moments_(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute covariance ``(2, 2, 8)`` and mean ``(2, 8)`` for all 8 ``mc`` combinations.

        Exact port of ``@IX_mod_shape_mono/private/moments_.m``.
        """
        t_av = np.zeros((2, 8), dtype=float)
        t_cov = np.zeros((2, 2, 8), dtype=float)

        if not self.valid_components_():
            return t_cov, t_av

        x1 = self._mono_chopper.distance
        x0 = self._moderator.distance - x1
        xa = self._shaping_chopper.distance - x1
        alf = xa / x0
        bet = (x0 - xa) / x0

        sig_m = float(self._moderator.pulse_width()[0])
        sig_sh = float(self._shaping_chopper.pulse_width()[0])
        sig_ch = float(self._mono_chopper.pulse_width()[0])

        # mod-shape-chop = [1, 1, 1] (index 7 in 0-based, 8 in MATLAB)
        t_cov[:, :, 7], t_av[:, 7] = self._moments_2D(alf)

        # mod-shape-chop = [1, 1, 0] (index 6 in 0-based, 7 in MATLAB)
        t_cov[0, 0, 6], t_av[0, 6] = self._moments_1D(alf)

        # mod-shape-chop = [1, 0, 1] (index 5 in 0-based, 6 in MATLAB)
        t_cov[:, :, 5] = np.array(
            [
                [(alf * sig_m) ** 2 + (bet * sig_ch) ** 2, bet * (sig_ch ** 2)],
                [bet * (sig_ch ** 2), sig_ch ** 2],
            ],
            dtype=float,
        )

        # mod-shape-chop = [1, 0, 0] (index 4 in 0-based, 5 in MATLAB)
        t_cov[0, 0, 4] = (alf * sig_m) ** 2

        # mod-shape-chop = [0, 1, 1] (index 3 in 0-based, 4 in MATLAB)
        t_cov[0, 0, 3] = sig_sh ** 2
        t_cov[1, 1, 3] = sig_ch ** 2

        # mod-shape-chop = [0, 1, 0] (index 2 in 0-based, 3 in MATLAB)
        t_cov[0, 0, 2] = sig_sh ** 2

        # mod-shape-chop = [0, 0, 1] (index 1 in 0-based, 2 in MATLAB)
        t_cov[1, 1, 1] = sig_ch ** 2

        return t_cov, t_av

    # ------------------------------------------------------------------
    # Public query & sampling methods
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_mc(mc: Optional[Sequence[Union[bool, int]]]) -> Tuple[List[bool], int]:
        if mc is None:
            mc_bool = [True, True, True]
        else:
            mc_arr = np.asarray(mc).ravel()
            if mc_arr.size != 3:
                raise ValueError(
                    "HEBERT:IX_mod_shape_mono:invalid_argument: "
                    "Monte Carlo switch array 'mc' must have 3 elements."
                )
            mc_bool = [bool(x) for x in mc_arr]
        ind = 4 * int(mc_bool[0]) + 2 * int(mc_bool[1]) + int(mc_bool[2])
        return mc_bool, ind

    def t_mod_offset(self, mc: Optional[Sequence[Union[bool, int]]] = None) -> float:
        """Return moderator offset time (microseconds) for the given ``mc`` switches."""
        self._ensure_current()
        if not self.valid_components_():
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:runtime_error: "
                "Component distances do not satisfy moderator > shaping_chopper > mono_chopper."
            )
        _, ind = self._parse_mc(mc)
        return float(self._t_m_offset[ind])

    def mean(self, mc: Optional[Sequence[Union[bool, int]]] = None) -> np.ndarray:
        """Return ``(2, 1)`` column vector of mean ``[t_ch0; t_ch1]`` (microseconds)."""
        self._ensure_current()
        if not self.valid_components_():
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:runtime_error: "
                "Component distances do not satisfy moderator > shaping_chopper > mono_chopper."
            )
        _, ind = self._parse_mc(mc)
        return self._t_chop_av[:, ind].reshape(2, 1).copy()

    def covariance(self, mc: Optional[Sequence[Union[bool, int]]] = None) -> np.ndarray:
        """Return ``(2, 2)`` covariance matrix of ``[t_ch0; t_ch1]`` (microseconds^2)."""
        self._ensure_current()
        if not self.valid_components_():
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:runtime_error: "
                "Component distances do not satisfy moderator > shaping_chopper > mono_chopper."
            )
        _, ind = self._parse_mc(mc)
        return self._t_chop_cov[:, :, ind].copy()

    def rand(
        self,
        *args: Any,
        mc: Optional[Sequence[Union[bool, int]]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Generate random deviations ``[t_ch0; t_ch1]`` from ``(2, *sz)``.

        Follows MATLAB ``@IX_mod_shape_mono/rand.m``:
        - ``obj.rand()`` -> shape ``(2, 1)``
        - ``obj.rand(n)`` -> shape ``(2, n, n)``
        - ``obj.rand(m, n)`` or ``obj.rand([m, n])`` -> shape ``(2, m, n)``
        -Trailing ``'mc', [mc_mod, mc_shape, mc_mono]`` supported both as keyword and positional pair.
        """
        self._ensure_current()
        if not self.valid_components_():
            raise ValueError(
                "HEBERT:IX_mod_shape_mono:runtime_error: "
                "Component distances do not satisfy moderator > shaping_chopper > mono_chopper."
            )

        pos_args = list(args)
        if len(pos_args) >= 2 and isinstance(pos_args[-2], str) and pos_args[-2].lower() == "mc":
            mc = pos_args[-1]
            pos_args = pos_args[:-2]

        mc_bool, _ = self._parse_mc(mc)
        shape = _parse_rand_shape(tuple(pos_args))
        if rng is None:
            rng = np.random.default_rng()

        return self._rand_internal(shape, mc_bool, rng)

    def _mod_pulse(
        self,
        x0: float,
        xa: float,
        t_m_offset: float,
        t_ch: np.ndarray,
        shaped: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Port of ``mod_pulse`` in ``@IX_mod_shape_mono/rand.m``."""
        n = t_ch.size
        if n == 0:
            return np.zeros_like(t_ch)
        t_m = np.asarray(self._moderator.rand(n, 1, rng=rng), dtype=float).ravel() - t_m_offset
        if np.all((t_m == 0.0) & (t_ch == 0.0)):
            return np.zeros_like(t_ch)
        t_sh = (xa * t_m + (x0 - xa) * t_ch) / x0
        if shaped:
            bad = ~np.asarray(self._shaping_chopper.retain(t_sh, rng=rng), dtype=bool).ravel()
            if np.any(bad):
                t_sh[bad] = self._mod_pulse(x0, xa, t_m_offset, t_ch[bad], True, rng)
        return t_sh

    def _chop_pulse(
        self,
        x0: float,
        xa: float,
        t_m_offset: float,
        t_ch: np.ndarray,
        shaped: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Port of ``chop_pulse`` in ``@IX_mod_shape_mono/rand.m``."""
        n = t_ch.size
        if n == 0:
            return np.zeros_like(t_ch)
        t_sh = np.asarray(self._shaping_chopper.rand(n, 1, rng=rng), dtype=float).ravel()
        if np.all((t_sh == 0.0) & (t_ch == 0.0)):
            return t_sh
        if shaped:
            t_m = (x0 * t_sh - (x0 - xa) * t_ch) / xa + t_m_offset
            bad = ~np.asarray(self._moderator.retain(t_m, rng=rng), dtype=bool).ravel()
            if np.any(bad):
                t_sh[bad] = self._chop_pulse(x0, xa, t_m_offset, t_ch[bad], True, rng)
        return t_sh

    def _rand_internal(
        self,
        shape: Tuple[int, ...],
        mc_bool: List[bool],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Core random number generator matching ``@IX_mod_shape_mono/rand.m``."""
        if len(shape) >= 2 and shape[-1] == 1:
            out_shape = (2,) + shape[:-1]
        elif len(shape) >= 2 and shape[0] == 1:
            out_shape = (2,) + shape[1:]
        else:
            out_shape = (2,) + shape

        n_total = int(np.prod(out_shape[1:]))
        if n_total == 0 or not self.valid_components_():
            return np.zeros(out_shape, dtype=float)

        ind = 4 * int(mc_bool[0]) + 2 * int(mc_bool[1]) + int(mc_bool[2])
        t_m_off = float(self._t_m_offset[ind])

        x1 = self._mono_chopper.distance
        x0 = self._moderator.distance - x1
        xa = self._shaping_chopper.distance - x1

        mc_mod, mc_shape, mc_mono = mc_bool

        if mc_mono:
            t_ch = np.asarray(self._mono_chopper.rand(n_total, 1, rng=rng), dtype=float).ravel()
        else:
            t_ch = np.zeros(n_total, dtype=float)

        if mc_mod or mc_shape:
            if not mc_shape:
                t_sh = self._mod_pulse(x0, xa, t_m_off, t_ch, False, rng)
            elif not mc_mod:
                t_sh = self._chop_pulse(x0, xa, t_m_off, t_ch, False, rng)
            else:
                if self._shaped_mod:
                    t_sh = self._chop_pulse(x0, xa, t_m_off, t_ch, True, rng)
                else:
                    t_sh = self._mod_pulse(x0, xa, t_m_off, t_ch, True, rng)
        else:
            t_sh = np.zeros(n_total, dtype=float)

        return np.vstack([t_sh, t_ch]).reshape(out_shape)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        return [
            "moderator",
            "shaping_chopper",
            "mono_chopper",
        ]

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_mod_shape_mono]:
        out = dict(inputs)
        if ("class_version_" in out and out["class_version_"] == 1) or "moderator_" in out:
            out.pop("class_version_", None)
            new_out: Dict[str, Any] = {}
            for k, v in out.items():
                new_key = k[:-1] if k.endswith("_") else k
                new_out[new_key] = v
            out = new_out
        return out, self

    def check_combo_arg(self, recompute_pdf: bool = True) -> None:
        self._shaped_mod = self.recompute_shaped_mod_()
        if recompute_pdf and self.valid_components_():
            try:
                self._t_m_offset = self.t_m_offset_calibrate_()
                self._t_chop_cov, self._t_chop_av = self.moments_()
            except Exception:
                self._t_m_offset = np.zeros(8, dtype=float)
                self._t_chop_av = np.zeros((2, 8), dtype=float)
                self._t_chop_cov = np.zeros((2, 2, 8), dtype=float)
        elif not self.valid_components_():
            self._t_m_offset = np.zeros(8, dtype=float)
            self._t_chop_av = np.zeros((2, 8), dtype=float)
            self._t_chop_cov = np.zeros((2, 2, 8), dtype=float)
        self._save_state_key()
