"""Fermi chopper class (port of MATLAB ``IX_fermi_chopper``)."""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.pdf_table import pdf_table


def _is_real_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str)):
        return False
    if isinstance(val, (int, float, np.number)):
        return bool(np.isfinite(val)) and not np.iscomplexobj(val)
    arr = np.asarray(val)
    if arr.size == 1 and arr.dtype.kind in "iuf" and not np.iscomplexobj(arr):
        return bool(np.isfinite(arr.item()))
    return False


def _to_float(val: Any) -> float:
    if isinstance(val, (int, float, np.number)):
        return float(val)
    return float(np.asarray(val).item())


def _is_log_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return True
    if _is_real_scalar(val):
        f = _to_float(val)
        return f in (0.0, 1.0)
    return False


def _is_key_match(arg: str, valid_keys: Tuple[str, ...]) -> bool:
    s = arg.lstrip("-") if arg.startswith("-") else arg
    if not s:
        return False
    s_low = s.lower()
    return any(k.lower().startswith(s_low) for k in valid_keys)


@register_serializable
class IX_fermi_chopper(hashable):
    """Fermi chopper object (port of MATLAB ``IX_fermi_chopper``)."""

    _c_e_to_t: float = 2.286271439537201e03

    _class_version: int = PrivateAttr(default=2)
    _name: str = PrivateAttr(default="")
    _distance: float = PrivateAttr(default=0.0)
    _frequency: float = PrivateAttr(default=0.0)
    _radius: float = PrivateAttr(default=0.0)
    _curvature: float = PrivateAttr(default=0.0)
    _slit_width: float = PrivateAttr(default=0.0)
    _slit_spacing: float = PrivateAttr(default=0.0)
    _width: float = PrivateAttr(default=0.0)
    _height: float = PrivateAttr(default=0.0)
    _energy: float = PrivateAttr(default=0.0)
    _phase: bool = PrivateAttr(default=True)
    _jitter: float = PrivateAttr(default=0.0)
    _mandatory_field_set: List[bool] = PrivateAttr(
        default_factory=lambda: [False, False, False, False, False]
    )
    _pdf: pdf_table = PrivateAttr(default_factory=lambda: pdf_table(0.0, np.inf))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 0 and len(kwargs) == 0:
            self._pdf = self._recompute_pdf()
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_fermi_chopper):
                self._copy_from(loaded)
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], IX_fermi_chopper):
            self._copy_from(args[0])
            return

        pos_params = self.saveable_fields()
        if (
            len(args) >= 1
            and isinstance(args[0], str)
            and not args[0].startswith("-")
            and not _is_key_match(args[0], pos_params)
        ):
            self.name = args[0]
            argi = args[1:]
        else:
            argi = args

        if len(argi) > 0 or len(kwargs) > 0:
            _, remains = self.set_positional_and_key_val_arguments(
                pos_params, True, *argi, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:IX_fermi_chopper:invalid_argument: "
                    f"Unrecognized extra parameters provided as input to IX_fermi_chopper constructor: {remains!r}"
                )
        else:
            self.check_combo_arg(True)

    def _copy_from(self, other: IX_fermi_chopper) -> None:
        self._name = other._name
        self._distance = other._distance
        self._frequency = other._frequency
        self._radius = other._radius
        self._curvature = other._curvature
        self._slit_width = other._slit_width
        self._slit_spacing = other._slit_spacing
        self._width = other._width
        self._height = other._height
        self._energy = other._energy
        self._phase = other._phase
        self._jitter = other._jitter
        self._mandatory_field_set = list(other._mandatory_field_set)
        self._pdf = copy.deepcopy(other._pdf)
        self.clear_hash()

    def _clone_for_hash(self) -> IX_fermi_chopper:
        cloned: IX_fermi_chopper = super()._clone_for_hash()
        cloned._mandatory_field_set = list(self._mandatory_field_set)
        return cloned

    @classmethod
    def class_version(cls) -> int:
        return 2

    def saveable_fields(self, mandatory: Optional[bool] = None) -> Tuple[str, ...]:
        fields = (
            "distance",
            "frequency",
            "radius",
            "curvature",
            "slit_width",
            "slit_spacing",
            "width",
            "height",
            "energy",
            "phase",
            "jitter",
            "name",
        )
        if mandatory:
            return fields[:5]
        return fields

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: Any) -> None:
        self.clear_hash()
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._name = ""
        elif isinstance(val, str):
            self._name = val
        else:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Chopper name must be a character string (or empty string)"
            )

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val):
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Distance must be a numeric scalar"
            )
        self._distance = _to_float(val)
        self._mandatory_field_set[0] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._frequency
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Frequency must be a non-negative numeric scalar"
            )
        self._frequency = _to_float(val)
        self._mandatory_field_set[1] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._frequency)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._radius
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Radius must be a non-negative numeric scalar"
            )
        self._radius = _to_float(val)
        self._mandatory_field_set[2] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._radius)

    @property
    def curvature(self) -> float:
        return self._curvature

    @curvature.setter
    def curvature(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._curvature
        if not _is_real_scalar(val):
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Radius of curvature of slits must be a numeric scalar"
            )
        self._curvature = _to_float(val)
        self._mandatory_field_set[3] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._curvature)

    @property
    def slit_width(self) -> float:
        return self._slit_width

    @slit_width.setter
    def slit_width(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._slit_width
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Slit width must be a non-negative numeric scalar"
            )
        self._slit_width = _to_float(val)
        if self._slit_spacing < self._slit_width:
            if self._do_check_combo_arg:
                warnings.warn(
                    "Slit width appears to be larger than the slit spacing. "
                    "Set slit spacing equal to the slit width",
                    UserWarning,
                    stacklevel=2,
                )
            self._slit_spacing = self._slit_width
        self._mandatory_field_set[4] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._slit_width)

    @property
    def slit_spacing(self) -> float:
        return self._slit_spacing

    @slit_spacing.setter
    def slit_spacing(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Slit spacing must be a non-negative numeric scalar"
            )
        self._slit_spacing = _to_float(val)
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Width must be a non-negative numeric scalar"
            )
        self._width = _to_float(val)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Height must be a non-negative numeric scalar"
            )
        self._height = _to_float(val)

    @property
    def energy(self) -> float:
        return self._energy

    @energy.setter
    def energy(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._energy
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Energy must be a non-negative numeric scalar"
            )
        self._energy = _to_float(val)
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._energy)

    @property
    def phase(self) -> bool:
        return self._phase

    @phase.setter
    def phase(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._phase
        if not _is_log_scalar(val):
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Phase must be true or false (or 1 or 0)"
            )
        self._phase = bool(_to_float(val)) if not isinstance(val, (bool, np.bool_)) else bool(val)
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._phase)

    @property
    def jitter(self) -> float:
        return self._jitter

    @jitter.setter
    def jitter(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Jitter must be a non-negative numeric scalar"
            )
        self._jitter = _to_float(val)

    @property
    def pdf(self) -> pdf_table:
        return self._pdf

    def pdf_table(self) -> pdf_table:
        """Return the probability distribution table for the pulse shape."""
        return self._pdf

    # ------------------------------------------------------------------
    # Validation & serialization
    # ------------------------------------------------------------------
    def check_combo_arg(self, do_recompute_pdf: bool = True) -> IX_fermi_chopper:
        self.clear_hash()
        if not all(self._mandatory_field_set):
            mandatory_field_names = self.saveable_fields(True)
            unset = [
                name
                for name, is_set in zip(mandatory_field_names, self._mandatory_field_set)
                if not is_set
            ]
            raise ValueError(
                f"HERBERT:IX_fermi_chopper:invalid_argument: "
                f"Must give all mandatory properties namely: {mandatory_field_names!r}.\n"
                f" Properties: {unset!r} have not been set"
            )
        if self._slit_width > self._slit_spacing:
            raise ValueError(
                f"HERBERT:IX_fermi_chopper:invalid_argument: "
                f"Unable to set slit spacing ({self._slit_spacing:g}) less than slit width ({self._slit_width:g})"
            )
        if do_recompute_pdf:
            self._pdf = self._recompute_pdf()
        return self

    def _recompute_pdf(self) -> pdf_table:
        tlo, thi = self.pulse_range()
        if tlo == 0.0:
            return pdf_table(0.0, np.inf)
        if not math.isnan(float(tlo)):
            t = np.linspace(float(tlo), float(thi), 100)
            y = self.pulse_shape(t)
            return pdf_table(t, y)
        return pdf_table()

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_fermi_chopper]:
        out = dict(inputs)
        if ("class_version_" in out and out["class_version_"] == 1) or "distance_" in out:
            out.pop("class_version_", None)
            out.pop("pdf_", None)
            new_out: Dict[str, Any] = {}
            for k, v in out.items():
                new_key = k[:-1] if k.endswith("_") else k
                new_out[new_key] = v
            out = new_out
        return out, self

    # ------------------------------------------------------------------
    # Private helpers (ports of private/*.m)
    # ------------------------------------------------------------------
    def _parse_ei_and_phase(self, *args: Any) -> Tuple[Union[float, np.ndarray], bool, bool]:
        """Parse optional ``(ei,)`` or ``(ei, phase)`` or ``('-reverse',)`` arguments."""
        narg = len(args)
        if narg == 0:
            return self._energy, self._phase, True
        if narg == 1 and isinstance(args[0], str):
            arg0 = args[0]
            if len(arg0) >= 2 and "-reverse".startswith(arg0.lower()):
                return self._energy, not self._phase, True
            raise ValueError(
                "HERBERT:IX_fermi_chopper:invalid_argument: "
                "Unrecognised character string argument"
            )
        if narg in (1, 2):
            raw_ei = args[0]
            if isinstance(raw_ei, (bool, np.bool_, str)):
                raise ValueError(
                    "HERBERT:IX_fermi_chopper:invalid_argument: "
                    "Incident energy must be numeric and every element >= 0"
                )
            if isinstance(raw_ei, (int, float, np.number)):
                if np.iscomplexobj(raw_ei) or math.isnan(float(raw_ei)) or float(raw_ei) < 0:
                    raise ValueError(
                        "HERBERT:IX_fermi_chopper:invalid_argument: "
                        "Incident energy must be numeric and every element >= 0"
                    )
                ei: Union[float, np.ndarray] = float(raw_ei)
                is_scalar = True
            else:
                arr = np.asarray(raw_ei)
                if arr.dtype.kind not in "iuf" or np.iscomplexobj(arr) or np.any(np.isnan(arr)) or np.any(arr < 0):
                    raise ValueError(
                        "HERBERT:IX_fermi_chopper:invalid_argument: "
                        "Incident energy must be numeric and every element >= 0"
                    )
                ei = arr.astype(np.float64)
                is_scalar = False
            if narg == 2:
                if not _is_log_scalar(args[1]):
                    raise ValueError(
                        "HERBERT:IX_fermi_chopper:invalid_argument: "
                        "phase must be true or false (or 1 or 0)"
                    )
                phase = bool(_to_float(args[1])) if not isinstance(args[1], (bool, np.bool_)) else bool(args[1])
            else:
                phase = self._phase
            return ei, phase, is_scalar
        raise ValueError(
            "HERBERT:IX_fermi_chopper:invalid_argument: "
            "Check number of input arguments"
        )

    def _parse_t_and_phase(
        self, default_t: Optional[float], *args: Any
    ) -> Tuple[Optional[Union[float, np.ndarray]], float, bool, bool, bool]:
        """Parse optional ``(t, ei, phase)`` or ``('-reverse',)`` arguments."""
        t: Optional[Union[float, np.ndarray]] = default_t
        ei: float = self._energy
        phase: bool = self._phase
        t_given: bool = False
        t_is_scalar: bool = True

        narg = len(args)
        if narg >= 1:
            if isinstance(args[0], str):
                arg0 = args[0]
                if narg == 1 and len(arg0) >= 2 and "-reverse".startswith(arg0.lower()):
                    phase = not self._phase
                else:
                    raise ValueError(
                        "HERBERT:IX_fermi_chopper:invalid_argument: "
                        "Unrecognised character string argument, or check number of arguments"
                    )
            else:
                raw_t = args[0]
                if isinstance(raw_t, (bool, np.bool_, str)):
                    raise ValueError(
                        "HERBERT:IX_fermi_chopper:invalid_argument: "
                        "Time value(s) cannot be NaN"
                    )
                if isinstance(raw_t, (int, float, np.number)):
                    if np.iscomplexobj(raw_t) or math.isnan(float(raw_t)):
                        raise ValueError(
                            "HERBERT:IX_fermi_chopper:invalid_argument: "
                            "Time value(s) cannot be NaN"
                        )
                    t = float(raw_t)
                    t_is_scalar = True
                else:
                    arr_t = np.asarray(raw_t)
                    if arr_t.dtype.kind not in "iuf" or np.iscomplexobj(arr_t) or np.any(np.isnan(arr_t)):
                        raise ValueError(
                            "HERBERT:IX_fermi_chopper:invalid_argument: "
                            "Time value(s) cannot be NaN"
                        )
                    t = arr_t.astype(np.float64)
                    t_is_scalar = False
                t_given = True

                if narg >= 2:
                    if isinstance(args[1], str):
                        arg1 = args[1]
                        if narg == 2 and len(arg1) >= 2 and "-reverse".startswith(arg1.lower()):
                            phase = not self._phase
                        else:
                            raise ValueError(
                                "HERBERT:IX_fermi_chopper:invalid_argument: "
                                "Unrecognised character string argument, or check number of arguments"
                            )
                    else:
                        if not _is_real_scalar(args[1]) or _to_float(args[1]) < 0:
                            raise ValueError(
                                "HERBERT:IX_fermi_chopper:invalid_argument: "
                                "Incident energy must be a scalar numeric >= 0"
                            )
                        ei = _to_float(args[1])
                        if narg == 3:
                            if not _is_log_scalar(args[2]):
                                raise ValueError(
                                    "HERBERT:IX_fermi_chopper:invalid_argument: "
                                    "Phase must be true or false (or 1 or 0)"
                                )
                            phase = (
                                bool(_to_float(args[2]))
                                if not isinstance(args[2], (bool, np.bool_))
                                else bool(args[2])
                            )
                        elif narg > 3:
                            raise ValueError(
                                "HERBERT:IX_fermi_chopper:invalid_argument: "
                                "Too many input arguments"
                            )
        return t, ei, phase, t_given, t_is_scalar

    def _get_pulse_props(
        self, ei: Union[float, np.ndarray], phase: bool
    ) -> Tuple[float, np.ndarray]:
        """Compute ``(pk_fwhh, gam)`` with ``pk_fwhh`` in seconds."""
        ei_arr = np.asarray(ei, dtype=np.float64)
        if (
            self._slit_width == 0.0
            and self._radius == 0.0
            and self._frequency == 0.0
            and self._curvature == 0.0
        ):
            pk_fwhh = 0.0
            gam = np.zeros_like(ei_arr, dtype=np.float64)
        else:
            vi = 1e6 * np.sqrt(ei_arr) / self._c_e_to_t
            omega = 2.0 * math.pi * self._frequency
            s = 2.0 * omega * self._curvature
            with np.errstate(divide="ignore", invalid="ignore"):
                pk_fwhh = (
                    self._slit_width / (2.0 * self._radius * omega)
                    if (self._radius * omega) != 0.0
                    else (0.0 if self._slit_width == 0.0 else math.inf)
                )
                inv_s = 1.0 / s if s != 0.0 else math.inf
                inv_vi = np.where(vi != 0.0, 1.0 / vi, np.inf)
                diff = np.abs(inv_s - inv_vi) if phase else np.abs(inv_s + inv_vi)
                if pk_fwhh != 0.0:
                    gam = (2.0 * self._radius / pk_fwhh) * diff
                else:
                    gam = np.where(diff == 0.0, 0.0, np.inf)
            if pk_fwhh == 0.0:
                gam = np.asarray(gam, dtype=np.float64)
                gam[vi == s] = 0.0
        return float(pk_fwhh), np.asarray(gam, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public physical methods
    # ------------------------------------------------------------------
    def transmission(self, *args: Any) -> Union[float, np.ndarray]:
        """Calculate the peak-normalised transmission of the Fermi chopper."""
        ei, phase, is_scalar = self._parse_ei_and_phase(*args)
        _, gam = self._get_pulse_props(ei, phase)

        f = np.zeros_like(gam, dtype=np.float64)
        i1 = gam < 1.0
        f[i1] = 1.0 - (gam[i1] ** 2) / 6.0

        i2 = (gam >= 1.0) & (gam < 4.0)
        rtgam = np.sqrt(gam[i2])
        f[i2] = rtgam * ((rtgam - 2.0) ** 2) * (rtgam + 4.0) / 6.0

        if is_scalar:
            return float(f.item())
        return f

    def pulse_width(
        self, *args: Any
    ) -> Tuple[Union[float, np.ndarray], float, Union[float, np.ndarray]]:
        """Calculate standard deviation, triangular FWHH, and actual FWHH (microseconds)."""
        ei, phase, is_scalar = self._parse_ei_and_phase(*args)
        pk_fwhh, gam = self._get_pulse_props(ei, phase)

        var = np.full(gam.shape, np.nan, dtype=np.float64)
        i1 = gam < 1.0
        var[i1] = ((pk_fwhh ** 2) / 6.0) * (
            (1.0 - (gam[i1] ** 4) / 10.0) / (1.0 - (gam[i1] ** 2) / 6.0)
        )

        i2 = (gam >= 1.0) & (gam < 4.0)
        rtgam = np.sqrt(gam[i2])
        var[i2] = ((pk_fwhh ** 2) / 6.0) * (
            0.6 * gam[i2] * ((rtgam - 2.0) ** 2) * (rtgam + 8.0) / (rtgam + 4.0)
        )

        dt = 1e6 * np.sqrt(var)
        pk_fwhh_us = float(1e6 * pk_fwhh)

        fwhh = np.full(gam.shape, np.nan, dtype=np.float64)
        ilo = gam < (4.0 / 7.0)
        fwhh[ilo] = pk_fwhh_us * (1.0 + gam[ilo] / 4.0)

        imed = (~ilo) & (gam < 4.0)
        g_med = gam[imed]
        fwhh[imed] = (
            pk_fwhh_us
            * (2.0 * g_med * (4.0 - g_med))
            / (2.0 * g_med + np.sqrt(2.0 * g_med * (4.0 + g_med)))
        )

        if is_scalar:
            return float(dt.item()), pk_fwhh_us, float(fwhh.item())
        return dt, pk_fwhh_us, fwhh

    def pulse_range(
        self, *args: Any
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Return ``(tlo, thi)`` in microseconds for the chopper pulse."""
        ei, phase, is_scalar = self._parse_ei_and_phase(*args)
        pk_fwhh, gam = self._get_pulse_props(ei, phase)

        thi = np.full(gam.shape, np.nan, dtype=np.float64)
        i1 = gam < 1.0
        thi[i1] = 1e6 * pk_fwhh

        i2 = (gam >= 1.0) & (gam < 4.0)
        rtgam = np.sqrt(gam[i2])
        thi[i2] = (1e6 * pk_fwhh) * (rtgam * (2.0 - rtgam))

        tlo = -thi
        if is_scalar:
            return float(tlo.item()), float(thi.item())
        return tlo, thi

    def pulse_shape(
        self, *args: Any, return_t: bool = False
    ) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate normalized pulse shape as a function of time (in microseconds)."""
        t, ei, phase, t_given, t_is_scalar = self._parse_t_and_phase(None, *args)
        pk_fwhh, gam_arr = self._get_pulse_props(ei, phase)
        gam = float(gam_arr.item())

        if not t_given:
            npnt = 500
            tlo, thi = self.pulse_range(ei, phase)
            if tlo == 0.0 and thi == 0.0:
                t_arr: Union[float, np.ndarray] = np.array([0.0], dtype=np.float64)
            else:
                t_arr = np.linspace(float(tlo), float(thi), npnt)
            t_is_scalar = False
        else:
            t_arr = np.asarray(t, dtype=np.float64)

        if pk_fwhh > 0.0:
            y = np.zeros_like(t_arr, dtype=np.float64)
            tau = np.abs(t_arr) / (1e6 * pk_fwhh)
            if gam < 1.0:
                ilo = tau < gam
                if gam > 0.0:
                    y[ilo] = 1.0 - ((gam + tau[ilo]) ** 2) / (4.0 * gam)
                ihi = (tau >= gam) & (tau < 1.0)
                y[ihi] = 1.0 - tau[ihi]
            elif gam < 4.0:
                rtgam = math.sqrt(gam)
                iok = tau < rtgam * (2.0 - rtgam)
                y[iok] = 1.0 - ((gam + tau[iok]) ** 2) / (4.0 * gam)
            y = y * (1e-6 / pk_fwhh)
        else:
            y = np.zeros_like(t_arr, dtype=np.float64)
            y[t_arr == 0.0] = np.inf

        if t_given and not return_t:
            if t_is_scalar:
                return float(y.item())
            return y
        return y, t_arr

    def partial_transmission(self, *args: Any) -> Union[float, np.ndarray]:
        """Integrated transmission from ``-inf`` to ``t`` (normalized to unity at peak energy)."""
        t, ei, phase, _, t_is_scalar = self._parse_t_and_phase(math.inf, *args)
        pk_fwhh, gam_arr = self._get_pulse_props(ei, phase)
        gam = float(gam_arr.item())

        t_arr = np.asarray(t, dtype=np.float64)
        f = np.zeros_like(t_arr, dtype=np.float64)
        pos_t = t_arr > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.abs(t_arr) / (1e6 * pk_fwhh) if pk_fwhh > 0.0 else np.where(t_arr == 0.0, 0.0, np.inf)

        if gam < 1.0:
            ilo = tau < gam
            if gam > 0.0:
                f[ilo] = 0.5 * ((1.0 - tau[ilo]) ** 2) - ((gam - tau[ilo]) ** 3) / (12.0 * gam)
            ihi = (tau >= gam) & (tau < 1.0)
            f[ihi] = 0.5 * ((1.0 - tau[ihi]) ** 2)
            f[pos_t] = 1.0 - (gam ** 2) / 6.0 - f[pos_t]
        elif gam < 4.0:
            rtgam = math.sqrt(gam)
            iok = tau < rtgam * (2.0 - rtgam)
            eta = rtgam * (2.0 - rtgam) - tau[iok]
            f[iok] = (eta ** 2) * (6.0 * rtgam - eta) / (12.0 * gam)
            f[pos_t] = rtgam * ((rtgam - 2.0) ** 2) * (rtgam + 4.0) / 6.0 - f[pos_t]

        if t_is_scalar:
            return float(f.item())
        return f

    def rand(
        self, *args: Any, rng: Optional[np.random.Generator] = None
    ) -> Union[float, np.ndarray]:
        """Generate random numbers from the chopper pulse distribution."""
        return self._pdf.rand(*args, rng=rng)

    def retain(
        self,
        t: Union[float, np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> Union[bool, np.ndarray]:
        """Retain or reject times based on the chopper pulse distribution."""
        return self._pdf.retain(t, rng=rng)
