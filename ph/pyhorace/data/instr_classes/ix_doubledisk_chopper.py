"""Double-disk chopper class (port of MATLAB ``IX_doubledisk_chopper``)."""

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


def _is_key_match(arg: str, valid_keys: Tuple[str, ...]) -> bool:
    s = arg.lstrip("-") if arg.startswith("-") else arg
    if not s:
        return False
    s_low = s.lower()
    return any(k.lower().startswith(s_low) for k in valid_keys)


def _conv_hh(t: np.ndarray, T1: float, T2: float) -> np.ndarray:
    """Convolution of two normalised hat functions of widths ``T1`` and ``T2``."""
    y = np.zeros_like(t, dtype=np.float64)
    Tmax = max(T1, T2)
    Tmin = min(T1, T2)
    if Tmax > 0.0:
        abs_t = np.abs(t)
        t_plateau = (Tmax - Tmin) / 2.0
        t_edge = (Tmax + Tmin) / 2.0
        i_flat = abs_t <= t_plateau
        y[i_flat] = 1.0 / Tmax
        if Tmin > 0.0:
            i_slope = (abs_t > t_plateau) & (abs_t < t_edge)
            y[i_slope] = (t_edge - abs_t[i_slope]) / (Tmax * Tmin)
    else:
        y[t == 0.0] = np.inf
    return y


def _conv_hh_cdf(t: np.ndarray, T1: float, T2: float) -> np.ndarray:
    """Cumulative integral from ``-inf`` to ``t`` of ``_conv_hh(t, T1, T2)``."""
    f = np.zeros_like(t, dtype=np.float64)
    Tmax = max(T1, T2)
    Tmin = min(T1, T2)
    if Tmax == 0.0:
        f[t > 0.0] = 1.0
        f[t == 0.0] = 0.5
        return f

    t_edge = (Tmax + Tmin) / 2.0
    t_plat = (Tmax - Tmin) / 2.0

    f[t >= t_edge] = 1.0

    if Tmin > 0.0:
        i_left = (t > -t_edge) & (t < -t_plat)
        u_left = t[i_left] + t_edge
        f[i_left] = 0.5 * (u_left ** 2) / (Tmax * Tmin)

        i_mid = (t >= -t_plat) & (t <= t_plat)
        f[i_mid] = 0.5 + t[i_mid] / Tmax

        i_right = (t > t_plat) & (t < t_edge)
        u_right = t_edge - t[i_right]
        f[i_right] = 1.0 - 0.5 * (u_right ** 2) / (Tmax * Tmin)
    else:
        i_mid = (t > -t_plat) & (t < t_plat)
        f[i_mid] = 0.5 + t[i_mid] / Tmax

    return f


@register_serializable
class IX_doubledisk_chopper(hashable):
    """Double-disk chopper object (port of MATLAB ``IX_doubledisk_chopper``)."""

    _class_version: int = PrivateAttr(default=2)
    _name: str = PrivateAttr(default="")
    _distance: float = PrivateAttr(default=0.0)
    _frequency: float = PrivateAttr(default=0.0)
    _radius: float = PrivateAttr(default=0.0)
    _slot_width: float = PrivateAttr(default=0.0)
    _aperture_width: float = PrivateAttr(default=0.0)
    _aperture_width_defined: bool = PrivateAttr(default=False)
    _aperture_height: float = PrivateAttr(default=0.0)
    _jitter: float = PrivateAttr(default=0.0)
    _mandatory_field_set: List[bool] = PrivateAttr(
        default_factory=lambda: [False, False, False, False]
    )
    _pdf: pdf_table = PrivateAttr(default_factory=lambda: pdf_table(0.0, np.inf))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 0 and len(kwargs) == 0:
            self._pdf = self._recompute_pdf()
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_doubledisk_chopper):
                self._copy_from(loaded)
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], IX_doubledisk_chopper):
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
                    f"HERBERT:IX_doubledisk_chopper:invalid_argument: "
                    f"Unrecognized extra parameters provided as input to IX_doubledisk_chopper constructor: {remains!r}"
                )
        else:
            self.check_combo_arg(True)

    def _copy_from(self, other: IX_doubledisk_chopper) -> None:
        self._name = other._name
        self._distance = other._distance
        self._frequency = other._frequency
        self._radius = other._radius
        self._slot_width = other._slot_width
        self._aperture_width = other._aperture_width
        self._aperture_width_defined = other._aperture_width_defined
        self._aperture_height = other._aperture_height
        self._jitter = other._jitter
        self._mandatory_field_set = list(other._mandatory_field_set)
        self._pdf = copy.deepcopy(other._pdf)
        self.clear_hash()

    def _clone_for_hash(self) -> IX_doubledisk_chopper:
        cloned: IX_doubledisk_chopper = super()._clone_for_hash()
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
            "slot_width",
            "aperture_width",
            "aperture_height",
            "jitter",
            "name",
        )
        if mandatory:
            return fields[:4]
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
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
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
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
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
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
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
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Radius must be a non-negative numeric scalar"
            )
        self._radius = _to_float(val)
        self._mandatory_field_set[2] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._radius)

    @property
    def slot_width(self) -> float:
        return self._slot_width

    @slot_width.setter
    def slot_width(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._slot_width
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Slot width must be a non-negative numeric scalar"
            )
        self._slot_width = _to_float(val)
        self._mandatory_field_set[3] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(val_old != self._slot_width)

    @property
    def aperture_width(self) -> float:
        if self._aperture_width_defined:
            return self._aperture_width
        return self._slot_width

    @aperture_width.setter
    def aperture_width(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._aperture_width
        was_defined = self._aperture_width_defined
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._aperture_width_defined = False
            self._aperture_width = 0.0
            recalc_pdf = was_defined and (val_old != self._slot_width)
        elif _is_real_scalar(val) and _to_float(val) >= 0:
            new_val = _to_float(val)
            self._aperture_width = new_val
            self._aperture_width_defined = True
            if was_defined:
                recalc_pdf = val_old != new_val
            else:
                recalc_pdf = self._slot_width != new_val
        else:
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Aperture width must be a non-negative numeric scalar (or empty, to set equal to slot_width)"
            )
        if self._do_check_combo_arg:
            self.check_combo_arg(recalc_pdf)

    @property
    def aperture_height(self) -> float:
        return self._aperture_height

    @aperture_height.setter
    def aperture_height(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Aperture height must be a non-negative numeric scalar"
            )
        self._aperture_height = _to_float(val)

    @property
    def jitter(self) -> float:
        return self._jitter

    @jitter.setter
    def jitter(self, val: Any) -> None:
        self.clear_hash()
        if not _is_real_scalar(val) or _to_float(val) < 0:
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Jitter must be a non-negative numeric scalar"
            )
        self._jitter = _to_float(val)

    @property
    def pdf(self) -> pdf_table:
        return self._pdf

    @property
    def half_width(self) -> float:
        """Half of the full base width of the chopper pulse in microseconds."""
        return float(self.pulse_range()[1])

    def pdf_table(self) -> pdf_table:
        """Return the probability distribution table for the pulse shape."""
        return self._pdf

    # ------------------------------------------------------------------
    # Validation & serialization
    # ------------------------------------------------------------------
    def check_combo_arg(self, do_recompute_pdf: bool = True) -> IX_doubledisk_chopper:
        self.clear_hash()
        if not all(self._mandatory_field_set):
            mandatory_field_names = self.saveable_fields(True)
            unset = [
                name
                for name, is_set in zip(mandatory_field_names, self._mandatory_field_set)
                if not is_set
            ]
            raise ValueError(
                f"HERBERT:IX_doubledisk_chopper:invalid_argument: "
                f"Must give all mandatory properties namely: {mandatory_field_names!r}.\n"
                f" Properties: {unset!r} have not been set"
            )
        if self._aperture_width_defined and self._aperture_width < self._slot_width:
            warnings.warn(
                f"aperture_width ({self._aperture_width:g}) have been set smaller than the slot_width ({self._slot_width:g}).\n"
                f" Setting aperture_width equal to slot_width",
                UserWarning,
                stacklevel=2,
            )
            self._aperture_width = self._slot_width
            do_recompute_pdf = True

        if do_recompute_pdf:
            self._pdf = self._recompute_pdf()
        return self

    def _recompute_pdf(self) -> pdf_table:
        tlo, thi = self.pulse_range()
        if tlo == 0.0:
            return pdf_table(0.0, np.inf)
        y, t = self.pulse_shape()
        return pdf_table(t, y)

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_doubledisk_chopper]:
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
    # Private helpers
    # ------------------------------------------------------------------
    def _hat_times(self) -> Tuple[float, float]:
        """Return ``(T1, T2)`` hat widths in microseconds."""
        if (
            self._slot_width == 0.0
            and self._radius == 0.0
            and self._frequency == 0.0
            and self.aperture_width == 0.0
        ):
            return 0.0, 0.0
        denom = 2.0 * math.pi * self._radius * self._frequency
        if denom == 0.0:
            T_slot = 0.0 if self._slot_width == 0.0 else math.inf
            T_aperture = 0.0 if self.aperture_width == 0.0 else math.inf
        else:
            T_slot = 1e6 * self._slot_width / denom
            T_aperture = 1e6 * self.aperture_width / denom
        if T_aperture <= T_slot:
            T1 = T_aperture / 2.0
            T2 = T_slot - T1
        else:
            T1 = T_slot / 2.0
            T2 = T1
        return float(T1), float(T2)

    @staticmethod
    def _validate_ei(ei: Any) -> Tuple[np.ndarray, bool]:
        if isinstance(ei, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Incident energy must be numeric and every element >= 0"
            )
        if isinstance(ei, (int, float, np.number)):
            if np.iscomplexobj(ei) or math.isnan(float(ei)) or float(ei) < 0:
                raise ValueError(
                    "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                    "Incident energy must be numeric and every element >= 0"
                )
            return np.asarray(float(ei), dtype=np.float64), True
        arr = np.asarray(ei)
        if arr.dtype.kind not in "iuf" or np.iscomplexobj(arr) or np.any(np.isnan(arr)) or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Incident energy must be numeric and every element >= 0"
            )
        return arr.astype(np.float64), False

    # ------------------------------------------------------------------
    # Public physical methods
    # ------------------------------------------------------------------
    def transmission(self, ei: Optional[Any] = None) -> Union[float, np.ndarray]:
        """Calculate the transmission of the double-disk chopper (always 1)."""
        if ei is None:
            return 1.0
        ei_arr, is_scalar = self._validate_ei(ei)
        if is_scalar:
            return 1.0
        return np.ones_like(ei_arr, dtype=np.float64)

    def pulse_width(
        self, ei: Optional[Any] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Return ``(dt, fwhh)`` in microseconds."""
        T1, T2 = self._hat_times()
        dt_val = math.sqrt((T1 ** 2 + T2 ** 2) / 12.0)
        fwhh_val = T2
        if ei is None:
            return float(dt_val), float(fwhh_val)
        ei_arr, is_scalar = self._validate_ei(ei)
        if is_scalar:
            return float(dt_val), float(fwhh_val)
        ones = np.ones_like(ei_arr, dtype=np.float64)
        return dt_val * ones, fwhh_val * ones

    def pulse_range(
        self, ei: Optional[Any] = None
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Return ``(tlo, thi)`` in microseconds."""
        T1, T2 = self._hat_times()
        thi_val = 0.5 * (T1 + T2)
        tlo_val = -thi_val
        if ei is None:
            return float(tlo_val), float(thi_val)
        ei_arr, is_scalar = self._validate_ei(ei)
        if is_scalar:
            return float(tlo_val), float(thi_val)
        ones = np.ones_like(ei_arr, dtype=np.float64)
        return tlo_val * ones, thi_val * ones

    def pulse_shape(
        self, t: Optional[Any] = None, return_t: bool = False
    ) -> Union[float, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Calculate normalized pulse shape as a function of time (in microseconds)."""
        T1, T2 = self._hat_times()
        if t is None:
            if T1 == 0.0 and T2 == 0.0:
                t_arr = np.array([0.0], dtype=np.float64)
            elif T1 == T2:
                t_arr = 0.5 * np.array([-(T1 + T2), 0.0, (T1 + T2)], dtype=np.float64)
            else:
                t_arr = 0.5 * np.array(
                    [-(T1 + T2), -(T2 - T1), (T2 - T1), (T1 + T2)], dtype=np.float64
                )
            y = _conv_hh(t_arr, T1, T2)
            return y, t_arr

        if isinstance(t, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Time array must be numeric and free of NaNs"
            )
        is_scalar = isinstance(t, (int, float, np.number))
        t_arr = np.asarray(t, dtype=np.float64)
        if np.iscomplexobj(t) or np.any(np.isnan(t_arr)):
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Time array must be numeric and free of NaNs"
            )
        y = _conv_hh(t_arr, T1, T2)
        if return_t:
            return y, t_arr
        if is_scalar:
            return float(y.item())
        return y

    def partial_transmission(
        self, tlo: Optional[Any] = None, thi: Optional[Any] = None
    ) -> Union[float, np.ndarray]:
        """Integrated transmission over time interval."""
        T1, T2 = self._hat_times()
        if tlo is None and thi is None:
            return 1.0
        if thi is None:
            is_scalar = isinstance(tlo, (int, float, np.number))
            t_arr = np.asarray(tlo, dtype=np.float64)
            f = _conv_hh_cdf(t_arr, T1, T2)
            return float(f.item()) if is_scalar else f

        is_scalar = isinstance(tlo, (int, float, np.number)) and isinstance(
            thi, (int, float, np.number)
        )
        tlo_arr, thi_arr = np.broadcast_arrays(
            np.asarray(tlo, dtype=np.float64), np.asarray(thi, dtype=np.float64)
        )
        if np.any(tlo_arr > thi_arr):
            raise ValueError(
                "HERBERT:IX_doubledisk_chopper:invalid_argument: "
                "Lower integration limit must be <= upper integration limit"
            )
        f = _conv_hh_cdf(thi_arr, T1, T2) - _conv_hh_cdf(tlo_arr, T1, T2)
        return float(f.item()) if is_scalar else f

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
