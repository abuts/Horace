"""Goniometer class defining sample crystal lattice and orientation angles."""

from __future__ import annotations

import copy
import warnings
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf import bmatrix, calc_proj_matrix, ubmatrix
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.hashable import HashableArray, hashable


class Goniometer(hashable):
    """Class defines the crystal lattice and orientation of a crystal in the spectrometer.

    Corresponds to MATLAB ``Goniometer`` in Herbert/Horace.
    """

    tol_: ClassVar[float] = 1e-9

    _ANG_PROP_NAMES: ClassVar[Tuple[str, ...]] = ("psi", "omega", "dpsi", "gl", "gs")
    _SHORT_CONSTR_FIELDS: ClassVar[Tuple[str, ...]] = (
        "psi",
        "u",
        "v",
        "omega",
        "dpsi",
        "gl",
        "gs",
    )
    _LAT_CONSTR_FIELDS: ClassVar[Tuple[str, ...]] = (
        "alatt",
        "angdeg",
        "psi",
        "u",
        "v",
        "omega",
        "dpsi",
        "gl",
        "gs",
    )
    _FULL_CONSTR_FIELDS: ClassVar[Tuple[str, ...]] = (
        "efix",
        "emode",
        "alatt",
        "angdeg",
        "u",
        "v",
        "psi",
        "omega",
        "dpsi",
        "gl",
        "gs",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._efix: float = 0.0
        self._emode: int = 0
        self._alatt: np.ndarray = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
        self._alatt_defined: bool = False
        self._angdeg: np.ndarray = np.array([90.0, 90.0, 90.0], dtype=float)
        self._angdeg_defined: bool = False
        self._u: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=float)
        self._v: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        # All angles are stored internally in degrees; psi_ starts as NaN (undefined -> 0)
        self._psi: float = float("nan")
        self._omega: float = 0.0
        self._dpsi: float = 0.0
        self._gl: float = 0.0
        self._gs: float = 0.0
        self._angular_is_degree: bool = True

        if len(args) == 0 and len(kwargs) == 0:
            return
        self.init(*args, **kwargs)

    # ------------------------------------------------------------------
    # Construction & initialization
    # ------------------------------------------------------------------
    def init(self, *args: Any, **kwargs: Any) -> "Goniometer":
        """Initialize or re-initialize the Goniometer instance."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, Goniometer):
                self._copy_goniometer_from(arg0)
                return self
            if isinstance(arg0, dict):
                return self._init_from_dict(arg0)

        # Handle mixed (struct/Goniometer, key-val...) calls
        arg_list = list(args)
        base_dict: Optional[Dict[str, Any]] = None
        if len(arg_list) >= 1 and isinstance(arg_list[0], (dict, Goniometer)):
            first = arg_list.pop(0)
            if isinstance(first, Goniometer):
                self._copy_goniometer_from(first)
            else:
                base_dict = dict(first)

        # Extract angular units from positional/key-value arguments and kwargs first
        arg_list, kwargs, unit_val = self.check_angular_units_present(arg_list, kwargs)

        self._do_check_combo_arg = False
        try:
            if base_dict is not None:
                self._apply_struct_dict(base_dict, override_units=(unit_val is None))

            if unit_val is not None:
                self.angular_units = unit_val

            pos_args, kv_dict = self._split_positional_and_kv(arg_list)

            # Merge Python kwargs into kv_dict
            for k, v in kwargs.items():
                kv_dict[k] = v

            # Determine positional field mapping
            n_pos = len(pos_args)
            if n_pos > 0:
                pos_fields = self._select_positional_fields(pos_args)
                if n_pos > len(pos_fields):
                    raise ValueError(
                        f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                        f"Too many positional arguments ({n_pos})."
                    )
                for fname, fval in zip(pos_fields[:n_pos], pos_args):
                    self._set_named_field(fname, fval)

            # Apply key-value pairs
            for fname, fval in kv_dict.items():
                self._set_named_field(fname, fval)
        finally:
            self._do_check_combo_arg = True

        self.check_combo_arg()
        return self

    def _copy_goniometer_from(self, other: "Goniometer") -> None:
        self._efix = float(other._efix)
        self._emode = int(other._emode)
        self._alatt = other._alatt.copy()
        self._alatt_defined = bool(other._alatt_defined)
        self._angdeg = other._angdeg.copy()
        self._angdeg_defined = bool(other._angdeg_defined)
        self._u = other._u.copy()
        self._v = other._v.copy()
        self._psi = float(other._psi)
        self._omega = float(other._omega)
        self._dpsi = float(other._dpsi)
        self._gl = float(other._gl)
        self._gs = float(other._gs)
        self._angular_is_degree = bool(other._angular_is_degree)
        self.clear_hash()

    def _init_from_dict(self, d: Dict[str, Any]) -> "Goniometer":
        if "serial_name" in d and "version" in d:
            loaded = self.loadobj(d)
            if isinstance(loaded, Goniometer):
                self._copy_goniometer_from(loaded)
                return self
        self._do_check_combo_arg = False
        try:
            self._apply_struct_dict(d, override_units=True)
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()
        return self

    def _apply_struct_dict(self, d: Dict[str, Any], override_units: bool = True) -> None:
        s = dict(d)
        if override_units:
            if "angular_units" in s:
                self.angular_units = s.pop("angular_units")
            elif "angular_is_degree" in s:
                self.angular_is_degree = bool(s.pop("angular_is_degree"))
            elif "degrees" in s:
                self.degrees = bool(s.pop("degrees"))
            elif "radians" in s:
                self.radians = bool(s.pop("radians"))
        else:
            s.pop("angular_units", None)
            s.pop("angular_is_degree", None)
            s.pop("degrees", None)
            s.pop("radians", None)

        for k, v in s.items():
            self._set_named_field(k, v)

    def _select_positional_fields(self, pos_args: List[Any]) -> Tuple[str, ...]:
        """Select positional field order based on argument shapes and count."""
        n_pos = len(pos_args)
        first = pos_args[0]
        first_size = np.size(first) if not isinstance(first, str) else -1

        # Short form: Goniometer(psi, u, v, omega, dpsi, gl, gs)
        if n_pos <= 7:
            if first_size == 1:
                if n_pos >= 4 and np.size(pos_args[3]) == 3:
                    # (efix, emode, alatt, angdeg, ...)
                    return self._FULL_CONSTR_FIELDS
                return self._SHORT_CONSTR_FIELDS
            elif first_size == 3:
                # (alatt, angdeg, psi, u, v, omega, dpsi, gl, gs)
                return self._LAT_CONSTR_FIELDS

        if first_size == 3:
            return self._LAT_CONSTR_FIELDS
        return self._FULL_CONSTR_FIELDS

    def _resolve_field_name(self, raw_name: str) -> str:
        if not isinstance(raw_name, str):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Field name must be a string, got {type(raw_name).__name__}"
            )
        name = raw_name.strip()
        if name.startswith("-"):
            warnings.warn(
                f"HERBERT:Serializable:deprecated: Keyword '{raw_name}' with leading '-' is deprecated; "
                f"use '{name.lstrip('-').strip()}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            name = name.lstrip("-").strip()

        if name == "cu":
            return "u"
        if name == "cv":
            return "v"

        valid_fields = set(self.construction_fields()) | {
            "u",
            "v",
            "cu",
            "cv",
            "angular_is_degree",
            "degrees",
            "radians",
        }
        if name in valid_fields:
            return name

        # Support unambiguous prefix matching only if exact match fails?
        # Wait: in test_goniometer.m, wrong key names must throw invalid_argument!
        raise ValueError(
            f"HERBERT:{self.__class__.__name__}:invalid_argument: "
            f"Unrecognized property or keyword '{raw_name}'."
        )

    def _set_named_field(self, raw_name: str, val: Any) -> None:
        fname = self._resolve_field_name(raw_name)
        setattr(self, fname, val)

    def _is_unit_string(self, val: Any) -> bool:
        if not isinstance(val, str):
            return False
        s = val.strip().lower()
        if s.startswith("-"):
            s = s[1:].strip()
        return s in ("d", "deg", "degree", "degrees", "r", "rad", "radian", "radians")

    def check_angular_units_present(
        self, arg_list: List[Any], kwargs: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, Any], Optional[str]]:
        """Extract angular units specification from positional/key-value args or kwargs."""
        out_args = list(arg_list)
        out_kwargs = dict(kwargs)
        unit_val: Optional[str] = None

        # Check kwargs first
        for k in list(out_kwargs.keys()):
            clean_k = k.lstrip("-").strip()
            if clean_k == "angular_units":
                if k.startswith("-"):
                    warnings.warn(
                        f"HERBERT:Serializable:deprecated: Keyword '{k}' is deprecated.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                unit_val = out_kwargs.pop(k)
            elif clean_k in ("angular_is_degree", "degrees"):
                val = out_kwargs.pop(k)
                unit_val = "deg" if bool(val) else "rad"
            elif clean_k == "radians":
                val = out_kwargs.pop(k)
                unit_val = "rad" if bool(val) else "deg"

        # Scan arg_list for ('angular_units', val) or ('-angular_units', val)
        i = 0
        while i < len(out_args):
            item = out_args[i]
            if isinstance(item, str):
                clean = item.strip()
                is_depr = clean.startswith("-")
                if is_depr:
                    clean = clean.lstrip("-").strip()
                if clean == "angular_units":
                    if i + 1 >= len(out_args):
                        raise ValueError(
                            f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                            f"Missing value for 'angular_units'."
                        )
                    if is_depr:
                        warnings.warn(
                            f"HERBERT:Serializable:deprecated: Keyword '{item}' is deprecated.",
                            DeprecationWarning,
                            stacklevel=3,
                        )
                    unit_val = out_args[i + 1]
                    del out_args[i : i + 2]
                    continue
                elif clean in ("angular_is_degree", "degrees"):
                    if i + 1 < len(out_args) and isinstance(out_args[i + 1], (bool, int, np.bool_)):
                        unit_val = "deg" if bool(out_args[i + 1]) else "rad"
                        del out_args[i : i + 2]
                        continue
                elif clean == "radians":
                    if i + 1 < len(out_args) and isinstance(out_args[i + 1], (bool, int, np.bool_)):
                        unit_val = "rad" if bool(out_args[i + 1]) else "deg"
                        del out_args[i : i + 2]
                        continue
            i += 1

        # Also check if a standalone trailing unit string ('deg' or 'rad') is present
        # e.g. Goniometer(10, [1,2,3], [3,2,1], 20, 2, 3, -3, 'deg')
        # or before key-value pairs
        i = 0
        while i < len(out_args):
            item = out_args[i]
            if self._is_unit_string(item):
                # Make sure it's not the value of a preceding key
                prev_is_key = i > 0 and isinstance(out_args[i - 1], str) and not self._is_unit_string(out_args[i - 1])
                if not prev_is_key:
                    s = item.strip()
                    if s.startswith("-"):
                        s = s.lstrip("-").strip()
                    unit_val = s
                    del out_args[i]
                    continue
            i += 1

        return out_args, out_kwargs, unit_val

    def _split_positional_and_kv(
        self, arg_list: List[Any]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Split a list of arguments into leading positional arguments and trailing key-value pairs."""
        # For Goniometer, all positional arguments are numeric (unless IX_experiment overrides string filename/filepath)
        first_kv_idx = len(arg_list)
        string_pos_fields = getattr(self, "_STRING_POS_FIELDS", ())

        for idx, item in enumerate(arg_list):
            if isinstance(item, str):
                # In IX_experiment, positional args 0 and 1 can be filename, filepath
                if idx < len(string_pos_fields):
                    # Check if item is a known keyword followed by a value when remaining count is even
                    clean = item.strip().lstrip("-").strip()
                    known_keys = set(self.construction_fields()) | {"u", "v", "cu", "cv"}
                    if clean in known_keys and (len(arg_list) - idx) % 2 == 0:
                        first_kv_idx = idx
                        break
                    continue
                first_kv_idx = idx
                break

        pos_args = arg_list[:first_kv_idx]
        kv_args = arg_list[first_kv_idx:]
        if len(kv_args) % 2 != 0:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Key-value arguments must come in pairs; got odd number ({len(kv_args)}) of trailing arguments: {kv_args}"
            )
        kv_dict: Dict[str, Any] = {}
        for i in range(0, len(kv_args), 2):
            k = kv_args[i]
            v = kv_args[i + 1]
            if not isinstance(k, str):
                raise ValueError(
                    f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                    f"Expected string keyword at position {first_kv_idx + i}, got {type(k).__name__}"
                )
            # Validate keyword immediately
            self._resolve_field_name(k)
            kv_dict[k] = v
        return pos_args, kv_dict

    # ------------------------------------------------------------------
    # Serializable & hashable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Tuple[str, ...]:
        return (
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
        )

    def construction_fields(self) -> Tuple[str, ...]:
        return (
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        )

    def constructionFields(self) -> Tuple[str, ...]:
        """MATLAB-compatible camelCase alias for construction_fields."""
        return self.construction_fields()

    @staticmethod
    def fields_with_defaults() -> List[str]:
        return ["omega", "dpsi", "gl", "gs", "u", "v"]

    @staticmethod
    def fieldsWithDefaults() -> List[str]:
        """MATLAB-compatible camelCase alias for fields_with_defaults."""
        return Goniometer.fields_with_defaults()

    def hashable_fields(self) -> Tuple[str, ...]:
        return self.construction_fields()

    def to_bare_struct(self, recursive_bare: bool = False) -> Dict[str, Any]:
        S = super().to_bare_struct(recursive_bare=recursive_bare)
        if type(self) is Goniometer:
            S["angular_is_degree"] = self._angular_is_degree
        return S

    def from_bare_struct(self, S: Dict[str, Any]) -> "Goniometer":
        s_copy = dict(S)
        self._do_check_combo_arg = False
        try:
            # Must set angular units BEFORE angle values
            if "angular_units" in s_copy:
                self.angular_units = s_copy.pop("angular_units")
            elif "angular_is_degree" in s_copy:
                self.angular_is_degree = bool(s_copy.pop("angular_is_degree"))
            elif "degrees" in s_copy:
                self.degrees = bool(s_copy.pop("degrees"))
            elif "radians" in s_copy:
                self.radians = bool(s_copy.pop("radians"))

            if "cu" in s_copy and "u" not in s_copy:
                s_copy["u"] = s_copy.pop("cu")
            else:
                s_copy.pop("cu", None)
            if "cv" in s_copy and "v" not in s_copy:
                s_copy["v"] = s_copy.pop("cv")
            else:
                s_copy.pop("cv", None)

            for field in self.saveable_fields():
                if field in s_copy:
                    setattr(self, field, s_copy[field])
        finally:
            self._do_check_combo_arg = True

        self.check_combo_arg()
        return self

    def check_combo_arg(self) -> "Goniometer":
        """Validate interdependent properties (u and v must not be collinear)."""
        cross_prod = np.cross(self._u, self._v)
        if float(np.linalg.norm(cross_prod)) < self.tol_:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vectors u ({self._u}) and v ({self._v}) cannot be collinear."
            )
        self.clear_hash()
        return self

    # ------------------------------------------------------------------
    # Properties: efix, emode, alatt, angdeg
    # ------------------------------------------------------------------
    @property
    def efix(self) -> float:
        return self._efix

    @efix.setter
    def efix(self, val: Any) -> None:
        if isinstance(val, (bool, np.bool_)) or val is None:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "efix must be a non-negative numeric scalar."
            )
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "efix must be a non-negative numeric scalar."
            ) from exc
        if arr.size != 1:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "efix must be a scalar."
            )
        fval = float(arr.item())
        if np.isnan(fval) or fval < 0.0:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"efix must be >= 0, got {fval}."
            )
        self._efix = fval
        self.clear_hash()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def emode(self) -> int:
        return self._emode

    @emode.setter
    def emode(self, val: Any) -> None:
        if isinstance(val, (bool, np.bool_)) or val is None:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "emode must be 0, 1, or 2."
            )
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "emode must be 0, 1, or 2."
            ) from exc
        if arr.size != 1:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "emode must be a scalar."
            )
        fval = float(arr.item())
        if fval not in (0.0, 1.0, 2.0):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"emode must be 0, 1, or 2, got {val}."
            )
        self._emode = int(fval)
        self.clear_hash()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def alatt(self) -> np.ndarray:
        return self._alatt.copy()

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._check_and_set_alatt(val)

    def _check_and_set_alatt(self, val: Any) -> None:
        if isinstance(val, (bool, np.bool_, str)) or val is None:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "alatt must be a 3-element positive numeric vector or scalar."
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "alatt must be numeric."
            ) from exc
        if arr.size == 0:
            self._alatt = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
            self._alatt_defined = False
            self.clear_hash()
            return
        if arr.size == 1:
            arr = np.repeat(arr[0], 3)
        elif arr.size != 3:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"alatt must have 1 or 3 elements, got {arr.size}."
            )
        if np.any(np.isnan(arr)) or np.any(arr <= self.tol_):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Lattice parameters alatt must be positive, got {arr}."
            )
        self._alatt = arr.astype(float)
        self._alatt_defined = True
        self.clear_hash()

    @property
    def angdeg(self) -> np.ndarray:
        return self._angdeg.copy()

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._check_and_set_angdeg(val)

    def _check_and_set_angdeg(self, val: Any) -> None:
        if isinstance(val, (bool, np.bool_, str)) or val is None:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "angdeg must be a 3-element positive numeric vector in (0, 180)."
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "angdeg must be numeric."
            ) from exc
        if arr.size == 0:
            self._angdeg = np.array([90.0, 90.0, 90.0], dtype=float)
            self._angdeg_defined = False
            self.clear_hash()
            return
        if arr.size == 1:
            arr = np.repeat(arr[0], 3)
        elif arr.size != 3:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"angdeg must have 1 or 3 elements, got {arr.size}."
            )
        if np.any(np.isnan(arr)) or np.any(arr <= self.tol_) or np.any(arr >= 180.0):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Lattice angles angdeg must be in (0, 180) degrees, got {arr}."
            )
        self._angdeg = arr.astype(float)
        self._angdeg_defined = True
        self.clear_hash()

    # ------------------------------------------------------------------
    # Properties: u, v, cu, cv
    # ------------------------------------------------------------------
    @property
    def u(self) -> np.ndarray:
        return self._u.copy()

    @u.setter
    def u(self, val: Any) -> None:
        self._u = self._check_and_set_uv(val, "u")
        self.clear_hash()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @v.setter
    def v(self, val: Any) -> None:
        self._v = self._check_and_set_uv(val, "v")
        self.clear_hash()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def cu(self) -> np.ndarray:
        return self.u

    @cu.setter
    def cu(self, val: Any) -> None:
        self.u = val

    @property
    def cv(self) -> np.ndarray:
        return self.v

    @cv.setter
    def cv(self, val: Any) -> None:
        self.v = val

    def _check_and_set_uv(self, val: Any, name: str) -> np.ndarray:
        if isinstance(val, (bool, np.bool_, str)) or val is None:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vector {name} must be a 3-element numeric vector."
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vector {name} must be numeric."
            ) from exc
        if arr.size == 1:
            arr = np.repeat(arr[0], 3)
        elif arr.size != 3:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vector {name} must have 3 elements, got {arr.size}."
            )
        if np.any(np.isnan(arr)):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vector {name} cannot contain NaN."
            )
        small = np.abs(arr) < self.tol_
        if np.all(small):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Vector {name} cannot be a 0-vector."
            )
        arr[small] = 0.0
        return arr.astype(float)

    # ------------------------------------------------------------------
    # Properties: psi, omega, dpsi, gl, gs & angular units
    # ------------------------------------------------------------------
    def _from_deg(self, deg_val: float) -> float:
        if self._angular_is_degree:
            return float(deg_val)
        return float(deg_val) * (np.pi / 180.0)

    def _check_angular_set(self, val: Any, name: str, allow_nan: bool = False) -> float:
        if isinstance(val, (bool, np.bool_, str)) or val is None:
            if allow_nan and val is None:
                return float("nan")
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Angle {name} must be a numeric scalar."
            )
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Angle {name} must be a numeric scalar."
            ) from exc
        if allow_nan and arr.size == 0:
            return float("nan")
        if arr.size != 1:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Angle {name} must be a scalar, got size {arr.size}."
            )
        fval = float(arr.item())
        if np.isnan(fval):
            if allow_nan:
                return float("nan")
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Angle {name} cannot be NaN."
            )
        max_angle = 360.0 if self._angular_is_degree else 2.0 * np.pi
        if abs(fval) > max_angle:
            unit_name = "degrees" if self._angular_is_degree else "radians"
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"Angle {name}={fval} out of range [-{max_angle}, {max_angle}] {unit_name}."
            )
        return fval if self._angular_is_degree else fval * (180.0 / np.pi)

    @property
    def psi(self) -> float:
        if np.isnan(self._psi):
            return 0.0
        return self._from_deg(self._psi)

    @psi.setter
    def psi(self, val: Any) -> None:
        self._psi = self._check_angular_set(val, "psi", allow_nan=True)
        self.clear_hash()

    @property
    def omega(self) -> float:
        return self._from_deg(self._omega)

    @omega.setter
    def omega(self, val: Any) -> None:
        self._omega = self._check_angular_set(val, "omega")
        self.clear_hash()

    @property
    def dpsi(self) -> float:
        return self._from_deg(self._dpsi)

    @dpsi.setter
    def dpsi(self, val: Any) -> None:
        self._dpsi = self._check_angular_set(val, "dpsi")
        self.clear_hash()

    @property
    def gl(self) -> float:
        return self._from_deg(self._gl)

    @gl.setter
    def gl(self, val: Any) -> None:
        self._gl = self._check_angular_set(val, "gl")
        self.clear_hash()

    @property
    def gs(self) -> float:
        return self._from_deg(self._gs)

    @gs.setter
    def gs(self, val: Any) -> None:
        self._gs = self._check_angular_set(val, "gs")
        self.clear_hash()

    @property
    def angular_units(self) -> str:
        return "deg" if self._angular_is_degree else "rad"

    @angular_units.setter
    def angular_units(self, val: Any) -> None:
        if not isinstance(val, str):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"angular_units must be 'deg' or 'rad', got {val!r}."
            )
        s = val.strip().lower()
        if s.startswith("-"):
            s = s[1:].strip()
        if s in ("d", "deg", "degree", "degrees"):
            self._angular_is_degree = True
        elif s in ("r", "rad", "radian", "radians"):
            self._angular_is_degree = False
        else:
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                f"angular_units must be 'deg' or 'rad', got {val!r}."
            )
        self.clear_hash()

    @property
    def angular_is_degree(self) -> bool:
        return self._angular_is_degree

    @angular_is_degree.setter
    def angular_is_degree(self, val: Any) -> None:
        if not isinstance(val, (bool, int, np.bool_)):
            raise ValueError(
                f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                "angular_is_degree must be a boolean."
            )
        self._angular_is_degree = bool(val)
        self.clear_hash()

    @property
    def degrees(self) -> bool:
        return self._angular_is_degree

    @degrees.setter
    def degrees(self, val: Any) -> None:
        self.angular_is_degree = bool(val)

    @property
    def radians(self) -> bool:
        return not self._angular_is_degree

    @radians.setter
    def radians(self, val: Any) -> None:
        self.angular_is_degree = not bool(val)

    @property
    def undef_fields(self) -> List[str]:
        if np.isnan(self._psi):
            return ["psi"]
        return []

    # ------------------------------------------------------------------
    # Query & conversion methods
    # ------------------------------------------------------------------
    def is_defined(self, field_name: str) -> bool:
        """Return True if the specified field has been explicitly defined."""
        if field_name == "psi":
            return not np.isnan(self._psi)
        if field_name == "alatt":
            return self._alatt_defined
        if field_name == "angdeg":
            return self._angdeg_defined
        if field_name in self.construction_fields():
            return True
        raise ValueError(
            f"HERBERT:{self.__class__.__name__}:invalid_argument: "
            f"Unknown field '{field_name}'."
        )

    def alatt_defined(self) -> bool:
        return self._alatt_defined

    def angdeg_defined(self) -> bool:
        return self._angdeg_defined

    def lattice_parameters_defined(self) -> bool:
        return self._alatt_defined and self._angdeg_defined

    def convert_to_radians(self) -> "Goniometer":
        self.angular_is_degree = False
        return self

    def convert_to_degrees(self) -> "Goniometer":
        self.angular_is_degree = True
        return self

    def get_gon_axis(self, nargout: Optional[int] = None) -> Any:
        """Return goniometer axes and angles as a dict (1 output) or tuple (multiple outputs)."""
        nout = _infer_nargout(default=1, explicit_nargout=nargout)
        if nout == 1:
            return {
                "u": self.u,
                "v": self.v,
                "psi": self.psi,
                "omega": self.omega,
                "dpsi": self.dpsi,
                "gl": self.gl,
                "gs": self.gs,
                "angular_units": self.angular_units,
            }
        vals = (self.u, self.v, self.psi, self.omega, self.dpsi, self.gl, self.gs)
        return vals[:nout]

    # ------------------------------------------------------------------
    # Crystallographic & projection matrix calculations
    # ------------------------------------------------------------------
    def bmatrix(self, nargout: Optional[int] = None) -> Any:
        """Calculate B-matrix from current lattice parameters."""
        return bmatrix(self._alatt, self._angdeg, nargout=nargout)

    def ubmatrix(self, b: Optional[np.ndarray] = None, nargout: Optional[int] = None) -> Any:
        """Calculate UB-matrix from u, v and B-matrix."""
        if b is None:
            b = bmatrix(self._alatt, self._angdeg, nargout=1)
        return ubmatrix(self._u, self._v, b, nargout=nargout)

    def get_gon_rot_matrix(self, b: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate 3x3 spectrometer-to-Crystal-Cartesian rotation matrix given B-matrix."""
        if b is None:
            b = bmatrix(self._alatt, self._angdeg, nargout=1)
        _, u_mat = ubmatrix(self._u, self._v, b, nargout=2)
        psi_rad = 0.0 if np.isnan(self._psi) else np.deg2rad(self._psi)
        omega_rad = np.deg2rad(self._omega)
        dpsi_rad = np.deg2rad(self._dpsi)
        gl_rad = np.deg2rad(self._gl)
        gs_rad = np.deg2rad(self._gs)
        return calc_proj_matrix(
            b, u_mat, "", "", psi_rad, omega_rad, dpsi_rad, gl_rad, gs_rad, 1, nargout=1
        )

    def calc_projections(
        self,
        n_matrix: Optional[int] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Calculate projection matrices (spec_to_cc, u_to_rlu, spec_to_rlu).

        Parameters
        ----------
        n_matrix : int, optional
            If 1, 2, or 3, returns only that matrix:
            1 -> spec_to_cc, 2 -> u_to_rlu, 3 -> spec_to_rlu.
        nargout : int, optional
            Explicit number of return values when ``n_matrix`` is None.
        """
        psi_rad = 0.0 if np.isnan(self._psi) else np.deg2rad(self._psi)
        omega_rad = np.deg2rad(self._omega)
        dpsi_rad = np.deg2rad(self._dpsi)
        gl_rad = np.deg2rad(self._gl)
        gs_rad = np.deg2rad(self._gs)

        spec_to_cc, u_to_rlu, spec_to_rlu = calc_proj_matrix(
            self._alatt,
            self._angdeg,
            self._u,
            self._v,
            psi_rad,
            omega_rad,
            dpsi_rad,
            gl_rad,
            gs_rad,
            nargout=3,
        )
        mats = (spec_to_cc, u_to_rlu, spec_to_rlu)
        if n_matrix is not None:
            idx = int(n_matrix)
            if idx not in (1, 2, 3):
                raise ValueError(
                    f"HERBERT:{self.__class__.__name__}:invalid_argument: "
                    f"Matrix index must be 1, 2, or 3, got {n_matrix}."
                )
            return mats[idx - 1]

        nout = _infer_nargout(default=3, explicit_nargout=nargout)
        if nout == 1:
            return spec_to_cc
        if nout == 2:
            return spec_to_cc, u_to_rlu
        return mats

    def calc_proj_matrix(
        self,
        alatt: Any = None,
        angdeg: Any = None,
        n_matrix: int = 1,
        nargout: Optional[int] = None,
    ) -> Any:
        """Set lattice parameters (if given), switch angular_units to 'rad', and compute projection matrix.

        Supports calling as:
        - ``ol.calc_proj_matrix(alatt, angdeg, n_matrix=1)``
        - ``ol.calc_proj_matrix(n_matrix)``
        - ``[mat, ol] = ol.calc_proj_matrix(...)``
        """
        if angdeg is None and alatt is not None and np.size(alatt) == 1:
            n_matrix = int(alatt)
        else:
            if alatt is not None:
                self.alatt = alatt
            if angdeg is not None:
                self.angdeg = angdeg

        self.angular_units = "rad"
        mat = self.calc_projections(n_matrix=n_matrix)

        default_nout = 2 if angdeg is not None else 1
        nout = _infer_nargout(default=default_nout, explicit_nargout=nargout)
        if nout == 2:
            return mat, self
        return mat


goniometer = Goniometer

