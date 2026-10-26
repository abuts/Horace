"""Divergence profile class (port of MATLAB ``IX_divergence_profile``)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.pdf_table import pdf_table


def _is_key_match(arg: str, valid_keys: Tuple[str, ...]) -> bool:
    """Return True if ``arg`` matches or is an abbreviation of any key in ``valid_keys``."""
    s = arg.lstrip("-") if arg.startswith("-") else arg
    if not s:
        return False
    s_low = s.lower()
    return any(k.lower().startswith(s_low) for k in valid_keys)


@register_serializable
class IX_divergence_profile(hashable):
    """Divergence profile object (port of MATLAB ``IX_divergence_profile``)."""

    _class_version: int = PrivateAttr(default=2)
    _name: str = PrivateAttr(default="")
    _angles: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0,), dtype=np.float64))
    _profile: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0,), dtype=np.float64))
    _mandatory_field_set: List[bool] = PrivateAttr(default_factory=lambda: [False, False])
    _pdf: pdf_table = PrivateAttr(default_factory=pdf_table)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 0 and len(kwargs) == 0:
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_divergence_profile):
                self._copy_from(loaded)
            return
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], IX_divergence_profile):
            self._copy_from(args[0])
            return

        pos_params = self.saveable_fields()
        if (
            len(args) >= 1
            and isinstance(args[0], str)
            and len(args[0]) > 0
            and not args[0].startswith("-")
            and not _is_key_match(args[0], pos_params)
        ):
            self.name = args[0]
            argi = args[1:]
        else:
            argi = args

        if "angle" in kwargs and "angles" not in kwargs:
            kwargs = dict(kwargs)
            kwargs["angles"] = kwargs.pop("angle")

        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, False, *argi, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_divergence_profile:invalid_argument: "
                f"Unrecognized extra parameters provided as input to IX_divergence_profile constructor: {remains!r}"
            )

    def _copy_from(self, other: IX_divergence_profile) -> None:
        self._name = other._name
        self._angles = other._angles.copy()
        self._profile = other._profile.copy()
        self._mandatory_field_set = list(other._mandatory_field_set)
        self._pdf = copy.deepcopy(other._pdf)
        self.clear_hash()

    def _clone_for_hash(self) -> IX_divergence_profile:
        cloned: IX_divergence_profile = super()._clone_for_hash()
        cloned._angles = self._angles.copy()
        cloned._profile = self._profile.copy()
        cloned._mandatory_field_set = list(self._mandatory_field_set)
        return cloned

    @classmethod
    def class_version(cls) -> int:
        return 2

    def saveable_fields(self, mandatory: Optional[bool] = None) -> Tuple[str, ...]:
        fields = ("angles", "profile", "name")
        if mandatory:
            return fields[:2]
        return fields

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
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "Profile name must be a character string (or empty string)"
            )

    @property
    def angle(self) -> np.ndarray:
        return self._angles

    @angle.setter
    def angle(self, val: Any) -> None:
        self.angles = val

    @property
    def angles(self) -> np.ndarray:
        return self._angles

    @angles.setter
    def angles(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._angles
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._angles = np.empty((0,), dtype=np.float64)
            self._mandatory_field_set[0] = False
            if self._do_check_combo_arg:
                self.check_combo_arg(not np.array_equal(val_old, self._angles))
            return
        if isinstance(val, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "Angles must be a vector of two or more monotonically increasing numbers"
            )
        arr = np.asarray(val)
        if (
            arr.ndim > 2
            or (arr.ndim == 2 and min(arr.shape) > 1)
            or arr.size < 2
            or np.iscomplexobj(arr)
            or arr.dtype.kind not in "iuf"
            or not np.all(np.isfinite(arr))
            or np.any(np.diff(arr.ravel()) < 0)
        ):
            raise ValueError(
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "Angles must be a vector of two or more monotonically increasing numbers"
            )
        self._angles = arr.astype(np.float64).ravel()
        self._mandatory_field_set[0] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(not np.array_equal(val_old, self._angles))

    @property
    def profile(self) -> np.ndarray:
        return self._profile

    @profile.setter
    def profile(self, val: Any) -> None:
        self.clear_hash()
        val_old = self._profile
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._profile = np.empty((0,), dtype=np.float64)
            self._mandatory_field_set[1] = False
            if self._do_check_combo_arg:
                self.check_combo_arg(not np.array_equal(val_old, self._profile))
            return
        if isinstance(val, (bool, np.bool_, str)):
            raise ValueError(
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "Profile must be a vector of non-negative numbers"
            )
        arr = np.asarray(val)
        if (
            arr.ndim > 2
            or (arr.ndim == 2 and min(arr.shape) > 1)
            or arr.size < 1
            or np.iscomplexobj(arr)
            or arr.dtype.kind not in "iuf"
            or not np.all(np.isfinite(arr))
            or np.any(arr.ravel() < 0)
        ):
            raise ValueError(
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "Profile must be a vector of non-negative numbers"
            )
        self._profile = arr.astype(np.float64).ravel()
        self._mandatory_field_set[1] = True
        if self._do_check_combo_arg:
            self.check_combo_arg(not np.array_equal(val_old, self._profile))

    @property
    def pdf(self) -> pdf_table:
        return self._pdf

    def pdf_table(self) -> pdf_table:
        """Return the internal ``pdf_table`` object."""
        return self._pdf

    def check_combo_arg(self, do_recompute_pdf: bool = True) -> IX_divergence_profile:
        """Validate interdependent properties and update internal PDF table."""
        self.clear_hash()
        if any(self._mandatory_field_set) and not all(self._mandatory_field_set):
            mandatory_field_names = self.saveable_fields(True)
            unset = [
                name
                for name, is_set in zip(mandatory_field_names, self._mandatory_field_set)
                if not is_set
            ]
            raise ValueError(
                f"HERBERT:IX_divergence_profile:invalid_argument: "
                f"Must give all mandatory properties namely: {mandatory_field_names!r}\n"
                f" Properties: {unset!r} have not been set"
            )
        if self._angles.size != self._profile.size:
            raise ValueError(
                "HERBERT:IX_divergence_profile:invalid_argument: "
                "The number of values of angles and profile must match"
            )
        if any(self._mandatory_field_set):
            if do_recompute_pdf:
                self._pdf = pdf_table(self._angles, self._profile)
        else:
            self._pdf = pdf_table()
        return self

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], IX_divergence_profile]:
        out = dict(inputs)
        if ("class_version_" in out and out["class_version_"] == 1) or "angle_" in out or "angles_" in out:
            out.pop("class_version_", None)
            out.pop("pdf_", None)
            new_out: Dict[str, Any] = {}
            for k, v in out.items():
                new_key = k[:-1] if k.endswith("_") else k
                new_out[new_key] = v
            out = new_out
        if "angle" in out:
            out["angles"] = out.pop("angle")
        return out, self

    def rand(
        self, *args: Any, rng: Optional[np.random.Generator] = None
    ) -> Union[float, np.ndarray]:
        """Return random numbers from the divergence profile distribution."""
        return self._pdf.rand(*args, rng=rng)

    def width(self, fac: float = 0.5) -> Tuple[float, float, float, float]:
        """Return ``(w, xmax, xlo, xhi)`` width of the divergence profile."""
        return self._pdf.width(fac)

    def mean(self) -> float:
        """Return mean of the divergence profile."""
        return self._pdf.mean()

    def var(self) -> Tuple[float, float]:
        """Return ``(variance, mean)`` of the divergence profile."""
        return self._pdf.var()

    def retain(
        self,
        angles: Union[float, np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> Union[bool, np.ndarray]:
        """Retain or reject angles based on the divergence profile."""
        return self._pdf.retain(angles, rng=rng)
