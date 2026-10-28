"""IX_samp base sample class definition.

Ported from Herbert MATLAB @IX_samp class:
- herbert_core/classes/instrument_classes/instrument/@IX_samp/IX_samp.m
- herbert_core/classes/instrument_classes/instrument/@IX_samp/init.m
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import hashable
from pyhorace.data.instr_classes.ix_mosaic import _is_numeric_array_or_scalar


class IX_samp(hashable):
    """
    Base sample class definition storing lattice parameters (`alatt`, `angdeg`)
    and sample `name`.
    Ported from Herbert MATLAB `@IX_samp`.
    """

    fields_to_save_: ClassVar[Tuple[str, ...]] = ("alatt", "angdeg", "name")

    _name: str = PrivateAttr(default="")
    _alatt: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([], dtype=float)
    )
    _angdeg: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([], dtype=float)
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name = ""
        self._alatt = np.array([], dtype=float)
        self._angdeg = np.array([], dtype=float)
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_samp):
                self._copy_base_state_from(loaded)
            return

        self.init(*args, **kwargs)

    def _copy_base_state_from(self, other: "IX_samp") -> None:
        self._name = other._name
        self._alatt = other._alatt.copy()
        self._angdeg = other._angdeg.copy()
        self._hash_value = other._hash_value

    def __copy__(self) -> "IX_samp":
        cloned = type(self)()
        cloned._copy_base_state_from(self)
        cloned._do_check_combo_arg = self._do_check_combo_arg
        return cloned

    def _clone_for_hash(self) -> "IX_samp":
        return self.__copy__()

    def init(self, *args: Any, **kwargs: Any) -> "IX_samp":
        """
        Initialize an `IX_samp` object from positional and/or key-value arguments.
        Ported from `@IX_samp/init.m`.
        """
        fields_to_set = list(self.saveable_fields())
        if (
            len(args) >= 1
            and isinstance(args[0], str)
            and not args[0].startswith("-")
            and args[0] not in fields_to_set
        ):
            self.name = args[0]
            args = args[1:]
            if not args and not kwargs:
                return self

        _, remains = self.set_positional_and_key_val_arguments(
            fields_to_set, False, *args, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_samp:invalid_argument: Unrecognized extra parameters provided "
                f"as input to IX_samp constructor: {remains}"
            )
        return self

    # -------------------------------------------------------------------------
    # Protected Getters & Setters (Overridable by IX_null_sample)
    # -------------------------------------------------------------------------

    def _get_name(self) -> str:
        return self._name

    def _set_name(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            val = ""
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:IX_samp:invalid_argument: "
                "Sample name must be a character string (or empty)"
            )
        self._name = val
        self.clear_hash()

    def _get_lattice(self) -> np.ndarray:
        return self._alatt

    def _set_lattice(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._alatt = np.array([], dtype=float)
            self.clear_hash()
            return

        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_samp:invalid_argument: "
                "Lattice parameters must be a three-vector of numbers all greater than zero"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.array([arr[0], arr[0], arr[0]], dtype=float)
        if arr.size != 3 or np.any(arr <= 0.0):
            raise ValueError(
                "HERBERT:IX_samp:invalid_argument: "
                "Lattice parameters must be a three-vector of numbers all greater than zero"
            )
        self._alatt = arr
        self.clear_hash()

    def _get_angles(self) -> np.ndarray:
        return self._angdeg

    def _set_angles(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._angdeg = np.array([], dtype=float)
            self.clear_hash()
            return

        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_samp:invalid_argument: "
                "Lattice angles must be a three-vector of numbers all greater than 0 and less than 180 degrees"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.array([arr[0], arr[0], arr[0]], dtype=float)
        if arr.size != 3 or np.any(arr <= 0.0) or np.any(arr >= 180.0):
            raise ValueError(
                "HERBERT:IX_samp:invalid_argument: "
                "Lattice angles must be a three-vector of numbers all greater than 0 and less than 180 degrees"
            )
        self._angdeg = arr
        self.clear_hash()

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._get_name()

    @name.setter
    def name(self, val: Any) -> None:
        self._set_name(val)

    @property
    def alatt(self) -> np.ndarray:
        return self._get_lattice()

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._set_lattice(val)

    @property
    def angdeg(self) -> np.ndarray:
        return self._get_angles()

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._set_angles(val)

    # Private property aliases
    @property
    def name_(self) -> str:
        return self._name

    @property
    def alatt_(self) -> np.ndarray:
        return self._alatt

    @property
    def angdeg_(self) -> np.ndarray:
        return self._angdeg

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def lattice_defined(self) -> bool:
        """Return True if both `alatt` and `angdeg` are non-empty."""
        return self.alatt.size > 0 and self.angdeg.size > 0

    def isempty(self) -> bool:
        """Return False for an instantiated sample object."""
        return False

    @staticmethod
    def cell_is_class(ca: Sequence[Any]) -> Tuple[bool, str]:
        """Check if all elements of a sequence are `IX_samp` instances of the same type."""
        if not isinstance(ca, (list, tuple)):
            return False, "Input is not a sequence"
        if not all(isinstance(x, IX_samp) for x in ca):
            return False, "Not all elements have class IX_samp"
        if len(ca) > 1:
            first_cls = type(ca[0])
            if not all(type(x) is first_cls for x in ca[1:]):
                return False, "Not all elements have the same IX_samp subclass"
        return True, ""

    def xxx(self) -> "IX_samp":
        """Test dummy method ported from `@IX_samp/IX_samp.m`."""
        self.name = "xxx"
        return self

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def saveable_fields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return ("alatt", "angdeg", "name")

    def saveableFields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return self.saveable_fields(mandatory=mandatory)

    def class_version(self) -> int:
        return 0


__all__ = ["IX_samp"]
