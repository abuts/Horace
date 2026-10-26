"""IX_source neutron source information class.

Ported from Herbert MATLAB @IX_source class:
- herbert_core/classes/instrument_classes/instrument/@IX_source/IX_source.m
- herbert_core/classes/instrument_classes/instrument/@IX_source/private/convert_old_struct_.m
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import HashableArray, hashable


def _is_empty_val(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple)) and len(val) == 0:
        return True
    if isinstance(val, np.ndarray) and val.size == 0 and val.dtype.kind != "U":
        return True
    return False


def _is_numeric_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(val, (int, float, np.number)):
        return not (np.isnan(val) or np.isinf(val))
    if isinstance(val, np.ndarray) and val.size == 1 and np.issubdtype(val.dtype, np.number):
        v = float(val.flat[0])
        return not (np.isnan(v) or np.isinf(v))
    return False


class IX_source(hashable):
    """
    Neutron source information (name, target_name, operating frequency).
    Ported from Herbert MATLAB `@IX_source`.
    """

    _name: str = PrivateAttr(default="")
    _target_name: str = PrivateAttr(default="")
    _frequency: float = PrivateAttr(default=0.0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name = ""
        self._target_name = ""
        self._frequency = 0.0
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], IX_source) and not kwargs:
            other = args[0]
            self._name = other._name
            self._target_name = other._target_name
            self._frequency = other._frequency
            self._hash_value = other._hash_value
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_source):
                self._name = loaded._name
                self._target_name = loaded._target_name
                self._frequency = loaded._frequency
                self._hash_value = loaded._hash_value
            return

        if len(args) == 1 and not kwargs and _is_numeric_scalar(args[0]):
            self.frequency = args[0]
            return

        if (
            len(args) == 2
            and not kwargs
            and isinstance(args[0], str)
            and _is_numeric_scalar(args[1])
        ):
            self.name = args[0]
            self.frequency = args[1]
            return

        if len(args) >= 1 and _is_numeric_scalar(args[0]):
            self.frequency = args[0]
            args = args[1:]

        accept_params = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            accept_params, True, *args, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_source:invalid_argument: Unrecognized extra parameters provided "
                f"as input to IX_source constructor: {remains}"
            )

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: Any) -> None:
        if _is_empty_val(val):
            val = ""
        if not isinstance(val, str):
            raise ValueError(
                f"HERBERT:IX_source:invalid_argument: The source name must be a character string. "
                f"Its type is: {type(val).__name__} and value {val!r}"
            )
        self._name = val
        self.clear_hash()

    @property
    def target_name(self) -> str:
        return self._target_name

    @target_name.setter
    def target_name(self, val: Any) -> None:
        if _is_empty_val(val):
            val = ""
        if not isinstance(val, str):
            raise ValueError(
                f"HERBERT:IX_source:invalid_argument: The target name must be a character string. "
                f"Its type is: {type(val).__name__} and value {val!r}"
            )
        self._target_name = val
        self.clear_hash()

    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, val: Any) -> None:
        if _is_empty_val(val):
            val = 0.0
        if not _is_numeric_scalar(val):
            raise ValueError(
                "HERBERT:IX_source:invalid_argument: The target frequency must be a non-negative number"
            )
        fval = float(np.asarray(val).flat[0])
        if fval < 0:
            raise ValueError(
                "HERBERT:IX_source:invalid_argument: The target frequency must be a non-negative number"
            )
        self._frequency = fval
        self.clear_hash()

    # Private aliases
    @property
    def name_(self) -> str:
        return self._name

    @property
    def target_name_(self) -> str:
        return self._target_name

    @property
    def frequency_(self) -> float:
        return self._frequency

    # -------------------------------------------------------------------------
    # Methods & Serializable Interface
    # -------------------------------------------------------------------------

    def isempty(self) -> bool:
        """Return True if the source object is in its default empty state."""
        return (
            self._name == ""
            and self._target_name == ""
            and self._frequency == 0.0
        )

    def saveable_fields(self) -> Sequence[str]:
        return ("name", "target_name", "frequency")

    def class_version(self) -> int:
        return 2

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "IX_source"]:
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "name_" in inputs:
            inputs.pop("class_version_", None)
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs
        return inputs, self


__all__ = ["IX_source"]

