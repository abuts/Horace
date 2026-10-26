"""Abstract base instrument class (IX_inst).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_inst/
"""
from __future__ import annotations

from typing import Any, List, Optional

from pyhorace.core.hashable import hashable
from pyhorace.data.instr_classes.ix_moderator import IX_moderator


class IX_inst(hashable):
    """Base instrument class.

    Converted from MATLAB ``IX_inst``.
    """

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name: str = ""
        self._source: Optional[IX_moderator] = None

        if not args and not kwargs:
            return

        if len(args) == 1 and not kwargs and isinstance(args[0], IX_inst):
            other = args[0]
            self._name = other._name
            self._source = IX_moderator(other._source) if other._source is not None else None
            return

        pos_params = ["source", "name"]
        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, False, *args, **kwargs
        )
        if remains:
            raise ValueError(f"Unrecognised extra arguments: {remains}")

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        if val is None:
            self._name = ""
            return
        if not isinstance(val, str):
            raise ValueError("HEBERT:IX_inst:invalid_argument: Instrument name must be a character string.")
        self._name = val

    @property
    def source(self) -> IX_moderator:
        if self._source is None:
            self._source = IX_moderator()
        return self._source

    @source.setter
    def source(self, val: Optional[IX_moderator]) -> None:
        if val is None:
            self._source = IX_moderator()
            return
        if not isinstance(val, IX_moderator):
            raise ValueError("HEBERT:IX_inst:invalid_argument: Instrument source must be an IX_moderator object.")
        self._source = val

    def saveable_fields(self) -> List[str]:
        if self._source is not None:
            return ["source", "name"]
        return ["name"]
