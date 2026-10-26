"""Null instrument class (IX_null_inst).

Converted from:
  herbert_core/classes/instrument_classes/instrument/instruments/@IX_null_inst/
"""
from __future__ import annotations

from typing import Any, List

from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_inst import IX_inst


@register_serializable
class IX_null_inst(IX_inst):
    """Null instrument class indicating no instrument information is stored.

    Converted from MATLAB ``IX_null_inst``.
    """

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name = ""
        self._source = None

    def saveable_fields(self) -> List[str]:
        return ["name"]
