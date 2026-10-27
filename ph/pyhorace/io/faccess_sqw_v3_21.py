"""Accessor for Horace SQW binary format version 3.21 (indirect geometry).

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v3_21/faccess_sqw_v3_21.m
"""

from __future__ import annotations

from typing import Any, Dict

from pyhorace.io.faccess_sqw_v3 import faccess_sqw_v3
from pyhorace.io.sqw_binfile_common import sqw_binfile_common
from pyhorace.io.sqw_fields_formatters import field_var_array


class faccess_sqw_v3_21(faccess_sqw_v3):
    """Accessor for Horace SQW binary format version 3.21 (indirect geometry)."""

    file_version_: float = 3.21

    def __init__(self, *args: Any) -> None:
        self.version_ = self.file_version_
        super().__init__(*args)

    def get_faccess_version(self) -> float:
        return 3.21

    @staticmethod
    def get_header_form(*args: Any) -> Dict[str, Any]:
        const = any(isinstance(a, str) and a.lower() == "-const" for a in args)
        header = sqw_binfile_common.get_header_form(*args)
        if not const:
            header["efix"] = field_var_array()
        return header

