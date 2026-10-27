"""Accessor for Horace SQW binary format version 3.2 (legacy indirect geometry).

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v3_2/faccess_sqw_v3_2.m
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from pyhorace.core.exceptions import HoraceError
from pyhorace.io.faccess_sqw_v3 import faccess_sqw_v3
from pyhorace.io.sqw_binfile_common import sqw_binfile_common
from pyhorace.io.sqw_fields_formatters import field_var_array


class faccess_sqw_v3_2(faccess_sqw_v3):
    """Read-only accessor for Horace SQW binary format version 3.2 (indirect geometry)."""

    file_version_: ClassVar[float] = 3.2

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.version_ = self.file_version_

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 3.2

    @staticmethod
    def get_header_form(*args: Any) -> Dict[str, Any]:
        const = any(isinstance(a, str) and a.lower() == "-const" for a in args)
        header = sqw_binfile_common.get_header_form(*args)
        if not const:
            header["efix"] = field_var_array()
        return header

    def put_sqw(self, *args: Any) -> "faccess_sqw_v3_2":
        raise HoraceError(
            "FACCESS_SQW_V3_2:not_implemented",
            "Writing indirect geometry files in format v3.2 is not supported. Use faccess_sqw_v3_21 or faccess_sqw_v4.",
        )

    def put_dnd(self, *args: Any) -> "faccess_sqw_v3_2":
        raise HoraceError(
            "FACCESS_SQW_V3_2:not_implemented",
            "Writing indirect geometry files in format v3.2 is not supported.",
        )

    def put_headers(self, *args: Any) -> "faccess_sqw_v3_2":
        raise HoraceError(
            "FACCESS_SQW_V3_2:not_implemented",
            "Writing indirect geometry files in format v3.2 is not supported.",
        )

