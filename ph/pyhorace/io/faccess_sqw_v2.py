"""SQW binary file accessor for Horace format v2 (and legacy v1/v3.0)."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

from pyhorace.io.sqw_binfile_common import sqw_binfile_common
from pyhorace.io.sqw_fopen import obj_init


class faccess_sqw_v2(sqw_binfile_common):
    """Class to access Horace sqw files written by Horace v1-v2."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 2.0

    def put_sqw_footer(self, *args: Any) -> "faccess_sqw_v2":
        return self

    def should_load_stream(
        self, head_struc: Dict[str, Any], fid: int
    ) -> Tuple[bool, Optional[obj_init], str]:
        hs = dict(head_struc)
        if hs.get("version") == 1:
            hs["version"] = 2
        elif hs.get("version") == 3:
            warnings.warn(
                "Horace File format version 3 is treated as version 2. Extra file version 3 data are lost",
                UserWarning,
                stacklevel=2,
            )
            hs["version"] = 2
        return super().should_load_stream(hs, fid)

