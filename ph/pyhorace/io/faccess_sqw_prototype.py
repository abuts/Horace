"""SQW binary file accessor for Horace prototype (pre-v1 / v0) format."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.io.sqw_binfile_common import sqw_binfile_common
from pyhorace.io.sqw_fopen import _ShouldLoadResult, do_fseek, obj_init, pack_io_outputs


def _convert_signal_error(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert cumulative signal and variance to per-pixel mean signal and variance."""
    if "s" in data and "e" in data and "npix" in data:
        s = np.asarray(data["s"]).copy()
        e = np.asarray(data["e"]).copy()
        npix = np.asarray(data["npix"])
        nonzero = npix != 0
        s[nonzero] = s[nonzero] / npix[nonzero]
        e[nonzero] = e[nonzero] / (npix[nonzero] ** 2)
        data["s"] = s
        data["e"] = e
    return data


class faccess_sqw_prototype(sqw_binfile_common):
    """Class to access legacy Horace prototype sqw files (written before 2008)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 0.0

    def get_sqw_type(self) -> bool:
        return True

    def init_from_sqw_obj(self, *args: Any) -> "faccess_sqw_prototype":
        raise HoraceError(
            "SQW_FILE_IO:runtime_error",
            "prototype dataset accessor can not be initiated by data object",
        )

    def init_from_sqw_file(self, *args: Any) -> "faccess_sqw_prototype":
        self.main_header_pos_ = 0
        do_fseek(self.file_id_, 0, "bof")
        self.check_error_report_fail_(
            "init_from_sqw_file: Error moving to the beginning of the file"
        )
        super().init_from_sqw_file(*args)
        if not isinstance(self.dnd_dimensions, str):
            self.num_dim_ = int(np.asarray(self.dnd_dimensions).size)
        return self

    def should_load_stream(
        self, header: Dict[str, Any], fid: int, *args: Any, nargout: Optional[int] = None
    ) -> Any:
        if header.get("version") == 0 and header.get("name") == "horace":
            if header.get("uncertain", False):
                header, fid = self.get_file_header(fid, 4098 + 22)
            return super().should_load_stream(header, fid, *args, nargout=nargout)
        return pack_io_outputs((False, obj_init(), "not Horace prototype file"), nargout, default=3)

    def get_dnd_form(self, *args: Any) -> Dict[str, Any]:
        form = super().get_dnd_form(*args)
        for k in ("filename", "filepath", "title", "alatt", "angdeg"):
            form.pop(k, None)
        return form

    def get_se_npix(
        self,
        data_str: Optional[Dict[str, Any]] = None,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        res = super().get_se_npix(data_str, *args, nargout=1)
        res = _convert_signal_error(res)
        return pack_io_outputs((res, self), nargout, default=1)

    def upgrade_file_format(self, *args: Any) -> "faccess_sqw_prototype":
        raise HoraceError(
            "SQW_FILE_IO:legacy_data",
            "Format 0 is the test format so update is not deployed",
        )

