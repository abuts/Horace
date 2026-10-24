"""Legacy DnD/SQW data structure class (data_sqw_dnd).

Ported from Horace MATLAB @data_sqw_dnd class
(horace_core/sqw/@data_sqw_dnd/data_sqw_dnd.m).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.axes_block_base import AxesBlockBase
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.sqw.dnd_data import dnd_data
from pyhorace.sqw.dnd_metadata import dnd_metadata


def _mat_struct_to_dict(obj: Any) -> Any:
    """Recursively convert scipy.io.matlab.mat_struct objects into dicts."""
    if hasattr(obj, "_fieldnames"):
        return {k: _mat_struct_to_dict(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, dict):
        return {k: _mat_struct_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray) and obj.size == 0 and obj.dtype.kind in ("U", "S"):
        return ""
    return obj


@register_serializable
class data_sqw_dnd(Serializable):
    """Legacy wrapper combining dnd_metadata and dnd_data."""

    _metadata: dnd_metadata = PrivateAttr(default_factory=dnd_metadata)
    _nd_data: dnd_data = PrivateAttr(default_factory=dnd_data)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._metadata = dnd_metadata()
        sz = self._metadata.dims_as_ssize
        self._nd_data = dnd_data(
            np.zeros(sz, dtype=float),
            np.zeros(sz, dtype=float),
            np.zeros(sz, dtype=float),
        )

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if hasattr(arg0, "_fieldnames"):
                arg0 = _mat_struct_to_dict(arg0)
            if isinstance(arg0, data_sqw_dnd):
                self._metadata = copy.deepcopy(arg0._metadata)
                self._nd_data = copy.deepcopy(arg0._nd_data)
                return
            if hasattr(arg0, "metadata") and hasattr(arg0, "nd_data") and not isinstance(arg0, dict):
                self._metadata = copy.deepcopy(arg0.metadata)
                self._nd_data = copy.deepcopy(arg0.nd_data)
                return
            if isinstance(arg0, dict):
                if "array_dat" in arg0 and isinstance(arg0["array_dat"], dict):
                    arg0 = arg0["array_dat"]
                if "data_" in arg0 and isinstance(arg0["data_"], dict):
                    arg0 = arg0["data_"]
                if "metadata" in arg0 and "nd_data" in arg0:
                    self.from_bare_struct(arg0)
                elif "serial_name" in arg0 and arg0.get("serial_name") == "data_sqw_dnd":
                    loaded = self.from_struct(arg0)
                    if isinstance(loaded, data_sqw_dnd):
                        self._metadata = loaded._metadata
                        self._nd_data = loaded._nd_data
                else:
                    self._metadata = dnd_metadata(arg0)
                    if "s" in arg0 or "sig" in arg0:
                        self._nd_data = dnd_data(arg0)
                    else:
                        sz = self._metadata.dims_as_ssize
                        self._nd_data = dnd_data(
                            np.zeros(sz, dtype=float),
                            np.zeros(sz, dtype=float),
                            np.zeros(sz, dtype=float),
                        )
                    self.check_combo_arg()
                return

        if len(args) == 2 and not kwargs:
            if isinstance(args[0], dnd_metadata) and isinstance(args[1], dnd_data):
                self._metadata = copy.deepcopy(args[0])
                self._nd_data = copy.deepcopy(args[1])
                self.check_combo_arg()
                return
            if isinstance(args[0], AxesBlockBase) and isinstance(args[1], aProjectionBase):
                self._metadata = dnd_metadata(args[0], args[1])
                sz = self._metadata.dims_as_ssize
                self._nd_data = dnd_data(
                    np.zeros(sz, dtype=float),
                    np.zeros(sz, dtype=float),
                    np.zeros(sz, dtype=float),
                )
                self.check_combo_arg()
                return

        if len(args) == 5 and not kwargs and isinstance(args[0], AxesBlockBase):
            self._metadata = dnd_metadata(args[0], args[1])
            self._nd_data = dnd_data(args[2], args[3], args[4])
            self.check_combo_arg()
            return

        flds = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *args, **kwargs
        )
        if remains:
            raise HoraceError(
                "HORACE:data_sqw_dnd:invalid_argument",
                f"Unrecognized arguments to data_sqw_dnd: {remains}",
            )

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return ["metadata", "nd_data"]

    def check_combo_arg(self) -> "data_sqw_dnd":
        self._metadata.check_combo_arg()
        self._nd_data.check_combo_arg()
        exp_sz = self._metadata.dims_as_ssize
        if self._nd_data.sig.size != int(np.prod(exp_sz)):
            raise HoraceError(
                "HORACE:data_sqw_dnd:invalid_argument",
                f"data_sqw_dnd axes size {exp_sz} does not match data size {self._nd_data.sig.shape}",
            )
        if self._nd_data.sig.shape != exp_sz:
            self._nd_data._sig = self._nd_data._sig.reshape(exp_sz, order="F")
            self._nd_data._err = self._nd_data._err.reshape(exp_sz, order="F")
            self._nd_data._npix = self._nd_data._npix.reshape(exp_sz, order="F")
        return self

    @property
    def metadata(self) -> dnd_metadata:
        return self._metadata

    @metadata.setter
    def metadata(self, val: dnd_metadata) -> None:
        if not isinstance(val, dnd_metadata):
            raise HoraceError(
                "HORACE:data_sqw_dnd:invalid_argument",
                "metadata must be a dnd_metadata instance",
            )
        self._metadata = copy.deepcopy(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def nd_data(self) -> dnd_data:
        return self._nd_data

    @nd_data.setter
    def nd_data(self, val: dnd_data) -> None:
        if not isinstance(val, dnd_data):
            raise HoraceError(
                "HORACE:data_sqw_dnd:invalid_argument",
                "nd_data must be a dnd_data instance",
            )
        self._nd_data = copy.deepcopy(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    # ------------------------------------------------------------------
    # Data properties
    # ------------------------------------------------------------------
    @property
    def s(self) -> np.ndarray:
        return self._nd_data.sig

    @s.setter
    def s(self, val: Any) -> None:
        self._nd_data.sig = val

    @property
    def e(self) -> np.ndarray:
        return self._nd_data.err

    @e.setter
    def e(self, val: Any) -> None:
        self._nd_data.err = val

    @property
    def npix(self) -> np.ndarray:
        return self._nd_data.npix

    @npix.setter
    def npix(self, val: Any) -> None:
        self._nd_data.npix = val

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------
    @property
    def axes(self) -> AxesBlockBase:
        return self._metadata.axes

    @axes.setter
    def axes(self, val: AxesBlockBase) -> None:
        self._metadata.axes = val

    @property
    def proj(self) -> aProjectionBase:
        return self._metadata.proj

    @proj.setter
    def proj(self, val: aProjectionBase) -> None:
        self._metadata.proj = val

    @property
    def title(self) -> str:
        return self._metadata.title

    @title.setter
    def title(self, val: str) -> None:
        self._metadata.title = val

    @property
    def filename(self) -> str:
        return self._metadata.filename

    @filename.setter
    def filename(self, val: str) -> None:
        self._metadata.filename = val

    @property
    def filepath(self) -> str:
        return self._metadata.filepath

    @filepath.setter
    def filepath(self, val: str) -> None:
        self._metadata.filepath = val

    @property
    def full_filename(self) -> str:
        return self._metadata.full_filename

    @full_filename.setter
    def full_filename(self, val: str) -> None:
        self._metadata.full_filename = val

    @property
    def alatt(self) -> np.ndarray:
        return self._metadata.alatt

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._metadata.alatt = val

    @property
    def angdeg(self) -> np.ndarray:
        return self._metadata.angdeg

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._metadata.angdeg = val

    @property
    def offset(self) -> np.ndarray:
        return self._metadata.offset

    @offset.setter
    def offset(self, val: Any) -> None:
        self._metadata.offset = val

    @property
    def uoffset(self) -> np.ndarray:
        return self._metadata.offset

    @uoffset.setter
    def uoffset(self, val: Any) -> None:
        self._metadata.offset = val

    @property
    def label(self) -> List[str]:
        return self._metadata.label

    @label.setter
    def label(self, val: Any) -> None:
        self._metadata.label = val

    @property
    def ulabel(self) -> List[str]:
        return self._metadata.label

    @ulabel.setter
    def ulabel(self, val: Any) -> None:
        self._metadata.label = val

    @property
    def ulen(self) -> np.ndarray:
        return self._metadata.ulen

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self._metadata.ulen = val

    @property
    def iax(self) -> np.ndarray:
        return self._metadata.iax

    @property
    def iint(self) -> np.ndarray:
        return self._metadata.iint

    @property
    def pax(self) -> np.ndarray:
        return self._metadata.pax

    @property
    def p(self) -> List[np.ndarray]:
        return self._metadata.p

    @property
    def dax(self) -> np.ndarray:
        return self._metadata.dax

    @dax.setter
    def dax(self, val: Any) -> None:
        self._metadata.dax = val

    @property
    def img_range(self) -> np.ndarray:
        return self._metadata.img_range

    @img_range.setter
    def img_range(self, val: Any) -> None:
        self._metadata.img_range = val

    @property
    def img_db_range(self) -> np.ndarray:
        return self._metadata.img_range

    @img_db_range.setter
    def img_db_range(self, val: Any) -> None:
        self._metadata.img_range = val

    @property
    def u_to_rlu(self) -> np.ndarray:
        if hasattr(self._metadata.proj, "u_to_rlu"):
            return self._metadata.proj.u_to_rlu
        return np.eye(4, dtype=float)

    @property
    def u_to_rlu_legacy(self) -> np.ndarray:
        if hasattr(self._metadata.proj, "u_to_rlu_legacy"):
            return self._metadata.proj.u_to_rlu_legacy
        return np.eye(4, dtype=float)

    @property
    def creation_date_defined(self) -> bool:
        return self._metadata.creation_date_defined

    @property
    def creation_date_str(self) -> str:
        return self._metadata.creation_date_str

    @creation_date_str.setter
    def creation_date_str(self, val: str) -> None:
        self._metadata.creation_date_str = val

    def dimensions(self) -> Tuple[int, Tuple[int, ...]]:
        nd = self._metadata.dimensions
        if nd == 0:
            return 0, ()
        sz = self._metadata.dims_as_ssize
        return nd, sz[:nd]

    def get_dnd_data(self, *args: Any) -> Any:
        if len(args) > 0:
            return self._nd_data, self._metadata
        return self._nd_data

    def clear_sqw_data(self) -> "data_sqw_dnd":
        return self

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "data_sqw_dnd"]:
        if "metadata" not in S:
            meta = dnd_metadata(S)
            dat = dnd_data(S)
            return {"metadata": meta, "nd_data": dat}, self
        return S, self

