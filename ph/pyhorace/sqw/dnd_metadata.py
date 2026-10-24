"""DnD metadata container class (dnd_metadata).

Ported from Horace MATLAB @dnd_metadata class
(horace_core/sqw/file_io/class_helpers/@dnd_metadata/dnd_metadata.m).
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.axes_block_base import AxesBlockBase
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.serializable import Serializable, register_serializable


def _parse_iso_datetime(date_str: str) -> str:
    """Validate and normalize ISO-like datetime string."""
    s = str(date_str).strip()
    if not s:
        return ""
    # Accept standard ISO 8601 strings or MATLAB datestr formats
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s[:26], fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError as exc:
        raise HoraceError(
            "HORACE:dnd_metadata:invalid_argument",
            f"Invalid creation_date_str format: {date_str!r}",
        ) from exc


@register_serializable
class dnd_metadata(Serializable):
    """Class holding metadata (axes, projection, creation date) of a DnD object."""

    _axes: AxesBlockBase = PrivateAttr(default_factory=line_axes)
    _proj: aProjectionBase = PrivateAttr(default_factory=line_proj)
    _creation_date: str = PrivateAttr(default="")
    _creation_date_defined: bool = PrivateAttr(default=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._axes = line_axes()
        self._proj = line_proj()
        self._creation_date = ""
        self._creation_date_defined = False

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, dnd_metadata):
                self._axes = copy.deepcopy(arg0._axes)
                self._proj = copy.deepcopy(arg0._proj)
                self._creation_date = arg0._creation_date
                self._creation_date_defined = arg0._creation_date_defined
                return
            if type(arg0).__name__ == "data_sqw_dnd" or (
                hasattr(arg0, "axes")
                and hasattr(arg0, "proj")
                and not isinstance(arg0, dict)
            ):
                self._axes = copy.deepcopy(arg0.axes)
                self._proj = copy.deepcopy(arg0.proj)
                if hasattr(arg0, "creation_date_defined") and arg0.creation_date_defined:
                    cd = getattr(arg0, "creation_date", None) or getattr(arg0, "creation_date_str", "")
                    self.creation_date_str = cd
                else:
                    self.creation_date_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                self.check_combo_arg()
                return
            if isinstance(arg0, dict):
                if "axes" not in arg0 and ("npax" in arg0 or "u_to_rlu" in arg0 or "u_to_rlu_legacy" in arg0):
                    self._proj = line_proj.get_from_old_data(arg0)
                    self._axes = line_axes.get_from_old_data(arg0)
                    self.check_combo_arg()
                elif "serial_name" in arg0:
                    loaded = self.from_struct(arg0)
                    if isinstance(loaded, dnd_metadata):
                        self._axes = loaded._axes
                        self._proj = loaded._proj
                        self._creation_date = loaded._creation_date
                        self._creation_date_defined = loaded._creation_date_defined
                else:
                    self.from_bare_struct(arg0)
                return

        flds = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *args, **kwargs
        )
        if remains:
            raise HoraceError(
                "HORACE:dnd_metadata:invalid_argument",
                f"Additional unrecognized parameters provided as input to dnd_metadata constructor: {remains}",
            )

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return ["axes", "proj", "creation_date_str"]

    def check_combo_arg(self) -> "dnd_metadata":
        self._axes = self._proj.copy_proj_defined_properties_to_axes(self._axes)
        return self

    # ------------------------------------------------------------------
    # Core properties: axes, proj, creation_date_str
    # ------------------------------------------------------------------
    @property
    def axes(self) -> AxesBlockBase:
        return self._axes

    @axes.setter
    def axes(self, val: Any) -> None:
        if not isinstance(val, AxesBlockBase):
            raise HoraceError(
                "HORACE:dnd_metadata:invalid_argument",
                f"Input moves for axes parameter have to be a subclass of AxesBlockBase class. It is {type(val).__name__}",
            )
        self._axes = copy.deepcopy(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def proj(self) -> aProjectionBase:
        return self._proj

    @proj.setter
    def proj(self, val: Any) -> None:
        if not isinstance(val, aProjectionBase):
            raise HoraceError(
                "HORACE:dnd_metadata:invalid_argument",
                f"Projection have to be instance of aProjectionBase class while in fact it is: {type(val).__name__}",
            )
        self._proj = copy.deepcopy(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def creation_date_defined(self) -> bool:
        return self._creation_date_defined

    @property
    def creation_date_str(self) -> str:
        if self._creation_date_defined:
            return self._creation_date
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    @creation_date_str.setter
    def creation_date_str(self, val: Any) -> None:
        if val is None or (isinstance(val, str) and not val.strip()) or (
            isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0
        ):
            self._creation_date = ""
            self._creation_date_defined = False
            return
        if not isinstance(val, str):
            raise HoraceError(
                "HORACE:dnd_metadata:invalid_argument",
                "creation_date_str must be a string",
            )
        parsed = _parse_iso_datetime(val)
        if parsed:
            self._creation_date = parsed
            self._creation_date_defined = True
        else:
            self._creation_date = ""
            self._creation_date_defined = False

    # ------------------------------------------------------------------
    # Convenience properties delegated to axes and proj
    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return self._axes.title

    @title.setter
    def title(self, val: str) -> None:
        self._axes.title = val

    @property
    def filename(self) -> str:
        return self._axes.filename

    @filename.setter
    def filename(self, val: str) -> None:
        self._axes.filename = val

    @property
    def filepath(self) -> str:
        return self._axes.filepath

    @filepath.setter
    def filepath(self, val: str) -> None:
        self._axes.filepath = val

    @property
    def full_filename(self) -> str:
        return self._axes.full_filename

    @full_filename.setter
    def full_filename(self, val: str) -> None:
        self._axes.full_filename = val

    @property
    def npax(self) -> int:
        return len(self._axes.pax)

    @property
    def iax(self) -> np.ndarray:
        return self._axes.iax

    @property
    def iint(self) -> np.ndarray:
        return self._axes.iint

    @property
    def pax(self) -> np.ndarray:
        return self._axes.pax

    @property
    def p(self) -> List[np.ndarray]:
        return self._axes.p

    @property
    def dax(self) -> np.ndarray:
        return self._axes.dax

    @dax.setter
    def dax(self, val: Any) -> None:
        self._axes.dax = val

    @property
    def label(self) -> List[str]:
        return self._axes.label

    @label.setter
    def label(self, val: Any) -> None:
        self._axes.label = val

    @property
    def ulen(self) -> np.ndarray:
        return self._axes.img_scales

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self._axes.img_scales = val

    @property
    def img_range(self) -> np.ndarray:
        return self._axes.img_range

    @img_range.setter
    def img_range(self, val: Any) -> None:
        self._axes.img_range = val

    @property
    def nbins_all_dims(self) -> np.ndarray:
        return self._axes.nbins_all_dims

    @nbins_all_dims.setter
    def nbins_all_dims(self, val: Any) -> None:
        self._axes.nbins_all_dims = val

    @property
    def dimensions(self) -> int:
        return self._axes.dimensions

    @property
    def dims_as_ssize(self) -> Tuple[int, ...]:
        return tuple(int(x) for x in self._axes.dims_as_ssize)

    @property
    def offset(self) -> np.ndarray:
        return self._proj.offset

    @offset.setter
    def offset(self, val: Any) -> None:
        self._proj.offset = val
        self._axes.offset = val

    @property
    def alatt(self) -> np.ndarray:
        return self._proj.alatt

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._proj.alatt = val

    @property
    def angdeg(self) -> np.ndarray:
        return self._proj.angdeg

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._proj.angdeg = val

    @property
    def latt_defined(self) -> bool:
        return self._proj.alatt_defined and self._proj.angdeg_defined

    def to_bare_struct(
        self, recursive_bare: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        res = super().to_bare_struct(recursive_bare=recursive_bare)
        if isinstance(res, dict) and not self._creation_date_defined:
            res["creation_date_str"] = ""
        return res

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "dnd_metadata"]:
        if "axes" not in S and ("npax" in S or "u_to_rlu" in S or "u_to_rlu_legacy" in S):
            proj = line_proj.get_from_old_data(S)
            axes = line_axes.get_from_old_data(S)
            S_new = {
                "axes": axes,
                "proj": proj,
                "creation_date_str": S.get("creation_date_str", ""),
            }
            return S_new, self
        return S, self
