"""Horace 3 legacy DnD property interface (horace3_dnd_interface).

Ported from Horace MATLAB horace3_dnd_interface class
(horace_core/sqw/horace3_dnd_interface.m).
"""

from __future__ import annotations

from abc import ABC
from typing import Any, List

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError


class horace3_dnd_interface(ABC):
    """Mixin providing Horace 3 DnD property accessors on DnDBase / SQWDnDBase."""

    # ------------------------------------------------------------------
    # Abstract hooks implemented by DnDBase
    # ------------------------------------------------------------------
    def _get_axes(self) -> Any:
        return self.axes

    def _get_proj(self) -> Any:
        return self.proj

    def _get_senpix(self, fld: str) -> np.ndarray:
        raise NotImplementedError

    def _set_senpix(self, val: Any, fld: str) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def filename(self) -> str:
        return self._get_axes().filename

    @filename.setter
    def filename(self, val: str) -> None:
        self._get_axes().filename = val

    @property
    def filepath(self) -> str:
        return self._get_axes().filepath

    @filepath.setter
    def filepath(self, val: str) -> None:
        self._get_axes().filepath = val

    @property
    def full_filename(self) -> str:
        return self._get_axes().full_filename

    @full_filename.setter
    def full_filename(self, val: str) -> None:
        self._get_axes().full_filename = val

    @property
    def title(self) -> str:
        return self._get_axes().title

    @title.setter
    def title(self, val: str) -> None:
        self._get_axes().title = val

    @property
    def alatt(self) -> np.ndarray:
        return self._get_proj().alatt

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._get_proj().alatt = val

    @property
    def angdeg(self) -> np.ndarray:
        return self._get_proj().angdeg

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._get_proj().angdeg = val

    @property
    def offset(self) -> np.ndarray:
        return self._get_proj().offset

    @offset.setter
    def offset(self, val: Any) -> None:
        self._get_proj().offset = val
        self._get_axes().offset = val

    @property
    def uoffset(self) -> np.ndarray:
        return self._get_proj().offset

    @uoffset.setter
    def uoffset(self, val: Any) -> None:
        self.offset = val

    @property
    def label(self) -> List[str]:
        return self._get_axes().label

    @label.setter
    def label(self, val: Any) -> None:
        self._get_axes().label = val

    @property
    def ulabel(self) -> List[str]:
        return self._get_axes().label

    @ulabel.setter
    def ulabel(self, val: Any) -> None:
        self._get_axes().label = val

    @property
    def ulen(self) -> np.ndarray:
        return self._get_axes().img_scales

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self._get_axes().img_scales = val

    @property
    def iax(self) -> np.ndarray:
        return self._get_axes().iax

    @iax.setter
    def iax(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "Cannot directly set the integration axes; use another constructor or cut.",
        )

    @property
    def iint(self) -> np.ndarray:
        return self._get_axes().iint

    @iint.setter
    def iint(self, val: Any) -> None:
        self._get_axes().iint = val

    @property
    def pax(self) -> np.ndarray:
        return self._get_axes().pax

    @pax.setter
    def pax(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "Cannot directly set the projection axes; use another constructor or cut.",
        )

    @property
    def p(self) -> List[np.ndarray]:
        return self._get_axes().p

    @p.setter
    def p(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "Cannot directly set the bin boundary values; use another constructor or cut.",
        )

    @property
    def dax(self) -> np.ndarray:
        return self._get_axes().dax

    @dax.setter
    def dax(self, val: Any) -> None:
        self._get_axes().dax = val

    @property
    def s(self) -> np.ndarray:
        return self._get_senpix("s")

    @s.setter
    def s(self, val: Any) -> None:
        self._set_senpix(val, "s")

    @property
    def e(self) -> np.ndarray:
        return self._get_senpix("e")

    @e.setter
    def e(self, val: Any) -> None:
        self._set_senpix(val, "e")

    @property
    def npix(self) -> np.ndarray:
        return self._get_senpix("npix")

    @npix.setter
    def npix(self, val: Any) -> None:
        self._set_senpix(val, "npix")

    @property
    def img_range(self) -> np.ndarray:
        return self._get_axes().img_range

    @img_range.setter
    def img_range(self, val: Any) -> None:
        self._get_axes().img_range = val

    @property
    def img_db_range(self) -> np.ndarray:
        return self._get_axes().img_range

    @img_db_range.setter
    def img_db_range(self, val: Any) -> None:
        self._get_axes().img_range = val

    @property
    def u_to_rlu(self) -> np.ndarray:
        proj = self._get_proj()
        if hasattr(proj, "u_to_rlu"):
            return proj.u_to_rlu
        return np.eye(4, dtype=float)

    @u_to_rlu.setter
    def u_to_rlu(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "Property u_to_rlu is deprecated and can not be set",
        )

    @property
    def u_to_rlu_legacy(self) -> np.ndarray:
        proj = self._get_proj()
        if hasattr(proj, "u_to_rlu_legacy"):
            return proj.u_to_rlu_legacy
        return np.eye(4, dtype=float)

    @u_to_rlu_legacy.setter
    def u_to_rlu_legacy(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "Property u_to_rlu_legacy is deprecated and can not be set",
        )

