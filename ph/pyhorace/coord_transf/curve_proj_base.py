"""Abstract base class for curvilinear coordinate projections (CurveProjBase)."""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.ubmatrix import ubmatrix


class CurveProjBase(aProjectionBase):
    """Abstract base class for curvilinear projections (spherical and cylindrical).

    Defines the polar/zenith axis ``u`` (alias ``ez``) and azimuthal reference axis
    ``v`` (alias ``ex``) in reciprocal lattice units (rlu), and provides common
    rotation and scaling matrix computations between Crystal Cartesian and
    curvilinear reference frames.
    """

    hor2matlab_transf_: ClassVar[np.ndarray] = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    positional_arg_names_: ClassVar[list[str]] = [
        "u",
        "v",
        "type",
        "alatt",
        "angdeg",
        "offset",
        "label",
        "title",
        "lab1",
        "lab2",
        "lab3",
        "lab4",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._u: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=float)
        self._v: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return [
            "u",
            "v",
            "type",
            "alatt",
            "angdeg",
            "offset",
            "label",
            "title",
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def u(self) -> np.ndarray:
        return self._u.copy()

    @u.setter
    def u(self, val: Any) -> None:
        arr = self._check_uv_vec(val, "u")
        self._u = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @v.setter
    def v(self, val: Any) -> None:
        arr = self._check_uv_vec(val, "v")
        self._v = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def ez(self) -> np.ndarray:
        """Alias for u (zenith/cylinder axis)."""
        return self.u

    @ez.setter
    def ez(self, val: Any) -> None:
        self.u = val

    @property
    def ex(self) -> np.ndarray:
        """Alias for v (azimuthal reference axis)."""
        return self.v

    @ex.setter
    def ex(self, val: Any) -> None:
        self.v = val

    @staticmethod
    def _check_uv_vec(val: Any, name: str) -> np.ndarray:
        if val is None or isinstance(val, (str, bytes, dict)):
            raise HoraceError(
                "HORACE:CurveProjBase:invalid_argument",
                f"Direction vector {name} must be a 3-element non-zero numeric vector",
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.any(np.isnan(arr)) or np.linalg.norm(arr) < 1e-12:
            raise HoraceError(
                "HORACE:CurveProjBase:invalid_argument",
                f"Direction vector {name} must be a 3-element non-zero numeric vector",
            )
        return arr.astype(float)

    def check_combo_arg(self) -> "CurveProjBase":
        """Ensure u and v are not collinear and update img_scales if lattice is defined."""
        if np.linalg.norm(np.cross(self._u, self._v)) < self.tol_:
            raise HoraceError(
                "HORACE:CurveProjBase:invalid_argument",
                "Direction vectors u and v cannot be collinear",
            )
        if self.lattice_defined:
            _, _, scales = self.get_pix_img_transformation(3, nargout=3)
            self._img_scales = np.array([scales[0], scales[1], scales[2], 1.0], dtype=float)
        return self

    def set_directions(
        self,
        u: Any,
        v: Any,
        offset: Any = None,
    ) -> "CurveProjBase":
        """Set u, v, and optionally offset at once."""
        old_check = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            self.u = u
            self.v = v
            if offset is not None:
                self.offset = offset
        finally:
            self._do_check_combo_arg = old_check
        self.check_combo_arg()
        return self

    def align_proj(self, alignment_info: Any) -> "CurveProjBase":
        """Align curvilinear projection using crystal_alignment_info."""
        corr_mat = alignment_info.get_corr_mat(self._alatt, self._angdeg)
        self._u = corr_mat @ self._u
        self._v = corr_mat @ self._v
        self._offset[:3] = corr_mat @ self._offset[:3]
        self.alatt = alignment_info.alatt
        self.angdeg = alignment_info.angdeg
        self.check_combo_arg()
        return self
