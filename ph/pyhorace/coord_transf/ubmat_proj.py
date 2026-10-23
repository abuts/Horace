"""Rectilinear projection class defined by u_to_rlu matrix and ulen (ubmat_proj)."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError, line_axes
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.line_proj_base import LineProjBase


class ubmat_proj(LineProjBase):
    """Rectilinear projection defined directly by the ``u_to_rlu`` matrix and ``ulen`` scales."""

    positional_arg_names_: ClassVar[list[str]] = [
        "u_to_rlu",
        "ulen",
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

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._u_to_rlu_mat: np.ndarray = np.eye(4, dtype=float)
        self._img_scales = np.ones(4, dtype=float)
        self._label = ["\\zeta", "\\xi", "\\eta", "E"]
        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return [
            "u_to_rlu",
            "ulen",
            "alatt",
            "angdeg",
            "offset",
            "label",
            "title",
        ]

    @property
    def u_to_rlu(self) -> np.ndarray:
        return self._u_to_rlu_mat.copy()

    @u_to_rlu.setter
    def u_to_rlu(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.shape == (3, 3):
            m4 = np.eye(4, dtype=float)
            m4[:3, :3] = arr
            arr = m4
        elif arr.shape != (4, 4):
            raise HoraceError(
                "HORACE:ubmat_proj:invalid_argument",
                "u_to_rlu must be a 3x3 or 4x4 invertible numeric matrix",
            )
        if abs(np.linalg.det(arr[:3, :3])) < self.tol_:
            raise HoraceError(
                "HORACE:ubmat_proj:invalid_argument",
                "u_to_rlu matrix is singular (determinant is zero)",
            )
        self._u_to_rlu_mat = arr.copy()

    @property
    def ulen(self) -> np.ndarray:
        return self._img_scales.copy()

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self._set_img_scales(val)

    def get_pix_img_transformation(
        self,
        ndim: int = 3,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Compute pixel-to-image transformation matrix, offset in CC, and img_scales."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        b = self.bmatrix(3, nargout=1)
        u_rlu_3 = self._u_to_rlu_mat[:3, :3]
        # q_cc = B @ u_rlu_3 @ q_img => q_img = inv(B @ u_rlu_3) @ q_cc
        u_to_cc = b @ u_rlu_3
        pix_to_img_3 = np.linalg.inv(u_to_cc)
        offset_cc_3 = b @ self._offset[:3]

        if ndim == 3:
            mat = pix_to_img_3
            offset_cc = offset_cc_3
            ulen = self._img_scales[:3].copy()
        elif ndim == 4:
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = pix_to_img_3
            offset_cc = np.array([offset_cc_3[0], offset_cc_3[1], offset_cc_3[2], self._offset[3]], dtype=float)
            ulen = self._img_scales.copy()
        else:
            raise HoraceError(
                "HORACE:ubmat_proj:invalid_argument",
                f"ndim must be 3 or 4, got {ndim}",
            )

        if nargout == 1:
            return mat
        if nargout == 2:
            return mat, offset_cc
        if nargout == 3:
            return mat, offset_cc, ulen
        offset_present = bool(np.any(np.abs(offset_cc) > self.tol_))
        return mat, offset_cc, ulen, offset_present

    def get_line_proj(self) -> Any:
        """Convert this ubmat_proj into an equivalent line_proj instance."""
        from pyhorace.coord_transf.line_proj import line_proj

        if not self.lattice_defined:
            raise HoraceError(
                "HORACE:ubmat_proj:runtime_error",
                "Cannot convert to line_proj when lattice (alatt, angdeg) is not defined",
            )
        u, v, w, ptype, nonortho = self.uv_from_u_to_rlu_legacy(
            self._u_to_rlu_mat,
            self._img_scales,
            self._alatt,
            self._angdeg,
        )
        lp = line_proj(
            u,
            v,
            w,
            nonorthogonal=nonortho,
            type=ptype,
            alatt=self.alatt,
            angdeg=self.angdeg,
            offset=self.offset,
            label=self.label,
            title=self.title,
        )
        lp.u_to_rlu_legacy = self._u_to_rlu_mat
        lp.ulen = self._img_scales
        return lp

    def get_ubmat_proj(self) -> "ubmat_proj":
        """Return a copy of this ubmat_proj."""
        return copy.deepcopy(self)

    def align_proj(self, alignment_info: Any) -> "ubmat_proj":
        """Align projection using crystal_alignment_info."""
        corr_mat = alignment_info.get_corr_mat(self)
        u_rlu_3 = corr_mat @ self._u_to_rlu_mat[:3, :3]
        self._u_to_rlu_mat[:3, :3] = u_rlu_3
        self._offset[:3] = corr_mat @ self._offset[:3]
        self.alatt = alignment_info.alatt
        self.angdeg = alignment_info.angdeg
        return self
