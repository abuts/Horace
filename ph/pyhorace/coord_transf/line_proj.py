"""Rectilinear projection class defined by crystal axes u, v, w (line_proj)."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError, line_axes
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.line_proj_base import LineProjBase


class line_proj(LineProjBase):
    """Projection class defining rectilinear (orthogonal or non-orthogonal) axes in reciprocal space.

    Defines coordinate transformations along crystal directions ``u``, ``v``, and
    optionally ``w``, with scaling ``type`` ('p', 'r', or 'a' for each Q-axis).
    """

    positional_arg_names_: ClassVar[list[str]] = [
        "u",
        "v",
        "w",
        "nonorthogonal",
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

    def class_version(self) -> int:
        return 3

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._u: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=float)
        self._v: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        self._w: Optional[np.ndarray] = None
        self._nonorthogonal: bool = False
        self._type_explicitly_set: bool = False
        self._type = ["p", "p", "r"]
        self._label = ["\\zeta", "\\xi", "\\eta", "E"]
        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return [
            "u",
            "v",
            "w",
            "nonorthogonal",
            "type",
            "alatt",
            "angdeg",
            "offset",
            "label",
            "title",
            "u_to_rlu_legacy",
            "ulen",
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def u(self) -> np.ndarray:
        return self._u.copy()

    @u.setter
    def u(self, val: Any) -> None:
        arr = self._check_vector_arg(val, "u", allow_empty=False)
        self._u = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @v.setter
    def v(self, val: Any) -> None:
        arr = self._check_vector_arg(val, "v", allow_empty=False)
        self._v = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def w(self) -> np.ndarray:
        if self._w is None:
            return np.array([], dtype=float)
        return self._w.copy()

    @w.setter
    def w(self, val: Any) -> None:
        arr = self._check_vector_arg(val, "w", allow_empty=True)
        if arr is None:
            self._w = None
            if not self._type_explicitly_set:
                self._type[2] = "r"
        else:
            self._w = arr
            if not self._type_explicitly_set and self._type == ["p", "p", "r"]:
                self._type[2] = "p"
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @staticmethod
    def _check_vector_arg(val: Any, name: str, allow_empty: bool = False) -> Optional[np.ndarray]:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            if allow_empty:
                return None
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                f"Direction vector {name} cannot be empty",
            )
        if isinstance(val, (str, bytes)) or isinstance(val, dict):
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                f"Direction vector {name} must be a 3-element non-zero numeric vector",
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.any(np.isnan(arr)) or np.linalg.norm(arr) < aProjectionBase_tol:
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                f"Direction vector {name} must be a 3-element non-zero numeric vector",
            )
        return arr.astype(float)

    @property
    def nonorthogonal(self) -> bool:
        return self._nonorthogonal

    @nonorthogonal.setter
    def nonorthogonal(self, val: Any) -> None:
        if not isinstance(val, (bool, np.bool_, int, np.integer, float, np.floating)):
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                "nonorthogonal must be a boolean or scalar 0/1",
            )
        self._nonorthogonal = bool(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_type(self, val: Any) -> None:
        if not isinstance(val, str) or len(val) != 3:
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                "type must be a 3-character string made of 'p', 'r', or 'a'",
            )
        chars = [c.lower() for c in val]
        if any(c not in ("p", "r", "a") for c in chars):
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                "type characters for line_proj must be 'p', 'r', or 'a'",
            )
        self._type = chars
        self._type_explicitly_set = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def img_scales(self) -> np.ndarray:
        if self.lattice_defined:
            _, _, ulen_3 = self.u_to_rlu_legacy_from_uvw(
                self.bmatrix(3, nargout=1),
                self._alatt,
                self._angdeg,
                self._u,
                self._v,
                self._w,
                self.type,
                self._nonorthogonal,
                nargout=3,
            )
            return np.array([ulen_3[0], ulen_3[1], ulen_3[2], 1.0], dtype=float)
        if self._ulen is not None:
            return self._ulen.copy()
        return self._img_scales.copy()

    @img_scales.setter
    def img_scales(self, val: Any) -> None:
        self._set_img_scales(val)

    def check_combo_arg(self) -> "line_proj":
        """Validate interdependent projection properties (u, v, w non-collinear/non-coplanar)."""
        if np.linalg.norm(np.cross(self._u, self._v)) < self.tol_:
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                "Vectors u and v cannot be collinear",
            )
        if self._w is not None:
            if np.linalg.norm(np.cross(self._u, self._w)) < self.tol_:
                raise HoraceError(
                    "HORACE:line_proj:invalid_argument",
                    "Vectors u and w cannot be collinear",
                )
            mat = np.column_stack([self._u, self._v, self._w])
            if abs(np.linalg.det(mat)) < self.tol_:
                raise HoraceError(
                    "HORACE:line_proj:invalid_argument",
                    "Vectors u, v, and w cannot be coplanar",
                )
        if self.lattice_defined:
            _, _, ulen_3 = self.u_to_rlu_legacy_from_uvw(
                self.bmatrix(3, nargout=1),
                self._alatt,
                self._angdeg,
                self._u,
                self._v,
                self._w,
                self.type,
                self._nonorthogonal,
                nargout=3,
            )
            self._img_scales = np.array([ulen_3[0], ulen_3[1], ulen_3[2], 1.0], dtype=float)
        if self._targ_proj is not None:
            self._check_and_set_targ_proj_optimization(self._targ_proj)
        return self

    def set_directions(
        self,
        u: Any,
        v: Any,
        w: Any = None,
        offset: Any = None,
    ) -> "line_proj":
        """Convenience method to set u, v, w, and optionally offset at once."""
        old_check = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            self.u = u
            self.v = v
            if w is not None:
                self.w = w
            if offset is not None:
                self.offset = offset
        finally:
            self._do_check_combo_arg = old_check
        self.check_combo_arg()
        return self

    # ------------------------------------------------------------------
    # Transformation Matrices & Conversions
    # ------------------------------------------------------------------
    def get_pix_img_transformation(
        self,
        ndim: int = 3,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Compute pixel-to-image transformation matrix, offset in CC, and u_to_rlu."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        b = self.bmatrix(3, nargout=1)
        pix_to_img_3, u_to_rlu_3, ulen_3 = self.u_to_rlu_legacy_from_uvw(
            b,
            self._alatt,
            self._angdeg,
            self._u,
            self._v,
            self._w,
            self.type,
            self._nonorthogonal,
            nargout=3,
        )
        offset_cc_3 = b @ self._offset[:3]

        if ndim == 3:
            mat = pix_to_img_3
            offset_cc = offset_cc_3
            u_to_rlu = u_to_rlu_3
            ulen = ulen_3
        elif ndim == 4:
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = pix_to_img_3
            offset_cc = np.array([offset_cc_3[0], offset_cc_3[1], offset_cc_3[2], self._offset[3]], dtype=float)
            u_to_rlu = np.eye(4, dtype=float)
            u_to_rlu[:3, :3] = u_to_rlu_3
            ulen = np.array([ulen_3[0], ulen_3[1], ulen_3[2], 1.0], dtype=float)
        else:
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                f"ndim must be 3 or 4, got {ndim}",
            )

        if nargout == 1:
            return mat
        if nargout == 2:
            # Note: 2 outputs in Horace get_pix_img_transformation can be (pix_to_img, u_to_rlu) or (pix_to_img, offset_cc)
            # Wait: let's check what MATLAB line_proj.get_pix_img_transformation returns!
            # In MATLAB LineProjBase:
            # [pix_to_img, offset_cc, img_scales, offset_present] = obj.get_pix_img_transformation(ndim, ...)
            # Wait: let's check both u_to_rlu property and get_pix_img_transformation callers!
            return mat, offset_cc
        if nargout == 3:
            return mat, offset_cc, ulen
        offset_present = bool(np.any(np.abs(offset_cc) > self.tol_))
        return mat, offset_cc, ulen, offset_present

    @property
    def u_to_rlu(self) -> np.ndarray:
        """4x4 matrix converting image coordinates into rlu + energy."""
        if not self.lattice_defined:
            return np.array([], dtype=float)
        b = self.bmatrix(3, nargout=1)
        _, u_to_rlu_3, _ = self.u_to_rlu_legacy_from_uvw(
            b,
            self._alatt,
            self._angdeg,
            self._u,
            self._v,
            self._w,
            self.type,
            self._nonorthogonal,
            nargout=3,
        )
        u4 = np.eye(4, dtype=float)
        u4[:3, :3] = u_to_rlu_3
        return u4

    def get_ubmat_proj(self) -> Any:
        """Convert this line_proj into an equivalent ubmat_proj instance."""
        from pyhorace.coord_transf.ubmat_proj import ubmat_proj

        if not self.lattice_defined:
            raise HoraceError(
                "HORACE:line_proj:runtime_error",
                "Cannot convert to ubmat_proj when lattice (alatt, angdeg) is not defined",
            )
        return ubmat_proj(
            self.u_to_rlu,
            self.img_scales,
            alatt=self.alatt,
            angdeg=self.angdeg,
            offset=self.offset,
            label=self.label,
            title=self.title,
        )

    def get_line_proj(self) -> "line_proj":
        """Return a copy of this line_proj."""
        return copy.deepcopy(self)

    def align_proj(self, alignment_info: Any) -> "line_proj":
        """Align projection axes and lattice using crystal_alignment_info."""
        ub_pr = self.get_ubmat_proj()
        ub_pr.align_proj(alignment_info)
        aligned_lp = ub_pr.get_line_proj()
        self._alatt = aligned_lp._alatt.copy()
        self._angdeg = aligned_lp._angdeg.copy()
        self._alatt_defined = aligned_lp._alatt_defined
        self._angdeg_defined = aligned_lp._angdeg_defined
        self._u = aligned_lp._u.copy()
        self._v = aligned_lp._v.copy()
        self._w = None if aligned_lp._w is None else aligned_lp._w.copy()
        self._offset = aligned_lp._offset.copy()
        self.check_combo_arg()
        return self


aProjectionBase_tol = 1e-12
