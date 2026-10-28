"""Interface class for symmetry operators defined by two vectors (u, v) or a normal vector (SymopSetPlaneInterface).

Translated from MATLAB ``horace_core/symop/@SymopSetPlaneInterface/SymopSetPlaneInterface.m``,
``@SymopSetPlaneInterface/private/check_and_caclulate_vectors_and_R_.m``, and
``@SymopSetPlaneInterface/private/get_uv_from_normvec_.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from pyhorace.coord_transf.symop.symop import Symop
from pyhorace.core.exceptions import HoraceError


def _is_diagonal_matr(mat: np.ndarray, tol: float = 1e-12) -> bool:
    """Return True if ``mat`` is diagonal within ``tol``."""
    off_diag = mat - np.diag(np.diag(mat))
    return bool(np.all(np.abs(off_diag) <= tol))


def _get_uv_from_normvec(
    normvec: np.ndarray,
    normvec_in_rlu: Optional[bool],
    bmat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[bool]]:
    """Calculate two orthogonal in-plane vectors (u, v) in rlu from ``normvec``.

    Ported from ``@SymopSetPlaneInterface/private/get_uv_from_normvec_.m``.
    """
    nrm = np.asarray(normvec, dtype=float).ravel().copy()
    if normvec_in_rlu is None:
        empty_normvec_in_rlu = True
        if _is_diagonal_matr(bmat):
            normvec_in_rlu = False
        else:
            raise HoraceError(
                "HORACE:SymopSetPlaneInterface:invalid_argument",
                "With normal vector specified on non-orthogonal lattice you have to explicitly mention what coordinate system the vector has been provided in",
            )
    else:
        empty_normvec_in_rlu = False

    if normvec_in_rlu:
        nrm = bmat @ nrm

    nrm = nrm / float(np.linalg.norm(nrm))

    u_suggestions = (
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
    )
    max_length = 0.0
    u_selected = u_suggestions[0]
    for s in u_suggestions:
        u_cand = np.linalg.solve(bmat, s)
        u_cand = u_cand / float(np.linalg.norm(u_cand))
        ortho_vec_length = float(np.linalg.norm(np.cross(u_cand, nrm)))
        if ortho_vec_length == 1.0:
            u_selected = u_cand
            break
        elif ortho_vec_length > max_length:
            max_length = ortho_vec_length
            u_selected = u_cand

    u_cc = bmat @ u_selected
    u_cc = u_cc - nrm * float(np.dot(nrm, u_cc))
    _, v = Symop.check_and_brush_3vector(np.linalg.solve(bmat, np.cross(nrm, u_cc)))
    _, u = Symop.check_and_brush_3vector(np.linalg.solve(bmat, u_cc))
    if empty_normvec_in_rlu:
        normvec_in_rlu = None
    return u, v, nrm, normvec_in_rlu


class SymopSetPlaneInterface(Symop):
    """Abstract base class for symmetry operators defined by a plane (u, v) or normal vector (normvec)."""

    def __init__(self) -> None:
        super().__init__()
        self._u: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=float)
        self._v: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        self._normvec: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float)
        self._input_nrmv_in_rlu: Optional[bool] = None
        self._set_from_normvec: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def u(self) -> np.ndarray:
        """First vector in reciprocal lattice (rlu) defining the symmetry plane."""
        return self._u.copy()

    @u.setter
    def u(self, val: Any) -> None:
        is_valid, brushed = self.check_and_brush_3vector(val)
        if not is_valid:
            raise HoraceError(
                "HORACE:SymopSetPlaneIntrerface:invalid_argument",
                f"u value has to be 3-elements numeric vector. Provided: {val!r}",
            )
        self._set_from_normvec = False
        self._u = brushed
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def v(self) -> np.ndarray:
        """Second vector in reciprocal lattice (rlu) defining the symmetry plane."""
        return self._v.copy()

    @v.setter
    def v(self, val: Any) -> None:
        is_valid, brushed = self.check_and_brush_3vector(val)
        if not is_valid:
            raise HoraceError(
                "HORACE:SymopSetPlaneIntrerface:invalid_argument",
                f"v value has to be 3-elements numeric vector. Provided: {val!r}",
            )
        self._set_from_normvec = False
        self._v = brushed
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def normvec(self) -> np.ndarray:
        """Unit normal vector to the plane (u, v) in Crystal Cartesian coordinates."""
        return self._normvec.copy()

    @normvec.setter
    def normvec(self, val: Any) -> None:
        is_valid, brushed = self.check_and_brush_3vector(val)
        if not is_valid or np.all(brushed == 0.0):
            raise HoraceError(
                "HORACE:SymopSetPlaneIntrerface:invalid_argument",
                f"normvec value has to be non-zero 3-element numeric vector. Provided: {val!r}",
            )
        self._set_from_normvec = True
        self._normvec = brushed
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def input_nrmv_in_rlu(self) -> bool:
        """True if input normal vector was provided in rlu, False if in Crystal Cartesian."""
        if self._input_nrmv_in_rlu is None:
            return True
        return self._input_nrmv_in_rlu

    @input_nrmv_in_rlu.setter
    def input_nrmv_in_rlu(self, val: Any) -> None:
        if not self._set_from_normvec:
            return
        if isinstance(val, (bool, np.bool_)):
            input_is_rlu = bool(val)
        elif isinstance(val, str):
            if val.lower() == "rlu":
                input_is_rlu = True
            elif val.lower() == "cc":
                input_is_rlu = False
            else:
                raise HoraceError(
                    "HORACE:SymopSetPlaneIntrerface:invalid_argument",
                    f"Only 'rlu' or 'cc' string parameters or logical flag may be used to specify normal coordinates type. Provided: {val!r}",
                )
        else:
            raise HoraceError(
                "HORACE:SymopSetPlaneIntrerface:invalid_argument",
                f"Only 'rlu' or 'cc' string parameters or logical flag may be used to specify normal coordinates type. Provided: {val!r}",
            )
        if self._do_check_combo_arg:
            self.check_combo_arg(input_is_rlu)
        else:
            self._input_nrmv_in_rlu = input_is_rlu

    # ------------------------------------------------------------------
    # Static & Protected Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def check_and_sanitize_coord(*varargin: Any) -> Tuple[List[Any], Optional[bool]]:
        """Extract 'rlu'/'cc' coordinate specifier from constructor arguments."""
        args = list(varargin)
        coord_idx: Optional[int] = None
        for idx, arg in enumerate(args):
            if isinstance(arg, str) and arg.lower() in ("cc", "rlu"):
                coord_idx = idx
                break
        if coord_idx is None:
            return args, None

        input_nrmv_in_rlu = args[coord_idx].lower() == "rlu"
        if (
            coord_idx > 0
            and isinstance(args[coord_idx - 1], str)
            and args[coord_idx - 1].lower().startswith("input")
        ):
            del args[coord_idx - 1 : coord_idx + 1]
        else:
            del args[coord_idx]
        return args, input_nrmv_in_rlu

    def check_and_caclulate_vectors_and_R_(self) -> "SymopSetPlaneInterface":
        """Synchronize (u, v), normvec, and R according to current b_matrix.

        Ported from ``@SymopSetPlaneInterface/private/check_and_caclulate_vectors_and_R_.m``.
        """
        use_bmat = self._b_matrix is not None
        bmat = self._b_matrix if use_bmat else np.eye(3, dtype=float)

        if self._set_from_normvec:
            self._u, self._v, self._normvec, input_nrmv_in_rlu = _get_uv_from_normvec(
                self._normvec, self._input_nrmv_in_rlu, bmat
            )
            if use_bmat:
                self._set_from_normvec = False
                self._input_nrmv_in_rlu = False
            else:
                self._input_nrmv_in_rlu = input_nrmv_in_rlu
        else:
            if use_bmat:
                self._input_nrmv_in_rlu = False
            if float(np.linalg.norm(np.cross(self._u, self._v))) < 1e-6:
                raise HoraceError(
                    "HORACE:SymopReflection:invalid_argument",
                    "vectors u and v must not be colinear",
                )
            e1 = bmat @ self._u if use_bmat else self._u
            e2 = bmat @ self._v if use_bmat else self._v
            n = np.cross(e1, e2)
            self._normvec = n / float(np.linalg.norm(n))

        self._R = self.calculate_transform(bmat)
        return self

    def check_combo_arg(
        self, input_nrmv_in_rlu: Optional[bool] = None
    ) -> "SymopSetPlaneInterface":
        """Validate interdependent properties (u, v, normvec, offset, b_matrix) and compute R."""
        if input_nrmv_in_rlu is not None:
            self._input_nrmv_in_rlu = input_nrmv_in_rlu
        self.check_offset_b_matrix_consistency()
        self.check_and_caclulate_vectors_and_R_()
        return self

