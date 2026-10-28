"""Rotation symmetry operator (SymopRotation).

Translated from MATLAB ``horace_core/symop/@SymopRotation/SymopRotation.m``
and ``@SymopRotation/private/parse_fold_inputs_.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import rotvec_to_rotmat
from pyhorace.coord_transf.symop.symop import Symop, SymopArray
from pyhorace.coord_transf.symop.symop_identity import SymopIdentity
from pyhorace.coord_transf.symop.symop_set_plane_interface import SymopSetPlaneInterface
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import register_serializable


def _is_numeric_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str, bytes, dict)) or val is None:
        return False
    if isinstance(val, (int, float, np.integer, np.floating)):
        return True
    try:
        arr = np.asarray(val, dtype=float)
    except (ValueError, TypeError):
        return False
    return arr.size == 1


def _parse_fold_inputs(*args: Any) -> Tuple[str, np.ndarray]:
    """Parse optional offset and coordinate system arguments for ``SymopRotation.fold``.

    Ported from ``@SymopRotation/private/parse_fold_inputs_.m``.
    """
    if len(args) > 2:
        raise HoraceError(
            "HORACE:SymopRotation:invalid_argument",
            f"Too many optional arguments ({len(args)}) provided to SymopRotation.fold",
        )
    is_char = [isinstance(a, str) for a in args]
    if sum(is_char) > 1:
        raise HoraceError(
            "HORACE:SymopRotation:invalid_argument",
            "Only one string argument ('rlu' or 'cc') can be provided to SymopRotation.fold",
        )
    if any(is_char):
        char_idx = is_char.index(True)
        coord_system = args[char_idx].lower()
        if coord_system not in ("rlu", "cc"):
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Coordinate system must be 'rlu' or 'cc'. Provided: {args[char_idx]!r}",
            )
        rest = [a for idx, a in enumerate(args) if idx != char_idx]
    else:
        coord_system = "cc"
        rest = list(args)

    if len(rest) == 0:
        offset = np.zeros(3, dtype=float)
    else:
        is_valid, offset = Symop.check_and_brush_3vector(rest[0])
        if not is_valid:
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Offset in SymopRotation.fold must be a 3-vector. Provided: {rest[0]!r}",
            )
    return coord_system, offset


@register_serializable
class SymopRotation(SymopSetPlaneInterface):
    """Rotation symmetry operation by ``theta_deg`` around an axis ``n`` (or normal to plane (u, v))."""

    def class_version(self) -> int:
        return 1

    classVersion = class_version

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._R = np.eye(3, dtype=float)
        self._theta_deg: Optional[float] = 0.0
        self._normvec_u: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        self._normvec_v: np.ndarray = np.array([0.0, -1.0, 0.0], dtype=float)

        raw_args = list(args)
        for k, v in kwargs.items():
            raw_args.extend([k, v])

        if len(raw_args) == 0:
            return

        if len(raw_args) == 1 and isinstance(raw_args[0], SymopRotation):
            other = raw_args[0]
            self._u = other._u.copy()
            self._v = other._v.copy()
            self._normvec = other._normvec.copy()
            self._input_nrmv_in_rlu = other._input_nrmv_in_rlu
            self._set_from_normvec = other._set_from_normvec
            self._offset = other._offset.copy()
            self._u_offset = other._u_offset.copy()
            self._offset_specified_uoffset_not = other._offset_specified_uoffset_not
            self._b_matrix = None if other._b_matrix is None else other._b_matrix.copy()
            self._R = other._R.copy()
            self._theta_deg = other._theta_deg
            self._normvec_u = other._normvec_u.copy()
            self._normvec_v = other._normvec_v.copy()
            return

        self._theta_deg = None
        argi, input_nrmv_in_rlu = self.check_and_sanitize_coord(*raw_args)
        flds = self.saveable_fields()
        if len(argi) > 0 and isinstance(argi[0], str):
            argi = [
                "normvec" if (isinstance(a, str) and a.lower() == "n") else a
                for a in argi
            ]
            flds = ["normvec"] + list(flds)
            self._input_nrmv_in_rlu = input_nrmv_in_rlu
        elif len(argi) > 1 and _is_numeric_scalar(argi[1]):
            flds = ["normvec"] + list(flds)[2:]
            self._input_nrmv_in_rlu = input_nrmv_in_rlu

        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *argi
        )
        if remains:
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Invalid/excess input argument(s) for SymopRotation: {remains!r}",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def theta_deg(self) -> float:
        """Rotation angle in degrees."""
        return 0.0 if self._theta_deg is None else float(self._theta_deg)

    @theta_deg.setter
    def theta_deg(self, val: Any) -> None:
        if not _is_numeric_scalar(val):
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Angle theta must be numeric scalar. It is: {val!r}",
            )
        self._theta_deg = float(np.asarray(val, dtype=float).item())
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def n(self) -> np.ndarray:
        """Alias for ``normvec`` (rotation axis unit vector in Crystal Cartesian coordinates)."""
        return self.normvec

    @n.setter
    def n(self, val: Any) -> None:
        self.normvec = val

    @property
    def normvec_u(self) -> np.ndarray:
        """Normal to the half-plane defined by n and u in Crystal Cartesian coordinates."""
        return self._normvec_u.copy()

    @property
    def normvec_v(self) -> np.ndarray:
        """Normal to the half-plane defined by n and rotated u in Crystal Cartesian coordinates."""
        return self._normvec_v.copy()

    # ------------------------------------------------------------------
    # Serialization & Validation
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        return ["u", "v", "theta_deg", "offset", "b_matrix"]

    saveableFields = saveable_fields

    def calculate_transform(self, b_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute active/passive rotation matrix for ``-theta_deg`` around ``normvec``."""
        theta = 0.0 if self._theta_deg is None else float(self._theta_deg)
        return rotvec_to_rotmat(-theta * self._normvec)

    def check_combo_arg(
        self, input_nrmv_in_rlu: Optional[bool] = None
    ) -> "SymopRotation":
        if self._theta_deg is None:
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                "Rotation angle is not defined",
            )
        super().check_combo_arg(input_nrmv_in_rlu)
        if self._b_matrix is None:
            u_cc = self._u.copy()
            use_rlu_offset = True
        else:
            u_cc = self._b_matrix @ self._u
            use_rlu_offset = False
        u_cc = u_cc / float(np.linalg.norm(u_cc))
        v_cc = self.transform_vec(u_cc, use_rlu_offset=use_rlu_offset)

        self._normvec_u = np.cross(self._normvec, u_cc)
        self._normvec_v = np.cross(v_cc, self._normvec)
        return self

    def in_irreducible(self, coords: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """Return True for points within the irreducible sector of this rotation."""
        arr = np.asarray(coords, dtype=float)
        diff = (arr - self._u_offset.reshape(3, 1)).T
        p_u = (diff @ self._normvec_u).ravel()
        p_v = (diff @ self._normvec_v).ravel()
        if tolerance <= 0.0:
            return (p_u >= 0.0) & (p_v > 0.0)
        return ((p_u + tolerance) >= 0.0) & ((p_v + tolerance) > 0.0)

    # ------------------------------------------------------------------
    # Static Factory: fold
    # ------------------------------------------------------------------
    @staticmethod
    def fold(nfold: Any, axis: Any, *args: Any) -> SymopArray:
        """Generate an n-fold rotation symmetry group starting with ``SymopIdentity``."""
        if (
            not _is_numeric_scalar(nfold)
            or float(nfold) <= 0.0
            or np.floor(float(nfold)) != float(nfold)
        ):
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Only positive integer fold is supported by this method. Provided: {nfold!r}",
            )
        nfold_int = int(float(nfold))
        coord_system, offset = _parse_fold_inputs(*args)

        is_3vec, axis_vec = Symop.check_and_brush_3vector(axis)
        use_uv = False
        u_vec = np.zeros(3, dtype=float)
        v_vec = np.zeros(3, dtype=float)
        if not is_3vec:
            try:
                arr = np.asarray(axis, dtype=float)
            except (ValueError, TypeError) as exc:
                raise HoraceError(
                    "HORACE:SymopRotation:invalid_argument",
                    f"Rotation axis must be a 3-vector or 2x3 matrix. Provided: {axis!r}",
                ) from exc
            if arr.shape != (2, 3):
                raise HoraceError(
                    "HORACE:SymopRotation:invalid_argument",
                    f"Rotation axis must be a 3-vector or 2x3 matrix. Provided: {axis!r}",
                )
            _, u_vec = Symop.check_and_brush_3vector(arr[0, :])
            _, v_vec = Symop.check_and_brush_3vector(arr[1, :])
            use_uv = True

        ang = 360.0 / float(nfold_int)
        ops: list[Symop] = [SymopIdentity()]
        for i in range(1, nfold_int):
            if use_uv:
                op = SymopRotation(u_vec, v_vec, ang * i, offset)
            else:
                op = SymopRotation(axis_vec, coord_system, ang * i, offset)
            ops.append(op)
        return SymopArray(ops)

