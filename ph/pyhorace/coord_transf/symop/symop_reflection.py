"""Reflection symmetry operator (SymopReflection).

Translated from MATLAB ``horace_core/symop/@SymopReflection/SymopReflection.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pyhorace.coord_transf.symop.symop_set_plane_interface import SymopSetPlaneInterface
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import register_serializable


@register_serializable
class SymopReflection(SymopSetPlaneInterface):
    """Reflection symmetry operation across a plane defined by (u, v) or normal vector."""

    def class_version(self) -> int:
        return 1

    classVersion = class_version

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._R = np.diag([1.0, 1.0, -1.0])

        raw_args = list(args)
        for k, v in kwargs.items():
            raw_args.extend([k, v])

        if len(raw_args) == 0:
            return

        if len(raw_args) == 1 and isinstance(raw_args[0], SymopReflection):
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
            return

        argi, input_nrmv_in_rlu = self.check_and_sanitize_coord(*raw_args)
        if len(raw_args) > 0 and isinstance(raw_args[0], str):
            flds = ["normvec", "offset", "b_matrix"]
            self._set_from_normvec = True
            self._input_nrmv_in_rlu = input_nrmv_in_rlu
        else:
            flds = self.saveable_fields()

        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *argi
        )
        if remains:
            raise HoraceError(
                "HORACE:SymopReflection:invalid_argument",
                f"Non-empty remaining arguments on SymopReflection constructor: {remains!r}",
            )

    def saveable_fields(self) -> List[str]:
        return ["u", "v", "offset", "b_matrix"]

    saveableFields = saveable_fields

    def calculate_transform(self, b_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute Householder reflection matrix I - 2 * n * n^T in Crystal Cartesian coordinates."""
        n = self._normvec.reshape(3, 1)
        return np.eye(3, dtype=float) - 2.0 * (n @ n.T)

    def in_irreducible(self, coords: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """Return True for points on the positive side of the reflection plane."""
        arr = np.asarray(coords, dtype=float)
        u_offset = self._u_offset.reshape(3, 1)
        proj_val = ((arr - u_offset).T @ self._normvec).ravel()
        if tolerance <= 0.0:
            return proj_val > 0.0
        return (proj_val + tolerance) > 0.0

