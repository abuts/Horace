"""General 3x3 orthogonal symmetry operator (SymopGeneral).

Translated from MATLAB ``horace_core/symop/@SymopGeneral/SymopGeneral.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pyhorace.coord_transf.symop.symop import Symop
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import register_serializable


@register_serializable
class SymopGeneral(Symop):
    """General 3x3 orthogonal transformation symmetry operator."""

    def class_version(self) -> int:
        return 1

    classVersion = class_version

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._R = np.eye(3, dtype=float)

        raw_args = list(args)
        for k, v in kwargs.items():
            raw_args.extend([k, v])

        if len(raw_args) == 0:
            return

        if len(raw_args) == 1 and isinstance(raw_args[0], SymopGeneral):
            other = raw_args[0]
            self._R = other._R.copy()
            self._offset = other._offset.copy()
            self._u_offset = other._u_offset.copy()
            self._offset_specified_uoffset_not = other._offset_specified_uoffset_not
            self._b_matrix = None if other._b_matrix is None else other._b_matrix.copy()
            return

        flds = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *raw_args
        )
        if remains:
            raise HoraceError(
                "HORACE:SymopRotation:invalid_argument",
                f"Invalid/excess input argument(s) for SymopGeneral: {remains!r}",
            )

    @property
    def W(self) -> np.ndarray:
        """Alias for ``R`` (transformation matrix in Crystal Cartesian coordinates)."""
        return self._R.copy()

    @W.setter
    def W(self, val: Any) -> None:
        self._set_R(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_R(self, val: Any) -> None:
        if not self.is_3x3matrix(val):
            raise HoraceError(
                "HORACE:symop:invalid_argument",
                "SymopGeneral transformation matrix has to be 3x3 matrix with determinant +-1",
            )
        mat = np.asarray(val, dtype=float).copy()
        det_val = float(np.linalg.det(mat))
        if abs(abs(det_val) - 1.0) > 1e-4:
            raise HoraceError(
                "HORACE:symop:invalid_argument",
                f"SymopGeneral transformation matrix has to be 3x3 matrix with determinant +-1. Got det={det_val}",
            )
        self._R = mat

    def saveable_fields(self) -> List[str]:
        return ["R", "offset", "b_matrix"]

    saveableFields = saveable_fields

    def calculate_transform(self, b_mat: Optional[np.ndarray] = None) -> np.ndarray:
        return self._R.copy()

    def in_irreducible(self, coords: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        raise HoraceError(
            "HORACE:symop:not_implemented",
            "Cannot compute the irreducible set from general 3x3 transform",
        )

    def check_combo_arg(self) -> "SymopGeneral":
        self.check_offset_b_matrix_consistency()
        return self

