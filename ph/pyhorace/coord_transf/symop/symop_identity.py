"""Identity symmetry operator (SymopIdentity).

Translated from MATLAB ``horace_core/symop/@SymopIdentity/SymopIdentity.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pyhorace.coord_transf.symop.symop import Symop
from pyhorace.core.serializable import register_serializable


@register_serializable
class SymopIdentity(Symop):
    """Identity symmetry operation (leaves vectors, pixels, and projections unchanged)."""

    def class_version(self) -> int:
        return 1

    classVersion = class_version

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._R = np.eye(3, dtype=float)

    def saveable_fields(self) -> List[str]:
        return []

    saveableFields = saveable_fields

    def calculate_transform(self, b_mat: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the 3x3 identity matrix (or ``b_mat`` if provided)."""
        if b_mat is None:
            return np.eye(3, dtype=float)
        return np.asarray(b_mat, dtype=float).copy()

    def in_irreducible(self, coords: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """All points are in the irreducible zone of the identity operator."""
        arr = np.asarray(coords)
        return np.ones(arr.shape[1], dtype=bool)

    def check_combo_arg(self) -> "SymopIdentity":
        self._R = np.eye(3, dtype=float)
        return self

