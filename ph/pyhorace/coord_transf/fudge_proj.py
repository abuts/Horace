"""Legacy spectrometer-to-rlu rectilinear projection class (fudge_proj)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.coord_transf.line_proj import line_proj


class fudge_proj(line_proj):
    """Subclass of line_proj supporting legacy spectrometer-to-rlu transformations."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._spec_to_rlu: np.ndarray = np.eye(3, dtype=float)
        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        fields = list(super().saveable_fields())
        if "spec_to_rlu" not in fields:
            fields.append("spec_to_rlu")
        return fields

    @property
    def spec_to_rlu(self) -> np.ndarray:
        return self._spec_to_rlu.copy()

    @spec_to_rlu.setter
    def spec_to_rlu(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.shape != (3, 3):
            raise HoraceError("HORACE:fudge_proj:invalid_argument", "spec_to_rlu must be a 3x3 matrix")
        self._spec_to_rlu = arr.copy()
