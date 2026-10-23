"""Helper class for testing aProjectionBase methods and get_nrange."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, line_axes
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.bmatrix import _infer_nargout


class proj_test_GET_NRANGE_helper(aProjectionBase):
    """Concrete test helper subclass of aProjectionBase with identity transformation."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._may_contribute_returns: Optional[tuple[np.ndarray, np.ndarray]] = None
        if args or kwargs:
            self.init(*args, **kwargs)

    def get_pix_img_transformation(
        self,
        ndim: int = 3,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        if nargout is None:
            nargout = _infer_nargout(default=1)
        mat = np.eye(ndim, dtype=float)
        off = self._offset[:ndim].copy()
        scales = self._img_scales[:ndim].copy()
        if nargout == 1:
            return mat
        if nargout == 2:
            return mat, off
        if nargout == 3:
            return mat, off, scales
        return mat, off, scales, bool(np.any(np.abs(off) > self.tol_))

    def transform_pix_to_img(self, pix_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_data)
        res_3d = q_cc - self._offset[:3].reshape(3, 1)
        if ndim == 4:
            res = np.vstack([res_3d, dE - self._offset[3]])
        else:
            res = res_3d
        if is_row:
            res = res.T
            if np.asarray(pix_data).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(img_data)
        res_3d = q_img + self._offset[:3].reshape(3, 1)
        if ndim == 4:
            res = np.vstack([res_3d, dE + self._offset[3]])
        else:
            res = res_3d
        if is_row:
            res = res.T
            if np.asarray(img_data).ndim == 1:
                return res.ravel()
        return res
