"""Single pixel page operation used by func_eval function.

Ported from ``horace_core/sqw/page_operations/PageOp_func_eval.m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config


class PageOp_func_eval(PageOpBase):
    """Single pixel page operation used by ``func_eval`` function.

    Ported from ``horace_core/sqw/page_operations/PageOp_func_eval.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._op_name: str = "func_eval"
        self.split_at_bin_edges: bool = True

    def init(self, sqw_obj: Any, *args: Any) -> "PageOp_func_eval":
        """Initialize ``PageOp_func_eval`` with target ``sqw`` object whose image was already evaluated."""
        super().init(sqw_obj)
        self._img.e = np.zeros_like(self._img.s)
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_func_eval":
        """Replicate pre-computed image signal into pixels for the current bin slice."""
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        s_flat = np.asarray(self._img.s, dtype=np.float64).ravel(order="F")
        npix_flat = np.asarray(self.npix, dtype=np.int64).ravel(order="F")
        s = np.repeat(s_flat[idx0:idx1], npix_flat[idx0:idx1])
        self._page_data[self.signal_idx, :] = s
        self._page_data[self.var_idx, :] = 0.0
        return self

    def get_changes_pix_only(self) -> bool:
        """``PageOp_func_eval`` calculates pixels only using image as source."""
        return True

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "func_eval_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_func_eval":
        log = log_config()
        log.func_eval_split_ratio = val
        return self
