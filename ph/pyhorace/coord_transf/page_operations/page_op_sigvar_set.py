"""Single pixel page operation used by ``sigvar_set`` and ``replicate`` methods.

Ported from ``horace_core/sqw/page_operations/PageOp_sigvar_set.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config


class PageOp_sigvar_set(PageOpBase):
    """Single pixel page operation used by ``sigvar_set`` and ``replicate`` methods.

    Ported from ``horace_core/sqw/page_operations/PageOp_sigvar_set.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.in_replicate: bool = bool(kwargs.pop("in_replicate", False))
        super().__init__(*args, **kwargs)
        if self.in_replicate:
            self.op_name_ = "sigvar_set"
        else:
            self.op_name_ = "replicate_with_pix"
        self.split_at_bin_edges = True

    def init(self, sqw_obj: Any = None) -> "PageOp_sigvar_set":
        """Initialize ``PageOp_sigvar_set`` on ``sqw_obj`` whose image is already updated."""
        super().init(sqw_obj)
        if sqw_obj is None:
            return self
        if self.changes_pix_only:
            self.sig_acc_ = np.empty(0, dtype=np.float64)
        else:
            self.var_acc_ = np.zeros_like(self.sig_acc_, dtype=np.float64, order="F")
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_sigvar_set":
        """Apply image signal and variance scaled by ``npix`` to pixel page."""
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        npix = np.asarray(self.npix[idx0:idx1], dtype=np.int64).ravel(order="F")
        s_bins = np.asarray(self.img_.s, dtype=np.float64).ravel(order="F")[idx0:idx1]
        e_bins = np.asarray(self.img_.e, dtype=np.float64).ravel(order="F")[idx0:idx1]
        s = np.repeat(s_bins, npix)
        e = np.repeat(e_bins * npix, npix)
        self.page_data_[self.signal_idx, :] = s
        self.page_data_[self.var_idx, :] = e
        if self.changes_pix_only:
            return self
        self.update_img_accumulators(npix_block, npix_idx, s, e)
        return self

    def get_changes_pix_only(self) -> bool:
        """Return ``not self.in_replicate``."""
        return not bool(self.in_replicate)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "sigvar_set_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_sigvar_set":
        log = log_config()
        log.sigvar_set_split_ratio = val
        return self

