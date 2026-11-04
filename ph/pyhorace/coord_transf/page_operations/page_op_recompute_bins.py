"""Page operation that recomputes binned image signal and variance from pixels.

Ported from:
- ``horace_core/sqw/page_operations/PageOp_recompute_bins.m``
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store


class PageOp_recompute_bins(PageOpBase):
    """Single-pixels page operation which does not change the pixel values
    unless pixels are misaligned. If they are misaligned, resulting pixels get
    realigned as the result of the operation.

    Used by ``recompute_bin_data``, ``finalize_alignment``, and
    ``recalc_data_range`` algorithms.

    Ported from ``horace_core/sqw/page_operations/PageOp_recompute_bins.m``.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.op_name_ = "recompute_bin_data_and_ranges"
        self.split_at_bin_edges = True
        self.changes_pix_only_: bool = False
        self.img_range: Any = None
        if len(args) > 0:
            self.init(*args)

    def init(self, in_obj: Any = None, *args: Any) -> "PageOp_recompute_bins":
        super().init(in_obj, *args)
        if in_obj is None:
            return self
        if not self.changes_pix_only:
            self.var_acc_ = np.zeros(np.asarray(self.npix).size, dtype=np.float64)
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_recompute_bins":
        if self.changes_pix_only:
            return self
        signal = self.page_data_[self.signal_idx_, :]
        error = self.page_data_[self.var_idx_, :]
        self.update_img_accumulators(npix_block, np.asarray(npix_idx), signal, error)
        return self

    # =========================================================================
    # Protected Overloadable Methods
    # =========================================================================

    def get_changes_pix_only(self) -> bool:
        return bool(self.changes_pix_only_ or self.img_ is None)

    def set_changes_pix_only(self, val: bool) -> "PageOp_recompute_bins":
        self.changes_pix_only_ = bool(val)
        return self

    def get_do_missing_range_warning(self) -> bool:
        # These operations are intended for computing missing range,
        # so no point in warning that range is missing.
        return False

    def get_info_split_log_ratio(self) -> int:
        try:
            return int(
                config_store.instance().get_value(
                    "log_config", "recompute_bins_split_ratio"
                )
            )
        except Exception:
            return super().get_info_split_log_ratio()

    def set_info_split_log_ratio(self, val: Union[int, float]) -> None:
        try:
            from pyhorace.core.configurations.log_config import log_config

            log = log_config()
            log.recompute_bins_split_ratio = val
        except Exception:
            super().set_info_split_log_ratio(val)
