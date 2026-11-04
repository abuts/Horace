"""Single pixel page operation used by sqw_eval algorithm.

Ported from ``horace_core/sqw/page_operations/PageOp_sqw_eval.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from pyhorace.coord_transf.page_operations._utils import (
    average_bin_data,
    compute_bin_data,
)
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace. core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.sqw.sqw_dnd_base import _call_user_func


class PageOp_sqw_eval(PageOpBase):
    """Single pixel page operation used by ``sqw_eval`` algorithm.

    Ported from ``horace_core/sqw/page_operations/PageOp_sqw_eval.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_name_: str = "sqw_eval"
        self.split_at_bin_edges: bool = True
        self.op_holder: Callable[..., Any] = lambda h, k, l, e: []
        self.average: bool = False
        self.proj: Any = None
        self.op_parms: Any = None
        self.sigvar_idx: np.ndarray = PixelDataBase.field_index("sig_var")
        self.npix_block_: Optional[np.ndarray] = None
        self.npix_idx_: Optional[np.ndarray] = None

    @property
    def img(self) -> Any:
        """Read-only access to internal image holder to use in ``sqw_op``."""
        return self.img_

    @property
    def npix_idx(self) -> Optional[np.ndarray]:
        """2-element array ``[nbin_min, nbin_max]`` of current image bin indices."""
        return self.npix_idx_

    @property
    def npix_block(self) -> Optional[np.ndarray]:
        """Read-only access to current ``npix_block``."""
        return self.npix_block_

    def init(
        self,
        sqw_obj: Any,
        operation: Callable[..., Any],
        op_param: Any = None,
        average: bool = False,
    ) -> "PageOp_sqw_eval":
        """Initialize ``PageOp_sqw_eval`` with target object, function, and parameters."""
        super().init(sqw_obj)
        self.average = bool(average)
        self.op_holder = operation
        self.op_parms = op_param
        self.split_at_bin_edges = self.average
        if hasattr(sqw_obj, "data") and hasattr(sqw_obj.data, "proj"):
            self.proj = sqw_obj.data.proj
        return self

    def update_img_accumulators(
        self,
        npix_block: np.ndarray,
        npix_idx: Sequence[int],
        new_signal: np.ndarray,
        *args: Any,
    ) -> "PageOp_sqw_eval":
        """Update image signal accumulator for ``sqw_eval`` (variance is zeroed at finish)."""
        img_signal, _ = compute_bin_data(npix_block, new_signal, None, True)
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        self.sig_acc_[idx0:idx1] = (
            self.sig_acc_[idx0:idx1] + img_signal.ravel(order="F")
        )
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_sqw_eval":
        """Apply user-defined ``sqw_eval`` operation over a page of pixels in memory."""
        qw = self.proj.transform_pix_to_hkl(self.page_data_[self.coord_idx, :])
        qw_pix_coord = [
            qw[0, :].reshape(-1, 1),
            qw[1, :].reshape(-1, 1),
            qw[2, :].reshape(-1, 1),
            qw[3, :].reshape(-1, 1),
        ]
        if self.average:
            qw_ave = average_bin_data(npix_block, qw_pix_coord)
            qw_ave = [np.asarray(x, dtype=np.float64).reshape(-1, 1) for x in qw_ave]
            new_signal = _call_user_func(
                self.op_holder, *qw_ave, pars=self.op_parms
            )
            new_signal = np.repeat(
                np.asarray(new_signal, dtype=np.float64).ravel(order="F"),
                np.asarray(npix_block, dtype=np.int64).ravel(order="F"),
            )
        else:
            new_signal = _call_user_func(
                self.op_holder, *qw_pix_coord, pars=self.op_parms
            )
            new_signal = np.asarray(new_signal, dtype=np.float64).ravel(order="F")

        self.page_data_[self.signal_idx, :] = new_signal
        self.page_data_[self.var_idx, :] = 0.0
        self.update_img_accumulators(npix_block, npix_idx, new_signal)
        return self

    def finish_op(self, out_obj: Any) -> Tuple[Any, "PageOp_sqw_eval"]:
        """Finalize ``sqw_eval`` page operation and zero image variance accumulator."""
        self.var_acc_ = np.zeros_like(self.sig_acc_)
        return super().finish_op(out_obj)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "sqw_eval_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_sqw_eval":
        log = log_config()
        log.sqw_eval_split_ratio = val
        return self
