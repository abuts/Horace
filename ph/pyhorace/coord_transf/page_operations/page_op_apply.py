"""Page operation class used by sqw.apply and PixelDataBase.apply methods.

Ported from ``horace_core/sqw/page_operations/PageOp_apply.m``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory


class PageOp_apply(PageOp_sqw_eval):
    """Single pixel page operation used by ``apply`` algorithm.

    Ported from ``horace_core/sqw/page_operations/PageOp_apply.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_name_: str = "apply"
        self.split_at_bin_edges: bool = False
        self.compute_variance_: bool = True
        self.change_var_: bool = True

    @property
    def compute_variance(self) -> bool:
        """If True, compute variance of image from pixel variances."""
        return self.compute_variance_

    @compute_variance.setter
    def compute_variance(self, val: bool) -> None:
        self.compute_variance_ = bool(val)
        self.change_var_ = self.compute_variance_

    def init(
        self,
        sqw_obj: Any,
        operation: Union[Callable[..., Any], Sequence[Callable[..., Any]]],
        op_param: Any = None,
        compute_variance: Optional[bool] = None,
        recompute_bins: Optional[bool] = None,
    ) -> "PageOp_apply":
        """Initialize ``PageOp_apply`` with target object, callable(s), and parameters."""
        if not isinstance(operation, (list, tuple)):
            op_list = [operation]
        else:
            op_list = list(operation)

        if (
            op_param is None
            or (isinstance(op_param, (list, tuple)) and len(op_param) == 0)
            or (
                isinstance(op_param, (list, tuple))
                and len(op_param) == 1
                and isinstance(op_param[0], (list, tuple))
                and len(op_param[0]) == 0
            )
        ):
            parm_list: List[Any] = [[] for _ in op_list]
        elif (
            isinstance(op_param, (list, tuple))
            and len(op_param) == len(op_list)
            and all(isinstance(x, (list, tuple)) for x in op_param)
        ):
            parm_list = [list(x) for x in op_param]
        elif isinstance(op_param, (list, tuple)):
            parm_list = [list(op_param) for _ in op_list]
        else:
            parm_list = [[op_param] for _ in op_list]

        super().init(sqw_obj, op_list, parm_list, False)
        if recompute_bins is None:
            recompute_bins = not isinstance(sqw_obj, PixelDataBase)

        self.split_at_bin_edges = False
        if compute_variance is not None:
            self.compute_variance = bool(compute_variance)

        self.does_img_op_ = bool(recompute_bins)
        if not self.does_img_op_:
            self.img_ = None
        else:
            self.var_acc_ = np.zeros(self.img_.npix.size, dtype=np.float64)
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_apply":
        """Apply user-defined function(s) to current page wrapped in ``PixelDataMemory``."""
        pix = PixelDataMemory()
        pix.set_raw_data(self.page_data_)
        for idx, fn in enumerate(self.op_holder):
            args = self.op_parms[idx] if idx < len(self.op_parms) else []
            if args is None:
                args_seq: List[Any] = []
            elif isinstance(args, (list, tuple)):
                args_seq = list(args)
            else:
                args_seq = [args]
            pix = fn(pix, *args_seq)

        self.page_data_ = pix.data
        if self.does_img_op_:
            from pyhorace.coord_transf.page_operations._utils import compute_bin_data

            new_signal = np.asarray(pix.signal, dtype=np.float64).ravel(order="F")
            if self.compute_variance_:
                img_signal, img_var, sig_variance = compute_bin_data(
                    npix_block, new_signal, None, True
                )
                self.page_data_[self.var_idx, :] = sig_variance
            else:
                new_var = np.asarray(pix.variance, dtype=np.float64).ravel(order="F")
                img_signal, img_var = compute_bin_data(
                    npix_block, new_signal, new_var, True
                )
            idx0 = int(npix_idx[0]) - 1
            idx1 = int(npix_idx[1])
            self.sig_acc_[idx0:idx1] = np.asarray(
                img_signal, dtype=np.float64
            ).reshape(self.sig_acc_[idx0:idx1].shape, order="F")
            self.var_acc_[idx0:idx1] = np.asarray(
                img_var, dtype=np.float64
            ).reshape(self.var_acc_[idx0:idx1].shape, order="F")
        return self

    def finish_op(self, out_obj: Any) -> Tuple[Any, "PageOp_apply"]:
        """Finalize ``apply`` operation without zeroing ``var_acc_``."""
        return PageOpBase.finish_op(self, out_obj)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "apply_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_apply":
        log = log_config()
        log.apply_split_ratio = val
        return self
