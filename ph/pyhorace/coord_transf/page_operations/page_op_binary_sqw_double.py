"""Single-page binary operation between an SQW/PixelData object and a scalar/double.

Ported from ``horace_core/sqw/page_operations/PageOp_binary_sqw_double.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_bin_base import PageOp_bin_Base
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.core.data_op_interface import data_op_interface
from pyhorace.data.sigvar import sigvar


class PageOp_binary_sqw_double(PageOp_bin_Base):
    """Single page pixel operation between an ``sqw``/``PixelData`` and a scalar operand."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def init(
        self,
        w1: Any,
        operand: Any = None,
        operation: Optional[Callable[..., Any]] = None,
        flip: bool = False,
        *args: Any,
    ) -> "PageOp_binary_sqw_double":
        """Initialize binary operation between ``w1`` and scalar ``operand``."""
        _, name_op1 = super().init(w1, operand, operation, flip, *args, nargout=2)
        self.split_at_bin_edges = False

        from pyhorace.sqw.sqw import sqw

        if isinstance(self.operand, (int, float, np.number)) and not isinstance(self.operand, bool):
            name_op2 = "scalar"
            self.sigvar2 = sigvar(np.array([[float(self.operand)]], dtype=np.float64), None)
        elif isinstance(self.operand, np.ndarray) and np.issubdtype(self.operand.dtype, np.number):
            name_op2 = "scalar"
            self.sigvar2 = sigvar(np.asarray(self.operand, dtype=np.float64), None)
        elif isinstance(self.operand, sqw):
            name_op2 = "0-dimensional sqw"
            self.sigvar2 = sigvar(self.operand.data.s, self.operand.data.e)
        elif isinstance(self.operand, sigvar):
            name_op2 = f"scalar {type(self.operand).__name__}"
            self.sigvar2 = sigvar(self.operand.s, self.operand.e)
        elif hasattr(self.operand, "sigvar"):
            name_op2 = f"scalar {type(self.operand).__name__}"
            self.sigvar2 = self.operand.sigvar()
        else:
            name_op2 = f"scalar {type(self.operand).__name__}"
            self.sigvar2 = sigvar(self.operand)

        self.set_op_name(name_op1, name_op2)
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_binary_sqw_double":
        """Perform binary operation between input object and scalar operand."""
        self.sigvar1.sig_var = self.page_data_[self.sigvar_idx_, :]

        if self.flip:
            res = data_op_interface.binary_op_manager_single(
                self.sigvar2, self.sigvar1, self.op_handle
            )
        else:
            res = data_op_interface.binary_op_manager_single(
                self.sigvar1, self.sigvar2, self.op_handle
            )

        self.page_data_[self.sigvar_idx_, :] = res.sig_var
        if self.changes_pix_only:
            return self

        self.update_img_accumulators(npix_block, npix_idx, res.s, res.e)
        return self

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "sqw_binary_double_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_binary_sqw_double":
        log = log_config()
        log.sqw_binary_double_split_ratio = val
        return self

