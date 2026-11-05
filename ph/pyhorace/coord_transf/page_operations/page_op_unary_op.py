"""Single pixel page operation used by unary operation manager.

Ported from ``horace_core/sqw/page_operations/PageOp_unary_op.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.data.sigvar import SigVar


class PageOp_unary_op(PageOpBase):
    """Single pixel page operation used by unary operation manager.

    Ported from ``horace_core/sqw/page_operations/PageOp_unary_op.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_holder: Callable[..., Any] = lambda x: x

    def init(self, in_obj: Any, operation: Callable[..., Any]) -> "PageOp_unary_op":
        """Initialize ``PageOp_unary_op`` with target object and unary operation."""
        super().init(in_obj)
        self.op_holder = operation
        op_name = getattr(operation, "__name__", str(operation))
        self._op_name = f"unary_op: {op_name}"
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_unary_op":
        """Apply unary operation on signal/variance of current pixel page."""
        pg_result = self.op_holder(
            SigVar(
                self._page_data[self.signal_idx, :],
                self._page_data[self.var_idx, :],
            )
        )
        self._page_data[self.signal_idx, :] = np.asarray(
            pg_result.s, dtype=np.float64
        ).ravel(order="F")
        self._page_data[self.var_idx, :] = np.asarray(
            pg_result.e, dtype=np.float64
        ).ravel(order="F")

        if self._does_img_op:
            self.update_img_accumulators(
                npix_block, npix_idx, pg_result.s, pg_result.e
            )
        return self

    def finish_op(self, in_obj: Any) -> Tuple[Any, "PageOp_unary_op"]:
        """Finalize unary operation and zero any negative variance values."""
        if self._does_img_op:
            self._var_acc[self._var_acc < 0] = 0.0
        return super().finish_op(in_obj)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "unary_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_unary_op":
        log = log_config()
        log.unary_split_ratio = val
        return self

