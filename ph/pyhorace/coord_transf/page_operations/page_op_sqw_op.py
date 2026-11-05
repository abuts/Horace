"""Single pixel page operation used by sqw_op algorithm.

Ported from ``horace_core/sqw/page_operations/PageOp_sqw_op.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval


class PageOp_sqw_op(PageOp_sqw_eval):
    """Single pixel page operation used by ``sqw_op`` algorithm.

    Ported from ``horace_core/sqw/page_operations/PageOp_sqw_op.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._op_name: str = "sqw_op"
        self.op_holder: Callable[..., Any] = lambda obj, *a: []
        self.sigvar_idx = np.array([self.signal_idx, self.var_idx], dtype=np.int64)

    def init(
        self,
        sqw_obj: Any,
        operation: Callable[..., Any],
        op_param: Any = None,
        pop_options: Any = None,
    ) -> "PageOp_sqw_op":
        """Initialize ``PageOp_sqw_op`` operation over input ``sqw`` object.

        Parameters
        ----------
        sqw_obj : sqw
            Instance of ``sqw`` object to perform operation on.
        operation : callable
            Function handle constructed according to ``sqw_op`` rules.
        op_param : sequence, optional
            Operation parameters passed as ``operation(obj, *op_param)``.
        pop_options : dict or object, optional
            Options structure with ``nopix`` boolean field.
        """
        from pyhorace.sqw.sqw import sqw

        if not isinstance(sqw_obj, sqw) or sqw_obj.pix.num_pixels == 0:
            if isinstance(sqw_obj, sqw):
                mess_out = "Provided sqw object with no pixels"
            else:
                mess_out = f"Provided {type(sqw_obj).__name__} object"
            raise HoraceError(
                "HORACE:PageOp_sqw_op:invalid_argument",
                f"sqw_op can be only applied to sqw objects which contains non-zero pixels.\n Provided: {mess_out}",
            )

        super().init(sqw_obj, operation, op_param, False)
        self._var_acc = np.zeros(self.npix.size, dtype=np.float64)
        self.split_at_bin_edges = True
        if pop_options is not None:
            if isinstance(pop_options, dict):
                self.do_nopix = bool(pop_options.get("nopix", False))
            else:
                self.do_nopix = bool(getattr(pop_options, "nopix", False))
        else:
            self.do_nopix = False
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: Sequence[int]
    ) -> "PageOp_sqw_op":
        """Apply user-defined operation over a page of pixels in memory."""
        if self.op_parms is None:
            parms = []
        elif isinstance(self.op_parms, (list, tuple)):
            parms = list(self.op_parms)
        else:
            parms = [self.op_parms]

        sig_var = self.op_holder(self, *parms)
        sig_var_arr = np.asarray(sig_var, dtype=np.float64)
        if sig_var_arr.ndim == 1:
            sig_var_arr = sig_var_arr.reshape(2, -1, order="F")
        self._page_data[[self.signal_idx, self.var_idx], :] = sig_var_arr
        self.update_img_accumulators(
            npix_block, npix_idx, sig_var_arr[0, :], sig_var_arr[1, :]
        )
        return self

    def update_img_accumulators(
        self,
        npix_block: np.ndarray,
        npix_idx: Sequence[int],
        new_signal: np.ndarray,
        variance: Any = None,
        *args: Any,
    ) -> "PageOp_sqw_op":
        """Re-overload to enable variance accumulator (unlike ``PageOp_sqw_eval``)."""
        PageOpBase.update_img_accumulators(
            self, npix_block, npix_idx, new_signal, variance
        )
        return self

    def finish_op(self, out_obj: Any) -> Tuple[Any, "PageOp_sqw_op"]:
        """Finalize ``sqw_op`` operation, preserving variance accumulator."""
        return PageOpBase.finish_op(self, out_obj)


__all__ = ["PageOp_sqw_op"]

