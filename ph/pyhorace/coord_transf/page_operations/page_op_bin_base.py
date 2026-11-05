"""Base class for single-page binary pixel operations.

Ported from ``horace_core/sqw/page_operations/PageOp_bin_Base.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.sigvar import sigvar


def _func_to_str(fn: Any) -> str:
    """Return a MATLAB ``func2str``-compatible name for a callable."""
    if isinstance(fn, str):
        return fn
    name = getattr(fn, "__name__", None)
    if name:
        return str(name)
    return str(fn)


class PageOp_bin_Base(PageOpBase):
    """Single page pixel operation used by binary operation manager.

    Applied to two ``sqw`` objects and a number or array of numbers with size
    1, ``numel(npix)``, or ``PixelData.num_pixels``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_handle: Optional[Callable[..., Any]] = None
        self.operand: Any = None
        self.flip: bool = False
        self._sigvar_idx: np.ndarray = PixelDataBase.field_index("sig_var") - 1
        self._all_idx: np.ndarray = PixelDataBase.field_index("all_indexes") - 1
        self.sigvar1: sigvar = sigvar()
        self.sigvar2: sigvar = sigvar()
        self._last_pix_idx: np.ndarray = np.array([], dtype=np.intp)
        self.split_at_bin_edges = True

    def init(
        self,
        w1: Any,
        operand: Any = None,
        operation: Optional[Callable[..., Any]] = None,
        flip: bool = False,
        npix: Any = None,
        nargout: int = 1,
    ) -> Union["PageOp_bin_Base", Tuple["PageOp_bin_Base", str]]:
        """Initialize binary page operation on ``w1`` and ``operand``."""
        super().init(w1)
        self.op_handle = operation
        self.operand = operand
        self.flip = bool(flip)

        if self._img is None:
            name1_obj = "pix"
        else:
            name1_obj = "sqw"

        if npix is not None and np.asarray(npix).size > 0:
            npix_arr = np.asarray(npix, dtype=np.int64).ravel(order="F")
            if self._pix.num_pixels != int(np.sum(npix_arr)):
                raise HoraceError(
                    f"HORACE:PageOp_bin_Base:invalid_argument - "
                    f"Number of pixels of the first operand ({self._pix.num_pixels}) inconsistent "
                    f"with their bin-distribution, provided as #5-th argument npix ({int(np.sum(npix_arr))})"
                )
            self.npix = npix_arr

        if not self.changes_pix_only:
            self._var_acc = np.zeros(self.npix.size, dtype=np.float64)

        self._pix_idx_start = 1
        if nargout == 2:
            return self, name1_obj
        return self

    def get_page_data(
        self,
        idx: int,
        npix_blocks: Sequence[np.ndarray],
        nargout: int = 1,
    ) -> Union["PageOp_bin_Base", Tuple["PageOp_bin_Base", np.ndarray]]:
        """Retrieve block of data used in page operation.

        Overloaded for dealing with two ``PixelData`` objects so returns
        1-based indices of the first data object when ``nargout == 2``.
        """
        npix_block = np.asarray(npix_blocks[idx - 1])
        npix_count = int(np.sum(npix_block))
        pix_idx_end = self._pix_idx_start + npix_count - 1

        if npix_count > 0:
            pix_idx = np.arange(self._pix_idx_start, pix_idx_end + 1, dtype=np.intp)
        else:
            pix_idx = np.array([], dtype=np.intp)
        self._page_data = np.asarray(
            self._pix.get_pixels(pix_idx, "-raw", "-align"), dtype=np.float64
        )
        self._pix_idx_start = pix_idx_end + 1
        self.page_num = idx
        self._last_pix_idx = pix_idx

        if nargout == 2:
            return self, pix_idx
        return self

    def set_op_name(self, obj1_name: str, obj2_name: str) -> "PageOp_bin_Base":
        """Define the name of the binary operation from object names and operation."""
        if self.flip:
            name1 = obj2_name
            name2 = obj1_name
        else:
            name1 = obj1_name
            name2 = obj2_name
        self._op_name = f"binary op: {_func_to_str(self.op_handle)} between {name1} and {name2}"
        return self

