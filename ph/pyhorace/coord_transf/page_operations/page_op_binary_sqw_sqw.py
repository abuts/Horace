"""Single-page binary operation between two SQW or PixelData objects.

Ported from ``horace_core/sqw/page_operations/PageOp_binary_sqw_sqw.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_bin_base import PageOp_bin_Base
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.core.data_op_interface import data_op_interface
from pyhorace.core.exceptions import HoraceError


class PageOp_binary_sqw_sqw(PageOp_bin_Base):
    """Single page pixel operation applied to two ``sqw``/``PixelData`` objects
    with identical pixel distributions over bins.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sort_pixels_in_bins: bool = True

    def init(
        self,
        w1: Any,
        operand: Any = None,
        operation: Optional[Callable[..., Any]] = None,
        flip: bool = False,
        *args: Any,
    ) -> "PageOp_binary_sqw_sqw":
        """Initialize binary operation between ``w1`` and ``operand``."""
        _, name_op1 = super().init(w1, operand, operation, flip, *args, nargout=2)

        from pyhorace.sqw.sqw import sqw

        if isinstance(operand, sqw):
            name_op2 = "sqw"
            self.operand = operand.pix
        else:
            name_op2 = "pix"
        self.set_op_name(name_op1, name_op2)

        if self.pix_.num_pixels != self.operand.num_pixels:
            raise HoraceError(
                f"HORACE:PageOp_binary_sqw_sqw:invalid_argument - "
                f"Pixels in {self.op_name_} are inconsistent as obj1 has {self.pix_.num_pixels} "
                f"pixels and obj2 has {self.operand.num_pixels} pixels"
            )

        if self.npix.size == 1:
            self.split_at_bin_edges = False
        else:
            self.split_at_bin_edges = True

        self.sort_pixels_in_bins = bool(
            config_store.instance().get_value("hpc_config", "sort_pix_in_binary_op")
        )
        return self

    def get_page_data(
        self,
        idx: int,
        npix_blocks: Sequence[np.ndarray],
        nargout: int = 1,
    ) -> "PageOp_binary_sqw_sqw":
        """Retrieve block of data from both ``PixelData`` objects for page ``idx``."""
        _, pix_idx = super().get_page_data(idx, npix_blocks, nargout=2)
        page_data2 = np.asarray(
            self.operand.get_pixels(pix_idx, "-raw", "-align"), dtype=np.float64
        )
        if self.sort_pixels_in_bins and self.page_data_.shape[1] > 0:
            npix_block = np.asarray(npix_blocks[idx - 1], dtype=np.int64).ravel(order="F")
            chunk_idx = np.repeat(
                np.arange(1, npix_block.size + 1, dtype=np.float64), npix_block
            )
            keys1 = np.vstack([chunk_idx, self.page_data_[self.all_idx_, :]])
            keys2 = np.vstack([chunk_idx, page_data2[self.all_idx_, :]])
            idx1 = np.lexsort(keys1[::-1, :])
            idx2 = np.lexsort(keys2[::-1, :])
            self.page_data_ = self.page_data_[:, idx1]
            page_data2 = page_data2[:, idx2]

        self.sigvar1.sig_var = self.page_data_[self.sigvar_idx_, :]
        self.sigvar2.sig_var = page_data2[self.sigvar_idx_, :]
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_binary_sqw_sqw":
        """Perform binary operation between the two pixel blocks."""
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
            config_store.instance().get_value("log_config", "sqw_binary_sqw_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_binary_sqw_sqw":
        log = log_config()
        log.sqw_binary_sqw_split_ratio = val
        return self

