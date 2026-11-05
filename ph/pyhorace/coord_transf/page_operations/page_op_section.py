"""
Page operation class used by the ``section`` method.

Ported from ``horace_core/sqw/page_operations/PageOp_section.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.page_operations._utils import (
    get_ind_from_ranges,
    split_data_blocks,
)
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store


class PageOp_section(PageOpBase):
    """
    Single pixel page operation used by the ``section`` algorithm.

    Ported from ``horace_core/sqw/page_operations/PageOp_section.m``.
    """

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._op_name = "section"
        self._block_starts: np.ndarray = np.empty(0, dtype=np.int64)
        self._block_sizes: np.ndarray = np.empty(0, dtype=np.int64)
        self._block_chunks: List[Tuple[np.ndarray, np.ndarray]] = []
        self.unique_runid_holder: np.ndarray = np.empty(0, dtype=np.int64)
        self.check_runid: bool = True

    def init(
        self,
        sqw_obj: Any = None,
        new_img: Any = None,
        ind_ranges: Optional[Union[np.ndarray, Sequence[Any]]] = None,
        *args: Any,
    ) -> "PageOp_section":
        """
        Initialize ``PageOp_section`` with the input ``sqw`` object, the sectioned
        ``DnDBase`` image ``new_img``, and the 1-based inclusive bin index ranges
        ``ind_ranges``.
        """
        if sqw_obj is None:
            return self
        super().init(sqw_obj)
        self._block_starts, self._block_sizes = self.find_section(
            self._img.npix, ind_ranges
        )
        self._block_chunks = [(self._block_starts, self._block_sizes)]
        self._img = new_img
        self.unique_runid_holder = np.empty(0, dtype=np.int64)
        self.check_runid = True
        return self

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: Union[int, float]
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, "PageOp_section"]:
        """
        Split ``_block_starts`` and ``_block_sizes`` into pages of maximum pixel count
        ``chunk_size``.
        """
        size_chunks, pos_chunks = split_data_blocks(
            self._block_starts, self._block_sizes, chunk_size
        )
        self._block_chunks = [
            (pos_ch, sz_ch) for pos_ch, sz_ch in zip(pos_chunks, size_chunks)
        ]
        npix_idx = np.ones((2, len(self._block_chunks)), dtype=np.int64)
        return self._block_chunks, npix_idx, self

    def get_page_data(
        self, idx: int = 1, npix_blocks: Optional[Sequence[Any]] = None
    ) -> "PageOp_section":
        """
        Load a page of sectioned pixel data for 1-based page index ``idx``.
        """
        self.page_num = idx
        if len(self._block_chunks) == 0:
            self._page_data = np.empty((9, 0), dtype=np.float64)
            return self
        bl_start, bl_size = self._block_chunks[idx - 1]
        ind = get_ind_from_ranges(bl_start, bl_size)
        self._page_data = np.asarray(
            self._pix.get_pixels(ind, "-raw", "-align"), dtype=np.float64
        )
        return self

    def apply_op(
        self,
        npix_block: Optional[Any] = None,
        npix_idx: Optional[np.ndarray] = None,
    ) -> "PageOp_section":
        """
        Selected pixel blocks are transferred to target without modifications.
        """
        return self

    def finish_op(self, out_obj: Any) -> Tuple[Any, "PageOp_section"]:
        """
        Finalize sectioning by pruning ``experiment_info`` if needed and transferring
        modifications to the target ``sqw`` object.
        """
        n_runs = int(getattr(out_obj.experiment_info, "n_runs", 0))
        if self._unique_run_id.size == n_runs:
            self.check_runid = False
        else:
            if self._unique_run_id.size < n_runs:
                out_obj = out_obj.copy()
                out_obj.experiment_info = out_obj.experiment_info.get_subobj(
                    self._unique_run_id
                )
            else:
                self.check_runid = False
        return super().finish_op(out_obj)

    # ------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------
    def get_exp_modified(self) -> bool:
        return bool(self._old_file_format or self.check_runid)

    def get_changes_pix_only(self) -> bool:
        return True

    def get_info_split_log_ratio(self) -> int:
        try:
            return int(
                config_store.instance().get_value("log_config", "section_split_ratio")
            )
        except Exception:
            return self._split_log_ratio

    def set_info_split_log_ratio(self, val: Union[int, float]) -> None:
        super().set_info_split_log_ratio(val)

    # ------------------------------------------------------------------
    # Static helper
    # ------------------------------------------------------------------
    @staticmethod
    def find_section(
        npix: Union[np.ndarray, Sequence[Any]],
        ind_range: Union[np.ndarray, Sequence[Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Using ``npix`` (which defines the number of pixels in each image cell)
        and ``ind_range`` (``(2, ND)`` 1-based inclusive bin index ranges),
        identify the 1-based starting pixel positions ``block_starts`` and sizes
        ``block_sizes`` of contiguous pixel ranges to keep.

        Ported from ``PageOp_section.find_section`` in ``PageOp_section.m``.
        """
        npix_arr = np.asarray(npix, dtype=np.int64)
        irange = np.asarray(ind_range, dtype=np.int64)
        if irange.ndim == 1:
            irange = irange.reshape(2, 1, order="F")

        n_dims = irange.shape[1]
        if (
            n_dims == 1
            and npix_arr.ndim == 2
            and (npix_arr.shape[0] == 1 or npix_arr.shape[1] == 1)
        ):
            npix_shaped = npix_arr.reshape(-1, order="F")
        elif npix_arr.ndim < n_dims:
            new_shape = list(npix_arr.shape) + [1] * (n_dims - npix_arr.ndim)
            npix_shaped = npix_arr.reshape(new_shape, order="F")
        elif npix_arr.ndim > n_dims:
            shape = list(npix_arr.shape)
            while len(shape) > n_dims and shape[-1] == 1:
                shape.pop()
            npix_shaped = npix_arr.reshape(shape, order="F")
        else:
            npix_shaped = npix_arr

        block_sizes_full = npix_shaped
        flat_sizes = block_sizes_full.ravel(order="F")
        if flat_sizes.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        block_ends_flat = np.cumsum(flat_sizes)
        pix_start_flat = np.concatenate(([0], block_ends_flat[:-1]))
        pix_start = pix_start_flat.reshape(block_sizes_full.shape, order="F")
        block_ends = block_ends_flat.reshape(block_sizes_full.shape, order="F")

        sel_slices = tuple(
            slice(int(irange[0, ii]) - 1, int(irange[1, ii])) for ii in range(n_dims)
        )
        sel_blocks = block_sizes_full[sel_slices]
        sel_pos = pix_start[sel_slices]
        sel_ends = block_ends[sel_slices]
        non_empty = sel_blocks.ravel(order="F") > 0

        block_starts = sel_pos.ravel(order="F")[non_empty]
        block_ends_sel = sel_ends.ravel(order="F")[non_empty]
        if block_starts.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Compress adjacent elements (following PageOp_section.m lines 118-126)
        edges = np.diff(block_starts) > 1
        start_mask = np.concatenate(([True], edges))
        end_mask = np.concatenate((edges, [True]))
        block_starts = block_starts[start_mask]
        block_ends_sel = block_ends_sel[end_mask]
        block_sizes = block_ends_sel - block_starts
        # MATLAB starts counting from 1
        block_starts = block_starts + 1
        return block_starts.astype(np.int64), block_sizes.astype(np.int64)
