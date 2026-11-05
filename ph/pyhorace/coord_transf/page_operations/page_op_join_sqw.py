"""PageOp_join_sqw: Single-page pixel operation for joining multiple SQW datasets.

Translated from:
    horace_core/sqw/page_operations/PageOp_join_sqw.m
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.configurations.log_config import log_config
from pyhorace.data.pixels.multipix_base import MultipixBase
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.data.pixels.pixfile_combine_info import pixfile_combine_info


def _fill_idx(bin_start: np.ndarray, page_bin_distr: np.ndarray) -> np.ndarray:
    """Find 0-based indices of sub-page pixels within a combined page with the same binning.

    Translated from ``fill_idx`` in ``PageOp_join_sqw.m`` (adapted to 0-based indexing).
    """
    distr = np.asarray(page_bin_distr, dtype=np.int64).ravel()
    starts = np.asarray(bin_start, dtype=np.int64).ravel()
    to_use = distr != 0
    distr = distr[to_use]
    starts = starts[to_use]

    total = int(np.sum(distr))
    if total == 0:
        return np.empty(0, dtype=np.int64)

    idx = np.zeros(total, dtype=np.int64)
    ic = 0
    for b_start, count in zip(starts, distr):
        c = int(count)
        bs = int(b_start) - 1  # 1-based bin_start -> 0-based index
        idx[ic : ic + c] = np.arange(bs, bs + c, dtype=np.int64)
        ic += c
    return idx


class PageOp_join_sqw(PageOpBase):
    """Single page pixel operation and main driver for ``sqw.join`` and ``write_nsqw_to_sqw``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_name_ = "join_sqw"
        self.split_at_bin_edges_ = True
        self.pix_combine_info: Optional[MultipixBase] = None
        self.new_runid: np.ndarray = np.empty(0, dtype=np.float64)
        self.use_mex: bool = False
        self.block_idx_: np.ndarray = np.empty((2, 0), dtype=np.int64)
        self.page_start_pos_: np.ndarray = np.empty(0, dtype=np.int64)
        self.npix_page_read_: np.ndarray = np.empty(0, dtype=np.int64)
        self.pixout_start_pos_: int = 0

    def init(
        self,
        in_sqw: Any = None,
        new_runid: Optional[Union[Sequence[Any], np.ndarray]] = None,
        use_mex: bool = False,
    ) -> Tuple["PageOp_join_sqw", Any]:
        """Initialize ``join_sqw`` algorithm on an ``sqw`` prepared by ``collect_sqw_metadata``."""
        if in_sqw is None:
            return self, None
        if not isinstance(getattr(in_sqw, "pix", None), MultipixBase):
            raise HoraceError(
                "HORACE:PageOp_join_sqw:invalid_argument",
                "Input sqw object does not contain information on how to combine input data",
            )
        self.pix_combine_info = in_sqw.pix

        hc = hor_config()
        mcs = int(hc.mem_chunk_size)
        fb = int(hc.fb_scale_factor)

        if (
            self.pix_combine_info.num_pixels > mcs * fb
            or bool(self.outfile)
            or isinstance(self.pix_combine_info, pixfile_combine_info)
        ):
            pix: PixelDataBase = PixelDataFileBacked()
            self.init_filebacked_output = True
        else:
            pix = PixelDataMemory()

        self.pix_ = pix
        if not self.outfile:
            self.outfile = self._build_file_name_for_join()

        self.use_mex = bool(
            use_mex and isinstance(self.pix_combine_info, pixfile_combine_info)
        )
        if self.use_mex:
            raise NotImplementedError(
                "# TODO(port-mex): Requires combine_files_using_mex"
            )

        pix.set_data_range(self.pix_combine_info.data_range)
        in_sqw.pix = pix

        super().init(in_sqw)

        if new_runid is None or np.size(new_runid) == 0:
            self.new_runid = np.empty(0, dtype=np.float64)
        else:
            self.new_runid = np.asarray(new_runid, dtype=np.float64).ravel()

        self.sig_acc_ = np.empty(0, dtype=np.float64)
        self.var_acc_ = np.empty(0, dtype=np.float64)

        if not self.use_mex:
            res = self.pix_combine_info.init_pix_access()
            if res is not None:
                self.pix_combine_info = res
            self.npix_page_read_ = np.zeros(
                self.pix_combine_info.nfiles, dtype=np.int64
            )
        else:
            self.pixout_start_pos_ = getattr(self.write_handle_, "pixout_start", 0)

        return self, in_sqw

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: int
    ) -> Tuple[List[np.ndarray], np.ndarray, "PageOp_join_sqw"]:
        """Split ``npix`` into large chunks at bin edges and record ``block_idx_``."""
        if self.use_mex:
            npix_flat = np.asarray(npix).ravel(order="F")
            npix_chunks = [npix_flat]
            npix_idx = np.array([[1], [npix_flat.size]], dtype=np.int64)
            return npix_chunks, npix_idx, self

        fb = int(hor_config().fb_scale_factor)
        large_chunk = int(chunk_size) * max(1, fb)
        npix_chunks, npix_idx, _ = super().split_into_pages(npix, large_chunk)
        self.block_idx_ = npix_idx
        page_sizes = np.array([int(np.sum(c)) for c in npix_chunks], dtype=np.int64)
        page_pos = np.cumsum(page_sizes)
        if page_pos.size > 0:
            self.page_start_pos_ = np.concatenate(
                ([1], page_pos[:-1] + 1)
            ).astype(np.int64)
        else:
            self.page_start_pos_ = np.array([1], dtype=np.int64)
        return npix_chunks, npix_idx, self

    def get_page_data(
        self, idx: int, npix_blocks: Sequence[np.ndarray]
    ) -> "PageOp_join_sqw":
        """Read data from multiple datasets and combine them into a single page of data."""
        self.page_num = idx
        if self.use_mex:
            return self

        block = np.asarray(npix_blocks[idx - 1], dtype=np.int64).ravel()
        bin_end = np.cumsum(block)
        page_size = int(bin_end[-1]) if bin_end.size > 0 else 0
        bin_start = (
            np.concatenate(([0], bin_end[:-1])) + 1
            if bin_end.size > 0
            else np.empty(0, dtype=np.int64)
        )

        page_data = np.zeros(
            (PixelDataBase.DEFAULT_NUM_PIX_FIELDS, page_size), dtype=np.float64
        )
        multi_data = self.pix_combine_info
        assert multi_data is not None
        n_datasets = multi_data.nfiles
        block_range = (
            self.block_idx_[:, idx - 1]
            if self.block_idx_.ndim == 2
            else self.block_idx_
        )

        for i in range(1, n_datasets + 1):
            contr_page_data, page_bin_distr = multi_data.get_dataset_page(
                i, int(self.npix_page_read_[i - 1]) + 1, block_range
            )
            n_page_pix = contr_page_data.shape[1]
            self.npix_page_read_[i - 1] += n_page_pix

            if self.new_runid.size > 0 and n_page_pix > 0:
                contr_page_data[self.run_idx_, :] = self.new_runid[i - 1]

            targ_bin_idx = _fill_idx(bin_start, page_bin_distr)
            if targ_bin_idx.size > 0:
                page_data[:, targ_bin_idx] = contr_page_data
            bin_start = bin_start + np.asarray(page_bin_distr, dtype=np.int64).ravel()

        self.page_data_ = page_data
        return self

    def apply_op(
        self, npix_block: Any = None, npix_idx: Any = None
    ) -> "PageOp_join_sqw":
        """Used only in MEX mode (no-op in Python mode since ``get_page_data`` combines the page)."""
        if self.use_mex:
            raise NotImplementedError(
                "# TODO(port-mex): Requires combine_files_using_mex"
            )
        return self

    def common_page_op(self) -> "PageOp_join_sqw":
        if self.use_mex:
            assert self.pix_combine_info is not None
            self.pix_data_range_ = self.pix_combine_info.data_range.copy()
            return self
        return super().common_page_op()

    def finish_op(self, in_obj: Any) -> Tuple[Any, "PageOp_join_sqw"]:
        assert self.pix_combine_info is not None
        extern_range = self.pix_combine_info.data_range
        if np.all(extern_range == PixelDataBase.EMPTY_RANGE):
            self.pix_data_range_ = extern_range.copy()

        res = self.pix_combine_info.close_faccessors()
        if res is not None:
            self.pix_combine_info = res

        out_obj, _ = super().finish_op(in_obj)
        return out_obj, self

    def get_page_num(self) -> int:
        if self.pix_combine_info is not None:
            return self.pix_combine_info.page_num
        return super().get_page_num()

    def set_page_num(self, val: int) -> PageOpBase:
        if self.pix_combine_info is not None:
            self.pix_combine_info.page_num = int(val)
            return self
        return super().set_page_num(val)

    def get_exp_modified(self) -> bool:
        return False

    def get_changes_pix_only(self) -> bool:
        return True

    def _build_file_name_for_join(self) -> str:
        wkdir = hor_config().working_directory or os.getcwd()
        assert self.pix_combine_info is not None
        first_in = (
            self.pix_combine_info.infiles[0]
            if len(self.pix_combine_info.infiles) > 0
            else ""
        )
        if isinstance(first_in, str):
            part_file = first_in
        else:
            part_file = getattr(first_in, "full_filename", "") or ""

        fb = os.path.splitext(os.path.basename(part_file))[0]
        parts = fb.split("_runID")
        fb = f"combined_{parts[0]}" if parts[0] else ""
        if not fb or fb == "combined_":
            fb = "combined_form_mem"

        if self.pix_ is not None and self.pix_.is_filebacked:
            return PixelDataBase.build_tmp_file_name(fb, wkdir)
        return os.path.join(wkdir, f"{fb}.sqw")

    def get_info_split_log_ratio(self) -> int:
        return int(
            config_store.instance().get_value("log_config", "join_sqw_split_ratio")
        )

    def set_info_split_log_ratio(self, val: Union[int, float]) -> PageOpBase:
        lc = log_config()
        lc.join_sqw_split_ratio = int(val)
        return self
