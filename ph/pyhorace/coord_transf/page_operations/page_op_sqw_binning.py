"""Single pixel page operation used by ``sqw_op_bin_pixels`` algorithm.

Ported from:
- ``horace_core/sqw/page_operations/PageOp_sqw_binning.m``
- ``horace_core/algorithms/@cut_data_from_file_job/private/accumulate_pix_.m``
- ``horace_core/algorithms/@cut_data_from_file_job/private/put_sqw_data_npix_and_pix_to_file_.m``
- ``horace_core/utilities/sort_pix.m``
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import uuid

import numpy as np

from pyhorace.coord_transf.page_operations._utils import split_vector_fixed_sum
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.data.pixels.multipix_base import MultipixBase
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.data.pixels.pixfile_combine_info import pixfile_combine_info
from pyhorace.sqw.dnd_base import DnDBase


def _put_sqw_data_npix_and_pix_to_file(
    outfile: str, npix: np.ndarray, pix: PixelDataBase
) -> Tuple[int, int, int]:
    """Write ``npix`` and ``pix`` to a partial binary cut file.

    Ported from ``horace_core/algorithms/@cut_data_from_file_job/private/put_sqw_data_npix_and_pix_to_file_.m``.
    """
    if os.path.isfile(outfile):
        try:
            os.remove(outfile)
        except OSError:
            pass

    npix_flat = np.asarray(npix, dtype=np.int64).ravel(order="F")
    pix_width = np.array([PixelDataBase.DEFAULT_NUM_PIX_FIELDS * 4], dtype=np.int32)
    npixtot = int(pix.num_pixels)
    npixtot_arr = np.array([npixtot], dtype=np.int64)

    with open(outfile, "wb") as f:
        pos_npix = int(f.tell())
        npix_flat.tofile(f)
        pix_width.tofile(f)
        npixtot_arr.tofile(f)
        pos_pix = int(f.tell())
        if npixtot > 0:
            raw_single = np.asarray(pix.data, dtype=np.float32).ravel(order="F")
            raw_single.tofile(f)

    return pos_npix, pos_pix, npixtot


def _sort_pix(
    pix_retained: Sequence[Any],
    pix_ix_retained: Sequence[np.ndarray],
    npix: np.ndarray,
    data_range: Optional[np.ndarray] = None,
    keep_precision: bool = False,
) -> PixelDataMemory:
    """Sort accumulated pixel chunks by 1-based bin index ``pix_ix_retained``.

    Ported from ``horace_core/utilities/sort_pix.m``.
    """
    _ = (npix, keep_precision)
    non_empty_pix: List[np.ndarray] = []
    non_empty_ix: List[np.ndarray] = []

    for p, ix in zip(pix_retained, pix_ix_retained):
        if p is None:
            continue
        if isinstance(p, PixelDataBase):
            if p.num_pixels == 0:
                continue
            raw = np.asarray(p.data, dtype=np.float64)
        else:
            raw = np.asarray(p, dtype=np.float64)
            if raw.size == 0:
                continue
        non_empty_pix.append(raw)
        non_empty_ix.append(np.asarray(ix, dtype=np.int64).ravel())

    if not non_empty_pix:
        return PixelDataMemory()

    all_raw = np.hstack(non_empty_pix)
    all_ix = np.concatenate(non_empty_ix)
    if all_ix.size > 1:
        order = np.argsort(all_ix, kind="mergesort")
        all_raw = all_raw[:, order]

    out = PixelDataMemory(np.asarray(all_raw, dtype=np.float64))
    if data_range is not None and not np.any(data_range == PixelDataBase.EMPTY_RANGE):
        out.set_data_range(data_range)
    else:
        out.recalc_data_range("all")
    return out


_ACCUM_STATE: Dict[str, Any] = {
    "n_writ_files": 0,
    "npix_prev": None,
    "n_mem_blocks": 0,
    "n_pix_in_memory": 0,
    "pix_mem_retained": [],
    "pix_mem_ix_retained": [],
}


def _clear_accum_memory() -> None:
    _ACCUM_STATE["n_writ_files"] = 0
    _ACCUM_STATE["npix_prev"] = None
    _ACCUM_STATE["n_mem_blocks"] = 0
    _ACCUM_STATE["n_pix_in_memory"] = 0
    _ACCUM_STATE["pix_mem_retained"] = []
    _ACCUM_STATE["pix_mem_ix_retained"] = []


def _save_pixels_to_file(
    pix_comb_info: pixfile_combine_info,
    npix_distr_in_mem: np.ndarray,
    log_level: int = -1,
) -> pixfile_combine_info:
    if _ACCUM_STATE["n_mem_blocks"] == 0:
        return pix_comb_info

    if log_level > 1:
        print("*** Sorting selected pixels by the processed  block of image bins:")

    pix_2write = _sort_pix(
        _ACCUM_STATE["pix_mem_retained"],
        _ACCUM_STATE["pix_mem_ix_retained"],
        npix_distr_in_mem,
        pix_comb_info.data_range,
        True,
    )
    _ACCUM_STATE["n_mem_blocks"] = 0
    _ACCUM_STATE["pix_mem_retained"] = []
    _ACCUM_STATE["pix_mem_ix_retained"] = []

    _ACCUM_STATE["n_writ_files"] += 1
    idx = _ACCUM_STATE["n_writ_files"] - 1
    file_name = pix_comb_info.infiles[idx]
    if log_level > 0:
        print(
            f"*** Storing sorted pixels to partial sqw file for combining: {file_name}"
        )

    pos_npix, pos_pix, _ = _put_sqw_data_npix_and_pix_to_file(
        file_name, npix_distr_in_mem, pix_2write
    )

    nf = pix_comb_info.nfiles
    if pix_comb_info._pos_npixstart.size < nf:
        new_pos_npix = np.zeros(nf, dtype=np.uint64)
        new_pos_npix[: pix_comb_info._pos_npixstart.size] = pix_comb_info._pos_npixstart
        pix_comb_info._pos_npixstart = new_pos_npix
    if pix_comb_info._pos_pixstart.size < nf:
        new_pos_pix = np.zeros(nf, dtype=np.uint64)
        new_pos_pix[: pix_comb_info._pos_pixstart.size] = pix_comb_info._pos_pixstart
        pix_comb_info._pos_pixstart = new_pos_pix
    if pix_comb_info._npix_each_file.size < nf:
        new_npix_each = np.zeros(nf, dtype=np.uint64)
        new_npix_each[: pix_comb_info._npix_each_file.size] = (
            pix_comb_info._npix_each_file
        )
        pix_comb_info._npix_each_file = new_npix_each

    pix_comb_info._pos_npixstart[idx] = np.uint64(pos_npix)
    pix_comb_info._pos_pixstart[idx] = np.uint64(pos_pix)
    pix_comb_info._npix_each_file[idx] = np.uint64(_ACCUM_STATE["n_pix_in_memory"])
    pix_comb_info._num_pixels = int(np.sum(pix_comb_info._npix_each_file))
    _ACCUM_STATE["n_pix_in_memory"] = 0
    return pix_comb_info


def _accumulate_pix(
    pix_comb_info: Any,
    finish_accum: bool = False,
    v: Optional[PixelDataBase] = None,
    ix_add: Optional[np.ndarray] = None,
    npix: Optional[np.ndarray] = None,
    max_buf_size: int = 0,
    log_level: Optional[int] = None,
) -> Any:
    """Accumulate pixel data in memory and spill to partial binary files when ``max_buf_size`` is reached.

    Ported from ``horace_core/algorithms/@cut_data_from_file_job/private/accumulate_pix_.m``.
    """
    if isinstance(pix_comb_info, str) and pix_comb_info == "cleanup":
        _clear_accum_memory()
        return None

    if log_level is None:
        log_level = int(config_store.instance().get_value("hor_config", "log_level"))

    if _ACCUM_STATE["npix_prev"] is None:
        assert npix is not None
        _ACCUM_STATE["npix_prev"] = np.zeros_like(npix, dtype=np.float64)
        _ACCUM_STATE["n_pix_in_memory"] = int(v.num_pixels) if v is not None else 0
        _ACCUM_STATE["n_mem_blocks"] = 0
        _ACCUM_STATE["n_writ_files"] = 0
    else:
        if v is not None:
            _ACCUM_STATE["n_pix_in_memory"] += int(v.num_pixels)

    if finish_accum:
        assert npix is not None
        npix_distr_in_mem = (
            np.asarray(npix, dtype=np.float64) - _ACCUM_STATE["npix_prev"]
        )
        if _ACCUM_STATE["n_writ_files"] > 0:
            pix_comb_info = _save_pixels_to_file(
                pix_comb_info, npix_distr_in_mem, log_level
            )
            pix_comb_info = pix_comb_info.trim_nfiles(_ACCUM_STATE["n_writ_files"])
            pix_comb_info.npix_cumsum = np.cumsum(
                np.asarray(npix, dtype=np.uint64).ravel(order="F")
            )
            res: Any = pix_comb_info
        else:
            if _ACCUM_STATE["n_pix_in_memory"] == 0:
                res = PixelDataMemory()
            else:
                res = _sort_pix(
                    _ACCUM_STATE["pix_mem_retained"],
                    _ACCUM_STATE["pix_mem_ix_retained"],
                    npix_distr_in_mem,
                    pix_comb_info.data_range,
                    False,
                )
        _clear_accum_memory()
        return res

    if v is not None and v.num_pixels > 0:
        _ACCUM_STATE["n_mem_blocks"] += 1
        _ACCUM_STATE["pix_mem_retained"].append(v)
        _ACCUM_STATE["pix_mem_ix_retained"].append(
            np.asarray(ix_add, dtype=np.int64).ravel()
        )
        if not v.is_range_valid():
            v.recalc_data_range()
        dr = pix_comb_info.data_range.copy()
        dr[0, :] = np.minimum(dr[0, :], v.data_range[0, :])
        dr[1, :] = np.maximum(dr[1, :], v.data_range[1, :])
        pix_comb_info.set_data_range(dr)

    if _ACCUM_STATE["n_pix_in_memory"] >= max_buf_size:
        assert npix is not None
        npix_distr_in_mem = (
            np.asarray(npix, dtype=np.float64) - _ACCUM_STATE["npix_prev"]
        )
        _ACCUM_STATE["npix_prev"] = np.asarray(npix, dtype=np.float64).copy()
        pix_comb_info = _save_pixels_to_file(
            pix_comb_info, npix_distr_in_mem, log_level
        )

    return pix_comb_info


def _gen_unique_file_paths(
    n_files: int, prefix: str, base_dir: str = "", ext: str = "tmp"
) -> List[str]:
    """Generate ``n_files`` unique temporary file paths."""
    if not base_dir or not os.path.isdir(base_dir):
        base_dir = tempfile.gettempdir()
    uid = uuid.uuid4().hex[:12]
    return [
        os.path.join(base_dir, f"{prefix}_{uid}_{i:05d}.{ext}")
        for i in range(1, n_files + 1)
    ]


class PageOp_sqw_binning(PageOp_sqw_eval):
    """Single pixel page operation used by ``sqw_op_bin_pixels`` algorithm.

    Ported from ``horace_core/sqw/page_operations/PageOp_sqw_binning.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._op_name: str = "sqw_op_bin_pixels"
        self.split_at_bin_edges = False
        self._npix_acc: np.ndarray = np.empty(0, dtype=np.float64)
        self._pix_page: PixelDataMemory = PixelDataMemory()
        self._pix_combine_info: Optional[pixfile_combine_info] = None
        self._targ_axes: Any = None
        self._buf_size: int = 0

    def init(
        self,
        sqw_obj: Any = None,
        operation: Optional[Callable[..., Any]] = None,
        op_param: Any = None,
        targ_ax_block: Any = None,
        targ_proj: Any = None,
        pop_options: Any = None,
    ) -> "PageOp_sqw_binning":
        """Initialize ``PageOp_sqw_binning`` operation over input ``sqw`` object."""
        if sqw_obj is None:
            return self

        super().init(sqw_obj, operation, op_param, False)
        self.split_at_bin_edges = False

        nopix = False
        if pop_options is not None:
            if isinstance(pop_options, dict):
                nopix = bool(pop_options.get("nopix", False))
            else:
                nopix = bool(getattr(pop_options, "nopix", False))
        self.do_nopix = nopix

        if nopix and self.init_filebacked_output:
            self.init_filebacked_output = False
            if self._write_handle is not None:
                if hasattr(self._write_handle, "delete"):
                    self._write_handle.delete()
                self._write_handle = None

        self._targ_axes = targ_ax_block
        self.proj = targ_proj
        self._img = DnDBase.dnd(targ_ax_block, targ_proj)

        self._img.do_check_combo_arg = False
        sz = tuple(int(x) for x in targ_ax_block.dims_as_ssize)
        self._npix_acc = np.zeros(sz, dtype=np.float64, order="F")
        self._sig_acc = np.zeros(sz, dtype=np.float64, order="F")
        self._var_acc = np.zeros(sz, dtype=np.float64, order="F")

        self._pix_page = PixelDataMemory()
        hc = hor_config()
        self._buf_size = int(hc.mem_chunk_size) * int(hc.fb_scale_factor)
        return self

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: int
    ) -> Tuple[List[np.ndarray], np.ndarray, "PageOp_sqw_binning"]:
        """Split input ``npix`` array into pages of size ``chunk_size``."""
        if isinstance(self._pix, MultipixBase):
            npix_flat = np.asarray(npix, dtype=np.int64).ravel(order="F")
            n_blocks = int(npix_flat.size)
            self._pix_idx_start = np.ones(n_blocks, dtype=np.int64)
            npix_chunks: List[np.ndarray] = []
            idx_list: List[np.ndarray] = []
            for i in range(n_blocks):
                ch_i, id_i = split_vector_fixed_sum(
                    np.array([npix_flat[i]], dtype=np.int64), chunk_size
                )
                id_i = id_i + i
                npix_chunks.extend(ch_i)
                idx_list.append(id_i)
            npix_idx = (
                np.hstack(idx_list)
                if idx_list
                else np.zeros((2, 0), dtype=np.int64)
            )
            self._init_filebacked_output = True
        else:
            npix_chunks, npix_idx = split_vector_fixed_sum(npix, chunk_size)

        self._npix_idx = npix_idx

        if self._do_nopix or not self._init_filebacked_output:
            return npix_chunks, npix_idx, self

        _accumulate_pix("cleanup")

        try:
            wk_dir = config_store.instance().get_value(
                "parallel_config", "working_directory"
            )
        except Exception:
            wk_dir = ""
        n_files = len(npix_chunks)
        tmp_file_names = _gen_unique_file_paths(
            n_files, "page_op_bin_pixels_part", str(wk_dir or "")
        )
        nbins = int(self._npix_acc.size)
        self._pix_combine_info = pixfile_combine_info(tmp_file_names, nbins)

        if not self.outfile:
            if self._write_handle is None:
                return npix_chunks, npix_idx, self
            self._outfile = self._write_handle.write_file_name

        if self._write_handle is not None:
            if hasattr(self._write_handle, "delete"):
                self._write_handle.delete()
            self._write_handle = None

        return npix_chunks, npix_idx, self

    def get_page_data(
        self, idx: int, npix_blocks: Sequence[np.ndarray]
    ) -> "PageOp_sqw_binning":
        """Load block of pixel data for page ``idx`` (1-based)."""
        self.page_num = idx
        if isinstance(self._pix, MultipixBase):
            assert self._npix_idx is not None
            n_dataset = int(self._npix_idx[0, idx - 1])
            pix_ds = self._pix.infiles[n_dataset - 1]
            pix_idx_0 = int(np.asarray(self._pix_idx_start).ravel()[n_dataset - 1])
            block_sz = int(np.sum(npix_blocks[idx - 1]))
            pix_idx_1 = pix_idx_0 + block_sz - 1
            if block_sz == 0:
                self._page_data = np.empty((9, 0), dtype=np.float64)
            else:
                indices = np.arange(pix_idx_0, pix_idx_1 + 1, dtype=np.int64)
                self._page_data = np.asarray(
                    pix_ds.get_pixels(indices, "-raw", "-align"),
                    dtype=np.float64,
                )
            np.asarray(self._pix_idx_start).ravel()[n_dataset - 1] = pix_idx_1 + 1
        else:
            self._page_data = np.asarray(self._pix.data, dtype=np.float64).copy()
        return self

    def common_page_op(self) -> "PageOp_sqw_binning":
        """Update ``_pix_data_range`` and store page data if output is purely in-memory."""
        self._pix_data_range = PixelDataBase.pix_minmax_ranges(
            self._page_data, self._pix_data_range
        )
        if not (self._init_filebacked_output or self._do_nopix):
            self._pix = self._pix.store_page_data(self._page_data, self._write_handle)
        return self

    def apply_op(self, *args: Any) -> "PageOp_sqw_binning":
        """Apply user-defined operation over a page of pixels and bin into target axes."""
        _ = args
        if self.op_parms is None:
            parms: List[Any] = []
        elif isinstance(self.op_parms, (list, tuple)):
            parms = list(self.op_parms)
        else:
            parms = [self.op_parms]

        try:
            page_data = self.op_holder(self, *parms)
        except TypeError:
            if (
                len(parms) == 1
                and isinstance(parms[0], (list, tuple, np.ndarray))
                and len(parms[0]) == 0
            ):
                page_data = self.op_holder(self)
            else:
                raise

        if page_data is None or np.size(page_data) == 0:
            self._page_data = np.empty((9, 0), dtype=np.float64)
            return self

        self._page_data = np.asarray(page_data, dtype=np.float64)
        pix = self._pix_page
        pix.set_raw_data(self._page_data)

        if self._do_nopix:
            self._npix_acc, self._sig_acc, self._var_acc = self.proj.bin_pixels(
                self._img.axes,
                pix,
                self._npix_acc,
                self._sig_acc,
                self._var_acc,
                "-nomex",
                nargout=3,
            )
        else:
            filebacked_processing = bool(
                self._init_filebacked_output and not self._do_nopix
            )
            if filebacked_processing:
                (
                    self._npix_acc,
                    self._sig_acc,
                    self._var_acc,
                    pix_ok,
                    unique_runid_l,
                    pix_indx,
                ) = self.proj.bin_pixels(
                    self._targ_axes,
                    pix,
                    self._npix_acc,
                    self._sig_acc,
                    self._var_acc,
                    "-nomex",
                    nargout=6,
                )
            else:
                (
                    self._npix_acc,
                    self._sig_acc,
                    self._var_acc,
                    pix_ok,
                    unique_runid_l,
                ) = self.proj.bin_pixels(
                    self._targ_axes,
                    pix,
                    self._npix_acc,
                    self._sig_acc,
                    self._var_acc,
                    "-nomex",
                    nargout=5,
                )
            self._page_data = np.asarray(pix_ok.data, dtype=np.float64)
            if (
                self.exp_modified
                and unique_runid_l is not None
                and np.size(unique_runid_l) > 0
            ):
                u_l = np.asarray(unique_runid_l, dtype=np.int64).ravel()
                if self._unique_run_id.size == 0:
                    self._unique_run_id = np.unique(u_l)
                else:
                    self._unique_run_id = np.unique(
                        np.concatenate([self._unique_run_id, u_l])
                    )
            if filebacked_processing:
                assert self._pix_combine_info is not None
                self._pix_combine_info = _accumulate_pix(
                    self._pix_combine_info,
                    False,
                    pix_ok,
                    pix_indx,
                    self._npix_acc,
                    self._buf_size,
                )
        return self

    def finish_op(self, out_obj: Any) -> Tuple[Any, "PageOp_sqw_binning"]:
        """Finalize pixel binning, update target image, and combine temporary partial files if needed."""
        from pyhorace.coord_transf.page_operations.page_op_join_sqw import (
            PageOp_join_sqw,
        )
        from pyhorace.core.configurations.hpc_config import hpc_config
        from pyhorace.sqw.sqw import sqw

        self.update_image(self._sig_acc, self._var_acc, self._npix_acc)
        pix_cmbn_info: Optional[MultipixBase] = None
        if self._pix_combine_info is not None:
            self._pix = _accumulate_pix(
                self._pix_combine_info, True, None, None, self._npix_acc
            )
            if isinstance(self._pix, MultipixBase):
                pix_cmbn_info = self._pix
            self._init_filebacked_output = False

        out_obj, _ = self.finish_core_op(out_obj)
        if self.outfile:
            out_obj.full_filename = self.outfile

        if pix_cmbn_info is None:
            return out_obj, self
        if isinstance(out_obj.pix, PixelDataMemory):
            return out_obj, self

        hpc = hpc_config()
        hc = hor_config()
        use_mex = bool(
            hc.use_mex and getattr(hpc, "combine_sqw_using", "") == "mex_code"
        )

        page_op = PageOp_join_sqw()
        page_op.outfile = self.outfile
        page_op, out_obj = page_op.init(out_obj, [], use_mex)
        out_obj = sqw.apply_op(out_obj, page_op)
        _accumulate_pix("cleanup")
        pix_cmbn_info.clean_up_tmp_files()
        return out_obj, self

    def get_exp_modified(self) -> bool:
        """Binning may remove pixels so experiment_info consistency must be checked."""
        return True

    def get_do_missing_range_warning(self) -> bool:
        """Suppress missing range warning since binning creates a new target dataset."""
        return False

    def set_page_num(self, val: int) -> PageOpBase:
        if isinstance(self._pix, MultipixBase):
            self._pix._page_num = int(val)
            return self
        return super().set_page_num(val)


__all__ = ["PageOp_sqw_binning"]

