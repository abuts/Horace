"""PageOp_split_sqw — Split an sqw object into individual contributing run sqw objects."""

from __future__ import annotations

import copy
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pyhorace.coord_transf.page_operations._utils import (
    compute_bin_data,
    normalize_signal,
    split_vector_max_sum_or_numel,
)
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.configurations.hpc_config import hpc_config
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

if TYPE_CHECKING:
    from pyhorace.sqw.sqw import sqw


class PageOp_split_sqw(PageOpBase):
    """Split an sqw object into a collection of sqw objects, each made from a single contributing run.

    Equivalent to MATLAB ``horace_core/sqw/page_operations/PageOp_split_sqw.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_name_ = "split_sqw"
        self.split_at_bin_edges = False
        self.write_handles: List[Any] = []
        self.out_img: List[Dict[str, np.ndarray]] = []
        self.out_pix: List[Any] = []
        self.out_sqw: List[Any] = []
        self.obj_pix_ranges: List[np.ndarray] = []

        self.runid_map_: Any = None
        self.n_obj_contrib_to_page_: np.ndarray = np.empty(0, dtype=np.int64)
        self.targ_files_map_: Dict[int, str] = {}
        self.run_contributes_: np.ndarray = np.empty(0, dtype=bool)
        self.img_filebacked_: bool = False
        self.results_are_tmp_files_: bool = True
        self.img_size_: int = 0
        self.npix_start_pos_: int = 0

    @property
    def img_filebacked(self) -> bool:
        """True if image changes are accumulated on disk rather than in memory."""
        return self.img_filebacked_

    @img_filebacked.setter
    def img_filebacked(self, val: bool) -> None:
        self.img_filebacked_ = bool(val)

    @property
    def n_outputs(self) -> int:
        """Total number of output objects produced by split."""
        return len(self.out_sqw)

    @property
    def run_contributes(self) -> np.ndarray:
        """Logical array containing True for runs that contributed pixels to the split."""
        return self.run_contributes_

    @property
    def targ_files_map(self) -> Dict[int, str]:
        """Map linking run_id of a contributing run to its target filename."""
        return self.targ_files_map_

    @property
    def results_are_tmp_files(self) -> bool:
        """True if output files are temporary files."""
        return self.results_are_tmp_files_

    def init(  # type: ignore[override]
        self,
        in_sqw: "sqw",
        pix_filebacked: bool = False,
        img_filebacked: bool = False,
    ) -> "PageOp_split_sqw":
        """Initialize the PageOp_split_sqw operation."""
        self.inplace = True
        super().init(in_sqw)
        # Clear single write_handle_ and img_ used by base class
        self.write_handle_ = None
        self.img_ = None
        self.img_filebacked_ = bool(img_filebacked)

        self.prepare_split_sqw(in_sqw, bool(pix_filebacked), bool(img_filebacked))

        self.obj_pix_ranges = [
            PixelDataBase.EMPTY_RANGE.copy() for _ in range(len(self.out_img))
        ]
        if img_filebacked:
            self.split_at_bin_edges = True
        return self

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: Union[int, float]
    ) -> Tuple[List[np.ndarray], np.ndarray, "PageOp_split_sqw"]:
        """Divide npix array into pages respecting both pixel chunk_size and image memory size."""
        npix_chunks, npix_idx, _ = super().split_into_pages(npix, chunk_size)
        if not self.img_filebacked_:
            return npix_chunks, npix_idx, self

        n_chunks = max(len(self.out_sqw), 1)
        n_elements_in_chunks = np.array([int(x.size) for x in npix_chunks], dtype=np.int64)
        chunk_byte_sizes = (3 * 8 * n_chunks) * n_elements_in_chunks
        mem_avail = float(hpc_config().phys_mem_available)
        if np.all(chunk_byte_sizes < mem_avail):
            return npix_chunks, npix_idx, self

        n_elements_fit_memory = int(np.floor(mem_avail / (3 * 8 * n_chunks)))
        if n_elements_fit_memory < 1:
            n_elements_fit_memory = 1
            warnings.warn(
                "split_sqw algorithm suggests page operation operating single image bin at a time.",
                RuntimeWarning,
                stacklevel=2,
            )
        npix_chunks, npix_idx = split_vector_max_sum_or_numel(
            npix, int(chunk_size), n_elements_fit_memory
        )
        return npix_chunks, npix_idx, self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_split_sqw":
        """Split a page of pixel data into per-run pages and accumulate per-run images/pixels."""
        data = self.page_data_
        signal = data[self.signal_idx_, :]
        variance = data[self.var_idx_, :]
        run_id = data[self.run_idx_, :]

        n_bins = int(npix_block.size)
        pix_bin_idx = np.repeat(
            np.arange(1, n_bins + 1, dtype=np.int64),
            np.asarray(npix_block, dtype=np.int64).ravel(order="F"),
        )

        contrib_ind, unique_run_id = self.get_run_indices(run_id)
        self.n_obj_contrib_to_page_ = contrib_ind
        if self.img_filebacked_:
            self.npix_start_pos_ = int(npix_idx[0]) - 1

        for id_pos, idx_zero in zip(unique_run_id, contrib_ind):
            self.run_contributes_[idx_zero] = True
            my_pix_mask = run_id.ravel(order="F") == id_pos

            my_data = data[:, my_pix_mask]
            my_bins = pix_bin_idx[my_pix_mask]
            my_npix = np.bincount(my_bins, minlength=n_bins + 1)[1:].astype(np.float64)

            my_sig = signal.ravel(order="F")[my_pix_mask]
            my_var = variance.ravel(order="F")[my_pix_mask]

            if self.img_filebacked_:
                s_ar, e_ar = compute_bin_data(
                    my_npix, my_sig, my_var
                )
                self.out_img[idx_zero]["s"] = s_ar.ravel(order="F")
                self.out_img[idx_zero]["e"] = (
                    e_ar.ravel(order="F")
                    if e_ar is not None
                    else np.zeros_like(my_npix)
                )
                self.out_img[idx_zero]["npix"] = my_npix.ravel(order="F")
            else:
                s_ar, e_ar = compute_bin_data(
                    my_npix, my_sig, my_var, no_normalization=True
                )
                idx0 = int(npix_idx[0]) - 1
                idx1 = int(npix_idx[1])
                self.out_img[idx_zero]["s"][idx0:idx1] += s_ar.ravel(order="F")
                if e_ar is not None:
                    self.out_img[idx_zero]["e"][idx0:idx1] += e_ar.ravel(order="F")
                self.out_img[idx_zero]["npix"][idx0:idx1] += my_npix.ravel(order="F")

            self.out_pix[idx_zero] = my_data

        return self

    def common_page_op(self) -> "PageOp_split_sqw":
        """Store per-run split pixel data and update pixel coordinate ranges."""
        for idx_zero in self.n_obj_contrib_to_page_:
            pix_obj_data = self.out_pix[idx_zero]
            self.obj_pix_ranges[idx_zero] = PixelDataBase.pix_minmax_ranges(
                pix_obj_data, self.obj_pix_ranges[idx_zero]
            )
            self.out_sqw[idx_zero].pix = self.out_sqw[idx_zero].pix.store_page_data(
                pix_obj_data, self.write_handles[idx_zero]
            )
            if self.img_filebacked_:
                self.write_handles[idx_zero].save_img_chunk(
                    self.out_img[idx_zero], self.npix_start_pos_
                )
        return self

    def get_run_indices(
        self, run_idx: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert pixel run_id values into 0-based indices into experiment_info."""
        unique_run_id = np.unique(run_idx.ravel(order="F")).astype(int)
        contrib_ind_list: List[int] = []
        for rid in unique_run_id:
            try:
                val = self.runid_map_.get(int(rid))
            except Exception as exc:
                raise RuntimeError(
                    f"HORACE:split:runtime_error: Pixel runs indices bypass experimental info indices: {rid}"
                ) from exc
            # MATLAB runid_map stores 1-based experiment_info indices
            contrib_ind_list.append(int(val) - 1)
        return np.array(contrib_ind_list, dtype=np.int64), unique_run_id

    def get_exp_modified(self) -> bool:
        """Experiment info is pre-split per run in prepare_split_sqw, so return False."""
        return False

    def finish_op(  # type: ignore[override]
        self, *args: Any
    ) -> Tuple[Union[DataOpArray, List[str]], "PageOp_split_sqw"]:
        """Finalize split operation and return array of split sqw objects (or list of filenames)."""
        # Switch off inplace flag so finish_dump runs on each output sqw object
        self.inplace_ = False

        # Clean up unused write handles (where npix_written == 0)
        for i, wh in enumerate(self.write_handles):
            if wh is not None and wh.npix_written == 0:
                wh.delete()
                self.write_handles[i] = None

        # Filter out runs that did not contribute any pixels
        contrib_mask = self.run_contributes_
        self.out_img = [
            img for img, flag in zip(self.out_img, contrib_mask) if flag
        ]
        self.out_sqw = [
            obj for obj, flag in zip(self.out_sqw, contrib_mask) if flag
        ]
        self.obj_pix_ranges = [
            rng for rng, flag in zip(self.obj_pix_ranges, contrib_mask) if flag
        ]
        self.write_handles = [
            wh for wh, flag in zip(self.write_handles, contrib_mask) if flag
        ]

        n_obj = len(self.out_img)
        if self.img_filebacked_:
            out_files: List[str] = []
            for i in range(n_obj):
                self.npix = self.out_img[i]["npix"]
                self.sig_acc_ = self.out_img[i]["s"]
                self.var_acc_ = self.out_img[i]["e"]
                self.write_handle_ = self.write_handles[i]
                self.pix_ = self.out_sqw[i].pix
                self.pix_data_range_ = self.obj_pix_ranges[i]
                out_file_i, self = self.finalize_fb_obj(self.out_sqw[i])
                out_files.append(out_file_i)
            self.out_img = []
            self.out_sqw = []
            return out_files, self
        else:
            out_objs: List[Any] = []
            for i in range(n_obj):
                self.npix = self.out_img[i]["npix"]
                self.sig_acc_ = self.out_img[i]["s"]
                self.var_acc_ = self.out_img[i]["e"]
                self.img_ = self.out_sqw[i].data
                self.write_handle_ = self.write_handles[i]
                self.pix_ = self.out_sqw[i].pix
                self.pix_data_range_ = self.obj_pix_ranges[i]
                run_id = int(self.out_sqw[i].experiment_info.expdata[0].run_id)
                self.outfile = self.targ_files_map_[run_id]
                out_obj_i, self = super().finish_op(self.out_sqw[i])
                out_objs.append(out_obj_i)
            self.out_img = []
            self.out_sqw = []
            return DataOpArray(out_objs), self

    def finalize_fb_obj(
        self, sqw_obj: "sqw"
    ) -> Tuple[str, "PageOp_split_sqw"]:
        """Finalize file-backed split object on disk and return its filename."""
        pix = self.pix_
        pix.set_data_range(self.pix_data_range_)
        out_split_sqw = sqw_obj.copy()
        out_split_sqw.pix = pix
        self.img_ = None
        self.outfile = ""
        self.write_handle_.is_tmp_file = False
        out_split_sqw = out_split_sqw.finish_dump(self)
        out_file = out_split_sqw.full_filename

        self.pix_ = PixelDataMemory()
        self.npix_ = np.empty(0, dtype=np.float64)
        self.write_handle_ = None
        return out_file, self

    def report_on_target_files(self, out_obj: Any) -> None:
        """Print target file information when hor_config.log_level > 0."""
        if not self.img_filebacked_ and not self.write_to_file:
            return
        print("--------------------------------------------------------------------------------")
        print("*** Split result goes into target files:")
        if isinstance(out_obj, (list, tuple)) and all(isinstance(x, str) for x in out_obj):
            for fname in out_obj:
                print(f"*** : {fname}")
        else:
            items = list(out_obj) if isinstance(out_obj, (list, tuple, DataOpArray)) else [out_obj]
            for item in items:
                print(f"*** : {item.full_filename}")
        print("--------------------------------------------------------------------------------")

    def prepare_split_sqw(
        self,
        in_sqw: "sqw",
        pix_filebacked: bool,
        img_filebacked: bool,
    ) -> "PageOp_split_sqw":
        """Prepare output sqw templates, image accumulators, and write handles."""
        from pyhorace.sqw.sqw import sqw as SQWClass

        self.runid_map_ = in_sqw.runid_map
        self.img_size_ = in_sqw.img_size_bytes
        n_bins = int(in_sqw.data.s.size)

        n_runs = int(in_sqw.main_header.nfiles)
        self.run_contributes_ = np.zeros(n_runs, dtype=bool)
        if img_filebacked:
            self.init_filebacked_output = False

        exp_info = in_sqw.experiment_info
        file_in = os.path.splitext(os.path.basename(in_sqw.data.filename))[0]
        self.gen_target_filenames(file_in, pix_filebacked)

        self.write_handles = [None] * n_runs
        self.out_img = [{} for _ in range(n_runs)]
        self.out_pix = [None] * n_runs
        self.out_sqw = [None] * n_runs

        zero_img = in_sqw.data.copy()
        zero_img.s = np.zeros_like(zero_img.s)
        zero_img.e = np.zeros_like(zero_img.e)
        zero_img.npix = np.zeros_like(zero_img.npix)

        for i in range(n_runs):
            obj_i = SQWClass()
            obj_i._main_header = copy.deepcopy(in_sqw._main_header)
            obj_i._main_header.nfiles = 1
            obj_i._data = zero_img.copy()
            obj_i.experiment_info = exp_info.get_subobj(i + 1, "-indexes")
            run_id = int(obj_i.experiment_info.expdata[0].run_id)
            targ_file = self.targ_files_map_[run_id]

            if pix_filebacked:
                obj_i.pix = PixelDataFileBacked()
            else:
                obj_i.pix = PixelDataMemory()
            obj_i.full_filename = targ_file

            self.out_sqw[i] = obj_i
            if img_filebacked:
                self.out_img[i] = {
                    "s": np.empty(0, dtype=np.float64),
                    "e": np.empty(0, dtype=np.float64),
                    "npix": np.empty(0, dtype=np.float64),
                }
            else:
                self.out_img[i] = {
                    "s": np.zeros(n_bins, dtype=np.float64),
                    "e": np.zeros(n_bins, dtype=np.float64),
                    "npix": np.zeros(n_bins, dtype=np.float64),
                }

            wh = obj_i.get_write_handle(targ_file, True)
            if wh is not None:
                wh.is_tmp_file = self.results_are_tmp_files_
            self.write_handles[i] = wh
        return self

    def gen_target_filenames(
        self, file_in: str, pix_filebacked: bool
    ) -> "PageOp_split_sqw":
        """Generate target filenames for each run_id in runid_map_."""
        self.results_are_tmp_files_ = True
        if self.outfile:
            self.results_are_tmp_files_ = False
        else:
            self.outfile = str(hor_config().working_directory)
        if self.img_filebacked_ and pix_filebacked:
            self.results_are_tmp_files_ = False

        if callable(getattr(self.runid_map_, "keys", None)):
            run_ids = [int(k) for k in self.runid_map_.keys()]
        else:
            run_ids = [int(k) for k in self.runid_map_.keys]

        self.targ_files_map_ = {}
        for rid in run_ids:
            filebase = f"{file_in}_runID{rid:07d}"
            if self.results_are_tmp_files_:
                split_filename = PixelDataBase.build_tmp_file_name(
                    filebase, self.outfile
                )
            else:
                split_filename = os.path.join(self.outfile, f"{filebase}.sqw")
            self.targ_files_map_[rid] = split_filename
        return self

    @property
    def op_name(self) -> str:
        """Operation name used in progress reporting."""
        return "splitting sqw object into parts"
