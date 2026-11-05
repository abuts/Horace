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
        self._op_name = "split_sqw"
        self.split_at_bin_edges = False
        self.write_handles: List[Any] = []
        self.out_img: List[Dict[str, np.ndarray]] = []
        self.out_pix: List[Any] = []
        self.out_sqw: List[Any] = []
        self.obj_pix_ranges: List[np.ndarray] = []

        self._runid_map: Any = None
        self._n_obj_contrib_to_page: np.ndarray = np.empty(0, dtype=np.int64)
        self._targ_files_map: Dict[int, str] = {}
        self._run_contributes: np.ndarray = np.empty(0, dtype=bool)
        self._img_filebacked: bool = False
        self._results_are_tmp_files: bool = True
        self._img_size: int = 0
        self._npix_start_pos: int = 0

    @property
    def img_filebacked(self) -> bool:
        """True if image changes are accumulated on disk rather than in memory."""
        return self._img_filebacked

    @img_filebacked.setter
    def img_filebacked(self, val: bool) -> None:
        self._img_filebacked = bool(val)

    @property
    def n_outputs(self) -> int:
        """Total number of output objects produced by split."""
        return len(self.out_sqw)

    @property
    def run_contributes(self) -> np.ndarray:
        """Logical array containing True for runs that contributed pixels to the split."""
        return self._run_contributes

    @property
    def targ_files_map(self) -> Dict[int, str]:
        """Map linking run_id of a contributing run to its target filename."""
        return self._targ_files_map

    @property
    def results_are_tmp_files(self) -> bool:
        """True if output files are temporary files."""
        return self._results_are_tmp_files

    def init(  # type: ignore[override]
        self,
        in_sqw: "sqw",
        pix_filebacked: bool = False,
        img_filebacked: bool = False,
    ) -> "PageOp_split_sqw":
        """Initialize the PageOp_split_sqw operation."""
        self.inplace = True
        super().init(in_sqw)
        # Clear single _write_handle and _img used by base class
        self._write_handle = None
        self._img = None
        self._img_filebacked = bool(img_filebacked)

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
        if not self._img_filebacked:
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
        data = self._page_data
        signal = data[self._signal_idx, :]
        variance = data[self._var_idx, :]
        run_id = data[self._run_idx, :]

        n_bins = int(npix_block.size)
        pix_bin_idx = np.repeat(
            np.arange(1, n_bins + 1, dtype=np.int64),
            np.asarray(npix_block, dtype=np.int64).ravel(order="F"),
        )

        contrib_ind, unique_run_id = self.get_run_indices(run_id)
        self._n_obj_contrib_to_page = contrib_ind
        if self._img_filebacked:
            self._npix_start_pos = int(npix_idx[0]) - 1

        for id_pos, idx_zero in zip(unique_run_id, contrib_ind):
            self._run_contributes[idx_zero] = True
            my_pix_mask = run_id.ravel(order="F") == id_pos

            my_data = data[:, my_pix_mask]
            my_bins = pix_bin_idx[my_pix_mask]
            my_npix = np.bincount(my_bins, minlength=n_bins + 1)[1:].astype(np.float64)

            my_sig = signal.ravel(order="F")[my_pix_mask]
            my_var = variance.ravel(order="F")[my_pix_mask]

            if self._img_filebacked:
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
        for idx_zero in self._n_obj_contrib_to_page:
            pix_obj_data = self.out_pix[idx_zero]
            self.obj_pix_ranges[idx_zero] = PixelDataBase.pix_minmax_ranges(
                pix_obj_data, self.obj_pix_ranges[idx_zero]
            )
            self.out_sqw[idx_zero].pix = self.out_sqw[idx_zero].pix.store_page_data(
                pix_obj_data, self.write_handles[idx_zero]
            )
            if self._img_filebacked:
                self.write_handles[idx_zero].save_img_chunk(
                    self.out_img[idx_zero], self._npix_start_pos
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
                val = self._runid_map.get(int(rid))
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
        self._inplace = False

        # Clean up unused write handles (where npix_written == 0)
        for i, wh in enumerate(self.write_handles):
            if wh is not None and wh.npix_written == 0:
                wh.delete()
                self.write_handles[i] = None

        # Filter out runs that did not contribute any pixels
        contrib_mask = self._run_contributes
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
        if self._img_filebacked:
            out_files: List[str] = []
            for i in range(n_obj):
                self.npix = self.out_img[i]["npix"]
                self._sig_acc = self.out_img[i]["s"]
                self._var_acc = self.out_img[i]["e"]
                self._write_handle = self.write_handles[i]
                self._pix = self.out_sqw[i].pix
                self._pix_data_range = self.obj_pix_ranges[i]
                out_file_i, self = self.finalize_fb_obj(self.out_sqw[i])
                out_files.append(out_file_i)
            self.out_img = []
            self.out_sqw = []
            return out_files, self
        else:
            out_objs: List[Any] = []
            for i in range(n_obj):
                self.npix = self.out_img[i]["npix"]
                self._sig_acc = self.out_img[i]["s"]
                self._var_acc = self.out_img[i]["e"]
                self._img = self.out_sqw[i].data
                self._write_handle = self.write_handles[i]
                self._pix = self.out_sqw[i].pix
                self._pix_data_range = self.obj_pix_ranges[i]
                run_id = int(self.out_sqw[i].experiment_info.expdata[0].run_id)
                self.outfile = self._targ_files_map[run_id]
                out_obj_i, self = super().finish_op(self.out_sqw[i])
                out_objs.append(out_obj_i)
            self.out_img = []
            self.out_sqw = []
            return DataOpArray(out_objs), self

    def finalize_fb_obj(
        self, sqw_obj: "sqw"
    ) -> Tuple[str, "PageOp_split_sqw"]:
        """Finalize file-backed split object on disk and return its filename."""
        pix = self._pix
        pix.set_data_range(self._pix_data_range)
        out_split_sqw = sqw_obj.copy()
        out_split_sqw.pix = pix
        self._img = None
        self.outfile = ""
        self._write_handle.is_tmp_file = False
        out_split_sqw = out_split_sqw.finish_dump(self)
        out_file = out_split_sqw.full_filename

        self._pix = PixelDataMemory()
        self._npix = np.empty(0, dtype=np.float64)
        self._write_handle = None
        return out_file, self

    def report_on_target_files(self, out_obj: Any) -> None:
        """Print target file information when hor_config.log_level > 0."""
        if not self._img_filebacked and not self.write_to_file:
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

        self._runid_map = in_sqw.runid_map
        self._img_size = in_sqw.img_size_bytes
        n_bins = int(in_sqw.data.s.size)

        n_runs = int(in_sqw.main_header.nfiles)
        self._run_contributes = np.zeros(n_runs, dtype=bool)
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
            targ_file = self._targ_files_map[run_id]

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
                wh.is_tmp_file = self._results_are_tmp_files
            self.write_handles[i] = wh
        return self

    def gen_target_filenames(
        self, file_in: str, pix_filebacked: bool
    ) -> "PageOp_split_sqw":
        """Generate target filenames for each run_id in _runid_map."""
        self._results_are_tmp_files = True
        if self.outfile:
            self._results_are_tmp_files = False
        else:
            self.outfile = str(hor_config().working_directory)
        if self._img_filebacked and pix_filebacked:
            self._results_are_tmp_files = False

        if callable(getattr(self._runid_map, "keys", None)):
            run_ids = [int(k) for k in self._runid_map.keys()]
        else:
            run_ids = [int(k) for k in self._runid_map.keys]

        self._targ_files_map = {}
        for rid in run_ids:
            filebase = f"{file_in}_runID{rid:07d}"
            if self._results_are_tmp_files:
                split_filename = PixelDataBase.build_tmp_file_name(
                    filebase, self.outfile
                )
            else:
                split_filename = os.path.join(self.outfile, f"{filebase}.sqw")
            self._targ_files_map[rid] = split_filename
        return self

    @property
    def op_name(self) -> str:
        """Operation name used in progress reporting."""
        return "splitting sqw object into parts"
