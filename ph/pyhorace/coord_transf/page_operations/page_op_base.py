"""
Abstract base class for page operations on SQW and PixelData objects.

Ported from:
- ``horace_core/sqw/page_operations/@PageOpBase/PageOpBase.m``
- ``horace_core/sqw/page_operations/@PageOpBase/private/init_.m``
- ``horace_core/sqw/page_operations/@PageOpBase/private/finish_op_.m``
- ``horace_core/sqw/page_operations/@PageOpBase/private/update_image_.m``
- ``horace_core/sqw/page_operations/@PageOpBase/private/print_range_warning_.m``
- ``horace_core/sqw/page_operations/@PageOpBase/private/report_on_target_files_.m``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import os
from typing import Any, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from pyhorace.coord_transf.page_operations._utils import (
    compute_bin_data,
    normalize_signal,
    split_vector_fixed_sum,
    split_vector_max_sum,
)
from pyhorace.core.configurations.config_store import config_store
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory


class PageOpBase(ABC):
    """
    Abstract base class defining the interface and common workflow for operations
    performed page-by-page over pixel data in ``sqw`` or ``PixelDataBase`` objects.
    """

    def __init__(self, *args: Any) -> None:
        # 0-based row indices into (9, N) pixel array
        self._coord_idx: List[int] = [0, 1, 2, 3]
        self._run_idx: int = 4
        self._signal_idx: int = 7
        self._var_idx: int = 8

        self._outfile: str = ""
        self._inplace: bool = False
        self._op_name: str = "undefined_operation"
        self._split_at_bin_edges: bool = False
        self._do_nopix: bool = False

        self._write_handle: Any = None
        self._pix: Any = PixelDataMemory()
        self._npix: Optional[np.ndarray] = None
        self._img: Any = None
        self._exp_info: Any = None

        self._pix_idx_start: int = 1
        self._pix_data_range: np.ndarray = PixelDataBase.EMPTY_RANGE.copy()
        self._unique_run_id: np.ndarray = np.empty(0, dtype=np.int64)
        self._sig_acc: np.ndarray = np.empty(0, dtype=np.float64)
        self._var_acc: np.ndarray = np.empty(0, dtype=np.float64)
        self._old_file_format: bool = False
        self._page_data: np.ndarray = np.empty((9, 0), dtype=np.float64)
        self._split_log_ratio: int = 1
        self._inform_about_target_file: bool = True
        self._init_filebacked_output: bool = False
        self._does_img_op: bool = False
        self._initialized: bool = False

        if len(args) > 0:
            self.init(*args)

    # =========================================================================
    # Abstract methods
    # =========================================================================

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> Union[PageOpBase, Tuple[PageOpBase, np.ndarray]]:
        """
        Apply the page operation to the current ``self._page_data`` chunk.
        """
        raise NotImplementedError

    # =========================================================================
    # Main operation methods
    # =========================================================================

    def init(self, in_obj: Any = None, *args: Any) -> PageOpBase:
        """
        Initialize page operation using parts of input ``sqw`` or ``PixelDataBase``
        object as the target for the operation.
        """
        if in_obj is None:
            return self
        self._init(in_obj)
        self._does_img_op = not self.changes_pix_only
        self._pix_idx_start = 1
        return self

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: Union[int, float]
    ) -> Tuple[List[np.ndarray], np.ndarray, PageOpBase]:
        """
        Split input ``npix`` array into pages of size ``chunk_size``.

        Parameters
        ----------
        npix : np.ndarray
            Image ``npix`` array defining pixel counts per bin.
        chunk_size : int or float
            Maximum pixel chunk size per page.

        Returns
        -------
        npix_chunks : list of np.ndarray
            List of 1D ``npix`` chunk arrays.
        npix_idx : np.ndarray
            ``(2, n_chunks)`` array of 1-based inclusive bin index ranges.
        self : PageOpBase
        """
        if self._split_at_bin_edges:
            npix_chunks, npix_idx = split_vector_max_sum(npix, chunk_size)
            chunk_sizes = [int(np.sum(ch)) for ch in npix_chunks]
            try:
                mchs = config_store.instance().get_value("hor_config", "mem_chunk_size")
                fb = config_store.instance().get_value("hor_config", "fb_scale_factor")
                mb_max = mchs * fb
            except Exception:
                mb_max = float("inf")
            if any(cs > mb_max for cs in chunk_sizes):
                warnings.warn(
                    f"\n*** The algorithm {self.op_name} requests input sqw object to be split on bin boundaries.\n"
                    f"*** Unfortunately input object contains bins that are so large,\n"
                    f"*** that even one bin may not fit to memory.\n"
                    f"*** This algorithm will try but probably fail processing such bins.\n"
                    f"*** Rebin input sqw object to smaller grid to be able to use this algorithm\n",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            npix_chunks, npix_idx = split_vector_fixed_sum(npix, chunk_size)
        return npix_chunks, npix_idx, self

    def get_page_data(
        self, idx: int, npix_blocks: Sequence[np.ndarray]
    ) -> PageOpBase:
        """
        Load and set into ``self._page_data`` the block of pixel data for page ``idx``
        (1-based).
        """
        self.page_num = idx
        if self._split_at_bin_edges:
            npix_block = npix_blocks[idx - 1]
            npix_in_block = int(np.sum(npix_block))
            pix_idx_end = self._pix_idx_start + npix_in_block - 1
            if npix_in_block == 0:
                self._page_data = np.empty((9, 0), dtype=np.float64)
            else:
                indices_1based = np.arange(
                    self._pix_idx_start, pix_idx_end + 1, dtype=np.int64
                )
                self._page_data = np.asarray(
                    self._pix.get_pixels(indices_1based, "-raw", "-align"),
                    dtype=np.float64,
                )
            self._pix_idx_start = pix_idx_end + 1
        else:
            raw = np.asarray(self._pix.data, dtype=np.float64)
            self._page_data = raw if self._inplace else raw.copy()
        return self

    def common_page_op(self) -> PageOpBase:
        """
        Common post-page operation executed after ``apply_op`` for each page:
        updates ``_pix_data_range``, ``_unique_run_id`` (if ``exp_modified``),
        and writes/stores the modified page data.
        """
        self._pix_data_range = PixelDataBase.pix_minmax_ranges(
            self._page_data, self._pix_data_range
        )
        if self.exp_modified and self._page_data.size > 0:
            page_runs = self._page_data[self._run_idx, :].astype(np.int64)
            if self._unique_run_id.size == 0:
                self._unique_run_id = np.unique(page_runs)
            else:
                self._unique_run_id = np.unique(
                    np.concatenate([self._unique_run_id, page_runs])
                )
        if not (self._inplace or self._do_nopix):
            self._pix = self._pix.store_page_data(self._page_data, self._write_handle)
        elif (
            self._inplace
            and not self._do_nopix
            and not getattr(self._pix, "is_filebacked", False)
        ):
            self._pix.set_raw_data(self._page_data)
        return self

    def update_img_accumulators(
        self,
        npix_block: np.ndarray,
        npix_idx: np.ndarray,
        signal: np.ndarray,
        variance: Optional[np.ndarray] = None,
    ) -> PageOpBase:
        """
        Accumulate modified pixel signal and variance into ``_sig_acc`` and ``_var_acc``.
        """
        has_var = variance is not None and np.asarray(variance).size > 0
        s_ar, e_ar = compute_bin_data(
            npix_block,
            signal,
            variance if has_var else None,
            no_normalization=True,
            calc_variance=has_var,
        )
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        if self._split_at_bin_edges:
            self._sig_acc[idx0:idx1] = s_ar.ravel(order="F")
            if has_var and e_ar is not None:
                self._var_acc[idx0:idx1] = e_ar.ravel(order="F")
        else:
            self._sig_acc[idx0:idx1] = self._sig_acc[idx0:idx1] + s_ar.ravel(order="F")
            if has_var and e_ar is not None:
                self._var_acc[idx0:idx1] = self._var_acc[idx0:idx1] + e_ar.ravel(
                    order="F"
                )
        return self

    def finish_op(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Finalize page operation: update image from accumulators and transfer
        modifications to the output object.
        """
        self.update_image(self._sig_acc, self._var_acc)
        return self.finish_core_op(in_obj)

    def finish_core_op(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Transfer modifications of new image and pixels to the target object
        without running ``update_image``.
        """
        return self._finish_op(in_obj)

    def print_range_warning(
        self, infile_name: str, is_old_file_format: bool
    ) -> None:
        """
        Print a warning informing the user that the source file contains an
        invalid data range and should be upgraded or finalized.
        """
        self._print_range_warning(infile_name, is_old_file_format)

    def report_on_target_files(self, output_obj: Any) -> None:
        """
        Print information about the target file(s) produced by a file-backed page operation.
        """
        self._report_on_target_files(output_obj)

    # =========================================================================
    # Properties & Getters / Setters
    # =========================================================================

    @property
    def changes_pix_only(self) -> bool:
        return self.get_changes_pix_only()

    @changes_pix_only.setter
    def changes_pix_only(self, val: bool) -> None:
        self.set_changes_pix_only(val)

    @property
    def outfile(self) -> str:
        return self._outfile

    @outfile.setter
    def outfile(self, val: Optional[str]) -> None:
        if val is None or val == "":
            self._outfile = ""
            return
        if not isinstance(val, (str, os.PathLike)):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: outfile type can only be str. "
                f"Provided: {type(val).__name__}"
            )
        self._outfile = str(val)

    @property
    def signal_idx(self) -> int:
        return self._signal_idx

    @property
    def var_idx(self) -> int:
        return self._var_idx

    @property
    def coord_idx(self) -> List[int]:
        return self._coord_idx

    @property
    def run_idx(self) -> int:
        return self._run_idx

    @property
    def inplace(self) -> bool:
        return self._inplace

    @inplace.setter
    def inplace(self, val: bool) -> None:
        self._inplace = bool(val)

    @property
    def split_log_ratio(self) -> int:
        return self.get_info_split_log_ratio()

    @split_log_ratio.setter
    def split_log_ratio(self, val: Union[int, float]) -> None:
        self.set_info_split_log_ratio(val)

    @property
    def exp_modified(self) -> bool:
        return self.get_exp_modified()

    @property
    def npix(self) -> Union[np.ndarray, int]:
        if self._npix is None or (isinstance(self._npix, np.ndarray) and self._npix.size == 0):
            return np.array([self._pix.num_pixels], dtype=np.int64)
        return self._npix

    @npix.setter
    def npix(self, val: Any) -> None:
        if val is None:
            self._npix = None
        else:
            self._npix = np.asarray(val).ravel(order="F")

    @property
    def page_num(self) -> int:
        return self.get_page_num()

    @page_num.setter
    def page_num(self, val: int) -> None:
        self.set_page_num(val)

    @property
    def pix(self) -> PixelDataBase:
        return self._pix

    @pix.setter
    def pix(self, val: PixelDataBase) -> None:
        if not isinstance(val, PixelDataBase):
            raise ValueError(
                "HORACE:PixelDataBase:invalid_argument: Pix can be an object of PixelDataBase class only"
            )
        self._pix = val

    @property
    def source_filename(self) -> str:
        full_fn = getattr(self._pix, "full_filename", "") or ""
        return os.path.basename(full_fn)

    @property
    def op_name(self) -> str:
        return self._op_name

    @op_name.setter
    def op_name(self, val: str) -> None:
        if not isinstance(val, str):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: op_name can be a text string only. "
                f"Provided {type(val).__name__}"
            )
        self._op_name = val

    @property
    def inform_about_target_file(self) -> bool:
        return self._inform_about_target_file

    @inform_about_target_file.setter
    def inform_about_target_file(self, val: bool) -> None:
        self._inform_about_target_file = bool(val)

    @property
    def split_at_bin_edges(self) -> bool:
        return self._split_at_bin_edges

    @split_at_bin_edges.setter
    def split_at_bin_edges(self, val: bool) -> None:
        self._split_at_bin_edges = bool(val)

    @property
    def is_range_valid(self) -> bool:
        return bool(self._pix.is_range_valid())

    @property
    def old_file_format(self) -> bool:
        return self._old_file_format

    @property
    def write_handle(self) -> Any:
        return self._write_handle

    @property
    def do_missing_range_warning(self) -> bool:
        return self.get_do_missing_range_warning()

    @property
    def page_data(self) -> np.ndarray:
        return self._page_data

    @page_data.setter
    def page_data(self, val: np.ndarray) -> None:
        self._page_data = np.asarray(val)

    @property
    def init_filebacked_output(self) -> bool:
        return self._init_filebacked_output

    @init_filebacked_output.setter
    def init_filebacked_output(self, val: bool) -> None:
        self._init_filebacked_output = bool(val)

    @property
    def do_nopix(self) -> bool:
        return self._do_nopix

    @do_nopix.setter
    def do_nopix(self, val: bool) -> None:
        self._do_nopix = bool(val)

    @property
    def img(self) -> Any:
        return self._img

    @img.setter
    def img(self, val: Any) -> None:
        self._img = val

    @property
    def pix_data_range(self) -> np.ndarray:
        return self._pix_data_range

    # =========================================================================
    # Protected Overloadable Methods
    # =========================================================================

    def get_page_num(self) -> int:
        return int(getattr(self._pix, "page_num", 1))

    def set_page_num(self, val: int) -> None:
        self._pix.page_num = val

    def get_exp_modified(self) -> bool:
        return self._old_file_format

    def get_changes_pix_only(self) -> bool:
        return self._img is None

    def set_changes_pix_only(self, *args: Any) -> None:
        pass

    def get_do_missing_range_warning(self) -> bool:
        return self._img is not None

    def update_image(
        self,
        sig_acc: np.ndarray,
        var_acc: Optional[np.ndarray] = None,
        npix_acc: Optional[np.ndarray] = None,
    ) -> PageOpBase:
        """
        Transfer accumulated signal and variance from ``sig_acc`` and ``var_acc``
        into ``self._img`` normalized by ``npix_acc``.
        """
        if self.changes_pix_only:
            return self
        if npix_acc is None:
            npix_acc = np.asarray(self.npix).ravel(order="F")
        return self._update_image(sig_acc, var_acc, npix_acc)

    def gen_old_file_message(self, infile_name: str) -> str:
        return (
            f"*** To upgrade original file run:\n"
            f"*** >> upgrade_file_format('{infile_name}',\"-upgrade_range\")\n"
        )

    def gen_misaligned_file_message(self, infile_name: str) -> str:
        return (
            f"*** To upgrade original file run:\n"
            f"*** >> fb_out = finalize_alignment('{infile_name}')\n"
        )

    def get_info_split_log_ratio(self) -> int:
        return self._split_log_ratio

    def set_info_split_log_ratio(self, val: Union[int, float]) -> None:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: log_split_ratio can have only numeric value. "
                f"Provided: {type(val).__name__}"
            )
        self._split_log_ratio = max(1, int(round(abs(float(val)))))

    # =========================================================================
    # Private Helpers (ported from @PageOpBase/private/)
    # =========================================================================

    def _init(self, in_obj: Any) -> PageOpBase:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/init_.m``.
        """
        from pyhorace.core.exceptions import HoraceError
        from pyhorace.sqw.sqw import sqw

        self._pix_data_range = PixelDataBase.EMPTY_RANGE.copy()

        if not (self._inplace or self._do_nopix):
            self._write_handle = in_obj.get_write_handle(self._outfile)
            if getattr(in_obj, "is_filebacked", False):
                self.init_filebacked_output = True

        if isinstance(in_obj, PixelDataBase):
            self._pix = (
                in_obj
                if (self._inplace or getattr(in_obj, "is_filebacked", False))
                else in_obj.copy()
            )
            self._img = None
            self._exp_info = None
            self._npix = np.array([in_obj.num_pixels], dtype=np.int64)
            if self._do_nopix:
                raise HoraceError(
                    "HORACE:PageOpBase:invalid_argument",
                    "-nopix option provided to pageOp but input contans pixels only. Can not continue.",
                )
        elif isinstance(in_obj, sqw):
            self._img = in_obj.data.copy() if hasattr(in_obj.data, "copy") else copy.deepcopy(in_obj.data)
            self._pix = (
                in_obj.pix
                if (self._inplace or getattr(in_obj.pix, "is_filebacked", False))
                else in_obj.pix.copy()
            )
            self._exp_info = in_obj.experiment_info
            self._npix = np.asarray(self._img.npix).ravel(order="F")
            self._sig_acc = np.zeros(self._npix.size, dtype=np.float64)
            self._var_acc = np.zeros(self._npix.size, dtype=np.float64)
        else:
            raise HoraceError(
                "HORACE:PageOpBase:invalid_argument",
                f"Init method accepts PixelData or SQW object input only. Provided {type(in_obj).__name__}",
            )

        self._pix.keep_precision = True
        self._old_file_format = bool(getattr(self._pix, "old_file_format", False))

        if self.exp_modified:
            self._unique_run_id = np.empty(0, dtype=np.int64)

        self._initialized = True
        return self

    def _update_image(
        self,
        sig_acc: np.ndarray,
        var_acc: Optional[np.ndarray],
        npix_acc: np.ndarray,
    ) -> PageOpBase:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/update_image_.m``.
        """
        if self.changes_pix_only and not self._do_nopix:
            return self

        s, e = normalize_signal(sig_acc, var_acc, npix_acc)
        img_shape = self._img.npix.shape
        self._img.do_check_combo_arg = False
        try:
            self._img.s = s.reshape(img_shape, order="F")
            self._img.npix = np.asarray(npix_acc, dtype=np.int64).reshape(img_shape, order="F")
            if e is not None:
                self._img.e = e.reshape(img_shape, order="F")
            else:
                self._img.e = np.zeros(img_shape, dtype=np.float64, order="F")
        finally:
            self._img.do_check_combo_arg = True
        self._img.check_combo_arg()
        return self

    def _finish_op(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/finish_op_.m``.
        """
        from pyhorace.sqw.sqw import sqw

        was_filebacked = bool(getattr(self._pix, "is_filebacked", False))
        if was_filebacked:
            orig_FName = getattr(self._pix, "full_filename", "") or ""
        else:
            orig_FName = ""
        source_range_valid = self.is_range_valid

        if isinstance(in_obj, sqw):
            out_obj = in_obj.copy()
            if self._img is not None:
                out_obj.data = self._img
            if self._do_nopix:
                out_obj = out_obj.data
                self._sig_acc = np.empty(0, dtype=np.float64)
                self._var_acc = np.empty(0, dtype=np.float64)
                self._initialized = False
                return out_obj, self

            if self.exp_modified and self._exp_info is not None:
                out_obj.experiment_info = out_obj.experiment_info.get_subobj(self._unique_run_id)

            from pyhorace.data.pixels.pixel_data_file_backed import (
                PixelDataFileBacked,
            )

            if isinstance(self._pix, PixelDataFileBacked) and not self._inplace:

                self._pix.keep_precision = False
                pix = PixelDataFileBacked()
                pix._num_pixels = self._pix.num_pixels
                pix._full_filename = self._pix.full_filename
                pix._base_page_size = self._pix._base_page_size
                pix._alignment_matr = self._pix._alignment_matr.copy()
                pix._is_corrected = self._pix._is_corrected
                pix._old_file_format = self._pix._old_file_format
                pix.set_data_range(self._pix_data_range)
            else:
                pix = self._pix
                pix.set_data_range(self._pix_data_range)
                pix.keep_precision = False
            if not (self._inplace or self._do_nopix):
                pix = pix.clear_alignment()
            out_obj.pix = pix
            if not self._inplace:
                out_obj = out_obj.finish_dump(self)
                self._report_on_target_files(out_obj)
        else:
            from pyhorace.data.pixels.pixel_data_file_backed import (
                PixelDataFileBacked,
            )

            if isinstance(self._pix, PixelDataFileBacked) and not self._inplace:

                self._pix.keep_precision = False
                pix = PixelDataFileBacked()
                pix._num_pixels = self._pix.num_pixels
                pix._full_filename = self._pix.full_filename
                pix._base_page_size = self._pix._base_page_size
                pix._alignment_matr = self._pix._alignment_matr.copy()
                pix._is_corrected = self._pix._is_corrected
                pix._old_file_format = self._pix._old_file_format
                pix.set_data_range(self._pix_data_range)
            else:
                pix = self._pix
                pix.set_data_range(self._pix_data_range)
                pix.keep_precision = False
            if self._inplace:
                out_obj = pix
            else:
                pix = pix.clear_alignment()
                out_obj = pix.finish_dump(self)
                self._report_on_target_files(out_obj)

        self._write_handle = None

        if was_filebacked and not source_range_valid and self.do_missing_range_warning:
            self._print_range_warning(orig_FName, self._old_file_format)

        self._sig_acc = np.empty(0, dtype=np.float64)
        self._var_acc = np.empty(0, dtype=np.float64)
        self._initialized = False
        return out_obj, self

    def _print_range_warning(
        self, infile_name: str, is_old_file_format: bool
    ) -> None:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/print_range_warning_.m``.
        """
        if is_old_file_format:
            reason = (
                "*** Legacy binary file has been upgraded to dynamic data structure, which requests knowledge of the pixel data range.\n"
                "*** Notice that operations with it were slow as full file scan was necessary to find its range.\n"
            )
            action = self.gen_old_file_message(infile_name)
            warn_id = "HORACE:old_file_format"
        else:
            reason = (
                "*** Source file supports dynamic data structure which was realigned without recalculating pixels data ranges.\n"
                "*** Notice that operations with this object were slow, as pixels alignment was performed at each file page access.\n"
            )
            action = self.gen_misaligned_file_message(infile_name)
            warn_id = "HORACE:invalid_data_range"

        warnings.warn(
            f"[{warn_id}]\n"
            f"*** Function '{self.op_name}' moved output data into pixel file with newly calculated range,\n"
            f"*** while input file: {infile_name} still does not contain precalculated full range values.\n"
            f"{reason}{action}",
            UserWarning,
            stacklevel=2,
        )

    def _report_on_target_files(self, output_obj: Any) -> None:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/report_on_target_files_.m``.
        """
        try:
            log_level = config_store.instance().get_value("hor_config", "log_level")
        except Exception:
            log_level = 1

        if (
            self._inform_about_target_file
            and getattr(output_obj, "is_filebacked", False)
            and log_level > 0
        ):
            print("********************************************************************************")
            wh_list = (
                self._write_handle
                if isinstance(self._write_handle, (list, tuple))
                else [self._write_handle]
            )
            out_list = (
                output_obj
                if isinstance(output_obj, (list, tuple))
                else [output_obj]
            )
            for idx, wh in enumerate(wh_list):
                if wh is None:
                    continue
                is_tmp = getattr(wh, "is_tmp_file", False)
                out_item = out_list[idx] if idx < len(out_list) else out_list[0]
                full_fn = getattr(out_item, "full_filename", "")
                if is_tmp:
                    print(
                        f"*** Operation {self.op_name} produced file-backed object grounded on temporary file:\n"
                        f"*** {full_fn}\n"
                        f"*** Use:\n"
                        f'*** >> save(out_obj,"new_permanent_file_name");\n'
                        f"*** command to save data file for permanent availability.                      ***"
                    )
                else:
                    print(
                        f"*** Operation {self.op_name} created constant output file:\n"
                        f"*** {full_fn}"
                    )
            print("********************************************************************************")
