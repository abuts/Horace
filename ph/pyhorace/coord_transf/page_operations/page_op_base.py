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
        self.coord_idx_: List[int] = [0, 1, 2, 3]
        self.run_idx_: int = 4
        self.signal_idx_: int = 7
        self.var_idx_: int = 8

        self.outfile_: str = ""
        self.inplace_: bool = False
        self.op_name_: str = "undefined_operation"
        self.split_at_bin_edges_: bool = False
        self.do_nopix_: bool = False

        self.write_handle_: Any = None
        self.pix_: Any = PixelDataMemory()
        self.npix_: Optional[np.ndarray] = None
        self.img_: Any = None
        self.exp_info_: Any = None

        self.pix_idx_start_: int = 1
        self.pix_data_range_: np.ndarray = PixelDataBase.EMPTY_RANGE.copy()
        self.unique_run_id_: np.ndarray = np.empty(0, dtype=np.int64)
        self.sig_acc_: np.ndarray = np.empty(0, dtype=np.float64)
        self.var_acc_: np.ndarray = np.empty(0, dtype=np.float64)
        self.old_file_format_: bool = False
        self.page_data_: np.ndarray = np.empty((9, 0), dtype=np.float64)
        self.split_log_ratio_: int = 1
        self.inform_about_target_file_: bool = True
        self.init_filebacked_output_: bool = False
        self.does_img_op_: bool = False
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
        Apply the page operation to the current ``self.page_data_`` chunk.
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
        self._init_(in_obj)
        self.does_img_op_ = not self.changes_pix_only
        self.pix_idx_start_ = 1
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
        if self.split_at_bin_edges_:
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
        Load and set into ``self.page_data_`` the block of pixel data for page ``idx``
        (1-based).
        """
        self.page_num = idx
        if self.split_at_bin_edges_:
            npix_block = npix_blocks[idx - 1]
            npix_in_block = int(np.sum(npix_block))
            pix_idx_end = self.pix_idx_start_ + npix_in_block - 1
            if npix_in_block == 0:
                self.page_data_ = np.empty((9, 0), dtype=np.float64)
            else:
                indices_1based = np.arange(
                    self.pix_idx_start_, pix_idx_end + 1, dtype=np.int64
                )
                self.page_data_ = np.asarray(
                    self.pix_.get_pixels(indices_1based, "-raw", "-align"),
                    dtype=np.float64,
                )
            self.pix_idx_start_ = pix_idx_end + 1
        else:
            raw = np.asarray(self.pix_.data, dtype=np.float64)
            self.page_data_ = raw if self.inplace_ else raw.copy()
        return self

    def common_page_op(self) -> PageOpBase:
        """
        Common post-page operation executed after ``apply_op`` for each page:
        updates ``pix_data_range_``, ``unique_run_id_`` (if ``exp_modified``),
        and writes/stores the modified page data.
        """
        self.pix_data_range_ = PixelDataBase.pix_minmax_ranges(
            self.page_data_, self.pix_data_range_
        )
        if self.exp_modified and self.page_data_.size > 0:
            page_runs = self.page_data_[self.run_idx_, :].astype(np.int64)
            if self.unique_run_id_.size == 0:
                self.unique_run_id_ = np.unique(page_runs)
            else:
                self.unique_run_id_ = np.unique(
                    np.concatenate([self.unique_run_id_, page_runs])
                )
        if not (self.inplace_ or self.do_nopix_):
            self.pix_ = self.pix_.store_page_data(self.page_data_, self.write_handle_)
        elif (
            self.inplace_
            and not self.do_nopix_
            and not getattr(self.pix_, "is_filebacked", False)
        ):
            self.pix_.set_raw_data(self.page_data_)
        return self

    def update_img_accumulators(
        self,
        npix_block: np.ndarray,
        npix_idx: np.ndarray,
        signal: np.ndarray,
        variance: Optional[np.ndarray] = None,
    ) -> PageOpBase:
        """
        Accumulate modified pixel signal and variance into ``sig_acc_`` and ``var_acc_``.
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
        if self.split_at_bin_edges_:
            self.sig_acc_[idx0:idx1] = s_ar.ravel(order="F")
            if has_var and e_ar is not None:
                self.var_acc_[idx0:idx1] = e_ar.ravel(order="F")
        else:
            self.sig_acc_[idx0:idx1] = self.sig_acc_[idx0:idx1] + s_ar.ravel(order="F")
            if has_var and e_ar is not None:
                self.var_acc_[idx0:idx1] = self.var_acc_[idx0:idx1] + e_ar.ravel(
                    order="F"
                )
        return self

    def finish_op(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Finalize page operation: update image from accumulators and transfer
        modifications to the output object.
        """
        self.update_image(self.sig_acc_, self.var_acc_)
        return self.finish_core_op(in_obj)

    def finish_core_op(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Transfer modifications of new image and pixels to the target object
        without running ``update_image``.
        """
        return self._finish_op_(in_obj)

    def print_range_warning(
        self, infile_name: str, is_old_file_format: bool
    ) -> None:
        """
        Print a warning informing the user that the source file contains an
        invalid data range and should be upgraded or finalized.
        """
        self._print_range_warning_(infile_name, is_old_file_format)

    def report_on_target_files(self, output_obj: Any) -> None:
        """
        Print information about the target file(s) produced by a file-backed page operation.
        """
        self._report_on_target_files_(output_obj)

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
        return self.outfile_

    @outfile.setter
    def outfile(self, val: Optional[str]) -> None:
        if val is None or val == "":
            self.outfile_ = ""
            return
        if not isinstance(val, (str, os.PathLike)):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: outfile type can only be str. "
                f"Provided: {type(val).__name__}"
            )
        self.outfile_ = str(val)

    @property
    def signal_idx(self) -> int:
        return self.signal_idx_

    @property
    def var_idx(self) -> int:
        return self.var_idx_

    @property
    def coord_idx(self) -> List[int]:
        return self.coord_idx_

    @property
    def run_idx(self) -> int:
        return self.run_idx_

    @property
    def inplace(self) -> bool:
        return self.inplace_

    @inplace.setter
    def inplace(self, val: bool) -> None:
        self.inplace_ = bool(val)

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
        if self.npix_ is None or (isinstance(self.npix_, np.ndarray) and self.npix_.size == 0):
            return np.array([self.pix_.num_pixels], dtype=np.int64)
        return self.npix_

    @npix.setter
    def npix(self, val: Any) -> None:
        if val is None:
            self.npix_ = None
        else:
            self.npix_ = np.asarray(val).ravel(order="F")

    @property
    def page_num(self) -> int:
        return self.get_page_num()

    @page_num.setter
    def page_num(self, val: int) -> None:
        self.set_page_num(val)

    @property
    def pix(self) -> PixelDataBase:
        return self.pix_

    @pix.setter
    def pix(self, val: PixelDataBase) -> None:
        if not isinstance(val, PixelDataBase):
            raise ValueError(
                "HORACE:PixelDataBase:invalid_argument: Pix can be an object of PixelDataBase class only"
            )
        self.pix_ = val

    @property
    def source_filename(self) -> str:
        full_fn = getattr(self.pix_, "full_filename", "") or ""
        return os.path.basename(full_fn)

    @property
    def op_name(self) -> str:
        return self.op_name_

    @op_name.setter
    def op_name(self, val: str) -> None:
        if not isinstance(val, str):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: op_name can be a text string only. "
                f"Provided {type(val).__name__}"
            )
        self.op_name_ = val

    @property
    def inform_about_target_file(self) -> bool:
        return self.inform_about_target_file_

    @inform_about_target_file.setter
    def inform_about_target_file(self, val: bool) -> None:
        self.inform_about_target_file_ = bool(val)

    @property
    def split_at_bin_edges(self) -> bool:
        return self.split_at_bin_edges_

    @split_at_bin_edges.setter
    def split_at_bin_edges(self, val: bool) -> None:
        self.split_at_bin_edges_ = bool(val)

    @property
    def is_range_valid(self) -> bool:
        return bool(self.pix_.is_range_valid())

    @property
    def old_file_format(self) -> bool:
        return self.old_file_format_

    @property
    def write_handle(self) -> Any:
        return self.write_handle_

    @property
    def do_missing_range_warning(self) -> bool:
        return self.get_do_missing_range_warning()

    @property
    def page_data(self) -> np.ndarray:
        return self.page_data_

    @page_data.setter
    def page_data(self, val: np.ndarray) -> None:
        self.page_data_ = np.asarray(val)

    @property
    def init_filebacked_output(self) -> bool:
        return self.init_filebacked_output_

    @init_filebacked_output.setter
    def init_filebacked_output(self, val: bool) -> None:
        self.init_filebacked_output_ = bool(val)

    @property
    def do_nopix(self) -> bool:
        return self.do_nopix_

    @do_nopix.setter
    def do_nopix(self, val: bool) -> None:
        self.do_nopix_ = bool(val)

    @property
    def img(self) -> Any:
        return self.img_

    @img.setter
    def img(self, val: Any) -> None:
        self.img_ = val

    @property
    def pix_data_range(self) -> np.ndarray:
        return self.pix_data_range_

    # =========================================================================
    # Protected Overloadable Methods
    # =========================================================================

    def get_page_num(self) -> int:
        return int(getattr(self.pix_, "page_num", 1))

    def set_page_num(self, val: int) -> None:
        self.pix_.page_num = val

    def get_exp_modified(self) -> bool:
        return self.old_file_format_

    def get_changes_pix_only(self) -> bool:
        return self.img_ is None

    def set_changes_pix_only(self, *args: Any) -> None:
        pass

    def get_do_missing_range_warning(self) -> bool:
        return self.img_ is not None

    def update_image(
        self,
        sig_acc: np.ndarray,
        var_acc: Optional[np.ndarray] = None,
        npix_acc: Optional[np.ndarray] = None,
    ) -> PageOpBase:
        """
        Transfer accumulated signal and variance from ``sig_acc`` and ``var_acc``
        into ``self.img_`` normalized by ``npix_acc``.
        """
        if self.changes_pix_only:
            return self
        if npix_acc is None:
            npix_acc = np.asarray(self.npix).ravel(order="F")
        return self._update_image_(sig_acc, var_acc, npix_acc)

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
        return self.split_log_ratio_

    def set_info_split_log_ratio(self, val: Union[int, float]) -> None:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise ValueError(
                f"HORACE:PageOpBase:invalid_argument: log_split_ratio can have only numeric value. "
                f"Provided: {type(val).__name__}"
            )
        self.split_log_ratio_ = max(1, int(round(abs(float(val)))))

    # =========================================================================
    # Private Helpers (ported from @PageOpBase/private/)
    # =========================================================================

    def _init_(self, in_obj: Any) -> PageOpBase:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/init_.m``.
        """
        from pyhorace.core.exceptions import HoraceError
        from pyhorace.sqw.sqw import sqw

        self.pix_data_range_ = PixelDataBase.EMPTY_RANGE.copy()

        if not (self.inplace_ or self.do_nopix_):
            self.write_handle_ = in_obj.get_write_handle(self.outfile_)
            if getattr(in_obj, "is_filebacked", False):
                self.init_filebacked_output = True

        if isinstance(in_obj, PixelDataBase):
            self.pix_ = (
                in_obj
                if (self.inplace_ or getattr(in_obj, "is_filebacked", False))
                else in_obj.copy()
            )
            self.img_ = None
            self.exp_info_ = None
            self.npix_ = np.array([in_obj.num_pixels], dtype=np.int64)
            if self.do_nopix_:
                raise HoraceError(
                    "HORACE:PageOpBase:invalid_argument",
                    "-nopix option provided to pageOp but input contans pixels only. Can not continue.",
                )
        elif isinstance(in_obj, sqw):
            self.img_ = in_obj.data.copy() if hasattr(in_obj.data, "copy") else copy.deepcopy(in_obj.data)
            self.pix_ = (
                in_obj.pix
                if (self.inplace_ or getattr(in_obj.pix, "is_filebacked", False))
                else in_obj.pix.copy()
            )
            self.exp_info_ = in_obj.experiment_info
            self.npix_ = np.asarray(self.img_.npix).ravel(order="F")
            self.sig_acc_ = np.zeros(self.npix_.size, dtype=np.float64)
            self.var_acc_ = np.zeros(self.npix_.size, dtype=np.float64)
        else:
            raise HoraceError(
                "HORACE:PageOpBase:invalid_argument",
                f"Init method accepts PixelData or SQW object input only. Provided {type(in_obj).__name__}",
            )

        self.pix_.keep_precision = True
        self.old_file_format_ = bool(getattr(self.pix_, "old_file_format", False))

        if self.exp_modified:
            self.unique_run_id_ = np.empty(0, dtype=np.int64)

        self._initialized = True
        return self

    def _update_image_(
        self,
        sig_acc: np.ndarray,
        var_acc: Optional[np.ndarray],
        npix_acc: np.ndarray,
    ) -> PageOpBase:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/update_image_.m``.
        """
        if self.changes_pix_only and not self.do_nopix_:
            return self

        s, e = normalize_signal(sig_acc, var_acc, npix_acc)
        img_shape = self.img_.npix.shape
        self.img_.do_check_combo_arg = False
        try:
            self.img_.s = s.reshape(img_shape, order="F")
            self.img_.npix = np.asarray(npix_acc, dtype=np.int64).reshape(img_shape, order="F")
            if e is not None:
                self.img_.e = e.reshape(img_shape, order="F")
            else:
                self.img_.e = np.zeros(img_shape, dtype=np.float64, order="F")
        finally:
            self.img_.do_check_combo_arg = True
        self.img_.check_combo_arg()
        return self

    def _finish_op_(self, in_obj: Any) -> Tuple[Any, PageOpBase]:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/finish_op_.m``.
        """
        from pyhorace.sqw.sqw import sqw

        was_filebacked = bool(getattr(self.pix_, "is_filebacked", False))
        if was_filebacked:
            orig_FName = getattr(self.pix_, "full_filename", "") or ""
        else:
            orig_FName = ""
        source_range_valid = self.is_range_valid

        if isinstance(in_obj, sqw):
            out_obj = in_obj.copy()
            if self.img_ is not None:
                out_obj.data = self.img_
            if self.do_nopix_:
                out_obj = out_obj.data
                self.sig_acc_ = np.empty(0, dtype=np.float64)
                self.var_acc_ = np.empty(0, dtype=np.float64)
                self._initialized = False
                return out_obj, self

            if self.exp_modified and self.exp_info_ is not None:
                out_obj.experiment_info = out_obj.experiment_info.get_subobj(self.unique_run_id_)

            if was_filebacked and not self.inplace_:
                from pyhorace.data.pixels.pixel_data_file_backed import (
                    PixelDataFileBacked,
                )

                self.pix_.keep_precision = False
                pix = PixelDataFileBacked()
                pix._num_pixels = self.pix_.num_pixels
                pix._full_filename = self.pix_.full_filename
                pix._base_page_size = self.pix_._base_page_size
                pix._alignment_matr = self.pix_._alignment_matr.copy()
                pix._is_corrected = self.pix_._is_corrected
                pix._old_file_format = self.pix_._old_file_format
                pix.set_data_range(self.pix_data_range_)
            else:
                pix = self.pix_
                pix.set_data_range(self.pix_data_range_)
                pix.keep_precision = False
            if not (self.inplace_ or self.do_nopix_):
                pix = pix.clear_alignment()
            out_obj.pix = pix
            if not self.inplace_:
                out_obj = out_obj.finish_dump(self)
                self._report_on_target_files_(out_obj)
        else:
            if was_filebacked and not self.inplace_:
                from pyhorace.data.pixels.pixel_data_file_backed import (
                    PixelDataFileBacked,
                )

                self.pix_.keep_precision = False
                pix = PixelDataFileBacked()
                pix._num_pixels = self.pix_.num_pixels
                pix._full_filename = self.pix_.full_filename
                pix._base_page_size = self.pix_._base_page_size
                pix._alignment_matr = self.pix_._alignment_matr.copy()
                pix._is_corrected = self.pix_._is_corrected
                pix._old_file_format = self.pix_._old_file_format
                pix.set_data_range(self.pix_data_range_)
            else:
                pix = self.pix_
                pix.set_data_range(self.pix_data_range_)
                pix.keep_precision = False
            if self.inplace_:
                out_obj = pix
            else:
                pix = pix.clear_alignment()
                out_obj = pix.finish_dump(self)
                self._report_on_target_files_(out_obj)

        self.write_handle_ = None

        if was_filebacked and not source_range_valid and self.do_missing_range_warning:
            self._print_range_warning_(orig_FName, self.old_file_format_)

        self.sig_acc_ = np.empty(0, dtype=np.float64)
        self.var_acc_ = np.empty(0, dtype=np.float64)
        self._initialized = False
        return out_obj, self

    def _print_range_warning_(
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

    def _report_on_target_files_(self, output_obj: Any) -> None:
        """
        Ported from ``horace_core/sqw/page_operations/@PageOpBase/private/report_on_target_files_.m``.
        """
        try:
            log_level = config_store.instance().get_value("hor_config", "log_level")
        except Exception:
            log_level = 1

        if (
            self.inform_about_target_file_
            and getattr(output_obj, "is_filebacked", False)
            and log_level > 0
        ):
            print("********************************************************************************")
            wh_list = (
                self.write_handle_
                if isinstance(self.write_handle_, (list, tuple))
                else [self.write_handle_]
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
