"""pixfile_combine_info class for combining multiple file-backed pixel datasets."""

from __future__ import annotations

import os
from typing import Any, ClassVar, List, Optional, Sequence
import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.data.pixels.multipix_base import MultipixBase
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked


def calc_job_indexes(n_tasks: int, n_workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Divide n_tasks among n_workers as evenly as possible (1-based inclusive indices)."""
    n_workers = min(max(1, int(n_workers)), max(1, int(n_tasks)))
    base = n_tasks // n_workers
    rem = n_tasks % n_workers
    counts = np.full(n_workers, base, dtype=int)
    counts[:rem] += 1
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    return starts, ends


@register_serializable
class pixfile_combine_info(MultipixBase):
    """Holds metadata for combining pixel data from multiple temporary/sqw files."""

    _positional_params: ClassVar[List[str]] = [
        "infiles",
        "nbins",
        "npix_each_file",
        "pos_npixstart",
        "pos_pixstart",
        "run_label",
        "filenum",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._pos_npixstart: np.ndarray = np.empty(0, dtype=np.uint64)
        self._pos_pixstart: np.ndarray = np.empty(0, dtype=np.uint64)
        self._npix_cumsum: np.ndarray = np.empty(0, dtype=np.uint64)
        self._f_accessors: List[Any] = []

        if len(args) == 1 and not kwargs and isinstance(args[0], pixfile_combine_info):
            other = args[0]
            self.from_bare_struct(other.to_bare_struct())
            return
        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            self.from_bare_struct(args[0])
            return

        if args or kwargs:
            self._init_from_args(args, kwargs)

    def _init_from_args(self, args: Sequence[Any], kwargs: dict[str, Any]) -> None:
        self._do_check_combo_arg = False
        try:
            for idx, val in enumerate(args):
                if idx < len(self._positional_params):
                    setattr(self, self._positional_params[idx], val)
                else:
                    raise HoraceError(
                        "HORACE:pixfile_combine_info:invalid_argument",
                        f"Too many positional arguments ({len(args)}).",
                    )
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise HoraceError(
                        "HORACE:pixfile_combine_info:invalid_argument",
                        f"Unknown parameter '{k}'.",
                    )
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()

    def saveable_fields(self) -> List[str]:
        return [
            "infiles",
            "nbins",
            "npix_each_file",
            "run_label",
            "filenum",
            "pos_npixstart",
            "pos_pixstart",
            "npix_cumsum",
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_filebacked(self) -> bool:
        return True

    def _set_infiles(self, val: Any) -> None:
        if val is None:
            self._infiles = []
        elif isinstance(val, str):
            self._infiles = [val]
        elif isinstance(val, (list, tuple, np.ndarray)):
            files = [str(x) for x in val]
            self._infiles = files
        else:
            raise HoraceError(
                "HORACE:pixfile_combine_info:invalid_argument",
                f"infiles must be a string or list of strings, got {type(val)}",
            )

    @property
    def pos_npixstart(self) -> np.ndarray:
        return self._pos_npixstart

    @pos_npixstart.setter
    def pos_npixstart(self, val: Any) -> None:
        if val is None:
            self._pos_npixstart = np.empty(0, dtype=np.uint64)
        else:
            arr = np.atleast_1d(np.asarray(val))
            if np.any(arr < 0):
                raise HoraceError(
                    "HORACE:pixfile_combine_info:invalid_argument",
                    "pos_npixstart must be non-negative.",
                )
            self._pos_npixstart = arr.astype(np.uint64).ravel()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def pos_pixstart(self) -> np.ndarray:
        return self._pos_pixstart

    @pos_pixstart.setter
    def pos_pixstart(self, val: Any) -> None:
        if val is None:
            self._pos_pixstart = np.empty(0, dtype=np.uint64)
        else:
            arr = np.atleast_1d(np.asarray(val))
            if np.any(arr < 0):
                raise HoraceError(
                    "HORACE:pixfile_combine_info:invalid_argument",
                    "pos_pixstart must be non-negative.",
                )
            self._pos_pixstart = arr.astype(np.uint64).ravel()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def npix_cumsum(self) -> np.ndarray:
        return self._npix_cumsum

    @npix_cumsum.setter
    def npix_cumsum(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._npix_cumsum = np.empty(0, dtype=np.uint64)
        else:
            self._npix_cumsum = np.atleast_1d(np.asarray(val, dtype=np.uint64)).ravel()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def check_combo_arg(self, recalc_data_range: bool = False) -> pixfile_combine_info:
        super().check_combo_arg(recalc_data_range=False)
        nf = self.nfiles
        if nf > 0:
            if self._pos_npixstart.size == 1 and nf > 1:
                self._pos_npixstart = np.full(nf, self._pos_npixstart[0], dtype=np.uint64)
            elif self._pos_npixstart.size not in (0, nf):
                raise HoraceError(
                    "HORACE:pixfile_combine_info:invalid_argument",
                    f"Length of pos_npixstart ({self._pos_npixstart.size}) does not match nfiles ({nf}).",
                )
            if self._pos_pixstart.size == 1 and nf > 1:
                self._pos_pixstart = np.full(nf, self._pos_pixstart[0], dtype=np.uint64)
            elif self._pos_pixstart.size not in (0, nf):
                raise HoraceError(
                    "HORACE:pixfile_combine_info:invalid_argument",
                    f"Length of pos_pixstart ({self._pos_pixstart.size}) does not match nfiles ({nf}).",
                )
        if recalc_data_range:
            self.recalc_data_range()
        return self

    def trim_nfiles(self, nfiles_to_leave: int) -> pixfile_combine_info:
        """Trim the file list and metadata arrays to the first nfiles_to_leave entries."""
        n = int(nfiles_to_leave)
        if n < 0:
            raise HoraceError(
                "HORACE:pixfile_combine_info:invalid_argument",
                "nfiles_to_leave must be non-negative.",
            )
        if n >= self.nfiles:
            return self
        self._do_check_combo_arg = False
        try:
            self._infiles = self._infiles[:n]
            if self._npix_each_file.size > n:
                self._npix_each_file = self._npix_each_file[:n]
            if self._pos_npixstart.size > n:
                self._pos_npixstart = self._pos_npixstart[:n]
            if self._pos_pixstart.size > n:
                self._pos_pixstart = self._pos_pixstart[:n]
            if self._filenum.size > n:
                self._filenum = self._filenum[:n]
            if isinstance(self._run_label, np.ndarray) and self._run_label.size > n:
                self._run_label = self._run_label[:n]
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()
        return self

    def split_into_parts(self, n_workers: int) -> List[pixfile_combine_info]:
        """Split pixfile_combine_info into up to n_workers parts for parallel processing."""
        if self.nfiles == 0:
            return [self.copy()]
        starts, ends = calc_job_indexes(self.nfiles, n_workers)
        parts: List[pixfile_combine_info] = []
        for s, e in zip(starts, ends):
            idx0 = s - 1
            idx1 = e
            sub_infiles = self._infiles[idx0:idx1]
            sub_npix = (
                self._npix_each_file[idx0:idx1].copy()
                if self._npix_each_file.size >= idx1
                else np.empty(0, dtype=np.uint64)
            )
            sub_pos_npix = (
                self._pos_npixstart[idx0:idx1].copy()
                if self._pos_npixstart.size >= idx1
                else np.empty(0, dtype=np.uint64)
            )
            sub_pos_pix = (
                self._pos_pixstart[idx0:idx1].copy()
                if self._pos_pixstart.size >= idx1
                else np.empty(0, dtype=np.uint64)
            )
            sub_filenum = (
                self.filenum[idx0:idx1].copy()
                if self.filenum.size >= idx1
                else np.arange(s, e + 1, dtype=int)
            )
            if isinstance(self._run_label, np.ndarray):
                sub_run_label: Any = self._run_label[idx0:idx1].copy()
            else:
                sub_run_label = self._run_label
            part = pixfile_combine_info(
                sub_infiles,
                self._nbins,
                sub_npix,
                sub_pos_npix,
                sub_pos_pix,
                sub_run_label,
                sub_filenum,
            )
            parts.append(part)
        return parts

    def init_pix_access(self) -> pixfile_combine_info:
        """Initialize PixelDataFileBacked accessors for each file."""
        self._f_accessors = []
        for idx, fname in enumerate(self._infiles):
            if os.path.isfile(fname):
                self._f_accessors.append(PixelDataFileBacked(fname))
            else:
                self._f_accessors.append(None)
        return self

    def get_dataset_page(
        self, n_dataset: int, pix_pos_start: int, npix_idx: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a page of pixel data and npix chunk from the n_dataset-th file (1-based)."""
        idx = int(n_dataset) - 1
        if idx < 0 or idx >= self.nfiles:
            raise HoraceError(
                "HORACE:pixfile_combine_info:invalid_argument",
                f"Dataset index {n_dataset} out of range (1..{self.nfiles}).",
            )
        fname = self._infiles[idx]
        if not os.path.isfile(fname):
            raise HoraceError(
                "HORACE:pixfile_combine_info:runtime_error",
                f"File '{fname}' does not exist.",
            )
        # Read npix chunk from pos_npixstart and pixel data from pos_pixstart
        b0, b1 = int(npix_idx[0]), int(npix_idx[1])
        n_bins_read = b1 - b0 + 1
        pos_npix = int(self._pos_npixstart[idx]) if self._pos_npixstart.size > idx else 0
        pos_pix = int(self._pos_pixstart[idx]) if self._pos_pixstart.size > idx else 0

        with open(fname, "rb") as f:
            f.seek(pos_npix + (b0 - 1) * 8)
            npix_chunk = np.fromfile(f, dtype=np.uint64, count=n_bins_read)
            n_pix_to_read = int(np.sum(npix_chunk))
            # pos_pixstart in Horace SQW binary format has a 12-byte header or direct float32 array;
            # seek to pos_pix + (pix_pos_start - 1) * 9 * 4
            f.seek(pos_pix + (int(pix_pos_start) - 1) * self.DEFAULT_NUM_PIX_FIELDS * 4)
            raw = np.fromfile(
                f, dtype=np.float32, count=n_pix_to_read * self.DEFAULT_NUM_PIX_FIELDS
            )
            pix_data = raw.reshape((self.DEFAULT_NUM_PIX_FIELDS, n_pix_to_read), order="F").astype(
                np.float64
            )

        if self._run_label == "filenum":
            pix_data[4, :] = self.filenum[idx]
        elif isinstance(self._run_label, np.ndarray):
            pix_data[4, :] += self._run_label[idx]

        return pix_data, npix_chunk

    def close_faccessors(self) -> None:
        for acc in self._f_accessors:
            if acc is not None and hasattr(acc, "deactivate"):
                acc.deactivate()
        self._f_accessors = []

    def recalc_data_range(self, fields: Any = "all") -> pixfile_combine_info:
        """Recalculate global data_range across all input files."""
        global_range = np.copy(self.EMPTY_RANGE)
        for idx, fname in enumerate(self._infiles):
            if not os.path.isfile(fname):
                continue
            pix = PixelDataFileBacked(fname)
            dr = pix.data_range.copy()
            if self._run_label == "filenum":
                dr[:, 4] = self.filenum[idx]
            elif isinstance(self._run_label, np.ndarray):
                dr[:, 4] += self._run_label[idx]
            global_range[0, :] = np.minimum(global_range[0, :], dr[0, :])
            global_range[1, :] = np.maximum(global_range[1, :], dr[1, :])
        self.set_data_range(global_range)
        return self
