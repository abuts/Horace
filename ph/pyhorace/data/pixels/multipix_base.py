"""MultipixBase class for multi-file and multi-object pixel containers in Horace."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, List, Sequence, Union
import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


class MultipixBase(PixelDataBase):
    """Base class for pixel data composed of multiple files or PixelDataBase objects."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._infiles: List[Any] = []
        self._nbins: int = 0
        self._npix_each_file: np.ndarray = np.empty(0, dtype=np.uint64)
        self._run_label: Union[str, np.ndarray] = "nochange"
        self._filenum: np.ndarray = np.empty(0, dtype=int)
        self._n_Accepted: int = 0
        self._page_num: int = 1
        self._page_size: int = PixelDataBase.DEFAULT_PAGE_SIZE

    def saveable_fields(self) -> List[str]:
        return ["infiles", "nbins", "npix_each_file", "run_label", "filenum"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def nfiles(self) -> int:
        return len(self._infiles)

    @property
    def num_pixels(self) -> int:
        return self.get_num_pixels()

    def get_num_pixels(self) -> int:
        if self._npix_each_file.size == 0:
            return 0
        return int(np.sum(self._npix_each_file))

    def copy(self) -> MultipixBase:
        out = self.__class__()
        out.from_bare_struct(self.to_bare_struct())
        return out

    def get_prop(self, fld: str) -> np.ndarray:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            f"Property access for '{fld}' is not supported on MultipixBase.",
        )

    def set_prop(self, fld: str, val: Any) -> MultipixBase:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            f"Property assignment for '{fld}' is not supported on MultipixBase.",
        )

    def get_raw_data(self, page_number: Any = None) -> np.ndarray:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "get_raw_data is not supported on MultipixBase.",
        )

    def set_raw_data(self, pix_data_arr: Any) -> MultipixBase:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "set_raw_data is not supported on MultipixBase.",
        )

    def get_raw_pix_data(
        self, row_pix_idx: np.ndarray, col_pix_idx: Any = None
    ) -> np.ndarray:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "get_raw_pix_data is not supported on MultipixBase.",
        )

    def set_alignment_matrix(self, val: Any) -> MultipixBase:
        self._set_alignment_matr_internal(val, self.invalidate_range)
        return self

    def set_data_wrap(self, val: Any) -> MultipixBase:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "set_data_wrap is not supported on MultipixBase.",
        )

    @property
    def infiles(self) -> List[Any]:
        return self._infiles

    @infiles.setter
    def infiles(self, val: Any) -> None:
        self._set_infiles(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_infiles(self, val: Any) -> None:
        if val is None:
            self._infiles = []
        elif isinstance(val, (str, PixelDataBase)):
            self._infiles = [val]
        elif isinstance(val, (list, tuple, np.ndarray)):
            self._infiles = list(val)
        else:
            raise HoraceError(
                "HORACE:MultipixBase:invalid_argument",
                f"Invalid infiles type: {type(val)}",
            )

    @property
    def nbins(self) -> int:
        return self._nbins

    @nbins.setter
    def nbins(self, val: Any) -> None:
        val_int = int(val)
        if val_int < 0:
            raise HoraceError(
                "HORACE:MultipixBase:invalid_argument",
                f"nbins must be a non-negative integer, got {val}",
            )
        self._nbins = val_int
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def npix_each_file(self) -> np.ndarray:
        return self._npix_each_file

    @npix_each_file.setter
    def npix_each_file(self, val: Any) -> None:
        self._set_npix_each_file(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_npix_each_file(self, val: Any) -> None:
        if val is None:
            self._npix_each_file = np.empty(0, dtype=np.uint64)
            return
        arr = np.atleast_1d(np.asarray(val))
        if np.any(arr < 0):
            raise HoraceError(
                "HORACE:MultipixBase:invalid_argument",
                "npix_each_file values cannot be negative.",
            )
        self._npix_each_file = arr.astype(np.uint64).ravel()

    @property
    def run_label(self) -> Union[str, np.ndarray]:
        return self._run_label

    @run_label.setter
    def run_label(self, val: Any) -> None:
        if isinstance(val, str):
            val_l = val.lower()
            if val_l not in ("nochange", "filenum", "fileno"):
                raise HoraceError(
                    "HORACE:MultipixBase:invalid_argument",
                    f"Invalid string value '{val}' for run_label. Must be 'nochange' or 'filenum'.",
                )
            if val_l == "fileno":
                val_l = "filenum"
            self._run_label = val_l
        elif isinstance(val, (list, tuple, np.ndarray, int, float, np.number)):
            arr = np.atleast_1d(np.asarray(val)).ravel()
            self._run_label = arr
        else:
            raise HoraceError(
                "HORACE:MultipixBase:invalid_argument",
                f"Invalid run_label: {val}",
            )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def filenum(self) -> np.ndarray:
        if self._filenum.size == 0 and self.nfiles > 0:
            return np.arange(1, self.nfiles + 1, dtype=int)
        return self._filenum

    @filenum.setter
    def filenum(self, val: Any) -> None:
        if val is None:
            self._filenum = np.empty(0, dtype=int)
        else:
            self._filenum = np.atleast_1d(np.asarray(val, dtype=int)).ravel()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def page_size(self) -> int:
        return self._page_size

    @page_size.setter
    def page_size(self, val: int) -> None:
        self._page_size = max(1, int(val))

    @property
    def num_pages(self) -> int:
        return max(1, self.nfiles)

    @property
    def page_num(self) -> int:
        return self._page_num

    @page_num.setter
    def page_num(self, val: int) -> None:
        val_int = int(val)
        if val_int < 1 or (self.nfiles > 0 and val_int > self.nfiles):
            raise HoraceError(
                "HORACE:MultipixBase:invalid_argument",
                f"page_num must be between 1 and {max(1, self.nfiles)}, got {val_int}.",
            )
        self._page_num = val_int

    @property
    def has_more(self) -> bool:
        return self._page_num < self.nfiles

    def advance(self) -> int:
        if not self.has_more:
            raise HoraceError(
                "HORACE:MultipixBase:runtime_error",
                "Cannot advance past the last file/object.",
            )
        self._page_num += 1
        return self._page_num

    def reset(self) -> None:
        self._page_num = 1

    # ------------------------------------------------------------------
    # Abstract / Stubbed PixelDataBase Methods
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "Direct data property access is not supported on MultipixBase.",
        )

    @data.setter
    def data(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "Direct data property assignment is not supported on MultipixBase.",
        )

    def get_pixels(self, pix_indices: Any, *args: Any, **kwargs: Any) -> PixelDataBase:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "get_pixels is not implemented on MultipixBase.",
        )

    def get_pix_in_ranges(
        self, pix_starts: Any, pix_ends: Any, rebind_idx: bool = False, keep_precision: bool = False
    ) -> PixelDataBase:
        raise HoraceError(
            "HORACE:MultipixBase:not_implemented",
            "get_pix_in_ranges is not implemented on MultipixBase.",
        )

    @abstractmethod
    def recalc_data_range(self, fields: Any = "all") -> MultipixBase:
        """Recalculate data range across all constituent datasets."""

    @abstractmethod
    def init_pix_access(self) -> MultipixBase:
        """Initialize accessors to underlying pixel datasets."""

    @abstractmethod
    def get_dataset_page(
        self, n_dataset: int, pix_pos_start: int, npix_idx: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a page of pixel data and corresponding npix chunk from dataset n_dataset."""

    @abstractmethod
    def close_faccessors(self) -> None:
        """Close open file accessors."""

    def check_combo_arg(self, recalc_data_range: bool = False) -> MultipixBase:
        nf = self.nfiles
        if nf > 0:
            if self._npix_each_file.size == 1 and nf > 1:
                self._npix_each_file = np.full(nf, self._npix_each_file[0], dtype=np.uint64)
            elif self._npix_each_file.size not in (0, nf):
                raise HoraceError(
                    "HORACE:MultipixBase:invalid_argument",
                    f"Length of npix_each_file ({self._npix_each_file.size}) does not match nfiles ({nf}).",
                )
            if isinstance(self._run_label, np.ndarray) and self._run_label.size != nf:
                raise HoraceError(
                    "HORACE:MultipixBase:invalid_argument",
                    f"Length of numeric run_label ({self._run_label.size}) must equal nfiles ({nf}).",
                )
            if self._filenum.size == 0:
                self._filenum = np.arange(1, nf + 1, dtype=int)
            elif self._filenum.size != nf:
                raise HoraceError(
                    "HORACE:MultipixBase:invalid_argument",
                    f"Length of filenum ({self._filenum.size}) must equal nfiles ({nf}).",
                )
        return self
