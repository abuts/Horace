"""File-backed pixel data class (translated from ``@PixelDataFileBacked``).

All interactions with the underlying pixel file go through
:class:`~pyhorace.data.pixels.filebacked_pix_interface.FileBackedPixInterface`
(via ``self._f_accessor``).
"""

from __future__ import annotations

import math
import os
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.data.pixels.filebacked_pix_interface import (
    FileBackedPixInterface,
    FileBackedPixStub,
)
from pyhorace.data.pixels.pix_data import pix_data
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pix_write_handle import pix_write_handle
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


class _TmpFileHolder:
    """Reference-counted temporary file holder (replaces ``file_obj_holder``)."""

    def __init__(self, file_name: str) -> None:
        self.file_name: str = str(file_name)
        self.ref_count: int = 1

    def acquire(self) -> "_TmpFileHolder":
        self.ref_count += 1
        return self

    def release(self) -> None:
        self.ref_count -= 1
        if self.ref_count <= 0 and self.file_name and os.path.isfile(self.file_name):
            try:
                os.remove(self.file_name)
            except OSError:
                pass


class PixelDataFileBacked(PixelDataBase):
    """File-backed implementation of :class:`PixelDataBase`."""

    DATA_FORMAT: ClassVar[str] = "single"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._f_accessor: Optional[FileBackedPixInterface] = None
        self._num_pixels: int = 0
        self._offset: int = 1
        self._tail: int = 0
        self._tmp_file_holder: Optional[_TmpFileHolder] = None

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            self.init(args[0])
            return

        self.set_positional_and_key_val_arguments(
            self.saveable_fields(), False, *args, **kwargs
        )

    def init(self, arg: Any, writable: bool = False) -> "PixelDataFileBacked":
        """Initialize ``PixelDataFileBacked`` from a file, accessor, array, or ``PixelDataBase``."""
        if arg is None:
            return self

        if isinstance(arg, PixelDataFileBacked):
            if arg._f_accessor is not None:
                if isinstance(arg._f_accessor, FileBackedPixStub):
                    self._f_accessor = arg._f_accessor.clone()
                else:
                    self._f_accessor = arg._f_accessor
            self._num_pixels = arg._num_pixels
            self._offset = arg._offset
            self._tail = arg._tail
            self._full_filename = arg._full_filename
            self._data_range = arg._data_range.copy()
            self._alignment_matr = arg._alignment_matr.copy()
            self._is_corrected = arg._is_corrected
            self._old_file_format = arg._old_file_format
            self._unique_run_id = arg._unique_run_id.copy()
            self._keep_precision = arg._keep_precision
            self._base_page_size = arg._base_page_size
            self._page_num = 1
            if arg._tmp_file_holder is not None:
                self._tmp_file_holder = arg._tmp_file_holder.acquire()
            return self

        if isinstance(arg, PixelDataBase):
            # Initialize from in-memory PixelDataMemory
            wh = pix_write_handle(arg.full_filename, keep_precision=arg.keep_precision)
            if arg.num_pixels > 0:
                wh.store_page(arg.data)
            self._full_filename = arg.full_filename
            self._alignment_matr = arg.alignment_matr.copy()
            self._is_corrected = arg.is_corrected
            self.finish_dump(wh)
            return self

        if isinstance(arg, FileBackedPixInterface):
            self._f_accessor = arg
            self._num_pixels = arg.num_pixels
            self._offset = arg.offset
            self._tail = arg.tail
            self._full_filename = arg.full_filename
            self._page_num = 1
            self.recalc_data_range("all")
            return self

        if isinstance(arg, (str, os.PathLike)):
            filepath = str(arg)
            stub = FileBackedPixStub(filepath, writable=writable)
            self._f_accessor = stub
            self._num_pixels = stub.num_pixels
            self._offset = stub.offset
            self._tail = stub.tail
            self._full_filename = stub.full_filename
            self._page_num = 1
            self.recalc_data_range("all")
            return self

        if hasattr(arg, "get_pix") or hasattr(arg, "npix") or hasattr(arg, "fake_data"):
            stub = FileBackedPixStub.from_faccess(arg, writable=writable)
            self._f_accessor = stub
            self._num_pixels = stub.num_pixels
            self._offset = stub.offset
            self._tail = stub.tail
            self._full_filename = stub.full_filename
            self._page_num = 1
            if hasattr(arg, "pix_range"):
                pr = np.asarray(getattr(arg, "pix_range"), dtype=np.float64)
                if pr.shape == (2, 4):
                    self._data_range[:, :4] = pr
            if hasattr(arg, "data_range"):
                dr = np.asarray(getattr(arg, "data_range"), dtype=np.float64)
                if dr.shape == (2, 9):
                    self._data_range = dr.copy()
            if not self.is_range_valid("all"):
                self.recalc_data_range("all")
            return self

        if isinstance(arg, pix_data):
            self.set_data_wrap(arg)
            return self

        if isinstance(arg, pix_metadata):
            self.set_metadata(arg)
            return self

        if isinstance(arg, dict):
            loaded = PixelDataBase.loadobj(arg)
            self.init(loaded)
            return self

        if isinstance(arg, (int, float, np.number)) and not isinstance(arg, bool):
            fval = float(arg)
            ival = int(fval)
            if fval < 0 or fval != ival or np.isnan(fval) or np.isinf(fval):
                raise HoraceError(
                    f"HORACE:PixelDataFileBacked:invalid_argument - "
                    f"Scalar pixel count must be a non-negative integer, got {arg!r}"
                )
            arr = np.zeros((self.DEFAULT_NUM_PIX_FIELDS, ival), dtype=np.float32)
            self._init_from_numeric(arr)
            return self

        if isinstance(arg, (np.ndarray, list, tuple)):
            arr = np.asarray(arg)
            if arr.dtype == object or np.issubdtype(arr.dtype, np.character) or arr.dtype == bool:
                raise HoraceError(
                    f"HORACE:PixelDataFileBacked:invalid_argument - "
                    f"Cannot construct PixelDataFileBacked from array of dtype {arr.dtype}"
                )
            if arr.ndim == 0 or (arr.size == 1 and arr.ndim <= 2 and arr.shape != (9, 1)):
                self.init(arr.item())
                return self
            self._init_from_numeric(arr)
            return self

        raise HoraceError(
            f"HORACE:PixelDataFileBacked:invalid_argument - "
            f"Cannot initialize PixelDataFileBacked from {type(arg).__name__}"
        )

    def _init_from_numeric(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            self._f_accessor = None
            self._num_pixels = 0
            self._data_range = self.EMPTY_RANGE.copy()
            return
        if arr.ndim == 1 and arr.size == self.DEFAULT_NUM_PIX_FIELDS:
            arr = arr.reshape((self.DEFAULT_NUM_PIX_FIELDS, 1))
        if arr.ndim != 2 or arr.shape[0] != self.DEFAULT_NUM_PIX_FIELDS:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"Pixel data array must be 2D with {self.DEFAULT_NUM_PIX_FIELDS} rows, got shape {arr.shape}"
            )
        wh = pix_write_handle(self._full_filename, keep_precision=self._keep_precision)
        wh.store_page(arr)
        self.finish_dump(wh)

    @property
    def is_filebacked(self) -> bool:
        return True

    @property
    def f_accessor(self) -> Optional[FileBackedPixInterface]:
        return self._f_accessor

    @property
    def offset(self) -> int:
        if self._f_accessor is not None:
            return self._f_accessor.offset
        return self._offset

    @property
    def tail(self) -> int:
        if self._f_accessor is not None:
            return self._f_accessor.tail
        return self._tail

    def copy(self) -> "PixelDataFileBacked":
        out = PixelDataFileBacked()
        out.init(self)
        out._page_num = 1
        return out

    def delete(self) -> None:
        """Close the file accessor and delete temporary file if owned."""
        if self._f_accessor is not None and self._f_accessor.is_activated:
            self._f_accessor.deactivate()
        if self._tmp_file_holder is not None:
            self._tmp_file_holder.release()
            self._tmp_file_holder = None

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass

    def deactivate(self) -> "PixelDataFileBacked":
        """Close file handle so the backing file can be moved or renamed."""
        if self._f_accessor is not None:
            self._f_accessor.deactivate()
        return self

    def activate(
        self,
        filename: Optional[str] = None,
        no_tmp_file: bool = False,
        writable: Optional[bool] = None,
    ) -> "PixelDataFileBacked":
        """Re-open file handle after ``deactivate()``, optionally at a new ``filename``."""
        if self._f_accessor is None:
            if not filename:
                raise HoraceError(
                    "HORACE:PixelDataFileBacked:invalid_argument - "
                    "Cannot activate empty PixelDataFileBacked without filename."
                )
            self.init(filename, writable=bool(writable))
        else:
            target_fn = filename if filename else self._f_accessor.full_filename
            self._f_accessor.activate(target_fn, writable=writable)
            self._full_filename = self._f_accessor.full_filename

        ext = os.path.splitext(self._full_filename)[1]
        if ext.startswith(".tmp_") and not no_tmp_file:
            self.set_as_tmp_obj(self._full_filename)
        else:
            self._tmp_file_holder = None
        return self

    def set_as_tmp_obj(self, filename: Optional[str] = None) -> "PixelDataFileBacked":
        """Mark the backing file for deletion when this object is destroyed."""
        fn = str(filename) if filename else self.full_filename
        if self._tmp_file_holder is not None and self._tmp_file_holder.file_name != fn:
            self._tmp_file_holder.release()
        self._tmp_file_holder = _TmpFileHolder(fn)
        return self

    def get_write_handle(self, targ_filename: str = "") -> pix_write_handle:
        targ_fn, _ = PixelDataBase.build_op_filename(self.full_filename, targ_filename)
        return pix_write_handle(targ_fn, keep_precision=self.keep_precision)

    def store_page_data(self, data: np.ndarray, wh: pix_write_handle) -> "PixelDataFileBacked":
        wh.store_page(data)
        return self

    def finish_dump(self, wh_or_op: Any) -> "PixelDataFileBacked":
        """Complete paged write from a :class:`pix_write_handle` and activate new accessor."""
        wh: pix_write_handle = getattr(wh_or_op, "write_handle", wh_or_op)
        if self._tmp_file_holder is not None:
            self._tmp_file_holder.release()
            self._tmp_file_holder = None

        self._f_accessor = wh.f_accessor
        self._num_pixels = wh.num_pixels
        self._offset = wh.f_accessor.offset
        self._tail = wh.f_accessor.tail
        self._full_filename = wh.write_FileName
        self._data_range = wh.pix_range.copy()
        self._page_num = 1
        if wh.is_tmp_obj:
            self.set_as_tmp_obj(wh.write_FileName)
        return self

    def get_is_tmp_obj(self) -> bool:
        return self._tmp_file_holder is not None

    def get_full_filename(self) -> str:
        if self._tmp_file_holder is not None:
            return self._tmp_file_holder.file_name
        if self._f_accessor is not None and self._f_accessor.full_filename:
            return self._f_accessor.full_filename
        return self._full_filename

    def get_num_pixels(self) -> int:
        if self._f_accessor is not None:
            return self._f_accessor.num_pixels
        return self._num_pixels

    def get_read_only(self) -> bool:
        if self._f_accessor is None:
            return True
        return not self._f_accessor.writable

    def get_pix_byte_size(self, keep_precision: Optional[bool] = None) -> int:
        kp = self.keep_precision if keep_precision is None else bool(keep_precision)
        return self.DEFAULT_NUM_PIX_FIELDS * (4 if kp else 8)

    # ------------------------------------------------------------------
    # Paging methods
    # ------------------------------------------------------------------
    def get_page_num(self) -> int:
        return self._page_num

    def set_page_num(self, val: Any) -> "PixelDataFileBacked":
        if isinstance(val, (str, bytes)) or not isinstance(
            val, (int, float, np.number, np.ndarray)
        ):
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"page number should be a positive numeric scalar, got {val!r}"
            )
        arr = np.asarray(val)
        if arr.size != 1:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"page number should be a positive numeric scalar, got shape {arr.shape}"
            )
        ival = int(math.floor(float(arr.item())))
        if ival < 1:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"page number should be a positive numeric scalar >= 1, got {ival}"
            )
        if ival > self.num_pages:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"page number ({ival}) should not exceed total number of pages ({self.num_pages})"
            )
        self._page_num = ival
        return self

    @property
    def page_number(self) -> int:
        return self.get_page_num()

    @page_number.setter
    def page_number(self, val: Any) -> None:
        self.set_page_num(val)

    def get_page_size(self) -> int:
        if self.num_pixels == 0:
            return 0
        if len(self._pix_page_chunks) > 0:
            return int(np.sum(self._pix_page_chunks[self._page_num - 1]))
        s_idx, e_idx = self.get_page_idx_(self._page_num)
        return max(0, e_idx - s_idx + 1)

    def get_num_pages(self) -> int:
        if self.num_pixels == 0:
            return 1
        if len(self._pix_page_chunks) > 0:
            return len(self._pix_page_chunks)
        base_pg = min(self.base_page_size, self.num_pixels)
        return max(1, int(math.ceil(self.num_pixels / base_pg)))

    def get_page_idx_(self, page_number: Optional[int] = None) -> Tuple[int, int]:
        """Return 1-based inclusive ``(pix_idx_start, pix_idx_end)`` for ``page_number``."""
        if page_number is None:
            pg = self._page_num
        else:
            self.set_page_num(page_number)
            pg = self._page_num

        if self.num_pixels == 0:
            return 1, 0

        if len(self._pix_page_chunks) == 0:
            base_pg = min(self.base_page_size, self.num_pixels)
            pix_idx_start = (pg - 1) * base_pg + 1
            pix_idx_end = min(pix_idx_start + base_pg - 1, self.num_pixels)
        else:
            pix_idx_start = int(self._pix_page_idx_start[pg - 1])
            pgs = int(np.sum(self._pix_page_chunks[pg - 1]))
            pix_idx_end = min(pix_idx_start + pgs - 1, self.num_pixels)

        if self.num_pixels > 0 and pix_idx_start > self.num_pixels:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:runtime_error - "
                f"pix_idx_start ({pix_idx_start}) exceeds number of pixels ({self.num_pixels})"
            )
        return pix_idx_start, pix_idx_end

    def move_to_first_page(self) -> "PixelDataFileBacked":
        self._page_num = 1
        return self

    @property
    def has_more(self) -> bool:
        return self._page_num < self.num_pages

    def advance(self) -> int:
        """Advance to the next page and return the new 1-based page number."""
        if self._page_num >= self.num_pages:
            raise HoraceError(
                "HORACE:PixelDataFileBacked:runtime_error - "
                "Cannot advance past the final page of pixels."
            )
        self._page_num += 1
        return self._page_num

    # ------------------------------------------------------------------
    # Property getters/setters via FileBackedPixInterface
    # ------------------------------------------------------------------
    def get_prop(self, fld: str) -> np.ndarray:
        idx = self.FIELD_INDEX_MAP[fld]
        idx0 = [i - 1 for i in idx]
        if self._f_accessor is None or not self._f_accessor.is_activated or self.num_pixels == 0:
            return np.empty(
                (len(idx0), 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )

        s_idx, e_idx = self.get_page_idx_(self._page_num)
        if self._is_corrected and any(i < 3 for i in idx0):
            raw_all = self._f_accessor.get_pix_slice(slice(None), s_idx, e_idx)
            pix_coord = self._alignment_matr @ raw_all[:3, :]
            for c in idx0:
                if c < 3:
                    raw_all[c, :] = pix_coord[c, :]
            out = raw_all[idx0, :]
        else:
            out = self._f_accessor.get_pix_slice(idx0, s_idx, e_idx)

        if self._keep_precision:
            return np.asarray(out, dtype=np.float32)
        return np.asarray(out, dtype=np.float64)

    def set_prop(self, fld: str, val: Any) -> "PixelDataFileBacked":
        if self.read_only:
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"File {self.full_filename} is opened in read-only mode. Cannot change its properties."
            )
        val_arr = self.check_set_prop(fld, val)
        s_idx, e_idx = self.get_page_idx_(self._page_num)
        e_idx = min(e_idx, s_idx - 1 + val_arr.shape[1])
        idx0 = [i - 1 for i in self.FIELD_INDEX_MAP[fld]]
        assert self._f_accessor is not None
        self._f_accessor.set_pix_slice(idx0, s_idx, e_idx, val_arr)

        if self._is_corrected:
            self._is_corrected = False
            self._alignment_matr = np.eye(3, dtype=np.float64)

        if self.num_pages == 1:
            self.recalc_data_range(fld)
        else:
            self.invalidate_range(fld)
        return self

    def get_raw_data(self, page_number: Optional[int] = None) -> np.ndarray:
        pg = self._page_num if page_number is None else int(page_number)
        if self._f_accessor is None or not self._f_accessor.is_activated or self.num_pixels == 0:
            return np.empty(
                (self.DEFAULT_NUM_PIX_FIELDS, 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )
        s_idx, e_idx = self.get_page_idx_(pg)
        out = self._f_accessor.get_pix_slice(slice(None), s_idx, e_idx)
        if self._keep_precision:
            return np.asarray(out, dtype=np.float32)
        return np.asarray(out, dtype=np.float64)

    def get_data(
        self,
        pix_fields_or_page: Any = None,
        page_num: Optional[int] = None,
    ) -> np.ndarray:
        out = super().get_data(pix_fields_or_page, page_num)
        if self._keep_precision:
            return np.asarray(out, dtype=np.float32)
        return out

    def set_raw_data(self, pix_data_arr: Any) -> "PixelDataFileBacked":
        if pix_data_arr is None:
            self._f_accessor = None
            self._num_pixels = 0
            return self
        arr = np.asarray(pix_data_arr)
        if not np.issubdtype(arr.dtype, np.number):
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"Pixel data must be numeric, got {arr.dtype}"
            )
        self._init_from_numeric(arr)
        return self

    def get_raw_pix_data(
        self,
        row_pix_idx: np.ndarray,
        col_pix_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._f_accessor is None or not self._f_accessor.is_activated or self.num_pixels == 0:
            n_rows = 9 if col_pix_idx is None else len(col_pix_idx)
            return np.empty(
                (n_rows, 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )
        fields: Union[slice, np.ndarray] = (
            slice(None) if col_pix_idx is None else np.asarray(col_pix_idx, dtype=np.intp)
        )
        out = self._f_accessor.get_pix_at_indices(fields, np.asarray(row_pix_idx, dtype=np.intp))
        if self._keep_precision:
            return np.asarray(out, dtype=np.float32)
        return np.asarray(out, dtype=np.float64)

    def recalc_data_range(
        self, fld_names: Union[str, Sequence[str], np.ndarray] = "all"
    ) -> "PixelDataFileBacked":
        ind = self.field_index(fld_names)
        idx0 = ind - 1
        self._data_range[:, idx0] = self.EMPTY_RANGE[:, idx0]
        if self._f_accessor is None or not self._f_accessor.is_activated or self.num_pixels == 0:
            return self

        saved_pg = self._page_num
        try:
            for pg in range(1, self.num_pages + 1):
                self._page_num = pg
                pg_data = self.get_data(fld_names, pg)
                self._data_range[:, idx0] = self.pix_minmax_ranges(
                    pg_data, self._data_range[:, idx0]
                )
        finally:
            self._page_num = saved_pg
        return self

    def set_alignment_matrix(self, val: Any) -> "PixelDataFileBacked":
        self._set_alignment_matr_internal(val, self.invalidate_range)
        return self

    def set_data_wrap(self, val: Any) -> "PixelDataFileBacked":
        if not isinstance(val, pix_data):
            raise HoraceError(
                f"HORACE:PixelDataFileBacked:invalid_argument - "
                f"pix_data_wrap can only be set with a pix_data instance, got {type(val).__name__}"
            )
        if val.data.size == 0:
            self._num_pixels = val.npix
            return self
        self.init(val.data)
        return self

    def set_metadata(self, val: Any) -> "PixelDataFileBacked":
        super().set_metadata(val)
        self._num_pixels = val.npix
        self._pix_page_chunks = []
        self._pix_page_idx_start = np.array([], dtype=np.intp)
        if val.full_filename and os.path.isfile(val.full_filename) and self._f_accessor is None:
            self._f_accessor = FileBackedPixStub(val.full_filename)
            self._num_pixels = self._f_accessor.num_pixels
        return self

    @staticmethod
    def apply_op(obj_in: "PixelDataFileBacked", page_op: Any) -> "PixelDataFileBacked":
        """Apply a ``PageOp`` across all pages of a file-backed pixel dataset."""
        npix_chunks = [obj_in.get_page_size()]
        for pg in range(1, obj_in.num_pages + 1):
            obj_in.page_num = pg
            page_op.get_page_data(pg, npix_chunks)
            page_op.common_op(pg)
            page_op.update_img_accumulators()
        return page_op.finish_op(obj_in)


register_serializable(PixelDataFileBacked)
