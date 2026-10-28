"""File-backed pixel data class (translated from ``@PixelDataFileBacked``).

All interactions with the underlying pixel file go through a file accessor
subscribed to :class:`~pyhorace.io.sqw_formats_factory.sqw_formats_factory`
(or a compatible :class:`~pyhorace.data.pixels.filebacked_pix_interface.FileBackedPixInterface`
fallback via ``self._f_accessor``), using ``get_pix``, ``get_pix_at_indices``,
``get_pix_in_ranges``, ``put_pix``, and ``put_raw_pix``.
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


def _open_faccess_for_file(filepath: str, writable: bool = False) -> Any:
    """Open a file accessor from ``sqw_formats_factory`` if possible, else fallback to ``FileBackedPixStub``."""
    filepath = os.path.abspath(str(filepath))
    if not os.path.isfile(filepath):
        raise HoraceError(
            f"HORACE:PixelDataFileBacked:invalid_argument - "
            f"Cannot find file '{filepath}' to initialize PixelDataFileBacked."
        )
    try:
        from pyhorace.io.sqw_formats_factory import sqw_formats_factory

        ldr = sqw_formats_factory.instance().get_loader(filepath)
        if writable and hasattr(ldr, "reopen_to_write"):
            ldr.reopen_to_write()
        return ldr
    except Exception:
        return FileBackedPixStub(filepath, writable=writable)


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
        self._f_accessor: Optional[Any] = None
        self._num_pixels: int = 0
        self._offset: int = 1
        self._tail: int = 0
        self._tmp_file_holder: Optional[_TmpFileHolder] = None
        self._f_accessor_activated: bool = False

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            self.init(args[0])
            return

        self.set_positional_and_key_val_arguments(
            self.saveable_fields(), False, *args, **kwargs
        )

    def _check_facc_activated(self) -> bool:
        """Return True if ``self._f_accessor`` is open/activated (re-opening if needed)."""
        if self._f_accessor is None:
            return False
        is_act_attr = getattr(self._f_accessor, "is_activated", None)
        if callable(is_act_attr):
            try:
                act = bool(is_act_attr("read"))
            except TypeError:
                act = bool(is_act_attr())
            if not act and self._f_accessor_activated:
                fn = getattr(self._f_accessor, "full_filename", "") or self._full_filename
                if fn and os.path.isfile(fn) and hasattr(self._f_accessor, "activate"):
                    try:
                        self._f_accessor.activate()
                        act = True
                    except Exception:
                        act = False
            return act
        if is_act_attr is not None:
            return bool(is_act_attr)
        return True

    def _facc_num_pixels(self, facc: Any) -> int:
        npix_val = getattr(facc, "npixels", None)
        if npix_val is None or (isinstance(npix_val, (list, tuple, np.ndarray)) and len(npix_val) == 0):
            npix_val = getattr(facc, "num_pixels", None)
        if npix_val is None:
            npix_val = getattr(facc, "npix", 0)
        if isinstance(npix_val, np.ndarray):
            return int(np.sum(npix_val)) if npix_val.size > 0 else 0
        return int(npix_val) if npix_val is not None else 0

    def _facc_offset(self, facc: Any) -> int:
        if hasattr(facc, "pix_position"):
            pos = getattr(facc, "pix_position")
            if pos is not None and not (isinstance(pos, (list, tuple, np.ndarray)) and len(pos) == 0):
                return int(pos)
        return int(getattr(facc, "offset", 1))

    def _facc_tail(self, facc: Any) -> int:
        if hasattr(facc, "eof_pix_position") and hasattr(facc, "pix_position"):
            eof_p = getattr(facc, "eof_pix_position")
            pix_p = getattr(facc, "pix_position")
            npix = self._facc_num_pixels(facc)
            if (
                eof_p is not None
                and pix_p is not None
                and not (isinstance(eof_p, (list, tuple, np.ndarray)) and len(eof_p) == 0)
                and not (isinstance(pix_p, (list, tuple, np.ndarray)) and len(pix_p) == 0)
            ):
                return max(0, int(eof_p) - (int(pix_p) + npix * 36))
        return int(getattr(facc, "tail", 0))

    def init(self, arg: Any, writable: bool = False) -> "PixelDataFileBacked":
        """Initialize ``PixelDataFileBacked`` from a file, accessor, array, or ``PixelDataBase``."""
        if arg is None:
            return self

        if isinstance(arg, PixelDataFileBacked):
            if arg._f_accessor is not None:
                if isinstance(arg._f_accessor, FileBackedPixStub):
                    self._f_accessor = arg._f_accessor.clone()
                elif hasattr(arg._f_accessor, "copy_contents"):
                    facc = type(arg._f_accessor)()
                    mode_args = ["-write_mode"] if not arg.read_only else []
                    res = facc.copy_contents(arg._f_accessor, *mode_args)
                    facc = res[0] if isinstance(res, tuple) else res
                    self._f_accessor = facc
                else:
                    self._f_accessor = arg._f_accessor
                self._f_accessor_activated = arg._f_accessor_activated
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

        if isinstance(arg, FileBackedPixStub):
            self._f_accessor = arg
            self._f_accessor_activated = bool(arg.is_activated)
            self._num_pixels = arg.num_pixels
            self._offset = arg.offset
            self._tail = arg.tail
            self._full_filename = arg.full_filename
            self._page_num = 1
            self.recalc_data_range("all")
            return self

        if isinstance(arg, (str, os.PathLike)):
            filepath = str(arg)
            facc = _open_faccess_for_file(filepath, writable=writable)
            return self.init(facc, writable=writable)

        from pyhorace.io.horace_binfile_interface import horace_binfile_interface
        from pyhorace.io.sqw_file_interface import sqw_file_interface

        if isinstance(arg, (sqw_file_interface, horace_binfile_interface)) or hasattr(arg, "get_pix"):
            if isinstance(arg, (sqw_file_interface, horace_binfile_interface)) and hasattr(arg, "copy_contents"):
                # Clone the file accessor so closing the caller's loader does not close our handle
                is_act = arg.is_activated("read") if callable(getattr(arg, "is_activated", None)) else True
                if is_act:
                    facc = type(arg)()
                    mode_args = ["-write_mode"] if writable else []
                    res = facc.copy_contents(arg, *mode_args)
                    facc = res[0] if isinstance(res, tuple) else res
                else:
                    facc = arg
                    if writable and hasattr(facc, "reopen_to_write"):
                        facc.reopen_to_write()
            elif hasattr(arg, "get_pix") and hasattr(arg, "get_pix_at_indices") and hasattr(arg, "get_pix_in_ranges"):
                facc = arg
                if writable and hasattr(facc, "reopen_to_write"):
                    facc.reopen_to_write()
            else:
                facc = FileBackedPixStub.from_faccess(arg, writable=writable)

            self._f_accessor = facc
            self._f_accessor_activated = True
            self._num_pixels = self._facc_num_pixels(facc)
            self._offset = self._facc_offset(facc)
            self._tail = self._facc_tail(facc)
            self._full_filename = getattr(facc, "full_filename", "") or getattr(arg, "full_filename", "")
            self._page_num = 1

            if hasattr(arg, "faccess_version"):
                try:
                    self._old_file_format = float(arg.faccess_version) < 4.0
                except Exception:
                    pass

            if hasattr(arg, "get_pix_metadata") and getattr(arg, "faccess_version", 0) >= 4:
                try:
                    meta = arg.get_pix_metadata(nargout=1)
                    if isinstance(meta, pix_metadata):
                        self._data_range = meta.data_range.copy()
                        self._alignment_matr = meta.alignment_matr.copy()
                        self._is_corrected = meta.is_corrected
                except Exception:
                    pass
            elif hasattr(arg, "get_pix_range"):
                try:
                    pr = np.asarray(arg.get_pix_range(), dtype=np.float64)
                    if pr.shape == (2, 4) and not np.any(np.isinf(pr)):
                        self._data_range[:, :4] = pr
                except Exception:
                    pass
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

        if hasattr(arg, "npixels") or hasattr(arg, "npix") or hasattr(arg, "fake_data"):
            stub = FileBackedPixStub.from_faccess(arg, writable=writable)
            self._f_accessor = stub
            self._f_accessor_activated = bool(stub.is_activated)
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
    def f_accessor(self) -> Optional[Any]:
        return self._f_accessor

    @property
    def offset(self) -> int:
        if self._f_accessor is not None:
            return self._facc_offset(self._f_accessor)
        return self._offset

    @property
    def tail(self) -> int:
        if self._f_accessor is not None:
            return self._facc_tail(self._f_accessor)
        return self._tail

    def copy(self) -> "PixelDataFileBacked":
        out = PixelDataFileBacked()
        out.init(self)
        out._page_num = 1
        return out

    def delete(self) -> None:
        """Close the file accessor and delete temporary file if owned."""
        if self._f_accessor is not None:
            self.deactivate()
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
            if hasattr(self._f_accessor, "deactivate"):
                self._f_accessor.deactivate()
            elif hasattr(self._f_accessor, "fclose"):
                self._f_accessor.fclose()
        self._f_accessor_activated = False
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
            if isinstance(self._f_accessor, FileBackedPixInterface):
                self._f_accessor.activate(target_fn, writable=writable)
            else:
                mode_arg = "-write" if writable else "-read"
                if filename:
                    fpath, fname = os.path.split(os.path.abspath(str(filename)))
                    self._f_accessor.filepath = fpath
                    self._f_accessor.filename = fname
                if hasattr(self._f_accessor, "activate"):
                    self._f_accessor.activate(mode_arg)
            self._f_accessor_activated = True
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
        self._f_accessor_activated = True
        self._num_pixels = wh.num_pixels
        self._offset = self._facc_offset(wh.f_accessor)
        self._tail = self._facc_tail(wh.f_accessor)
        self._full_filename = wh.write_FileName
        self._data_range = wh.pix_range.copy()
        self._page_num = 1
        if hasattr(self._f_accessor, "put_pix_metadata"):
            try:
                self._f_accessor.put_pix_metadata(self.metadata)
            except Exception:
                pass
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
            return self._facc_num_pixels(self._f_accessor)
        return self._num_pixels

    def get_read_only(self) -> bool:
        if self._f_accessor is None:
            return True
        if hasattr(self._f_accessor, "read_only"):
            return bool(self._f_accessor.read_only)
        return not getattr(self._f_accessor, "writable", False)

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
    # Property getters/setters via faccess interface
    # (get_pix, get_pix_at_indices, get_pix_in_ranges, put_pix, put_raw_pix)
    # ------------------------------------------------------------------
    def _read_raw_page(self, s_idx: int, e_idx: int) -> np.ndarray:
        """Read raw 9xN pixel slice [s_idx, e_idx] (1-based inclusive) via ``f_accessor.get_pix``."""
        assert self._f_accessor is not None
        if hasattr(self._f_accessor, "get_pix"):
            args: List[Any] = [s_idx, e_idx, "-raw_output"]
            if self._keep_precision:
                args.append("-keep_precision")
            return np.asarray(self._f_accessor.get_pix(*args))
        return np.asarray(self._f_accessor.get_pix_slice(slice(None), s_idx, e_idx))

    def get_prop(self, fld: str) -> np.ndarray:
        idx = self.FIELD_INDEX_MAP[fld]
        idx0 = [i - 1 for i in idx]
        if self._f_accessor is None or not self._check_facc_activated() or self.num_pixels == 0:
            return np.empty(
                (len(idx0), 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )

        s_idx, e_idx = self.get_page_idx_(self._page_num)
        raw_all = self._read_raw_page(s_idx, e_idx).copy()
        if self._is_corrected and any(i < 3 for i in idx0):
            pix_coord = self._alignment_matr @ raw_all[:3, :]
            for c in idx0:
                if c < 3:
                    raw_all[c, :] = pix_coord[c, :]
        out = raw_all[idx0, :]

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
        if hasattr(self._f_accessor, "put_raw_pix") and hasattr(self._f_accessor, "get_pix"):
            page_data = self._read_raw_page(s_idx, e_idx).copy()
            page_data[idx0, :] = val_arr[:, : e_idx - s_idx + 1]
            self._f_accessor.put_raw_pix(page_data, s_idx)
        else:
            self._f_accessor.set_pix_slice(idx0, s_idx, e_idx, val_arr)

        if self._is_corrected:
            self._is_corrected = False
            self._alignment_matr = np.eye(3, dtype=np.float64)

        if self.num_pages == 1:
            self.recalc_data_range(fld)
            if hasattr(self._f_accessor, "put_pix_metadata"):
                try:
                    self._f_accessor.put_pix_metadata(self.metadata)
                except Exception:
                    pass
        else:
            self.invalidate_range(fld)
        return self

    def get_raw_data(self, page_number: Optional[int] = None) -> np.ndarray:
        pg = self._page_num if page_number is None else int(page_number)
        if self._f_accessor is None or not self._check_facc_activated() or self.num_pixels == 0:
            return np.empty(
                (self.DEFAULT_NUM_PIX_FIELDS, 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )
        s_idx, e_idx = self.get_page_idx_(pg)
        out = self._read_raw_page(s_idx, e_idx)
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
        if (
            self._f_accessor is not None
            and not self.read_only
            and arr.ndim == 2
            and arr.shape[0] == self.DEFAULT_NUM_PIX_FIELDS
            and arr.shape[1] == self.num_pixels
            and hasattr(self._f_accessor, "put_pix")
        ):
            self._f_accessor.put_pix(arr)
            self.recalc_data_range("all")
            return self
        self._init_from_numeric(arr)
        return self

    def get_raw_pix_data(
        self,
        row_pix_idx: np.ndarray,
        col_pix_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._f_accessor is None or not self._check_facc_activated() or self.num_pixels == 0:
            n_rows = 9 if col_pix_idx is None else len(col_pix_idx)
            return np.empty(
                (n_rows, 0),
                dtype=np.float32 if self._keep_precision else np.float64,
            )
        indices = np.asarray(row_pix_idx, dtype=np.intp)
        try:
            raw = np.asarray(self._f_accessor.get_pix_at_indices(indices))
            if col_pix_idx is not None:
                col0 = np.asarray(col_pix_idx, dtype=np.intp).ravel()
                out = raw[col0, :]
            else:
                out = raw
        except TypeError:
            fields: Union[slice, np.ndarray] = (
                slice(None) if col_pix_idx is None else np.asarray(col_pix_idx, dtype=np.intp)
            )
            out = self._f_accessor.get_pix_at_indices(fields, indices)
        if self._keep_precision:
            return np.asarray(out, dtype=np.float32)
        return np.asarray(out, dtype=np.float64)

    def get_pix_in_ranges(
        self,
        pix_starts: Any,
        block_sizes: Any,
        skip_arg_validation: bool = False,
        keep_precision: bool = False,
    ) -> PixelDataBase:
        """Read pixels in the specified 1-based start/size ranges via ``f_accessor.get_pix_in_ranges``."""
        from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

        if self._f_accessor is None or not self._check_facc_activated() or self.num_pixels == 0:
            return PixelDataMemory()
        kp = bool(keep_precision or self._keep_precision)
        raw = np.asarray(
            self._f_accessor.get_pix_in_ranges(
                pix_starts,
                block_sizes,
                skip_validation=skip_arg_validation,
                keep_precision=kp,
            )
        )
        if self._is_corrected and raw.shape[1] > 0:
            raw = raw.copy()
            raw[:3, :] = self._alignment_matr @ raw[:3, :]
        out = PixelDataMemory(raw)
        out._keep_precision = kp
        return out

    def recalc_data_range(
        self, fld_names: Union[str, Sequence[str], np.ndarray] = "all"
    ) -> "PixelDataFileBacked":
        ind = self.field_index(fld_names)
        idx0 = ind - 1
        self._data_range[:, idx0] = self.EMPTY_RANGE[:, idx0]
        if self._f_accessor is None or not self._check_facc_activated() or self.num_pixels == 0:
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
        if isinstance(val.data, str):
            self._num_pixels = val.npix
            if os.path.isfile(val.data):
                self._f_accessor = _open_faccess_for_file(val.data)
                self._f_accessor_activated = True
                self._full_filename = val.data
            return self
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
            self._f_accessor = _open_faccess_for_file(val.full_filename)
            self._f_accessor_activated = True
            self._num_pixels = self._facc_num_pixels(self._f_accessor)
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
