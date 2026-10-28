"""In-memory pixel data class (translated from ``@PixelDataMemory``)."""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.data.pixels.filebacked_pix_interface import (
    FileBackedPixInterface,
    FileBackedPixStub,
)
from pyhorace.data.pixels.pix_data import pix_data
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


class PixelDataMemory(PixelDataBase):
    """In-memory implementation of :class:`PixelDataBase`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._data: np.ndarray = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            self._init_from_single_arg(args[0])
            return

        # Support (data_wrap, metadata) or keyword arguments
        self.set_positional_and_key_val_arguments(
            self.saveable_fields(), False, *args, **kwargs
        )

    def _init_from_single_arg(self, arg: Any) -> None:
        if arg is None:
            return

        if isinstance(arg, PixelDataMemory):
            self._data = arg._data.copy()
            self._data_range = arg._data_range.copy()
            self._full_filename = arg._full_filename
            self._alignment_matr = arg._alignment_matr.copy()
            self._is_corrected = arg._is_corrected
            self._old_file_format = arg._old_file_format
            self._unique_run_id = arg._unique_run_id.copy()
            self._keep_precision = arg._keep_precision
            return

        if isinstance(arg, PixelDataBase):
            # Copy from file-backed PixelDataBase into memory
            if arg.num_pixels == 0:
                self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
            else:
                pages = []
                saved_pg = arg.page_num
                try:
                    for pg in range(1, arg.num_pages + 1):
                        arg.page_num = pg
                        pages.append(np.asarray(arg.get_raw_data(pg), dtype=np.float64))
                finally:
                    arg.page_num = saved_pg
                self._data = np.hstack(pages) if pages else np.empty((9, 0), dtype=np.float64)
            self._full_filename = arg.full_filename
            self._alignment_matr = arg.alignment_matr.copy()
            self._is_corrected = arg.is_corrected
            if arg.is_range_valid("all"):
                self._data_range = arg.data_range.copy()
            else:
                self.recalc_data_range("all")
            return

        if isinstance(arg, pix_data):
            self.set_data_wrap(arg)
            return

        if isinstance(arg, pix_metadata):
            self.set_metadata(arg)
            return

        if isinstance(arg, (str, os.PathLike)):
            filepath = str(arg)
            if not os.path.isfile(filepath):
                raise HoraceError(
                    f"HORACE:PixelDataMemory:invalid_argument - "
                    f"Cannot find file '{filepath}' to initialize PixelDataMemory."
                )
            try:
                from pyhorace.io.sqw_formats_factory import sqw_formats_factory

                ldr = sqw_formats_factory.instance().get_loader(filepath)
                try:
                    self._init_from_single_arg(ldr)
                    return
                finally:
                    ldr.fclose()
            except Exception:
                stub = FileBackedPixStub(filepath)
                self._full_filename = stub.full_filename
                self._data = stub.get_pix(1, stub.num_pixels, "-raw_output")
                self.recalc_data_range("all")
                return

        if isinstance(arg, FileBackedPixStub):
            self._full_filename = arg.full_filename
            if hasattr(arg, "get_pix"):
                self._data = np.asarray(arg.get_pix(1, arg.num_pixels, "-raw_output"), dtype=np.float64)
            else:
                self._data = arg.get_pix_slice(slice(None), 1, arg.num_pixels)
            self.recalc_data_range("all")
            return

        if hasattr(arg, "get_pix") or hasattr(arg, "npix") or hasattr(arg, "npixels"):
            if hasattr(arg, "faccess_version"):
                try:
                    if float(arg.faccess_version) < 4.0:
                        self._old_file_format = True
                except Exception:
                    pass
            if hasattr(arg, "get_pix") and hasattr(arg, "npixels"):
                npix_val = getattr(arg, "npixels", 0)
                n_pix = int(npix_val) if npix_val is not None and not (isinstance(npix_val, (list, tuple, np.ndarray)) and len(npix_val) == 0) else 0
                self._full_filename = getattr(arg, "full_filename", "")
                if n_pix == 0:
                    self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
                else:
                    try:
                        self._data = np.asarray(
                            arg.get_pix(1, n_pix, "-raw_output"), dtype=np.float64
                        )
                    except TypeError:
                        raw = arg.get_pix(1, n_pix)
                        self._data = np.asarray(
                            raw.data if isinstance(raw, PixelDataBase) else raw,
                            dtype=np.float64,
                        )
            else:
                stub = FileBackedPixStub.from_faccess(arg)
                self._full_filename = stub.full_filename
                self._data = stub.get_pix_slice(slice(None), 1, stub.num_pixels)
            self.recalc_data_range("all")
            if self._old_file_format and self.num_pixels > 0:
                self._unique_run_id = np.unique(self.run_idx).astype(float)
            return

        if isinstance(arg, dict):
            loaded = PixelDataBase.loadobj(arg)
            self._init_from_single_arg(loaded)
            return

        if isinstance(arg, (int, float, np.number)) and not isinstance(arg, bool):
            fval = float(arg)
            ival = int(fval)
            if fval < 0 or fval != ival or np.isnan(fval) or np.isinf(fval):
                raise HoraceError(
                    f"HORACE:PixelDataMemory:invalid_argument - "
                    f"Scalar pixel count must be a non-negative integer, got {arg!r}"
                )
            self._data = np.zeros((self.DEFAULT_NUM_PIX_FIELDS, ival), dtype=np.float64)
            self.recalc_data_range("all")
            return

        if isinstance(arg, (np.ndarray, list, tuple)):
            arr = np.asarray(arg)
            if arr.dtype == object or np.issubdtype(arr.dtype, np.character) or arr.dtype == bool:
                raise HoraceError(
                    f"HORACE:PixelDataMemory:invalid_argument - "
                    f"Cannot construct PixelDataMemory from array of dtype {arr.dtype}"
                )
            if arr.ndim == 0 or (arr.size == 1 and arr.ndim <= 2 and arr.shape != (9, 1)):
                self._init_from_single_arg(arr.item())
                return
            self.data = arr
            return

        raise HoraceError(
            f"HORACE:PixelDataMemory:invalid_argument - "
            f"Cannot construct PixelDataMemory from {type(arg).__name__}"
        )

    @property
    def is_filebacked(self) -> bool:
        return False

    def copy(self) -> "PixelDataMemory":
        out = PixelDataMemory()
        out._data = self._data.copy()
        out._data_range = self._data_range.copy()
        out._full_filename = self._full_filename
        out._alignment_matr = self._alignment_matr.copy()
        out._is_corrected = self._is_corrected
        out._old_file_format = self._old_file_format
        out._unique_run_id = self._unique_run_id.copy()
        out._keep_precision = self._keep_precision
        out._base_page_size = self._base_page_size
        return out

    def get_num_pixels(self) -> int:
        return int(self._data.shape[1])

    def get_prop(self, fld: str) -> np.ndarray:
        idx = self.FIELD_INDEX_MAP[fld]
        idx0 = [i - 1 for i in idx]
        if self._data.shape[1] == 0:
            return np.empty((len(idx0), 0), dtype=np.float64)
        if self._is_corrected and any(i < 3 for i in idx0):
            data = self._data.copy()
            pix_coord = self._alignment_matr @ data[:3, :]
            for c in idx0:
                if c < 3:
                    data[c, :] = pix_coord[c, :]
            return data[idx0, :]
        return self._data[idx0, :].copy()

    def set_prop(self, fld: str, val: Any) -> "PixelDataMemory":
        val_arr = self.check_set_prop(fld, val)
        idx = self.FIELD_INDEX_MAP[fld]
        idx0 = [i - 1 for i in idx]
        self._data[idx0, :] = val_arr

        if self._is_corrected:
            self._is_corrected = False
            self._alignment_matr = np.eye(3, dtype=np.float64)

        self.calc_page_range(fld)
        return self

    def get_raw_data(self, page_number: Optional[int] = None) -> np.ndarray:
        return self._data

    def set_raw_data(self, pix_data_arr: Any) -> "PixelDataMemory":
        if pix_data_arr is None:
            self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
            return self
        if isinstance(pix_data_arr, (str, bytes)):
            raise HoraceError(
                "HORACE:PixelDataMemory:invalid_argument - "
                "Pixel data must be a numeric array, got string."
            )
        if isinstance(pix_data_arr, (int, float, np.number)) and not isinstance(pix_data_arr, bool):
            fval = float(pix_data_arr)
            ival = int(fval)
            if fval < 0 or fval != ival or np.isnan(fval) or np.isinf(fval):
                raise HoraceError(
                    f"HORACE:PixelDataMemory:invalid_argument - "
                    f"Scalar pixel count must be a non-negative integer, got {pix_data_arr!r}"
                )
            self._data = np.zeros((self.DEFAULT_NUM_PIX_FIELDS, ival), dtype=np.float64)
            return self

        arr = np.asarray(pix_data_arr)
        if not np.issubdtype(arr.dtype, np.number):
            raise HoraceError(
                f"HORACE:PixelDataMemory:invalid_argument - "
                f"Pixel data must be numeric, got {arr.dtype}"
            )
        if arr.size == 0:
            if arr.ndim == 2 and arr.shape[0] not in (0, self.DEFAULT_NUM_PIX_FIELDS) and arr.shape[1] != 0:
                raise HoraceError(
                    f"HORACE:PixelDataMemory:invalid_argument - "
                    f"Empty pixel data has invalid shape {arr.shape}"
                )
            self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
            return self
        if arr.ndim == 1 and arr.size == self.DEFAULT_NUM_PIX_FIELDS:
            arr = arr.reshape((self.DEFAULT_NUM_PIX_FIELDS, 1))
        if arr.ndim != 2 or arr.shape[0] != self.DEFAULT_NUM_PIX_FIELDS:
            raise HoraceError(
                f"HORACE:PixelDataMemory:invalid_argument - "
                f"Pixel data array must be 2D with {self.DEFAULT_NUM_PIX_FIELDS} rows, got shape {arr.shape}"
            )
        self._data = np.asarray(arr, dtype=np.float64).copy()
        return self

    def get_raw_pix_data(
        self,
        row_pix_idx: np.ndarray,
        col_pix_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self._data.size == 0:
            n_rows = 9 if col_pix_idx is None else len(col_pix_idx)
            return np.empty((n_rows, 0), dtype=np.float64)
        idx0 = np.asarray(row_pix_idx, dtype=np.intp).ravel() - 1
        if col_pix_idx is None:
            return self._data[:, idx0].copy()
        col0 = np.asarray(col_pix_idx, dtype=np.intp).ravel()
        return self._data[np.ix_(col0, idx0)].copy()

    def calc_page_range(
        self, field_name: Union[str, Sequence[str], np.ndarray] = "all"
    ) -> Tuple["PixelDataMemory", np.ndarray]:
        ind = self.field_index(field_name)
        idx0 = ind - 1
        if self._data.size == 0:
            self._data_range[:, idx0] = self.EMPTY_RANGE[:, idx0]
            return self, np.array([], dtype=np.float64)
        cur_data = self.get_data()
        self._data_range[:, idx0] = self.pix_minmax_ranges(cur_data[idx0, :])
        self._unique_run_id = np.unique(self._data[4, :])
        return self, self._unique_run_id

    def recalc_data_range(
        self, fld_names: Union[str, Sequence[str], np.ndarray] = "all"
    ) -> "PixelDataMemory":
        self.calc_page_range(fld_names)
        return self

    def _align_pixels(self, _fld: str = "q_coordinates") -> "PixelDataMemory":
        if self._data.shape[1] > 0:
            q_aligned = self._alignment_matr @ self._data[:3, :]
            self._data[:3, :] = q_aligned
            self._data_range[:, :3] = self.pix_minmax_ranges(q_aligned)
        return self

    def set_alignment_matrix(self, val: Any) -> "PixelDataMemory":
        _, alignment_changed = self._set_alignment_matr_internal(val, self._align_pixels)
        if alignment_changed:
            self._alignment_matr = np.eye(3, dtype=np.float64)
            self._is_corrected = False
        return self

    def set_data_wrap(self, val: Any) -> "PixelDataMemory":
        if not isinstance(val, pix_data):
            raise HoraceError(
                f"HORACE:PixelDataMemory:invalid_argument - "
                f"pix_data_wrap property can only be set with a pix_data instance, got {type(val).__name__}"
            )
        self.set_raw_data(val.data)
        return self

    def append(self, pix: PixelDataBase) -> "PixelDataMemory":
        """Append pixels from ``pix`` to this object and return it."""
        if not isinstance(pix, PixelDataBase):
            raise HoraceError(
                f"HORACE:PixelDataMemory:invalid_argument - "
                f"Input to append must be a PixelDataBase object, got {type(pix).__name__}"
            )
        if pix.num_pixels == 0:
            return self
        other_data = pix.get_pixels(np.arange(1, pix.num_pixels + 1, dtype=np.intp), "-raw_data")
        if self.num_pixels == 0:
            self._data = np.asarray(other_data, dtype=np.float64).copy()
        else:
            self._data = np.hstack([self._data, np.asarray(other_data, dtype=np.float64)])
        self.recalc_data_range("all")
        return self

    def tag(self, selected: Any = None) -> "PixelDataMemory":
        """Tag selected pixels by negating their ``detector_idx``."""
        if selected is None:
            self.detector_idx = -self.detector_idx
        else:
            sel = np.asarray(selected)
            det = self.detector_idx.copy()
            if sel.dtype == bool or np.issubdtype(sel.dtype, np.bool_):
                det[0, sel.ravel()] = -det[0, sel.ravel()]
            else:
                idx0 = sel.ravel().astype(np.intp) - 1
                det[0, idx0] = -det[0, idx0]
            self.detector_idx = det
        return self

    def untag(self, selected: Any = None) -> "PixelDataMemory":
        """Untag selected pixels by taking ``abs(detector_idx)``."""
        if selected is None:
            self.detector_idx = np.abs(self.detector_idx)
        else:
            sel = np.asarray(selected)
            det = self.detector_idx.copy()
            if sel.dtype == bool or np.issubdtype(sel.dtype, np.bool_):
                det[0, sel.ravel()] = np.abs(det[0, sel.ravel()])
            else:
                idx0 = sel.ravel().astype(np.intp) - 1
                det[0, idx0] = np.abs(det[0, idx0])
            self.detector_idx = det
        return self

    def sort(self, idx: Any = None) -> Tuple["PixelDataMemory", np.ndarray]:
        """Sort pixels by ``[abs(all_indexes); u1]`` or by 1-based ``idx``."""
        if idx is not None:
            idx_1based = np.asarray(idx, dtype=np.intp).ravel()
            idx0 = idx_1based - 1
        else:
            s_idx = np.vstack([np.abs(self.all_indexes), self.u1])
            # Sort columns lexicographically by rows 0, 1, 2, 3
            idx0 = np.lexsort((s_idx[3, :], s_idx[2, :], s_idx[1, :], s_idx[0, :]))
            idx_1based = idx0 + 1
        self._data = self._data[:, idx0]
        return self, idx_1based

    def move_to_first_page(self) -> "PixelDataMemory":
        self._page_num = 1
        return self

    @staticmethod
    def apply_op(obj_in: Any, page_op: Any) -> Any:
        """Apply a ``PageOp`` instance to ``obj_in``."""
        page_op.get_page_data(1, [1])
        page_op.common_op(1)
        page_op.update_img_accumulators()
        return page_op.finish_op(obj_in)


register_serializable(PixelDataMemory)

