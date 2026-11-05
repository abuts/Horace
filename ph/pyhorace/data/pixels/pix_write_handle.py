"""
Write handle for dumping pixel data pages to disk.

Ported from ``horace_core/sqw/PixelData/pix_write_handle.m``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np

from pyhorace.data.pixels.filebacked_pix_interface import (
    FileBackedPixInterface,
    FileBackedPixStub,
)
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
def _is_binfile_accessor(obj: Any) -> bool:
    if obj is None or isinstance(obj, (str, bytes, os.PathLike, int, float, np.number, FileBackedPixInterface)):
        return False
    from pyhorace.io.horace_binfile_interface import horace_binfile_interface

    return isinstance(obj, horace_binfile_interface)


class pix_write_handle:
    """
    Wraps write access handles for writing pixels (either via a
    ``horace_binfile_interface`` file accessor or :class:`FileBackedPixStub`)
    and provides a common interface for writing pixel and image pages.
    """

    def __init__(
        self,
        num_pix_or_source: Any = "in_mem",
        keep_precision: bool = False,
    ) -> None:
        self.write_FileName: str = ""
        self.num_pixels: int = 0
        self.keep_precision: bool = bool(keep_precision)
        self.handle_is_class: bool = False
        self.f_accessor: Any = None
        self.move_to_original: bool = False
        self.img_start_pos_: int = 0
        self.delete_target_file_: bool = True
        self._is_tmp_file: Optional[bool] = None
        self.pix_range: np.ndarray = PixelDataBase.EMPTY_RANGE.copy()
        self._closed: bool = False

        if _is_binfile_accessor(num_pix_or_source):
            self.f_accessor = num_pix_or_source
            self.handle_is_class = True
            self.write_FileName = str(num_pix_or_source.full_filename)
            return

        if isinstance(num_pix_or_source, FileBackedPixInterface):
            self.f_accessor = num_pix_or_source
            self.handle_is_class = False
            self.write_FileName = str(num_pix_or_source.full_filename)
            self._is_tmp_file = os.path.splitext(self.write_FileName)[1].startswith(
                ".tmp"
            )
            return

        if isinstance(num_pix_or_source, (int, np.integer)):
            self.write_FileName = PixelDataBase.build_tmp_file_name("in_mem")
            self._is_tmp_file = True
        elif isinstance(num_pix_or_source, (str, os.PathLike)):
            src = str(num_pix_or_source)
            if src == "" or src == "in_mem":
                self.write_FileName = PixelDataBase.build_tmp_file_name("in_mem")
                self._is_tmp_file = True
            else:
                _, ext = os.path.splitext(src)
                if ext.startswith(".tmp"):
                    self.write_FileName = os.path.abspath(src)
                    self._is_tmp_file = True
                elif os.path.isfile(src):
                    self.write_FileName = PixelDataBase.build_tmp_file_name(src)
                    self._is_tmp_file = True
                else:
                    self.write_FileName = os.path.abspath(src)
                    self._is_tmp_file = False
        elif hasattr(num_pix_or_source, "put_raw_pix"):
            self.f_accessor = num_pix_or_source
            self.handle_is_class = True
            self.write_FileName = getattr(num_pix_or_source, "full_filename", "")
            return
        else:
            raise ValueError(
                f"HORACE:PixelData:invalid_argument: pix_write_handle accepts a class which can "
                f"read/write pixel data or the name of the file containing these data. "
                f"Provided class {type(num_pix_or_source).__name__}"
            )

        self.f_accessor = FileBackedPixStub(
            filename=self.write_FileName,
            num_pixels=0,
            offset=1,
            tail=0,
            writable=True,
            create_file=True,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def write_file_name(self) -> str:
        return self.write_FileName

    @write_file_name.setter
    def write_file_name(self, val: str) -> None:
        self.write_FileName = str(val)

    @property
    def npix_written(self) -> int:
        return self.num_pixels

    @npix_written.setter
    def npix_written(self, val: int) -> None:
        self.num_pixels = int(val)

    @property
    def write_handle(self) -> Any:
        return self.f_accessor

    @property
    def pixout_start(self) -> int:
        if not self.handle_is_class:
            raise RuntimeError(
                "HORACE:pix_write_handle:runtime_error: Attempt to get pix position from incompatible binary file"
            )
        return int(self.f_accessor.pix_position)

    @property
    def is_tmp_file(self) -> bool:
        if self._is_tmp_file is None:
            _, ext = os.path.splitext(self.write_FileName)
            return ext.startswith(".tmp")
        return self._is_tmp_file

    @is_tmp_file.setter
    def is_tmp_file(self, val: bool) -> None:
        self._is_tmp_file = bool(val)

    @property
    def is_tmp_obj(self) -> bool:
        return self.is_tmp_file

    @is_tmp_obj.setter
    def is_tmp_obj(self, val: bool) -> None:
        self.is_tmp_file = val

    # =========================================================================
    # Write / Finalize Methods
    # =========================================================================

    def save_img_chunk(
        self, img_struc: dict, start_pos: Optional[int] = None
    ) -> None:
        """
        Store part of image ``{"s": s, "e": e, "npix": npix}`` changed due to
        pixel modifications at ``start_pos`` (0-based bin offset) in the image
        on disk.
        """
        if not self.handle_is_class:
            raise NotImplementedError(
                "HORACE:pix_write_handle:not_implemented: writing image using direct IO "
                "operations is not implemented. Use faccess_*** classes to modify image."
            )
        if start_pos is None:
            start_pos = self.img_start_pos_
        self.f_accessor.put_senpix_block(img_struc, start_pos)
        self.img_start_pos_ = int(start_pos) + int(np.asarray(img_struc["npix"]).size)

    def save_data(
        self, data: Any, start_pos: Optional[int] = None
    ) -> pix_write_handle:
        """
        Write a block of pixel data ``(9, N)`` to the underlying file or accessor.
        """
        raw = getattr(data, "data", data)
        arr = np.asarray(raw, dtype=np.float32 if self.keep_precision else np.float64)
        if arr.size == 0:
            return self
        if arr.ndim == 1 and arr.size == PixelDataBase.DEFAULT_NUM_PIX_FIELDS:
            arr = arr.reshape((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, 1))
        if arr.ndim != 2 or arr.shape[0] != PixelDataBase.DEFAULT_NUM_PIX_FIELDS:
            raise ValueError(
                f"Expected (9, N) pixel array, got shape {arr.shape}"
            )

        n_new = int(arr.shape[1])
        if start_pos is None:
            start_pos = self.num_pixels + 1

        if self.handle_is_class and self.f_accessor is not None:
            res = self.f_accessor.put_raw_pix(arr, start_pos)
            if res is not None:
                self.f_accessor = res
        elif isinstance(self.f_accessor, FileBackedPixInterface):
            end_pos = int(start_pos) + n_new - 1
            self.f_accessor.set_pix_slice(slice(None), int(start_pos), end_pos, arr)
        else:
            raise RuntimeError("pix_write_handle has already been closed.")

        self.num_pixels = max(self.num_pixels, int(start_pos) + n_new - 1)
        f32_arr = np.asarray(arr, dtype=np.float32).astype(np.float64)
        self.pix_range = PixelDataBase.pix_minmax_ranges(f32_arr, self.pix_range)
        return self

    def store_page(self, page_data: np.ndarray) -> pix_write_handle:
        """Append a ``9 x N`` page of pixel data to the target file."""
        return self.save_data(page_data)

    def write_pixels(self, page_data: np.ndarray) -> pix_write_handle:
        """Alias for :meth:`store_page`."""
        return self.store_page(page_data)

    def finalize(self, target_cls: Any = None) -> Any:
        """Finalize write handle into a ``PixelDataFileBacked`` instance."""
        if target_cls is None:
            from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked

            target_cls = PixelDataFileBacked
        obj = target_cls()
        return obj.finish_dump(self)

    def finish_pix_dump(
        self, pix_obj: Any, store_metadata: bool = True
    ) -> Optional[pix_metadata]:
        """
        Finalize multipage pixel write operation by recording the total number
        of written pixels and optionally updating ``bl_pix_metadata`` in the
        target SQW file.
        """
        if self.handle_is_class and self.f_accessor is not None:
            num_pixels = self.npix_written
            wh = self.f_accessor
            res = wh.put_num_pixels(num_pixels)
            if res is not None:
                wh = res

            if hasattr(pix_obj, "_pix_data_range"):
                pix_src = pix_obj.pix
                pix_src.set_data_range(pix_obj._pix_data_range)
                meta = pix_src.metadata
            elif hasattr(pix_obj, "metadata"):
                meta = pix_obj.metadata
            else:
                meta = pix_metadata(pix_obj)
            meta.full_filename = wh.full_filename
            meta.npix = num_pixels

            if store_metadata:
                res = wh.put_new_blocks_values(meta, "update", "bl_pix_metadata")
                if res is not None:
                    wh = res
            self.f_accessor = wh
            return meta

        if pix_obj is not None:
            if hasattr(pix_obj, "_pix_data_range"):
                pix_src = pix_obj.pix
                pix_src.set_data_range(pix_obj._pix_data_range)
                meta = pix_src.metadata
            elif hasattr(pix_obj, "metadata"):
                meta = pix_obj.metadata
            else:
                meta = pix_metadata(pix_obj)
            meta.full_filename = self.write_FileName
            meta.npix = self.num_pixels
            return meta
        return None

    def release_pixinit_info(self, keep_opened: bool = False) -> Any:
        """
        Return initialization info needed to construct a ``PixelDataFileBacked``
        over the written file and release ownership of the target file so it is
        not deleted on garbage collection.
        """
        init_info = self.f_accessor
        if self.handle_is_class and init_info is not None:
            _ = getattr(init_info, "eof_position", 0)
            if not keep_opened:
                self.close_handles()
        self.f_accessor = None
        self._closed = True
        self.delete_target_file_ = False
        return init_info

    def close_handles(self) -> pix_write_handle:
        """Close underlying file or accessor handle."""
        if self._closed:
            return self
        if self.f_accessor is not None:
            if self.handle_is_class:
                if hasattr(self.f_accessor, "delete"):
                    self.f_accessor.delete()
                elif hasattr(self.f_accessor, "fclose"):
                    self.f_accessor.fclose()
                self.f_accessor = None
            elif hasattr(self.f_accessor, "deactivate"):
                self.f_accessor.deactivate()
        self._closed = True
        return self

    def close(self) -> None:
        self.close_handles()

    def delete(self) -> None:
        """Close handles and delete incomplete temporary target file if not released."""
        self.close_handles()
        if self.delete_target_file_ and self.write_FileName and os.path.isfile(self.write_FileName):
            try:
                os.remove(self.write_FileName)
            except OSError:
                pass
            self.delete_target_file_ = False

    def __del__(self) -> None:
        try:
            self.delete()
        except Exception:
            pass
