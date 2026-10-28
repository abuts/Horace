"""Temporary file write handle for ``PixelDataFileBacked`` (translated from ``pix_write_handle.m``)."""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import Any, Optional

import numpy as np

from pyhorace.data.pixels.pix_metadata import pix_metadata


class pix_write_handle:
    """Helper class managing paged writes of pixels to a temporary backing file via ``faccess_sqw_v4``."""

    def __init__(
        self,
        num_pix_or_source: Any = 0,
        keep_precision: bool = False,
    ) -> None:
        from pyhorace.io.faccess_sqw_v4 import faccess_sqw_v4

        self._keep_precision: bool = bool(keep_precision)
        self.is_tmp_obj: bool = True
        self.pix_range: np.ndarray = pix_metadata.EMPTY_RANGE.copy()
        self.num_pixels: int = 0

        base_dir = tempfile.gettempdir()
        if isinstance(num_pix_or_source, str) and num_pix_or_source:
            base_dir = os.path.dirname(os.path.abspath(num_pix_or_source)) or base_dir
        elif hasattr(num_pix_or_source, "filepath") and getattr(num_pix_or_source, "filepath"):
            fp = getattr(num_pix_or_source, "filepath")
            if os.path.isdir(fp):
                base_dir = fp

        tmp_name = f"sqw_pix_{uuid.uuid4().hex}.tmp_pix"
        self.write_FileName: str = os.path.join(base_dir, tmp_name)
        facc = faccess_sqw_v4()
        facc.init_pix_file(self.write_FileName, npixels=0)
        self.f_accessor: Any = facc

    def store_page(
        self,
        data: np.ndarray,
        pos_start: Optional[int] = None,
    ) -> np.ndarray:
        """Write a page of pixel data to the backing file via ``f_accessor.put_raw_pix`` and update ``pix_range``."""
        arr = np.asarray(data)
        if arr.size == 0:
            return arr
        if arr.ndim == 1:
            arr = arr.reshape(9, -1, order="F")

        if not self._keep_precision:
            arr_single = np.asarray(arr, dtype=np.float32)
            range_src = np.asarray(arr_single, dtype=np.float64)
            ret_data = np.asarray(arr_single, dtype=np.float64)
        else:
            arr_single = np.asarray(arr, dtype=np.float32)
            range_src = np.asarray(arr, dtype=np.float64)
            ret_data = range_src

        n_page = arr.shape[1]
        start_pix = (self.num_pixels + 1) if pos_start is None else int(pos_start)
        end_pix = start_pix + n_page - 1

        self.f_accessor.put_raw_pix(arr_single, start_pix)
        self.num_pixels = max(self.num_pixels, end_pix)
        if hasattr(self.f_accessor, "put_num_pixels"):
            self.f_accessor.put_num_pixels(self.num_pixels)

        page_min = np.min(range_src, axis=1)
        page_max = np.max(range_src, axis=1)
        self.pix_range[0, :] = np.minimum(self.pix_range[0, :], page_min)
        self.pix_range[1, :] = np.maximum(self.pix_range[1, :], page_max)
        return ret_data

    def save_data(
        self,
        data: np.ndarray,
        pos_start: Optional[int] = None,
    ) -> np.ndarray:
        """Alias for :meth:`store_page` matching MATLAB ``pix_write_handle.save_data``."""
        return self.store_page(data, pos_start=pos_start)

    def write_pixels(
        self,
        data: np.ndarray,
        pos_start: Optional[int] = None,
    ) -> np.ndarray:
        """Alias for :meth:`store_page`."""
        return self.store_page(data, pos_start=pos_start)

    def finalize(self, target_cls_or_obj: Any = None) -> Any:
        """Finalize the write handle into a :class:`PixelDataFileBacked` instance."""
        from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked

        if target_cls_or_obj is None:
            fb = PixelDataFileBacked()
        elif isinstance(target_cls_or_obj, type):
            fb = target_cls_or_obj()
        else:
            fb = target_cls_or_obj
        fb.finish_dump(self)
        return fb

    def close(self) -> None:
        """Close the write handle."""
        if self.f_accessor is not None:
            if hasattr(self.f_accessor, "deactivate"):
                self.f_accessor.deactivate()
            elif hasattr(self.f_accessor, "fclose"):
                self.f_accessor.fclose()
