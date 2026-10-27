"""Accessor for Horace SQW binary format version 3.3.

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v3_3/faccess_sqw_v3_3.m
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.io.faccess_sqw_v3 import faccess_sqw_v3
from pyhorace.io.sqw_fields_formatters import field_not_in_structure, field_pix
from pyhorace.io.sqw_fopen import do_fread, do_fseek, do_ftell, do_fwrite
from pyhorace.io.sqw_serializer import sqw_serializer


class faccess_sqw_v3_3(faccess_sqw_v3):
    """Accessor for Horace SQW binary format version 3.3 (stores pixel data range)."""

    file_version_: float = 3.3

    _v3_3_fields_to_save: List[str] = ["pix_range_pos_"]
    fields_to_save_: List[str] = faccess_sqw_v3.fields_to_save_ + _v3_3_fields_to_save

    def __init__(self, *args: Any) -> None:
        self.version_ = self.file_version_
        self.pix_range_pos_: int = 0
        self.pix_range_: np.ndarray = PixelDataBase.EMPTY_RANGE.copy()
        super().__init__(*args)

    def get_faccess_version(self) -> float:
        return 3.3

    def delete(self) -> "faccess_sqw_v3_3":
        super().delete()
        self.pix_range_pos_ = 0
        self.pix_range_ = PixelDataBase.EMPTY_RANGE.copy()
        return self

    def from_bare_struct(self, indata: Dict[str, Any]) -> "faccess_sqw_v3_3":
        super().from_bare_struct(indata)
        if "pix_range_pos_" in indata:
            val = indata["pix_range_pos_"]
            if isinstance(val, (int, float, np.number)):
                self.pix_range_pos_ = int(val)
            elif isinstance(val, np.ndarray) and val.size == 1:
                self.pix_range_pos_ = int(val.flat[0])
        return self

    @staticmethod
    def get_pix_form() -> Dict[str, Any]:
        return {
            "img_range": np.array([2, 4], dtype=np.float32),
            "dummy": field_not_in_structure("img_range"),
            "pix_block": field_pix(),
            "pix_range": np.array([2, 4], dtype=np.float32),
        }

    def get_pix_range(self) -> np.ndarray:
        return self.pix_range_.copy()

    def get_data_range(self) -> np.ndarray:
        return self.get_pix_range()

    def get_pix_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        meta = pix_metadata()
        meta.full_filename = self.full_filename
        meta.npix = self.npixels if isinstance(self.npixels, int) else 0
        meta.data_range = self._expand_data_range(self.pix_range_)
        if nargout is not None and nargout > 1:
            return meta, self
        return meta

    @staticmethod
    def _expand_data_range(pix_range: np.ndarray) -> np.ndarray:
        pr = np.asarray(pix_range, dtype=np.float64)
        if pr.shape == (2, 9):
            return pr.copy()
        full_range = PixelDataBase.EMPTY_RANGE.copy()
        if pr.shape == (2, 4):
            full_range[:, :4] = pr
        return full_range

    def _init_pix_info(self, *args: Any) -> None:
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()

        base_pix_form = super().get_pix_form()
        from_obj = len(args) > 0 and self._is_sqw_or_dnd_obj(args[0])
        if from_obj:
            inobj = args[0]
            pix_obj, _ = self.extract_correct_subobj("pix", inobj)
            from pyhorace.data.pixels.pix_data import PixelDataMemory

            if pix_obj is None:
                pix_obj = PixelDataMemory()
            pix_input = {
                "img_range": np.zeros((2, 4), dtype=np.float32),
                "pix_block": pix_obj,
            }
            pix_pos, pos = self.sqw_serializer_.calculate_positions(
                base_pix_form, pix_input, self.dnd_eof_pos_
            )
            self.img_db_range_pos_ = int(pix_pos["img_range_pos_"])
            self.pix_pos_ = int(pix_pos["pix_block_pos_"]) + 8
            self.npixels_ = int(pix_obj.num_pixels)
            self.pix_range_pos_ = int(pos)
            self.eof_pix_pos_ = int(pos) + 8 * 4
            pr = getattr(pix_obj, "pix_range", PixelDataBase.EMPTY_RANGE[:, :4])
            self.pix_range_ = np.asarray(pr, dtype=np.float64)[:, :4].copy()
        else:
            pix_pos, pos, _, _ = self.sqw_serializer_.calculate_positions(
                base_pix_form, self.file_id_, self.dnd_eof_pos_, return_error=True
            )
            self.img_db_range_pos_ = int(pix_pos["img_range_pos_"])
            self.pix_pos_ = int(pix_pos["pix_block_pos_"]) + 8
            self.npixels_ = (int(pos) - int(self.pix_pos_)) // self.FILE_PIX_SIZE
            self.pix_range_pos_ = int(pos)
            self.eof_pix_pos_ = int(pos) + 8 * 4
            if self.file_id_ > 0:
                do_fseek(self.file_id_, int(self.pix_range_pos_), "bof")
                self.check_error_report_fail_("Unable to move to the start of the pixel range record")
                pr_arr, _ = do_fread(self.file_id_, (2, 4), "float32")
                self.check_error_report_fail_("Unable to read pixel range")
                self.pix_range_ = pr_arr.astype(np.float64)
        self.data_type_ = "a"

    def init_v3_specific(self) -> None:
        super().init_v3_specific()
        if self.file_id_ > 0 and self.pix_range_pos_ > 0:
            do_fseek(self.file_id_, int(self.pix_range_pos_), "bof")
            self.check_error_report_fail_("Unable to move to the start of the pixel range record")
            pr_arr, _ = do_fread(self.file_id_, (2, 4), "float32")
            self.check_error_report_fail_("Unable to read pixel range")
            self.pix_range_ = pr_arr.astype(np.float64)

    def put_pix(self, *args: Any) -> "faccess_sqw_v3_3":
        super().put_pix(*args)
        non_str = [a for a in args if not isinstance(a, str)]
        subobj, _ = self.extract_correct_subobj("pix", *non_str)
        pr = getattr(subobj, "pix_range", PixelDataBase.EMPTY_RANGE[:, :4])
        self.pix_range_ = np.asarray(pr, dtype=np.float64)[:, :4].copy()
        do_fseek(self.file_id_, int(self.pix_range_pos_), "bof")
        self.check_error_report_fail_("Unable to move to the start of the pixel range record")
        do_fwrite(self.file_id_, np.asarray(self.pix_range_, dtype=np.float32), "float32")
        self.check_error_report_fail_("Error writing pixels range")
        self.eof_pix_pos_ = do_ftell(self.file_id_)
        return self

