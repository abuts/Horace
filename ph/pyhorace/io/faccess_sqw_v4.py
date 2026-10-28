"""Accessor for Horace SQW binary format version 4.0.

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v4/faccess_sqw_v4.m
- horace_core/sqw/file_io/@faccess_sqw_v4/*.m
- horace_core/sqw/file_io/@faccess_sqw_v4/private/*.m
"""

from __future__ import annotations

import datetime
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes import (
    IX_detector_array,
    IX_inst,
    IX_null_inst,
    IX_null_sample,
    IX_samp,
)
from pyhorace.data.pixels.pix_data import pix_data
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.io.binfile_v4_common import binfile_v4_common
from pyhorace.io.data_block import data_block
from pyhorace.io.dnd_data_block import dnd_data_block
from pyhorace.io.horace_binfile_interface import horace_binfile_interface
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.io.pix_data_block import pix_data_block
from pyhorace.io.sqw_binfile_common import _validate_pixel_ranges, sqw_binfile_common
from pyhorace.io.sqw_file_interface import sqw_file_interface
from pyhorace.io.sqw_fopen import (
    do_fread,
    do_fseek,
    do_fwrite,
    is_holder,
    is_sqw_struct,
    pack_io_outputs,
)


class faccess_sqw_v4(binfile_v4_common, sqw_file_interface):
    """Class to access Horace binary SQW files in format v4.0.

    Uses `blockAllocationTable` (BAT) with 9 data blocks:
    1. `bl__main_header` (`data_block('', 'main_header')`)
    2. `bl__detpar` (`data_block('', 'detpar')`)
    3. `bl_data_metadata` (`data_block('data', 'metadata')`)
    4. `bl_data_nd_data` (`dnd_data_block()`)
    5. `bl_experiment_info_instruments` (`data_block('experiment_info', 'instruments')`)
    6. `bl_experiment_info_samples` (`data_block('experiment_info', 'samples')`)
    7. `bl_experiment_info_expdata` (`data_block('experiment_info', 'expdata')`)
    8. `bl_pix_metadata` (`data_block('pix', 'metadata')`)
    9. `bl_pix_data_wrap` (`pix_data_block()`)

    Replaces MATLAB's `file_closer_` handle-simulation with standard Python
    destructor (`__del__`), `delete()`, and `fclose()`.
    """

    file_version_: ClassVar[float] = 4.0
    num_pix_fields: ClassVar[int] = 9

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.version_ = self.file_version_
        self.num_contrib_files_: Union[int, str] = "undefined"
        self.eof_pos_cache_: Any = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Block Table Definition & Overloadable Getters
    # ------------------------------------------------------------------

    @staticmethod
    def get_data_blocks() -> List[data_block]:
        return [
            data_block("", "main_header"),
            data_block("", "detpar"),
            data_block("data", "metadata"),
            dnd_data_block(),
            data_block("experiment_info", "instruments"),
            data_block("experiment_info", "samples"),
            data_block("experiment_info", "expdata"),
            data_block("pix", "metadata"),
            pix_data_block(),
        ]

    def get_sqw_type(self) -> bool:
        return True

    def get_data_type(self) -> str:
        return "a"

    def get_format_for_object(self) -> str:
        return "sqw"

    def get_pix_position(self) -> Any:
        if self.bat_ is None or not self.bat_.initialized:
            return []
        bl = self.bat_.get_data_block("bl_pix_data_wrap", nargout=1)
        if bl.npixels == "undefined":
            if self.file_id_ > 0 and bl.position > 0:
                bl.read_n_pixels(self.file_id_)
            else:
                return []
        return int(bl.pix_position)

    def get_npixels(self) -> Any:
        if self.bat_ is None or not self.bat_.initialized:
            return []
        bl, bl_idx = self.bat_.get_data_block("bl_pix_data_wrap", nargout=2)
        if bl.npixels == "undefined":
            if self.file_id_ > 0 and bl.position > 0:
                bl.read_n_pixels(self.file_id_)
                self.bat_.set_changed_block(bl, bl_idx)
            else:
                return "undefined"
        return int(bl.npixels)

    @property
    def eof_pix_position(self) -> int:
        pos = self.pix_position
        npix = self.npixels
        if isinstance(pos, int) and isinstance(npix, int):
            return int(pos + npix * 4 * 9)
        return 0

    def get_creation_date(self) -> str:
        if self.bat_ is not None and self.bat_.initialized and self.file_id_ > 0:
            try:
                mh = self.get_main_header(nargout=1)
                if mh.creation_date_defined:
                    return str(mh.creation_date)
            except Exception:
                pass
        return super().get_creation_date()

    def delete(self) -> "faccess_sqw_v4":
        super().delete()
        self.num_contrib_files_ = "undefined"
        self.eof_pos_cache_ = []
        return self

    def saveable_fields(self) -> List[str]:
        flds = list(super().saveable_fields())
        if "num_contrib_files" not in flds:
            flds.append("num_contrib_files")
        return flds

    # ------------------------------------------------------------------
    # Initialization & Upgrade
    # ------------------------------------------------------------------

    def init_from_sqw_obj(self, *args: Any) -> "faccess_sqw_v4":
        super().init_from_sqw_obj(*args)
        if self.sqw_holder_ is not None:
            in_obj = args[0] if len(args) > 0 else self.sqw_holder_
            exp_info_bl = self.bat_.get_data_block("bl_experiment_info_expdata", nargout=1)
            sub = exp_info_bl.get_subobj(in_obj)
            if isinstance(sub, (list, tuple, np.ndarray)):
                self.num_contrib_files_ = int(len(sub))
            elif hasattr(in_obj, "exper_info") and in_obj.exper_info is not None:
                self.num_contrib_files_ = int(in_obj.exper_info.n_runs)
            elif hasattr(in_obj, "main_header") and in_obj.main_header is not None:
                self.num_contrib_files_ = int(in_obj.main_header.nfiles)
            else:
                self.num_contrib_files_ = 1
        return self

    def init_from_sqw_file(self, *args: Any) -> "faccess_sqw_v4":
        super().init_from_sqw_file(*args)
        try:
            exp_info_bl = self.bat_.get_data_block("bl_experiment_info_expdata", nargout=1)
            _, exp_info = exp_info_bl.get_sqw_block(self.file_id_)
            if isinstance(exp_info, (list, tuple, np.ndarray)):
                self.num_contrib_files_ = int(len(exp_info))
            else:
                self.num_contrib_files_ = 1
        except Exception:
            try:
                mh = self.get_main_header("-verbatim", nargout=1)
                self.num_contrib_files_ = int(mh.nfiles)
            except Exception:
                pass
        try:
            bl_pix, bl_idx = self.bat_.get_data_block("bl_pix_data_wrap", nargout=2)
            if bl_pix.npixels == "undefined" and bl_pix.position > 0 and self.file_id_ > 0:
                bl_pix.read_n_pixels(self.file_id_)
                self.bat_.set_changed_block(bl_pix, bl_idx)
        except Exception:
            pass
        return self

    def init_pix_file(
        self,
        filename: str,
        npixels: int = 0,
        pix_meta: Optional[pix_metadata] = None,
    ) -> "faccess_sqw_v4":
        """Initialize a standalone v4 pixel binary file on disk for file-backed pixels."""
        from pyhorace.io.sqw_fopen import sqw_fopen

        full_path = os.path.abspath(str(filename))
        parent = os.path.dirname(full_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self.filename_ = os.path.basename(full_path)
        self.filepath_ = (parent + os.sep) if parent else ""
        if self.file_id_ > 0:
            self.fclose()
        self.file_id_ = sqw_fopen(full_path, "wb+")
        self.read_only_ = False
        self.num_dim_ = 0
        self.num_contrib_files_ = 1
        self.data_in_file_ = True

        # Initialize BAT with pix_metadata and pix_data_block
        meta = pix_meta if isinstance(pix_meta, pix_metadata) else pix_metadata()
        meta.npix = int(npixels)
        bl_meta = data_block("pix", "metadata")
        bl_meta.calc_obj_size(meta, nocache=False)

        bl_pix = pix_data_block()
        bl_pix.npixels = int(npixels)
        bl_pix.n_rows = 9
        bl_pix.records_only = True

        self.bat_.init(int(self.max_header_size_), [bl_meta, bl_pix])
        _, pos_meta, _ = self.bat_.find_block_place(bl_meta, bl_meta.size)
        bl_meta.position = pos_meta

        _, pos_pix, _ = self.bat_.find_block_place(bl_pix, bl_pix.size)
        bl_pix.position = pos_pix
        self.bat_.initialized_ = True

        self.put_app_header()
        bl_meta.put_sqw_block(self.file_id_, meta)
        bl_pix.put_data_header(self.file_id_)
        self.bat_.put_bat(self.file_id_)
        return self

    def copy_contents(
        self, other_obj: "horace_binfile_interface", *args: Any, **kwargs: Any
    ) -> Tuple["faccess_sqw_v4", List[str]]:
        self, missing_fields = super().copy_contents(other_obj, *args, **kwargs)  # type: ignore[misc]
        upgrade_range = any(
            isinstance(a, str) and a.strip().lower().lstrip("-").startswith("upgrade")
            for a in args
        )
        npix_other = getattr(other_obj, "npixels", "undefined")
        is_fb = isinstance(npix_other, (int, np.integer)) and PixelDataBase.do_filebacked(int(npix_other))
        if not (upgrade_range or (is_fb and self.faccess_version != other_obj.faccess_version)):
            return self, missing_fields

        if isinstance(other_obj, sqw_binfile_common):
            bl_pix = self.bat_.get_data_block("bl_pix_data_wrap", nargout=1)
            bl_pix.pix_position = int(other_obj.pix_position)
            bl_pix.locked = True
            bl_pix.npixels = int(other_obj.npixels)
            self.bat_.set_data_block(bl_pix)

            sqw_obj = other_obj.get_sqw("-norange", nargout=1)
            if not sqw_obj.main_header.creation_date_defined:
                sqw_obj.main_header.creation_date = datetime.datetime.now().strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            self.bat_.place_undocked_blocks(sqw_obj, True)
            self.sqw_holder_ = sqw_obj
            return self, ["data_in_memory_write_result"]

        raise HoraceError(
            "HORACE:faccess_sqw_v4:not_implemented",
            f"Converting format {type(other_obj).__name__} to format faccess_sqw_v4 explicitly is not implemented",
        )

    def do_class_dependent_updates(
        self, new_ldr: "horace_binfile_interface", *args: Any
    ) -> "horace_binfile_interface":
        if self.faccess_version != new_ldr.faccess_version:
            sqw_obj = self.get_sqw(nargout=1)
            new_ldr.sqw_holder = sqw_obj
            if hasattr(new_ldr, "put_sqw"):
                new_ldr.put_sqw()
            else:
                new_ldr.put_dnd()
            self.delete()
        return new_ldr

    def _set_file_to_update_impl(
        self, filename: Optional[Union[str, os.PathLike]] = None
    ) -> Tuple["horace_binfile_interface", bool, Any]:
        obj, file_exist, old_ldr = super()._set_file_to_update_impl(filename)
        if file_exist and old_ldr is not None and isinstance(old_ldr, binfile_v4_common):
            if old_ldr.faccess_version == obj.faccess_version and old_ldr.sqw_type == obj.sqw_type:
                holder = obj.sqw_holder
                obj, _ = obj.copy_contents(old_ldr, "-write_mode")
                obj.sqw_holder_ = holder
                old_ldr.delete()
        return obj, file_exist, old_ldr

    # ------------------------------------------------------------------
    # Reading Methods
    # ------------------------------------------------------------------

    def get_main_header(self, *args: Any, nargout: Optional[int] = None) -> Any:
        keep_original = False
        rem_args: List[Any] = []
        for arg in args:
            if isinstance(arg, str) and arg.startswith("-"):
                s = arg.strip().lower().lstrip("-")
                if (
                    s in ("keep_original", "verbatim", "hverbatim", "hisverbatim")
                    or s.startswith("keep")
                    or s.startswith("verb")
                ):
                    keep_original = True
                elif s in ("head", "header", "his", "history", "legacy", "sqw_struc", "nopix", "all", "file_backed"):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"get_main_header: Unrecognised option {arg!r}",
                    )
            else:
                rem_args.append(arg)

        main_header = self.get_block_data("bl__main_header", *rem_args, nargout=1)
        if isinstance(main_header, dict):
            main_header = (
                main_header_cl.from_struct(main_header)
                if "serial_name" in main_header
                else main_header_cl(main_header)
            )
        if not keep_original:
            main_header.filename = self.filename
            main_header.filepath = self.filepath
        if self.num_contrib_files_ == "undefined":
            self.num_contrib_files_ = int(main_header.nfiles)
        return pack_io_outputs((main_header, self), nargout, default=1)

    def get_detpar(self, *args: Any, nargout: Optional[int] = None) -> Any:
        convert_to_struct = False
        rem_args: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                if a.lower() in ("-struct", "-legacy"):
                    convert_to_struct = True
            else:
                rem_args.append(a)
        det = self.get_block_data("bl__detpar", *rem_args, nargout=1)
        if convert_to_struct:
            if isinstance(det, IX_detector_array):
                det = det.get_detpar_representation()
            elif isinstance(det, (unique_objects_container, unique_references_container)) and det.n_runs > 0:
                det = det[0].get_detpar_representation()
        return pack_io_outputs((det, self), nargout, default=1)

    def _get_instr_or_sample_v4(
        self, block_name: str, base_cls: str, *args: Any, nargout: Optional[int] = None
    ) -> Any:
        get_all = False
        file_args: List[Any] = []
        idx_args: List[int] = []
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al in (
                    "-all",
                    "-verbatim",
                    "-keep_original",
                    "-head",
                    "-his",
                    "-nopix",
                    "-legacy",
                    "-sqw_struc",
                    "-file_backed",
                ):
                    if al == "-all":
                        get_all = True
                elif not a.startswith("-"):
                    file_args.append(a)
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"Unrecognised option {a!r}",
                    )
            elif isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                idx_args.append(int(a))
            else:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Run number must be positive integer, got {a!r}",
                )

        idx = idx_args[0] if idx_args else 1
        cont = self.get_block_data(block_name, *file_args, nargout=1)
        if not isinstance(cont, (unique_objects_container, unique_references_container)):
            res = unique_objects_container(baseclass=base_cls)
            if isinstance(cont, (list, tuple, np.ndarray)):
                for x in cont:
                    res.add(x)
            elif cont is not None:
                res.add(cont)
            cont = res

        if get_all:
            return pack_io_outputs((cont, self), nargout, default=1)
        if cont.n_runs == 0:
            empty_obj = IX_null_inst() if base_cls == "IX_inst" else IX_null_sample()
            return pack_io_outputs((empty_obj, self), nargout, default=1)
        if idx < 1 or idx > cont.n_runs:
            if cont.n_unique == 1:
                return pack_io_outputs((cont[0], self), nargout, default=1)
            raise HoraceError(
                "HORACE:faccess_sqw_v4:invalid_argument",
                f"Requested item {idx} when only {cont.n_runs} items are present in file",
            )
        return pack_io_outputs((cont[idx - 1], self), nargout, default=1)

    def get_instrument(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return self._get_instr_or_sample_v4(
            "bl_experiment_info_instruments", "IX_inst", *args, nargout=nargout
        )

    def get_sample(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return self._get_instr_or_sample_v4(
            "bl_experiment_info_samples", "IX_samp", *args, nargout=nargout
        )

    def get_exp_info(self, *args: Any, nargout: Optional[int] = None) -> Any:
        get_all = False
        no_is = False
        inst_number: Optional[int] = None
        file_args: List[Any] = []
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-all":
                    get_all = True
                elif al == "-no_is":
                    no_is = True
                elif al in (
                    "-verbatim",
                    "-keep_original",
                    "-head",
                    "-his",
                    "-nopix",
                    "-legacy",
                    "-sqw_struc",
                    "-file_backed",
                ):
                    pass
                elif not a.startswith("-"):
                    file_args.append(a)
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"get_exp_info: Unrecognised option {a!r}",
                    )
            elif isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                if a <= 0:
                    raise HoraceError(
                        "HORACE:sqw_file_interface:invalid_argument",
                        "Cannot request header number <=0",
                    )
                inst_number = int(a)
            else:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"get_exp_info: Invalid argument {a!r}",
                )

        expdata = self.get_block_data("bl_experiment_info_expdata", *file_args, nargout=1)
        if isinstance(expdata, list):
            expdata_list = expdata
        elif isinstance(expdata, np.ndarray):
            expdata_list = list(expdata.ravel())
        elif expdata is None:
            expdata_list = []
        else:
            expdata_list = [expdata]

        if inst_number is not None and not get_all:
            if inst_number < 1 or inst_number > len(expdata_list):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Header number {inst_number} requested, but only {len(expdata_list)} headers exist.",
                )
            expdata_sel = [expdata_list[inst_number - 1]]
        else:
            expdata_sel = expdata_list

        if no_is:
            inst = unique_references_container("IX_inst")
            samp = unique_references_container("IX_samp")
            for _ in range(len(expdata_sel)):
                inst.add(IX_null_inst())
                samp.add(IX_null_sample())
        else:
            if inst_number is not None and not get_all:
                inst_one = self.get_instrument(inst_number, nargout=1)
                samp_one = self.get_sample(inst_number, nargout=1)
                inst = unique_references_container("IX_inst")
                inst.add(inst_one)
                samp = unique_references_container("IX_samp")
                samp.add(samp_one)
            else:
                inst = self.get_instrument("-all", nargout=1)
                samp = self.get_sample("-all", nargout=1)

        det = self.get_detpar(nargout=1)
        if det is not None:
            det_cont = sqw_binfile_common.convert_old_det_forms(det, len(expdata_sel))
        else:
            det_cont = unique_references_container("IX_detector_array")

        exp = Experiment()
        exp.do_check_combo_arg = False
        exp.expdata = expdata_sel
        exp.instruments = inst
        exp.samples = samp
        if det_cont.n_runs == len(expdata_sel):
            exp.detector_arrays = det_cont
        exp.do_check_combo_arg = True
        exp.check_combo_arg()
        return pack_io_outputs((exp, self), nargout, default=1)

    def get_pix_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        meta = self.get_block_data("bl_pix_metadata", *args, nargout=1)
        if not isinstance(meta, pix_metadata):
            meta = (
                pix_metadata.from_struct(meta)
                if isinstance(meta, dict) and "serial_name" in meta
                else pix_metadata(meta)
            )
        meta.full_filename = self.full_filename
        return pack_io_outputs((meta, self), nargout, default=1)

    def get_pix_range(self, *args: Any) -> np.ndarray:
        meta = self.get_pix_metadata(*args, nargout=1)
        return np.asarray(meta.pix_range, dtype=np.float64)

    def get_data_range(self, *args: Any) -> np.ndarray:
        meta = self.get_pix_metadata(*args, nargout=1)
        return np.asarray(meta.data_range, dtype=np.float64)

    def get_raw_pix(
        self, pix_lo: Optional[int] = None, pix_hi: Optional[int] = None
    ) -> np.ndarray:
        if isinstance(self.npixels, str) or isinstance(self.npixels, list):
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "get_raw_pix called on un-initialized loader",
            )
        npix_tot = int(self.npixels)
        if npix_tot == 0:
            return np.zeros((9, 0), dtype=np.float64)
        lo = 1 if pix_lo is None else int(pix_lo)
        hi = npix_tot if pix_hi is None else int(pix_hi)
        if lo < 1 or hi > npix_tot or lo > hi:
            if lo > hi and lo == 1 and hi == 0:
                return np.zeros((9, 0), dtype=np.float64)
            raise HoraceError(
                "HORACE:faccess_sqw_v4:invalid_argument",
                f"Invalid pixel range [{lo}, {hi}] for file with {npix_tot} pixels",
            )
        n_read = hi - lo + 1
        offset = int(self.pix_position) + (lo - 1) * 9 * 4
        do_fseek(self.file_id_, offset, "bof")
        self.check_read_error(self.file_id_, "Move to start of requested pixel range")
        raw, count = do_fread(self.file_id_, (9, n_read), "*float32")
        self.check_read_error(self.file_id_, "Error reading pixel block")
        if count < 9 * n_read:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:io_error",
                f"Truncated read in get_raw_pix: expected {9 * n_read}, got {count}",
            )
        return raw.astype(np.float64)

    def get_pix(self, *args: Any, nargout: Optional[int] = None) -> Any:
        raw_output = any(
            isinstance(a, str) and a.strip().lower().lstrip("-") in ("raw_output", "raw")
            for a in args
        )
        keep_precision = any(
            isinstance(a, str) and a.strip().lower().lstrip("-") == "keep_precision"
            for a in args
        )
        num_args = [
            int(a)
            for a in args
            if isinstance(a, (int, np.integer, float, np.floating)) and not isinstance(a, bool)
        ]
        if len(num_args) > 0 or raw_output:
            raw = self.get_raw_pix(*num_args)
            if keep_precision:
                raw = raw.astype(np.float32, copy=False)
            return pack_io_outputs((raw, self), nargout, default=1)

        if isinstance(self.npixels, (str, list)):
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "Attempt to get pixels from non-initialized file-accessor",
            )

        pix_meta = self.get_pix_metadata(nargout=1)
        npix_tot = int(self.npixels)
        if PixelDataBase.do_filebacked(npix_tot):
            pix: PixelDataBase = PixelDataFileBacked(pix_meta)
            pix = pix.init(self)
        else:
            pix_block = self.bat_.get_data_block("bl_pix_data_wrap", nargout=1)
            _, pdata = pix_block.get_sqw_block(self.file_id_)
            pix = PixelDataMemory()
            pix.metadata = pix_meta
            pix.data_wrap = pdata
            pix.full_filename = self.full_filename
        return pack_io_outputs((pix, self), nargout, default=1)

    def get_pix_at_indices(
        self, indices: Union[Sequence[int], np.ndarray]
    ) -> np.ndarray:
        idx_arr = np.asarray(indices, dtype=np.int64).ravel()
        if idx_arr.size == 0:
            return np.zeros((9, 0), dtype=np.float64)
        npix_tot = int(self.npixels) if isinstance(self.npixels, int) else 0
        if np.any(idx_arr < 1) or np.any(idx_arr > npix_tot):
            raise HoraceError(
                "SQW_BINFILE_COMMON:get_pix_at_indices",
                f"Pixel indices out of bounds [1, {npix_tot}].",
            )
        pix_starts, block_sizes = self._indices_to_ranges(idx_arr)
        return self.get_pix_in_ranges(pix_starts, block_sizes, skip_validation=True)

    @staticmethod
    def _indices_to_ranges(idx_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if idx_arr.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        diffs = np.diff(idx_arr)
        breaks = np.where(diffs != 1)[0] + 1
        starts = np.concatenate(([idx_arr[0]], idx_arr[breaks]))
        ends = np.concatenate((idx_arr[breaks - 1], [idx_arr[-1]]))
        sizes = ends - starts + 1
        return starts, sizes

    def get_pix_in_ranges(
        self,
        pix_starts: Union[Sequence[int], np.ndarray],
        block_sizes: Union[Sequence[int], np.ndarray],
        skip_validation: bool = False,
        keep_precision: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        if "skip_arg_validation" in kwargs:
            skip_validation = bool(kwargs.pop("skip_arg_validation"))
        starts = np.asarray(pix_starts, dtype=np.int64).ravel()
        sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
        if not skip_validation:
            _validate_pixel_ranges(starts, sizes)
        npix_tot = int(self.npixels) if isinstance(self.npixels, int) else 0
        if starts.size == 0:
            return np.zeros((9, 0), dtype=np.float32 if keep_precision else np.float64)
        ends = starts + sizes - 1
        valid_mask = sizes > 0
        if np.any(starts[valid_mask] < 1) or np.any(ends[valid_mask] > npix_tot):
            raise HoraceError(
                "SQW_BINFILE_COMMON:get_pix_in_ranges",
                f"Pixel range exceeds total pixels ({npix_tot}).",
            )
        total_pix = int(np.sum(sizes))
        out = np.empty((9, total_pix), dtype=np.float32)
        col_offset = 0
        base_pos = int(self.pix_position)
        for st, sz in zip(starts, sizes):
            n_items = int(sz)
            if n_items == 0:
                continue
            byte_pos = base_pos + (int(st) - 1) * 9 * 4
            do_fseek(self.file_id_, byte_pos, "bof")
            block, cnt = do_fread(self.file_id_, (9, n_items), "*float32")
            if cnt < 9 * n_items:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:io_error",
                    "Failed to read pixel range from file",
                )
            out[:, col_offset : col_offset + n_items] = block
            col_offset += n_items
        return out if keep_precision else out.astype(np.float64)

    def get_sqw(self, *args: Any, nargout: Optional[int] = None) -> Any:
        from pyhorace. coord_transf.bmatrix import _infer_nargout
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import SQW

        n_out = (
            _infer_nargout(default=1, explicit_nargout=nargout, depth=1)
            if nargout is None
            else int(nargout)
        )

        opts = self._parse_get_sqw_args(*args)
        if opts["file"] is not None:
            self.init(opts["file"])

        sqw_skel = SQW.make_sqw(False)
        mh_arg = ("-verbatim",) if opts["verbatim"] else ()
        sqw_skel["main_header"] = self.get_main_header(*mh_arg, nargout=1)
        sqw_skel["experiment_info"] = self.get_exp_info("-all", nargout=1)
        detpar = self.get_detpar(nargout=1)
        sqw_skel["detpar"] = detpar
        if detpar is not None and sqw_skel["experiment_info"].detector_arrays.n_runs == 0:
            sqw_skel["experiment_info"].detector_arrays = sqw_binfile_common.convert_old_det_forms(
                detpar, sqw_skel["main_header"].nfiles
            )

        detpar_struct = (
            detpar.get_detpar_representation()
            if isinstance(detpar, IX_detector_array)
            else (
                detpar[0].get_detpar_representation()
                if isinstance(detpar, (unique_objects_container, unique_references_container))
                and detpar.n_runs > 0
                else detpar
            )
        )

        if opts["head"] or opts["his"]:
            data_opt = ("-verbatim", "-head") if opts["verbatim"] else ("-head",)
            sqw_skel["data"] = self.get_dnd(*data_opt, nargout=1)
            sqw_skel["npixels"] = self.npixels
            if opts["legacy"] or n_out > 1:
                outs = (
                    sqw_skel["main_header"],
                    sqw_skel["experiment_info"],
                    detpar_struct,
                    sqw_skel["data"],
                    sqw_skel["pix"],
                )
                return outs[:n_out]
            sqw_skel["detpar"] = detpar_struct
            return sqw_skel

        if opts["nopix"]:
            data_opt = ("-verbatim",) if opts["verbatim"] else ()
            sqw_skel["data"] = self.get_dnd(*data_opt, nargout=1)
        else:
            d_meta = self.get_dnd_metadata(nargout=1)
            if not opts["verbatim"]:
                d_meta.filename = self.filename
                d_meta.filepath = self.filepath
            d_data = self.get_dnd_data(nargout=1)
            sqw_skel["data"] = DnDBase.dnd(d_meta, d_data)

            npix_tot = int(self.npixels) if isinstance(self.npixels, int) else 0
            fb = (
                bool(opts["file_backed"])
                if opts["force_pix_location"]
                else PixelDataBase.do_filebacked(npix_tot)
            )
            if fb:
                pix_meta = self.get_pix_metadata(nargout=1)
                pix_obj: PixelDataBase = PixelDataFileBacked(pix_meta)
                pix_obj = pix_obj.init(self)
                sqw_skel["pix"] = pix_obj
            else:
                sqw_skel["pix"] = self.get_pix(nargout=1)

        if opts["legacy"] or n_out > 1:
            if opts["legacy"]:
                outs = (
                    sqw_skel["main_header"],
                    sqw_skel["experiment_info"],
                    detpar_struct,
                    sqw_skel["data"],
                    sqw_skel["pix"],
                )
                return outs[:n_out]
            sqw_obj = SQW(sqw_skel)
            outs_obj = (
                sqw_obj,
                sqw_skel["main_header"],
                sqw_skel["experiment_info"],
                detpar_struct,
                sqw_skel["data"],
                sqw_skel["pix"],
            )
            return outs_obj[:n_out]

        if opts["sqw_struc"]:
            sqw_skel["detpar"] = detpar_struct
            return sqw_skel

        return SQW(sqw_skel)

    def head(self, *args: Any) -> Dict[str, Any]:
        full = False
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("f"):
                full = True
            else:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Unrecognized option {a!r} in head",
                )
        hd = super().head(*args)
        hd["npixels"] = self.npixels
        hd["num_contrib_files"] = self.num_contrib_files
        if full:
            hd["main_header"] = self.get_main_header(nargout=1)
            hd["experiment_info"] = self.get_exp_info("-all", nargout=1)
            hd["detpar"] = self.get_detpar("-struct", nargout=1)
            hd["pix_range"] = self.get_pix_range()
            hd["data_range"] = self.get_data_range()
        return self.shuffle_fields_form_sqw_head(hd, full)

    @staticmethod
    def shuffle_fields_form_sqw_head(
        head_data: Dict[str, Any], full: bool = False
    ) -> Dict[str, Any]:
        from pyhorace.sqw.sqw import SQW

        if full and "main_header" in head_data:
            mh = head_data.pop("main_header")
            if isinstance(mh, main_header_cl):
                head_data["creation_date"] = mh.creation_date
                if not mh.creation_date_defined:
                    head_data["creation_date_defined"] = False
            elif isinstance(mh, dict):
                head_data["creation_date"] = mh.get("creation_date", "")

        form_fields = SQW.head_form(full)
        out: Dict[str, Any] = {}
        for fld in form_fields:
            if fld in head_data:
                out[fld] = head_data[fld]
        for k, v in head_data.items():
            if k not in out:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Writing / Updating Methods
    # ------------------------------------------------------------------

    def put_main_header(self, *args: Any) -> "faccess_sqw_v4":
        keep_original = False
        rem_args: List[Any] = []
        for a in args:
            if isinstance(a, str):
                s = a.strip().lower().lstrip("-")
                if s in ("keep_original", "verbatim") or s.startswith("keep") or s.startswith("verb"):
                    keep_original = True
                elif s == "update":
                    pass
                else:
                    rem_args.append(a)
            else:
                rem_args.append(a)

        inobj = rem_args[0] if len(rem_args) > 0 else self.sqw_holder_
        if inobj is not None and not keep_original:
            mh = getattr(inobj, "main_header", inobj)
            if isinstance(mh, main_header_cl):
                mh.filename = self.filename
                mh.filepath = self.filepath
                if not mh.creation_date_defined:
                    mh.creation_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        return self.put_block_data("bl__main_header", *rem_args)  # type: ignore[return-value]

    def put_det_info(self, *args: Any) -> "faccess_sqw_v4":
        rem_args = [a for a in args if not (isinstance(a, str) and a.lower() == "-update")]
        return self.put_block_data("bl__detpar", *rem_args)  # type: ignore[return-value]

    def put_headers(self, *args: Any) -> "faccess_sqw_v4":
        rem_args: List[Any] = []
        header_num: Optional[int] = None
        for a in args:
            if isinstance(a, str) and a.lower() == "-update":
                continue
            elif isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                header_num = int(a)
            else:
                rem_args.append(a)
        if header_num is not None and len(rem_args) > 0:
            in_obj = rem_args[0]
            exp_bl = self.bat_.get_data_block("bl_experiment_info_expdata", nargout=1)
            _, existing = exp_bl.get_sqw_block(self.file_id_)
            exp_list = list(existing) if isinstance(existing, (list, tuple, np.ndarray)) else [existing]
            new_sub = exp_bl.get_subobj(in_obj)
            new_list = list(new_sub) if isinstance(new_sub, (list, tuple, np.ndarray)) else [new_sub]
            exp_list[header_num - 1] = new_list[header_num - 1] if len(new_list) >= header_num else new_list[0]
            self.put_sqw_block("bl_experiment_info_expdata", exp_list, "-noinit")
            return self
        return self.put_block_data("bl_experiment_info_expdata", *rem_args)  # type: ignore[return-value]

    def put_instruments(self, *args: Any) -> "faccess_sqw_v4":
        rem_args = [a for a in args if not (isinstance(a, str) and a.lower() == "-update")]
        return self.put_block_data("bl_experiment_info_instruments", *rem_args)  # type: ignore[return-value]

    def put_samples(self, *args: Any) -> "faccess_sqw_v4":
        rem_args = [a for a in args if not (isinstance(a, str) and a.lower() == "-update")]
        return self.put_block_data("bl_experiment_info_samples", *rem_args)  # type: ignore[return-value]

    def put_pix_metadata(self, *args: Any) -> "faccess_sqw_v4":
        rem_args = [a for a in args if not (isinstance(a, str) and a.lower() == "-update")]
        if len(rem_args) > 0:
            pix_mt = rem_args[0]
            if isinstance(pix_mt, PixelDataBase):
                pix_mt = pix_mt.metadata
            if not isinstance(pix_mt, pix_metadata):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"This method accepts instance of PixelData or pix_metadata objects as input. In fact you provided object of class {type(pix_mt).__name__}",
                )
            self.put_sqw_block("bl_pix_metadata", pix_mt, "-noinit")
        else:
            self.put_sqw_block("bl_pix_metadata")
        return self

    def put_pix(self, *args: Any) -> "faccess_sqw_v4":
        _, nopix, reserve, low_limit, rem_args = self._parse_put_pix_args(*args)
        if nopix and not reserve:
            return self
        records_only = bool(nopix and reserve)

        in_obj: Any = None
        if len(rem_args) > 0:
            in_obj = rem_args[0]
            if not (isinstance(in_obj, (PixelDataBase, pix_data)) or is_holder(in_obj) or isinstance(in_obj, np.ndarray)):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"First argument of the function have to be clean or packaged PixelData object. In fact it is: {type(in_obj).__name__}",
                )
        else:
            in_obj = self.sqw_holder

        if isinstance(in_obj, np.ndarray):
            in_obj = PixelDataMemory(in_obj)

        pix_block = self.bat_.get_data_block("bl_pix_data_wrap", nargout=1)
        sub_obj = pix_block.get_subobj(in_obj)
        pix_meta = (
            in_obj.metadata
            if isinstance(in_obj, PixelDataBase)
            else (
                in_obj.pix.metadata
                if hasattr(in_obj, "pix") and isinstance(in_obj.pix, PixelDataBase)
                else None
            )
        )

        if low_limit is not None:
            pos = int(self.pix_position) + (int(low_limit) - 1) * 9 * 4
            pix_block.put_pix_chunk(self.file_id_, sub_obj, pos)
            if pix_meta is not None:
                self.put_pix_metadata(pix_meta)
            return self

        if not self.bat_.initialized:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "Can not write pixels block if BlockAllocationTable is not initialized",
            )
        pix_block.records_only = records_only
        self.put_sqw_block(pix_block, sub_obj, "-noinit")
        if not records_only and pix_meta is not None:
            self.put_pix_metadata(pix_meta)
        return self

    @staticmethod
    def _parse_put_pix_args(
        *args: Any,
    ) -> Tuple[bool, bool, bool, Optional[int], List[Any]]:
        update = False
        nopix = False
        reserve = False
        low_limit: Optional[int] = None
        rem: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                s = a.strip().lower().lstrip("-")
                if s == "update":
                    update = True
                elif s == "nopix":
                    nopix = True
                elif s == "reserve":
                    reserve = True
            elif isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                low_limit = int(a)
            else:
                rem.append(a)
        return update, nopix, reserve, low_limit, rem

    def put_raw_pix(
        self, pix_data_arr: np.ndarray, pix_idx: int = 1, *args: Any
    ) -> "faccess_sqw_v4":
        arr = np.asarray(pix_data_arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] != 9 and arr.shape[1] == 9:
            arr = arr.T
        pos = int(self.pix_position) + (int(pix_idx) - 1) * 9 * 4
        self.move_to_position(self.file_id_, pos)
        do_fwrite(self.file_id_, arr.ravel(order="F"), "float32")
        self.check_write_error(self.file_id_, "Error writing raw pixel data")
        return self

    def put_num_pixels(self, num_pixels: int) -> "faccess_sqw_v4":
        bl_pix, _ = self.bat_.get_data_block("bl_pix_data_wrap", nargout=2)
        bl_pix.npixels = int(num_pixels)
        self.bat_.set_data_block(bl_pix)
        bl_pix.put_data_header(self.file_id_)
        self.bat_.put_bat(self.file_id_)
        return self

    def put_sqw(self, *args: Any) -> "faccess_sqw_v4":
        from pyhorace.sqw.sqw import SQW

        verbatim = False
        nopix = False
        reserve = False
        in_obj: Any = None
        rem_args: List[Any] = []

        for a in args:
            if isinstance(a, str) and a.startswith("-"):
                s = a.strip().lower().lstrip("-")
                if s in ("verbatim", "keep_original") or s.startswith("verb") or s.startswith("keep"):
                    verbatim = True
                elif s == "nopix":
                    nopix = True
                elif s == "reserve":
                    reserve = True
                elif s == "update":
                    pass
                else:
                    rem_args.append(a)
            elif self._is_sqw_or_dnd_obj(a) or is_sqw_struct(a):
                in_obj = a
            else:
                rem_args.append(a)

        if in_obj is not None:
            if not isinstance(in_obj, SQW):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Input object to save in sqw v4 format must be sqw object, got {type(in_obj).__name__}",
                )
            self.init(in_obj, *rem_args)
            rem_args = []

        sqh = self.sqw_holder
        if sqh is None:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "Cannot save SQW object: no SQW object is attached to the file accessor.",
            )

        re_init_bat = False
        if isinstance(sqh, SQW):
            if not sqh.main_header.creation_date_defined:
                cr_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                sqh.main_header.creation_date = cr_date
                sqh.data.creation_date = cr_date
                re_init_bat = True
            if not verbatim:
                if sqh.main_header.filename != self.filename or sqh.main_header.filepath != self.filepath:
                    sqh.main_header.filename = self.filename
                    sqh.main_header.filepath = self.filepath
                    sqh.data.filename = self.filename
                    sqh.data.filepath = self.filepath
                    re_init_bat = True
            self.sqw_holder_ = sqh

        if re_init_bat or not self.bat_.initialized:
            self.bat_.init_obj_info(sqh, "-nocache")

        if nopix:
            bl_pix, bl_idx = self.bat_.get_data_block("bl_pix_data_wrap", nargout=2)
            bl_pix.records_only = bool(reserve)
            self.bat_.set_changed_block(bl_pix, bl_idx)
            if not reserve:
                rem_args = list(rem_args) + ["ignore_blocks", ["bl_pix_data_wrap"]]

        self.put_all_blocks(*rem_args)
        self.data_in_file_ = True
        return self

    def update_sqw_keep_pix(self, *args: Any) -> "faccess_sqw_v4":
        """Update all metadata blocks in place while preserving existing pixel data block."""
        inobj = args[0] if len(args) > 0 and self._is_sqw_or_dnd_obj(args[0]) else self.sqw_holder_
        if inobj is None:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "update_sqw_keep_pix requires an attached sqw_holder object",
            )
        self.put_new_blocks_values(
            inobj, "-nocache", "exclude", ["bl_pix_data_wrap"]
        )
        self.data_in_file_ = True
        return self
