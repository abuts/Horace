"""Accessor for Horace SQW binary format version 4.0.

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v4/faccess_sqw_v4.m
- horace_core/sqw/file_io/@faccess_sqw_v4/*.m
- horace_core/sqw/file_io/@faccess_sqw_v4/private/*.m
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes.detectors import IX_detector_array
from pyhorace.data.instr_classes.instruments import IX_inst, IX_null_inst
from pyhorace.data.instr_classes.samples import IX_null_samp, IX_samp
from pyhorace.data.pixels.pix_data import PixelDataFileBacked, PixelDataMemory
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.io.binfile_v4_common import binfile_v4_common
from pyhorace.io.data_block import data_block
from pyhorace.io.dnd_data_block import dnd_data_block
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.io.pix_data_block import pix_data_block
from pyhorace.io.sqw_binfile_common import _validate_pixel_ranges, sqw_binfile_common
from pyhorace.io.sqw_file_interface import sqw_file_interface
from pyhorace.io.sqw_fopen import do_ferror, do_fread, do_fseek, do_ftell, do_fwrite


class faccess_sqw_v4(binfile_v4_common, sqw_file_interface):
    """Class to access Horace binary SQW files in format v4.0.

    Uses `blockAllocationTable` (BAT) with 9 data blocks:
    1. `bl_main_header` (`data_block('', 'main_header')`)
    2. `bl_detpar` (`data_block('', 'detpar')`)
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

    file_version_: float = 4.0
    num_pix_fields: int = 9

    def __init__(self, *args: Any) -> None:
        self.version_ = self.file_version_
        self.num_contrib_files_: Union[int, str] = "undefined"
        self.eof_pos_cache_: Any = []
        super().__init__(*args)

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

    def get_pix_position(self) -> Union[int, str]:
        if not self._bat.initialized:
            return "undefined"
        bl = self._bat.get_data_block("bl_pix_data_wrap")
        if bl.npixels == "undefined":
            return "undefined"
        return int(bl.pix_position)

    def get_npixels(self) -> Union[int, str]:
        if not self._bat.initialized:
            return "undefined"
        bl = self._bat.get_data_block("bl_pix_data_wrap")
        if bl.npixels == "undefined":
            return "undefined"
        return int(bl.npixels)

    def get_creation_date(self) -> str:
        if not self.creation_date_defined:
            try:
                mh = self.get_main_header()
                if mh.creation_date_defined:
                    self._creation_date = mh.creation_date
            except Exception:
                pass
        return super().get_creation_date()

    @property
    def num_contrib_files(self) -> Union[int, str]:
        if isinstance(self.num_contrib_files_, int):
            return self.num_contrib_files_
        if self._bat.initialized:
            try:
                bl = self._bat.get_data_block("bl_experiment_info_expdata")
                if bl.size > 0 and self.file_id_ > 0:
                    mh = self.get_main_header()
                    self.num_contrib_files_ = int(mh.nfiles)
                    return self.num_contrib_files_
            except Exception:
                pass
        return self.num_contrib_files_

    @num_contrib_files.setter
    def num_contrib_files(self, val: Union[int, str, None]) -> None:
        sqw_file_interface.num_contrib_files.fset(self, val)  # type: ignore[attr-defined]

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
    # Initialization
    # ------------------------------------------------------------------

    def init_from_sqw_obj(self, *args: Any) -> "faccess_sqw_v4":
        super().init_from_sqw_obj(*args)
        if self.sqw_holder_ is not None:
            mh, _ = self.extract_correct_subobj("main_header")
            if mh is not None and hasattr(mh, "nfiles"):
                self.num_contrib_files_ = int(mh.nfiles)
            elif hasattr(self.sqw_holder_, "exper_info") and self.sqw_holder_.exper_info is not None:
                self.num_contrib_files_ = max(int(self.sqw_holder_.exper_info.n_runs), 1)
        return self

    def init_from_sqw_file(self, *args: Any) -> "faccess_sqw_v4":
        super().init_from_sqw_file(*args)
        bl_pix = self._bat.get_data_block("bl_pix_data_wrap")
        if bl_pix.npixels == "undefined" and bl_pix.position > 0 and self.file_id_ > 0:
            bl_pix = bl_pix.get_sqw_block(self.file_id_)
            self._bat = self._bat.set_data_block(bl_pix)
        try:
            mh = self.get_main_header("-verbatim")
            self.num_contrib_files_ = int(mh.nfiles)
            if mh.creation_date_defined:
                self._creation_date = mh.creation_date
        except Exception:
            pass
        return self

    def copy_contents(
        self, other_obj: Any, keep_internals: bool = False
    ) -> Tuple["faccess_sqw_v4", List[str]]:
        _, missing = super().copy_contents(other_obj, keep_internals)
        if hasattr(other_obj, "num_contrib_files"):
            self.num_contrib_files_ = other_obj.num_contrib_files
        return self, missing

    # ------------------------------------------------------------------
    # Reading Methods
    # ------------------------------------------------------------------

    def get_main_header(self, *args: Any) -> main_header_cl:
        keep_original = False
        for arg in args:
            if isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if s in ("keep_original", "verbatim", "hverbatim", "hisverbatim") or s.startswith("keep") or s.startswith("verb"):
                    keep_original = True
                elif s in ("head", "header", "his", "history", "legacy", "sqw_struc", "nopix", "all", "file_backed"):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"get_main_header: Unrecognised option {arg!r}",
                    )
        self.check_obj_initiated_properly()
        bl = self._bat.get_data_block("bl_main_header")
        _, main_header = bl.get_sqw_block(self.file_id_)
        if isinstance(main_header, dict):
            main_header = main_header_cl.from_struct(main_header) if "serial_name" in main_header else main_header_cl(main_header)
        if not keep_original:
            main_header.filename = self.filename
            main_header.filepath = self.filepath
        if self.num_contrib_files_ == "undefined":
            self.num_contrib_files_ = int(main_header.nfiles)
        return main_header

    def get_detpar(self, *args: Any) -> Any:
        self.check_obj_initiated_properly()
        bl = self._bat.get_data_block("bl_detpar")
        if bl.size == 0:
            return None
        _, det = bl.get_sqw_block(self.file_id_)
        return det

    def _get_instr_or_sample_v4(self, block_name: str, base_cls: str, *args: Any) -> Any:
        self.check_obj_initiated_properly()
        get_all = False
        rem_args = []
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al in ("-all", "-verbatim", "-keep_original", "-head", "-his", "-nopix", "-legacy", "-sqw_struc", "-file_backed"):
                    if al == "-all":
                        get_all = True
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"Unrecognised option {a!r}",
                    )
            else:
                rem_args.append(a)

        idx = 1
        if not get_all and len(rem_args) > 0:
            if isinstance(rem_args[0], (int, np.integer)) and not isinstance(rem_args[0], bool):
                idx = int(rem_args[0])
            else:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Run number must be positive integer, got {rem_args[0]!r}",
                )

        bl = self._bat.get_data_block(block_name)
        _, cont = bl.get_sqw_block(self.file_id_)
        if not isinstance(cont, unique_objects_container):
            res = unique_objects_container(baseclass=base_cls)
            if isinstance(cont, (list, tuple)):
                for x in cont:
                    res = res.add(x)
            elif cont is not None:
                res = res.add(cont)
            cont = res

        if get_all:
            return cont
        if cont.n_runs == 0:
            return IX_null_inst() if base_cls == "IX_inst" else IX_null_samp()
        if idx < 1 or idx > cont.n_runs:
            if cont.n_unique == 1:
                return cont[0]
            raise HoraceError(
                "HORACE:faccess_sqw_v4:invalid_argument",
                f"Requested item {idx} when only {cont.n_runs} items are present in file",
            )
        return cont[idx - 1]

    def get_instrument(self, *args: Any) -> Any:
        return self._get_instr_or_sample_v4("bl_experiment_info_instruments", "IX_inst", *args)

    def get_sample(self, *args: Any) -> Any:
        return self._get_instr_or_sample_v4("bl_experiment_info_samples", "IX_samp", *args)

    def get_exp_info(self, *args: Any) -> Experiment:
        self.check_obj_initiated_properly()
        get_all = False
        no_is = False
        inst_number: Optional[int] = None
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-all":
                    get_all = True
                elif al == "-no_is":
                    no_is = True
                elif al in ("-verbatim", "-keep_original", "-head", "-his", "-nopix", "-legacy", "-sqw_struc", "-file_backed"):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"get_exp_info: Unrecognised option {a!r}",
                    )
            elif isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                inst_number = int(a)
            else:
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"get_exp_info: Invalid argument {a!r}",
                )

        bl_exp = self._bat.get_data_block("bl_experiment_info_expdata")
        _, expdata = bl_exp.get_sqw_block(self.file_id_)
        if isinstance(expdata, list):
            expdata_list = expdata
        elif isinstance(expdata, np.ndarray):
            expdata_list = list(expdata.ravel())
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
            inst = unique_objects_container(baseclass="IX_inst")
            samp = unique_objects_container(baseclass="IX_samp")
            for _ in range(len(expdata_sel)):
                inst = inst.add(IX_null_inst())
                samp = samp.add(IX_null_samp())
        else:
            if inst_number is not None and not get_all:
                inst_one = self.get_instrument(inst_number)
                samp_one = self.get_sample(inst_number)
                inst = unique_objects_container(baseclass="IX_inst").add(inst_one)
                samp = unique_objects_container(baseclass="IX_samp").add(samp_one)
            else:
                inst = self.get_instrument("-all")
                samp = self.get_sample("-all")

        det = self.get_detpar()
        if det is not None:
            det_cont = sqw_binfile_common.convert_old_det_forms(det, len(expdata_sel))
        else:
            det_cont = unique_objects_container(baseclass="IX_detector_array")

        exp = Experiment()
        exp.do_check_combo_arg = False
        exp.expdata = expdata_sel
        exp.instruments = inst
        exp.samples = samp
        if det_cont.n_runs == len(expdata_sel):
            exp.detector_arrays = det_cont
        exp.do_check_combo_arg = True
        exp.check_combo_arg()
        return exp

    def get_pix_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        self.check_obj_initiated_properly()
        bl = self._bat.get_data_block("bl_pix_metadata")
        _, meta = bl.get_sqw_block(self.file_id_)
        if not isinstance(meta, pix_metadata):
            meta = pix_metadata.from_struct(meta) if isinstance(meta, dict) and "serial_name" in meta else pix_metadata(meta)
        meta.full_filename = self.full_filename
        if nargout is not None and nargout > 1:
            return meta, self
        return meta

    def get_pix_range(self) -> np.ndarray:
        meta = self.get_pix_metadata()
        return meta.pix_range

    def get_data_range(self) -> np.ndarray:
        meta = self.get_pix_metadata()
        return meta.data_range

    def get_raw_pix(
        self, pix_lo: Optional[int] = None, pix_hi: Optional[int] = None
    ) -> np.ndarray:
        self.check_obj_initiated_properly()
        npix_tot = int(self.npixels) if isinstance(self.npixels, int) else 0
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
        raw, count = do_fread(self.file_id_, (9, n_read), "*float32")
        self.check_error_report_fail_("Error reading pixel block")
        if count < 9 * n_read:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:io_error",
                f"Truncated read in get_raw_pix: expected {9 * n_read}, got {count}",
            )
        return raw.astype(np.float64)

    def get_pix(self, *args: Any) -> PixelDataBase:
        file_backed = False
        rem_args = []
        for a in args:
            if isinstance(a, str) and a.lower() in ("-file_backed", "file_backed"):
                file_backed = True
            elif isinstance(a, str):
                pass
            else:
                rem_args.append(a)

        if len(rem_args) == 0 and (file_backed or PixelDataBase.do_filebacked(self.npixels if isinstance(self.npixels, int) else 0)):
            return PixelDataBase.create(self, file_backed=True)

        raw = self.get_raw_pix(*rem_args)
        pix = PixelDataMemory(raw)
        if len(rem_args) == 0:
            try:
                meta = self.get_pix_metadata()
                pix.metadata = meta
            except Exception:
                pass
        pix.full_filename = self.full_filename
        return pix

    def get_pix_at_indices(self, indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
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
    ) -> np.ndarray:
        starts = np.asarray(pix_starts, dtype=np.int64).ravel()
        sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
        if not skip_validation:
            _validate_pixel_ranges(starts, sizes)
        npix_tot = int(self.npixels) if isinstance(self.npixels, int) else 0
        if starts.size == 0:
            return np.zeros((9, 0), dtype=np.float32 if keep_precision else np.float64)
        ends = starts + sizes - 1
        if np.any(starts < 1) or np.any(ends > npix_tot):
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
        from pyhorace.sqw.sqw import SQW

        head_only = False
        nopix = False
        legacy = False
        sqw_struc = False
        file_backed = False
        mh_args = []

        for arg in args:
            if isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if s in ("head", "header", "his", "history"):
                    head_only = True
                elif s in ("keep_original", "verbatim", "hverbatim", "hisverbatim") or s.startswith("keep") or s.startswith("verb"):
                    mh_args.append("-verbatim")
                    if s in ("hisverbatim", "hverbatim"):
                        head_only = True
                elif s == "nopix":
                    nopix = True
                elif s == "legacy":
                    legacy = True
                elif s == "sqw_struc":
                    sqw_struc = True
                elif s in ("file_backed", "filebacked"):
                    file_backed = True
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v4:invalid_argument",
                        f"get_sqw: Unrecognised option {arg!r}",
                    )

        self.check_obj_initiated_properly()
        main_header = self.get_main_header(*mh_args)
        exp_info = self.get_exp_info("-all")
        detpar = self.get_detpar()
        if detpar is not None and exp_info.detector_arrays.n_runs == 0:
            exp_info.detector_arrays = sqw_binfile_common.convert_old_det_forms(detpar, main_header.nfiles)
        detpar_struct = (
            detpar.get_detpar_representation()
            if isinstance(detpar, IX_detector_array)
            else (
                detpar[0].get_detpar_representation()
                if isinstance(detpar, unique_objects_container) and detpar.n_runs > 0
                else detpar
            )
        )

        if head_only:
            data = self.get_dnd_metadata(*mh_args)
            if legacy or (nargout is not None and nargout > 1):
                outs = (main_header, exp_info, detpar_struct, data)
                return outs[:nargout] if nargout is not None else outs
            return {
                "main_header": main_header,
                "experiment_info": exp_info,
                "detpar": detpar_struct,
                "data": data,
            }

        data = self.get_data(*mh_args)
        if nopix:
            pix: PixelDataBase = PixelDataMemory()
        else:
            pix_args = ("-file_backed",) if file_backed else ()
            pix = self.get_pix(*pix_args)

        if sqw_struc:
            return {
                "main_header": main_header,
                "experiment_info": exp_info,
                "detpar": detpar_struct,
                "data": data,
                "pix": pix,
            }

        sqw_obj = SQW()
        sqw_obj._main_header = main_header
        sqw_obj._exper_info = exp_info
        sqw_obj._detpar = detpar_struct if isinstance(detpar_struct, dict) else {}
        sqw_obj._data = data
        sqw_obj._pix = pix

        if legacy or (nargout is not None and nargout > 1):
            outs = (sqw_obj, main_header, exp_info, detpar_struct, data, pix)
            return outs[:nargout] if nargout is not None else outs
        return sqw_obj

    def head(self, *args: Any) -> Dict[str, Any]:
        full_data = False
        for a in args:
            if isinstance(a, str) and a.lower() == "-full":
                full_data = True
            elif isinstance(a, str):
                raise HoraceError(
                    "HORACE:faccess_sqw_v4:invalid_argument",
                    f"Unrecognised option {a!r} in head",
                )
        hd = super().head(*args)
        return self.shuffle_fields_form_sqw_head(hd, full_data)

    # ------------------------------------------------------------------
    # Writing / Updating Methods
    # ------------------------------------------------------------------

    def put_main_header(self, *args: Any) -> "faccess_sqw_v4":
        update = False
        keep_original = False
        obj_args = []
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-update":
                    update = True
                elif al in ("-keep_original", "-verbatim"):
                    keep_original = True
            else:
                obj_args.append(a)
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        if inobj is not None and not keep_original:
            mh = getattr(inobj, "main_header", inobj)
            if isinstance(mh, main_header_cl):
                mh.filename = self.filename
                mh.filepath = self.filepath
                if not mh.creation_date_defined:
                    mh.creation_date = self.creation_date
        self._put_block_by_name("bl_main_header", inobj, update=update)
        return self

    def put_det_info(self, *args: Any) -> "faccess_sqw_v4":
        update = any(isinstance(a, str) and a.lower() == "-update" for a in args)
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        self._put_block_by_name("bl_detpar", inobj, update=update)
        return self

    def put_headers(self, *args: Any) -> "faccess_sqw_v4":
        update = any(isinstance(a, str) and a.lower() == "-update" for a in args)
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        self._put_block_by_name("bl_experiment_info_expdata", inobj, update=update)
        return self

    def put_instruments(self, *args: Any) -> "faccess_sqw_v4":
        update = any(isinstance(a, str) and a.lower() == "-update" for a in args)
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        self._put_block_by_name("bl_experiment_info_instruments", inobj, update=update)
        return self

    def put_samples(self, *args: Any) -> "faccess_sqw_v4":
        update = any(isinstance(a, str) and a.lower() == "-update" for a in args)
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        self._put_block_by_name("bl_experiment_info_samples", inobj, update=update)
        return self

    def put_pix_metadata(self, *args: Any) -> "faccess_sqw_v4":
        update = any(isinstance(a, str) and a.lower() == "-update" for a in args)
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        self._put_block_by_name("bl_pix_metadata", inobj, update=update)
        return self

    def put_pix(self, *args: Any) -> "faccess_sqw_v4":
        self.check_write_ReadOnly()
        obj_args = [a for a in args if not isinstance(a, str)]
        inobj = obj_args[0] if len(obj_args) > 0 else self.sqw_holder_
        pix_obj, _ = self.extract_correct_subobj("pix", inobj)
        bl_pix = self._bat.get_data_block("bl_pix_data_wrap")
        if not self._bat.initialized:
            self._bat = self._bat.init_obj_info(inobj, self.Clear_BAT_Options)
            bl_pix = self._bat.get_data_block("bl_pix_data_wrap")
        if bl_pix.position == 0:
            self._bat, _ = self._bat.find_block_place(bl_pix)
            bl_pix = self._bat.get_data_block("bl_pix_data_wrap")
        bl_pix.put_sqw_block(self.file_id_, pix_obj)
        self._bat = self._bat.set_data_block(bl_pix)
        self._bat.put_bat(self.file_id_)
        return self

    def put_raw_pix(
        self, pix_data: np.ndarray, pix_idx: int = 1, *args: Any
    ) -> "faccess_sqw_v4":
        self.check_write_ReadOnly()
        arr = np.asarray(pix_data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] != 9 and arr.shape[1] == 9:
            arr = arr.T
        pos = int(self.pix_position) + (int(pix_idx) - 1) * 9 * 4
        do_fseek(self.file_id_, pos, "bof")
        do_fwrite(self.file_id_, arr.ravel(order="F"), "float32")
        self.check_error_report_fail_("Error writing raw pixel data")
        return self

    def put_num_pixels(self, num_pixels: int) -> "faccess_sqw_v4":
        self.check_write_ReadOnly()
        bl_pix = self._bat.get_data_block("bl_pix_data_wrap")
        bl_pix.npixels = int(num_pixels)
        self._bat = self._bat.set_data_block(bl_pix)
        do_fseek(self.file_id_, int(bl_pix.position), "bof")
        do_fwrite(self.file_id_, np.array([int(num_pixels)], dtype="<u8"), "uint64")
        self.check_error_report_fail_("Error writing number of pixels")
        self._bat.put_bat(self.file_id_)
        return self

    def put_sqw(self, *args: Any) -> "faccess_sqw_v4":
        self.check_write_ReadOnly()
        keep_original = False
        obj_args = []
        for a in args:
            if isinstance(a, str):
                al = a.lower().lstrip("-")
                if al in ("keep_original", "verbatim") or al.startswith("keep") or al.startswith("verb"):
                    keep_original = True
            else:
                obj_args.append(a)

        if len(obj_args) > 0 and self._is_sqw_or_dnd_obj(obj_args[0]):
            self.init_from_sqw_obj(obj_args[0])

        inobj = self.sqw_holder_
        if inobj is None:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "Cannot save SQW object: no SQW object is attached to the file accessor.",
            )

        mh = getattr(inobj, "main_header", None)
        if isinstance(mh, main_header_cl):
            if not keep_original:
                mh.filename = self.filename
                mh.filepath = self.filepath
            if not mh.creation_date_defined:
                mh.creation_date = self.creation_date

        self._bat = self._bat.init_obj_info(inobj, self.Clear_BAT_Options)
        self.put_app_header()
        self._bat.put_bat(self.file_id_)
        for bl in self._bat.blocks_list:
            bl.put_sqw_block(self.file_id_, inobj)
        self.data_in_file_ = True
        return self

    def update_sqw_keep_pix(self) -> "faccess_sqw_v4":
        """Update all metadata blocks in place while preserving existing pixel data block."""
        self.check_write_ReadOnly()
        inobj = self.sqw_holder_
        if inobj is None:
            raise HoraceError(
                "HORACE:faccess_sqw_v4:runtime_error",
                "update_sqw_keep_pix requires an attached sqw_holder object",
            )
        self.put_app_header()
        for bl in self._bat.blocks_list:
            if isinstance(bl, pix_data_block):
                continue
            self._bat, _ = self._bat.find_block_place(bl, inobj)
            bl_upd = self._bat.get_data_block(bl.block_name)
            bl_upd.put_sqw_block(self.file_id_, inobj)
        self._bat.put_bat(self.file_id_)
        self.data_in_file_ = True
        return self

