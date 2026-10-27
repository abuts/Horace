"""Common base class for Horace v2 and v3 binary SQW file accessors.

Ported from:
- horace_core/sqw/file_io/@sqw_binfile_common/sqw_binfile_common.m
- horace_core/sqw/file_io/@sqw_binfile_common/*.m
- horace_core/sqw/file_io/@sqw_binfile_common/private/*.m
"""

from __future__ import annotations

import copy
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.instr_classes.ix_experiment import IX_experiment
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.io.binfile_v2_common import (
    _do_convert_to_double,
    _parse_get_data_args,
    binfile_v2_common,
)
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.io.sqw_fields_formatters import (
    field_cellarray_of_strings,
    field_const_array_dependent,
    field_not_in_structure,
    field_pix,
    field_var_array,
)
from pyhorace.io.sqw_file_interface import sqw_file_interface
from pyhorace.io.sqw_fopen import (
    do_ferror,
    do_fread,
    do_fseek,
    do_ftell,
    do_fwrite,
    pack_io_outputs,
)
from pyhorace.io.sqw_serializer import sqw_serializer


def _validate_pixel_ranges(starts: np.ndarray, sizes: np.ndarray) -> None:
    """Validate 1-based pixel starts and block sizes arrays."""
    if starts.size != sizes.size:
        raise HoraceError(
            "HORACE:validate_ranges:invalid_argument",
            "pix_starts and block_sizes must have equal length",
        )
    if np.any(starts < 1) or np.any(sizes <= 0):
        raise HoraceError(
            "HORACE:validate_ranges:invalid_argument",
            "pix_starts must be >= 1 and block_sizes must be > 0",
        )


def _parse_const_flag(*args: Any) -> bool:
    """Return True if `-const` flag is present in `args`."""
    const_only = False
    for arg in args:
        if isinstance(arg, str):
            s = arg.strip().lower().lstrip("-")
            if s in ("c", "co", "con", "cons", "const"):
                const_only = True
            elif s in ("h", "he", "hea", "head", "heade", "header"):
                pass
            else:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:invalid_argument",
                    f"Unrecognized format option: {arg!r}",
                )
    return const_only


def _process_main_header_fields(*args: Any) -> Dict[str, Any]:
    """Build v2/v3 SQW main header format dictionary."""
    const_only = _parse_const_flag(*args)
    var_fields: Dict[str, Any] = {
        "filename_with_cdate": "",
        "filepath": "",
        "title": "",
    }
    const_fields: Dict[str, Any] = {
        "nfiles": np.int32(1),
    }
    if const_only:
        return const_fields
    res = dict(var_fields)
    res.update(const_fields)
    return res


def _process_header_fields(*args: Any) -> Dict[str, Any]:
    """Build v2/v3 SQW contributing run header (`IX_experiment`) format dictionary."""
    const_only = _parse_const_flag(*args)
    var_fields: Dict[str, Any] = {
        "filename": "",
        "filepath": "",
    }
    const_fields: Dict[str, Any] = {
        "efix": np.float32(1),
        "emode": np.int32(1),
        "alatt": np.array([1, 3], dtype=np.float32),
        "angdeg": np.array([1, 3], dtype=np.float32),
        "cu": np.array([1, 3], dtype=np.float32),
        "cv": np.array([1, 3], dtype=np.float32),
        "psi": np.float32(1),
        "omega": np.float32(1),
        "dpsi": np.float32(1),
        "gl": np.float32(1),
        "gs": np.float32(1),
        "en": field_var_array(1, "single"),
        "uoffset": np.array([4, 1], dtype=np.float32),
        "u_to_rlu": np.array([4, 4], dtype=np.float32),
        "ulen": np.array([1, 4], dtype=np.float32),
        "ulabel": field_cellarray_of_strings(),
    }
    if const_only:
        return const_fields
    res = dict(var_fields)
    res.update(const_fields)
    return res


def _process_detpar_fields(*args: Any) -> Dict[str, Any]:
    """Build v2/v3 SQW detector parameter (`detpar`) format dictionary."""
    const_only = _parse_const_flag(*args)
    var_fields: Dict[str, Any] = {
        "filename": "",
        "filepath": "",
    }
    const_fields: Dict[str, Any] = {
        "ndet": field_not_in_structure("group"),
        "group": field_const_array_dependent("ndet", 1, "single"),
        "x2": field_const_array_dependent("ndet", 1, "single"),
        "phi": field_const_array_dependent("ndet", 1, "single"),
        "azim": field_const_array_dependent("ndet", 1, "single"),
        "width": field_const_array_dependent("ndet", 1, "single"),
        "height": field_const_array_dependent("ndet", 1, "single"),
    }
    if const_only:
        return const_fields
    res = dict(var_fields)
    res.update(const_fields)
    return res


def _update_pixels_run_id(sqw_struc: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure `experiment_info.runid_map` matches pixel `run_idx` values when creation date is undefined."""
    pix = sqw_struc.get("pix")
    exp_info = sqw_struc.get("experiment_info")
    if pix is None or exp_info is None or not isinstance(exp_info, Experiment):
        return sqw_struc
    if pix.num_pixels == 0:
        return sqw_struc

    unique_run_id = np.unique(pix.run_idx).astype(int)
    file_id = np.asarray(exp_info.runid_map.keys, dtype=int)
    in_Pix_id = np.isin(unique_run_id, file_id)
    if np.all(in_Pix_id):
        return sqw_struc

    n_runs = exp_info.n_runs
    if len(unique_run_id) > n_runs:
        raise HoraceError(
            "HORACE:sqw_binfile_common:runtime_error",
            f"Number of unique run indexes in pixels ({len(unique_run_id)}) exceeds number of runs ({n_runs})",
        )

    if len(unique_run_id) == n_runs:
        exp_info.runid_map = dict(zip(unique_run_id.tolist(), range(1, n_runs + 1)))
        exp_info.runid_recalculated = True
    else:
        for idx, r_id in enumerate(unique_run_id, start=1):
            exp_info.expdata[idx - 1].run_id = int(r_id)
        exp_info.runid_recalculated = True

    sqw_struc["experiment_info"] = exp_info
    return sqw_struc


class sqw_binfile_common(binfile_v2_common, sqw_file_interface):
    """Common implementation for Horace v2 and v3 binary SQW file accessors."""

    FILE_PIX_SIZE: ClassVar[int] = 4 * 9

    data_fields_to_save_: ClassVar[List[str]] = [
        "main_header_pos_",
        "main_head_pos_info_",
        "header_pos_",
        "header_pos_info_",
        "detpar_pos_",
        "detpar_pos_info_",
    ]
    pixel_fields_to_save_: ClassVar[List[str]] = [
        "num_contrib_files_",
        "npixels_",
        "img_db_range_pos_",
        "pix_pos_",
        "eof_pix_pos_",
    ]
    fields_to_save_: ClassVar[List[str]] = (
        binfile_v2_common.fields_to_save_ + data_fields_to_save_ + pixel_fields_to_save_
    )

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.main_header_pos_: int = 26
        self.main_head_pos_info_: Any = []
        self.header_pos_: Any = 0
        self.header_pos_info_: Any = []
        self.detpar_pos_: int = 0
        self.detpar_pos_info_: Any = []
        self.img_db_range_pos_: int = 0
        self.pix_pos_: Union[int, str] = "undefined"
        self.eof_pix_pos_: int = 0
        self.contains_runid_in_header_: Any = []
        self.npixels_: Union[int, str] = "undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def main_header_position(self) -> int:
        return int(self.main_header_pos_)

    @property
    def header_position(self) -> Any:
        return self.header_pos_

    @property
    def detpar_position(self) -> int:
        return int(self.detpar_pos_)

    @property
    def img_db_range_pos(self) -> int:
        return self.get_img_db_range_pos()

    @property
    def eof_pix_pos(self) -> int:
        return int(self.eof_pix_pos_)

    @property
    def contains_runid_in_header(self) -> bool:
        if isinstance(self.contains_runid_in_header_, (list, tuple, np.ndarray)) and len(
            self.contains_runid_in_header_
        ) == 0:
            if isinstance(self.header_pos_info_, list) and len(self.header_pos_info_) > 0:
                self.check_header_mangilig(self.header_pos_info_[0])
            else:
                return False
        return bool(np.all(self.contains_runid_in_header_))

    @contains_runid_in_header.setter
    def contains_runid_in_header(self, val: bool) -> None:
        self.contains_runid_in_header_ = bool(val)

    # ------------------------------------------------------------------
    # Overloadable Getters
    # ------------------------------------------------------------------

    def get_npixels(self) -> Union[int, str]:
        return self.npixels_

    def get_pix_position(self) -> Union[int, str]:
        return self.pix_pos_

    def get_img_db_range_pos(self) -> int:
        return int(self.img_db_range_pos_)

    def get_sqw_type(self) -> bool:
        return True

    def get_format_for_object(self) -> str:
        return "sqw"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def delete(self) -> "sqw_binfile_common":
        super().delete()
        self.main_header_pos_ = 26
        self.main_head_pos_info_ = []
        self.header_pos_ = 0
        self.header_pos_info_ = []
        self.detpar_pos_ = 0
        self.detpar_pos_info_ = []
        self.img_db_range_pos_ = 0
        self.pix_pos_ = "undefined"
        self.eof_pix_pos_ = 0
        self.npixels_ = "undefined"
        self.contains_runid_in_header_ = []
        return self

    def from_bare_struct(self, indata: Dict[str, Any]) -> "sqw_binfile_common":
        super().from_bare_struct(indata)
        if isinstance(self.data_type_, bytes):
            self.data_type_ = self.data_type_.decode("windows-1252", errors="replace")
        elif isinstance(self.data_type_, np.ndarray):
            if self.data_type_.dtype.kind in ("U", "S"):
                self.data_type_ = "".join(str(x) for x in self.data_type_.ravel())
            else:
                self.data_type_ = "".join(chr(int(x)) for x in self.data_type_.ravel())

        if isinstance(self.header_pos_info_, dict):
            self.header_pos_info_ = [self.header_pos_info_] if self.header_pos_info_ else []
        elif isinstance(self.header_pos_info_, np.ndarray):
            self.header_pos_info_ = list(self.header_pos_info_.ravel())

        if isinstance(self.header_pos_, np.ndarray):
            self.header_pos_ = [int(x) for x in self.header_pos_.ravel()]
        elif isinstance(self.header_pos_, (int, float, np.number)) and not isinstance(
            self.header_pos_, (bool, np.bool_)
        ):
            hp_val = int(self.header_pos_)
            if hp_val > 0 and isinstance(self.header_pos_info_, list) and len(self.header_pos_info_) == 1:
                self.header_pos_ = [hp_val]
            else:
                self.header_pos_ = hp_val

        int_fields = (
            "main_header_pos_",
            "detpar_pos_",
            "data_pos_",
            "s_pos_",
            "e_pos_",
            "npix_pos_",
            "dnd_eof_pos_",
            "img_db_range_pos_",
            "pix_pos_",
            "eof_pix_pos_",
            "num_contrib_files_",
            "npixels_",
            "num_dim_",
        )
        for fld in int_fields:
            val = getattr(self, fld, None)
            if isinstance(val, (float, np.floating, np.integer)) and not isinstance(val, (bool, np.bool_)):
                setattr(self, fld, int(val))
            elif isinstance(val, np.ndarray) and val.size == 1 and val.dtype.kind in ("i", "u", "f"):
                setattr(self, fld, int(val.flat[0]))
        return self

    # ------------------------------------------------------------------
    # Format Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_main_header_form(*args: Any) -> Dict[str, Any]:
        return _process_main_header_fields(*args)

    @staticmethod
    def get_header_form(*args: Any) -> Dict[str, Any]:
        return _process_header_fields(*args)

    @staticmethod
    def get_detpar_form(*args: Any) -> Dict[str, Any]:
        return _process_detpar_fields(*args)

    @staticmethod
    def get_pix_form() -> Dict[str, Any]:
        return {
            "img_range": np.array([2, 4], dtype=np.float32),
            "dummy": field_not_in_structure("img_range"),
            "pix_block": field_pix(),
        }

    @staticmethod
    def make_pseudo_sqw(nfiles: int) -> Any:
        from pyhorace.sqw.sqw import SQW

        sq = SQW()
        head = IX_experiment()
        heads = [copy.deepcopy(head) for _ in range(max(int(nfiles), 1))]
        exper = Experiment()
        exper.do_check_combo_arg = False
        exper.expdata = heads
        sq.exper_info = exper
        return sq

    def check_header_mangilig(self, head_pos: Any) -> None:
        """Check whether contributing header 1 contains `$id$` in its filename."""
        if isinstance(head_pos, (list, tuple, np.ndarray)):
            if len(head_pos) == 0:
                self.contains_runid_in_header_ = False
                return
            head_pos = head_pos[0]
        if not isinstance(head_pos, dict) or "filepath_pos_" not in head_pos or "filename_pos_" not in head_pos:
            self.contains_runid_in_header_ = False
            return
        fn_size = int(head_pos["filepath_pos_"]) - int(head_pos["filename_pos_"])
        if fn_size <= 0 or self.file_id_ <= 0:
            self.contains_runid_in_header_ = False
            return
        do_fseek(self.file_id_, int(head_pos["filename_pos_"]), "bof")
        raw, _ = do_fread(self.file_id_, fn_size, "*char")
        _, res = do_ferror(self.file_id_)
        if res != 0:
            raise HoraceError(
                "HORACE:sqw_binfile_common:io_error",
                "Cannot read filename to check header mangling",
            )
        fname = raw.tobytes().decode("windows-1252", errors="replace")
        self.contains_runid_in_header_ = "$id$" in fname

    @staticmethod
    def convert_old_det_forms(det: Any, nfiles: int = 1) -> unique_objects_container:
        """Convert old detpar struct(s) into a `unique_objects_container` of `IX_detector_array`."""
        det_container = unique_objects_container(baseclass="IX_detector_array")
        if isinstance(det, dict):
            det_arr = IX_detector_array(det)
            det_container.add(det_arr)
            if nfiles > 1:
                det_container.replicate_runs(nfiles)
        elif isinstance(det, (list, tuple)):
            for d in det:
                det_arr = d if isinstance(d, IX_detector_array) else IX_detector_array(d)
                det_container.add(det_arr)
        elif isinstance(det, IX_detector_array):
            det_container.add(det)
            if nfiles > 1:
                det_container.replicate_runs(nfiles)
        elif isinstance(det, unique_objects_container) or (
            hasattr(det, "n_runs") and hasattr(det, "unique_objects")
        ):
            det_container = det
        return det_container

    # ------------------------------------------------------------------
    # Initialization (from SQW object or open file)
    # ------------------------------------------------------------------

    def extract_correct_subobj(
        self, obj_name: str, *args: Any
    ) -> Tuple[Any, bool]:
        from pyhorace.sqw.sqw import SQW

        if len(args) == 0 or args[0] is None:
            inobj = self.sqw_holder_
            external = False
        else:
            inobj = args[0]
            external = True

        if isinstance(inobj, SQW) or (
            isinstance(inobj, dict) and "main_header" in inobj and "data" in inobj
        ) or (hasattr(inobj, "main_header") and hasattr(inobj, "data")):
            if obj_name in ("experiment_info", "header"):
                if isinstance(inobj, dict):
                    sub_obj = inobj.get("experiment_info", inobj.get("exper_info", inobj.get("header")))
                else:
                    sub_obj = getattr(inobj, "experiment_info", getattr(inobj, "exper_info", None))
            elif obj_name == "pix":
                sub_obj = inobj.get("pix") if isinstance(inobj, dict) else getattr(inobj, "pix", None)
            elif obj_name == "detpar":
                if isinstance(inobj, dict):
                    sub_obj = inobj.get("detpar")
                    exp = inobj.get("experiment_info", inobj.get("exper_info"))
                else:
                    sub_obj = getattr(inobj, "detpar", None)
                    exp = getattr(inobj, "experiment_info", getattr(inobj, "exper_info", None))
                if sub_obj is None or (isinstance(sub_obj, dict) and not sub_obj):
                    if isinstance(exp, Experiment) and exp.detector_arrays.n_runs > 0:
                        sub_obj = exp.detector_arrays
                    elif hasattr(inobj, "my_detpar"):
                        sub_obj = inobj.my_detpar()
            elif isinstance(inobj, dict):
                sub_obj = inobj.get(obj_name)
            else:
                sub_obj = getattr(inobj, obj_name, None)
        elif isinstance(inobj, dict) and obj_name in inobj:
            sub_obj = inobj[obj_name]
        else:
            sub_obj = inobj
        return sub_obj, external

    def init_from_sqw_obj(self, *args: Any) -> "sqw_binfile_common":
        if len(args) == 0:
            raise HoraceError(
                "HORACE:sqw_binfile_common:invalid_argument",
                "init_from_sqw_obj method has to be invoked with at least one argument",
            )
        inobj = args[0]
        if not self._is_sqw_or_dnd_obj(inobj):
            raise HoraceError(
                "HORACE:sqw_binfile_common:invalid_argument",
                "First argument of init_from_sqw_obj method has to be an sqw object or structure",
            )
        self._init_sqw_structure_field_by_field(*args)
        super().init_from_sqw_obj(*args)
        self.sqw_holder_ = args[0]
        self._init_pix_info(*args)
        return self

    def init_from_sqw_file(self, *args: Any) -> "sqw_binfile_common":
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "File is not open when initializing SQW structure",
            )
        self._init_sqw_structure_field_by_field(*args)
        super().init_from_sqw_file(*args)
        self._init_pix_info(*args)
        return self

    def _init_sqw_structure_field_by_field(self, *args: Any) -> None:
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()

        from_obj = len(args) > 0 and self._is_sqw_or_dnd_obj(args[0])
        inobj = args[0] if from_obj else None

        # 1. Main header
        mh_form = self.get_main_header_form()
        if from_obj:
            mh_obj, _ = self.extract_correct_subobj("main_header", inobj)
            if isinstance(mh_obj, main_header_cl):
                if not mh_obj.creation_date_defined and self.upgrade_headers_:
                    import datetime

                    mh_obj.creation_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                mh_input: Any = mh_obj
            else:
                mh_input = mh_obj
            mh_pos, pos = self.sqw_serializer_.calculate_positions(
                mh_form, mh_input, self.main_header_pos_
            )
            n_files = int(mh_obj.nfiles) if hasattr(mh_obj, "nfiles") else int(mh_obj.get("nfiles", 0))
            self.num_contrib_files_ = n_files
        else:
            mh_pos, pos, io_err, _ = self.sqw_serializer_.calculate_positions(
                mh_form, self.file_id_, self.main_header_pos_, return_error=True
            )
            if io_err:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:io_error",
                    "IO error while parsing main header",
                )
            do_fseek(self.file_id_, int(mh_pos["nfiles_pos_"]), "bof")
            nf_arr, cnt = do_fread(self.file_id_, 1, "int32")
            if cnt < 1:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:io_error",
                    "IO error reading nfiles from main header",
                )
            n_files = int(nf_arr.flat[0])
            self.num_contrib_files_ = n_files

        self.main_head_pos_info_ = mh_pos

        # 2. Contributing headers
        if n_files < 0:
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                f"Negative number of contributing files ({n_files}) encountered",
            )
        elif n_files == 0:
            self.header_pos_ = int(pos)
            self.header_pos_info_ = []
        else:
            hdr_form = self.get_header_form()
            if from_obj:
                headers, _ = self.extract_correct_subobj("experiment_info", inobj)
                self.contains_runid_in_header_ = True
                if isinstance(headers, Experiment):
                    headers_list = headers.convert_to_old_headers()
                    if isinstance(headers_list, dict):
                        headers_list = [headers_list]
                    else:
                        headers_list = list(headers_list)
                elif isinstance(headers, (list, tuple)):
                    headers_list = list(headers)
                else:
                    headers_list = [headers]

                if len(headers_list) != n_files:
                    raise HoraceError(
                        "HORACE:sqw_binfile_common:runtime_error",
                        f"Main header nfiles ({n_files}) does not match number of experiment headers ({len(headers_list)})",
                    )

                header_positions: List[int] = []
                header_infos: List[Dict[str, Any]] = []
                for i in range(n_files):
                    header_positions.append(int(pos))
                    hdr_i = headers_list[i]
                    if hasattr(hdr_i, "convert_to_binfile_header"):
                        hdr_i = hdr_i.convert_to_binfile_header(
                            "-alatt_angdeg", not self.contains_runid_in_header
                        )
                    h_info, pos = self.sqw_serializer_.calculate_positions(
                        hdr_form, hdr_i, pos
                    )
                    header_infos.append(h_info)
                self.header_pos_ = header_positions
                self.header_pos_info_ = header_infos
            else:
                header_positions = []
                header_infos = []
                for i in range(n_files):
                    header_positions.append(int(pos))
                    h_info, pos, io_err, _ = self.sqw_serializer_.calculate_positions(
                        hdr_form, self.file_id_, pos, return_error=True
                    )
                    if io_err:
                        raise HoraceError(
                            "HORACE:sqw_binfile_common:io_error",
                            "IO error while parsing contributing experiment header",
                        )
                    if i == 0:
                        self.check_header_mangilig(h_info)
                        do_fseek(self.file_id_, int(pos), "bof")
                    header_infos.append(h_info)
                self.header_pos_ = header_positions
                self.header_pos_info_ = header_infos

        # 3. Detector parameters
        self.detpar_pos_ = int(pos)
        det_form = self.get_detpar_form()
        if from_obj:
            detpar_obj, _ = self.extract_correct_subobj("detpar", inobj)
            if isinstance(detpar_obj, unique_objects_container) or (
                hasattr(detpar_obj, "n_runs") and hasattr(detpar_obj, "unique_objects")
            ):
                if detpar_obj.n_runs > 0:
                    detpar_struct = detpar_obj[0].get_detpar_representation()
                else:
                    detpar_struct = IX_detector_array().get_detpar_representation()
            elif isinstance(detpar_obj, IX_detector_array):
                detpar_struct = detpar_obj.get_detpar_representation()
            elif isinstance(detpar_obj, dict) and detpar_obj:
                detpar_struct = detpar_obj
            else:
                detpar_struct = IX_detector_array().get_detpar_representation()
            det_info, pos = self.sqw_serializer_.calculate_positions(
                det_form, detpar_struct, pos
            )
        else:
            det_info, pos, io_err, _ = self.sqw_serializer_.calculate_positions(
                det_form, self.file_id_, pos, return_error=True
            )
            if io_err:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:io_error",
                    "IO error while parsing detector parameters",
                )
        self.detpar_pos_info_ = det_info
        self.data_pos_ = int(pos)

    def _init_pix_info(self, *args: Any) -> None:
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()

        pix_form = self.get_pix_form()
        from_obj = len(args) > 0 and self._is_sqw_or_dnd_obj(args[0])
        if from_obj:
            inobj = args[0]
            pix_obj, _ = self.extract_correct_subobj("pix", inobj)
            if pix_obj is None:
                pix_obj = PixelDataMemory()
            pix_input = {
                "img_range": np.zeros((2, 4), dtype=np.float32),
                "pix_block": pix_obj,
            }
            pix_pos, pos = self.sqw_serializer_.calculate_positions(
                pix_form, pix_input, self.dnd_eof_pos_
            )
        else:
            pix_pos, pos, io_err, _ = self.sqw_serializer_.calculate_positions(
                pix_form, self.file_id_, self.dnd_eof_pos_, return_error=True
            )
            if io_err and "pix_block_pos_" not in pix_pos:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:io_error",
                    "IO error while parsing pixel block header",
                )

        self.img_db_range_pos_ = int(pix_pos["img_range_pos_"])
        self.pix_pos_ = int(pix_pos["pix_block_pos_"]) + 8
        self.eof_pix_pos_ = int(pos)
        self.npixels_ = (self.eof_pix_pos_ - int(self.pix_pos_)) // self.FILE_PIX_SIZE
        self.data_type_ = "a"

    def do_class_dependent_updates(
        self, new_ldr: "sqw_binfile_common", *args: Any
    ) -> "sqw_binfile_common":
        if self.faccess_version != new_ldr.faccess_version:
            sqw_obj = self.get_sqw()
            new_ldr.sqw_holder = sqw_obj
            new_ldr = new_ldr.put_sqw()
            self.delete()
        return new_ldr

    # ------------------------------------------------------------------
    # Reading Methods
    # ------------------------------------------------------------------

    def get_main_header(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read `main_header_cl` from the binary SQW file."""
        keep_original = False
        for arg in args:
            if isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if (
                    s in ("keep_original", "verbatim", "hverbatim")
                    or s.startswith("keep")
                    or s.startswith("verb")
                ):
                    keep_original = True
                elif s in (
                    "head",
                    "header",
                    "his",
                    "history",
                    "legacy",
                    "sqw_struc",
                    "nopix",
                    "all",
                ):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:sqw_binfile_common:invalid_argument",
                        f"get_main_header: Unrecognized option {arg!r}",
                    )

        if isinstance(self.num_contrib_files, str):
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "get_main_header called on un-initialized loader",
            )

        if isinstance(self.header_pos_, list) and len(self.header_pos_) > 0:
            end_pos = int(self.header_pos_[0])
        elif isinstance(self.header_pos_, int) and self.header_pos_ > 0:
            end_pos = int(self.header_pos_)
        else:
            end_pos = int(self.detpar_pos_)
        sz = end_pos - int(self.main_header_pos_)

        do_fseek(self.file_id_, int(self.main_header_pos_), "bof")
        bytes_arr, _ = do_fread(self.file_id_, sz, "*uint8")
        self.check_error_report_fail_("get_main_header: Error reading main header")

        mh_form = self.get_main_header_form()
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        mh_dict, _ = self.sqw_serializer_.deserialize_bytes(bytes_arr, mh_form, 0)
        if self.convert_to_double:
            mh_dict = _do_convert_to_double(mh_dict)
        if not keep_original:
            orig_fn_cdate = str(
                mh_dict.get("filename_with_cdate", mh_dict.get("filename", ""))
            )
            pos = orig_fn_cdate.find("$")
            if pos == -1:
                mh_dict["filename_with_cdate"] = self.filename
            else:
                mh_dict["filename_with_cdate"] = f"{self.filename}${orig_fn_cdate[pos + 1 :]}"
            mh_dict["filepath"] = self.filepath

        res = main_header_cl(mh_dict)
        return pack_io_outputs((res, self), nargout, default=1)

    def get_exp_info(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read `Experiment` from the binary SQW file."""
        get_all = False
        header_number: Optional[int] = None

        for arg in args:
            if isinstance(arg, (int, np.integer)) and not isinstance(arg, (bool, np.bool_)):
                header_number = int(arg)
            elif isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if s in ("a", "al", "all"):
                    get_all = True
                elif s in (
                    "head",
                    "header",
                    "his",
                    "history",
                    "verbatim",
                    "hverbatim",
                    "keep_original",
                    "nopix",
                    "legacy",
                    "sqw_struc",
                    "no_isvalid_check",
                ):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:sqw_binfile_common:invalid_argument",
                        f"get_exp_info: Unrecognized option {arg!r}",
                    )
            else:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:invalid_argument",
                    f"get_exp_info: Invalid argument {arg!r}",
                )

        n_files = self.num_contrib_files
        if isinstance(n_files, str):
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "get_exp_info called on un-initialized loader",
            )
        if n_files == 0:
            return pack_io_outputs((Experiment(), self.detpar_pos_), nargout, default=1)

        if header_number is not None:
            if header_number < 1 or header_number > n_files:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:invalid_argument",
                    f"Requested header number {header_number} is outside range [1, {n_files}]",
                )
            indices = [header_number - 1]
        elif get_all and n_files > 1:
            indices = list(range(n_files))
        else:
            indices = [0]

        hdr_form = self.get_header_form()
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()

        headers_raw: List[Dict[str, Any]] = []
        last_pos = int(self.detpar_pos_)
        for idx0 in indices:
            start_pos = int(self.header_pos_[idx0])
            if idx0 + 1 < n_files:
                end_pos = int(self.header_pos_[idx0 + 1])
            else:
                end_pos = int(self.detpar_pos_)
            sz = end_pos - start_pos
            last_pos = end_pos

            do_fseek(self.file_id_, start_pos, "bof")
            bytes_arr, _ = do_fread(self.file_id_, sz, "*uint8")
            self.check_error_report_fail_(f"get_exp_info: Error reading header #{idx0 + 1}")

            hdr_dict, _ = self.sqw_serializer_.deserialize_bytes(bytes_arr, hdr_form, 0)
            if self.convert_to_double:
                hdr_dict = _do_convert_to_double(hdr_dict)
            if "u_to_rlu" in hdr_dict:
                hdr_dict["u_to_rlu_legacy"] = hdr_dict["u_to_rlu"]
            if "uoffset" in hdr_dict:
                hdr_dict["offset"] = hdr_dict["uoffset"]
            headers_raw.append(hdr_dict)

        header_arg: Any = headers_raw[0] if len(headers_raw) == 1 else headers_raw
        exp_info = Experiment.build_from_binfile_headers(header_arg)
        self.contains_runid_in_header_ = not bool(exp_info.runid_recalculated)

        detpar = self.get_detpar(nargout=1)
        detpar = detpar.replicate_runs(exp_info.n_runs)
        exp_info.detector_arrays = detpar

        return pack_io_outputs((exp_info, last_pos), nargout, default=1)

    def get_detpar(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read detector parameters as `unique_objects_container` of `IX_detector_array`."""
        if isinstance(self.num_contrib_files, str) or self.detpar_pos_ <= 0:
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "get_detpar called on un-initialized loader",
            )
        sz = int(self.data_pos_) - int(self.detpar_pos_)
        do_fseek(self.file_id_, int(self.detpar_pos_), "bof")
        bytes_arr, _ = do_fread(self.file_id_, sz, "*uint8")
        self.check_error_report_fail_("get_detpar: Error reading detector parameters")

        det_form = self.get_detpar_form()
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        det, _ = self.sqw_serializer_.deserialize_bytes(bytes_arr, det_form, 0)
        if self.convert_to_double:
            det = _do_convert_to_double(det)

        for k in ("group", "x2", "phi", "azim", "width", "height"):
            if k in det and isinstance(det[k], np.ndarray):
                det[k] = det[k].ravel()

        detpar = self.convert_old_det_forms(det, 1)
        return pack_io_outputs((detpar, self), nargout, default=1)

    def get_data(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read `data_sqw_dnd` (or header dict) from the binary SQW file and populate `axes.img_range`."""
        opts, argi = _parse_get_data_args(*args)
        data_str = super().get_data(*argi, nargout=1)
        if not (opts["header"] or opts["noclass"]):
            data_str.axes.img_range = self.get_img_db_range(data_str)
        return pack_io_outputs((data_str, self), nargout, default=1)

    def _read_img_range(self) -> np.ndarray:
        """Read `2 x 4` `img_db_range` array directly from the binary SQW file."""
        if self.img_db_range_pos_ > 0 and self.file_id_ > 0:
            do_fseek(self.file_id_, int(self.img_db_range_pos_), "bof")
            raw, cnt = do_fread(self.file_id_, 8, "float32")
            _, res = do_ferror(self.file_id_)
            if res == 0 and cnt == 8:
                return raw.reshape((2, 4), order="F").astype(np.float64)
        return super().get_img_db_range()

    def get_img_db_range(self, *args: Any) -> np.ndarray:
        """Get `2 x 4` min/max image range from the file (0 args) or from `data_str` axes (1 arg)."""
        if len(args) == 0:
            return self._read_img_range()
        data_str = args[0]
        if hasattr(data_str, "axes") and hasattr(data_str.axes, "img_range"):
            img_db_range = np.asarray(data_str.axes.img_range, dtype=np.float64).copy()
        else:
            img_db_range = line_axes.calc_img_db_range(data_str)
        if np.any(np.isinf(img_db_range)):
            img_data_range = self._read_img_range()
            undef = np.isinf(img_db_range)
            img_db_range[undef] = img_data_range[undef]
        return img_db_range

    def get_pix_range(self) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float64)

    def get_data_range(self, *args: Any) -> np.ndarray:
        return PixelDataBase.EMPTY_RANGE.copy()

    def get_pix_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        meta = pix_metadata()
        meta.full_filename = self.full_filename
        meta.npix = int(self.npixels) if isinstance(self.npixels, int) else 0
        meta.data_range = self.get_data_range()
        return pack_io_outputs((meta, self), nargout, default=1)

    def get_raw_pix(
        self, pix_lo: int = 1, pix_hi: Optional[int] = None
    ) -> np.ndarray:
        """Read raw `(9, N)` pixel block `[pix_lo, pix_hi]` (1-based inclusive)."""
        if isinstance(self.pix_position, str) or isinstance(self.npixels, str):
            raise HoraceError(
                "HORACE:sqw_binfile_common:runtime_error",
                "Cannot read pixels from uninitialized SQW loader",
            )
        total_pix = int(self.npixels)
        if pix_hi is None:
            pix_hi = total_pix
        pix_lo_i = int(pix_lo)
        pix_hi_i = int(pix_hi)
        if total_pix == 0 or pix_hi_i < pix_lo_i:
            return np.empty((9, 0), dtype=np.float64 if self.convert_to_double else np.float32)
        if pix_lo_i < 1 or pix_hi_i > total_pix:
            raise HoraceError(
                "HORACE:sqw_binfile_common:invalid_argument",
                f"Pixel range [{pix_lo_i}, {pix_hi_i}] out of bounds [1, {total_pix}]",
            )
        n_pix = pix_hi_i - pix_lo_i + 1
        seek_pos = int(self.pix_position) + (pix_lo_i - 1) * self.FILE_PIX_SIZE
        do_fseek(self.file_id_, seek_pos, "bof")
        raw, cnt = do_fread(self.file_id_, n_pix * 9, "*float32")
        self.check_error_report_fail_("get_raw_pix: Error reading pixel data")
        if cnt < n_pix * 9:
            raise HoraceError(
                "HORACE:sqw_binfile_common:io_error",
                f"Expected {n_pix * 9} float32 values, read {cnt}",
            )
        arr = raw.reshape((9, n_pix), order="F")
        return arr.astype(np.float64) if self.convert_to_double else arr

    def get_pix(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read `PixelDataMemory` for pixel range `[pix_lo, pix_hi]` (1-based inclusive)."""
        if isinstance(self.pix_position, str) or isinstance(self.npixels, str) or int(self.npixels) == 0:
            return pack_io_outputs((PixelDataBase.create(), self), nargout, default=1)

        total_pix = int(self.npixels)
        num_args = [a for a in args if isinstance(a, (int, np.integer, float, np.floating))]
        if len(num_args) == 0:
            pix_lo, pix_hi = 1, total_pix
        elif len(num_args) == 1:
            pix_lo = int(num_args[0])
            pix_hi = total_pix
        else:
            pix_lo = int(num_args[0])
            pix_hi = int(num_args[1])

        raw = self.get_raw_pix(pix_lo, pix_hi)
        pix = PixelDataBase.create(raw)
        pix.full_filename = self.full_filename
        return pack_io_outputs((pix, self), nargout, default=1)

    def get_pix_at_indices(self, indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Read pixels at 1-based `indices` as `(9, len(indices))` array."""
        idx_arr = np.asarray(indices, dtype=np.int64).ravel()
        if idx_arr.size == 0:
            return np.empty((9, 0), dtype=np.float64 if self.convert_to_double else np.float32)
        total_pix = int(self.npixels) if isinstance(self.npixels, int) else 0
        if np.any(idx_arr < 1) or np.any(idx_arr > total_pix):
            raise HoraceError(
                "SQW_BINFILE_COMMON:get_pix_at_indices",
                f"Pixel indices out of bounds [1, {total_pix}]",
            )
        diffs = np.diff(idx_arr)
        breaks = np.where(diffs != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [idx_arr.size]))
        pix_starts = idx_arr[starts]
        block_sizes = ends - starts
        return self.get_pix_in_ranges(pix_starts, block_sizes, skip_arg_validation=True)

    def get_pix_in_ranges(
        self,
        pix_starts: Union[Sequence[int], np.ndarray],
        block_sizes: Union[Sequence[int], np.ndarray],
        skip_arg_validation: bool = False,
        keep_precision: bool = False,
    ) -> np.ndarray:
        """Read multiple pixel blocks starting at 1-based `pix_starts` with lengths `block_sizes`."""
        starts = np.asarray(pix_starts, dtype=np.int64).ravel()
        sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
        if not skip_arg_validation:
            if starts.size != sizes.size:
                raise HoraceError(
                    "HORACE:validate_ranges:invalid_argument",
                    "pix_starts and block_sizes must have equal length",
                )
            if np.any(starts < 1) or np.any(sizes <= 0):
                raise HoraceError(
                    "HORACE:validate_ranges:invalid_argument",
                    "pix_starts must be >= 1 and block_sizes must be > 0",
                )
        blocks: List[np.ndarray] = []
        for st, sz in zip(starts, sizes):
            if sz > 0:
                blocks.append(self.get_raw_pix(int(st), int(st + sz - 1)))
        if not blocks:
            return np.empty((9, 0), dtype=np.float64 if self.convert_to_double else np.float32)
        return np.hstack(blocks)

    def get_sqw(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read full `SQW` object (or header/struct based on flags) from the binary SQW file."""
        from pyhorace.sqw.sqw import SQW

        opts = self.parse_get_sqw_args(*args)
        mh_args: List[str] = []
        data_args: List[str] = []
        if opts["verbatim"] or opts["hverbatim"]:
            mh_args.append("-verbatim")
            data_args.append("-verbatim")
        if opts["head"] or opts["his"]:
            data_args.append("-head")

        main_header = self.get_main_header(*mh_args, nargout=1)
        exp_info = self.get_exp_info("-all", nargout=1)
        detpar = exp_info.detector_arrays
        data = self.get_data(*data_args, nargout=1)

        if not (opts["head"] or opts["his"]):
            hav = exp_info.header_average()
            al_info = self._dnd_data_alignment(data, hav)
            if al_info is not None:
                data.proj = al_info

        if opts["head"] or opts["his"] or opts["nopix"] or not isinstance(self.npixels, int) or self.npixels == 0:
            pix: Any = PixelDataBase.create()
        elif opts["file_backed"]:
            pix = PixelDataFileBacked(self)
        elif opts["force_pix_location"] and not opts["file_backed"]:
            pix = PixelDataMemory(self)
        else:
            pix = PixelDataBase.create(self)

        sqw_struc: Dict[str, Any] = {
            "main_header": main_header,
            "experiment_info": exp_info,
            "detpar": detpar,
            "data": data,
            "pix": pix,
        }

        if (
            not main_header.creation_date_defined
            and not (opts["head"] or opts["his"] or opts["nopix"])
            and pix.num_pixels > 0
        ):
            sqw_struc = _update_pixels_run_id(sqw_struc)
            exp_info = sqw_struc["experiment_info"]

        if opts["head"] or opts["his"]:
            sqw_struc["npixels"] = self.npixels
            return pack_io_outputs(
                (sqw_struc, main_header, exp_info, detpar, data), nargout, default=1
            )

        if opts["sqw_struc"]:
            return pack_io_outputs((sqw_struc, self), nargout, default=1)

        sqw_obj = SQW(sqw_struc)
        if opts["legacy"]:
            return pack_io_outputs(
                (sqw_obj, main_header, exp_info, detpar, data, pix), nargout, default=1
            )
        return pack_io_outputs((sqw_obj, self), nargout, default=1)

    @staticmethod
    def _dnd_data_alignment(data_obj: Any, hav: Dict[str, Any]) -> Any:
        """Align legacy line_proj lattice parameters with experiment header average when needed."""
        proj = getattr(data_obj, "proj", None)
        if proj is None or not getattr(proj, "undeefined_fields_from_data_file", False):
            return None
        alatt = np.asarray(hav.get("alatt", proj.alatt), dtype=np.float64).ravel()
        angdeg = np.asarray(hav.get("angdeg", proj.angdeg), dtype=np.float64).ravel()
        if not np.allclose(alatt, np.asarray(proj.alatt).ravel(), atol=1e-6):
            proj.alatt = alatt
        if not np.allclose(angdeg, np.asarray(proj.angdeg).ravel(), atol=1e-6):
            proj.angdeg = angdeg
        return proj

    def head(self, *args: Any) -> Dict[str, Any]:
        """Return summary dictionary of SQW header information."""
        from pyhorace.sqw.sqw import SQW

        full = False
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("f"):
                full = True
            else:
                raise HoraceError(
                    "HORACE:sqw_binfile_common:invalid_argument",
                    f"Unrecognized option {a!r} in head",
                )
        sqw_data = self.get_sqw("-head", "-verbatim", nargout=1)
        flds = SQW.head_form(full)
        hd: Dict[str, Any] = {}
        mh = sqw_data["main_header"]
        data = sqw_data["data"]
        if isinstance(data, dict):
            if "u_to_rlu" not in data and "u_to_rlu_legacy" in data:
                data["u_to_rlu"] = data["u_to_rlu_legacy"]
            if "offset" not in data and "uoffset" in data:
                data["offset"] = data["uoffset"]
        for fld in flds:
            if hasattr(mh, fld):
                hd[fld] = getattr(mh, fld)
            elif fld == "experiment_info":
                hd[fld] = sqw_data["experiment_info"]
            elif fld == "detpar":
                hd[fld] = sqw_data["detpar"]
            elif fld == "npixels":
                hd[fld] = sqw_data["npixels"]
            elif isinstance(data, dict) and fld in data:
                hd[fld] = data[fld]
            elif hasattr(data, fld):
                hd[fld] = getattr(data, fld)
        return hd

    # ------------------------------------------------------------------
    # Writing Methods
    # ------------------------------------------------------------------

    def put_main_header(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write `main_header` record to the binary SQW file."""
        update, argi = self.parse_updt_directions(*args)
        self.check_obj_initated_properly()
        sub_obj, new_obj = self.extract_correct_subobj("main_header", *argi)
        if new_obj and update and not self.upgrade_mode:
            self.upgrade_mode = True

        if isinstance(sub_obj, main_header_cl):
            if not sub_obj.creation_date_defined and self.upgrade_headers_:
                import datetime

                sub_obj.creation_date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            mh_struct: Any = sub_obj
        else:
            mh_struct = sub_obj

        if update and not self.upgrade_headers_:
            mh_form = self.get_main_header_form("-const")
            start_pos = int(self.main_head_pos_info_["nfiles_pos_"])
        else:
            mh_form = self.get_main_header_form()
            start_pos = int(self.main_header_pos_)

        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        bytes_arr = self.sqw_serializer_.serialize(mh_struct, mh_form)

        if update and self.upgrade_mode:
            self.upgrade_map_.check_equal_sizes("main_header", len(bytes_arr))

        do_fseek(self.file_id_, start_pos, "bof")
        do_fwrite(self.file_id_, bytes_arr, "uint8")
        self.check_error_report_fail_("put_main_header: Error writing main header")
        return self

    def put_headers(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write contributing experiment headers to the binary SQW file."""
        update, argi = self.parse_updt_directions(*args)
        header_num = 0
        rem_args: List[Any] = []
        for arg in argi:
            if isinstance(arg, (int, np.integer)) and not isinstance(arg, (bool, np.bool_)):
                header_num = int(arg)
            else:
                rem_args.append(arg)

        self.check_obj_initated_properly()
        exp_info, new_obj = self.extract_correct_subobj("experiment_info", *rem_args)
        if new_obj and update and not self.upgrade_mode:
            self.upgrade_mode = True

        opt = () if self.contains_runid_in_header else ("-nomangle",)
        if isinstance(exp_info, Experiment):
            headers = exp_info.convert_to_old_headers(header_num, *opt)
            if isinstance(headers, dict):
                headers = [headers]
            else:
                headers = list(headers)
        elif isinstance(exp_info, (list, tuple)):
            headers = list(exp_info)
        else:
            headers = [exp_info]

        if update and not self.upgrade_headers_:
            hdr_form = self.get_header_form("-const")
            pos_list = [int(x["efix_pos_"]) for x in self.header_pos_info_]
        else:
            hdr_form = self.get_header_form()
            pos_list = (
                [int(x) for x in self.header_pos_]
                if isinstance(self.header_pos_, list)
                else [int(self.header_pos_)]
            )

        if header_num > 0:
            if header_num > len(pos_list):
                raise HoraceError(
                    "HORACE:sqw_binfile_common:invalid_argument",
                    f"Header number {header_num} out of range [1, {len(pos_list)}]",
                )
            pos_list = [pos_list[header_num - 1]]
            if len(headers) > 1:
                headers = [headers[header_num - 1]]

        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()

        for i, hdr_i in enumerate(headers):
            if hasattr(hdr_i, "convert_to_binfile_header"):
                hdr_i = hdr_i.convert_to_binfile_header(
                    "-alatt_angdeg", not self.contains_runid_in_header
                )
            bytes_arr = self.sqw_serializer_.serialize(hdr_i, hdr_form)
            if update and self.upgrade_mode:
                n_head = header_num if header_num > 0 else i + 1
                self.upgrade_map_.check_equal_sizes("exp_headers", len(bytes_arr), n_head)
            do_fseek(self.file_id_, pos_list[i], "bof")
            do_fwrite(self.file_id_, bytes_arr, "uint8")
            self.check_error_report_fail_(f"put_headers: Error writing header #{i + 1}")

        return self

    def put_det_info(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write detector parameter block (`detpar`) to the binary SQW file."""
        update, argi = self.parse_updt_directions(*args)
        self.check_obj_initated_properly()
        sub_obj, new_obj = self.extract_correct_subobj("detpar", *argi)
        if isinstance(sub_obj, unique_objects_container) or (
            hasattr(sub_obj, "n_runs") and hasattr(sub_obj, "unique_objects")
        ):
            if sub_obj.n_runs > 0:
                sub_obj = sub_obj[0].get_detpar_representation()
            else:
                sub_obj = IX_detector_array().get_detpar_representation()
        elif isinstance(sub_obj, IX_detector_array):
            sub_obj = sub_obj.get_detpar_representation()
        elif isinstance(sub_obj, Experiment):
            sub_obj = sub_obj.convert_to_old_detpar()

        if new_obj and update and not self.upgrade_mode:
            self.upgrade_mode = True

        if update and not self.upgrade_headers_:
            det_form = self.get_detpar_form("-const")
            start_pos = int(self.detpar_pos_info_["ndet_pos_"])
        else:
            det_form = self.get_detpar_form()
            start_pos = int(self.detpar_pos_)

        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        bytes_arr = self.sqw_serializer_.serialize(sub_obj, det_form)

        if update and self.upgrade_mode:
            self.upgrade_map_.check_equal_sizes("detpar", len(bytes_arr))

        do_fseek(self.file_id_, start_pos, "bof")
        do_fwrite(self.file_id_, bytes_arr, "uint8")
        self.check_error_report_fail_("put_det_info: Error writing detector parameters")
        return self

    def put_num_pixels(self, num_pixels: int) -> "sqw_binfile_common":
        """Write `uint64` pixel count at `pix_pos_`."""
        self.check_obj_initated_properly()
        do_fseek(self.file_id_, int(self.pix_pos_), "bof")
        do_fwrite(self.file_id_, np.array([int(num_pixels)], dtype="<u8"), "uint64")
        self.check_error_report_fail_("put_num_pixels: Error writing number of pixels")
        return self

    def put_raw_pix(
        self, pix_data: Union[np.ndarray, PixelDataBase], pix_idx: int = 1
    ) -> "sqw_binfile_common":
        """Write raw `(9, N)` pixel array starting at 1-based pixel index `pix_idx`."""
        self.check_obj_initated_properly()
        if isinstance(pix_data, PixelDataBase):
            arr = np.asarray(pix_data.data, dtype="<f4")
        else:
            arr = np.asarray(pix_data, dtype="<f4")
        if arr.size == 0:
            return self
        seek_pos = int(self.pix_position) + (int(pix_idx) - 1) * self.FILE_PIX_SIZE
        do_fseek(self.file_id_, seek_pos, "bof")
        do_fwrite(self.file_id_, arr.ravel(order="F"), "float32")
        self.check_error_report_fail_("put_raw_pix: Error writing pixel array")
        return self

    def put_pix_metadata(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write pixel metadata (`npixels` count in v2/v3 formats)."""
        if len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, pix_metadata):
                npix = int(arg0.npix)
            elif isinstance(arg0, PixelDataBase):
                npix = int(arg0.num_pixels)
            else:
                npix = int(self.npixels) if isinstance(self.npixels, int) else 0
        else:
            npix = int(self.npixels) if isinstance(self.npixels, int) else 0
        return self.put_num_pixels(npix)

    def put_pix(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write full pixel block (`img_range` + `dummy` + `uint64(npix)` + `(9, N)` float32 pixel data)."""
        update, argi = self.parse_updt_directions(*args)
        nopix = False
        reserve = False
        rem_args: List[Any] = []
        for arg in argi:
            if isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if s.startswith("nop"):
                    nopix = True
                elif s.startswith("res"):
                    reserve = True
            else:
                rem_args.append(arg)

        if nopix:
            return self

        self.check_obj_initated_properly()

        # Write img_range and dummy field before pixel block unless updating pixels only
        if not update:
            if len(rem_args) > 0 and isinstance(rem_args[0], PixelDataBase):
                data_obj, _ = self.extract_correct_subobj("data")
            else:
                data_obj, _ = self.extract_correct_subobj("data", *rem_args)
            if data_obj is not None:
                if hasattr(data_obj, "axes") and hasattr(data_obj.axes, "img_range"):
                    img_range = np.asarray(data_obj.axes.img_range, dtype=np.float32)
                elif isinstance(data_obj, dict) and "img_range" in data_obj:
                    img_range = np.asarray(data_obj["img_range"], dtype=np.float32)
                else:
                    img_range = np.zeros((2, 4), dtype=np.float32)
            else:
                img_range = np.zeros((2, 4), dtype=np.float32)

            head_pix_format = {
                "img_range": np.array([2, 4], dtype=np.float32),
                "dummy": field_not_in_structure("img_range"),
            }
            if self.sqw_serializer_ is None:
                self.sqw_serializer_ = sqw_serializer()
            head_bytes = self.sqw_serializer_.serialize(
                {"img_range": img_range}, head_pix_format
            )
            do_fseek(self.file_id_, int(self.img_db_range_pos_), "bof")
            do_fwrite(self.file_id_, head_bytes, "uint8")
            self.check_error_report_fail_("put_pix: Error writing img_range header")

        sub_obj, new_obj = self.extract_correct_subobj("pix", *rem_args)
        if sub_obj is None:
            sub_obj = PixelDataMemory()

        if new_obj and update and not self.upgrade_mode:
            self.upgrade_mode = True

        if isinstance(sub_obj, PixelDataBase):
            npix = int(sub_obj.num_pixels)
        else:
            arr = np.asarray(sub_obj)
            npix = int(arr.shape[1]) if arr.ndim == 2 else int(arr.size // 9)

        if update and self.upgrade_mode:
            self.upgrade_map_.check_equal_sizes("pix", npix * self.FILE_PIX_SIZE + 8)

        # Write uint64(npix) at pix_pos_ - 8
        npix_pos = int(self.pix_pos_) - 8
        do_fseek(self.file_id_, npix_pos, "bof")
        do_fwrite(self.file_id_, np.array([npix], dtype="<u8"), "uint64")
        self.check_error_report_fail_("put_pix: Error writing number of pixels")
        self.npixels_ = npix
        self.eof_pix_pos_ = int(self.pix_pos_) + npix * self.FILE_PIX_SIZE

        if npix == 0:
            return self

        if reserve:
            chunk_pix = min(npix, 100000)
            zeros = np.zeros((9, chunk_pix), dtype=np.float32)
            written = 0
            while written < npix:
                n_write = min(chunk_pix, npix - written)
                self.put_raw_pix(zeros[:, :n_write], written + 1)
                written += n_write
            return self

        if isinstance(sub_obj, PixelDataBase):
            if sub_obj.num_pages > 1:
                written = 0
                for page_idx in range(1, sub_obj.num_pages + 1):
                    sub_obj.page_num = page_idx
                    page_data = sub_obj.get_pixels(
                        range(1, sub_obj.page_size + 1), "-raw", "-align"
                    )
                    self.put_raw_pix(page_data, written + 1)
                    written += page_data.shape[1]
            else:
                self.put_raw_pix(sub_obj.data, 1)
        else:
            self.put_raw_pix(np.asarray(sub_obj), 1)
        return self

    def put_dnd(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Save SQW object (or its data section) into file as a DND object."""
        pos = self.get_pos_info()
        update_args: List[str] = []
        inobj: Any = None
        internal_obj = True
        for a in args:
            if isinstance(a, str):
                s = a.strip().lower().lstrip("-")
                if s.startswith("u"):
                    update_args.append("-update")
                elif s.startswith("nop"):
                    update_args.append("-nopix")
            elif self._is_sqw_or_dnd_obj(a):
                inobj = a
                internal_obj = False
        if inobj is None:
            inobj = self.sqw_holder_
            if inobj is None and self.file_id_ > 0:
                inobj = self.get_data("-verbatim", nargout=1)
        dnd_inobj, _ = self.extract_correct_subobj("data", inobj)
        if self.read_only and self.file_id_ > 0:
            self.reopen_to_write()
        self.data_pos_ = 26
        self.data_type_ = "b+"
        self._init_from_sqw(dnd_inobj)
        super().put_dnd(*update_args, dnd_inobj)
        for fld, val in pos.items():
            setattr(self, fld, val)
        if internal_obj:
            self.sqw_holder_ = inobj
        return self

    def put_sqw(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        """Write complete SQW object (main header, headers, detpar, data, pixels) to file."""
        update_args: List[str] = []
        nopix = False
        reserve = False
        obj_args: List[Any] = []
        for arg in args:
            if isinstance(arg, str):
                s = arg.strip().lower().lstrip("-")
                if s.startswith("u"):
                    update_args = ["-update"]
                elif s.startswith("nop"):
                    nopix = True
                elif s.startswith("res"):
                    reserve = True
                elif s in ("v2", "v3", "verbatim"):
                    pass
            elif self._is_sqw_or_dnd_obj(arg):
                obj_args = [arg]

        if obj_args and not update_args:
            self.init_from_sqw_obj(obj_args[0])

        self.check_obj_initated_properly()
        if not update_args:
            self.put_app_header()
        self.put_main_header(*update_args, *obj_args)
        self.put_headers(*update_args, *obj_args)
        self.put_det_info(*update_args, *obj_args)
        self.put_dnd_metadata(*update_args, *obj_args)
        self.put_dnd_data(*update_args, *obj_args)
        if not nopix:
            pix_flags: List[Any] = list(update_args) + list(obj_args)
            if reserve:
                pix_flags.append("-reserve")
            self.put_pix(*pix_flags)
        return self

    def put_instruments(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        raise HoraceError(
            "HORACE:sqw_binfile_common:not_implemented",
            f"Saving instruments is not supported by file format version {self.faccess_version}",
        )

    def put_samples(self, *args: Any, **kwargs: Any) -> "sqw_binfile_common":
        raise HoraceError(
            "HORACE:sqw_binfile_common:not_implemented",
            f"Saving samples is not supported by file format version {self.faccess_version}",
        )
