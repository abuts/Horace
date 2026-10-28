"""Common base class for Horace v2/v3 DND and SQW binary file accessors.

Ported from:
- horace_core/sqw/file_io/@binfile_v2_common/binfile_v2_common.m
- horace_core/sqw/file_io/@binfile_v2_common/*.m
- horace_core/sqw/file_io/@binfile_v2_common/private/*.m
"""

from __future__ import annotations

import copy
import datetime
import os
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import Serializable
from pyhorace.io.const_blocks_map import const_blocks_map
from pyhorace.io.horace_binfile_interface import horace_binfile_interface
from pyhorace.io.sqw_fields_formatters import (
    field_cellarray_of_axis,
    field_cellarray_of_strings,
    field_const_array_dependent,
    field_iax,
    field_iint,
    field_img,
    field_not_in_structure,
    field_p_size,
)
from pyhorace.io.sqw_fopen import (
    do_ferror,
    do_fread,
    do_fseek,
    do_ftell,
    do_fwrite,
    pack_io_outputs,
    sqw_fclose,
    sqw_fopen,
    sqw_fopen_info,
)
from pyhorace.io.sqw_serializer import sqw_serializer


def _parse_get_data_args(*args: Any) -> Tuple[Dict[str, bool], List[str]]:
    """Parse option flags passed to `get_data`."""
    header_only = False
    verbatim = False
    noclass = False
    argi: List[str] = []

    for arg in args:
        if not isinstance(arg, str):
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"get_data: Invalid argument: {arg!r}",
            )
        s = arg.strip().lower().lstrip("-")
        if s in ("h", "he", "hea", "head", "heade", "header", "his", "history"):
            header_only = True
            argi.append(arg)
        elif s in ("v", "ve", "ver", "verb", "verba", "verbat", "verbati", "verbatim"):
            verbatim = True
            argi.append(arg)
        elif s in ("hv", "hve", "hver", "hverb", "hverba", "hverbat", "hverbati", "hverbatim"):
            header_only = True
            verbatim = True
            argi.append(arg)
        elif s.startswith("noc") or s in ("legacy", "sqw_struc"):
            noclass = True
            argi.append(arg)
        elif s.startswith("nou") or s.startswith("nop") or s.startswith("slo"):
            argi.append(arg)
        else:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"get_data: Unrecognized option: {arg!r}",
            )
    return {"header": header_only, "verbatim": verbatim, "noclass": noclass}, argi


def _process_format_fields(*args: Any) -> Dict[str, Any]:
    """Build v2/v3 DND format descriptor dictionary."""
    head_only = False
    constant_len_fields = False
    data_only = False

    for arg in args:
        if not isinstance(arg, str):
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"get_head_form: invalid argument: {arg!r}",
            )
        s = arg.strip().lower().lstrip("-")
        if s in ("h", "he", "hea", "head", "heade", "header"):
            head_only = True
        elif s in ("c", "co", "con", "cons", "const"):
            constant_len_fields = True
        elif s in ("d", "da", "dat", "data"):
            data_only = True
        else:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"get_head_form: invalid argument: {arg!r}",
            )

    if (head_only or constant_len_fields) and data_only:
        raise HoraceError(
            "HORACE:binfile_v2_common:invalid_argument",
            "get_head_form: -data and -head or -const fields can not be provided together",
        )

    var_fields: Dict[str, Any] = {
        "filename": "",
        "filepath": "",
        "title": "",
    }
    const_fields: Dict[str, Any] = {
        "alatt": np.array([1, 3], dtype=np.float32),
        "angdeg": np.array([1, 3], dtype=np.float32),
        "uoffset": np.array([4, 1], dtype=np.float32),
        "u_to_rlu_legacy": np.array([4, 4], dtype=np.float32),
        "ulen": np.array([1, 4], dtype=np.float32),
        "label": field_cellarray_of_strings(),
        "npax": field_not_in_structure("pax"),
        "iax": field_iax(),
        "iint": field_iint(),
        "pax": field_const_array_dependent("npax", 1, "int32"),
        "p_size": field_p_size(),
        "p": field_cellarray_of_axis("npax"),
        "dax": field_const_array_dependent("npax", 1, "int32"),
    }
    data_fields: Dict[str, Any] = {
        "s": field_img(),
        "e": field_img(),
        "npix": field_img("uint64"),
    }

    res: Dict[str, Any] = {}
    if head_only:
        if not constant_len_fields:
            res.update(var_fields)
        res.update(const_fields)
    elif data_only:
        res.update(data_fields)
    else:
        if not constant_len_fields:
            res.update(var_fields)
        res.update(const_fields)
        res.update(data_fields)
    return res


def _do_convert_to_double(val: Any) -> Any:
    """Recursively convert single-precision (float32) and integer arrays in dict/list to float64."""
    if isinstance(val, dict):
        return {k: _do_convert_to_double(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_do_convert_to_double(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_do_convert_to_double(v) for v in val)
    if isinstance(val, np.ndarray):
        if val.dtype.kind in ("f", "i", "u"):
            return val.astype(np.float64)
        if val.dtype == object:
            out = np.empty_like(val, dtype=object)
            for idx, item in np.ndenumerate(val):
                out[idx] = _do_convert_to_double(item)
            return out
        return val
    if isinstance(val, (np.floating, np.integer)) and not isinstance(val, (bool, np.bool_)):
        return float(val)
    return val


class binfile_v2_common(horace_binfile_interface):
    """Common implementation for Horace v2/v3 DND and SQW binary file accessors."""

    _v2_fields_to_save: ClassVar[List[str]] = [
        "data_type_",
        "dnd_dimensions_",
        "data_pos_",
        "s_pos_",
        "e_pos_",
        "npix_pos_",
        "dnd_eof_pos_",
        "data_fields_locations_",
    ]
    fields_to_save_: ClassVar[List[str]] = horace_binfile_interface.fields_to_save_ + _v2_fields_to_save

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.data_pos_: int = 26
        self.s_pos_: int = 0
        self.e_pos_: int = 0
        self.npix_pos_: Union[int, str] = "undefined"
        self.dnd_eof_pos_: int = 0
        self.data_fields_locations_: Any = []
        self.real_eof_pos_: int = 0
        self.upgrade_map_: Any = None
        self.upgrade_headers_: bool = True
        self.convert_to_double_: bool = True
        self.data_type_: str = "undefined"
        self.dnd_dimensions_: Union[Tuple[int, ...], np.ndarray, str] = "undefined"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def upgrade_mode(self) -> bool:
        if self.upgrade_map_ is None:
            return False
        if isinstance(self.upgrade_map_, (list, tuple)) and len(self.upgrade_map_) == 0:
            return False
        return True

    @upgrade_mode.setter
    def upgrade_mode(self, mode: bool) -> None:
        self.upgrade_headers_ = False
        self._set_upgrade_mode(bool(mode))

    @property
    def data_position(self) -> int:
        return int(self.data_pos_)

    @property
    def s_position(self) -> int:
        return int(self.s_pos_)

    @property
    def e_position(self) -> int:
        return int(self.e_pos_)

    @property
    def data_type(self) -> str:
        return self.data_type_

    @property
    def convert_to_double(self) -> bool:
        return bool(self.convert_to_double_)

    @convert_to_double.setter
    def convert_to_double(self, val: bool) -> None:
        self.convert_to_double_ = bool(val)

    @property
    def dnd_dimensions(self) -> Union[Tuple[int, ...], np.ndarray, str]:
        return self.dnd_dimensions_

    # ------------------------------------------------------------------
    # Overloadable Getters
    # ------------------------------------------------------------------

    def get_sqw_type(self) -> bool:
        return False

    def get_format_for_object(self) -> str:
        return "dnd"

    def get_npix_position(self) -> Union[int, str]:
        return self.npix_pos_

    # ------------------------------------------------------------------
    # Lifecycle & State Management
    # ------------------------------------------------------------------

    def delete(self) -> "binfile_v2_common":
        super().delete()
        self.data_type_ = "undefined"
        self.dnd_dimensions_ = "undefined"
        self.real_eof_pos_ = 0
        self.upgrade_map_ = None
        self.upgrade_headers_ = True
        self.data_pos_ = 26
        self.s_pos_ = 0
        self.e_pos_ = 0
        self.npix_pos_ = "undefined"
        self.dnd_eof_pos_ = 0
        self.data_fields_locations_ = []
        return self

    def deactivate(self) -> "binfile_v2_common":
        super().deactivate()
        self.upgrade_headers_ = True
        return self

    def get_sqw_internal(self) -> Any:
        return self.sqw_holder_

    def get_pos_info(self) -> Dict[str, Any]:
        fields = self.saveable_fields()
        return {fld: getattr(self, fld) for fld in fields}

    # ------------------------------------------------------------------
    # Initialization & Copying
    # ------------------------------------------------------------------

    def extract_correct_subobj(
        self, obj_name: str, *args: Any
    ) -> Tuple[Any, bool]:
        """Extract sub-object (`data`, `main_header`, `experiment_info`, `detpar`, `pix`) from SQW or DND."""
        from pyhorace.sqw.sqw import SQW

        if len(args) == 0 or args[0] is None:
            inobj = self.sqw_holder_
            external = False
        else:
            inobj = args[0]
            external = True

        if isinstance(inobj, SQW) or (
            hasattr(inobj, "main_header") and hasattr(inobj, "data")
        ):
            if obj_name == "experiment_info":
                if hasattr(inobj, "experiment_info"):
                    sub_obj = inobj.experiment_info
                else:
                    sub_obj = inobj.exper_info
            elif isinstance(inobj, dict):
                sub_obj = inobj[obj_name]
            else:
                sub_obj = getattr(inobj, obj_name)
        else:
            sub_obj = inobj
        return sub_obj, external

    def init_from_sqw_obj(self, *args: Any) -> "binfile_v2_common":
        """Initialize accessor positions from an in-memory DND or SQW object."""
        if len(args) == 0:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                "init_from_sqw method have to be invoked with one or two arguments",
            )
        if not self._is_sqw_or_dnd_obj(args[0]):
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                "first argument of init_from_sqw method has to be an sqw or dnd object",
            )
        if self.sqw_type:
            inobj, _ = self.extract_correct_subobj("data", *args)
        else:
            inobj = args[0]
        return self._init_from_sqw(inobj, *args[1:])

    def _init_from_sqw(self, inobj: Any, *args: Any) -> "binfile_v2_common":
        data_form = self.get_dnd_form()
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        pos_str, pos = self.sqw_serializer_.calculate_positions(
            data_form, inobj, self.data_pos_
        )
        self._process_dnd_Positions(pos_str, pos, inobj)
        self.sqw_holder_ = inobj
        if len(args) > 0 and isinstance(args[0], (str, os.PathLike)):
            self.full_filename = str(args[0])
        elif not self.filename_ and getattr(inobj, "filename", ""):
            self.filename_ = getattr(inobj, "filename", "")
            self.filepath_ = getattr(inobj, "filepath", "")
        return self

    def init_from_sqw_file(self, *args: Any) -> "binfile_v2_common":
        """Initialize DND field positions from an open binary file."""
        return self._init_dnd_structure_field_by_field(*args)

    def _init_dnd_structure_field_by_field(self, *args: Any) -> "binfile_v2_common":
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "File is not open when initializing DND structure",
            )
        try:
            do_fseek(self.file_id_, self.data_pos_, "bof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:binfile_v2_common:io_error",
                "Error moving to the beginning of the data record",
            ) from exc

        data_form = self.get_dnd_form()
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        data_pos, pos, _, data_form = self.sqw_serializer_.calculate_positions(
            data_form, self.file_id_, self.data_pos_, return_error=True
        )
        self._process_dnd_Positions(data_pos, pos, data_form)
        return self

    def _process_dnd_Positions(
        self, data_pos: Dict[str, Any], pos: int, data_form_or_obj: Any
    ) -> None:
        self.dnd_eof_pos_ = int(pos)
        self.eof_pos_ = int(pos)
        self.data_fields_locations_ = data_pos

        self.s_pos_ = int(data_pos["s_pos_"])
        self.e_pos_ = int(data_pos["e_pos_"])

        if isinstance(data_form_or_obj, dict) and isinstance(data_form_or_obj.get("s"), field_img):
            s_fmt: field_img = data_form_or_obj["s"]
            dims = s_fmt.dimension
            self.dnd_dimensions_ = tuple(int(x) for x in dims.ravel(order="F"))
            self.num_dim_ = int(s_fmt.num_dims)
        else:
            s_arr = (
                data_form_or_obj["s"]
                if isinstance(data_form_or_obj, dict)
                else getattr(data_form_or_obj, "s")
            )
            p_arr = (
                data_form_or_obj["p"]
                if isinstance(data_form_or_obj, dict)
                else getattr(data_form_or_obj, "p")
            )
            self.num_dim_ = int(len(p_arr))
            if self.num_dim_ == 0:
                self.dnd_dimensions_ = (1,)
            elif self.num_dim_ == 1:
                self.dnd_dimensions_ = (int(np.asarray(s_arr).size),)
            else:
                self.dnd_dimensions_ = tuple(int(x) for x in np.asarray(s_arr).shape[: self.num_dim_])

        if "npix_pos_" in data_pos:
            self.npix_pos_ = int(data_pos["npix_pos_"])
            self.data_type_ = "b+"
        else:
            self.data_type_ = "b"
            self.npix_pos_ = "undefined"
            self.dnd_eof_pos_ = int(data_pos["e_pos_"]) + int(self.e_pos_ - self.s_pos_)

        if self.file_id_ > 0:
            try:
                cur = do_ftell(self.file_id_)
                do_fseek(self.file_id_, 0, "eof")
                self.real_eof_pos_ = do_ftell(self.file_id_)
                do_fseek(self.file_id_, cur, "bof")
            except Exception:
                pass

    def copy_contents(
        self, other_obj: "horace_binfile_interface", keep_internals: Any = False, *args: Any, **kwargs: Any
    ) -> Tuple["binfile_v2_common", List[str]]:
        """Copy file positions and handle from `other_obj`."""
        write_mode = False
        all_flags = [keep_internals] + list(args)
        keep_int_bool = False
        for flg in all_flags:
            if isinstance(flg, bool):
                keep_int_bool = keep_int_bool or flg
            elif isinstance(flg, str) and flg.lower().startswith("-write"):
                write_mode = True

        return self._copy_contents_impl(other_obj, keep_int_bool, write_mode)

    def _copy_contents_impl(
        self,
        other_obj: Union["horace_binfile_interface", Dict[str, Any]],
        keep_internals: bool = False,
        write_mode: bool = False,
    ) -> Tuple["binfile_v2_common", List[str]]:
        this_pos = self.get_pos_info()
        if isinstance(other_obj, dict):
            other_pos = other_obj
        else:
            other_pos = other_obj.get_pos_info() if hasattr(other_obj, "get_pos_info") else other_obj.to_bare_struct()
        missing_fields: List[str] = []
        for fld in this_pos:
            if fld in other_pos:
                setattr(self, fld, copy.deepcopy(other_pos[fld]))
            else:
                missing_fields.append(fld)

        if not keep_internals and not isinstance(other_obj, dict):
            self.sqw_holder_ = getattr(other_obj, "sqw_holder_", None)
            self.real_eof_pos_ = getattr(other_obj, "real_eof_pos_", 0)
            self.eof_pos_ = getattr(other_obj, "eof_pos_", 0)
            self.upgrade_map_ = copy.deepcopy(getattr(other_obj, "upgrade_map_", None))
            self.convert_to_double_ = getattr(other_obj, "convert_to_double_", True)
            self.upgrade_headers_ = getattr(other_obj, "upgrade_headers_", True)
            self.data_in_file_ = getattr(other_obj, "data_in_file_", False)

        if not isinstance(other_obj, dict):
            if other_obj.file_id > 0:
                file, acc, _, _ = sqw_fopen_info(other_obj.file_id)
                if file:
                    self.full_filename = file
                    if write_mode:
                        self.reopen_to_write(file)
                    else:
                        self.file_id_ = sqw_fopen(file, acc)
                        self.read_only_ = acc in ("rb", "r")
            elif other_obj.full_filename:
                self.full_filename = other_obj.full_filename
                if write_mode and os.path.isfile(self.full_filename):
                    self.reopen_to_write(self.full_filename)

        return self, missing_fields

    def do_class_dependent_updates(
        self, new_ldr: "horace_binfile_interface", *args: Any
    ) -> "horace_binfile_interface":
        if self.faccess_version != new_ldr.faccess_version:
            dnd_obj = self.get_dnd()
            new_ldr.sqw_holder = dnd_obj
            new_ldr = new_ldr.put_dnd()
            self.delete()
        return new_ldr

    # ------------------------------------------------------------------
    # File Opening / Upgrade Map
    # ------------------------------------------------------------------

    def set_file_to_update(
        self, filename: Optional[Union[str, os.PathLike]] = None, *args: Any
    ) -> "binfile_v2_common":
        obj, _, _ = self._set_file_to_update_v2(filename)
        return obj

    def _set_file_to_update_v2(
        self, filename: Optional[Union[str, os.PathLike]] = None
    ) -> Tuple["binfile_v2_common", bool, Any]:
        if filename is None:
            filename = self.full_filename
        obj, file_exist, old_ldr = self._set_file_to_update_impl(filename)
        if not isinstance(obj, binfile_v2_common):
            return obj, file_exist, old_ldr  # type: ignore[return-value]
        if type(obj) is type(old_ldr):
            if file_exist:
                obj.upgrade_headers_ = False
            else:
                obj.upgrade_headers_ = True
                return obj, file_exist, old_ldr
        obj._define_upgrade_map(file_exist, old_ldr)
        return obj, file_exist, old_ldr

    def reopen_to_write(
        self, filename: Optional[Union[str, os.PathLike, int]] = None
    ) -> "binfile_v2_common":
        obj, permissions = self._reopen_to_write_impl(filename)
        if permissions == "rb+":
            self.upgrade_headers_ = False
        else:
            self.upgrade_headers_ = True
        return self

    def _set_upgrade_mode(self, mode: bool) -> None:
        if mode:
            pos_info = self.get_pos_info()
            self.upgrade_map_ = const_blocks_map(pos_info)
        else:
            self.upgrade_map_ = None

    def _define_upgrade_map(self, file_exist: bool, old_ldr: Any) -> None:
        if file_exist and old_ldr is not None and hasattr(old_ldr, "get_pos_info"):
            pos_info = old_ldr.get_pos_info()
            self.upgrade_map_ = const_blocks_map(pos_info)
        else:
            self.upgrade_map_ = None

    # ------------------------------------------------------------------
    # Format Helpers
    # ------------------------------------------------------------------

    def get_dnd_form(self, *args: Any) -> Dict[str, Any]:
        argi = list(args)
        if str(self.data_type).startswith("un"):
            argi = []
        return _process_format_fields(*argi)

    @staticmethod
    def extract_field_range(
        pos_fields: Dict[str, Any], form_fields: Dict[str, Any]
    ) -> Tuple[str, str, bool]:
        """Extract first and last position field names in `pos_fields` corresponding to `form_fields`."""
        pos_keys = list(pos_fields.keys())
        form_keys = list(form_fields.keys())
        fn_start = f"{form_keys[0]}_pos_"
        fn_end_cand = f"{form_keys[-1]}_pos_"
        if fn_end_cand not in pos_fields:
            # Find last matching key
            matched = [f"{k}_pos_" for k in form_keys if f"{k}_pos_" in pos_fields]
            fn_end_cand = matched[-1]
        idx = pos_keys.index(fn_end_cand)
        if idx < len(pos_keys) - 1:
            is_last = False
            fn_end = pos_keys[idx + 1]
        else:
            is_last = True
            fn_end = fn_end_cand
        return fn_start, fn_end, is_last

    # ------------------------------------------------------------------
    # Reading Methods: get_data, get_se_npix, get_npix_block, get_sqw, get_dnd
    # ------------------------------------------------------------------

    def get_data(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read DND data (or header dict when `-head` or `-noclass` is given) from the binary file."""
        from pyhorace.coord_transf.line_proj import line_proj
        from pyhorace.sqw.dnd_base import DnDBase

        opts, _ = _parse_get_data_args(*args)
        header_only = opts["header"]
        verbatim = opts["verbatim"]
        noclass = opts["noclass"]

        if isinstance(self.num_dim, str):
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "get_data called on un-initialized loader",
            )

        try:
            do_fseek(self.file_id_, self.data_pos_, "bof")
        except Exception as exc:
            raise HoraceError(
                "COMBINE_SQW_PIX_JOB:io_error",
                "Can not move to the start of the main data block",
            ) from exc

        sz = self.s_pos_ - self.data_pos_
        bytes_arr, _ = do_fread(self.file_id_, sz, "*uint8")
        self.check_error_report_fail_("get_data: Can not read the main data block")

        form = self.get_dnd_form("-head")
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        data_str, _ = self.sqw_serializer_.deserialize_bytes(bytes_arr, form, 0)

        if "iax" in data_str and np.asarray(data_str["iax"]).size == 0:
            data_str["iax"] = np.zeros((1, 0), dtype=np.float64)
        if "iint" in data_str and np.asarray(data_str["iint"]).size == 0:
            data_str["iint"] = np.zeros((2, 0), dtype=np.float64)
        if "p" in data_str and not data_str["p"]:
            data_str["p"] = []

        if not verbatim:
            data_str["filename"] = self.filename
            data_str["filepath"] = self.filepath
        else:
            if "filename" not in data_str:
                data_str["filename"] = ""
            if "filepath" not in data_str:
                data_str["filepath"] = ""
        if "title" not in data_str:
            data_str["title"] = ""

        if isinstance(self.num_dim_, str):
            self.num_dim_ = len(data_str.get("p", []))
        if isinstance(self.dnd_dimensions_, str):
            self.dnd_dimensions_ = np.array(
                [np.asarray(x).size - 1 for x in data_str.get("p", [])], dtype=np.int64
            )

        if self.convert_to_double:
            data_str = _do_convert_to_double(data_str)

        if not header_only:
            data_str = self.get_se_npix(data_str, nargout=1)

        if header_only or noclass:
            data_str["img_range"] = line_axes.calc_img_db_range(data_str)
            data_str["dimensions"] = len(data_str.get("p", []))
            return pack_io_outputs((data_str, self), nargout, default=1)

        data_str["warn_on_legacy_data"] = False
        proj = line_proj.get_from_old_data(data_str)
        ax = line_axes.get_from_old_data(data_str)
        if "dnd" in type(self).__name__.lower():
            img_r = np.asarray(ax.img_range, dtype=np.float64).copy()
            if np.any(np.isinf(img_r)):
                img_r[np.isinf(img_r)] = 0.0
                ax.img_range = img_r

        if not proj.alatt_defined:
            proj.alatt = 2.0 * np.pi
        if not proj.angdeg_defined:
            proj.angdeg = 90.0

        ax.title = data_str.get("title", "")
        ax.filename = data_str.get("filename", "")
        ax.filepath = data_str.get("filepath", "")

        res = DnDBase.dnd(ax, proj, data_str["s"], data_str["e"], data_str["npix"])
        self.sqw_holder_ = res
        return pack_io_outputs((res, self), nargout, default=1)

    def _reshape_dnd_arrays_in_dict(self, data_str: Dict[str, Any]) -> Dict[str, Any]:
        dims = self.dnd_dimensions_
        if isinstance(dims, str):
            return data_str
        dims_t = tuple(int(x) for x in np.asarray(dims).ravel(order="F"))
        if len(dims_t) == 0 or (len(dims_t) == 1 and dims_t[0] == 1 and self.num_dim == 0):
            target_shape: Tuple[int, ...] = (1, 1)
        elif len(dims_t) == 1:
            target_shape = (dims_t[0], 1)
        else:
            target_shape = dims_t

        for fld in ("s", "e", "npix"):
            if fld in data_str:
                arr = np.asarray(data_str[fld])
                if arr.size == int(np.prod(target_shape)):
                    data_str[fld] = arr.reshape(target_shape, order="F")
        return data_str

    def get_se_npix(
        self, data_str: Optional[Dict[str, Any]] = None, *args: Any, nargout: Optional[int] = None
    ) -> Any:
        """Read signal (`s`), error (`e`), and `npix` arrays from the binary file."""
        if data_str is None or not isinstance(data_str, dict):
            out: Dict[str, Any] = {}
        else:
            out = dict(data_str)

        dims = self.dnd_dimensions_
        if isinstance(dims, str):
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "get_se_npix called on un-initialized loader",
            )
        dims_arr = np.asarray(dims, dtype=np.int64).ravel(order="F")
        n_elem = int(np.prod(dims_arr))
        if self.num_dim == 0:
            shape: Tuple[int, ...] = (1, 1)
        elif len(dims_arr) == 1:
            shape = (int(dims_arr[0]), 1)
        else:
            shape = tuple(int(x) for x in dims_arr)

        do_fseek(self.file_id_, self.s_pos_, "bof")
        s_raw, _ = do_fread(self.file_id_, n_elem, "float32")
        self.check_error_report_fail_("get_se_npix: Error reading signal array")

        do_fseek(self.file_id_, self.e_pos_, "bof")
        e_raw, _ = do_fread(self.file_id_, n_elem, "float32")
        self.check_error_report_fail_("get_se_npix: Error reading error array")

        if self.convert_to_double:
            s_arr = s_raw.astype(np.float64).reshape(shape, order="F")
            e_arr = e_raw.astype(np.float64).reshape(shape, order="F")
        else:
            s_arr = s_raw.astype(np.float32).reshape(shape, order="F")
            e_arr = e_raw.astype(np.float32).reshape(shape, order="F")

        out["s"] = s_arr
        out["e"] = e_arr

        if not isinstance(self.npix_pos_, str):
            do_fseek(self.file_id_, int(self.npix_pos_), "bof")
            npix_raw, _ = do_fread(self.file_id_, n_elem, "uint64")
            self.check_error_report_fail_("get_se_npix: Error reading npix array")
            if self.convert_to_double:
                out["npix"] = npix_raw.astype(np.float64).reshape(shape, order="F")
            else:
                out["npix"] = npix_raw.astype(np.uint64).reshape(shape, order="F")

        return pack_io_outputs((out, self), nargout, default=1)

    def get_npix_block(self, pos_start: int, pos_end: int) -> np.ndarray:
        """Read a contiguous range `[pos_start, pos_end]` (1-based inclusive) of `npix` bins."""
        if isinstance(self.npix_pos_, str):
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "Can not read npix block from uninitialized object or data type 'b'",
            )
        n_bins = int(pos_end - pos_start + 1)
        if n_bins <= 0:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"Invalid bin range [{pos_start}, {pos_end}]",
            )
        seek_pos = int(self.npix_pos_) + (int(pos_start) - 1) * 8
        try:
            do_fseek(self.file_id_, seek_pos, "bof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:binfile_v2_common:io_error",
                "Could not seek to start of npix block",
            ) from exc

        npix, count = do_fread(self.file_id_, n_bins, "uint64")
        if count < n_bins:
            raise HoraceError(
                "HORACE:binfile_v2_common:io_error",
                f"Could not read {n_bins} npix values (read {count})",
            )
        if self.convert_to_double:
            return npix.astype(np.float64)
        return npix

    def get_dnd(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Read DND object (`d0d`..`d4d`, or header struct when `-head`/`-his` is given) from file."""
        return self.get_data(*args, nargout=nargout)

    def get_sqw(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """For a DND file accessor, `get_sqw` delegates to `get_dnd`."""
        return self.get_dnd(*args, nargout=nargout)

    def get_dnd_metadata(self, *args: Any, nargout: Optional[int] = None) -> Any:
        from pyhorace.sqw.dnd_metadata import dnd_metadata

        ds = self.get_data("-head", nargout=1)
        meta = dnd_metadata(ds)
        return pack_io_outputs((meta, self), nargout, default=1)

    def get_pix_range(self) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float64)

    def get_img_db_range(self, *args: Any) -> np.ndarray:
        if len(args) == 0:
            ds = self.get_data("-head", nargout=1)
        else:
            ds = args[0]
        if isinstance(ds, dict) and "img_db_range" in ds:
            return np.asarray(ds["img_db_range"], dtype=np.float64)
        if hasattr(ds, "img_db_range"):
            return np.asarray(ds.img_db_range, dtype=np.float64)
        return line_axes.calc_img_db_range(ds)

    def head(self, *args: Any) -> Dict[str, Any]:
        from pyhorace.sqw.dnd_base import DnDBase

        full = False
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("f"):
                full = True
            else:
                raise HoraceError(
                    "HORACE:binfile_v2_common:invalid_argument",
                    f"Unrecognized option {a!r} in head",
                )
        if full:
            data = self.get_data("-verbatim", nargout=1)
        else:
            data = self.get_data("-verbatim", "-head", nargout=1)

        flds = [f for f in DnDBase.head_form(full) if f != "creation_date"]
        hd: Dict[str, Any] = {}
        if isinstance(data, dict):
            if "u_to_rlu" not in data and "u_to_rlu_legacy" in data:
                data["u_to_rlu"] = data["u_to_rlu_legacy"]
            if "offset" not in data and "uoffset" in data:
                data["offset"] = data["uoffset"]
            for fld in flds:
                if fld in data:
                    hd[fld] = data[fld]
                elif fld == "ulabel" and "label" in data:
                    hd[fld] = data["label"]
                elif fld == "label" and "ulabel" in data:
                    hd[fld] = data["ulabel"]
            hd["creation_date"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            if "offset" in hd:
                hd["offset"] = np.asarray(hd["offset"]).reshape(1, -1)
            if "p" in hd:
                hd["p"] = [np.asarray(px).reshape(-1, 1) for px in hd["p"]]
        else:
            for fld in flds:
                hd[fld] = getattr(data, fld)
            hd["creation_date"] = getattr(data, "creation_date", datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        return hd

    def get_instrument(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return pack_io_outputs(({}, self), nargout, default=1)

    def get_sample(self, *args: Any, nargout: Optional[int] = None) -> Any:
        return pack_io_outputs(({}, self), nargout, default=1)

    # ------------------------------------------------------------------
    # Writing Methods: put_image, put_dnd_metadata, put_dnd_data, put_dnd, put_sqw
    # ------------------------------------------------------------------

    def put_image(self, s: Any = None, e: Any = None) -> "binfile_v2_common":
        """Write only image signal `s` and error `e` arrays."""
        self.check_obj_initiated_properly()
        if s is None or e is None:
            sub_obj, _ = self.extract_correct_subobj("data")
            if sub_obj is None:
                raise HoraceError(
                    "HORACE:binfile_v2_common:invalid_argument",
                    "put_image requires signal and error arrays or initialized sqw_holder",
                )
            s = sub_obj["s"] if isinstance(sub_obj, dict) else sub_obj.s
            e = sub_obj["e"] if isinstance(sub_obj, dict) else sub_obj.e

        s_arr = np.asarray(s, dtype=np.float32).ravel(order="F")
        e_arr = np.asarray(e, dtype=np.float32).ravel(order="F")
        expected = int(np.prod(np.asarray(self.dnd_dimensions_)))
        if s_arr.size != expected or e_arr.size != expected:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"Signal/error array size ({s_arr.size}, {e_arr.size}) does not match expected dnd_dimensions size ({expected})",
            )

        do_fseek(self.file_id_, self.s_pos_, "bof")
        do_fwrite(self.file_id_, s_arr, "float32")
        self.check_error_report_fail_("put_image: Error writing signal array")

        do_fseek(self.file_id_, self.e_pos_, "bof")
        do_fwrite(self.file_id_, e_arr, "float32")
        self.check_error_report_fail_("put_image: Error writing error array")
        return self

    def _parse_update_args(self, *args: Any) -> Tuple[bool, bool, Any]:
        """Parse `-update` and `-nopix` flags and optional object argument."""
        update = False
        nopix = False
        input_obj = None
        for a in args:
            if isinstance(a, str):
                s = a.strip().lower().lstrip("-")
                if s.startswith("up"):
                    update = True
                elif s.startswith("nop"):
                    nopix = True
                else:
                    raise HoraceError(
                        "HORACE:binfile_v2_common:invalid_argument",
                        f"Unrecognized option: {a!r}",
                    )
            elif a is not None:
                input_obj = a
        return update, nopix, input_obj

    def parse_updt_directions(self, *args: Any) -> Tuple[bool, List[Any]]:
        update = False
        argi: List[Any] = []
        for a in args:
            if isinstance(a, str) and a.strip().lower().lstrip("-").startswith("up"):
                update = True
            else:
                argi.append(a)
        return update, argi

    def put_dnd_metadata(self, *args: Any) -> "binfile_v2_common":
        """Write DND metadata (header fields before `s`) to the binary file."""
        self.check_obj_initiated_properly()
        update, _, input_obj = self._parse_update_args(*args)
        if input_obj is not None:
            sub_obj, _ = self.extract_correct_subobj("data", input_obj)
        else:
            sub_obj, _ = self.extract_correct_subobj("data")

        if sub_obj is None:
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "put_dnd_metadata called without an object to save",
            )

        if update and not self.upgrade_headers_:
            form = self.get_dnd_form("-head", "-const")
            start_pos = int(self.data_fields_locations_["alatt_pos_"])
        else:
            form = self.get_dnd_form("-head")
            start_pos = int(self.data_pos_)

        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        bytes_arr = self.sqw_serializer_.serialize(sub_obj, form)

        try:
            do_fseek(self.file_id_, start_pos, "bof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:binfile_v2_common:io_error",
                "put_dnd_metadata: Error seeking to start of DND metadata",
            ) from exc

        do_fwrite(self.file_id_, bytes_arr, "uint8")
        self.check_error_report_fail_("put_dnd_metadata: Error writing DND metadata")
        return self

    def put_dnd_data(self, *args: Any) -> "binfile_v2_common":
        """Write DND image arrays (`s`, `e`, `npix`) to the binary file."""
        self.check_obj_initiated_properly()
        _, nopix, input_obj = self._parse_update_args(*args)
        if input_obj is not None:
            sub_obj, _ = self.extract_correct_subobj("data", input_obj)
        else:
            sub_obj, _ = self.extract_correct_subobj("data")

        if sub_obj is None:
            raise HoraceError(
                "HORACE:binfile_v2_common:runtime_error",
                "put_dnd_data called without an object to save",
            )

        form = self.get_dnd_form("-data")
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        bytes_arr = self.sqw_serializer_.serialize(sub_obj, form)

        try:
            do_fseek(self.file_id_, self.s_pos_, "bof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:binfile_v2_common:io_error",
                "put_dnd_data: Error seeking to s_pos_",
            ) from exc

        do_fwrite(self.file_id_, bytes_arr, "uint8")
        self.check_error_report_fail_("put_dnd_data: Error writing DND data arrays")
        return self

    def put_dnd(self, *args: Any) -> "binfile_v2_common":
        """Write complete DND record (application header + metadata + data) to file."""
        self.check_obj_initiated_properly()
        update, nopix, input_obj = self._parse_update_args(*args)
        if input_obj is not None:
            sub_obj, _ = self.extract_correct_subobj("data", input_obj)
            if (not update or self.upgrade_headers_) and (self.data_type_ != "b+" or isinstance(self.s_pos_, str)):
                self._init_from_sqw(sub_obj)
        else:
            sub_obj, _ = self.extract_correct_subobj("data")

        if not update or self.upgrade_headers_:
            if sub_obj is not None:
                self.put_app_header(sub_obj)
            else:
                self.put_app_header()

        meta_args: List[Any] = []
        if update:
            meta_args.append("-update")
        if sub_obj is not None:
            meta_args.append(sub_obj)
        self.put_dnd_metadata(*meta_args)

        data_args: List[Any] = []
        if update:
            data_args.append("-update")
        if nopix:
            data_args.append("-nopix")
        if sub_obj is not None:
            data_args.append(sub_obj)
        self.put_dnd_data(*data_args)
        self.data_in_file_ = True
        return self

    def put_sqw(self, *args: Any) -> "binfile_v2_common":
        """For a DND file accessor, `put_sqw` delegates to `put_dnd`."""
        return self.put_dnd(*args)


BinfileV2Common = binfile_v2_common

__all__ = [
    "BinfileV2Common",
    "binfile_v2_common",
]
