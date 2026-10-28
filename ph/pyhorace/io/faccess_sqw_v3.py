"""Accessor for Horace SQW binary format version 3.1.

Ported from:
- horace_core/sqw/file_io/@faccess_sqw_v3/faccess_sqw_v3.m
- horace_core/sqw/file_io/@faccess_sqw_v3/*.m
- horace_core/sqw/file_io/@faccess_sqw_v3/private/*.m
"""

from __future__ import annotations

import copy
import os
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import Serializable
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes import IX_inst, IX_null_inst, IX_null_sample as IX_null_samp, IX_samp
from pyhorace.io.horace_binfile_interface import horace_binfile_interface
from pyhorace.io.sqw_binfile_common import sqw_binfile_common
from pyhorace.io.sqw_fields_formatters import field_generic_class_hv3
from pyhorace.io.sqw_fopen import do_ferror, do_fread, do_fseek, do_ftell, do_fwrite, obj_init, pack_io_outputs, sqw_fclose


def _values_equal(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    return bool(a == b)


class faccess_sqw_v3(sqw_binfile_common):
    """Class to access Horace binary SQW files in format v3.1 (and base for v3.2/v3.21/v3.3).

    Format v3 stores instrument and sample blocks after the pixel block, followed by a
    serialized footer structure of all block byte-positions and a 4-byte uint32 footer size
    at the end of the file.

    Replaces MATLAB's `file_closer_` handle-simulation with standard Python destructor
    (`__del__`), `delete()`, and `fclose()`.
    """

    file_version_: ClassVar[float] = 3.1

    _v3_fields_to_save: ClassVar[List[str]] = [
        "instrument_head_pos_",
        "instrument_pos_",
        "sample_head_pos_",
        "sample_pos_",
        "instr_sample_end_pos_",
        "position_info_pos_",
        "eof_pos_",
    ]
    fields_to_save_: ClassVar[List[str]] = sqw_binfile_common.fields_to_save_ + _v3_fields_to_save

    def _init_default_attributes(self) -> None:
        super()._init_default_attributes()
        self.version_ = self.file_version_
        self.instrument_head_pos_: int = 0
        self.instrument_pos_: int = 0
        self.sample_head_pos_: int = 0
        self.sample_pos_: int = 0
        self.instr_sample_end_pos_: int = 0
        self.position_info_pos_: int = 0
        self.eof_pos_: int = 0
        self.file_ver_: Optional[float] = None
        self.use_faccess_ver_for_header_: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def instr_head_pos(self) -> int:
        return int(self.instrument_head_pos_)

    @property
    def instrument_head_pos(self) -> int:
        return int(self.instrument_head_pos_)

    @property
    def instrument_pos(self) -> int:
        return int(self.instrument_pos_)

    @property
    def sample_head_pos(self) -> int:
        return int(self.sample_head_pos_)

    @property
    def sample_pos(self) -> int:
        return int(self.sample_pos_)

    @property
    def instr_sample_end_pos(self) -> int:
        return int(self.instr_sample_end_pos_)

    @property
    def position_info_pos(self) -> int:
        return int(self.position_info_pos_)

    def get_faccess_version(self) -> float:
        return float(self.file_version_)

    def delete(self) -> "faccess_sqw_v3":
        super().delete()
        self.instrument_head_pos_ = 0
        self.instrument_pos_ = 0
        self.sample_head_pos_ = 0
        self.sample_pos_ = 0
        self.instr_sample_end_pos_ = 0
        self.position_info_pos_ = 0
        self.eof_pos_ = 0
        return self

    def from_bare_struct(self, indata: Dict[str, Any]) -> "faccess_sqw_v3":
        super().from_bare_struct(indata)
        for fld in self._v3_fields_to_save:
            val = getattr(self, fld, None)
            if isinstance(val, (float, np.floating, np.integer)) and not isinstance(val, (bool, np.bool_)):
                setattr(self, fld, int(val))
            elif isinstance(val, np.ndarray) and val.size == 1 and val.dtype.kind in ("i", "u", "f"):
                setattr(self, fld, int(val.flat[0]))
        return self

    # ------------------------------------------------------------------
    # Format Helpers for Instrument/Sample and Footer
    # ------------------------------------------------------------------

    @staticmethod
    def get_si_head_form(obj_name: str) -> Dict[str, Any]:
        return {
            "obj_name": str(obj_name),
            "version": np.int32(1),
            "nfiles": np.int32(1),
            "all_same": np.uint8(1),
        }

    @staticmethod
    def get_si_form(*args: Any) -> field_generic_class_hv3:
        return field_generic_class_hv3()

    @staticmethod
    def _build_block_descriptor(data: List[Any], block_type: str) -> Dict[str, Any]:
        nfiles = len(data)
        all_same = True
        if nfiles > 1:
            first = data[0]
            all_same = all(_values_equal(first, item) for item in data[1:])
        descr = faccess_sqw_v3.get_si_head_form(block_type)
        descr["nfiles"] = np.int32(nfiles)
        descr["all_same"] = np.uint8(1 if all_same else 0)
        return descr

    def _get_instr_sample_to_save(
        self, exp_info: Any
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if isinstance(exp_info, Experiment):
            instr_objs = [exp_info.instruments[i] for i in range(exp_info.instruments.n_runs)] or [IX_null_inst()]
            sampl_objs = [exp_info.samples[i] for i in range(exp_info.samples.n_runs)] or [IX_null_samp()]
            instr = [x.to_struct() if hasattr(x, "to_struct") else (x if isinstance(x, dict) else {}) for x in instr_objs]
            sampl = [x.to_struct() if hasattr(x, "to_struct") else (x if isinstance(x, dict) else {}) for x in sampl_objs]
            return instr, sampl
        if isinstance(exp_info, dict):
            instr_raw = exp_info.get("instruments", [])
            sampl_raw = exp_info.get("samples", [])
            if isinstance(instr_raw, unique_objects_container) or hasattr(instr_raw, "n_runs"):
                instr_raw = [instr_raw[i] for i in range(instr_raw.n_runs)]
            if isinstance(sampl_raw, unique_objects_container) or hasattr(sampl_raw, "n_runs"):
                sampl_raw = [sampl_raw[i] for i in range(sampl_raw.n_runs)]
            instr = [x.to_struct() if hasattr(x, "to_struct") else (x if isinstance(x, dict) else {}) for x in (instr_raw if isinstance(instr_raw, (list, tuple)) else [instr_raw])]
            sampl = [x.to_struct() if hasattr(x, "to_struct") else (x if isinstance(x, dict) else {}) for x in (sampl_raw if isinstance(sampl_raw, (list, tuple)) else [sampl_raw])]
            return instr, sampl
        if isinstance(exp_info, IX_inst):
            return [exp_info.to_struct()], [{}]
        if isinstance(exp_info, IX_samp):
            return [{}], [exp_info.to_struct()]
        if isinstance(exp_info, unique_objects_container) or hasattr(exp_info, "n_runs"):
            objs = [exp_info[i].to_struct() if hasattr(exp_info[i], "to_struct") else (exp_info[i] if isinstance(exp_info[i], dict) else {}) for i in range(exp_info.n_runs)]
            if getattr(exp_info, "baseclass", "") == "IX_inst":
                return objs, [{}]
            return [{}], objs
        if isinstance(exp_info, (list, tuple)):
            objs = [x.to_struct() if hasattr(x, "to_struct") else (x if isinstance(x, dict) else {}) for x in exp_info]
            if len(exp_info) > 0 and isinstance(exp_info[0], IX_samp):
                return [{}], objs
            return objs, [{}]
        return [{}], [{}]

    def _data_block_size(self, data: List[Any], block_type: str, pos: int) -> int:
        setattr(self, f"{block_type}_head_pos_", int(pos))
        if not data:
            setattr(self, f"{block_type}_pos_", int(pos))
            return int(pos)
        data_block = self._build_block_descriptor(data, block_type)
        head_form = self.get_si_head_form(block_type)
        _, pos = self.sqw_serializer_.calculate_positions(head_form, data_block, pos)
        setattr(self, f"{block_type}_pos_", int(pos))
        data_form = self.get_si_form(block_type)
        save_block = data[0] if bool(data_block["all_same"]) else data
        _, pos = self.sqw_serializer_.calculate_positions(data_form, save_block, pos)
        return int(pos)

    def _serialize_si_block(self, data: List[Any], block_type: str) -> np.ndarray:
        if not data:
            return np.array([], dtype=np.uint8)
        data_block = self._build_block_descriptor(data, block_type)
        data_block["version"] = np.int32(3)
        head_form = self.get_si_head_form(block_type)
        head_bytes = self.sqw_serializer_.serialize(data_block, head_form)
        data_form = self.get_si_form(block_type)
        save_block = data[0] if bool(data_block["all_same"]) else data
        block_bytes = self.sqw_serializer_.serialize(save_block, data_form)
        return np.concatenate([head_bytes, block_bytes])

    def _prepare_pos_info_for_hv3(self) -> Dict[str, Any]:
        pos_info = self.get_pos_info()
        # Normalize fields so field_generic_class_hv3 writes MATLAB-compatible types
        hp = pos_info.get("header_pos_")
        if isinstance(hp, (list, tuple)):
            if len(hp) == 1:
                pos_info["header_pos_"] = float(hp[0])
            else:
                pos_info["header_pos_"] = np.asarray(hp, dtype=np.float64).reshape((1, len(hp)))
        hpi = pos_info.get("header_pos_info_")
        if isinstance(hpi, (list, tuple)):
            if len(hpi) == 1:
                pos_info["header_pos_info_"] = hpi[0]
            elif len(hpi) > 1:
                arr = np.empty((1, len(hpi)), dtype=object)
                for idx, item in enumerate(hpi):
                    arr[0, idx] = item
                pos_info["header_pos_info_"] = arr
            else:
                pos_info["header_pos_info_"] = np.empty((0, 0), dtype=np.float64)
        dnd_dims = pos_info.get("dnd_dimensions_")
        if isinstance(dnd_dims, (list, tuple)):
            pos_info["dnd_dimensions_"] = np.asarray(dnd_dims, dtype=np.float64).reshape((1, len(dnd_dims)))
        return pos_info

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_sample_instr_records(self) -> None:
        exp_info, _ = self.extract_correct_subobj("experiment_info")
        instr, sampl = self._get_instr_sample_to_save(exp_info)
        pos = int(self.eof_pix_pos_)
        pos = self._data_block_size(instr, "instrument", pos)
        pos = self._data_block_size(sampl, "sample", pos)
        self.instr_sample_end_pos_ = int(pos)
        self.position_info_pos_ = int(pos)
        pos_info = self._prepare_pos_info_for_hv3()
        footer_form = self.get_si_form()
        _, eof = self.sqw_serializer_.calculate_positions(footer_form, pos_info, pos)
        self.eof_pos_ = int(eof)

    def init_v3_specific(self) -> None:
        self._get_sqw_footer()

    def _get_sqw_footer(self) -> None:
        try:
            do_fseek(self.file_id_, -4, "eof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:faccess_sqw_v3:io_error",
                "Cannot move to the footer size position at the end of the file",
            ) from exc
        foot_sz_arr, cnt = do_fread(self.file_id_, 1, "int32")
        mess, res = do_ferror(self.file_id_)
        if res != 0 or cnt < 1:
            raise HoraceError("HORACE:faccess_sqw_v3:io_error", mess)
        foot_sz = int(foot_sz_arr.flat[0])
        try:
            do_fseek(self.file_id_, -foot_sz - 4, "eof")
        except Exception as exc:
            raise HoraceError(
                "HORACE:faccess_sqw_v3:io_error",
                "Cannot move to the start of the footer block",
            ) from exc
        pos_info_location = do_ftell(self.file_id_)
        bytes_arr, cnt = do_fread(self.file_id_, foot_sz, "*uint8")
        mess, res = do_ferror(self.file_id_)
        if res != 0 or cnt < foot_sz:
            raise HoraceError("HORACE:faccess_sqw_v3:io_error", mess)

        data_form = self.get_si_form()
        fd_struct, _ = data_form.field_from_bytes(bytes_arr, 1)
        if not isinstance(fd_struct, dict):
            raise HoraceError(
                "HORACE:faccess_sqw_v3:io_error",
                "Corrupted v3 footer structure",
            )

        if "urange_pos_" in fd_struct:
            fd_struct["img_db_range_pos_"] = fd_struct.pop("urange_pos_")
        if "img_range_pos_" in fd_struct:
            fd_struct["img_db_range_pos_"] = fd_struct.pop("img_range_pos_")

        eof_pos = do_ftell(self.file_id_)
        self.position_info_pos_ = int(pos_info_location)
        self.eof_pos_ = int(eof_pos)
        self.real_eof_pos_ = int(eof_pos)

        saved_fn = self.filename_
        saved_fp = self.filepath_
        self.from_bare_struct(fd_struct)
        if saved_fn:
            self.filename_ = saved_fn
            self.filepath_ = saved_fp

        npix_calc = (int(self.instrument_head_pos_) - int(self.pix_pos_)) // 36
        if npix_calc != int(self.npixels):
            raise HoraceError(
                "HORACE:faccess_sqw_v3:io_error",
                "Number of pixels stored in workflow does not correspond to the shift between pixel and instrument blocks",
            )
        self.check_header_mangilig(self.header_pos_info_)

    def init_from_sqw_obj(self, *args: Any) -> "faccess_sqw_v3":
        super().init_from_sqw_obj(*args)
        self._init_sample_instr_records()
        return self

    def init_from_sqw_file(self, *args: Any) -> "faccess_sqw_v3":
        if self.file_id_ <= 0:
            raise HoraceError(
                "HORACE:faccess_sqw_v3:runtime_error",
                "File is not open when initializing v3 SQW structure",
            )
        try:
            self.init_v3_specific()
        except Exception:
            super().init_from_sqw_file(*args)
            self._init_sample_instr_records_from_file()
        return self

    def _init_sample_instr_records_from_file(self) -> None:
        pos = int(self.eof_pix_pos_)
        self.instrument_head_pos_ = pos
        try:
            do_fseek(self.file_id_, 0, "eof")
            eof_pos = do_ftell(self.file_id_)
        except Exception:
            eof_pos = pos
        if eof_pos <= pos:
            self.instrument_pos_ = pos
            self.sample_head_pos_ = pos
            self.sample_pos_ = pos
            self.instr_sample_end_pos_ = pos
            self.position_info_pos_ = pos
            self.eof_pos_ = pos
            return

        head_form = self.get_si_head_form("instrument")
        _, pos, err = self.sqw_serializer_.calculate_file_positions(head_form, self.file_id_, pos)
        if err:
            return
        self.instrument_pos_ = int(pos)
        si_form = self.get_si_form()
        _, pos, err = self.sqw_serializer_.calculate_file_positions(si_form, self.file_id_, pos)
        if err:
            return
        self.sample_head_pos_ = int(pos)

        head_form_s = self.get_si_head_form("sample")
        _, pos, err = self.sqw_serializer_.calculate_file_positions(head_form_s, self.file_id_, pos)
        if err:
            return
        self.sample_pos_ = int(pos)
        _, pos, err = self.sqw_serializer_.calculate_file_positions(si_form, self.file_id_, pos)
        if err:
            return
        self.instr_sample_end_pos_ = int(pos)
        self.position_info_pos_ = int(pos)
        _, pos, err = self.sqw_serializer_.calculate_file_positions(si_form, self.file_id_, pos)
        if not err:
            self.eof_pos_ = int(pos)

    def should_load_stream(
        self, head_struc: Dict[str, Any], fid: int, *args: Any, nargout: Optional[int] = None
    ) -> Any:
        from pyhorace.io.horace_binfile_interface import _ShouldLoadResult

        res_base = super().should_load_stream(head_struc, fid, *args, nargout=nargout)
        should, objinit, mess = res_base[0], res_base[1], res_base[2]
        if should:
            return res_base
        if not bool(head_struc.get("sqw_type", False)) or head_struc.get("version") != 3:
            return res_base

        # Check legacy version 3.0 vs 3.1
        try:
            do_fseek(fid, -4, "eof")
            foot_sz_arr, cnt = do_fread(fid, 1, "int32")
            if cnt == 1:
                foot_sz = int(foot_sz_arr.flat[0])
                if 900 < foot_sz < 100000:
                    do_fseek(fid, -foot_sz - 4, "eof")
                    bytes_arr, cnt_b = do_fread(fid, foot_sz, "*uint8")
                    if cnt_b == foot_sz:
                        data_form = self.get_si_form()
                        try:
                            res, _ = data_form.field_from_bytes(bytes_arr, 1)
                            if isinstance(res, dict) and "eof_pos_" in res:
                                return _ShouldLoadResult(True, obj_init(fid, head_struc["num_dim"]), "")
                        except Exception:
                            pass
        except Exception:
            pass
        return _ShouldLoadResult(False, obj_init(fid, head_struc["num_dim"]), "not Horace v3.1 file")

    # ------------------------------------------------------------------
    # Reading Instrument / Sample / SQW
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_legacy_instrument(res_struc: Any) -> unique_objects_container:
        res = unique_objects_container(baseclass="IX_inst")
        items = res_struc if isinstance(res_struc, list) else [res_struc]
        for item in items:
            if not item or (isinstance(item, dict) and len(item) == 0):
                res.add(IX_null_inst())
            elif isinstance(item, IX_inst):
                res.add(item)
            elif isinstance(item, dict) and "serial_name" in item:
                res.add(Serializable.from_struct(item))
            elif isinstance(item, dict) and "fermi_chopper" in item:
                from pyhorace.data.instr_classes.ix_inst_dgfermi import IX_inst_DGfermi
                res.add(IX_inst_DGfermi(item))
            elif isinstance(item, dict) and "disk_chopper" in item:
                from pyhorace.data.instr_classes.ix_inst_dgdisk import IX_inst_DGdisk
                res.add(IX_inst_DGdisk(item))
            else:
                res.add(IX_null_inst())
        return res

    def _get_instr_or_sample(
        self, field_name: str, *args: Any
    ) -> Tuple[Any, "faccess_sqw_v3"]:
        self.check_obj_initated_properly()
        get_all = False
        rem_args = []
        for a in args:
            if isinstance(a, str) and a.lower() == "-all":
                get_all = True
            else:
                rem_args.append(a)

        if not get_all:
            if len(rem_args) > 0:
                if isinstance(rem_args[0], (int, np.integer)) and not isinstance(rem_args[0], bool):
                    n_inst = int(rem_args[0])
                else:
                    raise HoraceError(
                        "HORACE:faccess_sqw_v3:invalid_argument",
                        f"get_{field_name}: Invalid argument {rem_args[0]!r}. Must be '-all' or an integer.",
                    )
            else:
                n_inst = 1
            if n_inst < 1 or (isinstance(self.num_contrib_files, int) and n_inst > self.num_contrib_files):
                raise HoraceError(
                    "HORACE:faccess_sqw_v3:invalid_argument",
                    f" Requested the {field_name} N{n_inst} when only {self.num_contrib_files} header(s) available in file",
                )
        else:
            n_inst = 1

        head_pos_attr = f"{field_name}_head_pos_"
        data_pos_attr = f"{field_name}_pos_"
        start_pos = int(getattr(self, head_pos_attr))
        data_pos = int(getattr(self, data_pos_attr))
        end_pos = int(self.sample_head_pos_) if field_name == "instrument" else int(self.instr_sample_end_pos_)

        sz_head = data_pos - start_pos
        is_instrument = field_name == "instrument"
        base_cls = "IX_inst" if is_instrument else "IX_samp"

        if sz_head <= 0 or end_pos <= data_pos:
            empty_obj = IX_null_inst() if is_instrument else IX_null_samp()
            if get_all:
                res = unique_objects_container(baseclass=base_cls)
                res.add(empty_obj)
                if isinstance(self.num_contrib_files, int) and self.num_contrib_files > 1:
                    res.replicate_runs(self.num_contrib_files)
                return res, self
            return empty_obj, self

        do_fseek(self.file_id_, start_pos, "bof")
        bytes_head, _ = do_fread(self.file_id_, sz_head, "*uint8")
        self.check_error_report_fail_(f"get_{field_name}: Concrete instrument or sample header has not been found")
        head_form = self.get_si_head_form(field_name)
        descr, _ = self.sqw_serializer_.deserialize_bytes(bytes_head, head_form, 1)
        version = int(descr["version"])
        nfiles_descr = int(descr["nfiles"])
        all_same = bool(descr["all_same"])

        sz_body = end_pos - data_pos
        do_fseek(self.file_id_, data_pos, "bof")
        bytes_body, _ = do_fread(self.file_id_, sz_body, "*uint8")
        self.check_error_report_fail_(f"Can not read the {field_name} data block")

        data_form = self.get_si_form()
        res_struc, _ = data_form.field_from_bytes(bytes_body, 1)

        if version < 2 and is_instrument:
            res = self._convert_legacy_instrument(res_struc)
        elif version == 3:
            res = unique_objects_container(baseclass=base_cls)
            items = res_struc if isinstance(res_struc, list) else [res_struc]
            for item in items:
                if item is None or (isinstance(item, dict) and len(item) == 0):
                    obj_val = IX_null_inst() if is_instrument else IX_null_samp()
                elif isinstance(item, Serializable):
                    obj_val = item
                elif isinstance(item, dict) and "serial_name" in item:
                    obj_val = Serializable.from_struct(item)
                else:
                    obj_val = IX_null_inst() if is_instrument else IX_null_samp()
                res.add(obj_val)
        else:
            res = unique_objects_container(baseclass=base_cls)
            items = res_struc if isinstance(res_struc, list) else [res_struc]
            for item in items:
                if item is None or (isinstance(item, dict) and len(item) == 0):
                    obj_val = IX_null_inst() if is_instrument else IX_null_samp()
                elif isinstance(item, Serializable):
                    obj_val = item
                elif isinstance(item, dict) and "serial_name" in item:
                    obj_val = Serializable.from_struct(item)
                else:
                    obj_val = IX_null_inst() if is_instrument else IX_null_samp()
                res.add(obj_val)

        if all_same and nfiles_descr > 1 and res.n_runs == 1:
            res.replicate_runs(nfiles_descr)

        if get_all:
            return res, self
        idx = 0 if all_same else (n_inst - 1)
        return res[idx], self

    def get_instrument(self, *args: Any, nargout: Optional[int] = None) -> Any:
        inst, obj = self._get_instr_or_sample("instrument", *args)
        return pack_io_outputs((inst, obj), nargout, default=1)

    def get_sample(self, *args: Any, nargout: Optional[int] = None) -> Any:
        samp, obj = self._get_instr_or_sample("sample", *args)
        return pack_io_outputs((samp, obj), nargout, default=1)

    def get_exp_info(self, *args: Any, nargout: Optional[int] = None) -> Any:
        base_args = [
            a
            for a in args
            if not (isinstance(a, str) and a.lower() in ("-no_is", "-no_samp_inst"))
        ]
        exp_info, last_pos = super().get_exp_info(*base_args, nargout=2)
        get_all = any(
            isinstance(a, str) and a.lower() in ("-all", "-no_is", "-no_samp_inst")
            for a in args
        )
        no_is = any(
            isinstance(a, str) and a.lower() in ("-no_is", "-no_samp_inst")
            for a in args
        )
        n_runs = exp_info.n_runs
        if not no_is:
            try:
                if get_all or not any(
                    isinstance(a, (int, np.integer)) and not isinstance(a, bool)
                    for a in args
                ):
                    inst = self.get_instrument("-all", nargout=1)
                    main_sampl = self.get_sample("-all", nargout=1)
                else:
                    n_runs = 1
                    num = next(
                        int(a)
                        for a in args
                        if isinstance(a, (int, np.integer)) and not isinstance(a, bool)
                    )
                    inst = self.get_instrument(num, nargout=1)
                    main_sampl = self.get_sample(num, nargout=1)

                is_null_samp = isinstance(main_sampl, IX_null_samp) or (
                    isinstance(main_sampl, unique_objects_container)
                    and (main_sampl.n_runs == 0 or isinstance(main_sampl[0], IX_null_samp))
                )
                if not is_null_samp:
                    if isinstance(main_sampl, (list, tuple)) and len(main_sampl) == 1:
                        main_sampl = main_sampl[0]
                    if isinstance(main_sampl, unique_objects_container):
                        if main_sampl.n_runs == 1 and n_runs > 1:
                            main_sampl = main_sampl.replicate_runs(n_runs)
                    else:
                        ms = unique_objects_container(baseclass="IX_samp")
                        ms.add(main_sampl)
                        if n_runs > 1:
                            ms = ms.replicate_runs(n_runs)
                        main_sampl = ms

                    bas_sample = exp_info.samples
                    n_samp = main_sampl.n_runs
                    for i in range(min(n_samp, bas_sample.n_runs)):
                        s_obj = main_sampl[i]
                        b_obj = bas_sample[i]
                        if getattr(s_obj, "alatt", None) is None or np.asarray(s_obj.alatt).size == 0:
                            s_obj.alatt = b_obj.alatt
                        if getattr(s_obj, "angdeg", None) is None or np.asarray(s_obj.angdeg).size == 0:
                            s_obj.angdeg = b_obj.angdeg
                        main_sampl[i] = s_obj
                    exp_info.samples = main_sampl

                is_null_inst = isinstance(inst, IX_null_inst) or (
                    isinstance(inst, unique_objects_container)
                    and (inst.n_runs == 0 or isinstance(inst[0], IX_null_inst))
                )
                if not is_null_inst:
                    if isinstance(inst, (list, tuple)) and len(inst) == 1:
                        inst = inst[0]
                    if isinstance(inst, unique_objects_container):
                        if inst.n_runs == 1 and n_runs > 1:
                            inst = inst.replicate_runs(n_runs)
                        exp_info.instruments = inst
                    else:
                        mi = unique_objects_container(baseclass="IX_inst")
                        mi.add(inst)
                        if n_runs > 1:
                            mi = mi.replicate_runs(n_runs)
                        exp_info.instruments = mi
            except Exception:
                pass
        return pack_io_outputs((exp_info, last_pos), nargout, default=1)

    # ------------------------------------------------------------------
    # Writing Instrument / Sample / Footer / SQW
    # ------------------------------------------------------------------

    def _put_sample_instr_records(self, *args: Any) -> "faccess_sqw_v3":
        update_arg: List[Any] = []
        input_obj: List[Any] = []
        save_instr = True
        save_sampl = True
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-update":
                    update_arg = ["-update"]
                elif al.startswith("-i"):
                    save_sampl = False
                elif al.startswith("-s"):
                    save_instr = False
            else:
                input_obj.append(a)

        if len(input_obj) == 1 and isinstance(
            input_obj[0], (IX_inst, IX_samp, unique_objects_container, list, tuple)
        ):
            exp_info = input_obj[0]
            new_obj = True
        else:
            exp_info, new_obj = self.extract_correct_subobj("experiment_info", *input_obj)
        instr, sampl = self._get_instr_sample_to_save(exp_info)
        if not save_instr:
            instr = []
        if not save_sampl:
            sampl = []

        if (new_obj or len(update_arg) > 0) and not self.upgrade_mode:
            pos = int(self.instrument_head_pos_)
            if save_instr and save_sampl:
                pos = self._data_block_size(instr, "instrument", pos)
                pos = self._data_block_size(sampl, "sample", pos)
                self.instr_sample_end_pos_ = int(pos)
                self.position_info_pos_ = int(pos)
            elif save_instr:
                pos = self._data_block_size(instr, "instrument", pos)
                if pos != int(self.sample_head_pos_):
                    # Shift sample block
                    do_fseek(self.file_id_, int(self.sample_head_pos_), "bof")
                    samp_sz = int(self.instr_sample_end_pos_) - int(self.sample_head_pos_)
                    samp_bytes, _ = do_fread(self.file_id_, samp_sz, "*uint8")
                    delta = pos - int(self.sample_head_pos_)
                    self.sample_head_pos_ = int(pos)
                    self.sample_pos_ = int(self.sample_pos_) + delta
                    self.instr_sample_end_pos_ = int(pos + samp_sz)
                    self.position_info_pos_ = int(pos + samp_sz)
                    do_fseek(self.file_id_, int(self.sample_head_pos_), "bof")
                    do_fwrite(self.file_id_, samp_bytes, "uint8")
            elif save_sampl:
                pos = int(self.sample_head_pos_)
                pos = self._data_block_size(sampl, "sample", pos)
                self.instr_sample_end_pos_ = int(pos)
                self.position_info_pos_ = int(pos)

        if save_instr:
            bytes_instr = self._serialize_si_block(instr, "instrument")
            do_fseek(self.file_id_, int(self.instrument_head_pos_), "bof")
            self.check_error_report_fail_("Error moving to the beginning of the instrument record")
            do_fwrite(self.file_id_, bytes_instr, "uint8")
            self.check_error_report_fail_("Error writing the instrument record")

        if save_sampl:
            bytes_sampl = self._serialize_si_block(sampl, "sample")
            do_fseek(self.file_id_, int(self.sample_head_pos_), "bof")
            self.check_error_report_fail_("Error moving to the beginning of the sample record")
            do_fwrite(self.file_id_, bytes_sampl, "uint8")
            self.check_error_report_fail_("Error writing the sample record")

        return self

    def _put_sqw_footer(self) -> "faccess_sqw_v3":
        pos_info = self._prepare_pos_info_for_hv3()
        footer_form = self.get_si_form()
        bytes_footer = self.sqw_serializer_.serialize(pos_info, footer_form)
        sz = len(bytes_footer)
        do_fseek(self.file_id_, int(self.position_info_pos_), "bof")
        self.check_error_report_fail_("Unable to move to the start of the side_block descriptor record")
        do_fwrite(self.file_id_, bytes_footer, "uint8")
        self.check_error_report_fail_("Unable to write side_block descriptor record")
        do_fwrite(self.file_id_, np.array([sz], dtype="<u4"), "uint32")
        self.check_error_report_fail_("Unable to write size of side_block descriptor")
        self.real_eof_pos_ = do_ftell(self.file_id_)
        return self

    def put_instruments(self, *args: Any) -> "faccess_sqw_v3":
        self.check_obj_initated_properly()
        self._put_sample_instr_records("-instrument", "-update", *args)
        self._put_sqw_footer()
        return self

    def put_samples(self, *args: Any) -> "faccess_sqw_v3":
        self.check_obj_initated_properly()
        self._put_sample_instr_records("-sample", "-update", *args)
        self._put_sqw_footer()
        return self

    def put_sqw(self, *args: Any) -> "faccess_sqw_v3":
        save_v2 = False
        argi = []
        for a in args:
            if isinstance(a, str) and a.lower() == "-v2":
                save_v2 = True
            else:
                argi.append(a)

        if save_v2:
            self.file_ver_ = 2.0
            self.use_faccess_ver_for_header_ = False
            super().put_sqw(*argi)
            self.deactivate()
            raise HoraceError(
                "FACCESS_SQW_V3:runtime_error",
                f"faccess_sqw_v3 with flag -v2 used to write sqw V2 format file {self.full_filename}. Use faccess_sqw_v2 to read this file",
            )

        super().put_sqw(*argi)
        self._put_sample_instr_records(*argi)
        self._put_sqw_footer()
        return self
