"""Helper class to serialize and deserialize SQW/DND data using predefined format structures.

Ported from `horace_core/sqw/file_io/@sqw_serializer`.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.io.hlp_serial import hlp_deserialize, hlp_serialize
from pyhorace.io.sqw_fields_formatters import (
    _slice_bytes,
    field_const_array_dependent,
    iVirt_field,
    sqw_field_format_interface,
)
from pyhorace.io.sqw_fopen import do_ferror, do_fread, do_fseek, do_ftell, get_stream, pack_io_outputs

_NUMPY_TO_MATLAB_TYPE = {
    "float64": ("double", np.float64, 8),
    "float32": ("single", np.float32, 4),
    "int8": ("int8", np.int8, 1),
    "uint8": ("uint8", np.uint8, 1),
    "int16": ("int16", np.int16, 2),
    "uint16": ("uint16", np.uint16, 2),
    "int32": ("int32", np.int32, 4),
    "uint32": ("uint32", np.uint32, 4),
    "int64": ("int64", np.int64, 8),
    "uint64": ("uint64", np.uint64, 8),
}


def _resolve_fixed_fmt(fmt: Any) -> Optional[Tuple[str, Any, int, int, Tuple[int, ...]]]:
    """If `fmt` is a numeric format specifier (like `np.float32([3, 1])` or `np.uint64(1)`),
    return `(matlab_type, np_dtype, elem_bytes, n_elem, shape)`.
    """
    if isinstance(fmt, (bool, np.bool_, str, bytes, dict, sqw_field_format_interface)):
        return None
    if isinstance(fmt, (np.ndarray, np.number)):
        dt_name = fmt.dtype.name
        if dt_name not in _NUMPY_TO_MATLAB_TYPE:
            return None
        m_type, np_dtype, elem_sz = _NUMPY_TO_MATLAB_TYPE[dt_name]
        arr = np.asarray(fmt)
        if arr.size > 0 and np.all(arr == 0):
            dims = tuple(int(x) for x in arr.shape) if arr.ndim > 0 else (1,)
        else:
            dims = tuple(int(x) for x in arr.ravel(order="F"))
        nel = int(np.prod(dims)) if dims else 1
        return m_type, np_dtype, elem_sz, nel, dims
    if isinstance(fmt, float):
        nel = int(fmt)
        return "double", np.float64, 8, nel, (nel,)
    if isinstance(fmt, int):
        nel = int(fmt)
        return "double", np.float64, 8, nel, (nel,)
    return None


_FIELD_ALIASES = {
    "u_to_rlu_legacy": ("u_to_rlu",),
    "u_to_rlu": ("u_to_rlu_legacy",),
    "label": ("ulabel",),
    "ulabel": ("label",),
    "uoffset": ("offset",),
    "offset": ("uoffset",),
    "img_db_range": ("img_range",),
    "img_range": ("img_db_range",),
    "filename_with_cdate": ("filename",),
    "filename": ("filename_with_cdate",),
}


def _get_field_val(struc: Any, field_n: str) -> Any:
    if isinstance(struc, dict):
        if field_n in struc:
            return struc[field_n]
        for alt in _FIELD_ALIASES.get(field_n, ()):
            if alt in struc:
                return struc[alt]
        raise KeyError(field_n)
    if hasattr(struc, field_n):
        return getattr(struc, field_n)
    for alt in _FIELD_ALIASES.get(field_n, ()):
        if hasattr(struc, alt):
            return getattr(struc, alt)
    return getattr(struc, field_n)


class sqw_serializer:
    """Serializer and byte-position calculator for Horace binary SQW/DND structures."""

    base_classes_: Tuple[str, ...] = (
        "double",
        "single",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float64",
    )
    class_sizes_: Tuple[int, ...] = (8, 4, 1, 1, 2, 2, 4, 4, 8, 8, 8)
    class_map_: Dict[str, int] = dict(zip(base_classes_, class_sizes_))

    def __init__(self) -> None:
        self._sqw_holder: Any = None
        self._n_header: int = 1
        self._input_is_struct: bool = False
        self._input_is_file: bool = False
        self._eof_pos: float = 0.0

    @property
    def input_is_file(self) -> bool:
        return self._input_is_file

    @property
    def input_is_struct(self) -> bool:
        return self._input_is_struct

    def serialize(self, struc: Any = None, format_struct: Any = None) -> np.ndarray:
        """Serialize `struc` according to `format_struct` into a 1D `uint8` numpy array."""
        if struc is None and format_struct is None:
            return self.saveobj()
        return self._serialize(struc, format_struct)

    def _serialize(self, struc: Any, format_struct: Any) -> np.ndarray:
        if isinstance(format_struct, sqw_field_format_interface):
            return np.asarray(format_struct.bytes_from_field(struc), dtype=np.uint8).ravel(order="F")

        if not isinstance(format_struct, dict):
            raise HoraceError(
                "STRUCT_SERIALIZER:invalid_argument",
                f"Expected dict or sqw_field_format_interface as format_struct, got {type(format_struct).__name__}",
            )

        chunks: List[bytes] = []
        for field_n, fmt in format_struct.items():
            if not isinstance(fmt, iVirt_field):
                val = _get_field_val(struc, field_n)
            else:
                val = None

            if isinstance(fmt, sqw_field_format_interface):
                if isinstance(fmt, iVirt_field):
                    ser_field = fmt.bytes_from_field(struc)
                else:
                    ser_field = fmt.bytes_from_field(val)
                chunks.append(np.asarray(ser_field, dtype=np.uint8).tobytes())
            elif isinstance(fmt, str):
                s_val = str(val) if val is not None else ""
                s_bytes = s_val.encode("windows-1252", errors="replace")
                chunks.append(np.array([len(s_bytes)], dtype="<u4").tobytes() + s_bytes)
            elif isinstance(val, (list, tuple)) and not isinstance(fmt, (np.ndarray, np.number, int, float)):
                for item in val:
                    chunks.append(self._serialize(item, fmt).tobytes())
            else:
                fixed = _resolve_fixed_fmt(fmt)
                if fixed is not None:
                    _, np_dtype, _, fnel, _ = fixed
                    arr = np.asarray(val, dtype=np.dtype(np_dtype).newbyteorder("<"))
                    nel = int(arr.size)
                    if nel != fnel:
                        raise HoraceError(
                            "STRUCT_SERIALIZER:invalid_argument",
                            f"format string for constant field {field_n} contains {fnel} elements but field itself has {nel} elements",
                        )
                    chunks.append(arr.ravel(order="F").tobytes())
                elif isinstance(fmt, dict) and isinstance(val, dict):
                    chunks.append(self._serialize(val, fmt).tobytes())
                else:
                    raise HoraceError(
                        "STRUCT_SERIALIZER:invalid_argument",
                        f"Unsupported type for: field {field_n}, type: {type(val).__name__}",
                    )

        return np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()

    def _calc_pos_check_input_set_defaults(
        self, input_data: Any, start_pos: Optional[int] = None
    ) -> int:
        if isinstance(input_data, dict):
            self._input_is_struct = True
            self._input_is_file = False
        elif isinstance(input_data, (bytes, bytearray)) or (
            isinstance(input_data, np.ndarray) and input_data.dtype == np.uint8
        ):
            self._input_is_file = False
            self._input_is_struct = False
        elif isinstance(input_data, (int, float, np.number)) and get_stream(int(input_data)) is not None:
            self._input_is_file = True
            self._input_is_struct = False
        else:
            # Class instance treated as structure
            self._input_is_struct = True
            self._input_is_file = False

        if start_pos is not None:
            pos = int(start_pos)
        else:
            if self._input_is_struct:
                pos = 1
            elif self._input_is_file:
                pos = 0
            else:
                pos = 1

        if self._input_is_struct:
            self._eof_pos = float("inf")
        elif self._input_is_file:
            fid = int(input_data)
            cur_fp = do_ftell(fid)
            do_fseek(fid, 0, "eof")
            self._eof_pos = float(do_ftell(fid))
            do_fseek(fid, cur_fp, "bof")
        else:
            n_bytes = len(input_data) if isinstance(input_data, (bytes, bytearray)) else int(input_data.size)
            self._eof_pos = float(n_bytes + 1)

        return pos

    def calculate_positions(
        self,
        format_struc: Any,
        input_data: Any,
        start_pos: Optional[int] = None,
        return_error: Optional[bool] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Calculate byte positions of fields in `format_struc` for `input_data`."""
        pos = self._calc_pos_check_input_set_defaults(input_data, start_pos)
        res = self._calculate_positions(format_struc, input_data, pos)
        effective_nargout = 4 if return_error else nargout
        return pack_io_outputs(res, nargout=effective_nargout, default=2)

    def _get_str_size(self, input_data: Any, pos: int) -> Tuple[int, bool]:
        if self._input_is_file:
            fid = int(input_data)
            try:
                do_fseek(fid, pos, "bof")
            except Exception:
                return 0, True
            arr, cnt = do_fread(fid, 1, "uint32")
            _, res = do_ferror(fid)
            if res != 0 or cnt < 1:
                return 0, True
            return int(arr.flat[0]), False
        else:
            raw = _slice_bytes(input_data, pos, 4)
            if len(raw) < 4:
                return 0, True
            sz = int(np.frombuffer(raw, dtype="<u4")[0])
            return sz, False

    def _calculate_positions(
        self, template_struc: Any, input_data: Any, pos: int
    ) -> Tuple[Dict[str, Any], int, bool, Any]:
        err = False
        size_str: Dict[str, Any] = {"start_pos_": pos}

        if isinstance(template_struc, sqw_field_format_interface):
            if self._input_is_struct:
                length = int(template_struc.size_of_field(input_data))
            else:
                length, template_struc, err = template_struc.field_size(input_data, pos)
                if err:
                    size_str["eof_pos_"] = pos
                    return size_str, pos, True, template_struc
            pos += length
            return size_str, pos, False, template_struc

        for field, fmt in list(template_struc.items()):
            field_name = f"{field}_pos_"
            size_str[field_name] = pos

            if isinstance(fmt, sqw_field_format_interface):
                if self._input_is_struct:
                    if isinstance(fmt, iVirt_field):
                        length = int(fmt.size_of_field(input_data))
                    else:
                        val = _get_field_val(input_data, field)
                        length = int(fmt.size_of_field(val))
                else:
                    if isinstance(fmt, field_const_array_dependent):
                        fmt.host = template_struc
                    length, fmt, err = fmt.field_size(input_data, pos)
                    if err:
                        size_str["eof_pos_"] = pos
                        return size_str, pos, True, template_struc
                    if isinstance(fmt, iVirt_field):
                        template_struc[field] = fmt
            elif isinstance(fmt, str):
                if self._input_is_struct:
                    val = _get_field_val(input_data, field)
                    sz = len(str(val)) if val is not None else 0
                else:
                    sz, err = self._get_str_size(input_data, pos)
                    if err:
                        size_str["eof_pos_"] = pos
                        return size_str, pos, True, template_struc
                length = 4 + sz
            else:
                fixed = _resolve_fixed_fmt(fmt)
                if fixed is not None:
                    _, _, elem_sz, nel, _ = fixed
                    length = elem_sz * nel
                elif isinstance(fmt, dict):
                    length = 0
                    sub_size_str, pos, err, _ = self._calculate_positions(fmt, input_data, pos)
                    size_str[field_name] = sub_size_str
                    if err:
                        size_str["eof_pos_"] = pos
                        return size_str, pos, True, template_struc
                else:
                    raise HoraceError(
                        "HORACE:sqw_serializer:runtime_error",
                        f"Unsupported type for: field {field}, type: {type(fmt).__name__}",
                    )

            pos += length
            if pos > self._eof_pos:
                pos -= length
                size_str["eof_pos_"] = pos
                err = True
                break

        return size_str, pos, err, template_struc

    def deserialize_bytes(
        self,
        input_bytes: Union[bytes, bytearray, np.ndarray],
        template_str: Any,
        pos: int = 1,
    ) -> Tuple[Any, int]:
        """Convert sequence of bytes into structure using `template_str` starting at 1-based `pos`."""
        start_pos = max(1, int(pos))
        return self._deserialize_bytes(input_bytes, copy.deepcopy(template_str), start_pos)

    def _deserialize_bytes(
        self,
        bytes_arr: Union[bytes, bytearray, np.ndarray],
        template_str: Any,
        pos: int,
    ) -> Tuple[Any, int]:
        if isinstance(template_str, sqw_field_format_interface):
            targ_str, length = template_str.field_from_bytes(bytes_arr, pos)
            return targ_str, pos + int(length)

        targ_str: Dict[str, Any] = {}
        for field, fmt in list(template_str.items()):
            if isinstance(fmt, sqw_field_format_interface):
                if isinstance(fmt, field_const_array_dependent):
                    fmt.host = template_str
                res, length = fmt.field_from_bytes(bytes_arr, pos)
                pos += int(length)
                if isinstance(fmt, iVirt_field):
                    fmt.field_value = res
                    template_str[field] = fmt
                    continue
            elif isinstance(fmt, str):
                raw_len = _slice_bytes(bytes_arr, pos, 4)
                length = int(np.frombuffer(raw_len, dtype="<i4")[0])
                pos += 4
                if length <= 0:
                    res = ""
                else:
                    res = _slice_bytes(bytes_arr, pos, length).decode("windows-1252", errors="replace")
                pos += max(0, length)
            else:
                fixed = _resolve_fixed_fmt(fmt)
                if fixed is not None:
                    _, np_dtype, elem_sz, nel, dims = fixed
                    length = elem_sz * nel
                    raw = _slice_bytes(bytes_arr, pos, length)
                    dtype = np.dtype(np_dtype).newbyteorder("<")
                    arr = np.frombuffer(raw, dtype=dtype).copy()
                    if nel > 1:
                        if len(dims) > 1:
                            res = arr.reshape(dims, order="F")
                        else:
                            res = arr
                    else:
                        res = arr[0] if arr.size == 1 else arr
                    pos += length
                elif isinstance(fmt, dict):
                    res, pos = self._deserialize_bytes(bytes_arr, fmt, pos)
                else:
                    raise HoraceError(
                        "STRUCT_SERIZLIZER:unsupported_data_type",
                        f"Unsupported type for: field {field}, type: {type(fmt).__name__}",
                    )
            targ_str[field] = res

        return targ_str, pos

    def saveobj(self) -> np.ndarray:
        return hlp_serialize("sqw_serializer")

    @staticmethod
    def loadobj(ls: Any) -> "sqw_serializer":
        ser_struc, _ = hlp_deserialize(ls)
        if ser_struc == "sqw_serializer":
            return sqw_serializer()
        raise HoraceError(
            "HORACE:sqw_serializer:runtime_error",
            "Attempt to recover sqw serializer from incorrect data",
        )

