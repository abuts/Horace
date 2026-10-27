"""Field formatters for Horace binary SQW/DND serialization (v2 and v3 formats).

Ported from `horace_core/sqw/file_io/sqw_fields_formatters/`.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import SERIALIZABLE_REGISTRY, Serializable
from pyhorace.io.sqw_fopen import do_ferror, do_fread, do_fseek


def _matlab_shape(val: Any) -> Tuple[int, ...]:
    """Return MATLAB-style shape (at least 2D) of `val`."""
    if val is None:
        return (0, 0)
    if isinstance(val, str):
        return (1, len(val)) if len(val) > 0 else (0, 0)
    if isinstance(val, (dict, Serializable)):
        return (1, 1)
    if isinstance(val, (list, tuple)):
        return (1, len(val)) if len(val) > 0 else (0, 0)
    arr = np.asarray(val)
    if arr.ndim == 0:
        return (1, 1)
    if arr.ndim == 1:
        if arr.size == 0:
            return (0, 0)
        return (1, arr.shape[0])
    return tuple(arr.shape)


def _matlab_class_name(val: Any) -> str:
    """Return MATLAB class name string for a Python value."""
    if val is None:
        return "double"
    if isinstance(val, Serializable):
        return type(val).__name__
    if isinstance(val, dict):
        if "serial_name" in val and isinstance(val["serial_name"], str):
            return "struct"
        return "struct"
    if isinstance(val, (bool, np.bool_)):
        return "logical"
    if isinstance(val, str):
        return "char"
    if isinstance(val, (list, tuple)):
        return "cell"
    if callable(val):
        return "function_handle"
    if isinstance(val, np.ndarray):
        dt = val.dtype
        if dt == np.bool_:
            return "logical"
        if dt.kind in ("U", "S"):
            return "char"
        if dt == object:
            if val.size > 0 and isinstance(val.flat[0], Serializable):
                return type(val.flat[0]).__name__
            if val.size > 0 and isinstance(val.flat[0], dict):
                return "struct"
            return "cell"
        name_map = {
            "float64": "double",
            "float32": "single",
            "int8": "int8",
            "uint8": "uint8",
            "int16": "int16",
            "uint16": "uint16",
            "int32": "int32",
            "uint32": "uint32",
            "int64": "int64",
            "uint64": "uint64",
        }
        return name_map.get(dt.name, "double")
    if isinstance(val, (np.floating, float)):
        if isinstance(val, np.float32):
            return "single"
        return "double"
    if isinstance(val, (np.integer, int)):
        if isinstance(val, np.int8):
            return "int8"
        if isinstance(val, np.uint8):
            return "uint8"
        if isinstance(val, np.int16):
            return "int16"
        if isinstance(val, np.uint16):
            return "uint16"
        if isinstance(val, np.int32):
            return "int32"
        if isinstance(val, np.uint32):
            return "uint32"
        if isinstance(val, np.int64):
            return "int64"
        if isinstance(val, np.uint64):
            return "uint64"
        return "double"
    return type(val).__name__


_PREC_TO_DTYPE = {
    "double": np.float64,
    "float64": np.float64,
    "single": np.float32,
    "float32": np.float32,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}


def _slice_bytes(bytes_arr: Union[bytes, bytearray, np.ndarray], pos1: int, length: int) -> bytes:
    """Slice `length` bytes starting at 1-based position `pos1`."""
    idx0 = int(pos1) - 1
    if isinstance(bytes_arr, np.ndarray):
        return bytes_arr.ravel(order="F")[idx0 : idx0 + int(length)].astype(np.uint8).tobytes()
    return bytes(bytes_arr[idx0 : idx0 + int(length)])


class sqw_field_format_interface(abc.ABC):
    """Abstract base class for Horace binary field formatters."""

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
    )
    class_sizes_: Tuple[int, ...] = (8, 4, 1, 1, 2, 2, 4, 4, 8, 8)
    class_map_: Dict[str, int] = dict(zip(base_classes_, class_sizes_))

    @abc.abstractmethod
    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[Any, int]:
        """Convert sequence of bytes starting at 1-based `pos` into `(val, size)`."""

    @abc.abstractmethod
    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "sqw_field_format_interface"]:
        """Identify byte size of field starting at 1-based `pos` in `bytes_arr`."""

    @abc.abstractmethod
    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "sqw_field_format_interface", bool]:
        """Identify byte size of field starting at 0-based `pos` in open binary file `fid`."""

    @abc.abstractmethod
    def size_of_field(self, val: Any) -> int:
        """Estimate byte size of `val`."""

    @abc.abstractmethod
    def bytes_from_field(self, val: Any) -> np.ndarray:
        """Convert `val` into 1D `uint8` numpy array."""

    def field_size(
        self, input_data: Union[bytes, bytearray, np.ndarray, int], pos: int
    ) -> Tuple[int, "sqw_field_format_interface", bool]:
        """Retrieve field size from either a byte array (1-based `pos`) or file `fid` (0-based `pos`)."""
        if isinstance(input_data, (bytes, bytearray)) or (
            isinstance(input_data, np.ndarray) and input_data.dtype == np.uint8
        ):
            sz, obj = self.size_from_bytes(input_data, pos)
            return int(sz), obj, False
        sz, obj, err = self.size_from_file(int(input_data), pos)
        return int(sz), obj, bool(err)


class iVirt_field(sqw_field_format_interface):
    """Interface for virtual format fields (stored in file or computed, not in target struct)."""

    def __init__(self) -> None:
        self.field_value: Any = None


class field_var_array(sqw_field_format_interface):
    """Variable-length numeric array preceded by `n_dims` int32 dimension sizes."""

    def __init__(self, n_dims: int = 1, precision: str = "single") -> None:
        if isinstance(n_dims, (bool, np.bool_)) or not isinstance(n_dims, (int, float, np.number)):
            raise HoraceError(
                "FIELD_VAR_ARRAY:invalid_argument",
                "first argument of field_var_array has to contain number of array dimensions",
            )
        self._n_dims: int = int(n_dims)
        if precision not in self.class_map_:
            raise HoraceError(
                "FIELD_VAR_ARRAY:invalid_argument",
                f"second input argument for field_var_array constructor can only belong to basic types.\n Got: {precision}",
            )
        self._precision: str = precision
        self._n_prec: int = self.class_map_[precision]

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def elem_byte_size(self) -> int:
        return self._n_prec

    def bytes_from_field(self, val: Any) -> np.ndarray:
        arr = np.asarray(val)
        nel = int(arr.size)
        if self._n_dims == 1:
            a_size = np.array([nel], dtype="<i4")
        else:
            sh = _matlab_shape(arr)
            if len(sh) < self._n_dims:
                sh = sh + (1,) * (self._n_dims - len(sh))
            elif len(sh) > self._n_dims:
                sh = sh[: self._n_dims]
            a_size = np.array(sh, dtype="<i4")
        dtype = np.dtype(_PREC_TO_DTYPE[self._precision]).newbyteorder("<")
        data = arr.astype(dtype).ravel(order="F")
        raw = a_size.tobytes() + data.tobytes()
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        arr = np.asarray(val)
        nel = int(arr.size)
        if self._n_dims == 1:
            n_sz = 1
        else:
            n_sz = max(len(_matlab_shape(arr)), self._n_dims)
        return int(n_sz * 4 + nel * self._n_prec)

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        start = self._n_dims * 4
        sz_raw = _slice_bytes(bytes_arr, pos, start)
        sz = np.frombuffer(sz_raw, dtype="<i4")
        n_elem = int(np.prod(sz.astype(np.int64))) if sz.size > 0 else 0
        data_len = n_elem * self._n_prec
        if data_len <= 0:
            val = np.array([], dtype=_PREC_TO_DTYPE[self._precision])
            return val, start
        data_raw = _slice_bytes(bytes_arr, pos + start, data_len)
        dtype = np.dtype(_PREC_TO_DTYPE[self._precision]).newbyteorder("<")
        val = np.frombuffer(data_raw, dtype=dtype).copy()
        if sz.size > 1:
            val = val.reshape(tuple(int(x) for x in sz), order="F")
        return val, start + data_len

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_var_array"]:
        size_length = self._n_dims * 4
        raw = _slice_bytes(bytes_arr, pos, size_length)
        if len(raw) < size_length:
            return size_length, self
        dims = np.frombuffer(raw, dtype="<u4").astype(np.int64)
        nelem = int(np.prod(dims))
        sz = int(size_length + nelem * self._n_prec)
        return sz, self

    def size_from_file(self, fid: int, pos: int) -> Tuple[int, "field_var_array", bool]:
        try:
            do_fseek(fid, pos, "bof")
        except Exception:
            return 0, self, True
        dims, cnt = do_fread(fid, self._n_dims, "uint32")
        _, res = do_ferror(fid)
        if res != 0 or cnt < self._n_dims:
            return 0, self, True
        nelem = int(np.prod(dims.astype(np.int64)))
        sz = int(nelem * self._n_prec + self._n_dims * 4)
        return sz, self, False


class field_const_array_dependent(field_var_array):
    """Array whose length/shape is determined by another field in the host format struct."""

    def __init__(self, host_field: str, n_dims: int = 1, precision: str = "single") -> None:
        super().__init__(n_dims, precision)
        self._host_struct: Optional[Dict[str, Any]] = None
        self._host_struct_field: str = host_field

    @property
    def host(self) -> Optional[Dict[str, Any]]:
        return self._host_struct

    @host.setter
    def host(self, struct: Dict[str, Any]) -> None:
        self._host_struct = struct

    def process_host_value(self) -> Any:
        if self._host_struct is None:
            return 0
        fmt = self._host_struct[self._host_struct_field]
        val = getattr(fmt, "field_value", 0)
        if isinstance(val, np.ndarray) and val.size == 1:
            return int(val.flat[0])
        return val

    def bytes_from_field(self, val: Any) -> np.ndarray:
        arr = np.asarray(val)
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        raw = arr.astype(dtype).ravel(order="F").tobytes()
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        arr = np.asarray(val)
        return int(arr.size * self.elem_byte_size)

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        nelem = int(self.process_host_value())
        sz = int(nelem * self.elem_byte_size)
        if sz <= 0:
            return np.array([], dtype=_PREC_TO_DTYPE[self.precision]), 0
        raw = _slice_bytes(bytes_arr, pos, sz)
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        val = np.frombuffer(raw, dtype=dtype).copy()
        return val, sz

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_const_array_dependent"]:
        nelem = int(self.process_host_value())
        sz = int(nelem * self.elem_byte_size)
        return sz, self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_const_array_dependent", bool]:
        sz, _ = self.size_from_bytes(b"", pos)
        return sz, self, False


class field_not_in_structure(iVirt_field):
    """Virtual uint32 length field stored in binary file but omitted from the target struct."""

    def __init__(self, root_field_name: str = "") -> None:
        super().__init__()
        self._precision: str = "uint32"
        self._root_field_name: str = root_field_name
        self._n_prec: int = 4

    @property
    def precision(self) -> str:
        return self._precision

    def bytes_from_field(self, strct: Any) -> np.ndarray:
        if not self._root_field_name:
            data = np.array([0], dtype="<u4")
        else:
            val = strct[self._root_field_name] if isinstance(strct, dict) else getattr(strct, self._root_field_name)
            nel = len(val) if isinstance(val, (list, tuple)) else int(np.asarray(val).size)
            data = np.array([nel], dtype="<u4")
        return np.frombuffer(data.tobytes(), dtype=np.uint8).copy()

    def size_of_field(self, struct: Any) -> int:
        return self._n_prec

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, int]:
        raw = _slice_bytes(bytes_arr, pos, self._n_prec)
        val = int(np.frombuffer(raw, dtype="<u4")[0])
        return val, self._n_prec

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_not_in_structure"]:
        raw = _slice_bytes(bytes_arr, pos, self._n_prec)
        if len(raw) >= self._n_prec:
            self.field_value = int(np.frombuffer(raw, dtype="<u4")[0])
        return self._n_prec, self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_not_in_structure", bool]:
        try:
            do_fseek(fid, pos, "bof")
        except Exception:
            return self._n_prec, self, True
        arr, cnt = do_fread(fid, 1, self._precision)
        _, res = do_ferror(fid)
        if res != 0 or cnt < 1:
            return self._n_prec, self, True
        self.field_value = int(arr.flat[0])
        return self._n_prec, self, False


class field_p_size(field_const_array_dependent, iVirt_field):
    """Virtual helper field that inspects `p` bin-edge arrays to determine image array dimensions."""

    def __init__(self) -> None:
        iVirt_field.__init__(self)
        field_const_array_dependent.__init__(self, "npax")

    def bytes_from_field(self, struct: Any) -> np.ndarray:
        return np.array([], dtype=np.uint8)

    def size_of_field(self, struct: Any) -> int:
        return 0

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        n_cells = int(self.process_host_value())
        if n_cells == 0:
            return np.array([], dtype=np.float64), 0
        val = np.zeros(n_cells, dtype=np.float64)
        cur_pos = pos
        for i in range(n_cells):
            raw = _slice_bytes(bytes_arr, cur_pos, 4)
            ndims = int(np.frombuffer(raw, dtype="<i4")[0])
            size_i = ndims * self.elem_byte_size
            val[i] = float(ndims - 1)
            cur_pos += 4 + size_i
        return val, 0

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_p_size"]:
        self.field_value, sz = self.field_from_bytes(bytes_arr, pos)
        return sz, self

    def size_from_file(self, fid: int, pos: int) -> Tuple[int, "field_p_size", bool]:
        n_cells = int(self.process_host_value())
        if n_cells == 0:
            self.field_value = np.array([], dtype=np.float64)
            return 0, self, False
        val = np.zeros(n_cells, dtype=np.float64)
        cur_pos = pos
        for i in range(n_cells):
            try:
                do_fseek(fid, cur_pos, "bof")
            except Exception:
                return 0, self, True
            ndims_arr, cnt = do_fread(fid, 1, "int32")
            _, res = do_ferror(fid)
            if res != 0 or cnt < 1:
                return 0, self, True
            ndims = int(ndims_arr.flat[0])
            size_i = ndims * self.elem_byte_size
            val[i] = float(ndims - 1)
            cur_pos += 4 + size_i
        self.field_value = val
        return 0, self, False


class field_img(field_const_array_dependent):
    """Formatter for `s`, `e`, and `npix` image arrays sized by `p_size`."""

    def __init__(self, precision: str = "single") -> None:
        super().__init__("p_size", 1, precision)

    @property
    def dimension(self) -> np.ndarray:
        if self._host_struct is None:
            return np.array([], dtype=np.float64)
        fmt = self._host_struct[self._host_struct_field]
        dim = getattr(fmt, "field_value", None)
        if dim is None:
            return np.array([], dtype=np.float64)
        return np.asarray(dim, dtype=np.float64).ravel()

    @property
    def num_dims(self) -> int:
        return int(self.dimension.size)

    def process_host_value(self) -> np.ndarray:
        if self._host_struct is None:
            return np.array([1, 1], dtype=np.int64)
        fmt = self._host_struct[self._host_struct_field]
        dim = getattr(fmt, "field_value", None)
        if dim is None or (isinstance(dim, (list, tuple, np.ndarray)) and len(dim) == 0):
            return np.array([1, 1], dtype=np.int64)
        return np.asarray(dim, dtype=np.int64).ravel()

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        dims = self.process_host_value()
        nelem = int(np.prod(dims))
        sz = int(nelem * self.elem_byte_size)
        if sz <= 0:
            return np.array([], dtype=_PREC_TO_DTYPE[self.precision]), 0
        raw = _slice_bytes(bytes_arr, pos, sz)
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        val = np.frombuffer(raw, dtype=dtype).copy()
        if dims.size > 1:
            val = val.reshape(tuple(int(x) for x in dims), order="F")
        return val, sz

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_img"]:
        dims = self.process_host_value()
        nelem = int(np.prod(dims))
        sz = int(nelem * self.elem_byte_size)
        return sz, self

    def size_from_file(self, fid: int, pos: int) -> Tuple[int, "field_img", bool]:
        sz, _ = self.size_from_bytes(b"", pos)
        return sz, self, False


class field_iax(field_const_array_dependent):
    """Formatter for `iax` (1 x niax) integration axis index array."""

    def __init__(self) -> None:
        super().__init__("npax", 1, "int32")

    def process_host_value(self) -> int:
        if self._host_struct is None:
            return 0
        fmt = self._host_struct[self._host_struct_field]
        npax = int(getattr(fmt, "field_value", 0))
        return 4 - npax

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        val, sz = super().field_from_bytes(bytes_arr, pos)
        return val.astype(np.float64), sz


class field_iint(field_const_array_dependent):
    """Formatter for `iint` (2 x niax) integration range array."""

    def __init__(self) -> None:
        super().__init__("npax", 2, "single")

    def process_host_value(self) -> int:
        if self._host_struct is None:
            return 0
        fmt = self._host_struct[self._host_struct_field]
        npax = int(getattr(fmt, "field_value", 0))
        niax = 4 - npax
        return 2 * niax

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        nelem = int(self.process_host_value())
        sz = int(nelem * self.elem_byte_size)
        if sz <= 0:
            return np.zeros((2, 0), dtype=_PREC_TO_DTYPE[self.precision]), 0
        raw = _slice_bytes(bytes_arr, pos, sz)
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        val = np.frombuffer(raw, dtype=dtype).copy().reshape((2, nelem // 2), order="F")
        return val, sz


class field_cellarray_of_axis(field_const_array_dependent):
    """Formatter for `p` cell array of bin-edge vectors."""

    def __init__(self, host_field: str = "npax", n_dims: int = 1, precision: str = "single") -> None:
        super().__init__(host_field, n_dims, precision)

    def bytes_from_field(self, val: Any) -> np.ndarray:
        if val is None or len(val) == 0:
            return np.array([], dtype=np.uint8)
        parts: List[bytes] = []
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        for item in val:
            arr = np.asarray(item, dtype=dtype).ravel(order="F")
            parts.append(np.array([arr.size], dtype="<i4").tobytes())
            parts.append(arr.tobytes())
        return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        if val is None or len(val) == 0:
            return 0
        total_elems = sum(int(np.asarray(x).size) for x in val)
        return int(len(val) * 4 + total_elems * self.elem_byte_size)

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[List[np.ndarray], int]:
        n_cells = int(self.process_host_value())
        if n_cells == 0:
            return [], 0
        val: List[np.ndarray] = []
        cur_pos = pos
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        for _ in range(n_cells):
            n_el = int(np.frombuffer(_slice_bytes(bytes_arr, cur_pos, 4), dtype="<i4")[0])
            size_i = n_el * self.elem_byte_size
            cur_pos += 4
            arr = np.frombuffer(_slice_bytes(bytes_arr, cur_pos, size_i), dtype=dtype).copy()
            val.append(arr)
            cur_pos += size_i
        return val, int(cur_pos - pos)

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_cellarray_of_axis"]:
        n_cells = int(self.process_host_value())
        if n_cells == 0:
            return 0, self
        cur_pos = pos
        for _ in range(n_cells):
            n_el = int(np.frombuffer(_slice_bytes(bytes_arr, cur_pos, 4), dtype="<i4")[0])
            size_i = n_el * self.elem_byte_size
            cur_pos += 4 + size_i
        return int(cur_pos - pos), self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_cellarray_of_axis", bool]:
        n_cells = int(self.process_host_value())
        if n_cells == 0:
            return 0, self, False
        cur_pos = pos
        for _ in range(n_cells):
            try:
                do_fseek(fid, cur_pos, "bof")
            except Exception:
                return 0, self, True
            n_el_arr, cnt = do_fread(fid, 1, "int32")
            _, res = do_ferror(fid)
            if res != 0 or cnt < 1:
                return 0, self, True
            size_i = int(n_el_arr.flat[0]) * self.elem_byte_size
            cur_pos += 4 + size_i
        return int(cur_pos - pos), self, False


class field_cellarray_of_strings(sqw_field_format_interface):
    """Formatter for cell array of strings stored as a 2D padded character matrix."""

    defstaticmethod_char_matrix(self, val: Sequence[str]) -> np.ndarray:
        strs = [str(s) for s in val]
        if not strs:
            return np.zeros((0, 0), dtype=np.uint8, order="F")
        n_rows = len(strs)
        n_cols = max((len(s) for s in strs), default=0)
        mat = np.full((n_rows, n_cols), ord(" "), dtype=np.uint8, order="F")
        for r, s in enumerate(strs):
            b = s.encode("windows-1252", errors="replace")
            for c, ch in enumerate(b):
                mat[r, c] = ch
        return mat

    def bytes_from_field(self, val: Any) -> np.ndarray:
        if isinstance(val, str):
            val = [val]
        mat = self.defstaticmethod_char_matrix(val)
        a_size = np.array(mat.shape, dtype="<i4")
        raw = a_size.tobytes() + mat.ravel(order="F").tobytes()
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        if isinstance(val, str):
            val = [val]
        mat = self.defstaticmethod_char_matrix(val)
        return int(2 * 4 + mat.size)

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[List[str], int]:
        sz = np.frombuffer(_slice_bytes(bytes_arr, pos, 8), dtype="<u4")
        n_rows, n_cols = int(sz[0]), int(sz[1])
        length = n_rows * n_cols
        if length == 0:
            return [""], 8
        raw = _slice_bytes(bytes_arr, pos + 8, length)
        mat = np.frombuffer(raw, dtype=np.uint8).reshape((n_rows, n_cols), order="F")
        out: List[str] = []
        for r in range(n_rows):
            row_str = mat[r, :].tobytes().decode("windows-1252", errors="replace").rstrip(" ")
            out.append(row_str)
        return out, length + 8

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_cellarray_of_strings"]:
        raw = _slice_bytes(bytes_arr, pos, 8)
        if len(raw) < 8:
            return 8, self
        sz = np.frombuffer(raw, dtype="<u4")
        return int(int(sz[0]) * int(sz[1]) + 8), self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_cellarray_of_strings", bool]:
        try:
            do_fseek(fid, pos, "bof")
        except Exception:
            return 0, self, True
        sizes, cnt = do_fread(fid, 2, "int32")
        _, res = do_ferror(fid)
        if res != 0 or cnt < 2:
            return 0, self, True
        n_elements = int(sizes.flat[0]) * int(sizes.flat[1])
        return n_elements + 8, self, False


class field_simple_class_hv3(sqw_field_format_interface):
    """Self-describing HV3 formatter for primitive MATLAB types and arrays."""

    def __init__(self) -> None:
        self._precision: str = ""
        self._n_prec: Optional[int] = None
        self.sclass_map_: Dict[str, int] = dict(self.class_map_)
        self.sclass_map_["logical"] = 1
        self.sclass_map_["char"] = 1

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def elem_byte_size(self) -> Optional[int]:
        return self._n_prec

    @staticmethod
    def head_to_bytes(type_str: str, shape: Sequence[int]) -> bytes:
        t_bytes = type_str.encode("windows-1252", errors="replace")
        sh_arr = np.asarray(shape, dtype="<f8")
        return (
            np.array([float(len(t_bytes))], dtype="<f8").tobytes()
            + t_bytes
            + np.array([float(sh_arr.size)], dtype="<f8").tobytes()
            + sh_arr.tobytes()
        )

    @staticmethod
    def head_size(type_str: str, shape: Sequence[int]) -> int:
        return 8 + len(type_str) + 8 + len(shape) * 8

    @staticmethod
    def head_from_bytes(
        bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[str, Tuple[int, ...], int]:
        type_size = int(np.frombuffer(_slice_bytes(bytes_arr, pos, 8), dtype="<f8")[0])
        sz = 8
        type_str = _slice_bytes(bytes_arr, pos + sz, type_size).decode("windows-1252", errors="replace")
        sz += type_size
        shape_size = int(np.frombuffer(_slice_bytes(bytes_arr, pos + sz, 8), dtype="<f8")[0])
        sz += 8
        shape_arr = np.frombuffer(_slice_bytes(bytes_arr, pos + sz, shape_size * 8), dtype="<f8")
        sz += shape_size * 8
        shape = tuple(int(x) for x in shape_arr)
        return type_str, shape, sz

    head_sz_from_bytes = head_from_bytes

    @staticmethod
    def head_sz_from_file(fid: int, pos: int) -> Tuple[str, Tuple[int, ...], int]:
        try:
            do_fseek(fid, pos, "bof")
        except Exception as exc:
            raise HoraceError("FIELD_SIMPLE_CLASS:io_error", "Error moving to start of a class header") from exc
        cs_arr, cnt = do_fread(fid, 1, "float64")
        mess, res = do_ferror(fid)
        if res != 0 or cnt < 1:
            raise HoraceError("FIELD_SIMPLE_CLASS:io_error", f"Error reading the size of a class name: {mess}")
        class_size = int(cs_arr.flat[0])
        t_arr, cnt = do_fread(fid, (1, class_size), "*char")
        mess, res = do_ferror(fid)
        if res != 0 or cnt < class_size:
            raise HoraceError("FIELD_SIMPLE_CLASS:io_error", f"Error reading the class name: {mess}")
        type_str = t_arr.tobytes().decode("windows-1252", errors="replace")
        sz = class_size + 8
        ss_arr, cnt = do_fread(fid, 1, "float64")
        mess, res = do_ferror(fid)
        if res != 0 or cnt < 1:
            raise HoraceError("FIELD_SIMPLE_CLASS:io_error", f"Error reading the size of a shape array {mess}")
        shape_size = int(ss_arr.flat[0])
        sz += 8 + shape_size * 8
        sh_arr, cnt = do_fread(fid, (1, shape_size), "float64")
        mess, res = do_ferror(fid)
        if res != 0 or cnt < shape_size:
            raise HoraceError("FIELD_SIMPLE_CLASS:io_error", f"Error reading the class shape array {mess}")
        shape = tuple(int(x) for x in sh_arr.ravel())
        return type_str, shape, sz

    def bytes_from_field(self, val: Any) -> np.ndarray:
        type_str = _matlab_class_name(val)
        shape = _matlab_shape(val)
        head = self.head_to_bytes(type_str, shape)
        if type_str == "char":
            s_bytes = str(val).encode("windows-1252", errors="replace")
            return np.frombuffer(head + s_bytes, dtype=np.uint8).copy()
        if type_str == "logical":
            b_arr = np.asarray(val, dtype=np.uint8).ravel(order="F")
            return np.frombuffer(head + b_arr.tobytes(), dtype=np.uint8).copy()
        if val is None:
            return np.frombuffer(head, dtype=np.uint8).copy()
        dtype = np.dtype(_PREC_TO_DTYPE[type_str]).newbyteorder("<")
        arr = np.asarray(val, dtype=dtype).ravel(order="F")
        return np.frombuffer(head + arr.tobytes(), dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        type_str = _matlab_class_name(val)
        if type_str not in self.sclass_map_:
            raise HoraceError(
                "FIELD_SIMPLE_CLASS_HV3:invalid_argument",
                f"size_of_field: Class {type_str} can not be converted as simple class",
            )
        shape = _matlab_shape(val)
        nel = int(np.prod(shape))
        bl_size = self.sclass_map_[type_str]
        return self.head_size(type_str, shape) + nel * bl_size

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[Any, int]:
        type_str, shape, sz = self.head_from_bytes(bytes_arr, pos)
        if type_str not in self.sclass_map_:
            raise HoraceError(
                "FIELD_SIMPLE_CLASS_HV3:invalid_argument",
                f"field_from_bytes: Class {type_str} can not be converted as simple class",
            )
        cl_size = self.sclass_map_[type_str]
        nel = int(np.prod(shape))
        raw = _slice_bytes(bytes_arr, pos + sz, nel * cl_size)
        if type_str == "logical":
            arr = np.frombuffer(raw, dtype=np.bool_).copy()
            if nel == 1:
                return bool(arr.flat[0]), sz + nel * cl_size
            return arr.reshape(shape, order="F"), sz + nel * cl_size
        if type_str == "char":
            s = raw.decode("windows-1252", errors="replace")
            return s, sz + nel * cl_size
        dtype = np.dtype(_PREC_TO_DTYPE[type_str]).newbyteorder("<")
        arr = np.frombuffer(raw, dtype=dtype).copy()
        if len(shape) == 2 and shape == (1, 1) and nel == 1:
            return arr.flat[0], sz + nel * cl_size
        if nel > 0:
            arr = arr.reshape(shape, order="F")
        return arr, sz + nel * cl_size

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_simple_class_hv3"]:
        type_str, shape, sz = self.head_sz_from_bytes(bytes_arr, pos)
        if type_str not in self.sclass_map_:
            raise HoraceError(
                "FIELD_SIMPLE_CLASS_HV3:invalid_argument",
                f"size_from_bytes: Class {type_str} can not be converted as simple class",
            )
        elem_size = self.sclass_map_[type_str]
        sz = sz + int(np.prod(shape)) * elem_size
        self._precision = type_str
        self._n_prec = elem_size
        return sz, self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_simple_class_hv3", bool]:
        try:
            type_str, shape, sz = self.head_sz_from_file(fid, pos)
        except HoraceError as exc:
            if exc.error_id == "FIELD_SIMPLE_CLASS:io_error":
                return 0, self, True
            raise
        if type_str not in self.sclass_map_:
            raise HoraceError(
                "FIELD_SIMPLE_CLASS_HV3:invalid_argument",
                f"size_from_file: Class {type_str} can not be converted as simple class",
            )
        elem_size = self.sclass_map_[type_str]
        sz = sz + int(np.prod(shape)) * elem_size
        self._precision = type_str
        self._n_prec = elem_size
        return sz, self, False


class field_generic_class_hv3(field_simple_class_hv3):
    """Self-describing HV3 formatter for arbitrary MATLAB structs, cell arrays, and classes."""

    def _to_indep_dict(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, Serializable):
            return item.to_struct()
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "to_dict") and callable(item.to_dict):
            return item.to_dict()
        if hasattr(item, "__dict__"):
            return {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
        return {}

    def bytes_from_field(self, val: Any) -> np.ndarray:
        type_str = _matlab_class_name(val)
        if type_str in self.sclass_map_:
            return super().bytes_from_field(val)
        shape = _matlab_shape(val)
        head = self.head_to_bytes(type_str, shape)
        nel = int(np.prod(shape))
        if type_str == "cell":
            items = list(val.ravel(order="F")) if isinstance(val, np.ndarray) else list(val)
            parts = [head] + [self.bytes_from_field(x).tobytes() for x in items]
            return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()
        if type_str == "function_handle":
            f_str = getattr(val, "__name__", str(val)).encode("windows-1252", errors="replace")
            raw = head + np.array([float(len(f_str))], dtype="<f8").tobytes() + f_str
            return np.frombuffer(raw, dtype=np.uint8).copy()

        # struct or custom class
        items = list(val.ravel(order="F")) if isinstance(val, np.ndarray) else [val]
        val1 = self._to_indep_dict(items[0]) if items else {}
        names = list(val1.keys())
        nn = len(names)
        parts = [head, np.array([float(nn)], dtype="<f8").tobytes()]
        for nm in names:
            nm_b = nm.encode("windows-1252", errors="replace")
            parts.append(np.array([float(len(nm_b))], dtype="<f8").tobytes())
            parts.append(nm_b)
        for item in items:
            d = self._to_indep_dict(item)
            for nm in names:
                parts.append(self.bytes_from_field(d[nm]).tobytes())
        return np.frombuffer(b"".join(parts), dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        type_str = _matlab_class_name(val)
        if type_str in self.sclass_map_:
            return super().size_of_field(val)
        return int(self.bytes_from_field(val).size)

    def _restore_fieldnames(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int, sz: int
    ) -> Tuple[List[str], int, int]:
        ntf = int(np.frombuffer(_slice_bytes(bytes_arr, pos, 8), dtype="<f8")[0])
        pos += 8
        sz += 8
        names: List[str] = []
        for _ in range(ntf):
            nf = int(np.frombuffer(_slice_bytes(bytes_arr, pos, 8), dtype="<f8")[0])
            pos += 8
            nm = _slice_bytes(bytes_arr, pos, nf).decode("windows-1252", errors="replace")
            names.append(nm)
            sz += 8 + nf
            pos += nf
        return names, sz, pos

    def _restore_structure(
        self,
        bytes_arr: Union[bytes, bytearray, np.ndarray],
        names: List[str],
        shape: Tuple[int, ...],
        pos: int,
        sz: int,
    ) -> Tuple[Any, int]:
        nel = int(np.prod(shape))
        if not names:
            if shape == (0, 0) or nel == 0:
                return {}, sz
            if shape == (1, 1):
                return {}, sz
            return [{} for _ in range(nel)], sz
        structs: List[Dict[str, Any]] = []
        for _ in range(nel):
            d: Dict[str, Any] = {}
            for nm in names:
                val, szi = self.field_from_bytes(bytes_arr, pos)
                d[nm] = val
                sz += szi
                pos += szi
            structs.append(d)
        if nel == 1:
            return structs[0], sz
        return structs, sz

    def _instantiate_class(self, type_str: str, struct_data: Dict[str, Any]) -> Any:
        if "serial_name" in struct_data:
            try:
                return Serializable.from_struct(struct_data)
            except Exception:
                pass
        target_cls = SERIALIZABLE_REGISTRY.get(type_str)
        if target_cls is not None:
            try:
                return target_cls.loadobj(struct_data)
            except Exception:
                try:
                    inst = target_cls()
                    return inst.from_bare_struct(struct_data)
                except Exception:
                    pass
        return struct_data

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[Any, int]:
        type_str, shape, sz = self.head_from_bytes(bytes_arr, pos)
        if type_str in self.sclass_map_:
            return super().field_from_bytes(bytes_arr, pos)
        cur_pos = pos + sz
        nel = int(np.prod(shape))
        if type_str == "cell":
            items: List[Any] = []
            for _ in range(nel):
                v, szi = self.field_from_bytes(bytes_arr, cur_pos)
                items.append(v)
                cur_pos += szi
                sz += szi
            return items, sz
        if type_str == "function_handle":
            fhl = int(np.frombuffer(_slice_bytes(bytes_arr, cur_pos, 8), dtype="<f8")[0])
            f_name = _slice_bytes(bytes_arr, cur_pos + 8, fhl).decode("windows-1252", errors="replace")
            return f_name, sz + 8 + fhl

        names, sz, cur_pos = self._restore_fieldnames(bytes_arr, cur_pos, sz)
        raw_structs, sz = self._restore_structure(bytes_arr, names, shape, cur_pos, sz)
        if type_str == "struct":
            return raw_structs, sz
        if isinstance(raw_structs, dict):
            return self._instantiate_class(type_str, raw_structs), sz
        if isinstance(raw_structs, list):
            return [self._instantiate_class(type_str, s) for s in raw_structs], sz
        return raw_structs, sz

    def _size_structure_from_bytes(
        self,
        bytes_arr: Union[bytes, bytearray, np.ndarray],
        n_fields: int,
        shape: Tuple[int, ...],
        pos: int,
        sz: int,
    ) -> int:
        nel = int(np.prod(shape))
        for _ in range(nel):
            for _ in range(n_fields):
                szi, _ = self.size_from_bytes(bytes_arr, pos)
                sz += szi
                pos += szi
        return sz

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_generic_class_hv3"]:
        type_str, shape, sz = self.head_sz_from_bytes(bytes_arr, pos)
        if type_str in self.sclass_map_:
            sz, _ = super().size_from_bytes(bytes_arr, pos)
            return sz, self
        self._precision = type_str
        cur_pos = pos + sz
        nel = int(np.prod(shape))
        if type_str == "cell":
            for _ in range(nel):
                szi, _ = self.size_from_bytes(bytes_arr, cur_pos)
                cur_pos += szi
                sz += szi
            self._n_prec = sz
            return sz, self
        if type_str == "function_handle":
            fn_len = int(np.frombuffer(_slice_bytes(bytes_arr, cur_pos, 8), dtype="<f8")[0])
            sz += 8 + fn_len
            return sz, self
        names, sz, cur_pos = self._restore_fieldnames(bytes_arr, cur_pos, sz)
        sz = self._size_structure_from_bytes(bytes_arr, len(names), shape, cur_pos, sz)
        self._n_prec = sz
        return sz, self

    def size_from_file(
        self, fid: int, pos: int
    ) -> Tuple[int, "field_generic_class_hv3", bool]:
        try:
            type_str, shape, sz = self.head_sz_from_file(fid, pos)
        except HoraceError as exc:
            if exc.error_id == "FIELD_SIMPLE_CLASS:io_error":
                return 0, self, True
            raise
        if type_str in self.sclass_map_:
            elem_size = self.sclass_map_[type_str]
            sz = sz + int(np.prod(shape)) * elem_size
            self._precision = type_str
            self._n_prec = elem_size
            return sz, self, False
        self._precision = type_str
        cur_pos = pos + sz
        nel = int(np.prod(shape))
        if type_str == "cell":
            for _ in range(nel):
                szi, _, err = self.size_from_file(fid, cur_pos)
                if err:
                    return sz, self, True
                cur_pos += szi
                sz += szi
            self._n_prec = sz
            return sz, self, False
        if type_str == "function_handle":
            try:
                do_fseek(fid, cur_pos, "bof")
                arr, cnt = do_fread(fid, 1, "float64")
                if cnt < 1:
                    return sz, self, True
                fn_len = int(arr.flat[0])
                return sz + 8 + fn_len, self, False
            except Exception:
                return sz, self, True
        # Read fieldnames from file
        try:
            do_fseek(fid, cur_pos, "bof")
        except Exception:
            return sz, self, True
        ntf_arr, cnt = do_fread(fid, 1, "float64")
        if cnt < 1:
            return sz, self, True
        n_fields = int(ntf_arr.flat[0])
        cur_pos += 8
        sz += 8
        for _ in range(n_fields):
            try:
                do_fseek(fid, cur_pos, "bof")
            except Exception:
                return sz, self, True
            nf_arr, cnt = do_fread(fid, 1, "float64")
            if cnt < 1:
                return sz, self, True
            nf = int(nf_arr.flat[0])
            cur_pos += 8 + nf
            sz += 8 + nf
        for _ in range(nel):
            for _ in range(n_fields):
                szi, _, err = self.size_from_file(fid, cur_pos)
                if err:
                    return sz, self, True
                cur_pos += szi
                sz += szi
        self._n_prec = sz
        return sz, self, False


class sqw_fields_formatters:
    """Factory and namespace for SQW binary field formatters."""

    @staticmethod
    def field_var_array(*args: Any) -> field_var_array:
        return field_var_array(*args)

    @staticmethod
    def field_const_array_dependent(*args: Any) -> field_const_array_dependent:
        return field_const_array_dependent(*args)

    @staticmethod
    def field_not_in_structure(*args: Any) -> field_not_in_structure:
        return field_not_in_structure(*args)

    @staticmethod
    def field_p_size(*args: Any) -> field_p_size:
        return field_p_size(*args)

    @staticmethod
    def field_img(*args: Any) -> field_img:
        return field_img(*args)

    @staticmethod
    def field_iax() -> field_iax:
        return field_iax()

    @staticmethod
    def field_iint() -> field_iint:
        return field_iint()

    @staticmethod
    def field_cellarray_of_axis(*args: Any) -> field_cellarray_of_axis:
        return field_cellarray_of_axis(*args)

    @staticmethod
    def field_cellarray_of_strings(*args: Any) -> field_cellarray_of_strings:
        return field_cellarray_of_strings(*args)

    @staticmethod
    def field_simple_class_hv3() -> field_simple_class_hv3:
        return field_simple_class_hv3()

    @staticmethod
    def field_generic_class_hv3() -> field_generic_class_hv3:
        return field_generic_class_hv3()

    @staticmethod
    def field_pix() -> "field_pix":
        return field_pix()


class field_pix(field_var_array):
    """Formatter for v2/v3 SQW pixel arrays (`uint64` npix count followed by `9 * npix` float32 values)."""

    DEFAULT_NUM_PIX_FIELDS: int = 9

    def __init__(self) -> None:
        super().__init__(1, "single")

    def bytes_from_field(self, val: Any) -> np.ndarray:
        if hasattr(val, "data") and hasattr(val, "num_pixels"):
            arr = np.asarray(val.data, dtype="<f4")
            num_pix = int(val.num_pixels)
        else:
            arr = np.asarray(val, dtype="<f4")
            sh = _matlab_shape(arr)
            num_pix = int(sh[1]) if arr.size > 0 else 0
        head = np.array([num_pix], dtype="<u8").tobytes()
        raw = head + arr.ravel(order="F").tobytes()
        return np.frombuffer(raw, dtype=np.uint8).copy()

    def size_of_field(self, val: Any) -> int:
        if hasattr(val, "num_pixels"):
            nel = int(val.num_pixels) * int(getattr(val, "DEFAULT_NUM_PIX_FIELDS", self.DEFAULT_NUM_PIX_FIELDS))
        else:
            nel = int(np.asarray(val).size)
        return int(8 + nel * self.elem_byte_size)

    def field_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[np.ndarray, int]:
        npix = int(np.frombuffer(_slice_bytes(bytes_arr, pos, 8), dtype="<u8")[0])
        numel = npix * self.DEFAULT_NUM_PIX_FIELDS
        length = numel * self.elem_byte_size
        start = 8
        if numel == 0:
            return np.empty((0, 0), dtype=np.float32), start
        raw = _slice_bytes(bytes_arr, pos + start, length)
        dtype = np.dtype(_PREC_TO_DTYPE[self.precision]).newbyteorder("<")
        val = np.frombuffer(raw, dtype=dtype).copy().reshape((self.DEFAULT_NUM_PIX_FIELDS, npix), order="F")
        return val, start + length

    def size_from_bytes(
        self, bytes_arr: Union[bytes, bytearray, np.ndarray], pos: int
    ) -> Tuple[int, "field_pix"]:
        raw = _slice_bytes(bytes_arr, pos, 8)
        if len(raw) < 8:
            return 8, self
        npix = int(np.frombuffer(raw, dtype="<u8")[0])
        numel = npix * self.DEFAULT_NUM_PIX_FIELDS
        return int(8 + numel * self.elem_byte_size), self

    def size_from_file(self, fid: int, pos: int) -> Tuple[int, "field_pix", bool]:
        try:
            do_fseek(fid, pos, "bof")
        except Exception:
            return 0, self, True
        npix_arr, cnt = do_fread(fid, 1, "uint64")
        _, res = do_ferror(fid)
        if res != 0 or cnt < 1:
            return 0, self, True
        nelem = int(npix_arr.flat[0]) * self.DEFAULT_NUM_PIX_FIELDS
        return int(nelem * self.elem_byte_size + 8), self, False


