"""MATLAB-compatible binary serialization (`hlp_serialize` / `hlp_deserialize`).

Ported from Herbert MATLAB utilities:
- `herbert_core/utilities/misc/hlp_serial_types.m`
- `herbert_core/utilities/misc/hlp_serialize.m`
- `herbert_core/utilities/misc/hlp_deserialize.m`
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError


class hlp_serial_types:
    """Type definitions and header packing/unpacking for `hlp_serialize` / `hlp_deserialize`."""

    types: Tuple[str, ...] = (
        "logical",
        "char",
        "string",
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
        "complex_double",
        "complex_single",
        "complex_int8",
        "complex_uint8",
        "complex_int16",
        "complex_uint16",
        "complex_int32",
        "complex_uint32",
        "complex_int64",
        "complex_uint64",
        "cell",
        "struct",
        "function_handle",
        "value_object",
        "handle_object_ref",
        "enum",
        "sparse_logical",
        "sparse_double",
        "sparse_complex_double",
        "serializable",
    )

    data_size: Tuple[int, ...] = (
        1, 1, 2, 8,
        4, 1, 1, 2, 2, 4, 4,
        8, 8, 16, 8, 2,
        2, 4, 4, 8,
        8, 16, 16, 0, 0,
        0, 0, 0, 0,
        1, 8, 16,
        0,
    )

    tags_map: Dict[str, int] = {name: i for i, name in enumerate(types)}
    sizes_map: Dict[str, int] = {name: sz for name, sz in zip(types, data_size)}

    NP_DTYPE_MAP: Dict[str, np.dtype] = {
        "logical": np.dtype("uint8"),
        "char": np.dtype("uint8"),
        "string": np.dtype("uint8"),
        "double": np.dtype("<f8"),
        "single": np.dtype("<f4"),
        "int8": np.dtype("<i1"),
        "uint8": np.dtype("<u1"),
        "int16": np.dtype("<i2"),
        "uint16": np.dtype("<u2"),
        "int32": np.dtype("<i4"),
        "uint32": np.dtype("<u4"),
        "int64": np.dtype("<i8"),
        "uint64": np.dtype("<u8"),
    }

    tag_size: int = 2
    ndims_size: int = 1
    dim_size: int = 4

    @classmethod
    def get_details(cls, type_name: str) -> Dict[str, Any]:
        tag = cls.tags_map[type_name]
        return {
            "name": type_name,
            "size": cls.data_size[tag],
            "tag": tag,
        }

    @classmethod
    def get_by_tag(cls, tag: int) -> Dict[str, Any]:
        if tag < 0 or tag >= len(cls.types):
            raise HoraceError(
                "HORACE:deserialize:invalid_argument",
                f"Cannot deserialize tag with ID: {tag}.",
            )
        return {
            "name": cls.types[tag],
            "size": cls.data_size[tag],
            "tag": tag,
        }

    @classmethod
    def pack_data_tag(
        cls,
        data_size: Sequence[int],
        type_struc: Dict[str, Any],
        full_header: bool = False,
    ) -> bytes:
        tag = int(type_struc["tag"])
        n_dims = len(data_size)
        n_elem = int(np.prod(data_size)) if n_dims > 0 else 0

        if n_elem == 0:
            return struct.pack("<BB", tag, 0)
        elif n_elem == 1 and not full_header:
            return struct.pack("<BBI", tag, 1, 1)
        elif n_dims == 2 and data_size[0] == 1 and not full_header:
            return struct.pack("<BBI", tag, 1, n_elem)
        else:
            return struct.pack(f"<BB{n_dims}I", tag, n_dims, *[int(d) for d in data_size])

    @classmethod
    def calc_tag_size(
        cls,
        data_size: Sequence[int],
        type_struc: Dict[str, Any],
        full_header: bool = False,
    ) -> int:
        if type_struc["name"] == "function_handle":
            return 1
        n_dims = len(data_size)
        n_elem = int(np.prod(data_size)) if n_dims > 0 else 0
        if n_elem == 0:
            return cls.tag_size
        elif n_elem == 1 and not full_header:
            return cls.tag_size + cls.dim_size
        elif n_dims == 2 and data_size[0] == 1 and not full_header:
            return cls.tag_size + cls.dim_size
        else:
            return cls.tag_size + cls.dim_size * n_dims

    @classmethod
    def unpack_data_tag(
        cls, head_bytes: Union[bytes, bytearray, memoryview], pos: int
    ) -> Tuple[Dict[str, Any], int, Tuple[int, ...], int]:
        tag = int(head_bytes[pos])
        if tag > 63 or tag == 25:
            fh_tag = tag & 192
            base_tag = tag - fh_tag
            type_str = cls.get_by_tag(base_tag)
            return type_str, 1, (fh_tag,), pos + 1

        type_str = cls.get_by_tag(tag)
        n_dims = int(head_bytes[pos + 1])
        pos += 2
        if n_dims == 0:
            return type_str, 0, (), pos
        elif n_dims == 1:
            (n_elem,) = struct.unpack_from("<I", head_bytes, pos)
            return type_str, 1, (1, int(n_elem)), pos + 4
        else:
            dims = struct.unpack_from(f"<{n_dims}I", head_bytes, pos)
            return type_str, n_dims, tuple(int(d) for d in dims), pos + 4 * n_dims


def _matlab_shape_for_array(arr: np.ndarray) -> Tuple[int, ...]:
    if arr.size == 0:
        return (0, 0)
    if arr.ndim == 0:
        return (1, 1)
    if arr.ndim == 1:
        return (1, int(arr.shape[0]))
    return tuple(int(d) for d in arr.shape)


def _infer_type_and_array(v: Any) -> Tuple[Dict[str, Any], Any, Tuple[int, ...]]:
    from pyhorace.core.serializable import Serializable

    if isinstance(v, Serializable):
        return hlp_serial_types.get_details("serializable"), v, (1, 1)
    if v is None:
        return hlp_serial_types.get_details("double"), np.empty((0, 0), dtype="<f8"), (0, 0)
    if isinstance(v, (bool, np.bool_)):
        arr = np.array([[1 if v else 0]], dtype=np.uint8)
        return hlp_serial_types.get_details("logical"), arr, (1, 1)
    if isinstance(v, str):
        raw = v.encode("latin1", errors="replace")
        n = len(raw)
        shape = (0, 0) if n == 0 else (1, n)
        return hlp_serial_types.get_details("char"), raw, shape
    if isinstance(v, (bytes, bytearray)):
        arr = np.frombuffer(v, dtype=np.uint8)
        shape = (0, 0) if arr.size == 0 else (1, int(arr.size))
        return hlp_serial_types.get_details("uint8"), arr, shape
    if isinstance(v, dict):
        return hlp_serial_types.get_details("struct"), v, (1, 1)
    if isinstance(v, (list, tuple)):
        if len(v) > 0 and all(isinstance(x, Serializable) and type(x) is type(v[0]) for x in v):
            return hlp_serial_types.get_details("serializable"), list(v), (1, len(v))
        if len(v) > 0 and all(isinstance(x, dict) for x in v):
            keys0 = tuple(v[0].keys())
            if all(tuple(x.keys()) == keys0 for x in v):
                return hlp_serial_types.get_details("struct"), list(v), (1, len(v))
        shape = (0, 0) if len(v) == 0 else (1, len(v))
        return hlp_serial_types.get_details("cell"), list(v), shape
    if isinstance(v, int):
        arr = np.array([[float(v)]], dtype="<f8")
        return hlp_serial_types.get_details("double"), arr, (1, 1)
    if isinstance(v, float):
        arr = np.array([[v]], dtype="<f8")
        return hlp_serial_types.get_details("double"), arr, (1, 1)
    if isinstance(v, complex):
        arr = np.array([[v]], dtype="<c16")
        return hlp_serial_types.get_details("complex_double"), arr, (1, 1)
    if isinstance(v, np.generic):
        return _infer_type_and_array(np.asarray(v))
    if isinstance(v, np.ndarray):
        shape = _matlab_shape_for_array(v)
        dt = v.dtype
        if dt.kind == "b":
            return hlp_serial_types.get_details("logical"), v.astype(np.uint8), shape
        if dt.kind in ("U", "S"):
            if v.size == 0:
                return hlp_serial_types.get_details("char"), b"", (0, 0)
            flat_chars = "".join(
                x.decode("latin1") if isinstance(x, bytes) else str(x)
                for x in v.ravel(order="F")
            )
            raw = flat_chars.encode("latin1", errors="replace")
            return hlp_serial_types.get_details("char"), raw, shape
        if dt.kind == "O":
            return hlp_serial_types.get_details("cell"), v, shape
        if dt.kind == "c":
            tname = "complex_single" if dt.itemsize == 8 else "complex_double"
            return hlp_serial_types.get_details(tname), v, shape
        dtype_to_name = {
            np.dtype("float64"): "double",
            np.dtype("float32"): "single",
            np.dtype("int8"): "int8",
            np.dtype("uint8"): "uint8",
            np.dtype("int16"): "int16",
            np.dtype("uint16"): "uint16",
            np.dtype("int32"): "int32",
            np.dtype("uint32"): "uint32",
            np.dtype("int64"): "int64",
            np.dtype("uint64"): "uint64",
        }
        base_dt = np.dtype(dt.name)
        tname = dtype_to_name.get(base_dt, "double")
        return hlp_serial_types.get_details(tname), v, shape

    raise HoraceError(
        "MATLAB:serialize:invalid_argument",
        f"Cannot serialize type {type(v).__name__}.",
    )


def hlp_serialize(v: Any) -> bytes:
    """Convert a Python/MATLAB data structure into a MATLAB `hlp_serialize` byte vector."""
    type_info, data, shape = _infer_type_and_array(v)
    tname = type_info["name"]

    if tname == "serializable":
        from pyhorace.core.serializable import Serializable

        if isinstance(data, list):
            S = Serializable.to_struct_array(data)
        else:
            S = data.to_struct()
        return bytes([32]) + hlp_serialize(S)

    if tname in (
        "logical",
        "char",
        "string",
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
    ):
        header = hlp_serial_types.pack_data_tag(shape, type_info)
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 0
        if n_elem == 0:
            return header
        if isinstance(data, (bytes, bytearray)):
            return header + bytes(data)
        target_dt = hlp_serial_types.NP_DTYPE_MAP[tname]
        arr_f = np.ascontiguousarray(np.asarray(data).ravel(order="F"), dtype=target_dt)
        return header + arr_f.tobytes()

    if tname.startswith("complex_"):
        header = hlp_serial_types.pack_data_tag(shape, type_info)
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 0
        if n_elem == 0:
            return header
        base_name = tname[len("complex_"):]
        target_dt = hlp_serial_types.NP_DTYPE_MAP[base_name]
        flat = np.asarray(data).ravel(order="F")
        real_bytes = np.ascontiguousarray(flat.real, dtype=target_dt).tobytes()
        imag_bytes = np.ascontiguousarray(flat.imag, dtype=target_dt).tobytes()
        return header + real_bytes + imag_bytes

    if tname == "cell":
        header = hlp_serial_types.pack_data_tag(shape, type_info)
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 0
        if n_elem == 0:
            return header
        if isinstance(data, np.ndarray):
            items = list(data.ravel(order="F"))
        else:
            items = list(data)
        parts = [header] + [hlp_serialize(item) for item in items]
        return b"".join(parts)

    if tname == "struct":
        header = hlp_serial_types.pack_data_tag(shape, type_info)
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 0
        if n_elem == 0:
            return header
        dicts: List[Dict[str, Any]] = [data] if isinstance(data, dict) else list(data)
        field_names = list(dicts[0].keys())
        n_fields = len(field_names)
        if n_fields == 0:
            cell_bytes = hlp_serialize([])
            return header + struct.pack("<I", 0) + cell_bytes
        fn_lengths = [len(fn) for fn in field_names]
        fn_chars = "".join(field_names).encode("latin1")
        struct_meta = struct.pack(f"<I{n_fields}I", n_fields, *fn_lengths) + fn_chars
        # Field values in MATLAB struct2cell: shape (nFields, nElem) in column-major order
        # i.e. for each element, all its fields in order
        cell_header = hlp_serial_types.pack_data_tag(
            (n_fields, n_elem) if n_elem > 1 else (n_fields, 1),
            hlp_serial_types.get_details("cell"),
        )
        val_parts = [cell_header]
        for d in dicts:
            for fn in field_names:
                val_parts.append(hlp_serialize(d[fn]))
        return header + struct_meta + b"".join(val_parts)

    raise HoraceError(
        "MATLAB:serialize:invalid_argument",
        f"Cannot serialize type {tname}.",
    )


def hlp_serial_size(v: Any) -> int:
    """Calculate the exact byte length of `hlp_serialize(v)`."""
    return len(hlp_serialize(v))


def hlp_deserialize(
    m: Union[bytes, bytearray, memoryview, np.ndarray], pos: int = 0
) -> Tuple[Any, int]:
    """Convert a serialized byte vector back into the corresponding data structure.

    Parameters
    ----------
    m : bytes, bytearray, memoryview, or uint8 ndarray
        Byte stream to deserialize.
    pos : int, optional
        0-based starting byte offset (default 0).

    Returns
    -------
    v : Any
        Deserialized data structure.
    nbytes : int
        Number of bytes consumed from `m`.
    """
    if isinstance(m, np.ndarray):
        buf = memoryview(np.ascontiguousarray(m, dtype=np.uint8)).cast("B")
    elif isinstance(m, (bytes, bytearray)):
        buf = memoryview(m)
    elif isinstance(m, memoryview):
        buf = m.cast("B")
    else:
        buf = memoryview(bytes(m))

    pos0 = int(pos)
    v, pos_end = _deserialize_value(buf, pos0)
    return v, pos_end - pos0


def _read_bytes(
    buf: memoryview, pos: int, type_name: str, n: int
) -> Tuple[np.ndarray, int]:
    comp_fac = 1
    if type_name.startswith("complex_"):
        type_name = type_name[len("complex_"):]
        comp_fac = 2
    details = hlp_serial_types.get_details(type_name)
    dt = hlp_serial_types.NP_DTYPE_MAP[type_name]
    n_bytes = int(details["size"]) * int(n) * comp_fac
    if pos + n_bytes > len(buf):
        raise HoraceError(
            "HORACE:deserialize:invalid_argument",
            f"Buffer underrun reading {n_bytes} bytes at position {pos} (buffer size {len(buf)}).",
        )
    arr = np.frombuffer(buf, dtype=dt, count=int(n) * comp_fac, offset=pos).copy()
    return arr, pos + n_bytes


def _deserialize_value(buf: memoryview, pos: int) -> Tuple[Any, int]:
    if pos >= len(buf):
        raise HoraceError(
            "HORACE:deserialize:invalid_argument",
            f"Position {pos} exceeds buffer length {len(buf)}.",
        )
    type_id = int(buf[pos])

    if 0 <= type_id <= 12:
        return _deserialize_simple_data(buf, pos)
    if 13 <= type_id <= 22:
        return _deserialize_complex_data(buf, pos)
    if type_id == 23:
        return _deserialize_cell(buf, pos)
    if type_id == 24:
        return _deserialize_struct(buf, pos)
    if type_id in (25, 25 + 64, 25 + 128, 25 + 192):
        return _deserialize_function_handle(buf, pos)
    if type_id in (26, 27):
        return _deserialize_object(buf, pos)
    if type_id in (29, 30, 31):
        return _deserialize_sparse(buf, pos)
    if type_id == 32:
        return _obj_deserialize_itself(buf, pos)

    raise HoraceError(
        "HORACE:deserialize:invalid_argument",
        f"Cannot deserialize tag with ID: {type_id} at position {pos}.",
    )


def _deserialize_simple_data(buf: memoryview, pos: int) -> Tuple[Any, int]:
    type_info, n_dims, size, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    tname = type_info["name"]
    if tname in ("logical", "char", "string"):
        field_width = "uint8"
    else:
        field_width = tname

    n_data = int(np.prod(size)) if n_dims > 0 else 0
    if n_data > 0:
        raw_arr, pos = _read_bytes(buf, pos, field_width, n_data)
    else:
        raw_arr = np.empty(0, dtype=hlp_serial_types.NP_DTYPE_MAP[field_width])

    if tname in ("char", "string"):
        if n_data == 0:
            return "", pos
        s = raw_arr.tobytes().decode("latin1")
        return s, pos

    if tname == "logical":
        arr = raw_arr.astype(bool)
        if n_data == 0:
            return np.empty(size if n_dims > 1 else (0,), dtype=bool), pos
        if n_dims == 1 and n_data == 1:
            return bool(arr[0]), pos
        if n_dims > 1:
            return arr.reshape(size, order="F"), pos
        return arr, pos

    # Numeric types (double, single, int*, uint*)
    if n_data == 0:
        return np.empty(size if n_dims > 1 else (0,), dtype=raw_arr.dtype), pos
    if n_dims == 1 and n_data == 1:
        if tname in ("double", "single"):
            val = float(raw_arr[0])
            if val.is_integer() and abs(val) < (1 << 53):
                # Keep float but int-compatible if needed; Python float compares equal to int (1.0 == 1)
                return val, pos
            return val, pos
        return int(raw_arr[0]), pos
    if n_dims > 1:
        if n_dims == 2 and (size[0] == 1 or size[1] == 1):
            return raw_arr.reshape((n_data,)), pos
        return raw_arr.reshape(size, order="F"), pos
    return raw_arr, pos


def _deserialize_complex_data(buf: memoryview, pos: int) -> Tuple[Any, int]:
    type_info, n_dims, size, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    if n_dims == 0:
        return np.empty(0, dtype=np.complex128), pos
    total_elem = int(np.prod(size))
    data, pos = _read_bytes(buf, pos, type_info["name"], total_elem)
    real_part = data[:total_elem]
    imag_part = data[total_elem:]
    c_arr = real_part + 1j * imag_part
    if n_dims == 1 and total_elem == 1:
        return complex(c_arr[0]), pos
    if n_dims > 1:
        return c_arr.reshape(size, order="F"), pos
    return c_arr, pos


def _deserialize_cell(buf: memoryview, pos: int) -> Tuple[Any, int]:
    _, n_dims, size, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    if n_dims == 0:
        return [], pos
    total_elem = int(np.prod(size))
    items: List[Any] = []
    for _ in range(total_elem):
        val, pos = _deserialize_value(buf, pos)
        items.append(val)
    return items, pos


def _deserialize_struct(buf: memoryview, pos: int) -> Tuple[Any, int]:
    _, n_dims, size, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    if n_dims == 0:
        return {}, pos
    n_elems = int(np.prod(size))
    n_fields_arr, pos = _read_bytes(buf, pos, "uint32", 1)
    n_fields = int(n_fields_arr[0])
    if n_fields > 0:
        fn_lengths_arr, pos = _read_bytes(buf, pos, "uint32", n_fields)
        fn_lengths = [int(x) for x in fn_lengths_arr]
        total_fn_chars = sum(fn_lengths)
        if total_fn_chars > 0:
            fn_chars_arr, pos = _read_bytes(buf, pos, "char", total_fn_chars)
            fn_chars = fn_chars_arr.tobytes().decode("latin1")
        else:
            fn_chars = ""
        field_names: List[str] = []
        offset = 0
        for length in fn_lengths:
            field_names.append(fn_chars[offset : offset + length])
            offset += length
    else:
        field_names = []

    raw_cells, pos = _deserialize_cell(buf, pos)
    if n_fields == 0:
        if n_elems == 1:
            return {}, pos
        return [{} for _ in range(n_elems)], pos

    structs: List[Dict[str, Any]] = []
    for elem_idx in range(n_elems):
        d: Dict[str, Any] = {}
        base = elem_idx * n_fields
        for f_idx, fn in enumerate(field_names):
            d[fn] = raw_cells[base + f_idx]
        structs.append(d)

    if n_elems == 1:
        return structs[0], pos
    return structs, pos


def _deserialize_function_handle(buf: memoryview, pos: int) -> Tuple[Any, int]:
    _, _, kind_tuple, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    kind = kind_tuple[0]
    if kind == 64:
        # simple function: db_file cell + name char
        _, pos = _deserialize_cell(buf, pos)
        name, pos = _deserialize_simple_data(buf, pos)
        return f"@{name}", pos
    elif kind == 128:
        # anonymous function: code char + workspace struct
        code, pos = _deserialize_simple_data(buf, pos)
        _, pos = _deserialize_struct(buf, pos)
        return code, pos
    elif kind == 192:
        # scoped/nested function: parentage cell
        parentage, pos = _deserialize_cell(buf, pos)
        return str(parentage[-1]) if parentage else "@func", pos
    raise HoraceError(
        "HORACE:deserialize:invalid_argument",
        f"Unsupported function handle type: {kind} at position {pos}.",
    )


def _deserialize_object(buf: memoryview, pos: int) -> Tuple[Any, int]:
    _, n_dims, size, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    cls_name, pos = _deserialize_simple_data(buf, pos)
    if n_dims == 0 or int(np.prod(size)) == 0:
        return None, pos
    ser_tag_arr, pos = _read_bytes(buf, pos, "uint8", 1)
    ser_tag = int(ser_tag_arr[0])
    contents, pos = _deserialize_value(buf, pos)
    from pyhorace.core.serializable import SERIALIZABLE_REGISTRY, Serializable

    target_cls = SERIALIZABLE_REGISTRY.get(cls_name)
    if target_cls is not None:
        if ser_tag == 1 and hasattr(target_cls, "loadobj"):
            return target_cls.loadobj(contents), pos
        if isinstance(contents, dict):
            if "serial_name" in contents:
                return Serializable.from_struct(contents), pos
            obj = target_cls()
            return obj.from_bare_struct(contents), pos
    return contents, pos


def _deserialize_sparse(buf: memoryview, pos: int) -> Tuple[Any, int]:
    type_info, _, sze, pos = hlp_serial_types.unpack_data_tag(buf, pos)
    n_elem_arr, pos = _read_bytes(buf, pos, "uint32", 1)
    n_elem = int(n_elem_arr[0])
    if not sze:
        return np.empty((0, 0)), pos
    tname = type_info["name"]
    data_format = "uint8" if tname == "sparse_logical" else tname[len("sparse_"):]
    i_arr, pos = _read_bytes(buf, pos, "uint64", n_elem)
    j_arr, pos = _read_bytes(buf, pos, "uint64", n_elem)
    if tname == "sparse_complex_double":
        s_arr, pos = _deserialize_complex_data(buf, pos)
    else:
        s_arr, pos = _read_bytes(buf, pos, data_format, n_elem)
    out = np.zeros(sze, dtype=bool if tname == "sparse_logical" else s_arr.dtype)
    if n_elem > 0:
        out[i_arr.astype(int) - 1, j_arr.astype(int) - 1] = s_arr
    return out, pos


def _obj_deserialize_itself(buf: memoryview, pos: int) -> Tuple[Any, int]:
    from pyhorace.core.serializable import Serializable

    pos += 1  # advance past tag 32
    S, nbytes = hlp_deserialize(buf, pos)
    pos += nbytes
    if isinstance(S, Serializable):
        return S, pos
    if isinstance(S, dict) and "serial_name" in S:
        return Serializable.from_struct(S), pos
    return S, pos

