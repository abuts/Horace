"""Binary file handle management and low-level I/O helpers for Horace SQW/DND files.

Ported from:
- horace_core/sqw/file_io/sqw_fopen.m
- horace_core/sqw/file_io/class_helpers/obj_init.m
- horace_core/sqw/file_io/class_helpers/do_fseek.m
- horace_core/sqw/file_io/class_helpers/is_sqw_struct.m
- horace_core/sqw/file_io/class_helpers/pack_io_outputs.m
- horace_core/sqw/file_io/class_helpers/is_holder.m
"""

from __future__ import annotations

import io
import os
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError

# Global registry mapping integer fid (>= 3) -> metadata dict
_OPENED_FILES: Dict[int, Dict[str, Any]] = {}
_NEXT_FID: int = 3

_PERM_MAP = {
    "r": "rb",
    "rb": "rb",
    "w": "wb+",
    "wb": "wb+",
    "w+": "wb+",
    "wb+": "wb+",
    "r+": "rb+",
    "rb+": "rb+",
    "a": "ab+",
    "ab": "ab+",
    "a+": "ab+",
    "ab+": "ab+",
}


def _register_stream(stream: BinaryIO, filename: str, permission: str) -> int:
    global _NEXT_FID
    # Clean up any closed streams first
    closed_fids = [fid for fid, info in _OPENED_FILES.items() if info["stream"].closed]
    for fid in closed_fids:
        _OPENED_FILES.pop(fid, None)

    fid = _NEXT_FID
    _NEXT_FID += 1
    _OPENED_FILES[fid] = {
        "stream": stream,
        "filename": str(filename),
        "permission": str(permission),
        "machinefmt": "ieee-le",
        "encoding": "windows-1252",
        "last_error": ("", 0),
    }
    return fid


def sqw_fopen(file: Union[str, os.PathLike], permissions: str = "rb") -> int:
    """Open a Horace binary SQW/DND file and return an integer file identifier (fid >= 3)."""
    file_str = str(file)
    fp = os.path.dirname(file_str)
    fn = os.path.basename(file_str)
    if fp and not os.path.isdir(fp):
        try:
            os.makedirs(fp, exist_ok=True)
        except OSError as exc:
            raise HoraceError(
                "HERBERT:utilities:runtime_error",
                f"Folder {fp} to write file: {fn} does not exist and you can not create it because of: {exc}",
            ) from exc

    py_mode = _PERM_MAP.get(permissions, permissions)
    if "b" not in py_mode:
        py_mode += "b"

    try:
        stream = open(file_str, py_mode)
    except OSError as exc:
        raise HoraceError(
            "HERBERT:utilities:runtime_error",
            f"Can not open file: {file_str} with permissions: {permissions}",
        ) from exc

    return _register_stream(stream, file_str, permissions)


def matlab_fopen(
    file_or_fid: Union[str, os.PathLike, int],
    permissions: str = "r",
) -> Union[Tuple[int, str], Tuple[str, str, str, str]]:
    """Emulate MATLAB `fopen(filename, permission)` or `fopen(fid)`."""
    if isinstance(file_or_fid, (int, np.integer)):
        return sqw_fopen_info(int(file_or_fid))
    if file_or_fid == "all":
        return opened_files()  # type: ignore[return-value]
    file_str = str(file_or_fid)
    py_mode = _PERM_MAP.get(permissions, permissions)
    if "b" not in py_mode:
        py_mode += "b"
    try:
        stream = open(file_str, py_mode)
    except OSError as exc:
        return -1, str(exc)
    fid = _register_stream(stream, file_str, permissions)
    return fid, ""


def sqw_fclose(fid: Union[int, str, Any]) -> int:
    """Close an open file identifier (or 'all' to close all open files). Returns 0 on success, -1 on error."""
    if isinstance(fid, str) and fid == "all":
        for f_id in list(_OPENED_FILES.keys()):
            sqw_fclose(f_id)
        return 0
    if hasattr(fid, "close") and callable(fid.close):
        try:
            fid.close()
            for f_id, info in list(_OPENED_FILES.items()):
                if info["stream"] is fid:
                    _OPENED_FILES.pop(f_id, None)
            return 0
        except Exception:
            return -1
    if not isinstance(fid, (int, np.integer)):
        return -1
    fid_int = int(fid)
    info = _OPENED_FILES.pop(fid_int, None)
    if info is None:
        return -1
    try:
        info["stream"].close()
        return 0
    except Exception:
        return -1


def sqw_fopen_info(fid: Union[int, Any]) -> Tuple[str, str, str, str]:
    """Emulate MATLAB `[filename, permission, machinefmt, encoding] = fopen(fid)`."""
    if hasattr(fid, "name") and hasattr(fid, "read"):
        if getattr(fid, "closed", False):
            return "", "", "", ""
        return str(getattr(fid, "name", "")), getattr(fid, "mode", "rb"), "ieee-le", "windows-1252"
    if not isinstance(fid, (int, np.integer)):
        return "", "", "", ""
    info = _OPENED_FILES.get(int(fid))
    if info is None or info["stream"].closed:
        _OPENED_FILES.pop(int(fid), None)
        return "", "", "", ""
    return (
        info["filename"],
        info["permission"],
        info["machinefmt"],
        info["encoding"],
    )


def opened_files() -> List[int]:
    """Emulate MATLAB `fopen('all')`, returning list of open integer FIDs."""
    closed = [fid for fid, info in _OPENED_FILES.items() if info["stream"].closed]
    for fid in closed:
        _OPENED_FILES.pop(fid, None)
    return list(_OPENED_FILES.keys())


def get_stream(fid: Union[int, BinaryIO]) -> Optional[BinaryIO]:
    """Return the underlying binary stream for `fid`, or None if closed/invalid."""
    if hasattr(fid, "read") and hasattr(fid, "seek"):
        if getattr(fid, "closed", False):
            return None
        return fid  # type: ignore[return-value]
    if not isinstance(fid, (int, np.integer)):
        return None
    info = _OPENED_FILES.get(int(fid))
    if info is None or info["stream"].closed:
        return None
    return info["stream"]


def do_fseek(fid: Union[int, BinaryIO], offset: int, origin: Union[str, int] = "bof") -> int:
    """Seek to `offset` in `fid` and raise `HORACE:file_io:io_error` on failure."""
    whence_map = {"bof": io.SEEK_SET, "cof": io.SEEK_CUR, "eof": io.SEEK_END, -1: io.SEEK_SET, 0: io.SEEK_CUR, 1: io.SEEK_END}
    whence = whence_map.get(origin, io.SEEK_SET)
    stream = get_stream(fid)
    if stream is None:
        raise HoraceError(
            "HORACE:file_io:io_error",
            f"Error seeking to position {offset}: file (fid={fid}) is not open",
        )
    try:
        pos = int(offset)
        if pos < 0 and whence == io.SEEK_SET:
            if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
                _OPENED_FILES[int(fid)]["last_error"] = ("Invalid offset", -25)
            raise HoraceError(
                "HORACE:file_io:io_error",
                f"Unable to seek to offset {offset} from {origin}: negative offset",
            )
        stream.seek(pos, whence)
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = ("", 0)
        return 0
    except HoraceError:
        raise
    except Exception as exc:
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = (str(exc), -1)
        raise HoraceError(
            "HORACE:file_io:io_error",
            f"Error seeking to position {offset}: {exc}",
        ) from exc


def do_ftell(fid: Union[int, BinaryIO]) -> int:
    """Return current stream position in bytes (or -1 if invalid)."""
    stream = get_stream(fid)
    if stream is None:
        return -1
    try:
        return int(stream.tell())
    except Exception:
        return -1


def do_ferror(fid: Union[int, BinaryIO]) -> Tuple[str, int]:
    """Return `(mess, res)` error status for `fid`."""
    if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
        return _OPENED_FILES[int(fid)].get("last_error", ("", 0))
    stream = get_stream(fid)
    if stream is None:
        return ("Invalid file identifier", -1)
    return ("", 0)


_DTYPE_MAP = {
    "uint8": np.uint8,
    "*uint8": np.uint8,
    "int8": np.int8,
    "*int8": np.int8,
    "char": np.uint8,
    "*char": np.uint8,
    "uint16": np.uint16,
    "*uint16": np.uint16,
    "int16": np.int16,
    "*int16": np.int16,
    "uint32": np.uint32,
    "*uint32": np.uint32,
    "int32": np.int32,
    "*int32": np.int32,
    "uint64": np.uint64,
    "*uint64": np.uint64,
    "int64": np.int64,
    "*int64": np.int64,
    "single": np.float32,
    "*single": np.float32,
    "float32": np.float32,
    "*float32": np.float32,
    "double": np.float64,
    "*double": np.float64,
    "float64": np.float64,
    "*float64": np.float64,
}


def do_fread(
    fid: Union[int, BinaryIO],
    count: Union[int, Tuple[int, ...], List[int]],
    precision: str = "uint8",
) -> Tuple[np.ndarray, int]:
    """Emulate MATLAB `[dat, cnt] = fread(fid, count, precision)`."""
    stream = get_stream(fid)
    if stream is None:
        return np.array([], dtype=np.float64), 0

    prec_clean = precision.strip()
    out_as_double = not prec_clean.startswith("*") and "=>" not in prec_clean
    if "=>" in prec_clean:
        src_p, dst_p = [p.strip() for p in prec_clean.split("=>", 1)]
        dtype = np.dtype(_DTYPE_MAP.get(src_p, np.uint8)).newbyteorder("<")
        out_dtype = _DTYPE_MAP.get(dst_p, dtype)
        out_as_double = False
    else:
        dtype = np.dtype(_DTYPE_MAP.get(prec_clean, np.uint8)).newbyteorder("<")
        out_dtype = np.float64 if out_as_double else dtype

    shape: Optional[Tuple[int, ...]] = None
    if isinstance(count, (tuple, list, np.ndarray)):
        shape = tuple(int(x) for x in count)
        n_elem = int(np.prod(shape))
    else:
        n_elem = int(count)

    n_bytes = n_elem * dtype.itemsize
    try:
        raw = stream.read(n_bytes)
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = ("", 0)
    except Exception as exc:
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = (str(exc), -1)
        return np.array([], dtype=out_dtype), 0

    actual_elems = len(raw) // dtype.itemsize
    if actual_elems == 0:
        return np.array([], dtype=out_dtype), 0
    arr = np.frombuffer(raw[: actual_elems * dtype.itemsize], dtype=dtype).copy()
    if out_as_double and prec_clean not in ("*char",):
        arr = arr.astype(np.float64)
    elif out_dtype != dtype:
        arr = arr.astype(out_dtype)

    if shape is not None and actual_elems == n_elem:
        arr = arr.reshape(shape, order="F")
    return arr, actual_elems


def do_fwrite(
    fid: Union[int, BinaryIO],
    data: Any,
    precision: str = "uint8",
) -> int:
    """Emulate MATLAB `cnt = fwrite(fid, data, precision)`."""
    stream = get_stream(fid)
    if stream is None:
        raise HoraceError(
            "HORACE:file_io:io_error",
            f"Cannot write to closed or invalid file (fid={fid})",
        )
    prec_clean = precision.strip().lstrip("*")
    if prec_clean == "char" and isinstance(data, str):
        raw = data.encode("windows-1252", errors="replace")
        n_elem = len(raw)
    else:
        dtype = np.dtype(_DTYPE_MAP.get(prec_clean, np.uint8)).newbyteorder("<")
        arr = np.asarray(data, dtype=dtype)
        raw = arr.ravel(order="F").tobytes()
        n_elem = int(arr.size)

    try:
        stream.write(raw)
        stream.flush()
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = ("", 0)
        return n_elem
    except Exception as exc:
        if isinstance(fid, (int, np.integer)) and int(fid) in _OPENED_FILES:
            _OPENED_FILES[int(fid)]["last_error"] = (str(exc), -1)
        raise HoraceError(
            "HORACE:file_io:io_error",
            f"Error writing binary data to file (fid={fid}): {exc}",
        ) from exc


class obj_init:
    """Holder of initialization information `(file_id, num_dim)` for binary SQW/DND file accessors.

    Ported from `horace_core/sqw/file_io/class_helpers/obj_init.m`.
    """

    def __init__(self, fid: Optional[int] = None, numdim: Any = "undefined") -> None:
        self._file_id: int = -1
        self._num_dim: Union[int, str] = "undefined"
        if fid is None:
            return
        self.file_id = fid
        self.num_dim = numdim

    def defined(self) -> bool:
        return self._file_id > 0 and isinstance(self._num_dim, (int, np.integer))

    @property
    def file_id(self) -> int:
        return self._file_id

    @file_id.setter
    def file_id(self, fid: Any) -> None:
        if isinstance(fid, (bool, np.bool_)) or not isinstance(fid, (int, np.integer)) or int(fid) < 1:
            raise HoraceError(
                "HORACE:sqw_file_interface:invalid_argument",
                f"First argument of the constructor must contain open file_id. It is: {fid!r}",
            )
        self._file_id = int(fid)

    @property
    def num_dim(self) -> Union[int, str]:
        return self._num_dim

    @num_dim.setter
    def num_dim(self, numdim: Any) -> None:
        if numdim is None or (isinstance(numdim, (list, tuple, np.ndarray)) and len(numdim) == 0) or (
            isinstance(numdim, str) and numdim == "undefined"
        ):
            self._num_dim = "undefined"
            return
        if isinstance(numdim, (bool, np.bool_)) or not isinstance(numdim, (int, float, np.number)) or not (0 <= int(numdim) <= 4):
            raise HoraceError(
                "HORACE:sqw_file_interface:invalid_argument",
                f'Second argument of the constructor has to be a number of dimensions in range [0-4] or word "undefined". It is: {numdim!r}',
            )
        self._num_dim = int(numdim)


def is_sqw_struct(in_obj: Any) -> bool:
    """Return True if `in_obj` is a legacy SQW/DND dict structure."""
    if not isinstance(in_obj, dict):
        return False
    sqw_fields = {"main_header", "header", "detpar", "data", "experiment_info", "pix"}
    dnd_fields = {"filename", "filepath", "title", "alatt", "angdeg", "uoffset", "u_to_rlu", "ulen", "label", "iax", "iint", "pax", "p", "dax", "s", "e", "npix"}
    keys = set(in_obj.keys())
    return bool(keys.intersection(sqw_fields)) or len(keys.intersection(dnd_fields)) >= 5


def is_holder(obj: Any) -> bool:
    """Return True if `obj` is a wrapper/holder structure with header/data/pix fields."""
    if isinstance(obj, dict):
        return any(k in obj for k in ("main_header", "exp_info", "experiment_info", "detpar", "data", "pix"))
    return any(hasattr(obj, k) for k in ("main_header", "exper_info", "experiment_info", "detpar", "data", "pix"))


def pack_io_outputs(outputs: Sequence[Any], nargout: int = 1) -> Any:
    """Pack file accessor outputs according to requested `nargout`."""
    if nargout <= 1:
        return outputs[0] if outputs else None
    return tuple(outputs[:nargout])

