"""Base abstract class for Horace binary file accessors (SQW and DND).

Ported from:
- horace_core/sqw/file_io/@horace_binfile_interface/horace_binfile_interface.m
- horace_core/sqw/file_io/@horace_binfile_interface/*.m
- horace_core/sqw/file_io/@horace_binfile_interface/private/*.m
"""

from __future__ import annotations

import copy
import datetime
import os
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import Serializable
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.instr_classes.detectors import IX_detector_array
from pyhorace.io.sqw_fopen import (
    do_ferror,
    do_fread,
    do_fseek,
    do_ftell,
    do_fwrite,
    is_sqw_struct,
    obj_init,
    sqw_fclose,
    sqw_fopen,
    sqw_fopen_info,
)
from pyhorace.io.sqw_serializer import sqw_serializer

_VALID_VARNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _is_varname(s: str) -> bool:
    return bool(_VALID_VARNAME_RE.match(s))


class horace_binfile_interface(Serializable):
    """Abstract base class for Horace binary file accessors (DND and SQW).

    Replaces MATLAB's `file_closer_` handle-simulation with standard Python
    destructor (`__del__`), `delete()`, and `fclose()` methods.
    """

    app_header_form_: Dict[str, Any] = {
        "appname": "horace",
        "version": float(1.0),
        "sqw_type": np.uint32(1),
        "num_dim": np.uint32(1),
    }
    max_header_size_: int = 4 + 6 + 8 + 4 + 4  # 26 bytes

    fields_to_save_: List[str] = ["filename_", "filepath_", "num_dim_"]

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.version_: Union[int, float, str] = "undefined"
        self.sqw_serializer_: Optional[sqw_serializer] = sqw_serializer()
        self.filename_: str = ""
        self.filepath_: str = ""
        self.file_id_: int = -1
        self.num_dim_: Union[int, str] = "undefined"
        self.sqw_holder_: Any = None
        self.eof_pos_: int = 0
        self.real_eof_pos_: int = 0
        self.upgrade_map_: Any = None
        self.read_only_: bool = True
        self.data_in_file_: bool = False
        self._do_init_on_load: bool = False
        if len(args) > 0:
            self.init(*args)

    def __del__(self) -> None:
        try:
            self.fclose()
        except Exception:
            pass

    def __enter__(self) -> "horace_binfile_interface":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.fclose()

    def __copy__(self) -> "horace_binfile_interface":
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.file_id_ = -1
        if self.file_id_ > 0 and self.full_filename and os.path.isfile(self.full_filename):
            try:
                result.activate(self.read_only_)
            except Exception:
                result.file_id_ = -1
        return result

    def __deepcopy__(self, memo: Dict[int, Any]) -> "horace_binfile_interface":
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "file_id_":
                setattr(result, k, -1)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        if self.file_id_ > 0 and result.full_filename and os.path.isfile(result.full_filename):
            try:
                result.activate(result.read_only_)
            except Exception:
                result.file_id_ = -1
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.filepath_

    @property
    def filename(self) -> str:
        return self.filename_

    @property
    def filepath(self) -> str:
        return self.filepath_

    @property
    def full_filename(self) -> str:
        if not self.filename_ and not self.filepath_:
            return ""
        if not self.filepath_:
            return self.filename_
        if not self.filename_:
            return self.filepath_
        return os.path.join(self.filepath_, self.filename_)

    @full_filename.setter
    def full_filename(self, val: Union[str, os.PathLike]) -> None:
        if not isinstance(val, (str, os.PathLike)):
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"The full filename can be only string or char array. It is: {type(val).__name__}",
            )
        val_str = str(val)
        fp, fn = os.path.split(val_str)
        base, ext = os.path.splitext(fn)
        self.filepath_ = fp
        self.filename_ = base + ext.lower()

    @property
    def data_in_file(self) -> bool:
        return bool(self.data_in_file_)

    @property
    def io_mode(self) -> str:
        if self.file_id_ <= 0:
            return ""
        _, mode, _, _ = sqw_fopen_info(self.file_id_)
        return mode

    @property
    def read_only(self) -> bool:
        if self.file_id_ > 0:
            return self.io_mode in ("rb", "r")
        return bool(self.read_only_)

    @read_only.setter
    def read_only(self, val: bool) -> None:
        self.read_only_ = bool(val)

    @property
    def format_for_object(self) -> str:
        return self.get_format_for_object()

    @property
    def file_version(self) -> str:
        ver = self.faccess_version
        if isinstance(ver, str) or ver is None or (isinstance(ver, float) and np.isnan(ver)):
            return "-v_undefined"
        if float(ver) == int(float(ver)):
            ver_str = f"{int(float(ver)):d}"
        else:
            ver_str = f"{float(ver):g}".replace(".", "_")
        return f"-v{ver_str}"

    @property
    def faccess_version(self) -> Union[int, float]:
        return self.get_faccess_version()

    @property
    def file_id(self) -> int:
        return self.file_id_

    @property
    def file_closer(self) -> int:
        return self.file_id_

    @property
    def num_dim(self) -> Union[int, str]:
        return self.get_num_dim()

    @num_dim.setter
    def num_dim(self, val: Union[int, str]) -> None:
        self.num_dims_to_save = val

    @property
    def num_dims_to_save(self) -> Union[int, str]:
        return self.num_dim_

    @num_dims_to_save.setter
    def num_dims_to_save(self, val: Union[int, str]) -> None:
        if isinstance(val, str):
            if val == "undefined":
                self.num_dim_ = "undefined"
                return
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"num_dim variable can be only number in the range [0:4] or 'undefined'. It is: {val!r}",
            )
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"num_dim variable can be only number in the range [0:4]. It is: {val!r}",
            )
        ival = int(val)
        if ival < 0 or ival > 4 or ival != val:
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"num_dim variable can be only integer in the range [0:4]. It is: {val!r}",
            )
        self.num_dim_ = ival

    @property
    def sqw_type(self) -> Union[bool, str]:
        return self.get_sqw_type()

    @property
    def creation_date(self) -> str:
        return self.get_creation_date()

    @property
    def npix_position(self) -> Union[int, str]:
        return self.get_npix_position()

    @property
    def npixels(self) -> Union[int, str]:
        return self.get_npixels()

    @property
    def sqw_holder(self) -> Any:
        return self.sqw_holder_

    @sqw_holder.setter
    def sqw_holder(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            self.sqw_holder_ = None
            return
        if not self._is_sqw_or_dnd_obj(val):
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"sqw_holder can be initialized by an sqw family of objects only. Trying to assign: {type(val).__name__}",
            )
        self.init_from_sqw_obj(val)

    @property
    def write_allowed(self) -> bool:
        return self.io_mode in ("wb+", "rb+", "r+b", "w+b", "wb", "w")

    @property
    def eof_pos(self) -> int:
        return self.eof_pos_

    # ------------------------------------------------------------------
    # Overloadable Getters
    # ------------------------------------------------------------------

    def get_num_dim(self) -> Union[int, str]:
        if isinstance(self.num_dim_, str):
            return self.num_dim_
        return int(self.num_dim_)

    def get_faccess_version(self) -> Union[int, float]:
        return self.version_  # type: ignore[return-value]

    def get_sqw_type(self) -> Union[bool, str]:
        return False

    def get_format_for_object(self) -> str:
        return "dnd"

    def get_npixels(self) -> Union[int, str]:
        return 0

    def get_npix_position(self) -> Union[int, str]:
        return "undefined"

    def get_creation_date(self) -> str:
        fname = self.full_filename
        if fname and os.path.isfile(fname):
            mtime = os.path.getmtime(fname)
            dt = datetime.datetime.fromtimestamp(mtime)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        return ""

    # ------------------------------------------------------------------
    # File Lifecycle / Destructor / Activation Methods
    # ------------------------------------------------------------------

    def fclose(self) -> "horace_binfile_interface":
        """Close the underlying binary file if open."""
        if getattr(self, "file_id_", -1) > 0:
            sqw_fclose(self.file_id_)
            self.file_id_ = -1
        return self

    def delete(self) -> "horace_binfile_interface":
        """Close file and reset accessor state to uninitialized."""
        self.fclose()
        self.num_dim_ = "undefined"
        self.sqw_holder_ = None
        self.filename_ = ""
        self.filepath_ = ""
        self.real_eof_pos_ = 0
        self.eof_pos_ = 0
        self.upgrade_map_ = None
        self.read_only_ = True
        self.data_in_file_ = False
        return self

    def deactivate(self) -> "horace_binfile_interface":
        """Close open file handle while preserving in-memory metadata."""
        return self.fclose()

    def activate(self, read_only: Optional[ Union[bool, str] ] = None) -> "horace_binfile_interface":
        """Reopen the file associated with this accessor."""
        perm = "rb"
        if isinstance(read_only, str):
            if read_only in ("-write", "write", "rb+", "wb+"):
                perm = "rb+"
                self.read_only_ = False
            else:
                perm = "rb"
                self.read_only_ = True
        elif isinstance(read_only, bool):
            self.read_only_ = read_only
            perm = "rb" if read_only else "rb+"
        else:
            perm = "rb" if self.read_only_ else "rb+"

        if self.file_id_ > 0:
            fn, cur_mode, _, _ = sqw_fopen_info(self.file_id_)
            if fn and cur_mode == perm:
                return self
            self.fclose()

        fname = self.full_filename
        if not fname or not os.path.isfile(fname):
            raise HoraceError(
                "HORACE:horace_binfile_interface:runtime_error",
                f"Can not activate object as file '{fname}' does not exist.",
            )
        self.file_id_ = sqw_fopen(fname, perm)
        return self

    def is_activated(self, read_or_write: Optional[str] = None) -> bool:
        """Return True if the underlying file stream is currently open."""
        if self.file_id_ <= 0:
            return False
        fn, mode, _, _ = sqw_fopen_info(self.file_id_)
        if not fn:
            return False
        if read_or_write is None:
            return True
        rw = str(read_or_write).lower().lstrip("-")
        if rw.startswith("read"):
            return mode in ("rb", "r")
        if rw.startswith("write"):
            return mode in ("wb+", "rb+", "r+b", "w+b", "wb", "w")
        raise HoraceError(
            "HORACE:horace_binfile_interface:invalid_argument",
            f"Unrecognised is_activated option: {read_or_write!r}",
        )

    # ------------------------------------------------------------------
    # Initialization Logic
    # ------------------------------------------------------------------

    def init(self, *args: Any) -> "horace_binfile_interface":
        """Initialize sqw/dnd accessor from `obj_init`, filename, fid, or SQW/DND object."""
        if len(args) == 0:
            return self

        argi = list(args)
        if isinstance(argi[0], horace_binfile_interface):
            self.copy_contents(argi[0])
            argi = argi[1:]

        if len(argi) == 0:
            return self

        inp = argi[0]
        rest = argi[1:]

        if isinstance(inp, obj_init):
            if inp.file_id < 0:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    "Incorrect initialization information from obj_init class. Input file is not opened",
                )
            self.init_input_stream(inp)
            self.init_from_sqw_file(*rest)
            return self
        elif isinstance(inp, (str, os.PathLike, int, np.integer)) and not isinstance(inp, bool):
            ok, objinit, mess = self.should_load(inp)
            if isinstance(inp, (str, os.PathLike)):
                fname = str(inp)
            else:
                fname, _, _, _ = sqw_fopen_info(int(inp))
            if ok:
                self.full_filename = fname
            else:
                if objinit is not None and getattr(objinit, "file_id", -1) > 0:
                    sqw_fclose(objinit.file_id)
                raise HoraceError(
                    "HORACE:horace_binfile_interface:runtime_error",
                    f"Can not initialize loader by input file: {fname}\n Reason: {mess}",
                )
            self.init_input_stream(objinit)
            self.init_from_sqw_file(*rest)
            return self
        elif isinstance(inp, dict) and ("class_name" in inp or "serial_name" in inp):
            return Serializable.loadobj(inp)
        elif self._is_sqw_or_dnd_obj(inp):
            if not isinstance(self.num_dim, str) and self.file_id_ > 0:
                raise HoraceError(
                    "HORACE:binfile_v2_common:runtime_error",
                    "Upgrade of existing object with new sqw/dnd object is not yet implemented",
                )
            if len(rest) >= 1 and hasattr(inp, "full_filename"):
                try:
                    inp.full_filename = str(rest[0])
                except Exception:
                    pass
            self.init_from_sqw_obj(inp)
            if len(rest) >= 1:
                self.set_file_to_update(*rest)
            elif self.filename:
                self.set_file_to_update()
            return self
        else:
            raise HoraceError(
                "HORACE:binfile_v2_common:invalid_argument",
                f"Input can be only sqw/dnd object or sqw file name. In fact, its class is: {type(inp).__name__}",
            )

    def init_input_stream(self, objinit: obj_init) -> "horace_binfile_interface":
        """Initialize file_id and num_dim from `obj_init`."""
        if not objinit.defined() and self.faccess_version != 0:
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                "Attempt to initialize f-accessor using undefined objinit information",
            )
        self.file_id_ = objinit.file_id
        self.num_dim_ = objinit.num_dim
        if self.file_id_ > 0:
            fn, perm, _, _ = sqw_fopen_info(self.file_id_)
            if fn and not self.filename_:
                self.full_filename = fn
            self.read_only_ = perm in ("rb", "r")
            self.data_in_file_ = True
        return self

    @abstractmethod
    def init_from_sqw_obj(self, *args: Any) -> "horace_binfile_interface":
        """Initialize accessor from in-memory SQW or DND object."""

    @abstractmethod
    def init_from_sqw_file(self, *args: Any) -> "horace_binfile_interface":
        """Initialize accessor from open binary SQW or DND file."""

    @staticmethod
    def _is_sqw_or_dnd_obj(val: Any) -> bool:
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import SQW

        if isinstance(val, (SQW, DnDBase)) or is_sqw_struct(val):
            return True
        if isinstance(val, dict):
            if ("main_header" in val and "data" in val) or ("s" in val and "e" in val and "p" in val):
                return True
        if hasattr(val, "s") and hasattr(val, "e") and hasattr(val, "npix"):
            return True
        if hasattr(val, "main_header") and hasattr(val, "data"):
            return True
        # Also support test mock objects like binfile_v4_block_tester
        if hasattr(val, "data") and hasattr(val, "level2_a"):
            return True
        return False

    # ------------------------------------------------------------------
    # Application Header Reading / Writing
    # ------------------------------------------------------------------

    def build_app_header(self, obj_to_save: Any = None) -> Dict[str, Any]:
        """Build Horace sqw-file header dict to write to disk."""
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import SQW

        if obj_to_save is not None and not (isinstance(obj_to_save, (list, tuple)) and len(obj_to_save) == 0):
            if isinstance(obj_to_save, SQW) or (hasattr(obj_to_save, "main_header") and hasattr(obj_to_save, "pix")):
                sqw_type = True
                ndim = obj_to_save.data.NUM_DIMS if hasattr(obj_to_save.data, "NUM_DIMS") else obj_to_save.dimensions()
            elif isinstance(obj_to_save, DnDBase) or is_sqw_struct(obj_to_save) or hasattr(obj_to_save, "dimensions"):
                sqw_type = False
                ndim = obj_to_save.dimensions() if callable(getattr(obj_to_save, "dimensions", None)) else getattr(obj_to_save, "NUM_DIMS", len(getattr(obj_to_save, "p", [])))
            elif isinstance(obj_to_save, dict) and "p" in obj_to_save:
                sqw_type = False
                ndim = len(obj_to_save["p"])
            else:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Unsupported class '{type(obj_to_save).__name__}' to save in sqw binary format.",
                )
        else:
            sqw_type = bool(self.sqw_type is True)
            ndim = self.num_dim

        if isinstance(ndim, str):
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                "Trying to build_app_header but no object to build header from is provided and faccessor itself is not defined",
            )

        ver = (
            self.file_ver_
            if (not getattr(self, "use_faccess_ver_for_header_", True) and getattr(self, "file_ver_", None) is not None)
            else self.faccess_version
        )
        return {
            "appname": "horace",
            "version": float(ver),
            "sqw_type": np.uint32(1 if sqw_type else 0),
            "num_dim": np.uint32(int(ndim)),
        }

    def put_app_header(self, *args: Any) -> "horace_binfile_interface":
        """Write binary application header at the start of the file."""
        self.check_obj_initiated_properly()
        head = self.build_app_header(*args)
        if self.sqw_serializer_ is None:
            self.sqw_serializer_ = sqw_serializer()
        stream_bytes = self.sqw_serializer_.serialize(head, self.app_header_form_)
        try:
            do_fseek(self.file_id_, 0, "bof")
        except Exception as exc:
            raise HoraceError(
                "COMBINE_SQW_PIX_JOB:io_error",
                "Error moving to the beginning of the file",
            ) from exc
        do_fwrite(self.file_id_, stream_bytes, "uint8")
        self.check_error_report_fail_("Error writing the sqw file header")
        return self

    @staticmethod
    def get_file_header(
        file: Union[str, os.PathLike, int], *args: Any
    ) -> Tuple[Dict[str, Any], int]:
        """Open existing file and read the Horace binary file header.

        Returns `(header_dict, fid)` with `fid` left open for subsequent reading.
        """
        max_buffer_size = horace_binfile_interface.max_header_size_
        rem_args = []
        for a in args:
            if isinstance(a, (int, np.integer)) and not isinstance(a, bool):
                max_buffer_size = int(a)
            else:
                rem_args.append(a)

        stream, fid, message = horace_binfile_interface._read_header(file, max_buffer_size, *rem_args)
        if message or fid <= 0:
            if fid > 0:
                sqw_fclose(fid)
            raise HoraceError(
                "HORACE:horace_binfile_interface:io_error",
                f"file: {file}\n Error: {message}",
            )

        header, mess = horace_binfile_interface._extract_hor_version(stream)
        if mess:
            if fid > 0:
                sqw_fclose(fid)
            raise HoraceError(
                "HORACE:horace_binfile_interface:runtime_error",
                f"File: {file}  {mess}",
            )
        return header, fid

    @staticmethod
    def _read_header(
        file_name: Union[str, os.PathLike, int], buf_size: int, *args: Any
    ) -> Tuple[np.ndarray, int, str]:
        open_for_update = False
        for a in args:
            if isinstance(a, str) and a.lower().startswith("-u"):
                open_for_update = True
            elif a is not None:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Unrecognized option {a!r} in read_header_",
                )

        permission_req = "rb+" if open_for_update else "rb"
        if isinstance(file_name, (int, np.integer)) and not isinstance(file_name, bool):
            fid = int(file_name)
            fn, perm, _, _ = sqw_fopen_info(fid)
            if not fn:
                return np.array([], dtype=np.uint8), -1, "No open file with the given file identifier"
            if perm.lower() != permission_req.lower() and not (
                permission_req == "rb" and perm.lower() in ("rb+", "wb+")
            ):
                return (
                    np.array([], dtype=np.uint8),
                    fid,
                    f"Read permission of file that is already open must be {permission_req}",
                )
            do_fseek(fid, 0, "bof")
        else:
            fname = str(file_name)
            if not os.path.isfile(fname):
                raise HoraceError(
                    "HORACE:horace_binfile_interface:io_error",
                    f"file {fname} does not exist",
                )
            fid = sqw_fopen(fname, permission_req)

        stream, count = do_fread(fid, buf_size, "*uint8")
        if count < buf_size:
            mess = (
                f"read_header_::io_error: Can not read first {buf_size} bytes of file. "
                f"File is too small to be an sqw file"
            )
            return stream, fid, mess
        mess, res = do_ferror(fid)
        if res != 0:
            raise HoraceError(
                "HORACE:horace_binfile_interface:io_error",
                f"IO error reading first {buf_size} bytes of the file: Reason {mess}",
            )
        return stream, fid, ""

    @staticmethod
    def _extract_hor_version(data_stream: np.ndarray) -> Tuple[Dict[str, Any], str]:
        ver_struct: Dict[str, Any] = {
            "appname": "horace",
            "version": 0.0,
            "sqw_type": False,
            "num_dim": "undefined",
            "name": "unknown",
            "typestart": 0,
            "uncertain": False,
        }
        raw = np.asarray(data_stream, dtype=np.uint8).tobytes()
        if len(raw) < 4:
            return ver_struct, "is not recognized as Horace binary file"

        n = int(np.frombuffer(raw[:4], dtype="<i4")[0])
        if 0 < n < 1024:
            if len(raw) < 4 + n:
                return ver_struct, "is not recognized as Horace binary file"
            name = raw[4 : 4 + n].decode("latin1")
            if not _is_varname(name):
                ok, uncertain = horace_binfile_interface._check_hor_v0(name, raw)
                if ok:
                    ver_struct["name"] = "horace"
                    ver_struct["typestart"] = 0
                    ver_struct["sqw_type"] = True
                    ver_struct["version"] = 0
                    ver_struct["uncertain"] = bool(uncertain)
                    if not uncertain:
                        ver_struct["num_dim"] = "undefined"
                    return ver_struct, ""
                return ver_struct, "Application name is not valid Matlab variable name"

            if len(raw) < 4 + n + 8:
                return ver_struct, "is not recognized as Horace binary file"
            version = float(np.frombuffer(raw[4 + n : 4 + n + 8], dtype="<f8")[0])
            if np.isnan(version) or version < 0 or version > 99999999:
                return ver_struct, f"Application version: {version} is not allowed version number"
            ver_struct["typestart"] = 4 + n + 8
            ver_struct["version"] = int(version) if version == int(version) else version
            ver_struct["name"] = name
        else:
            return ver_struct, "is not recognized as Horace binary file"

        if n == 6 and name == "horace":
            ts = ver_struct["typestart"]
            if len(raw) >= ts + 8:
                sqw_type_val = int(np.frombuffer(raw[ts : ts + 4], dtype="<i4")[0])
                num_dim_val = int(np.frombuffer(raw[ts + 4 : ts + 8], dtype="<i4")[0])
                if sqw_type_val in (0, 1) and 0 <= num_dim_val <= 4:
                    ver_struct["sqw_type"] = bool(sqw_type_val == 1)
                    ver_struct["num_dim"] = num_dim_val
                    ver_struct["uncertain"] = False
                else:
                    ver_struct["name"] = "horace"
                    ver_struct["typestart"] = 0
                    ver_struct["sqw_type"] = True
                    ver_struct["ver_uncertain"] = True
        return ver_struct, ""

    @staticmethod
    def _check_hor_v0(name: str, raw: bytes) -> Tuple[bool, bool]:
        uncertain = False
        fp, fn = os.path.split(name)
        base, fext = os.path.splitext(fn)
        if not (fp == "" and len(base) > 0 and len(fext) > 0):
            return False, False
        nfn = len(name) + 4
        if len(raw) <= nfn + 4:
            return True, True
        n = int(np.frombuffer(raw[nfn : nfn + 4], dtype="<i4")[0])
        if n < 0 or n > 2046:
            return False, False
        if n > 0:
            if len(raw) < nfn + n + 4:
                return True, True
            path = raw[nfn + 4 : nfn + 4 + n].decode("latin1")
            if _is_varname(path):
                return False, False
        else:
            return False, False
        return True, uncertain

    # ------------------------------------------------------------------
    # Format Probing: should_load / should_load_stream
    # ------------------------------------------------------------------

    def should_load(
        self, filename: Union[str, os.PathLike, int]
    ) -> Tuple[bool, Optional[obj_init], str]:
        """Check if this loader should load `filename`."""
        if not isinstance(filename, (int, np.integer)) or isinstance(filename, bool):
            if not isinstance(filename, (str, os.PathLike)) or not str(filename).strip():
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    "should_load function: filename is empty or not a string",
                )
            full_data_name: Union[str, int] = os.path.abspath(str(filename))
            if not os.path.isfile(full_data_name):
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"should_load function: File {full_data_name.replace(chr(92), '/')} does not exist",
                )
        else:
            full_data_name = int(filename)

        header, fh = horace_binfile_interface.get_file_header(full_data_name)
        return self.should_load_stream(header, fh)

    def should_load_stream(
        self, head_struc: Dict[str, Any], fid: int
    ) -> Tuple[bool, Optional[obj_init], str]:
        """Check if this loader should load the open stream described by `head_struc`."""
        if not isinstance(head_struc, dict) or "sqw_type" not in head_struc or "version" not in head_struc:
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                "the input head_struc structure for should_load_stream function does not have correct format",
            )
        my_sqw = bool(self.sqw_type is True) if not isinstance(self.sqw_type, bool) else self.sqw_type
        if bool(head_struc["sqw_type"]) == my_sqw and head_struc["version"] == self.faccess_version:
            return True, obj_init(fid, head_struc["num_dim"]), ""
        else:
            t_str = "sqw" if my_sqw else "dnd"
            mess = f"not Horace type: {t_str} version: {self.faccess_version:g} file"
            return False, obj_init(), mess

    # ------------------------------------------------------------------
    # File Opening for Writing / Updating
    # ------------------------------------------------------------------

    def set_file_to_update(
        self, filename: Optional[Union[str, os.PathLike]] = None, *args: Any
    ) -> "horace_binfile_interface":
        """Set filename to save sqw/dnd data and open file for write/update operations."""
        obj, _, _ = self._set_file_to_update_impl(filename)
        return obj

    def _set_file_to_update_impl(
        self, filename: Optional[Union[str, os.PathLike]] = None
    ) -> Tuple["horace_binfile_interface", bool, Any]:
        self.data_in_file_ = False
        file_exist = False
        old_ldr = None

        if filename is not None and str(filename) != "":
            if not isinstance(filename, (str, os.PathLike)):
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    "new filename to save needs to be a string",
                )
            new_filename = str(filename)
            if os.path.isfile(new_filename):
                file_exist = True
                try:
                    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

                    old_ldr = sqw_formats_factory.instance().get_loader(new_filename, "-update")
                    self.data_in_file_ = True
                except Exception:
                    file_exist = False
        else:
            if not self.filename:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Trying to reopen loader {self.__class__.__name__}-defined file for writing but loader's filename is empty",
                )
            new_filename = self.full_filename
            if self.file_id_ > 0:
                file_exist = True
                old_filename, access_rights, _, _ = sqw_fopen_info(self.file_id_)
                if old_filename and os.path.abspath(new_filename) == os.path.abspath(old_filename):
                    if access_rights in ("wb+", "rb+", "+wb", "r+b", "w+b"):
                        self.data_in_file_ = True
                        return self, file_exist, self
                    else:
                        self.fclose()
            elif os.path.isfile(new_filename):
                file_exist = True

        if file_exist:
            self.data_in_file_ = True
            if isinstance(self.num_dim, str) and old_ldr is not None:
                self.fclose()
                old_ldr.reopen_to_write()
                return old_ldr, file_exist, old_ldr
            perm = "rb+"
        else:
            perm = "wb+"
            self.data_in_file_ = False

        self.full_filename = new_filename
        if old_ldr is None:
            old_ldr = self
        elif old_ldr is not self:
            old_ldr.fclose()

        self.fclose()
        self.file_id_ = sqw_fopen(self.full_filename, perm)
        self.read_only_ = False
        return self, file_exist, old_ldr

    def reopen_to_write(
        self, filename: Optional[Union[str, os.PathLike, int]] = None
    ) -> "horace_binfile_interface":
        """Reopen already opened file in read-write mode ('rb+') or open new file ('wb+')."""
        obj, _ = self._reopen_to_write_impl(filename)
        return obj

    def _reopen_to_write_impl(
        self, filename: Optional[Union[str, os.PathLike, int]] = None
    ) -> Tuple["horace_binfile_interface", str]:
        if filename is not None and not isinstance(filename, bool) and isinstance(filename, (int, np.integer)):
            fid = int(filename)
            fname, permissions, _, _ = sqw_fopen_info(fid)
            if not fname:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    "Wrong (probably closed) file handle specified as input",
                )
            if permissions not in ("wb+", "rb+", "r+b", "w+b"):
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Get input file handle with incorrect file access for file {fname}",
                )
            self.fclose()
            self.full_filename = fname
            self.file_id_ = fid
            self.read_only_ = False
            return self, permissions

        if filename is not None and not isinstance(filename, (str, os.PathLike)):
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"Wrong type ({type(filename).__name__}) variable 'filename' specified as input",
            )

        fname_str = str(filename) if filename is not None else ""
        if not fname_str and not self.full_filename:
            raise HoraceError(
                "HORACE:horace_binfile_interface:runtime_error",
                "Can not reopen file: The filename is not defined either as input or as faccessor property",
            )

        if self.file_id_ > 0 and not fname_str:
            cur_fn, permissions, _, _ = sqw_fopen_info(self.file_id_)
            if permissions in ("wb+", "rb+", "r+b", "w+b"):
                self.full_filename = cur_fn
                self.read_only_ = False
                return self, permissions

        self.fclose()
        if fname_str:
            self.full_filename = fname_str

        target = self.full_filename
        if os.path.isfile(target):
            permissions = "rb+"
            self.data_in_file_ = True
        else:
            permissions = "wb+"
            self.data_in_file_ = False

        self.file_id_ = sqw_fopen(target, permissions)
        self.read_only_ = False
        return self, permissions

    def check_obj_initiated_properly(self) -> "horace_binfile_interface":
        """Verify if data access object is properly initialized for writing data in it."""
        perm = self.io_mode
        if perm not in ("wb+", "rb+", "r+b", "w+b"):
            self.reopen_to_write()
            fop, perm, _, _ = sqw_fopen_info(self.file_id_)
            if not fop:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:runtime_error",
                    f"put_sqw called but the file {self.full_filename} to write can not be opened in write mode.\n"
                    f"Current permissions are: '{perm}'.\n"
                    f"Initialize writer with sqw||dnd object to save data",
                )
        return self

    def check_obj_initated_properly(self) -> "horace_binfile_interface":
        """Alias preserving MATLAB typo `check_obj_initated_properly`."""
        return self.check_obj_initiated_properly()

    def check_error_report_fail_(self, pos_mess: str) -> None:
        mess, res = do_ferror(self.file_id_)
        if res != 0:
            raise HoraceError(
                "HORACE:sqw_file_interface:io_error",
                f"{pos_mess} -- Reason: {mess}",
            )

    # ------------------------------------------------------------------
    # Static Binary IO Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def move_to_position(fid: int, pos: int) -> None:
        try:
            do_fseek(fid, pos, "bof")
        except Exception as exc:
            file, _, _, _ = sqw_fopen_info(fid)
            raise HoraceError(
                "HORACE:data_block:io_error",
                f'file: "{file}". Error "{exc}"',
            ) from exc

    @staticmethod
    def check_write_error(fid: int, add_info: str = "") -> None:
        horace_binfile_interface._check_io_error(fid, "writing", add_info)

    @staticmethod
    def check_read_error(fid: int, add_info: str = "") -> None:
        horace_binfile_interface._check_io_error(fid, "reading", add_info)

    @staticmethod
    def _check_io_error(fid: int, read_or_write: str, add_info: str = "") -> None:
        mess, res = do_ferror(fid)
        if res != 0:
            file, acc, _, _ = sqw_fopen_info(fid)
            if not add_info:
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f'file: "{file}", IO mode: "{acc}". {read_or_write} error: "{mess}"',
                )
            else:
                raise HoraceError(
                    "HORACE:data_block:io_error",
                    f'file: "{file}", {acc} {read_or_write}. {add_info} Error: "{mess}"',
                )

    @staticmethod
    def parse_get_sqw_args(*args: Any) -> Dict[str, bool]:
        """Parse options for `get_sqw`."""
        flags = [
            "head",
            "his",
            "verbatim",
            "hverbatim",
            "hisverbatim",
            "noupgrade",
            "norange",
            "keep_original",
            "nopix",
            "legacy",
            "sqw_struc",
            "file_backed",
            "force_pix_location",
        ]
        opts: Dict[str, bool] = {f: False for f in flags}
        if len(args) == 1 and isinstance(args[0], dict):
            for k, v in args[0].items():
                opts[k] = bool(v)
            opts["verbatim"] = opts["verbatim"] or opts["hverbatim"]
            return opts

        for arg in args:
            if not isinstance(arg, str):
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Unrecognized non-string option {arg!r} in get_sqw",
                )
            s = arg.strip()
            if s == "-h":
                s = "-his"
            key = s.lstrip("-").lower()
            matched = [f for f in flags if f.startswith(key)]
            if len(matched) == 1:
                opts[matched[0]] = True
                if matched[0] == "file_backed":
                    opts["force_pix_location"] = True
            elif key in opts:
                opts[key] = True
            else:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Unrecognized option {arg!r}",
                )
        opts["verbatim"] = opts["verbatim"] or opts["hverbatim"]
        return opts

    # ------------------------------------------------------------------
    # Detector Conversion Utility
    # ------------------------------------------------------------------

    @staticmethod
    def convert_old_det_forms(
        detpar: Any, n_instances: Optional[Any] = None
    ) -> Union[unique_objects_container, unique_references_container]:
        """Convert legacy detector structures or `IX_detector_array` into `unique_objects_container`."""
        if isinstance(detpar, dict) or (hasattr(detpar, "group") and not isinstance(detpar, IX_detector_array)):
            detpar = IX_detector_array(detpar)

        if isinstance(detpar, IX_detector_array):
            det = unique_objects_container(baseclass="IX_detector_array")
            det.add(detpar)
            det.replicate_runs(n_instances)
            return det
        elif isinstance(detpar, (unique_objects_container, unique_references_container)):
            if detpar.n_objects == 1:
                det = copy.copy(detpar)
                det.replicate_runs(n_instances)
                return det
            elif (
                n_instances is not None
                and not isinstance(n_instances, (bool, str, list, tuple, np.ndarray))
                and isinstance(n_instances, (int, float, np.number))
                and detpar.n_objects == n_instances
            ):
                return detpar
            else:
                raise HoraceError(
                    "HORACE:horace_binfile_interface:invalid_argument",
                    f"Number of IX_detector_array objects ({detpar.n_objects}) provided in "
                    f"the input container is not consistent with number or type of objects "
                    f"requested to return ({n_instances!r}).",
                )
        else:
            raise HoraceError(
                "HORACE:horace_binfile_interface:invalid_argument",
                f"Unrecognized input class: '{type(detpar).__name__}' provided to the conversion method.",
            )

    # ------------------------------------------------------------------
    # Format Upgrade
    # ------------------------------------------------------------------

    def upgrade_file_format(self, *args: Any) -> "horace_binfile_interface":
        """Upgrade file format to the current preferred file format."""
        upgrade_range = False
        rem_args = []
        for a in args:
            if isinstance(a, str) and a.lower().startswith("-upgrade"):
                upgrade_range = True
            else:
                rem_args.append(a)

        copy_arg = ["-write_mode", "-upgrade_range"] if upgrade_range else ["-write_mode"]
        ff_obj = self.format_for_object
        from pyhorace.io.sqw_formats_factory import sqw_formats_factory

        new_obj = sqw_formats_factory.instance().get_pref_access(ff_obj)
        if isinstance(self.num_dim, str):
            return new_obj

        res = new_obj.copy_contents(self, *copy_arg)
        if isinstance(res, tuple):
            new_obj, missing = res
        else:
            new_obj, missing = res, []
        if not missing and type(new_obj) is type(self):
            return new_obj

        acc = new_obj.io_mode
        file_exist = bool(self.full_filename and os.path.isfile(self.full_filename))
        if acc not in ("wb+", "rb+", "r+b", "w+b"):
            new_obj.fclose()
            if self.full_filename:
                new_obj.reopen_to_write(self.full_filename)

        if not file_exist:
            return new_obj
        return self.do_class_dependent_updates(new_obj, upgrade_range, *rem_args)

    @abstractmethod
    def copy_contents(
        self, other_obj: "horace_binfile_interface", *args: Any
    ) -> Any:
        """Copy file positions and open handle from `other_obj`."""

    @abstractmethod
    def do_class_dependent_updates(
        self, new_obj: "horace_binfile_interface", *args: Any
    ) -> "horace_binfile_interface":
        """Perform format-specific migration from `self` to `new_obj`."""

    # ------------------------------------------------------------------
    # Main Abstract Data Access Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_data(self, *args: Any, **kwargs: Any) -> Any:
        """Read data block from file."""

    @abstractmethod
    def get_se_npix(self, *args: Any, **kwargs: Any) -> Any:
        """Read signal, error, and npix arrays from file."""

    @abstractmethod
    def get_instrument(self, *args: Any, **kwargs: Any) -> Any:
        """Read instrument from file."""

    @abstractmethod
    def get_sample(self, *args: Any, **kwargs: Any) -> Any:
        """Read sample from file."""

    @abstractmethod
    def get_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Read SQW or DND object from file."""

    @abstractmethod
    def get_dnd(self, *args: Any, **kwargs: Any) -> Any:
        """Read DND object from file."""

    @abstractmethod
    def get_dnd_metadata(self, *args: Any, **kwargs: Any) -> Any:
        """Read DND metadata from file."""

    @abstractmethod
    def get_pix_range(self) -> np.ndarray:
        """Read 2x4 pixel coordinate range."""

    @abstractmethod
    def get_img_db_range(self, *args: Any) -> np.ndarray:
        """Read 2x4 image database range."""

    @abstractmethod
    def put_sqw(self, *args: Any, **kwargs: Any) -> "horace_binfile_interface":
        """Write SQW or DND object to file."""

    @abstractmethod
    def put_dnd(self, *args: Any, **kwargs: Any) -> "horace_binfile_interface":
        """Write DND object to file."""

    @abstractmethod
    def put_dnd_metadata(self, *args: Any, **kwargs: Any) -> "horace_binfile_interface":
        """Write DND metadata to file."""

    @abstractmethod
    def put_dnd_data(self, *args: Any, **kwargs: Any) -> "horace_binfile_interface":
        """Write DND image arrays to file."""

    @abstractmethod
    def head(self, *args: Any, **kwargs: Any) -> Any:
        """Return summary header structure."""

    # ------------------------------------------------------------------
    # Serializable Interface
    # ------------------------------------------------------------------

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return list(self.fields_to_save_)

    def to_bare_struct(self, *args: Any) -> Dict[str, Any]:
        return {fld: getattr(self, fld) for fld in self.fields_to_save_}

    def from_bare_struct(self, indata: Dict[str, Any]) -> "horace_binfile_interface":
        for name in self.fields_to_save_:
            if name in indata:
                setattr(self, name, indata[name])
        if "sqw_holder" in indata and indata["sqw_holder"] is not None:
            sh = indata["sqw_holder"]
            if isinstance(sh, dict) and ("serial_name" in sh or "class_name" in sh):
                self.sqw_holder_ = Serializable.from_struct(sh)
            else:
                self.sqw_holder_ = sh
        return self


HoraceBinfileInterface = horace_binfile_interface

__all__ = [
    "HoraceBinfileInterface",
    "horace_binfile_interface",
]
