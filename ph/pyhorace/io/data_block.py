"""
``data_block`` — description of a binary data block in a v4 Horace file.

Ported from ``horace_core/sqw/file_io/class_helpers/@data_block``.
"""
from __future__ import annotations

import copy
from typing import Any, List, Optional, Tuple

import numpy as np

from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.io.hlp_serial import hlp_deserialize, hlp_serial_size, hlp_serialize
from pyhorace.io.sqw_fopen import do_ferror, do_fread, do_fseek, do_fwrite
from pyhorace.io.sqw_serializer import sqw_serializer


@register_serializable
class data_block(Serializable):
    """Defines a binary block in a v4 Horace file and links to a part of an SQW/DND object.

    In MATLAB: ``data_block < serializable``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._sqw_prop_name: str = ""
        self._level2_prop_name: str = ""
        self._position: int = 0
        self._size: int = 0
        self._serialized_obj_cache: Any = None
        self._locked: bool = False

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return

        if len(args) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict):
            self.from_bare_struct(args[0])
            return

        if len(args) == 1 and isinstance(args[0], data_block):
            other = args[0]
            self._sqw_prop_name = other._sqw_prop_name
            self._level2_prop_name = other._level2_prop_name
            self._position = other._position
            self._size = other._size
            self._locked = other._locked
            return

        fld = self.saveable_fields()
        if len(args) > len(fld):
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Expected at most {len(fld)} positional arguments, got {len(args)}.",
            )
        self._init_from_args(fld, args)

    def _init_from_args(self, fld: List[str], args: Tuple[Any, ...]) -> None:
        self._do_check_combo_arg = False
        for i, val in enumerate(args):
            setattr(self, fld[i], val)
        self._do_check_combo_arg = True
        self.check_combo_arg()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sqw_prop_name(self) -> str:
        return self._sqw_prop_name

    @sqw_prop_name.setter
    def sqw_prop_name(self, val: Any) -> None:
        self._set_prop_name(val, "_sqw_prop_name")

    @property
    def level2_prop_name(self) -> str:
        return self._level2_prop_name

    @level2_prop_name.setter
    def level2_prop_name(self, val: Any) -> None:
        self._set_prop_name(val, "_level2_prop_name")

    def _set_prop_name(self, val: Any, attr: str) -> None:
        if val is None or (isinstance(val, (list, np.ndarray)) and len(val) == 0):
            setattr(self, attr, "")
            return
        if not isinstance(val, str):
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Property name must be a character string, got {type(val).__name__}.",
            )
        setattr(self, attr, val)

    @property
    def position(self) -> int:
        return int(self._position)

    @position.setter
    def position(self, val: Any) -> None:
        if val is None or isinstance(val, (bool, str, dict, list, tuple)):
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block position must be a non-negative integer scalar, got {val!r}.",
            )
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise HoraceError(
                    "HORACE:data_block:invalid_argument",
                    "Block position must be a scalar.",
                )
            val = val.item()
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block position must be numeric, got {val!r}.",
            ) from exc
        if np.isnan(fval) or np.isinf(fval) or fval < 0:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block position must be non-negative, got {val!r}.",
            )
        self._position = int(np.uint64(round(fval)))

    @property
    def size(self) -> int:
        return int(self.get_size())

    @size.setter
    def size(self, val: Any) -> None:
        self.set_size(val)

    def get_size(self) -> int:
        return int(self._size)

    def set_size(self, val: Any) -> None:
        if val is None or isinstance(val, (bool, str, dict, list, tuple)):
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block size must be a non-negative integer scalar, got {val!r}.",
            )
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise HoraceError(
                    "HORACE:data_block:invalid_argument",
                    "Block size must be a scalar.",
                )
            val = val.item()
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block size must be numeric, got {val!r}.",
            ) from exc
        if np.isnan(fval) or np.isinf(fval) or fval < 0:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Block size must be non-negative, got {val!r}.",
            )
        self._size = int(np.uint64(round(fval)))

    @property
    def locked(self) -> bool:
        return bool(self._locked)

    @locked.setter
    def locked(self, val: Any) -> None:
        self._locked = bool(val)

    @property
    def allocated(self) -> bool:
        return self.position > 0 and self.size > 0

    @property
    def block_name(self) -> str:
        return f"bl_{self._sqw_prop_name}_{self._level2_prop_name}"

    @property
    def serial_name(self) -> str:
        return type(self).__name__

    @property
    def bat_format(self) -> dict:
        return {
            "serial_name": "",
            "sqw_prop_name": "",
            "level2_prop_name": "",
            "position": np.uint64(1),
            "size": np.uint64(1),
        }

    @property
    def bat_record_size(self) -> int:
        rec = {
            "serial_name": self.serial_name,
            "sqw_prop_name": self.sqw_prop_name,
            "level2_prop_name": self.level2_prop_name,
            "position": np.uint64(self.position),
            "size": np.uint64(self.size),
        }
        ser = sqw_serializer()
        _, pos, *_ = ser.calculate_positions(self.bat_format, rec, 1)
        return int(pos - 1)

    @property
    def bat_record(self) -> np.ndarray:
        rec = {
            "serial_name": self.serial_name,
            "sqw_prop_name": self.sqw_prop_name,
            "level2_prop_name": self.level2_prop_name,
            "position": np.uint64(self.position),
            "size": np.uint64(self.size),
        }
        ser = sqw_serializer()
        raw = ser.serialize(rec, self.bat_format)
        return np.frombuffer(raw, dtype=np.uint8).copy()

    @staticmethod
    def deserialize_bat_record(bindata: Any, pos: int = 1) -> Tuple["data_block", int]:
        from pyhorace.io.dnd_data_block import dnd_data_block
        from pyhorace.io.pix_data_block import pix_data_block

        ser = sqw_serializer()
        fmt = data_block().bat_format
        res, next_pos = ser.deserialize_bytes(bindata, fmt, pos)
        cls_name = res.pop("serial_name")
        cls_map = {
            "data_block": data_block,
            "dnd_data_block": dnd_data_block,
            "pix_data_block": pix_data_block,
        }
        cls = cls_map.get(cls_name)
        if cls is None:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Unknown data_block class '{cls_name}'.",
            )
        obj = cls(res)
        return obj, int(next_pos)

    # ------------------------------------------------------------------
    # Sub-object extraction and insertion
    # ------------------------------------------------------------------

    def get_subobj(self, sqw_dnd_obj: Any) -> Any:
        """Extract the sub-object targeted by this block from *sqw_dnd_obj*."""
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import sqw

        prop1 = self._sqw_prop_name
        prop2 = self._level2_prop_name

        if isinstance(sqw_dnd_obj, dict):
            has_prop1 = prop1 in sqw_dnd_obj or (
                prop1 == "experiment_info" and "exper_info" in sqw_dnd_obj
            ) or (
                prop1 == "exper_info" and "experiment_info" in sqw_dnd_obj
            )
            if has_prop1:
                if prop1 in sqw_dnd_obj:
                    subobj = sqw_dnd_obj[prop1]
                elif prop1 == "experiment_info":
                    subobj = sqw_dnd_obj["exper_info"]
                else:
                    subobj = sqw_dnd_obj["experiment_info"]
                if prop2:
                    if isinstance(subobj, dict) and prop2 in subobj:
                        subobj = subobj[prop2]
                    elif hasattr(subobj, prop2):
                        subobj = getattr(subobj, prop2)
                return subobj
            return sqw_dnd_obj

        if isinstance(sqw_dnd_obj, sqw):
            if not prop1:
                subobj = sqw_dnd_obj
            elif hasattr(sqw_dnd_obj, prop1):
                subobj = getattr(sqw_dnd_obj, prop1)
            elif prop1 == "experiment_info" and hasattr(sqw_dnd_obj, "exper_info"):
                subobj = getattr(sqw_dnd_obj, "exper_info")
            else:
                raise HoraceError(
                    "HORACE:data_block:invalid_argument",
                    f"SQW object does not have property '{prop1}'.",
                )
            if prop2:
                subobj = getattr(subobj, prop2)
            return subobj

        if isinstance(sqw_dnd_obj, DnDBase):
            if prop1 in ("", "data"):
                subobj = sqw_dnd_obj
            elif hasattr(sqw_dnd_obj, prop1):
                subobj = getattr(sqw_dnd_obj, prop1)
            else:
                subobj = sqw_dnd_obj
            if prop2 and hasattr(subobj, prop2):
                subobj = getattr(subobj, prop2)
            return subobj

        # Otherwise sqw_dnd_obj is already a sub-object (or level-1 sub-object)
        subobj = sqw_dnd_obj
        if prop1 and hasattr(subobj, prop1):
            subobj = getattr(subobj, prop1)
        if prop2 and hasattr(subobj, prop2):
            subobj = getattr(subobj, prop2)
        return subobj

    def set_subobj(self, sqw_dnd_obj: Any, part_to_set: Any) -> Any:
        """Set the sub-object targeted by this block on *sqw_dnd_obj*."""
        from pyhorace.sqw.dnd_base import DnDBase
        from pyhorace.sqw.sqw import sqw

        prop1 = self._sqw_prop_name
        prop2 = self._level2_prop_name

        if isinstance(sqw_dnd_obj, dict):
            if not prop1:
                return part_to_set
            if not prop2:
                sqw_dnd_obj[prop1] = part_to_set
            else:
                if prop1 not in sqw_dnd_obj or sqw_dnd_obj[prop1] is None:
                    sqw_dnd_obj[prop1] = {}
                target = sqw_dnd_obj[prop1]
                if isinstance(target, dict):
                    target[prop2] = part_to_set
                else:
                    setattr(target, prop2, part_to_set)
            return sqw_dnd_obj

        if isinstance(sqw_dnd_obj, sqw):
            actual_prop1 = prop1
            if prop1 == "experiment_info" and not hasattr(type(sqw_dnd_obj), "experiment_info"):
                actual_prop1 = "exper_info"
            if not actual_prop1:
                sqw_dnd_obj.do_check_combo_arg = False
                if prop2:
                    setattr(sqw_dnd_obj, prop2, part_to_set)
                return sqw_dnd_obj
            if not prop2:
                sqw_dnd_obj.do_check_combo_arg = False
                setattr(sqw_dnd_obj, actual_prop1, part_to_set)
            else:
                lvl1 = getattr(sqw_dnd_obj, actual_prop1)
                if hasattr(lvl1, "do_check_combo_arg"):
                    lvl1.do_check_combo_arg = False
                setattr(lvl1, prop2, part_to_set)
            return sqw_dnd_obj

        if isinstance(sqw_dnd_obj, DnDBase):
            sqw_dnd_obj.do_check_combo_arg = False
            if prop1 in ("", "data"):
                if prop2:
                    setattr(sqw_dnd_obj, prop2, part_to_set)
                else:
                    return part_to_set
                return sqw_dnd_obj
            if hasattr(sqw_dnd_obj, prop1):
                if not prop2:
                    setattr(sqw_dnd_obj, prop1, part_to_set)
                else:
                    lvl1 = getattr(sqw_dnd_obj, prop1)
                    if hasattr(lvl1, "do_check_combo_arg"):
                        lvl1.do_check_combo_arg = False
                    setattr(lvl1, prop2, part_to_set)
            return sqw_dnd_obj

        # Generic object
        if hasattr(sqw_dnd_obj, "do_check_combo_arg"):
            sqw_dnd_obj.do_check_combo_arg = False
        if prop1 and hasattr(sqw_dnd_obj, prop1):
            if not prop2:
                setattr(sqw_dnd_obj, prop1, part_to_set)
            else:
                lvl1 = getattr(sqw_dnd_obj, prop1)
                if hasattr(lvl1, "do_check_combo_arg"):
                    lvl1.do_check_combo_arg = False
                setattr(lvl1, prop2, part_to_set)
        elif prop2 and hasattr(sqw_dnd_obj, prop2):
            setattr(sqw_dnd_obj, prop2, part_to_set)
        return sqw_dnd_obj

    # ------------------------------------------------------------------
    # Size calculation and file I/O
    # ------------------------------------------------------------------

    def calc_obj_size(
        self,
        sqw_obj: Any,
        nocache: bool = False,
        check_size: bool = False,
    ) -> "data_block":
        """Calculate serialized size of the targeted sub-object."""
        subobj = self.get_subobj(sqw_obj)
        if nocache:
            if isinstance(subobj, Serializable):
                sz = int(subobj.serial_size())
            else:
                sz = int(hlp_serial_size(subobj))
            self._serialized_obj_cache = None
        else:
            if isinstance(subobj, Serializable):
                ser = subobj.serialize()
            else:
                ser = hlp_serialize(subobj)
            if isinstance(ser, np.ndarray):
                ser = ser.tobytes()
            sz = len(ser)
            self._serialized_obj_cache = ser

        if check_size and self.size != sz:
            raise HoraceError(
                "HORACE:data_block:runtime_error",
                f"Block {self.block_name}: expected size {self.size}, actual size {sz}.",
            )
        self.size = sz
        return self

    def put_sqw_block(
        self,
        fid: int,
        sqw_obj: Any = None,
        check_size: bool = False,
    ) -> "data_block":
        """Serialize (if needed) and write this block into open file *fid*."""
        if sqw_obj is not None:
            self.calc_obj_size(sqw_obj, nocache=False, check_size=check_size)
        if self._serialized_obj_cache is None:
            raise HoraceError(
                "HORACE:data_block:runtime_error",
                f"Block {self.block_name} has no cached serialized data to write.",
            )
        self.put_bindata_in_file(fid, self._serialized_obj_cache)
        self._serialized_obj_cache = None
        return self

    def get_sqw_block(self, fid: int, sqw_dnd_obj: Any = None, nargout: Optional[int] = None) -> Any:
        """Read and deserialize this block from open file *fid*."""
        if nargout is None:
            nargout = _infer_nargout(default=1)
        subobj = self.get_bindata_from_file(fid)
        if sqw_dnd_obj is None or (isinstance(sqw_dnd_obj, (list, np.ndarray)) and len(sqw_dnd_obj) == 0):
            res = subobj
        else:
            res = self.set_subobj(sqw_dnd_obj, subobj)
        if nargout >= 2:
            return self, res
        return res

    def put_bindata_in_file(self, fid: int, obj_data: Any) -> None:
        """Write raw serialized bytes *obj_data* at ``self.position``."""
        self.move_to_position(fid)
        if isinstance(obj_data, np.ndarray):
            raw = obj_data.tobytes()
        elif isinstance(obj_data, (bytes, bytearray, memoryview)):
            raw = bytes(obj_data)
        else:
            raw = hlp_serialize(obj_data)
        do_fwrite(fid, np.frombuffer(raw, dtype=np.uint8), "uint8")
        self.check_write_error(fid)

    def get_bindata_from_file(self, fid: int) -> Any:
        """Read ``self.size`` bytes from ``self.position`` and deserialize."""
        # Ensure common serializable classes are registered
        import pyhorace.sqw.dnd_metadata  # noqa: F401
        import pyhorace.data.pixels.pix_metadata  # noqa: F401
        import pyhorace.io.main_header_cl  # noqa: F401
        import pyhorace.data.experiment  # noqa: F401

        self.move_to_position(fid)
        raw_arr = do_fread(fid, self.size, "*uint8")
        self.check_read_error(fid)
        val, _ = hlp_deserialize(raw_arr)
        if isinstance(val, dict) and "serial_name" in val:
            return Serializable.from_struct(val)
        return val

    def move_to_position(self, fid: int, pos: Optional[int] = None) -> None:
        target = self.position if pos is None else int(pos)
        try:
            do_fseek(fid, target, "bof")
        except HoraceError as exc:
            raise HoraceError(
                "HORACE:data_block:io_error",
                f"Unable to seek to position {target} for block {self.block_name}: {exc}",
            ) from exc

    def check_write_error(self, fid: int, add_info: str = "") -> None:
        msg, errnum = do_ferror(fid)
        if errnum != 0:
            raise HoraceError(
                "HORACE:data_block:io_error",
                f"Write error in block {self.block_name} {add_info}: {msg}",
            )

    def check_read_error(self, fid: int, add_info: str = "") -> None:
        msg, errnum = do_ferror(fid)
        if errnum != 0:
            raise HoraceError(
                "HORACE:data_block:io_error",
                f"Read error in block {self.block_name} {add_info}: {msg}",
            )

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return ["sqw_prop_name", "level2_prop_name", "position", "size"]

    def from_bare_struct(self, s: dict) -> "data_block":
        if "sqw_prop_name" in s:
            self.sqw_prop_name = s["sqw_prop_name"]
        if "level2_prop_name" in s:
            self.level2_prop_name = s["level2_prop_name"]
        if "position" in s:
            self.position = s["position"]
        if "size" in s:
            self.size = s["size"]
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(block_name={self.block_name!r}, "
            f"position={self.position}, size={self.size})"
        )
