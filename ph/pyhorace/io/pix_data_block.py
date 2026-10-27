"""
``pix_data_block`` — binary block descriptor for ``pix_data`` in v4 Horace files.

Ported from ``horace_core/sqw/file_io/class_helpers/@pix_data_block``.
"""
from __future__ import annotations

from typing import Any, List, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.data.pixels.pix_data import pix_data
from pyhorace.io.data_block import data_block
from pyhorace.io.sqw_fopen import do_fread, do_fwrite


@register_serializable
class pix_data_block(data_block):
    """Binary block descriptor for ``pix_data`` in v4 SQW files.

    Binary layout at ``position``:
    - ``uint32``: ``n_rows`` (typically 9)
    - ``uint64``: ``npixels`` (at ``num_pix_position = position + 4``)
    - ``float32[n_rows * npixels]``: pixel array (at ``pix_position = position + 12``)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._npixels: Union[int, str] = "undefined"
        self._n_rows: int = 9
        super().__init__()
        self._sqw_prop_name = "pix"
        self._level2_prop_name = "data_wrap"
        self._locked = True

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return

        if len(args) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict):
            self.from_bare_struct(args[0])
            return

        if len(args) == 1 and isinstance(args[0], pix_data_block):
            other = args[0]
            self._position = other._position
            self._npixels = other._npixels
            self._n_rows = other._n_rows
            return

        if len(args) == 4:
            # Called with (sqw_prop_name, level2_prop_name, position, size)
            self.sqw_prop_name = args[0]
            self.level2_prop_name = args[1]
            self.position = args[2]
            self.size = args[3]
            return

        fld = self.saveable_fields()
        if len(args) > len(fld):
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Expected at most {len(fld)} positional arguments, got {len(args)}.",
            )
        self._init_from_args(fld, args)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npixels(self) -> Union[int, str]:
        return self._npixels

    @npixels.setter
    def npixels(self, val: Any) -> None:
        if isinstance(val, str):
            if val == "undefined":
                self._npixels = "undefined"
                return
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"npixels string value can only be 'undefined', got {val!r}.",
            )
        if val is None or isinstance(val, (bool, dict, list, tuple)):
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"npixels must be a non-negative integer scalar, got {val!r}.",
            )
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise HoraceError(
                    "HORACE:pix_data_block:invalid_argument",
                    "npixels must be a scalar.",
                )
            val = val.item()
        try:
            fval = float(val)
        except (TypeError, ValueError) as exc:
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"npixels must be numeric, got {val!r}.",
            ) from exc
        if np.isnan(fval) or np.isinf(fval) or fval < 0:
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"npixels must be non-negative, got {val!r}.",
            )
        self._npixels = int(np.uint64(round(fval)))

    @property
    def n_rows(self) -> int:
        return int(self._n_rows)

    @n_rows.setter
    def n_rows(self, val: Any) -> None:
        if val is None or isinstance(val, (bool, str, dict, list, tuple)):
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"n_rows must be a positive integer scalar, got {val!r}.",
            )
        if isinstance(val, np.ndarray):
            if val.size != 1:
                raise HoraceError(
                    "HORACE:pix_data_block:invalid_argument",
                    "n_rows must be a scalar.",
                )
            val = val.item()
        ival = int(val)
        if ival <= 0 or ival > 1024:
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"n_rows must be in (0, 1024], got {val!r}.",
            )
        self._n_rows = ival

    @property
    def num_pix_position(self) -> int:
        return int(self.position + 4)

    @property
    def pix_position(self) -> int:
        return int(self.position + 12)

    @pix_position.setter
    def pix_position(self, val: Any) -> None:
        ival = int(val)
        if ival < 12:
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"pix_position must be >= 12, got {val!r}.",
            )
        self.position = ival - 12

    def get_size(self) -> int:
        if isinstance(self._npixels, str):
            return 12
        return int(4 + 8 + int(self._npixels) * self._n_rows * 4)

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
        ival = int(val)
        all_pix_size = ival - 12
        pix_size = self._n_rows * 4
        if all_pix_size < 0 or all_pix_size % pix_size != 0:
            raise HoraceError(
                "HORACE:data_block:invalid_argument",
                f"Invalid pix_data_block size {ival}: (size - 12) must be a non-negative multiple of {pix_size}.",
            )
        self._npixels = all_pix_size // pix_size

    # ------------------------------------------------------------------
    # Size calculation and file I/O
    # ------------------------------------------------------------------

    def calc_obj_size(
        self,
        sqw_obj: Any,
        nocache: bool = False,
        check_size: bool = False,
    ) -> "pix_data_block":
        subobj = self.get_subobj(sqw_obj)
        if not isinstance(subobj, pix_data):
            raise HoraceError(
                "HORACE:pix_data_block:invalid_argument",
                f"Expected pix_data instance, got {type(subobj).__name__}.",
            )
        new_npix = int(subobj.npix)
        new_nrows = int(subobj.n_rows)
        new_size = 4 + 8 + new_npix * new_nrows * 4
        if check_size and self.size != new_size:
            raise HoraceError(
                "HORACE:data_block:runtime_error",
                f"Block {self.block_name}: expected size {self.size}, actual size {new_size}.",
            )
        self.npixels = new_npix
        self.n_rows = new_nrows
        self._serialized_obj_cache = None if nocache else subobj
        return self

    def put_data_header(self, fid: int) -> "pix_data_block":
        """Write ``n_rows`` (uint32) and ``npixels`` (uint64) at ``self.position``."""
        if isinstance(self._npixels, str):
            raise HoraceError(
                "HORACE:pix_data_block:runtime_error",
                "Attempt to write pixel data header block while BAT has not been initialized.",
            )
        self.move_to_position(fid)
        do_fwrite(fid, np.uint32(self._n_rows), "uint32")
        self.check_write_error(fid, "num pix rows")
        do_fwrite(fid, np.uint64(self._npixels), "uint64")
        self.check_write_error(fid, "num_pixels")
        return self

    def put_bindata_in_file(self, fid: int, obj_data: Any) -> None:
        self.move_to_position(fid)
        n_rows = int(obj_data.n_rows)
        npix = int(obj_data.npix)
        do_fwrite(fid, np.uint32(n_rows), "uint32")
        self.check_write_error(fid, "num pix rows")
        do_fwrite(fid, np.uint64(npix), "uint64")
        self.check_write_error(fid, "num_pixels")
        data = getattr(obj_data, "data", None)
        if isinstance(data, np.ndarray) and data.size > 0:
            do_fwrite(fid, np.asarray(data, dtype=np.float32).ravel(order="F"), "float32")
            self.check_write_error(fid, "writing pixel data block")

    def get_bindata_from_file(self, fid: int) -> pix_data:
        self.move_to_position(fid)
        n_rows_arr = do_fread(fid, 1, "*uint32")
        self.check_read_error(fid, "num pix rows")
        n_rows = int(n_rows_arr[0])
        n_pix_arr = do_fread(fid, 1, "*uint64")
        self.check_read_error(fid, "num pixels")
        n_pix = int(n_pix_arr[0])
        if n_pix > 0:
            raw = do_fread(fid, n_rows * n_pix, "*float32")
            pix_arr = raw.reshape((n_rows, n_pix), order="F")
        else:
            pix_arr = np.zeros((n_rows, 0), dtype=np.float32)
        return pix_data(pix_arr)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------

    def saveable_fields(self) -> List[str]:
        return ["position", "npixels", "n_rows"]

    def from_bare_struct(self, s: dict) -> "pix_data_block":
        self._sqw_prop_name = "pix"
        self._level2_prop_name = "data_wrap"
        self._locked = True
        if "position" in s:
            self.position = s["position"]
        if "n_rows" in s:
            self.n_rows = s["n_rows"]
        if "npixels" in s:
            self.npixels = s["npixels"]
        elif "size" in s:
            self.size = s["size"]
        return self
