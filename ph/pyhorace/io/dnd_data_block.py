"""
``dnd_data_block`` — binary block descriptor for ``dnd_data`` in v4 Horace files.

Ported from ``horace_core/sqw/file_io/class_helpers/@dnd_data_block``.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.io.data_block import data_block
from pyhorace.io.sqw_fopen import do_fread, do_fwrite
from pyhorace.sqw.dnd_data import dnd_data


@register_serializable
class dnd_data_block(data_block):
    """Binary block descriptor for ``dnd_data`` arrays (``sig``, ``err``, ``npix``).

    Binary layout at ``position``:
    - ``uint32``: number of dimensions (``len(data_size)``, always >= 2)
    - ``uint32[n_dims]``: ``data_size`` dimensions
    - ``float64[prod(data_size)]``: ``sig`` (Fortran order)
    - ``float64[prod(data_size)]``: ``err`` (Fortran order)
    - ``uint64[prod(data_size)]``: ``npix`` (Fortran order)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._dimensions: Optional[int] = None
        self._data_size: Optional[Tuple[int, ...]] = None
        self._sqw_prop_name = "data"
        self._level2_prop_name = "nd_data"

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)
            return

        if len(args) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict):
            self.from_bare_struct(args[0])
            return

        if len(args) == 1 and isinstance(args[0], dnd_data_block):
            other = args[0]
            self._position = other._position
            self._size = other._size
            self._dimensions = other._dimensions
            self._data_size = other._data_size
            return

        fld = self.saveable_fields()
        if len(args) == 4:
            # Called with (sqw_prop_name, level2_prop_name, position, size)
            self.sqw_prop_name = args[0]
            self.level2_prop_name = args[1]
            self.position = args[2]
            self.size = args[3]
            return
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
    def dnd_info_defined(self) -> bool:
        return self._dimensions is not None and self._data_size is not None

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, val: Optional[int]) -> None:
        self._dimensions = int(val) if val is not None else None

    @property
    def sig_position(self) -> int:
        if not self.dnd_info_defined:
            return 0
        return int(self.position + 4 + self._dimensions * 4)

    @property
    def err_position(self) -> int:
        if not self.dnd_info_defined:
            return 0
        return int(self.sig_position + 8 * self.data_size)

    @property
    def npix_position(self) -> int:
        if not self.dnd_info_defined:
            return 0
        return int(self.sig_position + 2 * 8 * self.data_size)

    @property
    def data_size(self) -> int:
        if not self.dnd_info_defined:
            return 0
        return int(np.prod(self._data_size, dtype=np.int64))

    # ------------------------------------------------------------------
    # Size calculation and file I/O
    # ------------------------------------------------------------------

    def calc_obj_size(
        self,
        sqw_obj: Any,
        nocache: bool = False,
        check_size: bool = False,
    ) -> "dnd_data_block":
        subobj = self.get_subobj(sqw_obj)
        if not isinstance(subobj, dnd_data):
            if hasattr(subobj, "nd_data"):
                subobj = subobj.nd_data
            elif isinstance(subobj, dict) and ("s" in subobj or "sig" in subobj):
                s = subobj.get("sig", subobj.get("s"))
                e = subobj.get("err", subobj.get("e"))
                npix = subobj.get("npix")
                subobj = dnd_data(s, e, npix)
            else:
                raise HoraceError(
                    "HORACE:dnd_data_block:invalid_argument",
                    f"Expected dnd_data instance, got {type(subobj).__name__}.",
                )
        dsize = tuple(int(x) for x in subobj.data_size)
        if len(dsize) < 2:
            dsize = dsize + (1,) * (2 - len(dsize))
        self._data_size = dsize
        self._dimensions = len(dsize)
        n_el = int(np.prod(dsize, dtype=np.int64))
        sz = int(4 + self._dimensions * 4 + (3 * 8) * n_el)

        if check_size and self.size != sz:
            raise HoraceError(
                "HORACE:data_block:runtime_error",
                f"Block {self.block_name}: expected size {self.size}, actual size {sz}.",
            )
        self._size = sz
        self._serialized_obj_cache = None if nocache else subobj
        return self

    def put_bindata_in_file(self, fid: int, obj_data: Any) -> None:
        if not isinstance(obj_data, dnd_data):
            raise HoraceError(
                "HORACE:dnd_data_block:invalid_argument",
                f"Expected dnd_data instance, got {type(obj_data).__name__}.",
            )
        self.move_to_position(fid)
        dsize = tuple(int(x) for x in obj_data.data_size)
        if len(dsize) < 2:
            dsize = dsize + (1,) * (2 - len(dsize))
        header = np.array([len(dsize), *dsize], dtype=np.uint32)
        do_fwrite(fid, header, "uint32")
        self.check_write_error(fid, "writing dimensions")

        do_fwrite(fid, np.asarray(obj_data.sig, dtype=np.float64).ravel(order="F"), "float64")
        self.check_write_error(fid, "writing signal")

        do_fwrite(fid, np.asarray(obj_data.err, dtype=np.float64).ravel(order="F"), "float64")
        self.check_write_error(fid, "writing error")

        do_fwrite(fid, np.asarray(obj_data.npix, dtype=np.uint64).ravel(order="F"), "uint64")
        self.check_write_error(fid, "writing npix")

    def get_bindata_from_file(self, fid: int) -> dnd_data:
        self.move_to_position(fid)
        n_dims_arr = do_fread(fid, 1, "*uint32")
        self.check_read_error(fid, "reading number of dimensions")
        n_dims = int(n_dims_arr[0])

        dims_arr = do_fread(fid, n_dims, "*uint32")
        self.check_read_error(fid, "reading dimensions array")
        dsize = tuple(int(x) for x in dims_arr)
        self._dimensions = n_dims
        self._data_size = dsize

        n_el = int(np.prod(dsize, dtype=np.int64))
        sig_flat = do_fread(fid, n_el, "*float64")
        self.check_read_error(fid, "reading signal")

        err_flat = do_fread(fid, n_el, "*float64")
        self.check_read_error(fid, "reading error")

        npix_flat = do_fread(fid, n_el, "*uint64")
        self.check_read_error(fid, "reading npix")

        sig = sig_flat.reshape(dsize, order="F")
        err = err_flat.reshape(dsize, order="F")
        npix = npix_flat.reshape(dsize, order="F")
        return dnd_data(sig, err, npix)

    def read_dnd_info(self, fid: int) -> "dnd_data_block":
        """Read the dimension header at ``self.position`` to populate ``npix_position`` etc."""
        self.move_to_position(fid)
        n_dims_arr = do_fread(fid, 1, "*uint32")
        n_dims = int(n_dims_arr[0])
        dims_arr = do_fread(fid, n_dims, "*uint32")
        self.check_read_error(fid, "header")
        self._dimensions = n_dims
        self._data_size = tuple(int(x) for x in dims_arr)
        return self

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------

    def saveable_fields(self) -> List[str]:
        return ["position", "size"]

    def from_bare_struct(self, s: dict) -> "dnd_data_block":
        self._sqw_prop_name = "data"
        self._level2_prop_name = "nd_data"
        if "position" in s:
            self.position = s["position"]
        if "size" in s:
            self.size = s["size"]
        return self
