"""Binary .sqw / .dnd reader and lightweight sqw wrapper for DnD operations."""

from __future__ import annotations

import copy
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from pyhorace.coord_transf._dependencies import HoraceError, PixelDataMemory
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.data_op_interface import data_op_interface, matlab_size
from pyhorace.core.serializable import Serializable
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw.data_sqw_dnd import data_sqw_dnd
from pyhorace.sqw.dnd_base import DnDBase
from pyhorace.sqw.sqw_dnd_base import SQWDnDBase


def _read_string(f) -> str:
    raw = f.read(4)
    if len(raw) < 4:
        raise EOFError("Unexpected EOF reading string length")
    (n,) = struct.unpack("<i", raw)
    if n <= 0:
        return ""
    data = f.read(n)
    return data.decode("latin1", errors="replace")


def _read_binary_sqw_dnd(path: str, read_pixels: bool = False) -> Tuple[DnDBase, Optional[PixelDataMemory], int]:
    """Read a Horace v2 / v3 binary .sqw or .dnd file and return (dnd_obj, pix_obj, sqw_type)."""
    with open(path, "rb") as f:
        raw4 = f.read(4)
        if len(raw4) < 4:
            raise HoraceError("HORACE:read_dnd:io_error", f"File is too short: {path}")
        (napp,) = struct.unpack("<i", raw4)
        if napp != 6:
            raise HoraceError("HORACE:read_dnd:io_error", f"Not a valid Horace binary file: {path}")
        app = f.read(napp).decode("latin1", errors="replace")
        if app != "horace":
            raise HoraceError("HORACE:read_dnd:io_error", f"Not a valid Horace binary file: {path}")

        ver, sqw_type, _ndims = struct.unpack("<dii", f.read(16))

        if sqw_type == 1:
            # Main header: filename, filepath, title, nfiles
            _read_string(f)
            _read_string(f)
            _read_string(f)
            (nfiles,) = struct.unpack("<i", f.read(4))
            for _ in range(nfiles):
                _read_string(f)  # filename
                _read_string(f)  # filepath
                # efix(1f), emode(1i), alatt(3f), angdeg(3f), cu(3f), cv(3f), psi,omega,dpsi,gl,gs(5f) -> 19*4 = 76 bytes
                f.read(76)
                (nen,) = struct.unpack("<i", f.read(4))
                f.read(4 * nen)
                # uoffset(4f), u_to_rlu(16f), ulen(4f) -> 24*4 = 96 bytes
                f.read(96)
                r, c = struct.unpack("<ii", f.read(8))
                if r * c > 0:
                    f.read(r * c)
            # detpar: filename, filepath, ndet, 6*ndet floats
            _read_string(f)
            _read_string(f)
            (ndet,) = struct.unpack("<i", f.read(4))
            if ndet > 0:
                f.read(24 * ndet)

        # Data (DnD) block
        fn = _read_string(f)
        fp = _read_string(f)
        title = _read_string(f)
        alatt = np.frombuffer(f.read(12), dtype="<f4").astype(float)
        angdeg = np.frombuffer(f.read(12), dtype="<f4").astype(float)
        uoffset = np.frombuffer(f.read(16), dtype="<f4").astype(float)
        u_to_rlu = np.frombuffer(f.read(64), dtype="<f4").astype(float).reshape((4, 4), order="F")
        ulen = np.frombuffer(f.read(16), dtype="<f4").astype(float)
        lr, lc = struct.unpack("<ii", f.read(8))
        if lr * lc > 0:
            raw_lbl = f.read(lr * lc)
            arr_lbl = np.frombuffer(raw_lbl, dtype="S1").reshape((lr, lc), order="F")
            labels = [b"".join(arr_lbl[i, :]).decode("latin1").rstrip() for i in range(lr)]
        else:
            labels = ["h", "k", "l", "en"]

        (npax,) = struct.unpack("<i", f.read(4))
        niax = 4 - npax
        iax = np.frombuffer(f.read(4 * niax), dtype="<i4").astype(int) if niax > 0 else np.array([], dtype=int)
        iint = (
            np.frombuffer(f.read(8 * niax), dtype="<f4").astype(float).reshape((2, niax), order="F")
            if niax > 0
            else np.zeros((2, 0), dtype=float)
        )
        pax = np.frombuffer(f.read(4 * npax), dtype="<i4").astype(int) if npax > 0 else np.array([], dtype=int)
        p: List[np.ndarray] = []
        dims: List[int] = []
        for _ in range(npax):
            (npi,) = struct.unpack("<i", f.read(4))
            pi = np.frombuffer(f.read(4 * npi), dtype="<f4").astype(float)
            p.append(pi)
            dims.append(npi - 1)
        dax = np.frombuffer(f.read(4 * npax), dtype="<i4").astype(int) if npax > 0 else np.array([], dtype=int)

        if npax == 0:
            sz: Tuple[int, ...] = (1, 1)
        elif npax == 1:
            sz = (dims[0], 1)
        else:
            sz = tuple(dims)
        nelem = int(np.prod(sz))
        s = np.frombuffer(f.read(4 * nelem), dtype="<f4").astype(float).reshape(sz, order="F")
        e = np.frombuffer(f.read(4 * nelem), dtype="<f4").astype(float).reshape(sz, order="F")
        npix = np.frombuffer(f.read(8 * nelem), dtype="<u8").astype(float).reshape(sz, order="F")

        img_range: Optional[np.ndarray] = None
        pix_obj: Optional[PixelDataMemory] = None
        if sqw_type == 1:
            rem_range = f.read(32)
            if len(rem_range) == 32:
                ur = np.frombuffer(rem_range, dtype="<f4").astype(float).reshape((2, 4), order="F")
                if not np.any(np.isinf(ur)) and not np.any(np.isnan(ur)):
                    img_range = ur
            if read_pixels:
                rem_npix = f.read(8)
                if len(rem_npix) == 8:
                    (num_pixels,) = struct.unpack("<Q", rem_npix)
                    raw_pix = f.read(36 * num_pixels)
                    if len(raw_pix) == 36 * num_pixels:
                        pix_arr = np.frombuffer(raw_pix, dtype="<f4").astype(float).reshape((9, num_pixels), order="F")
                        pix_obj = PixelDataMemory(pix_arr)

        abs_dir, abs_name = os.path.split(os.path.abspath(path))
        old_data: Dict[str, Any] = {
            "filename": abs_name if abs_name else fn,
            "filepath": abs_dir if abs_dir else fp,
            "title": title,
            "alatt": alatt,
            "angdeg": angdeg,
            "uoffset": uoffset,
            "u_to_rlu": u_to_rlu,
            "ulen": ulen,
            "label": labels,
            "iax": iax,
            "iint": iint,
            "pax": pax,
            "p": p,
            "dax": dax,
        }
        if img_range is not None:
            old_data["img_range"] = img_range

        proj = line_proj.get_from_old_data(old_data)
        ax = line_axes.get_from_old_data(old_data)

        dnd_obj = DnDBase.dnd(ax, proj, s, e, npix)
        return dnd_obj, pix_obj, sqw_type


def read_dnd(file: str) -> Any:
    """Read DnD data from a Horace .sqw / .dnd binary file or .mat file."""
    if isinstance(file, (list, tuple)):
        from pyhorace.core.data_op_interface import DataOpArray

        return DataOpArray([read_dnd(f) for f in file])

    if not isinstance(file, (str, os.PathLike)):
        raise HoraceError("HORACE:read_dnd:invalid_argument", "Filename must be a string")
    path = str(file)
    if not os.path.isfile(path):
        raise HoraceError("HORACE:read_dnd:invalid_argument", f"File does not exist: {path}")

    with open(path, "rb") as f:
        header = f.read(16)

    if len(header) >= 10 and header[:4] == b"\x06\x00\x00\x00" and header[4:10] == b"horace":
        dnd_obj, _, _ = _read_binary_sqw_dnd(path, read_pixels=False)
        return dnd_obj

    # Try loading as MATLAB .mat file
    try:
        mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        for k, v in mat.items():
            if not k.startswith("__"):
                if isinstance(v, dict) and "serial_name" in v:
                    return Serializable.from_struct(v)
                try:
                    return Serializable.from_struct(v)
                except Exception:
                    dsd = data_sqw_dnd(v)
                    return DnDBase.dnd(dsd)
    except Exception as exc:
        raise HoraceError("HORACE:read_dnd:io_error", f"Could not read DnD from {path}: {exc}") from exc

    raise HoraceError("HORACE:read_dnd:io_error", f"No DnD object found in {path}")


class sqw(SQWDnDBase):
    """Lightweight SQW class wrapper for loading .sqw files and supporting DnD conversion and binary operations."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Serializable.__init__(self)
        self._data: DnDBase = DnDBase.dnd()
        self._pix: PixelDataMemory = PixelDataMemory()
        self._main_header: Dict[str, Any] = {}
        self._experiment_info: Dict[str, Any] = {}
        self._detpar: Dict[str, Any] = {}

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (str, os.PathLike)):
                path = str(arg)
                if not os.path.isfile(path):
                    raise HoraceError("HORACE:sqw:invalid_argument", f"File does not exist: {path}")
                dnd_obj, pix_obj, _ = _read_binary_sqw_dnd(path, read_pixels=False)
                self._data = dnd_obj
                if pix_obj is not None:
                    self._pix = pix_obj
                return
            if isinstance(arg, sqw):
                self._data = arg._data.copy()
                self._pix = copy.deepcopy(arg._pix)
                return
            if isinstance(arg, DnDBase):
                self._data = arg.copy()
                return
            if isinstance(arg, dict):
                if "data" in arg:
                    d = arg["data"]
                    self._data = DnDBase.dnd(d) if not isinstance(d, DnDBase) else d.copy()
                    if "pix" in arg.get("data", {}):
                        self._pix = PixelDataMemory(arg["data"]["pix"])
                    return

    @property
    def data(self) -> DnDBase:
        return self._data

    @data.setter
    def data(self, val: DnDBase) -> None:
        self._data = DnDBase.dnd(val) if not isinstance(val, DnDBase) else val

    @property
    def pix(self) -> PixelDataMemory:
        return self._pix

    @property
    def axes(self):
        return self._data.axes

    @property
    def proj(self):
        return self._data.proj

    @property
    def dimensions(self) -> int:
        return self._data.dimensions

    def dnd(self) -> DnDBase:
        return self._data.copy()

    def get_NUM_DIMS(self) -> int:
        return int(self._data.dimensions)

    def get_img_size(self) -> Tuple[int, ...]:
        return self.sigvar_size()

    def has_pixels(self) -> bool:
        return self._pix.num_pixels > 0

    def sigvar(self) -> sigvar:
        return self._data.sigvar()

    def sigvar_get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._data.sigvar_get()

    def sigvar_set(self, sigvar_obj: Any) -> "sqw":
        self._data.sigvar_set(sigvar_obj)
        return self

    def sigvar_size(self) -> Tuple[int, ...]:
        return self._data.sigvar_size()

    def classVersion(self) -> int:
        return 4

    def saveableFields(self) -> List[str]:
        return ["data"]


def read_sqw(file: str) -> Any:
    """Read an SQW or DnD object from a Horace binary file."""
    if isinstance(file, (list, tuple)):
        from pyhorace.core.data_op_interface import DataOpArray

        return DataOpArray([read_sqw(f) for f in file])
    dnd_obj, pix_obj, sqw_type = _read_binary_sqw_dnd(str(file), read_pixels=False)
    if sqw_type == 1:
        obj = sqw(dnd_obj)
        return obj
    return dnd_obj


def read_horace(file: str) -> Any:
    """Read an SQW or DnD object from a Horace file."""
    return read_sqw(file)

