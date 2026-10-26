"""Binary .sqw / .dnd reader and lightweight sqw wrapper for DnD operations."""

from __future__ import annotations

import copy
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.io

from pyhorace.coord_transf._dependencies import HoraceError, PixelDataMemory, fast_map
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.data_op_interface import data_op_interface, matlab_size
from pyhorace.core.serializable import Serializable
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.sigvar import sigvar
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.sqw.data_sqw_dnd import data_sqw_dnd
from pyhorace.sqw.dnd_base import DnDBase
from pyhorace.sqw.sqw import sqw
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
                hdr12 = f.read(12)
                if len(hdr12) == 12:
                    _, num_pixels = struct.unpack("<iQ", hdr12)
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
            "img_db_range": img_range,
        }

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


def _read_binary_sqw_full(
    path: str,
    file_backed: Optional[bool] = None,
    pixel_page_size: Optional[int] = None,
) -> Tuple[main_header_cl, Experiment, DnDBase, Optional[PixelDataBase], int]:
    """Read a Horace v2 / v3 binary .sqw or .dnd file and return
    ``(main_header, exper_info, dnd_obj, pix_obj, sqw_type)``.
    """
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

        mh = main_header_cl()
        exp_info = Experiment()

        if sqw_type == 1:
            mh_fn = _read_string(f)
            mh_fp = _read_string(f)
            mh_title = _read_string(f)
            (nfiles,) = struct.unpack("<i", f.read(4))
            mh = main_header_cl(
                {
                    "filename": mh_fn,
                    "filepath": mh_fp,
                    "title": mh_title,
                    "nfiles": int(nfiles),
                }
            )
            hdrs: List[Dict[str, Any]] = []
            for _ in range(nfiles):
                exp_fn = _read_string(f)
                exp_fp = _read_string(f)
                efix = float(struct.unpack("<f", f.read(4))[0])
                emode = int(struct.unpack("<i", f.read(4))[0])
                exp_alatt = np.frombuffer(f.read(12), dtype="<f4").astype(float)
                exp_angdeg = np.frombuffer(f.read(12), dtype="<f4").astype(float)
                cu = np.frombuffer(f.read(12), dtype="<f4").astype(float)
                cv = np.frombuffer(f.read(12), dtype="<f4").astype(float)
                psi, omega, dpsi, gl, gs = [
                    float(x) for x in struct.unpack("<5f", f.read(20))
                ]
                (nen,) = struct.unpack("<i", f.read(4))
                en = np.frombuffer(f.read(4 * nen), dtype="<f4").astype(float)
                exp_uoffset = np.frombuffer(f.read(16), dtype="<f4").astype(float)
                exp_u_to_rlu = (
                    np.frombuffer(f.read(64), dtype="<f4")
                    .astype(float)
                    .reshape((4, 4), order="F")
                )
                exp_ulen = np.frombuffer(f.read(16), dtype="<f4").astype(float)
                r, c = struct.unpack("<ii", f.read(8))
                if r * c > 0:
                    raw_ulbl = f.read(r * c)
                    arr_ulbl = np.frombuffer(raw_ulbl, dtype="S1").reshape(
                        (r, c), order="F"
                    )
                    ulabel = [
                        b"".join(arr_ulbl[i, :]).decode("latin1").rstrip()
                        for i in range(r)
                    ]
                else:
                    ulabel = ["h", "k", "l", "en"]
                hdrs.append(
                    {
                        "filename": exp_fn,
                        "filepath": exp_fp,
                        "efix": efix,
                        "emode": emode,
                        "alatt": exp_alatt,
                        "angdeg": exp_angdeg,
                        "cu": cu,
                        "cv": cv,
                        "psi": psi,
                        "omega": omega,
                        "dpsi": dpsi,
                        "gl": gl,
                        "gs": gs,
                        "en": en,
                        "uoffset": exp_uoffset,
                        "u_to_rlu": exp_u_to_rlu,
                        "ulen": exp_ulen,
                        "ulabel": ulabel,
                    }
                )
            if nfiles > 0:
                exp_info = Experiment.build_from_binfile_headers(hdrs)

            # detpar: filename, filepath, ndet, 6*ndet floats
            det_fn = _read_string(f)
            det_fp = _read_string(f)
            (ndet,) = struct.unpack("<i", f.read(4))
            if ndet > 0:
                group = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                x2 = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                phi = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                azim = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                width = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                height = np.frombuffer(f.read(4 * ndet), dtype="<f4").astype(float)
                det_arr = IX_detector_array(
                    {
                        "filename": det_fn,
                        "filepath": det_fp,
                        "group": group,
                        "x2": x2,
                        "phi": phi,
                        "azim": azim,
                        "width": width,
                        "height": height,
                    }
                )
                urc = unique_references_container("IX_detector_array")
                urc.add(det_arr)
                if exp_info.n_runs > 1:
                    urc = urc.replicate_runs(exp_info.n_runs)
                exp_info.detector_arrays = urc

        # Data (DnD) block
        fn = _read_string(f)
        fp = _read_string(f)
        title = _read_string(f)
        alatt = np.frombuffer(f.read(12), dtype="<f4").astype(float)
        angdeg = np.frombuffer(f.read(12), dtype="<f4").astype(float)
        uoffset = np.frombuffer(f.read(16), dtype="<f4").astype(float)
        u_to_rlu = (
            np.frombuffer(f.read(64), dtype="<f4")
            .astype(float)
            .reshape((4, 4), order="F")
        )
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
        iax = (
            np.frombuffer(f.read(4 * niax), dtype="<i4").astype(int)
            if niax > 0
            else np.array([], dtype=int)
        )
        iint = (
            np.frombuffer(f.read(8 * niax), dtype="<f4")
            .astype(float)
            .reshape((2, niax), order="F")
            if niax > 0
            else np.zeros((2, 0), dtype=float)
        )
        pax = (
            np.frombuffer(f.read(4 * npax), dtype="<i4").astype(int)
            if npax > 0
            else np.array([], dtype=int)
        )
        p: List[np.ndarray] = []
        dims: List[int] = []
        for _ in range(npax):
            (npi,) = struct.unpack("<i", f.read(4))
            pi = np.frombuffer(f.read(4 * npi), dtype="<f4").astype(float)
            p.append(pi)
            dims.append(npi - 1)
        dax = (
            np.frombuffer(f.read(4 * npax), dtype="<i4").astype(int)
            if npax > 0
            else np.array([], dtype=int)
        )

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
        pix_obj: Optional[PixelDataBase] = None
        if sqw_type == 1:
            rem_range = f.read(32)
            if len(rem_range) == 32:
                ur = np.frombuffer(rem_range, dtype="<f4").astype(float).reshape((2, 4), order="F")
                if not np.any(np.isinf(ur)) and not np.any(np.isnan(ur)):
                    img_range = ur
            hdr12 = f.read(12)
            if len(hdr12) == 12:
                _, num_pixels = struct.unpack("<iQ", hdr12)
                use_fb = (
                    bool(file_backed)
                    if file_backed is not None
                    else PixelDataBase.do_filebacked(int(num_pixels))
                )
                raw_pix = f.read(36 * num_pixels)
                if len(raw_pix) == 36 * num_pixels:
                    pix_arr = (
                        np.frombuffer(raw_pix, dtype="<f4")
                        .astype(float)
                        .reshape((9, num_pixels), order="F")
                    )
                    if use_fb:
                        fb_pix = PixelDataFileBacked(path)
                        fb_pix.old_file_format = not mh.creation_date_defined
                        if (
                            pixel_page_size is not None
                            and pixel_page_size < PixelDataBase.DEFAULT_PAGE_SIZE
                        ):
                            fb_pix._base_page_size = int(pixel_page_size)
                        pix_obj = fb_pix
                    else:
                        mem_pix = PixelDataMemory(pix_arr)
                        mem_pix.full_filename = path
                        if not mh.creation_date_defined:
                            mem_pix.old_file_format = True
                            mem_pix._unique_run_id = np.unique(pix_arr[4, :]).astype(float)
                        pix_obj = mem_pix

                    # Align run_ids with pixel run_idx if creation_date is not defined (v2 / early v3 files)
                    if (
                        not mh.creation_date_defined
                        and num_pixels > 0
                        and exp_info.n_runs > 0
                    ):
                        u_ids = np.unique(pix_arr[4, :]).astype(float)
                        if u_ids.size == exp_info.n_runs:
                            cur_keys = np.asarray(exp_info.runid_map.keys, dtype=float)
                            if not np.array_equal(cur_keys, u_ids):
                                exp_info.runid_map = fast_map(
                                    u_ids,
                                    np.arange(1, exp_info.n_runs + 1, dtype=int),
                                )
                                exp_info.runid_recalculated = False

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
            "img_db_range": img_range,
        }

        proj = line_proj.get_from_old_data(old_data)
        ax = line_axes.get_from_old_data(old_data)

        dnd_obj = DnDBase.dnd(ax, proj, s, e, npix)
        return mh, exp_info, dnd_obj, pix_obj, sqw_type


def read_sqw(file: str, file_backed: Optional[bool] = None) -> Any:
    """Read an SQW or DnD object from a Horace binary file."""
    if isinstance(file, (list, tuple)):
        from pyhorace.core.data_op_interface import DataOpArray

        return DataOpArray([read_sqw(f, file_backed=file_backed) for f in file])
    path = str(file)
    if not os.path.isfile(path):
        raise HoraceError("HORACE:read_sqw:invalid_argument", f"File does not exist: {path}")
    with open(path, "rb") as f:
        header = f.read(30)
    if len(header) >= 26 and header[:4] == b"\x06\x00\x00\x00" and header[4:10] == b"horace":
        _, sqw_type, _ = struct.unpack("<dii", header[10:26])
        if sqw_type == 1:
            if file_backed is not None:
                return sqw(path, file_backed=file_backed)
            return sqw(path)
        dnd_obj, _, _ = _read_binary_sqw_dnd(path, read_pixels=False)
        return dnd_obj
    return sqw(path, file_backed=file_backed) if file_backed is not None else sqw(path)


def read_horace(file: str, file_backed: Optional[bool] = None) -> Any:
    """Read an SQW or DnD object from a Horace file."""
    return read_sqw(file, file_backed=file_backed)

