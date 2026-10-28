"""Binary .sqw / .dnd reader functions using ``sqw_formats_factory``."""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import scipy.io

from pyhorace.coord_transf._dependencies import HoraceError, PixelDataMemory
from pyhorace.core.serializable import Serializable
from pyhorace.data.experiment import Experiment
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.sqw.data_sqw_dnd import data_sqw_dnd
from pyhorace.sqw.dnd_base import DnDBase
from pyhorace.sqw.sqw import sqw


def _read_binary_sqw_dnd(
    path: str, read_pixels: bool = False
) -> Tuple[DnDBase, Optional[PixelDataMemory], int]:
    """Read a Horace binary .sqw or .dnd file via ``sqw_formats_factory``
    and return ``(dnd_obj, pix_obj, sqw_type)``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

    ldr = sqw_formats_factory.instance().get_loader(str(path))
    try:
        sqw_type = 1 if bool(ldr.sqw_type) else 0
        dnd_obj = ldr.get_dnd()
        pix_obj: Optional[PixelDataMemory] = None
        if sqw_type == 1 and read_pixels:
            pix_obj = PixelDataMemory(ldr)
        return dnd_obj, pix_obj, sqw_type
    finally:
        ldr.fclose()


def read_dnd(file: Any, *args: Any) -> Any:
    """Read DnD data from a Horace .sqw / .dnd binary file (via ``sqw_formats_factory``) or .mat file."""
    if isinstance(file, (list, tuple)):
        from pyhorace.core.data_op_interface import DataOpArray

        return DataOpArray([read_dnd(f, *args) for f in file])

    if not isinstance(file, (str, os.PathLike)):
        raise HoraceError("HORACE:read_dnd:invalid_argument", "Filename must be a string")
    path = str(file)
    if not os.path.isfile(path):
        raise HoraceError("HORACE:read_dnd:invalid_argument", f"File does not exist: {path}")

    try:
        from pyhorace.io.sqw_formats_factory import sqw_formats_factory

        ldr = sqw_formats_factory.instance().get_loader(path)
        try:
            return ldr.get_dnd(*args)
        finally:
            ldr.fclose()
    except Exception:
        pass

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
    """Read a Horace binary .sqw or .dnd file via ``sqw_formats_factory`` and return
    ``(main_header, exper_info, dnd_obj, pix_obj, sqw_type)``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

    ldr = sqw_formats_factory.instance().get_loader(str(path))
    try:
        sqw_type = 1 if bool(ldr.sqw_type) else 0
        if sqw_type == 0:
            dnd_obj = ldr.get_dnd()
            return main_header_cl(), Experiment(), dnd_obj, None, 0
        sqw_obj = sqw(ldr, file_backed=file_backed, pixel_page_size=pixel_page_size)
        return sqw_obj.main_header, sqw_obj.exper_info, sqw_obj.data, sqw_obj.pix, 1
    finally:
        ldr.fclose()


def read_horace(file: Any, *args: Any, file_backed: Optional[bool] = None) -> Any:
    """Read an SQW or DnD object (or list/DataOpArray of objects) from Horace binary file(s)."""
    get_dnd = False
    force_sqw = False
    rem_args: List[Any] = []
    for a in args:
        if isinstance(a, str):
            al = a.strip().lower()
            if al == "-get_dnd":
                get_dnd = True
            elif al == "-force_sqw":
                force_sqw = True
            elif al in ("-file_backed", "-filebacked"):
                file_backed = True
            else:
                rem_args.append(a)
        else:
            rem_args.append(a)

    if get_dnd and force_sqw:
        raise HoraceError(
            "HORACE:read_horace:invalid_argument",
            "only one option allowed i.e. -get_dnd and -force_sqw cannot be provided simultaneously",
        )

    if isinstance(file, (list, tuple)):
        from pyhorace.core.data_op_interface import DataOpArray

        items = [read_horace(f, *args, file_backed=file_backed) for f in file]
        if len(items) > 0 and all(type(x) is type(items[0]) for x in items):
            return DataOpArray(items)
        return items

    path = str(file)
    if not os.path.isfile(path):
        raise HoraceError("HORACE:read_horace:invalid_argument", f"File does not exist: {path}")

    from pyhorace.io.sqw_formats_factory import sqw_formats_factory

    ldr = sqw_formats_factory.instance().get_loader(path)
    try:
        is_sqw = bool(ldr.sqw_type)
        if force_sqw and not is_sqw:
            raise HoraceError(
                "HORACE:read_horace:invalid_argument",
                f"Files {path} only contain dnd information but sqw file required.",
            )
        if get_dnd or not is_sqw:
            return ldr.get_dnd(*rem_args)
        fb_args = list(rem_args)
        if file_backed is not None:
            fb_args.extend(["-file_backed"] if file_backed else [])
        return ldr.get_sqw(*fb_args)
    finally:
        ldr.fclose()


def read_sqw(file: Any, *args: Any, file_backed: Optional[bool] = None) -> Any:
    """Read an SQW object from a Horace binary file (raises if file contains only DnD)."""
    return read_horace(file, *args, "-force_sqw", file_backed=file_backed)

