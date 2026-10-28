"""Standalone and file-based cut algorithms (cut, cut_sqw, cut_dnd, cut_horace, cut_sym).

Ported from Horace MATLAB:
- ``horace_core/algorithms/cut.m``
- ``horace_core/algorithms/private/parse_cut_inputs_.m``
- ``horace_core/algorithms/private/pack_output_.m``
- ``horace_core/algorithms/cut_sqw.m``
- ``horace_core/algorithms/cut_dnd.m``
- ``horace_core/algorithms/cut_horace.m``
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.sqw.dnd_base import DnDBase
from pyhorace.sqw.sqw import sqw, symmetrise_sqw
from pyhorace.sqw.sqw_dnd_base import SQWDnDBase


def _is_filename_arg(arg: Any) -> bool:
    """Return True if ``arg`` is a non-flag filename string/path or list/tuple of them."""
    if isinstance(arg, (str, os.PathLike)):
        s = str(arg).strip()
        return bool(s) and not s.startswith("-")
    if isinstance(arg, (list, tuple)) and len(arg) > 0:
        return all(
            isinstance(x, (str, os.PathLike))
            and bool(str(x).strip())
            and not str(x).strip().startswith("-")
            for x in arg
        )
    return False


def _parse_cut_inputs(
    nin: int, nout: int, *argi: Any
) -> Tuple[int, int, bool, List[Tuple[Any, ...]]]:
    """Validate input/output counts and expand output filenames for ``cut``.

    Ported from ``horace_core/algorithms/private/parse_cut_inputs_.m``.
    Returns ``(nin, nout, fn_present, argi_per_source)``.
    """
    cleaned_args: List[Any] = []
    idx = 0
    while idx < len(argi):
        a = argi[idx]
        if isinstance(a, str) and a.lower() == "-nargout":
            if idx + 1 >= len(argi):
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "Option '-nargout' requires an integer value",
                )
            nout = int(argi[idx + 1])
            idx += 2
            continue
        cleaned_args.append(a)
        idx += 1

    fn_indices = [i for i, a in enumerate(cleaned_args) if _is_filename_arg(a)]
    fn_present = len(fn_indices) > 0

    if not fn_present and nout == 0:
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            "cut(s) without output objects are requested but no file(s) to save their results are provided",
        )

    filenames: List[str] = []
    fn_idx = -1
    if fn_present:
        fn_idx = fn_indices[-1]
        raw_fn = cleaned_args[fn_idx]
        if isinstance(raw_fn, (str, os.PathLike)):
            filenames = [str(raw_fn).strip()]
        else:
            filenames = [str(x).strip() for x in raw_fn]

        if len(filenames) == 1 and nin > 1:
            root, ext = os.path.splitext(filenames[0])
            filenames = [f"{root}_cutN{i + 1}{ext}" for i in range(nin)]
        elif len(filenames) < nin:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Multiple cuts with output to file are requested, but number of output files is smaller then number of cuts to do",
            )

    if nout > 1:
        if nout > nin:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                f"Number of input cut sources ({nin}) is smaller than the number of requested outputs ({nout})",
            )
        nin = nout

    argi_per_source: List[Tuple[Any, ...]] = []
    for i in range(nin):
        cur = list(cleaned_args)
        if fn_present:
            cur[fn_idx] = filenames[i]
        argi_per_source.append(tuple(cur))

    return nin, nout, fn_present, argi_per_source


def _pack_output(
    results: List[Any],
    nin: int,
    nout: int,
    cell_output: bool,
    source_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Pack cut results from one or more sources (port of ``pack_output_.m``)."""
    if nout == 0:
        return None

    if cell_output:
        if nout == 1:
            return list(results)
        return tuple([results[i]] for i in range(nout))

    if nout == 1:
        if nin == 1:
            return results[0]
        flat_objs: List[Any] = []
        for r in results:
            if isinstance(r, (DataOpArray, list, tuple)):
                flat_objs.extend(r)
            else:
                flat_objs.append(r)
        types = {type(x) for x in flat_objs}
        if len(types) > 1:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Cannot combine cuts of different types into a single array; use '-cell_output' option.",
            )
        shape = source_shape if len(flat_objs) == len(results) else None
        return DataOpArray.from_list(flat_objs, shape=shape)

    return tuple(results[:nout])


def cut(*args: Any, **kwargs: Any) -> Any:
    """Take a cut from an ``sqw`` or ``DnDBase`` object, file, or sequence of them.

    Ported from ``horace_core/algorithms/cut.m``.
    """
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)

    dnd_only = False
    sqw_only = False
    cell_output = False
    filtered_args: List[Any] = []
    for a in args:
        if isinstance(a, str) and a.startswith("-"):
            a_low = a.lower()
            if "-dnd_only".startswith(a_low) and len(a_low) >= 4:
                dnd_only = True
                continue
            if "-sqw_only".startswith(a_low) and len(a_low) >= 4:
                sqw_only = True
                continue
            if "-cell_output".startswith(a_low) and len(a_low) >= 4:
                cell_output = True
                continue
        filtered_args.append(a)

    if dnd_only and sqw_only:
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            "can not request only sqw cut and only dnd cut simultaneously",
        )

    if len(filtered_args) == 0:
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            "Call to cut must include a file or sqw/dnd object to take a cut from.",
        )

    source = filtered_args[0]
    rest_args = filtered_args[1:]

    if isinstance(source, (str, os.PathLike)):
        s_src = str(source).strip()
        if not s_src:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Source filename cannot be empty.",
            )
        raw_items: List[Any] = [s_src]
        is_files = True
        source_shape: Optional[Tuple[int, ...]] = None
    elif (
        isinstance(source, (list, tuple))
        and len(source) > 0
        and all(isinstance(x, (str, os.PathLike)) for x in source)
    ):
        raw_items = [str(x).strip() for x in source]
        if any(not s for s in raw_items):
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Source filename in cell/list cannot be empty.",
            )
        is_files = True
        source_shape = (1, len(raw_items)) if len(raw_items) > 1 else None
    elif isinstance(source, SQWDnDBase):
        raw_items = [source]
        is_files = False
        source_shape = None
    elif isinstance(source, (DataOpArray, np.ndarray)):
        if source.size == 0:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Cannot take cut from an empty array.",
            )
        raw_items = list(source.ravel(order="F"))
        if not all(isinstance(x, SQWDnDBase) for x in raw_items):
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Input array must contain only SQW or DnD objects.",
            )
        is_files = False
        source_shape = source.shape if source.size > 1 else None
    elif isinstance(source, (list, tuple)):
        if len(source) == 0:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Cannot take cut from an empty list/cell.",
            )
        if not all(isinstance(x, SQWDnDBase) for x in source):
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Input list/cell must contain only filenames or SQW/DnD objects.",
            )
        raw_items = list(source)
        is_files = False
        source_shape = (1, len(raw_items)) if len(raw_items) > 1 else None
    else:
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            f"Cannot take cut from {type(source).__name__} object.",
        )

    nin, nout, _, argi_per_source = _parse_cut_inputs(
        len(raw_items), int(nargout), *rest_args
    )
    return_cut = bool(nout > 0)

    results: List[Any] = []
    for i in range(nin):
        item = raw_items[i]
        cur_args = argi_per_source[i]
        was_loaded = False
        obj: SQWDnDBase
        if is_files:
            from pyhorace.io.sqw_formats_factory import sqw_formats_factory

            fname = str(item)
            if not os.path.isfile(fname):
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    f"Source file '{fname}' does not exist.",
                )
            ldr = sqw_formats_factory.instance().get_loader(fname)
            if ldr.sqw_type:
                if dnd_only:
                    ldr.delete()
                    raise HoraceError(
                        "HORACE:cut:invalid_argument",
                        f"cut_dnd requested on sqw file '{fname}'",
                    )
                obj = sqw(ldr, file_backed=True)
                was_loaded = True
            else:
                try:
                    obj = ldr.get_dnd()
                finally:
                    ldr.delete()
                if sqw_only:
                    raise HoraceError(
                        "HORACE:cut:invalid_argument",
                        f"cut_sqw requested on dnd file '{fname}'",
                    )
        else:
            obj = item
            if sqw_only and (
                isinstance(obj, DnDBase)
                or (isinstance(obj, sqw) and bool(obj.dnd_type))
            ):
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "cut_sqw requested on dnd object",
                )
            if dnd_only and isinstance(obj, sqw) and not bool(obj.dnd_type):
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "cut_dnd requested on sqw object",
                )

        try:
            res = obj.cut(*cur_args, nargout=1 if return_cut else 0, **kwargs)
            if return_cut:
                results.append(res)
        finally:
            if was_loaded and isinstance(obj, sqw):
                try:
                    obj.pix.deactivate()
                except Exception:
                    pass

    return _pack_output(results, nin, nout, cell_output, source_shape)


def cut_sqw(source: Any, *args: Any, **kwargs: Any) -> Any:
    """Take a cut from an ``sqw`` object or file (port of ``horace_core/algorithms/cut_sqw.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return cut(source, "-sqw_only", *args, nargout=nargout, **kwargs)


def cut_dnd(source: Any, *args: Any, **kwargs: Any) -> Any:
    """Take a cut from a ``DnDBase`` object or file (port of ``horace_core/algorithms/cut_dnd.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return cut(source, "-dnd_only", *args, nargout=nargout, **kwargs)


def cut_horace(*args: Any, **kwargs: Any) -> Any:
    """Legacy alias for :func:`cut` (port of ``horace_core/algorithms/cut_horace.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return cut(*args, nargout=nargout, **kwargs)


def cut_sym(*args: Any, **kwargs: Any) -> Any:
    """Take a symmetry-enabled cut from an ``sqw`` or ``DnDBase`` object or file."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return cut(*args, nargout=nargout, **kwargs)


__all__ = [
    "_pack_output",
    "_parse_cut_inputs",
    "cut",
    "cut_dnd",
    "cut_horace",
    "cut_sqw",
    "cut_sym",
    "symmetrise_sqw",
]
