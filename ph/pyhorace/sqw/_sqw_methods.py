"""Implementation of SQW and DnD cut and evaluation methods (cut, sqw_eval, func_eval, disp2sqw_eval).

Ported from Horace MATLAB:
- ``horace_core/sqw/@sqw/cut.m``
- ``horace_core/sqw/@sqw/cut_dnd.m``
- ``horace_core/sqw/@sqw/cut_sqw.m``
- ``horace_core/sqw/@sqw/cut_sym.m``
- ``horace_core/sqw/@sqw/private/cut_single_.m``
- ``horace_core/sqw/@sqw/private/cut_accumulate_data_.m``
- ``horace_core/sqw/@SQWDnDBase/sqw_eval.m``
- ``horace_core/sqw/@SQWDnDBase/sqw_eval_nopix.m``
- ``horace_core/sqw/@SQWDnDBase/disp2sqw_eval.m``
- ``horace_core/sqw/@SQWDnDBase/private/parse_eval_args_.m``
- ``horace_core/sqw/@sqw/sqw_eval_pix.m``
- ``horace_core/sqw/@sqw/func_eval.m``
- ``horace_core/sqw/@DnDBase/func_eval.m``
- ``horace_core/sqw/@DnDBase/private/sqw_eval_nopix_.m``
- ``horace_core/sqw/file_io/load_sqw_dnd.m``
- ``horace_core/algorithms/sqw_eval.m``
- ``horace_core/algorithms/func_eval.m``
- ``horace_core/algorithms/disp2sqw_eval.m``
"""

from __future__ import annotations

import copy
import inspect
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError, fast_map
from pyhorace.coord_transf.axes_block_base import normalize_signal
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.experiment import Experiment
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

if TYPE_CHECKING:
    from pyhorace.sqw.dnd_base import DnDBase
    from pyhorace.sqw.sqw import sqw
    from pyhorace.sqw.sqw_dnd_base import SQWDnDBase


def _invoke_eval_func(
    func: Callable[..., Any],
    *coord_arrays: Any,
    pars: Any = None,
) -> Any:
    """Invoke a user evaluation function with coordinate column arrays and parameters.

    Supports both unpacked parameter lists ``func(x1, ..., xn, *pars)`` (MATLAB ``pars{:}``)
    and single packed parameter vector ``func(x1, ..., xn, pars)``.
    """
    if pars is None:
        pars_list: List[Any] = []
    elif isinstance(pars, (list, tuple)):
        pars_list = list(pars)
    else:
        pars_list = [pars]

    n_coords = len(coord_arrays)
    try:
        sig = inspect.signature(func)
        pos_params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )
        if not has_varargs:
            if len(pos_params) == n_coords:
                return func(*coord_arrays)
            if len(pos_params) == n_coords + 1:
                if len(pars_list) == 0:
                    return func(*coord_arrays, [])
                if len(pars_list) == 1:
                    return func(*coord_arrays, pars_list[0])
                return func(*coord_arrays, pars_list)
            if len(pos_params) == n_coords + len(pars_list) and len(pars_list) > 1:
                return func(*coord_arrays, *pars_list)
    except (ValueError, TypeError):
        pass

    if len(pars_list) == 0:
        try:
            return func(*coord_arrays)
        except TypeError:
            return func(*coord_arrays, [])

    if len(pars_list) == 1:
        p0 = pars_list[0]
        if isinstance(p0, (list, tuple)) and len(p0) == 0:
            try:
                return func(*coord_arrays)
            except TypeError:
                return func(*coord_arrays, [])
        try:
            return func(*coord_arrays, p0)
        except TypeError:
            if isinstance(p0, (list, tuple, np.ndarray)):
                try:
                    return func(*coord_arrays, *p0)
                except TypeError:
                    pass
            return func(*coord_arrays, pars_list)

    # len(pars_list) > 1: try unpacked (*pars_list) first, then packed (pars_list)
    try:
        return func(*coord_arrays, *pars_list)
    except TypeError:
        return func(*coord_arrays, pars_list)


def _flatten_objects(
    win: Any,
) -> Tuple[List[Any], bool, Optional[Tuple[int, ...]]]:
    """Normalize single object or sequence/array of objects into ``(obj_list, is_seq, shape)``."""
    if isinstance(win, DataOpArray):
        return list(win.ravel(order="F")), True, win.shape
    if isinstance(win, np.ndarray) and win.dtype == object:
        shape = win.shape if win.ndim >= 2 else (1, int(win.size))
        return list(win.ravel(order="F")), True, shape
    if isinstance(win, (list, tuple)):
        return list(win), True, (1, len(win))
    return [win], False, None


def _pack_eval_output(
    wout: List[Any],
    is_seq: bool,
    shape: Optional[Tuple[int, ...]],
) -> Any:
    """Pack evaluated output objects matching input container structure."""
    if not is_seq and len(wout) == 1:
        return wout[0]
    if len(wout) == 0:
        return DataOpArray([])
    types = {type(x) for x in wout}
    if len(types) == 1:
        return DataOpArray.from_list(wout, shape=shape)
    return list(wout)


# ==============================================================================
# Cut Methods (ported from @sqw/cut.m, cut_single_.m, cut_accumulate_data_.m)
# ==============================================================================


def cut_sqw_method(self: "sqw", *args: Any, **kwargs: Any) -> Any:
    """Take a cut from an ``sqw`` object along existing or new projection axes.

    Ported from ``horace_core/sqw/@sqw/cut.m``, ``@sqw/private/cut_single_.m``, and
    ``@sqw/private/cut_accumulate_data_.m``.
    """
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return_cut = bool(nargout > 0)

    if bool(self.dnd_type):
        return self._data.cut(*args, nargout=nargout, **kwargs)

    log_level = int(hor_config().log_level)
    targ_proj, pbin_list, sym, opt = self.process_and_validate_cut_inputs(
        return_cut, *args, nargout=4
    )

    n_cuts = len(pbin_list)
    if n_cuts > 1 and opt.get("outfile"):
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            "Guard against single file output with multiple cuts",
        )

    results: List[Any] = []
    for pbin in pbin_list:
        targ_ax_block, cur_proj, cur_sym = self.define_target_axes_block(
            targ_proj, pbin, sym, nargout=3
        )
        wout = _cut_single(
            self,
            cur_proj,
            targ_ax_block,
            opt,
            log_level,
            cur_sym,
            return_cut=return_cut,
        )
        if return_cut:
            results.append(wout)

    if not return_cut:
        return None
    return results[0] if len(results) == 1 else DataOpArray(results)


def cut_dnd_method(self: "sqw", *args: Any, **kwargs: Any) -> Any:
    """Take a DND cut (no pixels retained) from an ``sqw`` object (port of ``@sqw/cut_dnd.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return self.cut(*args, "-nopix", nargout=nargout, **kwargs)


def cut_sqw_only_method(self: "sqw", *args: Any, **kwargs: Any) -> Any:
    """Take an SQW cut from an ``sqw`` object (port of ``@sqw/cut_sqw.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    if bool(self.dnd_type):
        raise HoraceError(
            "HORACE:cut:invalid_argument",
            "Cannot call cut_sqw on an sqw object of dnd_type (without pixels)",
        )
    return self.cut(*args, nargout=nargout, **kwargs)


def cut_sym_method(self: "sqw", *args: Any, **kwargs: Any) -> Any:
    """Take a symmetry-enabled cut from an ``sqw`` object (port of ``@sqw/cut_sym.m``)."""
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)
    return self.cut(*args, nargout=nargout, **kwargs)


def _cut_single(
    self: "sqw",
    targ_proj: Any,
    targ_axes: Any,
    opt: Dict[str, Any],
    log_level: int,
    sym: Any,
    return_cut: bool = True,
) -> Optional[Union["sqw", "DnDBase"]]:
    """Execute a single cut on this ``sqw`` object (port of ``@sqw/private/cut_single_.m``)."""
    from pyhorace.sqw.dnd_base import DnDBase
    from pyhorace.sqw.sqw import sqw

    keep_pix = bool(opt.get("keep_pix", True))
    outfile = str(opt.get("outfile", "") or "")

    s, e, npix, pix_out, runid_contributed = _cut_accumulate_data(
        self, targ_proj, targ_axes, keep_pix, log_level, return_cut, sym
    )

    proj_0 = (
        targ_proj[0]
        if isinstance(targ_proj, (list, tuple, np.ndarray))
        else targ_proj
    )
    data_dnd = DnDBase.dnd(targ_axes, proj_0, s, e, npix)
    data_dnd.title = self.title

    if not keep_pix:
        wout: Union[sqw, DnDBase] = data_dnd
    else:
        main_header = copy.deepcopy(self._main_header)
        exp_info = copy.deepcopy(self._exper_info)

        if (
            not main_header.creation_date_defined
            and exp_info.n_runs > 0
            and len(runid_contributed) > 0
        ):
            cur_keys = set(int(k) for k in np.asarray(exp_info.runid_map.keys).ravel())
            contrib_set = set(int(r) for r in np.asarray(runid_contributed).ravel())
            if not contrib_set.issubset(cur_keys) and self._pix.num_pixels > 0:
                u_ids = np.unique(self._pix.run_idx).astype(float)
                if u_ids.size == exp_info.n_runs:
                    exp_info.runid_map = fast_map(
                        u_ids, np.arange(1, exp_info.n_runs + 1, dtype=int)
                    )
                    exp_info.runid_recalculated = False

        if len(runid_contributed) > 0 and exp_info.n_runs > 0:
            exp_sub = exp_info.get_subobj(
                np.asarray(runid_contributed, dtype=int).ravel()
            )
        else:
            exp_sub = Experiment()

        if self._main_header.nfiles != exp_sub.n_runs:
            main_header.creation_date = datetime.now()
        main_header.nfiles = exp_sub.n_runs
        wout_sqw = sqw()
        wout_sqw._do_check_combo_arg = False
        wout_sqw._main_header = main_header
        wout_sqw._exper_info = exp_sub
        wout_sqw._data = data_dnd
        wout_sqw._pix = pix_out
        wout_sqw._do_check_combo_arg = True
        wout_sqw.check_combo_arg()
        wout = wout_sqw

    if outfile:
        wout.save(outfile)
        wout.full_filename = outfile

    if not return_cut:
        return None
    return wout


def _cut_accumulate_data(
    self: "sqw",
    targ_proj: Any,
    targ_axes: Any,
    keep_pixels: bool,
    log_level: int,
    return_cut: bool,
    sym: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PixelDataBase, np.ndarray]:
    """Accumulate binned signal, error, npix, and pixels for a cut.

    Ported from ``horace_core/sqw/@sqw/private/cut_accumulate_data_.m``.
    """
    targ_proj_list: List[Any] = (
        list(targ_proj)
        if isinstance(targ_proj, (list, tuple, np.ndarray))
        else [targ_proj]
    )
    sym_list: List[Any] = (
        list(sym) if isinstance(sym, (list, tuple, np.ndarray)) else [sym]
    )
    n_sym = len(targ_proj_list)
    apply_symmetries = n_sym > 1

    sproj = copy.deepcopy(self._data.proj)
    saxes = self._data.axes
    data_npix = self._data.npix

    starts_per_sym: List[np.ndarray] = []
    sizes_per_sym: List[np.ndarray] = []
    for proj in targ_proj_list:
        b_starts, b_sizes = sproj.get_nrange(data_npix, saxes, targ_axes, proj)
        starts_per_sym.append(np.asarray(b_starts, dtype=np.int64).ravel())
        sizes_per_sym.append(np.asarray(b_sizes, dtype=np.int64).ravel())

    if apply_symmetries:
        block_starts, block_sizes = _merge_ranges(starts_per_sym, sizes_per_sym)
    else:
        block_starts = starts_per_sym[0]
        block_sizes = sizes_per_sym[0]

    targ_dims = tuple(int(x) for x in targ_axes.dims_as_ssize)
    s = np.zeros(targ_dims, dtype=np.float64, order="F")
    e = np.zeros(targ_dims, dtype=np.float64, order="F")
    npix = np.zeros(targ_dims, dtype=np.float64, order="F")

    buf_size = int(hor_config().mem_chunk_size)
    chunks_starts, chunks_sizes = _split_data_blocks(
        block_starts, block_sizes, buf_size
    )

    pix_retained: List[PixelDataBase] = []
    pix_ix_retained: List[np.ndarray] = []
    unique_runid = np.array([], dtype=np.int64)

    num_chunks = len(chunks_starts)
    is_fb = bool(self._pix.is_filebacked)
    access_type = "Read" if is_fb else "Access"
    pixel_contrib_name = "retained" if keep_pixels else "contributed"

    t_start = time.perf_counter()
    t_read_total = 0.0
    t_bin_total = 0.0
    t_sort_total = 0.0

    for iter_idx, (ch_starts, ch_sizes) in enumerate(
        zip(chunks_starts, chunks_sizes), start=1
    ):
        t0 = time.perf_counter()
        candidate_pix = self._pix.get_pix_in_ranges(ch_starts, ch_sizes, False, True)
        t_read_total += time.perf_counter() - t0
        if candidate_pix.num_pixels == 0:
            continue

        if log_level >= 1:
            print(
                f"*** Step {iter_idx:3d} of {num_chunks:3d}; Read data for {candidate_pix.num_pixels:8d} pixels -- processing data...",
                end="",
            )

        t1 = time.perf_counter()
        npix_before = float(np.sum(npix))
        if keep_pixels:
            for i in range(n_sym):
                proj = targ_proj_list[i]
                if apply_symmetries:
                    (
                        npix,
                        s,
                        e,
                        pix_ok,
                        unique_runid,
                        pix_indx,
                        selected,
                    ) = proj.bin_pixels(
                        targ_axes,
                        candidate_pix,
                        npix,
                        s,
                        e,
                        unique_runid,
                        "-nomex",
                        nargout=7,
                    )
                    if pix_ok.num_pixels > 0:
                        pix_ok.q_coordinates = sym_list[i].transform_vec(
                            pix_ok.q_coordinates
                        )
                    candidate_pix = candidate_pix.tag(selected)
                else:
                    (
                        npix,
                        s,
                        e,
                        pix_ok,
                        unique_runid,
                        pix_indx,
                    ) = proj.bin_pixels(
                        targ_axes,
                        candidate_pix,
                        npix,
                        s,
                        e,
                        unique_runid,
                        "-nomex",
                        nargout=6,
                    )
                if pix_ok.num_pixels > 0:
                    pix_retained.append(pix_ok)
                    pix_ix_retained.append(
                        np.asarray(pix_indx, dtype=np.int64).ravel()
                    )
        else:
            for i in range(n_sym):
                proj = targ_proj_list[i]
                if apply_symmetries:
                    npix, s, e, selected = proj.bin_pixels(
                        targ_axes,
                        candidate_pix,
                        npix,
                        s,
                        e,
                        "-selected_only",
                        "-nomex",
                        nargout=4,
                    )
                    candidate_pix = candidate_pix.tag(selected)
                else:
                    npix, s, e = proj.bin_pixels(
                        targ_axes,
                        candidate_pix,
                        npix,
                        s,
                        e,
                        "-nomex",
                        nargout=3,
                    )
        t_bin_total += time.perf_counter() - t1
        npix_step = int(round(float(np.sum(npix)) - npix_before))
        if log_level >= 1:
            print(f" ----->  {pixel_contrib_name}  {npix_step:8d} pixels")

    s, e = normalize_signal(s, e, npix)

    t2 = time.perf_counter()
    if keep_pixels:
        pix_out = _sort_pix(pix_retained, pix_ix_retained, npix)
    else:
        pix_out = PixelDataMemory()
    t_sort_total += time.perf_counter() - t2

    if log_level >= 2:
        t_total = max(time.perf_counter() - t_start, 1e-9)
        t_read_safe = max(t_read_total, 1e-9)
        n_read_bytes = float(np.sum(block_sizes)) * 9.0 * 4.0
        speed_mbs = (n_read_bytes / (1024.0 * 1024.0)) / t_read_safe
        prep_pct = (100.0 * t_sort_total / t_total) if keep_pixels else 0.0
        print(
            f"*** Cut timings: Total={t_total:.3f}s | {access_type} speed: {speed_mbs:.2f} MB/sec\n"
            f"    Resulting pix preparation: {prep_pct:5.1f}%"
        )

    return s, e, npix, pix_out, unique_runid


def _merge_ranges(
    starts_cell: Sequence[np.ndarray], sizes_cell: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge overlapping or adjacent 1-based pixel block ranges across symmetries.

    Ported from ``merge_ranges`` in ``@sqw/private/cut_accumulate_data_.m``.
    """
    non_empty_starts = []
    non_empty_ends = []
    for st, sz in zip(starts_cell, sizes_cell):
        st_arr = np.asarray(st, dtype=np.int64).ravel()
        sz_arr = np.asarray(sz, dtype=np.int64).ravel()
        valid = sz_arr > 0
        if np.any(valid):
            s_v = st_arr[valid]
            e_v = s_v + sz_arr[valid] - 1
            non_empty_starts.append(s_v)
            non_empty_ends.append(e_v)

    if not non_empty_starts:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    all_starts = np.concatenate(non_empty_starts)
    all_ends = np.concatenate(non_empty_ends)

    order = np.lexsort((all_ends, all_starts))
    all_starts = all_starts[order]
    all_ends = all_ends[order]

    merged_starts = [int(all_starts[0])]
    merged_ends = [int(all_ends[0])]
    for s_cur, e_cur in zip(all_starts[1:], all_ends[1:]):
        if int(s_cur) <= merged_ends[-1] + 1:
            if int(e_cur) > merged_ends[-1]:
                merged_ends[-1] = int(e_cur)
        else:
            merged_starts.append(int(s_cur))
            merged_ends.append(int(e_cur))

    ms = np.asarray(merged_starts, dtype=np.int64)
    me = np.asarray(merged_ends, dtype=np.int64)
    return ms, me - ms + 1


def _split_data_blocks(
    block_starts: np.ndarray, block_sizes: np.ndarray, buf_size: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split pixel block ranges into chunks of at most ``buf_size`` pixels.

    Ported from ``split_data_blocks.m``.
    """
    starts = np.asarray(block_starts, dtype=np.int64).ravel()
    sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
    valid = sizes > 0
    starts = starts[valid]
    sizes = sizes[valid]

    if starts.size == 0:
        return [], []

    buf_sz = max(int(buf_size), 1)
    chunks_starts: List[np.ndarray] = []
    chunks_sizes: List[np.ndarray] = []

    cur_starts: List[int] = []
    cur_sizes: List[int] = []
    cur_sum = 0

    for s_val, sz_val in zip(starts, sizes):
        s = int(s_val)
        sz = int(sz_val)
        while sz > 0:
            space = buf_sz - cur_sum
            if sz <= space:
                cur_starts.append(s)
                cur_sizes.append(sz)
                cur_sum += sz
                sz = 0
                if cur_sum >= buf_sz:
                    chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
                    chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))
                    cur_starts = []
                    cur_sizes = []
                    cur_sum = 0
            else:
                cur_starts.append(s)
                cur_sizes.append(space)
                chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
                chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))
                cur_starts = []
                cur_sizes = []
                cur_sum = 0
                s += space
                sz -= space

    if cur_starts:
        chunks_starts.append(np.asarray(cur_starts, dtype=np.int64))
        chunks_sizes.append(np.asarray(cur_sizes, dtype=np.int64))

    return chunks_starts, chunks_sizes


def _sort_pix(
    pix_retained: Sequence[PixelDataBase],
    pix_ix_retained: Sequence[np.ndarray],
    npix: np.ndarray,
) -> PixelDataBase:
    """Sort accumulated pixel chunks by 1-based bin index ``pix_ix_retained``.

    Ported from ``sort_pix.m``.
    """
    non_empty_pix = []
    non_empty_ix = []
    for p, ix in zip(pix_retained, pix_ix_retained):
        if p is not None and p.num_pixels > 0:
            non_empty_pix.append(
                p.get_pixels(
                    np.arange(1, p.num_pixels + 1, dtype=np.intp), "-raw_data"
                )
            )
            non_empty_ix.append(np.asarray(ix, dtype=np.int64).ravel())

    if not non_empty_pix:
        return PixelDataMemory()

    all_raw = np.hstack(non_empty_pix)
    all_ix = np.concatenate(non_empty_ix)
    if all_ix.size > 1:
        order = np.argsort(all_ix, kind="mergesort")
        all_raw = all_raw[:, order]

    out = PixelDataMemory(np.asarray(all_raw, dtype=np.float64))
    out.recalc_data_range("all")
    return out


# ==============================================================================
# Evaluation Argument Parsing (ported from @SQWDnDBase/private/parse_eval_args_.m)
# ==============================================================================


def parse_eval_args(
    win: Any,
    func_handle: Callable[..., Any],
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Callable[..., Any], List[Any], Dict[str, Any]]:
    """Parse arguments for ``sqw_eval``, ``func_eval``, and ``disp2sqw_eval``.

    Ported from ``horace_core/sqw/@SQWDnDBase/private/parse_eval_args_.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval

    if not callable(func_handle):
        raise HoraceError(
            "HORACE:SQWDnDBase:invalid_argument",
            "MATLAB:InputParser:ArgumentFailedValidation - "
            "HORACE:DnDBase:invalid_argument: First argument must be a function handle",
        )

    win_list, _, _ = _flatten_objects(win)
    n_win = max(len(win_list), 1)

    the_page_op: Optional[PageOp_sqw_eval] = None
    rem_args = list(args)
    if len(rem_args) > 0 and isinstance(rem_args[0], PageOp_sqw_eval):
        the_page_op = rem_args.pop(0)

    all_flag = bool(kwargs.pop("all", False) or kwargs.pop("all_bins", False))
    ave_flag = bool(kwargs.pop("average", False) or kwargs.pop("ave", False))
    filebacked_flag = bool(
        kwargs.pop("filebacked", False) or kwargs.pop("file_backed", False)
    )
    combine_flag = bool(kwargs.pop("combine", False))
    nopix_flag = bool(kwargs.pop("nopix", False))
    test_inputs = bool(kwargs.pop("test_input_parsing", False))
    outfile: Any = kwargs.pop("outfile", None)
    if "op" in kwargs and the_page_op is None:
        the_page_op = kwargs.pop("op")

    if len(kwargs) > 0:
        bad_key = next(iter(kwargs.keys()))
        raise HoraceError(
            "MATLAB:InputParser:ParamMissingValue",
            f"HORACE:sqw:invalid_argument - HORACE:DnDBase:invalid_argument: Unrecognised parameter '{bad_key}'",
        )

    idx = 0
    while idx < len(rem_args):
        arg = rem_args[idx]
        if isinstance(arg, PageOp_sqw_eval):
            the_page_op = arg
            idx += 1
            continue
        if isinstance(arg, dict):
            all_flag = bool(
                arg.get("all", arg.get("all_bins", all_flag))
            )
            ave_flag = bool(arg.get("average", arg.get("ave", ave_flag)))
            filebacked_flag = bool(
                arg.get("filebacked", arg.get("file_backed", filebacked_flag))
            )
            combine_flag = bool(arg.get("combine", combine_flag))
            nopix_flag = bool(arg.get("nopix", nopix_flag))
            if "outfile" in arg and arg["outfile"]:
                outfile = arg["outfile"]
            if "pageop_processor" in arg and arg["pageop_processor"] is not None:
                the_page_op = arg["pageop_processor"]
            idx += 1
            continue
        if isinstance(arg, str):
            a_low = arg.lower().strip()
            if a_low.startswith("-"):
                flag = a_low[1:]
                if "all".startswith(flag) and len(flag) >= 2:
                    if idx + 1 < len(rem_args) and isinstance(
                        rem_args[idx + 1], (bool, np.bool_)
                    ):
                        all_flag = bool(rem_args[idx + 1])
                        idx += 2
                    else:
                        all_flag = True
                        idx += 1
                    continue
                if "average".startswith(flag) and len(flag) >= 2:
                    if idx + 1 < len(rem_args) and isinstance(
                        rem_args[idx + 1], (bool, np.bool_)
                    ):
                        ave_flag = bool(rem_args[idx + 1])
                        idx += 2
                    else:
                        ave_flag = True
                        idx += 1
                    continue
                if flag in ("filebacked", "file_backed") or (
                    "filebacked".startswith(flag) and len(flag) >= 2
                ):
                    if idx + 1 < len(rem_args) and isinstance(
                        rem_args[idx + 1], (bool, np.bool_)
                    ):
                        filebacked_flag = bool(rem_args[idx + 1])
                        idx += 2
                    else:
                        filebacked_flag = True
                        idx += 1
                    continue
                if "combine".startswith(flag) and len(flag) >= 2:
                    if idx + 1 < len(rem_args) and isinstance(
                        rem_args[idx + 1], (bool, np.bool_)
                    ):
                        combine_flag = bool(rem_args[idx + 1])
                        idx += 2
                    else:
                        combine_flag = True
                        idx += 1
                    continue
                if "nopix".startswith(flag) and len(flag) >= 2:
                    if idx + 1 < len(rem_args) and isinstance(
                        rem_args[idx + 1], (bool, np.bool_)
                    ):
                        nopix_flag = bool(rem_args[idx + 1])
                        idx += 2
                    else:
                        nopix_flag = True
                        idx += 1
                    continue
                if "test_input_parsing".startswith(flag) and len(flag) >= 2:
                    test_inputs = True
                    idx += 1
                    continue
                if "outfile".startswith(flag) and len(flag) >= 3:
                    if idx + 1 >= len(rem_args):
                        raise HoraceError(
                            "MATLAB:InputParser:ParamMissingValue",
                            "HORACE:sqw:invalid_argument - HORACE:DnDBase:invalid_argument: Missing filename after 'outfile'",
                        )
                    outfile = rem_args[idx + 1]
                    idx += 2
                    continue
                raise HoraceError(
                    "MATLAB:InputParser:ParamMissingValue",
                    f"HORACE:sqw:invalid_argument - HORACE:DnDBase:invalid_argument: Unrecognised flag '{arg}'",
                )
            else:
                k_low = a_low
                # Support bare 'all' or 'ave' without value if at end of args or followed by another string option
                if ("all".startswith(k_low) and len(k_low) >= 2) and (
                    idx + 1 >= len(rem_args)
                    or not isinstance(
                        rem_args[idx + 1], (bool, np.bool_, int, np.integer)
                    )
                ):
                    all_flag = True
                    idx += 1
                    continue
                if ("average".startswith(k_low) and len(k_low) >= 2) and (
                    idx + 1 >= len(rem_args)
                    or not isinstance(
                        rem_args[idx + 1], (bool, np.bool_, int, np.integer)
                    )
                ):
                    ave_flag = True
                    idx += 1
                    continue

                if idx + 1 >= len(rem_args):
                    raise HoraceError(
                        "MATLAB:InputParser:ParamMissingValue",
                        f"HORACE:sqw:invalid_argument - HORACE:DnDBase:invalid_argument: Parameter '{arg}' requires a value",
                    )
                val = rem_args[idx + 1]
                if "average".startswith(k_low) and len(k_low) >= 2:
                    ave_flag = bool(val)
                elif "all".startswith(k_low) and len(k_low) >= 2:
                    all_flag = bool(val)
                elif k_low in ("filebacked", "file_backed") or (
                    "filebacked".startswith(k_low) and len(k_low) >= 2
                ):
                    filebacked_flag = bool(val)
                elif "combine".startswith(k_low) and len(k_low) >= 2:
                    combine_flag = bool(val)
                elif "nopix".startswith(k_low) and len(k_low) >= 2:
                    nopix_flag = bool(val)
                elif "test_input_parsing".startswith(k_low) and len(k_low) >= 2:
                    test_inputs = bool(val)
                elif "outfile".startswith(k_low) and len(k_low) >= 3:
                    outfile = val
                else:
                    raise HoraceError(
                        "MATLAB:InputParser:ParamMissingValue",
                        f"HORACE:sqw:invalid_argument - HORACE:DnDBase:invalid_argument: Unrecognised parameter '{arg}'",
                    )
                idx += 2
                continue
        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            f"HORACE:DnDBase:invalid_argument: Unrecognised argument {arg!r}",
        )

    if isinstance(pars, (list, tuple)):
        pars_cell = list(pars)
    elif pars is None or (isinstance(pars, np.ndarray) and pars.size == 0):
        pars_cell = [[]]
    else:
        pars_cell = [pars]

    if outfile is None:
        outfile_cell = [""] * n_win
        outfiles_empty = True
    elif isinstance(outfile, (list, tuple, np.ndarray)):
        outfile_cell = [str(x) if x is not None else "" for x in outfile]
        outfiles_empty = len(outfile_cell) == 0 or all(len(x) == 0 for x in outfile_cell)
        if outfiles_empty:
            outfile_cell = [""] * n_win
    elif isinstance(outfile, (str, os.PathLike)):
        s_out = str(outfile)
        if s_out == "":
            outfile_cell = [""] * n_win
            outfiles_empty = True
        else:
            outfile_cell = [s_out]
            outfiles_empty = False
    else:
        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            f"'outfile' must be a string or sequence of strings, got {type(outfile).__name__}",
        )

    if not outfiles_empty and len(outfile_cell) != n_win:
        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            f"Number of outfiles specified must match number of input objects.\n"
            f"Found {len(outfile_cell)} outfile(s), but {n_win} sqw object(s).",
        )

    opts: Dict[str, Any] = {
        "func_handle": func_handle,
        "pars": pars_cell,
        "average": ave_flag,
        "all": all_flag,
        "all_bins": all_flag,
        "filebacked": filebacked_flag,
        "combine": combine_flag,
        "nopix": nopix_flag,
        "test_input_parsing": test_inputs,
        "outfile": outfile_cell,
        "pageop_processor": the_page_op,
    }
    return func_handle, pars_cell, opts


# ==============================================================================
# File Loader Helper (ported from horace_core/sqw/file_io/load_sqw_dnd.m)
# ==============================================================================


def load_sqw_dnd(sources: Any, *args: Any, **kwargs: Any) -> List[Any]:
    """Load SQW or DnD objects from file paths, or pass through in-memory objects.

    Ported from ``horace_core/sqw/file_io/load_sqw_dnd.m``.
    """
    from pyhorace.io.sqw_formats_factory import sqw_formats_factory
    from pyhorace.sqw.sqw import sqw
    from pyhorace.sqw.sqw_dnd_base import SQWDnDBase

    if isinstance(sources, (str, os.PathLike)):
        src_list: List[Any] = [str(sources)]
    elif isinstance(sources, SQWDnDBase):
        src_list = [sources]
    elif isinstance(sources, DataOpArray):
        src_list = list(sources.ravel(order="F"))
    elif isinstance(sources, np.ndarray) and sources.dtype == object:
        src_list = list(sources.ravel(order="F"))
    elif isinstance(sources, (list, tuple)):
        src_list = list(sources)
    else:
        raise HoraceError(
            "HORACE:load_sqw_dnd:invalid_argument",
            f"Cannot load sqw/dnd from input of type {type(sources).__name__}",
        )

    # Extract filebacked flag from args/kwargs if present (while validating unknown flags via parse_eval_args later)
    fb_val: Optional[bool] = None
    if "filebacked" in kwargs:
        fb_val = bool(kwargs["filebacked"])
    elif "file_backed" in kwargs:
        fb_val = bool(kwargs["file_backed"])

    idx = 0
    while idx < len(args):
        a = args[idx]
        if isinstance(a, str):
            a_low = a.lower().strip()
            if a_low.startswith("-"):
                flag = a_low[1:]
                if flag in ("filebacked", "file_backed") or (
                    "filebacked".startswith(flag) and len(flag) >= 2
                ):
                    if idx + 1 < len(args) and isinstance(
                        args[idx + 1], (bool, np.bool_)
                    ):
                        fb_val = bool(args[idx + 1])
                        idx += 2
                    else:
                        fb_val = True
                        idx += 1
                    continue
            else:
                if (
                    a_low in ("filebacked", "file_backed")
                    or ("filebacked".startswith(a_low) and len(a_low) >= 2)
                ) and idx + 1 < len(args):
                    fb_val = bool(args[idx + 1])
                    idx += 2
                    continue
        idx += 1

    loaded: List[Any] = []
    for item in src_list:
        if isinstance(item, SQWDnDBase):
            loaded.append(item)
        elif isinstance(item, (str, os.PathLike)):
            fname = str(item).strip()
            if not fname or not os.path.isfile(fname):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"File does not exist: {fname}",
                )
            ldr = sqw_formats_factory.instance().get_loader(fname)
            try:
                if bool(ldr.sqw_type):
                    if fb_val is not None:
                        w_item = sqw(ldr, file_backed=fb_val)
                    else:
                        w_item = sqw(ldr)
                else:
                    w_item = ldr.get_dnd()
            finally:
                ldr.fclose()
            loaded.append(w_item)
        else:
            raise HoraceError(
                "HORACE:func_eval:invalid_argument",
                f"Cannot evaluate function on input element of type {type(item).__name__}",
            )

    return loaded


# ==============================================================================
# sqw_eval and disp2sqw_eval Methods
# ==============================================================================


def sqw_eval_pix_method(
    self: "sqw",
    sqwfunc: Callable[..., Any],
    pars: Any,
    options: Union[Dict[str, Any], bool],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Helper function for ``sqw_eval`` executed on a full ``sqw`` object containing pixels.

    Ported from ``horace_core/sqw/@sqw/sqw_eval_pix.m``.
    Uses :class:`~pyhorace.coord_transf.page_operations.page_op_sqw_eval.PageOp_sqw_eval`.
    """
    from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
    from pyhorace.sqw.sqw import sqw

    if isinstance(options, dict):
        ave_pix = bool(options.get("average", False))
        outfile = options.get("outfile", "")
        if isinstance(outfile, (list, tuple)):
            outfile = outfile[0] if len(outfile) > 0 else ""
        outfile = str(outfile or "")
        filebacked = bool(options.get("filebacked", False))
        nopix = bool(options.get("nopix", False))
        eval_op = options.get("pageop_processor") or options.get("op") or kwargs.get("op")
    else:
        ave_pix = bool(options)
        nopix = bool(args[0]) if len(args) > 0 else bool(kwargs.get("nopix", False))
        outfile = str(args[1] or "") if len(args) > 1 else str(kwargs.get("outfile", "") or "")
        filebacked = bool(args[2]) if len(args) > 2 else bool(kwargs.get("filebacked", False))
        eval_op = args[3] if len(args) > 3 else kwargs.get("op")

    if eval_op is None:
        eval_op = PageOp_sqw_eval()
    else:
        eval_op = copy.copy(eval_op)

    if outfile:
        eval_op.outfile = outfile

    eval_op.do_nopix = nopix
    eval_op.init_filebacked_output = bool(filebacked or self.is_filebacked)
    eval_op = eval_op.init(self, sqwfunc, pars, ave_pix)
    eval_op.do_nopix = nopix
    eval_op.init_filebacked_output = bool(filebacked or self.is_filebacked)

    return sqw.apply_op(self, eval_op)


def sqw_eval_nopix_method(
    self: "SQWDnDBase",
    sqwfunc: Callable[..., Any],
    pars: Any = None,
    options: Optional[Dict[str, Any]] = None,
) -> "SQWDnDBase":
    """Evaluate S(Q, w) model function at bin centres of a pixel-less ``sqw`` or ``DnDBase`` object.

    Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval_nopix.m`` and
    ``horace_core/sqw/@DnDBase/private/sqw_eval_nopix_.m``.
    """
    from pyhorace.sqw.sqw import sqw

    if isinstance(self, sqw):
        self._data = sqw_eval_nopix_method(self._data, sqwfunc, pars, options)
        return self

    opts = options or {}
    all_bins = bool(opts.get("all_bins", opts.get("all", False)))
    qw = self.calculate_qw_bins(nargout=1)
    empty = self.npix == 0
    has_any_pix = not bool(np.all(empty))

    if not all_bins and has_any_pix:
        ok = (~empty).ravel(order="F")
        qw_eval = [q.ravel(order="F")[ok].reshape(-1, 1) for q in qw]
        val_sub = np.asarray(
            _invoke_eval_func(sqwfunc, *qw_eval, pars=pars), dtype=np.float64
        ).ravel(order="F")
        s_flat = np.zeros(self.s.size, dtype=np.float64)
        s_flat[ok] = val_sub
        s_new = s_flat.reshape(self.s.shape, order="F")
    elif not all_bins and not has_any_pix:
        s_new = np.zeros_like(self.s, dtype=np.float64)
    else:
        qw_eval = [q.reshape(-1, 1) for q in qw]
        val = np.asarray(
            _invoke_eval_func(sqwfunc, *qw_eval, pars=pars), dtype=np.float64
        )
        s_new = val.reshape(self.s.shape, order="F").copy()

    self.s = s_new
    self.e = np.zeros_like(s_new)
    if all_bins and np.any(empty):
        npix_new = self.npix.copy()
        npix_new[empty] = 1
        self.npix = npix_new
    return self


def sqw_eval_method(
    win: Any,
    sqwfunc: Callable[..., Any],
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Calculate ``sqw`` or ``DnDBase`` for a model scattering function ``S(Q, w)``.

    Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval.m`` and
    ``horace_core/algorithms/sqw_eval.m``.
    """
    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)

    # Support file path or list of file paths (algorithms/sqw_eval.m)
    is_file_input = False
    if isinstance(win, (str, os.PathLike)):
        is_file_input = True
    elif isinstance(win, (list, tuple)) and len(win) > 0 and any(
        isinstance(x, (str, os.PathLike)) for x in win
    ):
        is_file_input = True

    if is_file_input:
        # Validate eval arguments first so unknown flags raise immediately
        sqwfunc_parsed, pars_cell, opts = parse_eval_args(
            win if isinstance(win, (list, tuple)) else [win],
            sqwfunc,
            pars,
            *args,
            **kwargs,
        )
        loaded_objs = load_sqw_dnd(win, *args, **kwargs)
        is_seq = isinstance(win, (list, tuple))
        shape = (1, len(loaded_objs)) if is_seq else None
        win_list = loaded_objs
    else:
        win_list, is_seq, shape = _flatten_objects(win)
        sqwfunc_parsed, pars_cell, opts = parse_eval_args(
            win, sqwfunc, pars, *args, **kwargs
        )

    outfiles: List[str] = list(opts["outfile"])
    outfiles_empty = len(outfiles) == 0 or all(len(x) == 0 for x in outfiles)

    if outfiles_empty or bool(opts["filebacked"]):
        if nargout != 1:
            raise HoraceError(
                "HORACE:sqw_eval:invalid_argument",
                f"This method request single output argument. Got: {nargout}",
            )

    if bool(opts["filebacked"]) and nargout > 0:
        outfiles = [""] * len(win_list)

    for obj_i in win_list:
        has_pix = hasattr(obj_i, "has_pixels") and bool(obj_i.has_pixels())
        if not has_pix and bool(opts["all_bins"]):
            for dim_size in obj_i.s.shape:
                if dim_size == 0:
                    raise HoraceError(
                        "HORACE:sqw_eval:invalid_argument",
                        "Empty object cannot be evaluated using option 'all'",
                    )

    wout: List[Any] = []
    for i, obj_i in enumerate(win_list):
        has_pix = hasattr(obj_i, "has_pixels") and bool(obj_i.has_pixels())
        outfile_i = outfiles[i] if i < len(outfiles) else ""
        optl = dict(opts)
        optl["outfile"] = outfile_i
        if has_pix:
            res_i = obj_i.sqw_eval_pix(sqwfunc_parsed, pars_cell, optl)
            if bool(opts["nopix"]) and outfile_i:
                res_i.save(outfile_i)
            wout.append(res_i)
        else:
            res_i = obj_i.copy()
            res_i = res_i.sqw_eval_nopix(sqwfunc_parsed, pars_cell, optl)
            if outfile_i:
                res_i.save(outfile_i)
            wout.append(res_i)

    if nargout == 0:
        return None
    return _pack_eval_output(wout, is_seq, shape)


def disp2sqw_eval_method(
    win: Any,
    dispreln: Callable[..., Any],
    pars: Any,
    fwhh: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Calculate ``sqw`` or ``DnDBase`` for a model dispersion relation.

    Ported from ``horace_core/sqw/@SQWDnDBase/disp2sqw_eval.m`` and
    ``horace_core/algorithms/disp2sqw_eval.m``.
    """
    from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
    from pyhorace.sqw.sqw_dnd_base import disp2sqw

    nargout = kwargs.pop("nargout", None)
    if nargout is None:
        nargout = _infer_nargout(default=1)

    pars_list = (
        list(pars)
        if isinstance(pars, (list, tuple))
        else ([pars] if pars is not None else [])
    )
    meta_param = [dispreln, pars_list, fwhh]
    op = PageOp_sqw_eval()
    op.op_name = "disp2sqw_eval"
    return sqw_eval_method(
        win, disp2sqw, meta_param, op, *args, nargout=nargout, **kwargs
    )


# ==============================================================================
# func_eval Methods (ported from @sqw/func_eval.m, @DnDBase/func_eval.m, algorithms/func_eval.m)
# ==============================================================================


def func_eval_dnd(
    win: Any,
    func_handle: Callable[..., Any],
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate a function at the plotting axis bin centres of a ``DnDBase`` object or array of ``DnDBase`` objects.

    Ported from ``horace_core/sqw/@DnDBase/func_eval.m``.
    """
    win_list, is_seq, shape = _flatten_objects(win)

    if len(args) == 1 and isinstance(args[0], dict) and len(kwargs) == 0:
        opts = args[0]
        if not callable(func_handle):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "MATLAB:InputParser:ArgumentFailedValidation: First argument must be a function handle",
            )
        if isinstance(pars, (list, tuple)):
            pars_cell = list(pars)
        elif pars is None or (isinstance(pars, np.ndarray) and pars.size == 0):
            pars_cell = [[]]
        else:
            pars_cell = [pars]
    else:
        func_handle, pars_cell, opts = parse_eval_args(
            win, func_handle, pars, *args, **kwargs
        )

    if len(win_list) > 1:
        input_dims = [int(x.dimensions()) for x in win_list]
        if any(d != input_dims[0] for d in input_dims):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Input dnd objects must have equal image dimensions.\n"
                f"Found dimensions {input_dims}.",
            )

    if any(int(x.dimensions()) == 0 for x in win_list):
        raise HoraceError(
            "HORACE:DnDBase:invalid_argument",
            "func_eval not supported for zero dimensional objects.",
        )

    all_flag = bool(opts.get("all", False) or opts.get("all_bins", False))
    if all_flag:
        for obj_i in win_list:
            if any(dim_sz == 0 for dim_sz in obj_i.s.shape):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    "One of the datasets has a zero-length dimension, which cannot be evaluated with the -all option",
                )

    outfiles: List[str] = (
        list(opts["outfile"])
        if isinstance(opts.get("outfile"), (list, tuple))
        else [str(opts.get("outfile", "") or "")] * len(win_list)
    )

    wout: List[Any] = []
    for i, obj_i in enumerate(win_list):
        ndim = int(obj_i.dimensions())
        wout_i = obj_i.copy()
        empty_npix = obj_i.npix == 0
        has_any_pix = not bool(np.all(empty_npix))

        pcent = [0.5 * (pi[:-1] + pi[1:]) for pi in obj_i.p]
        if ndim == 1:
            coord_cols = [pcent[0].reshape(-1, 1)]
        else:
            grids = np.meshgrid(*pcent, indexing="ij")
            coord_cols = [g.ravel(order="F").reshape(-1, 1) for g in grids]

        if not all_flag and has_any_pix:
            ok = (~empty_npix).ravel(order="F")
            coord_sub = [c[ok].reshape(-1, 1) for c in coord_cols]
            val_sub = np.asarray(
                _invoke_eval_func(func_handle, *coord_sub, pars=pars_cell),
                dtype=np.float64,
            ).ravel(order="F")
            s_flat = np.zeros(obj_i.s.size, dtype=np.float64)
            s_flat[ok] = val_sub
            s_new = s_flat.reshape(obj_i.s.shape, order="F")
        elif not all_flag and not has_any_pix:
            s_new = np.zeros_like(obj_i.s, dtype=np.float64)
        else:
            val = np.asarray(
                _invoke_eval_func(func_handle, *coord_cols, pars=pars_cell),
                dtype=np.float64,
            )
            s_new = val.reshape(obj_i.s.shape, order="F").copy()

        wout_i.s = s_new
        wout_i.e = np.zeros_like(s_new)
        if all_flag:
            wout_i.npix = np.ones_like(wout_i.npix)

        outfile_i = outfiles[i] if i < len(outfiles) else ""
        if outfile_i:
            wout_i.save(outfile_i)
        wout.append(wout_i)

    return _pack_eval_output(wout, is_seq, shape)


def func_eval_sqw(
    win: Any,
    func_handle: Callable[..., Any],
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate a function at the plotting bin centres of an ``sqw`` object or array of ``sqw`` objects.

    Ported from ``horace_core/sqw/@sqw/func_eval.m``.
    Uses :class:`~pyhorace.coord_transf.page_operations.page_op_func_eval.PageOp_func_eval`.
    """
    from pyhorace.coord_transf.page_operations.page_op_func_eval import (
        PageOp_func_eval,
    )
    from pyhorace.sqw.sqw import sqw

    win_list, is_seq, shape = _flatten_objects(win)
    func_handle_parsed, pars_cell, opts = parse_eval_args(
        win, func_handle, pars, *args, **kwargs
    )

    if len(win_list) > 1:
        input_dims = [int(x.dimensions()) for x in win_list]
        if any(d != input_dims[0] for d in input_dims):
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Input sqw objects must have equal image dimensions.\n"
                f"Found dimensions {input_dims}.",
            )

    if any(int(x.dimensions()) == 0 for x in win_list):
        raise HoraceError(
            "HORACE:sqw:invalid_argument",
            "func_eval not supported for zero dimensional objects.",
        )

    outfiles: List[str] = list(opts["outfile"])

    wout: List[Any] = []
    for i, obj_i in enumerate(win_list):
        sqw_type = bool(obj_i.has_pixels())
        opts_i = dict(opts)
        if sqw_type:
            opts_i["all"] = False
            opts_i["all_bins"] = False
        opts_i["outfile"] = ""

        wout_i = obj_i.copy()
        wout_i.data = func_eval_dnd(
            obj_i.data, func_handle_parsed, pars_cell, opts_i
        )

        outfile_i = outfiles[i] if i < len(outfiles) else ""
        if sqw_type:
            page_op = PageOp_func_eval()
            if outfile_i:
                page_op.outfile = outfile_i
            page_op.do_nopix = bool(opts.get("nopix", False))
            page_op.init_filebacked_output = bool(
                opts.get("filebacked", False) or obj_i.is_filebacked
            )
            page_op = page_op.init(wout_i)
            page_op.do_nopix = bool(opts.get("nopix", False))
            page_op.init_filebacked_output = bool(
                opts.get("filebacked", False) or obj_i.is_filebacked
            )
            wout_i = sqw.apply_op(wout_i, page_op)
            if bool(opts.get("nopix", False)) and outfile_i:
                wout_i.save(outfile_i)
        else:
            if bool(opts_i.get("all", False)):
                wout_i.data.npix = np.ones_like(wout_i.data.npix)
            if bool(opts.get("nopix", False)):
                wout_i = wout_i.data
            if outfile_i:
                wout_i.save(outfile_i)

        wout.append(wout_i)

    return _pack_eval_output(wout, is_seq, shape)


def func_eval(
    sources: Any,
    func_handle: Callable[..., Any],
    pars: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate a function at the plotting bin centres of ``sqw``/``DnDBase`` object(s) or file(s).

    Ported from ``horace_core/algorithms/func_eval.m``.
    """
    from pyhorace.sqw.dnd_base import DnDBase
    from pyhorace.sqw.sqw import sqw

    if isinstance(sources, (list, tuple)):
        for item in sources:
            if isinstance(item, (list, tuple, DataOpArray)) or (
                isinstance(item, np.ndarray) and item.size > 1
            ):
                raise HoraceError(
                    "HORACE:func_eval:invalid_argument",
                    "Input cell/list cannot contain arrays of objects.",
                )

    loaded = load_sqw_dnd(sources, *args, **kwargs)
    is_sqw = [isinstance(x, sqw) for x in loaded]
    is_dnd = [isinstance(x, DnDBase) for x in loaded]
    if any(is_sqw) and any(is_dnd):
        raise HoraceError(
            "HORACE:func_eval:invalid_argument",
            "Cannot call func_eval with a mix of sqw and dnd objects.",
        )

    is_seq = isinstance(sources, (list, tuple, DataOpArray)) or (
        isinstance(sources, np.ndarray) and sources.dtype == object
    )
    shape = (
        sources.shape
        if isinstance(sources, (DataOpArray, np.ndarray))
        else ((1, len(loaded)) if is_seq else None)
    )
    target = DataOpArray.from_list(loaded, shape=shape) if is_seq else loaded[0]

    if all(is_sqw):
        return func_eval_sqw(target, func_handle, pars, *args, **kwargs)
    return func_eval_dnd(target, func_handle, pars, *args, **kwargs)
