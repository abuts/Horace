"""Common abstract base class for SQW and DnD objects (SQWDnDBase).

Ported from Horace MATLAB @SQWDnDBase class
(horace_core/sqw/@SQWDnDBase) and disp2sqw (horace_core/disp2sqw.m).
"""

from __future__ import annotations

import copy
import inspect
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import (
    HoraceError,
    crystal_alignment_info,
)
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.axes_block_base import AxesBlockBase
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.configurations.config_store import warn_with_id
from pyhorace.core.data_op_interface import (
    DataOpArray,
    _is_numeric,
    data_op_interface,
    matlab_size,
)
from pyhorace.core.serializable import Serializable
from pyhorace.sqw.horace3_dnd_interface import horace3_dnd_interface


def _parse_char_options(
    args: Sequence[Any], valid_options: Sequence[str]
) -> Tuple[bool, str, List[bool], List[Any]]:
    """Parse MATLAB-style prefix-matched string flags from *args."""
    flags = [False] * len(valid_options)
    rem: List[Any] = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("-"):
            a_low = arg.lower()
            matched = False
            for idx, opt in enumerate(valid_options):
                if opt.lower().startswith(a_low) or a_low == opt.lower().lstrip("-"):
                    flags[idx] = True
                    matched = True
                    break
            if not matched:
                return False, f"Unrecognised option: {arg!r}", flags, rem
        else:
            rem.append(arg)
    return True, "", flags, rem


def _call_user_func(func: Callable[..., Any], *coord_arrays: Any, pars: Any = None) -> Any:
    """Invoke a user function handling both MATLAB-style (coords..., pars) and unpacked (*coords, *pars)."""
    if pars is None or (isinstance(pars, (list, tuple)) and len(pars) == 0):
        try:
            return func(*coord_arrays)
        except TypeError:
            return func(*coord_arrays, [])
    try:
        return func(*coord_arrays, pars)
    except TypeError:
        if isinstance(pars, (list, tuple)):
            return func(*coord_arrays, *pars)
        raise


def disp2sqw(*args: Any, **kwargs: Any) -> np.ndarray:
    """Calculate spectral weight on grids of (qh, qk, ql, en) for a dispersion relation.

    Ported from ``horace_core/disp2sqw.m``.
    Supports:
    - ``disp2sqw(q, dispreln, pars, fwhh)`` (4 args: ``q = [qh, qk, ql, en]``)
    - ``disp2sqw(q, en, dispreln, pars, fwhh)`` (5 args: ``q = [qh, qk, ql]``, ``expand_qe = True``)
    - ``disp2sqw(qh, qk, ql, en, [dispreln, pars, fwhh])`` (5 args from ``sqw_eval``)
    - ``disp2sqw(qh, qk, ql, en, dispreln, pars, fwhh)`` (7 args)
    """
    if len(args) == 7:
        expand_qe = False
        qh, qk, ql, en = args[0], args[1], args[2], args[3]
        dispreln, pars, fwhh = args[4], args[5], args[6]
    elif len(args) == 5:
        if isinstance(args[4], (list, tuple)) and len(args[4]) == 3 and callable(args[4][0]):
            expand_qe = False
            qh, qk, ql, en = args[0], args[1], args[2], args[3]
            dispreln, pars, fwhh = args[4][0], args[4][1], args[4][2]
        else:
            expand_qe = True
            q_seq = args[0]
            qh, qk, ql = q_seq[0], q_seq[1], q_seq[2]
            en = args[1]
            dispreln, pars, fwhh = args[2], args[3], args[4]
    elif len(args) == 4:
        expand_qe = False
        q_seq = args[0]
        qh, qk, ql, en = q_seq[0], q_seq[1], q_seq[2], q_seq[3]
        dispreln, pars, fwhh = args[1], args[2], args[3]
    else:
        raise HoraceError(
            "HORACE:disp2sqw:invalid_argument",
            "Check number of input arguments",
        )

    res = _call_user_func(dispreln, qh, qk, ql, pars=pars)
    if isinstance(res, tuple):
        wdisp, sf = res[0], res[1]
    else:
        wdisp = res
        sf = None

    if not isinstance(wdisp, (list, tuple)):
        wdisp_list = [np.asarray(wdisp, dtype=float)]
    else:
        wdisp_list = [np.asarray(w, dtype=float) for w in wdisp]

    if sf is None:
        sf_list = [np.ones_like(w) for w in wdisp_list]
    elif not isinstance(sf, (list, tuple)):
        sf_list = [np.asarray(sf, dtype=float)]
    else:
        sf_list = [np.asarray(s, dtype=float) for s in sf]

    qh_arr = np.asarray(qh, dtype=float)
    en_arr = np.asarray(en, dtype=float)
    if not expand_qe:
        weight = np.zeros(qh_arr.size, dtype=float)
        en_flat = en_arr.ravel(order="F")
        for w_mode, s_mode in zip(wdisp_list, sf_list):
            w_flat = w_mode.ravel(order="F")
            s_flat = s_mode.ravel(order="F")
            if callable(fwhh):
                try:
                    sig = np.asarray(fwhh(w_flat), dtype=float).ravel(order="F") / np.sqrt(np.log(256.0))
                    weight = weight + s_flat * np.exp(-((w_flat - en_flat) ** 2) / (2.0 * sig**2)) / (
                        sig * np.sqrt(2.0 * np.pi)
                    )
                except TypeError:
                    weight = weight + s_flat * np.asarray(fwhh(en_flat, w_flat), dtype=float).ravel(order="F")
            else:
                sig = float(fwhh) / np.sqrt(np.log(256.0))
                weight = weight + s_flat * np.exp(-((w_flat - en_flat) ** 2) / (2.0 * sig**2)) / (
                    sig * np.sqrt(2.0 * np.pi)
                )
        return weight.reshape(qh_arr.shape, order="F")
    else:
        nq = qh_arr.size
        ne = en_arr.size
        weight = np.zeros((nq, ne), dtype=float)
        en_mat = np.tile(en_arr.ravel(order="F").reshape(1, ne), (nq, 1))
        for w_mode, s_mode in zip(wdisp_list, sf_list):
            w_col = w_mode.ravel(order="F").reshape(nq, 1)
            s_col = s_mode.ravel(order="F").reshape(nq, 1)
            if callable(fwhh):
                try:
                    sig_col = np.asarray(fwhh(w_col), dtype=float).reshape(nq, 1) / np.sqrt(np.log(256.0))
                    sig_mat = np.tile(sig_col, (1, ne))
                    g_mat = np.exp(-((w_col - en_mat) ** 2) / (2.0 * sig_mat**2)) / (
                        sig_mat * np.sqrt(2.0 * np.pi)
                    )
                except TypeError:
                    g_mat = np.asarray(fwhh(en_mat, w_col), dtype=float)
            else:
                sig = float(fwhh) / np.sqrt(np.log(256.0))
                g_mat = np.exp(-((w_col - en_mat) ** 2) / (2.0 * sig**2)) / (sig * np.sqrt(2.0 * np.pi))
            weight = weight + s_col * g_mat
        return weight


def _disp2sqw_legacy(
    q: Sequence[np.ndarray],
    dispreln: Callable[..., Any],
    pars: Any,
    fwhh: float,
) -> np.ndarray:
    """Calculate spectral weight on grids of (qh, qk, ql, en) for a dispersion relation.

    Ported from ``horace_core/disp2sqw.m``.
    """
    sig = float(fwhh) / np.sqrt(np.log(256.0))
    qh, qk, ql, en = q[0], q[1], q[2], q[3]

    res = _call_user_func(dispreln, qh, qk, ql, pars=pars)
    if isinstance(res, tuple):
        wdisp, sf = res[0], res[1]
    else:
        wdisp = res
        sf = None

    if not isinstance(wdisp, (list, tuple)):
        wdisp_list = [np.asarray(wdisp, dtype=float)]
    else:
        wdisp_list = [np.asarray(w, dtype=float) for w in wdisp]

    if sf is None:
        sf_list = [np.ones_like(w) for w in wdisp_list]
    elif not isinstance(sf, (list, tuple)):
        sf_list = [np.asarray(sf, dtype=float)]
    else:
        sf_list = [np.asarray(s, dtype=float) for s in sf]

    en_arr = np.asarray(en, dtype=float)
    weight = np.zeros_like(en_arr, dtype=float)
    norm_factor = sig * np.sqrt(2.0 * np.pi)
    for w_mode, s_mode in zip(wdisp_list, sf_list):
        w_b = w_mode.reshape(en_arr.shape, order="F") if w_mode.size == en_arr.size else w_mode
        s_b = s_mode.reshape(en_arr.shape, order="F") if s_mode.size == en_arr.size else s_mode
        weight = weight + s_b * np.exp(-((en_arr - w_b) ** 2) / (2.0 * sig**2)) / norm_factor
    return weight


class SQWDnDBase(
    data_op_interface, horace3_dnd_interface, Serializable, ABC
):
    """Abstract base class common to SQW and DnD objects."""

    # ------------------------------------------------------------------
    # Abstract methods implemented by DnDBase / sqw
    # ------------------------------------------------------------------
    @abstractmethod
    def has_pixels(self) -> bool:
        """Return True if object contains pixel data."""

    @abstractmethod
    def dimensions(self, nargout: Optional[int] = None) -> Any:
        """Return (nd, sz) dimensionality and extent along each dimension."""

    @abstractmethod
    def calculate_q_bins(self, nargout: Optional[int] = None) -> Any:
        """Calculate qh, qk, ql bin centers in rlu."""

    @abstractmethod
    def calculate_qw_bins(
        self, boundaries: bool = False, edges: bool = False, nargout: Optional[int] = None
    ) -> Any:
        """Calculate qh, qk, ql, en bin centers in rlu and meV."""

    # ------------------------------------------------------------------
    # Shared SQWDnDBase methods
    # ------------------------------------------------------------------
    def copy(self) -> "SQWDnDBase":
        """Return a deep copy of this object."""
        res = copy.deepcopy(self)
        if getattr(res, "_obj_holder", None) is not None:
            res._obj_holder = None
        return res

    def equal_to_tol(
        self,
        other: Any,
        *args: Any,
        return_msg: bool = False,
        nargout: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Compare two SQWDnDBase objects within tolerance."""
        if nargout is None:
            try:
                caller_name = sys._getframe(1).f_code.co_name
            except ValueError:
                caller_name = ""
            if caller_name == "compare_values":
                nargout = 2
            else:
                nargout = _infer_nargout(default=1)
        if type(self) is not type(other):
            iseq = False
            mess = f"Objects being compared are not of the same type ({type(self).__name__} vs {type(other).__name__})"
        else:
            iseq, mess = Serializable.equal_to_tol(self, other, *args, **kwargs)
        if return_msg or nargout >= 2:
            return bool(iseq), mess
        return bool(iseq)

    @staticmethod
    def dimensions_match(
        objs: Union["SQWDnDBase", Sequence["SQWDnDBase"], np.ndarray],
        match_dim: Optional[int] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Check that all objects in an array have the same dimensionality."""
        if nargout is None:
            nargout = _infer_nargout(default=2)
        if isinstance(objs, SQWDnDBase):
            seq = [objs]
        elif isinstance(objs, (np.ndarray, DataOpArray)):
            seq = list(objs.ravel(order="F"))
        else:
            seq = list(objs)

        dims = []
        for item in seq:
            d = item.dimensions(nargout=2)
            nd = d[0] if isinstance(d, tuple) else int(d)
            dims.append(nd)

        if len(set(dims)) > 1:
            equal_dims = False
            nd_out = None
        else:
            nd_out = dims[0] if dims else 0
            if match_dim is None:
                equal_dims = True
            else:
                equal_dims = nd_out == int(match_dim)

        if nargout <= 1:
            return equal_dims
        return equal_dims, nd_out

    def calculate_qsqr_bins(self, nargout: Optional[int] = None) -> np.ndarray:
        """Calculate |Q|^2 at bin centers in Angstrom^-2."""
        q = self.calculate_q_bins(nargout=1)
        return q[0] ** 2 + q[1] ** 2 + q[2] ** 2

    def calculate_qsqr_w_bins(
        self,
        *args: Any,
        boundaries: bool = False,
        edges: bool = False,
        nargout: Optional[int] = None,
    ) -> List[Any]:
        """Calculate |Q|^2 and energy at bin centers (or boundaries/edges)."""
        for arg in args:
            if isinstance(arg, str):
                a_low = arg.lower().lstrip("-")
                if a_low.startswith("bound"):
                    boundaries = True
                elif a_low.startswith("edge"):
                    edges = True
        qw = self.calculate_qw_bins(boundaries=boundaries, edges=edges, nargout=1)
        if boundaries and isinstance(qw[0], list):
            qsqr = [
                q1**2 + q2**2 + q3**2
                for q1, q2, q3 in zip(qw[0], qw[1], qw[2])
            ]
            return [qsqr, qw[3]]
        qsqr = qw[0] ** 2 + qw[1] ** 2 + qw[2] ** 2
        return [qsqr, qw[3]]

    def _sqw_eval_nopix_impl(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> "SQWDnDBase":
        """Evaluate 4D S(Q, w) model function at bin centers (ignoring pixels).

        Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval_nopix.m``.
        """
        opts = options or {}
        if hasattr(self, "data") and getattr(self, "data") is not self:
            self.data = self.data.sqw_eval_nopix(sqwfunc, pars, opts)
            return self
        all_bins = bool(opts.get("all_bins", opts.get("all", False)))
        qw = self.calculate_qw_bins(nargout=1)
        empty_npix = self.npix == 0
        has_any_pix = not np.all(empty_npix)
        if not all_bins and has_any_pix:
            keep = (~empty_npix).ravel(order="F")
            qw_eval = [q[keep].reshape(-1, 1) for q in qw]
            val_sub = np.asarray(
                _call_user_func(sqwfunc, *qw_eval, pars=pars), dtype=float
            ).ravel(order="F")
            s_flat = np.zeros(self.s.size, dtype=float)
            s_flat[keep] = val_sub
            val_arr = s_flat.reshape(self.s.shape, order="F")
        else:
            qw_eval = [q.reshape(-1, 1) for q in qw]
            val = _call_user_func(sqwfunc, *qw_eval, pars=pars)
            val_arr = np.asarray(val, dtype=float).reshape(self.s.shape, order="F")
        self.s = val_arr
        self.e = np.zeros_like(self.s)
        if all_bins:
            self.npix = np.ones_like(self.npix)
        return self

    def sqw_eval(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "SQWDnDBase":
        """Calculate sqw for a model scattering function.

        Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval.m``.
        """
        from pyhorace.coord_transf.bmatrix import _infer_nargout
        from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)

        rem_args = list(args)
        if len(rem_args) > 0 and isinstance(rem_args[0], PageOp_sqw_eval):
            op = rem_args.pop(0)
        else:
            op = PageOp_sqw_eval()

        sqwfunc_parsed, pars_cell, opts = self._parse_eval_options(
            sqwfunc, pars, *rem_args, **kwargs
        )
        outfiles = opts["outfile"]
        if bool(opts["filebacked"]) and nargout > 0:
            opts["outfile"] = ""

        has_pix = hasattr(self, "has_pixels") and bool(self.has_pixels())
        if not has_pix and bool(opts["all_bins"]):
            for dim_size in self.s.shape:
                if dim_size == 0:
                    raise HoraceError(
                        "HORACE:sqw_eval:invalid_argument",
                        "Empty object cannot be evaluated using option 'all'",
                    )

        if has_pix and not bool(opts["nopix"]):
            wout = self.sqw_eval_pix(
                sqwfunc_parsed,
                pars_cell,
                opts,
                bool(opts["average"]),
                outfiles,
                op=op,
            )
            return wout
        else:
            wout = self.copy()
            if has_pix:
                from pyhorace.data.pixels.pixel_data_base import PixelDataBase

                wout.pix = PixelDataBase.create()
            wout = wout.sqw_eval_nopix(sqwfunc_parsed, pars_cell, opts)
            if outfiles:
                wout.save(outfiles)
            return wout

    def disp2sqw_eval(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "SQWDnDBase":
        """Calculate sqw for a model dispersion relation.

        Ported from ``horace_core/sqw/@SQWDnDBase/disp2sqw_eval.m``.
        """
        from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval

        pars_list = list(pars) if isinstance(pars, (list, tuple)) else ([pars] if pars is not None else [])
        meta_param = [dispreln, pars_list, fwhh]
        op = PageOp_sqw_eval(op_name="disp2sqw_eval")
        return self.sqw_eval(disp2sqw, meta_param, op, *args, **kwargs)

    def _legacy_sqw_eval_nopix_unused(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> "SQWDnDBase":
        """Evaluate 4D S(Q, w) model function at bin centers (ignoring pixels)."""
        opts = options or {}
        all_bins = bool(opts.get("all_bins", False))
        qw = self.calculate_qw_bins(nargout=1)
        val = _call_user_func(sqwfunc, qw[0], qw[1], qw[2], qw[3], pars=pars)
        val_arr = np.asarray(val, dtype=float).reshape(self.s.shape, order="F")
        if not all_bins:
            val_arr = val_arr.copy()
            val_arr[self.npix == 0] = 0.0
        self.s = val_arr
        self.e = np.zeros_like(self.s)
        return self

    def sqw_eval(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "SQWDnDBase":
        """Evaluate S(Q, w) function on a DnD or SQW object."""
        opts = self._parse_eval_options(*args, **kwargs)
        wout = self.copy()
        wout = wout.sqw_eval_nopix(sqwfunc, pars, opts)
        return wout

    def disp2sqw_eval(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: float,
        all_bins: bool = False,
    ) -> "SQWDnDBase":
        """Evaluate dispersion relation spectral weight on a single object."""
        qw = self.calculate_qw_bins(nargout=1)
        val = disp2sqw(qw, dispreln, pars, fwhh)
        val_arr = np.asarray(val, dtype=float).reshape(self.s.shape, order="F")
        if not all_bins:
            val_arr = val_arr.copy()
            val_arr[self.npix == 0] = 0.0
        wout = self.copy()
        wout.s = val_arr
        wout.e = np.zeros_like(wout.s)
        if all_bins:
            wout.npix = np.ones_like(wout.npix)
        return wout

    def disp2sqw(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: Any,
        *args: Any,
    ) -> "SQWDnDBase":
        """Evaluate dispersion relation on a DnD or SQW object (ignoring pixels).

        Ported from ``horace_core/sqw/@SQWDnDBase/disp2sqw.m``.
        """
        wout = self.copy()
        if hasattr(wout, "data") and getattr(wout, "data") is not wout:
            wout.data = wout.data.disp2sqw(dispreln, pars, fwhh, *args)
            return wout
        return wout.disp2sqw_eval(dispreln, pars, fwhh, *args)

    def _LegacyDisp2SqwUnused(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: float,
        *args: Any,
    ) -> "SQWDnDBase":
        """Evaluate dispersion relation on a DnD or SQW object to produce spectral weight."""
        all_bins = any(
            isinstance(a, str) and a.lower().lstrip("-").startswith("al")
            for a in args
        )
        return self.disp2sqw_eval(dispreln, pars, fwhh, all_bins=all_bins)

    def mask_points(
        self,
        *args: Any,
        keep: Any = None,
        remove: Any = None,
        mask: Any = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Determine points to keep/mask on the basis of ranges and mask array.

        Ported from ``@SQWDnDBase/mask_points.m``.
        """
        if nargout is None:
            nargout = _infer_nargout(default=3)

        # Parse MATLAB-style key-value arguments ('keep', xkeep, 'remove', xremove, 'mask', mask)
        idx = 0
        while idx < len(args):
            key = args[idx]
            if isinstance(key, str):
                k_low = key.lower().lstrip("-")
                if idx + 1 >= len(args):
                    raise HoraceError(
                        "HORACE:mask_points:invalid_argument",
                        f"Option '{key}' requires a value",
                    )
                val = args[idx + 1]
                if k_low.startswith("keep"):
                    keep = val
                elif k_low.startswith("remove") or k_low.startswith("rem"):
                    remove = val
                elif k_low.startswith("mask"):
                    mask = val
                else:
                    raise HoraceError(
                        "HORACE:mask_points:invalid_argument",
                        f"Unrecognized option '{key}'",
                    )
                idx += 2
            else:
                raise HoraceError(
                    "HORACE:mask_points:invalid_argument",
                    f"Expected option string, got {type(key).__name__}",
                )

        nd, sz = self.dimensions(nargout=2)
        if nd == 0:
            mess = "Cannot mask a zero dimensional object"
            if nargout <= 1:
                warn_with_id("HORACE:mask_points:invalid_argument", mess)
                return np.array([True], dtype=bool)
            return np.array([], dtype=bool), False, mess

        sel = np.ones(self.s.shape, dtype=bool)
        if keep is None and remove is None and mask is None:
            warn_with_id(
                "HORACE:mask_points:invalid_argument",
                "No masking ranges or mask array have been given.",
            )
            if nargout <= 1:
                return sel
            return sel, True, ""

        # Compute bin centers along display axes
        p_bins = self.p
        dax = self.dax
        centers_per_dim = [
            0.5 * (p_bins[d - 1][:-1] + p_bins[d - 1][1:]) for d in dax
        ]
        if nd == 1:
            grids = [centers_per_dim[0].reshape(self.s.shape, order="F")]
        else:
            mg = np.meshgrid(*centers_per_dim, indexing="ij")
            grids = [g.reshape(self.s.shape, order="F") for g in mg]

        def _parse_range_boxes(rng_in: Any) -> Optional[np.ndarray]:
            arr = np.asarray(rng_in, dtype=float)
            if arr.size == 0:
                return None
            # For nd dimensions: can be 1D of length 2*nd, or 2D of shape (M, 2*nd) or (nd, 2)
            if arr.ndim == 1:
                if arr.size != 2 * nd:
                    return None
                arr = arr.reshape((1, 2 * nd))
            elif arr.ndim == 2:
                if arr.shape[1] == 2 * nd:
                    pass
                elif arr.shape == (nd, 2):
                    arr = arr.reshape((1, 2 * nd))
                elif arr.shape == (2, nd) and nd != 2:
                    arr = arr.T.reshape((1, 2 * nd))
                else:
                    return None
            else:
                return None
            return arr

        if keep is not None:
            boxes = _parse_range_boxes(keep)
            if boxes is None:
                mess = f"xkeep must specify min/max ranges for each of the {nd} dimensions"
                if nargout <= 1:
                    raise HoraceError("HORACE:mask_points:invalid_argument", mess)
                return np.array([], dtype=bool), False, mess
            sel_keep = np.zeros(self.s.shape, dtype=bool)
            for row in boxes:
                in_box = np.ones(self.s.shape, dtype=bool)
                for d_idx in range(nd):
                    lo = min(row[2 * d_idx], row[2 * d_idx + 1])
                    hi = max(row[2 * d_idx], row[2 * d_idx + 1])
                    in_box &= (grids[d_idx] >= lo) & (grids[d_idx] <= hi)
                sel_keep |= in_box
            sel &= sel_keep

        if remove is not None:
            boxes = _parse_range_boxes(remove)
            if boxes is None:
                mess = f"xremove must specify min/max ranges for each of the {nd} dimensions"
                if nargout <= 1:
                    raise HoraceError("HORACE:mask_points:invalid_argument", mess)
                return np.array([], dtype=bool), False, mess
            sel_rem = np.zeros(self.s.shape, dtype=bool)
            for row in boxes:
                in_box = np.ones(self.s.shape, dtype=bool)
                for d_idx in range(nd):
                    lo = min(row[2 * d_idx], row[2 * d_idx + 1])
                    hi = max(row[2 * d_idx], row[2 * d_idx + 1])
                    in_box &= (grids[d_idx] >= lo) & (grids[d_idx] <= hi)
                sel_rem |= in_box
            sel &= ~sel_rem

        if mask is not None:
            msk_arr = np.asarray(mask)
            if msk_arr.dtype != bool and not _is_numeric(msk_arr):
                mess = "Mask array must be a logical or numeric array"
                if nargout <= 1:
                    raise HoraceError("HORACE:mask_points:invalid_argument", mess)
                return np.array([], dtype=bool), False, mess
            if msk_arr.size != self.s.size:
                mess = "Mask array must have the same number of elements as the data array"
                if nargout <= 1:
                    raise HoraceError("HORACE:mask_points:invalid_argument", mess)
                return np.array([], dtype=bool), False, mess
            sel &= msk_arr.astype(bool).reshape(self.s.shape, order="F")

        if nargout <= 1:
            return sel
        return sel, True, ""

    @staticmethod
    def _parse_eval_options(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        all_bins = bool(kwargs.get("all", False) or kwargs.get("all_bins", False))
        average = bool(kwargs.get("average", False) or kwargs.get("ave", False))
        filebacked = bool(kwargs.get("filebacked", False))
        outfile = kwargs.get("outfile", "")

        idx = 0
        while idx < len(args):
            arg = args[idx]
            if isinstance(arg, str):
                a_low = arg.lower().lstrip("-")
                if a_low in ("all", "al"):
                    all_bins = True
                    idx += 1
                    continue
                if a_low in ("average", "ave", "av"):
                    average = True
                    idx += 1
                    continue
                if a_low.startswith("fileback"):
                    filebacked = True
                    idx += 1
                    continue
                if a_low.startswith("outfile"):
                    if idx + 1 < len(args):
                        outfile = str(args[idx + 1])
                        idx += 2
                        continue
            idx += 1
        return {
            "all_bins": all_bins,
            "average": average,
            "filebacked": filebacked,
            "outfile": outfile,
        }

    @staticmethod
    def parse_change_crystal_arguments(*args: Any) -> crystal_alignment_info:
        """Validate and normalize change_crystal arguments to crystal_alignment_info."""
        if len(args) == 0:
            raise HoraceError(
                "HORACE:change_crystal:invalid_argument",
                "change_crystal requires crystal_alignment_info or lattice parameters",
            )
        if len(args) == 1 and isinstance(args[0], crystal_alignment_info):
            return copy.deepcopy(args[0])
        if len(args) >= 2:
            alatt = args[0]
            angdeg = args[1]
            info = crystal_alignment_info(alatt, angdeg)
            if len(args) >= 3 and args[2] is not None:
                info.rotmat = args[2]
            return info
        raise HoraceError(
            "HORACE:change_crystal:invalid_argument",
            "Invalid arguments to change_crystal",
        )

    def save(self, *args: Any) -> "SQWDnDBase":
        """Save a SQW or DnD object to a binary file using ``sqw_formats_factory``.

        Ported from ``@SQWDnDBase/save.m``.
        """
        import os
        from pyhorace.io.horace_binfile_interface import horace_binfile_interface
        from pyhorace.io.sqw_formats_factory import sqw_formats_factory

        full_fn = getattr(self, "full_filename", "")
        file_internal = isinstance(full_fn, str) and len(full_fn) > 0

        ldw = None
        rem_args = list(args)
        for idx, arg in enumerate(rem_args):
            if isinstance(arg, horace_binfile_interface):
                ldw = arg
                rem_args.pop(idx)
                break

        if len(rem_args) == 0:
            if not file_internal:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "Must give filename if internal filename is not present",
                )
            file = full_fn
        elif len(rem_args) == 1:
            if not isinstance(rem_args[0], (str, os.PathLike)) or not str(rem_args[0]):
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    "Filename must be a non-empty string",
                )
            file = str(rem_args[0])
        else:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                "Too many input arguments provided to save",
            )

        if ldw is None or not ldw.is_Compatible(self):
            ldw = sqw_formats_factory.instance().get_pref_access(self)

        ldw = ldw.init(self, file)
        try:
            ldw.put_sqw()
        finally:
            ldw.delete()
        return self

    # ------------------------------------------------------------------
    # Cut input parsing (@SQWDnDBase/process_and_validate_cut_inputs)
    # ------------------------------------------------------------------
    def process_and_validate_cut_inputs(
        self, *args: Any, nargout: Optional[int] = None
    ) -> Any:
        """Parse and validate cut parameters (port of ``@SQWDnDBase/private/cut_parse_inputs_.m``).

        Can be invoked either as an instance method ``obj.process_and_validate_cut_inputs(return_cut, *args)``
        or via class/static style ``SQWDnDBase.process_and_validate_cut_inputs(in_obj, return_cut, *args)``
        or ``tester.process_and_validate_cut_inputs(in_obj, return_cut, *args)``.
        """
        import os
        from pyhorace.coord_transf.line_proj import line_proj
        from pyhorace.coord_transf.symop import Symop, SymopArray, SymopIdentity

        if nargout is None:
            nargout = _infer_nargout(default=4)

        if len(args) >= 2 and isinstance(args[0], SQWDnDBase):
            in_obj = args[0]
            return_cut = bool(args[1])
            cut_args = list(args[2:])
        elif len(args) >= 1:
            in_obj = self
            return_cut = bool(args[0])
            cut_args = list(args[1:])
        else:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Insufficient arguments passed to process_and_validate_cut_inputs",
            )

        d_res = in_obj.dimensions(nargout=1) if callable(getattr(in_obj, "dimensions", None)) else in_obj.dimensions
        ndims = int(d_res[0] if isinstance(d_res, tuple) else d_res)
        has_pix_fn = getattr(in_obj, "has_pixels", None)
        has_pix = bool(has_pix_fn()) if callable(has_pix_fn) else False
        if ndims == 0 and not has_pix:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Cannot take a cut from a zero-dimensional (d0d or 0D sqw) object",
            )

        # Parse recognized string flags ('-pix', '-nopix', '-parallel', '-save');
        # any unrecognized '-flag' is left in par to trigger the extra-binning-args error
        # matching parse_Args(..., '-keys_at_end', false) in cut_parse_inputs_.m.
        pix_flag = True
        pix_present = False
        nopix_flag = False
        nopix_present = False
        parallel_flag = False
        save_flag = False
        save_present = False
        par: List[Any] = []

        for arg in cut_args:
            if isinstance(arg, str) and arg.startswith("-"):
                a_low = arg.lower()
                if "-pix".startswith(a_low) and len(a_low) >= 2:
                    pix_flag = True
                    pix_present = True
                elif "-nopix".startswith(a_low) and len(a_low) >= 2:
                    nopix_flag = True
                    nopix_present = True
                elif "-parallel".startswith(a_low) and len(a_low) >= 3:
                    parallel_flag = True
                elif "-save".startswith(a_low) and len(a_low) >= 2:
                    save_flag = True
                    save_present = True
                else:
                    par.append(arg)
            else:
                par.append(arg)

        if pix_present and nopix_present:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Options '-pix' and '-nopix' cannot both be given",
            )
        keep_pix = pix_flag and (not nopix_flag)

        # Check for trailing output filename
        outfile = ""
        if (
            len(par) > 0
            and isinstance(par[-1], (str, os.PathLike))
            and str(par[-1])
            and not str(par[-1]).startswith("-")
        ):
            outfile = str(par.pop()).strip()

        # Check for leading projection (aProjectionBase or legacy dict)
        if len(par) > 0 and (
            isinstance(par[0], aProjectionBase) or isinstance(par[0], dict)
        ):
            first = par.pop(0)
            if isinstance(first, aProjectionBase):
                proj = copy.deepcopy(first)
            else:
                proj = line_proj(first)
            npbin_expected = 4
            proj_given = True
        else:
            src_proj = in_obj.data.proj if hasattr(in_obj, "data") else in_obj.proj
            proj = copy.deepcopy(src_proj)
            npbin_expected = ndims
            proj_given = False

        # Check for trailing symmetry argument (Symop, SymopArray, or sequence of them)
        def _is_sym_arg(val: Any) -> bool:
            if isinstance(val, (Symop, SymopArray)):
                return True
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
                seq = list(val.flat) if isinstance(val, np.ndarray) else list(val)
                return any(isinstance(x, (Symop, SymopArray)) for x in seq)
            return False

        if len(par) > 0 and _is_sym_arg(par[-1]):
            sym = SQWDnDBase._check_sym_arg(par.pop())
        else:
            sym = [SymopIdentity()]

        # Validate number of binning arguments
        if len(par) < npbin_expected:
            if proj_given:
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "Must give binning arguments for all four dimensions if new projection axes",
                )
            else:
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "Number of binning arguments must match the number of dimensions of the sqw data being cut",
                )
        elif len(par) > npbin_expected:
            extra_str = " ".join(str(x) for x in par[npbin_expected:])
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                f"Object has {npbin_expected} dimensions but cutting with {len(par)} binning parameters\n"
                f'Unrecognised additional input(s): "{extra_str}" were provided to cut',
            )

        # Validate that all binning arguments are empty or numeric
        for pb in par:
            if pb is None or (isinstance(pb, (list, tuple, np.ndarray)) and len(pb) == 0):
                continue
            if isinstance(pb, (bool, str, dict)) or not _is_numeric(pb):
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "Binning arguments must all be numeric (check also if any optional arguments were mis-spelt)",
                )

        if proj_given:
            proj = proj.copy_proj_param_from_source(in_obj)
            pbin_4d: List[np.ndarray] = []
            for pb in par:
                if pb is None or (isinstance(pb, (list, tuple, np.ndarray)) and np.asarray(pb).size == 0):
                    pbin_4d.append(np.array([], dtype=float))
                else:
                    pbin_4d.append(np.asarray(pb, dtype=float).ravel())
        else:
            img = in_obj.data if hasattr(in_obj, "data") else in_obj
            pbin_4d = [np.array([], dtype=float) for _ in range(4)]
            for k, pax_num in enumerate(img.pax):
                pbin_4d[int(pax_num) - 1] = SQWDnDBase._select_pbin(par[k], img.p[k])
            for k, iax_num in enumerate(img.iax):
                pbin_4d[int(iax_num) - 1] = np.asarray(img.iint[:, k], dtype=float).ravel().copy()

        # Expand multicuts (any 4-element binning descriptor [plo, rdiff, phi, rwidth])
        pbin_list = SQWDnDBase._expand_multicuts(pbin_4d)

        # Validate output / save options
        save_to_file = save_flag or (not save_present and bool(outfile))
        if save_present and not outfile:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Use of '-save' option and/or provision of output file name are not consistent",
            )
        if not save_to_file and not return_cut:
            raise HoraceError(
                "HORACE:cut:invalid_argument",
                "Neither output cut object nor output file requested - routine is not being asked to do anything",
            )
        if save_to_file:
            _, file_ext = os.path.splitext(outfile)
            if len(file_ext) <= 1:
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    f"Output filename '{outfile}' has no extension - check optional arguments",
                )
            try:
                existed = os.path.exists(outfile)
                with open(outfile, "ab"):
                    pass
                if not existed and os.path.exists(outfile):
                    os.remove(outfile)
            except OSError as exc:
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    f"Cannot open output file {outfile}",
                ) from exc

        opt = {
            "keep_pix": bool(keep_pix),
            "parallel": bool(parallel_flag),
            "outfile": outfile,
            "proj_given": bool(proj_given),
        }
        if nargout == 3:
            return proj, pbin_list, opt
        return proj, pbin_list, sym, opt

    process_and_validate_cut_inputs_public = process_and_validate_cut_inputs

    @staticmethod
    def _check_sym_arg(sym: Any) -> List[Any]:
        """Validate and normalize symmetry arguments (port of ``check_sym_arg`` in ``cut_parse_inputs_.m``)."""
        from pyhorace.coord_transf.symop import Symop, SymopArray, SymopIdentity

        if not isinstance(sym, (list, tuple)):
            sym_seq = [sym]
        else:
            sym_seq = list(sym)

        valid_items: List[Any] = []
        for item in sym_seq:
            if item is None or (isinstance(item, (list, tuple, np.ndarray)) and len(item) == 0):
                continue
            if isinstance(item, (Symop, SymopArray)):
                s_item = item
            elif isinstance(item, (list, tuple, np.ndarray)):
                flat = list(item.flat) if isinstance(item, np.ndarray) else list(item)
                if not all(isinstance(x, Symop) for x in flat):
                    raise HoraceError(
                        "HORACE:cut:invalid_argument",
                        "Symmetry argument items must all be Symop objects",
                    )
                s_item = flat[0] if len(flat) == 1 else SymopArray(flat)
            else:
                raise HoraceError(
                    "HORACE:cut:invalid_argument",
                    "Symmetry argument items must all be Symop objects",
                )

            if isinstance(s_item, SymopIdentity):
                continue
            if isinstance(s_item, SymopArray) and all(isinstance(x, SymopIdentity) for x in s_item):
                continue
            valid_items.append(s_item)

        return [SymopIdentity(), *valid_items]

    @staticmethod
    def _select_pbin(upbin: Any, p: np.ndarray) -> np.ndarray:
        """Merge user binning descriptor ``upbin`` with existing plot axis boundaries ``p``
        when cutting without a new projection (port of ``select_pbin`` in ``cut_parse_inputs_.m``).
        """
        eps_s = float(np.finfo(np.float32).eps)
        if upbin is None:
            u = np.array([], dtype=float)
        else:
            u = np.asarray(upbin, dtype=float).ravel()

        p_arr = np.asarray(p, dtype=float).ravel()

        if u.size == 0:
            bin_width = float(p_arr[1] - p_arr[0])
            bin_centers = 0.5 * (p_arr[:-1] + p_arr[1:])
            return np.array([float(bin_centers[0]), bin_width, float(bin_centers[-1])], dtype=float)
        if u.size == 1:
            pstep_given = float(u[0])
            if abs(pstep_given) < eps_s:
                new_axis = p_arr
            else:
                n_steps = int(np.floor((p_arr[-1] - p_arr[0]) / pstep_given))
                paxis_end = float(p_arr[0]) + pstep_given * n_steps
                if float(p_arr[-1]) - paxis_end > 4.0 * eps_s:
                    n_steps += 1
                paxis_end = float(p_arr[0]) + pstep_given * n_steps
                new_axis = np.linspace(float(p_arr[0]), paxis_end, n_steps + 1)
            bin_centers = 0.5 * (new_axis[:-1] + new_axis[1:])
            pbins = new_axis[1:] - new_axis[:-1]
            pstep = float(np.sum(pbins) / pbins.size)
            return np.array([float(bin_centers[0]), pstep, float(bin_centers[-1])], dtype=float)
        if u.size == 2:
            lo = float(p_arr[0]) if np.isneginf(u[0]) else float(u[0])
            hi = float(p_arr[-1]) if np.isposinf(u[1]) else float(u[1])
            return np.array([lo, hi], dtype=float)
        if u.size == 3:
            pstep = float(p_arr[-1] - p_arr[0]) / float(len(p_arr) - 1)
            pcent_lo = float(0.5 * (p_arr[0] + p_arr[1]))
            pcent_hi = float(0.5 * (p_arr[-2] + p_arr[-1]))
            if abs(float(u[1])) < eps_s:
                lo = pcent_lo if np.isneginf(u[0]) else float(u[0])
                hi = pcent_hi if np.isposinf(u[2]) else float(u[2])
                return np.array([lo, pstep, hi], dtype=float)
            lo = float(p_arr[0]) if np.isneginf(u[0]) else float(u[0])
            hi = float(p_arr[-1]) if np.isposinf(u[2]) else float(u[2])
            return np.array([lo, float(u[1]), hi], dtype=float)
        return u.copy()

    @staticmethod
    def _expand_multicuts(pbin_4d: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Expand 4-element multicut descriptors ``[plo, rdiff, phi, rwidth]`` into a list of 4-axis pbin lists."""
        axis_options: List[List[np.ndarray]] = []
        for pb in pbin_4d:
            arr = np.asarray(pb, dtype=float).ravel()
            if arr.size == 4:
                plo, rdiff, phi, rwidth = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                if plo > phi or rdiff <= 0.0 or rwidth <= 0.0:
                    raise HoraceError(
                        "HORACE:cut:invalid_argument",
                        f"Invalid multicut binning descriptor: {arr.tolist()}",
                    )
                n_cuts = int(np.floor((phi - plo) / rdiff))
                if abs(phi - plo - n_cuts * rdiff) > 4.0 * float(np.finfo(float).eps):
                    n_cuts += 1
                cuts_for_ax = [
                    np.array(
                        [
                            plo + j * rdiff - 0.5 * rwidth,
                            plo + j * rdiff + 0.5 * rwidth,
                        ],
                        dtype=float,
                    )
                    for j in range(n_cuts + 1)
                ]
                axis_options.append(cuts_for_ax)
            else:
                axis_options.append([arr])

        # Build cartesian product in Fortran/MATLAB column-major order across axes
        expanded: List[List[np.ndarray]] = [[]]
        for ax_cuts in axis_options:
            new_expanded: List[List[np.ndarray]] = []
            for cut_item in ax_cuts:
                for prev in expanded:
                    new_expanded.append([*prev, cut_item.copy()])
            expanded = new_expanded
        return expanded


    def sqw_eval_nopix(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        options: Optional[Dict[str, Any]] = None,
    ) -> "SQWDnDBase":
        """Evaluate 4D S(Q, w) model function at bin centers (ignoring pixels).

        Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval_nopix.m``.
        """
        opts = options or {}
        if hasattr(self, "data") and getattr(self, "data") is not self:
            self.data = self.data.sqw_eval_nopix(sqwfunc, pars, opts)
            return self
        all_bins = bool(opts.get("all_bins", opts.get("all", False)))
        qw = self.calculate_qw_bins(nargout=1)
        empty_npix = self.npix == 0
        has_any_pix = not np.all(empty_npix)
        if not all_bins and has_any_pix:
            keep = (~empty_npix).ravel(order="F")
            qw_eval = [q[keep].reshape(-1, 1) for q in qw]
            val_sub = np.asarray(
                _call_user_func(sqwfunc, *qw_eval, pars=pars), dtype=float
            ).ravel(order="F")
            s_flat = np.zeros(self.s.size, dtype=float)
            s_flat[keep] = val_sub
            val_arr = s_flat.reshape(self.s.shape, order="F")
        else:
            qw_eval = [q.reshape(-1, 1) for q in qw]
            val = _call_user_func(sqwfunc, *qw_eval, pars=pars)
            val_arr = np.asarray(val, dtype=float).reshape(self.s.shape, order="F")
        self.s = val_arr
        self.e = np.zeros_like(self.s)
        if all_bins:
            self.npix = np.ones_like(self.npix)
        return self

    def parse_eval_args(
        self,
        func_handle: Callable[..., Any],
        pars: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Callable[..., Any], List[Any], Dict[str, Any]]:
        """Parser for ``sqw_eval`` and ``sqw_op`` algorithms input parameters.

        Ported from ``horace_core/sqw/@SQWDnDBase/private/parse_eval_args_.m``.
        """
        from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval

        if not callable(func_handle):
            raise HoraceError(
                "HORACE:sqw_eval:invalid_argument",
                "func_handle must be a callable function handle",
            )

        rem_args = list(args)
        the_page_op = None
        if len(rem_args) > 0 and isinstance(rem_args[0], PageOp_sqw_eval):
            the_page_op = rem_args.pop(0)

        all_flag = bool(kwargs.get("all", False) or kwargs.get("all_bins", False))
        ave_flag = bool(kwargs.get("average", False) or kwargs.get("ave", False))
        filebacked_flag = bool(kwargs.get("filebacked", False))
        combine_flag = bool(kwargs.get("combine", False))
        nopix_flag = bool(kwargs.get("nopix", False))
        test_inputs = bool(kwargs.get("test_input_parsing", False))
        outfile = kwargs.get("outfile", "")

        idx = 0
        while idx < len(rem_args):
            arg = rem_args[idx]
            if isinstance(arg, str):
                a_low = arg.lower()
                if a_low.startswith("-"):
                    flag = a_low.lstrip("-")
                    if "all".startswith(flag) and len(flag) >= 2:
                        all_flag = True
                        idx += 1
                        continue
                    if "average".startswith(flag) and len(flag) >= 2:
                        ave_flag = True
                        idx += 1
                        continue
                    if "filebacked".startswith(flag) and len(flag) >= 2:
                        filebacked_flag = True
                        idx += 1
                        continue
                    if "combine".startswith(flag) and len(flag) >= 2:
                        combine_flag = True
                        idx += 1
                        continue
                    if "nopix".startswith(flag) and len(flag) >= 2:
                        nopix_flag = True
                        idx += 1
                        continue
                    if "test_input_parsing".startswith(flag) and len(flag) >= 2:
                        test_inputs = True
                        idx += 1
                        continue
                    raise HoraceError(
                        "HORACE:sqw:invalid_argument",
                        f"Unrecognised flag '{arg}'",
                    )
                else:
                    if idx + 1 >= len(rem_args):
                        raise HoraceError(
                            "HORACE:sqw:invalid_argument",
                            f"Parameter '{arg}' requires a value",
                        )
                    val = rem_args[idx + 1]
                    k_low = a_low.strip()
                    if "average".startswith(k_low):
                        ave_flag = bool(val)
                    elif "all".startswith(k_low):
                        all_flag = bool(val)
                    elif "filebacked".startswith(k_low):
                        filebacked_flag = bool(val)
                    elif "combine".startswith(k_low):
                        combine_flag = bool(val)
                    elif "nopix".startswith(k_low):
                        nopix_flag = bool(val)
                    elif "test_input_parsing".startswith(k_low):
                        test_inputs = bool(val)
                    elif "outfile".startswith(k_low):
                        outfile = val
                    else:
                        raise HoraceError(
                            "HORACE:sqw:invalid_argument",
                            f"Unrecognised parameter '{arg}'",
                        )
                    idx += 2
                    continue
            idx += 1

        if isinstance(pars, (list, tuple)):
            pars_cell = list(pars)
        elif pars is None or (isinstance(pars, np.ndarray) and pars.size == 0):
            pars_cell = [[]]
        else:
            pars_cell = [pars]

        if isinstance(outfile, (list, tuple)):
            outfile_cell = [str(x) if x is not None else "" for x in outfile]
        else:
            outfile_cell = [str(outfile) if outfile else ""]

        outfiles_empty = all(len(x) == 0 for x in outfile_cell)
        if not outfiles_empty and len(outfile_cell) != 1:
            raise HoraceError(
                "HORACE:sqw:invalid_argument",
                f"Number of outfiles specified must match number of input objects.\n"
                f"Found {len(outfile_cell)} outfile(s), but 1 sqw object(s).",
            )
        if outfiles_empty and filebacked_flag:
            import os
            import tempfile
            import uuid

            outfile_cell = [
                os.path.join(
                    tempfile.gettempdir(), f"horace_eval_{uuid.uuid4().hex}.sqw"
                )
            ]

        opts = {
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

    def sqw_eval(
        self,
        sqwfunc: Callable[..., Any],
        pars: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "SQWDnDBase":
        """Calculate sqw for a model scattering function over input sqw/dnd object.

        Ported from ``horace_core/sqw/@SQWDnDBase/sqw_eval.m``.
        """
        from pyhorace.coord_transf.bmatrix import _infer_nargout

        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)

        sqwfunc_parsed, pars_cell, opts = self.parse_eval_args(
            sqwfunc, pars, *args, **kwargs
        )
        outfile_str = opts["outfile"][0] if opts["outfile"] else ""
        if not outfile_str or bool(opts["filebacked"]):
            if nargout == 0:
                raise HoraceError(
                    "HORACE:sqw_eval:invalid_argument",
                    f"This method request single output argument. Got: {nargout}",
                )

        has_pix = hasattr(self, "has_pixels") and bool(self.has_pixels())
        if not has_pix and bool(opts["all_bins"]):
            for dim_size in self.s.shape:
                if dim_size == 0:
                    raise HoraceError(
                        "HORACE:sqw_eval:invalid_argument",
                        "Empty object cannot be evaluated using option 'all'",
                    )

        if has_pix and not bool(opts["nopix"]):
            optl = dict(opts)
            optl["outfile"] = outfile_str
            return self.sqw_eval_pix(sqwfunc_parsed, pars_cell, optl)
        else:
            wout = self.copy()
            if has_pix:
                from pyhorace.data.pixels.pixel_data_base import PixelDataBase

                wout.pix = PixelDataBase.create()
            wout = wout.sqw_eval_nopix(sqwfunc_parsed, pars_cell, opts)
            if outfile_str:
                wout.save(outfile_str)
            return wout

    def disp2sqw_eval(
        self,
        dispreln: Callable[..., Any],
        pars: Any,
        fwhh: Any,
        *args: Any,
        **kwargs: Any,
    ) -> "SQWDnDBase":
        """Calculate sqw for a model dispersion relation.

        Ported from ``horace_core/sqw/@SQWDnDBase/disp2sqw_eval.m``.
        """
        all_bins = bool(kwargs.get("all", False) or kwargs.get("all_bins", False))
        ave_pix = bool(kwargs.get("average", False) or kwargs.get("ave", False))
        for arg in args:
            if isinstance(arg, str):
                a_low = arg.lower().lstrip("-")
                if "all".startswith(a_low) and len(a_low) >= 2:
                    all_bins = True
                elif "average".startswith(a_low) and len(a_low) >= 2:
                    ave_pix = True
                else:
                    raise HoraceError(
                        "HORACE:sqw:invalid_arguments",
                        f"Unrecognised option '{arg}'.\n"
                        "Valid values for optional arguments are '-al[l]' or '-av[erage]'.",
                    )

        pars_cell = list(pars) if isinstance(pars, (list, tuple)) else [pars]
        extra_args: List[str] = []
        if ave_pix:
            extra_args.append("-ave")
        if all_bins:
            extra_args.append("-all")

        has_pix = hasattr(self, "has_pixels") and bool(self.has_pixels())
        if has_pix:
            return self.sqw_eval(
                disp2sqw, [dispreln, pars_cell, fwhh], *extra_args
            )
        else:
            wout = self.copy()
            return wout.disp2sqw(dispreln, pars_cell, fwhh, *extra_args)

    def unary_op_manager(self, *args: Any) -> Any:
        """Apply a unary operation to a DnD or SQW object.

        Ported from ``horace_core/sqw/@SQWDnDBase/unary_op_manager.m``.
        """
        unary_op = args[-1]
        target = args[0] if len(args) > 1 and isinstance(args[0], SQWDnDBase) else self
        has_pix = hasattr(target, "has_pixels") and bool(target.has_pixels())
        if has_pix:
            from pyhorace.coord_transf.page_operations.page_op_unary_op import (
                PageOp_unary_op,
            )

            page_op = PageOp_unary_op()
            page_op = page_op.init(target, unary_op)
            return target.apply_op(page_op)
        else:
            return data_op_interface.unary_op_manager(target, unary_op)

    def do_unary_op(self, unary_op: Callable[..., Any]) -> Any:
        """Apply a unary operation to this object."""
        has_pix = hasattr(self, "has_pixels") and bool(self.has_pixels())
        if has_pix:
            return self.unary_op_manager(unary_op)
        sv = self.sigvar()
        sv = unary_op(sv)
        wout = self.copy()
        wout = wout.sigvar_set(sv)
        return wout

    def bose(self, T: float) -> "SQWDnDBase":
        """Correct a dataset for the Bose population factor.

        Ported from ``horace_core/GUI/bose.m``.
        """
        from pyhorace.data.sigvar import sigvar

        def _bose_factor(
            h: np.ndarray,
            k: np.ndarray,
            l: np.ndarray,
            en: np.ndarray,
            temp: float,
        ) -> np.ndarray:
            return 1.0 - np.exp(-11.6044 / float(temp) * np.asarray(en, dtype=np.float64))

        sqw_bose = self.sqw_eval(_bose_factor, [float(T)])
        return data_op_interface.binary_op_manager(self, sqw_bose, sigvar.mtimes, True)


def bose(win: SQWDnDBase, T: float) -> SQWDnDBase:
    """Correct a dataset for the Bose population factor (port of ``bose.m``)."""
    return win.bose(T)
