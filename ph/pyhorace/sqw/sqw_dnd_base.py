"""Common abstract base class for SQW and DnD objects (SQWDnDBase).

Ported from Horace MATLAB @SQWDnDBase class
(horace_core/sqw/@SQWDnDBase) and disp2sqw (horace_core/disp2sqw.m).
"""

from __future__ import annotations

import copy
import inspect
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
from pyhorace.sqw.dnd_plot_interface import dnd_plot_interface
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


def disp2sqw(
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
    data_op_interface, horace3_dnd_interface, dnd_plot_interface, Serializable, ABC
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
        return copy.deepcopy(self)

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

    def sqw_eval_nopix(
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
