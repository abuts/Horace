"""2-dimensional DnD class (`d2d`), translated from `horace_core/sqw/@d2d/`."""

from __future__ import annotations

import math
import warnings
from typing import Any, ClassVar, Dict

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import DataOpArray, _is_numeric
from pyhorace.core.serializable import register_serializable
from pyhorace.sqw.dnd_base import DnDBase


@register_serializable
class d2d(DnDBase):
    """2-dimensional DnD object, translated from `horace_core/sqw/@d2d/d2d.m`."""

    NUM_DIMS: ClassVar[int] = 2

    def IX_dataset_2d(self) -> Dict[str, Any]:
        """Convert 2D DnD object into IX_dataset_2d dictionary representation."""
        title_main, title_pax, *_ = self.data_plot_titles()
        s_axis = "Intensity"
        axis_1 = title_pax[0] if len(title_pax) > 0 else ""
        axis_2 = title_pax[1] if len(title_pax) > 1 else ""

        nopix = self._npix == 0
        signal = np.array(self._s, dtype=float, copy=True)
        signal[nopix] = np.nan
        err = np.sqrt(np.maximum(self._e, 0.0))
        err[nopix] = 0.0

        p_list = self.p
        p1 = p_list[0].copy() if len(p_list) > 0 else np.array([], dtype=float)
        p2 = p_list[1].copy() if len(p_list) > 1 else np.array([], dtype=float)

        if np.array_equal(self.dax, [2, 1]):
            return {
                "title": title_main,
                "signal": signal.T,
                "error": err.T,
                "s_axis": s_axis,
                "x": p2,
                "x_axis": axis_1,
                "x_distribution": True,
                "y": p1,
                "y_axis": axis_2,
                "y_distribution": True,
            }
        return {
            "title": title_main,
            "signal": signal,
            "error": err,
            "s_axis": s_axis,
            "x": p1,
            "x_axis": axis_1,
            "x_distribution": True,
            "y": p2,
            "y_axis": axis_2,
            "y_distribution": True,
        }

    def get(self, index: str | None = None) -> Any:
        """Get a named field from an object, or a structure with all fields."""
        if index is None:
            return self.to_struct()
        if not isinstance(index, str):
            raise HoraceError("MATLAB:nonExistentField", f"Invalid field name: {index!r}")
        if hasattr(self, index):
            return getattr(self, index)
        raise HoraceError(
            "MATLAB:nonExistentField",
            f"Reference to non-existent field {index}.",
        )

    def combine_horace_2d(self, w2: Any, *args: Any) -> d2d:
        warnings.warn(
            "combine_horace_2d is deprecated. Use combine instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.combine(w2, *args)

    def combine(self, w2: Any, *args: Any) -> d2d:
        """Combine two d2d datasets."""
        ndims1 = int(self.dimensions)
        ndims2 = int(w2.dimensions) if hasattr(w2, "dimensions") else -1
        if ndims1 != 2 or ndims2 != 2:
            raise HoraceError(
                "HORACE:d2d:invalid_argument",
                f"Both objects have the same dimensionality. Actually, ndims(obj1) = {ndims1}, ndims(obj2)={ndims2} ",
            )

        if len(args) == 0:
            route = 1
            tol = None
        elif len(args) == 1:
            tol = args[0]
            if not _is_numeric(tol):
                raise HoraceError(
                    "HORACE:d2d:invalid_argument",
                    "the combining tolerance should be specified by a scalar",
                )
            tol_arr = np.asarray(tol, dtype=float).ravel()
            if tol_arr.size != 2:
                raise HoraceError(
                    "HORACE:d2d:invalid_argument",
                    "The combining tolerance should be specified by a scalar",
                )
            tol = tol_arr
            route = 2
        else:
            raise HoraceError(
                "HORACE:d2d:invalid_argument",
                "Horace error: check number and format of input arguments",
            )

        ok, same_axes, mess = _check_rebinning_axes(self, w2)
        if not ok:
            raise HoraceError("HORACE:d2d:invalid_argument", mess)

        if same_axes:
            irange, uoff = _calculate_integration_range(self, w2)
            pax1 = int(self.pax[0]) - 1
            pax2 = int(w2.pax[0]) - 1
            if np.array_equal(self.u_to_rlu[:, pax1], w2.u_to_rlu[:, pax2]):
                x1, y1 = np.meshgrid(self.p[0].ravel(), self.p[1].ravel(), indexing="ij")
                x2, y2 = np.meshgrid(w2.p[0].ravel(), w2.p[1].ravel(), indexing="ij")
                s1, s2 = self.s, w2.s
                e1, e2 = self.e, w2.e
                n1, n2 = self.npix, w2.npix
            else:
                x1, y1 = np.meshgrid(self.p[0].ravel(), self.p[1].ravel(), indexing="ij")
                x2, y2 = np.meshgrid(w2.p[1].ravel(), w2.p[0].ravel(), indexing="ij")
                s1, s2 = self.s, w2.s.T
                e1, e2 = self.e, w2.e.T
                n1, n2 = self.npix, w2.npix.T

            xout, yout, sout, eout, nout = _combine_2d(
                x1, y1, s1, e1, n1, x2, y2, s2, e2, n2, tol if route == 2 else None
            )
            wout = self.copy()
            wout._do_check_combo_arg = False
            paxis = wout._axes.nbins_all_dims > 1
            rng = np.array(
                [[np.min(xout), np.min(yout)], [np.max(xout), np.max(yout)]],
                dtype=float,
            )
            img_rng = wout._axes.img_range.copy()
            img_rng[:, paxis] = rng
            img_rng[:, ~paxis] = irange
            wout._axes.img_range = img_rng
            nbins_all = wout._axes.nbins_all_dims.copy()
            pax_indices = np.where(paxis)[0]
            nbins_all[pax_indices[0]] = xout.shape[0] - 1
            nbins_all[pax_indices[1]] = yout.shape[1] - 1
            wout._axes.nbins_all_dims = nbins_all
            wout._s = sout
            wout._e = eout
            wout._npix = nout
            wout.title = f"{wout.title} COMBINED "
            wout._proj.offset = uoff
            wout._do_check_combo_arg = True
            wout.check_combo_arg()
            return wout
        else:
            irange, uoff = _calculate_integration_range(self, w2)
            w2tmp = w2.rebin_horace_2d(self)
            w1tmp = self.rebin_horace_2d(w2tmp)
            if route == 2 and tol is not None:
                tolx = None if np.isnan(tol[0]) else float(tol[0])
                toly = None if np.isnan(tol[1]) else float(tol[1])
                w2tmp = w2tmp.rebin_horace_2d(tolx, toly)
                w1tmp = w1tmp.rebin_horace_2d(tolx, toly)
            s1, e1, n1 = w1tmp.s.copy(), w1tmp.e.copy(), w1tmp.npix.copy()
            s2, e2, n2 = w2tmp.s.copy(), w2tmp.e.copy(), w2tmp.npix.copy()
            e1_old, e2_old = e1.copy(), e2.copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                e1 = np.where((s1 == 0) | np.isnan(e1 / s1) | np.isinf(e1 / s1), 0.0, e1 / s1)
                e2 = np.where((s2 == 0) | np.isnan(e2 / s2) | np.isinf(e2 / s2), 0.0, e2 / s2)
            biggest = max(float(np.max(s1)), float(np.max(s2)))
            e1 = e1 + (e1 == 0).astype(float) * 1e5 * biggest
            e2 = e2 + (e2 == 0).astype(float) * 1e5 * biggest
            sout = (s2 / e2 + s1 / e1) / (1.0 / e2 + 1.0 / e1)
            eout = 1.0 / (1.0 / e2 + 1.0 / e1)
            nout = n1 + n2
            zero_mask = (e1_old == 0) & (e2_old == 0)
            nout[zero_mask] = 0.0
            sout[zero_mask] = 0.0
            eout[zero_mask] = 0.0
            eout = eout * sout
            wout = w1tmp.copy()
            wout._s = sout
            wout._e = eout
            wout._npix = nout
            wout.iint = irange
            wout._proj.offset = uoff
            wout.title = f"{self.title} COMBINED"
            return wout

    def rebin_horace_2d(self, *args: Any) -> d2d:
        """Rebin a d2d object."""
        eps = float(np.finfo(float).eps)
        xin_vec = self.p[0].ravel()
        yin_vec = self.p[1].ravel()
        inmin_x, inmax_x = float(np.min(xin_vec)), float(np.max(xin_vec))
        inmin_y, inmax_y = float(np.min(yin_vec)), float(np.max(yin_vec))
        xin, yin = np.meshgrid(xin_vec, yin_vec, indexing="ij")

        if len(args) == 1:
            arg1 = args[0]
            if isinstance(arg1, DnDBase) or hasattr(arg1, "dnd"):
                w2 = arg1 if isinstance(arg1, d2d) else d2d(arg1)
                ok, same_axes, mess = _check_rebinning_axes(self, w2)
                if not ok:
                    raise HoraceError("HORACE:d2d:invalid_argument", mess)
                if same_axes:
                    pax1 = int(self.pax[0]) - 1
                    pax2 = int(w2.pax[0]) - 1
                    xout_w2 = w2.p[0].ravel()
                    yout_w2 = w2.p[1].ravel()
                    outmin_x, outmax_x = float(np.min(xout_w2)), float(np.max(xout_w2))
                    outmin_y, outmax_y = float(np.min(yout_w2)), float(np.max(yout_w2))
                    outbin_x = float(xout_w2[1] - xout_w2[0])
                    outbin_y = float(yout_w2[1] - yout_w2[0])
                    lo_x, hi_x = min(inmin_x, outmin_x), max(inmax_x, outmax_x)
                    lo_y, hi_y = min(inmin_y, outmin_y), max(inmax_y, outmax_y)
                    xout = np.arange(lo_x - eps, hi_x + eps + 0.5 * outbin_x, outbin_x)
                    yout = np.arange(lo_y - eps, hi_y + eps + 0.5 * outbin_y, outbin_y)
                    if np.array_equal(self.u_to_rlu[:, pax1], w2.u_to_rlu[:, pax2]):
                        xnew, ynew, sout, eout, nout = _rebin_2d(
                            xin, yin, self.s, self.e, self.npix, xout, yout
                        )
                        return _update_d2d_from_rebin(
                            self, xnew[:, 0], ynew[0, :], sout, eout, nout, f"{self.title} REBINNED "
                        )
                    else:
                        xnew, ynew, sout, eout, nout = _rebin_2d(
                            yin.T, xin.T, self.s.T, self.e.T, self.npix.T, xout, yout
                        )
                        return _update_d2d_from_rebin(
                            w2, xnew[:, 0], ynew[0, :], sout, eout, nout, f"{self.title} REBINNED "
                        )
                else:
                    raise NotImplementedError("Shoelace 2D non-collinear rebinning is not yet implemented.")
            elif _is_numeric(arg1):
                a1 = np.asarray(arg1, dtype=float).ravel()
                if a1.size == 1:
                    dx = float(a1[0])
                    xout = np.arange(inmin_x - eps, inmax_x + eps, dx)
                    if np.max(xout) < inmax_x:
                        xout = np.append(xout, xout[-1] + dx)
                elif a1.size == 3:
                    xout = np.arange(a1[0] - eps, a1[2] + eps, a1[1])
                else:
                    raise HoraceError("HORACE:d2d:invalid_argument", "Horace error: check the format of input arguments")
                xnew, ynew, sout, eout, nout = _rebin_2d(xin, yin, self.s, self.e, self.npix, xout, None)
                return _update_d2d_from_rebin(
                    self, xnew[:, 0], ynew[0, :], sout, eout, nout, f"{self.title} REBINNED "
                )
            else:
                raise HoraceError("HORACE:d2d:invalid_argument", "Horace error: check the format of input arguments")

        elif len(args) == 2:
            arg1, arg2 = args
            a1 = np.array([], dtype=float) if arg1 is None else np.asarray(arg1, dtype=float).ravel()
            a2 = np.array([], dtype=float) if arg2 is None else np.asarray(arg2, dtype=float).ravel()

            if a1.size == 0:
                xout = None
            elif a1.size == 1:
                dx = float(a1[0])
                xout = np.arange(inmin_x - eps, inmax_x + eps, dx)
                if np.max(xout) < inmax_x:
                    xout = np.append(xout, xout[-1] + dx)
            elif a1.size == 3:
                xout = np.arange(a1[0] - eps, a1[2] + eps, a1[1])
            else:
                raise HoraceError("HORACE:d2d:invalid_argument", "Horace error: check the format of input arguments")

            if a2.size == 0:
                yout = None
            elif a2.size == 1:
                dy = float(a2[0])
                yout = np.arange(inmin_y - eps, inmax_y + eps, dy)
                if np.max(yout) < inmax_y:
                    yout = np.append(yout, yout[-1] + dy)
            elif a2.size == 3:
                yout = np.arange(a2[0] - eps, a2[2] + eps, a2[1])
            else:
                raise HoraceError("HORACE:d2d:invalid_argument", "Horace error: check the format of input arguments")

            xnew, ynew, sout, eout, nout = _rebin_2d(xin, yin, self.s, self.e, self.npix, xout, yout)
            return _update_d2d_from_rebin(
                self, xnew[:, 0], ynew[0, :], sout, eout, nout, f"{self.title} REBINNED "
            )
        else:
            raise HoraceError("HORACE:d2d:invalid_argument", "Horace error: check the format of input arguments")

    def compare_sym_axes(
        self, v1: Any, v2: Any, v3: Any
    ) -> tuple[bool, np.ndarray]:
        """Determine whether reflection plane is perpendicular to data plane along a data axis."""
        v1_arr = np.asarray(v1, dtype=float).ravel()
        v2_arr = np.asarray(v2, dtype=float).ravel()
        v3_arr = np.asarray(v3, dtype=float).ravel()
        normvec = np.cross(v1_arr, v2_arr)
        pax1 = int(self.pax[0]) - 1
        pax2 = int(self.pax[1]) - 1
        datavec1 = self.u_to_rlu[:3, pax1]
        datavec2 = self.u_to_rlu[:3, pax2]

        c1 = np.all(np.abs(np.cross(normvec, datavec1)) < 1e-5)
        c2 = np.all(np.abs(np.cross(normvec, datavec2)) < 1e-5)
        speedup = False
        if bool(c1) ^ bool(c2):
            speedup = True
        elif np.all(np.cross(normvec, datavec1) < 1e-5) and np.all(datavec2 < 1e-5):
            speedup = True
        elif np.all(np.cross(normvec, datavec2) < 1e-5) and np.all(datavec1 < 1e-5):
            speedup = True

        midpoint = np.array([], dtype=float)
        if speedup:
            inv_u = np.linalg.inv(self.u_to_rlu[:3, :3])
            v3new = inv_u @ v3_arr
            if np.all(np.cross(normvec, datavec1) < 1e-5) and np.any(datavec1 > 1e-5):
                midpoint = np.array([v3new[pax1], np.nan], dtype=float)
            elif np.all(np.cross(normvec, datavec2) < 1e-5) and np.any(datavec2 > 1e-5):
                midpoint = np.array([np.nan, v3new[pax2]], dtype=float)
            else:
                raise HoraceError(
                    "HORACE:d2d:runtime_error",
                    "Horace error: symmetrisation logic flaw. Contact R. Ewings",
                )
        return speedup, midpoint

    def calculate_transformation_matrix(
        self, v1: Any, v2: Any, v3: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Determine the reflection matrix and translation vector in terms of the basis of self."""
        vec1 = np.asarray(v1, dtype=float).ravel()
        vec2 = np.asarray(v2, dtype=float).ravel()
        vec3 = np.asarray(v3, dtype=float).ravel()
        inv_u = np.linalg.inv(self.u_to_rlu[:3, :3])
        vec1p = inv_u @ vec1
        vec2p = inv_u @ vec2
        vec3p = inv_u @ vec3

        normvec = np.cross(vec1p, vec2p)
        norm_sq = float(np.sum(normvec**2))
        R = np.eye(3, dtype=float) - 2.0 * np.outer(normvec, normvec) / norm_sq
        trans = vec3p

        pax1 = int(self.pax[0]) - 1
        pax2 = int(self.pax[1]) - 1
        test_trans = float(
            np.dot(
                np.cross(self.u_to_rlu[:3, pax1], self.u_to_rlu[:3, pax2]),
                trans,
            )
        )
        if test_trans > 1e-5:
            raise HoraceError(
                "HORACE:d2d:invalid_argument",
                f"Horace error: offset vector {vec3.tolist()} for reflection plane does not lie in the data plane",
            )
        return R, trans

    def symmetrise_horace_2d(self, *args: Any) -> d2d:
        """Symmetrise d2d data."""
        if len(args) == 1:
            if _is_numeric(args[0]) and np.asarray(args[0]).size == 2:
                midpoint = np.asarray(args[0], dtype=float).ravel()
                xin, yin = np.meshgrid(self.p[0].ravel(), self.p[1].ravel(), indexing="ij")
                xout, yout, sout, eout, nout = _symmetrise_2d(
                    xin, yin, self.s, self.e, self.npix, midpoint
                )
                return _update_d2d_from_rebin(
                    self, xout[:, 0], yout[0, :], sout, eout, nout, self.title
                )
            raise HoraceError(
                "HORACE:d2d:invalid_argument",
                "Horace error: check form of input arguments for symmetrise_horace_2d",
            )
        elif len(args) in (2, 3):
            v1 = np.asarray(args[0], dtype=float).ravel()
            v2 = np.asarray(args[1], dtype=float).ravel()
            v3 = np.zeros(3, dtype=float) if len(args) == 2 else np.asarray(args[2], dtype=float).ravel()
            if v1.size != 3 or v2.size != 3 or v3.size != 3:
                raise HoraceError(
                    "HORACE:d2d:invalid_argument",
                    "Horace error: check form of input arguments for symmetrise_horace_2d",
                )
            speedup, midpoint = self.compare_sym_axes(v1, v2, v3)
            if speedup:
                xin, yin = np.meshgrid(self.p[0].ravel(), self.p[1].ravel(), indexing="ij")
                xout, yout, sout, eout, nout = _symmetrise_2d(
                    xin, yin, self.s, self.e, self.npix, midpoint
                )
                return _update_d2d_from_rebin(
                    self, xout[:, 0], yout[0, :], sout, eout, nout, self.title
                )
            raise NotImplementedError("Non-axis-aligned 2D symmetrisation is not yet implemented.")
        else:
            raise HoraceError(
                "HORACE:d2d:invalid_argument",
                "Horace error: check form of input arguments for symmetrise_horace_2d",
            )


def _update_d2d_from_rebin(
    win: d2d,
    xvec: np.ndarray,
    yvec: np.ndarray,
    sout: np.ndarray,
    eout: np.ndarray,
    nout: np.ndarray,
    title: str,
) -> d2d:
    wout = win.copy()
    wout._do_check_combo_arg = False
    pax_idx = [int(p) - 1 for p in wout.pax]
    img_range = wout._axes.img_range.copy()
    img_range[0, pax_idx[0]] = float(np.min(xvec))
    img_range[1, pax_idx[0]] = float(np.max(xvec))
    img_range[0, pax_idx[1]] = float(np.min(yvec))
    img_range[1, pax_idx[1]] = float(np.max(yvec))
    wout._axes.img_range = img_range
    nbins_all = wout._axes.nbins_all_dims.copy()
    nbins_all[pax_idx[0]] = max(len(xvec) - 1, 1)
    nbins_all[pax_idx[1]] = max(len(yvec) - 1, 1)
    wout._axes.nbins_all_dims = nbins_all
    wout._s = np.asarray(sout, dtype=float)
    wout._e = np.asarray(eout, dtype=float)
    wout._npix = np.asarray(nout, dtype=float)
    wout.title = title
    wout._do_check_combo_arg = True
    wout.check_combo_arg()
    return wout


def _check_rebinning_axes(w1: d2d, w2: d2d) -> tuple[bool, bool, str]:
    ok = False
    same_axes = False
    mess = ""
    if not np.array_equal(w1.alatt, w2.alatt) or not np.array_equal(w1.angdeg, w2.angdeg):
        return False, False, "Horace error: rebinning impossible for 2 datasets where lattice parameters are different"

    w1_plane = w1.u_to_rlu[:, [int(w1.pax[0]) - 1, int(w1.pax[1]) - 1]].copy()
    w2_plane = w2.u_to_rlu[:, [int(w2.pax[0]) - 1, int(w2.pax[1]) - 1]].copy()

    if np.array_equal(w1_plane[3, :], [0.0, 0.0]) and np.array_equal(w2_plane[3, :], [0.0, 0.0]):
        w1_p = w1_plane[:3, :]
        w2_p = w2_plane[:3, :]
        perp1 = np.cross(w1_p[:, 0], w1_p[:, 1])
        perp2 = np.cross(w2_p[:, 0], w2_p[:, 1])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        if np.allclose(perp1, perp2, atol=1e-12) or np.allclose(perp1, -perp2, atol=1e-12):
            ok = True
        else:
            return False, False, "Horace error: rebinning impossible when 2 objects have different scattering planes"

        w1_p[w1_p < 1e-5] = 0.0
        w2_p[w2_p < 1e-5] = 0.0
        c11 = np.all(np.cross(w1_p[:, 0], w2_p[:, 0]) == 0)
        c12 = np.all(np.cross(w1_p[:, 0], w2_p[:, 1]) == 0)
        c21 = np.all(np.cross(w1_p[:, 1], w2_p[:, 0]) == 0)
        c22 = np.all(np.cross(w1_p[:, 1], w2_p[:, 1]) == 0)
        same_axes = (c11 or c12) and (c21 or c22)
    elif np.sum(w1_plane[3, :]) == np.sum(w2_plane[3, :]) and np.sum(w1_plane[3, :]) == 1:
        w1_p = w1_plane[:3, :]
        w2_p = w2_plane[:3, :]
        if (
            np.all(np.cross(w1_p[:, 0], w2_p[:, 0]) == 0)
            and np.all(np.cross(w1_p[:, 0], w2_p[:, 1]) == 0)
            and np.all(np.cross(w1_p[:, 1], w2_p[:, 0]) == 0)
            and np.all(np.cross(w1_p[:, 1], w2_p[:, 1]) == 0)
        ):
            same_axes = True
            ok = True
    return ok, same_axes, mess


def _calculate_integration_range(w1: d2d, w2: d2d) -> tuple[np.ndarray, np.ndarray]:
    iax11 = w1.u_to_rlu[:, int(w1.iax[0]) - 1]
    iax12 = w1.u_to_rlu[:, int(w1.iax[1]) - 1]
    off11 = float(w1.uoffset[int(w1.iax[0]) - 1])
    off12 = float(w1.uoffset[int(w1.iax[1]) - 1])

    iax21 = w2.u_to_rlu[:, int(w2.iax[0]) - 1]
    off21 = float(w2.uoffset[int(w2.iax[0]) - 1])
    off22 = float(w2.uoffset[int(w2.iax[1]) - 1])

    iint1 = w1.iint
    iint2 = w2.iint

    if np.array_equal(1e-5 * np.round(1e5 * iax11), 1e-5 * np.round(1e5 * iax21)):
        lo1 = min(float(iint1[0, 0]), float(iint2[0, 0]))
        hi1 = max(float(iint1[1, 0]), float(iint2[1, 0]))
        lo2 = min(float(iint1[0, 1]), float(iint2[0, 1]))
        hi2 = max(float(iint1[1, 1]), float(iint2[1, 1]))
        uoff1 = (off11 + off21) / 2.0
        uoff2 = (off12 + off22) / 2.0
    elif np.array_equal(1e-5 * np.round(1e5 * iax12), 1e-5 * np.round(1e5 * iax21)):
        lo1 = min(float(iint1[0, 0]), float(iint2[0, 1]))
        hi1 = max(float(iint1[1, 0]), float(iint2[1, 1]))
        lo2 = min(float(iint1[0, 1]), float(iint2[0, 0]))
        hi2 = max(float(iint1[1, 1]), float(iint2[1, 0]))
        uoff1 = (off11 + off22) / 2.0
        uoff2 = (off12 + off21) / 2.0
    else:
        raise HoraceError(
            "HORACE:d2d:runtime_error",
            "Horace error: logic flaw in horace_combine_2d. Contact R. Ewings for help",
        )
    irange = np.array([[lo1, lo2], [hi1, hi2]], dtype=float)
    utmp = w1.uoffset.copy()
    utmp[int(w1.iax[0]) - 1] = uoff1
    utmp[int(w1.iax[1]) - 1] = uoff2
    return irange, utmp


def _combine_2d(
    x1: np.ndarray,
    y1: np.ndarray,
    s1: np.ndarray,
    e1: np.ndarray,
    n1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    s2: np.ndarray,
    e2: np.ndarray,
    n2: np.ndarray,
    tol: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = float(np.finfo(float).eps)
    xlo1, xhi1 = float(np.min(x1)), float(np.max(x1))
    xlo2, xhi2 = float(np.min(x2)), float(np.max(x2))
    ylo1, yhi1 = float(np.min(y1)), float(np.max(y1))
    ylo2, yhi2 = float(np.min(y2)), float(np.max(y2))

    if tol is None or len(tol) == 0:
        tol_arr = np.array([0.0, 0.0], dtype=float)
    else:
        tol_arr = np.asarray(tol, dtype=float).ravel()

    if np.isnan(tol_arr[0]) or tol_arr[0] == 0:
        if xlo1 <= xlo2 and xhi1 >= xhi2:
            xnew = x1[:, 0]
        elif xlo2 <= xlo1 and xhi2 >= xhi1:
            xnew = x2[:, 0]
        elif xlo2 < xlo1 and xhi2 <= xhi1:
            xnew = np.concatenate([x2[x2[:, 0] < xlo1, 0], x1[:, 0]])
        else:
            xnew = np.concatenate([x1[x1[:, 0] < xlo2, 0], x2[:, 0]])
    else:
        tx = float(tol_arr[0])
        xnew = np.arange(min(xlo1, xlo2), max(xhi1, xhi2) + tx - eps, tx)

    if np.isnan(tol_arr[1]) or tol_arr[1] == 0:
        if ylo1 <= ylo2 and yhi1 >= yhi2:
            ynew = y1[0, :]
        elif ylo2 <= ylo1 and yhi2 >= yhi1:
            ynew = y2[0, :]
        elif ylo2 < ylo1 and yhi2 <= yhi1:
            ynew = np.concatenate([y2[0, y2[0, :] < ylo1], y1[0, :]])
        else:
            ynew = np.concatenate([y1[0, y1[0, :] < ylo2], y2[0, :]])
    else:
        ty = float(tol_arr[1])
        ynew = np.arange(min(ylo1, ylo2), max(yhi1, yhi2) + ty - eps, ty)

    _, _, sig1, err1, npix1 = _rebin_2d(x1, y1, s1, e1, n1, xnew, ynew)
    _, _, sig2, err2, npix2 = _rebin_2d(x2, y2, s2, e2, n2, xnew, ynew)
    xout, yout = np.meshgrid(xnew, ynew, indexing="ij")

    err1_old = err1.copy()
    err2_old = err2.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        err1 = np.where((sig1 == 0) | np.isnan(err1 / sig1) | np.isinf(err1 / sig1), 0.0, err1 / sig1)
        err2 = np.where((sig2 == 0) | np.isnan(err2 / sig2) | np.isinf(err2 / sig2), 0.0, err2 / sig2)

    biggest = max(float(np.max(sig1)), float(np.max(sig2)))
    err1 = err1 + (err1 == 0).astype(float) * 1e5 * biggest
    err2 = err2 + (err2 == 0).astype(float) * 1e5 * biggest
    sout = (sig2 / err2 + sig1 / err1) / (1.0 / err2 + 1.0 / err1)
    eout = 1.0 / (1.0 / err2 + 1.0 / err1)
    nout = npix1 + npix2
    zero_mask = (err1_old == 0) & (err2_old == 0)
    nout[zero_mask] = 0.0
    sout[zero_mask] = 0.0
    eout[zero_mask] = 0.0
    eout = eout * sout
    return xout, yout, sout, eout, nout


def _rebin_2d(
    xin: np.ndarray,
    yin: np.ndarray,
    sin: np.ndarray,
    ein: np.ndarray,
    nin: np.ndarray,
    xout: np.ndarray | None,
    yout: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_empty = xout is None or len(xout) == 0
    y_empty = yout is None or len(yout) == 0
    if x_empty and y_empty:
        return xin, yin, sin, ein, nin
    if not x_empty and not y_empty:
        xnew1, ynew1, sout1, eout1, nout1 = _rebin_2d_1axis(xin, yin, xout, sin, ein, nin)
        ynew2, xnew2, sout2, eout2, nout2 = _rebin_2d_1axis(
            ynew1.T, xnew1.T, yout, sout1.T, eout1.T, nout1.T
        )
        return xnew2.T, ynew2.T, sout2.T, eout2.T, nout2.T
    if y_empty and not x_empty:
        return _rebin_2d_1axis(xin, yin, xout, sin, ein, nin)
    ynew1, xnew1, sout1, eout1, nout1 = _rebin_2d_1axis(
        yin.T, xin.T, yout, sin.T, ein.T, nin.T
    )
    return xnew1.T, ynew1.T, sout1.T, eout1.T, nout1.T


def _rebin_2d_1axis(
    xin: np.ndarray,
    yin: np.ndarray,
    xout: np.ndarray,
    sin: np.ndarray,
    ein: np.ndarray,
    nin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xout_vec = np.sort(np.asarray(xout, dtype=float).ravel())
    small = 1e-5 * float(np.max(np.abs(xout_vec)))

    if float(np.max(xin)) > float(np.max(xout_vec)) + small:
        raise HoraceError(
            "HORACE:d2d:invalid_argument",
            "Rebinning error: maximum of output bin boundaries must be >= max of input bin boundaries",
        )
    if float(np.min(xin)) < float(np.min(xout_vec)) - small:
        raise HoraceError(
            "HORACE:d2d:invalid_argument",
            "Rebinning error: minimum of output bin boundaries must be <= min of input bin boundaries",
        )

    nin_c = np.array(nin, dtype=float, copy=True)
    sin_c = np.array(sin, dtype=float, copy=True)
    ein_c = np.array(ein, dtype=float, copy=True)

    nin_test = (nin_c > 0).astype(float)
    sin_c *= nin_test
    ein_c *= nin_test
    okerr = (ein_c > 0).astype(float)
    sin_c *= okerr
    nin_c *= okerr

    nan_s = np.isnan(sin_c)
    ein_c[nan_s] = 0.0
    nin_c[nan_s] = 0.0
    sin_c[nan_s] = 0.0
    nan_e = np.isnan(ein_c)
    sin_c[nan_e] = 0.0
    nin_c[nan_e] = 0.0
    ein_c[nan_e] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        ein_frac = np.where((sin_c == 0) | np.isnan(ein_c / sin_c) | np.isinf(ein_c / sin_c), 0.0, ein_c / sin_c)

    xtosort = xin[:-1, 0]
    sort_idx = np.argsort(xtosort, kind="mergesort")
    snew = sin_c[sort_idx, :]
    enew = ein_frac[sort_idx, :]
    nnew = nin_c[sort_idx, :]

    xinnew = xin[:, 0]
    dx = np.diff(xinnew)
    eps_val = 1e-5 * float(np.min(dx))

    xlo = xout_vec[:-1]
    xhi = xout_vec[1:]
    sz2 = sin_c.shape[1]
    lonew = xinnew[:-1]
    hinew = xinnew[1:]
    filler = np.zeros((1, sz2), dtype=float)

    xnew, ynew = np.meshgrid(xout_vec, yin[0, :], indexing="ij")

    sout_rows: list[np.ndarray] = []
    eout_rows: list[np.ndarray] = []
    nout_rows: list[np.ndarray] = []

    for i in range(len(xlo)):
        inbin = (xinnew > xlo[i]) & (xinnew < xhi[i])
        ind = np.where(inbin)[0]
        if len(ind) == 1:
            binfrac = np.empty((0, 1), dtype=float)
            cumulsum, esum, nsum = _rebin_2d_1axis_multibins(
                ind, binfrac, snew, enew, nnew, xinnew, eps_val, xlo, xhi, i
            )
            sout_rows.append(cumulsum if cumulsum.size > 0 else filler)
            eout_rows.append(esum if esum.size > 0 else filler)
            nout_rows.append(nsum if nsum.size > 0 else filler)
        elif len(ind) == 0:
            lorange = xlo[i] < float(np.min(xinnew))
            hirange = xhi[i] > float(np.max(xinnew))
            if lorange:
                sout_rows.insert(0, filler)
                eout_rows.insert(0, filler)
                nout_rows.insert(0, filler)
            if hirange:
                sout_rows.append(filler)
                eout_rows.append(filler)
                nout_rows.append(filler)
            if not lorange and not hirange:
                cumulsum, esum, nsum = _rebin_2d_1axis_finegrid(
                    snew, enew, nnew, xinnew, lonew, hinew, eps_val, xlo, xhi, i
                )
                sout_rows.append(cumulsum if cumulsum.size > 0 else filler)
                eout_rows.append(esum if esum.size > 0 else filler)
                nout_rows.append(nsum if nsum.size > 0 else filler)
        else:
            binfrac = np.ones((len(ind) - 1, 1), dtype=float)
            cumulsum, esum, nsum = _rebin_2d_1axis_multibins(
                ind, binfrac, snew, enew, nnew, xinnew, eps_val, xlo, xhi, i
            )
            sout_rows.append(cumulsum if cumulsum.size > 0 else filler)
            eout_rows.append(esum if esum.size > 0 else filler)
            nout_rows.append(nsum if nsum.size > 0 else filler)

    sout = np.vstack(sout_rows) if sout_rows else np.zeros((0, sz2), dtype=float)
    eout = np.vstack(eout_rows) if eout_rows else np.zeros((0, sz2), dtype=float)
    nout = np.vstack(nout_rows) if nout_rows else np.zeros((0, sz2), dtype=float)

    rowout, colout = sout.shape
    rowx = len(xout_vec)
    if rowout <= rowx - 2:
        difflen = (rowx - rowout) - 1
        extra = np.zeros((difflen, colout), dtype=float)
        sout = np.vstack([sout, extra])
        eout = np.vstack([eout, extra])
        nout = np.vstack([nout, extra])

    nan_out = np.isnan(sout)
    nout[nan_out] = 0.0
    eout[nan_out] = 0.0
    sout[nan_out] = 0.0

    nout[eout == 0] = 0.0
    sout[eout == 0] = 0.0
    sout[nout == 0] = 0.0
    eout[nout == 0] = 0.0

    max_ein = float(np.max(ein_frac)) if ein_frac.size > 0 else 0.0
    huge_err = eout > (1e2 * max_ein)
    sout[huge_err] = 0.0
    nout[huge_err] = 0.0
    eout[huge_err] = 0.0

    eout = eout * sout
    return xnew, ynew, sout, eout, nout


def _rebin_2d_1axis_multibins(
    ind: np.ndarray,
    binfrac: np.ndarray,
    snew: np.ndarray,
    enew: np.ndarray,
    nnew: np.ndarray,
    xinnew: np.ndarray,
    eps: float,
    xlo: np.ndarray,
    xhi: np.ndarray,
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enew_c = enew.copy()
    max_s = float(np.max(snew)) if snew.size > 0 else 0.0
    enew_c[(snew == 0) & (enew_c == 0)] = 1e5 * max_s if max_s > 0 else 1e5

    sz2 = snew.shape[1]
    bf = np.tile(binfrac, (1, sz2))

    dlo = xinnew[ind[0]] - xlo[i]
    dhi = xhi[i] - xinnew[ind[-1]]
    has_lo = ind[0] != 0
    has_hi = ind[-1] != len(xinnew) - 1

    lofrac = np.full((1, sz2), eps + dlo / (xinnew[ind[0]] - xinnew[ind[0] - 1]), dtype=float) if has_lo else None
    hifrac = np.full((1, sz2), eps + dhi / (xinnew[ind[-1] + 1] - xinnew[ind[-1]]), dtype=float) if has_hi else None

    if has_lo and has_hi:
        bf_all = np.vstack([lofrac, bf, hifrac])
        sl = slice(ind[0] - 1, ind[-1] + 1)
    elif has_hi:
        bf_all = np.vstack([bf, hifrac])
        sl = slice(ind[0], ind[-1] + 1)
    elif has_lo:
        bf_all = np.vstack([lofrac, bf])
        sl = slice(ind[0] - 1, ind[-1])
    else:
        bf_all = bf
        sl = slice(ind[0], ind[-1])

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = bf_all / enew_c[sl, :]
        w_sum = np.sum(weights, axis=0, keepdims=True)
        cumulsum = np.sum(bf_all * snew[sl, :] / enew_c[sl, :], axis=0, keepdims=True) / w_sum
        esum = 1.0 / w_sum
    nsum = np.sum(bf_all * nnew[sl, :], axis=0, keepdims=True)
    return cumulsum, esum, nsum


def _rebin_2d_1axis_finegrid(
    snew: np.ndarray,
    enew: np.ndarray,
    nnew: np.ndarray,
    xinnew: np.ndarray,
    lonew: np.ndarray,
    hinew: np.ndarray,
    eps: float,
    xlo: np.ndarray,
    xhi: np.ndarray,
    i: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = xlo[i]
    hi = xhi[i]
    binind = np.where((lonew <= lo) & (hinew >= hi))[0]
    if len(binind) == 0:
        return np.empty((0, snew.shape[1])), np.empty((0, snew.shape[1])), np.empty((0, snew.shape[1]))
    idx = binind[0]
    insize = hinew[idx] - lonew[idx]
    outsize = hi - lo
    frac = np.full((1, snew.shape[1]), outsize / insize, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_row = snew[idx : idx + 1, :]
        e_row = enew[idx : idx + 1, :]
        n_row = nnew[idx : idx + 1, :]
        cumulsum = (frac * s_row / e_row) / (frac / e_row)
        esum = 1.0 / (frac / e_row)
    nsum = frac * n_row
    return cumulsum, esum, nsum


def _symmetrise_2d(
    xin: np.ndarray,
    yin: np.ndarray,
    sin: np.ndarray,
    ein: np.ndarray,
    nin: np.ndarray,
    midpoint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if midpoint.size == 0:
        return xin, yin, sin, ein, nin
    xtemp, ytemp, stemp, etemp, ntemp = xin.copy(), yin.copy(), sin.copy(), ein.copy(), nin.copy()
    if not np.isnan(midpoint[0]):
        xtemp, ytemp, stemp, etemp, ntemp = _symmetrise_2d_1axis(
            xtemp, ytemp, stemp, etemp, ntemp, float(midpoint[0])
        )
    if not np.isnan(midpoint[1]):
        ytemp2, xtemp2, stemp2, etemp2, ntemp2 = _symmetrise_2d_1axis(
            ytemp.T, xtemp.T, stemp.T, etemp.T, ntemp.T, float(midpoint[1])
        )
        xtemp, ytemp, stemp, etemp, ntemp = xtemp2.T, ytemp2.T, stemp2.T, etemp2.T, ntemp2.T
    return xtemp, ytemp, stemp, etemp, ntemp


def _symmetrise_2d_1axis(
    xin: np.ndarray,
    yin: np.ndarray,
    sin: np.ndarray,
    ein: np.ndarray,
    nin: np.ndarray,
    midpoint: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eps = float(np.finfo(float).eps)
    xval = xin[:, 0]
    yval = yin[0, :]
    x2 = 2.0 * midpoint - xval
    sort_idx = np.argsort(x2[1:], kind="mergesort")
    reflsin = sin[sort_idx, :]
    reflein = ein[sort_idx, :]
    reflnin = nin[sort_idx, :]
    x2_sorted = np.sort(x2)
    reflxin, reflyin = np.meshgrid(x2_sorted, yval, indexing="ij")

    max1, max2 = float(np.max(xval)), float(np.max(x2))
    x_rebin = xval[xval >= midpoint] if max1 >= max2 else x2_sorted[x2_sorted >= midpoint]
    if float(np.min(x_rebin)) > midpoint + eps:
        x_rebin = np.concatenate([[midpoint], x_rebin])

    xin_c = xin[xval >= midpoint, :]
    yin_c = yin[xval >= midpoint, :]
    sin_c = sin[xval[:-1] >= midpoint, :]
    ein_c = ein[xval[:-1] >= midpoint, :]
    nin_c = nin[xval[:-1] >= midpoint, :]

    reflxin_c = reflxin[x2_sorted >= midpoint, :]
    reflyin_c = reflyin[x2_sorted >= midpoint, :]
    reflsin_c = reflsin[x2_sorted[:-1] >= midpoint, :]
    reflein_c = reflein[x2_sorted[:-1] >= midpoint, :]
    reflnin_c = reflnin[x2_sorted[:-1] >= midpoint, :]

    xtemp1, ytemp1, stemp1, etemp1, ntemp1 = _rebin_2d_1axis(xin_c, yin_c, x_rebin, sin_c, ein_c, nin_c)
    xtemp2, ytemp2, stemp2, etemp2, ntemp2 = _rebin_2d_1axis(
        reflxin_c, reflyin_c, x_rebin, reflsin_c, reflein_c, reflnin_c
    )
    return _combine_2d(
        xtemp1, ytemp1, stemp1, etemp1, ntemp1, xtemp2, ytemp2, stemp2, etemp2, ntemp2, None
    )
