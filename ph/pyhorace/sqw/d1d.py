"""1-dimensional Horace DnD dataset class (d1d).

Ported from Horace MATLAB @d1d class (horace_core/sqw/@d1d).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.configurations.config_store import warn_with_id
from pyhorace.core.data_op_interface import _is_numeric
from pyhorace.core.serializable import register_serializable
from pyhorace.sqw.dnd_base import DnDBase


@register_serializable
class d1d(DnDBase):
    """1-dimensional Horace DnD dataset."""

    NUM_DIMS: ClassVar[int] = 1

    def _set_senpix(self, val: Any, field: str) -> None:
        if _is_numeric(val):
            arr = np.asarray(val, dtype=float)
            if arr.size > 0:
                val = arr.reshape(-1, 1, order="F")
        super()._set_senpix(val, field)

    def IX_dataset_1d(self) -> Dict[str, Any]:
        """Convert 1D DnD object into an IX_dataset_1d dictionary representation."""
        title_main, title_pax, *_ = self.data_plot_titles()
        nopix = self._npix == 0
        signal = self._s.copy().astype(float)
        signal[nopix] = np.nan
        err = np.sqrt(self._e.copy().astype(float))
        err[nopix] = 0.0
        return {
            "title": title_main,
            "signal": signal,
            "error": err,
            "s_axis": "Intensity",
            "x": self.p[0].copy(),
            "x_axis": title_pax[0] if title_pax else "",
            "x_distribution": True,
        }

    # ------------------------------------------------------------------
    # Combine, rebin_horace_1d, symmetrise_horace_1d
    # ------------------------------------------------------------------
    def combine_horace_1d(self, w2: Any, *args: Any) -> "d1d":
        """Deprecated alias for combine()."""
        warn_with_id(
            "HORACE:d1d:deprecated",
            "combine_horace_1d is deprecated. Use combine instead",
        )
        return self.combine(w2, *args)

    def combine(self, w2: Any, *args: Any) -> "d1d":
        """Combine two 1D datasets."""
        w2_dnd = getattr(w2, "data") if hasattr(w2, "data") and not isinstance(w2, DnDBase) else w2
        ndims1 = int(self.dimensions)
        ndims2 = int(w2_dnd.dimensions) if hasattr(w2_dnd, "dimensions") else -1
        if ndims1 != 1 or ndims2 != 1:
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                f"Both objects must be 1D. Got ndims(obj1)={ndims1}, ndims(obj2)={ndims2}",
            )

        if len(args) == 0:
            tol = None
        elif len(args) == 1:
            tol_arg = args[0]
            if not _is_numeric(tol_arg) or np.asarray(tol_arg).size != 1:
                raise HoraceError(
                    "HORACE:d1d:invalid_argument",
                    "The combining tolerance should be specified by a scalar",
                )
            tol = float(np.asarray(tol_arg).flat[0])
        else:
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                f"Incorrect number of input arguments. 2 or 3 allowed, provided: {2 + len(args)}",
            )

        eq_proj, _ = self.proj.equal_to_tol(w2_dnd.proj)
        if not eq_proj:
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                "1d objects must have same x-axis type and direction",
            )

        x1, x2 = self.p[0], w2_dnd.p[0]
        s1, s2 = self.s, w2_dnd.s
        e1, e2 = self.e, w2_dnd.e
        n1, n2 = self.npix, w2_dnd.npix

        xout, sout, eout, nout = self._combine_1d(x1, s1, e1, n1, x2, s2, e2, n2, tol)

        nbins = xout.size - 1
        nbins_all_dims = self.axes.nbins_all_dims.copy()
        tot_range = self.img_range.copy()
        pax0 = int(self.pax[0]) - 1
        nbins_all_dims[pax0] = nbins
        tot_range[0, pax0] = float(np.min(xout))
        tot_range[1, pax0] = float(np.max(xout))

        wout = d1d(self)
        wout._do_check_combo_arg = False
        wout._axes.nbins_all_dims = nbins_all_dims
        wout._axes.img_range = tot_range
        wout._s = sout.reshape(-1, 1, order="F")
        wout._e = eout.reshape(-1, 1, order="F")
        wout._npix = nout.reshape(-1, 1, order="F")
        wout.title = f"{wout.title} COMBINED "
        wout._do_check_combo_arg = True
        wout.check_combo_arg()
        return wout

    def rebin_horace_1d(self, *args: Any) -> "d1d":
        """Rebin 1D dataset along its x-axis."""
        if int(self.dimensions) != 1:
            raise HoraceError(
                "HORACE:rebin_horace_1d:invalid_argument",
                "Rebinning currently only implemented for d1d, d2d, and sqw objects",
            )
        if len(args) != 1:
            raise HoraceError(
                "HORACE:rebin_horace_1d:invalid_argument",
                "Check the format of input arguments",
            )

        arg = args[0]
        xin_vec = self.p[0].ravel()
        inmin_x = float(np.min(xin_vec))
        inmax_x = float(np.max(xin_vec))
        eps = float(np.finfo(float).eps)

        if isinstance(arg, DnDBase) or hasattr(arg, "data"):
            w2 = d1d(arg)
            ok, same_axes, mess = self._check_rebinning_axes_1d(w2)
            if not ok:
                raise HoraceError("HORACE:rebin_horace_1d:invalid_argument", mess)
            if not same_axes:
                raise HoraceError(
                    "HORACE:rebin_horace_1d:array_mismatch",
                    "1d objects must have same x-axis projection",
                )
            x2 = w2.p[0].ravel()
            outmin_x = float(np.min(x2))
            outmax_x = float(np.max(x2))
            outbin_x = float(x2[1] - x2[0])
            lo_x = min(inmin_x, outmin_x)
            hi_x = max(inmax_x, outmax_x)
            xout = np.arange(lo_x - outbin_x + eps, hi_x + outbin_x, outbin_x)
        elif _is_numeric(arg):
            vec = np.asarray(arg, dtype=float).ravel()
            if vec.size == 1:
                step = float(vec[0])
                xout = np.arange(inmin_x - step + eps, inmax_x + step, step)
                if np.max(xout) < inmax_x:
                    xout = np.append(xout, xout[-1] + step)
            elif vec.size == 3:
                lo, step, hi = float(vec[0]), float(vec[1]), float(vec[2])
                xout = np.arange(lo - step - eps, hi + step, step)
            else:
                raise HoraceError(
                    "HORACE:rebin_horace_1d:invalid_argument",
                    "Check the format of input arguments",
                )
        else:
            raise HoraceError(
                "HORACE:rebin_horace_1d:invalid_argument",
                "Check the format of input arguments",
            )

        sout, eout, nout = self._rebin_1d_general(xin_vec, xout, self.s, self.e, self.npix)
        wout = d1d(self)
        pax0 = int(self.pax[0]) - 1
        nbins_all_dims = self.axes.nbins_all_dims.copy()
        tot_range = self.img_range.copy()
        nbins_all_dims[pax0] = xout.size - 1
        tot_range[0, pax0] = float(np.min(xout))
        tot_range[1, pax0] = float(np.max(xout))
        wout._do_check_combo_arg = False
        wout._axes.nbins_all_dims = nbins_all_dims
        wout._axes.img_range = tot_range
        wout._s = sout.reshape(-1, 1, order="F")
        wout._e = eout.reshape(-1, 1, order="F")
        wout._npix = nout.reshape(-1, 1, order="F")
        wout.title = f"{wout.title} REBINNED "
        wout._do_check_combo_arg = True
        wout.check_combo_arg()
        return wout

    def symmetrise_horace_1d(self, midpoint: Any = 0.0) -> "d1d":
        """Symmetrise a 1D dataset about a specified midpoint."""
        if int(self.dimensions) != 1:
            raise HoraceError("HORACE:d1d:invalid_argument", "Ensure input object is 1-dimensional")
        mid = 0.0 if midpoint is None or np.asarray(midpoint).size == 0 else float(np.asarray(midpoint).flat[0])
        xout, sout, eout, nout = self._symmetrise_1d(self.p[0], self.s, self.e, self.npix, mid)

        wout = d1d(self)
        pax0 = int(self.pax[0]) - 1
        nbins_all_dims = self.axes.nbins_all_dims.copy()
        tot_range = self.img_range.copy()
        nbins_all_dims[pax0] = xout.size - 1
        tot_range[0, pax0] = float(np.min(xout))
        tot_range[1, pax0] = float(np.max(xout))
        wout._do_check_combo_arg = False
        wout._axes.nbins_all_dims = nbins_all_dims
        wout._axes.img_range = tot_range
        wout._s = sout.reshape(-1, 1, order="F")
        wout._e = eout.reshape(-1, 1, order="F")
        wout._npix = nout.reshape(-1, 1, order="F")
        wout.title = f"{wout.title} SYMMETRISED "
        wout._do_check_combo_arg = True
        wout.check_combo_arg()
        return wout

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _check_rebinning_axes_1d(self, w2: "d1d") -> Tuple[bool, bool, str]:
        if not np.allclose(self.alatt, w2.alatt) or not np.allclose(self.angdeg, w2.angdeg):
            return (
                False,
                False,
                "Horace error: rebinning impossible for 2 datasets where lattice parameters are different",
            )
        p1 = int(self.pax[0]) - 1
        p2 = int(w2.pax[0]) - 1
        w1_plane = self.u_to_rlu[:, p1]
        w2_plane = w2.u_to_rlu[:, p2]
        if np.allclose(w1_plane, w2_plane):
            return True, True, ""
        return (
            False,
            False,
            "Horace error: rebinning impossible for 2 1d datasets where x-axis is not the same",
        )

    @staticmethod
    def _combine_1d(
        x1: np.ndarray,
        y1: np.ndarray,
        e1: np.ndarray,
        n1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        e2: np.ndarray,
        n2: np.ndarray,
        tol: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x1 = np.asarray(x1, dtype=float).ravel()
        y1 = np.asarray(y1, dtype=float).ravel()
        e1 = np.asarray(e1, dtype=float).ravel()
        n1 = np.asarray(n1, dtype=float).ravel()
        x2 = np.asarray(x2, dtype=float).ravel()
        y2 = np.asarray(y2, dtype=float).ravel()
        e2 = np.asarray(e2, dtype=float).ravel()
        n2 = np.asarray(n2, dtype=float).ravel()

        lo1, hi1 = float(np.min(x1)), float(np.max(x1))
        lo2, hi2 = float(np.min(x2)), float(np.max(x2))
        eps = float(np.finfo(float).eps)

        if tol is None:
            if lo1 <= lo2 and hi1 >= hi2:
                xnew = x1.copy()
            elif lo2 <= lo1 and hi2 >= hi1:
                xnew = x2.copy()
            elif lo2 < lo1 and hi2 <= hi1:
                xnew = np.concatenate([x2[x2 < lo1], x1])
            else:
                xnew = np.concatenate([x1[x1 < lo2], x2])
        else:
            lo = min(lo1, lo2)
            hi = max(hi1, hi2)
            xnew = np.arange(lo, hi + tol - eps, tol)

        sig1, err1, npix1 = d1d._rebin_1d_general(x1, xnew, y1, e1, n1)
        sig2, err2, npix2 = d1d._rebin_1d_general(x2, xnew, y2, e2, n2)

        err1_old = err1.copy()
        err2_old = err2.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            f_err1 = np.where((sig1 != 0) & np.isfinite(err1 / sig1), err1 / sig1, 0.0)
            f_err2 = np.where((sig2 != 0) & np.isfinite(err2 / sig2), err2 / sig2, 0.0)

        biggest = max(float(np.max(sig1)), float(np.max(sig2)), 1.0)
        f_err1[f_err1 == 0] = 1e5 * biggest
        f_err2[f_err2 == 0] = 1e5 * biggest

        yout = (sig2 / f_err2 + sig1 / f_err1) / (1.0 / f_err2 + 1.0 / f_err1)
        eout = 1.0 / (1.0 / f_err2 + 1.0 / f_err1)
        nout = npix1 + npix2

        both_zero = (err1_old == 0) & (err2_old == 0)
        nout[both_zero] = 0.0
        yout[both_zero] = 0.0
        eout[both_zero] = 0.0
        eout = eout * yout
        return xnew, yout, eout, nout

    @staticmethod
    def _symmetrise_1d(
        xin: np.ndarray,
        yin: np.ndarray,
        ein: np.ndarray,
        nin: np.ndarray,
        midpoint: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        xin = np.asarray(xin, dtype=float).ravel()
        yin = np.asarray(yin, dtype=float).ravel()
        ein = np.asarray(ein, dtype=float).ravel()
        nin = np.asarray(nin, dtype=float).ravel()

        order1 = np.argsort(xin[1:])
        xin = np.sort(xin)
        yin = yin[order1]
        ein = ein[order1]
        nin = nin[order1]

        x2 = 2.0 * midpoint - xin
        order2 = np.argsort(x2[1:])
        x2 = np.sort(x2)
        y2 = yin[order2]
        e2 = ein[order2]
        n2 = nin[order2]

        eps = float(np.finfo(float).eps)
        if np.max(xin) >= np.max(x2):
            x_rebin = xin[xin >= midpoint]
        else:
            x_rebin = x2[x2 >= midpoint]
        if x_rebin.size == 0 or np.min(x_rebin) > midpoint + eps:
            x_rebin = np.concatenate([[midpoint], x_rebin])

        keep1 = xin[:-1] >= midpoint
        xin_sub = xin[xin >= midpoint]
        yin_sub, ein_sub, nin_sub = yin[keep1], ein[keep1], nin[keep1]

        keep2 = x2[:-1] >= midpoint
        x2_sub = x2[x2 >= midpoint]
        y2_sub, e2_sub, n2_sub = y2[keep2], e2[keep2], n2[keep2]

        if xin_sub.size > 1 and yin_sub.size > 0:
            stemp1, etemp1, ntemp1 = d1d._rebin_1d_general(xin_sub, x_rebin, yin_sub, ein_sub, nin_sub)
        else:
            stemp1 = np.zeros(x_rebin.size - 1, dtype=float)
            etemp1 = np.zeros_like(stemp1)
            ntemp1 = np.zeros_like(stemp1)

        if x2_sub.size > 1 and y2_sub.size > 0:
            stemp2, etemp2, ntemp2 = d1d._rebin_1d_general(x2_sub, x_rebin, y2_sub, e2_sub, n2_sub)
        else:
            stemp2 = np.zeros(x_rebin.size - 1, dtype=float)
            etemp2 = np.zeros_like(stemp2)
            ntemp2 = np.zeros_like(stemp2)

        return d1d._combine_1d(x_rebin, stemp1, etemp1, ntemp1, x_rebin, stemp2, etemp2, ntemp2, None)

    @staticmethod
    def _rebin_1d_general(
        xin: np.ndarray,
        xout: np.ndarray,
        s_in: np.ndarray,
        e_in: np.ndarray,
        n_in: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xin = np.asarray(xin, dtype=float).ravel()
        xout = np.asarray(xout, dtype=float).ravel()
        sin = np.asarray(s_in, dtype=float).ravel().copy()
        ein = np.asarray(e_in, dtype=float).ravel().copy()
        nin = np.asarray(n_in, dtype=float).ravel().copy()

        if sin.size != ein.size or sin.size != nin.size:
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                "Rebinning error: input arrays of signals, errors and npix must all be the same size",
            )
        if xin.size != sin.size + 1:
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                "Rebinning error: input array of signals must have 1 element fewer than array of bin boundaries",
            )
        if np.max(xin) > np.max(xout) or np.min(xin) < np.min(xout):
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                "Rebinning error: output bin boundaries must encompass input bin boundaries",
            )

        nin_test = (nin > 0).astype(float)
        ein = ein * nin_test
        sin = sin * nin_test
        zero_e = ein == 0
        sin[zero_e] = 0.0
        nin[zero_e] = 0.0
        bad = np.isnan(sin) | np.isnan(ein)
        sin[bad] = 0.0
        ein[bad] = 0.0
        nin[bad] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            ein_frac = np.where((sin != 0) & np.isfinite(ein / sin), ein / sin, 0.0)

        binlo = xin[:-1]
        binhi = xin[1:]
        order = np.argsort(binlo)
        lonew = binlo[order]
        hinew = binhi[order]
        snew = sin[order]
        enew = ein_frac[order]
        nnew = nin[order]
        xinnew = np.sort(xin)

        dx = np.diff(xinnew)
        if np.any(dx == 0):
            raise HoraceError(
                "HORACE:d1d:invalid_argument",
                "Rebinning error: input dataset should not have repeated bin boundaries",
            )
        eps = 1e-5 * float(np.min(dx))

        xout_s = np.sort(xout)
        xlo = xout_s[:-1]
        xhi = xout_s[1:]
        nout_bins = xlo.size

        sout = np.zeros(nout_bins, dtype=float)
        eout = np.zeros(nout_bins, dtype=float)
        nout = np.zeros(nout_bins, dtype=float)

        for i in range(nout_bins):
            inbin = (xinnew > xlo[i]) & (xinnew < xhi[i])
            ind = np.nonzero(inbin)[0]
            if ind.size == 1:
                binfrac = np.array([], dtype=float)
                cs, es, ns = d1d._rebin_1d_multibins(ind, binfrac, snew, enew, nnew, xinnew, eps, xlo, xhi, i)
                sout[i], eout[i], nout[i] = cs, es, ns
            elif ind.size == 0:
                lorange = xlo[i] < np.min(xinnew)
                hirange = xhi[i] > np.max(xinnew)
                if not lorange and not hirange:
                    cs, es, ns = d1d._rebin_1d_finegrid(snew, enew, nnew, xinnew, lonew, hinew, eps, xlo, xhi, i)
                    sout[i], eout[i], nout[i] = cs, es, ns
            else:
                binfrac = np.ones(ind.size - 1, dtype=float)
                cs, es, ns = d1d._rebin_1d_multibins(ind, binfrac, snew, enew, nnew, xinnew, eps, xlo, xhi, i)
                sout[i], eout[i], nout[i] = cs, es, ns

        bad_out = np.isnan(sout) | (eout == 0) | (nout == 0)
        sout[bad_out] = 0.0
        eout[bad_out] = 0.0
        nout[bad_out] = 0.0

        max_ein = float(np.max(ein_frac)) if ein_frac.size > 0 else 0.0
        if max_ein > 0:
            huge = eout > (1e3 * max_ein)
            sout[huge] = 0.0
            eout[huge] = 0.0
            nout[huge] = 0.0

        eout = eout * sout
        return sout, eout, nout

    @staticmethod
    def _rebin_1d_finegrid(
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
    ) -> Tuple[float, float, float]:
        lo, hi = float(xlo[i]), float(xhi[i])
        binind = np.nonzero((lonew <= lo) & (hinew >= hi))[0]
        if binind.size == 0:
            return 0.0, 0.0, 0.0
        idx = int(binind[0])
        insize = float(hinew[idx] - lonew[idx])
        outsize = hi - lo
        frac = outsize / insize if insize != 0 else 0.0
        if enew[idx] == 0 or frac == 0:
            return 0.0, 0.0, 0.0
        cumulsum = float(snew[idx])
        esum = float(enew[idx] / frac)
        nsum = float(frac * nnew[idx])
        return cumulsum, esum, nsum

    @staticmethod
    def _rebin_1d_multibins(
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
    ) -> Tuple[float, float, float]:
        enew_loc = enew.copy()
        max_s = float(np.max(snew)) if snew.size > 0 and np.max(snew) > 0 else 1.0
        enew_loc[(snew == 0) & (enew_loc == 0)] = 1e5 * max_s

        i_first = int(ind[0])
        i_last = int(ind[-1])
        dlo = float(xinnew[i_first] - xlo[i])
        dhi = float(xhi[i] - xinnew[i_last])

        has_lo = i_first != 0
        has_hi = i_last != (xinnew.size - 1)
        lofrac = eps + dlo / float(xinnew[i_first] - xinnew[i_first - 1]) if has_lo else 0.0
        hifrac = eps + dhi / float(xinnew[i_last + 1] - xinnew[i_last]) if has_hi else 0.0

        if has_lo and has_hi:
            bf = np.concatenate([[lofrac], binfrac, [hifrac]])
            sl = slice(i_first - 1, i_last + 1)
        elif has_hi:
            bf = np.concatenate([binfrac, [hifrac]])
            sl = slice(i_first, i_last + 1)
        elif has_lo:
            bf = np.concatenate([[lofrac], binfrac])
            sl = slice(i_first - 1, i_last)
        else:
            bf = binfrac
            sl = slice(i_first, i_last)

        s_sl = snew[sl]
        e_sl = enew_loc[sl]
        n_sl = nnew[sl]
        if bf.size == 0 or s_sl.size == 0:
            return 0.0, 0.0, 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            w = bf / e_sl
            w_sum = float(np.sum(w))
            if w_sum == 0 or not np.isfinite(w_sum):
                return 0.0, 0.0, 0.0
            cumulsum = float(np.sum(bf * s_sl / e_sl) / w_sum)
            esum = 1.0 / w_sum
            nout = float(np.sum(bf * n_sl))
        return cumulsum, esum, nout

