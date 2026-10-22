"""IX_dataset abstract base class and dataset utilities.

Translated from:
- ``@IX_dataset/IX_dataset.m`` and ``@IX_dataset/*.m`` (including ``private/``)
- ``bin_index.m``, ``upper_index.m``, ``lower_index.m``, ``bin_boundaries_simple.m``
- ``IX_dataset_nd.m``, ``isa_IX_dataset_nd.m``, ``rebin_IX_dataset_nd.m``,
  ``delta_IX_dataset_nd.m``, ``compact_array.m``
- ``integrate_1d_points.m``, ``deriv_xye.m``, ``deriv_xye_n.m``,
  ``unspike_xye.m``, ``unspike_xye_n.m``, ``combine_xye.m``
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any

import numpy as np

from pyhorace.core.data_op_interface import DataOpArray, data_op_interface
from pyhorace.core.serializable import Serializable
from pyhorace.data.data_plot_interface import data_plot_interface
from pyhorace.data.ix_axis import IX_axis, _is_cellstr, _is_empty, _is_numeric_array
from pyhorace.data.sigvar import SigVar, sigvar


# ======================================================================
# Index & Bin Boundary Utilities
# ======================================================================


def upper_index(arr: Any, val: Any) -> np.ndarray | int:
    """Return largest 1-based index ``m`` such that ``arr(m) <= val``.

    Returns ``0`` if ``val < arr(1)``, or ``zeros(size(val))`` if ``arr`` is empty.
    """
    a = np.asarray(arr, dtype=float).ravel()
    is_scalar = np.ndim(val) == 0
    v = np.asarray(val, dtype=float)
    if a.size == 0:
        res = np.zeros(v.shape, dtype=int)
        return int(res.item()) if is_scalar else res
    res = np.searchsorted(a, v, side="right").astype(int)
    return int(res.item()) if is_scalar else res


def lower_index(arr: Any, val: Any) -> np.ndarray | int:
    """Return smallest 1-based index ``m`` such that ``val <= arr(m)``.

    Returns ``numel(arr) + 1`` if ``arr(end) < val``, or ``ones(size(val))`` if ``arr`` is empty.
    """
    a = np.asarray(arr, dtype=float).ravel()
    is_scalar = np.ndim(val) == 0
    v = np.asarray(val, dtype=float)
    if a.size == 0:
        res = np.ones(v.shape, dtype=int)
        return int(res.item()) if is_scalar else res
    res = (np.searchsorted(a, v, side="left") + 1).astype(int)
    return int(res.item()) if is_scalar else res


def bin_index(x: Any, b: Any, inclusive: bool = False) -> np.ndarray:
    """Find 1-based bin index for a set of points.

    Parameters
    ----------
    x : array_like
        Array of points (must be monotonic increasing, but need not be strictly monotonic).
    b : array_like
        Bin boundaries (must be monotonic increasing, ``numel(b) >= 2``).
    inclusive : bool, optional
        Logical flag to indicate how to treat points on final bin boundary:
        - ``True``: points where ``x(i) == b(end)`` are included in final bin (``numel(b) - 1``).
        - ``False``: points where ``x(i) == b(end)`` are not included (``numel(b)``).

    Returns
    -------
    indx : np.ndarray
        1-based index of the bin to which the points belong, with the same shape as ``x``.
    """
    x_arr = np.asarray(x, dtype=float)
    b_arr = np.asarray(b, dtype=float).ravel()
    nb = b_arr.size
    if nb < 2:
        raise ValueError(
            "BIN_INDEX:invalid_argument: Must have at least two bin boundaries"
        )
    if x_arr.size == 0:
        return np.zeros(x_arr.shape, dtype=int)

    indx = np.searchsorted(b_arr, x_arr, side="right").astype(int)
    if inclusive:
        indx = np.where(x_arr == b_arr[-1], nb - 1, indx)
    return indx


def bin_boundaries_simple(xc: Any) -> tuple[np.ndarray, bool, str]:
    """Create simple bin boundaries halfway between point coordinates ``xc``."""
    arr = np.asarray(xc, dtype=float).ravel()
    nx = arr.size
    if nx > 1:
        del_x = np.diff(arr)
        if np.any(del_x <= 0):
            return np.empty(0, dtype=float), False, "Points must be strictly monotonic increasing"
        del0 = 0.5 * (arr[1] - arr[0])
        del1 = 0.5 * (arr[-1] - arr[-2])
        xb = np.concatenate(
            ([arr[0] - del0], 0.5 * (arr[1:] + arr[:-1]), [arr[-1] + del1])
        )
        return xb, True, ""
    if nx == 1:
        xb = np.array([arr[0] - 0.5, arr[0] + 0.5], dtype=float)
        return xb, True, ""
    return np.empty(0, dtype=float), False, "No points in input array"


def _is_stringmatchi(s: Any, ref: str) -> bool:
    """Return True if *s* is a non-empty case-insensitive prefix of *ref*."""
    if not isinstance(s, str) or len(s) == 0:
        return False
    return ref.lower().startswith(s.lower())


# ======================================================================
# Low-level Histogram Rebinning & Point Integration (1D, 2D, 3D)
# ======================================================================


def rebin_nd_hist(
    x: np.ndarray,
    s: np.ndarray,
    e: np.ndarray,
    xout: np.ndarray,
    iax: int = 1,
    ndim: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebin histogram distribution data along axis ``iax`` (1-based)."""
    x_arr = np.asarray(x, dtype=float).ravel()
    xout_arr = np.asarray(xout, dtype=float).ravel()
    s_arr = np.asarray(s, dtype=float)
    e_arr = np.asarray(e, dtype=float)

    mx = x_arr.size - 1
    if s_arr.ndim < ndim:
        s_arr = s_arr.reshape(s_arr.shape + (1,) * (ndim - s_arr.ndim))
        e_arr = e_arr.reshape(e_arr.shape + (1,) * (ndim - e_arr.ndim))

    ax0 = iax - 1
    if mx < 1 or s_arr.shape[ax0] != mx or s_arr.shape != e_arr.shape:
        raise ValueError("Check sizes of input arrays")

    nx = xout_arr.size - 1
    if nx < 1:
        raise ValueError("Check size of output axis axis")

    s_work = np.moveaxis(s_arr, ax0, 0)
    e_work = np.moveaxis(e_arr, ax0, 0)
    sz_out = (nx,) + s_work.shape[1:]
    sout = np.zeros(sz_out, dtype=float)
    eout = np.zeros(sz_out, dtype=float)

    iin = max(1, int(upper_index(x_arr, xout_arr[0])))
    iout = max(1, int(upper_index(xout_arr, x_arr[0])))
    if iin == mx + 1 or iout == nx + 1:
        sout = np.moveaxis(sout, 0, ax0)
        eout = np.moveaxis(eout, 0, ax0)
        return sout, eout

    while True:
        dx_ov = min(xout_arr[iout], x_arr[iin]) - max(
            xout_arr[iout - 1], x_arr[iin - 1]
        )
        sout[iout - 1] = sout[iout - 1] + dx_ov * s_work[iin - 1]
        eout[iout - 1] = eout[iout - 1] + (dx_ov * e_work[iin - 1]) ** 2
        if xout_arr[iout] >= x_arr[iin]:
            if iin < mx:
                iin += 1
            else:
                dx_bin = xout_arr[iout] - xout_arr[iout - 1]
                sout[iout - 1] = sout[iout - 1] / dx_bin
                eout[iout - 1] = np.sqrt(eout[iout - 1]) / dx_bin
                break
        else:
            dx_bin = xout_arr[iout] - xout_arr[iout - 1]
            sout[iout - 1] = sout[iout - 1] / dx_bin
            eout[iout - 1] = np.sqrt(eout[iout - 1]) / dx_bin
            if iout < nx:
                iout += 1
            else:
                break

    sout = np.moveaxis(sout, 0, ax0)
    eout = np.moveaxis(eout, 0, ax0)
    return sout, eout


def integrate_nd_points(
    x: np.ndarray,
    s: np.ndarray,
    e: np.ndarray,
    xout: np.ndarray,
    iax: int = 1,
    ndim: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoidal integration of point data along axis ``iax`` (1-based).

    Translated from ``integrate_1d_points.m`` / ``integrate_2d_x_points.m``.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    xout_arr = np.asarray(xout, dtype=float).ravel()
    s_arr = np.asarray(s, dtype=float)
    e_arr = np.asarray(e, dtype=float)

    nx = x_arr.size
    if nx < 2:
        raise ValueError(
            "Must have at least two data points to perform integration of point data"
        )
    if s_arr.ndim < ndim:
        s_arr = s_arr.reshape(s_arr.shape + (1,) * (ndim - s_arr.ndim))
        e_arr = e_arr.reshape(e_arr.shape + (1,) * (ndim - e_arr.ndim))

    ax0 = iax - 1
    if s_arr.shape[ax0] != nx or s_arr.shape != e_arr.shape:
        raise ValueError("Check sizes of input arrays")

    nb = xout_arr.size - 1
    if nb < 1:
        raise ValueError("Sizes of output integration limits array too small")

    s_work = np.moveaxis(s_arr, ax0, 0)
    e_work = np.moveaxis(e_arr, ax0, 0)
    trail_shape = s_work.shape[1:]
    sout = np.zeros((nb,) + trail_shape, dtype=float)
    eout = np.zeros((nb,) + trail_shape, dtype=float)

    if x_arr[-1] <= xout_arr[0] or xout_arr[-1] <= x_arr[0]:
        return np.moveaxis(sout, 0, ax0), np.moveaxis(eout, 0, ax0)

    if xout_arr[0] >= x_arr[0]:
        ml = int(lower_index(x_arr, xout_arr[0]))
        ib = 1
    else:
        ml = 1
        ib = int(upper_index(xout_arr, x_arr[0]))

    while ib <= nb:
        mu = ml - 1
        while mu < nx and x_arr[mu] <= xout_arr[ib]:
            mu += 1
        val, errbar = _single_integrate_points_axis0(
            x_arr, s_work, e_work, xout_arr[ib - 1], xout_arr[ib], ml, mu, trail_shape
        )
        sout[ib - 1] = val
        eout[ib - 1] = errbar
        if mu == nx or ib == nb:
            break
        ib += 1
        if x_arr[mu - 1] < xout_arr[ib - 1]:
            ml = mu + 1
        else:
            ml = mu

    return np.moveaxis(sout, 0, ax0), np.moveaxis(eout, 0, ax0)


def _single_integrate_points_axis0(
    x: np.ndarray,
    s: np.ndarray,
    e: np.ndarray,
    xmin: float,
    xmax: float,
    ml: int,
    mu: int,
    trail_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Single bin trapezoidal integration along axis 0 (1-based ``ml``, ``mu``)."""
    nx = x.size
    if mu < ml:
        ilo = max(ml - 1, 1) - 1
        ihi = min(mu + 1, nx) - 1
        dx = x[ihi] - x[ilo]
        w_hi = (xmax - x[ilo]) + (xmin - x[ilo])
        w_lo = (x[ihi] - xmax) + (x[ihi] - xmin)
        fac = 0.5 * ((xmax - xmin) / dx)
        val = fac * (s[ihi] * w_hi + s[ilo] * w_lo)
        errbar = fac * np.sqrt((e[ihi] * w_hi) ** 2 + (e[ilo] * w_lo) ** 2)
        return val, errbar

    ml0 = ml - 1
    mu0 = mu - 1
    if ml > 1:
        x_ml_m1 = x[ml0 - 1]
        x_ml = x[ml0]
        x1eff = (xmin * (xmin - x_ml_m1) + x_ml_m1 * (x_ml - xmin)) / (x_ml - x_ml_m1)
        denom1 = (x_ml - x_ml_m1) + (xmin - x_ml_m1)
        s1eff = s[ml0 - 1] * (x_ml - xmin) / denom1
        e1eff = e[ml0 - 1] * (x_ml - xmin) / denom1
    else:
        x1eff = x[ml0]
        s1eff = np.zeros(trail_shape, dtype=float)
        e1eff = np.zeros(trail_shape, dtype=float)

    if mu < nx:
        x_mu = x[mu0]
        x_mu_p1 = x[mu0 + 1]
        xneff = (xmax * (x_mu_p1 - xmax) + x_mu_p1 * (xmax - x_mu)) / (x_mu_p1 - x_mu)
        denomn = (x_mu_p1 - x_mu) + (x_mu_p1 - xmax)
        sneff = s[mu0 + 1] * (xmax - x_mu) / denomn
        eneff = e[mu0 + 1] * (xmax - x_mu) / denomn
    else:
        xneff = x[-1]
        sneff = np.zeros(trail_shape, dtype=float)
        eneff = np.zeros(trail_shape, dtype=float)

    val = (x[ml0] - x1eff) * (s[ml0] + s1eff)
    errbar = (e1eff * (x[ml0] - x1eff)) ** 2

    if mu == ml:
        errbar = errbar + (e[ml0] * (xneff - x1eff)) ** 2
    elif mu == ml + 1:
        val = val + (s[mu0] + s[ml0]) * (x[mu0] - x[ml0])
        errbar = (
            errbar
            + (e[ml0] * (x[ml0 + 1] - x1eff)) ** 2
            + (e[mu0] * (xneff - x[mu0 - 1])) ** 2
        )
    else:
        bcast = (-1,) + (1,) * len(trail_shape)
        dx1 = (x[ml0 + 1 : mu0 + 1] - x[ml0:mu0]).reshape(bcast)
        val = val + np.sum((s[ml0 + 1 : mu0 + 1] + s[ml0:mu0]) * dx1, axis=0)
        dx2 = (x[ml0 + 2 : mu0 + 1] - x[ml0 : mu0 - 1]).reshape(bcast)
        errbar = (
            errbar
            + (e[ml0] * (x[ml0 + 1] - x1eff)) ** 2
            + (e[mu0] * (xneff - x[mu0 - 1])) ** 2
            + np.sum((e[ml0 + 1 : mu0] * dx2) ** 2, axis=0)
        )

    val = val + (xneff - x[mu0]) * (s[mu0] + sneff)
    errbar = errbar + (eneff * (xneff - x[mu0])) ** 2

    return 0.5 * val, 0.5 * np.sqrt(errbar)


# ======================================================================
# Numerical Derivative, Unspike, and Combine Utilities
# ======================================================================


def deriv_xye(
    x: np.ndarray, y: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical first derivative of ``(x, y, e)`` points (translated from ``deriv_xye.m``)."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_in = np.asarray(y, dtype=float)
    e_in = np.asarray(e, dtype=float)
    np_pts = x_arr.size
    if y_in.size != np_pts or e_in.size != np_pts:
        raise ValueError("x,y,e arrays must have equal lengths")
    if np_pts == 0:
        return y_in.copy(), e_in.copy()
    if np_pts == 1:
        return np.zeros_like(y_in), np.zeros_like(e_in)

    y_vec = y_in.ravel()
    e_vec = e_in.ravel()
    dx = x_arr[2:] - x_arr[:-2]
    dy = y_vec[2:] - y_vec[:-2]
    ybeg = (y_vec[1] - y_vec[0]) / (x_arr[1] - x_arr[0])
    yend = (y_vec[-1] - y_vec[-2]) / (x_arr[-1] - x_arr[-2])
    yd = np.concatenate(([ybeg], dy / dx, [yend]))

    ebeg = np.sqrt(e_vec[1] ** 2 + e_vec[0] ** 2) / (x_arr[1] - x_arr[0])
    eend = np.sqrt(e_vec[-1] ** 2 + e_vec[-2] ** 2) / (x_arr[-1] - x_arr[-2])
    ed = np.concatenate(
        ([ebeg], np.sqrt(e_vec[2:] ** 2 + e_vec[:-2] ** 2) / dx, [eend])
    )
    return yd.reshape(y_in.shape), ed.reshape(e_in.shape)


def deriv_xye_n(
    iax: int, x: np.ndarray, y: np.ndarray, e: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical first derivative along 1-based axis ``iax`` (translated from ``deriv_xye_n.m``)."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    if y_arr.shape != e_arr.shape:
        raise ValueError("Check y, e array sizes are the same")
    nax = y_arr.ndim
    if iax < 1 or iax > nax or int(iax) != iax:
        raise ValueError(f"Axis number to unspike must lie in range 1-{nax}")
    ax0 = iax - 1
    if x_arr.size != y_arr.shape[ax0]:
        raise ValueError(
            "Number of points along axis to be differentiated does not match size of signal and error arrays along that axis"
        )
    if y_arr.size == 0:
        return y_arr.copy(), e_arr.copy()
    if x_arr.size == 1:
        return np.zeros_like(y_arr), np.zeros_like(e_arr)

    y_work = np.moveaxis(y_arr, ax0, 0)
    e_work = np.moveaxis(e_arr, ax0, 0)
    orig_perm_shape = y_work.shape
    y_2d = y_work.reshape(x_arr.size, -1)
    e_2d = e_work.reshape(x_arr.size, -1)
    yd_2d = np.zeros_like(y_2d)
    ed_2d = np.zeros_like(e_2d)
    for col in range(y_2d.shape[1]):
        yd_2d[:, col], ed_2d[:, col] = deriv_xye(x_arr, y_2d[:, col], e_2d[:, col])
    yd = np.moveaxis(yd_2d.reshape(orig_perm_shape), 0, ax0)
    ed = np.moveaxis(ed_2d.reshape(orig_perm_shape), 0, ax0)
    return yd, ed


def unspike_xye(
    xin: np.ndarray,
    yin: np.ndarray,
    ein: np.ndarray,
    ymin: float | None = None,
    ymax: float | None = None,
    fac: float | None = None,
    sfac: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove isolated spikes from 1D ``(x, y, e)`` data (translated from ``unspike_xye.m``)."""
    x_in = np.asarray(xin, dtype=float).ravel()
    y_in = np.asarray(yin, dtype=float)
    e_in = np.asarray(ein, dtype=float)
    np_pts = x_in.size
    if y_in.size != np_pts or e_in.size != np_pts:
        raise ValueError("x,y,e arrays must have equal lengths")
    if np_pts == 0:
        return y_in.copy(), e_in.copy()

    y_vec = y_in.ravel()
    e_vec = e_in.ravel()
    ymin_val = np.nan if ymin is None or _is_empty(ymin) else float(ymin)
    ymax_val = np.nan if ymax is None or _is_empty(ymax) else float(ymax)
    fac_val = 2.0 if fac is None or _is_empty(fac) else abs(float(fac))
    sfac_val = 5.0 if sfac is None or _is_empty(sfac) else abs(float(sfac))

    filter_low = not (np.isnan(ymin_val) or ymin_val == -np.inf)
    filter_high = not (np.isnan(ymax_val) or ymax_val == np.inf)
    if not filter_low and not filter_high:
        ind_ok = np.arange(np_pts)
    else:
        ok_mask = np.ones(np_pts, dtype=bool)
        if filter_low:
            ok_mask &= y_vec >= ymin_val
        if filter_high:
            ok_mask &= y_vec <= ymax_val
        ind_ok = np.flatnonzero(ok_mask)

    x = x_in[ind_ok]
    y = y_vec[ind_ok]
    e = e_vec[ind_ok]

    if x.size >= 3:
        y0, e0 = y[1:-1], e[1:-1]
        ylo, elo = y[:-2], e[:-2]
        yhi, ehi = y[2:], e[2:]
        dlo = y0 - ylo
        dhi = y0 - yhi
        bad_sigma = (
            (np.sign(dlo) == np.sign(dhi))
            & (np.abs(dlo) > sfac_val * np.sqrt(e0**2 + elo**2))
            & (np.abs(dhi) > sfac_val * np.sqrt(e0**2 + ehi**2))
        )
        bad_dev = (
            (np.sign(y0) * np.sign(ylo) >= 0)
            & (np.sign(y0) * np.sign(yhi) >= 0)
            & (
                ((np.abs(y0) > fac_val * np.abs(ylo)) & (np.abs(y0) > fac_val * np.abs(yhi)))
                | (
                    (np.abs(y0) < (1.0 / fac_val) * np.abs(ylo))
                    & (np.abs(y0) < (1.0 / fac_val) * np.abs(yhi))
                )
            )
        )
        ok_sub = np.concatenate(([True], ~(bad_sigma & bad_dev), [True]))
        ind_ok = ind_ok[ok_sub]
        x, y, e = x[ok_sub], y[ok_sub], e[ok_sub]

    if x.size == 0:
        return np.full_like(y_in, np.nan), np.full_like(e_in, np.nan)
    if x.size == 1:
        return np.full_like(y_in, y[0]), np.full_like(e_in, e[0])

    ok_full = np.zeros(np_pts, dtype=bool)
    ok_full[ind_ok] = True
    sum_ok = np.cumsum(ok_full)
    ind_bad = np.flatnonzero(~ok_full)
    ix_lo = sum_ok[ind_bad]

    yout = y_vec.copy()
    eout = e_vec.copy()

    extrap_lo = ind_bad[ix_lo == 0]
    extrap_hi = ind_bad[ix_lo == ind_ok.size]
    if extrap_lo.size > 0:
        yout[extrap_lo] = y_vec[ind_ok[0]]
        eout[extrap_lo] = e_vec[ind_ok[0]]
    if extrap_hi.size > 0:
        yout[extrap_hi] = y_vec[ind_ok[-1]]
        eout[extrap_hi] = e_vec[ind_ok[-1]]

    i_interp = (ix_lo > 0) & (ix_lo < ind_ok.size)
    if np.any(i_interp):
        ind_interp = ind_bad[i_interp]
        lo_idx = ind_ok[ix_lo[i_interp] - 1]
        hi_idx = ind_ok[ix_lo[i_interp]]
        x1, y1, e1 = x_in[lo_idx], y_vec[lo_idx], e_vec[lo_idx]
        x2, y2, e2 = x_in[hi_idx], y_vec[hi_idx], e_vec[hi_idx]
        xi = x_in[ind_interp]
        a1 = (x2 - xi) / (x2 - x1)
        a2 = (xi - x1) / (x2 - x1)
        yout[ind_interp] = a1 * y1 + a2 * y2
        eout[ind_interp] = np.sqrt((a1 * e1) ** 2 + (a2 * e2) ** 2)

    return yout.reshape(y_in.shape), eout.reshape(e_in.shape)


def unspike_xye_n(
    iax: int, x: np.ndarray, y: np.ndarray, e: np.ndarray, *args: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Remove spikes along 1-based axis ``iax`` (translated from ``unspike_xye_n.m``)."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    ax0 = iax - 1
    y_work = np.moveaxis(y_arr, ax0, 0)
    e_work = np.moveaxis(e_arr, ax0, 0)
    orig_perm_shape = y_work.shape
    y_2d = y_work.reshape(x_arr.size, -1)
    e_2d = e_work.reshape(x_arr.size, -1)
    yo_2d = np.zeros_like(y_2d)
    eo_2d = np.zeros_like(e_2d)
    for col in range(y_2d.shape[1]):
        yo_2d[:, col], eo_2d[:, col] = unspike_xye(
            x_arr, y_2d[:, col], e_2d[:, col], *args
        )
    return (
        np.moveaxis(yo_2d.reshape(orig_perm_shape), 0, ax0),
        np.moveaxis(eo_2d.reshape(orig_perm_shape), 0, ax0),
    )


# ======================================================================
# Rebin / Integrate / Cut Descriptor Parsing Helpers
# ======================================================================


def _rebin_point_integration_check(
    nax: int, option: Any
) -> tuple[np.ndarray, bool, str]:
    """Validate point integration option ('integrate' or 'average')."""
    if isinstance(option, str):
        opts = [option] * nax
    elif _is_cellstr(option):
        flat = list(np.asarray(option).flat) if isinstance(option, np.ndarray) else list(option)
        if len(flat) == 1:
            opts = [flat[0]] * nax
        elif len(flat) == nax:
            opts = flat
        else:
            return (
                np.empty(0, dtype=bool),
                False,
                "Option must be 'integrate' or 'average' or a cell array of those options",
            )
    else:
        return (
            np.empty(0, dtype=bool),
            False,
            "Option must be 'integrate' or 'average' or a cell array of those options",
        )

    point_integration = np.zeros(nax, dtype=bool)
    for i, opt_i in enumerate(opts):
        if _is_stringmatchi(opt_i, "integrate"):
            point_integration[i] = True
        elif len(opt_i) == 0 or _is_stringmatchi(opt_i, "average"):
            point_integration[i] = False
        else:
            return (
                np.empty(0, dtype=bool),
                False,
                "Option must be 'integrate' or 'average'",
            )
    return point_integration, True, ""


def _rebin_boundaries_description_parse_single(
    opt: dict[str, bool], xvals: Any
) -> tuple[bool, np.ndarray, bool, bool, bool, str]:
    """Parse a single axis rebin/integration boundary descriptor."""
    if _is_empty(xvals):
        if opt["empty_is_full_range"]:
            return True, np.array([-np.inf, np.inf]), True, False, False, ""
        return True, np.array([-np.inf, 0.0, np.inf]), True, True, True, ""

    if not _is_numeric_array(xvals) or isinstance(xvals, str):
        return False, np.empty(0), False, False, False, "Rebin descriptor must be a numeric vector"

    arr = np.asarray(xvals, dtype=float)
    if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
        return False, np.empty(0), False, False, False, "Rebin descriptor must be a numeric vector"
    arr = arr.ravel()
    if np.any(np.isnan(arr)):
        return False, np.empty(0), False, False, False, "Rebin descriptor cannot contain any NaNs"

    n = arr.size
    if n == 1:
        if arr[0] == 0:
            return True, np.array([-np.inf, 0.0, np.inf]), True, True, True, ""
        return True, np.array([-np.inf, arr[0], np.inf]), True, True, False, ""

    if n == 2:
        if arr[0] >= arr[1]:
            return (
                False,
                np.empty(0),
                False,
                False,
                False,
                "Upper limit must be greater than lower limit in a rebin descriptor",
            )
        any_inf = bool(np.any(np.isinf(arr)))
        if opt["range_is_one_bin"]:
            return True, arr.copy(), any_inf, False, False, ""
        return True, np.array([arr[0], 0.0, arr[1]]), any_inf, True, True, ""

    # n >= 3
    if opt["array_is_descriptor"]:
        if opt["bin_boundaries"]:
            if n % 2 != 1:
                return (
                    False,
                    np.empty(0),
                    False,
                    False,
                    False,
                    "Check rebin descriptor has correct number of elements",
                )
            if not np.all(np.diff(arr[0::2]) > 0):
                return (
                    False,
                    np.empty(0),
                    False,
                    False,
                    False,
                    "Bin ranges in rebin descriptor must be strictly monotonic increasing",
                )
            xvals_lo = arr[0:-1:2].copy()
            if np.isinf(xvals_lo[0]):
                xvals_lo = np.array([arr[2]])
            steps = arr[1:-1:2]
            if not np.all((xvals_lo > 0) | (steps >= 0)):
                return (
                    False,
                    np.empty(0),
                    False,
                    False,
                    False,
                    "Rebin descriptor cannot have logarithmic bins for negative axis values",
                )
            any_inf = bool(np.isinf(arr[0]) or np.isinf(arr[-1]))
            any_dx0 = bool(np.any(steps == 0))
            return True, arr.copy(), any_inf, True, any_dx0, ""
        else:
            if not (n == 3 and arr[0] < arr[2] and arr[1] > 0):
                return (
                    False,
                    np.empty(0),
                    False,
                    False,
                    False,
                    "Rebin descriptor for bin centres must have three elements in the form [xlo,dx,xhi], xlo<xhi and dx>0",
                )
            xb = np.array([arr[0] - arr[1] / 2.0, arr[1], arr[2] + arr[1] / 2.0])
            any_inf = bool(np.isinf(xb[0]) or np.isinf(xb[-1]))
            return True, xb, any_inf, True, False, ""
    else:
        if not np.all(np.diff(arr) > 0):
            return (
                False,
                np.empty(0),
                False,
                False,
                False,
                "Rebin boundaries must be strictly monotonic increasing vector i.e. all bin widths > 0",
            )
        any_inf = bool(np.isinf(arr[0]) or np.isinf(arr[-1]))
        return True, arr.copy(), any_inf, False, False, ""


def _rebin_boundaries_description_parse(
    nax: int, opt: dict[str, bool], *args: Any
) -> tuple[bool, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, str]:
    """Parse multi-axis rebin/integration boundary descriptors."""
    if len(args) == 0:
        xbounds = []
        any_lim_inf = np.ones(nax, dtype=bool)
        is_descriptor = np.zeros(nax, dtype=bool)
        any_dx_zero = np.zeros(nax, dtype=bool)
        for _ in range(nax):
            if opt["empty_is_full_range"]:
                xbounds.append(np.array([-np.inf, np.inf]))
            else:
                xbounds.append(np.array([-np.inf, 0.0, np.inf]))
                is_descriptor[:] = True
                any_dx_zero[:] = True
        return True, xbounds, any_lim_inf, is_descriptor, any_dx_zero, ""

    if nax == 1 and not (
        len(args) == 1 and (_is_empty(args[0]) or _is_numeric_array(args[0]))
    ):
        if all(
            _is_numeric_array(a) and np.asarray(a).size == 1 for a in args
        ):
            vec = np.array([float(np.asarray(a).item()) for a in args])
            ok, xb, inf_f, desc_f, dx0_f, mess = (
                _rebin_boundaries_description_parse_single(opt, vec)
            )
            if not ok:
                return False, [], np.empty(0, dtype=bool), np.empty(0, dtype=bool), np.empty(0, dtype=bool), mess
            return (
                True,
                [xb],
                np.array([inf_f]),
                np.array([desc_f]),
                np.array([dx0_f]),
                "",
            )
        return (
            False,
            [],
            np.empty(0, dtype=bool),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=bool),
            "Check bin boundary descriptor is a numeric vector or list of scalars",
        )

    if len(args) == nax:
        xbounds = []
        any_lim_inf = np.zeros(nax, dtype=bool)
        is_descriptor = np.zeros(nax, dtype=bool)
        any_dx_zero = np.zeros(nax, dtype=bool)
        for i in range(nax):
            ok, xb, inf_f, desc_f, dx0_f, mess = (
                _rebin_boundaries_description_parse_single(opt, args[i])
            )
            if not ok:
                return False, [], np.empty(0, dtype=bool), np.empty(0, dtype=bool), np.empty(0, dtype=bool), mess
            xbounds.append(xb)
            any_lim_inf[i] = inf_f
            is_descriptor[i] = desc_f
            any_dx_zero[i] = dx0_f
        return True, xbounds, any_lim_inf, is_descriptor, any_dx_zero, ""

    return (
        False,
        [],
        np.empty(0, dtype=bool),
        np.empty(0, dtype=bool),
        np.empty(0, dtype=bool),
        "Check number of bin boundary descriptors matches number of rebin axes",
    )


def _low_limit(x: float, xref: float, dx: float) -> tuple[float, bool]:
    if dx == 0:
        return x, True
    if dx > 0:
        xlo = xref + dx * np.floor((x - xref) / dx)
        if xlo > x:
            xlo -= dx
        elif xlo < x and xlo + dx <= x:
            xlo += dx
        return float(xlo), True
    if dx < 0 and x > 0 and xref > 0:
        step = np.log(1.0 + abs(dx))
        n = np.floor(np.log(xref / x) / step)
        xlo = np.exp(np.log(xref) - n * step)
        if xlo > x:
            xlo /= 1.0 + abs(dx)
        elif xlo < x and xlo * (1.0 + abs(dx)) <= x:
            xlo *= 1.0 + abs(dx)
        return float(xlo), True
    return 0.0, False


def _high_limit(x: float, xref: float, dx: float) -> tuple[float, bool]:
    if dx == 0:
        return x, True
    if dx > 0:
        xhi = xref + dx * np.ceil((x - xref) / dx)
        if xhi < x:
            xhi += dx
        elif xhi > x and xhi - dx >= x:
            xhi -= dx
        return float(xhi), True
    if dx < 0 and x > 0 and xref > 0:
        step = np.log(1.0 + abs(dx))
        n = np.ceil(np.log(x / xref) / step)
        xhi = np.exp(np.log(xref) + n * step)
        if xhi < x:
            xhi *= 1.0 + abs(dx)
        elif xhi > x and xhi / (1.0 + abs(dx)) >= x:
            xhi /= 1.0 + abs(dx)
        return float(xhi), True
    return 0.0, False


def _rebin_boundaries_description_resolve_inf(
    xbounds: np.ndarray, is_descriptor: bool, xlo: float, xhi: float
) -> tuple[np.ndarray, bool, str]:
    """Resolve -Inf and +Inf in bin boundaries or rebin descriptor."""
    xb = np.asarray(xbounds, dtype=float).copy()
    if xb[0] == -np.inf or xb[-1] == np.inf:
        if is_descriptor:
            if xb.size == 3 and np.isinf(xb[0]) and np.isinf(xb[2]):
                dx = xb[1]
                if dx == 0:
                    return np.empty(0, dtype=float), True, ""
                if dx > 0:
                    lo, _ = _low_limit(xlo, 0.0, dx)
                    hi, _ = _high_limit(xhi, 0.0, dx)
                    return np.array([lo, dx, hi]), True, ""
                if xlo > 0:
                    lo, _ = _low_limit(xlo, 1.0, dx)
                    hi, _ = _high_limit(xhi, 1.0, dx)
                    return np.array([lo, dx, hi]), True, ""
                return (
                    np.empty(0, dtype=float),
                    False,
                    "If lower limit is less than or equal to zero, the rebin descriptor [-Inf,dx,Inf] with dx<0 is invalid",
                )
            else:
                if xb[0] == -np.inf:
                    if xb[2] > xlo:
                        lo, ok = _low_limit(xlo, xb[2], xb[1])
                        if not ok:
                            return (
                                np.empty(0, dtype=float),
                                False,
                                "Unable to resolve rebin descriptor beginning [-Inf,dx,...] - check dx and lower limit of data",
                            )
                        xb[0] = lo
                    else:
                        xb = xb[2:]
                        if xb.size < 3:
                            return xb, True, ""
                if xb[-1] == np.inf:
                    if xb[-3] < xhi:
                        hi, ok = _high_limit(xhi, xb[-3], xb[-2])
                        if not ok:
                            return (
                                np.empty(0, dtype=float),
                                False,
                                "Unable to resolve rebin descriptor ending [...dx,Inf] - check dx and upper limit of data",
                            )
                        xb[-1] = hi
                    else:
                        xb = xb[:-2]
                        if xb.size < 3:
                            return xb, True, ""
        else:
            if xb[0] == -np.inf:
                if xb[1] > xlo:
                    xb[0] = xlo
                else:
                    xb = xb[1:]
                    if xb.size < 2:
                        return xb, True, ""
            if xb[-1] == np.inf:
                if xb[-2] < xhi:
                    xb[-1] = xhi
                else:
                    xb = xb[:-1]
                    if xb.size < 2:
                        return xb, True, ""
    return xb, True, ""


def _bin_boundaries_from_descriptor(
    xbounds: Any, x_in: Any = None
) -> tuple[np.ndarray, bool, str]:
    """Construct explicit bin boundaries from a descriptor ``[x1, dx1, x2, ...]``."""
    xb = np.asarray(xbounds, dtype=float).ravel()
    mx = xb.size
    small = 1.0e-10
    if mx < 2 or (mx > 2 and mx % 2 != 1):
        return np.empty(0, dtype=float), False, "Check size of xbounds array"

    x_in_arr = None if x_in is None else np.asarray(x_in, dtype=float).ravel()
    if x_in_arr is not None and x_in_arr.size > 1 and np.min(np.diff(x_in_arr)) <= 0:
        return (
            np.empty(0, dtype=float),
            False,
            "Input x array is not strictly monotonic increasing",
        )

    parts: list[np.ndarray] = []
    n_seg = mx // 2
    xhi = xb[-1]
    for i in range(n_seg):
        if mx == 2:
            xlo, del_x, xhi = xb[0], 0.0, xb[1]
        else:
            xlo, del_x, xhi = xb[2 * i], xb[2 * i + 1], xb[2 * i + 2]
        if xhi <= xlo:
            return (
                np.empty(0, dtype=float),
                False,
                "Check boundaries strictly monotonically increasing",
            )
        if del_x > 0:
            n = int(np.floor((xhi - xlo) / del_x - small))
            if xlo + n * del_x < xhi:
                n += 1
            parts.append(np.array([xlo]))
            if n > 1:
                parts.append(xlo + del_x * np.arange(1, n, dtype=float))
        elif del_x < 0:
            if xlo <= 0:
                return (
                    np.empty(0, dtype=float),
                    False,
                    "Logarithmic bins starting with XLO <= 0 forbidden",
                )
            logdel = np.log(1.0 - del_x)
            n = int(np.floor(np.log(xhi / xlo) / logdel - small))
            if xlo * np.exp(n * logdel) < xhi:
                n += 1
            parts.append(np.array([xlo]))
            if n > 1:
                parts.append(xlo * np.exp(np.arange(1, n, dtype=float) * logdel))
        else:
            if x_in_arr is None:
                return (
                    np.empty(0, dtype=float),
                    False,
                    "No input x array provided to supply bin boundaries",
                )
            m_in = x_in_arr.size
            imin = int(lower_index(x_in_arr, xlo))
            imax = int(upper_index(x_in_arr, xhi))
            parts.append(np.array([xlo]))
            if imin <= m_in and imax >= 1:
                if x_in_arr[imin - 1] == xlo:
                    imin += 1
                if x_in_arr[imax - 1] == xhi:
                    imax -= 1
                if imax >= imin:
                    parts.append(x_in_arr[imin - 1 : imax])

    parts.append(np.array([xhi]))
    return np.concatenate(parts), True, ""


# ======================================================================
# IX_dataset Base Class
# ======================================================================


class IX_dataset(data_op_interface, Serializable, data_plot_interface):
    """Abstract base class for ``IX_dataset_1d``, ``IX_dataset_2d``, and ``IX_dataset_3d``.

    Combines ``IX_dataset``, ``IX_data_Nd``, and ``data_plot_interface``.
    """

    _fields_to_save = ["title", "signal", "error", "s_axis"]

    def __init__(self) -> None:
        data_op_interface.__init__(self)
        Serializable.__init__(self)
        data_plot_interface.__init__(self)
        self._title: list[str] | str = []
        self._signal: np.ndarray = np.zeros((0, 1), dtype=float)
        self._error: np.ndarray = np.zeros((0, 1), dtype=float)
        self._s_axis: IX_axis = IX_axis("Counts")
        self._xyz: list[np.ndarray] = []
        self._xyz_axis: list[IX_axis] = []
        self._xyz_distribution: np.ndarray = np.empty(0, dtype=bool)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def title(self) -> list[str] | str:
        """Dataset title (cellstr / list of strings)."""
        return self._title

    @title.setter
    def title(self, val: Any) -> None:
        self._check_and_set_title(val)

    @property
    def signal(self) -> np.ndarray:
        """Signal array."""
        return self._signal

    @signal.setter
    def signal(self, val: Any) -> None:
        self._check_and_set_sig_err("signal", val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def error(self) -> np.ndarray:
        """Standard deviations on signal array."""
        return self._error

    @error.setter
    def error(self, val: Any) -> None:
        self._check_and_set_sig_err("error", val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def s_axis(self) -> IX_axis:
        """Signal ``IX_axis`` description."""
        return self._s_axis

    @s_axis.setter
    def s_axis(self, val: Any) -> None:
        self._s_axis = self.check_and_build_axis(val)

    # ------------------------------------------------------------------
    # Abstract Interface for Subclasses
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def ndim() -> int:
        """Return dimensionality (1, 2, or 3)."""

    @abstractmethod
    def _check_and_set_sig_err(self, field_name: str, value: Any) -> None:
        """Validate and set ``_signal`` or ``_error``."""

    @staticmethod
    @abstractmethod
    def rebin_hist(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rebin histogram data along axis ``iax``."""

    @staticmethod
    @abstractmethod
    def integrate_points(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate point data along axis ``iax``."""

    @abstractmethod
    def init(self, *args: Any, **kwargs: Any) -> IX_dataset:
        """Re-initialise dataset in-place."""

    @abstractmethod
    def dimensions(self) -> tuple[int, Any]:
        """Return ``(nd, sz)`` of dataset."""

    @abstractmethod
    def ishistogram(self, n: int | None = None) -> Any:
        """Return True where axis ``n`` (or all axes) contains histogram bin boundaries."""

    @abstractmethod
    def axis(self, n: int | None = None) -> tuple[Any, Any]:
        """Return ``(ax_struct, hist_flag)`` for axis ``n`` (or all axes)."""

    # ------------------------------------------------------------------
    # Concrete Methods on IX_dataset
    # ------------------------------------------------------------------

    @staticmethod
    def bin_boundaries_from_descriptor(
        xbounds: Any, x_in: Any = None
    ) -> np.ndarray:
        """Construct bin boundaries from a descriptor."""
        x_out, ok, mess = _bin_boundaries_from_descriptor(xbounds, x_in)
        if not ok:
            raise ValueError(f"IX_dataset:invalid_argument: {mess}")
        return x_out

    def get_xyz(self, nd: int | None = None) -> list[np.ndarray] | np.ndarray:
        """Return cell array of axis arrays or the ``nd``-th axis array (1-based)."""
        if nd is None:
            return [arr.copy() for arr in self._xyz]
        return self._xyz[nd - 1]

    def get_signal(self) -> np.ndarray:
        """Return ``_signal`` array."""
        return self._signal

    def get_error(self) -> np.ndarray:
        """Return ``_error`` array."""
        return self._error

    def get_isdistribution(self) -> np.ndarray:
        """Return boolean array of distribution flags for each axis."""
        return self._xyz_distribution.copy()

    def copy(self) -> IX_dataset:
        """Return a deep copy of this dataset."""
        return copy.deepcopy(self)

    def abs(self) -> IX_dataset:
        """Take absolute value of signal, leaving error unchanged."""
        wout = self.copy()
        wout._signal = np.abs(wout._signal)
        return wout

    def __abs__(self) -> IX_dataset:
        return self.abs()

    @staticmethod
    def check_and_build_axis(val: Any) -> IX_axis:
        """Validate or construct an ``IX_axis`` from ``val``."""
        if isinstance(val, IX_axis):
            return val.copy()
        if isinstance(val, str) or _is_cellstr(val):
            return IX_axis(val)
        if isinstance(val, dict):
            return IX_axis(val)
        if _is_numeric_array(val) and not isinstance(val, (bool, np.bool_)):
            return IX_axis(str(val))
        raise ValueError(
            "HERBERT:IX_dataset:invalid_argument: Axis annotation must be character array or IX_axis object (type help IX_axis)"
        )

    @staticmethod
    def check_xyz(val: Any) -> np.ndarray:
        """Validate an axis coordinate / bin boundary vector."""
        if _is_empty(val):
            return np.zeros((0,), dtype=float)
        if not _is_numeric_array(val):
            raise ValueError(
                "HERBERT:IX_dataset:invalid_argument: axis values array must be a numeric vector"
            )
        arr = np.asarray(val, dtype=float)
        if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
            raise ValueError(
                "HERBERT:IX_dataset:invalid_argument: axis values array must be a numeric vector"
            )
        arr = arr.ravel()
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                "HERBERT:IX_dataset:invalid_argument: axis values must all be finite (i.e. no Inf or NaN)"
            )
        if arr.size > 1 and np.any(np.diff(arr) < 0):
            raise ValueError(
                "HERBERT:IX_dataset:invalid_argument: axis values or Histogram bin boundaries along x-axis must be strictly monotonically increasing"
            )
        return arr

    def _get_xyz_data(self, nax: int) -> np.ndarray:
        return self._xyz[nax - 1]

    def _set_xyz_data(self, nax: int, val: Any) -> None:
        self._xyz[nax - 1] = self.check_xyz(val)

    def _check_and_set_title(self, val: Any) -> None:
        if _is_empty(val):
            self._title = []
        elif isinstance(val, str):
            lines = val.splitlines() if "\n" in val else [val]
            self._title = [line.rstrip() for line in lines]
        elif _is_cellstr(val):
            if isinstance(val, np.ndarray):
                self._title = [str(x) for x in val.flat]
            else:
                self._title = [str(x) for x in val]
        elif _is_numeric_array(val) and not isinstance(val, (bool, np.bool_)):
            self._title = str(val)
        else:
            raise ValueError(
                "HERBERT:IX_dataset_1d:invalid_argument: Title must be character array or cell array of strings or numeric value"
            )

    def set_simple_xsigerr(
        self,
        iax: int | list[int] | np.ndarray,
        x: Any,
        signal: np.ndarray,
        err: np.ndarray,
        xdistr: Any = None,
    ) -> IX_dataset:
        """Set signal, error and selected axes without full combo check."""
        wout = self.copy()
        wout._signal = np.asarray(signal, dtype=float)
        wout._error = np.asarray(err, dtype=float)
        if wout.ndim() == 1 and wout._signal.ndim == 1:
            if wout._signal.size == 0:
                wout._signal = np.zeros((1, 0), dtype=float)
                wout._error = np.zeros((1, 0), dtype=float)
            else:
                wout._signal = wout._signal.reshape(-1, 1)
                wout._error = wout._error.reshape(-1, 1)

        iax_arr = np.atleast_1d(np.asarray(iax, dtype=int)).ravel()
        if np.any(iax_arr > self.ndim()) or np.any(iax_arr < 1):
            raise ValueError(
                "IX_dataset:invalid_argument: Axis index(es) exceeds maximal object dimension(s)"
            )

        if iax_arr.size == 1 and not isinstance(x, (list, tuple)):
            x_list = [x]
        else:
            x_list = list(x)

        if xdistr is not None and not _is_empty(xdistr):
            xd_arr = np.asarray(xdistr, dtype=bool).ravel()
            if xd_arr.size == 1:
                xd_arr = np.repeat(xd_arr, iax_arr.size)
        else:
            xd_arr = None

        for idx_i, ax_num in enumerate(iax_arr):
            wout._xyz[ax_num - 1] = np.asarray(x_list[idx_i], dtype=float).ravel()
            if xd_arr is not None:
                wout._xyz_distribution[ax_num - 1] = bool(xd_arr[idx_i])
        return wout

    def init_from_structure(self, in_struct: dict[str, Any]) -> IX_dataset:
        """Initialise dataset fields from a structure/dictionary."""
        prev = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            for fld, val in in_struct.items():
                if not fld.startswith("serializable"):
                    setattr(self, fld, val)
        finally:
            self._do_check_combo_arg = prev
        self.check_combo_arg()
        return self

    # ------------------------------------------------------------------
    # sigvar Interface
    # ------------------------------------------------------------------

    def sigvar(self) -> SigVar:
        """Create a ``sigvar`` object with signal and variance ``error**2``."""
        return sigvar(self._signal, self._error**2)

    def sigvar_get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(signal, variance, mask_not_nan)``."""
        s = self._signal
        var = self._error**2
        msk = ~np.isnan(s)
        return s, var, msk

    def sigvar_getx(self) -> Any:
        """Return bin centers or point coordinates along x."""
        x = self._xyz[0]
        if self._signal.size != x.size:
            return (0.5 * (x[1:] + x[:-1])).reshape(-1, 1)
        return x.reshape(-1, 1)

    def sigvar_set(self, sigvarobj: SigVar) -> IX_dataset:
        """Set signal and error from a ``sigvar`` object."""
        s_new = np.asarray(sigvarobj.s, dtype=float)
        v_new = np.asarray(sigvarobj.e, dtype=float)
        if self._signal.shape != s_new.shape:
            if self._signal.size == s_new.size and self.ndim() == 1:
                s_new = s_new.reshape(self._signal.shape)
                v_new = v_new.reshape(self._signal.shape)
            else:
                raise ValueError(
                    f"IX_dataset:invalid_argument: {type(self).__name__} and sigvar object have inconsistent sizes"
                )
        prev = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            self.signal = s_new
            self.error = np.sqrt(v_new)
        finally:
            self._do_check_combo_arg = prev
        self.check_combo_arg()
        return self

    def sigvar_size(self) -> tuple[int, ...]:
        """Return the MATLAB-style size of the signal array."""
        if self._signal.size == 0:
            return (1, 0) if self.ndim() <= 2 else (0, 0, 0)
        if self._signal.ndim == 1:
            return (self._signal.size, 1)
        return self._signal.shape

    # ------------------------------------------------------------------
    # Rebinning, Integrating, Cutting, Squeezing
    # ------------------------------------------------------------------

    def _rebin_xyz(
        self, array_is_descriptor: bool, dir_axes: int | list[int] | np.ndarray, *args: Any
    ) -> Any:
        if len(args) == 0 and dir_axes is None:
            return self.copy()
        iax = np.atleast_1d(np.asarray(dir_axes, dtype=int)).ravel()
        ndims = self.ndim()
        if np.any(iax > ndims) or np.any(iax < 1):
            bad = iax[iax > ndims]
            raise ValueError(
                f"IX_dataset:invalid_argument: Attempting to rebin  {ndims}D object along {bad} direction(s)"
            )
        opt = {
            "empty_is_full_range": False,
            "range_is_one_bin": False,
            "array_is_descriptor": array_is_descriptor,
            "bin_boundaries": True,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, False, False, iax, opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset:invalid_argument: {mess}")
        return wout

    def _integrate_xyz(
        self, array_is_descriptor: bool, dir_axes: int | list[int] | np.ndarray, *args: Any
    ) -> Any:
        iax = np.atleast_1d(np.asarray(dir_axes, dtype=int)).ravel()
        ndims = self.ndim()
        if np.max(iax) > ndims or np.any(iax < 1):
            raise ValueError(
                f"IX_dataset:invalid_argument: Attempting to inegrate {ndims}D object along {np.max(iax)} direction"
            )
        opt = {
            "empty_is_full_range": True,
            "range_is_one_bin": True,
            "array_is_descriptor": array_is_descriptor,
            "bin_boundaries": True,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, True, True, iax, opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset:invalid_argument: {mess}")
        wout = wout.squeeze_IX_dataset(iax)
        if isinstance(wout, IX_dataset):
            wout.check_combo_arg()
        return wout

    def _cut_xyz(
        self, dir_axes: int | list[int] | np.ndarray, *args: Any
    ) -> Any:
        iax = np.atleast_1d(np.asarray(dir_axes, dtype=int)).ravel()
        ndims = self.ndim()
        if np.max(iax) > ndims or np.any(iax < 1):
            raise ValueError(
                f"IX_dataset:invalid_argument: Attempting to cut from {ndims}D object along {np.max(iax)} direction"
            )
        opt = {
            "empty_is_full_range": False,
            "range_is_one_bin": True,
            "array_is_descriptor": True,
            "bin_boundaries": False,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, False, False, iax, opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset:invalid_argument: {mess}")
        wout = wout.squeeze_IX_dataset(iax)
        if isinstance(wout, IX_dataset):
            wout.check_combo_arg()
        return wout

    def squeeze_IX_dataset(self, iax: int | list[int] | np.ndarray | None = None) -> Any:
        """Squeeze singleton dimensions in ``IX_dataset_nd`` object."""
        ndim = self.ndim()
        if iax is None:
            iax_arr = np.arange(1, ndim + 1, dtype=int)
        else:
            iax_arr = np.atleast_1d(np.asarray(iax, dtype=int)).ravel()
            if np.any(iax_arr < 1) or np.any(iax_arr > ndim):
                raise ValueError(
                    f"IX_dataset:invalid_argument: List of axes to squeeze can contain only integers in range 1-{ndim}"
                )

        remove = np.zeros(ndim, dtype=bool)
        remove[iax_arr - 1] = True

        sig_shape = list(self._signal.shape[:ndim])
        while len(sig_shape) < ndim:
            sig_shape.append(1)
        sz = np.array(sig_shape, dtype=int)
        keep = (sz != 1) | (~remove)
        ind = np.flatnonzero(keep) + 1

        if np.all(keep):
            return self.copy()
        if not np.any(keep):
            return {
                "val": float(self._signal.flat[0]) if self._signal.size == 1 else self._signal.copy(),
                "err": float(self._error.flat[0]) if self._error.size == 1 else self._error.copy(),
            }

        ndim_squeeze = int(np.sum(keep))
        sz_squeeze = tuple(sz[keep]) if ndim_squeeze > 1 else (int(sz[keep][0]), 1)
        ax_kept, _ = self.axis(ind)
        if isinstance(ax_kept, dict):
            ax_kept = [ax_kept]
        return IX_dataset_nd(
            self.title,
            self._signal.reshape(sz_squeeze),
            self._error.reshape(sz_squeeze),
            self.s_axis,
            ax_kept,
        )

    @staticmethod
    def rebin_IX_dataset(
        win: Any,
        integrate_data: bool,
        point_integration_default: bool,
        iax: int | list[int] | np.ndarray,
        descriptor_opt: dict[str, bool],
        *args: Any,
    ) -> tuple[Any, bool, str]:
        """Rebin or integrate ``win`` along axes ``iax``."""
        iax_arr = np.atleast_1d(np.asarray(iax, dtype=int)).ravel()
        nax = iax_arr.size

        rem_args = list(args)
        if not (len(rem_args) == 1 and isinstance(rem_args[0], IX_dataset)) and (
            len(rem_args) >= 1
            and not _is_numeric_array(rem_args[-1])
            and not _is_empty(rem_args[-1])
        ):
            point_integration, ok, mess = _rebin_point_integration_check(
                nax, rem_args[-1]
            )
            if not ok:
                return None, False, mess
            rem_args = rem_args[:-1]
        else:
            point_integration = np.full(nax, bool(point_integration_default), dtype=bool)

        if len(rem_args) == 1 and isinstance(rem_args[0], type(win)):
            wref = rem_args[0]
            ishist_ref = np.atleast_1d(wref.ishistogram()).ravel()
            xbounds: list[np.ndarray] = []
            for i in range(nax):
                ax_idx = iax_arr[i]
                ref_x = wref._xyz[ax_idx - 1]
                if ref_x.size <= 1:
                    raise ValueError(
                        "IX_dataset:invalid_argument: Reference dataset must have at least one bin (histogram data) or two points (point data)"
                    )
                if ishist_ref[ax_idx - 1]:
                    xbounds.append(ref_x.copy())
                else:
                    xb_i, ok, mess = bin_boundaries_simple(ref_x)
                    if not ok:
                        raise ValueError(
                            f"IX_dataset:invalid_argument: Unable to construct bin boundaries for point data axis number {ax_idx} : {mess}"
                        )
                    xbounds.append(xb_i)
            true_values = np.ones(nax, dtype=bool)
            is_descriptor = np.zeros(nax, dtype=bool)
        else:
            ok, xbounds, any_lim_inf, is_descriptor, any_dx_zero, mess = (
                _rebin_boundaries_description_parse(nax, descriptor_opt, *rem_args)
            )
            if not ok:
                return None, False, mess
            true_values = ~(any_lim_inf | any_dx_zero)
            for i in range(nax):
                if true_values[i] and is_descriptor[i]:
                    xb_i, ok, mess = _bin_boundaries_from_descriptor(xbounds[i])
                    if not ok:
                        return None, False, mess
                    xbounds[i] = xb_i

        return _rebin_ix_dataset_single(
            win,
            integrate_data,
            point_integration,
            iax_arr,
            xbounds,
            true_values,
            is_descriptor,
        )

    # ------------------------------------------------------------------
    # data_op_interface & Serializable Interface
    # ------------------------------------------------------------------

    def get_NUM_DIMS(self) -> int:
        """Return number of object dimensions."""
        return self.ndim()

    def get_img_size(self) -> tuple[int, ...]:
        """Return size of the ND image object represents."""
        return self.sigvar_size()

    def class_version(self) -> int:
        return 2

    def classVersion(self) -> int:
        return self.class_version()

    def saveable_fields(self) -> list[str]:
        return list(self._fields_to_save)

    def saveableFields(self) -> list[str]:
        return self.saveable_fields()

    def convert_old_struct(
        self, S: dict[str, Any], ver: int | None = None
    ) -> tuple[dict[str, Any], IX_dataset]:
        S_updated: dict[str, Any] = {}
        for k, v in S.items():
            key = k[:-1] if k.endswith("_") else k
            S_updated[key] = v
        if "xyz" in S_updated:
            xyz = S_updated.pop("xyz")
            xyz_dist = S_updated.pop("xyz_distribution", [True] * len(xyz))
            xyz_ax = S_updated.pop("xyz_axis", [IX_axis()] * len(xyz))
            ax_names = ["x", "y", "z"]
            for i in range(min(len(xyz), 3)):
                nm = ax_names[i]
                S_updated[nm] = xyz[i]
                S_updated[f"{nm}_distribution"] = xyz_dist[i]
                S_updated[f"{nm}_axis"] = xyz_ax[i]
        return S_updated, self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IX_dataset) or self.ndim() != other.ndim():
            return NotImplemented
        if self._title != other._title or self._s_axis != other._s_axis:
            return False
        if self._signal.shape != other._signal.shape or self._error.shape != other._error.shape:
            return False
        if not np.allclose(self._signal, other._signal, rtol=1e-12, atol=1e-12, equal_nan=True):
            return False
        if not np.allclose(self._error, other._error, rtol=1e-12, atol=1e-12, equal_nan=True):
            return False
        if not np.array_equal(self._xyz_distribution, other._xyz_distribution):
            return False
        for a1, a2, x1, x2 in zip(
            self._xyz_axis, other._xyz_axis, self._xyz, other._xyz
        ):
            if a1 != a2:
                return False
            if x1.shape != x2.shape or not np.allclose(x1, x2, rtol=1e-12, atol=1e-12, equal_nan=True):
                return False
        return True


def _rebin_ix_dataset_single(
    win: IX_dataset,
    integrate_data: bool,
    point_integration: np.ndarray,
    iax: np.ndarray,
    xbounds: list[np.ndarray],
    true_values: np.ndarray,
    is_descriptor: np.ndarray,
) -> tuple[Any, bool, str]:
    """Perform single-dataset rebinning/integration along ``iax``."""
    wout_x: list[np.ndarray] = []
    wout_s = win._signal.copy()
    wout_e = win._error.copy()
    ndim = win.ndim()

    for i in range(iax.size):
        ax_num = int(iax[i])
        wx, ws, we, ok, mess = _rebin_one_axis(
            win,
            ax_num,
            ndim,
            win._xyz[ax_num - 1],
            win._xyz_distribution[ax_num - 1],
            wout_s,
            wout_e,
            integrate_data,
            bool(point_integration[i]),
            xbounds[i],
            bool(true_values[i]),
            bool(is_descriptor[i]),
        )
        if not ok:
            return None, False, mess
        wout_x.append(wx)
        wout_s = ws
        wout_e = we

    if not integrate_data:
        wout = win.set_simple_xsigerr(iax, wout_x, wout_s, wout_e)
    else:
        wout = win.set_simple_xsigerr(iax, wout_x, wout_s, wout_e, False)
    return wout, True, ""


def _rebin_one_axis(
    win: IX_dataset,
    iax: int,
    ndim: int,
    win_x: np.ndarray,
    win_xdist: bool,
    win_s: np.ndarray,
    win_e: np.ndarray,
    integrate_data: bool,
    point_integration: bool,
    xbounds: np.ndarray,
    true_values: bool,
    is_descriptor: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    """Rebin or integrate a single axis ``iax`` (1-based)."""
    ax0 = iax - 1
    sig_shape = list(win_s.shape[:ndim])
    while len(sig_shape) < ndim:
        sig_shape.append(1)
    sz_full = list(sig_shape)

    # Branch 1: Histogram data
    if win_x.size != sz_full[ax0]:
        if sz_full[ax0] == 0:
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
                False,
                f"No data points along axis {iax}: cannot perform rebin or integration of histogram data along this axis",
            )
        if true_values:
            wout_x = xbounds.copy()
        else:
            xb, ok, mess = _rebin_boundaries_description_resolve_inf(
                xbounds, is_descriptor, float(win_x[0]), float(win_x[-1])
            )
            if not ok:
                return np.empty(0), np.empty(0), np.empty(0), False, mess
            if is_descriptor:
                if xb.size > 0:
                    wout_x, ok, mess = _bin_boundaries_from_descriptor(xb, win_x)
                    if not ok:
                        return np.empty(0), np.empty(0), np.empty(0), False, mess
                else:
                    wout_x = win_x.copy()
            else:
                wout_x = xb

        if win_s.size == 0 or wout_x.size == 1:
            sz_out = list(sz_full)
            sz_out[ax0] = max(0, wout_x.size - 1)
            return (
                wout_x,
                np.zeros(sz_out, dtype=float),
                np.zeros(sz_out, dtype=float),
                True,
                "",
            )

        bounds_unchanged = wout_x.size == win_x.size and np.all(wout_x == win_x)
        bcast = [1] * max(win_s.ndim, ndim)
        bcast[ax0] = -1

        if win_xdist:
            if bounds_unchanged:
                wout_s = win_s.copy()
                wout_e = win_e.copy()
            else:
                wout_s, wout_e = win.rebin_hist(iax, win_x, win_s, win_e, wout_x)
            if integrate_data:
                dx_out = np.diff(wout_x).reshape(bcast[: wout_s.ndim])
                wout_s = wout_s * dx_out
                wout_e = wout_e * dx_out
        else:
            if bounds_unchanged:
                wout_s = win_s.copy()
                wout_e = win_e.copy()
            else:
                dx_in = np.diff(win_x).reshape(bcast[: win_s.ndim])
                wout_s, wout_e = win.rebin_hist(
                    iax, win_x, win_s / dx_in, win_e / dx_in, wout_x
                )
                dx_out = np.diff(wout_x).reshape(bcast[: wout_s.ndim])
                wout_s = wout_s * dx_out
                wout_e = wout_e * dx_out
        return wout_x, wout_s, wout_e, True, ""

    # Branch 2: Point data
    if true_values:
        xbounds_true = xbounds.copy()
    elif win_x.size > 0:
        xb, ok, mess = _rebin_boundaries_description_resolve_inf(
            xbounds, is_descriptor, float(win_x[0]), float(win_x[-1])
        )
        if not ok:
            return np.empty(0), np.empty(0), np.empty(0), False, mess
        if is_descriptor:
            win_xb, ok, mess = bin_boundaries_simple(win_x)
            if not ok:
                return np.empty(0), np.empty(0), np.empty(0), False, mess
            if xb.size > 0:
                xbounds_true, ok, mess = _bin_boundaries_from_descriptor(xb, win_xb)
                if not ok:
                    return np.empty(0), np.empty(0), np.empty(0), False, mess
            else:
                xbounds_true = win_xb
        else:
            xbounds_true = xb
            if xbounds_true.size == 2 and xbounds_true[1] == xbounds_true[0] and win_x.size == 1:
                xbounds_true, _, _ = bin_boundaries_simple(xbounds_true[0])
        if win_x.size > 1 and np.any(np.diff(win_x) <= 0):
            return (
                np.empty(0),
                np.empty(0),
                np.empty(0),
                False,
                f"One or more output bins has zero width along axis {iax}: cannot perform rebin or integration of point data along this axis",
            )
    else:
        xbounds_true = xbounds.copy()

    # Sub-branch 2A: Point averaging
    if not point_integration:
        if win_x.size == 0:
            return win_x.copy(), win_s.copy(), win_e.copy(), True, ""
        if xbounds_true.size <= 1:
            sz_out = list(sz_full)
            sz_out[ax0] = 0
            return (
                np.zeros((0,), dtype=float),
                np.zeros(sz_out, dtype=float),
                np.zeros(sz_out, dtype=float),
                True,
                "",
            )
        nb = xbounds_true.size
        ind = bin_index(win_x, xbounds_true, inclusive=True)
        ok_pts = (ind > 0) & (ind < nb)
        ind_ok = ind[ok_pts] - 1
        if ind_ok.size == 0:
            sz_out = list(sz_full)
            sz_out[ax0] = 0
            return (
                np.zeros((0,), dtype=float),
                np.zeros(sz_out, dtype=float),
                np.zeros(sz_out, dtype=float),
                True,
                "",
            )

        nout = np.bincount(ind_ok, minlength=nb - 1)
        xsum = np.bincount(ind_ok, weights=win_x[ok_pts], minlength=nb - 1)
        keep = nout != 0
        nout_keep = nout[keep].astype(float)
        wout_x = xsum[keep] / nout_keep

        s_work = np.moveaxis(win_s, ax0, 0)[ok_pts]
        e_work = np.moveaxis(win_e, ax0, 0)[ok_pts]
        trail_shape = s_work.shape[1:]
        s_2d = s_work.reshape(ind_ok.size, -1)
        e_2d = e_work.reshape(ind_ok.size, -1)

        n_keep = int(np.sum(keep))
        ws_2d = np.zeros((n_keep, s_2d.shape[1]), dtype=float)
        we_2d = np.zeros((n_keep, e_2d.shape[1]), dtype=float)
        for col in range(s_2d.shape[1]):
            ysum_c = np.bincount(ind_ok, weights=s_2d[:, col], minlength=nb - 1)[keep]
            esum_c = np.bincount(ind_ok, weights=e_2d[:, col] ** 2, minlength=nb - 1)[keep]
            ws_2d[:, col] = ysum_c / nout_keep
            we_2d[:, col] = np.sqrt(esum_c) / nout_keep

        if integrate_data:
            dx_keep = np.diff(xbounds_true)[keep].reshape(-1, 1)
            ws_2d *= dx_keep
            we_2d *= dx_keep

        wout_s = np.moveaxis(ws_2d.reshape((n_keep,) + trail_shape), 0, ax0)
        wout_e = np.moveaxis(we_2d.reshape((n_keep,) + trail_shape), 0, ax0)
        return wout_x, wout_s, wout_e, True, ""

    # Sub-branch 2B: Trapezoidal integration
    if win_x.size == 0:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
            False,
            f"No data points along axis {iax}: cannot perform rebin or integration of point data along this axis",
        )
    if win_x.size == 1:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
            False,
            f"Only one data point along axis {iax}: cannot perform rebin or integration of point data along this axis",
        )
    if xbounds_true.size <= 1:
        sz_out = list(sz_full)
        sz_out[ax0] = 0
        return (
            np.zeros((0,), dtype=float),
            np.zeros(sz_out, dtype=float),
            np.zeros(sz_out, dtype=float),
            True,
            "",
        )

    wout_x = 0.5 * (xbounds_true[1:] + xbounds_true[:-1])
    wout_s, wout_e = win.integrate_points(iax, win_x, win_s, win_e, xbounds_true)
    if not integrate_data:
        bcast = [1] * wout_s.ndim
        bcast[ax0] = -1
        dx_out = np.diff(xbounds_true).reshape(bcast)
        wout_s = wout_s / dx_out
        wout_e = wout_e / dx_out
    return wout_x, wout_s, wout_e, True, ""


# ======================================================================
# Gateway Functions & Standalone Classes
# ======================================================================


def IX_dataset_nd(*args: Any) -> IX_dataset:
    """Gateway constructor for ``IX_dataset_1d``, ``IX_dataset_2d``, or ``IX_dataset_3d``."""
    from pyhorace.data.ix_dataset_1d import IX_dataset_1d
    from pyhorace.data.ix_dataset_2d import IX_dataset_2d
    from pyhorace.data.ix_dataset_3d import IX_dataset_3d

    if len(args) == 1 and isinstance(args[0], (int, np.integer)):
        ndim = int(args[0])
        if ndim == 1:
            return IX_dataset_1d()
        if ndim == 2:
            return IX_dataset_2d()
        if ndim == 3:
            return IX_dataset_3d()
        raise ValueError(f"IX_dataset_nd with dimensionality n={ndim} is not supported")

    if len(args) == 5:
        title, signal, err, s_axis, ax = args
        if isinstance(ax, dict):
            ax = [ax]
        if not isinstance(ax, (list, tuple)) or not all(
            isinstance(a, dict) and set(a.keys()) == {"values", "axis", "distribution"}
            for a in ax
        ):
            raise ValueError("axis description does not have correct fields")
        ndim = len(ax)
        if ndim == 1:
            return IX_dataset_1d(
                title,
                signal,
                err,
                s_axis,
                ax[0]["values"],
                ax[0]["axis"],
                ax[0]["distribution"],
            )
        if ndim == 2:
            return IX_dataset_2d(
                title,
                signal,
                err,
                s_axis,
                ax[0]["values"],
                ax[0]["axis"],
                ax[0]["distribution"],
                ax[1]["values"],
                ax[1]["axis"],
                ax[1]["distribution"],
            )
        if ndim == 3:
            return IX_dataset_3d(
                title,
                signal,
                err,
                s_axis,
                ax[0]["values"],
                ax[0]["axis"],
                ax[0]["distribution"],
                ax[1]["values"],
                ax[1]["axis"],
                ax[1]["distribution"],
                ax[2]["values"],
                ax[2]["axis"],
                ax[2]["distribution"],
            )
        raise ValueError(f"IX_dataset_nd with dimensionality n={ndim} is not supported")

    raise ValueError("Check number and type of arguments to IX_dataset_nd")


def isa_IX_dataset_nd(w: Any) -> bool:
    """Return True if *w* is a valid ``IX_dataset`` object."""
    return isinstance(w, IX_dataset)


def rebin_IX_dataset_nd(
    win: Any,
    integrate_data: bool,
    point_integration_default: bool,
    iax: Any,
    descriptor_opt: dict[str, bool],
    *args: Any,
) -> tuple[Any, bool, str]:
    """Rebin an ``IX_dataset_nd`` object or structure."""
    if not isinstance(win, IX_dataset):
        raise ValueError(
            "IX_dataset_nd:invalid_argument: Dataset to be rebinned has unrecognised type"
        )
    wout, ok, message = IX_dataset.rebin_IX_dataset(
        win, integrate_data, point_integration_default, iax, descriptor_opt, *args
    )
    if not ok:
        raise RuntimeError(f"IX_dataset_nd:runtime_error: {message}")
    return wout, ok, message


def delta_IX_dataset_nd(
    w1: Any, w2: Any, tol: float = 0.0
) -> tuple[bool, str, np.ndarray]:
    """Report the difference between two ``IX_dataset_nd`` objects against ``tol``."""
    def _del_calc(v1: np.ndarray, v2: np.ndarray) -> tuple[float, float]:
        a = np.asarray(v1, dtype=float).ravel()
        b = np.asarray(v2, dtype=float).ravel()
        if a.size == 0 and b.size == 0:
            return 0.0, 0.0
        num = np.abs(a - b)
        den = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0)
        return float(np.nanmax(num)), float(np.nanmax(num / den))

    if isinstance(w1, IX_dataset) and isinstance(w2, IX_dataset):
        h1 = np.atleast_1d(w1.ishistogram()).ravel()
        h2 = np.atleast_1d(w2.ishistogram()).ravel()
        nd1, nd2 = h1.size, h2.size
        if nd1 != nd2:
            return False, "Datasets have different dimensionality", np.empty(0)
        if not np.all(h1 == h2):
            return (
                False,
                "One or more corresponding axes are not both histogram or point data",
                np.empty(0),
            )
        del_abs = np.zeros(nd1 + 2, dtype=float)
        del_rel = np.zeros(nd1 + 2, dtype=float)
        for i in range(1, nd1 + 1):
            x1, _ = w1.axis(i)
            x2, _ = w2.axis(i)
            if x1["distribution"] != x2["distribution"]:
                return (
                    False,
                    f"Axis {i}: one axis is a distribution, the other not",
                    np.empty(0),
                )
            if np.asarray(x1["values"]).size != np.asarray(x2["values"]).size:
                return (
                    False,
                    f"Axis {i}: different number of data points along this axis",
                    np.empty(0),
                )
            if x1["axis"] != x2["axis"]:
                return False, f"Axis {i}: IX_axis descriptions differ", np.empty(0)
            del_abs[i - 1], del_rel[i - 1] = _del_calc(x1["values"], x2["values"])
        del_abs[nd1], del_rel[nd1] = _del_calc(w1.signal, w2.signal)
        del_abs[nd1 + 1], del_rel[nd1 + 1] = _del_calc(w1.error, w2.error)
    elif isinstance(w1, dict) and isinstance(w2, dict) and set(w1.keys()) == {"val", "err"} and set(w2.keys()) == {"val", "err"}:
        del_abs = np.zeros(2, dtype=float)
        del_rel = np.zeros(2, dtype=float)
        del_abs[0], del_rel[0] = _del_calc(w1["val"], w2["val"])
        del_abs[1], del_rel[1] = _del_calc(w1["err"], w2["err"])
    else:
        return False, "Unrecognised objects for comparison in this function", np.empty(0)

    if tol < 0:
        del_out = del_rel
        del_max = float(np.max(del_out))
        if del_max > abs(tol):
            return (
                False,
                f"Relative tolerance criterion failed: maximum relative error = {del_max}",
                del_out,
            )
        return True, "", del_out
    else:
        del_out = del_abs
        del_max = float(np.max(del_out))
        if del_max > abs(tol):
            return (
                False,
                f"Absolute tolerance criterion failed: maximum error = {del_max}",
                del_out,
            )
        return True, "", del_out


class compact_array(Serializable):
    """Store array with non-unique elements compactly (translated from ``compact_array.m``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._unique_val: list[Any] = []
        self._nonunq_idx: list[np.ndarray] = []
        self._unique_value_used: np.ndarray = np.empty(0, dtype=bool)
        self._lidx_cache: np.ndarray | None = None
        if len(args) > 0 or kwargs:
            self.set_positional_and_key_val_arguments(
                self.saveableFields(), False, *args, **kwargs
            )

    @property
    def n_unique(self) -> int:
        return len(self._nonunq_idx)

    @property
    def n_objects(self) -> int:
        lc, _ = self.get_lidx()
        return lc.size

    @property
    def unique_val(self) -> list[Any]:
        return self._unique_val

    @unique_val.setter
    def unique_val(self, val: Any) -> None:
        if not isinstance(val, (list, tuple)):
            raise ValueError(
                "HERBERT:compact_array:invalid_argument: Compact array values have to be wrapped in cellarray"
            )
        self._unique_val = list(val)
        self._unique_value_used = np.zeros(len(self._unique_val), dtype=bool)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def nonunq_idx(self) -> list[np.ndarray]:
        return self._nonunq_idx

    @nonunq_idx.setter
    def nonunq_idx(self, idx: Any) -> None:
        if not isinstance(idx, (list, tuple)):
            raise ValueError(
                "HERBERT:compact_array:invalid_argument: Compact array Indices have to be wrapped in cellarray"
            )
        self._nonunq_idx = [np.asarray(x, dtype=int).ravel() for x in idx]
        self._lidx_cache = None
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def get_lidx(self) -> tuple[np.ndarray, compact_array]:
        if self._lidx_cache is None:
            if not self._nonunq_idx:
                self._lidx_cache = np.empty(0, dtype=float)
            else:
                pos = np.concatenate(self._nonunq_idx)
                n_pos = int(np.max(pos))
                lidx = np.full(n_pos, np.nan, dtype=float)
                for i, idxs in enumerate(self._nonunq_idx, start=1):
                    lidx[idxs - 1] = i
                self._lidx_cache = lidx
        return self._lidx_cache, self

    def get(self, lidx: int) -> tuple[Any, compact_array, int, bool]:
        lic, _ = self.get_lidx()
        used_unique_num = int(lic[lidx - 1])
        val = self._unique_val[used_unique_num - 1]
        was_used = bool(self._unique_value_used[used_unique_num - 1])
        self._unique_value_used[used_unique_num - 1] = True
        return val, self, used_unique_num, was_used

    def get_subobj(self, idx_to_select: Any) -> compact_array:
        sel = np.asarray(idx_to_select, dtype=int).ravel()
        lidx_out = []
        val_out = []
        for uval, idxs in zip(self._unique_val, self._nonunq_idx):
            mask = np.isin(idxs, sel)
            if np.any(mask):
                val_out.append(uval)
                lidx_out.append(idxs[mask])
        return compact_array(lidx_out, val_out)

    def class_version(self) -> int:
        return 1

    def classVersion(self) -> int:
        return self.class_version()

    def saveable_fields(self) -> list[str]:
        return ["nonunq_idx", "unique_val"]

    def saveableFields(self) -> list[str]:
        return self.saveable_fields()

    def check_combo_arg(self) -> compact_array:
        if len(self._unique_val) != len(self._nonunq_idx):
            raise ValueError(
                "HERBERT:compact_array:invalid_argument: Number of unique value fields have to be equal to number of its indices"
            )
        self.get_lidx()
        return self

    @classmethod
    def loadobj(cls, S: dict[str, Any]) -> Any:
        obj = cls()
        return Serializable.loadobj(S, obj)


# ======================================================================
# Standalone Dataset Dispatchers & DataOpArray Method Extensions
# ======================================================================


def ishistogram(w: Any, n: int | None = None) -> np.ndarray | bool:
    """Return histogram status for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.ishistogram(n)
    if isinstance(w, (DataOpArray, list, tuple, np.ndarray)):
        arr = DataOpArray(w)
        elems = list(arr.ravel(order="F"))
        if not elems:
            return np.empty(0, dtype=bool)
        nd = elems[0].ndim()
        if nd == 1:
            return np.array([bool(el.ishistogram(n)) for el in elems], dtype=bool)
        if n is not None:
            return np.array([[bool(el.ishistogram(n)) for el in elems]], dtype=bool)
        return np.column_stack(
            [np.asarray(el.ishistogram(), dtype=bool).ravel() for el in elems]
        )
    raise TypeError(f"Unsupported type for ishistogram: {type(w).__name__}")


def axis(w: Any, n: int | None = None) -> tuple[Any, Any]:
    """Return ``(ax_struct, hist_flag)`` for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.axis(n)
    if isinstance(w, (DataOpArray, list, tuple, np.ndarray)):
        arr = DataOpArray(w)
        elems = list(arr.ravel(order="F"))
        ax_list = []
        hist_list = []
        for el in elems:
            a_i, h_i = el.axis(n)
            ax_list.append(a_i)
            hist_list.append(h_i)
        if elems and elems[0].ndim() == 1:
            return ax_list, np.array(hist_list, dtype=bool)
        return ax_list, np.column_stack([np.atleast_1d(h).ravel() for h in hist_list])
    raise TypeError(f"Unsupported type for axis: {type(w).__name__}")


def cnt2dist(w: Any, *args: Any) -> Any:
    """Convert counts to distribution for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.cnt2dist(*args)
    arr = DataOpArray(w)
    out = np.empty(arr.shape, dtype=object)
    out_flat = out.ravel(order="F")
    for i, el in enumerate(arr.ravel(order="F")):
        out_flat[i] = el.cnt2dist(*args)
    return out.reshape(arr.shape, order="F").view(DataOpArray)


def dist2cnt(w: Any, *args: Any) -> Any:
    """Convert distribution to counts for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.dist2cnt(*args)
    arr = DataOpArray(w)
    out = np.empty(arr.shape, dtype=object)
    out_flat = out.ravel(order="F")
    for i, el in enumerate(arr.ravel(order="F")):
        out_flat[i] = el.dist2cnt(*args)
    return out.reshape(arr.shape, order="F").view(DataOpArray)


def point2hist(w: Any, *args: Any) -> Any:
    """Convert point axes to histogram axes for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.point2hist(*args)
    arr = DataOpArray(w)
    out = np.empty(arr.shape, dtype=object)
    out_flat = out.ravel(order="F")
    for i, el in enumerate(arr.ravel(order="F")):
        out_flat[i] = el.point2hist(*args)
    return out.reshape(arr.shape, order="F").view(DataOpArray)


def hist2point(w: Any, *args: Any) -> Any:
    """Convert histogram axes to point axes for dataset or array of datasets."""
    if isinstance(w, IX_dataset):
        return w.hist2point(*args)
    arr = DataOpArray(w)
    out = np.empty(arr.shape, dtype=object)
    out_flat = out.ravel(order="F")
    for i, el in enumerate(arr.ravel(order="F")):
        out_flat[i] = el.hist2point(*args)
    return out.reshape(arr.shape, order="F").view(DataOpArray)


def _install_dataoparray_dataset_methods() -> None:
    """Attach dataset and plotting methods onto ``DataOpArray``."""
    DataOpArray.ishistogram = lambda self, n=None: ishistogram(self, n)  # type: ignore[attr-defined]
    DataOpArray.axis = lambda self, n=None: axis(self, n)  # type: ignore[attr-defined]
    DataOpArray.cnt2dist = lambda self, *args: cnt2dist(self, *args)  # type: ignore[attr-defined]
    DataOpArray.dist2cnt = lambda self, *args: dist2cnt(self, *args)  # type: ignore[attr-defined]
    DataOpArray.point2hist = lambda self, *args: point2hist(self, *args)  # type: ignore[attr-defined]
    DataOpArray.hist2point = lambda self, *args: hist2point(self, *args)  # type: ignore[attr-defined]

    def _map_method(name: str):
        def _wrapper(self: DataOpArray, *args: Any, **kwargs: Any) -> Any:
            out = np.empty(self.shape, dtype=object)
            out_flat = out.ravel(order="F")
            for i, el in enumerate(self.ravel(order="F")):
                out_flat[i] = getattr(el, name)(*args, **kwargs)
            return out.reshape(self.shape, order="F").view(DataOpArray)
        return _wrapper

    for mname in (
        "rebin",
        "rebin2",
        "integrate",
        "integrate2",
        "cut",
        "deriv",
        "unspike",
        "func_eval",
        "noisify",
        "scale",
        "shift",
        "smooth",
        "rebunch",
        "interp",
        "linspace",
        "mask",
        "moments",
        "xye",
        "abs",
        "squeeze_IX_dataset",
        "rebin_x",
        "rebin_y",
        "rebin_z",
        "rebin_xy",
        "rebin_xyz",
        "integrate_x",
        "integrate_y",
        "integrate_z",
        "integrate_xy",
        "integrate_xyz",
        "cut_x",
        "cut_y",
        "cut_z",
        "cut_xy",
        "cut_xyz",
        "deriv_x",
        "deriv_y",
        "deriv_z",
        "expand_d1d",
        "expand_d2d",
        "contract_d2d",
        "contract_d3d",
    ):
        setattr(DataOpArray, mname, _map_method(mname))

    from pyhorace.data import data_plot_interface as _dpi

    for pname in (
        "dd",
        "de",
        "dh",
        "dl",
        "dm",
        "dp",
        "pd",
        "pdoc",
        "pe",
        "peoc",
        "ph",
        "phoc",
        "pl",
        "ploc",
        "pm",
        "pmoc",
        "pp",
        "ppoc",
        "da",
        "ds",
        "ds2",
        "pa",
        "paoc",
        "ps",
        "psoc",
        "ps2",
        "ps2oc",
        "sliceomatic",
        "sliceomatic_overview",
        "plot",
        "plotover",
    ):
        fn = getattr(_dpi, pname, None)
        if fn is not None:
            setattr(DataOpArray, pname, fn)


_install_dataoparray_dataset_methods()


