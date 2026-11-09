"""Delta function in momentum along one direction of momentum, constant for all energies.

Translated from MATLAB ``horace_core/sqw_models/sheet_sqw.m``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from pyhorace.coord_transf.bmatrix import bmatrix


def sheet_sqw(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    pars: Sequence[float] | np.ndarray,
    lattice: Sequence[float] | np.ndarray,
    normal: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Delta function in momentum along one direction of momentum, constant for all energies.

    Translated from MATLAB ``horace_core/sqw_models/sheet_sqw.m``.

    Parameters
    ----------
    qh, qk, ql, en : array_like
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : sequence of float or np.ndarray
        ``[Amplitude, fwhh]`` - the peak intensity and width (FWHH in inverse
        Angstroms).
    lattice : sequence of float or np.ndarray
        Crystal lattice parameters ``[a, b, c, alpha, beta, gamma]``.
    normal : sequence of float or np.ndarray
        Direction of normal to sheet in rlu ``[hnorm, knorm, lnorm]``.

    Returns
    -------
    np.ndarray
        Spectral weight with the same shape as ``qh``.
    """
    qh_arr = np.asarray(qh, dtype=np.float64)
    qk_arr = np.asarray(qk, dtype=np.float64)
    ql_arr = np.asarray(ql, dtype=np.float64)
    lat_arr = np.asarray(lattice, dtype=np.float64).ravel()
    norm_arr = np.asarray(normal, dtype=np.float64).reshape(3, 1)

    amp = float(pars[0])
    qsig = float(pars[1]) / math.sqrt(math.log(256.0))

    b = bmatrix(lat_arr[:3], lat_arr[3:6], nargout=1)

    # Get normal and Q in crystal Cartesian coordinates
    n = b @ norm_arr
    n = n / np.linalg.norm(n)
    qcryst = b @ np.vstack(
        [
            qh_arr.ravel(order="F"),
            qk_arr.ravel(order="F"),
            ql_arr.ravel(order="F"),
        ]
    )

    # Note: matches MATLAB `qperp_sqr = sum(bsxfun(@times, n, qcryst).^2, 1)`
    qperp_sqr = np.sum((n * qcryst) ** 2, axis=0)
    weight_flat = amp * np.exp(qperp_sqr / (-2.0 * qsig**2))
    return weight_flat.reshape(qh_arr.shape, order="F")

