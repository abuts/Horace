"""Delta function in momentum along one direction of momentum, constant for all energies.

Translated from MATLAB ``horace_core/sqw_models/sheet_sqw.m``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from pyhorace.coord_transf.bmatrix import bmatrix


def sheet_sqw(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    pars: ArrayLike,
    lattice: ArrayLike,
    normal: ArrayLike,
) -> np.ndarray:
    """Delta function in momentum along one direction of momentum, constant for all energies.

    Translated from MATLAB ``horace_core/sqw_models/sheet_sqw.m``.

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : ArrayLike
        ``[Amplitude, fwhh]`` - the peak intensity and width (FWHH in inverse
        Angstroms).
    lattice : ArrayLike
        Crystal lattice parameters ``[a, b, c, alpha, beta, gamma]``.
    normal : ArrayLike
        Direction of normal to sheet in rlu ``[hnorm, knorm, lnorm]``.

    Returns
    -------
    np.ndarray
        Spectral weight with the same shape as ``qh``.
    """
    lat_arr = np.asarray(lattice, dtype=np.float64).ravel()
    norm_arr = np.asarray(normal, dtype=np.float64).reshape(3, 1)
    pars_arr = np.asarray(pars, dtype=np.float64).ravel()

    amp = float(pars_arr[0])
    qsig = float(pars_arr[1]) / math.sqrt(math.log(256.0))

    b = bmatrix(lat_arr[:3], lat_arr[3:6], nargout=1)

    # Get normal and Q in crystal Cartesian coordinates
    n = b @ norm_arr
    n = n / np.linalg.norm(n)
    qcryst = b @ np.vstack(
        [
            qh.ravel(order="F"),
            qk.ravel(order="F"),
            ql.ravel(order="F"),
        ]
    )

    # Note: matches MATLAB `qperp_sqr = sum(bsxfun(@times, n, qcryst).^2, 1)`
    qperp_sqr = np.sum((n * qcryst) ** 2, axis=0)
    weight_flat = amp * np.exp(qperp_sqr / (-2.0 * qsig**2))
    return weight_flat.reshape(qh.shape, order="F")

