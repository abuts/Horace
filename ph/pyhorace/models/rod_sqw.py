"""Delta function rod in momentum, constant for all energies.

Translated from MATLAB ``horace_core/sqw_models/rod_sqw.m``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from pyhorace.coord_transf.bmatrix import bmatrix


def rod_sqw(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    pars: ArrayLike,
    lattice: ArrayLike,
    rlp: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Delta function in momentum, constant for all energies.

    Translated from MATLAB ``horace_core/sqw_models/rod_sqw.m``.

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : ArrayLike
        ``[Amplitude, fwhh]`` - the peak intensity and width (FWHH in inverse
        Angstroms) of an energy-independent mode.
    lattice : ArrayLike
        Crystal lattice parameters ``[a, b, c, alpha, beta, gamma]``.
    rlp : ArrayLike, optional
        Values of ``[qh, qk, ql]`` at which the rods of scattering occur:
        array of shape ``(np, 3)`` where ``np`` is the number of points.
        If not given then assumes a Gaussian at every integer ``[h, k, l]``,
        truncated to the nearest rlp.

    Returns
    -------
    np.ndarray
        Spectral weight with the same shape as ``qh``.
    """
    lat_arr = np.asarray(lattice, dtype=np.float64).ravel()
    pars_arr = np.asarray(pars, dtype=np.float64).ravel()

    amp = float(pars_arr[0])
    qsig = float(pars_arr[1]) / math.sqrt(math.log(256.0))

    b = bmatrix(lat_arr[:3], lat_arr[3:6], nargout=1)

    qrlu = np.vstack(
        [
            qh.ravel(order="F"),
            qk.ravel(order="F"),
            ql.ravel(order="F"),
        ]
    )

    if rlp is None:
        # MATLAB `round` rounds half away from zero
        qrlu_rounded = np.sign(qrlu) * np.floor(np.abs(qrlu) + 0.5)
        dqrlu = qrlu - qrlu_rounded
        dq = b @ dqrlu
        weight_flat = amp * np.exp(-np.sum(dq**2, axis=0) / (2.0 * qsig**2))
    else:
        rlp_arr = np.atleast_2d(np.asarray(rlp, dtype=np.float64))
        weight_flat = np.zeros(qh.size, dtype=np.float64)
        for i in range(rlp_arr.shape[0]):
            dqrlu = qrlu - rlp_arr[i, :3].reshape(3, 1)
            dq = b @ dqrlu
            weight_flat = weight_flat + amp * np.exp(
                -np.sum(dq**2, axis=0) / (2.0 * qsig**2)
            )

    return weight_flat.reshape(qh.shape, order="F")

