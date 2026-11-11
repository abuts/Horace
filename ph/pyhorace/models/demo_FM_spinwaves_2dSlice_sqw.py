"""Calculate spectral weight from a Heisenberg ferromagnet for a Q-E slice.

Translated from MATLAB ``horace_core/functions/demo_FM_spinwaves_2dSlice_sqw.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def demo_FM_spinwaves_2dSlice_sqw(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    pars: ArrayLike,
) -> np.ndarray:
    """Calculate the spectral weight from a Heisenberg ferromagnet for a Q-E slice with nearest-neighbour interactions only.

    Same model as in ``sqw_broad`` in Tobyfit.

    Syntax:
        ``wout = sqw_eval(win, demo_FM_spinwaves_2dSlice_sqw, pars, options)``

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Arrays that are all the same size that specify the coordinates for
        points in ``(Q, E)``-space where you wish to calculate the intensity.
    pars : ArrayLike
        Model parameters ``[js, delta, gam, temp, amp]``.

    Returns
    -------
    np.ndarray
        Array the same size as the input arrays ``qh`` and ``en``.
    """
    pars_arr = np.asarray(pars, dtype=np.float64).ravel()
    js = float(pars_arr[0])
    delta = float(pars_arr[1])
    gam = float(pars_arr[2])
    temp = float(pars_arr[3])
    amp = float(pars_arr[4])

    omega0 = delta + (4.0 * js) * (
        (np.sin(np.pi * qh)) ** 2
        + (np.sin(np.pi * qk)) ** 2
        + (np.sin(np.pi * ql)) ** 2
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bose = en / (1.0 - np.exp(-11.602 * en / temp))
        y = amp * (
            bose
            * (4.0 * gam * omega0)
            / (np.pi * ((en**2 - omega0**2) ** 2 + 4.0 * (gam * en) ** 2))
        )

    return y

