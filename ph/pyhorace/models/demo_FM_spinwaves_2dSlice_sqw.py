"""Calculate spectral weight from a Heisenberg ferromagnet for a Q-E slice.

Translated from MATLAB ``horace_core/functions/demo_FM_spinwaves_2dSlice_sqw.m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def demo_FM_spinwaves_2dSlice_sqw(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    pars: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Calculate the spectral weight from a Heisenberg ferromagnet for a Q-E slice with nearest-neighbour interactions only.

    Same model as in ``sqw_broad`` in Tobyfit.

    Syntax:
        ``wout = sqw_eval(win, demo_FM_spinwaves_2dSlice_sqw, pars, options)``

    Parameters
    ----------
    qh, qk, ql, en : array_like or float
        Arrays that are all the same size (or are scalars) that specify the
        coordinates for points in ``(Q, E)``-space where you wish to calculate
        the intensity.
    pars : sequence of float or np.ndarray
        Model parameters ``[js, delta, gam, temp, amp]``.

    Returns
    -------
    np.ndarray
        Array the same size as the input arrays ``qh`` and ``en``.
    """
    qh_arr = np.asarray(qh, dtype=np.float64)
    qk_arr = np.asarray(qk, dtype=np.float64)
    ql_arr = np.asarray(ql, dtype=np.float64)
    en_arr = np.asarray(en, dtype=np.float64)

    js = float(pars[0])
    delta = float(pars[1])
    gam = float(pars[2])
    temp = float(pars[3])
    amp = float(pars[4])

    omega0 = delta + (4.0 * js) * (
        (np.sin(np.pi * qh_arr)) ** 2
        + (np.sin(np.pi * qk_arr)) ** 2
        + (np.sin(np.pi * ql_arr)) ** 2
    )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        bose = en_arr / (1.0 - np.exp(-11.602 * en_arr / temp))
        y = amp * (
            bose
            * (4.0 * gam * omega0)
            / (np.pi * ((en_arr**2 - omega0**2) ** 2 + 4.0 * (gam * en_arr) ** 2))
        )

    return y

