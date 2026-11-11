"""Calculate 4 Gaussians at symmetric positions in momentum space.

Translated from MATLAB ``horace_core/functions/demo_4gauss_2dQ.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def demo_4gauss_2dQ(
    q1: np.ndarray, q2: np.ndarray, p: ArrayLike
) -> np.ndarray:
    """Calculate 4 Gaussians at symmetric positions in momentum space.

    This function is designed only to work for constant-energy 2D slices.

    Syntax:
        ``wout = func_eval(win, demo_4gauss_2dQ, pars, options)``

    Parameters
    ----------
    q1 : np.ndarray
        Variable component ``q1`` (meshgrid style matrix or vector).
    q2 : np.ndarray
        Variable component ``q2``.
    p : ArrayLike
        Parameters ``[amp, posx, posy, wid, symposx, symposy, bg]``.

    Returns
    -------
    np.ndarray
        Array the same size as the input arrays ``q1`` and ``q2``.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    amp = float(p_arr[0])
    posx = float(p_arr[1])
    posy = float(p_arr[2])
    wid = float(p_arr[3])
    symposx = float(p_arr[4])
    symposy = float(p_arr[5])
    bg = float(p_arr[6])

    pos1 = (symposx + posx, symposy)
    pos2 = (symposx, symposy - posy)
    pos3 = (symposx - posx, symposy)
    pos4 = (symposx, symposy + posy)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = (
            amp
            * np.exp(-0.5 * ((q1 - pos1[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2 - pos1[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1 - pos2[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2 - pos2[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1 - pos3[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2 - pos3[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1 - pos4[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2 - pos4[1]) / wid) ** 2)
            + bg
        )

    return y

