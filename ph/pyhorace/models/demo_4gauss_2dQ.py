"""Calculate 4 Gaussians at symmetric positions in momentum space.

Translated from MATLAB ``horace_core/functions/demo_4gauss_2dQ.m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def demo_4gauss_2dQ(
    q1: Any, q2: Any, p: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Calculate 4 Gaussians at symmetric positions in momentum space.

    This function is designed only to work for constant-energy 2D slices.

    Syntax:
        ``wout = func_eval(win, demo_4gauss_2dQ, pars, options)``

    Parameters
    ----------
    q1 : array_like
        Variable component ``q1`` (meshgrid style matrix or vector).
    q2 : array_like
        Variable component ``q2``.
    p : sequence of float or np.ndarray
        Parameters ``[amp, posx, posy, wid, symposx, symposy, bg]``.

    Returns
    -------
    np.ndarray
        Array the same size as the input arrays ``q1`` and ``q2``.
    """
    q1_arr = np.asarray(q1, dtype=np.float64)
    q2_arr = np.asarray(q2, dtype=np.float64)

    amp = float(p[0])
    posx = float(p[1])
    posy = float(p[2])
    wid = float(p[3])
    symposx = float(p[4])
    symposy = float(p[5])
    bg = float(p[6])

    pos1 = (symposx + posx, symposy)
    pos2 = (symposx, symposy - posy)
    pos3 = (symposx - posx, symposy)
    pos4 = (symposx, symposy + posy)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = (
            amp
            * np.exp(-0.5 * ((q1_arr - pos1[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2_arr - pos1[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1_arr - pos2[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2_arr - pos2[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1_arr - pos3[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2_arr - pos3[1]) / wid) ** 2)
            + amp
            * np.exp(-0.5 * ((q1_arr - pos4[0]) / wid) ** 2)
            * np.exp(-0.5 * ((q2_arr - pos4[1]) / wid) ** 2)
            + bg
        )

    return y

