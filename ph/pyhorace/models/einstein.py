"""Einstein function helper for neutron scattering cross-sections.

Translated from MATLAB ``herbert_core/utilities/maths_functions/einstein.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def einstein(y: np.ndarray) -> np.ndarray:
    """Calculate the function ``y / (1 - exp(-y))``, carefully handling negative ``y`` and ``|y|`` near zero.

    Translated from MATLAB ``herbert_core/utilities/maths_functions/einstein.m``.

    This is one of several variants of functions known as Einstein functions. It
    crops up in numerically robust computation of neutron scattering cross-sections.

    Parameters
    ----------
    y : array_like or float
        Input value or array of values.

    Returns
    -------
    np.ndarray
        Evaluated ``y / (1 - exp(-y))`` with the same shape as ``y``.
    """
    y_arr = y.ravel()
    p = np.zeros_like(y_arr, dtype=np.float64)

    yabs = np.abs(y_arr)
    ibig = yabs > 0.1
    ibigneg = ibig & (y_arr < 0.0)

    if np.any(ibig):
        p[ibig] = yabs[ibig] / (1.0 - np.exp(-yabs[ibig]))
        if np.any(ibigneg):
            p[ibigneg] = p[ibigneg] * np.exp(-yabs[ibigneg])

    if not np.all(ibig):
        ys = y_arr[~ibig]
        ys2 = ys**2
        p[~ibig] = 1.0 + 0.5 * ys * (
            1.0
            + (1.0 / 6.0)
            * ys
            * (
                1.0
                - (1.0 / 60.0)
                * ys2
                * (1.0 - (1.0 / 42.0) * ys2 * (1.0 - (1.0 / 40.0) * ys2))
            )
        )

    return p.ravel()

