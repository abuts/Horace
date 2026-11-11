"""Bose occupation factor multiplied by energy transfer.

Translated from MATLAB ``herbert_core/utilities/maths_functions/bose_times_eps.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyhorace.models.einstein import einstein


def bose_times_eps(eps: np.ndarray, T: float) -> np.ndarray:
    """Compute ``(<n> + 1) * eps``, where ``<n>`` is the Bose occupation function and ``eps`` is energy.

    Translated from MATLAB ``herbert_core/utilities/maths_functions/bose_times_eps.m``.

    Parameters
    ----------
    eps : array_like or float
        Energy transfer, or array of energy transfers (meV).
    T : float
        Temperature (K).

    Returns
    -------
    np.ndarray
        ``(<n> + 1) * eps``, where ``<n>`` is the Bose occupation function.
    """
    eps_arr = eps.ravel()
    kB = 8.6173324e-2

    if T != 0:
        y = (kB * T) * einstein(eps_arr / (kB * T))
    else:
        if np.all(eps_arr > 0):
            y = eps_arr.copy()
        else:
            y = np.zeros_like(eps_arr, dtype=np.float64)
            mask = eps_arr > 0
            y[mask] = eps_arr[mask]
    return y.ravel()

