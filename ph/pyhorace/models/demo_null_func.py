"""Null S(Q, w) model returning zeros of the same size as the input arrays.

Translated from MATLAB ``horace_core/functions/demo_null_func.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def demo_null_func(
    qh: Any, qk: Any, ql: Any, en: Any, p: Any = None
) -> np.ndarray:
    """Return an array of zeros the same size as the input array ``qh``.

    Translated from MATLAB ``horace_core/functions/demo_null_func.m``.

    Parameters
    ----------
    qh, qk, ql, en : array_like
        Coordinate arrays in ``(Q, E)``-space.
    p : Any, optional
        Model parameters (unused).

    Returns
    -------
    np.ndarray
        Array of zeros with the same shape as ``qh``.
    """
    qh_arr = np.asarray(qh, dtype=np.float64)
    return np.zeros_like(qh_arr, dtype=np.float64)

