"""Null S(Q, w) model returning zeros of the same size as the input arrays.

Translated from MATLAB ``horace_core/functions/demo_null_func.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def demo_null_func(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, en: np.ndarray, p: Any = None
) -> np.ndarray:
    """Return an array of zeros the same size as the input array ``qh``.

    Translated from MATLAB ``horace_core/functions/demo_null_func.m``.

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Coordinate arrays in ``(Q, E)``-space.
    p : Any, optional
        Model parameters (unused).

    Returns
    -------
    np.ndarray
        Array of zeros with the same shape as ``qh``.
    """
    return np.zeros_like(qh, dtype=np.float64)

