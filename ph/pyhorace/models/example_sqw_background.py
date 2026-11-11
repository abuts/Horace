"""Background S(Q, w) model for a cubic system.

Translated from MATLAB ``horace_core/sqw_models/example_sqw_background.m``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def example_sqw_background(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    pars: ArrayLike,
) -> np.ndarray:
    """Background S(Q, w) model for cubic system (Gaussian in energy).

    Translated from MATLAB ``horace_core/sqw_models/example_sqw_background.m``.

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Arrays of Q and energy values at which to evaluate the model.
    pars : ArrayLike
        Parameters ``[ampl, en0, fwhh]``.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    en_arr = en.ravel()
    pars_arr = np.asarray(pars, dtype=np.float64).ravel()
    ampl = float(pars_arr[0])
    en0 = float(pars_arr[1])
    sig = float(pars_arr[2]) / math.sqrt(math.log(256.0))

    weight = (ampl / (sig * math.sqrt(2.0 * math.pi))) * np.exp(
        -((en_arr - en0) ** 2) / (2.0 * sig**2)
    )
    return weight

