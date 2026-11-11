"""Dispersionless mode at non-zero energy en0 approximated as a Gaussian in energy.

Translated from MATLAB ``horace_core/sqw_models/van_sqw.m``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike


def van_sqw(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    pars: ArrayLike,
) -> np.ndarray:
    """Dispersionless mode at non-zero energy ``en0`` approximated as a Gaussian in energy.

    Translated from MATLAB ``horace_core/sqw_models/van_sqw.m``.

    Parameters
    ----------
    qh, qk, ql, en : np.ndarray
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : ArrayLike
        ``[Amplitude, en0, fwhh]`` - the area, centre and FWHH of a
        dispersionless mode.

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

