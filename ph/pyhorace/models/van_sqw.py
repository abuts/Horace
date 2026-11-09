"""Dispersionless mode at non-zero energy en0 approximated as a Gaussian in energy.

Translated from MATLAB ``horace_core/sqw_models/van_sqw.m``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def van_sqw(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    pars: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Dispersionless mode at non-zero energy ``en0`` approximated as a Gaussian in energy.

    Translated from MATLAB ``horace_core/sqw_models/van_sqw.m``.

    Parameters
    ----------
    qh, qk, ql, en : array_like
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : sequence of float or np.ndarray
        ``[Amplitude, en0, fwhh]`` - the area, centre and FWHH of a
        dispersionless mode.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    en_arr = np.asarray(en, dtype=np.float64)
    ampl = float(pars[0])
    en0 = float(pars[1])
    sig = float(pars[2]) / math.sqrt(math.log(256.0))

    weight = (ampl / (sig * math.sqrt(2.0 * math.pi))) * np.exp(
        -((en_arr - en0) ** 2) / (2.0 * sig**2)
    )
    return weight

