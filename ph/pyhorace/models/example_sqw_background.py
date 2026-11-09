"""Background S(Q, w) model for a cubic system.

Translated from MATLAB ``horace_core/sqw_models/example_sqw_background.m``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def example_sqw_background(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    pars: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Background S(Q, w) model for cubic system (Gaussian in energy).

    Translated from MATLAB ``horace_core/sqw_models/example_sqw_background.m``.

    Parameters
    ----------
    qh, qk, ql, en : array_like
        Arrays of Q and energy values at which to evaluate the model.
    pars : sequence of float or np.ndarray
        Parameters ``[ampl, en0, fwhh]``.

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

