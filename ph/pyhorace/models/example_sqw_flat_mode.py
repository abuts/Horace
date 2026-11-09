"""Example S(Q, w) model for a dispersionless mode.

Translated from MATLAB ``horace_core/sqw_models/example_sqw_flat_mode.m``.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def example_sqw_flat_mode(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    pars: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Dispersionless mode at non-zero energy ``en0`` approximated as a Gaussian in energy (plus optional Lorentzian).

    Translated from MATLAB ``horace_core/sqw_models/example_sqw_flat_mode.m``.

    Parameters
    ----------
    qh, qk, ql, en : array_like
        Arrays of Q and energy values at which to evaluate dispersion.
    pars : sequence of float or np.ndarray
        Either ``[Amplitude, en0, fwhh]`` (3 parameters) for a pure Gaussian
        dispersionless mode, or ``[HE, CE, fwhh, H1, G1, T]`` (6 parameters)
        including an inelastic detailed-balance Lorentzian component.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    en_arr = np.asarray(en, dtype=np.float64)
    pars_arr = np.asarray(pars, dtype=np.float64).ravel()

    HE = float(pars_arr[0])
    CE = float(pars_arr[1])
    GE = float(pars_arr[2]) / math.sqrt(math.log(256.0))

    elastic = (HE / (GE * math.sqrt(2.0 * math.pi))) * np.exp(
        -((en_arr - CE) ** 2) / (2.0 * GE**2)
    )

    if pars_arr.size < 6:
        return elastic

    H1 = float(pars_arr[3])
    G1 = float(pars_arr[4])
    T = float(pars_arr[5])

    kB = 1.0 / 11.606
    Eb = en_arr / (kB * T)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dbal = np.where(en_arr != 0.0, en_arr / (1.0 - np.exp(-Eb)), kB * T)

    inelastic = dbal * H1 / ((en_arr - CE) ** 2 + G1**2)
    lor = elastic + inelastic
    return lor

