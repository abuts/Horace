"""Spectral weight for face-centred cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/sqw_fcc_hfm.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyhorace.models.bose_times_eps import bose_times_eps
from pyhorace.models.disp_fcc_hfm import disp_fcc_hfm
from pyhorace.models.dsho_over_eps import dsho_over_eps


def sqw_fcc_hfm(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    par: ArrayLike,
) -> np.ndarray:
    r"""Spectral weight for face-centred cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/sqw_fcc_hfm.m``.

    The spectral weight is for:

        ``(S/2) * (<n(en) + 1> * delta(en - en(q)) + <n(en)> * delta(en + en(q)))``

    broadened by the response for a damped simple harmonic oscillator with
    inverse lifetime ``gamma``.

    Parameters
    ----------
    qh, qk, ql : np.ndarray
        Arrays of ``h, k, l``.
    en : np.ndarray
        Array of energy transfers.
    par : ArrayLike
        Parameters ``[T, gamma, Seff, gap, JS_p5p50, JS_100, JS_1p5p5, JS_110, ...]``:
        - ``T``: Temperature (K)
        - ``gamma``: Inverse lifetime (meV)
        - ``Seff``: Intensity scale factor
        - ``gap``: Gap at zone centre
        - ``JS_p5p50, ...``: First, second, etc. neighbour exchange constants.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    par_arr = np.asarray(par, dtype=np.float64).ravel()
    T = float(par_arr[0])
    gamma = float(par_arr[1])

    wdisp, idisp = disp_fcc_hfm(qh, qk, ql, par_arr[2:])
    weight = idisp * (
        dsho_over_eps(en, wdisp, gamma) * bose_times_eps(en, T)
    )
    return weight

