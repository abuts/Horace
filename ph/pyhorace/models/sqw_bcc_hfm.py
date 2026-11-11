"""Spectral weight for body-centred cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/sqw_bcc_hfm.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyhorace.models.bose_times_eps import bose_times_eps
from pyhorace.models.disp_bcc_hfm import disp_bcc_hfm
from pyhorace.models.dsho_over_eps import dsho_over_eps


def sqw_bcc_hfm(
    qh: np.ndarray,
    qk: np.ndarray,
    ql: np.ndarray,
    en: np.ndarray,
    par: ArrayLike,
) -> np.ndarray:
    r"""Spectral weight for body-centred cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/sqw_bcc_hfm.m``.

    The spectral weight is for:

        ``(S/2) * (<n(en) + 1> * delta(en - en(q)) + <n(en)> * delta(en + en(q)))``

    broadened by the response for a damped simple harmonic oscillator with
    inverse lifetime ``gamma``.

    Parameters
    ----------
    qh, qk, ql : array_like
        Arrays of ``h, k, l``.
    en : array_like
        Array of energy transfers.
    par : sequence of float or np.ndarray
        Parameters ``[T, gamma, Seff, gap, JS_p5p5p5, JS_100, JS_110, JS_3p5p5p5, JS_111, ...]``:
        - ``T``: Temperature (K)
        - ``gamma``: Inverse lifetime (meV)
        - ``Seff``: Intensity scale factor
        - ``gap``: Gap at zone centre
        - ``JS_p5p5p5, ...``: First, second, etc. neighbour exchange constants.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    par_arr = np.asarray(par, dtype=np.float64).ravel()
    T = float(par_arr[0])
    gamma = float(par_arr[1])

    wdisp, idisp = disp_bcc_hfm(qh, qk, ql, par_arr[2:])
    weight = idisp * (
        dsho_over_eps(en, wdisp, gamma) * bose_times_eps(en, T)
    )
    return weight

