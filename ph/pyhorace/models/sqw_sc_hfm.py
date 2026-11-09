"""Spectral weight for simple cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/sqw_sc_hfm.m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyhorace.models.bose_times_eps import bose_times_eps
from pyhorace.models.disp_sc_hfm import disp_sc_hfm
from pyhorace.models.dsho_over_eps import dsho_over_eps


def sqw_sc_hfm(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    par: Sequence[float] | np.ndarray,
) -> np.ndarray:
    r"""Spectral weight for simple cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/sqw_sc_hfm.m``.

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
        Parameters ``[T, gamma, Seff, gap, JS_100, JS_110, JS_111, JS_200, ...]``:
        - ``T``: Temperature (K)
        - ``gamma``: Inverse lifetime (meV)
        - ``Seff``: Intensity scale factor
        - ``gap``: Gap at zone centre
        - ``JS_100, ...``: First, second, etc. neighbour exchange constants.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    par_arr = np.asarray(par, dtype=np.float64).ravel()
    T = float(par_arr[0])
    gamma = float(par_arr[1])

    wdisp, idisp = disp_sc_hfm(qh, qk, ql, par_arr[2:])
    weight = idisp[0] * (
        dsho_over_eps(en, wdisp[0], gamma) * bose_times_eps(en, T)
    )
    return weight

