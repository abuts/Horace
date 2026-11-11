"""Legacy 5-neighbour spin wave dispersion relation for body-centred cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/disp_bcc_hfm0.m``.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike


def disp_bcc_hfm0(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, par: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    """Spin wave dispersion relation for body-centred cubic Heisenberg ferromagnet (up to 5th neighbour).

    Translated from MATLAB ``horace_core/sqw_models/disp_bcc_hfm0.m``.

    Parameters
    ----------
    qh, qk, ql : np.ndarray
        Arrays of ``h, k, l``.
    par : ArrayLike
        Parameters ``[Seff, gap, JS_p5p5p5, JS_100, JS_110, JS_3p5p5p5, JS_111]``:
        - ``Seff``: Intensity scale factor
        - ``gap``: Gap at zone centre
        - ``JS_p5p5p5``: First neighbour exchange constant
        - ``JS_100``: Second neighbour exchange constant
        - ``JS_110``: Third neighbour exchange constant
        - ``JS_3p5p5p5``: Fourth neighbour exchange constant
        - ``JS_111``: Fifth neighbour exchange constant

        Note: each pair of spins in the Hamiltonian appears only once.

    Returns
    -------
    wdisp : np.ndarray
        Array of energies for the dispersion.
    sf : np.ndarray
        Array of spectral weights.
    """
    qh_arr = qh.ravel()
    qk_arr = qk.ravel()
    ql_arr = ql.ravel()
    par_arr = np.asarray(par).ravel()

    Seff = float(par_arr[0])
    gap = float(par_arr[1])
    JS_p5p5p5 = float(par_arr[2])
    JS_100 = float(par_arr[3])
    JS_110 = float(par_arr[4])
    JS_3p5p5p5 = float(par_arr[5])
    JS_111 = float(par_arr[6])

    w = gap * np.ones_like(qh_arr, dtype=np.float64)

    if JS_p5p5p5 != 0.0 or JS_3p5p5p5 != 0.0:
        cos1h = np.cos(math.pi * qh_arr)
        cos1k = np.cos(math.pi * qk_arr)
        cos1l = np.cos(math.pi * ql_arr)

    if JS_110 != 0.0 or JS_111 != 0.0:
        cos2h = np.cos((2.0 * math.pi) * qh_arr)
        cos2k = np.cos((2.0 * math.pi) * qk_arr)
        cos2l = np.cos((2.0 * math.pi) * ql_arr)

    if JS_p5p5p5 != 0.0:
        w = w + (8.0 * JS_p5p5p5) * (1.0 - cos1h * cos1k * cos1l)

    if JS_100 != 0.0:
        w = w + (4.0 * JS_100) * (
            np.sin(math.pi * qh_arr) ** 2
            + np.sin(math.pi * qk_arr) ** 2
            + np.sin(math.pi * ql_arr) ** 2
        )

    if JS_110 != 0.0:
        w = w + (4.0 * JS_110) * (
            3.0 - cos2h * cos2k - cos2k * cos2l - cos2l * cos2h
        )

    if JS_3p5p5p5 != 0.0:
        cos3h = np.cos((3.0 * math.pi) * qh_arr)
        cos3k = np.cos((3.0 * math.pi) * qk_arr)
        cos3l = np.cos((3.0 * math.pi) * ql_arr)
        w = w + (8.0 * JS_3p5p5p5) * (
            3.0
            - cos3h * cos1k * cos1l
            - cos3k * cos1l * cos1h
            - cos3l * cos1h * cos1k
        )

    if JS_111 != 0.0:
        w = w + (8.0 * JS_111) * (1.0 - cos2h * cos2k * cos2l)

    sf = (Seff / 2.0) * np.ones_like(w, dtype=np.float64)
    return w.ravel(), sf.ravel()

