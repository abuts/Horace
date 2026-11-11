"""Damped simple harmonic oscillator (DSHO) response divided by energy.

Translated from MATLAB ``herbert_core/utilities/maths_functions/dsho_over_eps.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def dsho_over_eps(en: np.ndarray, en0: np.ndarray, gam: ArrayLike) -> np.ndarray:
    """Delta function response function divided by energy, broadened as DSHO.

    Translated from MATLAB ``herbert_core/utilities/maths_functions/dsho_over_eps.m``.

    This function broadens the delta function response:

        ``(delta(en - en0) + delta(en + en0)) / en``

    as a damped simple harmonic oscillator.
    Multiply by ``en / (1 - exp(-en / (kB * T)))`` as returned by the output of
    :func:`bose_times_eps` to get the DSHO broadened response of:

        ``<n(en) + 1> * delta(en - en0) + <n(en)> * delta(en + en0)``

    Parameters
    ----------
    en : np.ndarray
        Array of energy transfers.
    en0 : np.ndarray
        Array of energies of the dispersion. Array with the same size as ``en``.
    gam : float or ArrayLike
        Inverse lifetime (same units as ``en`` and ``en0``).
        If ``gam`` is scalar, then expanded to the same size as ``en`` and ``en0``.

    Returns
    -------
    np.ndarray
        Array of calculated response (has the same size as ``en``).
    """
    en_arr = en.ravel()
    en0_arr = en0.ravel()
    gam_arr = np.asarray(gam, dtype=np.float64)
    if gam_arr.ndim > 0:
        gam_arr = gam_arr.ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        y = ((4.0 / np.pi) * np.abs(gam_arr * en0_arr)) / (
            ((en_arr - en0_arr) * (en_arr + en0_arr)) ** 2
            + 4.0 * (gam_arr * en_arr) ** 2
        )
    return y.ravel()

