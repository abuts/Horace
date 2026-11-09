"""Damped simple harmonic oscillator (DSHO) response divided by energy.

Translated from MATLAB ``herbert_core/utilities/maths_functions/dsho_over_eps.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def dsho_over_eps(en: Any, en0: Any, gam: Any) -> np.ndarray:
    r"""Delta function response function divided by energy, broadened as DSHO.

    Translated from MATLAB ``herbert_core/utilities/maths_functions/dsho_over_eps.m``.

    This function broadens the delta function response:

        ``(delta(en - en0) + delta(en + en0)) / en``

    as a damped simple harmonic oscillator.
    Multiply by ``en / (1 - exp(-en / (kB * T)))`` as returned by the output of
    :func:`bose_times_eps` to get the DSHO broadened response of:

        ``<n(en) + 1> * delta(en - en0) + <n(en)> * delta(en + en0)``

    Parameters
    ----------
    en : array_like
        Array of energy transfers.
    en0 : array_like
        Array of energies of the dispersion. Array with the same size as ``en``.
    gam : float or array_like
        Inverse lifetime (same units as ``en`` and ``en0``).
        If ``gam`` is scalar, then expanded to the same size as ``en`` and ``en0``.

    Returns
    -------
    np.ndarray
        Array of calculated response (has the same size as ``en``).
    """
    en_arr = np.asarray(en, dtype=np.float64)
    en0_arr = np.asarray(en0, dtype=np.float64)
    gam_arr = np.asarray(gam, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        y = ((4.0 / np.pi) * np.abs(gam_arr * en0_arr)) / (
            (en_arr**2 - en0_arr**2) ** 2 + 4.0 * (gam_arr * en_arr) ** 2
        )
    return y

