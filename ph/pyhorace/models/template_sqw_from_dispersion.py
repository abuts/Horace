"""Template for a function to broaden a dispersion relation.

Translated from MATLAB ``horace_core/sqw_models/template_sqw_from_dispersion.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from pyhorace.models.bose_times_eps import bose_times_eps
from pyhorace.models.dsho_over_eps import dsho_over_eps


def template_sqw_from_dispersion(
    qh: Any,
    qk: Any,
    ql: Any,
    en: Any,
    par: Sequence[float] | np.ndarray,
    my_dispersion_relation: Optional[Callable[..., Tuple[Any, Any]]] = None,
) -> np.ndarray:
    r"""Template for a function to broaden a dispersion relation.

    Translated from MATLAB ``horace_core/sqw_models/template_sqw_from_dispersion.m``.

    The template shows how to take a delta-function dispersion relation

        ``(<n(en) + 1> * delta(en - en(q)) + <n(en)> * delta(en + en(q)))``

    where
        ``<n(en) + 1> = 1 / (1 - exp(-en / (kB * T)))``
        ``<n(en)>     = 1 / (exp(en / (kB * T)) - 1)``

    and broaden it by the response for a damped simple harmonic oscillator with
    inverse lifetime ``gamma``.

    Parameters
    ----------
    qh, qk, ql : array_like
        Arrays of ``h, k, l``.
    en : array_like
        Array of energy transfers.
    par : sequence of float or np.ndarray
        Parameters ``[T, gamma, par[2], par[3], ...]``:
        - ``T``: Temperature (K)
        - ``gamma``: Inverse lifetime (meV)
        - ``par[2:]``: Parameters required to calculate the dispersion relations.
    my_dispersion_relation : callable, optional
        Function with signature ``(qh, qk, ql, par[2:]) -> (wdisp, idisp)``.

    Returns
    -------
    np.ndarray
        Spectral weight.
    """
    par_arr = np.asarray(par, dtype=np.float64).ravel()
    T = float(par_arr[0])
    gamma = float(par_arr[1])

    disp_fn = my_dispersion_relation or globals().get("my_dispersion_relation")
    if disp_fn is None or not callable(disp_fn):
        raise NotImplementedError(
            "template_sqw_from_dispersion requires a callable `my_dispersion_relation`."
        )

    wdisp, idisp = disp_fn(qh, qk, ql, par_arr[2:])
    bose = bose_times_eps(en, T)

    if isinstance(wdisp, (list, tuple)):
        w0 = np.asarray(wdisp[0], dtype=np.float64)
        weight = np.zeros_like(w0, dtype=np.float64)
        for w_i, id_i in zip(wdisp, idisp):
            weight = weight + np.asarray(id_i, dtype=np.float64) * (
                dsho_over_eps(en, w_i, gamma) * bose
            )
    else:
        weight = np.asarray(idisp, dtype=np.float64) * (
            dsho_over_eps(en, wdisp, gamma) * bose
        )

    return weight

