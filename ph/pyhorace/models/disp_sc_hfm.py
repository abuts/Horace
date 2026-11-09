"""Spin wave dispersion relation for the simple cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/disp_sc_hfm.m``.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from pyhorace.models.disp_bcc_hfm import _fdisp

# Module-level cache matching MATLAB `persistent rho`
_rho: Optional[np.ndarray] = None


def _reset_rho() -> None:
    """Reset the cached nearest-neighbour site array ``_rho``."""
    global _rho
    _rho = None


def disp_sc_hfm(
    qh: Any, qk: Any, ql: Any, par: Sequence[float] | np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    r"""Spin wave dispersion relation for the simple cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/disp_sc_hfm.m``.

    The dispersion is for the following Hamiltonian:

        ``H = -(1/2) Sum(J(i,j) S(i).S(j))``

    so that positive ``J`` favours ferromagnetism and each pair of spins appears only
    once. In addition, a single ion anisotropy is included that opens a gap at the
    magnetic zone centre.

    The algorithm works for intersite interactions out to arbitrarily great
    distance, as determined by the length of the vector containing the interaction
    parameters, input argument ``par``, below.

    Parameters
    ----------
    qh, qk, ql : array_like or float
        Arrays of ``h, k, l`` at which to compute the spin wave energies and
        spectral weights.
    par : sequence of float or np.ndarray
        Parameters ``[Seff, gap, JS1, JS2, JS3, JS4, ...]``:
        - ``Seff``: Intensity scale factor as an effective local moment spin with
          value ``Seff`` on each atomic site.
        - ``gap``: Gap at zone centre.
        - ``JS1, JS2, ...``: First, second, etc. neighbour exchange constants.

        The first 15 neighbours are:
        - ``JS1``:  JS to ``[1, 0, 0]``     (nearest neighbour)
        - ``JS2``:  JS to ``[1, 1, 0]``     (2nd neighbour)
        - ``JS3``:  JS to ``[1, 1, 1]``     (3rd neighbour)
        - ``JS4``:  JS to ``[2, 0, 0]``     (4th neighbour)
        - ``JS5``:  JS to ``[2, 1, 0]``     (5th neighbour)
        - ``JS6``:  JS to ``[2, 1, 1]``     (6th neighbour)
        - ``JS7``:  JS to ``[2, 2, 0]``     (7th neighbour)
        - ``JS8``:  JS to ``[3, 0, 0]``     (equal 8th neighbour)
        - ``JS9``:  JS to ``[2, 2, 1]``     (equal 8th neighbour)
        - ``JS10``: JS to ``[3, 1, 0]``     (10th neighbour)
        - ``JS11``: JS to ``[3, 1, 1]``     (11th neighbour)
        - ``JS12``: JS to ``[2, 2, 2]``     (12th neighbour)
        - ``JS13``: JS to ``[3, 2, 0]``     (13th neighbour)
        - ``JS14``: JS to ``[3, 2, 1]``     (14th neighbour)
        - ``JS15``: JS to ``[4, 0, 0]``     (15th neighbour)

    Returns
    -------
    wdisp : list of np.ndarray
        Single-element list containing the array of spin wave energies at the
        points defined by ``qh, qk, ql``.
    sf : list of np.ndarray
        Single-element list containing the array of corresponding spectral
        weights, ``Seff / 2`` (i.e. ``par[0] / 2``).
    """
    global _rho

    qh_arr = np.asarray(qh, dtype=np.float64)
    qk_arr = np.asarray(qk, dtype=np.float64)
    ql_arr = np.asarray(ql, dtype=np.float64)
    par_arr = np.asarray(par, dtype=np.float64).ravel()

    Seff = float(par_arr[0])
    gap = float(par_arr[1])
    JS = par_arr[2:]
    npar = JS.size

    n_cached = 0 if _rho is None else _rho.shape[0]
    if npar > n_cached:
        d_upper_bound = _d_upper_bound_sc(max(npar, 15))
        _rho, _ = _sc_atom_sites(d_upper_bound)

    w = gap * np.ones_like(qh_arr, dtype=np.float64)
    if npar > 0 and _rho is not None:
        for i in range(npar):
            if JS[i] != 0.0:
                w = w + JS[i] * _fdisp(qh_arr, qk_arr, ql_arr, _rho[i, :])

    wdisp = [w]
    sf = [(Seff / 2.0) * np.ones_like(w, dtype=np.float64)]
    return wdisp, sf


def _d_upper_bound_sc(N: int) -> float:
    """Return an upper bound for the distance of the Nth neighbour interaction for a simple cubic lattice.

    Expressed in units of the cube side: ``d_upper_bound = (36 * N / pi) ** (1 / 3)``.
    """
    return float((36.0 * N / math.pi) ** (1.0 / 3.0))


def _sc_atom_sites(rmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get distances to neighbouring atoms for the sc lattice in a sphere of radius ``rmax``.

    Parameters
    ----------
    rmax : float
        Maximum distance from origin in units of the cube side length.

    Returns
    -------
    rho : np.ndarray
        Array of unique sites of shape ``(npar, 3)`` sorted by increasing distance
        from the origin, with ``rho[i, 0] >= rho[i, 1] >= rho[i, 2]`` and ties broken
        by decreasing ``rho[:, 0]``, then decreasing ``rho[:, 1]``.
    dist : np.ndarray
        1D array of distances from the origin of length ``npar``.
    """
    small = 1e-10
    R = int(math.ceil(rmax))

    N_vertices = ((R + 1) * (R + 2) * (R + 3)) // 6 - 1
    x = np.zeros(N_vertices, dtype=np.float64)
    y = np.zeros(N_vertices, dtype=np.float64)
    z = np.zeros(N_vertices, dtype=np.float64)

    i = 0
    for ix in range(1, R + 1):
        for iy in range(0, ix + 1):
            for iz in range(0, iy + 1):
                x[i] = ix
                y[i] = iy
                z[i] = iz
                i += 1

    dsqr = x**2 + y**2 + z**2
    ok = dsqr <= (rmax**2 + small)
    dsqr = dsqr[ok]
    x = x[ok]
    y = y[ok]
    z = z[ok]

    order = np.lexsort((-z, -y, -x, dsqr))
    rho = np.column_stack([x[order], y[order], z[order]])
    dist = np.sqrt(dsqr[order])
    return rho, dist


d_upper_bound_sc = _d_upper_bound_sc
sc_atom_sites = _sc_atom_sites

