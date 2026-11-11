"""Spin wave dispersion relation for the face-centred cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/disp_fcc_hfm.m``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike

from pyhorace.models.disp_bcc_hfm import _fdisp

# Module-level cache matching MATLAB `persistent rho`
_rho: Optional[np.ndarray] = None


def _reset_rho() -> None:
    """Reset the cached nearest-neighbour site array ``_rho``."""
    global _rho
    _rho = None


def disp_fcc_hfm(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, par: ArrayLike
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Spin wave dispersion relation for the face-centred cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/disp_fcc_hfm.m``.

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
    qh, qk, ql : np.ndarray
        Arrays of ``h, k, l`` coordinates at which to compute the spin wave
        energies and spectral weights.
    par : ArrayLike
        Parameters ``[Seff, gap, JS1, JS2, JS3, JS4, ...]``:
        - ``Seff``: Intensity scale factor as an effective local moment spin with
          value ``Seff`` on each atomic site.
        - ``gap``: Gap at zone centre.
        - ``JS1, JS2, ...``: First, second, etc. neighbour exchange constants.

        The first 15 neighbours are:
        - ``JS1``:  JS to ``[0.5, 0.5, 0]``     (nearest neighbour)
        - ``JS2``:  JS to ``[1, 0, 0]``         (2nd neighbour)
        - ``JS3``:  JS to ``[1, 0.5, 0.5]``     (3rd neighbour)
        - ``JS4``:  JS to ``[1, 1, 0]``         (4th neighbour)
        - ``JS5``:  JS to ``[1.5, 0.5, 0]``     (5th neighbour)
        - ``JS6``:  JS to ``[1, 1, 1]``         (6th neighbour)
        - ``JS7``:  JS to ``[1.5, 1, 0.5]``     (7th neighbour)
        - ``JS8``:  JS to ``[2, 0, 0]``         (8th neighbour)
        - ``JS9``:  JS to ``[2, 0.5, 0.5]``     (equal 9th neighbour)
        - ``JS10``: JS to ``[1.5, 1.5, 0]``     (equal 9th neighbour)
        - ``JS11``: JS to ``[2, 1, 0]``         (11th neighbour)
        - ``JS12``: JS to ``[1.5, 1.5, 1]``     (12th neighbour)
        - ``JS13``: JS to ``[2, 1, 1]``         (13th neighbour)
        - ``JS14``: JS to ``[2.5, 0.5, 0]``     (equal 14th neighbour)
        - ``JS15``: JS to ``[2, 1.5, 0.5]``     (equal 14th neighbour)

    Returns
    -------
    wdisp : np.ndarray
        Array of spin wave energies at the points defined by ``qh, qk, ql``.
    sf : np.ndarray
        Array of corresponding spectral weights, ``Seff / 2`` (i.e. ``par[0] / 2``).
    """
    global _rho

    qh_arr = qh.ravel()
    qk_arr = qk.ravel()
    ql_arr = ql.ravel()
    par_arr = np.asarray(par).ravel()

    Seff = float(par_arr[0])
    gap = float(par_arr[1])
    JS = par_arr[2:]
    npar = JS.size

    n_cached = 0 if _rho is None else _rho.shape[0]
    if npar > n_cached:
        rmax = _d_upper_bound_fcc(max(npar, 15))
        _rho, _ = _fcc_atom_sites(rmax)

    w = gap * np.ones_like(qh_arr, dtype=np.float64)
    if npar > 0 and _rho is not None:
        for i in range(npar):
            if JS[i] != 0.0:
                w = w + JS[i] * _fdisp(qh_arr, qk_arr, ql_arr, _rho[i, :])

    sf = (Seff / 2.0) * np.ones_like(w, dtype=np.float64)
    return w.ravel(), sf.ravel()


def _d_upper_bound_fcc(N: int) -> float:
    """Return an upper bound for the distance of the Nth neighbour interaction for an fcc lattice.

    Expressed in units of the cube side: ``d_upper_bound = (9 * N / pi) ** (1 / 3)``.
    """
    return float((9.0 * N / math.pi) ** (1.0 / 3.0))


def _fcc_atom_sites(rmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get distances to neighbouring atoms for the fcc lattice in a sphere of radius ``rmax``.

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

    N_vertices = ((R + 1) * (R + 2) * (R + 3)) // 6
    x = np.zeros(N_vertices, dtype=np.float64)
    y = np.zeros(N_vertices, dtype=np.float64)
    z = np.zeros(N_vertices, dtype=np.float64)

    i = 0
    for ix in range(0, 2 * R + 1, 2):
        for iy in range(0, ix + 1, 2):
            for iz in range(0, iy + 1, 2):
                x[i] = ix
                y[i] = iy
                z[i] = iz
                i += 1

    N_1by2_1by2_0 = (R * (R + 1) * (R + 2)) // 6
    N_1_1by2_1by2 = (R * (R + 1) * (R + 2)) // 6
    N_3by2_1_1by2 = (R * (R - 1) * (R + 1)) // 6

    x = np.concatenate(
        [
            x,
            x[:N_1by2_1by2_0] + 1.0,
            x[:N_1_1by2_1by2] + 2.0,
            x[:N_3by2_1_1by2] + 3.0,
        ]
    )
    y = np.concatenate(
        [
            y,
            y[:N_1by2_1by2_0] + 1.0,
            y[:N_1_1by2_1by2] + 1.0,
            y[:N_3by2_1_1by2] + 2.0,
        ]
    )
    z = np.concatenate(
        [
            z,
            z[:N_1by2_1by2_0],
            z[:N_1_1by2_1by2] + 1.0,
            z[:N_3by2_1_1by2] + 1.0,
        ]
    )

    dsqr = x**2 + y**2 + z**2
    ok = (dsqr > 0.0) & (dsqr <= 4.0 * (rmax**2 + small))
    dsqr = dsqr[ok]
    x = x[ok]
    y = y[ok]
    z = z[ok]

    order = np.lexsort((-z, -y, -x, dsqr))
    rho = np.column_stack([x[order], y[order], z[order]]) / 2.0
    dist = np.sqrt(dsqr[order]) / 2.0
    return rho, dist


d_upper_bound_fcc = _d_upper_bound_fcc
fcc_atom_sites = _fcc_atom_sites

