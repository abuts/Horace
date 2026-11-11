"""Spin wave dispersion relation for the body-centred cubic Heisenberg ferromagnet.

Translated from MATLAB ``horace_core/sqw_models/disp_bcc_hfm.m``.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

# Module-level cache matching MATLAB `persistent rho`
_rho: Optional[np.ndarray] = None


def _reset_rho() -> None:
    """Reset the cached nearest-neighbour site array ``_rho``."""
    global _rho
    _rho = None


def disp_bcc_hfm(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, par: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Spin wave dispersion relation for the body-centred cubic Heisenberg ferromagnet.

    Translated from MATLAB ``horace_core/sqw_models/disp_bcc_hfm.m``.

    The dispersion is for the following Hamiltonian:

        ``H = -(1/2) Sum(J(i,j) S(i).S(j))``

    so that positive ``J`` favours ferromagnetism and each pair of spins appears only
    once. In addition, a single ion anisotropy is included that opens a gap at the
    magnetic zone centre.

    The algorithm works for intersite interactions out to arbitrarily great
    distance, as determined by the length of the vector containing the interaction
    parameters, input argument ``par``, below.

    Reference:
    For a derivation of the linear spin wave theory dispersion relation and
    spectral weight see e.g.
    "Principles of Neutron Scattering from Condensed Matter",
    Author: A.T. Boothroyd (Oxford University Press, 2020).
    The general expression for the spin wave energies for a Heisenberg ferromagnet
    is given in Eq. 8.56 (page 273). Note the opposite sign convention for the
    exchange constants ``J``.

    Parameters
    ----------
    qh, qk, ql : float numpy.ndarray of ``h, k, l`` coordinates at which to compute 
        the spin wave energies and  spectral weights.
    par : sequence ofnp.ndarray
        Parameters ``[Seff, gap, JS1, JS2, JS3, JS4, ...]``:
        - ``Seff``: Intensity scale factor as an effective local moment spin with
          value ``Seff`` on each atomic site.
        - ``gap``: Gap at zone centre.
        - ``JS1, JS2, ...``: First, second, etc. neighbour exchange constants.
          Each is the product of ``J`` and ``Seff`` for the first, second, third
          neighbour, etc.

        The length of the parameters vector determines the range of the
        interactions. The vector can have an unlimited number of interaction
        parameters.

        The first 15 neighbours are:
        - ``JS1``:  JS to ``[0.5, 0.5, 0.5]``   (nearest neighbour)
        - ``JS2``:  JS to ``[1, 0, 0]``         (2nd neighbour)
        - ``JS3``:  JS to ``[1, 1, 0]``         (3rd neighbour)
        - ``JS4``:  JS to ``[1.5, 0.5, 0.5]``   (4th neighbour)
        - ``JS5``:  JS to ``[1, 1, 1]``         (5th neighbour)
        - ``JS6``:  JS to ``[2, 0, 0]``         (6th neighbour)
        - ``JS7``:  JS to ``[1.5, 1.5, 0.5]``   (7th neighbour)
        - ``JS8``:  JS to ``[2, 1, 0]``         (8th neighbour)
        - ``JS9``:  JS to ``[2, 1, 1]``         (9th neighbour)
        - ``JS10``: JS to ``[2.5, 0.5, 0.5]``   (equal 10th neighbour)
        - ``JS11``: JS to ``[1.5, 1.5, 1.5]``   (equal 10th neighbour)
        - ``JS12``: JS to ``[2, 2, 0]``         (12th neighbour)
        - ``JS13``: JS to ``[2.5, 1.5, 0.5]``   (13th neighbour)
        - ``JS14``: JS to ``[3, 0, 0]``         (equal 14th neighbour)
        - ``JS15``: JS to ``[2, 2, 1]``         (equal 14th neighbour)

        Note that if there are inequivalent sites at the same distance then the
        order is always decreasing ``x``, followed by decreasing ``y``.

    Returns
    -------
    wdisp :  np.ndarray of spin wave energies at the
        points defined by ``qh, qk, ql``.
    sf : np.ndarray of corresponding spectral
        weights, ``Seff / 2`` (i.e. ``par[0] / 2``).
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
        d_upper_bound = _d_upper_bound_bcc(max(npar, 15))
        _rho, _ = _bcc_atom_sites(d_upper_bound)

    w = gap * np.ones_like(qh_arr, dtype=np.float64)
    if npar > 0 and _rho is not None:
        for i in range(npar):
            if JS[i] != 0.0:
                w = w + JS[i] * _fdisp(qh_arr, qk_arr, ql_arr, _rho[i, :])

    sf = (Seff / 2.0) * np.ones_like(w, dtype=np.float64)
    return w.ravel(), sf.ravel()


def _d_upper_bound_bcc(N: int) -> float:
    """Return an upper bound for the distance of the Nth neighbour interaction for a bcc lattice.

    Expressed in units of the cube side: ``d_upper_bound = (18 * N / pi) ** (1 / 3)``.
    """
    return float((18.0 * N / math.pi) ** (1.0 / 3.0))


def _bcc_atom_sites(rmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Get distances to neighbouring atoms for the bcc lattice in a sphere of radius ``rmax``.

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

    N_centres = (R * (R + 1) * (R + 2)) // 6
    x = np.concatenate([x, x[:N_centres] + 1.0])
    y = np.concatenate([y, y[:N_centres] + 1.0])
    z = np.concatenate([z, z[:N_centres] + 1.0])

    dsqr = x**2 + y**2 + z**2
    ok = (dsqr > 0.0) & (dsqr <= 4.0 * (rmax**2 + small))
    dsqr = dsqr[ok]
    x = x[ok]
    y = y[ok]
    z = z[ok]

    # Sort: dsqr ascending, x descending, y descending, z descending
    order = np.lexsort((-z, -y, -x, dsqr))
    rho = np.column_stack([x[order], y[order], z[order]]) / 2.0
    dist = np.sqrt(dsqr[order]) / 2.0
    return rho, dist


def _fdisp(qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return the contribution from symmetry equivalent sites to cubic Heisenberg ferromagnet.

    Returns ``w = M - Sum({r"}, exp(2*pi*i*(h*x" + k*y" + l*z")))`` where ``M`` is
    the number of atomic sites symmetry equivalent to ``r = [x, y, z]``.
    """
    ind = np.flatnonzero(r != 0.0)
    num_non_zero = ind.size

    if num_non_zero == 1:
        return _f_x00(qh, qk, ql, float(r[ind[0]]))
    elif num_non_zero == 2:
        r0 = float(r[ind[0]])
        r1 = float(r[ind[1]])
        if r0 == r1:
            return _f_xx0(qh, qk, ql, r0)
        else:
            return _f_xy0(qh, qk, ql, r0, r1)
    elif num_non_zero == 3:
        r0 = float(r[ind[0]])
        r1 = float(r[ind[1]])
        r2 = float(r[ind[2]])
        if r0 == r1 and r1 == r2:
            return _f_xxx(qh, qk, ql, r0)
        elif r0 != r1 and r1 != r2 and r2 != r0:
            return _f_xyz(qh, qk, ql, r0, r1, r2)
        else:
            if r0 == r1:
                return _f_xxy(qh, qk, ql, r1, r2)
            elif r1 == r2:
                return _f_xxy(qh, qk, ql, r2, r0)
            else:
                return _f_xxy(qh, qk, ql, r0, r1)
    return np.zeros_like(qh, dtype=np.float64)


def _f_x00(qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float) -> np.ndarray:
    """Contribution from interactions along ``[x, 0, 0]`` in real space (6-fold degeneracy)."""
    px = math.pi * x
    return 4.0 * (np.sin(px * qh) ** 2 + np.sin(px * qk) ** 2 + np.sin(px * ql) ** 2)


def _f_xx0(qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float) -> np.ndarray:
    """Contribution from interactions along ``[x, x, 0]`` in real space (12-fold degeneracy)."""
    tpx = 2.0 * math.pi * x
    cos_xh = np.cos(tpx * qh)
    cos_xk = np.cos(tpx * qk)
    cos_xl = np.cos(tpx * ql)
    return 4.0 * (3.0 - cos_xh * cos_xk - cos_xk * cos_xl - cos_xl * cos_xh)


def _f_xxx(qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float) -> np.ndarray:
    """Contribution from interactions along ``[x, x, x]`` in real space (8-fold degeneracy)."""
    tpx = 2.0 * math.pi * x
    return 8.0 * (1.0 - np.cos(tpx * qh) * np.cos(tpx * qk) * np.cos(tpx * ql))


def _f_xy0(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float, y: float
) -> np.ndarray:
    """Contribution from interactions along ``[x, y, 0]`` (``x != y``) in real space (24-fold degeneracy)."""
    tpx = 2.0 * math.pi * x
    tpy = 2.0 * math.pi * y
    cos_xh = np.cos(tpx * qh)
    cos_xk = np.cos(tpx * qk)
    cos_xl = np.cos(tpx * ql)
    cos_yh = np.cos(tpy * qh)
    cos_yk = np.cos(tpy * qk)
    cos_yl = np.cos(tpy * ql)
    return 4.0 * (
        6.0
        - cos_xh * cos_yk
        - cos_yh * cos_xk
        - cos_xk * cos_yl
        - cos_yk * cos_xl
        - cos_xl * cos_yh
        - cos_yl * cos_xh
    )


def _f_xxy(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float, y: float
) -> np.ndarray:
    """Contribution from interactions along ``[x, x, y]`` (``x != y``) in real space (24-fold degeneracy)."""
    tpx = 2.0 * math.pi * x
    tpy = 2.0 * math.pi * y
    cos_xh = np.cos(tpx * qh)
    cos_xk = np.cos(tpx * qk)
    cos_xl = np.cos(tpx * ql)
    return 8.0 * (
        3.0
        - cos_xh * cos_xk * np.cos(tpy * ql)
        - cos_xk * cos_xl * np.cos(tpy * qh)
        - cos_xl * cos_xh * np.cos(tpy * qk)
    )


def _f_xyz(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, x: float, y: float, z: float
) -> np.ndarray:
    """Contribution from interactions along ``[x, y, z]`` (all different) in real space (48-fold degeneracy)."""
    tpx = 2.0 * math.pi * x
    tpy = 2.0 * math.pi * y
    tpz = 2.0 * math.pi * z
    cos_xh = np.cos(tpx * qh)
    cos_xk = np.cos(tpx * qk)
    cos_xl = np.cos(tpx * ql)
    cos_yh = np.cos(tpy * qh)
    cos_yk = np.cos(tpy * qk)
    cos_yl = np.cos(tpy * ql)
    cos_zh = np.cos(tpz * qh)
    cos_zk = np.cos(tpz * qk)
    cos_zl = np.cos(tpz * ql)
    return 8.0 * (
        6.0
        - cos_xh * cos_yk * cos_zl
        - cos_yh * cos_zk * cos_xl
        - cos_zh * cos_xk * cos_yl
        - cos_xh * cos_zk * cos_yl
        - cos_yh * cos_xk * cos_zl
        - cos_zh * cos_yk * cos_xl
    )


# Public aliases for helper functions
d_upper_bound_bcc = _d_upper_bound_bcc
bcc_atom_sites = _bcc_atom_sites
fdisp = _fdisp

