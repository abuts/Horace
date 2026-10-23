"""Calculate matrix that converts momentum from spectrometer frame to crystal Cartesian / r.l.u.

Translated from MATLAB ``herbert_core/utilities/scientific/calc_proj_matrix.m``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from .bmatrix import _infer_nargout, bmatrix
from .ubmatrix import ubmatrix


def calc_proj_matrix(
    var1: Any,
    var2: Any,
    u: Any = None,
    v: Any = None,
    psi: float = 0.0,
    omega: float = 0.0,
    dpsi: float = 0.0,
    gl: float = 0.0,
    gs: float = 0.0,
    *args: Any,
    nargout: Optional[int] = None,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Calculate matrix that converts momentum from coordinates in spectrometer
    frame to projection axes defined by ``u1 || a*``, ``u2`` in plane of ``a*``
    and ``b*`` i.e. crystal Cartesian axes. Allows for correction of scattering
    plane ``(omega, dpsi, gl, gs)`` - see Tobyfit for conventions.

    Mode 1::

        >> spec_to_cc, u_to_rlu, spec_to_rlu = calc_proj_matrix(
               alatt, angdeg, u, v, psi, omega, dpsi, gl, gs
           )

    Mode 2::

        >> spec_to_cc, u_to_rlu, spec_to_rlu = calc_proj_matrix(
               b_matrix, u_matrix, '', '', psi, omega, dpsi, gl, gs
           )

    Parameters
    ----------
    var1 : array_like
        Mode 1: ``alatt`` - Lattice parameters (Ang) [3-element vector].
        Mode 2: ``b_matrix`` - 3x3 B-matrix transforming hkl to crystal Cartesian.
    var2 : array_like
        Mode 1: ``angdeg`` - Lattice angles (deg) [3-element vector].
        Mode 2: ``u_matrix`` - 3x3 U-matrix transforming crystal Cartesian to
        orthonormal frame defined by vectors ``u, v``.
    u : array_like or str
        Mode 1: First vector (1x3) defining scattering plane (r.l.u.).
        Mode 2: Empty ``''`` or ``None`` to distinguish Mode 2 from Mode 1.
    v : array_like or str
        Mode 1: Second vector (1x3) defining scattering plane (r.l.u.).
        Mode 2: Not used in Mode 2.
    psi : float, default 0.0
        Angle of ``u`` w.r.t. ``ki`` (rad).
    omega : float, default 0.0
        Angle of axis of small goniometer arc w.r.t. notional ``u`` (rad).
    dpsi : float, default 0.0
        Correction to ``psi`` (rad).
    gl : float, default 0.0
        Large goniometer arc angle (rad).
    gs : float, default 0.0
        Small goniometer arc angle (rad).
    *args : int, optional
        Optional ``n_matrix``: if present and lies in the range 1..3, only the
        selected matrix (1: ``spec_to_cc``, 2: ``u_to_rlu``, 3: ``spec_to_rlu``)
        is returned as a single output.
    nargout : int, optional
        Explicit number of outputs (1, 2, or 3) when ``n_matrix`` is not specified.

    Returns
    -------
    spec_to_cc : np.ndarray
        Matrix (3x3) to convert momentum from coordinates in spectrometer
        frame to crystal Cartesian axes: ``v_crystal_Cart = spec_to_cc @ v_spec``.
    u_to_rlu : np.ndarray, optional
        Matrix (3x3) of crystal Cartesian axes in reciprocal lattice units:
        ``v_rlu = u_to_rlu @ v_crystal_Cart`` (same as ``inv(B)``).
    spec_to_rlu : np.ndarray, optional
        Matrix (3x3) to convert from spectrometer coordinates to r.l.u.:
        ``v_rlu = spec_to_rlu @ v_spec``.
    """
    mat_to_return = 0
    if len(args) > 0 and args[0] is not None:
        mat_to_return = int(args[0])
        if mat_to_return < 1 or mat_to_return > 3:
            mat_to_return = 0

    u_is_empty = (
        u is None
        or (isinstance(u, (str, list, tuple)) and len(u) == 0)
        or (isinstance(u, np.ndarray) and u.size == 0)
    )
    if u_is_empty:
        b_matrix = np.asarray(var1, dtype=float)
        u_matrix = np.asarray(var2, dtype=float)
    else:
        alatt = var1
        angdeg = var2
        b_matrix = bmatrix(alatt, angdeg, nargout=1)
        _, u_matrix = ubmatrix(u, v, b_matrix, nargout=2)

    psi = float(psi)
    omega = float(omega)
    dpsi = float(dpsi)
    gl = float(gl)
    gs = float(gs)

    rot_dpsi = np.array(
        [
            [np.cos(dpsi), -np.sin(dpsi), 0.0],
            [np.sin(dpsi), np.cos(dpsi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rot_gl = np.array(
        [
            [np.cos(gl), 0.0, np.sin(gl)],
            [0.0, 1.0, 0.0],
            [-np.sin(gl), 0.0, np.cos(gl)],
        ],
        dtype=float,
    )
    rot_gs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(gs), -np.sin(gs)],
            [0.0, np.sin(gs), np.cos(gs)],
        ],
        dtype=float,
    )
    rot_om = np.array(
        [
            [np.cos(omega), -np.sin(omega), 0.0],
            [np.sin(omega), np.cos(omega), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    corr = (rot_om @ (rot_dpsi @ rot_gl @ rot_gs) @ rot_om.T).T

    cryst = np.array(
        [
            [np.cos(psi), np.sin(psi), 0.0],
            [-np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    if mat_to_return == 0:
        spec_to_cc = np.linalg.solve(u_matrix, corr @ cryst)
        nout = _infer_nargout(default=1, explicit_nargout=nargout)
        if nout == 1:
            return spec_to_cc
        u_to_rlu = np.linalg.inv(b_matrix)
        if nout == 2:
            return spec_to_cc, u_to_rlu
        spec_to_rlu = np.linalg.solve(b_matrix, spec_to_cc)
        return spec_to_cc, u_to_rlu, spec_to_rlu
    else:
        if mat_to_return == 1:
            return np.linalg.solve(u_matrix, corr @ cryst)
        elif mat_to_return == 2:
            return np.linalg.inv(b_matrix)
        else:
            return np.linalg.solve(u_matrix @ b_matrix, corr @ cryst)

