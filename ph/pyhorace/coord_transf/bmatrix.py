"""Calculate B matrix of Busing and Levy.

Translated from MATLAB ``herbert_core/utilities/scientific/bmatrix.m``.
"""

from __future__ import annotations

import dis
import inspect
from typing import Any, Optional, Tuple, Union

import numpy as np


def _infer_nargout(
    default: int = 1,
    explicit_nargout: Optional[int] = None,
    depth: int = 2,
) -> int:
    """Infer the number of output arguments expected by the caller.

    Parameters
    ----------
    default : int, default 1
        Default number of return values when the caller does not unpack.
    explicit_nargout : int or None, default None
        Explicit ``nargout`` override if provided by caller.
    depth : int, default 2
        Call stack depth to inspect.

    Returns
    -------
    int
        Number of unpacked outputs at the call site, or ``default``.
    """
    if explicit_nargout is not None:
        return int(explicit_nargout)
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return default
        lasti = frame.f_lasti
        found_call = False
        for instr in dis.get_instructions(frame.f_code):
            if instr.offset == lasti:
                found_call = True
                continue
            if found_call:
                if instr.opname == "CACHE":
                    continue
                if instr.opname == "UNPACK_SEQUENCE":
                    return int(instr.argval)
                break
    except Exception:
        pass
    return default


def bmatrix(
    alatt: Any,
    angdeg: Any,
    *,
    nargout: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calculate B matrix of Busing and Levy, returning also the reciprocal
    lattice vector in Angstrom^-1 and the reciprocal lattice angles in degrees.

    Calling forms::

        >> b = bmatrix(alatt, ang)
        >> b, arlu, angrlu = bmatrix(alatt, ang)

    Parameters
    ----------
    alatt : array_like
        Vector containing lattice parameters (Ang) [row or column vector, length 3].
    angdeg : array_like
        Vector containing lattice angles (degrees) [row or column vector, length 3].
    nargout : int, optional
        Number of outputs to return (1, 2, or 3). Automatically inferred when
        unpacking in Python.

    Returns
    -------
    b : np.ndarray
        B matrix of Busing & Levy [3x3 matrix].
    arlu : np.ndarray, optional
        Reciprocal lattice vectors (Ang^-1) [1D array of length 3].
    angrlu : np.ndarray, optional
        Reciprocal lattice angles (deg) [1D array of length 3].

    Raises
    ------
    ValueError
        ``HORACE:bmatrix:invalid_argument`` if lattice angles are negative or
        larger than 180 deg, or lattice parameters are less than or equal to zero,
        or inputs are not 3-element numeric vectors of matching shape.

    Notes
    -----
    Matrix B is used to transform components of a vector in r.l.u. to those
    in crystal Cartesian coordinates, that is, an orthonormal frame in which
    ``x || a*``, ``z || cross(a*, b*)``, ``y perp. x, z``::

        Vcryst[i] = B[i, j] * Vrlu[j]
    """
    if (
        alatt is None
        or angdeg is None
        or isinstance(alatt, (str, bool, dict))
        or isinstance(angdeg, (str, bool, dict))
    ):
        raise ValueError(
            f"HORACE:bmatrix:invalid_argument: bmatrix input should contain two 3-element "
            f"vectors containing lattice parameters and lattice angles. It is: {alatt}, {angdeg}"
        )
    try:
        alatt_arr = np.asarray(alatt, dtype=float)
        angdeg_arr = np.asarray(angdeg, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"HORACE:bmatrix:invalid_argument: bmatrix input should contain two 3-element "
            f"vectors containing lattice parameters and lattice angles. It is: {alatt}, {angdeg}"
        ) from exc

    if (
        alatt_arr.size != 3
        or angdeg_arr.size != 3
        or alatt_arr.shape != angdeg_arr.shape
    ):
        raise ValueError(
            f"HORACE:bmatrix:invalid_argument: bmatrix input should contain two 3-element "
            f"vectors containing lattice parameters and lattice angles. It is: {alatt}, {angdeg}"
        )

    alatt_1d = alatt_arr.ravel()
    angdeg_1d = angdeg_arr.ravel()

    if np.max(angdeg_1d) >= 180.0 or np.min(angdeg_1d) <= 0.0:
        raise ValueError(
            "HORACE:bmatrix:invalid_argument: some lattice angles bigger than 180 deg or less than 0 deg"
        )

    if np.min(alatt_1d) <= 0.0:
        raise ValueError(
            "HORACE:bmatrix:invalid_argument: Some lattice parameters are less than 0"
        )

    ang = np.deg2rad(angdeg_1d)
    cosa = np.cos(ang)
    sina = np.abs(np.sin(ang))

    a_mat = np.array(
        [
            [1.0, cosa[2], cosa[1]],
            [cosa[2], 1.0, cosa[0]],
            [cosa[1], cosa[0], 1.0],
        ],
        dtype=float,
    )

    q = np.sqrt(np.abs(np.linalg.det(a_mat)))
    arlu = (2.0 * np.pi / q) * (sina / alatt_1d)

    # Reciprocal lattice angles
    aa = np.arccos((cosa[1] * cosa[2] - cosa[0]) / (sina[1] * sina[2]))
    bb = np.arccos((cosa[2] * cosa[0] - cosa[1]) / (sina[2] * sina[0]))
    cc = np.arccos((cosa[0] * cosa[1] - cosa[2]) / (sina[0] * sina[1]))

    # B-matrix as in Acta Cryst. (1967). 22, 457
    b = np.array(
        [
            [arlu[0], arlu[1] * np.cos(cc), arlu[2] * np.cos(bb)],
            [0.0, arlu[1] * np.abs(np.sin(cc)), -arlu[2] * np.abs(np.sin(bb)) * cosa[0]],
            [0.0, 0.0, 2.0 * np.pi / alatt_1d[2]],
        ],
        dtype=float,
    )

    angrlu = np.rad2deg(np.array([aa, bb, cc], dtype=float))

    nout = _infer_nargout(default=1, explicit_nargout=nargout)
    if nout == 1:
        return b
    elif nout == 2:
        return b, arlu
    else:
        return b, arlu, angrlu

