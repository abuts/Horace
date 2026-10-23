"""Calculate UB matrix that transforms components of a vector given in r.l.u.

Translated from MATLAB ``herbert_core/utilities/scientific/ubmatrix.m``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from .bmatrix import _infer_nargout


def ubmatrix(
    u: Any,
    v: Any,
    b: Any,
    *,
    nargout: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Calculate UB matrix that transforms components of a vector given in r.l.u.
    into the components in an orthonormal frame defined by the two vectors
    ``u`` and ``v`` (each given in r.l.u.).

    Calling forms::

        >> ub = ubmatrix(u, v, b)
        >> ub, umat = ubmatrix(u, v, b)

    Parameters
    ----------
    u, v : array_like
        Two 3-element vectors expressed in r.l.u.
    b : np.ndarray
        B-matrix of Busing and Levy (as calculated by :func:`bmatrix`).
    nargout : int, optional
        Number of outputs to return (1 or 2). Automatically inferred when
        unpacking in Python.

    Returns
    -------
    ub : np.ndarray
        UB matrix (3x3).
    umat : np.ndarray, optional
        U matrix (3x3).

    Raises
    ------
    ValueError
        ``HORACE:ubmatrix:invalid_argument`` if ``u`` and ``v`` are parallel or invalid.

    Notes
    -----
    The orthonormal frame defined by vectors ``u`` and ``v`` is:
      - ``e1`` parallel to ``u``
      - ``e2`` in the plane of ``u`` and ``v``, with a +ve component along ``v``
      - ``e3`` perpendicular to ``u`` and ``v``

    Use the matrix ``ub`` to convert components of a vector as follows::

        Vuv[i] = UB[i, j] * Vrlu[j]

    Also::

        Vuv[i] = U[i, j] * Vcryst[j]   # NOTE: inv(U) == U.T
    """
    try:
        u_vec = np.asarray(u, dtype=float).reshape(3)
        v_vec = np.asarray(v, dtype=float).reshape(3)
        b_mat = np.asarray(b, dtype=float)
        if b_mat.shape != (3, 3):
            raise ValueError("B matrix must be 3x3")

        uc = b_mat @ u_vec
        vc = b_mat @ v_vec

        norm_uc = np.linalg.norm(uc)
        if norm_uc == 0.0 or not np.isfinite(norm_uc):
            raise ValueError("Zero norm u vector")
        e1 = uc / norm_uc

        cross_uv = np.cross(uc, vc)
        norm_cross = np.linalg.norm(cross_uv)
        if norm_cross == 0.0 or not np.isfinite(norm_cross):
            raise ValueError("Parallel u and v vectors")
        e3 = cross_uv / norm_cross
        e2 = np.cross(e3, e1)

        e_mat = np.vstack((e1, e2, e3))
        det_e = np.linalg.det(e_mat)
        if det_e == 0.0 or not np.isfinite(det_e):
            raise ValueError("Singular orientation matrix")
        umat = e_mat / det_e
        ub = umat @ b_mat
        if not (np.all(np.isfinite(ub)) and np.all(np.isfinite(umat))):
            raise ValueError("Non-finite UB matrix")
    except Exception as exc:
        raise ValueError(
            f"HORACE:ubmatrix:invalid_argument: Problem calculating UB matrix. "
            f"Check u ({u}) and v ({v}) are not parallel"
        ) from exc

    nout = _infer_nargout(default=1, explicit_nargout=nargout)
    if nout == 1:
        return ub
    return ub, umat

