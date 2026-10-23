"""Expand min/max box boundaries into full vertex node coordinates.

Translated from MATLAB ``herbert_core/utilities/geometry/expand_box.m`` and
``herbert_core/utilities/geometry/get_geometry.m``.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .bmatrix import _infer_nargout


def get_geometry(n_dims: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return a box structure in 2, 3, or 4 dimensions, necessary to convert
    compact ``(min_value, max_value)`` aligned box representation into the whole
    shape representation (coordinates of all nodes and arrays of coordinates of
    all edges).

    Parameters
    ----------
    n_dims : int
        Number of dimensions (2, 3, or 4).

    Returns
    -------
    nodes_ind : list of np.ndarray
        List of ``2^n_dims`` 1-based index arrays (each of length ``n_dims``),
        matching MATLAB ``ind2sub`` column-major order.
    edges_ind : np.ndarray
        ``(2, N_edges)`` array of 1-based node indices defining edges of the box.
    """
    if n_dims not in (2, 3, 4):
        raise ValueError(
            f"GET_GEOMETRY:invalid_argument: Function recognises only 2 to 4 dimensions, "
            f"Requested: {n_dims}"
        )

    shape = (2,) * n_dims
    n_nodes = 2**n_dims
    nodes_ind: List[np.ndarray] = []
    for lin_idx in range(n_nodes):
        sub_0 = np.unravel_index(lin_idx, shape, order="F")
        nodes_ind.append(np.array([s + 1 for s in sub_0], dtype=int))

    # Build edges
    ind = np.arange(1, n_nodes + 1, dtype=int).reshape(shape, order="F")
    nb_pairs = np.vstack((np.eye(n_dims, dtype=int), -np.eye(n_dims, dtype=int)))
    n_edges_alloc = n_nodes * n_dims
    edges_ind = np.zeros((2, n_edges_alloc), dtype=int)
    ic = 0
    for ii in range(n_nodes):
        sub0 = np.array(np.unravel_index(ii, shape, order="F"), dtype=int)
        for jj in range(nb_pairs.shape[0]):
            sub1 = sub0 + nb_pairs[jj]
            if np.any(sub1 < 0) or np.any(sub1 >= 2):
                continue
            n1 = ii + 1
            n2 = int(ind[tuple(sub1)])
            edges_ind[:, ic] = np.sort([n1, n2])
            ic += 1

    edges_ind = edges_ind[:, :ic]
    uni_num = edges_ind[0, :] + n_nodes * edges_ind[1, :]
    _, unii = np.unique(uni_num, return_index=True)
    edges_ind = edges_ind[:, unii]
    return nodes_ind, edges_ind


def expand_box(
    min_value: Any,
    max_value: Any,
    *,
    nargout: Optional[int] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Build array of nodes of 2D, 3D or 4D box based on its minimal and maximal
    points. Each node is a column vector with ``n_dims`` rows.

    Parameters
    ----------
    min_value : array_like
        Vector of minimal coordinate values (length 2, 3, or 4).
    max_value : array_like
        Vector of maximal coordinate values (length 2, 3, or 4).
    nargout : int, optional
        Number of outputs (1 or 2). Automatically inferred when unpacking.

    Returns
    -------
    nodes : np.ndarray
        ``(2, 4)``, ``(3, 8)``, or ``(4, 16)`` array of box vertex coordinates.
    perm : np.ndarray, optional
        ``(n_dims, n_nodes)`` array of 1-based permutations used to produce nodes.
    """
    min_arr = np.asarray(min_value, dtype=float)
    max_arr = np.asarray(max_value, dtype=float)

    if min_arr.ndim == 1:
        min_arr = min_arr.reshape(-1, 1)
    elif min_arr.shape[1] > 1:
        min_arr = min_arr.T

    if max_arr.ndim == 1:
        max_arr = max_arr.reshape(-1, 1)
    elif max_arr.shape[1] > 1:
        max_arr = max_arr.T

    if min_arr.ndim != 2 or min_arr.shape[1] != 1:
        raise ValueError(
            f"EXPAND_BOX:invalid_argument: The expansion routine accepts only vectors. "
            f"Actual size is: {min_arr.shape}"
        )
    if min_arr.shape != max_arr.shape:
        raise ValueError(
            f"EXPAND_BOX:invalid_argument: The size of vector of min values {min_arr.shape} "
            f"and max values {max_arr.shape} have to be the same"
        )
    if not np.any(min_arr < max_arr):
        raise ValueError(
            f"EXPAND_BOX:invalid_argument: all min values of the input {min_arr.ravel()} "
            f"have to be smaller than all max values of the input {max_arr.ravel()}"
        )

    ndims = min_arr.shape[0]
    if ndims not in (2, 3, 4):
        raise ValueError(
            f"EXPAND_BOX:invalid_argument: The routine accepts data in 2, 3 or 4 dimensions only, "
            f"Got: {min_arr.size}"
        )

    perm_list, _ = get_geometry(ndims)
    source = np.hstack((min_arr, max_arr))  # (ndims, 2)

    cols = []
    for sel in perm_list:
        col = np.array([source[d, sel[d] - 1] for d in range(ndims)], dtype=float)
        cols.append(col)
    nodes = np.column_stack(cols)

    nout = _infer_nargout(default=1, explicit_nargout=nargout)
    if nout == 1:
        return nodes
    perm = np.column_stack(perm_list)
    return nodes, perm

