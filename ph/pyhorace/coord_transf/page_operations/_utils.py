"""
Utility functions for SQW and PixelData page operations.

Ported from:
- ``horace_core/utilities/split_vector_fixed_sum.m``
- ``horace_core/utilities/split_vector_max_sum.m``
- ``horace_core/utilities/compute_bin_data.m``
- ``horace_core/utilities/private/compute_bin_data_matlab_.m``
- ``horace_core/utilities/normalize_signal.m``
- ``horace_core/utilities/average_bin_data.m``
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


def split_vector_fixed_sum(
    vector: Union[np.ndarray, List[int], List[float]],
    chunk_sum: Union[int, float],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Split a vector of non-negative integers into sub-vectors whose sums do not
    exceed ``chunk_sum`` (splitting individual elements across chunks if needed).

    Parameters
    ----------
    vector : array_like
        1D or ND array of non-negative integers.
    chunk_sum : int or float
        Maximum sum allowed per chunk (must be a positive scalar).

    Returns
    -------
    chunks : list of np.ndarray
        List of 1D arrays representing the split chunks.
    idxs : np.ndarray
        ``(2, n_chunks)`` array of 1-based inclusive start and end indices into
        ``vector`` for each chunk.
    """
    vec = np.asarray(vector).ravel(order="F")
    if vec.ndim != 1 or not np.issubdtype(vec.dtype, np.number):
        raise ValueError("HORACE:utilities:invalid_argument: vector must be numeric.")
    if np.any(vec < 0):
        raise ValueError(
            "HORACE:utilities:invalid_argument: vector elements must be non-negative."
        )
    if np.ndim(chunk_sum) != 0 or chunk_sum <= 0:
        raise ValueError(
            "HORACE:utilities:invalid_argument: chunk_sum must be a positive scalar."
        )

    if vec.size == 0:
        return [], np.zeros((2, 0), dtype=np.int64)

    chunk_sum_val = int(chunk_sum)
    vec_work = vec.astype(np.int64, copy=True)
    cum_sum = np.cumsum(vec_work)
    tot_sum = int(cum_sum[-1])

    if tot_sum == 0:
        return [vec_work.copy()], np.array([[1], [vec_work.size]], dtype=np.int64)

    num_chunks = int(np.ceil(tot_sum / chunk_sum_val))
    chunks: List[np.ndarray] = []
    idxs = np.ones((2, num_chunks), dtype=np.int64)

    start_idx = 0  # 0-based index into vec_work
    for i in range(num_chunks):
        target_cum = (i + 1) * chunk_sum_val
        ge_mask = cum_sum[start_idx:] >= target_cum
        if np.any(ge_mask):
            rel_end = int(np.argmax(ge_mask))
            end_idx = start_idx + rel_end
            if i == num_chunks - 1:
                end_idx = vec_work.size - 1
        else:
            end_idx = vec_work.size - 1

        sub_vec = vec_work[start_idx : end_idx + 1].copy()
        sub_sum = int(np.sum(sub_vec))
        if sub_sum > chunk_sum_val:
            overflow = sub_sum - chunk_sum_val
            sub_vec[-1] -= overflow
            vec_work[end_idx] = overflow
            next_start_idx = end_idx
        else:
            next_start_idx = end_idx + 1
            if i == num_chunks - 1 and end_idx < vec_work.size - 1:
                sub_vec = vec_work[start_idx:].copy()
                end_idx = vec_work.size - 1
                next_start_idx = vec_work.size

        chunks.append(sub_vec)
        idxs[0, i] = start_idx + 1
        idxs[1, i] = end_idx + 1
        start_idx = next_start_idx

    return chunks, idxs


def split_vector_max_sum(
    vector: Union[np.ndarray, List[int], List[float]],
    max_sum: Union[int, float],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Split a vector of non-negative numbers into contiguous sub-vectors without
    splitting individual elements, such that each sub-vector has sum <= ``max_sum``
    (unless a single element exceeds ``max_sum``, in which case it forms its own chunk).

    Parameters
    ----------
    vector : array_like
        1D or ND array of non-negative values.
    max_sum : int or float
        Target maximum sum per chunk (must be a positive scalar).

    Returns
    -------
    chunks : list of np.ndarray
        List of 1D arrays representing the contiguous chunks.
    idxs : np.ndarray
        ``(2, n_chunks)`` array of 1-based inclusive start and end indices into
        ``vector`` for each chunk.
    """
    vec = np.asarray(vector).ravel(order="F")
    if vec.ndim != 1 or not np.issubdtype(vec.dtype, np.number):
        raise ValueError("HORACE:utilities:invalid_argument: vector must be numeric.")
    if np.any(vec < 0):
        raise ValueError(
            "HORACE:utilities:invalid_argument: vector elements must be non-negative."
        )
    if np.ndim(max_sum) != 0 or max_sum <= 0:
        raise ValueError(
            "HORACE:utilities:invalid_argument: max_sum must be a positive scalar."
        )

    if vec.size == 0:
        return [], np.zeros((2, 0), dtype=np.int64)

    n = vec.size
    cum_sum = np.cumsum(vec, dtype=np.float64)

    chunks: List[np.ndarray] = []
    idx_list: List[Tuple[int, int]] = []

    start_idx = 0  # 0-based
    base_sum = 0.0
    while start_idx < n:
        target = base_sum + float(max_sum)
        rel_pos = int(np.searchsorted(cum_sum[start_idx:], target, side="right"))
        if rel_pos == 0:
            end_idx = start_idx
        else:
            end_idx = start_idx + rel_pos - 1

        chunks.append(vec[start_idx : end_idx + 1].copy())
        idx_list.append((start_idx + 1, end_idx + 1))
        base_sum = float(cum_sum[end_idx])
        start_idx = end_idx + 1

    idxs = np.array(idx_list, dtype=np.int64).T
    return chunks, idxs


def split_vector_max_sum_or_numel(
    numeric_vector: Union[np.ndarray, List[int], List[float]],
    max_chunk_sum: Union[int, float],
    max_numel: Union[int, float],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Split a vector of non-negative numbers into contiguous sub-vectors such that
    each sub-vector has sum <= ``max_chunk_sum`` (unless a single element exceeds
    ``max_chunk_sum``) AND length <= ``max_numel``.

    Ported from ``horace_core/utilities/split_vector_max_sum_or_numel.m``.

    Parameters
    ----------
    numeric_vector : array_like
        1D or ND array of non-negative values.
    max_chunk_sum : int or float
        Target maximum sum per chunk (must be a positive scalar).
    max_numel : int or float
        Maximum number of elements per chunk (must be a positive scalar).

    Returns
    -------
    chunks : list of np.ndarray
        List of 1D arrays representing the contiguous chunks.
    idxs : np.ndarray
        ``(2, n_chunks)`` array of 1-based inclusive start and end indices into
        ``numeric_vector`` for each chunk.
    """
    vec = np.asarray(numeric_vector).ravel(order="F")
    if vec.ndim != 1 or not np.issubdtype(vec.dtype, np.number):
        raise ValueError("HORACE:utilities:invalid_argument: numeric_vector must be numeric.")
    if np.any(vec < 0):
        raise ValueError(
            "HORACE:utilities:invalid_argument: numeric_vector elements must be non-negative."
        )
    if np.ndim(max_chunk_sum) != 0 or max_chunk_sum <= 0:
        raise ValueError(
            "HORACE:utilities:invalid_argument: max_chunk_sum must be a positive scalar."
        )
    if np.ndim(max_numel) != 0 or max_numel <= 0:
        raise ValueError(
            "HORACE:utilities:invalid_argument: max_numel must be a positive scalar."
        )

    if vec.size == 0:
        return [], np.zeros((2, 0), dtype=np.int64)

    n = vec.size
    cum_sum = np.cumsum(vec, dtype=np.float64)
    max_numel_int = int(max_numel)

    chunks: List[np.ndarray] = []
    idx_list: List[Tuple[int, int]] = []

    start_idx = 0  # 0-based
    prev_sum = 0.0
    while start_idx < n:
        target = prev_sum + float(max_chunk_sum)
        end_sum_count = int(np.searchsorted(cum_sum[start_idx:], target, side="right"))
        end_num_count = min(max_numel_int, n - start_idx)
        rel_count = min(end_sum_count, end_num_count)
        if rel_count == 0:
            end_idx = start_idx
        else:
            end_idx = start_idx + rel_count - 1

        chunks.append(vec[start_idx : end_idx + 1].copy())
        idx_list.append((start_idx + 1, end_idx + 1))
        prev_sum = float(cum_sum[end_idx])
        start_idx = end_idx + 1

    idxs = np.array(idx_list, dtype=np.int64).T
    return chunks, idxs



def normalize_signal(
    s: np.ndarray,
    e: Optional[np.ndarray],
    npix: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Normalize accumulated bin signal and variance sums by pixel counts ``npix``.

    Parameters
    ----------
    s : np.ndarray
        Accumulated signal sum per bin.
    e : np.ndarray or None
        Accumulated variance sum per bin, or None.
    npix : np.ndarray
        Number of pixels per bin.

    Returns
    -------
    s_norm : np.ndarray
        Mean signal per bin (0 where ``npix == 0``).
    e_norm : np.ndarray or None
        Normalized variance per bin (``e / npix^2``, 0 where ``npix == 0``).
    """
    npix_arr = np.asarray(npix, dtype=np.float64)
    s_arr = np.asarray(s, dtype=np.float64)
    no_pix = npix_arr == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        s_norm = np.where(no_pix, 0.0, s_arr / npix_arr)
        if e is not None and np.asarray(e).size > 0:
            e_arr = np.asarray(e, dtype=np.float64)
            e_norm = np.where(no_pix, 0.0, e_arr / (npix_arr * npix_arr))
        else:
            e_norm = None

    return s_norm, e_norm


def compute_bin_data(
    npix: np.ndarray,
    signal: np.ndarray,
    variance: Optional[np.ndarray] = None,
    no_normalization: bool = False,
    calc_variance: bool = True,
    calc_signal_msd: bool = False,
) -> Union[
    Tuple[np.ndarray, Optional[np.ndarray]],
    Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]],
]:
    """
    Compute bin mean signal and variance from per-pixel signal and variance arrays.

    Ported from ``horace_core/utilities/compute_bin_data.m`` and
    ``horace_core/utilities/private/compute_bin_data_matlab_.m``.

    Parameters
    ----------
    npix : np.ndarray
        Number of pixels contributing to each bin.
    signal : np.ndarray
        1D array of pixel signal values of length ``sum(npix)``.
    variance : np.ndarray or None, optional
        1D array of pixel variance values of length ``sum(npix)``.
    no_normalization : bool, default False
        If True, return bin sums without dividing by ``npix`` or ``npix^2``.
    calc_variance : bool, default True
        Whether to compute variance.
    calc_signal_msd : bool, default False
        If True, also return ``signal_msd`` as the third return value.

    Returns
    -------
    mean_signal : np.ndarray
        Average (or summed) signal for each bin, shaped like ``npix``.
    mean_variance : np.ndarray or None
        Average (or summed) variance for each bin, shaped like ``npix``, or None.
    signal_msd : np.ndarray or None, optional
        Returned only when ``calc_signal_msd=True``.
    """
    npix_arr = np.asarray(npix)
    s_in = np.asarray(signal, dtype=np.float64).ravel(order="F")
    tot_npix = int(np.sum(npix_arr))

    if s_in.size == 0 or npix_arr.size == 0:
        empty_s = np.zeros(npix_arr.shape, dtype=np.float64, order="F")
        empty_v = (
            np.zeros(npix_arr.shape, dtype=np.float64, order="F")
            if calc_variance
            else None
        )
        if calc_signal_msd:
            return empty_s, empty_v, np.empty(0, dtype=np.float64)
        return empty_s, empty_v

    if s_in.size == 1 and tot_npix != 1:
        s_in = np.full(tot_npix, s_in[0], dtype=np.float64)
    elif tot_npix != s_in.size:
        raise ValueError(
            f"HORACE:utilities:invalid_argument: number of elements in signal array "
            f"({s_in.size}) must equal total npix ({tot_npix})"
        )

    has_var = variance is not None and np.asarray(variance).size > 0
    if has_var:
        v_in = np.asarray(variance, dtype=np.float64).ravel(order="F")
        if v_in.size == 1 and tot_npix != 1:
            v_in = np.full(tot_npix, v_in[0], dtype=np.float64)
        elif v_in.size != s_in.size:
            raise ValueError(
                f"HORACE:utilities:invalid_argument: number of elements in signal array "
                f"({s_in.size}) must equal number of elements in variance array ({v_in.size})"
            )
    else:
        v_in = None

    normalize = not bool(no_normalization)
    npix_flat = npix_arr.ravel(order="F").astype(np.int64)
    no_pix = npix_flat == 0
    non_empty = ~no_pix

    img_signal_sum = np.zeros(npix_flat.size, dtype=np.float64)
    if np.any(non_empty):
        npix_nz = npix_flat[non_empty]
        starts = np.concatenate(([0], np.cumsum(npix_nz)[:-1]))
        img_signal_sum[non_empty] = np.add.reduceat(s_in, starts)

    with np.errstate(divide="ignore", invalid="ignore"):
        if normalize:
            mean_signal_flat = np.where(no_pix, 0.0, img_signal_sum / npix_flat)
        else:
            mean_signal_flat = np.where(no_pix, 0.0, img_signal_sum)

    if not calc_variance:
        mean_signal = mean_signal_flat.reshape(npix_arr.shape, order="F")
        if calc_signal_msd:
            return mean_signal, None, None
        return mean_signal, None

    signal_msd: Optional[np.ndarray] = None
    if not has_var:
        with np.errstate(divide="ignore", invalid="ignore"):
            msi = np.where(no_pix, 0.0, img_signal_sum / npix_flat)
        rep_msi = np.repeat(msi, npix_flat)
        signal_msd = (s_in - rep_msi) ** 2
    else:
        signal_msd = v_in

    if signal_msd is not None and signal_msd.size > 0:
        img_variance_sum = np.zeros(npix_flat.size, dtype=np.float64)
        if np.any(non_empty):
            npix_nz = npix_flat[non_empty]
            starts = np.concatenate(([0], np.cumsum(npix_nz)[:-1]))
            img_variance_sum[non_empty] = np.add.reduceat(signal_msd, starts)

        with np.errstate(divide="ignore", invalid="ignore"):
            if normalize:
                mean_var_flat = np.where(
                    no_pix, 0.0, img_variance_sum / (npix_flat.astype(np.float64) ** 2)
                )
            else:
                mean_var_flat = np.where(no_pix, 0.0, img_variance_sum)
        mean_variance = mean_var_flat.reshape(npix_arr.shape, order="F")
    else:
        mean_variance = np.zeros(npix_arr.shape, dtype=np.float64, order="F")

    mean_signal = mean_signal_flat.reshape(npix_arr.shape, order="F")
    if calc_signal_msd:
        return mean_signal, mean_variance, signal_msd
    return mean_signal, mean_variance


def average_bin_data(
    npix: np.ndarray,
    vals: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
    return_variance: bool = False,
    return_devsqr: bool = False,
) -> Union[
    Union[np.ndarray, List[np.ndarray]],
    Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]],
    Tuple[
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
        Union[np.ndarray, List[np.ndarray]],
    ],
]:
    """
    Given ``npix`` array, average one or more arrays of ``npixtot`` values to
    give one value per bin.

    Ported from ``horace_core/utilities/average_bin_data.m``.

    Parameters
    ----------
    npix : np.ndarray
        ``npix`` array of bins to average data over.
    vals : np.ndarray or sequence of np.ndarray
        Array, or list/tuple of arrays, each with ``sum(npix)`` elements, to be
        averaged over the bins.
    return_variance : bool, default False
        If True (or ``return_devsqr`` is True), also compute and return ``vals_var``.
    return_devsqr : bool, default False
        If True, also compute and return ``vals_devsqr`` (square deviations for all pixels).

    Returns
    -------
    vals_av : np.ndarray or list of np.ndarray
        Array (or list of arrays) containing the averaged values with shape ``npix.shape``
        (0 where ``npix == 0``).
    vals_var : np.ndarray or list of np.ndarray, optional
        Variance of the values in each bin with shape ``npix.shape``.
    vals_devsqr : np.ndarray or list of np.ndarray, optional
        Square deviation for all of the pixel values.
    """
    is_list = isinstance(vals, (list, tuple))
    vals_list = list(vals) if is_list else [vals]

    npix_arr = np.asarray(npix)
    npix_flat = npix_arr.ravel(order="F").astype(np.int64)
    nbin = npix_flat.size
    npixtot = int(np.sum(npix_flat))

    for v in vals_list:
        if np.asarray(v).size != npixtot:
            raise ValueError(
                "HORACE:average_bin_data:invalid_argument: "
                "Invalid number of elements in input array(s)"
            )

    stddev_request = return_variance or return_devsqr
    ind = np.repeat(np.arange(nbin, dtype=np.int64), npix_flat)
    nopix = npix_flat == 0
    npix_f = npix_flat.astype(np.float64)

    vals_av_out: List[np.ndarray] = []
    vals_var_out: List[np.ndarray] = []
    vals_devsqr_out: List[np.ndarray] = []

    for v in vals_list:
        v_flat = np.asarray(v, dtype=np.float64).ravel(order="F")
        if nbin == 0 or npixtot == 0:
            av_flat = np.zeros(nbin, dtype=np.float64)
        else:
            av_sum = np.bincount(ind, weights=v_flat, minlength=nbin)
            with np.errstate(divide="ignore", invalid="ignore"):
                av_flat = av_sum / npix_f
            av_flat[nopix] = 0.0
        vals_av_out.append(av_flat.reshape(npix_arr.shape, order="F"))

        if stddev_request:
            rep_av = np.repeat(av_flat, npix_flat)
            devsqr = (v_flat - rep_av) ** 2
            vals_devsqr_out.append(devsqr)
            if nbin == 0 or npixtot == 0:
                var_flat = np.zeros(nbin, dtype=np.float64)
            else:
                var_sum = np.bincount(ind, weights=devsqr, minlength=nbin)
                with np.errstate(divide="ignore", invalid="ignore"):
                    var_flat = var_sum / (npix_f**2)
                var_flat[nopix] = 0.0
            vals_var_out.append(var_flat.reshape(npix_arr.shape, order="F"))

    if not is_list:
        if return_devsqr:
            return vals_av_out[0], vals_var_out[0], vals_devsqr_out[0]
        if return_variance:
            return vals_av_out[0], vals_var_out[0]
        return vals_av_out[0]

    if return_devsqr:
        return vals_av_out, vals_var_out, vals_devsqr_out
    if return_variance:
        return vals_av_out, vals_var_out
    return vals_av_out


def split_data_blocks(
    pos: Union[np.ndarray, List[int], List[float]],
    block_sizes: Union[np.ndarray, List[int], List[float]],
    buf_size: Union[int, float],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Divide an array of block sizes and positions into chunks such that the sum
    of sizes of each sub-chunk ensures ``sum(chunk_i) <= buf_size`` while
    splitting blocks larger than ``buf_size`` and adjusting their positions.

    Ported from ``horace_core/utilities/split_data_blocks.m``.

    Parameters
    ----------
    pos : array_like
        1-based (or 0-based) starting positions of data blocks.
    block_sizes : array_like
        Sizes of data blocks.
    buf_size : int or float
        Maximum sum of ``block_sizes`` in a single chunk.

    Returns
    -------
    chunks : list of np.ndarray
        1D integer arrays of block sizes in each chunk.
    positions : list of np.ndarray
        1D integer arrays of block starting positions in each chunk.
    """
    pos_arr = np.asarray(pos).ravel(order="F")
    sizes_arr = np.asarray(block_sizes).ravel(order="F")
    if pos_arr.size != sizes_arr.size:
        raise ValueError(
            f"HORACE:utilities:invalid_argument: Block positions array size ({pos_arr.size}) "
            f"has to be equal to block_sizes array size ({sizes_arr.size})"
        )
    bsize = int(buf_size)
    if bsize <= 0:
        raise ValueError(
            "HORACE:utilities:invalid_argument: The second argument must be non-negative numeric scalar."
        )
    if pos_arr.size == 0:
        return [], []

    pos_int = pos_arr.astype(np.int64)
    sizes_int = sizes_arr.astype(np.int64)

    chunks, idx = split_vector_fixed_sum(sizes_int, bsize)
    n_chunks = len(chunks)
    positions: List[np.ndarray] = []

    prev_idx = -1
    prev_shift = 0
    for i in range(n_chunks):
        ch_ind = idx[:, i]
        cur_start = int(ch_ind[0]) - 1  # 0-based
        cur_end = int(ch_ind[1]) - 1  # 0-based inclusive
        pos_part = pos_int[cur_start : cur_end + 1].copy()
        if prev_idx == cur_start:
            pos_part[0] += prev_shift
        prev_idx = cur_end
        if prev_idx == cur_start:
            prev_shift = int(chunks[i][-1]) + prev_shift
        else:
            prev_shift = int(chunks[i][-1])
        positions.append(pos_part)

    return chunks, positions


def get_ind_from_ranges(
    ranges_beg: Union[np.ndarray, List[int]],
    block_sizes: Union[np.ndarray, List[int]],
) -> np.ndarray:
    """
    Generate flat array of indices interpolated between ``ranges_beg`` and
    ``ranges_beg + block_sizes - 1``.

    Ported from ``horace_core/utilities/get_ind_from_ranges.m``.

    Parameters
    ----------
    ranges_beg : array_like
        Starting indices of blocks.
    block_sizes : array_like
        Sizes of blocks (same number of elements as ``ranges_beg``).

    Returns
    -------
    ind : np.ndarray
        1D int64 array containing expanded contiguous ranges for each block.
    """
    beg_arr = np.asarray(ranges_beg, dtype=np.int64).ravel(order="F")
    sz_arr = np.asarray(block_sizes, dtype=np.int64).ravel(order="F")
    if beg_arr.size != sz_arr.size:
        raise ValueError(
            f"HORACE:utilities:invalid_argument: Numer of elements in ranges_beg ({beg_arr.size}) "
            f"not equal to number of elements in block_sizes ({sz_arr.size})"
        )
    valid = sz_arr > 0
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    beg_v = beg_arr[valid]
    sz_v = sz_arr[valid]
    tot = int(np.sum(sz_v))
    # Vectorized expansion of [beg_v[k], ..., beg_v[k] + sz_v[k] - 1]
    starts = np.cumsum(sz_v) - sz_v
    steps = np.ones(tot, dtype=np.int64)
    steps[0] = beg_v[0]
    if beg_v.size > 1:
        prev_ends = beg_v[:-1] + sz_v[:-1] - 1
        steps[starts[1:]] = beg_v[1:] - prev_ends
    return np.cumsum(steps)


def minmax_ranges(range1: np.ndarray, range2: np.ndarray) -> np.ndarray:
    """Compute bounding box (2xN min/max range) enclosing ``range1`` and ``range2``."""
    r1 = np.asarray(range1, dtype=np.float64)
    r2 = np.asarray(range2, dtype=np.float64)
    return np.vstack([np.minimum(r1[0, :], r2[0, :]), np.maximum(r1[1, :], r2[1, :])])


def parse_char_options(
    args: Union[Tuple, List], valid_options: List[str]
) -> Tuple:
    """Parse character option flags (e.g. ``'-keep_runid'``) from ``args``.

    Ported from ``herbert_core/utilities/general/parse_char_options.m``.
    Returns ``(ok, mess, *flag_bools, remaining_args)``.
    """
    flags = [False] * len(valid_options)
    rem: List = []
    opts_lower = [opt.lower() for opt in valid_options]

    for arg in args:
        if isinstance(arg, str) and arg.startswith("-"):
            a_low = arg.lower()
            exact = [i for i, opt in enumerate(opts_lower) if opt == a_low]
            if len(exact) == 1:
                flags[exact[0]] = True
                continue
            prefix = [i for i, opt in enumerate(opts_lower) if opt.startswith(a_low)]
            if len(prefix) == 1:
                flags[prefix[0]] = True
                continue
            return (False, f"Unrecognised or ambiguous option: {arg}", *flags, rem)
        rem.append(arg)

    return (True, "", *flags, rem)


