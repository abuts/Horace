import numpy as np
from numba import njit
from typing import List, Tuple, Optional, Union
from ..sqw.sqw import SQW
from ..sqw.dnd import D0d, D1d, D2d, D3d, D4d, DnDBase
from ..projection.base import ProjectionBase

@njit
def bin_pixels_kernel(proj_coords: np.ndarray, signal: np.ndarray, variance: np.ndarray,
                     p1_bins: np.ndarray, p2_bins: np.ndarray, p3_bins: np.ndarray, p4_bins: np.ndarray,
                     out_signal: np.ndarray, out_error: np.ndarray, out_npix: np.ndarray):
    """
    Numba-accelerated core binning kernel.
    proj_coords: (N, 4) projected coordinates
    signal, variance: (N,) arrays
    p1_bins, ..., p4_bins: Bin boundaries for each axis. If an axis is integrating, it has 2 elements [min, max].
                           If it's a plot axis, it has N+1 elements.
    out_signal, out_error: Output arrays of shape (N1, N2, N3, N4) where Ni = len(p_bins)-1
    out_npix: Output array of uint64 shape (N1, N2, N3, N4)
    """
    n_pixels = proj_coords.shape[0]
    
    n1 = len(p1_bins) - 1
    n2 = len(p2_bins) - 1
    n3 = len(p3_bins) - 1
    n4 = len(p4_bins) - 1

    for i in range(n_pixels):
        c1 = proj_coords[i, 0]
        c2 = proj_coords[i, 1]
        c3 = proj_coords[i, 2]
        c4 = proj_coords[i, 3]

        # Quick reject if out of global bounds
        if not (p1_bins[0] <= c1 < p1_bins[-1] and
                p2_bins[0] <= c2 < p2_bins[-1] and
                p3_bins[0] <= c3 < p3_bins[-1] and
                p4_bins[0] <= c4 < p4_bins[-1]):
            continue
            
        # Find bin indices using np.searchsorted
        # searchsorted returns insertion index; for bins we want index-1
        idx1 = np.searchsorted(p1_bins, c1, side='right') - 1
        idx2 = np.searchsorted(p2_bins, c2, side='right') - 1
        idx3 = np.searchsorted(p3_bins, c3, side='right') - 1
        idx4 = np.searchsorted(p4_bins, c4, side='right') - 1
        
        # Handle edge cases exactly at the upper bound
        if idx1 == n1 and c1 == p1_bins[-1]: idx1 -= 1
        if idx2 == n2 and c2 == p2_bins[-1]: idx2 -= 1
        if idx3 == n3 and c3 == p3_bins[-1]: idx3 -= 1
        if idx4 == n4 and c4 == p4_bins[-1]: idx4 -= 1

        if 0 <= idx1 < n1 and 0 <= idx2 < n2 and 0 <= idx3 < n3 and 0 <= idx4 < n4:
            out_signal[idx1, idx2, idx3, idx4] += signal[i]
            out_error[idx1, idx2, idx3, idx4] += variance[i]
            out_npix[idx1, idx2, idx3, idx4] += 1

def _parse_binning_arg(arg: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Parses binning arguments.
    [min, max] -> Integration axis
    [min, step, max] -> Plot axis (generates arange)
    """
    arg = np.asarray(arg, dtype=float)
    if len(arg) == 2:
        return arg # Integration axis
    elif len(arg) == 3:
        # Plot axis: generate bins
        start, step, stop = arg
        # To avoid floating point issues, calculate number of bins
        nbins = int(np.round((stop - start) / step))
        # Generate exactly nbins+1 edges
        return np.linspace(start, start + nbins * step, nbins + 1)
    else:
        raise ValueError(f"Invalid binning argument: {arg}")

def cut_sqw(sqw_obj: SQW, proj: ProjectionBase, 
            p1_bin: Union[List[float], np.ndarray],
            p2_bin: Union[List[float], np.ndarray],
            p3_bin: Union[List[float], np.ndarray],
            p4_bin: Union[List[float], np.ndarray],
            nopix: bool = True) -> Union[SQW, DnDBase]:
    """
    Takes an N-dimensional cut from a 4D SQW object.
    
    Args:
        sqw_obj: Source SQW object
        proj: Projection instance
        p1_bin..p4_bin: Binning definitions. [min, max] for integration, [min, step, max] for plotting.
        nopix: If True, returns a lightweight DnD object instead of a full SQW.
    """
    if sqw_obj.pix is None:
        raise ValueError("Cannot cut an SQW object without pixel data.")
        
    p1_edges = _parse_binning_arg(p1_bin)
    p2_edges = _parse_binning_arg(p2_bin)
    p3_edges = _parse_binning_arg(p3_bin)
    p4_edges = _parse_binning_arg(p4_bin)
    
    edges = [p1_edges, p2_edges, p3_edges, p4_edges]
    
    # Determine dimensionality based on number of plot axes (length > 2)
    plot_axes = [i for i, e in enumerate(edges) if len(e) > 2]
    ndim = len(plot_axes)
    
    # Output arrays
    shape = (len(p1_edges)-1, len(p2_edges)-1, len(p3_edges)-1, len(p4_edges)-1)
    out_sig = np.zeros(shape, dtype=float)
    out_err = np.zeros(shape, dtype=float)
    out_npix = np.zeros(shape, dtype=np.uint64)
    
    # Project coordinates
    u1 = sqw_obj.pix.u1
    u2 = sqw_obj.pix.u2
    u3 = sqw_obj.pix.u3
    dE = sqw_obj.pix.dE
    sig = sqw_obj.pix.signal
    var = sqw_obj.pix.variance
    
    proj_coords = proj.project(u1, u2, u3, dE)
    
    # Run Numba binning kernel
    bin_pixels_kernel(proj_coords, sig, var, p1_edges, p2_edges, p3_edges, p4_edges,
                     out_sig, out_err, out_npix)
                     
    # Normalize by npix (Horace behavior: signal is average per bin)
    # Avoid division by zero
    mask = out_npix > 0
    out_sig[mask] = out_sig[mask] / out_npix[mask]
    out_err[mask] = out_err[mask] / (out_npix[mask] ** 2) # Variance of the mean
    
    # Squeeze out integration dimensions (length 1)
    out_sig_sq = np.squeeze(out_sig)
    out_err_sq = np.squeeze(out_err)
    out_npix_sq = np.squeeze(out_npix)
    
    # Ensure scalars become 0D arrays
    if np.isscalar(out_sig_sq): out_sig_sq = np.array([out_sig_sq])
    if np.isscalar(out_err_sq): out_err_sq = np.array([out_err_sq])
    if np.isscalar(out_npix_sq): out_npix_sq = np.array([out_npix_sq])
    
    # Collect the plot axes boundaries for the result
    plot_edges = [edges[i] for i in plot_axes]
    
    # Construct the appropriate DnD object
    if ndim == 0: dnd_cls = D0d
    elif ndim == 1: dnd_cls = D1d
    elif ndim == 2: dnd_cls = D2d
    elif ndim == 3: dnd_cls = D3d
    elif ndim == 4: dnd_cls = D4d
    else: raise ValueError(f"Invalid dimensionality: {ndim}")
    
    dnd_obj = dnd_cls(
        signal=out_sig_sq,
        error=out_err_sq,
        npix=out_npix_sq,
        p=plot_edges
    )
    
    if nopix:
        return dnd_obj
    else:
        # To retain pixels, we would need to filter sqw_obj.pix based on the bounding box.
        # This requires a sort_pixels_by_bins equivalent. For now, we return nopix=True logic.
        raise NotImplementedError("nopix=False is not yet implemented (requires pixel sorting/filtering).")
