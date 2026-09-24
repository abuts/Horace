import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from typing import Optional, Union, Tuple, Any
from ..dataset.ix_dataset_2d import IXDataset2D
from ..sqw.dnd import D2d

def _prepare_data_2d(dataset: Union[IXDataset2D, D2d]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts X and Y bin edges, and Z signal array from a 2D dataset."""
    if isinstance(dataset, D2d):
        x = dataset.p[0]
        y = dataset.p[1]
    else:
        x = dataset.x
        y = dataset.y
        
    z = dataset.signal
    
    # pcolormesh expects bin edges. If we have centers, we need to approximate edges.
    if len(x) == z.shape[0]:
        # Approximate edges by assuming uniform spacing based on first two points
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        x = np.append(x - dx/2, x[-1] + dx/2)
        
    if len(y) == z.shape[1]:
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        y = np.append(y - dy/2, y[-1] + dy/2)
        
    return x, y, z

def da(dataset: Union[IXDataset2D, D2d], ax: Optional[plt.Axes] = None, 
       log_scale: bool = False, vmin: Optional[float] = None, vmax: Optional[float] = None, 
       cmap: str = 'viridis', **kwargs) -> Tuple[plt.Axes, Any]:
    """
    Draw Area (da): 2D color map using pcolormesh.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x, y, z = _prepare_data_2d(dataset)
    
    # Handle NaNs and masking
    z_masked = np.ma.masked_invalid(z)
    
    if log_scale:
        # Avoid log of negative/zero
        z_masked = np.ma.masked_less_equal(z_masked, 0)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        
    # Note: pcolormesh expects (X, Y) where X, Y broadcast to shape (N+1, M+1)
    # and Z has shape (M, N). Horace/NumPy matrices are typically (X, Y) ordered
    # so we need to transpose Z for Matplotlib which expects (Y, X) ordering.
    im = ax.pcolormesh(x, y, z_masked.T, norm=norm, cmap=cmap, shading='flat', **kwargs)
    
    # Add colorbar if this is a new figure
    if ax.figure:
        ax.figure.colorbar(im, ax=ax, label=getattr(dataset, 'z_label', 'Intensity'))
        
    _apply_labels_2d(ax, dataset)
    return ax, im

def dc(dataset: Union[IXDataset2D, D2d], ax: Optional[plt.Axes] = None, 
       levels: int = 10, cmap: str = 'viridis', **kwargs) -> Tuple[plt.Axes, Any]:
    """
    Draw Contour (dc): 2D contour plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x, y, z = _prepare_data_2d(dataset)
    
    # Contours require bin centers, not edges
    x_c = 0.5 * (x[:-1] + x[1:])
    y_c = 0.5 * (y[:-1] + y[1:])
    
    X, Y = np.meshgrid(x_c, y_c, indexing='ij')
    
    cs = ax.contour(X, Y, z, levels=levels, cmap=cmap, **kwargs)
    
    _apply_labels_2d(ax, dataset)
    return ax, cs

def _apply_labels_2d(ax: plt.Axes, dataset: Union[IXDataset2D, D2d]):
    """Applies title and axis labels from dataset metadata."""
    if hasattr(dataset, 'title') and dataset.title:
        ax.set_title(dataset.title)
    if hasattr(dataset, 'x_label') and dataset.x_label:
        ax.set_xlabel(dataset.x_label)
    if hasattr(dataset, 'y_label') and dataset.y_label:
        ax.set_ylabel(dataset.y_label)
