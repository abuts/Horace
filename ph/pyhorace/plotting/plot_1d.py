import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple
from ..dataset.ix_dataset_1d import IXDataset1D
from ..sqw.dnd import D1d

def _get_x_centers(x: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Helper to convert bin edges to bin centers if necessary."""
    if len(x) == len(signal) + 1:
        return 0.5 * (x[:-1] + x[1:])
    return x

def _prepare_data(dataset: Union[IXDataset1D, D1d]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts x, y, and y_err from a 1D dataset."""
    if isinstance(dataset, D1d):
        # D1d stores axes in p[0]
        x = dataset.p[0]
    else:
        x = dataset.x
        
    y = dataset.signal
    y_err = np.sqrt(dataset.error)
    x_c = _get_x_centers(x, y)
    
    return x_c, y, y_err

def dd(dataset: Union[IXDataset1D, D1d], ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Draw Data (dd): Scatter plot with error bars.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x, y, y_err = _prepare_data(dataset)
    
    # Default styling mimicking GENIE/Horace
    kwargs.setdefault('fmt', 'o')
    kwargs.setdefault('markersize', 4)
    kwargs.setdefault('capsize', 2)
    kwargs.setdefault('color', 'black')
    
    ax.errorbar(x, y, yerr=y_err, **kwargs)
    
    _apply_labels(ax, dataset)
    return ax

def dl(dataset: Union[IXDataset1D, D1d], ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Draw Line (dl): Continuous line plot (usually for models/fits).
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x, y, _ = _prepare_data(dataset)
    
    kwargs.setdefault('color', 'red')
    kwargs.setdefault('linewidth', 1.5)
    
    ax.plot(x, y, **kwargs)
    
    _apply_labels(ax, dataset)
    return ax

def dp(dataset: Union[IXDataset1D, D1d], ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Draw Points (dp): Scatter plot without error bars.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    x, y, _ = _prepare_data(dataset)
    
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('linestyle', 'None')
    kwargs.setdefault('markersize', 4)
    kwargs.setdefault('color', 'black')
    
    ax.plot(x, y, **kwargs)
    
    _apply_labels(ax, dataset)
    return ax

def _apply_labels(ax: plt.Axes, dataset: Union[IXDataset1D, D1d]):
    """Applies title and axis labels from dataset metadata."""
    if hasattr(dataset, 'title') and dataset.title:
        ax.set_title(dataset.title)
    if hasattr(dataset, 'x_label') and dataset.x_label:
        ax.set_xlabel(dataset.x_label)
    if hasattr(dataset, 'y_label') and dataset.y_label:
        ax.set_ylabel(dataset.y_label)
