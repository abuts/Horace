import numpy as np
from typing import Callable, List, Optional, Tuple, Dict, Any
from scipy.optimize import least_squares
from ..dataset.base import IXDataset
from ..dataset.ix_dataset_1d import IXDataset1D

class Multifit:
    """
    Multi-dataset fitting engine.
    Allows fitting multiple IXDataset objects simultaneously using Levenberg-Marquardt.
    """
    def __init__(self, datasets: List[IXDataset]):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets = datasets
        self.funcs: List[Callable] = []
        self.initial_params: np.ndarray = np.array([])
        self.free_params: np.ndarray = np.array([], dtype=bool)
        
        # Fit results
        self.popt: Optional[np.ndarray] = None
        self.pcov: Optional[np.ndarray] = None
        self.chisq: Optional[float] = None
        
    def set_fun(self, func: Callable) -> 'Multifit':
        """Sets the model function for all datasets."""
        self.funcs = [func] * len(self.datasets)
        return self
        
    def set_pin(self, p: List[float]) -> 'Multifit':
        """Sets initial parameters."""
        self.initial_params = np.asarray(p, dtype=float)
        # By default, all parameters are free
        if len(self.free_params) != len(self.initial_params):
            self.free_params = np.ones(len(self.initial_params), dtype=bool)
        return self
        
    def set_free(self, free: List[bool]) -> 'Multifit':
        """Sets which parameters are free (1/True) or fixed (0/False)."""
        self.free_params = np.asarray(free, dtype=bool)
        return self

    def _residuals(self, p_free: np.ndarray) -> np.ndarray:
        """Computes concatenated, error-weighted residuals for all datasets."""
        # Reconstruct full parameter array
        p_full = self.initial_params.copy()
        p_full[self.free_params] = p_free
        
        all_res = []
        for i, (ds, func) in enumerate(zip(self.datasets, self.funcs)):
            if not isinstance(ds, IXDataset1D):
                raise NotImplementedError("Multifit currently only supports IXDataset1D")
            
            # Extract bin centers if x is bin edges
            x = ds.x
            if len(x) == len(ds.signal) + 1:
                x = 0.5 * (x[:-1] + x[1:])
                
            y_data = ds.signal
            y_err = np.sqrt(ds.error)
            
            # Avoid division by zero in weights
            weights = np.where(y_err > 0, 1.0 / y_err, 0.0)
            
            # Evaluate model
            y_model = func(x, *p_full)
            
            # Valid mask (e.g. ignoring NaNs or zero errors)
            mask = (weights > 0) & np.isfinite(y_data) & np.isfinite(y_model)
            
            res = (y_data[mask] - y_model[mask]) * weights[mask]
            all_res.append(res)
            
        return np.concatenate(all_res)

    def fit(self) -> Tuple[List[IXDataset], Dict[str, Any]]:
        """
        Executes the fit using scipy.optimize.least_squares (Levenberg-Marquardt).
        Returns:
            List of evaluated model IXDatasets
            Dictionary of fit results
        """
        if not self.funcs or len(self.initial_params) == 0:
            raise ValueError("Must set model function and initial parameters before fitting.")
            
        p0_free = self.initial_params[self.free_params]
        
        result = least_squares(
            self._residuals,
            p0_free,
            method='lm'
        )
        
        # Reconstruct full optimal parameters
        self.popt = self.initial_params.copy()
        self.popt[self.free_params] = result.x
        
        # Calculate covariance matrix
        J = result.jac
        self.pcov = np.zeros((len(self.popt), len(self.popt)))
        if J is not None and J.size > 0:
            cov_free = np.linalg.inv(J.T @ J)
            # Scatter free covariance into full covariance matrix
            free_idx = np.where(self.free_params)[0]
            for i, r_idx in enumerate(free_idx):
                for j, c_idx in enumerate(free_idx):
                    self.pcov[r_idx, c_idx] = cov_free[i, j]
                    
        # Calculate reduced chi-squared
        dof = len(result.fun) - len(p0_free)
        self.chisq = np.sum(result.fun**2) / dof if dof > 0 else float('inf')
        
        # Generate model datasets
        model_datasets = []
        for ds, func in zip(self.datasets, self.funcs):
            # Same logic to get x
            x = ds.x
            if len(x) == len(ds.signal) + 1:
                x = 0.5 * (x[:-1] + x[1:])
                
            y_model = func(x, *self.popt)
            
            mds = IXDataset1D(
                signal=y_model,
                error=np.zeros_like(y_model),
                x=ds.x,
                title=f"Fit to {ds.title}"
            )
            model_datasets.append(mds)
            
        res_dict = {
            'popt': self.popt,
            'pcov': self.pcov,
            'chisq': self.chisq,
            'success': result.success,
            'message': result.message
        }
        
        return model_datasets, res_dict
