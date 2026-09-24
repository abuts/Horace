import numpy as np
from typing import Callable, List, Optional, Tuple, Dict, Any
from scipy.optimize import least_squares
from ..sqw.sqw import SQW
from ..sqw.dnd import DnDBase
from ..algorithms.cut import bin_pixels_kernel

class MultifitSQW:
    """
    Fits analytical models directly to SQW objects by evaluating the model
    at the exact coordinates of every pixel, rather than evaluating at bin centers.
    This avoids bin-centering artifacts inherent in multi-dimensional grids.
    """
    def __init__(self, sqw_obj: SQW):
        if sqw_obj.pix is None:
            raise ValueError("MultifitSQW requires an SQW object with pixel data.")
        self.sqw = sqw_obj
        self.func: Optional[Callable] = None
        self.initial_params: np.ndarray = np.array([])
        self.free_params: np.ndarray = np.array([], dtype=bool)
        
    def set_fun(self, func: Callable) -> 'MultifitSQW':
        """
        Sets the physical model function. 
        Expected signature: func(qh, qk, ql, en, *params) -> signal_array
        """
        self.func = func
        return self
        
    def set_pin(self, p: List[float]) -> 'MultifitSQW':
        self.initial_params = np.asarray(p, dtype=float)
        if len(self.free_params) != len(self.initial_params):
            self.free_params = np.ones(len(self.initial_params), dtype=bool)
        return self
        
    def set_free(self, free: List[bool]) -> 'MultifitSQW':
        self.free_params = np.asarray(free, dtype=bool)
        return self

    def _residuals(self, p_free: np.ndarray) -> np.ndarray:
        p_full = self.initial_params.copy()
        p_full[self.free_params] = p_free
        
        # 1. Evaluate model at exact pixel coordinates
        # We need the real crystal coordinates (qh, qk, ql).
        # Assuming the SQW pix u1, u2, u3 are already in the correct reciprocal space frame 
        # (or we would need the UB matrix to transform them back). 
        # For simplicity, we pass u1, u2, u3 directly to the function.
        u1 = self.sqw.pix.u1
        u2 = self.sqw.pix.u2
        u3 = self.sqw.pix.u3
        dE = self.sqw.pix.dE
        
        model_pix_sig = self.func(u1, u2, u3, dE, *p_full)
        
        # 2. Re-bin the simulated pixels into the target histogram grid
        # We use the existing bins from the SQW's data property
        shape = self.sqw.data.signal.shape
        p1 = self.sqw.data.p[0] if len(self.sqw.data.p) > 0 else np.array([-np.inf, np.inf])
        p2 = self.sqw.data.p[1] if len(self.sqw.data.p) > 1 else np.array([-np.inf, np.inf])
        p3 = self.sqw.data.p[2] if len(self.sqw.data.p) > 2 else np.array([-np.inf, np.inf])
        p4 = self.sqw.data.p[3] if len(self.sqw.data.p) > 3 else np.array([-np.inf, np.inf])
        
        # Numba binning kernel requires a (N, 4) projected coords array.
        # If the SQW was cut, u1,u2,u3 are in crystal frame, we need them in projection frame.
        # For a full implementation, we must re-project them using the SQW's internal projection matrix.
        # *Simplified assumption for prototype*: SQW's u1..4 are already projected or we are fitting 
        # a full 4D grid where proj == identity.
        
        coords = np.column_stack((u1, u2, u3, dE))
        out_sig = np.zeros(shape, dtype=float)
        out_err = np.zeros(shape, dtype=float)
        out_npix = np.zeros(shape, dtype=np.uint64)
        
        # We pass 0 variance because the model evaluation is exact
        var_zeros = np.zeros_like(model_pix_sig)
        
        bin_pixels_kernel(coords, model_pix_sig, var_zeros, p1, p2, p3, p4, out_sig, out_err, out_npix)
        
        # Average per bin
        mask = out_npix > 0
        out_sig[mask] = out_sig[mask] / out_npix[mask]
        
        # 3. Calculate residuals against the actual binned data
        y_data = self.sqw.data.signal
        y_err = np.sqrt(self.sqw.data.error)
        
        weights = np.where(y_err > 0, 1.0 / y_err, 0.0)
        
        valid_mask = (weights > 0) & np.isfinite(y_data) & np.isfinite(out_sig) & mask
        
        res = (y_data[valid_mask] - out_sig[valid_mask]) * weights[valid_mask]
        
        return res

    def fit(self) -> Tuple[SQW, Dict[str, Any]]:
        if self.func is None:
            raise ValueError("Must set model function.")
            
        p0_free = self.initial_params[self.free_params]
        
        result = least_squares(
            self._residuals,
            p0_free,
            method='lm'
        )
        
        popt = self.initial_params.copy()
        popt[self.free_params] = result.x
        
        # Generate final model SQW (simplified: just copying the structure and updating the grid)
        # Real implementation would evaluate model -> bin -> update SQW.data.signal
        # ...
        
        return self.sqw, {'popt': popt, 'success': result.success}
