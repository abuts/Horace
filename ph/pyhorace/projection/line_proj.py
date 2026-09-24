import numpy as np
from typing import Tuple, Optional
from .base import ProjectionBase

class LineProj(ProjectionBase):
    """
    Linear projection defined by vectors u, v (and optionally w).
    """
    def __init__(self, u: np.ndarray, v: np.ndarray, w: Optional[np.ndarray] = None, 
                 proj_type: str = 'rrr', u_matrix: Optional[np.ndarray] = None):
        """
        Args:
            u: First projection vector (e.g. [1, 0, 0])
            v: Second projection vector (e.g. [0, 1, 0])
            w: Optional third projection vector
            proj_type: Normalization strings, e.g., 'rrr' or 'aaa'.
            u_matrix: Optional 3x3 U matrix from the lattice to convert rlu to Cartesian.
        """
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        if w is not None:
            self.w = np.asarray(w, dtype=float)
        else:
            self.w = np.cross(self.u, self.v)
            
        self.proj_type = proj_type
        
        # Build the 3x3 projection matrix that maps (u1, u2, u3) crystal cartesian
        # into the new projection coordinates.
        # This is a simplified version. A full Horace implementation uses the B and U matrices
        # and normalizes according to proj_type ('a' = Angstrom^-1, 'r' = rlu, 'p' = proj)
        
        # For this prototype, we'll construct a simple change of basis.
        basis = np.column_stack((self.u, self.v, self.w))
        # If the vectors are independent, we can invert them to find the projection matrix
        try:
            self.P_mat = np.linalg.inv(basis)
        except np.linalg.LinAlgError:
            raise ValueError("Projection vectors u, v, w must be linearly independent.")

    def project(self, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, dE: np.ndarray) -> np.ndarray:
        # Stack coordinates into (3, N)
        coords = np.vstack((u1, u2, u3))
        
        # Apply projection matrix: (3, 3) @ (3, N) -> (3, N)
        proj_coords = self.P_mat @ coords
        
        # Output shape (N, 4): [p1, p2, p3, dE]
        out = np.empty((len(u1), 4), dtype=float)
        out[:, 0] = proj_coords[0, :]
        out[:, 1] = proj_coords[1, :]
        out[:, 2] = proj_coords[2, :]
        out[:, 3] = dE
        
        return out
