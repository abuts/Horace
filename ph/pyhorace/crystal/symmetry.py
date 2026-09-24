import numpy as np
from abc import ABC, abstractmethod

class Symop(ABC):
    """
    Base class for symmetry operations in reciprocal space.
    """
    @abstractmethod
    def get_matrix(self) -> np.ndarray:
        """Return the 3x3 transformation matrix for this symmetry operation."""
        pass

    def apply(self, q: np.ndarray) -> np.ndarray:
        """
        Apply the symmetry operation to an array of Q vectors.
        q: (N, 3) or (3,) array
        """
        R = self.get_matrix()
        q_arr = np.asarray(q)
        if q_arr.ndim == 1:
            return R @ q_arr
        else:
            return (R @ q_arr.T).T

class SymopIdentity(Symop):
    def get_matrix(self) -> np.ndarray:
        return np.eye(3)

class SymopReflection(Symop):
    def __init__(self, normal: np.ndarray):
        """
        Reflection across a plane defined by its normal vector.
        """
        n = np.asarray(normal, dtype=float)
        n = n / np.linalg.norm(n)
        self.R = np.eye(3) - 2 * np.outer(n, n)

    def get_matrix(self) -> np.ndarray:
        return self.R

class SymopRotation(Symop):
    def __init__(self, axis: np.ndarray, angle_deg: float):
        """
        Rotation around an axis by a given angle.
        """
        u = np.asarray(axis, dtype=float)
        u = u / np.linalg.norm(u)
        theta = np.deg2rad(angle_deg)
        
        ux, uy, uz = u
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Rodrigues' rotation formula
        self.R = np.array([
            [cos_t + ux**2*(1-cos_t), ux*uy*(1-cos_t) - uz*sin_t, ux*uz*(1-cos_t) + uy*sin_t],
            [uy*ux*(1-cos_t) + uz*sin_t, cos_t + uy**2*(1-cos_t), uy*uz*(1-cos_t) - ux*sin_t],
            [uz*ux*(1-cos_t) - uy*sin_t, uz*uy*(1-cos_t) + ux*sin_t, cos_t + uz**2*(1-cos_t)]
        ])

    def get_matrix(self) -> np.ndarray:
        return self.R

class SymopGeneral(Symop):
    def __init__(self, matrix: np.ndarray):
        self.R = np.asarray(matrix, dtype=float)
        if self.R.shape != (3, 3):
            raise ValueError("Symmetry matrix must be 3x3")

    def get_matrix(self) -> np.ndarray:
        return self.R
