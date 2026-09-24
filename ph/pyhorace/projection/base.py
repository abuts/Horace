import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

class ProjectionBase(ABC):
    """
    Base class for coordinate projections from Crystal Cartesian (u1, u2, u3, dE)
    to the target projection space.
    """
    
    @abstractmethod
    def project(self, u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, dE: np.ndarray) -> np.ndarray:
        """
        Projects 4D coordinates to the new basis.
        Args:
            u1, u2, u3, dE: 1D arrays of length N
        Returns:
            Projected coordinates of shape (N, 4)
        """
        pass
