import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from .ub_matrix import calc_B_matrix, calc_U_matrix

@dataclass
class OrientedLattice:
    """
    Represents a crystal lattice and its orientation.
    """
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    
    # Orientation vectors in rlu
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    
    # U matrix directly provided
    U: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.u is None:
            self.u = np.array([1.0, 0.0, 0.0])
        else:
            self.u = np.asarray(self.u, dtype=float)
            
        if self.v is None:
            self.v = np.array([0.0, 1.0, 0.0])
        else:
            self.v = np.asarray(self.v, dtype=float)

    @property
    def alatt(self) -> Tuple[float, float, float]:
        return (self.a, self.b, self.c)

    @property
    def angdeg(self) -> Tuple[float, float, float]:
        return (self.alpha, self.beta, self.gamma)

    @property
    def B(self) -> np.ndarray:
        return calc_B_matrix(self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    @property
    def ub_matrix(self) -> np.ndarray:
        if self.U is not None:
            return self.U @ self.B
        U = calc_U_matrix(self.u, self.v, self.B)
        return U @ self.B
