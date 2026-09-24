import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple, Any

@dataclass
class SigVar:
    """
    Signal and variance pair with Gaussian error propagation.
    """
    s: np.ndarray  # Signal
    e: np.ndarray  # Variance (error squared)

    def __post_init__(self):
        self.s = np.asarray(self.s, dtype=float)
        self.e = np.asarray(self.e, dtype=float)
        if self.s.shape != self.e.shape:
            raise ValueError(f"Signal shape {self.s.shape} must match variance shape {self.e.shape}")

    def __add__(self, other: Union['SigVar', float, int, np.ndarray]) -> 'SigVar':
        if isinstance(other, SigVar):
            return SigVar(self.s + other.s, self.e + other.e)
        else:
            return SigVar(self.s + other, self.e.copy())

    def __radd__(self, other: Union[float, int, np.ndarray]) -> 'SigVar':
        return self.__add__(other)

    def __sub__(self, other: Union['SigVar', float, int, np.ndarray]) -> 'SigVar':
        if isinstance(other, SigVar):
            return SigVar(self.s - other.s, self.e + other.e)
        else:
            return SigVar(self.s - other, self.e.copy())

    def __rsub__(self, other: Union[float, int, np.ndarray]) -> 'SigVar':
        return SigVar(other - self.s, self.e.copy())

    def __mul__(self, other: Union['SigVar', float, int, np.ndarray]) -> 'SigVar':
        if isinstance(other, SigVar):
            # C = A * B => var_C = A^2 * var_B + B^2 * var_A
            s_new = self.s * other.s
            e_new = (self.s ** 2) * other.e + (other.s ** 2) * self.e
            return SigVar(s_new, e_new)
        else:
            other_arr = np.asarray(other)
            return SigVar(self.s * other_arr, self.e * (other_arr ** 2))

    def __rmul__(self, other: Union[float, int, np.ndarray]) -> 'SigVar':
        return self.__mul__(other)

    def __truediv__(self, other: Union['SigVar', float, int, np.ndarray]) -> 'SigVar':
        if isinstance(other, SigVar):
            # C = A / B => var_C = (var_A + (A/B)^2 * var_B) / B^2
            s_new = self.s / other.s
            e_new = (self.e + (s_new ** 2) * other.e) / (other.s ** 2)
            return SigVar(s_new, e_new)
        else:
            other_arr = np.asarray(other)
            return SigVar(self.s / other_arr, self.e / (other_arr ** 2))

    def __rtruediv__(self, other: Union[float, int, np.ndarray]) -> 'SigVar':
        # C = other / self => var_C = (other / self^2)^2 * self.e
        other_arr = np.asarray(other)
        s_new = other_arr / self.s
        e_new = ((other_arr / (self.s ** 2)) ** 2) * self.e
        return SigVar(s_new, e_new)

    def __pow__(self, power: Union[float, int]) -> 'SigVar':
        # C = A^p => var_C = (p * A^(p-1))^2 * var_A
        s_new = self.s ** power
        e_new = ((power * (self.s ** (power - 1))) ** 2) * self.e
        return SigVar(s_new, e_new)
