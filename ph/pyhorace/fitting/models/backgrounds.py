import numpy as np

def flat_bg(x: np.ndarray, c: float) -> np.ndarray:
    """Flat background."""
    return np.full_like(x, c, dtype=float)

def linear_bg(x: np.ndarray, c: float, m: float) -> np.ndarray:
    """Linear background: c + m*x"""
    return c + m * x

def quadratic_bg(x: np.ndarray, c: float, m: float, q: float) -> np.ndarray:
    """Quadratic background: c + m*x + q*x^2"""
    return c + m * x + q * x**2
