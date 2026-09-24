import numpy as np
from typing import Callable

def gaussian_1d(x: np.ndarray, height: float, center: float, sigma: float) -> np.ndarray:
    """
    1D Gaussian peak.
    """
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)

def lorentzian_1d(x: np.ndarray, height: float, center: float, gamma: float) -> np.ndarray:
    """
    1D Lorentzian peak.
    gamma is the Half Width at Half Maximum (HWHM).
    """
    return height * (gamma**2 / ((x - center)**2 + gamma**2))

def dho_1d(x: np.ndarray, height: float, center: float, gamma: float, temperature: float) -> np.ndarray:
    """
    1D Damped Harmonic Oscillator (DHO) multiplied by Bose population factor.
    x is energy transfer in meV.
    temperature in K.
    """
    # kB = 0.08617 meV/K
    kB = 0.08617
    # Bose factor (detailed balance)
    # Using 1 - exp(-E/kT) to handle both E>0 and E<0
    with np.errstate(divide='ignore', invalid='ignore'):
        bose = 1.0 / (1.0 - np.exp(-x / (kB * temperature)))
    
    # Handle x=0 explicitly if needed, limit is kT/x
    bose = np.where(x == 0, kB * temperature, bose)
    
    # DHO spectral weight
    dho = (4 * height * center * gamma) / (((x**2 - center**2)**2) + 4 * (gamma * x)**2)
    return dho * bose * x
