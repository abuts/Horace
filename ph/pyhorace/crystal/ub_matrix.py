import numpy as np
from typing import Tuple

def deg2rad(deg: float) -> float:
    return deg * np.pi / 180.0

def rad2deg(rad: float) -> float:
    return rad * 180.0 / np.pi

def calc_B_matrix(a: float, b: float, c: float, alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """
    Calculate the Busing-Levy B matrix from lattice parameters.
    The B matrix converts from reciprocal lattice units (rlu) to a Cartesian 
    coordinate system attached to the reciprocal lattice.
    
    Args:
        a, b, c: Lattice constants in Angstroms.
        alpha_deg, beta_deg, gamma_deg: Lattice angles in degrees.
        
    Returns:
        B: 3x3 numpy array (Angstrom^-1)
    """
    alpha = deg2rad(alpha_deg)
    beta = deg2rad(beta_deg)
    gamma = deg2rad(gamma_deg)

    # Volume of the real-space unit cell
    v = a * b * c * np.sqrt(
        1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    # Reciprocal lattice parameters (crystallographic definition, without 2*pi)
    # Horace typically uses scattering convention which includes 2*pi in Q
    # We will compute a*, b*, c* with 2*pi
    astar = 2 * np.pi * b * c * np.sin(alpha) / v
    bstar = 2 * np.pi * a * c * np.sin(beta) / v
    cstar = 2 * np.pi * a * b * np.sin(gamma) / v

    cos_alphastar = (np.cos(beta) * np.cos(gamma) - np.cos(alpha)) / (np.sin(beta) * np.sin(gamma))
    cos_betastar = (np.cos(alpha) * np.cos(gamma) - np.cos(beta)) / (np.sin(alpha) * np.sin(gamma))
    cos_gammastar = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))

    sin_gammastar = np.sqrt(1 - cos_gammastar**2)

    # Busing and Levy convention: 
    # a* is along x
    # b* is in the xy plane
    # c* is in the xyz space
    B = np.zeros((3, 3), dtype=float)
    B[0, 0] = astar
    B[0, 1] = bstar * cos_gammastar
    B[0, 2] = cstar * cos_betastar
    B[1, 1] = bstar * sin_gammastar
    # B_23 = c* (cos alpha* - cos beta* cos gamma*) / sin gamma*
    B[1, 2] = cstar * (cos_alphastar - cos_betastar * cos_gammastar) / sin_gammastar
    B[2, 2] = np.sqrt(cstar**2 - B[0, 2]**2 - B[1, 2]**2)

    return B

def calc_U_matrix(u: np.ndarray, v: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Calculate the U matrix given two orientation vectors in rlu (u and v) and the B matrix.
    u is the primary vector, v is the secondary vector.
    """
    # Cartesian coordinates of u and v
    uc = B @ np.asarray(u, dtype=float)
    vc = B @ np.asarray(v, dtype=float)
    
    uc = uc / np.linalg.norm(uc)
    vc = vc / np.linalg.norm(vc)
    
    # Orthogonalize v wrt u
    vc = vc - np.dot(uc, vc) * uc
    vc = vc / np.linalg.norm(vc)
    
    # t3 = u x v
    wc = np.cross(uc, vc)
    
    # U matrix describes the rotation of the Cartesian system attached to the 
    # reciprocal lattice to the spectrometer Cartesian system.
    # In Horace, U matrix aligns u with the beam (or specific axis).
    # Assuming u || x, v in xy plane
    T_c = np.column_stack((uc, vc, wc))
    
    # T_phi is the target frame (e.g., standard spectrometer frame)
    # Typically T_phi is the identity matrix if we just want the crystal Cartesian frame
    T_phi = np.eye(3)
    
    U = T_phi @ np.linalg.inv(T_c)
    return U
