import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from ..crystal.oriented_lattice import OrientedLattice

@dataclass
class Instrument:
    """Metadata for an instrument."""
    name: str = ""
    energy_mode: str = ""  # 'direct', 'indirect', 'elastic'
    # Additional instrument specific parameters could be added here

@dataclass
class Sample:
    """Metadata for the sample."""
    name: str = ""
    lattice: OrientedLattice = field(default_factory=OrientedLattice)

@dataclass
class Experiment:
    """
    Metadata for a single experimental run.
    """
    run_id: int = 0
    efix: float = 0.0  # Incident or final energy
    emode: int = 1     # 1: direct, 2: indirect, 3: elastic
    psi: float = 0.0   # Goniometer angle
    
    instrument: Instrument = field(default_factory=Instrument)
    sample: Sample = field(default_factory=Sample)
    
    # Detector parameters
    det_theta: Optional[np.ndarray] = None
    det_phi: Optional[np.ndarray] = None

