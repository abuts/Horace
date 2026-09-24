import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..dataset.base import IXDataset

@dataclass
class DnDBase(IXDataset):
    """
    Base class for binned grid objects (d0d, d1d, d2d, d3d, d4d)
    Extends IXDataset with npix (number of contributing pixels per bin) and axes boundaries.
    """
    npix: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # List of 1D arrays for bin boundaries along each dimension
    p: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.npix = np.asarray(self.npix, dtype=np.uint64)
        
        if self.npix.size > 0 and self.signal.size > 0:
            if self.npix.shape != self.signal.shape:
                raise ValueError("npix shape must match signal shape")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d['npix'] = self.npix.tolist()
        d['p'] = [ax.tolist() for ax in self.p]
        return d

@dataclass
class D0d(DnDBase):
    pass

@dataclass
class D1d(DnDBase):
    def __post_init__(self):
        super().__post_init__()
        if self.signal.size > 0 and self.signal.ndim != 1:
            raise ValueError("D1d signal must be 1D")

@dataclass
class D2d(DnDBase):
    def __post_init__(self):
        super().__post_init__()
        if self.signal.size > 0 and self.signal.ndim != 2:
            raise ValueError("D2d signal must be 2D")

@dataclass
class D3d(DnDBase):
    def __post_init__(self):
        super().__post_init__()
        if self.signal.size > 0 and self.signal.ndim != 3:
            raise ValueError("D3d signal must be 3D")

@dataclass
class D4d(DnDBase):
    def __post_init__(self):
        super().__post_init__()
        if self.signal.size > 0 and self.signal.ndim != 4:
            raise ValueError("D4d signal must be 4D")
