import numpy as np
from dataclasses import dataclass, field
from .base import IXDataset

@dataclass
class IXDataset3D(IXDataset):
    """
    3D dataset containing x, y, and z axis values and associated signal/error.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    z: np.ndarray = field(default_factory=lambda: np.array([]))
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    v_label: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        
        # Check shapes (signal should be 3D)
        if self.signal.size > 0:
            if self.signal.ndim != 3:
                raise ValueError("Signal must be 3D for IXDataset3D")
            
            nx, ny, nz = self.signal.shape
            
            if self.x.size > 0 and self.x.shape[0] not in (nx, nx + 1):
                raise ValueError("x dimension length incompatible with signal shape")
            if self.y.size > 0 and self.y.shape[0] not in (ny, ny + 1):
                raise ValueError("y dimension length incompatible with signal shape")
            if self.z.size > 0 and self.z.shape[0] not in (nz, nz + 1):
                raise ValueError("z dimension length incompatible with signal shape")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'z': self.z.tolist(),
            'x_label': self.x_label,
            'y_label': self.y_label,
            'z_label': self.z_label,
            'v_label': self.v_label
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'IXDataset3D':
        return cls(
            signal=np.array(data.get('signal', [])),
            error=np.array(data.get('error', [])),
            title=data.get('title', ''),
            x=np.array(data.get('x', [])),
            y=np.array(data.get('y', [])),
            z=np.array(data.get('z', [])),
            x_label=data.get('x_label', ''),
            y_label=data.get('y_label', ''),
            z_label=data.get('z_label', ''),
            v_label=data.get('v_label', '')
        )
