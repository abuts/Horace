import numpy as np
from dataclasses import dataclass, field
from .base import IXDataset

@dataclass
class IXDataset2D(IXDataset):
    """
    2D dataset containing x and y axis values and associated signal/error.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        
        # Check shapes (signal should be 2D)
        if self.signal.size > 0:
            if self.signal.ndim != 2:
                raise ValueError("Signal must be 2D for IXDataset2D")
            
            nx, ny = self.signal.shape
            
            if self.x.size > 0 and self.x.shape[0] not in (nx, nx + 1):
                raise ValueError("x dimension length incompatible with signal shape")
            if self.y.size > 0 and self.y.shape[0] not in (ny, ny + 1):
                raise ValueError("y dimension length incompatible with signal shape")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'x': self.x.tolist(),
            'y': self.y.tolist(),
            'x_label': self.x_label,
            'y_label': self.y_label,
            'z_label': self.z_label
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'IXDataset2D':
        return cls(
            signal=np.array(data.get('signal', [])),
            error=np.array(data.get('error', [])),
            title=data.get('title', ''),
            x=np.array(data.get('x', [])),
            y=np.array(data.get('y', [])),
            x_label=data.get('x_label', ''),
            y_label=data.get('y_label', ''),
            z_label=data.get('z_label', '')
        )
