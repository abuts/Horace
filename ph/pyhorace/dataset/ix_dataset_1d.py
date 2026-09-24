import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .base import IXDataset

@dataclass
class IXDataset1D(IXDataset):
    """
    1D dataset containing x-axis values and associated signal/error.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    x_label: str = ""
    y_label: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.x = np.asarray(self.x, dtype=float)
        
        # Check shapes
        # In typical histogram data, len(x) might be len(signal) + 1 if x represents bin boundaries
        # Or len(x) == len(signal) if x represents bin centers.
        if self.x.size > 0 and self.signal.size > 0:
            if self.x.shape[0] not in (self.signal.shape[0], self.signal.shape[0] + 1):
                raise ValueError(f"x shape {self.x.shape} incompatible with signal shape {self.signal.shape}")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            'x': self.x.tolist(),
            'x_label': self.x_label,
            'y_label': self.y_label
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'IXDataset1D':
        return cls(
            signal=np.array(data.get('signal', [])),
            error=np.array(data.get('error', [])),
            title=data.get('title', ''),
            x=np.array(data.get('x', [])),
            x_label=data.get('x_label', ''),
            y_label=data.get('y_label', '')
        )
