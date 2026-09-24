import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from ..core.data_op_interface import DataOpInterface
from ..core.serializable import Serializable

@dataclass
class IXDataset(DataOpInterface, Serializable):
    """
    Base class for N-dimensional datasets (equivalent to Herbert's IX_dataset_*d).
    """
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    error: np.ndarray = field(default_factory=lambda: np.array([]))
    title: str = ""

    def __post_init__(self):
        self.signal = np.asarray(self.signal, dtype=float)
        self.error = np.asarray(self.error, dtype=float)
        if self.signal.shape != self.error.shape:
            if self.error.size == 0:
                self.error = np.zeros_like(self.signal)
            else:
                raise ValueError("Signal and error shapes must match.")

    @property
    def ndim(self) -> int:
        return self.signal.ndim

    def to_dict(self) -> dict:
        return {
            'signal': self.signal.tolist(),
            'error': self.error.tolist(),
            'title': self.title
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'IXDataset':
        return cls(
            signal=np.array(data.get('signal', [])),
            error=np.array(data.get('error', [])),
            title=data.get('title', '')
        )
