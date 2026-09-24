import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from .dnd import DnDBase
from .pixel_data import PixelDataBase
from .experiment import Experiment
from .main_header import MainHeader
from ..core.data_op_interface import DataOpInterface
from ..core.serializable import Serializable

class SQW(DataOpInterface, Serializable):
    """
    Core 4D dataset container (equivalent to Horace @sqw).
    Contains:
    - data: A DnD grid object representing the binned volume.
    - pix: A PixelDataBase instance containing individual neutron events.
    - experiment_info: List of Experiment metadata (one per run).
    - main_header: Provenance and creation info.
    """
    def __init__(self, data: DnDBase, pix: Optional[PixelDataBase] = None, 
                 experiment_info: Optional[List[Experiment]] = None, 
                 main_header: Optional[MainHeader] = None):
        self.data = data
        self.pix = pix
        self.experiment_info = experiment_info if experiment_info is not None else []
        self.main_header = main_header if main_header is not None else MainHeader()
        
    @property
    def signal(self) -> np.ndarray:
        return self.data.signal
        
    @signal.setter
    def signal(self, val: np.ndarray):
        self.data.signal = val

    @property
    def error(self) -> np.ndarray:
        return self.data.error
        
    @error.setter
    def error(self, val: np.ndarray):
        self.data.error = val
        
    @property
    def ndim(self) -> int:
        return self.data.ndim
        
    def to_dict(self) -> dict:
        # Note: serializing pixels could be huge, typically we just serialize the structure/path
        return {
            'main_header': self.main_header.__dict__,
            'data': self.data.to_dict()
            # Omit pix and experiment_info for basic dict serialization
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'SQW':
        raise NotImplementedError("Deserialization of SQW from dict not yet implemented")

    def __repr__(self) -> str:
        s = f"SQW Object [{self.ndim}D]\n"
        s += f"  Title: {self.main_header.title}\n"
        s += f"  Grid shape: {self.signal.shape}\n"
        n_pix = self.pix.num_pixels if self.pix is not None else 0
        s += f"  Total pixels: {n_pix}\n"
        s += f"  Runs: {len(self.experiment_info)}"
        return s
