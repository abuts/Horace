import copy
import numpy as np
from typing import Any, List, Optional, Tuple
from .dnd import DnDBase
from .pixel_data import PixelDataBase
from .experiment import Experiment
from .main_header import MainHeader
from ..core.data_op_interface import DataOpInterface, matlab_size
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
        Serializable.__init__(self)
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

    def get_NUM_DIMS(self) -> int:
        return int(self.data.ndim)

    def get_img_size(self) -> Tuple[int, ...]:
        return matlab_size(self.data.signal)

    def has_pixels(self) -> bool:
        return self.pix is not None and getattr(self.pix, "num_pixels", 0) > 0

    def sigvar(self) -> Any:
        from ..data.sigvar import sigvar
        return sigvar(self.signal, self.error)

    def sigvar_size(self) -> Tuple[int, ...]:
        return matlab_size(self.signal)

    def sigvar_set(self, sigvarobj: Any) -> "SQW":
        from ..data.sigvar import sigvar
        if not isinstance(sigvarobj, sigvar):
            raise ValueError("Source of new signal and variance data must be a sigvar object")
        self.signal = sigvarobj.s.reshape(self.signal.shape, order="F")
        self.error = sigvarobj.e.reshape(self.error.shape, order="F")
        return self

    def copy(self) -> "SQW":
        return copy.deepcopy(self)

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
