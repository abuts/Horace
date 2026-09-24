import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple
import h5py

class PixelDataBase(ABC):
    """
    Abstract base class for 9-column pixel records.
    Columns: u1, u2, u3, dE, run_idx, detector_idx, energy_idx, signal, variance
    """
    
    @property
    @abstractmethod
    def num_pixels(self) -> int:
        pass

    @abstractmethod
    def get_pixels(self, start: int = 0, count: int = -1) -> np.ndarray:
        """Get a chunk of pixel data. If count is -1, get all from start."""
        pass
        
    @property
    def u1(self) -> np.ndarray: return self.get_pixels()[:, 0]
    @property
    def u2(self) -> np.ndarray: return self.get_pixels()[:, 1]
    @property
    def u3(self) -> np.ndarray: return self.get_pixels()[:, 2]
    @property
    def dE(self) -> np.ndarray: return self.get_pixels()[:, 3]
    @property
    def run_idx(self) -> np.ndarray: return self.get_pixels()[:, 4]
    @property
    def detector_idx(self) -> np.ndarray: return self.get_pixels()[:, 5]
    @property
    def energy_idx(self) -> np.ndarray: return self.get_pixels()[:, 6]
    @property
    def signal(self) -> np.ndarray: return self.get_pixels()[:, 7]
    @property
    def variance(self) -> np.ndarray: return self.get_pixels()[:, 8]

class PixelDataMemory(PixelDataBase):
    """
    In-memory storage for pixel data.
    """
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: np.ndarray of shape (N, 9)
        """
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != 9:
            if self.data.size == 0:
                self.data = np.empty((0, 9), dtype=float)
            else:
                raise ValueError(f"PixelData must be Nx9 array, got {self.data.shape}")

    @property
    def num_pixels(self) -> int:
        return self.data.shape[0]

    def get_pixels(self, start: int = 0, count: int = -1) -> np.ndarray:
        if count == -1:
            return self.data[start:]
        return self.data[start:start+count]

class PixelDataFileBacked(PixelDataBase):
    """
    File-backed storage for pixel data, reading chunks from an HDF5 file on demand.
    """
    def __init__(self, filepath: str, dataset_path: str = 'pix'):
        """
        Args:
            filepath: Path to HDF5 file.
            dataset_path: Path to the pixel dataset within the HDF5 file.
        """
        self.filepath = filepath
        self.dataset_path = dataset_path
        
        # Read shape without loading into memory
        with h5py.File(self.filepath, 'r') as f:
            if self.dataset_path not in f:
                raise KeyError(f"Dataset {self.dataset_path} not found in {self.filepath}")
            dset = f[self.dataset_path]
            self._shape = dset.shape
            
        if len(self._shape) != 2 or self._shape[1] != 9:
            raise ValueError(f"Dataset must be Nx9, got {self._shape}")

    @property
    def num_pixels(self) -> int:
        return self._shape[0]

    def get_pixels(self, start: int = 0, count: int = -1) -> np.ndarray:
        with h5py.File(self.filepath, 'r') as f:
            dset = f[self.dataset_path]
            if count == -1:
                return dset[start:]
            else:
                return dset[start:start+count]
