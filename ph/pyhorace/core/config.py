from dataclasses import dataclass
from typing import Optional

@dataclass
class HoraceConfig:
    """
    Global configuration for Horace operations.
    """
    # Multithreading / performance
    use_c_extensions: bool = True
    num_threads: int = -1  # -1 means all available cores
    
    # Memory and paging
    page_size_mb: float = 512.0
    mem_fraction: float = 0.5
    
    # Parallel computing
    parallel_backend: str = 'serial' # 'serial', 'dask', 'mpi'
    
    # File I/O
    default_file_format: str = 'v4' # hdf5
    
    # Logging
    log_level: str = 'INFO'

# Global singleton instance
config = HoraceConfig()
