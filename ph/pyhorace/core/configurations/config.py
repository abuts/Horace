"""Configuration management facade for PyHorace.

Re-exports all configuration classes from their individual modules:
- `.config_store` (`config_store`, `ConfigData`, `HoraceConfigError`, `HoraceConfigWarning`, `warn_with_id`, `lastwarn`)
- `.config_base` (`config_base`, `get`, `set`)
- `.log_config` (`log_config`)
- `.parallel_config` (`parallel_config`)
- `.hpc_config` (`hpc_config`)
- `.hor_config` (`hor_config`)
- `.herbert_config` (`herbert_config`)
- `.planner_config` (`planner_config`)
- `.graph_config` (`graph_config`)
"""

from __future__ import annotations

from dataclasses import dataclass

from .config_base import config_base, get, set
from .config_store import (
    CONFIG_CLASS_REGISTRY,
    ConfigData,
    HoraceConfigError,
    HoraceConfigWarning,
    _sys_memory,
    config_store,
    lastwarn,
    warn_with_id,
)
from .graph_config import graph_config
from .herbert_config import herbert_config
from .hor_config import hor_config
from .hpc_config import hpc_config
from .log_config import log_config
from .parallel_config import parallel_config
from .planner_config import planner_config


@dataclass
class HoraceConfig:
    """Legacy dataclass alias for global Horace configuration."""

    use_c_extensions: bool = True
    num_threads: int = -1
    page_size_mb: float = 512.0
    mem_fraction: float = 0.5
    parallel_backend: str = "serial"
    default_file_format: str = "v4"
    log_level: str = "INFO"


config = HoraceConfig()

__all__ = [
    "CONFIG_CLASS_REGISTRY",
    "ConfigData",
    "HoraceConfig",
    "HoraceConfigError",
    "HoraceConfigWarning",
    "config",
    "config_base",
    "config_store",
    "get",
    "graph_config",
    "herbert_config",
    "hor_config",
    "hpc_config",
    "lastwarn",
    "log_config",
    "parallel_config",
    "planner_config",
    "set",
    "warn_with_id",
]
