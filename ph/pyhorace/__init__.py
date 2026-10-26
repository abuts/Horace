"""PyHorace: Python library for multi-dimensional neutron spectroscopy data analysis."""

from pyhorace.core.configurations import (
    config_base,
    config_store,
    graph_config,
    herbert_config,
    hor_config,
    hpc_config,
    log_config,
    parallel_config,
    planner_config,
)
from pyhorace.core.data_op_interface import DataOpArray, DataOpInterface, data_op_interface
from pyhorace.core.hashable import (
    Hashable,
    HashableArray,
    build_hash,
    hashable,
    isequal,
)
from pyhorace.core.serializable import Serializable
from pyhorace.data.sigvar import SigVar, sigvar
from pyhorace.io import main_header_cl, parse_full_filename
import pyhorace.coord_transf as coord_transf  # Ensure coord_transf initializes before sqw
from pyhorace.sqw import (
    DnDBase,
    SQWDnDBase,
    d0d,
    d1d,
    d2d,
    d3d,
    d4d,
    data_sqw_dnd,
    dnd,
    dnd_data,
    dnd_metadata,
    read_dnd,
    read_horace,
    read_sqw,
    sqw,
)

__version__ = "0.1.0"
__all__ = [
    "DataOpArray",
    "DataOpInterface",
    "DnDBase",
    "Hashable",
    "HashableArray",
    "SQWDnDBase",
    "Serializable",
    "SigVar",
    "build_hash",
    "config_base",
    "config_store",
    "d0d",
    "d1d",
    "d2d",
    "d3d",
    "d4d",
    "data_op_interface",
    "data_sqw_dnd",
    "dnd",
    "dnd_data",
    "dnd_metadata",
    "graph_config",
    "hashable",
    "herbert_config",
    "hor_config",
    "hpc_config",
    "isequal",
    "log_config",
    "parallel_config",
    "planner_config",
    "read_dnd",
    "read_horace",
    "read_sqw",
    "sigvar",
    "sqw",
]
