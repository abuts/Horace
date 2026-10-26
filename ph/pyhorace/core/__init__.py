"""Core data models and abstractions for pyhorace."""

from pyhorace.core.configurations import (
    ConfigData,
    HoraceConfig,
    HoraceConfigError,
    HoraceConfigWarning,
    config,
    config_base,
    config_store,
    get,
    graph_config,
    herbert_config,
    hor_config,
    hpc_config,
    lastwarn,
    log_config,
    parallel_config,
    planner_config,
    set,
    warn_with_id,
)
from pyhorace.core.data_op_interface import (
    DataOpArray,
    DataOpInterface,
    data_op_interface,
    sigvar_size,
    size_array_split,
)
from pyhorace.core.hashable import (
    Hashable,
    HashableArray,
    build_hash,
    hashable,
    isequal,
)
from pyhorace.core.obj_containers_base import (
    ObjContainersBase,
    is_instance_of_baseclass,
    resolve_matlab_type_name,
)
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.core.unique_obj_store import UniqueObjStore, unique_obj_store
from pyhorace.core.unique_objects_container import (
    UniqueObjectsContainer,
    unique_objects_container,
)
from pyhorace.core.unique_only_obj_container import (
    UniqueOnlyObjContainer,
    unique_only_obj_container,
)
from pyhorace.core.unique_references_container import (
    UniqueReferencesContainer,
    unique_references_container,
)

__all__ = [
    "ConfigData",
    "DataOpArray",
    "DataOpInterface",
    "Hashable",
    "HashableArray",
    "HoraceConfig",
    "HoraceConfigError",
    "HoraceConfigWarning",
    "ObjContainersBase",
    "Serializable",
    "UniqueObjStore",
    "UniqueObjectsContainer",
    "UniqueOnlyObjContainer",
    "UniqueReferencesContainer",
    "build_hash",
    "config",
    "config_base",
    "config_store",
    "data_op_interface",
    "get",
    "graph_config",
    "hashable",
    "herbert_config",
    "hor_config",
    "hpc_config",
    "is_instance_of_baseclass",
    "isequal",
    "lastwarn",
    "log_config",
    "parallel_config",
    "planner_config",
    "register_serializable",
    "resolve_matlab_type_name",
    "set",
    "sigvar_size",
    "size_array_split",
    "unique_obj_store",
    "unique_objects_container",
    "unique_only_obj_container",
    "unique_references_container",
    "warn_with_id",
]
