"""Global singleton store holding unique objects per baseclass for reference containers.

Ported from Herbert/Horace MATLAB @unique_obj_store class:
- herbert_core/utilities/classes/@unique_obj_store/unique_obj_store.m
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Union

import numpy as np

from pyhorace.core.unique_only_obj_container import unique_only_obj_container


class unique_obj_store:
    """
    Singleton handle store holding one `unique_only_obj_container` per `baseclass`
    name, serving as the global backing store for `unique_references_container`.
    """

    _instance_holder: ClassVar[Optional["unique_obj_store"]] = None

    def __init__(self, *args: Any) -> None:
        self._stor_holder: Dict[str, unique_only_obj_container] = {}

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def n_types(self) -> int:
        """Number of unique class types currently stored."""
        return len(self._stor_holder)

    @property
    def typenames(self) -> List[str]:
        """List of class names currently stored."""
        return list(self._stor_holder.keys())

    @property
    def n_unique_per_type(self) -> List[int]:
        """List of unique object counts for each stored class type."""
        return [stor.n_unique for stor in self._stor_holder.values()]

    @property
    def n_obj_per_type(self) -> List[int]:
        """List of object counts for each stored class type."""
        return [stor.n_objects for stor in self._stor_holder.values()]

    @property
    def stor_holder_(self) -> Dict[str, unique_only_obj_container]:
        return self._stor_holder

    # -------------------------------------------------------------------------
    # Store Operations
    # -------------------------------------------------------------------------

    def get_objects(self, class_name: str) -> unique_only_obj_container:
        """
        Return the `unique_only_obj_container` for `class_name`.
        If no objects of `class_name` are stored yet, returns a new empty
        `unique_only_obj_container(class_name)`.
        """
        if class_name in self._stor_holder:
            return self._stor_holder[class_name]
        return unique_only_obj_container(class_name)

    def set_objects(
        self, unique_storage: unique_only_obj_container
    ) -> "unique_obj_store":
        """
        Store or update the `unique_only_obj_container` for its `baseclass`.
        """
        if not isinstance(unique_storage, unique_only_obj_container):
            raise ValueError(
                "HERBERT:unique_obj_storage:invalid_argument: "
                f"Only unique_only_obj_container may be set as storage of unique objects. "
                f"Attempt to set {type(unique_storage).__name__}"
            )
        fldname = unique_storage.baseclass
        if unique_storage.n_objects == 0 and fldname not in self._stor_holder:
            return self
        self._stor_holder[fldname] = unique_storage
        return self

    def get_value(
        self, class_name: str, glidx: Union[int, Sequence[int], np.ndarray]
    ) -> Any:
        """
        Return the value(s) stored in global memory for `class_name` at 0-based
        global index `glidx`.
        """
        if class_name not in self._stor_holder:
            raise ValueError(
                "HERBERT:unique_obj_store:invalid_argument: "
                f"Attempt to get value from empty store: {class_name}"
            )
        stor = self._stor_holder[class_name]
        if stor.is_in(glidx):
            return stor.get_at_direct_idx(glidx)
        raise ValueError(
            "HERBERT:unique_obj_store:invalid_argument: "
            f"Global index {glidx} is outside of the ranges [0,{stor.n_objects - 1}] "
            f"of global store: {class_name}"
        )

    def clear(self, class_name: Optional[str] = None) -> "unique_obj_store":
        """
        Remove the container for `class_name` from the store (or clear all if `class_name` is None).
        """
        if class_name is None:
            self._stor_holder.clear()
        elif class_name in self._stor_holder:
            del self._stor_holder[class_name]
        return self

    # -------------------------------------------------------------------------
    # Singleton Accessor
    # -------------------------------------------------------------------------

    @classmethod
    def instance(cls, *args: Any) -> Optional["unique_obj_store"]:
        """
        Return the global singleton `unique_obj_store` instance.
        Calling `unique_obj_store.instance('clear')` clears the singleton and returns `None`.
        """
        if len(args) > 0:
            cls._instance_holder = None
            if args[0] == "clear":
                return None
        if cls._instance_holder is None:
            cls._instance_holder = cls(*args)
        return cls._instance_holder


UniqueObjStore = unique_obj_store

__all__ = [
    "UniqueObjStore",
    "unique_obj_store",
]

