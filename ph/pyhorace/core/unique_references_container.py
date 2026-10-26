"""Container holding local indices referencing unique objects in global `unique_obj_store`.

Ported from Herbert/Horace MATLAB @unique_references_container class:
- herbert_core/utilities/classes/@unique_references_container/unique_references_container.m
- herbert_core/utilities/classes/@unique_references_container/private/add_single_.m
- herbert_core/utilities/classes/@unique_references_container/private/set_container_from_saved_objects_.m
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.obj_containers_base import (
    ObjContainersBase,
    is_instance_of_baseclass,
    resolve_matlab_type_name,
)
from pyhorace.core.serializable import register_serializable
from pyhorace.core.unique_obj_store import unique_obj_store
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_only_obj_container import unique_only_obj_container


@register_serializable
class unique_references_container(ObjContainersBase):
    """
    Container storing references (global indices in `unique_obj_store`) to unique
    objects of a specified `baseclass`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def global_container(self) -> unique_only_obj_container:
        """Return the backing `unique_only_obj_container` from `unique_obj_store`."""
        return unique_obj_store.instance().get_objects(self._baseclass)

    @property
    def hashes(self) -> List[str]:
        """List of MD5 hashes of the unique objects in the global store."""
        return self.global_container.stored_hashes

    # -------------------------------------------------------------------------
    # Core Container Operations
    # -------------------------------------------------------------------------

    def expose_unique_objects(self) -> List[Any]:
        """
        Return the list of unique objects in the global store for `self.baseclass`
        (matching MATLAB `expose_unique_objects` on `unique_references_container`).
        """
        return self.global_container.unique_objects

    def get_unique_objects_and_indices(
        self, get_lidx: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """
        Return `(unique_objects, unique_idx)` for only the unique objects referenced
        by this container instance.

        Parameters
        ----------
        get_lidx : bool, default False
            If False, `unique_idx` is the sorted list of unique global indices.
            If True, `unique_idx` is a list of lists of 0-based local indices in `self`
            where each unique object occurs.
        """
        unique_gidx = sorted(set(self._idx))
        storage = self.global_container
        unique_objs = [storage.get_at_direct_idx(g) for g in unique_gidx]
        if get_lidx:
            unique_lidx = [
                [i for i, g in enumerate(self._idx) if g == ug]
                for ug in unique_gidx
            ]
            return unique_objs, unique_lidx
        return unique_objs, unique_gidx

    def get_n_unique(self) -> int:
        """Return the number of unique objects referenced by this container."""
        return len(set(self._idx))

    def get_n_objects(self) -> int:
        """Return the total number of objects (runs) in this container."""
        return len(self._idx)

    def get_n_duplicates(self) -> List[int]:
        """Return the duplicate count for each unique object referenced by this container."""
        if not self._idx:
            return []
        unique_gidx = sorted(set(self._idx))
        return [sum(1 for g in self._idx if g == ug) for ug in unique_gidx]

    def get_unique_objects(self, nuidx: Optional[Any] = None) -> Any:
        """
        If `nuidx is None`, returns a `unique_objects_container` containing the
        objects referenced by `self` (used for serialization and `.unique_objects`).
        Otherwise, dereferences and returns the object(s) at 0-based local index `nuidx`.
        """
        if nuidx is None:
            uoc = unique_objects_container(baseclass=self._baseclass)
            if not self._idx:
                return uoc
            storage = self.global_container
            for gidx in self._idx:
                uoc.add_single(storage.get_at_direct_idx(gidx))
            return uoc

        self.check_if_range_allowed(nuidx)
        if isinstance(nuidx, (int, np.integer)):
            glindex = self._idx[int(nuidx)]
            return unique_obj_store.instance().get_value(
                self._baseclass, glindex
            )

        indices = self.check_if_range_allowed(nuidx)
        glindices = [self._idx[i] for i in indices]
        return unique_obj_store.instance().get_value(self._baseclass, glindices)

    def set_unique_objects(self, val: Any) -> "unique_references_container":
        """
        Populate this reference container from a saved `unique_objects_container`.
        Ported from @unique_references_container/private/set_container_from_saved_objects_.m.
        """
        if isinstance(val, (list, tuple)):
            storage = self.global_container
            storage.unique_objects = val
            unique_obj_store.instance().set_objects(storage)
            return self

        if not isinstance(val, unique_objects_container):
            raise ValueError(
                "HERBERT:unique_references_container:invalid_argument: "
                f"Input for set.unique_objects must be a unique_objects_container, got {type(val).__name__}"
            )

        if not self._baseclass:
            self._baseclass = val.baseclass
        elif val.baseclass and val.baseclass != self._baseclass:
            raise ValueError(
                "HERBERT:unique_references_container:invalid_argument: "
                f'Baseclass of input unique_objects_container ("{val.baseclass}") '
                f'does not match container baseclass ("{self._baseclass}")'
            )

        self._idx = []
        if val.n_objects > 0:
            self.add(val)
        return self

    def find_in_container(
        self, obj: Any, return_global: bool = False
    ) -> Tuple[Optional[int], str, Any]:
        """
        Check whether `obj` is referenced by `self`.

        Parameters
        ----------
        obj : Any
            Object to search for.
        return_global : bool, default False
            If True, return the 0-based global index in `unique_obj_store` if present in `self`.
            If False, return the 0-based position in `sorted(set(self._idx))` (local unique index).
        """
        storage = self.global_container
        gidx, hash_val, obj = storage.find_in_container(obj, return_global=True)
        if gidx is None or gidx not in self._idx:
            return None, hash_val, obj
        if return_global:
            return gidx, hash_val, obj
        unique_gidx = sorted(set(self._idx))
        return unique_gidx.index(gidx), hash_val, obj

    def hash(self, index: int) -> str:
        """Return stored MD5 hash for the object at 0-based local `index`."""
        self.check_if_range_allowed(index)
        gidx = self._idx[int(index)]
        storage = self.global_container
        return storage.stored_hashes_[gidx]

    def add_single(self, obj: Any) -> Tuple["unique_references_container", int]:
        """
        Add a single object to the container via `unique_obj_store`.
        Ported from @unique_references_container/private/add_single_.m.

        Returns `(self, nuidx)` where `nuidx` is the 0-based local index in `self`.
        """
        self._ensure_baseclass_for_object(obj)
        storage = unique_obj_store.instance().get_objects(self._baseclass)
        storage, gidx = storage.add_single(obj)
        nuidx = len(self._idx)
        self._idx.append(gidx)
        unique_obj_store.instance().set_objects(storage)
        return self, nuidx

    def replace(
        self, obj: Any, nuix: int, *args: Any
    ) -> Tuple["unique_references_container", int]:
        """
        Replace the object at 0-based local index `nuix` with `obj`.
        Note: Matches MATLAB `unique_references_container.replace` (Re #1816), which
        adds `obj` to the global `unique_only_obj_container` and updates `self._idx[nuix]`
        without removing the previous object from global storage (as other containers
        may reference it).
        """
        self._ensure_baseclass_for_object(obj)
        self.check_if_range_allowed(nuix, *args)
        nuix_int = int(nuix)

        if nuix_int == self.n_objects:
            _, added_idx = self.add_single(obj)
            return self, added_idx

        storage = unique_obj_store.instance().get_objects(self._baseclass)
        storage, gidx = storage.add_single(obj)
        self._idx[nuix_int] = gidx
        unique_obj_store.instance().set_objects(storage)
        return self, nuix_int

    def replicate_runs(
        self, n_objects: int, *args: Any
    ) -> "unique_references_container":
        """
        Expand a single-run container to `n_objects` identical runs and update
        global duplicate counts in `unique_obj_store`.
        """
        n_obj = int(n_objects)
        if n_obj == 0 or self.n_objects == n_obj:
            return self
        if self.n_unique == 1 and self.n_objects == 1 and n_obj > 1:
            gidx = self._idx[0]
            self._idx = [gidx] * n_obj
            storage = self.global_container
            storage.replicate_runs(n_obj, gidx)
            unique_obj_store.instance().set_objects(storage)
            return self
        raise ValueError(
            "HERBERT:ObjContainerBase:invalid_argument: "
            "replicate_runs requires a container with 1 unique object and n_objects > 1"
        )

    def get_subset(
        self, indices: Union[int, slice, Sequence[int], np.ndarray]
    ) -> "unique_references_container":
        """
        Return a new `unique_references_container` containing the subset of items at `indices`.
        """
        sub = unique_references_container(self._baseclass)
        if isinstance(indices, slice):
            idx_list = list(range(*indices.indices(self.n_objects)))
        elif isinstance(indices, (int, np.integer)):
            idx_list = [int(indices)]
        else:
            idx_list = [int(x) for x in np.asarray(indices).ravel()]

        if not idx_list:
            return sub

        self.check_if_range_allowed(idx_list)
        for i in idx_list:
            sub.add_single(self.get_unique_objects(i))
        return sub

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> Sequence[str]:
        return ["baseclass", "unique_objects"]


UniqueReferencesContainer = unique_references_container

__all__ = [
    "UniqueReferencesContainer",
    "unique_references_container",
]

