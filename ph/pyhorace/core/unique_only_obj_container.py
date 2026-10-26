"""Container storing unique objects only with permanent global reference indices.

Ported from Herbert/Horace MATLAB @unique_only_obj_container class:
- herbert_core/utilities/classes/@unique_only_obj_container/unique_only_obj_container.m
- herbert_core/utilities/classes/@unique_only_obj_container/private/add_if_new_single_.m
- herbert_core/utilities/classes/@unique_only_obj_container/private/replace_.m
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import build_hash
from pyhorace.core.obj_containers_base import (
    ObjContainersBase,
    is_instance_of_baseclass,
    resolve_matlab_type_name,
)
from pyhorace.core.serializable import register_serializable


@register_serializable
class unique_only_obj_container(ObjContainersBase):
    """
    Container holding only unique objects and providing permanent global indices (`gidx`)
    for external reference containers (`unique_references_container`) via `unique_obj_store`.

    Uses a Python `dict[str, int]` (`_hash_to_gidx`) for O(1) hash-to-global-index lookup,
    combined with the Alen & Tildesley free-list memory management (`lidx_` / `idx_`).
    """

    _n_unique: int = PrivateAttr(default=0)
    _stored_hashes: List[str] = PrivateAttr(default_factory=list)
    _n_duplicates: List[int] = PrivateAttr(default_factory=list)
    _lidx: List[int] = PrivateAttr(default_factory=list)
    _total_allocated: int = PrivateAttr(default=0)
    _max_obj_idx: int = PrivateAttr(default=0)
    _mem_expansion_chunk: int = PrivateAttr(default=100)
    _hash_to_gidx: Dict[str, int] = PrivateAttr(default_factory=dict)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._n_unique = 0
        self._stored_hashes = []
        self._n_duplicates = []
        self._lidx = []
        self._total_allocated = 0
        self._max_obj_idx = 0
        self._mem_expansion_chunk = 100
        self._hash_to_gidx = {}
        if args or kwargs:
            self.init(*args, **kwargs)

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def stored_hashes(self) -> List[str]:
        """List of stored hashes for unique objects."""
        return [self._stored_hashes[self._lidx[i]] for i in range(self._n_unique)]

    @property
    def stored_hashes_(self) -> List[str]:
        return self._stored_hashes

    @property
    def allocated_mem_size(self) -> int:
        """Total number of allocated slots in internal arrays."""
        return len(self._idx)

    @property
    def total_allocated(self) -> int:
        """Total allocated capacity."""
        return self._total_allocated

    @total_allocated.setter
    def total_allocated(self, val: int) -> None:
        self._total_allocated = int(val)

    @property
    def max_obj_idx(self) -> int:
        """High-water count of occupied global indices (valid global indices are 0..max_obj_idx-1)."""
        return self._max_obj_idx

    @property
    def mem_expansion_chunk(self) -> int:
        """Number of slots allocated per memory expansion step."""
        return self._mem_expansion_chunk

    @mem_expansion_chunk.setter
    def mem_expansion_chunk(self, val: int) -> None:
        if int(val) < 1:
            raise ValueError(
                "HERBERT:unique_only_obj_container:invalid_argument: "
                "mem_expansion_chunk must be a positive integer"
            )
        self._mem_expansion_chunk = int(val)

    @property
    def mem_expansion_chunk_(self) -> int:
        return self._mem_expansion_chunk

    @mem_expansion_chunk_.setter
    def mem_expansion_chunk_(self, val: int) -> None:
        self.mem_expansion_chunk = val

    @property
    def lidx(self) -> List[int]:
        """Internal local-to-global index array."""
        return list(self._lidx)

    @property
    def lidx_(self) -> List[int]:
        return self._lidx

    # -------------------------------------------------------------------------
    # Memory Management
    # -------------------------------------------------------------------------

    def _check_and_expand_memory_if_necessary(self) -> int:
        """
        Expand internal storage arrays by `mem_expansion_chunk` if all allocated
        slots are occupied. Returns the 0-based local index of the first empty slot.
        """
        lidx_first_empty = self._n_unique
        if lidx_first_empty < self._total_allocated:
            return lidx_first_empty

        chunk = self._mem_expansion_chunk
        start = self._total_allocated
        self._unique_objects.extend([None] * chunk)
        self._stored_hashes.extend([""] * chunk)
        self._n_duplicates.extend([0] * chunk)
        new_idx = list(range(start, start + chunk))
        self._idx.extend(new_idx)
        self._lidx.extend(new_idx)
        self._total_allocated = len(self._idx)
        return lidx_first_empty

    def is_in(self, gidx: Union[int, Sequence[int], np.ndarray]) -> bool:
        """Return True if 0-based global index `gidx` is within `[0, max_obj_idx)`."""
        arr = np.asarray(gidx, dtype=int).ravel()
        return bool(arr.size > 0 and np.all((arr >= 0) & (arr < self._max_obj_idx)))

    def get_at_direct_idx(
        self, gidx: Union[int, slice, Sequence[int], np.ndarray]
    ) -> Any:
        """
        Return object(s) given their direct 0-based global index `gidx` in internal storage.
        Used by `unique_references_container` to dereference global indices.
        """
        if isinstance(gidx, slice):
            indices = list(range(*gidx.indices(self._max_obj_idx)))
        elif isinstance(gidx, (int, np.integer)):
            idx_int = int(gidx)
            self.check_if_range_allowed(idx_int)
            return self._unique_objects[idx_int]
        else:
            indices = [int(x) for x in np.asarray(gidx).ravel()]

        self.check_if_range_allowed(indices)
        out = [self._unique_objects[i] for i in indices]
        if len(out) == 1 and not isinstance(gidx, (list, tuple, slice, np.ndarray)):
            return out[0]
        if all(isinstance(x, str) for x in out):
            return "".join(out)
        return out

    # -------------------------------------------------------------------------
    # Containers Interface Implementation
    # -------------------------------------------------------------------------

    def expose_unique_objects(self) -> List[Any]:
        """Return list of unique objects stored in this container."""
        return self.get_unique_objects()

    def find_in_container(
        self, obj: Any, return_global: bool = False
    ) -> Tuple[Optional[int], str, Any]:
        """
        Check whether `obj` is stored in `self` in O(1) time via `_hash_to_gidx`.

        Parameters
        ----------
        obj : Any
            Object to search for.
        return_global : bool, default False
            If True, return the permanent 0-based global index (`gidx`).
            If False, return the 0-based local index (`lidx` in `0..n_unique-1`).

        Returns
        -------
        idx : int or None
            0-based index if found, else `None`.
        hash_val : str
            32-character hex MD5 hash of `obj`.
        obj : Any
            Input object (with cached hash populated if `hashable`).
        """
        obj, hash_val, _ = build_hash(obj)
        assert isinstance(hash_val, str)
        gidx = self._hash_to_gidx.get(hash_val)
        if gidx is None:
            return None, hash_val, obj
        if return_global:
            return gidx, hash_val, obj
        return self._idx[gidx], hash_val, obj

    def replicate_runs(
        self, n_duplicates: int, gidx: int = 0
    ) -> "unique_only_obj_container":
        """
        Increase the reference count at global index `gidx` by `n_duplicates - 1`.
        """
        self._n_duplicates[int(gidx)] += int(n_duplicates) - 1
        return self

    def get_subset(self, indices: Any) -> Any:
        raise NotImplementedError(
            "HERBERT:unique_objects_container:not_implemented: "
            "This function is possible but does not make sense on unique_only_obj_container"
        )

    def reorder(self) -> "unique_only_obj_container":
        raise NotImplementedError(
            "HERBERT:unique_objects_container:not_implemented: "
            "This function is possible but does not make sense on unique_only_obj_container"
        )

    def hash(self, lidx: int) -> str:
        """Return stored hash for 0-based local index `lidx`."""
        self.check_if_range_allowed(lidx)
        return self._stored_hashes[self._lidx[int(lidx)]]

    def replace(
        self, obj: Any, gidx: int, *args: Any
    ) -> Tuple["unique_only_obj_container", int]:
        """
        Replace the object at 0-based global index `gidx` with `obj`.
        Ported from @unique_only_obj_container/private/replace_.m.
        """
        self._ensure_baseclass_for_object(obj)
        self.check_if_range_allowed(gidx, *args)

        old_gidx = int(gidx)
        obj = self._clean_item_for_storage(obj)
        obj, objhash, _ = build_hash(obj)
        assert isinstance(objhash, str)
        curhash = self._stored_hashes[old_gidx]
        if objhash == curhash:
            return self, old_gidx

        # Decrement duplicate count of replaced item
        self._n_duplicates[old_gidx] -= 1
        no_more_duplicates = self._n_duplicates[old_gidx] == 0
        if no_more_duplicates and curhash in self._hash_to_gidx:
            del self._hash_to_gidx[curhash]

        lidx_n, hash_val, obj = self.find_in_container(obj, return_global=False)

        if lidx_n is None:
            # New unique object not currently in container
            if no_more_duplicates:
                self._unique_objects[old_gidx] = obj
                self._stored_hashes[old_gidx] = hash_val
                self._n_duplicates[old_gidx] = 1
                self._hash_to_gidx[hash_val] = old_gidx
                new_gidx = old_gidx
            else:
                lidx_first_empty = self._check_and_expand_memory_if_necessary()
                idx_free = self._lidx[lidx_first_empty]
                self._unique_objects[idx_free] = obj
                self._stored_hashes[idx_free] = hash_val
                self._n_duplicates[idx_free] = 1
                self._idx[idx_free] = lidx_first_empty
                self._hash_to_gidx[hash_val] = idx_free
                new_gidx = idx_free
                self._n_unique += 1
            self._max_obj_idx = max(self._max_obj_idx, new_gidx + 1)
        else:
            # Object already exists in container at local index lidx_n
            new_gidx = self._lidx[lidx_n]
            if no_more_duplicates:
                old_lidx_n = self._idx[old_gidx]
                last_lidx_n = self._n_unique - 1
                gi_tmp = self._lidx[last_lidx_n]
                self._lidx[last_lidx_n] = self._lidx[old_lidx_n]
                self._lidx[old_lidx_n] = gi_tmp

                self._idx[old_gidx] = last_lidx_n
                self._idx[gi_tmp] = old_lidx_n

                self._n_unique -= 1
                self._unique_objects[old_gidx] = None
                self._stored_hashes[old_gidx] = ""

                if old_gidx == self._max_obj_idx - 1:
                    active = [
                        k
                        for k in range(self._max_obj_idx)
                        if self._n_duplicates[k] > 0
                    ]
                    self._max_obj_idx = (active[-1] + 1) if active else 0

            self._n_duplicates[new_gidx] += 1

        return self, new_gidx

    # -------------------------------------------------------------------------
    # ObjContainersBase Protected Interface
    # -------------------------------------------------------------------------

    def get_unique_objects(self, nuidx: Optional[Any] = None) -> Any:
        if nuidx is None:
            return [self._unique_objects[self._lidx[i]] for i in range(self._n_unique)]

        if isinstance(nuidx, (int, np.integer)):
            idx_int = int(nuidx)
            self.check_if_range_allowed(idx_int)
            uidx = self._lidx[idx_int]
            return self._unique_objects[uidx]

        indices = self.check_if_range_allowed(nuidx)
        items = [self._unique_objects[self._lidx[i]] for i in indices]
        if all(isinstance(x, str) for x in items):
            return "".join(items)
        return items

    def check_if_range_allowed(
        self, nuix: Union[int, Sequence[int], np.ndarray], *args: Any
    ) -> List[int]:
        if isinstance(nuix, (int, np.integer)):
            indices = [int(nuix)]
        else:
            indices = [int(x) for x in np.asarray(nuix).ravel()]

        if not indices:
            return []

        if len(args) > 0:
            upper_range = self._max_obj_idx
            if any(i == upper_range for i in indices):
                raise ValueError(
                    "HERBERT:unique_only_obj_container:invalid_argument: "
                    f"Some or all input indices: [{indices[0]}..{indices[-1]}] are at the range {upper_range}.\n"
                    "This container can not be extended by addressing its elements behind its boundaries"
                )
        upper_bound = self._max_obj_idx - 1
        if any(i < 0 or i > upper_bound for i in indices):
            raise ValueError(
                "HERBERT:ObjContainersBase:invalid_argument: "
                f"Some or all input indices: [{indices[0]}..{indices[-1]}] are outside allowed range "
                f"[0:{max(upper_bound, 0)}] for this container"
            )
        return indices

    def get_idx(self) -> List[int]:
        if not self._n_duplicates or self._max_obj_idx == 0:
            return []
        return [
            i if self._n_duplicates[i] != 0 else -1
            for i in range(self._max_obj_idx)
        ]

    def set_unique_objects(self, val: Any) -> "unique_only_obj_container":
        raise NotImplementedError(
            "HERBERT:unique_only_obj_container:not_implemented: "
            "This method is not implemented"
        )

    def get_n_duplicates(self) -> List[int]:
        return list(self._n_duplicates[: self._max_obj_idx])

    def get_n_unique(self) -> int:
        return self._n_unique

    def get_n_objects(self) -> int:
        return self._n_unique

    def add_single(self, obj: Any) -> Tuple["unique_only_obj_container", int]:
        """
        Add a single object to the container if not already present, or increment
        its reference count if present. Returns `(self, gidx)`.
        Ported from @unique_only_obj_container/private/add_if_new_single_.m.
        """
        self._ensure_baseclass_for_object(obj)
        obj = self._clean_item_for_storage(obj)
        lidx, hash_val, obj = self.find_in_container(obj, return_global=False)

        if lidx is None:
            lidx_new = self._check_and_expand_memory_if_necessary()
            gidx_free = self._lidx[lidx_new]
            self._stored_hashes[gidx_free] = hash_val
            self._unique_objects[gidx_free] = obj
            self._n_duplicates[gidx_free] = 1
            self._idx[gidx_free] = lidx_new
            self._hash_to_gidx[hash_val] = gidx_free
            self._n_unique += 1
            self._max_obj_idx = max(self._max_obj_idx, gidx_free + 1)
            gidx = gidx_free
        else:
            gidx = self._lidx[lidx]
            self._n_duplicates[gidx] += 1

        return self, gidx

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return ["baseclass"]


UniqueOnlyObjContainer = unique_only_obj_container

__all__ = [
    "UniqueOnlyObjContainer",
    "unique_only_obj_container",
]

