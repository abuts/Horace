"""Abstract base class for Herbert/Horace unique object containers.

Ported from Herbert/Horace MATLAB @ObjContainersBase class:
- herbert_core/utilities/classes/@ObjContainersBase/ObjContainersBase.m
- herbert_core/utilities/classes/@ObjContainersBase/get_unique_field.m
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.configurations import warn_with_id
from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import SERIALIZABLE_REGISTRY, Serializable


def resolve_matlab_type_name(obj: Any) -> str:
    """Return canonical MATLAB/Horace type name for a Python object."""
    if isinstance(obj, str):
        return "char"
    if isinstance(obj, (bool, np.bool_)):
        return "logical"
    if isinstance(obj, (int, float, np.number)):
        return "double"
    if isinstance(obj, dict):
        return "struct"
    return type(obj).__name__


def is_instance_of_baseclass(obj: Any, baseclass: str) -> bool:
    """Check whether `obj` satisfies the container's `baseclass` constraint."""
    if not baseclass:
        return True
    if baseclass in ("char", "str", "string"):
        return isinstance(obj, str)
    if baseclass in ("double", "float", "numeric", "int"):
        return isinstance(obj, (int, float, np.number)) and not isinstance(
            obj, (bool, np.bool_)
        )
    if baseclass in ("logical", "bool"):
        return isinstance(obj, (bool, np.bool_))
    if baseclass in ("struct", "dict"):
        return isinstance(obj, dict)

    # Check class hierarchy by name or registered Serializable class
    target_cls = SERIALIZABLE_REGISTRY.get(baseclass)
    if target_cls is not None and isinstance(obj, target_cls):
        return True
    return any(cls.__name__ == baseclass for cls in type(obj).__mro__)


class ObjContainersBase(Serializable, ABC):
    """
    Abstract base class for containers holding unique objects or references to unique objects.
    Uses standard 0-based Python indexing via `[]` (`__getitem__` / `__setitem__`).
    """

    _baseclass: str = PrivateAttr(default="")
    _unique_objects: List[Any] = PrivateAttr(default_factory=list)
    _idx: List[int] = PrivateAttr(default_factory=list)
    _hashes: List[str] = PrivateAttr(default_factory=list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._baseclass = ""
        self._unique_objects = []
        self._idx = []
        self._hashes = []
        if args or kwargs:
            self.init(*args, **kwargs)

    def init(self, *args: Any, **kwargs: Any) -> "ObjContainersBase":
        """Initialize container with positional and/or keyword arguments."""
        pos_fields = list(self.saveable_fields())
        self.set_positional_and_key_val_arguments(
            pos_fields, False, *args, **kwargs
        )
        return self

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def baseclass(self) -> str:
        """Base class name of objects stored in this container."""
        return self.get_baseclass()

    @baseclass.setter
    def baseclass(self, val: Any) -> None:
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:ObjContainerBase:invalid_argument: "
                "baseclass must be a string defining class name"
            )
        if self.n_objects > 0 and self._baseclass and val != self._baseclass:
            # Verify whether existing objects conform to the new baseclass
            first_obj = self._peek_first_object()
            if first_obj is not None and not is_instance_of_baseclass(first_obj, val):
                raise ValueError(
                    "HERBERT:ObjContainerBase:invalid_argument: "
                    f'Existing objects in container are incompatible with new baseclass "{val}"'
                )
        self._baseclass = val
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def baseclass_(self) -> str:
        return self._baseclass

    @baseclass_.setter
    def baseclass_(self, val: str) -> None:
        self._baseclass = val

    @property
    def unique_objects(self) -> Any:
        """Container or list of unique objects stored in this container."""
        return self.get_unique_objects()

    @unique_objects.setter
    def unique_objects(self, val: Any) -> None:
        self.set_unique_objects(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def unique_objects_(self) -> List[Any]:
        return self._unique_objects

    @unique_objects_.setter
    def unique_objects_(self, val: List[Any]) -> None:
        self._unique_objects = list(val)

    @property
    def idx(self) -> List[int]:
        """List of 0-based indices linking non-unique positions to unique objects."""
        return self.get_idx()

    @idx.setter
    def idx(self, val: Sequence[int]) -> None:
        self.set_idx(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def idx_(self) -> List[int]:
        return self._idx

    @idx_.setter
    def idx_(self, val: Sequence[int]) -> None:
        self._idx = [int(x) for x in val]

    @property
    def hashes(self) -> List[str]:
        """List of hashes of stored unique objects."""
        return list(self._hashes)

    @property
    def n_unique(self) -> int:
        """Number of unique objects in the container."""
        return self.get_n_unique()

    @property
    def n_objects(self) -> int:
        """Total number of (non-unique) objects in the container."""
        return self.get_n_objects()

    @property
    def n_runs(self) -> int:
        """Number of runs (alias for n_objects in Horace Experiment)."""
        return self.get_n_objects()

    @n_runs.setter
    def n_runs(self, val: int) -> None:
        self.replicate_runs(val)

    @property
    def n_duplicates(self) -> List[int]:
        """Number of occurrences of each unique object in the container."""
        return self.get_n_duplicates()

    # -------------------------------------------------------------------------
    # Abstract & Protected Interface
    # -------------------------------------------------------------------------

    def get_baseclass(self) -> str:
        return self._baseclass

    def get_idx(self) -> List[int]:
        return list(self._idx)

    def set_idx(self, val: Sequence[int]) -> "ObjContainersBase":
        if val is None or len(val) == 0:
            self._idx = []
            return self
        arr = np.asarray(val, dtype=int).ravel()
        if np.any(arr < 0):
            raise ValueError(
                "HERBERT:ObjContainerBase:invalid_argument: "
                "indices must be non-negative integers"
            )
        self._idx = [int(x) for x in arr]
        return self

    def _peek_first_object(self) -> Any:
        if self.n_objects == 0:
            return None
        try:
            return self._raw_get_single(0)
        except Exception:
            return None

    def _ensure_baseclass_for_object(self, obj: Any) -> None:
        """Initialize baseclass with warning if empty, or validate obj against baseclass."""
        if not self._baseclass:
            if isinstance(obj, ObjContainersBase):
                self._baseclass = obj.baseclass
            else:
                self._baseclass = resolve_matlab_type_name(obj)
            warn_with_id(
                "HERBERT:ObjContainerBase:incomplete_setup",
                f'baseclass not initialised, using first assigned type: "{self._baseclass}"',
            )
            return

        if not is_instance_of_baseclass(obj, self._baseclass):
            obj_cls = resolve_matlab_type_name(obj)
            raise ValueError(
                "HERBERT:ObjContainerBase:invalid_argument: "
                f'Assigning object of class: "{obj_cls}" to container with baseclass: "{self._baseclass}" is prohibited'
            )

    @staticmethod
    def _clean_item_for_storage(obj: Any) -> Any:
        """Ensure stored Serializable/hashable object does not hold container callback."""
        if isinstance(obj, hashable):
            cloned = obj._clone_for_hash()
            cloned._container_callback = None
            return cloned
        elif isinstance(obj, Serializable):
            cloned = obj.model_copy()
            if obj.__pydantic_extra__ is not None:
                cloned.__pydantic_extra__ = dict(obj.__pydantic_extra__)
            cloned._do_check_combo_arg = obj._do_check_combo_arg
            cloned._container_callback = None
            return cloned
        return obj

    def _attach_callback_for_return(self, obj: Any, idx: int) -> Any:
        """Return a shallow clone of a Serializable object with a mutation callback to (self, idx)."""
        if isinstance(obj, hashable):
            cloned = obj._clone_for_hash()
            cloned._container_callback = (self, int(idx))
            return cloned
        elif isinstance(obj, Serializable):
            cloned = obj.model_copy()
            if obj.__pydantic_extra__ is not None:
                cloned.__pydantic_extra__ = dict(obj.__pydantic_extra__)
            cloned._do_check_combo_arg = obj._do_check_combo_arg
            cloned._container_callback = (self, int(idx))
            return cloned
        return obj

    def _on_item_mutated(self, idx: int, mutated_obj: Any) -> None:
        """Callback invoked when a property is mutated on an item retrieved via container[idx]."""
        clean_obj = self._clean_item_for_storage(mutated_obj)
        self.replace(clean_obj, idx)

    def check_if_range_allowed(
        self, nuix: Union[int, Sequence[int], np.ndarray], *args: Any
    ) -> List[int]:
        """
        Validate that 0-based index/indices `nuix` are within allowed range for this container.
        If `'+'` is passed in `*args`, index `self.n_objects` is also allowed (append position).
        """
        if isinstance(nuix, (int, np.integer)):
            indices = [int(nuix)]
        else:
            indices = [int(x) for x in np.asarray(nuix).ravel()]

        if not indices:
            return []

        allow_plus = len(args) > 0 and args[0] == "+"
        upper_bound = self.n_objects if allow_plus else (self.n_objects - 1)
        max_disp = self.n_objects + 1 if allow_plus else max(self.n_objects, 1)

        if any(i < 0 or i > upper_bound for i in indices):
            first_i = indices[0]
            last_i = indices[-1]
            raise ValueError(
                "HERBERT:ObjContainersBase:invalid_argument: "
                f"Some or all input indices: [{first_i}..{last_i}] are outside allowed range "
                f"[0:{max(upper_bound, 0)}] for container: {type(self).__name__}"
            )
        return indices

    @abstractmethod
    def get_unique_objects(self, nuidx: Optional[Any] = None) -> Any:
        """Return unique objects or object(s) at specified non-unique index."""

    @abstractmethod
    def set_unique_objects(self, val: Any) -> "ObjContainersBase":
        """Set unique objects in the container."""

    @abstractmethod
    def get_n_unique(self) -> int:
        """Return number of unique objects in the container."""

    @abstractmethod
    def get_n_objects(self) -> int:
        """Return total number of objects in the container."""

    @abstractmethod
    def get_n_duplicates(self) -> List[int]:
        """Return list of duplicate counts for each unique object."""

    @abstractmethod
    def add_single(self, obj: Any) -> Tuple["ObjContainersBase", int]:
        """Add a single object to the container and return (self, index)."""

    @abstractmethod
    def expose_unique_objects(self) -> List[Any]:
        """Return list of unique objects stored in or referenced by this container."""

    @abstractmethod
    def replicate_runs(self, n_objects: int, *args: Any) -> "ObjContainersBase":
        """Expand a single-unique-object container to `n_objects` identical runs."""

    @abstractmethod
    def find_in_container(
        self, obj: Any, *args: Any
    ) -> Tuple[Optional[int], str, Any]:
        """Find `obj` in the container, returning `(index_or_None, hash_str, obj_with_hash)`."""

    @abstractmethod
    def get_subset(self, indices: Union[int, Sequence[int], slice]) -> Any:
        """Return a sub-container for the specified indices."""

    @abstractmethod
    def replace(
        self, obj: Any, nuix: int, *args: Any
    ) -> Tuple["ObjContainersBase", int]:
        """Replace object at index `nuix` with `obj`."""

    @abstractmethod
    def hash(self, index: int) -> str:
        """Return stored hash for the object at `index`."""

    # -------------------------------------------------------------------------
    # Concrete Shared Methods
    # -------------------------------------------------------------------------

    def add(self, obj: Any) -> Tuple["ObjContainersBase", Union[int, List[int]]]:
        """
        Add one or more objects (scalar, list, tuple, numpy array, or container)
        to the end of the container.

        Returns `(self, added_index_or_indices)`.
        """
        if isinstance(obj, ObjContainersBase):
            n_items = obj.n_objects
            added_indices: List[int] = []
            for i in range(n_items):
                _, idx_single = self.add_single(obj.get(i))
                added_indices.append(idx_single)
            return self, (added_indices[0] if len(added_indices) == 1 else added_indices)

        if isinstance(obj, (list, tuple)):
            added_indices = []
            for item in obj:
                _, idx_single = self.add_single(item)
                added_indices.append(idx_single)
            return self, (added_indices[0] if len(added_indices) == 1 else added_indices)

        if isinstance(obj, np.ndarray) and obj.ndim > 0 and obj.size > 1:
            # Multi-element numpy array: if baseclass is numeric/double or empty or object array,
            # add element by element
            if obj.dtype == object or self._baseclass in ("", "double", "float", "int", "numeric"):
                added_indices = []
                for item in obj.flat:
                    val = item.item() if isinstance(item, np.generic) else item
                    _, idx_single = self.add_single(val)
                    added_indices.append(idx_single)
                return self, (added_indices[0] if len(added_indices) == 1 else added_indices)

        if isinstance(obj, np.generic):
            obj = obj.item()

        _, idx_single = self.add_single(obj)
        return self, idx_single

    def _raw_get_single(self, nuidx: int) -> Any:
        return self.get_unique_objects(nuidx)

    def get(self, nuidx: Union[int, slice, Sequence[int], np.ndarray] = 0) -> Any:
        """
        Retrieve object(s) at 0-based non-unique index `nuidx`.
        Supports a single integer, a `slice`, or a sequence of integer indices.
        """
        if isinstance(nuidx, slice):
            indices = list(range(*nuidx.indices(self.n_objects)))
            return self._get_multiple(indices)
        if isinstance(nuidx, (list, tuple, np.ndarray)):
            indices = [int(x) for x in np.asarray(nuidx).ravel()]
            return self._get_multiple(indices)

        idx_int = int(nuidx)
        obj = self.get_unique_objects(idx_int)
        return self._attach_callback_for_return(obj, idx_int)

    def _get_multiple(self, indices: List[int]) -> Any:
        if not indices:
            return []
        self.check_if_range_allowed(indices)
        items = [self._attach_callback_for_return(self.get_unique_objects(i), i) for i in indices]
        if all(isinstance(x, str) for x in items):
            return "".join(items)
        return items

    def get_unique_field(self, fieldname: str) -> "ObjContainersBase":
        """
        Create a new container of the same type as `self`, populated with
        `getattr(item, fieldname)` (or `item[fieldname]` for dicts) for every item in `self`.
        Ported from @ObjContainersBase/get_unique_field.m.
        """
        subc = type(self)()
        if self.n_objects == 0:
            return subc

        first_item = self.get(0)
        first_sub = (
            first_item[fieldname]
            if isinstance(first_item, dict)
            else getattr(first_item, fieldname)
        )
        subc.baseclass = resolve_matlab_type_name(first_sub)
        for i in range(self.n_objects):
            item = self.get(i)
            subobj = (
                item[fieldname]
                if isinstance(item, dict)
                else getattr(item, fieldname)
            )
            subc.add(subobj)
        return subc

    def contains(self, obj_or_classname: Any) -> Tuple[bool, List[int]]:
        """
        Check whether the container contains an instance matching `obj_or_classname`
        (either a class name string or a specific object value).
        Returns `(is_present, matching_0_based_indices)`.
        """
        if isinstance(obj_or_classname, str):
            matches = [
                i
                for i in range(self.n_objects)
                if is_instance_of_baseclass(self.get(i), obj_or_classname)
            ]
            return (len(matches) > 0), matches

        _, h_target, _ = self.find_in_container(obj_or_classname)
        matches = [
            i for i in range(self.n_objects) if self.hash(i) == h_target
        ]
        return (len(matches) > 0), matches

    # -------------------------------------------------------------------------
    # Python Indexing (`[]`), Iteration, Length
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_objects

    def __iter__(self) -> Iterator[Any]:
        for i in range(self.n_objects):
            yield self.get(i)

    def __getitem__(self, key: Union[int, slice, Sequence[int], np.ndarray]) -> Any:
        return self.get(key)

    def __setitem__(
        self, key: Union[int, slice, Sequence[int], np.ndarray], value: Any
    ) -> None:
        if isinstance(key, (int, np.integer)):
            idx = int(key)
            self.replace(value, idx, "+")
            return

        if isinstance(key, slice):
            indices = list(range(*key.indices(self.n_objects)))
        else:
            indices = [int(x) for x in np.asarray(key).ravel()]

        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(indices):
            for idx, val_item in zip(indices, value):
                self.replace(val_item, idx, "+")
        else:
            for idx in indices:
                self.replace(value, idx, "+")


__all__ = [
    "ObjContainersBase",
    "is_instance_of_baseclass",
    "resolve_matlab_type_name",
]
