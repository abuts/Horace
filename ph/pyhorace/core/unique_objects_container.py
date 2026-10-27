"""Self-contained container storing unique objects and their non-unique indices.

Ported from Herbert/Horace MATLAB @unique_objects_container class:
- herbert_core/utilities/classes/@unique_objects_container/unique_objects_container.m
- herbert_core/utilities/classes/@unique_objects_container/private/add_single_.m
- herbert_core/utilities/classes/@unique_objects_container/private/replace_.m
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
class unique_objects_container(ObjContainersBase):
    """
    Self-contained container storing unique objects, their MD5 hashes, duplicate counts,
    and the 0-based index map `idx` linking each non-unique position to its unique object.

    Uses a Python `dict[str, int]` (`_hash_to_lidx`) for O(1) hash-to-unique-index lookup.
    """

    _stored_hashes: List[str] = PrivateAttr(default_factory=list)
    _n_duplicates: List[int] = PrivateAttr(default_factory=list)
    _hash_to_lidx: Dict[str, int] = PrivateAttr(default_factory=dict)
    _pending_with_checks: bool = PrivateAttr(default=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._stored_hashes = []
        self._n_duplicates = []
        self._hash_to_lidx = {}
        self._pending_with_checks = False
        if args or kwargs:
            self.init(*args, **kwargs)

    def init(self, *args: Any, **kwargs: Any) -> "unique_objects_container":
        """Initialize container, filtering out legacy `convert_to_stream_f` arguments."""
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k != "convert_to_stream_f"
        }
        filtered_args: List[Any] = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if isinstance(arg, str) and arg == "convert_to_stream_f":
                skip_next = True
                continue
            filtered_args.append(arg)

        if filtered_args or filtered_kwargs:
            pos_fields = list(self.saveable_fields())
            self.set_positional_and_key_val_arguments(
                pos_fields, False, *filtered_args, **filtered_kwargs
            )
        return self

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def stored_hashes(self) -> List[str]:
        """List of MD5 hashes for the stored unique objects."""
        return list(self._stored_hashes)

    @property
    def stored_hashes_(self) -> List[str]:
        return self._stored_hashes

    @property
    def hashes(self) -> List[str]:
        """List of MD5 hashes for the stored unique objects."""
        return list(self._stored_hashes)

    # -------------------------------------------------------------------------
    # Core Container Operations
    # -------------------------------------------------------------------------

    def clear(self) -> "unique_objects_container":
        """Clear all objects, hashes, and indices while retaining `baseclass`."""
        self._unique_objects = []
        self._idx = []
        self._stored_hashes = []
        self._n_duplicates = []
        self._hash_to_lidx = {}
        return self

    def expose_unique_objects(self) -> List[Any]:
        """Return the list of unique objects stored in this container."""
        return list(self._unique_objects)

    def get_n_unique(self) -> int:
        return len(self._unique_objects)

    def get_n_objects(self) -> int:
        return len(self._idx)

    def get_n_duplicates(self) -> List[int]:
        return list(self._n_duplicates)

    def get_unique_objects(self, nuidx: Optional[Any] = None) -> Any:
        """
        Return the list of unique objects if `nuidx is None`, or the object(s)
        corresponding to 0-based non-unique index/indices `nuidx`.
        """
        if nuidx is None:
            return list(self._unique_objects)

        if isinstance(nuidx, (int, np.integer)):
            idx_int = int(nuidx)
            self.check_if_range_allowed(idx_int)
            uidx = self._idx[idx_int]
            return self._unique_objects[uidx]

        indices = self.check_if_range_allowed(nuidx)
        items = [self._unique_objects[self._idx[i]] for i in indices]
        if all(isinstance(x, str) for x in items):
            return "".join(items)
        return items

    def set_unique_objects(self, val: Any) -> "unique_objects_container":
        """
        Replace the list of unique objects in the container.
        If the container is non-empty, `len(val)` must equal `self.n_unique`.
        """
        if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
            self.clear()
            return self

        if not isinstance(val, (list, tuple)):
            val_list = [val]
        else:
            val_list = list(val)

        if self.n_objects > 0 and self.n_unique != len(val_list):
            raise ValueError(
                "HERBERT:unique_objects_container:invalid_argument: "
                f"If container is not empty, number of unique objects to set ({len(val_list)}) "
                f"must be equal to the number of existing unique objects ({self.n_unique})"
            )

        cleaned = [self._clean_item_for_storage(x) for x in val_list]
        if self.do_check_combo_arg and self.n_objects > 0:
            # Validate baseclass and uniqueness before replacing existing state
            new_hashes: List[str] = []
            new_hash_to_lidx: Dict[str, int] = {}
            for i, obj in enumerate(cleaned):
                if self._baseclass and not is_instance_of_baseclass(obj, self._baseclass):
                    obj_cls = resolve_matlab_type_name(obj)
                    raise ValueError(
                        "HERBERT:unique_objects_container:invalid_argument: "
                        f'Object of class "{obj_cls}" at unique index {i} is incompatible '
                        f'with container baseclass "{self._baseclass}"'
                    )
                obj, h, _ = build_hash(obj)
                assert isinstance(h, str)
                if h in new_hash_to_lidx:
                    raise ValueError(
                        "HERBERT:unique_objects_container:invalid_argument: "
                        "Non-unique objects were assigned to unique_objects property"
                    )
                cleaned[i] = obj
                new_hashes.append(h)
                new_hash_to_lidx[h] = i
            self._unique_objects = cleaned
            self._stored_hashes = new_hashes
            self._hash_to_lidx = new_hash_to_lidx
            self._pending_with_checks = False
            return self

        with_checks = bool(self._stored_hashes) or not bool(self._idx)
        self._unique_objects = cleaned
        self._stored_hashes = []
        self._hash_to_lidx = {}
        self._pending_with_checks = with_checks
        return self

    def find_in_container(
        self, obj: Any, *args: Any
    ) -> Tuple[Optional[int], str, Any]:
        """
        Find `obj` in the container in O(1) time via `_hash_to_lidx`.

        Returns
        -------
        lidx : int or None
            0-based index in `unique_objects` if present, else `None`.
        hash_val : str
            32-character hex MD5 hash of `obj`.
        obj : Any
            Input object (with cached hash populated if `hashable`).
        """
        obj, hash_val, _ = build_hash(obj)
        assert isinstance(hash_val, str)
        lidx = self._hash_to_lidx.get(hash_val)
        return lidx, hash_val, obj

    def hash(self, index: int) -> str:
        """Return stored MD5 hash for the object at 0-based non-unique `index`."""
        self.check_if_range_allowed(index)
        uidx = self._idx[int(index)]
        return self._stored_hashes[uidx]

    def add_single(self, obj: Any) -> Tuple["unique_objects_container", int]:
        """
        Add a single object to the end of the container.
        Ported from @unique_objects_container/private/add_single_.m.

        Returns `(self, nuix)` where `nuix` is the 0-based position in `self`.
        """
        self._ensure_baseclass_for_object(obj)
        obj = self._clean_item_for_storage(obj)
        lidx, hash_val, obj = self.find_in_container(obj)

        if lidx is None:
            lidx = len(self._unique_objects)
            self._unique_objects.append(obj)
            self._stored_hashes.append(hash_val)
            self._n_duplicates.append(1)
            self._hash_to_lidx[hash_val] = lidx
        else:
            self._n_duplicates[lidx] += 1

        nuix = len(self._idx)
        self._idx.append(lidx)
        return self, nuix

    def replace(
        self, obj: Any, nuix: int, *args: Any
    ) -> Tuple["unique_objects_container", int]:
        """
        Replace the object at 0-based non-unique index `nuix` with `obj`.
        If `'+'` is passed in `*args` and `nuix == self.n_objects`, appends `obj`.
        Ported from @unique_objects_container/private/replace_.m.
        """
        self.check_if_range_allowed(nuix, *args)
        self._ensure_baseclass_for_object(obj)
        nuix_int = int(nuix)

        if nuix_int == self.n_objects:
            return self.add_single(obj)

        obj = self._clean_item_for_storage(obj)
        old_uix = self._idx[nuix_int]
        obj, objhash, _ = build_hash(obj)
        assert isinstance(objhash, str)
        curhash = self._stored_hashes[old_uix]
        if objhash == curhash:
            # Update object in-place in case non-hashable properties changed
            self._unique_objects[old_uix] = obj
            return self, nuix_int

        self._n_duplicates[old_uix] -= 1
        no_more_duplicates = self._n_duplicates[old_uix] == 0

        if no_more_duplicates:
            self._unique_objects.pop(old_uix)
            self._stored_hashes.pop(old_uix)
            self._n_duplicates.pop(old_uix)
            self._hash_to_lidx = {
                h: i for i, h in enumerate(self._stored_hashes)
            }
            self._idx = [
                (i - 1 if i > old_uix else i) for i in self._idx
            ]

        ix, hash_val, obj = self.find_in_container(obj)
        if ix is None:
            ix = len(self._unique_objects)
            self._unique_objects.append(obj)
            self._stored_hashes.append(hash_val)
            self._n_duplicates.append(1)
            self._hash_to_lidx[hash_val] = ix
        else:
            self._n_duplicates[ix] += 1

        self._idx[nuix_int] = ix
        return self, nuix_int

    def replicate_runs(
        self, n_objects: int, *args: Any
    ) -> "unique_objects_container":
        """
        Expand a container holding a single object into `n_objects` identical runs.
        """
        n_obj = int(n_objects)
        if n_obj == 0 or self.n_objects == n_obj:
            return self
        if self.n_unique == 1 and self.n_objects == 1 and n_obj > 1:
            self._idx = [0] * n_obj
            self._n_duplicates = [n_obj]
            return self
        raise ValueError(
            "HERBERT:ObjContainerBase:invalid_argument: "
            "replicate_runs requires a container with 1 unique object and n_objects > 1"
        )

    def get_subset(
        self, indices: Union[int, slice, Sequence[int], np.ndarray]
    ) -> "unique_objects_container":
        """
        Return a new `unique_objects_container` containing only the items at `indices`.
        """
        sub = unique_objects_container(baseclass=self._baseclass)
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
            sub.add_single(self._unique_objects[self._idx[i]])
        return sub

    def reorder(self) -> "unique_objects_container":
        """
        Reorder `unique_objects` so that unique objects appear in order of their
        first occurrence in `idx`.
        """
        if self.n_objects <= 1 or self.n_unique <= 1:
            return self

        old_to_new: Dict[int, int] = {}
        new_unique_objects: List[Any] = []
        new_stored_hashes: List[str] = []
        new_n_duplicates: List[int] = []

        for old_uix in self._idx:
            if old_uix not in old_to_new:
                new_uix = len(new_unique_objects)
                old_to_new[old_uix] = new_uix
                new_unique_objects.append(self._unique_objects[old_uix])
                new_stored_hashes.append(self._stored_hashes[old_uix])
                new_n_duplicates.append(self._n_duplicates[old_uix])

        self._unique_objects = new_unique_objects
        self._stored_hashes = new_stored_hashes
        self._n_duplicates = new_n_duplicates
        self._idx = [old_to_new[old_uix] for old_uix in self._idx]
        self._hash_to_lidx = {h: i for i, h in enumerate(self._stored_hashes)}
        return self

    def rehashify_all(self, with_checks: bool = False) -> "unique_objects_container":
        """
        Recalculate MD5 hashes for all objects in `_unique_objects`.
        If `with_checks` is True, validates that all objects conform to `baseclass`
        and that all hashes are unique.
        """
        new_hashes: List[str] = []
        new_hash_to_lidx: Dict[str, int] = {}

        for i, obj in enumerate(self._unique_objects):
            if with_checks and self._baseclass:
                if not is_instance_of_baseclass(obj, self._baseclass):
                    obj_cls = resolve_matlab_type_name(obj)
                    raise ValueError(
                        "HERBERT:unique_objects_container:invalid_argument: "
                        f'Object of class "{obj_cls}" at unique index {i} is incompatible '
                        f'with container baseclass "{self._baseclass}"'
                    )
            obj = self._clean_item_for_storage(obj)
            obj, h, _ = build_hash(obj)
            assert isinstance(h, str)
            if with_checks and h in new_hash_to_lidx:
                raise ValueError(
                    "HERBERT:unique_objects_container:invalid_argument: "
                    "Non-unique objects were assigned to unique_objects property"
                )
            self._unique_objects[i] = obj
            new_hashes.append(h)
            new_hash_to_lidx[h] = i

        self._stored_hashes = new_hashes
        self._hash_to_lidx = new_hash_to_lidx
        return self

    @staticmethod
    def concatenate(
        objs: Union["unique_objects_container", Sequence["unique_objects_container"]],
        *args: Any,
    ) -> "unique_objects_container":
        """
        Concatenate a sequence of `unique_objects_container` instances into a single container.
        """
        if isinstance(objs, unique_objects_container):
            return objs
        if not isinstance(objs, (list, tuple)):
            raise ValueError(
                "HERBERT:unique_objects_container:invalid_argument: "
                "Input for this method may be a list/tuple or instance of unique_objects_container"
            )
        if not all(isinstance(x, unique_objects_container) for x in objs):
            raise ValueError(
                "HERBERT:unique_objects_container:invalid_argument: "
                "Input sequence may contain unique_objects_container members only"
            )
        if len(objs) == 0:
            return unique_objects_container()

        # Find first non-empty container to determine baseclass
        base_cls = ""
        for c in objs:
            if c.baseclass:
                base_cls = c.baseclass
                break

        result = unique_objects_container(baseclass=base_cls)
        for c in objs:
            if c.n_objects > 0:
                result.add(c)
        return result

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return ["baseclass", "unique_objects", "idx"]

    def _populate_single_bare(self, S: Dict[str, Any]) -> "unique_objects_container":
        if "idx" in S and S["idx"] is not None:
            idx_arr = np.asarray(S["idx"], dtype=int).ravel()
            uobjs = S.get("unique_objects", self._unique_objects)
            n_uniq = len(uobjs) if uobjs is not None else 0
            if idx_arr.size > 0 and n_uniq > 0:
                if int(np.min(idx_arr)) >= 1 and int(np.max(idx_arr)) == n_uniq and 0 not in idx_arr:
                    S = dict(S)
                    S["idx"] = [int(x) - 1 for x in idx_arr]
        return super()._populate_single_bare(S)

    def check_combo_arg(
        self, with_checks: bool = False
    ) -> "unique_objects_container":
        """
        Validate interdependent properties (`baseclass`, `unique_objects`, `idx`)
        and rebuild hashes and duplicate counts.
        """
        do_checks = with_checks or self._pending_with_checks
        self._pending_with_checks = False

        if len(self._unique_objects) == 0 and len(self._idx) == 0:
            self._stored_hashes = []
            self._n_duplicates = []
            self._hash_to_lidx = {}
            return self

        if len(self._idx) == 0 and len(self._unique_objects) > 0:
            items = list(self._unique_objects)
            self._unique_objects = []
            self._stored_hashes = []
            self._n_duplicates = []
            self._hash_to_lidx = {}
            self.add(items)
            return self

        if len(self._unique_objects) == 0 and len(self._idx) > 0:
            raise ValueError(
                "HERBERT:unique_objects_container:invalid_argument: "
                "Container has non-empty idx but empty unique_objects"
            )

        n_uniq = len(self._unique_objects)
        if any(i < 0 or i >= n_uniq for i in self._idx):
            raise ValueError(
                "HERBERT:unique_objects_container:invalid_argument: "
                f"Index in idx is out of range [0..{n_uniq - 1}]"
            )

        if len(self._stored_hashes) != n_uniq or do_checks:
            self.rehashify_all(with_checks=do_checks)

        counts = [0] * n_uniq
        for i in self._idx:
            counts[i] += 1
        self._n_duplicates = counts
        return self


UniqueObjectsContainer = unique_objects_container

__all__ = [
    "UniqueObjectsContainer",
    "unique_objects_container",
]

