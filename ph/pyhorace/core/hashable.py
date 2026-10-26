"""Hashable base class for MD5-based object hashing and fast equality comparison.

Ported from Herbert/Horace MATLAB @hashable class and build_hash utility:
- herbert_core/utilities/classes/@hashable/hashable.m
- herbert_core/utilities/classes/@hashable/eq.m
- herbert_core/utilities/classes/@hashable/ne.m
- herbert_core/utilities/classes/@hashable/isequal.m
- herbert_core/utilities/classes/@hashable/private/build_single_hash_.m
- herbert_core/utilities/classes/@hashable/private/equal_to_tol_single_.m
- herbert_core/utilities/classes/@hashable/private/to_hashable_array_.m
- herbert_core/utilities/misc/build_hash.m
"""

from __future__ import annotations

import copy
import hashlib
import pickle
from abc import ABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.serializable import Serializable, compare_values


class hashable(Serializable, ABC):
    """
    Parent class for objects that calculate and cache an MD5 hash of their state
    to accelerate equality checks and container deduplication (e.g. unique_objects_container).

    Inherits versioned serialization and validation from :class:`Serializable`.
    """

    _hash_value: Optional[str] = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Hash Properties
    # -------------------------------------------------------------------------

    @property
    def hash_value(self) -> Optional[str]:
        """Return the cached 32-character lowercase hex MD5 hash string, or None if not calculated."""
        return self._hash_value

    @property
    def hash_value_(self) -> Optional[str]:
        """Protected alias matching MATLAB's hash_value_ property."""
        return self._hash_value

    @hash_value_.setter
    def hash_value_(self, val: Optional[str]) -> None:
        self._hash_value = val if (isinstance(val, str) and len(val) > 0) else None

    @property
    def hash_defined(self) -> bool:
        """Return True if the object's hash is calculated and cached, False otherwise."""
        return bool(self._hash_value)

    # -------------------------------------------------------------------------
    # Hashable Fields & Cache Invalidation
    # -------------------------------------------------------------------------

    def hashable_fields(self) -> Sequence[str]:
        """
        Return sequence of property names used to construct the object hash.
        Defaults to saveable_fields(). Subclasses may override hashable_fields()
        or hashableFields() to exclude non-essential metadata (e.g. run_id, filepath).
        """
        if type(self).hashableFields is not hashable.hashableFields:
            return self.hashableFields()
        return self.saveable_fields()

    def hashableFields(self) -> Sequence[str]:
        """MATLAB camelCase alias for hashable_fields."""
        if type(self).hashable_fields is not hashable.hashable_fields:
            return self.hashable_fields()
        return self.saveable_fields()

    def clear_hash(self) -> "hashable":
        """Reset the cached MD5 hash value."""
        self._hash_value = None
        return self

    def check_combo_arg(self) -> "hashable":
        """Validate interdependent properties and reset cached hash."""
        super().check_combo_arg()
        self.clear_hash()
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        is_private_or_flag = name.startswith("_") or name in (
            "do_check_combo_arg",
            "do_check_combo_arg_",
            "hash_value_",
        )
        prev_hash = getattr(self, "_hash_value", None) if not is_private_or_flag else None

        super().__setattr__(name, value)

        if not is_private_or_flag:
            try:
                h_fields = self.hashable_fields()
                s_fields = self.saveable_fields()
                if name in h_fields or name not in s_fields:
                    self.clear_hash()
                elif prev_hash is not None:
                    # Modifying a non-hashable saveable property (e.g. run_id or filepath)
                    # should not invalidate an already-calculated hash even if
                    # Pydantic validate_assignment ran _validate_combo_arguments.
                    self._hash_value = prev_hash
            except Exception:
                self.clear_hash()
            self._notify_container_mutation()

    # -------------------------------------------------------------------------
    # Structure Conversion (to_struct / from_struct)
    # -------------------------------------------------------------------------

    def to_struct(self) -> Dict[str, Any]:
        """
        Convert object to a structured dictionary including 'serial_name', 'version',
        and 'hash_value' (stored as a list matching MATLAB's cell array representation).
        """
        S = super().to_struct()
        S["hash_value"] = [self._hash_value]
        return S

    @classmethod
    def to_struct_array(cls, objs: Sequence[Serializable]) -> Dict[str, Any]:
        """
        Convert a sequence of hashable objects to a structured dictionary including
        'array_dat' and 'hash_value' list.
        """
        S = super().to_struct_array(objs)
        S["hash_value"] = [getattr(o, "_hash_value", None) for o in objs]
        return S

    @classmethod
    def from_struct(
        cls,
        S: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        obj_template: Optional[Serializable] = None,
    ) -> Union["hashable", "HashableArray"]:
        """
        Restore a hashable object or HashableArray of objects from a structure created
        by to_struct() / to_struct_array(), preserving any stored hash_value.
        """
        res = super().from_struct(S, obj_template=obj_template)
        if isinstance(S, dict) and "hash_value" in S:
            hashes = S["hash_value"]
            if isinstance(res, list):
                for i, item in enumerate(res):
                    if isinstance(item, hashable):
                        h = (
                            hashes[i]
                            if isinstance(hashes, (list, tuple)) and i < len(hashes)
                            else None
                        )
                        item._hash_value = (
                            h if (isinstance(h, str) and len(h) > 0) else None
                        )
                return HashableArray(res)
            elif isinstance(res, hashable):
                if isinstance(hashes, (list, tuple)):
                    h = hashes[0] if len(hashes) > 0 else None
                elif isinstance(hashes, str):
                    h = hashes
                else:
                    h = None
                res._hash_value = h if (isinstance(h, str) and len(h) > 0) else None
                return res

        if isinstance(res, list):
            return HashableArray(res)
        return res  # type: ignore[return-value]

    to_dict = to_struct
    from_dict = from_struct

    # -------------------------------------------------------------------------
    # Hash Construction (to_hashable_array / build_hash)
    # -------------------------------------------------------------------------

    def _clone_for_hash(self) -> "hashable":
        """Create an independent shallow copy of self with its own private attributes."""
        cloned = self.model_copy()
        if self.__pydantic_extra__ is not None:
            cloned.__pydantic_extra__ = dict(self.__pydantic_extra__)
        cloned._do_check_combo_arg = self._do_check_combo_arg
        cloned._hash_value = self._hash_value
        cloned._container_callback = None
        return cloned

    @staticmethod
    def _set_hashed_field(obj: Any, fld: str, val: Any) -> None:
        """Store updated nested hashable field without triggering reconstructing property setters."""
        old_flag = obj.do_check_combo_arg
        try:
            obj.do_check_combo_arg = False
            prop = getattr(type(obj), fld, None)
            priv_name = f"_{fld}"
            if isinstance(prop, property) and hasattr(obj, priv_name):
                if getattr(obj, "__pydantic_extra__", None) is not None and priv_name in obj.__pydantic_extra__:
                    obj.__pydantic_extra__[priv_name] = val
                else:
                    object.__setattr__(obj, priv_name, val)
            else:
                object.__setattr__(obj, fld, val)
        finally:
            obj.do_check_combo_arg = old_flag

    def to_hashable_array(self) -> Tuple["hashable", bytes]:
        """
        Convert a single hashable object into a byte array suitable for MD5 hashing.
        Ported from @hashable/private/to_hashable_array_.m.

        Returns
        -------
        obj : hashable
            Updated object copy (with any nested hashable fields having their hashes cached).
        bytestream : bytes
            Byte stream encoding the object's class name and hashable_fields values.
        """
        obj = self._clone_for_hash()
        flds = obj.hashable_fields()
        chunks: List[bytes] = [obj.__class__.__name__.encode("utf-8")]

        for fld in flds:
            val = getattr(obj, fld, None)
            if val is None:
                continue
            elif isinstance(val, hashable):
                if not val.hash_defined:
                    val, h_val, new_hash = build_hash(val)
                    if new_hash:
                        self._set_hashed_field(obj, fld, val)
                else:
                    h_val = val.hash_value
                if isinstance(h_val, list):
                    chunks.append("".join(x or "" for x in h_val).encode("utf-8"))
                elif isinstance(h_val, str):
                    chunks.append(h_val.encode("utf-8"))
            elif isinstance(val, (list, tuple, HashableArray)) and (
                len(val) > 0 and all(isinstance(x, hashable) for x in val)
            ):
                updated_seq, h_val, new_hash = build_hash(val)
                if new_hash:
                    self._set_hashed_field(obj, fld, updated_seq)
                if isinstance(h_val, list):
                    chunks.append("".join(x or "" for x in h_val).encode("utf-8"))
                elif isinstance(h_val, str):
                    chunks.append(h_val.encode("utf-8"))
            elif isinstance(val, (bool, np.bool_)):
                chunks.append(np.uint8(1 if val else 0).tobytes())
            elif isinstance(val, (int, np.integer)):
                chunks.append(np.asarray(val).tobytes())
            elif isinstance(val, (float, np.floating)):
                if isinstance(val, np.float32):
                    chunks.append(np.float32(np.round(val, 6)).tobytes())
                else:
                    chunks.append(np.float32(np.round(float(val), 7)).tobytes())
            elif isinstance(val, str):
                chunks.append(val.encode("utf-8"))
            elif isinstance(val, (bytes, bytearray)):
                chunks.append(bytes(val))
            elif isinstance(val, np.ndarray) and val.dtype != object:
                flat = val.ravel(order="F")
                if flat.dtype == np.bool_:
                    chunks.append(flat.astype(np.uint8).tobytes())
                elif np.issubdtype(flat.dtype, np.floating):
                    if flat.dtype == np.float32:
                        chunks.append(np.round(flat, 6).astype(np.float32).tobytes())
                    else:
                        chunks.append(
                            np.round(flat.astype(np.float64), 7)
                            .astype(np.float32)
                            .tobytes()
                        )
                elif np.issubdtype(flat.dtype, np.number):
                    chunks.append(flat.tobytes())
                else:
                    chunks.append(pickle.dumps(val, protocol=5))
            elif isinstance(val, (list, tuple)) and (
                len(val) > 0
                and all(
                    isinstance(x, (int, float, bool, np.number, np.bool_)) for x in val
                )
            ):
                arr = np.asarray(val).ravel(order="F")
                if arr.dtype == np.bool_:
                    chunks.append(arr.astype(np.uint8).tobytes())
                elif np.issubdtype(arr.dtype, np.floating):
                    if arr.dtype == np.float32:
                        chunks.append(np.round(arr, 6).astype(np.float32).tobytes())
                    else:
                        chunks.append(
                            np.round(arr.astype(np.float64), 7)
                            .astype(np.float32)
                            .tobytes()
                        )
                else:
                    chunks.append(arr.tobytes())
            elif isinstance(val, dict):
                _, dict_hash, _ = build_hash(val)
                if isinstance(dict_hash, str):
                    chunks.append(dict_hash.encode("utf-8"))
            elif isinstance(val, Serializable):
                chunks.append(val.serialize())
            else:
                chunks.append(pickle.dumps(val, protocol=5))

        return obj, b"".join(chunks)

    def build_hash(self) -> Tuple["hashable", str, bool]:
        """
        Calculate and cache the 32-character lowercase hexadecimal MD5 hash for this object.
        Ported from @hashable/hashable.m and private/build_single_hash_.m.

        Returns
        -------
        obj : hashable
            Object instance with hash_value populated (leaves original unmutated if newly built,
            matching MATLAB value-class semantics).
        hash_val : str
            32-character lowercase hexadecimal MD5 digest.
        is_calculated : bool
            True if the hash was newly calculated, False if retrieved from cache.
        """
        if self.hash_defined and self._hash_value is not None:
            return self, self._hash_value, False

        obj, bytestream = self.to_hashable_array()
        h = hashlib.md5(bytestream).hexdigest()
        obj._hash_value = h
        return obj, h, True

    # -------------------------------------------------------------------------
    # Equality & Tolerance Comparison (equal_to_tol / eq / ne / isequal)
    # -------------------------------------------------------------------------

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """
        Compare two hashable objects within tolerance across hashable_fields().
        Ported from @hashable/private/equal_to_tol_single_.m.

        If both objects already have their hashes defined and equal, immediately
        returns (True, ""). Otherwise compares each property in hashable_fields().
        """
        if type(self) is not type(other):
            return (
                False,
                f'Objects have different types. First is "{type(self).__name__}" and second is "{type(other).__name__}"',
            )

        if (
            self.hash_defined
            and other.hash_defined
            and self._hash_value == other._hash_value
        ):
            return True, ""

        flds = self.hashable_fields()
        for fld in flds:
            val1 = getattr(self, fld, None)
            val2 = getattr(other, fld, None)
            fld_name_a = f"{name_a}.{fld}"
            fld_name_b = f"{name_b}.{fld}"

            if ignore_str and isinstance(val1, str) and isinstance(val2, str):
                continue

            iseq, mess = compare_values(
                val1,
                val2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=fld_name_a,
                name_b=fld_name_b,
            )
            if not iseq:
                return False, mess

        return True, ""

    def eq(
        self, other: Any, return_message: bool = False
    ) -> Union[bool, Tuple[bool, str]]:
        """
        Check equality of two hashable objects by comparing their MD5 hashes.
        Ported from @hashable/eq.m.
        """
        if type(self) is not type(other):
            mess = f'Objects have different types: "{type(self).__name__}" vs "{type(other).__name__}"'
            return (False, mess) if return_message else False

        _, hash1, _ = self.build_hash()
        _, hash2, _ = other.build_hash()
        is_eq = hash1 == hash2
        return (is_eq, "") if return_message else is_eq

    def ne(self, other: Any) -> bool:
        """Check inequality of two hashable objects. Ported from @hashable/ne.m."""
        return not bool(self.eq(other))

    def isequal(self, *others: Any) -> bool:
        """
        Check whether this object and all provided objects have equal hashable_fields,
        regardless of whether their hashes have been precalculated.
        Ported from @hashable/isequal.m.
        """
        return isequal(self, *others)

    def __eq__(self, other: Any) -> bool:
        return bool(self.eq(other))

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        _, h, _ = self.build_hash()
        return int(h, 16)


# PEP 8 alias
Hashable = hashable


class HashableArray(list):
    """
    List subclass representing an array of hashable objects, supporting array-level
    operations (.to_struct(), .build_hash(), .hash_defined, .hash_value, .equal_to_tol(),
    .eq(), .ne(), .isequal()) matching MATLAB's object array semantics.
    """

    @property
    def hash_defined(self) -> List[bool]:
        """Return list of booleans indicating whether each element's hash is defined."""
        return [bool(getattr(x, "hash_defined", False)) for x in self]

    @property
    def hash_value(self) -> List[Optional[str]]:
        """Return list of cached hash values for each element."""
        return [getattr(x, "hash_value", None) for x in self]

    def clear_hash(self) -> "HashableArray":
        """Clear cached hashes on all elements."""
        for x in self:
            if isinstance(x, hashable):
                x.clear_hash()
        return self

    def to_struct(self) -> Dict[str, Any]:
        """Convert array of hashable objects to a struct dict with 'array_dat' and 'hash_value'."""
        return hashable.to_struct_array(self)

    def build_hash(
        self,
    ) -> Tuple["HashableArray", Union[str, List[str], None], bool]:
        """Calculate hashes for all elements in the array."""
        return build_hash(self)

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """Compare two sequences of hashable objects element-by-element."""
        if not isinstance(other, (list, tuple, HashableArray)):
            return (
                False,
                f'Objects have different types: "{type(self).__name__}" vs "{type(other).__name__}"',
            )
        if len(self) != len(other):
            return (
                False,
                f"Sizes of first and second objects keys are different: [{len(self)}] vs [{len(other)}]",
            )
        for i, (a, b) in enumerate(zip(self, other)):
            if isinstance(a, hashable):
                ok, mess = a.equal_to_tol(
                    b,
                    tol=tol,
                    nan_equal=nan_equal,
                    ignore_str=ignore_str,
                    name_a=f"{name_a}({i + 1})",
                    name_b=f"{name_b}({i + 1})",
                )
            else:
                ok, mess = compare_values(
                    a,
                    b,
                    tol=tol,
                    nan_equal=nan_equal,
                    ignore_str=ignore_str,
                    name_a=f"{name_a}({i + 1})",
                    name_b=f"{name_b}({i + 1})",
                )
            if not ok:
                return False, mess
        return True, ""

    def eq(
        self, other: Any, return_message: bool = False
    ) -> Union[bool, Tuple[bool, str]]:
        """Compare equality of two hashable object arrays via their hashes."""
        if not isinstance(other, (list, tuple, HashableArray)):
            mess = f'Objects have different types: "{type(self).__name__}" vs "{type(other).__name__}"'
            return (False, mess) if return_message else False
        if len(self) != len(other):
            mess = f"Sizes of first and second objects keys are different: [{len(self)}] vs [{len(other)}]"
            return (False, mess) if return_message else False
        if len(self) == 0:
            return (True, "") if return_message else True

        is_eq = all(a == b for a, b in zip(self, other))
        return (is_eq, "") if return_message else is_eq

    def ne(self, other: Any) -> bool:
        return not bool(self.eq(other))

    def isequal(self, *others: Any) -> bool:
        return isequal(self, *others)

    def __eq__(self, other: Any) -> bool:
        return bool(self.eq(other))

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


def _unwrap_build_hash_result(val: Any) -> Any:
    """
    If a user passed the 3-tuple result (obj, hash_val, is_calculated) of build_hash(obj)
    directly as if it were the single-output MATLAB call `obj = build_hash(obj)`,
    extract the first element `obj`.
    """
    if (
        isinstance(val, tuple)
        and len(val) == 3
        and isinstance(val[0], (hashable, HashableArray))
        and isinstance(val[2], bool)
    ):
        return val[0]
    return val


def build_hash(
    obj: Any,
) -> Tuple[Any, Union[str, List[str], None], bool]:
    """
    Build MD5 hash (32-character lowercase hex string) from any object, hashable instance,
    or sequence of hashable instances.

    Ported from:
    - herbert_core/utilities/classes/@hashable/hashable.m (build_hash method)
    - herbert_core/utilities/misc/build_hash.m (standalone function)

    Parameters
    ----------
    obj : Any
        A `hashable` instance, a sequence (`list`, `tuple`, `HashableArray`, `np.ndarray`)
        of `hashable` instances, a byte buffer (`bytes`, `bytearray`, `uint8` ndarray),
        or any Python/Serializable object.

    Returns
    -------
    obj : Any
        Updated object or `HashableArray` with cached hash value(s).
    hash_val : str | list[str] | None
        32-character lowercase hexadecimal MD5 digest (for a single object or 1-element sequence),
        a list of 32-character hex digests (for a multi-element sequence of `hashable` objects),
        or None (for an empty sequence).
    is_calculated : bool
        True if any hash was newly calculated during this call, False if all were already cached.
    """
    obj = _unwrap_build_hash_result(obj)

    if isinstance(obj, hashable):
        return obj.build_hash()

    if isinstance(obj, HashableArray) or (
        isinstance(obj, (list, tuple))
        and (len(obj) == 0 or all(isinstance(x, hashable) for x in obj))
    ):
        if len(obj) == 0:
            return HashableArray(), None, False

        is_calculated = False
        new_items: List[hashable] = []
        hashes: List[str] = []
        for item in obj:
            if item.hash_defined and item.hash_value is not None:
                new_items.append(item)
                hashes.append(item.hash_value)
            else:
                updated_item, h, _ = item.build_hash()
                new_items.append(updated_item)
                hashes.append(h)
                is_calculated = True

        out_arr = HashableArray(new_items)
        if len(hashes) == 1:
            return out_arr, hashes[0], is_calculated
        return out_arr, hashes, is_calculated

    if (
        isinstance(obj, np.ndarray)
        and obj.dtype == object
        and obj.size > 0
        and all(isinstance(x, hashable) for x in obj.flat)
    ):
        is_calculated = False
        new_items_arr = np.empty(obj.shape, dtype=object)
        hashes = []
        for idx, item in np.ndenumerate(obj):
            if item.hash_defined and item.hash_value is not None:
                new_items_arr[idx] = item
                hashes.append(item.hash_value)
            else:
                updated_item, h, _ = item.build_hash()
                new_items_arr[idx] = updated_item
                hashes.append(h)
                is_calculated = True
        if len(hashes) == 1:
            return new_items_arr, hashes[0], is_calculated
        return new_items_arr, hashes, is_calculated

    # General non-hashable object: serialize to bytes and compute MD5
    if isinstance(obj, (bytes, bytearray)):
        bytestream = bytes(obj)
    elif isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
        bytestream = obj.ravel(order="F").tobytes()
    elif isinstance(obj, Serializable):
        bytestream = obj.serialize()
    elif isinstance(obj, dict):
        # Deterministic serialization of dictionary keys/values
        sorted_items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        bytestream = pickle.dumps(sorted_items, protocol=5)
    else:
        bytestream = pickle.dumps(obj, protocol=5)

    h = hashlib.md5(bytestream).hexdigest()
    return obj, h, True


def isequal(*objs: Any) -> bool:
    """
    Compare two or more hashable objects (or sequences of hashable objects) for equality
    within zero tolerance, independent of whether their hashes have been precalculated.

    Ported from @hashable/isequal.m.
    """
    if len(objs) < 2:
        raise ValueError("isequal requires at least two input arguments")

    first = _unwrap_build_hash_result(objs[0])
    for other_raw in objs[1:]:
        other = _unwrap_build_hash_result(other_raw)
        if isinstance(first, (hashable, HashableArray)):
            ok, _ = first.equal_to_tol(other)
        elif isinstance(first, (list, tuple)) and isinstance(other, (list, tuple)):
            if len(first) != len(other):
                return False
            ok = all(isequal(a, b) for a, b in zip(first, other))
        else:
            ok, _ = compare_values(first, other)
        if not ok:
            return False
    return True


HashableObj = hashable

__all__ = [
    "Hashable",
    "HashableObj",
    "HashableArray",
    "build_hash",
    "hashable",
    "isequal",
]

