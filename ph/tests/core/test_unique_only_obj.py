"""Unit tests for unique_only_obj_container.

Ported from Herbert/Horace MATLAB unit tests:
- _test/test_object_containers/test_unique_only_obj.m
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
from pydantic import Field

from pyhorace.core.configurations import lastwarn
from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.core.unique_only_obj_container import (
    UniqueOnlyObjContainer,
    unique_only_obj_container,
)
from pyhorace.data.instr_classes import (
    IX_aperture,
    IX_fermi_chopper,
    IX_inst,
    IX_inst_DGfermi,
    IX_moderator,
    IX_null_inst,
)


@register_serializable
class hashable_tester_for_uoc(hashable):
    """Ported from test_hashable_obj in test_object_containers."""

    data: Any = Field(default=10)
    non_hash_prop: Any = Field(default="aux")

    def __init__(
        self, data: Any = 10, non_hash_prop: Any = "aux", **kwargs: Any
    ) -> None:
        super().__init__(data=data, non_hash_prop=non_hash_prop, **kwargs)

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return ["data", "non_hash_prop"]

    def hashable_fields(self) -> Sequence[str]:
        return ["data"]


class TestUniqueOnlyObjContainer:
    """Tests ported from test_unique_only_obj.m."""

    def test_empty_constructor_and_properties(self):
        uooc = unique_only_obj_container()
        assert uooc.baseclass == ""
        assert uooc.n_unique == 0
        assert uooc.n_objects == 0
        assert uooc.max_obj_idx == 0
        assert uooc.total_allocated == 0
        assert uooc.allocated_mem_size == 0
        assert uooc.idx == []
        assert uooc.n_duplicates == []
        assert uooc.unique_objects == []

        uooc_typed = unique_only_obj_container("char")
        assert uooc_typed.baseclass == "char"

    def test_mem_expansion_and_add_objects(self):
        uooc = unique_only_obj_container("char")
        uooc.mem_expansion_chunk = 3
        assert uooc.mem_expansion_chunk == 3

        with pytest.raises(ValueError, match="HERBERT:unique_only_obj_container:invalid_argument"):
            uooc.mem_expansion_chunk = 0

        # Add 4 unique strings, triggering memory expansion from 3 to 6
        _, gidx0 = uooc.add("A")
        _, gidx1 = uooc.add("B")
        _, gidx2 = uooc.add("C")
        assert uooc.total_allocated == 3
        assert uooc.n_unique == 3
        assert [gidx0, gidx1, gidx2] == [0, 1, 2]

        # Adding duplicate does not expand memory
        _, gidx_dup = uooc.add("B")
        assert gidx_dup == 1
        assert uooc.total_allocated == 3
        assert uooc.n_unique == 3
        assert uooc.n_duplicates == [1, 2, 1]

        # Adding 4th unique string expands memory to 6
        _, gidx3 = uooc.add("D")
        assert gidx3 == 3
        assert uooc.total_allocated == 6
        assert uooc.n_unique == 4
        assert uooc.max_obj_idx == 4
        assert uooc.n_duplicates == [1, 2, 1, 1]
        assert uooc.unique_objects == ["A", "B", "C", "D"]
        assert uooc.idx == [0, 1, 2, 3]

    def test_replace_unique_and_duplicated_elements_and_free_list_reuse(self):
        uooc = unique_only_obj_container("char")
        uooc.mem_expansion_chunk = 4

        # Add 'A', 'B', 'C', and duplicate 'B'
        uooc.add(["A", "B", "C", "B"])
        assert uooc.n_unique == 3
        assert uooc.n_duplicates == [1, 2, 1]
        assert uooc.max_obj_idx == 3

        # 1. Replace 'A' (at gidx 0, 1 duplicate) with new object 'E':
        # Since 'A' had only 1 duplicate, slot 0 is reused in-place
        _, new_gidx = uooc.replace("E", 0)
        assert new_gidx == 0
        assert uooc.n_unique == 3
        assert uooc.get_at_direct_idx(0) == "E"
        assert uooc.n_duplicates == [1, 2, 1]

        # 2. Replace one instance of 'B' (at gidx 1, 2 duplicates) with new object 'F':
        # Since 'B' still has 1 remaining duplicate at gidx 1, 'F' takes free slot gidx 3
        _, new_gidx_f = uooc.replace("F", 1)
        assert new_gidx_f == 3
        assert uooc.n_unique == 4
        assert uooc.max_obj_idx == 4
        assert uooc.n_duplicates == [1, 1, 1, 1]
        assert uooc.get_at_direct_idx(1) == "B"
        assert uooc.get_at_direct_idx(3) == "F"

        # 3. Replace 'C' (at gidx 2, 1 duplicate) with existing object 'B' (at gidx 1):
        # Slot 2 becomes free (n_duplicates[2] == 0), 'B' duplicate count becomes 2,
        # and n_unique decreases from 4 to 3!
        _, new_gidx_b = uooc.replace("B", 2)
        assert new_gidx_b == 1
        assert uooc.n_unique == 3
        assert uooc.max_obj_idx == 4
        assert uooc.n_duplicates == [1, 2, 0, 1]
        assert uooc.idx == [0, 1, -1, 3]
        assert sorted(uooc.unique_objects) == ["B", "E", "F"]

        # 4. Now add a new unique object 'G':
        # It must reuse the freed global slot 2 from the free list!
        _, gidx_g = uooc.add("G")
        assert gidx_g == 2
        assert uooc.n_unique == 4
        assert uooc.max_obj_idx == 4
        assert uooc.n_duplicates == [1, 2, 1, 1]
        assert uooc.get_at_direct_idx(2) == "G"

    def test_replace_last_occupied_slot_shrinks_max_obj_idx(self):
        uooc = unique_only_obj_container("char")
        uooc.add(["A", "B", "C"])
        assert uooc.max_obj_idx == 3

        # Replace 'C' (at last slot gidx 2) with existing 'A' (at gidx 0)
        _, new_gidx = uooc.replace("A", 2)
        assert new_gidx == 0
        assert uooc.n_unique == 2
        assert uooc.max_obj_idx == 2
        assert uooc.n_duplicates == [2, 1]
        assert uooc.idx == [0, 1]

    def test_indexing_and_unsupported_methods(self):
        uooc = unique_only_obj_container("hashable_tester_for_uoc")
        obj1 = hashable_tester_for_uoc(10, "first")
        obj2 = hashable_tester_for_uoc(20, "second")
        obj1_dup = hashable_tester_for_uoc(10, "ignored_aux")

        uooc, g0 = uooc.add(obj1)
        uooc, g1 = uooc.add(obj2)
        uooc, g2 = uooc.add(obj1_dup)
        assert [g0, g1, g2] == [0, 1, 0]
        assert uooc.n_unique == 2
        assert uooc[0] == obj1
        assert uooc[1] == obj2

        # Cannot extend container by indexing at max_obj_idx
        with pytest.raises(ValueError, match="HERBERT:unique_only_obj_container:invalid_argument"):
            uooc[2] = hashable_tester_for_uoc(30)

        # Unsupported methods raise NotImplementedError
        with pytest.raises(NotImplementedError, match="HERBERT:unique_only_obj_container:not_implemented"):
            uooc.unique_objects = [obj1]
        with pytest.raises(NotImplementedError, match="HERBERT:unique_objects_container:not_implemented"):
            uooc.get_subset([0])
        with pytest.raises(NotImplementedError, match="HERBERT:unique_objects_container:not_implemented"):
            uooc.reorder()

