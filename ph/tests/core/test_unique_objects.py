"""Unit tests for unique_objects_container.

Ported from Herbert/Horace MATLAB unit tests:
- _test/test_object_containers/test_unique_objects.m
- _test/test_object_containers/test_unique_objects_2.m
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
from pydantic import Field

from pyhorace.core.configurations import lastwarn
from pyhorace.core.hashable import build_hash, hashable
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.core.unique_objects_container import (
    UniqueObjectsContainer,
    unique_objects_container,
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
class thingy(Serializable):
    """Test helper class ported from _test/test_object_containers/thingy.m."""

    data: Any = Field(default=111)

    def __init__(self, data: Any = 111, **kwargs: Any) -> None:
        super().__init__(data=data, **kwargs)

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Sequence[str]:
        return ["data"]


@register_serializable
class thingy_child(thingy):
    """Subclass of thingy for baseclass inheritance tests."""

    extra: Any = Field(default=0)

    def __init__(self, data: Any = 111, extra: Any = 0, **kwargs: Any) -> None:
        super().__init__(data=data, extra=extra, **kwargs)

    def saveable_fields(self) -> Sequence[str]:
        return ["data", "extra"]


def _make_merlin_instrument(efix: float = 100.0) -> IX_inst_DGfermi:
    """Create a realistic IX_inst_DGfermi instrument for container tests."""
    mod = IX_moderator(10.0, 12.0, "ikcarp", [10.0, 20.0, 0.5])
    ap = IX_aperture(-2.0, 0.05, 0.05)
    fc = IX_fermi_chopper(1.5, 600.0, 0.05, 1.3, 0.002, energy=efix)
    return IX_inst_DGfermi(mod, ap, fc, efix, name="MERLIN")


class TestUniqueObjectsContainer:
    """Tests ported from test_unique_objects.m."""

    def test_concatenate_empty_or_invalid(self):
        with pytest.raises(ValueError, match="HERBERT:unique_objects_container:invalid_argument"):
            unique_objects_container.concatenate("invalid")

        with pytest.raises(ValueError, match="HERBERT:unique_objects_container:invalid_argument"):
            unique_objects_container.concatenate(["invalid"])

        uoc1 = unique_objects_container("char")
        uoc1.add(["a", "b", "a"])
        res_single = unique_objects_container.concatenate(uoc1)
        assert res_single == uoc1

        uoc2 = unique_objects_container("char")
        uoc2.add(["b", "c"])
        combined = unique_objects_container.concatenate([uoc1, uoc2])
        assert combined.n_objects == 5
        assert combined.n_unique == 3
        assert combined.unique_objects == ["a", "b", "c"]
        assert combined.idx == [0, 1, 0, 1, 2]
        assert combined.n_duplicates == [2, 2, 1]

    def test_constructor_arguments_and_legacy_serializer(self):
        # Legacy 'convert_to_stream_f' argument should be ignored cleanly
        uoc = unique_objects_container(
            baseclass="thingy", convert_to_stream_f=lambda x: x
        )
        assert uoc.baseclass == "thingy"
        assert uoc.n_objects == 0
        assert uoc.n_unique == 0

        uoc2 = unique_objects_container(
            "convert_to_stream_f", lambda x: x, "baseclass", "thingy"
        )
        assert uoc2.baseclass == "thingy"

    def test_add_identical_and_non_identical_returns_correct_index(self):
        uoc = unique_objects_container(baseclass="thingy")
        uoc, ix0 = uoc.add(thingy(111))
        assert ix0 == 0
        assert uoc.n_objects == 1
        assert uoc.n_unique == 1

        # Add non-identical object
        uoc, ix1 = uoc.add(thingy(222))
        assert ix1 == 1
        assert uoc.n_objects == 2
        assert uoc.n_unique == 2

        # Add duplicate of first object
        uoc, ix2 = uoc.add(thingy(111))
        assert ix2 == 2
        assert uoc.n_objects == 3
        assert uoc.n_unique == 2
        assert uoc.idx == [0, 1, 0]
        assert uoc.n_duplicates == [2, 1]

        # Add multiple objects at once
        uoc, ix_multi = uoc.add([thingy(222), thingy(333)])
        assert ix_multi == [3, 4]
        assert uoc.n_objects == 5
        assert uoc.n_unique == 3
        assert uoc.idx == [0, 1, 0, 1, 2]
        assert uoc.n_duplicates == [2, 2, 1]

    def test_python_indexing_and_in_place_property_mutation(self):
        uoc = unique_objects_container(baseclass="thingy")
        for i in range(1, 7):
            uoc.add(thingy(i * 100))
        for i in range(3, 7):
            uoc.add(thingy(i * 100))

        assert len(uoc) == 10
        assert uoc.n_unique == 6
        assert uoc[0] == thingy(100)
        assert uoc[2] == thingy(300)
        assert uoc[6] == thingy(300)

        # Mutate property on element 0 (which has only 1 duplicate)
        uoc[0].data = 222
        assert uoc[0] == thingy(222)
        assert uoc.n_unique == 6

        # Mutate property on element 2 (which had 2 duplicates: at index 2 and index 6)
        uoc[2].data = 999
        assert uoc[2] == thingy(999)
        assert uoc[6] == thingy(300)
        assert uoc.n_unique == 7

        # Assign directly via __setitem__
        uoc[2] = thingy(300)
        assert uoc[2] == thingy(300)
        assert uoc.n_unique == 6

        # Append at n_objects position via __setitem__
        uoc[10] = thingy(777)
        assert len(uoc) == 11
        assert uoc[10] == thingy(777)
        assert uoc.n_unique == 7

        # Out of range raises ValueError
        with pytest.raises(ValueError, match="HERBERT:ObjContainersBase:invalid_argument"):
            _ = uoc[11]
        with pytest.raises(ValueError, match="HERBERT:ObjContainersBase:invalid_argument"):
            uoc[12] = thingy(888)

    def test_contains_object_and_classname(self):
        uoc = unique_objects_container(baseclass="IX_inst")
        inst_null = IX_null_inst()
        inst_dg1 = _make_merlin_instrument(100.0)
        inst_dg2 = _make_merlin_instrument(200.0)

        uoc.add([inst_null, inst_dg1, inst_null, inst_dg2])
        assert uoc.n_objects == 4
        assert uoc.n_unique == 3

        is_in, idx = uoc.contains(inst_null)
        assert is_in is True
        assert idx == [0, 2]

        is_in_dg1, idx_dg1 = uoc.contains(inst_dg1)
        assert is_in_dg1 is True
        assert idx_dg1 == [1]

        inst_dg3 = _make_merlin_instrument(300.0)
        is_in_dg3, idx_dg3 = uoc.contains(inst_dg3)
        assert is_in_dg3 is False
        assert idx_dg3 == []

        # Contains by class name
        has_dg, idx_dg = uoc.contains("IX_inst_DGfermi")
        assert has_dg is True
        assert idx_dg == [1, 3]

        has_null, idx_null = uoc.contains("IX_null_inst")
        assert has_null is True
        assert idx_null == [0, 2]

    def test_serialization_roundtrip(self):
        uoc = unique_objects_container(baseclass="thingy")
        uoc.add([thingy(10), thingy(20), thingy(10), thingy(30)])

        s = uoc.to_struct()
        rec = Serializable.from_struct(s)
        assert isinstance(rec, unique_objects_container)
        assert rec == uoc
        assert rec.baseclass == "thingy"
        assert rec.n_objects == 4
        assert rec.n_unique == 3
        assert rec.idx == [0, 1, 0, 2]
        assert rec.n_duplicates == [2, 1, 1]
        assert rec.stored_hashes == uoc.stored_hashes


class TestUniqueObjectsContainer2:
    """Tests ported from test_unique_objects_2.m."""

    def test_default_constructor_and_warning_on_uninitialized_baseclass(self):
        lastwarn("", "")
        uoc = unique_objects_container()
        assert uoc.baseclass == ""
        assert uoc.n_objects == 0

        uoc.add("first_string")
        msg, wid = lastwarn()
        assert wid == "HERBERT:ObjContainerBase:incomplete_setup"
        assert uoc.baseclass == "char"
        assert uoc.n_objects == 1
        assert uoc[0] == "first_string"

        # Adding incompatible type after baseclass is set raises ValueError
        with pytest.raises(ValueError, match="HERBERT:ObjContainerBase:invalid_argument"):
            uoc.add(123.0)

    def test_set_unique_objects_validation(self):
        uoc = unique_objects_container("char", ["AA", "BB"])
        assert uoc.n_objects == 2
        assert uoc.n_unique == 2
        assert uoc[0] == "AA"
        assert uoc[1] == "BB"

        # Replicate runs through idx
        uoc.add("AA")
        assert uoc.n_objects == 3
        assert uoc.n_unique == 2
        assert uoc.idx == [0, 1, 0]

        # Setting unique_objects with wrong count raises ValueError
        with pytest.raises(ValueError, match="HERBERT:unique_objects_container:invalid_argument"):
            uoc.unique_objects = ["CC"]

        # Setting unique_objects with non-unique items raises ValueError
        with pytest.raises(ValueError, match="HERBERT:unique_objects_container:invalid_argument"):
            uoc.unique_objects = ["CC", "CC"]

        # Setting unique_objects with incompatible type raises ValueError
        with pytest.raises(ValueError, match="HERBERT:unique_objects_container:invalid_argument"):
            uoc.unique_objects = [10, 20]

        # Setting valid replacement unique_objects works
        uoc.unique_objects = ["CC", "DD"]
        assert uoc[0] == "CC"
        assert uoc[1] == "DD"
        assert uoc[2] == "CC"

    def test_replicate_runs_and_get_subset_and_reorder(self):
        uoc = unique_objects_container("thingy")
        uoc.add(thingy(42))
        uoc.replicate_runs(5)
        assert uoc.n_objects == 5
        assert uoc.n_unique == 1
        assert uoc.n_duplicates == [5]
        for i in range(5):
            assert uoc[i] == thingy(42)

        # Replace middle items and test get_subset
        uoc[1] = thingy(100)
        uoc[3] = thingy(200)
        sub = uoc.get_subset([1, 3, 4])
        assert sub.n_objects == 3
        assert sub.n_unique == 3
        assert sub[0] == thingy(100)
        assert sub[1] == thingy(200)
        assert sub[2] == thingy(42)

        # Test reorder when first element is replaced so unique order != first occurrence order
        uoc2 = unique_objects_container("char")
        uoc2.add(["A", "B", "A"])
        uoc2[0] = "B"
        # Now idx is [1, 1, 0] where unique_objects is ['A', 'B']
        assert uoc2.idx == [1, 1, 0]
        uoc2.reorder()
        assert uoc2.unique_objects == ["B", "A"]
        assert uoc2.idx == [0, 0, 1]
        assert uoc2[0] == "B"
        assert uoc2[1] == "B"
        assert uoc2[2] == "A"

