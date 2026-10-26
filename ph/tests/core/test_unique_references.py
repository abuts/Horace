"""Unit tests for unique_references_container, unique_obj_store, and get_unique_field.

Ported from Herbert/Horace MATLAB unit tests:
- _test/test_object_containers/test_unique_references.m
- _test/test_object_containers/test_get_unique_fields.m
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest
from pydantic import Field

from pyhorace.core.configurations import lastwarn
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.core.unique_obj_store import UniqueObjStore, unique_obj_store
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import (
    UniqueReferencesContainer,
    unique_references_container,
)
from pyhorace.data.instr_classes import (
    IX_aperture,
    IX_fermi_chopper,
    IX_inst,
    IX_inst_DGfermi,
    IX_moderator,
    IX_null_inst,
)
from tests.core.test_unique_objects import _make_merlin_instrument, thingy


@pytest.fixture(autouse=True)
def clean_global_store():
    """Ensure a clean global unique_obj_store before and after each test."""
    unique_obj_store.instance().clear()
    yield
    unique_obj_store.instance().clear()


class TestUniqueReferencesContainer:
    """Tests ported from test_unique_references.m."""

    def test_empty_constructor_and_global_store_interaction(self):
        urc1 = unique_references_container("thingy")
        urc2 = unique_references_container("thingy")
        assert urc1.baseclass == "thingy"
        assert urc1.n_objects == 0
        assert urc1.n_unique == 0

        # Add objects to urc1 and urc2 sharing global unique_obj_store
        urc1.add([thingy(10), thingy(20), thingy(10)])
        urc2.add([thingy(20), thingy(30)])

        assert urc1.n_objects == 3
        assert urc1.n_unique == 2
        assert urc1.idx == [0, 1, 0]
        assert urc1.n_duplicates == [2, 1]

        assert urc2.n_objects == 2
        assert urc2.n_unique == 2
        assert urc2.idx == [1, 2]
        assert urc2.n_duplicates == [1, 1]

        # Check global unique_obj_store state
        store = unique_obj_store.instance()
        assert "thingy" in store.typenames
        glob = store.get_objects("thingy")
        assert glob.n_unique == 3
        assert glob.n_duplicates == [2, 2, 1]

        # Test get_unique_objects_and_indices on urc2
        u_objs, u_gidx = urc2.get_unique_objects_and_indices(get_lidx=False)
        assert u_objs == [thingy(20), thingy(30)]
        assert u_gidx == [1, 2]

        u_objs_l, u_lidx = urc2.get_unique_objects_and_indices(get_lidx=True)
        assert u_objs_l == [thingy(20), thingy(30)]
        assert u_lidx == [[0], [1]]

    def test_python_indexing_and_in_place_mutation(self):
        urc1 = unique_references_container("thingy")
        urc2 = unique_references_container("thingy")

        for i in range(1, 7):
            urc1.add(thingy(i * 100))
            urc2.add(thingy(i * 100 + 1))
        for i in range(3, 7):
            urc1.add(thingy(i * 100))
            urc2.add(thingy(i * 100 + 1))

        assert len(urc1) == 10
        assert urc1.n_unique == 6
        assert urc1[0] == thingy(100)
        assert urc2[0] == thingy(101)

        # In-place property mutation via Python [] indexing
        urc1[0].data = 222
        assert urc1[0] == thingy(222)
        assert urc2[0] == thingy(101)

        # Direct index assignment and append at end
        urc1[1] = thingy(999)
        assert urc1[1] == thingy(999)

        urc1[10] = thingy(1111)
        assert len(urc1) == 11
        assert urc1[10] == thingy(1111)

        # Slice retrieval
        first_three = urc1[:3]
        assert first_three == [thingy(222), thingy(999), thingy(300)]

    def test_serialization_roundtrip_and_set_unique_objects(self):
        urc = unique_references_container("IX_inst")
        inst1 = _make_merlin_instrument(100.0)
        inst2 = _make_merlin_instrument(250.0)
        inst3 = IX_null_inst()

        urc.add([inst1, inst2, inst1, inst3])
        assert urc.n_objects == 4
        assert urc.n_unique == 3

        # unique_objects property returns a unique_objects_container
        uoc = urc.unique_objects
        assert isinstance(uoc, unique_objects_container)
        assert uoc.n_objects == 4
        assert uoc.n_unique == 3

        # Serialize to struct, clear global store, and reconstruct
        s = urc.to_struct()
        unique_obj_store.instance().clear()
        assert unique_obj_store.instance().n_types == 0

        rec = Serializable.from_struct(s)
        assert isinstance(rec, unique_references_container)
        assert rec.baseclass == "IX_inst"
        assert rec.n_objects == 4
        assert rec.n_unique == 3
        assert rec[0] == inst1
        assert rec[1] == inst2
        assert rec[2] == inst1
        assert rec[3] == inst3

        # Setting unique_objects with a list raises NotImplementedError (from unique_only_obj_container)
        with pytest.raises(NotImplementedError, match="HERBERT:unique_only_obj_container:not_implemented"):
            rec.unique_objects = [inst1, inst2, inst3]

        # Setting unique_objects with wrong type raises ValueError
        with pytest.raises(ValueError, match="HERBERT:unique_references_container:invalid_argument"):
            rec.unique_objects = "not_a_container"

    def test_replicate_runs_and_get_subset(self):
        urc = unique_references_container("char")
        urc.add("run_file_1")
        urc.replicate_runs(4)
        assert urc.n_objects == 4
        assert urc.n_unique == 1
        assert urc.n_duplicates == [4]
        assert urc.global_container.n_duplicates == [4]

        urc[2] = "run_file_2"
        sub = urc.get_subset([0, 2, 3])
        assert isinstance(sub, unique_references_container)
        assert sub.n_objects == 3
        assert sub.n_unique == 2
        assert sub[0] == "run_file_1"
        assert sub[1] == "run_file_2"
        assert sub[2] == "run_file_1"


class TestGetUniqueFields:
    """Tests ported from _test/test_object_containers/test_get_unique_fields.m."""

    def test_get_unique_field_on_objects_and_references_containers(self):
        s1 = {"a": "a", "b": "d"}
        s2 = {"a": "a", "b": "e"}
        s3 = {"a": "b", "b": "f"}
        s4 = {"a": "b", "b": "f"}
        s5 = {"a": "c", "b": "g"}
        s6 = {"a": "b", "b": "f"}

        uoc = unique_objects_container("struct")
        for s in (s1, s2, s3, s4, s5, s6):
            uoc.add(s)
        assert uoc.n_objects == 6
        assert uoc.n_unique == 4

        sub_a = uoc.get_unique_field("a")
        assert isinstance(sub_a, unique_objects_container)
        assert sub_a.baseclass == "char"
        assert sub_a.n_objects == 6
        assert sub_a.n_unique == 3
        assert sub_a.unique_objects == ["a", "b", "c"]
        assert [sub_a[i] for i in range(6)] == ["a", "a", "b", "b", "c", "b"]

        sub_b = uoc.get_unique_field("b")
        assert isinstance(sub_b, unique_objects_container)
        assert sub_b.baseclass == "char"
        assert sub_b.n_objects == 6
        assert sub_b.n_unique == 4
        assert sub_b.unique_objects == ["d", "e", "f", "g"]

        # Test get_unique_field on unique_references_container with IX_inst_DGfermi objects
        urc = unique_references_container("IX_inst")
        inst1 = _make_merlin_instrument(100.0)
        inst2 = _make_merlin_instrument(200.0)
        urc.add([inst1, inst2, inst1])

        ap_urc = urc.get_unique_field("aperture")
        assert isinstance(ap_urc, unique_references_container)
        assert ap_urc.baseclass == "IX_aperture"
        assert ap_urc.n_objects == 3
        # Both instruments share the identical aperture!
        assert ap_urc.n_unique == 1
        assert ap_urc[0] == inst1.aperture

        fc_urc = urc.get_unique_field("fermi_chopper")
        assert isinstance(fc_urc, unique_references_container)
        assert fc_urc.baseclass == "IX_fermi_chopper"
        assert fc_urc.n_objects == 3
        # The two instruments have different Fermi chopper energies (100 vs 200)
        assert fc_urc.n_unique == 2
        assert fc_urc[0] == inst1.fermi_chopper
        assert fc_urc[1] == inst2.fermi_chopper
        assert fc_urc[2] == inst1.fermi_chopper

