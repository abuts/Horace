"""Unit tests for Serializable base class.

Translated from MATLAB _test/test_serializers/test_serializable_class.m and
_test/test_serializers/test_serializable_class_2.m.
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import List

import numpy as np
import pytest

from pyhorace.core.serializable import Serializable
from tests.serializable_testers import (
    SerializableTester1,
    SerializableTester2,
    SerializableTester3,
    SerializableTester4SetKeyValConstructor,
    SerializableTesterWithInterdepProp,
    make_tester2,
)


@pytest.fixture(autouse=True)
def reset_tester_versions():
    """Reset static/persistent version numbers before each test."""
    SerializableTester1.ver_holder(2)
    SerializableTester2.version_holder(1)
    SerializableTester3.version_holder(2)
    yield
    SerializableTester1.ver_holder(2)
    SerializableTester2.version_holder(1)
    SerializableTester3.version_holder(2)


# =============================================================================
# Tests from test_serializable_class.m
# =============================================================================


def test_partial_match_works():
    tob = SerializableTesterWithInterdepProp(
        0.5, 1, 2, "partial_match_1", 20, "partial_match_3", 11
    )
    assert tob.Prop_class2_1 == 0.5
    assert tob.Prop_class2_2 == 1
    assert tob.Prop_class2_3 == 2
    assert tob.partial_match_1_blue == 20
    assert tob.partial_match_2_green is None
    assert tob.partial_match_3_yellow == 11


def test_partial_match_multi_throw():
    with pytest.raises(ValueError, match="non-uniquely define more then one"):
        SerializableTesterWithInterdepProp(
            10, 1, 0, "partial_match", 20, "partial_match", 11
        )


def test_right_interdep_prop_pass():
    tob = SerializableTesterWithInterdepProp(10, 1, 0, "Prop_class2_2", 20)
    assert tob.Prop_class2_1 == 10
    assert tob.Prop_class2_2 == 20
    assert tob.Prop_class2_3 == 0


def test_eq_operator_level1_ne():
    tc1 = SerializableTester2(1, list(range(1, 21)), 3)
    tc2 = SerializableTester2(1, 2, 3)

    assert not (tc1 == tc2)
    assert not tc1.eq(tc2)


def test_eq_operator_level2():
    tc1 = SerializableTester2(1, 2, SerializableTester1())
    tc2 = SerializableTester2(1, 2, SerializableTester1())

    assert tc1 == tc2
    assert tc1.eq(tc2)


def test_eq_operator_level1_with_mess():
    tc1 = SerializableTester2(1, list(range(1, 21)), 3)
    tc2 = SerializableTester2(1, list(range(1, 21)), 3)

    iseq, mess = tc1.equal_to_tol(tc2)
    assert iseq
    assert mess == ""


def test_eq_operator_level1():
    tc1 = SerializableTester2(1, 2, 3)
    tc2 = SerializableTester2(1, 2, 3)

    assert tc1 == tc2
    assert tc1.eq(tc2)


def test_wrong_interdep_prop_fail_differently():
    tob = SerializableTesterWithInterdepProp(0, 1, 0)
    with pytest.raises(ValueError, match="inconsistent interdependent properties"):
        tob.Prop_class2_1 = 10


def test_wrong_interdep_prop_fail():
    tob = SerializableTesterWithInterdepProp()
    with pytest.raises(ValueError, match="inconsistent interdependent properties"):
        tob.Prop_class2_1 = 10


def test_ser_serializable_obj_array_class2():
    ser_cl: List[SerializableTester1] = []
    set_cl2 = SerializableTester2()
    set_cl2.Prop_class2_1 = 10
    set_cl2.Prop_class2_2 = 20

    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [copy.deepcopy(set_cl2) for _ in range(2 * i)]
        ser_cl.append(item)

    # Serialize using Python Serializable
    raw_bytes = Serializable.serialize_array(ser_cl)
    ser_cl_rec, nbytes = Serializable.deserialize(raw_bytes)

    assert isinstance(ser_cl_rec, list)
    assert len(ser_cl_rec) == len(ser_cl)
    for orig, rec in zip(ser_cl, ser_cl_rec):
        assert orig == rec


def test_ser_serializable_obj_array_class1_obj_class2():
    ser_cl: List[SerializableTester1] = []
    for i in range(1, 5):
        set_cl2 = SerializableTester2()
        set_cl2.Prop_class2_1 = 5 * i
        set_cl2.Prop_class2_2 = 20 * i
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = set_cl2
        ser_cl.append(item)

    raw_bytes = Serializable.serialize_array(ser_cl)
    ser_cl_rec, nbytes = Serializable.deserialize(raw_bytes)

    assert isinstance(ser_cl_rec, list)
    assert len(ser_cl_rec) == len(ser_cl)
    for orig, rec in zip(ser_cl, ser_cl_rec):
        assert orig == rec


def test_ser_serializable_obj_array_class1():
    ser_cl: List[SerializableTester1] = []
    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [None] * i
        ser_cl.append(item)

    raw_bytes = Serializable.serialize_array(ser_cl)
    ser_cl_rec, nbytes = Serializable.deserialize(raw_bytes)

    assert isinstance(ser_cl_rec, list)
    assert len(ser_cl_rec) == len(ser_cl)
    for orig, rec in zip(ser_cl, ser_cl_rec):
        assert orig == rec


def test_ser_serializable_obj():
    ser_cl = SerializableTester2()
    ser_cl.Prop_class2_1 = 100
    ser_cl.Prop_class2_2 = SerializableTester1()

    raw_bytes = ser_cl.serialize()
    assert ser_cl.serial_size() == len(raw_bytes)

    ser_cl_rec, nbytes = Serializable.deserialize(raw_bytes)
    assert nbytes == len(raw_bytes)
    assert ser_cl == ser_cl_rec
    assert isinstance(ser_cl_rec.Prop_class2_2, type(ser_cl.Prop_class2_2))


def test_ser_serializable_obj_level0():
    ser_cl = SerializableTester2()
    ser_cl.Prop_class2_1 = 100
    ser_cl.Prop_class2_2 = [1, 2, 4]

    raw_bytes = ser_cl.serialize()
    assert ser_cl.serial_size() == len(raw_bytes)

    ser_cl_rec, nbytes = Serializable.deserialize(raw_bytes)
    assert nbytes == len(raw_bytes)
    assert ser_cl == ser_cl_rec
    assert isinstance(ser_cl_rec.Prop_class2_2, type(ser_cl.Prop_class2_2))


def test_saveobj_old_version_loadobj_new_version():
    tc = SerializableTester2()
    tc2 = SerializableTester1()
    tc2.ver_holder(1)
    assert tc2.class_version() == 1

    tc.Prop_class2_1 = 10
    tc.Prop_class2_2 = [copy.deepcopy(tc2), copy.deepcopy(tc2)]

    tc_struct = tc.saveobj()

    # Now rebuild to have new version (version 2)
    SerializableTester1.ver_holder(2)
    tc_rec = SerializableTester2.loadobj(tc_struct)

    assert isinstance(tc_rec, SerializableTester2)
    tc2_lev2 = tc_rec.Prop_class2_2
    assert len(tc2_lev2) == 2

    assert tc2_lev2[0].class_version() == 2
    assert tc2_lev2[0].Prop_class1_3 == "recovered_new_from_old_value"

    assert tc2_lev2[1].class_version() == 2
    assert tc2_lev2[1].Prop_class1_3 == "recovered_new_from_old_value"


def test_new_version_saveobj_loadobj_array_recursive():
    tc_list: List[SerializableTester2] = []
    tc2 = SerializableTester1()
    for i in range(1, 5):
        item = SerializableTester2()
        item.Prop_class2_1 = i * 10
        item.Prop_class2_2 = [copy.deepcopy(tc2) for _ in range(2 * i)]
        tc_list.append(item)

    tc_struct = Serializable.to_struct_array(tc_list)

    # Bump version
    ver = SerializableTester2.version_holder()
    SerializableTester2.version_holder(ver + 1)

    tc_rec = Serializable.from_struct(tc_struct, SerializableTester2())
    assert isinstance(tc_rec, list)
    assert len(tc_rec) == len(tc_list)
    for orig, rec in zip(tc_list, tc_rec):
        assert orig == rec


def test_serialize_classes_array_recursive():
    tc_list: List[SerializableTester1] = []
    tc2 = SerializableTester2()
    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [copy.deepcopy(tc2) for _ in range(2 * i)]
        tc_list.append(item)

    tc_bytes = Serializable.serialize_array(tc_list)
    tc_size = len(tc_bytes)

    tc_rec, nbytes = Serializable.deserialize(tc_bytes)
    assert isinstance(tc_rec, list)
    assert len(tc_rec) == len(tc_list)
    assert tc_size == nbytes
    for orig, rec in zip(tc_list, tc_rec):
        assert orig == rec
        assert type(orig) is type(rec)
        assert type(orig.Prop_class1_2[0]) is type(rec.Prop_class1_2[0])


def test_to_from_to_struct_classes_array_recursive():
    tc_list: List[SerializableTester1] = []
    tc2 = SerializableTester2()
    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [copy.deepcopy(tc2) for _ in range(2 * i)]
        tc_list.append(item)

    tc_struct = Serializable.to_struct_array(tc_list)
    tc_rec = Serializable.from_struct(tc_struct)
    assert isinstance(tc_rec, list)
    assert len(tc_rec) == len(tc_list)
    for orig, rec in zip(tc_list, tc_rec):
        assert orig == rec


def test_serialize_classes_array():
    tc_list: List[SerializableTester1] = []
    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [None] * (2 * i)
        tc_list.append(item)

    tc_bytes = Serializable.serialize_array(tc_list)
    tc_rec, nbytes = Serializable.deserialize(tc_bytes)
    assert isinstance(tc_rec, list)
    assert len(tc_rec) == len(tc_list)
    for orig, rec in zip(tc_list, tc_rec):
        assert orig == rec


def test_to_from_to_struct_classes_array():
    tc_list: List[SerializableTester1] = []
    for i in range(1, 5):
        item = SerializableTester1()
        item.Prop_class1_1 = i * 10
        item.Prop_class1_2 = [None] * (2 * i)
        tc_list.append(item)

    tc_struct = Serializable.to_struct_array(tc_list)
    tc_rec = Serializable.from_struct(tc_struct)
    assert isinstance(tc_rec, list)
    assert len(tc_rec) == len(tc_list)
    for orig, rec in zip(tc_list, tc_rec):
        assert orig == rec


def test_serialize_native_single_class():
    tc = SerializableTester1()
    tc.Prop_class1_1 = 20
    tc.Prop_class1_2 = [None] * 10

    tc_bytes = tc.serialize()
    tc_size = tc.serial_size()
    assert len(tc_bytes) == tc_size

    tc_rec, nbytes = Serializable.deserialize(tc_bytes)
    assert tc == tc_rec
    assert tc_size == nbytes


def test_to_from_to_struct_single_class():
    tc = SerializableTester1()
    tc.Prop_class1_1 = 20
    tc.Prop_class1_2 = [None] * 10
    tc_struct = tc.to_struct()

    tc_rec = Serializable.from_struct(tc_struct)
    assert tc == tc_rec


def test_eq_false_with_message():
    tc = SerializableTester1()
    tc.Prop_class1_1 = 20
    tc.Prop_class1_2 = [None] * 10

    tc1 = copy.deepcopy(tc)
    tc1.Prop_class1_1 = 10

    iseq, mess = tc.equal_to_tol(tc1, name_a="tc", name_b="tc1")
    assert not iseq
    assert mess == (
        "tc.Prop_class1_1 and tc1.Prop_class1_1: Not all elements are equal; max. error = 10 at element (1)"
    )


# -----------------------------------------------------------------------------
# Constructor argument parsing tests
# -----------------------------------------------------------------------------


def test_pos_constructor_char_pos_sets_key():
    tc, rem = make_tester2(
        11, 20, "Prop_class2_3", "aaa", "Prop_class2_2", 30, "blabla"
    )
    assert tc.Prop_class2_1 == 11
    assert tc.Prop_class2_2 == 30
    assert tc.Prop_class2_3 == "aaa"
    assert rem == ["blabla"]


def test_keyval_constructor_nokey_throws_at_the_end():
    with pytest.raises(ValueError, match="should be even number of key-value pairs"):
        make_tester2("Prop_class2_1", "a", "Prop_class2_2")


def test_keyval_constructor_nokey_return_remains():
    st, remains = make_tester2(
        "Prop_class2_1", 10, "Prop_class2_2", 20, "blabla"
    )
    assert st.Prop_class2_1 == 10
    assert st.Prop_class2_2 == 20
    assert remains == ["blabla"]


def test_deprecated_keys_provided():
    with pytest.deprecated_call():
        tc = SerializableTesterWithInterdepProp(
            "Prop_class2_1",
            2,
            "-Prop_class2_2",
            10,
            "-Prop_class2_3",
            [1, 2, 3],
        )

    assert tc.Prop_class2_1 == 2
    assert tc.Prop_class2_2 == 10
    assert tc.Prop_class2_3 == [1, 2, 3]


def test_val_keyval_constructor_returns_keyval_remaining():
    tc, rem = make_tester2(
        11, "Prop_class2_1", 10, "blabla", "Prop_class2_2", 20
    )
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert tc.Prop_class2_3 is None
    assert rem == ["blabla"]


def test_keyval_constructor_middle_extra_val_ignored():
    tc, rem = make_tester2(
        "Prop_class2_1", 10, "blabla", "Prop_class2_2", 20
    )
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert rem == ["blabla"]


def test_keyval_constructor_last_extra_val_ignored():
    tc, rem = make_tester2(
        "Prop_class2_1", 10, "Prop_class2_2", 20, "blabla"
    )
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert rem == ["blabla"]


def test_keyval_constructor_first_pos_reset_later():
    tc, rem = make_tester2(
        "blabla", "Prop_class2_1", 10, "Prop_class2_2", 20
    )
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert len(rem) == 0


def test_keyval_constructor():
    tc, rem = make_tester2("Prop_class2_1", 10, "Prop_class2_2", 20)
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert len(rem) == 0


def test_val_constructor():
    tc, rem = make_tester2(10, 20)
    assert tc.Prop_class2_1 == 10
    assert tc.Prop_class2_2 == 20
    assert len(rem) == 0


def test_key_val_constructor_mix_dash():
    tc = SerializableTester4SetKeyValConstructor(
        True, "a", "prop3", "a", "prop2", "b"
    )
    assert tc.prop1_char == "a"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "a"


def test_key_val_constructor_m_keys_dash():
    with pytest.deprecated_call():
        tc = SerializableTester4SetKeyValConstructor(
            True,
            "-prop3_char",
            "a",
            "-prop2_char",
            "b",
            "-prop1",
            "c",
        )
    assert tc.prop1_char == "c"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "a"


def test_key_val_constructor_keys_dash():
    tc = SerializableTester4SetKeyValConstructor(
        True, "prop3_char", "a", "prop2_char", "b", "prop1", "c"
    )
    assert tc.prop1_char == "c"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "a"


def test_key_val_constructor_mix_no_dash():
    tc = SerializableTester4SetKeyValConstructor(
        False, "a", "prop3", "a", "prop2", "b"
    )
    assert tc.prop1_char == "a"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "a"


def test_two_keys_in_a_row_throw():
    with pytest.raises(
        ValueError, match="should be even number of key-value pairs"
    ):
        SerializableTester4SetKeyValConstructor(
            False, "b", "prop3_char", "c", "prop2_char"
        )


def test_key_val_constructor_m_keys_no_dash_throw():
    with pytest.raises(
        ValueError, match="unrecognized property provided as input"
    ):
        SerializableTester4SetKeyValConstructor(
            False, "-prop3", "a", "-prop2", "b", "-prop1", "c"
        )


def test_key_val_constructor_keys_no_dash():
    tc = SerializableTester4SetKeyValConstructor(
        False, "prop3_char", "a", "prop2_char", "b", "prop1", "c"
    )
    assert tc.prop1_char == "c"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "a"


def test_key_val_constructor_keys_no_dash_throws():
    with pytest.raises(ValueError, match="More positional arguments"):
        SerializableTester4SetKeyValConstructor(
            False, "prop3", "a", "prop2", "b", "prop1", "c"
        )


def test_key_val_constructor_values_no_dash():
    tc = SerializableTester4SetKeyValConstructor(False, "a", "b", "c")
    assert tc.prop1_char == "a"
    assert tc.prop2_char == "b"
    assert tc.prop3_char == "c"


# =============================================================================
# Tests from test_serializable_class_2.m
# =============================================================================


def _construct_detectors() -> List[SerializableTester3]:
    dias = [0.0254, 0.0300, 0.0400, 0.0400, 0.0400]
    heights = [0.015, 0.025, 0.035, 0.035, 0.035]
    walls = [6.35e-4, 10.0e-4, 15.0e-4, 15.0e-4, 15.0e-4]
    atmss = [10.0, 6.0, 4.0, 7.0, 9.0]
    return [
        SerializableTester3(dia=d, height=h, wall=w, atms=a)
        for d, h, w, a in zip(dias, heights, walls, atmss)
    ]


def test_to_bare_struct_1():
    dets_ref = _construct_detectors()

    S = dets_ref[0].to_struct()
    Sbare = dets_ref[0].to_bare_struct()

    det_1 = Serializable.from_struct(S)
    det_1_bare = SerializableTester3().from_bare_struct(Sbare)

    assert dets_ref[0] == det_1
    assert dets_ref[0] == det_1_bare


def test_to_bare_struct_2():
    dets_ref = _construct_detectors()

    S = Serializable.to_struct_array(dets_ref)
    Sbare = [d.to_bare_struct() for d in dets_ref]

    dets = Serializable.from_struct(S, SerializableTester3())
    dets_bare = SerializableTester3().from_bare_struct(Sbare)

    assert dets_ref == dets
    assert dets_ref == dets_bare


def test_ver1_save_load():
    dets_ref = _construct_detectors()

    # Save detector array as ver 1
    SerializableTester3.version_holder(1)
    dets_ver1 = _construct_detectors()

    # Check saveable fields for ver 1
    s_ver1 = dets_ver1[0].to_bare_struct()
    assert set(s_ver1.keys()) == {"dia", "height", "atms"}

    # Struct representation with ver 1
    struct_ver1 = Serializable.to_struct_array(dets_ver1)

    # Return class version to 2
    SerializableTester3.version_holder(2)

    # Recover detector array as ver 2
    recovered = Serializable.from_struct(struct_ver1, SerializableTester3())

    assert isinstance(recovered, list)
    for i in range(len(dets_ref)):
        expected = copy.deepcopy(dets_ref[i])
        expected.wall = 1e-6
        assert recovered[i] == expected


def test_ver1_reserialize():
    SerializableTester3.version_holder(2)
    dets_ref = _construct_detectors()

    # Serialize detector array as ver 1
    SerializableTester3.version_holder(1)
    dets_ver1 = _construct_detectors()
    byte_array = Serializable.serialize_array(dets_ver1)

    # Return to ver 2
    SerializableTester3.version_holder(2)

    dets_ver1_rec, _ = Serializable.deserialize(byte_array)
    assert isinstance(dets_ver1_rec, list)

    for i in range(len(dets_ref)):
        expected = copy.deepcopy(dets_ref[i])
        expected.wall = 1e-6
        assert dets_ver1_rec[i] == expected


def test_verNaN_from_struct():
    dets_ref = _construct_detectors()

    bare_structs = []
    for d in dets_ref:
        bare_structs.append({"dia": d.dia, "height": d.height})

    # Recover detector array from version NaN structure
    recovered = Serializable.from_struct(bare_structs, SerializableTester3())
    assert isinstance(recovered, list)

    for i in range(len(dets_ref)):
        expected = copy.deepcopy(dets_ref[i])
        expected.wall = 1e-7
        expected.atms = 27.0
        assert recovered[i] == expected


# =============================================================================
# Additional Edge Cases and Error Paths
# =============================================================================


def test_equal_to_tol_different_classes():
    t1 = SerializableTester1()
    t2 = SerializableTester2()
    iseq, mess = t1.equal_to_tol(t2)
    assert not iseq
    assert "different classes" in mess


def test_equal_to_tol_with_tolerances():
    t1 = SerializableTester1()
    t1.Prop_class1_1 = 10.001
    t2 = SerializableTester1()
    t2.Prop_class1_1 = 10.000

    # Fails with zero tolerance
    iseq, _ = t1.equal_to_tol(t2, tol=(0.0, 0.0))
    assert not iseq

    # Passes with atol = 0.01
    iseq, _ = t1.equal_to_tol(t2, tol=(0.01, 0.0))
    assert iseq

    # Passes with rtol = 0.001
    iseq, _ = t1.equal_to_tol(t2, tol=(0.0, 0.001))
    assert iseq


def test_equal_to_tol_nan_handling():
    t1 = SerializableTester1()
    t1.Prop_class1_1 = float("nan")
    t2 = SerializableTester1()
    t2.Prop_class1_1 = float("nan")

    # When nan_equal is True (default)
    assert t1 == t2
    iseq, _ = t1.equal_to_tol(t2, nan_equal=True)
    assert iseq

    # When nan_equal is False
    iseq, mess = t1.equal_to_tol(t2, nan_equal=False)
    assert not iseq
    assert "NaN" in mess


def test_to_struct_protected_names_error():
    class BadSerializable(Serializable):
        def class_version(self) -> int:
            return 1

        def saveable_fields(self):
            return ["serial_name", "normal_field"]

    bad = BadSerializable()
    bad.serial_name = "test"
    bad.normal_field = 123
    with pytest.raises(ValueError, match="protected names"):
        bad.to_struct()


def test_from_struct_missing_serial_name():
    with pytest.raises(ValueError, match="to_struct"):
        Serializable.from_struct({"version": 1, "field": 10})


def test_from_struct_unknown_class():
    with pytest.raises(ValueError, match="Unknown serializable class"):
        Serializable.from_struct({"serial_name": "NonExistentClass123", "version": 1})


def test_deserialize_with_offset():
    tc = SerializableTester1(prop1=42, prop2=84, prop3="offset_test")
    raw = tc.serialize()
    padded_bytes = b"HEADER_PADDING_BYTES_" + raw

    recovered, nbytes = Serializable.deserialize(
        padded_bytes, pos=len(b"HEADER_PADDING_BYTES_")
    )
    assert recovered == tc
    assert nbytes == len(raw)


def test_to_from_dict_aliases():
    tc = SerializableTester1(prop1=100, prop2=200, prop3="dict_test")
    d = tc.to_dict()
    assert d["serial_name"] == "SerializableTester1"
    assert d["Prop_class1_1"] == 100

    rec = Serializable.from_dict(d)
    assert rec == tc


def test_mandatory_properties_validation():
    t = SerializableTester2()
    # Mandatory props: Prop_class2_1 and Prop_class2_2
    opts = {"key_dash": False, "mandatory_props": [True, True, False]}

    # Missing Prop_class2_2 should fail
    with pytest.raises(ValueError, match="mandatory properties have not been provided"):
        t.set_positional_and_key_val_arguments(
            t.saveable_fields(), opts, 10
        )

    # Providing both should succeed
    t2 = SerializableTester2()
    t2.set_positional_and_key_val_arguments(
        t2.saveable_fields(), opts, 10, 20
    )
    assert t2.Prop_class2_1 == 10
    assert t2.Prop_class2_2 == 20

