"""Unit tests for hashable base class and build_hash / isequal functions.

Translated from MATLAB _test/test_object_containers/test_hashable_methods.m.
"""

from __future__ import annotations

import copy
import uuid
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest
from pydantic import Field

from pyhorace.core.hashable import (
    Hashable,
    HashableArray,
    build_hash,
    hashable,
    isequal,
)


class IX_experiment(hashable):
    """
    Test helper class matching MATLAB's IX_experiment subclass of hashable
    used in test_hashable_methods.m.
    """

    filename: str = ""
    filepath: str = ""
    run_id: float = float("nan")
    efix: float = 0.0
    emode: int = 0
    en: np.ndarray = Field(default_factory=lambda: np.zeros((0, 1), dtype=np.float64))
    psi: float = float("nan")
    cu: np.ndarray = Field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    cv: np.ndarray = Field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    omega: float = float("nan")
    dpsi: float = float("nan")
    gl: float = float("nan")
    gs: float = float("nan")
    angular_units: str = "deg"

    def class_version(self) -> int:
        return 3

    def saveable_fields(self) -> Sequence[str]:
        return [
            "filename",
            "filepath",
            "run_id",
            "efix",
            "emode",
            "en",
            "psi",
            "cu",
            "cv",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        ]

    def hashable_fields(self) -> Sequence[str]:
        # Matches IX_experiment.m: excludes run_id and filepath
        return [
            "filename",
            "efix",
            "emode",
            "en",
            "psi",
            "cu",
            "cv",
            "omega",
            "dpsi",
            "gl",
            "gs",
        ]


def build_IX_array(
    n_elements: int,
) -> Tuple[Union[IX_experiment, HashableArray], np.ndarray]:
    """
    Build 1 or more randomized IX_experiment objects matching
    test_hashable_methods.build_IX_array(n_elements) in MATLAB.
    """
    rng = np.random.default_rng()
    par_names = [
        "filename",
        "run_id",
        "efix",
        "en",
        "psi",
        "omega",
        "dpsi",
        "gl",
        "gs",
    ]
    par_val = [
        "my_file",
        6666,
        10.0,
        np.array([[1.0], [2.0], [4.0], [8.0]]),
        70.0,
        5.0,
        5.0,
        5.0,
        5.0,
    ]

    items: List[IX_experiment] = []
    for _ in range(n_elements):
        expd = IX_experiment()
        expd.do_check_combo_arg = False
        for name, pval in zip(par_names, par_val):
            if isinstance(pval, str):
                val = f"nxspe_file_{uuid.uuid4().hex}"
            elif isinstance(pval, np.ndarray) and pval.size > 1:
                expd.efix = expd.efix + 5.0
                val = np.sort(rng.random(pval.shape) * expd.efix, axis=0)
            else:
                val = float(np.round(rng.random() * float(pval)))
            setattr(expd, name, val)

        expd.filepath = "some_file_path"
        expd.emode = 1
        expd.do_check_combo_arg = True
        expd = expd.check_combo_arg()
        items.append(expd)

    run_id = np.array([x.run_id for x in items], dtype=np.float64)
    uniq_id = np.unique(run_id)
    was_nonunique = False
    while uniq_id.size != run_id.size:
        was_nonunique = True
        run_id = np.round(rng.random(n_elements) * float(par_val[1]))
        uniq_id = np.unique(run_id)

    if was_nonunique:
        for i, item in enumerate(items):
            item.run_id = float(run_id[i])

    if n_elements == 1:
        return items[0], run_id
    return HashableArray(items), run_id


# =============================================================================
# Tests translated from _test/test_object_containers/test_hashable_methods.m
# =============================================================================


class TestHashableMethods:
    """Test suite translated from test_hashable_methods.m."""

    def test_exposing_hash_array(self):
        data, _ = build_IX_array(10)
        data, hashes, is_new = build_hash(data)
        assert isinstance(hashes, list)
        assert is_new

        hash_defined = [x.hash_defined for x in data]
        assert all(hash_defined)

        S = data.to_struct()

        assert "hash_value" in S
        assert isinstance(S["hash_value"], list)

        is_empty = [not bool(x) for x in S["hash_value"]]
        assert not any(is_empty)

        rec_data = hashable.from_struct(S)

        assert data == rec_data
        hash_defined = [x.hash_defined for x in rec_data]
        assert all(hash_defined)

    def test_exposing_hash_value(self):
        data, _ = build_IX_array(1)
        data, hash_val, is_new = build_hash(data)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 32
        assert is_new

        assert data.hash_defined

        S = data.to_struct()

        assert "hash_value" in S
        assert isinstance(S["hash_value"], list)

        is_empty = [not bool(x) for x in S["hash_value"]]
        assert not any(is_empty)

        rec_data = hashable.from_struct(S)

        assert data == rec_data
        assert rec_data.hash_defined

    def test_exposing_empty_hash_array(self):
        data, _ = build_IX_array(10)

        hash_defined = [x.hash_defined for x in data]
        assert not any(hash_defined)

        S = data.to_struct()

        assert "hash_value" in S
        assert isinstance(S["hash_value"], list)

        is_empty = [not bool(x) for x in S["hash_value"]]
        assert all(is_empty)

        rec_data = hashable.from_struct(S)

        assert data == rec_data
        assert not any(x.hash_defined for x in rec_data)

    def test_exposing_empty_hash_value(self):
        data, _ = build_IX_array(1)

        assert not data.hash_defined

        S = data.to_struct()

        assert "hash_value" in S
        assert isinstance(S["hash_value"], list)

        is_empty = [not bool(x) for x in S["hash_value"]]
        assert all(is_empty)

        rec_data = hashable.from_struct(S)

        assert data == rec_data
        assert not rec_data.hash_defined

    # -------------------------------------------------------------------------
    # isequal tests
    # -------------------------------------------------------------------------

    def test_is_equal_four(self):
        data, _ = build_IX_array(4)
        data[1] = copy.deepcopy(data[0])
        data[2], _, _ = build_hash(data[0])
        data[3], _, _ = build_hash(data[0])

        assert not data[0].hash_defined
        assert not data[1].hash_defined
        assert data[2].hash_defined
        assert data[3].hash_defined

        assert isequal(data[0], data[1], data[2], data[3])
        assert data[0].isequal(data[1], data[2], data[3])

    def test_is_equal_both_hashes(self):
        data, _ = build_IX_array(1)
        assert not data.hash_defined
        data1, _, _ = build_hash(data)
        data, _, _ = build_hash(data)

        assert data.hash_defined
        assert data1.hash_defined
        assert isequal(data, data1)

    def test_is_equal_first_nohash_second_hash(self):
        data, _ = build_IX_array(1)
        assert not data.hash_defined
        data1, _, _ = build_hash(data)

        assert not data.hash_defined
        assert data1.hash_defined
        assert isequal(data, data1)

    def test_is_equal_first_hash_second_no_hash(self):
        data, _ = build_IX_array(1)
        assert not data.hash_defined
        data1, _, _ = build_hash(data)

        assert data1.hash_defined
        assert not data.hash_defined
        assert isequal(data1, data)

    def test_is_equal_both_no_hash(self):
        data, _ = build_IX_array(1)
        assert not data.hash_defined
        data1 = copy.deepcopy(data)

        assert not data1.hash_defined
        assert isequal(data, data1)


# =============================================================================
# Additional Unit Tests for hashable, HashableArray, and build_hash
# =============================================================================


def test_build_hash_caching_and_invalidation():
    exp, _ = build_IX_array(1)
    assert not exp.hash_defined
    assert exp.hash_value is None

    # First call computes the hash
    exp_hashed, h1, is_new1 = build_hash(exp)
    assert is_new1
    assert exp_hashed.hash_defined
    assert exp_hashed.hash_value == h1
    assert len(h1) == 32

    # Second call returns cached hash with is_new == False
    exp_hashed2, h2, is_new2 = build_hash(exp_hashed)
    assert not is_new2
    assert h2 == h1

    # Modifying a non-hashable field (run_id, filepath) preserves cached hash
    exp_hashed2.run_id = 99999.0
    exp_hashed2.filepath = "/new/path"
    assert exp_hashed2.hash_defined
    assert exp_hashed2.hash_value == h1

    # Modifying a hashable field (efix) automatically invalidates cached hash
    exp_hashed2.efix = exp_hashed2.efix + 100.0
    assert not exp_hashed2.hash_defined
    assert exp_hashed2.hash_value is None

    # Rebuilding produces a different MD5 hash
    exp_hashed3, h3, is_new3 = build_hash(exp_hashed2)
    assert is_new3
    assert h3 != h1


def test_eq_ne_and_python_hash():
    exp1, _ = build_IX_array(1)
    exp2 = copy.deepcopy(exp1)
    # Different run_id (not in hashable_fields) should still be equal!
    exp2.run_id = exp1.run_id + 1234.0

    assert exp1 == exp2
    assert exp1.eq(exp2)
    assert not (exp1 != exp2)
    assert not exp1.ne(exp2)
    assert hash(exp1) == hash(exp2)

    # Can be used in a set/dict
    s = {exp1, exp2}
    assert len(s) == 1

    # Changing a hashable field makes them unequal
    exp2.efix = exp1.efix + 50.0
    assert exp1 != exp2
    assert exp1.ne(exp2)
    assert not (exp1 == exp2)


def test_nested_hashable_objects():
    class OuterHashable(Hashable):
        name: str = "outer"
        inner: Optional[IX_experiment] = None
        items: List[IX_experiment] = Field(default_factory=list)

    inner_exp, _ = build_IX_array(1)
    arr_exp, _ = build_IX_array(2)
    outer = OuterHashable(name="container", inner=inner_exp, items=list(arr_exp))

    assert not outer.hash_defined
    assert not outer.inner.hash_defined

    outer_hashed, h, is_new = outer.build_hash()
    assert is_new
    assert outer_hashed.hash_defined
    assert len(h) == 32
    # Nested hashable objects also have their hashes populated on outer_hashed
    assert outer_hashed.inner.hash_defined
    assert all(x.hash_defined for x in outer_hashed.items)


def test_build_hash_standalone_types():
    # Bytes
    raw = b"hello horace md5"
    _, h_bytes, is_calc = build_hash(raw)
    assert is_calc
    assert len(h_bytes) == 32

    # uint8 numpy array with same bytes produces identical MD5
    arr = np.frombuffer(raw, dtype=np.uint8)
    _, h_arr, _ = build_hash(arr)
    assert h_arr == h_bytes

    # Dictionary
    d1 = {"a": 1, "b": [1, 2, 3]}
    d2 = {"b": [1, 2, 3], "a": 1}
    _, hd1, _ = build_hash(d1)
    _, hd2, _ = build_hash(d2)
    assert hd1 == hd2

