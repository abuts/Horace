"""Tests for equal_to_tol on DnD objects, ported from test_equal_to_tol_dnd.m."""

import os
import numpy as np
import pytest

from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.sqw import d2d, read_dnd, sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestEqualToTolDnD:
    @pytest.fixture
    def d2d_obj(self):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        return read_dnd(f2d)

    def test_the_same_d2d_objects_are_equal(self, d2d_obj):
        dnd_copy = d2d_obj.copy()
        assert d2d_obj.equal_to_tol(dnd_copy)

    def test_d2d_and_sqw_objects_are_not_equal(self, d2d_obj):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        sqw_2d = sqw(f2d)
        ok, mess = d2d_obj.equal_to_tol(sqw_2d, return_msg=True)
        assert not ok
        assert "Objects being compared are not of the same type" in mess

    def test_not_equal_if_one_field_different(self, d2d_obj):
        dnd_copy = d2d_obj.copy()
        dnd_copy.npix = np.zeros_like(dnd_copy.npix)
        assert not d2d_obj.equal_to_tol(dnd_copy)

    def test_equal_within_absolute_tolerance(self, d2d_obj):
        dnd_copy = d2d_obj.copy()
        dnd_copy.s = d2d_obj.s + 1e-5
        assert not d2d_obj.equal_to_tol(dnd_copy, tol=[1e-8, 0.0])
        assert d2d_obj.equal_to_tol(dnd_copy, tol=[1e-4, 0.0])

    def test_equal_to_tol_for_dnd_arrays(self, d2d_obj):
        arr1 = DataOpArray([d2d_obj.copy(), d2d_obj.copy()])
        arr2 = DataOpArray([d2d_obj.copy(), d2d_obj.copy()])
        assert arr1.equal_to_tol(arr2)

        arr2[1].title = "different"
        assert not arr1.equal_to_tol(arr2)

