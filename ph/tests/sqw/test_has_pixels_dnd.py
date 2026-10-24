"""Tests for has_pixels on DnD objects, ported from test_has_pixels_dnd.m."""

import os
import pytest

from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d, read_dnd, sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestHasPixelsDnD:
    def test_has_pixels_returns_false_for_empty_dnd_objects(self):
        dnd_objs = [d0d(), d1d(), d2d(), d3d(), d4d()]
        for obj in dnd_objs:
            assert obj.has_pixels() is False

    def test_has_pixels_returns_false_for_dnd_array(self):
        arr = DataOpArray([d2d(), d2d(), d2d()])
        res = [x.has_pixels() for x in arr]
        assert res == [False, False, False]

    def test_has_pixels_returns_false_for_loaded_dnd(self):
        f2d = os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw")
        w2d = read_dnd(f2d)
        assert w2d.has_pixels() is False

