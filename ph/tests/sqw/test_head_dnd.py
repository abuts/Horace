"""Tests for head on DnD objects, ported from test_head_dnd.m."""

import os
import numpy as np
import pytest
import scipy.io

from pyhorace.sqw import DnDBase, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)
TEST_DND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "test_dnd_class")
)


class TestHeadDnD:
    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw"))

    def test_head_returns_expected_fields(self, d2d_obj):
        h = d2d_obj.head(nargout=1)
        assert isinstance(h, dict)
        h_fields, d_fields = DnDBase.head_form(False)
        for f in h_fields + d_fields:
            assert f in h
        assert "s" not in h
        assert "e" not in h
        assert "npix" not in h
        np.testing.assert_allclose(h["alatt"], d2d_obj.alatt)
        np.testing.assert_allclose(h["angdeg"], d2d_obj.angdeg)

    def test_head_full_returns_data_arrays(self, d2d_obj):
        h_full = d2d_obj.head("-full", nargout=1)
        assert isinstance(h_full, dict)
        h_fields, d_fields = DnDBase.head_form(True)
        for f in h_fields + d_fields:
            assert f in h_full
        np.testing.assert_allclose(h_full["s"], d2d_obj.s)
        np.testing.assert_allclose(h_full["e"], d2d_obj.e)
        np.testing.assert_allclose(h_full["npix"], d2d_obj.npix)

