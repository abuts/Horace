"""Tests for copying DnD objects, ported from test_dnd_copy.m."""

import copy
import os
import numpy as np
import pytest

from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestDnDCopy:
    @pytest.mark.parametrize("cls", [d0d, d1d, d2d, d3d, d4d])
    def test_empty_dnd_copy_is_independent(self, cls):
        obj = cls()
        cp = obj.copy()
        assert obj.equal_to_tol(cp)
        cp.title = "Modified title"
        assert obj.title != cp.title
        assert not obj.equal_to_tol(cp)

    def test_d2d_copy_from_file_is_deep_copy(self):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        w2d = d2d(f2d)
        w2d_copy = copy.deepcopy(w2d)

        assert w2d.equal_to_tol(w2d_copy)

        # Mutate signal on copy and verify original is unchanged
        orig_s = w2d.s.copy()
        s_mod = w2d_copy.s
        s_mod[0, 0] += 999.0
        w2d_copy.s = s_mod
        np.testing.assert_array_equal(w2d.s, orig_s)
        assert not w2d.equal_to_tol(w2d_copy)

        # Mutate projection/axes property on copy and verify original is unchanged
        orig_alatt = w2d.alatt.copy()
        w2d_copy.alatt = [5.0, 6.0, 7.0]
        np.testing.assert_array_almost_equal(w2d.alatt, orig_alatt)

