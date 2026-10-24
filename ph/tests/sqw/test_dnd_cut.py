"""Tests for cut on DnD objects, ported from test_dnd_cut.m."""

import os
import numpy as np
import pytest
import scipy.io

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import DnDBase, d0d, d1d, d2d, d3d, data_sqw_dnd, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)
TEST_DND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "test_dnd_class")
)


def _load_ref_dnd(mat_path: str, var_name: str) -> DnDBase:
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    raw = mat[var_name]
    if hasattr(raw, "data"):
        raw = raw.data
    return DnDBase.dnd(data_sqw_dnd(raw))


class TestDnDCut:
    @pytest.fixture(scope="class")
    def d1d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw"))

    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw"))

    @pytest.fixture(scope="class")
    def d3d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"))

    def test_cut_d1d_matches_reference(self, d1d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "d1d_cut")
        res = d1d_obj.cut([0.4, 0.03, 0.7])
        assert isinstance(res, d1d)
        np.testing.assert_allclose(res.s, ref.s, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.e, ref.e, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.npix, ref.npix, rtol=1e-6, atol=1e-6)

    def test_cut_d2d_to_d1d_matches_reference(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "d2d_cut")
        res = d2d_obj.cut([-0.4, 0.05, 0.5], [10.0, 20.0])
        assert isinstance(res, d1d)
        np.testing.assert_allclose(res.s, ref.s, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.e, ref.e, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.npix, ref.npix, rtol=1e-6, atol=1e-6)

    def test_cut_d3d_to_d2d_matches_reference(self, d3d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "d3d_cut")
        res = d3d_obj.cut([-0.8, 0.1, 0.8], [-1.0, 0.5], [-0.8, 0.1, 0.8])
        assert isinstance(res, d2d)
        np.testing.assert_allclose(res.s, ref.s, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.e, ref.e, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(res.npix, ref.npix, rtol=1e-6, atol=1e-6)

    def test_cut_invalid_number_of_arguments_raises_error(self, d2d_obj):
        with pytest.raises(HoraceError):
            d2d_obj.cut([0, 0.1, 1])

