"""Tests for smooth on DnD objects, ported from test_smooth.m."""

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


class TestSmoothDnD:
    @pytest.fixture(scope="class")
    def d1d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw"))

    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw"))

    def test_smooth_d0d_raises_error(self):
        w0 = d0d()
        with pytest.raises(HoraceError) as excinfo:
            w0.smooth()
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"

    def test_smooth_invalid_width_raises_error(self, d2d_obj):
        with pytest.raises(HoraceError) as excinfo:
            d2d_obj.smooth([10, 10, 10])
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "Width parameter must be a scalar or a vector with length equal to the dimensions" in str(excinfo.value)

    def test_smooth_invalid_shape_raises_error(self, d2d_obj):
        with pytest.raises(HoraceError) as excinfo:
            d2d_obj.smooth(10, "invalid_shape")
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "'invalid_shape' is not recognised as a valid smoothing shape" in str(excinfo.value)

    def test_smooth_resolution_shape_raises_error(self, d2d_obj):
        with pytest.raises(HoraceError) as excinfo:
            d2d_obj.smooth(10, "resolution")
        assert excinfo.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "'resolution' smoothing is not currently implemented" in str(excinfo.value)

    def test_smooth_d2d_matches_reference_mat(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_smooth.mat")
        mat = scipy.io.loadmat(ref_file, squeeze_me=True, struct_as_record=False)

        if "d2d_smooth_default" in mat:
            ref_def = _load_ref_dnd(ref_file, "d2d_smooth_default")
            res_def = d2d_obj.smooth()
            np.testing.assert_allclose(res_def.s, ref_def.s, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(res_def.e, ref_def.e, rtol=1e-5, atol=1e-6)

        if "d2d_smooth_gaussian" in mat:
            ref_g = _load_ref_dnd(ref_file, "d2d_smooth_gaussian")
            res_g = d2d_obj.smooth([3, 4], "gaussian")
            np.testing.assert_allclose(res_g.s, ref_g.s, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(res_g.e, ref_g.e, rtol=1e-5, atol=1e-6)

