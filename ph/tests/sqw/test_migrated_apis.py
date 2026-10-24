"""Tests for migrated DnD APIs, ported from test_migrated_apis.m."""

import os
import numpy as np
import pytest
import scipy.io

from pyhorace.coord_transf._dependencies import crystal_alignment_info
from pyhorace.sqw import DnDBase, d1d, d2d, data_sqw_dnd, read_dnd

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)
TEST_DND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "test_dnd_class")
)


class TestMigratedAPIsDnD:
    @pytest.fixture(scope="class")
    def d2d_obj(self):
        return read_dnd(os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw"))

    @pytest.fixture(scope="class")
    def ref_mat(self):
        return scipy.io.loadmat(
            os.path.join(TEST_DND_DIR, "test_migrated_apis.mat"),
            squeeze_me=True,
            struct_as_record=False,
        )

    def test_calculate_q_bins(self, d2d_obj, ref_mat):
        q = d2d_obj.calculate_q_bins(nargout=1)
        assert len(q) == 3
        if "q_bins" in ref_mat:
            for i in range(3):
                np.testing.assert_allclose(q[i].ravel(order="F"), ref_mat["q_bins"][i].ravel(order="F"), rtol=1e-5, atol=1e-6)

    def test_calculate_qw_bins(self, d2d_obj, ref_mat):
        qw = d2d_obj.calculate_qw_bins(nargout=1)
        assert len(qw) == 4
        if "qw_bins" in ref_mat:
            for i in range(4):
                np.testing.assert_allclose(qw[i].ravel(order="F"), ref_mat["qw_bins"][i].ravel(order="F"), rtol=1e-5, atol=1e-6)

    def test_compact(self, d2d_obj):
        w = d2d_obj.copy()
        s = w.s.copy()
        npix = w.npix.copy()
        npix[0, :] = 0
        npix[-1, :] = 0
        s[0, :] = 0
        s[-1, :] = 0
        w.npix = npix
        w.s = s
        w_comp = w.compact()
        assert w_comp.s.shape[0] <= w.s.shape[0] - 2

    def test_data_bin_limits(self, d2d_obj):
        val, n = d2d_obj.data_bin_limits(nargout=2)
        assert val.shape == (2, 2)
        assert n.shape == (2, 2)

    def test_get_proj_and_pbin(self, d2d_obj):
        proj, pbin = d2d_obj.get_proj_and_pbin(nargout=2)
        assert len(pbin) == 4
        assert proj.equal_to_tol(d2d_obj.proj)

    def test_mask(self, d2d_obj):
        msk = np.ones_like(d2d_obj.s, dtype=bool)
        msk[0, :] = False
        w_masked = d2d_obj.mask(msk)
        assert np.all(w_masked.npix[0, :] == 0)
        assert np.all(w_masked.s[0, :] == 0)

    def test_save_xye_and_xye(self, d2d_obj, tmp_path):
        res = d2d_obj.xye()
        assert "x" in res and "y" in res and "e" in res
        assert len(res["x"]) == 2
        assert res["y"].shape == d2d_obj.s.shape

        out_path = tmp_path / "test_out.xye"
        d2d_obj.save_xye(str(out_path))
        assert out_path.is_file()
        loaded = np.loadtxt(str(out_path))
        assert loaded.shape == (d2d_obj.s.size, 4)

    def test_section(self, d2d_obj):
        p1 = d2d_obj.p[0]
        p2 = d2d_obj.p[1]
        mid1 = 0.5 * (p1[0] + p1[-1])
        mid2 = 0.5 * (p2[0] + p2[-1])
        w_sec = d2d_obj.section([float(p1[0]), float(mid1)], [float(p2[0]), float(mid2)])
        assert isinstance(w_sec, d2d)
        assert w_sec.s.shape[0] <= d2d_obj.s.shape[0]
        assert w_sec.s.shape[1] <= d2d_obj.s.shape[1]

    def test_value(self, d2d_obj):
        p1 = d2d_obj.p[0]
        p2 = d2d_obj.p[1]
        c1 = 0.5 * (p1[0] + p1[1])
        c2 = 0.5 * (p2[0] + p2[1])
        y, sig = d2d_obj.value([[c1, c2]])
        assert y.shape == (1,)
        assert sig.shape == (1,)
        if d2d_obj.npix[0, 0] > 0:
            np.testing.assert_allclose(y[0], d2d_obj.s[0, 0])
            np.testing.assert_allclose(sig[0], np.sqrt(d2d_obj.e[0, 0]))

    def test_change_crystal(self, d2d_obj):
        info = crystal_alignment_info([4.0, 5.0, 6.0], [90.0, 90.0, 90.0])
        w_mod = d2d_obj.change_crystal(info)
        np.testing.assert_allclose(w_mod.alatt, [4.0, 5.0, 6.0])
        np.testing.assert_allclose(w_mod.angdeg, [90.0, 90.0, 90.0])

