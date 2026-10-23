"""Unit tests for aProjectionBase (converted from test_aProjectionBase.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, PixelDataMemory, line_axes, line_proj
from tests.coord_transf.proj_test_GET_NRANGE_helper import proj_test_GET_NRANGE_helper


class TestAProjectionBase:
    """Tests for abstract base projection class aProjectionBase."""

    def test_empty_constructor_properties(self):
        proj = proj_test_GET_NRANGE_helper()
        assert proj.alatt.size == 0
        assert proj.angdeg.size == 0
        assert not proj.alatt_defined
        assert not proj.angdeg_defined
        np.testing.assert_allclose(proj.offset, [0, 0, 0, 0])
        assert proj.label == ["\\zeta", "\\xi", "\\eta", "E"]
        assert proj.title == ""
        assert proj.type == "ppr"
        assert proj.do_3D_transformation is True
        assert proj.do_generic is False

    def test_alatt_angdeg_setters(self):
        proj = proj_test_GET_NRANGE_helper()
        proj.alatt = 3.5
        assert proj.alatt_defined
        np.testing.assert_allclose(proj.alatt, [3.5, 3.5, 3.5])

        proj.alatt = [2.0, 3.0, 4.0]
        np.testing.assert_allclose(proj.alatt, [2.0, 3.0, 4.0])

        proj.angdeg = 90
        assert proj.angdeg_defined
        np.testing.assert_allclose(proj.angdeg, [90.0, 90.0, 90.0])

        proj.angdeg = [80.0, 90.0, 110.0]
        np.testing.assert_allclose(proj.angdeg, [80.0, 90.0, 110.0])

        proj.alatt = []
        assert not proj.alatt_defined
        assert proj.alatt.size == 0

    def test_invalid_alatt_angdeg_throws(self):
        proj = proj_test_GET_NRANGE_helper()
        with pytest.raises(HoraceError):
            proj.alatt = -1.0
        with pytest.raises(HoraceError):
            proj.alatt = [1.0, 2.0]
        with pytest.raises(HoraceError):
            proj.angdeg = 0.0
        with pytest.raises(HoraceError):
            proj.angdeg = 185.0
        with pytest.raises(HoraceError):
            proj.angdeg = [130.0, 130.0, 130.0]

    def test_offset_and_labels(self):
        proj = proj_test_GET_NRANGE_helper()
        proj.offset = [1, 2, 3]
        np.testing.assert_allclose(proj.offset, [1.0, 2.0, 3.0, 0.0])

        proj.offset = [1, 2, 3, 4]
        np.testing.assert_allclose(proj.offset, [1.0, 2.0, 3.0, 4.0])

        proj.offset = []
        np.testing.assert_allclose(proj.offset, [0.0, 0.0, 0.0, 0.0])

        proj.lab1 = "Qx"
        proj.lab2 = "Qy"
        proj.lab3 = "Qz"
        proj.lab4 = "En"
        assert proj.label == ["Qx", "Qy", "Qz", "En"]

        with pytest.raises(HoraceError):
            proj.label = ["only_one"]

    def test_pix_hkl_transform_roundtrip(self):
        proj = proj_test_GET_NRANGE_helper(alatt=[2.8, 3.5, 4.2], angdeg=[85, 95, 105])
        pts_cc = np.array(
            [
                [1.0, 2.0, 3.0, 0.5],
                [0.5, -1.0, 2.0, 1.5],
                [-0.5, 0.0, 1.0, -2.0],
                [5.0, 10.0, 15.0, 20.0],
            ],
            dtype=float,
        )
        pts_hkl = proj.transform_pix_to_hkl(pts_cc)
        pts_rec = proj.transform_hkl_to_pix(pts_hkl)
        np.testing.assert_allclose(pts_rec, pts_cc, atol=1e-12)

    def test_get_nrange_simple(self):
        proj1 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        proj2 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        ax2 = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 2.0, 2.0]]),
            nbins_all_dims=[1, 1, 1, 1],
        )
        # 4 bins in ax1; bin 1 ([0..1, 0..1]) overlaps ax2
        npix = np.array([[3, 4], [5, 6]], dtype=int)
        nstart, nlength = proj1.get_nrange(npix, ax1, ax2, proj2)
        assert nstart.size >= 1
        assert nstart[0] == 1
        assert nlength[0] >= 3

    def test_bin_pixels_with_pixel_data(self):
        proj = proj_test_GET_NRANGE_helper(alatt=2 * np.pi, angdeg=90)
        ax = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 10.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        raw = np.zeros((9, 4), dtype=float)
        raw[0, :] = [0.5, 1.5, 0.5, 1.5]
        raw[1, :] = [0.5, 0.5, 1.5, 1.5]
        raw[2, :] = [1.0, 1.0, 1.0, 1.0]
        raw[3, :] = [5.0, 5.0, 5.0, 5.0]
        raw[4, :] = 1.0
        raw[7, :] = [10.0, 20.0, 30.0, 40.0]
        raw[8, :] = [1.0, 2.0, 3.0, 4.0]
        pix = PixelDataMemory(raw)

        npix, s, e, pix_ok, uniq_runid = proj.bin_pixels(ax, pix, [], [], [], [1], nargout=5)
        np.testing.assert_array_equal(npix, [[1, 1], [1, 1]])
        np.testing.assert_allclose(s, [[10.0, 30.0], [20.0, 40.0]])
        np.testing.assert_allclose(e, [[1.0, 3.0], [2.0, 4.0]])
        assert pix_ok.num_pixels == 4
