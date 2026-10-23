"""Unit tests for line_proj methods (converted from test_line_proj_methods.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    crystal_alignment_info,
    line_axes,
    line_proj,
    rotvec_to_rotmat_rad,
    ubmat_proj,
)


class TestLineProjMethods:
    """Tests for line_proj helper methods, legacy conversion, and alignment."""

    def test_get_ubmat_proj_and_back(self):
        lp = line_proj(
            [1, 1, 0],
            [-1, 1, 0],
            [0, 0, 1],
            alatt=[2.8, 2.8, 5.4],
            angdeg=[90, 90, 90],
            offset=[0.1, 0.2, 0.3, 4.0],
            type="ppr",
        )
        ubp = lp.get_ubmat_proj()
        assert isinstance(ubp, ubmat_proj)
        np.testing.assert_allclose(ubp.u_to_rlu, lp.u_to_rlu, atol=1e-12)
        np.testing.assert_allclose(ubp.ulen, lp.img_scales, atol=1e-12)

        lp2 = ubp.get_line_proj()
        pts = np.array([[1.0, 2.0], [0.5, -1.0], [0.2, 0.8], [3.0, 5.0]])
        np.testing.assert_allclose(
            lp.transform_pix_to_img(pts),
            ubp.transform_pix_to_img(pts),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            lp.transform_pix_to_img(pts),
            lp2.transform_pix_to_img(pts),
            atol=1e-12,
        )

    def test_from_old_data_struct(self):
        lp_orig = line_proj([1, 0, 0], [0, 1, 0], alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi], angdeg=[90, 90, 90])
        old_struct = lp_orig.compat_struct
        lp_rec = line_proj.get_from_old_data(old_struct)
        np.testing.assert_allclose(lp_rec.u_to_rlu, lp_orig.u_to_rlu, atol=1e-12)
        np.testing.assert_allclose(lp_rec.ulen, lp_orig.ulen, atol=1e-12)

    def test_align_proj(self):
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=[3.0, 3.0, 3.0], angdeg=[90, 90, 90])
        rotvec = np.deg2rad([0.0, 0.0, 15.0])
        align_info = crystal_alignment_info([3.1, 3.1, 3.1], [90, 90, 90], rotvec)
        align_info.hkl_mode = True
        lp.align_proj(align_info)
        np.testing.assert_allclose(lp.alatt, [3.1, 3.1, 3.1])
        rm = rotvec_to_rotmat_rad(rotvec)
        # With cubic lattice, B^-1 * R * B == R
        np.testing.assert_allclose(lp.u / np.linalg.norm(lp.u), rm[:, 0], atol=1e-12)

    def test_copy_proj_defined_properties_to_axes(self):
        lp = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            alatt=[3.0, 3.0, 3.0],
            angdeg=[90, 90, 90],
            offset=[0.5, 0, 0, 0],
            label=["H", "K", "L", "E"],
            title="Test Title",
        )
        ax = line_axes()
        lp.copy_proj_defined_properties_to_axes(ax)
        np.testing.assert_allclose(ax.offset, [0.5, 0, 0, 0])
        assert ax.label == ["H", "K", "L", "E"]
        assert ax.title == "Test Title"
        np.testing.assert_allclose(ax.img_scales, lp.img_scales)
        np.testing.assert_allclose(ax.hkle_axes_directions, lp.u_to_rlu)

