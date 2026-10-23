"""Unit tests for line_proj coordinate transformations (converted from test_line_proj_transforms.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import PixelDataMemory, bmatrix, line_proj


class TestLineProjTransforms:
    """Tests for line_proj pixel/image/hkl/target coordinate transformations."""

    def test_cubic_identity_transform(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        pts = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=float,
        )
        img = proj.transform_pix_to_img(pts)
        np.testing.assert_allclose(img, pts, atol=1e-12)
        rec = proj.transform_img_to_pix(img)
        np.testing.assert_allclose(rec, pts, atol=1e-12)

    def test_orthogonal_rotated_axes_with_offset(self):
        proj = line_proj(
            [1, 1, 0],
            [-1, 1, 0],
            [0, 0, 1],
            alatt=[3.0, 3.0, 5.0],
            angdeg=[90, 90, 90],
            offset=[1.0, 0.0, 0.0, 2.0],
            type="ppp",
        )
        # HKL point at (2, 1, 3, 10) -> relative to offset (1, 0, 0, 2) is dHKL = (1, 1, 3), dE_rel = 8
        # Since u = [1, 1, 0], dHKL = 1*u + 0*v + 3*w -> image coords should be (1, 0, 3, 8)!
        hkl_pt = np.array([[2.0], [1.0], [3.0], [10.0]])
        pix_pt = proj.transform_hkl_to_pix(hkl_pt)
        img_pt = proj.transform_pix_to_img(pix_pt)
        np.testing.assert_allclose(img_pt, [[1.0], [0.0], [3.0], [8.0]], atol=1e-12)

        pix_rec = proj.transform_img_to_pix(img_pt)
        np.testing.assert_allclose(pix_rec, pix_pt, atol=1e-12)

        hkl_rec = proj.transform_img_to_hkl(img_pt)
        np.testing.assert_allclose(hkl_rec, hkl_pt, atol=1e-12)

    def test_nonorthogonal_triclinic_transform(self):
        proj = line_proj(
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            nonorthogonal=True,
            alatt=[3.2, 4.1, 5.3],
            angdeg=[80, 95, 110],
            offset=[0.2, -0.1, 0.5, 1.5],
            type="rar",
        )
        rng = np.random.default_rng(42)
        pts_4d = rng.standard_normal((4, 20))
        img_4d = proj.transform_pix_to_img(pts_4d)
        rec_4d = proj.transform_img_to_pix(img_4d)
        np.testing.assert_allclose(rec_4d, pts_4d, atol=1e-12)

        hkl_4d = proj.transform_img_to_hkl(img_4d)
        img_from_hkl = proj.transform_hkl_to_img(hkl_4d)
        np.testing.assert_allclose(img_from_hkl, img_4d, atol=1e-12)

    def test_from_this_to_targ_coord_optimization(self):
        p1 = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            alatt=[3.0, 4.0, 5.0],
            angdeg=[90, 90, 90],
            offset=[0.5, 0.2, 0.0, 1.0],
        )
        p2 = line_proj(
            [1, 0, 0],
            [0, 1, 1],
            alatt=[3.0, 4.0, 5.0],
            angdeg=[90, 90, 90],
            offset=[-0.2, 0.1, 0.3, 2.0],
        )
        p1.targ_proj = p2

        pts_origin = np.array([[1.0, 2.0, -1.0], [0.5, -0.5, 1.5], [2.0, 0.0, -2.0], [5.0, 6.0, 7.0]])
        targ_opt = p1.from_this_to_targ_coord(pts_origin)

        p1.disable_srce_to_targ_optimization = True
        targ_no_opt = p1.from_this_to_targ_coord(pts_origin)
        np.testing.assert_allclose(targ_opt, targ_no_opt, atol=1e-12)

    def test_pixel_data_input(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90, offset=[1, 2, 3, 4])
        raw = np.ones((9, 5), dtype=float)
        raw[0, :] = 2.0
        raw[1, :] = 4.0
        raw[2, :] = 6.0
        raw[3, :] = 10.0
        pix = PixelDataMemory(raw)
        img = proj.transform_pix_to_img(pix)
        assert img.shape == (4, 5)
        np.testing.assert_allclose(img[0, :], 1.0)
        np.testing.assert_allclose(img[1, :], 2.0)
        np.testing.assert_allclose(img[2, :], 3.0)
        np.testing.assert_allclose(img[3, :], 6.0)

