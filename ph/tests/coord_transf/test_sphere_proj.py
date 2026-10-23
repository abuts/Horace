"""Unit tests for sphere_proj (converted from test_sphere_proj.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, crystal_alignment_info, sphere_axes, sphere_proj


class TestSphereProj:
    """Tests for spherical coordinate projection sphere_proj."""

    def test_default_constructor(self):
        proj = sphere_proj()
        np.testing.assert_allclose(proj.u, [1, 0, 0])
        np.testing.assert_allclose(proj.v, [0, 1, 0])
        np.testing.assert_allclose(proj.ez, [1, 0, 0])
        np.testing.assert_allclose(proj.ex, [0, 1, 0])
        assert proj.type == "pdd"
        assert proj.label == ["|Q|", "\\theta", "\\phi", "En"]

    def test_invalid_types_throw(self):
        proj = sphere_proj()
        with pytest.raises(HoraceError):
            proj.type = "pdr"
        with pytest.raises(HoraceError):
            proj.type = "ddd"
        with pytest.raises(HoraceError):
            proj.type = "ppp"

    def test_cardinal_directions_transform_deg(self):
        proj = sphere_proj([1, 0, 0], [0, 1, 0], "add", alatt=2 * np.pi, angdeg=90)
        # Points along u=[1,0,0], v=[0,1,0], w=[0,0,1], -u=[-1,0,0]
        pts = np.array(
            [
                [2.0, 0.0, 0.0, -3.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
            dtype=float,
        )
        img = proj.transform_pix_to_img(pts)
        # Along u: |Q|=2, theta=0, phi=0
        # Along v: |Q|=2, theta=90, phi=0
        # Along w: |Q|=2, theta=90, phi=90
        # Along -u: |Q|=3, theta=180, phi=0
        expected = np.array(
            [
                [2.0, 2.0, 2.0, 3.0],
                [0.0, 90.0, 90.0, 180.0],
                [0.0, 0.0, 90.0, 0.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(img, expected, atol=1e-12)

        rec = proj.transform_img_to_pix(img)
        np.testing.assert_allclose(rec, pts, atol=1e-12)

    def test_transform_radians_and_scaling(self):
        proj = sphere_proj(
            [1, 1, 0],
            [-1, 1, 0],
            "prr",
            alatt=[3.0, 3.0, 4.0],
            angdeg=[90, 90, 90],
            offset=[0.5, 0.5, 0.0, 2.0],
        )
        rng = np.random.default_rng(123)
        pts_4d = rng.standard_normal((4, 15))
        img_4d = proj.transform_pix_to_img(pts_4d)
        rec_4d = proj.transform_img_to_pix(img_4d)
        np.testing.assert_allclose(rec_4d, pts_4d, atol=1e-12)

        hkl_4d = proj.transform_img_to_hkl(img_4d)
        img_from_hkl = proj.transform_hkl_to_img(hkl_4d)
        np.testing.assert_allclose(img_from_hkl, img_4d, atol=1e-12)

    def test_serialization_and_alignment(self):
        proj = sphere_proj([1, 0, 0], [0, 1, 0], "add", alatt=[3.0, 3.0, 3.0], angdeg=[90, 90, 90])
        s = proj.to_struct()
        rec = sphere_proj.from_struct(s)
        assert proj == rec

        align = crystal_alignment_info([3.1, 3.1, 3.1], [90, 90, 90], [0.0, 0.0, 0.05])
        proj.align_proj(align)
        np.testing.assert_allclose(proj.alatt, [3.1, 3.1, 3.1])
