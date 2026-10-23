"""Unit tests for get_targ_range across projections and axes (converted from test_targ_ranges.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    DnDBase,
    HoraceError,
    cylinder_axes,
    cylinder_proj,
    d1d,
    d2d,
    d3d,
    d4d,
    line_axes,
    line_proj,
    sphere_axes,
    sphere_proj,
)


class TestTargRanges:
    """Test suite for AxesBlockBase.get_targ_range and DnDBase.get_targ_range."""

    def test_same_projection_returns_exact_range_and_binning(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=[3, 3, 3], angdeg=[90, 90, 90])
        ab = line_axes([-1, 0.1, 1], [-2, 0.2, 2], [-3, 3], [0, 1, 10])
        ab = proj.copy_proj_defined_properties_to_axes(ab)
        dnd_obj = d3d(ab, proj)

        rng = dnd_obj.get_targ_range()
        np.testing.assert_allclose(rng, ab.img_range)

        pbin = dnd_obj.get_targ_range(proj, "-binning")
        assert len(pbin) == 4
        # With '-binning', get_targ_range returns plot ranges [min, step, max] or [min, max]
        assert pbin[2].size == 2
        np.testing.assert_allclose(pbin[2], [-3.0, 3.0])

        with pytest.raises(HoraceError):
            dnd_obj.get_targ_range(proj, "-invalid_flag")

    def test_line_to_line_rotated_and_shifted(self):
        proj1 = line_proj([1, 0, 0], [0, 1, 0], alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi], angdeg=[90, 90, 90])
        ab1 = line_axes([-1, 1], [-1, 1], [-1, 1], [0, 10])
        d_obj = DnDBase.dnd(ab1, proj1)

        # Shifted target projection
        proj2 = line_proj(
            [1, 0, 0],
            [0, 1, 0],
            offset=[1, 1, 1, 0],
            alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi],
            angdeg=[90, 90, 90],
        )
        rng_shift = d_obj.get_targ_range(proj2)
        expected_shift = np.array([[-2.0, -2.0, -2.0, 0.0], [0.0, 0.0, 0.0, 10.0]])
        np.testing.assert_allclose(rng_shift, expected_shift, atol=1e-12)

        # Rotated 45 degrees in hk plane (type='aaa' so unit vectors in A^-1)
        proj_rot = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            type="aaa",
            alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi],
            angdeg=[90, 90, 90],
        )
        rng_rot = d_obj.get_targ_range(proj_rot)
        s2 = np.sqrt(2.0)
        expected_rot = np.array([[-s2, -s2, -1.0, 0.0], [s2, s2, 1.0, 10.0]])
        np.testing.assert_allclose(rng_rot, expected_rot, atol=1e-12)

    def test_line_to_sphere_and_cylinder(self):
        alatt = [2 * np.pi, 2 * np.pi, 2 * np.pi]
        angdeg = [90, 90, 90]
        lproj = line_proj([1, 0, 0], [0, 1, 0], alatt=alatt, angdeg=angdeg)
        # Cube [-1, 1]^3 with step 0.5 so origin and face centers are included in hull
        ab_cube = line_axes([-0.75, 0.5, 0.75], [-0.75, 0.5, 0.75], [-0.75, 0.5, 0.75], [0, 10])
        d_cube = d3d(ab_cube, lproj)

        sproj = sphere_proj(alatt=alatt, angdeg=angdeg, type="add")
        sph_rng = d_cube.get_targ_range(sproj)
        # Max |Q| at corners (+/-1, +/-1, +/-1) is sqrt(3)
        assert sph_rng[1, 0] == pytest.approx(np.sqrt(3.0), rel=1e-6)
        assert sph_rng[0, 1] >= 0.0
        assert sph_rng[1, 1] <= 180.0
        assert sph_rng[0, 2] >= -180.0
        assert sph_rng[1, 2] <= 180.0

        cproj = cylinder_proj(alatt=alatt, angdeg=angdeg, type="aad")
        cyl_rng = d_cube.get_targ_range(cproj)
        # Max Q_perp at corners (+/-1, +/-1) is sqrt(2), Q_|| in [-1, 1]
        assert cyl_rng[1, 0] == pytest.approx(np.sqrt(2.0), rel=1e-6)
        np.testing.assert_allclose(cyl_rng[:, 1], [-1.0, 1.0], atol=1e-6)
        assert cyl_rng[0, 2] >= -180.0
        assert cyl_rng[1, 2] <= 180.0

