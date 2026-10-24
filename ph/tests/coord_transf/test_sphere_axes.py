"""Unit tests for sphere_axes class (converted from test_sphere_axes.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, sphere_axes
from pyhorace.core.serializable import Serializable


class TestSphereAxes:
    """Test suite for sphere_axes construction, angular units, volumes, and ranges."""

    def test_default_construction_deg_and_rad(self):
        sa = sphere_axes()
        assert sa.angular_unit_is_rad is False
        assert sa.type == "pdde"
        assert sa.changes_aspect_ratio is False
        np.testing.assert_allclose(
            sa.img_range,
            np.array([[0.0, 0.0, -180.0, 0.0], [1.0, 180.0, 180.0, 0.0]]),
        )

        sa.angular_unit_is_rad = True
        assert sa.type == "prre"
        np.testing.assert_allclose(
            sa.img_range,
            np.array([[0.0, 0.0, -np.pi, 0.0], [1.0, np.pi, np.pi, 0.0]]),
        )

    def test_pbin_constrained_ranges_and_limits(self):
        sa = sphere_axes([0, 0.1, 2], [0, 10, 180], [-180, 10, 180], [-5, 1, 5])
        assert sa.dimensions == 4
        # Notice: at 0 for |Q| and theta, min_v = 0 - half_step is clamped to 0,
        # and since p[0] == 0 == lim[0], max_v is shifted by +half_step or clamped to 180!
        # For |Q|: [0, 0.1, 2] -> min=0, max=2+0.05+0.05=2.1, nbins=21
        # For theta: [0, 10, 180] -> min=0, max=180, nbins=18
        # For phi: [-180, 10, 180] -> min=-180, max=180, nbins=36
        assert sa.img_range[0, 0] == pytest.approx(0.0)
        assert sa.img_range[1, 0] == pytest.approx(2.1)
        assert sa.nbins_all_dims[0] == 20
        np.testing.assert_allclose(sa.img_range[:, 1], [0.0, 180.0])
        assert sa.nbins_all_dims[1] == 18
        np.testing.assert_allclose(sa.img_range[:, 2], [-180.0, 180.0])
        assert sa.nbins_all_dims[2] == 36

        # Out-of-bounds spherical limits throw
        with pytest.raises(HoraceError):
            sphere_axes([-1, 0.1, 2], [0, 180], [-180, 180], [0, 1])
        with pytest.raises(HoraceError):
            sphere_axes([0, 1], [-10, 180], [-180, 180], [0, 1])
        with pytest.raises(HoraceError):
            sphere_axes([0, 1], [0, 180], [-190, 180], [0, 1])

    def test_offset_validation(self):
        sa = sphere_axes()
        sa.offset = [1.0, 45.0, -90.0, 0.0]
        np.testing.assert_allclose(sa.offset, [1.0, 45.0, -90.0, 0.0])

        with pytest.raises(HoraceError):
            sa.offset = [-0.5, 45.0, 0.0, 0.0]
        with pytest.raises(HoraceError):
            sa.offset = [1.0, 200.0, 0.0, 0.0]

    def test_calc_bin_volume_full_sphere(self):
        # Full sphere of radius R=2 with 1 bin in Q, theta, phi, and dE=[0, 1]:
        # Volume should be (4/3)*pi*R^3 * dE = (4/3)*pi*8 * 1 = 32*pi/3
        sa = sphere_axes([0, 2], [0, 180], [-180, 180], [0, 1])
        vol = sa.calc_bin_volume()
        expected = (4.0 / 3.0) * np.pi * (2.0 ** 3) * 1.0
        assert vol == pytest.approx(expected, rel=1e-12)

        # Subdivided sphere: sum of all bin volumes must equal full sphere volume
        sa_multi = sphere_axes([0, 0.5, 1.5], [5, 10, 175], [-175, 10, 175], [0, 1])
        # Force exact [0, 2] on Q
        sa_multi.img_range = np.array([[0.0, 0.0, -180.0, 0.0], [2.0, 180.0, 180.0, 1.0]])
        vols = sa_multi.calc_bin_volume()
        assert np.sum(vols) == pytest.approx(expected, rel=1e-12)

    def test_serialization(self):
        sa = sphere_axes(
            [0, 0.2, 2],
            [0, 15, 180],
            [-180, 30, 180],
            [-2, 2],
            angular_unit_is_rad=False,
        )
        s = sa.to_struct()
        sa_rec = Serializable.from_struct(s)
        assert sa == sa_rec

