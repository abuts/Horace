"""Unit tests for cylinder_axes class (converted from test_cylinder_axes.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, cylinder_axes
from pyhorace.core.serializable import Serializable


class TestCylinderAxes:
    """Test suite for cylinder_axes construction, angular units, volumes, and ranges."""

    def test_default_construction_deg_and_rad(self):
        ca = cylinder_axes()
        assert ca.angular_unit_is_rad is False
        assert ca.type == "ppde"
        assert ca.changes_aspect_ratio is False
        np.testing.assert_allclose(
            ca.img_range,
            np.array([[0.0, -1.0, -180.0, 0.0], [1.0, 1.0, 180.0, 0.0]]),
        )

        ca.angular_unit_is_rad = True
        assert ca.type == "ppre"
        np.testing.assert_allclose(
            ca.img_range,
            np.array([[0.0, -1.0, -np.pi, 0.0], [1.0, 1.0, np.pi, 0.0]]),
        )

    def test_pbin_constrained_ranges_and_limits(self):
        ca = cylinder_axes([0, 0.1, 2], [-1, 0.2, 1], [-180, 10, 180], [-5, 1, 5])
        assert ca.dimensions == 4
        assert ca.img_range[0, 0] == pytest.approx(0.0)
        assert ca.img_range[1, 0] == pytest.approx(2.1)
        assert ca.nbins_all_dims[0] == 21
        np.testing.assert_allclose(ca.img_range[:, 1], [-1.1, 1.1])
        assert ca.nbins_all_dims[1] == 11
        np.testing.assert_allclose(ca.img_range[:, 2], [-180.0, 180.0])
        assert ca.nbins_all_dims[2] == 36

        with pytest.raises(HoraceError):
            cylinder_axes([-1, 0.1, 2], [-1, 1], [-180, 180], [0, 1])
        with pytest.raises(HoraceError):
            cylinder_axes([0, 1], [-1, 1], [-190, 180], [0, 1])

    def test_offset_validation(self):
        ca = cylinder_axes()
        ca.offset = [1.0, -5.0, 90.0, 0.0]
        np.testing.assert_allclose(ca.offset, [1.0, -5.0, 90.0, 0.0])

        with pytest.raises(HoraceError):
            ca.offset = [-0.1, 0.0, 0.0, 0.0]
        with pytest.raises(HoraceError):
            ca.offset = [1.0, 0.0, 190.0, 0.0]

    def test_calc_bin_volume_full_cylinder(self):
        # Cylinder of radius R=2, height H=4 ([-2, 2]), phi=[-180, 180], dE=[0, 1]:
        # Volume = pi * R^2 * H * dE = pi * 4 * 4 * 1 = 16 * pi
        ca = cylinder_axes([0, 2], [-2, 2], [-180, 180], [0, 1])
        vol = ca.calc_bin_volume()
        expected = np.pi * (2.0 ** 2) * 4.0 * 1.0
        assert vol == pytest.approx(expected, rel=1e-12)

        ca_multi = cylinder_axes([0, 0.5, 1.5], [-1.5, 1.0, 1.5], [-175, 10, 175], [0, 1])
        ca_multi.img_range = np.array([[0.0, -2.0, -180.0, 0.0], [2.0, 2.0, 180.0, 1.0]])
        vols = ca_multi.calc_bin_volume()
        assert np.sum(vols) == pytest.approx(expected, rel=1e-12)

    def test_serialization(self):
        ca = cylinder_axes(
            [0, 0.2, 2],
            [-2, 0.5, 2],
            [-np.pi, np.pi / 6, np.pi],
            [-2, 2],
            angular_unit_is_rad=True,
        )
        s = ca.to_struct()
        ca_rec = Serializable.from_struct(s)
        assert ca == ca_rec
