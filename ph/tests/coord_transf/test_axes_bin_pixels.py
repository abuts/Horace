"""Unit tests for line_axes, sphere_axes, cylinder_axes, and bin_pixels."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, PixelDataMemory, cylinder_axes, line_axes, sphere_axes


class TestAxesBlocks:
    """Tests for line_axes, sphere_axes, cylinder_axes, and pixel binning."""

    def test_line_axes_from_pbin(self):
        ax = line_axes([-1, 0.5, 1], [-2, 2], [0, 1, 3], [10, 2, 20])
        np.testing.assert_array_equal(ax.nbins_all_dims, [5, 1, 4, 6])
        np.testing.assert_array_equal(ax.pax, [1, 3, 4])
        np.testing.assert_array_equal(ax.iax, [2])
        np.testing.assert_allclose(ax.iint, [[-2.0], [2.0]])
        assert ax.dimensions == 3
        np.testing.assert_allclose(ax.img_range[:, 0], [-1.25, 1.25])

    def test_sphere_axes_limits(self):
        ax = sphere_axes()
        assert ax.type == "pdde"
        with pytest.raises(HoraceError):
            ax.img_range = np.array([[-1.0, 0.0, -180.0, 0.0], [1.0, 180.0, 180.0, 10.0]])
        with pytest.raises(HoraceError):
            ax.img_range = np.array([[0.0, 0.0, -180.0, 0.0], [1.0, 200.0, 180.0, 10.0]])

    def test_cylinder_axes_limits(self):
        ax = cylinder_axes()
        assert ax.type == "ppde"
        with pytest.raises(HoraceError):
            ax.img_range = np.array([[-1.0, -5.0, -180.0, 0.0], [1.0, 5.0, 180.0, 10.0]])
        with pytest.raises(HoraceError):
            ax.img_range = np.array([[0.0, -5.0, -200.0, 0.0], [1.0, 5.0, 180.0, 10.0]])

    def test_bins_in_1Drange(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ok, rng = line_axes.bins_in_1Drange(edges, [0.5, 2.5])
        assert ok is True
        np.testing.assert_array_equal(rng, [True, True, True, False])

        ok2, rng2 = line_axes.bins_in_1Drange(edges, [5.0, 6.0])
        assert ok2 is False
        np.testing.assert_array_equal(rng2, [False, False, False, False])

    def test_axes_bin_pixels(self):
        ax = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        # 5 points: 4 in each of the 2x2 bins, 1 outside
        coords = np.array(
            [
                [0.5, 1.5, 0.5, 1.5, 3.0],
                [0.5, 0.5, 1.5, 1.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            dtype=float,
        )
        npix = ax.bin_pixels(coords, nargout=1)
        np.testing.assert_array_equal(npix, [[1, 1], [1, 1]])
