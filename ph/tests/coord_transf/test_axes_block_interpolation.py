"""Unit tests for axes block density and interpolation (converted from test_axes_block_interpolation.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    cylinder_axes,
    line_axes,
    sphere_axes,
)


class TestAxesBlockInterpolation:
    """Test suite for get_density and interpolate_data."""

    def test_get_density_invalid_inputs_throw(self):
        ax = line_axes([-1, 0.5, 1], [-1, 0.5, 1], [-1, 1], [0, 1])
        with pytest.raises(HoraceError):
            ax.get_density([np.ones((5, 5)), np.ones((5, 5))])
        with pytest.raises(HoraceError):
            ax.get_density([np.ones((4, 4)), np.ones((5, 5)), np.ones((5, 5))])

    def test_get_density_and_self_interpolation_4D_and_2D(self):
        # 2D line_axes
        ax2 = line_axes([-1, 0.5, 1], [-1, 0.5, 1], [0, 2], [0, 2])
        # nbins = [5, 5, 1, 1], step = [0.5, 0.5, 2, 2], bin_vol = 0.5 * 0.5 * 2 * 2 = 1.0
        sz = tuple(int(x) for x in ax2.dims_as_ssize)
        s = np.ones(sz, dtype=float) * 4.0
        e = np.ones(sz, dtype=float) * 2.0
        npix = np.ones(sz, dtype=float) * 10.0

        sd, ed, nd = ax2.get_density([s, e, npix])
        assert sd.shape == (7, 7, 1, 1)
        # Interior points have density = value / vol (vol = 1.0), halo points are 0
        np.testing.assert_allclose(sd[1:-1, 1:-1, 0, 0], 4.0)
        np.testing.assert_allclose(sd[0, :, 0, 0], 0.0)

        # Interpolating onto identical axes reproduces original data
        si, ei, ni = ax2.interpolate_data(ax2, [s, e, npix])
        np.testing.assert_allclose(si, s)
        np.testing.assert_allclose(ei, e)
        np.testing.assert_allclose(ni, npix)

        # 4D line_axes
        ax4 = line_axes([0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2])
        sz4 = tuple(int(x) for x in ax4.dims_as_ssize)
        s4 = np.ones(sz4, dtype=float) * 6.0
        e4 = np.ones(sz4, dtype=float) * 3.0
        n4 = np.ones(sz4, dtype=float) * 8.0
        si4, ei4, ni4 = ax4.interpolate_data(ax4, [s4, e4, n4])
        np.testing.assert_allclose(si4, s4)
        np.testing.assert_allclose(ei4, e4)
        np.testing.assert_allclose(ni4, n4)

    def test_interpolate_data_invalid_inputs_throw(self):
        ax = line_axes([0, 1, 2], [0, 1, 2], [0, 1], [0, 1])
        sz = tuple(int(x) for x in ax.dims_as_ssize)
        s = np.ones(sz)
        with pytest.raises(HoraceError):
            ax.interpolate_data(ax, [s, s])
        with pytest.raises(HoraceError):
            ax.interpolate_data(ax, [s, s, s], [np.ones((2, 2))])
        with pytest.raises(HoraceError):
            ax.interpolate_data(
                ax,
                [s, s, s],
                [np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2))],
            )

