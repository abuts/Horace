"""Unit tests for line_axes, sphere_axes, cylinder_axes, and bin_pixels."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    PixelDataMemory,
    cylinder_axes,
    line_axes,
    line_proj,
    sphere_axes,
)
from pyhorace.core.configurations import hor_config


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

    def test_axes_bin_pixels_mex_selection(self):
        ax = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 1.0, 1.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        coords = np.array(
            [
                [0.5, 1.5, 0.5, 1.5],
                [0.5, 0.5, 1.5, 1.5],
                [0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=float,
        )

        # Incompatible flags '-nomex' and '-force_mex'
        with pytest.raises(HoraceError, match="cannot be used together"):
            ax.bin_pixels(coords, "-nomex", "-force_mex", nargout=1)

        # '-force_mex' with '-test_mex_inputs' returns mex_par dict
        mex_par = ax.bin_pixels(coords, "-force_mex", "-test_mex_inputs", nargout=1)
        assert isinstance(mex_par, dict)
        assert mex_par["mode"] == 1
        assert mex_par["test_inputs"] is True

        # '-force_mex' raises NotImplementedError
        with pytest.raises(NotImplementedError, match="bin_pixels_with_mex_code"):
            ax.bin_pixels(coords, "-force_mex", nargout=1)

        # hor_config().use_mex = True raises NotImplementedError for multi-output binning
        hc = hor_config()
        prev_mex = hc.use_mex
        try:
            hc.use_mex = True
            with pytest.raises(NotImplementedError, match="bin_pixels_with_mex_code"):
                ax.bin_pixels(coords, nargout=2)

            # '-nomex' overrides hor_config().use_mex = True
            npix, ok = ax.bin_pixels(coords, "-nomex", nargout=2)
            np.testing.assert_array_equal(npix, [[1, 1], [1, 1]])
            assert np.all(ok)
        finally:
            hc.use_mex = prev_mex

    def test_line_proj_bin_pixels_mex_selection(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
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

        with pytest.raises(NotImplementedError, match="bin_pixels_with_mex_code"):
            proj.bin_pixels(ax, pix, [], [], [], "-force_mex", nargout=5)

        hc = hor_config()
        prev_mex = hc.use_mex
        try:
            hc.use_mex = True
            with pytest.raises(NotImplementedError, match="bin_pixels_with_mex_code"):
                proj.bin_pixels(ax, pix, [], [], [], nargout=5)

            # '-nomex' overrides hor_config().use_mex = True
            npix, s, e, pix_ok, _ = proj.bin_pixels(
                ax, pix, [], [], [], "-nomex", nargout=5
            )
            np.testing.assert_array_equal(npix, [[1, 1], [1, 1]])
            assert pix_ok.num_pixels == 4
        finally:
            hc.use_mex = prev_mex

