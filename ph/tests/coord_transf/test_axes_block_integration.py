"""Unit tests for axes block integration and rebinning (converted from test_axes_block_integration.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    cylinder_axes,
    line_axes,
    sphere_axes,
)


class TestAxesBlockIntegration:
    """Test suite for realign_bin_edges and rebin_data across 1D-4D axes."""

    def test_realign_different_axes_types_throws(self):
        la = line_axes([-1, 0.1, 1], [-1, 1], [-1, 1], [0, 1])
        sa = sphere_axes([0, 0.1, 1], [0, 180], [-180, 180], [0, 1])
        with pytest.raises(HoraceError):
            la.realign_bin_edges(sa)

    def test_realign_bin_edges_commensurate_and_noncommensurate(self):
        ax_src = line_axes([-1.0, 0.1, 1.0], [-2.0, 0.2, 2.0], [-1.0, 1.0], [0.0, 1.0, 10.0])
        # Non-commensurate step 0.15 on axis 1 throws
        ax_bad = line_axes([-0.5, 0.15, 0.5], [-1.0, 1.0], [-1.0, 1.0], [0.0, 1.0, 10.0])
        with pytest.raises(HoraceError):
            ax_src.realign_bin_edges(ax_bad)

        # Commensurate step 0.2 on axis 1, integration on axis 2
        ax_cut = line_axes([-0.5, 0.2, 0.5], [-0.95, 0.95], [-1.0, 1.0], [1.0, 2.0, 9.0])
        ax_aligned = ax_src.realign_bin_edges(ax_cut)
        assert ax_aligned.step[0] == pytest.approx(0.2)
        assert ax_aligned.step[1] == pytest.approx(2.0)
        # Integration range [-0.95, 0.95] on source axis 2 (edges at -2.1, -1.9, ..., -1.1, -0.9, ..., 0.9, 1.1)
        # overlaps bins [-1.1, -0.9] through [0.9, 1.1] -> [-1.1, 1.1]
        np.testing.assert_allclose(ax_aligned.img_range[:, 1], [-1.1, 1.1], atol=1e-12)

    def test_rebin_data_1D_and_2D(self):
        ax_src = line_axes([0.5, 1.0, 9.5], [0.5, 1.0, 3.5], [0, 1], [0, 1])
        # 10 x 4 bins on axes 1 & 2, img_range = [[0, 0, 0, 0], [10, 4, 1, 1]]
        s_in = np.ones((10, 4), dtype=float) * 2.0
        e_in = np.ones((10, 4), dtype=float) * 4.0
        npix_in = np.ones((10, 4), dtype=float) * 3.0

        # Cut integrating axis 2 over [0, 4] and doubling step on axis 1 to 2.0:
        ax_cut = line_axes([1.0, 2.0, 9.0], [0.0, 4.0], [0, 1], [0, 1])
        ax_aligned = ax_src.realign_bin_edges(ax_cut)
        assert ax_aligned.dimensions == 1
        assert ax_aligned.nbins_all_dims[0] == 5

        s_out, e_out, npix_out = ax_src.rebin_data([s_in, e_in, npix_in], ax_aligned)
        # Each target bin combines 2 x 4 = 8 source bins, each with npix=3 -> npix_out = 24
        # Signal is weighted average of constant 2.0 -> 2.0
        # Variance is sum(8 * (4 * 3^2)) / 24^2 = (8 * 36) / 576 = 288 / 576 = 0.5
        np.testing.assert_allclose(npix_out.ravel(), np.full(5, 24.0))
        np.testing.assert_allclose(s_out.ravel(), np.full(5, 2.0))
        np.testing.assert_allclose(e_out.ravel(), np.full(5, 0.5))

    def test_rebin_data_partial_overlap_and_empty_region(self):
        ax_src = line_axes([0.5, 1.0, 3.5], [0, 1], [0, 1], [0, 1])
        # 4 bins in [0, 4]
        s_in = np.array([[10.0], [20.0], [30.0], [40.0]])
        e_in = np.array([[1.0], [1.0], [1.0], [1.0]])
        npix_in = np.array([[1.0], [1.0], [1.0], [1.0]])

        # Target axes shifted to [2, 6] with step 1.0 (4 bins: 2 inside [2, 4], 2 outside [4, 6])
        ax_targ = line_axes([2.5, 1.0, 5.5], [0, 1], [0, 1], [0, 1])
        s_out, e_out, npix_out = ax_src.rebin_data([s_in, e_in, npix_in], ax_targ)
        np.testing.assert_allclose(s_out.ravel(), [30.0, 40.0, 0.0, 0.0])
        np.testing.assert_allclose(e_out.ravel(), [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(npix_out.ravel(), [1.0, 1.0, 0.0, 0.0])

