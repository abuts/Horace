"""Unit tests for AxesBlockBase.bins_in_1Drange (converted from test_axes_block_binIn1D.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import AxesBlockBase


class TestAxesBlockBinIn1D:
    """Test suite for AxesBlockBase.bins_in_1Drange."""

    def test_in_range_full(self):
        bins = np.arange(-1.0, 11.0, 2.0)  # [-1, 1, 3, 5, 7, 9]
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [-1.0, 9.0])
        assert any_within is True
        np.testing.assert_array_equal(
            is_within, np.array([True, True, True, True, True])
        )

    def test_in_range_partial_internal(self):
        bins = np.arange(-1.0, 11.0, 2.0)  # [-1, 1, 3, 5, 7, 9]
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [1.5, 6.5])
        assert any_within is True
        np.testing.assert_array_equal(
            is_within, np.array([False, True, True, True, False])
        )

    def test_in_range_single_point_on_edge(self):
        bins = np.arange(-1.0, 11.0, 2.0)  # [-1, 1, 3, 5, 7, 9]
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [3.0, 3.0])
        assert any_within is True
        np.testing.assert_array_equal(
            is_within, np.array([False, True, True, False, False])
        )

    def test_in_range_single_point_internal(self):
        bins = np.arange(-1.0, 11.0, 2.0)
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [2.0, 2.0])
        assert any_within is True
        np.testing.assert_array_equal(
            is_within, np.array([False, True, False, False, False])
        )

    def test_out_of_range_low(self):
        bins = np.arange(-1.0, 11.0, 2.0)
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [-10.0, -2.0])
        assert any_within is False
        np.testing.assert_array_equal(
            is_within, np.array([False, False, False, False, False])
        )

    def test_out_of_range_high(self):
        bins = np.arange(-1.0, 11.0, 2.0)
        any_within, is_within = AxesBlockBase.bins_in_1Drange(bins, [10.0, 20.0])
        assert any_within is False
        np.testing.assert_array_equal(
            is_within, np.array([False, False, False, False, False])
        )

