"""Unit tests for line_axes class (converted from test_line_axes.m & line_axesTester.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import AxesBlockBase, HoraceError, line_axes
from pyhorace.core.serializable import Serializable


class line_axesTester(line_axes):
    """Helper subclass exposing p_defines_bin_centers for testing both center and edge modes."""

    def __init__(self, *args, **kwargs):
        self._p_centers = np.ones(4, dtype=bool)
        super().__init__(*args, **kwargs)

    @property
    def _p_defines_bin_centers(self) -> np.ndarray:
        return self._p_centers

    @property
    def p_defines_bin_centers(self) -> np.ndarray:
        return self._p_centers.copy()

    @p_defines_bin_centers.setter
    def p_defines_bin_centers(self, val):
        self._p_centers = np.asarray(val, dtype=bool).ravel()


class TestLineAxes:
    """Test suite for line_axes construction, properties, pbin_parse, and serialization."""

    def test_default_constructor_and_ndim_constructor(self):
        ax0 = line_axes()
        assert ax0.dimensions == 0
        np.testing.assert_allclose(
            ax0.img_range, np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        )
        np.testing.assert_array_equal(ax0.nbins_all_dims, np.array([1, 1, 1, 1]))
        assert ax0.pax.size == 0
        np.testing.assert_array_equal(ax0.iax, np.array([1, 2, 3, 4]))

        for ndim in (0, 1, 2, 3, 4):
            ax = line_axes(ndim)
            assert ax.dimensions == ndim
            assert ax.pax.size == ndim
            assert ax.iax.size == 4 - ndim

        with pytest.raises(HoraceError):
            line_axes(5)
        with pytest.raises(HoraceError):
            line_axes(-1)

    def test_pbin_parse_centers_and_edges(self):
        tester = line_axesTester()
        lim = np.array([-np.inf, np.inf])

        # Empty
        r, nb, ok, msg = tester.pbin_parse([], True, lim)
        assert ok is True and nb == 1
        np.testing.assert_allclose(r, [0.0, 0.0])

        # 1 element
        r, nb, ok, msg = tester.pbin_parse([2.5], True, lim)
        assert ok is True and nb == 1
        np.testing.assert_allclose(r, [2.5, 2.5])

        # 2 elements
        r, nb, ok, msg = tester.pbin_parse([-1.0, 3.0], True, lim)
        assert ok is True and nb == 1
        np.testing.assert_allclose(r, [-1.0, 3.0])

        # 2 elements invalid (min > max)
        _, _, ok, _ = tester.pbin_parse([3.0, -1.0], True, lim)
        assert ok is False

        # 3 elements with bin centers = True
        r, nb, ok, msg = tester.pbin_parse([-1.0, 0.5, 1.0], True, lim)
        assert ok is True and nb == 5
        np.testing.assert_allclose(r, [-1.25, 1.25])

        # 3 elements with bin centers = False (bin edges)
        r, nb, ok, msg = tester.pbin_parse([-1.0, 0.5, 1.0], False, lim)
        assert ok is True and nb == 4
        np.testing.assert_allclose(r, [-1.0, 1.0])

        # Non-commensurate step with bin centers = True
        r, nb, ok, msg = tester.pbin_parse([0.0, 0.3, 1.0], True, lim)
        assert ok is True and nb == 5
        np.testing.assert_allclose(r, [-0.15, 1.35])

    def test_bin_Initialization_and_properties(self):
        ax = line_axes([-1, 0.1, 1], [-2, 2], [-0.5, 0.5], [0, 2, 10])
        assert ax.dimensions == 2
        np.testing.assert_array_equal(ax.pax, np.array([1, 4]))
        np.testing.assert_array_equal(ax.iax, np.array([2, 3]))
        np.testing.assert_allclose(ax.iint, np.array([[-2.0, -0.5], [2.0, 0.5]]))
        np.testing.assert_array_equal(ax.nbins_all_dims, np.array([21, 1, 1, 6]))
        np.testing.assert_array_equal(ax.data_nbins, np.array([21, 6]))
        np.testing.assert_array_equal(ax.dims_as_ssize, np.array([21, 6]))
        assert len(ax.p) == 2
        assert len(ax.p[0]) == 22
        assert len(ax.p[1]) == 7

        # Test default_pbin roundtrip
        pbin = ax.default_pbin()
        ax2 = line_axes(*pbin)
        assert ax == ax2

    def test_property_validation_errors(self):
        ax = line_axes(2)
        with pytest.raises(HoraceError):
            ax.label = ["a", "b", "c"]
        with pytest.raises(HoraceError):
            ax.ulen = [1, 2, -1, 1]
        with pytest.raises(HoraceError):
            ax.offset = [1, 2]
        with pytest.raises(HoraceError):
            ax.dax = [1, 3]
        with pytest.raises(HoraceError):
            ax.nbins_all_dims = [1, 0, 1, 1]
        with pytest.raises(HoraceError):
            ax.img_range = [[1, 0, 0, 0], [0, 1, 1, 1]]

    def test_serialization_and_legacy_struct(self):
        ax = line_axes(
            [-1, 0.1, 1],
            [-2, 2],
            [-0.5, 0.5],
            [0, 2, 10],
            title="my_cut",
            filename="test.sqw",
            offset=[1, 0, 0, 0],
        )
        ax.hkle_axes_directions = np.eye(3)
        s = ax.to_struct()
        ax_rec = Serializable.from_struct(s)
        assert ax == ax_rec
        assert ax_rec.axes_directions_defined is True

        # Legacy struct conversion
        old_data = {
            "title": "legacy",
            "filename": "old.sqw",
            "filepath": "/tmp",
            "label": ["h", "k", "l", "e"],
            "ulen": np.array([1.5, 1.5, 1.5, 1.0]),
            "uoffset": np.array([1.0, 2.0, 0.0, 0.0]),
            "pax": np.array([1, 4]),
            "iax": np.array([2, 3]),
            "iint": np.array([[-1.0, -0.5], [1.0, 0.5]]),
            "p": [np.linspace(-2.0, 2.0, 21), np.linspace(0.0, 10.0, 11)],
            "dax": np.array([2, 1]),
        }
        ax_old = line_axes.get_from_old_data(old_data)
        assert ax_old.dimensions == 2
        np.testing.assert_array_equal(ax_old.pax, np.array([1, 4]))
        np.testing.assert_array_equal(ax_old.dax, np.array([2, 1]))
        np.testing.assert_array_equal(ax_old.nbins_all_dims, np.array([20, 1, 1, 10]))
        np.testing.assert_allclose(
            ax_old.img_range,
            np.array([[-2.0, -1.0, -0.5, 0.0], [2.0, 1.0, 0.5, 10.0]]),
        )

