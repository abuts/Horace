"""Unit tests for IX_dataset_1d, IX_dataset_2d, IX_dataset_3d, and IX_dataset_nd.

Translates test_IX_dataset_1d.m, test_IX_dataset_2d.m, test_IX_dataset_3d.m,
IX_dataset_tester.m, make_IX_dataset_nd_random.m, make_testdata_IX_dataset_1d.m,
and make_testdata_IX_dataset_2d.m from _test/test_IX_classes, plus comprehensive
tests for 1D/2D/3D data manipulation methods (rebin, integrate, cut, deriv,
unspike, func_eval, noisify, scale, shift, smooth, rebunch, interp, linspace,
mask, moments, xye, expand_d1d, expand_d2d, compact_array).
"""

import numpy as np
import pytest

from pyhorace.data import (
    DataOpArray,
    HoraceError,
    IX_axis,
    IX_dataset,
    IX_dataset_1d,
    IX_dataset_2d,
    IX_dataset_3d,
    IX_dataset_nd,
    compact_array,
)


def make_IX_dataset_nd_random(sz):
    """Create IX_dataset_nd with random signal and error (from make_IX_dataset_nd_random.m)."""
    rng = np.random.default_rng(12345)
    sz = list(sz)
    ndim = len(sz)
    ax = []
    for i in range(ndim):
        ax.append({
            "values": np.arange(1, sz[i] + 1, dtype=float),
            "axis": IX_axis(f"Axis {i + 1}"),
            "distribution": False,
        })
    shape = sz if ndim > 1 else [sz[0], 1]
    signal = 10.0 * rng.random(shape)
    err = rng.random(shape)
    title = f"Test IX_dataset_{ndim}d"
    s_axis = IX_axis("Counts")
    return IX_dataset_nd(title, signal, err, s_axis, ax)


def make_testdata_IX_dataset_1d(nx0, nw):
    """Create arrays of IX_dataset_1d with random x axes and Gaussian signal (from make_testdata_IX_dataset_1d.m)."""
    rng = np.random.default_rng(12345)
    xrange = 10.0
    nx = nx0 + np.round(0.2 * nx0 * rng.random(nw)).astype(int)
    pp_1d_gau = []
    hh_1d_gau = []
    hp_1d_gau = []
    for i in range(1, nw + 1):
        n = nx[i - 1]
        x = xrange * ((1.0 + 0.1 * rng.random()) * np.sort(rng.random(n)) - 0.05 * rng.random())
        y = 10.0 * np.exp(-0.5 * (((x - xrange / 2) / (xrange / 4)) ** 2 + ((i - nw / 2) / (nw / 4)) ** 2))
        e = 0.5 + rng.random(n)
        pp_1d_gau.append(
            IX_dataset_1d(x, y, e, "Point data, not distribution", IX_axis("Energy transfer", "meV", "$w"), "Counts", False)
        )
        hh_1d_gau.append(
            IX_dataset_1d(x, y, e, "Point data, not distribution", IX_axis("Energy transfer", "meV", "$w"), "Counts", False)
        )
        dn = int(np.round(rng.random()))
        y_hp = y[:-dn] if dn > 0 else y
        e_hp = e[:-dn] if dn > 0 else e
        hp_1d_gau.append(
            IX_dataset_1d(x, y_hp, e_hp, "Point data, distribution", IX_axis("Energy transfer", "meV", "$w"), "Counts", True)
        )
    return DataOpArray(hh_1d_gau), DataOpArray(hp_1d_gau), DataOpArray(pp_1d_gau)


def make_testdata_IX_dataset_2d(nx, ny):
    """Create IX_dataset_2d objects with random x and y axes and Gaussian signal (from make_testdata_IX_dataset_2d.m)."""
    rng = np.random.default_rng(12345)
    xrange = 10.0
    yrange = 6.0

    def _gen():
        x = xrange * np.sort(rng.random(nx))
        y = yrange * np.sort(rng.random(ny))
        xx, yy = np.meshgrid(x, y, indexing="ij")
        sig = 10.0 * np.exp(-0.5 * (((xx - xrange / 2) / (xrange / 4)) ** 2 + ((yy - yrange / 2) / (yrange / 4)) ** 2))
        err = 0.5 + rng.random((nx, ny))
        return x, y, sig, err

    x, y, sig, err = _gen()
    hh_gau = IX_dataset_2d(
        x, y, sig[:-1, :-1], err[:-1, :-1], "hist-hist", IX_axis("Energy transfer", "meV", "$w"), "Temperature", "Counts", False, False
    )
    x, y, sig, err = _gen()
    hp_gau = IX_dataset_2d(
        x, y, sig[:-1, :], err[:-1, :], "hist-pnt", IX_axis("Energy transfer", "meV", "$w"), "Temperature", "Counts", False, False
    )
    x, y, sig, err = _gen()
    pp_gau = IX_dataset_2d(
        x, y, sig, err, "pnt-pnt", IX_axis("Energy transfer", "meV", "$w"), "Temperature", "Counts", False, False
    )
    return hh_gau, hp_gau, pp_gau


# ==============================================================================
# IX_dataset_tester (translated from IX_dataset_tester.m)
# ==============================================================================
class IX_dataset_tester(IX_dataset):
    """Class to test some IX_dataset methods."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def init(self, *args):
        raise HoraceError("IX_dataset_tester:not_implemented", "init not implemented")

    def dimensions(self):
        return np.nan, np.nan

    def ishistogram(self, n=None):
        raise HoraceError("IX_dataset_tester:not_implemented", "ishistogram not implemented")

    def axis(self, n=None):
        raise HoraceError("IX_dataset_tester:not_implemented", "axis is not implemented")

    @staticmethod
    def loadobj(data):
        raise HoraceError("IX_dataset_tester:not_implemented", "loadobj not implemented")

    @staticmethod
    def ndim():
        return np.nan

    def _check_and_set_sig_err(self, field_name, value):
        raise HoraceError("IX_dataset_tester:not_implemented", "check_and_set_sig_err not implemented")

    @staticmethod
    def _rebin_hist(iax, wout_x, *args):
        raise HoraceError("IX_dataset_tester:not_implemented", "rebin_hist not implemented")

    @staticmethod
    def _integrate_points(iax, xbounds_true, *args):
        raise HoraceError("IX_dataset_tester:not_implemented", "integrate_points not implemented")


# ==============================================================================
# TestIXDataset1D (translated from test_IX_dataset_1d.m)
# ==============================================================================
class TestIXDataset1D:
    """Test class to test IX_dataset_1d methods."""

    def test_properties(self):
        id_obj = IX_dataset_1d()
        id_obj.title = "my title"
        assert id_obj.title == ["my title"]

        id_obj.x_axis = "Coord"
        ax = id_obj.x_axis
        assert isinstance(ax, IX_axis)
        assert ax.caption == ["Coord"]
        ax.units = "A^-1"
        id_obj.s_axis = ax

        ay = id_obj.s_axis
        assert isinstance(ay, IX_axis)

        id_obj.do_check_combo_arg = False
        id_obj.x = np.arange(1, 11)

        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_1d:invalid_argument" in str(excinfo.value)
        assert (
            "numel(signal)=0, numel(x)=10; numel(signal)  must be equal to numel(x) or numel(x)+1"
            in str(excinfo.value)
        )

        id_obj.signal = np.ones((1, 10))
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_1d:invalid_argument" in str(excinfo.value)
        assert "numel(signal)=10, numel(error)=0; numel(signal)~=numel(error)" in str(excinfo.value)

        id_obj.error = np.ones((1, 10))
        id_obj = id_obj.check_combo_arg()
        id_obj.do_check_combo_arg = True

        val = id_obj.signal
        np.testing.assert_array_equal(val, np.ones((10, 1)))
        np.testing.assert_array_equal(id_obj.error, np.ones((10, 1)))

    def test_fraction_split_NaNs(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        ss = np.ones(10)
        in_idx = [1, 3, 6, 8]  # 1-based [2, 4, 7, 9]
        ss[in_idx] = np.nan
        ds.signal = ss

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 2 / 6)
        assert n_points == 6

    def test_fraction_edge_NaNs(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        ss = np.ones(10)
        in_idx = [1, 4, 8]  # 1-based [2, 5, 9]
        ss[in_idx] = np.nan
        ds.signal = ss

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 5 / 7)
        assert n_points == 7

    def test_fraction_other_NaNs(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        ss = np.ones(10)
        in_idx = [2, 3, 6, 9]  # 1-based [3, 4, 7, 10]
        ss[in_idx] = np.nan
        ds.signal = ss

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 1.0)
        assert n_points == 6

    def test_fraction_some_NaNs(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        ss = np.ones(10)
        in_idx = [1, 2, 6, 9]  # 1-based [2, 3, 7, 10]
        ss[in_idx] = np.nan
        ds.signal = ss

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 5 / 6)
        assert n_points == 6

    def test_fraction_half_NaNs(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        ss = np.ones(10)
        in_idx = np.arange(0, 10, 2)  # 1-based 1:2:10
        ss[in_idx] = np.nan
        ds.signal = ss

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 0.0)
        assert n_points == 5

    def test_fraction_no_NANs(self):
        ds = IX_dataset_1d(np.arange(1, 11))

        frac, n_points = ds.calc_continuous_fraction()
        assert np.isclose(frac, 1.0)
        assert n_points == 10

    def test_constructor_x(self):
        ds = IX_dataset_1d(np.arange(1, 11))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.zeros((10, 1)))
        np.testing.assert_array_equal(ds.error, np.zeros((10, 1)))

    def test_constructor_xs(self):
        ds = IX_dataset_1d(np.arange(1, 11), np.ones(9))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((9, 1)))
        np.testing.assert_array_equal(ds.error, np.zeros((9, 1)))

    def test_constructor_xse(self):
        ds = IX_dataset_1d(np.arange(1, 11), np.ones(10), np.ones(10))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 1)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 1)))

        data = np.vstack([np.arange(1, 11), 2 * np.ones(10), np.ones(10)])
        ds = IX_dataset_1d(data)
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, 2 * np.ones((10, 1)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 1)))

    def test_constructor_xse_title_axis(self):
        ds = IX_dataset_1d(
            np.arange(1, 11), np.ones(10), np.ones(10), "my object", "x-axis name", "y-axis name"
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 1)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 1)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.s_axis.caption == ["y-axis name"]

    def test_constructor_xse_title_axis_distribution(self):
        ds = IX_dataset_1d(
            np.arange(1, 11),
            np.ones(10),
            np.ones(10),
            "my object",
            "x-axis name",
            "y-axis name",
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 1)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 1)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.s_axis.caption == ["y-axis name"]
        assert ds.x_distribution is False

    def test_constructor_xse_title_axis_distr_random_order(self):
        ds = IX_dataset_1d(
            "my object",
            np.ones(10),
            np.ones(10),
            "y-axis name",
            np.arange(1, 11),
            "x-axis name",
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 1)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 1)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.s_axis.caption == ["y-axis name"]
        assert ds.x_distribution is False

    def test_methods(self):
        ds = IX_dataset_1d(
            np.arange(1, 11), np.ones(10), np.ones(10), "my object", "x-axis name", "y-axis name"
        )
        ax, hist = ds.axis()
        assert hist is False
        np.testing.assert_array_equal(ax["values"], np.arange(1, 11))
        assert isinstance(ax["axis"], IX_axis)
        assert ax["distribution"] is True

        dsa = DataOpArray([ds.copy(), ds.copy()])
        dsa[1].x = np.arange(0.5, 11.0, 1.0)

        ax, hist = dsa.axis()
        np.testing.assert_array_equal(hist, [False, True])
        np.testing.assert_array_equal(ax[0]["values"], np.arange(1, 11))
        np.testing.assert_array_equal(ax[1]["values"], np.arange(0.5, 11.0, 1.0))

        is_hist = dsa.ishistogram()
        is_hist1 = dsa.ishistogram(1)
        np.testing.assert_array_equal(is_hist, is_hist1)
        assert not is_hist[0]
        assert is_hist[1]

        ids = dsa.cnt2dist()
        idr = ids.dist2cnt()
        assert len(idr) == 2

    def test_op_managers(self):
        ds = IX_dataset_1d(
            np.arange(1, 11), np.ones(10), np.ones(10), "my object", "x-axis name", "y-axis name"
        )
        dsa = DataOpArray([ds.copy(), ds.copy()])

        dss = dsa[0] + dsa[1]
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 1)))
        np.testing.assert_allclose(dss.error, np.sqrt(2 * np.ones((10, 1))))

        dsm = -ds
        dss = dss + dsm
        np.testing.assert_array_equal(dss.signal, np.ones((10, 1)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 1))))

        dss = dss + 1
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 1)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 1))))

        dss = 1 + dss
        np.testing.assert_array_equal(dss.signal, 3 * np.ones((10, 1)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 1))))


# ==============================================================================
# TestIXDataset2D (translated from test_IX_dataset_2d.m)
# ==============================================================================
class TestIXDataset2D:
    """Test class to test IX_dataset_2d methods."""

    def test_prop_title(self):
        id_obj = IX_dataset_2d()
        id_obj.title = "my title"
        assert id_obj.title == ["my title"]

    def test_prop_x_axis(self):
        id_obj = IX_dataset_2d()
        id_obj.x_axis = "Coord"
        ax = id_obj.x_axis
        assert isinstance(ax, IX_axis)
        assert ax.caption == ["Coord"]
        ax.units = "A^-1"
        id_obj.s_axis = ax

        ay = id_obj.s_axis
        assert isinstance(ay, IX_axis)

    def test_prop_y_axis(self):
        id_obj = IX_dataset_2d()
        id_obj.y_axis = "dist"
        ay = id_obj.y_axis
        assert isinstance(ay, IX_axis)
        assert ay.caption == ["dist"]
        ay.units = "A^-1"
        id_obj.y_axis = ay
        assert isinstance(id_obj.y_axis, IX_axis)
        assert id_obj.y_axis.caption == ["dist"]

    def test_combo_properties_set(self):
        id_obj = IX_dataset_2d()
        id_obj.do_check_combo_arg = False
        id_obj.x = np.arange(1, 11)
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_2d:invalid_argument" in str(excinfo.value)
        assert (
            "size(signal,1)=0, numel(x)=10; size(signal,1) must be equal to numel(x) or numel(x)+1"
            in str(excinfo.value)
        )

        id_obj.signal = np.ones((20, 10))
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_2d:invalid_argument" in str(excinfo.value)
        assert "size(signal)=[20,10], size(error)=[0,1]; size(signal)~=size(error)" in str(
            excinfo.value
        )

        id_obj.error = np.ones((20, 10))
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_2d:invalid_argument" in str(excinfo.value)
        assert (
            "size(signal,2)=20, numel(y)=0; size(signal,2)  must be equal to numel(y) or numel(y)+1"
            in str(excinfo.value)
        )

        id_obj.y = np.arange(1, 21)
        id_obj = id_obj.check_combo_arg()
        id_obj.do_check_combo_arg = True

        val = id_obj.signal
        np.testing.assert_array_equal(val, np.ones((10, 20)))
        np.testing.assert_array_equal(id_obj.error, np.ones((10, 20)))

    def test_constructor_xy(self):
        ds = IX_dataset_2d(np.arange(1, 11), np.arange(1, 21))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 21))
        np.testing.assert_array_equal(ds.signal, np.zeros((10, 20)))
        np.testing.assert_array_equal(ds.error, np.zeros((10, 20)))

    def test_constructor_xys(self):
        ds = IX_dataset_2d(np.arange(1, 11), np.arange(1, 21), np.ones((9, 19)))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 21))
        np.testing.assert_array_equal(ds.signal, np.ones((9, 19)))
        np.testing.assert_array_equal(ds.error, np.zeros((9, 19)))

    def test_constructor_xyse(self):
        ds = IX_dataset_2d(
            np.arange(1, 11), np.arange(1, 21), np.ones((10, 20)), np.ones((10, 20))
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 21))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 20)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 20)))

    def test_constructor_xyse_title_axis(self):
        ds = IX_dataset_2d(
            np.arange(1, 21),
            np.arange(1, 11),
            np.ones((20, 10)),
            np.ones((20, 10)),
            "my object",
            "x-axis name",
            "y-axis name",
            "signal",
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 21))
        np.testing.assert_array_equal(ds.y, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((20, 10)))
        np.testing.assert_array_equal(ds.error, np.ones((20, 10)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.y_axis.caption == ["y-axis name"]
        assert ds.s_axis.caption == ["signal"]

    def test_constructor_xyse_title_axis_no_distr(self):
        ds = IX_dataset_2d(
            np.arange(1, 21),
            np.arange(1, 11),
            np.ones((20, 10)),
            np.ones((20, 10)),
            "my object",
            "x-axis name",
            "y-axis name",
            "signal",
            False,
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 21))
        np.testing.assert_array_equal(ds.y, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((20, 10)))
        np.testing.assert_array_equal(ds.error, np.ones((20, 10)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.y_axis.caption == ["y-axis name"]
        assert ds.s_axis.caption == ["signal"]
        assert ds.x_distribution is False
        assert ds.y_distribution is False

    def test_constructor_xyse_title_axis_names(self):
        ds = IX_dataset_2d(
            "my object",
            np.ones((15, 10)),
            np.ones((15, 10)),
            "signal",
            np.arange(1, 16),
            "x-axis name",
            False,
            np.arange(1, 11),
            "y-axis name",
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 16))
        np.testing.assert_array_equal(ds.y, np.arange(1, 11))
        np.testing.assert_array_equal(ds.signal, np.ones((15, 10)))
        np.testing.assert_array_equal(ds.error, np.ones((15, 10)))
        assert ds.title == ["my object"]
        assert ds.x_axis.caption == ["x-axis name"]
        assert ds.y_axis.caption == ["y-axis name"]
        assert ds.s_axis.caption == ["signal"]
        assert ds.x_distribution is False
        assert ds.y_distribution is False

    def test_methods(self):
        ds = IX_dataset_2d(
            np.arange(1, 11),
            np.arange(1, 16),
            np.ones((10, 15)),
            np.ones((10, 15)),
            "my object",
            "x-axis name",
            "y-axis name",
            "signal",
        )
        ax, hist = ds.axis(2)
        assert hist is False
        np.testing.assert_array_equal(ax["values"], np.arange(1, 16))
        assert isinstance(ax["axis"], IX_axis)
        assert ax["distribution"] is True

        dsa = DataOpArray([ds.copy(), ds.copy()])
        dsa[1].x = np.arange(0.5, 11.0, 1.0)

        ax, hist = dsa[0].axis()
        assert not np.any(hist)
        np.testing.assert_array_equal(ax[0]["values"], np.arange(1, 11))
        np.testing.assert_array_equal(ax[1]["values"], np.arange(1, 16))

        is_hist = dsa.ishistogram()
        is_hist1 = dsa.ishistogram(1)
        is_hist2 = dsa.ishistogram(2)
        np.testing.assert_array_equal(is_hist, np.vstack([is_hist1, is_hist2]))
        assert not is_hist[0, 0]
        assert is_hist[0, 1]
        assert not is_hist[1, 0]
        assert not is_hist[1, 1]

        ids = dsa.point2hist()
        idr = ids.hist2point()
        assert len(idr) == 2

    def test_op_managers(self):
        ds = IX_dataset_2d(
            np.arange(1, 11),
            np.arange(1, 21),
            np.ones((10, 20)),
            np.ones((10, 20)),
            "my object",
            "x-axis name",
            "y-axis name",
            "signal",
        )
        dsa = DataOpArray([ds.copy(), ds.copy()])

        dss = dsa[0] + dsa[1]
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 20)))
        np.testing.assert_allclose(dss.error, np.sqrt(2 * np.ones((10, 20))))

        dsm = -ds
        dss = dss + dsm
        np.testing.assert_array_equal(dss.signal, np.ones((10, 20)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 20))))

        dss = dss + 1
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 20)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 20))))

        dss = 1 + dss
        np.testing.assert_array_equal(dss.signal, 3 * np.ones((10, 20)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 20))))


# ==============================================================================
# TestIXDataset3D (translated from test_IX_dataset_3d.m)
# ==============================================================================
class TestIXDataset3D:
    """Test class to test IX_dataset_3d methods."""

    def test_properties_title(self):
        id_obj = IX_dataset_3d()
        id_obj.title = "my title"
        assert id_obj.title == ["my title"]

    def test_properties_x_axis(self):
        id_obj = IX_dataset_3d()
        id_obj.x_axis = "Coord"
        ax = id_obj.x_axis
        assert isinstance(ax, IX_axis)
        assert ax.caption == ["Coord"]
        ax.units = "A^-1"
        id_obj.s_axis = ax

        ay = id_obj.s_axis
        assert isinstance(ay, IX_axis)

    def test_properties_y_axis(self):
        id_obj = IX_dataset_3d()
        id_obj.y_axis = "dist"
        ay = id_obj.y_axis
        assert isinstance(ay, IX_axis)
        assert ay.caption == ["dist"]
        ay.units = "A^-1"
        id_obj.y_axis = ay
        assert isinstance(id_obj.y_axis, IX_axis)
        assert id_obj.y_axis.caption == ["dist"]

    def test_combo_properties_set(self):
        id_obj = IX_dataset_3d()
        id_obj.do_check_combo_arg = False

        id_obj.x = np.arange(1, 11)
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_3d:invalid_argument" in str(excinfo.value)
        assert (
            "size(signal,1)=0, numel(x)=10; size(signal,1) must be equal to numel(x) or numel(x)+1"
            in str(excinfo.value)
        )

        id_obj.signal = np.ones((10, 20))
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_3d:invalid_argument" in str(excinfo.value)
        assert "size(signal)=[10,20,1], size(error)=[0,0,0]; size(signal)~=size(error)" in str(
            excinfo.value
        )

        id_obj.error = np.ones((20, 10))
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_3d:invalid_argument" in str(excinfo.value)
        assert "size(signal)=[10,20,1], size(error)=[20,10,1]; size(signal)~=size(error)" in str(
            excinfo.value
        )

        id_obj.error = np.ones((10, 20))
        id_obj.y = np.arange(1, 21)
        with pytest.raises((ValueError, HoraceError)) as excinfo:
            id_obj.check_combo_arg()
        assert "HERBERT:IX_data_3d:invalid_argument" in str(excinfo.value)
        assert (
            "size(signal,3)=1, numel(z)=0; size(signal,3) must be equal to numel(z) or numel(z)+1"
            in str(excinfo.value)
        )

        id_obj.z = 0.5
        id_obj.do_check_combo_arg = False
        id_obj = id_obj.check_combo_arg()

        val = id_obj.signal
        assert not isinstance(val, str)
        np.testing.assert_array_equal(val.squeeze(), np.ones((10, 20)))
        np.testing.assert_array_equal(id_obj.error.squeeze(), np.ones((10, 20)))

    def test_constructor_xyz(self):
        ds = IX_dataset_3d(np.arange(1, 11), np.arange(1, 6), np.arange(1, 8))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.zeros((10, 5, 7)))
        np.testing.assert_array_equal(ds.error, np.zeros((10, 5, 7)))

    def test_constructor_xyzs(self):
        ds = IX_dataset_3d(np.arange(1, 11), np.arange(1, 6), np.arange(1, 8), np.ones((9, 4, 6)))
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.ones((9, 4, 6)))
        np.testing.assert_array_equal(ds.error, np.zeros((9, 4, 6)))

    def test_constructor_xyzse(self):
        ds = IX_dataset_3d(
            np.arange(1, 11),
            np.arange(1, 6),
            np.arange(1, 8),
            np.ones((10, 5, 7)),
            np.ones((10, 5, 7)),
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 5, 7)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 5, 7)))

    def test_constructor_xyzse_title_ax(self):
        ds = IX_dataset_3d(
            np.arange(1, 11),
            np.arange(1, 6),
            np.arange(1, 8),
            np.ones((10, 5, 7)),
            np.ones((10, 5, 7)),
            "my 3D obj",
            "x-axis",
            "y-axis",
            "z-axis",
            "signal",
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 5, 7)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 5, 7)))
        assert ds.title == ["my 3D obj"]
        assert ds.x_axis.caption == ["x-axis"]
        assert ds.y_axis.caption == ["y-axis"]
        assert ds.z_axis.caption == ["z-axis"]
        assert ds.s_axis.caption == ["signal"]

    def test_constructor_xyzse_title_ax_distr(self):
        ds = IX_dataset_3d(
            np.arange(1, 11),
            np.arange(1, 6),
            np.arange(1, 8),
            np.ones((10, 5, 7)),
            np.ones((10, 5, 7)),
            "my 3D obj",
            "x-axis",
            "y-axis",
            "z-axis",
            "signal",
            False,
            False,
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 5, 7)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 5, 7)))
        assert ds.title == ["my 3D obj"]
        assert ds.x_axis.caption == ["x-axis"]
        assert ds.y_axis.caption == ["y-axis"]
        assert ds.z_axis.caption == ["z-axis"]
        assert ds.s_axis.caption == ["signal"]
        assert ds.x_distribution is False
        assert ds.y_distribution is False
        assert ds.z_distribution is False

    def test_constructor_title_se_ax_distr(self):
        ds = IX_dataset_3d(
            "my 3D obj",
            np.ones((10, 5, 7)),
            np.ones((10, 5, 7)),
            "signal",
            np.arange(1, 11),
            "x-axis",
            False,
            np.arange(1, 6),
            "y-axis",
            False,
            np.arange(1, 8),
            "z-axis",
            False,
        )
        np.testing.assert_array_equal(ds.x, np.arange(1, 11))
        np.testing.assert_array_equal(ds.y, np.arange(1, 6))
        np.testing.assert_array_equal(ds.z, np.arange(1, 8))
        np.testing.assert_array_equal(ds.signal, np.ones((10, 5, 7)))
        np.testing.assert_array_equal(ds.error, np.ones((10, 5, 7)))
        assert ds.title == ["my 3D obj"]
        assert ds.x_axis.caption == ["x-axis"]
        assert ds.y_axis.caption == ["y-axis"]
        assert ds.z_axis.caption == ["z-axis"]
        assert ds.s_axis.caption == ["signal"]
        assert ds.x_distribution is False
        assert ds.y_distribution is False
        assert ds.z_distribution is False

    def test_methods(self):
        ds = IX_dataset_3d(
            np.arange(1, 11),
            np.arange(1, 6),
            np.arange(1, 8),
            np.ones((10, 5, 7)),
            np.ones((10, 5, 7)),
            "my object",
            "x-axis",
            "y-axis",
            "z-axis",
            "signal",
        )
        ax, hist = ds.axis(2)
        assert hist is False
        np.testing.assert_array_equal(ax["values"], np.arange(1, 6))
        assert isinstance(ax["axis"], IX_axis)
        assert ax["distribution"] is True

        dsa = DataOpArray([ds.copy(), ds.copy()])
        dsa[1].x = np.arange(0.5, 11.0, 1.0)

        ax, hist = dsa[0].axis()
        assert not np.any(hist)
        np.testing.assert_array_equal(ax[0]["values"], np.arange(1, 11))
        np.testing.assert_array_equal(ax[1]["values"], np.arange(1, 6))
        np.testing.assert_array_equal(ax[2]["values"], np.arange(1, 8))

        is_hist = dsa.ishistogram()
        is_hist1 = dsa.ishistogram(1)
        is_hist2 = dsa.ishistogram(2)
        is_hist3 = dsa.ishistogram(3)
        np.testing.assert_array_equal(is_hist, np.vstack([is_hist1, is_hist2, is_hist3]))
        assert not is_hist[0, 0]
        assert is_hist[0, 1]
        assert not is_hist[1, 0]
        assert not is_hist[1, 1]
        assert not is_hist[2, 0]
        assert not is_hist[2, 1]

        ids = dsa.point2hist()
        idr = ids.hist2point()
        assert len(idr) == 2

    def test_op_managers(self):
        ds = IX_dataset_3d(
            np.arange(1, 11),
            np.arange(1, 6),
            np.arange(1, 16),
            np.ones((10, 5, 15)),
            np.ones((10, 5, 15)),
            "test 3D object",
            "x-axis",
            "y-axis",
            "z-axis",
            "signal",
        )
        dsa = DataOpArray([ds.copy(), ds.copy()])

        dss = dsa[0] + dsa[1]
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 5, 15)))
        np.testing.assert_allclose(dss.error, np.sqrt(2 * np.ones((10, 5, 15))))

        dsm = -ds
        dss = dss + dsm
        np.testing.assert_array_equal(dss.signal, np.ones((10, 5, 15)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 5, 15))))

        dss = dss + 1
        np.testing.assert_array_equal(dss.signal, 2 * np.ones((10, 5, 15)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 5, 15))))

        dss = 1 + dss
        np.testing.assert_array_equal(dss.signal, 3 * np.ones((10, 5, 15)))
        np.testing.assert_allclose(dss.error, np.sqrt(3 * np.ones((10, 5, 15))))


# ==============================================================================
# TestUtilitiesAndManipulationMethods
# ==============================================================================
class TestUtilitiesAndManipulationMethods:
    """Test helper functions and manipulation methods on 1D, 2D, and 3D datasets."""

    def test_make_IX_dataset_nd_random(self):
        w1 = make_IX_dataset_nd_random([17])
        assert isinstance(w1, IX_dataset_1d)
        assert w1.signal.shape == (17, 1)

        w2 = make_IX_dataset_nd_random([3, 4])
        assert isinstance(w2, IX_dataset_2d)
        assert w2.signal.shape == (3, 4)

        w3 = make_IX_dataset_nd_random([5, 4, 3])
        assert isinstance(w3, IX_dataset_3d)
        assert w3.signal.shape == (5, 4, 3)

    def test_make_testdata_1d_and_2d(self):
        hh1, hp1, pp1 = make_testdata_IX_dataset_1d(10, 3)
        assert len(hh1) == 3
        assert len(hp1) == 3
        assert len(pp1) == 3

        hh2, hp2, pp2 = make_testdata_IX_dataset_2d(8, 6)
        assert isinstance(hh2, IX_dataset_2d)
        assert isinstance(hp2, IX_dataset_2d)
        assert isinstance(pp2, IX_dataset_2d)
        assert np.array_equal(hh2.ishistogram(), [True, True])
        assert np.array_equal(hp2.ishistogram(), [True, False])
        assert np.array_equal(pp2.ishistogram(), [False, False])

    def test_compact_array(self):
        ca = compact_array([[1, 2, 4], [3, 5]], [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])])
        assert ca.n_unique == 2
        assert ca.n_objects == 5
        val, _, u_idx, was_used = ca.get(4)
        np.testing.assert_array_equal(val, np.array([1.0, 2.0, 3.0]))
        assert u_idx == 1
        assert not was_used

    def test_1d_manipulation_methods(self):
        x = np.linspace(0.0, 10.0, 11)
        s = 2.0 * np.arange(1, 11, dtype=float)
        e = 0.5 * np.ones(10)
        w_hist = IX_dataset_1d(x, s, e, "Hist 1D", "X", "Signal", True)

        # rebin & integrate & cut
        w_reb = w_hist.rebin([0.0, 2.0, 10.0])
        assert len(w_reb.x) == 6
        w_int = w_hist.integrate([2.0, 8.0])
        assert np.isscalar(w_int["val"]) or np.size(w_int["val"]) == 1
        w_cut = w_hist.cut([2.0, 2.0, 8.0])
        assert isinstance(w_cut, IX_dataset_1d)

        # scale, shift, rebunch, smooth, mask, xye, moments
        w_scaled = w_hist.scale(2.0)
        np.testing.assert_allclose(w_scaled.x, 2.0 * x)
        w_shifted = w_hist.shift(1.5)
        np.testing.assert_allclose(w_shifted.x, x + 1.5)
        w_rebunched = w_hist.rebunch(2)
        assert len(w_rebunched.signal) == 5

        w_pt = w_hist.hist2point()
        w_der = w_pt.deriv()
        np.testing.assert_allclose(w_der.signal.ravel(), 2.0)

        w_masked = w_pt.mask([True, False] * 5)
        assert np.isnan(w_masked.signal[1, 0])

        xye_res = w_pt.xye()
        assert "x" in xye_res and "y" in xye_res and "e" in xye_res

    def test_2d_and_3d_manipulation_methods(self):
        x = np.linspace(0.0, 10.0, 11)
        y = np.linspace(0.0, 6.0, 7)
        s = np.ones((10, 6))
        e = 0.1 * np.ones((10, 6))
        w2 = IX_dataset_2d(x, y, s, e, "Hist 2D", "X", "Y", "Signal", True, True)

        w2_reb = w2.rebin([0.0, 2.0, 10.0], [0.0, 2.0, 6.0])
        assert w2_reb.signal.shape == (5, 3)

        w1_int = w2.integrate_x([2.0, 8.0])
        assert isinstance(w1_int, IX_dataset_1d)

        w2_scaled = w2.scale(2.0, 3.0)
        np.testing.assert_allclose(w2_scaled.x, 2.0 * x)
        np.testing.assert_allclose(w2_scaled.y, 3.0 * y)

        w2_shifted = w2.shift(1.0, -1.0)
        np.testing.assert_allclose(w2_shifted.x, x + 1.0)
        np.testing.assert_allclose(w2_shifted.y, y - 1.0)

        # expand_d1d and expand_d2d
        w1_arr = w2.expand_d1d()
        assert len(w1_arr) == 6
        assert isinstance(w1_arr[0], IX_dataset_1d)

        w3 = IX_dataset_3d(x, y, np.array([1.0, 2.0]), np.ones((10, 6, 2)), 0.1 * np.ones((10, 6, 2)))
        w2_arr = w3.expand_d2d()
        assert len(w2_arr) == 2
        assert isinstance(w2_arr[0], IX_dataset_2d)
