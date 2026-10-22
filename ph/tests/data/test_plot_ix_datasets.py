"""Unit tests for genieplot, data_plot_interface, and plotting of IX_dataset_1d/2d/3d.

Translates test_plot_singleton.m, test_plot_IX_dataset.m, test_plot_IX_dataset_2.m,
and test_plot_IX_dataset_3.m from _test/test_IX_classes using Matplotlib (Agg backend).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyhorace.data import (
    DataOpArray,
    HoraceError,
    IX_dataset_1d,
    IX_dataset_2d,
    IX_dataset_3d,
    acolor,
    aline,
    amark,
    clear_figures,
    da,
    dd,
    de,
    dh,
    dl,
    dm,
    dp,
    ds,
    ds2,
    genieplot,
    is_genie_figure,
    keep_figure,
    lc,
    linx,
    liny,
    linz,
    logx,
    logy,
    logz,
    lx,
    ly,
    lz,
    make_current,
    pa,
    paoc,
    pd,
    pdoc,
    pe,
    peoc,
    ph,
    phoc,
    pl,
    ploc,
    plot,
    plotover,
    pm,
    pmoc,
    pp,
    ppoc,
    ps,
    ps2,
    ps2oc,
    psoc,
    sliceomatic,
    sliceomatic_overview,
    spaghetti_plot,
    src,
)
from pyhorace.data import DataPlotInterface


# ==============================================================================
# Helpers & Fixtures
# ==============================================================================
class DataPlotInterfaceTester(DataPlotInterface):
    """Concrete tester for DataPlotInterface that matches data_plot_interface_tester.m."""

    methodsND_plot = [plot]
    methodsND_plotOver = [plotover]
    methods1D_plot = [dd, de, dh, dl, dm, dp]
    methods1D_plotOver = [pd, pe, ph, pl, pm, pp]
    methods1D_plotOverCurr = [pdoc, peoc, phoc, ploc, pmoc, ppoc]
    methods2D_plot = [da, ds, ds2]
    methods2D_plotOver = [pa, ps, ps2]
    methods2D_plotOverCurr = [paoc, psoc, ps2oc]
    methods3D_plot = [sliceomatic, sliceomatic_overview]

    def dimensions(self):
        return 0, ()


def make_data():
    """Make a selection of datasets for testing plot methods (from test_plot_IX_dataset_2.m)."""
    # One-dimensional datasets
    x1_1_point = np.arange(1, 11, dtype=float)
    x1_1_hist = np.arange(0, 11, dtype=float) + 0.5
    y1_1 = np.arange(40, 140, 10, dtype=float)
    e1_1 = 5.0 + np.arange(20, 0, -2, dtype=float)
    w1_1_point = IX_dataset_1d(
        x1_1_point,
        y1_1,
        e1_1,
        "Title: w1_1_point",
        "xlabel: w1_1_point - x",
        "signal: w1_1_point - signal",
    )
    w1_1_hist = IX_dataset_1d(
        x1_1_hist,
        y1_1,
        e1_1,
        "Title: w1_1_hist",
        "xlabel: w1_1_hist - x",
        "signal: w1_1_hist - signal",
    )

    x1_2_point = np.arange(5, 15, dtype=float)
    x1_2_hist = np.arange(4, 15, dtype=float) + 0.5
    y1_2 = np.arange(150, 250, 10, dtype=float)
    e1_2 = 5.0 + np.arange(50, 0, -5, dtype=float)
    w1_2_point = IX_dataset_1d(
        x1_2_point,
        y1_2,
        e1_2,
        "Title: w1_2_point",
        "xlabel: w1_2_point - x",
        "signal: w1_2_point - signal",
    )
    w1_2_hist = IX_dataset_1d(
        x1_2_hist,
        y1_2,
        e1_2,
        "Title: w1_2_hist",
        "xlabel: w1_2_hist - x",
        "signal: w1_2_hist - signal",
    )

    w1_point = DataOpArray([w1_1_point, w1_2_point])
    w1_hist = DataOpArray([w1_1_hist, w1_2_hist])

    # Two-dimensional datasets
    w2oneD_point_xy = IX_dataset_2d(
        np.arange(11, 21, dtype=float), np.array([5.0]), np.arange(101, 121, 2, dtype=float)
    )
    w2oneD_point_x_hist_y = IX_dataset_2d(
        np.arange(11, 21, dtype=float), np.array([2.0, 8.0]), np.arange(101, 121, 2, dtype=float)
    )

    x2_1 = np.arange(1, 16, dtype=float)
    y2_1 = np.arange(101, 116, dtype=float)
    xx2_1, yy2_1 = np.meshgrid(x2_1, y2_1, indexing="ij")
    s2_1 = 2 * xx2_1 + yy2_1
    e2_1 = np.exp(-((xx2_1 - 5) ** 2 + (yy2_1 - 110) ** 2) / 4)
    w2_1_point = IX_dataset_2d(
        x2_1,
        y2_1,
        s2_1,
        e2_1,
        "Title: w2_1_point",
        "xlabel: w2_1_point - x",
        "ylabel: w2_1_point - y",
        "signal: w2_1_point - signal",
    )

    x2_2 = np.arange(11, 31, dtype=float)
    y2_2 = np.arange(120, 131, dtype=float)
    xx2_2, yy2_2 = np.meshgrid(x2_2, y2_2, indexing="ij")
    s2_2 = -2 * xx2_2 + yy2_2
    e2_2 = 2 * np.exp(-((xx2_2 - 20) ** 2 + (yy2_2 - 125) ** 2) / 8)
    w2_2_point = IX_dataset_2d(
        x2_2,
        y2_2,
        s2_2,
        e2_2,
        "Title: w2_2_point",
        "xlabel: w2_2_point - x",
        "ylabel: w2_2_point - y",
        "signal: w2_2_point - signal",
    )

    w2_point = DataOpArray([w2_1_point, w2_2_point])

    x2_3 = np.arange(1, 16, dtype=float)
    x2_3_bins = np.arange(0, 16, dtype=float) + 0.5
    y2_3 = np.arange(101, 116, dtype=float)
    y2_3_bins = np.arange(100, 116, dtype=float) + 0.5
    xx2_3, yy2_3 = np.meshgrid(x2_3, y2_3, indexing="ij")
    s2_3 = np.exp(-((xx2_3 - 5) ** 2 + (yy2_3 - 110) ** 2) / 4)
    e2_3 = 2 * xx2_3 + yy2_3
    w2_1_hist = IX_dataset_2d(
        x2_3_bins,
        y2_3_bins,
        s2_3,
        e2_3,
        "Title: w2_1_hist",
        "xlabel: w2_1_hist - x",
        "ylabel: w2_1_hist - y",
        "signal: w2_1_hist - signal",
    )

    x2_4 = np.arange(11, 31, dtype=float)
    x2_4_bins = np.arange(10, 31, dtype=float) + 0.5
    y2_4 = np.arange(120, 131, dtype=float)
    y2_4_bins = np.arange(119, 131, dtype=float) + 0.5
    xx2_4, yy2_4 = np.meshgrid(x2_4, y2_4, indexing="ij")
    s2_4 = 2 * np.exp(-((xx2_4 - 20) ** 2 + (yy2_4 - 125) ** 2) / 8)
    e2_4 = -2 * xx2_4 + yy2_4
    w2_2_hist = IX_dataset_2d(
        x2_4_bins,
        y2_4_bins,
        s2_4,
        e2_4,
        "Title: w2_2_hist",
        "xlabel: w2_2_hist - x",
        "ylabel: w2_2_hist - y",
        "signal: w2_2_hist - signal",
    )

    w2_hist = DataOpArray([w2_1_hist, w2_2_hist])

    # Three-dimensional datasets
    x3_1 = np.arange(1, 4, dtype=float)
    y3_1 = np.arange(101, 105, dtype=float)
    z3_1 = np.arange(201, 206, dtype=float)
    xx3_1, yy3_1, zz3_1 = np.meshgrid(x3_1, y3_1, z3_1, indexing="ij")
    s3_1 = 2 * xx3_1 + 3 * (yy3_1 - 100) + (zz3_1 - 200)
    e3_1 = np.exp(-((xx3_1 - 2) ** 2 + (yy3_1 - 102) ** 2 + (zz3_1 - 203) ** 2) / 4)
    w3_1_point = IX_dataset_3d(x3_1, y3_1, z3_1, s3_1, e3_1)
    w3_1_point.x_axis = "xlabel: w3_1_point - x"
    w3_1_point.y_axis = "ylabel: w3_1_point - y"
    w3_1_point.z_axis = "zlabel: w3_1_point - z"
    w3_1_point.title = ["Title: w3_1_point"]

    x3_2 = np.arange(11, 15, dtype=float)
    y3_2 = np.arange(111, 116, dtype=float)
    z3_2 = np.arange(211, 217, dtype=float)
    xx3_2, yy3_2, zz3_2 = np.meshgrid(x3_2, y3_2, z3_2, indexing="ij")
    s3_2 = 2 * (xx3_2 - 10) + 3 * (yy3_2 - 110) + (zz3_2 - 210)
    e3_2 = np.exp(-((xx3_2 - 12) ** 2 + (yy3_2 - 113) ** 2 + (zz3_2 - 213) ** 2) / 4)
    w3_2_point = IX_dataset_3d(x3_2, y3_2, z3_2, s3_2, e3_2)
    w3_2_point.x_axis = "xlabel: w3_2_point - x"
    w3_2_point.y_axis = "ylabel: w3_2_point - y"
    w3_2_point.z_axis = "zlabel: w3_2_point - z"
    w3_2_point.title = ["Title: w3_2_point"]

    w3_point = DataOpArray([w3_1_point, w3_2_point])
    w3_hist = DataOpArray([w3_1_point.copy(), w3_2_point.copy()])

    # Small 3D dataset with unequal step sizes
    x3_unequal = np.array([1.0, 2.0, 4.0])
    y3_unequal = np.array([101.0, 103.0, 104.0, 107.0])
    z3_unequal = np.array([201.0, 201.5, 203.0, 204.0, 206.0])
    xx3_u, yy3_u, zz3_u = np.meshgrid(x3_unequal, y3_unequal, z3_unequal, indexing="ij")
    s3_unequal = 2 * xx3_u + 3 * (yy3_u - 100) + (zz3_u - 200)
    w3_unequal = IX_dataset_3d(x3_unequal, y3_unequal, z3_unequal, s3_unequal)
    w3_unequal.x_axis = "x-axis caption"
    w3_unequal.y_axis = "y-axis caption"
    w3_unequal.z_axis = "z-axis caption"
    w3_unequal.title = ["The Title"]

    return (
        w1_point,
        w1_hist,
        w2_point,
        w2_hist,
        w3_point,
        w3_hist,
        w2oneD_point_xy,
        w2oneD_point_x_hist_y,
        w3_unequal,
    )


# ==============================================================================
# TestPlotSingleton (translated from test_plot_singleton.m)
# ==============================================================================
class TestPlotSingleton:
    """Test genieplot singleton."""

    S_default = {
        "XScale": "linear",
        "YScale": "linear",
        "ZScale": "linear",
        "color_cycle": "with",
        "colors": ["k"],
        "default_fig_name": None,
        "line_styles": ["-"],
        "line_widths": 0.5,
        "marker_sizes": 6,
        "marker_types": ["o"],
        "maxspec_1D": 1000,
        "maxspec_2D": 1000,
        "use_original_horace_plot_colours": True,
    }

    def setup_method(self):
        self.current_graphics_settings = genieplot.get()

    def teardown_method(self):
        genieplot.set(self.current_graphics_settings)
        clear_figures()

    def test_get_all(self):
        S = genieplot.get()
        assert isinstance(S, dict)
        assert sorted(S.keys()) == sorted(self.S_default.keys())

    def test_set_all(self):
        S1 = dict(self.S_default)
        S2 = dict(self.S_default)
        S1["colors"] = ["k", "#121314", "y"]
        S2["colors"] = ["g", "b", "y"]

        genieplot.set(S1)
        S1_retrieved = genieplot.get()
        genieplot.set(S2)
        S2_retrieved = genieplot.get()

        assert S1_retrieved == S1
        assert S2_retrieved == S2

    def test_reset(self):
        genieplot.reset()
        S = genieplot.get()
        assert S == self.S_default

        S1 = dict(S)
        S1["colors"] = ["g", "b", "y"]
        genieplot.set(S1)
        assert genieplot.get() != self.S_default

        genieplot.reset()
        S2 = genieplot.get()
        assert S2 == self.S_default

    def test_set_nonExistentProperty_ERROR(self):
        with pytest.raises((Exception,)):
            genieplot.set("local_only", True)

    def test_get_nonExistentProperty_ERROR(self):
        with pytest.raises((Exception,)):
            genieplot.get("local_only")

    def test_setAndGet_XScale(self):
        genieplot.reset()
        genieplot.set("XScale", "linear")
        assert genieplot.get("XScale") == "linear"
        genieplot.set("XScale", "log")
        assert genieplot.get("XScale") == "log"

    def test_setAndGet_XScale_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("XScale", "Peter")
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_YScale(self):
        genieplot.reset()
        genieplot.set("YScale", "linear")
        assert genieplot.get("YScale") == "linear"
        genieplot.set("YScale", "log")
        assert genieplot.get("YScale") == "log"

    def test_setAndGet_YScale_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("YScale", False)
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_ZScale(self):
        genieplot.reset()
        genieplot.set("ZScale", "linear")
        assert genieplot.get("ZScale") == "linear"
        genieplot.set("ZScale", "log")
        assert genieplot.get("ZScale") == "log"

    def test_setAndGet_ZScale_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("ZScale", [34, 35])
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_colors(self):
        genieplot.reset()
        genieplot.set("colors", ["b"])
        assert genieplot.get("colors") == ["b"]
        genieplot.set("colors", ["k", "#121314", "y"])
        assert genieplot.get("colors") == ["k", "#121314", "y"]

    def test_setAndGet_colors_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("colors", 666)
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_color_cycle(self):
        genieplot.reset()
        genieplot.set("color_cycle", "with")
        assert genieplot.get("color_cycle") == "with"
        genieplot.set("color_cycle", "fast")
        assert genieplot.get("color_cycle") == "fast"

    def test_setAndGet_color_cycle_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("color_cycle", 666)
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_default_fig_name(self):
        genieplot.reset()
        genieplot.set("default_fig_name", "A silly name")
        assert genieplot.get("default_fig_name") == "A silly name"
        genieplot.set("default_fig_name", "Lets boogie")
        assert genieplot.get("default_fig_name") == "Lets boogie"

    def test_setAndGet_default_fig_name_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("default_fig_name", 37)
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_line_styles(self):
        genieplot.reset()
        genieplot.set("line_styles", "--")
        assert genieplot.get("line_styles") in ("--", ["--"])
        genieplot.set("line_styles", ["-", "-."])
        assert genieplot.get("line_styles") == ["-", "-."]

    def test_setAndGet_line_styles_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("line_styles", "ddot")
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_line_widths(self):
        genieplot.reset()
        genieplot.set("line_widths", 1.5)
        np.testing.assert_allclose(genieplot.get("line_widths"), 1.5)
        genieplot.set("line_widths", [0.4, 4.6])
        np.testing.assert_allclose(genieplot.get("line_widths"), [0.4, 4.6])

    def test_setAndGet_line_widths_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("line_widths", "thick")
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_marker_sizes(self):
        genieplot.reset()
        genieplot.set("marker_sizes", 1.5)
        np.testing.assert_allclose(genieplot.get("marker_sizes"), 1.5)
        genieplot.set("marker_sizes", [0.4, 4.6])
        np.testing.assert_allclose(genieplot.get("marker_sizes"), [0.4, 4.6])

    def test_setAndGet_marker_sizes_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("marker_sizes", "big")
        assert "invalid_argument" in str(excinfo.value)

    def test_setAndGet_marker_types(self):
        genieplot.reset()
        genieplot.set("marker_types", "^")
        assert genieplot.get("marker_types") in ("^", ["^"])
        genieplot.set("marker_types", ["+", "p"])
        assert genieplot.get("marker_types") == ["+", "p"]

    def test_setAndGet_marker_types_ERROR(self):
        with pytest.raises(Exception) as excinfo:
            genieplot.set("marker_types", "diamond")
        assert "invalid_argument" in str(excinfo.value)

    @pytest.mark.parametrize("prop", ["maxspec_1D", "maxspec_2D"])
    def test_setAndGet_maxspec(self, prop):
        genieplot.reset()
        genieplot.set(prop, 468)
        assert genieplot.get(prop) == 468

        for bad_val in [0, 314.15, -3, -np.inf, np.nan]:
            with pytest.raises(Exception) as excinfo:
                genieplot.set(prop, bad_val)
            assert "invalid_argument" in str(excinfo.value)

    def test_setterGetterSyntax_interoperability(self):
        genieplot.reset()
        a = genieplot.instance()
        b = genieplot.instance()

        assert a.colors == b.colors == genieplot.get("colors") == self.S_default["colors"]

        new_colors = ["b", "g", "r"]
        genieplot.set("colors", new_colors)
        assert a.colors == b.colors == genieplot.get("colors") == new_colors

        new_colors2 = ["r", "y", "k"]
        a.colors = new_colors2
        assert a.colors == b.colors == genieplot.get("colors") == new_colors2

        a.reset()
        assert a.colors == b.colors == genieplot.get("colors") == self.S_default["colors"]


# ==============================================================================
# TestPlotIXDataset3 (translated from test_plot_IX_dataset_3.m)
# ==============================================================================
def _genieplot_initialise(color_cycle):
    genieplot.set("color_cycle", color_cycle)
    genieplot.set("colors", ["#0071BA", "#D75319", "#EDB113"])
    genieplot.set("line_styles", ["-.", ":", "-.", ":", "-.", ":", "-.", ":"])
    genieplot.set("line_widths", [1.3, 6, 34, 36, 5.3368, 3.7, 5.4, 7.5, 5, 15, 14, 6, 66, 56, 5])
    genieplot.set("marker_types", ["x", "<", "*", "x", "<", "+", "x", "<", "*", "x", "^", "*"])
    genieplot.set("marker_sizes", [1.3, 6.34327, 3.35784, 36, 5.3368, 3.1856377, 5.4, 7])


def _genie_line_properties_test(styles_ref, widths_ref):
    styles = genieplot.get("line_styles")
    widths = genieplot.get("line_widths")
    if styles_ref:
        if isinstance(styles, str):
            styles = [styles]
        assert list(styles) == list(styles_ref)
    if widths_ref:
        np.testing.assert_allclose(widths, widths_ref)


def _genie_marker_properties_test(types_ref, sizes_ref):
    types = genieplot.get("marker_types")
    sizes = genieplot.get("marker_sizes")
    if types_ref:
        if isinstance(types, str):
            types = [types]
        assert list(types) == list(types_ref)
    if sizes_ref:
        np.testing.assert_allclose(sizes, sizes_ref)


def _genie_colors_test(colors_ref, cycle_ref):
    colors = genieplot.get("colors")
    cycle = genieplot.get("color_cycle")
    if colors_ref:
        if isinstance(colors, str):
            colors = [colors]
        assert list(colors) == list(colors_ref)
    if cycle_ref:
        assert cycle == cycle_ref


class TestPlotIXDataset3:
    """Test of marker, lines and colours control (functions amark, aline, acolor)."""

    colorNames = [
        "red", "green", "blue", "cyan", "magenta", "yellow", "black", "white",
        "r", "g", "b", "c", "m", "y", "k", "w",
        "denim", "carrot", "marigold", "purple", "grass", "babyblue", "brickred",
    ]
    colorCodes = [
        "r", "g", "b", "c", "m", "y", "k", "w",
        "r", "g", "b", "c", "m", "y", "k", "w",
        "#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F",
    ]
    line_style_names = ["solid", "dashed", "dotted", "ddot"]
    line_style_codes = ["-", "--", ":", "-."]
    marker_type_names = [
        "o", "+", "*", ".", "x", "_", "|", "square", "diamond",
        "^", "v", ">", "<", "pentagram", "hexagram",
    ]
    marker_type_codes = [
        "o", "+", "*", ".", "x", "_", "|", "s", "d",
        "^", "v", ">", "<", "p", "h",
    ]

    def setup_method(self):
        w1_p, *_ = make_data()
        self.data1D = w1_p[0]

    def teardown_method(self):
        genieplot.reset()
        clear_figures()

    def test_aline(self):
        for name, code in zip(self.line_style_names, self.line_style_codes):
            genieplot.reset()
            _genieplot_initialise("fast")
            aline(name)
            _genie_line_properties_test([code], [])

        for code in self.line_style_codes:
            genieplot.reset()
            _genieplot_initialise("fast")
            aline(code)
            _genie_line_properties_test([code], [])

        genieplot.reset()
        _genieplot_initialise("fast")
        aline(2.719)
        _genie_line_properties_test([], 2.719)

        genieplot.reset()
        _genieplot_initialise("fast")
        aline(0.5, "dot")
        _genie_line_properties_test([":"], 0.5)

        genieplot.reset()
        _genieplot_initialise("fast")
        aline("dot", 0.5)
        _genie_line_properties_test([":"], 0.5)

        genieplot.reset()
        _genieplot_initialise("fast")
        aline(":", 1, 2, "--", "-", "-.")
        _genie_line_properties_test([":", "--", "-", "-."], [1, 2])

        genieplot.reset()
        _genieplot_initialise("fast")
        aline(["DOT", "SOL"], np.arange(1, 8, 2))
        _genie_line_properties_test([":", "-"], [1, 3, 5, 7])

        width_out, style_out = aline("dot", "4.5")
        assert np.isclose(width_out, 4.5)
        assert style_out in (":", [":"])

        width_out, style_out = aline("dot", [5, 6, 8], "sol")
        np.testing.assert_array_equal(width_out, [5, 6, 8])
        assert list(style_out) == [":", "-"]

    def test_amark(self):
        for name, code in zip(self.marker_type_names, self.marker_type_codes):
            genieplot.reset()
            _genieplot_initialise("fast")
            amark(name)
            _genie_marker_properties_test([code], [])

        for code in self.marker_type_codes:
            genieplot.reset()
            _genieplot_initialise("fast")
            amark(code)
            _genie_marker_properties_test([code], [])

        genieplot.reset()
        _genieplot_initialise("fast")
        amark(9.5)
        _genie_marker_properties_test([], 9.5)

        genieplot.reset()
        _genieplot_initialise("fast")
        amark(4.5, "diam")
        _genie_marker_properties_test(["d"], 4.5)

        genieplot.reset()
        _genieplot_initialise("fast")
        amark("+", 5, 10, "*", "diam", ">")
        _genie_marker_properties_test(["+", "*", "d", ">"], [5, 10])

        genieplot.reset()
        _genieplot_initialise("fast")
        amark(["V", "P"], np.arange(1, 8, 2))
        _genie_marker_properties_test(["v", "p"], [1, 3, 5, 7])

        size_out, marker_out = amark("pent", 4.5)
        assert np.isclose(size_out, 4.5)
        assert marker_out in ("p", ["p"])

        size_out, marker_out = amark("pent", [5, 6, 8], "diam")
        np.testing.assert_array_equal(size_out, [5, 6, 8])
        assert list(marker_out) == ["p", "d"]

    def test_acolor(self):
        genieplot.reset()
        _genieplot_initialise("fast")
        acolor("-fast")
        _genie_colors_test([], "fast")

        genieplot.reset()
        _genieplot_initialise("fast")
        acolor("-wi")
        _genie_colors_test([], "with")

        with pytest.raises(HoraceError) as excinfo:
            acolor("wi")
        assert "HERBERT:graphics:invalid_argument" in str(excinfo.value)

        for name, code in zip(self.colorNames, self.colorCodes):
            genieplot.reset()
            _genieplot_initialise("fast")
            acolor(name)
            _genie_colors_test([code], [])

        for code in self.colorCodes:
            genieplot.reset()
            _genieplot_initialise("fast")
            acolor(code)
            _genie_colors_test([code], [])

        genieplot.reset()
        _genieplot_initialise("fast")
        acolor("r", "b", "bla", "g", "-wi")
        _genie_colors_test(["r", "b", "k", "g"], "with")

        with pytest.raises(HoraceError) as excinfo:
            acolor("r", "b", "bla", "-wi", "g")
        assert "HERBERT:graphics:invalid_argument" in str(excinfo.value)

        col_out, cycle_out = acolor("g", "-fa")
        assert col_out == "green" or col_out == ["green"]
        assert cycle_out == "fast"

        col_out, cycle_out = acolor("r", "gem", "bla", "-wi")
        assert list(col_out) == [
            "red", "denim", "carrot", "marigold",
            "purple", "grass", "babyblue", "brickred", "black",
        ]
        assert cycle_out == "with"

    def test_cyclePlotProperties_colorsFast_and_With(self):
        for mode in ["fast", "with"]:
            genieplot.reset()
            _genieplot_initialise("fast")
            acolor(f"-{mode}")
            acolor("r", "b", "y")
            aline("-", "--", 0.5, 1, 2)
            amark("o", "*", "x", "s", 3, 6)

            data1D_arr = DataOpArray([self.data1D * k for k in range(1, 13)])
            fig_h, axes_h, plot_h = dd(data1D_arr)
            assert isinstance(fig_h, matplotlib.figure.Figure)
            assert len(plot_h) == 12
            clear_figures()


# ==============================================================================
# TestPlotIXDataset1And2 (translated from test_plot_IX_dataset.m & test_plot_IX_dataset_2.m)
# ==============================================================================
class TestPlotIXDataset1And2:
    """Test plotting of IX_dataset_1d, IX_dataset_2d, and IX_dataset_3d objects."""

    def setup_method(self):
        genieplot.reset()
        clear_figures()
        (
            self.w1_point,
            self.w1_hist,
            self.w2_point,
            self.w2_hist,
            self.w3_point,
            self.w3_hist,
            self.w2oneD_point_xy,
            self.w2oneD_point_x_hist_y,
            self.w3_unequal,
        ) = make_data()
        self.data1D = self.w1_hist[0]
        self.data2D = self.w2_hist[0]
        self.data3D = self.w3_hist[0]
        self.T = DataPlotInterfaceTester()

    def teardown_method(self):
        genieplot.reset()
        clear_figures()

    # ------------------------------------------------------------------
    # Spaghetti_plot tests
    # ------------------------------------------------------------------
    def test_spaghetti_plot_noplot(self):
        wdisp = DataOpArray([self.data2D, self.data2D * 2, self.data2D * 0.5])
        wdisp_out, cuts, fig_h, axes_h, plot_h = spaghetti_plot(wdisp, "noplot")
        assert len(wdisp_out) == 3
        assert not cuts
        assert fig_h is None or fig_h == []
        assert axes_h is None or axes_h == []
        assert plot_h is None or plot_h == []

    def test_spaghetti_plot_labels(self):
        wdisp = DataOpArray([self.data2D, self.data2D * 2, self.data2D * 0.5])
        wdisp_out, cuts, fig_h, axes_h, plot_h = spaghetti_plot(
            wdisp, "lab", ["A", "B", "C", "D"]
        )
        assert len(wdisp_out) == 3
        assert isinstance(fig_h, matplotlib.figure.Figure)
        assert isinstance(axes_h, matplotlib.axes.Axes)
        assert len(plot_h) == 3

    # ------------------------------------------------------------------
    # 3D plot tests
    # ------------------------------------------------------------------
    def test_IX3D_plot3D_methods_work(self):
        methods = self.T.methodsND_plot + self.T.methods3D_plot
        fig_h_ref = None
        for i, meth in enumerate(methods):
            fig_h, axes_h, plot_h = meth(self.data3D)
            assert isinstance(fig_h, matplotlib.figure.Figure)
            assert isinstance(axes_h, matplotlib.axes.Axes)
            assert isinstance(plot_h, dict)
            if i > 0:
                assert fig_h is fig_h_ref
            else:
                fig_h_ref = fig_h

    def test_IX3D_other_plot_methods_throw(self):
        other_methods = (
            self.T.methods1D_plot
            + self.T.methods1D_plotOver
            + self.T.methods1D_plotOverCurr
            + self.T.methods2D_plot
            + self.T.methods2D_plotOver
            + self.T.methods2D_plotOverCurr
        )
        for meth in other_methods:
            with pytest.raises(HoraceError) as excinfo:
                meth(self.data3D)
            assert "IX_dataset_3d:invalid_argument" in str(excinfo.value)

    def test_IX3D_plot3D_methods_do_not_work_with_array(self):
        methods = self.T.methodsND_plot + self.T.methods3D_plot
        for meth in methods:
            with pytest.raises(HoraceError) as excinfo:
                meth(self.w3_point)
            assert "IX_dataset_3d:invalid_argument" in str(excinfo.value)

    def test_3D_nonUniformBins_noFigurePlotted(self):
        with pytest.raises(HoraceError) as excinfo:
            sliceomatic(self.w3_unequal)
        assert "HERBERT:IX_dataset_3d:invalid_argument" in str(excinfo.value)

    # ------------------------------------------------------------------
    # 2D plot tests
    # ------------------------------------------------------------------
    def test_IX2D_plot2D_methods_work(self):
        methods = (
            self.T.methodsND_plot
            + self.T.methods2D_plot
            + self.T.methodsND_plotOver
            + self.T.methods2D_plotOver
            + self.T.methods2D_plotOverCurr
        )
        for meth in methods:
            clear_figures()
            fig_h, axes_h, plot_h = meth(self.data2D)
            assert isinstance(fig_h, matplotlib.figure.Figure)
            assert isinstance(axes_h, matplotlib.axes.Axes)
            assert len(plot_h) >= 1

    def test_IX2D_other_plot_methods_throw(self):
        other_methods = (
            self.T.methods1D_plot
            + self.T.methods1D_plotOver
            + self.T.methods1D_plotOverCurr
            + self.T.methods3D_plot
        )
        for meth in other_methods:
            with pytest.raises(HoraceError) as excinfo:
                meth(self.data2D)
            assert "IX_dataset_2d:invalid_argument" in str(excinfo.value)

    def test_IX2D_plot2D_methods_reuseFig_newAxes(self):
        methods = self.T.methodsND_plot + self.T.methods2D_plot
        for meth in methods:
            clear_figures()
            fig_h_ref, axes_h_ref, _ = meth(self.w2_point[0])
            fig_h, axes_h, _ = meth(self.w2_point[1])
            assert fig_h is fig_h_ref
            assert axes_h is not axes_h_ref

    def test_IX2D_plotOver2D_methods_reuseFigAndAxes(self):
        methods = self.T.methodsND_plot + self.T.methods2D_plot
        methods_over = self.T.methodsND_plotOver + self.T.methods2D_plotOver
        for meth, meth_over in zip(methods, methods_over):
            clear_figures()
            fig_h_ref, axes_h_ref, plot_h_ref = meth(self.w2_point[0])
            fig_h, axes_h, plot_h = meth_over(self.w2_point[1])
            assert fig_h is fig_h_ref
            assert axes_h is axes_h_ref
            assert len(plot_h) == len(plot_h_ref) + 1

    def test_IX2D_plot2D_methods_work_with_array(self):
        methods = self.T.methodsND_plot + self.T.methods2D_plot
        for meth in methods:
            clear_figures()
            fig_h, axes_h, plot_h = meth(self.w2_point)
            assert isinstance(fig_h, matplotlib.figure.Figure)
            assert isinstance(axes_h, matplotlib.axes.Axes)
            assert len(plot_h) == 2

    def test_2D_areaAndsurfacePlotsDifferentFigures(self):
        fig_h_ref, _, _ = da(self.w2_point[0])
        fig_h, _, _ = ps(self.w2_point[1])
        assert fig_h is not fig_h_ref

    def test_2D_plotOverCurrent_on2D(self):
        fig_h_ref, axes_h_ref, plot_h_ref = ds(self.w2_point[0])
        fig_h, axes_h, plot_h = paoc(self.w2_point[1])
        assert fig_h is fig_h_ref
        assert axes_h is axes_h_ref
        assert len(plot_h) == len(plot_h_ref) + 1

    def test_IX2D_limits_area_and_surface(self):
        _, axes_h, _ = da(self.data2D)
        xlim_ref = axes_h.get_xlim()
        ylim_ref = axes_h.get_ylim()

        xlim_new = (xlim_ref[0] - 2.0, xlim_ref[1] + 2.0)
        ylim_new = (ylim_ref[0] - 2.0, ylim_ref[1] + 2.0)
        lx(xlim_new)
        ly(ylim_new)
        np.testing.assert_allclose(lx(), xlim_new)
        np.testing.assert_allclose(ly(), ylim_new)

        lx()
        ly()
        np.testing.assert_allclose(lx(), xlim_ref)
        np.testing.assert_allclose(ly(), ylim_ref)

        clear_figures()
        _, axes_h, _ = ds2(self.data2D, np.pi * self.data2D)
        lc("padded")

    # ------------------------------------------------------------------
    # 1D plot tests
    # ------------------------------------------------------------------
    def test_IX1D_all_plot1D_methods_work(self):
        methods = (
            self.T.methodsND_plot
            + self.T.methods1D_plot
            + self.T.methodsND_plotOver
            + self.T.methods1D_plotOver
            + self.T.methods1D_plotOverCurr
        )
        for meth in methods:
            clear_figures()
            fig_h, axes_h, plot_h = meth(self.data1D)
            assert isinstance(fig_h, matplotlib.figure.Figure)
            assert isinstance(axes_h, matplotlib.axes.Axes)
            assert len(plot_h) >= 1

    def test_IX1D_other_plot_methods_throw(self):
        other_methods = (
            self.T.methods2D_plot
            + self.T.methods2D_plotOver
            + self.T.methods2D_plotOverCurr
            + self.T.methods3D_plot
        )
        for meth in other_methods:
            with pytest.raises(HoraceError) as excinfo:
                meth(self.data1D)
            assert "IX_dataset_1d:invalid_argument" in str(excinfo.value)

    def test_IX1D_plotOver1D_methods_reuseFigAndAxes(self):
        methods = self.T.methodsND_plotOver + self.T.methods1D_plotOver
        fig_h_prev, axes_h_prev, plot_h_prev = plot(self.data1D)
        for i, meth in enumerate(methods):
            fig_h, axes_h, plot_h = meth((2 + i) * self.data1D)
            assert fig_h is fig_h_prev
            assert axes_h is axes_h_prev
            assert len(plot_h) == len(plot_h_prev) + 1
            plot_h_prev = plot_h

    def test_src_one_and_two_obj(self):
        plot(self.data1D)
        res1 = src(1)
        assert isinstance(res1, IX_dataset_1d)
        plotover(2 * self.data1D)
        res2 = src(1)
        assert len(res2) == 2

    def test_1D_axisScale_and_keep_figure(self):
        linx()
        fig1, ax1, _ = dl(self.w1_point[0])
        keep_figure()
        _, ax2, _ = dh(2 * self.w1_point[1])
        logx()

        make_current(fig1)
        _, is_current, _ = is_genie_figure(fig1)
        assert is_current

        ftmp, _, _ = ph(self.w1_point[1])
        assert ftmp is fig1
        assert ax1.get_xscale() == "linear"
        assert ax2.get_xscale() == "log"

    def test_1D_limits_and_auto_rescaling(self):
        dl(self.w1_point[0])
        pl(self.w1_point[1])
        np.testing.assert_allclose(lx(), [1, 14])
        np.testing.assert_allclose(ly(), [40, 240])

        # Reduce x-axis range; y range should be reduced
        lx(3, 8)
        np.testing.assert_allclose(lx(), [3, 8])
        np.testing.assert_allclose(ly(), [60, 180])

        # Increase x range; y range should be increased
        lx(2, 12)
        np.testing.assert_allclose(lx(), [2, 12])
        np.testing.assert_allclose(ly(), [50, 220])

        # Default x range
        lx()
        np.testing.assert_allclose(lx(), [1, 14])
        np.testing.assert_allclose(ly(), [40, 240])

        # Reduce y-axis range; x range should be reduced
        ly(70, 210)
        np.testing.assert_allclose(lx(), [4, 11])
        np.testing.assert_allclose(ly(), [70, 210])

        # Increase y range
        ly(60, 220)
        np.testing.assert_allclose(lx(), [3, 12])
        np.testing.assert_allclose(ly(), [60, 220])

        # Default y range
        ly()
        np.testing.assert_allclose(lx(), [1, 14])
        np.testing.assert_allclose(ly(), [40, 240])

    def test_limits_tight_tickaligned_padded_rounded(self):
        _, axes_h, _ = dh(self.w1_hist)

        lx("tight")
        np.testing.assert_allclose(axes_h.get_xlim(), [0.5, 14.5], atol=1e-12)

        lx("tickaligned")
        np.testing.assert_allclose(axes_h.get_xlim(), [0.0, 15.0], atol=1e-12)

        lx("padded")
        np.testing.assert_allclose(axes_h.get_xlim(), [-0.48, 15.48], atol=1e-12)

        lx("rounded")
        np.testing.assert_allclose(axes_h.get_xlim(), [-0.48, 15.48], atol=1e-12)

    def test_plotInterface_throws(self):
        all_methods = (
            self.T.methodsND_plot
            + self.T.methodsND_plotOver
            + self.T.methods1D_plot
            + self.T.methods1D_plotOver
            + self.T.methods1D_plotOverCurr
            + self.T.methods2D_plot
            + self.T.methods2D_plotOver
            + self.T.methods2D_plotOverCurr
            + self.T.methods3D_plot
        )
        for meth in all_methods:
            with pytest.raises(HoraceError) as excinfo:
                meth(self.T)
            assert "invalid_argument" in str(excinfo.value)
