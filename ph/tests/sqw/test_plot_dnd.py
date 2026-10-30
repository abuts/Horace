"""Pytest port of `_test/test_dnd_class/test_plot_dnd.m` and `data_plot_interface_tester.m`."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.data_plot_interface import _dispatch_plot_method, data_plot_interface
from pyhorace.data.graphics import Axes, Figure, PlotHandle, clearfigs, genieplot, spaghetti_plot, src
from pyhorace.data.ix_dataset_2d import IX_dataset_2d
from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d, dnd_plot_interface, read_dnd

COMMON_DATA_DIR = Path(__file__).resolve().parents[3] / "_test" / "common_data"
SQW_1D_FILE = COMMON_DATA_DIR / "sqw_1d_1.sqw"
SQW_2D_FILE = COMMON_DATA_DIR / "sqw_2d_1.sqw"
SQW_3D_FILE = COMMON_DATA_DIR / "w3d_sqw.sqw"
SQW_4D_FILE = COMMON_DATA_DIR / "sqw_4d.sqw"

METHODS_ND_PLOT = ["plot"]
METHODS_ND_PLOT_OVER = ["plotover"]
METHODS_1D_PLOT = ["dd", "de", "dh", "dl", "dm", "dp"]
METHODS_1D_PLOT_OVER = ["pd", "pe", "ph", "pl", "pm", "pp"]
METHODS_1D_PLOT_OVER_CURR = ["pdoc", "peoc", "phoc", "ploc", "pmoc", "ppoc"]
METHODS_2D_PLOT = ["da", "ds", "ds2"]
METHODS_2D_PLOT_OVER = ["pa", "ps", "ps2"]
METHODS_2D_PLOT_OVER_CURR = ["paoc", "psoc", "ps2oc"]
METHODS_3D_PLOT = ["sliceomatic", "sliceomatic_overview"]

TWO_INPUT_METHODS = {"ds2", "ps2", "ps2oc"}


def _call_plot_method(f_name: str, data_obj: Any) -> Any:
    if f_name in TWO_INPUT_METHODS:
        return _dispatch_plot_method(f_name, data_obj, data_obj)
    return _dispatch_plot_method(f_name, data_obj)


def _assert_valid_1d_2d_handles(fig_h: Any, ax_h: Any, plot_h: Any) -> None:
    assert isinstance(fig_h, Figure)
    assert isinstance(ax_h, Axes)
    assert isinstance(plot_h, PlotHandle) or (
        isinstance(plot_h, list) and all(isinstance(h, PlotHandle) for h in plot_h)
    )


@pytest.fixture(scope="module")
def sample_dnd_objects() -> dict[str, Any]:
    """Load 1D, 2D, 3D, and 4D DnD objects from `_test/common_data` matching `test_plot_dnd.m`."""
    return {
        "data1D": read_dnd(str(SQW_1D_FILE)),
        "data2D": read_dnd(str(SQW_2D_FILE)),
        "data3D": read_dnd(str(SQW_3D_FILE)),
        "data4D": read_dnd(str(SQW_4D_FILE)),
    }


@pytest.fixture(autouse=True)
def _cleanup_figures():
    genieplot.reset()
    clearfigs()
    yield
    clearfigs()


class TestPlotDnD:
    """Port of `test_plot_dnd.m`."""

    def test_inheritance_hierarchy(self):
        assert issubclass(dnd_plot_interface, data_plot_interface)
        for cls in (d0d, d1d, d2d, d3d, d4d):
            assert issubclass(cls, dnd_plot_interface)

    def test_spaghetti_default_labels(self, sample_dnd_objects: dict[str, Any]):
        data2d: d2d = sample_dnd_objects["data2D"]
        wdisp = DataOpArray([data2d, data2d * 2, data2d * 0.5])
        wdisp_out, cuts, fig_h, axes_h, plot_h = spaghetti_plot(wdisp)

        assert isinstance(wdisp_out, DataOpArray)
        assert all(isinstance(x, IX_dataset_2d) for x in wdisp_out.flat)
        assert len(cuts) == len(wdisp)
        for c, w in zip(cuts.flat, wdisp.flat):
            assert c.equal_to_tol(w)
        assert isinstance(fig_h, Figure)
        assert isinstance(axes_h, Axes)
        assert len(plot_h) == 3
        assert all(isinstance(h, PlotHandle) and h.Type == "patch" for h in plot_h)
        assert axes_h.XTickLabel == [
            "[-0.71, -0.71, 0]",
            "[-0.71, -0.71, 0]",
            "[-0.71, -0.71, 0]",
            "[-0.39, -0.39, 0]",
        ]

    def test_d4d_all_plot_methods_throw(self, sample_dnd_objects: dict[str, Any]):
        data4d: d4d = sample_dnd_objects["data4D"]
        other_methods = [
            *METHODS_ND_PLOT,
            *METHODS_1D_PLOT,
            *METHODS_1D_PLOT_OVER,
            *METHODS_1D_PLOT_OVER_CURR,
            *METHODS_2D_PLOT,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
            *METHODS_3D_PLOT,
        ]
        for f_name in other_methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data4d)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_d3d_plot3d_methods_work(self, sample_dnd_objects: dict[str, Any]):
        data3d: d3d = sample_dnd_objects["data3D"]
        methods = [*METHODS_ND_PLOT, *METHODS_3D_PLOT]
        fig_h_ref = None
        for idx, f_name in enumerate(methods):
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data3d)
            assert isinstance(fig_h, Figure)
            assert isinstance(axes_h, Axes)
            assert isinstance(plot_h, dict)
            if idx > 0:
                assert fig_h is fig_h_ref
            else:
                fig_h_ref = fig_h

    def test_d3d_other_plot_methods_throw(self, sample_dnd_objects: dict[str, Any]):
        data3d: d3d = sample_dnd_objects["data3D"]
        other_methods = [
            *METHODS_1D_PLOT,
            *METHODS_1D_PLOT_OVER,
            *METHODS_1D_PLOT_OVER_CURR,
            *METHODS_2D_PLOT,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
        ]
        for f_name in other_methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data3d)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_d3d_plot3d_methods_do_not_work_with_array(self, sample_dnd_objects: dict[str, Any]):
        data3d: d3d = sample_dnd_objects["data3D"]
        methods = [*METHODS_ND_PLOT, *METHODS_3D_PLOT]
        data3d_arr = DataOpArray([data3d, 2 * data3d])
        for f_name in methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data3d_arr)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_d2d_plot2d_methods_work(self, sample_dnd_objects: dict[str, Any]):
        data2d: d2d = sample_dnd_objects["data2D"]
        methods = [
            *METHODS_ND_PLOT,
            *METHODS_2D_PLOT,
            *METHODS_ND_PLOT_OVER,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
        ]
        clearfigs()
        for f_name in methods:
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data2d)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            if len(plot_h) == 1:
                assert fig_h.UserData.equal_to_tol(data2d)
            else:
                assert fig_h.UserData[-1].equal_to_tol(data2d)
                assert len(fig_h.UserData) == len(plot_h)

    def test_d2d_other_plot_methods_throw(self, sample_dnd_objects: dict[str, Any]):
        data2d: d2d = sample_dnd_objects["data2D"]
        other_methods = [
            *METHODS_1D_PLOT,
            *METHODS_1D_PLOT_OVER,
            *METHODS_1D_PLOT_OVER_CURR,
            *METHODS_3D_PLOT,
        ]
        for f_name in other_methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data2d)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_d2d_plot2d_methods_work_with_array(self, sample_dnd_objects: dict[str, Any]):
        data2d: d2d = sample_dnd_objects["data2D"]
        methods = [*METHODS_ND_PLOT, *METHODS_2D_PLOT]
        data2d_arr = DataOpArray([data2d, 2 * data2d])
        for f_name in methods:
            clearfigs()
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data2d_arr)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            assert len(plot_h) == 2
            assert isinstance(fig_h.UserData, list) and len(fig_h.UserData) == 2
            for item, exp in zip(fig_h.UserData, data2d_arr.flat):
                assert item.equal_to_tol(exp)

    def test_d1d_all_plot1d_methods_work(self, sample_dnd_objects: dict[str, Any]):
        data1d: d1d = sample_dnd_objects["data1D"]
        methods = [
            *METHODS_ND_PLOT,
            *METHODS_1D_PLOT,
            *METHODS_ND_PLOT_OVER,
            *METHODS_1D_PLOT_OVER,
            *METHODS_1D_PLOT_OVER_CURR,
        ]
        for f_name in methods:
            clearfigs()
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data1d)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            if len(plot_h) > 1:
                assert fig_h.UserData[-1].equal_to_tol(data1d)
            else:
                assert fig_h.UserData.equal_to_tol(data1d)

    def test_d1d_other_plot_methods_throw(self, sample_dnd_objects: dict[str, Any]):
        data1d: d1d = sample_dnd_objects["data1D"]
        other_methods = [
            *METHODS_2D_PLOT,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
            *METHODS_3D_PLOT,
        ]
        for f_name in other_methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data1d)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_d1d_plot1d_methods_work_with_array(self, sample_dnd_objects: dict[str, Any]):
        data1d: d1d = sample_dnd_objects["data1D"]
        methods = [*METHODS_ND_PLOT, *METHODS_1D_PLOT]
        data1d_arr = DataOpArray([data1d, 2 * data1d])
        for f_name in methods:
            clearfigs()
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data1d_arr)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            assert len(plot_h) == 2
            assert isinstance(fig_h.UserData, list) and len(fig_h.UserData) == 2
            for item, exp in zip(fig_h.UserData, data1d_arr.flat):
                assert item.equal_to_tol(exp)

    def test_src_two_obj(self, sample_dnd_objects: dict[str, Any]):
        data1d: d1d = sample_dnd_objects["data1D"]
        clearfigs()
        data1d.plot()
        (2 * data1d).plotover()
        res = src(1)
        assert isinstance(res, DataOpArray) and len(res) == 2
        assert res[0].equal_to_tol(data1d)
        assert res[1].equal_to_tol(2 * data1d)

    def test_src_one_obj(self, sample_dnd_objects: dict[str, Any]):
        data1d: d1d = sample_dnd_objects["data1D"]
        clearfigs()
        data1d.plot()
        res = src(1)
        assert isinstance(res, d1d)
        assert res.equal_to_tol(data1d)

    def test_adjust_aspect(self, sample_dnd_objects: dict[str, Any]):
        w2d = copy.deepcopy(sample_dnd_objects["data2D"])
        assert bool(w2d.axes.changes_aspect_ratio) is True

        fig_h, axes_h, plot_h = _dispatch_plot_method("da", w2d)
        _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
        assert axes_h.DataAspectRatioMode == "manual"

        w2d.axes.changes_aspect_ratio = False
        assert bool(w2d.axes.changes_aspect_ratio) is False
        fig_h2, axes_h2, plot_h2 = _dispatch_plot_method("da", w2d)
        _assert_valid_1d_2d_handles(fig_h2, axes_h2, plot_h2)
        assert axes_h2.DataAspectRatioMode == "auto"
