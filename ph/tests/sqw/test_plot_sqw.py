"""Pytest port of `_test/test_sqw_class/test_plot_sqw.m` and `data_plot_interface_tester.m`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.data_plot_interface import _dispatch_plot_method, data_plot_interface
from pyhorace.data.graphics import Axes, Figure, PlotHandle, clearfigs, genieplot, src
from pyhorace.sqw import read_sqw, sqw, sqw_plot_interface

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
def sample_sqw_objects() -> dict[str, sqw]:
    """Load 1D, 2D, 3D, and 4D sqw objects from `_test/common_data` matching `test_plot_sqw.m`."""
    return {
        "data1D": read_sqw(str(SQW_1D_FILE)),
        "data2D": read_sqw(str(SQW_2D_FILE)),
        "data3D": read_sqw(str(SQW_3D_FILE)),
        "data4D": read_sqw(str(SQW_4D_FILE)),
    }


@pytest.fixture(autouse=True)
def _cleanup_figures():
    genieplot.reset()
    clearfigs()
    yield
    clearfigs()


class TestPlotSqw:
    """Port of `test_plot_sqw.m`."""

    def test_inheritance_hierarchy(self):
        assert issubclass(sqw_plot_interface, data_plot_interface)
        assert issubclass(sqw, sqw_plot_interface)

    def test_sqw4d_all_plot_methods_throw(self, sample_sqw_objects: dict[str, sqw]):
        data4d = sample_sqw_objects["data4D"]
        all_methods = [
            *METHODS_ND_PLOT,
            *METHODS_ND_PLOT_OVER,
            *METHODS_1D_PLOT,
            *METHODS_1D_PLOT_OVER,
            *METHODS_1D_PLOT_OVER_CURR,
            *METHODS_2D_PLOT,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
            *METHODS_3D_PLOT,
        ]
        for f_name in all_methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data4d)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_sqw3d_plot3d_methods_work(self, sample_sqw_objects: dict[str, sqw]):
        data3d = sample_sqw_objects["data3D"]
        methods = [*METHODS_ND_PLOT, *METHODS_3D_PLOT]
        fig_h_ref = None
        for idx, f_name in enumerate(methods):
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data3d)
            assert isinstance(fig_h, Figure)
            assert isinstance(axes_h, Axes)
            assert isinstance(plot_h, dict)
            assert isinstance(fig_h.UserData, sqw)
            assert fig_h.UserData.equal_to_tol(data3d)
            if idx > 0:
                assert fig_h is fig_h_ref
            else:
                fig_h_ref = fig_h

    def test_sqw3d_other_plot_methods_throw(self, sample_sqw_objects: dict[str, sqw]):
        data3d = sample_sqw_objects["data3D"]
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

    def test_sqw3d_plot3d_methods_do_not_work_with_array(self, sample_sqw_objects: dict[str, sqw]):
        data3d = sample_sqw_objects["data3D"]
        methods = [*METHODS_ND_PLOT, *METHODS_3D_PLOT]
        data3d_arr = DataOpArray([data3d, 2 * data3d])
        for f_name in methods:
            with pytest.raises(HoraceError) as exc_info:
                _call_plot_method(f_name, data3d_arr)
            assert exc_info.value.error_id == "HORACE:graphics:invalid_argument"

    def test_sqw2d_plot2d_methods_work(self, sample_sqw_objects: dict[str, sqw]):
        data2d = sample_sqw_objects["data2D"]
        methods = [*METHODS_ND_PLOT, *METHODS_2D_PLOT]
        for f_name in methods:
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data2d)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            assert isinstance(fig_h.UserData, sqw)
            assert fig_h.UserData.equal_to_tol(data2d)

    def test_sqw2d_overplot2d_methods_work(self, sample_sqw_objects: dict[str, sqw]):
        data2d = sample_sqw_objects["data2D"]
        methods = [
            *METHODS_ND_PLOT_OVER,
            *METHODS_2D_PLOT_OVER,
            *METHODS_2D_PLOT_OVER_CURR,
        ]
        clearfigs()
        _call_plot_method(METHODS_2D_PLOT[0], data2d)
        for f_name in methods:
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data2d)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            if len(plot_h) == 1:
                assert fig_h.UserData.equal_to_tol(data2d)
            else:
                assert fig_h.UserData[-1].equal_to_tol(data2d)
                assert len(fig_h.UserData) == len(plot_h)

    def test_sqw2d_other_plot_methods_throw(self, sample_sqw_objects: dict[str, sqw]):
        data2d = sample_sqw_objects["data2D"]
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

    def test_sqw2d_plot2d_methods_work_with_array(self, sample_sqw_objects: dict[str, sqw]):
        data2d = sample_sqw_objects["data2D"]
        methods = [*METHODS_ND_PLOT, *METHODS_2D_PLOT]
        data2d_arr = DataOpArray([data2d, 2 * data2d])
        for f_name in methods:
            clearfigs()
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data2d_arr)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            assert len(plot_h) == 2
            assert isinstance(fig_h.UserData, list) and len(fig_h.UserData) == 2
            for item, exp in zip(fig_h.UserData, data2d_arr.flat):
                assert isinstance(item, sqw) and item.equal_to_tol(exp)

    def test_sqw1d_all_plot1d_methods_work(self, sample_sqw_objects: dict[str, sqw]):
        data1d = sample_sqw_objects["data1D"]
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
                assert isinstance(fig_h.UserData[-1], sqw)
                assert fig_h.UserData[-1].equal_to_tol(data1d)
            else:
                assert isinstance(fig_h.UserData, sqw)
                assert fig_h.UserData.equal_to_tol(data1d)

    def test_sqw1d_other_plot_methods_throw(self, sample_sqw_objects: dict[str, sqw]):
        data1d = sample_sqw_objects["data1D"]
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

    def test_sqw1d_plot1d_methods_work_with_array(self, sample_sqw_objects: dict[str, sqw]):
        data1d = sample_sqw_objects["data1D"]
        methods = [*METHODS_ND_PLOT, *METHODS_1D_PLOT]
        data1d_arr = DataOpArray([data1d, 2 * data1d])
        for f_name in methods:
            clearfigs()
            fig_h, axes_h, plot_h = _call_plot_method(f_name, data1d_arr)
            _assert_valid_1d_2d_handles(fig_h, axes_h, plot_h)
            assert len(plot_h) == 2
            assert isinstance(fig_h.UserData, list) and len(fig_h.UserData) == 2
            for item, exp in zip(fig_h.UserData, data1d_arr.flat):
                assert isinstance(item, sqw) and item.equal_to_tol(exp)

    def test_src_two_obj(self, sample_sqw_objects: dict[str, sqw]):
        data1d = sample_sqw_objects["data1D"]
        clearfigs()
        data1d.plot()
        (2 * data1d).plotover()
        res = src(1)
        assert isinstance(res, DataOpArray) and len(res) == 2
        assert isinstance(res[0], sqw) and res[0].equal_to_tol(data1d)
        assert isinstance(res[1], sqw) and res[1].equal_to_tol(2 * data1d)

    def test_src_one_obj(self, sample_sqw_objects: dict[str, sqw]):
        data1d = sample_sqw_objects["data1D"]
        clearfigs()
        data1d.plot()
        res = src(1)
        assert isinstance(res, sqw)
        assert res.equal_to_tol(data1d)
