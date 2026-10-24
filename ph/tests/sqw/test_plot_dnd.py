"""Tests for dnd_plot_interface methods on DnD objects, ported from test_plot_dnd.m."""

import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import d0d, d1d, d2d, d3d, d4d

METHODS_1D = ["dd", "de", "dh", "dl", "dm", "dp", "pd", "pdoc", "pe", "peoc", "ph", "phoc", "pl", "ploc", "pm", "pmoc", "pp", "ppoc"]
METHODS_2D = ["da", "ds", "ds2", "pa", "paoc", "ps", "psoc", "ps2", "ps2oc"]
METHODS_3D = ["sliceomatic", "sliceomatic_overview"]
METHODS_ANY = ["plot", "plotover"]


class TestPlotDnD:
    @pytest.mark.parametrize("method_name", METHODS_1D)
    def test_1d_plot_methods_succeed_on_d1d_and_fail_on_others(self, method_name):
        w1 = d1d()
        fig, ax, pl = getattr(w1, method_name)(nargout=3)
        assert fig is not None

        for cls in (d0d, d2d, d3d, d4d):
            w_other = cls()
            with pytest.raises(HoraceError):
                getattr(w_other, method_name)()

    @pytest.mark.parametrize("method_name", METHODS_2D)
    def test_2d_plot_methods_succeed_on_d2d_and_fail_on_others(self, method_name):
        w2 = d2d()
        fig, ax, pl = getattr(w2, method_name)(nargout=3)
        assert fig is not None

        for cls in (d0d, d1d, d3d, d4d):
            w_other = cls()
            with pytest.raises(HoraceError):
                getattr(w_other, method_name)()

    @pytest.mark.parametrize("method_name", METHODS_3D)
    def test_3d_plot_methods_succeed_on_d3d_and_fail_on_others(self, method_name):
        w3 = d3d()
        fig, ax, pl = getattr(w3, method_name)(nargout=3)
        assert fig is not None

        for cls in (d0d, d1d, d2d, d4d):
            w_other = cls()
            with pytest.raises(HoraceError):
                getattr(w_other, method_name)()

    @pytest.mark.parametrize("method_name", METHODS_ANY)
    def test_generic_plot_methods_succeed_on_1d_2d_3d_and_fail_on_0d_4d(self, method_name):
        for cls in (d1d, d2d, d3d):
            w = cls()
            fig, ax, pl = getattr(w, method_name)(nargout=3)
            assert fig is not None

        for cls in (d0d, d4d):
            w_bad = cls()
            with pytest.raises(HoraceError):
                getattr(w_bad, method_name)()

