"""Unit tests for `graph_config` (translated from `genieplot` / `test_plot_singleton.m`).

Translated from `_test/test_IX_classes/test_plot_singleton.m`.
Tests the setting, getting, resetting, and `config_store` JSON persistence of
the `graph_config` properties, as well as interoperability between instance
attribute syntax, `config_base` `get`/`set` functions, and `graph_config`
methods.
"""

from __future__ import annotations

import copy
import math
import os

import pytest

from pyhorace.core.configurations import (
    ConfigData,
    HoraceConfigError,
    config_store,
    get,
    graph_config,
    set,
)


class TestGraphConfig:
    """Translated from `_test/test_IX_classes/test_plot_singleton.m`."""

    def setup_method(self) -> None:
        self.S_default = ConfigData(
            {
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
        )
        self.current_graphics_settings = graph_config.get()

    def teardown_method(self) -> None:
        graph_config.set(self.current_graphics_settings)

    # ------------------------------------------------------------------
    # graph_config tests - for the syntax type:
    #
    #   graph_config.set(<property_name>, <value>)
    #   val = graph_config.get('marker_sizes')
    #   S = graph_config.get()
    #   graph_config.reset()
    # ------------------------------------------------------------------

    def test_get_all(self) -> None:
        """Test `graph_config.get()` returns a structure with the fields of `graph_config`."""
        S = graph_config.get()
        assert isinstance(S, dict), "Output is not a scalar structure/dict"
        assert list(S.keys()) == list(self.S_default.keys()), "Property names not correct"

    def test_set_all(self) -> None:
        """Test `graph_config.set(S)` (`S` a structure/dict) works."""
        S1 = copy.deepcopy(self.S_default)
        S2 = copy.deepcopy(self.S_default)
        S1["colors"] = ["k", "#121314", "y"]
        S2["colors"] = ["g", "b", "y"]

        graph_config.set(S1)
        S1_retrieved = graph_config.get()
        graph_config.set(S2)
        S2_retrieved = graph_config.get()

        assert S1 == S1_retrieved, "S1 and S1_retrieved do not match"
        assert S2 == S2_retrieved, "S2 and S2_retrieved do not match"

    def test_reset(self) -> None:
        """Test `graph_config.reset()`."""
        graph_config.reset()
        S = graph_config.get()
        assert S == self.S_default, "Expected defaults not recovered"

        # Set to something else
        S1 = copy.deepcopy(S)
        S1["colors"] = ["g", "b", "y"]
        graph_config.set(S1)
        assert S1 != self.S_default, "set to new properties failed"

        # Test reset does its job
        graph_config.reset()
        S2 = graph_config.get()
        assert S2 == self.S_default, "Expected defaults not recovered"

    # ------------------------------------------------------------------
    # Non-existent property errors
    # ------------------------------------------------------------------

    def test_set_nonExistentProperty_ERROR(self) -> None:
        """Test attempting to set unrecognised property."""
        with pytest.raises(HoraceConfigError, match="local_only"):
            graph_config.set("local_only", True)

    def test_get_nonExistentProperty_ERROR(self) -> None:
        """Test attempting to get unrecognised property."""
        with pytest.raises(HoraceConfigError, match="local_only"):
            graph_config.get("local_only")

    # ------------------------------------------------------------------
    # XScale
    # ------------------------------------------------------------------

    def test_setAndGet_XScale(self) -> None:
        """Test setting and getting XScale."""
        graph_config.reset()

        graph_config.set("XScale", "linear")
        val = graph_config.get("XScale")
        assert val == "linear"

        graph_config.set("XScale", "log")
        val = graph_config.get("XScale")
        assert val == "log"

    def test_setAndGet_XScale_ERROR(self) -> None:
        """Test attempting to set invalid XScale."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("XScale", "Peter")

    # ------------------------------------------------------------------
    # YScale
    # ------------------------------------------------------------------

    def test_setAndGet_YScale(self) -> None:
        """Test setting and getting YScale."""
        graph_config.reset()

        graph_config.set("YScale", "linear")
        val = graph_config.get("YScale")
        assert val == "linear"

        graph_config.set("YScale", "log")
        val = graph_config.get("YScale")
        assert val == "log"

    def test_setAndGet_YScale_ERROR(self) -> None:
        """Test attempting to set invalid YScale."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("YScale", False)

    # ------------------------------------------------------------------
    # ZScale
    # ------------------------------------------------------------------

    def test_setAndGet_ZScale(self) -> None:
        """Test setting and getting ZScale."""
        graph_config.reset()

        graph_config.set("ZScale", "linear")
        val = graph_config.get("ZScale")
        assert val == "linear"

        graph_config.set("ZScale", "log")
        val = graph_config.get("ZScale")
        assert val == "log"

    def test_setAndGet_ZScale_ERROR(self) -> None:
        """Test attempting to set invalid ZScale."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("ZScale", [34, 35])

    # ------------------------------------------------------------------
    # colors
    # ------------------------------------------------------------------

    def test_setAndGet_colors(self) -> None:
        """Test setting and getting colors."""
        graph_config.reset()

        graph_config.set("colors", ["b"])
        val = graph_config.get("colors")
        assert val == ["b"]

        graph_config.set("colors", ["k", "#121314", "y"])
        val = graph_config.get("colors")
        assert val == ["k", "#121314", "y"]

    def test_setAndGet_colors_ERROR(self) -> None:
        """Test attempting to set invalid colors."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("colors", 666)

    # ------------------------------------------------------------------
    # color_cycle
    # ------------------------------------------------------------------

    def test_setAndGet_color_cycle(self) -> None:
        """Test setting and getting color_cycle."""
        graph_config.reset()

        graph_config.set("color_cycle", "with")
        val = graph_config.get("color_cycle")
        assert val == "with"

        graph_config.set("color_cycle", "fast")
        val = graph_config.get("color_cycle")
        assert val == "fast"

    def test_setAndGet_color_cycle_ERROR(self) -> None:
        """Test attempting to set invalid color_cycle."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("color_cycle", 666)

    # ------------------------------------------------------------------
    # default_fig_name
    # ------------------------------------------------------------------

    def test_setAndGet_default_fig_name(self) -> None:
        """Test setting and getting default_fig_name."""
        graph_config.reset()

        graph_config.set("default_fig_name", "A silly name")
        val = graph_config.get("default_fig_name")
        assert val == "A silly name"

        graph_config.set("default_fig_name", "Lets boogie")
        val = graph_config.get("default_fig_name")
        assert val == "Lets boogie"

        graph_config.set("default_fig_name", [])
        assert graph_config.get("default_fig_name") is None

    def test_setAndGet_default_fig_name_ERROR(self) -> None:
        """Test attempting to set invalid default_fig_name."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("default_fig_name", 37)

    # ------------------------------------------------------------------
    # line_styles
    # ------------------------------------------------------------------

    def test_setAndGet_line_styles(self) -> None:
        """Test setting and getting line_styles."""
        graph_config.reset()

        graph_config.set("line_styles", "--")
        val = graph_config.get("line_styles")
        assert val == "--"

        graph_config.set("line_styles", ["-", "-."])
        val = graph_config.get("line_styles")
        assert val == ["-", "-."]

    def test_setAndGet_line_styles_ERROR(self) -> None:
        """Test attempting to set invalid line_styles."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("line_styles", "ddot")

    # ------------------------------------------------------------------
    # line_widths
    # ------------------------------------------------------------------

    def test_setAndGet_line_widths(self) -> None:
        """Test setting and getting line_widths."""
        graph_config.reset()

        graph_config.set("line_widths", 1.5)
        val = graph_config.get("line_widths")
        assert val == 1.5

        graph_config.set("line_widths", [0.4, 4.6])
        val = graph_config.get("line_widths")
        assert val == [0.4, 4.6]

    def test_setAndGet_line_widths_ERROR(self) -> None:
        """Test attempting to set invalid line_widths."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("line_widths", "thick")

    # ------------------------------------------------------------------
    # marker_sizes
    # ------------------------------------------------------------------

    def test_setAndGet_marker_sizes(self) -> None:
        """Test setting and getting marker_sizes."""
        graph_config.reset()

        graph_config.set("marker_sizes", 1.5)
        val = graph_config.get("marker_sizes")
        assert val == 1.5

        graph_config.set("marker_sizes", [0.4, 4.6])
        val = graph_config.get("marker_sizes")
        assert val == [0.4, 4.6]

    def test_setAndGet_marker_sizes_ERROR(self) -> None:
        """Test attempting to set invalid marker_sizes."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("marker_sizes", "big")

    # ------------------------------------------------------------------
    # marker_types
    # ------------------------------------------------------------------

    def test_setAndGet_marker_types(self) -> None:
        """Test setting and getting marker_types."""
        graph_config.reset()

        graph_config.set("marker_types", "^")
        val = graph_config.get("marker_types")
        assert val == "^"

        graph_config.set("marker_types", ["+", "p"])
        val = graph_config.get("marker_types")
        assert val == ["+", "p"]

    def test_setAndGet_marker_types_ERROR(self) -> None:
        """Test attempting to set invalid marker_types."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("marker_types", "diamond")

    # ------------------------------------------------------------------
    # maxspec_1D
    # ------------------------------------------------------------------

    def test_setAndGet_maxspec_1D(self) -> None:
        """Test setting and getting maxspec_1D."""
        graph_config.reset()

        graph_config.set("maxspec_1D", 468)
        val = graph_config.get("maxspec_1D")
        assert val == 468

        graph_config.set("maxspec_1D", math.inf)
        assert math.isinf(graph_config.get("maxspec_1D"))

    def test_setAndGet_maxspec_1D_ERROR_zero(self) -> None:
        """Test attempting to set invalid maxspec_1D (0)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_1D", 0)

    def test_setAndGet_maxspec_1D_ERROR_nonInteger(self) -> None:
        """Test attempting to set invalid maxspec_1D (non-integer)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_1D", 314.15)

    def test_setAndGet_maxspec_1D_ERROR_negative(self) -> None:
        """Test attempting to set invalid maxspec_1D (negative)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_1D", -3)

    def test_setAndGet_maxspec_1D_ERROR_negativeInf(self) -> None:
        """Test attempting to set invalid maxspec_1D (-Inf)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_1D", -math.inf)

    def test_setAndGet_maxspec_1D_ERROR_NaN(self) -> None:
        """Test attempting to set invalid maxspec_1D (NaN)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_1D", math.nan)

    # ------------------------------------------------------------------
    # maxspec_2D
    # ------------------------------------------------------------------

    def test_setAndGet_maxspec_2D(self) -> None:
        """Test setting and getting maxspec_2D."""
        graph_config.reset()

        graph_config.set("maxspec_2D", 468)
        val = graph_config.get("maxspec_2D")
        assert val == 468

        graph_config.set("maxspec_2D", math.inf)
        assert math.isinf(graph_config.get("maxspec_2D"))

    def test_setAndGet_maxspec_2D_ERROR_zero(self) -> None:
        """Test attempting to set invalid maxspec_2D (0)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_2D", 0)

    def test_setAndGet_maxspec_2D_ERROR_nonInteger(self) -> None:
        """Test attempting to set invalid maxspec_2D (non-integer)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_2D", 314.15)

    def test_setAndGet_maxspec_2D_ERROR_negative(self) -> None:
        """Test attempting to set invalid maxspec_2D (negative)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_2D", -3)

    def test_setAndGet_maxspec_2D_ERROR_negativeInf(self) -> None:
        """Test attempting to set invalid maxspec_2D (-Inf)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_2D", -math.inf)

    def test_setAndGet_maxspec_2D_ERROR_NaN(self) -> None:
        """Test attempting to set invalid maxspec_2D (NaN)."""
        with pytest.raises(
            HoraceConfigError, match="HERBERT:graph_config:invalid_argument"
        ):
            graph_config.set("maxspec_2D", math.nan)

    # ------------------------------------------------------------------
    # graph_config tests - for the instance syntax type:
    #
    #   g = graph_config() / g = graph_config.instance()
    #   g.<property_name> = <value>
    #   <value> = g.<property_name>
    #   g.reset()
    #
    # and its interoperability with graph_config.<method> and config_base
    # ------------------------------------------------------------------

    def test_setterGetterSyntax_get_instance(self) -> None:
        """Tests that more than one instance of `graph_config` can be created
        and share state via `config_store`.
        """
        graph_config.reset()

        a = graph_config.instance()
        b = graph_config()

        acol = a.colors
        bcol = b.colors
        gcol = graph_config.get("colors")

        assert gcol == self.S_default["colors"], (
            "Base singleton 'color' does not have the default value"
        )
        assert acol == bcol, "'colors' mismatch - two instances"
        assert acol == gcol, "'colors' mismatch - with base singleton"
        assert bcol == gcol, "'colors' mismatch - with base singleton"

    def test_setterGetterSyntax_set_property(self) -> None:
        """Tests that the property setters and getters work - set on the base
        singleton.
        """
        graph_config.reset()

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != self.S_default["colors"], (
            "New colors are the same as the defaults - invalidates the test"
        )
        graph_config.set("colors", new_colors)

        acol = a.colors
        bcol = b.colors
        gcol = graph_config.get("colors")

        assert gcol == new_colors, (
            "Base singleton 'color' does not have the expected value"
        )
        assert acol == bcol, "'colors' mismatch - two instances"
        assert acol == gcol, "'colors' mismatch - with base singleton"
        assert bcol == gcol, "'colors' mismatch - with base singleton"

    def test_setterGetterSyntax_set_property_on_an_instance(self) -> None:
        """Tests that the property setters and getters work - set on an
        instance of `graph_config`.
        """
        graph_config.reset()

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != self.S_default["colors"], (
            "New colors are the same as the defaults - invalidates the test"
        )
        a.colors = new_colors

        acol = a.colors
        bcol = b.colors
        gcol = graph_config.get("colors")

        assert gcol == new_colors, (
            "Base singleton 'color' does not have the expected value"
        )
        assert acol == bcol, "'colors' mismatch - two instances"
        assert acol == gcol, "'colors' mismatch - with base singleton"
        assert bcol == gcol, "'colors' mismatch - with base singleton"

    def test_setterGetterSyntax_reset(self) -> None:
        """Tests that `reset` acting on the base class properly resets the
        values of all instances to the default values.
        """
        graph_config.reset()

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != self.S_default["colors"], (
            "New colors are the same as the defaults - invalidates the test"
        )
        a.colors = new_colors

        graph_config.reset()

        acol = a.colors
        bcol = b.colors
        gcol = graph_config.get("colors")

        assert gcol == self.S_default["colors"], (
            "Base singleton 'color' does not have the expected value"
        )
        assert acol == bcol, "'colors' mismatch - two instances"
        assert acol == gcol, "'colors' mismatch - with base singleton"
        assert bcol == gcol, "'colors' mismatch - with base singleton"

    def test_setterGetterSyntax_reset_on_an_instance(self) -> None:
        """Tests that `reset` acting on an instance properly resets the values
        of all instances to the default values.
        """
        graph_config.reset()

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != self.S_default["colors"], (
            "New colors are the same as the defaults - invalidates the test"
        )
        a.colors = new_colors

        a.reset()

        acol = a.colors
        bcol = b.colors
        gcol = graph_config.get("colors")

        assert gcol == self.S_default["colors"], (
            "Base singleton 'color' does not have the expected value"
        )
        assert acol == bcol, "'colors' mismatch - two instances"
        assert acol == gcol, "'colors' mismatch - with base singleton"
        assert bcol == gcol, "'colors' mismatch - with base singleton"

    def test_clear_instances(self) -> None:
        """Tests that deleting all instances created by `a = graph_config.instance()`
        does not clear the stored `config_store` configuration.
        """
        graph_config.reset()
        default_colors = graph_config.get("colors")

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != default_colors, (
            "New colors are the same as the defaults - invalidates the test"
        )
        a.colors = new_colors

        del a, b

        colors = graph_config.get("colors")
        assert colors == new_colors, "Singleton was incorrectly reset"

    def test_clear_instances_and_graph_config(self) -> None:
        """Tests that clearing `graph_config` from `config_store` (including
        saved file) resets the properties to the defaults when next queried,
        while clearing only in-memory cache restores from JSON file on disk.
        """
        graph_config.reset()
        default_colors = graph_config.get("colors")

        a = graph_config.instance()
        b = graph_config.instance()

        new_colors = ["b", "g", "r"]
        assert new_colors != default_colors, (
            "New colors are the same as the defaults - invalidates the test"
        )
        a.colors = new_colors

        cfg_file = os.path.join(
            config_store.instance().config_folder, "graph_config.json"
        )
        assert os.path.isfile(cfg_file)

        # Clearing in-memory only reloads from JSON file on disk
        config_store.instance().clear_config("graph_config")
        assert get(graph_config(), "colors") == new_colors

        # Clearing instances AND stored config (memory + file) resets to defaults
        del a, b
        config_store.instance().clear_config("graph_config", "-files")

        colors = graph_config.get("colors")
        assert colors == default_colors, "Singleton was incorrectly reset"

