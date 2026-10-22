"""Graphics configuration class for PyHorace.

Translated from `herbert_core/graphics/genieplot.m` (renamed to `graph_config`)
and modified to inherit from `config_base` and persist configurations via
`config_store`.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .config_base import config_base
from .config_store import HoraceConfigError, config_store


def _is_hex_color(val: Any) -> bool:
    """Determine if a string has the form `'#rrggbb'` with each character
    pair being a hexadecimal number in the range 0 to 255.
    """
    if isinstance(val, str) and len(val) == 7 and val.startswith("#"):
        try:
            r = int(val[1:3], 16)
            g = int(val[3:5], 16)
            b = int(val[5:7], 16)
            return 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
        except ValueError:
            return False
    return False


class graph_config(config_base):
    """Configuration class to hold the state of Horace/Herbert graphics
    (translated from `herbert_core/graphics/genieplot.m`).

    It permits the setting and getting of graphics configuration values via
    `config_base` and `config_store`. This can be done in two ways:

    - Follow the standard configuration syntax implemented in Horace, where
      an instance pointing to the configuration store is created, and the
      properties are retrieved or changed for that instance (and all others)
      using the usual Python object attribute syntax:
          `<object>.<property> = <value>`
          `<value> = <object>.<property>`

    - Set or get property values using the `set` / `get` syntax:
          `graph_config.set(<name>, <value>)` or `set(graph_config, <name>, <value>)`
          `<value> = graph_config.get(<name>)` or `<value> = get(graph_config, <name>)`
      thereby being consistent with graphics property setting/getting and
      `config_base`.

    The two syntaxes can be used together. In more detail:

    Use: standard method
    --------------------
    Get an instance pointing to the `graph_config` configuration:
        >>> g = graph_config()
    or:
        >>> g = graph_config.instance()

    To set a property:
        >>> g.<property_name> = <value>

        EXAMPLE:
        >>> g.colors = ['b', 'r', 'g']

    To retrieve a property:
        >>> <value> = g.<property_name>

        EXAMPLE:
        >>> val = g.marker_sizes
        >>> val
        6

    To reset all properties to the defaults:
        >>> g.reset()
    or:
        >>> set(g, 'defaults')

    Use: set and get syntax directly on the class or instance
    ---------------------------------------------------------
    To set a property:
        >>> graph_config.set(<property_name>, <value>)
    or:
        >>> set(graph_config, <property_name>, <value>)

        EXAMPLE:
        >>> graph_config.set('colors', ['b', 'r', 'g'])

    To retrieve a property:
        >>> value = graph_config.get(<property_name>)
    or:
        >>> value = get(graph_config, <property_name>)

        EXAMPLE:
        >>> val = graph_config.get('marker_sizes')
        >>> val
        6

    Get a structure (`ConfigData`) with all properties:
        >>> S = graph_config.get()

    Set all properties from a dictionary/structure:
        >>> graph_config.set(S)

    To reset all properties to the defaults:
        >>> graph_config.reset()

    Example of mixed use
    --------------------
    The two syntaxes can be intermixed. For example:
        >>> a = graph_config.instance()    # get an instance of graph_config
        >>> b = graph_config()             # get a second instance
        >>> a.colors = ['b', 'g', 'r']     # set the colors with instance a
        >>> bcol = b.colors                # confirm colors have changed for b too
        >>> bcol
        ['b', 'g', 'r']
        >>> graph_config.reset()           # set all properties to the default values
        >>> bcol = b.colors                # confirm they have changed for b too
        >>> bcol
        ['k']
        >>> graph_config.set('colors', ['g', 'k'])  # change colors directly
        >>> bcol = b.colors                # confirm they have changed for b too
        >>> bcol
        ['g', 'k']

    Notes for Developers
    --------------------
    - The `reset` method resets all properties to the default values (`set('defaults')`),
      equivalent to issuing the `set` command for every property with the
      corresponding default value.
    - If all instances created by `a = graph_config()`, `b = graph_config()`
      are deleted (`del a, b`), this has no effect on the underlying
      `config_store` singleton values. If an instance `c = graph_config()` is
      created afterwards, the values of the properties are those that `a`, `b`
      had at the time they were deleted.

    `graph_config` properties are:
    ------------------------------
    General graph properties:
    - `default_fig_name` - Default name for plot (`str` or `None`)
    - `XScale`           - x-axis scaling: `'linear'` or `'log'`
    - `YScale`           - y-axis scaling: `'linear'` or `'log'`
    - `ZScale`           - z-axis scaling: `'linear'` or `'log'`

    One-dimensional graph properties:
    - `maxspec_1D`       - Maximum number of 1D datasets in a plottable array
    - `colors`           - List of default colors (`'r'`, `'g'`, `'b'`, `'c'`,
                           `'m'`, `'y'`, `'k'`, `'w'`, or `'#rrggbb'`)
    - `color_cycle`      - `'fast'` or `'with'`: colors cycle faster or with
                           line styles & widths and marker types & sizes
    - `line_styles`      - Default line style(s) (`'-'`, `'--'`, `':'`, `'-.'`)
    - `line_widths`      - Default line width(s) (positive real number(s))
    - `marker_types`     - Default marker type(s) (`'o'`, `'+'`, `'*'`, `'.'`,
                           `'x'`, `'_'`, `'|'`, `'s'`, `'d'`, `'^'`, `'v'`,
                           `'>'`, `'<'`, `'p'`, `'h'`)
    - `marker_sizes`     - Default marker size(s) (positive real number(s))

    Two-dimensional graph properties:
    - `maxspec_2D`       - Maximum number of 2D datasets in a plottable array

    Color theme properties:
    - `use_original_horace_plot_colours` - Revert to original Horace light color
                                           scheme (`bool`).
    """

    _SAVED_PROPERTIES: List[str] = [
        "XScale",
        "YScale",
        "ZScale",
        "color_cycle",
        "colors",
        "default_fig_name",
        "line_styles",
        "line_widths",
        "marker_sizes",
        "marker_types",
        "maxspec_1D",
        "maxspec_2D",
        "use_original_horace_plot_colours",
    ]

    _VALID_SCALES: Tuple[str, ...] = ("linear", "log")
    _VALID_COLOR_CODES: Tuple[str, ...] = ("r", "g", "b", "c", "m", "y", "k", "w")
    _VALID_COLOR_CYCLES: Tuple[str, ...] = ("fast", "with")
    _VALID_LINE_STYLES: Tuple[str, ...] = ("-", "--", ":", "-.")
    _VALID_MARKER_TYPES: Tuple[str, ...] = (
        "o",
        "+",
        "*",
        ".",
        "x",
        "_",
        "|",
        "s",
        "d",
        "^",
        "v",
        ">",
        "<",
        "p",
        "h",
    )

    def __init__(self) -> None:
        super().__init__("graph_config")
        # General graph properties
        self.default_fig_name_: Optional[str] = None
        self.XScale_: str = "linear"
        self.YScale_: str = "linear"
        self.ZScale_: str = "linear"

        # One-dimensional graph properties
        self.maxspec_1D_: Union[int, float] = 1000
        self.colors_: List[str] = ["k"]
        self.color_cycle_: str = "with"
        self.line_styles_: Union[str, List[str]] = ["-"]
        self.line_widths_: Union[float, List[float]] = 0.5
        self.marker_types_: Union[str, List[str]] = ["o"]
        self.marker_sizes_: Union[int, float, List[float]] = 6

        # Two-dimensional graph properties
        self.maxspec_2D_: Union[int, float] = 1000

        # Use standard Horace colour scheme for graphics
        self.use_original_horace_plot_colours_: bool = True

    # ------------------------------------------------------------------
    # Validation Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_scale(cls, name: str, val: Any) -> str:
        if isinstance(val, str) and val in cls._VALID_SCALES:
            return val
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            f"{name} must be 'linear' or 'log'",
        )

    @staticmethod
    def _validate_maxspec(name: str, val: Any) -> Union[int, float]:
        if isinstance(val, np.ndarray) and val.size == 1 and not np.issubdtype(val.dtype, np.bool_):
            val = val.item()
        if (
            isinstance(val, (int, float, np.number))
            and not isinstance(val, (bool, np.bool_))
            and not isinstance(val, complex)
        ):
            fval = float(val)
            if fval > 0 and (
                math.isinf(fval) or (math.isfinite(fval) and fval % 1 == 0)
            ):
                return float("inf") if math.isinf(fval) else int(fval)
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            f"{name} must be a positive integer or Inf",
        )

    @staticmethod
    def _validate_positive_reals(
        name: str, val: Any, preserve_int: bool = False
    ) -> Union[int, float, List[float]]:
        if (
            isinstance(val, (int, float, np.number))
            and not isinstance(val, (bool, np.bool_, complex))
        ):
            fval = float(val)
            if math.isfinite(fval) and fval > 0:
                if preserve_int and isinstance(val, (int, np.integer)):
                    return int(val)
                return fval
        elif isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(val)
            except Exception as exc:
                raise HoraceConfigError(
                    "HERBERT:graph_config:invalid_argument",
                    f"'{name}' must be reals greater than zero",
                ) from exc
            if (
                arr.size > 0
                and np.issubdtype(arr.dtype, np.number)
                and not np.issubdtype(arr.dtype, np.bool_)
                and not np.issubdtype(arr.dtype, np.complexfloating)
            ):
                flat = arr.reshape(-1)
                if bool(np.all(np.isfinite(flat))) and bool(np.all(flat > 0)):
                    if arr.ndim == 0:
                        return float(flat[0])
                    return [float(x) for x in flat]
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            f"'{name}' must be reals greater than zero",
        )

    # ------------------------------------------------------------------
    # Getters & Setters for Public Properties
    # ------------------------------------------------------------------

    @property
    def default_fig_name(self) -> Optional[str]:
        """The default name for a genie figure.
        - The empty string `''` is a valid name.
        - If the figure name is `None` (or set to `[]`), the hard-wired value
          for the relevant plot type will be used (one-dimensional plot,
          area plot, surface plot, etc.).
        """
        return self.get_or_restore_field("default_fig_name")

    @default_fig_name.setter
    def default_fig_name(self, val: Any) -> None:
        if isinstance(val, str):
            clean: Optional[str] = val.strip()
        elif val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0
        ):
            clean = None
        else:
            raise HoraceConfigError(
                "HERBERT:graph_config:invalid_argument",
                "The default name can only be set to a character vector or []",
            )
        config_store.instance().store_config(self, "default_fig_name", clean)

    @property
    def XScale(self) -> str:
        """x-axis scaling: `'linear'` or `'log'`."""
        return str(self.get_or_restore_field("XScale"))

    @XScale.setter
    def XScale(self, val: Any) -> None:
        checked = self._validate_scale("XScale", val)
        config_store.instance().store_config(self, "XScale", checked)

    @property
    def YScale(self) -> str:
        """y-axis scaling: `'linear'` or `'log'`."""
        return str(self.get_or_restore_field("YScale"))

    @YScale.setter
    def YScale(self, val: Any) -> None:
        checked = self._validate_scale("YScale", val)
        config_store.instance().store_config(self, "YScale", checked)

    @property
    def ZScale(self) -> str:
        """z-axis scaling: `'linear'` or `'log'`."""
        return str(self.get_or_restore_field("ZScale"))

    @ZScale.setter
    def ZScale(self, val: Any) -> None:
        checked = self._validate_scale("ZScale", val)
        config_store.instance().store_config(self, "ZScale", checked)

    @property
    def maxspec_1D(self) -> Union[int, float]:
        """Maximum number of 1D datasets in a plottable array."""
        val = self.get_or_restore_field("maxspec_1D")
        return float("inf") if math.isinf(float(val)) else int(val)

    @maxspec_1D.setter
    def maxspec_1D(self, val: Any) -> None:
        checked = self._validate_maxspec("maxspec_1D", val)
        config_store.instance().store_config(self, "maxspec_1D", checked)

    @property
    def colors(self) -> List[str]:
        """Row list of default colors (`'r'`, `'g'`, `'b'`, `'c'`, `'m'`, `'y'`,
        `'k'`, `'w'`, or hexadecimal `'#rrggbb'`).
        """
        return list(self.get_or_restore_field("colors"))

    @colors.setter
    def colors(self, val: Any) -> None:
        if isinstance(val, (list, tuple)) and len(val) > 0:
            items = list(val)
            if all(
                isinstance(x, str)
                and len(x) > 0
                and (x in self._VALID_COLOR_CODES or _is_hex_color(x))
                for x in items
            ):
                config_store.instance().store_config(self, "colors", items)
                return
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            "colors must be a cell array with elements from 'r','g','b','c','m',"
            "'y','k','w'\nor a valid hexadecimal of the form '#rrggbb'",
        )

    @property
    def color_cycle(self) -> str:
        """`'fast'` or `'with'`: colours cycle faster or with line styles & widths
        and marker types & sizes.
        """
        return str(self.get_or_restore_field("color_cycle"))

    @color_cycle.setter
    def color_cycle(self, val: Any) -> None:
        if isinstance(val, str):
            trimmed = val.rstrip()
            if trimmed.lower() in self._VALID_COLOR_CYCLES:
                config_store.instance().store_config(self, "color_cycle", trimmed)
                return
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            "'color_cycle' must be 'fast' or 'with'",
        )

    @property
    def line_styles(self) -> Union[str, List[str]]:
        """Default line style(s) from `'-'`, `'--'`, `':'`, `'-.'`."""
        val = self.get_or_restore_field("line_styles")
        if isinstance(val, list):
            return list(val)
        return str(val)

    @line_styles.setter
    def line_styles(self, val: Any) -> None:
        if isinstance(val, str) and val in self._VALID_LINE_STYLES:
            config_store.instance().store_config(self, "line_styles", val)
            return
        if (
            isinstance(val, (list, tuple))
            and len(val) > 0
            and all(isinstance(x, str) and x in self._VALID_LINE_STYLES for x in val)
        ):
            config_store.instance().store_config(self, "line_styles", list(val))
            return
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            "'line_styles 'must be a cell array with elements one of '-',  '--',  ':',  '-.'",
        )

    @property
    def line_widths(self) -> Union[float, List[float]]:
        """Row vector or scalar of default line widths (reals greater than zero)."""
        val = self.get_or_restore_field("line_widths")
        if isinstance(val, list):
            return list(val)
        return float(val)

    @line_widths.setter
    def line_widths(self, val: Any) -> None:
        checked = self._validate_positive_reals("line_widths", val)
        config_store.instance().store_config(self, "line_widths", checked)

    @property
    def marker_types(self) -> Union[str, List[str]]:
        """Default marker type(s) from `'o'`, `'+'`, `'*'`, `'.'`, `'x'`, `'_'`,
        `'|'`, `'s'`, `'d'`, `'^'`, `'v'`, `'>'`, `'<'`, `'p'`, `'h'`.
        """
        val = self.get_or_restore_field("marker_types")
        if isinstance(val, list):
            return list(val)
        return str(val)

    @marker_types.setter
    def marker_types(self, val: Any) -> None:
        if isinstance(val, str) and val in self._VALID_MARKER_TYPES:
            config_store.instance().store_config(self, "marker_types", val)
            return
        if (
            isinstance(val, (list, tuple))
            and len(val) > 0
            and all(isinstance(x, str) and x in self._VALID_MARKER_TYPES for x in val)
        ):
            config_store.instance().store_config(self, "marker_types", list(val))
            return
        raise HoraceConfigError(
            "HERBERT:graph_config:invalid_argument",
            "'marker_types 'must be a cell array of valid marker types",
        )

    @property
    def marker_sizes(self) -> Union[int, float, List[float]]:
        """Row vector or scalar of default marker sizes (reals greater than zero)."""
        val = self.get_or_restore_field("marker_sizes")
        if isinstance(val, list):
            return list(val)
        return val

    @marker_sizes.setter
    def marker_sizes(self, val: Any) -> None:
        checked = self._validate_positive_reals(
            "marker_sizes", val, preserve_int=True
        )
        config_store.instance().store_config(self, "marker_sizes", checked)

    @property
    def maxspec_2D(self) -> Union[int, float]:
        """Maximum number of 2D datasets in a plottable array."""
        val = self.get_or_restore_field("maxspec_2D")
        return float("inf") if math.isinf(float(val)) else int(val)

    @maxspec_2D.setter
    def maxspec_2D(self, val: Any) -> None:
        checked = self._validate_maxspec("maxspec_2D", val)
        config_store.instance().store_config(self, "maxspec_2D", checked)

    @property
    def use_original_horace_plot_colours(self) -> bool:
        """MATLAB 2025 introduced a new colour scheme which, if the OS theme is
        dark, makes Horace graphics unusable. If this option is True, Horace
        reverts to the original light colour scheme.
        """
        return bool(self.get_or_restore_field("use_original_horace_plot_colours"))

    @use_original_horace_plot_colours.setter
    def use_original_horace_plot_colours(self, val: Any) -> None:
        if not isinstance(val, (bool, int, float, np.bool_, np.number)):
            raise HoraceConfigError(
                "HERBERT:graph_config:invalid_argument",
                "use_original_horace_plot_colours must be convertible to logical/boolean",
            )
        config_store.instance().store_config(
            self, "use_original_horace_plot_colours", bool(val)
        )

    # ------------------------------------------------------------------
    # Abstract config_base Implementation
    # ------------------------------------------------------------------

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public names of the fields
        which should be saved.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set
        methods interface. Relies on assumption that each public field has a
        private field with name different by underscore.
        """
        val = getattr(self, f"{field_name}_")
        if isinstance(val, list):
            return list(val)
        return val


__all__ = ["graph_config"]

