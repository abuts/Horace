"""Matplotlib-based graphics and plotting utilities for PyHorace.

Translated from `herbert_core/graphics/` (including `genieplot`, `aline`,
`amark`, `acolor`, `property_index`, `lx`, `ly`, `lz`, `lc`, `linx`, `liny`,
`linz`, `logx`, `logy`, `logz`, `keep_figure`, `make_current`, `clearfigs`,
`is_genie_figure`, `src`, `spaghetti_plot`, and 1D/2D/3D dataset plotting
routines).
"""

from __future__ import annotations

import dis
import inspect
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np

# Use non-interactive Agg backend by default if no display or in headless test runs
try:
    matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from pyhorace.core.configurations import HoraceConfigError, graph_config, hor_config
from pyhorace.core.data_op_interface import DataOpArray


class HoraceError(ValueError):
    """Exception raised for invalid Horace/Herbert graphics or data arguments."""

    def __init__(self, err_id: str, message: str = "") -> None:
        self.err_id = err_id
        self.msg = message
        full_msg = f"{err_id}: {message}" if message else err_id
        super().__init__(full_msg)


# Alias genieplot to graph_config
genieplot = graph_config


# ---------------------------------------------------------------------------
# Color, Line Style, and Marker Tables
# ---------------------------------------------------------------------------

_SINGLE_LETTER_COLORS: Dict[str, str] = {
    "r": "r",
    "g": "g",
    "b": "b",
    "c": "c",
    "m": "m",
    "y": "y",
    "k": "k",
    "w": "w",
}

_COLOR_NAME_TO_CODE: List[Tuple[str, str]] = [
    ("red", "r"),
    ("green", "g"),
    ("blue", "b"),
    ("cyan", "c"),
    ("magenta", "m"),
    ("yellow", "y"),
    ("black", "k"),
    ("white", "w"),
    ("denim", "#0072BD"),
    ("carrot", "#D95319"),
    ("marigold", "#EDB120"),
    ("purple", "#7E2F8E"),
    ("grass", "#77AC30"),
    ("babyblue", "#4DBEEE"),
    ("brickred", "#A2142F"),
]

_CODE_TO_COLOR_NAME: Dict[str, str] = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white",
    "#0072BD": "denim",
    "#D95319": "carrot",
    "#EDB120": "marigold",
    "#7E2F8E": "purple",
    "#77AC30": "grass",
    "#4DBEEE": "babyblue",
    "#A2142F": "brickred",
}

_PALETTES: Dict[str, List[str]] = {
    "gem": [
        "#0072BD",
        "#D95319",
        "#EDB120",
        "#7E2F8E",
        "#77AC30",
        "#4DBEEE",
        "#A2142F",
    ],
}

_LINE_STYLE_NAMES: List[Tuple[str, str]] = [
    ("solid", "-"),
    ("dashed", "--"),
    ("dotted", ":"),
    ("ddot", "-."),
]

_VALID_LINE_CODES: Tuple[str, ...] = ("-", "--", ":", "-.")

_MARKER_NAMES: List[Tuple[str, str]] = [
    ("circle", "o"),
    ("plus", "+"),
    ("asterisk", "*"),
    ("point", "."),
    ("cross", "x"),
    ("horizontal", "_"),
    ("vertical", "|"),
    ("square", "s"),
    ("diamond", "d"),
    ("up", "^"),
    ("down", "v"),
    ("right", ">"),
    ("left", "<"),
    ("pentagram", "p"),
    ("hexagram", "h"),
]

_VALID_MARKER_CODES: Tuple[str, ...] = (
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

_MARKER_CODE_TO_MATLAB_PROP: Dict[str, str] = {
    "o": "o",
    "+": "+",
    "*": "*",
    ".": ".",
    "x": "x",
    "_": "_",
    "|": "|",
    "s": "square",
    "d": "diamond",
    "^": "^",
    "v": "v",
    ">": ">",
    "<": "<",
    "p": "pentagram",
    "h": "hexagram",
    "none": "none",
}

_COLOR_CODE_TO_RGB: Dict[str, List[float]] = {
    "r": [1.0, 0.0, 0.0],
    "g": [0.0, 1.0, 0.0],
    "b": [0.0, 0.0, 1.0],
    "c": [0.0, 1.0, 1.0],
    "m": [1.0, 0.0, 1.0],
    "y": [1.0, 1.0, 0.0],
    "k": [0.0, 0.0, 0.0],
    "w": [1.0, 1.0, 1.0],
}


def _color_to_rgb(c: str) -> List[float]:
    if c in _COLOR_CODE_TO_RGB:
        return list(_COLOR_CODE_TO_RGB[c])
    if c.startswith("#") and len(c) == 7:
        return [
            int(c[1:3], 16) / 255.0,
            int(c[3:5], 16) / 255.0,
            int(c[5:7], 16) / 255.0,
        ]
    rgb = mcolors.to_rgb(c)
    return [float(rgb[0]), float(rgb[1]), float(rgb[2])]


def _is_hex_color(val: Any) -> bool:
    if isinstance(val, str) and len(val) == 7 and val.startswith("#"):
        try:
            int(val[1:3], 16)
            int(val[3:5], 16)
            int(val[5:7], 16)
            return True
        except ValueError:
            return False
    return False


def _caller_expects_output() -> bool:
    """Detect whether the caller of `lx`/`ly`/`lz`/`lc` (1 or 2 frames up)
    assigns or uses the return value, vs calling it as a standalone statement.
    """
    try:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            return True
        caller = frame.f_back.f_back
        info = inspect.getframeinfo(caller)
        if info.code_context:
            stmt = info.code_context[0].strip()
            if stmt in ("lx()", "ly()", "lz()", "lc()"):
                return False
        lasti = caller.f_lasti
        instructions = list(dis.get_instructions(caller.f_code))
        call_idx = None
        for idx, instr in enumerate(instructions):
            if instr.offset <= lasti and "CALL" in instr.opname:
                call_idx = idx
        if call_idx is not None:
            for next_instr in instructions[call_idx + 1 :]:
                if next_instr.opname in ("CACHE", "NOP", "PRECALL"):
                    continue
                if next_instr.opname == "POP_TOP":
                    return False
                return True
    except Exception:
        pass
    return True


# ---------------------------------------------------------------------------
# PlotHandle and PlotHandleArray
# ---------------------------------------------------------------------------


class PlotHandle:
    """Wrapper around a Matplotlib artist providing MATLAB-compatible properties
    (`.Color`, `.LineStyle`, `.LineWidth`, `.Marker`, `.MarkerSize`, `.XData`,
    `.YData`, `.ZData`, `.CData`, `.kind`).
    """

    def __init__(
        self,
        artist: Any,
        kind: str = "line",
        color: Optional[Sequence[float]] = None,
        line_style: str = "-",
        line_width: float = 0.5,
        marker: str = "none",
        marker_size: float = 6.0,
        xdata: Optional[np.ndarray] = None,
        ydata: Optional[np.ndarray] = None,
        zdata: Optional[np.ndarray] = None,
        cdata: Optional[np.ndarray] = None,
        yerr: Optional[np.ndarray] = None,
    ) -> None:
        self.artist = artist
        self.kind = kind  # 'line', 'errorbar', 'patch', 'surface'
        self.Color = list(color) if color is not None else [0.0, 0.0, 0.0]
        self.LineStyle = line_style
        self.LineWidth = float(line_width)
        self.Marker = _MARKER_CODE_TO_MATLAB_PROP.get(marker, marker)
        self.MarkerSize = float(marker_size)
        self.XData = np.asarray(xdata, dtype=float) if xdata is not None else np.array([])
        self.YData = np.asarray(ydata, dtype=float) if ydata is not None else np.array([])
        self.ZData = np.asarray(zdata, dtype=float) if zdata is not None else np.array([])
        self.CData = np.asarray(cdata, dtype=float) if cdata is not None else np.array([])
        self.YErr = np.asarray(yerr, dtype=float) if yerr is not None else None
        self._is_deleted = False


class PlotHandleArray(list):
    """1D sequence of `PlotHandle` objects supporting both 0-based Python
    indexing `plot_h[i]` and 1-based MATLAB callable indexing `plot_h(i)`.
    """

    def __call__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, int):
            if idx >= 1:
                return self[idx - 1]
            return self[idx]
        return self[idx]

    @property
    def size(self) -> int:
        return len(self)


# ---------------------------------------------------------------------------
# Figure and Axes Extensions for MATLAB Graphics Compatibility
# ---------------------------------------------------------------------------


def _ensure_fig_attrs(fig: Figure) -> None:
    if not hasattr(fig, "_horace_initialized"):
        fig._horace_initialized = True
        fig._horace_name = ""
        fig._horace_tag = ""
        fig._horace_userdata = None
        fig._is_deleted = False


def _fig_get_name(self: Figure) -> str:
    _ensure_fig_attrs(self)
    return self._horace_name


def _fig_set_name(self: Figure, val: Optional[str]) -> None:
    _ensure_fig_attrs(self)
    self._horace_name = "" if val is None else str(val)


def _fig_get_tag(self: Figure) -> str:
    _ensure_fig_attrs(self)
    return self._horace_tag


def _fig_set_tag(self: Figure, val: Optional[str]) -> None:
    _ensure_fig_attrs(self)
    self._horace_tag = "" if val is None else str(val)


def _fig_get_userdata(self: Figure) -> Any:
    _ensure_fig_attrs(self)
    return self._horace_userdata


def _fig_set_userdata(self: Figure, val: Any) -> None:
    _ensure_fig_attrs(self)
    self._horace_userdata = val


def _fig_get_current_axes(self: Figure) -> Optional[Axes]:
    _ensure_fig_attrs(self)
    if len(self.axes) == 0:
        return None
    return self.gca()


Figure.Name = property(_fig_get_name, _fig_set_name)
Figure.Tag = property(_fig_get_tag, _fig_set_tag)
Figure.UserData = property(_fig_get_userdata, _fig_set_userdata)
Figure.CurrentAxes = property(_fig_get_current_axes)


def _ensure_ax_attrs(ax: Axes) -> None:
    if not hasattr(ax, "_horace_initialized"):
        ax._horace_initialized = True
        ax._is_deleted = False
        ax._horace_plot_handles: List[PlotHandle] = []
        ax._tracked_mpl_artists: set = set()
        ax._horace_plot_dim = 1  # 1, 2, or 3
        ax._horace_has_z = False
        ax._horace_has_c = False
        ax._horace_xlim: List[float] = [0.0, 1.0]
        ax._horace_ylim: List[float] = [0.0, 1.0]
        ax._horace_zlim: List[float] = [0.0, 1.0]
        ax._horace_clim: List[float] = [0.0, 1.0]
        ax._horace_xlim_mode: str = "auto"
        ax._horace_ylim_mode: str = "auto"
        ax._horace_zlim_mode: str = "auto"
        ax._horace_clim_mode: str = "auto"
        ax._horace_xlimit_method: str = "tight"
        ax._horace_ylimit_method: str = "tight"
        ax._horace_zlimit_method: str = "tight"
        ax._horace_xscale: str = "linear"
        ax._horace_yscale: str = "linear"
        ax._horace_zscale: str = "linear"
        ax._horace_xticklabel: List[str] = []


def _sync_raw_mpl_artists(ax: Axes) -> None:
    """Ensure any artists added directly via Matplotlib (`plt.plot`, `ax.plot_surface`, etc.)
    are also wrapped in `ax._horace_plot_handles`.
    """
    _ensure_ax_attrs(ax)
    for line in ax.lines:
        if id(line) not in ax._tracked_mpl_artists:
            ax._tracked_mpl_artists.add(id(line))
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            ph = PlotHandle(
                artist=line,
                kind="line",
                xdata=xd,
                ydata=yd,
            )
            ax._horace_plot_handles.append(ph)
    for coll in ax.collections:
        if id(coll) not in ax._tracked_mpl_artists:
            ax._tracked_mpl_artists.add(id(coll))
            ph = PlotHandle(
                artist=coll,
                kind="surface",
            )
            ax._horace_plot_handles.append(ph)


def _get_ax_plot_handles_lifo(ax: Axes) -> PlotHandleArray:
    _sync_raw_mpl_artists(ax)
    return PlotHandleArray(reversed(ax._horace_plot_handles))


def _nice_tick_step(span: float) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw_step = span / 4.0
    mag = 10.0 ** math.floor(math.log10(raw_step))
    residual = raw_step / mag
    if residual <= 1.5:
        step = 1.0 * mag
    elif residual <= 3.0:
        step = 2.0 * mag
    elif residual <= 7.0:
        step = 5.0 * mag
    else:
        step = 10.0 * mag
    return step


def _apply_limit_method(lo: float, hi: float, method: str) -> List[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    span = hi - lo
    method = method.lower()
    if method == "tight":
        return [float(lo), float(hi)]
    elif method in ("padded", "rounded"):
        pad = 0.07 * span
        return [float(lo - pad), float(hi + pad)]
    elif method in ("tickaligned", "stretch"):
        step = _nice_tick_step(span)
        aligned_lo = math.floor(lo / step) * step
        aligned_hi = math.ceil(hi / step) * step
        return [float(aligned_lo), float(aligned_hi)]
    return [float(lo), float(hi)]


def _recompute_ax_auto_limits(ax: Axes) -> None:
    """Recompute automatic axis limits on `ax` based on plotted data and
    current `*LimMode` and `*LimitMethod` settings.
    """
    _sync_raw_mpl_artists(ax)
    handles = ax._horace_plot_handles
    if not handles:
        return

    if ax._horace_plot_dim == 1 and not ax._horace_has_c:
        # 1D coupled auto-scaling:
        # If XLimMode is 'auto' and YLimMode is 'auto': full data range for both
        # If XLimMode is 'manual' and YLimMode is 'auto': YLim fits points in XLim
        # If YLimMode is 'manual' and XLimMode is 'auto': XLim fits points in YLim
        all_x: List[np.ndarray] = []
        all_y_lo: List[np.ndarray] = []
        all_y_hi: List[np.ndarray] = []
        for h in handles:
            if h.XData.size > 0 and h.YData.size > 0 and h.XData.size == h.YData.size:
                x = h.XData.ravel()
                y = h.YData.ravel()
                if h.YErr is not None and h.YErr.size == y.size:
                    ye = h.YErr.ravel()
                    ylo = y - ye
                    yhi = y + ye
                else:
                    ylo = y
                    yhi = y
                valid = np.isfinite(x) & np.isfinite(y)
                if np.any(valid):
                    all_x.append(x[valid])
                    all_y_lo.append(ylo[valid])
                    all_y_hi.append(yhi[valid])

        if not all_x:
            return

        x_concat = np.concatenate(all_x)
        ylo_concat = np.concatenate(all_y_lo)
        yhi_concat = np.concatenate(all_y_hi)

        if ax._horace_xlim_mode == "auto" and ax._horace_ylim_mode == "auto":
            ax._horace_xlim = _apply_limit_method(
                float(np.min(x_concat)), float(np.max(x_concat)), ax._horace_xlimit_method
            )
            ax._horace_ylim = _apply_limit_method(
                float(np.min(ylo_concat)), float(np.max(yhi_concat)), ax._horace_ylimit_method
            )
        elif ax._horace_xlim_mode == "manual" and ax._horace_ylim_mode == "auto":
            xlo, xhi = ax._horace_xlim
            mask = (x_concat >= xlo) & (x_concat <= xhi)
            if np.any(mask):
                ax._horace_ylim = _apply_limit_method(
                    float(np.min(ylo_concat[mask])),
                    float(np.max(yhi_concat[mask])),
                    ax._horace_ylimit_method,
                )
        elif ax._horace_ylim_mode == "manual" and ax._horace_xlim_mode == "auto":
            ylo, yhi = ax._horace_ylim
            mask = (ylo_concat >= ylo) & (yhi_concat <= yhi)
            if np.any(mask):
                ax._horace_xlim = _apply_limit_method(
                    float(np.min(x_concat[mask])),
                    float(np.max(x_concat[mask])),
                    ax._horace_xlimit_method,
                )
    else:
        # 2D / 3D auto-scaling
        all_x = [h.XData.ravel() for h in handles if h.XData.size > 0]
        all_y = [h.YData.ravel() for h in handles if h.YData.size > 0]
        all_z = [h.ZData.ravel() for h in handles if h.ZData.size > 0]
        all_c = [h.CData.ravel() for h in handles if h.CData.size > 0]

        if ax._horace_xlim_mode == "auto" and all_x:
            xc = np.concatenate(all_x)
            xc = xc[np.isfinite(xc)]
            if xc.size > 0:
                ax._horace_xlim = _apply_limit_method(
                    float(np.min(xc)), float(np.max(xc)), ax._horace_xlimit_method
                )
        if ax._horace_ylim_mode == "auto" and all_y:
            yc = np.concatenate(all_y)
            yc = yc[np.isfinite(yc)]
            if yc.size > 0:
                ax._horace_ylim = _apply_limit_method(
                    float(np.min(yc)), float(np.max(yc)), ax._horace_ylimit_method
                )
        if ax._horace_zlim_mode == "auto" and all_z:
            zc = np.concatenate(all_z)
            zc = zc[np.isfinite(zc)]
            if zc.size > 0:
                ax._horace_zlim = _apply_limit_method(
                    float(np.min(zc)), float(np.max(zc)), ax._horace_zlimit_method
                )
        if ax._horace_clim_mode == "auto" and all_c:
            cc = np.concatenate(all_c)
            cc = cc[np.isfinite(cc)]
            if cc.size > 0:
                ax._horace_clim = [float(np.min(cc)), float(np.max(cc))]

    try:
        if ax._horace_xscale != "log" or ax._horace_xlim[0] > 0:
            ax.set_xlim(ax._horace_xlim)
        if ax._horace_yscale != "log" or ax._horace_ylim[0] > 0:
            ax.set_ylim(ax._horace_ylim)
    except Exception:
        pass


def _ax_get_xlim(self: Axes) -> np.ndarray:
    _ensure_ax_attrs(self)
    return np.array(self._horace_xlim, dtype=float)


def _ax_set_xlim(self: Axes, val: Sequence[float]) -> None:
    _ensure_ax_attrs(self)
    self._horace_xlim = [float(val[0]), float(val[1])]
    self._horace_xlim_mode = "manual"
    _recompute_ax_auto_limits(self)
    try:
        self.set_xlim(self._horace_xlim)
    except Exception:
        pass


def _ax_get_ylim(self: Axes) -> np.ndarray:
    _ensure_ax_attrs(self)
    return np.array(self._horace_ylim, dtype=float)


def _ax_set_ylim(self: Axes, val: Sequence[float]) -> None:
    _ensure_ax_attrs(self)
    self._horace_ylim = [float(val[0]), float(val[1])]
    self._horace_ylim_mode = "manual"
    _recompute_ax_auto_limits(self)
    try:
        self.set_ylim(self._horace_ylim)
    except Exception:
        pass


def _ax_get_zlim(self: Axes) -> np.ndarray:
    _ensure_ax_attrs(self)
    return np.array(self._horace_zlim, dtype=float)


def _ax_set_zlim(self: Axes, val: Sequence[float]) -> None:
    _ensure_ax_attrs(self)
    self._horace_zlim = [float(val[0]), float(val[1])]
    self._horace_zlim_mode = "manual"
    _recompute_ax_auto_limits(self)


def _ax_get_clim(self: Axes) -> np.ndarray:
    _ensure_ax_attrs(self)
    return np.array(self._horace_clim, dtype=float)


def _ax_set_clim(self: Axes, val: Sequence[float]) -> None:
    _ensure_ax_attrs(self)
    self._horace_clim = [float(val[0]), float(val[1])]
    self._horace_clim_mode = "manual"


def _ax_get_xscale(self: Axes) -> str:
    _ensure_ax_attrs(self)
    return self._horace_xscale


def _ax_set_xscale(self: Axes, val: str) -> None:
    _ensure_ax_attrs(self)
    self._horace_xscale = str(val)
    try:
        self.set_xscale(self._horace_xscale)
    except Exception:
        pass


def _ax_get_yscale(self: Axes) -> str:
    _ensure_ax_attrs(self)
    return self._horace_yscale


def _ax_set_yscale(self: Axes, val: str) -> None:
    _ensure_ax_attrs(self)
    self._horace_yscale = str(val)
    try:
        self.set_yscale(self._horace_yscale)
    except Exception:
        pass


def _ax_get_zscale(self: Axes) -> str:
    _ensure_ax_attrs(self)
    return self._horace_zscale


def _ax_set_zscale(self: Axes, val: str) -> None:
    _ensure_ax_attrs(self)
    self._horace_zscale = str(val)


def _ax_get_xticklabel(self: Axes) -> List[str]:
    _ensure_ax_attrs(self)
    return list(self._horace_xticklabel)


def _ax_set_xticklabel(self: Axes, val: Sequence[str]) -> None:
    _ensure_ax_attrs(self)
    self._horace_xticklabel = [str(v) for v in val]


Axes.XLim = property(_ax_get_xlim, _ax_set_xlim)
Axes.YLim = property(_ax_get_ylim, _ax_set_ylim)
Axes.ZLim = property(_ax_get_zlim, _ax_set_zlim)
Axes.CLim = property(_ax_get_clim, _ax_set_clim)
Axes.XScale = property(_ax_get_xscale, _ax_set_xscale)
Axes.YScale = property(_ax_get_yscale, _ax_set_yscale)
Axes.ZScale = property(_ax_get_zscale, _ax_set_zscale)
Axes.XTickLabel = property(_ax_get_xticklabel, _ax_set_xticklabel)
Axes.Children = property(_get_ax_plot_handles_lifo)


# ---------------------------------------------------------------------------
# Style Control Functions: aline, amark, acolor, property_index
# ---------------------------------------------------------------------------


def _try_parse_numeric_arg(arg: Any) -> Optional[List[float]]:
    """Try to parse `arg` as a numeric scalar, array, or MATLAB-style colon
    range string (e.g. `'1:2:7'` or `'4.5'`). Returns `None` if not numeric.
    """
    if isinstance(arg, (int, float, np.number)) and not isinstance(arg, (bool, np.bool_)):
        return [float(arg)]
    if isinstance(arg, np.ndarray) and np.issubdtype(arg.dtype, np.number):
        return [float(x) for x in arg.ravel()]
    if isinstance(arg, (list, tuple)) and len(arg) > 0:
        if all(
            isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_))
            for x in arg
        ):
            return [float(x) for x in arg]
    if isinstance(arg, str):
        s = arg.strip()
        # Check colon expression like '1:2:7' or '1:5'
        if ":" in s and not s.startswith(":"):
            parts = s.split(":")
            try:
                nums = [float(p) for p in parts]
                if len(nums) == 2:
                    arr = np.arange(nums[0], nums[1] + 0.5 * 1.0, 1.0)
                    return [float(x) for x in arr]
                elif len(nums) == 3:
                    arr = np.arange(nums[0], nums[2] + 0.5 * nums[1], nums[1])
                    return [float(x) for x in arr]
            except ValueError:
                pass
        try:
            val = float(s)
            return [val]
        except ValueError:
            return None
    return None


def _match_line_style(s: str) -> str:
    st = s.strip()
    if st in _VALID_LINE_CODES:
        return st
    st_low = st.lower()
    matches = [code for name, code in _LINE_STYLE_NAMES if name.startswith(st_low)]
    if len(matches) == 1:
        return matches[0]
    raise HoraceError(
        "HERBERT:graphics:invalid_argument",
        f"Unrecognised or ambiguous line style '{s}'",
    )


def aline(*args: Any) -> Tuple[Union[float, List[float]], Union[str, List[str]]]:
    """Set and/or retrieve default line widths and line styles in `genieplot`."""
    widths: List[float] = []
    styles: List[str] = []

    for arg in args:
        nums = _try_parse_numeric_arg(arg)
        if nums is not None:
            widths.extend(nums)
        elif isinstance(arg, str):
            styles.append(_match_line_style(arg))
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                item_nums = _try_parse_numeric_arg(item)
                if item_nums is not None:
                    widths.extend(item_nums)
                elif isinstance(item, str):
                    styles.append(_match_line_style(item))
                else:
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        f"Invalid argument to aline: {item!r}",
                    )
        else:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                f"Invalid argument to aline: {arg!r}",
            )

    if widths:
        if any(not math.isfinite(w) or w <= 0 for w in widths):
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "Line widths must be positive real numbers",
            )
        graph_config.set("line_widths", widths[0] if len(widths) == 1 else widths)

    if styles:
        graph_config.set("line_styles", styles)

    cur_widths = graph_config.get("line_widths")
    cur_styles = graph_config.get("line_styles")
    if isinstance(cur_styles, list):
        style_out: Union[str, List[str]] = (
            cur_styles[0] if len(cur_styles) == 1 else list(cur_styles)
        )
    else:
        style_out = str(cur_styles)

    return cur_widths, style_out


def _match_marker_type(s: str) -> str:
    st = s.strip()
    st_low = st.lower()
    if st_low in _VALID_MARKER_CODES:
        return st_low
    matches = [code for name, code in _MARKER_NAMES if name.startswith(st_low)]
    if len(matches) == 1:
        return matches[0]
    raise HoraceError(
        "HERBERT:graphics:invalid_argument",
        f"Unrecognised or ambiguous marker type '{s}'",
    )


def amark(*args: Any) -> Tuple[Union[int, float, List[float]], Union[str, List[str]]]:
    """Set and/or retrieve default marker sizes and marker types in `genieplot`."""
    sizes: List[float] = []
    markers: List[str] = []

    for arg in args:
        nums = _try_parse_numeric_arg(arg)
        if nums is not None:
            sizes.extend(nums)
        elif isinstance(arg, str):
            markers.append(_match_marker_type(arg))
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                item_nums = _try_parse_numeric_arg(item)
                if item_nums is not None:
                    sizes.extend(item_nums)
                elif isinstance(item, str):
                    markers.append(_match_marker_type(item))
                else:
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        f"Invalid argument to amark: {item!r}",
                    )
        else:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                f"Invalid argument to amark: {arg!r}",
            )

    if sizes:
        if any(not math.isfinite(sz) or sz <= 0 for sz in sizes):
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "Marker sizes must be positive real numbers",
            )
        graph_config.set("marker_sizes", sizes[0] if len(sizes) == 1 else sizes)

    if markers:
        graph_config.set("marker_types", markers)

    cur_sizes = graph_config.get("marker_sizes")
    cur_markers = graph_config.get("marker_types")
    if isinstance(cur_markers, list):
        marker_out: Union[str, List[str]] = (
            cur_markers[0] if len(cur_markers) == 1 else list(cur_markers)
        )
    else:
        marker_out = str(cur_markers)

    return cur_sizes, marker_out


def _match_color_or_palette(s: str) -> List[str]:
    st = s.strip()
    if _is_hex_color(st):
        return [st.upper()]
    st_low = st.lower()
    # Exact single-letter match takes priority over prefix match (e.g. 'g' vs 'gem')
    if st_low in _SINGLE_LETTER_COLORS:
        return [_SINGLE_LETTER_COLORS[st_low]]

    # Prefix match against color names and palettes
    color_matches = [code for name, code in _COLOR_NAME_TO_CODE if name.startswith(st_low)]
    palette_matches = [
        pal_codes for pal_name, pal_codes in _PALETTES.items() if pal_name.startswith(st_low)
    ]
    total_matches = len(color_matches) + len(palette_matches)
    if total_matches == 1:
        if color_matches:
            return [color_matches[0]]
        return list(palette_matches[0])

    raise HoraceError(
        "HERBERT:graphics:invalid_argument",
        f"Unrecognised or ambiguous color or palette name '{s}'",
    )


def acolor(*args: Any) -> Tuple[Union[str, List[str]], str]:
    """Set and/or retrieve default colors and color cycle mode in `genieplot`."""
    colors: List[str] = []
    cycle: Optional[str] = None

    nargs = len(args)
    for idx, arg in enumerate(args):
        if isinstance(arg, str) and arg.strip().startswith("-"):
            opt = arg.strip()[1:].lower()
            if opt and "fast".startswith(opt):
                if idx != nargs - 1:
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        "Color cycle option must be the last input argument",
                    )
                cycle = "fast"
            elif opt and "with".startswith(opt):
                if idx != nargs - 1:
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        "Color cycle option must be the last input argument",
                    )
                cycle = "with"
            else:
                raise HoraceError(
                    "HERBERT:graphics:invalid_argument",
                    f"Unrecognised option '{arg}'",
                )
        elif isinstance(arg, str):
            colors.extend(_match_color_or_palette(arg))
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                if not isinstance(item, str) or item.strip().startswith("-"):
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        f"Invalid color specification {item!r}",
                    )
                colors.extend(_match_color_or_palette(item))
        else:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                f"Invalid argument to acolor: {arg!r}",
            )

    if colors:
        graph_config.set("colors", colors)
    if cycle is not None:
        graph_config.set("color_cycle", cycle)

    cur_codes = graph_config.get("colors")
    cur_names = [_CODE_TO_COLOR_NAME.get(c, c) for c in cur_codes]
    col_out: Union[str, List[str]] = cur_names[0] if len(cur_names) == 1 else cur_names
    cycle_out = graph_config.get("color_cycle")
    return col_out, cycle_out


def property_index(
    nw: int, color_cycle: str, ncol: int, *prop_lengths: int
) -> Tuple[np.ndarray, ...]:
    """Compute 0-based cycling indices for colors and other line/marker properties."""
    idx = np.arange(nw, dtype=int)
    icol = idx % ncol
    if color_cycle.lower() == "with":
        iprop_base = idx
    elif color_cycle.lower() == "fast":
        iprop_base = idx // ncol
    else:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            f"Unrecognised color_cycle '{color_cycle}'",
        )
    out = [icol]
    for n_k in prop_lengths:
        out.append(iprop_base % n_k)
    return tuple(out)


# ---------------------------------------------------------------------------
# Figure Management: is_genie_figure, keep_figure, make_current, clearfigs,
#                    figure, gcf, gca, isgraphics, src
# ---------------------------------------------------------------------------


def _all_open_figures() -> List[Figure]:
    figs: List[Figure] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        _ensure_fig_attrs(fig)
        if not fig._is_deleted:
            figs.append(fig)
    return figs


def gcf() -> Optional[Figure]:
    fignums = plt.get_fignums()
    if not fignums:
        return None
    fig = plt.gcf()
    _ensure_fig_attrs(fig)
    return fig


def gca() -> Optional[Axes]:
    fig = gcf()
    if fig is None or len(fig.axes) == 0:
        return None
    ax = fig.gca()
    _ensure_ax_attrs(ax)
    return ax


def figure(fig: Optional[Union[int, Figure]] = None) -> Figure:
    """Make `fig` the current active Matplotlib figure (or create a new one)."""
    if fig is None:
        f = plt.figure()
    elif isinstance(fig, Figure):
        f = plt.figure(fig.number)
    else:
        f = plt.figure(int(fig))
    _ensure_fig_attrs(f)
    return f


def isgraphics(h: Any) -> Union[bool, np.ndarray]:
    """Check whether handle(s) `h` refer to valid (non-deleted) graphics objects."""
    if isinstance(h, (list, tuple, np.ndarray, PlotHandleArray)):
        return np.array([bool(isgraphics(item)) for item in h], dtype=bool)
    if isinstance(h, Figure):
        _ensure_fig_attrs(h)
        return (not h._is_deleted) and plt.fignum_exists(h.number)
    if isinstance(h, Axes):
        _ensure_ax_attrs(h)
        return (
            (not h._is_deleted)
            and (h.figure is not None)
            and isgraphics(h.figure)
            and (h in h.figure.axes)
        )
    if isinstance(h, PlotHandle):
        return not h._is_deleted
    return False


def is_genie_figure(
    fig_handle: Optional[Union[Figure, int, Sequence[Any]]] = None,
) -> Tuple[Union[bool, np.ndarray], Union[bool, np.ndarray], Union[str, List[str]]]:
    """Determine if figure handle(s) are genie figures and whether they have
    `'current'` status.
    """
    if fig_handle is None:
        fig_handle = gcf()
        if fig_handle is None:
            return False, False, ""

    if isinstance(fig_handle, (list, tuple, np.ndarray)):
        ok_list: List[bool] = []
        cur_list: List[bool] = []
        name_list: List[str] = []
        for fh in fig_handle:
            ok_i, cur_i, name_i = is_genie_figure(fh)
            ok_list.append(bool(ok_i))
            cur_list.append(bool(cur_i))
            name_list.append(str(name_i))
        return np.array(ok_list, dtype=bool), np.array(cur_list, dtype=bool), name_list

    if isinstance(fig_handle, (int, np.integer)):
        if not plt.fignum_exists(int(fig_handle)):
            return False, False, ""
        fig_obj = plt.figure(int(fig_handle))
    elif isinstance(fig_handle, Figure):
        fig_obj = fig_handle
    else:
        return False, False, ""

    if not isgraphics(fig_obj):
        return False, False, ""

    _ensure_fig_attrs(fig_obj)
    tag = fig_obj.Tag
    if tag.endswith("$current$"):
        name = tag[: -len("$current$")]
        return True, True, name
    elif tag.endswith("$keep$"):
        name = tag[: -len("$keep$")]
        return True, False, name
    return False, False, ""


def keep_figure(fig: Optional[Union[Figure, int, str, Sequence[Any]]] = None) -> None:
    """Change the status of target genie figure(s) from `'current'` to `'keep'`."""
    targets = _resolve_genie_figure_targets(fig)
    for f in targets:
        ok, is_cur, name = is_genie_figure(f)
        if ok and is_cur:
            f.Tag = f"{name}$keep$"


def make_current(fig: Optional[Union[Figure, int]] = None) -> None:
    """Give a kept genie figure `'current'` status and make it the active figure."""
    if fig is None:
        target = gcf()
    elif isinstance(fig, Figure):
        target = fig
    elif isinstance(fig, (int, np.integer)) and plt.fignum_exists(int(fig)):
        target = plt.figure(int(fig))
    else:
        raise HoraceError("HERBERT:graphics:invalid_argument", f"Invalid figure {fig!r}")

    if target is None:
        return
    ok, _, name = is_genie_figure(target)
    if not ok:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "Figure is not a genie figure",
        )
    # Any other genie figure with the same name becomes 'keep'
    for other in _all_open_figures():
        ok_o, cur_o, name_o = is_genie_figure(other)
        if ok_o and cur_o and name_o == name and other is not target:
            other.Tag = f"{name}$keep$"
    target.Tag = f"{name}$current$"
    plt.figure(target.number)


def _mark_figure_deleted(f: Figure) -> None:
    _ensure_fig_attrs(f)
    f._is_deleted = True
    for ax in list(f.axes):
        _ensure_ax_attrs(ax)
        ax._is_deleted = True
        for ph in ax._horace_plot_handles:
            ph._is_deleted = True


def clearfigs(fig: Optional[Any] = None) -> None:
    """Close and delete specified figure(s) or all open figures."""
    if fig is None or (isinstance(fig, str) and fig == "-all"):
        for f in _all_open_figures():
            _mark_figure_deleted(f)
        plt.close("all")
        return

    targets = _resolve_genie_figure_targets(fig)
    for f in targets:
        _mark_figure_deleted(f)
        plt.close(f)


def _resolve_genie_figure_targets(fig: Optional[Any]) -> List[Figure]:
    if fig is None:
        cur = gcf()
        return [cur] if cur is not None else []
    if isinstance(fig, Figure):
        return [fig] if isgraphics(fig) else []
    if isinstance(fig, (int, np.integer)):
        if plt.fignum_exists(int(fig)):
            return [plt.figure(int(fig))]
        return []
    if isinstance(fig, str):
        return [f for f in _all_open_figures() if f.Name == fig]
    if isinstance(fig, (list, tuple, np.ndarray)):
        out: List[Figure] = []
        for item in fig:
            out.extend(_resolve_genie_figure_targets(item))
        return out
    return []


def src(fig: Optional[Union[Figure, int, str]] = None) -> Any:
    """Extract the source dataset(s) stored in `fig.UserData`."""
    targets = _resolve_genie_figure_targets(fig)
    if len(targets) == 0:
        raise HoraceError("HERBERT:src:invalid_argument", "No figure found")
    if len(targets) > 1:
        raise HoraceError("HERBERT:src:invalid_argument", "More than one figure matches")
    f = targets[0]
    data = f.UserData
    if data is None:
        return None
    if isinstance(data, list):
        if len(data) == 0:
            return None
        if len(data) == 1:
            return data[0]
        first_cls = type(data[0])
        if all(type(x) is first_cls for x in data):
            return DataOpArray(data)
        return list(data)
    return data


def _get_or_create_target_figure_and_axes(
    default_name: str,
    new_axes: bool,
    force_current_axes: bool,
    custom_name: Optional[str] = None,
    custom_axes: Optional[Axes] = None,
    projection_3d: bool = False,
) -> Tuple[Figure, Axes, bool]:
    """Locate or create the target `(fig_h, axes_h, created_new_axes)` according to
    Herbert figure targeting rules.
    """
    if custom_axes is not None and isgraphics(custom_axes):
        fig_h = custom_axes.figure
        _ensure_fig_attrs(fig_h)
        plt.figure(fig_h.number)
        if new_axes:
            # Clear old axes and create fresh axes
            for ax_old in list(fig_h.axes):
                _ensure_ax_attrs(ax_old)
                ax_old._is_deleted = True
                for ph in ax_old._horace_plot_handles:
                    ph._is_deleted = True
            fig_h.clf()
            ax_h = fig_h.add_subplot(111)
            _ensure_ax_attrs(ax_h)
            return fig_h, ax_h, True
        return fig_h, custom_axes, False

    cfg_name = graph_config.get("default_fig_name")
    target_name = (
        custom_name
        if custom_name is not None
        else (cfg_name if cfg_name is not None else default_name)
    )

    if force_current_axes:
        # 'plot over current' (*oc): use gcf / gca if any figure exists
        cur_fig = gcf()
        if cur_fig is not None and isgraphics(cur_fig):
            cur_ax = gca()
            if cur_ax is not None and isgraphics(cur_ax):
                return cur_fig, cur_ax, False
            ax_h = cur_fig.add_subplot(111)
            _ensure_ax_attrs(ax_h)
            return cur_fig, ax_h, True
        # No figure exists at all -> create new genie figure
        fig_h = plt.figure()
        _ensure_fig_attrs(fig_h)
        fig_h.Name = target_name
        fig_h.Tag = f"{target_name}$current$"
        ax_h = fig_h.add_subplot(111)
        _ensure_ax_attrs(ax_h)
        return fig_h, ax_h, True

    # Look for an existing genie figure with target_name and '$current$' status
    found_fig: Optional[Figure] = None
    for f in _all_open_figures():
        ok, is_cur, name = is_genie_figure(f)
        if ok and is_cur and name == target_name:
            found_fig = f
            break

    if found_fig is not None:
        plt.figure(found_fig.number)
        if new_axes or len(found_fig.axes) == 0:
            for ax_old in list(found_fig.axes):
                _ensure_ax_attrs(ax_old)
                ax_old._is_deleted = True
                for ph in ax_old._horace_plot_handles:
                    ph._is_deleted = True
            found_fig.clf()
            ax_h = found_fig.add_subplot(111)
            _ensure_ax_attrs(ax_h)
            return found_fig, ax_h, True
        else:
            ax_h = found_fig.gca()
            _ensure_ax_attrs(ax_h)
            return found_fig, ax_h, False

    # Create a new genie figure
    fig_h = plt.figure()
    _ensure_fig_attrs(fig_h)
    fig_h.Name = target_name
    fig_h.Tag = f"{target_name}$current$"
    ax_h = fig_h.add_subplot(111)
    _ensure_ax_attrs(ax_h)
    return fig_h, ax_h, True


# ---------------------------------------------------------------------------
# Scale & Limit Functions: linx, liny, linz, logx, logy, logz,
#                          lx, ly, lz, lc, xlim, ylim, zlim, caxis
# ---------------------------------------------------------------------------


def linx() -> None:
    graph_config.set("XScale", "linear")
    ax = gca()
    if ax is not None:
        ax.XScale = "linear"


def logx() -> None:
    graph_config.set("XScale", "log")
    ax = gca()
    if ax is not None:
        ax.XScale = "log"


def liny() -> None:
    graph_config.set("YScale", "linear")
    ax = gca()
    if ax is not None:
        ax.YScale = "linear"


def logy() -> None:
    graph_config.set("YScale", "log")
    ax = gca()
    if ax is not None:
        ax.YScale = "log"


def linz() -> None:
    graph_config.set("ZScale", "linear")
    ax = gca()
    if ax is not None:
        ax.ZScale = "linear"


def logz() -> None:
    graph_config.set("ZScale", "log")
    ax = gca()
    if ax is not None:
        ax.ZScale = "log"


def xlim(lims: Optional[Sequence[float]] = None) -> np.ndarray:
    ax = gca()
    if ax is None:
        return np.array([0.0, 1.0])
    if lims is not None:
        ax.XLim = lims
    return ax.XLim


def ylim(lims: Optional[Sequence[float]] = None) -> np.ndarray:
    ax = gca()
    if ax is None:
        return np.array([0.0, 1.0])
    if lims is not None:
        ax.YLim = lims
    return ax.YLim


def zlim(lims: Optional[Sequence[float]] = None) -> np.ndarray:
    ax = gca()
    if ax is None:
        return np.array([0.0, 1.0])
    if lims is not None:
        ax.ZLim = lims
    return ax.ZLim


def caxis(lims: Optional[Sequence[float]] = None) -> np.ndarray:
    ax = gca()
    if ax is None:
        return np.array([0.0, 1.0])
    if lims is not None:
        ax.CLim = lims
    return ax.CLim


_LIMIT_METHODS = ("padded", "rounded", "tickaligned", "tight")


def _set_limits(ax_name: str, *args: Any, _expects_output: bool = False) -> Optional[Tuple[float, float]]:
    fig = gcf()
    if fig is None:
        return None
    ax = gca()
    if ax is None:
        return None
    _ensure_ax_attrs(ax)
    _sync_raw_mpl_artists(ax)

    ax_upper = ax_name.upper()
    if ax_upper == "Z" and not ax._horace_has_z:
        if not ax._horace_has_c:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "No z-axis or colour data on the current figure and axes - change limits ignored",
            )
        ax_upper = "C"
    elif ax_upper == "C":
        if not ax._horace_has_c:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "No colour data on the current figure and axes - change limits ignored",
            )

    narg = len(args)

    # If no arguments and caller expects output (`xlo, xhi = lx()`): return current limits
    if narg == 0 and _expects_output:
        lims = getattr(ax, f"{ax_upper}Lim")
        return float(lims[0]), float(lims[1])

    # Reset to auto or set LimitMethod ('tight', 'padded', 'rounded', 'tickaligned')
    if narg == 0 or (narg == 1 and isinstance(args[0], str) and len(args[0].strip()) > 0):
        if narg == 1:
            opt = args[0].strip().lower()
            matches = [m for m in _LIMIT_METHODS if m.startswith(opt)]
            if len(matches) != 1:
                raise HoraceError(
                    "HERBERT:graphics:invalid_argument",
                    "Check the validity and uniqueness of the automatic limit method option",
                )
            chosen = "padded" if matches[0] == "rounded" else matches[0]
            if ax_upper != "C":
                setattr(ax, f"_horace_{ax_upper.lower()}limit_method", chosen)

        setattr(ax, f"_horace_{ax_upper.lower()}lim_mode", "auto")
        _recompute_ax_auto_limits(ax)
        lims = getattr(ax, f"{ax_upper}Lim")
        return float(lims[0]), float(lims[1])

    # Parse numeric range(s)
    if narg == 2:
        a0 = _try_parse_numeric_arg(args[0]) if isinstance(args[0], str) else None
        a1 = _try_parse_numeric_arg(args[1]) if isinstance(args[1], str) else None
        v0 = a0[0] if a0 is not None and len(a0) == 1 else args[0]
        v1 = a1[0] if a1 is not None and len(a1) == 1 else args[1]
        try:
            lo = float(np.asarray(v0).item())
            hi = float(np.asarray(v1).item())
        except Exception as exc:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "Check the input arguments are two numeric scalars",
            ) from exc
    elif narg == 1:
        arr = np.asarray(args[0], dtype=float).ravel()
        if arr.size != 2:
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "Input argument must be a 2-element numeric pair [lo, hi]",
            )
        lo, hi = float(arr[0]), float(arr[1])
    else:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "Invalid arguments to limit function",
        )

    if math.isnan(lo) or math.isnan(hi) or lo >= hi:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "Check the input arguments: the lower limit must be less than the higher",
        )

    setattr(ax, f"{ax_upper}Lim", [lo, hi])
    lims = getattr(ax, f"{ax_upper}Lim")
    return float(lims[0]), float(lims[1])


def lx(*args: Any) -> Optional[Tuple[float, float]]:
    """Set or get x-axis limits on the current axes."""
    expects = _caller_expects_output() if len(args) == 0 else False
    return _set_limits("X", *args, _expects_output=expects)


def ly(*args: Any) -> Optional[Tuple[float, float]]:
    """Set or get y-axis limits on the current axes."""
    expects = _caller_expects_output() if len(args) == 0 else False
    return _set_limits("Y", *args, _expects_output=expects)


def lz(*args: Any) -> Optional[Tuple[float, float]]:
    """Set or get z-axis (or colour scale on 2D area plot) limits on the current axes."""
    expects = _caller_expects_output() if len(args) == 0 else False
    return _set_limits("Z", *args, _expects_output=expects)


def lc(*args: Any) -> Optional[Tuple[float, float]]:
    """Set or get colour scale limits on the current axes."""
    expects = _caller_expects_output() if len(args) == 0 else False
    return _set_limits("C", *args, _expects_output=expects)


# ---------------------------------------------------------------------------
# Parse optional plot arguments (limits, 'name', 'axes')
# ---------------------------------------------------------------------------


def _parse_plot_args(
    args: Sequence[Any],
    new_axes: bool,
    force_current_axes: bool,
    max_limit_pairs: int = 2,
) -> Tuple[List[Tuple[float, float]], Optional[str], Optional[Axes]]:
    if force_current_axes and len(args) > 0:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "No optional arguments are permitted when overplotting on current figure",
        )

    custom_name: Optional[str] = None
    custom_axes: Optional[Axes] = None
    rem_args = list(args)

    # Check trailing keyword-value pairs ('name', val) or ('axes', val)
    i = 0
    numeric_args: List[Any] = []
    while i < len(rem_args):
        arg = rem_args[i]
        if isinstance(arg, str):
            low = arg.strip().lower()
            if low and "name".startswith(low):
                if i + 1 >= len(rem_args) or not isinstance(rem_args[i + 1], str):
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        "Expected figure name string after 'name' option",
                    )
                custom_name = rem_args[i + 1]
                i += 2
                continue
            elif low and "axes".startswith(low):
                if i + 1 >= len(rem_args) or not isinstance(rem_args[i + 1], Axes):
                    raise HoraceError(
                        "HERBERT:graphics:invalid_argument",
                        "Expected Axes handle after 'axes' option",
                    )
                custom_axes = rem_args[i + 1]
                i += 2
                continue
        numeric_args.append(arg)
        i += 1

    if not new_axes and len(numeric_args) > 0:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "Plot limits cannot be specified when overplotting",
        )

    if len(numeric_args) % 2 != 0 or len(numeric_args) // 2 > max_limit_pairs:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            "Invalid number of plot limit arguments",
        )

    limits: List[Tuple[float, float]] = []
    for k in range(0, len(numeric_args), 2):
        lo = float(numeric_args[k])
        hi = float(numeric_args[k + 1])
        if lo >= hi or math.isnan(lo) or math.isnan(hi):
            raise HoraceError(
                "HERBERT:graphics:invalid_argument",
                "Lower plot limit must be strictly less than upper plot limit",
            )
        limits.append((lo, hi))

    return limits, custom_name, custom_axes


# ---------------------------------------------------------------------------
# 1D, 2D, 3D Dataset Plotting Implementations
# ---------------------------------------------------------------------------


def _ensure_list(val: Any) -> List[Any]:
    if isinstance(val, (list, tuple, np.ndarray)):
        return list(val)
    return [val]


def plot_1d_datasets(
    datasets: Sequence[Any],
    plot_type: str,
    new_axes: bool,
    force_current_axes: bool,
    *args: Any,
    default_fig_name: str = "Herbert 1D plot",
) -> Tuple[Figure, Axes, PlotHandleArray]:
    """Core 1D dataset plotting routine (`dd`, `de`, `dh`, `dl`, `dm`, `dp`,
    `pd`..`pp`, `pdoc`..`ppoc`).
    """
    nw = len(datasets)
    if nw == 0:
        raise HoraceError("HERBERT:graphics:invalid_argument", "Empty dataset array")
    maxspec = graph_config.get("maxspec_1D")
    if nw > maxspec:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            f"Number of 1D datasets ({nw}) exceeds maxspec_1D ({maxspec})",
        )

    limits, custom_name, custom_axes = _parse_plot_args(
        args, new_axes=new_axes, force_current_axes=force_current_axes, max_limit_pairs=2
    )

    fig_h, ax_h, created_new_axes = _get_or_create_target_figure_and_axes(
        default_fig_name,
        new_axes=new_axes,
        force_current_axes=force_current_axes,
        custom_name=custom_name,
        custom_axes=custom_axes,
    )

    if created_new_axes:
        ax_h._horace_plot_dim = 1
        ax_h.XScale = graph_config.get("XScale")
        ax_h.YScale = graph_config.get("YScale")
        xlab, slab = datasets[0].make_label()
        ax_h.set_xlabel("\n".join(xlab) if isinstance(xlab, list) else str(xlab))
        ax_h.set_ylabel("\n".join(slab) if isinstance(slab, list) else str(slab))
        if datasets[0].title:
            ax_h.set_title(
                "\n".join(datasets[0].title)
                if isinstance(datasets[0].title, list)
                else str(datasets[0].title)
            )

    colors = _ensure_list(graph_config.get("colors"))
    lstyles = _ensure_list(graph_config.get("line_styles"))
    lwidths = [float(x) for x in _ensure_list(graph_config.get("line_widths"))]
    mtypes = _ensure_list(graph_config.get("marker_types"))
    msizes = [float(x) for x in _ensure_list(graph_config.get("marker_sizes"))]
    cycle = graph_config.get("color_cycle")

    ptype = plot_type.lower()
    if ptype == "d":
        icol, ils, ilw, imt, ims = property_index(
            nw, cycle, len(colors), len(lstyles), len(lwidths), len(mtypes), len(msizes)
        )
    elif ptype == "p":
        icol, ilw, imt, ims = property_index(
            nw, cycle, len(colors), len(lwidths), len(mtypes), len(msizes)
        )
    elif ptype == "e":
        icol, ilw = property_index(nw, cycle, len(colors), len(lwidths))
    elif ptype in ("h", "l"):
        icol, ils, ilw = property_index(
            nw, cycle, len(colors), len(lstyles), len(lwidths)
        )
    elif ptype == "m":
        icol, imt, ims = property_index(
            nw, cycle, len(colors), len(mtypes), len(msizes)
        )
    else:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument", f"Unknown 1D plot type '{plot_type}'"
        )

    for i, w in enumerate(datasets):
        x = np.asarray(w.x, dtype=float).ravel()
        y = np.asarray(w.signal, dtype=float).ravel()
        e = np.asarray(w.error, dtype=float).ravel()
        ny = y.size
        nx = x.size

        col_code = colors[icol[i]]
        rgb = _color_to_rgb(col_code)

        if ptype == "h":
            ls = lstyles[ils[i]]
            lw = lwidths[ilw[i]]
            if ny == 0:
                xb = np.array([], dtype=float)
                yb = np.array([], dtype=float)
            else:
                if nx == ny + 1:
                    xb = np.empty(2 * ny, dtype=float)
                    xb[0] = x[0]
                    if ny > 1:
                        xb[1 : 2 * ny - 1 : 2] = x[1 : nx - 1]
                        xb[2 : 2 * ny - 1 : 2] = x[1 : nx - 1]
                    xb[-1] = x[-1]
                else:
                    xb = np.empty(2 * ny, dtype=float)
                    if nx > 1:
                        xb[0] = x[0] - 0.5 * (x[1] - x[0])
                        mid = 0.5 * (x[1:] + x[:-1])
                        xb[1 : 2 * ny - 1 : 2] = mid
                        xb[2 : 2 * ny - 1 : 2] = mid
                        xb[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
                    else:
                        xb[0] = x[0] - 0.5
                        xb[1] = x[0] + 0.5
                yb = np.empty(2 * ny, dtype=float)
                yb[0::2] = y
                yb[1::2] = y

            (line_art,) = ax_h.plot(xb, yb, linestyle=ls, linewidth=lw, color=rgb)
            ax_h._tracked_mpl_artists.add(id(line_art))
            ph = PlotHandle(
                artist=line_art,
                kind="line",
                color=rgb,
                line_style=ls,
                line_width=lw,
                marker="none",
                marker_size=6.0,
                xdata=xb,
                ydata=yb,
            )
            ax_h._horace_plot_handles.append(ph)
        else:
            xtemp = 0.5 * (x[:-1] + x[1:]) if (nx == ny + 1 and ny > 0) else x
            if ptype == "d":
                ls = lstyles[ils[i]]
                lw = lwidths[ilw[i]]
                mt = mtypes[imt[i]]
                ms = msizes[ims[i]]
                yerr = e
            elif ptype == "p":
                ls = "none"
                lw = lwidths[ilw[i]]
                mt = mtypes[imt[i]]
                ms = msizes[ims[i]]
                yerr = e
            elif ptype == "e":
                ls = "none"
                lw = lwidths[ilw[i]]
                mt = "none"
                ms = 6.0
                yerr = e
            elif ptype == "l":
                ls = lstyles[ils[i]]
                lw = lwidths[ilw[i]]
                mt = "none"
                ms = 6.0
                yerr = None
            elif ptype == "m":
                ls = "none"
                lw = 0.5
                mt = mtypes[imt[i]]
                ms = msizes[ims[i]]
                yerr = None

            mpl_marker = "None" if mt == "none" else mt
            mpl_ls = "None" if ls == "none" else ls
            (line_art,) = ax_h.plot(
                xtemp,
                y,
                linestyle=mpl_ls,
                linewidth=lw,
                 color=rgb,
                marker=mpl_marker,
                markersize=ms,
            )
            ax_h._tracked_mpl_artists.add(id(line_art))
            ph = PlotHandle(
                artist=line_art,
                kind="errorbar" if yerr is not None else "line",
                color=rgb,
                line_style=ls,
                line_width=lw,
                marker=mt,
                marker_size=ms,
                xdata=xtemp,
                ydata=y,
                yerr=yerr,
            )
            ax_h._horace_plot_handles.append(ph)

    _recompute_ax_auto_limits(ax_h)
    if len(limits) >= 1:
        ax_h.XLim = list(limits[0])
    if len(limits) >= 2:
        ax_h.YLim = list(limits[1])

    # Attach source data to figure handle
    datasets[0]._attach_source_to_fig(fig_h, datasets, created_new_axes)

    return fig_h, ax_h, _get_ax_plot_handles_lifo(ax_h)


def _bin_boundaries_for_plot(x: np.ndarray, n_sig: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    if x.size == n_sig + 1:
        return x
    if x.size == 0:
        return np.array([0.0, 1.0])
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5])
    dx = np.diff(x)
    b = np.empty(x.size + 1, dtype=float)
    b[0] = x[0] - 0.5 * dx[0]
    b[1:-1] = 0.5 * (x[:-1] + x[1:])
    b[-1] = x[-1] + 0.5 * dx[-1]
    return b


def plot_2d_datasets(
    datasets: Sequence[Any],
    plot_type: str,
    new_axes: bool,
    force_current_axes: bool,
    *args: Any,
    color_datasets: Optional[Sequence[Any]] = None,
) -> Tuple[Figure, Axes, PlotHandleArray]:
    """Core 2D dataset plotting routine (`da`, `ds`, `ds2`, `pa`, `ps`, `ps2`,
    `paoc`, `psoc`, `ps2oc`).
    """
    nw = len(datasets)
    if nw == 0:
        raise HoraceError("HERBERT:graphics:invalid_argument", "Empty dataset array")
    maxspec = graph_config.get("maxspec_2D")
    if nw > maxspec:
        raise HoraceError(
            "HERBERT:graphics:invalid_argument",
            f"Number of 2D datasets ({nw}) exceeds maxspec_2D ({maxspec})",
        )

    ptype = plot_type.lower()  # 'a', 's', 's2'
    default_fig_name = "Herbert area plot" if ptype == "a" else "Herbert surface plot"

    limits, custom_name, custom_axes = _parse_plot_args(
        args, new_axes=new_axes, force_current_axes=force_current_axes, max_limit_pairs=3
    )

    fig_h, ax_h, created_new_axes = _get_or_create_target_figure_and_axes(
        default_fig_name,
        new_axes=new_axes,
        force_current_axes=force_current_axes,
        custom_name=custom_name,
        custom_axes=custom_axes,
    )

    if created_new_axes:
        ax_h._horace_plot_dim = 2
        ax_h.XScale = graph_config.get("XScale")
        ax_h.YScale = graph_config.get("YScale")
        ax_h.ZScale = graph_config.get("ZScale")
        xlab, ylab, _ = datasets[0].make_label()
        ax_h.set_xlabel("\n".join(xlab) if isinstance(xlab, list) else str(xlab))
        ax_h.set_ylabel("\n".join(ylab) if isinstance(ylab, list) else str(ylab))
        if datasets[0].title:
            ax_h.set_title(
                "\n".join(datasets[0].title)
                if isinstance(datasets[0].title, list)
                else str(datasets[0].title)
            )

    ax_h._horace_has_c = True
    if ptype in ("s", "s2"):
        ax_h._horace_has_z = True

    for i, w in enumerate(datasets):
        sig = np.asarray(w.signal, dtype=float)
        nx_s, ny_s = sig.shape
        x = np.asarray(w.x, dtype=float).ravel()
        y = np.asarray(w.y, dtype=float).ravel()

        if ptype == "a":
            xb = _bin_boundaries_for_plot(x, nx_s)
            yb = _bin_boundaries_for_plot(y, ny_s)
            mesh = ax_h.pcolormesh(xb, yb, sig.T, shading="flat")
            ax_h._tracked_mpl_artists.add(id(mesh))
            ph = PlotHandle(
                artist=mesh,
                kind="patch",
                xdata=xb,
                ydata=yb,
                cdata=sig,
            )
            ax_h._horace_plot_handles.append(ph)
        else:
            # Surface plot ('s' or 's2')
            xc = 0.5 * (x[:-1] + x[1:]) if x.size == nx_s + 1 else x
            yc = 0.5 * (y[:-1] + y[1:]) if y.size == ny_s + 1 else y
            if ptype == "s":
                cdata = sig
            else:
                if color_datasets is not None:
                    cw = color_datasets[i] if len(color_datasets) > 1 else color_datasets[0]
                    cdata = (
                        np.asarray(cw.signal, dtype=float)
                        if hasattr(cw, "signal")
                        else np.asarray(cw, dtype=float)
                    )
                else:
                    cdata = np.asarray(w.error, dtype=float)
            mesh = ax_h.pcolormesh(
                _bin_boundaries_for_plot(xc, nx_s),
                _bin_boundaries_for_plot(yc, ny_s),
                cdata.T,
                shading="flat",
            )
            ax_h._tracked_mpl_artists.add(id(mesh))
            ph = PlotHandle(
                artist=mesh,
                kind="surface",
                xdata=xc,
                ydata=yc,
                zdata=sig,
                cdata=cdata,
            )
            ax_h._horace_plot_handles.append(ph)

    _recompute_ax_auto_limits(ax_h)
    if len(limits) >= 1:
        ax_h.XLim = list(limits[0])
    if len(limits) >= 2:
        ax_h.YLim = list(limits[1])
    if len(limits) >= 3:
        if ptype == "a":
            ax_h.CLim = list(limits[2])
        else:
            ax_h.ZLim = list(limits[2])
            if ptype == "s":
                ax_h.CLim = list(limits[2])

    datasets[0]._attach_source_to_fig(fig_h, datasets, created_new_axes)

    return fig_h, ax_h, _get_ax_plot_handles_lifo(ax_h)


def _check_uniform_axis(x: np.ndarray, is_hist: bool, tol: float = 1e-4) -> bool:
    x = np.asarray(x, dtype=float).ravel()
    min_len = 2 if is_hist else 1
    if x.size < min_len:
        return False
    if x.size <= 2:
        return True
    dx = np.diff(x)
    dx_ref = (x[-1] - x[0]) / (x.size - 1)
    if dx_ref == 0:
        return False
    return bool(np.all(np.abs((dx - dx_ref) / dx_ref) <= tol))


def plot_3d_dataset(
    datasets: Sequence[Any],
    overview: bool = False,
    *args: Any,
) -> Tuple[Figure, Axes, Dict[str, Any]]:
    """Core 3D dataset plotting routine (`sliceomatic`, `sliceomatic_overview`)."""
    if len(datasets) != 1:
        raise HoraceError(
            "HERBERT:IX_dataset_3d:invalid_argument",
            "Sliceomatic only works for a single 3D dataset, not an array of datasets",
        )
    w = datasets[0]
    sig = np.asarray(w.signal, dtype=float)
    if sig.size == 0 or any(s == 0 for s in sig.shape):
        raise HoraceError(
            "HERBERT:IX_dataset_3d:invalid_argument",
            "Cannot plot an empty 3D dataset",
        )

    for iax, x_arr in enumerate((w.x, w.y, w.z), start=1):
        if not _check_uniform_axis(x_arr, w.ishistogram(iax)):
            raise HoraceError(
                "HERBERT:IX_dataset_3d:invalid_argument",
                "Data points must be equally spaced for sliceomatic.",
            )

    fig_h, ax_h, _ = _get_or_create_target_figure_and_axes(
        "Sliceomatic",
        new_axes=True,
        force_current_axes=False,
    )
    ax_h._horace_plot_dim = 3
    ax_h._horace_has_z = True
    ax_h._horace_has_c = True

    # Plot middle Z-slice onto ax_h as representative 2D view
    mid_z = sig.shape[2] // 2
    xb = _bin_boundaries_for_plot(w.x, sig.shape[0])
    yb = _bin_boundaries_for_plot(w.y, sig.shape[1])
    mesh = ax_h.pcolormesh(xb, yb, sig[:, :, mid_z].T, shading="flat")
    ax_h._tracked_mpl_artists.add(id(mesh))
    zc = 0.5 * (w.z[:-1] + w.z[1:]) if w.ishistogram(3) else np.asarray(w.z, dtype=float)
    ph = PlotHandle(
        artist=mesh,
        kind="surface",
        xdata=xb,
        ydata=yb,
        zdata=zc,
        cdata=sig,
    )
    ax_h._horace_plot_handles.append(ph)
    _recompute_ax_auto_limits(ax_h)

    w._attach_source_to_fig(fig_h, [w], True)

    plot_h_struct: Dict[str, Any] = {
        "figure": fig_h,
        "axes": ax_h,
        "overview": overview,
        "surface": ph,
    }
    return fig_h, ax_h, plot_h_struct


def spaghetti_plot(
    wdisp: Union[Any, Sequence[Any]],
    *args: Any,
) -> Tuple[Any, List[Any], Any, Any, Any]:
    """Plot a sequence of 2D dispersion datasets side-by-side (`spaghetti_plot`)."""
    if isinstance(wdisp, DataOpArray):
        w_list = list(wdisp.flat)
        w_out = wdisp.copy()
    elif isinstance(wdisp, (list, tuple, np.ndarray)):
        w_list = list(np.asarray(wdisp, dtype=object).flat)
        w_out = DataOpArray([w.copy() for w in w_list])
    else:
        w_list = [wdisp]
        w_out = wdisp.copy()

    noplot = False
    labels: Optional[List[str]] = None
    i = 0
    while i < len(args):
        arg = args[i]
        if isinstance(arg, str):
            low = arg.strip().lower().lstrip("-")
            if low == "noplot":
                noplot = True
                i += 1
                continue
            elif low.startswith("lab"):
                if i + 1 >= len(args):
                    raise HoraceError(
                        "HERBERT:spaghetti_plot:invalid_argument",
                        "Expected cell array of labels after 'labels' option",
                    )
                labels = [str(x) for x in args[i + 1]]
                i += 2
                continue
        i += 1

    if noplot:
        return w_out, [], [], [], []

    fig_h, ax_h, _ = _get_or_create_target_figure_and_axes(
        "Spaghetti plot",
        new_axes=True,
        force_current_axes=False,
    )
    ax_h._horace_plot_dim = 2
    ax_h._horace_has_c = True

    x_offset = 0.0
    for w in w_list:
        sig = np.asarray(w.signal, dtype=float)
        nx_s, ny_s = sig.shape
        xb = _bin_boundaries_for_plot(w.x, nx_s)
        yb = _bin_boundaries_for_plot(w.y, ny_s)
        span_x = float(xb[-1] - xb[0]) if xb.size > 1 else 1.0
        xb_shifted = (xb - xb[0]) + x_offset
        x_offset += span_x
        mesh = ax_h.pcolormesh(xb_shifted, yb, sig.T, shading="flat")
        ax_h._tracked_mpl_artists.add(id(mesh))
        ph = PlotHandle(
            artist=mesh,
            kind="patch",
            xdata=xb_shifted,
            ydata=yb,
            cdata=sig,
        )
        ax_h._horace_plot_handles.append(ph)

    _recompute_ax_auto_limits(ax_h)

    if labels is not None:
        ax_h.XTickLabel = labels
    else:
        default_labs: List[str] = []
        for w in w_list:
            if w.title and len(w.title) > 0:
                default_labs.append(str(w.title[0]))
            else:
                default_labs.append("")
        ax_h.XTickLabel = default_labs

    return w_out, [], fig_h, ax_h, _get_ax_plot_handles_lifo(ax_h)


__all__ = [
    "PlotHandle",
    "PlotHandleArray",
    "acolor",
    "aline",
    "amark",
    "caxis",
    "clearfigs",
    "figure",
    "gca",
    "gcf",
    "genieplot",
    "is_genie_figure",
    "isgraphics",
    "keep_figure",
    "lc",
    "linx",
    "liny",
    "linz",
    "logx",
    "logy",
    "logz",
    "lx",
    "ly",
    "lz",
    "make_current",
    "plot_1d_datasets",
    "plot_2d_datasets",
    "plot_3d_dataset",
    "property_index",
    "spaghetti_plot",
    "src",
    "xlim",
    "ylim",
    "zlim",
]
