"""Abstract plotting interface for IX_dataset_nd and dnd classes.

Translated from ``herbert_core/graphics/plot_interfaces/data_plot_interface.m``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.graphics import HoraceError


class data_plot_interface(ABC):
    """Abstract class containing plot interface which works with ``IX_dataset_nd``
    and ``dnd`` classes.

    Subclasses override the methods appropriate for their dimensionality (1D,
    2D, or 3D). Calling an unimplemented plotting method raises a
    ``ValueError`` with ID ``HORACE:<class_name>:invalid_argument``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._obj_holder: Any = None

    @abstractmethod
    def dimensions(self) -> tuple[int, Any]:
        """Return ``(nd, sz)`` dimensionality and sizes of the object."""

    # ------------------------------------------------------------------
    # 1D Plot Methods
    # ------------------------------------------------------------------

    def dd(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw markers, errorbars and lines for 1D dataset(s)."""
        return self._throw_unavailable("dd", *args, **kwargs)

    def de(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw errorbars for 1D dataset(s)."""
        return self._throw_unavailable("de", *args, **kwargs)

    def dh(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw histogram for 1D dataset(s)."""
        return self._throw_unavailable("dh", *args, **kwargs)

    def dl(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw lines for 1D dataset(s)."""
        return self._throw_unavailable("dl", *args, **kwargs)

    def dm(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw markers for 1D dataset(s)."""
        return self._throw_unavailable("dm", *args, **kwargs)

    def dp(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw markers and errorbars for 1D dataset(s)."""
        return self._throw_unavailable("dp", *args, **kwargs)

    # ------------------------------------------------------------------
    # 1D Overplot Methods
    # ------------------------------------------------------------------

    def pd(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers, errorbars and lines on current 1D plot."""
        return self._throw_unavailable("pd", *args, **kwargs)

    def pdoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers, errorbars and lines on current figure/axes."""
        return self._throw_unavailable("pdoc", *args, **kwargs)

    def pe(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot errorbars on current 1D plot."""
        return self._throw_unavailable("pe", *args, **kwargs)

    def peoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot errorbars on current figure/axes."""
        return self._throw_unavailable("peoc", *args, **kwargs)

    def ph(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot histogram on current 1D plot."""
        return self._throw_unavailable("ph", *args, **kwargs)

    def phoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot histogram on current figure/axes."""
        return self._throw_unavailable("phoc", *args, **kwargs)

    def pl(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot lines on current 1D plot."""
        return self._throw_unavailable("pl", *args, **kwargs)

    def ploc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot lines on current figure/axes."""
        return self._throw_unavailable("ploc", *args, **kwargs)

    def pm(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers on current 1D plot."""
        return self._throw_unavailable("pm", *args, **kwargs)

    def pmoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers on current figure/axes."""
        return self._throw_unavailable("pmoc", *args, **kwargs)

    def pp(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers and errorbars on current 1D plot."""
        return self._throw_unavailable("pp", *args, **kwargs)

    def ppoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot markers and errorbars on current figure/axes."""
        return self._throw_unavailable("ppoc", *args, **kwargs)

    # ------------------------------------------------------------------
    # 2D Plot Methods
    # ------------------------------------------------------------------

    def da(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw area (patch) plot of 2D dataset(s)."""
        return self._throw_unavailable("da", *args, **kwargs)

    def ds(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw surface plot of 2D dataset(s)."""
        return self._throw_unavailable("ds", *args, **kwargs)

    def ds2(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Draw surface plot of 2D dataset(s) with separate colour scale."""
        return self._throw_unavailable("ds2", *args, **kwargs)

    # ------------------------------------------------------------------
    # 2D Overplot Methods
    # ------------------------------------------------------------------

    def pa(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot area (patch) plot on current 2D area plot."""
        return self._throw_unavailable("pa", *args, **kwargs)

    def paoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot area (patch) plot on current figure/axes."""
        return self._throw_unavailable("paoc", *args, **kwargs)

    def ps(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot surface plot on current 2D surface plot."""
        return self._throw_unavailable("ps", *args, **kwargs)

    def psoc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot surface plot on current figure/axes."""
        return self._throw_unavailable("psoc", *args, **kwargs)

    def ps2(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot surface plot with separate colour scale on current 2D surface plot."""
        return self._throw_unavailable("ps2", *args, **kwargs)

    def ps2oc(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot surface plot with separate colour scale on current figure/axes."""
        return self._throw_unavailable("ps2oc", *args, **kwargs)

    # ------------------------------------------------------------------
    # 3D Plot Methods
    # ------------------------------------------------------------------

    def sliceomatic(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Plot 3D dataset(s) using Sliceomatic."""
        return self._throw_unavailable("sliceomatic", *args, **kwargs)

    def sliceomatic_overview(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Plot 3D dataset(s) using Sliceomatic overview."""
        return self._throw_unavailable("sliceomatic_overview", *args, **kwargs)

    # ------------------------------------------------------------------
    # Generic ND Plot Methods
    # ------------------------------------------------------------------

    def plot(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Plot 1D, 2D or 3D dataset(s) or array of datasets."""
        res = self.dimensions()
        nd = res[0] if isinstance(res, tuple) else int(res)
        if nd == 1:
            return self.dp(*args, **kwargs)
        if nd == 2:
            return self.da(*args, **kwargs)
        if nd == 3:
            return self.sliceomatic(*args, **kwargs)
        raise HoraceError(
            "HORACE:graphics:invalid_argument",
            f"Method 'plot' is not available for {nd}-dimensional objects",
        )

    def plotover(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Overplot 1D or 2D dataset(s) or array of datasets."""
        res = self.dimensions()
        nd = res[0] if isinstance(res, tuple) else int(res)
        if nd == 1:
            return self.pp(*args, **kwargs)
        if nd == 2:
            return self.pa(*args, **kwargs)
        raise HoraceError(
            "HORACE:graphics:invalid_argument",
            f"Method 'plotover' is not available for {nd}-dimensional objects",
        )

    # ------------------------------------------------------------------
    # Protected / Internal Helpers
    # ------------------------------------------------------------------

    def add_source_data_to_fig_handle(
        self,
        fig_h: Any,
        new_axes: bool,
        datasets: list[Any] | None = None,
    ) -> Any:
        """Attach source dataset(s) to ``fig_h.UserData``."""
        items = datasets if datasets is not None else [self]
        src_list: list[Any] = []
        for item in items:
            holder = getattr(item, "_obj_holder", None)
            src_list.append(holder if holder is not None else item)

        existing = getattr(fig_h, "UserData", None)
        if not existing or new_axes:
            fig_h.UserData = list(src_list)
        else:
            if isinstance(existing, list):
                fig_h.UserData = list(existing) + list(src_list)
            else:
                fig_h.UserData = [existing] + list(src_list)
        return fig_h

    def _attach_source_to_fig(
        self,
        fig_h: Any,
        datasets: Any,
        new_axes: bool,
    ) -> Any:
        """Attach source dataset(s) to ``fig_h.UserData`` (used by ``graphics.py``)."""
        return self.add_source_data_to_fig_handle(fig_h, new_axes, list(datasets))

    def _throw_unavailable(
        self, method: str, *args: Any, **kwargs: Any
    ) -> tuple[Any, Any, Any]:
        cls_name = type(self).__name__
        raise HoraceError(
            f"HORACE:{cls_name}:invalid_argument",
            f"Method '{method}' is not available for objects of class '{cls_name}'",
        )


def _extract_datasets(w: Any) -> list[data_plot_interface]:
    """Flatten scalar or array/list/DataOpArray of ``data_plot_interface`` objects."""
    if isinstance(w, data_plot_interface):
        return [w]
    if isinstance(w, DataOpArray):
        return [x for x in w.flat]
    if isinstance(w, np.ndarray):
        return [x for x in w.flat]
    if isinstance(w, (list, tuple)):
        out: list[data_plot_interface] = []
        for item in w:
            out.extend(_extract_datasets(item))
        return out
    raise HoraceError(
        "HERBERT:graphics:invalid_argument",
        f"Cannot plot object of type {type(w).__name__}",
    )


def _dispatch_plot_method(method_name: str, w: Any, *args: Any, **kwargs: Any) -> Any:
    """Dispatch module-level plot function to ``w`` (scalar or array of datasets)."""
    if isinstance(w, data_plot_interface):
        return getattr(w, method_name)(*args, **kwargs)
    items = _extract_datasets(w)
    if not items:
        raise HoraceError("HERBERT:graphics:invalid_argument", "Empty dataset array")
    first = items[0]
    return getattr(first, method_name)(*args, _dataset_array=items, **kwargs)


# Module-level functional plot wrappers (matching MATLAB method dispatch on arrays)
def dd(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("dd", w, *args, **kwargs)


def de(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("de", w, *args, **kwargs)


def dh(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("dh", w, *args, **kwargs)


def dl(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("dl", w, *args, **kwargs)


def dm(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("dm", w, *args, **kwargs)


def dp(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("dp", w, *args, **kwargs)


def pd(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pd", w, *args, **kwargs)


def pdoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pdoc", w, *args, **kwargs)


def pe(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pe", w, *args, **kwargs)


def peoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("peoc", w, *args, **kwargs)


def ph(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ph", w, *args, **kwargs)


def phoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("phoc", w, *args, **kwargs)


def pl(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pl", w, *args, **kwargs)


def ploc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ploc", w, *args, **kwargs)


def pm(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pm", w, *args, **kwargs)


def pmoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pmoc", w, *args, **kwargs)


def pp(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pp", w, *args, **kwargs)


def ppoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ppoc", w, *args, **kwargs)


def da(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("da", w, *args, **kwargs)


def ds(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ds", w, *args, **kwargs)


def ds2(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ds2", w, *args, **kwargs)


def mplot(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("da", w, *args, **kwargs)


def pa(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("pa", w, *args, **kwargs)


def paoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("paoc", w, *args, **kwargs)


def ps(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ps", w, *args, **kwargs)


def psoc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("psoc", w, *args, **kwargs)


def ps2(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ps2", w, *args, **kwargs)


def ps2oc(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("ps2oc", w, *args, **kwargs)


def sliceomatic(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("sliceomatic", w, *args, **kwargs)


def sliceomatic_overview(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("sliceomatic_overview", w, *args, **kwargs)


def plot(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("plot", w, *args, **kwargs)


def plotover(w: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
    return _dispatch_plot_method("plotover", w, *args, **kwargs)

