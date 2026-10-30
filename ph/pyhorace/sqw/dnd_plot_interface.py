"""Interface for plotting DnD objects (`d1d`, `d2d`, `d3d`).

Ported from Herbert MATLAB `dnd_plot_interface`
(`herbert_core/graphics/plot_interfaces/dnd_plot_interface.m`).
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.data_plot_interface import (
    _dispatch_plot_method,
    data_plot_interface as _BaseDataPlotInterface,
)
from pyhorace.data.graphics import aspect, genieplot
from pyhorace.data.ix_dataset_1d import IX_dataset_1d
from pyhorace.data.ix_dataset_2d import IX_dataset_2d
from pyhorace.data.ix_dataset_3d import IX_dataset_3d


def _extract_input_items(self_obj: Any, kwargs: dict[str, Any]) -> Any:
    """Return `_dataset_array` if provided by DataOpArray dispatch, else `self_obj`."""
    dataset_array = kwargs.pop("_dataset_array", None)
    if dataset_array is not None:
        return DataOpArray(dataset_array)
    return self_obj


def _flatten_objects(w: Any) -> List[Any]:
    """Flatten scalar, DataOpArray, list, tuple, or ndarray of objects into a list."""
    if isinstance(w, DataOpArray):
        return [item for item in w.flat]
    if isinstance(w, (list, tuple)):
        out: List[Any] = []
        for item in w:
            out.extend(_flatten_objects(item))
        return out
    if isinstance(w, np.ndarray) and w.dtype == object:
        return [item for item in w.ravel()]
    return [w]


def _with_default_fig_name(fig_name: str, fn: Callable[[], Any]) -> Any:
    """Temporarily set `genieplot.default_fig_name` while executing `fn`."""
    gp = genieplot.instance()
    prev_name = gp.default_fig_name
    gp.default_fig_name = fig_name
    try:
        return fn()
    finally:
        gp.default_fig_name = prev_name


def _convert_2d_signal_arg(arg: Any) -> Any:
    """Convert second dataset argument in ds2/ps2/ps2oc to IX_dataset_2d if it is a DnD/sqw."""
    from pyhorace.sqw.dnd_base import DnDBase

    items = _flatten_objects(arg)
    if items and all(hasattr(x, "IX_dataset_2d") and not isinstance(x, IX_dataset_2d) for x in items):
        for x in items:
            holder = getattr(x, "_obj_holder", None)
            if holder is None or isinstance(holder, DnDBase):
                x._obj_holder = x
        if len(items) == 1:
            return IX_dataset_2d(items[0])
        return IX_dataset_2d(DataOpArray(items))
    return arg


class dnd_plot_interface(_BaseDataPlotInterface):
    """Plot interface for DnD (1D, 2D, and 3D) objects.

    Mirrors `herbert_core/graphics/plot_interfaces/dnd_plot_interface.m`.
    """

    def data_plot_titles(self) -> Tuple[Any, ...]:
        """Return `(title_main, title_pax, title_iax, ...)` from `self.axes.data_plot_titles`."""
        if hasattr(self, "axes") and hasattr(self.axes, "data_plot_titles"):
            return self.axes.data_plot_titles(self)
        return ("", [], [])

    # ------------------------------------------------------------------
    # Static validation & aspect helpers
    # ------------------------------------------------------------------
    @staticmethod
    def check_data(w: Any, ndim: int, scalar_only: bool, func_name: str) -> Any:
        """Check dimensionality and scalar constraint, and attach `_obj_holder`."""
        w_list = _flatten_objects(w)
        if not w_list or (scalar_only and len(w_list) > 1) or any(
            int(x.dimensions()) != int(ndim) for x in w_list
        ):
            if scalar_only:
                string = f"a single {ndim}D object"
            else:
                string = f"{ndim}D object(s)"
            raise HoraceError(
                "HORACE:graphics:invalid_argument",
                f"Plot method '{func_name}' only works for {string}",
            )
        from pyhorace.sqw.dnd_base import DnDBase

        for x in w_list:
            holder = getattr(x, "_obj_holder", None)
            if holder is None or isinstance(holder, DnDBase):
                x._obj_holder = x
        if len(w_list) == 1:
            return w_list[0]
        return DataOpArray(w_list)

    @staticmethod
    def strip_aspect_option(*args: Any) -> Tuple[Tuple[Any, ...], bool]:
        """Strip `'-aspect'` option from arguments and return `(remaining_args, adjust_aspect)`."""
        adjust_aspect = True
        adjust_present = False
        remaining: List[Any] = []
        target = "-aspect"
        for arg in args:
            if isinstance(arg, str) and len(arg) > 0 and target.startswith(arg.lower()):
                if not adjust_present:
                    adjust_present = True
                else:
                    raise HoraceError(
                        "HORACE:graphics:invalid_argument",
                        "Input key: '-aspect' appears at least twice in the argument list",
                    )
            else:
                remaining.append(arg)
        if adjust_present:
            adjust_aspect = True
        return tuple(remaining), adjust_aspect

    @staticmethod
    def adjust_aspect_ratio(w: Any) -> None:
        """Adjust plot aspect ratio for 2D and 3D DnD datasets if required."""
        w_list = _flatten_objects(w)
        if not w_list:
            return
        ndims = int(w_list[0].dimensions())
        if ndims not in (2, 3):
            return
        if not all(bool(getattr(x.axes, "changes_aspect_ratio", True)) for x in w_list):
            return

        ulen_ref: np.ndarray | None = None
        is_energy_axis_ref: np.ndarray | None = None
        for idx, w_i in enumerate(w_list):
            pax_dax = [int(w_i.pax[int(d) - 1]) for d in w_i.dax]
            ulen = np.array([float(w_i.axes.ulen[p_idx - 1]) for p_idx in pax_dax], dtype=float)
            is_energy_axis = np.array([p_idx == 4 for p_idx in pax_dax], dtype=bool)
            if idx == 0:
                ulen_ref = ulen.copy()
                is_energy_axis_ref = is_energy_axis.copy()
            else:
                assert ulen_ref is not None and is_energy_axis_ref is not None
                if not np.array_equal(is_energy_axis, is_energy_axis_ref) or float(
                    np.max(np.abs((ulen - ulen_ref) / ulen_ref))
                ) > 1e-8:
                    return

        if ulen_ref is not None and is_energy_axis_ref is not None:
            ulen_plot = ulen_ref.copy()
            ulen_plot[is_energy_axis_ref] = 0.0
            aspect(ulen_plot)

    # ------------------------------------------------------------------
    # 1D plotting methods
    # ------------------------------------------------------------------
    def _plot_1d(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 1, False, method_name)
        return _with_default_fig_name(
            "Horace 1D plot",
            lambda: _dispatch_plot_method(method_name, IX_dataset_1d(w_checked), *args, **kwargs),
        )

    def dd(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("dd", *args, **kwargs)

    def de(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("de", *args, **kwargs)

    def dh(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("dh", *args, **kwargs)

    def dl(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("dl", *args, **kwargs)

    def dm(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("dm", *args, **kwargs)

    def dp(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("dp", *args, **kwargs)

    def pd(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pd", *args, **kwargs)

    def pdoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pdoc", *args, **kwargs)

    def pe(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pe", *args, **kwargs)

    def peoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("peoc", *args, **kwargs)

    def ph(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("ph", *args, **kwargs)

    def phoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("phoc", *args, **kwargs)

    def pl(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pl", *args, **kwargs)

    def ploc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("ploc", *args, **kwargs)

    def pm(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pm", *args, **kwargs)

    def pmoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pmoc", *args, **kwargs)

    def pp(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("pp", *args, **kwargs)

    def ppoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._plot_1d("ppoc", *args, **kwargs)

    # ------------------------------------------------------------------
    # 2D plotting methods
    # ------------------------------------------------------------------
    def da(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "da")
        rem_args, adjust_aspect = dnd_plot_interface.strip_aspect_option(*args)

        def _do_plot() -> Any:
            res = _dispatch_plot_method("da", IX_dataset_2d(w_checked), *rem_args, **kwargs)
            if adjust_aspect:
                dnd_plot_interface.adjust_aspect_ratio(w_checked)
            return res

        return _with_default_fig_name("Horace area plot", _do_plot)

    def ds(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "ds")
        rem_args, adjust_aspect = dnd_plot_interface.strip_aspect_option(*args)

        def _do_plot() -> Any:
            res = _dispatch_plot_method("ds", IX_dataset_2d(w_checked), *rem_args, **kwargs)
            if adjust_aspect:
                dnd_plot_interface.adjust_aspect_ratio(w_checked)
            return res

        return _with_default_fig_name("Horace surface plot", _do_plot)

    def ds2(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "ds2")
        rem_args, adjust_aspect = dnd_plot_interface.strip_aspect_option(*args)
        if len(rem_args) > 0:
            rem_args = (_convert_2d_signal_arg(rem_args[0]), *rem_args[1:])

        def _do_plot() -> Any:
            res = _dispatch_plot_method("ds2", IX_dataset_2d(w_checked), *rem_args, **kwargs)
            if adjust_aspect:
                dnd_plot_interface.adjust_aspect_ratio(w_checked)
            return res

        return _with_default_fig_name("Horace surface plot", _do_plot)

    def pa(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "pa")
        return _with_default_fig_name(
            "Horace area plot",
            lambda: _dispatch_plot_method("pa", IX_dataset_2d(w_checked), *args, **kwargs),
        )

    def paoc(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "poc")
        return _with_default_fig_name(
            "Horace area plot",
            lambda: _dispatch_plot_method("paoc", IX_dataset_2d(w_checked), *args, **kwargs),
        )

    def ps(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "ps")
        return _with_default_fig_name(
            "Horace surface plot",
            lambda: _dispatch_plot_method("ps", IX_dataset_2d(w_checked), *args, **kwargs),
        )

    def psoc(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "psoc")
        return _with_default_fig_name(
            "Horace surface plot",
            lambda: _dispatch_plot_method("psoc", IX_dataset_2d(w_checked), *args, **kwargs),
        )

    def ps2(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "ps2")
        conv_args = (_convert_2d_signal_arg(args[0]), *args[1:]) if len(args) > 0 else args
        return _with_default_fig_name(
            "Horace surface plot",
            lambda: _dispatch_plot_method("ps2", IX_dataset_2d(w_checked), *conv_args, **kwargs),
        )

    def ps2oc(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 2, False, "ps2oc")
        conv_args = (_convert_2d_signal_arg(args[0]), *args[1:]) if len(args) > 0 else args
        return _with_default_fig_name(
            "Horace surface plot",
            lambda: _dispatch_plot_method("ps2oc", IX_dataset_2d(w_checked), *conv_args, **kwargs),
        )

    # ------------------------------------------------------------------
    # 3D plotting methods
    # ------------------------------------------------------------------
    def sliceomatic(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 3, True, "sliceomatic")
        rem_args, adjust_aspect = dnd_plot_interface.strip_aspect_option(*args)
        pax_dax = [int(w_checked.pax[int(d) - 1]) for d in w_checked.dax]
        ulabel = [w_checked.label[p_idx - 1] for p_idx in pax_dax]
        res = _dispatch_plot_method(
            "sliceomatic",
            IX_dataset_3d(w_checked),
            "x_axis",
            ulabel[0],
            "y_axis",
            ulabel[1],
            "z_axis",
            ulabel[2],
            "name",
            "Horace sliceomatic",
            *rem_args,
            **kwargs,
        )
        if adjust_aspect:
            dnd_plot_interface.adjust_aspect_ratio(w_checked)
        return res

    def sliceomatic_overview(self, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        w_checked = dnd_plot_interface.check_data(w_in, 3, True, "sliceomatic")
        rem_args, adjust_aspect = dnd_plot_interface.strip_aspect_option(*args)
        pax_dax = [int(w_checked.pax[int(d) - 1]) for d in w_checked.dax]
        ulabel = [w_checked.label[p_idx - 1] for p_idx in pax_dax]
        res = _dispatch_plot_method(
            "sliceomatic_overview",
            IX_dataset_3d(w_checked),
            "x_axis",
            ulabel[0],
            "y_axis",
            ulabel[1],
            "z_axis",
            ulabel[2],
            "name",
            "Horace sliceomatic",
            *rem_args,
            **kwargs,
        )
        if adjust_aspect:
            dnd_plot_interface.adjust_aspect_ratio(w_checked)
        return res


data_plot_interface = _BaseDataPlotInterface


def _bind_array_plot_methods() -> None:
    """Ensure DataOpArray dispatches plot calls through `_dispatch_plot_method`."""
    for name in (
        "dd",
        "de",
        "dh",
        "dl",
        "dm",
        "dp",
        "pd",
        "pdoc",
        "pe",
        "peoc",
        "ph",
        "phoc",
        "pl",
        "ploc",
        "pm",
        "pmoc",
        "pp",
        "ppoc",
        "da",
        "ds",
        "ds2",
        "pa",
        "paoc",
        "ps",
        "ps2",
        "psoc",
        "ps2oc",
        "sliceomatic",
        "sliceomatic_overview",
        "plot",
        "plotover",
    ):
        def _make_method(m_name: str) -> Callable[..., Any]:
            def _method(self_arr: Any, *args: Any, **kwargs: Any) -> Any:
                return _dispatch_plot_method(m_name, self_arr, *args, **kwargs)
            return _method

        setattr(DataOpArray, name, _make_method(name))


_bind_array_plot_methods()


__all__ = ["data_plot_interface", "dnd_plot_interface"]
