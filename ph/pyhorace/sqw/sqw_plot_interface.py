"""Interface for plotting `sqw` objects (1D, 2D, and 3D).

Ported from Herbert MATLAB `sqw_plot_interface`
(`herbert_core/graphics/plot_interfaces/sqw_plot_interface.m`).
"""

from __future__ import annotations

import copy
from typing import Any, List

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.data_plot_interface import data_plot_interface as _BaseDataPlotInterface
from pyhorace.sqw.dnd_plot_interface import _extract_input_items, _flatten_objects


class sqw_plot_interface(_BaseDataPlotInterface):
    """Plot interface for `sqw` objects.

    Converts `sqw` objects into their corresponding `DnDBase` (`d1d`, `d2d`, `d3d`)
    representation while preserving a reference to the original `sqw` object in
    `_obj_holder` so `figure.UserData` stores the source `sqw` object(s).
    """

    @staticmethod
    def convert_to_dnd(w: Any) -> Any:
        """Convert scalar or array of `sqw` objects into `DnDBase` object(s) with `_obj_holder` set."""
        w_list = _flatten_objects(w)
        if len(w_list) > 1:
            dims = [int(x.dimensions()) for x in w_list]
            if any(d != dims[0] for d in dims[1:]):
                raise HoraceError(
                    "HORACE:graphics:invalid_argument",
                    "Cannot plot an array of sqw objects with different images dimensionality",
                )
        d_list: List[Any] = []
        for x in w_list:
            data_obj = x.data
            d_i = data_obj.copy() if hasattr(data_obj, "copy") else copy.copy(data_obj)
            d_i._obj_holder = x
            d_list.append(d_i)
        if len(d_list) == 1:
            return d_list[0]
        return DataOpArray(d_list)

    def _delegate_to_dnd(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        w_in = _extract_input_items(self, kwargs)
        d = sqw_plot_interface.convert_to_dnd(w_in)
        return getattr(d, method_name)(*args, **kwargs)

    # ------------------------------------------------------------------
    # 1D plotting methods
    # ------------------------------------------------------------------
    def dd(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("dd", *args, **kwargs)

    def de(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("de", *args, **kwargs)

    def dh(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("dh", *args, **kwargs)

    def dl(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("dl", *args, **kwargs)

    def dm(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("dm", *args, **kwargs)

    def dp(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("dp", *args, **kwargs)

    def pd(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pd", *args, **kwargs)

    def pdoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pdoc", *args, **kwargs)

    def pe(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pe", *args, **kwargs)

    def peoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("peoc", *args, **kwargs)

    def ph(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ph", *args, **kwargs)

    def phoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("phoc", *args, **kwargs)

    def pl(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pl", *args, **kwargs)

    def ploc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ploc", *args, **kwargs)

    def pm(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pm", *args, **kwargs)

    def pmoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pmoc", *args, **kwargs)

    def pp(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pp", *args, **kwargs)

    def ppoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ppoc", *args, **kwargs)

    # ------------------------------------------------------------------
    # 2D plotting methods
    # ------------------------------------------------------------------
    def da(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("da", *args, **kwargs)

    def ds(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ds", *args, **kwargs)

    def ds2(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ds2", *args, **kwargs)

    def pa(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("pa", *args, **kwargs)

    def paoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("paoc", *args, **kwargs)

    def ps(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ps", *args, **kwargs)

    def ps2(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ps2", *args, **kwargs)

    def psoc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("psoc", *args, **kwargs)

    def ps2oc(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("ps2oc", *args, **kwargs)

    # ------------------------------------------------------------------
    # 3D plotting methods
    # ------------------------------------------------------------------
    def sliceomatic(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("sliceomatic", *args, **kwargs)

    def sliceomatic_overview(self, *args: Any, **kwargs: Any) -> Any:
        return self._delegate_to_dnd("sliceomatic_overview", *args, **kwargs)


__all__ = ["sqw_plot_interface"]

