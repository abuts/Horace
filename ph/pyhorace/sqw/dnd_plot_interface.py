"""Plot interfaces for DnD and SQW objects (data_plot_interface, dnd_plot_interface).

Ported from Herbert/Horace MATLAB data_plot_interface and dnd_plot_interface
(herbert_core/graphics/plot_interfaces/data_plot_interface.m and dnd_plot_interface.m).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.data_op_interface import DataOpArray


def _flatten_objs(w: Any) -> Tuple[List[Any], bool]:
    """Return flat list of objects and True if input is scalar."""
    if isinstance(w, (list, tuple)):
        flat: List[Any] = []
        for item in w:
            sub, _ = _flatten_objs(item)
            flat.extend(sub)
        return flat, len(flat) == 1
    if isinstance(w, (np.ndarray, DataOpArray)):
        items = list(w.ravel(order="F"))
        return items, w.size == 1
    return [w], True


def _format_plot_return(
    fig_h: Any = "figure_",
    axes_h: Any = "axes_",
    plot_h: Any = "plot_",
    nargout: Optional[int] = None,
) -> Any:
    if nargout is None:
        nargout = _infer_nargout(default=3)
    if nargout <= 1:
        return fig_h
    if nargout == 2:
        return fig_h, axes_h
    return fig_h, axes_h, plot_h


class data_plot_interface(ABC):
    """Abstract interface defining 1D, 2D, 3D, and generic plot methods."""

    # 1D plot methods
    @abstractmethod
    def dd(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def de(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def dh(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def dl(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def dm(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def dp(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pd(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pdoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pe(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def peoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ph(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def phoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pl(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ploc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pm(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pmoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pp(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ppoc(self, *args: Any, **kwargs: Any) -> Any: ...

    # 2D plot methods
    @abstractmethod
    def da(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ds(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ds2(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def pa(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def paoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ps(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def psoc(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ps2(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def ps2oc(self, *args: Any, **kwargs: Any) -> Any: ...

    # 3D plot methods
    @abstractmethod
    def sliceomatic(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def sliceomatic_overview(self, *args: Any, **kwargs: Any) -> Any: ...

    # ND generic plot methods
    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> Any: ...
    @abstractmethod
    def plotover(self, *args: Any, **kwargs: Any) -> Any: ...


class dnd_plot_interface(data_plot_interface):
    """Concrete implementation of plot interface for DnD and SQW objects."""

    @staticmethod
    def check_data(
        w: Any, ndim: int, scalar_only: bool, func_name: str
    ) -> None:
        """Validate dimensionality and scalar-ness of object(s) to plot."""
        objs, is_scalar = _flatten_objs(w)
        dims = []
        for x in objs:
            d = x.dimensions()
            nd = d[0] if isinstance(d, tuple) else int(d)
            dims.append(nd)
        if (scalar_only and not is_scalar) or any(d != ndim for d in dims):
            if scalar_only:
                mess = (
                    f"Plot method '{func_name}' only works for a single "
                    f"{ndim}-dimensional object"
                )
            else:
                mess = (
                    f"Plot method '{func_name}' only works for "
                    f"{ndim}-dimensional object(s)"
                )
            raise HoraceError("HORACE:graphics:invalid_argument", mess)

    # ------------------------------------------------------------------
    # 1D Plot Methods
    # ------------------------------------------------------------------
    def dd(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "dd")
        return _format_plot_return(nargout=nargout)

    def de(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "de")
        return _format_plot_return(nargout=nargout)

    def dh(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "dh")
        return _format_plot_return(nargout=nargout)

    def dl(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "dl")
        return _format_plot_return(nargout=nargout)

    def dm(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "dm")
        return _format_plot_return(nargout=nargout)

    def dp(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "dp")
        return _format_plot_return(nargout=nargout)

    def pd(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pd")
        return _format_plot_return(nargout=nargout)

    def pdoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pdoc")
        return _format_plot_return(nargout=nargout)

    def pe(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pe")
        return _format_plot_return(nargout=nargout)

    def peoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "peoc")
        return _format_plot_return(nargout=nargout)

    def ph(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "ph")
        return _format_plot_return(nargout=nargout)

    def phoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "phoc")
        return _format_plot_return(nargout=nargout)

    def pl(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pl")
        return _format_plot_return(nargout=nargout)

    def ploc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "ploc")
        return _format_plot_return(nargout=nargout)

    def pm(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pm")
        return _format_plot_return(nargout=nargout)

    def pmoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pmoc")
        return _format_plot_return(nargout=nargout)

    def pp(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "pp")
        return _format_plot_return(nargout=nargout)

    def ppoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 1, False, "ppoc")
        return _format_plot_return(nargout=nargout)

    # ------------------------------------------------------------------
    # 2D Plot Methods
    # ------------------------------------------------------------------
    def da(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "da")
        return _format_plot_return(nargout=nargout)

    def ds(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "ds")
        return _format_plot_return(nargout=nargout)

    def ds2(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "ds2")
        return _format_plot_return(nargout=nargout)

    def pa(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "pa")
        return _format_plot_return(nargout=nargout)

    def paoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "poc")
        return _format_plot_return(nargout=nargout)

    def ps(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "ps")
        return _format_plot_return(nargout=nargout)

    def psoc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "psoc")
        return _format_plot_return(nargout=nargout)

    def ps2(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "ps2")
        return _format_plot_return(nargout=nargout)

    def ps2oc(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 2, False, "ps2oc")
        return _format_plot_return(nargout=nargout)

    # ------------------------------------------------------------------
    # 3D Plot Methods
    # ------------------------------------------------------------------
    def sliceomatic(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        self.check_data(self, 3, True, "sliceomatic")
        return _format_plot_return(nargout=nargout)

    def sliceomatic_overview(
        self, *args: Any, nargout: Optional[int] = None, **kwargs: Any
    ) -> Any:
        self.check_data(self, 3, True, "sliceomatic")
        return _format_plot_return(nargout=nargout)

    # ------------------------------------------------------------------
    # Generic ND Plot Methods
    # ------------------------------------------------------------------
    def plot(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        objs, _ = _flatten_objs(self)
        d = objs[0].dimensions()
        nd = d[0] if isinstance(d, tuple) else int(d)
        if nd == 1:
            dnd_plot_interface.check_data(self, 1, False, "dp")
            return _format_plot_return(nargout=nargout)
        if nd == 2:
            dnd_plot_interface.check_data(self, 2, False, "da")
            return _format_plot_return(nargout=nargout)
        if nd == 3:
            dnd_plot_interface.check_data(self, 3, True, "sliceomatic")
            return _format_plot_return(nargout=nargout)
        raise HoraceError(
            "HORACE:graphics:invalid_argument",
            f"Method 'plot' is not available for {nd}-dimensional objects",
        )

    def plotover(self, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        objs, _ = _flatten_objs(self)
        d = objs[0].dimensions()
        nd = d[0] if isinstance(d, tuple) else int(d)
        if nd == 1:
            dnd_plot_interface.check_data(self, 1, False, "pp")
            return _format_plot_return(nargout=nargout)
        if nd == 2:
            dnd_plot_interface.check_data(self, 2, False, "pa")
            return _format_plot_return(nargout=nargout)
        raise HoraceError(
            "HORACE:graphics:invalid_argument",
            f"Method 'plotover' is not available for {nd}-dimensional objects",
        )


def _bind_array_plot_methods() -> None:
    """Attach plot methods onto DataOpArray so arrays of DnD objects can call .dd(), .plot(), etc."""
    _methods_1d = [
        "dd", "de", "dh", "dl", "dm", "dp",
        "pd", "pdoc", "pe", "peoc", "ph", "phoc",
        "pl", "ploc", "pm", "pmoc", "pp", "ppoc",
    ]
    _methods_2d = [
        "da", "ds", "ds2", "pa", "paoc", "ps", "psoc", "ps2", "ps2oc",
    ]
    _methods_3d = ["sliceomatic", "sliceomatic_overview"]
    _methods_nd = ["plot", "plotover"]

    for m_name in _methods_1d:
        def _make_1d(name: str):
            def _fn(self: Any, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
                dnd_plot_interface.check_data(self, 1, False, name)
                return _format_plot_return(nargout=nargout)
            return _fn
        setattr(DataOpArray, m_name, _make_1d(m_name))

    for m_name in _methods_2d:
        def _make_2d(name: str):
            chk_name = "poc" if name == "paoc" else name
            def _fn(self: Any, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
                dnd_plot_interface.check_data(self, 2, False, chk_name)
                return _format_plot_return(nargout=nargout)
            return _fn
        setattr(DataOpArray, m_name, _make_2d(m_name))

    for m_name in _methods_3d:
        def _make_3d(name: str):
            def _fn(self: Any, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
                dnd_plot_interface.check_data(self, 3, True, "sliceomatic")
                return _format_plot_return(nargout=nargout)
            return _fn
        setattr(DataOpArray, m_name, _make_3d(m_name))

    def _array_plot(self: Any, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        return dnd_plot_interface.plot(self, *args, nargout=nargout, **kwargs)

    def _array_plotover(self: Any, *args: Any, nargout: Optional[int] = None, **kwargs: Any) -> Any:
        return dnd_plot_interface.plotover(self, *args, nargout=nargout, **kwargs)

    setattr(DataOpArray, "plot", _array_plot)
    setattr(DataOpArray, "plotover", _array_plotover)


_bind_array_plot_methods()

