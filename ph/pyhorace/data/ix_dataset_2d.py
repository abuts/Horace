"""IX_dataset_2d / IX_data_2d class definition.

Translated from:
- ``@IX_data_2d/IX_data_2d.m`` and ``@IX_data_2d/*.m`` (including ``private/``)
- ``@IX_dataset_2d/IX_dataset_2d.m`` and ``@IX_dataset_2d/*.m``
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.data.ix_axis import IX_axis, _is_empty, _is_numeric_array
from pyhorace.data.ix_dataset import (
    IX_dataset,
    _is_stringmatchi,
    bin_boundaries_simple,
    deriv_xye_n,
    integrate_nd_points,
    rebin_nd_hist,
    unspike_xye_n,
)
from pyhorace.data.sigvar import SigVar


@register_serializable
class IX_dataset_2d(IX_dataset):
    """2D dataset with x/y axes, signal, error, and axis annotations.

    Combines ``@IX_data_2d`` and ``@IX_dataset_2d``.
    """

    _fields_to_save = [
        "title",
        "signal",
        "error",
        "s_axis",
        "x",
        "x_axis",
        "x_distribution",
        "y",
        "y_axis",
        "y_distribution",
    ]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, IX_dataset) and arg0.ndim() == 3:
                return cls._build_from_ix_data_3d(arg0)
            if isinstance(arg0, DataOpArray) and arg0.size > 0:
                first = arg0.ravel(order="F")[0]
                if isinstance(first, IX_dataset) and first.ndim() == 3:
                    parts = [cls._build_from_ix_data_3d(el) for el in arg0.ravel(order="F")]
                    return DataOpArray(np.concatenate([p.ravel(order="F") for p in parts]))
                if hasattr(first, "IX_dataset_2d") and callable(first.IX_dataset_2d):
                    return DataOpArray([cls(el) for el in arg0.flat]).reshape(arg0.shape)
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._xyz = [np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)]
        self._xyz_axis = [IX_axis(), IX_axis()]
        self._xyz_distribution = np.array([True, True], dtype=bool)
        self._signal = np.zeros((0, 1), dtype=float)
        self._error = np.zeros((0, 1), dtype=float)
        self._do_check_combo_arg = True

        if len(args) > 0:
            self._build_ixdataset_2d(*args)
        if kwargs:
            prev = self._do_check_combo_arg
            self._do_check_combo_arg = False
            try:
                for k, v in kwargs.items():
                    setattr(self, k, v)
            finally:
                self._do_check_combo_arg = prev
            if self._do_check_combo_arg:
                self.check_combo_arg()

    def init(self, *args: Any, **kwargs: Any) -> IX_dataset_2d:
        """Re-initialise dataset in-place."""
        self.__init__(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """x-axis coordinates or bin boundaries."""
        return self._get_xyz_data(1)

    @x.setter
    def x(self, val: Any) -> None:
        self._set_xyz_data(1, val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def x_axis(self) -> IX_axis:
        """x-axis ``IX_axis`` description."""
        return self._xyz_axis[0]

    @x_axis.setter
    def x_axis(self, val: Any) -> None:
        self._xyz_axis[0] = self.check_and_build_axis(val)

    @property
    def x_distribution(self) -> bool:
        """True if signal is a distribution along x."""
        return bool(self._xyz_distribution[0])

    @x_distribution.setter
    def x_distribution(self, val: Any) -> None:
        self._xyz_distribution[0] = bool(
            np.asarray(val).item() if isinstance(val, np.ndarray) else val
        )

    @property
    def y(self) -> np.ndarray:
        """y-axis coordinates or bin boundaries."""
        return self._get_xyz_data(2)

    @y.setter
    def y(self, val: Any) -> None:
        self._set_xyz_data(2, val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def y_axis(self) -> IX_axis:
        """y-axis ``IX_axis`` description."""
        return self._xyz_axis[1]

    @y_axis.setter
    def y_axis(self, val: Any) -> None:
        self._xyz_axis[1] = self.check_and_build_axis(val)

    @property
    def y_distribution(self) -> bool:
        """True if signal is a distribution along y."""
        return bool(self._xyz_distribution[1])

    @y_distribution.setter
    def y_distribution(self, val: Any) -> None:
        self._xyz_distribution[1] = bool(
            np.asarray(val).item() if isinstance(val, np.ndarray) else val
        )

    # ------------------------------------------------------------------
    # Validation & Construction
    # ------------------------------------------------------------------

    @staticmethod
    def ndim() -> int:
        return 2

    def _check_and_set_sig_err(self, field_name: str, val: Any) -> None:
        if _is_empty(val) and not isinstance(val, str):
            arr = np.zeros((0, 1), dtype=float)
        else:
            if not _is_numeric_array(val):
                raise ValueError(
                    f"IX_dataset_2d:invalid_argument: {field_name} - array must be a numeric two-dimensional array"
                )
            arr = np.asarray(val, dtype=float)
            if arr.ndim > 2:
                raise ValueError(
                    f"IX_dataset_2d:invalid_argument: {field_name} - array must be a numeric two-dimensional array"
                )
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            elif arr.shape[0] == 1 and arr.shape[1] > 1:
                arr = arr.reshape(-1, 1)
        setattr(self, f"_{field_name}", arr)

    def check_combo_arg(self) -> IX_dataset_2d:
        """Validate interdependent properties ``x``, ``y``, ``signal``, and ``error``."""
        s0, s1 = self._signal.shape
        e0, e1 = self._error.shape
        if (s0, s1) != (e0, e1):
            raise ValueError(
                f"HERBERT:IX_data_2d:invalid_argument: size(signal)=[{s0},{s1}], size(error)=[{e0},{e1}]; size(signal)~=size(error)"
            )

        nx = self._xyz[0].size
        ny = self._xyz[1].size

        if (nx == s1 or nx == s1 + 1) and not (nx == s0 or nx == s0 + 1):
            self._signal = self._signal.T.copy()
            self._error = self._error.T.copy()
            s0, s1 = self._signal.shape

        if not (nx == s0 or nx == s0 + 1):
            raise ValueError(
                f"HERBERT:IX_data_2d:invalid_argument: size(signal,1)={s0}, numel(x)={nx}; size(signal,1) must be equal to numel(x) or numel(x)+1"
            )
        if not (ny == s1 or ny == s1 + 1):
            raise ValueError(
                f"HERBERT:IX_data_2d:invalid_argument: size(signal,2)={s1}, numel(y)={ny}; size(signal,2)  must be equal to numel(y) or numel(y)+1"
            )

        if nx == 0 or ny == 0:
            self._signal = np.zeros((0, 1), dtype=float)
            self._error = np.zeros((0, 1), dtype=float)
            self._xyz[0] = np.zeros((0,), dtype=float)
            self._xyz[1] = np.zeros((0,), dtype=float)
        return self

    def _build_ixdataset_2d(self, *args: Any) -> None:
        narg = len(args)
        prev = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            if narg == 0:
                return
            if narg == 1:
                in_val = args[0]
                if isinstance(in_val, IX_dataset_2d):
                    self._title = (
                        list(in_val._title)
                        if isinstance(in_val._title, list)
                        else in_val._title
                    )
                    self._signal = in_val._signal.copy()
                    self._error = in_val._error.copy()
                    self._s_axis = in_val._s_axis.copy()
                    self._xyz = [arr.copy() for arr in in_val._xyz]
                    self._xyz_axis = [ax.copy() for ax in in_val._xyz_axis]
                    self._xyz_distribution = in_val._xyz_distribution.copy()
                    self._obj_holder = getattr(in_val, "_obj_holder", None)
                    return
                if isinstance(in_val, dict):
                    self.init_from_structure(in_val)
                    return
                if isinstance(in_val, IX_dataset) and in_val.ndim() == 1:
                    self._build_from_ix_data_1d([in_val])
                    return
                if (
                    isinstance(in_val, (DataOpArray, list, tuple, np.ndarray))
                    and not _is_numeric_array(in_val)
                ):
                    arr_in = DataOpArray(in_val)
                    elems = list(arr_in.ravel(order="F"))
                    if elems and isinstance(elems[0], IX_dataset) and elems[0].ndim() == 1:
                        self._build_from_ix_data_1d(elems)
                        return
                if isinstance(in_val, SigVar):
                    s_arr = np.asarray(in_val.s, dtype=float)
                    e_arr = np.asarray(in_val.e, dtype=float)
                    if s_arr.ndim == 1:
                        s_arr = s_arr.reshape(-1, 1)
                        e_arr = e_arr.reshape(-1, 1)
                    self.x = np.arange(1, s_arr.shape[0] + 1, dtype=float)
                    self.y = np.arange(1, s_arr.shape[1] + 1, dtype=float)
                    self.signal = s_arr
                    self.error = np.sqrt(e_arr)
                elif _is_numeric_array(in_val):
                    arr = np.asarray(in_val, dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(-1, 1)
                    self.x = np.arange(1, arr.shape[0] + 1, dtype=float)
                    self.y = np.arange(1, arr.shape[1] + 1, dtype=float)
                    self.signal = arr
                    self.error = np.zeros_like(arr)
                elif hasattr(in_val, "IX_dataset_2d") and callable(in_val.IX_dataset_2d):
                    res = in_val.IX_dataset_2d()
                    self._build_ixdataset_2d(res)
                    if getattr(self, "_obj_holder", None) is None:
                        self._obj_holder = getattr(in_val, "_obj_holder", None)
                    return
                else:
                    raise ValueError(
                        "HERBERT:IX_data_2d:invalid_argument: Check type(s) of input argument(s)"
                    )
            elif narg == 2:
                self.x = args[0]
                self.y = args[1]
                nx, ny = self._xyz[0].size, self._xyz[1].size
                self._signal = np.zeros((nx, ny), dtype=float) if (nx > 0 and ny > 0) else np.zeros((0, 1), dtype=float)
                self._error = np.zeros_like(self._signal)
            elif narg == 3:
                self.x = args[0]
                self.y = args[1]
                self.signal = args[2]
                self._error = np.zeros(self._signal.shape, dtype=float)
            elif narg == 4:
                self.x = args[0]
                self.y = args[1]
                self.signal = args[2]
                self.error = args[3]
            elif narg == 6:
                self.x = args[0]
                self.y = args[1]
                self.signal = args[2]
                self.error = args[3]
                self.x_distribution = args[4]
                self.y_distribution = args[5]
            elif narg == 8:
                self.x = args[0]
                self.y = args[1]
                self.signal = args[2]
                self.error = args[3]
                self.title = args[4]
                self.x_axis = args[5]
                self.y_axis = args[6]
                self.s_axis = args[7]
            elif narg == 10 and isinstance(args[6], (bool, np.bool_)) and not isinstance(args[8], (bool, np.bool_)):
                self.title = args[0]
                self.signal = args[1]
                self.error = args[2]
                self.s_axis = args[3]
                self.x = args[4]
                self.x_axis = args[5]
                self.x_distribution = args[6]
                self.y = args[7]
                self.y_axis = args[8]
                self.y_distribution = args[9]
            elif narg == 10 and _is_numeric_array(args[0]):
                self.x = args[0]
                self.y = args[1]
                self.signal = args[2]
                self.error = args[3]
                self.title = args[4]
                self.x_axis = args[5]
                self.y_axis = args[6]
                self.s_axis = args[7]
                self.x_distribution = args[8]
                self.y_distribution = args[9]
            else:
                raise ValueError(
                    "HERBERT:IX_data_2d:invalid_argument: Wrong number of input arguments"
                )
        finally:
            self._do_check_combo_arg = prev
        self.check_combo_arg()

    def _build_from_ix_data_1d(self, elems: list[IX_dataset]) -> None:
        n = len(elems)
        first = elems[0]
        nx_sig = first._signal.size
        nx_x = first._xyz[0].size
        for el in elems[1:]:
            if el._signal.size != nx_sig or el._xyz[0].size != nx_x:
                raise ValueError(
                    "HERBERT:IX_data_2d:invalid_argument: All 1D datasets must have the same x-axis and signal length"
                )
            if not np.allclose(el._xyz[0], first._xyz[0]):
                raise ValueError(
                    "HERBERT:IX_data_2d:invalid_argument: All 1D datasets must have identical x-axis values"
                )
        self._title = (
            list(first._title) if isinstance(first._title, list) else first._title
        )
        self._s_axis = first._s_axis.copy()
        self._xyz[0] = first._xyz[0].copy()
        self._xyz_axis[0] = first._xyz_axis[0].copy()
        self._xyz_distribution[0] = first._xyz_distribution[0]
        self._xyz[1] = np.arange(1, n + 1, dtype=float)
        self._xyz_axis[1] = IX_axis("Spectrum index")
        self._xyz_distribution[1] = False
        self._signal = np.column_stack([el._signal.ravel() for el in elems])
        self._error = np.column_stack([el._error.ravel() for el in elems])

    @classmethod
    def _build_from_ix_data_3d(cls, in_obj: IX_dataset) -> DataOpArray:
        sig = in_obj._signal
        err = in_obj._error
        if sig.size == 0:
            return DataOpArray([cls()])
        sz = list(sig.shape) + [1] * (3 - sig.ndim)
        nz = sz[2]
        sig_3d = sig.reshape(sz)
        err_3d = err.reshape(sz)
        out = []
        for k in range(nz):
            w = cls(
                in_obj.title,
                sig_3d[:, :, k],
                err_3d[:, :, k],
                in_obj.s_axis,
                in_obj._xyz[0],
                in_obj._xyz_axis[0],
                in_obj._xyz_distribution[0],
                in_obj._xyz[1],
                in_obj._xyz_axis[1],
                in_obj._xyz_distribution[1],
            )
            out.append(w)
        return DataOpArray(out)

    # ------------------------------------------------------------------
    # Core Dataset Inquiry & Conversion Methods
    # ------------------------------------------------------------------

    def axis(self, n: int | list[int] | np.ndarray | None = None) -> tuple[Any, Any]:
        """Return ``(ax_struct, hist_flag)`` for axis ``n`` (or both axes)."""
        if n is None:
            ind = [1, 2]
            single = False
        else:
            ind_arr = np.asarray(n, dtype=int).ravel()
            if np.any(ind_arr < 1) or np.any(ind_arr > 2):
                raise ValueError(
                    "HERBERT:IX_data_2d:invalid_argument: Axis index must be 1 or 2"
                )
            ind = [int(x) for x in ind_arr]
            single = len(ind) == 1

        ax_list = []
        hist_list = []
        for idx in ind:
            h_i = self.ishistogram(idx)
            ax_list.append(
                {
                    "values": self._xyz[idx - 1].copy(),
                    "axis": self._xyz_axis[idx - 1].copy(),
                    "distribution": bool(self._xyz_distribution[idx - 1]),
                }
            )
            hist_list.append(bool(h_i))
        if single:
            return ax_list[0], hist_list[0]
        if hist_list[0] == hist_list[1]:
            return ax_list, hist_list[0]
        return ax_list, np.array(hist_list, dtype=bool)

    def ishistogram(self, n: int | None = None) -> np.ndarray | bool:
        """Return histogram status for axis ``n`` (1 or 2) or both axes."""
        if self._signal.size == 0:
            res = np.array(
                [self._xyz[0].size == 1, self._xyz[1].size == 1], dtype=bool
            )
        else:
            res = np.array(
                [
                    self._xyz[0].size == self._signal.shape[0] + 1,
                    self._xyz[1].size == self._signal.shape[1] + 1,
                ],
                dtype=bool,
            )
        if n is None:
            return res
        n_int = int(np.asarray(n).item())
        if n_int not in (1, 2):
            raise ValueError(
                "HERBERT:IX_data_2d:invalid_argument: Axis index must be 1 or 2"
            )
        return bool(res[n_int - 1])

    def dimensions(self) -> tuple[int, tuple[int, ...]]:
        """Return ``(nd, sz)`` of dataset."""
        if self._xyz[0].size == 0 or self._xyz[1].size == 0:
            return 0, (0, 0)
        return 2, self._signal.shape

    def get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, y)`` arrays."""
        return self._xyz[0].copy(), self._xyz[1].copy()

    def get_xysigerr(self) -> dict[str, Any]:
        """Return dictionary with ``x``, ``y``, ``signal``, ``err``, ``x_distribution``, ``y_distribution``."""
        return {
            "x": [self._xyz[0].copy(), self._xyz[1].copy()],
            "signal": self._signal.copy(),
            "err": self._error.copy(),
            "x_distribution": self._xyz_distribution.copy(),
        }

    def sigvar_getx(self) -> tuple[np.ndarray, np.ndarray]:
        """Return meshgrid ``(X, Y)`` of bin centers or point coordinates."""
        xc = (
            0.5 * (self._xyz[0][1:] + self._xyz[0][:-1])
            if self.ishistogram(1)
            else self._xyz[0]
        )
        yc = (
            0.5 * (self._xyz[1][1:] + self._xyz[1][:-1])
            if self.ishistogram(2)
            else self._xyz[1]
        )
        return np.meshgrid(xc, yc, indexing="ij")

    def cnt2dist(self, *args: Any) -> IX_dataset_2d:
        """Convert count data to distribution data along specified or all axes."""
        iax = [1, 2] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        for ax in iax:
            if ax not in (1, 2):
                raise ValueError("IX_dataset_2d:invalid_argument: Axis index must be 1 or 2")
            if wout.ishistogram(ax) and not wout._xyz_distribution[ax - 1]:
                d = np.diff(wout._xyz[ax - 1])
                shape = (-1, 1) if ax == 1 else (1, -1)
                wout._signal = wout._signal / d.reshape(shape)
                wout._error = wout._error / d.reshape(shape)
            wout._xyz_distribution[ax - 1] = True
        return wout

    def dist2cnt(self, *args: Any) -> IX_dataset_2d:
        """Convert distribution data to count data along specified or all axes."""
        iax = [1, 2] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        for ax in iax:
            if ax not in (1, 2):
                raise ValueError("IX_dataset_2d:invalid_argument: Axis index must be 1 or 2")
            if wout.ishistogram(ax) and wout._xyz_distribution[ax - 1]:
                d = np.diff(wout._xyz[ax - 1])
                shape = (-1, 1) if ax == 1 else (1, -1)
                wout._signal = wout._signal * d.reshape(shape)
                wout._error = wout._error * d.reshape(shape)
            wout._xyz_distribution[ax - 1] = False
        return wout

    def point2hist(self, *args: Any) -> IX_dataset_2d:
        """Convert point axes to histogram axes."""
        iax = [1, 2] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        for ax in iax:
            if ax not in (1, 2):
                raise ValueError("IX_dataset_2d:invalid_argument: Axis index must be 1 or 2")
            if not wout.ishistogram(ax):
                xb, ok, mess = bin_boundaries_simple(wout._xyz[ax - 1])
                if not ok:
                    raise ValueError(f"IX_dataset_2d:invalid_argument: {mess}")
                if not wout._xyz_distribution[ax - 1]:
                    d = np.diff(xb)
                    shape = (-1, 1) if ax == 1 else (1, -1)
                    wout._signal = wout._signal * d.reshape(shape)
                    wout._error = wout._error * d.reshape(shape)
                wout._xyz[ax - 1] = xb
        return wout

    def hist2point(self, *args: Any) -> IX_dataset_2d:
        """Convert histogram axes to point axes (at bin centres)."""
        iax = [1, 2] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        for ax in iax:
            if ax not in (1, 2):
                raise ValueError("IX_dataset_2d:invalid_argument: Axis index must be 1 or 2")
            if wout.ishistogram(ax):
                x = wout._xyz[ax - 1]
                if not wout._xyz_distribution[ax - 1]:
                    d = np.diff(x)
                    shape = (-1, 1) if ax == 1 else (1, -1)
                    wout._signal = wout._signal / d.reshape(shape)
                    wout._error = wout._error / d.reshape(shape)
                wout._xyz[ax - 1] = 0.5 * (x[1:] + x[:-1])
        return wout

    # ------------------------------------------------------------------
    # Rebinning, Integrating, Cutting, Manipulation Methods
    # ------------------------------------------------------------------

    @staticmethod
    def rebin_hist(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rebin_nd_hist(x, s, e, xout, iax=iax, ndim=2)

    @staticmethod
    def integrate_points(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return integrate_nd_points(x, s, e, xout, iax=iax, ndim=2)

    def rebin(self, *args: Any) -> IX_dataset_2d:
        """Rebin 2D dataset along both axes."""
        return self.rebin_xy(*args)

    def rebin2(self, *args: Any) -> IX_dataset_2d:
        """Rebin 2D dataset using explicit bin boundaries."""
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(False, [1, 2], *args)

    def rebin_x(self, *args: Any) -> IX_dataset_2d:
        return self._rebin_xyz(True, 1, *args)

    def rebin_y(self, *args: Any) -> IX_dataset_2d:
        return self._rebin_xyz(True, 2, *args)

    def rebin_xy(self, *args: Any) -> IX_dataset_2d:
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(True, [1, 2], *args)

    def integrate(self, *args: Any) -> Any:
        """Integrate 2D dataset along both axes."""
        return self.integrate_xy(*args)

    def integrate2(self, *args: Any) -> IX_dataset_2d:
        """Integrate 2D dataset using explicit bin boundaries without squeezing."""
        opt = {
            "empty_is_full_range": True,
            "range_is_one_bin": True,
            "array_is_descriptor": False,
            "bin_boundaries": True,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, True, True, [1, 2], opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset_2d:invalid_argument: {mess}")
        return wout

    def integrate_x(self, *args: Any) -> Any:
        return self._integrate_xyz(True, 1, *args)

    def integrate_y(self, *args: Any) -> Any:
        return self._integrate_xyz(True, 2, *args)

    def integrate_xy(self, *args: Any) -> Any:
        return self._integrate_xyz(True, [1, 2], *args)

    def cut(self, *args: Any) -> Any:
        """Take a 2D cut along both axes."""
        return self.cut_xy(*args)

    def cut_x(self, *args: Any) -> Any:
        return self._cut_xyz(1, *args)

    def cut_y(self, *args: Any) -> Any:
        return self._cut_xyz(2, *args)

    def cut_xy(self, *args: Any) -> Any:
        return self._cut_xyz([1, 2], *args)

    def deriv_x(self) -> IX_dataset_2d:
        wout = self.copy()
        xc = (
            0.5 * (self._xyz[0][1:] + self._xyz[0][:-1])
            if self.ishistogram(1)
            else self._xyz[0]
        )
        wout._signal, wout._error = deriv_xye_n(1, xc, self._signal, self._error)
        return wout

    def deriv_y(self) -> IX_dataset_2d:
        wout = self.copy()
        yc = (
            0.5 * (self._xyz[1][1:] + self._xyz[1][:-1])
            if self.ishistogram(2)
            else self._xyz[1]
        )
        wout._signal, wout._error = deriv_xye_n(2, yc, self._signal, self._error)
        return wout

    def deriv(self, iax: int = 1) -> IX_dataset_2d:
        if iax == 1:
            return self.deriv_x()
        if iax == 2:
            return self.deriv_y()
        raise ValueError("IX_dataset_2d:invalid_argument: iax must be 1 or 2")

    def unspike(
        self,
        ymin: float | None = None,
        ymax: float | None = None,
        fac: float | None = None,
        sfac: float | None = None,
    ) -> IX_dataset_2d:
        """Remove spikes along x-axis of 2D dataset."""
        wout = self.copy()
        xc = (
            0.5 * (self._xyz[0][1:] + self._xyz[0][:-1])
            if self.ishistogram(1)
            else self._xyz[0]
        )
        wout._signal, wout._error = unspike_xye_n(
            1, xc, self._signal, self._error, ymin, ymax, fac, sfac
        )
        return wout

    def func_eval(self, func_handle: Callable[..., Any], *args: Any) -> IX_dataset_2d:
        """Evaluate ``func_handle(x, y, *pars)`` at bin centers or points."""
        all_flag = False
        pars = list(args)
        if len(pars) >= 2 and isinstance(pars[-1], str) and _is_stringmatchi(pars[-1], "all"):
            all_flag = True
            pars = pars[:-1]
        if len(pars) == 1 and isinstance(pars[0], (list, tuple)):
            pars = list(pars[0])

        wout = self.copy()
        X, Y = self.sigvar_getx()
        s_eval = np.asarray(
            func_handle(X.ravel(), Y.ravel(), *pars), dtype=float
        ).reshape(self._signal.shape)
        if not all_flag:
            nan_mask = np.isnan(self._signal)
            s_eval = np.where(nan_mask, np.nan, s_eval)
        wout._signal = s_eval
        wout._error = np.zeros_like(wout._signal)
        return wout

    def noisify(self, *args: Any, **kwargs: Any) -> IX_dataset_2d:
        from pyhorace.utilities.math import noisify as _noisify

        wout = self.copy()
        s_noisy, e_noisy = _noisify(
            self._signal, self._error, *args, **kwargs
        )
        wout._signal = np.asarray(s_noisy, dtype=float).reshape(self._signal.shape)
        wout._error = np.asarray(e_noisy, dtype=float).reshape(self._error.shape)
        return wout

    def scale(self, xscale: float = 1.0, yscale: float = 1.0) -> IX_dataset_2d:
        return self.scale_x(xscale).scale_y(yscale)

    def scale_x(self, xscale: float) -> IX_dataset_2d:
        sc = float(np.asarray(xscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_2d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[0] = wout._xyz[0] * sc
        else:
            wout._xyz[0] = (wout._xyz[0] * sc)[::-1].copy()
            wout._signal = wout._signal[::-1, :].copy()
            wout._error = wout._error[::-1, :].copy()
        if wout.ishistogram(1) and wout.x_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def scale_y(self, yscale: float) -> IX_dataset_2d:
        sc = float(np.asarray(yscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_2d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[1] = wout._xyz[1] * sc
        else:
            wout._xyz[1] = (wout._xyz[1] * sc)[::-1].copy()
            wout._signal = wout._signal[:, ::-1].copy()
            wout._error = wout._error[:, ::-1].copy()
        if wout.ishistogram(2) and wout.y_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def shift(self, xshift: float = 0.0, yshift: float = 0.0) -> IX_dataset_2d:
        return self.shift_x(xshift).shift_y(yshift)

    def shift_x(self, xshift: float = 0.0) -> IX_dataset_2d:
        wout = self.copy()
        wout._xyz[0] = wout._xyz[0] + float(np.asarray(xshift).item())
        return wout

    def shift_y(self, yshift: float = 0.0) -> IX_dataset_2d:
        wout = self.copy()
        wout._xyz[1] = wout._xyz[1] + float(np.asarray(yshift).item())
        return wout

    def xye(self) -> dict[str, Any]:
        """Return dict ``{'x': [xc, yc], 'y': signal, 'e': error}``."""
        xc = (
            0.5 * (self._xyz[0][1:] + self._xyz[0][:-1])
            if self.ishistogram(1)
            else self._xyz[0].copy()
        )
        yc = (
            0.5 * (self._xyz[1][1:] + self._xyz[1][:-1])
            if self.ishistogram(2)
            else self._xyz[1].copy()
        )
        return {
            "x": [xc.reshape(-1, 1), yc.reshape(-1, 1)],
            "y": self._signal.copy(),
            "e": self._error.copy(),
        }

    def make_label(self) -> tuple[list[str], list[str], list[str]]:
        """Return ``(xlabel, ylabel, slabel)`` formatted axis labels."""
        xlabel, _ = self._xyz_axis[0].make_label()
        ylabel, _ = self._xyz_axis[1].make_label()
        slabel, units_appended = self._s_axis.make_label()
        denom_parts = []
        if self.x_distribution and self._xyz_axis[0].units:
            denom_parts.append(f" / {self._xyz_axis[0].units}")
        if self.y_distribution and self._xyz_axis[1].units:
            denom_parts.append(f" / {self._xyz_axis[1].units}")
        denom = "".join(denom_parts)
        if denom:
            if slabel:
                if units_appended:
                    slabel[-1] = f"{slabel[-1][:-1]}{denom})"
                else:
                    slabel[-1] = f"{slabel[-1]} (1{denom})"
            else:
                slabel = [f"(1{denom})"]
        return xlabel, ylabel, slabel

    def expand_d1d(self) -> DataOpArray:
        """Expand 2D dataset into an array of 1D datasets along y."""
        from pyhorace.data.ix_dataset_1d import IX_dataset_1d
        return IX_dataset_1d._build_from_ix_data_2d(self)

    def contract_d2d(self) -> IX_dataset_2d:
        return self.copy()

    # ------------------------------------------------------------------
    # 2D Graphics / Plotting Methods (delegating to graphics.py)
    # ------------------------------------------------------------------

    def _plot_2d(
        self,
        plot_type: str,
        new_axes: bool,
        force_current_axes: bool,
        *args: Any,
        _dataset_array: Any = None,
        **kwargs: Any,
    ) -> Any:
        from pyhorace.data.data_plot_interface import _extract_datasets, data_plot_interface
        from pyhorace.data.graphics import plot_2d_datasets

        datasets = _dataset_array if _dataset_array is not None else [self]
        color_datasets = None
        rem_args = args
        if plot_type == "s2" and len(args) > 0:
            arg0 = args[0]
            if isinstance(arg0, (data_plot_interface, DataOpArray)) or (
                isinstance(arg0, (list, tuple, np.ndarray))
                and len(arg0) > 0
                and isinstance(np.asarray(arg0, dtype=object).flat[0], data_plot_interface)
            ):
                color_datasets = _extract_datasets(arg0)
                rem_args = args[1:]
        return plot_2d_datasets(
            datasets,
            plot_type,
            new_axes,
            force_current_axes,
            *rem_args,
            color_datasets=color_datasets,
        )

    def da(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("a", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def ds(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def ds2(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s2", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def pa(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("a", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def paoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("a", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def ps(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def psoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def ps2(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s2", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def ps2oc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_2d("s2", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    @classmethod
    def loadobj(cls, S: dict[str, Any] | IX_dataset_2d) -> Any:
        if isinstance(S, cls):
            return S
        obj = cls()
        return Serializable.loadobj(S, obj)


IX_data_2d = IX_dataset_2d
register_serializable(IX_data_2d)
