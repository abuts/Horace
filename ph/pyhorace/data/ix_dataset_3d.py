"""IX_dataset_3d / IX_data_3d class definition.

Translated from:
- ``@IX_data_3d/IX_data_3d.m`` and ``@IX_data_3d/*.m`` (including ``private/``)
- ``@IX_dataset_3d/IX_dataset_3d.m`` and ``@IX_dataset_3d/*.m``
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


def _size_3d(arr: np.ndarray) -> list[int]:
    """Return 3-element size list matching MATLAB ``size(arr, 1..3)``."""
    if arr.shape == (0, 0, 0):
        return [0, 0, 0]
    if arr.size == 0 and arr.ndim < 3:
        return [0, 0, 0]
    sz = list(arr.shape)
    while len(sz) < 3:
        sz.append(1)
    return sz[:3]


@register_serializable
class IX_dataset_3d(IX_dataset):
    """3D dataset with x/y/z axes, signal, error, and axis annotations.

    Combines ``@IX_data_3d`` and ``@IX_dataset_3d``.
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
        "z",
        "z_axis",
        "z_distribution",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._xyz = [
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        ]
        self._xyz_axis = [IX_axis(), IX_axis(), IX_axis()]
        self._xyz_distribution = np.array([True, True, True], dtype=bool)
        self._signal = np.zeros((0, 0, 0), dtype=float)
        self._error = np.zeros((0, 0, 0), dtype=float)
        self._do_check_combo_arg = True

        if len(args) > 0:
            self._build_ixdataset_3d(*args)
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

    def init(self, *args: Any, **kwargs: Any) -> IX_dataset_3d:
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

    @property
    def z(self) -> np.ndarray:
        """z-axis coordinates or bin boundaries."""
        return self._get_xyz_data(3)

    @z.setter
    def z(self, val: Any) -> None:
        self._set_xyz_data(3, val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def z_axis(self) -> IX_axis:
        """z-axis ``IX_axis`` description."""
        return self._xyz_axis[2]

    @z_axis.setter
    def z_axis(self, val: Any) -> None:
        self._xyz_axis[2] = self.check_and_build_axis(val)

    @property
    def z_distribution(self) -> bool:
        """True if signal is a distribution along z."""
        return bool(self._xyz_distribution[2])

    @z_distribution.setter
    def z_distribution(self, val: Any) -> None:
        self._xyz_distribution[2] = bool(
            np.asarray(val).item() if isinstance(val, np.ndarray) else val
        )

    # ------------------------------------------------------------------
    # Validation & Construction
    # ------------------------------------------------------------------

    @staticmethod
    def ndim() -> int:
        return 3

    def _check_and_set_sig_err(self, field_name: str, val: Any) -> None:
        if _is_empty(val) and not isinstance(val, str):
            arr = np.zeros((0, 0, 0), dtype=float)
        else:
            if not _is_numeric_array(val):
                raise ValueError(
                    f"IX_dataset_3d:invalid_argument: {field_name} - array must be a numeric three-dimensional array"
                )
            arr = np.asarray(val, dtype=float)
            if arr.ndim > 3:
                raise ValueError(
                    f"IX_dataset_3d:invalid_argument: {field_name} - array must be a numeric three-dimensional array"
                )
            if arr.ndim == 0:
                arr = arr.reshape(1, 1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(-1, 1)
        setattr(self, f"_{field_name}", arr)

    def check_combo_arg(self) -> IX_dataset_3d:
        """Validate interdependent properties ``x``, ``y``, ``z``, ``signal``, and ``error``."""
        sz_s = _size_3d(self._signal)
        sz_e = _size_3d(self._error)
        if sz_s != sz_e:
            raise ValueError(
                f"HERBERT:IX_data_3d:invalid_argument: size(signal)=[{sz_s[0]},{sz_s[1]},{sz_s[2]}], size(error)=[{sz_e[0]},{sz_e[1]},{sz_e[2]}]; size(signal)~=size(error)"
            )

        ax_names = ("x", "y", "z")
        for i, name in enumerate(ax_names):
            n_ax = self._xyz[i].size
            s_i = sz_s[i]
            if not (n_ax == s_i or n_ax == s_i + 1):
                raise ValueError(
                    f"HERBERT:IX_data_3d:invalid_argument: size(signal,{i+1})={s_i}, numel({name})={n_ax}; size(signal,{i+1}) must be equal to numel({name}) or numel({name})+1"
                )

        if any(self._xyz[i].size == 0 for i in range(3)):
            self._signal = np.zeros((0, 0, 0), dtype=float)
            self._error = np.zeros((0, 0, 0), dtype=float)
            for i in range(3):
                self._xyz[i] = np.zeros((0,), dtype=float)
        return self

    def _build_ixdataset_3d(self, *args: Any) -> None:
        narg = len(args)
        prev = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            if narg == 0:
                return
            if narg == 1:
                in_val = args[0]
                if isinstance(in_val, IX_dataset_3d):
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
                    return
                if isinstance(in_val, dict):
                    self.init_from_structure(in_val)
                    return
                if isinstance(in_val, IX_dataset) and in_val.ndim() == 1:
                    self._build_from_ix_data_1d(DataOpArray([[in_val]]))
                    return
                if isinstance(in_val, IX_dataset) and in_val.ndim() == 2:
                    self._build_from_ix_data_2d([in_val])
                    return
                if (
                    isinstance(in_val, (DataOpArray, list, tuple, np.ndarray))
                    and not _is_numeric_array(in_val)
                ):
                    arr_in = DataOpArray(in_val)
                    elems = list(arr_in.ravel(order="F"))
                    if elems and isinstance(elems[0], IX_dataset):
                        if elems[0].ndim() == 2:
                            self._build_from_ix_data_2d(elems)
                            return
                        if elems[0].ndim() == 1:
                            self._build_from_ix_data_1d(arr_in)
                            return
                if isinstance(in_val, SigVar):
                    s_arr = np.asarray(in_val.s, dtype=float)
                    e_arr = np.asarray(in_val.e, dtype=float)
                    sz = _size_3d(s_arr)
                    self.x = np.arange(1, sz[0] + 1, dtype=float)
                    self.y = np.arange(1, sz[1] + 1, dtype=float)
                    self.z = np.arange(1, sz[2] + 1, dtype=float)
                    self.signal = s_arr
                    self.error = np.sqrt(e_arr)
                elif _is_numeric_array(in_val):
                    arr = np.asarray(in_val, dtype=float)
                    sz = _size_3d(arr)
                    self.x = np.arange(1, sz[0] + 1, dtype=float)
                    self.y = np.arange(1, sz[1] + 1, dtype=float)
                    self.z = np.arange(1, sz[2] + 1, dtype=float)
                    self.signal = arr
                    self.error = np.zeros_like(arr)
                elif hasattr(in_val, "IX_dataset_3d") and callable(in_val.IX_dataset_3d):
                    res = in_val.IX_dataset_3d()
                    self._build_ixdataset_3d(res)
                    return
                else:
                    raise ValueError(
                        "HERBERT:IX_data_3d:invalid_argument: Check type(s) of input argument(s)"
                    )
            elif narg == 3:
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                nx, ny, nz = self._xyz[0].size, self._xyz[1].size, self._xyz[2].size
                self._signal = (
                    np.zeros((nx, ny, nz), dtype=float)
                    if (nx > 0 and ny > 0 and nz > 0)
                    else np.zeros((0, 0, 0), dtype=float)
                )
                self._error = np.zeros_like(self._signal)
            elif narg == 4:
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                self.signal = args[3]
                self._error = np.zeros(self._signal.shape, dtype=float)
            elif narg == 5:
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                self.signal = args[3]
                self.error = args[4]
            elif narg == 8:
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                self.signal = args[3]
                self.error = args[4]
                self.x_distribution = args[5]
                self.y_distribution = args[6]
                self.z_distribution = args[7]
            elif narg == 10:
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                self.signal = args[3]
                self.error = args[4]
                self.title = args[5]
                self.x_axis = args[6]
                self.y_axis = args[7]
                self.z_axis = args[8]
                self.s_axis = args[9]
            elif narg == 13 and isinstance(args[6], (bool, np.bool_)) and isinstance(args[9], (bool, np.bool_)):
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
                self.z = args[10]
                self.z_axis = args[11]
                self.z_distribution = args[12]
            elif narg == 13 and _is_numeric_array(args[0]):
                self.x = args[0]
                self.y = args[1]
                self.z = args[2]
                self.signal = args[3]
                self.error = args[4]
                self.title = args[5]
                self.x_axis = args[6]
                self.y_axis = args[7]
                self.z_axis = args[8]
                self.s_axis = args[9]
                self.x_distribution = args[10]
                self.y_distribution = args[11]
                self.z_distribution = args[12]
            else:
                raise ValueError(
                    "HERBERT:IX_data_3d:invalid_argument: Wrong number of input arguments"
                )
        finally:
            self._do_check_combo_arg = prev
        self.check_combo_arg()

    def _build_from_ix_data_2d(self, elems: list[IX_dataset]) -> None:
        nz = len(elems)
        first = elems[0]
        s_shape = first._signal.shape
        for el in elems[1:]:
            if (
                el._signal.shape != s_shape
                or not np.allclose(el._xyz[0], first._xyz[0])
                or not np.allclose(el._xyz[1], first._xyz[1])
            ):
                raise ValueError(
                    "HERBERT:IX_data_3d:invalid_argument: All 2D datasets must have identical x and y axes and signal sizes"
                )
        self._title = (
            list(first._title) if isinstance(first._title, list) else first._title
        )
        self._s_axis = first._s_axis.copy()
        self._xyz[0] = first._xyz[0].copy()
        self._xyz_axis[0] = first._xyz_axis[0].copy()
        self._xyz_distribution[0] = first._xyz_distribution[0]
        self._xyz[1] = first._xyz[1].copy()
        self._xyz_axis[1] = first._xyz_axis[1].copy()
        self._xyz_distribution[1] = first._xyz_distribution[1]
        self._xyz[2] = np.arange(1, nz + 1, dtype=float)
        self._xyz_axis[2] = IX_axis("Index")
        self._xyz_distribution[2] = False
        self._signal = np.stack([el._signal for el in elems], axis=2)
        self._error = np.stack([el._error for el in elems], axis=2)

    def _build_from_ix_data_1d(self, arr_in: DataOpArray) -> None:
        ny, nz = arr_in.shape[:2] if arr_in.ndim >= 2 else (arr_in.size, 1)
        first = arr_in.ravel(order="F")[0]
        nx = first._signal.size
        sig_3d = np.zeros((nx, ny, nz), dtype=float)
        err_3d = np.zeros((nx, ny, nz), dtype=float)
        for j in range(ny):
            for k in range(nz):
                el = arr_in[j, k]
                if el._signal.size != nx or not np.allclose(el._xyz[0], first._xyz[0]):
                    raise ValueError(
                        "HERBERT:IX_data_3d:invalid_argument: All 1D datasets must have identical x-axis and signal length"
                    )
                sig_3d[:, j, k] = el._signal.ravel()
                err_3d[:, j, k] = el._error.ravel()
        self._title = (
            list(first._title) if isinstance(first._title, list) else first._title
        )
        self._s_axis = first._s_axis.copy()
        self._xyz[0] = first._xyz[0].copy()
        self._xyz_axis[0] = first._xyz_axis[0].copy()
        self._xyz_distribution[0] = first._xyz_distribution[0]
        self._xyz[1] = np.arange(1, ny + 1, dtype=float)
        self._xyz_axis[1] = IX_axis("Index_1")
        self._xyz_distribution[1] = False
        self._xyz[2] = np.arange(1, nz + 1, dtype=float)
        self._xyz_axis[2] = IX_axis("Index_2")
        self._xyz_distribution[2] = False
        self._signal = sig_3d
        self._error = err_3d

    # ------------------------------------------------------------------
    # Core Dataset Inquiry & Conversion Methods
    # ------------------------------------------------------------------

    def axis(self, n: int | list[int] | np.ndarray | None = None) -> tuple[Any, Any]:
        """Return ``(ax_struct, hist_flag)`` for axis ``n`` (or all 3 axes)."""
        if n is None:
            ind = [1, 2, 3]
            single = False
        else:
            ind_arr = np.asarray(n, dtype=int).ravel()
            if np.any(ind_arr < 1) or np.any(ind_arr > 3):
                raise ValueError(
                    "HERBERT:IX_data_3d:invalid_argument: Axis index must be 1, 2, or 3"
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
        if len(set(hist_list)) == 1:
            return ax_list, hist_list[0]
        return ax_list, np.array(hist_list, dtype=bool)

    def ishistogram(self, n: int | None = None) -> np.ndarray | bool:
        """Return histogram status for axis ``n`` (1, 2, or 3) or all 3 axes."""
        sz = _size_3d(self._signal)
        if self._signal.size == 0:
            res = np.array(
                [self._xyz[i].size == 1 for i in range(3)], dtype=bool
            )
        else:
            res = np.array(
                [self._xyz[i].size == sz[i] + 1 for i in range(3)], dtype=bool
            )
        if n is None:
            return res
        n_int = int(np.asarray(n).item())
        if n_int not in (1, 2, 3):
            raise ValueError(
                "HERBERT:IX_data_3d:invalid_argument: Axis index must be 1, 2, or 3"
            )
        return bool(res[n_int - 1])

    def dimensions(self) -> tuple[int, tuple[int, ...]]:
        """Return ``(nd, sz)`` of dataset."""
        if any(self._xyz[i].size == 0 for i in range(3)):
            return 0, (0, 0)
        return 3, tuple(_size_3d(self._signal))

    def get_xyz(self, nd: int | None = None) -> Any:
        """Return ``(x, y, z)`` or single axis array."""
        if nd is None:
            return [arr.copy() for arr in self._xyz]
        return self._xyz[nd - 1].copy()

    def get_xyzsigerr(self) -> dict[str, Any]:
        """Return dictionary with ``x``, ``signal``, ``err``, ``x_distribution``."""
        return {
            "x": [arr.copy() for arr in self._xyz],
            "signal": self._signal.copy(),
            "err": self._error.copy(),
            "x_distribution": self._xyz_distribution.copy(),
        }

    def sigvar_getx(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 3D meshgrid ``(X, Y, Z)`` of bin centers or point coordinates."""
        coords = []
        for i in range(3):
            x_i = self._xyz[i]
            coords.append(
                0.5 * (x_i[1:] + x_i[:-1]) if self.ishistogram(i + 1) else x_i
            )
        return np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")  # type: ignore[return-value]

    def cnt2dist(self, *args: Any) -> IX_dataset_3d:
        """Convert count data to distribution data along specified or all axes."""
        iax = [1, 2, 3] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        shapes = {1: (-1, 1, 1), 2: (1, -1, 1), 3: (1, 1, -1)}
        for ax in iax:
            if ax not in (1, 2, 3):
                raise ValueError("IX_dataset_3d:invalid_argument: Axis index must be 1, 2, or 3")
            if wout.ishistogram(ax) and not wout._xyz_distribution[ax - 1]:
                d = np.diff(wout._xyz[ax - 1]).reshape(shapes[ax])
                if wout._signal.ndim < 3 and ax == 3:
                    wout._signal = wout._signal.reshape(_size_3d(wout._signal))
                    wout._error = wout._error.reshape(_size_3d(wout._error))
                elif wout._signal.ndim == 2:
                    d = d.squeeze(axis=2)
                wout._signal = wout._signal / d
                wout._error = wout._error / d
            wout._xyz_distribution[ax - 1] = True
        return wout

    def dist2cnt(self, *args: Any) -> IX_dataset_3d:
        """Convert distribution data to count data along specified or all axes."""
        iax = [1, 2, 3] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        shapes = {1: (-1, 1, 1), 2: (1, -1, 1), 3: (1, 1, -1)}
        for ax in iax:
            if ax not in (1, 2, 3):
                raise ValueError("IX_dataset_3d:invalid_argument: Axis index must be 1, 2, or 3")
            if wout.ishistogram(ax) and wout._xyz_distribution[ax - 1]:
                d = np.diff(wout._xyz[ax - 1]).reshape(shapes[ax])
                if wout._signal.ndim < 3 and ax == 3:
                    wout._signal = wout._signal.reshape(_size_3d(wout._signal))
                    wout._error = wout._error.reshape(_size_3d(wout._error))
                elif wout._signal.ndim == 2:
                    d = d.squeeze(axis=2)
                wout._signal = wout._signal * d
                wout._error = wout._error * d
            wout._xyz_distribution[ax - 1] = False
        return wout

    def point2hist(self, *args: Any) -> IX_dataset_3d:
        """Convert point axes to histogram axes."""
        iax = [1, 2, 3] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        shapes = {1: (-1, 1, 1), 2: (1, -1, 1), 3: (1, 1, -1)}
        for ax in iax:
            if ax not in (1, 2, 3):
                raise ValueError("IX_dataset_3d:invalid_argument: Axis index must be 1, 2, or 3")
            if not wout.ishistogram(ax):
                xb, ok, mess = bin_boundaries_simple(wout._xyz[ax - 1])
                if not ok:
                    raise ValueError(f"IX_dataset_3d:invalid_argument: {mess}")
                if not wout._xyz_distribution[ax - 1]:
                    d = np.diff(xb).reshape(shapes[ax])
                    if wout._signal.ndim < 3 and ax == 3:
                        wout._signal = wout._signal.reshape(_size_3d(wout._signal))
                        wout._error = wout._error.reshape(_size_3d(wout._error))
                    elif wout._signal.ndim == 2:
                        d = d.squeeze(axis=2)
                    wout._signal = wout._signal * d
                    wout._error = wout._error * d
                wout._xyz[ax - 1] = xb
        return wout

    def hist2point(self, *args: Any) -> IX_dataset_3d:
        """Convert histogram axes to point axes (at bin centres)."""
        iax = [1, 2, 3] if len(args) == 0 else [int(x) for x in np.asarray(args[0]).ravel()]
        wout = self.copy()
        shapes = {1: (-1, 1, 1), 2: (1, -1, 1), 3: (1, 1, -1)}
        for ax in iax:
            if ax not in (1, 2, 3):
                raise ValueError("IX_dataset_3d:invalid_argument: Axis index must be 1, 2, or 3")
            if wout.ishistogram(ax):
                x = wout._xyz[ax - 1]
                if not wout._xyz_distribution[ax - 1]:
                    d = np.diff(x).reshape(shapes[ax])
                    if wout._signal.ndim < 3 and ax == 3:
                        wout._signal = wout._signal.reshape(_size_3d(wout._signal))
                        wout._error = wout._error.reshape(_size_3d(wout._error))
                    elif wout._signal.ndim == 2:
                        d = d.squeeze(axis=2)
                    wout._signal = wout._signal / d
                    wout._error = wout._error / d
                wout._xyz[ax - 1] = 0.5 * (x[1:] + x[:-1])
        return wout

    # ------------------------------------------------------------------
    # Rebinning, Integrating, Cutting, Manipulation Methods
    # ------------------------------------------------------------------

    @staticmethod
    def rebin_hist(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rebin_nd_hist(x, s, e, xout, iax=iax, ndim=3)

    @staticmethod
    def integrate_points(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return integrate_nd_points(x, s, e, xout, iax=iax, ndim=3)

    def rebin(self, *args: Any) -> IX_dataset_3d:
        return self.rebin_xyz(*args)

    def rebin2(self, *args: Any) -> IX_dataset_3d:
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(False, [1, 2, 3], *args)

    def rebin_x(self, *args: Any) -> IX_dataset_3d:
        return self._rebin_xyz(True, 1, *args)

    def rebin_y(self, *args: Any) -> IX_dataset_3d:
        return self._rebin_xyz(True, 2, *args)

    def rebin_z(self, *args: Any) -> IX_dataset_3d:
        return self._rebin_xyz(True, 3, *args)

    def rebin_xy(self, *args: Any) -> IX_dataset_3d:
        return self._rebin_xyz(True, [1, 2], *args)

    def rebin_xyz(self, *args: Any) -> IX_dataset_3d:
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(True, [1, 2, 3], *args)

    def integrate(self, *args: Any) -> Any:
        return self.integrate_xyz(*args)

    def integrate2(self, *args: Any) -> IX_dataset_3d:
        opt = {
            "empty_is_full_range": True,
            "range_is_one_bin": True,
            "array_is_descriptor": False,
            "bin_boundaries": True,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, True, True, [1, 2, 3], opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset_3d:invalid_argument: {mess}")
        return wout

    def integrate_x(self, *args: Any) -> Any:
        return self._integrate_xyz(True, 1, *args)

    def integrate_y(self, *args: Any) -> Any:
        return self._integrate_xyz(True, 2, *args)

    def integrate_z(self, *args: Any) -> Any:
        return self._integrate_xyz(True, 3, *args)

    def integrate_xy(self, *args: Any) -> Any:
        return self._integrate_xyz(True, [1, 2], *args)

    def integrate_xyz(self, *args: Any) -> Any:
        return self._integrate_xyz(True, [1, 2, 3], *args)

    def cut(self, *args: Any) -> Any:
        return self.cut_xyz(*args)

    def cut_x(self, *args: Any) -> Any:
        return self._cut_xyz(1, *args)

    def cut_y(self, *args: Any) -> Any:
        return self._cut_xyz(2, *args)

    def cut_z(self, *args: Any) -> Any:
        return self._cut_xyz(3, *args)

    def cut_xy(self, *args: Any) -> Any:
        return self._cut_xyz([1, 2], *args)

    def cut_xyz(self, *args: Any) -> Any:
        return self._cut_xyz([1, 2, 3], *args)

    def deriv_x(self) -> IX_dataset_3d:
        wout = self.copy()
        xc = (
            0.5 * (self._xyz[0][1:] + self._xyz[0][:-1])
            if self.ishistogram(1)
            else self._xyz[0]
        )
        wout._signal, wout._error = deriv_xye_n(1, xc, self._signal, self._error)
        return wout

    def deriv_y(self) -> IX_dataset_3d:
        wout = self.copy()
        yc = (
            0.5 * (self._xyz[1][1:] + self._xyz[1][:-1])
            if self.ishistogram(2)
            else self._xyz[1]
        )
        wout._signal, wout._error = deriv_xye_n(2, yc, self._signal, self._error)
        return wout

    def deriv_z(self) -> IX_dataset_3d:
        wout = self.copy()
        zc = (
            0.5 * (self._xyz[2][1:] + self._xyz[2][:-1])
            if self.ishistogram(3)
            else self._xyz[2]
        )
        sig_3d = self._signal.reshape(_size_3d(self._signal))
        err_3d = self._error.reshape(_size_3d(self._error))
        s_out, e_out = deriv_xye_n(3, zc, sig_3d, err_3d)
        wout._signal = s_out.reshape(self._signal.shape)
        wout._error = e_out.reshape(self._error.shape)
        return wout

    def deriv(self, iax: int = 1) -> IX_dataset_3d:
        if iax == 1:
            return self.deriv_x()
        if iax == 2:
            return self.deriv_y()
        if iax == 3:
            return self.deriv_z()
        raise ValueError("IX_dataset_3d:invalid_argument: iax must be 1, 2, or 3")

    def unspike(
        self,
        ymin: float | None = None,
        ymax: float | None = None,
        fac: float | None = None,
        sfac: float | None = None,
    ) -> IX_dataset_3d:
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

    def func_eval(self, func_handle: Callable[..., Any], *args: Any) -> IX_dataset_3d:
        all_flag = False
        pars = list(args)
        if len(pars) >= 2 and isinstance(pars[-1], str) and _is_stringmatchi(pars[-1], "all"):
            all_flag = True
            pars = pars[:-1]
        if len(pars) == 1 and isinstance(pars[0], (list, tuple)):
            pars = list(pars[0])

        wout = self.copy()
        X, Y, Z = self.sigvar_getx()
        s_eval = np.asarray(
            func_handle(X.ravel(), Y.ravel(), Z.ravel(), *pars), dtype=float
        ).reshape(self._signal.shape)
        if not all_flag:
            nan_mask = np.isnan(self._signal)
            s_eval = np.where(nan_mask, np.nan, s_eval)
        wout._signal = s_eval
        wout._error = np.zeros_like(wout._signal)
        return wout

    def noisify(self, *args: Any, **kwargs: Any) -> IX_dataset_3d:
        from pyhorace.utilities.math import noisify as _noisify

        wout = self.copy()
        s_noisy, e_noisy = _noisify(
            self._signal, self._error, *args, **kwargs
        )
        wout._signal = np.asarray(s_noisy, dtype=float).reshape(self._signal.shape)
        wout._error = np.asarray(e_noisy, dtype=float).reshape(self._error.shape)
        return wout

    def scale(
        self, xscale: float = 1.0, yscale: float = 1.0, zscale: float = 1.0
    ) -> IX_dataset_3d:
        return self.scale_x(xscale).scale_y(yscale).scale_z(zscale)

    def scale_x(self, xscale: float) -> IX_dataset_3d:
        sc = float(np.asarray(xscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_3d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[0] = wout._xyz[0] * sc
        else:
            wout._xyz[0] = (wout._xyz[0] * sc)[::-1].copy()
            wout._signal = np.flip(wout._signal, axis=0).copy()
            wout._error = np.flip(wout._error, axis=0).copy()
        if wout.ishistogram(1) and wout.x_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def scale_y(self, yscale: float) -> IX_dataset_3d:
        sc = float(np.asarray(yscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_3d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[1] = wout._xyz[1] * sc
        else:
            wout._xyz[1] = (wout._xyz[1] * sc)[::-1].copy()
            wout._signal = np.flip(wout._signal, axis=1).copy()
            wout._error = np.flip(wout._error, axis=1).copy()
        if wout.ishistogram(2) and wout.y_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def scale_z(self, zscale: float) -> IX_dataset_3d:
        sc = float(np.asarray(zscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_3d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[2] = wout._xyz[2] * sc
        else:
            wout._xyz[2] = (wout._xyz[2] * sc)[::-1].copy()
            if wout._signal.ndim >= 3:
                wout._signal = np.flip(wout._signal, axis=2).copy()
                wout._error = np.flip(wout._error, axis=2).copy()
        if wout.ishistogram(3) and wout.z_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def shift(
        self, xshift: float = 0.0, yshift: float = 0.0, zshift: float = 0.0
    ) -> IX_dataset_3d:
        return self.shift_x(xshift).shift_y(yshift).shift_z(zshift)

    def shift_x(self, xshift: float = 0.0) -> IX_dataset_3d:
        wout = self.copy()
        wout._xyz[0] = wout._xyz[0] + float(np.asarray(xshift).item())
        return wout

    def shift_y(self, yshift: float = 0.0) -> IX_dataset_3d:
        wout = self.copy()
        wout._xyz[1] = wout._xyz[1] + float(np.asarray(yshift).item())
        return wout

    def shift_z(self, zshift: float = 0.0) -> IX_dataset_3d:
        wout = self.copy()
        wout._xyz[2] = wout._xyz[2] + float(np.asarray(zshift).item())
        return wout

    def xye(self) -> dict[str, Any]:
        """Return dict ``{'x': [xc, yc, zc], 'y': signal, 'e': error}``."""
        coords = []
        for i in range(3):
            x_i = self._xyz[i]
            xc_i = 0.5 * (x_i[1:] + x_i[:-1]) if self.ishistogram(i + 1) else x_i.copy()
            coords.append(xc_i.reshape(-1, 1))
        return {
            "x": coords,
            "y": self._signal.copy(),
            "e": self._error.copy(),
        }

    def make_label(self) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return ``(xlabel, ylabel, zlabel, slabel)`` formatted axis labels."""
        xlabel, _ = self._xyz_axis[0].make_label()
        ylabel, _ = self._xyz_axis[1].make_label()
        zlabel, _ = self._xyz_axis[2].make_label()
        slabel, units_appended = self._s_axis.make_label()
        denom_parts = []
        for i in range(3):
            if self._xyz_distribution[i] and self._xyz_axis[i].units:
                denom_parts.append(f" / {self._xyz_axis[i].units}")
        denom = "".join(denom_parts)
        if denom:
            if slabel:
                if units_appended:
                    slabel[-1] = f"{slabel[-1][:-1]}{denom})"
                else:
                    slabel[-1] = f"{slabel[-1]} (1{denom})"
            else:
                slabel = [f"(1{denom})"]
        return xlabel, ylabel, zlabel, slabel

    def expand_d1d(self) -> DataOpArray:
        """Expand 3D dataset into an array of 1D datasets along y and z."""
        from pyhorace.data.ix_dataset_1d import IX_dataset_1d
        return IX_dataset_1d._build_from_ix_data_3d(self)

    def expand_d2d(self) -> DataOpArray:
        """Expand 3D dataset into an array of 2D datasets along z."""
        from pyhorace.data.ix_dataset_2d import IX_dataset_2d
        return IX_dataset_2d._build_from_ix_data_3d(self)

    def contract_d3d(self) -> IX_dataset_3d:
        return self.copy()

    # ------------------------------------------------------------------
    # 3D Graphics / Plotting Methods (delegating to graphics.py)
    # ------------------------------------------------------------------

    def sliceomatic(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        from pyhorace.data.graphics import plot_3d_dataset

        datasets = _dataset_array if _dataset_array is not None else [self]
        return plot_3d_dataset(datasets, False, *args)

    def sliceomatic_overview(
        self, *args: Any, _dataset_array: Any = None, **kwargs: Any
    ) -> Any:
        from pyhorace.data.graphics import plot_3d_dataset

        datasets = _dataset_array if _dataset_array is not None else [self]
        return plot_3d_dataset(datasets, True, *args)

    @classmethod
    def loadobj(cls, S: dict[str, Any] | IX_dataset_3d) -> Any:
        if isinstance(S, cls):
            return S
        obj = cls()
        return Serializable.loadobj(S, obj)


IX_data_3d = IX_dataset_3d
register_serializable(IX_data_3d)
