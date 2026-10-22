"""IX_dataset_1d / IX_data_1d class definition.

Translated from:
- ``@IX_data_1d/IX_data_1d.m`` and ``@IX_data_1d/*.m`` (including ``private/``)
- ``@IX_dataset_1d/IX_dataset_1d.m`` and ``@IX_dataset_1d/*.m``
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
    deriv_xye,
    integrate_nd_points,
    rebin_nd_hist,
    unspike_xye,
)
from pyhorace.data.sigvar import SigVar


@register_serializable
class IX_dataset_1d(IX_dataset):
    """1D dataset with x-axis, signal, error, and axis annotations.

    Combines ``@IX_data_1d`` and ``@IX_dataset_1d``.
    """

    _fields_to_save = [
        "title",
        "signal",
        "error",
        "s_axis",
        "x",
        "x_axis",
        "x_distribution",
    ]

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, IX_dataset) and arg0.ndim() == 2:
                return cls._build_from_ix_data_2d(arg0)
            if isinstance(arg0, IX_dataset) and arg0.ndim() == 3:
                return cls._build_from_ix_data_3d(arg0)
            if isinstance(arg0, DataOpArray) and arg0.size > 0:
                first = arg0.ravel(order="F")[0]
                if isinstance(first, IX_dataset) and first.ndim() == 2:
                    parts = [cls._build_from_ix_data_2d(el) for el in arg0.ravel(order="F")]
                    return DataOpArray(np.concatenate([p.ravel(order="F") for p in parts]))
                if isinstance(first, IX_dataset) and first.ndim() == 3:
                    parts = [cls._build_from_ix_data_3d(el) for el in arg0.ravel(order="F")]
                    return DataOpArray(np.concatenate([p.ravel(order="F") for p in parts]))
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._xyz = [np.zeros((0,), dtype=float)]
        self._xyz_axis = [IX_axis()]
        self._xyz_distribution = np.array([True], dtype=bool)
        self._signal = np.zeros((1, 0), dtype=float)
        self._error = np.zeros((1, 0), dtype=float)
        self._do_check_combo_arg = True

        if len(args) > 0:
            self._build_ixdataset_1d(*args)
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

    def init(self, *args: Any, **kwargs: Any) -> IX_dataset_1d:
        """Re-initialise dataset in-place."""
        self.__init__(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        """x-axis coordinates or bin boundaries (row/1D vector)."""
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
        """True if signal is a distribution along x, False if bin counts."""
        return bool(self._xyz_distribution[0])

    @x_distribution.setter
    def x_distribution(self, val: Any) -> None:
        self._xyz_distribution[0] = bool(np.asarray(val).item() if isinstance(val, np.ndarray) else val)

    # ------------------------------------------------------------------
    # Validation & Construction
    # ------------------------------------------------------------------

    @staticmethod
    def ndim() -> int:
        return 1

    def _check_and_set_sig_err(self, field_name: str, val: Any) -> None:
        if _is_empty(val) and not isinstance(val, str):
            arr = np.zeros((1, 0), dtype=float)
        else:
            if not _is_numeric_array(val):
                raise ValueError(
                    f"IX_dataset_1d:invalid_argument: {field_name}- array must be a numeric vector"
                )
            arr = np.asarray(val, dtype=float)
            if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
                raise ValueError(
                    f"IX_dataset_1d:invalid_argument: {field_name}- array must be a numeric vector"
                )
            arr = arr.reshape(-1, 1)
        setattr(self, f"_{field_name}", arr)

    def check_combo_arg(self) -> IX_dataset_1d:
        """Validate interdependent properties ``x``, ``signal``, and ``error``."""
        if self._signal.size != self._error.size:
            raise ValueError(
                f"HERBERT:IX_data_1d:invalid_argument: numel(signal)={self._signal.size}, numel(error)={self._error.size}; numel(signal)~=numel(error)"
            )
        nx = self._xyz[0].size
        ns = self._signal.size
        if not (nx == ns or nx == ns + 1):
            raise ValueError(
                f"HERBERT:IX_data_1d:invalid_argument: numel(signal)={ns}, numel(x)={nx}; numel(signal)  must be equal to numel(x) or numel(x)+1"
            )
        if nx == 0:
            self._signal = np.zeros((1, 0), dtype=float)
            self._error = np.zeros((1, 0), dtype=float)
        return self

    def _build_ixdataset_1d(self, *args: Any) -> None:
        narg = len(args)
        prev = self._do_check_combo_arg
        self._do_check_combo_arg = False
        try:
            if narg == 0:
                return
            if narg == 1:
                in_val = args[0]
                if isinstance(in_val, IX_dataset_1d):
                    self._title = list(in_val._title) if isinstance(in_val._title, list) else in_val._title
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
                if isinstance(in_val, SigVar):
                    s_arr = np.asarray(in_val.s, dtype=float)
                    e_arr = np.asarray(in_val.e, dtype=float)
                    self.x = np.arange(1, s_arr.size + 1, dtype=float)
                    self.signal = s_arr.ravel()
                    self.error = np.sqrt(e_arr.ravel())
                elif _is_numeric_array(in_val):
                    arr = np.asarray(in_val, dtype=float)
                    if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] > 1:
                        self.x = arr[0, :]
                        self.signal = arr[1, :]
                        self.error = arr[2, :]
                    else:
                        self.x = arr
                        n = self._xyz[0].size
                        self.signal = np.zeros((n, 1), dtype=float) if n > 0 else np.zeros((1, 0), dtype=float)
                        self.error = np.zeros((n, 1), dtype=float) if n > 0 else np.zeros((1, 0), dtype=float)
                elif hasattr(in_val, "IX_dataset_1d") and callable(in_val.IX_dataset_1d):
                    res = in_val.IX_dataset_1d()
                    self._build_ixdataset_1d(res)
                    return
                else:
                    raise ValueError(
                        "HERBERT:IX_data_1d:invalid_argument: Check type(s) of input argument(s)"
                    )
            elif narg == 2:
                self.x = args[0]
                self.signal = args[1]
                self.error = np.zeros(self._signal.shape, dtype=float)
            elif narg == 3:
                self.x = args[0]
                self.signal = args[1]
                self.error = args[2]
            elif narg == 4:
                self.x = args[0]
                self.signal = args[1]
                self.error = args[2]
                self.x_distribution = args[3]
            elif narg == 6:
                self.x = args[0]
                self.signal = args[1]
                self.error = args[2]
                self.title = args[3]
                self.x_axis = args[4]
                self.s_axis = args[5]
            elif narg == 7 and (
                isinstance(args[0], str)
                or (isinstance(args[0], (list, tuple)) and all(isinstance(t, str) for t in args[0]))
            ) and not isinstance(args[4], (str, IX_axis)):
                self.title = args[0]
                self.signal = args[1]
                self.error = args[2]
                self.s_axis = args[3]
                self.x = args[4]
                self.x_axis = args[5]
                self.x_distribution = args[6]
            elif narg == 7 and _is_numeric_array(args[0]):
                self.x = args[0]
                self.signal = args[1]
                self.error = args[2]
                self.title = args[3]
                self.x_axis = args[4]
                self.s_axis = args[5]
                self.x_distribution = args[6]
            else:
                raise ValueError(
                    "HERBERT:IX_data_1d:invalid_argument: Wrong number of input arguments"
                )
        finally:
            self._do_check_combo_arg = prev
        self.check_combo_arg()

    @classmethod
    def _build_from_ix_data_2d(cls, in_obj: IX_dataset) -> DataOpArray:
        ny = in_obj._signal.shape[1] if in_obj._signal.ndim >= 2 else 0
        if ny == 0:
            return DataOpArray([cls()])
        out = []
        for i in range(ny):
            w = cls(
                in_obj.title,
                in_obj._signal[:, i],
                in_obj._error[:, i],
                in_obj.s_axis,
                in_obj._xyz[0],
                in_obj._xyz_axis[0],
                in_obj._xyz_distribution[0],
            )
            out.append(w)
        return DataOpArray(out)

    @classmethod
    def _build_from_ix_data_3d(cls, in_obj: IX_dataset) -> DataOpArray:
        sig = in_obj._signal
        err = in_obj._error
        if sig.size == 0:
            return DataOpArray([cls()])
        sz = list(sig.shape) + [1] * (3 - sig.ndim)
        ny, nz = sz[1], sz[2]
        sig_3d = sig.reshape(sz)
        err_3d = err.reshape(sz)
        out = np.empty((ny, nz), dtype=object)
        for j in range(ny):
            for k in range(nz):
                out[j, k] = cls(
                    in_obj.title,
                    sig_3d[:, j, k],
                    err_3d[:, j, k],
                    in_obj.s_axis,
                    in_obj._xyz[0],
                    in_obj._xyz_axis[0],
                    in_obj._xyz_distribution[0],
                )
        return out.view(DataOpArray)

    # ------------------------------------------------------------------
    # Core Dataset Inquiry & Conversion Methods
    # ------------------------------------------------------------------

    def axis(self, n: int | None = None) -> tuple[dict[str, Any], bool]:
        """Return ``(ax_struct, hist_flag)`` for the 1D dataset."""
        if n is not None:
            n_arr = np.asarray(n).ravel()
            if n_arr.size != 1 or int(n_arr[0]) != 1:
                raise ValueError(
                    "HERBERT:IX_data_1d:invalid_argument: Axis index must be 1 for 1D dataset"
                )
        is_hist = self.ishistogram()
        ax_dict = {
            "values": self._xyz[0].copy(),
            "axis": self._xyz_axis[0].copy(),
            "distribution": bool(self._xyz_distribution[0]),
        }
        return ax_dict, is_hist

    def ishistogram(self, n: int | None = None) -> bool:
        """Return True if x-axis is histogram bin boundaries, False if point data."""
        if n is not None:
            n_arr = np.asarray(n).ravel()
            if n_arr.size != 1 or int(n_arr[0]) != 1:
                raise ValueError(
                    "HERBERT:IX_data_1d:invalid_argument: Axis index must be 1 for 1D dataset"
                )
        return bool(self._xyz[0].size == self._signal.size + 1 and self._signal.size > 0) or (
            self._xyz[0].size == 1 and self._signal.size == 0
        )

    def dimensions(self) -> tuple[int, tuple[int, ...]]:
        """Return ``(nd, sz)`` of dataset."""
        if self._xyz[0].size == 0:
            return 0, (1, 0)
        return 1, self._signal.shape

    def get_x(self) -> np.ndarray:
        """Return ``x`` array."""
        return self._xyz[0].copy()

    def get_xsigerr(self) -> dict[str, Any]:
        """Return dictionary with ``x``, ``signal``, ``err``, `` x_distribution``."""
        return {
            "x": self._xyz[0].copy(),
            "signal": self._signal.copy(),
            "err": self._error.copy(),
            "x_distribution": bool(self._xyz_distribution[0]),
        }

    def calc_continuous_fraction(self) -> tuple[float, int]:
        """Calculate the fraction of non-NaN points that have an adjacent non-NaN neighbor."""
        valid = ~np.isnan(self._signal.ravel())
        n_points = int(np.sum(valid))
        if n_points == 0:
            return 0.0, 0
        if n_points == self._signal.size:
            return 1.0, n_points

        idx = np.flatnonzero(valid)
        if idx.size <= 1:
            return 0.0, n_points

        diffs = np.diff(idx)
        adj = np.zeros(idx.size, dtype=bool)
        adj[:-1] |= diffs == 1
        adj[1:] |= diffs == 1
        n_cont = int(np.sum(adj))
        return float(n_cont) / float(n_points), n_points

    def cnt2dist(self, *args: Any) -> IX_dataset_1d:
        """Convert histogram count data to distribution data."""
        wout = self.copy()
        if self.ishistogram() and not self.x_distribution:
            dx = np.diff(wout._xyz[0]).reshape(-1, 1)
            wout._signal = wout._signal / dx
            wout._error = wout._error / dx
            wout._xyz_distribution[0] = True
        elif not self.ishistogram() and not self.x_distribution:
            wout._xyz_distribution[0] = True
        return wout

    def dist2cnt(self, *args: Any) -> IX_dataset_1d:
        """Convert histogram distribution data to count data."""
        wout = self.copy()
        if self.ishistogram() and self.x_distribution:
            dx = np.diff(wout._xyz[0]).reshape(-1, 1)
            wout._signal = wout._signal * dx
            wout._error = wout._error * dx
            wout._xyz_distribution[0] = False
        elif not self.ishistogram() and self.x_distribution:
            wout._xyz_distribution[0] = False
        return wout

    def point2hist(self, *args: Any) -> IX_dataset_1d:
        """Convert point data to histogram data."""
        if len(args) > 0:
            iax = np.asarray(args[0], dtype=int).ravel()
            if iax.size != 1 or iax[0] != 1:
                raise ValueError("IX_dataset_1d:invalid_argument: Invalid axis index")
        if self.ishistogram():
            return self.copy()
        xb, ok, mess = bin_boundaries_simple(self._xyz[0])
        if not ok:
            raise ValueError(f"IX_dataset_1d:invalid_argument: {mess}")
        wout = self.copy()
        if not wout.x_distribution:
            dx = np.diff(xb).reshape(-1, 1)
            wout._signal = wout._signal * dx
            wout._error = wout._error * dx
        wout._xyz[0] = xb
        return wout

    def hist2point(self, *args: Any) -> IX_dataset_1d:
        """Convert histogram data to point data (at bin centres)."""
        if len(args) > 0:
            iax = np.asarray(args[0], dtype=int).ravel()
            if iax.size != 1 or iax[0] != 1:
                raise ValueError("IX_dataset_1d:invalid_argument: Invalid axis index")
        if not self.ishistogram():
            return self.copy()
        wout = self.copy()
        x = wout._xyz[0]
        if not wout.x_distribution:
            dx = np.diff(x).reshape(-1, 1)
            wout._signal = wout._signal / dx
            wout._error = wout._error / dx
        wout._xyz[0] = 0.5 * (x[1:] + x[:-1])
        return wout

    # ------------------------------------------------------------------
    # Rebinning, Integrating, Cutting, Manipulation Methods
    # ------------------------------------------------------------------

    @staticmethod
    def rebin_hist(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rebin_nd_hist(x, s, e, xout, iax=iax, ndim=1)

    @staticmethod
    def integrate_points(
        iax: int, x: np.ndarray, s: np.ndarray, e: np.ndarray, xout: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return integrate_nd_points(x, s, e, xout, iax=iax, ndim=1)

    def rebin(self, *args: Any) -> IX_dataset_1d:
        """Rebin 1D dataset using rebin descriptor(s) or reference dataset."""
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(True, 1, *args)

    def rebin2(self, *args: Any) -> IX_dataset_1d:
        """Rebin 1D dataset using explicit bin boundaries."""
        if len(args) == 0:
            return self.copy()
        return self._rebin_xyz(False, 1, *args)

    def rebin_x(self, *args: Any) -> IX_dataset_1d:
        return self.rebin(*args)

    def integrate(self, *args: Any) -> dict[str, Any]:
        """Integrate 1D dataset over descriptor range, returning ``{'val': val, 'err': err}``."""
        return self._integrate_xyz(True, 1, *args)

    def integrate2(self, *args: Any) -> IX_dataset_1d:
        """Integrate 1D dataset over explicit bin boundaries."""
        opt = {
            "empty_is_full_range": True,
            "range_is_one_bin": True,
            "array_is_descriptor": False,
            "bin_boundaries": True,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, True, True, 1, opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset_1d:invalid_argument: {mess}")
        return wout

    def integrate_x(self, *args: Any) -> dict[str, Any]:
        return self.integrate(*args)

    def cut(self, *args: Any) -> IX_dataset_1d | dict[str, Any]:
        """Take a 1D cut using bin center descriptor ``[xlo, dx, xhi]`` or ``[xlo, xhi]``."""
        opt = {
            "empty_is_full_range": False,
            "range_is_one_bin": True,
            "array_is_descriptor": True,
            "bin_boundaries": False,
        }
        wout, ok, mess = self.rebin_IX_dataset(self, False, False, 1, opt, *args)
        if not ok:
            raise ValueError(f"IX_dataset_1d:invalid_argument: {mess}")
        if len(args) == 1 and _is_numeric_array(args[0]) and np.asarray(args[0]).size == 2:
            return wout.squeeze_IX_dataset(1)
        return wout

    def cut_x(self, *args: Any) -> Any:
        return self.cut(*args)

    def deriv(self) -> IX_dataset_1d:
        """Numerical first derivative of dataset."""
        wout = self.copy()
        xc = self.sigvar_getx().ravel()
        dy, de = deriv_xye(xc, self._signal.ravel(), self._error.ravel())
        wout._signal = dy.reshape(-1, 1)
        wout._error = de.reshape(-1, 1)
        return wout

    def deriv_x(self) -> IX_dataset_1d:
        return self.deriv()

    def unspike(
        self,
        ymin: float | None = None,
        ymax: float | None = None,
        fac: float | None = None,
        sfac: float | None = None,
    ) -> IX_dataset_1d:
        """Remove spikes from 1D dataset and interpolate."""
        wout = self.copy()
        xc = self.sigvar_getx().ravel()
        yo, eo = unspike_xye(
            xc, self._signal.ravel(), self._error.ravel(), ymin, ymax, fac, sfac
        )
        wout._signal = yo.reshape(-1, 1)
        wout._error = eo.reshape(-1, 1)
        return wout

    def func_eval(self, func_handle: Callable[..., Any], *args: Any) -> IX_dataset_1d:
        """Evaluate ``func_handle(x, *pars)`` at bin centers or points."""
        all_flag = False
        pars = list(args)
        if len(pars) >= 2 and isinstance(pars[-1], str) and _is_stringmatchi(pars[-1], "all"):
            all_flag = True
            pars = pars[:-1]
        if len(pars) == 1 and isinstance(pars[0], (list, tuple)):
            pars = list(pars[0])

        wout = self.copy()
        xc = self.sigvar_getx().ravel()
        y_eval = np.asarray(func_handle(xc, *pars), dtype=float).reshape(-1, 1)
        if not all_flag:
            nan_mask = np.isnan(self._signal)
            y_eval = np.where(nan_mask, np.nan, y_eval)
        wout._signal = y_eval
        wout._error = np.zeros_like(wout._signal)
        return wout

    def noisify(self, *args: Any, **kwargs: Any) -> IX_dataset_1d:
        """Add noise to dataset using ``pyhorace.utilities.math.noisify``."""
        from pyhorace.utilities.math import noisify as _noisify

        wout = self.copy()
        s_noisy, e_noisy = _noisify(
            self._signal.ravel(), self._error.ravel(), *args, **kwargs
        )
        wout._signal = np.asarray(s_noisy, dtype=float).reshape(-1, 1)
        wout._error = np.asarray(e_noisy, dtype=float).reshape(-1, 1)
        return wout

    def scale(self, xscale: float) -> IX_dataset_1d:
        """Scale x-axis by scalar factor ``xscale``."""
        sc = float(np.asarray(xscale).item())
        if sc == 0:
            raise ValueError("IX_dataset_1d:invalid_argument: Scale factor must be non-zero")
        wout = self.copy()
        if sc > 0:
            wout._xyz[0] = wout._xyz[0] * sc
        else:
            wout._xyz[0] = (wout._xyz[0] * sc)[::-1].copy()
            wout._signal = wout._signal[::-1].copy()
            wout._error = wout._error[::-1].copy()
        if wout.ishistogram() and wout.x_distribution:
            wout._signal = wout._signal / abs(sc)
            wout._error = wout._error / abs(sc)
        return wout

    def shift(self, xshift: float = 0.0) -> IX_dataset_1d:
        """Shift x-axis by ``xshift``."""
        wout = self.copy()
        wout._xyz[0] = wout._xyz[0] + float(np.asarray(xshift).item())
        return wout

    def smooth(self, *args: Any, **kwargs: Any) -> IX_dataset_1d:
        """Smooth 1D dataset using a convolution kernel."""
        width = int(args[0]) if len(args) >= 1 else 3
        if width <= 1:
            return self.copy()
        kernel = np.ones(width, dtype=float) / float(width)
        wout = self.copy()
        s = self._signal.ravel()
        e = self._error.ravel()
        wout._signal = np.convolve(s, kernel, mode="same").reshape(-1, 1)
        wout._error = np.sqrt(np.convolve(e**2, kernel**2, mode="same")).reshape(-1, 1)
        return wout

    def rebunch(self, nbins: int = 1) -> IX_dataset_1d:
        """Rebunch adjacent bins or points in groups of ``nbins``."""
        nb = int(nbins)
        if nb < 1:
            raise ValueError("IX_dataset_1d:invalid_argument: nbins must be >= 1")
        if nb == 1:
            return self.copy()
        wout = self.copy()
        n_sig = self._signal.size
        if self.ishistogram():
            xb_new = self._xyz[0][0::nb]
            if xb_new[-1] != self._xyz[0][-1]:
                xb_new = np.append(xb_new, self._xyz[0][-1])
            return self.rebin2(xb_new)
        else:
            n_groups = (n_sig + nb - 1) // nb
            x_new = np.zeros(n_groups, dtype=float)
            s_new = np.zeros((n_groups, 1), dtype=float)
            e_new = np.zeros((n_groups, 1), dtype=float)
            for g in range(n_groups):
                sl = slice(g * nb, min((g + 1) * nb, n_sig))
                cnt = sl.stop - sl.start
                x_new[g] = float(np.mean(self._xyz[0][sl]))
                if self.x_distribution:
                    s_new[g, 0] = float(np.mean(self._signal[sl]))
                    e_new[g, 0] = float(np.sqrt(np.sum(self._error[sl] ** 2))) / cnt
                else:
                    s_new[g, 0] = float(np.sum(self._signal[sl]))
                    e_new[g, 0] = float(np.sqrt(np.sum(self._error[sl] ** 2)))
            wout._xyz[0] = x_new
            wout._signal = s_new
            wout._error = e_new
            return wout

    def interp(self, x_new: Any, method: str = "linear") -> IX_dataset_1d:
        """Interpolate dataset onto new point coordinates ``x_new``."""
        xc = self.sigvar_getx().ravel()
        if isinstance(x_new, IX_dataset_1d):
            x_out = x_new.sigvar_getx().ravel()
        else:
            x_out = np.asarray(x_new, dtype=float).ravel()
        s_out = np.interp(x_out, xc, self._signal.ravel(), left=np.nan, right=np.nan)
        e_out = np.zeros_like(s_out)
        wout = self.copy()
        wout._xyz[0] = x_out
        wout._signal = s_out.reshape(-1, 1)
        wout._error = e_out.reshape(-1, 1)
        return wout

    def linspace(self, n: int) -> IX_dataset_1d:
        """Divide each interval in x into ``n`` equal sub-intervals."""
        n_int = int(n)
        if n_int < 1:
            raise ValueError("IX_dataset_1d:invalid_argument: n must be >= 1")
        if n_int == 1 or self._xyz[0].size <= 1:
            return self.copy()
        x = self._xyz[0]
        parts = []
        for i in range(x.size - 1):
            parts.append(np.linspace(x[i], x[i + 1], n_int + 1)[:-1])
        parts.append(np.array([x[-1]]))
        x_new = np.concatenate(parts)
        wout = self.copy()
        if self.ishistogram():
            wout._xyz[0] = x_new
            wout._signal = np.repeat(self._signal, n_int, axis=0)
            wout._error = np.repeat(self._error, n_int, axis=0)
            if not self.x_distribution:
                wout._signal = wout._signal / n_int
                wout._error = wout._error / n_int
        else:
            n_sig_new = x_new.size
            wout._xyz[0] = x_new
            wout._signal = np.zeros((n_sig_new, 1), dtype=float)
            wout._error = np.zeros((n_sig_new, 1), dtype=float)
        return wout

    def mask(self, mask_array: Any) -> IX_dataset_1d:
        """Mask points where ``mask_array`` is False/0 by setting signal=NaN and error=0."""
        msk = np.asarray(mask_array, dtype=bool).ravel()
        if msk.size != self._signal.size:
            raise ValueError(
                "IX_dataset_1d:invalid_argument: Mask array must have same number of elements as signal"
            )
        wout = self.copy()
        wout._signal[~msk] = np.nan
        wout._error[~msk] = 0.0
        return wout

    def moments(self, x_min: float = -np.inf, x_max: float = np.inf) -> dict[str, Any]:
        """Calculate area, mean, and standard deviation of peak."""
        xc = self.sigvar_getx().ravel()
        s = self._signal.ravel()
        e = self._error.ravel()
        if self.ishistogram() and self.x_distribution:
            dx = np.diff(self._xyz[0])
            s = s * dx
            e = e * dx
        valid = (~np.isnan(s)) & (xc >= x_min) & (xc <= x_max)
        if not np.any(valid):
            return {"area": np.nan, "area_err": np.nan, "mean": np.nan, "std": np.nan}
        xv = xc[valid]
        sv = s[valid]
        ev = e[valid]
        area = float(np.sum(sv))
        area_err = float(np.sqrt(np.sum(ev**2)))
        if area == 0:
            return {"area": 0.0, "area_err": area_err, "mean": np.nan, "std": np.nan}
        mean = float(np.sum(xv * sv) / area)
        var = float(np.sum(((xv - mean) ** 2) * sv) / area)
        return {
            "area": area,
            "area_err": area_err,
            "mean": mean,
            "std": float(np.sqrt(max(0.0, var))),
        }

    def xye(self) -> dict[str, np.ndarray]:
        """Return dict ``{'x': xc, 'y': signal, 'e': error}`` with bin centres ``xc``."""
        return {
            "x": self.sigvar_getx(),
            "y": self._signal.copy(),
            "e": self._error.copy(),
        }

    def make_label(self) -> tuple[list[str], list[str]]:
        """Return ``(xlabel, slabel)`` formatted axis labels."""
        xlabel, _ = self._xyz_axis[0].make_label()
        slabel, units_appended = self._s_axis.make_label()
        denom = f" / {self._xyz_axis[0].units}" if (self.x_distribution and self._xyz_axis[0].units) else ""
        if denom:
            if slabel:
                if units_appended:
                    slabel[-1] = f"{slabel[-1][:-1]}{denom})"
                else:
                    slabel[-1] = f"{slabel[-1]} (1{denom})"
            else:
                slabel = [f"(1{denom})"]
        return xlabel, slabel

    def set_simple(
        self, x: Any, signal: Any, err: Any, xdistr: bool | None = None
    ) -> IX_dataset_1d:
        """Fast setter for x, signal, error, and x_distribution."""
        return self.set_simple_xsigerr(1, x, signal, err, xdistr)  # type: ignore[return-value]

    def set_simple_se(self, signal: Any, err: Any) -> IX_dataset_1d:
        """Fast setter for signal and error only."""
        wout = self.copy()
        wout._signal = np.asarray(signal, dtype=float).reshape(-1, 1)
        wout._error = np.asarray(err, dtype=float).reshape(-1, 1)
        return wout

    def expand_d0d(self) -> Any:
        """Placeholder for d0d expansion."""
        return self.copy()

    # ------------------------------------------------------------------
    # 1D Graphics / Plotting Methods (delegating to graphics.py)
    # ------------------------------------------------------------------

    def _plot_1d(
        self,
        plot_type: str,
        new_axes: bool,
        force_current_axes: bool,
        *args: Any,
        _dataset_array: Any = None,
        **kwargs: Any,
    ) -> Any:
        from pyhorace.data.graphics import plot_1d_datasets

        datasets = _dataset_array if _dataset_array is not None else [self]
        return plot_1d_datasets(
            datasets, plot_type, new_axes, force_current_axes, *args
        )

    def dd(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("d", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def de(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("e", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def dh(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("h", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def dl(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("l", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def dm(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("m", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def dp(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("p", True, False, *args, _dataset_array=_dataset_array, **kwargs)

    def pd(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("d", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def pdoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("d", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def pe(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("e", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def peoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("e", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def ph(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("h", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def phoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("h", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def pl(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("l", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def ploc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("l", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def pm(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("m", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def pmoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("m", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    def pp(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("p", False, False, *args, _dataset_array=_dataset_array, **kwargs)

    def ppoc(self, *args: Any, _dataset_array: Any = None, **kwargs: Any) -> Any:
        return self._plot_1d("p", False, True, *args, _dataset_array=_dataset_array, **kwargs)

    @classmethod
    def loadobj(cls, S: dict[str, Any] | IX_dataset_1d) -> Any:
        if isinstance(S, cls):
            return S
        obj = cls()
        return Serializable.loadobj(S, obj)


IX_data_1d = IX_dataset_1d
register_serializable(IX_data_1d)
SERIALIZABLE_ALIAS_1D = "IX_data_1d"
