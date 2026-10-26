"""1D Probability Distribution Function lookup table class and statistical utilities.

Ported from Herbert MATLAB @pdf_table class and math utilities:
- herbert_core/utilities/classes/@pdf_table/pdf_table.m
- herbert_core/utilities/classes/@pdf_table/IX_dataset_1d.m
- herbert_core/utilities/classes/@pdf_table/mean.m
- herbert_core/utilities/classes/@pdf_table/var.m
- herbert_core/utilities/classes/@pdf_table/width.m
- herbert_core/utilities/classes/@pdf_table/rand.m
- herbert_core/utilities/classes/@pdf_table/retain.m
- herbert_core/utilities/classes/@pdf_table/private/moments_.m
- herbert_core/utilities/classes/@pdf_table/private/convert_old_struct_.m
- herbert_core/utilities/maths/upper_index.m
- herbert_core/utilities/maths/sampling_statistics.m
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.serializable import Serializable
from pyhorace.data.ix_dataset_1d import IX_dataset_1d


def upper_index(arr: np.ndarray, val: Union[float, np.ndarray]) -> np.ndarray:
    """
    Return the largest 0-based index `m` such that `arr[m] <= val` given a monotonically
    increasing array `arr`. If `val < arr[0]`, returns -1.
    Note: when duplicate values exist in `arr` (e.g. step discontinuities), returns the
    LAST index `m` where `arr[m] <= val`, matching MATLAB `upper_index.m`.
    """
    arr_flat = np.asarray(arr, dtype=float).ravel()
    val_arr = np.asarray(val, dtype=float)
    n = arr_flat.size
    if n == 0:
        return np.full(val_arr.shape, -1, dtype=int)

    # searchsorted(..., side='right') - 1 gives the largest index m where arr[m] <= val
    idx = np.searchsorted(arr_flat, val_arr, side="right") - 1
    idx = np.clip(idx, -1, n - 1)
    return idx


def sampling_statistics(
    X: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Get estimate of statistical quantities for a set of observations `X` of shape `(N, nvar)`.
    Ported from `herbert_core/utilities/maths/sampling_statistics.m`.

    Returns
    -------
    mean_val : (nvar,) ndarray
    sig_mean : (nvar,) ndarray
    sig_val : (nvar,) ndarray
    sig_sig : (nvar,) ndarray
    cov_val : (nvar, nvar) ndarray
    sig_cov : (nvar, nvar) ndarray
    minval : (nvar,) ndarray
    maxval : (nvar,) ndarray
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr[:, np.newaxis]
    if X_arr.ndim == 2 and X_arr.shape[0] < X_arr.shape[1] and X_arr.shape[0] <= 10 and X_arr.shape[1] >= 100:
        X_arr = X_arr.T
    if X_arr.ndim > 2 or X_arr.shape[0] < 2 or X_arr.shape[1] < 1:
        raise ValueError(
            "HERBERT:sampling_statistics:invalid_argument: "
            "Observations must form a 2D array of at least one column and two observations"
        )
    N, nvar = X_arr.shape
    minval = np.min(X_arr, axis=0)
    maxval = np.max(X_arr, axis=0)

    Xmean = np.mean(X_arr, axis=0)
    if nvar == 1:
        Xcov = np.array([[float(np.cov(X_arr[:, 0], ddof=1))]])
    else:
        Xcov = np.cov(X_arr, rowvar=False, ddof=1)
    Xsig = np.sqrt(np.diag(Xcov))

    mean_val = Xmean
    sig_val = Xsig
    cov_val = Xcov
    sig_mean = Xsig / np.sqrt(max(1, N - 1))

    dX = X_arr - Xmean
    mu4 = np.mean(dX**4, axis=0)
    var_var = (mu4 - ((N - 3.0) / (N - 1.0)) * (Xsig**4)) / N
    var_var = np.maximum(var_var, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_sig = np.where(Xsig > 0, np.sqrt(var_var) / (2.0 * Xsig), 0.0)

    dX2 = dX**2
    # Outer product per observation averaged over observations: E[(x_i - mu_i)^2 (x_j - mu_j)^2]
    mean_dX2Y2 = (dX2.T @ dX2) / N
    var_cov = np.maximum((mean_dX2Y2 - Xcov**2) / N, 0.0)
    sig_cov = np.sqrt(var_cov)

    if nvar == 1:
        return (
            float(mean_val[0]),
            float(sig_mean[0]),
            float(sig_val[0]),
            float(sig_sig[0]),
            float(cov_val[0, 0]),
            float(sig_cov[0, 0]),
            float(minval[0]),
            float(maxval[0]),
        )

    return mean_val, sig_mean, sig_val, sig_sig, cov_val, sig_cov, minval, maxval


def _parse_rand_shape(args: Tuple[Any, ...]) -> Tuple[int, ...]:
    """Convert MATLAB-style `rand(obj, ...)` size arguments to a NumPy shape tuple."""
    if len(args) == 0:
        return (1, 1)
    if len(args) == 1:
        a = args[0]
        if isinstance(a, (int, np.integer)):
            n = int(a)
            return (n, n)
        arr = np.asarray(a, dtype=int).ravel()
        if arr.size == 1:
            return (int(arr[0]), int(arr[0]))
        return tuple(int(x) for x in arr)
    return tuple(int(x) for x in args)


class pdf_table(Serializable):
    """
    One-dimensional probability distribution function lookup table.
    Ported from Herbert MATLAB `@pdf_table`.
    """

    _x: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 1), dtype=float))
    _f: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 1), dtype=float))
    _fmax: float = PrivateAttr(default=float("nan"))
    _A: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 1), dtype=float))
    _m: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((0, 1), dtype=float))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._x = np.zeros((0, 1), dtype=float)
        self._f = np.zeros((0, 1), dtype=float)
        self._fmax = float("nan")
        self._A = np.zeros((0, 1), dtype=float)
        self._m = np.zeros((0, 1), dtype=float)
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, pdf_table):
                self._copy_from(loaded)
            return

        self._do_check_combo_arg = False
        try:
            if len(args) == 1 and not kwargs:
                self.x = args[0]
                self.f = 1.0
            elif len(args) == 2 and not kwargs:
                self.x = args[0]
                self.f = args[1]
            elif kwargs and not args:
                if "x" in kwargs:
                    self.x = kwargs["x"]
                if "f" in kwargs:
                    self.f = kwargs["f"]
                elif "x" in kwargs and np.asarray(kwargs["x"]).size == 1:
                    self.f = 1.0
            else:
                raise ValueError(
                    "HERBERT:pdf_table:invalid_argument: Too many or invalid input arguments"
                )
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()

    def _copy_from(self, other: "pdf_table") -> None:
        self._x = other._x.copy()
        self._f = other._f.copy()
        self._fmax = other._fmax
        self._A = other._A.copy()
        self._m = other._m.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, pdf_table):
            return False
        if self.filled != other.filled:
            return False
        if self._x.shape != other._x.shape or self._f.shape != other._f.shape:
            return False
        return bool(
            np.allclose(self._x, other._x, rtol=1e-12, atol=1e-12, equal_nan=True)
            and np.allclose(self._f, other._f, rtol=1e-12, atol=1e-12, equal_nan=True)
        )

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._x

    @x.setter
    def x(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if (
            arr.size == 0
            or arr.ndim > 2
            or (arr.ndim == 2 and min(arr.shape) > 1)
            or np.any(np.isnan(arr))
            or np.any(np.isinf(arr))
        ):
            raise ValueError(
                "HERBERT:pdf_table:invalid_argument: The array of x coordinates must be a numeric vector"
            )
        self._x = arr.reshape(-1, 1)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def f(self) -> np.ndarray:
        return self._f

    @f.setter
    def f(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        # Allow +inf when restoring a scalar delta function from struct
        is_scalar_inf = arr.size == 1 and np.isposinf(arr.flat[0]) and self._x.size == 1
        if (
            arr.size == 0
            or arr.ndim > 2
            or (arr.ndim == 2 and min(arr.shape) > 1)
            or np.any(np.isnan(arr))
            or (np.any(np.isinf(arr)) and not is_scalar_inf)
            or np.any(arr < 0)
        ):
            raise ValueError(
                "HERBERT:pdf_table:invalid_argument: The pdf must be a vector of non-negative numbers"
            )
        self._f = arr.reshape(-1, 1)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def fmax(self) -> float:
        return self._fmax

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def m(self) -> np.ndarray:
        return self._m

    @property
    def npnt(self) -> int:
        return int(self._x.size)

    @property
    def filled(self) -> bool:
        return self._x.size > 0

    # Private property aliases for MATLAB compatibility
    @property
    def x_(self) -> np.ndarray:
        return self._x

    @property
    def f_(self) -> np.ndarray:
        return self._f

    @property
    def fmax_(self) -> float:
        return self._fmax

    @property
    def A_(self) -> np.ndarray:
        return self._A

    @property
    def m_(self) -> np.ndarray:
        return self._m

    # -------------------------------------------------------------------------
    # Validation & Normalisation
    # -------------------------------------------------------------------------

    def check_combo_arg(self) -> "pdf_table":
        if self._x.size == 0 and self._f.size == 0:
            self._fmax = float("nan")
            self._A = np.zeros((0, 1), dtype=float)
            self._m = np.zeros((0, 1), dtype=float)
            return self

        if self._x.size != self._f.size:
            raise ValueError(
                "HERBERT:pdf_table:invalid_argument: The number of x values and function values do not match"
            )

        x_flat = self._x.ravel().copy()
        f_flat = self._f.ravel().copy()

        if x_flat.size > 1:
            dx = np.diff(x_flat)
            if np.any(dx < 0):
                raise ValueError(
                    "HERBERT:pdf_table:invalid_argument: Independent variable values must increase monotonically"
                )
            df = np.diff(f_flat)
            area = 0.5 * (f_flat[1:] + f_flat[:-1]) * dx
            if np.all(dx == 0):
                raise ValueError(
                    "HERBERT:pdf_table:invalid_argument: Independent variable cannot have all values identical; "
                    "must span a range if there are two or more points"
                )
            A = np.concatenate([[0.0], np.cumsum(area)])
            Amax = A[-1]
            if Amax == 0:
                raise ValueError(
                    "HERBERT:pdf_table:invalid_argument: Probability distribution function values are all zero, "
                    "or only non-zero at discontinuities"
                )

            # Remove leading and trailing zero-area segments
            pos_indices = np.where(A > 0)[0]
            ibeg = int(pos_indices[0]) - 1
            max_indices = np.where(A == Amax)[0]
            iend = int(max_indices[0])

            x_flat = x_flat[ibeg : iend + 1]
            f_flat = f_flat[ibeg : iend + 1]
            A = A[ibeg : iend + 1]
            dx = dx[ibeg:iend]
            df = df[ibeg:iend]

            # Remove internal points where adjacent segments both have zero f at both ends
            if df.size >= 3:
                zero_area = (df[1:-1] == 0) & (f_flat[1:-2] == 0)
                if np.any(zero_area):
                    keep = np.ones(x_flat.size, dtype=bool)
                    keep[1:-2][zero_area] = False
                    x_flat = x_flat[keep]
                    f_flat = f_flat[keep]
                    A = A[keep]
                    dx = np.diff(x_flat)
                    df = np.diff(f_flat)

            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(dx == 0, np.inf, df / dx)

            f_flat = f_flat / Amax
            fmax = float(np.max(f_flat))
            A = A / Amax
            A[-1] = 1.0
            m = m / Amax

            self._x = x_flat.reshape(-1, 1)
            self._f = f_flat.reshape(-1, 1)
            self._fmax = fmax
            self._A = A.reshape(-1, 1)
            self._m = m.reshape(-1, 1)
        else:
            if f_flat[0] <= 0:
                raise ValueError(
                    "HERBERT:pdf_table:invalid_argument: The pdf value must be greater than zero for a scalar x value"
                )
            self._x = x_flat.reshape(1, 1)
            self._f = np.array([[np.inf]], dtype=float)
            self._fmax = float("inf")
            self._A = np.array([[1.0]], dtype=float)
            self._m = np.zeros((0, 1), dtype=float)

        return self

    # -------------------------------------------------------------------------
    # Distribution Methods (moments, mean, var, width, rand, retain, IX_dataset_1d)
    # -------------------------------------------------------------------------

    def _moments(self) -> Tuple[float, float]:
        if self._x.size > 1:
            x_flat = self._x.ravel()
            f_flat = self._f.ravel()
            dx = np.diff(x_flat)
            df = np.diff(f_flat)
            xbar = 0.5 * (x_flat[1:] + x_flat[:-1])
            fbar = 0.5 * (f_flat[1:] + f_flat[:-1])

            area = fbar * dx
            ok = area > 0
            dx = dx[ok]
            df = df[ok]
            xbar = xbar[ok]
            area = area[ok]

            area_tot = float(np.sum(area))
            dxsqr = dx * dx
            area_lambda = area * xbar + (df * dxsqr) / 12.0
            x_av = float(np.sum(area_lambda) / area_tot)

            del_xbar = xbar - x_av
            tmp = area * dxsqr / 12.0 + del_xbar * (area * del_xbar + df * dxsqr / 6.0)
            x_var = float(np.sum(tmp) / area_tot)
            return x_av, x_var
        elif self._x.size == 1:
            return float(self._x.flat[0]), 0.0
        else:
            return float("nan"), float("nan")

    def mean(self) -> float:
        """First moment (mean) of the probability distribution."""
        x_av, _ = self._moments()
        return x_av

    def var(self) -> Tuple[float, float]:
        """Return `(x_var, x_av)` (variance and mean of the distribution)."""
        x_av, x_var = self._moments()
        return x_var, x_av

    def width(self, fac: float = 0.5) -> Tuple[float, float, float, float]:
        """
        Return `(w, xmax, xlo, xhi)` where `w` is full width at fraction `fac` of maximum height.
        Ported from `@pdf_table/width.m`.
        """
        if self._x.size > 1:
            x = self._x.ravel()
            f = self._f.ravel()
            fref = fac * self._fmax

            above = np.where(f > fref)[0]
            if above.size == 0:
                xlo = float(x[0])
                xhi = float(x[-1])
            else:
                ilo = int(above[0])
                if ilo > 0:
                    xlo = float(
                        (x[ilo - 1] * (f[ilo] - fref) + x[ilo] * (fref - f[ilo - 1]))
                        / (f[ilo] - f[ilo - 1])
                    )
                else:
                    xlo = float(x[0])

                ihi = int(above[-1])
                if ihi < x.size - 1:
                    xhi = float(
                        (x[ihi] * (fref - f[ihi + 1]) + x[ihi + 1] * (f[ihi] - fref))
                        / (f[ihi] - f[ihi + 1])
                    )
                else:
                    xhi = float(x[-1])

            w = xhi - xlo
            xmid = 0.5 * (xlo + xhi)
            ind = np.where(f == self._fmax)[0]
            dx = np.abs(x[ind] - xmid)
            ix = np.where(dx == np.min(dx))[0]
            xmax = float(np.mean(x[ind[ix]]))
            return w, xmax, xlo, xhi
        elif self._x.size == 1:
            xmax = float(self._x.flat[0])
            return 0.0, xmax, xmax, xmax
        else:
            nan = float("nan")
            return nan, nan, nan, nan

    def rand(self, *args: Any, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate random numbers from the probability distribution function.
        Ported from `@pdf_table/rand.m`.
        """
        shape = _parse_rand_shape(args)
        if self._x.size > 1:
            if rng is not None:
                Asamp = rng.random(shape)
            else:
                Asamp = np.random.rand(*shape)
            xx = self._x.ravel()
            ff = self._f.ravel()
            AA = self._A.ravel()
            mm = self._m.ravel()

            Asamp_flat = Asamp.ravel(order="F")
            ix = upper_index(AA, Asamp_flat)
            # Clip ix to [0, len(mm)-1] so Asamp == 1.0 uses the last segment
            ix = np.clip(ix, 0, mm.size - 1)
            twice_dA = 2.0 * (Asamp_flat - AA[ix])
            ffix = ff[ix]
            disc = np.maximum(ffix**2 + mm[ix] * twice_dA, 0.0)
            denom = ffix + np.sqrt(disc)
            with np.errstate(divide="ignore", invalid="ignore"):
                X_flat = np.where(denom > 0, xx[ix] + twice_dA / denom, xx[ix])
            return X_flat.reshape(shape, order="F")
        elif self._x.size == 1:
            return float(self._x.flat[0]) * np.ones(shape, dtype=float)
        else:
            return np.full(shape, np.nan, dtype=float)

    def retain(
        self, x: Union[float, np.ndarray], rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Retain independent variable values `x` according to the probability distribution
        using rejection ratio sampling. Ported from `@pdf_table/retain.m`.
        """
        if not self.filled:
            raise ValueError(
                "HERBERT:pdf_table:uninitialised: The probability distribution function is not initialised"
            )
        x_arr = np.asarray(x, dtype=float)
        x_table = self._x.ravel()
        f_table = self._f.ravel()

        ok = (x_arr >= x_table[0]) & (x_arr <= x_table[-1])
        if np.any(ok):
            f_interp = np.interp(x_arr[ok], x_table, f_table)
            u = (
                rng.random(f_interp.shape)
                if rng is not None
                else np.random.rand(*f_interp.shape)
            )
            ok_sub = f_interp >= self._fmax * u
            ok = ok.copy()
            ok[ok] = ok_sub
        return ok

    def IX_dataset_1d(self) -> Tuple[IX_dataset_1d, IX_dataset_1d]:
        """
        Return `(wpdf, wcdf)` as `IX_dataset_1d` objects.
        Ported from `@pdf_table/IX_dataset_1d.m`.
        """
        if not self.filled:
            raise ValueError(
                "HERBERT:pdf_table:uninitialised: The probability distribution function is not initialised"
            )
        x_1d = self._x.ravel()
        f_1d = self._f.ravel()
        A_1d = self._A.ravel()
        wpdf = IX_dataset_1d(
            x_1d,
            f_1d,
            np.zeros_like(f_1d),
            "Probability distribution function",
            "Random variable",
            "Function value",
        )
        wcdf = IX_dataset_1d(
            x_1d,
            A_1d,
            np.zeros_like(A_1d),
            "Cumulative distribution function",
            "Random variable",
            "Function value",
        )
        return wpdf, wcdf

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> Sequence[str]:
        return ("x", "f")

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "pdf_table"]:
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "x_" in inputs:
            inputs.pop("class_version_", None)
            inputs.pop("pdf_", None)
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs
        return inputs, self


__all__ = ["pdf_table", "sampling_statistics", "upper_index"]

