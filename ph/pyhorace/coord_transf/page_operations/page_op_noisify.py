"""Single pixel page operation used by ``noisify`` function.

Ported from ``horace_core/sqw/page_operations/PageOp_noisify.m`` and
``herbert_core/utilities/maths/noisify.m``.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config


def randn(size: tuple[int, ...] | list[int] | int) -> np.ndarray:
    """Standard normal random number generator matching MATLAB ``randn(size)``."""
    if isinstance(size, (tuple, list, np.ndarray)):
        shape = tuple(int(s) for s in size)
        return np.random.standard_normal(shape)
    n = int(size)
    return np.random.standard_normal((n, n))


def _parse_noisify_args(
    y: Any,
    e: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[float, bool, Any, Callable[..., Any], dict[str, Any]]:
    """Parse arguments for ``noisify`` (normal and driven modes)."""
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
        in_dat = args[0]
        if "driven_mode" not in in_dat:
            raise ValueError(
                "HERBERT:utilities:invalid_argument: "
                "A structure is provided as input of noisify routine but "
                "the structure does not contain correct initialization information."
            )
        return (
            float(in_dat["fac"]),
            bool(in_dat["is_poisson"]),
            in_dat["ymax"],
            in_dat["randfunc"],
            in_dat,
        )

    fac: float = 0.1
    use_local_max = -np.inf
    ymax: Any = use_local_max
    is_poisson: bool = False
    randfunc: Callable[..., Any] = randn

    idx = 0
    if idx < len(args):
        first = args[idx]
        if isinstance(first, (int, float, np.number)) and not isinstance(first, bool):
            fac = float(first)
            idx += 1
        elif isinstance(first, str):
            low = first.lower()
            if low == "poisson":
                is_poisson = True
                idx += 1
            elif low in ("maximum_value", "random_number_function"):
                pass
            else:
                raise ValueError(
                    "HERBERT:noisify: 3rd argument cannot be interpreted as a "
                    "Gaussian factor or legal probability distribution"
                )
        else:
            raise ValueError(
                "HERBERT:noisify: 3rd argument cannot be interpreted as a "
                "Gaussian factor or legal probability distribution"
            )

    while idx < len(args):
        key = args[idx]
        if not isinstance(key, str) or idx + 1 >= len(args):
            raise ValueError(f"HERBERT:noisify: Invalid parameter argument: {key!r}")
        val = args[idx + 1]
        k_low = key.lower()
        if k_low == "maximum_value":
            ymax = float(val)
        elif k_low == "random_number_function":
            if not callable(val):
                raise ValueError("HERBERT:noisify: random_number_function must be callable")
            randfunc = val
        else:
            raise ValueError(f"HERBERT:noisify: Unrecognized parameter {key!r}")
        idx += 2

    if "poisson" in kwargs:
        is_poisson = bool(kwargs["poisson"])
    if "factor" in kwargs:
        fac = float(kwargs["factor"])
    if "maximum_value" in kwargs:
        ymax = float(kwargs["maximum_value"])
    if "random_number_function" in kwargs:
        randfunc = kwargs["random_number_function"]

    y_arr = np.asarray(y, dtype=np.float64) if y is not None else np.empty(0, dtype=np.float64)
    if not is_poisson and (isinstance(ymax, (int, float, np.floating)) and ymax == use_local_max):
        if y_arr.size == 0:
            ymax = np.array([], dtype=np.float64)
        else:
            ymax = float(np.max(np.abs(y_arr)))

    out_par: dict[str, Any] = {
        "driven_mode": True,
        "fac": fac,
        "is_poisson": is_poisson,
        "ymax": ymax,
        "randfunc": randfunc,
    }
    return fac, is_poisson, ymax, randfunc, out_par


def noisify(
    y: Any,
    e: Any = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Add noise to ``y`` values and associated variance ``e``.

    Ported from ``herbert_core/utilities/maths/noisify.m``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any]]
        ``(yout, eout, outpar)``
    """
    fac, is_poisson, ymax, randfunc, outpar = _parse_noisify_args(y, e, *args, **kwargs)
    y_arr = np.asarray(y, dtype=np.float64) if y is not None else np.empty(0, dtype=np.float64)
    if y_arr.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), outpar

    if is_poisson:
        lam = np.abs(y_arr)
        yout = np.random.poisson(lam).astype(np.float64)
        eout = np.abs(y_arr).astype(np.float64)
    else:
        ymax_val = float(ymax) if np.asarray(ymax).size > 0 else float(np.max(np.abs(y_arr)))
        if y_arr.ndim == 1:
            matlab_size: tuple[int, ...] = (1, int(y_arr.size))
        else:
            matlab_size = y_arr.shape
        raw_rnd = np.asarray(randfunc(matlab_size), dtype=np.float64)
        dy = (fac * ymax_val) * raw_rnd.reshape(y_arr.shape, order="F")
        yout = y_arr + dy
        eout = np.ones_like(y_arr, dtype=np.float64) * ((fac * ymax_val) ** 2)

    if e is not None:
        e_arr = np.asarray(e, dtype=np.float64)
        if e_arr.size > 0:
            if e_arr.shape != eout.shape and e_arr.size == eout.size:
                e_arr = e_arr.reshape(eout.shape, order="F")
            if e_arr.shape != eout.shape:
                raise ValueError(
                    "HERBERT:noisify: Input array of error bars must have same size as input y array"
                )
            eout = eout + e_arr

    return yout, eout, outpar


class PageOp_noisify(PageOpBase):
    """Single pixel page operation used by ``noisify`` function.

    Ported from ``horace_core/sqw/page_operations/PageOp_noisify.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._op_name = "noisify"
        self.split_at_bin_edges = False
        self.noisify_par: dict[str, Any] = {}

    def init(self, sqw_obj: Any = None, *args: Any, **kwargs: Any) -> "PageOp_noisify":
        """Initialize ``PageOp_noisify`` on ``sqw_obj`` with noisify arguments."""
        super().init(sqw_obj)
        if sqw_obj is not None and not self.changes_pix_only:
            self._var_acc = np.zeros(
                int(np.asarray(self.npix).size), dtype=np.float64, order="F"
            )
        _, _, self.noisify_par = noisify([], [], *args, **kwargs)
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_noisify":
        """Apply ``noisify`` to current pixel page and update image accumulators."""
        signal, var, _ = noisify(
            self._page_data[self.signal_idx, :],
            self._page_data[self.var_idx, :],
            self.noisify_par,
        )
        sig_flat = np.asarray(signal, dtype=np.float64).ravel(order="F")
        var_flat = np.asarray(var, dtype=np.float64).ravel(order="F")
        self._page_data[self.signal_idx, :] = sig_flat
        self._page_data[self.var_idx, :] = var_flat

        self.update_img_accumulators(npix_block, npix_idx, sig_flat, var_flat)
        return self

    def print_range_warning(
        self, infile_name: str, is_old_file_format: bool
    ) -> None:
        """Print missing range message specific for ``noisify`` function."""
        fn_fe = os.path.basename(infile_name or "")
        if is_old_file_format:
            upgrade_message = self.gen_old_file_message(infile_name)
        else:
            upgrade_message = self.gen_misaligned_file_message(infile_name)
        sys.stderr.write(
            f"\n*** Source SQW file {fn_fe} does not contain correct pixel data ranges.\n"
            f"*** noisify uses these ranges so pre-calculates them using recompute_bin_data algorithm.\n"
            f"*** Upgrade your original sqw object to contain these averages\n"
            f"    and not to recalculate them each time the averages are requested\n"
            f"{upgrade_message}"
        )

    def get_do_missing_range_warning(self) -> bool:
        """``noisify`` calculates missing range beforehand, so suppress warning in ``finish_op``."""
        return False

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "noisify_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_noisify":
        log = log_config()
        log.noisify_split_ratio = val
        return self

