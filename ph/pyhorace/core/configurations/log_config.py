"""Adaptive logging configuration for PyHorace.

Translated from `herbert_core/configuration/log_config.m`.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .config_base import config_base
from .config_store import HoraceConfigError, config_store, warn_with_id


class log_config(config_base):
    """The class containing information on how often a `page_op` should be
    logged to provide reasonable user experience.

    To see the list of current configuration option values:
        >>> log_config()

    To set values:
        >>> lc = log_config()
        >>> lc.name1 = val1
    or:
        >>> set(log_config, 'name1', val1, 'name2', val2, ...)

    To fetch values:
        >>> val1 = log_config().name1
    or:
        >>> val1, val2, ... = get(log_config, 'name1', 'name2', ...)

    `log_config` properties are:
    ----------------------------
    GENERAL PROPERTIES:
    - `info_log_print_time` -- How often (in sec) progress info log should be printed
      when operation is performed.
    - `info_log_split_ratio` -- This value defines automatic split step, calculated
      during logging from `info_log_print_time` and time spent on doing every step
      of the algorithm.

    PAGEOP SPECIFIC SETTINGS:
    Store logging ratios for various algorithms which use `apply_op` on paging operations:
    - `recompute_bins_split_ratio`
    - `apply_split_ratio`
    - `sqw_binary_double_split_ratio`
    - `sqw_binary_img_split_ratio`
    - `sqw_binary_sqw_split_ratio`
    - `cat_pix_split_ratio`
    - `coord_calc_split_ratio`
    - `func_eval_split_ratio`
    - `join_sqw_split_ratio`
    - `mask_split_ratio`
    - `noisify_split_ratio`
    - `section_split_ratio`
    - `sigvar_set_split_ratio`
    - `split_sqw_split_ratio`
    - `sqw_eval_split_ratio`
    - `unary_op_split_ratio`
    """

    _SAVED_PROPERTIES: List[str] = [
        "info_log_print_time",
        "recompute_bins_split_ratio",
        "apply_split_ratio",
        "sqw_binary_double_split_ratio",
        "sqw_binary_img_split_ratio",
        "sqw_binary_sqw_split_ratio",
        "cat_pix_split_ratio",
        "coord_calc_split_ratio",
        "func_eval_split_ratio",
        "join_sqw_split_ratio",
        "mask_split_ratio",
        "noisify_split_ratio",
        "section_split_ratio",
        "sigvar_set_split_ratio",
        "split_sqw_split_ratio",
        "sqw_eval_split_ratio",
        "unary_op_split_ratio",
    ]

    def __init__(self) -> None:
        super().__init__("log_config")
        self.warn_if_missing_config_ = False
        # normally info log should be printed every 5 minutes
        self.info_log_print_time_: float = 300.0
        for prop in self._SAVED_PROPERTIES[1:]:
            setattr(self, f"{prop}_", 1)

        # helper property which helps organize consistent logging in
        # apply_op routine. It is set to true when log algorithm has
        # printed alive-log dot and to false when other logging has been
        # performed. Introduced to simplify log prints in transitions
        # between different parts of apply_op routine
        self.dot_printed: bool = False
        self.info_log_split_ratio_: int = 1
        self.log_time_holder_: Optional[float] = None

    @property
    def info_log_split_ratio(self) -> int:
        """Automatic split step calculated during logging from `info_log_print_time`
        and time spent on doing every step of the algorithm.
        """
        return self.info_log_split_ratio_

    @property
    def info_log_print_time(self) -> float:
        """How often (in sec) progress info log should be printed when operation is performed."""
        return float(self.get_or_restore_field("info_log_print_time"))

    @info_log_print_time.setter
    def info_log_print_time(self, val: float) -> None:
        if not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise HoraceConfigError(
                "HERBERT:Log_config:invalid_argument",
                f"Only numeric value allowed for info_log_print_time. Provided: {type(val).__name__}",
            )
        if val < 1:  # no point of making it smaller than 1 sec
            val = 1.0
        config_store.instance().store_config(self, "info_log_print_time", float(val))

    @staticmethod
    def _check_integer_larger_then_one(
        val: Any, method_name: str, warn_user: bool
    ) -> int:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val)
            if not np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_:
                raise HoraceConfigError(
                    "HERBERT:Log_config:invalid_argument",
                    f"Only numeric value allowed for {method_name}. Provided: {type(val).__name__}",
                )
            if arr.size > 1:
                raise HoraceConfigError(
                    "HERBERT:Log_config:invalid_argument",
                    f"Input for property {method_name} can be scalar only. Attempt to set value with {arr.size} elements",
                )
            val = float(arr.item())
        elif not isinstance(val, (int, float, np.number)) or isinstance(val, bool):
            raise HoraceConfigError(
                "HERBERT:Log_config:invalid_argument",
                f"Only numeric value allowed for {method_name}. Provided: {type(val).__name__}",
            )
        if val < 1 and warn_user:
            warn_with_id(
                "LOG_CONFIG:setting_log_split_ratio",
                f"*** Attempt to set {method_name} to be smaller then 1\n"
                "    Changed to be 1. Think about decreasing calculations chunk size\n"
                "    as time of calculating single chunk exceeds info_log_print_time",
            )
        return max(1, int(round(float(val))))

    def _get_split_ratio(self, field_name: str) -> int:
        return int(config_store.instance().get_value("log_config", field_name))

    def _set_split_ratio(self, field_name: str, val: Union[int, float]) -> None:
        checked = self._check_integer_larger_then_one(val, field_name, True)
        config_store.instance().store_config(self, field_name, checked)

    def init_adaptive_logging(self, prefix_name: str = "") -> "log_config":
        """Initialize adaptive logging procedure."""
        self.dot_printed = False
        self.log_time_holder_ = time.perf_counter()
        if prefix_name:
            prop_name = f"{prefix_name}_split_ratio"
            if prop_name in self._SAVED_PROPERTIES:
                self.info_log_split_ratio_ = getattr(self, prop_name)
        return self

    def adapt_logging(self, n_step: int) -> Tuple["log_config", float]:
        """Adapt `info_log_split_ratio` based on elapsed time since `init_adaptive_logging`.

        Inputs:
        - `n_step` -- Number of steps performed from the start of the logging loop.

        Output:
        - `obj` -- Instance of `log_config` class containing modified `info_log_split_ratio`.
        - `pass_time` -- Time passed from the time when `init_adaptive_logging` was called
          to the call of this function.
        """
        now = time.perf_counter()
        if self.log_time_holder_ is None:
            self.log_time_holder_ = now
        pass_time = max(now - self.log_time_holder_, 1e-9)
        time_per_step = pass_time / max(1, n_step)
        split_ratio = self.info_log_print_time / time_per_step
        self.info_log_split_ratio_ = self._check_integer_larger_then_one(
            split_ratio, "info_log_split_ratio", False
        )
        return self, pass_time

    def get_storage_field_names(self) -> List[str]:
        """Helper function returns the list of the public names of the fields
        which should be saved.
        """
        return list(self._SAVED_PROPERTIES)

    def get_default_value(self, field_name: str) -> Any:
        """Method gets internal field value bypassing standard get/set methods interface.
        Relies on assumption that each public field has a private field with name
        different by underscore.
        """
        return getattr(self, f"{field_name}_")


# Dynamically attach all 16 *_split_ratio properties to log_config
def _make_split_ratio_prop(prop_name: str) -> property:
    return property(
        lambda self: self._get_split_ratio(prop_name),
        lambda self, val: self._set_split_ratio(prop_name, val),
        doc=f"PageOp logging split ratio for `{prop_name[:-12]}` operations.",
    )


for _prop in log_config._SAVED_PROPERTIES[1:]:
    setattr(log_config, _prop, _make_split_ratio_prop(_prop))


__all__ = ["log_config"]
