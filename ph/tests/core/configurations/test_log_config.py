"""Unit tests for `log_config`, `hor_config`, `planner_config`, and `herbert_config`.

Translated from `_test/test_utilities_herbert/test_log_config.m`.
"""

from __future__ import annotations

import os
import time

import pytest

from pyhorace.core.configurations import (
    HoraceConfigError,
    config_store,
    herbert_config,
    hor_config,
    lastwarn,
    log_config,
    planner_config,
    warn_with_id,
)


class TestLogAndOtherConfigs:
    """Translated from `_test/test_utilities_herbert/test_log_config.m` and tests
    for `hor_config`, `planner_config`, and `herbert_config`.
    """

    def test_all_split_log_ratios_warn_small(self) -> None:
        tc = log_config()
        tc.saveable = False
        all_prop = tc.get_storage_field_names()
        all_prop = all_prop[1:]  # remove info_log_print_time
        for prop in all_prop:
            warn_with_id("HORACE:clear_test_warn", "clear possible warnings")
            setattr(tc, prop, 0.1)
            assert (
                getattr(tc, prop) == 1
            ), f"Property {prop} has not been set correctly"
            mess, lw = lastwarn()
            assert lw == "LOG_CONFIG:setting_log_split_ratio"
            assert prop in mess
        config_store.instance().clear_config(tc)

    def test_all_split_log_ratios_settable(self) -> None:
        tc = log_config()
        tc.saveable = False
        all_prop = tc.get_storage_field_names()
        all_prop = all_prop[1:]
        for prop in all_prop:
            setattr(tc, prop, 100)
            assert getattr(tc, prop) == 100
        tc.info_log_print_time = 60
        assert tc.info_log_print_time == 60
        config_store.instance().clear_config(tc)

    def test_auto_logging(self) -> None:
        tc = log_config()
        tc.saveable = False
        tc.info_log_print_time = 2.0
        tc = tc.init_adaptive_logging()
        time.sleep(0.2)
        tc, elapsed = tc.adapt_logging(2)
        assert elapsed >= 0.15
        assert tc.info_log_split_ratio >= 1
        config_store.instance().clear_config(tc)

    def test_info_log_print_time(self) -> None:
        tc = log_config()
        tc.info_log_print_time = 30
        assert tc.info_log_print_time == 30

    def test_info_log_split_ratio(self) -> None:
        tc = log_config()
        assert tc.info_log_split_ratio == 1

    def test_herbert_config_deprecated(self) -> None:
        hc = herbert_config()
        _, lw = lastwarn()
        assert lw == "HORACE:herbert_config:deprecated"
        assert isinstance(hc, hor_config)

    def test_planner_config_json_persistence(self) -> None:
        pc = planner_config()
        pc.ei = 250.0
        pc.u = [1, 1, 0]
        pc.par_file = "maps_4to1.par"

        cfg_file = os.path.join(
            config_store.instance().config_folder, "planner_config.json"
        )
        assert os.path.isfile(cfg_file)

        config_store.instance().clear_config("planner_config")
        pc2 = planner_config()
        assert pc2.ei == 250.0
        assert list(pc2.u) == [1, 1, 0]
        assert pc2.par_file == "maps_4to1.par"

        with pytest.raises(HoraceConfigError):
            pc2.ei = -5
        with pytest.raises(HoraceConfigError):
            pc2.u = [1, 2]

