"""Unit tests for general configuration classes and `get`/`set` operations.

Translated from `_test/test_config/test_config_classes.m`.
"""

from __future__ import annotations

import os

import pytest

from pyhorace.core.configurations import (
    HoraceConfigError,
    config_store,
    get,
    hor_config,
    lastwarn,
    set,
    warn_with_id,
)

from .conftest import config_base_tester, tgp_test_class2


class TestConfigClasses:
    """Translated from `_test/test_config/test_config_classes.m`."""

    def setup_method(self) -> None:
        set(tgp_test_class2(), "default")
        self.s2_def = get(tgp_test_class2())
        config_store.instance().clear_config(tgp_test_class2(), "-files")

    def teardown_method(self) -> None:
        config_store.instance().clear_config(tgp_test_class2(), "-files")

    def test_getstruct(self) -> None:
        s2_def_pub = get(tgp_test_class2())
        assert list(s2_def_pub.keys()) == ["v1", "v2", "v3", "v4"]
        assert self.s2_def["v1"] == s2_def_pub["v1"]
        assert self.s2_def["v2"] == s2_def_pub["v2"]

        v1, v3 = get(tgp_test_class2(), "v1", "v3")
        assert self.s2_def["v1"] == v1
        assert self.s2_def["v3"] == v3

    def test_get_wrongCase(self) -> None:
        with pytest.raises((HoraceConfigError, KeyError, AttributeError, ValueError)):
            get(tgp_test_class2(), "v1", "V3")

    def test_get_and_save(self) -> None:
        set(tgp_test_class2(), "v1", 55, "-buffer")
        s2_buf = get(tgp_test_class2())

        set(tgp_test_class2(), "def", "-buffer")
        s2_tmp = get(tgp_test_class2())

        assert s2_tmp != s2_buf
        assert s2_tmp == self.s2_def

    def test_set_withbuffer(self) -> None:
        set(tgp_test_class2(), "v1", -30)
        s2_sav = get(tgp_test_class2())

        # Change the config without saving, change to saved values, see this done properly
        set(tgp_test_class2(), "v1", 55, "-buffer")
        s2_buf = get(tgp_test_class2())

        set(tgp_test_class2(), "save")
        s2_tmp = get(tgp_test_class2())

        assert s2_tmp != s2_buf
        assert s2_tmp == s2_sav

    def test_set_tests(self) -> None:
        hc = hor_config()
        set(hc, "init_tests", 0, "-buffer")
        assert not hc.init_tests
        assert hc.unit_test_folder in (None, [], "")

        set(hc, "init_tests", 1, "-buffer")
        assert hc.init_tests
        assert bool(hc.unit_test_folder)

    def test_is_configured_on_first_call(self) -> None:
        warn_with_id(
            "HERBERT:fake_warning", "ensure no unwanted warnings have been issued"
        )
        config_store.instance().clear_config("tgp_test_class2", "-files")

        stc = tgp_test_class2()
        assert not stc.is_field_configured("v1")
        assert not stc.is_field_configured("v2")
        assert not stc.is_field_configured("v3")
        assert not stc.is_field_configured("v4")

        _, lw = lastwarn()
        assert lw == "HERBERT:fake_warning"

        assert stc.v1 == 10000000
        _, lw = lastwarn()
        assert lw == "HERBERT:config_store:default_configuration"

        warn_with_id(
            "HERBERT:fake_warning", "ensure no unwanted warnings have been issued"
        )
        assert stc.is_field_configured("v1")
        assert stc.is_field_configured("v2")
        assert stc.is_field_configured("v3")
        assert stc.is_field_configured("v4")

        _, lw = lastwarn()
        assert lw == "HERBERT:fake_warning"

    def test_unsaveable_recovered_from_store(self) -> None:
        cf = config_base_tester()
        config_class_file = os.path.join(cf.config_folder, "config_base_tester.json")
        config_store.instance().clear_config(config_base_tester(), "-file")
        assert not os.path.isfile(config_class_file)

        warn_with_id(
            "HERBERT:test_warning", "this is to set warning to defined state"
        )

        val = config_store.instance().get_value(
            "config_base_tester", "unsaveable_property"
        )
        assert val == "abra_cadabra"
        _, lv = lastwarn()
        assert lv == "HERBERT:config_store:default_configuration"

    def test_unsaveable_sets_gets_clears(self) -> None:
        cf = config_base_tester()
        config_class_file = os.path.join(cf.config_folder, "config_base_tester.json")
        config_store.instance().clear_config(config_base_tester(), "-file")
        assert not os.path.isfile(config_class_file)

        warn_with_id(
            "HERBERT:test_warning",
            "this is to check if no other warning was issued",
        )
        cf.unsaveable_property = "ha_ha_ha"
        # nothing should be saved to disk when only unsaveable property is set
        assert not os.path.isfile(config_class_file)
        cf.my_prop = "AAA"

        assert cf.unsaveable_property == "ha_ha_ha"
        assert cf.my_prop == "AAA"

        config_store.instance().clear_config(config_base_tester())
        assert os.path.isfile(config_class_file)

        cf = config_base_tester()
        assert cf.unsaveable_property == "abra_cadabra"
        assert cf.my_prop == "AAA"

        _, lv = lastwarn()
        assert lv == "HERBERT:config_store:default_configuration"

    def test_unsaveable_property_works_with_no_warning(self) -> None:
        cf = config_base_tester()
        config_class_file = os.path.join(cf.config_folder, "config_base_tester.json")
        if not os.path.isfile(config_class_file):
            cf.my_prop = "something"
        assert os.path.isfile(config_class_file)

        warn_with_id(
            "HERBERT:test_warning",
            "this is to check if no other warning was issued",
        )

        assert cf.unsaveable_property == "abra_cadabra"
        _, lv = lastwarn()
        assert lv == "HERBERT:test_warning"

