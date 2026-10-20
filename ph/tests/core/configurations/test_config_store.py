"""Unit tests for `config_store` (JSON-backed singleton storage).

Translated from `_test/test_config/test_config_store.m`.
"""

from __future__ import annotations

import copy
import json
import os
import warnings

from pyhorace.core.configurations import (
    HoraceConfigWarning,
    config_store,
    lastwarn,
    warn_with_id,
)

from .conftest import some_test_class, some_test_class2, tgp_test_class2


class TestConfigStore:
    """Translated from `_test/test_config/test_config_store.m` (using `.json` files)."""

    def setup_method(self) -> None:
        config_store.instance().clear_config("tgp_test_class2", "-files")

    def teardown_method(self) -> None:
        config_store.instance().clear_config("tgp_test_class2", "-files")
        config_store.instance().clear_config("config_base_tester", "-file")

    def test_store_restore_new_class(self) -> None:
        tsc = some_test_class()
        config_store.instance().clear_config(tsc)
        test_class_name = tsc.class_name

        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc)
        assert os.path.isfile(config_file)

        # Verify file is valid JSON containing config_data
        with open(config_file, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        assert "config_data" in raw_json
        assert raw_json["config_data"] == {"a": 10, "b": "beee"}

        tsc1_data = config_store.instance().get_config(tsc)
        tsc1 = copy.deepcopy(tsc).set_stored_data(tsc1_data)
        assert tsc1 == tsc

        config_store.instance().clear_config(tsc1)
        assert not config_store.instance().is_configured(tsc1, "-in_mem")
        assert config_store.instance().is_configured(tsc1)

        tsc1_data = config_store.instance().get_config(tsc)
        tsc1 = copy.deepcopy(tsc).set_stored_data(tsc1_data)
        assert tsc1 == tsc

        assert config_store.instance().is_configured(tsc, "-in_mem")

        # clean up
        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

    def test_two_classes(self) -> None:
        tsc1 = some_test_class()
        tsc2 = some_test_class2()

        config_store.instance().clear_config(tsc1)
        config_store.instance().clear_config(tsc2)

        test_class_name1 = tsc1.class_name
        test_class_name2 = tsc2.class_name

        config_file1 = os.path.join(
            config_store.instance().config_folder, f"{test_class_name1}.json"
        )
        config_file2 = os.path.join(
            config_store.instance().config_folder, f"{test_class_name2}.json"
        )
        if os.path.isfile(config_file1):
            os.unlink(config_file1)
        if os.path.isfile(config_file2):
            os.unlink(config_file2)

        config_store.instance().store_config(tsc1)
        config_store.instance().store_config(tsc2)
        assert os.path.isfile(config_file1)
        assert os.path.isfile(config_file2)

        tsc2a_data = config_store.instance().get_config(tsc2)
        tsc2a = copy.deepcopy(tsc2).set_stored_data(tsc2a_data)
        assert tsc2a == tsc2

        tsc1a_data = config_store.instance().get_config(tsc1)
        tsc1a = copy.deepcopy(tsc1).set_stored_data(tsc1a_data)
        assert tsc1a == tsc1

        config_store.instance().clear_config(tsc1, "-file")
        assert not os.path.isfile(config_file1)

        config_store.instance().clear_config(tsc2, "-file")
        assert not os.path.isfile(config_file2)

        assert not config_store.instance().is_configured(tsc1, "-in_mem")
        assert not config_store.instance().is_configured(tsc2, "-in_mem")

        assert not config_store.instance().is_configured(tsc1)
        assert not config_store.instance().is_configured(tsc2)

    def test_force_save(self) -> None:
        tsc = some_test_class()
        config_store.instance().clear_config(tsc)

        test_class_name = tsc.class_name
        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc)
        assert os.path.isfile(config_file)

        if os.path.isfile(config_file):
            os.unlink(config_file)

        # Unchanged in memory -> does not re-save without -force
        config_store.instance().store_config(tsc)
        assert not os.path.isfile(config_file)

        config_store.instance().store_config(tsc, "-force")
        assert os.path.isfile(config_file)

        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

    def test_changed_config(self) -> None:
        tsc = some_test_class()
        config_store.instance().clear_config(tsc)

        test_class_name = tsc.class_name
        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc)
        assert os.path.isfile(config_file)
        if os.path.isfile(config_file):
            os.unlink(config_file)

        tsc1 = copy.deepcopy(tsc)
        tsc1.a = 20
        config_store.instance().store_config(tsc1)
        assert os.path.isfile(config_file)

        tsc1a_data = config_store.instance().get_config(tsc)
        tsc1a = copy.deepcopy(tsc).set_stored_data(tsc1a_data)
        assert tsc1 == tsc1a

        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

    def test_changed_class(self) -> None:
        tsc = some_test_class()
        config_store.instance().clear_config(tsc)
        test_class_name = tsc.class_name

        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc)
        assert os.path.isfile(config_file)

        tsc1 = some_test_class2()
        # Fool the loader forcing it to think that the class has changed
        tsc1 = tsc1.set_class_name(test_class_name)
        tsc1.a = 100

        config_store.instance().clear_config(tsc1)
        assert not config_store.instance().is_configured(tsc1, "-in_mem")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", HoraceConfigWarning)
            tsc1a_data = config_store.instance().get_config(tsc1)

        assert not os.path.isfile(config_file)
        tsc1a = copy.deepcopy(tsc1).set_stored_data(tsc1a_data)
        assert tsc1 == tsc1a

        tsc1b_data = config_store.instance().get_config(tsc1)
        tsc1b = copy.deepcopy(tsc1).set_stored_data(tsc1b_data)
        assert tsc1 == tsc1b

        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

    def test_set_restore_fields(self) -> None:
        tsc = some_test_class2()
        config_store.instance().clear_config(tsc)

        test_class_name = tsc.class_name
        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc, "a", "new_val")

        tsc1 = config_store.instance().get_config(tsc)
        assert tsc1.a == "new_val"
        assert tsc1.b == "beee"
        assert tsc1.c == "other_property"

        config_store.instance().store_config(tsc, "b", "meee")

        tsc1 = config_store.instance().get_config(tsc)
        assert tsc1.a == "new_val"
        assert tsc1.b == "meee"
        assert tsc1.c == "other_property"

        config_store.instance().clear_config(tsc)
        tsc1 = config_store.instance().get_config(tsc)
        assert tsc1.a == "new_val"
        assert tsc1.b == "meee"
        assert tsc1.c == "other_property"

        config_store.instance().clear_config(tsc)
        config_store.instance().store_config(tsc, "c", 100)

        tsc1 = config_store.instance().get_config(tsc)
        assert tsc1.a == "new_val"
        assert tsc1.b == "meee"
        assert tsc1.c == 100

        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

    def test_is_configured_on_first_call(self) -> None:
        warn_with_id(
            "HERBERT:fake_warning", "ensure no unwanted warnings have been issued"
        )

        assert not config_store.instance().is_field_configured(
            "tgp_test_class2", "v1"
        )
        stc = tgp_test_class2()
        assert not config_store.instance().is_field_configured(stc, "v1")
        assert not config_store.instance().is_field_configured(stc, "v2")
        assert not config_store.instance().is_field_configured(stc, "v3")
        assert not config_store.instance().is_field_configured(stc, "v4")

        _, lw = lastwarn()
        assert lw == "HERBERT:fake_warning"

        # first call to property loads it to memory
        assert stc.v1 == 10000000
        _, lw = lastwarn()
        assert lw == "HERBERT:config_store:default_configuration"

        warn_with_id(
            "HERBERT:fake_warning", "ensure no unwanted warnings have been issued"
        )
        assert config_store.instance().is_field_configured(stc, "v1")
        assert config_store.instance().is_field_configured(stc, "v2")
        assert config_store.instance().is_field_configured(stc, "v3")
        assert config_store.instance().is_field_configured(stc, "v4")

        _, lw = lastwarn()
        assert lw == "HERBERT:fake_warning"

        config_store.instance().clear_config(stc, "-files")

    def test_restore_some_fields(self) -> None:
        tsc = some_test_class2()
        config_store.instance().clear_config(tsc)

        test_class_name = tsc.class_name
        config_file = os.path.join(
            config_store.instance().config_folder, f"{test_class_name}.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        config_store.instance().store_config(tsc, "a", "new_val")

        a_val = config_store.instance().get_config_field(tsc, "a")
        assert a_val == "new_val"

        config_store.instance().store_config(tsc, "b", "meee")

        a_val, b_val, c_val = config_store.instance().get_config_field(
            tsc, "a", "b", "c"
        )
        assert a_val == "new_val"
        assert b_val == "meee"
        assert c_val == "other_property"

        config_store.instance().clear_config(tsc)
        b_val, c_val, a_val = config_store.instance().get_config_field(
            tsc, "b", "c", "a"
        )
        assert b_val == "meee"
        assert c_val == "other_property"
        assert a_val == "new_val"

        config_store.instance().clear_config(tsc)
        config_store.instance().store_config(tsc, "c", 100)

        c_val = config_store.instance().get_config_field(tsc, "c")
        assert c_val == 100

        config_store.instance().clear_config(tsc, "-file")
        assert not os.path.isfile(config_file)

