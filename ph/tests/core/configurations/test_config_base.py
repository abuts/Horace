"""Unit tests for `config_base`.

Translated from `_test/test_config/test_config_base.m`.
"""

from __future__ import annotations

import os
import shutil

from pyhorace.core.configurations import config_store, set

from .conftest import config_base_tester


class TestConfigBase:
    """Translated from `_test/test_config/test_config_base.m`."""

    def setup_method(self) -> None:
        self.stored_config = config_store.instance().get_all_configs()

    def teardown_method(self) -> None:
        config_store.instance().set_all_configs(self.stored_config)
        config_store.instance().clear_config("config_base_tester", "-file")

    def test_store_restore_in_memory(self) -> None:
        config = config_base_tester()
        config_store.instance().clear_config(config)
        config_file = os.path.join(
            config_store.instance().config_folder, "config_base_tester.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        assert config.my_prop == "beee"

        config.my_prop = "meee"
        assert config.my_prop == "meee"

        config1 = config_base_tester()
        assert config1.my_prop == "meee"

        # clear configuration from memory
        config_store.instance().clear_config(config)

        config2 = config_base_tester()
        assert config2.my_prop == "meee"

        config2.returns_defaults = True
        assert config2.my_prop == "beee"

        config_store.instance().clear_config(config, "-file")

    def test_set_extern_config_folder_with_config_name(self, tmp_path) -> None:
        wkf = os.path.join(str(tmp_path), "mprogs_config_blabla")
        config_store.set_config_folder(wkf)
        cfn = config_store.instance().config_folder_name
        assert cfn == "mprogs_config_blabla"
        assert config_store.instance().config_folder == wkf

    def test_set_extern_folder_with_config_name(self, tmp_path) -> None:
        wkf = os.path.join(str(tmp_path), "mprogs_config_blabla")
        cfn_orig = config_store.instance().config_folder
        try:
            config_store.instance().set_config_path(wkf)
            cfn = config_store.instance().config_folder_name
            assert cfn == "mprogs_config_blabla"
            assert config_store.instance().config_folder == wkf
        finally:
            config_store.instance().set_config_path(cfn_orig)
            if os.path.isdir(wkf):
                shutil.rmtree(wkf)

    def test_set_extern_folder_no_config_name(self, tmp_path) -> None:
        cfn_orig = config_store.instance().config_folder
        wkf = str(tmp_path / "ext_root")
        os.makedirs(wkf, exist_ok=True)
        try:
            config_store.instance().set_config_path(wkf)
            cfn = config_store.instance().config_folder_name
            assert config_store.instance().config_folder == os.path.join(wkf, cfn)
        finally:
            config_store.instance().set_config_path(cfn_orig)
            target = os.path.join(wkf, "mprogs_config")
            if os.path.isdir(target):
                shutil.rmtree(target)

    def test_store_restore_in_file(self) -> None:
        config = config_base_tester()
        config_store.instance().clear_config(config)
        config_file = os.path.join(
            config_store.instance().config_folder, "config_base_tester.json"
        )
        if os.path.isfile(config_file):
            os.unlink(config_file)

        assert config.my_prop == "beee"

        config.my_prop = "meee"
        assert config.my_prop == "meee"

        config1 = config_base_tester()
        assert config1.my_prop == "meee"

        assert os.path.isfile(config_file)

        set(config_base_tester(), "my_prop", "veee", "-buffer")
        config1 = config_base_tester()
        assert config1.my_prop == "veee"

        # clear configuration from memory
        config_store.instance().clear_config(config)

        config2 = config_base_tester()
        assert config2.my_prop == "meee"

        config2.returns_defaults = True
        assert config2.my_prop == "beee"

        config_store.instance().clear_config(config, "-file")
        assert not os.path.isfile(config_file)

