"""Shared fixtures and helper test configuration classes for `tests/core/configurations/`.

Translated from MATLAB helper classes in `_test/test_config/`:
- `some_test_class.m`
- `some_test_class2.m`
- `tgp_test_class.m`
- `tgp_test_class2.m`
- `config_base_tester.m`
- `parallel_config_tester.m`
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from pyhorace.core.configurations import (
    config_base,
    config_store,
    parallel_config,
)


class some_test_class(config_base):
    """Test class to store/restore (translated from `some_test_class.m`)."""

    def __init__(self, class_name: str | None = None) -> None:
        if class_name is None:
            class_name = "some_test_class"
        super().__init__(class_name)
        self.a: Any = 10
        self.b: Any = "beee"

    def set_class_name(self, new_name: str) -> "some_test_class":
        self.class_name_ = new_name
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, some_test_class):
            return False
        flds = self.get_storage_field_names()
        flds2 = other.get_storage_field_names()
        if len(flds) != len(flds2):
            return False
        for field in flds:
            try:
                if getattr(self, field) != getattr(other, field):
                    return False
            except Exception:
                return False
        return True

    def get_data_to_store(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def get_storage_field_names(self) -> List[str]:
        return ["a", "b"]

    def get_default_value(self, field_name: str) -> Any:
        return getattr(self, field_name)


class some_test_class2(some_test_class):
    """Test class 2 to validate configuration (translated from `some_test_class2.m`)."""

    def __init__(self) -> None:
        super().__init__("some_test_class2")
        self.c: Any = "other_property"

    def get_data_to_store(self) -> Dict[str, Any]:
        data = super().get_data_to_store()
        data["c"] = self.c
        return data

    def get_storage_field_names(self) -> List[str]:
        return ["a", "b", "c"]

    def get_default_value(self, field_name: str) -> Any:
        return getattr(self, field_name)


class tgp_test_class(config_base):
    """Test configuration class (translated from `tgp_test_class.m`)."""

    def __init__(self, class_name: str | None = None) -> None:
        if class_name is None:
            class_name = "tgp_test_class"
        super().__init__(class_name)
        self.v1_ = 10000000
        self.v2_ = 9

    @property
    def v1(self) -> Any:
        return self.get_or_restore_field("v1")

    @v1.setter
    def v1(self, val: Any) -> None:
        config_store.instance().store_config(self, "v1", val)

    @property
    def v2(self) -> Any:
        return self.get_or_restore_field("v2")

    @v2.setter
    def v2(self, val: Any) -> None:
        config_store.instance().store_config(self, "v2", val)

    def set_class_name(self, new_name: str) -> "tgp_test_class":
        self.class_name_ = new_name
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, tgp_test_class):
            return False
        flds = self.get_storage_field_names()
        flds2 = other.get_storage_field_names()
        if len(flds) != len(flds2):
            return False
        for field in flds:
            try:
                if getattr(self, field) != getattr(other, field):
                    return False
            except Exception:
                return False
        return True

    def get_storage_field_names(self) -> List[str]:
        return ["v1", "v2"]

    def get_default_value(self, field_name: str) -> Any:
        return getattr(self, f"{field_name}_")


class tgp_test_class2(tgp_test_class):
    """Test configuration subclass (translated from `tgp_test_class2.m`)."""

    def __init__(self) -> None:
        super().__init__("tgp_test_class2")
        self.v3_ = "hello"
        self.v4_ = [13, 14]

    @property
    def v3(self) -> Any:
        return self.get_or_restore_field("v3")

    @v3.setter
    def v3(self, val: Any) -> None:
        config_store.instance().store_config(self, "v3", val)

    @property
    def v4(self) -> Any:
        return self.get_or_restore_field("v4")

    @v4.setter
    def v4(self, val: Any) -> None:
        config_store.instance().store_config(self, "v4", val)

    def get_storage_field_names(self) -> List[str]:
        fields = super().get_storage_field_names()
        return fields + ["v3", "v4"]

    def get_default_value(self, field_name: str) -> Any:
        return getattr(self, f"{field_name}_")


class config_base_tester(config_base):
    """Test class representing generic configuration (translated from `config_base_tester.m`)."""

    def __init__(self) -> None:
        super().__init__("config_base_tester")
        self.my_prop_ = "beee"
        self.my_prop2_ = 10
        self.unsaveable_property_ = "abra_cadabra"
        self.mem_only_prop_list_ = ["unsaveable_property"]

    @property
    def my_prop(self) -> Any:
        return self.get_or_restore_field("my_prop")

    @my_prop.setter
    def my_prop(self, val: Any) -> None:
        config_store.instance().store_config(self, "my_prop", val)

    @property
    def my_prop2(self) -> Any:
        return self.get_or_restore_field("my_prop2")

    @my_prop2.setter
    def my_prop2(self, val: Any) -> None:
        config_store.instance().store_config(self, "my_prop2", val)

    @property
    def unsaveable_property(self) -> Any:
        return self.get_or_restore_field("unsaveable_property")

    @unsaveable_property.setter
    def unsaveable_property(self, val: Any) -> None:
        config_store.instance().store_config(self, "unsaveable_property", val)

    def get_storage_field_names(self) -> List[str]:
        return ["my_prop", "my_prop2"]

    def get_default_value(self, field_name: str) -> Any:
        return getattr(self, f"{field_name}_")


class parallel_config_tester(parallel_config):
    """Class used to test parallel configuration (translated from `parallel_config_tester.m`)."""

    def __init__(self) -> None:
        super().__init__()
        config_store.instance().set_saveable(parallel_config(), False)

    def set_worker(self, worker_name: str) -> "parallel_config_tester":
        self.worker_ = worker_name
        config_store.instance().store_config("parallel_config", "worker", worker_name)
        return self


@pytest.fixture(autouse=True)
def isolated_config_store(tmp_path):
    """Ensure every test runs with an isolated temporary `config_store` folder."""
    orig_folder = config_store.instance().config_folder
    test_cfg_dir = str(tmp_path / "mprogs_config")
    config_store.set_config_folder(test_cfg_dir)
    yield test_cfg_dir
    config_store.instance("clear")
    if os.path.isdir(orig_folder):
        config_store.set_config_folder(orig_folder)

