"""Unit tests for `hpc_config`.

Translated from `_test/test_config/test_hpc_config.m`.
"""

from __future__ import annotations

from pyhorace.core.configurations import config_store, hpc_config


class TestHpcConfig:
    """Translated from `_test/test_config/test_hpc_config.m`."""

    def test_get_default_memory_from_config_store(self) -> None:
        config_store.instance().clear_config("hpc_config")
        mem = config_store.instance().get_value("hpc_config", "phys_mem_available")
        assert mem is not None
        assert mem > 0

    def test_get_free_memory_from_empty(self) -> None:
        hpc = hpc_config()
        hpc.phys_mem_available = []
        mem = hpc.phys_mem_available
        assert hpc.is_field_configured("phys_mem_available")
        assert mem > 0

    def test_get_free_memory(self) -> None:
        hpc = hpc_config()
        mem = hpc.phys_mem_available
        assert hpc.is_field_configured("phys_mem_available")
        assert mem > 0

    def test_mem_works(self) -> None:
        fm = hpc_config.calc_free_memory()
        assert fm > 0

