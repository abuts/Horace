"""Unit tests for `parallel_config`.

Translated from `_test/test_config/test_parallel_config.m`.
"""

from __future__ import annotations

import pytest

from pyhorace.core.configurations import (
    HoraceConfigError,
    lastwarn,
    parallel_config,
    set,
)

from .conftest import parallel_config_tester


class TestParallelConfig:
    """Translated from `_test/test_config/test_parallel_config.m`."""

    def test__fake_worker(self) -> None:
        pc = parallel_config_tester()
        pc = pc.set_worker("non_existing_worker")

        pc.worker = "non_existing_worker"
        _, wid = lastwarn()
        assert wid == "HORACE:parallel_config:invalid_argument"
        assert pc.worker == "non_existing_worker"
        assert pc.parallel_cluster == "none"
        assert pc.cluster_config == "none"

    def test__missing_worker(self) -> None:
        pc = parallel_config()
        with pytest.raises(HoraceConfigError) as exc_info:
            set(pc, "worker", "non_existing_worker")
        assert "HORACE:parallel_config:invalid_argument" in str(exc_info.value)

    def test__slurm_commands_parser(self) -> None:
        pc = parallel_config()

        new_commands = {"-A": "account", "-p": "partition"}
        new_commands_app = {"-p": "new_part", "-q": "queue"}
        new_commands_app_check = {"-A": "account", "-p": "new_part", "-q": "queue"}

        # Destructive: Set as map/dict
        pc.slurm_commands = new_commands
        assert list(pc.slurm_commands.keys()) == list(new_commands.keys())
        assert list(pc.slurm_commands.values()) == list(new_commands.values())

        # Set empty (check clearing works)
        pc.slurm_commands = []
        assert len(pc.slurm_commands) == 0

        # Set as char/str
        pc.slurm_commands = []
        pc.slurm_commands = "-A account -p=partition"
        assert dict(pc.slurm_commands) == new_commands

        # Set as list of strings
        pc.slurm_commands = []
        pc.slurm_commands = ["-A", "account", "-p", "partition"]
        assert dict(pc.slurm_commands) == new_commands

        # Set as list of pairs
        pc.slurm_commands = []
        pc.slurm_commands = [["-A", "account"], ["-p", "partition"]]
        assert dict(pc.slurm_commands) == new_commands

        # Using update_slurm_commands
        pc.slurm_commands = []
        pc.update_slurm_commands("-A account -p=partition", False)
        assert dict(pc.slurm_commands) == new_commands

        # Using update_slurm_commands omitting append
        pc.slurm_commands = []
        pc.update_slurm_commands(new_commands)
        assert dict(pc.slurm_commands) == new_commands

        # Non-destructive: Set through update_slurm_commands as map
        pc.slurm_commands = new_commands
        pc.update_slurm_commands(new_commands_app, True)
        assert dict(pc.slurm_commands) == new_commands_app_check

        # Set through update_slurm_commands as char
        pc.slurm_commands = new_commands
        pc.update_slurm_commands("-q queue -p=new_part", True)
        assert dict(pc.slurm_commands) == new_commands_app_check

