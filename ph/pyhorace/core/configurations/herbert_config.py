"""Deprecated Herbert configuration class (alias for `hor_config`).

Translated from `herbert_core/configuration/herbert_config.m`.
"""

from __future__ import annotations

from .config_store import CONFIG_CLASS_REGISTRY, warn_with_id
from .hor_config import hor_config


class herbert_config(hor_config):
    """Deprecated function/class alias for `hor_config`.

    `herbert_config` has been rendered obsolete and all features have moved
    to `hor_config`. Calling `herbert_config()` issues a
    `HORACE:herbert_config:deprecated` warning and returns a `hor_config`
    instance.
    """

    def __init__(self) -> None:
        super().__init__()
        CONFIG_CLASS_REGISTRY["hor_config"] = hor_config
        warn_with_id(
            "HORACE:herbert_config:deprecated",
            "herbert_config has been rendered obsolete and has been removed. \nAll features have moved to hor_config",
        )

    @property
    def is_default(self) -> bool:
        """Return True if no custom configuration has been set."""
        warn_with_id(
            "HORACE:herbert_config:deprecated",
            "Class herbert_config is deprecated. Use hor_config directly.",
        )
        return super().is_default


__all__ = ["herbert_config"]
