"""0-dimensional Horace DnD dataset class (d0d).

Ported from Horace MATLAB @d0d class (horace_core/sqw/@d0d/d0d.m).
"""

from __future__ import annotations

from typing import Any, ClassVar

from pyhorace.core.serializable import register_serializable
from pyhorace.sqw.dnd_base import DnDBase


@register_serializable
class d0d(DnDBase):
    """0-dimensional Horace DnD dataset."""

    NUM_DIMS: ClassVar[int] = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._in_d0d_init = True
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._in_d0d_init = False
