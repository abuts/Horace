"""4-dimensional Horace DnD dataset class (d4d).

Ported from Horace MATLAB @d4d class (horace_core/sqw/@d4d/d4d.m).
"""

from __future__ import annotations

from typing import ClassVar

from pyhorace.core.serializable import register_serializable
from pyhorace.sqw.dnd_base import DnDBase


@register_serializable
class d4d(DnDBase):
    """4-dimensional Horace DnD dataset."""

    NUM_DIMS: ClassVar[int] = 4

