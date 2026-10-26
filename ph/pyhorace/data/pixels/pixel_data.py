"""Legacy ``PixelData`` factory/wrapper class (translated from ``PixelData.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pyhorace.core.configurations.config_store import warn_with_id
from pyhorace.core.serializable import SERIALIZABLE_REGISTRY
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


class _PixelDataMeta(type):
    """Metaclass allowing ``isinstance(x, PixelData)`` to match any ``PixelDataBase``."""

    def __instancecheck__(cls, instance: Any) -> bool:
        return isinstance(instance, PixelDataBase)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, PixelDataBase)


class PixelData(metaclass=_PixelDataMeta):
    """Compatibility wrapper around :class:`PixelDataBase`."""

    DATA_POINT_SIZE: int = PixelDataBase.DATA_POINT_SIZE
    DEFAULT_NUM_PIX_FIELDS: int = PixelDataBase.DEFAULT_NUM_PIX_FIELDS
    EMPTY_RANGE: np.ndarray = PixelDataBase.EMPTY_RANGE
    EMPTY_RANGE_: np.ndarray = PixelDataBase.EMPTY_RANGE_
    FIELD_INDEX_MAP = PixelDataBase.FIELD_INDEX_MAP

    def __new__(cls, *args: Any, **kwargs: Any) -> PixelDataBase:
        warn_with_id(
            "HORACE:PixelData:deprecated",
            "PixelData is deprecated; use PixelDataBase.create, PixelDataMemory, or PixelDataFileBacked.",
        )
        return PixelDataBase.create(*args, **kwargs)

    @staticmethod
    def create(*args: Any, **kwargs: Any) -> PixelDataBase:
        return PixelDataBase.create(*args, **kwargs)

    @staticmethod
    def cat(*args: Any, **kwargs: Any) -> PixelDataBase:
        return PixelDataBase.cat(*args, **kwargs)

    @staticmethod
    def field_index(fld_names: Any) -> np.ndarray:
        return PixelDataBase.field_index(fld_names)

    @staticmethod
    def do_filebacked(num_pixels: int, mem_chunk_size: Any = None) -> bool:
        return PixelDataBase.do_filebacked(num_pixels, mem_chunk_size)

    @classmethod
    def loadobj(cls, S: Any) -> PixelDataBase:
        return PixelDataBase.loadobj(S)


SERIALIZABLE_REGISTRY["PixelData"] = PixelData  # type: ignore[assignment]
