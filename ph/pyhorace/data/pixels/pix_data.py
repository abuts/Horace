"""Helper class for saving/loading pixel data (translated from ``pix_data.m``)."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.coord_transf._dependencies import HoraceError


class pix_data(Serializable):
    """Serializable wrapper for raw pixel data arrays."""

    DEFAULT_NUM_PIX_FIELDS: int = 9

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Serializable.__init__(self)
        self._data: Any = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
        self._npix: int = 0

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            in_obj = args[0]
            if isinstance(in_obj, pix_data):
                self._data = in_obj.data.copy() if isinstance(in_obj.data, np.ndarray) else in_obj.data
                self._npix = in_obj.npix
                return
            if hasattr(in_obj, "is_filebacked") and hasattr(in_obj, "num_pixels"):
                if getattr(in_obj, "is_filebacked", False):
                    self._npix = int(in_obj.num_pixels)
                    self._data = getattr(in_obj, "full_filename", "") or np.empty(
                        (self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64
                    )
                else:
                    self.data = np.asarray(in_obj.data, dtype=np.float64)
                return
            if isinstance(in_obj, np.ndarray) or (
                isinstance(in_obj, (list, tuple)) and not isinstance(in_obj, dict)
            ):
                self.data = in_obj
                return
            if isinstance(in_obj, dict):
                for k in self.saveable_fields():
                    if k in in_obj:
                        setattr(self, k, in_obj[k])
                return

        self.set_positional_and_key_val_arguments(
            self.saveable_fields(), False, *args, **kwargs
        )

    @property
    def npix(self) -> int:
        if isinstance(self._data, str) or self._data.size == 0:
            return self._npix
        return int(self._data.shape[1])

    @npix.setter
    def npix(self, val: Any) -> None:
        if val is None or (isinstance(val, np.ndarray) and val.size == 0):
            self._npix = 0
            return
        try:
            ival = int(np.asarray(val).item())
        except Exception as exc:
            raise HoraceError(
                f"HORACE:pix_data:invalid_argument - "
                f"npix must be a non-negative integer scalar, got {val!r}"
            ) from exc
        if ival < 0 or float(np.asarray(val).item()) != ival:
            raise HoraceError(
                f"HORACE:pix_data:invalid_argument - "
                f"npix must be a non-negative integer scalar, got {val!r}"
            )
        if isinstance(self._data, np.ndarray) and self._data.size > 0 and self._data.shape[1] != ival:
            raise HoraceError(
                f"HORACE:pix_data:invalid_argument - "
                f"Cannot set npix={ival} when data has {self._data.shape[1]} columns."
            )
        self._npix = ival

    @property
    def num_pixels(self) -> int:
        return self.npix

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, val: Any) -> None:
        if val is None:
            self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
            return
        if isinstance(val, str):
            self._data = val
            return
        arr = np.asarray(val, dtype=np.float64)
        if arr.size == 0:
            self._data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
            return
        if arr.ndim != 2 or arr.shape[0] != self.DEFAULT_NUM_PIX_FIELDS:
            raise HoraceError(
                f"HORACE:pix_data:invalid_argument - "
                f"Data array must have shape (9, N), got {arr.shape}"
            )
        self._data = arr
        self._npix = int(arr.shape[1])

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return ["data", "npix"]


register_serializable(pix_data)

