"""Pixel metadata container (translated from ``@pix_metadata``)."""

from __future__ import annotations

import os
from typing import Any, ClassVar, List, Sequence, Union

import numpy as np

from pyhorace.core.configurations.config_store import warn_with_id
from pyhorace.core.serializable import Serializable, register_serializable
from pyhorace.coord_transf._dependencies import HoraceError


class _RangeValidBool(int):
    """Boolean-like value usable both as a bool property and as a callable method."""

    def __new__(cls, val: bool, owner: "pix_metadata") -> "_RangeValidBool":
        obj = super().__new__(cls, int(bool(val)))
        obj._owner = owner
        return obj

    def __call__(self, fld_names: Union[str, Sequence[str]] = "all") -> bool:
        if fld_names == "pix_range":
            fld_names = "coordinates"
        return self._owner.check_range_valid(fld_names)

    def __bool__(self) -> bool:
        return bool(int(self))

    def __repr__(self) -> str:
        return repr(bool(int(self)))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (bool, np.bool_, int, np.integer)):
            return bool(int(self)) == bool(other)
        return False


class pix_metadata(Serializable):
    """Metadata describing a pixel dataset (filename, pixel count, ranges, alignment)."""

    DEFAULT_NUM_PIX_FIELDS: ClassVar[int] = 9
    EMPTY_RANGE: ClassVar[np.ndarray] = np.vstack(
        [np.full((1, 9), np.inf, dtype=np.float64), np.full((1, 9), -np.inf, dtype=np.float64)]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Serializable.__init__(self)
        self._full_filename: str = ""
        self._npix: int = 0
        self._npix_set: bool = False
        self._data_range: np.ndarray = self.EMPTY_RANGE.copy()
        self._alignment_matr: np.ndarray = np.eye(3, dtype=np.float64)

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, pix_metadata):
                self._full_filename = arg.full_filename
                self._npix = arg.npix
                self._npix_set = arg._npix_set
                self._data_range = arg.data_range.copy()
                self._alignment_matr = arg.alignment_matr.copy()
                return
            # Check if arg is a PixelDataBase-like object
            if hasattr(arg, "num_pixels") and hasattr(arg, "data_range") and hasattr(arg, "full_filename"):
                self.full_filename = arg.full_filename
                self.npix = arg.num_pixels
                self.data_range = arg.data_range.copy()
                if hasattr(arg, "alignment_matr") and getattr(arg, "is_misaligned", getattr(arg, "is_corrected", False)):
                    self.alignment_matr = np.asarray(arg.alignment_matr, dtype=np.float64).copy()
                return
            if isinstance(arg, dict):
                self._init_from_dict(arg)
                return

        self.set_positional_and_key_val_arguments(
            self.saveable_fields(), False, *args, **kwargs
        )

    def _init_from_dict(self, d: dict) -> None:
        for k in self.saveable_fields():
            if k in d:
                setattr(self, k, d[k])

    @property
    def filename(self) -> str:
        if not self._full_filename:
            return ""
        return os.path.basename(self._full_filename)

    @property
    def filepath(self) -> str:
        if not self._full_filename:
            return ""
        return os.path.dirname(self._full_filename)

    @property
    def full_filename(self) -> str:
        return self._full_filename

    @full_filename.setter
    def full_filename(self, val: Any) -> None:
        if val is None or val == "":
            self._full_filename = ""
            return
        if not isinstance(val, (str, os.PathLike)):
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"full_filename should be a string, got {type(val).__name__}"
            )
        self._full_filename = str(val)

    @property
    def npix(self) -> int:
        return self._npix

    @npix.setter
    def npix(self, val: Any) -> None:
        if val is None or (isinstance(val, np.ndarray) and val.size == 0):
            self._npix = 0
            self._npix_set = False
            return
        try:
            ival = int(np.asarray(val).item())
        except Exception as exc:
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"npix must be a non-negative integer scalar, got {val!r}"
            ) from exc
        if ival < 0 or float(np.asarray(val).item()) != ival:
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"npix must be a non-negative integer scalar, got {val!r}"
            )
        self._npix = ival
        self._npix_set = True

    @property
    def num_pixels(self) -> int:
        return self._npix

    @num_pixels.setter
    def num_pixels(self, val: Any) -> None:
        self.npix = val

    @property
    def pix_range(self) -> np.ndarray:
        return self._data_range[:, :4].copy()

    @pix_range.setter
    def pix_range(self, val: Any) -> None:
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != (2, 4):
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"pix_range must be a 2x4 array, got shape {arr.shape}"
            )
        self._data_range[:, :4] = arr

    @property
    def data_range(self) -> np.ndarray:
        return self._data_range.copy()

    @data_range.setter
    def data_range(self, val: Any) -> None:
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != (2, 9):
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"data_range must be a 2x9 array, got shape {arr.shape}"
            )
        self._data_range = arr.copy()

    @property
    def is_misaligned(self) -> bool:
        dif = np.abs(self._alignment_matr - np.eye(3, dtype=np.float64))
        return bool(np.max(dif) > 1e-6)

    @property
    def is_corrected(self) -> bool:
        return self.is_misaligned

    @property
    def alignment_matr(self) -> np.ndarray:
        return self._alignment_matr.copy()

    @alignment_matr.setter
    def alignment_matr(self, val: Any) -> None:
        if val is None or (isinstance(val, np.ndarray) and val.size == 0):
            self._alignment_matr = np.eye(3, dtype=np.float64)
            return
        arr = np.asarray(val, dtype=np.float64)
        if arr.shape != (3, 3):
            raise HoraceError(
                f"HORACE:pix_metadata:invalid_argument - "
                f"alignment_matr must be a 3x3 matrix, got shape {arr.shape}"
            )
        self._alignment_matr = arr.copy()

    @property
    def is_range_valid(self) -> _RangeValidBool:
        return _RangeValidBool(self.check_range_valid("all"), self)

    @property
    def is_pix_range_valid(self) -> bool:
        return self.check_range_valid("coordinates")

    def reset_range(self, field_names: Union[str, Sequence[str]] = "all") -> None:
        """Reset specified fields of ``data_range`` to ``EMPTY_RANGE``."""
        from pyhorace.data.pixels.pixel_data_base import PixelDataBase

        ind = PixelDataBase.field_index(field_names)
        idx0 = np.asarray(ind, dtype=np.intp) - 1
        self._data_range[:, idx0] = self.EMPTY_RANGE[:, idx0]

    def check_range_valid(self, fld_names: Union[str, Sequence[str]] = "all") -> bool:
        """Return True if all specified fields in ``data_range`` are finite and min <= max."""
        from pyhorace.data.pixels.pixel_data_base import PixelDataBase

        if fld_names == "pix_range":
            fld_names = "coordinates"
        ind = PixelDataBase.field_index(fld_names)
        idx0 = np.asarray(ind, dtype=np.intp) - 1
        range_slice = self._data_range[:, idx0]
        invalid = np.any(np.isinf(range_slice)) or np.any(range_slice[0, :] > range_slice[1, :])
        return not bool(invalid)

    def recalc_data_range(self, fld_names: Union[str, Sequence[str]] = "all") -> None:
        """Warn when attempting to recalculate range on metadata alone."""
        if self.check_range_valid(fld_names):
            return
        warn_with_id(
            "HORACE:pix_metadata:UnableToRecalcRange",
            "Passing misaligned fluid data array via file is not supported. Data range was not recalculated.",
        )

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        if self.is_misaligned or not self._npix_set:
            return ["full_filename", "npix", "data_range", "alignment_matr"]
        return ["full_filename", "npix", "data_range"]


register_serializable(pix_metadata)
