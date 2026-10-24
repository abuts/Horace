"""DnD image data container class (dnd_data).

Ported from Horace MATLAB @dnd_data class
(horace_core/sqw/file_io/class_helpers/dnd_data.m).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.data_op_interface import _is_numeric, matlab_size
from pyhorace.core.serializable import Serializable, register_serializable


def _normalize_dnd_array(val: Any, prop_name: str, non_negative: bool = False) -> np.ndarray:
    """Validate and convert input to a float64 numpy array."""
    if val is None:
        return np.empty((0, 0), dtype=float)
    if not _is_numeric(val):
        raise HoraceError(
            "HORACE:dnd_data:invalid_argument",
            f"{prop_name} must be a numeric array, got {type(val).__name__}",
        )
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape((1, 1))
    if non_negative and arr.size > 0:
        with np.errstate(invalid="ignore"):
            if np.any(arr < 0):
                raise HoraceError(
                    "HORACE:dnd_data:invalid_argument",
                    f"{prop_name} cannot contain negative values",
                )
    return arr.copy()


@register_serializable
class dnd_data(Serializable):
    """Class holding multidimensional image data arrays (sig, err, npix) of a DnD object."""

    _sig: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=float))
    _err: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=float))
    _npix: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=float))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._sig = np.empty((0, 0), dtype=float)
        self._err = np.empty((0, 0), dtype=float)
        self._npix = np.empty((0, 0), dtype=float)

        if len(args) == 0 and not kwargs:
            return

        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, dnd_data):
                self._sig = arg0._sig.copy()
                self._err = arg0._err.copy()
                self._npix = arg0._npix.copy()
                return
            if hasattr(arg0, "s") and hasattr(arg0, "e") and hasattr(arg0, "npix") and not isinstance(arg0, dict):
                self._do_check_combo_arg = False
                self.sig = arg0.s
                self.err = arg0.e
                self.npix = arg0.npix
                self._do_check_combo_arg = True
                self.check_combo_arg()
                return
            if isinstance(arg0, dict):
                if "s" in arg0 and "sig" not in arg0:
                    self._do_check_combo_arg = False
                    self.sig = arg0["s"]
                    self.err = arg0["e"]
                    self.npix = arg0["npix"]
                    self._do_check_combo_arg = True
                    self.check_combo_arg()
                elif "serial_name" in arg0:
                    loaded = self.from_struct(arg0)
                    if isinstance(loaded, dnd_data):
                        self._sig = loaded._sig
                        self._err = loaded._err
                        self._npix = loaded._npix
                else:
                    self.from_bare_struct(arg0)
                return

        flds = self.saveable_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *args, **kwargs
        )
        if remains:
            raise HoraceError(
                "HORACE:dnd_data:invalid_argument",
                f"Additional unrecognized parameters provided to dnd_data constructor: {remains}",
            )

    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> List[str]:
        return ["sig", "err", "npix"]

    def check_combo_arg(self) -> "dnd_data":
        sz_s = matlab_size(self._sig)
        sz_e = matlab_size(self._err)
        sz_n = matlab_size(self._npix)
        if sz_s != sz_e or sz_s != sz_n:
            # Also check if all are 1D vs 2D column/row vector of same length
            if not (
                self._sig.size == self._err.size == self._npix.size
                and self.dimensions <= 1
            ):
                raise HoraceError(
                    "HORACE:dnd_data:invalid_argument",
                    f" Sizes of sig: {sz_s}, err: {sz_e} and npix: {sz_n} arrays must be equal",
                )
        # Align internal shapes
        if self._sig.shape != self._err.shape and self._sig.size == self._err.size:
            self._err = self._err.reshape(self._sig.shape, order="F")
        if self._sig.shape != self._npix.shape and self._sig.size == self._npix.size:
            self._npix = self._npix.reshape(self._sig.shape, order="F")
        return self

    @property
    def sig(self) -> np.ndarray:
        return self._sig

    @sig.setter
    def sig(self, val: Any) -> None:
        self._sig = _normalize_dnd_array(val, "sig", non_negative=False)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def s(self) -> np.ndarray:
        return self._sig

    @s.setter
    def s(self, val: Any) -> None:
        self.sig = val

    @property
    def err(self) -> np.ndarray:
        return self._err

    @err.setter
    def err(self, val: Any) -> None:
        self._err = _normalize_dnd_array(val, "err", non_negative=True)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def e(self) -> np.ndarray:
        return self._err

    @e.setter
    def e(self, val: Any) -> None:
        self.err = val

    @property
    def npix(self) -> np.ndarray:
        return self._npix

    @npix.setter
    def npix(self, val: Any) -> None:
        self._npix = _normalize_dnd_array(val, "npix", non_negative=True)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def dimensions(self) -> int:
        if self._sig.size <= 1:
            return 0
        sz = matlab_size(self._sig)
        if len(sz) == 2 and (sz[0] == 1 or sz[1] == 1):
            return 1
        return len(sz)

    @property
    def data_size(self) -> Tuple[int, ...]:
        if self._sig.size == 0:
            return (0, 0)
        if self._sig.size == 1:
            return (1, 1)
        if self.dimensions == 1:
            return (int(self._sig.size), 1)
        return matlab_size(self._sig)

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "dnd_data"]:
        if "s" in S and "sig" not in S:
            S_new = {
                "sig": S["s"],
                "err": S["e"],
                "npix": S["npix"],
            }
            return S_new, self
        return S, self

