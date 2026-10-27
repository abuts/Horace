"""Abstract interface for Horace binary SQW file accessors.

Ported from:
- horace_core/sqw/file_io/@sqw_file_interface/sqw_file_interface.m
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.exceptions import HoraceError
from pyhorace.io.horace_binfile_interface import horace_binfile_interface


class sqw_file_interface(horace_binfile_interface):
    """Abstract base class for Horace SQW binary file accessors (v2, v3, v4)."""

    format_for_object: str = "sqw"

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.npixels_: Union[int, str] = "undefined"
        self.num_contrib_files_: Union[int, str] = "undefined"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def npixels(self) -> Union[int, str]:
        return self.get_npixels()

    @npixels.setter
    def npixels(self, val: Union[int, str]) -> None:
        self.set_npixels(val)

    @property
    def num_contrib_files(self) -> Union[int, str]:
        return self.get_num_contrib_files()

    @property
    def pix_pos(self) -> Union[int, str]:
        return self.get_pix_pos()

    @pix_pos.setter
    def pix_pos(self, val: Union[int, str]) -> None:
        self.set_pix_pos(val)

    @property
    def data_type(self) -> str:
        return self.get_data_type()

    # ------------------------------------------------------------------
    # Protected Getters / Setters
    # ------------------------------------------------------------------

    def get_sqw_type(self) -> Union[bool, str]:
        if self.num_dim == "undefined":
            return "undefined"
        return True

    def get_npixels(self) -> Union[int, str]:
        return self.npixels_

    def set_npixels(self, val: Union[int, str]) -> None:
        if isinstance(val, str):
            if val == "undefined":
                self.npixels_ = "undefined"
                return
            raise HoraceError(
                "HORACE:sqw_file_interface:invalid_argument",
                f"npixels has to be non-negative integer or 'undefined', got {val!r}",
            )
        if isinstance(val, bool) or not isinstance(val, (int, float, np.number)) or int(val) < 0:
            raise HoraceError(
                "HORACE:sqw_file_interface:invalid_argument",
                f"npixels has to be non-negative integer, got {val!r}",
            )
        self.npixels_ = int(val)

    def get_num_contrib_files(self) -> Union[int, str]:
        return self.num_contrib_files_

    def get_pix_pos(self) -> Union[int, str]:
        return "undefined"

    def set_pix_pos(self, val: Union[int, str]) -> None:
        pass

    def get_data_type(self) -> str:
        if self.num_dim == "undefined":
            return "undefined"
        return "a"

    def delete(self) -> "sqw_file_interface":
        super().delete()
        self.npixels_ = "undefined"
        self.num_contrib_files_ = "undefined"
        return self

    def copy_contents(
        self, other_obj: horace_binfile_interface, keep_internals: bool = False
    ) -> "sqw_file_interface":
        super().copy_contents(other_obj, keep_internals=keep_internals)
        if isinstance(other_obj, sqw_file_interface):
            self.npixels_ = other_obj.npixels_
            self.num_contrib_files_ = other_obj.num_contrib_files_
        return self

    # ------------------------------------------------------------------
    # Abstract SQW File Access Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_main_header(self, *args: Any, **kwargs: Any) -> Any:
        """Read main_header from SQW file."""

    @abstractmethod
    def get_exp_info(self, *args: Any, **kwargs: Any) -> Any:
        """Read Experiment info from SQW file."""

    @abstractmethod
    def get_detpar(self, *args: Any, **kwargs: Any) -> Any:
        """Read legacy detector parameter structure from SQW file."""

    @abstractmethod
    def get_pix(self, *args: Any, **kwargs: Any) -> Any:
        """Read PixelData or raw pixel array from SQW file."""

    @abstractmethod
    def get_pix_at_indices(self, indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Read raw 9 x N pixel array at 1-based (or 0-based) pixel indices."""

    @abstractmethod
    def get_pix_in_ranges(
        self,
        pix_starts: Union[Sequence[int], np.ndarray],
        pix_ends: Union[Sequence[int], np.ndarray],
        skip_validation: bool = False,
        use_mex: bool = False,
    ) -> np.ndarray:
        """Read raw 9 x N pixel array across multiple [start, end] pixel ranges."""

    @abstractmethod
    def get_instrument(self, *args: Any, **kwargs: Any) -> Any:
        """Read instrument(s) from SQW file."""

    @abstractmethod
    def get_sample(self, *args: Any, **kwargs: Any) -> Any:
        """Read sample(s) from SQW file."""

    @abstractmethod
    def put_main_header(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write main_header to SQW file."""

    @abstractmethod
    def put_headers(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write experiment run headers to SQW file."""

    @abstractmethod
    def put_det_info(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write detector parameters to SQW file."""

    @abstractmethod
    def put_pix(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write pixel data to SQW file."""

    @abstractmethod
    def put_instruments(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write instrument(s) to SQW file."""

    @abstractmethod
    def put_samples(self, *args: Any, **kwargs: Any) -> "sqw_file_interface":
        """Write sample(s) to SQW file."""

    def extract_correct_subobj(self, obj_name: str, *args: Any) -> Tuple[Any, bool]:
        """Extract subobject `obj_name` from input arguments or `self.sqw_holder_`."""
        if len(args) > 0 and args[0] is not None and not isinstance(args[0], str):
            candidate = args[0]
            if isinstance(candidate, dict) and obj_name in candidate:
                return candidate[obj_name], True
            if hasattr(candidate, obj_name):
                return getattr(candidate, obj_name), True
            return candidate, True
        if self.sqw_holder_ is not None:
            if isinstance(self.sqw_holder_, dict) and obj_name in self.sqw_holder_:
                return self.sqw_holder_[obj_name], False
            if hasattr(self.sqw_holder_, obj_name):
                return getattr(self.sqw_holder_, obj_name), False
            if obj_name == "exp_info" and hasattr(self.sqw_holder_, "exper_info"):
                return getattr(self.sqw_holder_, "exper_info"), False
            if obj_name == "experiment_info" and hasattr(self.sqw_holder_, "exper_info"):
                return getattr(self.sqw_holder_, "exper_info"), False
        return None, False


SqwFileInterface = sqw_file_interface

__all__ = [
    "SqwFileInterface",
    "sqw_file_interface",
]

