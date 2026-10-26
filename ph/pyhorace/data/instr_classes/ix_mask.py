"""IX_mask class defining masked spectrum numbers."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.serializable import Serializable


def _str_to_iarray(text: str) -> np.ndarray:
    """Parse a string of integers and ranges (e.g. '1:5, 8, 10-12, 20:2:26') into a 1D int array.

    Port of Herbert ``str_to_iarray.m``.
    """
    # Strip comments starting with %, !, or #
    for comment_char in ("%", "!", "#"):
        idx = text.find(comment_char)
        if idx != -1:
            text = text[:idx]
    text = text.strip()
    if not text:
        return np.zeros(0, dtype=int)

    # Normalize range separators with surrounding spaces like "10 - 20" or "10 : 20"
    text = re.sub(r"\s*:\s*", ":", text)
    # In ISIS .msk files, "10-20" or "10 - 20" is a range 10:20 (for positive integers)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", ":", text)

    tokens = [tok for tok in re.split(r"[\s,;]+", text) if tok]
    result: List[np.ndarray] = []
    for tok in tokens:
        parts = tok.split(":")
        try:
            if len(parts) == 1:
                val = int(parts[0])
                result.append(np.array([val], dtype=int))
            elif len(parts) == 2:
                start, stop = int(parts[0]), int(parts[1])
                if stop >= start:
                    result.append(np.arange(start, stop + 1, dtype=int))
                else:
                    result.append(np.zeros(0, dtype=int))
            elif len(parts) == 3:
                start, step, stop = int(parts[0]), int(parts[1]), int(parts[2])
                if step == 0:
                    raise ValueError("Zero step in integer range.")
                if (step > 0 and stop >= start) or (step < 0 and stop <= start):
                    stop_adj = stop + (1 if step > 0 else -1)
                    result.append(np.arange(start, stop_adj, step, dtype=int))
                else:
                    result.append(np.zeros(0, dtype=int))
            else:
                raise ValueError(f"Invalid range token '{tok}'.")
        except ValueError as exc:
            raise ValueError(
                f"HERBERT:IX_mask:invalid_argument: Cannot parse integer array from '{tok}'."
            ) from exc

    if not result:
        return np.zeros(0, dtype=int)
    return np.concatenate(result)


def _iarray_to_str(arr: np.ndarray, max_len: int = 80) -> List[str]:
    """Format a sorted 1D integer array into compact range strings (e.g. '1:5 8 10:2:16').

    Port of Herbert ``iarray_to_str.m``.
    """
    a = np.asarray(arr, dtype=int).ravel()
    if a.size == 0:
        return [""]

    tokens: List[str] = []
    n = a.size
    i = 0
    while i < n:
        # Check for contiguous run (step == 1)
        j = i
        while j + 1 < n and a[j + 1] - a[j] == 1:
            j += 1
        if j - i >= 2:
            tokens.append(f"{a[i]}:{a[j]}")
            i = j + 1
            continue

        # Check for constant-step run (step > 1)
        if i + 2 < n:
            step = a[i + 1] - a[i]
            if step > 1 and a[i + 2] - a[i + 1] == step:
                k = i + 2
                while k + 1 < n and a[k + 1] - a[k] == step:
                    k += 1
                tokens.append(f"{a[i]}:{step}:{a[k]}")
                i = k + 1
                continue

        tokens.append(str(a[i]))
        i += 1

    lines: List[str] = []
    curr_line = ""
    for tok in tokens:
        if not curr_line:
            curr_line = tok
        elif len(curr_line) + 1 + len(tok) <= max_len:
            curr_line = f"{curr_line} {tok}"
        else:
            lines.append(curr_line)
            curr_line = tok
    if curr_line:
        lines.append(curr_line)
    return lines


class IX_mask(Serializable):
    """Mask object containing sorted unique positive spectrum numbers to mask.

    Corresponds to MATLAB ``IX_mask`` in Herbert/Horace.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._msk: np.ndarray = np.zeros(0, dtype=int)
        if len(args) == 0 and len(kwargs) == 0:
            return
        self.init(*args, **kwargs)

    def init(self, *args: Any, **kwargs: Any) -> "IX_mask":
        """Initialize or re-initialize the IX_mask instance."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, IX_mask):
                self._msk = arg0._msk.copy()
                return self
            if isinstance(arg0, dict):
                if "serial_name" in arg0 and "version" in arg0:
                    loaded = self.loadobj(arg0)
                    if isinstance(loaded, IX_mask):
                        self._msk = loaded._msk.copy()
                        return self
                return self.from_bare_struct(arg0)
            if isinstance(arg0, str):
                if os.path.isfile(arg0):
                    self._msk = self.get_mask_ascii(arg0)
                else:
                    # Try parsing as range description string first
                    try:
                        parsed = _str_to_iarray(arg0)
                    except ValueError as exc:
                        raise ValueError(
                            f"HERBERT:IX_mask:invalid_argument: "
                            f"Mask file '{arg0}' not found and string is not a valid integer range."
                        ) from exc
                    self.msk = parsed
                return self
            if isinstance(arg0, (list, tuple)) and len(arg0) > 0 and all(isinstance(x, str) for x in arg0):
                arrays = [_str_to_iarray(line) for line in arg0]
                combined = np.concatenate(arrays) if arrays else np.zeros(0, dtype=int)
                self.msk = combined
                return self
            self.msk = arg0
            return self

        if "msk" in kwargs and len(args) == 0 and len(kwargs) == 1:
            self.msk = kwargs["msk"]
            return self

        if len(args) == 2 and isinstance(args[0], str) and args[0].lstrip("-").strip() == "msk":
            self.msk = args[1]
            return self

        raise ValueError(
            "HERBERT:IX_mask:invalid_argument: "
            "Invalid arguments to IX_mask constructor."
        )

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Tuple[str, ...]:
        return ("msk",)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def msk(self) -> np.ndarray:
        return self._msk.copy()

    @msk.setter
    def msk(self, val: Any) -> None:
        if val is None:
            self._msk = np.zeros(0, dtype=int)
            return
        if isinstance(val, str):
            val = _str_to_iarray(val)
        elif isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(x, str) for x in val):
            val = np.concatenate([_str_to_iarray(x) for x in val])

        if isinstance(val, (bool, np.bool_)):
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra must be positive integers, not bool."
            )
        try:
            arr = np.asarray(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra must be a numeric array of positive integers."
            ) from exc

        if arr.dtype == bool:
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra must be positive integers, not boolean array."
            )
        arr_flat = arr.ravel()
        if arr_flat.size == 0:
            self._msk = np.zeros(0, dtype=int)
            return

        try:
            farr = arr_flat.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra must be numeric."
            ) from exc

        if np.any(np.isnan(farr)) or np.any( np.isinf(farr)):
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra cannot contain NaN or Inf."
            )
        if np.any(farr <= 0) or np.any(np.mod(farr, 1.0) != 0.0):
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Mask spectra must be strictly positive integers (> 0)."
            )
        self._msk = np.unique(farr.astype(int))

    # ------------------------------------------------------------------
    # Operations & ASCII I/O
    # ------------------------------------------------------------------
    def combine(self, *other_masks: Union["IX_mask", Sequence[int], np.ndarray]) -> "IX_mask":
        """Combine this mask with one or more other IX_mask objects (or integer arrays)."""
        arrays: List[np.ndarray] = [self._msk]
        for item in other_masks:
            if isinstance(item, IX_mask):
                arrays.append(item._msk)
            else:
                tmp = IX_mask(item)
                arrays.append(tmp._msk)
        all_msk = np.concatenate(arrays) if arrays else np.zeros(0, dtype=int)
        return IX_mask(np.unique(all_msk))

    @staticmethod
    def get_mask_ascii(filename: str) -> np.ndarray:
        """Read an ASCII .msk file and return sorted unique positive spectrum numbers."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Filename must be a non-empty string."
            )
        if not os.path.isfile(filename):
            raise ValueError(
                f"HERBERT:IX_mask:invalid_argument: "
                f"Mask file '{filename}' does not exist."
            )
        arrays: List[np.ndarray] = []
        with open(filename, "r", encoding="utf-8") as fh:
            for line in fh:
                arr = _str_to_iarray(line)
                if arr.size > 0:
                    if np.any(arr <= 0):
                        raise ValueError(
                            f"HERBERT:IX_mask:invalid_argument: "
                            f"Non-positive spectrum number in mask file '{filename}'."
                        )
                    arrays.append(arr)
        if not arrays:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate(arrays))

    @staticmethod
    def read_ascii(filename: str) -> "IX_mask":
        """Read an IX_mask object from an ASCII .msk file."""
        return IX_mask(IX_mask.get_mask_ascii(filename))

    @staticmethod
    def load_ascii(filename: str) -> "IX_mask":
        """Alias for read_ascii."""
        return IX_mask.read_ascii(filename)

    def put_mask_ascii(self, filename: str) -> None:
        """Write mask spectrum numbers to an ASCII .msk file."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                "HERBERT:IX_mask:invalid_argument: "
                "Filename must be a non-empty string."
            )
        lines = _iarray_to_str(self._msk)
        with open(filename, "w", encoding="utf-8") as fh:
            for line in lines:
                if line:
                    fh.write(line + "\n")

    def save_ascii(self, filename: str) -> None:
        """Save IX_mask to an ASCII .msk file."""
        self.put_mask_ascii(filename)
