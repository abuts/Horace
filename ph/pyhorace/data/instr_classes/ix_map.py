"""IX_map class defining detector-to-workspace spectrum mapping."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_mask import IX_mask, _iarray_to_str, _str_to_iarray


def _validate_int_vector(
    val: Any,
    name: str,
    allow_zero: bool = False,
    allow_empty: bool = True,
) -> np.ndarray:
    """Validate that ``val`` is a 1D integer array with elements > 0 (or >= 0 if allow_zero)."""
    if isinstance(val, (bool, np.bool_, str)) or val is None:
        if allow_empty and val is None:
            return np.zeros(0, dtype=int)
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' must be an integer array."
        )
    try:
        arr = np.asarray(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' must be an integer array."
        ) from exc

    if arr.dtype == bool or arr.dtype == object:
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' must be a numeric integer array."
        )
    flat = arr.ravel()
    if flat.size == 0:
        if allow_empty:
            return np.zeros(0, dtype=int)
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' cannot be empty."
        )
    try:
        farr = flat.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' must be numeric."
        ) from exc

    if np.any(np.isnan(farr)) or np.any(np.isinf(farr)):
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' cannot contain NaN or Inf."
        )
    min_val = 0.0 if allow_zero else 1.0
    if np.any(farr < min_val) or np.any(np.mod(farr, 1.0) != 0.0):
        cond = ">= 0" if allow_zero else "> 0"
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: '{name}' must contain integers {cond}."
        )
    return farr.astype(int)


def _sort_s_w(
    s: np.ndarray,
    w: np.ndarray,
    wkno_empty: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort (s, w) pairs by workspace then spectrum, remove duplicates, and check uniqueness of s across workspaces.

    Port of ``@IX_map/private/sort_s_w.m``.
    """
    s_arr = _validate_int_vector(s, "s", allow_zero=False, allow_empty=True)
    w_arr = _validate_int_vector(w, "w", allow_zero=False, allow_empty=True)
    if s_arr.size != w_arr.size:
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: "
            f"Spectrum array 's' ({s_arr.size}) and workspace array 'w' ({w_arr.size}) must have the same length."
        )
    empty_w = (
        _validate_int_vector(wkno_empty, "wkno", allow_zero=False, allow_empty=True)
        if wkno_empty is not None
        else np.zeros(0, dtype=int)
    )

    if s_arr.size > 0:
        pairs = np.column_stack((w_arr, s_arr))
        # Sort by w (col 0) then s (col 1) and remove duplicate (w, s) pairs
        unique_pairs = np.unique(pairs, axis=0)
        w_sorted = unique_pairs[:, 0]
        s_sorted = unique_pairs[:, 1]

        # Check that no spectrum belongs to two different workspaces
        if np.unique(s_sorted).size != s_sorted.size:
            raise ValueError(
                "HERBERT:IX_map:invalid_argument: "
                "One or more spectra belong to at least two different workspaces."
            )
    else:
        w_sorted = np.zeros(0, dtype=int)
        s_sorted = np.zeros(0, dtype=int)

    all_wkno = np.unique(np.concatenate([w_sorted, empty_w]))
    if all_wkno.size == 0:
        return (
            np.array([1], dtype=int),
            np.array([0], dtype=int),
            np.zeros(0, dtype=int),
        )

    if w_sorted.size > 0:
        w_vals, w_counts = np.unique(w_sorted, return_counts=True)
        ns = np.zeros(all_wkno.size, dtype=int)
        idx = np.searchsorted(all_wkno, w_vals)
        ns[idx] = w_counts
    else:
        ns = np.zeros(all_wkno.size, dtype=int)

    return all_wkno, ns, s_sorted


def _repeat_s_w_arrays(
    s: np.ndarray,
    w: np.ndarray,
    wkno_empty: np.ndarray,
    n_repeat: int,
    s_inc: int,
    w_inc: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Repeat (s, w, wkno_empty) blocks n_repeat times with increments s_inc and w_inc.

    Port of ``@IX_map/private/repeat_s_w_arrays.m``.
    """
    if n_repeat < 1:
        raise ValueError(
            f"HERBERT:IX_map:invalid_argument: Number of repeats must be >= 1, got {n_repeat}."
        )
    if n_repeat == 1:
        return s, w, wkno_empty

    s_list: List[np.ndarray] = []
    w_list: List[np.ndarray] = []
    empty_list: List[np.ndarray] = []
    for k in range(n_repeat):
        if s.size > 0:
            s_list.append(s + k * s_inc)
            w_list.append(w + k * w_inc)
        if wkno_empty.size > 0:
            empty_list.append(wkno_empty + k * w_inc)

    s_out = np.concatenate(s_list) if s_list else np.zeros(0, dtype=int)
    w_out = np.concatenate(w_list) if w_list else np.zeros(0, dtype=int)
    empty_out = np.concatenate(empty_list) if empty_list else np.zeros(0, dtype=int)
    return s_out, w_out, empty_out


class IX_map(Serializable):
    """Mapping of detector spectra to workspaces.

    Corresponds to MATLAB ``IX_map`` in Herbert/Horace.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._wkno: np.ndarray = np.array([1], dtype=int)
        self._ns: np.ndarray = np.array([0], dtype=int)
        self._s: np.ndarray = np.zeros(0, dtype=int)
        if len(args) == 0 and len(kwargs) == 0:
            return
        self.init(*args, **kwargs)

    # ------------------------------------------------------------------
    # Construction & initialization
    # ------------------------------------------------------------------
    def init(self, *args: Any, **kwargs: Any) -> "IX_map":
        """Initialize or re-initialize the IX_map instance."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, IX_map):
                self._wkno = arg0._wkno.copy()
                self._ns = arg0._ns.copy()
                self._s = arg0._s.copy()
                return self
            if isinstance(arg0, dict):
                if "serial_name" in arg0 and "version" in arg0:
                    loaded = self.loadobj(arg0)
                    if isinstance(loaded, IX_map):
                        self._wkno = loaded._wkno.copy()
                        self._ns = loaded._ns.copy()
                        self._s = loaded._s.copy()
                        return self
                return self.from_bare_struct(arg0)
            if isinstance(arg0, str):
                wkno, ns, s = self.get_map_ascii(arg0)
                self._wkno, self._ns, self._s = _sort_s_w(
                    s, np.repeat(wkno, ns), wkno[ns == 0]
                )
                return self

        # Check for direct ('wkno', ..., 'ns', ..., 's', ...) via kwargs or key-value args
        if set(kwargs.keys()) and set(kwargs.keys()) <= {"wkno", "ns", "s"} and len(args) == 0:
            self._do_check_combo_arg = False
            try:
                if "wkno" in kwargs:
                    self.wkno = kwargs["wkno"]
                if "ns" in kwargs:
                    self.ns = kwargs["ns"]
                if "s" in kwargs:
                    self.s = kwargs["s"]
            finally:
                self._do_check_combo_arg = True
            self.check_combo_arg()
            return self

        arg_list = list(args)
        # Check if 'wkno', 'ns', 's' key-value pairs are used positionally
        if len(arg_list) == 6 and all(isinstance(arg_list[i], str) for i in (0, 2, 4)):
            kv = {arg_list[i].lstrip("-").strip(): arg_list[i + 1] for i in (0, 2, 4)}
            if set(kv.keys()) == {"wkno", "ns", "s"}:
                return self.from_bare_struct(kv)

        # Check for trailing 'repeat' keyword
        repeat_params: Optional[Tuple[int, int, int]] = None
        for idx, item in enumerate(arg_list):
            if isinstance(item, str) and item.lstrip("-").strip().lower().startswith("rep"):
                rem = arg_list[idx + 1 :]
                arg_list = arg_list[:idx]
                if len(rem) == 1 and np.size(rem[0]) == 3:
                    r_arr = _validate_int_vector(rem[0], "repeat", allow_zero=True, allow_empty=False)
                    repeat_params = (int(r_arr[0]), int(r_arr[1]), int(r_arr[2]))
                elif len(rem) == 3:
                    n_rep = int(np.asarray(rem[0]).item())
                    s_inc = int(np.asarray(rem[1]).item())
                    w_inc = int(np.asarray(rem[2]).item())
                    repeat_params = (n_rep, s_inc, w_inc)
                else:
                    raise ValueError(
                        "HERBERT:IX_map:invalid_argument: "
                        "'repeat' must be followed by (n_repeat, s_inc, w_inc)."
                    )
                break

        if "repeat" in kwargs:
            r_val = kwargs.pop("repeat")
            r_arr = _validate_int_vector(r_val, "repeat", allow_zero=True, allow_empty=False)
            if r_arr.size != 3:
                raise ValueError(
                    "HERBERT:IX_map:invalid_argument: 'repeat' must have 3 elements (n_repeat, s_inc, w_inc)."
                )
            repeat_params = (int(r_arr[0]), int(r_arr[1]), int(r_arr[2]))

        if kwargs:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Unrecognized keyword arguments {list(kwargs.keys())}."
            )

        if len(arg_list) == 0 or len(arg_list) > 2:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Expected 1 or 2 positional arguments, got {len(arg_list)}."
            )

        first = arg_list[0]
        is_cell_of_s = self._is_cell_of_spectra(first)

        if is_cell_of_s:
            cell_s = [
                _validate_int_vector(el, "s", allow_zero=False, allow_empty=True)
                for el in first
            ]
            n_wk = len(cell_s)
            if len(arg_list) == 2:
                w_nums = _validate_int_vector(arg_list[1], "wkno", allow_zero=False, allow_empty=True)
                if w_nums.size != n_wk:
                    raise ValueError(
                        f"HERBERT:IX_map:invalid_argument: "
                        f"Number of workspace numbers ({w_nums.size}) must match number of spectrum blocks ({n_wk})."
                    )
            else:
                w_nums = np.arange(1, n_wk + 1, dtype=int)

            s_parts: List[np.ndarray] = []
            w_parts: List[np.ndarray] = []
            empty_wks: List[int] = []
            for wk_id, spec_block in zip(w_nums, cell_s):
                if spec_block.size == 0:
                    empty_wks.append(int(wk_id))
                else:
                    s_parts.append(spec_block)
                    w_parts.append(np.full(spec_block.size, wk_id, dtype=int))

            s_arr = np.concatenate(s_parts) if s_parts else np.zeros(0, dtype=int)
            w_arr = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=int)
            wkno_empty = np.array(empty_wks, dtype=int) if empty_wks else np.zeros(0, dtype=int)
        else:
            s_arr = _validate_int_vector(first, "s", allow_zero=False, allow_empty=True)
            if len(arg_list) == 2:
                w_arr = _validate_int_vector(arg_list[1], "w", allow_zero=False, allow_empty=True)
            else:
                w_arr = np.arange(1, s_arr.size + 1, dtype=int)
            wkno_empty = np.zeros(0, dtype=int)

        if repeat_params is not None:
            s_arr, w_arr, wkno_empty = _repeat_s_w_arrays(
                s_arr, w_arr, wkno_empty, *repeat_params
            )

        self._wkno, self._ns, self._s = _sort_s_w(s_arr, w_arr, wkno_empty)
        return self

    @staticmethod
    def _is_cell_of_spectra(arg: Any) -> bool:
        """Return True if ``arg`` is a list/tuple of spectrum arrays (MATLAB cell array of vectors)."""
        if isinstance(arg, np.ndarray):
            return arg.dtype == object
        if not isinstance(arg, (list, tuple)):
            return False
        if len(arg) == 0:
            return False
        return any(isinstance(el, (list, tuple, np.ndarray)) for el in arg)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> Tuple[str, ...]:
        return ("wkno", "ns", "s")

    def check_combo_arg(self) -> "IX_map":
        """Validate consistency of wkno, ns, and s and canonicalize order."""
        if self._wkno.size != self._ns.size:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: "
                f"Length of 'wkno' ({self._wkno.size}) must equal length of 'ns' ({self._ns.size})."
            )
        if np.unique(self._wkno).size != self._wkno.size:
            raise ValueError(
                "HERBERT:IX_map:invalid_argument: "
                "Workspace numbers in 'wkno' must be unique."
            )
        if int(np.sum(self._ns)) != self._s.size:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: "
                f"sum(ns) ({int(np.sum(self._ns))}) must equal length of 's' ({self._s.size})."
            )
        w_expanded = np.repeat(self._wkno, self._ns)
        empty_wks = self._wkno[self._ns == 0]
        self._wkno, self._ns, self._s = _sort_s_w(self._s, w_expanded, empty_wks)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def wkno(self) -> np.ndarray:
        return self._wkno.copy()

    @wkno.setter
    def wkno(self, val: Any) -> None:
        self._wkno = _validate_int_vector(val, "wkno", allow_zero=False, allow_empty=False)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def ns(self) -> np.ndarray:
        return self._ns.copy()

    @ns.setter
    def ns(self, val: Any) -> None:
        self._ns = _validate_int_vector(val, "ns", allow_zero=True, allow_empty=False)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def s(self) -> np.ndarray:
        return self._s.copy()

    @s.setter
    def s(self, val: Any) -> None:
        self._s = _validate_int_vector(val, "s", allow_zero=False, allow_empty=True)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def w(self) -> np.ndarray:
        """Workspace number for every spectrum in ``s``."""
        return np.repeat(self._wkno, self._ns)

    @property
    def total_spec(self) -> int:
        """Total number of mapped spectra."""
        return int(self._s.size)

    @property
    def nwork(self) -> int:
        """Number of workspaces in the map."""
        return int(self._wkno.size)

    @property
    def wend(self) -> np.ndarray:
        """1-based ending index in ``s`` for each workspace."""
        return np.cumsum(self._ns)

    @property
    def wbeg(self) -> np.ndarray:
        """1-based starting index in ``s`` for each workspace."""
        return self.wend - self._ns + 1

    @property
    def w_ind(self) -> List[np.ndarray]:
        """List of 1-based index arrays into ``s`` for each workspace."""
        begs = self.wbeg
        ends = self.wend
        out: List[np.ndarray] = []
        for b, e in zip(begs, ends):
            if e >= b:
                out.append(np.arange(b, e + 1, dtype=int))
            else:
                out.append(np.zeros(0, dtype=int))
        return out

    # ------------------------------------------------------------------
    # Operations: combine, concatenate, mask, section, product
    # ------------------------------------------------------------------
    def combine(self, *other_maps: "IX_map") -> "IX_map":
        """Combine this map with one or more IX_map objects.

        Spectra belonging to the same workspace number across maps are merged;
        if a spectrum belongs to two different workspaces, a ValueError is raised.
        Port of ``@IX_map/combine.m``.
        """
        maps = [self] + list(other_maps)
        for m in maps:
            if not isinstance(m, IX_map):
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: Expected IX_map, got {type(m).__name__}."
                )
        s_all = np.concatenate([m._s for m in maps])
        w_all = np.concatenate([m.w for m in maps])
        empty_all = np.concatenate([m._wkno[m._ns == 0] for m in maps])

        wkno, ns, s = _sort_s_w(s_all, w_all, empty_all)
        res = IX_map()
        res._wkno, res._ns, res._s = wkno, ns, s
        return res

    def concatenate(self, *other_maps: "IX_map") -> "IX_map":
        """Concatenate workspaces of multiple IX_map objects by offsetting workspace numbers.

        Port of ``@IX_map/concatenate.m``.
        """
        maps = [self] + list(other_maps)
        for m in maps:
            if not isinstance(m, IX_map):
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: Expected IX_map, got {type(m).__name__}."
                )
        s_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        empty_list: List[np.ndarray] = []
        w_offset = 0

        for m in maps:
            if m._s.size == 0 and m._wkno.size == 1 and m._ns[0] == 0 and m._wkno[0] == 1 and len(maps) > 1:
                # Still offset if non-empty or standard empty
                pass
            s_list.append(m._s)
            w_list.append(m.w + w_offset)
            empty_wks = m._wkno[m._ns == 0]
            if empty_wks.size > 0:
                empty_list.append(empty_wks + w_offset)
            if m._wkno.size > 0:
                w_offset += int(np.max(m._wkno))

        s_all = np.concatenate(s_list) if s_list else np.zeros(0, dtype=int)
        w_all = np.concatenate(w_list) if w_list else np.zeros(0, dtype=int)
        empty_all = np.concatenate(empty_list) if empty_list else np.zeros(0, dtype=int)

        wkno, ns, s = _sort_s_w(s_all, w_all, empty_all)
        res = IX_map()
        res._wkno, res._ns, res._s = wkno, ns, s
        return res

    def mask(self, mask_arg: Union[IX_mask, Sequence[int], np.ndarray]) -> "IX_map":
        """Remove masked spectra from this map while keeping all workspace numbers.

        Port of ``@IX_map/mask.m``.
        """
        if isinstance(mask_arg, IX_mask):
            msk_spectra = mask_arg.msk
        else:
            msk_spectra = IX_mask(mask_arg).msk

        if msk_spectra.size == 0 or self._s.size == 0:
            return IX_map(self)

        keep = ~np.isin(self._s, msk_spectra)
        s_kept = self._s[keep]
        w_kept = self.w[keep]

        wkno, ns, s = _sort_s_w(s_kept, w_kept, self._wkno)
        res = IX_map()
        res._wkno, res._ns, res._s = wkno, ns, s
        return res

    def section(self, wkno_keep: Union[int, Sequence[int], np.ndarray]) -> "IX_map":
        """Retain only the specified workspace numbers ``wkno_keep``.

        Port of ``@IX_map/section.m``.
        """
        keep_wks = np.unique(_validate_int_vector(wkno_keep, "wkno_keep", allow_zero=False, allow_empty=True))
        if keep_wks.size == 0:
            return IX_map()

        if not np.all(np.isin(keep_wks, self._wkno)):
            missing = keep_wks[~np.isin(keep_wks, self._wkno)]
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: "
                f"Workspace number(s) {missing} are not present in this IX_map."
            )

        w_curr = self.w
        sel = np.isin(w_curr, keep_wks)
        wkno, ns, s = _sort_s_w(self._s[sel], w_curr[sel], keep_wks)
        res = IX_map()
        res._wkno, res._ns, res._s = wkno, ns, s
        return res

    def product(self, map2: "IX_map") -> "IX_map":
        """Compute the composite map ``map2(self)`` where workspaces of ``self`` are spectra of ``map2``.

        Port of ``@IX_map/product.m``.
        """
        if not isinstance(map2, IX_map):
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Expected IX_map, got {type(map2).__name__}."
            )
        w1 = self.w
        s1 = self._s
        s2 = map2._s
        w2 = map2.w

        if s1.size == 0 or s2.size == 0:
            res = IX_map()
            res._wkno = map2._wkno.copy()
            res._ns = np.zeros(map2._wkno.size, dtype=int)
            res._s = np.zeros(0, dtype=int)
            return res

        # Since s2 has unique elements, map each w1 in s2 to its corresponding w2
        in_map2 = np.isin(w1, s2)
        s_prod = s1[in_map2]
        w1_matched = w1[in_map2]

        sort_idx = np.argsort(s2)
        s2_sorted = s2[sort_idx]
        w2_sorted = w2[sort_idx]
        pos = np.searchsorted(s2_sorted, w1_matched)
        w_prod = w2_sorted[pos]

        wkno, ns, s = _sort_s_w(s_prod, w_prod, map2._wkno)
        res = IX_map()
        res._wkno, res._ns, res._s = wkno, ns, s
        return res

    # ------------------------------------------------------------------
    # ASCII .map file I/O (get_map_ascii.m & put_map_ascii.m)
    # ------------------------------------------------------------------
    @staticmethod
    def get_map_ascii(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read an ISIS ASCII .map file and return (wkno, ns, s)."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                "HERBERT:IX_map:invalid_argument: Filename must be a non-empty string."
            )
        if not os.path.isfile(filename):
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Map file '{filename}' does not exist."
            )

        lines: List[str] = []
        with open(filename, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                # Strip comments starting with !, #, %
                line = raw_line
                for c in ("!", "#", "%"):
                    pos = line.find(c)
                    if pos != -1:
                        line = line[:pos]
                line = line.strip()
                if line:
                    lines.append(line)

        if not lines:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Map file '{filename}' is empty."
            )

        try:
            header_vals = _str_to_iarray(lines[0])
            if header_vals.size < 1 or header_vals[0] < 0:
                raise ValueError("Invalid workspace count in header.")
            n_wk = int(header_vals[0])
        except ValueError as exc:
            raise ValueError(
                f"HERBERT:IX_map:invalid_argument: Invalid header in map file '{filename}'."
            ) from exc

        if n_wk == 0:
            return (
                np.array([1], dtype=int),
                np.array([0], dtype=int),
                np.zeros(0, dtype=int),
            )

        wkno_list: List[int] = []
        ns_list: List[int] = []
        s_list: List[np.ndarray] = []

        line_idx = 1
        for _ in range(n_wk):
            if line_idx >= len(lines):
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: Unexpected end of map file '{filename}'."
                )
            wk_vals = _str_to_iarray(lines[line_idx])
            line_idx += 1
            if wk_vals.size == 1:
                wk_num = int(wk_vals[0])
                if line_idx >= len(lines):
                    raise ValueError(
                        f"HERBERT:IX_map:invalid_argument: Missing spectrum count for workspace {wk_num}."
                    )
                ns_vals = _str_to_iarray(lines[line_idx])
                line_idx += 1
                if ns_vals.size != 1:
                    raise ValueError(
                        f"HERBERT:IX_map:invalid_argument: Invalid spectrum count for workspace {wk_num}."
                    )
                n_spec = int(ns_vals[0])
            elif wk_vals.size == 2:
                wk_num = int(wk_vals[0])
                n_spec = int(wk_vals[1])
            else:
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: Invalid workspace header in '{filename}'."
                )

            if wk_num <= 0 or n_spec < 0:
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: Invalid workspace number ({wk_num}) or spectrum count ({n_spec})."
                )

            spec_collected: List[np.ndarray] = []
            count_collected = 0
            while count_collected < n_spec:
                if line_idx >= len(lines):
                    raise ValueError(
                        f"HERBERT:IX_map:invalid_argument: "
                        f"Expected {n_spec} spectra for workspace {wk_num}, got {count_collected}."
                    )
                sp = _str_to_iarray(lines[line_idx])
                line_idx += 1
                spec_collected.append(sp)
                count_collected += sp.size

            if count_collected != n_spec:
                raise ValueError(
                    f"HERBERT:IX_map:invalid_argument: "
                    f"Workspace {wk_num} declared {n_spec} spectra, but read {count_collected}."
                )

            wkno_list.append(wk_num)
            ns_list.append(n_spec)
            if spec_collected:
                s_list.append(np.concatenate(spec_collected))

        wkno_arr = np.array(wkno_list, dtype=int)
        ns_arr = np.array(ns_list, dtype=int)
        s_arr = np.concatenate(s_list) if s_list else np.zeros(0, dtype=int)
        return wkno_arr, ns_arr, s_arr

    @staticmethod
    def read_ascii(filename: str) -> "IX_map":
        """Read an IX_map from an ISIS ASCII .map file."""
        return IX_map(filename)

    @staticmethod
    def load_ascii(filename: str) -> "IX_map":
        """Alias for read_ascii."""
        return IX_map.read_ascii(filename)

    def put_map_ascii(self, filename: str) -> None:
        """Write this IX_map to an ISIS ASCII .map file."""
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(
                "HERBERT:IX_map:invalid_argument: Filename must be a non-empty string."
            )
        begs = self.wbeg
        ends = self.wend
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(f"{self.nwork}\n")
            for wk_num, n_sp, b, e in zip(self._wkno, self._ns, begs, ends):
                fh.write(f"! Workspace {wk_num}\n")
                fh.write(f"{wk_num}\n")
                fh.write(f"{n_sp}\n")
                if n_sp > 0:
                    sp_slice = self._s[b - 1 : e]
                    for line in _iarray_to_str(sp_slice):
                        if line:
                            fh.write(line + "\n")

    def save_ascii(self, filename: str) -> None:
        """Save this IX_map to an ISIS ASCII .map file."""
        self.put_map_ascii(filename)
