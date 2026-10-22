"""IX_axis class definition (translated from @IX_axis/IX_axis.m)."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pyhorace.core.serializable import Serializable


class TicksDict(dict):
    """Dictionary with attribute-style access for ``positions`` and ``labels``.

    Matches MATLAB struct field access (``ticks.positions``, ``ticks.labels``)
    while remaining a standard Python ``dict`` (``ticks['positions']``).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'TicksDict' object has no attribute {name!r}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(
                f"'TicksDict' object has no attribute {name!r}"
            ) from exc

    def copy(self) -> TicksDict:
        return TicksDict(
            positions=np.copy(self.get("positions", np.empty(0, dtype=float))),
            labels=list(self.get("labels", [])),
        )


def _is_empty(val: Any) -> bool:
    """Return True if *val* is empty in the MATLAB ``isempty`` sense."""
    if val is None:
        return True
    if isinstance(val, (str, list, tuple, dict)):
        return len(val) == 0
    if isinstance(val, np.ndarray):
        return val.size == 0
    return False


def _is_numeric_array(val: Any) -> bool:
    """Return True if *val* is a numeric scalar, list/tuple of numbers, or numeric ndarray."""
    if isinstance(val, (bool, np.bool_)):
        return False
    if isinstance(val, (int, float, np.number)):
        return True
    if isinstance(val, np.ndarray):
        return np.issubdtype(val.dtype, np.number)
    if isinstance(val, (list, tuple)):
        if len(val) == 0:
            return True
        return all(_is_numeric_array(x) for x in val)
    return False


def _is_cellstr(val: Any) -> bool:
    """Return True if *val* is a list/tuple/object-array of strings."""
    if isinstance(val, (list, tuple)):
        return all(isinstance(x, str) for x in val)
    if isinstance(val, np.ndarray) and (
        val.dtype.kind in ("U", "S") or val.dtype == object
    ):
        return all(isinstance(x, str) for x in val.flat)
    return False


class IX_axis(Serializable):
    """IX_axis object contains axis description and tick information.

    The elements can be set from constructor and also accessed/modified from
    the object properties:

      caption     str or list[str]   Caption for axis (can be multiline input in
                                     the form of a list of strings or a string)
      units       str                Units for axis e.g. 'meV'
      code        str                Code for units (see documentation for
                                     built-in units; can also be user-defined
                                     unit code)

      vals        numeric            Array of tick positions
      labels      str or list[str]   Character array or list of tick labels
      ticks       dict / TicksDict   Tick information with two fields:
                                         positions    tick positions (numeric array)
                                         labels       list of tick labels
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create IX_axis object.

        Calling forms::

            >> w = IX_axis(caption)
            >> w = IX_axis(caption, units)
            >> w = IX_axis(code)           # set caption and units via a standard units code
            >> w = IX_axis(..., code)      # override caption and/or units from the defined code

        Setting custom tick positions and labels::

            >> w = IX_axis(..., vals)           # positions
            >> w = IX_axis(..., vals, labels)   # positions and labels
            >> w = IX_axis(..., ticks)          # structure/dict with positions and labels
        """
        super().__init__()
        self._caption: list[str] = []
        self._units: str = ""
        self._code: str = ""
        self._ticks: TicksDict = TicksDict(
            positions=np.empty(0, dtype=float),
            labels=[],
        )

        if len(args) > 0:
            self._build_ix_axis(*args)
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def caption(self) -> list[str]:
        """Caption for axis (always a list of strings)."""
        return self._caption

    @caption.setter
    def caption(self, cap: Any) -> None:
        self._check_and_set_caption(cap)

    @property
    def units(self) -> str:
        """Units for axis (character string)."""
        return self._units

    @units.setter
    def units(self, un: Any) -> None:
        self._check_and_set_units(un)

    @property
    def code(self) -> str:
        """Units code for axis (character string)."""
        return self._code

    @code.setter
    def code(self, cd: Any) -> None:
        self._check_and_set_code(cd)

    @property
    def ticks(self) -> TicksDict:
        """Tick information structure/dict with fields 'positions' and 'labels'."""
        return self._ticks

    @ticks.setter
    def ticks(self, tks: Any) -> None:
        self._check_and_set_ticks(tks)

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def make_label(self) -> tuple[list[str], bool]:
        """Create axis annotation from IX_axis object.

        Always returns a list of strings (cellstr) and a boolean indicating if
        the units string was appended.

        Returns
        -------
        label : list[str]
            Axis label lines.
        units_appended : bool
            True if units string was appended to the label.
        """
        if self.units:
            if self.caption:
                label = list(self.caption)
                label[-1] = f"{self.caption[-1]} ({self.units})"
            else:
                label = [f"({self.units})"]
            units_appended = True
        else:
            label = list(self.caption)
            units_appended = False
        return label, units_appended

    def init_from_structure(self, in_struct: dict[str, Any]) -> IX_axis:
        """Initialise IX_axis from a dictionary / structure."""
        for fld, val in in_struct.items():
            if not fld.startswith("serializable"):
                setattr(self, fld, val)
        return self

    def copy(self) -> IX_axis:
        """Return a deep copy of this IX_axis instance."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Private Validation & Construction Helpers
    # ------------------------------------------------------------------

    def _build_ix_axis(self, *args: Any) -> None:
        """Build IX_axis from constructor arguments (translated from buildIX_axis_.m)."""
        narg = len(args)
        if narg == 1 and isinstance(args[0], IX_axis):
            other = args[0]
            self._caption = list(other._caption)
            self._units = other._units
            self._code = other._code
            self._ticks = other._ticks.copy()
            return

        if narg == 1 and isinstance(args[0], dict) and not (
            set(args[0].keys()) == {"positions", "labels"}
        ):
            loaded = self.loadobj(args[0])
            self._caption = list(loaded._caption)
            self._units = loaded._units
            self._code = loaded._code
            self._ticks = loaded._ticks.copy()
            return

        if narg <= 4 and isinstance(args[-1], dict):
            nch = narg - 1
            if nch >= 1:
                self.caption = args[0]
            if nch >= 2:
                self.units = args[1]
            if nch >= 3:
                self.code = args[2]
            self.ticks = args[-1]
        elif narg <= 4 and _is_numeric_array(args[-1]) and not isinstance(args[-1], str):
            nch = narg - 1
            if nch >= 1:
                self.caption = args[0]
            if nch >= 2:
                self.units = args[1]
            if nch >= 3:
                self.code = args[2]
            if not _is_empty(args[-1]):
                pos = np.asarray(args[-1], dtype=float).ravel()
                tc = TicksDict(
                    positions=pos,
                    labels=[""] * pos.size,
                )
                self.ticks = tc
        elif (
            1 < narg <= 5
            and _is_numeric_array(args[-2])
            and not isinstance(args[-2], str)
        ):
            nch = narg - 2
            if nch >= 1:
                self.caption = args[0]
            if nch >= 2:
                self.units = args[1]
            if nch >= 3:
                self.code = args[2]
            tc = TicksDict(positions=args[-2], labels=args[-1])
            self.ticks = tc
            if not _is_empty(args[-2]):
                self._ticks["positions"] = np.asarray(args[-2], dtype=float).ravel()
            if not _is_empty(args[-1]):
                if isinstance(args[-1], str):
                    self._ticks["labels"] = [args[-1].rstrip()]
                elif isinstance(args[-1], (list, tuple, np.ndarray)):
                    self._ticks["labels"] = [str(x) for x in np.asarray(args[-1]).flat]
        elif narg <= 3:
            nch = narg
            if nch >= 1:
                self.caption = args[0]
            if nch >= 2:
                self.units = args[1]
            if nch >= 3:
                self.code = args[2]
        else:
            raise ValueError(
                "IX_axis:invalid_argument: wrong number and type of input arguments"
            )

    def _check_and_set_caption(self, cap: Any) -> None:
        """Check and set axis caption (translated from check_and_set_caption_.m)."""
        if _is_empty(cap):
            self._caption = []
            return
        if isinstance(cap, str):
            lines = cap.splitlines() if "\n" in cap else [cap]
            self._caption = [line.rstrip() for line in lines]
            return
        if _is_cellstr(cap):
            if isinstance(cap, np.ndarray):
                self._caption = [str(x) for x in cap.flat]
            else:
                self._caption = [str(x) for x in cap]
            return
        raise ValueError(
            f"HERBERT:IX_axis:invalid_argument: Caption must be character or cell array of strings. It is {type(cap).__name__}"
        )

    def _check_and_set_units(self, units: Any) -> None:
        """Check and set axis units (translated from check_and_set_units_.m)."""
        if _is_empty(units):
            self._units = ""
            return
        if isinstance(units, str) and "\n" not in units:
            self._units = units
            return
        raise ValueError(
            "IX_axis:invalid_argument: Axis units must be a character string"
        )

    def _check_and_set_code(self, code: Any) -> None:
        """Check and set axis units code (translated from check_and_set_code_.m)."""
        if _is_empty(code):
            self._code = ""
            return
        if isinstance(code, str) and "\n" not in code:
            self._code = code
            return
        raise ValueError(
            f"HERBERT:IX_axis:invalid_argument: Units code must be a character string. It is {type(code).__name__}"
        )

    def _check_and_set_ticks(self, ticks: Any) -> None:
        """Check and set axis ticks (translated from check_and_set_ticks_.m)."""
        if not isinstance(ticks, dict) and _is_empty(ticks):
            self._ticks = TicksDict(
                positions=np.empty(0, dtype=float),
                labels=[],
            )
            return

        if not isinstance(ticks, dict) or set(ticks.keys()) != {"positions", "labels"}:
            raise ValueError(
                "IX_axis:invalid_argument: ticks information must be a structure with fields 'positions' and 'labels'"
            )

        raw_pos = ticks["positions"]
        raw_lbl = ticks["labels"]

        if _is_empty(raw_pos):
            pos_arr = np.empty(0, dtype=float)
        elif _is_numeric_array(raw_pos):
            pos_arr = np.asarray(raw_pos, dtype=float).ravel()
        else:
            raise ValueError(
                "IX_axis:invalid_argument: tick positions must be a numeric vector"
            )

        if not _is_empty(raw_lbl):
            if isinstance(raw_lbl, str):
                n_lbl = len(raw_lbl)
            elif isinstance(raw_lbl, np.ndarray):
                n_lbl = raw_lbl.size
            elif isinstance(raw_lbl, (list, tuple)):
                n_lbl = len(raw_lbl)
            else:
                n_lbl = -1

            if n_lbl != pos_arr.size:
                raise ValueError(
                    "IX_axis:invalid_argument: If tick labels are provided, the number of labels must match the number of tick positions"
                )

        if _is_empty(raw_lbl):
            if pos_arr.size > 0:
                lbl_list: list[Any] = [None] * pos_arr.size
            else:
                lbl_list = []
        elif _is_cellstr(raw_lbl):
            if isinstance(raw_lbl, np.ndarray):
                lbl_list = [str(x) for x in raw_lbl.flat]
            else:
                lbl_list = [str(x) for x in raw_lbl]
        elif isinstance(raw_lbl, str):
            lbl_list = [c.strip() for c in raw_lbl]
        elif isinstance(raw_lbl, (list, tuple)) and all(x is None for x in raw_lbl):
            lbl_list = list(raw_lbl)
        else:
            raise ValueError(
                f"HORACE:IX_axis:invalid_argument: tick labels must be a cellstr or character array. It is {ticks!r}"
            )

        self._ticks = TicksDict(positions=pos_arr, labels=lbl_list)

    # ------------------------------------------------------------------
    # Serializable Interface
    # ------------------------------------------------------------------

    def class_version(self) -> int:
        """Return the serialization version of IX_axis."""
        return 1

    def classVersion(self) -> int:
        """Return the serialization version of IX_axis."""
        return self.class_version()

    def saveable_fields(self) -> list[str]:
        """Return the list of independent fields that define the state of IX_axis."""
        return ["caption", "units", "code", "ticks"]

    def saveableFields(self) -> list[str]:
        """Return the list of independent fields that define the state of IX_axis."""
        return self.saveable_fields()

    @classmethod
    def loadobj(cls, S: dict[str, Any] | IX_axis) -> Any:
        """Reconstruct IX_axis from a serialized dictionary."""
        if isinstance(S, cls):
            return S
        obj = cls()
        return Serializable.loadobj(S, obj)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IX_axis):
            return NotImplemented
        if (
            self._caption != other._caption
            or self._units != other._units
            or self._code != other._code
        ):
            return False
        if not np.array_equal(
            self._ticks["positions"], other._ticks["positions"]
        ):
            return False
        return self._ticks["labels"] == other._ticks["labels"]

    def __repr__(self) -> str:
        return (
            f"IX_axis(caption={self._caption!r}, units={self._units!r}, "
            f"code={self._code!r}, ticks={dict(self._ticks)!r})"
        )

