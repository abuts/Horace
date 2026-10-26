"""FixedNameList helper class for validating and resolving abbreviated model/shape names.

Ported from Herbert MATLAB @fixedNameList class:
- herbert_core/utilities/classes/@fixedNameList/fixedNameList.m
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


class fixedNameList:
    """
    Helper class storing a list of valid names and resolving case-insensitive
    unambiguous prefix abbreviations.
    """

    def __init__(self, names: Optional[Sequence[str]] = None) -> None:
        self._model_names: List[str] = []
        if names is not None:
            self.model_names = names

    @property
    def model_names(self) -> List[str]:
        """Return list of valid lowercase model names."""
        return list(self._model_names)

    @model_names.setter
    def model_names(self, val: Sequence[str]) -> None:
        if not isinstance(val, (list, tuple)) or not all(isinstance(x, str) for x in val):
            raise ValueError(
                f"HERBERT:fixedNameList:invalid_argument: Names must be a sequence of strings, got {val!r}"
            )
        self._model_names = [x.lower() for x in val]

    @property
    def model_names_(self) -> List[str]:
        return self._model_names

    @model_names_.setter
    def model_names_(self, val: Sequence[str]) -> None:
        self.model_names = val

    def valid_name(self, mod_name: Any) -> Tuple[str, int]:
        """
        Verify if the provided name is a member of the allowed names list
        (supporting case-insensitive prefix matching) and return the full name
        and its 1-based index in model_names.
        """
        if not isinstance(mod_name, str):
            raise ValueError(
                f"HERBERT:fixedNameList:invalid_argument: Name has to be a character string. "
                f"Provided {type(mod_name).__name__}"
            )
        mod_lower = mod_name.lower()
        matches = [
            (idx + 1, name)
            for idx, name in enumerate(self._model_names)
            if name.startswith(mod_lower)
        ]
        if len(matches) == 0:
            raise ValueError(
                f"HERBERT:fixedNameList:invalid_argument: Unrecognised name {mod_name}"
            )
        if len(matches) > 1:
            exact = [(idx, name) for idx, name in matches if name == mod_lower]
            if len(exact) == 1:
                return exact[0][1], exact[0][0]
            opts = [name for _, name in matches]
            raise ValueError(
                f"HERBERT:fixedNameList:invalid_argument: Ambiguous name: {mod_name}. "
                f"Acceptable options are: {opts}"
            )
        return matches[0][1], matches[0][0]


FixedNameList = fixedNameList

__all__ = ["FixedNameList", "fixedNameList"]

