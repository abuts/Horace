"""IX_null_sample class definition.

Ported from Herbert MATLAB @IX_null_sample class:
- herbert_core/classes/instrument_classes/instrument/@IX_null_sample/IX_null_sample.m
- herbert_core/classes/instrument_classes/instrument/@IX_null_sample/private/convert_old_struct_.m
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_samp import IX_samp


@register_serializable
class IX_null_sample(IX_samp):
    """
    Null sample object representing the absence of sample data.
    Ported from Herbert MATLAB `@IX_null_sample`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_null_sample):
                self._copy_base_state_from(loaded)
            return

        if len(args) == 3 and not kwargs:
            self.name = args[0]
            self.alatt = args[1]
            self.angdeg = args[2]
            return

        raise ValueError(
            "HORACE:IX_null_sample:invalid_argument: Invalid number of input arguments"
        )

    def _get_lattice(self) -> np.ndarray:
        return np.array([], dtype=float)

    def _get_angles(self) -> np.ndarray:
        return np.array([], dtype=float)

    def _get_name(self) -> str:
        return ""

    def null_struct(self) -> Dict[str, Any]:
        """Return an empty structure dictionary."""
        return {}

    def class_version(self) -> int:
        return 2

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "IX_null_sample"]:
        """
        Convert old structure versions (v0, v1) to current v2 format.
        Ported from `@IX_null_sample/private/convert_old_struct_.m`.
        """
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "alatt_" in inputs:
            inputs.pop("class_version_", None)
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs
        return inputs, self


__all__ = ["IX_null_sample"]
