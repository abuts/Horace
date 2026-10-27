"""IX_aperture class definition.

Ported from Herbert MATLAB @IX_aperture class:
- herbert_core/classes/instrument_classes/instrument/@IX_aperture/IX_aperture.m
- herbert_core/classes/instrument_classes/instrument/@IX_aperture/covariance.m
- herbert_core/classes/instrument_classes/instrument/@IX_aperture/rand.m
- herbert_core/classes/instrument_classes/instrument/@IX_aperture/private/convert_old_struct_.m
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import hashable
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.pdf_table import _parse_rand_shape


def _is_numeric_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(val, (int, float, np.number)):
        return not (np.isnan(val) or np.isinf(val))
    if isinstance(val, np.ndarray) and val.size == 1 and np.issubdtype(val.dtype, np.number):
        v = float(val.flat[0])
        return not (np.isnan(v) or np.isinf(v))
    return False


@register_serializable
class IX_aperture(hashable):
    """
    Aperture class definition (distance, width, height, name).
    Ported from Herbert MATLAB `@IX_aperture`.
    """

    _name: str = PrivateAttr(default="")
    _distance: float = PrivateAttr(default=0.0)
    _width: float = PrivateAttr(default=0.0)
    _height: float = PrivateAttr(default=0.0)
    _mandatory_field_set: List[bool] = PrivateAttr(
        default_factory=lambda: [False, False, False]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._name = ""
        self._distance = 0.0
        self._width = 0.0
        self._height = 0.0
        self._mandatory_field_set = [False, False, False]
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], IX_aperture) and not kwargs:
            other = args[0]
            self._name = other._name
            self._distance = other._distance
            self._width = other._width
            self._height = other._height
            self._mandatory_field_set = list(other._mandatory_field_set)
            self._hash_value = other._hash_value
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_aperture):
                self._name = loaded._name
                self._distance = loaded._distance
                self._width = loaded._width
                self._height = loaded._height
                self._mandatory_field_set = list(loaded._mandatory_field_set)
                self._hash_value = loaded._hash_value
            return

        pos_params = list(self.saveable_fields())
        argi = list(args)
        if (
            len(argi) > 0
            and isinstance(argi[0], str)
            and not argi[0].startswith("-")
            and argi[0] not in pos_params
            and not any(
                p.startswith(argi[0]) and len(argi[0]) >= min(4, len(p))
                for p in pos_params
            )
        ):
            self.name = argi[0]
            argi = argi[1:]

        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, True, *argi, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_aperture:invalid_argument: Unrecognized extra parameters provided "
                f"as input to IX_aperture constructor: {remains}"
            )

    def _clone_for_hash(self) -> "IX_aperture":
        cloned = super()._clone_for_hash()
        if isinstance(cloned, IX_aperture):
            cloned._mandatory_field_set = list(self._mandatory_field_set)
        return cloned

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, np.ndarray)) and len(val) == 0):
            val = ""
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:IX_apperture:invalid_argument: "
                "Sample name must be a character string (or empty string)"
            )
        self._name = val
        self.clear_hash()

    @property
    def distance(self) -> float:
        return self._distance

    @distance.setter
    def distance(self, val: Any) -> None:
        if not _is_numeric_scalar(val):
            raise ValueError(
                "HERBERT:IX_apperture:invalid_argument: Distance must be a numeric scalar"
            )
        self._distance = float(np.asarray(val).flat[0])
        self._mandatory_field_set[0] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, val: Any) -> None:
        if not _is_numeric_scalar(val) or float(np.asarray(val).flat[0]) < 0:
            raise ValueError(
                "HERBERT:IX_apperture:invalid_argument: "
                "Aperture width must be a numeric scalar greater than or equal to zero"
            )
        self._width = float(np.asarray(val).flat[0])
        self._mandatory_field_set[1] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, val: Any) -> None:
        if not _is_numeric_scalar(val) or float(np.asarray(val).flat[0]) < 0:
            raise ValueError(
                "HERBERT:IX_apperture:invalid_argument: "
                "Aperture height must be a numeric scalar greater than or equal to zero"
            )
        self._height = float(np.asarray(val).flat[0])
        self._mandatory_field_set[2] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    # Private property aliases
    @property
    def name_(self) -> str:
        return self._name

    @property
    def distance_(self) -> float:
        return self._distance

    @property
    def width_(self) -> float:
        return self._width

    @property
    def height_(self) -> float:
        return self._height

    # -------------------------------------------------------------------------
    # Calculation Methods (covariance, rand)
    # -------------------------------------------------------------------------

    def covariance(self) -> np.ndarray:
        """
        Return 2x2 covariance matrix for the aperture shape.
        Ported from `@IX_aperture/covariance.m`.
        """
        C = np.zeros((2, 2), dtype=float)
        C[0, 0] = (self._width**2) / 12.0
        C[1, 1] = (self._height**2) / 12.0
        return C

    def rand(self, *args: Any, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate random points in the aperture.
        Returns an array of shape `(2, *sz)` with singleton dimensions in `sz` squeezed away.
        Ported from `@IX_aperture/rand.m`.
        """
        sz = _parse_rand_shape(args)
        if rng is not None:
            rx = rng.random(sz)
            ry = rng.random(sz)
        else:
            rx = np.random.rand(*sz)
            ry = np.random.rand(*sz)
        x = self._width * (rx - 0.5)
        y = self._height * (ry - 0.5)

        X = np.vstack([x.ravel(order="F"), y.ravel(order="F")])
        X = X.reshape((2, *sz), order="F")
        return np.squeeze(X)

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def saveable_fields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        flds = ("distance", "width", "height", "name")
        if mandatory is not None and mandatory is not False:
            return flds[:3]
        return flds

    def saveableFields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return self.saveable_fields(mandatory=mandatory)

    def class_version(self) -> int:
        return 2

    def check_combo_arg(self) -> "IX_aperture":
        self.clear_hash()
        if not all(self._mandatory_field_set):
            mandatory_names = self.saveable_fields(mandatory=True)
            unset = [
                n
                for n, is_set in zip(mandatory_names, self._mandatory_field_set)
                if not is_set
            ]
            raise ValueError(
                f"HERBERT:IX_aperture:invalid_argument: Must give all mandatory properties namely: "
                f"{mandatory_names}. Properties: {unset} have not been set"
            )
        return self

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "IX_aperture"]:
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "name_" in inputs:
            inputs.pop("class_version_", None)
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs
        return inputs, self


__all__ = ["IX_aperture"]

