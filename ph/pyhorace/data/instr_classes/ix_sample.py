"""IX_sample class definition.

Ported from Herbert MATLAB @IX_sample class:
- herbert_core/classes/instrument_classes/instrument/@IX_sample/IX_sample.m
- herbert_core/classes/instrument_classes/instrument/@IX_sample/covariance.m
- herbert_core/classes/instrument_classes/instrument/@IX_sample/rand.m
- herbert_core/classes/instrument_classes/instrument/@IX_sample/rand_mosaic.m
- herbert_core/classes/instrument_classes/instrument/@IX_sample/private/convert_old_struct_.m
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.coord_transf import (
    bmatrix,
    rotvec_to_rotmat,
    ubmatrix,
)
from pyhorace.data.instr_classes.fixed_name_list import fixedNameList
from pyhorace.data.instr_classes.ix_mosaic import (
    IX_mosaic,
    _is_numeric_array_or_scalar,
)
from pyhorace.core.serializable import register_serializable
from pyhorace.data.instr_classes.ix_samp import IX_samp
from pyhorace.data.instr_classes.pdf_table import _parse_rand_shape


def _is_numeric_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(val, (int, float, np.number)):
        return not (np.isnan(val) or np.isinf(val))
    if (
        isinstance(val, np.ndarray)
        and val.size == 1
        and np.issubdtype(val.dtype, np.number)
    ):
        v = float(val.flat[0])
        return not (np.isnan(v) or np.isinf(v))
    return False


def _is_lognumscalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_)):
        return True
    if _is_numeric_scalar(val):
        v = float(np.asarray(val).flat[0])
        return v in (0.0, 1.0)
    return False


@register_serializable
class IX_sample(IX_samp):
    """
    Sample class definition (crystal orientation, geometry, shape, mosaic, temperature).
    Ported from Herbert MATLAB `@IX_sample`.
    """

    _shapes: ClassVar[fixedNameList] = fixedNameList(["point", "cuboid"])
    _n_ps: ClassVar[Dict[str, int]] = {"point": 0, "cuboid": 3}

    _hall_symbol: str = PrivateAttr(default="")
    _single_crystal: bool = PrivateAttr(default=True)
    _xgeom: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )
    _ygeom: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float)
    )
    _shape: str = PrivateAttr(default="point")
    _ps: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([], dtype=float)
    )
    _eta: IX_mosaic = PrivateAttr(default_factory=IX_mosaic)
    _temperature: float = PrivateAttr(default=0.0)
    _mandatory_field_set: List[bool] = PrivateAttr(
        default_factory=lambda: [False, False, False, False]
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._hall_symbol = ""
        self._single_crystal = True
        self._xgeom = np.array([1.0, 0.0, 0.0], dtype=float)
        self._ygeom = np.array([0.0, 1.0, 0.0], dtype=float)
        self._shape = "point"
        self._ps = np.array([], dtype=float)
        self._eta = IX_mosaic()
        self._temperature = 0.0
        self._mandatory_field_set = [False, False, False, False]
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            loaded = self.loadobj(args[0])
            if isinstance(loaded, IX_sample):
                self._copy_sample_state_from(loaded)
            return

        if len(args) == 2 and not kwargs and not isinstance(args[0], str):
            self.alatt = args[0]
            self.angdeg = args[1]
            return

        interface1_par = [
            "single_crystal",
            "xgeom",
            "ygeom",
            "shape",
            "ps",
            "eta",
            "temperature",
            "hall_symbol",
        ]
        interface2_par = [
            "xgeom",
            "ygeom",
            "shape",
            "ps",
            "eta",
            "temperature",
            "hall_symbol",
        ]
        base_par = list(IX_samp.fields_to_save_)

        argi = list(args)
        if len(argi) > 0:
            arg1 = argi[0]
            is_name_first = False
            if (
                isinstance(arg1, str)
                and not arg1.startswith("-")
                and arg1 not in interface1_par
                and arg1 not in base_par
                and not any(
                    p.startswith(arg1) and len(arg1) >= min(4, len(p))
                    for p in interface1_par + base_par
                )
            ):
                is_name_first = True
            elif (
                isinstance(arg1, str)
                and arg1 == "name"
                and len(argi) >= 2
                and isinstance(argi[1], (bool, np.bool_))
            ):
                # Legacy constructor form where sample name is literally 'name'
                # followed by boolean single_crystal
                is_name_first = True

            if is_name_first:
                self.name = arg1
                argi = argi[1:]
                pos_params = interface1_par + base_par
            elif isinstance(arg1, (bool, np.bool_)):
                pos_params = interface1_par + base_par
            else:
                pos_params = interface2_par + base_par
        else:
            pos_params = interface2_par + base_par

        _, remains = self.set_positional_and_key_val_arguments(
            pos_params, True, *argi, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_sample:invalid_argument: Unrecognized extra parameters provided "
                f"as input to IX_sample constructor: {remains}"
            )

    def _copy_sample_state_from(self, other: "IX_sample") -> None:
        self._copy_base_state_from(other)
        self._hall_symbol = other._hall_symbol
        self._single_crystal = other._single_crystal
        self._xgeom = other._xgeom.copy()
        self._ygeom = other._ygeom.copy()
        self._shape = other._shape
        self._ps = other._ps.copy()
        self._eta = other._eta.__copy__()
        self._temperature = other._temperature
        self._mandatory_field_set = list(other._mandatory_field_set)
        self._hash_value = other._hash_value

    def __copy__(self) -> "IX_sample":
        cloned = IX_sample()
        cloned._copy_sample_state_from(self)
        cloned._do_check_combo_arg = self._do_check_combo_arg
        return cloned

    def _clone_for_hash(self) -> "IX_sample":
        return self.__copy__()

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def eta(self) -> IX_mosaic:
        return self._eta

    @eta.setter
    def eta(self, val: Any) -> None:
        if isinstance(val, IX_mosaic):
            self._eta = val
        elif isinstance(val, dict):
            loaded = IX_mosaic.loadobj(val)
            if isinstance(loaded, IX_mosaic):
                self._eta = loaded
            else:
                raise ValueError(
                    "HERBERT:IX_sample:invalid_argument: "
                    "Mosaic spread must be numeric or an IX_mosaic object"
                )
        elif _is_numeric_array_or_scalar(val):
            self._eta = IX_mosaic(val)
        else:
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Mosaic spread must be numeric or an IX_mosaic object"
            )
        self.clear_hash()

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._temperature = 0.0
            self.clear_hash()
            return
        if not _is_numeric_scalar(val) or float(np.asarray(val).flat[0]) < 0.0:
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Temperature must be numeric scalar greater than or equal to zero"
            )
        self._temperature = float(np.asarray(val).flat[0])
        self.clear_hash()

    @property
    def hall_symbol(self) -> str:
        return self._hall_symbol

    @hall_symbol.setter
    def hall_symbol(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._hall_symbol = ""
            self.clear_hash()
            return
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Sample Hall symbol must be a character string (or empty string)"
            )
        self._hall_symbol = val
        self.clear_hash()

    @property
    def xgeom(self) -> np.ndarray:
        return self._xgeom

    @xgeom.setter
    def xgeom(self, val: Any) -> None:
        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: 'xgeom' must be a three-vector"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.all(arr == 0.0):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: 'xgeom' must be a three-vector"
            )
        self._xgeom = arr
        self._mandatory_field_set[0] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def ygeom(self) -> np.ndarray:
        return self._ygeom

    @ygeom.setter
    def ygeom(self, val: Any) -> None:
        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: 'ygeom' must be a three-vector"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.all(arr == 0.0):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: 'ygeom' must be a three-vector"
            )
        self._xgeom = self._xgeom
        self._ygeom = arr
        self._mandatory_field_set[1] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def xy_geom(self) -> np.ndarray:
        return np.vstack([self._xgeom, self._ygeom])

    @xy_geom.setter
    def xy_geom(self, val: Any) -> None:
        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "'xy_geom' must be a 2x3 matrix, combining two non-zero 3-vectors as strings"
            )
        arr = np.asarray(val, dtype=float)
        if (
            arr.shape != (2, 3)
            or np.all(arr[0] == 0.0)
            or np.all(arr[1] == 0.0)
        ):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "'xy_geom' must be a 2x3 matrix, combining two non-zero 3-vectors as strings"
            )
        self._xgeom = arr[0].copy()
        self._ygeom = arr[1].copy()
        self._mandatory_field_set[0] = True
        self._mandatory_field_set[1] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def single_crystal(self) -> bool:
        return self._single_crystal

    @single_crystal.setter
    def single_crystal(self, val: Any) -> None:
        if not _is_lognumscalar(val):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "single_crystal must true or false (or 1 or 0)"
            )
        if isinstance(val, (bool, np.bool_)):
            self._single_crystal = bool(val)
        else:
            self._single_crystal = bool(float(np.asarray(val).flat[0]) != 0.0)
        self.clear_hash()

    @property
    def shape(self) -> str:
        return self._shape

    @shape.setter
    def shape(self, val: Any) -> None:
        if not isinstance(val, str) or len(val.strip()) == 0:
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Sample shape must be a non-empty character string"
            )
        try:
            fullname, _ = self._shapes.valid_name(val)
        except ValueError as exc:
            raise ValueError(
                f"HERBERT:IX_sample:invalid_argument: Sample shape: {exc}"
            ) from exc
        self._shape = fullname
        self._mandatory_field_set[2] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def ps(self) -> np.ndarray:
        return self._ps

    @ps.setter
    def ps(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._ps = np.array([], dtype=float)
            self._mandatory_field_set[3] = True
            if self._do_check_combo_arg:
                self.check_combo_arg()
            else:
                self.clear_hash()
            return

        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Sample parameters must be a numeric vector"
            )
        arr = np.asarray(val, dtype=float)
        if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 1):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Sample parameters must be a numeric vector"
            )
        self._ps = arr.ravel()
        self._mandatory_field_set[3] = True
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    # Private property aliases
    @property
    def hall_symbol_(self) -> str:
        return self._hall_symbol

    @property
    def single_crystal_(self) -> bool:
        return self._single_crystal

    @property
    def xgeom_(self) -> np.ndarray:
        return self._xgeom

    @property
    def ygeom_(self) -> np.ndarray:
        return self._ygeom

    @property
    def shape_(self) -> str:
        return self._shape

    @property
    def ps_(self) -> np.ndarray:
        return self._ps

    @property
    def eta_(self) -> IX_mosaic:
        return self._eta

    @property
    def temperature_(self) -> float:
        return self._temperature

    # -------------------------------------------------------------------------
    # Calculation Methods (covariance, rand, rand_mosaic)
    # -------------------------------------------------------------------------

    def covariance(self) -> np.ndarray:
        """
        Return 3x3 covariance matrix for the sample shape.
        Ported from `@IX_sample/covariance.m`.
        """
        if self._shape == "cuboid":
            return np.diag((self._ps[:3] ** 2) / 12.0)
        if self._shape == "point":
            return np.zeros((3, 3), dtype=float)
        raise ValueError(
            f"HERBERT:IX_sample:invalid_argument: Unrecognised sample shape '{self._shape}'"
        )

    def rand(
        self, *args: Any, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        Generate random points in the sample shape.
        Returns an array of shape `(3, *sz)` with singleton dimensions squeezed away.
        Ported from `@IX_sample/rand.m`.
        """
        sz = _parse_rand_shape(args)
        if self._shape == "cuboid":
            if rng is not None:
                rx = rng.random(sz)
                ry = rng.random(sz)
                rz = rng.random(sz)
            else:
                rx = np.random.rand(*sz)
                ry = np.random.rand(*sz)
                rz = np.random.rand(*sz)
            x = self._ps[0] * (rx - 0.5)
            y = self._ps[1] * (ry - 0.5)
            z = self._ps[2] * (rz - 0.5)
            X = np.vstack(
                [x.ravel(order="F"), y.ravel(order="F"), z.ravel(order="F")]
            )
            X = X.reshape((3, *sz), order="F")
            return np.squeeze(X)
        if self._shape == "point":
            X = np.zeros((3, *sz), dtype=float)
            return np.squeeze(X)
        raise ValueError(
            f"HERBERT:IX_sample:invalid_argument: Unrecognised sample shape '{self._shape}'"
        )

    def rand_mosaic(
        self,
        sz: Any,
        alatt: Union[Sequence[float], np.ndarray],
        angdeg: Union[Sequence[float], np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate random rotation matrices representing mosaic spread in the crystal RLU frame.
        Ported from `@IX_sample/rand_mosaic.m`.

        Parameters
        ----------
        sz : int or Sequence[int]
            Size of the random sample array.
        alatt : array_like
            Lattice parameters `[a, b, c]` in Angstroms.
        angdeg : array_like
            Lattice angles `[alpha, beta, gamma]` in degrees.
        rng : np.random.Generator, optional
            Optional NumPy random generator.

        Returns
        -------
        Rrlu : np.ndarray
            Array of 3x3 rotation matrices in reciprocal lattice units with shape `(3, *X.shape)`.
        """
        mosaic = self._eta
        B = bmatrix(alatt, angdeg, nargout=1)
        UB = ubmatrix(mosaic.xaxis, mosaic.yaxis, B, nargout=1)
        UBinv = np.linalg.inv(UB)

        X = mosaic.rand_rot_vect(sz, rng=rng)
        n_pts = int(X.size // 3)
        neg_X_2d = np.reshape(-X, (3, n_pts), order="F")
        tmp = rotvec_to_rotmat(neg_X_2d)
        if tmp.ndim == 2:
            Rrlu = UBinv @ tmp @ UB
        else:
            Rrlu = np.einsum("ij,jkn,kl->iln", UBinv, tmp, UB)
        return np.reshape(Rrlu, (3, *X.shape), order="F")

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def saveable_fields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        if mandatory is not None and mandatory is not False:
            return ("xgeom", "ygeom", "shape", "ps")
        baseflds = super().saveable_fields()
        return (
            baseflds[0],
            baseflds[1],
            "hall_symbol",
            "single_crystal",
            "xy_geom",
            "shape",
            "ps",
            "eta",
            "temperature",
            baseflds[-1],
        )

    def saveableFields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return self.saveable_fields(mandatory=mandatory)

    def class_version(self) -> int:
        return 3

    def check_combo_arg(self) -> "IX_sample":
        if any(self._mandatory_field_set) and not all(self._mandatory_field_set):
            mandatory_names = self.saveable_fields(mandatory=True)
            unset = [
                n
                for n, is_set in zip(mandatory_names, self._mandatory_field_set)
                if not is_set
            ]
            raise ValueError(
                f"HERBERT:IX_sample:invalid_argument: If any of the mandatory properties "
                f"({mandatory_names}) is set, all mandatory properties must be set. "
                f"Properties: {unset} have not been set"
            )

        if self._xgeom.size > 0 and self._ygeom.size > 0:
            norm_x = float(np.linalg.norm(self._xgeom))
            norm_y = float(np.linalg.norm(self._ygeom))
            cross_norm = float(np.linalg.norm(np.cross(self._xgeom, self._ygeom)))
            if (
                norm_x == 0.0
                or norm_y == 0.0
                or (cross_norm / (norm_x * norm_y)) < 1e-5
            ):
                raise ValueError(
                    f"HERBERT:IX_sample:invalid_argument: "
                    f'"xgeom={self._xgeom}" and "ygeom={self._ygeom}" vectors are colinear, or almost colinear'
                )

        expected_n_ps = self._n_ps[self._shape]
        if self._ps.size != expected_n_ps:
            raise ValueError(
                f"HERBERT:IX_sample:invalid_argument: "
                f'Sample shape "{self._shape}" requires {expected_n_ps} parameters, '
                f"but {self._ps.size} parameters are provided"
            )

        if np.any(self._ps < 0.0):
            raise ValueError(
                "HERBERT:IX_sample:invalid_argument: "
                "Sample shape parameters must all be greater than or equal to zero"
            )

        self.clear_hash()
        return self

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "IX_sample"]:
        """
        Convert old structure versions (v0, v1, v2) to current v3 format.
        Ported from `@IX_sample/private/convert_old_struct_.m`.
        """
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "name_" in inputs:
            inputs.pop("class_version_", None)
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs

        if "xgeom" in inputs and "ygeom" in inputs:
            xg = np.asarray(inputs.pop("xgeom"), dtype=float).ravel()
            yg = np.asarray(inputs.pop("ygeom"), dtype=float).ravel()
            inputs["xy_geom"] = np.vstack([xg, yg])

        return inputs, self


__all__ = ["IX_sample"]
