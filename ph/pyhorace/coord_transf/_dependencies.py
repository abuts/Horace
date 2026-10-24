"""Lightweight support dependencies for coordinate transformation classes and unit tests."""

from __future__ import annotations

import copy
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.calc_proj_matrix import calc_proj_matrix
from pyhorace.coord_transf.ubmatrix import ubmatrix
from pyhorace.core.serializable import Serializable


class HoraceError(ValueError, RuntimeError):
    """Exception class for Horace and Herbert errors with MATLAB-style error IDs."""

    def __init__(self, error_id: str, message: str = "") -> None:
        self.error_id = error_id
        self.message = message or error_id
        full_msg = f"{error_id}: {message}" if message else error_id
        super().__init__(full_msg)


def rotvec_to_rotmat_rad(
    theta: Union[Sequence[float], np.ndarray], algorithm: int = 0
) -> np.ndarray:
    """Convert rotation vector(s) in radians to rotation matrix/matrices."""
    th = np.asarray(theta, dtype=float)
    if th.size == 3 and th.ndim <= 2 and (th.ndim == 1 or 1 in th.shape):
        t = th.reshape(3)
        theta0 = float(np.linalg.norm(t))
        if theta0 < 1e-12:
            return np.eye(3, dtype=float)
        nv = t / theta0
        c = float(np.cos(theta0))
        s = float(np.sin(theta0))
        a = 1.0 - c
        skew = np.array(
            [
                [0.0, nv[2], -nv[1]],
                [-nv[2], 0.0, nv[0]],
                [nv[1], -nv[0], 0.0],
            ],
            dtype=float,
        )
        return a * np.outer(nv, nv) + c * np.eye(3, dtype=float) + s * skew

    if th.ndim == 2 and th.shape[0] == 3:
        m = th.shape[1]
        out = np.zeros((3, 3, m), dtype=float)
        for i in range(m):
            out[:, :, i] = rotvec_to_rotmat_rad(th[:, i], algorithm=algorithm)
        return out

    raise HoraceError(
        "HERBERT:rotvec_to_rotmat_rad:invalid_argument",
        "theta must be a 3-vector or 3xN array",
    )


def rotmat_to_rotvec_rad(rotmat: np.ndarray, algorithm: int = 0) -> np.ndarray:
    """Convert rotation matrix/matrices to rotation vector(s) in radians."""
    rm = np.asarray(rotmat, dtype=float)
    if rm.ndim == 2 and rm.shape == (3, 3):
        sN = np.array(
            [
                rm[1, 2] - rm[2, 1],
                rm[2, 0] - rm[0, 2],
                rm[0, 1] - rm[1, 0],
            ],
            dtype=float,
        )
        S = 0.5 * (rm + rm.T)
        costh = float(np.clip(0.5 * (np.trace(S) - 1.0), -1.0, 1.0))
        theta0 = float(np.arccos(costh))
        S[0, 0] -= costh
        S[1, 1] -= costh
        S[2, 2] -= costh
        icol = int(np.argmax(np.diag(S)))
        v = S[:, icol].copy()
        vn = float(np.linalg.norm(v))
        if vn > 1e-20:
            v = v / vn
        sgn = -1.0 if sN[icol] < 0.0 else 1.0
        return theta0 * sgn * v

    if rm.ndim == 3 and rm.shape[:2] == (3, 3):
        m = rm.shape[2]
        out = np.zeros((3, m), dtype=float)
        for i in range(m):
            out[:, i] = rotmat_to_rotvec_rad(rm[:, :, i], algorithm=algorithm)
        return out

    raise HoraceError(
        "HERBERT:rotmat_to_rotvec_rad:invalid_argument",
        "rotmat must be a 3x3 or 3x3xN array",
    )


def rotvec_to_rotmat(
    theta_deg: Union[Sequence[float], np.ndarray], algorithm: int = 0
) -> np.ndarray:
    """Convert rotation vector(s) in degrees to rotation matrix/matrices."""
    return rotvec_to_rotmat_rad(np.deg2rad(theta_deg), algorithm=algorithm)


def rotmat_to_rotvec(rotmat: np.ndarray, algorithm: int = 0) -> np.ndarray:
    """Convert rotation matrix/matrices to rotation vector(s) in degrees."""
    return np.rad2deg(rotmat_to_rotvec_rad(rotmat, algorithm=algorithm))


class crystal_alignment_info(Serializable):
    """Helper class holding crystal alignment outputs to modify misaligned files."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._alatt: np.ndarray = np.array(
            [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float
        )
        self._angdeg: np.ndarray = np.array([90.0, 90.0, 90.0], dtype=float)
        self._rotvec: np.ndarray = np.zeros(3, dtype=float)
        self._distance: np.ndarray = np.zeros(3, dtype=float)
        self._hkl_mode: bool = False

        if (
            len(args) == 1
            and isinstance(args[0], crystal_alignment_info)
            and not kwargs
        ):
            other = args[0]
            self._alatt = other._alatt.copy()
            self._angdeg = other._angdeg.copy()
            self._rotvec = other._rotvec.copy()
            self._distance = other._distance.copy()
            self._hkl_mode = other._hkl_mode
            return
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            self.from_bare_struct(args[0])
            return
        if len(args) == 2 and not kwargs:
            self.alatt = args[0]
            self.angdeg = args[1]
        elif len(args) == 3 and not kwargs:
            self.alatt = args[0]
            self.angdeg = args[1]
            self.rotvec = args[2]
        elif len(args) > 0:
            pos_names = ["alatt", "angdeg", "rotvec", "distance"]
            self.set_positional_and_key_val_arguments(
                pos_names, False, *args, **kwargs
            )
            return

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def saveable_fields(self) -> list[str]:
        return ["alatt", "angdeg", "rotvec", "distance"]

    @property
    def alatt(self) -> np.ndarray:
        return self._alatt.copy()

    @alatt.setter
    def alatt(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(arr <= 0):
            raise HoraceError(
                "HORACE:crystal_alignment_info:invalid_argument",
                "alatt must be a 3-vector of positive lattice parameters",
            )
        self._alatt = arr.astype(float)

    @property
    def angdeg(self) -> np.ndarray:
        return self._angdeg.copy()

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(arr <= 0) or np.any(arr >= 180):
            raise HoraceError(
                "HORACE:crystal_alignment_info:invalid_argument",
                "angdeg must be a 3-vector of angles in (0, 180) degrees",
            )
        self._angdeg = arr.astype(float)

    @property
    def rotvec(self) -> np.ndarray:
        return self._rotvec.copy()

    @rotvec.setter
    def rotvec(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3:
            raise HoraceError(
                "HORACE:crystal_alignment_info:invalid_argument",
                "rotvec must be a 3-element numeric vector",
            )
        self._rotvec = arr.astype(float)

    @property
    def distance(self) -> np.ndarray:
        return self._distance.copy()

    @distance.setter
    def distance(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3:
            raise HoraceError(
                "HORACE:crystal_alignment_info:invalid_argument",
                "distance must be a 3-element numeric vector",
            )
        self._distance = arr.astype(float)

    @property
    def rotmat(self) -> np.ndarray:
        return rotvec_to_rotmat_rad(self._rotvec)

    @rotmat.setter
    def rotmat(self, val: Any) -> None:
        rm = np.asarray(val, dtype=float)
        if rm.shape != (3, 3):
            raise HoraceError(
                "HORACE:crystal_alignment_info:invalid_argument",
                "rotmat must be a 3x3 orthogonal matrix",
            )
        self._rotvec = rotmat_to_rotvec_rad(rm)

    @property
    def rotangle(self) -> float:
        return float(np.rad2deg(np.linalg.norm(self._rotvec)))

    @property
    def hkl_mode(self) -> bool:
        return bool(self._hkl_mode)

    @hkl_mode.setter
    def hkl_mode(self, val: Any) -> None:
        self._hkl_mode = bool(val)

    @property
    def legacy_mode(self) -> bool:
        return bool(self._hkl_mode)

    @legacy_mode.setter
    def legacy_mode(self, val: Any) -> None:
        self._hkl_mode = bool(val)

    def get_corr_mat(self, proj_or_alatt0: Any, angdeg0: Any = None) -> np.ndarray:
        """Return 3x3 crystal alignment correction matrix."""
        if hasattr(proj_or_alatt0, "bmatrix") and angdeg0 is None:
            proj = proj_or_alatt0
            b0 = proj.bmatrix(3)
            hkle_mode_ = self._hkl_mode
        else:
            b0 = bmatrix(proj_or_alatt0, angdeg0, nargout=1)
            hkle_mode_ = True
            proj = None

        if hkle_mode_:
            b = bmatrix(self._alatt, self._angdeg, nargout=1)
            return np.linalg.solve(b, self.rotmat @ b0)
        else:
            q_to_img = proj.get_pix_img_transformation(3, nargout=1)
            return np.linalg.solve(b0 @ q_to_img, self.rotmat)


class fast_map(Serializable):
    """Fast key-value map for integer/double run IDs and indices."""

    def class_version(self) -> int:
        return 2

    def __init__(
        self,
        keys: Optional[Union[Sequence[Any], np.ndarray]] = None,
        values: Optional[Union[Sequence[Any], np.ndarray]] = None,
        key_type: str = "uint32",
    ) -> None:
        super().__init__()
        self._key_type = str(key_type)
        self._keys: np.ndarray = np.array([], dtype=float)
        self._values: np.ndarray = np.array([], dtype=float)
        self._optimized: bool = False
        if keys is not None and values is not None:
            k_arr = np.asarray(keys, dtype=float).ravel()
            v_arr = np.asarray(values, dtype=float).ravel()
            if k_arr.size != v_arr.size:
                raise HoraceError(
                    "HERBERT:fast_map:invalid_argument",
                    "keys and values must have the same number of elements",
                )
            order = np.argsort(k_arr)
            self._keys = k_arr[order]
            self._values = v_arr[order]
            self._optimized = True

    def saveable_fields(self) -> list[str]:
        return ["keys", "values", "KeyType"]

    @property
    def keys(self) -> np.ndarray:
        return self._keys.copy()

    @keys.setter
    def keys(self, val: Any) -> None:
        self._keys = np.asarray(val, dtype=float).ravel()

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    @values.setter
    def values(self, val: Any) -> None:
        self._values = np.asarray(val, dtype=float).ravel()

    @property
    def KeyType(self) -> str:
        return self._key_type

    @KeyType.setter
    def KeyType(self, val: str) -> None:
        self._key_type = str(val)

    @property
    def optimized(self) -> bool:
        return self._optimized

    @optimized.setter
    def optimized(self, val: Any) -> None:
        self._optimized = bool(val)

    @property
    def n_members(self) -> int:
        return int(self._keys.size)

    @property
    def min_max_key(self) -> np.ndarray:
        if self._keys.size == 0:
            return np.array([np.nan, np.nan], dtype=float)
        return np.array([np.min(self._keys), np.max(self._keys)], dtype=float)

    def isKey(self, key: Any) -> Union[bool, np.ndarray]:
        k = np.asarray(key, dtype=float)
        res = np.isin(k, self._keys)
        if np.ndim(key) == 0:
            return bool(res)
        return res

    def add(self, key: Any, value: Any) -> "fast_map":
        k = float(key)
        v = float(value)
        idx = np.where(self._keys == k)[0]
        if idx.size > 0:
            self._values[idx[0]] = v
        else:
            self._keys = np.append(self._keys, k)
            self._values = np.append(self._values, v)
            order = np.argsort(self._keys)
            self._keys = self._keys[order]
            self._values = self._values[order]
        return self

    def get(self, key: Any) -> Any:
        return self.get_values_for_keys(key)

    def get_values_for_keys(
        self,
        keys: Any,
        no_validity_checks: bool = False,
        mode: int = 1,
    ) -> np.ndarray:
        k_arr = np.asarray(keys, dtype=float)
        orig_shape = k_arr.shape
        k_flat = k_arr.ravel()
        pos = np.searchsorted(self._keys, k_flat)
        valid = (pos < self._keys.size) & (
            self._keys[np.minimum(pos, max(self._keys.size - 1, 0))] == k_flat
        )
        if not np.all(valid):
            if not no_validity_checks:
                raise HoraceError(
                    "HERBERT:fast_map:invalid_argument",
                    "Some requested keys are not present in fast_map",
                )
            out = np.full(k_flat.shape, np.nan, dtype=float)
            out[valid] = self._values[pos[valid]]
        else:
            out = self._values[pos]
        if orig_shape == ():
            return out[0] if out.size == 1 else out
        if (
            out.dtype == float
            and np.all(np.isfinite(out))
            and np.all(out == np.floor(out))
        ):
            return out.astype(int).reshape(orig_shape)
        return out.reshape(orig_shape)


class oriented_lattice(Serializable):
    """Class holding oriented crystal lattice parameters and goniometer angles."""

    fields_to_define_: ClassVar[list[str]] = [
        "alatt",
        "angdeg",
        "psi",
        "u",
        "v",
        "omega",
        "dpsi",
        "gl",
        "gs",
    ]

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._alatt: np.ndarray = np.array(
            [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float
        )
        self._angdeg: np.ndarray = np.array([90.0, 90.0, 90.0], dtype=float)
        self._psi: float = float("nan")
        self._u: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=float)
        self._v: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=float)
        self._omega: float = 0.0
        self._dpsi: float = 0.0
        self._gl: float = 0.0
        self._gs: float = 0.0
        self._angular_units: str = "deg"
        self._defined_fields: set[str] = set()
        self._reason_for_invalid: str = "Lattice has not been initialized"

        if len(args) == 0 and not kwargs:
            return
        if len(args) == 1 and isinstance(args[0], oriented_lattice) and not kwargs:
            other = args[0]
            self._alatt = other._alatt.copy()
            self._angdeg = other._angdeg.copy()
            self._psi = other._psi
            self._u = other._u.copy()
            self._v = other._v.copy()
            self._omega = other._omega
            self._dpsi = other._dpsi
            self._gl = other._gl
            self._gs = other._gs
            self._angular_units = other._angular_units
            self._defined_fields = set(other._defined_fields)
            self._check_combo()
            return
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            self.from_bare_struct(args[0])
            self._check_combo()
            return

        pos_names = [
            "alatt",
            "angdeg",
            "psi",
            "u",
            "v",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        ]
        self.set_positional_and_key_val_arguments(pos_names, False, *args, **kwargs)
        self._check_combo()

    def saveable_fields(self) -> list[str]:
        return [
            "alatt",
            "angdeg",
            "psi",
            "u",
            "v",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        ]

    def _check_combo(self) -> None:
        if "alatt" not in self._defined_fields or "angdeg" not in self._defined_fields:
            self._reason_for_invalid = "alatt and angdeg must be defined"
            return
        if np.linalg.norm(np.cross(self._u, self._v)) < 1e-6:
            self._reason_for_invalid = "u and v vectors are collinear"
            return
        self._reason_for_invalid = ""

    @property
    def isvalid(self) -> bool:
        self._check_combo()
        return self._reason_for_invalid == ""

    @property
    def reason_for_invalid(self) -> str:
        self._check_combo()
        return self._reason_for_invalid

    @property
    def alatt(self) -> np.ndarray:
        if "alatt" not in self._defined_fields:
            return np.array([], dtype=float)
        return self._alatt.copy()

    @alatt.setter
    def alatt(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._defined_fields.discard("alatt")
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(arr <= 0):
            raise HoraceError(
                "HORACE:oriented_lattice:invalid_argument",
                "alatt must be 3 positive numbers",
            )
        self._alatt = arr
        self._defined_fields.add("alatt")
        self._check_combo()

    @property
    def angdeg(self) -> np.ndarray:
        if "angdeg" not in self._defined_fields:
            return np.array([], dtype=float)
        return self._angdeg.copy()

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._defined_fields.discard("angdeg")
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if self._angular_units == "rad":
            arr = np.rad2deg(arr)
        if arr.size != 3 or np.any(arr <= 0) or np.any(arr >= 180):
            raise HoraceError(
                "HORACE:oriented_lattice:invalid_argument",
                "angdeg must be 3 angles in (0, 180)",
            )
        self._angdeg = arr
        self._defined_fields.add("angdeg")
        self._check_combo()

    @property
    def psi(self) -> float:
        return (
            self._psi
            if self._angular_units == "deg"
            else float(np.deg2rad(self._psi))
        )

    @psi.setter
    def psi(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._psi = float("nan")
            self._defined_fields.discard("psi")
            return
        v = float(np.asarray(val, dtype=float).item())
        self._psi = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        if not np.isnan(self._psi):
            self._defined_fields.add("psi")

    @property
    def u(self) -> np.ndarray:
        return self._u.copy()

    @u.setter
    def u(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.linalg.norm(arr) < 1e-12:
            raise HoraceError(
                "HORACE:oriented_lattice:invalid_argument",
                "u must be a non-zero 3-vector",
            )
        self._u = arr
        self._defined_fields.add("u")
        self._check_combo()

    @property
    def v(self) -> np.ndarray:
        return self._v.copy()

    @v.setter
    def v(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.linalg.norm(arr) < 1e-12:
            raise HoraceError(
                "HORACE:oriented_lattice:invalid_argument",
                "v must be a non-zero 3-vector",
            )
        self._v = arr
        self._defined_fields.add("v")
        self._check_combo()

    @property
    def omega(self) -> float:
        return (
            self._omega
            if self._angular_units == "deg"
            else float(np.deg2rad(self._omega))
        )

    @omega.setter
    def omega(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._omega = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("omega")

    @property
    def dpsi(self) -> float:
        return (
            self._dpsi
            if self._angular_units == "deg"
            else float(np.deg2rad(self._dpsi))
        )

    @dpsi.setter
    def dpsi(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._dpsi = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("dpsi")

    @property
    def gl(self) -> float:
        return (
            self._gl
            if self._angular_units == "deg"
            else float(np.deg2rad(self._gl))
        )

    @gl.setter
    def gl(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._gl = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("gl")

    @property
    def gs(self) -> float:
        return (
            self._gs
            if self._angular_units == "deg"
            else float(np.deg2rad(self._gs))
        )

    @gs.setter
    def gs(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._gs = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("gs")

    @property
    def angular_units(self) -> str:
        return self._angular_units

    @angular_units.setter
    def angular_units(self, val: str) -> None:
        v = str(val).lower()
        if v.startswith("deg"):
            self._angular_units = "deg"
        elif v.startswith("rad"):
            self._angular_units = "rad"
        else:
            raise HoraceError(
                "HORACE:oriented_lattice:invalid_argument",
                "angular_units must be 'deg' or 'rad'",
            )

    def is_defined(self, field_name: str) -> bool:
        return field_name in self._defined_fields

    def bmatrix(self, nargout: Optional[int] = None) -> Any:
        if nargout is None:
            nargout = _infer_nargout(default=1)
        return bmatrix(self._alatt, self._angdeg, nargout=nargout)

    def ubmatrix(self, nargout: Optional[int] = None) -> Any:
        if nargout is None:
            nargout = _infer_nargout(default=1)
        b = bmatrix(self._alatt, self._angdeg, nargout=1)
        return ubmatrix(self._u, self._v, b, nargout=nargout)

    def calc_proj_matrix(
        self, n_matrix: int = 0, nargout: Optional[int] = None
    ) -> Any:
        if nargout is None:
            nargout = _infer_nargout(default=3)
        psi_rad = 0.0 if np.isnan(self._psi) else float(np.deg2rad(self._psi))
        return calc_proj_matrix(
            self._alatt,
            self._angdeg,
            self._u,
            self._v,
            psi_rad,
            float(np.deg2rad(self._omega)),
            float(np.deg2rad(self._dpsi)),
            float(np.deg2rad(self._gl)),
            float(np.deg2rad(self._gs)),
            n_matrix,
            nargout=nargout,
        )


class PixelDataBase:
    """Base interface for Horace pixel data containers."""

    DEFAULT_NUM_PIX_FIELDS = 9
    EMPTY_RANGE_ = np.array(
        [[np.inf, np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf, -np.inf]],
        dtype=float,
    )

    def __init__(self, data: Any = None) -> None:
        self._is_corrected: bool = False
        self._alignment_matr: np.ndarray = np.eye(3, dtype=float)
        self._raw_data: np.ndarray = np.zeros((9, 0), dtype=float)
        if data is None:
            self._data = np.zeros((9, 0), dtype=float)
        elif isinstance(data, PixelDataBase):
            self._data = data._data.copy()
            self._raw_data = data._raw_data.copy()
            self._is_corrected = data._is_corrected
            self._alignment_matr = data._alignment_matr.copy()
        elif isinstance(data, (int, np.integer)):
            n = int(data)
            self._data = np.zeros((9, n), dtype=float)
            self._data[4, :] = 1.0
            self._raw_data = self._data.copy()
        else:
            arr = np.asarray(data, dtype=float)
            if arr.size == 0:
                self._data = np.zeros((9, 0), dtype=float)
            elif arr.ndim == 2 and arr.shape[0] == 9:
                self._data = arr.copy()
            elif arr.ndim == 2 and arr.shape[1] == 9:
                self._data = arr.T.copy()
            else:
                raise HoraceError(
                    "HORACE:PixelDataBase:invalid_argument",
                    f"Pixel data array must have 9 rows, got shape {arr.shape}",
                )
            self._raw_data = self._data.copy()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 9:
            self._data = arr.copy()
            self._raw_data = arr.copy()
        else:
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument", "data must be 9xN"
            )

    @property
    def num_pixels(self) -> int:
        return int(self._data.shape[1])

    @property
    def q_coordinates(self) -> np.ndarray:
        return self._data[0:3, :]

    @q_coordinates.setter
    def q_coordinates(self, val: Any) -> None:
        self._data[0:3, :] = np.asarray(val, dtype=float)

    @property
    def dE(self) -> np.ndarray:
        return self._data[3:4, :]

    @dE.setter
    def dE(self, val: Any) -> None:
        self._data[3:4, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def coordinates(self) -> np.ndarray:
        return self._data[0:4, :]

    @coordinates.setter
    def coordinates(self, val: Any) -> None:
        self._data[0:4, :] = np.asarray(val, dtype=float)

    @property
    def run_idx(self) -> np.ndarray:
        return self._data[4:5, :]

    @run_idx.setter
    def run_idx(self, val: Any) -> None:
        self._data[4:5, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def detector_idx(self) -> np.ndarray:
        return self._data[5:6, :]

    @detector_idx.setter
    def detector_idx(self, val: Any) -> None:
        self._data[5:6, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def energy_idx(self) -> np.ndarray:
        return self._data[6:7, :]

    @energy_idx.setter
    def energy_idx(self, val: Any) -> None:
        self._data[6:7, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def signal(self) -> np.ndarray:
        return self._data[7:8, :]

    @signal.setter
    def signal(self, val: Any) -> None:
        self._data[7:8, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def variance(self) -> np.ndarray:
        return self._data[8:9, :]

    @variance.setter
    def variance(self, val: Any) -> None:
        self._data[8:9, :] = np.asarray(val, dtype=float).reshape(1, -1)

    @property
    def pix_range(self) -> np.ndarray:
        if self.num_pixels == 0:
            return np.full((2, 4), np.nan, dtype=float)
        return np.vstack(
            [np.min(self._data[0:4, :], axis=1), np.max(self._data[0:4, :], axis=1)]
        )

    @property
    def is_corrected(self) -> bool:
        return self._is_corrected

    @is_corrected.setter
    def is_corrected(self, val: bool) -> None:
        self._is_corrected = bool(val)

    @property
    def alignment_matr(self) -> np.ndarray:
        return self._alignment_matr.copy()

    @alignment_matr.setter
    def alignment_matr(self, val: Any) -> None:
        rm = np.asarray(val, dtype=float)
        if rm.shape != (3, 3):
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument",
                "alignment_matr must be 3x3",
            )
        self._alignment_matr = rm.copy()
        self._is_corrected = not np.allclose(rm, np.eye(3))
        if self._is_corrected:
            self._data[0:3, :] = rm @ self._raw_data[0:3, :]
        else:
            self._data[0:3, :] = self._raw_data[0:3, :].copy()

    def get_raw_data(self, field_name: str = "q_coordinates") -> np.ndarray:
        if field_name == "q_coordinates":
            return self._raw_data[0:3, :].copy()
        if field_name == "coordinates":
            return self._raw_data[0:4, :].copy()
        return getattr(self, field_name).copy()


class PixelDataMemory(PixelDataBase):
    """In-memory PixelData container."""

    pass


def __getattr__(name: str) -> Any:
    if name in ("DnDBase", "SQWDnDBase", "d0d", "d1d", "d2d", "d3d", "d4d", "dnd", "dnd_metadata", "dnd_data", "data_sqw_dnd"):
        import importlib

        _sqw = importlib.import_module("pyhorace.sqw")
        return getattr(_sqw, name)
    if name == "AxesBlockBase":
        from pyhorace.coord_transf.axes_block_base import AxesBlockBase

        return AxesBlockBase
    if name == "line_axes":
        from pyhorace.coord_transf.line_axes import line_axes

        return line_axes
    if name == "sphere_axes":
        from pyhorace.coord_transf.sphere_axes import sphere_axes

        return sphere_axes
    if name == "cylinder_axes":
        from pyhorace.coord_transf.cylinder_axes import cylinder_axes

        return cylinder_axes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
