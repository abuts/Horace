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


def rotvec_to_rotmat_rad(theta: Union[Sequence[float], np.ndarray], algorithm: int = 0) -> np.ndarray:
    """Convert rotation vector(s) in radians to rotation matrix/matrices.

    Parameters
    ----------
    theta : array_like
        3-element vector or (3, m) array of rotation vectors in radians.
        Gives the rotation matrix of a coordinate frame (passive rotation).
    algorithm : int, optional
        0 for explicit Rodrigues formula (fast), 1 for matrix exponential.

    Returns
    -------
    rotmat : np.ndarray
        (3, 3) or (3, 3, m) array of rotation matrices.
    """
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

    raise HoraceError("HERBERT:rotvec_to_rotmat_rad:invalid_argument", "theta must be a 3-vector or 3xN array")


def rotmat_to_rotvec_rad(rotmat: np.ndarray, algorithm: int = 0) -> np.ndarray:
    """Convert rotation matrix/matrices to rotation vector(s) in radians.

    Parameters
    ----------
    rotmat : np.ndarray
        (3, 3) or (3, 3, m) rotation matrix/matrices relating two coordinate frames.
    algorithm : int, optional
        Algorithm selector (unused, kept for signature compatibility).

    Returns
    -------
    theta : np.ndarray
        (3,) or (3, m) rotation vector(s) in radians.
    """
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

    raise HoraceError("HERBERT:rotmat_to_rotvec_rad:invalid_argument", "rotmat must be a 3x3 or 3x3xN array")


def rotvec_to_rotmat(theta_deg: Union[Sequence[float], np.ndarray], algorithm: int = 0) -> np.ndarray:
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
        self._alatt: np.ndarray = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
        self._angdeg: np.ndarray = np.array([90.0, 90.0, 90.0], dtype=float)
        self._rotvec: np.ndarray = np.zeros(3, dtype=float)
        self._distance: np.ndarray = np.zeros(3, dtype=float)
        self._hkl_mode: bool = False

        if len(args) == 1 and isinstance(args[0], crystal_alignment_info) and not kwargs:
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
            self.set_positional_and_key_val_arguments(pos_names, False, *args, **kwargs)
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
        valid = (pos < self._keys.size) & (self._keys[np.minimum(pos, max(self._keys.size - 1, 0))] == k_flat)
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
        # In MATLAB, values returned for row/col keys match integer/float vector
        if out.dtype == float and np.all(np.isfinite(out)) and np.all(out == np.floor(out)):
            return out.astype(int).reshape(orig_shape)
        return out.reshape(orig_shape)


class oriented_lattice(Serializable):
    """Class holding oriented crystal lattice parameters and goniometer angles."""

    fields_to_define_: ClassVar[list[str]] = ["alatt", "angdeg", "psi", "u", "v", "omega", "dpsi", "gl", "gs"]

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._alatt: np.ndarray = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
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

        pos_names = ["alatt", "angdeg", "psi", "u", "v", "omega", "dpsi", "gl", "gs", "angular_units"]
        self.set_positional_and_key_val_arguments(pos_names, False, *args, **kwargs)
        self._check_combo()

    def saveable_fields(self) -> list[str]:
        return ["alatt", "angdeg", "psi", "u", "v", "omega", "dpsi", "gl", "gs", "angular_units"]

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
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._defined_fields.discard("alatt")
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(arr <= 0):
            raise HoraceError("HORACE:oriented_lattice:invalid_argument", "alatt must be 3 positive numbers")
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
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._defined_fields.discard("angdeg")
            return
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if self._angular_units == "rad":
            arr = np.rad2deg(arr)
        if arr.size != 3 or np.any(arr <= 0) or np.any(arr >= 180):
            raise HoraceError("HORACE:oriented_lattice:invalid_argument", "angdeg must be 3 angles in (0, 180)")
        self._angdeg = arr
        self._defined_fields.add("angdeg")
        self._check_combo()

    @property
    def psi(self) -> float:
        return self._psi if self._angular_units == "deg" else float(np.deg2rad(self._psi))

    @psi.setter
    def psi(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
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
            raise HoraceError("HORACE:oriented_lattice:invalid_argument", "u must be a non-zero 3-vector")
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
            raise HoraceError("HORACE:oriented_lattice:invalid_argument", "v must be a non-zero 3-vector")
        self._v = arr
        self._defined_fields.add("v")
        self._check_combo()

    @property
    def omega(self) -> float:
        return self._omega if self._angular_units == "deg" else float(np.deg2rad(self._omega))

    @omega.setter
    def omega(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._omega = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("omega")

    @property
    def dpsi(self) -> float:
        return self._dpsi if self._angular_units == "deg" else float(np.deg2rad(self._dpsi))

    @dpsi.setter
    def dpsi(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._dpsi = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("dpsi")

    @property
    def gl(self) -> float:
        return self._gl if self._angular_units == "deg" else float(np.deg2rad(self._gl))

    @gl.setter
    def gl(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        self._gl = float(np.rad2deg(v)) if self._angular_units == "rad" else v
        self._defined_fields.add("gl")

    @property
    def gs(self) -> float:
        return self._gs if self._angular_units == "deg" else float(np.deg2rad(self._gs))

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
            raise HoraceError("HORACE:oriented_lattice:invalid_argument", "angular_units must be 'deg' or 'rad'")

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

    def calc_proj_matrix(self, n_matrix: int = 0, nargout: Optional[int] = None) -> Any:
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
            raise HoraceError("HORACE:PixelDataBase:invalid_argument", "data must be 9xN")

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
        return np.vstack([np.min(self._data[0:4, :], axis=1), np.max(self._data[0:4, :], axis=1)])

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
            raise HoraceError("HORACE:PixelDataBase:invalid_argument", "alignment_matr must be 3x3")
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


class AxesBlockBase(Serializable):
    """Base class for axes block definitions (line_axes, sphere_axes, cylinder_axes)."""

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._img_range: np.ndarray = np.zeros((2, 4), dtype=float)
        self._nbins_all_dims: np.ndarray = np.ones(4, dtype=int)
        self._single_bin_defines_iax: np.ndarray = np.ones(4, dtype=bool)
        self._dax: Optional[np.ndarray] = None
        self._offset: np.ndarray = np.zeros(4, dtype=float)
        self._label: list[str] = ["\\zeta", "\\xi", "\\eta", "E"]
        self._title: str = ""
        self._filename: str = ""
        self._filepath: str = ""
        self._img_scales: np.ndarray = np.ones(4, dtype=float)
        self._type: str = "ppr"
        self._hkle_axes_directions: Optional[np.ndarray] = None
        self._max_img_range: Optional[np.ndarray] = None

        if len(args) == 0 and not kwargs:
            return
        self._init_axes_args(*args, **kwargs)

    def _init_axes_args(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 0 and not kwargs:
            return
        if len(args) == 1 and isinstance(args[0], AxesBlockBase) and not kwargs:
            other = args[0]
            self._img_range = other._img_range.copy()
            self._nbins_all_dims = other._nbins_all_dims.copy()
            self._single_bin_defines_iax = other._single_bin_defines_iax.copy()
            self._dax = None if other._dax is None else other._dax.copy()
            self._offset = other._offset.copy()
            self._label = list(other._label)
            self._title = other._title
            self._filename = other._filename
            self._filepath = other._filepath
            self._img_scales = other._img_scales.copy()
            self._type = other._type
            self._hkle_axes_directions = (
                None if other._hkle_axes_directions is None else other._hkle_axes_directions.copy()
            )
            return
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            self.from_bare_struct(args[0])
            return

        # Check if initialized with 4 bin descriptors: e.g. line_axes([-1, 0.1, 1], [-2, 2], ...)
        if (
            len(args) == 4
            and all(isinstance(a, (list, tuple, np.ndarray, int, float)) for a in args)
            and not (
                isinstance(args[0], np.ndarray) and args[0].shape == (2, 4)
            )
            and not (
                isinstance(args[0], (list, tuple))
                and len(args[0]) == 2
                and isinstance(args[0][0], (list, tuple, np.ndarray))
                and len(args[0][0]) == 4
            )
        ):
            self._init_from_pbin(args, **kwargs)
            return

        if len(args) == 2 and not kwargs:
            r0 = np.asarray(args[0], dtype=float)
            n1 = np.asarray(args[1])
            if r0.shape == (2, 4) and n1.size == 4:
                self.img_range = r0
                self.nbins_all_dims = n1
                return

        pos_names = [
            "img_range",
            "nbins_all_dims",
            "single_bin_defines_iax",
            "dax",
            "offset",
            "label",
            "title",
            "filename",
            "filepath",
        ]
        self.set_positional_and_key_val_arguments(pos_names, False, *args, **kwargs)

    def _init_from_pbin(self, pbin_list: Sequence[Any], **kwargs: Any) -> None:
        img_range = np.zeros((2, 4), dtype=float)
        nbins = np.ones(4, dtype=int)
        sbd = np.ones(4, dtype=bool)
        for i in range(4):
            pb = np.asarray(pbin_list[i], dtype=float).ravel()
            if pb.size == 0:
                img_range[:, i] = [0.0, 0.0]
                nbins[i] = 1
                sbd[i] = True
            elif pb.size == 1:
                step = pb[0]
                img_range[:, i] = [0.0, step]
                nbins[i] = 1
                sbd[i] = False
            elif pb.size == 2:
                img_range[:, i] = [pb[0], pb[1]]
                nbins[i] = 1
                sbd[i] = True
            elif pb.size == 3:
                lo, step, hi = pb[0], pb[1], pb[2]
                if step == 0 or hi < lo:
                    img_range[:, i] = [lo, hi]
                    nbins[i] = 1
                    sbd[i] = True
                else:
                    sbd[i] = False
                    n_steps = int(np.floor((hi - lo + 1e-12 * abs(step)) / step)) + 1
                    if n_steps < 1:
                        n_steps = 1
                    actual_hi = lo + (n_steps - 1) * step
                    img_range[:, i] = [lo - 0.5 * step, actual_hi + 0.5 * step]
                    nbins[i] = n_steps
            else:
                img_range[:, i] = [pb[0], pb[-1]]
                nbins[i] = max(1, pb.size - 1)
                sbd[i] = False
        self._img_range = img_range
        self._nbins_all_dims = nbins
        self._single_bin_defines_iax = sbd
        for k, v in kwargs.items():
            setattr(self, k, v)

    def saveable_fields(self) -> list[str]:
        return [
            "img_range",
            "nbins_all_dims",
            "single_bin_defines_iax",
            "dax",
            "offset",
            "label",
            "title",
            "filename",
            "filepath",
            "img_scales",
            "type",
        ]

    @property
    def img_range(self) -> np.ndarray:
        return self._img_range.copy()

    @img_range.setter
    def img_range(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.shape != (2, 4):
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "img_range must be a 2x4 array")
        if self._max_img_range is not None:
            for i in range(4):
                lo_lim, hi_lim = self._max_img_range[0, i], self._max_img_range[1, i]
                if not np.isinf(lo_lim) and arr[0, i] < lo_lim - 1e-9:
                    raise HoraceError(
                        "HORACE:AxesBlockBase:invalid_argument",
                        f"img_range lower bound {arr[0, i]} is below minimum {lo_lim} for axis {i + 1}",
                    )
                if not np.isinf(hi_lim) and arr[1, i] > hi_lim + 1e-9:
                    raise HoraceError(
                        "HORACE:AxesBlockBase:invalid_argument",
                        f"img_range upper bound {arr[1, i]} exceeds maximum {hi_lim} for axis {i + 1}",
                    )
        if np.any(arr[0, :] > arr[1, :] + 1e-12):
            raise HoraceError(
                "HORACE:AxesBlockBase:invalid_argument",
                "img_range lower bounds must not exceed upper bounds",
            )
        self._img_range = arr.copy()

    @property
    def nbins_all_dims(self) -> np.ndarray:
        return self._nbins_all_dims.copy()

    @nbins_all_dims.setter
    def nbins_all_dims(self, val: Any) -> None:
        arr = np.asarray(val, dtype=int).ravel()
        if arr.size != 4 or np.any(arr < 1):
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "nbins_all_dims must be 4 positive integers")
        self._nbins_all_dims = arr.copy()
        for i in range(4):
            if arr[i] > 1:
                self._single_bin_defines_iax[i] = False

    @property
    def single_bin_defines_iax(self) -> np.ndarray:
        return self._single_bin_defines_iax.copy()

    @single_bin_defines_iax.setter
    def single_bin_defines_iax(self, val: Any) -> None:
        arr = np.asarray(val, dtype=bool).ravel()
        if arr.size != 4:
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "single_bin_defines_iax must have 4 elements")
        self._single_bin_defines_iax = arr.copy()

    @property
    def iax(self) -> np.ndarray:
        mask = (self._nbins_all_dims == 1) & self._single_bin_defines_iax
        return np.where(mask)[0] + 1

    @property
    def pax(self) -> np.ndarray:
        mask = ~((self._nbins_all_dims == 1) & self._single_bin_defines_iax)
        return np.where(mask)[0] + 1

    @property
    def iint(self) -> np.ndarray:
        idx = self.iax - 1
        if idx.size == 0:
            return np.zeros((2, 0), dtype=float)
        return self._img_range[:, idx].copy()

    @property
    def p(self) -> list[np.ndarray]:
        out = []
        for idx in self.pax - 1:
            n = int(self._nbins_all_dims[idx])
            out.append(np.linspace(self._img_range[0, idx], self._img_range[1, idx], n + 1))
        return out

    @property
    def pbin(self) -> list[np.ndarray]:
        out = []
        for i in range(4):
            n = int(self._nbins_all_dims[i])
            lo, hi = self._img_range[0, i], self._img_range[1, i]
            if n == 1 and self._single_bin_defines_iax[i]:
                out.append(np.array([lo, hi], dtype=float))
            else:
                step = (hi - lo) / n
                fl = lo + 0.5 * step
                fh = hi - 0.5 * step
                out.append(np.array([fl, step, fh], dtype=float))
        return out

    @property
    def step(self) -> np.ndarray:
        return (self._img_range[1, :] - self._img_range[0, :]) / self._nbins_all_dims

    @property
    def dimensions(self) -> int:
        return int(self.pax.size)

    @property
    def data_nbins(self) -> np.ndarray:
        idx = self.pax - 1
        if idx.size == 0:
            return np.array([], dtype=int)
        return self._nbins_all_dims[idx].copy()

    @property
    def dims_as_ssize(self) -> Tuple[int, ...]:
        dn = self.data_nbins
        if dn.size == 0:
            return (1, 1)
        if dn.size == 1:
            return (int(dn[0]), 1)
        return tuple(int(x) for x in dn)

    @property
    def dax(self) -> np.ndarray:
        if self._dax is None or self._dax.size != self.pax.size:
            return np.arange(1, self.pax.size + 1, dtype=int)
        return self._dax.copy()

    @dax.setter
    def dax(self, val: Any) -> None:
        if val is None:
            self._dax = None
            return
        self._dax = np.asarray(val, dtype=int).ravel()

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.append(arr, 0.0)
        if arr.size != 4:
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "offset must have 3 or 4 elements")
        self._offset = arr.copy()

    @property
    def label(self) -> list[str]:
        return list(self._label)

    @label.setter
    def label(self, val: Any) -> None:
        if isinstance(val, (list, tuple)) and len(val) == 4:
            self._label = [str(x) for x in val]
        else:
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "label must be a 4-element sequence of strings")

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, val: Any) -> None:
        self._title = "" if val is None else str(val)

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, val: Any) -> None:
        self._filename = "" if val is None else str(val)

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, val: Any) -> None:
        self._filepath = "" if val is None else str(val)

    @property
    def img_scales(self) -> np.ndarray:
        return self._img_scales.copy()

    @img_scales.setter
    def img_scales(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.append(arr, 1.0)
        if arr.size != 4:
            raise HoraceError("HORACE:AxesBlockBase:invalid_argument", "img_scales must have 3 or 4 elements")
        self._img_scales = arr.copy()

    @property
    def ulen(self) -> np.ndarray:
        return self.img_scales

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self.img_scales = val

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, val: str) -> None:
        s = str(val)
        if len(s) == 3:
            s = s + "p"
        self._type = s

    @property
    def axes_units(self) -> str:
        return self.type

    @axes_units.setter
    def axes_units(self, val: str) -> None:
        self.type = val

    @property
    def hkle_axes_directions(self) -> Optional[np.ndarray]:
        return None if self._hkle_axes_directions is None else self._hkle_axes_directions.copy()

    @hkle_axes_directions.setter
    def hkle_axes_directions(self, val: Any) -> None:
        if val is None:
            self._hkle_axes_directions = None
        else:
            self._hkle_axes_directions = np.asarray(val, dtype=float).copy()

    @staticmethod
    def bins_in_1Drange(bin_edges: np.ndarray, range_1d: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Find 1-based inclusive bin index range overlapping with range_1d."""
        edges = np.asarray(bin_edges, dtype=float).ravel()
        rng = np.asarray(range_1d, dtype=float).ravel()
        r_min, r_max = float(rng[0]), float(rng[1])
        tol = 1e-12
        if r_max < edges[0] + tol or r_min > edges[-1] - tol:
            return False, np.array([], dtype=int)
        # Bin i (0-based, i in 0..N-1) spans [edges[i], edges[i+1]]
        # Overlaps [r_min, r_max] if edges[i+1] > r_min + tol and edges[i] < r_max - tol
        n_bins = edges.size - 1
        lower_edges = edges[:-1]
        upper_edges = edges[1:]
        overlap = (upper_edges > r_min + tol) & (lower_edges < r_max - tol)
        if not np.any(overlap):
            # Fallback if r_min == r_max inside an edge
            overlap = (upper_edges >= r_min - tol) & (lower_edges <= r_max + tol)
        idx = np.where(overlap)[0]
        if idx.size == 0:
            return False, np.array([], dtype=int)
        return True, np.array([int(idx[0]) + 1, int(idx[-1]) + 1], dtype=int)

    def get_bin_nodes(self, *args: Any, nargout: Optional[int] = None) -> Any:
        """Compute grid of bin boundaries or bin centers."""
        if nargout is None:
            nargout = _infer_nargout(default=1)
        is_3d = False
        bin_centre = False
        halo = False
        axes_only = False
        custom_nbins = None

        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-3d":
                    is_3d = True
                elif al == "-bin_centre":
                    bin_centre = True
                elif al == "-halo":
                    halo = True
                elif al == "-axes_only":
                    axes_only = True
                elif al == "-plot_edges":
                    pass
            elif isinstance(a, (list, tuple, np.ndarray)):
                custom_nbins = np.asarray(a, dtype=int).ravel()

        nbins = self._nbins_all_dims.copy()
        if custom_nbins is not None:
            if custom_nbins.size == 3:
                nbins[:3] = custom_nbins
            elif custom_nbins.size == 4:
                nbins[:] = custom_nbins

        axes_1d = []
        ndim = 3 if (is_3d or nargout >= 2) else 4
        for i in range(4):
            lo, hi = self._img_range[0, i], self._img_range[1, i]
            n = int(nbins[i])
            edges = np.linspace(lo, hi, n + 1)
            if bin_centre:
                pts = 0.5 * (edges[:-1] + edges[1:])
                if halo and i < 3:
                    step = (hi - lo) / n
                    if n > 1:
                        pts = np.concatenate([[pts[0] - step], pts, [pts[-1] + step]])
            else:
                pts = edges
                if halo and i < 3 and n > 1:
                    step = (hi - lo) / n
                    pts = np.concatenate([[pts[0] - step], pts, [pts[-1] + step]])
            axes_1d.append(pts)

        if axes_only:
            if nargout >= 2:
                return axes_1d[:3], axes_1d[3]
            return axes_1d[:ndim]

        if ndim == 3:
            g0, g1, g2 = np.meshgrid(axes_1d[0], axes_1d[1], axes_1d[2], indexing="ij")
            nodes = np.vstack([g0.ravel(order="F"), g1.ravel(order="F"), g2.ravel(order="F")])
            if nargout >= 2:
                return nodes, axes_1d[3]
            return nodes
        else:
            g0, g1, g2, g3 = np.meshgrid(axes_1d[0], axes_1d[1], axes_1d[2], axes_1d[3], indexing="ij")
            nodes = np.vstack(
                [g0.ravel(order="F"), g1.ravel(order="F"), g2.ravel(order="F"), g3.ravel(order="F")]
            )
            return nodes

    def get_char_size(self, this_proj: Any) -> np.ndarray:
        """Compute characteristic cell size of this axes block in hkl coordinates."""
        steps = self.step[:3]
        corners_local = np.array(
            [
                [0.0, 0.0, 0.0],
                [steps[0], 0.0, 0.0],
                [0.0, steps[1], 0.0],
                [0.0, 0.0, steps[2]],
                [steps[0], steps[1], steps[2]],
            ],
            dtype=float,
        ).T
        base_pt = self._img_range[0, :3].reshape(3, 1)
        pts_img = base_pt + corners_local
        pts_hkl = this_proj.transform_img_to_hkl(pts_img)
        min_hkl = np.min(pts_hkl, axis=1)
        max_hkl = np.max(pts_hkl, axis=1)
        return np.maximum(max_hkl - min_hkl, 1e-12)

    def get_interp_nodes(self, targ_proj: Any, char_sizes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate interpolation grid in target image coordinates so step <= char_sizes."""
        corners, dE_edges = self.get_bin_nodes("-3D", nargout=2)
        corners_hkl = targ_proj.transform_img_to_hkl(corners)
        span_hkl = np.max(corners_hkl, axis=1) - np.min(corners_hkl, axis=1)
        n_sub = np.maximum(1, np.ceil(span_hkl / np.maximum(char_sizes, 1e-6)).astype(int))
        n_nodes = np.maximum(self._nbins_all_dims[:3], n_sub)
        n_nodes = np.minimum(n_nodes, 100)  # cap to keep fast
        axes_1d = [
            np.linspace(self._img_range[0, i], self._img_range[1, i], int(n_nodes[i]) + 1)
            for i in range(3)
        ]
        g0, g1, g2 = np.meshgrid(axes_1d[0], axes_1d[1], axes_1d[2], indexing="ij")
        nodes = np.vstack([g0.ravel(order="F"), g1.ravel(order="F"), g2.ravel(order="F")])
        return nodes, dE_edges

    def bin_pixels(
        self,
        coord_transf: np.ndarray,
        *args: Any,
        nargout: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Bin transformed pixel coordinates into the axes block grid."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        coords = np.asarray(coord_transf, dtype=float)
        ndim = coords.shape[0]
        n_pts = coords.shape[1]

        npix_accum = None
        s_accum = None
        e_accum = None
        pix_cand = None
        unique_runid = None

        force_double = False
        nomex = False
        test_input = False

        pos_args = []
        for a in args:
            if isinstance(a, str):
                al = a.lower()
                if al == "-force_double":
                    force_double = True
                elif al == "-nomex":
                    nomex = True
                elif al == "-test_input":
                    test_input = True
            else:
                pos_args.append(a)

        if len(pos_args) >= 1 and pos_args[0] is not None and np.asarray(pos_args[0]).size > 0:
            npix_accum = np.asarray(pos_args[0], dtype=float).copy()
        if len(pos_args) >= 2 and pos_args[1] is not None and np.asarray(pos_args[1]).size > 0:
            s_accum = np.asarray(pos_args[1], dtype=float).copy()
        if len(pos_args) >= 3 and pos_args[2] is not None and np.asarray(pos_args[2]).size > 0:
            e_accum = np.asarray(pos_args[2], dtype=float).copy()
        if len(pos_args) >= 4:
            pix_cand = pos_args[3]
        if len(pos_args) >= 5:
            unique_runid = pos_args[4]

        grid_shape = tuple(int(x) for x in self._nbins_all_dims[:ndim])
        if npix_accum is None:
            npix_accum = np.zeros(grid_shape, dtype=float)
        if s_accum is None and len(pos_args) >= 2:
            s_accum = np.zeros(grid_shape, dtype=float)
        if e_accum is None and len(pos_args) >= 3:
            e_accum = np.zeros(grid_shape, dtype=float)

        valid = np.ones(n_pts, dtype=bool)
        bin_indices = np.zeros((ndim, n_pts), dtype=int)
        tol = 1e-12

        for d in range(ndim):
            lo = self._img_range[0, d]
            hi = self._img_range[1, d]
            nb = int(self._nbins_all_dims[d])
            c_d = coords[d, :]
            in_d = (c_d >= lo - tol) & (c_d <= hi + tol)
            valid &= in_d
            if nb == 1:
                bin_indices[d, :] = 0
            else:
                step = (hi - lo) / nb
                idx_d = np.floor((c_d - lo) / step).astype(int)
                # Points right on upper boundary hi belong to the last bin (nb - 1)
                idx_d = np.where((c_d >= hi - tol) & (c_d <= hi + tol), nb - 1, idx_d)
                idx_d = np.where((c_d >= lo - tol) & (c_d < lo), 0, idx_d)
                in_idx = (idx_d >= 0) & (idx_d < nb)
                valid &= in_idx
                bin_indices[d, :] = np.clip(idx_d, 0, nb - 1)

        valid_pos = np.where(valid)[0]
        if valid_pos.size > 0:
            # Compute Fortran-order flat indices into grid_shape
            strides = [1]
            for s_dim in grid_shape[:-1]:
                strides.append(strides[-1] * s_dim)
            flat_idx = np.zeros(valid_pos.size, dtype=int)
            for d in range(ndim):
                flat_idx += bin_indices[d, valid_pos] * strides[d]

            npix_flat = npix_accum.ravel(order="F")
            counts = np.bincount(flat_idx, minlength=npix_flat.size)
            npix_flat += counts
            npix_accum = npix_flat.reshape(grid_shape, order="F")

            if s_accum is not None and pix_cand is not None:
                if isinstance(pix_cand, PixelDataBase):
                    sig = pix_cand.signal.ravel()[valid_pos]
                    var = pix_cand.variance.ravel()[valid_pos]
                else:
                    arr_p = np.asarray(pix_cand, dtype=float)
                    sig = arr_p[7, valid_pos] if arr_p.shape[0] >= 9 else np.zeros(valid_pos.size)
                    var = arr_p[8, valid_pos] if arr_p.shape[0] >= 9 else np.zeros(valid_pos.size)
                s_flat = s_accum.ravel(order="F")
                e_flat = e_accum.ravel(order="F")
                s_flat += np.bincount(flat_idx, weights=sig, minlength=s_flat.size)
                e_flat += np.bincount(flat_idx, weights=var, minlength=e_flat.size)
                s_accum = s_flat.reshape(grid_shape, order="F")
                e_accum = e_flat.reshape(grid_shape, order="F")

            if unique_runid is not None and pix_cand is not None:
                if isinstance(pix_cand, PixelDataBase):
                    r_ids = pix_cand.run_idx.ravel()[valid_pos]
                else:
                    r_ids = np.asarray(pix_cand)[4, valid_pos]
                unique_runid = np.unique(np.concatenate([np.asarray(unique_runid).ravel(), r_ids]))

            if pix_cand is not None:
                order = np.argsort(flat_idx, kind="stable")
                sorted_pos = valid_pos[order]
                if isinstance(pix_cand, PixelDataBase):
                    if test_input:
                        pix_out_data = pix_cand.data[:, sorted_pos].copy()
                        pix_out_data[0:ndim, :] = coords[:, sorted_pos]
                        pix_ok = PixelDataMemory(pix_out_data)
                    else:
                        pix_ok = PixelDataMemory(pix_cand.data[:, sorted_pos].copy())
                else:
                    pix_ok = np.asarray(pix_cand)[:, sorted_pos]
                indx = flat_idx[order] + 1
                ok = valid
            else:
                pix_ok = np.array([])
                indx = flat_idx + 1
                ok = valid
        else:
            pix_ok = PixelDataMemory() if isinstance(pix_cand, PixelDataBase) else np.array([])
            indx = np.array([], dtype=int)
            ok = valid

        # Squeeze singleton integration dimensions to match dims_as_ssize when appropriate
        out_shape = self.dims_as_ssize
        if npix_accum.size == int(np.prod(out_shape)):
            npix_out = npix_accum.reshape(out_shape, order="F")
            s_out = None if s_accum is None else s_accum.reshape(out_shape, order="F")
            e_out = None if e_accum is None else e_accum.reshape(out_shape, order="F")
        else:
            npix_out = npix_accum
            s_out = s_accum
            e_out = e_accum

        outputs = (npix_out, s_out, e_out, pix_ok, unique_runid, indx, ok)
        if nargout == 1:
            return npix_out
        return outputs[:nargout]

    @staticmethod
    def get_from_old_data(data_struct: dict[str, Any]) -> "line_axes":
        """Construct line_axes from legacy Horace data structure."""
        p = data_struct.get("p", [])
        pax = np.asarray(data_struct.get("pax", []), dtype=int).ravel()
        iax = np.asarray(data_struct.get("iax", []), dtype=int).ravel()
        iint = np.asarray(data_struct.get("iint", np.zeros((2, 0))), dtype=float)

        img_range = np.zeros((2, 4), dtype=float)
        nbins = np.ones(4, dtype=int)
        sbd = np.ones(4, dtype=bool)

        for idx_i, ax_num in enumerate(iax):
            ax0 = int(ax_num) - 1
            if iint.ndim == 2 and idx_i < iint.shape[1]:
                img_range[:, ax0] = iint[:, idx_i]
            nbins[ax0] = 1
            sbd[ax0] = True

        for idx_p, ax_num in enumerate(pax):
            ax0 = int(ax_num) - 1
            edges = np.asarray(p[idx_p], dtype=float).ravel()
            img_range[:, ax0] = [edges[0], edges[-1]]
            nbins[ax0] = max(1, edges.size - 1)
            sbd[ax0] = False

        ax = line_axes(img_range=img_range, nbins_all_dims=nbins, single_bin_defines_iax=sbd)
        if "dax" in data_struct and data_struct["dax"] is not None:
            ax.dax = data_struct["dax"]
        if "uoffset" in data_struct and data_struct["uoffset"] is not None:
            ax.offset = data_struct["uoffset"]
        if "ulabel" in data_struct and data_struct["ulabel"] is not None:
            ax.label = list(data_struct["ulabel"])
        if "ulen" in data_struct and data_struct["ulen"] is not None:
            ax.img_scales = data_struct["ulen"]
        if "title" in data_struct and data_struct["title"] is not None:
            ax.title = data_struct["title"]
        if "filename" in data_struct and data_struct["filename"] is not None:
            ax.filename = data_struct["filename"]
        if "filepath" in data_struct and data_struct["filepath"] is not None:
            ax.filepath = data_struct["filepath"]
        return ax


class line_axes(AxesBlockBase):
    """Axes block for rectilinear (line_proj / ubmat_proj) coordinate systems."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class sphere_axes(AxesBlockBase):
    """Axes block for spherical (sphere_proj / kf_sphere_proj) coordinate systems."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._label = ["|Q|", "\\theta", "\\phi", "En"]
        self._type = "pddp"
        self._max_img_range = np.array(
            [
                [0.0, 0.0, -180.0, -np.inf],
                [np.inf, 180.0, 180.0, np.inf],
            ],
            dtype=float,
        )
        self._img_range = np.array(
            [
                [0.0, 0.0, -180.0, 0.0],
                [1.0, 180.0, 180.0, 0.0],
            ],
            dtype=float,
        )
        if args or kwargs:
            self._init_axes_args(*args, **kwargs)
            self._validate_spherical_ranges()

    def _validate_spherical_ranges(self) -> None:
        is_deg = len(self._type) >= 3 and self._type[1].lower() == "d"
        max_th = 180.0 if is_deg else np.pi
        max_ph = 180.0 if is_deg else np.pi
        self._max_img_range = np.array(
            [
                [0.0, 0.0, -max_ph, -np.inf],
                [np.inf, max_th, max_ph, np.inf],
            ],
            dtype=float,
        )
        # Re-validate img_range against updated limits
        self.img_range = self._img_range


class cylinder_axes(AxesBlockBase):
    """Axes block for cylindrical (cylinder_proj) coordinate systems."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._label = ["Q_{\\perp}", "Q_{||}", "\\phi", "En"]
        self._type = "ppdp"
        self._max_img_range = np.array(
            [
                [0.0, -np.inf, -180.0, -np.inf],
                [np.inf, np.inf, 180.0, np.inf],
            ],
            dtype=float,
        )
        self._img_range = np.array(
            [
                [0.0, 0.0, -180.0, 0.0],
                [1.0, 0.0, 180.0, 0.0],
            ],
            dtype=float,
        )
        if args or kwargs:
            self._init_axes_args(*args, **kwargs)
            self._validate_cylindrical_ranges()

    def _validate_cylindrical_ranges(self) -> None:
        is_deg = len(self._type) >= 3 and self._type[2].lower() == "d"
        max_ph = 180.0 if is_deg else np.pi
        self._max_img_range = np.array(
            [
                [0.0, -np.inf, -max_ph, -np.inf],
                [np.inf, np.inf, max_ph, np.inf],
            ],
            dtype=float,
        )
        self.img_range = self._img_range
