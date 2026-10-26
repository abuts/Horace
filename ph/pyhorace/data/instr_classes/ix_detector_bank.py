"""Detector bank class (IX_detector_bank)."""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from pyhorace.coord_transf import rotmat_to_rotvec, rotvec_to_rotmat
from pyhorace.core.hashable import HashableObj
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_abstract_type import (
    IX_det_abstractType,
    expand_args_by_ref,
    is_integer_id,
    matlab_size,
    size_array_split,
)
from pyhorace.data.instr_classes.ix_det_he3tube import IX_det_He3tube
from pyhorace.data.instr_classes.ix_det_slab import IX_det_slab
from pyhorace.data.instr_classes.ix_det_tobyfit_classic import IX_det_TobyfitClassic


def _det_orient_trans(opt: str, val_in: Any) -> np.ndarray:
    """Validate and convert detector orientation ('dmat' or 'rotvec') into internal dmat array.

    Equivalent to MATLAB ``det_orient_trans(opt, val_in)``.
    """
    if val_in is None:
        raise ValueError(
            f"HERBERT:IX_detector_bank:invalid_argument: {opt} cannot be empty"
        )
    arr = np.asarray(val_in, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"HERBERT:IX_detector_bank:invalid_argument: {opt} must be finite numeric array"
        )

    if opt == "dmat":
        if arr.ndim not in (2, 3) or arr.shape[0] != 3 or arr.shape[1] != 3:
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "dmat must be a 3x3 matrix or 3x3xN array of matrices"
            )
        # Validate rotation matrix/matrices (orthogonal with determinant +1)
        mats = arr.reshape(3, 3, -1)
        for i in range(mats.shape[2]):
            m = mats[:, :, i]
            if not (
                np.allclose(m @ m.T, np.eye(3), atol=1e-5)
                and np.isclose(np.linalg.det(m), 1.0, atol=1e-5)
            ):
                raise ValueError(
                    "HERBERT:IX_detector_bank:invalid_argument: "
                    "Input matrix is not a valid rotation matrix"
                )
        try:
            rotmat_to_rotvec(arr if arr.ndim == 2 else mats)
        except Exception as exc:
            raise ValueError(
                f"HERBERT:IX_detector_bank:invalid_argument: {exc}"
            ) from exc

        if arr.ndim == 3 and arr.shape[2] == 1:
            return arr.reshape(3, 3).copy()
        return arr.copy()

    if opt == "rotvec":
        if arr.size == 3 and (
            arr.ndim == 1
            or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1))
        ):
            arr = arr.ravel(order="F").reshape(3, 1)
        elif arr.ndim != 2 or arr.shape[0] != 3:
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "rotvec must be a vector length 3 or array size [3,N]"
            )
        try:
            rm = rotvec_to_rotmat(arr)
        except Exception as exc:
            raise ValueError(
                f"HERBERT:IX_detector_bank:invalid_argument: {exc}"
            ) from exc
        return np.swapaxes(rm, 0, 1).copy()

    raise ValueError(
        "HERBERT:IX_detector_bank:invalid_argument: Invalid option in det_orient_trans"
    )


class IX_detector_bank(HashableObj):
    """Detector bank storing positions, orientations, and physical detector elements."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._id: np.ndarray = np.ones((1, 1), dtype=int)
        self._x2: np.ndarray = np.zeros((1, 1), dtype=float)
        self._phi: np.ndarray = np.zeros((1, 1), dtype=float)
        self._azim: np.ndarray = np.zeros((1, 1), dtype=float)
        self._dmat: np.ndarray = np.eye(3, dtype=float)
        self._det: IX_det_abstractType = IX_det_slab()

        if len(args) > 0 or len(kwargs) > 0:
            property_names = ["id", "x2", "phi", "azim", "det", "dmat", "rotvec"]
            mandatory = [True, True, True, True, True, False, False]

            all_tokens = list(args)
            for k, v in kwargs.items():
                all_tokens.extend([k, v])

            orient_set = [
                isinstance(x, str)
                and len(x) > 0
                and (
                    "dmat".startswith(x.lstrip("-").lower())
                    or "rotvec".startswith(x.lstrip("-").lower())
                )
                for x in all_tokens
            ]
            if sum(orient_set) > 1 or (len(all_tokens) == 7 and not any(orient_set)):
                raise ValueError(
                    "HERBERT:IX_detector_bank:invalid_argument: "
                    "Ambiguous input: Check number of input arguments and options"
                )

            options = {"key_dash": False, "mandatory_props": mandatory}
            _, remains = self.set_positional_and_key_val_arguments(
                property_names, options, *args, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:IX_detector_bank:invalid_argument: "
                    f"Unrecognised extra parameters provided as input to "
                    f"IX_detector_bank constructor: {remains}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def id(self) -> np.ndarray:
        return self._id.copy()

    @id.setter
    def id(self, val: Any) -> None:
        ok, mess, _, _, _ = is_integer_id(val)
        if not ok:
            raise ValueError(
                f"HERBERT:IX_detector_bank:invalid_argument: "
                f"Invalid detector array identifier numbers: {mess}"
            )
        self._id = np.asarray(val, dtype=int).ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def x2(self) -> np.ndarray:
        return self._x2.copy()

    @x2.setter
    def x2(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Sample - detector distance(s) must be greater or equal to zero"
            )
        self._x2 = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def phi(self) -> np.ndarray:
        return self._phi.copy()

    @phi.setter
    def phi(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if (
            arr.size == 0
            or not np.all(np.isfinite(arr))
            or np.any(arr < 0)
            or np.any(arr > 180)
        ):
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Scattering angle(s) 'phi' must lie between 0 and 180 degrees"
            )
        self._phi = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def azim(self) -> np.ndarray:
        return self._azim.copy()

    @azim.setter
    def azim(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Azimuthal angle(s) 'azim' must be numeric"
            )
        self._azim = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def dmat(self) -> np.ndarray:
        return self._dmat.copy()

    @dmat.setter
    def dmat(self, val: Any) -> None:
        self._dmat = _det_orient_trans("dmat", val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def rotvec(self) -> np.ndarray:
        rv = rotmat_to_rotvec(np.swapaxes(self._dmat, 0, 1))
        if rv.ndim == 1:
            return rv.reshape(3, 1)
        return rv

    @rotvec.setter
    def rotvec(self, val: Any) -> None:
        self._dmat = _det_orient_trans("rotvec", val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def det(self) -> IX_det_abstractType:
        return copy.deepcopy(self._det)

    @det.setter
    def det(self, val: Any) -> None:
        if isinstance(val, dict):
            if "serial_name" in val:
                val = Serializable.from_struct(val)
            elif "wall_" in val or "wall" in val:
                val = IX_det_He3tube().from_old_struct(val)
            elif "depth_" in val or "depth" in val:
                val = IX_det_slab().from_old_struct(val)
            elif "dia_" in val or "dia" in val:
                val = IX_det_TobyfitClassic().from_old_struct(val)
            else:
                raise ValueError(
                    f"HERBERT:IX_detector_bank:invalid_argument: "
                    f"Unrecognised old detector structure with fields: {list(val.keys())}"
                )
        if not isinstance(val, IX_det_abstractType):
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Detector type must be a scalar instance of a subclass of IX_det_abstractType"
            )
        self._det = copy.deepcopy(val)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def width(self) -> np.ndarray:
        return self._det.width

    @property
    def height(self) -> np.ndarray:
        return self._det.height

    @property
    def ndet(self) -> int:
        return self._det.ndet

    # ------------------------------------------------------------------
    # Detector evaluation methods
    # ------------------------------------------------------------------
    def func_eval(
        self,
        func_handle: Callable[..., Any],
        *args: Any,
        return_sz_val: bool = False,
    ) -> Any:
        """Evaluate a method of the underlying detector type object.

        Equivalent to MATLAB ``[val, sz_val] = func_eval(obj, func_handle, varargin{:})``.
        """
        sz, ind, wvec = self._det.parse_ind_wvec_(*args)
        ind_flat = ind.ravel(order="F")
        if np.any(ind_flat < 1) or np.any(ind_flat > self.ndet):
            raise IndexError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Detector index out of bounds"
            )
        ind_0 = ind_flat - 1
        dmat_3d = self._dmat[:, :, np.newaxis] if self._dmat.ndim == 2 else self._dmat
        npath = dmat_3d[0][:, ind_0]

        if isinstance(func_handle, str):
            val = getattr(self._det, func_handle)(ind, npath, wvec)
        elif (
            hasattr(func_handle, "__name__")
            and getattr(IX_det_abstractType, func_handle.__name__, None) is func_handle
        ):
            val = getattr(self._det, func_handle.__name__)(ind, npath, wvec)
        else:
            val = func_handle(self._det, ind, npath, wvec)
        if return_sz_val:
            sz_val = size_array_split(matlab_size(val), sz)
            return val, sz_val
        return val

    def effic(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.effic, *args)

    def mean_d(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.mean_d, *args)

    def var_d(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.var_d, *args)

    def mean_x(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.mean_x, *args)

    def mean_y(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.mean_y, *args)

    def mean_z(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.mean_z, *args)

    def mean(self, *args: Any) -> np.ndarray:
        return self.func_eval(IX_det_abstractType.mean, *args)

    def var_x(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.var_x, *args)

    def var_y(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.var_y, *args)

    def var_z(self, *args: Any) -> float | np.ndarray:
        return self.func_eval(IX_det_abstractType.var_z, *args)

    def covariance(self, *args: Any) -> np.ndarray:
        return self.func_eval(IX_det_abstractType.covariance, *args)

    def rand(self, *args: Any) -> np.ndarray:
        return self.func_eval(IX_det_abstractType.rand, *args)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> list[str]:
        return ["id", "x2", "phi", "azim", "dmat", "det"]

    def check_combo_arg(self) -> IX_detector_bank:
        nd = self._id.size

        self._x2, self._phi, self._azim = expand_args_by_ref(
            self._id, self._x2, self._phi, self._azim
        )
        self._x2 = self._x2.astype(float).reshape(-1, 1)
        self._phi = self._phi.astype(float).reshape(-1, 1)
        self._azim = self._azim.astype(float).reshape(-1, 1)

        if self._dmat.ndim == 2 or self._dmat.shape[2] == 1:
            if nd > 1:
                self._dmat = np.repeat(self._dmat.reshape(3, 3, 1), nd, axis=2)
            else:
                self._dmat = self._dmat.reshape(3, 3)
        elif self._dmat.shape[2] != nd:
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Rotation matrix array argument must refer to a single rotation matrix "
                "or a stack with length equal to the number of detectors"
            )

        if self._det.ndet == 1 and nd > 1:
            self._det = self._det.replicate(nd)
        elif self._det.ndet != nd:
            raise ValueError(
                "HERBERT:IX_detector_bank:invalid_argument: "
                "Number of detectors in the detector definitions object ('det') "
                "is inconsistent with the number of detector identifiers ('id')"
            )

        ok, mess, ix, _, _ = is_integer_id(self._id)
        if not ok:
            raise ValueError(
                f"HERBERT:IX_detector_bank:invalid_argument: "
                f"Invalid detector array identifier numbers: {mess}"
            )
        if ix.size > 0:
            ix_0 = ix.ravel(order="F") - 1
            self._id = self._id[ix_0].reshape(-1, 1)
            self._x2 = self._x2[ix_0].reshape(-1, 1)
            self._phi = self._phi[ix_0].reshape(-1, 1)
            self._azim = self._azim[ix_0].reshape(-1, 1)
            self._dmat = self._dmat[:, :, ix_0]
            self._det = self._det.reorder(ix)

        self.clear_hash()
        return self

    def convert_old_struct(
        self, s: dict[str, Any], ver: float
    ) -> tuple[dict[str, Any], IX_detector_bank]:
        inputs = dict(s)
        if "class_version_" in inputs and inputs["class_version_"] == 1:
            inputs.pop("class_version_", None)
            new_inputs = {}
            for key, val in inputs.items():
                new_key = key[:-1] if key.endswith("_") else key
                new_inputs[new_key] = val
            return new_inputs, self
        new_inputs = {}
        for key, val in inputs.items():
            if key in ("id_", "x2_", "phi_", "azim_", "dmat_", "det_"):
                new_inputs[key[:-1]] = val
            else:
                new_inputs[key] = val
        return new_inputs, self

