"""Old Tobyfit approximation to a 3He cylindrical tube detector."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_abstract_type import (
    IX_det_abstractType,
    format_scalar_result,
    is_integer_id,
    matlab_size,
    size_array_stack,
)


class IX_det_TobyfitClassic(IX_det_abstractType):
    """Old Tobyfit approximation to a cylindrical 3He detector tube."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._dia: np.ndarray = np.zeros((1, 1), dtype=float)
        self._height: np.ndarray = np.zeros((1, 1), dtype=float)

        if len(args) > 0 or len(kwargs) > 0:
            property_names = ["dia", "height"]
            options = {"key_dash": True, "mandatory_props": [True, True]}
            _, remains = self.set_positional_and_key_val_arguments(
                property_names, options, *args, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:IX_det_TobyfitClassic:invalid_argument: "
                    f"Unrecognised extra parameters provided as input to "
                    f"IX_det_TobyfitClassic constructor: {remains}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def dia(self) -> np.ndarray:
        return self._dia.copy()

    @dia.setter
    def dia(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_TobyfitClassic:invalid_argument: "
                "Tube diameter(s) must be greater or equal to zero"
            )
        self._dia = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def width(self) -> np.ndarray:
        return self._dia.copy()

    @property
    def height(self) -> np.ndarray:
        return self._height.copy()

    @height.setter
    def height(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_TobyfitClassic:invalid_argument: "
                "Detector element height(s) must be greater or equal to zero"
            )
        self._height = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    def get_ndet(self) -> int:
        return int(self._dia.size)

    # ------------------------------------------------------------------
    # Reorder and replicate
    # ------------------------------------------------------------------
    def reorder(self, ix: Any) -> IX_det_TobyfitClassic:
        ndet = self.ndet
        ix_arr = np.asarray(ix, dtype=int).ravel(order="F")
        ok, _, _, perm, _ = is_integer_id(ix_arr)
        if not (ok and perm and ix_arr.size == ndet):
            raise ValueError(
                "HERBERT:IX_det_TobyfitClassic:invalid_argument: "
                "Index array is not a permutation of the integers 1 to the "
                "number of detector elements"
            )
        ix_0 = ix_arr - 1
        obj_out = copy.deepcopy(self)
        obj_out._dia = self._dia[ix_0].reshape(-1, 1)
        obj_out._height = self._height[ix_0].reshape(-1, 1)
        obj_out.clear_hash()
        return obj_out

    def replicate(self, n: int) -> IX_det_TobyfitClassic:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError(
                "HERBERT:IX_det_TobyfitClassic:invalid_argument: "
                "Replication factor must be a positive integer"
            )
        obj_out = copy.deepcopy(self)
        if n > 1:
            obj_out._dia = np.tile(self._dia, (int(n), 1))
            obj_out._height = np.tile(self._height, (int(n), 1))
            obj_out.clear_hash()
        return obj_out

    # ------------------------------------------------------------------
    # Detector evaluation methods
    # ------------------------------------------------------------------
    def effic(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.ones(sz, dtype=float), sz)

    def mean_d(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean_x(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean_y(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean_z(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean(self, *args: Any) -> np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        sz_full = size_array_stack((3, 1), sz)
        return np.zeros(sz_full, dtype=float, order="F")

    def var_d(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        fac = 75.0 / 127.0
        ind_0 = ind.ravel(order="F") - 1
        depth = fac * self._dia[ind_0, 0]
        if ind.size != 1:
            val = (depth.reshape(sz, order="F") ** 2) / 12.0
        else:
            val = ((depth.item() ** 2) / 12.0) * np.ones(sz, dtype=float)
        return format_scalar_result(val, sz)

    def var_x(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        fac = 75.0 / 127.0
        ind_0 = ind.ravel(order="F") - 1
        depth = fac * self._dia[ind_0, 0]
        if ind.size != 1:
            val = (depth.reshape(sz, order="F") ** 2) / 12.0
        else:
            val = ((depth.item() ** 2) / 12.0) * np.ones(sz, dtype=float)
        return format_scalar_result(val, sz)

    def var_y(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        ind_0 = ind.ravel(order="F") - 1
        width = self._dia[ind_0, 0]
        if ind.size != 1:
            val = (width.reshape(sz, order="F") ** 2) / 12.0
        else:
            val = ((width.item() ** 2) / 12.0) * np.ones(sz, dtype=float)
        return format_scalar_result(val, sz)

    def var_z(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        ind_0 = ind.ravel(order="F") - 1
        height = self._height[ind_0, 0]
        if ind.size != 1:
            val = (height.reshape(sz, order="F") ** 2) / 12.0
        else:
            val = ((height.item() ** 2) / 12.0) * np.ones(sz, dtype=float)
        return format_scalar_result(val, sz)

    def covariance(self, *args: Any) -> np.ndarray:
        cxx = np.asarray(self.var_x(*args), dtype=float)
        cyy = np.asarray(self.var_y(*args), dtype=float)
        czz = np.asarray(self.var_z(*args), dtype=float)
        sz_cxx = matlab_size(cxx)

        n = cxx.size
        val = np.zeros((3, 3, n), dtype=float, order="F")
        val[0, 0, :] = cxx.ravel(order="F")
        val[1, 1, :] = cyy.ravel(order="F")
        val[2, 2, :] = czz.ravel(order="F")

        sz_full = size_array_stack((3, 3), sz_cxx)
        return val.reshape(sz_full, order="F")

    def rand(self, *args: Any) -> np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        fac = 75.0 / 127.0
        n = int(np.prod(sz))
        ind_0 = ind.ravel(order="F") - 1
        if ind.size != 1:
            x = (fac * self._dia[ind_0, 0]).reshape(1, n) * (np.random.rand(1, n) - 0.5)
            y = self._dia[ind_0, 0].reshape(1, n) * (np.random.rand(1, n) - 0.5)
            z = self._height[ind_0, 0].reshape(1, n) * (np.random.rand(1, n) - 0.5)
        else:
            x = (fac * self._dia[ind_0[0], 0]) * (np.random.rand(1, n) - 0.5)
            y = self._dia[ind_0[0], 0] * (np.random.rand(1, n) - 0.5)
            z = self._height[ind_0[0], 0] * (np.random.rand(1, n) - 0.5)

        xyz = np.vstack([x, y, z])
        sz_full = size_array_stack((3, 1), sz)
        return xyz.reshape(sz_full, order="F")

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> list[str]:
        return ["dia", "height"]

    def check_combo_arg(self) -> IX_det_TobyfitClassic:
        flds = self.saveable_fields()
        self.expand_internal_properties_to_max_length(flds)
        self.clear_hash()
        return self

    def convert_old_struct(
        self, s: dict[str, Any], ver: float
    ) -> tuple[dict[str, Any], IX_det_TobyfitClassic]:
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
            if key in ("dia_", "height_"):
                new_inputs[key[:-1]] = val
            else:
                new_inputs[key] = val
        return new_inputs, self

