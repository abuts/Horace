"""Cuboidal (slab) detector class (IX_det_slab)."""

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
    rand_truncexp2,
    size_array_stack,
)


def _series_eval(x: np.ndarray, c: np.ndarray, nmin: int = 0) -> np.ndarray:
    """Evaluate factored series expansion (equivalent to MATLAB ``series_eval``)."""
    val = np.full(x.shape, float(c[-1]), dtype=float)
    for i in range(c.size - 2, -1, -1):
        val = float(c[i]) * (1.0 + x * val)
    if nmin != 0:
        if nmin > 0:
            val = val * (x**nmin)
        else:
            val = val / (x**nmin)
    return val


def _effic_alf(alf: np.ndarray) -> np.ndarray:
    """Efficiency of a slab detector given dimensionless thickness alf."""
    small = alf < 0.1
    c = np.concatenate([[1.0], -1.0 / np.arange(2.0, 11.0)])
    val = np.zeros_like(alf, dtype=float)
    if np.any(~small):
        val[~small] = 1.0 - np.exp(-alf[~small])
    if np.any(small):
        val[small] = _series_eval(alf[small], c, 1)
    return val


def _mean_d_alf(alf: np.ndarray) -> np.ndarray:
    """Mean depth of absorption in a slab detector in reduced units."""
    small = alf < 0.1
    c = np.array([-1.0 / 12.0, -1.0 / 60.0, -1.0 / 42.0, -1.0 / 40.0, -5.0 / 198.0], dtype=float)
    val = np.zeros_like(alf, dtype=float)
    if np.any(~small):
        alfbig = alf[~small]
        expon = np.exp(-alfbig)
        val[~small] = -0.5 + 1.0 / alfbig - expon / (1.0 - expon)
    if np.any(small):
        alfsmall = alf[small]
        val[small] = alfsmall * _series_eval(alfsmall**2, c, 0)
    return val


def _var_d_alf(alf: np.ndarray) -> np.ndarray:
    """Variance of depth of absorption in a slab detector in reduced units."""
    small = alf < 0.35
    c = np.array(
        [
            1.0 / 12.0,
            -1.0 / 20.0,
            -5.0 / 126.0,
            -7.0 / 200.0,
            -5.0 / 154.0,
            -7601.0 / 245700.0,
            -455.0 / 15202.0,
        ],
        dtype=float,
    )
    val = np.zeros_like(alf, dtype=float)
    if np.any(~small):
        alfbig = alf[~small]
        expon = np.exp(-alfbig)
        val[~small] = 1.0 / (alfbig**2) - expon / ((1.0 - expon) ** 2)
    if np.any(small):
        alfsmall = alf[small]
        val[small] = _series_eval(alfsmall**2, c, 0)
    return val


class IX_det_slab(IX_det_abstractType):
    """Array of cuboidal (slab) detectors."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._depth: np.ndarray = np.zeros((1, 1), dtype=float)
        self._width: np.ndarray = np.zeros((1, 1), dtype=float)
        self._height: np.ndarray = np.zeros((1, 1), dtype=float)
        self._atten: np.ndarray = np.zeros((1, 1), dtype=float)

        if len(args) > 0 or len(kwargs) > 0:
            property_names = ["depth", "width", "height", "atten"]
            options = {"key_dash": True, "mandatory_props": [True, True, True, True]}
            _, remains = self.set_positional_and_key_val_arguments(
                property_names, options, *args, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:IX_det_slab:invalid_argument: "
                    f"Unrecognised extra parameters provided as input to "
                    f"IX_det_slab constructor: {remains}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def depth(self) -> np.ndarray:
        return self._depth.copy()

    @depth.setter
    def depth(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Detector element depth(s) must be greater or equal to zero"
            )
        self._depth = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def width(self) -> np.ndarray:
        return self._width.copy()

    @width.setter
    def width(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Detector element width(s) must be greater or equal to zero"
            )
        self._width = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def height(self) -> np.ndarray:
        return self._height.copy()

    @height.setter
    def height(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Detector element height(s) must be greater or equal to zero"
            )
        self._height = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def atten(self) -> np.ndarray:
        return self._atten.copy()

    @atten.setter
    def atten(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Detector element attenuation length(s) must be greater or equal to zero"
            )
        self._atten = arr.ravel(order="F").reshape(-1, 1)
        if self.do_check_combo_arg:
            self.check_combo_arg()

    def get_ndet(self) -> int:
        return int(self._depth.size)

    # ------------------------------------------------------------------
    # Reorder and replicate
    # ------------------------------------------------------------------
    def reorder(self, ix: Any) -> IX_det_slab:
        ndet = self.ndet
        ix_arr = np.asarray(ix, dtype=int).ravel(order="F")
        ok, _, _, perm, _ = is_integer_id(ix_arr)
        if not (ok and perm and ix_arr.size == ndet):
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Index array is not a permutation of the integers 1 to the "
                "number of detector elements"
            )
        ix_0 = ix_arr - 1
        obj_out = copy.deepcopy(self)
        obj_out._depth = self._depth[ix_0].reshape(-1, 1)
        obj_out._width = self._width[ix_0].reshape(-1, 1)
        obj_out._height = self._height[ix_0].reshape(-1, 1)
        obj_out._atten = self._atten[ix_0].reshape(-1, 1)
        obj_out.clear_hash()
        return obj_out

    def replicate(self, n: int) -> IX_det_slab:
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 1:
            raise ValueError(
                "HERBERT:IX_det_slab:invalid_argument: "
                "Replication factor must be a positive integer"
            )
        obj_out = copy.deepcopy(self)
        if n > 1:
            obj_out._depth = np.tile(self._depth, (int(n), 1))
            obj_out._width = np.tile(self._width, (int(n), 1))
            obj_out._height = np.tile(self._height, (int(n), 1))
            obj_out._atten = np.tile(self._atten, (int(n), 1))
            obj_out.clear_hash()
        return obj_out

    # ------------------------------------------------------------------
    # Internal helper: macro_xs_thick
    # ------------------------------------------------------------------
    def _macro_xs_thick(
        self, ind: np.ndarray, npath: np.ndarray, wvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        wvec0 = 3.494157974647365
        ind_0 = ind.ravel(order="F") - 1
        atten0 = self._atten[ind_0, 0]
        thickness = (self._depth[ind_0, 0] / npath[0, :]) / atten0

        if ind.size == 1:
            sz_w = matlab_size(wvec)
            w_arr = wvec.reshape(sz_w, order="F")
            alf = (wvec0 * thickness[0]) / w_arr
            atten = w_arr * (atten0[0] / wvec0)
        elif wvec.size == 1:
            sz_i = matlab_size(ind)
            w_val = float(wvec.item())
            alf = (wvec0 / w_val) * thickness.reshape(sz_i, order="F")
            atten = (w_val / wvec0) * atten0.reshape(sz_i, order="F")
        else:
            sz_w = matlab_size(wvec)
            w_arr = wvec.reshape(sz_w, order="F")
            alf = wvec0 * (thickness.reshape(sz_w, order="F") / w_arr)
            atten = (w_arr * atten0.reshape(sz_w, order="F")) / wvec0

        return alf, atten

    # ------------------------------------------------------------------
    # Detector evaluation methods
    # ------------------------------------------------------------------
    def effic(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, _ = self._macro_xs_thick(ind, npath, wvec)
        val = _effic_alf(alf)
        return format_scalar_result(val, sz)

    def mean_d(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, _ = self._macro_xs_thick(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        thickness = self._depth[ind_0, 0] / npath[0, :]
        if ind.size != 1:
            thickness = thickness.reshape(alf.shape, order="F")
        else:
            thickness = float(thickness[0])
        val = thickness * _mean_d_alf(alf)
        return format_scalar_result(val, sz)

    def var_d(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, _ = self._macro_xs_thick(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        thickness = self._depth[ind_0, 0] / npath[0, :]
        if ind.size != 1:
            thickness = thickness.reshape(alf.shape, order="F")
        else:
            thickness = float(thickness[0])
        val = (thickness**2) * _var_d_alf(alf)
        return format_scalar_result(val, sz)

    def mean_x(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, _ = self._macro_xs_thick(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        thickness = self._depth[ind_0, 0]
        if ind.size != 1:
            thickness = thickness.reshape(alf.shape, order="F")
        else:
            thickness = float(thickness[0])
        val = thickness * _mean_d_alf(alf)
        return format_scalar_result(val, sz)

    def mean_y(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean_z(self, *args: Any) -> float | np.ndarray:
        sz, _, _, _ = self.parse_ind_npath_wvec_(*args)
        return format_scalar_result(np.zeros(sz, dtype=float), sz)

    def mean(self, *args: Any) -> np.ndarray:
        mx = np.asarray(self.mean_x(*args), dtype=float)
        my = np.asarray(self.mean_y(*args), dtype=float)
        mz = np.asarray(self.mean_z(*args), dtype=float)
        sz_mx = matlab_size(mx)
        xyz = np.vstack(
            [
                mx.ravel(order="F").reshape(1, -1),
                my.ravel(order="F").reshape(1, -1),
                mz.ravel(order="F").reshape(1, -1),
            ]
        )
        sz_full = size_array_stack((3, 1), sz_mx)
        return xyz.reshape(sz_full, order="F")

    def var_x(self, *args: Any) -> float | np.ndarray:
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, _ = self._macro_xs_thick(ind, npath, wvec)
        ind_0 = ind.ravel(order="F") - 1
        thickness = self._depth[ind_0, 0]
        if ind.size != 1:
            thickness = thickness.reshape(alf.shape, order="F")
        else:
            thickness = float(thickness[0])
        val = (thickness**2) * _var_d_alf(alf)
        return format_scalar_result(val, sz)

    def var_y(self, *args: Any) -> float | np.ndarray:
        sz, ind, _, _ = self.parse_ind_npath_wvec_(*args)
        ind_0 = ind.ravel(order="F") - 1
        width = self._width[ind_0, 0]
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
        sz, ind, npath, wvec = self.parse_ind_npath_wvec_(*args)
        alf, atten = self._macro_xs_thick(ind, npath, wvec)
        alf_col = alf.ravel(order="F").reshape(-1, 1)
        atten_col = atten.ravel(order="F").reshape(-1, 1)
        npath_x = npath[0, :].reshape(-1, 1)
        xscalefactor = atten_col * npath_x

        ind_0 = ind.ravel(order="F") - 1
        thickness = self._depth[ind_0, 0].reshape(-1, 1)
        width = self._width[ind_0, 0].reshape(-1, 1)
        height = self._height[ind_0, 0].reshape(-1, 1)
        n = int(np.prod(sz))

        xyz = np.vstack(
            [
                (xscalefactor * rand_truncexp2(alf_col) - 0.5 * thickness).T,
                (width * (np.random.rand(n, 1) - 0.5)).T,
                (height * (np.random.rand(n, 1) - 0.5)).T,
            ]
        )
        sz_full = size_array_stack((3, 1), sz)
        return xyz.reshape(sz_full, order="F")

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> list[str]:
        return ["depth", "width", "height", "atten"]

    def check_combo_arg(self) -> IX_det_slab:
        flds = self.saveable_fields()
        self.expand_internal_properties_to_max_length(flds)
        self.clear_hash()
        return self

    def convert_old_struct(
        self, s: dict[str, Any], ver: float
    ) -> tuple[dict[str, Any], IX_det_slab]:
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
            if key in ("depth_", "width_", "height_", "atten_"):
                new_inputs[key[:-1]] = val
            else:
                new_inputs[key] = val
        return new_inputs, self

