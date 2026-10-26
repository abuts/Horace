"""Detector array class (IX_detector_array) composed of one or more IX_detector_bank objects."""

from __future__ import annotations

import copy
from typing import Any, Callable, Sequence

import numpy as np

from pyhorace.core.hashable import HashableObj
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_abstract_type import (
    IX_det_abstractType,
    is_integer_id,
    size_array_stack,
)
from pyhorace.data.instr_classes.ix_det_he3tube import IX_det_He3tube
from pyhorace.data.instr_classes.ix_det_tobyfit_classic import IX_det_TobyfitClassic
from pyhorace.data.instr_classes.ix_detector_bank import (
    IX_detector_bank,
    _det_orient_trans,
)

_DETPAR_FIELDS = (
    "filename",
    "filepath",
    "group",
    "x2",
    "phi",
    "azim",
    "width",
    "height",
)


def _is_detpar_dict(val: Any) -> bool:
    """Return True if *val* is a dict with the exact fields of a Horace detpar struct."""
    if not isinstance(val, dict):
        return False
    return set(val.keys()) == set(_DETPAR_FIELDS)


def _check_det_bank(val: Any) -> list[IX_detector_bank]:
    """Validate and normalise detector bank input into a list of IX_detector_bank objects.

    Equivalent to MATLAB ``check_det_bank_(val)``.
    """
    if isinstance(val, dict) and "serial_name" in val:
        val = Serializable.from_struct(val)

    if isinstance(val, IX_detector_bank):
        banks = [copy.deepcopy(val)]
    elif isinstance(val, (list, tuple, np.ndarray)) and len(val) > 0:
        banks = []
        for item in val:
            if isinstance(item, dict) and "serial_name" in item:
                item = Serializable.from_struct(item)
            if not isinstance(item, IX_detector_bank):
                raise ValueError(
                    "HERBERT:IX_detector_array:invalid_argument: "
                    "All detector banks must be IX_detector_bank objects"
                )
            banks.append(copy.deepcopy(item))
    else:
        raise ValueError(
            "HERBERT:IX_detector_array:invalid_argument: "
            "Detector bank argument must be one or more IX_detector_bank objects"
        )

    if len(banks) > 1:
        all_ids = np.vstack([b.id for b in banks])
        ok, mess, _, _, _ = is_integer_id(all_ids)
        if not ok:
            raise ValueError(
                f"HERBERT:IX_detector_array:invalid_argument: "
                f"Detector identifiers across banks are not unique: {mess}"
            )

    return banks


class IX_detector_array(HashableObj):
    """Array of one or more detector banks (`IX_detector_bank`)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._det_bank: list[IX_detector_bank] = [IX_detector_bank()]
        self._filename: str = ""
        self._filepath: str = ""
        self._detdcn_cache: np.ndarray | None = None

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, IX_detector_array):
                self._det_bank = [copy.deepcopy(b) for b in arg0._det_bank]
                self._filename = arg0._filename
                self._filepath = arg0._filepath
                return
            if isinstance(arg0, IX_detector_bank) or (
                isinstance(arg0, (list, tuple, np.ndarray))
                and len(arg0) > 0
                and all(isinstance(x, IX_detector_bank) for x in arg0)
            ):
                self._det_bank = _check_det_bank(arg0)
                return
            if _is_detpar_dict(arg0):
                dtype, dparms = IX_det_abstractType.tobyfit_defaults()
                if dtype == "IX_det_He3tube":
                    det: IX_det_abstractType = IX_det_He3tube(
                        arg0["width"],
                        arg0["height"],
                        dparms["wall"],
                        dparms["p"],
                    )
                elif dtype in ("IX_det_TobyfitClassic", None, ""):
                    det = IX_det_TobyfitClassic(arg0["width"], arg0["height"])
                else:
                    raise ValueError(
                        f"HORACE:IX_detector_array:invalid_type: "
                        f"unsupported detector type {dtype!r}"
                    )
                self._det_bank = [
                    IX_detector_bank(
                        arg0["group"],
                        arg0["x2"],
                        arg0["phi"],
                        arg0["azim"],
                        det,
                    )
                ]
                self.filename = arg0["filename"]
                self.filepath = arg0["filepath"]
                return

        if len(kwargs) == 0 and all(isinstance(a, IX_detector_bank) for a in args):
            self._det_bank = _check_det_bank(list(args))
            return

        # Otherwise construct a single IX_detector_bank from the arguments
        self._det_bank = [IX_detector_bank(*args, **kwargs)]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def det_bank(self) -> IX_detector_bank | list[IX_detector_bank]:
        if len(self._det_bank) == 1:
            return copy.deepcopy(self._det_bank[0])
        return [copy.deepcopy(b) for b in self._det_bank]

    @det_bank.setter
    def det_bank(self, val: Any) -> None:
        self._det_bank = _check_det_bank(val)
        self._detdcn_cache = None
        self.clear_hash()

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, val: Any) -> None:
        if val is None:
            self._filename = ""
            return
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: filename must be a string"
            )
        self._filename = val

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, val: Any) -> None:
        if val is None:
            self._filepath = ""
            return
        if not isinstance(val, str):
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: filepath must be a string"
            )
        self._filepath = val

    @property
    def nbank(self) -> int:
        return len(self._det_bank)

    @property
    def ndet(self) -> int:
        return int(sum(b.ndet for b in self._det_bank))

    @property
    def ndet_bank(self) -> int | np.ndarray:
        if len(self._det_bank) == 1:
            return int(self._det_bank[0].ndet)
        return np.array([b.ndet for b in self._det_bank], dtype=int)

    @property
    def id(self) -> np.ndarray:
        return np.vstack([b.id for b in self._det_bank])

    @id.setter
    def id(self, val: Any) -> None:
        arr = np.asarray(val)
        if arr.size != self.ndet:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of elements must match the total number of detectors "
                "in the detector bank(s)"
            )
        ok, mess, _, _, _ = is_integer_id(arr)
        if not ok:
            raise ValueError(
                f"HERBERT:IX_detector_array:invalid_argument: "
                f"Invalid detector array identifier numbers: {mess}"
            )
        flat = arr.ravel(order="F")
        counts = [b.ndet for b in self._det_bank]
        ends = np.cumsum(counts)
        begs = ends - counts
        for b, beg, end in zip(self._det_bank, begs, ends):
            b.id = flat[beg:end]
        self._detdcn_cache = None
        self.clear_hash()

    @property
    def group(self) -> np.ndarray:
        return self.id.T

    @property
    def x2(self) -> np.ndarray:
        return np.vstack([b.x2 for b in self._det_bank])

    @x2.setter
    def x2(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 1:
            scalar = float(arr.ravel()[0])
            for b in self._det_bank:
                b.x2 = scalar
        elif arr.size == self.ndet:
            flat = arr.ravel(order="F")
            counts = [b.ndet for b in self._det_bank]
            ends = np.cumsum(counts)
            begs = ends - counts
            for b, beg, end in zip(self._det_bank, begs, ends):
                b.x2 = flat[beg:end]
        else:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of elements must equal unity or match the total "
                "number of detectors in the detector bank(s)"
            )
        self.clear_hash()

    @property
    def phi(self) -> np.ndarray:
        return np.vstack([b.phi for b in self._det_bank])

    @phi.setter
    def phi(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 1:
            scalar = float(arr.ravel()[0])
            for b in self._det_bank:
                b.phi = scalar
        elif arr.size == self.ndet:
            flat = arr.ravel(order="F")
            counts = [b.ndet for b in self._det_bank]
            ends = np.cumsum(counts)
            begs = ends - counts
            for b, beg, end in zip(self._det_bank, begs, ends):
                b.phi = flat[beg:end]
        else:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of elements must equal unity or match the total "
                "number of detectors in the detector bank(s)"
            )
        self._detdcn_cache = None
        self.clear_hash()

    @property
    def azim(self) -> np.ndarray:
        return np.vstack([b.azim for b in self._det_bank])

    @azim.setter
    def azim(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.size == 1:
            scalar = float(arr.ravel()[0])
            for b in self._det_bank:
                b.azim = scalar
        elif arr.size == self.ndet:
            flat = arr.ravel(order="F")
            counts = [b.ndet for b in self._det_bank]
            ends = np.cumsum(counts)
            begs = ends - counts
            for b, beg, end in zip(self._det_bank, begs, ends):
                b.azim = flat[beg:end]
        else:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of elements must equal unity or match the total "
                "number of detectors in the detector bank(s)"
            )
        self._detdcn_cache = None
        self.clear_hash()

    @property
    def dmat(self) -> np.ndarray:
        mats = []
        for b in self._det_bank:
            dm = b.dmat
            if dm.ndim == 2:
                dm = dm[:, :, np.newaxis]
            mats.append(dm)
        out = np.concatenate(mats, axis=2)
        if out.shape[2] == 1:
            return out.reshape(3, 3)
        return out

    @dmat.setter
    def dmat(self, val: Any) -> None:
        arr = _det_orient_trans("dmat", val)
        if arr.ndim == 2:
            for b in self._det_bank:
                b.dmat = arr
        elif arr.ndim == 3 and arr.shape[2] == self.ndet:
            counts = [b.ndet for b in self._det_bank]
            ends = np.cumsum(counts)
            begs = ends - counts
            for b, beg, end in zip(self._det_bank, begs, ends):
                b.dmat = arr[:, :, beg:end]
        else:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of rotation matrices must equal unity or match the total "
                "number of detectors in the detector bank(s)"
            )
        self.clear_hash()

    @property
    def rotvec(self) -> np.ndarray:
        return np.concatenate([b.rotvec for b in self._det_bank], axis=1)

    @rotvec.setter
    def rotvec(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        is_single_vec = arr.size == 3 and (
            arr.ndim == 1
            or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1))
        )
        if is_single_vec:
            vec = arr.ravel(order="F")
            for b in self._det_bank:
                b.rotvec = vec
        elif arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] == self.ndet:
            counts = [b.ndet for b in self._det_bank]
            ends = np.cumsum(counts)
            begs = ends - counts
            for b, beg, end in zip(self._det_bank, begs, ends):
                b.rotvec = arr[:, beg:end]
        else:
            raise ValueError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "The number of rotation vectors must equal unity or match the total "
                "number of detectors in the detector bank(s)"
            )
        self.clear_hash()

    @property
    def width(self) -> np.ndarray:
        return np.vstack([b.width for b in self._det_bank])

    @property
    def height(self) -> np.ndarray:
        return np.vstack([b.height for b in self._det_bank])

    # ------------------------------------------------------------------
    # Detector evaluation methods
    # ------------------------------------------------------------------
    def _repackage_ind_wvec(
        self, *args: Any
    ) -> tuple[
        tuple[int, ...],
        list[int],
        list[np.ndarray],
        list[np.ndarray],
        np.ndarray,
    ]:
        """Partition detector indices and wavevectors across banks.

        Equivalent to MATLAB ``[sz_stack, ibank, ind_bank, wvec_bank, ix] = repackage_ind_wvec_(obj, varargin{:})``.
        """
        counts = np.array([b.ndet for b in self._det_bank], dtype=int)
        ind2bank = np.repeat(np.arange(len(counts), dtype=int), counts)
        ind_offset = np.cumsum(counts) - counts + 1  # 1-based offset of each bank

        sz_stack, ind, wvec = self._det_bank[0]._det.parse_ind_wvec_(*args)
        ind_flat = ind.ravel(order="F")
        if np.any(ind_flat < 1) or np.any(ind_flat > self.ndet):
            raise IndexError(
                "HERBERT:IX_detector_array:invalid_argument: "
                "Detector indices are outside the range of the detector array"
            )

        bank_for_el = ind2bank[ind_flat - 1]
        unique_banks = np.unique(bank_for_el)

        if unique_banks.size == 1:
            b_idx = int(unique_banks[0])
            ibank = [b_idx]
            ind_bank = [ind - int(ind_offset[b_idx]) + 1]
            wvec_bank = [wvec]
            ix = np.array([], dtype=int)
            return sz_stack, ibank, ind_bank, wvec_bank, ix

        # Stable sort so detectors belonging to the same bank stay in original relative order
        ix = np.argsort(bank_for_el, kind="mergesort")
        sorted_banks = bank_for_el[ix]
        sorted_ind = ind_flat[ix]

        ibank = [int(b) for b in unique_banks]
        ind_bank = []
        wvec_bank = []
        wvec_flat = wvec.ravel(order="F")
        scalar_wvec = wvec.size == 1

        for b_idx in ibank:
            mask = sorted_banks == b_idx
            local_ind = (sorted_ind[mask] - int(ind_offset[b_idx]) + 1).reshape(-1, 1)
            ind_bank.append(local_ind)
            if scalar_wvec:
                wvec_bank.append(wvec)
            else:
                wvec_bank.append(wvec_flat[ix[mask]].reshape(-1, 1))

        return sz_stack, ibank, ind_bank, wvec_bank, ix

    def func_eval(self, func_handle: Callable[..., Any], *args: Any) -> Any:
        """Evaluate a method of the underlying detector type across banks.

        Equivalent to MATLAB ``val = func_eval(obj, func_handle, varargin{:})``.
        """
        sz_stack, ibank, ind_bank, wvec_bank, ix = self._repackage_ind_wvec(*args)
        nbank_eval = len(ibank)

        if nbank_eval == 1:
            return self._det_bank[ibank[0]].func_eval(
                func_handle, ind_bank[0], wvec_bank[0]
            )

        vals = []
        sz_roots = []
        nind_bank = [arr.size for arr in ind_bank]

        for i in range(nbank_eval):
            v, sz_r = self._det_bank[ibank[i]].func_eval(
                func_handle, ind_bank[i], wvec_bank[i], return_sz_val=True
            )
            sz_roots.append(sz_r)
            nel = int(np.prod(sz_r))
            vals.append( np.asarray(v, dtype=float).reshape((nel, nind_bank[i]), order="F") )

        sz_root = sz_roots[0]
        if not all(sr == sz_root for sr in sz_roots[1:]):
            raise ValueError(
                "HERBERT:IX_detector_array:runtime_error: "
                "Different detector types give different return argument sizes "
                "for a single detector"
            )

        nel = int(np.prod(sz_root))
        val_cat = np.concatenate(vals, axis=1)
        val_out = np.empty_like(val_cat)
        val_out[:, ix] = val_cat

        out_shape = size_array_stack(sz_root, sz_stack)
        if out_shape == (1, 1):
            return float(val_out.ravel()[0])
        return val_out.reshape(out_shape, order="F")

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
    # Horace detpar / direction helpers
    # ------------------------------------------------------------------
    def calc_detdcn(self, idx: Any = None) -> np.ndarray:
        """Compute ``[cosd(phi); sind(phi)*cosd(azim); sind(phi)*sind(azim); id]`` (4 x ndet).

        Equivalent to MATLAB ``detdcn = obj.calc_detdcn([idx])``.
        """
        if idx is None or (isinstance(idx, (list, tuple, np.ndarray)) and len(idx) == 0):
            if self._detdcn_cache is not None and self._detdcn_cache.shape[1] == self.ndet:
                return self._detdcn_cache.copy()
            phi_rad = np.deg2rad(self.phi.ravel(order="F"))
            azim_rad = np.deg2rad(self.azim.ravel(order="F"))
            ids = self.id.ravel(order="F").astype(float)
            s_phi = np.sin(phi_rad)
            detdcn = np.vstack(
                [
                    np.cos(phi_rad),
                    s_phi * np.cos(azim_rad),
                    s_phi * np.sin(azim_rad),
                    ids,
                ]
            )
            self._detdcn_cache = detdcn
            return detdcn.copy()

        idx_arr = np.asarray(idx, dtype=int).ravel(order="F") - 1
        phi_rad = np.deg2rad(self.phi.ravel(order="F")[idx_arr])
        azim_rad = np.deg2rad(self.azim.ravel(order="F")[idx_arr])
        ids = self.id.ravel(order="F")[idx_arr].astype(float)
        s_phi = np.sin(phi_rad)
        return np.vstack(
            [
                np.cos(phi_rad),
                s_phi * np.cos(azim_rad),
                s_phi * np.sin(azim_rad),
                ids,
            ]
        )

    def get_detpar_representation(self) -> dict[str, Any]:
        """Return a Horace detpar dictionary representation of the detector array."""
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "group": self.group.copy(),
            "x2": self.x2.T.copy(),
            "phi": self.phi.T.copy(),
            "azim": self.azim.T.copy(),
            "width": self.width.T.copy(),
            "height": self.height.T.copy(),
        }

    @staticmethod
    def check_detpar_parms(dp: Any) -> IX_detector_array:
        """Convert a detpar structure or IX_detector_array into an IX_detector_array."""
        if isinstance(dp, IX_detector_array):
            return dp
        if isinstance(dp, dict):
            return IX_detector_array(dp)
        raise ValueError(
            "HORACE:IX_detector_array:invalid_argument: "
            "Input must be an IX_detector_array or a detpar dictionary"
        )

    # ------------------------------------------------------------------
    # Serializable / HashableObj interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 1

    def saveable_fields(self) -> list[str]:
        return ["det_bank", "filename", "filepath"]

    def hashable_fields(self) -> list[str]:
        return ["det_bank"]

    def convert_old_struct(
        self, s: dict[str, Any], ver: float
    ) -> tuple[dict[str, Any], IX_detector_array]:
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
            if key in ("det_bank_", "filename_", "filepath_"):
                new_inputs[key[:-1]] = val
            else:
                new_inputs[key] = val
        return new_inputs, self

