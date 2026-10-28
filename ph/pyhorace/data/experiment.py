"""Experiment container class holding per-run metadata, instruments, samples, and detectors.

Converted from:
  horace_core/sqw/@Experiment/
"""
from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf import (
    HoraceError,
    LineProjBase,
    bmatrix,
    crystal_alignment_info,
    fast_map,
    ubmatrix,
)
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.hashable import build_hash
from pyhorace.core.serializable import Serializable, compare_values, register_serializable
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.instr_classes import (
    IX_detector_array,
    IX_experiment,
    IX_inst,
    IX_null_inst,
    IX_null_sample,
    IX_samp,
    IX_sample,
)
from pyhorace.data.ix_dataset import compact_array


def calc_idx_for_eq_to_tol(
    data: Sequence[Any],
    tolerance: Sequence[float] = (1e-9, 1e-9),
) -> List[np.ndarray]:
    """Group indices (1-based) of elements in ``data`` that are equal within ``tolerance``.

    Translated from ``herbert_core/utilities/misc/calc_idx_for_eq_to_tol.m``.
    """
    if isinstance(data, np.ndarray) and data.dtype != object:
        items = [float(x) for x in data.ravel()]
    elif isinstance(data, (list, tuple)):
        items = list(data)
    else:
        items = list(data)

    n_items = len(items)
    if n_items == 0:
        return []

    visited = np.zeros(n_items, dtype=bool)
    groups: List[np.ndarray] = []

    for i in range(n_items):
        if visited[i]:
            continue
        visited[i] = True
        ref_item = items[i]
        cur_group = [i + 1]
        for j in range(i + 1, n_items):
            if visited[j]:
                continue
            cand = items[j]
            if isinstance(ref_item, (int, float, np.number)) and isinstance(
                cand, (int, float, np.number)
            ):
                atol = float(tolerance[0]) if np.size(tolerance) >= 1 else 1e-9
                rtol = float(tolerance[1]) if np.size(tolerance) >= 2 else atol
                diff = abs(float(ref_item) - float(cand))
                scale = max(abs(float(ref_item)), abs(float(cand)))
                is_eq = diff <= max(atol, rtol * scale)
            elif isinstance(ref_item, np.ndarray) and isinstance(cand, np.ndarray):
                if ref_item.shape != cand.shape:
                    is_eq = False
                else:
                    is_eq, _ = compare_values(ref_item, cand, tol=tolerance)
            else:
                is_eq, _ = compare_values(ref_item, cand, tol=tolerance)
            if is_eq:
                visited[j] = True
                cur_group.append(j + 1)
        groups.append(np.array(cur_group, dtype=int))

    return groups


def crystal_pars_correct(
    u: Any,
    v: Any,
    alatt0: Any,
    angdeg0: Any,
    omega0: float,
    dpsi0: float,
    gl0: float,
    gs0: float,
    rlu_corr: Any,
    *args: Any,
    nargout: Optional[int] = None,
) -> Any:
    """Return new lattice parameters and orientation angles corresponding to ``rlu_corr``.

    Translated from ``horace_core/lattice_functions/crystal_pars_correct.m``.
    All angles ``omega0, dpsi0, gl0, gs0`` and returned ``dpsi, gl, gs`` are in degrees.
    """
    nout = _infer_nargout(default=5, explicit_nargout=nargout)

    u_vec = np.asarray(u, dtype=float).ravel()
    v_vec = np.asarray(v, dtype=float).ravel()
    alatt0_vec = np.asarray(alatt0, dtype=float).ravel()
    angdeg0_vec = np.asarray(angdeg0, dtype=float).ravel()
    rlu_corr_mat = np.asarray(rlu_corr, dtype=float)[:3, :3]

    b0 = bmatrix(alatt0_vec, angdeg0_vec, nargout=1)
    ub0 = ubmatrix(u_vec, v_vec, b0, nargout=1)

    if len(args) == 0:
        ub = np.linalg.solve(rlu_corr_mat.T, ub0.T).T  # ub0 / rlu_corr
        ginv = ub.T @ ub  # (b^-1)^T * (b^-1)
        g = np.linalg.inv(ginv)
        abc = np.sqrt(np.diag(g))
        cosang = (g / np.outer(abc, abc) + (g / np.outer(abc, abc)).T) / 2.0
        angdeg = np.rad2deg(
            np.arccos(np.array([cosang[1, 2], cosang[0, 2], cosang[0, 1]], dtype=float))
        )
        alatt = (2.0 * math.pi) * abc
    elif len(args) == 2:
        alatt = np.asarray(args[0], dtype=float).ravel()
        angdeg = np.asarray(args[1], dtype=float).ravel()
    else:
        raise ValueError(
            "HORACE:crystal_pars_correct:invalid_argument: Check number of input arguments"
        )

    deg_to_rad = math.pi / 180.0
    c_omega = math.cos(deg_to_rad * float(omega0))
    s_omega = math.sin(deg_to_rad * float(omega0))
    c_dpsi0 = math.cos(deg_to_rad * float(dpsi0))
    s_dpsi0 = math.sin(deg_to_rad * float(dpsi0))
    c_gl0 = math.cos(deg_to_rad * float(gl0))
    s_gl0 = math.sin(deg_to_rad * float(gl0))
    c_gs0 = math.cos(deg_to_rad * float(gs0))
    s_gs0 = math.sin(deg_to_rad * float(gs0))

    R0 = (
        np.array([[c_omega, s_omega, 0.0], [-s_omega, c_omega, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[c_dpsi0, -s_dpsi0, 0.0], [s_dpsi0, c_dpsi0, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[c_gl0, 0.0, s_gl0], [0.0, 1.0, 0.0], [-s_gl0, 0.0, c_gl0]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, c_gs0, -s_gs0], [0.0, s_gs0, c_gs0]])
        @ np.array([[c_omega, -s_omega, 0.0], [s_omega, c_omega, 0.0], [0.0, 0.0, 1.0]])
    )

    ub = np.linalg.solve(rlu_corr_mat.T, ub0.T).T
    b = bmatrix(alatt, angdeg, nargout=1)
    umat = np.linalg.solve(b.T, ub.T).T  # ub / b
    u0 = np.linalg.solve(b0.T, ub0.T).T  # ub0 / b0

    R = (R0 @ u0) @ umat.T
    Rp = (
        np.array([[c_omega, -s_omega, 0.0], [s_omega, c_omega, 0.0], [0.0, 0.0, 1.0]])
        @ R
        @ np.array([[c_omega, s_omega, 0.0], [-s_omega, c_omega, 0.0], [0.0, 0.0, 1.0]])
    )

    if abs(Rp[2, 0]) != 1.0:
        s_gl = -Rp[2, 0]
        c_gl = math.sqrt(max(0.0, 1.0 - s_gl**2))
        gl = math.asin(np.clip(s_gl, -1.0, 1.0)) / deg_to_rad
        gs = math.atan2(Rp[2, 1] / c_gl, Rp[2, 2] / c_gl) / deg_to_rad
        dpsi = math.atan2(Rp[1, 0] / c_gl, Rp[0, 0] / c_gl) / deg_to_rad
    else:
        raise RuntimeError(
            "HORACE:crystal_pars_correct:runtime_error: Cannot invert rotation matrix (|gl| == 90 deg)"
        )

    if nout <= 5:
        return alatt, angdeg, dpsi, gl, gs

    del_alatt = alatt - alatt0_vec
    del_angdeg = angdeg - angdeg0_vec
    if nout == 7:
        return alatt, angdeg, dpsi, gl, gs, del_alatt, del_angdeg

    u_to_rlu = np.linalg.solve(b, umat.T)
    ulen = np.ones(3, dtype=float)
    return alatt, angdeg, dpsi, gl, gs, del_alatt, del_angdeg, u_to_rlu, ulen


class ExpDataList(list):
    """List subclass for ``Experiment.expdata`` providing MATLAB array methods on ``IX_experiment``."""

    def get_run_ids(self) -> np.ndarray:
        """Return 1D numpy array of ``run_id`` values across all ``IX_experiment`` objects."""
        return np.array([float(x.run_id) for x in self], dtype=float)

    def set_run_ids(self, ids: Sequence[float]) -> "ExpDataList":
        """Set ``run_id`` values across all ``IX_experiment`` objects."""
        ids_arr = np.asarray(ids, dtype=float).ravel()
        if ids_arr.size != len(self):
            raise ValueError(
                "HERBERT:IX_experiment:invalid_argument: Number of run_ids must match number of experiments."
            )
        for item, rid in zip(self, ids_arr):
            item.run_id = float(rid)
        return self

    def get_efix(self, extract_unique: bool = False) -> np.ndarray:
        """Return array of ``efix`` across all ``IX_experiment`` objects."""
        efix_arr = np.array([float(x.efix) for x in self], dtype=float)
        if extract_unique:
            _, idx = np.unique(efix_arr, return_index=True)
            return efix_arr[np.sort(idx)]
        return efix_arr

    def get_emode(self) -> np.ndarray:
        """Return array of ``emode`` across all ``IX_experiment`` objects."""
        return np.array([int(x.emode) for x in self], dtype=int)

    def get_en_transfer(
        self,
        bin_centre: bool = False,
        get_lidx: bool = False,
        nargout: Optional[int] = None,
    ) -> Any:
        """Return energy transfer array(s) across all ``IX_experiment`` objects.

        Translated from ``@IX_experiment/get_en_transfer.m``.
        """
        nout = _infer_nargout(default=2 if get_lidx else 1, explicit_nargout=nargout)
        en_all = [x.get_en(bin_centers=bin_centre) for x in self]
        if len(en_all) == 0:
            if nout >= 2 or get_lidx:
                return [], []
            return np.empty(0, dtype=float)
        if len(en_all) == 1 and not get_lidx and nout == 1:
            return en_all[0]

        idx_groups = calc_idx_for_eq_to_tol(en_all)
        unique_en = [en_all[int(grp[0]) - 1] for grp in idx_groups]

        if nout >= 2:
            return unique_en, idx_groups
        if get_lidx:
            if len(unique_en) == 1:
                return unique_en[0]
            return unique_en
        return en_all

    def check_SameBins(self, other: Any = None, tol: float = 1e-8) -> bool:
        """Return True if all ``IX_experiment`` objects have identical energy bins within ``tol``."""
        if len(self) == 0:
            return True
        target = self if other is None else other
        return self[0].check_SameBins(target, tol=tol)

    def combine(
        self,
        exp_cellarray: Sequence[Any],
        allow_equal_headers: bool = False,
        keep_runid: bool = True,
        this_runid_map: Optional[fast_map] = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Combine this ``ExpDataList`` with a list of ``ExpDataList`` / ``list[IX_experiment]``.

        Translated from ``@IX_experiment/private/combine_.m``.
        """
        nout = _infer_nargout(default=2, explicit_nargout=nargout)

        all_groups: List[List[IX_experiment]] = [list(self)]
        for item in exp_cellarray:
            if isinstance(item, IX_experiment):
                all_groups.append([item])
            elif isinstance(item, (list, tuple, np.ndarray)):
                all_groups.append(list(item))
            else:
                raise ValueError(
                    f"HERBERT:IX_experiment:invalid_argument: Expected sequence of IX_experiment, got {type(item).__name__}."
                )

        if keep_runid:
            if this_runid_map is None:
                keys = [float(x.run_id) for x in self]
                vals = list(range(1, len(self) + 1))
                runid_map_out = fast_map(keys, vals)
            else:
                runid_map_out = copy.deepcopy(this_runid_map)
        else:
            keys = list(range(1, len(self) + 1))
            vals = list(range(1, len(self) + 1))
            runid_map_out = fast_map(keys, vals)

        combined = ExpDataList([copy.deepcopy(x) for x in self])
        if not keep_runid:
            for idx_0, exp_item in enumerate(combined):
                exp_item.run_id = float(idx_0 + 1)

        if not allow_equal_headers:
            hashes_seen = set()
            for x in combined:
                _, h, _ = build_hash(x)
                if h in hashes_seen:
                    raise ValueError(
                        "HORACE:Experiment:invalid_argument: Equal experiments within single sqw are prohibited."
                    )
                hashes_seen.add(h)
        else:
            hashes_seen = set()

        nsqw_change_runid: List[Optional[fast_map]] = []
        tot_nruns = len(combined)

        for group in all_groups[1:]:
            n_cur = len(group)
            if not allow_equal_headers:
                for x in group:
                    _, h, _ = build_hash(x)
                    if h in hashes_seen:
                        raise ValueError(
                            "HORACE:Experiment:invalid_argument: Equal experiments in combined objects are prohibited when allow_equal_headers is False."
                        )
                    hashes_seen.add(h)

            cur_ids = np.array([float(x.run_id) for x in group], dtype=float)
            existing_keys = set(float(k) for k in runid_map_out.keys)
            change_runid = (not keep_runid) or any(
                float(cid) in existing_keys or np.isnan(cid) for cid in cur_ids
            )

            if change_runid:
                if keep_runid:
                    max_id = int(max(existing_keys)) if existing_keys else 0
                    new_ids = np.arange(max_id + 1, max_id + n_cur + 1, dtype=int)
                else:
                    new_ids = np.arange(tot_nruns + 1, tot_nruns + n_cur + 1, dtype=int)
                nsqw_change_runid.append(fast_map(cur_ids, new_ids))
                for exp_item, new_id in zip(group, new_ids):
                    ec = copy.deepcopy(exp_item)
                    ec.run_id = float(new_id)
                    combined.append(ec)
                runid_map_out = runid_map_out.add(
                    new_ids, np.arange(tot_nruns + 1, tot_nruns + n_cur + 1, dtype=int)
                )
            else:
                nsqw_change_runid.append(None)
                for exp_item in group:
                    combined.append(copy.deepcopy(exp_item))
                runid_map_out = runid_map_out.add(
                    cur_ids, np.arange(tot_nruns + 1, tot_nruns + n_cur + 1, dtype=int)
                )

            tot_nruns += n_cur

        if nout == 1:
            return combined
        if nout == 2:
            return combined, nsqw_change_runid
        return combined, nsqw_change_runid, runid_map_out


@register_serializable
class Experiment(Serializable):
    """Experiment container holding per-run metadata, instruments, samples, and detector arrays.

    Translated from ``horace_core/sqw/@Experiment/Experiment.m``.
    """

    def class_version(self) -> int:
        return 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._instruments: unique_references_container = unique_references_container("IX_inst")
        self._detectors: unique_references_container = unique_references_container(
            "IX_detector_array"
        )
        self._samples: unique_references_container = unique_references_container("IX_samp")
        self._expdata: ExpDataList = ExpDataList()
        self._runid_map: fast_map = fast_map()
        self._runid_recalculated: bool = False

        self._samples_set: bool = False
        self._instruments_set: bool = False
        self._detectors_set: bool = False
        self._expdata_set: bool = False
        self._old_lattice_holder: Optional[unique_references_container] = None

        if len(args) == 0 and len(kwargs) == 0:
            return
        self.init(*args, **kwargs)

    def init(self, *args: Any, **kwargs: Any) -> "Experiment":
        """Initialize or re-initialize the Experiment container (port of ``init_.m``)."""
        if len(args) == 0 and len(kwargs) == 0:
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg0 = args[0]
            if isinstance(arg0, Experiment):
                self._copy_from(arg0)
                return self
            if isinstance(arg0, IX_experiment):
                self.expdata = [arg0]
                return self
            if isinstance(arg0, dict):
                if "serial_name" in arg0:
                    loaded = Serializable.loadobj(arg0)
                    if isinstance(loaded, Experiment):
                        self._copy_from(loaded)
                        return self
                if (
                    "samples" in arg0
                    and "instruments" in arg0
                    and "expdata" in arg0
                ):
                    loaded = Experiment.from_struct(arg0)
                    if isinstance(loaded, Experiment):
                        self._copy_from(loaded)
                        return self
                built = Experiment.build_from_binfile_headers(arg0)
                self._copy_from(built)
                return self
            if isinstance(arg0, (list, tuple, np.ndarray)):
                seq = list(arg0)
                if len(seq) == 0:
                    return self
                if all(isinstance(x, IX_experiment) for x in seq):
                    self.expdata = seq
                    return self
                if all(isinstance(x, dict) for x in seq):
                    built = Experiment.build_from_binfile_headers(seq)
                    self._copy_from(built)
                    return self
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Unrecognised single argument of type {type(arg0).__name__}"
            )

        pos_args = list(args)
        if len(pos_args) == 3 and len(kwargs) == 0:
            first = pos_args[0]
            is_det = isinstance(first, IX_detector_array) or (
                isinstance(first, unique_references_container)
                and first.stored_baseclass == "IX_detector_array"
            ) or (
                isinstance(first, (list, tuple, np.ndarray))
                and len(first) > 0
                and isinstance(first[0], IX_detector_array)
            )
            if is_det:
                self._do_check_combo_arg = False
                try:
                    self.detector_arrays = pos_args[0]
                    self.instruments = pos_args[1]
                    self.samples = pos_args[2]
                finally:
                    self._do_check_combo_arg = True
                self.check_combo_arg()
                return self
            else:
                pos_args = [None, pos_args[0], pos_args[1], pos_args[2]]

        flds = self.construction_fields()
        _, remains = self.set_positional_and_key_val_arguments(
            flds, False, *pos_args, **kwargs
        )
        if remains:
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Extra unknown parameters provided: {remains}"
            )
        return self

    def _copy_from(self, other: "Experiment") -> None:
        self._instruments = copy.deepcopy(other._instruments)
        self._detectors = copy.deepcopy(other._detectors)
        self._samples = copy.deepcopy(other._samples)
        self._expdata = ExpDataList([copy.deepcopy(x) for x in other._expdata])
        self._runid_map = copy.deepcopy(other._runid_map)
        self._runid_recalculated = bool(other._runid_recalculated)
        self._samples_set = bool(other._samples_set)
        self._instruments_set = bool(other._instruments_set)
        self._detectors_set = bool(other._detectors_set)
        self._expdata_set = bool(other._expdata_set)
        self._old_lattice_holder = (
            None
            if other._old_lattice_holder is None
            else copy.deepcopy(other._old_lattice_holder)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_runs(self) -> int:
        if self._expdata_set:
            return len(self._expdata)
        if self._samples_set:
            return self._samples.n_runs
        if self._instruments_set:
            return self._instruments.n_runs
        if self._detectors_set:
            return self._detectors.n_runs
        return 0

    @property
    def detector_arrays(self) -> unique_references_container:
        return self._detectors

    @detector_arrays.setter
    def detector_arrays(self, val: Any) -> None:
        self._detectors, self._detectors_set = self._check_and_set_container(
            val, "IX_detector_array", IX_detector_array, None
        )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def instruments(self) -> unique_references_container:
        return self._instruments

    @instruments.setter
    def instruments(self, val: Any) -> None:
        self._instruments, self._instruments_set = self._check_and_set_container(
            val, "IX_inst", IX_inst, IX_null_inst
        )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def samples(self) -> unique_references_container:
        return self._samples

    @samples.setter
    def samples(self, val: Any) -> None:
        self._check_lattice_defined()
        self._samples, self._samples_set = self._check_and_set_container(
            val, "IX_samp", IX_samp, IX_null_sample
        )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def expdata(self) -> ExpDataList:
        return self._expdata

    @expdata.setter
    def expdata(self, val: Any) -> None:
        self._expdata, self._runid_map, self._runid_recalculated, self._expdata_set = (
            self._check_expdata_and_form_runid_map(val)
        )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def runid_map(self) -> fast_map:
        return self._runid_map

    @runid_map.setter
    def runid_map(self, val: Any) -> None:
        if not isinstance(val, fast_map):
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: runid_map must be a fast_map instance, got {type(val).__name__}"
            )
        keys = np.asarray(val.keys, dtype=float).ravel()
        values = np.asarray(val.values, dtype=int).ravel()
        if keys.size != len(self._expdata):
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Number of elements in runid_map ({keys.size}) differs from n_runs ({len(self._expdata)})"
            )
        if len(self._expdata) > 0:
            min_v, max_v = int(np.min(values)), int(np.max(values))
            if min_v != 1 or max_v != len(self._expdata) or np.unique(values).size != len(self._expdata):
                raise ValueError(
                    "HORACE:Experiment:invalid_argument: runid_map values must be a permutation of 1..n_runs"
                )
            for k, v in zip(keys, values):
                self._expdata[int(v) - 1].run_id = float(k)
        self._runid_map = copy.deepcopy(val)
        self._runid_recalculated = False
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def runid_recalculated(self) -> bool:
        return self._runid_recalculated

    @runid_recalculated.setter
    def runid_recalculated(self, val: bool) -> None:
        self._runid_recalculated = bool(val)

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------
    def _check_lattice_defined(self) -> bool:
        """Port of ``check_lattice_defined_.m``."""
        if not self._samples_set or self._samples.n_runs == 0:
            return False
        u_samp = self._samples.get_unique_objects_and_indices()[0]
        lat_def = [s.alatt.size > 0 for s in u_samp]
        ang_def = [s.angdeg.size > 0 for s in u_samp]
        is_def = any(lat_def) and any(ang_def)
        if is_def:
            self._old_lattice_holder = copy.deepcopy(self._samples)
        return is_def

    @staticmethod
    def _check_and_set_container(
        val: Any,
        base_class_name: str,
        base_type: type,
        null_cls: Optional[type],
    ) -> Tuple[unique_references_container, bool]:
        """Port of ``check_and_set_type_.m`` and ``check_sample_or_inst_array_and_return_std_form_.m``."""
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            return unique_references_container(base_class_name), False

        if isinstance(val, unique_references_container):
            if val.baseclass != base_class_name:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Container baseclass '{val.baseclass}' does not match '{base_class_name}'"
                )
            c = copy.deepcopy(val)
            return c, c.n_runs > 0

        if isinstance(val, unique_objects_container):
            if val.baseclass != base_class_name:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Container baseclass '{val.baseclass}' does not match '{base_class_name}'"
                )
            urc = unique_references_container(base_class_name)
            for i in range(val.n_runs):
                urc.add(val[i])
            return urc, urc.n_runs > 0

        if isinstance(val, base_type):
            urc = unique_references_container(base_class_name)
            urc.add(copy.deepcopy(val))
            return urc, True

        if isinstance(val, (list, tuple, np.ndarray)):
            urc = unique_references_container(base_class_name)
            for item in val:
                if item is None or (
                    isinstance(item, (list, tuple, np.ndarray, dict)) and len(item) == 0
                ):
                    if null_cls is None:
                        raise ValueError(
                            f"HORACE:Experiment:invalid_argument: Empty element not allowed for {base_class_name}"
                        )
                    urc.add(null_cls())
                elif isinstance(item, base_type):
                    urc.add(copy.deepcopy(item))
                else:
                    raise ValueError(
                        f"HORACE:Experiment:invalid_argument: Element of type {type(item).__name__} is not an instance of {base_class_name}"
                    )
            return urc, urc.n_runs > 0

        raise ValueError(
            f"HORACE:Experiment:invalid_argument: Cannot set {base_class_name} container from {type(val).__name__}"
        )

    @staticmethod
    def _check_expdata_and_form_runid_map(
        val: Any,
    ) -> Tuple[ExpDataList, fast_map, bool, bool]:
        """Port of ``check_expdata_and_FormRunidMap_.m``."""
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            return ExpDataList(), fast_map(), False, False

        if isinstance(val, IX_experiment):
            items = [copy.deepcopy(val)]
        elif isinstance(val, (list, tuple, np.ndarray)):
            items = []
            for x in val:
                if not isinstance(x, IX_experiment):
                    raise ValueError(
                        f"HORACE:Experiment:invalid_argument: expdata elements must be IX_experiment, got {type(x).__name__}"
                    )
                items.append(copy.deepcopy(x))
        else:
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: expdata must be IX_experiment or sequence of IX_experiment, got {type(val).__name__}"
            )

        exp_list = ExpDataList(items)
        n_runs = len(exp_list)
        if n_runs == 0:
            return ExpDataList(), fast_map(), False, False

        run_ids = np.array([float(x.run_id) for x in exp_list], dtype=float)
        nan_mask = np.isnan(run_ids)
        recalculated = False

        if np.all(nan_mask):
            run_ids = np.arange(1, n_runs + 1, dtype=float)
            recalculated = True
        else:
            valid_ids = run_ids[~nan_mask]
            unique_ids = np.unique(valid_ids)
            if np.any(nan_mask) or unique_ids.size != n_runs:
                if np.any(nan_mask):
                    warnings.warn(
                        "HORACE:Experiment:invalid_argument: Some run_ids were NaN; resetting all run_ids to 1..n_runs",
                        UserWarning,
                        stacklevel=3,
                    )
                run_ids = np.arange(1, n_runs + 1, dtype=float)
                recalculated = True

        if recalculated:
            for exp_obj, rid in zip(exp_list, run_ids):
                exp_obj.run_id = float(rid)

        rmap = fast_map(run_ids, np.arange(1, n_runs + 1, dtype=int))
        return exp_list, rmap, recalculated, True

    def check_combo_arg(self) -> "Experiment":
        """Validate and synchronize component containers across runs (port of ``check_combo_arg_.m``)."""
        if self._expdata_set:
            n_runs = len(self._expdata)
        else:
            ni = self._instruments.n_runs if self._instruments_set else 0
            ns = self._samples.n_runs if self._samples_set else 0
            nd = self._detectors.n_runs if self._detectors_set else 0
            if ni > 1 and ns > 1 and ni != ns:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Number of instruments ({ni}) and samples ({ns}) do not match"
                )
            n_runs = max(ni, ns, nd)

        if n_runs == 0:
            return self

        # Synchronize instruments
        if not self._instruments_set:
            urc_inst = unique_references_container("IX_inst")
            urc_inst.add(IX_null_inst())
            self._instruments = urc_inst.replicate_runs(n_runs)
        elif self._instruments.n_runs != n_runs:
            if self._instruments.n_runs == 1:
                self._instruments = self._instruments.replicate_runs(n_runs)
            else:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Number of instruments ({self._instruments.n_runs}) does not match n_runs ({n_runs})"
                )

        # Synchronize detector_arrays
        if self._detectors_set and self._detectors.n_runs != n_runs:
            if self._detectors.n_runs == 1:
                self._detectors = self._detectors.replicate_runs(n_runs)
            else:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Number of detector_arrays ({self._detectors.n_runs}) does not match n_runs ({n_runs})"
                )

        # Synchronize samples
        if not self._samples_set:
            urc_samp = unique_references_container("IX_samp")
            urc_samp.add(IX_null_sample())
            self._samples = urc_samp.replicate_runs(n_runs)
        elif self._samples.n_runs != n_runs:
            if self._samples.n_runs == 1:
                self._samples = self._samples.replicate_runs(n_runs)
            else:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Number of samples ({self._samples.n_runs}) does not match n_runs ({n_runs})"
                )

        if self._samples_set:
            u_samp = self._samples.get_unique_objects_and_indices()[0]
            defined = [s.alatt.size > 0 and s.angdeg.size > 0 for s in u_samp]
            if not all(defined):
                if self._old_lattice_holder is not None:
                    if (
                        self._old_lattice_holder.n_runs == self._samples.n_runs
                        or self._old_lattice_holder.n_unique == 1
                    ):
                        if self._old_lattice_holder.n_runs != n_runs:
                            self._old_lattice_holder = (
                                self._old_lattice_holder.replicate_runs(n_runs)
                            )
                        new_samples = unique_references_container("IX_samp")
                        for i in range(n_runs):
                            cs = copy.deepcopy(self._samples[i])
                            if cs.alatt.size == 0:
                                cs.alatt = self._old_lattice_holder[i].alatt
                            if cs.angdeg.size == 0:
                                cs.angdeg = self._old_lattice_holder[i].angdeg
                            new_samples.add(cs)
                        self._samples = new_samples
                    else:
                        warnings.warn(
                            "HORACE:Experiment:lattice_undefined: Samples in experiment are defined but their lattice is undefined and the old samples define different lattice",
                            UserWarning,
                            stacklevel=2,
                        )
                    self._old_lattice_holder = None
                else:
                    is_null = [isinstance(s, IX_null_sample) for s in u_samp]
                    if not all(is_null):
                        warnings.warn(
                            "HORACE:Experiment:lattice_undefined: Samples in experiment are defined but their lattice is undefined",
                            UserWarning,
                            stacklevel=2,
                        )
            else:
                self._old_lattice_holder = None

        # Synchronize expdata if not set yet
        if not self._expdata_set and n_runs > 0:
            default_exps = [IX_experiment() for _ in range(n_runs)]
            self._expdata, self._runid_map, self._runid_recalculated, _ = (
                self._check_expdata_and_form_runid_map(default_exps)
            )
            self._expdata_set = False

        if self._expdata_set and len(self._runid_map.keys) != n_runs:
            raise ValueError(
                "HORACE:Experiment:invalid_argument: runid_map size does not match expdata length"
            )

        return self

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------
    def get_efix(
        self,
        compact: Union[bool, str] = False,
        nargout: Optional[int] = None,
    ) -> Any:
        """Return fixed/incident energies across all runs (or as a ``compact_array``).

        Translated from ``@Experiment/get_efix.m``.
        """
        nout = _infer_nargout(default=1, explicit_nargout=nargout)
        is_compact = (
            nout > 1
            or (isinstance(compact, str) and compact == "-compact")
            or (isinstance(compact, bool) and compact)
        )
        efix_list = [float(x.efix) for x in self._expdata]
        if is_compact:
            idx = calc_idx_for_eq_to_tol(efix_list, (1e-8, 1e-8))
            uval = [efix_list[int(grp[0]) - 1] for grp in idx]
            ca = compact_array(idx, uval)
            if nout == 1:
                return ca
            if nout == 2:
                return ca, idx
            return ca, idx, self
        if len(efix_list) == 1:
            return float(efix_list[0])
        return np.array(efix_list, dtype=float)

    def get_emode(self) -> Union[int, np.ndarray]:
        """Return spectrometer energy mode(s) ``emode``."""
        emodes = [int(x.emode) for x in self._expdata]
        if len(emodes) == 1:
            return int(emodes[0])
        return np.array(emodes, dtype=int)

    def set_efix_emode(
        self, efix: Any, emode: Union[int, str] = "-keep_emode"
    ) -> "Experiment":
        """Set ``efix`` (and optionally ``emode``) across all runs and instruments.

        Translated from ``@Experiment/Experiment.m:set_efix_emode``.
        """
        keep_emode = isinstance(emode, str) and emode == "-keep_emode"
        if not keep_emode:
            emode_int = int(emode)
            if emode_int not in (0, 1, 2):
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: emode must be 0, 1, or 2, got {emode}"
                )
        else:
            emode_int = -1

        n = self.n_runs
        efix_arr = np.asarray(efix, dtype=float).ravel()
        if efix_arr.size == 1:
            efix_arr = np.repeat(efix_arr, n)
        elif efix_arr.size != n:
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Number of efix values ({efix_arr.size}) does not match n_runs ({n})"
            )

        if np.any(np.isnan(efix_arr)) or np.any(efix_arr < 0):
            raise ValueError(
                "HORACE:Experiment:invalid_argument: efix values must be finite and non-negative"
            )

        for i in range(n):
            em_i = self._expdata[i].emode if keep_emode else emode_int
            if efix_arr[i] == 0.0 and em_i != 0:
                raise ValueError(
                    "HORACE:Experiment:invalid_argument: Zero efix is only allowed for elastic mode (emode == 0)"
                )
            if not keep_emode:
                self._expdata[i].emode = emode_int
            self._expdata[i].efix = float(efix_arr[i])

        # Also update instrument energy if instruments have an energy property
        if self._instruments_set and self._instruments.n_runs == n:
            new_insts = unique_references_container("IX_inst")
            for i in range(n):
                inst = copy.deepcopy(self._instruments[i])
                if not isinstance(inst, IX_null_inst) and hasattr(inst, "energy"):
                    inst.energy = float(efix_arr[i])
                new_insts.add(inst)
            self._instruments = new_insts

        return self

    def get_en_transfer(
        self,
        bin_centre: bool = False,
        compact: bool = False,
    ) -> Any:
        """Return energy transfer array(s) or ``compact_array`` across all runs.

        Translated from ``@Experiment/get_en_transfer.m``.
        """
        if self.n_runs == 0:
            return []
        if compact:
            en_unique, u_idx = self._expdata.get_en_transfer(
                bin_centre, True, nargout=2
            )
            return compact_array(u_idx, en_unique)
        return self._expdata.get_en_transfer(bin_centre, True, nargout=1)

    def is_same_ebins(self) -> bool:
        """Return True if all runs have identical energy bins."""
        return self._expdata.check_SameBins()

    def get_inst_class(self, nargout: Optional[int] = None) -> Any:
        """Retrieve the main instrument class type from the instruments container.

        Translated from ``@Experiment/Experiment.m:get_inst_class``.
        """
        nout = _infer_nargout(default=2, explicit_nargout=nargout)
        u_inst = self._instruments.unique_objects
        non_null = [inst for inst in u_inst if not isinstance(inst, IX_null_inst)]
        all_inst = len(non_null) == len(u_inst) and len(u_inst) > 0

        if len(non_null) == 0:
            inst_class = IX_null_inst()
        else:
            classes = {type(inst) for inst in non_null}
            if len(classes) != 1:
                raise RuntimeError(
                    "HORACE:Experiment:runtime_error: Different non-empty instrument classes present in Experiment"
                )
            inst_class = copy.deepcopy(non_null[0])

        if nout == 1:
            return inst_class
        return inst_class, all_inst

    def get_mod_pulse(self, nargout: Optional[int] = None) -> Tuple[str, np.ndarray, bool]:
        """Get moderator pulse model names and parameters across all runs.

        Translated from ``@Experiment/Experiment.m:get_mod_pulse``.
        """
        n = self.n_runs
        pm_list: List[str] = []
        pm_par_rows: List[np.ndarray] = []
        present = True

        for i in range(n):
            inst = self._instruments[i]
            try:
                mod = inst.moderator
                pm_list.append(mod.pulse_model)
                pp_i = np.asarray(mod.pp, dtype=float).ravel()
                pm_par_rows.append(pp_i)
            except Exception:
                present = False
                return "", np.empty((0, 0), dtype=float), False

        unique_models = set(pm_list)
        if len(unique_models) > 1:
            warnings.warn(
                "HORACE:Experiment:invalid_argument: Different moderator pulse models present across runs",
                UserWarning,
                stacklevel=2,
            )
            return "", np.empty((0, 0), dtype=float), False

        pulsemodel = pm_list[0] if pm_list else ""
        pm_par = np.vstack(pm_par_rows) if pm_par_rows else np.empty((0, 0), dtype=float)
        return pulsemodel, pm_par, present

    def set_mod_pulse(self, pulse_model: str, pm_par: Any) -> "Experiment":
        """Set moderator pulse model and parameters across all instruments in Experiment.

        Translated from ``@Experiment/set_mod_pulse.m``.
        """
        if not isinstance(pulse_model, str) or not pulse_model.strip():
            raise ValueError(
                "HORACE:Experiment:invalid_argument: pulse_model must be a non-empty string"
            )
        n = self.n_runs
        par_arr = np.asarray(pm_par, dtype=float)
        if par_arr.ndim == 1:
            par_arr = np.tile(par_arr.reshape(1, -1), (n, 1))
        elif par_arr.ndim == 2:
            if par_arr.shape[0] == 1 and n > 1:
                par_arr = np.tile(par_arr, (n, 1))
            elif par_arr.shape[0] != n:
                if par_arr.shape[1] == n and par_arr.shape[0] != n:
                    par_arr = par_arr.T
                else:
                    raise ValueError(
                        f"HORACE:Experiment:invalid_argument: Number of parameter rows ({par_arr.shape[0]}) must be 1 or n_runs ({n})"
                    )

        new_insts = unique_references_container("IX_inst")
        for i in range(n):
            inst = copy.deepcopy(self._instruments[i])
            if isinstance(inst, IX_null_inst):
                raise RuntimeError(
                    "HORACE:Experiment:runtime_error: Cannot set moderator pulse on empty (IX_null_inst) instrument"
                )
            inst = inst.set_mod_pulse(pulse_model, par_arr[i, :])
            new_insts.add(inst)
        self._instruments = new_insts
        return self

    def set_instrument(
        self, instrument: Any, *args: Any, **kwargs: Any
    ) -> "Experiment":
        """Set instrument across all runs (either an ``IX_inst`` array/object or a function handle).

        Translated from ``@Experiment/set_instrument.m``.
        """
        n = self.n_runs
        if isinstance(instrument, IX_inst):
            if n <= 1:
                self.instruments = instrument
            else:
                urc = unique_references_container("IX_inst")
                urc.add(copy.deepcopy(instrument))
                self.instruments = urc.replicate_runs(n)
            return self

        if isinstance(instrument, (unique_references_container, unique_objects_container, list, tuple, np.ndarray)):
            self.instruments = instrument
            return self

        if callable(instrument):
            efix_present = False
            clean_args: List[Any] = []
            i = 0
            while i < len(args):
                a = args[i]
                if isinstance(a, str) and a.lower() in ("-efix", "efix"):
                    efix_present = True
                    i += 1
                else:
                    clean_args.append(a)
                    i += 1

            # Evaluate each arg for each run
            parsed_args_per_run: List[List[Any]] = [[] for _ in range(n)]
            if efix_present:
                for r in range(n):
                    parsed_args_per_run[r].append(float(self._expdata[r].efix))

            for arg in clean_args:
                arr = np.asarray(arg)
                if arr.ndim == 0 or arr.shape[0] == 1:
                    val = arg if arr.ndim == 0 else (arr[0] if arr.ndim == 1 else arr[0, :])
                    for r in range(n):
                        parsed_args_per_run[r].append(val)
                elif arr.shape[0] == n:
                    for r in range(n):
                        parsed_args_per_run[r].append(arr[r] if arr.ndim == 1 else arr[r, :])
                elif arr.ndim == 1 and arr.size != n:
                    for r in range(n):
                        parsed_args_per_run[r].append(arr)
                else:
                    raise ValueError(
                        f"HORACE:Experiment:invalid_argument: Instrument function argument shape {arr.shape} incompatible with n_runs={n}"
                    )

            urc = unique_references_container("IX_inst")
            if n > 0 and len(clean_args) == 0 and not efix_present:
                inst_obj = instrument(**kwargs)
                urc.add(inst_obj)
                self.instruments = urc.replicate_runs(n)
            else:
                for r in range(n):
                    inst_obj = instrument(*parsed_args_per_run[r], **kwargs)
                    urc.add(inst_obj)
                self.instruments = urc
            return self

        raise ValueError(
            f"HORACE:Experiment:invalid_argument: Unrecognised instrument specification of type {type(instrument).__name__}"
        )

    def calc_efix_avrgs(self, nargout: Optional[int] = None) -> Tuple[float, float, bool]:
        """Calculate average fixed energy ``efix`` and ``emode`` across runs.

        Translated from ``@Experiment/Experiment.m:calc_efix_avrgs``.
        """
        if self.n_runs == 0:
            return float("nan"), float("nan"), False
        efix_arr = np.array([float(x.efix) for x in self._expdata], dtype=float)
        emode_arr = np.array([int(x.emode) for x in self._expdata], dtype=int)

        u_efix = calc_idx_for_eq_to_tol(efix_arr)
        u_emode = np.unique(emode_arr)
        ave_efix = float(np.mean(efix_arr))
        ave_emode = float(np.mean(emode_arr))

        present = True
        if len(u_efix) > 1:
            warnings.warn(
                "HORACE:Experiment:invalid_argument: Different efix values across runs; using mean efix",
                UserWarning,
                stacklevel=2,
            )
            present = False
        if u_emode.size > 1:
            warnings.warn(
                "HORACE:Experiment:invalid_argument: Different emode values across runs",
                UserWarning,
                stacklevel=2,
            )
            present = False
        return ave_efix, ave_emode, present

    def calc_mod_pulse_avrgs(
        self, nargout: Optional[int] = None
    ) -> Tuple[np.ndarray, str, bool]:
        """Calculate average moderator pulse parameters across runs.

        Translated from ``@Experiment/Experiment.m:calc_mod_pulse_avrgs``.
        """
        pulsemodel, pm_par, present = self.get_mod_pulse()
        if present and pm_par.size > 0:
            pm_par_avrg = np.mean(pm_par, axis=0)
        else:
            pm_par_avrg = np.empty(0, dtype=float)
        return pm_par_avrg, pulsemodel, present

    def header_average(self, nargout: Optional[int] = None) -> Any:
        """Return the first run's legacy header dict (and average ebins Flag).

        Translated from ``@Experiment/Experiment.m:header_average``.
        """
        nout = _infer_nargout(default=1, explicit_nargout=nargout)
        if self.n_runs == 0:
            if nout >= 2:
                return {}, True
            return {}
        hdr_av = self.convert_to_old_headers(header_num=1)[0]
        if self._expdata[0]._u_to_rlu is not None:
            hdr_av["u_to_rlu"] = self._expdata[0].u_to_rlu
        else:
            hdr_av["u_to_rlu"] = np.zeros(0, dtype=float)
        if nout >= 2:
            ebins_are_same = self.is_same_ebins()
            return hdr_av, ebins_are_same
        return hdr_av

    def get_UXZ(self, get_U: bool = True) -> np.ndarray:
        """Calculate rotation matrices transforming momentum from spectrometer to crystal Cartesian frame.

        Translated from ``@Experiment/Experiment.m:get_UXZ``.
        """
        n = self.n_runs
        u_mat = np.zeros((3, 3, n), dtype=float)
        for i in range(n):
            s = self._samples[i]
            e = self._expdata[i]
            b = bmatrix(s.alatt, s.angdeg, nargout=1)
            corr = e.get_gon_rot_matrix(b)
            if get_U:
                u_mat[:, :, i] = corr @ b
            else:
                u_mat[:, :, i] = corr
        return u_mat

    def change_crystal(
        self, alignment_info: crystal_alignment_info, nargout: Optional[int] = None
    ) -> "Experiment":
        """Change the crystal lattice and orientation in the Experiment using ``crystal_alignment_info``.

        Translated from ``@Experiment/change_crystal.m``.
        """
        if not isinstance(alignment_info, crystal_alignment_info):
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Expected crystal_alignment_info, got {type(alignment_info).__name__}"
            )

        n = self.n_runs
        alatt = alignment_info.alatt
        angdeg = alignment_info.angdeg
        new_samples = unique_references_container("IX_samp")

        for i in range(n):
            exp_i = copy.deepcopy(self._expdata[i])
            samp_i = copy.deepcopy(self._samples[i])
            deg_mode = exp_i.angular_is_degree
            exp_i.angular_is_degree = True

            alatt0 = samp_i.alatt
            angdeg0 = samp_i.angdeg

            if alignment_info.hkl_mode:
                rlu_corr = alignment_info.get_corr_mat(alatt0, angdeg0)
            else:
                proj = exp_i.get_projection()
                proj.alatt = alatt0
                proj.angdeg = angdeg0
                rlu_corr = alignment_info.get_corr_mat(proj)

            _, _, dpsi, gl, gs = crystal_pars_correct(
                exp_i.u,
                exp_i.v,
                alatt0,
                angdeg0,
                exp_i.omega,
                exp_i.dpsi,
                exp_i.gl,
                exp_i.gs,
                rlu_corr,
                nargout=5,
            )

            samp_i.alatt = alatt
            samp_i.angdeg = angdeg
            exp_i.dpsi = dpsi
            exp_i.gl = gl
            exp_i.gs = gs

            if exp_i._u_to_rlu is not None:
                exp_i.u_to_rlu = rlu_corr @ exp_i.u_to_rlu[:3, :3]

            exp_i.angular_is_degree = deg_mode
            self._expdata[i] = exp_i
            new_samples.add(samp_i)

        self._samples = new_samples
        return self

    def get_subobj(
        self,
        runids_to_keep: Union[int, float, Sequence[Union[int, float]], np.ndarray],
        *flags: str,
    ) -> "Experiment":
        """Extract a sub-Experiment containing only the specified runs or indices.

        Translated from ``@Experiment/Experiment.m:get_subobj`` and ``private/get_subobj_.m``.
        """
        idx_mode = any(
            isinstance(f, str) and f.startswith("-ind") for f in flags
        )
        keep_ids = np.asarray(runids_to_keep).ravel()
        if keep_ids.size == 0 or self.n_runs == 0:
            return Experiment()

        recalc_flag = self._runid_recalculated
        expdata_source = self._expdata

        if idx_mode:
            ind_1based = np.asarray(keep_ids, dtype=int)
            if np.any(ind_1based < 1) or np.any(ind_1based > self.n_runs):
                raise ValueError(
                    "HORACE:Experiment:invalid_argument: Some or all input indexes exceed the number of runs"
                )
        else:
            existing_keys = np.asarray(self._runid_map.keys, dtype=float)
            if not np.any(np.isin(keep_ids.astype(float), existing_keys)):
                ind_1based = np.asarray(keep_ids, dtype=int)
                if np.any(ind_1based < 1) or np.any(ind_1based > self.n_runs):
                    raise ValueError(
                        "HORACE:Experiment:invalid_argument: requested run_indexes to extract lie outside the range of existing run indexes"
                    )
                expdata_source = ExpDataList(
                    [copy.deepcopy(x) for x in self._expdata]
                )
                for i_0, exp_i in enumerate(expdata_source):
                    exp_i.run_id = float(i_0 + 1)
                recalc_flag = True
            else:
                mapped = self._runid_map.get_values_for_keys(keep_ids, True)
                mapped_arr = np.asarray(mapped, dtype=float).ravel()
                if np.any(np.isnan(mapped_arr)):
                    raise ValueError(
                        "HORACE:Experiment:invalid_argument: Some or all input run_ids are not present in runid_map"
                    )
                ind_1based = mapped_arr.astype(int)

        ind_0based = [int(i) - 1 for i in ind_1based]
        expdata_sub = [copy.deepcopy(expdata_source[i]) for i in ind_0based]
        samples_sub = self._samples.get_subset(ind_0based)
        inst_sub = self._instruments.get_subset(ind_0based)
        if self._detectors.n_runs == self.n_runs:
            det_sub = self._detectors.get_subset(ind_0based)
        elif self._detectors.n_runs == 0:
            det_sub = unique_references_container("IX_detector_array")
        else:
            det_sub = self._detectors.get_subset([0])

        subexper = Experiment(det_sub, inst_sub, samples_sub, expdata_sub)
        if recalc_flag:
            subexper._runid_recalculated = True
        return subexper

    def get_experiments(
        self, ind: Union[int, Sequence[int], np.ndarray]
    ) -> Union[IX_experiment, List[IX_experiment]]:
        """Return ``IX_experiment`` object(s) at 1-based experiment index/indices ``ind``."""
        ind_arr = np.asarray(ind, dtype=int).ravel()
        res = [copy.deepcopy(self._expdata[int(i) - 1]) for i in ind_arr]
        if np.ndim(ind) == 0:
            return res[0]
        return res

    def get_experiment_idx(
        self,
        run_idx: Union[int, float, Sequence[Union[int, float]], np.ndarray],
        sparse_lookup: bool = True,
    ) -> np.ndarray:
        """Convert pixel ``run_id`` array into 1-based contiguous experiment header indices."""
        return self._runid_map.get_values_for_keys(run_idx, sparse_lookup)

    def get_unique_samples(self) -> Any:
        """Return unique sample objects container."""
        return self._samples.unique_objects

    def get_unique_instruments(self) -> Any:
        """Return unique instrument objects container."""
        return self._instruments.unique_objects

    def combine_experiments(
        self,
        exp_cellarray: Sequence["Experiment"],
        allow_equal_headers: bool = False,
        keep_runid: bool = True,
        nargout: Optional[int] = None,
    ) -> Any:
        """Combine multiple ``Experiment`` objects into a single ``Experiment``.

        Translated from ``@Experiment/private/combine_experiments_.m``.
        """
        nout = _infer_nargout(default=2, explicit_nargout=nargout)

        all_exps = [self]
        for item in exp_cellarray:
            if not isinstance(item, Experiment):
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Expected Experiment objects, got {type(item).__name__}"
                )
            all_exps.append(item)

        non_empty = [e for e in all_exps if e.n_runs > 0]
        if len(non_empty) == 0:
            if nout == 1:
                return Experiment()
            return Experiment(), []

        if len(non_empty) == 1:
            single = copy.deepcopy(non_empty[0])
            if nout == 1:
                return single
            return single, [None]

        first = non_empty[0]
        other_exps = non_empty[1:]

        expdata_comb, nsqw_change_runid, runid_map_comb = first._expdata.combine(
            [e._expdata for e in other_exps],
            allow_equal_headers=allow_equal_headers,
            keep_runid=keep_runid,
            this_runid_map=first._runid_map,
            nargout=3,
        )

        inst_comb = unique_references_container("IX_inst")
        samp_comb = unique_references_container("IX_samp")
        det_comb = unique_references_container("IX_detector_array")

        has_dets = any(e._detectors.n_runs > 0 for e in non_empty)
        all_dets = all(e._detectors.n_runs == e.n_runs for e in non_empty)

        for e in non_empty:
            for i in range(e.n_runs):
                inst_comb.add(copy.deepcopy(e._instruments[i]))
                samp_comb.add(copy.deepcopy(e._samples[i]))
                if has_dets and all_dets:
                    det_comb.add(copy.deepcopy(e._detectors[i]))

        exp_out = Experiment()
        exp_out._do_check_combo_arg = False
        try:
            exp_out._detectors = det_comb
            exp_out._detectors_set = det_comb.n_runs > 0
            exp_out._instruments = inst_comb
            exp_out._instruments_set = inst_comb.n_runs > 0
            exp_out._samples = samp_comb
            exp_out._samples_set = samp_comb.n_runs > 0
            exp_out._expdata = expdata_comb
            exp_out._expdata_set = len(expdata_comb) > 0
            exp_out._runid_map = runid_map_comb
            exp_out._runid_recalculated = not keep_runid
        finally:
            exp_out._do_check_combo_arg = True
        exp_out.check_combo_arg()

        if nout == 1:
            return exp_out
        return exp_out, nsqw_change_runid

    # ------------------------------------------------------------------
    # Legacy binary file header conversion
    # ------------------------------------------------------------------
    def convert_to_old_headers(
        self,
        *args: Any,
        header_num: Optional[int] = None,
        nomangle: bool = False,
    ) -> List[Dict[str, Any]]:
        """Convert Experiment into a list of legacy binary file header dicts.

        Translated from ``@Experiment/Experiment.m:convert_to_old_headers`` and ``private/convert_to_old_headers_.m``.
        """
        for a in args:
            if isinstance(a, str) and a == "-nomangle":
                nomangle = True
            elif isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_)) and int(a) > 0:
                header_num = int(a)

        if header_num is not None and not isinstance(header_num, (bool, np.bool_)) and int(header_num) > 0:
            indices_0 = [int(header_num) - 1]
        else:
            indices_0 = list(range(self.n_runs))

        headers: List[Dict[str, Any]] = []
        for i in indices_0:
            inst_i = self._instruments[i] if self._instruments.n_runs > i else IX_null_inst()
            samp_i = self._samples[i] if self._samples.n_runs > i else IX_null_sample()
            hdr = self._expdata[i].convert_to_binfile_header(
                "-inst_samp", inst_i, samp_i, nomangle=nomangle
            )
            headers.append(hdr)
        return headers

    @staticmethod
    def build_from_binfile_headers(
        headers: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ) -> "Experiment":
        """Build an ``Experiment`` object from one or more legacy binfile header dictionaries.

        Translated from ``@Experiment/Experiment.m:build_from_binfile_headers`` and ``private/build_from_binfile_headers_.m``.
        """
        if isinstance(headers, dict):
            hdr_list = [headers]
        elif isinstance(headers, (list, tuple, np.ndarray)):
            hdr_list = list(headers)
        else:
            raise ValueError(
                f"HORACE:Experiment:invalid_argument: Expected header dict or list of dicts, got {type(headers).__name__}"
            )

        if len(hdr_list) == 0:
            return Experiment()

        inst_container = unique_references_container("IX_inst")
        samp_container = unique_references_container("IX_samp")
        exp_list: List[IX_experiment] = []

        for hdr in hdr_list:
            if not isinstance(hdr, dict):
                raise ValueError(
                    "HORACE:Experiment:invalid_argument: Input headers must be dictionaries"
                )
            h_copy = dict(hdr)
            raw_inst = h_copy.pop("instrument", None)
            raw_samp = h_copy.pop("sample", None)

            exp_i, alatt, angdeg = IX_experiment.build_from_binfile_header(
                h_copy, nargout=3
            )
            exp_list.append(exp_i)

            # Convert instrument
            if raw_inst is None or (isinstance(raw_inst, dict) and len(raw_inst) == 0) or (isinstance(raw_inst, str) and raw_inst == ""):
                inst_obj: IX_inst = IX_null_inst()
            elif isinstance(raw_inst, IX_inst):
                inst_obj = copy.deepcopy(raw_inst)
            elif isinstance(raw_inst, dict) and "serial_name" in raw_inst:
                inst_obj = Serializable.loadobj(raw_inst)
            else:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Unrecognised instrument in header: {type(raw_inst).__name__}"
                )
            inst_container.add(inst_obj)

            # Convert sample
            if raw_samp is None or (isinstance(raw_samp, dict) and len(raw_samp) == 0) or (isinstance(raw_samp, str) and raw_samp == ""):
                samp_obj: IX_samp = IX_samp("", alatt, angdeg)
            elif isinstance(raw_samp, IX_samp):
                samp_obj = copy.deepcopy(raw_samp)
                if samp_obj.alatt.size == 0:
                    samp_obj.alatt = alatt
                if samp_obj.angdeg.size == 0:
                    samp_obj.angdeg = angdeg
            elif isinstance(raw_samp, dict):
                if "serial_name" in raw_samp:
                    samp_obj = Serializable.loadobj(raw_samp)
                else:
                    samp_obj = IX_sample(raw_samp)
                if samp_obj.alatt.size == 0:
                    samp_obj.alatt = alatt
                if samp_obj.angdeg.size == 0:
                    samp_obj.angdeg = angdeg
            else:
                raise ValueError(
                    f"HORACE:Experiment:invalid_argument: Unrecognised sample in header: {type(raw_samp).__name__}"
                )
            samp_container.add(samp_obj)

        return Experiment(
            unique_references_container("IX_detector_array"),
            inst_container,
            samp_container,
            exp_list,
        )

    # ------------------------------------------------------------------
    # Serializable Interface
    # ------------------------------------------------------------------
    def saveable_fields(self) -> List[str]:
        return ["instruments", "samples", "expdata", "detector_arrays", "runid_map"]

    def construction_fields(self) -> List[str]:
        return ["detector_arrays", "instruments", "samples", "expdata"]

    def convert_old_struct(
        self, inputs: Dict[str, Any], ver: Optional[float] = None
    ) -> Tuple[Dict[str, Any], "Experiment"]:
        """Convert legacy struct representations (ver 0 or 1) to version 2."""
        out = dict(inputs)
        if ver is not None and ver == 1:
            return out, self
        if (
            "filename" in out
            and "filepath" in out
            and "efix" in out
        ):
            exp_built = Experiment.build_from_binfile_headers(out)
            return exp_built.to_bare_struct(), self
        return out, self

    @classmethod
    def loadobj(cls, S: Any) -> "Experiment":
        """Load an Experiment instance from a saved struct or legacy header."""
        if isinstance(S, Experiment):
            return copy.deepcopy(S)
        if isinstance(S, list):
            if len(S) > 0 and isinstance(S[0], dict) and "filename" in S[0] and "efix" in S[0]:
                return cls.build_from_binfile_headers(S)
        if isinstance(S, dict):
            if "filename" in S and "filepath" in S and "efix" in S and "serial_name" not in S:
                return cls.build_from_binfile_headers(S)
            if "array_dat" in S and "version" not in S and "serial_name" not in S:
                ad = S["array_dat"]
                if isinstance(ad, dict) and "filename" in ad:
                    return cls.build_from_binfile_headers(ad)
                if isinstance(ad, list) and len(ad) > 0 and isinstance(ad[0], dict) and "filename" in ad[0]:
                    return cls.build_from_binfile_headers(ad)
        obj = cls()
        return Serializable.loadobj(S, obj)

