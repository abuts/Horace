"""Unit tests for saving and loading the Experiment class across versions.

Translated from:
  _test/test_experiment/test_experiment_loadsave.m
"""
from __future__ import annotations

import copy
import pytest
import numpy as np

from pyhorace.core.serializable import Serializable, compare_values


def equal_to_tol(a, b, tol=(1e-9, 1e-9)) -> bool:
    ok, _ = compare_values(a, b, tol=tol)
    return ok
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data import Experiment
from pyhorace.data.instr_classes import (
    IX_aperture,
    IX_det_He3tube,
    IX_detector_array,
    IX_detector_bank,
    IX_experiment,
    IX_fermi_chopper,
    IX_inst_DGfermi,
    IX_moderator,
    IX_null_inst,
    IX_null_sample,
    IX_sample,
)


def _make_detector_array(n_det: int = 4) -> IX_detector_array:
    det = IX_det_He3tube(0.0254, 1.0, 0.002, 10.0)
    x2 = np.linspace(2.0, 3.0, n_det)
    phi = np.linspace(10.0, 40.0, n_det)
    azim = np.zeros(n_det)
    det_nums = np.arange(1, n_det + 1)
    bank = IX_detector_bank(det_nums, x2, phi, azim, det)
    return IX_detector_array(bank)


def _make_multi_run_experiment(with_detectors: bool = False) -> Experiment:
    exp1 = IX_experiment(
        filename="run1.nxspe",
        filepath="/data/",
        efix=35.0,
        emode=1,
        cu=[1.0, 0.0, 0.0],
        cv=[0.0, 1.0, 0.0],
        psi=10.0,
        omega=1.0,
        dpsi=0.5,
        gl=0.2,
        gs=-0.1,
        en=[0.0, 2.0, 4.0, 6.0],
        run_id=101,
    )
    exp1.angular_units = "rad"

    exp2 = IX_experiment(
        filename="run2.nxspe",
        filepath="/data/",
        efix=35.0,
        emode=1,
        cu=[1.0, 0.0, 0.0],
        cv=[0.0, 1.0, 0.0],
        psi=25.0,
        omega=1.0,
        dpsi=0.5,
        gl=0.2,
        gs=-0.1,
        en=[0.0, 2.0, 4.0, 6.0],
        run_id=102,
    )
    exp2.angular_units = "rad"

    mod = IX_moderator(
        12.0, 32.0, "ikcarp", [3.0, 25.0, 0.2],
        temperature=300.0, thickness=0.05, width=0.1, height=0.1, energy=35.0, name="mod",
    )
    ap = IX_aperture(-10.0, 0.09, 0.09)
    fc = IX_fermi_chopper(1.5, 600.0, 0.049, 1.3, 0.002, energy=35.0, name="sloppy")
    inst1 = IX_inst_DGfermi(mod, ap, fc, energy=35.0, name="MAPS")
    inst2 = IX_inst_DGfermi(mod, ap, fc, energy=35.0, name="MAPS")

    samp1 = IX_sample([2.8, 2.8, 2.8], [90.0, 90.0, 90.0])
    samp2 = IX_sample([2.8, 2.8, 2.8], [90.0, 90.0, 90.0])

    dets = [_make_detector_array(4), _make_detector_array(4)] if with_detectors else []
    return Experiment(dets, [inst1, inst2], [samp1, samp2], [exp1, exp2])


class TestExperimentLoadSave:
    """Port of ``test_experiment_loadsave`` from MATLAB."""

    def test_load_from_old_v0_header_structs(self) -> None:
        exp_orig = _make_multi_run_experiment(with_detectors=False)
        old_hdrs = exp_orig.convert_to_old_headers()

        # Load via Experiment.loadobj on list of old header dicts
        exp_loaded = Experiment.loadobj(old_hdrs)
        assert equal_to_tol(exp_loaded, exp_orig)

        # Load via single old header dict
        sub1 = exp_orig.get_subobj(101)
        exp_single = Experiment.loadobj(old_hdrs[0])
        assert equal_to_tol(exp_single, sub1)

        # Load via array_dat wrapper (unversioned legacy struct)
        wrapped = {"array_dat": old_hdrs}
        exp_wrapped = Experiment.loadobj(wrapped)
        assert equal_to_tol(exp_wrapped, exp_orig)

    def test_save_load_version_1_struct(self) -> None:
        exp_orig = _make_multi_run_experiment(with_detectors=False)
        s = exp_orig.to_struct()
        s["version"] = 1

        exp_rec = Serializable.from_struct(s)
        assert isinstance(exp_rec, Experiment)
        assert equal_to_tol(exp_rec, exp_orig)

    def test_save_load_version_2_struct_with_detectors(self) -> None:
        exp_orig = _make_multi_run_experiment(with_detectors=True)
        assert exp_orig.detector_arrays.n_runs == 2
        assert exp_orig.detector_arrays.n_unique == 1

        s = exp_orig.to_struct()
        assert s["version"] == 2

        exp_rec = Serializable.from_struct(s)
        assert isinstance(exp_rec, Experiment)
        assert exp_rec.detector_arrays.n_runs == 2
        assert exp_rec.detector_arrays.n_unique == 1
        assert equal_to_tol(exp_rec, exp_orig)

    def test_constructor_with_detector_arrays_3_args(self) -> None:
        det = _make_detector_array(4)
        inst = IX_null_inst()
        samp = IX_sample([3.0, 3.0, 3.0], [90.0, 90.0, 90.0])

        exp = Experiment(det, inst, samp)
        assert exp.n_runs == 1
        assert exp.detector_arrays.n_runs == 1
        assert exp.instruments.n_runs == 1
        assert exp.samples.n_runs == 1
        assert len(exp.expdata) == 1

