"""Unit tests for the Experiment class methods.

Translated from:
  _test/test_experiment/test_experiment_methods.m
"""
from __future__ import annotations

import copy
import math
import pytest
import numpy as np

from pyhorace.coord_transf import crystal_alignment_info, fast_map
from pyhorace.core.serializable import Serializable, compare_values


def equal_to_tol(a, b, tol=(1e-9, 1e-9)) -> bool:
    ok, _ = compare_values(a, b, tol=tol)
    return ok
from pyhorace.core.unique_objects_container import unique_objects_container
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data import (
    Experiment,
    IX_dataset,
    calc_idx_for_eq_to_tol,
    compact_array,
)
from pyhorace.data.instr_classes import (
    IX_aperture,
    IX_det_He3tube,
    IX_detector_array,
    IX_detector_bank,
    IX_experiment,
    IX_fermi_chopper,
    IX_inst,
    IX_inst_DGfermi,
    IX_moderator,
    IX_null_inst,
    IX_null_sample,
    IX_samp,
    IX_sample,
)


def _create_maps_like_inst(ei: float = 100.0) -> IX_inst_DGfermi:
    mod = IX_moderator(
        12.0, 32.0, "ikcarp", [3.0, 25.0, 0.2],
        temperature=300.0, thickness=0.05, width=0.1, height=0.1, energy=ei, name="moderator",
    )
    ap = IX_aperture(-10.0, 0.09, 0.09)
    fc = IX_fermi_chopper(1.5, 600.0, 0.049, 1.3, 0.002, energy=ei, name="sloppy")
    return IX_inst_DGfermi(mod, ap, fc, energy=ei, name="MAPS")


def _create_merlin_like_inst(ei: float = 100.0) -> IX_inst_DGfermi:
    mod = IX_moderator(
        10.0, 15.0, "ikcarp", [2.5, 20.0, 0.1],
        temperature=100.0, thickness=0.05, width=0.1, height=0.1, energy=ei, name="mod_merlin",
    )
    ap = IX_aperture(-8.0, 0.08, 0.08)
    fc = IX_fermi_chopper(1.8, 400.0, 0.05, 1.2, 0.0015, energy=ei, name="gast")
    return IX_inst_DGfermi(mod, ap, fc, energy=ei, name="MERLIN")


class TestExperimentMethods:
    """Port of ``test_experiment_methods`` from MATLAB."""

    @pytest.fixture(autouse=True)
    def setup_experiments(self) -> None:
        exp1 = IX_experiment()
        exp1.filename = "a1"
        exp1.efix = 10.0
        exp1.psi = 10.0
        exp1.angular_units = "rad"
        exp1.en = [1.0, 2.0, 4.0]
        exp1.run_id = 10

        exp2 = IX_experiment()
        exp2.filename = "a2"
        exp2.efix = 10.0 + 0.5e-9
        exp2.psi = 20.0
        exp2.angular_units = "rad"
        exp2.en = [1.0, 2.0, 4.0]
        exp2.run_id = 20

        exp3 = IX_experiment()
        exp3.filename = "a3"
        exp3.efix = 20.0
        exp3.psi = 30.0
        exp3.angular_units = "rad"
        exp3.en = [1.0, 2.0, 4.0]
        exp3.run_id = 30

        self.Sample1 = IX_sample([1.0, 2.0, 3.0], [90.0, 91.0, 92.0])
        self.Sample2 = IX_sample([1.1, 2.1, 3.1], [90.0, 91.0, 92.0])
        self.Sample3 = IX_sample([1.2, 2.2, 3.2], [90.0, 91.0, 92.0])

        self.Inst1 = IX_null_inst()
        self.Inst2 = _create_maps_like_inst(100.0)
        self.Inst3 = _create_merlin_like_inst(150.0)

        self.test_exp = Experiment(
            [],
            [self.Inst1, self.Inst2, self.Inst3],
            [self.Sample1, self.Sample2, self.Sample3],
            [exp1, exp2, exp3],
        )

    def test_get_efix(self) -> None:
        efix = self.test_exp.get_efix()
        np.testing.assert_allclose(efix, [10.0, 10.0, 20.0], atol=1e-7)

        efix_c = self.test_exp.get_efix("-compact")
        assert isinstance(efix_c, compact_array)
        assert efix_c.n_unique == 2
        assert efix_c.n_objects == 3
        assert efix_c.unique_val == [10.0, 20.0]

        efix_c2, idx2 = self.test_exp.get_efix(True)
        assert isinstance(efix_c2, compact_array)
        assert len(idx2) == 2
        np.testing.assert_array_equal(idx2[0], [1, 2])
        np.testing.assert_array_equal(idx2[1], [3])

    def test_find_same_within_tol(self) -> None:
        data = [self.test_exp.expdata[i].efix for i in range(3)]
        idx = calc_idx_for_eq_to_tol(data)
        assert len(idx) == 2
        np.testing.assert_array_equal(idx[0], [1, 2])
        np.testing.assert_array_equal(idx[1], [3])

    def test_empty_constructor_properties(self) -> None:
        ex = Experiment()
        assert ex.expdata == []
        assert ex.samples.n_runs == 0
        assert ex.instruments.n_runs == 0
        assert ex.detector_arrays.n_runs == 0

    def test_one_run_constructor_MakesSingleLengthProperties(self) -> None:
        ex = Experiment(IX_null_inst(), IX_null_sample(), IX_experiment())
        assert len(ex.expdata) == 1
        assert ex.samples.n_runs == 1
        assert ex.instruments.n_runs == 1

    def test_SettingContainerWaitForCompletion_OneRun(self) -> None:
        ex = Experiment()
        ex.instruments = [self.Inst1]
        ex.samples = [self.Sample1]
        ex.expdata = [self.test_exp.expdata[0]]
        assert len(ex.expdata) == 1
        assert ex.samples.n_runs == 1
        assert ex.instruments.n_runs == 1

    def test_SettingContainerWaitForCompletion_MultiRun(self) -> None:
        ex = Experiment()
        ex.instruments = [self.Inst1, self.Inst2, self.Inst3]
        ex.samples = [self.Sample1, self.Sample2, self.Sample3]
        ex.expdata = self.test_exp.expdata
        assert len(ex.expdata) == 3
        assert ex.samples.n_runs == 3
        assert ex.instruments.n_runs == 3

    def test_MultiRunConstructor_PropertiesHaveMultipleElements(self) -> None:
        assert len(self.test_exp.expdata) == 3
        assert self.test_exp.samples.n_runs == 3
        assert self.test_exp.instruments.n_runs == 3

    def test_SaveLoad_RestoresIdenticalObject(self) -> None:
        s = self.test_exp.to_struct()
        ex2 = Serializable.from_struct(s)
        assert equal_to_tol(self.test_exp, ex2)

    def test_WrongAssignment_ThrowsException(self) -> None:
        ex = Experiment()
        with pytest.raises(ValueError, match="HORACE:Experiment:invalid_argument"):
            ex.instruments = "not_an_instrument"
        with pytest.raises(ValueError, match="HORACE:Experiment:invalid_argument"):
            ex.samples = "not_a_sample"
        with pytest.raises(ValueError, match="HORACE:Experiment:invalid_argument"):
            ex.expdata = "not_expdata"

    def test_CorrectAssignment_Succeeds(self) -> None:
        ex = Experiment()
        ex.expdata = self.test_exp.expdata
        ex.instruments = [self.Inst1, self.Inst2, self.Inst3]
        ex.samples = [self.Sample1, self.Sample2, self.Sample3]
        assert equal_to_tol(ex, self.test_exp)

        # Also test assigning unique_references_container directly
        ex2 = Experiment()
        ex2.expdata = self.test_exp.expdata
        urc_i = unique_references_container("IX_inst")
        for inst in (self.Inst1, self.Inst2, self.Inst3):
            urc_i.add(inst)
        ex2.instruments = urc_i

        urc_s = unique_references_container("IX_samp")
        for samp in (self.Sample1, self.Sample2, self.Sample3):
            urc_s.add(samp)
        ex2.samples = urc_s
        assert equal_to_tol(ex2, self.test_exp)

    def test_set_sample_preserves_lattice_in_sample(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        new_samp = IX_sample()  # empty alatt and angdeg
        assert new_samp.alatt.size == 0
        assert new_samp.angdeg.size == 0

        ex.samples = new_samp
        assert ex.samples.n_runs == 3
        np.testing.assert_allclose(ex.samples[0].alatt, self.Sample1.alatt)
        np.testing.assert_allclose(ex.samples[1].alatt, self.Sample2.alatt)
        np.testing.assert_allclose(ex.samples[2].alatt, self.Sample3.alatt)
        np.testing.assert_allclose(ex.samples[0].angdeg, self.Sample1.angdeg)

    def test_old_header_conversions(self) -> None:
        exp = copy.deepcopy(self.test_exp)
        hdrs = exp.convert_to_old_headers()
        assert len(hdrs) == 3
        assert hdrs[0]["filename"] == "a1$id$10"
        assert hdrs[1]["filename"] == "a2$id$20"
        assert hdrs[2]["filename"] == "a3$id$30"
        np.testing.assert_allclose(hdrs[0]["alatt"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(hdrs[1]["alatt"], [1.1, 2.1, 3.1])
        np.testing.assert_allclose(hdrs[2]["alatt"], [1.2, 2.2, 3.2])

        reconstructed_exp = Experiment.build_from_binfile_headers(hdrs)
        assert equal_to_tol(reconstructed_exp, exp)

    def test_old_header_conversions_with_nan_runids_resets_runids(self) -> None:
        exp = copy.deepcopy(self.test_exp)
        for i in range(3):
            exp.expdata[i].run_id = float("nan")

        hdrs = exp.convert_to_old_headers()
        assert hdrs[0]["filename"] == "a1"
        assert hdrs[1]["filename"] == "a2"
        assert hdrs[2]["filename"] == "a3"

        reconstructed_exp = Experiment.build_from_binfile_headers(hdrs)
        np.testing.assert_array_equal(reconstructed_exp.expdata.get_run_ids(), [1, 2, 3])
        np.testing.assert_array_equal(reconstructed_exp.runid_map.keys, [1, 2, 3])
        np.testing.assert_array_equal(reconstructed_exp.runid_map.values, [1, 2, 3])

        # After setting run_ids to 1,2,3 on exp, reconstructed_exp equals exp
        for i in range(3):
            exp.expdata[i].run_id = float(i + 1)
        exp.runid_map = fast_map([1, 2, 3], [1, 2, 3])
        assert equal_to_tol(reconstructed_exp, exp)

    def test_init_ResetDuplicateRunids(self) -> None:
        exp1 = copy.deepcopy(self.test_exp.expdata[0])
        exp2 = copy.deepcopy(self.test_exp.expdata[1])
        exp3 = copy.deepcopy(self.test_exp.expdata[2])
        exp1.run_id = 10
        exp2.run_id = 10  # duplicate!
        exp3.run_id = 30

        ex = Experiment(
            [],
            [self.Inst1, self.Inst2, self.Inst3],
            [self.Sample1, self.Sample2, self.Sample3],
            [exp1, exp2, exp3],
        )
        assert ex.runid_recalculated is True
        np.testing.assert_array_equal(ex.expdata.get_run_ids(), [1, 2, 3])
        np.testing.assert_array_equal(ex.runid_map.keys, [1, 2, 3])
        np.testing.assert_array_equal(ex.runid_map.values, [1, 2, 3])

    def test_get_subobj_works(self) -> None:
        sub = self.test_exp.get_subobj([10, 30])
        assert sub.n_runs == 2
        np.testing.assert_array_equal(sub.runid_map.keys, [10, 30])
        np.testing.assert_array_equal(sub.runid_map.values, [1, 2])
        assert sub.expdata[0] == self.test_exp.expdata[0]
        assert sub.expdata[1] == self.test_exp.expdata[2]
        assert sub.samples[0] == self.Sample1
        assert sub.samples[1] == self.Sample3
        assert sub.instruments[0] == self.Inst1
        assert sub.instruments[1] == self.Inst3

        # Also test with '-indexes' / '-ind' flag
        sub_idx = self.test_exp.get_subobj([2, 3], "-indexes")
        assert sub_idx.n_runs == 2
        assert sub_idx.runid_recalculated is False
        np.testing.assert_array_equal(sub_idx.runid_map.keys, [20, 30])
        np.testing.assert_array_equal(sub_idx.runid_map.values, [1, 2])
        assert sub_idx.expdata[0] == self.test_exp.expdata[1]
        assert sub_idx.expdata[1] == self.test_exp.expdata[2]

        # Test fallback to index mode with recalculated runids when keys are not in runid_map
        sub_recalc = self.test_exp.get_subobj([2, 3])
        assert sub_recalc.n_runs == 2
        assert sub_recalc.runid_recalculated is True
        np.testing.assert_array_equal(sub_recalc.runid_map.keys, [2, 3])
        np.testing.assert_array_equal(sub_recalc.runid_map.values, [1, 2])
        assert sub_recalc.expdata[0].filename == "a2"
        assert sub_recalc.expdata[0].run_id == 2
        assert sub_recalc.expdata[1].filename == "a3"
        assert sub_recalc.expdata[1].run_id == 3

    def test_get_experiment_idx_gives_expected_indices(self) -> None:
        run_idx = [30, 20, 10, 10, 30]
        expt_idx = self.test_exp.get_experiment_idx(run_idx, sparse_lookup=True)
        np.testing.assert_array_equal(expt_idx, [3, 2, 1, 1, 3])

        expt_idx_dense = self.test_exp.get_experiment_idx(run_idx, sparse_lookup=False)
        np.testing.assert_array_equal(expt_idx_dense, [3, 2, 1, 1, 3])

    def test_runid_DirectResetInvalid_throws(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        with pytest.raises(ValueError, match="HORACE:Experiment:invalid_argument"):
            ex.runid_map = fast_map([10, 20], [1, 2])  # wrong size
        with pytest.raises(ValueError, match="HORACE:Experiment:invalid_argument"):
            ex.runid_map = fast_map([10, 20, 30], [1, 2, 4])  # not 1..3

    def test_runid_DirectResetValid_reset_ids(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        ex.runid_map = fast_map([100, 200, 300], [3, 1, 2])
        assert ex.runid_recalculated is False
        np.testing.assert_array_equal(ex.expdata.get_run_ids(), [200, 300, 100])

    # ------------------------------------------------------------------
    # Additional tests for Experiment methods
    # ------------------------------------------------------------------
    def test_set_instrument_and_mod_pulse(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        ex.set_instrument(_create_maps_like_inst, "-efix")
        inst_cls, all_inst = ex.get_inst_class()
        assert isinstance(inst_cls, IX_inst_DGfermi)
        assert all_inst is True

        pm, pm_par, present = ex.get_mod_pulse()
        assert present is True
        assert pm == "ikcarp"
        assert pm_par.shape == (3, 3)

        ex.set_mod_pulse("ikcarp", [5.0, 30.0, 0.3])
        pm_avrg, pm2, present2 = ex.calc_mod_pulse_avrgs()
        assert present2 is True
        assert pm2 == "ikcarp"
        np.testing.assert_allclose(pm_avrg, [5.0, 30.0, 0.3])

    def test_set_efix_emode_and_calc_efix_avrgs(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        ex.set_efix_emode([50.0, 50.0, 50.0], 1)
        ave_efix, ave_emode, present = ex.calc_efix_avrgs()
        assert present is True
        assert ave_efix == pytest.approx(50.0)
        assert ave_emode == pytest.approx(1.0)
        np.testing.assert_array_equal(ex.get_emode(), [1, 1, 1])

    def test_get_en_transfer_and_is_same_ebins(self) -> None:
        assert self.test_exp.is_same_ebins() is True
        en_tr = self.test_exp.get_en_transfer(bin_centre=True, compact=False)
        np.testing.assert_allclose(en_tr, [1.5, 3.0])

        en_c = self.test_exp.get_en_transfer(bin_centre=True, compact=True)
        assert isinstance(en_c, compact_array)
        assert en_c.n_unique == 1
        assert en_c.n_objects == 3

    def test_combine_experiments(self) -> None:
        sub1 = self.test_exp.get_subobj([10, 20])
        sub2 = self.test_exp.get_subobj([30])
        comb, change_map = sub1.combine_experiments([sub2], allow_equal_headers=False, keep_runid=True)
        assert comb.n_runs == 3
        assert change_map == [None]
        np.testing.assert_array_equal(comb.runid_map.keys, [10, 20, 30])

    def test_change_crystal(self) -> None:
        ex = copy.deepcopy(self.test_exp)
        for i in range(3):
            ex.expdata[i].u = [1.0, 0.0, 0.0]
            ex.expdata[i].v = [0.0, 1.0, 0.0]
        align = crystal_alignment_info([2.0, 2.0, 2.0], [90.0, 90.0, 90.0], [0.01, -0.02, 0.03])
        align.hkl_mode = True
        ex.change_crystal(align)
        for i in range(3):
            np.testing.assert_allclose(ex.samples[i].alatt, [2.0, 2.0, 2.0])
            np.testing.assert_allclose(ex.samples[i].angdeg, [90.0, 90.0, 90.0])

    def test_get_UXZ_and_header_average(self) -> None:
        u_mat = self.test_exp.get_UXZ(get_U=True)
        assert u_mat.shape == (3, 3, 3)
        hdr_av, same_ebins = self.test_exp.header_average(nargout=2)
        assert same_ebins is True
        assert hdr_av["filename"] == "a1$id$10"

