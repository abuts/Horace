"""Unit tests for IX_moderator.

Converted from:
  _test/test_instrument_classes/test_IX_moderator.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_moderator import IX_moderator
from pyhorace.data.instr_classes.pdf_table import sampling_statistics


class TestIXModerator:
    """Test suite for IX_moderator."""

    def setup_method(self) -> None:
        self.mik = IX_moderator(
            15.0,
            30.0,
            "ikcarp",
            [5.0, 25.0, 0.13],
            temperature=100.0,
            thickness=0.05,
            width=0.1,
            height=0.12,
            energy=80.0,
            name="Chilled methane",
        )
        self.mikp = IX_moderator(
            12.0,
            23.0,
            "ikcarp_param",
            [0.05, 25.0, 200.0],
            temperature=300.0,
            thickness=0.04,
            width=0.12,
            height=0.11,
            energy=120.0,
            name="Water",
        )
        self.t_tab = np.array([0.0, 5.0, 15.0, 25.0, 40.0, 65.0, 100.0])
        self.y_tab = np.array([0.0, 0.5, 1.0, 0.8, 0.4, 0.1, 0.0])
        self.mtab = IX_moderator(
            10.0,
            15.0,
            "table",
            [self.t_tab, self.y_tab],
            name="Table mod",
        )

    def test_empty_constructor(self) -> None:
        m = IX_moderator()
        assert m.name == ""
        assert m.distance == 0.0
        assert m.angle == 0.0
        assert m.pulse_model == "ikcarp"
        assert np.allclose(m.pp, [0.0, 0.0, 0.0])

    def test_ikcarp_pulse_width_exact_reference(self) -> None:
        dt, t_av, fwhh = self.mik.pulse_width()
        assert np.isclose(dt, 15.064444895182829, rtol=1e-12)
        assert np.isclose(t_av, 18.25, rtol=1e-12)
        assert np.isclose(fwhh, 17.796795013023157, rtol=1e-12)

        w, tmax, tlo, thi = self.mik.pulse_width2(0.25)
        assert np.isclose(w, 26.151479900731516, rtol=1e-12)
        assert np.isclose(tmax, 10.239533277555076, rtol=1e-12)
        assert np.isclose(tlo, 2.359637371190054, rtol=1e-12)
        assert np.isclose(thi, 28.511117271921570, rtol=1e-12)

    def test_ikcarp_param_pulse_width_exact_reference(self) -> None:
        dt, t_av, fwhh = self.mikp.pulse_width()
        assert np.isclose(dt, 24.141435934481127, rtol=1e-12)
        assert np.isclose(t_av, 29.692839193676942, rtol=1e-12)
        assert np.isclose(fwhh, 26.406900251721297, rtol=1e-12)

        w, tmax, tlo, thi = self.mikp.pulse_width2(0.25)
        assert np.isclose(w, 42.980743939855834, rtol=1e-12)
        assert np.isclose(tmax, 12.997016317476614, rtol=1e-7)
        assert np.isclose(tlo, 2.860652447599696, rtol=1e-12)
        assert np.isclose(thi, 45.841396387455532, rtol=1e-12)

    def test_table_pulse_width_and_shape(self) -> None:
        dt, t_av, fwhh = self.mtab.pulse_width()
        var_ref, tav_ref = self.mtab.pdf.var()
        fwhh_ref, _, _, _ = self.mtab.pdf.width(0.5)
        assert np.isclose(dt, np.sqrt(var_ref), rtol=1e-12)
        assert np.isclose(t_av, tav_ref, rtol=1e-12)
        assert np.isclose(fwhh, fwhh_ref, rtol=1e-12)

        w, tmax, tlo, thi = self.mtab.pulse_width2(0.25)
        w_ref, tmax_ref, tlo_ref, thi_ref = self.mtab.pdf.width(0.25)
        assert np.isclose(w, w_ref, rtol=1e-12)
        assert np.isclose(tmax, tmax_ref, rtol=1e-12)
        assert np.isclose(tlo, tlo_ref, rtol=1e-12)
        assert np.isclose(thi, thi_ref, rtol=1e-12)

    def test_pulse_shape_evaluation(self) -> None:
        y_grid, t_grid = self.mik.pulse_shape()
        assert y_grid.shape == t_grid.shape
        assert np.all(y_grid >= 0.0)

        t_test = np.array([5.0, 10.0, 20.0, 40.0])
        y_test = self.mik.pulse_shape(t_test)
        assert y_test.shape == t_test.shape
        assert np.all(y_test > 0.0)

    def test_rand_statistics(self) -> None:
        rng = np.random.default_rng(2024)
        for mod in (self.mik, self.mikp, self.mtab):
            X = mod.rand(100000, 1, rng=rng)
            mean_val, sig_mean, sig_val, sig_sig, _, _, _, _ = sampling_statistics(X)
            dt_ref, tav_ref, _ = mod.pulse_width()
            # Table pdf for ikcarp has small discretization difference (< 0.05 us), allow 4*sigma + 0.05
            assert abs(float(mean_val) - tav_ref) < 4.0 * float(sig_mean) + 0.05
            assert abs(float(sig_val) - dt_ref) < 4.0 * float(sig_sig) + 0.05

    def test_in_place_pp_mutation(self) -> None:
        m = IX_moderator(self.mik)
        dt_orig, _, _ = m.pulse_width()
        m.pp[0] = 12.0
        dt_new, _, _ = m.pulse_width()
        assert dt_new > dt_orig

    def test_serialization_roundtrip(self) -> None:
        for mod in (self.mik, self.mikp, self.mtab):
            d = mod.to_dict()
            mod_rec = IX_moderator.from_dict(d)
            assert mod_rec == mod
            assert np.allclose(mod_rec.pulse_width(), mod.pulse_width(), rtol=1e-12)
