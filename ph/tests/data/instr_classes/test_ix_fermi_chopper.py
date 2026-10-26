"""Unit tests for IX_fermi_chopper.

Converted from:
  _test/test_instrument_classes/test_IX_fermi_chopper.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_fermi_chopper import IX_fermi_chopper
from pyhorace.data.instr_classes.pdf_table import sampling_statistics


class TestIXFermiChopper:
    """Test suite for IX_fermi_chopper."""

    def setup_method(self) -> None:
        self.S = IX_fermi_chopper(
            10.0,
            250.0,
            0.05,
            1.3,
            0.002,
            width=0.04,
            height=0.06,
            energy=120.0,
            phase=True,
            jitter=3.0,
            name="Sloppy",
        )
        self.B = IX_fermi_chopper(
            12.0,
            600.0,
            0.049,
            0.92,
            0.00123,
            width=0.052,
            height=0.065,
            energy=550.0,
            phase=False,
            jitter=0.0,
            name="B chopper",
        )

    def test_empty_constructor(self) -> None:
        fc = IX_fermi_chopper()
        assert fc.name == ""
        assert fc.distance == 0.0
        assert fc.frequency == 0.0
        assert fc.radius == 0.0
        assert fc.curvature == 0.0
        assert fc.slit_width == 0.0
        assert fc.phase is True

    def test_constructor_and_properties(self) -> None:
        assert self.S.name == "Sloppy"
        assert self.S.distance == 10.0
        assert self.S.frequency == 250.0
        assert self.S.radius == 0.05
        assert self.S.curvature == 1.3
        assert self.S.slit_width == 0.002
        assert self.S.width == 0.04
        assert self.S.height == 0.06
        assert self.S.energy == 120.0
        assert self.S.phase is True
        assert self.S.jitter == 3.0

        assert self.B.name == "B chopper"
        assert self.B.phase is False
        assert self.B.jitter == 0.0

    def test_pulse_width_and_transmission(self) -> None:
        dt_S, fwhh_S, _ = self.S.pulse_width()
        trans_S = self.S.transmission()
        assert np.isclose(dt_S, 12.192013561396197, rtol=1e-12)
        assert np.isclose(fwhh_S, 27.621358640099585, rtol=1e-12)
        assert np.isclose(trans_S, 0.728215526458732, rtol=1e-12)

        dt_B, fwhh_B, _ = self.B.pulse_width()
        trans_B = self.B.transmission()
        assert np.isclose(dt_B, 2.345746353257828, rtol=1e-12)
        assert np.isclose(fwhh_B, 5.716058325673466, rtol=1e-12)
        assert np.isclose(trans_B, 0.369324618442608, rtol=1e-12)

    def test_array_energy_pulse_width_and_transmission(self) -> None:
        ei_arr = np.array([[100.0, 120.0, 150.0], [200.0, 250.0, 300.0]])
        dt_arr, fwhh_arr, _ = self.S.pulse_width(ei_arr)
        trans_arr = self.S.transmission(ei_arr)
        assert dt_arr.shape == ei_arr.shape
        assert fwhh_arr.shape == ei_arr.shape
        assert trans_arr.shape == ei_arr.shape

        dt_120, fwhh_120, _ = self.S.pulse_width(120.0)
        assert np.isclose(dt_arr[0, 1], dt_120, rtol=1e-12)
        assert np.isclose(fwhh_arr[0, 1], fwhh_120, rtol=1e-12)
        assert np.isclose(trans_arr[0, 1], self.S.transmission(120.0), rtol=1e-12)

    def test_gamma_regime_regimes(self) -> None:
        # Test 0 <= gamma < 1, 1 <= gamma < 4, and gamma >= 4
        fc = IX_fermi_chopper(10.0, 250.0, 0.05, 1.3, 0.002, energy=120.0, phase=True, jitter=0.0)
        # gamma = 0 (at optimum energy: omega = 2 * v * curvature => v = pi*250/1.3)
        v_opt = (2.0 * np.pi * 250.0) / (2.0 * 1.3)
        ei_opt = (v_opt / 437.393377) ** 2
        dt0, fwhh0, _ = fc.pulse_width(ei_opt)
        assert dt0 > 0
        assert np.isclose(fwhh0, np.sqrt(6.0) * dt0, rtol=1e-12)
        assert np.isclose(fc.transmission(ei_opt), 1.0, rtol=1e-12)

        # Very low energy where gamma >= 4 -> zero transmission and pulse width
        dt_closed, fwhh_closed, _ = fc.pulse_width(1.0)
        assert dt_closed == 0.0
        assert fwhh_closed == 0.0
        assert fc.transmission(1.0) == 0.0

    def test_rand_statistics(self) -> None:
        rng = np.random.default_rng(1234)
        X_S = self.S.rand(100000, 1, rng=rng)
        mean_S, sig_mean_S, sig_S, sig_sig_S, _, _, _, _ = sampling_statistics(X_S)
        dt_S, _, _ = self.S.pulse_width()
        assert abs(float(mean_S)) < 3.5 * float(sig_mean_S)
        assert abs(float(sig_S) - dt_S) < 3.5 * float(sig_sig_S)

        X_B = self.B.rand(100000, 1, rng=rng)
        mean_B, sig_mean_B, sig_B, sig_sig_B, _, _, _, _ = sampling_statistics(X_B)
        dt_B, _, _ = self.B.pulse_width()
        assert abs(float(mean_B)) < 3.5 * float(sig_mean_B)
        assert abs(float(sig_B) - dt_B) < 3.5 * float(sig_sig_B)

    def test_serialization_roundtrip(self) -> None:
        d = self.S.to_dict()
        S_rec = IX_fermi_chopper.from_dict(d)
        assert S_rec == self.S
        assert np.isclose(S_rec.pulse_width()[0], self.S.pulse_width()[0], rtol=1e-12)
