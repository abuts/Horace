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
            phase=True,
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
        assert self.B.phase is True
        assert self.B.jitter == 0.0

    def test_pulse_width_and_transmission(self) -> None:
        dt_S, pk_fwhh_S, fwhh_S = self.S.pulse_width()
        trans_S = self.S.transmission()
        assert np.isclose(dt_S, 5.2315460104771905, rtol=1e-12)
        assert np.isclose(pk_fwhh_S, 12.732395447351628, rtol=1e-12)
        assert np.isclose(fwhh_S, 13.636062579031469, rtol=1e-12)
        assert np.isclose(trans_S, 0.986567233433285, rtol=1e-12)

    def test_array_energy_pulse_width_and_transmission(self) -> None:
        ei_arr = np.array([[100.0, 120.0, 150.0], [200.0, 250.0, 300.0]])
        dt_arr, pk_fwhh, fwhh_arr = self.S.pulse_width(ei_arr)
        trans_arr = self.S.transmission(ei_arr)
        assert dt_arr.shape == ei_arr.shape
        assert isinstance(pk_fwhh, float)
        assert fwhh_arr.shape == ei_arr.shape
        assert trans_arr.shape == ei_arr.shape

        dt_120, pk_fwhh_120, fwhh_120 = self.S.pulse_width(120.0)
        assert np.isclose(dt_arr[0, 1], dt_120, rtol=1e-12)
        assert np.isclose(pk_fwhh, pk_fwhh_120, rtol=1e-12)
        assert np.isclose(fwhh_arr[0, 1], fwhh_120, rtol=1e-12)
        assert np.isclose(trans_arr[0, 1], self.S.transmission(120.0), rtol=1e-12)

    def test_gamma_regime_regimes(self) -> None:
        # Test 0 <= gamma < 1, 1 <= gamma < 4, and gamma >= 4
        fc = IX_fermi_chopper(
            10.0, 250.0, 0.05, 1.3, 0.002, slit_spacing=0.0025, energy=120.0, phase=True, jitter=0.0
        )
        # gamma = 0 (at optimum energy: s = 2 * omega * curvature => v_opt = 2 * omega * curvature)
        v_opt = 2.0 * (2.0 * np.pi * 250.0) * 1.3
        ei_opt = (v_opt * fc._c_e_to_t / 1e6) ** 2
        dt0, pk_fwhh0, fwhh0 = fc.pulse_width(ei_opt)
        assert dt0 > 0
        assert np.isclose(pk_fwhh0, np.sqrt(6.0) * dt0, rtol=1e-12)
        assert np.isclose(fwhh0, np.sqrt(6.0) * dt0, rtol=1e-12)
        assert np.isclose(fc.transmission(ei_opt), 1.0, rtol=1e-12)

        # Very High energy where gamma >= 4 -> zero transmission and NaN pulse width
        fc_narrow = IX_fermi_chopper(
            10.0, 600.0, 0.049, 1.3, 0.0005, slit_spacing=0.001, energy=50.0, phase=True, jitter=0.0
        )
        dt_closed, _, fwhh_closed = fc_narrow.pulse_width(50.0)
        assert np.isnan(dt_closed)
        assert np.isnan(fwhh_closed)
        assert fc_narrow.transmission(50.0) == 0.0

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

    def test_matlab_suite_methods(self) -> None:
        f500 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 500)
        f200 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 200)
        f163 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 163)
        f162 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 162)
        f100 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 100)
        f50 = IX_fermi_chopper(10, 600, 0.049, 1.3, 0.0028, 0.1, 0.2, 0.9, 50)

        # test_hashable_prop
        f500, hash_val, _ = f500.build_hash()
        assert f500.hash_defined
        assert f500.hash_value == hash_val
        f500.slit_width = 0.01
        assert not f500.hash_defined
        f500.slit_width = 0.0028

        # test_pulse_shape
        t = np.arange(-12.0, 13.0, 1.0)
        y1 = f500.pulse_shape(t, 200)
        y2 = f200.pulse_shape(t)
        np.testing.assert_allclose(y1, y2)

        # test_auto_pulse_shape
        y1_auto, t1_auto = f500.pulse_shape(200, return_t=True)
        y2_auto, t2_auto = f200.pulse_shape()
        assert y2_auto.shape == (500,)
        assert t2_auto.shape == (500,)

        # test_pdf
        pdf = f200.pdf
        assert np.isclose(pdf.var()[0], f200.pulse_width()[0] ** 2, atol=1e-2, rtol=1e-2)

        # test_default_pulse_shape
        f0 = IX_fermi_chopper()
        y0, t0 = f0.pulse_shape()
        assert y0 == np.inf
        assert t0 == 0.0
        t_arr = np.arange(-2.0, 3.0, 1.0)
        y_arr = f0.pulse_shape(t_arr)
        np.testing.assert_array_equal(y_arr, np.array([0.0, 0.0, np.inf, 0.0, 0.0]))

        # test_default_pulse_range
        tlo, thi = f0.pulse_range()
        assert tlo == 0.0
        assert thi == 0.0

        # test_default_partial_transmission
        T = f0.partial_transmission(np.array([-1e-15, 0.0, 1e-15]))
        np.testing.assert_array_equal(T, np.array([0.0, 0.0, 1.0]))
        assert f163.transmission() > 0
        assert f162.transmission() > 0
        assert f100.transmission() > 0
        assert f50.transmission() > 0

    def test_serialization_roundtrip(self) -> None:
        d = self.S.to_dict()
        S_rec = IX_fermi_chopper.from_dict(d)
        assert S_rec == self.S
        assert np.isclose(S_rec.pulse_width()[0], self.S.pulse_width()[0], rtol=1e-12)
