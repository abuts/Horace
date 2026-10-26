"""Unit tests for IX_doubledisk_chopper.

Converted from:
  _test/test_instrument_classes/test_IX_doubledisk_chopper.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_doubledisk_chopper import IX_doubledisk_chopper
from pyhorace.data.instr_classes.pdf_table import sampling_statistics


class TestIXDoublediskChopper:
    """Test suite for IX_doubledisk_chopper."""

    def setup_method(self) -> None:
        # Three regimes:
        # C1: slot_width < aperture_width
        self.C1 = IX_doubledisk_chopper(
            15.0,
            200.0,
            0.35,
            0.03,
            aperture_width=0.05,
            aperture_height=0.06,
            jitter=4.0,
            name="Chopper 1",
        )
        # C2: slot_width == aperture_width (triangle with jitter)
        self.C2 = IX_doubledisk_chopper(
            12.0,
            150.0,
            0.25,
            0.04,
            aperture_width=0.04,
            aperture_height=0.05,
            jitter=0.0,
            name="Chopper 2",
        )
        # C3: slot_width > aperture_width
        self.C3 = IX_doubledisk_chopper(
            18.0,
            250.0,
            0.30,
            0.06,
            aperture_width=0.04,
            aperture_height=0.05,
            jitter=2.5,
            name="Chopper 3",
        )

    def test_empty_constructor(self) -> None:
        dc = IX_doubledisk_chopper()
        assert dc.name == ""
        assert dc.distance == 0.0
        assert dc.frequency == 0.0
        assert dc.radius == 0.0
        assert dc.slot_width == 0.0
        assert dc.aperture_width == 0.0
        assert dc.aperture_height == 0.0
        assert dc.jitter == 0.0

    def test_properties_and_aperture_width_default(self) -> None:
        # When aperture_width is not given, it defaults to slot_width
        dc = IX_doubledisk_chopper(15.0, 200.0, 0.35, 0.03)
        assert dc.slot_width == 0.03
        assert dc.aperture_width == 0.03

        # Changing slot_width when aperture_width was not explicitly set updates aperture_width
        dc.slot_width = 0.045
        assert dc.aperture_width == 0.045

        # Setting aperture_width explicitly decouples it from slot_width
        dc.aperture_width = 0.06
        dc.slot_width = 0.03
        assert dc.aperture_width == 0.06

    def test_pulse_width_and_transmission(self) -> None:
        for ch in (self.C1, self.C2, self.C3):
            dt, fwhh = ch.pulse_width()
            tlo, thi = ch.pulse_range()
            trans = ch.transmission()
            assert dt > 0
            assert fwhh > 0
            assert tlo < 0 < thi
            assert np.isclose(tlo, -thi, rtol=1e-12)
            assert np.isclose(thi, ch.half_width, rtol=1e-12)
            assert 0 < trans <= 1.0

        # C2 has slot_width == aperture_width and jitter == 0:
        # omega = 2*pi*150, tau_s = 1e6 * 0.04 / (2 * 0.25 * omega)
        omega2 = 2.0 * np.pi * 150.0
        tau2 = 1e6 * 0.04 / (2.0 * 0.25 * omega2)
        dt2, fwhh2 = self.C2.pulse_width()
        tlo2, thi2 = self.C2.pulse_range()
        assert np.isclose(thi2, tau2, rtol=1e-12)
        assert np.isclose(fwhh2, tau2, rtol=1e-12)
        assert np.isclose(dt2, tau2 / np.sqrt(6.0), rtol=1e-12)

    def test_rand_statistics(self) -> None:
        rng = np.random.default_rng(9876)
        for ch in (self.C1, self.C2, self.C3):
            X = ch.rand(100000, 1, rng=rng)
            mean_val, sig_mean, sig_val, sig_sig, _, _, _, _ = sampling_statistics(X)
            dt_ref = ch.pulse_width()[0]
            assert abs(float(mean_val)) < 3.5 * float(sig_mean)
            assert abs(float(sig_val) - dt_ref) < 3.5 * float(sig_sig)

    def test_serialization_roundtrip(self) -> None:
        for ch in (self.C1, self.C2, self.C3):
            d = ch.to_dict()
            ch_rec = IX_doubledisk_chopper.from_dict(d)
            assert ch_rec == ch
            assert np.allclose(ch_rec.pulse_width(), ch.pulse_width(), rtol=1e-12)
