"""Unit tests for IX_mod_shape_mono.

Converted from:
  _test/test_instrument_classes/test_IX_mod_shape_mono.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_doubledisk_chopper import IX_doubledisk_chopper
from pyhorace.data.instr_classes.ix_mod_shape_mono import IX_mod_shape_mono
from pyhorace.data.instr_classes.ix_moderator import IX_moderator
from pyhorace.data.instr_classes.pdf_table import sampling_statistics


def _make_let_components():
    """Construct LET moderator, shaping chopper and mono chopper matching let_instrument_struct_for_tests.m."""
    moderator = IX_moderator(
        23.5,
        0.0,
        "ikcarp",
        [33.876, 80.0, 0.0],
        temperature=25.0,
        thickness=0.03,
        width=0.04,
        height=0.04,
        energy=3.7,
        name="Hydrogen",
    )
    shaping_chopper = IX_doubledisk_chopper(
        15.67,
        70.0,
        0.29,
        0.031,
        aperture_width=0.04,
        aperture_height=0.056,
        jitter=0.0,
        name="Pulse shaping",
    )
    mono_chopper = IX_doubledisk_chopper(
        1.5,
        240.0,
        0.29,
        0.015,
        aperture_width=0.04,
        aperture_height=0.045,
        jitter=0.0,
        name="Monochromating",
    )
    return moderator, shaping_chopper, mono_chopper


class TestIXModShapeMono:
    """Test suite for IX_mod_shape_mono."""

    def setup_method(self) -> None:
        self.mod, self.shape_ch, self.mono_ch = _make_let_components()
        self.msm = IX_mod_shape_mono(self.mod, self.shape_ch, self.mono_ch)

    def test_empty_constructor(self) -> None:
        m = IX_mod_shape_mono()
        assert m.shaped_mod is False
        assert np.all(m.t_m_offset == 0.0)

    def test_constructor_and_shaped_mod(self) -> None:
        # For LET parameters at 3.7 meV:
        # Moderator FWHH ~ 115 us, shaping chopper FWHH ~ 228 us -> shaped_mod is False
        assert self.msm.shaped_mod is False
        assert self.msm.energy == 3.7

        # Increasing shaping chopper frequency to 300 Hz narrows shaping chopper below moderator FWHH
        msm2 = IX_mod_shape_mono(self.msm)
        msm2.shaping_chopper.frequency = 300.0
        assert msm2.shaped_mod is True

    def test_analytic_and_numerical_moments_vs_monte_carlo(self) -> None:
        rng = np.random.default_rng(123456)
        mc_cases = [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
        for mc in mc_cases:
            mean_calc = self.msm.mean(mc).ravel()
            cov_calc = self.msm.covariance(mc)
            X = self.msm.rand([50000, 1], mc=mc, rng=rng)
            assert X.shape == (2, 50000)

            mean_mc, sig_mean, _, _, cov_mc, sig_cov, _, _ = sampling_statistics(X.T)
            mean_mc = np.asarray(mean_mc, dtype=float).ravel()
            sig_mean = np.asarray(sig_mean, dtype=float).ravel()
            cov_mc = np.asarray(cov_mc, dtype=float)
            sig_cov = np.asarray(sig_cov, dtype=float)

            for i in range(2):
                assert abs(mean_mc[i] - mean_calc[i]) <= 4.0 * sig_mean[i] + 0.5
                for j in range(2):
                    assert abs(cov_mc[i, j] - cov_calc[i, j]) <= 4.0 * sig_cov[i, j] + 5.0

    def test_shaped_mod_true_moments_vs_monte_carlo(self) -> None:
        msm_wide = IX_mod_shape_mono(self.msm)
        msm_wide.moderator.pp[0] = 200.0  # Wide moderator -> shaped_mod == True
        assert msm_wide.shaped_mod is True

        rng = np.random.default_rng(654321)
        mean_calc = msm_wide.mean([1, 1, 1]).ravel()
        cov_calc = msm_wide.covariance([1, 1, 1])
        X = msm_wide.rand([40000, 1], mc=[1, 1, 1], rng=rng)

        mean_mc, sig_mean, _, _, cov_mc, sig_cov, _, _ = sampling_statistics(X.T)
        mean_mc = np.asarray(mean_mc, dtype=float).ravel()
        sig_mean = np.asarray(sig_mean, dtype=float).ravel()
        cov_mc = np.asarray(cov_mc, dtype=float)
        sig_cov = np.asarray(sig_cov, dtype=float)

        for i in range(2):
            assert abs(mean_mc[i] - mean_calc[i]) <= 4.0 * sig_mean[i] + 1.0
            for j in range(2):
                assert abs(cov_mc[i, j] - cov_calc[i, j]) <= 4.0 * sig_cov[i, j] + 15.0

    def test_serialization_roundtrip(self) -> None:
        d = self.msm.to_dict()
        msm_rec = IX_mod_shape_mono.from_dict(d)
        assert msm_rec == self.msm
        assert np.allclose(msm_rec.mean(), self.msm.mean(), rtol=1e-10)
        assert np.allclose(msm_rec.covariance(), self.msm.covariance(), rtol=1e-10)
