"""Unit tests for IX_divergence_profile.

Converted from:
  _test/test_instrument_classes/test_IX_divergence_profile.m
"""
import numpy as np
import pytest

from pyhorace.data.instr_classes.ix_divergence_profile import IX_divergence_profile
from pyhorace.data.instr_classes.pdf_table import pdf_table, sampling_statistics


class TestIXDivergenceProfile:
    """Test suite for IX_divergence_profile."""

    def setup_method(self) -> None:
        self.ang1 = np.array([-3.0, -2.0, 0.0, 2.0, 3.0]) / 100.0
        self.prof1 = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        self.pdf1 = pdf_table(self.ang1, self.prof1)

        self.ang2 = np.array([-3.0, -1.0, 1.0, 2.0, 4.0]) / 100.0
        self.prof2 = np.array([0.0, 1.0, 1.0, 2.0, 0.0])
        self.pdf2 = pdf_table(self.ang2, self.prof2)

    def test_empty_constructor(self) -> None:
        div = IX_divergence_profile()
        assert div.name == ""
        assert div.angle.size == 0
        assert div.profile.size == 0

    def test_construct_key_val_1(self) -> None:
        div = IX_divergence_profile(self.ang1, profile=self.prof1)
        assert np.allclose(div.angle.ravel(), self.ang1)
        assert np.allclose(div.profile.ravel(), self.prof1)

    def test_construct_key_val_2(self) -> None:
        div = IX_divergence_profile(self.ang1, self.prof1, name="div1")
        assert div.name == "div1"
        assert np.allclose(div.angle.ravel(), self.ang1)
        assert np.allclose(div.profile.ravel(), self.prof1)

    def test_construct_full(self) -> None:
        div1 = IX_divergence_profile(self.ang1, self.prof1, "div1")
        assert div1.name == "div1"
        assert np.allclose(div1.angle.ravel(), self.ang1)
        assert np.allclose(div1.profile.ravel(), self.prof1)
        assert div1.pdf == self.pdf1

    def test_mutation_updates_pdf(self) -> None:
        div = IX_divergence_profile(self.ang1, self.prof1, "div1")
        assert div.pdf == self.pdf1

        div.angle = self.ang2
        div.profile = self.prof2
        assert div.pdf == self.pdf2

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError):
            IX_divergence_profile(np.array([0.03, 0.01, 0.02]), np.array([1.0, 1.0, 1.0]))
        with pytest.raises(ValueError):
            IX_divergence_profile(self.ang1, np.array([0.0, -1.0, 1.0, 1.0, 0.0]))
        with pytest.raises(ValueError):
            IX_divergence_profile(self.ang1, np.zeros_like(self.ang1))
        with pytest.raises(ValueError):
            IX_divergence_profile(self.ang1, np.array([1.0, 2.0]))

    def test_width_and_mean_and_var(self) -> None:
        div1 = IX_divergence_profile(self.ang1, self.prof1, "div1")
        w, xmax, xlo, xhi = div1.width(0.5)
        w_ref, xmax_ref, xlo_ref, xhi_ref = self.pdf1.width(0.5)
        assert np.isclose(w, w_ref, rtol=1e-12)
        assert np.isclose(xmax, xmax_ref, rtol=1e-12)
        assert np.isclose(xlo, xlo_ref, rtol=1e-12)
        assert np.isclose(xhi, xhi_ref, rtol=1e-12)
        assert np.isclose(div1.mean(), self.pdf1.mean(), rtol=1e-12)
        var1, av1 = div1.var()
        var_ref, av_ref = self.pdf1.var()
        assert np.isclose(var1, var_ref, rtol=1e-12)
        assert np.isclose(av1, av_ref, rtol=1e-12)

    def test_rand_statistics(self) -> None:
        div2 = IX_divergence_profile(self.ang2, self.prof2, "div2")
        rng = np.random.default_rng(42)
        X = div2.rand(100000, 1, rng=rng)
        assert X.shape == (100000, 1)

        mean_val, sig_mean, sig_val, sig_sig, _, _, _, _ = sampling_statistics(X)
        var_ref, mean_ref = self.pdf2.var()
        sig_ref = np.sqrt(var_ref)

        assert abs(float(mean_val) - mean_ref) < 3.5 * float(sig_mean)
        assert abs(float(sig_val) - sig_ref) < 3.5 * float(sig_sig)

    def test_serialization_roundtrip(self) -> None:
        div1 = IX_divergence_profile(self.ang1, self.prof1, "div1")
        s = div1.to_dict()
        div_rec = IX_divergence_profile.from_dict(s)
        assert div_rec == div1
        assert div_rec.pdf == div1.pdf
