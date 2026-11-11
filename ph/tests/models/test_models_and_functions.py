"""Unit tests for Horace functions and S(Q, w) models in ``pyhorace.models``."""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.models import (
    bose_times_eps,
    demo_4gauss_2dQ,
    demo_FM_spinwaves_2dSlice_sqw,
    demo_null_func,
    disp_bcc_hfm,
    disp_bcc_hfm0,
    disp_fcc_hfm,
    disp_sc_hfm,
    dsho_over_eps,
    einstein,
    example_sqw_background,
    example_sqw_flat_mode,
    example_sqw_spin_waves,
    rod_sqw,
    sheet_sqw,
    sqw_bcc_hfm,
    sqw_fcc_hfm,
    sqw_sc_hfm,
    template_sqw_from_dispersion,
    van_sqw,
)


class TestMathHelpers:
    """Test ``einstein``, ``bose_times_eps``, and ``dsho_over_eps``."""

    def test_einstein_regimes(self) -> None:
        # Zero limit: y / (1 - exp(-y)) -> 1 as y -> 0
        y_zero = np.array([0.0])
        np.testing.assert_allclose(einstein(y_zero), [1.0], atol=1e-15)

        # Small |y| <= 0.1 (Taylor expansion) vs direct formula at boundary y = 0.05, -0.05
        y_small = np.array([-0.05, -0.01, 0.01, 0.05])
        expected_small = y_small / (1.0 - np.exp(-y_small))
        np.testing.assert_allclose(einstein(y_small), expected_small, atol=1e-14, rtol=1e-14)

        # Large positive and negative |y| > 0.1
        y_big = np.array([-5.0, -0.5, 0.5, 5.0])
        expected_big = y_big / (1.0 - np.exp(-y_big))
        np.testing.assert_allclose(einstein(y_big), expected_big, atol=1e-14, rtol=1e-14)

    def test_bose_times_eps_finite_and_zero_temp(self) -> None:
        eps = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        # At T = 0, y = eps for eps > 0 and 0 for eps <= 0
        y_t0 = bose_times_eps(eps, 0.0)
        np.testing.assert_allclose(y_t0, [0.0, 0.0, 0.0, 1.0, 10.0], atol=1e-15)

        # All positive at T = 0
        eps_pos = np.array([2.0, 5.0])
        np.testing.assert_allclose(bose_times_eps(eps_pos, 0.0), eps_pos, atol=1e-15)

        # Finite T > 0
        T = 10.0
        kB = 8.6173324e-2
        y_t10 = bose_times_eps(eps, T)
        np.testing.assert_allclose(y_t10, (kB * T) * einstein(eps / (kB * T)), atol=1e-14)

    def test_dsho_over_eps(self) -> None:
        en = np.array([0.0, 5.0, 10.0])
        en0 = np.array([5.0, 5.0, 5.0])
        gam = 1.5
        y = dsho_over_eps(en, en0, gam)
        expected = ((4.0 / np.pi) * np.abs(gam * en0)) / (
            (en**2 - en0**2) ** 2 + 4.0 * (gam * en) ** 2
        )
        np.testing.assert_allclose(y, expected, atol=1e-15)


class TestHoraceFunctions:
    """Test functions in ``horace_core/functions``."""

    def test_demo_4gauss_2dQ(self) -> None:
        # [amp, posx, posy, wid, symposx, symposy, bg]
        p = [10.0, 0.5, 0.5, 0.1, 1.0, 2.0, 1.5]
        # Peak 1 at (1.5, 2.0), Peak 2 at (1.0, 1.5), Peak 3 at (0.5, 2.0), Peak 4 at (1.0, 2.5)
        q1 = np.array([1.5, 1.0, 0.5, 1.0, 10.0])
        q2 = np.array([2.0, 1.5, 2.0, 2.5, 10.0])
        y = demo_4gauss_2dQ(q1, q2, p)
        assert y.shape == q1.shape
        # Far from all peaks, y -> bg
        np.testing.assert_allclose(y[4], 1.5, atol=1e-12)
        # Symmetry: all 4 peak centres must have identical intensity
        np.testing.assert_allclose(y[0], y[1], atol=1e-14)
        np.testing.assert_allclose(y[0], y[2], atol=1e-14)
        np.testing.assert_allclose(y[0], y[3], atol=1e-14)
        assert y[0] > 11.5

    def test_demo_FM_spinwaves_2dSlice_sqw(self) -> None:
        # pars = [js, delta, gam, temp, amp]
        pars = [25.0, 2.0, 1.5, 10.0, 3.0]
        qh = np.array([0.0, 0.25, 0.5])
        qk = np.array([0.0, 0.0, 0.0])
        ql = np.array([0.0, 0.0, 0.0])
        en = np.array([2.0, 52.0, 102.0])
        y = demo_FM_spinwaves_2dSlice_sqw(qh, qk, ql, en, pars)
        assert y.shape == qh.shape

        # Manual verification of formula
        omega0 = 2.0 + 100.0 * (np.sin(np.pi * qh) ** 2)
        bose = en / (1.0 - np.exp(-11.602 * en / 10.0))
        expected = 3.0 * (
            bose * (4.0 * 1.5 * omega0) / (np.pi * ((en**2 - omega0**2) ** 2 + 4.0 * (1.5 * en) ** 2))
        )
        np.testing.assert_allclose(y, expected, atol=1e-14)

    def test_demo_null_func(self) -> None:
        qh = np.ones((3, 4))
        qk = np.ones((3, 4))
        ql = np.ones((3, 4))
        en = np.ones((3, 4))
        yout = demo_null_func(qh, qk, ql, en, [1, 2, 3])
        assert yout.shape == (3, 4)
        np.testing.assert_array_equal(yout, np.zeros((3, 4)))


class TestSqwModels:
    """Test models in ``horace_core/sqw_models``."""

    def test_disp_bcc_hfm0_matches_disp_bcc_hfm(self) -> None:
        qh = np.array([0.1, 0.25, 0.4])
        qk = np.array([0.2, 0.15, 0.3])
        ql = np.array([0.05, 0.35, 0.2])
        par = [10.0, 2.5, 5.0, -1.2, 0.8, 0.4, -0.3]

        w0, sf0 = disp_bcc_hfm0(qh, qk, ql, par)
        w1, sf1 = disp_bcc_hfm(qh, qk, ql, par)
        np.testing.assert_allclose(w0, w1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(sf0, sf1, atol=1e-13, rtol=1e-13)

    def test_sqw_bcc_fcc_sc_hfm_and_spin_waves(self) -> None:
        qh = np.array([0.1, 0.2, 0.3])
        qk = np.array([0.15, 0.25, 0.05])
        ql = np.array([0.2, 0.1, 0.4])
        en = np.array([15.0, 25.0, 35.0])
        par = [12.0, 1.5, 10.0, 2.0, 4.0, 1.0, 0.5]

        # sqw_bcc_hfm and example_sqw_spin_waves have identical implementations
        wbcc = sqw_bcc_hfm(qh, qk, ql, en, par)
        wex = example_sqw_spin_waves(qh, qk, ql, en, par)
        np.testing.assert_allclose(wbcc, wex, atol=1e-14)

        # template_sqw_from_dispersion with disp_bcc_hfm matches sqw_bcc_hfm
        wtmpl = template_sqw_from_dispersion(qh, qk, ql, en, par, disp_bcc_hfm)
        np.testing.assert_allclose(wbcc, wtmpl, atol=1e-14)

        # sqw_fcc_hfm and sqw_sc_hfm
        wfcc = sqw_fcc_hfm(qh, qk, ql, en, par)
        wsc = sqw_sc_hfm(qh, qk, ql, en, par)
        assert wfcc.shape == qh.shape
        assert wsc.shape == qh.shape
        assert np.all(wfcc > 0)
        assert np.all(wsc > 0)

        # template_sqw_from_dispersion with disp_sc_hfm (direct ndarray return)
        wtmpl_sc = template_sqw_from_dispersion(qh, qk, ql, en, par, disp_sc_hfm)
        np.testing.assert_allclose(wsc, wtmpl_sc, atol=1e-14)

        # template_sqw_from_dispersion with multi-branch (list of ndarray) dispersion function
        def multi_branch_disp(h: np.ndarray, k: np.ndarray, l: np.ndarray, p: np.ndarray):
            w_arr, sf_arr = disp_sc_hfm(h, k, l, p)
            return [w_arr], [sf_arr]

        wtmpl_multi = template_sqw_from_dispersion(qh, qk, ql, en, par, multi_branch_disp)
        np.testing.assert_allclose(wsc, wtmpl_multi, atol=1e-14)

    def test_van_sqw_and_example_sqw_background(self) -> None:
        qh = np.zeros(5)
        qk = np.zeros(5)
        ql = np.zeros(5)
        en = np.linspace(-1.0, 1.0, 5)
        pars = [10.0, 0.0, 1.0]  # [ampl, en0, fwhh]

        w_van = van_sqw(qh, qk, ql, en, pars)
        w_bg = example_sqw_background(qh, qk, ql, en, pars)
        np.testing.assert_allclose(w_van, w_bg, atol=1e-15)

        # Peak is at en = 0 with value ampl / (sig * sqrt(2*pi)) where sig = fwhh / sqrt(log(256))
        sig = 1.0 / np.sqrt(np.log(256.0))
        np.testing.assert_allclose(w_van[2], 10.0 / (sig * np.sqrt(2.0 * np.pi)), atol=1e-14)
        # At en = +/- 0.5 (half-width at half-maximum), weight is half of peak
        np.testing.assert_allclose(w_van[1], 0.5 * w_van[2], atol=1e-14)
        np.testing.assert_allclose(w_van[3], 0.5 * w_van[2], atol=1e-14)

    def test_example_sqw_flat_mode(self) -> None:
        qh = np.zeros(3)
        qk = np.zeros(3)
        ql = np.zeros(3)
        en = np.array([0.0, 2.0, 5.0])
        # 6-parameter call: [HE, CE, fwhh, H1, G1, T]
        pars6 = [10.0, 2.0, 1.0, 0.5, 0.8, 15.0]
        w6 = example_sqw_flat_mode(qh, qk, ql, en, pars6)
        assert w6.shape == en.shape
        assert np.all(np.isfinite(w6))
        assert np.all(w6 > 0)

        # 3-parameter call (elastic Gaussian only): matches van_sqw
        pars3 = [10.0, 2.0, 1.0]
        w3 = example_sqw_flat_mode(qh, qk, ql, en, pars3)
        np.testing.assert_allclose(w3, van_sqw(qh, qk, ql, en, pars3), atol=1e-14)

    def test_rod_sqw(self) -> None:
        # Cubic lattice with a = 2*pi so bmatrix = I
        a = 2.0 * np.pi
        lattice = [a, a, a, 90.0, 90.0, 90.0]
        pars = [5.0, 0.2]  # [amp, fwhh]
        qh = np.array([[1.0, 1.1], [2.0, 2.0]])
        qk = np.array([[0.0, 0.0], [1.0, 1.0]])
        ql = np.array([[0.0, 0.0], [3.0, 3.5]])
        en = np.zeros_like(qh)

        # Default rlp (nearest integer h, k, l)
        w = rod_sqw(qh, qk, ql, en, pars, lattice)
        assert w.shape == qh.shape
        # At exact integer rlp (1,0,0) and (2,1,3), weight == amp == 5.0
        np.testing.assert_allclose(w[0, 0], 5.0, atol=1e-14)
        np.testing.assert_allclose(w[1, 0], 5.0, atol=1e-14)
        # At (1.1, 0, 0), dq = 0.1 = fwhh/2 => half maximum = 2.5
        np.testing.assert_allclose(w[0, 1], 2.5, atol=1e-14)

        # Explicit rlp list
        rlp = np.array([[1.0, 0.0, 0.0]])
        w_rlp = rod_sqw(qh, qk, ql, en, pars, lattice, rlp)
        assert w_rlp.shape == qh.shape
        np.testing.assert_allclose(w_rlp[0, 0], 5.0, atol=1e-14)
        np.testing.assert_allclose(w_rlp[0, 1], 2.5, atol=1e-14)

    def test_sheet_sqw(self) -> None:
        a = 2.0 * np.pi
        lattice = [a, a, a, 90.0, 90.0, 90.0]
        pars = [8.0, 0.4]  # [amp, fwhh]
        normal = [1.0, 0.0, 0.0]  # Sheet normal along h
        qh = np.array([0.0, 0.2, -0.2])
        qk = np.array([1.5, -3.2, 0.7])
        ql = np.array([-2.1, 4.0, 1.1])
        en = np.array([5.0, 10.0, 15.0])

        w = sheet_sqw(qh, qk, ql, en, pars, lattice, normal)
        assert w.shape == qh.shape
        # At qh = 0, qperp = 0 => weight = amp = 8.0 regardless of qk, ql, en
        np.testing.assert_allclose(w[0], 8.0, atol=1e-14)
        # At qh = +/- 0.2 = +/- fwhh/2 => weight = amp / 2 = 4.0
        np.testing.assert_allclose(w[1], 4.0, atol=1e-14)
        np.testing.assert_allclose(w[2], 4.0, atol=1e-14)

    def test_integration_with_sqw_and_dnd_eval(self) -> None:
        from pathlib import Path
        from pyhorace.sqw import d2d, sqw

        horace_root = Path(__file__).resolve().parents[3]
        sqw_path = horace_root / "_test" / "common_data" / "sqw_2d_1.sqw"
        w2 = sqw(str(sqw_path))
        d2 = d2d(w2)

        # func_eval with demo_4gauss_2dQ on 2D slice
        d2_gauss = d2.func_eval(demo_4gauss_2dQ, [10.0, 0.1, 0.1, 0.05, -0.5, -0.5, 1.0])
        assert np.all(np.isfinite(d2_gauss.s))

        # sqw_eval with demo_null_func sets signal to zero
        w2_null = w2.sqw_eval(demo_null_func, [])
        np.testing.assert_array_equal(w2_null.data.s, np.zeros_like(w2_null.data.s))

        # sqw_eval with demo_FM_spinwaves_2dSlice_sqw
        w2_sw = w2.sqw_eval(demo_FM_spinwaves_2dSlice_sqw, [25.0, 2.0, 1.5, 10.0, 3.0])
        assert np.all(np.isfinite(w2_sw.pix.signal[w2.pix.dE != 0.0]))

        # disp2sqw_eval with disp_bcc_hfm
        w2_disp = w2.disp2sqw_eval(disp_bcc_hfm, [10.0, 2.0, 5.0], 1.5)
        assert np.all(np.isfinite(w2_disp.data.s))

        # sqw_eval with sqw_bcc_hfm and van_sqw
        w2_bcc = w2.sqw_eval(sqw_bcc_hfm, [10.0, 1.5, 10.0, 2.0, 5.0])
        w2_van = w2.sqw_eval(van_sqw, [10.0, 150.0, 20.0])
        assert np.all(np.isfinite(w2_bcc.data.s))
        assert np.all(np.isfinite(w2_van.data.s))

