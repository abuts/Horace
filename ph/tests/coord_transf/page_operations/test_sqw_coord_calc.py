"""Unit tests for ``PageOp_coord_calc``, ``sqw.coordinates_calc``, and ``sqw.signal``.

Ported from ``_test/test_sqw_pageOpMethods/test_sqw_coord_calc.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.coord_transf.page_operations._utils import compute_bin_data
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.exceptions import HoraceError
from pyhorace.sqw import read_sqw, sqw


TEST_COMMON = Path(__file__).resolve().parents[4] / "_test" / "common_data"


class TestSqwCoordCalc:
    """Ported from ``_test/test_sqw_pageOpMethods/test_sqw_coord_calc.m``."""

    @classmethod
    def setup_class(cls) -> None:
        cls.sqw_1d_path = str(TEST_COMMON / "sqw_1d_1.sqw")
        cls.sqw_2d_path = str(TEST_COMMON / "sqw_2d_1.sqw")
        cls.sqw_1d = read_sqw(cls.sqw_1d_path)
        cls.sqw_2d = read_sqw(cls.sqw_2d_path)

    def test_signal_fb(self) -> None:
        w1modE = self.sqw_2d.coordinates_calc("E")

        hc = hor_config()
        old_mcs = hc.mem_chunk_size
        old_fbs = hc.fb_scale_factor
        try:
            hc.mem_chunk_size = 4000
            hc.fb_scale_factor = 4

            fb_sqw = sqw(self.sqw_2d_path)
            assert fb_sqw.pix.is_filebacked

            w1modE_fb = fb_sqw.coordinates_calc("E")

            tol = float(2 * np.finfo(np.float32).eps)
            ok_d, mess_d = w1modE.data.equal_to_tol(
                w1modE_fb.data, tol=tol, ignore_str=True
            )
            assert ok_d, mess_d
            ok_p, mess_p = w1modE.pix.equal_to_tol(
                w1modE_fb.pix, tol=tol, ignore_str=True
            )
            assert ok_p, mess_p
        finally:
            hc.mem_chunk_size = old_mcs
            hc.fb_scale_factor = old_fbs

    def test_w2E_option(self) -> None:
        w1modE = self.sqw_2d.coordinates_calc("E")

        expected_sig = (
            np.asarray(self.sqw_2d.pix.dE, dtype=np.float64).ravel(order="F")
            + float(np.asarray(self.sqw_2d.data.proj.offset).ravel(order="F")[3])
        )
        exp_s, exp_e, exp_msd = compute_bin_data(
            self.sqw_2d.data.npix,
            expected_sig,
            None,
            no_normalization=False,
            calc_variance=True,
            calc_signal_msd=True,
        )
        np.testing.assert_allclose(w1modE.pix.signal.ravel(order="F"), expected_sig, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modE.pix.variance.ravel(order="F"), exp_msd, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modE.data.s, exp_s, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modE.data.e, exp_e, rtol=1e-7, atol=1e-7)

    def test_w2Q_option(self) -> None:
        w1modQ = self.sqw_2d.coordinates_calc("Q")

        qq = np.asarray(self.sqw_2d.pix.q_coordinates, dtype=np.float64)
        expected_sig = np.linalg.norm(qq, ord=2, axis=0)
        exp_s, exp_e, exp_msd = compute_bin_data(
            self.sqw_2d.data.npix,
            expected_sig,
            None,
            no_normalization=False,
            calc_variance=True,
            calc_signal_msd=True,
        )
        np.testing.assert_allclose(w1modQ.pix.signal.ravel(order="F"), expected_sig, rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(w1modQ.pix.variance.ravel(order="F"), exp_msd, rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(w1modQ.data.s, exp_s, rtol=3e-7, atol=3e-7)
        np.testing.assert_allclose(w1modQ.data.e, exp_e, rtol=3e-7, atol=3e-7)

    def test_w2l_option(self) -> None:
        w2modL = self.sqw_2d.coordinates_calc("l")

        uhkl = self.sqw_2d.data.proj.transform_pix_to_hkl(self.sqw_2d.pix.q_coordinates)
        expected_sig = uhkl[2, :]
        exp_s, exp_e, exp_msd = compute_bin_data(
            self.sqw_2d.data.npix,
            expected_sig,
            None,
            no_normalization=False,
            calc_variance=True,
            calc_signal_msd=True,
        )
        np.testing.assert_allclose(w2modL.pix.signal.ravel(order="F"), expected_sig, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modL.pix.variance.ravel(order="F"), exp_msd, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modL.data.s, exp_s, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modL.data.e, exp_e, rtol=1e-7, atol=1e-7)

    def test_w2d2_option(self) -> None:
        w2modD2 = self.sqw_2d.coordinates_calc("d2")

        pax = np.asarray(self.sqw_2d.data.pax, dtype=np.int64).ravel(order="F")
        dax = np.asarray(self.sqw_2d.data.dax, dtype=np.int64).ravel(order="F")
        get_ind = int(pax[int(dax[1]) - 1]) - 1
        uhkl = self.sqw_2d.data.proj.transform_pix_to_img(self.sqw_2d.pix.q_coordinates)
        expected_sig = uhkl[get_ind, :]
        exp_s, exp_e, exp_msd = compute_bin_data(
            self.sqw_2d.data.npix,
            expected_sig,
            None,
            no_normalization=False,
            calc_variance=True,
            calc_signal_msd=True,
        )
        np.testing.assert_allclose(w2modD2.pix.signal.ravel(order="F"), expected_sig, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modD2.pix.variance.ravel(order="F"), exp_msd, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modD2.data.s, exp_s, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w2modD2.data.e, exp_e, rtol=1e-7, atol=1e-7)

    def test_w1d2_throws(self) -> None:
        with pytest.raises(HoraceError, match="HORACE:sqw:invalid_argument"):
            self.sqw_1d.coordinates_calc("d2")

    def test_w1d1_option(self) -> None:
        w1modP1 = self.sqw_1d.coordinates_calc("d1")

        pax = np.asarray(self.sqw_1d.data.pax, dtype=np.int64).ravel(order="F")
        dax = np.asarray(self.sqw_1d.data.dax, dtype=np.int64).ravel(order="F")
        get_ind = int(pax[int(dax[0]) - 1]) - 1
        uhkl = self.sqw_1d.data.proj.transform_pix_to_img(self.sqw_1d.pix.q_coordinates)
        expected_sig = uhkl[get_ind, :]
        exp_s, exp_e, exp_msd = compute_bin_data(
            self.sqw_1d.data.npix,
            expected_sig,
            None,
            no_normalization=False,
            calc_variance=True,
            calc_signal_msd=True,
        )
        np.testing.assert_allclose(w1modP1.pix.signal.ravel(order="F"), expected_sig, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modP1.pix.variance.ravel(order="F"), exp_msd, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modP1.data.s, exp_s, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(w1modP1.data.e, exp_e, rtol=1e-7, atol=1e-7)
