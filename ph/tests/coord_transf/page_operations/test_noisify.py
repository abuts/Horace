"""Tests for ``PageOp_noisify`` and ``sqw.noisify``.

Ported from ``_test/test_sqw_pageOpMethods/test_noisify.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pyhorace.coord_transf.page_operations.page_op_noisify import noisify
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.sqw import read_sqw, sqw


REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DATA_DIR = REPO_ROOT / "_test" / "common_data"
SQW_FILE_PATH = str(COMMON_DATA_DIR / "deterministic_sqw_fake_data_for_testing.sqw")


class DeterministicPseudorandomSequence:
    """Port of ``_test/common_functions/deterministic_pseudorandom_sequence.m``."""

    def __init__(self) -> None:
        self.init: int = 1
        self.got_to: int = -666

    def myrand(self, data_length: tuple[int, ...] | list[int]) -> np.ndarray:
        if self.init == 1:
            self.init = 0
            self.got_to = 1
        length = int(data_length[1])
        vals = (np.arange(self.got_to, self.got_to + length, dtype=np.float64) % 1000) * 1e-3
        self.got_to = self.got_to + length
        return vals

    def reset(self) -> None:
        self.init = 1
        self.got_to = -666


class TestNoisify:
    """Test suite for ``sqw.noisify`` and ``noisify`` parser."""

    def test_noisify_returns_equivalent_sqw_for_paged_pixel_data(self) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        old_fb_scale = hc.fb_scale_factor
        try:
            hc.mem_chunk_size = int(100337 // 5)
            hc.fb_scale_factor = 3

            noise_factor = 1.0 / 999.0
            sqw_obj1 = sqw(SQW_FILE_PATH, file_backed=True)
            assert sqw_obj1.pix.num_pixels > sqw_obj1.pix.page_size

            seq = DeterministicPseudorandomSequence()
            myrng = seq.myrand

            noisy_obj1 = sqw_obj1.noisify(
                noise_factor, "random_number_function", myrng
            )

            sqw_obj2 = read_sqw(SQW_FILE_PATH)
            seq.reset()
            noisy_obj2 = sqw_obj2.noisify(
                noise_factor, "random_number_function", myrng
            )

            ok1, msg1 = sqw_obj1.equal_to_tol(
                sqw_obj2, tol=[5e-4, 5e-4], ignore_str=True, ignore_date=True
            )
            assert ok1, msg1

            ok2, msg2 = noisy_obj1.equal_to_tol(
                noisy_obj2, tol=[5e-4, 5e-4], ignore_str=True, ignore_date=True
            )
            assert ok2, msg2

            ok_diff1, _ = sqw_obj1.equal_to_tol(noisy_obj1, 5e-4)
            assert not ok_diff1
            ok_diff2, _ = sqw_obj2.equal_to_tol(noisy_obj2, 5e-4)
            assert not ok_diff2

            assert not np.allclose(sqw_obj1.data.s, noisy_obj1.data.s, atol=5e-4, rtol=5e-4)
            assert not np.allclose(sqw_obj2.data.s, noisy_obj2.data.s, atol=5e-4, rtol=5e-4)
        finally:
            hc.mem_chunk_size = old_chunk
            hc.fb_scale_factor = old_fb_scale

    def test_noisify_adds_gaussian_noise_to_data_with_given_stddev(self) -> None:
        old_state = np.random.get_state()
        try:
            np.random.seed(0)
            sqw_obj = sqw(SQW_FILE_PATH)
            assert sqw_obj.pix.page_size == sqw_obj.pix.num_pixels

            noise_factor = 0.01
            noisy_obj = sqw_obj.noisify(noise_factor)

            original_signal = np.asarray(sqw_obj.pix.signal, dtype=np.float64).ravel(order="F")
            noisy_signal = np.asarray(noisy_obj.pix.signal, dtype=np.float64).ravel(order="F")
            signal_diff = original_signal - noisy_signal

            n = signal_diff.size
            mu_hat = float(np.mean(signal_diff))
            sigma_hat = float(np.std(signal_diff, ddof=1))

            # 95% confidence intervals matching MATLAB fitdist(..., 'normal') / paramci
            t_crit = float(stats.t.ppf(0.975, df=n - 1))
            se_mu = sigma_hat / np.sqrt(n)
            mu_interval = (mu_hat - t_crit * se_mu, mu_hat + t_crit * se_mu)
            assert mu_interval[1] - mu_interval[0] < 2.0
            assert mu_interval[0] <= 0.0 <= mu_interval[1]

            chi2_low = float(stats.chi2.ppf(0.975, df=n - 1))
            chi2_high = float(stats.chi2.ppf(0.025, df=n - 1))
            sigma_interval = (
                sigma_hat * np.sqrt((n - 1) / chi2_low),
                sigma_hat * np.sqrt((n - 1) / chi2_high),
            )
            expected_stddev = float(noise_factor * np.max(original_signal))
            assert sigma_interval[1] - sigma_interval[0] < expected_stddev / 10.0
            assert sigma_interval[0] <= expected_stddev <= sigma_interval[1]
        finally:
            np.random.set_state(old_state)

    def test_npoisify_parser_maxval_frac(self) -> None:
        _, _, outpar = noisify([], [], 0.01, "maximum_value", 1)
        assert "driven_mode" in outpar
        assert outpar["fac"] == 0.01
        assert not outpar["is_poisson"]
        assert outpar["ymax"] == 1
        assert outpar["randfunc"].__name__ == "randn"

    def test_npoisify_parser_maxval(self) -> None:
        _, _, outpar = noisify([], [], "maximum_value", 1)
        assert "driven_mode" in outpar
        assert outpar["fac"] == 0.1
        assert not outpar["is_poisson"]
        assert outpar["ymax"] == 1
        assert outpar["randfunc"].__name__ == "randn"

    def test_npoisify_parser_poisson(self) -> None:
        _, _, outpar = noisify([], [], "poisson")
        assert "driven_mode" in outpar
        assert outpar["fac"] == 0.1
        assert outpar["is_poisson"]
        assert outpar["ymax"] == -np.inf
        assert outpar["randfunc"].__name__ == "randn"

    def test_npoisify_parser_ymax(self) -> None:
        _, _, outpar = noisify([], [], 0.01)
        assert "driven_mode" in outpar
        assert outpar["fac"] == 0.01
        assert not outpar["is_poisson"]
        assert np.asarray(outpar["ymax"]).size == 0
        assert outpar["randfunc"].__name__ == "randn"

    def test_npoisify_parser_default(self) -> None:
        _, _, outpar = noisify([], [])
        assert "driven_mode" in outpar
        assert outpar["fac"] == 0.1
        assert not outpar["is_poisson"]
        assert np.asarray(outpar["ymax"]).size == 0
        assert outpar["randfunc"].__name__ == "randn"
