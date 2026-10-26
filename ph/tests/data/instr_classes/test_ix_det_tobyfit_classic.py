"""Unit tests for IX_det_TobyfitClassic translated from test_IX_det_TobyfitClassic.m."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_tobyfit_classic import IX_det_TobyfitClassic

REF_MAT_FILE = (
    Path(__file__).resolve().parents[4]
    / "_test"
    / "test_instrument_classes"
    / "test_IX_det_TobyfitClassic_output.mat"
)


def _load_ref_mat() -> dict[str, Any]:
    if REF_MAT_FILE.is_file():
        try:
            return sio.loadmat(str(REF_MAT_FILE), simplify_cells=True)
        except Exception:
            return {}
    return {}


def _check_ref(
    mat_data: dict[str, Any],
    method_name: str,
    var_name: str,
    actual: Any,
    atol: float = 1e-13,
    rtol: float = 1e-13,
) -> None:
    """Compare actual value with saved MATLAB reference if present and numeric."""
    block = mat_data.get(method_name)
    if block is None and "test_IX_det_TobyfitClassic" in mat_data:
        cls_block = mat_data["test_IX_det_TobyfitClassic"]
        if isinstance(cls_block, dict):
            block = cls_block.get(method_name)
    if isinstance(block, dict) and var_name in block:
        expected = block[var_name]
        if isinstance(expected, (int, float, np.number, np.ndarray)) and not (
            isinstance(expected, np.ndarray) and expected.dtype == object
        ):
            np.testing.assert_allclose(
                np.asarray(actual, dtype=float).ravel(order="F"),
                np.asarray(expected, dtype=float).ravel(order="F"),
                atol=atol,
                rtol=rtol,
            )


def sampling_statistics(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute sample mean, standard deviation, covariance, and their standard errors."""
    N = X.shape[0]
    Xmean = np.mean(X, axis=0)
    Xcov = np.cov(X, rowvar=False, ddof=1)
    Xsig = np.sqrt(np.diag(Xcov))

    mean_val = Xmean
    sig_val = Xsig
    cov_val = Xcov
    sig_mean = Xsig / np.sqrt(max(1, N - 1))

    dX = X - Xmean
    mu4 = np.mean(dX**4, axis=0)
    var_var = (mu4 - ((N - 3.0) / (N - 1.0)) * (Xsig**4)) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_sig = np.where(Xsig > 0, np.sqrt(np.maximum(var_var, 0.0)) / (2.0 * Xsig), 0.0)

    dX2 = dX**2
    mean_dX2Y2 = (dX2.T @ dX2) / N
    var_cov = (mean_dX2Y2 - Xcov**2) / N
    sig_cov = np.sqrt(np.maximum(var_cov, 0.0))
    return mean_val, sig_mean, sig_val, sig_sig, cov_val, sig_cov


def validate_det_rand(
    det: Any, nsamples: int, ntol: float, npath: Any, wvec: float
) -> tuple[bool, str]:
    """Validate detector random point generation against analytical mean and covariance."""
    npath_arr = np.asarray(npath, dtype=float)
    npath_arr = npath_arr / np.linalg.norm(npath_arr)
    X = det.rand(np.ones((int(nsamples), 1), dtype=int), npath_arr, wvec)

    mean_val, sig_mean, sig_val, sig_sig, cov_val, sig_cov = sampling_statistics(X.T)

    mean_pos = np.asarray(det.mean(npath_arr, wvec), dtype=float).ravel()
    cov_pos = np.asarray(det.covariance(npath_arr, wvec), dtype=float)
    sig_pos = np.sqrt(np.diag(cov_pos))

    ok = True
    mess = ""
    if not np.all(np.abs(mean_val - mean_pos) < ntol * sig_mean):
        ok = False
        mess += "mean point of absorption from random sampling is outside tolerance\n"
    if not np.all(np.abs(sig_val - sig_pos) < ntol * sig_sig):
        ok = False
        mess += "std of absorption from random sampling is outside tolerance\n"
    if not np.all(np.abs(cov_val - cov_pos) < ntol * sig_cov):
        ok = False
        mess += "covariance of absorption from random sampling is outside tolerance\n"
    return ok, mess


class TestIXDetTobyfitClassic:
    """Test suite for IX_det_TobyfitClassic."""

    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
        self.dia = np.array([0.0254, 0.0300, 0.0400, 0.0400, 0.0400], dtype=float)
        self.height = np.array([0.015, 0.025, 0.035, 0.035, 0.035], dtype=float)
        th = np.array([np.pi / 2.0, 0.9, 0.775, 0.775, 0.775], dtype=float)
        self.path = np.vstack([np.sin(th), np.zeros_like(th), np.cos(th)])
        self.seed = 0
        np.random.seed(self.seed)
        self.mat_data = _load_ref_mat()

    def construct_detectors(self) -> tuple[list[IX_det_TobyfitClassic], IX_det_TobyfitClassic]:
        dets = [
            IX_det_TobyfitClassic(self.dia[i], self.height[i])
            for i in range(5)
        ]
        det_array = IX_det_TobyfitClassic(self.dia, self.height)
        return dets, det_array

    # ------------------------------------------------------------------
    # Constructor tests
    # ------------------------------------------------------------------
    def test_det_constructor(self) -> None:
        _, det_array = self.construct_detectors()
        np.testing.assert_array_equal(self.dia.reshape(-1, 1), det_array.dia)
        np.testing.assert_array_equal(self.height.reshape(-1, 1), det_array.height)
        assert det_array.ndet == 5

    def test_det_constructor_argExpand(self) -> None:
        val = 0.00344
        det_array = IX_det_TobyfitClassic(val, self.height)
        np.testing.assert_array_equal(
            np.full((5, 1), val, dtype=float), det_array.dia
        )
        np.testing.assert_array_equal(self.height.reshape(-1, 1), det_array.height)

    def test_det_constructor_tooFewArgs_THROW(self) -> None:
        with pytest.raises(ValueError):
            IX_det_TobyfitClassic(self.dia)

    def test_det_constructor_tooManyArgs_THROW(self) -> None:
        with pytest.raises(ValueError, match="IX_det_TobyfitClassic"):
            IX_det_TobyfitClassic(self.dia, self.height, 0.1)

    def test_det_constructor_mixPosKeywrdArgs(self) -> None:
        _, det_array_ref = self.construct_detectors()
        det_array = IX_det_TobyfitClassic(self.dia, "height", self.height)
        assert det_array_ref == det_array

    # ------------------------------------------------------------------
    # Method tests
    # ------------------------------------------------------------------
    def test_effic_allDets_singleWvec(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = 10.0
        effs = np.array(
            [dets[i].effic(self.path[:, i], wvec) for i in range(5)],
            dtype=float,
        ).reshape(1, 5)
        eff_array = det_array.effic(self.path, wvec)

        np.testing.assert_allclose(np.ones((1, 5)), eff_array, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(
            self.mat_data, "test_effic_allDets_singleWvec", "eff_array", eff_array
        )

    def test_effic_arrInd_singleWvec(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = 10.0
        effs = np.array(
            [dets[i].effic(self.path[:, i], wvec) for i in range(5)],
            dtype=float,
        )
        ind = np.array([3, 1, 4, 1], dtype=int)
        ind_0 = ind - 1
        eff_array = det_array.effic(ind, self.path[:, ind_0], wvec)

        np.testing.assert_allclose(
            effs[ind_0].reshape(1, 4), eff_array, atol=1e-13, rtol=1e-13
        )
        np.testing.assert_allclose(np.ones((1, 4)), eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(
            self.mat_data, "test_effic_arrInd_singleWvec", "eff_array", eff_array
        )

    def test_effic_singleInd_arrWvec(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = np.array([10.0, 9.0, 8.0, 7.0, 6.0], dtype=float)
        effs = np.array(
            [dets[3].effic(self.path[:, 3], wvec[i]) for i in range(5)],
            dtype=float,
        ).reshape(1, 5)
        ind = np.array([4, 4, 4, 4, 4], dtype=int)
        eff_array = det_array.effic(ind, self.path[:, ind - 1], wvec)

        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(np.ones((1, 5)), eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(
            self.mat_data, "test_effic_singleInd_arrWvec", "eff_array", eff_array
        )

    def test_effic_arrInd_arrWvec(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = np.array([10.0, 9.0, 8.0, 7.0, 6.0], dtype=float)
        effs = np.array(
            [dets[i].effic(self.path[:, i], wvec[i]) for i in range(5)],
            dtype=float,
        )
        ind = np.array([3, 1, 4, 1], dtype=int)
        ind_0 = ind - 1
        eff_array = det_array.effic(ind, self.path[:, ind_0], wvec[ind_0])

        np.testing.assert_allclose(
            effs[ind_0].reshape(1, 4), eff_array, atol=1e-13, rtol=1e-13
        )
        np.testing.assert_allclose(np.ones((1, 4)), eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(
            self.mat_data, "test_effic_arrInd_arrWvec", "eff_array", eff_array
        )

    def test_mean_d(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = 10.0
        means_d = np.array(
            [dets[i].mean_d(self.path[:, i], wvec) for i in range(5)],
            dtype=float,
        ).reshape(1, 5)
        mean_d_array = det_array.mean_d(self.path, wvec)

        np.testing.assert_allclose(means_d, mean_d_array, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(np.zeros((1, 5)), mean_d_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_mean_d", "mean_d_array", mean_d_array)

    def test_var_d(self) -> None:
        dets, det_array = self.construct_detectors()
        wvec = 10.0
        vars_d = np.array(
            [dets[i].var_d(self.path[:, i], wvec) for i in range(5)],
            dtype=float,
        ).reshape(1, 5)
        var_d_array = det_array.var_d(self.path, wvec)

        np.testing.assert_allclose(vars_d, var_d_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_var_d", "var_d_array", var_d_array)

    def test_random_points(self) -> None:
        dia0 = 0.01
        height0 = 0.03
        det = IX_det_TobyfitClassic(dia0, height0)

        nsamples = int(1e5)
        ntol = 4.0
        npath = [1.0, 1.0, 0.0]
        wvec = 1.0
        ok, mess = validate_det_rand(det, nsamples, ntol, npath, wvec)
        assert ok, mess

    def test_serialization_and_reorder_replicate(self) -> None:
        _, det_array = self.construct_detectors()
        s = det_array.to_struct()
        restored = Serializable.from_struct(s)
        assert restored == det_array

        det_single = IX_det_TobyfitClassic(0.0254, 0.015)
        rep = det_single.replicate(3)
        assert rep.ndet == 3
        np.testing.assert_allclose(rep.dia, np.full((3, 1), 0.0254))

        reord = det_array.reorder([5, 4, 3, 2, 1])
        np.testing.assert_allclose(reord.dia, det_array.dia[::-1])
