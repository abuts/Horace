"""Unit tests for IX_det_He3tube translated from test_IX_det_He3tube.m."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_he3tube import IX_det_He3tube
from tests.data.instr_classes.test_ix_det_tobyfit_classic import validate_det_rand

REF_MAT_FILE = (
    Path(__file__).resolve().parents[4]
    / "_test"
    / "test_instrument_classes"
    / "test_IX_det_He3tube_output.mat"
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
    block = mat_data.get(method_name)
    if block is None and "test_IX_det_He3tube" in mat_data:
        cls_block = mat_data["test_IX_det_He3tube"]
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


class TestIXDetHe3tube:
    """Test suite for IX_det_He3tube."""

    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
        self.dia = np.array([0.0254, 0.0300, 0.0400, 0.0400, 0.0400], dtype=float)
        self.height = np.array([0.015, 0.025, 0.035, 0.035, 0.035], dtype=float)
        self.wall = np.array([6.35e-4, 10.0e-4, 15.0e-4, 15.0e-4, 15.0e-4], dtype=float)
        self.atms = np.array([10.0, 6.0, 4.0, 7.0, 9.0], dtype=float)
        th = np.array([np.pi / 2.0, 0.9, 0.775, 0.775, 0.775], dtype=float)
        self.path = np.vstack([np.sin(th), np.zeros_like(th), np.cos(th)])
        self.seed = 0
        np.random.seed(self.seed)
        self.mat_data = _load_ref_mat()

    def construct_detectors(self) -> tuple[list[IX_det_He3tube], IX_det_He3tube]:
        dets = [
            IX_det_He3tube(self.dia[i], self.height[i], self.wall[i], self.atms[i])
            for i in range(5)
        ]
        det_array = IX_det_He3tube(self.dia, self.height, self.wall, self.atms)
        return dets, det_array

    # ------------------------------------------------------------------
    # Constructor tests
    # ------------------------------------------------------------------
    def test_det_constructor(self) -> None:
        _, det_array = self.construct_detectors()
        np.testing.assert_array_equal(self.dia.reshape(-1, 1), det_array.dia)
        np.testing.assert_array_equal(self.height.reshape(-1, 1), det_array.height)
        np.testing.assert_array_equal(self.wall.reshape(-1, 1), det_array.wall)
        np.testing.assert_array_equal(self.atms.reshape(-1, 1), det_array.atms)
        assert det_array.ndet == 5

    def test_det_constructor_argExpand(self) -> None:
        val = 0.00344
        det_array = IX_det_He3tube(self.dia, self.height, val, self.atms)
        np.testing.assert_array_equal(self.dia.reshape(-1, 1), det_array.dia)
        np.testing.assert_array_equal(self.height.reshape(-1, 1), det_array.height)
        np.testing.assert_array_equal(
            np.full((5, 1), val, dtype=float), det_array.wall
        )
        np.testing.assert_array_equal(self.atms.reshape(-1, 1), det_array.atms)

    def test_det_constructor_tooFewArgs_THROW(self) -> None:
        with pytest.raises(ValueError):
            IX_det_He3tube(self.dia, self.height, self.wall)

    def test_det_constructor_tooManyArgs_THROW(self) -> None:
        with pytest.raises(ValueError):
            IX_det_He3tube(self.dia, self.height, self.wall, self.atms, 0.1)

    def test_det_constructor_mixPosKeywrdArgs(self) -> None:
        _, det_array_ref = self.construct_detectors()
        det_array = IX_det_He3tube(
            self.dia, self.height, "atms", self.atms, "wall", self.wall
        )
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

    # ------------------------------------------------------------------
    # Random number tests
    # ------------------------------------------------------------------
    def test_random_points_noAtten(self) -> None:
        dia0 = 0.01
        height0 = 0.03
        wall0 = 0.0001
        atms0 = 1e-8
        det = IX_det_He3tube(dia0, height0, wall0, atms0)

        nsamples = int(1e5)
        ntol = 4.0
        npath = [1.0, 1.0, 0.0]
        wvec = 1.0
        ok, mess = validate_det_rand(det, nsamples, ntol, npath, wvec)
        assert ok, mess

    def test_random_points_fullAtten(self) -> None:
        dia0 = 0.01
        height0 = 0.03
        wall0 = 0.0001
        atms0 = 1e8
        det = IX_det_He3tube(dia0, height0, wall0, atms0)

        nsamples = int(1e5)
        ntol = 4.0
        npath = [1.0, 1.0, 0.0]
        wvec = 1.0
        ok, mess = validate_det_rand(det, nsamples, ntol, npath, wvec)
        assert ok, mess

    def test_random_points_intermediateAtten(self) -> None:
        dia0 = 0.01
        height0 = 0.03
        wall0 = 0.0001
        atms0 = 10.0
        det = IX_det_He3tube(dia0, height0, wall0, atms0)

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

        det_single = IX_det_He3tube(0.0254, 0.015, 6.35e-4, 10.0)
        rep = det_single.replicate(3)
        assert rep.ndet == 3
        np.testing.assert_allclose(rep.atms, np.full((3, 1), 10.0))

        reord = det_array.reorder([5, 4, 3, 2, 1])
        np.testing.assert_allclose(reord.atms, det_array.atms[::-1])
