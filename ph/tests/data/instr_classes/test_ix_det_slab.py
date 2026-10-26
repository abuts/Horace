"""Unit tests for IX_det_slab translated from test_IX_det_slab.m."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_slab import IX_det_slab
from tests.data.instr_classes.test_ix_det_tobyfit_classic import validate_det_rand

REF_MAT_FILE = (
    Path(__file__).resolve().parents[4]
    / "_test"
    / "test_instrument_classes"
    / "test_IX_det_slab_output.mat"
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
    if block is None and "test_IX_det_slab" in mat_data:
        cls_block = mat_data["test_IX_det_slab"]
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


class TestIXDetSlab:
    """Test suite for IX_det_slab."""

    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
        self.depth = np.array([0.1000, 0.1200, 0.1400, 0.1600, 0.1800], dtype=float)
        self.width = np.array([0.2000, 0.2200, 0.2400, 0.2600, 0.2800], dtype=float)
        self.height = np.array([0.3000, 0.3200, 0.3400, 0.3600, 0.3800], dtype=float)
        self.atten = np.array([0.1000, 0.0800, 0.0600, 0.0400, 0.0200], dtype=float)
        th = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=float)
        self.path = np.vstack([np.cos(th), np.zeros_like(th), np.sin(th)])
        self.seed = 0
        np.random.seed(self.seed)
        self.mat_data = _load_ref_mat()

    def construct_detectors(self) -> tuple[list[IX_det_slab], IX_det_slab]:
        dets = [
            IX_det_slab(
                self.depth[i], self.width[i], self.height[i], self.atten[i]
            )
            for i in range(5)
        ]
        det_array = IX_det_slab(self.depth, self.width, self.height, self.atten)
        return dets, det_array

    # ------------------------------------------------------------------
    # Constructor tests
    # ------------------------------------------------------------------
    def test_det_constructor(self) -> None:
        _, det_array = self.construct_detectors()
        np.testing.assert_array_equal(self.depth.reshape(-1, 1), det_array.depth)
        np.testing.assert_array_equal(self.width.reshape(-1, 1), det_array.width)
        np.testing.assert_array_equal(self.height.reshape(-1, 1), det_array.height)
        np.testing.assert_array_equal(self.atten.reshape(-1, 1), det_array.atten)
        assert det_array.ndet == 5

    def test_det_constructor_argExpand(self) -> None:
        val = 0.00344
        det_array = IX_det_slab(self.depth, self.width, val, self.atten)
        np.testing.assert_array_equal(self.depth.reshape(-1, 1), det_array.depth)
        np.testing.assert_array_equal(self.width.reshape(-1, 1), det_array.width)
        np.testing.assert_array_equal(
            np.full((5, 1), val, dtype=float), det_array.height
        )
        np.testing.assert_array_equal(self.atten.reshape(-1, 1), det_array.atten)

    def test_det_constructor_tooFewArgs_THROW(self) -> None:
        with pytest.raises(ValueError):
            IX_det_slab(self.depth, self.width, self.height)

    def test_det_constructor_tooManyArgs_THROW(self) -> None:
        with pytest.raises(ValueError):
            IX_det_slab(self.depth, self.width, self.height, self.atten, 0.1)

    def test_det_constructor_mixPosKeywrdArgs(self) -> None:
        _, det_array_ref = self.construct_detectors()
        det_array = IX_det_slab(
            self.depth, self.width, "atten", self.atten, "height", self.height
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
        depth0 = 0.1
        width0 = 0.03
        height0 = 0.5
        atten0 = 1e-8
        det = IX_det_slab(depth0, width0, height0, atten0)

        nsamples = int(1e5)
        ntol = 4.0
        npath = [1.0, 1.0, 0.0]
        wvec = 1.0
        ok, mess = validate_det_rand(det, nsamples, ntol, npath, wvec)
        assert ok, mess

    def test_random_points_fullAtten(self) -> None:
        depth0 = 0.1
        width0 = 0.03
        height0 = 0.5
        atten0 = 1e8
        det = IX_det_slab(depth0, width0, height0, atten0)

        nsamples = int(1e5)
        ntol = 4.0
        npath = [1.0, 1.0, 0.0]
        wvec = 1.0
        ok, mess = validate_det_rand(det, nsamples, ntol, npath, wvec)
        assert ok, mess

    def test_random_points_intermediateAtten(self) -> None:
        depth0 = 0.1
        width0 = 0.03
        height0 = 0.5
        atten0 = 0.05
        det = IX_det_slab(depth0, width0, height0, atten0)

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

        det_single = IX_det_slab(0.1, 0.2, 0.3, 0.05)
        rep = det_single.replicate(3)
        assert rep.ndet == 3
        np.testing.assert_allclose(rep.depth, np.full((3, 1), 0.1))

        reord = det_array.reorder([5, 4, 3, 2, 1])
        np.testing.assert_allclose(reord.depth, det_array.depth[::-1])
