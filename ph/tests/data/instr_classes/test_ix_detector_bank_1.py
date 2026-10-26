"""Unit tests for IX_detector_bank translated from test_IX_detector_bank_1.m."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io as sio

from pyhorace.coord_transf import rotvec_to_rotmat
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_he3tube import IX_det_He3tube
from pyhorace.data.instr_classes.ix_detector_bank import IX_detector_bank

REF_MAT_FILE = (
    Path(__file__).resolve().parents[4]
    / "_test"
    / "test_instrument_classes"
    / "test_IX_detector_bank_1_output.mat"
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
    if block is None and "test_IX_detector_bank_1" in mat_data:
        cls_block = mat_data["test_IX_detector_bank_1"]
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


class TestIXDetectorBank1:
    """Test suite for IX_detector_bank."""

    @pytest.fixture(autouse=True)
    def setup_data(self) -> None:
        dia = np.array([0.0254, 0.0300, 0.0400, 0.0400, 0.0400], dtype=float)
        height = np.array([0.015, 0.025, 0.035, 0.035, 0.035], dtype=float)
        wall = np.array([6.35e-4, 10.0e-4, 15.0e-4, 15.0e-4, 15.0e-4], dtype=float)
        atms = np.array([10.0, 6.0, 4.0, 7.0, 4.0], dtype=float)
        self.dets = IX_det_He3tube(dia, height, wall, atms)

        self.id = np.array([10, 20, 30, 40, 50], dtype=int)
        self.x2 = np.array([2.1, 2.2, 2.3, 2.4, 2.5], dtype=float)
        self.phi = np.array([10.0, 30.0, 50.0, 70.0, 90.0], dtype=float)
        self.azim = np.array([0.0, 11.0, 22.0, 33.0, 44.0], dtype=float)
        self.rotvec = np.zeros((3, 5), dtype=float)
        self.rotvec[:, 4] = np.array([20.0, 45.0, 15.0], dtype=float)
        self.mat_data = _load_ref_mat()

    def test_hashable_prop_det_bank(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        assert not bank.hash_defined
        bank, h1, is_calc = bank.build_hash()
        assert is_calc
        assert isinstance(h1, str) and len(h1) == 32
        assert bank.hash_defined
        assert bank.hash_value == h1
        # Modify property -> hash should change
        bank2 = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        bank2, h2, _ = bank2.build_hash()
        assert h2 == h1
        bank2.x2 = self.x2 + 0.1
        assert not bank2.hash_defined
        bank2, h3, _ = bank2.build_hash()
        assert h3 != h1

    # ------------------------------------------------------------------
    # Constructor tests
    # ------------------------------------------------------------------
    def test_constructor_1(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets
        )
        np.testing.assert_array_equal(bank.id, self.id.reshape(-1, 1))
        np.testing.assert_array_equal(bank.x2, self.x2.reshape(-1, 1))
        np.testing.assert_array_equal(bank.phi, self.phi.reshape(-1, 1))
        np.testing.assert_array_equal(bank.azim, self.azim.reshape(-1, 1))
        assert bank.det == self.dets

    def test_constructor_2(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        np.testing.assert_array_equal(bank.id, self.id.reshape(-1, 1))
        np.testing.assert_array_equal(bank.x2, self.x2.reshape(-1, 1))
        np.testing.assert_array_equal(bank.phi, self.phi.reshape(-1, 1))
        np.testing.assert_array_equal(bank.azim, self.azim.reshape(-1, 1))
        dmat = np.swapaxes(rotvec_to_rotmat(self.rotvec), 0, 1)
        np.testing.assert_allclose(bank.dmat, dmat, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(bank.rotvec, self.rotvec, atol=1e-13, rtol=0.0)
        assert bank.det == self.dets

    def test_constructor_3(self) -> None:
        dmat = np.swapaxes(rotvec_to_rotmat(self.rotvec), 0, 1)
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "dmat", dmat
        )
        np.testing.assert_array_equal(bank.id, self.id.reshape(-1, 1))
        np.testing.assert_array_equal(bank.x2, self.x2.reshape(-1, 1))
        np.testing.assert_array_equal(bank.phi, self.phi.reshape(-1, 1))
        np.testing.assert_array_equal(bank.azim, self.azim.reshape(-1, 1))
        np.testing.assert_allclose(bank.dmat, dmat, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(bank.rotvec, self.rotvec, atol=1e-13, rtol=0.0)
        assert bank.det == self.dets

    # ------------------------------------------------------------------
    # Save and load / serialization
    # ------------------------------------------------------------------
    def test_save_load_1(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        s = bank.to_struct()
        recovered = Serializable.from_struct(s)
        assert bank == recovered

        raw = bank.serialize()
        recovered2, _ = Serializable.deserialize(raw)
        assert bank == recovered2

    # ------------------------------------------------------------------
    # Method tests
    # ------------------------------------------------------------------
    def test_effic_1(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        wvec = 10.0

        ind = np.array([1, 2, 3, 4, 5], dtype=int)
        paths = bank.dmat[0][:, ind - 1]
        effs = self.dets.effic(ind, paths, wvec)

        eff_array = bank.effic(wvec)

        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_effic_1", "eff_array", eff_array)

    def test_effic_2(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        wvec = 10.0

        ind = np.array([1, 2, 3, 4, 5], dtype=int)
        paths = bank.dmat[0][:, ind - 1]
        effs = self.dets.effic(ind, paths, wvec)

        eff_array = bank.effic(ind, wvec)

        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_effic_2", "eff_array", eff_array)

    def test_effic_3(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )
        wvec = 10.0

        ind = np.array([5, 3], dtype=int)
        paths = bank.dmat[0][:, ind - 1]
        effs = self.dets.effic(ind, paths, wvec)

        eff_array = bank.effic(ind, wvec)

        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_effic_3", "eff_array", eff_array)

    def test_effic_4(self) -> None:
        bank = IX_detector_bank(
            self.id, self.x2, self.phi, self.azim, self.dets, "rotvec", self.rotvec
        )

        ind = np.array([[3, 1, 4, 1], [3, 1, 1, 3]], dtype=int).T
        wvec = np.array([[10.0, 9.0, 8.0, 7.0], [10.0, 9.0, 9.0, 5.0]], dtype=float).T
        paths = bank.dmat[0][:, ind.ravel(order="F") - 1]
        effs = self.dets.effic(ind, paths, wvec)

        eff_array = bank.effic(ind, wvec)

        np.testing.assert_allclose(effs, eff_array, atol=1e-13, rtol=1e-13)
        _check_ref(self.mat_data, "test_effic_4", "eff_array", eff_array)
