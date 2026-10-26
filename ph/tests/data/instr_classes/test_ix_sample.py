"""Unit tests for IX_sample, IX_samp, and IX_null_sample classes.

Ported from:
- _test/test_instrument_classes/test_IX_sample.m
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.core.hashable import HashableArray, hashable
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_mosaic import IX_mosaic
from pyhorace.data.instr_classes.ix_null_sample import IX_null_sample
from pyhorace.data.instr_classes.ix_samp import IX_samp
from pyhorace.data.instr_classes.ix_sample import IX_sample
from tests.data.instr_classes._helpers import (
    MATLAB_TEST_DIR,
    check_matfile_IO,
    hashable_obj_tester,
    load_mat_reference,
)


class TestIXSample:
    """Test suite for `IX_sample`, `IX_samp`, and `IX_null_sample` ported from `test_IX_sample.m`."""

    @pytest.fixture
    def sample_fixtures(self):
        sam1 = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4])
        sam2 = IX_sample([0, 1, 0], [0, 0, 1], "cuboid", [12, 13, 34])
        sam3 = IX_sample([1, 1, 0], [0, 0, 1], "cuboid", [22, 23, 24])

        s1 = [sam1, sam1, sam2, sam2, sam2]
        s2 = [sam3, sam1, sam2, sam3, sam1]
        s3 = [sam2, sam3, sam1, sam2, sam3]
        return {
            "sam1": sam1,
            "sam2": sam2,
            "sam3": sam3,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }

    def test_to_from_struct(self) -> None:
        sample = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        sample.alatt = [1, 2, 3]
        sample.angdeg = [91, 89, 91]
        s_dict = sample.to_struct()

        samp_rec = Serializable.from_struct(s_dict)
        assert sample == samp_rec

    def test_hashable_prop(self) -> None:
        ap = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "eta", 4134)
        hashable_obj_tester(ap)

        ap, _, _ = ap.build_hash()
        S = ap.to_struct()
        rec_obj = hashable.from_struct(S)
        assert ap.hash_defined
        assert ap == rec_obj

        obj_arr = HashableArray([ap, ap.__copy__()])
        S_arr = obj_arr.to_struct()
        rec_arr = hashable.from_struct(S_arr)
        assert all(rec_arr.hash_defined)
        assert obj_arr.eq(rec_arr)

    def test_IX_sample_constructor_error_if_required_args_missing(self) -> None:
        with pytest.raises(ValueError, match="HERBERT:IX_sample:invalid_argument"):
            IX_sample([1, 0, 0], [0, 1, 0], "cuboid")

    def test_IX_sample_constructor_error_if_invalid_shape(self) -> None:
        with pytest.raises(ValueError, match="HERBERT:IX_sample:invalid_argument"):
            IX_sample([1, 0, 0], [0, 1, 0], "banana", [2, 3, 4])

    def test_IX_sample_constructor_accepts_and_sets_hall_symbol(self) -> None:
        sample = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        assert sample.hall_symbol == "hsymbol"

    def test_IX_sample_constructor_accepts_and_sets_temperature(self) -> None:
        sample = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "temperature", 1234.5
        )
        assert sample.temperature == 1234.5

    def test_IX_sample_constructor_errors_for_non_numeric_temperature(self) -> None:
        with pytest.raises(ValueError, match="HERBERT:IX_sample:invalid_argument"):
            IX_sample(
                [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "temperature", "string"
            )

    def test_IX_sample_constructor_accepts_and_sets_name(self) -> None:
        sample = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "name", "test name"
        )
        assert sample.name == "test name"

    def test_IX_sample_constructor_accepts_and_sets_mosaic_eta(self) -> None:
        eta = IX_mosaic(1234)
        sample = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "eta", eta)
        assert sample.eta == eta

    def test_IX_sample_constructorsets_sets_numeric_eta_as_mosaic(self) -> None:
        sample = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "eta", 4134)
        assert sample.eta == IX_mosaic(4134)

    def test_covariance(self, sample_fixtures) -> None:
        s2 = sample_fixtures["s2"]
        # 1-based indices [2, 2, 1, 4, 3] in s2 -> 0-based [1, 1, 0, 3, 2]
        indices = [1, 1, 0, 3, 2]
        cov = np.stack([s2[i].covariance() for i in indices], axis=2)

        expected = np.stack(
            [
                np.diag(np.array([2.0, 3.0, 4.0]) ** 2 / 12.0),
                np.diag(np.array([2.0, 3.0, 4.0]) ** 2 / 12.0),
                np.diag(np.array([22.0, 23.0, 24.0]) ** 2 / 12.0),
                np.diag(np.array([22.0, 23.0, 24.0]) ** 2 / 12.0),
                np.diag(np.array([12.0, 13.0, 34.0]) ** 2 / 12.0),
            ],
            axis=2,
        )
        np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-12)

        if (MATLAB_TEST_DIR / "test_IX_sample_output.mat").exists():
            ref = load_mat_reference("test_IX_sample_output.mat")
            if "test_covariance" in ref and isinstance(ref["test_covariance"], dict):
                ref_cov = ref["test_covariance"].get("cov")
                if isinstance(ref_cov, np.ndarray):
                    np.testing.assert_allclose(cov, ref_cov, rtol=1e-12, atol=1e-12)

    def test_identical_samples_are_equal(self) -> None:
        samp1 = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4])
        samp2 = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4])
        assert samp1 == samp2
        assert not (samp1 != samp2)

    def test_different_samples_are_not_equal(self) -> None:
        samp1 = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4])
        samp2 = IX_sample([1, 1, 0], [0, 0, 1], "cuboid", [22, 23, 24])
        assert not (samp1 == samp2)
        assert samp1 != samp2

    def test_identical_samples_with_matching_hall_symbol_are_equal(self) -> None:
        samp1 = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        samp2 = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        assert samp1 == samp2
        assert not (samp1 != samp2)

    def test_matching_samples_with_missing_hall_symbol_are_not_equal(self) -> None:
        samp1 = IX_sample([1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4])
        samp2 = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        assert not (samp1 == samp2)
        assert samp1 != samp2

    def test_matching_samples_with_different_hall_symbols_are_not_equal(self) -> None:
        samp1 = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "other"
        )
        samp2 = IX_sample(
            [1, 0, 0], [0, 1, 0], "cuboid", [2, 3, 4], "hall_symbol", "hsymbol"
        )
        assert not (samp1 == samp2)
        assert samp1 != samp2

    def test_pdf(self, sample_fixtures) -> None:
        sam1 = sample_fixtures["sam1"]
        sam2 = sample_fixtures["sam2"]

        rng = np.random.default_rng(42)
        nsamp = 500_000
        samp2 = sam1.rand(1, nsamp, rng=rng)
        samp3 = sam2.rand(1, nsamp, rng=rng)

        mean2 = np.mean(samp2, axis=1)
        mean3 = np.mean(samp3, axis=1)
        std2 = np.std(samp2, axis=1, ddof=0)
        std3 = np.std(samp3, axis=1, ddof=0)

        np.testing.assert_allclose(mean2, [0.0, 0.0, 0.0], atol=0.005)
        np.testing.assert_allclose(mean3, [0.0, 0.0, 0.0], atol=0.03)

        np.testing.assert_allclose(std2, sam1.ps / np.sqrt(12.0), rtol=0.005)
        np.testing.assert_allclose(std3, sam2.ps / np.sqrt(12.0), atol=0.02)

    def test_prev_versions_array(self) -> None:
        sample_arr = [
            IX_sample(
                "", False, [1, 1, 1], [0, 1, 1], "cuboid", [0.005, 0.005, 0.0005]
            ),
            IX_sample(
                "FeSi",
                True,
                [1, 1, 0],
                [0, 1, 3],
                "cuboid",
                [0.020, 0.024, 0.028],
                0.5,
                120,
            ),
        ]

        rec_arr = check_matfile_IO(
            sample_arr[0].class_version(), False, sample_arr
        )
        assert len(rec_arr) == 2
        assert rec_arr[0] == sample_arr[0]
        assert rec_arr[1] == sample_arr[1]

        # Test old v1 struct array conversion
        v1_arr_struct = {
            "serial_name": "IX_sample",
            "version": 1,
            "array_dat": [
                {
                    "name_": "",
                    "single_crystal_": False,
                    "xgeom_": np.array([1.0, 1.0, 1.0]),
                    "ygeom_": np.array([0.0, 1.0, 1.0]),
                    "shape_": "cuboid",
                    "ps_": np.array([0.005, 0.005, 0.0005]),
                    "eta_": IX_mosaic().to_struct(),
                    "temperature_": 0.0,
                    "hall_symbol_": "",
                    "alatt_": np.array([], dtype=float),
                    "angdeg_": np.array([], dtype=float),
                    "class_version_": 1,
                },
                {
                    "name_": "FeSi",
                    "single_crystal_": True,
                    "xgeom_": np.array([1.0, 1.0, 0.0]),
                    "ygeom_": np.array([0.0, 1.0, 3.0]),
                    "shape_": "cuboid",
                    "ps_": np.array([0.020, 0.024, 0.028]),
                    "eta_": IX_mosaic(0.5).to_struct(),
                    "temperature_": 120.0,
                    "hall_symbol_": "",
                    "alatt_": np.array([], dtype=float),
                    "angdeg_": np.array([], dtype=float),
                    "class_version_": 1,
                },
            ],
        }
        rec_v1 = IX_sample.from_struct(v1_arr_struct)
        assert len(rec_v1) == 2
        assert rec_v1[0] == sample_arr[0]
        assert rec_v1[1] == sample_arr[1]

    def test_prev_versions(self) -> None:
        sample = IX_sample(
            "Fe", True, [1, 1, 0], [0, 1, 3], "cuboid", [0.020, 0.024, 0.028]
        )

        rec = check_matfile_IO(sample.class_version(), False, sample)
        assert rec == sample

        v1_struct = {
            "serial_name": "IX_sample",
            "version": 1,
            "array_dat": {
                "name_": "Fe",
                "single_crystal_": True,
                "xgeom_": np.array([1.0, 1.0, 0.0]),
                "ygeom_": np.array([0.0, 1.0, 3.0]),
                "shape_": "cuboid",
                "ps_": np.array([0.020, 0.024, 0.028]),
                "eta_": IX_mosaic().to_struct(),
                "temperature_": 0.0,
                "hall_symbol_": "",
                "alatt_": np.array([], dtype=float),
                "angdeg_": np.array([], dtype=float),
                "class_version_": 1,
            },
        }
        rec_v1 = IX_sample.from_struct(v1_struct)
        assert rec_v1 == sample

    def test_additional_constructors_and_base_classes(self) -> None:
        # 2-arg lattice constructor on IX_sample
        sam_lat = IX_sample([1, 2, 3], [90, 89, 91])
        np.testing.assert_array_equal(sam_lat.alatt, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sam_lat.angdeg, [90.0, 89.0, 91.0])
        assert sam_lat.lattice_defined() is True

        # Legacy constructor with 'name' as first arg followed by bool single_crystal
        sam4 = IX_sample(
            "name",
            False,
            [1, 1, 0],
            [0, 0, 1],
            "cuboid",
            [10, 11, 12],
            [0.5, 0.6, 0.4],
            300,
        )
        assert sam4.name == "name"
        assert sam4.single_crystal is False
        assert sam4.temperature == 300.0
        np.testing.assert_array_equal(sam4.eta.parameters, [0.5, 0.6, 0.4])

        # IX_samp tests
        base = IX_samp([2.5, 2.5, 2.5], [90, 90, 90], "base_samp")
        assert base.name == "base_samp"
        assert base.lattice_defined() is True
        hashable_obj_tester(base)

        # IX_null_sample tests
        null_samp = IX_null_sample()
        assert null_samp.name == ""
        assert null_samp.alatt.size == 0
        assert null_samp.angdeg.size == 0
        assert null_samp.lattice_defined() is False
        assert null_samp.null_struct() == {}
        rec_null = Serializable.from_struct(null_samp.to_struct())
        assert rec_null == null_samp
        with pytest.raises(ValueError, match="HORACE:IX_null_sample:invalid_argument"):
            IX_null_sample("only_one_arg")
