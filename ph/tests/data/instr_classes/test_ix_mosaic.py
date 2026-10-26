"""Unit tests for IX_mosaic class and mosaic helpers.

Ported from:
- _test/test_instrument_classes/test_IX_mosaic.m
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.core.hashable import HashableArray, hashable
from pyhorace.core.serializable import Serializable
from pyhorace.coord_transf import bmatrix, ubmatrix
from pyhorace.data.instr_classes.ix_mosaic import (
    IX_mosaic,
    check_mosaic_matrix,
    rand_mosaic_gaussian,
)
from pyhorace.data.instr_classes.ix_sample import IX_sample
from tests.data.instr_classes._helpers import (
    check_matfile_IO,
    hashable_obj_tester,
)


class TestIXMosaic:
    """Test suite for `IX_mosaic` ported from `test_IX_mosaic.m`."""

    def test_mosaic_fwhh_deg(self) -> None:
        mosaic = IX_mosaic(10)
        assert mosaic.parameters == 10.0
        np.testing.assert_array_equal(mosaic.xaxis, [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(mosaic.yaxis, [0.0, 1.0, 0.0])
        assert mosaic.mosaic_pdf_string == "rand_mosaic_gaussian"
        assert mosaic.mosaic_crystal() is True

        S = mosaic.to_struct()
        rec = Serializable.from_struct(S)
        assert mosaic == rec

    def test_mosaic_anisotropic(self) -> None:
        mosaic = IX_mosaic([1, 1, 0], [1, -1, 0], [10, 12, 7])
        np.testing.assert_array_equal(mosaic.xaxis, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(mosaic.yaxis, [1.0, -1.0, 0.0])
        np.testing.assert_array_equal(mosaic.parameters, [10.0, 12.0, 7.0])
        assert mosaic.mosaic_pdf_string == "rand_mosaic_gaussian"

        S = mosaic.to_struct()
        rec = Serializable.from_struct(S)
        assert mosaic == rec

    def test_hashable_prop(self) -> None:
        ap = IX_mosaic([1, 1, 0], [1, -1, 0], [10, 12, 7])
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

    def test_mosaic_matrix(self) -> None:
        xmos = [1, 1, 0]
        ymos = [0, 0, 1]
        eta = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 4.0, 5.0],
                [0.0, 5.0, 9.0],
            ]
        )
        mos = IX_mosaic(xmos, ymos, eta)
        np.testing.assert_array_equal(mos.xaxis, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(mos.yaxis, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(mos.parameters, eta)

        S = mos.to_struct()
        rec = Serializable.from_struct(S)
        assert mos == rec

    def test_mosaic_spread(self) -> None:
        """Test random rotation matrices created by mosaic spread."""
        alatt = [4.0, 4.0, 4.0]
        angdeg = [90.0, 90.0, 90.0]

        bragg1 = np.array([1.0, 1.0, 0.0])
        bragg2 = np.array([0.0, 0.0, 1.0])

        npnt = 500_000

        xgeom = [1, 1, 0]
        ygeom = [0, 0, 1]
        shape = "cuboid"
        pshape = [0.02, 0.03, 0.04]

        xmos = [1, 1, 0]
        ymos = [0, 0, 1]
        eta = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 4.0, 5.0],
                [0.0, 5.0, 9.0],
            ]
        )

        mos = IX_mosaic(xmos, ymos, eta)
        sample = IX_sample(xgeom, ygeom, shape, pshape, mos)

        rng = np.random.default_rng(12345)
        R = sample.rand_mosaic([1, npnt], alatt, angdeg, rng=rng)
        assert R.shape == (3, 3, npnt)

        ub = ubmatrix(bragg1, bragg2, bmatrix(alatt, angdeg, nargout=1), nargout=1)
        hkl = np.einsum("ijk,j->ik", R, bragg1)
        xyz = ub @ hkl

        modQ = float(np.linalg.norm(ub @ bragg1))
        xyz = ((180.0 / np.pi) / modQ) * xyz

        fwhh_cov_rand = np.cov(xyz[1:3, :]) * np.log(256.0)
        fwhh_cov_expect = np.array([[9.0, -5.0], [-5.0, 4.0]])
        np.testing.assert_allclose(fwhh_cov_rand, fwhh_cov_expect, rtol=0.01)

    def test_prev_versions(self) -> None:
        def gauss(*args, **kwargs):
            return rand_mosaic_gaussian(*args, **kwargs)

        mosaic = IX_mosaic([1, 1, 0], [1, -1, 0], gauss, [1, 2, 3])
        assert mosaic.mosaic_pdf_string == "gauss"

        rec = check_matfile_IO(mosaic.class_version(), False, mosaic)
        assert rec == mosaic

        # Verify v1 struct conversion
        v1_struct = {
            "serial_name": "IX_mosaic",
            "version": 1,
            "array_dat": {
                "xaxis_": np.array([1.0, 1.0, 0.0]),
                "yaxis_": np.array([1.0, -1.0, 0.0]),
                "mosaic_pdf_": "@gauss",
                "parameters_": np.array([1.0, 2.0, 3.0]),
                "class_version_": 1,
            },
        }
        rec_v1 = IX_mosaic.from_struct(v1_struct)
        assert rec_v1 == mosaic

    def test_constructors_and_validation(self) -> None:
        mos0 = IX_mosaic()
        assert mos0.parameters == 0.0
        assert mos0.mosaic_crystal() is False

        mos2 = IX_mosaic(rand_mosaic_gaussian, [1, 2, 3])
        np.testing.assert_array_equal(mos2.parameters, [1.0, 2.0, 3.0])

        mat = np.array([[2.0, 1.0, -0.5], [1.0, 4.0, 0.2], [-0.5, 0.2, 6.0]])
        mos3 = IX_mosaic([1, 1, 0], [0, 0, 1], rand_mosaic_gaussian, mat)
        mos3_2 = IX_mosaic([1, 1, 0], [0, 0, 1], mat)
        assert mos3 == mos3_2

        # Collinear xaxis and yaxis must raise
        with pytest.raises(ValueError, match="HERBERT:IX_mosaic:invalid_argument"):
            IX_mosaic([1, 1, 0], [2, 2, 0], 10)

        # Negative mosaic parameter must raise
        with pytest.raises(ValueError, match="HERBERT:IX_mosaic:invalid_argument"):
            IX_mosaic(-1.0)

        # Non-positive-definite 3x3 matrix must raise
        bad_mat = np.array([[1.0, 5.0, 0.0], [5.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        assert check_mosaic_matrix(bad_mat)[0] is None
        with pytest.raises(ValueError, match="HERBERT:IX_mosaic:invalid_argument"):
            IX_mosaic(bad_mat)
