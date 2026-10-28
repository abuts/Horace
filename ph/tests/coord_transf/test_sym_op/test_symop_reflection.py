"""Pytest suite for SymopReflection and SymopSetPlaneInterface.

Translated from MATLAB ``_test/test_sym_op/test_symop_reflection.m``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    Symop,
    SymopReflection,
    bmatrix,
)
from pyhorace.core.serializable import Serializable


class TestSymopReflection:
    """Port of MATLAB ``test_symop_reflection`` test cases."""

    points2transform = np.column_stack(
        [np.eye(3), [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    )

    def test_symop_reflection_with_genBM_and_normal111_inRlu(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [1, 1, 1], "b_matrix", bm, "rlu")
        expected_nrm = (bm @ np.array([1.0, 1.0, 1.0])) / np.linalg.norm(
            bm @ np.array([1.0, 1.0, 1.0])
        )
        np.testing.assert_allclose(out.normvec, expected_nrm, atol=1e-14)
        np.testing.assert_allclose(out.u, [-0.3831, 0.8722, -0.0811], atol=1e-4)
        np.testing.assert_allclose(out.v, [0.0004, 0.3275, 1.4364], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_and_normal111(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [1, 1, 1], "b_matrix", bm, "cc")
        np.testing.assert_allclose(out.normvec, np.ones(3) / np.sqrt(3), atol=1e-14)
        np.testing.assert_allclose(out.u, [-0.3633, 0.5199, -0.5073], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.1789, 0.2003, 0.8786], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_and_normal010(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [0, 1, 0], "b_matrix", bm, "cc")
        np.testing.assert_array_equal(out.normvec, [0, 1, 0])
        np.testing.assert_allclose(out.u, [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(out.v, [-0.2213, -0.8719, -3.8240], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_and_normal100(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [1, 0, 0], "b_matrix", bm, "cc")
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [0.0563, 0.2220, 0.9734], atol=1e-4)
        np.testing.assert_allclose(out.v, [0.1932, -0.6098, 0.0], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_and_normal001rlu(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [0, 0, 1], "b_matrix", bm, "rlu")
        expected_nrm = (bm @ np.array([0.0, 0.0, 1.0])) / np.linalg.norm(
            bm @ np.array([0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(out.normvec, expected_nrm, atol=1e-14)
        np.testing.assert_allclose(out.u, [-0.3020, 0.9533, 0.4014], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.3761, -0.1470, -0.6445], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_and_normal001(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopReflection("normvec", [0, 0, 1], "b_matrix", bm, "cc")
        np.testing.assert_array_equal(out.normvec, [0, 0, 1])
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_allclose(out.v, [-0.7588, 2.3956, 0.0], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_genBM_fails_without_descr(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneInterface:invalid_argument"
        ):
            SymopReflection("normvec", [0, 0, 1], "b_matrix", bm)

    def test_symop_reflection_with_orthoBM_and_normal111_inRlu(self):
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out = SymopReflection("normvec", [1, 1, 1], "b_matrix", bm, "rlu")
        expected_nrm = (bm @ np.array([1.0, 1.0, 1.0])) / np.linalg.norm(
            bm @ np.array([1.0, 1.0, 1.0])
        )
        np.testing.assert_allclose(out.normvec, expected_nrm, atol=1e-14)
        np.testing.assert_allclose(out.u, [-0.0816, -0.0816, 0.9184], atol=1e-4)
        np.testing.assert_allclose(out.v, [0.1429, -0.5714, 0.0], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_orthoBM_and_normal111(self):
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out = SymopReflection("normvec", [1, 1, 1], "b_matrix", bm)
        np.testing.assert_allclose(out.normvec, np.ones(3) / np.sqrt(3), atol=1e-14)
        np.testing.assert_allclose(out.u, [0.6667, -0.6667, -1.000], atol=1e-4)
        np.testing.assert_allclose(out.v, [0.0, 1.1547, -1.7321], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_orthoBM_and_normal010(self):
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out = SymopReflection("normvec", [0, 1, 0], "b_matrix", bm)
        np.testing.assert_array_equal(out.normvec, [0, 1, 0])
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_allclose(out.v, [0, 0, -3], atol=1e-4)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_orthoBM_and_normal100(self):
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out = SymopReflection("normvec", [1, 0, 0], "b_matrix", bm)
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_array_equal(out.u, [0, 1, 0])
        np.testing.assert_allclose(out.v, [0, 0, 1.5], atol=1e-14)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_orthoBM_and_normal001(self):
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out = SymopReflection("normvec", [0, 0, 1], "b_matrix", bm)
        np.testing.assert_array_equal(out.normvec, [0, 0, 1])
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_allclose(out.v, [0, 2, 0], atol=1e-14)
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert not out.input_nrmv_in_rlu

        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.u, out.normvec), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.dot(bm @ out.v, out.normvec), 0.0, atol=1e-14)

    def test_symop_reflection_with_normal001(self):
        out = SymopReflection("normvec", [0, 0, 1])
        np.testing.assert_array_equal(out.normvec, [0, 0, 1])
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_array_equal(out.v, [0, 1, 0])
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert out.input_nrmv_in_rlu

    def test_symop_reflection_with_normal010(self):
        out = SymopReflection("normvec", [0, 1, 0])
        np.testing.assert_array_equal(out.normvec, [0, 1, 0])
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_array_equal(out.v, [0, 0, -1])
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert out.input_nrmv_in_rlu

    def test_symop_reflection_with_normal100(self):
        out = SymopReflection("normvec", [1, 0, 0])
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_array_equal(out.u, [0, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 0, 1])
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert out.input_nrmv_in_rlu

    def test_symop_create_reflection(self):
        out = Symop.create("Refl", [1, 0, 0], [0, 1, 0])
        assert isinstance(out, SymopReflection)
        np.testing.assert_array_equal(out.u, [1, 0, 0])
        np.testing.assert_array_equal(out.v, [0, 1, 0])
        np.testing.assert_array_equal(out.offset, [0, 0, 0])
        assert out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [0, 0, 1])

    def test_reflection_constructor_with_bmat(self):
        bm = bmatrix([1, 2, 3], [70, 80, 110])
        out = SymopReflection([1, 1, 0], [0, 1, 1], [3, 3, 3], bm)
        np.testing.assert_array_equal(out.u, [1, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 1, 1])
        np.testing.assert_array_equal(out.offset, [3, 3, 3])
        assert not out.input_nrmv_in_rlu
        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)

    def test_reflection_constructor_bm_later(self):
        out = SymopReflection([1, 1, 0], [0, 1, 1], [3, 3, 3])
        np.testing.assert_array_equal(out.u, [1, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 1, 1])
        np.testing.assert_array_equal(out.offset, [3, 3, 3])
        assert out.input_nrmv_in_rlu
        old_normvec = out.normvec.copy()
        bm = bmatrix([1, 2, 3], [70, 80, 110])
        out.b_matrix = bm
        assert not out.input_nrmv_in_rlu
        c1 = np.cross(bm @ out.u, bm @ out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)
        assert not np.all(old_normvec == out.normvec)

    def test_reflection_constructor_no_bm(self):
        out = SymopReflection([1, 1, 0], [0, 1, 1], [3, 3, 3])
        np.testing.assert_array_equal(out.u, [1, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 1, 1])
        np.testing.assert_array_equal(out.offset, [3, 3, 3])
        assert out.input_nrmv_in_rlu
        c1 = np.cross(out.u, out.v)
        c1 = c1 / np.linalg.norm(c1)
        np.testing.assert_allclose(c1, out.normvec, atol=1e-14)

    def test_set_coord_in_orth_using_normvec_rlu(self):
        out = SymopReflection("norm", [1, 0, 0], "offset", [3, 3, 3], "rlu")
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.u, [0, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 0, 1])
        bm = bmatrix([1, 2, 3], [90, 90, 90])
        out.b_matrix = bm
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(out.v, [0, 0, 1.5], atol=1e-14)

        out.input_nrmv_in_rlu = True  # ignored once b_matrix is set
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(out.v, [0, 0, 1.5], atol=1e-14)

    def test_set_coord_in_nonorth_using_normvec_cc(self):
        out = SymopReflection("norm", [1, 0, 0], "offset", [3, 3, 3], "cc")
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.u, [0, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 0, 1])
        bm = bmatrix([1, 2, 3], [70, 120, 80])
        out.b_matrix = bm
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [-0.1604, 0.2194, 0.96240], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.1177, -0.6029, 0.0], atol=1e-4)

        out.input_nrmv_in_rlu = True
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [-0.1604, 0.2194, 0.96240], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.1177, -0.6029, 0.0], atol=1e-4)

    def test_set_coord_in_nonorth_using_normvec_rlu(self):
        out = SymopReflection("norm", [1, 0, 0], "offset", [3, 3, 3], "rlu")
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.u, [0, 1, 0])
        np.testing.assert_array_equal(out.v, [0, 0, 1])
        bm = bmatrix([1, 2, 3], [70, 120, 80])
        out.b_matrix = bm
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [-0.1604, 0.2194, 0.96240], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.1177, -0.6029, 0.0], atol=1e-4)

        out.input_nrmv_in_rlu = True
        assert not out.input_nrmv_in_rlu
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        np.testing.assert_allclose(out.u, [-0.1604, 0.2194, 0.96240], atol=1e-4)
        np.testing.assert_allclose(out.v, [-0.1177, -0.6029, 0.0], atol=1e-4)

    def test_set_coord_in_nonorth_using_uv(self):
        out = SymopReflection([0, 1, 0], [0, 0, 1], [3, 3, 3])
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert out.input_nrmv_in_rlu
        bm = bmatrix([1, 2, 3], [70, 120, 80])
        out.b_matrix = bm
        assert not out.input_nrmv_in_rlu
        np.testing.assert_allclose(out.normvec, [0.7845, 0.3668, -0.5000], atol=1e-4)

    def test_reflection_construct_normal_fails_on_nonort_without_coord(self):
        out = SymopReflection("norm", [1, 1, 0], "offset", [3, 3, 3])
        bm = bmatrix([1, 2, 3], [70, 90, 90])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneInterface:invalid_argument"
        ):
            out.b_matrix = bm
        out.input_nrmv_in_rlu = False
        out.b_matrix = bm
        np.testing.assert_allclose(out.b_matrix, bm)

    def test_reflection_constructor_fail(self):
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopReflection(1)
        with pytest.raises(HoraceError, match="HORACE:SymopReflection:invalid_argument"):
            SymopReflection([0, 1, 0])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopReflection(1, 90)
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopReflection([1, 0, 0], 90)
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopReflection(np.eye(3))
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopReflection([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], 90)
        with pytest.raises(HoraceError, match="HORACE:SymopReflection:invalid_argument"):
            SymopReflection([1, 0, 0], [1, 0, 0])

    def test_serializable_and_in_irreducible(self):
        bm = bmatrix([2, 3, 4], [80, 100, 110])
        sr = SymopReflection([1, 0, 0], [0, 1, 0], [1, 1, 1], bm)
        u_off = bm @ np.array([1.0, 1.0, 1.0])
        nrm = sr.normvec
        pts = np.column_stack(
            [u_off + 0.5 * nrm, u_off - 0.5 * nrm, u_off - 1e-4 * nrm]
        )
        in_irr = sr.in_irreducible(pts)
        np.testing.assert_array_equal(in_irr, [True, False, False])

        in_irr_tol = sr.in_irreducible(pts, 1e-3)
        np.testing.assert_array_equal(in_irr_tol, [True, False, True])

        s_dict = sr.to_struct()
        sr_rec = Serializable.from_struct(s_dict)
        assert sr_rec.equal_to_tol(sr)[0]
