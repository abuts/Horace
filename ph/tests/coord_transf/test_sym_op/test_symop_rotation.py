"""Pytest suite for SymopRotation.

Translated from MATLAB ``_test/test_sym_op/test_symop_rotation.m``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    Symop,
    SymopArray,
    SymopIdentity,
    SymopRotation,
    bmatrix,
)
from pyhorace.core.serializable import Serializable


class TestSymopRotation:
    """Port of MATLAB ``test_symop_rotation`` test cases."""

    def test_symop_create_rotation(self):
        out = Symop.create("Rot", [1, 0, 0], 120)
        assert isinstance(out, SymopRotation)
        np.testing.assert_array_equal(out.n, [1, 0, 0])
        assert out.theta_deg == 120
        np.testing.assert_array_equal(out.offset, [0, 0, 0])

    def test_rot_with_genBM_and_keyvalue_param(self):
        bm = bmatrix([1, 2, 3], [70, 80, 120])
        out = SymopRotation(
            "normvec",
            [0, 0, 1],
            "theta_deg",
            90,
            "b_matrix",
            bm,
            "input_nrmv_in_rlu",
            "cc",
        )
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

    def test_rotation_constructor_with_uv(self):
        out = SymopRotation([1, 0, 0], [0, 1, 0], 120, [3, 3, 3])
        np.testing.assert_array_equal(out.normvec, [0, 0, 1])
        assert out.theta_deg == 120
        np.testing.assert_array_equal(out.offset, [3, 3, 3])

    def test_rotation_constructor_with_bm(self):
        out = SymopRotation([1, 0, 0], 120, [3, 3, 3], np.pi * np.eye(3))
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert out.theta_deg == 120
        np.testing.assert_array_equal(out.offset, [3, 3, 3])

    def test_rotation_constructor(self):
        out = SymopRotation([1, 0, 0], 120, [3, 3, 3])
        np.testing.assert_array_equal(out.normvec, [1, 0, 0])
        assert out.theta_deg == 120
        np.testing.assert_array_equal(out.offset, [3, 3, 3])

    def test_rotation_constructor_fail(self):
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopRotation(1)
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation([1, 0, 0])
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation([1, 0, 0], [0, 1, 0])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopRotation(np.eye(3), [1, 0, 0])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopRotation([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [1, 1, 0])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneIntrerface:invalid_argument"
        ):
            SymopRotation([[0, 1, 0], [-1, 0, 0]])

        bm = bmatrix([1, 2, 3], [70, 80, 90])
        with pytest.raises(
            HoraceError, match="HORACE:SymopSetPlaneInterface:invalid_argument"
        ):
            SymopRotation([0, 1, 0], 90, [-1, 0, 0], bm)

    def test_rot_fold_constructor(self):
        sym = SymopRotation.fold(4, [0, 0, 1])
        assert isinstance(sym, SymopArray)
        assert len(sym) == 4
        assert isinstance(sym[0], SymopIdentity)
        for i in range(1, 4):
            assert isinstance(sym[i], SymopRotation)
            assert sym[i].theta_deg == 90.0 * i
            np.testing.assert_allclose(sym[i].n, [0, 0, 1])
            assert not sym[i].input_nrmv_in_rlu

        s_dict = sym.to_struct()
        sym_rec = Serializable.from_struct(s_dict)
        assert sym.equal_to_tol(sym_rec)[0]

        sym_uv = SymopRotation.fold(3, np.array([[1, 0, 0], [0, 1, 0]]), [1, 1, 0])
        assert len(sym_uv) == 3
        assert sym_uv[1].theta_deg == 120.0
        np.testing.assert_allclose(sym_uv[1].offset, [1, 1, 0])

        sym_rlu = SymopRotation.fold(2, [0, 0, 1], "rlu", [0, 0, 1])
        assert len(sym_rlu) == 2
        assert sym_rlu[1].input_nrmv_in_rlu
        np.testing.assert_allclose(sym_rlu[1].offset, [0, 0, 1])

        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2.5, [0, 0, 1])
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2, [0, 0])
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2, [0, 0, 1], "rlu", "cc")
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2, [0, 0, 1], "bad")
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2, [0, 0, 1], [1, 2])
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopRotation.fold(2, [0, 0, 1], "rlu", [0, 0, 0], "extra")

    def test_in_irreducible_and_serializable(self):
        sr = SymopRotation([0, 0, 1], 90)
        pts = np.array(
            [
                [1.0, 1.0, 0.0],    # 1st quadrant -> True
                [-1.0, 1.0, 0.0],   # 2nd quadrant -> False
                [-1.0, -1.0, 0.0],  # 3rd quadrant -> False
                [1.0, -1.0, 0.0],   # 4th quadrant -> False
                [1.0, 0.0, 0.0],    # on +x axis (p_u = 0, p_v = 1) -> True
                [0.0, 1.0, 0.0],    # on +y axis (p_u = 1, p_v = 0) -> False (without tol)
            ]
        ).T
        in_irr = sr.in_irreducible(pts)
        np.testing.assert_array_equal(in_irr, [True, False, False, False, True, False])

        in_irr_tol = sr.in_irreducible(pts, 1e-5)
        np.testing.assert_array_equal(in_irr_tol, [True, False, False, False, True, True])

        sym3 = SymopArray([sr, sr, sr])
        pts_quads = pts[:, :4]
        tr_pts = sym3.transform_pix(pts_quads)
        expected = np.ones((2, 4))
        np.testing.assert_allclose(tr_pts[:2, :], expected, atol=1e-12)

        sr_kv = SymopRotation("n", [0, 1, 0], "theta_deg", 180)
        assert sr_kv.theta_deg == 180.0
        np.testing.assert_allclose(sr_kv.n, [0, 1, 0])

        s_dict = sr.to_struct()
        sr_rec = Serializable.from_struct(s_dict)
        assert sr_rec.equal_to_tol(sr)[0]
