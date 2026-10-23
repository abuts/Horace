"""Unit tests for calc_proj_matrix, bmatrix, ubmatrix, and instr_proj (converted from test_calc_proj_matrix.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import bmatrix, calc_proj_matrix, expand_box, instr_proj, oriented_lattice, ubmatrix


class TestCalcProjMatrix:
    """Tests for bmatrix, ubmatrix, calc_proj_matrix, oriented_lattice, and instr_proj."""

    def test_bmatrix_cubic_and_triclinic(self):
        b, arlu, angrlu = bmatrix([2 * np.pi, 2 * np.pi, 2 * np.pi], [90, 90, 90], nargout=3)
        np.testing.assert_allclose(b, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(arlu, [1.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(angrlu, [90.0, 90.0, 90.0], atol=1e-12)

        b2, arlu2, angrlu2 = bmatrix([3.0, 4.0, 5.0], [80.0, 95.0, 110.0], nargout=3)
        # Reciprocal of reciprocal lattice should recover direct lattice constants
        b_inv, a_dir, ang_dir = bmatrix(arlu2, angrlu2, nargout=3)
        np.testing.assert_allclose(a_dir, [3.0, 4.0, 5.0], atol=1e-12)
        np.testing.assert_allclose(ang_dir, [80.0, 95.0, 110.0], atol=1e-12)

    def test_ubmatrix_orthogonal(self):
        b = bmatrix([3.0, 4.0, 5.0], [90, 90, 90], nargout=1)
        ub, umat = ubmatrix([1, 0, 0], [0, 1, 0], b, nargout=2)
        np.testing.assert_allclose(umat @ umat.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(umat), 1.0)
        np.testing.assert_allclose(ub, umat @ b, atol=1e-12)

    def test_calc_proj_matrix_modes(self):
        alatt = [3.0, 4.0, 5.0]
        angdeg = [90.0, 90.0, 90.0]
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        psi = np.deg2rad(30.0)
        omega = np.deg2rad(5.0)
        dpsi = np.deg2rad(2.0)
        gl = np.deg2rad(1.0)
        gs = np.deg2rad(-1.0)

        spec_to_cc, u_to_rlu, spec_to_rlu = calc_proj_matrix(
            alatt, angdeg, u, v, psi, omega, dpsi, gl, gs, 0, nargout=3
        )
        np.testing.assert_allclose(spec_to_cc @ spec_to_cc.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(u_to_rlu @ spec_to_cc, spec_to_rlu, atol=1e-12)

        # Single-matrix selectors n_matrix = 1, 2, 3
        m1 = calc_proj_matrix(alatt, angdeg, u, v, psi, omega, dpsi, gl, gs, 1)
        b = bmatrix(alatt, angdeg, nargout=1)
        np.testing.assert_allclose(b @ spec_to_rlu, m1, atol=1e-12)
        np.testing.assert_allclose(m1, spec_to_cc, atol=1e-12)

        m2 = calc_proj_matrix(alatt, angdeg, u, v, psi, omega, dpsi, gl, gs, 2)
        np.testing.assert_allclose(m2, u_to_rlu, atol=1e-12)

        m3 = calc_proj_matrix(alatt, angdeg, u, v, psi, omega, dpsi, gl, gs, 3)
        np.testing.assert_allclose(m3, spec_to_rlu, atol=1e-12)

    def test_instr_proj_transform_roundtrip(self):
        lat = oriented_lattice([3.0, 4.0, 5.0], [90, 90, 90], 25.0, [1, 1, 0], [0, 0, 1])
        ip = instr_proj(lat)
        pts = np.array([[1.0, 2.0], [0.5, -0.5], [0.2, 1.2], [5.0, 10.0]])
        cc = ip.transform_pix_to_img(pts)
        rec = ip.transform_img_to_pix(cc)
        np.testing.assert_allclose(rec, pts, atol=1e-12)

    def test_expand_box(self):
        r_min = np.array([0.0, 0.0, 0.0])
        r_max = np.array([1.0, 2.0, 3.0])
        box = expand_box(r_min, r_max)
        assert box.shape == (3, 8)
        np.testing.assert_allclose(np.min(box, axis=1), r_min)
        np.testing.assert_allclose(np.max(box, axis=1), r_max)
