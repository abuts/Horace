"""Pytest suite for Symop base class, SymopIdentity, SymopGeneral, and symop factory.

Translated from MATLAB ``_test/test_sym_op/test_symop.m``.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from pyhorace.coord_transf import (
    HoraceError,
    Symop,
    SymopArray,
    SymopGeneral,
    SymopIdentity,
    SymopReflection,
    SymopRotation,
    bmatrix,
    line_proj,
    sphere_proj,
    symop,
    ubmat_proj,
)
from pyhorace.core.serializable import Serializable
from pyhorace.data.pixels import PixelDataMemory


class TestSymop:
    """Port of MATLAB ``test_symop`` test cases."""

    refl_op = SymopReflection([1, 0, 0], [0, 0, 1])  # Reflection in Y
    rot_op = SymopRotation([1, 0, 0], 90)  # Rotation 90deg about X
    rot_op_mat = SymopGeneral([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    mot_op_comp = SymopArray(
        [
            SymopRotation([0, 1, 0], 90),
            SymopRotation([0, 0, 1], 90),
            SymopReflection([1, 0, 0], [0, 1, 0]),
            SymopReflection([0, 0, 1], [1, 0, 0]),
        ]
    )
    mot_op = SymopGeneral([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

    proj = line_proj([1, 0, 0], [0, 1, 0], alatt=[3, 3, 3], angdeg=[90, 90, 90])
    nort_proj = line_proj([1, 0, 0], [0, 1, 0], alatt=[1, 2, 3], angdeg=[80, 70, 120])

    points2transform = np.column_stack(
        [np.eye(3), [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    )

    def _check_proj_transformation_correct(self, op, proj):
        out_proj, op_out = op.transform_proj(proj, nargout=2)
        sym_pts = op_out.transform_vec(self.points2transform)
        sym_img_pts = proj.transform_pix_to_img(sym_pts)
        proj_sym_pts = out_proj.transform_pix_to_img(self.points2transform)
        np.testing.assert_allclose(sym_img_pts, proj_sym_pts, atol=1e-14)
        rev_pts = out_proj.transform_img_to_pix(proj_sym_pts)
        np.testing.assert_allclose(self.points2transform, rev_pts, atol=1e-14)
        return out_proj, op_out

    def test_symop_transform_projection_eq_symop(self):
        refl = SymopReflection([1, 1, 0], [0, 0, 1])
        assert refl.input_nrmv_in_rlu

        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=1, angdeg=90)
        lp1, refl = refl.transform_proj(lp, nargout=2)
        assert not refl.input_nrmv_in_rlu

        pix = np.column_stack([np.eye(3), [1, 1, 0], [0, 1, 1], [1, 0, 1]])
        transf_pix = lp1.transform_pix_to_img(pix)
        pix_cc = lp.transform_img_to_pix(transf_pix)

        sym_pix = refl.transform_vec(pix)
        np.testing.assert_allclose(pix_cc, sym_pix, atol=1e-14)

        sym_pix = refl.transform_pix(pix)
        nrefl_pix = ~(pix.T @ refl.normvec > 0)
        np.testing.assert_allclose(
            pix_cc[:, nrefl_pix], sym_pix[:, nrefl_pix], atol=1e-14
        )

    def test_symop_transform_projection_eq_symop_100(self):
        refl = SymopReflection([1, 0, 0], [0, 0, 1])
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=1, angdeg=90)
        lp1, refl = refl.transform_proj(lp, nargout=2)
        assert not refl.input_nrmv_in_rlu

        pix = np.column_stack([np.eye(3), [1, 1, 0], [0, 1, 1], [1, 0, 1]])
        transf_pix = lp1.transform_pix_to_img(pix)
        pix_cc = lp.transform_img_to_pix(transf_pix)

        sym_pix = refl.transform_pix(pix)
        np.testing.assert_allclose(pix_cc, sym_pix, atol=1e-14)

    def test_symop_create_identity(self):
        out = Symop.create("I", np.eye(3))
        assert isinstance(out, SymopIdentity)

    def test_identity_constructor(self):
        out = SymopIdentity()
        assert isinstance(out, SymopIdentity)

        out = SymopIdentity(np.eye(3))
        assert isinstance(out, SymopIdentity)

        out = SymopIdentity(np.eye(3), [0, 0, 0])
        assert isinstance(out, SymopIdentity)

    def test_symop_create_matrix(self):
        out = Symop.create("General", [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        assert isinstance(out, Symop)
        np.testing.assert_array_equal(out.R, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(out.offset, [0, 0, 0])

    def test_matrix_constructor(self):
        out = Symop.create(
            "General", [[0, 0, -1], [-1, 0, 0], [0, 1, 0]], [3, 3, 3]
        )
        assert isinstance(out, Symop)
        np.testing.assert_array_equal(out.W, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        np.testing.assert_array_equal(out.offset, [3, 3, 3])

    def test_apply_vec_indentity(self):
        op = SymopIdentity()
        testvec = np.array([1.0, 0.0, 0.0])
        outvec = op.transform_vec(testvec)
        np.testing.assert_allclose(outvec, testvec)

    def test_apply_vec_rotation(self):
        op = SymopRotation([0, 1, 0], 90)
        testvec = np.array([1.0, 0.0, 0.0])
        outvec = op.transform_vec(testvec)
        np.testing.assert_allclose(outvec, [0.0, 0.0, -1.0], atol=1e-10)

    def test_apply_matrix_rotation(self):
        op = SymopRotation([0, 1, 0], 90)
        testvec = np.eye(3)
        outvec = op.transform_vec(testvec)
        np.testing.assert_allclose(
            outvec, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-10
        )

    def test_apply_matrix_reflection(self):
        op = SymopReflection([0, 1, 0], [0, 0, 1])
        testvec = self.points2transform.copy()
        outvec = op.transform_vec(testvec)
        ref_v = testvec.copy()
        ref_v[0, :] = -ref_v[0, :]
        np.testing.assert_allclose(outvec, ref_v)

    def test_apply_vec_matrix(self):
        op = SymopGeneral([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        testvec = np.array([1.0, 0.0, 0.0])
        outvec = op.transform_vec(testvec)
        np.testing.assert_allclose(outvec, [0.0, -1.0, 0.0])

    def test_apply_matrix_matrix(self):
        op = SymopGeneral([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        testvec = np.eye(3)
        outvec = op.transform_vec(testvec)
        np.testing.assert_allclose(outvec, [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    def test_apply_matrix_comp(self):
        outvec_comp = self.mot_op_comp.transform_vec(self.points2transform)
        outvec = self.mot_op.transform_vec(self.points2transform)
        np.testing.assert_allclose(outvec, outvec_comp, atol=1e-10)

    def test_apply_proj_rotation(self):
        out_proj, _ = self._check_proj_transformation_correct(self.rot_op, self.proj)
        out_proj_mat, _ = self._check_proj_transformation_correct(
            self.rot_op_mat, self.proj
        )
        np.testing.assert_allclose(
            out_proj.sym_transf, out_proj_mat.sym_transf, atol=1e-12
        )

    def test_combined_with_shift_works_on_non_orth(self):
        op = SymopArray(
            [
                SymopReflection([1, 0, 0], [0, 0, 1], [0, 1, 0], "cc"),
                SymopRotation([1, 0, 0], 60, "cc"),
            ]
        )
        out_proj, _ = self._check_proj_transformation_correct(op, self.nort_proj)
        assert not np.all(out_proj.offset == 0)

    def test_folding_with_shift_works_on_non_orth(self):
        op = SymopReflection([1, 0, 0], [0, 0, 1], [0, 1, 0])
        out_proj, op = self._check_proj_transformation_correct(op, self.nort_proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R)
        assert not np.all(out_proj.offset == 0)

    def test_folding_with_shift100_works(self):
        op = SymopReflection([1, 0, 0], [0, 0, 1], [1, 0, 0])
        out_proj, op = self._check_proj_transformation_correct(op, self.proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R)
        assert np.all(out_proj.offset == 0)

    def test_folding_with_shift010_works(self):
        op = SymopReflection([1, 0, 0], [0, 0, 1], [0, 1, 0])
        out_proj, op = self._check_proj_transformation_correct(op, self.proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R)
        np.testing.assert_allclose(out_proj.offset, [0, -2, 0, 0], atol=1e-14)

    def test_rotation_with_shift_works_on_non_orth(self):
        op = SymopRotation([0, 1, 0], 60, [0, 1, 0], "rlu")
        out_proj, op = self._check_proj_transformation_correct(op, self.nort_proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        assert np.all(np.abs(out_proj.offset) < 1e-14)

    def test_rotation_with_shift_on_proj_100_works(self):
        op = SymopRotation([0, 1, 0], 90)
        sproj = copy.deepcopy(self.proj)
        sproj.offset = [1, 0, 0, 0]
        out_proj, op = self._check_proj_transformation_correct(op, sproj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        np.testing.assert_allclose(out_proj.offset, [1, 0, 0, 0], atol=1e-14)

    def test_rotation_with_shift100_works(self):
        op = SymopRotation([0, 1, 0], 60, [1, 0, 0])
        out_proj, op = self._check_proj_transformation_correct(op, self.proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        assert not np.all(out_proj.offset == 0)

    def test_rotation_with_2shifts010_works(self):
        op = SymopRotation([0, 1, 0], 60, [0, 1, 0])
        sproj = copy.deepcopy(self.proj)
        sproj.offset = [0, 1, 0, 0]
        out_proj, op = self._check_proj_transformation_correct(op, sproj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        np.testing.assert_allclose(out_proj.offset, [0, 1, 0, 0], atol=1e-14)

    def test_rotation_with_shift010_on_proj_works(self):
        op = SymopRotation([0, 1, 0], 60)
        sproj = copy.deepcopy(self.proj)
        sproj.offset = [0, 1, 0, 0]
        out_proj, op = self._check_proj_transformation_correct(op, sproj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        np.testing.assert_allclose(out_proj.offset, [0, 1, 0, 0], atol=1e-14)

    def test_rotation_with_shift010_works(self):
        op = SymopRotation([0, 1, 0], 60, [0, 1, 0])
        out_proj, op = self._check_proj_transformation_correct(op, self.proj)
        np.testing.assert_allclose(out_proj.sym_transf, op.R, atol=1e-14)
        np.testing.assert_allclose(out_proj.offset, [0, 0, 0, 0], atol=1e-14)

    def test_apply_proj_comp(self):
        out_proj = self.mot_op.transform_proj(self.proj)
        out_proj2 = self.mot_op_comp.transform_proj(self.proj)
        np.testing.assert_allclose(
            out_proj.sym_transf, out_proj2.sym_transf, atol=1e-12
        )

    def test_apply_ref_ref_back(self):
        out_proj, _ = self._check_proj_transformation_correct(
            self.refl_op, self.proj
        )
        ref_transf = np.eye(3)
        ref_transf[1, 1] = -1
        np.testing.assert_allclose(out_proj.sym_transf, ref_transf, atol=1e-14)

        out_proj2, _ = self._check_proj_transformation_correct(
            self.refl_op, out_proj
        )
        assert out_proj2.sym_transf.size == 0

    def test_apply_ref_ref_back_comp(self):
        op = SymopArray([self.refl_op, self.refl_op])
        out_proj, _ = self._check_proj_transformation_correct(op, self.proj)
        assert out_proj.sym_transf.size == 0

    def test_apply_rot_180_180(self):
        op = SymopArray(
            [SymopRotation([1, 0, 0], 180), SymopRotation([1, 0, 0], 180)]
        )
        out_proj, _ = self._check_proj_transformation_correct(op, self.proj)
        assert out_proj.sym_transf.size == 0

    def test_apply_rot_360(self):
        op = SymopArray([SymopRotation([1, 0, 0], 360)])
        out_proj, _ = self._check_proj_transformation_correct(op, self.proj)
        assert out_proj.sym_transf.size == 0

    def test_apply_rot_60_min_60_nonortho(self):
        op = SymopArray(
            [
                SymopRotation([1, 0, 0], 60, "cc"),
                SymopRotation([1, 0, 0], -60, "cc"),
            ]
        )
        out_proj, _ = self._check_proj_transformation_correct(op, self.nort_proj)
        assert out_proj.sym_transf.size == 0

    def test_apply_rot60_min60_with_symOffset(self):
        op = SymopArray(
            [
                SymopRotation([1, 0, 0], 60, [1, 1, 0]),
                SymopRotation([1, 0, 0], -60, [1, 1, 0]),
            ]
        )
        out_proj, _ = self._check_proj_transformation_correct(op, self.proj)
        assert out_proj.sym_transf.size == 0
        assert np.all(out_proj.offset == 0)

    def test_apply_rot_60_min_60(self):
        op = SymopArray(
            [SymopRotation([1, 0, 0], 60), SymopRotation([1, 0, 0], -60)]
        )
        out_proj, _ = self._check_proj_transformation_correct(op, self.proj)
        assert out_proj.sym_transf.size == 0

    def test_create_unknown_operation(self):
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument") as exc_info:
            Symop.create("SomeOperation", 1, 2, 3)
        assert "Short name" in str(exc_info.value)

    def test_validate_and_generate_sym_errors(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym("invalid", proj)

        s1 = SymopReflection([1, 0, 0], [0, 1, 0])
        r1 = SymopRotation([0, 0, 1], 90)
        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:not_implemented"):
            Symop.validate_and_generate_sym([s1, r1], proj)

    def test_validate_and_generate_sym_identity(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2, angdeg=90)
        sym = SymopIdentity()
        sym_res, fold = Symop.validate_and_generate_sym(sym, proj)
        assert fold == 1
        assert isinstance(sym_res, SymopIdentity)
        np.testing.assert_allclose(sym_res.b_matrix, proj.bmatrix(3))

    def test_validate_and_generate_sym_reflection(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2, angdeg=90)
        sym = SymopReflection([1, 0, 0], [0, 1, 0])
        sym_res, fold = Symop.validate_and_generate_sym(sym, proj)
        assert fold == 1
        assert isinstance(sym_res, SymopReflection)
        np.testing.assert_allclose(sym_res.b_matrix, proj.bmatrix(3))

        s1 = SymopReflection([1, 0, 0], [0, 1, 0])
        s2 = SymopReflection([0, 1, 0], [0, 0, 1])
        sym_arr = SymopArray([s1, s2])
        sym_res, fold = Symop.validate_and_generate_sym(sym_arr, proj)
        assert fold == 2
        assert isinstance(sym_res, SymopArray)
        np.testing.assert_allclose(sym_res[0].b_matrix, proj.bmatrix(3))
        np.testing.assert_allclose(sym_res[1].b_matrix, proj.bmatrix(3))

        s2_off = SymopReflection([0, 1, 0], [0, 0, 1], [1, 0, 0])
        sym_arr_diff = SymopArray([s1, s2_off])
        sym_res_diff, fold = Symop.validate_and_generate_sym(sym_arr_diff, proj)
        assert fold == 2
        assert isinstance(sym_res_diff, list) and not isinstance(sym_res_diff, SymopArray)
        np.testing.assert_allclose(sym_res_diff[0].b_matrix, proj.bmatrix(3))
        np.testing.assert_allclose(sym_res_diff[1].b_matrix, proj.bmatrix(3))

        sym_cell = [s1, s2]
        sym_res_cell, fold = Symop.validate_and_generate_sym(sym_cell, proj)
        assert fold == 2
        assert isinstance(sym_res_cell, list)

    def test_validate_and_generate_sym_rotation(self):
        proj = line_proj([1, 0, 0], [0, 1, 0], alatt=2, angdeg=90)
        sym = SymopRotation([0, 0, 1], 90)
        sym_res, fold = Symop.validate_and_generate_sym(sym, proj)
        assert fold == 3
        assert isinstance(sym_res, list)
        assert len(sym_res) == 3
        for op in sym_res:
            assert isinstance(op, SymopRotation)
            assert op.theta_deg == 90
            np.testing.assert_allclose(op.b_matrix, proj.bmatrix(3))

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(SymopRotation([0, 0, 1], 70), proj)

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(
                SymopArray([SymopRotation([0, 0, 1], 90), SymopRotation([0, 0, 1], 180)]),
                proj,
            )

        sym_fold_cell = list(SymopRotation.fold(4, [0, 0, 1]))
        sym_res, fold = Symop.validate_and_generate_sym(sym_fold_cell, proj)
        assert fold == 3
        assert len(sym_res) == 3
        for op in sym_res:
            assert op.theta_deg == 90

        sym_rep_cell = [
            SymopIdentity(),
            SymopRotation([0, 0, 1], 90),
            SymopRotation([0, 0, 1], 90),
            SymopRotation([0, 0, 1], 90),
        ]
        sym_res, fold = Symop.validate_and_generate_sym(sym_rep_cell, proj)
        assert fold == 3
        assert len(sym_res) == 3

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(
                [SymopRotation([0, 0, 1], 90), SymopRotation([0, 0, 1], 180)], proj
            )

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(
                [SymopIdentity(), SymopRotation([0, 0, 1], 70)], proj
            )

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(
                [
                    SymopIdentity(),
                    SymopRotation([0, 0, 1], 90),
                    SymopRotation([0, 0, 1], 180),
                    SymopRotation([0, 0, 1], 200),
                ],
                proj,
            )

        with pytest.raises(HoraceError, match="HORACE:symmetrise_sqw:invalid_argument"):
            Symop.validate_and_generate_sym(
                [
                    SymopIdentity(),
                    SymopRotation([0, 0, 1], 90),
                    SymopRotation([0, 0, 1], 180),
                    SymopRotation([0, 0, 1], 270, [1, 0, 0]),
                ],
                proj,
            )

    def test_transform_proj_errors(self):
        s = SymopIdentity()
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            s.transform_proj("not_a_proj")

        sp = sphere_proj(alatt=2 * np.pi, angdeg=90)
        sp_tr = s.transform_proj(sp)
        assert sp_tr.equal_to_tol(sp)[0]

        sr = SymopReflection([1, 0, 0], [0, 1, 0])
        with pytest.raises(HoraceError, match="HORACE:Symop:not_implemented"):
            sr.transform_proj(sp)

        s1 = SymopReflection([1, 0, 0], [0, 1, 0], [1, 0, 0])
        s2 = SymopReflection([0, 1, 0], [0, 0, 1], [0, 1, 0])
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        with pytest.raises(HoraceError, match="HORACE:Symop:not_implemented"):
            SymopArray([s1, s2]).transform_proj(lp)

    def test_transform_proj_offset_propagated_on_other_symmetries(self):
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        s1 = SymopReflection([1, 0, 0], [0, 1, 0], [1, 0, 0])
        s2 = SymopReflection([0, 1, 0], [0, 0, 1])
        sym = SymopArray([s1, s2])
        lp_tr, sym_tr = sym.transform_proj(lp, nargout=2)
        assert isinstance(sym_tr, SymopGeneral)
        np.testing.assert_allclose(sym_tr.offset, [1, 0, 0])
        np.testing.assert_allclose(lp_tr.offset, [-2, 0, 0, 0], atol=1e-12)

    def test_transform_proj_2refl_no_offset(self):
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        s1 = SymopReflection([1, 0, 0], [0, 0, 1])
        s2 = SymopReflection([0, 1, 0], [0, 0, 1])
        sym = SymopArray([s1, s2])
        lp_tr, sym_tr = sym.transform_proj(lp, nargout=2)
        np.testing.assert_allclose(lp_tr.sym_transf, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)
        np.testing.assert_allclose(sym_tr.R, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)

        ubp = ubmat_proj(alatt=2 * np.pi, angdeg=90)
        lp_from_ubp = sym.transform_proj(ubp)
        assert isinstance(lp_from_ubp, line_proj)
        np.testing.assert_allclose(
            lp_from_ubp.sym_transf, np.diag([-1.0, -1.0, 1.0]), atol=1e-12
        )

    def test_transform_pix_1op(self):
        s = SymopReflection([1, 0, 0], [0, 1, 0])
        coords = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, -3.0]]).T
        res = s.transform_pix(coords)
        expected = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).T
        np.testing.assert_allclose(res, expected)

        pix = PixelDataMemory(np.vstack([coords, np.ones((6, 2))]))
        pix_res = s.transform_pix(pix)
        np.testing.assert_allclose(pix_res.q_coordinates, expected)

        with pytest.raises(HoraceError, match="HORACE:symop:invalid_argument"):
            s.transform_pix("invalid")
        with pytest.raises(HoraceError, match="HORACE:symop:invalid_argument"):
            s.transform_pix(np.ones((2, 3)))

    def test_symop_general_constructor(self):
        sg = SymopGeneral()
        np.testing.assert_allclose(sg.R, np.eye(3))
        np.testing.assert_allclose(sg.W, np.eye(3))
        np.testing.assert_allclose(sg.offset, [0, 0, 0])
        assert sg.b_matrix.size == 0

        r_mat = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        bm = bmatrix([2, 2, 2], [90, 90, 90])
        sg2 = SymopGeneral(r_mat, [1, 0, 0], bm)
        np.testing.assert_allclose(sg2.R, r_mat)
        np.testing.assert_allclose(sg2.W, r_mat)
        np.testing.assert_allclose(sg2.offset, [1, 0, 0])
        np.testing.assert_allclose(sg2.b_matrix, bm)

        sg_copy = SymopGeneral(sg2)
        assert sg_copy.equal_to_tol(sg2)[0]

        s_dict = sg2.to_struct()
        sg_rec = Serializable.from_struct(s_dict)
        assert sg_rec.equal_to_tol(sg2)[0]

        with pytest.raises(HoraceError, match="HORACE:symop:invalid_argument"):
            SymopGeneral(2.0 * np.eye(3))
        with pytest.raises(HoraceError, match="HORACE:symop:invalid_argument"):
            SymopGeneral(np.ones((2, 3)))
        with pytest.raises(HoraceError, match="HORACE:SymopRotation:invalid_argument"):
            SymopGeneral(np.eye(3), [0, 0, 0], np.eye(3), "extra")
        with pytest.raises(HoraceError, match="HORACE:symop:not_implemented"):
            sg2.in_irreducible(np.zeros((3, 2)))

    def test_generic_symop_create(self):
        with pytest.warns(UserWarning, match="deprecated"):
            si = symop("identity")
        assert isinstance(si, SymopIdentity)

        s_id = Symop.create("SymopIdentity")
        assert isinstance(s_id, SymopIdentity)

        s_refl = Symop.create("reflection", [1, 0, 0], [0, 1, 0])
        assert isinstance(s_refl, SymopReflection)

        s_rot = Symop.create("rot", [0, 0, 1], 90)
        assert isinstance(s_rot, SymopRotation)
        assert s_rot.theta_deg == 90

        s_fold = Symop.create("rotation", "fold", 4, [0, 0, 1])
        assert isinstance(s_fold, SymopArray)
        assert len(s_fold) == 4
        assert isinstance(s_fold[0], SymopIdentity)
        assert isinstance(s_fold[1], SymopRotation)

        s_gen = Symop.create("general", np.eye(3))
        assert isinstance(s_gen, SymopGeneral)

        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            Symop.create(123)
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            Symop.create("r")
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            Symop.create("symop")
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            Symop.create("rote")
        with pytest.raises(HoraceError, match="HORACE:Symop:invalid_argument"):
            Symop.create("unknown")
