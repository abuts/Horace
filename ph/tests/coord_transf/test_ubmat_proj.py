"""Unit tests for ubmat_proj (converted from test_ubmat_proj.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, crystal_alignment_info, line_proj, ubmat_proj


class TestUbmatProj:
    """Tests for ubmat_proj construction, transformations, and alignment."""

    def test_default_constructor(self):
        proj = ubmat_proj()
        np.testing.assert_allclose(proj.u_to_rlu, np.eye(4))
        np.testing.assert_allclose(proj.ulen, np.ones(4))
        assert proj.label == ["\\zeta", "\\xi", "\\eta", "E"]

    def test_invalid_u_to_rlu_throws(self):
        with pytest.raises(HoraceError):
            ubmat_proj(np.zeros((3, 3)))
        with pytest.raises(HoraceError):
            ubmat_proj(np.ones((2, 2)))

    def test_equivalence_with_line_proj(self):
        lp = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 1],
            alatt=[3.2, 3.2, 6.4],
            angdeg=[90, 90, 90],
            offset=[0.2, 0.3, 0.4, 1.0],
        )
        ubp = ubmat_proj(
            lp.u_to_rlu,
            lp.img_scales,
            alatt=lp.alatt,
            angdeg=lp.angdeg,
            offset=lp.offset,
            label=lp.label,
        )
        pts = np.array([[1.0, -0.5, 2.0], [0.3, 1.2, -0.8], [0.5, 0.0, 1.1], [2.0, 4.0, 6.0]])
        img_lp = lp.transform_pix_to_img(pts)
        img_ub = ubp.transform_pix_to_img(pts)
        np.testing.assert_allclose(img_ub, img_lp, atol=1e-12)

        pix_rec = ubp.transform_img_to_pix(img_ub)
        np.testing.assert_allclose(pix_rec, pts, atol=1e-12)

    def test_serialization_roundtrip(self):
        lp = line_proj([1, 2, 0], [-2, 1, 0], alatt=[3.0, 4.0, 5.0], angdeg=[90, 90, 90])
        ubp = lp.get_ubmat_proj()
        s = ubp.to_struct()
        rec = ubmat_proj.from_struct(s)
        assert ubp == rec
        np.testing.assert_allclose(rec.u_to_rlu, ubp.u_to_rlu)
        np.testing.assert_allclose(rec.ulen, ubp.ulen)

    def test_align_proj(self):
        lp = line_proj([1, 0, 0], [0, 1, 0], alatt=[3.0, 3.0, 3.0], angdeg=[90, 90, 90], offset=[1, 0, 0, 0])
        ubp = lp.get_ubmat_proj()
        align_info = crystal_alignment_info([3.1, 3.1, 3.1], [90, 90, 90], [0.0, 0.0, 0.1])
        align_info.hkl_mode = True
        ubp.align_proj(align_info)
        np.testing.assert_allclose(ubp.alatt, [3.1, 3.1, 3.1])
