"""Unit tests for kf_sphere_proj (converted from test_kf_sphere_proj.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, PixelDataMemory, fast_map, kf_sphere_proj
from pyhorace.coord_transf.kf_sphere_proj import K_TO_E_MEV


class TestKfSphereProj:
    """Tests for kf_sphere_proj construction and transformations."""

    def test_default_constructor(self):
        proj = kf_sphere_proj()
        assert proj.type == "add"
        assert proj.label == ["|kf|", "\\theta", "\\phi", "En"]
        assert proj.do_3D_transformation is False
        assert proj.Ei.size == 0
        assert proj.emode == 1

    def test_nonzero_offset_throws_not_implemented(self):
        proj = kf_sphere_proj()
        with pytest.raises(HoraceError) as exc_info:
            proj.offset = [1, 0, 0, 0]
        assert "not_implemented" in exc_info.value.error_id

    def test_simple_transform_and_roundtrip(self):
        proj = kf_sphere_proj(alatt=2 * np.pi, angdeg=90, type="add")
        # Set Ei such that ki_mod == 1.0 A^-1
        proj.Ei = K_TO_E_MEV
        proj.cc_to_spec_mat = np.eye(3)
        proj.run_id_mapper = fast_map([1], [1])

        # 4 pixels at q = (1,0,0), (0,1,0), (0,0,1), (2,0,0) with dE = 10
        raw = np.zeros((9, 4), dtype=float)
        raw[0, :] = [1.0, 0.0, 0.0, 2.0]
        raw[1, :] = [0.0, 1.0, 0.0, 0.0]
        raw[2, :] = [0.0, 0.0, 1.0, 0.0]
        raw[3, :] = [0.0, 0.0, 0.0, 10.0]
        raw[4, :] = 1.0
        pix = PixelDataMemory(raw)

        img = proj.transform_pix_to_img(pix)
        # Pixel 0: q=(1,0,0), ki=(1,0,0) -> kf=(0,0,0) -> |kf|=0
        assert np.isclose(img[0, 0], 0.0)
        # Pixel 1: q=(0,1,0), ki=(1,0,0) -> kf=(-1,1,0) -> |kf|=sqrt(2), theta=135, phi=0
        assert np.isclose(img[0, 1], np.sqrt(2.0))
        assert np.isclose(img[1, 1], 135.0)
        assert np.isclose(img[2, 1], 0.0)
        # Pixel 3: q=(2,0,0), ki=(1,0,0) -> kf=(1,0,0) -> |kf|=1, theta=0, phi=0, dE=10
        assert np.isclose(img[0, 3], 1.0)
        assert np.isclose(img[1, 3], 0.0)
        assert np.isclose(img[3, 3], 10.0)

        rec = proj.transform_img_to_pix(img, pix.run_idx)
        np.testing.assert_allclose(rec, pix.coordinates, atol=1e-12)

    def test_multi_run_transform_roundtrip(self):
        proj = kf_sphere_proj(alatt=[3.0, 3.0, 3.0], angdeg=[90, 90, 90])
        proj.Ei = [25.0, 50.0]
        mats = np.zeros((3, 3, 2), dtype=float)
        mats[:, :, 0] = np.eye(3)
        mats[:, :, 1] = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        proj.cc_to_spec_mat = mats
        proj.run_id_mapper = fast_map([10, 20], [1, 2])

        rng = np.random.default_rng(99)
        raw = np.zeros((9, 6), dtype=float)
        raw[0:4, :] = rng.standard_normal((4, 6))
        raw[4, :] = [10, 20, 10, 20, 10, 20]
        pix = PixelDataMemory(raw)

        img = proj.transform_pix_to_img(pix)
        rec = proj.transform_img_to_pix(img, pix.run_idx)
        np.testing.assert_allclose(rec, pix.coordinates, atol=1e-12)
