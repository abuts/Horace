"""Unit tests for IX_aperture class, translated from MATLAB test_IX_aperture.m."""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.core.hashable import HashableArray, isequal
from pyhorace.data.instr_classes.ix_aperture import IX_aperture
from tests.data.instr_classes._helpers import (
    check_matfile_IO,
    hashable_obj_tester,
)


class TestIXAperture:
    """Port of MATLAB ``test_IX_aperture`` test suite."""

    def test_IX_aperture_all(self) -> None:
        ap = IX_aperture(-2, 0.06, 0.07)
        assert ap.distance == pytest.approx(-2.0)
        assert ap.width == pytest.approx(0.06)
        assert ap.height == pytest.approx(0.07)

        ver_ap = check_matfile_IO(ap.class_version(), False, ap)
        assert isequal(ap, ver_ap)

        ap2 = IX_aperture(-2, 0.06, 0.07, "distance", -10)
        assert ap2.distance == pytest.approx(-10.0)
        assert ap2.width == pytest.approx(0.06)
        assert ap2.height == pytest.approx(0.07)

        ap3 = IX_aperture("in-pile", 12, 0.1, 0.06)
        assert ap3.name == "in-pile"
        assert ap3.distance == pytest.approx(12.0)
        assert ap3.width == pytest.approx(0.1)
        assert ap3.height == pytest.approx(0.06)

    def test_cov_and_pdf(self) -> None:
        ap = IX_aperture("in-pile", 12, 0.1, 0.06)
        cov = ap.covariance()
        np.testing.assert_allclose(cov, np.diag([0.1**2, 0.06**2]) / 12.0, atol=1e-12)

        rng = np.random.default_rng(12345)
        X = ap.rand(1, 200000, rng=rng)
        stdev = np.std(X, axis=1)
        np.testing.assert_allclose(stdev**2, np.array([0.1**2, 0.06**2]) / 12.0, rtol=2e-2)

    def test_IX_aperture_wrong_Dist_throws(self) -> None:
        ap = IX_aperture()
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.distance = "a"
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.distance = [1, 2]

    def test_IX_aperture_wrong_width_throws(self) -> None:
        ap = IX_aperture()
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.width = "a"
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.width = -10
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.width = [1, 2]

    def test_IX_aperture_wrong_height_throws(self) -> None:
        ap = IX_aperture()
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.height = "a"
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.height = -10
        with pytest.raises(ValueError, match="invalid_argument"):
            ap.height = [1, 2]

    def test_IX_aperture_ser_WrongKey(self) -> None:
        with pytest.raises(ValueError, match="invalid_argument"):
            IX_aperture(-2, 0.06, "wrong_key", 0.07)

    def test_IX_aperture_ser_DeprKey(self) -> None:
        with pytest.warns(DeprecationWarning):
            ap = IX_aperture(-2, 0.06, "-height", 0.07)
        assert ap.distance == pytest.approx(-2.0)
        assert ap.width == pytest.approx(0.06)
        assert ap.height == pytest.approx(0.07)

    def test_hashable_prop(self) -> None:
        ap = IX_aperture(-2, 0.06, 0.07)
        hashable_obj_tester(ap)

    def test_save_load_prev_versions(self) -> None:
        ap_expect = IX_aperture("Ap0", 11, 0.2, 0.25)
        ver0_struct = {
            "name_": "Ap0",
            "distance_": 11.0,
            "width_": 0.2,
            "height_": 0.25,
        }
        ver1_struct = {
            "serial_name": "IX_aperture",
            "version": 1,
            "array_dat": {
                "name_": "Ap0",
                "distance_": 11.0,
                "width_": 0.2,
                "height_": 0.25,
                "class_version_": 1,
            },
        }
        ver0_ap = IX_aperture.loadobj(ver0_struct)
        ver1_ap = IX_aperture.from_struct(ver1_struct)
        assert isinstance(ver0_ap, IX_aperture)
        assert isinstance(ver1_ap, IX_aperture)
        assert isequal(ver0_ap, ap_expect)
        assert isequal(ver1_ap, ap_expect)

        ap_arr = HashableArray(
            [
                IX_aperture("Ap0", 11, 0.2, 0.25),
                IX_aperture("Ap1", 121, 0.22, 0.225),
            ]
        )
        arr_rec = check_matfile_IO(ap_expect.class_version(), False, ap_arr)
        assert ap_arr.eq(arr_rec)
