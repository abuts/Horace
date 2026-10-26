"""Unit tests for Goniometer class, translated from MATLAB test_goniometer.m."""

from __future__ import annotations

import warnings
import numpy as np
import pytest

from pyhorace.coord_transf import calc_proj_matrix
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.goniometer import Goniometer
from pyhorace.data.instr_classes.ix_experiment import IX_experiment
from pyhorace.data.instr_classes.ix_map import IX_map
from pyhorace.data.instr_classes.ix_mask import IX_mask
from tests.data.instr_classes._helpers import hashable_obj_tester


class TestGoniometer:
    """Port of MATLAB ``test_goniometer`` test suite."""

    def test_constructor_defaults(self) -> None:
        ol = Goniometer()
        assert ol.psi == 0.0
        assert ol.undef_fields == ["psi"]
        assert not ol.is_defined("psi")
        assert not ol.is_defined("alatt")
        assert not ol.is_defined("angdeg")
        assert not ol.alatt_defined()
        assert not ol.angdeg_defined()
        assert not ol.lattice_parameters_defined()
        np.testing.assert_allclose(ol.u, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(ol.v, [0.0, 1.0, 0.0])
        assert ol.omega == 0.0
        assert ol.dpsi == 0.0
        assert ol.gl == 0.0
        assert ol.gs == 0.0
        assert ol.angular_units == "deg"
        assert ol.angular_is_degree is True
        assert ol.degrees is True
        assert ol.radians is False

    def test_serial_fields(self) -> None:
        ol = Goniometer()
        sf = ol.saveable_fields()
        assert sf == (
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
        )
        cf = ol.construction_fields()
        assert cf == (
            "efix",
            "emode",
            "alatt",
            "angdeg",
            "u",
            "v",
            "psi",
            "omega",
            "dpsi",
            "gl",
            "gs",
            "angular_units",
        )
        assert Goniometer.fields_with_defaults() == [
            "omega",
            "dpsi",
            "gl",
            "gs",
            "u",
            "v",
        ]

    def test_degrees_rad(self) -> None:
        ol = Goniometer()
        ol.psi = 10.0
        ol.omega = 20.0
        ol.dpsi = 2.0
        ol.gl = 3.0
        ol.gs = -3.0

        assert ol.is_defined("psi")
        assert ol.undef_fields == []
        assert ol.angular_units == "deg"

        ol.angular_units = "rad"
        assert ol.angular_units == "rad"
        assert ol.radians is True
        assert ol.degrees is False
        d2r = np.pi / 180.0
        assert ol.psi == pytest.approx(10.0 * d2r)
        assert ol.omega == pytest.approx(20.0 * d2r)
        assert ol.dpsi == pytest.approx(2.0 * d2r)
        assert ol.gl == pytest.approx(3.0 * d2r)
        assert ol.gs == pytest.approx(-3.0 * d2r)

        # Setting in radians and reading back in degrees
        ol.psi = 0.5
        ol.convert_to_degrees()
        assert ol.angular_units == "deg"
        assert ol.psi == pytest.approx(0.5 * (180.0 / np.pi))

        # Out-of-bounds angle throws
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.psi = 361.0
        ol.convert_to_radians()
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.omega = 2.0 * np.pi + 0.1
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.angular_units = "invalid_unit"

    def test_3Dvectors(self) -> None:
        ol = Goniometer()
        ol.u = [1, 2, 3]
        ol.v = [[3], [2], [1]]
        np.testing.assert_allclose(ol.u, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.v, [3.0, 2.0, 1.0])
        np.testing.assert_allclose(ol.cu, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.cv, [3.0, 2.0, 1.0])

        # Collinear u and v must throw
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.v = [2, 4, 6]

        # Zero vector must throw
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.u = [0, 0, 0]

        # Invalid length must throw
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.u = [1, 2]

        # Lattice parameters
        ol.alatt = [2, 3, 4]
        ol.angdeg = [80, 90, 100]
        assert ol.alatt_defined()
        assert ol.angdeg_defined()
        assert ol.lattice_parameters_defined()
        np.testing.assert_allclose(ol.alatt, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(ol.angdeg, [80.0, 90.0, 100.0])

        # Scalar expansion for alatt and angdeg
        ol.alatt = 5.0
        ol.angdeg = 90.0
        np.testing.assert_allclose(ol.alatt, [5.0, 5.0, 5.0])
        np.testing.assert_allclose(ol.angdeg, [90.0, 90.0, 90.0])

        with pytest.raises(ValueError, match="invalid_argument"):
            ol.alatt = [1, -2, 3]
        with pytest.raises(ValueError, match="invalid_argument"):
            ol.angdeg = [90, 180, 90]

    def test_full_constructor(self) -> None:
        ol = Goniometer(10, [1, 2, 3], [3, 2, 1], 20, 2, 3, -3)
        assert ol.psi == pytest.approx(10.0)
        np.testing.assert_allclose(ol.u, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.v, [3.0, 2.0, 1.0])
        assert ol.omega == pytest.approx(20.0)
        assert ol.dpsi == pytest.approx(2.0)
        assert ol.gl == pytest.approx(3.0)
        assert ol.gs == pytest.approx(-3.0)
        assert ol.angular_units == "deg"

        # With trailing 'rad'
        d2r = np.pi / 180.0
        ol_rad = Goniometer(
            10 * d2r,
            [1, 2, 3],
            [3, 2, 1],
            20 * d2r,
            2 * d2r,
            3 * d2r,
            -3 * d2r,
            "rad",
        )
        assert ol_rad.angular_units == "rad"
        ol_rad.angular_units = "deg"
        assert ol_rad == ol

    def test_full_constructor_with_keyval(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ol = Goniometer(
                "psi",
                10,
                "u",
                [1, 2, 3],
                "-v",
                [3, 2, 1],
                "omega",
                20,
                "dpsi",
                2,
                "-gl",
                3,
                "gs",
                -3,
            )
        assert ol.psi == pytest.approx(10.0)
        np.testing.assert_allclose(ol.u, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.v, [3.0, 2.0, 1.0])
        assert ol.omega == pytest.approx(20.0)
        assert ol.dpsi == pytest.approx(2.0)
        assert ol.gl == pytest.approx(3.0)
        assert ol.gs == pytest.approx(-3.0)

    def test_constructor_with_wrong_keyval_throw(self) -> None:
        with pytest.raises(ValueError, match="invalid_argument"):
            Goniometer("psi", 10, "wrong_field", 20)

        with pytest.raises(ValueError, match="invalid_argument"):
            Goniometer(10, [1, 2, 3], "psi")

    def test_mixed_constructor_with_keyval(self) -> None:
        ol = Goniometer(10, [1, 2, 3], [3, 2, 1], "gl", 3, "gs", -3)
        assert ol.psi == pytest.approx(10.0)
        np.testing.assert_allclose(ol.u, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.v, [3.0, 2.0, 1.0])
        assert ol.omega == pytest.approx(0.0)
        assert ol.dpsi == pytest.approx(0.0)
        assert ol.gl == pytest.approx(3.0)
        assert ol.gs == pytest.approx(-3.0)

    def test_mixed_constructor_with_structure(self) -> None:
        s = {"psi": 10, "u": [1, 2, 3], "v": [3, 2, 1], "gl": 2}
        ol = Goniometer(s)
        assert ol.psi == pytest.approx(10.0)
        np.testing.assert_allclose(ol.u, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ol.v, [3.0, 2.0, 1.0])
        assert ol.gl == pytest.approx(2.0)

        # Override struct fields with key-value arguments
        ol2 = Goniometer(s, "gl", 5, "gs", -2)
        assert ol2.psi == pytest.approx(10.0)
        assert ol2.gl == pytest.approx(5.0)
        assert ol2.gs == pytest.approx(-2.0)

    def test_save_load_serializable_obj(self) -> None:
        ol = Goniometer({"psi": 10, "u": [1, 0, 0], "v": [0, 1, 0], "gl": 2})
        struc = ol.to_struct()
        ol_rec = Serializable.from_struct(struc)
        assert ol == ol_rec

        ol.angular_units = "rad"
        struc_rad = ol.to_struct()
        ol_rad_rec = Serializable.from_struct(struc_rad)
        assert ol_rad_rec.angular_units == "rad"
        assert ol == ol_rad_rec

    def test_unit_matrixes(self) -> None:
        ol = Goniometer(0, [1, 0, 0], [0, 1, 0])
        mat1, ol = ol.calc_proj_matrix([2 * np.pi, 2 * np.pi, 2 * np.pi], [90, 90, 90], 1)
        assert ol.angular_units == "rad"
        np.testing.assert_allclose(mat1, np.eye(3), atol=1e-12)

        mat2 = ol.calc_proj_matrix(2)
        np.testing.assert_allclose(mat2, np.eye(3), atol=1e-12)

        mat3 = ol.calc_proj_matrix(3)
        np.testing.assert_allclose(mat3, np.eye(3), atol=1e-12)

    def test_different_matrixes(self) -> None:
        ol = Goniometer(10, [1, 2, 3], [3, 2, 1], 20, 2, 3, -3, "deg")
        alatt = [2.0, 3.0, 4.0]
        angdeg = [30.0, 40.0, 50.0]

        mat1, ol = ol.calc_proj_matrix(alatt, angdeg, 1)
        assert ol.angular_units == "rad"
        mat2 = ol.calc_proj_matrix(2)
        mat3 = ol.calc_proj_matrix(3)

        spec_to_cc, u_to_rlu, spec_to_rlu = ol.calc_projections()
        np.testing.assert_allclose(mat1, spec_to_cc, atol=1e-12)
        np.testing.assert_allclose(mat2, u_to_rlu, atol=1e-12)
        np.testing.assert_allclose(mat3, spec_to_rlu, atol=1e-12)

        # Compare against direct calc_proj_matrix function call
        d2r = np.pi / 180.0
        ref1, ref2, ref3 = calc_proj_matrix(
            alatt,
            angdeg,
            [1, 2, 3],
            [3, 2, 1],
            10 * d2r,
            20 * d2r,
            2 * d2r,
            3 * d2r,
            -3 * d2r,
            nargout=3,
        )
        np.testing.assert_allclose(mat1, ref1, atol=1e-12)
        np.testing.assert_allclose(mat2, ref2, atol=1e-12)
        np.testing.assert_allclose(mat3, ref3, atol=1e-12)

    def test_hashable_prop(self) -> None:
        gon = Goniometer()
        hashable_obj_tester(gon)


class TestIXExperiment:
    """Unit tests for IX_experiment."""

    def test_defaults_and_properties(self) -> None:
        exp = IX_experiment()
        assert exp.filename == ""
        assert exp.filepath == ""
        assert np.isnan(exp.run_id)
        assert exp.en.size == 0
        assert exp.efix == 0.0
        assert exp.emode == 0
        assert exp.angular_units == "deg"

        exp.filename = "run1.nxspe"
        exp.filepath = "/data/"
        exp.run_id = 10
        exp.en = [1.0, 2.0, 3.0]
        exp.efix = 25.0
        exp.emode = 1
        exp.alatt = [3.0, 4.0, 5.0]
        exp.angdeg = [90.0, 90.0, 90.0]
        exp.psi = 15.0

        assert exp.filename == "run1.nxspe"
        assert exp.filepath == "/data/"
        assert exp.run_id == 10.0
        np.testing.assert_allclose(exp.get_en(bin_centers=True), [1.5, 2.5])
        assert exp.get_efix() == 25.0
        assert exp.get_emode() == 1
        assert exp.get_run_ids() == 10.0

        # Serialization round-trip
        s = exp.to_struct()
        rec = Serializable.from_struct(s)
        assert rec == exp

        # Binfile header round-trip
        hdr = exp.convert_to_binfile_header()
        exp_from_hdr = IX_experiment.build_from_binfile_header(hdr)
        assert exp_from_hdr == exp
        assert exp_from_hdr.run_id == 10.0

    def test_efix_emode_combo_validation(self) -> None:
        with pytest.raises(ValueError, match="invalid_argument"):
            IX_experiment(efix=0.0, emode=1)

    def test_hashable_prop(self) -> None:
        exp = IX_experiment(filename="a.spe", filepath="/tmp", efix=10.0, emode=1, psi=5.0)
        hashable_obj_tester(exp)


class TestIXMaskAndMap:
    """Unit tests for IX_mask and IX_map."""

    def test_ix_mask(self, tmp_path) -> None:
        m0 = IX_mask()
        assert m0.msk.size == 0

        m1 = IX_mask([5, 2, 5, 1, 3, 4])
        np.testing.assert_array_equal(m1.msk, [1, 2, 3, 4, 5])

        m2 = IX_mask("10:12, 15, 20:2:24")
        np.testing.assert_array_equal(m2.msk, [10, 11, 12, 15, 20, 22, 24])

        mc = m1.combine(m2)
        np.testing.assert_array_equal(mc.msk, [1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 22, 24])

        mask_file = str(tmp_path / "test.msk")
        mc.save_ascii(mask_file)
        mc_loaded = IX_mask.read_ascii(mask_file)
        assert mc_loaded == mc

    def test_ix_map(self, tmp_path) -> None:
        mp = IX_map([[1, 2, 3], [5, 4]], [10, 20])
        np.testing.assert_array_equal(mp.wkno, [10, 20])
        np.testing.assert_array_equal(mp.ns, [3, 2])
        np.testing.assert_array_equal(mp.s, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(mp.w, [10, 10, 10, 20, 20])
        assert mp.total_spec == 5
        assert mp.nwork == 2

        # Masking spectra
        mp_masked = mp.mask([2, 4, 5])
        np.testing.assert_array_equal(mp_masked.wkno, [10, 20])
        np.testing.assert_array_equal(mp_masked.ns, [2, 0])
        np.testing.assert_array_equal(mp_masked.s, [1, 3])

        # Sectioning
        mp_sec = mp.section([20])
        np.testing.assert_array_equal(mp_sec.wkno, [20])
        np.testing.assert_array_equal(mp_sec.ns, [2])
        np.testing.assert_array_equal(mp_sec.s, [4, 5])

        # ASCII I/O round-trip
        map_file = str(tmp_path / "test.map")
        mp.save_ascii(map_file)
        mp_loaded = IX_map.read_ascii(map_file)
        assert mp_loaded == mp

