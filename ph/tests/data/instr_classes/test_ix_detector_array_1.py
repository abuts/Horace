"""Unit tests for IX_detector_array translated from test_IX_detector_array_1.m."""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.coord_transf import rotvec_to_rotmat
from pyhorace.core.serializable import Serializable
from pyhorace.data.instr_classes.ix_det_he3tube import IX_det_He3tube
from pyhorace.data.instr_classes.ix_det_slab import IX_det_slab
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.instr_classes.ix_detector_bank import IX_detector_bank


def _make_he3_banks():
    """Construct the three He3tube detector banks used in test_IX_detector_array_1.m."""
    # Bank 1 (2 detectors)
    dia1 = [0.025, 0.040]
    height1 = [0.015, 0.02]
    wall1 = [0.001, 0.0015]
    atms1 = [10, 6]
    det1 = IX_det_He3tube(dia1, height1, wall1, atms1)

    id1 = [19, 13]
    x2_1 = [2, 2.2]
    phi1 = [30, 40]
    azim1 = [45, 60]
    ang11 = [20, 30, 50]
    ang12 = [-15, 25, 35]
    rotvec1 = np.deg2rad(np.column_stack([ang11, ang12]))
    dmat1 = np.swapaxes(rotvec_to_rotmat(rotvec1), 0, 1)
    dbank1 = IX_detector_bank(id1, x2_1, phi1, azim1, det1, "rotvec", rotvec1)

    # Bank 2 (2 detectors)
    dia2 = [0.027, 0.047]
    height2 = [0.017, 0.027]
    wall2 = [0.0017, 0.00157]
    atms2 = [10.7, 6.7]
    det2 = IX_det_He3tube(dia2, height2, wall2, atms2)

    id2 = [28, 22]
    x2_2 = [2.7, 2.27]
    phi2 = [37, 47]
    azim2 = [47, 67]
    ang21 = [27, 37, 57]
    ang22 = [-17, 27, 37]
    rotvec2 = np.deg2rad(np.column_stack([ang21, ang22]))
    dmat2 = np.swapaxes(rotvec_to_rotmat(rotvec2), 0, 1)
    dbank2 = IX_detector_bank(id2, x2_2, phi2, azim2, det2, "rotvec", rotvec2)

    # Bank 3 (3 detectors)
    dia3 = [0.028, 0.048, 0.058]
    height3 = [0.018, 0.028, 0.038]
    wall3 = [0.0018, 0.00158, 0.0028]
    atms3 = [10.8, 6.8, 7.8]
    det3 = IX_det_He3tube(dia3, height3, wall3, atms3)

    id3 = [35, 39, 31]
    x2_3 = [2.8, 2.28, 2.58]
    phi3 = [38, 48, 58]
    azim3 = [48, 68, 88]
    ang31 = [28, 38, 58]
    ang32 = [-18, 28, 38]
    ang33 = [-18.5, 28.5, 38.5]
    rotvec3 = np.deg2rad(np.column_stack([ang31, ang32, ang33]))
    dmat3 = np.swapaxes(rotvec_to_rotmat(rotvec3), 0, 1)
    dbank3 = IX_detector_bank(id3, x2_3, phi3, azim3, det3, "rotvec", rotvec3)

    # Combined single bank (detectors sorted within each bank as in dbank1, dbank2, dbank3)
    det_all = IX_det_He3tube(
        np.vstack([det1.reorder([2, 1]).dia, det2.reorder([2, 1]).dia, det3.reorder([3, 1, 2]).dia]),
        np.vstack([det1.reorder([2, 1]).height, det2.reorder([2, 1]).height, det3.reorder([3, 1, 2]).height]),
        np.vstack([det1.reorder([2, 1]).wall, det2.reorder([2, 1]).wall, det3.reorder([3, 1, 2]).wall]),
        np.vstack([det1.reorder([2, 1]).atms, det2.reorder([2, 1]).atms, det3.reorder([3, 1, 2]).atms]),
    )
    id_all = np.vstack([dbank1.id, dbank2.id, dbank3.id])
    x2_all = np.vstack([dbank1.x2, dbank2.x2, dbank3.x2])
    phi_all = np.vstack([dbank1.phi, dbank2.phi, dbank3.phi])
    azim_all = np.vstack([dbank1.azim, dbank2.azim, dbank3.azim])
    rotvec_all = np.hstack([dbank1.rotvec, dbank2.rotvec, dbank3.rotvec])
    dmat_all = np.concatenate([dbank1.dmat, dbank2.dmat, dbank3.dmat], axis=2)
    dbank_all = IX_detector_bank(
        id_all, x2_all, phi_all, azim_all, det_all, "rotvec", rotvec_all
    )

    return (
        dbank1,
        dbank2,
        dbank3,
        dbank_all,
        (id1, x2_1, phi1, azim1, det1, rotvec1, dmat1),
        (id_all, x2_all, phi_all, azim_all, rotvec_all, dmat_all),
    )


class TestIXDetectorArray1:
    """Tests for IX_detector_array translated from test_IX_detector_array_1.m."""

    def test_he3tube_array(self):
        (
            dbank1,
            dbank2,
            dbank3,
            dbank_all,
            (id1, x2_1, phi1, azim1, det1, rotvec1, dmat1),
            (id_all, x2_all, phi_all, azim_all, rotvec_all, dmat_all),
        ) = _make_he3_banks()

        # Single-bank detector array constructed from bank or directly from bank arguments
        d1 = IX_detector_array(dbank1)
        d1_direct = IX_detector_array(
            id1, x2_1, phi1, azim1, det1, "rotvec", rotvec1
        )
        assert d1 == d1_direct
        assert d1.nbank == 1
        assert d1.ndet == 2
        assert d1.ndet_bank == 2
        assert d1.det_bank == dbank1
        np.testing.assert_array_equal(d1.id, np.array([[13], [19]]))
        np.testing.assert_allclose(d1.x2, np.array([[2.2], [2.0]]), atol=1e-13)
        np.testing.assert_allclose(d1.phi, np.array([[40.0], [30.0]]), atol=1e-13)
        np.testing.assert_allclose(d1.azim, np.array([[60.0], [45.0]]), atol=1e-13)
        np.testing.assert_allclose(d1.rotvec, rotvec1[:, [1, 0]], atol=1e-13)
        np.testing.assert_allclose(d1.dmat, dmat1[:, :, [1, 0]], atol=1e-13)

        # Multi-bank detector array
        d = IX_detector_array([dbank1, dbank2, dbank3])
        assert d.nbank == 3
        assert d.ndet == 7
        np.testing.assert_array_equal(d.ndet_bank, np.array([2, 2, 3]))
        assert d.det_bank == [dbank1, dbank2, dbank3]
        np.testing.assert_array_equal(d.id, id_all)
        np.testing.assert_allclose(d.x2, x2_all, atol=1e-13)
        np.testing.assert_allclose(d.phi, phi_all, atol=1e-13)
        np.testing.assert_allclose(d.azim, azim_all, atol=1e-13)
        np.testing.assert_allclose(d.rotvec, rotvec_all, atol=1e-13)
        np.testing.assert_allclose(d.dmat, dmat_all, atol=1e-13)

        # Function evaluation on a subset of detectors in a single bank vs multi-bank
        ind_sub = np.array([[3, 4], [4, 3]])  # all in bank 2 (indices 3 and 4)
        wvec_sub = np.array([[5.1, 5.2], [5.3, 5.4]])
        for meth in (
            "effic",
            "mean_d",
            "mean_x",
            "mean_y",
            "mean_z",
            "mean",
            "var_d",
            "var_x",
            "var_y",
            "var_z",
            "covariance",
        ):
            val_arr = getattr(d, meth)(ind_sub, wvec_sub)
            val_ref = getattr(dbank_all, meth)(ind_sub, wvec_sub)
            np.testing.assert_allclose(val_arr, val_ref, atol=1e-13)

        # Function evaluation across multiple banks with 2D index and wvec arrays
        ind_multi = np.array([[6, 3, 2], [1, 7, 4]])
        wvec_multi = np.array([[5.1, 5.2, 5.3], [5.4, 5.5, 5.6]])
        for meth in (
            "effic",
            "mean_d",
            "mean_x",
            "mean_y",
            "mean_z",
            "mean",
            "var_d",
            "var_x",
            "var_y",
            "var_z",
            "covariance",
        ):
            val_arr = getattr(d, meth)(ind_multi, wvec_multi)
            val_ref = getattr(dbank_all, meth)(ind_multi, wvec_multi)
            np.testing.assert_allclose(val_arr, val_ref, atol=1e-13)
            # Also test scalar wavevector
            val_arr_sc = getattr(d, meth)(ind_multi, 5.25)
            val_ref_sc = getattr(dbank_all, meth)(ind_multi, 5.25)
            np.testing.assert_allclose(val_arr_sc, val_ref_sc, atol=1e-13)

    def test_he3tube_array_throws(self):
        dbank1, dbank2, dbank3, _, _, _ = _make_he3_banks()

        # Duplicate bank IDs must raise ValueError
        with pytest.raises(
            ValueError, match="HERBERT:IX_detector_array:invalid_argument"
        ):
            IX_detector_array([dbank1, dbank2, dbank2])

        d = IX_detector_array([dbank1, dbank2, dbank3])

        # Duplicate detector ID when setting d.id
        with pytest.raises(
            ValueError, match="HERBERT:IX_detector_array:invalid_argument"
        ):
            d.id = [11, 21, 13, 14, 15, 16, 21]

        # Wrong number of detector IDs when setting d.id
        with pytest.raises(
            ValueError, match="HERBERT:IX_detector_array:invalid_argument"
        ):
            d.id = [11, 21, 13, 14, 15, 16]

        # Valid reassignment of detector IDs across banks (sorted within each bank)
        d.id = [21, 11, 14, 13, 16, 15, 12]
        np.testing.assert_array_equal(
            d.id, np.array([[11], [21], [13], [14], [12], [15], [16]])
        )

    def test_mixed_array(self):
        dbank1, _, dbank3, _, _, _ = _make_he3_banks()

        # Bank 2 is an IX_det_slab bank (2 detectors)
        depth2 = [0.012, 0.018]
        width2 = [0.027, 0.047]
        height2 = [0.017, 0.027]
        atten2 = [4.5, 5.5]
        det2_slab = IX_det_slab(depth2, width2, height2, atten2)

        id2 = [28, 22]
        x2_2 = [2.7, 2.27]
        phi2 = [37, 47]
        azim2 = [47, 67]
        ang21 = [27, 37, 57]
        ang22 = [-17, 27, 37]
        rotvec2 = np.deg2rad(np.column_stack([ang21, ang22]))
        dbank2_slab = IX_detector_bank(
            id2, x2_2, phi2, azim2, det2_slab, "rotvec", rotvec2
        )

        d = IX_detector_array([dbank1, dbank2_slab, dbank3])
        assert d.nbank == 3
        assert d.ndet == 7

        ind = np.array([[6, 2, 7], [3, 1, 4]])
        wvec = np.array([[5.1, 5.2, 5.3], [5.4, 5.5, 5.6]])

        # Compare mixed-bank evaluation against element-by-element individual bank calls
        # Bank 1 covers global indices 1..2; Bank 2 covers 3..4; Bank 3 covers 5..7
        for meth in (
            "effic",
            "mean_d",
            "mean_x",
            "mean_y",
            "mean_z",
            "var_d",
            "var_x",
            "var_y",
            "var_z",
        ):
            val = getattr(d, meth)(ind, wvec)
            assert val.shape == (2, 3)
            # (0,0): ind=6 -> bank3 local ind=2, wvec=5.1
            np.testing.assert_allclose(
                val[0, 0], getattr(dbank3, meth)(2, 5.1), atol=1e-13
            )
            # (1,0): ind=3 -> bank2_slab local ind=1, wvec=5.4
            np.testing.assert_allclose(
                val[1, 0], getattr(dbank2_slab, meth)(1, 5.4), atol=1e-13
            )
            # (0,1): ind=2 -> bank1 local ind=2, wvec=5.2
            np.testing.assert_allclose(
                val[0, 1], getattr(dbank1, meth)(2, 5.2), atol=1e-13
            )
            # (1,1): ind=1 -> bank1 local ind=1, wvec=5.5
            np.testing.assert_allclose(
                val[1, 1], getattr(dbank1, meth)(1, 5.5), atol=1e-13
            )
            # (0,2): ind=7 -> bank3 local ind=3, wvec=5.3
            np.testing.assert_allclose(
                val[0, 2], getattr(dbank3, meth)(3, 5.3), atol=1e-13
            )
            # (1,2): ind=4 -> bank2_slab local ind=2, wvec=5.6
            np.testing.assert_allclose(
                val[1, 2], getattr(dbank2_slab, meth)(2, 5.6), atol=1e-13
            )

        # Also check vector-valued mean (3 x 2 x 3) and matrix-valued covariance (3 x 3 x 2 x 3)
        mean_val = d.mean(ind, wvec)
        assert mean_val.shape == (3, 2, 3)
        np.testing.assert_allclose(
            mean_val[:, 1, 0], dbank2_slab.mean(1, 5.4).ravel(), atol=1e-13
        )
        np.testing.assert_allclose(
            mean_val[:, 0, 2], dbank3.mean(3, 5.3).ravel(), atol=1e-13
        )

        cov_val = d.covariance(ind, wvec)
        assert cov_val.shape == (3, 3, 2, 3)
        np.testing.assert_allclose(
            cov_val[:, :, 1, 0], dbank2_slab.covariance(1, 5.4), atol=1e-13
        )
        np.testing.assert_allclose(
            cov_val[:, :, 0, 2], dbank3.covariance(3, 5.3), atol=1e-13
        )

    def test_serializable_and_detpar(self):
        dbank1, dbank2, dbank3, _, _, _ = _make_he3_banks()
        d = IX_detector_array([dbank1, dbank2, dbank3])
        d.filename = "my_det.par"
        d.filepath = "/path/to/det"

        s = d.to_struct()
        d_rec = Serializable.from_struct(s)
        assert d == d_rec
        assert d_rec.filename == "my_det.par"
        assert d_rec.filepath == "/path/to/det"

        # Detpar dictionary round-trip
        dp = d.get_detpar_representation()
        d_from_dp = IX_detector_array(dp)
        assert d_from_dp.nbank == 1
        assert d_from_dp.ndet == 7
        np.testing.assert_array_equal(d_from_dp.id, np.sort(d.id, axis=0))

        detdcn = d.calc_detdcn()
        assert detdcn.shape == (4, 7)
        np.testing.assert_array_equal(detdcn[3, :], d.id.ravel())
        detdcn_sub = d.calc_detdcn([2, 5])
        np.testing.assert_allclose(detdcn_sub, detdcn[:, [1, 4]], atol=1e-13)
