"""Common pixel operations test suite and in-memory pixel operations test runner.

Translated from `_test/test_sqw_pixels/PixOpsTestCommon.m` and
`_test/test_sqw_pixels/test_pix_ops_in_memory.m`.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import (
    DataOpArray,
    minus,
    mldivide,
    mpower,
    mrdivide,
    mtimes,
    plus,
)
from pyhorace.data.ix_axis import IX_axis
from pyhorace.data.ix_dataset_1d import IX_dataset_1d
from pyhorace.data.pixels import (
    PixelDataBase,
    PixelDataFileBacked,
    PixelDataMemory,
)
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw import DnDBase, d1d, read_dnd, read_sqw, sqw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
COMMON_DATA_DIR = os.path.join(REPO_ROOT, "_test", "common_data")
SQW_1D_1 = os.path.join(COMMON_DATA_DIR, "sqw_1d_1.sqw")
SQW_1D_2 = os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")
SQW_2D_1 = os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw")
W2D_QE_SQW = os.path.join(COMMON_DATA_DIR, "w2d_qe_sqw.sqw")


def _make_sample_pix(n_pix: int = 30, file_backed: bool = False, page_size: int = 10000) -> PixelDataBase:
    rng = np.random.default_rng(12345)
    raw = np.abs(rng.random((9, n_pix), dtype=np.float64)) + 0.5
    raw[4, :] = 1.0
    raw[5, :] = np.arange(1, n_pix + 1, dtype=np.float64)
    raw[6, :] = 1.0
    if file_backed:
        pix: PixelDataBase = PixelDataFileBacked(raw)
        pix.page_size = page_size
    else:
        pix = PixelDataMemory(raw)
    return pix


class PixOpsCommonSuite:
    """Shared test suite for both in-memory (`mem_chunk_size=10000`) and
    multi-page file-backed (`mem_chunk_size=11`) PixelDataBase operations.
    """

    file_backed: bool = False
    page_size: int = 10000

    def _create_pix(self, n_pix: int = 30) -> PixelDataBase:
        return _make_sample_pix(n_pix=n_pix, file_backed=self.file_backed, page_size=self.page_size)

    def _get_full_data(self, pix: PixelDataBase) -> np.ndarray:
        return PixelDataMemory(pix).data

    # ------------------------------------------------------------------
    # Unary operations
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "op_name,sig_fn,var_fn",
        [
            ("cos", lambda s: np.cos(s), lambda s, v: v * (np.sin(s) ** 2)),
            ("sin", lambda s: np.sin(s), lambda s, v: v * (np.cos(s) ** 2)),
            ("exp", lambda s: np.exp(s), lambda s, v: v * (np.exp(s) ** 2)),
            ("log", lambda s: np.log(s), lambda s, v: v / (s ** 2)),
            ("sqrt", lambda s: np.sqrt(s), lambda s, v: v / (4.0 * s)),
            ("uminus", lambda s: -s, lambda s, v: v),
            ("uplus", lambda s: s, lambda s, v: v),
        ],
    )
    def test_unary_ops(self, op_name, sig_fn, var_fn):
        pix = self._create_pix(30)
        orig_full = self._get_full_data(pix)
        orig_s = orig_full[7, :].copy()
        orig_v = orig_full[8, :].copy()

        res = getattr(pix, op_name)()
        res_full = self._get_full_data(res)

        expected_s = sig_fn(orig_s)
        expected_v = var_fn(orig_s, orig_v)
        np.testing.assert_allclose(res_full[7, :], expected_s, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(res_full[8, :], expected_v, rtol=1e-5, atol=1e-6)
        # Non-signal/variance rows are unchanged
        np.testing.assert_allclose(res_full[:7, :], orig_full[:7, :], rtol=1e-6)
        # Range for signal & variance is updated
        np.testing.assert_allclose(
            res.data_range[:, 7],
            [np.min(expected_s), np.max(expected_s)],
            rtol=1e-5,
            atol=1e-6,
        )

    # ------------------------------------------------------------------
    # Binary operations with scalar
    # ------------------------------------------------------------------
    def test_binary_scalar_ops(self):
        pix = self._create_pix(30)
        orig_full = self._get_full_data(pix)
        s0 = orig_full[7, :]
        v0 = orig_full[8, :]
        scalar = 3.0

        #pix + scalar
        r_plus = plus(pix, scalar)
        rf_plus = self._get_full_data(r_plus)
        np.testing.assert_allclose(rf_plus[7, :], s0 + scalar, rtol=1e-5)
        np.testing.assert_allclose(rf_plus[8, :], v0, rtol=1e-5)

        # scalar - pix
        r_rminus = minus(scalar, pix)
        rf_rminus = self._get_full_data(r_rminus)
        np.testing.assert_allclose(rf_rminus[7, :], scalar - s0, rtol=1e-5)
        np.testing.assert_allclose(rf_rminus[8, :], v0, rtol=1e-5)

        # pix * scalar
        r_mul = mtimes(pix, scalar)
        rf_mul = self._get_full_data(r_mul)
        np.testing.assert_allclose(rf_mul[7, :], s0 * scalar, rtol=1e-5)
        np.testing.assert_allclose(rf_mul[8, :], v0 * (scalar ** 2), rtol=1e-5)

        # pix / scalar
        r_div = mrdivide(pix, scalar)
        rf_div = self._get_full_data(r_div)
        np.testing.assert_allclose(rf_div[7, :], s0 / scalar, rtol=1e-5)
        np.testing.assert_allclose(rf_div[8, :], v0 / (scalar ** 2), rtol=1e-5)

        # pix ^ scalar
        r_pow = mpower(pix, 2.0)
        rf_pow = self._get_full_data(r_pow)
        np.testing.assert_allclose(rf_pow[7, :], s0 ** 2.0, rtol=1e-5)
        np.testing.assert_allclose(rf_pow[8, :], 4.0 * (s0 ** 2.0) * v0, rtol=1e-5)

    # ------------------------------------------------------------------
    # Binary operations between two PixelDataBase objects
    # ------------------------------------------------------------------
    def test_binary_pix_pix_ops(self):
        pix1 = self._create_pix(30)
        pix2 = self._create_pix(30)
        f1 = self._get_full_data(pix1)
        f2 = self._get_full_data(pix2)

        res_add = pix1 + pix2
        rf_add = self._get_full_data(res_add)
        np.testing.assert_allclose(rf_add[7, :], f1[7, :] + f2[7, :], rtol=1e-5)
        np.testing.assert_allclose(rf_add[8, :], f1[8, :] + f2[8, :], rtol=1e-5)

        res_sub = pix1 - pix2
        rf_sub = self._get_full_data(res_sub)
        np.testing.assert_allclose(rf_sub[7, :], f1[7, :] - f2[7, :], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(rf_sub[8, :], f1[8, :] + f2[8, :], rtol=1e-5)

        # Incompatible pixel counts raise HoraceError
        pix_short = self._create_pix(10)
        with pytest.raises(HoraceError):
            _ = pix1 + pix_short

    # ------------------------------------------------------------------
    # Binary operations with sigvar, 1D/2D numeric array + npix, DnD, IX_dataset_1d
    # ------------------------------------------------------------------
    def test_binary_with_sigvar_and_npix(self):
        pix = self._create_pix(30)
        orig = self._get_full_data(pix)
        npix = np.array([5, 10, 8, 7], dtype=int)  # sums to 30
        sv_s = np.array([1.0, 2.0, 3.0, 4.0])
        sv_e = np.array([0.1, 0.2, 0.3, 0.4])
        sv = sigvar(sv_s, sv_e)

        res = pix.do_binary_op(sv, plus, npix=npix)
        rf = self._get_full_data(res)
        rep_s = np.repeat(sv_s, npix)
        rep_e = np.repeat(sv_e, npix)
        np.testing.assert_allclose(rf[7, :], orig[7, :] + rep_s, rtol=1e-5)
        np.testing.assert_allclose(rf[8, :], orig[8, :] + rep_e, rtol=1e-5)

        # Flip = True (sv - pix)
        res_flip = pix.do_binary_op(sv, minus, flip=True, npix=npix)
        rf_flip = self._get_full_data(res_flip)
        np.testing.assert_allclose(rf_flip[7, :], rep_s - orig[7, :], rtol=1e-5)
        np.testing.assert_allclose(rf_flip[8, :], rep_e + orig[8, :], rtol=1e-5)

    def test_binary_with_ix_dataset_1d(self):
        pix = self._create_pix(20)
        orig = self._get_full_data(pix)
        npix = np.array([4, 6, 5, 5], dtype=int)
        ds = IX_dataset_1d(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 20.0, 30.0, 40.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )
        res = pix.do_binary_op(ds, plus, npix=npix)
        rf = self._get_full_data(res)
        rep_s = np.repeat(ds.signal.ravel(order="F"), npix)
        rep_v = np.repeat((ds.error.ravel(order="F")) ** 2, npix)
        np.testing.assert_allclose(rf[7, :], orig[7, :] + rep_s, rtol=1e-5)
        np.testing.assert_allclose(rf[8, :], orig[8, :] + rep_v, rtol=1e-5)

    # ------------------------------------------------------------------
    # compute_bin_data, noisify, and recalc_data_range
    # ------------------------------------------------------------------
    def test_compute_bin_data(self):
        pix = self._create_pix(30)
        full = self._get_full_data(pix)
        npix = np.array([[5, 10], [8, 7]], dtype=int, order="F")

        mean_s, mean_v = pix.compute_bin_data(npix)
        assert mean_s.shape == (2, 2)
        assert mean_v.shape == (2, 2)

        # Manually verify bin averages in column-major order
        npix_flat = npix.ravel(order="F")
        ends = np.cumsum(npix_flat)
        starts = ends - npix_flat
        exp_s = np.zeros(4)
        exp_v = np.zeros(4)
        for i, (st, en, c) in enumerate(zip(starts, ends, npix_flat)):
            exp_s[i] = np.mean(full[7, st:en])
            exp_v[i] = np.sum(full[8, st:en]) / (c ** 2)

        np.testing.assert_allclose(mean_s.ravel(order="F"), exp_s, rtol=1e-5)
        np.testing.assert_allclose(mean_v.ravel(order="F"), exp_v, rtol=1e-5)

    def test_noisify(self):
        pix = self._create_pix(30)
        orig = self._get_full_data(pix)
        noisy = pix.noisify(0.05)
        noisy_full = self._get_full_data(noisy)
        assert noisy.num_pixels == 30
        assert not np.allclose(noisy_full[7, :], orig[7, :])
        # Coordinates are untouched
        np.testing.assert_allclose(noisy_full[:7, :], orig[:7, :], rtol=1e-6)


class TestPixOpsInMemory(PixOpsCommonSuite):
    """Run PixOpsCommonSuite with in-memory PixelDataMemory (`mem_chunk_size = 10000`)."""

    file_backed = False
    page_size = 10000

