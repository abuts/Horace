"""Pytest port of ``_test/test_sqw_pageOpMethods/test_eval.m``."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pytest

from pyhorace.coord_transf.sphere_proj import sphere_proj
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.sqw.sqw import sqw
from pyhorace.sqw.sqw_dnd_base import bose

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_COMMON_DIR = REPO_ROOT / "_test" / "common_data"


def _disp2sqw_eval_tester2D(
    qh: np.ndarray, qk: np.ndarray, ql: np.ndarray, p: Any = None
) -> Tuple[np.ndarray, np.ndarray]:
    w = np.ones_like(qh, dtype=np.float64)
    s = np.ones_like(qh, dtype=np.float64)
    return w, s


def _sqw_eval_tester(
    h: np.ndarray, k: np.ndarray, l: np.ndarray, en: np.ndarray, par: Any = None
) -> np.ndarray:
    sz = h.shape
    if sz != k.shape or sz != l.shape or sz != en.shape:
        raise RuntimeError("SQW_EVAL_TESTER:runtime_error: unequal size input arrays")
    if len(sz) != 2 or sz[1] != 1:
        raise RuntimeError(
            f"SQW_EVAL_TESTER:runtime_error: incorrect shape of input arrays {sz}"
        )
    return 2.0 * np.ones(sz, dtype=np.float64)


def _funceval_tester2D(
    x: np.ndarray, en: np.ndarray, par: Any = None
) -> np.ndarray:
    sz = x.shape
    if sz != en.shape:
        raise RuntimeError("FUNC_EVAL_TESTER:runtime_error: unequal size input arrays")
    if len(sz) != 2 or sz[1] != 1:
        raise RuntimeError(
            f"FUNC_EVAL_TESTER:runtime_error: incorrect shape of input arrays {sz}"
        )
    return 3.0 * np.ones(sz, dtype=np.float64)


class TestEval:
    """Port of ``test_eval.m``."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        test_sqw_file = str(TEST_COMMON_DIR / "sqw_2d_2.sqw")
        self.sqw_obj = sqw(test_sqw_file)
        self.sqw_obj_fb = self.sqw_obj.copy()
        self.sqw_obj_fb.pix = PixelDataFileBacked(self.sqw_obj_fb.pix)

    def test_disp2sqw_eval(self) -> None:
        ds = self.sqw_obj.disp2sqw_eval(_disp2sqw_eval_tester2D, [], 1.0, "-all")
        assert isinstance(ds, sqw)
        assert ds.data.s.shape == self.sqw_obj.data.s.shape

    def test_sqw_eval_with_sphere_proj(self) -> None:
        spp = sphere_proj()
        spp.offset = [0, 1.5, 0]
        sp_cut = self.sqw_obj.cut(spp, [0, 0.01, 2], [0, 180], [-180, 180], 0)

        sp_bose = bose(sp_cut, 8)
        diff = sp_cut - sp_bose
        np.testing.assert_allclose(
            diff.data.s, np.zeros_like(diff.data.s), atol=1e-7
        )

    def test_func_eval_sqw(self) -> None:
        ds = self.sqw_obj.func_eval(_funceval_tester2D, [], "-all")

        sig = ds.data.s
        pix_s = ds.pix.signal

        np.testing.assert_array_equal(sig, 3.0 * np.ones_like(sig))
        np.testing.assert_array_equal(pix_s, 3.0 * np.ones_like(pix_s))

    def test_sqw_eval_average(self) -> None:
        ds = self.sqw_obj.sqw_eval(_sqw_eval_tester, [], "-average")

        sig = ds.data.s
        pix_s = ds.pix.signal

        np.testing.assert_array_equal(sig, 2.0 * np.ones_like(sig))
        np.testing.assert_array_equal(pix_s, 2.0 * np.ones_like(pix_s))

    def test_sqw_eval(self) -> None:
        ds = self.sqw_obj.sqw_eval(_sqw_eval_tester, [])

        sig = ds.pix.signal
        np.testing.assert_array_equal(sig, 2.0 * np.ones_like(sig))
        np.testing.assert_array_equal(ds.data.s, 2.0 * np.ones_like(ds.data.s))
        np.testing.assert_array_equal(ds.data.e, np.zeros_like(ds.data.s))

    def test_sqw_eval_average_fb(self) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 8000
            ds_mb = self.sqw_obj.sqw_eval(_sqw_eval_tester, [], "-average")
            ds_fb = self.sqw_obj_fb.sqw_eval(_sqw_eval_tester, [], "-average")
            assert isinstance(ds_fb.pix, PixelDataFileBacked)

            ok, mess = ds_mb.equal_to_tol(
                ds_fb, tol=1e-6, ignore_str=True, nargout=2
            )
            assert ok, mess
        finally:
            hc.mem_chunk_size = old_chunk

    def test_sqw_eval_fb(self) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 8000
            ds_mb = self.sqw_obj.sqw_eval(_sqw_eval_tester, [])
            ds_fb = self.sqw_obj_fb.sqw_eval(_sqw_eval_tester, [])
            assert ds_fb.is_filebacked

            ok, mess = ds_mb.equal_to_tol(
                ds_fb, tol=1e-6, ignore_str=True, nargout=2
            )
            assert ok, mess

            fb_res_file = ds_fb.full_filename
            assert os.path.isfile(fb_res_file)
            del ds_fb
            gc.collect()
            assert not os.path.isfile(fb_res_file)
        finally:
            hc.mem_chunk_size = old_chunk

    def test_sqw_eval_no_pix(self) -> None:
        sqw_nopix = self.sqw_obj.copy()
        sqw_nopix.pix = PixelDataBase.create()
        ds = sqw_nopix.sqw_eval(_sqw_eval_tester, [])

        assert ds.data.s.shape == sqw_nopix.data.s.shape
        expected = 2.0 * np.ones_like(sqw_nopix.data.s)
        np.testing.assert_array_equal(ds.data.s, expected)
