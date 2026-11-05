"""Pytest port of ``_test/test_sqw_pageOpMethods/test_apply.m``."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from pyhorace.coord_transf.symop import SymopReflection
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw.sqw import sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_COMMON_DIR = REPO_ROOT / "_test" / "common_data"


def _get_random_data_in_range(
    num_fields: int, num_pix: int, data_range: tuple[float, float], seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = data_range
    return lo + (hi - lo) * rng.random((num_fields, num_pix))


def _wrap_unary_func(unary_op: Callable[[Any], Any], pix: PixelDataBase) -> PixelDataBase:
    pg_result = unary_op(sigvar(pix.signal, pix.variance))
    pix.signal = pg_result.s
    pix.variance = pg_result.e
    pix = pix.recalc_data_range(["signal", "variance"])
    return pix


class TestApply:
    """Port of ``test_apply.m``."""

    def test_apply_against_unary_pix_no_dnd(self) -> None:
        num_pix = 30
        data_range = (0.0, 1.0)
        data = _get_random_data_in_range(
            PixelDataBase.DEFAULT_NUM_PIX_FIELDS, num_pix, data_range
        )
        pix_mb = PixelDataMemory(data)

        pix_un = pix_mb.do_unary_op(np.sin)
        pix_ap = pix_mb.apply(lambda p: _wrap_unary_func(lambda sv: sv.sin(), p))

        ok, mess = pix_un.equal_to_tol(pix_ap, nargout=2)
        assert ok, mess

        pix_fb = PixelDataFileBacked(data)
        pix_ap_fb = pix_fb.apply(lambda p: _wrap_unary_func(lambda sv: sv.sin(), p))

        ok_fb, mess_fb = pix_un.equal_to_tol(pix_ap_fb, tol=[1e-6, 1e-6], nargout=2)
        assert ok_fb, mess_fb

    def test_apply_against_unary_pix(self) -> None:
        num_pix = 30
        data_range = (0.0, 1.0)
        data = _get_random_data_in_range(
            PixelDataBase.DEFAULT_NUM_PIX_FIELDS, num_pix, data_range
        )
        sqw_mb = sqw()
        sqw_mb.pix = PixelDataMemory(data)
        sqw_mb.data.npix = np.array([[num_pix]], dtype=np.float64)
        sqw_mb.data.s = np.array([[-99.0]], dtype=np.float64)
        sqw_mb.data.e = np.array([[99.0]], dtype=np.float64)

        sqw_un = sqw_mb.sin()
        sqw_ap = sqw_mb.apply(lambda p: _wrap_unary_func(lambda sv: sv.sin(), p))

        ok, mess = sqw_un.equal_to_tol(sqw_ap, nargout=2)
        assert ok, mess

        sqw_fb = sqw()
        sqw_fb.pix = PixelDataFileBacked(data)
        sqw_fb.data.npix = np.array([[num_pix]], dtype=np.float64)
        sqw_fb.data.s = np.array([[-99.0]], dtype=np.float64)
        sqw_fb.data.e = np.array([[99.0]], dtype=np.float64)

        sqw_ap_fb = sqw_fb.apply(lambda p: _wrap_unary_func(lambda sv: sv.sin(), p))

        ok_fb, mess_fb = sqw_un.equal_to_tol(sqw_ap_fb, tol=[1e-6, 1e-6], nargout=2)
        assert ok_fb, mess_fb

    def test_apply_transform(self) -> None:
        pth = str(TEST_COMMON_DIR / "sqw_2d_1.sqw")
        sqw_mb = sqw(pth)

        assert sqw_mb.experiment_info.n_runs == 24
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        old_scale = hc.fb_scale_factor
        try:
            hc.mem_chunk_size = 10000
            hc.fb_scale_factor = 1
            sqw_fb = sqw(pth, file_backed=True)
            assert sqw_fb.experiment_info.n_runs == 85

            sym = SymopReflection([1, 0, 0], [0, 1, 0])
            sym.b_matrix = sqw_mb.data.proj.bmatrix(3)
            sqw_ap_fb = sqw_fb.apply(sym.transform_pix, [], False)
            assert sqw_ap_fb.experiment_info.n_runs == 24

            sqw_ap_mb = sqw_mb.apply(sym.transform_pix, [], False)

            ok, mess = sqw_ap_fb.equal_to_tol(sqw_ap_mb, nargout=2)
            assert ok, mess
        finally:
            hc.mem_chunk_size = old_chunk
            hc.fb_scale_factor = old_scale

    def test_apply_multiple_transform(self) -> None:
        sym = [
            SymopReflection([1, 0, 0], [0, 1, 0]),
            SymopReflection([1, 0, 0], [0, 0, 1]),
        ]
        sqw_mb = sqw.generate_cube_sqw(10)
        for s in sym:
            s.b_matrix = np.eye(3)

        func = [s.transform_pix for s in sym]

        sqw_ap_mb = sqw_mb.apply(func, [], False)

        sqw_fb = sqw_mb.copy()
        sqw_fb.pix = PixelDataFileBacked(sqw_fb.pix)
        sqw_ap_fb = sqw_fb.apply(func, [], False)

        sqw_op = sqw_mb.copy()
        sqw_op.pix = sym[0].transform_pix(sqw_op.pix)
        sqw_op.pix = sym[1].transform_pix(sqw_op.pix)

        ok1, mess1 = sqw_op.equal_to_tol(sqw_ap_mb, nargout=2)
        assert ok1, mess1
        ok2, mess2 = sqw_ap_fb.equal_to_tol(sqw_ap_mb, nargout=2)
        assert ok2, mess2
