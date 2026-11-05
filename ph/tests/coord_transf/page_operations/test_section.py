"""Pytest port of ``_test/test_sqw_pageOpMethods/test_section.m``."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.coord_transf.page_operations.page_op_section import PageOp_section
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.sqw import read_sqw, sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_COMMON_DIR = REPO_ROOT / "_test" / "common_data"


def _is_bins_subset_of(a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> bool:
    for pa, pb in zip(a, b):
        pa_arr = np.asarray(pa, dtype=float).ravel()
        pb_arr = np.asarray(pb, dtype=float).ravel()
        for val in pa_arr:
            if not np.any(np.isclose(val, pb_arr, atol=1e-12, rtol=1e-12)):
                return False
    return True


class TestSection:
    """Port of ``test_section.m``."""

    def test_section_works(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test_sec = w.section([-3, 3], [], [], [])
        test_cut = w.cut(line_proj([1, 0, 0], [0, 1, 0]), [-2.5, 1, 2.5], [], [], [])

        ok, mess = test_sec.equal_to_tol(test_cut)
        assert ok, mess
        assert _is_bins_subset_of(test_sec.data.axes.p, w.data.axes.p)

    def test_experiment_reduced_on_file(self) -> None:
        tf = str(TEST_COMMON_DIR / "sqw_2d_2.sqw")
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 2000
            w2 = sqw(tf, file_backed=True)
            assert w2.is_filebacked
            assert w2.experiment_info.n_runs == 186
            ws = w2.section([0, 0.6], [160, 180])
            assert ws.experiment_info.n_runs == 36
            wsc = ws.recompute_bin_data()
            ok, mess = wsc.equal_to_tol(ws, tol=float(2 * np.finfo(np.float32).eps))
            assert ok, mess
            w2.deactivate()
        finally:
            hc.mem_chunk_size = old_chunk

    def test_experiment_reduced_in_mem(self) -> None:
        tf = str(TEST_COMMON_DIR / "sqw_2d_2.sqw")
        w2 = read_sqw(tf)
        assert not w2.is_filebacked
        assert w2.experiment_info.n_runs == 68
        ws = w2.section([0, 0.6], [160, 180])
        assert ws.experiment_info.n_runs == 36
        wsc = ws.recompute_bin_data()
        ok, mess = wsc.equal_to_tol(ws, tol=float(2 * np.finfo(np.float32).eps))
        assert ok, mess

    def test_section_sqw_fb(self, tmp_path: Path) -> None:
        w = sqw.generate_cube_sqw(10)
        test_sec_mb = w.section([-3, 3], [], [], [])
        test_file = str(tmp_path / "test_section_fb.sqw")
        w.save(test_file)

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 500
            wfb = sqw(test_file, file_backed=True)
            test_sec_fb = wfb.section([-3, 3], [], [], [])
            ok, mess = test_sec_mb.equal_to_tol(
                test_sec_fb, "-ignore_date", ignore_str=True
            )
            assert ok, mess
            wfb.deactivate()
            test_sec_fb.deactivate()
        finally:
            hc.mem_chunk_size = old_chunk
            gc.collect()

    def test_section_collapse_dim(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test_sec = w.section([-1, 0], [], [], [])
        test_cut = w.cut(line_proj([1, 0, 0], [0, 1, 0]), [-1, 0], [], [], [])

        ok, mess = test_sec.equal_to_tol(test_cut)
        assert ok, mess
        combined_p = [test_sec.data.axes.iint.T.ravel()] + list(test_sec.data.axes.p)
        assert _is_bins_subset_of(combined_p, w.data.axes.p)

    def test_section_sqw_array(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test = [w, w]
        test_sec = sqw.section(test, [-3, 3], [], [], [])
        test_cut = w.cut(line_proj([1, 0, 0], [0, 1, 0]), [-2.5, 1, 2.5], [], [], [])

        ok1, mess1 = test_sec[0].equal_to_tol(test_cut)
        assert ok1, mess1
        ok2, mess2 = test_sec[1].equal_to_tol(test_cut)
        assert ok2, mess2
        assert _is_bins_subset_of(test_sec[0].data.axes.p, w.data.axes.p)
        assert _is_bins_subset_of(test_sec[1].data.axes.p, w.data.axes.p)

    def test_section_get_all_bins(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test = w.section([-10, 10], [], [], [])
        ok, mess = test.equal_to_tol(w)
        assert ok, mess

    def test_section_preserves_original(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test = w.section([], [], [], [])
        ok, mess = test.equal_to_tol(w, "-ignore_date")
        assert ok, mess

    def test_section_no_arg_in_unchanged(self) -> None:
        w = sqw.generate_cube_sqw(10)
        test = w.section()
        ok, mess = test.equal_to_tol(w)
        assert ok, mess

    def test_section_fails_0_dim(self) -> None:
        w = sqw.generate_cube_sqw(2)
        w = w.cut(line_proj([1, 0, 0], [0, 1, 0]), [-1, 1], [-1, 1], [-1, 1], [-1, 1])
        assert int(w.dimensions()) == 0
        with pytest.raises(HoraceError, match="Cannot section a zero dimensional object"):
            w.section([], [], [], [])

    def test_section_fails_diff_dim(self) -> None:
        w = sqw.generate_cube_sqw(2)
        test = [
            w.cut(line_proj([1, 0, 0], [0, 1, 0]), [], [], [-1, 1], [-1, 1]),
            w.copy(),
        ]
        assert int(test[0].dimensions()) == 2
        assert int(test[1].dimensions()) == 4
        with pytest.raises(
            HoraceError,
            match="All objects must have same dimensionality for sectioning to work",
        ):
            sqw.section(test, [], [], [], [])

    def test_section_fails_bad_narg(self) -> None:
        w = sqw.generate_cube_sqw(2)
        assert int(w.dimensions()) == 4
        with pytest.raises(HoraceError, match="Check number of arguments"):
            w.section([], [], [])

    def test_find_section_3D(self) -> None:
        nbins = np.ones((10, 10, 10), dtype=int)
        sel = np.array([[4, 4, 4], [8, 8, 8]], dtype=int)
        bl_start, bl_size = PageOp_section.find_section(nbins, sel)
        np.testing.assert_array_equal(bl_start[:6], np.array([334, 344, 354, 364, 374, 434]))
        np.testing.assert_array_equal(bl_size, 5 * np.ones(25, dtype=int))

    def test_find_section_2D(self) -> None:
        nbins = np.ones((10, 10), dtype=int)
        sel = np.array([[4, 4], [8, 8]], dtype=int)
        bl_start, bl_size = PageOp_section.find_section(nbins, sel)
        np.testing.assert_array_equal(bl_start, np.array([34, 44, 54, 64, 74]))
        np.testing.assert_array_equal(bl_size, 5 * np.ones(5, dtype=int))

    def test_find_section_1D(self) -> None:
        nbins = np.ones((1, 10), dtype=int)
        sel = np.array([[4], [8]], dtype=int)
        bl_start, bl_size = PageOp_section.find_section(nbins, sel)
        np.testing.assert_array_equal(bl_start, np.array([4]))
        np.testing.assert_array_equal(bl_size, np.array([5]))
