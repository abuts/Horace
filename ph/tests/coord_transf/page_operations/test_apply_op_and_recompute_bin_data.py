"""Tests for ``PageOpBase``, ``PageOp_recompute_bins``, ``sqw.recompute_bin_data``,
and ``sqw.finalize_alignment``.

Translated from ``_test/test_sqw_pageOpMethods/test_apply_op_and_recompute_bin_data.m``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pyhorace.coord_transf.page_operations import PageOp_recompute_bins, PageOpBase
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.sqw.sqw import sqw


class TestApplyOpAndRecomputeBinData:
    """Port of ``test_apply_op_and_recompute_bin_data``."""

    def test_page_op_on_pix(self) -> None:
        pix = PixelDataMemory(np.ones((9, 1000), dtype=np.float64))
        op = PageOpBase(pix)
        assert op.changes_pix_only
        assert not op.exp_modified
        assert op.img is None
        assert np.array_equal(op.npix, np.array([1000]))
        assert np.array_equal(op.pix_data_range, PixelDataBase.EMPTY_RANGE)

    def test_recompute_on_pix(self) -> None:
        pix = PixelDataMemory(np.ones((9, 1000), dtype=np.float64))
        op = PageOp_recompute_bins()
        op = op.init(pix)
        assert op.changes_pix_only

    def test_recompute_bin_data_empty(self) -> None:
        tsqw = sqw()
        new_sqw = tsqw.recompute_bin_data()
        assert tsqw == new_sqw

    def test_recompute_bin_data_populated(self) -> None:
        tsqw = sqw.generate_cube_sqw(2)
        tsqw.data.s = np.zeros((2, 2, 2, 2), dtype=np.float64)
        tsqw.data.e = np.zeros((2, 2, 2, 2), dtype=np.float64)

        new_sqw = tsqw.recompute_bin_data()
        expected = np.arange(1, 17, dtype=np.float64).reshape((2, 2, 2, 2), order="F")
        assert np.allclose(new_sqw.data.s, expected)
        assert np.allclose(new_sqw.data.e, expected)

    def test_recompute_bin_data_with_alignment(self) -> None:
        tsqw = sqw.generate_cube_sqw(2)
        s = tsqw.data.s.copy()
        e = tsqw.data.e.copy()
        tsqw.data.s = np.zeros((2, 2, 2, 2), dtype=np.float64)
        tsqw.data.e = np.zeros((2, 2, 2, 2), dtype=np.float64)

        rm = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        tsqw.pix = tsqw.pix.set_raw_alignment(rm)
        tsqw.pix = tsqw.pix.invalidate_range()

        assert tsqw.pix.is_corrected
        assert not tsqw.pix.is_range_valid()
        new_sqw = tsqw.finalize_alignment()
        assert not new_sqw.pix.is_corrected
        assert new_sqw.pix.is_range_valid()

        assert np.allclose(new_sqw.data.s, s)
        assert np.allclose(new_sqw.data.e, e)

    def test_recompute_bin_data_in_memory(self) -> None:
        tsqw = sqw.generate_cube_sqw(4)
        npix = tsqw.data.npix.copy()
        tsqw.pix.signal = 4.0
        tsqw.pix.variance = 1.0
        tsqw.pix = tsqw.pix.invalidate_range()
        assert not tsqw.pix.is_range_valid()

        new_sqw = tsqw.recompute_bin_data()
        assert np.allclose(new_sqw.data.s, 4.0 * npix)
        assert np.allclose(new_sqw.data.e, npix)
        assert new_sqw.pix.is_range_valid()

    def test_recompute_bin_data_on_file_with_same_name(self, tmp_path) -> None:
        tsqw = sqw.generate_cube_sqw(4)
        npix = tsqw.data.npix.copy()
        tsqw.pix.signal = 4.0
        tsqw.pix.variance = 1.0
        tsqw.pix = tsqw.pix.invalidate_range()
        test_file = str(tmp_path / "recompute_with_pages_same.sqw")
        tsqw.save(test_file)

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 20
            tsqw_fb = sqw(test_file, file_backed=True)
            assert isinstance(tsqw_fb.pix, PixelDataFileBacked)

            new_sqw = tsqw_fb.recompute_bin_data(test_file)
            assert np.allclose(new_sqw.data.s, 4.0 * npix)
            assert np.allclose(new_sqw.data.e, npix)
            assert new_sqw.pix.is_range_valid()
            assert os.path.normcase(new_sqw.full_filename) == os.path.normcase(test_file)
            new_sqw.deactivate()
            tsqw_fb.deactivate()

            assert os.path.isfile(test_file)
            tsqw_reloaded = sqw(test_file)
            assert np.allclose(tsqw_reloaded.data.s, 4.0 * npix)
            assert np.allclose(tsqw_reloaded.data.e, npix)
            assert tsqw_reloaded.pix.is_range_valid()
        finally:
            hc.mem_chunk_size = old_chunk

    def test_recompute_bin_data_on_file_with_other_name(self, tmp_path) -> None:
        tsqw = sqw.generate_cube_sqw(4)
        npix = tsqw.data.npix.copy()
        tsqw.pix.signal = 4.0
        tsqw.pix.variance = 1.0
        tsqw.pix = tsqw.pix.invalidate_range()
        test_source_file = str(tmp_path / "recompute_with_pages.sqw")
        test_targ_file = str(tmp_path / "recompute_with_pages_targ.sqw")
        tsqw.save(test_source_file)

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 20
            tsqw_fb = sqw(test_source_file, file_backed=True)
            assert isinstance(tsqw_fb.pix, PixelDataFileBacked)

            new_sqw = tsqw_fb.recompute_bin_data(test_targ_file)
            assert np.allclose(new_sqw.data.s, 4.0 * npix)
            assert np.allclose(new_sqw.data.e, npix)
            assert new_sqw.pix.is_range_valid()
            assert os.path.normcase(new_sqw.full_filename) == os.path.normcase(test_targ_file)
            new_sqw.deactivate()
            tsqw_fb.deactivate()

            assert os.path.isfile(test_targ_file)
            tsqw_reloaded = sqw(test_targ_file)
            assert np.allclose(tsqw_reloaded.data.s, 4.0 * npix, rtol=4 * np.finfo(np.float32).eps)
            assert np.allclose(tsqw_reloaded.data.e, npix, rtol=4 * np.finfo(np.float32).eps)
            assert tsqw_reloaded.pix.is_range_valid()
        finally:
            hc.mem_chunk_size = old_chunk

    def test_recompute_bin_data_on_file(self) -> None:
        tsqw = sqw.generate_cube_sqw(2)
        s = tsqw.data.s.copy()
        e = tsqw.data.e.copy()
        tsqw.data.s = np.zeros((2, 2, 2, 2), dtype=np.float64)
        tsqw.data.e = np.zeros((2, 2, 2, 2), dtype=np.float64)

        pix = PixelDataFileBacked(tsqw.pix.data)
        try:
            tsqw.pix = pix.invalidate_range()

            new_sqw = tsqw.recompute_bin_data()
            assert new_sqw.pix.is_range_valid()
            assert new_sqw.is_tmp_obj
            tmp_file = new_sqw.full_filename
            assert os.path.isfile(tmp_file)

            assert np.allclose(new_sqw.data.s, s)
            assert np.allclose(new_sqw.data.e, e)
            new_sqw.delete()
            assert not os.path.isfile(tmp_file)
        finally:
            pix.delete()

    def test_recompute_bin_data_on_file_with_paging(self) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 3
            tsqw = sqw.generate_cube_sqw(2)
            s = tsqw.data.s.copy()
            e = tsqw.data.e.copy()
            tsqw.data.s = np.zeros((2, 2, 2, 2), dtype=np.float64)
            tsqw.data.e = np.zeros((2, 2, 2, 2), dtype=np.float64)

            pix = PixelDataFileBacked(tsqw.pix.data)
            try:
                tsqw.pix = pix.invalidate_range()
                assert tsqw.pix.num_pages == 6

                new_sqw = tsqw.recompute_bin_data()
                assert new_sqw.pix.is_range_valid()
                assert new_sqw.is_tmp_obj
                tmp_file = new_sqw.full_filename
                assert os.path.isfile(tmp_file)

                assert np.allclose(new_sqw.data.s, s)
                assert np.allclose(new_sqw.data.e, e)
                new_sqw.delete()
                assert not os.path.isfile(tmp_file)
            finally:
                pix.delete()
        finally:
            hc.mem_chunk_size = old_chunk
