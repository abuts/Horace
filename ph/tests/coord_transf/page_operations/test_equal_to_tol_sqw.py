"""Tests for sqw.equal_to_tol (port of test_equal_to_tol_sqw.m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.core.configurations import hor_config
from pyhorace.core.serializable import compare_values
from pyhorace.data.pixels import PixelDataBase, PixelDataFileBacked, PixelDataMemory
from pyhorace.sqw.read_dnd import read_dnd
from pyhorace.sqw.sqw import sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
TEST_SQW_FILE = REPO_ROOT / "_test" / "common_data" / "sqw_2d_1.sqw"
TEST_DND_FILE = REPO_ROOT / "_test" / "common_data" / "dnd_2d.sqw"


@pytest.fixture(scope="module")
def sqw_objs():
    sqw_2d_paged = sqw(str(TEST_SQW_FILE), file_backed=True)
    assert sqw_2d_paged.pix.is_filebacked
    sqw_2d = sqw(str(TEST_SQW_FILE))
    return sqw_2d, sqw_2d_paged


class TestEqualToTolSqw:
    """Port of ``_test/test_sqw_pageOpMethods/test_equal_to_tol_sqw.m``."""

    @staticmethod
    def shuffle_pixel_bin_rows(pix: PixelDataBase, npix: np.ndarray) -> PixelDataBase:
        rng = np.random.default_rng(42)
        npix_non_empty = npix.ravel(order="F")
        npix_non_empty = npix_non_empty[npix_non_empty != 0].astype(np.int64)
        shuffled_pix = PixelDataBase.create(pix.num_pixels)

        bin_end_idxs = np.cumsum(npix_non_empty)
        bin_start_idxs = bin_end_idxs - npix_non_empty
        for i in range(npix_non_empty.size):
            bin_start = int(bin_start_idxs[i])
            bin_end = int(bin_end_idxs[i])
            pix_in_bin = pix.data[:, bin_start:bin_end]
            perm = rng.permutation(pix_in_bin.shape[1])
            shuffled_bin_pix = pix_in_bin[:, perm]
            shuffled_pix.data[:, bin_start:bin_end] = shuffled_bin_pix
            np.testing.assert_allclose(
                np.sum(shuffled_bin_pix, axis=1), np.sum(pix_in_bin, axis=1), atol=1e-8
            )
        return shuffled_pix

    @staticmethod
    def get_pix_with_fake_faccess(data: np.ndarray) -> PixelDataFileBacked:
        return PixelDataFileBacked(data)

    def test_datetime_equal_to_tol(self):
        ok, mess = compare_values(
            "2023-07-18T18:47:01",
            "2023-07-18T18:47:59",
            name_a="input_1",
            name_b="input_2",
        )
        assert ok, mess
        ok, mess = compare_values(
            "2023-07-18T18:47:41",
            "2023-07-18T18:49:41",
            name_a="input_1",
            name_b="input_2",
        )
        assert not ok
        assert (
            mess
            == "Object input_1 with Time=18-Jul-2023 18:47:41  and object input_2, with Time=18-Jul-2023 18:49:41 are different for more than a minute"
        )

    def test_the_same_sqw_objects_are_equal(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        sqw_copy = sqw_2d.copy()
        ok, mess = sqw_2d.equal_to_tol(sqw_copy)
        assert ok, mess

    def test_sqw_and_d2d_objects_are_not_equal(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        dnd_2d_ = read_dnd(str(TEST_DND_FILE))
        ok, mess = sqw_2d.equal_to_tol(dnd_2d_)
        assert not ok
        assert (
            mess
            == 'Objects have different types. "input_1" has class: "sqw" and "input_2" class: "d2d"'
        )

    def test_the_same_sqw_objects_are_equal_with_no_pix_reorder(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        sqw_copy = sqw_2d.copy()
        ok, mess = sqw_2d.equal_to_tol(sqw_copy, "reorder", False)
        assert ok, mess

    def test_the_same_sqw_objects_are_equal_if_testing_fraction_of_pix(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        sqw_copy = sqw_2d.copy()
        ok, mess = sqw_2d.equal_to_tol(sqw_copy, "fraction", 0.5)
        assert ok, mess

    def test_same_sqw_objs_equal_if_pixels_in_each_bin_shuffled(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        original_sqw = sqw_2d.copy()
        pix = original_sqw.pix
        npix = original_sqw.data.npix

        shuffled_sqw = original_sqw.copy()
        shuffled_sqw.pix = self.shuffle_pixel_bin_rows(pix, npix)

        ok, mess = shuffled_sqw.equal_to_tol(original_sqw)
        assert ok, mess

    def test_paged_mem_sqw_ne_change_2nd_page(self):
        new_sqw = sqw.generate_cube_sqw(10)
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 50
            new_sqw_paged = new_sqw.copy()
            new_sqw_paged.pix.data[3, 179] = -1.0
            new_sqw_paged.pix = PixelDataFileBacked(new_sqw_paged.pix)

            assert isinstance(new_sqw.pix, PixelDataMemory)
            assert isinstance(new_sqw_paged.pix, PixelDataFileBacked)

            ok1, _ = new_sqw.equal_to_tol(new_sqw_paged)
            assert not ok1
            ok2, _ = new_sqw_paged.equal_to_tol(new_sqw)
            assert not ok2
        finally:
            hc.mem_chunk_size = old_chunk

    def test_paged_sqw_objects_equal_if_pix_within_each_bin_shuffled(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        original_sqw = sqw_2d_paged.copy()
        npix = np.array([10, 5, 6, 3, 6], dtype=np.int64)

        rng = np.random.default_rng(42)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, 30))
        shuffled_data = self.shuffle_pixel_bin_rows(PixelDataBase.create(data), npix).data

        pix = self.get_pix_with_fake_faccess(data)
        shuffled_pix = self.get_pix_with_fake_faccess(shuffled_data)

        original_sqw.do_check_combo_arg = False
        original_sqw.data.do_check_combo_arg = False
        original_sqw.data.npix = npix
        original_sqw.pix = pix
        shuffled_sqw = original_sqw.copy()
        shuffled_sqw.do_check_combo_arg = False
        shuffled_sqw.pix = shuffled_pix

        ok, mess = shuffled_sqw.equal_to_tol(original_sqw)
        assert ok, mess

    def test_paged_sqw_ne_if_pix_within_bin_shuffled_and_reorder_false(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        original_sqw = sqw_2d_paged.copy()
        npix = np.array([10, 5, 6, 3, 6], dtype=np.int64)

        rng = np.random.default_rng(42)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, 30))
        shuffled_data = self.shuffle_pixel_bin_rows(PixelDataBase.create(data), npix).data

        pix = self.get_pix_with_fake_faccess(data)
        shuffled_pix = self.get_pix_with_fake_faccess(shuffled_data)

        original_sqw.do_check_combo_arg = False
        original_sqw.data.do_check_combo_arg = False
        original_sqw.data.npix = npix
        original_sqw.pix = pix
        shuffled_sqw = original_sqw.copy()
        shuffled_sqw.do_check_combo_arg = False
        shuffled_sqw.pix = shuffled_pix

        ok, _ = shuffled_sqw.equal_to_tol(original_sqw, "reorder", False)
        assert not ok

    def test_the_same_sqw_objects_are_equal_with_paged_pix(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        sqw_copy = sqw_2d_paged.copy()
        ok, mess = sqw_2d_paged.equal_to_tol(sqw_copy)
        assert ok, mess

    def test_the_same_sqw_objs_eq_if_fraction_of_pix_and_pix_are_paged(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        sqw_copy = sqw_2d_paged.copy()
        ok, mess = sqw_2d_paged.equal_to_tol(sqw_copy, "fraction", 0.5)
        assert ok, mess

    def test_paged_and_non_paged_version_of_same_sqw_file_are_equal(self, sqw_objs):
        sqw_2d, sqw_2d_paged = sqw_objs
        pobj = sqw_2d_paged.copy()
        pobj.experiment_info = sqw_2d.experiment_info
        pobj.main_header.nfiles = sqw_2d.main_header.nfiles
        ok, mess = pobj.equal_to_tol(sqw_2d, "fraction", 0.5, "-ignore_date")
        assert ok, mess

    def test_false_returned_if_NaNs_in_sqw_and_nan_equal_is_false(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        original_sqw = sqw_2d.copy()
        multi_idx = np.unravel_index(4, original_sqw.data.s.shape, order="F")
        original_sqw.data.s[multi_idx] = np.nan
        sqw_copy = original_sqw.copy()

        ok1, _ = original_sqw.equal_to_tol(sqw_2d, "nan_equal", False)
        assert not ok1
        ok2, _ = sqw_copy.equal_to_tol(original_sqw, "nan_equal", False)
        assert not ok2

    def test_true_returned_if_NaNs_in_sqw_and_nan_equal_is_true(self, sqw_objs):
        sqw_2d, _ = sqw_objs
        original_sqw = sqw_2d.copy()
        multi_idx = np.unravel_index(4, original_sqw.data.s.shape, order="F")
        original_sqw.data.s[multi_idx] = np.nan
        sqw_copy = original_sqw.copy()

        ok1, _ = original_sqw.equal_to_tol(sqw_2d, "nan_equal", True)
        assert not ok1
        ok2, mess = sqw_copy.equal_to_tol(original_sqw, "nan_equal", True)
        assert ok2, mess

    def test_using_fraction_argument_only_tests_fraction_of_the_bins(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        original_sqw = sqw_2d_paged.copy()
        npix = np.array([10, 5, 6, 3, 6], dtype=np.int64)
        rng = np.random.default_rng(42)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, 30))
        edited_data = data.copy()

        fraction = 0.5
        edited_data[:, 10:15] = 0.0
        edited_data[:, 21:24] = 0.0

        pix = self.get_pix_with_fake_faccess(data)
        edited_pix = self.get_pix_with_fake_faccess(edited_data)

        original_sqw.do_check_combo_arg = False
        original_sqw.data.do_check_combo_arg = False
        original_sqw.data.npix = npix
        original_sqw.pix = pix
        edited_sqw = original_sqw.copy()
        edited_sqw.do_check_combo_arg = False
        edited_sqw.pix = edited_pix

        ok1, _ = edited_sqw.equal_to_tol(original_sqw, "fraction", 1.0)
        assert not ok1
        ok2, mess = edited_sqw.equal_to_tol(original_sqw, "fraction", fraction, "reorder", True)
        assert ok2, mess

    def test_equal_to_tol_true_for_eq_sqw_reorder_false_with_fraction(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        sqw_copy = sqw_2d_paged.copy()
        ok, mess = sqw_2d_paged.equal_to_tol(sqw_copy, "fraction", 0.5, "reorder", False)
        assert ok, mess

    def test_equal_to_tol_can_be_called_with_negative_tol_for_rel_tol(self, sqw_objs):
        _, sqw_2d_paged = sqw_objs
        sqw_copy = sqw_2d_paged.copy()
        rel_tol = 1e-5
        ok, mess = sqw_copy.equal_to_tol(sqw_2d_paged, -rel_tol)
        assert ok, mess

        flat_s = sqw_copy.data.s.ravel(order="F")
        sig_idx = int(np.flatnonzero(flat_s > 0)[0])
        multi_idx = np.unravel_index(sig_idx, sqw_copy.data.s.shape, order="F")

        orig_val = float(sqw_2d_paged.data.s[multi_idx])
        value_diff = 1.01 * rel_tol * orig_val
        sqw_copy.data.s[multi_idx] = orig_val + value_diff

        ok1, _ = sqw_copy.equal_to_tol(sqw_2d_paged, -rel_tol)
        assert not ok1

        ok2, mess2 = sqw_copy.equal_to_tol(sqw_2d_paged, -rel_tol * 1.01)
        assert ok2, mess2

        ok3, mess3 = sqw_copy.equal_to_tol(sqw_2d_paged, value_diff + 1e-8)
        assert ok3, mess3
