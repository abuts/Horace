"""Tests for PageOp_mask and sqw.mask / mask_pixels / mask_random_pixels / slim.

Ported from ``_test/test_sqw_pageOpMethods/test_mask_sqw.m``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.coord_transf.page_operations.page_op_mask import PageOp_mask
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.sqw import sqw


REPO_ROOT = Path(__file__).resolve().parents[4]
COMMON_DATA_DIR = REPO_ROOT / "_test" / "common_data"
SQW_2D_FILE = str(COMMON_DATA_DIR / "sqw_2d_1.sqw")
SQW_3D_FILE = str(COMMON_DATA_DIR / "w3d_sqw.sqw")


def _concatenate_pixel_pages(pix) -> np.ndarray:
    pages = []
    saved = pix.page_num
    try:
        for pg in range(1, pix.num_pages + 1):
            pix.page_num = pg
            pages.append(pix.data.copy())
    finally:
        pix.page_num = saved
    return np.hstack(pages) if pages else np.empty((9, 0), dtype=np.float64)


def _check_filebacked_signal_averages(sqw_to_check: sqw) -> None:
    assert sqw_to_check.pix.num_pixels == int(np.sum(sqw_to_check.data.npix))
    ssignal = 0.0
    serror = 0.0
    pix = sqw_to_check.pix
    saved = pix.page_num
    try:
        for i in range(1, pix.num_pages + 1):
            pix.page_num = i
            ssignal += float(np.sum(pix.signal))
            serror += float(np.sum(pix.variance))
    finally:
        pix.page_num = saved

    s_flat = np.asarray(sqw_to_check.data.s, dtype=np.float64).ravel(order="F")
    e_flat = np.asarray(sqw_to_check.data.e, dtype=np.float64).ravel(order="F")
    npix_flat = np.asarray(sqw_to_check.data.npix, dtype=np.float64).ravel(order="F")

    reltol = float(4.0 * np.finfo(np.float32).eps)
    assert np.isclose(ssignal, float(np.sum(s_flat * npix_flat)), rtol=reltol, atol=1e-6)
    assert np.isclose(
        serror, float(np.sum(e_flat * (npix_flat**2))), rtol=reltol, atol=1e-6
    )


@pytest.fixture(scope="module")
def mask_fixture():
    sqw_2d = sqw(SQW_2D_FILE, file_backed=False).recompute_bin_data()
    sqw_2d_num_pixels = sqw_2d.pix.num_pixels

    idxs_to_mask_1based = np.array([2, 46, 91, 93, 94, 107, 123, 166], dtype=np.intp)
    mask_flat_2d = np.ones(sqw_2d.data.npix.size, dtype=bool)
    mask_flat_2d[idxs_to_mask_1based - 1] = False
    mask_array_2d = mask_flat_2d.reshape(sqw_2d.data.npix.shape, order="F")

    sqw_3d = sqw(SQW_3D_FILE, file_backed=False)
    num_bins = sqw_3d.data.npix.size
    mask_flat_3d = np.ones(num_bins, dtype=bool)
    mask_flat_3d[: num_bins // 2] = False
    mask_array_3d = mask_flat_3d.reshape(sqw_3d.data.npix.shape, order="F")
    sqw_3d = sqw_3d.recompute_bin_data()
    masked_3d = sqw_3d.mask(mask_array_3d)

    dummy_sample = [sqw.generate_cube_sqw(4), sqw.generate_cube_sqw(5)]

    return {
        "sqw_2d": sqw_2d,
        "sqw_2d_num_pixels": sqw_2d_num_pixels,
        "mask_array_2d": mask_array_2d,
        "sqw_3d": sqw_3d,
        "mask_array_3d": mask_array_3d,
        "masked_3d": masked_3d,
        "dummy_sample": dummy_sample,
    }


def _get_paged_sqw(file_path: str, mask_array: np.ndarray, monkeypatch):
    paged_sqw = sqw(file_path, file_backed=True)
    new_pg_size = max(1, paged_sqw.pix.num_pixels // 6)
    hc = hor_config()
    monkeypatch.setattr(hc, "mem_chunk_size", new_pg_size)
    masked_sqw = paged_sqw.mask(mask_array)
    _check_filebacked_signal_averages(paged_sqw)
    return paged_sqw, masked_sqw


class TestMaskSqw:
    """Ported from ``_test/test_sqw_pageOpMethods/test_mask_sqw.m``."""

    def test_mem_fb_equal_after_mask(self, mask_fixture, monkeypatch):
        sqw_2d = mask_fixture["sqw_2d"]
        mask_array_2d = mask_fixture["mask_array_2d"]
        masked_2d = sqw_2d.mask(mask_array_2d)
        _check_filebacked_signal_averages(masked_2d)
        _, masked_2d_paged = _get_paged_sqw(SQW_2D_FILE, mask_array_2d, monkeypatch)
        ok, mess = masked_2d_paged.pix.equal_to_tol(masked_2d.pix, tol=[1e-6, 1e-6])
        assert ok, mess

    def test_mask_works_with_paged(self, mask_fixture, monkeypatch):
        sqw_2d = mask_fixture["sqw_2d"]
        mask_array_2d = mask_fixture["mask_array_2d"]
        _, masked_2d_paged = _get_paged_sqw(SQW_2D_FILE, mask_array_2d, monkeypatch)
        _check_filebacked_signal_averages(masked_2d_paged)

        assert np.sum(masked_2d_paged.data.npix[~mask_array_2d]) == 0
        assert np.sum(masked_2d_paged.data.s[~mask_array_2d]) == 0
        assert np.sum(masked_2d_paged.data.e[~mask_array_2d]) == 0

        np.testing.assert_allclose(
            masked_2d_paged.data.s[mask_array_2d],
            sqw_2d.data.s[mask_array_2d],
            atol=0.01,
        )
        np.testing.assert_allclose(
            masked_2d_paged.data.e[mask_array_2d],
            sqw_2d.data.e[mask_array_2d],
            atol=0.13,
        )
        np.testing.assert_allclose(
            masked_2d_paged.data.npix[mask_array_2d],
            sqw_2d.data.npix[mask_array_2d],
        )

        expected_num_pix = int(np.sum(sqw_2d.data.npix[mask_array_2d]))
        assert masked_2d_paged.pix.num_pixels == expected_num_pix

        original_pix_range = sqw_2d.pix.pix_range
        new_pix_range = masked_2d_paged.pix.pix_range
        img_range_diff = np.abs(original_pix_range - new_pix_range)
        assert img_range_diff.ravel(order="F")[0] > 0.001
        np.testing.assert_allclose(
            original_pix_range[1, [0, 1, 3]],
            new_pix_range[1, [0, 1, 3]],
            atol=0.001,
        )
        np.testing.assert_allclose(
            original_pix_range[0, 2:],
            new_pix_range[0, 2:],
            atol=0.001,
        )

    def test_mask_sets_npix_in_masked_bins_to_zero_3d(self, mask_fixture):
        masked_3d = mask_fixture["masked_3d"]
        mask_array_3d = mask_fixture["mask_array_3d"]
        assert np.sum(masked_3d.data.npix[~mask_array_3d]) == 0

    def test_mask_works_with_paged_sqw_3d(self, mask_fixture, monkeypatch):
        sqw_3d = mask_fixture["sqw_3d"]
        mask_array_3d = mask_fixture["mask_array_3d"]
        masked_3d = mask_fixture["masked_3d"]

        _, masked_3d_paged = _get_paged_sqw(SQW_3D_FILE, mask_array_3d, monkeypatch)

        assert np.sum(masked_3d_paged.data.npix[~mask_array_3d]) == 0
        assert np.sum(masked_3d.data.s[~mask_array_3d]) == 0
        assert np.sum(masked_3d_paged.data.s[~mask_array_3d]) == 0
        assert np.sum(masked_3d.data.e[~mask_array_3d]) == 0
        assert np.sum(masked_3d_paged.data.e[~mask_array_3d]) == 0

        np.testing.assert_allclose(
            masked_3d.data.s[mask_array_3d],
            sqw_3d.data.s[mask_array_3d],
            atol=float(32 * np.finfo(np.float32).eps),
        )
        np.testing.assert_allclose(
            masked_3d_paged.data.s[mask_array_3d],
            sqw_3d.data.s[mask_array_3d],
            atol=float(32 * np.finfo(np.float32).eps),
        )
        np.testing.assert_allclose(
            masked_3d.data.e[mask_array_3d],
            sqw_3d.data.e[mask_array_3d],
            atol=float(8 * np.finfo(np.float32).eps),
        )
        np.testing.assert_array_equal(
            masked_3d_paged.data.npix[mask_array_3d],
            sqw_3d.data.npix[mask_array_3d],
        )
        np.testing.assert_allclose(
            masked_3d_paged.data.e[mask_array_3d],
            sqw_3d.data.e[mask_array_3d],
            atol=float(8 * np.finfo(np.float32).eps),
        )

        expected_num_pix = int(np.sum(sqw_3d.data.npix[mask_array_3d]))
        assert masked_3d_paged.pix.num_pixels == expected_num_pix

        raw_paged_pix = PixelDataMemory(masked_3d_paged.pix)
        ok, mess = raw_paged_pix.equal_to_tol(masked_3d.pix, tol=[1e-6, 1e-6])
        assert ok, mess

        original_img_range = sqw_3d.data.img_range
        img_range_diff = np.abs(original_img_range - masked_3d_paged.data.img_range)
        assert np.all(img_range_diff < 1e-7)

        paged_img_range = masked_3d_paged.data.img_range
        mem_img_range = masked_3d.data.img_range
        np.testing.assert_allclose(mem_img_range, paged_img_range, atol=0.001)

    def test_mask_does_not_change_unmasked_bins_npix_3d(self, mask_fixture):
        masked_3d = mask_fixture["masked_3d"]
        sqw_3d = mask_fixture["sqw_3d"]
        mask_array_3d = mask_fixture["mask_array_3d"]
        np.testing.assert_array_equal(
            masked_3d.data.npix[mask_array_3d],
            sqw_3d.data.npix[mask_array_3d],
        )

    def test_num_pixels_has_been_reduced_by_correct_amount_3d(self, mask_fixture):
        masked_3d = mask_fixture["masked_3d"]
        sqw_3d = mask_fixture["sqw_3d"]
        mask_array_3d = mask_fixture["mask_array_3d"]
        expected_num_pix = int(np.sum(sqw_3d.data.npix[mask_array_3d]))
        assert masked_3d.pix.num_pixels == expected_num_pix

    def test_img_range_recalculated_after_mask_3d(self, mask_fixture):
        masked_3d = mask_fixture["masked_3d"]
        sqw_3d = mask_fixture["sqw_3d"]
        original_img_range = sqw_3d.data.img_range
        img_range_diff = np.abs(original_img_range - masked_3d.data.img_range)
        assert np.all(img_range_diff < 1e-7)

    def test_mask_pixels_removes_pixels_given_in_mask_array(self):
        sqw_obj = sqw(SQW_2D_FILE, file_backed=False)
        mask_array = np.ones(sqw_obj.pix.num_pixels, dtype=bool)
        median_u1_range = float(np.median(sqw_obj.pix.u1))
        pix_to_remove = sqw_obj.pix.u1.ravel() > median_u1_range
        mask_array[pix_to_remove] = False

        new_sqw = sqw_obj.mask_pixels(mask_array)
        assert new_sqw.pix.num_pixels == int(np.sum(mask_array))
        np.testing.assert_allclose(
            new_sqw.data.img_range, sqw_obj.data.img_range, atol=1e-7
        )
        _check_filebacked_signal_averages(new_sqw)

    def test_mask_works_in_memory(self, mask_fixture, monkeypatch):
        sqw_2d = mask_fixture["sqw_2d"]
        mask_array_2d = mask_fixture["mask_array_2d"]
        test_sqw = sqw_2d.recompute_bin_data()
        masked_2d = test_sqw.mask(mask_array_2d)

        _check_filebacked_signal_averages(masked_2d)

        assert np.sum(masked_2d.data.npix[~mask_array_2d]) == 0
        assert np.sum(masked_2d.data.s[~mask_array_2d]) == 0
        assert np.sum(masked_2d.data.e[~mask_array_2d]) == 0

        np.testing.assert_allclose(
            masked_2d.data.s[mask_array_2d],
            test_sqw.data.s[mask_array_2d],
        )
        np.testing.assert_allclose(
            masked_2d.data.e[mask_array_2d],
            test_sqw.data.e[mask_array_2d],
        )
        np.testing.assert_array_equal(
            masked_2d.data.npix[mask_array_2d],
            test_sqw.data.npix[mask_array_2d],
        )

        expected_num_pix = int(np.sum(test_sqw.data.npix[mask_array_2d]))
        assert masked_2d.pix.num_pixels == expected_num_pix

        original_pix_range = sqw_2d.pix.pix_range
        new_pix_range = masked_2d.pix.pix_range
        pix_range_diff = np.abs(original_pix_range - new_pix_range)
        assert pix_range_diff.ravel(order="F")[0] > 0.001
        np.testing.assert_allclose(
            original_pix_range[1, :], new_pix_range[1, :], atol=0.001
        )
        np.testing.assert_allclose(
            original_pix_range[0, 2:], new_pix_range[0, 2:], atol=0.001
        )

        _, masked_2d_paged = _get_paged_sqw(SQW_2D_FILE, mask_array_2d, monkeypatch)
        raw_paged_pix = _concatenate_pixel_pages(masked_2d_paged.pix)
        np.testing.assert_allclose(raw_paged_pix, masked_2d.pix.data, atol=1e-6)

    def test_mask_random_fraction_equal_to_slim(self, mask_fixture):
        sqw_obj = mask_fixture["sqw_2d"]
        frac_to_keep = 1.0 / 5.0
        new_sqw = sqw_obj.mask_random_fraction_pixels(frac_to_keep)
        slim_sqw = sqw_obj.slim(5)
        assert new_sqw.num_pixels == slim_sqw.num_pixels

    def test_mask_random_fraction_removes_perc_of_pix_in_mem(self, mask_fixture):
        sqw_obj = mask_fixture["sqw_2d"]
        frac_to_keep = 0.8
        new_sqw = sqw_obj.mask_random_fraction_pixels(frac_to_keep)
        expected_num_pix = int(np.floor(frac_to_keep * sqw_obj.pix.num_pixels + 0.5))
        assert new_sqw.pix.num_pixels == expected_num_pix
        _check_filebacked_signal_averages(new_sqw)

    def test_mask_random_retains_correct_number_of_pix_on_file(
        self, mask_fixture, monkeypatch
    ):
        sqw_obj = mask_fixture["sqw_2d"]
        new_pg_size = max(1, mask_fixture["sqw_2d_num_pixels"] // 6)
        hc = hor_config()
        monkeypatch.setattr(hc, "mem_chunk_size", new_pg_size)

        num_pix_to_keep = 5000
        new_sqw = sqw_obj.mask_random_pixels(num_pix_to_keep)
        assert new_sqw.pix.num_pixels == num_pix_to_keep
        _check_filebacked_signal_averages(new_sqw)

    def test_mask_random_retains_correct_number_of_pix_in_mem(self, mask_fixture):
        sqw_obj = mask_fixture["sqw_2d"]
        num_pix_to_keep = 5000
        new_sqw = sqw_obj.mask_random_pixels(num_pix_to_keep)
        assert new_sqw.pix.num_pixels == num_pix_to_keep
        _check_filebacked_signal_averages(new_sqw)

    def test_mask_random_frac_pix_fails_on_nobj(self, mask_fixture):
        keep = [1, 1, 1]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_fraction_pixels(mask_fixture["dummy_sample"], keep)

    def test_mask_random_frac_pix_fails_on_large(self, mask_fixture):
        keep = [0.5, 1.1]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_fraction_pixels(mask_fixture["dummy_sample"], keep)

    def test_mask_random_frac_pix_fails_on_zero(self, mask_fixture):
        keep = [0.5, -1]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_fraction_pixels(mask_fixture["dummy_sample"], keep)

    def test_mask_random_pix_fails_on_nobj(self, mask_fixture):
        keep = [100, 100, 10]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_pixels(mask_fixture["dummy_sample"], keep)

    def test_mask_random_pix_fails_on_large(self, mask_fixture):
        keep = [100, 5**4 + 1]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_pixels(mask_fixture["dummy_sample"], keep)

    def test_mask_random_pix_fails_on_zero(self, mask_fixture):
        keep = [100, 0]
        with pytest.raises(HoraceError, match="invalid_argument"):
            sqw.mask_random_pixels(mask_fixture["dummy_sample"], keep)

    def test_split_in_pages_small_fraction(self):
        page_op = PageOp_mask()
        pieces = page_op.calc_page_share(300, 901, 300)
        assert pieces.size == 4
        assert int(np.sum(pieces)) == 300
        assert np.all(pieces > 0)

    def test_split_in_pages_large_fraction(self):
        page_op = PageOp_mask()
        pieces = page_op.calc_page_share(300, 890, 300)
        assert pieces.size == 3
        assert int(np.sum(pieces)) == 300
        assert np.all(pieces > 0)

    def test_split_in_pages_large_normal(self):
        page_op = PageOp_mask()
        pieces = page_op.calc_page_share(300, 1000, 300)
        assert pieces.size == 4
        assert int(np.sum(pieces)) == 300
        assert np.all(pieces > 0)
