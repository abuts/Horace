"""Tests for PageOp_cat_pix and PixelDataBase.cat.

Ported from ``_test/test_sqw_pageOpMethods/test_cat_combine.m``.
"""

from __future__ import annotations

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_cat_pix import PageOp_cat_pix
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory


def _get_random_pix_data(rows: int) -> PixelDataMemory:
    rng = np.random.default_rng(42 + rows)
    data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, rows))
    return PixelDataMemory(data)


class TestCatCombine:
    """Ported from ``_test/test_sqw_pageOpMethods/test_cat_combine.m``."""

    def test_cat_combines_given_PixelData_objects(self):
        pix_data_obj1 = _get_random_pix_data(10)
        pix_data_obj2 = _get_random_pix_data(5)

        combined_pix = pix_data_obj1.cat(pix_data_obj1, pix_data_obj2)

        assert combined_pix.num_pixels == 15
        np.testing.assert_array_equal(
            combined_pix.data,
            np.hstack([pix_data_obj1.data, pix_data_obj2.data]),
        )

    def test_get_page_data_4pix_large_odd_split_chunks(self):
        n_pix = 100
        rng = np.random.default_rng(101)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, n_pix))
        pix1 = PixelDataMemory(data[:, 0:25])
        pix2 = PixelDataMemory(data[:, 25:50])
        pix3 = PixelDataMemory(data[:, 50:75])
        pix4 = PixelDataMemory(data[:, 75:100])

        page_op = PageOp_cat_pix()
        page_op = page_op.init(pix1, pix2, pix3, pix4)
        page_op.pix = PixelDataFileBacked()

        assert page_op.npix.size == 4
        np.testing.assert_array_equal(page_op.npix, [25, 25, 25, 25])

        npix = page_op.npix
        npix_chunks, npix_idx, page_op = page_op.split_into_pages(npix, 33)
        assert len(npix_chunks) == 4
        assert list(npix_idx.shape) == [2, 4]
        np.testing.assert_array_equal(
            npix_idx, np.array([[1, 2, 3, 4], [2, 3, 4, 4]])
        )
        expected_chunks = [[25, 8], [17, 16], [9, 24], [1]]
        for got, exp in zip(npix_chunks, expected_chunks):
            np.testing.assert_array_equal(np.asarray(got).ravel(), exp)

        pb = 0
        for i in range(1, 5):
            page_op = page_op.get_page_data(i, npix_chunks)
            cs = int(np.sum(npix_chunks[i - 1]))
            pe = pb + cs
            np.testing.assert_array_equal(page_op.page_data, data[:, pb:pe])
            pb = pe

    def test_get_page_data_4pix_large_even_split_chunks(self):
        n_pix = 100
        rng = np.random.default_rng(102)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, n_pix))
        pix1 = PixelDataMemory(data[:, 0:25])
        pix2 = PixelDataMemory(data[:, 25:50])
        pix3 = PixelDataMemory(data[:, 50:75])
        pix4 = PixelDataMemory(data[:, 75:100])

        page_op = PageOp_cat_pix()
        page_op = page_op.init(pix1, pix2, pix3, pix4)
        page_op.pix = PixelDataFileBacked()

        assert page_op.npix.size == 4
        np.testing.assert_array_equal(page_op.npix, [25, 25, 25, 25])

        npix = page_op.npix
        npix_chunks, npix_idx, page_op = page_op.split_into_pages(npix, 50)
        assert len(npix_chunks) == 2
        assert list(npix_idx.shape) == [2, 2]
        np.testing.assert_array_equal(npix_idx, np.array([[1, 3], [2, 4]]))
        expected_chunks = [[25, 25], [25, 25]]
        for got, exp in zip(npix_chunks, expected_chunks):
            np.testing.assert_array_equal(np.asarray(got).ravel(), exp)

        pb = 0
        for i in range(1, 3):
            page_op = page_op.get_page_data(i, npix_chunks)
            cs = int(np.sum(npix_chunks[i - 1]))
            pe = pb + cs
            np.testing.assert_array_equal(page_op.page_data, data[:, pb:pe])
            pb = pe

    def test_get_page_data_2pix_smaller_odd_split_chunk(self):
        n_pix = 100
        rng = np.random.default_rng(103)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, n_pix))
        pix1 = PixelDataMemory(data[:, 0:50])
        pix2 = PixelDataMemory(data[:, 50:100])

        page_op = PageOp_cat_pix()
        page_op = page_op.init(pix1, pix2)
        page_op.pix = PixelDataFileBacked()

        assert page_op.npix.size == 2
        np.testing.assert_array_equal(page_op.npix, [50, 50])

        npix = page_op.npix
        npix_chunks, npix_idx, page_op = page_op.split_into_pages(npix, 33)
        assert len(npix_chunks) == 4
        assert list(npix_idx.shape) == [2, 4]
        np.testing.assert_array_equal(
            npix_idx, np.array([[1, 1, 2, 2], [1, 2, 2, 2]])
        )
        expected_chunks = [[33], [17, 16], [33], [1]]
        for got, exp in zip(npix_chunks, expected_chunks):
            np.testing.assert_array_equal(np.asarray(got).ravel(), exp)

        pb = 0
        for i in range(1, 5):
            page_op = page_op.get_page_data(i, npix_chunks)
            cs = int(np.sum(npix_chunks[i - 1]))
            pe = pb + cs
            np.testing.assert_array_equal(page_op.page_data, data[:, pb:pe])
            pb = pe

    def test_get_page_data_2pix_smaller_even_split_chunk(self):
        n_pix = 100
        rng = np.random.default_rng(104)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, n_pix))
        pix1 = PixelDataMemory(data[:, 0:50])
        pix2 = PixelDataMemory(data[:, 50:100])

        page_op = PageOp_cat_pix()
        page_op = page_op.init(pix1, pix2)
        page_op.pix = PixelDataFileBacked()

        assert page_op.npix.size == 2
        np.testing.assert_array_equal(page_op.npix, [50, 50])

        npix = page_op.npix
        npix_chunks, npix_idx, page_op = page_op.split_into_pages(npix, 25)
        assert len(npix_chunks) == 4
        assert list(npix_idx.shape) == [2, 4]
        np.testing.assert_array_equal(
            npix_idx, np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
        )
        expected_chunks = [[25], [25], [25], [25]]
        for got, exp in zip(npix_chunks, expected_chunks):
            np.testing.assert_array_equal(np.asarray(got).ravel(), exp)

        for i in range(1, 5):
            page_op = page_op.get_page_data(i, npix_chunks)
            pb = (i - 1) * 25
            np.testing.assert_array_equal(
                page_op.page_data, data[:, pb : pb + 25]
            )

    def test_get_page_data_2pix_eq_chunk(self):
        n_pix = 100
        rng = np.random.default_rng(105)
        data = rng.random((PixelDataBase.DEFAULT_NUM_PIX_FIELDS, n_pix))
        pix1 = PixelDataMemory(data[:, 0:50])
        pix2 = PixelDataMemory(data[:, 50:100])

        page_op = PageOp_cat_pix()
        page_op = page_op.init(pix1, pix2)
        page_op.pix = PixelDataFileBacked()

        assert page_op.npix.size == 2
        np.testing.assert_array_equal(page_op.npix, [50, 50])

        npix = page_op.npix
        npix_chunks, npix_idx, page_op = page_op.split_into_pages(npix, 50)
        assert len(npix_chunks) == 2
        assert list(npix_idx.shape) == [2, 2]
        np.testing.assert_array_equal(npix_idx, np.array([[1, 2], [1, 2]]))
        expected_chunks = [[50], [50]]
        for got, exp in zip(npix_chunks, expected_chunks):
            np.testing.assert_array_equal(np.asarray(got).ravel(), exp)

        page_op = page_op.get_page_data(1, npix_chunks)
        np.testing.assert_array_equal(page_op.page_data, data[:, 0:50])
        page_op = page_op.get_page_data(2, npix_chunks)
        np.testing.assert_array_equal(page_op.page_data, data[:, 50:100])
