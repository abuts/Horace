"""Unit tests for PixelDataBase, PixelDataMemory, PixelDataFileBacked, PixelData,
pix_metadata, pix_data, pix_write_handle, MultipixBase, pixfile_combine_info,
pixobj_combine_info, and test_pix_load_3_5_0.

Translated from `_test/test_sqw_pixels/test_PixelData.m` and
`_test/test_sqw_pixels/test_pix_load_3_5_0.m`.
"""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
import scipy.io

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.core.serializable import Serializable
from pyhorace.data.pixels import (
    FileBackedPixInterface,
    FileBackedPixStub,
    MultipixBase,
    PixelData,
    PixelDataBase,
    PixelDataFileBacked,
    PixelDataMemory,
    pix_data,
    pix_metadata,
    pix_write_handle,
    pixfile_combine_info,
    pixobj_combine_info,
)
from pyhorace.data.sigvar import sigvar

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
COMMON_DATA_DIR = os.path.join(REPO_ROOT, "_test", "common_data")
SQW_1D_1 = os.path.join(COMMON_DATA_DIR, "sqw_1d_1.sqw")
SQW_2D_1 = os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw")
V3_5_0_MAT = os.path.join(REPO_ROOT, "_test", "test_sqw_pixels", "data", "sqw_v3_5_0_collection.mat")


def _get_random_pix_data(num_pixels: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.random((9, num_pixels), dtype=np.float64)
    # Give integer-like run_idx, detector_idx, energy_idx
    if num_pixels > 0:
        data[4, :] = rng.integers(1, 5, size=num_pixels).astype(np.float64)
        data[5, :] = rng.integers(1, 100, size=num_pixels).astype(np.float64)
        data[6, :] = rng.integers(1, 50, size=num_pixels).astype(np.float64)
    return data


# ==============================================================================
# Construction & Factory Tests
# ==============================================================================
class TestPixelDataConstruction:
    def test_empty_construction(self):
        pix = PixelDataBase.create()
        assert isinstance(pix, PixelDataMemory)
        assert pix.num_pixels == 0
        assert pix.data.shape == (9, 0)
        assert pix.isempty()
        assert np.all(pix.data_range == PixelDataBase.EMPTY_RANGE)

    def test_int_construction_zeros(self):
        n = 15
        pix = PixelDataBase.create(n)
        assert isinstance(pix, PixelDataMemory)
        assert pix.num_pixels == n
        assert pix.data.shape == (9, n)
        np.testing.assert_array_equal(pix.data, np.zeros((9, n)))
        np.testing.assert_array_equal(pix.data_range, np.zeros((2, 9)))

    def test_numpy_array_construction(self):
        raw = _get_random_pix_data(20)
        pix = PixelDataBase.create(raw)
        assert isinstance(pix, PixelDataMemory)
        assert pix.num_pixels == 20
        np.testing.assert_allclose(pix.data, raw)
        expected_range = np.vstack([np.min(raw, axis=1), np.max(raw, axis=1)])
        np.testing.assert_allclose(pix.data_range, expected_range)

    def test_invalid_row_count_raises(self):
        with pytest.raises(HoraceError):
            PixelDataBase.create(np.zeros((8, 10)))
        with pytest.raises(HoraceError):
            PixelDataBase.create(np.zeros((10, 5)))

    def test_copy_construction(self):
        raw = _get_random_pix_data(12)
        pix1 = PixelDataMemory(raw)
        pix2 = PixelDataMemory(pix1)
        assert pix1 == pix2
        pix2.signal = np.ones(12) * 99.0
        assert not np.allclose(pix1.signal, pix2.signal)

    def test_filebacked_from_array_and_file(self, tmp_path):
        raw = _get_random_pix_data(25)
        fb = PixelDataFileBacked(raw)
        assert fb.is_filebacked
        assert fb.num_pixels == 25
        np.testing.assert_allclose(fb.data, raw.astype(np.float32), rtol=1e-6)

        if os.path.isfile(SQW_1D_1):
            fb_sqw = PixelDataBase.create(SQW_1D_1, file_backed=True)
            assert isinstance(fb_sqw, PixelDataFileBacked)
            assert fb_sqw.is_filebacked
            assert fb_sqw.num_pixels == 100337

            mem_sqw = PixelDataBase.create(SQW_1D_1, file_backed=False)
            assert isinstance(mem_sqw, PixelDataMemory)
            assert not mem_sqw.is_filebacked
            assert mem_sqw.num_pixels == 100337
            np.testing.assert_allclose(fb_sqw.data[:, :100], mem_sqw.data[:, :100])

    def test_legacy_pixel_data_wrapper(self):
        raw = _get_random_pix_data(10)
        p = PixelData(raw)
        assert isinstance(p, PixelDataMemory)
        assert p.num_pixels == 10


# ==============================================================================
# Field Getters, Setters, and Slice Assignment Tests
# ==============================================================================
class TestPixelFields:
    @pytest.mark.parametrize(
        "field,rows",
        [
            ("u1", [0]),
            ("u2", [1]),
            ("u3", [2]),
            ("dE", [3]),
            ("coordinates", [0, 1, 2, 3]),
            ("q_coordinates", [0, 1, 2]),
            ("run_idx", [4]),
            ("detector_idx", [5]),
            ("energy_idx", [6]),
            ("signal", [7]),
            ("variance", [8]),
        ],
    )
    def test_field_getters_and_setters(self, field, rows):
        raw = _get_random_pix_data(15)
        pix = PixelDataMemory(raw.copy())
        val = getattr(pix, field)
        assert val.shape == (len(rows), 15)
        np.testing.assert_allclose(val, raw[rows, :])

        # Set to new random values
        new_val = np.ones((len(rows), 15), dtype=np.float64) * 3.5
        setattr(pix, field, new_val)
        np.testing.assert_allclose(getattr(pix, field), new_val)
        np.testing.assert_allclose(pix.data_range[0, rows], 3.5)
        np.testing.assert_allclose(pix.data_range[1, rows], 3.5)

    def test_in_place_slice_assignment_updates_owner(self):
        raw = _get_random_pix_data(10)
        pix = PixelDataMemory(raw.copy())
        pix.signal[:, 2:5] = 42.0
        np.testing.assert_allclose(pix.data[7, 2:5], 42.0)
        assert pix.data_range[1, 7] == pytest.approx(42.0)

        pix.q_coordinates[:, 0] = np.array([-10.0, -20.0, -30.0])
        np.testing.assert_allclose(pix.u1[0, 0], -10.0)
        np.testing.assert_allclose(pix.u2[0, 0], -20.0)
        np.testing.assert_allclose(pix.u3[0, 0], -30.0)
        assert pix.data_range[0, 0] == pytest.approx(-10.0)

    def test_get_and_set_fields(self):
        raw = _get_random_pix_data(12)
        pix = PixelDataMemory(raw.copy())
        sub = pix.get_fields(["u1", "signal", "variance"])
        assert sub.shape == (3, 12)
        np.testing.assert_allclose(sub[0], raw[0])
        np.testing.assert_allclose(sub[1], raw[7])
        np.testing.assert_allclose(sub[2], raw[8])

        with pytest.raises(HoraceError):
            pix.get_fields(["nonexistent_field"])

    def test_pix_range_property(self):
        raw = _get_random_pix_data(20)
        pix = PixelDataMemory(raw)
        assert pix.pix_range.shape == (2, 4)
        np.testing.assert_allclose(pix.pix_range, pix.data_range[:, :4])


# ==============================================================================
# Subset Extraction, Masking, Cat, and Append Tests
# ==============================================================================
class TestPixelSubsetAndCombine:
    @pytest.mark.parametrize("file_backed", [False, True])
    def test_get_pixels_logical_and_numeric(self, file_backed):
        raw = _get_random_pix_data(20)
        pix = PixelDataFileBacked(raw) if file_backed else PixelDataMemory(raw)

        # 1-based numeric indices (including out-of-order and repeated indices)
        idx = [5, 2, 10, 2, 20]
        sub = pix.get_pixels(idx)
        assert sub.num_pixels == 5
        expected = raw[:, [4, 1, 9, 1, 19]]
        np.testing.assert_allclose(sub.data, expected, rtol=1e-6)

        # Boolean mask
        mask = np.zeros(20, dtype=bool)
        mask[[0, 3, 7, 15]] = True
        sub_mask = pix.get_pixels(mask)
        assert sub_mask.num_pixels == 4
        np.testing.assert_allclose(sub_mask.data, raw[:, mask], rtol=1e-6)

    @pytest.mark.parametrize("file_backed", [False, True])
    def test_get_pix_in_ranges(self, file_backed):
        raw = _get_random_pix_data(30)
        pix = PixelDataFileBacked(raw) if file_backed else PixelDataMemory(raw)

        starts = [1, 10, 25]
        block_sizes = [4, 3, 4]
        sub = pix.get_pix_in_ranges(starts, block_sizes)
        expected_idx = [0, 1, 2, 3, 9, 10, 11, 24, 25, 26, 27]
        assert sub.num_pixels == len(expected_idx)
        np.testing.assert_allclose(sub.data, raw[:, expected_idx], rtol=1e-6)

        # Overlapping ranges should also work
        sub_overlap = pix.get_pix_in_ranges([2, 3], [4, 4])
        expected_ov = [1, 2, 3, 4, 2, 3, 4, 5]
        assert sub_overlap.num_pixels == len(expected_ov)
        np.testing.assert_allclose(sub_overlap.data, raw[:, expected_ov], rtol=1e-6)

        # Invalid ranges (negative block_sizes or out of bounds) raise HoraceError
        with pytest.raises(HoraceError):
            pix.get_pix_in_ranges([5, 10], [-1, 3])
        with pytest.raises(HoraceError):
            pix.get_pix_in_ranges([5, 28], [3, 5])

    def test_mask_in_memory_and_paged(self):
        raw = _get_random_pix_data(25)
        pix = PixelDataMemory(raw)
        mask_arr = np.arange(25) % 2 == 0
        masked = pix.mask(mask_arr)
        assert masked.num_pixels == int(np.sum(mask_arr))
        np.testing.assert_allclose(masked.data, raw[:, mask_arr])

        # With npix chunking
        fb = PixelDataFileBacked(raw)
        fb.page_size = 7
        npix = np.array([5, 5, 5, 5, 5])
        keep_bins = np.array([True, False, True, False, True])
        masked_fb = fb.mask(keep_bins, npix=npix)
        expected_pix_mask = np.repeat(keep_bins, 5)
        assert masked_fb.num_pixels == 15
        np.testing.assert_allclose(masked_fb.data, raw[:, expected_pix_mask], rtol=1e-6)

    def test_cat_and_append(self):
        raw1 = _get_random_pix_data(10, seed=1)
        raw2 = _get_random_pix_data(15, seed=2)
        p1 = PixelDataMemory(raw1)
        p2 = PixelDataMemory(raw2)

        combined = PixelDataBase.cat(p1, p2)
        assert combined.num_pixels == 25
        np.testing.assert_allclose(combined.data, np.hstack([raw1, raw2]))
        expected_range = np.vstack(
            [
                np.minimum(p1.data_range[0], p2.data_range[0]),
                np.maximum(p1.data_range[1], p2.data_range[1]),
            ]
        )
        np.testing.assert_allclose(combined.data_range, expected_range)

        # Append in-place
        p1_copy = p1.copy()
        p1_copy.append(p2)
        assert p1_copy.num_pixels == 25
        np.testing.assert_allclose(p1_copy.data, combined.data)


# ==============================================================================
# Paging & Alignment Tests
# ==============================================================================
class TestPagingAndAlignment:
    def test_filebacked_paging_iteration(self):
        raw = _get_random_pix_data(23)
        fb = PixelDataFileBacked(raw)
        fb.page_size = 7
        assert fb.num_pages == 4
        assert fb.page_num == 1
        assert fb.has_more

        collected = []
        while True:
            collected.append(fb.data.copy())
            if not fb.has_more:
                break
            fb.advance()

        assert len(collected) == 4
        assert [c.shape[1] for c in collected] == [7, 7, 7, 2]
        np.testing.assert_allclose(np.hstack(collected), raw, rtol=1e-6)

        fb.reset()
        assert fb.page_num == 1

    def test_alignment_matrix_correction(self):
        raw = _get_random_pix_data(10)
        # 90-degree rotation around z-axis
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        expected_q = rot @ raw[:3, :]

        # File-backed pixels apply alignment lazily (keeping raw data unchanged)
        fb = PixelDataFileBacked(raw.copy())
        assert not fb.is_corrected
        fb.alignment_matr = rot
        assert fb.is_corrected
        np.testing.assert_allclose(fb.q_coordinates, expected_q, rtol=1e-6)
        np.testing.assert_allclose(fb.get_raw_data()[:3, :], raw[:3, :], rtol=1e-6)

        # Reset alignment to identity on file-backed
        fb.alignment_matr = np.eye(3)
        assert not fb.is_corrected
        np.testing.assert_allclose(fb.q_coordinates, raw[:3, :], rtol=1e-6)

        # In-memory pixels apply alignment immediately via alignment_matr, or lazily via set_raw_alignment
        pix = PixelDataMemory(raw.copy())
        assert not pix.is_corrected
        pix.set_raw_alignment(rot)
        assert pix.is_corrected
        np.testing.assert_allclose(pix.q_coordinates, expected_q)
        np.testing.assert_allclose(pix.get_raw_data()[:3, :], raw[:3, :])

        pix_eager = PixelDataMemory(raw.copy())
        pix_eager.alignment_matr = rot
        assert not pix_eager.is_corrected
        np.testing.assert_allclose(pix_eager.q_coordinates, expected_q)


# ==============================================================================
# Serialization, pix_metadata, pix_data, pix_write_handle, Multipix Tests
# ==============================================================================
class TestMetadataAndCombineInfo:
    def test_pix_metadata_and_pix_data_serialization(self):
        raw = _get_random_pix_data(14)
        pix = PixelDataMemory(raw)
        meta = pix.metadata
        assert isinstance(meta, pix_metadata)
        assert meta.npix == 14
        np.testing.assert_allclose(meta.data_range, pix.data_range)

        wrap = pix.data_wrap
        assert isinstance(wrap, pix_data)
        assert wrap.npix == 14

        s = pix.to_struct()
        reconstructed = Serializable.from_struct(s)
        assert isinstance(reconstructed, PixelDataMemory)
        assert reconstructed == pix

    def test_pix_write_handle(self, tmp_path):
        target = str(tmp_path / "target_pix.bin")
        wh = pix_write_handle(target)
        raw1 = _get_random_pix_data(8, seed=10)
        raw2 = _get_random_pix_data(6, seed=11)
        wh.write_pixels(raw1)
        wh.write_pixels(raw2)
        fb = wh.finalize(PixelDataFileBacked)
        assert isinstance(fb, PixelDataFileBacked)
        assert fb.num_pixels == 14
        np.testing.assert_allclose(fb.data, np.hstack([raw1, raw2]), rtol=1e-6)

    def test_pixfile_combine_info(self):
        files = ["file1.sqw", "file2.sqw", "file3.sqw", "file4.sqw"]
        nbins = 100
        npix_each = [10, 20, 30, 40]
        pos_npix = [100, 200, 300, 400]
        pos_pix = [1000, 2000, 3000, 4000]
        pfc = pixfile_combine_info(files, nbins, npix_each, pos_npix, pos_pix)

        assert pfc.nfiles == 4
        assert pfc.num_pixels == 100
        assert pfc.is_filebacked
        np.testing.assert_array_equal(pfc.filenum, [1, 2, 3, 4])

        parts = pfc.split_into_parts(2)
        assert len(parts) == 2
        assert parts[0].nfiles == 2
        assert parts[1].nfiles == 2
        assert parts[0].num_pixels == 30
        assert parts[1].num_pixels == 70

        pfc.trim_nfiles(2)
        assert pfc.nfiles == 2
        assert pfc.num_pixels == 30

    def test_pixobj_combine_info(self):
        p1 = PixelDataMemory(_get_random_pix_data(10, seed=1))
        p2 = PixelDataMemory(_get_random_pix_data(15, seed=2))
        npix1 = np.array([4, 6], dtype=np.uint64)
        npix2 = np.array([7, 8], dtype=np.uint64)

        poc = pixobj_combine_info([p1, p2], [npix1, npix2], "filenum")
        assert poc.nfiles == 2
        assert poc.nbins == 2
        assert poc.num_pixels == 25
        assert not poc.is_filebacked

        pix_page, npix_chunk = poc.get_dataset_page(2, 1, [1, 2])
        assert pix_page.shape == (9, 15)
        np.testing.assert_array_equal(npix_chunk, npix2)
        # Since run_label == 'filenum', run_idx of dataset 2 is 2
        np.testing.assert_allclose(pix_page[4, :], 2.0)

        with pytest.raises(HoraceError):
            poc.npix_each_file = [5, 5]

    @pytest.mark.skipif(not os.path.isfile(V3_5_0_MAT), reason="v3.5.0 test file not found")
    def test_pix_load_3_5_0(self):
        mat = scipy.io.loadmat(V3_5_0_MAT, squeeze_me=True, struct_as_record=False)
        assert len([k for k in mat if not k.startswith("__")]) > 0
