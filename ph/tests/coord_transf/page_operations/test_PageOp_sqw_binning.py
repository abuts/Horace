"""Unit tests for ``PageOp_sqw_binning`` and ``sqw_op_bin_pixels``.

Translated from ``_test/test_sqw_pageOpMethods/test_PageOp_sqw_binning.m``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

from pyhorace.coord_transf.page_operations.page_op_sqw_binning import (
    PageOp_sqw_binning,
)
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.serializable import compare_values
from pyhorace.data.pixels.pixobj_combine_info import pixobj_combine_info
from pyhorace.sqw import read_sqw, sqw, sqw_op_bin_pixels


def _assert_equal_to_tol(a: Any, b: Any, *args: Any, **kwargs: Any) -> None:
    if hasattr(a, "equal_to_tol"):
        res = a.equal_to_tol(b, *args, **kwargs, nargout=2)
    else:
        res = compare_values(a, b)
    if isinstance(res, tuple):
        ok, mess = res[0], res[1]
    else:
        ok, mess = bool(res), ""
    assert ok, f"equal_to_tol failed: {mess}"


def _null_op(pop: Any, *args: Any) -> np.ndarray:
    """No-op page operation returning ``pop.page_data`` unchanged."""
    _ = args
    return pop.page_data


def _get_test_multipix_ds(
    n_datasets: int, n_pixels: int, chunk_size: int
) -> Tuple[sqw, np.ndarray, List[Any], np.ndarray]:
    """Build a test multi-pixel dataset to check ``PageOp_sqw_binning`` paging."""
    all_ds: List[sqw] = []
    rng = np.random.default_rng(42)
    var = rng.random(n_datasets)
    npix_list: List[int] = []
    pix_list: List[Any] = []
    chunks_list: List[np.ndarray] = []

    for i in range(n_datasets):
        n_pix = int(np.floor(n_pixels * (1.0 + float(var[i]))))
        ds = sqw.generate_cube_sqw(n_pix)
        ds.data.do_check_combo_arg = False
        ds.data.axes.img_range = np.array(
            [[-1.0, -2.0, -3.0, -5.0], [1.0, 2.0, 3.0, 10.0]], dtype=np.float64
        )
        ds.data.npix = np.array([[ds.pix.num_pixels]], dtype=np.float64)
        ds.data.s = np.array([[1.0]], dtype=np.float64)
        ds.data.e = np.array([[1.0]], dtype=np.float64)
        all_ds.append(ds)
        pix_list.append(ds.pix)
        npix_i = int(ds.pix.num_pixels)
        npix_list.append(npix_i)

        nl_chunks = npix_i // chunk_size
        if nl_chunks * chunk_size < npix_i:
            nl_chunks += 1
            ch = np.ones(nl_chunks, dtype=np.int64) * chunk_size
            ch[-1] = npix_i - (nl_chunks - 1) * chunk_size
        else:
            ch = np.ones(nl_chunks, dtype=np.int64) * chunk_size
        chunks_list.append(ch)

    chunks = np.concatenate(chunks_list)
    pix_cobj = pixobj_combine_info(pix_list, npix_list)
    multipix_ds = all_ds[0]
    npix = np.asarray(npix_list, dtype=np.int64)
    multipix_ds.data.npix = npix.reshape(1, -1)
    multipix_ds.data.s = np.ones((1, n_datasets), dtype=np.float64)
    multipix_ds.data.e = np.ones((1, n_datasets), dtype=np.float64)
    multipix_ds._do_check_combo_arg = False
    multipix_ds.pix = pix_cobj
    multipix_ds._do_check_combo_arg = True
    return multipix_ds, npix, pix_list, chunks


@pytest.fixture(scope="module")
def binning_test_env():
    repo_root = Path(__file__).resolve().parents[4]
    test_dir = repo_root / "_test" / "common_data"
    sample_obj = read_sqw(str(test_dir / "sqw_2d_1.sqw"))

    sqw_to_join = sample_obj.split()
    n_parts = len(sqw_to_join)
    work_dir = tempfile.mkdtemp(prefix="test_binning_join_data_")
    files_to_join = []
    for i in range(n_parts):
        ids = sqw_to_join[i].runid_map.keys
        base_fn = os.path.splitext(
            os.path.basename(sqw_to_join[i].main_header.filename)
        )[0]
        fn = os.path.join(work_dir, f"{base_fn}_runID{int(ids[0])}.sqw")
        files_to_join.append(fn)
        if not os.path.isfile(fn):
            sqw_to_join[i].save(fn)

    ref_mobj = sqw_op_bin_pixels(sqw_to_join, _null_op, [], "-combine")

    yield {
        "test_dir": str(test_dir),
        "work_dir": work_dir,
        "sample_obj": sample_obj,
        "sqw_to_join": sqw_to_join,
        "files_to_join": files_to_join,
        "ref_mobj": ref_mobj,
    }

    shutil.rmtree(work_dir, ignore_errors=True)


class TestPageOpSqwBinning:
    """Test suite translated from ``test_PageOp_sqw_binning.m``."""

    def test_op_on_objects_equal_to_combine_in_memory_with_par_and_file(
        self, binning_test_env
    ):
        sample_obj = binning_test_env["sample_obj"]
        sqw_to_join = binning_test_env["sqw_to_join"]
        ref_mobj = binning_test_env["ref_mobj"]

        proj = sample_obj.data.proj
        bins = sample_obj.data.axes.get_cut_range("-full_range")

        mobj = None
        try:
            mobj = sqw_op_bin_pixels(
                sqw_to_join,
                _null_op,
                [],
                proj,
                *bins,
                "-combine",
                "outfile",
                "MyTestFile.sqw",
            )
            assert mobj.full_filename == "MyTestFile.sqw"
            _assert_equal_to_tol(
                mobj, ref_mobj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
            )
        finally:
            if mobj is not None:
                mobj.deactivate()
            if os.path.isfile("MyTestFile.sqw"):
                os.remove("MyTestFile.sqw")

    def test_op_on_objects_equal_to_combine_in_memory_with_par(
        self, binning_test_env
    ):
        sample_obj = binning_test_env["sample_obj"]
        sqw_to_join = binning_test_env["sqw_to_join"]
        ref_mobj = binning_test_env["ref_mobj"]

        proj = sample_obj.data.proj
        bins = sample_obj.data.axes.get_cut_range("-full_range")

        mobj = sqw_op_bin_pixels(
            sqw_to_join,
            _null_op,
            [],
            proj,
            *bins,
            "-combine",
        )
        _assert_equal_to_tol(
            mobj, ref_mobj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
        )

    def test_op_on_objects_equal_to_combine_on_file_without_par(
        self, binning_test_env
    ):
        files_to_join = binning_test_env["files_to_join"]
        ref_mobj = binning_test_env["ref_mobj"]

        hc = hor_config()
        old_mcs = hc.mem_chunk_size
        old_fbs = hc.fb_scale_factor
        old_mex = hc.use_mex
        mobj = None
        try:
            hc.mem_chunk_size = 200
            hc.fb_scale_factor = 3
            hc.use_mex = False
            mobj = sqw_op_bin_pixels(
                files_to_join, _null_op, [], "-combine"
            )
            _assert_equal_to_tol(
                mobj, ref_mobj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
            )
        finally:
            if mobj is not None:
                mobj.deactivate()
            hc.mem_chunk_size = old_mcs
            hc.fb_scale_factor = old_fbs
            hc.use_mex = old_mex

    def test_op_on_objects_equal_to_combine_in_memory_without_par(
        self, binning_test_env
    ):
        sqw_to_join = binning_test_env["sqw_to_join"]
        sample_obj = binning_test_env["sample_obj"]

        mobj = sqw_op_bin_pixels(sqw_to_join, _null_op, [], "-combine")
        ref_cut = sample_obj.cut(
            sample_obj.data.proj,
            *sample_obj.data.axes.get_cut_range("-full_range"),
        )
        _assert_equal_to_tol(
            mobj,
            ref_cut,
            [1e-7, 1e-7],
            "-ignore_str",
            "-ignore_date",
        )
        assert mobj.pix.num_pixels == sample_obj.pix.num_pixels

    def test_split_multipix_data_get_multipix_pages(self):
        chunk_size = 9000
        tds, npix, pix_data, chunks = _get_test_multipix_ds(4, 10, chunk_size)
        pop_obj = PageOp_sqw_binning()
        pop_obj = pop_obj.init(
            tds,
            _null_op,
            [],
            tds.data.axes,
            tds.data.proj,
            {"combine": True, "nopix": False},
        )

        npix_chunks, npix_idx, pop_obj = pop_obj.split_into_pages(
            tds.data.npix, chunk_size
        )

        assert sum(int(np.sum(c)) for c in npix_chunks) == int(np.sum(npix))
        assert npix_idx.shape[1] == chunks.size
        for i in range(1, 5):
            this_ds = npix_idx[0, :] == i
            these_chunks = np.concatenate(
                [np.asarray(npix_chunks[k]).ravel() for k in np.where(this_ds)[0]]
            )
            ref_chunks = chunks[this_ds]
            np.testing.assert_allclose(these_chunks, ref_chunks)

        all_idx = npix_idx[0, :]
        ds_num = int(all_idx[0])
        num_pixels = pix_data[ds_num - 1].num_pixels
        pix_idx0 = 0
        for i in range(1, len(npix_chunks) + 1):
            pop_obj = pop_obj.get_page_data(i, npix_chunks)
            if ds_num != int(all_idx[i - 1]):
                ds_num = int(all_idx[i - 1])
                num_pixels = pix_data[ds_num - 1].num_pixels
                pix_idx0 = 0
            block_size = min(chunk_size, num_pixels - pix_idx0)
            pix_idx1 = pix_idx0 + block_size

            expected = pix_data[ds_num - 1].get_pixels(
                np.arange(pix_idx0 + 1, pix_idx1 + 1, dtype=np.int64),
                "-raw",
                "-align",
            )
            np.testing.assert_array_equal(pop_obj.page_data, expected)
            pix_idx0 = pix_idx1
