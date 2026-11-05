"""Unit tests for ``PageOp_join_sqw``, ``sqw.join``, ``collect_sqw_metadata``, and ``write_nsqw_to_sqw``.

Translated from ``_test/test_sqw_pageOpMethods/test_join.m``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyhorace.algorithms.join_sqw import (
    collect_sqw_metadata,
    write_nsqw_to_sqw,
)
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.configurations.hpc_config import hpc_config
from pyhorace.core.serializable import compare_values
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixfile_combine_info import pixfile_combine_info
from pyhorace.data.pixels.pixobj_combine_info import pixobj_combine_info
from pyhorace.sqw import read_sqw, sqw


def _assert_equal_to_tol(a, b, *args, **kwargs):
    if hasattr(a, "equal_to_tol"):
        res = a.equal_to_tol(b, *args, **kwargs, nargout=2)
    else:
        res = compare_values(a, b)
    if isinstance(res, tuple):
        ok, mess = res[0], res[1]
    else:
        ok, mess = bool(res), ""
    assert ok, f"equal_to_tol failed: {mess}"


@pytest.fixture(scope="module")
def join_test_env():
    repo_root = Path(__file__).resolve().parents[4]
    test_dir = repo_root / "_test" / "common_data"
    sample_obj = read_sqw(str(test_dir / "sqw_2d_1.sqw"))

    sqw_to_join = sample_obj.split()
    n_parts = len(sqw_to_join)
    work_dir = tempfile.mkdtemp(prefix="test_join_data_")
    files_to_join = []
    for i in range(n_parts):
        ids = sqw_to_join[i].runid_map.keys
        base_fn = os.path.splitext(os.path.basename(sqw_to_join[i].main_header.filename))[0]
        fn = os.path.join(work_dir, f"{base_fn}_runID{int(ids[0])}")
        files_to_join.append(fn)
        if not os.path.isfile(fn):
            sqw_to_join[i].save(fn)

    yield {
        "test_dir": str(test_dir),
        "work_dir": work_dir,
        "sample_obj": sample_obj,
        "sqw_to_join": sqw_to_join,
        "files_to_join": files_to_join,
    }

    shutil.rmtree(work_dir, ignore_errors=True)


class TestJoin:
    """Test suite translated from ``test_join.m``."""

    def test_split_cube_1_run(self):
        sqw_obj = sqw.generate_cube_sqw(10)

        split_obj = sqw_obj.split()
        reformed_obj = sqw.join(sqw_obj)

        _assert_equal_to_tol(split_obj, reformed_obj)
        _assert_equal_to_tol(sqw_obj, reformed_obj)

    def test_split_cube_2_run(self):
        sqw_obj = sqw.generate_cube_sqw(10)

        run_idx = sqw_obj.pix.run_idx.copy()
        run_idx[7:] = 2
        sqw_obj.pix.run_idx = run_idx

        sqw_obj.experiment_info.expdata[1] = {
            "filename": "fake",
            "filepath": "/fake",
            "efix": 1,
            "emode": 1,
            "cu": [1, 0, 0],
            "cv": [0, 1, 0],
            "psi": 1,
            "omega": 1,
            "dpsi": 1,
            "gl": 1,
            "gs": 1,
            "en": 10,
            "run_id": 1,
        }
        sqw_obj.main_header.nfiles = 2
        rm = sqw_obj.experiment_info.runid_map
        rm = rm.add(2, 2)
        sqw_obj.experiment_info.runid_map = rm

        split_obj = sqw_obj.split()
        assert len(split_obj) == 2

        reformed_obj = sqw.join(split_obj, sqw_obj)

        _assert_equal_to_tol(
            sqw_obj, reformed_obj, "-ignore_str", "-ignore_date"
        )

    def test_join_eq_write_nsqw_to_sqw(self, join_test_env, tmp_path):
        hc = hor_config()
        old_mex = hc.use_mex
        outfile = str(tmp_path / "write_nsqw_to_sqw_test_join.sqw")
        try:
            hc.use_mex = False
            _, _, reformed_obj = write_nsqw_to_sqw(
                join_test_env["files_to_join"], outfile, "-keep", nargout=3
            )
            _assert_equal_to_tol(
                join_test_env["sample_obj"],
                reformed_obj,
                [1e-7, 1e-7],
                ignore_str=True,
                ignore_date=True,
            )
            reformed_obj.deactivate()
        finally:
            hc.use_mex = old_mex
            if os.path.isfile(outfile):
                try:
                    os.remove(outfile)
                except OSError:
                    pass

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-mex): Requires combine_sqw C++ MEX kernel",
        strict=True,
    )
    def test_join_changes_runid_on_files_mex(self, join_test_env):
        hc = hor_config()
        hpc = hpc_config()
        old_mex = hc.use_mex
        old_comb = hpc.combine_sqw_using
        try:
            hc.use_mex = True
            hpc.combine_sqw_using = "mex_code"
            reformed_obj = sqw.join(join_test_env["files_to_join"], "-recalc")
            runid = reformed_obj.runid_map.keys()
            np.testing.assert_array_equal(
                np.asarray(runid, dtype=float),
                np.arange(1, reformed_obj.main_header.nfiles + 1, dtype=float),
            )
        finally:
            hc.use_mex = old_mex
            hpc.combine_sqw_using = old_comb

    @pytest.mark.xfail(
        raises=NotImplementedError,
        reason="TODO(port-mex): Requires combine_sqw C++ MEX kernel",
        strict=True,
    )
    def test_join_works_with_file_list_with_mex(self, join_test_env):
        hc = hor_config()
        hpc = hpc_config()
        old_mex = hc.use_mex
        old_comb = hpc.combine_sqw_using
        try:
            hc.use_mex = True
            hpc.combine_sqw_using = "mex_code"
            reformed_obj = sqw.join(join_test_env["files_to_join"])
            assert join_test_env["sample_obj"].detpar == reformed_obj.detpar
        finally:
            hc.use_mex = old_mex
            hpc.combine_sqw_using = old_comb

    def test_join_works_with_file_list_with_nomex_pages(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        page_size = int(sample_obj.pix.num_pixels // 4)
        hc = hor_config()
        old_mex = hc.use_mex
        old_chunk = hc.mem_chunk_size
        old_fb = hc.fb_scale_factor
        reformed_obj = None
        try:
            hc.use_mex = False
            hc.mem_chunk_size = page_size
            hc.fb_scale_factor = 3

            reformed_obj = sqw.join(join_test_env["files_to_join"])
            assert sample_obj.detpar == reformed_obj.detpar
        finally:
            hc.use_mex = old_mex
            hc.mem_chunk_size = old_chunk
            hc.fb_scale_factor = old_fb

        try:
            _assert_equal_to_tol(
                sample_obj, reformed_obj, [1e-7, 1e-7], ignore_str=True, ignore_date=True
            )
        finally:
            if reformed_obj is not None:
                reformed_obj.delete()

    def test_join_works_with_file_list_with_nomex_and_sample(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        hc = hor_config()
        old_mex = hc.use_mex
        try:
            hc.use_mex = False
            reformed_obj = sqw.join(join_test_env["files_to_join"], sample_obj)
            assert sample_obj.detpar == reformed_obj.detpar
            _assert_equal_to_tol(
                sample_obj, reformed_obj, [1e-7, 1e-7], ignore_str=True, ignore_date=True
            )
        finally:
            hc.use_mex = old_mex

    def test_join_works_with_file_list_with_nomex(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        hc = hor_config()
        old_mex = hc.use_mex
        try:
            hc.use_mex = False
            reformed_obj = sqw.join(join_test_env["files_to_join"])
            assert sample_obj.detpar == reformed_obj.detpar
            reformed_obj.experiment_info.detector_arrays = (
                sample_obj.experiment_info.detector_arrays
            )
            _assert_equal_to_tol(
                sample_obj, reformed_obj, [1e-7, 1e-7], ignore_str=True, ignore_date=True
            )
        finally:
            hc.use_mex = old_mex

    def test_join_creates_tmp_filebacked_on_conditions(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        page_size = int(sample_obj.pix.num_pixels // 4)
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        old_fb = hc.fb_scale_factor
        split_obj = []
        reformed_obj = None
        targ_file = ""
        try:
            hc.mem_chunk_size = page_size
            hc.fb_scale_factor = 3

            files_to_join = join_test_env["files_to_join"]
            for fn in files_to_join:
                w_fb = sqw(fn, file_backed=True)
                assert w_fb.is_filebacked
                split_obj.append(w_fb)

            reformed_obj = sqw.join(split_obj)
            assert reformed_obj.is_filebacked
            targ_file = reformed_obj.full_filename
            assert os.path.isfile(targ_file)
        finally:
            hc.mem_chunk_size = old_chunk
            hc.fb_scale_factor = old_fb
            for w_fb in split_obj:
                w_fb.deactivate()

        _assert_equal_to_tol(
            sample_obj, reformed_obj, [1e-7, 1e-7], ignore_str=True, ignore_date=True
        )
        reformed_obj.delete()
        assert not os.path.isfile(targ_file)

    def test_join_saves_filebacked_on_request(self, join_test_env, tmp_path):
        sample_obj = join_test_env["sample_obj"]
        targ_file = str(tmp_path / "test_join_saves_filebacked.sqw")
        try:
            split_obj = join_test_env["sqw_to_join"]
            reformed_obj = sqw.join(split_obj, targ_file)

            assert reformed_obj.is_filebacked
            assert os.path.isfile(targ_file)
            assert reformed_obj.full_filename == targ_file

            _assert_equal_to_tol(
                sample_obj, reformed_obj, [1e-7, 1e-7], ignore_str=True, ignore_date=True
            )
            reformed_obj.delete()
            assert os.path.isfile(targ_file)
        finally:
            if os.path.isfile(targ_file):
                try:
                    os.remove(targ_file)
                except OSError:
                    pass

    def test_join_changes_runid_on_requests_in_mem_no_sample(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        split_obj = join_test_env["sqw_to_join"]

        assert all(x.main_header.nfiles == 1 for x in split_obj)

        reformed_obj = sqw.join(split_obj, "-recalc_runid")

        runid = reformed_obj.runid_map.keys()
        np.testing.assert_array_equal(
            np.asarray(runid, dtype=float),
            np.arange(1, reformed_obj.main_header.nfiles + 1, dtype=float),
        )

        sobj = sample_obj.copy()
        pix_data = sobj.pix.data.copy()
        idx = PixelDataBase.field_index("run_idx") - 1
        pix_data[idx, :] = pix_data[idx, :] - 18
        sobj.pix.data = pix_data
        sobj.experiment_info.runid_map = np.arange(1, len(split_obj) + 1)

        _assert_equal_to_tol(
            sobj, reformed_obj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
        )

    def test_join_changes_runid_on_requests_in_mem_with_sample(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        split_obj = join_test_env["sqw_to_join"]

        assert all(x.main_header.nfiles == 1 for x in split_obj)

        reformed_obj = sqw.join(split_obj, sample_obj, "-recalc_runid")

        runid = reformed_obj.runid_map.keys
        np.testing.assert_array_equal(
            np.asarray(runid, dtype=float),
            np.arange(1, reformed_obj.main_header.nfiles + 1, dtype=float),
        )

        sobj = sample_obj.copy()
        pix_data = sobj.pix.data.copy()
        idx = PixelDataBase.field_index("run_idx") - 1
        pix_data[idx, :] = pix_data[idx, :] - 18
        sobj.pix.data = pix_data
        sobj.experiment_info.runid_map = np.arange(1, len(split_obj) + 1)

        _assert_equal_to_tol(
            sobj, reformed_obj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
        )

    def test_split_and_join_returns_same_obj_in_mem(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        split_obj = join_test_env["sqw_to_join"]

        assert all(x.main_header.nfiles == 1 for x in split_obj)

        reformed_obj = sqw.join(split_obj)

        _assert_equal_to_tol(
            sample_obj, reformed_obj, [1e-7, 1e-7], "-ignore_str", "-ignore_date"
        )

    def test_collect_metadata_works_on_membased(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        sqw_t = collect_sqw_metadata(join_test_env["sqw_to_join"])
        assert sqw_t.main_header.nfiles == 24

        _assert_equal_to_tol(
            sqw_t.data,
            sample_obj.data,
            [1e-7, 1e-7],
            "-ignore_str",
            "-ignore_date",
        )
        assert isinstance(sqw_t.pix, pixobj_combine_info)

    def test_collect_metadata_works_on_files(self, join_test_env):
        sample_obj = join_test_env["sample_obj"]
        sqw_t = collect_sqw_metadata(join_test_env["files_to_join"])
        assert sqw_t.main_header.nfiles == 24

        _assert_equal_to_tol(
            sqw_t.data,
            sample_obj.data,
            [1e-7, 1e-7],
            "-ignore_str",
            "-ignore_date",
        )
        assert isinstance(sqw_t.pix, pixfile_combine_info)
