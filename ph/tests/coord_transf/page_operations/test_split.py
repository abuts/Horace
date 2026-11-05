"""Unit tests for sqw.split and PageOp_split_sqw (ported from _test/test_sqw_pageOpMethods/test_split.m)."""

from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import fast_map
from pyhorace.coord_transf.page_operations.page_op_split_sqw import PageOp_split_sqw
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.configurations.hpc_config import hpc_config
from pyhorace.sqw import read_sqw, sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
SQW_SOURCE = str(REPO_ROOT / "_test" / "common_data" / "sqw_4d.sqw")


class PageOp_split_sqw_tester(PageOp_split_sqw):
    """Helper class to test PageOp_split_sqw methods (port of PageOp_split_sqw_tester.m)."""

    def __init__(self) -> None:
        super().__init__()
        self._runid_map = fast_map([101, 102, 103], [1, 2, 3])
        self.img_filebacked = False

    def gen_target_filenames_public(
        self, file_in: str, pix_filebacked: bool
    ) -> "PageOp_split_sqw_tester":
        self.gen_target_filenames(file_in, pix_filebacked)
        return self

    def prepare_split_sqw_public(
        self, in_sqw: sqw, pix_filebacked: bool, img_filebacked: bool
    ) -> "PageOp_split_sqw_tester":
        self.img_filebacked = img_filebacked
        self.prepare_split_sqw(in_sqw, pix_filebacked, img_filebacked)
        return self

    @property
    def targ_files_list(self) -> Dict[int, str]:
        return self._targ_files_map

    @property
    def runid_map(self) -> Any:
        return self._runid_map

    @runid_map.setter
    def runid_map(self, val: Any) -> None:
        self._runid_map = val


@pytest.fixture(scope="module")
def source_sqw4D() -> sqw:
    return read_sqw(SQW_SOURCE)


class TestSplit:
    """Port of test_split.m."""

    def test_split_all_filebacked_mem_constrained_eq_membased(
        self, source_sqw4D: sqw, tmp_path: Path
    ) -> None:
        w_spl_mem = source_sqw4D.split()
        img_size = source_sqw4D.img_size_bytes
        phys_mem_req = img_size * 23 / 3.0

        hpc = hpc_config()
        old_phys_mem = hpc.phys_mem_available
        targ_folder = str(tmp_path / "split_fb_targ")
        try:
            hpc.phys_mem_available = phys_mem_req
            source = sqw(SQW_SOURCE, file_backed=True)
            assert source.is_filebacked

            w_splf = source.split("-files", targ_folder)
            assert len(w_splf) == 23

            tol = [float(8 * np.finfo(np.float32).eps), float(8 * np.finfo(np.float32).eps)]
            for i in range(len(w_splf)):
                assert os.path.isfile(w_splf[i])
                spl_obj = read_sqw(w_splf[i])
                ok, mess = w_spl_mem[i].equal_to_tol(
                    spl_obj, tol, "-ignore_date", ignore_str=True
                )
                assert ok, f"Mismatch at index {i}: {mess}"
        finally:
            hpc.phys_mem_available = old_phys_mem
            shutil.rmtree(targ_folder, ignore_errors=True)

    def test_split_all_filebacked_eq_membased(
        self, source_sqw4D: sqw, tmp_path: Path
    ) -> None:
        w_spl_mem = source_sqw4D.split()
        n_pix = source_sqw4D.npixels

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        targ_folder = str(tmp_path / "split_fb_targ")
        try:
            hc.mem_chunk_size = int(n_pix // 3)
            source = sqw(SQW_SOURCE, file_backed=True)
            assert source.is_filebacked

            w_splf = source.split("-files", targ_folder)
            assert len(w_splf) == 23

            tol = [float(8 * np.finfo(np.float32).eps), float(8 * np.finfo(np.float32).eps)]
            for i in range(len(w_splf)):
                assert os.path.isfile(w_splf[i])
                spl_obj = read_sqw(w_splf[i])
                ok, mess = w_spl_mem[i].equal_to_tol(
                    spl_obj, tol, "-ignore_date", ignore_str=True
                )
                assert ok, f"Mismatch at index {i}: {mess}"
        finally:
            hc.mem_chunk_size = old_chunk
            shutil.rmtree(targ_folder, ignore_errors=True)

    def test_split_all_filebacked_generates_files(
        self, source_sqw4D: sqw
    ) -> None:
        n_pix = source_sqw4D.npixels
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        wkdir = str(hc.working_directory)
        fb = os.path.splitext(os.path.basename(source_sqw4D.data.filename))[0]
        tf = [
            os.path.join(wkdir, f"{fb}_runID{91 + i:07d}.sqw")
            for i in range(1, 24)
        ]
        try:
            hc.mem_chunk_size = int(n_pix // 3)
            source = sqw(SQW_SOURCE, file_backed=True)
            assert source.is_filebacked

            w_splf = source.split("-files")
            assert len(w_splf) == 23

            for i in range(len(w_splf)):
                assert os.path.isfile(w_splf[i])
                assert tf[i] == w_splf[i]
        finally:
            hc.mem_chunk_size = old_chunk
            for f in tf:
                if os.path.isfile(f):
                    try:
                        os.remove(f)
                    except OSError:
                        pass

        for i in range(len(w_splf)):
            assert not os.path.isfile(w_splf[i])

    def test_split_pix_filebacked_permanent_res(
        self, source_sqw4D: sqw, tmp_path: Path
    ) -> None:
        n_pix = source_sqw4D.npixels
        targ_folder = str(tmp_path / "split_fb_targ")
        try:
            w_spl = source_sqw4D.split("-filebacked", targ_folder)
            assert len(w_spl) == 23

            n_split_pix = 0
            for i in range(len(w_spl)):
                assert w_spl[i].is_filebacked
                keys = w_spl[i].runid_map.keys
                assert len(keys) == 1
                rid = np.unique(w_spl[i].pix.run_idx)
                assert float(keys[0]) == float(rid[0])
                assert float(w_spl[i].experiment_info.expdata.run_id) == float(rid[0])
                n_split_pix += w_spl[i].npixels
            assert n_pix == n_split_pix
            assert os.path.isdir(targ_folder)

            files = []
            for i in range(len(w_spl)):
                f_i = w_spl[i].full_filename
                files.append(f_i)
                assert os.path.isfile(f_i)
                ext = os.path.splitext(f_i)[1]
                assert ext == ".sqw"
                w_spl[i].deactivate()

            del w_spl
            gc.collect()
            for f_i in files:
                assert os.path.isfile(f_i)
        finally:
            shutil.rmtree(targ_folder, ignore_errors=True)

    def test_split_filebacked_eq_split_membased(
        self, source_sqw4D: sqw
    ) -> None:
        w_spl_mem = source_sqw4D.split()
        n_pix = source_sqw4D.npixels
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        w_spl_fb = None
        try:
            hc.mem_chunk_size = int(n_pix // 3)
            source = sqw(SQW_SOURCE, file_backed=True)
            assert source.is_filebacked
            w_spl_fb = source.split("-filebacked")
            assert len(w_spl_mem) == 23
        finally:
            hc.mem_chunk_size = old_chunk

        tol = [float(8 * np.finfo(np.float32).eps), float(8 * np.finfo(np.float32).eps)]
        for i in range(len(w_spl_mem)):
            ok, mess = w_spl_mem[i].equal_to_tol(
                w_spl_fb[i], tol, "-ignore_str", "-ignore_date"
            )
            assert ok, f"Mismatch at index {i}: {mess}"

    def test_split_pix_filebacked(self, source_sqw4D: sqw) -> None:
        n_pix = source_sqw4D.npixels
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = int(n_pix // 3)
            source = sqw(SQW_SOURCE, file_backed=True)
            assert source.is_filebacked

            w_spl = source.split("-filebacked")
            assert len(w_spl) == 23

            n_split_pix = 0
            for i in range(len(w_spl)):
                assert w_spl[i].is_filebacked
                keys = w_spl[i].runid_map.keys
                assert len(keys) == 1
                rid = np.unique(w_spl[i].pix.run_idx)
                assert float(keys[0]) == float(rid[0])
                assert float(w_spl[i].experiment_info.expdata.run_id) == float(rid[0])
                n_split_pix += w_spl[i].npixels
            assert n_pix == n_split_pix

            files = []
            for i in range(len(w_spl)):
                f_i = w_spl[i].pix.full_filename
                files.append(f_i)
                assert os.path.isfile(f_i)

            del w_spl
            gc.collect()
            for f_i in files:
                assert not os.path.isfile(f_i)
        finally:
            hc.mem_chunk_size = old_chunk

    def test_split_all_in_memory(self, source_sqw4D: sqw) -> None:
        n_pix = source_sqw4D.npixels
        w_spl = source_sqw4D.split()
        assert len(w_spl) == 23

        n_split_pix = 0
        for i in range(len(w_spl)):
            keys = w_spl[i].runid_map.keys
            assert len(keys) == 1
            rid = np.unique(w_spl[i].pix.run_idx)
            assert float(keys[0]) == float(rid[0])
            assert float(w_spl[i].experiment_info.expdata.run_id) == float(rid[0])
            n_split_pix += w_spl[i].npixels
        assert n_pix == n_split_pix

    def test_prepare_split_sqw_in_mem(self, source_sqw4D: sqw) -> None:
        page_op = PageOp_split_sqw_tester()
        page_op = page_op.prepare_split_sqw_public(source_sqw4D, False, False)
        n_runs = int(source_sqw4D.main_header.nfiles)

        assert len(page_op.out_img) == n_runs
        assert len(page_op.out_pix) == n_runs
        assert len(page_op.write_handles) == n_runs
        assert all(wh is None for wh in page_op.write_handles)

    def test_target_filenames_img_filebacked_with_folder_are_sqw(self) -> None:
        tc = PageOp_split_sqw_tester()
        tc.img_filebacked = True
        tc.outfile = "some_folder_to_place_files"
        tc = tc.gen_target_filenames_public("My_sqw_file", True)

        assert not tc.results_are_tmp_files
        file = tc.targ_files_list[101]
        fp = os.path.dirname(file)
        fn, fe = os.path.splitext(os.path.basename(file))
        assert fp == "some_folder_to_place_files"
        assert fn == "My_sqw_file_runID0000101"
        assert fe == ".sqw"

    def test_target_filenames_img_filebacked_are_sqw(self) -> None:
        tc = PageOp_split_sqw_tester()
        tc.img_filebacked = True
        tc = tc.gen_target_filenames_public("My_sqw_file", True)

        assert not tc.results_are_tmp_files
        wk_dir = str(hor_config().working_directory).rstrip("/\\")
        file = tc.targ_files_list[101]
        fp = os.path.dirname(file)
        fn, fe = os.path.splitext(os.path.basename(file))
        assert fp == wk_dir
        assert fn == "My_sqw_file_runID0000101"
        assert fe == ".sqw"

    def test_target_filenames_pix_filebacked_with_folder_are_sqw(self) -> None:
        tc = PageOp_split_sqw_tester()
        tc.outfile = "some_folder_to_place_files"
        tc = tc.gen_target_filenames_public("My_sqw_file", True)

        assert not tc.results_are_tmp_files
        file = tc.targ_files_list[101]
        fp = os.path.dirname(file)
        fn, fe = os.path.splitext(os.path.basename(file))
        assert fp == "some_folder_to_place_files"
        assert fn == "My_sqw_file_runID0000101"
        assert fe == ".sqw"

    def test_target_filenames_pix_filebacked_are_tmp(self) -> None:
        tc = PageOp_split_sqw_tester()
        tc = tc.gen_target_filenames_public("My_sqw_file", True)

        wk_dir = str(hor_config().working_directory).rstrip("/\\")
        assert tc.results_are_tmp_files
        file = tc.targ_files_list[101]
        fp = os.path.dirname(file)
        fn, fe = os.path.splitext(os.path.basename(file))
        assert fp == wk_dir
        assert fn == "My_sqw_file_runID0000101"
        assert fe.startswith(".tmp_")

    def test_target_filenames_all_memory(self) -> None:
        tc = PageOp_split_sqw_tester()
        tc = tc.gen_target_filenames_public("My_sqw_file", False)

        wk_dir = str(hor_config().working_directory).rstrip("/\\")
        assert tc.results_are_tmp_files
        file = tc.targ_files_list[101]
        fp = os.path.dirname(file)
        fn, fe = os.path.splitext(os.path.basename(file))
        assert fp == wk_dir
        assert fn == "My_sqw_file_runID0000101"
        assert fe.startswith(".tmp")
