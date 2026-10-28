"""Pytest suite for cut algorithm (ported from _test/test_algorithms/test_cut.m)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from pyhorace.algorithms.cut import (
    _parse_cut_inputs,
    cut,
    cut_dnd,
    cut_sqw,
)
from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.data.pixels import PixelDataBase, PixelDataFileBacked, PixelDataMemory
from pyhorace.io.sqw_formats_factory import sqw_formats_factory
from pyhorace.sqw import d3d, read_dnd, read_sqw, sqw

_TEST_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "_test"
_SQW_FILE = str(_TEST_ROOT / "test_sym_op" / "test_cut_sqw_sym.sqw")
_DND_FILE = str(_TEST_ROOT / "common_data" / "w1d_d1d.sqw")
_REF_CUT_FILE = str(_TEST_ROOT / "test_algorithms" / "test_cut_ref_sqw.sqw")

FLOAT_TOL = 1e-5


def _make_ref_params() -> List[Any]:
    return [
        line_proj(
            [1, -1, 0],
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            "offset",
            [1, 1, 0],
            "type",
            "paa",
        ),
        [-0.1, 0.025, 0.1],
        [-0.1, 0.025, 0.1],
        [-0.1, 0.1],
        [105, 1, 114],
    ]


@pytest.fixture(scope="module")
def sqw_4d() -> sqw:
    return read_sqw(_SQW_FILE)


@pytest.fixture(scope="module")
def ref_sqw() -> sqw:
    return read_sqw(_REF_CUT_FILE)


class TestCutAlgorithm:
    """Port of MATLAB ``test_cut`` from ``_test/test_algorithms/test_cut.m``."""

    def test_cut_preselection_disabled(self, sqw_4d: sqw) -> None:
        ref_par = _make_ref_params()
        sqw_cut_short = cut(sqw_4d, *ref_par)

        ref_par_long = _make_ref_params()
        ref_par_long[0].disable_pix_preselection = True
        sqw_cut_long = cut(sqw_4d, *ref_par_long)

        is_eq, mess = sqw_cut_short.equal_to_tol(
            sqw_cut_long, tol=FLOAT_TOL, ignore_str=True
        )
        assert is_eq, mess

    def test_nrange_is_whole_without_preselection(self, sqw_4d: sqw) -> None:
        ref_par = _make_ref_params()
        ax = line_axes(*ref_par[1:5])
        saxes = sqw_4d.data.axes
        sproj = sqw_4d.data.proj
        data_npix = sqw_4d.data.npix

        proj = ref_par[0].copy_proj_param_from_source(sqw_4d.data)
        block_start, block_size = sproj.get_nrange(data_npix, saxes, ax, proj)
        assert len(block_start) == 260
        assert len(block_size) == len(block_start)
        assert np.sum(block_size) < np.sum(data_npix)

        proj.disable_pix_preselection = True
        block_start_all, block_size_all = sproj.get_nrange(data_npix, saxes, ax, proj)
        assert len(block_start_all) == 1
        assert int(block_start_all[0]) == 1
        assert int(block_size_all[0]) == int(np.sum(data_npix))

    def test_cut_sqw_file_single_chunk(self, ref_sqw: sqw) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 8000
            sqw_cut = cut(_SQW_FILE, *_make_ref_params())
        finally:
            hc.mem_chunk_size = old_chunk

        is_eq, mess = sqw_cut.equal_to_tol(ref_sqw, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_bin_given(self, ref_sqw: sqw) -> None:
        cut_res = cut(ref_sqw, 0.025, 3 * 0.025, 2)
        np.testing.assert_allclose(
            ref_sqw.data.axes.img_range, cut_res.data.axes.img_range, atol=FLOAT_TOL
        )
        np.testing.assert_allclose(
            ref_sqw.data.p[0], cut_res.data.p[0], atol=FLOAT_TOL
        )

    def test_cut_default_ranges(self, ref_sqw: sqw) -> None:
        cut_res = cut(ref_sqw, [-0.05, 0.05], [-0.05, 0.05], [])
        np.testing.assert_allclose(ref_sqw.data.p[2], cut_res.data.p[0], atol=FLOAT_TOL)

    def test_cut_sqw_object(self, sqw_4d: sqw, ref_sqw: sqw) -> None:
        ref_par = _make_ref_params()
        sqw_cut = cut(sqw_4d, *ref_par)

        np.testing.assert_allclose(
            sqw_cut.data.offset[:3], ref_par[0].offset[:3], atol=FLOAT_TOL
        )
        is_eq, mess = sqw_cut.equal_to_tol(ref_sqw, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_sqw_object_mb_fb(self, sqw_4d: sqw) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 10000
            sqw_obj_fb = sqw_4d.copy()
            sqw_obj_fb.pix = PixelDataFileBacked(sqw_4d.pix)
            try:
                sqw_cut_fb = cut(sqw_obj_fb, *_make_ref_params())
            finally:
                sqw_obj_fb.pix.deactivate()

            sqw_obj_mb = sqw_4d.copy()
            sqw_obj_mb.pix = PixelDataMemory(sqw_4d.pix)
            sqw_cut_mb = cut(sqw_obj_mb, *_make_ref_params())
        finally:
            hc.mem_chunk_size = old_chunk

        is_eq, mess = sqw_cut_mb.data.equal_to_tol(
            sqw_cut_fb.data, tol=FLOAT_TOL, ignore_str=True
        )
        assert is_eq, mess

    def test_cut_sqw_with_pix_in_memory_advanced_logging(
        self, sqw_4d: sqw, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hc = hor_config()
        old_level = hc.log_level
        try:
            hc.log_level = 0
            capsys.readouterr()
            cut(sqw_4d, *_make_ref_params())
            log_1 = capsys.readouterr().out

            hc.log_level = 2
            cut(sqw_4d, *_make_ref_params())
            log_2 = capsys.readouterr().out
        finally:
            hc.log_level = old_level

        assert len(log_2) > len(log_1)
        assert "Access speed" not in log_1
        assert "Access speed" in log_2

    def test_cut_sqw_with_pix_advanced_logging(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hc = hor_config()
        old_level = hc.log_level
        try:
            hc.log_level = 0
            capsys.readouterr()
            cut(_SQW_FILE, *_make_ref_params())
            log_1 = capsys.readouterr().out

            hc.log_level = 2
            cut(_SQW_FILE, *_make_ref_params())
            log_2 = capsys.readouterr().out
        finally:
            hc.log_level = old_level

        assert len(log_2) > len(log_1)
        assert "Read speed" not in log_1
        assert "Read speed" in log_2

    def test_cut_sqw_nopix_advanced_logging(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hc = hor_config()
        old_level = hc.log_level
        try:
            hc.log_level = 0
            capsys.readouterr()
            cut(_SQW_FILE, *_make_ref_params(), "-nopix")
            log_1 = capsys.readouterr().out

            hc.log_level = 2
            cut(_SQW_FILE, *_make_ref_params(), "-nopix")
            log_2 = capsys.readouterr().out
        finally:
            hc.log_level = old_level

        assert len(log_2) > len(log_1)
        assert "Read speed" not in log_1
        assert "Read speed" in log_2
        assert "Resulting pix preparation:   0.0%" in log_2

    def test_cut_sqw_nopix(self) -> None:
        sqw_cut = cut(_SQW_FILE, *_make_ref_params(), "-nopix")
        ref_dnd = read_dnd(_REF_CUT_FILE)
        is_eq, mess = sqw_cut.equal_to_tol(ref_dnd, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_sqw_array(self, sqw_4d: sqw) -> None:
        sqw_obj1 = sqw_4d.copy()
        sqw_obj2 = sqw_4d.copy()
        out_sqw = cut(DataOpArray([sqw_obj1, sqw_obj2]), *_make_ref_params())
        assert len(out_sqw) == 2
        is_eq, mess = out_sqw[0].equal_to_tol(
            out_sqw[1], tol=FLOAT_TOL, ignore_str=True
        )
        assert is_eq, mess

    def test_cut_sqw_integrating_multi_axes(self) -> None:
        proj = line_proj([1, -1, 0], [1, 1, 0], "offset", [1, 1, 0], "type", "paa")
        u_axis_lims = [-0.1, 0.025, 0.1]
        v_axis_lims = [-0.1, 0.1]
        w_axis_lims = [-0.1, 0.1]
        en_axis_lims = [105, 1, 114]

        dnd_cut = cut(
            _SQW_FILE,
            proj,
            u_axis_lims,
            v_axis_lims,
            w_axis_lims,
            en_axis_lims,
            "-nopix",
        )
        assert len(dnd_cut.pax) == 2

    def test_cut_sqw_file_to_file(self, tmp_path: Path) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        outfile = str(tmp_path / "cut_sqw_file_to_file_out.sqw")
        try:
            hc.mem_chunk_size = 500
            ret_sqw = cut(_SQW_FILE, *_make_ref_params(), outfile)
        finally:
            hc.mem_chunk_size = old_chunk

        runid = np.unique(ret_sqw.pix.run_idx)
        np.testing.assert_array_equal(
            runid, ret_sqw.experiment_info.expdata.get_run_ids()
        )
        loaded_cut = read_sqw(outfile)
        is_eq, mess = ret_sqw.equal_to_tol(loaded_cut, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_sqw_file_to_sqw_file_combined_nomex(
        self, sqw_4d: sqw, tmp_path: Path
    ) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        outfile = str(tmp_path / "nomex_combine_cut_from_file_to_file.sqw")
        try:
            hc.mem_chunk_size = 500
            cut(_SQW_FILE, *_make_ref_params(), outfile)
        finally:
            hc.mem_chunk_size = old_chunk

        loaded_cut = read_sqw(outfile)
        ref_cut = cut(sqw_4d, *_make_ref_params())
        is_eq, mess = ref_cut.equal_to_tol(loaded_cut, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_sqw_object_to_file(
        self, sqw_4d: sqw, ref_sqw: sqw, tmp_path: Path
    ) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        outfile = str(tmp_path / "test_cut_from_obj_to_file.sqw")
        try:
            hc.mem_chunk_size = 4000
            cut(sqw_4d, *_make_ref_params(), outfile, nargout=0)
        finally:
            hc.mem_chunk_size = old_chunk

        loaded_cut = read_sqw(outfile)
        is_eq, mess = loaded_cut.equal_to_tol(ref_sqw, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_dnd(self) -> None:
        dnd_obj = read_dnd(_SQW_FILE)
        u_axis_lims = [-0.1, 0.024, 0.1]
        v_axis_lims = [-0.1, 0.024, 0.1]
        w_axis_lims = [-0.1, 0.1]
        en_axis_lims = [105, 1, 114]

        res = cut(dnd_obj, u_axis_lims, v_axis_lims, w_axis_lims, en_axis_lims)
        assert isinstance(res, d3d)
        assert res.s.shape == (11, 11, 10)

    def test_cut_fail_outfile_not_created(self) -> None:
        outfile = str(Path("P:/not/a_valid/path.sqw"))
        with pytest.raises(HoraceError) as exc_info:
            cut(_SQW_FILE, *_make_ref_params(), outfile)
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

    def test_out_of_memory_cut_tmp_files_no_mex(
        self, ref_sqw: sqw, tmp_path: Path
    ) -> None:
        num_pixels = 200118
        mem_chunk_size = int(np.floor(num_pixels / 1.9 / 10.0))
        outfile = str(tmp_path / "tmp_outfile.sqw")

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = mem_chunk_size
            assert PixelDataBase.do_filebacked(num_pixels)
            cut(_SQW_FILE, *_make_ref_params(), outfile, nargout=0)
        finally:
            hc.mem_chunk_size = old_chunk

        output_sqw = read_sqw(outfile)
        contrib_ind = np.asarray(output_sqw.runid_map.keys, dtype=int)
        real_contr_ind = np.unique(ref_sqw.pix.run_idx).astype(int)
        assert np.all(np.isin(contrib_ind, real_contr_ind))

        contr_headers = output_sqw.experiment_info.get_subobj(contrib_ind)
        is_eq_hdr, mess_hdr = contr_headers.equal_to_tol(output_sqw.experiment_info)
        assert is_eq_hdr, mess_hdr

        is_eq, mess = output_sqw.equal_to_tol(ref_sqw, tol=FLOAT_TOL, ignore_str=True)
        assert is_eq, mess

    def test_cut_fail_no_outfile_no_nargout(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            cut(_SQW_FILE, *_make_ref_params(), nargout=0)
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

    def test_cut_nopix_to_file(self, tmp_path: Path) -> None:
        outfile = str(tmp_path / "tmp_outfile.sqw")
        cut(_SQW_FILE, *_make_ref_params(), outfile, "-nopix", nargout=0)
        assert Path(outfile).is_file()

        ldr = sqw_formats_factory.instance().get_loader(outfile)
        try:
            output_obj = ldr.get_dnd()
        finally:
            ldr.delete()
        ref_object = read_dnd(_REF_CUT_FILE)
        is_eq, mess = output_obj.equal_to_tol(ref_object, tol=2e-7, ignore_str=True)
        assert is_eq, mess

    def test_cut_fail_dnd_sqw_file(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            cut_dnd(_SQW_FILE, *_make_ref_params())
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

    def test_cut_fail_sqw_dnd_file(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            cut_sqw(_DND_FILE, *_make_ref_params())
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

    def test_cut_fail_no_output_no_file(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            cut(_DND_FILE, *_make_ref_params(), nargout=0)
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

    def test_cut_fail_not_enough_outs(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            _parse_cut_inputs(2, 3, *_make_ref_params())
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"
        assert (
            "Number of input cut sources (2) is smaller than the number of requested outputs (3)"
            in str(exc_info.value)
        )

    def test_cut_fail_not_enough_files(self) -> None:
        inputs = [*_make_ref_params(), ["file_name1.sqw", "file_name2.sqw"]]
        with pytest.raises(HoraceError) as exc_info:
            _parse_cut_inputs(3, 0, *inputs)
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"
        assert "Multiple cuts" in str(exc_info.value)

    def test_cut_fail_no_nargout_no_file(self) -> None:
        with pytest.raises(HoraceError) as exc_info:
            _parse_cut_inputs(1, 0, *_make_ref_params())
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"

