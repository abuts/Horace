"""Pytest suite for sqw_eval algorithm and methods (ported from _test/test_algorithms/test_sqw_eval.m)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyhorace.algorithms import sqw_eval
from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.sqw import d2d, read_sqw, sqw

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMON_DATA_DIR = REPO_ROOT / "_test" / "common_data"
ALG_TEST_DIR = REPO_ROOT / "_test" / "test_algorithms"
SQW_2D_FILE_PATH = str(COMMON_DATA_DIR / "sqw_2d_1.sqw")
SQW_2D_REF_FILE_PATH = str(ALG_TEST_DIR / "test_sqw_eval_gauss_ref.sqw")


def _gauss_1d(x: np.ndarray, pars: Any) -> np.ndarray:
    """1D Gaussian matching herbert_core/applications/mfit_funcs/gauss.m."""
    p = np.asarray(pars, dtype=np.float64).ravel()
    return p[0] * np.exp(-0.5 * ((np.asarray(x, dtype=np.float64) - p[1]) / p[2]) ** 2)


def gauss_sqw(
    u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, dE: np.ndarray, pars: Any
) -> np.ndarray:
    """Sum of the Gaussian of each coordinate (matching test_sqw_eval.m)."""
    coords = np.column_stack([u1, u2, u3, dE])
    return np.sum(_gauss_1d(coords, pars), axis=1).reshape(-1, 1)


def linear_func(
    u1: np.ndarray, u2: np.ndarray, u3: np.ndarray, dE: np.ndarray, pars: Any
) -> np.ndarray:
    """Sum of multiple of each coordinate (matching test_sqw_eval.m)."""
    coords = np.column_stack([u1, u2, u3, dE])
    p = np.asarray(pars, dtype=np.float64).reshape(1, -1)
    return np.sum(coords * p, axis=1).reshape(-1, 1)


def build_fake_dnd() -> d2d:
    """Build 2D test dnd object (ported from test_sqw_eval.build_fake_dnd)."""
    fake_dnd = d2d(
        line_axes(nbins_all_dims=[3, 1, 2, 1]),
        line_proj(alatt=[1, 1, 1], angdeg=[90, 90, 90]),
    )
    fake_dnd.s = np.array([[1.0, 0.0, 2.0], [7.0, 1.0, 2.0]], dtype=np.float64).T
    fake_dnd.npix = np.array([[2.0, 0.0, 6.0], [8.0, 3.0, 4.0]], dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        e_arr = np.where(fake_dnd.npix > 0, np.sqrt(fake_dnd.s) / fake_dnd.npix, 0.0)
    fake_dnd.e = e_arr
    fake_dnd.axes.img_range = np.array(
        [[0.5, -1.0, 0.4, -1.0], [3.5, 1.0, 1.2, 1.0]], dtype=np.float64
    )
    return fake_dnd


@pytest.fixture(scope="module")
def sqw_eval_fixtures() -> dict[str, Any]:
    """Load shared sqw_2d_1.sqw and reference test_sqw_eval_gauss_ref.sqw objects."""
    return {
        "sqw_2d_obj": read_sqw(SQW_2D_FILE_PATH),
        "sqw_2d_ref_obj": read_sqw(SQW_2D_REF_FILE_PATH),
        "gauss_params": [10.0, 0.1, 0.05],
        "linear_params": [2.0, 1.0, 1.0, 4.0],
        "sqw_2d_pix_pg_size": int(3e5 // 36),
    }


class TestSqwEval:
    """Port of _test/test_algorithms/test_sqw_eval.m."""

    FLOAT_TOL = 1e-5
    DOUBLE_TOL = 1e-8

    # ------------------------------------------------------------------
    # Argument validation tests
    # ------------------------------------------------------------------
    def test_invalid_argument_error_if_unknown_flag(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        with pytest.raises(HoraceError, match="MATLAB:InputParser:ParamMissingValue"):
            sqw_eval(
                sqw_eval_fixtures["sqw_2d_obj"],
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "-notaflag",
            )

    def test_notEnoughOutputs_error_if_no_ret_value_and_no_outfile(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        with pytest.raises(HoraceError, match="HORACE:sqw_eval:invalid_argument"):
            sqw_eval(
                sqw_eval_fixtures["sqw_2d_obj"],
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                nargout=0,
            )

    def test_notEnoughOutputs_error_if_no_ret_value_and_filebacked(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        with pytest.raises(HoraceError, match="HORACE:sqw_eval:invalid_argument"):
            sqw_eval(
                sqw_eval_fixtures["sqw_2d_obj"],
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "filebacked",
                True,
                nargout=0,
            )

    def test_error_if_num_outfiles_ne_to_num_input_objects(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        with pytest.raises(HoraceError, match="HORACE:sqw:invalid_argument"):
            sqw_eval(
                [w, w],
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "outfile",
                "some_path",
            )

    # ------------------------------------------------------------------
    # SQW object tests
    # ------------------------------------------------------------------
    def test_gauss_on_sqw_object_matches_reference_file(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        out_sqw = sqw_eval(w, gauss_sqw, sqw_eval_fixtures["gauss_params"])
        ok, mess = out_sqw.equal_to_tol(ref, self.FLOAT_TOL, ignore_str=True)
        assert ok, mess

    def test_gauss_on_sqw_with_nopix_matches_reference_dnd(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        out_dnd = sqw_eval(
            w, gauss_sqw, sqw_eval_fixtures["gauss_params"], "-nopix"
        )
        ok, mess = out_dnd.equal_to_tol(ref.data, self.FLOAT_TOL, ignore_str=True)
        assert ok, mess

    def test_gauss_on_array_of_sqw_objects_matches_reference_file(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        sqws_in = DataOpArray([w, w])
        out_sqw = sqw_eval(sqws_in, gauss_sqw, sqw_eval_fixtures["gauss_params"])
        assert isinstance(out_sqw, DataOpArray)
        assert out_sqw.shape == sqws_in.shape
        for i in range(len(sqws_in)):
            ok, mess = out_sqw[i].equal_to_tol(ref, self.FLOAT_TOL, ignore_str=True)
            assert ok, mess

    def test_calling_with_average_flag_sets_each_pix_signal_to_average(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        out_sqw = sqw_eval(
            w, gauss_sqw, sqw_eval_fixtures["gauss_params"], "average", True
        )
        s_flat = out_sqw.data.s.ravel(order="F")
        npix_flat = out_sqw.data.npix.ravel(order="F").astype(np.int64)
        mask = npix_flat != 0
        non_empty_s = s_flat[mask]
        non_empty_npix = npix_flat[mask]
        pix_bin_bounds = np.cumsum(non_empty_npix)
        pix_sig = out_sqw.pix.signal.ravel(order="F")
        for i in range(len(non_empty_s)):
            ave_sig = non_empty_s[i]
            num_pix_in_bin = int(non_empty_npix[i])
            pix_bin_start = int(pix_bin_bounds[i]) - num_pix_in_bin
            np.testing.assert_allclose(
                pix_sig[pix_bin_start : int(pix_bin_bounds[i])],
                ave_sig * np.ones(num_pix_in_bin),
                atol=self.DOUBLE_TOL,
            )
        np.testing.assert_array_equal(
            out_sqw.pix.variance, np.zeros((1, w.pix.num_pixels))
        )
        np.testing.assert_array_equal(
            out_sqw.data.e, np.zeros_like(w.data.npix, dtype=np.float64)
        )

    def test_output_isnot_file_anymore_if_flbckd_true_and_pix_in_memory(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        w = sqw_eval_fixtures["sqw_2d_obj"]
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        out_sqw = sqw_eval(
            w, gauss_sqw, sqw_eval_fixtures["gauss_params"], "filebacked", True
        )
        ok, mess = out_sqw.equal_to_tol(
            ref, self.FLOAT_TOL, "-ignore_str", "-ignore_date"
        )
        assert ok, mess

    # ------------------------------------------------------------------
    # SQW file tests
    # ------------------------------------------------------------------
    def test_gauss_on_sqw_file_matches_reference_file(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        out_sqw = sqw_eval(
            SQW_2D_FILE_PATH, gauss_sqw, sqw_eval_fixtures["gauss_params"]
        )
        ok, mess = out_sqw.equal_to_tol(ref, self.FLOAT_TOL, "-ignore_str")
        assert ok, mess

    def test_gauss_on_sqw_file_with_all_flag_ignores_the_flag(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        out_sqw = sqw_eval(
            SQW_2D_FILE_PATH, gauss_sqw, sqw_eval_fixtures["gauss_params"], "-all"
        )
        ok, mess = out_sqw.equal_to_tol(ref, self.FLOAT_TOL, ignore_str=True)
        assert ok, mess

    def test_gauss_on_cell_of_sqw_files_matches_reference_file(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        ref = sqw_eval_fixtures["sqw_2d_ref_obj"]
        sqws_in = [SQW_2D_FILE_PATH, SQW_2D_FILE_PATH]
        out_sqw = sqw_eval(sqws_in, gauss_sqw, sqw_eval_fixtures["gauss_params"])
        assert len(out_sqw) == len(sqws_in)
        for i in range(len(sqws_in)):
            ok, mess = out_sqw[i].equal_to_tol(ref, self.FLOAT_TOL, ignore_str=True)
            assert ok, mess

    def test_filebacked_pix_matches_reference_object_with_no_mex(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        old_mex = hc.use_mex
        try:
            hc.mem_chunk_size = sqw_eval_fixtures["sqw_2d_pix_pg_size"]
            hc.use_mex = False

            out_sqw = sqw_eval(
                SQW_2D_FILE_PATH,
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "filebacked",
                True,
            )
            assert out_sqw.is_filebacked
            ref_obj = sqw_eval_fixtures["sqw_2d_ref_obj"]
            ok, mess = out_sqw.equal_to_tol(
                ref_obj, self.FLOAT_TOL, "-ignore_str", "-ignore_date"
            )
            assert ok, mess
        finally:
            hc.mem_chunk_size = old_chunk
            hc.use_mex = old_mex

    def test_gauss_on_sqw_w_filebacked_and_ave_equal_to_in_memory(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = sqw_eval_fixtures["sqw_2d_pix_pg_size"]
            fb_out_sqw = sqw_eval(
                SQW_2D_FILE_PATH,
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "average",
                True,
                "filebacked",
                True,
            )
            assert fb_out_sqw.is_filebacked

            ref_out_sqw = sqw_eval(
                sqw_eval_fixtures["sqw_2d_obj"],
                gauss_sqw,
                sqw_eval_fixtures["gauss_params"],
                "average",
                True,
            )
            assert not ref_out_sqw.is_filebacked

            ok, mess = fb_out_sqw.equal_to_tol(
                ref_out_sqw, self.FLOAT_TOL, "-ignore_str", "-ignore_date"
            )
            assert ok, mess
        finally:
            hc.mem_chunk_size = old_chunk

    # ------------------------------------------------------------------
    # DND tests
    # ------------------------------------------------------------------
    def test_func_on_dnd_file_acts_on_signal_and_sets_e_to_zeros(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        fake_dnd = build_fake_dnd()
        dnd_out = sqw_eval(
            fake_dnd, linear_func, sqw_eval_fixtures["linear_params"]
        )
        expected_signal = np.array(
            [[2.6, 0.0, 6.6], [3.0, 5.0, 7.0]], dtype=np.float64
        ).T
        np.testing.assert_allclose(dnd_out.s, expected_signal, atol=self.DOUBLE_TOL)
        np.testing.assert_array_equal(dnd_out.e, np.zeros_like(fake_dnd.npix))

    def test_func_on_dnd_file_acts_on_non_empty_bins_if_all_flag_true(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        fake_dnd = build_fake_dnd()
        dnd_out = sqw_eval(
            fake_dnd, linear_func, sqw_eval_fixtures["linear_params"], "all", True
        )
        expected_signal = np.array(
            [[2.6, 4.6, 6.6], [3.0, 5.0, 7.0]], dtype=np.float64
        ).T
        np.testing.assert_allclose(dnd_out.s, expected_signal, atol=self.DOUBLE_TOL)
        np.testing.assert_array_equal(dnd_out.e, np.zeros_like(fake_dnd.npix))

    def test_all_option_can_be_name_val_pair_of_flag(
        self, sqw_eval_fixtures: dict[str, Any]
    ) -> None:
        fake_dnd = build_fake_dnd()
        dnd_nvp = sqw_eval(
            fake_dnd, linear_func, sqw_eval_fixtures["linear_params"], "all", True
        )
        dnd_flag = sqw_eval(
            fake_dnd, linear_func, sqw_eval_fixtures["linear_params"], "-all"
        )
        ok, mess = dnd_nvp.equal_to_tol(dnd_flag)
        assert ok, mess

