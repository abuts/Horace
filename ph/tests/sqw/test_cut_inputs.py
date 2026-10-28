"""Pytest suite for SQW cut input validation (ported from _test/test_sqw_class/test_cut_inputs.m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.sqw import d3d, sqw

_TEST_SQW_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "_test" / "test_sqw"
)
_TEST_FILE = str(_TEST_SQW_DIR / "96dets.par")


class _SqwCutInputsTester(sqw):
    """Subclass exposing process_and_validate_cut_inputs as a public method."""


class TestCutInputs:
    """Port of MATLAB ``test_cut_inputs``."""

    def test_cut_inputs_ndims_no_proj(self) -> None:
        sobj = _SqwCutInputsTester()
        ax = line_axes([-1, 0.02, 1], [-1, 0.02, 1], [-2, 2], [0, 1, 10])
        proj = line_proj([1, 0, 0], [0, 1, 0])
        sobj.data = d3d(ax, proj)

        return_cut = True
        targ_proj, pbin, sym, opt = sobj.process_and_validate_cut_inputs(
            return_cut, [-0.5, 0.5], [-0.5, 0.5], [2, 8], nargout=4
        )
        assert targ_proj == proj
        assert len(pbin) == 1
        assert len(pbin[0]) == 4
        np.testing.assert_allclose(pbin[0][0], [-0.5, 0.5])
        np.testing.assert_allclose(pbin[0][1], [-0.5, 0.5])
        np.testing.assert_allclose(pbin[0][2], [-2.0, 2.0])
        np.testing.assert_allclose(pbin[0][3], [2.0, 8.0])
        assert len(sym) == 1
        assert opt["keep_pix"] is True
        assert opt["parallel"] is False
        assert opt["outfile"] == ""
        assert opt["proj_given"] is False

    def test_cut_inputs_excess_params_throws(self) -> None:
        sobj = _SqwCutInputsTester()
        ax = line_axes([-1, 0.02, 1], [-1, 0.02, 1], [-2, 2], [0, 1, 10])
        proj = line_proj([1, 0, 0], [0, 1, 0])
        sobj.data = d3d(ax, proj)

        with pytest.raises(HoraceError) as exc_info:
            sobj.process_and_validate_cut_inputs(
                True, [-0.5, 0.5], [-0.5, 0.5], [-1, 1], [2, 8]
            )
        assert exc_info.value.error_id == "HORACE:cut:invalid_argument"
        assert "Object has 3 dimensions but cutting with 4 binning parameters" in str(
            exc_info.value
        )

    def test_cut_inputs_with_proj_and_flags(self, tmp_path: Path) -> None:
        sobj = _SqwCutInputsTester()
        ax = line_axes([-1, 0.02, 1], [-1, 0.02, 1], [-2, 2], [0, 1, 10])
        proj = line_proj([1, 0, 0], [0, 1, 0])
        sobj.data = d3d(ax, proj)

        new_proj = line_proj([1, 1, 0], [1, -1, 0])
        outfile = str(tmp_path / "test_cut_out.sqw")

        targ_proj, pbin, sym, opt = sobj.process_and_validate_cut_inputs(
            False,
            new_proj,
            [-0.5, 0.05, 0.5],
            [-0.5, 0.5],
            [-1, 1],
            [2, 1, 8],
            "-nopix",
            outfile,
            nargout=4,
        )
        assert opt["keep_pix"] is False
        assert opt["outfile"] == outfile
        assert opt["proj_given"] is True
        assert len(pbin) == 1
        np.testing.assert_allclose(pbin[0][0], [-0.5, 0.05, 0.5])
        np.testing.assert_allclose(pbin[0][1], [-0.5, 0.5])
        np.testing.assert_allclose(pbin[0][2], [-1.0, 1.0])
        np.testing.assert_allclose(pbin[0][3], [2.0, 1.0, 8.0])

