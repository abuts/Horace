"""Pytest suite for symmetry-enabled SQW cuts (ported from _test/test_sym_op/test_cut_sqw_sym.m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.algorithms.cut import cut, cut_dnd, cut_sqw, symmetrise_sqw
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.coord_transf.symop import SymopIdentity, SymopReflection, SymopRotation
from pyhorace.sqw import d1d, read_dnd, read_sqw, sqw

_SYM_OP_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent / "_test" / "test_sym_op"
)
_SQW_FILE = str(_SYM_OP_DIR / "test_cut_sqw_sym.sqw")
_SQW_FILE_2 = str(_SYM_OP_DIR / "test_sym_op.sqw")


@pytest.fixture(scope="module")
def data_4d() -> sqw:
    return read_sqw(_SQW_FILE)


@pytest.fixture(scope="module")
def proj_default() -> line_proj:
    return line_proj([1, -1, 0], [1, 1, 0], "offset", [1, 1, 0], "type", "paa")


class TestCutSqwSym:
    """Port of MATLAB ``test_cut_sqw_sym``."""

    u_axis_lims = [-0.1, 0.025, 0.1]
    v_axis_lims = [-0.1, 0.025, 0.1]
    w_axis_lims = [-0.1, 0.1]
    en_axis_lims = [105, 1, 114]

    def test_cut_sym_identity(self, data_4d: sqw, proj_default: line_proj) -> None:
        sym = SymopIdentity()
        w_sym = cut(
            data_4d,
            proj_default,
            self.u_axis_lims,
            self.v_axis_lims,
            self.w_axis_lims,
            self.en_axis_lims,
            sym,
        )
        w_nosym = cut(
            data_4d,
            proj_default,
            self.u_axis_lims,
            self.v_axis_lims,
            self.w_axis_lims,
            self.en_axis_lims,
        )
        is_eq, mess = w_sym.equal_to_tol(w_nosym, tol=1e-6, ignore_str=True)
        assert is_eq, mess

    def test_cut_sym_reflection_single(self, data_4d: sqw) -> None:
        proj = line_proj([1, 0, 0], [0, 1, 0], "type", "rrr")
        sym = SymopReflection([0, 1, 0], [0, 0, 1], [1, 1, 0])

        w_pos = cut(
            data_4d,
            proj,
            [1.0, 0.05, 1.3],
            [0.9, 1.1],
            [-0.1, 0.1],
            [105, 115],
        )
        w_neg = cut(
            data_4d,
            proj,
            [0.7, 0.05, 1.0],
            [0.9, 1.1],
            [-0.1, 0.1],
            [105, 115],
        )
        w_sym = cut(
            data_4d,
            proj,
            [1.0, 0.05, 1.3],
            [0.9, 1.1],
            [-0.1, 0.1],
            [105, 115],
            sym,
        )

        expected_npix = w_pos.data.npix + w_neg.data.npix[::-1]
        np.testing.assert_allclose(w_sym.data.npix, expected_npix)
        assert w_sym.pix.num_pixels == int(np.sum(expected_npix))

    def test_cut_sym_reflection_nopix(self, data_4d: sqw) -> None:
        proj = line_proj([1, 0, 0], [0, 1, 0], "type", "rrr")
        sym = SymopReflection([0, 1, 0], [0, 0, 1], [1, 1, 0])

        w_sym_pix = cut(
            data_4d,
            proj,
            [1.0, 0.05, 1.3],
            [0.9, 1.1],
            [-0.1, 0.1],
            [105, 115],
            sym,
        )
        w_sym_nopix = cut_sqw(
            data_4d,
            proj,
            [1.0, 0.05, 1.3],
            [0.9, 1.1],
            [-0.1, 0.1],
            [105, 115],
            sym,
            "-nopix",
        )
        assert isinstance(w_sym_nopix, d1d)
        is_eq, mess = w_sym_pix.data.equal_to_tol(
            w_sym_nopix, tol=1e-6, ignore_str=True
        )
        assert is_eq, mess

    def test_cut_sym_rotation_90(self, data_4d: sqw) -> None:
        proj = line_proj([1, 0, 0], [0, 1, 0], "type", "rrr")
        sym = SymopRotation([0, 0, 1], 90, [1, 1, 0])

        w_orig = cut(
            data_4d,
            proj,
            [1.0, 0.05, 1.2],
            [0.95, 1.05],
            [-0.1, 0.1],
            [105, 115],
        )
        w_rot = cut(
            data_4d,
            proj,
            [0.95, 1.05],
            [1.0, 0.05, 1.2],
            [-0.1, 0.1],
            [105, 115],
        )
        w_sym = cut(
            data_4d,
            proj,
            [1.0, 0.05, 1.2],
            [0.95, 1.05],
            [-0.1, 0.1],
            [105, 115],
            sym,
        )

        expected_npix = w_orig.data.npix + w_rot.data.npix
        np.testing.assert_allclose(w_sym.data.npix, expected_npix)
        assert w_sym.pix.num_pixels == int(np.sum(expected_npix))

    def test_cut_sym_reflect_half_to_whole_cut(self, data_4d: sqw) -> None:
        op = SymopReflection([-1, 1, 0], [0, 0, 1], [0, 0, 0])
        proj = line_proj([1, -1, 0], np.array([1, 1, 0]) / np.sqrt(2.0), "offset", [1, 1, 0], "type", "paa")
        ubin_half = [-0.25, 0.05, 0.0]
        width = [-0.15, 0.15]
        ebins = [105, 0, 115]

        wtmp = symmetrise_sqw(data_4d, op)
        w1sym = cut(wtmp, proj, ubin_half, width, width, ebins, "-nopix")
        w2sym = cut(
            data_4d,
            proj,
            ubin_half,
            width,
            width,
            ebins,
            [SymopIdentity(), op],
            "-nopix",
        )

        is_eq, mess = w1sym.equal_to_tol(w2sym, tol=1e-5, ignore_str=True)
        assert is_eq, mess

    def test_cut_sym_from_file(self, data_4d: sqw) -> None:
        op = SymopReflection([-1, 1, 0], [0, 0, 1], [0, 0, 0])
        proj = line_proj([1, -1, 0], np.array([1, 1, 0]) / np.sqrt(2.0), "offset", [1, 1, 0], "type", "paa")
        ubin_half = [-0.25, 0.05, 0.0]
        width = [-0.15, 0.15]
        ebins = [105, 0, 115]

        wtmp = symmetrise_sqw(data_4d, op)
        w1sym = cut(wtmp, proj, ubin_half, width, width, ebins, "-nopix")

        w2sym = cut(
            _SQW_FILE,
            proj,
            ubin_half,
            width,
            width,
            ebins,
            [SymopIdentity(), op],
            "-nopix",
        )
        w3sym = cut_sqw(
            _SQW_FILE,
            proj,
            ubin_half,
            width,
            width,
            ebins,
            [SymopIdentity(), op],
            "-nopix",
        )

        is_eq2, mess2 = w1sym.equal_to_tol(w2sym, tol=1e-5, ignore_str=True)
        assert is_eq2, mess2
        is_eq3, mess3 = w1sym.equal_to_tol(w3sym, tol=1e-5, ignore_str=True)
        assert is_eq3, mess3

