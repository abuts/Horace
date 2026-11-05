"""Pytest port of ``_test/test_sqw_pageOpMethods/test_compact.m``."""

from __future__ import annotations

import numpy as np

from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.sqw import sqw


def _is_subset(a: np.ndarray, b: np.ndarray) -> bool:
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    return all(bool(np.any(np.isclose(v, b_arr, atol=1e-12, rtol=1e-12))) for v in a_arr)


class TestCompact:
    """Port of ``test_compact.m``."""

    def test_compact_slice_cube_remove_dim(self) -> None:
        data = sqw.generate_cube_sqw(10)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]), [0.2, 0.05, 0.7], [], [], []
        )
        comp = test_data.compact()

        assert int(comp.data.dimensions) < int(test_data.data.dimensions)
        np.testing.assert_array_equal(comp.data.iax, np.array([1]))
        np.testing.assert_array_equal(comp.data.pax, np.array([2, 3, 4]))

        for ind in range(len(comp.data.pax)):
            ind2 = int(comp.data.pax[ind]) - 1
            assert _is_subset(comp.data.p[ind], test_data.data.p[ind2])

        np.testing.assert_array_equal(
            test_data.data.s[test_data.data.s != 0],
            comp.data.s[comp.data.s != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.e[test_data.data.e != 0],
            comp.data.e[comp.data.e != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.npix[test_data.data.npix != 0],
            comp.data.npix[comp.data.npix != 0],
        )

    def test_compact_oversized_cut_3d(self) -> None:
        data = sqw.generate_cube_sqw(2)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]),
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
            [-9.5, 9.5],
        )
        comp = test_data.compact()

        np.testing.assert_array_equal(comp.data.iax, np.array([4]))
        np.testing.assert_array_equal(comp.data.pax, np.array([1, 2, 3]))

        for ind in range(len(comp.data.pax)):
            ind2 = int(comp.data.pax[ind]) - 1
            assert _is_subset(comp.data.p[ind], test_data.data.p[ind2])

        np.testing.assert_array_equal(
            test_data.data.s[test_data.data.s != 0],
            comp.data.s[comp.data.s != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.e[test_data.data.e != 0],
            comp.data.e[comp.data.e != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.npix[test_data.data.npix != 0],
            comp.data.npix[comp.data.npix != 0],
        )

    def test_compact_oversized_cut_2d(self) -> None:
        data = sqw.generate_cube_sqw(2)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]),
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
            [-9.5, 9.5],
            [-9.5, 9.5],
        )
        comp = test_data.compact()

        np.testing.assert_array_equal(comp.data.iax, np.array([3, 4]))
        np.testing.assert_array_equal(comp.data.pax, np.array([1, 2]))

        for ind in range(len(comp.data.pax)):
            ind2 = int(comp.data.pax[ind]) - 1
            assert _is_subset(comp.data.p[ind], test_data.data.p[ind2])

        np.testing.assert_array_equal(
            test_data.data.s[test_data.data.s != 0],
            comp.data.s[comp.data.s != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.e[test_data.data.e != 0],
            comp.data.e[comp.data.e != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.npix[test_data.data.npix != 0],
            comp.data.npix[comp.data.npix != 0],
        )

    def test_compact_oversized_cut_1d(self) -> None:
        data = sqw.generate_cube_sqw(2)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]),
            [-9.5, 1, 9.5],
            [-9.5, 9.5],
            [-9.5, 9.5],
            [-9.5, 9.5],
        )
        comp = test_data.compact()

        np.testing.assert_array_equal(comp.data.iax, np.array([2, 3, 4]))
        np.testing.assert_array_equal(comp.data.pax, np.array([1]))

        for ind in range(len(comp.data.pax)):
            ind2 = int(comp.data.pax[ind]) - 1
            assert _is_subset(comp.data.p[ind], test_data.data.p[ind2])

        np.testing.assert_array_equal(
            test_data.data.s[test_data.data.s != 0],
            comp.data.s[comp.data.s != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.e[test_data.data.e != 0],
            comp.data.e[comp.data.e != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.npix[test_data.data.npix != 0],
            comp.data.npix[comp.data.npix != 0],
        )

    def test_compact_oversized_cut_0d(self) -> None:
        data = sqw.generate_cube_sqw(2)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]),
            [-9.5, 9.5],
            [-9.5, 9.5],
            [-9.5, 9.5],
            [-9.5, 9.5],
        )
        comp = test_data.compact()

        np.testing.assert_array_equal(comp.data.iax, np.array([1, 2, 3, 4]))
        assert len(comp.data.pax) == 0

        np.testing.assert_array_equal(
            test_data.data.s[test_data.data.s != 0],
            comp.data.s[comp.data.s != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.e[test_data.data.e != 0],
            comp.data.e[comp.data.e != 0],
        )
        np.testing.assert_array_equal(
            test_data.data.npix[test_data.data.npix != 0],
            comp.data.npix[comp.data.npix != 0],
        )

    def test_compact_empty_sqw(self) -> None:
        data = sqw()
        comp = data.compact()
        ok, mess = data.equal_to_tol(comp, "-ignore_date")
        assert ok, mess

    def test_compact_not_alter_full_data(self) -> None:
        data = sqw.generate_cube_sqw(10)
        comp = data.compact()
        ok, mess = data.equal_to_tol(comp, "-ignore_date")
        assert ok, mess

    def test_compact_oversized_cut_4d(self) -> None:
        data = sqw.generate_cube_sqw(2)
        test_data = data.cut(
            line_proj([1, 0, 0], [0, 1, 0]),
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
            [-9.5, 1, 9.5],
        )
        comp = test_data.compact()
        ok, mess = data.equal_to_tol(comp, "-ignore_date")
        assert ok, mess
