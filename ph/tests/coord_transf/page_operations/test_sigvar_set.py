"""Tests for ``sqw.sigvar_set`` and ``PageOp_sigvar_set``.

Translated from ``_test/test_sqw_pageOpMethods/test_sigvar_set.m``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw.d1d import d1d
from pyhorace.sqw.d2d import d2d
from pyhorace.sqw.sqw import sqw


class TestSigvarSet:
    """Port of ``test_sigvar_set.m``."""

    def test_sigvar_set_raises_error_if_s_not_same_size_as_dnd_object(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d2d(
            line_axes(
                "nbins_all_dims",
                [1, 3, 5, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros((3, 5), dtype=np.float64, order="F")
        sqw_obj.data.e = np.zeros((3, 5), dtype=np.float64, order="F")

        sigvar_obj = sigvar({"s": np.array([[1, 2, 3]]), "e": np.array([[44, 55, 66]])})

        with pytest.raises(HoraceError) as exc_info:
            sqw_obj.sigvar_set(sigvar_obj)
        assert exc_info.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "size" in str(exc_info.value)

    def test_sigvar_set_raises_error_if_e_not_same_size_as_dnd_object(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d2d(
            line_axes(
                "nbins_all_dims",
                [3, 5, 1, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros((3, 5), dtype=np.float64, order="F")
        sqw_obj.data.e = np.zeros((3, 5), dtype=np.float64, order="F")
        sigvar_obj = sigvar({"s": np.array([[1, 2, 3]]), "e": np.array([[44, 55, 66]])})

        with pytest.raises(HoraceError) as exc_info:
            sqw_obj.sigvar_set(sigvar_obj)
        assert exc_info.value.error_id == "HORACE:DnDBase:invalid_argument"
        assert "size" in str(exc_info.value)

    def test_sigvar_set_updates_s_and_e_values_with_pix(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d2d(
            line_axes(
                "nbins_all_dims",
                [2, 3, 1, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros((2, 3), dtype=np.float64, order="F")
        sqw_obj.data.e = np.zeros((2, 3), dtype=np.float64, order="F")
        sqw_obj.data.npix = 10.0 * np.ones((2, 3), dtype=np.float64, order="F")
        sqw_obj.pix = PixelDataFileBacked(np.ones((9, 60), dtype=np.float64, order="F"))

        sigvar_obj = sigvar(
            {
                "s": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order="F"),
                "e": np.array([[44, 55, 66], [77, 88, 99]], dtype=np.float64, order="F"),
            }
        )

        result = sqw_obj.sigvar_set(sigvar_obj)
        ss = result.sigvar()
        assert np.array_equal(ss.s, sigvar_obj.s)
        assert np.array_equal(ss.e, sigvar_obj.e)

        np.testing.assert_allclose(result.data.s, sigvar_obj.s)
        np.testing.assert_allclose(result.data.e, sigvar_obj.e)
        np.testing.assert_allclose(
            np.asarray(result.pix.signal).ravel()[:10], np.ones(10, dtype=np.float64)
        )
        np.testing.assert_allclose(
            np.asarray(result.pix.signal).ravel()[50:60],
            6.0 * np.ones(10, dtype=np.float64),
        )

        sobj_test = result.recompute_bin_data()
        ok, mess = result.equal_to_tol(sobj_test)
        assert ok, mess

    def test_sigvar_set_updates_s_and_e_values(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d2d(
            line_axes(
                "nbins_all_dims",
                [2, 3, 1, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros((2, 3), dtype=np.float64, order="F")
        sqw_obj.data.e = np.zeros((2, 3), dtype=np.float64, order="F")
        sqw_obj.data.npix = np.ones((2, 3), dtype=np.float64, order="F")

        sigvar_obj = sigvar(
            {
                "s": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64, order="F"),
                "e": np.array([[44, 55, 66], [77, 88, 99]], dtype=np.float64, order="F"),
            }
        )

        result = sqw_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(result.data.s, sigvar_obj.s)
        np.testing.assert_allclose(result.data.e, sigvar_obj.e)

    def test_sigvar_set_sets_pixel_data_as_npix_replica_of_image(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d1d(
            line_axes(
                "nbins_all_dims",
                [1, 3, 1, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros(3, dtype=np.float64)
        sqw_obj.data.e = np.zeros(3, dtype=np.float64)
        sqw_obj.data.npix = np.array([3, 5, 1], dtype=np.float64)
        sqw_obj.pix = PixelDataBase.create(9)

        sigvar_obj = sigvar(
            {
                "s": np.array([1, 2, 3], dtype=np.float64),
                "e": np.array([44, 55, 66], dtype=np.float64),
            }
        )

        expected_signal = np.array([1, 1, 1, 2, 2, 2, 2, 2, 3], dtype=np.float64)
        expected_variance = np.array(
            [3 * 44, 3 * 44, 3 * 44, 5 * 55, 5 * 55, 5 * 55, 5 * 55, 5 * 55, 66],
            dtype=np.float64,
        )

        result = sqw_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(
            np.asarray(result.pix.signal).ravel(), expected_signal
        )
        np.testing.assert_allclose(
            np.asarray(result.pix.variance).ravel(), expected_variance
        )

    def test_sigvar_set_zero_s_and_e_where_npix_zero(self) -> None:
        sqw_obj = sqw()
        sqw_obj.data = d1d(
            line_axes(
                "nbins_all_dims",
                [1, 3, 1, 1],
                "img_range",
                np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            ),
            line_proj("alatt", 3, "angdeg", 90),
        )
        sqw_obj.data.s = np.zeros(3, dtype=np.float64)
        sqw_obj.data.e = np.zeros(3, dtype=np.float64)
        sqw_obj.data.npix = np.array([3, 0, 1], dtype=np.float64)
        sqw_obj.pix = PixelDataBase.create(4)

        sigvar_obj = sigvar(
            {
                "s": np.array([1, 2, 3], dtype=np.float64),
                "e": np.array([44, 55, 66], dtype=np.float64),
            }
        )

        expected_signal = np.array([1, 1, 1, 3], dtype=np.float64)
        expected_variance = np.array([3 * 44, 3 * 44, 3 * 44, 66], dtype=np.float64)
        expected_img_s = np.array([1, 0, 3], dtype=np.float64)
        expected_img_e = np.array([44, 0, 66], dtype=np.float64)

        result = sqw_obj.sigvar_set(sigvar_obj)
        np.testing.assert_allclose(
            np.asarray(result.data.s).ravel(), expected_img_s
        )
        np.testing.assert_allclose(
            np.asarray(result.data.e).ravel(), expected_img_e
        )
        np.testing.assert_allclose(
            np.asarray(result.pix.signal).ravel(), expected_signal
        )
        np.testing.assert_allclose(
            np.asarray(result.pix.variance).ravel(), expected_variance
        )
