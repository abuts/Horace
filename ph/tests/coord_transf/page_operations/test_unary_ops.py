"""Pytest port of ``_test/test_sqw_pageOpMethods/test_unary_ops.m``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.sqw.d1d import d1d
from pyhorace.sqw.sqw import sqw


def _get_random_data_in_range(
    num_fields: int, num_pix: int, data_range: tuple[float, float], seed: int = 123
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = data_range
    return lo + (hi - lo) * rng.random((num_fields, num_pix))


class TestUnaryOps:
    """Port of ``test_unary_ops.m``."""

    def test_all_functions_are_defined(self) -> None:
        unary_ops = [
            ("acos", (0.0, 1.0)),
            ("acosh", (1.0, 3.0)),
            ("acot", (0.0, 1.0)),
            ("acoth", (10.0, 15.0)),
            ("acsc", (1.0, 3.0)),
            ("acsch", (1.0, 3.0)),
            ("asec", (1.5, 3.0)),
            ("asech", (0.0, 1.0)),
            ("asin", (0.0, 1.0)),
            ("asinh", (1.0, 3.0)),
            ("atan", (0.0, 1.0)),
            ("atanh", (0.0, 0.5)),
            ("cos", (0.0, 1.0)),
            ("cosh", (0.0, 1.0)),
            ("cot", (0.0, 1.0)),
            ("coth", (1.5, 3.0)),
            ("csc", (0.5, 2.5)),
            ("csch", (1.0, 3.0)),
            ("exp", (0.0, 1.0)),
            ("log", (1.0, 3.0)),
            ("log10", (1.0, 3.0)),
            ("sec", (2.0, 4.0)),
            ("sech", (0.0, 1.4)),
            ("sin", (0.0, 3.0)),
            ("sinh", (0.0, 3.0)),
            ("sqrt", (0.0, 3.0)),
            ("tan", (0.0, 1.0)),
            ("tanh", (0.0, 3.0)),
        ]

        num_pix = 7
        for idx, (op_name, data_range) in enumerate(unary_ops):
            data = _get_random_data_in_range(
                PixelDataBase.DEFAULT_NUM_PIX_FIELDS, num_pix, data_range, seed=100 + idx
            )

            sqw_obj = sqw()
            sqw_obj.pix = PixelDataBase.create(data)
            sqw_obj.data.npix = np.array([[num_pix]], dtype=np.float64)
            sqw_obj.data.s = np.array([[-99.0]], dtype=np.float64)
            sqw_obj.data.e = np.array([[99.0]], dtype=np.float64)

            result = getattr(sqw_obj, op_name)()

            assert not np.allclose(result.pix.signal, sqw_obj.pix.signal)
            np.testing.assert_allclose(
                float(result.data.s.flat[0]), float(np.mean(result.pix.signal))
            )
            np.testing.assert_allclose(
                float(result.data.e.flat[0]),
                float(np.sum(result.pix.variance) / (num_pix**2)),
            )

    def test_unary_op_updates_image_signal_and_error_if_no_pixeldata(self) -> None:
        sqw_obj = sqw()
        ax = line_axes(
            img_range=np.array([[-1, -1, -1, -1], [1, 1, 1, 1]], dtype=np.float64),
            nbins_all_dims=[2, 1, 1, 1],
        )
        sqw_obj.data = d1d(ax, line_proj(alatt=3, angdeg=90))
        sqw_obj.pix = PixelDataBase.create()
        sqw_obj.data.s = np.array([[2.0], [21951.0]], dtype=np.float64)
        sqw_obj.data.e = np.array([[1.5], [4123.0]], dtype=np.float64)
        sqw_obj.data.npix = np.array([[1.0], [1.0]], dtype=np.float64)

        result = sqw_obj.log10()

        expected_signal = np.log10(sqw_obj.data.s)
        expected_var = sqw_obj.data.e / (sqw_obj.data.s * np.log(10.0)) ** 2

        np.testing.assert_allclose(result.data.s, expected_signal)
        np.testing.assert_allclose(result.data.e, expected_var)

    def test_unary_op_updates_signal_and_variance_on_file(
        self, tmp_path: Path
    ) -> None:
        num_pix = 2003
        data = _get_random_data_in_range(
            PixelDataBase.DEFAULT_NUM_PIX_FIELDS, num_pix, (1.0, 3.0)
        )

        sqw_obj = sqw()
        sqw_obj.pix = PixelDataBase.create(data)
        sqw_obj.data.npix = np.array([[num_pix]], dtype=np.float64)
        sqw_obj.data.s = np.array([[-99.0]], dtype=np.float64)
        sqw_obj.data.e = np.array([[99.0]], dtype=np.float64)

        test_file = str(tmp_path / "unary_log_op_on_file.sqw")
        sqw_obj.save(test_file)

        hc = hor_config()
        old_chunk = hc.mem_chunk_size
        try:
            hc.mem_chunk_size = 500
            sqw_fb = sqw(test_file, file_backed=True)
            assert sqw_fb.is_filebacked
            result = sqw_fb.log10()
        finally:
            hc.mem_chunk_size = old_chunk

        expected_signal = np.log10(sqw_fb.pix.signal)
        expected_var = sqw_fb.pix.variance / (sqw_fb.pix.signal * np.log(10.0)) ** 2

        eps_s = float(np.finfo(np.float32).eps)
        np.testing.assert_allclose(result.pix.signal, expected_signal, rtol=2 * eps_s, atol=2 * eps_s)
        np.testing.assert_allclose(result.pix.variance, expected_var, rtol=2 * eps_s, atol=2 * eps_s)
        np.testing.assert_allclose(
            float(result.data.s.flat[0]),
            float(np.sum(expected_signal) / num_pix),
            rtol=8 * eps_s,
            atol=8 * eps_s,
        )
        np.testing.assert_allclose(
            float(result.data.e.flat[0]),
            float(np.sum(expected_var) / (num_pix**2)),
            rtol=8 * eps_s,
            atol=8 * eps_s,
        )

    def test_unary_op_updates_signal_and_variance_in_mem(self) -> None:
        num_pix = 23
        data = _get_random_data_in_range(
            PixelDataBase.DEFAULT_NUM_PIX_FIELDS, num_pix, (1.0, 3.0)
        )

        sqw_obj = sqw()
        sqw_obj.pix = PixelDataBase.create(data)
        sqw_obj.data.npix = np.array([[num_pix]], dtype=np.float64)
        sqw_obj.data.s = np.array([[-99.0]], dtype=np.float64)
        sqw_obj.data.e = np.array([[99.0]], dtype=np.float64)

        result = sqw_obj.log10()

        expected_signal = np.log10(sqw_obj.pix.signal)
        expected_var = sqw_obj.pix.variance / (sqw_obj.pix.signal * np.log(10.0)) ** 2

        np.testing.assert_allclose(result.pix.signal, expected_signal)
        np.testing.assert_allclose(result.pix.variance, expected_var)
        np.testing.assert_allclose(
            float(result.data.s.flat[0]), float(np.sum(expected_signal) / num_pix)
        )
        np.testing.assert_allclose(
            float(result.data.e.flat[0]),
            float(np.sum(expected_var) / (num_pix**2)),
        )
