"""Unit tests for ``sqw.copy``.

Translated from ``_test/test_sqw_pageOpMethods/test_sqw_copy.m``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyhorace.core.serializable import compare_values
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.sqw import sqw


@pytest.fixture(scope="module")
def sqw_1d_path() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    return str(repo_root / "_test" / "common_data" / "sqw_1d_1.sqw")


class TestSqwCopy:
    """Test suite translated from ``test_sqw_copy.m``."""

    def test_copy_returns_object_with_identical_data(self, sqw_1d_path: str):
        sqw_obj = sqw(sqw_1d_path)
        sqw_copy = sqw_obj.copy()

        ok, mess = sqw_copy.equal_to_tol(sqw_obj, nargout=2)
        assert ok, mess

    def test_copy_returns_distinct_object(self, sqw_1d_path: str):
        sqw_obj = sqw(sqw_1d_path)
        sqw_copy = sqw(sqw_obj)

        sqw_copy.main_header.title = "test_copy"

        detector = sqw_copy.experiment_info.detector_arrays[0]
        dtp = detector.get_detpar_representation()
        dtp["azim"][:10] = 0
        detector = IX_detector_array(dtp)
        sqw_copy.experiment_info.detector_arrays[0] = detector

        sqw_copy.pix.signal = 1
        sqw_copy.data.s.flat[0] = 100

        assert not compare_values(sqw_copy.main_header, sqw_obj.main_header)[0]
        assert not compare_values(sqw_copy.detpar, sqw_obj.detpar)[0]
        assert not compare_values(sqw_copy.data, sqw_obj.data)[0]
        assert not compare_values(sqw_copy.pix, sqw_obj.pix)[0]

    def test_copy_excluding_pix_returns_empty_pix_data(self, sqw_1d_path: str):
        sqw_obj = sqw(sqw_1d_path)
        sqw_copy = sqw_obj.copy("exclude_pix", True)

        assert compare_values(sqw_copy.pix, PixelDataBase.create())[0]
        assert sqw_copy.main_header.title == sqw_obj.main_header.title
        assert compare_values(sqw_copy.experiment_info, sqw_obj.experiment_info)[0]
        assert compare_values(sqw_copy.detpar, sqw_obj.detpar)[0]
