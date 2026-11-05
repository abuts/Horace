"""Unit tests for sqw.run_inspector (ported from _test/test_sqw_pageOpMethods/test_run_inspector.m)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorace.core.data_op_interface import DataOpArray
from pyhorace.core.exceptions import HoraceError
from pyhorace.sqw import read_sqw, sqw

REPO_ROOT = Path(__file__).resolve().parents[4]
SQW_SOURCE = str(REPO_ROOT / "_test" / "common_data" / "sqw_4d.sqw")


@pytest.fixture(scope="module")
def source_sqw4D() -> sqw:
    return read_sqw(SQW_SOURCE)


@pytest.fixture(scope="module")
def source_sqw2D(source_sqw4D: sqw) -> sqw:
    return source_sqw4D.cut([-0.2, 0.2], [-0.2, 0.2], [], [])


@pytest.fixture(scope="module")
def source_sqw1D(source_sqw4D: sqw) -> sqw:
    return source_sqw4D.cut([-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2], [])


class TestRunInspector:
    """Port of test_run_inspector.m."""

    def test_run_inspector_2D(self, source_sqw2D: sqw) -> None:
        _, nd, split_data = source_sqw2D.run_inspector("test_videofig", True)
        assert nd == 2
        assert len(split_data) == 21

    def test_run_inspector_1D(self, source_sqw1D: sqw) -> None:
        _, nd, split_data = source_sqw1D.run_inspector("test_videofig", True)
        assert nd == 1
        assert len(split_data) == 21

    def test_invalid_param_throw(
        self, source_sqw4D: sqw, source_sqw1D: sqw
    ) -> None:
        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw4D.run_inspector()

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            sqw.run_inspector(DataOpArray([source_sqw1D, source_sqw1D]))

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw1D.run_inspector("ax")

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw1D.run_inspector("col", [1, -1])

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw1D.run_inspector("ax", [1, -1])

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw1D.run_inspector("ax", [-1, 1, -2, -2])

        with pytest.raises(HoraceError, match="HORACE:run_inspector:invalid_argument"):
            source_sqw1D.run_inspector("ax", [1, -1, -2, 2])

    def test_parse_col(self, source_sqw1D: sqw) -> None:
        pr, nd, _ = source_sqw1D.run_inspector("colour", [-1, 1], "test_parser", True)
        assert nd == 1
        assert pr["ax"].size == 0
        np.testing.assert_array_equal(pr["col"], [-1, 1])

    def test_parse_ax(self, source_sqw2D: sqw) -> None:
        pr, nd, _ = source_sqw2D.run_inspector(
            "axis", [-1, 1, -2, 2], "test_parser", True
        )
        assert nd == 2
        assert pr["ax"].size != 0
        np.testing.assert_array_equal(pr["ax"], [-1, 1, -2, 2])
        assert pr["col"].size == 0

    def test_parse_default_par(self, source_sqw2D: sqw) -> None:
        pr, nd, _ = source_sqw2D.run_inspector("test_parser", True)
        assert nd == 2
        assert pr["ax"].size == 0
        assert pr["col"].size == 0
