"""Unit tests for IX_axis and bin_index.

Translated from:
- _test/test_IX_classes/test_IX_axis.m
- _test/test_IX_classes/test_bin_index.m
"""

import numpy as np
import pytest

from pyhorace.data import IX_axis, bin_index


class TestIXAxis:
    """Tests for IX_axis methods and constructors (from test_IX_axis.m)."""

    def test_methods(self):
        ia = IX_axis()
        ia.caption = "my axis name"
        assert ia.caption == ["my axis name"]

        ia.units = "meV"
        ia.code = "blabla"
        assert ia.units == "meV"
        assert ia.code == "blabla"

        ia.ticks = []
        assert len(ia.ticks["positions"]) == 0
        assert ia.ticks["labels"] == []

        with pytest.raises(ValueError, match="IX_axis:invalid_argument"):
            ia.ticks = {}

        data = {"positions": [], "labels": []}
        ia.ticks = data
        assert len(ia.ticks["positions"]) == 0
        assert ia.ticks["labels"] == []

        data["labels"] = ["1", "2", "3"]
        with pytest.raises(ValueError, match="IX_axis:invalid_argument"):
            ia.ticks = data

        data["labels"] = []
        data["positions"] = [1, 2, 3]
        ia.ticks = data
        da = ia.ticks
        np.testing.assert_array_equal(da.positions, data["positions"])

        data["labels"] = ["a", "b", "c"]
        ia.ticks = data
        np.testing.assert_array_equal(ia.ticks.positions, data["positions"])
        assert ia.ticks.labels == ["a", "b", "c"]

    def test_constructor(self):
        ia = IX_axis("my axis name")
        assert ia.caption == ["my axis name"]

        ia = IX_axis("my axis name", "meV")
        assert ia.caption == ["my axis name"]
        assert ia.units == "meV"

        ia = IX_axis("my axis name", "meV", "code")
        assert ia.caption == ["my axis name"]
        assert ia.units == "meV"
        assert ia.code == "code"

        ia = IX_axis("my axis name", "meV", "", [1, 2, 3])
        assert ia.caption == ["my axis name"]
        assert ia.units == "meV"
        assert ia.code == ""
        da = ia.ticks
        np.testing.assert_array_equal(da.positions, [1, 2, 3])
        assert isinstance(da.labels, list)
        assert len(da.labels) == 3

        data = {"positions": [1, 2, 3], "labels": ["a", "b", "c"]}
        ia = IX_axis("my axis name", "meV", "", data["positions"], data["labels"])
        assert ia.caption == ["my axis name"]
        assert ia.units == "meV"
        assert ia.code == ""
        da = ia.ticks
        np.testing.assert_array_equal(da.positions, [1, 2, 3])
        assert ia.ticks.labels == ["a", "b", "c"]

        ia = IX_axis("my axis name", "meV", "", data)
        assert ia.caption == ["my axis name"]
        assert ia.units == "meV"
        assert ia.code == ""
        da = ia.ticks
        np.testing.assert_array_equal(da.positions, [1, 2, 3])
        assert ia.ticks.labels == ["a", "b", "c"]

        ias = ia.to_struct()
        ia_restored = IX_axis(ias)
        assert ia_restored.caption == ["my axis name"]
        assert ia_restored.units == "meV"
        assert ia_restored.code == ""
        da = ia_restored.ticks
        np.testing.assert_array_equal(da.positions, [1, 2, 3])
        assert ia_restored.ticks.labels == ["a", "b", "c"]


class TestBinIndex:
    """Tests for bin_index function (from test_bin_index.m)."""

    @staticmethod
    def _check_all_ok(x, xbounds, ibin_t, ibin_f):
        np.testing.assert_array_equal(
            bin_index(x, xbounds, True),
            ibin_t,
            err_msg="Unexpected result for inclusive==True",
        )
        np.testing.assert_array_equal(
            bin_index(x, xbounds, False),
            ibin_f,
            err_msg="Unexpected result for inclusive==False",
        )

    def test_bin_index_cases(self):
        xbounds = [1, 4, 5]
        self._check_all_ok(
            [0.9, 1, 1.1, 4, 5, 6],
            xbounds,
            [0, 1, 1, 2, 2, 3],
            [0, 1, 1, 2, 3, 3],
        )
        self._check_all_ok([15, 16, 17], xbounds, [3, 3, 3], [3, 3, 3])
        self._check_all_ok(
            [2, 3, 3, 4, 4, 4, 5, 5],
            xbounds,
            [1, 1, 1, 2, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 3, 3],
        )

        xbounds = [1, 4, 4, 5]
        self._check_all_ok(
            [2, 3, 3, 4, 4, 4, 5, 5],
            xbounds,
            [1, 1, 1, 3, 3, 3, 3, 3],
            [1, 1, 1, 3, 3, 3, 4, 4],
        )

        xbounds = [1, 4, 5]
        self._check_all_ok([4.2, 4.3, 4.7], xbounds, [2, 2, 2], [2, 2, 2])
        self._check_all_ok([2, 5, 5, 5], xbounds, [1, 2, 2, 2], [1, 3, 3, 3])
        self._check_all_ok([2, 4, 4, 4], xbounds, [1, 2, 2, 2], [1, 2, 2, 2])
        self._check_all_ok([4, 4, 4], xbounds, [2, 2, 2], [2, 2, 2])
        self._check_all_ok([5, 5, 5], xbounds, [2, 2, 2], [3, 3, 3])

        xbounds = [1, 2, 3, 4, 5]
        self._check_all_ok([1.3, 2, 4, 6], xbounds, [1, 2, 4, 5], [1, 2, 4, 5])
        self._check_all_ok([1.3, 2, 4, 5], xbounds, [1, 2, 4, 4], [1, 2, 4, 5])

