"""Tests for data_op_interface operation order and priority (port of test_data_op_order.m)."""

from __future__ import annotations

import numpy as np
import pytest

from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.data_op_interface import data_op_interface
from pyhorace.data.ix_dataset_1d import IX_dataset_1d
from pyhorace.data.pixels import PixelDataBase, PixelDataMemory
from pyhorace.data.sigvar import SigVar
from pyhorace.sqw.d0d import d0d
from pyhorace.sqw.d2d import d2d
from pyhorace.sqw.sqw import sqw


class TestDataOpOrder:
    """Port of ``_test/test_sqw_pageOpMethods/test_data_op_order.m``."""

    @pytest.fixture
    def known_classes(self):
        base_sqw = sqw.generate_cube_sqw(4)
        sqw0 = sqw.generate_cube_sqw(1)
        sqw0.pix = PixelDataBase.create()
        sqw_nopix = base_sqw.copy()
        sqw_nopix.pix = PixelDataBase.create()
        dd = d2d(
            line_axes("nbins_all_dims", [4, 4, 1, 1], "img_range", [[0, 0, 0, 0], [1, 1, 1, 1]]),
            line_proj("alatt", 3, "angdeg", 90),
        )
        return [
            base_sqw,
            PixelDataMemory(np.ones((9, 5))),
            sqw_nopix,
            dd,
            IX_dataset_1d(np.arange(1, 11, dtype=np.float64), np.arange(1, 11, dtype=np.float64)),
            SigVar(np.ones((2, 10)), np.ones((2, 10))),
            np.ones((3, 5)),
            sqw0,
            d0d(),
            IX_dataset_1d(np.array([1.0]), np.array([1.0])),
            SigVar(np.array([[1.0]]), np.array([[1.0]])),
            10,
        ]

    def test_num_img_order(self):
        obj1 = IX_dataset_1d()
        obj2 = 4
        flip, page_op_order = data_op_interface.get_operation_order(obj2, obj1)
        assert flip is True
        assert page_op_order == 0

    def test_img_num_order(self):
        obj1 = IX_dataset_1d()
        obj2 = 4
        flip, page_op_order = data_op_interface.get_operation_order(obj1, obj2)
        assert flip is False
        assert page_op_order == 0

    def test_op_kind_pix_img(self):
        op_kind = data_op_interface.get_operation_kind(True, False, False, True)
        assert op_kind == 2

    def test_op_kind_sqw_img(self):
        op_kind = data_op_interface.get_operation_kind(True, True, False, True)
        assert op_kind == 2

    def test_op_kind_sqw_sqw(self):
        op_kind = data_op_interface.get_operation_kind(True, True, True, True)
        assert op_kind == 3

    def test_op_kind_sqw_pix(self):
        op_kind = data_op_interface.get_operation_kind(True, True, True, False)
        assert op_kind == 3

    def test_gen_priorities(self, known_classes):
        pri = np.array([data_op_interface.get_priority(x)[0] for x in known_classes])
        assert pri.size == 12
        uni_pri = np.flip(np.unique(pri))
        np.testing.assert_array_equal(pri, uni_pri)
