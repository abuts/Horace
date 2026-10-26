"""Unit tests for IX_source class, translated from MATLAB test_IX_source.m."""

from __future__ import annotations

import pytest

from pyhorace.core.hashable import HashableArray, isequal
from pyhorace.data.instr_classes.ix_source import IX_source
from tests.data.instr_classes._helpers import (
    check_matfile_IO,
    hashable_obj_tester,
)


class TestIXSource:
    """Port of MATLAB ``test_IX_source`` test suite."""

    def test_IX_source_1(self) -> None:
        src = IX_source()
        assert src.name == ""
        assert src.target_name == ""
        assert src.frequency == 0.0
        assert src.isempty() is True

        ver_src = check_matfile_IO(src.class_version(), False, src)
        assert isequal(src, ver_src)

    def test_IX_source_2(self) -> None:
        src = IX_source("ISIS")
        assert src.name == "ISIS"
        assert src.target_name == ""
        assert src.frequency == 0.0
        assert src.isempty() is False

        ver_src = check_matfile_IO(src.class_version(), False, src)
        assert isequal(src, ver_src)

    def test_IX_source_3(self) -> None:
        src = IX_source("ISIS", "TS2")
        assert src.name == "ISIS"
        assert src.target_name == "TS2"
        assert src.frequency == 0.0

        ver_src = check_matfile_IO(src.class_version(), False, src)
        assert isequal(src, ver_src)

    def test_IX_source_4(self) -> None:
        src = IX_source("ISIS", "TS2", 40)
        assert src.name == "ISIS"
        assert src.target_name == "TS2"
        assert src.frequency == 40.0

        ver_src = check_matfile_IO(src.class_version(), False, src)
        assert isequal(src, ver_src)

    def test_IX_source_freq_Only(self) -> None:
        src = IX_source(40)
        assert src.name == ""
        assert src.target_name == ""
        assert src.frequency == 40.0

    def test_IX_source_NameAndfreq_Only(self) -> None:
        src = IX_source("ISIS", 40)
        assert src.name == "ISIS"
        assert src.target_name == ""
        assert src.frequency == 40.0

    def test_IX_source_KeyVal(self) -> None:
        with pytest.warns(DeprecationWarning):
            src = IX_source(40, "-target_name", "TS2")
        assert src.name == ""
        assert src.target_name == "TS2"
        assert src.frequency == 40.0

        src2 = IX_source("ISIS", target_name="TS1", frequency=50)
        assert src2.name == "ISIS"
        assert src2.target_name == "TS1"
        assert src2.frequency == 50.0

    def test_hashable_prop(self) -> None:
        src = IX_source("ISIS", "TS1", 50)
        hashable_obj_tester(src)

    def test_save_load_prev_versions(self) -> None:
        src_expect = IX_source("ISIS", "TS2", 40)
        ver0_struct = {
            "name_": "ISIS",
            "target_name_": "TS2",
            "frequency_": 40.0,
        }
        ver1_struct = {
            "serial_name": "IX_source",
            "version": 1,
            "array_dat": {
                "name_": "ISIS",
                "target_name_": "TS2",
                "frequency_": 40.0,
                "class_version_": 1,
            },
        }
        ver0_src = IX_source.loadobj(ver0_struct)
        ver1_src = IX_source.from_struct(ver1_struct)
        assert isinstance(ver0_src, IX_source)
        assert isinstance(ver1_src, IX_source)
        assert isequal(ver0_src, src_expect)
        assert isequal(ver1_src, src_expect)

        srs_arr = HashableArray(
            [
                IX_source("ISIS", "TS2", 40),
                IX_source("name", "SNS", "target_name", "TS1", "frequency", 50),
                IX_source(),
            ]
        )
        arr_rec = check_matfile_IO(src_expect.class_version(), False, srs_arr)
        assert srs_arr.eq(arr_rec)
