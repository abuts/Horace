"""Unit tests for main_header_cl and parse_full_filename.

Translated from:
- ``_test/test_main_header/test_main_header_constructor.m``
- ``_test/test_utilities_herbert/test_parse_full_filename.m``
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from pyhorace.core.serializable import Serializable
from pyhorace.io.main_header_cl import main_header_cl, parse_full_filename


class TestMainHeaderConstructor:
    """Tests for main_header_cl construction, properties, and serialization."""

    @staticmethod
    def get_close_date(date_tested: str) -> str:
        """Helper routine to get current date and time close to the one obtained from main_header_cl.

        Use after ``main_header_cl.creation_date`` is called.

        Parameters
        ----------
        date_tested : str
            The date string obtained from ``main_header_cl.creation_date``.
        """
        date_now = datetime.now()
        date_now_m = date_now - timedelta(seconds=1)

        date_now_str = main_header_cl.convert_datetime_to_str(date_now)
        date_now_m_str = main_header_cl.convert_datetime_to_str(date_now_m)
        if date_now_str == date_tested:
            return date_now_str
        return date_now_m_str

    def test_constructor_from_sqw_file_keeps_defined_date(self) -> None:
        inputs = {
            "filename_with_cdate": "test_file$2021-01-01T20:10:10",
            "filepath": "test_path",
            "title": "my_test_title",
            "nfiles": 10,
        }
        th = main_header_cl(inputs)
        assert th.filename == "test_file"
        assert th.filepath == "test_path"
        assert th.title == "my_test_title"
        assert th.nfiles == 10
        assert th.creation_date_defined is True
        assert th.creation_date == "2021-01-01T20:10:10"

    def test_constructor_from_sqw_file_keeps_undefined_date(self) -> None:
        inputs = {
            "filename_with_cdate": "test_file",
            "filepath": "test_path",
            "title": "my_test_title",
            "nfiles": 10,
        }
        th = main_header_cl(inputs)
        assert th.filename == "test_file"
        assert th.filepath == "test_path"
        assert th.title == "my_test_title"
        assert th.nfiles == 10
        assert th.creation_date_defined is False

    def test_serialize_deserialize_keeps_defined_date(self) -> None:
        inputs = {
            "filename": "test_file",
            "filepath": "test_path",
            "title": "my_test_title",
            "nfiles": 10,
        }
        th = main_header_cl(inputs)
        assert th.filename == "test_file.sqw"
        assert th.filepath == "test_path"
        assert th.title == "my_test_title"
        assert th.nfiles == 10
        assert th.creation_date_defined is False

        date_tested = th.creation_date
        close_date = self.get_close_date(date_tested)

        th.creation_date = date_tested
        assert th.creation_date_defined is True

        assert date_tested == close_date
        assert th.filename_with_cdate == f"test_file.sqw${date_tested}"

        dat = th.saveobj()
        th_rec = Serializable.loadobj(dat)

        assert th == th_rec

    def test_serialize_deserialize_keeps_undefined_date(self) -> None:
        inputs = {
            "filename": "test_file",
            "filepath": "test_path",
            "title": "my_test_title",
            "nfiles": 10,
        }
        th = main_header_cl(inputs)
        assert th.filename == "test_file.sqw"
        assert th.filepath == "test_path"
        assert th.title == "my_test_title"
        assert th.nfiles == 10
        assert th.creation_date_defined is False

        date_tested = th.creation_date
        close_date = self.get_close_date(date_tested)

        assert date_tested == close_date
        assert th.filename_with_cdate == "test_file.sqw"

        dat = th.saveobj()
        th_rec = Serializable.loadobj(dat)

        assert th == th_rec

    def test_partial_constructor(self) -> None:
        th = main_header_cl("my_file", "my_path")
        assert th.filename == "my_file"
        assert th.filepath == "my_path"
        assert th.title == ""
        assert th.nfiles == 0
        assert th.creation_date_defined is False

    def test_invalid_setter_throws(self) -> None:
        th = main_header_cl()

        def setter(prop_name: str, prop_val: object) -> None:
            setattr(th, prop_name, prop_val)

        with pytest.raises(ValueError, match="HORACE:main_header:invalid_argument"):
            setter("nfiles", "a")

        with pytest.raises(ValueError, match="HORACE:main_header:invalid_argument"):
            setter("filename", 10)

        with pytest.raises(ValueError, match="HORACE:main_header:invalid_argument"):
            setter("filepath", 10)

        with pytest.raises(ValueError, match="HORACE:main_header:invalid_argument"):
            setter("title", 10)

        with pytest.raises(ValueError, match="HORACE:main_header:invalid_argument"):
            setter("creation_date", 10)

        with pytest.raises((AttributeError, ValueError), match="noSetMethod"):
            setter("creation_date_defined", 1)

    def test_existing_file_creation_date(self) -> None:
        th = main_header_cl()
        the_file = os.path.abspath(__file__)
        fp, file = os.path.split(the_file)

        th.filename = file
        th.filepath = fp

        assert th.creation_date_defined is False

        # Test the mechanism of defining new creation date:
        # the creation date is taken from the existing file modification/creation date
        ct_tested = th.creation_date
        assert th.filename_with_cdate == file

        # Set up creation date:
        th.creation_date = ct_tested
        assert th.creation_date_defined is True
        assert th.creation_date == ct_tested
        assert th.filename_with_cdate == f"{file}${ct_tested}"

    def test_non_existing_file_and_other_properties(self) -> None:
        th = main_header_cl()
        th.nfiles = 10
        assert th.nfiles == 10
        th.filename = "some_file"
        assert th.filename == "some_file"
        th.filepath = "/path"
        assert th.filepath == "/path"
        th.title = "some title"
        assert th.title == "some title"

        assert th.creation_date_defined is False

        # Test the mechanism of defining new creation date
        ct_tested = th.creation_date
        assert th.filename_with_cdate == "some_file"

        th.creation_date = ct_tested
        assert th.creation_date_defined is True
        assert th.creation_date == ct_tested
        assert th.filename_with_cdate == f"some_file${ct_tested}"

    def test_empty_constructor(self) -> None:
        th = main_header_cl()
        assert th.nfiles == 0
        assert th.filename == ""
        assert th.filepath == ""
        assert th.title == ""
        assert th.creation_date_defined is False

        date_tested = th.creation_date
        date_selected = self.get_close_date(date_tested)
        assert date_tested == date_selected


class TestParseFullFilename:
    """Tests for parse_full_filename utility."""

    def test_parse_real_path(self) -> None:
        ts = os.path.splitext(os.path.abspath(__file__))[0]
        refp = os.path.dirname(ts)
        ref_base = os.path.basename(ts)
        fp, fn = parse_full_filename(ts)

        assert isinstance(fp, str)
        assert isinstance(fn, str)
        assert fp == refp
        assert fn == f"{ref_base}.sqw"

    def test_not_a_path_throw(self) -> None:
        with pytest.raises(ValueError, match="HERBERT:utilities:invalid_argument"):
            parse_full_filename(10)  # type: ignore[arg-type]

    def test_multistring_throw(self) -> None:
        ts = ["filepath", os.sep, "filename.tmp1"]
        with pytest.raises(ValueError, match="HERBERT:utilities:invalid_argument"):
            parse_full_filename(ts)  # type: ignore[arg-type]

    def test_name_and_extension(self) -> None:
        ts = os.sep.join(["filepath", "filename.tmp1"])
        fp, fn = parse_full_filename(ts)

        assert fp == "filepath"
        assert fn == "filename.tmp1"

    def test_parse_simple_path(self) -> None:
        ts = os.sep.join(["filepath", "filename"])
        fp, fn = parse_full_filename(ts)

        assert fp == "filepath"
        assert fn == "filename.sqw"

    def test_parse_just_string(self) -> None:
        ts = "abcde"
        fp, fn = parse_full_filename(ts)

        assert fp == ""
        assert fn == "abcde.sqw"

