"""Unit tests for the ``sqw`` class properties, constructor, and destructor.

Translated from:
- ``_test/test_sqw_class/test_sqw_constructor.m``
- ``_test/test_sqw/test_sqw_main.m``
plus property validation and Python destructor (``__del__``) tests for ``is_tmp_obj``.
"""

from __future__ import annotations

import gc
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError, fast_map
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.unique_references_container import unique_references_container
from pyhorace.data.experiment import Experiment
from pyhorace.data.instr_classes.ix_detector_array import IX_detector_array
from pyhorace.data.instr_classes.ix_experiment import IX_experiment
from pyhorace.data.instr_classes.ix_null_inst import IX_null_inst
from pyhorace.data.instr_classes.ix_samp import IX_samp
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory
from pyhorace.io.main_header_cl import main_header_cl
from pyhorace.sqw import (
    DnDBase,
    SQWDnDBase,
    d0d,
    d1d,
    d2d,
    read_sqw,
    sqw,
)

HORACE_ROOT = Path(__file__).resolve().parents[3]
COMMON_DATA_DIR = HORACE_ROOT / "_test" / "common_data"
SQW_1D_1_PATH = str(COMMON_DATA_DIR / "sqw_1d_1.sqw")
SQW_1D_2_PATH = str(COMMON_DATA_DIR / "sqw_1d_2.sqw")
WREF_SQW_PATH = str(HORACE_ROOT / "_test" / "test_change_crystal" / "wref.sqw")
OLD_MAT_SQW_PATH = str(
    HORACE_ROOT / "_test" / "test_sqw_class" / "data" / "sqw_loadobj_test_v3_6_1.mat"
)
if not os.path.isfile(OLD_MAT_SQW_PATH):
    OLD_MAT_SQW_PATH = str(
        HORACE_ROOT / "_test" / "test_sqw_class" / "sqw_loadobj_test_v3_6_1.mat"
    )

DATA_NPIX_1D_1 = 100337


@pytest.fixture(autouse=True)
def _preserve_hor_config():
    """Save and restore ``hor_config().mem_chunk_size`` around each test."""
    hc = hor_config()
    saved_chunk = hc.mem_chunk_size
    hc.mem_chunk_size = int(1e6)
    try:
        yield
    finally:
        hc.mem_chunk_size = saved_chunk


# ==============================================================================
# 1. Tests translated from _test/test_sqw_class/test_sqw_constructor.m
# ==============================================================================


class TestSqwConstructor:
    """Port of ``_test/test_sqw_class/test_sqw_constructor.m``."""

    def test_sqw_class_follows_expected_class_hierarchy(self):
        sqw_obj = sqw()
        assert isinstance(sqw_obj, sqw)
        assert isinstance(sqw_obj, SQWDnDBase)

    def test_default_constructor_returns_empty_instance(self):
        sqw_obj = sqw()

        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.main_header == main_header_cl()
        assert isinstance(sqw_obj.exper_info, Experiment)
        # Compatibility alias check
        assert sqw_obj.experiment_info is sqw_obj.exper_info

        assert sqw_obj.exper_info.instruments.n_runs == 0
        with pytest.raises((IndexError, ValueError)):
            _ = sqw_obj.exper_info.instruments[0]

        assert sqw_obj.exper_info.samples.n_runs == 0
        with pytest.raises((IndexError, ValueError)):
            _ = sqw_obj.exper_info.samples[0]

        assert sqw_obj.exper_info.detector_arrays.n_runs == 0
        assert sqw_obj.pix == PixelDataBase.create()
        assert len(sqw_obj.data.pax) == 0

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_filename_constructor_returns_populated_class(self):
        sqw_obj = sqw(SQW_1D_1_PATH)

        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.main_header.nfiles == 14
        assert len(sqw_obj.exper_info.expdata) == 14
        assert sqw_obj.exper_info.instruments.n_runs == 14
        assert sqw_obj.exper_info.samples.n_runs == 14
        assert sqw_obj.exper_info.detector_arrays[0].group.size == 36864
        assert len(sqw_obj.data.pax) == 1
        assert sqw_obj.pix.num_pixels == DATA_NPIX_1D_1
        assert sqw_obj.npixels == DATA_NPIX_1D_1
        assert sqw_obj.num_pixels == DATA_NPIX_1D_1

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_filename_constructor_sets_pixel_page_size_if_passed(self):
        hc = hor_config()
        pagesize_pixels = 6666
        hc.mem_chunk_size = pagesize_pixels

        sqw_obj = sqw(SQW_1D_1_PATH, "file_backed", True)

        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.is_filebacked
        assert sqw_obj.pix.num_pixels > pagesize_pixels
        assert sqw_obj.pix.num_pixels == DATA_NPIX_1D_1
        assert sqw_obj.pix.page_size == pagesize_pixels

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_filename_constructor_sets_all_data_default_pagesize(self):
        sqw_obj = sqw(SQW_1D_1_PATH)

        hc = hor_config()
        mmc = hc.mem_chunk_size

        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.pix.num_pixels == DATA_NPIX_1D_1
        assert sqw_obj.pix.page_size == min(mmc, DATA_NPIX_1D_1)

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_copy_constructor_clones_object(self):
        sqw_obj = sqw(SQW_1D_1_PATH)
        sqw_copy = sqw(sqw_obj)

        assert isinstance(sqw_obj, sqw)
        is_eq, msg = sqw_copy.equal_to_tol(sqw_obj)
        assert is_eq, msg

        # Ensure deep copy semantics
        sqw_copy.main_header.title = "modified_title"
        assert sqw_obj.main_header.title != "modified_title"

        # Test copy with exclude_pix=True
        sqw_no_pix = sqw_obj.copy("-exclude_pix")
        assert sqw_no_pix.npixels == 0
        assert bool(sqw_no_pix.dnd_type) is True
        assert sqw_obj.npixels == DATA_NPIX_1D_1

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_save_load_returns_identical_object(self):
        sqw_obj = read_sqw(SQW_1D_1_PATH)
        keys = np.asarray(sqw_obj.runid_map.keys, dtype=float)
        ids = sqw_obj.exper_info.expdata.get_run_ids()
        np.testing.assert_array_equal(keys, ids)

        pix_ids = np.unique(sqw_obj.pix.run_idx)
        np.testing.assert_array_equal(ids, pix_ids)

        saved_struct = sqw_obj.saveobj()
        assert saved_struct["serial_name"] == "sqw"
        assert saved_struct["version"] == 6
        assert "exper_info" in saved_struct

        mat_stored_new = sqw.loadobj(saved_struct)
        assert isinstance(mat_stored_new, sqw)
        is_eq, msg = mat_stored_new.equal_to_tol(sqw_obj, tol=[1e-15, 1e-15])
        assert is_eq, msg

        # Test loading from legacy v3.6.1 / v4 struct format (experiment_info + nested data.pix)
        legacy_data = dict(saved_struct["data"])
        legacy_data["pix"] = saved_struct["pix"]
        legacy_struct = {
            "serial_name": "sqw",
            "version": 4,
            "main_header": saved_struct["main_header"],
            "experiment_info": saved_struct["exper_info"],
            "data": legacy_data,
        }
        mat_stored_old = sqw.loadobj(legacy_struct)
        assert isinstance(mat_stored_old, sqw)
        is_eq_old, msg_old = mat_stored_old.equal_to_tol(sqw_obj, tol=[1e-15, 1e-15])
        assert is_eq_old, msg_old

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_1_PATH), reason="sqw_1d_1.sqw test file not found"
    )
    def test_init_from_file_membacked_and_force_filebacked(self):
        in_data_mem = {
            "file": SQW_1D_1_PATH,
            "file_backed": False,
            "force_pix_location": True,
        }
        so_mem = sqw()
        so_mem.init_from_file(in_data_mem)
        assert isinstance(so_mem.pix, PixelDataMemory)
        assert not so_mem.is_filebacked

        in_data_fb = {
            "file": SQW_1D_1_PATH,
            "file_backed": True,
            "force_pix_location": True,
        }
        so_fb = sqw()
        so_fb.init_from_file(in_data_fb)
        assert isinstance(so_fb.pix, PixelDataFileBacked)
        assert so_fb.is_filebacked

    def test_input_args_filebacked_specified_false(self):
        args = sqw.parse_sqw_args(SQW_1D_1_PATH, "file_backed", False)
        assert args["force_pix_location"] is True
        assert args["file_backed"] is False

    def test_input_args_filebacked_specified_true(self):
        args = sqw.parse_sqw_args(SQW_1D_1_PATH, "file_backed", True)
        assert args["force_pix_location"] is True
        assert args["file_backed"] is True

    def test_input_args_filebacked_default(self):
        args = sqw.parse_sqw_args()
        assert args["force_pix_location"] is False
        assert args["file_backed"] is False


# ==============================================================================
# 2. Tests translated from _test/test_sqw/test_sqw_main.m
# ==============================================================================


class TestSqwMainProperties:
    """Property and constructor tests from ``_test/test_sqw/test_sqw_main.m``."""

    @pytest.mark.skipif(
        not os.path.isfile(WREF_SQW_PATH), reason="wref.sqw test file not found"
    )
    def test_read_sqw(self):
        sqw_data = read_sqw(WREF_SQW_PATH)
        assert isinstance(sqw_data, sqw)

        np.testing.assert_allclose(sqw_data.data.alatt, [2.87, 2.87, 2.87], atol=1e-4)
        assert sqw_data.data.npix.shape == (21, 20)
        np.testing.assert_array_equal(sqw_data.data.pax, [2, 4])
        np.testing.assert_array_equal(sqw_data.data.iax, [1, 3])

        test_dnd = sqw_data.d2d()
        assert isinstance(test_dnd, d2d)

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_2_PATH), reason="sqw_1d_2.sqw test file not found"
    )
    def test_setting_pix_page_size_in_constructor_pages_pixels(self):
        sq_obj = sqw(SQW_1D_2_PATH, "file_backed", True)
        assert isinstance(sq_obj.pix, PixelDataFileBacked)
        assert sq_obj.is_filebacked

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_2_PATH), reason="sqw_1d_2.sqw test file not found"
    )
    def test_pixels_not_paged_if_pixel_page_size_arg_not_given(self):
        sq_obj = sqw(SQW_1D_2_PATH)
        assert sq_obj.pix.page_size == sq_obj.pix.num_pixels
        assert not sq_obj.is_filebacked

    def test_sqw_constructor_from_dnd(self):
        data = d2d()
        tsqw_obj = sqw(data)
        assert bool(tsqw_obj.dnd_type) is True
        assert tsqw_obj.dnd_type() is True
        assert not tsqw_obj.has_pixels()


# ==============================================================================
# 3. Property Getter/Setter Validation & Python Destructor (is_tmp_obj) Tests
# ==============================================================================


class TestSqwPropertiesAndDestructor:
    """Tests for ``sqw`` property getters/setters, ``exper_info`` rename, and ``__del__`` destructor."""

    def test_saveable_fields_uses_exper_info(self):
        s = sqw()
        assert s.saveableFields() == ["main_header", "exper_info", "data", "pix"]
        assert s.classVersion() == 6

    def test_main_header_property_and_inconsistency_warning(self):
        s = sqw()
        mh = main_header_cl(filename="test.sqw", filepath="/tmp", title="my_sqw", nfiles=2)
        s.main_header = mh
        assert s.main_header.filename == "test.sqw"
        assert s.main_header.title == "my_sqw"
        assert s.main_header.nfiles == 2

        # Setting exper_info with 1 run updates main_header.nfiles to 1
        exp1 = IX_experiment(filename="r1.nxspe", filepath="/tmp", efix=100.0, emode=1, en=[0.0, 10.0], run_id=1)
        samp1 = IX_samp([3.0, 3.0, 3.0], [90.0, 90.0, 90.0])
        s.exper_info = Experiment(None, [IX_null_inst()], [samp1], [exp1])
        assert s.main_header.nfiles == 1

        # Setting main_header with mismatched nfiles warns HORACE:inconsitent_sqw
        with pytest.warns(UserWarning, match="not equal to nfiles in main_header"):
            s.main_header = main_header_cl(nfiles=5)

        with pytest.raises(HoraceError, match="Main header has to be an instance"):
            s.main_header = 12345

    def test_exper_info_and_detpar_properties(self):
        s = sqw()
        exp1 = IX_experiment(filename="r1.nxspe", filepath="/tmp", efix=100.0, emode=1, en=[0.0, 10.0], run_id=10)
        exp2 = IX_experiment(filename="r2.nxspe", filepath="/tmp", efix=100.0, emode=1, en=[0.0, 10.0], run_id=20)
        samp = IX_samp([3.0, 3.0, 3.0], [90.0, 90.0, 90.0])
        s.exper_info = Experiment(None, [IX_null_inst(), IX_null_inst()], [samp, samp], [exp1, exp2])

        assert s.exper_info.n_runs == 2
        assert s.main_header.nfiles == 2
        np.testing.assert_array_equal(s.runid_map.keys, [10, 20])
        np.testing.assert_array_equal(s.runid_map.values, [1, 2])

        # Setting detpar via legacy dict populates exper_info.detector_arrays replicated for 2 runs
        det_dict = {
            "filename": "det.par",
            "filepath": "/tmp",
            "group": np.array([1.0, 2.0, 3.0]),
            "x2": np.array([4.0, 4.0, 4.0]),
            "phi": np.array([10.0, 20.0, 30.0]),
            "azim": np.array([0.0, 0.0, 0.0]),
            "width": np.array([0.02, 0.02, 0.02]),
            "height": np.array([0.02, 0.02, 0.02]),
        }
        s.detpar = det_dict
        assert isinstance(s.detpar, unique_references_container)
        assert s.detpar.n_runs == 2
        assert s.detpar[0].group.size == 3

        # Legacy header property returns list of 2 dicts without 'instrument' or 'sample'
        hdrs = s.header
        assert isinstance(hdrs, list)
        assert len(hdrs) == 2
        assert "instrument" not in hdrs[0]
        assert "sample" not in hdrs[0]
        assert hdrs[0]["efix"] == 100.0

        with pytest.raises(HoraceError, match="Experiment info can be only an instance"):
            s.exper_info = "invalid"

    def test_data_and_pix_properties(self):
        s = sqw()
        d1 = d1d()
        s.data = d1
        assert isinstance(s.data, d1d)
        assert s.dimensions == 1

        with pytest.raises(HoraceError, match="Data field has to be a DnDBase"):
            s.data = "not_a_dnd"

        raw_pix = np.ones((9, 25), dtype=float)
        s.pix = raw_pix
        assert isinstance(s.pix, PixelDataMemory)
        assert s.npixels == 25
        assert s.num_pixels == 25
        assert s.has_pixels() is True
        assert bool(s.dnd_type) is False
        assert s.dnd_type() is False

        s.pix = []
        assert s.npixels == 0
        assert bool(s.dnd_type) is True
        assert s.dnd_type() is True

    def test_full_filename_and_creation_date_propagation(self):
        s = sqw()
        target_path = os.path.abspath("some_folder/my_test_file.sqw")
        s.full_filename = target_path
        assert s.full_filename == target_path
        assert s.main_header.full_filename == target_path
        assert s.data.full_filename == target_path
        assert s.pix.full_filename == target_path

        dt_str = "2026-05-01T12:30:00"
        s.creation_date = dt_str
        assert s.main_header.creation_date_defined
        assert s.creation_date == s.main_header.creation_date
        assert s.data.creation_date == s.main_header.creation_date

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_2_PATH), reason="sqw_1d_2.sqw test file not found"
    )
    def test_tmp_obj_destructor_deletes_backing_file_on_gc(self, tmp_path):
        """Verify that replacing ``tmp_file_holder_`` with Python's ``__del__`` destructor
        deletes the temporary backing file when ``is_tmp_obj`` is True and the object is
        garbage-collected, while leaving non-temporary files untouched.
        """
        tmp_sqw_file = str(tmp_path / "temp_backed.tmp")
        shutil.copyfile(SQW_1D_2_PATH, tmp_sqw_file)
        assert os.path.isfile(tmp_sqw_file)

        # 1. Non-temporary file-backed sqw does NOT delete the file on destruction
        obj_normal = sqw(tmp_sqw_file, file_backed=True)
        assert obj_normal.is_filebacked
        assert obj_normal.is_tmp_obj is False
        del obj_normal
        gc.collect()
        assert os.path.isfile(tmp_sqw_file)

        # 2. Marking as temporary via set_as_tmp_obj causes __del__ to delete the file on GC
        obj_tmp = sqw(tmp_sqw_file, file_backed=True)
        obj_tmp.set_as_tmp_obj()
        assert obj_tmp.is_tmp_obj is True
        assert obj_tmp.full_filename == tmp_sqw_file

        # Copy of a tmp object should NOT steal temporary file deletion ownership
        obj_copy = obj_tmp.copy()
        assert obj_copy.is_tmp_obj is False
        obj_copy.deactivate()
        del obj_copy
        gc.collect()
        assert os.path.isfile(tmp_sqw_file)

        # Destroying obj_tmp triggers __del__ -> delete() -> removes tmp_sqw_file
        del obj_tmp
        gc.collect()
        assert not os.path.isfile(tmp_sqw_file)

    @pytest.mark.skipif(
        not os.path.isfile(SQW_1D_2_PATH), reason="sqw_1d_2.sqw test file not found"
    )
    def test_activate_with_tmp_extension_sets_is_tmp_obj(self, tmp_path):
        """Verify ``activate()`` on a ``.tmp`` file sets ``is_tmp_obj`` and ``deactivate()`` clears it."""
        tmp_sqw_file = str(tmp_path / "activated_file.tmp")
        shutil.copyfile(SQW_1D_2_PATH, tmp_sqw_file)

        obj = sqw(tmp_sqw_file, file_backed=True)
        obj.deactivate()
        assert obj.is_tmp_obj is False

        obj.activate(tmp_sqw_file)
        assert obj.is_tmp_obj is True

        # Explicit delete() removes the temporary file immediately
        obj.delete()
        assert obj.is_tmp_obj is False
        assert not os.path.isfile(tmp_sqw_file)

