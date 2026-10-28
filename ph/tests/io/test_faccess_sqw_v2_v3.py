import os
import shutil
import numpy as np
import pytest

from pyhorace.core.exceptions import HoraceError
from pyhorace.io import (
    sqw_formats_factory,
    faccess_sqw_prototype,
    faccess_sqw_v2,
    faccess_sqw_v3,
    faccess_sqw_v3_2,
    faccess_sqw_v3_3,
    faccess_sqw_v4,
    faccess_dnd_v2,
)
from pyhorace.io.sqw_fopen import sqw_fclose
from pyhorace.sqw import sqw, d1d, d2d
from pyhorace.data.instr_classes import (
    IX_experiment,
    IX_null_inst,
    IX_null_sample,
    IX_sample,
    IX_inst_DGfermi,
    IX_moderator,
    IX_aperture,
    IX_fermi_chopper,
)

from pathlib import Path

HORACE_ROOT = Path(__file__).resolve().parents[3]
COMMON_DATA_DIR = str(HORACE_ROOT / "_test" / "common_data")
TEST_SQW_FILE_DIR = str(HORACE_ROOT / "_test" / "test_sqw_file")


def create_test_instrument():
    mod = IX_moderator(10.0, 2.0, "ikcarp", [10.0, 20.0, 0.5])
    ap = IX_aperture(-8.0, 0.05, 0.05)
    fc = IX_fermi_chopper(1.5, 600.0, 0.05, 1.3, 0.002)
    return IX_inst_DGfermi(mod, ap, fc, 100.0, name="test_fermi")


# =============================================================================
# test_faccess_sqw_prototype.m
# =============================================================================
class TestFaccessSqwPrototype:
    @pytest.fixture
    def sample_v0_file(self):
        return os.path.join(TEST_SQW_FILE_DIR, "test_sqw_read_write_v0_t.sqw")

    def test_should_load_stream(self, sample_v0_file):
        to = faccess_sqw_prototype()
        assert to.faccess_version == 0
        assert to.file_version == "-v0"
        stream, pos = to.get_file_header(sample_v0_file)
        ok, initobj = to.should_load_stream(stream, pos)
        if hasattr(initobj, "file_id") and initobj.file_id is not None and initobj.file_id > 0:
            sqw_fclose(initobj.file_id)
        assert ok
        assert initobj.file_id is not None and initobj.file_id > 0

    def test_init_and_get(self, sample_v0_file):
        to = faccess_sqw_prototype(sample_v0_file)
        assert to.num_dim == 4
        assert to.filename == "test_sqw_read_write_v0_t.sqw"
        assert to.data_type == "a"
        assert to.num_contrib_files == 2
        assert list(to.dnd_dimensions) == [4, 4, 4, 4]
        assert to.npixels == 16

        exp_info = to.get_exp_info("-all")
        assert exp_info.n_runs == 2

        det = to.get_detpar()
        assert det[0].ndet > 0

        data = to.get_data()
        assert list(data.s.shape) == list(to.dnd_dimensions)

        pix = to.get_pix()
        assert pix.num_pixels == 16

        sqw_obj = to.get_sqw()
        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.pix.num_pixels == 16
        assert sqw_obj.exper_info.n_runs == 1
        to.delete()

    def test_upgrade_v0_to_v4(self, sample_v0_file):
        to = faccess_sqw_prototype(sample_v0_file)
        try:
            with pytest.raises(HoraceError, match="SQW_FILE_IO:legacy_data"):
                to.upgrade_file_format()
        finally:
            to.delete()


# =============================================================================
# test_faccess_sqw_v2.m
# =============================================================================
class TestFaccessSqwV2:
    @pytest.fixture
    def sample_file(self):
        return os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw")

    def test_should_load_stream(self, sample_file):
        to = faccess_sqw_v2()
        assert to.faccess_version == 2
        stream, pos = to.get_file_header(sample_file)
        ok, initobj = to.should_load_stream(stream, pos)
        if hasattr(initobj, "file_id") and initobj.file_id is not None and initobj.file_id > 0:
            sqw_fclose(initobj.file_id)
        assert ok
        assert initobj.file_id is not None and initobj.file_id > 0

    def test_should_load_file(self, sample_file):
        to = faccess_sqw_v2()
        ok, initobj, mess = to.should_load(sample_file)
        if hasattr(initobj, "file_id") and initobj.file_id is not None and initobj.file_id > 0:
            sqw_fclose(initobj.file_id)
        assert ok
        assert mess == ""

    def test_init_and_get(self, sample_file):
        to = faccess_sqw_v2()
        ok, initobj, mess = to.should_load(sample_file)
        assert ok
        to = to.init(initobj)

        assert to.num_dim == 2
        assert to.filename == "sqw_2d_1.sqw"
        assert to.data_type == "a"
        assert to.num_contrib_files == 85
        assert list(to.dnd_dimensions) == [16, 11]
        assert to.npixels == 24689

        mhead = to.get_main_header("-verbatim")
        assert mhead.title == ""
        assert mhead.nfiles == 85

        exp_info, pos = to.get_exp_info("-all")
        assert isinstance(exp_info.instruments[0], IX_null_inst)
        assert exp_info.samples[0].alatt is not None
        exp_h = exp_info.expdata
        assert len(exp_h) == 85
        assert isinstance(exp_h[0], IX_experiment)

        det = to.get_detpar()
        assert det[0].ndet == 36864

        data = to.get_data("-ver")
        assert list(data.s.shape) == [16, 11]
        assert list(data.s.shape) == [16, 11]
        assert list(data.e.shape) == [16, 11]
        assert list(data.npix.shape) == [16, 11]

        pix = to.get_pix()
        assert pix.num_pixels == 24689
        to.delete()

    def test_get_pix_range_and_indices(self, sample_file):
        to = faccess_sqw_v2(sample_file)
        pix_all = to.get_pix()
        raw_all = pix_all.data

        pix_part = to.get_raw_pix(1, 20)
        assert np.allclose(raw_all[:, 0:20], pix_part)

        pix_part2 = to.get_raw_pix(10, 50)
        assert np.allclose(raw_all[:, 9:50], pix_part2)

        indices = np.array([1, 5, 10, 100, 24689])
        pix_idx = to.get_pix_at_indices(indices)
        assert np.allclose(raw_all[:, indices - 1], pix_idx)

        starts = np.array([1, 20, 100])
        ends = np.array([5, 25, 110])
        sizes = ends - starts + 1
        pix_ranges = to.get_pix_in_ranges(starts, sizes)
        expected = np.hstack([raw_all[:, 0:5], raw_all[:, 19:25], raw_all[:, 99:110]])
        assert np.allclose(expected, pix_ranges)
        to.delete()

    def test_save_and_load_sqw_v2(self, sample_file, tmp_path):
        to = faccess_sqw_v2(sample_file)
        sqw_obj = to.get_sqw("-verbatim")
        to.delete()

        tf = str(tmp_path / "test_save_load_sqw_v2.sqw")
        fa = faccess_sqw_v2(sqw_obj, tf)
        fa = fa.put_sqw()
        assert os.path.exists(tf)
        fa.delete()

        fa_read = faccess_sqw_v2(tf)
        assert fa_read.npixels == 24689
        assert fa_read.num_contrib_files == 24
        sqw_rec = fa_read.get_sqw("-verbatim")
        fa_read.delete()

        assert sqw_rec.main_header.filename == sqw_obj.main_header.filename
        assert np.allclose(sqw_rec.data.s, sqw_obj.data.s)
        assert np.allclose(sqw_rec.pix.data, sqw_obj.pix.data)

    def test_save_sqw_v2_to_v3(self, sample_file, tmp_path):
        to = faccess_sqw_v2(sample_file)
        tf = str(tmp_path / "test_save_v2_as_v3.sqw")
        v3_fa = faccess_sqw_v3(to, tf)
        v3_fa = v3_fa.put_sqw()
        to.delete()
        v3_fa.delete()

        ldr = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(ldr, faccess_sqw_v3)
        assert ldr.npixels == 24689
        ldr.delete()

    def test_upgrade_v2_to_v4(self, sample_file, tmp_path):
        tf = str(tmp_path / "test_upgrade_v2_to_v4.sqw")
        shutil.copyfile(sample_file, tf)

        to = faccess_sqw_v2(tf)
        sqw_orig = to.get_sqw()
        upg = to.upgrade_file_format()
        assert isinstance(upg, faccess_sqw_v4)
        sqw_upg = upg.get_sqw()
        upg.delete()
        to.delete()

        assert sqw_upg.pix.num_pixels == sqw_orig.pix.num_pixels
        assert np.allclose(sqw_upg.pix.data, sqw_orig.pix.data)
        assert np.allclose(sqw_upg.data.s, sqw_orig.data.s)


# =============================================================================
# test_faccess_sqw_v3.m & test_faccess_sqw_v3_3.m
# =============================================================================
class TestFaccessSqwV3AndV33:
    @pytest.fixture
    def sample_v3_file(self):
        return os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_1.sqw")

    @pytest.fixture
    def sample_v3_3_file(self):
        return os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_3.sqw")

    def test_load_v3(self, sample_v3_file):
        fo = faccess_sqw_v3(sample_v3_file)
        assert fo.faccess_version == 3.1
        assert fo.npixels == 7680
        assert fo.num_contrib_files == 1
        assert fo.num_dim == 4

        sqw_obj = fo.get_sqw()
        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.pix.num_pixels == 7680
        inst = sqw_obj.exper_info.instruments[0]
        assert isinstance(inst, IX_null_inst)
        samp = sqw_obj.exper_info.samples[0]
        assert isinstance(samp, IX_sample)
        fo.delete()

    def test_save_and_load_v3(self, sample_v3_file, tmp_path):
        fo = faccess_sqw_v3(sample_v3_file)
        sqw_obj = fo.get_sqw("-verbatim")
        fo.delete()

        sqw_obj.exper_info.instruments[0] = create_test_instrument()

        tf = str(tmp_path / "test_save_load_v3.sqw")
        fa = faccess_sqw_v3(sqw_obj, tf)
        fa = fa.put_sqw()
        fa.delete()

        fa_read = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(fa_read, faccess_sqw_v3)
        sqw_rec = fa_read.get_sqw("-verbatim")
        fa_read.delete()

        assert np.allclose(sqw_rec.data.s, sqw_obj.data.s)
        assert np.allclose(sqw_rec.pix.data, sqw_obj.pix.data)
        assert sqw_rec.exper_info.instruments[0] == sqw_obj.exper_info.instruments[0]
        assert sqw_rec.exper_info.samples[0] == sqw_obj.exper_info.samples[0]

    def test_load_v3_3(self, sample_v3_3_file):
        fo = faccess_sqw_v3_3(sample_v3_3_file)
        assert fo.faccess_version == 3.3
        assert fo.npixels == 7680
        assert fo.num_contrib_files == 1
        assert fo.num_dim == 4

        sqw_obj = fo.get_sqw()
        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.pix.num_pixels == 7680
        inst = sqw_obj.exper_info.instruments[0]
        assert isinstance(inst, IX_null_inst)
        samp = sqw_obj.exper_info.samples[0]
        assert isinstance(samp, IX_sample)
        fo.delete()

    def test_save_and_load_v3_3(self, sample_v3_3_file, tmp_path):
        fo = faccess_sqw_v3_3(sample_v3_3_file)
        sqw_obj = fo.get_sqw("-verbatim")
        fo.delete()

        sqw_obj.exper_info.instruments[0] = create_test_instrument()

        tf = str(tmp_path / "test_save_load_v3_3.sqw")
        fa = faccess_sqw_v3_3(sqw_obj, tf)
        fa = fa.put_sqw()
        fa.delete()

        fa_read = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(fa_read, faccess_sqw_v3_3)
        sqw_rec = fa_read.get_sqw("-verbatim")
        fa_read.delete()

        assert np.allclose(sqw_rec.data.s, sqw_obj.data.s)
        assert np.allclose(sqw_rec.pix.data, sqw_obj.pix.data)
        assert sqw_rec.exper_info.instruments[0] == sqw_obj.exper_info.instruments[0]
        assert sqw_rec.exper_info.samples[0] == sqw_obj.exper_info.samples[0]

    def test_put_instruments_v3_3(self, sample_v3_3_file, tmp_path):
        tf = str(tmp_path / "test_put_inst_v3_3.sqw")
        shutil.copyfile(sample_v3_3_file, tf)

        fo = faccess_sqw_v3_3(tf)
        new_inst = create_test_instrument()
        fo = fo.reopen_to_write()
        fo = fo.put_instruments(new_inst)
        fo.delete()

        fo_check = faccess_sqw_v3_3(tf)
        inst_rec = fo_check.get_instrument()
        fo_check.delete()
        assert inst_rec == new_inst

    def test_upgrade_v3_to_v4(self, sample_v3_file, tmp_path):
        tf = str(tmp_path / "test_upgrade_v3_to_v4.sqw")
        shutil.copyfile(sample_v3_file, tf)

        fo = faccess_sqw_v3(tf)
        sqw_orig = fo.get_sqw()
        upg = fo.upgrade_file_format()
        assert isinstance(upg, faccess_sqw_v4)

        sqw_upg = upg.get_sqw()
        upg.delete()
        fo.delete()

        assert np.allclose(sqw_upg.data.s, sqw_orig.data.s)
        assert np.allclose(sqw_upg.pix.data, sqw_orig.pix.data)
        assert sqw_upg.exper_info.instruments[0] == sqw_orig.exper_info.instruments[0]

