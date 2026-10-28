import os
import shutil
import numpy as np
import pytest

from pyhorace.io import (
    sqw_formats_factory,
    faccess_sqw_v4,
    faccess_sqw_v2,
    faccess_sqw_v3_3,
    faccess_dnd_v4,
)
from pyhorace.io.sqw_fopen import sqw_fopen
from pyhorace.sqw import sqw, d1d, d2d
from pyhorace.data.instr_classes import (
    IX_experiment,
    IX_null_inst,
    IX_sample,
    IX_inst_DGfermi,
    IX_moderator,
    IX_aperture,
    IX_fermi_chopper,
)
from pathlib import Path
from pyhorace.data.pixels.pix_metadata import pix_metadata

HORACE_ROOT = Path(__file__).resolve().parents[3]
COMMON_DATA_DIR = str(HORACE_ROOT / "_test" / "common_data")
TEST_SQW_FILE_DIR = str(HORACE_ROOT / "_test" / "test_sqw_file")


def create_test_instrument():
    mod = IX_moderator(12.0, 2.5, "ikcarp", [11.0, 22.0, 0.6])
    ap = IX_aperture(-9.0, 0.06, 0.06)
    fc = IX_fermi_chopper(1.6, 500.0, 0.06, 1.4, 0.002)
    return IX_inst_DGfermi(mod, ap, fc, 100.0, name="test_fermi_v4")


class TestFaccessSqwV4:
    @pytest.fixture
    def sample_v4_file(self):
        return os.path.join(COMMON_DATA_DIR, "faccess_sqw_v4_sample.sqw")

    @pytest.fixture
    def sample_v2_file(self):
        return os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")

    def test_load_v4_sample(self, sample_v4_file):
        fo = sqw_formats_factory.instance().get_loader(sample_v4_file)
        assert isinstance(fo, faccess_sqw_v4)
        assert fo.faccess_version == 4.0
        assert fo.sqw_type is True
        assert fo.num_dim == 2
        assert fo.npixels == 4324
        assert fo.num_contrib_files == 109
        assert fo.file_id > 0

        sqw_obj = fo.get_sqw()
        assert isinstance(sqw_obj, sqw)
        assert sqw_obj.pix.num_pixels == 4324
        assert sqw_obj.exper_info.n_runs == 109
        fo.delete()

    def test_save_and_load_sqw_v4(self, sample_v4_file, tmp_path):
        fo = faccess_sqw_v4(sample_v4_file)
        sqw_orig = fo.get_sqw("-verbatim")
        fo.delete()

        tf = str(tmp_path / "test_save_load_v4.sqw")
        fa = faccess_sqw_v4(sqw_orig, tf)
        fa = fa.put_sqw()
        assert os.path.exists(tf)
        fa.delete()

        fa_read = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(fa_read, faccess_sqw_v4)
        assert fa_read.npixels == 4324
        assert fa_read.num_contrib_files == 109
        assert fa_read.num_dim == 2

        sqw_rec = fa_read.get_sqw("-verbatim")
        fa_read.delete()

        assert sqw_rec.main_header.title == sqw_orig.main_header.title
        assert sqw_rec.main_header.filename == sqw_orig.main_header.filename
        assert np.allclose(sqw_rec.data.s, sqw_orig.data.s)
        assert np.allclose(sqw_rec.data.e, sqw_orig.data.e)
        assert np.array_equal(sqw_rec.data.npix, sqw_orig.data.npix)
        assert np.allclose(sqw_rec.pix.data, sqw_orig.pix.data)
        assert np.allclose(sqw_rec.pix.data_range, sqw_orig.pix.data_range)

    def test_get_parts_v4(self, sample_v4_file):
        fo = faccess_sqw_v4(sample_v4_file)
        mh = fo.get_main_header()
        assert mh.nfiles == 109
        assert mh.filename == "faccess_sqw_v4_sample.sqw"

        exp_info = fo.get_exp_info("-all")
        assert exp_info.n_runs == 109

        inst = fo.get_instrument(1)
        assert inst is not None
        inst_all = fo.get_instrument("-all")
        assert inst_all.n_runs == 109

        samp = fo.get_sample(1)
        assert samp is not None
        samp_all = fo.get_sample("-all")
        assert samp_all.n_runs == 109

        det = fo.get_detpar()
        assert "group" in det

        pmeta = fo.get_pix_metadata()
        assert isinstance(pmeta, pix_metadata)
        assert pmeta.npix == 4324

        pix = fo.get_pix()
        assert pix.num_pixels == 4324

        raw_part = fo.get_pix(1, 50)
        assert np.allclose(pix.data[:, 0:50], raw_part)

        indices = np.array([1, 10, 100, 4324])
        raw_idx = fo.get_pix_at_indices(indices)
        assert np.allclose(pix.data[:, indices - 1], raw_idx)

        starts = np.array([1, 50, 200])
        ends = np.array([10, 60, 210])
        sizes = ends - starts + 1
        raw_ranges = fo.get_pix_in_ranges(starts, sizes)
        expected = np.hstack([pix.data[:, 0:10], pix.data[:, 49:60], pix.data[:, 199:210]])
        assert np.allclose(expected, raw_ranges)
        fo.delete()

    def test_put_subblocks_v4(self, sample_v4_file, tmp_path):
        tf = str(tmp_path / "test_put_subblocks_v4.sqw")
        shutil.copyfile(sample_v4_file, tf)

        fo = faccess_sqw_v4(tf)
        sqw_obj = fo.get_sqw()

        # 1. Update main header
        sqw_obj.main_header.title = "Updated V4 Main Header Title"
        fo = fo.reopen_to_write()
        fo = fo.put_main_header(sqw_obj)

        # 2. Update instruments
        new_inst = create_test_instrument()
        fo = fo.put_instruments(new_inst)

        # 3. Update samples
        new_samp = IX_sample(alatt=[4.0, 4.0, 4.0], angdeg=[90.0, 90.0, 90.0], name="test_samp_v4")
        fo = fo.put_samples(new_samp)

        # 4. Update pix_metadata
        pmeta = fo.get_pix_metadata()
        new_range = np.ones((2, 9), dtype=np.float64)
        new_range[0, :] = -10.0
        new_range[1, :] = 10.0
        pmeta.data_range = new_range
        fo = fo.put_pix_metadata(pmeta)
        fo.delete()

        # Verify updates
        fo_check = faccess_sqw_v4(tf)
        mh_rec = fo_check.get_main_header()
        assert mh_rec.title == "Updated V4 Main Header Title"

        inst_rec = fo_check.get_instrument(1)
        assert inst_rec == new_inst

        samp_rec = fo_check.get_sample(1)
        assert samp_rec == new_samp

        pmeta_rec = fo_check.get_pix_metadata()
        assert np.allclose(pmeta_rec.data_range, new_range)
        fo_check.delete()

    def test_put_dnd_from_sqw_v4(self, sample_v4_file, tmp_path):
        fo = faccess_sqw_v4(sample_v4_file)
        sqw_obj = fo.get_sqw()
        fo.delete()

        tf = str(tmp_path / "test_put_dnd_from_v4.sqw")
        fa = faccess_dnd_v4(sqw_obj, tf)
        fa = fa.put_dnd()
        fa.delete()

        ldr = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(ldr, faccess_dnd_v4)
        assert ldr.sqw_type is False
        assert ldr.num_dim == 2
        d1d_rec = ldr.get_sqw()
        ldr.delete()

        assert np.allclose(d1d_rec.s, sqw_obj.data.s)
        assert np.allclose(d1d_rec.e, sqw_obj.data.e)
        assert np.array_equal(d1d_rec.npix, sqw_obj.data.npix)

    def test_destructor_closes_file(self, sample_v4_file):
        fo = faccess_sqw_v4(sample_v4_file)
        fid = fo.file_id
        assert fid > 0 and sqw_fopen(fid) != ""
        del fo
        assert sqw_fopen(fid) == ""

