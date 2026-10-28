import os
import shutil
import numpy as np
import pytest

from pyhorace.core.exceptions import HoraceError
from pyhorace.io import (
    sqw_formats_factory,
    faccess_dnd_v2,
    faccess_dnd_v4,
    faccess_sqw_v2,
    sqw_fclose,
)
from pyhorace.sqw import d1d, d2d, d3d, sqw

COMMON_DATA_DIR = r"c:\Users\abuts\Documents\DEVELOPMENT\Horace_fork\_test\common_data"
TEST_SQW_FILE_DIR = r"c:\Users\abuts\Documents\DEVELOPMENT\Horace_fork\_test\test_sqw_file"


# =============================================================================
# test_faccess_dnd_v2.m
# =============================================================================
class TestFaccessDndV2:
    @pytest.fixture
    def sample_file(self):
        return os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw")

    def test_should_load_stream(self, sample_file):
        to = faccess_dnd_v2()
        assert to.faccess_version == 2
        stream, pos = to.get_file_header(sample_file)
        ok, initobj = to.should_load_stream(stream, pos, nargout=2)
        if hasattr(initobj, "file_id") and initobj.file_id is not None and initobj.file_id > 0:
            sqw_fclose(initobj.file_id)
        assert ok
        assert initobj.file_id is not None
        assert initobj.num_dim == 2

    def test_should_load_file(self, sample_file):
        to = faccess_dnd_v2()
        ok, initobj, mess = to.should_load(sample_file)
        if hasattr(initobj, "file_id") and initobj.file_id is not None and initobj.file_id > 0:
            sqw_fclose(initobj.file_id)
        assert ok
        assert mess == ""
        assert initobj.file_id is not None
        assert initobj.num_dim == 2

    def test_init_Wrong(self):
        to = faccess_dnd_v2()
        with pytest.raises((HoraceError, ValueError, RuntimeError), match="HORACE:horace_binfile_interface:invalid_argument"):
            to.init("wrong_init_param")

    def test_init_and_get(self, sample_file):
        to = faccess_dnd_v2()
        ok, initobj, mess = to.should_load(sample_file)
        assert ok
        assert mess == ""

        to = to.init(initobj)
        assert to.num_dim == 2
        assert os.path.normcase(to.filepath) == os.path.normcase(COMMON_DATA_DIR + os.sep) or os.path.normcase(to.filepath.rstrip("\\/")) == os.path.normcase(COMMON_DATA_DIR)
        assert to.filename == "w2d_qe_d2d.sqw"
        assert to.data_type == "b+"
        assert list(to.dnd_dimensions) == [81, 72]

        data = to.get_data()
        assert data.filename == to.filename
        assert np.allclose(data.s.shape, to.dnd_dimensions)
        assert np.allclose(data.e.shape, to.dnd_dimensions)
        assert np.allclose(data.npix.shape, to.dnd_dimensions)

        d2d_inst = to.get_sqw()
        assert isinstance(d2d_inst, d2d)
        assert d2d_inst.filename == data.filename
        assert np.allclose(d2d_inst.s, data.s)
        assert np.allclose(d2d_inst.e, data.e)
        assert np.array_equal(d2d_inst.npix, data.npix)
        to.delete()

    def test_get_npix_block(self, sample_file):
        to = faccess_dnd_v2(sample_file)
        data = to.get_data()
        npix_tot = data.npix.ravel(order="F")

        block = to.get_npix_block(1, 50)
        assert np.array_equal(block, npix_tot[0:50])

        block2 = to.get_npix_block(10, 100)
        assert np.array_equal(block2, npix_tot[9:100])
        to.delete()

    def test_get_data_Throws(self):
        to = faccess_dnd_v2()
        with pytest.raises((HoraceError, RuntimeError), match="HORACE:binfile_v2_common:runtime_error"):
            to.get_data("-ver")

    def test_get_data_AndSaveDnd(self, sample_file, tmp_path):
        to = faccess_dnd_v2(sample_file)
        assert to.filename == "w2d_qe_d2d.sqw"
        assert to.data_type == "b+"
        assert to.num_dim == 2
        assert list(to.dnd_dimensions) == [81, 72]

        d2d_obj = to.get_sqw("-verbatim")
        assert isinstance(d2d_obj, d2d)
        assert d2d_obj.filename == "ei140.sqw"
        to.delete()

        tf = str(tmp_path / "test_save_load_dnd_v2.sqw")
        so = faccess_dnd_v2(d2d_obj)
        so = so.set_file_to_update(tf)
        so = so.put_sqw()
        assert os.path.exists(tf)
        so.delete()

        check_obj = faccess_dnd_v2(tf)
        assert check_obj.data_type == "b+"
        assert check_obj.num_dim == 2
        assert list(check_obj.dnd_dimensions) == [81, 72]
        assert check_obj.filename == "test_save_load_dnd_v2.sqw"

        d2d_rec = check_obj.get_sqw("-verbatim")
        check_obj.delete()
        assert d2d_rec.filename == "ei140.sqw"
        assert np.allclose(d2d_obj.s, d2d_rec.s)
        assert np.allclose(d2d_obj.e, d2d_rec.e)
        assert np.array_equal(d2d_obj.npix, d2d_rec.npix)

    def test_SaveSqw(self, tmp_path):
        samp_f = os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw")
        so = faccess_sqw_v2(samp_f)
        sqw_ob = so.get_sqw()
        so.delete()

        assert isinstance(sqw_ob, sqw)
        tf = str(tmp_path / "test_save_sqw_as_dnd_v2.sqw")

        tob = faccess_dnd_v2(sqw_ob, tf)
        tob = tob.put_sqw()
        assert os.path.exists(tf)
        tob.delete()

        chob = faccess_dnd_v2(tf)
        d1d_Ver = chob.get_sqw("-verbatim")
        d1d_fname = chob.get_sqw()
        chob.delete()

        assert d1d_fname.filename == "test_save_sqw_as_dnd_v2.sqw"
        assert sqw_ob.data.filename == d1d_Ver.filename
        assert np.allclose(sqw_ob.data.s, d1d_fname.s)

    def test_put_dnd_from_sqw(self, tmp_path):
        samp_f = os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw")
        tf = str(tmp_path / "test_put_dnd_from_sqw_v2.sqw")
        shutil.copyfile(samp_f, tf)

        so = faccess_sqw_v2(tf)
        assert so.sqw_type is True
        assert so.num_dim == 1

        # Save as dnd
        so = so.put_dnd()
        so.delete()

        to = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(to, faccess_dnd_v2)
        assert to.sqw_type is False
        assert to.num_dim == 1
        to.delete()

    def test_put_dnd_data_only(self, sample_file, tmp_path):
        tf = str(tmp_path / "test_put_dnd_data_v2.sqw")
        shutil.copyfile(sample_file, tf)

        to = faccess_dnd_v2(tf)
        d2d_obj = to.get_sqw()
        d2d_obj.s = np.ones_like(d2d_obj.s) * 5.0
        d2d_obj.e = np.ones_like(d2d_obj.e) * 2.0
        d2d_obj.npix = np.ones_like(d2d_obj.npix) * 10

        to = to.reopen_to_write()
        to = to.put_dnd_data(d2d_obj)
        to.delete()

        ch = faccess_dnd_v2(tf)
        d2d_rec = ch.get_sqw()
        ch.delete()
        assert np.allclose(d2d_rec.s, 5.0)
        assert np.allclose(d2d_rec.e, 2.0)
        assert np.all(d2d_rec.npix == 10)

    def test_upgrade_file_format(self, sample_file, tmp_path):
        tf = str(tmp_path / "test_upgrade_dnd_v2_to_v4.sqw")
        shutil.copyfile(sample_file, tf)

        to = faccess_dnd_v2(tf)
        d2d_orig = to.get_sqw()
        new_obj = to.upgrade_file_format()
        assert isinstance(new_obj, faccess_dnd_v4)
        assert new_obj.faccess_version == 4.0

        d2d_upg = new_obj.get_sqw()
        new_obj.delete()
        to.delete()

        assert np.allclose(d2d_orig.s, d2d_upg.s)
        assert np.allclose(d2d_orig.e, d2d_upg.e)
        assert np.array_equal(d2d_orig.npix, d2d_upg.npix)


# =============================================================================
# test_faccess_dnd_v4.m
# =============================================================================
class TestFaccessDndV4:
    @pytest.fixture
    def sample_v2_file(self):
        return os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw")

    @pytest.fixture
    def sample_v4_file(self):
        return os.path.join(TEST_SQW_FILE_DIR, "faccess_dnd_v4_sample.sqw")

    def test_load_v4_sample(self, sample_v4_file):
        if not os.path.exists(sample_v4_file):
            pytest.skip("faccess_dnd_v4_sample.sqw not present")
        ldr = sqw_formats_factory.instance().get_loader(sample_v4_file)
        assert isinstance(ldr, faccess_dnd_v4)
        assert ldr.faccess_version == 4.0
        assert ldr.sqw_type is False
        assert ldr.num_dim == 2

        d2d_obj = ldr.get_sqw()
        assert isinstance(d2d_obj, d2d)
        ldr.delete()

    def test_save_and_load_dnd_v4(self, sample_v2_file, tmp_path):
        ldr = faccess_dnd_v2(sample_v2_file)
        d2d_orig = ldr.get_sqw("-verbatim")
        ldr.delete()

        tf = str(tmp_path / "test_save_load_dnd_v4.sqw")
        fa = faccess_dnd_v4(d2d_orig, tf)
        fa = fa.put_sqw()
        assert os.path.exists(tf)
        fa.delete()

        fa_read = sqw_formats_factory.instance().get_loader(tf)
        assert isinstance(fa_read, faccess_dnd_v4)
        assert fa_read.faccess_version == 4.0
        assert fa_read.sqw_type is False
        assert fa_read.num_dim == 2
        assert list(fa_read.dnd_dimensions) == [81, 72]

        d2d_rec = fa_read.get_sqw("-verbatim")
        assert d2d_rec.filename == d2d_orig.filename
        assert d2d_rec.title == d2d_orig.title
        assert np.allclose(d2d_rec.s, d2d_orig.s)
        assert np.allclose(d2d_rec.e, d2d_orig.e)
        assert np.array_equal(d2d_rec.npix, d2d_orig.npix)

        d2d_upd_name = fa_read.get_sqw()
        assert d2d_upd_name.filename == "test_save_load_dnd_v4.sqw"
        fa_read.delete()

    def test_get_npix_block_v4(self, sample_v2_file, tmp_path):
        ldr = faccess_dnd_v2(sample_v2_file)
        d2d_orig = ldr.get_sqw()
        ldr.delete()

        tf = str(tmp_path / "test_npix_block_v4.sqw")
        fa = faccess_dnd_v4(d2d_orig, tf)
        fa = fa.put_sqw()

        npix_tot = d2d_orig.npix.ravel(order="F")
        block = fa.get_npix_block(1, 50)
        assert np.array_equal(block, npix_tot[0:50])

        block2 = fa.get_npix_block(10, 100)
        assert np.array_equal(block2, npix_tot[9:100])
        fa.delete()

    def test_put_dnd_metadata_and_data_v4(self, sample_v2_file, tmp_path):
        ldr = faccess_dnd_v2(sample_v2_file)
        d2d_orig = ldr.get_sqw()
        ldr.delete()

        tf = str(tmp_path / "test_put_meta_data_v4.sqw")
        fa = faccess_dnd_v4(d2d_orig, tf)
        fa = fa.put_sqw()

        # Modify metadata
        d2d_orig.title = "New V4 DND Title"
        fa = fa.put_dnd_metadata(d2d_orig)

        # Modify data
        d2d_orig.s = np.ones_like(d2d_orig.s) * 7.5
        fa = fa.put_dnd_data(d2d_orig)
        fa.delete()

        fa_check = faccess_dnd_v4(tf)
        d2d_rec = fa_check.get_sqw()
        fa_check.delete()
        assert d2d_rec.title == "New V4 DND Title"
        assert np.allclose(d2d_rec.s, 7.5)

