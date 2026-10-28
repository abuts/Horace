import os
import copy
import struct
import numpy as np
import pytest

from pyhorace.core.exceptions import HoraceError
from pyhorace.io import (
    data_block,
    dnd_data_block,
    pix_data_block,
    blockAllocationTable,
    binfile_v4_common,
    sqw_formats_factory,
    faccess_sqw_v4,
    faccess_dnd_v4,
)
from pyhorace.io.sqw_fopen import sqw_fopen, do_fclose
from pyhorace.sqw import sqw, d2d, d4d
from pyhorace.sqw.dnd_data import dnd_data
from pyhorace.data.pixels.pix_data import pix_data

COMMON_DATA_DIR = r"c:\Users\abuts\Documents\DEVELOPMENT\Horace_fork\_test\common_data"
TEST_SQW_FILE_DIR = r"c:\Users\abuts\Documents\DEVELOPMENT\Horace_fork\_test\test_sqw_file"


class _BinfileV4BlockTester(binfile_v4_common):
    """Port of ``binfile_v4_block_tester.m``."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.num_dim_ = 2
        if args or kwargs:
            self.init(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 4.0

    def get_data_blocks(self):
        return [
            data_block("", "level2_a"),
            data_block("level1", "level2_b"),
            data_block("level1_c", "level2_c"),
            data_block("level1_d", "level2_d"),
        ]

    def get_data_type(self):
        return "a"

    def get_sqw_type(self):
        return False

    def get_pix_range(self, *args):
        return np.empty((0, 4))

    def get_instrument(self, *args, **kwargs):
        return None

    def get_sample(self, *args, **kwargs):
        return None

    def get_sqw(self, *args, **kwargs):
        return self.get_all_blocks(*args, **kwargs)

    def put_sqw(self, *args, **kwargs):
        return self.put_all_blocks(*args, **kwargs)


class _BinfileV4CommonTester(binfile_v4_common):
    """Port of ``binfile_v4_common_tester.m``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 4.0

    def get_sqw_type(self) -> bool:
        return False

    def get_data_blocks(self):
        return [
            data_block("data", "metadata"),
            dnd_data_block(),
        ]

    def get_data_type(self):
        return "b+"

    def get_pix_range(self, *args):
        return np.empty((0, 4))

    def get_instrument(self, *args, **kwargs):
        return None

    def get_sample(self, *args, **kwargs):
        return None

    def get_sqw(self, *args, **kwargs):
        return self.get_dnd(*args, **kwargs)

    def put_sqw(self, *args, **kwargs):
        return self.put_dnd(*args, **kwargs)


# =============================================================================
# test_data_block.m
# =============================================================================
class TestDataBlock:
    def test_pix_block_serialization(self):
        pdb = pix_data_block()
        pdb.position = 100
        pdb.npixels = 1000
        assert pdb.size == 1000 * 9 * 4 + 12
        assert pdb.n_rows == 9

        rec = pdb.bat_record
        pdb_rec, pos = data_block.deserialize_bat_record(rec)
        assert isinstance(pdb_rec, pix_data_block)
        assert pdb_rec.position == 100
        assert pdb_rec.size == pdb.size
        assert pdb_rec.npixels == 1000
        assert pdb_rec.n_rows == 9
        assert pdb_rec.locked is True

    def test_pix_block_properties(self):
        pdb = pix_data_block(100, 50)
        assert pdb.position == 100
        assert pdb.npixels == 50
        assert pdb.n_rows == 9
        assert pdb.size == 12 + 9 * 4 * 50

        pdb.n_rows = 10
        assert pdb.size == 12 + 10 * 4 * 50

        with pytest.raises(HoraceError, match="HORACE:pix_data_block:invalid_argument"):
            pdb.n_rows = 2000

    def test_dnd_data_block_serialization(self):
        ddb = dnd_data_block(200, 3 * 8 * 20)
        assert ddb.position == 200
        assert ddb.size == 3 * 8 * 20
        assert ddb.locked is False

        rec = ddb.bat_record
        ddb_rec, pos = data_block.deserialize_bat_record(rec)
        assert isinstance(ddb_rec, dnd_data_block)
        assert ddb_rec.position == 200
        assert ddb_rec.size == 3 * 8 * 20
        assert ddb_rec.sqw_prop_name == "data"
        assert ddb_rec.level2_prop_name == "nd_data"

    def test_data_block_serialization(self):
        db = data_block("det_par", "", 50, 500)
        assert db.block_name == "bl_det_par_"
        assert db.position == 50
        assert db.size == 500
        assert db.locked is False

        db.locked = True
        assert db.locked is True
        rec = db.bat_record
        assert len(rec) == db.bat_record_size
        db_rec, pos = data_block.deserialize_bat_record(rec)
        assert isinstance(db_rec, data_block)
        assert db_rec.block_name == "bl_det_par_"
        assert db_rec.sqw_prop_name == "det_par"
        assert db_rec.level2_prop_name == ""
        assert db_rec.position == 50
        assert db_rec.size == 500
        assert db_rec.locked is False  # locked is transient, not serialized in bat_record

    def test_data_block_get_set_subobj(self):
        db = data_block("main_header", "title")
        dummy = {"main_header": {"title": "hello_world", "nfiles": 5}}
        sub = db.get_subobj(dummy)
        assert sub == "hello_world"

        dummy = db.set_subobj(dummy, "new_title")
        assert dummy["main_header"]["title"] == "new_title"

    def test_dnd_data_block_get_set_subobj(self):
        ddb = dnd_data_block()
        s = np.ones((4, 5), dtype=np.float64)
        e = np.ones((4, 5), dtype=np.float64) * 2.0
        npix = np.ones((4, 5), dtype=np.uint64) * 3
        nd = dnd_data(s, e, npix)
        dummy = {"data": {"nd_data": nd}}

        sub = ddb.get_subobj(dummy)
        assert isinstance(sub, dnd_data)
        assert np.array_equal(sub.sig, s)
        assert np.array_equal(sub.err, e)
        assert np.array_equal(sub.npix, npix)

        ddb.calc_obj_size(dummy)
        assert ddb.size == 4 + 2 * 4 + 4 * 5 * 3 * 8

    def test_put_get_pix_data_block(self, tmp_path):
        f_sqw = os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw")
        to = sqw(f_sqw)

        dp1 = data_block("pix", "metadata")
        dp1 = dp1.calc_obj_size(to)

        dp2 = pix_data_block()
        dp2 = dp2.calc_obj_size(to)
        dp2.position = dp1.size

        test_file = str(tmp_path / "test_put_get_pix_data_block.bin")
        fid = sqw_fopen(test_file, "wb+")
        try:
            dp1 = dp1.put_sqw_block(fid)
            dp2 = dp2.put_sqw_block(fid, to)

            rec_obj = sqw()
            _, rec_obj = dp1.get_sqw_block(fid, rec_obj, nargout=2)
            _, rec_obj = dp2.get_sqw_block(fid, rec_obj, nargout=2)
            assert to.pix.metadata == rec_obj.pix.metadata
            assert np.allclose(to.pix.data, rec_obj.pix.data)
        finally:
            do_fclose(fid)

    def test_put_get_dnd_info(self, tmp_path):
        f_sqw = os.path.join(COMMON_DATA_DIR, "sqw_2d_1.sqw")
        to = sqw(f_sqw)

        dp1 = data_block("data", "metadata")
        dp1 = dp1.calc_obj_size(to)

        dp2 = dnd_data_block()
        dp2 = dp2.calc_obj_size(to)
        dp2.position = dp1.size

        test_file = str(tmp_path / "test_put_get_dnd_info.bin")
        fid = sqw_fopen(test_file, "wb+")
        try:
            dp1 = dp1.put_sqw_block(fid)
            dp2 = dp2.put_sqw_block(fid)

            rec_obj = sqw()
            _, rec_obj = dp1.get_sqw_block(fid, rec_obj, nargout=2)
            assert to.data.metadata == rec_obj.data.metadata

            _, rec_obj = dp2.get_sqw_block(fid, rec_obj, nargout=2)
            assert to.data.nd_data == rec_obj.data.nd_data
        finally:
            do_fclose(fid)


# =============================================================================
# test_block_allocation_table.m
# =============================================================================
class TestBlockAllocationTable:
    def test_empty_bat_construction(self):
        bat = blockAllocationTable()
        assert not bat.initialized
        assert bat.blocks_start_position == 4
        assert bat.end_of_file_pos == 0
        assert len(bat.block_names) == 0

        with pytest.raises(HoraceError, match="HORACE:blockAllocationTable:invalid_argument"):
            bat.blocks_list = "a"

        with pytest.raises(HoraceError, match="HORACE:blockAllocationTable:invalid_argument"):
            bat.blocks_list = ["a", "b"]

        with pytest.raises(HoraceError, match="HORACE:blockAllocationTable:invalid_argument"):
            bat.position = "a"

        with pytest.raises(HoraceError, match="HORACE:blockAllocationTable:invalid_argument"):
            bat.position = -10

        with pytest.raises(HoraceError, match="HORACE:blockAllocationTable:invalid_argument"):
            bat.ba_table = "a"

    def test_bat_construction_with_blocks(self):
        blocks = [
            data_block("main_header", ""),
            data_block("header", "instruments"),
            dnd_data_block(),
            pix_data_block(),
        ]
        bat = blockAllocationTable(64, blocks)
        assert not bat.initialized
        assert len(bat.blocks_list) == 4
        assert bat.block_names == [
            "bl_main_header_",
            "bl_header_instruments",
            "bl_data_nd_data",
            "bl_pix_data_wrap",
        ]
        assert bat.blocks_start_position > 64

    def test_bat_init_obj_info_and_serialize(self):
        f_sqw = os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")
        sqw_obj = sqw(f_sqw)

        fa = faccess_sqw_v4()
        blocks = fa.get_data_blocks()
        bat = blockAllocationTable(37, blocks)
        bat = bat.init_obj_info(sqw_obj)
        assert bat.initialized
        assert bat.end_of_file_pos > bat.blocks_start_position

        bin_data = bat.ba_table
        assert len(bin_data) + 4 == bat.bat_bin_size

        bat2 = blockAllocationTable(37, [])
        bat2.ba_table = bin_data
        assert bat2.initialized
        assert bat2.blocks_start_position == bat.blocks_start_position
        assert bat2.end_of_file_pos == bat.end_of_file_pos
        assert bat2.block_names == bat.block_names

        for b1, b2 in zip(bat.blocks_list, bat2.blocks_list):
            assert b1.block_name == b2.block_name
            assert b1.position == b2.position
            assert b1.size == b2.size

    def test_bat_free_spaces_and_find_block_place(self):
        blocks_uninit = [
            data_block("main_header", ""),
            data_block("det_par", ""),
            dnd_data_block(),
        ]
        bat0 = blockAllocationTable(0, blocks_uninit)
        start_pos = bat0.blocks_start_position
        blocks = [
            data_block("main_header", "", start_pos, 100),
            data_block("det_par", "", start_pos + 100, 200),
            dnd_data_block(start_pos + 300, 300),
        ]
        bat = blockAllocationTable(0, blocks)
        assert bat.initialized
        assert bat.end_of_file_pos == start_pos + 600

        # Replace middle block with a smaller one (size 100 instead of 200)
        new_det_bl = data_block("det_par", "", 0, 100)
        bat, new_pos, compress = bat.find_block_place(new_det_bl, 100)
        assert not compress
        assert new_pos == start_pos + 100
        placed_bl = bat.get_data_block("bl_det_par_", nargout=1)
        assert placed_bl.position == start_pos + 100
        assert placed_bl.size == 100
        # Free space of 100 bytes at start_pos + 200 should now exist
        assert bat.free_spaces_and_size.shape == (2, 1)
        assert bat.free_spaces_and_size[0, 0] == start_pos + 200
        assert bat.free_spaces_and_size[1, 0] == 100

        # Replace with a larger block (size 250 > 200) -> goes to end of file (start_pos + 600), leaving 200 bytes free at start_pos + 100
        larger_det_bl = data_block("det_par", "", 0, 250)
        bat, new_pos, compress = bat.find_block_place(larger_det_bl, 250)
        assert new_pos == start_pos + 600
        placed_bl = bat.get_data_block("bl_det_par_", nargout=1)
        assert placed_bl.position == start_pos + 600
        assert placed_bl.size == 250
        assert bat.end_of_file_pos == start_pos + 850
        # Merged free space at start_pos + 100 of size 100 + 100 = 200!
        assert bat.free_spaces_and_size.shape == (2, 1)
        assert bat.free_spaces_and_size[0, 0] == start_pos + 100
        assert bat.free_spaces_and_size[1, 0] == 200

    def test_bat_set_data_block_overlap_throws(self):
        blocks = [
            data_block("main_header", "", 200, 100),
            data_block("det_par", "", 300, 200),
        ]
        bat = blockAllocationTable(0, blocks)

        # Try setting main_header to overlap with det_par (pos 250, size 100 -> ends at 349 >= 300)
        bad_bl = data_block("main_header", "", 250, 100)
        with pytest.raises(HoraceError, match="overlaps with location of existing blocks"):
            bat.set_data_block(bad_bl)


# =============================================================================
# test_binfile_v4_common.m
# =============================================================================
class TestBinfileV4Common:
    def test_serialize_deserialize_faccess(self):
        f_sqw = os.path.join(COMMON_DATA_DIR, "sqw_1d_2.sqw")
        sqw_ob = sqw(f_sqw)

        acc = _BinfileV4CommonTester(sqw_ob.data)
        assert acc.bat.initialized

        dat = acc.serialize()
        rec, _ = _BinfileV4CommonTester.deserialize(dat)
        assert acc == rec

    def test_init_and_put_get_blocks(self, tmp_path):
        test_obj = {
            "level2_a": "Some string within",
            "level1": {"level2_b": np.array([1, 2, 3, 4], dtype=np.float64)},
            "level1_c": {"level2_c": [10, 20]},
            "level1_d": {"level2_d": 42.0},
        }
        test_file = str(tmp_path / "binfile_v4_put_get_blocks.bin")
        blk_io = _BinfileV4BlockTester(test_obj, test_file)
        assert blk_io.bat.initialized

        blk_io = blk_io.put_all_blocks()
        assert os.path.isfile(test_file)

        _, rec_obj = blk_io.get_all_blocks()
        assert rec_obj["level2_a"] == test_obj["level2_a"]
        assert np.array_equal(rec_obj["level1"]["level2_b"], test_obj["level1"]["level2_b"])
        assert rec_obj["level1_c"]["level2_c"] == test_obj["level1_c"]["level2_c"]
        assert rec_obj["level1_d"]["level2_d"] == test_obj["level1_d"]["level2_d"]
        blk_io.delete()

    def test_dnd_v4_blocks_read_write(self, tmp_path):
        f_dnd = os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw")
        ldr = sqw_formats_factory.instance().get_loader(f_dnd)
        d2d_obj = ldr.get_sqw()
        ldr.delete()

        test_file = str(tmp_path / "test_dnd_v4_blocks.sqw")
        fa = faccess_dnd_v4(d2d_obj, test_file)
        fa = fa.put_sqw()
        fa.delete()

        fa_read = faccess_dnd_v4(test_file)
        d2d_rec = fa_read.get_sqw()
        assert d2d_rec.title == d2d_obj.title
        assert np.allclose(d2d_rec.s, d2d_obj.s)
        assert np.allclose(d2d_rec.e, d2d_obj.e)
        assert np.array_equal(d2d_rec.npix, d2d_obj.npix)

        # Modify a single block in place
        d2d_rec.title = "Updated title in place via put_sqw_block"
        fa_upd = fa_read.reopen_to_write()
        fa_upd = fa_upd.put_sqw_block("bl_data_metadata", d2d_rec)
        fa_upd.delete()

        fa_check = faccess_dnd_v4(test_file)
        _, meta = fa_check.get_sqw_block("bl_data_metadata")
        assert meta.title == "Updated title in place via put_sqw_block"
        fa_check.delete()
