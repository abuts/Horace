import os
import shutil
import copy
import numpy as np
import pytest
import scipy.io as sio

from pyhorace.io import (
    sqw_formats_factory,
    horace_binfile_interface,
    const_blocks_map,
    binfile_v2_common,
    sqw_binfile_common,
    faccess_dnd_v2,
    faccess_dnd_v4,
    faccess_sqw_prototype,
    faccess_sqw_v2,
    faccess_sqw_v3,
    faccess_sqw_v3_3,
    faccess_sqw_v4,
)
from pyhorace.sqw import sqw, d0d, d1d, d2d, d3d, read_dnd, read_sqw, read_horace
from pyhorace.data.instr_classes import IX_detector_array
from pathlib import Path
from pyhorace.core.unique_objects_container import unique_objects_container

HORACE_ROOT = Path(__file__).resolve().parents[3]
COMMON_DATA_DIR = str(HORACE_ROOT / "_test" / "common_data")
TEST_SQW_FILE_DIR = str(HORACE_ROOT / "_test" / "test_sqw_file")


class _BinfileV2CommonTester(binfile_v2_common):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_faccess_version(self) -> float:
        return 2.0

    def get_sqw_type(self) -> bool:
        return False

    def set_datatype(self, val: str):
        self.data_type_ = val
        return self

    def get_file_header(self, file, *args):
        return binfile_v2_common.get_file_header(file, *args)

    def get_dnd_form(self, *args):
        return super().get_dnd_form(*args)

    def extract_field_range(self, pos_fields, form_fields):
        return binfile_v2_common.extract_field_range(pos_fields, form_fields)

    def set_file_to_update(self, filename=None):
        return super().set_file_to_update(filename)


class _SqwBinfileCommonTester(sqw_binfile_common):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._faccess_version = 100.0
        if len(args) > 0:
            self.init(*args, **kwargs)

    def get_main_header_form(self, *args):
        return super().get_main_header_form(*args)

    def get_header_form(self, *args):
        return super().get_header_form(*args)

    def get_detpar_form(self, *args):
        return super().get_detpar_form(*args)

    def get_dnd_form(self, *args):
        return super().get_dnd_form(*args)


class _ConstBlocksMapTester(const_blocks_map):
    pass


# =============================================================================
# test_sqw_formats_factory.m
# =============================================================================
class TestSqwFormatsFactory:
    def test_pref_accessor_d0d(self):
        fa = sqw_formats_factory.instance().get_pref_access()
        assert isinstance(fa, faccess_sqw_v4)

        fa = sqw_formats_factory.instance().get_pref_access(d0d())
        assert isinstance(fa, faccess_dnd_v4)

        try:
            sqw_formats_factory.instance().set_pref_access(d0d(), faccess_dnd_v2())
            fa = sqw_formats_factory.instance().get_pref_access(d0d())
            assert isinstance(fa, faccess_dnd_v2)

            sqw_formats_factory.instance().set_pref_access(d0d(), "")
            fa = sqw_formats_factory.instance().get_pref_access(d0d())
            assert isinstance(fa, faccess_dnd_v4)
        finally:
            sqw_formats_factory.instance().set_pref_access(d0d(), "")

    def test_selection_v3_3(self):
        file_v3_3 = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_3.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v3_3)
        assert isinstance(fo, faccess_sqw_v3_3)
        assert fo.faccess_version == 3.3
        assert fo.file_closer_ > 0
        assert fo.npixels == 7680
        fo.delete()

    def test_selection_v2(self):
        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v2)
        assert isinstance(fo, faccess_sqw_v2)
        assert fo.faccess_version == 2
        assert fo.file_closer_ > 0
        assert fo.npixels == 8031
        fo.delete()

    def test_selection_v3_old(self):
        file_v3_old = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v3_old)
        assert isinstance(fo, faccess_sqw_v2)
        assert fo.faccess_version == 2
        assert fo.file_closer_ > 0
        assert fo.npixels == 7680
        fo.delete()

    def test_selection_v3(self):
        file_v3 = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_1.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v3)
        assert isinstance(fo, faccess_sqw_v3)
        assert fo.faccess_version == 3.1
        assert fo.file_closer_ > 0
        assert fo.npixels == 7680
        fo.delete()

    def test_wrong_selection_wrong_format(self):
        file_non_hor = os.path.join(TEST_SQW_FILE_DIR, "pos_to_test.mat")
        with pytest.raises(RuntimeError, match="HORACE:horace_binfile_interface:runtime_error"):
            sqw_formats_factory.instance().get_loader(file_non_hor)

    def test_selection_v0(self):
        file_v0 = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_read_write_v0_t.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v0)
        assert isinstance(fo, faccess_sqw_prototype)
        assert fo.faccess_version == 0
        assert fo.file_closer_ > 0
        assert fo.npixels == 16
        fo.delete()

    def test_selection_dnd(self):
        file_dnd = os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_dnd)
        assert isinstance(fo, faccess_dnd_v2)
        assert fo.faccess_version == 2
        assert fo.file_closer_ > 0
        assert fo.sqw_type is False
        assert fo.data_type == "b+"
        assert fo.num_dim == 2
        assert list(fo.dnd_dimensions) == [81, 72]
        fo.delete()

    def test_wrong_selection_wrong_version(self):
        file_wrong_ver = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_fictional_ver.sqw")
        with pytest.raises(RuntimeError, match="HORACE:file_io:runtime_error"):
            sqw_formats_factory.instance().get_loader(file_wrong_ver)

    def test_selection_v1(self):
        file_v1 = os.path.join(TEST_SQW_FILE_DIR, "w2_small_v1.sqw")
        fo = sqw_formats_factory.instance().get_loader(file_v1)
        assert isinstance(fo, faccess_sqw_v2)
        assert fo.faccess_version == 2
        assert fo.file_closer_ > 0
        assert fo.npixels == 179024
        fo.delete()

    def test_pref_access(self):
        obj1 = sqw()
        obj2 = d1d()
        fo1 = sqw_formats_factory.instance().get_pref_access(obj1)
        fo2 = sqw_formats_factory.instance().get_pref_access(obj2)
        assert isinstance(fo1, faccess_sqw_v4)
        assert isinstance(fo2, faccess_dnd_v4)

    def test_load_range(self):
        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw")
        file_v3 = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_1.sqw")
        files = [file_v2, file_v3]
        loaders = sqw_formats_factory.instance().get_loader(files)
        assert isinstance(loaders[0], faccess_sqw_v2)
        assert isinstance(loaders[1], faccess_sqw_v3)
        for ldr in loaders:
            ldr.delete()


# =============================================================================
# test_horace_binfile_interface.m
# =============================================================================
class TestHoraceBinfileInterface:
    @pytest.fixture
    def det_struct(self):
        return {
            "filename": "",
            "filepath": "",
            "group": np.array([1.0]),
            "x2": np.array([2.0]),
            "phi": np.array([3.0]),
            "azim": np.array([4.0]),
            "width": np.array([5.0]),
            "height": np.array([6.0]),
        }

    def test_convert_old_det_forms_container_det_arrays(self, det_struct):
        det = IX_detector_array(det_struct)
        uoc = unique_objects_container(baseclass="IX_detector_array")
        uoc, _ = uoc.add(det)
        uoc = uoc.replicate_runs(3)
        new_dets = horace_binfile_interface.convert_old_det_forms(uoc, 3)
        assert isinstance(new_dets, unique_objects_container)
        assert new_dets.n_objects == 3
        assert new_dets.n_unique == 1
        assert new_dets[0] == det

    def test_convert_old_det_forms_det_arrays_single_with_n(self, det_struct):
        det = IX_detector_array(det_struct)
        new_dets = horace_binfile_interface.convert_old_det_forms(det, 3)
        assert isinstance(new_dets, unique_objects_container)
        assert new_dets.n_objects == 3
        assert new_dets.n_unique == 1
        assert new_dets[0] == det

    def test_convert_old_det_forms_structs_single(self, det_struct):
        det = IX_detector_array(det_struct)
        new_dets = horace_binfile_interface.convert_old_det_forms(det_struct)
        assert isinstance(new_dets, unique_objects_container)
        assert new_dets.n_objects == 1
        assert new_dets.n_unique == 1
        assert new_dets[0] == det

    def test_convert_old_det_forms_container_n_not_matching(self, det_struct):
        det = IX_detector_array(det_struct)
        uoc = unique_objects_container(baseclass="IX_detector_array")
        uoc, _ = uoc.add(det)
        uoc = uoc.replicate_runs(2)
        with pytest.raises((ValueError, RuntimeError), match="HORACE:horace_binfile_interface:invalid_argument"):
            horace_binfile_interface.convert_old_det_forms(uoc, 4)

    def test_convert_old_det_forms_null_inputs(self):
        detr = unique_objects_container(baseclass="IX_detector_array")
        deta = horace_binfile_interface.convert_old_det_forms(detr, 0)
        assert detr == deta
        assert deta.n_objects == 0
        assert deta.n_runs == 0

    def test_convert_old_det_forms_exceptions(self, det_struct):
        detr = unique_objects_container(baseclass="IX_detector_array")
        with pytest.raises((ValueError, RuntimeError), match="HORACE:horace_binfile_interface:invalid_argument"):
            horace_binfile_interface.convert_old_det_forms("wrong", 0)

        with pytest.raises((ValueError, RuntimeError), match="HORACE:horace_binfile_interface:invalid_argument"):
            horace_binfile_interface.convert_old_det_forms(detr, [1, 1])

        with pytest.raises((ValueError, RuntimeError), match="HORACE:horace_binfile_interface:invalid_argument"):
            horace_binfile_interface.convert_old_det_forms(detr, "a")

        with pytest.raises((ValueError, RuntimeError), match="HERBERT:unique_objects_container:invalid_argument"):
            horace_binfile_interface.convert_old_det_forms(det_struct, 0)


# =============================================================================
# test_const_blocks_map.m
# =============================================================================
class TestConstBlocksMap:
    @staticmethod
    def _modify_test_pos(raw_pos) -> dict:
        pos: dict = {}
        for name in raw_pos.dtype.names:
            val = raw_pos[name].item()
            if isinstance(val, np.ndarray) and val.dtype.names is not None:
                if val.ndim == 0:
                    pos[name] = {fn: int(val[fn].item()) for fn in val.dtype.names}
                else:
                    pos[name] = [
                        {fn: int(elem[fn].item()) for fn in val.dtype.names}
                        for elem in val.ravel()
                    ]
            else:
                pos[name] = int(val) if isinstance(val, (int, np.integer, float, np.floating)) else val

        pos["num_contrib_files_"] = 4
        sz = int(pos["detpar_pos_"] - pos["header_pos_"])
        base_hdr_pos = int(pos["header_pos_"])
        base_hdr_inf = dict(pos["header_pos_info_"])
        header_pos = np.zeros(4, dtype=np.int64)
        header_inf = []
        for i in range(4):
            header_pos[i] = base_hdr_pos + i * sz
            header_inf.append({k: v + i * sz for k, v in base_hdr_inf.items()})
        pos["header_pos_"] = header_pos
        pos["header_pos_info_"] = header_inf
        pos["detpar_pos_"] = int(pos["detpar_pos_"]) + 3 * sz
        return pos

    def test_sqw_4head_block_map_construct(self):
        mat_path = os.path.join(TEST_SQW_FILE_DIR, "pos_to_test.mat")
        pos_data = sio.loadmat(mat_path, squeeze_me=True)
        pos_dict = self._modify_test_pos(pos_data["pos"])

        bsc = _ConstBlocksMapTester(pos_dict)
        cblocks = bsc.cblocks_map
        assert len(cblocks) == 10
        assert "dnd_methadata" in cblocks
        np.testing.assert_array_equal(cblocks["dnd_data"].ravel(), [3414, 10000])
        assert "header" in cblocks
        assert cblocks["header"].shape == (2, 4)
        np.testing.assert_array_equal(
            cblocks["header"],
            np.array([[147, 691, 1235, 1779], [536, 536, 536, 536]]),
        )
        must_fit = bsc.get_must_fit()
        assert len(must_fit) == 6

        bsc2 = _ConstBlocksMapTester(pos_dict)
        assert bsc.check_equal_sizes(bsc2)

    def test_sqw_2head_block_map_construct(self):
        test_file = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_read_write_v0_t.sqw")
        ldr = sqw_formats_factory.instance().get_loader(test_file)
        pos = ldr.get_pos_info()
        ldr.delete()

        bsc = _ConstBlocksMapTester(pos)
        cblocks = bsc.cblocks_map
        assert len(cblocks) == 5
        np.testing.assert_array_equal(cblocks["dnd_data"].ravel(), [677080, 4096])
        assert "dnd_methadata" not in cblocks
        val = cblocks["header"]
        assert val.shape == (2, 2)
        np.testing.assert_array_equal(val[:, 0], [152, 336])
        np.testing.assert_array_equal(val[:, 1], [562, 336])

    def test_sqw_block_map_construct(self):
        test_file = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_1.sqw")
        ldr = sqw_formats_factory.instance().get_loader(test_file)
        pos = ldr.get_pos_info()
        ldr.delete()

        bsc = _ConstBlocksMapTester(pos)
        cblocks = bsc.cblocks_map
        assert len(cblocks) == 10
        val1 = cblocks["dnd_data"].ravel()
        val2 = cblocks["dnd_methadata"].ravel()
        assert val1[1] == 10000
        assert val2[1] == 304
        assert val1[0] == val2[0] + val2[1]
        np.testing.assert_array_equal(cblocks["header"].ravel(), [147, 536])

        bsc1 = _ConstBlocksMapTester(pos)
        assert bsc.check_equal_sizes(bsc1)

    def test_block_map_construct(self):
        test_file = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        tob = _BinfileV2CommonTester(test_file)
        pos = tob.get_pos_info()
        tob.delete()

        bsc = _ConstBlocksMapTester(pos)
        cblocks = bsc.cblocks_map
        assert len(cblocks) == 2
        val1 = cblocks["dnd_data"].ravel()
        val2 = cblocks["dnd_methadata"].ravel()
        assert val1[1] == 1296
        assert val2[1] == 528
        assert val1[0] == val2[0] + val2[1]
        assert len(bsc.get_must_fit()) == 2


# =============================================================================
# test_binfile_v2_common.m & test_sqw_binfile_common.m
# =============================================================================
class TestBinfileV2AndSqwCommon:
    def test_wrong_header(self):
        from pyhorace.io.sqw_fopen import sqw_fclose

        file_non_hor = os.path.join(COMMON_DATA_DIR, "96dets.par")
        obj = _BinfileV2CommonTester()
        with pytest.raises(RuntimeError, match="HORACE:horace_binfile_interface:io_error"):
            obj.get_file_header("non-existing_file.sqw")
        with pytest.raises(RuntimeError, match="HORACE:horace_binfile_interface:runtime_error"):
            obj.get_file_header(file_non_hor)

    def test_Unk_header(self):
        from pyhorace.io.sqw_fopen import sqw_fclose

        file_wrong = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_fictional_ver.sqw")
        obj = _BinfileV2CommonTester()
        header, fid = obj.get_file_header(file_wrong)
        sqw_fclose(fid)
        assert header["version"] == pytest.approx(2.666, rel=1e-4)
        assert header["name"] == "horace"
        assert header["typestart"] == 18

    def test_v2_header(self):
        from pyhorace.io.sqw_fopen import sqw_fclose

        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        obj = _BinfileV2CommonTester()
        header, fid = obj.get_file_header(file_v2)
        sqw_fclose(fid)
        assert header["version"] == 2
        assert header["name"] == "horace"
        assert header["sqw_type"] is False
        assert header["num_dim"] == 1
        assert header["typestart"] == 18

    def test_v3_header(self):
        from pyhorace.io.sqw_fopen import sqw_fclose

        file_v3 = os.path.join(TEST_SQW_FILE_DIR, "test_sqw_file_read_write_v3_1.sqw")
        obj = _BinfileV2CommonTester()
        header, fid = obj.get_file_header(file_v3)
        sqw_fclose(fid)
        assert np.isclose(header["version"], 3.1, atol=1e-6)
        assert header["name"] == "horace"
        assert header["sqw_type"] is True
        assert header["num_dim"] == 4
        assert header["typestart"] == 18

    def test_form_extracter(self):
        obj = _BinfileV2CommonTester()
        full_form = obj.get_dnd_form()
        assert len(full_form) == 19

        obj.set_datatype("a")
        meta_form = obj.get_dnd_form("-head")
        assert len(meta_form) == 16

        const_form = obj.get_dnd_form("-const")
        assert len(const_form) == 16

        const_head_form = obj.get_dnd_form("-const", "-head")
        assert len(const_head_form) == 13

        base_str = {f"f{i}_pos_": "" for i in range(1, 11)}
        filter1 = {"f1": "", "f2": "", "f3": ""}
        f1, f2, is_last = obj.extract_field_range(base_str, filter1)
        assert not is_last
        assert f1 == "f1_pos_"
        assert f2 == "f4_pos_"

        filter3 = {"f4": "", "f7": "", "f10": ""}
        f1, f2, is_last = obj.extract_field_range(base_str, filter3)
        assert is_last
        assert f1 == "f4_pos_"
        assert f2 == "f10_pos_"

    def test_copy_constructor(self):
        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        obj = _BinfileV2CommonTester()
        should_load, obj_init, mess = obj.should_load(file_v2)
        assert should_load
        obj = obj.init(obj_init)

        obj1 = _BinfileV2CommonTester(obj)
        assert obj.filename == obj1.filename
        assert obj.filepath == obj1.filepath
        assert obj.num_dim == obj1.num_dim
        assert obj.data_type == obj1.data_type
        assert list(obj.dnd_dimensions) == list(obj1.dnd_dimensions)
        assert obj.file_closer_ > 0
        assert obj1.file_closer_ > 0

        obj.delete()
        assert obj.file_closer_ is None
        assert obj.num_dim == "undefined"
        assert obj1.file_closer_ > 0
        obj1.delete()

    def test_activate_deactivate(self):
        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        obj = _BinfileV2CommonTester()
        should_load, obj_init, _ = obj.should_load(file_v2)
        obj = obj.init(obj_init)

        assert obj.is_activated("read")
        assert not obj.is_activated("write")
        obj.deactivate()
        assert not obj.is_activated("read")
        assert not obj.is_activated("write")
        obj.activate("read")
        assert obj.is_activated("read")
        assert not obj.is_activated("write")
        obj.deactivate()
        obj.activate("write")
        assert obj.is_activated("write")
        assert obj.is_activated("read")
        obj.delete()

    def test_reopen_to_write(self, tmp_path):
        file_v2 = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        test_file = str(tmp_path / "test_reopen_to_write.sqw")
        if os.path.exists(test_file):
            os.remove(test_file)

        obj = _BinfileV2CommonTester()
        should_load, obj_init, _ = obj.should_load(file_v2)
        obj = obj.init(obj_init)

        obj = obj.reopen_to_write(test_file)
        assert obj.filename == "test_reopen_to_write.sqw"
        assert os.path.exists(test_file)
        assert obj.file_closer_ > 0
        obj.delete()

    def test_sqw_binfile_common_forms(self):
        obj = _SqwBinfileCommonTester()
        mh_form = obj.get_main_header_form()
        assert list(mh_form.keys()) == ["filename_with_cdate", "filepath", "title", "nfiles"]
        mh_const = obj.get_main_header_form("-const")
        assert list(mh_const.keys()) == ["nfiles"]

        h_form = obj.get_header_form()
        assert "efix" in h_form and "en" in h_form and "filename" in h_form
        h_const = obj.get_header_form("-const")
        assert "efix" in h_const and "filename" not in h_const

        dp_form = obj.get_detpar_form()
        assert list(dp_form.keys()) == ["filename", "filepath", "ndet", "group", "x2", "phi", "azim", "width", "height"]
        dp_const = obj.get_detpar_form("-const")
        assert list(dp_const.keys()) == ["ndet", "group", "x2", "phi", "azim", "width", "height"]


# =============================================================================
# test_read_all.m
# =============================================================================
class TestReadAll:
    def test_read_dnd(self):
        files = [
            os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw"),
            os.path.join(COMMON_DATA_DIR, "w2d_qq_small_sqw.sqw"),
        ]
        res = read_dnd(files[0])
        assert isinstance(res, d1d)

        res = read_dnd(files[:2])
        assert len(res) == 2
        assert isinstance(res[0], d1d)
        assert isinstance(res[1], d2d)

        res = read_dnd(files)
        assert len(res) == 5
        assert isinstance(res[0], d1d)
        assert isinstance(res[1], d2d)
        assert isinstance(res[2], d3d)
        assert isinstance(res[3], d1d)
        assert isinstance(res[4], d2d)

    def test_read_horace(self):
        files = [
            os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w2d_qe_d2d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w3d_d3d.sqw"),
            os.path.join(COMMON_DATA_DIR, "w1d_sqw.sqw"),
            os.path.join(COMMON_DATA_DIR, "w2d_qq_small_sqw.sqw"),
        ]
        res = read_horace(files[0])
        assert isinstance(res, d1d)

        res = read_horace(files[3])
        assert isinstance(res, sqw)

        res = read_horace(files)
        assert len(res) == 5
        assert isinstance(res[0], d1d)
        assert isinstance(res[1], d2d)
        assert isinstance(res[2], d3d)
        assert isinstance(res[3], sqw)
        assert isinstance(res[4], sqw)

    def test_read_sqw_errors_on_dnd(self):
        dnd_file = os.path.join(COMMON_DATA_DIR, "w1d_d1d.sqw")
        with pytest.raises((ValueError, RuntimeError), match="HORACE:read_horace:invalid_argument"):
            read_sqw(dnd_file)

