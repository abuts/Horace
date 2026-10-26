"""Tests for cut on DnD objects, ported from test_dnd_cut.m."""

import os
import numpy as np
import pytest
import scipy.io

from pyhorace.coord_transf import cylinder_proj, line_proj, sphere_proj
from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.sqw import DnDBase, d0d, d1d, d2d, d3d, data_sqw_dnd, read_dnd, read_sqw

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)
TEST_DND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "test_dnd_class")
)


def _load_ref_dnd(mat_path: str, test_name: str, var_name: str = "w2") -> DnDBase:
    import io
    from scipy.io.matlab._mio5 import MatFile5Reader
    from pyhorace.core.serializable import Serializable
    from pyhorace.sqw.data_sqw_dnd import _mat_struct_to_dict

    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    tc = mat[test_name]
    raw = getattr(tc, var_name) if hasattr(tc, var_name) else tc[var_name]
    if isinstance(raw, np.ndarray) and raw.dtype.names and "_ObjectMetadata" in raw.dtype.names:
        obj_id = int(raw[0]["_ObjectMetadata"].flat[4])
        fw = mat["__function_workspace__"]
        hdr = b"MATLAB 5.0 MAT-file" + b" " * (124 - 19) + b"\x00\x01IM"
        bio = io.BytesIO(hdr + fw.tobytes()[8:])
        rdr = MatFile5Reader(bio, squeeze_me=True, struct_as_record=False)
        rdr.initialize_read()
        rdr.read_file_header()
        h, _ = rdr.read_var_header()
        arr = rdr.read_var_array(h)
        om = arr.MCOS[0]["_ObjectMetadata"]
        d = _mat_struct_to_dict(om[obj_id + 1])
        return Serializable.from_struct(d)
    if hasattr(raw, "data_"):
        raw = raw.data_
    elif hasattr(raw, "data"):
        raw = raw.data
    return DnDBase.dnd(data_sqw_dnd(raw))


@pytest.fixture(scope="module")
def _sqw_data():
    sqw_ref = read_sqw(os.path.join(COMMON_DATA_DIR, "w2d_qe_sqw.sqw"))
    return sqw_ref.data


class TestDnDCut:
    @pytest.fixture
    def d2d_obj(self, _sqw_data):
        return _sqw_data.copy()

    def test_2D_to2D_cut_with_cylinder_proj_23(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to2D_cut_with_cylinder_proj_23", "w2")
        proj = cylinder_proj([1, 1, 0], [0, 0, 1], type="aad")
        cut_range = d2d_obj.get_targ_range(proj)
        bin_range = [
            [0, 4],
            [-2, 0.1, 2],
            [cut_range[0, 2], 2, cut_range[1, 2]],
            [0, 10],
        ]
        w2 = d2d_obj.cut(proj, *bin_range)
        w2.s[:,0] = 0.0
        w2.s[:,-1] = 0.0
        w2.s[0,:] = 0.0
        w2.s[-1,:] = 0.0        
        assert w2.equal_to_tol(ref, ignore_str=True, tol=[1e-3, 2e-3])

    def test_2D_to2D_cut_with_cylinder_proj_24(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to2D_cut_with_cylinder_proj_24", "w2")
        proj = cylinder_proj([1, 1, 0], [0, 0, 1], type="aad")
        cut_range = d2d_obj.get_targ_range(proj)
        bin_range = [
            [0, 4],
            [-2, 0.1, 2],
            [cut_range[0, 2], cut_range[1, 2]],
            [],
        ]
        w2 = d2d_obj.cut(proj, *bin_range)
        s = w2.s.copy()
        s[:, 0] = 0.0
        s[:, -1] = 0.0
        s[0, :] = 0.0
        s[-1, :] = 0.0
        w2.s = s
        assert w2.equal_to_tol(ref, ignore_str=True, tol=[1e-9, 1e-9])

    def test_2D_to2D_cut_with_spher_proj(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to2D_cut_with_spher_proj", "w2")
        proj = sphere_proj(type="add")
        w2 = d2d_obj.cut(proj, [0, 0.1, 5], [0, 2, 180], [-180, 180], [20, 50])
        assert w2.equal_to_tol(ref, ignore_str=True, tol=[1e-9, 1e-9])

    def test_2D_to2D_cut_with_proj(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to2D_cut_with_proj", "w2")
        proj = line_proj([1, 1, 1], [0, 0, 1])
        w2 = d2d_obj.cut(proj, [0, 0.02, 0.9], [-0.3, 0.02, 0.9], [-0.1, 0.1], [20, 50])
        assert w2.equal_to_tol(ref, ignore_str=True, tol=[1e-9, 1e-9])

    def test_2D_to2D_cut(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to2D_cut", "w2")
        w2 = d2d_obj.cut([-0.9, 0.04, 0.9], [0, 4, 100])
        assert w2.equal_to_tol(ref, ignore_str=True, tol=[1e-9, 1e-9])

    def test_2D_to1D_cut(self, d2d_obj):
        ref_file = os.path.join(TEST_DND_DIR, "test_dnd_cut_output.mat")
        ref = _load_ref_dnd(ref_file, "test_2D_to1D_cut", "w1")
        w1 = d2d_obj.cut([-0.9, 0.04, 0.9], [20, 50])
        assert w1.equal_to_tol(ref, ignore_str=True, tol=[1e-9, 1e-9])

    def test_cut_invalid_number_of_arguments_raises_error(self, d2d_obj):
        with pytest.raises(HoraceError):
            d2d_obj.cut([0, 0.1, 1])


