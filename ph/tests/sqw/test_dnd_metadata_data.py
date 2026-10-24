"""Tests for dnd_metadata and dnd_data classes, ported from test_dnd_metadata_data.m."""

import os
import numpy as np
import pytest

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.core.serializable import Serializable
from pyhorace.sqw import DnDBase, d2d, data_sqw_dnd, dnd_data, dnd_metadata

COMMON_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "_test", "common_data")
)


class TestDnDMetadataData:
    def test_empty_metadata_and_data(self):
        meta = dnd_metadata()
        assert meta.dimensions == 0
        assert isinstance(meta.axes, line_axes)
        assert isinstance(meta.proj, line_proj)
        assert not meta.creation_date_defined

        dat = dnd_data()
        assert dat.dimensions == 0
        assert dat.s.shape == (0, 0)
        assert dat.e.shape == (0, 0)
        assert dat.npix.shape == (0, 0)

    def test_construct_from_d2d_and_roundtrip(self):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        w2d = d2d(f2d)

        meta = w2d.metadata
        dat = w2d.nd_data

        assert meta.dimensions == 2
        assert dat.dimensions == 2
        np.testing.assert_array_equal(dat.s, w2d.s)
        np.testing.assert_array_equal(dat.e, w2d.e)
        np.testing.assert_array_equal(dat.npix, w2d.npix)

        w2d_rec = DnDBase.dnd(meta, dat)
        assert isinstance(w2d_rec, d2d)
        assert w2d.equal_to_tol(w2d_rec)

    def test_metadata_creation_date_from_dnd(self):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        w2d = d2d(f2d)
        meta = dnd_metadata(w2d)
        assert meta.creation_date_defined
        assert len(meta.creation_date_str) > 0

    def test_metadata_and_data_serialization(self):
        f2d = os.path.join(COMMON_DATA_DIR, "sqw_2d_2.sqw")
        w2d = d2d(f2d)
        meta = w2d.metadata
        dat = w2d.nd_data

        meta_s = meta.to_struct()
        meta_rec = Serializable.from_struct(meta_s)
        assert meta.equal_to_tol(meta_rec)

        dat_s = dat.to_struct()
        dat_rec = Serializable.from_struct(dat_s)
        assert dat.equal_to_tol(dat_rec)

    def test_dnd_data_validation(self):
        dat = dnd_data()
        with pytest.raises(HoraceError):
            dat.s = np.array([-1.0, 2.0])
            dat.e = np.array([-1.0, 2.0])  # negative variance should fail
