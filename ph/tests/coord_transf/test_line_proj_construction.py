"""Unit tests for line_proj construction and properties (converted from test_line_proj_construction.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import HoraceError, line_proj


class TestLineProjConstruction:
    """Tests for line_proj constructor, properties, validation, and serialization."""

    def test_default_constructor(self):
        proj = line_proj()
        np.testing.assert_allclose(proj.u, [1, 0, 0])
        np.testing.assert_allclose(proj.v, [0, 1, 0])
        assert proj.w.size == 0
        assert proj.type == "ppr"
        assert proj.nonorthogonal is False
        np.testing.assert_allclose(proj.offset, [0, 0, 0, 0])
        assert proj.label == ["\\zeta", "\\xi", "\\eta", "E"]

    def test_positional_and_kwarg_construction(self):
        proj = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 1],
            True,
            "rar",
            alatt=[2, 3, 4],
            angdeg=[90, 90, 90],
            offset=[0.5, 0.5, 0, 1],
            label=["a", "b", "c", "d"],
            title="my_cut",
        )
        np.testing.assert_allclose(proj.u, [1, 1, 0])
        np.testing.assert_allclose(proj.v, [1, -1, 0])
        np.testing.assert_allclose(proj.w, [0, 0, 1])
        assert proj.nonorthogonal is True
        assert proj.type == "rar"
        np.testing.assert_allclose(proj.alatt, [2, 3, 4])
        np.testing.assert_allclose(proj.angdeg, [90, 90, 90])
        np.testing.assert_allclose(proj.offset, [0.5, 0.5, 0, 1])
        assert proj.label == ["a", "b", "c", "d"]
        assert proj.title == "my_cut"

    def test_w_auto_updates_default_type(self):
        proj = line_proj([1, 0, 0], [0, 1, 0])
        assert proj.type == "ppr"
        proj.w = [0, 0, 1]
        assert proj.type == "ppp"
        proj.w = []
        assert proj.type == "ppr"

    def test_collinear_or_coplanar_vectors_throw(self):
        with pytest.raises(HoraceError):
            line_proj([1, 1, 0], [2, 2, 0])
        with pytest.raises(HoraceError):
            line_proj([1, 0, 0], [0, 1, 0], [1, 1, 0])
        with pytest.raises(HoraceError):
            line_proj([0, 0, 0], [0, 1, 0])

    def test_invalid_type_throws(self):
        proj = line_proj()
        with pytest.raises(HoraceError):
            proj.type = "xyz"
        with pytest.raises(HoraceError):
            proj.type = "pp"
        with pytest.raises(HoraceError):
            proj.type = "pdd"

    def test_serialization_roundtrip(self):
        proj = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            alatt=[3.0, 3.0, 5.0],
            angdeg=[90, 90, 90],
            offset=[0.1, 0.2, 0.3, 0.0],
            type="apr",
            title="test_ser",
        )
        s = proj.to_struct()
        rec = line_proj.from_struct(s)
        assert proj == rec
        np.testing.assert_allclose(rec.u, proj.u)
        np.testing.assert_allclose(rec.v, proj.v)
        np.testing.assert_allclose(rec.alatt, proj.alatt)
        np.testing.assert_allclose(rec.angdeg, proj.angdeg)
        np.testing.assert_allclose(rec.offset, proj.offset)
        assert rec.type == proj.type
        assert rec.title == proj.title
