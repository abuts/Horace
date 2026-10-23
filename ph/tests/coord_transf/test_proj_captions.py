"""Unit tests for axes block plot captions (converted from test_proj_captions.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    cylinder_axes,
    cylinder_proj,
    line_axes,
    line_proj,
    sphere_axes,
    sphere_proj,
)


class TestProjCaptions:
    """Test suite for data_plot_titles across line_axes, sphere_axes, and cylinder_axes."""

    def test_line_axes_captions_orthogonal_and_offset(self):
        proj = line_proj(
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 1],
            offset=[1, 0, 0, 0],
            alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi],
            angdeg=[90, 90, 90],
            label=["\\zeta", "\\xi", "\\eta", "En"],
            title="Test Linear Plot",
        )
        ax = line_axes([-1, 0.1, 1], [-2, 0.2, 2], [-0.5, 0.5], [0, 10])
        ax = proj.copy_proj_defined_properties_to_axes(ax)
        ax.filename = "sample.sqw"
        ax.filepath = "C:\\data"

        title_main, title_pax, title_iax = ax.data_plot_titles()
        assert "sample.sqw" in title_main
        assert "Test Linear Plot" in title_main
        assert len(title_pax) == 2
        assert len(title_iax) == 2
        assert "\\zeta" in title_pax[0]
        assert "\\xi" in title_pax[1]
        assert "\\eta" in title_iax[0]
        assert "En" in title_iax[1]

    def test_sphere_axes_captions_deg_and_rad(self):
        sproj = sphere_proj(
            alatt=[3, 3, 3],
            angdeg=[90, 90, 90],
            type="add",
            title="Sphere Cut",
        )
        sax = sphere_axes([0, 0.1, 2], [0, 5, 180], [-10, 10], [0, 5])
        sax = sproj.copy_proj_defined_properties_to_axes(sax)
        main_d, pax_d, iax_d = sax.data_plot_titles()
        assert "Sphere Cut" in main_d
        assert len(pax_d) == 2
        assert "deg" in pax_d[1]
        assert len(iax_d) == 2

        sax.angular_unit_is_rad = True
        _, pax_r, _ = sax.data_plot_titles()
        assert "rad" in pax_r[1]

    def test_cylinder_axes_captions_deg_and_rad(self):
        cproj = cylinder_proj(
            alatt=[3, 3, 3],
            angdeg=[90, 90, 90],
            type="aad",
            title="Cylinder Cut",
        )
        cax = cylinder_axes([0, 0.1, 2], [-1, 1], [-180, 10, 180], [0, 5])
        cax = cproj.copy_proj_defined_properties_to_axes(cax)
        main_c, pax_c, iax_c = cax.data_plot_titles()
        assert "Cylinder Cut" in main_c
        assert len(pax_c) == 2
        assert "deg" in pax_c[1]
        assert len(iax_c) == 2

        cax.angular_unit_is_rad = True
        _, pax_cr, _ = cax.data_plot_titles()
        assert "rad" in pax_cr[1]

