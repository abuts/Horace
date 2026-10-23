"""Unit tests for line_proj cell intersections (converted from test_line_proj_cell_intersects.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import cylinder_axes, cylinder_proj, line_axes, line_proj, sphere_axes, sphere_proj


class TestLineProjCellIntersects:
    """Tests for get_contrib_cell_ind and may_contribute across projections."""

    def test_identical_axes_all_cells_contribute(self):
        p1 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        p2 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 10.0]]),
            nbins_all_dims=[2, 2, 2, 2],
        )
        ax2 = line_axes(
            img_range=np.array([[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 10.0]]),
            nbins_all_dims=[2, 2, 2, 2],
        )
        ind = p1.get_contrib_cell_ind(ax1, p2, ax2)
        np.testing.assert_array_equal(ind, np.arange(1, 17))

    def test_disjoint_energy_returns_empty(self):
        p1 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        p2 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[-1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 5.0]]),
            nbins_all_dims=[2, 2, 2, 2],
        )
        ax2 = line_axes(
            img_range=np.array([[-1.0, -1.0, -1.0, 10.0], [1.0, 1.0, 1.0, 15.0]]),
            nbins_all_dims=[2, 2, 2, 2],
        )
        ind = p1.get_contrib_cell_ind(ax1, p2, ax2)
        assert ind.size == 0

    def test_subvolume_intersection(self):
        p1 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        p2 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 1.0, 10.0]]),
            nbins_all_dims=[4, 4, 1, 2],
        )
        # Target covers only first bin in x and y ([0.1, 0.9]), and second energy bin ([6, 9])
        ax2 = line_axes(
            img_range=np.array([[0.1, 0.1, 0.0, 6.0], [0.9, 0.9, 1.0, 9.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        ind = p1.get_contrib_cell_ind(ax1, p2, ax2)
        # 3D cell 1 in 2nd energy slice (16 + 1 = 17)
        np.testing.assert_array_equal(ind, [17])

    def test_rotated_line_proj_intersection(self):
        p1 = line_proj([1, 0, 0], [0, 1, 0], alatt=[3.0, 3.0, 3.0], angdeg=90)
        p2 = line_proj([1, 1, 0], [1, -1, 0], alatt=[3.0, 3.0, 3.0], angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[-2.0, -2.0, -1.0, 0.0], [2.0, 2.0, 1.0, 10.0]]),
            nbins_all_dims=[4, 4, 1, 1],
        )
        ax2 = line_axes(
            img_range=np.array([[-0.5, -0.5, -0.5, 0.0], [0.5, 0.5, 0.5, 10.0]]),
            nbins_all_dims=[2, 2, 1, 1],
        )
        ind = p1.get_contrib_cell_ind(ax1, p2, ax2)
        # Central 2x2 cells in 4x4 grid are indices 6, 7, 10, 11 (1-based Fortran order)
        for c in [6, 7, 10, 11]:
            assert c in ind

    def test_line_to_sphere_proj_intersection(self):
        p1 = line_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)
        p2 = sphere_proj([1, 0, 0], [0, 1, 0], alatt=2 * np.pi, angdeg=90)

        ax1 = line_axes(
            img_range=np.array([[-2.0, -2.0, -2.0, 0.0], [2.0, 2.0, 2.0, 10.0]]),
            nbins_all_dims=[2, 2, 2, 1],
        )
        ax2 = sphere_axes(
            img_range=np.array([[0.0, 0.0, -180.0, 0.0], [1.5, 180.0, 180.0, 10.0]]),
            nbins_all_dims=[3, 3, 3, 1],
        )
        ind = p1.get_contrib_cell_ind(ax1, p2, ax2)
        assert ind.size == 8

