"""Unit tests for AxesBlockBase methods (converted from test_AxesBlockBase_methods.m)."""

import numpy as np
import pytest

from pyhorace.coord_transf import (
    AxesBlockBase,
    HoraceError,
    PixelDataBase,
    PixelDataMemory,
    cylinder_axes,
    cylinder_proj,
    expand_box,
    line_axes,
    line_proj,
    sphere_axes,
    sphere_proj,
)


class line_axes_char_size_tester(AxesBlockBase):
    """Helper class inheriting directly from AxesBlockBase to test base get_char_size."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._init_from_args(*args, **kwargs)

    @property
    def _default_img_range(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=float)

    @property
    def _p_defines_bin_centers(self) -> np.ndarray:
        return np.ones(4, dtype=bool)


class TestAxesBlockBaseMethods:
    """Tests for AxesBlockBase methods and properties."""

    def test_build_from_input_binning_1D_and_3D(self):
        cur_range = [[-1.0, 0.1, 1.0], [-2.0, 2.0], [-3.0, 0.2, 3.0], [0.0, 1.0, 10.0]]
        pbin = [[], [-1.0, 0.2, 1.0], [-2.0, 2.0], [2.0]]
        ab = AxesBlockBase.build_from_input_binning("line_axes", cur_range, pbin)
        assert isinstance(ab, line_axes)
        assert ab.dimensions == 3
        np.testing.assert_array_equal(ab.pax, np.array([1, 2, 4]))
        np.testing.assert_array_equal(ab.iax, np.array([3]))
        np.testing.assert_allclose(ab.iint, np.array([[-2.0], [2.0]]))
        np.testing.assert_allclose(
            ab.img_range,
            np.array(
                [[-1.05, -1.1, -2.0, -0.5], [1.05, 1.1, 2.0, 11.5]]
            ),
        )

    def test_build_from_input_binning_zero_step_reuses_current_step(self):
        cur_range = [[-1.0, 0.1, 1.0], [-2.0, 0.2, 2.0], [-3.0, 3.0], [0.0, 1.0, 10.0]]
        pbin = [[-0.5, 0.0, 0.5], [0.0], [], [2.0, 0.0, 8.0]]
        ab = line_axes.build_from_input_binning(cur_range, pbin)
        assert ab.dimensions == 3
        np.testing.assert_allclose(ab.step, np.array([0.1, 0.2, 1.0]))
        np.testing.assert_allclose(
            ab.img_range,
            np.array([[-0.55, -2.1, -3.0, 1.5], [0.55, 2.1, 3.0, 8.5]]),
        )

    def test_build_from_input_binning_zero_step_on_integrated_axis_throws(self):
        cur_range = [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], [0.0, 10.0]]
        pbin = [[0.0], [], [], []]
        with pytest.raises(HoraceError) as exc_info:
            line_axes.build_from_input_binning(cur_range, pbin)
        assert "invalid_argument" in exc_info.value.error_id

    def test_get_img_range_from_cut_range(self):
        rng = AxesBlockBase.get_img_range_from_cut_range(
            [-1.0, 0.1, 1.0], [-2.0, 2.0], [], [-5.0, 1.0, 5.0]
        )
        expected = np.array(
            [[-1.05, -2.0, -np.inf, -5.5], [1.05, 2.0, np.inf, 5.5]]
        )
        np.testing.assert_allclose(rng, expected)

        with pytest.raises(HoraceError):
            AxesBlockBase.get_img_range_from_cut_range(
                [-1.0, 0.0, 1.0], [-2.0, 2.0], [], [-5.0, 1.0, 5.0]
            )

    def test_get_char_size_line_vs_base(self):
        proj = line_proj([1, 1, 0], [1, -1, 0], alatt=[2 * np.pi, 2 * np.pi, 2 * np.pi], angdeg=[90, 90, 90])
        ab_line = line_axes([-1, 0.1, 1], [-1, 0.2, 1], [-1, 0.5, 1], [0, 1, 10])
        ab_base = line_axes_char_size_tester(
            [-1, 0.1, 1], [-1, 0.2, 1], [-1, 0.5, 1], [0, 1, 10]
        )

        cs_line_no_proj = ab_line.get_char_size()
        cs_base_no_proj = ab_base.get_char_size()
        np.testing.assert_allclose(cs_line_no_proj, cs_base_no_proj)

        cs_line = ab_line.get_char_size(proj)
        cs_base = ab_base.get_char_size(proj)
        np.testing.assert_allclose(cs_line[:3], cs_base[:3], rtol=1e-12)
        assert cs_line[3] == pytest.approx(11.0)
        assert cs_base[3] == pytest.approx(1.0)

    def test_get_char_size_sphere_and_cylinder(self):
        sproj = sphere_proj(alatt=[3, 3, 3], angdeg=[90, 90, 90])
        sab = sphere_axes([0, 0.1, 2], [0, 10, 180], [-180, 10, 180], [0, 1, 10])
        cs_sph = sab.get_char_size(sproj)
        assert cs_sph.shape == (4,)
        assert np.all(cs_sph > 0)

        cproj = cylinder_proj(alatt=[3, 3, 3], angdeg=[90, 90, 90])
        cab = cylinder_axes([0, 0.1, 2], [-1, 0.1, 1], [-180, 10, 180], [0, 1, 10])
        cs_cyl = cab.get_char_size(cproj)
        assert cs_cyl.shape == (4,)
        assert np.all(cs_cyl > 0)

    def test_get_bin_nodes_invalid_inputs_throw(self):
        ab = line_axes([-1, 0.5, 1], [-1, 1], [-1, 1], [0, 2, 4])
        with pytest.raises(HoraceError) as exc1:
            ab.get_bin_nodes("-wrong")
        assert "invalid_argument" in exc1.value.error_id

        with pytest.raises(HoraceError) as exc2:
            ab.get_bin_nodes(np.ones((3, 1)))
        assert "invalid_argument" in exc2.value.error_id

        with pytest.raises(HoraceError) as exc3:
            ab.get_bin_nodes("-bin_edges", "-bin_centre")
        assert "invalid_argument" in exc3.value.error_id

    def test_get_bin_nodes_options(self):
        ab = line_axes([-1, 1, 1], [-2, 2], [-1, 1, 1], [0, 5, 10])
        # nbins_all_dims = [3, 1, 3, 3]
        nodes_4d, en, npts, vol = ab.get_bin_nodes(nargout=4)
        np.testing.assert_array_equal(npts, np.array([4, 2, 4, 4]))
        assert nodes_4d.shape == (4, 4 * 2 * 4 * 4)
        assert en.size == 4
        assert vol == pytest.approx(1.0 * 4.0 * 1.0 * 5.0)

        nodes_3d, en_3d, npts_3d = ab.get_bin_nodes("-3D", nargout=3)
        assert nodes_3d.shape == (3, 4 * 2 * 4)
        np.testing.assert_array_equal(npts_3d, np.array([4, 2, 4, 4]))

        nodes_centre, _, npts_c, vol_c = ab.get_bin_nodes("-bin_centre", nargout=4)
        np.testing.assert_array_equal(npts_c, np.array([3, 1, 3, 3]))
        assert nodes_centre.shape == (4, 3 * 1 * 3 * 3)
        assert vol_c == pytest.approx(vol)

        plot_edges = ab.get_bin_nodes("-plot_edges")
        assert isinstance(plot_edges, list) and len(plot_edges) == 4
        assert [len(a) for a in plot_edges] == [4, 2, 4, 4]

        dens_nodes, _, npts_d = ab.get_bin_nodes("-dens_interp", nargout=3)
        assert isinstance(dens_nodes, list) and len(dens_nodes) == 4
        # pax are [1, 3, 4], so halo adds +2 to axes 1, 3, 4 bin centres (3+2=5), axis 2 is iax (1)
        np.testing.assert_array_equal(npts_d, np.array([5, 1, 5, 5]))
        assert dens_nodes[0].shape == (5, 1, 5, 5)

        hull_3d = ab.get_bin_nodes("-3D", "-hull")
        nb3 = np.array([4, 2, 4])
        expected_hull_3d = 2 * (nb3[0] * nb3[1] + nb3[0] * nb3[2] + nb3[1] * nb3[2])
        assert hull_3d.shape == (3, expected_hull_3d)

        hull_4d = ab.get_bin_nodes("-hull")
        nb4 = np.array([4, 2, 4, 4])
        expected_hull_4d = 2 * (
            nb4[0] * nb4[1] * nb4[2]
            + nb4[0] * nb4[1] * nb4[3]
            + nb4[0] * nb4[2] * nb4[3]
            + nb4[1] * nb4[2] * nb4[3]
        )
        assert hull_4d.shape == (4, expected_hull_4d)

    def test_default_empty_range_on_base_tester(self):
        class EmptyAxesTester(AxesBlockBase):
            pass

        ab = EmptyAxesTester()
        np.testing.assert_array_equal(ab.img_range, PixelDataBase.EMPTY_RANGE_)

    def test_bin_pixels_variants(self):
        ab = line_axes([0, 1, 2], [0, 1, 2], [0, 1], [0, 1])
        # img_range: [-0.5, 2.5] with 3 bins on axes 1 and 2; [0, 1] on axes 3 and 4
        coords = np.array(
            [
                [0.0, 1.0, 2.0, 0.0, 5.0],
                [0.0, 1.0, 2.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        npix, ok, ix = ab.bin_pixels(coords, nargout=3)
        assert npix.shape == (3, 3)
        np.testing.assert_array_equal(ok, np.array([True, True, True, True, False]))
        assert npix[0, 0] == 2
        assert npix[1, 1] == 1
        assert npix[2, 2] == 1
        assert np.sum(npix) == 4

        # Bin with PixelDataMemory
        raw_pix = np.zeros((9, 5), dtype=float)
        raw_pix[0:4, :] = coords
        raw_pix[7, :] = [10.0, 20.0, 30.0, 30.0, 100.0]
        raw_pix[8, :] = [1.0, 4.0, 9.0, 3.0, 10.0]
        pix = PixelDataMemory(raw_pix)

        npix2, s2, e2, pix_out, uniq_pos = ab.bin_pixels(coords, pix, nargout=5)
        np.testing.assert_array_equal(npix2, npix)
        assert s2[0, 0] == pytest.approx(20.0)  # (10 + 30) / 2
        assert e2[0, 0] == pytest.approx(1.0)  # (1 + 3) / 4
        assert s2[1, 1] == pytest.approx(20.0)
        assert s2[2, 2] == pytest.approx(30.0)
        assert pix_out.num_pixels == 4
        np.testing.assert_array_equal(uniq_pos, np.array([1, 5, 9]))

