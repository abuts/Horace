"""Spherical coordinate axes block (sphere_axes)."""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.axes_block_base import EPS_SINGLE, AxesBlockBase


class sphere_axes(AxesBlockBase):
    """Axes block for spherical coordinates (|Q|, theta, phi, En)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._angular_unit_is_rad: bool = False
        self._label = ["|Q|", "\\theta", "\\phi", "En"]
        self._type = "pdde"
        self._changes_aspect_ratio = False
        self._img_range = self._default_img_range.copy()
        self._init_from_args(*args, **kwargs)

    @property
    def _max_img_range(self) -> np.ndarray:
        ang = np.pi if self._angular_unit_is_rad else 180.0
        return np.array(
            [[0.0, 0.0, -ang, -np.inf], [np.inf, ang, ang, np.inf]],
            dtype=float,
        )

    @property
    def _default_img_range(self) -> np.ndarray:
        ang = np.pi if self._angular_unit_is_rad else 180.0
        return np.array(
            [[0.0, 0.0, -ang, 0.0], [1.0, ang, ang, 0.0]],
            dtype=float,
        )

    @property
    def _p_defines_bin_centers(self) -> np.ndarray:
        return np.ones(4, dtype=bool)

    def saveable_fields(self) -> List[str]:
        return ["angular_unit_is_rad"] + list(self._base_fields)

    @property
    def angular_unit_is_rad(self) -> bool:
        return self._angular_unit_is_rad

    @angular_unit_is_rad.setter
    def angular_unit_is_rad(self, val: Any) -> None:
        new_val = bool(val)
        if new_val != self._angular_unit_is_rad:
            old_val = self._angular_unit_is_rad
            self._angular_unit_is_rad = new_val
            self._type = "prre" if new_val else "pdde"
            if not np.any(np.isinf(self._img_range[:, 1:3])):
                factor = (np.pi / 180.0) if (new_val and not old_val) else (180.0 / np.pi)
                self._img_range[:, 1:3] *= factor
                max_r = self._max_img_range
                self._img_range[:, 1:3] = np.clip(
                    self._img_range[:, 1:3], max_r[0:1, 1:3], max_r[1:2, 1:3]
                )
            if not self._do_check_combo_arg:
                self.check_combo_arg()
        else:
            self._type = "prre" if new_val else "pdde"

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                "HORACE:sphere_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                "HORACE:sphere_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            ) from exc
        if arr.size == 3:
            arr = np.append(arr, 0.0)
        elif arr.size != 4:
            raise HoraceError(
                "HORACE:sphere_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        arr[np.abs(arr) < EPS_SINGLE] = 0.0
        max_r = self._max_img_range
        if (
            arr[0] < max_r[0, 0] - EPS_SINGLE
            or arr[1] < max_r[0, 1] - EPS_SINGLE
            or arr[1] > max_r[1, 1] + EPS_SINGLE
            or arr[2] < max_r[0, 2] - EPS_SINGLE
            or arr[2] > max_r[1, 2] + EPS_SINGLE
        ):
            raise HoraceError(
                "HORACE:sphere_axes:invalid_argument",
                f"Spherical offset {arr} is outside allowed spherical bounds {max_r}",
            )
        self._offset = arr.copy()

    def default_pbin(self) -> List[np.ndarray]:
        res = super().default_pbin()
        max_r = self._max_img_range
        for i in (1, 2):
            if res[i].size == 0:
                res[i] = max_r[:, i].copy()
        return res

    def get_volume_scale(self) -> float:
        return float(self._ulen[0] ** 3)

    def _calc_bin_volume_from_axes(
        self, axes: List[np.ndarray]
    ) -> Union[float, np.ndarray]:
        q_edges = np.asarray(axes[0], dtype=float).ravel()
        th_edges = np.asarray(axes[1], dtype=float).ravel()
        phi_edges = np.asarray(axes[2], dtype=float).ravel()
        en_edges = np.asarray(axes[3], dtype=float).ravel()

        if not self._angular_unit_is_rad:
            th_rad = np.deg2rad(th_edges)
            phi_rad = np.deg2rad(phi_edges)
        else:
            th_rad = th_edges
            phi_rad = phi_edges

        dq3 = np.diff(q_edges ** 3) / 3.0
        dcos_th = -np.diff(np.cos(th_rad))
        dphi = np.diff(phi_rad)
        den = np.diff(en_edges)

        g_q, g_th, g_phi, g_en = np.meshgrid(
            dq3, dcos_th, dphi, den, indexing="ij"
        )
        vol = (g_q * g_th * g_phi * g_en) * self.get_volume_scale()
        if vol.size == 1:
            return float(vol.ravel()[0])
        return vol

    def calc_bin_volume(self, *args) -> Union[float, np.ndarray]:
        if len(args) == 0:
            axes = self._build_bin_nodes_axes(
                self._nbins_all_dims, is_halo=False, is_centre=False
            )
            return self._calc_bin_volume_from_axes(axes)
        return super().calc_bin_volume(*args)

    def data_plot_titles(
        self, data_obj: Any = None
    ) -> Tuple[str, List[str], List[str]]:
        ang_unit = "rad" if self._angular_unit_is_rad else "deg"
        unit_labels = [
            f"in {self._ulen[0]:.3g} \\AA^{{-1}}",
            ang_unit,
            ang_unit,
            "meV",
        ]
        return self._format_plot_titles(unit_labels)
