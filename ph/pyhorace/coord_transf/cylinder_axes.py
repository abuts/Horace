"""Cylindrical coordinate axes block (cylinder_axes)."""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.axes_block_base import EPS_SINGLE, AxesBlockBase


class cylinder_axes(AxesBlockBase):
    """Axes block for cylindrical coordinates (Q_perp, Q_par, phi, En)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._angular_unit_is_rad: bool = False
        self._label = ["Q_{\\perp}", "Q_{||}", "\\phi", "En"]
        self._type = "ppde"
        self._changes_aspect_ratio = False
        self._img_range = self._default_img_range.copy()
        self._init_from_args(*args, **kwargs)

    @property
    def _max_img_range(self) -> np.ndarray:
        ang = np.pi if self._angular_unit_is_rad else 180.0
        return np.array(
            [[0.0, -np.inf, -ang, -np.inf], [np.inf, np.inf, ang, np.inf]],
            dtype=float,
        )

    @property
    def _default_img_range(self) -> np.ndarray:
        ang = np.pi if self._angular_unit_is_rad else 180.0
        return np.array(
            [[0.0, -1.0, -ang, 0.0], [1.0, 1.0, ang, 0.0]],
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
            self._type = "ppre" if new_val else "ppde"
            if not np.any(np.isinf(self._img_range[:, 2])):
                factor = (np.pi / 180.0) if (new_val and not old_val) else (180.0 / np.pi)
                self._img_range[:, 2] *= factor
                max_r = self._max_img_range
                self._img_range[:, 2] = np.clip(
                    self._img_range[:, 2], max_r[0, 2], max_r[1, 2]
                )
            if not self._do_check_combo_arg:
                self.check_combo_arg()
        else:
            self._type = "ppre" if new_val else "ppde"

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                "HORACE:cylinder_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                "HORACE:cylinder_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            ) from exc
        if arr.size == 3:
            arr = np.append(arr, 0.0)
        elif arr.size != 4:
            raise HoraceError(
                "HORACE:cylinder_axes:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        arr[np.abs(arr) < EPS_SINGLE] = 0.0
        max_r = self._max_img_range
        if (
            arr[0] < max_r[0, 0] - EPS_SINGLE
            or arr[2] < max_r[0, 2] - EPS_SINGLE
            or arr[2] > max_r[1, 2] + EPS_SINGLE
        ):
            raise HoraceError(
                "HORACE:cylinder_axes:invalid_argument",
                f"Cylindrical offset {arr} is outside allowed cylindrical bounds {max_r}",
            )
        self._offset = arr.copy()

    def default_pbin(self) -> List[np.ndarray]:
        res = super().default_pbin()
        max_r = self._max_img_range
        if res[2].size == 0:
            res[2] = max_r[:, 2].copy()
        return res

    def get_volume_scale(self) -> float:
        return float((self._ulen[0] ** 2) * self._ulen[1])

    def _calc_bin_volume_from_axes(
        self, axes: List[np.ndarray]
    ) -> Union[float, np.ndarray]:
        qr_edges = np.asarray(axes[0], dtype=float).ravel()
        qz_edges = np.asarray(axes[1], dtype=float).ravel()
        phi_edges = np.asarray(axes[2], dtype=float).ravel()
        en_edges = np.asarray(axes[3], dtype=float).ravel()

        if not self._angular_unit_is_rad:
            phi_rad = np.deg2rad(phi_edges)
        else:
            phi_rad = phi_edges

        dqr2 = 0.5 * np.diff(qr_edges ** 2)
        dqz = np.diff(qz_edges)
        dphi = np.diff(phi_rad)
        den = np.diff(en_edges)

        g_qr, g_qz, g_phi, g_en = np.meshgrid(
            dqr2, dqz, dphi, den, indexing="ij"
        )
        vol = (g_qr * g_qz * g_phi * g_en) * self.get_volume_scale()
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
            f"in {self._ulen[1]:.3g} \\AA^{{-1}}",
            ang_unit,
            "meV",
        ]
        return self._format_plot_titles(unit_labels)
