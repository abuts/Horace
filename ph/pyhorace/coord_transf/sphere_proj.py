"""Spherical coordinate projection class (sphere_proj)."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError, sphere_axes
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.curve_proj_base import CurveProjBase
from pyhorace.coord_transf.ubmatrix import ubmatrix


class sphere_proj(CurveProjBase):
    """Spherical coordinate projection class: (|Q|, theta, phi, E).

    Polar angle ``theta`` in [0, 180] deg (or [0, pi] rad) is measured from
    direction ``u``; azimuthal angle ``phi`` in [-180, 180] deg (or [-pi, pi] rad)
    is measured in the plane perpendicular to ``u`` starting from the projection
    of ``v`` onto that plane.
    """

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._type = ["p", "d", "d"]
        self._label = ["|Q|", "\\theta", "\\phi", "En"]
        self._cur_axes_block_class_name = "sphere_axes"
        if args or kwargs:
            self.init(*args, **kwargs)

    def _get_axes_class(self) -> type:
        return sphere_axes

    def _set_type(self, val: Any) -> None:
        if not isinstance(val, str) or len(val) != 3:
            raise HoraceError(
                "HORACE:sphere_proj:invalid_argument",
                "type must be a 3-character string",
            )
        c0, c1, c2 = val[0].lower(), val[1].lower(), val[2].lower()
        if c0 not in ("a", "r", "p"):
            raise HoraceError(
                "HORACE:sphere_proj:invalid_argument",
                f"First character of sphere_proj type must be 'a', 'r', or 'p', got '{c0}'",
            )
        if c1 not in ("d", "r") or c2 not in ("d", "r") or c1 != c2:
            raise HoraceError(
                "HORACE:sphere_proj:invalid_argument",
                f"Second and third characters of sphere_proj type must both be 'd' or both be 'r', got '{c1}{c2}'",
            )
        self._type = [c0, c1, c2]
        if self._do_check_combo_arg and self.lattice_defined:
            self.check_combo_arg()

    def get_pix_img_transformation(
        self,
        ndim: int = 3,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Return rotation matrix rot_to_img, offset_cc, and scale factors for sphere_proj."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        b, arlu, _ = bmatrix(self._alatt, self._angdeg, nargout=3)
        _, umat = ubmatrix(self._u, self._v, b, nargout=2)
        rot_to_img_3 = self.hor2matlab_transf_ @ umat

        c0 = self._type[0]
        if c0 == "a":
            q_scale = 1.0
        elif c0 == "r":
            q_scale = float(np.max(arlu))
        else:  # 'p'
            q_scale = float(np.linalg.norm(b @ self._u))

        ang_scale = float(np.deg2rad(1.0)) if self._type[1] == "d" else 1.0
        scales_3 = np.array([q_scale, ang_scale, ang_scale], dtype=float)
        offset_cc_3 = b @ self._offset[:3]

        if ndim == 3:
            rot_mat = rot_to_img_3
            offset_cc = offset_cc_3
            scales = scales_3
        elif ndim == 4:
            rot_mat = np.eye(4, dtype=float)
            rot_mat[:3, :3] = rot_to_img_3
            offset_cc = np.array([offset_cc_3[0], offset_cc_3[1], offset_cc_3[2], self._offset[3]], dtype=float)
            scales = np.array([q_scale, ang_scale, ang_scale, 1.0], dtype=float)
        else:
            raise HoraceError(
                "HORACE:sphere_proj:invalid_argument",
                f"ndim must be 3 or 4, got {ndim}",
            )

        if nargout == 1:
            return rot_mat
        if nargout == 2:
            return rot_mat, offset_cc
        if nargout == 3:
            return rot_mat, offset_cc, scales
        offset_present = bool(np.any(np.abs(offset_cc) > self.tol_))
        return rot_mat, offset_cc, scales, offset_present

    def transform_pix_to_img(self, pix_input: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform pixel coordinates (Crystal Cartesian) into spherical coordinates (|Q|, theta, phi [, dE])."""
        q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_input)
        rot_to_img, offset_cc, scales, offset_present = self.get_pix_img_transformation(3, nargout=4)
        if offset_present:
            q_shifted = q_cc - offset_cc[:3].reshape(3, 1)
        else:
            q_shifted = q_cc

        xyz = rot_to_img @ q_shifted
        x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
        xy = np.hypot(x, y)
        r = np.hypot(xy, z)
        elev = np.arctan2(z, xy)
        theta = (0.5 * np.pi - elev) / scales[1]
        phi = np.arctan2(y, x) / scales[2]
        r_scaled = r / scales[0]

        res_3d = np.vstack([r_scaled, theta, phi])
        if ndim == 4:
            res = np.vstack([res_3d, dE - self._offset[3]])
        else:
            res = res_3d

        if is_row:
            res = res.T
            if np.asarray(pix_input).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform spherical coordinates (|Q|, theta, phi [, dE]) into Crystal Cartesian coordinates."""
        q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(img_data)
        rot_to_img, offset_cc, scales, offset_present = self.get_pix_img_transformation(3, nargout=4)

        r = q_img[0, :] * scales[0]
        theta = q_img[1, :] * scales[1]
        phi = q_img[2, :] * scales[2]

        sin_th = np.sin(theta)
        x = r * sin_th * np.cos(phi)
        y = r * sin_th * np.sin(phi)
        z = r * np.cos(theta)

        xyz = np.vstack([x, y, z])
        q_cc = rot_to_img.T @ xyz
        if offset_present:
            q_cc = q_cc + offset_cc[:3].reshape(3, 1)

        if ndim == 4:
            res = np.vstack([q_cc, dE + self._offset[3]])
        else:
            res = q_cc

        if is_row:
            res = res.T
            if np.asarray(img_data).ndim == 1:
                return res.ravel()
        return res
