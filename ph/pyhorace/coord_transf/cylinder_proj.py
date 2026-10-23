"""Cylindrical coordinate projection class (cylinder_proj)."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError, cylinder_axes
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.curve_proj_base import CurveProjBase
from pyhorace.coord_transf.ubmatrix import ubmatrix


class cylinder_proj(CurveProjBase):
    """Cylindrical coordinate projection class: (Q_perp, Q_par, phi, En).

    ``Q_par`` is along the cylinder axis ``u`` (zenith); ``Q_perp`` is the radial
    projection perpendicular to ``u``; ``phi`` in [-180, 180] deg (or [-pi, pi] rad)
    is the azimuthal angle measured from the component of ``v`` perpendicular to ``u``.
    """

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._type = ["p", "p", "d"]
        self._label = ["Q_{\\perp}", "Q_{||}", "\\phi", "En"]
        self._cur_axes_block_class_name = "cylinder_axes"
        if args or kwargs:
            self.init(*args, **kwargs)

    def _get_axes_class(self) -> type:
        return cylinder_axes

    def _set_type(self, val: Any) -> None:
        if not isinstance(val, str) or len(val) != 3:
            raise HoraceError(
                "HORACE:cylinder_proj:invalid_argument",
                "type must be a 3-character string",
            )
        c0, c1, c2 = val[0].lower(), val[1].lower(), val[2].lower()
        if c0 not in ("a", "r", "p") or c1 not in ("a", "r", "p"):
            raise HoraceError(
                "HORACE:cylinder_proj:invalid_argument",
                f"First two characters of cylinder_proj type must be 'a', 'r', or 'p', got '{c0}{c1}'",
            )
        if c2 not in ("d", "r"):
            raise HoraceError(
                "HORACE:cylinder_proj:invalid_argument",
                f"Third character of cylinder_proj type must be 'd' or 'r', got '{c2}'",
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
        """Return rotation matrix rot_to_img, offset_cc, and scale factors for cylinder_proj."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        b, arlu, _ = bmatrix(self._alatt, self._angdeg, nargout=3)
        _, umat = ubmatrix(self._u, self._v, b, nargout=2)
        rot_to_img_3 = self.hor2matlab_transf_ @ umat

        # Scale 0 (Q_perp, along e2 = umat[1, :])
        c0 = self._type[0]
        if c0 == "a":
            s0 = 1.0
        elif c0 == "r":
            s0 = float(np.max(arlu))
        else:  # 'p'
            v_cc = b @ self._v
            s0 = float(abs(np.dot(v_cc, umat[1, :])))

        # Scale 1 (Q_par, along e1 = ub[0, :] = u / |u|)
        c1 = self._type[1]
        if c1 == "a":
            s1 = 1.0
        elif c1 == "r":
            s1 = float(np.max(arlu))
        else:  # 'p'
            s1 = float(np.linalg.norm(b @ self._u))

        # Scale 2 (phi)
        s2 = float(np.deg2rad(1.0)) if self._type[2] == "d" else 1.0

        scales_3 = np.array([s0, s1, s2], dtype=float)
        offset_cc_3 = b @ self._offset[:3]

        if ndim == 3:
            rot_mat = rot_to_img_3
            offset_cc = offset_cc_3
            scales = scales_3
        elif ndim == 4:
            rot_mat = np.eye(4, dtype=float)
            rot_mat[:3, :3] = rot_to_img_3
            offset_cc = np.array([offset_cc_3[0], offset_cc_3[1], offset_cc_3[2], self._offset[3]], dtype=float)
            scales = np.array([s0, s1, s2, 1.0], dtype=float)
        else:
            raise HoraceError(
                "HORACE:cylinder_proj:invalid_argument",
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
        """Transform pixel coordinates (Crystal Cartesian) into cylindrical coordinates (Q_perp, Q_par, phi [, dE])."""
        q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_input)
        rot_to_img, offset_cc, scales, offset_present = self.get_pix_img_transformation(3, nargout=4)
        if offset_present:
            q_shifted = q_cc - offset_cc[:3].reshape(3, 1)
        else:
            q_shifted = q_cc

        xyz = rot_to_img @ q_shifted
        x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
        q_perp = np.hypot(x, y) / scales[0]
        q_par = z / scales[1]
        phi = np.arctan2(y, x) / scales[2]

        res_3d = np.vstack([q_perp, q_par, phi])
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
        """Transform cylindrical coordinates (Q_perp, Q_par, phi [, dE]) into Crystal Cartesian coordinates."""
        q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(img_data)
        rot_to_img, offset_cc, scales, offset_present = self.get_pix_img_transformation(3, nargout=4)

        rho = q_img[0, :] * scales[0]
        z = q_img[1, :] * scales[1]
        phi = q_img[2, :] * scales[2]

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
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
