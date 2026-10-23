"""Spherical coordinate projection centered at scattered wavevector kf (kf_sphere_proj)."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError, PixelDataBase, fast_map, sphere_axes
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.sphere_proj import sphere_proj

# Conversion factor hbar^2 / (2 * m_n) in meV * Angstrom^2
K_TO_E_MEV = 2.072124836987059


class kf_sphere_proj(sphere_proj):
    """Spherical projection in the scattered wavevector ``kf`` frame: (|kf|, theta, phi, E)."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._Ei: np.ndarray = np.array([], dtype=float)
        self._emode: int = 1
        self._cc_to_spec_mat: np.ndarray = np.eye(3, dtype=float).reshape((3, 3, 1))
        self._run_id_mapper: Optional[fast_map] = None
        self._type = ["a", "d", "d"]
        self._label = ["|kf|", "\\theta", "\\phi", "En"]
        self._do_3D_transformation = False
        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return [
            "u",
            "v",
            "type",
            "alatt",
            "angdeg",
            "offset",
            "label",
            "title",
            "Ei",
            "emode",
            "cc_to_spec_mat",
            "run_id_mapper",
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def _set_offset(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._offset = np.zeros(4, dtype=float)
            return
        arr = np.asarray(val, dtype=float).ravel()
        if np.any(np.abs(arr) > self.tol_):
            raise HoraceError(
                "HORACE:kf_sphere_proj:not_implemented",
                "Non-zero offset is not implemented for kf_sphere_proj",
            )
        self._offset = np.zeros(4, dtype=float)

    @property
    def Ei(self) -> np.ndarray:
        return self._Ei.copy()

    @Ei.setter
    def Ei(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._Ei = np.array([], dtype=float)
            return
        arr = np.asarray(val, dtype=float).ravel()
        if np.any(arr <= 0):
            raise HoraceError(
                "HORACE:kf_sphere_proj:invalid_argument",
                "Incident energy Ei must be positive",
            )
        self._Ei = arr.copy()

    @property
    def ki_mod(self) -> np.ndarray:
        if self._Ei.size == 0:
            return np.array([], dtype=float)
        return np.sqrt(self._Ei / K_TO_E_MEV)

    @property
    def emode(self) -> int:
        return self._emode

    @emode.setter
    def emode(self, val: Any) -> None:
        v = int(val)
        if v not in (1, 2):
            raise HoraceError(
                "HORACE:kf_sphere_proj:invalid_argument",
                "emode must be 1 (direct geometry) or 2 (indirect geometry)",
            )
        self._emode = v

    @property
    def cc_to_spec_mat(self) -> np.ndarray:
        return self._cc_to_spec_mat.copy()

    @cc_to_spec_mat.setter
    def cc_to_spec_mat(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 2 and arr.shape == (3, 3):
            arr = arr.reshape((3, 3, 1))
        elif arr.ndim != 3 or arr.shape[:2] != (3, 3):
            raise HoraceError(
                "HORACE:kf_sphere_proj:invalid_argument",
                "cc_to_spec_mat must have shape (3, 3) or (3, 3, n_runs)",
            )
        self._cc_to_spec_mat = arr.copy()

    @property
    def run_id_mapper(self) -> Optional[fast_map]:
        return self._run_id_mapper

    @run_id_mapper.setter
    def run_id_mapper(self, val: Optional[fast_map]) -> None:
        if val is not None and not isinstance(val, fast_map):
            raise HoraceError(
                "HORACE:kf_sphere_proj:invalid_argument",
                "run_id_mapper must be a fast_map instance",
            )
        self._run_id_mapper = val

    # ------------------------------------------------------------------
    # Coordinate Transformations
    # ------------------------------------------------------------------
    def _extract_pix_and_run_indices(
        self,
        pix_input: Any,
        *args: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Extract q_cc (3xN), dE (1xN), and 0-based run matrix indices (N,)."""
        if isinstance(pix_input, PixelDataBase):
            q_cc = pix_input.q_coordinates
            dE = pix_input.dE
            run_ids = pix_input.run_idx.ravel()
            is_row = False
        else:
            arr = np.asarray(pix_input, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == 9:
                q_cc = arr[0:3, :]
                dE = arr[3:4, :]
                run_ids = arr[4, :]
                is_row = False
            else:
                q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_input)
                if ndim < 4:
                    raise HoraceError(
                        "HORACE:kf_sphere_proj:invalid_argument",
                        "kf_sphere_proj requires 4D coordinates or PixelData input",
                    )
                if len(args) >= 1 and args[0] is not None:
                    run_ids = np.asarray(args[0], dtype=float).ravel()
                else:
                    run_ids = np.ones(q_cc.shape[1], dtype=float)

        if self._run_id_mapper is not None and self._run_id_mapper.n_members > 0:
            mapped = np.asarray(self._run_id_mapper.get_values_for_keys(run_ids), dtype=int).ravel() - 1
        else:
            mapped = np.zeros(q_cc.shape[1], dtype=int)
        return q_cc, dE, mapped, is_row

    def transform_pix_to_img(self, pix_input: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform pixel coordinates into (|kf|, theta, phi, dE)."""
        q_cc, dE, run_idx_0, is_row = self._extract_pix_and_run_indices(pix_input, *args)
        _, _, scales, _ = self.get_pix_img_transformation(3, nargout=4)
        rot_to_img = self.hor2matlab_transf_

        n_pts = q_cc.shape[1]
        kf_spec = np.zeros((3, n_pts), dtype=float)
        ki_all = self.ki_mod

        unique_runs = np.unique(run_idx_0)
        for r_i in unique_runs:
            mask = run_idx_0 == r_i
            mat_i = self._cc_to_spec_mat[:, :, min(r_i, self._cc_to_spec_mat.shape[2] - 1)]
            ki_i = ki_all[min(r_i, ki_all.size - 1)] if ki_all.size > 0 else 0.0
            q_sp = mat_i @ q_cc[:, mask]
            # In spectrometer coordinates: q_spec = kf - ki (or ki - kf);
            # kf_x = q_sp[0] - ki_i (so at q_sp = [ki, 0, 0], kf = 0)
            q_sp[0, :] = q_sp[0, :] - ki_i
            kf_spec[:, mask] = q_sp

        xyz = rot_to_img @ kf_spec
        x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
        xy = np.hypot(x, y)
        r = np.hypot(xy, z)
        elev = np.arctan2(z, xy)
        theta = (0.5 * np.pi - elev) / scales[1]
        phi = np.arctan2(y, x) / scales[2]
        r_scaled = r / scales[0]

        res = np.vstack([r_scaled, theta, phi, dE - self._offset[3]])
        if is_row:
            res = res.T
            if np.asarray(pix_input).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform (|kf|, theta, phi, dE) back into Crystal Cartesian coordinates."""
        q_img, dE, run_idx_0, is_row = self._extract_pix_and_run_indices(img_data, *args)
        _, _, scales, _ = self.get_pix_img_transformation(3, nargout=4)
        rot_to_img = self.hor2matlab_transf_

        r = q_img[0, :] * scales[0]
        theta = q_img[1, :] * scales[1]
        phi = q_img[2, :] * scales[2]

        sin_th = np.sin(theta)
        x = r * sin_th * np.cos(phi)
        y = r * sin_th * np.sin(phi)
        z = r * np.cos(theta)

        kf_spec = rot_to_img.T @ np.vstack([x, y, z])
        n_pts = q_img.shape[1]
        q_cc = np.zeros((3, n_pts), dtype=float)
        ki_all = self.ki_mod

        unique_runs = np.unique(run_idx_0)
        for r_i in unique_runs:
            mask = run_idx_0 == r_i
            mat_i = self._cc_to_spec_mat[:, :, min(r_i, self._cc_to_spec_mat.shape[2] - 1)]
            ki_i = ki_all[min(r_i, ki_all.size - 1)] if ki_all.size > 0 else 0.0
            q_sp = kf_spec[:, mask].copy()
            q_sp[0, :] = q_sp[0, :] + ki_i
            q_cc[:, mask] = np.linalg.solve(mat_i, q_sp)

        res = np.vstack([q_cc, dE + self._offset[3]])
        if is_row:
            res = res.T
            if np.asarray(img_data).ndim == 1:
                return res.ravel()
        return res
