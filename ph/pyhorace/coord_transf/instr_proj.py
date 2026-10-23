"""Instrument-to-Crystal-Cartesian coordinate projection class (instr_proj)."""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError, oriented_lattice
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.coord_transf.kf_sphere_proj import K_TO_E_MEV


class instr_proj(aProjectionBase):
    """Projection converting detector/spectrometer coordinates into Crystal Cartesian pixel coordinates."""

    def class_version(self) -> int:
        return 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._lattice: oriented_lattice = oriented_lattice()
        self._efix: float = 0.0
        self._emode: int = 1
        self._detdcn_cache: Optional[np.ndarray] = None
        self._label = ["Q_\\zeta", "Q_\\xi", "Q_\\eta", "E"]
        if len(args) == 1 and isinstance(args[0], oriented_lattice) and not kwargs:
            self.lattice = args[0]
        elif args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return ["lattice", "efix", "emode", "offset", "label", "title"]

    @property
    def lattice(self) -> oriented_lattice:
        return self._lattice

    @lattice.setter
    def lattice(self, val: oriented_lattice) -> None:
        if not isinstance(val, oriented_lattice):
            raise HoraceError(
                "HORACE:instr_proj:invalid_argument",
                "lattice must be an oriented_lattice instance",
            )
        self._lattice = copy.deepcopy(val)
        if val.is_defined("alatt"):
            self._set_alatt(val.alatt)
        if val.is_defined("angdeg"):
            self._set_angdeg(val.angdeg)

    @property
    def efix(self) -> float:
        return self._efix

    @efix.setter
    def efix(self, val: Any) -> None:
        v = float(np.asarray(val, dtype=float).item())
        if v < 0:
            raise HoraceError("HORACE:instr_proj:invalid_argument", "efix must be non-negative")
        self._efix = v

    @property
    def emode(self) -> int:
        return self._emode

    @emode.setter
    def emode(self, val: Any) -> None:
        v = int(val)
        if v not in (0, 1, 2):
            raise HoraceError("HORACE:instr_proj:invalid_argument", "emode must be 0, 1, or 2")
        self._emode = v

    @property
    def detdcn_cache(self) -> Optional[np.ndarray]:
        return None if self._detdcn_cache is None else self._detdcn_cache.copy()

    @detdcn_cache.setter
    def detdcn_cache(self, val: Optional[np.ndarray]) -> None:
        self._detdcn_cache = None if val is None else np.asarray(val, dtype=float).copy()

    def get_spec_to_cc_mat(self, n_matrix: int = 1) -> np.ndarray:
        """Return 3x3 matrix transforming spectrometer coordinates to Crystal Cartesian (or rlu)."""
        return self._lattice.calc_proj_matrix(n_matrix, nargout=1)

    def transform_pix_to_img(self, pix_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform detector/energy data into Crystal Cartesian coordinates."""
        q_in, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_data)
        spec_to_cc = self.get_spec_to_cc_mat(1)
        res_3d = spec_to_cc @ q_in
        if ndim == 4:
            res = np.vstack([res_3d, dE - self._offset[3]])
        else:
            res = res_3d
        if is_row:
            res = res.T
            if np.asarray(pix_data).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform Crystal Cartesian coordinates back to spectrometer coordinates."""
        q_in, dE, is_row, ndim = self.check_and_preprocess_input_coords(img_data)
        spec_to_cc = self.get_spec_to_cc_mat(1)
        res_3d = np.linalg.solve(spec_to_cc, q_in)
        if ndim == 4:
            res = np.vstack([res_3d, dE + self._offset[3]])
        else:
            res = res_3d
        if is_row:
            res = res.T
            if np.asarray(img_data).ndim == 1:
                return res.ravel()
        return res
