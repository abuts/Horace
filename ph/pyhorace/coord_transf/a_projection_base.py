"""Abstract base class for Horace coordinate projections (aProjectionBase)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import (
    AxesBlockBase,
    HoraceError,
    PixelDataBase,
    PixelDataMemory,
    line_axes,
)
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.core.serializable import Serializable


class aProjectionBase(Serializable, ABC):
    """Abstract base class defining coordinate projections between pixel and image spaces.

    Defines crystal lattice parameters, projection offsets, axis labels, and
    the common interface for transforming coordinates between Crystal Cartesian
    (pixel), reciprocal lattice (hkl), and image coordinate systems, as well
    as finding contributing cells and binning pixels.
    """

    tol_: ClassVar[float] = 1e-12
    init_params: ClassVar[list[str]] = ["alatt", "angdeg", "offset", "label", "title", "lab1", "lab2", "lab3", "lab4"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._alatt: np.ndarray = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
        self._angdeg: np.ndarray = np.array([90.0, 90.0, 90.0], dtype=float)
        self._alatt_defined: bool = False
        self._angdeg_defined: bool = False
        self._offset: np.ndarray = np.zeros(4, dtype=float)
        self._label: list[str] = ["\\zeta", "\\xi", "\\eta", "E"]
        self._title: str = ""
        self._type: list[str] = ["p", "p", "r"]
        self._img_scales: np.ndarray = np.ones(4, dtype=float)
        self._do_3D_transformation: bool = True
        self._do_generic: bool = False
        self._targ_proj: Optional["aProjectionBase"] = None
        self._disable_srce_to_targ_optimization: bool = False
        self._disable_pix_preselection: bool = False
        self._cur_axes_block_class_name: str = "line_axes"

        if args or kwargs:
            self.init(*args, **kwargs)

    def saveable_fields(self) -> list[str]:
        return ["alatt", "angdeg", "offset", "label", "title"]

    def init(self, *args: Any, **kwargs: Any) -> "aProjectionBase":
        """Initialize or update projection parameters."""
        if len(args) == 0 and not kwargs:
            return self
        if len(args) == 1 and isinstance(args[0], aProjectionBase) and not kwargs:
            other = args[0]
            for f in self.saveable_fields():
                if f == "alatt" and not other.alatt_defined:
                    continue
                if f == "angdeg" and not other.angdeg_defined:
                    continue
                if hasattr(other, f) and hasattr(self, f):
                    setattr(self, f, copy.deepcopy(getattr(other, f)))
            return self
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            self.from_bare_struct(args[0])
            return self

        pos_names = getattr(self, "positional_arg_names_", self.init_params)
        self.set_positional_and_key_val_arguments(pos_names, False, *args, **kwargs)
        if self._do_check_combo_arg:
            self.check_combo_arg()
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def alatt(self) -> np.ndarray:
        if not self._alatt_defined:
            return np.array([], dtype=float)
        return self._alatt.copy()

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._set_alatt(val)

    def _set_alatt(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._alatt = np.array([2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi], dtype=float)
            self._alatt_defined = False
            return
        if isinstance(val, (str, bytes)) or isinstance(val, dict):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "alatt must be a 1 or 3 element positive numeric vector",
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(np.isnan(arr)) or np.any(arr <= self.tol_):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "alatt must be a 1 or 3 element positive numeric vector",
            )
        self._alatt = arr.astype(float)
        self._alatt_defined = True

    @property
    def angdeg(self) -> np.ndarray:
        if not self._angdeg_defined:
            return np.array([], dtype=float)
        return self._angdeg.copy()

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._set_angdeg(val)

    def _set_angdeg(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._angdeg = np.array([90.0, 90.0, 90.0], dtype=float)
            self._angdeg_defined = False
            return
        if isinstance(val, (str, bytes)) or isinstance(val, dict):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "angdeg must be a 1 or 3 element numeric vector in (0, 180)",
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1:
            arr = np.repeat(arr, 3)
        if arr.size != 3 or np.any(np.isnan(arr)) or np.any(arr <= self.tol_) or np.any(arr >= 180.0 - self.tol_):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "angdeg must be a 1 or 3 element numeric vector in (0, 180)",
            )
        # Check triangle inequality for crystal angles: sum < 360 and each < sum of other two
        if np.sum(arr) >= 360.0 - self.tol_:
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "Sum of lattice angles must be less than 360 degrees",
            )
        self._angdeg = arr.astype(float)
        self._angdeg_defined = True

    @property
    def alatt_defined(self) -> bool:
        return self._alatt_defined

    @property
    def angdeg_defined(self) -> bool:
        return self._angdeg_defined

    @property
    def lattice_defined(self) -> bool:
        return self._alatt_defined and self._angdeg_defined

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        self._set_offset(val)

    def _set_offset(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._offset = np.zeros(4, dtype=float)
            return
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 1 and arr[0] == 0.0:
            arr = np.zeros(4, dtype=float)
        elif arr.size == 3:
            arr = np.append(arr, 0.0)
        elif arr.size != 4:
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        arr[np.abs(arr) < self.tol_] = 0.0
        self._offset = arr.astype(float)

    @property
    def label(self) -> list[str]:
        return list(self._label)

    @label.setter
    def label(self, val: Any) -> None:
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 4:
            if not all(isinstance(x, str) for x in val):
                raise HoraceError(
                    "HORACE:aProjectionBase:invalid_argument",
                    "label must be a 4-element sequence of strings",
                )
            self._label = [str(x) for x in val]
        else:
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "label must be a 4-element cell array / sequence of strings",
            )

    @property
    def lab1(self) -> str:
        return self._label[0]

    @lab1.setter
    def lab1(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceError("HORACE:aProjectionBase:invalid_argument", "lab1 must be a string")
        self._label[0] = val

    @property
    def lab2(self) -> str:
        return self._label[1]

    @lab2.setter
    def lab2(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceError("HORACE:aProjectionBase:invalid_argument", "lab2 must be a string")
        self._label[1] = val

    @property
    def lab3(self) -> str:
        return self._label[2]

    @lab3.setter
    def lab3(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceError("HORACE:aProjectionBase:invalid_argument", "lab3 must be a string")
        self._label[2] = val

    @property
    def lab4(self) -> str:
        return self._label[3]

    @lab4.setter
    def lab4(self, val: str) -> None:
        if not isinstance(val, str):
            raise HoraceError("HORACE:aProjectionBase:invalid_argument", "lab4 must be a string")
        self._label[3] = val

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, val: Any) -> None:
        if val is None or (isinstance(val, np.ndarray) and val.size == 0):
            self._title = ""
        elif isinstance(val, str):
            self._title = val
        else:
            raise HoraceError("HORACE:aProjectionBase:invalid_argument", "title must be a string")

    @property
    def type(self) -> str:
        return "".join(self._type)

    @type.setter
    def type(self, val: Any) -> None:
        self._set_type(val)

    def _set_type(self, val: Any) -> None:
        if not isinstance(val, str) or len(val) != 3:
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "type must be a 3-character string",
            )
        self._type = [c.lower() for c in val]

    @property
    def img_scales(self) -> np.ndarray:
        return self._img_scales.copy()

    @img_scales.setter
    def img_scales(self, val: Any) -> None:
        self._set_img_scales(val)

    def _set_img_scales(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.append(arr, 1.0)
        if arr.size != 4 or np.any(arr <= 0):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "img_scales must be 3 or 4 positive numbers",
            )
        self._img_scales = arr.copy()

    @property
    def do_3D_transformation(self) -> bool:
        return self._do_3D_transformation

    @do_3D_transformation.setter
    def do_3D_transformation(self, val: Any) -> None:
        self._do_3D_transformation = bool(val)

    @property
    def do_generic(self) -> bool:
        return self._do_generic

    @do_generic.setter
    def do_generic(self, val: Any) -> None:
        self._do_generic = bool(val)

    @property
    def targ_proj(self) -> Optional["aProjectionBase"]:
        return self._targ_proj

    @targ_proj.setter
    def targ_proj(self, val: Optional["aProjectionBase"]) -> None:
        if val is not None and not isinstance(val, aProjectionBase):
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                "targ_proj must be an instance of aProjectionBase or None",
            )
        if val is not None:
            if not val.alatt_defined and self.alatt_defined:
                val.alatt = self.alatt
            if not val.angdeg_defined and self.angdeg_defined:
                val.angdeg = self.angdeg
            self._do_3D_transformation = val.do_3D_transformation
        self._targ_proj = val
        self._check_and_set_targ_proj_optimization(val)

    def _check_and_set_targ_proj_optimization(self, targ: Optional["aProjectionBase"]) -> None:
        pass

    @property
    def disable_srce_to_targ_optimization(self) -> bool:
        return self._disable_srce_to_targ_optimization

    @disable_srce_to_targ_optimization.setter
    def disable_srce_to_targ_optimization(self, val: Any) -> None:
        self._disable_srce_to_targ_optimization = bool(val)
        if self._targ_proj is not None:
            self._check_and_set_targ_proj_optimization(self._targ_proj)

    @property
    def disable_pix_preselection(self) -> bool:
        return self._disable_pix_preselection

    @disable_pix_preselection.setter
    def disable_pix_preselection(self, val: Any) -> None:
        self._disable_pix_preselection = bool(val)

    @property
    def can_mex_cut(self) -> bool:
        return False

    def copy_proj_param_from_source(self, cut_source: Any) -> "aProjectionBase":
        """Copy lattice parameters (alatt, angdeg) from cut_source projection if not already defined."""
        source_proj = (
            cut_source.data.proj
            if hasattr(cut_source, "data") and hasattr(cut_source.data, "proj")
            else cut_source.proj
        )
        self._do_check_combo_arg = False
        try:
            if not self.alatt_defined and source_proj.alatt_defined:
                self.alatt = source_proj.alatt
            if not self.angdeg_defined and source_proj.angdeg_defined:
                self.angdeg = source_proj.angdeg
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()
        return self

    # ------------------------------------------------------------------
    # Abstract & Core Coordinate Transformation Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def transform_pix_to_img(self, pix_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform pixel coordinates (Crystal Cartesian) into image coordinates."""

    @abstractmethod
    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform image coordinates into pixel coordinates (Crystal Cartesian)."""

    def bmatrix(self, ndim: int = 3, nargout: Optional[int] = None) -> Any:
        """Return B-matrix (3x3 or 4x4) for this projection's lattice."""
        if nargout is None:
            nargout = _infer_nargout(default=1)
        b, arlu, angrlu = bmatrix(self._alatt, self._angdeg, nargout=3)
        if ndim == 4:
            b4 = np.eye(4, dtype=float)
            b4[:3, :3] = b
            b = b4
        elif ndim != 3:
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                f"bmatrix dimension must be 3 or 4, got {ndim}",
            )
        if nargout == 1:
            return b
        if nargout == 2:
            return b, arlu
        return b, arlu, angrlu

    @staticmethod
    def check_and_preprocess_input_coords(
        pix_input: Any,
    ) -> Tuple[np.ndarray, np.ndarray, bool, int]:
        """Extract numeric coordinate array (3xN or 4xN) and energy row if present."""
        if isinstance(pix_input, PixelDataBase):
            coords = pix_input.coordinates
            return coords[:3, :], coords[3:4, :], False, 4
        arr = np.asarray(pix_input, dtype=float)
        if arr.ndim == 1:
            if arr.size in (3, 4):
                col = arr.reshape(-1, 1)
                is_row = True
                if arr.size == 4:
                    return col[:3, :], col[3:4, :], is_row, 4
                return col, np.zeros((1, 1), dtype=float), is_row, 3
            raise HoraceError(
                "HORACE:aProjectionBase:invalid_argument",
                f"Coordinate vector must have 3 or 4 elements, got {arr.size}",
            )
        if arr.shape[0] in (3, 4):
            if arr.shape[0] == 4:
                return arr[:3, :], arr[3:4, :], False, 4
            return arr, np.zeros((1, arr.shape[1]), dtype=float), False, 3
        if arr.shape[0] == 9:
            return arr[:3, :], arr[3:4, :], False, 4
        if arr.shape[1] in (3, 4):
            col = arr.T
            if col.shape[0] == 4:
                return col[:3, :], col[3:4, :], True, 4
            return col, np.zeros((1, col.shape[1]), dtype=float), True, 3
        raise HoraceError(
            "HORACE:aProjectionBase:invalid_argument",
            f"Invalid coordinate array shape {arr.shape}; expected (3, N) or (4, N)",
        )

    def transform_pix_to_hkl(self, pix_input: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform pixel coordinates (Crystal Cartesian) to Miller indices (hkl)."""
        q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_input)
        b = self.bmatrix(3, nargout=1)
        hkl = np.linalg.solve(b, q_cc)
        if ndim == 4 and not self._do_3D_transformation:
            res = np.vstack([hkl, dE - self._offset[3]])
        elif ndim == 4:
            res = np.vstack([hkl, dE])
        else:
            res = hkl
        if is_row:
            res = res.T
            if np.asarray(pix_input).ndim == 1:
                return res.ravel()
        return res

    def transform_hkl_to_pix(self, pix_hkl: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform Miller indices (hkl) to pixel coordinates (Crystal Cartesian)."""
        hkl, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_hkl)
        b = self.bmatrix(3, nargout=1)
        q_cc = b @ hkl
        if ndim == 4 and not self._do_3D_transformation:
            res = np.vstack([q_cc, dE + self._offset[3]])
        elif ndim == 4:
            res = np.vstack([q_cc, dE])
        else:
            res = q_cc
        if is_row:
            res = res.T
            if np.asarray(pix_hkl).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_hkl(self, pix_img: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform image coordinates into hkl coordinates."""
        pix_cc = self.transform_img_to_pix(pix_img, *args, **kwargs)
        return self.transform_pix_to_hkl(pix_cc, *args, **kwargs)

    def transform_hkl_to_img(self, pix_hkl: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform hkl coordinates into image coordinates."""
        pix_cc = self.transform_hkl_to_pix(pix_hkl, *args, **kwargs)
        return self.transform_pix_to_img(pix_cc, *args, **kwargs)

    def from_this_to_targ_coord(self, pix_origin: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform coordinates from this projection's image space to targ_proj's image space."""
        if self._targ_proj is None:
            raise HoraceError(
                "HORACE:aProjectionBase:runtime_error",
                "Target projection (targ_proj) is not set",
            )
        pix_cc = self.transform_img_to_pix(pix_origin, *args, **kwargs)
        return self._targ_proj.transform_pix_to_img(pix_cc, *args, **kwargs)

    # ------------------------------------------------------------------
    # Cell Contribution & Range Methods
    # ------------------------------------------------------------------
    @staticmethod
    def convert_3Dplus1Ind_to_4Dind_ranges(
        bin_inside3D: np.ndarray,
        en_inside: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 3D contributing cell mask + 1D energy mask into 4D start/end block ranges."""
        arr3 = np.asarray(bin_inside3D)
        arr1 = np.asarray(en_inside)
        if arr3.dtype == bool and arr1.dtype == bool:
            b3_flat = arr3.ravel(order="F")
            if not np.any(b3_flat) or not np.any(arr1):
                return np.array([], dtype=int), np.array([], dtype=int)
            padded = np.concatenate([[False], b3_flat, [False]])
            edges = np.diff(padded.astype(int))
            istart_3d = np.where(edges == 1)[0] + 1
            iend_3d = np.where(edges == -1)[0]
            q_stride = np.where(arr1.ravel(order="F"))[0] * b3_flat.size
            istart = (istart_3d[:, np.newaxis] + q_stride[np.newaxis, :]).ravel(order="F")
            iend = (iend_3d[:, np.newaxis] + q_stride[np.newaxis, :]).ravel(order="F")
            if istart.size > 1:
                adj = iend[:-1] + 1 == istart[1:]
                if np.any(adj):
                    keep_start = np.concatenate([[True], ~adj])
                    keep_end = np.concatenate([~adj, [True]])
                    istart = istart[keep_start]
                    iend = iend[keep_end]
            return istart.astype(int), iend.astype(int)

        ind3 = np.asarray(bin_inside3D, dtype=int).ravel()
        rng1 = np.asarray(en_inside, dtype=int).ravel()
        if ind3.size == 0 or rng1.size < 2:
            return np.array([], dtype=int), np.array([], dtype=int)

        ind3 = np.sort(ind3)
        diffs = np.diff(ind3)
        split_pts = np.where(diffs > 1)[0] + 1
        runs = np.split(ind3, split_pts)
        starts_3d = np.array([r[0] for r in runs], dtype=int)
        ends_3d = np.array([r[-1] for r in runs], dtype=int)
        lengths_3d = ends_3d - starts_3d + 1
        return starts_3d, lengths_3d

    @staticmethod
    def convert_contrib_cell_into_pix_indexes(
        cell_ind: Any,
        npix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert contributing cell indices or (istart, iend) tuple into pixel block (start, length)."""
        if cell_ind is None:
            return np.array([], dtype=int), np.array([], dtype=int)
        if isinstance(cell_ind, tuple) and len(cell_ind) == 2:
            c_start = np.asarray(cell_ind[0], dtype=int).ravel(order="F")
            c_end = np.asarray(cell_ind[1], dtype=int).ravel(order="F")
            if c_start.size == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
        else:
            ind_arr = np.asarray(cell_ind, dtype=int).ravel(order="F")
            if ind_arr.size == 0:
                return np.array([], dtype=int), np.array([], dtype=int)
            adj = (ind_arr[:-1] + 1 == ind_arr[1:]) if ind_arr.size > 1 else np.array([], dtype=bool)
            c_start = ind_arr[np.concatenate([[True], ~adj])]
            c_end = ind_arr[np.concatenate([~adj, [True]])]

        npix_flat = np.asarray(npix, dtype=int).ravel(order="F")
        pix_start = np.concatenate([[0], np.cumsum(npix_flat)])
        bl_start = pix_start[c_start - 1]
        bl_end = pix_start[c_end]
        bl_size = bl_end - bl_start
        non_empty = bl_size != 0
        if not np.any(non_empty):
            return np.array([], dtype=int), np.array([], dtype=int)
        return (bl_start[non_empty] + 1).astype(int), bl_size[non_empty].astype(int)

    def may_contribute(
        self,
        source_axes_block: AxesBlockBase,
        targ_proj: "aProjectionBase",
        targ_axes_block: AxesBlockBase,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find 3D cell indices and 1D energy bin range of source_axes_block that may contribute to targ_axes_block."""
        # Energy range check (4th axis)
        s_en_edges = np.linspace(
            source_axes_block.img_range[0, 3],
            source_axes_block.img_range[1, 3],
            int(source_axes_block.nbins_all_dims[3]) + 1,
        ) + self._offset[3]
        t_en_range = targ_axes_block.img_range[:, 3] + targ_proj.offset[3]
        any_en, is_within_en = AxesBlockBase.bins_in_1Drange(s_en_edges, t_en_range)
        if not any_en:
            return np.array([], dtype=int), np.array([], dtype=int)
        idx_en = np.where(is_within_en)[0]
        en_ind_range = np.array([int(idx_en[0]) + 1, int(idx_en[-1]) + 1], dtype=int)

        self.targ_proj = targ_proj
        targ_proj.targ_proj = self

        # 1. Forward check: transform source bin centers AND bin corners into target image space
        s_centres = source_axes_block.get_bin_nodes("-3D", "-bin_centre", nargout=1)
        t_from_s_centres = self.from_this_to_targ_coord(s_centres)
        t_range_3d = targ_axes_block.img_range[:, :3]
        s_nbins_3d = source_axes_block.nbins_all_dims[:3]
        n_s_cells = int(np.prod(s_nbins_3d))

        # Check which source centres fall inside target range (with half-step tolerance)
        s_steps = source_axes_block.step[:3]
        in_targ = np.ones(n_s_cells, dtype=bool)
        for d in range(3):
            in_targ &= (t_from_s_centres[d, :] >= t_range_3d[0, d] - 1e-9) & (
                t_from_s_centres[d, :] <= t_range_3d[1, d] + 1e-9
            )
        forward_ind = np.where(in_targ)[0] + 1

        # Also check all 8 corners of each source cell transformed to target space
        # If the bounding box of a source cell in target space overlaps target img_range, it may contribute!
        s_nodes = source_axes_block.get_bin_nodes("-3D", nargout=1)
        t_nodes = self.from_this_to_targ_coord(s_nodes)
        nx, ny, nz = int(s_nbins_3d[0]), int(s_nbins_3d[1]), int(s_nbins_3d[2])
        t_nodes_grid = t_nodes.reshape((3, nx + 1, ny + 1, nz + 1), order="F")

        # Compute min/max over the 8 corners of each cell (nx, ny, nz)
        corners_list = [
            t_nodes_grid[:, 0:nx, 0:ny, 0:nz],
            t_nodes_grid[:, 1 : nx + 1, 0:ny, 0:nz],
            t_nodes_grid[:, 0:nx, 1 : ny + 1, 0:nz],
            t_nodes_grid[:, 0:nx, 0:ny, 1 : nz + 1],
            t_nodes_grid[:, 1 : nx + 1, 1 : ny + 1, 0:nz],
            t_nodes_grid[:, 1 : nx + 1, 0:ny, 1 : nz + 1],
            t_nodes_grid[:, 0:nx, 1 : ny + 1, 1 : nz + 1],
            t_nodes_grid[:, 1 : nx + 1, 1 : ny + 1, 1 : nz + 1],
        ]
        c_stack = np.stack(corners_list, axis=0)  # (8, 3, nx, ny, nz)
        cell_min = np.min(c_stack, axis=0).reshape((3, n_s_cells), order="F")
        cell_max = np.max(c_stack, axis=0).reshape((3, n_s_cells), order="F")

        # Handle angular wrap-around in spherical/cylindrical phi if span > 180
        tol = 1e-9
        bbox_overlap = np.ones(n_s_cells, dtype=bool)
        for d in range(3):
            bbox_overlap &= (cell_max[d, :] >= t_range_3d[0, d] - tol) & (
                cell_min[d, :] <= t_range_3d[1, d] + tol
            )
        bbox_ind = np.where(bbox_overlap)[0] + 1

        # 2. Inverse check: sample target grid nodes and bin them into source_axes_block
        char_sizes = source_axes_block.get_char_size(self)
        interp_nodes, _ = targ_axes_block.get_interp_nodes(targ_proj, char_sizes)
        s_from_t_nodes = targ_proj.from_this_to_targ_coord(interp_nodes)

        ax_3d = copy.deepcopy(source_axes_block)
        npix_3d = ax_3d.bin_pixels(s_from_t_nodes, nargout=1)
        inv_ind = np.where(npix_3d.ravel(order="F") > 0)[0] + 1

        contrib_3d = np.unique(np.concatenate([forward_ind, bbox_ind, inv_ind])).astype(int)
        return contrib_3d, en_ind_range

    def get_contrib_cell_ind(
        self,
        cur_axes_block: AxesBlockBase,
        targ_proj: "aProjectionBase",
        targ_axes_block: AxesBlockBase,
    ) -> np.ndarray:
        """Return 1-based flat 4D cell indices of cur_axes_block that may contribute to targ_axes_block."""
        contrib_3d, en_range = self.may_contribute(cur_axes_block, targ_proj, targ_axes_block)
        if contrib_3d.size == 0 or en_range.size == 0:
            return np.array([], dtype=int)

        n_3d_total = int(np.prod(cur_axes_block.nbins_all_dims[:3]))
        en_indices = np.arange(en_range[0] - 1, en_range[1], dtype=int)
        out = []
        for e_idx in en_indices:
            out.append(contrib_3d + e_idx * n_3d_total)
        return np.concatenate(out).astype(int)

    def get_nrange(
        self,
        npix: np.ndarray,
        cur_axes_block: AxesBlockBase,
        targ_axes_block: AxesBlockBase,
        targ_proj: Optional["aProjectionBase"] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate start positions and lengths of pixel blocks in npix contributing to the cut."""
        if targ_proj is not None:
            if isinstance(targ_proj, type(self)):
                self.do_generic = targ_proj.do_generic
                self.disable_srce_to_targ_optimization = targ_proj.disable_srce_to_targ_optimization
            targ_proj.targ_proj = self
            self.targ_proj = targ_proj
            if targ_proj.disable_pix_preselection:
                return np.array([1], dtype=int), np.array([int(np.sum(npix))], dtype=int)

        cell_ind = self.get_contrib_cell_ind(cur_axes_block, targ_proj, targ_axes_block)
        return self.convert_contrib_cell_into_pix_indexes(cell_ind, npix)

    # ------------------------------------------------------------------
    # Axes Block Construction & Binning Methods
    # ------------------------------------------------------------------
    def get_proj_axes_block(
        self,
        default_bounds_cell_ind: Any,
        req_bin_range: Sequence[Any],
        *args: Any,
    ) -> AxesBlockBase:
        """Build an AxesBlockBase for this projection from default bounds and requested bin descriptors."""
        axes_cls = self._get_axes_class()
        ax = axes_cls()
        if hasattr(ax, "angular_unit_is_rad") and hasattr(self, "type"):
            t_str = str(self.type).lower()
            if len(t_str) >= 3 and "r" in t_str[1:3]:
                ax.angular_unit_is_rad = True
            elif len(t_str) >= 3 and "d" in t_str[1:3]:
                ax.angular_unit_is_rad = False

        try:
            arr_bounds = np.asarray(default_bounds_cell_ind)
            is_bool_4 = arr_bounds.dtype == bool and arr_bounds.size == 4
        except ValueError:
            arr_bounds = None
            is_bool_4 = False

        if is_bool_4:
            # MATLAB signature: get_proj_axes_block(cell_is_iax, pbin_or_ranges)
            sbd_in = arr_bounds.ravel()
            pbin_in = list(req_bin_range)
            ax.init(*pbin_in)
            ax.single_bin_defines_iax = sbd_in
            return self.copy_proj_defined_properties_to_axes(ax)

        if isinstance(default_bounds_cell_ind, (list, tuple)) and len(default_bounds_cell_ind) == 4:
            ax = AxesBlockBase.build_from_input_binning(ax, default_bounds_cell_ind, req_bin_range)
            return self.copy_proj_defined_properties_to_axes(ax)

        img_db_range = np.asarray(default_bounds_cell_ind, dtype=float)
        if img_db_range.shape != (2, 4):
            img_db_range = ax._default_img_range.copy()

        img_range = np.zeros((2, 4), dtype=float)
        nbins = np.ones(4, dtype=int)
        sbd = np.ones(4, dtype=bool)

        for i in range(4):
            pb = np.asarray(req_bin_range[i], dtype=float).ravel() if i < len(req_bin_range) else np.array([])
            def_lo, def_hi = img_db_range[0, i], img_db_range[1, i]
            if pb.size == 0:
                img_range[:, i] = [def_lo, def_hi]
                nbins[i] = 1
                sbd[i] = True
            elif pb.size == 1:
                step = pb[0]
                if step <= 0:
                    img_range[:, i] = [def_lo, def_hi]
                    nbins[i] = 1
                    sbd[i] = False
                else:
                    sbd[i] = False
                    n_steps = max(1, int(np.ceil((def_hi - def_lo) / step)))
                    img_range[:, i] = [def_lo, def_lo + n_steps * step]
                    nbins[i] = n_steps
            elif pb.size == 2:
                lo = def_lo if np.isnan(pb[0]) or np.isinf(pb[0]) else pb[0]
                hi = def_hi if np.isnan(pb[1]) or np.isinf(pb[1]) else pb[1]
                img_range[:, i] = [lo, hi]
                nbins[i] = 1
                sbd[i] = True
            elif pb.size == 3:
                lo = def_lo if np.isnan(pb[0]) or np.isinf(pb[0]) else pb[0]
                step = pb[1]
                hi = def_hi if np.isnan(pb[2]) or np.isinf(pb[2]) else pb[2]
                if step == 0:
                    img_range[:, i] = [lo, hi]
                    nbins[i] = 1
                    sbd[i] = True
                else:
                    sbd[i] = False
                    n_steps = max(1, int(np.floor((hi - lo + 1e-12 * abs(step)) / step)) + 1)
                    actual_hi = lo + (n_steps - 1) * step
                    img_range[:, i] = [lo - 0.5 * step, actual_hi + 0.5 * step]
                    nbins[i] = n_steps
            else:
                img_range[:, i] = [pb[0], pb[-1]]
                nbins[i] = max(1, pb.size - 1)
                sbd[i] = False

        max_r = ax._max_img_range
        img_range[0, :] = np.maximum(img_range[0, :], max_r[0, :])
        img_range[1, :] = np.minimum(img_range[1, :], max_r[1, :])

        ax = axes_cls(img_range=img_range, nbins_all_dims=nbins, single_bin_defines_iax=sbd)
        return self.copy_proj_defined_properties_to_axes(ax)

    def _get_axes_class(self) -> type:
        return line_axes

    @property
    def axes_name(self) -> str:
        return self._get_axes_class().__name__

    def copy_proj_defined_properties_to_axes(self, axes_bl: AxesBlockBase) -> AxesBlockBase:
        """Copy projection-defined properties (offset, label, title, img_scales, type) to axes_bl."""
        if hasattr(axes_bl, "angular_unit_is_rad") and hasattr(self, "type"):
            t_str = str(self.type).lower()
            if len(t_str) >= 3 and "r" in t_str[1:3]:
                axes_bl.angular_unit_is_rad = True
            elif len(t_str) >= 3 and "d" in t_str[1:3]:
                axes_bl.angular_unit_is_rad = False
        axes_bl.offset = self.offset
        axes_bl.label = self.label
        if self.title:
            axes_bl.title = self.title
        axes_bl.img_scales = self.img_scales
        t_val = str(self.type)
        if len(t_val) == 3:
            t_val = t_val + "e"
        axes_bl.type = t_val[:4]
        return axes_bl

    def bin_pixels(
        self,
        axes_bl: AxesBlockBase,
        in_pix: Any,
        npix: Any = None,
        s: Any = None,
        e: Any = None,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Transform input pixels into image space and bin them into axes_bl."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        if isinstance(npix, str):
            extra = [npix]
            if s is not None:
                extra.append(s)
            if e is not None:
                extra.append(e)
            args = tuple(extra) + args
            npix = None
            s = None
            e = None

        test_input = False
        for a in args:
            if isinstance(a, str) and a.lower() == "-test_input":
                test_input = True

        ndim = 3 if (s is None and e is None and self._do_3D_transformation and not test_input) else 4
        if isinstance(in_pix, PixelDataBase):
            if in_pix.num_pixels == 0:
                coord_img = np.zeros((ndim, 0), dtype=float)
            else:
                coord_img = self.transform_pix_to_img(in_pix)
                if coord_img.shape[0] == 3 and ndim == 4:
                    coord_img = np.vstack([coord_img, in_pix.dE - self._offset[3]])
                elif coord_img.shape[0] == 4 and ndim == 3:
                    coord_img = coord_img[:3, :]
        else:
            coord_img = self.transform_pix_to_img(in_pix)

        if nargout == 1 and s is None and e is None:
            if npix is None:
                return axes_bl.bin_pixels(coord_img, *args, nargout=1)
            return axes_bl.bin_pixels(coord_img, npix, *args, nargout=1)

        return axes_bl.bin_pixels(coord_img, npix, s, e, in_pix, *args, nargout=nargout)

    def align_proj(self, alignment_info: Any) -> "aProjectionBase":
        """Align projection using crystal_alignment_info."""
        self.alatt = alignment_info.alatt
        self.angdeg = alignment_info.angdeg
        return self
