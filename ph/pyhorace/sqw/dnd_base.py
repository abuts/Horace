"""Abstract base class for n-dimensional Horace DnD datasets (DnDBase).

Ported from Horace MATLAB @DnDBase class
(horace_core/sqw/@DnDBase).
"""

from __future__ import annotations

import copy
import os
from abc import ABC
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.ndimage
from pydantic import PrivateAttr

from pyhorace.coord_transf._dependencies import HoraceError, PixelDataMemory
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.axes_block_base import AxesBlockBase
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.coord_transf.cylinder_axes import cylinder_axes
from pyhorace.coord_transf.line_axes import line_axes
from pyhorace.coord_transf.line_proj import line_proj
from pyhorace.coord_transf.sphere_axes import sphere_axes
from pyhorace.core.configurations.config_store import warn_with_id
from pyhorace.core.data_op_interface import (
    DataOpArray,
    _is_numeric,
    _is_object_or_object_array,
    matlab_size,
)
from pyhorace.core.serializable import Serializable, compare_values, register_serializable
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw.data_sqw_dnd import data_sqw_dnd
from pyhorace.sqw.dnd_data import dnd_data
from pyhorace.sqw.dnd_metadata import dnd_metadata
from pyhorace.sqw.dnd_plot_interface import dnd_plot_interface
from pyhorace.sqw.horace3_dnd_interface import horace3_dnd_interface
from pyhorace.sqw.sqw_dnd_base import SQWDnDBase, _call_user_func, _parse_char_options


def _axes_cls_for_name(name: str) -> type:
    mapping = {
        "line_axes": line_axes,
        "sphere_axes": sphere_axes,
        "cylinder_axes": cylinder_axes,
    }
    return mapping.get(name, AxesBlockBase)


class _DimensionsInt(int):
    """Integer subclass that can also be called as a method returning `(nd, sz)` when nargout >= 2."""

    def __new__(cls, nd: int, sz: Tuple[int, ...] = ()) -> "_DimensionsInt":
        obj = super().__new__(cls, nd)
        obj._sz = sz
        return obj

    def __call__(self, nargout: Optional[int] = None) -> Any:
        if nargout is None:
            nargout = _infer_nargout(default=1)
        if nargout >= 2:
            return int(self), self._sz
        return int(self)


def _fmt_matlab_sz(sz: Tuple[int, ...]) -> str:
    return "  ".join(str(x) for x in sz)


def _reshape_data_for_axes(arr: np.ndarray, target_dims: Tuple[int, ...]) -> np.ndarray:
    """Reshape array to match target_dims in Fortran order."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape == target_dims:
        return arr.copy()
    if len(target_dims) == 2 and target_dims[1] == 1 and arr.ndim == 1 and arr.shape[0] == target_dims[0]:
        return arr.copy()
    if len(target_dims) == 2 and target_dims == (1, 1) and arr.size == 1:
        return arr.reshape((1, 1)).copy()
    if arr.size == int(np.prod(target_dims)):
        if len(target_dims) == 2 and target_dims[1] == 1 and target_dims[0] > 1:
            return arr.reshape((target_dims[0],), order="F").copy()
        return arr.reshape(target_dims, order="F").copy()
    return arr.copy()


@register_serializable
class DnDBase(SQWDnDBase, horace3_dnd_interface, dnd_plot_interface, ABC):
    """Abstract base class for 0D-4D Horace DnD objects (d0d, d1d, d2d, d3d, d4d)."""

    class_version_: ClassVar[int] = 4
    NUM_DIMS: ClassVar[Optional[int]] = None

    _fields_to_save: ClassVar[List[str]] = [
        "axes",
        "proj",
        "s",
        "e",
        "npix",
        "creation_date",
    ]

    _axes: AxesBlockBase = PrivateAttr(default=None)
    _proj: aProjectionBase = PrivateAttr(default=None)
    _s: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((1, 1), dtype=float))
    _e: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((1, 1), dtype=float))
    _npix: np.ndarray = PrivateAttr(default_factory=lambda: np.zeros((1, 1), dtype=float))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        in_d0d_init = getattr(self, "_in_d0d_init", False)
        Serializable.__init__(self)
        self._in_d0d_init = in_d0d_init
        self._do_check_combo_arg = False

        ndims = self.NUM_DIMS if self.NUM_DIMS is not None else 0
        self._axes = line_axes(ndims)
        self._proj = line_proj(alatt=2 * np.pi, angdeg=90)
        self._init_arrays()

        if not args and not kwargs:
            self._do_check_combo_arg = True
            self.check_combo_arg(False)
            return

        if args and not kwargs:
            self._init_from_args(*args)
            return

        # Keyword or mixed arguments via Serializable
        pos_names = ["axes", "proj", "s", "e", "npix", "creation_date"]
        merged = dict(kwargs)
        for i, arg in enumerate(args):
            if i < len(pos_names) and pos_names[i] not in merged:
                merged[pos_names[i]] = arg
        self.set_positional_and_key_val_arguments(pos_names, False, **merged)
        self._do_check_combo_arg = True
        self.check_combo_arg(True)

    def _err_cls_name(self) -> str:
        return type(self).__name__

    def _init_arrays(self) -> None:
        sz = tuple(int(x) for x in self._axes.dims_as_ssize)
        self._s = np.zeros(sz, dtype=float)
        self._e = np.zeros(sz, dtype=float)
        self._npix = np.zeros(sz, dtype=float)

    def _init_from_args(self, *args: Any) -> None:
        # Check for spinw constructor: first arg is 6-element lattice_const vector
        if len(args) >= 1 and _is_numeric(args[0]) and np.asarray(args[0]).size == 6:
            res = self.spinw_dnd_obj_constructor(*args)
            if self.NUM_DIMS is not None and int(res.dimensions) != self.NUM_DIMS:
                if self.NUM_DIMS == 0 and getattr(self, "_in_d0d_init", False):
                    raise HoraceError(
                        "MATLAB:class:mustReturnObject",
                        "Constructor for class 'd0d' must return an object of that class.",
                    )
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Constructed {int(res.dimensions)}D object but expected {self.NUM_DIMS}D",
                )
            self._copy_from_dnd(res)
            self._do_check_combo_arg = True
            return

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (str, os.PathLike)):
                from pyhorace.sqw.read_dnd import read_dnd

                loaded = read_dnd(str(arg))
                if self.NUM_DIMS is not None and int(loaded.dimensions) != self.NUM_DIMS:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"File {arg} contains {int(loaded.dimensions)}D data, cannot load into {self._err_cls_name()}",
                    )
                self._copy_from_dnd(loaded)
                self._do_check_combo_arg = True
                return

            if isinstance(arg, DnDBase):
                if self.NUM_DIMS is not None and int(arg.dimensions) != self.NUM_DIMS:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Cannot construct {self._err_cls_name()} from {type(arg).__name__} ({int(arg.dimensions)}D)",
                    )
                self._copy_from_dnd(arg)
                self._do_check_combo_arg = True
                return

            # Check for sqw object (has .data attribute which is DnDBase or data_sqw_dnd)
            if hasattr(arg, "data") and isinstance(getattr(arg, "data"), (DnDBase, data_sqw_dnd)):
                d_inner = getattr(arg, "data")
                if isinstance(d_inner, DnDBase):
                    if self.NUM_DIMS is not None and int(d_inner.dimensions) != self.NUM_DIMS:
                        raise HoraceError(
                            f"HORACE:{self._err_cls_name()}:invalid_argument",
                            f"Cannot construct {self._err_cls_name()} from {int(d_inner.dimensions)}D sqw object",
                        )
                    self._copy_from_dnd(d_inner)
                    self._do_check_combo_arg = True
                    return
                arg = d_inner

            if isinstance(arg, data_sqw_dnd):
                if self.NUM_DIMS is not None and int(arg.dimensions) != self.NUM_DIMS:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Cannot construct {self._err_cls_name()} from {int(arg.dimensions)}D data_sqw_dnd",
                    )
                self._axes = copy.deepcopy(arg.axes)
                self._proj = copy.deepcopy(arg.proj)
                self._set_data_arrays_raw(arg.s, arg.e, arg.npix)
                self._do_check_combo_arg = True
                self.check_combo_arg(False)
                return

            if isinstance(arg, dict):
                self._init_from_dict(arg)
                return

            if hasattr(arg, "get_dnd"):
                loaded = arg.get_dnd()
                if self.NUM_DIMS is not None and int(loaded.dimensions) != self.NUM_DIMS:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Loader contains {int(loaded.dimensions)}D data, cannot load into {self._err_cls_name()}",
                    )
                self._copy_from_dnd(loaded)
                self._do_check_combo_arg = True
                return

            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Cannot construct {self._err_cls_name()} from single argument of type {type(arg).__name__}",
            )

        if len(args) == 2 and isinstance(args[0], dnd_metadata) and isinstance(args[1], dnd_data):
            meta, dat = args[0], args[1]
            self._axes = copy.deepcopy(meta.axes)
            self._proj = copy.deepcopy(meta.proj)
            if meta.creation_date_defined:
                self._axes.creation_date = meta.creation_date_str
            self._set_data_arrays_raw(dat.s, dat.e, dat.npix)
            self._do_check_combo_arg = True
            self.check_combo_arg(False)
            return

        if len(args) == 2 and isinstance(args[0], AxesBlockBase) and isinstance(args[1], aProjectionBase):
            self._axes = copy.deepcopy(args[0])
            self._proj = copy.deepcopy(args[1])
            self._init_arrays()
            self._do_check_combo_arg = True
            self.check_combo_arg(False)
            return

        if len(args) == 5 and isinstance(args[0], AxesBlockBase) and isinstance(args[1], aProjectionBase):
            self._axes = copy.deepcopy(args[0])
            self._proj = copy.deepcopy(args[1])
            self._set_data_arrays_raw(args[2], args[3], args[4])
            self._do_check_combo_arg = True
            self.check_combo_arg(False)
            return

        # Positional serializable arguments
        pos_names = ["axes", "proj", "s", "e", "npix", "creation_date"]
        if len(args) <= len(pos_names):
            kw = {pos_names[i]: args[i] for i in range(len(args))}
            self.set_positional_and_key_val_arguments(pos_names, False, **kw)
            self._do_check_combo_arg = True
            self.check_combo_arg(True)
            return

        raise HoraceError(
            f"HORACE:{self._err_cls_name()}:invalid_argument",
            f"Unsupported number of arguments ({len(args)}) to {self._err_cls_name()}",
        )

    def _init_from_dict(self, d: Dict[str, Any]) -> None:
        if "serial_name" in d or ("axes" in d and "proj" in d):
            if "axes" in d:
                ax = d["axes"]
                if isinstance(ax, dict):
                    ax = Serializable.from_struct(ax) if "serial_name" in ax else line_axes.get_from_old_data(ax)
                self._axes = copy.deepcopy(ax)
            if "proj" in d:
                pr = d["proj"]
                if isinstance(pr, dict):
                    pr = Serializable.from_struct(pr) if "serial_name" in pr else line_proj.get_from_old_data(pr)
                self._proj = copy.deepcopy(pr)
            if "s" in d and "e" in d and "npix" in d:
                self._set_data_arrays_raw(d["s"], d["e"], d["npix"])
            else:
                self._init_arrays()
            if "creation_date" in d and d["creation_date"]:
                self._axes.creation_date = d["creation_date"]
            self._do_check_combo_arg = True
            self.check_combo_arg(False)
            return

        # Legacy data_sqw_dnd structure
        dsd = data_sqw_dnd(d)
        if self.NUM_DIMS is not None and int(dsd.dimensions) != self.NUM_DIMS:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Expected {self.NUM_DIMS}D structure, got {int(dsd.dimensions)}D",
            )
        self._axes = copy.deepcopy(dsd.axes)
        self._proj = copy.deepcopy(dsd.proj)
        self._set_data_arrays_raw(dsd.s, dsd.e, dsd.npix)
        self._do_check_combo_arg = True
        self.check_combo_arg(False)

    def _copy_from_dnd(self, other: "DnDBase") -> None:
        self._axes = copy.deepcopy(other._axes)
        self._proj = copy.deepcopy(other._proj)
        self._s = other._s.copy()
        self._e = other._e.copy()
        self._npix = other._npix.copy()

    def _set_data_arrays_raw(self, s: Any, e: Any, npix: Any) -> None:
        for name, val, non_neg in (("s", s, False), ("e", e, True), ("npix", npix, True)):
            if not _is_numeric(val):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Field {name} must be a numeric array",
                )
            arr = np.asarray(val, dtype=float)
            if non_neg and np.any(arr < 0):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Field {name} must not contain negative values",
                )
            setattr(self, f"_{name}", arr.copy())

    # ------------------------------------------------------------------
    # Static factories
    # ------------------------------------------------------------------
    @staticmethod
    def dnd(*args: Any, **kwargs: Any) -> "DnDBase":
        """Construct the appropriate concrete DnD subclass (d0d..d4d) based on dimensions."""
        from pyhorace.sqw.d0d import d0d
        from pyhorace.sqw.d1d import d1d
        from pyhorace.sqw.d2d import d2d
        from pyhorace.sqw.d3d import d3d
        from pyhorace.sqw.d4d import d4d

        dim_map = {0: d0d, 1: d1d, 2: d2d, 3: d3d, 4: d4d}

        if not args and not kwargs:
            return d0d()

        # Check spinw constructor
        if len(args) >= 1 and _is_numeric(args[0]) and np.asarray(args[0]).size == 6:
            return DnDBase.spinw_dnd_obj_constructor(*args)

        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, DnDBase):
                cls_t = dim_map[int(arg.dimensions)]
                return cls_t(arg)
            if isinstance(arg, (str, os.PathLike)):
                from pyhorace.sqw.read_dnd import read_dnd

                return read_dnd(str(arg))
            if hasattr(arg, "data") and isinstance(getattr(arg, "data"), (DnDBase, data_sqw_dnd)):
                d_in = getattr(arg, "data")
                return DnDBase.dnd(d_in)
            if isinstance(arg, data_sqw_dnd):
                cls_t = dim_map[int(arg.dimensions)]
                return cls_t(arg)
            if isinstance(arg, dict):
                dsd = data_sqw_dnd(arg)
                cls_t = dim_map[int(dsd.dimensions)]
                return cls_t(dsd)

        # Find axes argument to determine dimensions
        ax_arg = kwargs.get("axes", args[0] if args else None)
        if isinstance(ax_arg, dnd_metadata):
            ax_arg = ax_arg.axes
        if isinstance(ax_arg, AxesBlockBase):
            cls_t = dim_map.get(int(ax_arg.dimensions), d0d)
            return cls_t(*args, **kwargs)

        return d0d(*args, **kwargs)

    @staticmethod
    def spinw_dnd_obj_constructor(lattice_const: Any, *varargin: Any) -> "DnDBase":
        """Construct DnD object from SpinW/legacy lattice and axis-step pairs."""
        if not _is_numeric(lattice_const) or np.asarray(lattice_const).size != 6:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "First argument (lattice_const) must be a 6-element vector [a,b,c,alpha,beta,gamma]",
            )
        lat = np.asarray(lattice_const, dtype=float).ravel()
        if len(varargin) == 0:
            proj = line_proj(alatt=lat[:3], angdeg=lat[3:6])
            ax = line_axes(0)
            return DnDBase.dnd(ax, proj)

        nargi = len(varargin)
        if nargi > 8 or nargi % 2 != 0:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Expected pairs of (direction, binning) arguments (up to 4 pairs), got {nargi} extra arguments",
            )

        uvw: List[Optional[np.ndarray]] = [None, None, None]
        binning: List[Any] = [0, 0, 0, 0]
        uvw_idx = 0
        for i in range(0, nargi, 2):
            if not _is_numeric(varargin[i]) or not _is_numeric(varargin[i + 1]):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    "Direction and binning arguments in SpinW constructor must be numeric vectors",
                )
            dir_vec = np.asarray(varargin[i], dtype=float).ravel()
            bin_vec = np.asarray(varargin[i + 1], dtype=float).ravel()
            if dir_vec.size != 4 or bin_vec.size not in (2, 3):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Expected 4-element direction vector and 2- or 3-element binning vector, got sizes {dir_vec.size} and {bin_vec.size}",
                )
            if dir_vec[3] > 0:
                if np.any(dir_vec[:3] != 0):
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        "Energy axis direction vector cannot have non-zero Q components",
                    )
                binning[3] = bin_vec
            else:
                if uvw_idx >= 3:
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        "At most 3 Q-direction vectors can be specified",
                    )
                uvw[uvw_idx] = dir_vec[:3]
                binning[uvw_idx] = bin_vec
                uvw_idx += 1

        if uvw[0] is None:
            uvw[0] = np.array([1.0, 0.0, 0.0], dtype=float)
        if uvw[1] is None:
            v = np.zeros(3, dtype=float)
            v[[1, 2, 0]] = uvw[0]
            uvw[1] = v

        uvw_args = [x for x in uvw if x is not None]
        proj = line_proj(*uvw_args, alatt=lat[:3], angdeg=lat[3:6])
        ax = line_axes(*binning)
        return DnDBase.dnd(ax, proj)

    # ------------------------------------------------------------------
    # Core properties: axes, proj, s, e, npix
    # ------------------------------------------------------------------
    @property
    def axes(self) -> AxesBlockBase:
        return self._axes

    @axes.setter
    def axes(self, val: Any) -> None:
        if isinstance(val, dict):
            val = Serializable.from_struct(val) if "serial_name" in val else line_axes.get_from_old_data(val)
        if not isinstance(val, AxesBlockBase):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"axes must be an instance of AxesBlockBase, got {type(val).__name__}",
            )
        self._axes = copy.deepcopy(val)
        if not self._do_check_combo_arg:
            # When initializing via axes setter before s/e/npix, resize arrays if needed
            sz = tuple(int(x) for x in self._axes.dims_as_ssize)
            if matlab_size(self._s) != sz:
                self._init_arrays()
        else:
            self.check_combo_arg(False)

    @property
    def proj(self) -> aProjectionBase:
        return self._proj

    @proj.setter
    def proj(self, val: Any) -> None:
        if isinstance(val, dict):
            val = Serializable.from_struct(val) if "serial_name" in val else line_proj.get_from_old_data(val)
        if not isinstance(val, aProjectionBase):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"proj must be an instance of aProjectionBase, got {type(val).__name__}",
            )
        self._proj = copy.deepcopy(val)
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def s(self) -> np.ndarray:
        return self._s

    @s.setter
    def s(self, val: Any) -> None:
        self._set_senpix(val, "s")

    @property
    def e(self) -> np.ndarray:
        return self._e

    @e.setter
    def e(self, val: Any) -> None:
        self._set_senpix(val, "e")

    @property
    def npix(self) -> np.ndarray:
        return self._npix

    @npix.setter
    def npix(self, val: Any) -> None:
        self._set_senpix(val, "npix")

    def _set_senpix(self, val: Any, field: str) -> None:
        if not _is_numeric(val):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Numeric array is expected as value for field '{field}', got {type(val).__name__}",
            )
        arr = np.asarray(val, dtype=float)
        if field in ("e", "npix") and np.any(arr < 0):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Field '{field}' cannot have negative elements",
            )
        # Scalar expansion if target array already exists
        current = getattr(self, f"_{field}")
        if arr.size == 1 and current is not None and current.size > 0 and self._do_check_combo_arg:
            new_arr = np.full(current.shape, float(arr.flat[0]), dtype=float)
            setattr(self, f"_{field}", new_arr)
            return

        setattr(self, f"_{field}", arr.copy())
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    # ------------------------------------------------------------------
    # Dependent / convenience properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> "DnDBase":
        return self

    @data.setter
    def data(self, val: Any) -> None:
        if isinstance(val, DnDBase):
            self._copy_from_dnd(val)
            if self._do_check_combo_arg:
                self.check_combo_arg(False)
        elif isinstance(val, data_sqw_dnd):
            self._axes = copy.deepcopy(val.axes)
            self._proj = copy.deepcopy(val.proj)
            self._set_data_arrays_raw(val.s, val.e, val.npix)
            if self._do_check_combo_arg:
                self.check_combo_arg(False)
        else:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"data must be a DnDBase or data_sqw_dnd instance, got {type(val).__name__}",
            )

    @property
    def metadata(self) -> dnd_metadata:
        return dnd_metadata(self)

    @metadata.setter
    def metadata(self, val: Any) -> None:
        if not isinstance(val, dnd_metadata):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"metadata must be a dnd_metadata object, got {type(val).__name__}",
            )
        self._axes = copy.deepcopy(val.axes)
        self._proj = copy.deepcopy(val.proj)
        if val.creation_date_defined:
            self._axes.creation_date = val.creation_date_str
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def nd_data(self) -> dnd_data:
        return dnd_data(self._s, self._e, self._npix)

    @nd_data.setter
    def nd_data(self, val: Any) -> None:
        if not isinstance(val, dnd_data):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"nd_data must be a dnd_data object, got {type(val).__name__}",
            )
        self._s = val.s.copy()
        self._e = val.e.copy()
        self._npix = val.npix.copy()
        if self._do_check_combo_arg:
            self.check_combo_arg(True)

    @property
    def dimensions(self) -> _DimensionsInt:
        nd = int(self._axes.dimensions)
        if nd == 0:
            sz: Tuple[int, ...] = (1, 1)
        else:
            sz = tuple(int(x) for x in self._axes.data_nbins)
        return _DimensionsInt(nd, sz)

    @property
    def do_check_combo_arg(self) -> bool:
        return bool(self._do_check_combo_arg)

    @do_check_combo_arg.setter
    def do_check_combo_arg(self, val: bool) -> None:
        bval = bool(val)
        self._do_check_combo_arg = bval
        if hasattr(self._axes, "do_check_combo_arg"):
            self._axes.do_check_combo_arg = bval
        if hasattr(self._proj, "do_check_combo_arg"):
            self._proj.do_check_combo_arg = bval

    @property
    def pix(self) -> PixelDataMemory:
        return PixelDataMemory()

    @property
    def title(self) -> str:
        return self._axes.title

    @title.setter
    def title(self, val: str) -> None:
        self._axes.title = val

    @property
    def filename(self) -> str:
        return self._axes.filename

    @filename.setter
    def filename(self, val: str) -> None:
        self._axes.filename = val

    @property
    def filepath(self) -> str:
        return self._axes.filepath

    @filepath.setter
    def filepath(self, val: str) -> None:
        self._axes.filepath = val

    @property
    def full_filename(self) -> str:
        return os.path.join(self._axes.filepath, self._axes.filename)

    @full_filename.setter
    def full_filename(self, val: str) -> None:
        if not isinstance(val, (str, os.PathLike)):
            raise HoraceError("HORACE:DnDBase:invalid_argument", "full_filename must be a string")
        fp, fn = os.path.split(str(val))
        self._axes.filepath = fp
        self._axes.filename = fn

    @property
    def label(self) -> List[str]:
        return self._axes.label

    @label.setter
    def label(self, val: Sequence[str]) -> None:
        self._axes.label = val
        if hasattr(self._proj, "label"):
            self._proj.label = val

    @property
    def iax(self) -> np.ndarray:
        return self._axes.iax

    @property
    def iint(self) -> np.ndarray:
        return self._axes.iint

    @property
    def pax(self) -> np.ndarray:
        return self._axes.pax

    @property
    def p(self) -> List[np.ndarray]:
        return self._axes.p

    @property
    def dax(self) -> np.ndarray:
        return self._axes.dax

    @dax.setter
    def dax(self, val: Any) -> None:
        self._axes.dax = val

    @property
    def nbins(self) -> np.ndarray:
        return self._axes.data_nbins

    @property
    def nbins_all_dims(self) -> np.ndarray:
        return self._axes.nbins_all_dims

    @property
    def img_range(self) -> np.ndarray:
        return self._axes.img_range

    @property
    def line_at_direct_ang(self) -> bool:
        return bool(getattr(self._axes, "line_at_direct_ang", True))

    @property
    def offset(self) -> np.ndarray:
        return self._proj.offset

    @property
    def img_offset(self) -> np.ndarray:
        if hasattr(self._axes, "img_offset"):
            return self._axes.img_offset
        off = self._proj.offset
        q_img = self._proj.transform_hkl_to_img(off[:3].reshape(3, 1)).ravel()
        return np.append(q_img, off[3])

    @property
    def is_filebacked(self) -> bool:
        return False

    @property
    def border_size(self) -> float:
        if hasattr(self._axes, "border_size"):
            return float(self._axes.border_size)
        return -4.0 * float(np.finfo(np.float32).eps)

    @property
    def alatt(self) -> np.ndarray:
        return self._proj.alatt

    @alatt.setter
    def alatt(self, val: Any) -> None:
        self._proj.alatt = val
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def angdeg(self) -> np.ndarray:
        return self._proj.angdeg

    @angdeg.setter
    def angdeg(self, val: Any) -> None:
        self._proj.angdeg = val
        if self._do_check_combo_arg:
            self.check_combo_arg(False)

    @property
    def creation_date(self) -> str:
        return self._axes.creation_date

    @creation_date.setter
    def creation_date(self, val: str) -> None:
        if val:
            self._axes.creation_date = val

    @property
    def CreationDate(self) -> str:
        return self._axes.creation_date

    @CreationDate.setter
    def CreationDate(self, val: str) -> None:
        if val:
            self._axes.creation_date = val

    @property
    def creation_date_defined(self) -> bool:
        return bool(self._axes.creation_date_defined)

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return self.class_version_

    def classVersion(self) -> int:
        return self.class_version_

    def saveable_fields(self) -> List[str]:
        if self.creation_date_defined:
            return list(self._fields_to_save)
        return [f for f in self._fields_to_save if f != "creation_date"]

    def saveableFields(self) -> List[str]:
        return self.saveable_fields()

    def check_combo_arg(self, no_proj_copy: bool = False) -> "DnDBase":
        """Validate interdependent properties axes, proj, s, e, npix."""
        if self.NUM_DIMS is not None and int(self._axes.dimensions) != self.NUM_DIMS:
            if self.NUM_DIMS == 0:
                if getattr(self, "_in_d0d_init", False):
                    raise HoraceError(
                        "MATLAB:class:mustReturnObject",
                        "Constructor for class 'd0d' must return an object of that class.",
                    )
                res = DnDBase.dnd(self._axes, self._proj, self._s, self._e, self._npix)
                return res
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Axes dimensions ({self._axes.dimensions}) do not match {self._err_cls_name()} dimensions ({self.NUM_DIMS})",
            )

        expected_sz = tuple(int(x) for x in self._axes.dims_as_ssize)
        expected_matlab_sz = matlab_size(np.empty(expected_sz))
        for name, arr in (("s", self._s), ("e", self._e), ("npix", self._npix)):
            actual_sz = matlab_size(arr)
            if actual_sz != expected_matlab_sz and arr.shape != expected_sz:
                if (
                    len(expected_sz) == 2
                    and expected_sz[1] == 1
                    and actual_sz == (1, expected_sz[0])
                ):
                    pass
                else:
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        f"Size of field '{name}' {actual_sz} does not match axes dimensions {expected_sz}",
                    )

        if self._s.shape != expected_sz:
            self._s = self._s.reshape(expected_sz, order="F")
        if self._e.shape != expected_sz:
            self._e = self._e.reshape(expected_sz, order="F")
        if self._npix.shape != expected_sz:
            self._npix = self._npix.reshape(expected_sz, order="F")

        # Validate axes type matches proj.axes_name
        axes_name = getattr(self._proj, "axes_name", None)
        if axes_name is None and hasattr(self._proj, "_get_axes_class"):
            axes_name = self._proj._get_axes_class().__name__
        expected_axes_cls = _axes_cls_for_name(axes_name or "line_axes")
        if not isinstance(self._axes, expected_axes_cls):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Projection {type(self._proj).__name__} requires axes of class "
                f"{axes_name}, but got {type(self._axes).__name__}",
            )

        if not no_proj_copy:
            self._axes = self._proj.copy_proj_defined_properties_to_axes(self._axes)

        return self

    @classmethod
    def from_old_struct(cls, inputs: Dict[str, Any]) -> "DnDBase":
        if "data_" in inputs and isinstance(inputs["data_"], (dict, data_sqw_dnd)):
            d = inputs["data_"]
            dsd = d if isinstance(d, data_sqw_dnd) else data_sqw_dnd(d)
            if cls.NUM_DIMS is not None and int(dsd.dimensions) != cls.NUM_DIMS:
                return DnDBase.dnd(dsd)
            return cls(dsd)
        if "axes" in inputs and "proj" in inputs:
            return cls(**inputs)
        dsd = data_sqw_dnd(inputs)
        if cls.NUM_DIMS is not None and int(dsd.dimensions) != cls.NUM_DIMS:
            return DnDBase.dnd(dsd)
        return cls(dsd)

    # ------------------------------------------------------------------
    # SQWDnDBase & data_op_interface abstract methods
    # ------------------------------------------------------------------
    def get_NUM_DIMS(self) -> int:
        if isinstance(self.NUM_DIMS, int):
            return self.NUM_DIMS
        if self._axes is not None:
            return int(self._axes.dimensions)
        return 0

    def get_img_size(self) -> Tuple[int, ...]:
        return self.sigvar_size()

    def has_pixels(self) -> bool:
        return False

    def sigvar(self) -> sigvar:
        return sigvar(self._s, self._e)

    def sigvar_get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._s.copy(), self._e.copy(), (self._npix != 0)

    def sigvar_set(self, sigvar_obj: Any) -> "DnDBase":
        s_in = np.asarray(sigvar_obj.s, dtype=float)
        e_in = np.asarray(sigvar_obj.e, dtype=float)
        target_sz = matlab_size(self._npix)
        s_sz = matlab_size(s_in)
        if s_sz != target_sz:
            if not (len(target_sz) == 2 and target_sz[1] == 1 and s_sz == (1, target_sz[0]) and s_in.ndim == 1):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"sigvar_set: size of npix array [{_fmt_matlab_sz(target_sz)}] different from size of signal array [{_fmt_matlab_sz(s_sz)}]",
                )
        if e_in.size > 0:
            e_sz = matlab_size(e_in)
            if e_sz != target_sz:
                if not (len(target_sz) == 2 and target_sz[1] == 1 and e_sz == (1, target_sz[0]) and e_in.ndim == 1):
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        f"sigvar_set: size of npix array [{_fmt_matlab_sz(target_sz)}] different from size of error array [{_fmt_matlab_sz(e_sz)}]",
                    )
        s_new = s_in.reshape(self._s.shape, order="F").copy()
        if e_in.size == 0:
            e_new = np.zeros_like(s_new)
        else:
            e_new = e_in.reshape(self._e.shape, order="F").copy()

        nopix = self._npix == 0
        s_new[nopix] = 0.0
        e_new[nopix] = 0.0
        self._s = s_new
        self._e = e_new
        return self

    def sigvar_size(self) -> Tuple[int, ...]:
        return matlab_size(self._s)

    def equal_to_tol_single_(
        self,
        other: "DnDBase",
        tol: Any = (1e-5, 1e-5),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "input_1",
        name_b: str = "input_2",
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        return Serializable.equal_to_tol(
            self,
            other,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            name_a=name_a,
            name_b=name_b,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Coordinate & bin methods
    # ------------------------------------------------------------------
    def get_proj_and_pbin(
        self, nargout: Optional[int] = None
    ) -> Union[aProjectionBase, Tuple[aProjectionBase, List[np.ndarray]]]:
        """Retrieve the projection and binning descriptors of this DnD dataset."""
        if nargout is None:
            nargout = _infer_nargout(default=2)
        pbin = self._axes.get_cut_range("-full_range")
        proj = copy.deepcopy(self._proj)
        if nargout <= 1:
            return proj
        return proj, pbin

    def get_targ_range(self, targ_proj: Any = None, *args: Any) -> Any:
        """Compute the range of this dataset's axes in a target projection."""
        if targ_proj is None:
            targ_proj = self._proj
        return self._axes.get_targ_range(self._proj, targ_proj, *args)

    def calculate_q_bins(
        self, nargout: Optional[int] = None
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]:
        """Calculate qh, qk, ql, en for the centres of the bins of an n-dimensional dataset."""
        if nargout is None:
            nargout = _infer_nargout(default=2)
        pts = self._axes.get_bin_nodes("-bin_centre")
        hkle = self._proj.transform_img_to_hkl(pts)
        q = [
            hkle[0, :].reshape(-1, 1),
            hkle[1, :].reshape(-1, 1),
            hkle[2, :].reshape(-1, 1),
        ]
        if nargout <= 1:
            return q
        if self._axes.dimensions > 0 and 4 in list(self._axes.pax):
            pax_list = list(self._axes.pax)
            en_idx = pax_list.index(4)
            p_en = self._axes.p[en_idx]
            en = 0.5 * (p_en[:-1] + p_en[1:]).reshape(-1, 1)
        else:
            iax_list = list(self._axes.iax)
            en_idx = iax_list.index(4)
            en_iint = self._axes.iint[:, en_idx]
            if np.any(np.isinf(en_iint)):
                en = np.array([[0.0]])
            else:
                en = np.array([[0.5 * (en_iint[0] + en_iint[1])]])
        return q, en

    def calculate_qw_bins(
        self,
        *args: str,
        boundaries: bool = False,
        edges: bool = False,
        nargout: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Calculate qh, qk, ql, en for the centres (or boundaries/edges) of bins."""
        ok, mess, flags, _ = _parse_char_options(args, ["-boundaries", "-edges"])
        if not ok:
            raise HoraceError("HORACE:DnDBase:invalid_argument", mess)
        boundaries = boundaries or flags[0]
        edges = edges or flags[1]
        if boundaries and edges:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Cannot specify both '-boundaries' and '-edges'",
            )
        if boundaries:
            pts = self._axes.get_bin_nodes("-boundaries")
        elif edges:
            pts = self._axes.get_bin_nodes("-edges")
        else:
            pts = self._axes.get_bin_nodes("-bin_centre")
        if isinstance(pts, tuple):
            pts = pts[0]
        hkle = self._proj.transform_img_to_hkl(pts)
        return [
            hkle[0, :].reshape(-1, 1),
            hkle[1, :].reshape(-1, 1),
            hkle[2, :].reshape(-1, 1),
            hkle[3, :].reshape(-1, 1),
        ]

    def data_bin_limits(
        self, nargout: Optional[int] = None
    ) -> Union[Optional[np.ndarray], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """Get limits of the data in an n-dimensional dataset along the plot axes
        within which there are non-zero values of npix.
        """
        if nargout is None:
            nargout = _infer_nargout(default=2)
        ndim = self.dimensions
        if ndim == 0:
            return (None, None) if nargout >= 2 else None
        npix_arr = self._npix
        if np.all(npix_arr == 0):
            return (None, None) if nargout >= 2 else None

        val = np.zeros((ndim, 2), dtype=float)
        n = np.zeros((ndim, 2), dtype=int)
        for i in range(ndim):
            if ndim == 1:
                any_along = npix_arr.ravel() != 0
            else:
                other_axes = tuple(ax for ax in range(npix_arr.ndim) if ax != i)
                any_along = np.any(npix_arr != 0, axis=other_axes)
            nz_indices = np.nonzero(any_along)[0]
            if nz_indices.size == 0:
                return (None, None) if nargout >= 2 else None
            ilo = int(nz_indices[0])
            ihi = int(nz_indices[-1])
            n[i, 0] = ilo + 1  # 1-based index
            n[i, 1] = ihi + 1
            p_i = self._axes.p[i]
            val[i, 0] = float(p_i[ilo])
            val[i, 1] = float(p_i[ihi + 1])
        if nargout <= 1:
            return val
        return val, n

    def compact(self) -> "DnDBase":
        """Squeeze the data range in a DnD object to eliminate empty outer bins."""
        wout = self.copy()
        if self.dimensions == 0:
            return wout
        val, n = self.data_bin_limits(nargout=2)
        if n is None:
            return wout

        ndim = self.dimensions
        slices = []
        pbin_all: List[Any] = [None] * 4
        for idx, iax_num in enumerate(self._axes.iax):
            iint_col = self._axes.iint[:, idx]
            pbin_all[int(iax_num) - 1] = 0 if np.any(np.isinf(iint_col)) else iint_col.tolist()

        for i in range(ndim):
            ilo = int(n[i, 0]) - 1
            ihi = int(n[i, 1])
            slices.append(slice(ilo, ihi))
            pax_num = int(self._axes.pax[i])
            p_i = self._axes.p[i]
            step = float(p_i[1] - p_i[0]) if len(p_i) > 1 else 0.0
            centres = 0.5 * (p_i[ilo:ihi] + p_i[ilo + 1 : ihi + 1])
            pbin_all[pax_num - 1] = [float(centres[0]), step, float(centres[-1])]

        ax_cls = type(self._axes)
        new_axes = ax_cls(*pbin_all)
        for idx, iax_num in enumerate(self._axes.iax):
            new_axes._img_range[:, int(iax_num) - 1] = self._axes.iint[:, idx]
        new_axes = self._proj.copy_proj_defined_properties_to_axes(new_axes)
        new_axes.title = self._axes.title
        new_axes.filename = self._axes.filename
        new_axes.filepath = self._axes.filepath
        new_axes.label = self._axes.label
        new_axes.dax = self._axes.dax

        if ndim == 1 and self._s.ndim == 2 and self._s.shape[1] == 1:
            idx_tuple: Any = (slices[0], slice(None))
        elif ndim == 1 and self._s.ndim == 1:
            idx_tuple = (slices[0],)
        else:
            idx_tuple = tuple(slices)

        s_new = self._s[idx_tuple].copy()
        e_new = self._e[idx_tuple].copy()
        npix_new = self._npix[idx_tuple].copy()

        wout._do_check_combo_arg = False
        wout._axes = new_axes
        wout._s = s_new
        wout._e = e_new
        wout._npix = npix_new
        wout._do_check_combo_arg = True
        wout.check_combo_arg(False)
        return wout

    def section(self, *args: Any) -> "DnDBase":
        """Take a section out of an n-dimensional dataset along its plot axes."""
        ndim = self.dimensions
        if ndim == 0:
            raise HoraceError("HORACE:DnDBase:invalid_argument", "Cannot section a zero-dimensional object")
        if len(args) != ndim:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Number of section arguments ({len(args)}) must equal number of dimensions ({ndim})",
            )

        pbin_all: List[Any] = [None] * 4
        for idx, iax_num in enumerate(self._axes.iax):
            iint_col = self._axes.iint[:, idx]
            pbin_all[int(iax_num) - 1] = 0 if np.any(np.isinf(iint_col)) else iint_col.tolist()

        slices = []
        tol = 1e-5
        for i in range(ndim):
            arg = args[i]
            p_i = self._axes.p[i]
            centres = 0.5 * (p_i[:-1] + p_i[1:])
            step = float(p_i[1] - p_i[0])
            if arg is None or (isinstance(arg, (list, tuple, np.ndarray)) and len(arg) == 0) or (
                _is_numeric(arg) and np.asarray(arg).size == 1 and float(np.asarray(arg).flat[0]) == 0.0
            ):
                ilo = 0
                ihi = len(centres)
            else:
                arr = np.asarray(arg, dtype=float).ravel()
                if arr.size != 2:
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        f"Axis {i + 1} section limits must be a 2-element vector [lo, hi] or empty/0",
                    )
                lo, hi = min(float(arr[0]), float(arr[1])), max(float(arr[0]), float(arr[1]))
                in_range = np.nonzero((centres >= lo - tol * abs(step)) & (centres <= hi + tol * abs(step)))[0]
                if in_range.size == 0:
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        f"No bin centres lie within specified section range [{lo}, {hi}] on axis {i + 1}",
                    )
                ilo = int(in_range[0])
                ihi = int(in_range[-1]) + 1

            slices.append(slice(ilo, ihi))
            pax_num = int(self._axes.pax[i])
            pbin_all[pax_num - 1] = [float(centres[ilo]), step, float(centres[ihi - 1])]

        ax_cls = type(self._axes)
        new_axes = ax_cls(*pbin_all)
        for idx, iax_num in enumerate(self._axes.iax):
            new_axes._img_range[:, int(iax_num) - 1] = self._axes.iint[:, idx]
        new_axes = self._proj.copy_proj_defined_properties_to_axes(new_axes)
        new_axes.title = self._axes.title
        new_axes.filename = self._axes.filename
        new_axes.filepath = self._axes.filepath
        new_axes.label = self._axes.label
        new_axes.dax = self._axes.dax

        if ndim == 1 and self._s.ndim == 2 and self._s.shape[1] == 1:
            idx_tuple: Any = (slices[0], slice(None))
        elif ndim == 1 and self._s.ndim == 1:
            idx_tuple = (slices[0],)
        else:
            idx_tuple = tuple(slices)

        wout = self.copy()
        wout._do_check_combo_arg = False
        wout._axes = new_axes
        wout._s = self._s[idx_tuple].copy()
        wout._e = self._e[idx_tuple].copy()
        wout._npix = self._npix[idx_tuple].copy()
        wout._do_check_combo_arg = True
        wout.check_combo_arg(False)
        return wout

    def mask(self, mask_array: Any = None) -> "DnDBase":
        """Remove the bins indicated by the mask array (0/False = remove, 1/True = keep)."""
        wout = self.copy()
        if mask_array is None or (isinstance(mask_array, (list, tuple, np.ndarray)) and np.asarray(mask_array).size == 0):
            return wout
        msk = np.asarray(mask_array)
        if msk.size != self._s.size:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Mask array must have the same number of elements ({msk.size}) as the data ({self._s.size})",
            )
        keep = msk.astype(bool).reshape(self._s.shape, order="F")
        wout._s = self._s.copy()
        wout._e = self._e.copy()
        wout._npix = self._npix.copy()
        wout._s[~keep] = 0.0
        wout._e[~keep] = 0.0
        wout._npix[~keep] = 0.0
        return wout

    def value(self, x: Any = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get the signal and standard deviation (sqrt(variance)) for points along the plot axes."""
        ndim = self.dimensions
        if ndim == 0:
            if x is not None and np.asarray(x).size > 0:
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    "Cannot specify coordinates for a 0D object",
                )
            y = np.array([float(self._s.flat[0])])
            sig = np.array([float(np.sqrt(self._e.flat[0]))])
            return y, sig

        x_arr = np.asarray(x, dtype=float)
        if ndim == 1:
            pts = x_arr.reshape(-1, 1)
        else:
            if x_arr.ndim == 1 and x_arr.size == ndim:
                pts = x_arr.reshape(1, ndim)
            elif x_arr.ndim == 2 and x_arr.shape[1] == ndim:
                pts = x_arr
            else:
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Coordinates x must be an m x {ndim} array, got shape {x_arr.shape}",
                )

        npts = pts.shape[0]
        y = np.full(npts, np.nan, dtype=float)
        sig = np.full(npts, np.nan, dtype=float)

        # Note: x is given in DISPLAY axes order (dax)!
        dax_0 = [int(d) - 1 for d in self._axes.dax]
        bin_indices = np.zeros((npts, ndim), dtype=int)
        valid = np.ones(npts, dtype=bool)

        for idisp in range(ndim):
            ipax = dax_0[idisp]
            edges = self._axes.p[ipax]
            coords = pts[:, idisp]
            # Find bin index: edges[k] <= coord <= edges[k+1]
            idx = np.searchsorted(edges, coords, side="right") - 1
            # Handle exact right boundary of the last bin
            at_right = coords == edges[-1]
            idx[at_right] = len(edges) - 2
            in_bounds = (coords >= edges[0]) & (coords <= edges[-1]) & (idx >= 0) & (idx < len(edges) - 1)
            valid &= in_bounds
            bin_indices[:, ipax] = np.clip(idx, 0, len(edges) - 2)

        s_arr = self._s.reshape([ len(p) - 1 for p in self._axes.p ], order="F") if ndim > 0 else self._s
        e_arr = self._e.reshape(s_arr.shape, order="F")
        npix_arr = self._npix.reshape(s_arr.shape, order="F")

        for k in range(npts):
            if valid[k]:
                b_idx = tuple(int(bin_indices[k, d]) for d in range(ndim))
                if npix_arr[b_idx] > 0:
                    y[k] = float(s_arr[b_idx])
                    sig[k] = float(np.sqrt(e_arr[b_idx]))
        return y, sig

    def xye(self, null_value: Any = np.nan) -> Dict[str, Any]:
        """Return the bin centres, intensity (y) and error bar (e = sqrt(variance)) of a DnD dataset."""
        if null_value is None:
            null_val = np.nan
        else:
            if not _is_numeric(null_value) or np.asarray(null_value).size != 1:
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    "Null value must be a numeric scalar",
                )
            null_val = float(np.asarray(null_value).flat[0])

        ndim = self.dimensions
        if ndim == 0:
            x: Any = []
        elif ndim == 1:
            p0 = self._axes.p[0]
            x = 0.5 * (p0[:-1] + p0[1:])
        else:
            x = [0.5 * (pi[:-1] + pi[1:]) for pi in self._axes.p]

        y = self._s.copy().astype(float)
        e = np.sqrt(self._e.copy().astype(float))
        empty = self._npix == 0
        y[empty] = null_val
        e[empty] = 0.0
        return {"x": x, "y": y, "e": e}

    def save_xye(self, file: Optional[str] = None, null_value: Any = None) -> None:
        """Save 1D, 2D, 3D, or 4D data to ASCII file in x-y-e format."""
        if self.dimensions == 0:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Cannot save a zero-dimensional dataset to an xye file",
            )
        if not file or not isinstance(file, (str, os.PathLike)):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Output filename must be a non-empty character string",
            )
        if null_value is None:
            null_val = -1e30
        else:
            if not _is_numeric(null_value) or np.asarray(null_value).size != 1:
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    "Null value must be a numeric scalar",
                )
            null_val = float(np.asarray(null_value).flat[0])

        data = self.xye(null_val)
        ndim = self.dimensions
        if ndim == 1:
            x_col = np.asarray(data["x"]).reshape(-1, 1)
            y_col = np.asarray(data["y"]).reshape(-1, 1)
            e_col = np.asarray(data["e"]).reshape(-1, 1)
            table = np.hstack([x_col, y_col, e_col])
        else:
            grids = np.meshgrid(*data["x"], indexing="ij")
            cols = [g.ravel(order="F").reshape(-1, 1) for g in grids]
            cols.append(np.asarray(data["y"]).ravel(order="F").reshape(-1, 1))
            cols.append(np.asarray(data["e"]).ravel(order="F").reshape(-1, 1))
            table = np.hstack(cols)

        np.savetxt(str(file), table, fmt="%-.16g", delimiter="\t")

    @staticmethod
    def head_form(keep_data_arrays: bool = False) -> Tuple[List[str], List[str]]:
        """Return list of field names kept by head()."""
        h_fields = [
            "alatt",
            "angdeg",
            "offset",
            "u_to_rlu_legacy",
            "ulen",
            "label",
            "iax",
            "iint",
            "pax",
            "p",
            "dax",
            "img_range",
            "dimensions",
            "creation_date",
        ]
        if keep_data_arrays:
            d_fields = ["filename", "filepath", "title", "s", "e", "npix"]
        else:
            d_fields = ["filename", "filepath", "title"]
        return h_fields, d_fields

    def head(self, *args: str, nargout: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Display or return a summary of a DnD object or file."""
        ok, mess, flags, rem = _parse_char_options(args, ["-full"])
        if not ok:
            raise HoraceError("HORACE:DnDBase:invalid_argument", mess)
        if rem:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Unrecognized argument(s) to head: {rem}",
            )
        full_flag = flags[0]
        if nargout is None:
            nargout = _infer_nargout(default=1)

        h_fields, d_fields = self.head_form(full_flag)
        all_fields = d_fields + h_fields
        out: Dict[str, Any] = {}
        for f in all_fields:
            val = getattr(self, f)
            out[f] = copy.deepcopy(val)
        if nargout == 0:
            return None
        return out

    # ------------------------------------------------------------------
    # Cut, rebin, replicate, smooth
    # ------------------------------------------------------------------
    def cut_dnd(self, *args: Any, **kwargs: Any) -> Any:
        """Take a cut from a DnD object (port of ``@DnDBase/cut_dnd.m``)."""
        return self.cut(*args, **kwargs)

    def cut_sym(self, *args: Any, **kwargs: Any) -> Any:
        """Take a cut from a DnD object (port of ``@DnDBase/cut_sym.m``)."""
        return self.cut(*args, **kwargs)

    def cut_sqw(self, *args: Any, **kwargs: Any) -> Any:
        """Raise error when ``cut_sqw`` is called on a DnD object (port of ``@DnDBase/cut_sqw.m``)."""
        raise HoraceError(
            "HORACE:cut_sqw:invalid_argument",
            f"Cannot call cut_sqw on a {type(self).__name__} (dnd) object",
        )

    def cut(self, *args: Any, **kwargs: Any) -> Any:
        """Take a cut from a DnD object by integrating/rebinning over one or more axes.

        Ported from ``@DnDBase/cut.m`` and ``@DnDBase/private/cut_single_.m``.
        """
        nargout = kwargs.pop("nargout", None)
        if nargout is None:
            nargout = _infer_nargout(default=1)
        return_cut = bool(nargout > 0)

        targ_proj, pbin_list, sym, opt = self.process_and_validate_cut_inputs(
            return_cut, *args, nargout=4
        )

        results = []
        for pbin in pbin_list:
            targ_axes, cur_proj = self.define_target_axes_block(
                targ_proj, pbin, sym, nargout=2
            )
            proj_single = (
                cur_proj[0]
                if isinstance(cur_proj, (list, tuple, np.ndarray))
                else cur_proj
            )
            if not opt["proj_given"] and not isinstance(cur_proj, (list, tuple, np.ndarray)):
                targ_axes = self._axes.realign_bin_edges(targ_axes)
                s_out, e_out, npix_out = self._axes.rebin_data(
                    (self._s, self._e, self._npix), targ_axes
                )
            else:
                s_out, e_out, npix_out = self._interpolate_cut_data(
                    targ_axes, proj_single
                )
            wout = DnDBase.dnd(targ_axes, proj_single, s_out, e_out, npix_out)
            wout.title = self.title
            if opt.get("outfile"):
                wout.save(opt["outfile"])
            if return_cut:
                results.append(wout)

        if not return_cut:
            return None
        return results[0] if len(results) == 1 else DataOpArray(results)

    def _interpolate_cut_data(
        self, targ_axes: AxesBlockBase, targ_proj: aProjectionBase
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate signal and variance from self onto targ_axes with targ_proj.
        Ported from @AxesBlockBase/private/calculate_density_.m and interpolate_data_.m.
        """
        from scipy.interpolate import RegularGridInterpolator
        from ..coord_transf.line_axes import line_axes
        from ..coord_transf.line_proj import line_proj

        ref_axes = self._axes
        ref_proj = self._proj
        if not targ_proj.alatt_defined and ref_proj.alatt_defined:
            targ_proj.alatt = ref_proj.alatt
        if not targ_proj.angdeg_defined and ref_proj.angdeg_defined:
            targ_proj.angdeg = ref_proj.angdeg
        ref_proj.targ_proj = targ_proj
        targ_proj.targ_proj = ref_proj

        # 1. Calculate source signal density at source bin edges (calculate_density_.m)
        is_pax = np.zeros(4, dtype=bool)
        if ref_axes.pax.size > 0:
            is_pax[ref_axes.pax - 1] = True

        ref_nbins = ref_axes.nbins_all_dims.astype(int)
        ref_range = ref_axes.img_range
        dens_axes = [
            np.linspace(ref_range[0, i], ref_range[1, i], int(ref_nbins[i]) + 1)
            for i in range(4)
        ]

        if isinstance(ref_axes, line_axes):
            base_cell_volume: Any = float(
                np.prod((ref_range[1, :] - ref_range[0, :]) / ref_nbins)
                * ref_axes.get_volume_scale()
            )
        else:
            edges_3d_ref = [
                dens_axes[i] if is_pax[i] else np.array([ref_range[0, i], ref_range[1, i]])
                for i in range(3)
            ]
            de_edges_ref = (
                dens_axes[3] if is_pax[3] else np.array([ref_range[0, 3], ref_range[1, 3]])
            )
            mg_ref = np.meshgrid(*edges_3d_ref, indexing="ij")
            coords_ref = np.stack(mg_ref, axis=0)
            v1_r = coords_ref[:, 1:, :-1, :-1] - coords_ref[:, :-1, :-1, :-1]
            v2_r = coords_ref[:, :-1, 1:, :-1] - coords_ref[:, :-1, :-1, :-1]
            v3_r = coords_ref[:, :-1, :-1, 1:] - coords_ref[:, :-1, :-1, :-1]
            vol_3d_r = (
                np.abs(np.sum(np.cross(v1_r, v2_r, axis=0) * v3_r, axis=0))
                * ref_axes.get_volume_scale()
            )
            de_diff_r = np.diff(de_edges_ref)
            base_cell_volume = (
                vol_3d_r.ravel(order="F")[:, None] * de_diff_r[None, :]
            ).reshape(self._s.shape, order="F")

        base_shape_1 = [int(ref_nbins[i]) if is_pax[i] else 1 for i in range(4)]
        ref_ds = (self._s / base_cell_volume).reshape(base_shape_1, order="F")
        tile_reps = [1 if is_pax[i] else 2 for i in range(4)]
        ref_ds_tiled = np.tile(ref_ds, tile_reps)
        density = ref_ds_tiled
        nearest_density = ref_ds_tiled

        # Interpolate from bin centers (length N) to bin edges (length N+1) along pax
        # with 'linear' interior interpolation and MATLAB 'nearest' boundary extrapolation
        # (matching interpn + griddedInterpolant(..., 'linear', 'nearest') in calculate_density_.m)
        for i in range(4):
            if is_pax[i]:
                n_i = density.shape[i]
                sl_first = [slice(None)] * 4
                sl_last = [slice(None)] * 4
                sl_left = [slice(None)] * 4
                sl_right = [slice(None)] * 4
                sl_first[i] = slice(0, 1)
                sl_last[i] = slice(-1, None)
                sl_left[i] = slice(0, -1)
                sl_right[i] = slice(1, None)
                if n_i == 1:
                    density = np.concatenate(
                        [density[tuple(sl_first)], density[tuple(sl_last)]], axis=i
                    )
                else:
                    mid = 0.5 * (density[tuple(sl_left)] + density[tuple(sl_right)])
                    density = np.concatenate(
                        [density[tuple(sl_first)], mid, density[tuple(sl_last)]],
                        axis=i,
                    )
                nearest_idx = np.minimum(np.arange(n_i + 1), n_i - 1)
                nearest_density = np.take(nearest_density, nearest_idx, axis=i)

        for i in range(4):
            if is_pax[i]:
                sl_b0 = [slice(None)] * 4
                sl_b1 = [slice(None)] * 4
                sl_b0[i] = 0
                sl_b1[i] = -1
                density[tuple(sl_b0)] = nearest_density[tuple(sl_b0)]
                density[tuple(sl_b1)] = nearest_density[tuple(sl_b1)]

        # 2. Determine target grid refinement (interpolate_data_.m lines 65-91)
        hull_3d = ref_axes.get_bin_nodes("-3D", "-hull")
        edges_pix = ref_proj.transform_img_to_pix(hull_3d)
        range_3d = np.vstack([np.min(edges_pix, axis=1), np.max(edges_pix, axis=1)])
        range_4d = np.hstack([range_3d, ref_range[:, 3:4]])
        char_size = (range_4d[1, :] - range_4d[0, :]) / ref_nbins

        range_cc = targ_axes.get_targ_range(targ_proj, line_proj(type="aaa"))
        # Restore cross-linked projections (mutated by get_targ_range in Python)
        ref_proj.targ_proj = targ_proj
        targ_proj.targ_proj = ref_proj

        targ_axes_span = range_cc[1, :] - range_cc[0, :]
        nbins_all_dims = targ_axes.nbins_all_dims.astype(int).copy()
        targ_step = targ_axes_span / nbins_all_dims
        too_coarse = targ_step > (0.5 * char_size)
        while np.any(too_coarse):
            nbins_all_dims[too_coarse] *= 2
            targ_step = targ_axes_span / nbins_all_dims
            too_coarse = targ_step > (0.5 * char_size)

        # 3. Compute target cell volumes in reference coordinate system (lines 95-106)
        targ_range = targ_axes.img_range
        edges_3d_1d = [
            np.linspace(targ_range[0, i], targ_range[1, i], int(nbins_all_dims[i]) + 1)
            for i in range(3)
        ]
        de_edges = np.linspace(
            targ_range[0, 3], targ_range[1, 3], int(nbins_all_dims[3]) + 1
        )
        mg_edges = np.meshgrid(*edges_3d_1d, indexing="ij")
        nodes_3d_edges = np.vstack([g.ravel(order="F") for g in mg_edges])
        inodes_edges = targ_proj.from_this_to_targ_coord(nodes_3d_edges)

        npts_e = [int(nbins_all_dims[i]) + 1 for i in range(3)]
        coords_e = inodes_edges.reshape((3, npts_e[0], npts_e[1], npts_e[2]), order="F")
        v1 = coords_e[:, 1:, :-1, :-1] - coords_e[:, :-1, :-1, :-1]
        v2 = coords_e[:, :-1, 1:, :-1] - coords_e[:, :-1, :-1, :-1]
        v3 = coords_e[:, :-1, :-1, 1:] - coords_e[:, :-1, :-1, :-1]
        vol_3d = (
            np.abs(np.sum(np.cross(v1, v2, axis=0) * v3, axis=0))
            * ref_axes.get_volume_scale()
        )
        de_diff = np.diff(de_edges)
        targ_cell_volume = (
            vol_3d.ravel(order="F")[:, None] * de_diff[None, :]
        ).ravel(order="F")

        # 4. Evaluate interpolated density at target bin centers and bin into targ_axes
        centers_3d_1d = [0.5 * (e[:-1] + e[1:]) for e in edges_3d_1d]
        de_centers = 0.5 * (de_edges[:-1] + de_edges[1:])
        mg_centers = np.meshgrid(*centers_3d_1d, indexing="ij")
        nodes_3d_centers = np.vstack([g.ravel(order="F") for g in mg_centers])
        inodes_centers = targ_proj.from_this_to_targ_coord(nodes_3d_centers)

        n_3d = nodes_3d_centers.shape[1]
        n_de = de_centers.size
        de_row = np.repeat(de_centers, n_3d).reshape(1, -1)
        inodes_4d = np.vstack([np.tile(inodes_centers, (1, n_de)), de_row])
        nodes_4d = np.vstack([np.tile(nodes_3d_centers, (1, n_de)), de_row])

        rgi = RegularGridInterpolator(
            tuple(dens_axes),
            density,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        interp_ds = rgi(inodes_4d.T)
        signal = interp_ds.ravel(order="F") * targ_cell_volume

        targ_sz = tuple(int(x) for x in targ_axes.dims_as_ssize)
        if float(np.sum(signal)) == 0.0:
            return (
                np.zeros(targ_sz, dtype=float),
                np.zeros(targ_sz, dtype=float),
                np.ones(targ_sz, dtype=float),
            )

        _, s_list, _ = targ_axes.bin_pixels(
            nodes_4d, None, None, None, [signal], nargout=3
        )
        s_out = s_list[0] if isinstance(s_list, list) else s_list
        e_out = np.full(targ_sz, np.inf, dtype=float)
        npix_out = np.ones(targ_sz, dtype=float)
        return s_out, e_out, npix_out

    def define_target_axes_block(
        self,
        targ_proj: Any,
        pbin: Sequence[Any],
        sym: Any = None,
        nargout: Optional[int] = None,
    ) -> Any:
        """Build the target AxesBlockBase and symmetry-transformed projections for a cut operation.

        Ported from ``@DnDBase/define_target_axes_block.m``.
        """
        from ..coord_transf.line_proj_base import LineProjBase
        from ..coord_transf.symop import SymopIdentity

        if nargout is None:
            nargout = _infer_nargout(default=1)

        if sym is None:
            sym_in = [SymopIdentity()]
        elif isinstance(sym, (list, tuple)):
            sym_in = list(sym)
        else:
            sym_in = [sym]

        base_proj = (
            targ_proj[0]
            if isinstance(targ_proj, (list, tuple, np.ndarray))
            else targ_proj
        )

        if isinstance(base_proj, LineProjBase):
            targ_proj_list: List[aProjectionBase] = []
            sym_out: List[Any] = []
            for s in sym_in:
                tp, s_tr = s.transform_proj(base_proj, nargout=2)
                targ_proj_list.append(tp)
                sym_out.append(s_tr)
        else:
            targ_proj_list = [copy.deepcopy(base_proj)]
            sym_out = list(sym_in)

        default_needed = []
        for pb in pbin:
            if pb is None:
                default_needed.append(True)
            else:
                arr = np.asarray(pb, dtype=float).ravel()
                need = (
                    arr.size < 2
                    or np.isinf(arr[0])
                    or np.isinf(arr[-1])
                    or (arr.size == 3 and arr[1] == 0)
                )
                default_needed.append(bool(need))

        first_proj = targ_proj_list[0]
        if any(default_needed):
            source_binning = self.get_targ_range(
                first_proj, default_needed, "-binning"
            )
        else:
            source_binning = [
                np.array([]),
                np.array([]),
                np.array([]),
                np.array([]),
            ]

        targ_ax_block = first_proj.get_proj_axes_block(source_binning, list(pbin))
        targ_ax_block.filename = self.filename
        targ_ax_block.filepath = self.filepath

        targ_proj_out: Any = (
            targ_proj_list[0] if len(targ_proj_list) == 1 else targ_proj_list
        )
        if nargout <= 1:
            return targ_ax_block
        if nargout == 2:
            return targ_ax_block, targ_proj_out
        return targ_ax_block, targ_proj_out, sym_out

    def rebin(self, *args: Any) -> "DnDBase":
        """Rebin a DnD dataset along its plot axes.

        Usage:
            >> wout = w.rebin(wref)
            >> wout = w.rebin(targ_axes)
            >> wout = w.rebin(pbin1, pbin2, ...)
        """
        ok, mess, flags, rem_args = _parse_char_options(args, ["-keep_contents"])
        if not ok:
            raise HoraceError("HORACE:DnDBase:invalid_argument", mess)
        keep_contents = flags[0]
        args = tuple(rem_args)

        ndim = int(self.dimensions)
        if ndim == 0:
            if len(args) >= 1 and isinstance(args[0], DnDBase):
                other_obj = args[0]
                if len(args) > 1:
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        f"Unrecognised option(s) to 0D rebin: {args[1:]}",
                    )
                wout = self.copy()
                ax = copy.deepcopy(other_obj._axes)
                ax._do_check_combo_arg = True
                ax._nbins_all_dims = np.ones(4, dtype=int)
                ax._single_bin_defines_iax = np.ones(4, dtype=bool)
                ax._dax = np.array([], dtype=int)
                ax._do_check_combo_arg = False
                ax.check_combo_arg()
                wout._do_check_combo_arg = False
                wout._axes = ax
                wout._proj = copy.deepcopy(other_obj._proj)
                if keep_contents:
                    npix_sum = float(np.sum(self._npix)) + float(np.sum(other_obj._npix))
                    s_sum = float(np.sum(self._s * self._npix)) + float(np.sum(other_obj._s * other_obj._npix))
                    e_sum = float(np.sum(self._e * (self._npix**2))) + float(np.sum(other_obj._e * (other_obj._npix**2)))
                else:
                    npix_sum = float(np.sum(other_obj._npix))
                    s_sum = float(np.sum(other_obj._s * other_obj._npix))
                    e_sum = float(np.sum(other_obj._e * (other_obj._npix**2)))
                wout._npix = np.array([[npix_sum]], dtype=float)
                if npix_sum > 0:
                    wout._s = np.array([[s_sum / npix_sum]], dtype=float)
                    wout._e = np.array([[e_sum / (npix_sum**2)]], dtype=float)
                else:
                    wout._s = np.zeros((1, 1), dtype=float)
                    wout._e = np.zeros((1, 1), dtype=float)
                wout._do_check_combo_arg = True
                wout.check_combo_arg(False)
                return wout
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Cannot rebin a zero-dimensional (d0d) dataset without a source DnDBase object",
            )
        if len(args) == 0:
            from pyhorace.sqw.d0d import d0d

            return d0d().rebin(self)
        if len(args) == 1 and isinstance(args[0], (DnDBase, AxesBlockBase)):
            other = args[0]
            if isinstance(other, DnDBase):
                if not isinstance(other._proj, type(self._proj)):
                    raise HoraceError(
                        "HORACE:DnDBase:invalid_argument",
                        "Template object must have the same projection class",
                    )
                targ_axes = copy.deepcopy(other._axes)
                targ_proj = copy.deepcopy(other._proj)
            else:
                targ_axes = copy.deepcopy(other)
                targ_proj = copy.deepcopy(self._proj)
        elif 1 <= len(args) <= ndim:
            # Binning description for plot axes
            pbin_all: List[Any] = [None] * 4
            for idx, iax_num in enumerate(self._axes.iax):
                iint_col = self._axes.iint[:, idx]
                pbin_all[int(iax_num) - 1] = [] if np.any(np.isinf(iint_col)) else iint_col.tolist()
            for idx, pax_num in enumerate(self._axes.pax):
                if idx < len(args):
                    pbin_all[int(pax_num) - 1] = args[idx]
                else:
                    pbin_all[int(pax_num) - 1] = []
            targ_proj = copy.deepcopy(self._proj)
            targ_axes = self.define_target_axes_block(targ_proj, pbin_all)
            for idx, iax_num in enumerate(self._axes.iax):
                targ_axes._img_range[:, int(iax_num) - 1] = self._axes.iint[:, idx]
        else:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Expected 1 template argument or up to {ndim} binning descriptors, got {len(args)}",
            )

        s_out, e_out, npix_out = self._axes.rebin_data(
            (self._s, self._e, self._npix), targ_axes
        )
        wout = DnDBase.dnd(targ_axes, targ_proj, s_out, e_out, npix_out)
        wout.title = self.title
        return wout

    def replicate(self, wref: Any) -> "DnDBase":
        """Make a higher dimensional dataset from a lower dimensional dataset by
        replicating the data along the extra dimensions of a reference dataset `wref`.
        """
        if hasattr(wref, "data") and isinstance(getattr(wref, "data"), DnDBase) and not isinstance(wref, DnDBase):
            wref_dnd = getattr(wref, "data")
        elif isinstance(wref, DnDBase):
            wref_dnd = wref
        else:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Reference object must be a DnDBase or sqw object",
            )

        ndim_in = self.dimensions
        ndim_ref = wref_dnd.dimensions
        if ndim_in > ndim_ref:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                f"Cannot replicate {ndim_in}D dataset into lower-dimensional {ndim_ref}D dataset",
            )

        pax_in = [int(x) for x in self._axes.pax]
        pax_ref = [int(x) for x in wref_dnd._axes.pax]

        # Check all plot axes of self are in wref.pax and have identical bin boundaries
        tol = 4.0 * float(np.finfo(np.float32).eps)
        in_positions = []
        for idx_in, ax_num in enumerate(pax_in):
            if ax_num not in pax_ref:
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Plot axis {ax_num} of input dataset is not a plot axis of reference dataset",
                )
            idx_ref = pax_ref.index(ax_num)
            p_in = self._axes.p[idx_in]
            p_ref = wref_dnd._axes.p[idx_ref]
            if len(p_in) != len(p_ref) or not np.allclose(p_in, p_ref, rtol=tol, atol=tol):
                raise HoraceError(
                    "HORACE:DnDBase:invalid_argument",
                    f"Bin boundaries along axis {ax_num} do not match reference dataset",
                )
            in_positions.append(idx_ref)

        wout = wref_dnd.copy()
        ref_shape = [len(p) - 1 for p in wref_dnd._axes.p]
        if ndim_in == 0:
            s_rep = np.full(wout._s.shape, float(self._s.flat[0]), dtype=float)
            e_rep = np.full(wout._e.shape, float(self._e.flat[0]), dtype=float)
        else:
            # Build broadcast shape of length ndim_ref:
            bcast_shape = [1] * ndim_ref
            for idx_in, pos_ref in enumerate(in_positions):
                bcast_shape[pos_ref] = ref_shape[pos_ref]
            # Permute self._s if needed so its axes match increasing order in in_positions
            in_shape = [len(p) - 1 for p in self._axes.p]
            s_src = self._s.reshape(in_shape, order="F")
            e_src = self._e.reshape(in_shape, order="F")
            order = np.argsort(in_positions)
            if not np.array_equal(order, np.arange(ndim_in)):
                s_src = np.transpose(s_src, order)
                e_src = np.transpose(e_src, order)
            s_rep = np.broadcast_to(s_src.reshape(bcast_shape), ref_shape).reshape(wout._s.shape, order="F").copy()
            e_rep = np.broadcast_to(e_src.reshape(bcast_shape), ref_shape).reshape(wout._e.shape, order="F").copy()

        nopix = wout._npix == 0
        s_rep[nopix] = 0.0
        e_rep[nopix] = 0.0
        wout._s = s_rep
        wout._e = e_rep
        return wout

    def smooth(self, *args: Any) -> "DnDBase":
        """Smooth a DnD dataset.

        Syntax:
            >> wout = w.smooth()
            >> wout = w.smooth(width)
            >> wout = w.smooth(width, shape)
        """
        ndim = self.dimensions
        if ndim == 0:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Cannot smooth a zero-dimensional (d0d) dataset",
            )
        if not self._axes.line_at_direct_ang:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "Smoothing is only supported for datasets with mutually orthogonal axes",
            )

        ok, mess, width, shape_func = self._parse_smooth_args(ndim, *args)
        if not ok:
            raise HoraceError("HORACE:DnDBase:invalid_argument", mess)

        # Build convolution kernel c along each dimension
        kernels = [shape_func(float(w)) for w in width]
        c = kernels[0].reshape((-1,) + (1,) * (ndim - 1))
        for i in range(1, ndim):
            sh = [1] * ndim
            sh[i] = kernels[i].size
            c = c * kernels[i].reshape(sh)
        c_sum = np.sum(c)
        if c_sum != 0:
            c = c / c_sum

        grid_shape = tuple(len(p) - 1 for p in self._axes.p)
        s_arr = self._s.reshape(grid_shape, order="F").copy()
        e_arr = self._e.reshape(grid_shape, order="F").copy()
        npix_arr = self._npix.reshape(grid_shape, order="F")

        valid = (npix_arr > 0) & (~np.isnan(s_arr))
        s_masked = np.where(valid, s_arr, 0.0)
        e_masked = np.where(valid, e_arr, 0.0)
        w_mask = valid.astype(float)

        weight = scipy.ndimage.convolve(w_mask, c, mode="constant", cval=0.0)
        s_conv = scipy.ndimage.convolve(s_masked, c, mode="constant", cval=0.0)
        e_conv = scipy.ndimage.convolve(e_masked, c**2, mode="constant", cval=0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            s_out = np.where(valid & (weight != 0), s_conv / weight, 0.0)
            e_out = np.where(valid & (weight != 0), e_conv / (weight**2), 0.0)

        wout = self.copy()
        wout._s = s_out.reshape(self._s.shape, order="F")
        wout._e = e_out.reshape(self._e.shape, order="F")
        return wout

    @staticmethod
    def _parse_smooth_args(
        ndim: int, *args: Any
    ) -> Tuple[bool, str, np.ndarray, Callable[[float], np.ndarray]]:
        if len(args) > 2:
            return False, "Too many arguments passed to smooth", np.array([]), lambda w: np.array([1.0])

        if len(args) == 0 or args[0] is None or (isinstance(args[0], (list, tuple, np.ndarray)) and len(args[0]) == 0):
            width = np.full(ndim, 3.0, dtype=float)
        else:
            if not _is_numeric(args[0]):
                return False, "Width parameter must be a scalar or a vector with length equal to the dimensions of the dataset", np.array([]), lambda w: np.array([1.0])
            w_in = np.asarray(args[0], dtype=float).ravel()
            if w_in.size == 1:
                width = np.full(ndim, float(w_in[0]), dtype=float)
            elif w_in.size == ndim:
                width = w_in
            else:
                return False, f"Width parameter must be a scalar or a vector with length equal to the dimensions of the dataset ({ndim})", np.array([]), lambda w: np.array([1.0])
            if np.any(width < 0):
                return False, "Width parameter(s) must be non-negative", np.array([]), lambda w: np.array([1.0])

        def _hat_kernel(fwhh: float) -> np.ndarray:
            n = int(np.floor(fwhh / 2.0))
            return np.ones(2 * n + 1, dtype=float) / float(2 * n + 1)

        def _gauss_kernel(fwhh: float) -> np.ndarray:
            if fwhh == 0:
                return np.array([1.0], dtype=float)
            sig = fwhh / np.sqrt(np.log(256.0))
            n = int(np.round(2.0 * fwhh))
            x = np.arange(-n, n + 1, dtype=float)
            g = np.exp(-(x**2) / (2.0 * sig**2))
            return g / np.sum(g)

        if len(args) < 2 or args[1] is None:
            return True, "", width, _hat_kernel

        shape_arg = args[1]
        if callable(shape_arg):
            return True, "", width, shape_arg
        if isinstance(shape_arg, str):
            s_low = shape_arg.strip().lower()
            if not s_low:
                return False, f"'{shape_arg}' is not recognised as a valid smoothing shape", np.array([]), _hat_kernel
            if "hat".startswith(s_low):
                return True, "", width, _hat_kernel
            if "gaussian".startswith(s_low):
                return True, "", width, _gauss_kernel
            if "resolution".startswith(s_low):
                return False, "'resolution' smoothing is not currently implemented", np.array([]), _hat_kernel
        return False, f"'{shape_arg}' is not recognised as a valid smoothing shape", np.array([]), _hat_kernel

    # ------------------------------------------------------------------
    # Function evaluation & dispersion
    # ------------------------------------------------------------------
    def func_eval(self, func_handle: Callable[..., Any], pars: Any, *args: Any) -> "DnDBase":
        """Evaluate a function at the plotting axis bin centres of a DnD object.

        Syntax:
            >> wout = w.func_eval(func_handle, pars)
            >> wout = w.func_eval(func_handle, pars, '-all')
            >> wout = w.func_eval(func_handle, pars, 'outfile', filename)
        """
        if not callable(func_handle):
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "First argument to func_eval must be a callable function handle",
            )
        if self.dimensions == 0:
            raise HoraceError(
                "HORACE:DnDBase:invalid_argument",
                "func_eval cannot be called on a zero-dimensional (d0d) object",
            )

        all_flag = False
        outfile = None
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, str) and a.lower().startswith("-al"):
                all_flag = True
                i += 1
            elif isinstance(a, str) and a.lower() in ("all", "-all"):
                if i + 1 < len(args) and isinstance(args[i + 1], (bool, int)):
                    all_flag = bool(args[i + 1])
                    i += 2
                else:
                    all_flag = True
                    i += 1
            elif isinstance(a, str) and a.lower() in ("outfile", "-outfile"):
                if i + 1 >= len(args):
                    raise HoraceError("HORACE:DnDBase:invalid_argument", "Missing filename after 'outfile'")
                outfile = args[i + 1]
                i += 2
            else:
                raise HoraceError("HORACE:DnDBase:invalid_argument", f"Unrecognised option to func_eval: {a!r}")

        pars_list = list(pars) if isinstance(pars, (list, tuple)) else [pars]

        ndim = self.dimensions
        centres = [0.5 * (pi[:-1] + pi[1:]) for pi in self._axes.p]
        if ndim == 1:
            coord_cols = [centres[0].reshape(-1, 1)]
        else:
            grids = np.meshgrid(*centres, indexing="ij")
            coord_cols = [g.ravel(order="F").reshape(-1, 1) for g in grids]

        val = np.asarray(_call_user_func(func_handle, *coord_cols, pars=pars), dtype=float)
        wout = self.copy()
        s_new = val.reshape(self._s.shape, order="F").copy()
        e_new = np.zeros_like(s_new)

        if all_flag:
            wout._npix = np.ones_like(self._npix)
        else:
            nopix = self._npix == 0
            s_new[nopix] = 0.0

        wout._s = s_new
        wout._e = e_new
        if outfile:
            wout.save(outfile)
        return wout

    def sqw_eval_nopix(self, sqwfunc: Callable[..., Any], pars: Sequence[Any], opts: Dict[str, Any]) -> "DnDBase":
        """Evaluate S(Q, w) model function at bin centres of a DnD object."""
        qw = self.calculate_qw_bins()
        all_bins = bool(opts.get("all_bins", opts.get("all", False)))
        empty_npix = self._npix == 0
        has_any_pix = not np.all(empty_npix)

        if not all_bins and has_any_pix:
            keep = (~empty_npix).ravel(order="F")
            qw_eval = [q[keep] for q in qw]
            val_sub = np.asarray(_call_user_func(sqwfunc, *qw_eval, pars=pars), dtype=float).ravel(order="F")
            s_flat = np.zeros(self._s.size, dtype=float)
            s_flat[keep] = val_sub
            s_new = s_flat.reshape(self._s.shape, order="F")
        else:
            val = np.asarray(_call_user_func(sqwfunc, *qw, pars=pars), dtype=float)
            s_new = val.reshape(self._s.shape, order="F").copy()

        self._s = s_new
        self._e = np.zeros_like(s_new)
        if all_bins:
            self._npix = np.ones_like(self._npix)
        return self

    def dispersion(
        self, dispreln: Callable[..., Any], pars: Any, nargout: Optional[int] = None
    ) -> Union["DnDBase", DataOpArray, Tuple[Any, Any]]:
        """Calculate dispersion relation and spectral weight on a dataset or array of datasets."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        pars_list = list(pars) if isinstance(pars, (list, tuple)) else [pars]
        qw = self.calculate_qw_bins()

        # Project onto Q space (omit energy axis if energy is a plot axis)
        ndim = self.dimensions
        pax_list = [int(x) for x in self._axes.pax]
        if 4 in pax_list:
            en_axis_pos = pax_list.index(4)
            # Take first slice along energy axis to get unique Q points
            grid_shape = tuple(len(p) - 1 for p in self._axes.p)
            idx_sl = [slice(None)] * ndim
            idx_sl[en_axis_pos] = 0
            q_pts = [q.reshape(grid_shape, order="F")[tuple(idx_sl)].reshape(-1, 1, order="F") for q in qw[:3]]
            npix_q = np.sum(self._npix.reshape(grid_shape, order="F"), axis=en_axis_pos)
            pbin_all: List[Any] = [None] * 4
            for idx, iax_num in enumerate(self._axes.iax):
                iint_col = self._axes.iint[:, idx]
                pbin_all[int(iax_num) - 1] = 0 if np.any(np.isinf(iint_col)) else iint_col.tolist()
            for idx, pax_num in enumerate(pax_list):
                p_i = self._axes.p[idx]
                if pax_num == 4:
                    pbin_all[3] = [float(p_i[0]), float(p_i[-1])]
                else:
                    step = float(p_i[1] - p_i[0])
                    centres = 0.5 * (p_i[:-1] + p_i[1:])
                    pbin_all[pax_num - 1] = [float(centres[0]), step, float(centres[-1])]
            q_axes = type(self._axes)(*pbin_all)
            for idx, iax_num in enumerate(self._axes.iax):
                q_axes._img_range[:, int(iax_num) - 1] = self._axes.iint[:, idx]
            q_axes = self._proj.copy_proj_defined_properties_to_axes(q_axes)
            q_axes.title = self._axes.title
            q_axes.filename = self._axes.filename
            q_axes.filepath = self._axes.filepath
            q_axes.label = self._axes.label
            # Preserve display axis order excluding energy axis
            old_dax_pax = [pax_list[int(d) - 1] for d in self._axes.dax if pax_list[int(d) - 1] != 4]
            new_pax = [int(x) for x in q_axes.pax]
            if old_dax_pax:
                q_axes.dax = [new_pax.index(p) + 1 for p in old_dax_pax]
        else:
            q_pts = qw[:3]
            npix_q = self._npix.copy()
            q_axes = copy.deepcopy(self._axes)

        try:
            res = dispreln(q_pts[0], q_pts[1], q_pts[2], *pars_list)
        except Exception:
            res = dispreln(q_pts[0], q_pts[1], q_pts[2], pars)

        if isinstance(res, tuple) and len(res) >= 2:
            w_disp, s_weight = res[0], res[1]
        else:
            w_disp = res
            s_weight = None

        w_list = list(w_disp) if isinstance(w_disp, (list, tuple)) else [w_disp]
        if s_weight is not None:
            s_list = list(s_weight) if isinstance(s_weight, (list, tuple)) else [s_weight]
        else:
            s_list = [np.zeros_like(w) for w in w_list]

        wout_disp = []
        wout_weight = []
        target_sz = tuple(int(x) for x in q_axes.dims_as_ssize)
        for wi, si in zip(w_list, s_list):
            w_arr = np.asarray(wi, dtype=float).reshape(target_sz, order="F")
            s_arr = np.asarray(si, dtype=float).reshape(target_sz, order="F")
            npix_arr = npix_q.reshape(target_sz, order="F").copy()
            wout_disp.append(DnDBase.dnd(q_axes, self._proj, w_arr, s_arr, npix_arr))
            wout_weight.append(DnDBase.dnd(q_axes, self._proj, s_arr, np.zeros_like(s_arr), npix_arr))

        out_d = wout_disp[0] if len(wout_disp) == 1 else DataOpArray(wout_disp)
        out_w = wout_weight[0] if len(wout_weight) == 1 else DataOpArray(wout_weight)
        if nargout >= 2:
            return out_d, out_w
        return out_d

    # ------------------------------------------------------------------
    # Crystal alignment, aspect, distribute, noisify, save
    # ------------------------------------------------------------------
    def change_crystal(self, *args: Any) -> "DnDBase":
        """Change the crystal lattice and orientation of a DnD object."""
        alignment_info = self.parse_change_crystal_arguments(*args)
        wout = self.copy()
        wout._proj = wout._proj.align_proj(alignment_info)
        wout._axes = wout._proj.copy_proj_defined_properties_to_axes(wout._axes)
        return wout

    def adjust_aspect(self) -> bool:
        """Return True if plot aspect ratio should be adjusted automatically."""
        if hasattr(self._proj, "type") and isinstance(self._proj.type, str) and len(self._proj.type) >= 2:
            return self._proj.type[1] == "a"
        return False

    def distribute(self, nWorkers: int = 1, split_bins: bool = True) -> Tuple[Any, Dict[str, Any]]:
        """Distribute a DnD object among parallel workers."""
        return [self.copy()], {"n_workers": 1, "split_bins": split_bins}

    def noisify(self, *args: Any, **kwargs: Any) -> "DnDBase":
        """Add noise to the signal and error of a DnD object.

        Syntax:
            >> wout = w.noisify()
            >> wout = w.noisify(factor)
            >> wout = w.noisify(factor, 'maximum_value', max_val)
            >> wout = w.noisify('poisson')
        """
        wout = self.copy()
        rng = np.random.default_rng()
        poisson = False
        factor = 0.1
        max_val = None

        i = 0
        while i < len(args):
            a = args[i]
            if _is_numeric(a) and np.asarray(a).size == 1:
                factor = float(np.asarray(a).flat[0])
                i += 1
            elif isinstance(a, str) and a.lower().startswith("pois"):
                poisson = True
                i += 1
            elif isinstance(a, str) and a.lower().startswith("max"):
                if i + 1 < len(args):
                    max_val = float(args[i + 1])
                    i += 2
                else:
                    i += 1
            elif callable(a):
                i += 1
            else:
                i += 1

        if "poisson" in kwargs:
            poisson = bool(kwargs["poisson"])
        if "factor" in kwargs:
            factor = float(kwargs["factor"])

        valid = self._npix > 0
        s_new = self._s.copy()
        e_new = self._e.copy()

        if poisson:
            lam = np.maximum(0.0, s_new[valid])
            noisy = rng.poisson(lam).astype(float)
            s_new[valid] = noisy
            e_new[valid] = noisy
        else:
            if max_val is None:
                max_val = float(np.max(np.abs(s_new[valid]))) if np.any(valid) else 1.0
                if max_val == 0.0:
                    max_val = 1.0
            sigma = factor * max_val
            s_new[valid] = s_new[valid] + rng.normal(0.0, sigma, size=int(np.sum(valid)))
            e_new[valid] = sigma**2

        wout._s = s_new
        wout._e = e_new
        return wout


def dnd(*args: Any, **kwargs: Any) -> DnDBase:
    """Factory function for constructing a d0d..d4d object."""
    return DnDBase.dnd(*args, **kwargs)

