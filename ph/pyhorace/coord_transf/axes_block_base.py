"""Abstract base class for coordinate axes blocks (AxesBlockBase)."""

from __future__ import annotations

import copy
from abc import ABC
from typing import Any, ClassVar, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyhorace.coord_transf._dependencies import (
    HoraceError,
    PixelDataBase,
    PixelDataMemory,
)
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.coord_transf.expand_box import expand_box
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.serializable import Serializable

EPS_SINGLE = float(np.finfo(np.float32).eps)


def _is_range_2x4(val: Any) -> bool:
    if isinstance(val, np.ndarray) and val.shape == (2, 4):
        return True
    if (
        isinstance(val, (list, tuple))
        and len(val) == 2
        and all(
            isinstance(r, (list, tuple, np.ndarray)) and len(r) == 4 for r in val
        )
    ):
        return True
    return False


def _is_nbins_4(val: Any) -> bool:
    if isinstance(val, np.ndarray) and val.size == 4 and val.ndim <= 2:
        if val.shape in ((4,), (1, 4), (4, 1)):
            return True
    if (
        isinstance(val, (list, tuple))
        and len(val) == 4
        and all(isinstance(x, (int, float, np.number)) for x in val)
    ):
        return True
    return False


class AxesBlockBase(Serializable, ABC):
    """Abstract base class defining 4D coordinate axes blocks in Horace."""

    EMPTY_RANGE_: ClassVar[np.ndarray] = np.array(
        [[np.inf, np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf, -np.inf]],
        dtype=float,
    )

    _base_fields: ClassVar[List[str]] = [
        "title",
        "filename",
        "filepath",
        "label",
        "ulen",
        "img_scales",
        "img_range",
        "nbins_all_dims",
        "single_bin_defines_iax",
        "dax",
        "offset",
        "changes_aspect_ratio",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._angular_unit_is_rad: bool = False
        self._title: str = ""
        self._filename: str = ""
        self._filepath: str = ""
        self._label: List[str] = ["Q_h", "Q_k", "Q_l", "En"]
        self._ulen: np.ndarray = np.ones(4, dtype=float)
        self._type: str = "pppe"
        self._img_range: np.ndarray = self._default_img_range.copy()
        self._nbins_all_dims: np.ndarray = np.ones(4, dtype=int)
        self._single_bin_defines_iax: np.ndarray = np.ones(4, dtype=bool)
        self._dax: Optional[np.ndarray] = None
        self._offset: np.ndarray = np.zeros(4, dtype=float)
        self._changes_aspect_ratio: bool = True
        self._dimensions: Optional[int] = None
        self._pax: Optional[np.ndarray] = None
        self._iax: Optional[np.ndarray] = None
        self._iint: Optional[np.ndarray] = None
        self._p: Optional[List[np.ndarray]] = None
        self._F_interp: Optional[Any] = None
        self._interp_grid_size: Optional[np.ndarray] = None
        self._creation_date: str = ""

    # ------------------------------------------------------------------
    # Class-level defaults overridden by subclasses
    # ------------------------------------------------------------------
    @property
    def _max_img_range(self) -> np.ndarray:
        return np.array(
            [[-np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]],
            dtype=float,
        )

    @property
    def _default_img_range(self) -> np.ndarray:
        return self.EMPTY_RANGE_.copy()

    @property
    def _p_defines_bin_centers(self) -> np.ndarray:
        return np.ones(4, dtype=bool)

    @property
    def creation_date(self) -> str:
        if self._creation_date:
            return self._creation_date
        import datetime

        return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    @creation_date.setter
    def creation_date(self, val: Any) -> None:
        self._creation_date = "" if val is None else str(val)

    @property
    def creation_date_defined(self) -> bool:
        return bool(self._creation_date)

    @property
    def line_at_direct_ang(self) -> bool:
        return True

    def class_version(self) -> int:
        return 3

    def saveable_fields(self) -> List[str]:
        return list(self._base_fields)

    def _err_cls_name(self) -> str:
        name = self.__class__.__name__
        if name in ("line_axesTester", "line_axes_char_size_tester"):
            return "AxesBlockBase"
        return name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, val: Any) -> None:
        if val is None:
            self._title = ""
        elif isinstance(val, (str, np.str_)):
            self._title = str(val)
        elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0:
            self._title = ""
        else:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Title must be a character string",
            )

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, val: Any) -> None:
        if val is None:
            self._filename = ""
        elif isinstance(val, (str, np.str_)):
            self._filename = str(val)
        elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0:
            self._filename = ""
        else:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Filename must be a character string",
            )

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, val: Any) -> None:
        if val is None:
            self._filepath = ""
        elif isinstance(val, (str, np.str_)):
            self._filepath = str(val)
        elif isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0:
            self._filepath = ""
        else:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Filepath must be a character string",
            )

    @property
    def label(self) -> List[str]:
        return list(self._label)

    @label.setter
    def label(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._label = ["", "", "", ""]
            return
        if isinstance(val, (list, tuple, np.ndarray)) and len(val) == 4:
            if all(isinstance(x, (str, np.str_)) for x in val):
                self._label = [str(x) for x in val]
                return
        raise HoraceError(
            f"HORACE:{self._err_cls_name()}:invalid_argument",
            "label must be a 4-element cell array of strings",
        )

    @property
    def ulen(self) -> np.ndarray:
        return self._ulen.copy()

    @ulen.setter
    def ulen(self, val: Any) -> None:
        self.img_scales = val

    @property
    def img_scales(self) -> np.ndarray:
        return self._ulen.copy()

    @img_scales.setter
    def img_scales(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_scales must be a 4-element numeric vector of positive numbers",
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_scales must be a 4-element numeric vector",
            ) from exc
        if arr.size != 4 or np.any(arr <= 0):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_scales must be a 4-element numeric vector of positive numbers",
            )
        self._ulen = arr.copy()

    @property
    def type(self) -> str:
        return self._type

    @type.setter
    def type(self, val: str) -> None:
        if not isinstance(val, str) or len(val) not in (3, 4):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "type must be a 3- or 4-character string",
            )
        if len(val) == 3:
            val = val + "e"
        if isinstance(getattr(type(self), "angular_unit_is_rad", None), property):
            self.angular_unit_is_rad = val[2].lower() == "r"
        self._type = val

    @property
    def uoffset(self) -> np.ndarray:
        return self._offset.copy()

    @uoffset.setter
    def uoffset(self, val: Any) -> None:
        self.offset = val

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            ) from exc
        if arr.size == 3:
            arr = np.append(arr, 0.0)
        elif arr.size != 4:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "offset must be a 3 or 4 element numeric vector",
            )
        arr[np.abs(arr) < EPS_SINGLE] = 0.0
        self._offset = arr.copy()

    @property
    def changes_aspect_ratio(self) -> bool:
        return bool(self._changes_aspect_ratio)

    @changes_aspect_ratio.setter
    def changes_aspect_ratio(self, val: Any) -> None:
        self._changes_aspect_ratio = bool(val)

    @property
    def img_range(self) -> np.ndarray:
        return self._img_range.copy()

    @img_range.setter
    def img_range(self, val: Any) -> None:
        self._set_img_range(val)
        if not self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_img_range(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_range must be a 2x4 numeric array",
            )
        try:
            arr = np.asarray(val, dtype=float)
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_range must be a 2x4 numeric array",
            ) from exc
        if arr.shape != (2, 4):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"img_range must be a 2x4 numeric array, got shape {arr.shape}",
            )
        if np.array_equal(arr, self.EMPTY_RANGE_):
            self._img_range = self.EMPTY_RANGE_.copy()
            return
        if np.any(np.isnan(arr)) or np.any(arr[0, :] == np.inf) or np.any(arr[1, :] == -np.inf):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "img_range cannot contain Inf or NaN unless it is EMPTY_RANGE_",
            )
        if np.any(arr[0, :] > arr[1, :] + EPS_SINGLE):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Lower limits in img_range must not exceed upper limits: {arr}",
            )
        max_r = self._max_img_range
        if np.any(arr[0, :] < max_r[0, :] - EPS_SINGLE) or np.any(
            arr[1, :] > max_r[1, :] + EPS_SINGLE
        ):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"img_range {arr} exceeds allowed limits {max_r}",
            )
        arr = np.maximum(arr, max_r[0:1, :])
        arr = np.minimum(arr, max_r[1:2, :])
        self._img_range = arr.copy()

    @property
    def img_db_range(self) -> np.ndarray:
        return self.img_range

    @img_db_range.setter
    def img_db_range(self, val: Any) -> None:
        self.img_range = val

    @property
    def nbins_all_dims(self) -> np.ndarray:
        return self._nbins_all_dims.copy()

    @nbins_all_dims.setter
    def nbins_all_dims(self, val: Any) -> None:
        self._set_nbins_all_dims(val)
        if not self._do_check_combo_arg:
            self.check_combo_arg()

    def _set_nbins_all_dims(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "nbins_all_dims must be a 4-element positive integer vector",
            )
        try:
            arr = np.asarray(val, dtype=float).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "nbins_all_dims must be a 4-element positive integer vector",
            ) from exc
        if arr.size != 4 or np.any(arr < 1) or np.any(np.floor(arr) != arr):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "nbins_all_dims must be a 4-element vector of positive integers >= 1",
            )
        int_arr = arr.astype(int)
        is_multibin = int_arr > 1
        if np.any(is_multibin):
            self._single_bin_defines_iax[is_multibin] = False
        self._nbins_all_dims = int_arr

    @property
    def single_bin_defines_iax(self) -> np.ndarray:
        return self._single_bin_defines_iax.copy()

    @single_bin_defines_iax.setter
    def single_bin_defines_iax(self, val: Any) -> None:
        if isinstance(val, (str, bytes)):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "single_bin_defines_iax must be a 4-element logical vector",
            )
        try:
            arr = np.asarray(val, dtype=bool).ravel()
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "single_bin_defines_iax must be a 4-element logical vector",
            ) from exc
        if arr.size != 4:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "single_bin_defines_iax must be a 4-element logical vector",
            )
        self._single_bin_defines_iax = arr.copy()
        if not self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def dax(self) -> np.ndarray:
        if self._dax is None:
            return np.array([], dtype=int)
        return self._dax.copy()

    @dax.setter
    def dax(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._dax = np.array([], dtype=int)
        else:
            if isinstance(val, (str, bytes)):
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    "dax must be a permutation of 1..dimensions",
                )
            try:
                arr = np.asarray(val, dtype=int).ravel()
            except (ValueError, TypeError) as exc:
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    "dax must be an integer permutation vector",
                ) from exc
            self._dax = arr.copy()
        if not self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self.check_combo_arg()
        return int(self._dimensions)

    @property
    def pax(self) -> np.ndarray:
        if self._pax is None:
            self.check_combo_arg()
        return self._pax.copy()

    @property
    def iax(self) -> np.ndarray:
        if self._iax is None:
            self.check_combo_arg()
        return self._iax.copy()

    @property
    def iint(self) -> np.ndarray:
        if self._iint is None:
            self.check_combo_arg()
        return self._iint.copy()

    @property
    def p(self) -> List[np.ndarray]:
        if self._p is None:
            self.check_combo_arg()
        return [arr.copy() for arr in self._p]

    @property
    def dims_as_ssize(self) -> np.ndarray:
        dims = self._nbins_all_dims[self.pax - 1]
        if dims.size == 0:
            return np.array([1, 1], dtype=int)
        if dims.size == 1:
            return np.array([ int(dims[0]), 1], dtype=int)
        return dims.astype(int).copy()

    @property
    def data_nbins(self) -> np.ndarray:
        return self._nbins_all_dims[self.pax - 1].copy()

    @property
    def step(self) -> np.ndarray:
        if self.pax.size == 0:
            return np.array([], dtype=float)
        idx = self.pax - 1
        return (self._img_range[1, idx] - self._img_range[0, idx]) / self._nbins_all_dims[idx]

    @property
    def full_img_AutoScale(self) -> np.ndarray:
        return self._img_range * self._ulen[np.newaxis, :]

    # ------------------------------------------------------------------
    # Construction & Parsing
    # ------------------------------------------------------------------
    def _init_from_args(self, *args, **kwargs) -> None:
        if len(args) == 0 and len(kwargs) == 0:
            self.check_combo_arg()
            return
        self.init(*args, **kwargs)

    def init(self, *args, **kwargs) -> "AxesBlockBase":
        if len(args) == 0 and len(kwargs) == 0:
            self.check_combo_arg()
            return self

        if len(args) == 1 and len(kwargs) == 0:
            arg = args[0]
            if isinstance(arg, AxesBlockBase):
                for fld in self.saveable_fields():
                    if hasattr(arg, fld):
                        setattr(self, fld, copy.deepcopy(getattr(arg, fld)))
                self.check_combo_arg()
                return self
            if isinstance(arg, dict):
                self._init_from_dict(arg)
                return self
            if isinstance(arg, (int, np.integer)):
                ndim = int(arg)
                if ndim not in (0, 1, 2, 3, 4):
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Number of dimensions must be 0, 1, 2, 3, or 4, got {ndim}",
                    )
                self._do_check_combo_arg = True
                try:
                    sbd = np.ones(4, dtype=bool)
                    if ndim > 0:
                        sbd[:ndim] = False
                    self._single_bin_defines_iax = sbd
                    self._nbins_all_dims = np.ones(4, dtype=int)
                    self._dax = np.arange(1, ndim + 1, dtype=int)
                finally:
                    self._do_check_combo_arg = False
                self.check_combo_arg()
                return self
            if (
                isinstance(arg, (list, tuple))
                and len(arg) == 4
                and not _is_range_2x4(arg)
                and not _is_nbins_4(arg)
            ):
                args = tuple(arg)

        # Separate positional pbin / range_and_nbins args from keyword-value pairs
        pos_args = list(args)
        kv_dict = dict(kwargs)

        # Check if first 2 args are (img_range_2x4, nbins_all_dims_4)
        pbin_args = None
        range_nbins_pair = None
        if (
            len(pos_args) >= 2
            and _is_range_2x4(pos_args[0])
            and _is_nbins_4(pos_args[1])
        ):
            range_nbins_pair = (pos_args[0], pos_args[1])
            rem = pos_args[2:]
        elif len(pos_args) >= 4 and not isinstance(pos_args[0], str):
            pbin_args = pos_args[:4]
            rem = pos_args[4:]
        else:
            rem = pos_args

        # Parse remaining positional args as key-value pairs or single-arg struct
        i = 0
        while i < len(rem):
            k = rem[i]
            if isinstance(k, str):
                key = k.lstrip("-")
                if i + 1 < len(rem):
                    kv_dict[key] = rem[i + 1]
                    i += 2
                else:
                    kv_dict[key] = True
                    i += 1
            else:
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    f"Unrecognized positional arguments: {rem}",
                )

        self._do_check_combo_arg = True
        try:
            # Apply non-range/nbins keyword arguments first (e.g. angular_unit_is_rad)
            deferred_keys = {}
            for k, v in kv_dict.items():
                if k in ("img_range", "nbins_all_dims", "single_bin_defines_iax", "dax"):
                    deferred_keys[k] = v
                elif hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Unknown property '{k}' for {self.__class__.__name__}",
                    )

            if range_nbins_pair is not None:
                self._set_img_range(range_nbins_pair[0])
                self._set_nbins_all_dims(range_nbins_pair[1])
            elif pbin_args is not None:
                self._set_axis_bins(pbin_args)

            for k, v in deferred_keys.items():
                setattr(self, k, v)
        finally:
            self._do_check_combo_arg = False

        self.check_combo_arg()
        return self

    def _init_from_dict(self, d: dict) -> None:
        if "serializable_full_name" in d or ("version" in d and "array_dat" in d):
            loaded = Serializable.from_struct(d)
            for fld in self.saveable_fields():
                if hasattr(loaded, fld):
                    setattr(self, fld, getattr(loaded, fld))
            self.check_combo_arg()
            return
        # Check if legacy struct
        if "p" in d and "pax" in d and "img_range" not in d:
            loaded = self.get_from_old_data(d)
            for fld in self.saveable_fields():
                if hasattr(loaded, fld):
                    setattr(self, fld, getattr(loaded, fld))
            self.check_combo_arg()
            return

        self._do_check_combo_arg = True
        try:
            for k, v in d.items():
                if hasattr(self, k):
                    setattr(self, k, v)
        finally:
            self._do_check_combo_arg = False
        self.check_combo_arg()

    def _set_axis_bins(self, pbin_list: Sequence[Any]) -> None:
        if len(pbin_list) != 4:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Four binning arguments are required to initialize axes",
            )
        p_centers = self._p_defines_bin_centers
        max_r = self._max_img_range
        img_r = np.zeros((2, 4), dtype=float)
        nbins = np.ones(4, dtype=int)
        sbd = np.ones(4, dtype=bool)

        for i in range(4):
            pb = pbin_list[i]
            r_i, nb_i, ok, mess = self.pbin_parse(pb, bool(p_centers[i]), max_r[:, i])
            if not ok:
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    f"Axis {i + 1}: {mess}",
                )
            img_r[:, i] = r_i
            nbins[i] = nb_i
            arr_pb = np.asarray(pb, dtype=float).ravel() if pb is not None else np.array([])
            if arr_pb.size == 3 or arr_pb.size == 0:
                sbd[i] = False
            else:
                sbd[i] = True

        self._img_range = img_r
        self._nbins_all_dims = nbins
        self._single_bin_defines_iax = sbd
        self._dax = None

    def pbin_parse(
        self,
        p: Any,
        p_defines_bin_centers: bool,
        range_limits: Sequence[float],
    ) -> Tuple[Optional[np.ndarray], Optional[int], bool, str]:
        """Parse a single axis binning specification into (range_2x1, nbins, ok, mess)."""
        if isinstance(p, (str, bytes)):
            return (
                None,
                None,
                False,
                "Binning description must be a numeric vector of length 0, 1, 2, or 3",
            )
        if p is None:
            arr = np.array([], dtype=float)
        else:
            try:
                arr = np.asarray(p, dtype=float)
            except (ValueError, TypeError):
                return (
                    None,
                    None,
                    False,
                    "Binning description must be a numeric vector",
                )
        if arr.ndim > 2 or (arr.ndim == 2 and min(arr.shape) > 1):
            return (
                None,
                None,
                False,
                "Binning description must be a row or column vector",
            )
        arr = arr.ravel()
        lim = np.asarray(range_limits, dtype=float).ravel()

        if arr.size == 0:
            return np.array([0.0, 0.0], dtype=float), 1, True, ""

        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            return None, None, False, "Binning description cannot contain NaN or Inf"

        if arr.size == 1:
            val = float(arr[0])
            if val < lim[0] - EPS_SINGLE or val > lim[1] + EPS_SINGLE:
                return (
                    None,
                    None,
                    False,
                    f"Integration point {val} is outside allowed range [{lim[0]}, {lim[1]}]",
                )
            val = float(np.clip(val, lim[0], lim[1]))
            return np.array([val, val], dtype=float), 1, True, ""

        if arr.size == 2:
            if arr[0] > arr[1]:
                return (
                    None,
                    None,
                    False,
                    f"Lower integration limit {arr[0]} exceeds upper limit {arr[1]}",
                )
            if arr[0] < lim[0] - EPS_SINGLE or arr[1] > lim[1] + EPS_SINGLE:
                return (
                    None,
                    None,
                    False,
                    f"Integration range [{arr[0]}, {arr[1]}] exceeds limits [{lim[0]}, {lim[1]}]",
                )
            r0 = max(float(arr[0]), float(lim[0]))
            r1 = min(float(arr[1]), float(lim[1]))
            return np.array([r0, r1], dtype=float), 1, True, ""

        if arr.size == 3:
            if arr[0] > arr[2]:
                return (
                    None,
                    None,
                    False,
                    f"Lower plot limit {arr[0]} exceeds upper limit {arr[2]}",
                )
            if arr[1] <= 0:
                return (
                    None,
                    None,
                    False,
                    f"Plot step must be strictly positive, got {arr[1]}",
                )
            if arr[0] < lim[0] - EPS_SINGLE or arr[2] > lim[1] + EPS_SINGLE:
                return (
                    None,
                    None,
                    False,
                    f"Plot range [{arr[0]}, {arr[2]}] exceeds limits [{lim[0]}, {lim[1]}]",
                )
            if p_defines_bin_centers:
                if abs(arr[0] - arr[2]) < EPS_SINGLE and arr[1] > abs(arr[2] - arr[0]) + EPS_SINGLE:
                    # When min == max and step is specified, check if step exceeds range limits
                    pass
                step = float(arr[1])
                half_step = 0.5 * step
                min_v = float(arr[0]) - half_step
                max_v = float(arr[2]) + half_step
                range_constrained = False
                if min_v < lim[0]:
                    if abs(float(arr[0]) - lim[0]) < EPS_SINGLE:
                        max_v = max_v + half_step
                    min_v = float(lim[0])
                    range_constrained = True
                if max_v > lim[1]:
                    max_v = float(lim[1])
                    range_constrained = True
                if max_v - min_v < step - EPS_SINGLE and range_constrained:
                    return (
                        None,
                        None,
                        False,
                        f"Step {step} exceeds available range [{min_v}, {max_v}]",
                    )
                nbin = int(np.floor((max_v - min_v) / step))
                if abs(min_v + nbin * step - max_v) > EPS_SINGLE:
                    if range_constrained:
                        if nbin < 1:
                            return (
                                None,
                                None,
                                False,
                                f"Step {step} is too large for constrained range [{min_v}, {max_v}]",
                            )
                        step = (max_v - min_v) / nbin
                    else:
                        nbin += 1
                max_v = min_v + nbin * step
                return np.array([min_v, max_v], dtype=float), nbin, True, ""
            else:
                min_v = float(arr[0])
                step_in = float(arr[1])
                max_v = float(arr[2])
                if abs(max_v - min_v) < step_in - EPS_SINGLE:
                    return (
                        None,
                        None,
                        False,
                        f"Step {step_in} exceeds range [{min_v}, {max_v}]",
                    )
                nbin = max(int(np.floor((max_v - min_v) / step_in)), 1)
                step = (max_v - min_v) / nbin
                max_v = min_v + nbin * step
                return np.array([min_v, max_v], dtype=float), nbin, True, ""

        return (
            None,
            None,
            False,
            f"Binning description must have 0, 1, 2, or 3 elements, got {arr.size}",
        )

    def check_combo_arg(self) -> "AxesBlockBase":
        """Validate interdependent properties and compute pax, iax, iint, p, dax, dimensions."""
        self._F_interp = None
        self._interp_grid_size = None

        is_iax = (self._nbins_all_dims == 1) & self._single_bin_defines_iax
        is_pax = ~is_iax

        # Ensure single_bin_defines_iax is consistent with nbins_all_dims > 1
        self._single_bin_defines_iax = is_iax.copy()

        self._pax = np.where(is_pax)[0] + 1
        self._iax = np.where(is_iax)[0] + 1
        self._dimensions = int(self._pax.size)

        if self._iax.size > 0:
            self._iint = self._img_range[:, self._iax - 1].copy()
        else:
            self._iint = np.zeros((2, 0), dtype=float)

        p_list: List[np.ndarray] = []
        for idx0 in self._pax - 1:
            nb = int(self._nbins_all_dims[idx0])
            r0 = float(self._img_range[0, idx0])
            r1 = float(self._img_range[1, idx0])
            if np.isinf(r0) or np.isinf(r1):
                edges = np.zeros(nb + 1, dtype=float)
            else:
                edges = np.linspace(r0, r1, nb + 1, dtype=float)
            p_list.append(edges)
        self._p = p_list

        if self._dax is None or (self._dax.size == 0 and self._dimensions > 0):
            self._dax = np.arange(1, self._dimensions + 1, dtype=int)
        else:
            if self._dax.size != self._dimensions or sorted(self._dax.tolist()) != list(
                range(1, self._dimensions + 1)
            ):
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    f"dax {self._dax} is incompatible with {self._dimensions} plot axes",
                )
        return self

    # ------------------------------------------------------------------
    # Binning & Cut Range Helpers
    # ------------------------------------------------------------------
    def default_pbin(self) -> List[np.ndarray]:
        """Return default pbin cell array describing current axes binning."""
        res: List[np.ndarray] = []
        is_iax = self._single_bin_defines_iax
        p_centers = self._p_defines_bin_centers
        for i in range(4):
            r0 = float(self._img_range[0, i])
            r1 = float(self._img_range[1, i])
            nb = int(self._nbins_all_dims[i])
            if np.isinf(r0) or np.isinf(r1):
                res.append(np.array([], dtype=float))
            elif is_iax[i]:
                res.append(np.array([r0, r1], dtype=float))
            else:
                step = (r1 - r0) / nb
                if step == 0.0:
                    res.append(np.array([], dtype=float))
                elif p_centers[i]:
                    res.append(
                        np.array([r0 + 0.5 * step, step, r1 - 0.5 * step], dtype=float)
                    )
                else:
                    res.append(np.array([r0, step, r1], dtype=float))
        return res

    def get_cut_range(self, *args: str) -> List[np.ndarray]:
        """Return binning range of existing data object so that cut within this range returns the same cut."""
        full_range = False
        for a in args:
            if isinstance(a, str) and "-full_range".startswith(a.lower()) and a.startswith("-"):
                full_range = True
            else:
                raise HoraceError(
                    "HORACE:AxesBlockBase:invalid_argument",
                    f"Unrecognized option to get_cut_range: {a!r}",
                )

        ndims = int(self.pax.size)
        bin_range: List[np.ndarray] = []
        for i in range(ndims):
            p_i = self.p[i]
            bin_centers = 0.5 * (p_i[:-1] + p_i[1:])
            min_unref = float(np.min(bin_centers))
            max_unref = float(np.max(bin_centers))
            step = float(p_i[1] - p_i[0])
            bin_range.append(np.array([min_unref, step, max_unref], dtype=float))

        if not full_range:
            return bin_range

        fl_range: List[np.ndarray] = [np.array([], dtype=float) for _ in range(4)]
        for idx, pax_num in enumerate(self.pax):
            fl_range[int(pax_num) - 1] = bin_range[idx]
        for idx, iax_num in enumerate(self.iax):
            fl_range[int(iax_num) - 1] = self.iint[:, idx].copy()
        return fl_range

    @staticmethod
    def get_img_range_from_cut_range(*args) -> np.ndarray:
        """Compute [2, N] image range from N cut binning descriptors."""
        if (
            len(args) == 1
            and isinstance(args[0], (list, tuple))
            and len(args[0]) > 0
            and isinstance(args[0][0], (list, tuple, np.ndarray))
        ):
            bin_pars = list(args[0])
        else:
            bin_pars = list(args)

        n_pars = len(bin_pars)
        rng = np.vstack([np.full(n_pars, -np.inf), np.full(n_pars, np.inf)])
        for i, bp in enumerate(bin_pars):
            if bp is None:
                continue
            arr = np.asarray(bp, dtype=float).ravel()
            if arr.size == 2:
                rng[:, i] = [arr[0], arr[1]]
            elif arr.size == 3:
                if arr[1] == 0:
                    raise HoraceError(
                        "HORACE:AxesBlock:invalid_argument",
                        f"Bin step cannot be zero in axis {i + 1}: {arr}",
                    )
                rng[0, i] = arr[0] - 0.5 * arr[1]
                rng[1, i] = arr[2] + 0.5 * arr[1]
        return rng

    @classmethod
    def build_from_input_binning(
        cls,
        arg1: Any,
        arg2: Any = None,
        arg3: Any = None,
    ) -> "AxesBlockBase":
        """Build a target AxesBlockBase instance by merging existing range/steps with requested pbin."""
        from pyhorace.coord_transf.cylinder_axes import cylinder_axes
        from pyhorace.coord_transf.line_axes import line_axes
        from pyhorace.coord_transf.sphere_axes import sphere_axes

        cls_map = {
            "line_axes": line_axes,
            "sphere_axes": sphere_axes,
            "cylinder_axes": cylinder_axes,
        }

        if arg3 is not None:
            # Called with (axes_class_or_name_or_unit, cur_img_range_and_steps, pbin)
            if isinstance(arg1, AxesBlockBase):
                new_ax = copy.deepcopy(arg1)
            elif isinstance(arg1, type) and issubclass(arg1, AxesBlockBase):
                new_ax = arg1()
            elif isinstance(arg1, str) and arg1 in cls_map:
                new_ax = cls_map[arg1]()
            elif isinstance(arg1, (str, bool, np.bool_)):
                # Subclass called with angular unit (e.g. sphere_axes.build_from_input_binning('deg', ...))
                new_ax = cls()
                if hasattr(new_ax, "angular_unit_is_rad"):
                    if isinstance(arg1, (bool, np.bool_)):
                        new_ax.angular_unit_is_rad = bool(arg1)
                    elif str(arg1).lower().startswith("r"):
                        new_ax.angular_unit_is_rad = True
                    else:
                        new_ax.angular_unit_is_rad = False
            else:
                raise HoraceError(
                    "HORACE:AxesBlockBase:invalid_argument",
                    f"Unrecognized first argument to build_from_input_binning: {arg1}",
                )
            cur_img_range_and_steps = arg2
            pbin = arg3
        else:
            # 2 arguments: cls.build_from_input_binning(cur_img_range_and_steps, pbin)
            if cls is AxesBlockBase:
                new_ax = line_axes()
            else:
                new_ax = cls()
            cur_img_range_and_steps = arg1
            pbin = arg2

        if len(cur_img_range_and_steps) != 4 or len(pbin) != 4:
            raise HoraceError(
                f"HORACE:{new_ax._err_cls_name()}:invalid_argument",
                "cur_img_range_and_steps and pbin must both have 4 elements",
            )

        data_range = new_ax.get_img_range_from_cut_range(*cur_img_range_and_steps)
        default_bins = new_ax.default_pbin()
        targ_pbin: List[np.ndarray] = []

        for ind in range(4):
            bin_req = pbin[ind]
            cur_bin = cur_img_range_and_steps[ind]
            bin_def = default_bins[ind]
            targ_pbin.append(
                AxesBlockBase._parse_pbin_item(
                    ind, bin_req, cur_bin, data_range[:, ind], bin_def
                )
            )

        return new_ax.init(*targ_pbin)

    @staticmethod
    def _parse_pbin_item(
        ind: int,
        bin_req: Any,
        cur_bin: Any,
        data_range_1d: np.ndarray,
        bin_default: np.ndarray,
    ) -> np.ndarray:
        req = (
            np.array([], dtype=float)
            if bin_req is None
            else np.asarray(bin_req, dtype=float).ravel()
        )
        cur = (
            np.array([], dtype=float)
            if cur_bin is None
            else np.asarray(cur_bin, dtype=float).ravel()
        )
        def_b = (
            np.array([], dtype=float)
            if bin_default is None
            else np.asarray(bin_default, dtype=float).ravel()
        )

        cur_step = float(cur[1]) if cur.size == 3 else 0.0
        r_min = float(data_range_1d[0])
        r_max = float(data_range_1d[1])
        if np.isinf(r_min) and def_b.size >= 2:
            r_min = float(def_b[0])
        if np.isinf(r_max) and def_b.size >= 2:
            r_max = float(def_b[-1])

        if req.size == 0:
            if cur.size == 3 and cur_step > 0:
                return cur.copy()
            elif cur.size == 2 and not np.any(np.isinf(cur)):
                return cur.copy()
            else:
                return np.array([r_min, r_max], dtype=float)

        if req.size == 1:
            step = float(req[0])
            if step == 0.0:
                if cur_step > 0:
                    step = cur_step
                else:
                    raise HoraceError(
                        "HORACE:AxesBlockBase:invalid_argument",
                        f"Cannot use zero step for axis {ind + 1} when original data is integrated",
                    )
            if cur.size == 3 and abs(step - cur_step) < EPS_SINGLE:
                return cur.copy()
            return np.array([r_min + 0.5 * step, step, r_max - 0.5 * step], dtype=float)

        if req.size == 2:
            lo = r_min if np.isinf(req[0]) else float(req[0])
            hi = r_max if np.isinf(req[1]) else float(req[1])
            return np.array([lo, hi], dtype=float)

        if req.size == 3:
            step = float(req[1])
            if step == 0.0:
                if cur_step > 0:
                    step = cur_step
                else:
                    raise HoraceError(
                        "HORACE:AxesBlockBase:invalid_argument",
                        f"Cannot use zero step for axis {ind + 1} when original data is integrated",
                    )
            lo = r_min + 0.5 * step if np.isinf(req[0]) else float(req[0])
            hi = r_max - 0.5 * step if np.isinf(req[2]) else float(req[2])
            return np.array([lo, step, hi], dtype=float)

        raise HoraceError(
            "HORACE:AxesBlockBase:invalid_argument",
            f"Invalid binning request for axis {ind + 1}: {bin_req}",
        )

    @staticmethod
    def bins_in_1Drange(
        bins: Sequence[float], minmax: Sequence[float]
    ) -> Tuple[bool, np.ndarray]:
        """Check which 1D bins defined by bin edges overlap with [minmax[0], minmax[1]]."""
        b = np.asarray(bins, dtype=float).ravel()
        r = np.asarray(minmax, dtype=float).ravel()
        if b.size < 2 or r.size < 2:
            return False, np.array([], dtype=bool)
        wk_low = b < r[0]
        all_low = wk_low[:-1] & wk_low[1:]
        wk_high = b > r[1]
        all_high = wk_high[:-1] & wk_high[1:]
        is_within = ~(all_low | all_high)
        any_within = bool(np.any(is_within))
        return any_within, is_within

    # ------------------------------------------------------------------
    # Characteristic Size & Interpolation Nodes
    # ------------------------------------------------------------------
    def get_char_size(self, this_proj: Any = None) -> np.ndarray:
        """Return 4-element characteristic bin size of the axes block."""
        if this_proj is None:
            return (self._img_range[1, :] - self._img_range[0, :]) / self._nbins_all_dims

        img_coords, dE_nodes, nbs, _ = self.get_bin_nodes("-3D", nargout=4)
        img_coords = this_proj.transform_img_to_pix(img_coords)
        nx, ny, nz = int(nbs[0]), int(nbs[1]), int(nbs[2])
        x = img_coords[0, :].reshape((nx, ny, nz), order="F")
        y = img_coords[1, :].reshape((nx, ny, nz), order="F")
        z = img_coords[2, :].reshape((nx, ny, nz), order="F")
        i0 = max(nx - 2, 0)
        xb = x[i0 : i0 + 2, 0:2, 0:2].ravel()
        yb = y[i0 : i0 + 2, 0:2, 0:2].ravel()
        zb = z[i0 : i0 + 2, 0:2, 0:2].ravel()
        pts = np.vstack([xb, yb, zb])
        sz_3d = np.max(pts, axis=1) - np.min(pts, axis=1)
        de = float(dE_nodes[1] - dE_nodes[0])
        return np.array([sz_3d[0], sz_3d[1], sz_3d[2], de], dtype=float)

    def get_interp_nodes(
        self, this_proj: Any = None, char_sizes: Any = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (nodes_3d, en_edges) for interpolation or contribution checks."""
        if isinstance(char_sizes, AxesBlockBase):
            # Legacy call signature (source_proj, targ_proj, targ_axes)
            targ_axes = char_sizes
            targ_proj = this_proj
            sz = targ_axes.get_char_size(targ_proj)
            nodes_3d, en_edges = self.get_bin_nodes("-3D", nargout=2)
            return nodes_3d, en_edges

        if char_sizes is not None:
            cs = np.asarray(char_sizes, dtype=float).ravel()
            my_sz = self.get_char_size(this_proj)
            ratio = np.ceil(my_sz[:3] / np.maximum(cs[:3], 1e-9)).astype(int)
            ratio = np.clip(ratio, 1, 16)
            nb = np.append(self._nbins_all_dims[:3] * ratio, self._nbins_all_dims[3])
            nodes_3d, en_edges = self.get_bin_nodes(nb, "-3D", nargout=2)
            return nodes_3d, en_edges

        nodes_3d, en_edges = self.get_bin_nodes("-3D", nargout=2)
        return nodes_3d, en_edges

    # ------------------------------------------------------------------
    # Volume & Scale Calculations
    # ------------------------------------------------------------------
    def get_volume_scale(self) -> float:
        """Return conversion factor from axis step product to physical volume."""
        return float(np.prod(self._ulen[:3]))

    def calc_bin_volume(self, *args) -> Union[float, np.ndarray]:
        """Compute bin volume(s) of the 4D grid or supplied axes/nodes."""
        if len(args) == 0:
            steps = (self._img_range[1, :] - self._img_range[0, :]) / self._nbins_all_dims
            return float(np.prod(steps) * self.get_volume_scale())

        arg0 = args[0]
        if isinstance(arg0, (list, tuple)) and len(arg0) == 4:
            axes = [np.asarray(a, dtype=float).ravel() for a in arg0]
        elif isinstance(arg0, np.ndarray) and arg0.ndim == 2 and arg0.shape[0] in (3, 4):
            # 3xN or 4xN grid of nodes
            nodes = arg0
            ax_list: List[np.ndarray] = []
            for i in range(4):
                if i < nodes.shape[0]:
                    u = np.unique(np.round(nodes[i, :], 12))
                    if u.size >= 2:
                        ax_list.append(u)
                    else:
                        ax_list.append(
                            np.linspace(
                                self._img_range[0, i],
                                self._img_range[1, i],
                                int(self._nbins_all_dims[i]) + 1,
                            )
                        )
                else:
                    ax_list.append(
                        np.linspace(
                            self._img_range[0, i],
                            self._img_range[1, i],
                            int(self._nbins_all_dims[i]) + 1,
                        )
                    )
            axes = ax_list
        else:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Invalid input to calc_bin_volume",
            )

        return self._calc_bin_volume_from_axes(axes)

    def _calc_bin_volume_from_axes(
        self, axes: List[np.ndarray]
    ) -> Union[float, np.ndarray]:
        diffs = [np.diff(a) for a in axes]
        all_uniform = all(
            d.size == 1 or np.allclose(d, d[0], rtol=1e-6, atol=1e-12) for d in diffs
        )
        scale = self.get_volume_scale()
        if all_uniform:
            return float(np.prod([d[0] for d in diffs]) * scale)
        g0, g1, g2, g3 = np.meshgrid(
            diffs[0], diffs[1], diffs[2], diffs[3], indexing="ij"
        )
        vol = (g0 * g1 * g2 * g3) * scale
        return vol

    # ------------------------------------------------------------------
    # get_bin_nodes & Private Helpers (calc_bin_nodes_)
    # ------------------------------------------------------------------
    def get_bin_nodes(self, *args, nargout: Optional[int] = None) -> Any:
        """Compute coordinates of bin edges, centers, interpolation grids, or hull."""
        if nargout is None:
            nargout = _infer_nargout(default=1)

        known_flags = {
            "-3D",
            "-halo",
            "-bin_edges",
            "-bin_centre",
            "-bin_center",
            "-dens_interp",
            "-plot_edges",
            "-axes_only",
            "-ngrid",
            "-hull",
        }

        flags = set()
        other_args = []
        for a in args:
            if isinstance(a, str) and a in known_flags:
                if a == "-bin_center":
                    flags.add("-bin_centre")
                else:
                    flags.add(a)
            else:
                other_args.append(a)

        # Compound option expansions matching calc_bin_nodes_.m
        if "-dens_interp" in flags:
            flags.add("-halo")
            flags.add("-bin_centre")
            flags.add("-ngrid")
        if "-plot_edges" in flags:
            flags.add("-bin_edges")
            flags.add("-axes_only")

        if "-bin_edges" in flags and "-bin_centre" in flags:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "Cannot specify both -bin_edges and -bin_centre",
            )

        # Parse numeric binning argument if present
        nbins = self._parse_bin_nodes_inputs(*other_args)

        is_3d = "-3D" in flags
        is_halo = "-halo" in flags
        is_centre = "-bin_centre" in flags
        is_axes_only = "-axes_only" in flags
        is_ngrid = "-ngrid" in flags
        is_hull = "-hull" in flags

        # Build the 4 1D coordinate arrays
        axes_4d = self._build_bin_nodes_axes(nbins, is_halo, is_centre)
        en_axis = axes_4d[3].copy()
        npoints_in_axes = np.array([a.size for a in axes_4d], dtype=int)

        bin_volume = None
        if nargout >= 4:
            if is_centre:
                edge_axes = self._build_bin_nodes_axes(nbins, is_halo, False)
                bin_volume = self.calc_bin_volume(edge_axes)
            else:
                bin_volume = self.calc_bin_volume(axes_4d)

        if is_axes_only:
            if is_3d:
                nodes_out: Any = axes_4d[:3]
            else:
                nodes_out = axes_4d
        elif is_hull:
            nodes_out = self._build_hull_nodes(axes_4d, is_3d)
        elif is_ngrid:
            if is_3d:
                X, Y, Z = np.meshgrid(
                    axes_4d[0], axes_4d[1], axes_4d[2], indexing="ij"
                )
                nodes_out = [X, Y, Z]
            else:
                X, Y, Z, E = np.meshgrid(
                    axes_4d[0], axes_4d[1], axes_4d[2], axes_4d[3], indexing="ij"
                )
                nodes_out = [X, Y, Z, E]
        else:
            if is_3d:
                X, Y, Z = np.meshgrid(
                    axes_4d[0], axes_4d[1], axes_4d[2], indexing="ij"
                )
                nodes_out = np.vstack(
                    [
                        X.ravel(order="F"),
                        Y.ravel(order="F"),
                        Z.ravel(order="F"),
                    ]
                )
            else:
                X, Y, Z, E = np.meshgrid(
                    axes_4d[0], axes_4d[1], axes_4d[2], axes_4d[3], indexing="ij"
                )
                nodes_out = np.vstack(
                    [
                        X.ravel(order="F"),
                        Y.ravel(order="F"),
                        Z.ravel(order="F"),
                        E.ravel(order="F"),
                    ]
                )

        if nargout <= 1:
            return nodes_out
        if nargout == 2:
            return nodes_out, en_axis
        if nargout == 3:
            return nodes_out, en_axis, npoints_in_axes
        return nodes_out, en_axis, npoints_in_axes, bin_volume

    def _parse_bin_nodes_inputs(self, *other_args) -> np.ndarray:
        if len(other_args) == 0:
            return self._nbins_all_dims.copy()
        if len(other_args) > 1:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Unrecognized arguments to get_bin_nodes: {other_args}",
            )
        arg = other_args[0]
        if isinstance(arg, (str, bytes)) or not isinstance(
            arg, (int, float, np.number, list, tuple, np.ndarray)
        ):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "axis binning, if present, should be single numeric value or 4x1 vector of numeric values",
            )
        arr = np.asarray(arg)
        if arr.dtype.kind not in ("i", "u", "f"):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "axis binning, if present, should be single numeric value or 4x1 vector of numeric values",
            )
        arr_flat = arr.ravel()
        if arr_flat.size == 1:
            val = int(arr_flat[0])
            if val < 1:
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    "axis binning must be positive integer(s)",
                )
            nb = self._nbins_all_dims.copy()
            if self.pax.size > 0:
                nb[self.pax - 1] = val
            return nb
        if arr_flat.size == 4:
            if np.any(arr_flat < 1):
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    "axis binning must be positive integer(s)",
                )
            return arr_flat.astype(int)
        raise HoraceError(
            f"HORACE:{self._err_cls_name()}:invalid_argument",
            "axis binning should be 1x4 vector or single value",
        )

    def _build_bin_nodes_axes(
        self, nbins: np.ndarray, is_halo: bool, is_centre: bool
    ) -> List[np.ndarray]:
        axes: List[np.ndarray] = []
        max_r = self._max_img_range
        is_pax = np.zeros(4, dtype=bool)
        if self.pax.size > 0:
            is_pax[self.pax - 1] = True

        for i in range(4):
            r0 = float(self._img_range[0, i])
            r1 = float(self._img_range[1, i])
            if np.isinf(r0) and np.isinf(r1):
                r0, r1 = 0.0, 0.0
            elif np.isinf(r0):
                r0 = r1
            elif np.isinf(r1):
                r1 = r0
            nb = int(nbins[i])
            step = (r1 - r0) / nb
            edges = np.linspace(r0, r1, nb + 1, dtype=float)

            if is_halo and is_pax[i] and step > 0:
                lo_lim = float(max_r[0, i])
                hi_lim = float(max_r[1, i])
                if r0 - step >= lo_lim - EPS_SINGLE:
                    edges = np.insert(edges, 0, r0 - step)
                elif r0 > lo_lim + EPS_SINGLE:
                    edges = np.insert(edges, 0, lo_lim)
                if r1 + step <= hi_lim + EPS_SINGLE:
                    edges = np.append(edges, r1 + step)
                elif r1 < hi_lim - EPS_SINGLE:
                    edges = np.append(edges, hi_lim)

            if is_centre:
                pts = 0.5 * (edges[:-1] + edges[1:])
            else:
                pts = edges
            axes.append(pts)
        return axes

    @staticmethod
    def _build_hull_nodes(axes_4d: List[np.ndarray], is_3d: bool) -> np.ndarray:
        ndim = 3 if is_3d else 4
        ax = axes_4d[:ndim]
        blocks = []
        for fixed_dim in range(ndim):
            if ax[fixed_dim].size >= 2:
                ends = np.array([ax[fixed_dim][0], ax[fixed_dim][-1]], dtype=float)
            else:
                ends = ax[fixed_dim].copy()
            grid_inputs = []
            for d in range(ndim):
                if d == fixed_dim:
                    grid_inputs.append(ends)
                else:
                    # For already visited dimensions, exclude boundary endpoints if desired,
                    # but MATLAB calc_hull Notice: MATLAB's calc_hull uses full axes on each face pair!
                    # Let's check: in MATLAB calc_bin_nodes_.m:
                    # For 3D hull: 6 faces: X(ends)xYxZ, X(2:end-1)xY(ends)xZ, X(2:end-1)xY(2:end-1)xZ(ends)
                    # Wait: in test_AxesBlockBase_methods.m:
                    # n_tot = 2*(nb(1)*nb(2) + nb(1)*nb(3) + nb(2)*nb(3)) for 3D,
                    # and 2*(nb(1)*nb(2)*nb(3) + nb(1)*nb(2)*nb(4) + nb(1)*nb(3)*nb(4) + nb(2)*nb(3)*nb(4)) for 4D!
                    grid_inputs.append(ax[d])
            grids = np.meshgrid(*grid_inputs, indexing="ij")
            block = np.vstack([g.ravel(order="F") for g in grids])
            blocks.append(block)
        return np.hstack(blocks)

    # ------------------------------------------------------------------
    # Binning Pixels (bin_pixels)
    # ------------------------------------------------------------------
    def bin_pixels(
        self,
        coord_transf: np.ndarray,
        *args,
        nargout: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Bin transformed pixel coordinates into the axes grid."""
        if nargout is None:
            nargout = _infer_nargout(default=3)

        opts = list(args)
        force_double = False
        nomex = False
        force_mex = False
        selected_only = False
        test_mex_inputs = False
        nosort = False
        extra_inputs = []
        for a in opts:
            if isinstance(a, str) and a == "-force_double":
                force_double = True
            elif isinstance(a, str) and a == "-nomex":
                nomex = True
            elif isinstance(a, str) and a == "-force_mex":
                force_mex = True
            elif isinstance(a, str) and a == "-selected_only":
                selected_only = True
            elif isinstance(a, str) and a == "-test_mex_inputs":
                test_mex_inputs = True
            elif isinstance(a, str) and a == "-nosort":
                nosort = True
            else:
                extra_inputs.append(a)

        if nomex and force_mex:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "invalid arguments to bin_pixels: '-nomex' and '-force_mex' cannot be used together",
            )

        # Normalize empty 3-accumulator call: bin_pixels(coord, [], [], [])
        if (
            nargout == 1
            and len(extra_inputs) == 3
            and all(x is None or np.asarray(x).size == 0 for x in extra_inputs)
        ):
            extra_inputs = []

        # Determine whether to use MEX code (matching normalize_bin_input_.m)
        cand_is_cell = (
            len(extra_inputs) >= 4 and isinstance(extra_inputs[3], (list, tuple))
        ) or (len(extra_inputs) == 1 and isinstance(extra_inputs[0], (list, tuple)))

        if nargout == 1 and len(extra_inputs) == 0:
            use_mex = force_mex
        elif cand_is_cell:
            use_mex = False
        elif force_mex:
            use_mex = True
        elif nomex:
            use_mex = False
        else:
            use_mex = bool(config_store.instance().get_value("hor_config", "use_mex"))

        if use_mex:
            mode_to_bin = 4 if nargout >= 4 else nargout
            npix_in = extra_inputs[0] if len(extra_inputs) >= 1 else None
            s_in = extra_inputs[1] if len(extra_inputs) >= 2 else None
            e_in = extra_inputs[2] if len(extra_inputs) >= 3 else None
            pix_cand = (
                extra_inputs[3]
                if len(extra_inputs) >= 4
                else (extra_inputs[0] if len(extra_inputs) == 1 else None)
            )
            mex_flags = []
            if force_double:
                mex_flags.append("-force_double")
            if selected_only:
                mex_flags.append("-selected_only")
            if test_mex_inputs:
                mex_flags.append("-test_mex_inputs")
            if nosort:
                mex_flags.append("-nosort")
            return self.bin_pixels_with_mex_code(
                coord_transf,
                mode_to_bin,
                npix_in,
                s_in,
                e_in,
                pix_cand,
                *mex_flags,
                nargout=nargout,
            )

        coords = np.asarray(coord_transf, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, np.newaxis]
        n_dims_in, n_pix = coords.shape

        # Determine active dimensions to check
        r0 = self._img_range[0, :n_dims_in]
        r1 = self._img_range[1, :n_dims_in]
        nb = self._nbins_all_dims[:n_dims_in]

        if n_pix == 0:
            ok = np.zeros(0, dtype=bool)
            ix = np.zeros((0, max(self.dimensions, 1)), dtype=int)
        else:
            ok = np.ones(n_pix, dtype=bool)
            for d in range(n_dims_in):
                c = coords[d, :]
                ok &= (c >= r0[d]) & (c <= r1[d])

            valid_coords = coords[:, ok]
            n_valid = valid_coords.shape[1]
            bin_idx_all = np.ones((4, n_valid), dtype=int)
            for d in range(n_dims_in):
                if nb[d] > 1 and r1[d] > r0[d]:
                    step_d = (r1[d] - r0[d]) / nb[d]
                    idx_d = np.floor((valid_coords[d, :] - r0[d]) / step_d).astype(int) + 1
                    idx_d = np.clip(idx_d, 1, int(nb[d]))
                    bin_idx_all[d, :] = idx_d
                else:
                    bin_idx_all[d, :] = 1

            if self.pax.size > 0:
                ix = bin_idx_all[self.pax - 1, :].T
            else:
                ix = np.ones((n_valid, 1), dtype=int)

        grid_shape = tuple(int(x) for x in self.dims_as_ssize)
        n_bins_total = int(np.prod(grid_shape))

        if n_pix == 0 or ix.shape[0] == 0:
            lin_idx = np.zeros(0, dtype=int)
            npix_flat = np.zeros(n_bins_total, dtype=float)
        else:
            if self.pax.size <= 1:
                lin_idx = ix[:, 0] - 1
            else:
                multi = tuple(ix[:, k] - 1 for k in range(self.pax.size))
                lin_idx = np.ravel_multi_index(multi, grid_shape, order="F")
            npix_flat = np.bincount(lin_idx, minlength=n_bins_total).astype(float)

        npix_grid = npix_flat.reshape(grid_shape, order="F")

        # Handle extra inputs: either (npix_accum, s_accum, e_accum, pix_cand) or (pix_input,)
        if len(extra_inputs) == 0:
            if nargout == 1:
                return npix_grid
            if nargout == 2:
                return npix_grid, ok
            return npix_grid, ok, ix

        if len(extra_inputs) == 1:
            pix_in = extra_inputs[0]
            return self._bin_pixels_with_pix(
                pix_in,
                ok,
                lin_idx,
                ix,
                npix_grid,
                grid_shape,
                n_bins_total,
                force_double,
                nargout,
            )

        if len(extra_inputs) >= 3:
            npix_acc = (
                np.zeros(grid_shape, dtype=float)
                if extra_inputs[0] is None or np.asarray(extra_inputs[0]).size == 0
                else np.asarray(extra_inputs[0], dtype=float).copy()
            )
            s_acc = (
                np.zeros(grid_shape, dtype=float)
                if extra_inputs[1] is None or np.asarray(extra_inputs[1]).size == 0
                else np.asarray(extra_inputs[1], dtype=float).copy()
            )
            e_acc = (
                np.zeros(grid_shape, dtype=float)
                if extra_inputs[2] is None or np.asarray(extra_inputs[2]).size == 0
                else np.asarray(extra_inputs[2], dtype=float).copy()
            )
            pix_in = extra_inputs[3] if len(extra_inputs) >= 4 else None
            return self._bin_pixels_accumulate(
                npix_acc,
                s_acc,
                e_acc,
                pix_in,
                ok,
                lin_idx,
                ix,
                npix_grid,
                grid_shape,
                n_bins_total,
                force_double,
                nargout,
            )

        raise HoraceError(
            f"HORACE:{self._err_cls_name()}:invalid_argument",
            "Invalid number of arguments to bin_pixels",
        )

    def _bin_pixels_with_pix(
        self,
        pix_in: Any,
        ok: np.ndarray,
        lin_idx: np.ndarray,
        ix: np.ndarray,
        npix_grid: np.ndarray,
        grid_shape: Tuple[int, ...],
        n_bins_total: int,
        force_double: bool,
        nargout: int,
    ) -> Any:
        is_pix_obj = isinstance(pix_in, PixelDataBase)
        if is_pix_obj:
            raw_data = pix_in.data
            sig = np.asarray(pix_in.signal, dtype=float).ravel()[ok]
            var = np.asarray(pix_in.variance, dtype=float).ravel()[ok]
            valid_data = raw_data[:, ok]
        else:
            raw_data = np.asarray(pix_in)
            if raw_data.ndim == 2 and raw_data.shape[0] >= 9:
                sig = np.asarray(raw_data[7, ok], dtype=float).ravel()
                var = np.asarray(raw_data[8, ok], dtype=float).ravel()
                valid_data = raw_data[:, ok]
            else:
                # 1D or cell signal array
                sig = np.asarray(raw_data, dtype=float).ravel()[ok]
                var = np.zeros_like(sig)
                valid_data = raw_data.ravel()[ok]

        dtype_out = np.float64 if (force_double or raw_data.dtype == np.float64) else raw_data.dtype

        if lin_idx.size > 0:
            s_sum = np.bincount(lin_idx, weights=sig, minlength=n_bins_total)
            e_sum = np.bincount(lin_idx, weights=var, minlength=n_bins_total)
            npix_flat = np.bincount(lin_idx, minlength=n_bins_total).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                s_mean = np.where(npix_flat > 0, s_sum / npix_flat, 0.0)
                e_mean = np.where(npix_flat > 0, e_sum / (npix_flat ** 2), 0.0)
        else:
            s_mean = np.zeros(n_bins_total, dtype=float)
            e_mean = np.zeros(n_bins_total, dtype=float)

        s_grid = s_mean.reshape(grid_shape, order="F")
        e_grid = e_mean.reshape(grid_shape, order="F")

        # Sort pixels by bin index (stable sort)
        if lin_idx.size > 0:
            sort_order = np.argsort(lin_idx, kind="mergesort")
            uniq_pos = np.unique(lin_idx[sort_order]) + 1
        else:
            sort_order = np.zeros(0, dtype=int)
            uniq_pos = np.zeros(0, dtype=int)

        if is_pix_obj:
            sorted_data = valid_data[:, sort_order].astype(dtype_out, copy=False)
            pix_out: Any = PixelDataMemory(sorted_data)
        elif valid_data.ndim == 2:
            pix_out = valid_data[:, sort_order].astype(dtype_out, copy=False)
        else:
            pix_out = valid_data[sort_order].astype(dtype_out, copy=False)

        outs = (npix_grid, s_grid, e_grid, pix_out, uniq_pos, ok, ix)
        return outs[:nargout]

    def _bin_pixels_accumulate(
        self,
        npix_acc: np.ndarray,
        s_acc: np.ndarray,
        e_acc: np.ndarray,
        pix_in: Any,
        ok: np.ndarray,
        lin_idx: np.ndarray,
        ix: np.ndarray,
        npix_grid: np.ndarray,
        grid_shape: Tuple[int, ...],
        n_bins_total: int,
        force_double: bool,
        nargout: int,
    ) -> Any:
        npix_out = npix_acc.reshape(grid_shape, order="F") + npix_grid

        if isinstance(pix_in, (list, tuple)):
            # Cell array of signal arrays (e.g. magnetic form factor components)
            s_list = []
            e_list = []
            for p_arr in pix_in:
                arr = np.asarray(p_arr, dtype=float).ravel()[ok]
                if lin_idx.size > 0:
                    s_sum = np.bincount(lin_idx, weights=arr, minlength=n_bins_total)
                    e_sum = np.bincount(
                        lin_idx, weights=arr ** 2, minlength=n_bins_total
                    )
                else:
                    s_sum = np.zeros(n_bins_total, dtype=float)
                    e_sum = np.zeros(n_bins_total, dtype=float)
                s_list.append(s_sum.reshape(grid_shape, order="F"))
                e_list.append(e_sum.reshape(grid_shape, order="F"))
            return (npix_out, s_list, e_list)[:nargout]

        is_pix_obj = isinstance(pix_in, PixelDataBase)
        if is_pix_obj:
            raw_data = pix_in.data
            sig = np.asarray(pix_in.signal, dtype=float).ravel()[ok]
            var = np.asarray(pix_in.variance, dtype=float).ravel()[ok]
            valid_data = raw_data[:, ok]
        elif pix_in is not None:
            raw_data = np.asarray(pix_in)
            if raw_data.ndim == 2 and raw_data.shape[0] >= 9:
                sig = np.asarray(raw_data[7, ok], dtype=float).ravel()
                var = np.asarray(raw_data[8, ok], dtype=float).ravel()
                valid_data = raw_data[:, ok]
            else:
                sig = np.asarray(raw_data, dtype=float).ravel()[ok]
                var = sig ** 2
                valid_data = raw_data.ravel()[ok]
        else:
            raw_data = np.zeros((9, 0))
            sig = np.zeros(0)
            var = np.zeros(0)
            valid_data = raw_data

        if lin_idx.size > 0:
            s_sum = np.bincount(lin_idx, weights=sig, minlength=n_bins_total)
            e_sum = np.bincount(lin_idx, weights=var, minlength=n_bins_total)
            sort_order = np.argsort(lin_idx, kind="mergesort")
            uniq_pos = np.unique(lin_idx[sort_order]) + 1
        else:
            s_sum = np.zeros(n_bins_total, dtype=float)
            e_sum = np.zeros(n_bins_total, dtype=float)
            sort_order = np.zeros(0, dtype=int)
            uniq_pos = np.zeros(0, dtype=int)

        s_out = s_acc.reshape(grid_shape, order="F") + s_sum.reshape(
            grid_shape, order="F"
        )
        e_out = e_acc.reshape(grid_shape, order="F") + e_sum.reshape(
            grid_shape, order="F"
        )

        dtype_out = (
            np.float64
            if (force_double or raw_data.dtype == np.float64)
            else raw_data.dtype
        )
        if is_pix_obj:
            pix_ok: Any = PixelDataMemory(
                valid_data[:, sort_order].astype(dtype_out, copy=False)
            )
        elif valid_data.ndim == 2:
            pix_ok = valid_data[:, sort_order].astype(dtype_out, copy=False)
        else:
            pix_ok = valid_data[sort_order].astype(dtype_out, copy=False)

        outs = (npix_out, s_out, e_out, pix_ok, uniq_pos, ok, ix)
        return outs[:nargout]

    def bin_pixels_with_mex_code(
        self,
        coord: Any,
        mode_to_bin: int,
        npix: Any = None,
        s: Any = None,
        err: Any = None,
        pix_cand: Any = None,
        *args,
        nargout: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Bin pixel coordinates using C++/MEX routine ``bin_pixels_c``."""
        _ = (coord, npix, s, err, pix_cand, nargout, kwargs)
        mex_par = self.prepare_mex_code_input(mode_to_bin, *args)
        if "-test_mex_inputs" in args:
            return mex_par
        # TODO(port-mex): Requires C++ MEX routine bin_pixels_c
        raise NotImplementedError(
            f"{self._err_cls_name()}.bin_pixels_with_mex_code (MEX bin_pixels_c) is not yet implemented."
        )

    def prepare_mex_code_input(self, mode_to_bin: int, *args) -> dict:
        """Prepare parameter dictionary for C++/MEX routine ``bin_pixels_c``."""
        force_double = "-force_double" in args
        selected_only = "-selected_only" in args
        test_mex_inputs = "-test_mex_inputs" in args
        nosort = "-nosort" in args
        return {
            "range": self.img_range,
            "nbins_all_dims": self.nbins_all_dims,
            "pax": self.pax,
            "mode": int(mode_to_bin),
            "force_double": bool(force_double),
            "selected_only": bool(selected_only),
            "test_inputs": bool(test_mex_inputs),
            "sort_pixels": not bool(nosort),
            "thread_mode": 0,
            "num_threads": 1,
        }

    # ------------------------------------------------------------------
    # Integration & Rebinning (realign_bin_edges, rebin_data)
    # ------------------------------------------------------------------
    def realign_bin_edges(self, ax_other: "AxesBlockBase") -> "AxesBlockBase":
        """Align ax_other bin edges to match the bin boundaries of self."""
        if not isinstance(ax_other, self.__class__) and not isinstance(
            self, ax_other.__class__
        ):
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Cannot realign axes of different types: {self.__class__.__name__} and {ax_other.__class__.__name__}",
            )

        ax_nbins = ax_other._nbins_all_dims.copy()
        ax_range = ax_other._img_range.copy()
        sbd = np.zeros(4, dtype=bool)

        want_pax = np.zeros(4, dtype=bool)
        if ax_other.pax.size > 0:
            want_pax[ax_other.pax - 1] = True
        this_is_pax = np.zeros(4, dtype=bool)
        if self.pax.size > 0:
            this_is_pax[self.pax - 1] = True

        this_range = self._img_range.copy()
        this_nbins = self._nbins_all_dims.copy()

        for i in range(4):
            origin_range = this_range[:, i].copy()
            origin_nbins = int(this_nbins[i])
            if want_pax[i]:
                if this_is_pax[i]:
                    req_range = ax_range[:, i].copy()
                    req_nbins = int(ax_nbins[i])
                    if req_range[0] >= origin_range[1] or req_range[1] <= origin_range[0]:
                        raise HoraceError(
                            "HORACE:realign_bin_edges:invalid_argument",
                            "Input projection ranges lie outside of original axis edges",
                        )
                    base_step = (origin_range[1] - origin_range[0]) / origin_nbins
                    if (req_range[1] - req_range[0]) < base_step:
                        req_range[1] = req_range[0] + base_step
                    req_step0 = (req_range[1] - req_range[0]) / req_nbins
                    if req_range[1] > origin_range[1]:
                        req_range[1] = origin_range[1]
                    step_scaling_factor = int(
                        np.floor(np.round(req_step0 / base_step, 12) + 0.5)
                    )
                    if step_scaling_factor < 1:
                        req_step = base_step
                    else:
                        req_step = base_step * step_scaling_factor

                    if origin_nbins <= 1:
                        bin_edges = origin_range.copy()
                    else:
                        bin_edges = np.linspace(
                            origin_range[0], origin_range[1], origin_nbins + 1
                        )
                    new_edge_1 = max(req_range[0], origin_range[0])
                    ind = np.where(bin_edges <= new_edge_1)[0][-1]
                    new_edge_1 = float(bin_edges[ind])

                    req_range[0] = new_edge_1
                    nbins = int(np.floor((req_range[1] - req_range[0]) / req_step))
                    test_range = req_range[0] + nbins * req_step
                    if test_range - req_range[1] < -2.0e-8:
                        nbins = nbins + 1
                        req_range[1] = req_range[0] + nbins * req_step
                    else:
                        req_range[1] = test_range
                    ax_range[:, i] = req_range
                    ax_nbins[i] = max(nbins, 1)
                else:
                    ax_nbins[i] = 1
                    ax_range[:, i] = origin_range
                    sbd[i] = True
            else:
                ax_nbins[i] = 1
                sbd[i] = True
                iax_req = ax_range[:, i].copy()
                if iax_req[0] < origin_range[0]:
                    iax_req[0] = origin_range[0]
                if iax_req[1] > origin_range[1]:
                    iax_req[1] = origin_range[1]
                if iax_req[1] < iax_req[0]:
                    raise HoraceError(
                        "HORACE:realign_bin_edges:invalid_argument",
                        "Input integration ranges lie outside of original axis edges",
                    )
                if origin_nbins <= 1:
                    bin_edges = origin_range.copy()
                else:
                    bin_edges = np.linspace(
                        origin_range[0], origin_range[1], origin_nbins + 1
                    )
                min_ind = np.where(bin_edges <= iax_req[0])[0][-1]
                max_ind = np.where(bin_edges >= iax_req[1])[0][0]
                ax_range[:, i] = [bin_edges[min_ind], bin_edges[max_ind]]

        res = copy.deepcopy(ax_other)
        res._do_check_combo_arg = True
        try:
            res._set_img_range(ax_range)
            res._set_nbins_all_dims(ax_nbins)
            res._single_bin_defines_iax = sbd
        finally:
            res._do_check_combo_arg = False
        res.check_combo_arg()
        return res

    def rebin_data(
        self,
        data_in: Sequence[np.ndarray],
        other_ax: "AxesBlockBase",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rebin (s, e, npix) arrays defined on self onto aligned other_ax."""
        s_in = np.asarray(data_in[0], dtype=float)
        e_in = np.asarray(data_in[1], dtype=float)
        npix_in = np.asarray(data_in[2], dtype=float)

        full_shape_self = tuple(int(x) for x in self._nbins_all_dims)
        s_4d = s_in.reshape(full_shape_self, order="F")
        e_4d = e_in.reshape(full_shape_self, order="F")
        npix_4d = npix_in.reshape(full_shape_self, order="F")

        # Weight signals by npix (and variances by npix^2) before summing
        sig_weighted = s_4d * npix_4d
        var_weighted = e_4d * (npix_4d ** 2)

        for i in range(4):
            r0_s, r1_s = float(self._img_range[0, i]), float(self._img_range[1, i])
            nb_s = int(self._nbins_all_dims[i])
            if nb_s <= 1:
                if np.isinf(r0_s) and np.isinf(r1_s):
                    centers_s = np.array([0.0], dtype=float)
                else:
                    centers_s = np.array([0.5 * (r0_s + r1_s)], dtype=float)
            else:
                edges_s = np.linspace(r0_s, r1_s, nb_s + 1)
                centers_s = 0.5 * (edges_s[:-1] + edges_s[1:])

            r0_o, r1_o = float(other_ax._img_range[0, i]), float(other_ax._img_range[1, i])
            nb_o = int(other_ax._nbins_all_dims[i])
            if nb_o <= 1:
                edges_o = np.array([r0_o, r1_o], dtype=float)
            else:
                edges_o = np.linspace(r0_o, r1_o, nb_o + 1)

            # Map each source bin center into target bin [0 .. nb_o - 1]
            new_shape = list(sig_weighted.shape)
            new_shape[i] = nb_o
            s_next = np.zeros(new_shape, dtype=float)
            e_next = np.zeros(new_shape, dtype=float)
            n_next = np.zeros(new_shape, dtype=float)

            for b_o in range(nb_o):
                lo, hi = edges_o[b_o], edges_o[b_o + 1]
                mask = (centers_s >= lo - 1e-9) & (centers_s <= hi + 1e-9)
                if np.any(mask):
                    sl_src = [slice(None)] * 4
                    sl_src[i] = mask
                    sl_dst = [slice(None)] * 4
                    sl_dst[i] = b_o
                    s_next[tuple(sl_dst)] = np.sum(
                        sig_weighted[tuple(sl_src)], axis=i
                    )
                    e_next[tuple(sl_dst)] = np.sum(
                        var_weighted[tuple(sl_src)], axis=i
                    )
                    n_next[tuple(sl_dst)] = np.sum(npix_4d[tuple(sl_src)], axis=i)

            sig_weighted = s_next
            var_weighted = e_next
            npix_4d = n_next

        with np.errstate(divide="ignore", invalid="ignore"):
            s_out_4d = np.where(npix_4d > 0, sig_weighted / npix_4d, 0.0)
            e_out_4d = np.where(npix_4d > 0, var_weighted / (npix_4d ** 2), 0.0)

        out_shape = tuple(int(x) for x in other_ax.dims_as_ssize)
        return (
            s_out_4d.reshape(out_shape, order="F"),
            e_out_4d.reshape(out_shape, order="F"),
            npix_4d.reshape(out_shape, order="F"),
        )

    # ------------------------------------------------------------------
    # Density & Interpolation (get_density, interpolate_data)
    # ------------------------------------------------------------------
    def get_density(
        self, data_in: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert (s, e, npix) arrays into signal/error/npix density arrays with 1-cell halo."""
        if not isinstance(data_in, (list, tuple)) or len(data_in) != 3:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "get_density requires a 3-element cell/list of (s, e, npix) arrays",
            )
        expected_shape = tuple(int(x) for x in self.dims_as_ssize)
        expected_size = int(np.prod(expected_shape))
        for arr in data_in:
            a = np.asarray(arr)
            if a.size != expected_size or (
                a.shape != expected_shape and a.squeeze().shape != np.zeros(expected_shape).squeeze().shape
            ):
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    f"Input data shape {a.shape} does not match axes dimensions {expected_shape}",
                )

        _, _, npts_halo, bin_vol = self.get_bin_nodes("-dens_interp", nargout=4)
        edge_axes_no_halo = self._build_bin_nodes_axes(
            self._nbins_all_dims, is_halo=False, is_centre=False
        )
        vol_interior = self.calc_bin_volume(edge_axes_no_halo)

        full_shape = tuple(int(x) for x in self._nbins_all_dims)
        halo_shape = tuple(int(x) for x in npts_halo)

        # Determine padding on each axis
        max_r = self._max_img_range
        is_pax = np.zeros(4, dtype=bool)
        if self.pax.size > 0:
            is_pax[self.pax - 1] = True

        pad_widths = []
        for i in range(4):
            r0 = float(self._img_range[0, i])
            r1 = float(self._img_range[1, i])
            nb = int(self._nbins_all_dims[i])
            step = (r1 - r0) / nb
            lo_pad = 0
            hi_pad = 0
            if is_pax[i] and step > 0:
                if r0 > float(max_r[0, i]) + EPS_SINGLE:
                    lo_pad = 1
                if r1 < float(max_r[1, i]) - EPS_SINGLE:
                    hi_pad = 1
            pad_widths.append((lo_pad, hi_pad))

        out_arrays = []
        for idx_arr, arr in enumerate(data_in):
            a_4d = np.asarray(arr, dtype=float).reshape(full_shape, order="F")
            if idx_arr == 1:
                # Variance density scales with 1 / vol^2
                dens = a_4d / (vol_interior ** 2)
            else:
                dens = a_4d / vol_interior
            padded = np.pad(dens, pad_widths, mode="constant", constant_values=0.0)
            out_arrays.append(padded.reshape(halo_shape, order="F"))

        return out_arrays[0], out_arrays[1], out_arrays[2]

    def interpolate_data(
        self,
        source_axes: "AxesBlockBase",
        data_in: Sequence[np.ndarray],
        proj_ref_nodes: Optional[Sequence[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate (s, e, npix) from source_axes onto self using density interpolation."""
        if not isinstance(data_in, (list, tuple)) or len(data_in) != 3:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                "interpolate_data requires a 3-element cell/list of (s, e, npix) arrays",
            )

        s_dens, e_dens, n_dens = source_axes.get_density(data_in)
        src_nodes_grid = source_axes.get_bin_nodes("-dens_interp")
        src_axes_1d = [
            src_nodes_grid[0][:, 0, 0, 0],
            src_nodes_grid[1][0, :, 0, 0],
            src_nodes_grid[2][0, 0, :, 0],
            src_nodes_grid[3][0, 0, 0, :],
        ]

        if proj_ref_nodes is None:
            targ_pts, _, _, targ_vol = self.get_bin_nodes("-bin_centre", nargout=4)
        else:
            if not isinstance(proj_ref_nodes, (list, tuple)) or len(proj_ref_nodes) != 4:
                raise HoraceError(
                    f"HORACE:{self._err_cls_name()}:invalid_argument",
                    "Reference nodes must be a 4-element cell array of grid arrays",
                )
            targ_pts = np.vstack([ np.asarray(g, dtype=float).ravel(order="F") for g in proj_ref_nodes])
            _, _, _, targ_vol = self.get_bin_nodes("-bin_centre", nargout=4)

        expected_npts = int(np.prod(self._nbins_all_dims))
        if targ_pts.shape[1] != expected_npts:
            raise HoraceError(
                f"HORACE:{self._err_cls_name()}:invalid_argument",
                f"Number of target points ({targ_pts.shape[1]}) does not match target bins ({expected_npts})",
            )

        # Build 4D interpolators (or handle singleton dimensions)
        active_dims = [i for i in range(4) if src_axes_1d[i].size > 1]
        pts_query = targ_pts.T

        def _eval_interp(field_4d: np.ndarray) -> np.ndarray:
            if len(active_dims) == 4:
                interp = RegularGridInterpolator(
                    tuple(src_axes_1d),
                    field_4d,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                return interp(pts_query)
            elif len(active_dims) > 0:
                sl = [0] * 4
                for d in active_dims:
                    sl[d] = slice(None)
                sub_field = field_4d[tuple(sl)]
                sub_axes = tuple(src_axes_1d[d] for d in active_dims)
                interp = RegularGridInterpolator(
                    sub_axes,
                    sub_field,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                res = interp(pts_query[:, active_dims])
                # Check bounds on singleton dimensions
                for d in range(4):
                    if d not in active_dims:
                        r0, r1 = float(source_axes._img_range[0, d]), float(source_axes._img_range[1, d])
                        mask_out = (pts_query[:, d] < r0 - EPS_SINGLE) | (pts_query[:, d] > r1 + EPS_SINGLE)
                        res[mask_out] = 0.0
                return res
            else:
                return np.full(pts_query.shape[0], float(field_4d.ravel()[0]))

        s_interp = _eval_interp(s_dens)
        e_interp = _eval_interp(e_dens)
        n_interp = _eval_interp(n_dens)

        full_targ_shape = tuple(int(x) for x in self._nbins_all_dims)
        s_4d = s_interp.reshape(full_targ_shape, order="F") * targ_vol
        e_4d = e_interp.reshape(full_targ_shape, order="F") * (targ_vol ** 2)
        n_4d = np.floor(n_interp.reshape(full_targ_shape, order="F") * targ_vol)

        zero_mask = n_4d < 1
        s_4d[zero_mask] = 0.0
        e_4d[zero_mask] = 0.0
        n_4d[zero_mask] = 0.0

        out_shape = tuple(int(x) for x in self.dims_as_ssize)
        return (
            s_4d.reshape(out_shape, order="F"),
            e_4d.reshape(out_shape, order="F"),
            n_4d.reshape(out_shape, order="F"),
        )

    # ------------------------------------------------------------------
    # Target Range Transformation (get_targ_range)
    # ------------------------------------------------------------------
    def get_targ_range(
        self,
        source_proj: Any,
        targ_proj: Any = None,
        *args: Any,
    ) -> Any:
        """Compute range or default pbin of this axes block in target projection coordinate system.
        Ported from @AxesBlockBase/get_targ_range.m.
        """
        return_binning = False
        range_requested = np.ones(4, dtype=bool)

        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, str):
                if arg.startswith("-bin"):
                    return_binning = True
                else:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Unrecognized option '{arg}' to get_targ_range",
                    )
            else:
                arr_req = np.asarray(arg, dtype=bool).ravel()
                if arr_req.size == 4:
                    range_requested = arr_req
                else:
                    raise HoraceError(
                        f"HORACE:{self._err_cls_name()}:invalid_argument",
                        f"Unrecognized argument '{arg}' to get_targ_range",
                    )

        if targ_proj is None or (
            isinstance(targ_proj, (list, tuple, np.ndarray)) and len(targ_proj) == 0
        ):
            targ_proj = source_proj

        if source_proj == targ_proj:
            targ_range = self.img_range.copy()
        else:
            if not targ_proj.alatt_defined and source_proj.alatt_defined:
                targ_proj.alatt = source_proj.alatt
            if not targ_proj.angdeg_defined and source_proj.angdeg_defined:
                targ_proj.angdeg = source_proj.angdeg
            source_proj.targ_proj = targ_proj
            targ_proj.targ_proj = source_proj

            current_range = self.img_range.copy()
            off_diff = source_proj.offset[:3] - targ_proj.offset[:3]
            is_zero_off = float(np.max(np.abs(off_diff))) < 4.0 * float(
                np.finfo(np.float32).eps
            )
            if is_zero_off:
                img_targ_center = np.zeros((3, 1), dtype=float)
            else:
                img_targ_center = source_proj.transform_hkl_to_img(
                    np.asarray(targ_proj.offset[:3], dtype=float).reshape(3, 1)
                )

            # in_range(current_range[:, :3], img_targ_center)
            coord = img_targ_center.ravel()[:3]
            r_min = current_range[0, :3]
            r_max = current_range[1, :3]
            outside = (coord < r_min) | (coord > r_max)
            equal = (coord == r_min) | (coord == r_max)
            is_in = 1
            if np.any(outside):
                is_in = -1
            if np.any(equal):
                is_in = 0

            from .line_proj_base import LineProjBase
            from .curve_proj_base import CurveProjBase

            if isinstance(source_proj, type(targ_proj)) and (
                is_zero_off or isinstance(source_proj, LineProjBase)
            ):
                full_range = expand_box(current_range[0, :], current_range[1, :])
                full_targ_range = source_proj.from_this_to_targ_coord(full_range)
                targ_range = np.vstack(
                    [np.min(full_targ_range, axis=1), np.max(full_targ_range, axis=1)]
                )
            else:
                # search_for_range
                range_defined = np.zeros((2, 4), dtype=bool)
                out_range = np.vstack([np.full(4, -np.inf), np.full(4, np.inf)])
                if is_in == -1:
                    nbins_per_dim = np.ones(4, dtype=int)
                    max_out_range = out_range.copy()
                elif is_in == 0:
                    nbins_per_dim = np.full(4, 10, dtype=int)
                    max_out_range = out_range.copy()
                    if isinstance(targ_proj, CurveProjBase):
                        max_out_range[0, 0] = 0.0
                        range_defined[0, 0] = True
                else:  # is_in == 1
                    nbins_per_dim = np.full(4, 10, dtype=int)
                    axes_name = getattr(targ_proj, "axes_name", None)
                    if axes_name is None and hasattr(targ_proj, "_get_axes_class"):
                        t_cls = targ_proj._get_axes_class()
                    else:
                        from .line_axes import line_axes
                        from .cylinder_axes import cylinder_axes
                        from .sphere_axes import sphere_axes

                        t_cls = {
                            "line_axes": line_axes,
                            "cylinder_axes": cylinder_axes,
                            "sphere_axes": sphere_axes,
                        }.get(axes_name, line_axes)
                    taxes = t_cls(type=targ_proj.type)
                    max_out_range = taxes._max_img_range.copy()
                    range_defined = ~np.isinf(max_out_range)

                do_3d = bool(getattr(source_proj, "do_3D_transformation", True))
                if do_3d:
                    img_r_work = current_range[:, :3]
                    nbins_work = nbins_per_dim[:3].copy()
                    out_range[:, 3] = current_range[:, 3]
                    range_defined_work = range_defined[:, :3]
                    max_out_work = max_out_range[:, :3]
                    req_work = range_requested[:3]
                else:
                    img_r_work = current_range.copy()
                    nbins_work = nbins_per_dim.copy()
                    range_defined_work = range_defined
                    max_out_work = max_out_range
                    req_work = range_requested

                range_known = np.all(range_defined_work, axis=0)
                if not np.any(req_work[~range_known]):
                    targ_range = out_range
                else:
                    def _build_hull_nd(rng: np.ndarray, nb: np.ndarray) -> np.ndarray:
                        nd = rng.shape[1]
                        ax = [
                            np.array([rng[1, d]], dtype=float)
                            if int(nb[d]) == 1
                            else np.linspace(rng[0, d], rng[1, d], int(nb[d]))
                            for d in range(nd)
                        ]
                        ax_rng = [np.array([rng[0, d], rng[1, d]]) for d in range(nd)]
                        blocks = []
                        for face_d in range(nd):
                            grids_1d = [
                                ax_rng[d] if d == face_d else ax[d] for d in range(nd)
                            ]
                            mg = np.meshgrid(*grids_1d, indexing="ij")
                            block = np.vstack([g.ravel(order="F") for g in mg])
                            blocks.append(block)
                        return np.hstack(blocks)

                    def _transf_range(nb: np.ndarray) -> np.ndarray:
                        shell_pts = _build_hull_nd(img_r_work, nb)
                        tr_shell = source_proj.from_this_to_targ_coord(shell_pts)
                        loc_r = np.vstack(
                            [np.min(tr_shell, axis=1), np.max(tr_shell, axis=1)]
                        )
                        if np.any(range_defined_work):
                            loc_r[range_defined_work] = max_out_work[range_defined_work]
                        return loc_r

                    loc_range = _transf_range(nbins_work)
                    difr = 1.0
                    ic = 1
                    while difr > 1e-3 and ic <= 5:
                        nbins_work = 2 * nbins_work
                        new_loc_range = _transf_range(nbins_work)
                        min_difr = float(np.max(-(new_loc_range[0, :] - loc_range[0, :])))
                        max_difr = float(np.max(new_loc_range[1, :] - loc_range[1, :]))
                        difr = max(min_difr, max_difr)
                        loc_range = new_loc_range
                        ic += 1

                    if do_3d:
                        out_range[:, :3] = loc_range
                    else:
                        out_range = loc_range
                    targ_range = out_range

        if not np.all(range_requested):
            targ_range[0, ~range_requested] = -np.inf
            targ_range[1, ~range_requested] = np.inf

        if not return_binning:
            return targ_range

        nsteps = self._nbins_all_dims
        binning = []
        for i in range(4):
            min_r = float(targ_range[0, i])
            max_r = float(targ_range[1, i])
            ns = int(nsteps[i])
            if np.isinf(min_r) or np.isinf(max_r):
                binning.append(np.array([], dtype=float))
            elif ns == 1:
                binning.append(np.array([min_r, max_r], dtype=float))
            else:
                step = (max_r - min_r) / ns
                binning.append(
                    np.array(
                        [min_r + 0.5 * step, step, max_r - 0.5 * step], dtype=float
                    )
                )
        return binning

    # ------------------------------------------------------------------
    # Plot Titles (data_plot_titles)
    # ------------------------------------------------------------------
    def data_plot_titles(
        self, data_obj: Any = None
    ) -> Tuple[str, List[str], List[str]]:
        """Generate (title_main, title_pax, title_iax) strings for plotting."""
        return self._format_plot_titles()

    def _format_plot_titles(
        self, axis_unit_labels: Optional[List[str]] = None
    ) -> Tuple[str, List[str], List[str]]:
        if axis_unit_labels is None:
            axis_unit_labels = [
                f"in {self._ulen[i]:.3g} \\AA^{{-1}}" if i < 3 else "meV"
                for i in range(4)
            ]

        title_pax: List[str] = []
        pax_ordered = self.pax[self.dax - 1] if self.pax.size > 0 else np.array([], dtype=int)
        for ax_num in pax_ordered:
            idx = int(ax_num) - 1
            off = float(self._offset[idx])
            lbl = self._label[idx]
            u_str = axis_unit_labels[idx]
            if abs(off) > EPS_SINGLE:
                if off > 0:
                    prefix = f"{off:.3g} + "
                else:
                    prefix = f"{off:.3g} + "
            else:
                prefix = ""
            if u_str:
                title_pax.append(f"{prefix}{lbl} ({u_str})")
            else:
                title_pax.append(f"{prefix}{lbl}")

        title_iax: List[str] = []
        for idx_i, ax_num in enumerate(self.iax):
            idx = int(ax_num) - 1
            off = float(self._offset[idx])
            r0 = float(self._img_range[0, idx]) + off
            r1 = float(self._img_range[1, idx]) + off
            lbl = self._label[idx]
            title_iax.append(f"{r0:.3g} \\leq {lbl} \\leq {r1:.3g}")

        main_lines = []
        if self._filename:
            main_lines.append(self._filename)
        if self._title:
            main_lines.append(self._title)
        if title_iax:
            main_lines.append(", ".join(title_iax))
        title_main = "\n".join(main_lines) if main_lines else ""

        return title_main, title_pax, title_iax

    # ------------------------------------------------------------------
    # Legacy Deserialization Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_from_old_data(inputs: dict) -> "AxesBlockBase":
        from pyhorace.coord_transf.line_axes import line_axes

        return line_axes.get_from_old_data(inputs)
