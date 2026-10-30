"""Abstract base class for pixel data (translated from ``@PixelDataBase``)."""

from __future__ import annotations

import abc
import copy
import math
import os
import weakref
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.configurations.hor_config import hor_config
from pyhorace.core.data_op_interface import (
    DataOpArray,
    data_op_interface,
    mldivide,
    minus,
    mpower,
    mrdivide,
    mtimes,
    plus,
)
from pyhorace.core.serializable import Serializable, compare_values, register_serializable
from pyhorace.data.pixels.pix_data import pix_data
from pyhorace.data.pixels.pix_metadata import pix_metadata
from pyhorace.data.sigvar import sigvar


class _PixelFieldView(np.ndarray):
    """NumPy array subclass that writes in-place slice assignments back to ``PixelDataBase``."""

    _owner_ref: Optional[weakref.ReferenceType]
    _field_name: Optional[str]

    def __new__(
        cls,
        input_array: np.ndarray,
        owner: Optional["PixelDataBase"] = None,
        field_name: Optional[str] = None,
    ) -> "_PixelFieldView":
        arr = np.asarray(input_array)
        if not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64)
        obj = arr.view(cls)
        obj._owner_ref = weakref.ref(owner) if owner is not None else None
        obj._field_name = field_name
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self._owner_ref = None
        self._field_name = None

    def _sync_to_owner(self) -> None:
        if self._owner_ref is not None and self._field_name is not None:
            owner = self._owner_ref()
            if owner is not None:
                if self._field_name == "all":
                    owner.data = np.asarray(self, dtype=np.float64)
                else:
                    owner.set_prop(self._field_name, np.asarray(self, dtype=np.float64))

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        raw_inputs = [np.asarray(x) if isinstance(x, _PixelFieldView) else x for x in inputs]
        if "out" in kwargs and kwargs["out"] is not None:
            out_tuple = kwargs["out"]
            raw_out = tuple(
                np.asarray(o) if isinstance(o, _PixelFieldView) else o for o in out_tuple
            )
            kwargs = dict(kwargs, out=raw_out)
            res = getattr(ufunc, method)(*raw_inputs, **kwargs)
            for o in out_tuple:
                if isinstance(o, _PixelFieldView):
                    o._sync_to_owner()
            return res
        return getattr(ufunc, method)(*raw_inputs, **kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        # Support 1D indexing on 1xN views (e.g. pix.run_idx[1:4] = [10, 20, 30])
        if (
            self.ndim == 2
            and self.shape[0] == 1
            and not isinstance(key, tuple)
            and not (isinstance(key, np.ndarray) and key.dtype == bool and key.ndim == 2)
        ):
            super().__setitem__((0, key), value)
        else:
            super().__setitem__(key, value)

        self._sync_to_owner()


class PixelDataBase(Serializable, data_op_interface, abc.ABC):
    """Abstract base class for memory-backed and file-backed pixel datasets."""

    DATA_POINT_SIZE: ClassVar[int] = 4
    DEFAULT_NUM_PIX_FIELDS: ClassVar[int] = 9
    DEFAULT_PAGE_SIZE: ClassVar[int] = 10000000
    EMPTY_RANGE: ClassVar[np.ndarray] = np.vstack(
        [np.full((1, 9), np.inf, dtype=np.float64), np.full((1, 9), -np.inf, dtype=np.float64)]
    )
    EMPTY_RANGE_: ClassVar[np.ndarray] = EMPTY_RANGE[:, :4].copy()
    EMPTY_PIXELS: ClassVar[np.ndarray] = np.empty((9, 0), dtype=np.float64)
    COLS: ClassVar[Tuple[str, ...]] = (
        "u1",
        "u2",
        "u3",
        "dE",
        "run_idx",
        "detector_idx",
        "energy_idx",
        "signal",
        "variance",
    )
    FIELD_INDEX_MAP: ClassVar[Dict[str, List[int]]] = {
        "u1": [1],
        "u2": [2],
        "u3": [3],
        "dE": [4],
        "coordinates": [1, 2, 3, 4],
        "q_coordinates": [1, 2, 3],
        "run_idx": [5],
        "detector_idx": [6],
        "energy_idx": [7],
        "signal": [8],
        "variance": [9],
        "sig_var": [8, 9],
        "all_indexes": [5, 6, 7],
        "all_experiment": [5, 6, 7, 8, 9],
        "all": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    _fields_to_save: ClassVar[Tuple[str, ...]] = ("data_wrap", "metadata")

    def __init__(self) -> None:
        Serializable.__init__(self)
        self._full_filename: str = ""
        self._page_num: int = 1
        self._data_range: np.ndarray = self.EMPTY_RANGE.copy()
        self._is_corrected: bool = False
        self._alignment_matr: np.ndarray = np.eye(3, dtype=np.float64)
        self._old_file_format: bool = False
        self._unique_run_id: np.ndarray = np.array([], dtype=np.float64)
        self._keep_precision: bool = False
        self._pix_page_chunks: List[np.ndarray] = []
        self._pix_page_idx_start: np.ndarray = np.array([], dtype=np.intp)
        self._base_page_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Static factory & utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def create(*args: Any, **kwargs: Any) -> "PixelDataBase":
        """Factory method creating :class:`PixelDataMemory` or :class:`PixelDataFileBacked`."""
        from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
        from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

        pos_args = list(args)
        filebacked: Optional[bool] = kwargs.pop(
            "filebacked", kwargs.pop("file_backed", None)
        )
        norange: bool = bool(kwargs.pop("norange", False))

        # Parse trailing MATLAB-style key-value or flag arguments
        cleaned_args: List[Any] = []
        i = 0
        while i < len(pos_args):
            arg = pos_args[i]
            if isinstance(arg, str):
                low = arg.lower().lstrip("-")
                if low in ("filebacked", "file_backed"):
                    if i + 1 < len(pos_args) and isinstance(pos_args[i + 1], (bool, np.bool_)):
                        filebacked = bool(pos_args[i + 1])
                        i += 2
                        continue
                    filebacked = True
                    i += 1
                    continue
                if low == "norange":
                    if i + 1 < len(pos_args) and isinstance(pos_args[i + 1], (bool, np.bool_)):
                        norange = bool(pos_args[i + 1])
                        i += 2
                        continue
                    norange = True
                    i += 1
                    continue
            cleaned_args.append(arg)
            i += 1

        if len(cleaned_args) == 0:
            if filebacked:
                return PixelDataFileBacked()
            return PixelDataMemory()

        init_arg = cleaned_args[0]
        rest_args = cleaned_args[1:]

        if isinstance(init_arg, PixelDataBase):
            if filebacked is None:
                filebacked = init_arg.is_filebacked
            if filebacked:
                return PixelDataFileBacked(init_arg, *rest_args)
            return PixelDataMemory(init_arg, *rest_args)

        if filebacked is None:
            # Determine number of pixels to decide whether to use filebacked
            num_pix = 0
            if isinstance(init_arg, (int, np.integer)) and not isinstance(init_arg, bool):
                num_pix = int(init_arg)
            elif isinstance(init_arg, np.ndarray):
                if init_arg.ndim == 0 or (init_arg.size == 1 and init_arg.ndim <= 2):
                    num_pix = int(init_arg.item())
                elif init_arg.ndim == 2:
                    num_pix = int(init_arg.shape[1])
            elif isinstance(init_arg, (str, os.PathLike)):
                tmp_fb = PixelDataFileBacked(init_arg, *rest_args)
                if PixelDataBase.do_filebacked(tmp_fb.num_pixels):
                    return tmp_fb
                return PixelDataMemory(tmp_fb)
            elif hasattr(init_arg, "npix") or hasattr(init_arg, "num_pixels"):
                num_pix = int(getattr(init_arg, "npix", getattr(init_arg, "num_pixels", 0)) or 0)
            filebacked = PixelDataBase.do_filebacked(num_pix)

        if filebacked:
            obj = PixelDataFileBacked(init_arg, *rest_args)
        else:
            obj = PixelDataMemory(init_arg, *rest_args)

        if norange:
            obj.invalidate_range("all")
        return obj

    @staticmethod
    def cat(*args: Any, **kwargs: Any) -> "PixelDataBase":
        """Concatenate multiple ``PixelDataBase`` objects along the pixel dimension."""
        from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
        from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

        force_fb: Optional[bool] = kwargs.get("filebacked", None)
        objs: List[PixelDataBase] = []
        for a in args:
            if isinstance(a, (list, tuple, DataOpArray)):
                objs.extend(a)
            elif isinstance(a, np.ndarray) and a.dtype == object:
                objs.extend(a.ravel(order="F"))
            elif isinstance(a, str) and a.lower().lstrip("-") == "filebacked":
                force_fb = True
            elif isinstance(a, (bool, np.bool_)):
                force_fb = bool(a)
            else:
                objs.append(a)

        if len(objs) == 0:
            return PixelDataMemory()

        for o in objs:
            if not isinstance(o, PixelDataBase):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Can only concatenate PixelDataBase objects, got {type(o).__name__}"
                )

        total_pix = sum(o.num_pixels for o in objs)
        any_fb = any(o.is_filebacked for o in objs)
        if force_fb is None:
            use_fb = any_fb or PixelDataBase.do_filebacked(total_pix)
        else:
            use_fb = bool(force_fb)

        arrays: List[np.ndarray] = []
        comb_range = PixelDataBase.EMPTY_RANGE.copy()
        for o in objs:
            if o.num_pixels == 0:
                continue
            if o.is_filebacked:
                saved_pg = o.page_num
                for pg in range(1, o.num_pages + 1):
                    o.page_num = pg
                    arrays.append(np.asarray(o.data, dtype=np.float64))
                o.page_num = saved_pg
            else:
                arrays.append(np.asarray(o.data, dtype=np.float64))
            if o.is_range_valid("all"):
                comb_range[0, :] = np.minimum(comb_range[0, :], o.data_range[0, :])
                comb_range[1, :] = np.maximum(comb_range[1, :], o.data_range[1, :])

        if len(arrays) == 0:
            combined = np.empty((9, 0), dtype=np.float64)
        else:
            combined = np.hstack(arrays)

        if use_fb:
            res: PixelDataBase = PixelDataFileBacked(combined)
        else:
            res = PixelDataMemory(combined)
        return res

    @staticmethod
    def do_filebacked(num_pixels: int, mem_chunk_size: Optional[int] = None) -> bool:
        """Return True if ``num_pixels`` exceeds ``mem_chunk_size``."""
        if mem_chunk_size is None:
            mem_chunk_size = int(hor_config().mem_chunk_size)
        return bool(int(num_pixels) > int(mem_chunk_size))

    @staticmethod
    def field_index(fld_names: Union[str, Sequence[str], int, Sequence[int]]) -> np.ndarray:
        """Return 1-based column indices in the 9-row pixel array for ``fld_names``."""
        if isinstance(fld_names, (int, np.integer)):
            idx = int(fld_names)
            if idx < 1 or idx > 9:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Numeric field index out of range 1..9: {idx}"
                )
            return np.array([idx], dtype=np.intp)

        if isinstance(fld_names, np.ndarray) and np.issubdtype(fld_names.dtype, np.integer):
            arr = fld_names.ravel().astype(np.intp)
            if np.any((arr < 1) | (arr > 9)):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Numeric field indices out of range 1..9: {arr}"
                )
            return arr

        if isinstance(fld_names, str):
            if fld_names not in PixelDataBase.FIELD_INDEX_MAP:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Unknown pixel field name: {fld_names!r}"
                )
            return np.array(PixelDataBase.FIELD_INDEX_MAP[fld_names], dtype=np.intp)

        if isinstance(fld_names, (list, tuple, np.ndarray)):
            indices: List[int] = []
            for item in fld_names:
                if isinstance(item, (int, np.integer)):
                    indices.append(int(item))
                elif isinstance(item, str):
                    if item not in PixelDataBase.FIELD_INDEX_MAP:
                        raise HoraceError(
                            f"HORACE:PixelDataBase:invalid_argument - "
                            f"Unknown pixel field name: {item!r}"
                        )
                    indices.extend(PixelDataBase.FIELD_INDEX_MAP[item])
                else:
                    raise HoraceError(
                        f"HORACE:PixelDataBase:invalid_argument - "
                        f"Invalid field descriptor: {item!r}"
                    )
            return np.array(indices, dtype=np.intp)

        raise HoraceError(
            f"HORACE:PixelDataBase:invalid_argument - "
            f"Invalid field name specification: {fld_names!r}"
        )

    @staticmethod
    def pix_minmax_ranges(
        pix_data_arr: np.ndarray, prev_range: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute ``(2, k)`` min/max range array for ``(k, N)`` ``pix_data_arr``."""
        arr = np.asarray(pix_data_arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        n_fields = arr.shape[0]
        if arr.shape[1] == 0:
            if prev_range is not None:
                return np.asarray(prev_range, dtype=np.float64).copy()
            return np.vstack(
                [
                    np.full((1, n_fields), np.inf, dtype=np.float64),
                    np.full((1, n_fields), -np.inf, dtype=np.float64),
                ]
            )
        cur_min = np.min(arr, axis=1)
        cur_max = np.max(arr, axis=1)
        loc_range = np.vstack([cur_min, cur_max])
        if prev_range is not None:
            pr = np.asarray(prev_range, dtype=np.float64)
            loc_range[0, :] = np.minimum(loc_range[0, :], pr[0, :])
            loc_range[1, :] = np.maximum(loc_range[1, :], pr[1, :])
        return loc_range

    @staticmethod
    def get_memmap_format(num_pixels: int, tail: int = 0) -> List[Tuple[Any, ...]]:
        """Return descriptor tuple list representing the binary layout of pixels."""
        fmt: List[Tuple[Any, ...]] = [
            ("single", (PixelDataBase.DEFAULT_NUM_PIX_FIELDS, int(num_pixels)), "data")
        ]
        if int(tail) > 0:
            fmt.append(("uint8", (int(tail), 1), "tail"))
        return fmt

    @staticmethod
    def build_tmp_file_name(fn_base: str = "in_mem", filepath: str = "") -> str:
        """Build a temporary filename with ``.tmp_<uuid>`` extension."""
        import tempfile
        import uuid

        fp = os.path.dirname(fn_base)
        base = os.path.basename(fn_base)
        stem, _ = os.path.splitext(base)
        if not stem:
            stem = "in_mem"
        if not filepath:
            if fp:
                filepath = fp
            else:
                try:
                    filepath = str(hor_config().working_directory or tempfile.gettempdir())
                except Exception:
                    filepath = tempfile.gettempdir()
        return os.path.join(filepath, f"{stem}.tmp_{uuid.uuid4().hex[:12]}")

    @staticmethod
    def build_op_filename(original_fn: str = "", target_fn: str = "") -> Tuple[str, bool]:
        """Construct an operation target filename and ``move_to_orig`` flag."""
        original_fn = str(original_fn or "")
        target_fn = str(target_fn or "")
        if not target_fn:
            fn_base = original_fn if original_fn else "in_mem"
            move_to_orig = False
            filename = PixelDataBase.build_tmp_file_name(fn_base)
            return filename, move_to_orig

        filepath = os.path.dirname(target_fn)
        base = os.path.basename(target_fn)
        fn_base, fext = os.path.splitext(base)
        if not filepath:
            filepath = os.getcwd()
        fext_is_tmp = fext.startswith(".tmp")
        if original_fn == target_fn:
            move_to_orig = True
            if fext_is_tmp:
                filename = target_fn
            else:
                filename = PixelDataBase.build_tmp_file_name(fn_base, filepath)
        else:
            move_to_orig = False
            filename = target_fn
        return filename, move_to_orig

    # ------------------------------------------------------------------
    # Abstract / Protected interface for subclasses
    # ------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def is_filebacked(self) -> bool:
        """True if pixel data are backed by a file on disk, False if in memory."""

    @abc.abstractmethod
    def get_num_pixels(self) -> int:
        """Return total number of pixels in the dataset."""

    @abc.abstractmethod
    def get_prop(self, fld: str) -> np.ndarray:
        """Return ``(k, page_size)`` array for the named field on the current page."""

    @abc.abstractmethod
    def set_prop(self, fld: str, val: Any) -> "PixelDataBase":
        """Set the named field on the current page."""

    @abc.abstractmethod
    def get_raw_data(self, page_number: Optional[int] = None) -> np.ndarray:
        """Return unaligned ``(9, page_size)`` pixel array for ``page_number``."""

    @abc.abstractmethod
    def set_raw_data(self, pix_data_arr: Any) -> "PixelDataBase":
        """Set raw ``(9, N)`` pixel array without recalculating range."""

    @abc.abstractmethod
    def get_raw_pix_data(
        self,
        row_pix_idx: np.ndarray,
        col_pix_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return unaligned pixel columns at 1-based ``row_pix_idx``."""

    @abc.abstractmethod
    def recalc_data_range(
        self, fld_names: Union[str, Sequence[str]] = "all"
    ) -> "PixelDataBase":
        """Recalculate min/max ``data_range`` for ``fld_names`` across all pages."""

    @abc.abstractmethod
    def copy(self) -> "PixelDataBase":
        """Return a copy of the pixel data object."""

    def __copy__(self) -> "PixelDataBase":
        return self.copy()

    def __deepcopy__(self, memo: Any) -> "PixelDataBase":
        return self.copy()

    # ------------------------------------------------------------------
    # Data field properties (returning _PixelFieldView for indexed writes)
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        arr = self.get_data(self.get_page_num())
        return _PixelFieldView(arr, owner=self, field_name="all")

    @data.setter
    def data(self, pixel_data: Any) -> None:
        self.set_raw_data(pixel_data)
        self.recalc_data_range("all")

    @property
    def raw_data(self) -> np.ndarray:
        return self.get_raw_data(self.get_page_num())

    @property
    def u1(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("u1"), owner=self, field_name="u1")

    @u1.setter
    def u1(self, val: Any) -> None:
        self.set_prop("u1", val)

    @property
    def u2(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("u2"), owner=self, field_name="u2")

    @u2.setter
    def u2(self, val: Any) -> None:
        self.set_prop("u2", val)

    @property
    def u3(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("u3"), owner=self, field_name="u3")

    @u3.setter
    def u3(self, val: Any) -> None:
        self.set_prop("u3", val)

    @property
    def dE(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("dE"), owner=self, field_name="dE")

    @dE.setter
    def dE(self, val: Any) -> None:
        self.set_prop("dE", val)

    @property
    def q_coordinates(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("q_coordinates"), owner=self, field_name="q_coordinates"
        )

    @q_coordinates.setter
    def q_coordinates(self, val: Any) -> None:
        self.set_prop("q_coordinates", val)

    @property
    def coordinates(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("coordinates"), owner=self, field_name="coordinates"
        )

    @coordinates.setter
    def coordinates(self, val: Any) -> None:
        self.set_prop("coordinates", val)

    @property
    def run_idx(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("run_idx"), owner=self, field_name="run_idx")

    @run_idx.setter
    def run_idx(self, val: Any) -> None:
        self.set_prop("run_idx", val)

    @property
    def detector_idx(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("detector_idx"), owner=self, field_name="detector_idx"
        )

    @detector_idx.setter
    def detector_idx(self, val: Any) -> None:
        self.set_prop("detector_idx", val)

    @property
    def energy_idx(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("energy_idx"), owner=self, field_name="energy_idx"
        )

    @energy_idx.setter
    def energy_idx(self, val: Any) -> None:
        self.set_prop("energy_idx", val)

    @property
    def signal(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("signal"), owner=self, field_name="signal")

    @signal.setter
    def signal(self, val: Any) -> None:
        self.set_prop("signal", val)

    @property
    def variance(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("variance"), owner=self, field_name="variance")

    @variance.setter
    def variance(self, val: Any) -> None:
        self.set_prop("variance", val)

    @property
    def sig_var(self) -> np.ndarray:
        return _PixelFieldView(self.get_prop("sig_var"), owner=self, field_name="sig_var")

    @sig_var.setter
    def sig_var(self, val: Any) -> None:
        self.set_prop("sig_var", val)

    @property
    def all_indexes(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("all_indexes"), owner=self, field_name="all_indexes"
        )

    @all_indexes.setter
    def all_indexes(self, val: Any) -> None:
        self.set_prop("all_indexes", val)

    @property
    def all_experiment(self) -> np.ndarray:
        return _PixelFieldView(
            self.get_prop("all_experiment"), owner=self, field_name="all_experiment"
        )

    @all_experiment.setter
    def all_experiment(self, val: Any) -> None:
        self.set_prop("all_experiment", val)

    # ------------------------------------------------------------------
    # Metadata, Ranges, Alignment & Paging properties
    # ------------------------------------------------------------------
    @property
    def num_pixels(self) -> int:
        return self.get_num_pixels()

    @property
    def npix(self) -> int:
        return self.get_num_pixels()

    def isempty(self) -> bool:
        """Return True if the pixel dataset has zero pixels."""
        return self.num_pixels == 0

    @property
    def pix_range(self) -> np.ndarray:
        return self.get_data_range([1, 2, 3, 4])

    @pix_range.setter
    def pix_range(self, range_val: Any) -> None:
        self.set_pix_range(range_val)

    def set_pix_range(self, range_val: Any) -> "PixelDataBase":
        if isinstance(range_val, (str, bytes)) or not isinstance(
            range_val, (np.ndarray, list, tuple)
        ):
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "pixel range must be a numeric array of size 2x4"
            )
        arr = np.asarray(range_val, dtype=np.float64)
        if arr.shape != (2, 4):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"pixel range must be a numeric array of size 2x4, got {arr.shape}"
            )
        self._data_range[:, :4] = arr
        return self

    @property
    def data_range(self) -> np.ndarray:
        return self.get_data_range()

    @data_range.setter
    def data_range(self, val: Any) -> None:
        self.set_data_range(val)

    def set_data_range(self, data_range: Any) -> "PixelDataBase":
        if isinstance(data_range, (str, bytes)) or not isinstance(
            data_range, (np.ndarray, list, tuple)
        ):
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "data_range should be [2x9] array of data ranges"
            )
        arr = np.asarray(data_range, dtype=np.float64)
        if arr.shape != (2, 9):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"data_range should be [2x9] array of data ranges, got {arr.shape}"
            )
        self._data_range = arr.copy()
        return self

    def get_data_range(self, field_id: Any = None) -> np.ndarray:
        if field_id is None:
            return self._data_range.copy()
        idx = self.field_index(field_id)
        return self._data_range[:, idx - 1].copy()

    def invalidate_range(self, fld: Union[str, Sequence[str], np.ndarray] = "all") -> "PixelDataBase":
        idx = self.field_index(fld)
        self._data_range[:, idx - 1] = self.EMPTY_RANGE[:, idx - 1]
        return self

    def is_range_valid(self, fld: Union[str, Sequence[str], np.ndarray] = "all") -> bool:
        idx = self.field_index(fld)
        sub = self._data_range[:, idx - 1]
        invalid = (sub == self.EMPTY_RANGE[:, idx - 1]) | np.isinf(sub) | (sub[0:1, :] > sub[1:2, :])
        return not bool(np.any(invalid))

    @property
    def is_corrected(self) -> bool:
        return self._is_corrected

    @property
    def is_misaligned(self) -> bool:
        return self._is_corrected

    @property
    def alignment_matr(self) -> np.ndarray:
        return self._alignment_matr.copy()

    @alignment_matr.setter
    def alignment_matr(self, val: Any) -> None:
        self.set_alignment_matrix(val)

    def clear_alignment(self) -> "PixelDataBase":
        self._is_corrected = False
        self._alignment_matr = np.eye(3, dtype=np.float64)
        return self

    def set_raw_alignment(self, al_mat: Any) -> "PixelDataBase":
        """Set alignment matrix without modifying in-memory pixels directly."""
        self._set_alignment_matr_internal(al_mat, self.invalidate_range)
        return self

    def _set_alignment_matr_internal(
        self,
        val: Any,
        pix_proc_function: Callable[[ str], Any],
    ) -> Tuple["PixelDataBase", bool]:
        prev_matr = self._alignment_matr
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            mat = np.eye(3, dtype=np.float64)
        elif hasattr(val, "rotmat"):
            mat = np.asarray(val.rotmat, dtype=np.float64)
        else:
            if isinstance(val, (str, bytes)) or not isinstance(val, (np.ndarray, list, tuple)):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Alignment matrix must be 3x3 numeric matrix, got {type(val).__name__}"
                )
            mat = np.asarray(val, dtype=np.float64)
            if mat.shape != (3, 3):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Alignment matrix must be 3x3 matrix, got shape {mat.shape}"
                )

        difr = mat - prev_matr
        if np.max(np.abs(difr)) > 1e-8:
            alignment_changed = True
            self._alignment_matr = mat.copy()
            mal_value = mat - np.eye(3, dtype=np.float64)
            if np.any(np.abs(mal_value) > np.finfo(np.float32).eps):
                self._is_corrected = True
            else:
                self._is_corrected = False
            pix_proc_function("q_coordinates")
        else:
            alignment_changed = False
        return self, alignment_changed

    @abc.abstractmethod
    def set_alignment_matrix(self, val: Any) -> "PixelDataBase":
        """Set alignment matrix and update or invalidate pixel ranges."""

    def finalize_alignment(self, outfile: str = "") -> "PixelDataBase":
        """Apply any pending alignment to all pages and recompute ranges."""
        targ_obj = self.copy()
        if not targ_obj.is_corrected and targ_obj.is_range_valid("all"):
            return targ_obj
        if targ_obj.is_filebacked:
            from pyhorace.data.pixels.pix_write_handle import pix_write_handle

            wh = pix_write_handle(outfile or targ_obj.full_filename, keep_precision=targ_obj.keep_precision)
            for pg in range(1, targ_obj.num_pages + 1):
                targ_obj.page_num = pg
                wh.store_page(targ_obj.data)
            targ_obj.clear_alignment()
            targ_obj.finish_dump(wh)
        else:
            if targ_obj.is_corrected:
                targ_obj._align_pixels()
                targ_obj.clear_alignment()
            targ_obj.recalc_data_range("all")
        return targ_obj

    @property
    def default_page_size(self) -> int:
        return int(hor_config().mem_chunk_size)

    @property
    def base_page_size(self) -> int:
        if self._base_page_size is not None:
            return self._base_page_size
        return self.default_page_size

    @base_page_size.setter
    def base_page_size(self, val: int) -> None:
        ival = int(val)
        if ival < 1:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"base_page_size must be >= 1, got {val}"
            )
        self._base_page_size = ival

    @property
    def full_filename(self) -> str:
        return self.get_full_filename()

    @full_filename.setter
    def full_filename(self, val: Any) -> None:
        self.set_full_filename(val)

    @property
    def file_path(self) -> str:
        return self.get_full_filename()

    @file_path.setter
    def file_path(self, val: Any) -> None:
        self.set_full_filename(val)

    @property
    def filename(self) -> str:
        ff = self.full_filename
        return os.path.basename(ff) if ff else ""

    @property
    def filepath(self) -> str:
        ff = self.full_filename
        return os.path.dirname(ff) if ff else ""

    def get_full_filename(self) -> str:
        return self._full_filename

    def set_full_filename(self, val: Any) -> "PixelDataBase":
        if val is None or val == "":
            self._full_filename = ""
            return self
        if not isinstance(val, (str, os.PathLike)):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"full_filename must be a string, got {type(val).__name__}"
            )
        self._full_filename = str(val)
        return self

    @property
    def data_wrap(self) -> pix_data:
        return self.get_data_wrap()

    @data_wrap.setter
    def data_wrap(self, val: Any) -> None:
        self.set_data_wrap(val)

    @property
    def pix_data_wrap(self) -> pix_data:
        return self.get_data_wrap()

    @pix_data_wrap.setter
    def pix_data_wrap(self, val: Any) -> None:
        self.set_data_wrap(val)

    def get_data_wrap(self) -> pix_data:
        return pix_data(self)

    @abc.abstractmethod
    def set_data_wrap(self, val: Any) -> "PixelDataBase":
        """Set pixel data from a :class:`pix_data` instance."""

    @property
    def metadata(self) -> pix_metadata:
        return pix_metadata(self)

    @metadata.setter
    def metadata(self, val: Any) -> None:
        self.set_metadata(val)

    def set_metadata(self, val: Any) -> "PixelDataBase":
        if not isinstance(val, pix_metadata):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"metadata can only be set to an instance of pix_metadata, got {type(val).__name__}"
            )
        self._full_filename = val.full_filename
        self._data_range = val.data_range.copy()
        if val.is_misaligned:
            self._alignment_matr = val.alignment_matr.copy()
            self._is_corrected = True
        else:
            self._alignment_matr = np.eye(3, dtype=np.float64)
            self._is_corrected = False
        return self

    @property
    def page_size(self) -> int:
        return self.get_page_size()

    @page_size.setter
    def page_size(self, val: int) -> None:
        self.base_page_size = val

    def reset(self) -> "PixelDataBase":
        """Reset current page number to 1."""
        self._page_num = 1
        return self

    def move_to_first_page(self) -> "PixelDataBase":
        """Move to the first page of pixels."""
        self._page_num = 1
        return self

    @property
    def page_num(self) -> int:
        return self.get_page_num()

    @page_num.setter
    def page_num(self, val: int) -> None:
        self.set_page_num(val)

    @property
    def num_pages(self) -> int:
        return self.get_num_pages()

    @property
    def read_only(self) -> bool:
        return self.get_read_only()

    @property
    def old_file_format(self) -> bool:
        return self._old_file_format

    @old_file_format.setter
    def old_file_format(self, val: Any) -> None:
        self._old_file_format = bool(val)

    @property
    def unique_run_id(self) -> np.ndarray:
        return self._unique_run_id

    @property
    def keep_precision(self) -> bool:
        return self._keep_precision

    @keep_precision.setter
    def keep_precision(self, val: Any) -> None:
        self._keep_precision = bool(val)

    @property
    def is_tmp_obj(self) -> bool:
        return self.get_is_tmp_obj()

    @property
    def pix_byte_size(self) -> int:
        return self.get_pix_byte_size()

    def get_page_size(self) -> int:
        return self.num_pixels

    def get_page_num(self) -> int:
        return self._page_num

    def set_page_num(self, val: int) -> "PixelDataBase":
        self._page_num = 1
        return self

    def get_num_pages(self) -> int:
        return 1

    def get_read_only(self) -> bool:
        return False

    def get_is_tmp_obj(self) -> bool:
        return False

    def get_pix_byte_size(self, keep_precision: Optional[bool] = None) -> int:
        return self.DEFAULT_NUM_PIX_FIELDS * 8

    def get_field_count(self, fld: Union[str, Sequence[str]]) -> int:
        return int(len(self.field_index(fld)))

    def check_set_prop(self, fld: str, val: Any) -> np.ndarray:
        """Validate and reshape input array for ``set_prop(fld, val)``."""
        if isinstance(val, (str, bytes)) or not isinstance(
            val, (int, float, np.number, np.ndarray, list, tuple)
        ):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"Value for property {fld!r} must be numeric, got {type(val).__name__}"
            )
        arr = np.asarray(val, dtype=np.float64)
        n_rows = self.get_field_count(fld)
        n_cols = self.page_size
        block_size = (n_rows, n_cols)

        if arr.size == 1:
            return np.full(block_size, arr.item(), dtype=np.float64)

        if arr.ndim == 1:
            # Allow 1D vector of length page_size when setting a 1-row property (u1, signal, etc.)
            if n_rows == 1 and arr.size == n_cols:
                return arr.reshape((1, n_cols))
            if n_cols == 1 and arr.size == n_rows:
                return arr.reshape((n_rows, 1))
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"{fld} value must be scalar or [{n_rows} {n_cols}] numeric array. "
                f"Received shape {arr.shape}"
            )

        if arr.shape != block_size:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"{fld} value must be scalar or [{n_rows} {n_cols}] numeric array. "
                f"Received shape {arr.shape}"
            )
        return arr

    # ------------------------------------------------------------------
    # Data access methods: get_data, get_fields, get_pixels, get_pix_in_ranges
    # ------------------------------------------------------------------
    def get_fields(
        self,
        pix_fields: Union[str, Sequence[str]],
        page_num: Optional[int] = None,
    ) -> np.ndarray:
        """Retrieve one or more pixel fields for the given ``page_num``."""
        return self.get_data(pix_fields, page_num)

    def get_data(
        self,
        pix_fields_or_page: Any = None,
        page_num: Optional[int] = None,
    ) -> np.ndarray:
        """Retrieve pixel fields for a specified page (or current page)."""
        if pix_fields_or_page is None:
            pg = self.get_page_num() if page_num is None else int(page_num)
            fld_idx = self.field_index("all")
        elif isinstance(pix_fields_or_page, (int, np.integer)) and page_num is None:
            pg = int(pix_fields_or_page)
            fld_idx = self.field_index("all")
        else:
            fld_idx = self.field_index(pix_fields_or_page)
            pg = self.get_page_num() if page_num is None else int(page_num)

        if pg < 1 or pg > self.num_pages:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"Page number {pg} out of range 1..{self.num_pages}"
            )

        raw = self.get_raw_data(pg)
        if raw.size == 0:
            return np.empty((len(fld_idx), 0), dtype=np.float64)
        if self._is_corrected and np.any(fld_idx <= 3):
            raw = raw.copy()
            raw[:3, :] = self._alignment_matr @ raw[:3, :]
        return raw[fld_idx - 1, :]

    def _parse_get_pix_args(
        self,
        accepts_logical: bool,
        abs_pix_indices: Any,
        *args: Any,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool, bool, bool, bool]:
        ignore_range = False
        raw_data = False
        keep_precision = False
        align = False
        pix_col_idx: Optional[np.ndarray] = None

        for a in args:
            if isinstance(a, str):
                low = a.lower()
                if low == "-ignore_range":
                    ignore_range = True
                elif low == "-raw_data":
                    raw_data = True
                elif low == "-keep_precision":
                    keep_precision = True
                elif low == "-align":
                    align = True
                else:
                    pix_col_idx = self.field_index(a)
            elif isinstance(a, (list, tuple, np.ndarray)):
                pix_col_idx = self.field_index(a)
            else:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Unrecognized argument to get_pixels: {a!r}"
                )

        if abs_pix_indices is None:
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "Pixel indices must be provided to get_pixels."
            )

        idx_arr = np.asarray(abs_pix_indices)
        if idx_arr.dtype == bool or np.issubdtype(idx_arr.dtype, np.bool_):
            if not accepts_logical:
                raise HoraceError(
                    "HORACE:PixelDataBase:invalid_argument - "
                    "Logical indexing is not supported here."
                )
            flat_bool = idx_arr.ravel(order="F")
            if flat_bool.size > self.num_pixels:
                if np.any(flat_bool[self.num_pixels :]):
                    raise HoraceError(
                        f"HORACE:PixelDataBase:invalid_argument - "
                        f"Logical index length ({flat_bool.size}) exceeds num_pixels ({self.num_pixels})."
                    )
                flat_bool = flat_bool[: self.num_pixels]
            idx_1based = np.flatnonzero(flat_bool) + 1
        elif np.issubdtype(idx_arr.dtype, np.number):
            flat_num = idx_arr.ravel(order="F")
            if flat_num.size == 0:
                idx_1based = np.array([], dtype=np.intp)
            else:
                if np.any(flat_num < 1) or np.any(flat_num != np.floor(flat_num)):
                    raise HoraceError(
                        "HORACE:PixelDataBase:invalid_argument - "
                        "Pixel indices must be positive integers >= 1."
                    )
                if np.any(flat_num > self.num_pixels):
                    raise HoraceError(
                        f"HORACE:PixelDataBase:invalid_argument - "
                        f"Pixel index exceeds num_pixels ({self.num_pixels})."
                    )
                idx_1based = flat_num.astype(np.intp)
        else:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"Pixel indices must be numeric or logical, got {idx_arr.dtype}"
            )

        return idx_1based, pix_col_idx, ignore_range, raw_data, keep_precision, align

    def _pack_get_pix_result(
        self,
        pix_data_arr: np.ndarray,
        pix_col_idx: Optional[np.ndarray],
        ignore_range: bool,
        raw_data: bool,
        keep_precision: bool,
        align: bool,
    ) -> Union["PixelDataBase", np.ndarray]:
        from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

        out_arr = pix_data_arr
        if self._is_corrected and align:
            if pix_col_idx is None:
                out_arr = out_arr.copy()
                out_arr[:3, :] = self._alignment_matr @ out_arr[:3, :]
            elif np.any(pix_col_idx <= 3):
                # Re-align Q coordinates if any of u1, u2, u3 requested
                pass

        if not keep_precision:
            out_arr = np.asarray(out_arr, dtype=np.float64)

        if raw_data:
            return out_arr

        if ignore_range:
            pix_out = PixelDataMemory()
            pix_out.set_raw_data(np.asarray(out_arr, dtype=np.float64))
        else:
            pix_out = PixelDataMemory(np.asarray(out_arr, dtype=np.float64))
        pix_out.full_filename = self.full_filename
        if self._is_corrected and not align:
            pix_out.alignment_matr = self._alignment_matr.copy()
        return pix_out

    def get_pixels(
        self, abs_pix_indices: Any, *args: Any
    ) -> Union["PixelDataBase", np.ndarray]:
        """Extract a subset of pixels at 1-based ``abs_pix_indices`` (or boolean mask)."""
        (
            idx_1based,
            pix_col_idx,
            ignore_range,
            raw_data,
            keep_precision,
            align,
        ) = self._parse_get_pix_args(True, abs_pix_indices, *args)

        if idx_1based.size == 0:
            n_rows = 9 if pix_col_idx is None else len(pix_col_idx)
            dtype = np.float32 if (keep_precision and self.is_filebacked) else np.float64
            raw_pix = np.empty((n_rows, 0), dtype=dtype)
        else:
            if self._is_corrected and (align or not raw_data) and pix_col_idx is not None and np.any(pix_col_idx <= 3):
                full_raw = self.get_raw_pix_data(idx_1based, None)
                if not keep_precision:
                    full_raw = np.asarray(full_raw, dtype=np.float64)
                full_raw = full_raw.copy()
                full_raw[:3, :] = self._alignment_matr @ full_raw[:3, :]
                raw_pix = full_raw[pix_col_idx - 1, :]
                if raw_data:
                    return raw_pix
            else:
                col_0 = None if pix_col_idx is None else (pix_col_idx - 1)
                raw_pix = self.get_raw_pix_data(idx_1based, col_0)

        return self._pack_get_pix_result(
            raw_pix, pix_col_idx, ignore_range, raw_data, keep_precision, align
        )

    def get_pix_in_ranges(
        self,
        abs_indices_starts: Any,
        block_sizes: Any,
        recalculate_pix_ranges: bool = False,
        keep_precision: bool = False,
    ) -> "PixelDataBase":
        """Extract contiguous blocks of pixels defined by 1-based starts and sizes."""
        starts = np.asarray(abs_indices_starts)
        sizes = np.asarray(block_sizes)
        if starts.size != sizes.size:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"abs_indices_starts ({starts.size}) and block_sizes ({sizes.size}) must have equal number of elements."
            )
        starts_flat = starts.ravel(order="F").astype(np.intp)
        sizes_flat = sizes.ravel(order="F").astype(np.intp)
        ends_flat = starts_flat + sizes_flat - 1

        if np.any(sizes_flat < 0):
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "block_sizes must be non-negative."
            )
        non_empty = sizes_flat > 0
        if np.any(starts_flat[non_empty] < 1) or np.any(ends_flat[non_empty] > self.num_pixels):
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"Pixel block ranges exceed bounds [1, {self.num_pixels}]."
            )

        segments = [
            np.arange(s, s + sz, dtype=np.intp)
            for s, sz in zip(starts_flat, sizes_flat)
            if sz > 0
        ]
        if len(segments) == 0:
            indices = np.array([], dtype=np.intp)
        else:
            indices = np.concatenate(segments)

        flags: List[str] = []
        if not recalculate_pix_ranges:
            flags.append("-ignore_range")
        if keep_precision:
            flags.append("-keep_precision")
        res = self.get_pixels(indices, *flags)
        assert isinstance(res, PixelDataBase)
        if not recalculate_pix_ranges:
            res.set_data_range(self.data_range)
        return res

    # ------------------------------------------------------------------
    # Algorithms on pixels: mask, compute_bin_data, noisify, unary/binary ops
    # ------------------------------------------------------------------
    def mask(self, mask_array: Any, npix: Any = None) -> "PixelDataBase":
        """Remove pixels where ``mask_array`` is False (0), keeping True (1)."""
        if mask_array is None:
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "mask_array cannot be None."
            )
        m_arr = np.asarray(mask_array)
        if not (np.issubdtype(m_arr.dtype, np.bool_) or np.issubdtype(m_arr.dtype, np.number)):
            raise HoraceError(
                "HORACE:PixelDataBase:invalid_argument - "
                "mask_array must be logical or numeric."
            )
        m_bool = m_arr.astype(bool).ravel(order="F")

        if npix is not None:
            npix_arr = np.asarray(npix, dtype=np.intp).ravel(order="F")
            if npix_arr.size != m_bool.size:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"mask_array size ({m_bool.size}) must equal npix size ({npix_arr.size})."
                )
            if int(np.sum(npix_arr)) != self.num_pixels:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"sum(npix) ({int(np.sum(npix_arr))}) must equal num_pixels ({self.num_pixels})."
                )
            full_mask = np.repeat(m_bool, npix_arr)
        else:
            if m_bool.size != self.num_pixels:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"mask_array numel ({m_bool.size}) must equal num_pixels ({self.num_pixels})."
                )
            full_mask = m_bool

        if np.all(full_mask):
            return self.copy()
        if not np.any(full_mask):
            from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

            out = PixelDataMemory()
            out.full_filename = self.full_filename
            return out

        n_keep = int(np.sum(full_mask))
        if self.is_filebacked and PixelDataBase.do_filebacked(n_keep):
            from pyhorace.data.pixels.pix_write_handle import pix_write_handle
            from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked

            wh = pix_write_handle(self.full_filename, keep_precision=self.keep_precision)
            saved_pg = self.page_num
            for pg in range(1, self.num_pages + 1):
                self.page_num = pg
                s_idx, e_idx = self.get_page_idx_(pg)
                pg_mask = full_mask[s_idx - 1 : e_idx]
                if np.any(pg_mask):
                    wh.store_page(self.data[:, pg_mask])
            self.page_num = saved_pg
            out_fb = PixelDataFileBacked()
            out_fb.finish_dump(wh)
            return out_fb

        res = self.get_pixels(full_mask)
        assert isinstance(res, PixelDataBase)
        return res

    def compute_bin_data(
        self,
        npix: Any,
        pix_idx: Any = None,
        return_msd: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute mean signal and mean variance per bin given ``npix``."""
        npix_arr = np.asarray(npix, dtype=np.float64)
        orig_shape = npix_arr.shape
        npix_flat = npix_arr.ravel(order="F").astype(np.intp)

        if self.num_pixels == 0 or np.sum(npix_flat) == 0:
            zeros = np.zeros(orig_shape, dtype=np.float64, order="F")
            if return_msd:
                return zeros.copy(), zeros.copy(), zeros.copy()
            return zeros.copy(), zeros.copy()

        s_sum = np.zeros(npix_flat.size, dtype=np.float64)
        e_sum = np.zeros(npix_flat.size, dtype=np.float64)

        if pix_idx is not None and np.asarray(pix_idx).size > 0:
            sv = self.get_pixels(pix_idx, "sig_var", "-raw_data")
            sig = np.asarray(sv[0, :], dtype=np.float64)
            var = np.asarray(sv[1, :], dtype=np.float64)
            bin_ids = np.repeat(np.arange(npix_flat.size, dtype=np.intp), npix_flat)
            np.add.at(s_sum, bin_ids, sig)
            np.add.at(e_sum, bin_ids, var)
        elif self.is_filebacked and self.num_pages > 1:
            bin_ids = np.repeat(np.arange(npix_flat.size, dtype=np.intp), npix_flat)
            saved_pg = self.page_num
            for pg in range(1, self.num_pages + 1):
                self.page_num = pg
                s_idx, e_idx = self.get_page_idx_(pg)
                pg_bins = bin_ids[s_idx - 1 : e_idx]
                np.add.at(s_sum, pg_bins, self.signal.ravel())
                np.add.at(e_sum, pg_bins, self.variance.ravel())
            self.page_num = saved_pg
        else:
            bin_ids = np.repeat(np.arange(npix_flat.size, dtype=np.intp), npix_flat)
            np.add.at(s_sum, bin_ids, self.signal.ravel())
            np.add.at(e_sum, bin_ids, self.variance.ravel())

        nonzero = npix_flat > 0
        mean_s = np.zeros_like(s_sum)
        mean_e = np.zeros_like(e_sum)
        mean_s[nonzero] = s_sum[nonzero] / npix_flat[nonzero]
        mean_e[nonzero] = e_sum[nonzero] / (npix_flat[nonzero].astype(np.float64) ** 2)

        mean_signal = mean_s.reshape(orig_shape, order="F")
        mean_variance = mean_e.reshape(orig_shape, order="F")

        if return_msd:
            # Compute mean square deviation around bin mean
            msd_sum = np.zeros(npix_flat.size, dtype=np.float64)
            bin_ids = np.repeat(np.arange(npix_flat.size, dtype=np.intp), npix_flat)
            if pix_idx is not None and np.asarray(pix_idx).size > 0:
                sig = np.asarray(self.get_pixels(pix_idx, "signal", "-raw_data")).ravel()
            else:
                sig = self.signal.ravel()
            diff_sq = (sig - mean_s[bin_ids]) ** 2
            np.add.at(msd_sum, bin_ids, diff_sq)
            msd = np.zeros_like(msd_sum)
            msd[nonzero] = msd_sum[nonzero] / npix_flat[nonzero]
            return mean_signal, mean_variance, msd.reshape(orig_shape, order="F")

        return mean_signal, mean_variance

    def recompute_bin_data(self) -> "PixelDataBase":
        """Recalculate signal and variance data ranges across all pages."""
        return self.recalc_data_range("sig_var")

    def noisify(self, *args: Any, **kwargs: Any) -> "PixelDataBase":
        """Add random noise to pixel signal and update variance."""
        pix_out = self.copy()
        if pix_out.num_pixels == 0:
            return pix_out

        factor = 0.1
        is_poisson = False
        ymax: Optional[float] = None
        rng_func: Callable[[Tuple[int, ...]], np.ndarray] = np.random.standard_normal

        i = 0
        pos_args = list(args)
        while i < len(pos_args):
            a = pos_args[i]
            if isinstance(a, (int, float, np.number)):
                factor = float(a)
                i += 1
            elif isinstance(a, str):
                low = a.lower()
                if low == "poisson":
                    is_poisson = True
                    i += 1
                elif low in ("maximum_value", "ymax") and i + 1 < len(pos_args):
                    ymax = float(pos_args[i + 1])
                    i += 2
                else:
                    i += 1
            elif callable(a):
                rng_func = a
                i += 1
            else:
                i += 1

        if "poisson" in kwargs and kwargs["poisson"]:
            is_poisson = True
        if "factor" in kwargs:
            factor = float(kwargs["factor"])
        if "maximum_value" in kwargs:
            ymax = float(kwargs["maximum_value"])

        if not is_poisson and ymax is None:
            if not pix_out.is_range_valid("signal"):
                pix_out.recalc_data_range("signal")
            ymax = float(np.max(np.abs(pix_out.get_data_range("signal"))))

        def _noisify_page(sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if is_poisson:
                abs_sig = np.abs(sig)
                new_sig = np.random.poisson(abs_sig).astype(np.float64)
                new_var = abs_sig.copy()
            else:
                scale = factor * (ymax if ymax is not None else 1.0)
                noise = scale * rng_func(sig.shape)
                new_sig = sig + noise
                new_var = np.full(sig.shape, scale ** 2, dtype=np.float64)
            return new_sig, new_var

        if pix_out.is_filebacked:
            from pyhorace.data.pixels.pix_write_handle import pix_write_handle

            wh = pix_write_handle(pix_out.full_filename, keep_precision=pix_out.keep_precision)
            for pg in range(1, pix_out.num_pages + 1):
                pix_out.page_num = pg
                pg_data = pix_out.get_raw_data(pg).copy()
                ns, nv = _noisify_page(pg_data[7:8, :])
                pg_data[7:8, :] = ns
                pg_data[8:9, :] = nv
                wh.store_page(pg_data)
            pix_out.finish_dump(wh)
        else:
            ns, nv = _noisify_page(pix_out.signal)
            pix_out.signal = ns
            pix_out.variance = nv
        return pix_out

    # ------------------------------------------------------------------
    # data_op_interface & sigvar integration (do_unary_op, do_binary_op)
    # ------------------------------------------------------------------
    def get_NUM_DIMS(self) -> int:
        return 2

    def get_img_size(self) -> Tuple[int, ...]:
        return (self.DEFAULT_NUM_PIX_FIELDS, self.num_pixels)

    def sigvar_size(self) -> Tuple[int, int]:
        return (1, 1)

    def sigvar(self) -> sigvar:
        return sigvar(self.signal, self.variance)

    def sigvar_set(self, sv: sigvar) -> "PixelDataBase":
        self.signal = sv.s
        self.variance = sv.e
        return self

    def do_unary_op(self, unary_op: Callable[..., Any]) -> "PixelDataBase":
        """Apply ``unary_op`` (a ``sigvar`` unary operation) across all pixel pages."""
        out_obj = self.copy()
        if out_obj.num_pixels == 0:
            return out_obj

        if out_obj.is_filebacked:
            from pyhorace.data.pixels.pix_write_handle import pix_write_handle

            wh = pix_write_handle(out_obj.full_filename, keep_precision=out_obj.keep_precision)
            for pg in range(1, out_obj.num_pages + 1):
                out_obj.page_num = pg
                pg_data = out_obj.get_data(pg).copy()
                res_sv = unary_op(sigvar(pg_data[7:8, :], pg_data[8:9, :]))
                pg_data[7:8, :] = res_sv.s
                pg_data[8:9, :] = res_sv.e
                wh.store_page(pg_data)
            out_obj.finish_dump(wh)
        else:
            res_sv = unary_op(sigvar(out_obj.signal, out_obj.variance))
            out_obj.signal = res_sv.s
            out_obj.variance = res_sv.e
        return out_obj

    @staticmethod
    def binary_op_manager_single(
        w1: Any,
        w2: Any,
        binary_op: Callable[..., Any],
        ignore_cell_sorting: bool = False,
    ) -> Any:
        """Dispatch binary operations involving ``PixelDataBase``."""
        op_name = getattr(binary_op, "__name__", str(binary_op))
        flip, _ = data_op_interface.get_operation_order(w1, w2, op_name)
        if flip:
            lead = w2
            operand = w1
        else:
            lead = w1
            operand = w2
        if isinstance(lead, PixelDataBase):
            return lead.do_binary_op(operand, binary_op, flip=flip)
        return data_op_interface.binary_op_manager_single(w1, w2, binary_op, ignore_cell_sorting)

    def do_binary_op(
        self,
        operand: Any,
        binary_op: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> "PixelDataBase":
        """Perform a binary operation between ``self`` and ``operand``."""
        flip: bool = bool(kwargs.get("flip", False))
        npix: Any = kwargs.get("npix", None)

        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, str):
                low = a.lower()
                if low == "flip" and i + 1 < len(args):
                    flip = bool(args[i + 1])
                    i += 2
                    continue
                if low == "npix" and i + 1 < len(args):
                    npix = args[i + 1]
                    i += 2
                    continue
            elif isinstance(a, (bool, np.bool_)) and i == 0:
                flip = bool(a)
                i += 1
                continue
            i += 1

        # Validate operand type
        from pyhorace.data.ix_dataset import IX_dataset
        from pyhorace.sqw.dnd_base import DnDBase

        is_valid_op = isinstance(
            operand,
            (
                PixelDataBase,
                DnDBase,
                IX_dataset,
                sigvar,
                int,
                float,
                np.number,
                np.ndarray,
                list,
                tuple,
            ),
        ) and not isinstance(operand, (str, bytes))
        if not is_valid_op:
            raise HoraceError(
                f"HORACE:PixelDataBase:invalid_argument - "
                f"Invalid operand of type {type(operand).__name__} for binary operation."
            )

        def _apply_sv_op(sv_self: sigvar, sv_other: sigvar) -> sigvar:
            if flip:
                return binary_op(sv_other, sv_self)
            return binary_op(sv_self, sv_other)

        pix_out = self.copy()

        # Case 1: another PixelDataBase
        if isinstance(operand, PixelDataBase):
            if operand.num_pixels != self.num_pixels:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"PixelDataBase operands must have equal num_pixels ({self.num_pixels} != {operand.num_pixels})."
                )
            if pix_out.is_filebacked:
                from pyhorace.data.pixels.pix_write_handle import pix_write_handle

                wh = pix_write_handle(pix_out.full_filename, keep_precision=pix_out.keep_precision)
                for pg in range(1, pix_out.num_pages + 1):
                    pix_out.page_num = pg
                    s_idx, e_idx = pix_out.get_page_idx_(pg)
                    pg_data = pix_out.get_data(pg).copy()
                    sv1 = sigvar(pg_data[7:8, :], pg_data[8:9, :])
                    if operand.is_filebacked:
                        op_sv_arr = operand.get_pixels(
                            np.arange(s_idx, e_idx + 1, dtype=np.intp), "sig_var", "-raw_data"
                        )
                    else:
                        op_sv_arr = operand.sig_var[:, s_idx - 1 : e_idx]
                    sv2 = sigvar(op_sv_arr[0:1, :], op_sv_arr[1:2, :])
                    res_sv = _apply_sv_op(sv1, sv2)
                    pg_data[7:8, :] = res_sv.s
                    pg_data[8:9, :] = res_sv.e
                    wh.store_page(pg_data)
                pix_out.finish_dump(wh)
            else:
                sv1 = sigvar(pix_out.signal, pix_out.variance)
                if operand.is_filebacked:
                    op_sv_arr = operand.get_pixels(
                        np.arange(1, operand.num_pixels + 1, dtype=np.intp), "sig_var", "-raw_data"
                    )
                    sv2 = sigvar(op_sv_arr[0:1, :], op_sv_arr[1:2, :])
                else:
                    sv2 = sigvar(operand.signal, operand.variance)
                res_sv = _apply_sv_op(sv1, sv2)
                pix_out.signal = res_sv.s
                pix_out.variance = res_sv.e
            return pix_out

        # Case 2: DnDBase, IX_dataset, sigvar, or numeric scalar/array
        if isinstance(operand, DnDBase):
            op_s = np.asarray(operand.s, dtype=np.float64)
            op_e = np.asarray(operand.e, dtype=np.float64)
            if npix is None:
                npix = operand.npix
        elif isinstance(operand, IX_dataset):
            op_s = np.asarray(operand.signal, dtype=np.float64)
            op_e = np.asarray(operand.error, dtype=np.float64) ** 2
        elif isinstance(operand, sigvar):
            op_s = np.asarray(operand.s, dtype=np.float64)
            op_e = np.asarray(operand.e, dtype=np.float64)
        else:
            op_s = np.asarray(operand, dtype=np.float64)
            op_e = np.zeros_like(op_s)

        if op_s.size == 1 and (npix is None or not isinstance(operand, (DnDBase, IX_dataset))):
            # Scalar operation
            s_val = np.array([[float(op_s.item())]], dtype=np.float64)
            e_val = np.array([[float(op_e.item())]], dtype=np.float64)
            sv_scalar = sigvar(s_val, e_val)
            if pix_out.is_filebacked:
                from pyhorace.data.pixels.pix_write_handle import pix_write_handle

                wh = pix_write_handle(pix_out.full_filename, keep_precision=pix_out.keep_precision)
                for pg in range(1, pix_out.num_pages + 1):
                    pix_out.page_num = pg
                    pg_data = pix_out.get_data(pg).copy()
                    sv1 = sigvar(pg_data[7:8, :], pg_data[8:9, :])
                    res_sv = _apply_sv_op(sv1, sv_scalar)
                    pg_data[7:8, :] = res_sv.s
                    pg_data[8:9, :] = res_sv.e
                    wh.store_page(pg_data)
                pix_out.finish_dump(wh)
            else:
                sv1 = sigvar(pix_out.signal, pix_out.variance)
                res_sv = _apply_sv_op(sv1, sv_scalar)
                pix_out.signal = res_sv.s
                pix_out.variance = res_sv.e
            return pix_out

        # Array / image operation (with or without npix)
        if npix is not None:
            npix_arr = np.asarray(npix, dtype=np.intp)
            if op_s.size != npix_arr.size and not (
                op_s.shape == npix_arr.shape
            ):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Operand size {op_s.shape} must match npix size {npix_arr.shape}."
                )
            if npix_arr.ndim >= 2 and op_s.ndim >= 2 and op_s.shape != npix_arr.shape:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Operand shape {op_s.shape} must match npix shape {npix_arr.shape}."
                )
            npix_flat = npix_arr.ravel(order="F")
            if int(np.sum(npix_flat)) != self.num_pixels:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"sum(npix) ({int(np.sum(npix_flat))}) must equal num_pixels ({self.num_pixels})."
                )
            exp_s = np.repeat(op_s.ravel(order="F"), npix_flat).reshape(1, -1)
            exp_e = np.repeat(op_e.ravel(order="F"), npix_flat).reshape(1, -1)
        else:
            if op_s.size != self.num_pixels:
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Operand numel ({op_s.size}) must equal num_pixels ({self.num_pixels})."
                )
            if op_s.ndim == 2 and op_s.shape not in ((1, self.num_pixels), (self.num_pixels, 1)):
                raise HoraceError(
                    f"HORACE:PixelDataBase:invalid_argument - "
                    f"Operand shape {op_s.shape} incompatible with (1, {self.num_pixels})."
                )
            exp_s = op_s.ravel(order="F").reshape(1, -1)
            exp_e = op_e.ravel(order="F").reshape(1, -1)

        if pix_out.is_filebacked:
            from pyhorace.data.pixels.pix_write_handle import pix_write_handle

            wh = pix_write_handle(pix_out.full_filename, keep_precision=pix_out.keep_precision)
            for pg in range(1, pix_out.num_pages + 1):
                pix_out.page_num = pg
                s_idx, e_idx = pix_out.get_page_idx_(pg)
                pg_data = pix_out.get_data(pg).copy()
                sv1 = sigvar(pg_data[7:8, :], pg_data[8:9, :])
                sv2 = sigvar(exp_s[:, s_idx - 1 : e_idx], exp_e[:, s_idx - 1 : e_idx])
                res_sv = _apply_sv_op(sv1, sv2)
                pg_data[7:8, :] = res_sv.s
                pg_data[8:9, :] = res_sv.e
                wh.store_page(pg_data)
            pix_out.finish_dump(wh)
        else:
            sv1 = sigvar(pix_out.signal, pix_out.variance)
            sv2 = sigvar(exp_s, exp_e)
            res_sv = _apply_sv_op(sv1, sv2)
            pix_out.signal = res_sv.s
            pix_out.variance = res_sv.e

        return pix_out

    # ------------------------------------------------------------------
    # Equality & Comparison (equal_to_tol)
    # ------------------------------------------------------------------
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PixelDataBase):
            return False
        ok, _ = self.equal_to_tol(other)
        return bool(ok)

    def eq_to_tol_type_equal(
        self,
        other: Any,
        name_a: str = "input_1",
        name_b: str = "input_2",
    ) -> Tuple[bool, str]:
        if isinstance(other, PixelDataBase):
            return True, ""
        return (
            False,
            f"Objects '{name_a}' ({type(self).__name__}) and '{name_b}' ({type(other).__name__}) have different types",
        )

    def equal_to_tol(
        self,
        other: Any,
        tol: Any = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[bool, str]:
        """Compare two ``PixelDataBase`` objects within tolerance ``tol``."""
        name_a = str(kwargs.get("name_a", "input_1"))
        name_b = str(kwargs.get("name_b", "input_2"))
        nan_equal = bool(kwargs.get("nan_equal", True))
        reorder = bool(kwargs.get("reorder", False))
        npix = kwargs.get("npix", None)
        fraction = float(kwargs.get("fraction", 1.0))

        # Parse optional MATLAB-style key-value args in *args
        i = 0
        while i < len(args):
            a = args[i]
            if isinstance(a, (int, float, np.number, list, tuple, np.ndarray)) and i == 0 and tol == 0.0:
                tol = a
                i += 1
                continue
            if isinstance(a, str):
                low = a.lower().lstrip("-")
                if low == "nan_equal" and i + 1 < len(args):
                    nan_equal = bool(args[i + 1])
                    i += 2
                    continue
                if low == "reorder" and i + 1 < len(args):
                    reorder = bool(args[i + 1])
                    i += 2
                    continue
                if low == "npix" and i + 1 < len(args):
                    npix = args[i + 1]
                    i += 2
                    continue
                if low == "fraction" and i + 1 < len(args):
                    fraction = float(args[i + 1])
                    i += 2
                    continue
                if low == "name_a" and i + 1 < len(args):
                    name_a = str(args[i + 1])
                    i += 2
                    continue
                if low == "name_b" and i + 1 < len(args):
                    name_b = str(args[i + 1])
                    i += 2
                    continue
            i += 1

        ok_type, mess_type = self.eq_to_tol_type_equal(other, name_a, name_b)
        if not ok_type:
            return False, mess_type

        if self.num_pixels != other.num_pixels:
            return (
                False,
                f"Number of pixels differs: {name_a}.num_pixels={self.num_pixels} vs {name_b}.num_pixels={other.num_pixels}",
            )

        if self.num_pixels == 0:
            return True, ""

        # Parse tolerance [abs_tol, rel_tol]
        if isinstance(tol, (int, float, np.number)):
            f_tol = float(tol)
            if f_tol >= 0:
                abs_tol, rel_tol = f_tol, 0.0
            else:
                abs_tol, rel_tol = 0.0, abs(f_tol)
        else:
            t_arr = np.asarray(tol, dtype=np.float64).ravel()
            abs_tol = float(t_arr[0]) if t_arr.size >= 1 else 0.0
            rel_tol = float(t_arr[1]) if t_arr.size >= 2 else 0.0

        def _cmp_arrays(a: np.ndarray, b: np.ndarray, label: str) -> Tuple[bool, str]:
            return compare_values(
                a, b, name_a=f"{name_a}.{label}", name_b=f"{name_b}.{label}",
                tol=(abs_tol, rel_tol), nan_equal=nan_equal
            )

        if reorder:
            if npix is None:
                npix_flat = np.array([self.num_pixels], dtype=np.intp)
            else:
                npix_flat = np.asarray(npix, dtype=np.intp).ravel(order="F")
            bin_ids = np.repeat(np.arange(npix_flat.size, dtype=np.intp), npix_flat)
            d1 = self.get_pixels(np.arange(1, self.num_pixels + 1), "-raw_data")
            d2 = other.get_pixels(np.arange(1, other.num_pixels + 1), "-raw_data")
            order1 = np.lexsort((d1[6, :], d1[5, :], d1[4, :], bin_ids))
            order2 = np.lexsort((d2[6, :], d2[5, :], d2[4, :], bin_ids))
            return _cmp_arrays(d1[:, order1], d2[:, order2], "data")

        if fraction < 1.0 and fraction > 0.0:
            spacing = max(1, int(math.floor(1.0 / fraction)))
            idx = np.arange(1, self.num_pixels + 1, spacing, dtype=np.intp)
            d1 = self.get_pixels(idx, "-raw_data")
            d2 = other.get_pixels(idx, "-raw_data")
            return _cmp_arrays(d1, d2, "data")

        if not self.is_filebacked and not other.is_filebacked:
            return _cmp_arrays(self.data, other.data, "data")

        # At least one is filebacked: compare page by page
        fb = self if self.is_filebacked else other
        saved_pg1 = self.page_num
        saved_pg2 = other.page_num
        try:
            for pg in range(1, fb.num_pages + 1):
                s_idx, e_idx = fb.get_page_idx_(pg)
                idx = np.arange(s_idx, e_idx + 1, dtype=np.intp)
                d1 = self.get_pixels(idx, "-raw_data")
                d2 = other.get_pixels(idx, "-raw_data")
                ok, mess = _cmp_arrays(d1, d2, f"data[:, {s_idx}:{e_idx}]")
                if not ok:
                    return False, mess
        finally:
            self.page_num = saved_pg1
            other.page_num = saved_pg2
        return True, ""

    def get_pix_byte_size(self, keep_precision: bool = False) -> int:
        """Return size in bytes of a single pixel in memory or on disk."""
        if self.is_filebacked and keep_precision:
            return self.DEFAULT_NUM_PIX_FIELDS * 4
        return self.DEFAULT_NUM_PIX_FIELDS * 8

    @property
    def pix_byte_size(self) -> int:
        return self.get_pix_byte_size()

    def tag(self, selected: Any = None) -> "PixelDataBase":
        """Tag pixels by negating their detector_idx so they are ignored in subsequent binning."""
        if self.num_pixels == 0:
            return self
        det = self.get_prop("detector_idx").copy()
        if selected is None:
            det = -det
        else:
            sel = np.asarray(selected).ravel()
            if sel.size > 0:
                if sel.dtype == bool:
                    det[:, sel] = -det[:, sel]
                else:
                    det[:, sel.astype(int) - 1] = -det[:, sel.astype(int) - 1]
        self.set_prop("detector_idx", det)
        return self

    def untag(self, selected: Any = None) -> "PixelDataBase":
        """Untag pixels by restoring positive detector_idx."""
        if self.num_pixels == 0:
            return self
        det = self.get_prop("detector_idx").copy()
        if selected is None:
            det = np.abs(det)
        else:
            sel = np.asarray(selected).ravel()
            if sel.size > 0:
                if sel.dtype == bool:
                    det[:, sel] = np.abs(det[:, sel])
                else:
                    det[:, sel.astype(int) - 1] = np.abs(det[:, sel.astype(int) - 1])
        self.set_prop("detector_idx", det)
        return self

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def class_version(self) -> int:
        return 2

    def saveable_fields(self) -> List[str]:
        return list(self._fields_to_save)

    @classmethod
    def loadobj(cls, S: Any) -> "PixelDataBase":
        """Restore ``PixelDataMemory`` or ``PixelDataFileBacked`` from a saved struct/dict."""
        from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
        from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory

        if isinstance(S, PixelDataBase):
            return S

        if isinstance(S, np.ndarray) and S.dtype == object:
            out = np.empty(S.shape, dtype=object)
            out_flat = out.ravel(order="F")
            for i, elem in enumerate(S.ravel(order="F")):
                out_flat[i] = cls.loadobj(elem)
            return out.reshape(S.shape, order="F").view(DataOpArray)  # type: ignore[return-value]

        if not isinstance(S, dict):
            return PixelDataMemory(S)

        d = dict(S)
        if "array_dat" in d and isinstance(d["array_dat"], (dict, np.ndarray, list)):
            ad = d["array_dat"]
            if isinstance(ad, np.ndarray) and ad.size > 1:
                out = np.empty(ad.shape, dtype=object)
                out_flat = out.ravel(order="F")
                for i, elem in enumerate(ad.ravel(order="F")):
                    out_flat[i] = cls.loadobj(elem)
                return out.reshape(ad.shape, order="F").view(DataOpArray)  # type: ignore[return-value]
            if isinstance(ad, np.ndarray) and ad.size == 1:
                ad = ad.item()
            if isinstance(ad, dict):
                d = dict(ad)

        if "data_" in d and "data" not in d:
            d["data"] = d["data_"]
        if "raw_data_" in d and "data" not in d:
            d["data"] = d["raw_data_"]

        serial_name = str(d.get("serial_name", ""))
        if serial_name == "PixelDataFileBacked" or (
            "data" in d and isinstance(d["data"], str)
        ):
            obj: PixelDataBase = PixelDataFileBacked()
        else:
            obj = PixelDataMemory()

        if "data_wrap" in d or "pix_data_wrap" in d:
            dw = d.get("data_wrap", d.get("pix_data_wrap"))
            if isinstance(dw, dict):
                dw = pix_data.loadobj(dw)
            elif not isinstance(dw, pix_data):
                dw = pix_data(dw)
            obj.set_data_wrap(dw)

        if "metadata" in d:
            md = d["metadata"]
            if isinstance(md, dict):
                md = pix_metadata.loadobj(md)
            elif not isinstance(md, pix_metadata):
                md = pix_metadata(md)
            obj.set_metadata(md)

        if "data" in d and "data_wrap" not in d and "pix_data_wrap" not in d:
            raw_d = d["data"]
            if isinstance(raw_d, str):
                obj = PixelDataFileBacked(raw_d)
            else:
                arr = np.asarray(raw_d, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] == 9:
                    obj.data = arr
                elif arr.size == 0:
                    obj.data = np.empty((9, 0), dtype=np.float64)

        for fp_key in ("full_filename", "file_path_", "file_path"):
            if fp_key in d and d[fp_key] and not obj.full_filename:
                val = d[fp_key]
                if isinstance(val, np.ndarray) and val.size > 0:
                    val = str(val.item())
                if isinstance(val, str):
                    obj.full_filename = val

        if "pix_range" in d and "metadata" not in d:
            pr = np.asarray(d["pix_range"], dtype=np.float64)
            if pr.shape == (2, 4):
                obj.pix_range = pr
        if "data_range" in d and "metadata" not in d:
            dr = np.asarray(d["data_range"], dtype=np.float64)
            if dr.shape == (2, 9):
                obj.data_range = dr

        return obj
