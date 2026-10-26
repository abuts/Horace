"""pixobj_combine_info class for combining multiple in-memory or file-backed PixelDataBase objects."""

from __future__ import annotations

from typing import Any, ClassVar, List, Sequence
import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.core.serializable import register_serializable
from pyhorace.data.pixels.multipix_base import MultipixBase
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


@register_serializable
class pixobj_combine_info(MultipixBase):
    """Combines multiple PixelDataBase objects and their associated npix bin arrays."""

    _positional_params: ClassVar[List[str]] = ["infiles", "npix_list", "run_label"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._npix_list: List[np.ndarray] = []

        if len(args) == 1 and not kwargs and isinstance(args[0], pixobj_combine_info):
            other = args[0]
            self.from_bare_struct(other.to_bare_struct())
            return
        if len(args) == 1 and not kwargs and isinstance(args[0], dict):
            self.from_bare_struct(args[0])
            return

        if args or kwargs:
            self._init_from_args(args, kwargs)

    def _init_from_args(self, args: Sequence[Any], kwargs: dict[str, Any]) -> None:
        self._do_check_combo_arg = False
        try:
            for idx, val in enumerate(args):
                if idx < len(self._positional_params):
                    setattr(self, self._positional_params[idx], val)
                else:
                    raise HoraceError(
                        "HORACE:pixobj_combine_info:invalid_argument",
                        f"Too many positional arguments ({len(args)}).",
                    )
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                else:
                    raise HoraceError(
                        "HORACE:pixobj_combine_info:invalid_argument",
                        f"Unknown parameter '{k}'.",
                    )
        finally:
            self._do_check_combo_arg = True
        self.check_combo_arg()

    def saveable_fields(self) -> List[str]:
        return ["infiles", "npix_list", "run_label"]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_filebacked(self) -> bool:
        return any(getattr(x, "is_filebacked", False) for x in self._infiles)

    def _set_infiles(self, val: Any) -> None:
        if val is None:
            self._infiles = []
            self._npix_each_file = np.empty(0, dtype=np.uint64)
            return
        if isinstance(val, PixelDataBase):
            objs = [val]
        elif isinstance(val, (list, tuple, np.ndarray)):
            objs = list(val)
        else:
            raise HoraceError(
                "HORACE:pixobj_combine_info:invalid_argument",
                f"infiles must be a PixelDataBase or list of PixelDataBase objects, got {type(val)}",
            )
        for obj in objs:
            if not isinstance(obj, PixelDataBase):
                raise HoraceError(
                    "HORACE:pixobj_combine_info:invalid_argument",
                    f"All elements of infiles must be PixelDataBase instances, got {type(obj)}",
                )
        self._infiles = objs
        self._npix_each_file = np.array([obj.num_pixels for obj in objs], dtype=np.uint64)

    def _set_npix_each_file(self, val: Any) -> None:
        raise HoraceError(
            "HORACE:pixobj_combine_info:runtime_error",
            "npix_each_file cannot be set directly on pixobj_combine_info; it is derived from infiles.",
        )

    @property
    def npix_list(self) -> List[np.ndarray]:
        return self._npix_list

    @npix_list.setter
    def npix_list(self, val: Any) -> None:
        if val is None:
            self._npix_list = []
        elif isinstance(val, np.ndarray) and val.dtype != object:
            arr = np.asarray(val, dtype=np.uint64).ravel()
            n_rep = max(1, self.nfiles)
            self._npix_list = [arr.copy() for _ in range(n_rep)]
            self._nbins = int(arr.size)
        elif isinstance(val, (list, tuple, np.ndarray)):
            items = list(val)
            if len(items) > 0 and all(isinstance(x, (int, float, np.number)) for x in items):
                arr = np.asarray(items, dtype=np.uint64).ravel()
                n_rep = max(1, self.nfiles)
                self._npix_list = [arr.copy() for _ in range(n_rep)]
                self._nbins = int(arr.size)
            else:
                self._npix_list = [np.asarray(x, dtype=np.uint64).ravel() for x in items]
                if self._npix_list:
                    self._nbins = int(self._npix_list[0].size)
        else:
            raise HoraceError(
                "HORACE:pixobj_combine_info:invalid_argument",
                f"npix_list must be a list of numeric arrays or a single numeric array, got {type(val)}",
            )
        if self._do_check_combo_arg:
            self.check_combo_arg()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def check_combo_arg(self, recalc_data_range: bool = True) -> pixobj_combine_info:
        super().check_combo_arg(recalc_data_range=False)
        nf = self.nfiles
        if nf > 0 and len(self._npix_list) > 0:
            if len(self._npix_list) == 1 and nf > 1:
                self._npix_list = [self._npix_list[0].copy() for _ in range(nf)]
            if len(self._npix_list) != nf:
                raise HoraceError(
                    "HORACE:pixobj_combine_info:invalid_argument",
                    f"Number of npix arrays ({len(self._npix_list)}) does not match number of pixel objects ({nf}).",
                )
            nb0 = self._npix_list[0].size
            self._nbins = int(nb0)
            for idx, (pix_obj, npix_arr) in enumerate(zip(self._infiles, self._npix_list)):
                if npix_arr.size != nb0:
                    raise HoraceError(
                        "HORACE:pixobj_combine_info:invalid_argument",
                        f"All npix arrays must have the same number of bins ({nb0}), but element {idx} has {npix_arr.size}.",
                    )
                tot_npix = int(np.sum(npix_arr))
                if tot_npix != pix_obj.num_pixels:
                    raise HoraceError(
                        "HORACE:pixobj_combine_info:invalid_argument",
                        f"Sum of npix[{idx}] ({tot_npix}) does not match num_pixels ({pix_obj.num_pixels}) of input object {idx}.",
                    )
            if recalc_data_range:
                self.recalc_data_range()
        return self

    def init_pix_access(self) -> pixobj_combine_info:
        for obj in self._infiles:
            obj.reset()
        return self

    def get_dataset_page(
        self, n_dataset: int, pix_pos_start: int, npix_idx: Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve a page of pixel data and npix chunk from dataset n_dataset (1-based)."""
        idx = int(n_dataset) - 1
        if idx < 0 or idx >= self.nfiles:
            raise HoraceError(
                "HORACE:pixobj_combine_info:invalid_argument",
                f"Dataset index {n_dataset} out of range (1..{self.nfiles}).",
            )
        b0, b1 = int(npix_idx[0]), int(npix_idx[1])
        npix_chunk = self._npix_list[idx][b0 - 1 : b1].copy()
        n_pix_to_read = int(np.sum(npix_chunk))

        pix_obj: PixelDataBase = self._infiles[idx]
        p_start = int(pix_pos_start)
        p_end = p_start + n_pix_to_read - 1
        if n_pix_to_read == 0:
            pix_data = np.empty((self.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64)
        else:
            sub_pix = pix_obj.get_pix_in_ranges([p_start], [n_pix_to_read], keep_precision=False)
            pix_data = sub_pix.data.copy()

        if self._run_label == "filenum":
            pix_data[4, :] = self.filenum[idx]
        elif isinstance(self._run_label, np.ndarray):
            pix_data[4, :] += self._run_label[idx]

        return pix_data, npix_chunk

    def close_faccessors(self) -> None:
        for obj in self._infiles:
            if hasattr(obj, "deactivate"):
                obj.deactivate()

    def recalc_data_range(self, fields: Any = "all") -> pixobj_combine_info:
        global_range = np.copy(self.EMPTY_RANGE)
        for idx, obj in enumerate(self._infiles):
            if obj.num_pixels == 0:
                continue
            if not obj. is_range_valid():
                obj.recalc_data_range()
            dr = obj.data_range.copy()
            if self._run_label == "filenum":
                dr[:, 4] = self.filenum[idx]
            elif isinstance(self._run_label, np.ndarray):
                dr[:, 4] += self._run_label[idx]
            global_range[0, :] = np.minimum(global_range[0, :], dr[0, :])
            global_range[1, :] = np.maximum(global_range[1, :], dr[1, :])
        self.set_data_range(global_range)
        return self
