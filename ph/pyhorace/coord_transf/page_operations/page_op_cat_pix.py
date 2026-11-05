"""PageOp_cat_pix class for concatenating multiple PixelDataBase objects.

Ported from ``horace_core/sqw/page_operations/PageOp_cat_pix.m``.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.data.pixels.pixel_data_base import PixelDataBase
from pyhorace.data.pixels.pixel_data_file_backed import PixelDataFileBacked
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory


class PageOp_cat_pix(PageOpBase):
    """Single page pixel operations specific for cat pixels algorithm.

    This version works with pixels only.
    Ported from ``horace_core/sqw/page_operations/PageOp_cat_pix.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.op_name_ = "cat_pixels"
        self.split_at_bin_edges_ = False
        self.in_objects: List[PixelDataBase] = []
        self.npix_tot: int = 0
        self.force_cat_in_memory: bool = False
        self.pix_block_start_: np.ndarray = np.empty(0, dtype=np.int64)
        self.block_idx_: np.ndarray = np.empty((2, 0), dtype=np.int64)

    @property
    def in_pix(self) -> List[PixelDataBase]:
        """Alias for ``in_objects``."""
        return self.in_objects

    def init(self, *args: Any) -> "PageOp_cat_pix":
        """Initialize cat pix operations.

        Parameters
        ----------
        *args : PixelDataBase
            Sequence of ``PixelDataBase`` objects to concatenate together.
        """
        self.init_pix_only_data_obj(*args)

        mem_chunk_size = config_store.instance().get_value(
            "hor_config", "mem_chunk_size"
        )
        pf = config_store.instance().get_value("hor_config", "fb_scale_factor")
        fb_pix_limit = pf * mem_chunk_size
        if self.npix_tot > fb_pix_limit and not self.force_cat_in_memory:
            in_obj: PixelDataBase = PixelDataFileBacked()
            self.inform_about_target_file = True
        else:
            in_obj = PixelDataMemory()
            self.inform_about_target_file = False

        saved_npix = self.npix_.copy() if self.npix_ is not None else None
        super().init(in_obj)
        if saved_npix is not None:
            self.npix_ = saved_npix
        return self

    def split_into_pages(
        self, npix: np.ndarray, chunk_size: Union[int, float]
    ) -> Tuple[List[np.ndarray], np.ndarray, "PageOp_cat_pix"]:
        """Overload of ``split_into_pages`` as we also need ``npix_idx`` to process pages."""
        npix_flat = np.asarray(npix, dtype=np.int64).ravel(order="F")
        if isinstance(self.pix_, PixelDataMemory):
            npix_chunks = [npix_flat]
            npix_idx = np.array([[1], [npix_flat.size]], dtype=np.int64)
        else:
            npix_chunks, npix_idx, _ = super().split_into_pages(
                npix_flat, chunk_size
            )
        self.block_idx_ = npix_idx
        return npix_chunks, npix_idx, self

    def get_page_data(
        self, page_idx: int, npix_blocks: Sequence[np.ndarray]
    ) -> "PageOp_cat_pix":
        """Return block of data used in page operation.

        Here each ``npix`` element contains the number of pixels in an input
        pixel dataset and ``block_idx_`` defines how many datasets the page of
        pixels occupies.
        """
        chunks = np.asarray(npix_blocks[page_idx - 1], dtype=np.int64).ravel(
            order="F"
        )
        n_chunks = chunks.size
        npix_idx = self.block_idx_[:, page_idx - 1]

        if n_chunks == 1:
            obj_idx = int(npix_idx[0]) - 1
            c0 = int(chunks[0])
            pix_idx_start = int(self.pix_block_start_[obj_idx])
            pix_idx_end = pix_idx_start + c0 - 1
            if c0 == 0:
                self.page_data_ = np.empty(
                    (PixelDataBase.DEFAULT_NUM_PIX_FIELDS, 0), dtype=np.float64
                )
            else:
                indices_1based = np.arange(
                    pix_idx_start, pix_idx_end + 1, dtype=np.int64
                )
                self.page_data_ = np.asarray(
                    self.in_objects[obj_idx].get_pixels(
                        indices_1based, "-raw", "-align"
                    ),
                    dtype=np.float64,
                )
            self.pix_block_start_[obj_idx] = pix_idx_end + 1
        else:
            this_chunk = int(np.sum(chunks))
            self.page_data_ = np.zeros(
                (PixelDataBase.DEFAULT_NUM_PIX_FIELDS, this_chunk),
                dtype=np.float64,
            )
            page_idx_start = 0
            n_accessor = list(range(int(npix_idx[0]) - 1, int(npix_idx[1])))
            for i in range(n_chunks):
                obj_idx = n_accessor[i]
                c_i = int(chunks[i])
                pix_idx_start = int(self.pix_block_start_[obj_idx])
                pix_idx_end = pix_idx_start + c_i - 1
                accessor = self.in_objects[obj_idx]
                page_idx_end = page_idx_start + c_i
                if c_i > 0:
                    indices_1based = np.arange(
                        pix_idx_start, pix_idx_end + 1, dtype=np.int64
                    )
                    self.page_data_[:, page_idx_start:page_idx_end] = (
                        np.asarray(
                            accessor.get_pixels(
                                indices_1based, "-raw", "-align"
                            ),
                            dtype=np.float64,
                        )
                    )
                self.pix_block_start_[obj_idx] = pix_idx_end + 1
                page_idx_start = page_idx_end
        return self

    def apply_op(self, *args: Any) -> "PageOp_cat_pix":
        """Cat does not change pixels."""
        return self

    # =========================================================================
    # Protected methods
    # =========================================================================

    def init_pix_only_data_obj(self, *args: Any) -> "PageOp_cat_pix":
        """Process and prepare input sequence of PixelDataBase objects."""
        n_inputs = len(args)
        self.in_objects = []
        npix_arr = np.zeros(n_inputs, dtype=np.int64)
        self.pix_block_start_ = np.ones(n_inputs, dtype=np.int64)
        npix_tot_ = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, PixelDataBase):
                raise HoraceError(
                    "HORACE:PixelDataBase:invalid_argument",
                    f"cat requested arguments are PixelDatBase sub-classes. "
                    f"Class of the input N{i + 1} is: {type(arg).__name__}",
                )
            self.in_objects.append(arg)
            npix = int(arg.num_pixels)
            npix_arr[i] = npix
            npix_tot_ += npix
        self.npix = npix_arr
        self.npix_tot = npix_tot_
        return self

    def get_exp_modified(self) -> bool:
        """Control calculations of unique run_ids during page_op."""
        return self.img_ is not None

    def get_changes_pix_only(self) -> bool:
        """PageOp calculates pixels only."""
        return True

    def get_info_split_log_ratio(self) -> int:
        return int(
            config_store.instance().get_value(
                "log_config", "cat_pix_split_ratio"
            )
        )

    def set_info_split_log_ratio(self, val: Union[int, float]) -> None:
        log = log_config()
        log.cat_pix_split_ratio = int(val)


__all__ = ["PageOp_cat_pix"]
