"""Single pixel page operation used by ``mask`` and ``mask_pixels`` methods.

Ported from ``horace_core/sqw/page_operations/PageOp_mask.m``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_base import PixelDataBase


class PageOp_mask(PageOpBase):
    """Single page pixel operation used by ``mask``, ``mask_pixels``,
    ``mask_random_fraction_pixels``, and ``mask_random_pixels`` methods.

    Ported from ``horace_core/sqw/page_operations/PageOp_mask.m``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.keep_info_obj: Any = None
        self.mask_by_obj: bool = False
        self.mask_by_bins: bool = False
        self.mask_by_num: bool = False
        self.npix_acc: np.ndarray = np.empty(0, dtype=np.int64)
        self.check_runid: bool = False
        super().__init__(*args, **kwargs)
        self._op_name = "masking"

    @property
    def num_pix_original(self) -> int:
        """Number of pixels in the masked dataset."""
        return int(self._pix.num_pixels)

    @property
    def num_bins(self) -> int:
        """Number of bins in the masked dataset."""
        return int(np.asarray(self.npix).size)

    def init(self, in_obj: Any = None, keep_info: Any = None) -> "PageOp_mask":
        """Initialize ``PageOp_mask`` with ``in_obj`` and ``keep_info``."""
        super().init(in_obj)
        if in_obj is None:
            return self

        from pyhorace.sqw.sqw_dnd_base import SQWDnDBase

        if isinstance(keep_info, SQWDnDBase):
            self.keep_info_obj = keep_info.pix
        elif isinstance(keep_info, PixelDataBase):
            self.keep_info_obj = keep_info
        else:
            self.keep_info_obj = np.asarray(keep_info).ravel(order="F")

        if self._img is not None:
            nb = int(np.asarray(self._img.npix).size)
            self._var_acc = np.zeros(nb, dtype=np.float64, order="F")
            self.npix_acc = np.zeros(nb, dtype=np.int64, order="F")
            self.check_runid = True
        else:
            self.check_runid = False

        if isinstance(self.keep_info_obj, PixelDataBase):
            self.mask_by_obj = True
            self.mask_by_bins = False
            self.mask_by_num = False
            self.split_at_bin_edges = False
        elif isinstance(
            keep_info, (int, float, np.number, bool, np.bool_, list, tuple, np.ndarray)
        ):
            ki_arr = np.asarray(keep_info)
            is_scalar_num = (
                ki_arr.size == 1
                and not isinstance(keep_info, (bool, np.bool_))
                and not np.issubdtype(ki_arr.dtype, np.bool_)
            )
            if is_scalar_num:
                self.mask_by_obj = False
                self.mask_by_bins = False
                self.mask_by_num = True
                self.split_at_bin_edges = False
                num_keep_val = int(ki_arr.item())
                if self._pix.num_pages > 1:
                    self.keep_info_obj = self.calc_page_share(
                        num_keep_val, self.num_pix_original, self._pix.page_size
                    )
                else:
                    self.keep_info_obj = np.array([num_keep_val], dtype=np.int64)
            else:
                self.mask_by_obj = False
                self.mask_by_num = False
                if self.num_bins == ki_arr.size:
                    self.mask_by_bins = True
                    self.split_at_bin_edges = True
                elif self.num_pix_original == ki_arr.size:
                    self.mask_by_bins = False
                    self.split_at_bin_edges = False
                else:
                    raise HoraceError(
                        "HORACE:PageOp_mask:invalid_argument",
                        f"Number of masking elements in array must be equal to number of pixels "
                        f"({self.num_pix_original}) or number of bins ({self.num_bins}). "
                        f"It is {ki_arr.size}",
                    )
        else:
            raise HoraceError(
                "HORACE:PageOp_mask:invalid_argument",
                "keep_info can be either sqw/PixData object, containing information about masking\n"
                "or array/logical array or numbers specifying pixels/bins to mask",
            )
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_mask":
        """Apply pixel or bin mask to current page."""
        counts = np.asarray(npix_block, dtype=np.int64).ravel(order="F")
        if self.mask_by_num:
            num_keep_here = int(
                np.asarray(self.keep_info_obj).ravel(order="F")[self.page_num - 1]
            )
            num_pix_here = int(np.sum(counts))
            keep_pix = np.zeros(num_pix_here, dtype=bool)
            if num_keep_here > 0 and num_pix_here > 0:
                chosen = np.random.choice(
                    num_pix_here, size=min(num_keep_here, num_pix_here), replace=False
                )
                keep_pix[chosen] = True
        elif self.mask_by_bins:
            idx0 = int(npix_idx[0]) - 1
            idx1 = int(npix_idx[1])
            bins_keep = np.asarray(self.keep_info_obj[idx0:idx1], dtype=bool).ravel(
                order="F"
            )
            keep_pix = np.repeat(bins_keep, counts)
        elif self.mask_by_obj:
            self.keep_info_obj.page_num = self.page_num
            mask_data = np.asarray(
                self.keep_info_obj.signal, dtype=np.float64
            ).ravel(order="F")
            keep_pix = mask_data >= 1.0
        else:
            if hasattr(self._pix, "get_page_idx_"):
                pix_idx_start, pix_idx_end = self._pix.get_page_idx_()
            else:
                pix_idx_start, pix_idx_end = 1, int(self._pix.num_pixels)
            keep_pix = np.asarray(
                self.keep_info_obj[pix_idx_start - 1 : pix_idx_end], dtype=bool
            ).ravel(order="F")

        self._page_data = self._page_data[:, keep_pix]
        if self.changes_pix_only:
            return self

        nbins = int(counts.size)
        ibin = np.repeat(np.arange(nbins, dtype=np.int64), counts)
        npix_block_new = np.bincount(ibin[keep_pix], minlength=nbins).astype(np.int64)

        signal = self._page_data[self.signal_idx, :]
        error = self._page_data[self.var_idx, :]
        self.update_img_accumulators(npix_block_new, npix_idx, signal, error)

        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        self.npix_acc[idx0:idx1] = self.npix_acc[idx0:idx1] + npix_block_new
        return self

    def finish_op(self, out_obj: Any) -> tuple[Any, "PageOp_mask"]:
        """Update ``npix`` and ``experiment_info`` and finalize operation."""
        self.npix = self.npix_acc
        if not self.changes_pix_only:
            n_runs = int(getattr(out_obj.experiment_info, "n_runs", 0))
            if self._unique_run_id.size == n_runs:
                self.check_runid = False
            elif self._unique_run_id.size < n_runs and hasattr(
                out_obj.experiment_info, "get_subobj"
            ):
                out_obj.experiment_info = out_obj.experiment_info.get_subobj(
                    self._unique_run_id
                )
        res_obj, _ = super().finish_op(out_obj)
        return res_obj, self

    def calc_page_share(
        self, num_pix_to_keep: int, tot_num_pix: int, page_size: int
    ) -> np.ndarray:
        """Split total number of pixels to keep across pages."""
        pix_share = float(num_pix_to_keep) / float(tot_num_pix)
        pages_list, *_ = self.split_into_pages(tot_num_pix, page_size)
        pages = np.array(
            [int(np.asarray(p).item()) for p in pages_list], dtype=np.int64
        )
        num_pages = int(pages.size)
        pieces = np.floor(pages * pix_share).astype(np.int64)
        nothing = pieces == 0
        pieces[nothing] = pieces[nothing] + 1

        splitted = int(np.sum(pieces))
        ic = num_pages - 1
        while ic >= 0 and splitted < num_pix_to_keep:
            pieces[ic] = pieces[ic] + 1
            if pieces[ic] < pages[ic]:
                splitted = splitted + 1
            else:
                pieces[ic] = pages[ic]
            ic -= 1
        return pieces

    def get_exp_modified(self) -> bool:
        return bool(self._old_file_format or self.check_runid)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "mask_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_mask":
        log = log_config()
        log.mask_split_ratio = val
        return self
