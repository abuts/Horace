"""Single pixel page operation used by ``signal`` and ``coordinates_calc`` methods.

Ported from ``horace_core/sqw/page_operations/PageOp_coord_calc.m``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyhorace.coord_transf.page_operations._utils import compute_bin_data
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.pixels.pixel_data_memory import PixelDataMemory


class PageOp_coord_calc(PageOpBase):
    """Single pixel page operation used by ``signal`` and ``coordinates_calc`` methods.

    Ported from ``horace_core/sqw/page_operations/PageOp_coord_calc.m``.
    """

    xname: tuple[str, ...] = ("d1", "d2", "d3", "d4", "h", "k", "l", "E", "Q")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.op_type: str | None = None
        self.ind: int = 0
        self.proj: Any = None
        self.pix_page: PixelDataMemory | None = None
        super().__init__(*args, **kwargs)
        self.split_at_bin_edges = True

    def init(self, in_obj: Any = None, ind: int | str | None = None) -> "PageOp_coord_calc":
        """Initialize ``PageOp_coord_calc`` with ``in_obj`` and 1-based ``ind`` (or name)."""
        super().init(in_obj)
        if in_obj is None:
            return self
        self.split_at_bin_edges = True
        if self.changes_pix_only:
            raise HoraceError(
                "HORACE:PageOp_coord_calc:invalid_argument",
                "This method requests full sqw object as input but only pixels were provided",
            )

        if isinstance(ind, str):
            if ind not in self.xname:
                raise HoraceError(
                    "HORACE:sqw:invalid_argument",
                    f"Input parameter must be one or group of parameters from list {self.xname}\n"
                    f". It is: {ind}",
                )
            ind_val = self.xname.index(ind) + 1
        elif ind is not None:
            ind_val = int(ind)
        else:
            raise HoraceError(
                "HORACE:PageOp_coord_calc:invalid_argument",
                "Coordinate index or name is required.",
            )

        self.ind = ind_val
        self.op_type = self.xname[self.ind - 1]
        self.op_name_ = f'"set signal to {self.op_type}"'
        self.proj = self.img_.proj
        self.pix_page = PixelDataMemory()
        self.var_acc_ = np.zeros(
            int(np.asarray(self.npix).size), dtype=np.float64, order="F"
        )
        return self

    def get_page_data(
        self, idx: int, npix_blocks: Sequence[np.ndarray]
    ) -> "PageOp_coord_calc":
        """Load page data and wrap it in ``self.pix_page``."""
        super().get_page_data(idx, npix_blocks)
        if self.pix_page is None:
            self.pix_page = PixelDataMemory()
        self.pix_page.set_raw_data(self.page_data_)
        return self

    def update_img_accumulators(
        self,
        npix_block: np.ndarray,
        npix_idx: np.ndarray,
        signal: np.ndarray,
        variance: np.ndarray | None = None,
    ) -> "PageOp_coord_calc":
        """Compute bin sums and mean square deviations for pixel coordinates."""
        s_ar, v_ar, s_msd = compute_bin_data(
            npix_block,
            signal,
            None,
            no_normalization=True,
            calc_variance=True,
            calc_signal_msd=True,
        )
        if s_msd is not None and s_msd.size > 0:
            self.page_data_[self.var_idx, :] = s_msd
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        self.sig_acc_[idx0:idx1] = s_ar.ravel(order="F")
        if v_ar is not None:
            self.var_acc_[idx0:idx1] = v_ar.ravel(order="F")
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_coord_calc":
        """Set pixel signal to the requested coordinate and compute bin variance."""
        this_proj = self.proj
        ctype = self.op_type
        lind = self.ind

        if ctype in ("h", "k", "l"):
            get_ind = (lind - 1) % 4
            uhkl = this_proj.transform_pix_to_hkl(self.pix_page.q_coordinates)
            signal = uhkl[get_ind, :]
        elif ctype == "E":
            offset = np.asarray(self.proj.offset, dtype=np.float64).ravel(order="F")
            signal = (
                np.asarray(self.pix_page.dE, dtype=np.float64).ravel(order="F")
                + offset[3]
            )
        elif ctype == "Q":
            qq = np.asarray(self.pix_page.q_coordinates, dtype=np.float64)
            signal = np.linalg.norm(qq, ord=2, axis=0)
        elif ctype in ("d1", "d2", "d3", "d4"):
            pax = np.asarray(self.img_.pax, dtype=np.int64).ravel(order="F")
            dax = np.asarray(self.img_.dax, dtype=np.int64).ravel(order="F")
            d_idx = (lind - 1) % 4
            get_ind = int(pax[int(dax[d_idx]) - 1]) - 1
            uhkl = this_proj.transform_pix_to_img(self.pix_page.q_coordinates)
            signal = uhkl[get_ind, :]
        else:
            raise HoraceError(
                "HORACE:PageOp_coord_calc:invalid_argument",
                f"Unsupported coordinate type: {ctype}",
            )

        self.page_data_[self.signal_idx, :] = signal
        if self.changes_pix_only:
            return self
        self.update_img_accumulators(npix_block, npix_idx, signal)
        return self

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "coord_calc_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_coord_calc":
        log = log_config()
        log.coord_calc_split_ratio = val
        return self
