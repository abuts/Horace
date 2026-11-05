"""Single-page binary operation between an SQW/PixelData object and an image-sized operand.

Ported from ``horace_core/sqw/page_operations/PageOp_binary_sqw_img.m``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import numpy as np

from pyhorace.coord_transf.page_operations.page_op_bin_base import PageOp_bin_Base
from pyhorace.core.configurations.config_store import config_store
from pyhorace.core.configurations.log_config import log_config
from pyhorace.core.data_op_interface import data_op_interface
from pyhorace.core.exceptions import HoraceError
from pyhorace.data.ix_dataset import IX_dataset
from pyhorace.data.sigvar import sigvar
from pyhorace.sqw.dnd_base import DnDBase


class PageOp_binary_sqw_img(PageOp_bin_Base):
    """Single page pixel operation between an ``sqw``/``PixelData`` and an image-like operand."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.keep_array: np.ndarray = np.array([], dtype=bool)

    def init(
        self,
        w1: Any,
        operand: Any = None,
        operation: Optional[Callable[..., Any]] = None,
        flip: bool = False,
        npix: Any = None,
    ) -> "PageOp_binary_sqw_img":
        """Initialize binary operation between ``w1`` and image-sized ``operand``."""
        _, name1_op = super().init(w1, operand, operation, flip, npix, nargout=2)

        is_num_arr = (
            isinstance(operand, (np.ndarray, list, tuple, int, float, np.number))
            and not isinstance(operand, bool)
            and (
                not isinstance(operand, np.ndarray)
                or (np.issubdtype(operand.dtype, np.number) and operand.dtype != bool)
            )
        )
        if is_num_arr:
            name2_op = "image-size array"
        else:
            name2_op = type(operand).__name__
        self.set_op_name(name1_op, name2_op)

        npix_provided = npix is not None and np.asarray(npix).size > 0

        if isinstance(self.operand, DnDBase):
            if npix_provided:
                raise HoraceError(
                    "HORACE:PageOp_binary_sqw_img:invalid_argument - "
                    "external npix input and second operand of DnD-type are not compatible"
                )
            self.keep_array = np.asarray(self.operand.npix, dtype=bool).ravel(order="F")
            if self.npix.size == 1:
                if int(np.sum(self.operand.npix)) != self._pix.num_pixels:
                    nobj1_elements = self._pix.num_pixels
                    nobj2_elements = int(np.sum(self.operand.npix))
                    raise HoraceError(
                        f"HORACE:PageOp_binary_sqw_img:invalid_argument - "
                        f"{self._op_name} attempted between inconsistent objects.\n"
                        f"{name1_op} contains {nobj1_elements} pixels and obj {name2_op} addresses {nobj2_elements} pixels"
                    )
                self.npix = np.asarray(self.operand.npix, dtype=np.int64).ravel(order="F")
                if not self.changes_pix_only:
                    self._var_acc = np.zeros(self.npix.size, dtype=np.float64)
            nobj2_elements = int(np.asarray(self.operand.npix).size)
        else:
            self.keep_array = np.ones(self.npix.shape, dtype=bool)
            if isinstance(self.operand, sigvar):
                if self.operand.is_mask_defined:
                    self.keep_array = np.asarray(self.operand.mask, dtype=bool).ravel(order="F")
                nobj2_elements = int(self.operand.n_elements)
            elif isinstance(self.operand, IX_dataset) or is_num_arr:
                self.operand = sigvar(self.operand)
                nobj2_elements = int(self.operand.n_elements)
            else:
                raise HoraceError(
                    f"HORACE:PageOp_binary_sqw_img:invalid_argument - "
                    f"Unsupported class {type(self.operand).__name__} of second operand for operation {self.op_name}"
                )

        nobj1_elements = int(self.npix.size)
        if nobj1_elements != nobj2_elements:
            name1_obj = type(w1).__name__
            name2_obj = type(operand).__name__
            raise HoraceError(
                f"HORACE:PageOp_binary_sqw_img:invalid_argument - "
                f"{self._op_name} attempted between inconsistent objects. "
                f"Image of operand1: {name1_obj} has {nobj1_elements} elements "
                f"and img of op2 {name2_obj} has {nobj2_elements} elements"
            )
        return self

    def apply_op(
        self, npix_block: np.ndarray, npix_idx: np.ndarray
    ) -> "PageOp_binary_sqw_img":
        """Perform binary operation between input object and image-like operand."""
        idx0 = int(npix_idx[0]) - 1
        idx1 = int(npix_idx[1])
        npix_blk = np.asarray(npix_block, dtype=np.int64).ravel(order="F").copy()

        keep_page_bins = np.asarray(self.keep_array[idx0:idx1], dtype=bool).ravel(order="F")
        keep_pix = np.repeat(keep_page_bins, npix_blk)
        npix_blk[~keep_page_bins] = 0

        page_data = self._page_data[:, keep_pix]
        self.sigvar1.sig_var = page_data[self._sigvar_idx, :]

        op_s_flat = np.asarray(self.operand.s, dtype=np.float64).ravel(order="F")
        fp_sig = np.repeat(op_s_flat[idx0:idx1], npix_blk)
        if getattr(self.operand, "e", None) is not None and np.asarray(self.operand.e).size > 0:
            op_e_flat = np.asarray(self.operand.e, dtype=np.float64).ravel(order="F")
            fp_var = np.repeat(op_e_flat[idx0:idx1], npix_blk)
            fake_pix = sigvar(fp_sig.reshape(1, -1), fp_var.reshape(1, -1))
        else:
            fake_pix = sigvar(fp_sig.reshape(1, -1), None)

        if self.flip:
            res = data_op_interface.binary_op_manager_single(
                fake_pix, self.sigvar1, self.op_handle
            )
        else:
            res = data_op_interface.binary_op_manager_single(
                self.sigvar1, fake_pix, self.op_handle
            )

        page_data[self._sigvar_idx, :] = res.sig_var
        self._page_data = page_data
        if self.changes_pix_only:
            return self

        self.update_img_accumulators(npix_blk, npix_idx, res.s, res.e)
        return self

    def finish_op(self, in_obj: Any) -> Tuple[Any, "PageOp_binary_sqw_img"]:
        """Reduce total number of pixels in final image for masked bins."""
        self._npix[~self.keep_array] = 0
        return super().finish_op(in_obj)

    def get_info_split_log_ratio(self) -> float:
        return float(
            config_store.instance().get_value("log_config", "sqw_binary_img_split_ratio")
        )

    def set_info_split_log_ratio(self, val: float) -> "PageOp_binary_sqw_img":
        log = log_config()
        log.sqw_binary_img_split_ratio = val
        return self

