"""Page operations for transforming SQW and PixelData objects page-by-page.

Ported from ``horace_core/sqw/page_operations/``.
"""

from pyhorace.coord_transf.page_operations._utils import (
    average_bin_data,
    compute_bin_data,
    normalize_signal,
    split_vector_fixed_sum,
    split_vector_max_sum,
)
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.coord_transf.page_operations.page_op_recompute_bins import (
    PageOp_recompute_bins,
)

__all__ = [
    "PageOpBase",
    "PageOp_recompute_bins",
    "average_bin_data",
    "compute_bin_data",
    "normalize_signal",
    "split_vector_fixed_sum",
    "split_vector_max_sum",
]
