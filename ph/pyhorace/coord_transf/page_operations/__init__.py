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
from pyhorace.coord_transf.page_operations.page_op_apply import PageOp_apply
from pyhorace.coord_transf.page_operations.page_op_base import PageOpBase
from pyhorace.coord_transf.page_operations.page_op_bin_base import PageOp_bin_Base
from pyhorace.coord_transf.page_operations.page_op_binary_sqw_double import (
    PageOp_binary_sqw_double,
)
from pyhorace.coord_transf.page_operations.page_op_binary_sqw_img import (
    PageOp_binary_sqw_img,
)
from pyhorace.coord_transf.page_operations.page_op_binary_sqw_sqw import (
    PageOp_binary_sqw_sqw,
)
from pyhorace.coord_transf.page_operations.page_op_cat_pix import PageOp_cat_pix
from pyhorace.coord_transf.page_operations.page_op_coord_calc import (
    PageOp_coord_calc,
)
from pyhorace.coord_transf.page_operations.page_op_func_eval import PageOp_func_eval
from pyhorace.coord_transf.page_operations.page_op_join_sqw import PageOp_join_sqw
from pyhorace.coord_transf.page_operations.page_op_mask import PageOp_mask
from pyhorace.coord_transf.page_operations.page_op_noisify import PageOp_noisify
from pyhorace.coord_transf.page_operations.page_op_recompute_bins import (
    PageOp_recompute_bins,
)
from pyhorace.coord_transf.page_operations.page_op_section import PageOp_section
from pyhorace.coord_transf.page_operations.page_op_sigvar_set import (
    PageOp_sigvar_set,
)
from pyhorace.coord_transf.page_operations.page_op_split_sqw import PageOp_split_sqw
from pyhorace.coord_transf.page_operations.page_op_sqw_binning import (
    PageOp_sqw_binning,
)
from pyhorace.coord_transf.page_operations.page_op_sqw_eval import PageOp_sqw_eval
from pyhorace.coord_transf.page_operations.page_op_sqw_op import PageOp_sqw_op
from pyhorace.coord_transf.page_operations.page_op_unary_op import PageOp_unary_op

__all__ = [
    "PageOpBase",
    "PageOp_apply",
    "PageOp_bin_Base",
    "PageOp_binary_sqw_double",
    "PageOp_binary_sqw_img",
    "PageOp_binary_sqw_sqw",
    "PageOp_cat_pix",
    "PageOp_coord_calc",
    "PageOp_func_eval",
    "PageOp_join_sqw",
    "PageOp_mask",
    "PageOp_noisify",
    "PageOp_recompute_bins",
    "PageOp_section",
    "PageOp_sigvar_set",
    "PageOp_split_sqw",
    "PageOp_sqw_binning",
    "PageOp_sqw_eval",
    "PageOp_sqw_op",
    "PageOp_unary_op",
    "average_bin_data",
    "compute_bin_data",
    "normalize_signal",
    "split_vector_fixed_sum",
    "split_vector_max_sum",
]
