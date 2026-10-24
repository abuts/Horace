"""3-dimensional Horace DnD dataset class (d3d).

Ported from Horace MATLAB @d3d class (horace_core/sqw/@d3d/d3d.m).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

import numpy as np

from pyhorace.core.serializable import register_serializable
from pyhorace.sqw.dnd_base import DnDBase


@register_serializable
class d3d(DnDBase):
    """3-dimensional Horace DnD dataset."""

    NUM_DIMS: ClassVar[int] = 3

    def IX_dataset_3d(self) -> Dict[str, Any]:
        """Convert 3D DnD object into IX_dataset_3d dictionary representation."""
        title_main, title_pax, *_ = self.data_plot_titles()
        nopix = self._npix == 0
        signal = np.array(self._s, dtype=float, copy=True)
        signal[nopix] = np.nan
        err = np.sqrt(np.maximum(self._e, 0.0))
        err[nopix] = 0.0
        p_list = self.p
        return {
            "title": title_main,
            "signal": signal,
            "error": err,
            "s_axis": "Intensity",
            "x": p_list[0].copy() if len(p_list) > 0 else np.array([], dtype=float),
            "x_axis": title_pax[0] if len(title_pax) > 0 else "",
            "x_distribution": True,
            "y": p_list[1].copy() if len(p_list) > 1 else np.array([], dtype=float),
            "y_axis": title_pax[1] if len(title_pax) > 1 else "",
            "y_distribution": True,
            "z": p_list[2].copy() if len(p_list) > 2 else np.array([], dtype=float),
            "z_axis": title_pax[2] if len(title_pax) > 2 else "",
            "z_distribution": True,
        }
