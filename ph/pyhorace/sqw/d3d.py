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
        p_raw = self.p
        p_full = [
            p_raw[k].copy() if k < len(p_raw) else np.array([-0.5, 0.5], dtype=float)
            for k in range(3)
        ]
        target_shape = tuple(max(len(p_k) - 1, 1) for p_k in p_full)
        if signal.ndim < 3 and signal.size == int(np.prod(target_shape)):
            signal = signal.reshape(target_shape, order="F")
            err = err.reshape(target_shape, order="F")
        dax_0 = [int(d) - 1 for d in self.dax]
        if len(dax_0) == 3 and signal.ndim == 3:
            p_perm = [p_full[idx] for idx in dax_0]
            signal = np.transpose(signal, dax_0)
            err = np.transpose(err, dax_0)
        else:
            p_perm = p_full
        return {
            "title": title_main,
            "signal": signal,
            "error": err,
            "s_axis": "Intensity",
            "x": p_perm[0],
            "x_axis": title_pax[0] if len(title_pax) > 0 else "",
            "x_distribution": True,
            "y": p_perm[1],
            "y_axis": title_pax[1] if len(title_pax) > 1 else "",
            "y_distribution": True,
            "z": p_perm[2],
            "z_axis": title_pax[2] if len(title_pax) > 2 else "",
            "z_distribution": True,
        }
