"""Orthogonal / non-orthogonal linear coordinate axes block (line_axes)."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from pyhorace.coord_transf._dependencies import HoraceError
from pyhorace.coord_transf.axes_block_base import EPS_SINGLE, AxesBlockBase
from pyhorace.coord_transf.expand_box import expand_box


class line_axes(AxesBlockBase):
    """Axes block for linear reciprocal space / energy coordinates (Q_h, Q_k, Q_l, En)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._hkle_axes_directions: Optional[np.ndarray] = None
        self._init_from_args(*args, **kwargs)

    @property
    def _default_img_range(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=float)

    @property
    def _p_defines_bin_centers(self) -> np.ndarray:
        return np.ones(4, dtype=bool)

    def saveable_fields(self) -> List[str]:
        fields = list(self._base_fields)
        if self.axes_directions_defined:
            fields.append("hkle_axes_directions")
        return fields

    @property
    def axes_directions_defined(self) -> bool:
        return self._hkle_axes_directions is not None and self._hkle_axes_directions.size > 0

    @property
    def hkle_axes_directions(self) -> np.ndarray:
        if self._hkle_axes_directions is None:
            return np.eye(4, dtype=float)
        return self._hkle_axes_directions.copy()

    @hkle_axes_directions.setter
    def hkle_axes_directions(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and len(val) == 0):
            self._hkle_axes_directions = None
            return
        try:
            arr = np.asarray(val, dtype=float)
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                "HORACE:line_axes:invalid_argument",
                "hkle_axes_directions must be a 3x3 or 4x4 numeric matrix",
            ) from exc
        if arr.shape == (3, 3):
            mat = np.eye(4, dtype=float)
            mat[:3, :3] = arr
            self._hkle_axes_directions = mat
        elif arr.shape == (4, 4):
            self._hkle_axes_directions = arr.copy()
        else:
            raise HoraceError(
                "HORACE:line_axes:invalid_argument",
                f"hkle_axes_directions must be a 3x3 or 4x4 numeric matrix, got {arr.shape}",
            )

    def get_char_size(self, this_proj: Any = None) -> np.ndarray:
        """Compute characteristic bin size in image or pixel coordinates."""
        sz = (self._img_range[1, :] - self._img_range[0, :]) / self._nbins_all_dims
        if this_proj is None:
            return sz
        q_min = self._img_range[0, :3]
        q_max = q_min + sz[:3]
        img_coords = expand_box(q_min, q_max)
        pix_coords = this_proj.transform_img_to_pix(img_coords)
        sz_3d = np.max(pix_coords[:3, :], axis=1) - np.min(pix_coords[:3, :], axis=1)
        de = float(self._img_range[1, 3] - self._img_range[0, 3])
        return np.array([sz_3d[0], sz_3d[1], sz_3d[2], de], dtype=float)

    def data_plot_titles(
        self, data_obj: Any = None
    ) -> Tuple[str, List[str], List[str]]:
        """Generate (title_main, title_pax, title_iax) for linear axes."""
        dirs = self.hkle_axes_directions

        def _fmt_num(x: float) -> str:
            if abs(x) < 1e-6:
                return "0"
            if abs(x - round(x)) < 1e-6:
                return f"{int(round(x))}"
            return f"{x:.3g}"

        def _fmt_vec_with_var(off_vec: np.ndarray, dir_vec: np.ndarray, lbl: str) -> str:
            terms = []
            for j in range(3):
                o_j = float(off_vec[j])
                d_j = float(dir_vec[j])
                has_o = abs(o_j) > 1e-6
                has_d = abs(d_j) > 1e-6
                if not has_o and not has_d:
                    terms.append("0")
                elif has_o and not has_d:
                    terms.append(_fmt_num(o_j))
                elif not has_o and has_d:
                    if abs(d_j - 1.0) < 1e-6:
                        terms.append(lbl)
                    elif abs(d_j + 1.0) < 1e-6:
                        terms.append(f"-{lbl}")
                    else:
                        terms.append(f"{_fmt_num(d_j)}{lbl}")
                else:
                    if d_j > 0:
                        d_part = lbl if abs(d_j - 1.0) < 1e-6 else f"{_fmt_num(d_j)}{lbl}"
                        terms.append(f"{_fmt_num(o_j)}+{d_part}")
                    else:
                        d_part = lbl if abs(d_j + 1.0) < 1e-6 else f"{_fmt_num(abs(d_j))}{lbl}"
                        terms.append(f"{_fmt_num(o_j)}-{d_part}")
            return "[" + ", ".join(terms) + "]"

        title_pax: List[str] = []
        pax_ordered = self.pax[self.dax - 1] if self.pax.size > 0 else np.array([], dtype=int)

        # Compute base offset in hkl from offset + integrated Q axes midpoints
        base_hkl_off = dirs[:3, :3] @ self._offset[:3]
        for iax_num in self.iax:
            idx_i = int(iax_num) - 1
            if idx_i < 3:
                mid_i = 0.5 * (self._img_range[0, idx_i] + self._img_range[1, idx_i])
                base_hkl_off = base_hkl_off + dirs[:3, idx_i] * mid_i

        for ax_num in pax_ordered:
            idx = int(ax_num) - 1
            lbl = self._label[idx]
            u_val = float(self._ulen[idx])
            if idx < 3:
                dir_vec = dirs[:3, idx]
                vec_str = _fmt_vec_with_var(base_hkl_off, dir_vec, lbl)
                title_pax.append(f"{vec_str} in {u_val:.3g} \\AA^{{-1}}")
            else:
                off_e = float(self._offset[3])
                prefix = f"{_fmt_num(off_e)} + " if abs(off_e) > 1e-6 else ""
                title_pax.append(f"{prefix}{lbl} (meV)")

        title_iax: List[str] = []
        for ax_num in self.iax:
            idx = int(ax_num) - 1
            lbl = self._label[idx]
            r0 = float(self._img_range[0, idx])
            r1 = float(self._img_range[1, idx])
            if idx < 3:
                dir_vec = dirs[:3, idx]
                vec_str = _fmt_vec_with_var(np.zeros(3), dir_vec, lbl)
                title_iax.append(f"{r0:.3g} \\leq {vec_str} \\leq {r1:.3g}")
            else:
                off_e = float(self._offset[3])
                title_iax.append(
                    f"{r0 + off_e:.3g} \\leq {lbl} \\leq {r1 + off_e:.3g} (meV)"
                )

        main_lines = []
        if self._filename:
            if self._filepath:
                main_lines.append(f"{self._filepath}\\{self._filename}")
            else:
                main_lines.append(self._filename)
        if self._title:
            main_lines.append(self._title)
        if not np.allclose(self._offset[:3], 0.0):
            off_terms = [_fmt_num(x) for x in base_hkl_off]
            main_lines.append("Offset: [" + ", ".join(off_terms) + "]")
        if title_iax:
            main_lines.append(", ".join(title_iax))
        title_main = "\n".join(main_lines) if main_lines else ""

        return title_main, title_pax, title_iax

    @staticmethod
    def get_from_old_data(inputs: dict) -> "line_axes":
        """Reconstruct a line_axes object from legacy Horace data struct."""
        ax = line_axes()
        d = dict(inputs)

        img_range = np.zeros((2, 4), dtype=float)
        nbins = np.ones(4, dtype=int)
        sbd = np.ones(4, dtype=bool)

        pax = np.asarray(d.get("pax", []), dtype=int).ravel()
        iax = np.asarray(d.get("iax", []), dtype=int).ravel()
        p_cell = d.get("p", [])
        if pax.size == 1 and isinstance(p_cell, np.ndarray) and p_cell.dtype != object:
            p_cell = [p_cell]
        iint = np.asarray(d.get("iint", np.zeros((2, 0))), dtype=float)
        if iint.ndim == 1 and iint.size == 2 and iax.size == 1:
            iint = iint.reshape((2, 1))

        for k, ax_num in enumerate(pax):
            idx = int(ax_num) - 1
            edges = np.asarray(p_cell[k], dtype=float).ravel()
            img_range[0, idx] = float(edges[0])
            img_range[1, idx] = float(edges[-1])
            nbins[idx] = max(edges.size - 1, 1)
            sbd[idx] = False

        for k, ax_num in enumerate(iax):
            idx = int(ax_num) - 1
            if iint.ndim == 2 and k < iint.shape[1]:
                img_range[0, idx] = float(iint[0, k])
                img_range[1, idx] = float(iint[1, k])
            nbins[idx] = 1
            sbd[idx] = True

        if pax.size == 0 and iax.size == 0:
            if "img_db_range" in d and d["img_db_range"] is not None:
                db_r = np.asarray(d["img_db_range"], dtype=float)
                if db_r.shape == (2, 4) and not np.any(np.isinf(db_r)):
                    img_range = db_r.copy()
            elif "img_range" in d and d["img_range"] is not None:
                db_r = np.asarray(d["img_range"], dtype=float)
                if db_r.shape == (2, 4) and not np.any(np.isinf(db_r)):
                    img_range = db_r.copy()
        elif np.any(np.isinf(img_range)):
            inf_mask = np.isinf(img_range)
            if "img_db_range" in d and d["img_db_range"] is not None:
                db_r = np.asarray(d["img_db_range"], dtype=float)
                if db_r.shape == (2, 4):
                    img_range[inf_mask] = db_r[inf_mask]
            elif "img_range" in d and d["img_range"] is not None:
                db_r = np.asarray(d["img_range"], dtype=float)
                if db_r.shape == (2, 4):
                    img_range[inf_mask] = db_r[inf_mask]

        ax._do_check_combo_arg = True
        try:
            ax._set_img_range(img_range)
            ax._set_nbins_all_dims(nbins)
            ax._single_bin_defines_iax = sbd
            if "dax" in d and d["dax"] is not None:
                ax.dax = d["dax"]
            if "title" in d and d["title"] is not None:
                ax.title = d["title"]
            if "filename" in d and d["filename"] is not None:
                ax.filename = d["filename"]
            if "filepath" in d and d["filepath"] is not None:
                ax.filepath = d["filepath"]
            if "label" in d and d["label"] is not None:
                ax.label = [str(x).strip() for x in np.asarray(d["label"]).ravel()]
            elif "ulabel" in d and d["ulabel"] is not None:
                ax.label = [str(x).strip() for x in np.asarray(d["ulabel"]).ravel()]
            if "ulen" in d and d["ulen"] is not None:
                ax.ulen = d["ulen"]
            if "uoffset" in d and d["uoffset"] is not None:
                ax.offset = d["uoffset"]
            elif "offset" in d and d["offset"] is not None:
                ax.offset = d["offset"]
        finally:
            ax._do_check_combo_arg = False
        ax.check_combo_arg()
        return ax

    def from_bare_struct(self, s: dict) -> "line_axes":
        if "p" in s and "pax" in s and "img_range" not in s:
            loaded = self.get_from_old_data(s)
            for fld in self.saveable_fields():
                if hasattr(loaded, fld):
                    setattr(self, fld, getattr(loaded, fld))
            self.check_combo_arg()
            return self
        super().from_bare_struct(s)
        if "hkle_axes_directions" in s and s["hkle_axes_directions"] is not None:
            self.hkle_axes_directions = s["hkle_axes_directions"]
        return self
