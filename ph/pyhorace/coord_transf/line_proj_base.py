"""Base class for linear/rectilinear coordinate projections (LineProjBase)."""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf._dependencies import AxesBlockBase, HoraceError, PixelDataBase, line_axes
from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.bmatrix import _infer_nargout, bmatrix
from pyhorace.coord_transf.ubmatrix import ubmatrix
from pyhorace.core.configurations.config_store import config_store


class LineProjBase(aProjectionBase):
    """Base class for rectilinear projections (line_proj and ubmat_proj)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._u_to_rlu: Optional[np.ndarray] = None
        self._u_to_rlu_legacy: Optional[np.ndarray] = None
        self._ulen: Optional[np.ndarray] = None
        self._ortho_ortho_transf_mat: Optional[np.ndarray] = None
        self._ortho_ortho_offset: Optional[np.ndarray] = None
        self._convert_targ_to_source: Optional[bool] = None
        if args or kwargs:
            self.init(*args, **kwargs)

    # ------------------------------------------------------------------
    # Properties & Legacy Aliases
    # ------------------------------------------------------------------
    @property
    def uoffset(self) -> np.ndarray:
        """Legacy alias for offset."""
        return self.offset

    @uoffset.setter
    def uoffset(self, val: Any) -> None:
        self.offset = val

    @property
    def u_to_rlu(self) -> np.ndarray:
        """4x4 matrix converting image coordinates into rlu + energy."""
        if not self.lattice_defined:
            return np.array([], dtype=float)
        _, u_to_rlu = self.get_pix_img_transformation(4, nargout=2)
        return u_to_rlu

    @property
    def u_to_rlu_legacy(self) -> np.ndarray:
        if self._u_to_rlu_legacy is not None:
            return self._u_to_rlu_legacy.copy()
        if not self.lattice_defined:
            return np.array([], dtype=float)
        return self.u_to_rlu

    @u_to_rlu_legacy.setter
    def u_to_rlu_legacy(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            self._u_to_rlu_legacy = None
            return
        arr = np.asarray(val, dtype=float)
        if arr.shape == (3, 3):
            m4 = np.eye(4, dtype=float)
            m4[:3, :3] = arr
            arr = m4
        self._u_to_rlu_legacy = arr.copy()

    @property
    def ulen(self) -> np.ndarray:
        """4-element vector of unit lengths along image axes in inverse Angstroms / meV."""
        if self._ulen is not None and not self.lattice_defined:
            return self._ulen.copy()
        return self.img_scales

    @ulen.setter
    def ulen(self, val: Any) -> None:
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size == 3:
            arr = np.append(arr, 1.0)
        self._ulen = arr.copy()
        self._img_scales = arr.copy()

    @property
    def compat_struct(self) -> dict[str, Any]:
        return {
            "alatt": self.alatt,
            "angdeg": self.angdeg,
            "uoffset": self.offset,
            "ulabel": self.label,
            "ulen": self.ulen,
            "u_to_rlu": self.u_to_rlu,
        }

    @property
    def convert_targ_to_source(self) -> Optional[bool]:
        return self._convert_targ_to_source

    @convert_targ_to_source.setter
    def convert_targ_to_source(self, val: Optional[bool]) -> None:
        self._convert_targ_to_source = None if val is None else bool(val)

    @property
    def can_mex_cut(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Transformation Matrices & Coordinate Transformations
    # ------------------------------------------------------------------
    @abstractmethod
    def get_pix_img_transformation(
        self,
        ndim: int,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Return transformation matrix/offset between pixel (Crystal Cartesian) and image space."""

    def _check_and_set_targ_proj_optimization(self, targ: Optional[aProjectionBase]) -> None:
        if targ is None or self._disable_srce_to_targ_optimization or not isinstance(targ, LineProjBase):
            self._ortho_ortho_transf_mat = None
            self._ortho_ortho_offset = None
            return
        if not self.lattice_defined or not targ.lattice_defined:
            self._ortho_ortho_transf_mat = None
            self._ortho_ortho_offset = None
            return
        # Direct affine transformation from this image space to targ image space:
        # q_cc = (pix_to_img_self)^(-1) @ x_self + offset_cc_self
        # x_targ = pix_to_img_targ @ (q_cc - offset_cc_targ)
        mat_self, off_cc_self, _ = self.get_pix_img_transformation(4, nargout=3)
        mat_targ, off_cc_targ, _ = targ.get_pix_img_transformation(4, nargout=3)
        # x_self = mat_self @ (q_cc - off_cc_self) => q_cc = inv(mat_self) @ x_self + off_cc_self
        # x_targ = mat_targ @ (inv(mat_self) @ x_self + off_cc_self - off_cc_targ)
        inv_self = np.linalg.inv(mat_self)
        self._ortho_ortho_transf_mat = mat_targ @ inv_self
        self._ortho_ortho_offset = (off_cc_self.ravel() - off_cc_targ.ravel()).reshape(4, 1)
        # Convert offset into self's image space or target's image space:
        # x_targ = mat_targ @ inv_self @ x_self + mat_targ @ (off_cc_self - off_cc_targ)
        self._ortho_ortho_offset = (mat_targ @ (off_cc_self.ravel() - off_cc_targ.ravel())).reshape(4, 1)

    def transform_pix_to_img(self, pix_input: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform pixel coordinates (Crystal Cartesian) into image coordinates."""
        q_cc, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_input)
        use_4d = ndim == 4 and not self._do_3D_transformation
        if use_4d:
            mat, offset_cc, _ = self.get_pix_img_transformation(4, *args, nargout=3)
            pts = np.vstack([q_cc, dE])
            shifted = pts - offset_cc.reshape(4, 1)
            res = mat @ shifted
        else:
            mat, offset_cc, _ = self.get_pix_img_transformation(3, *args, nargout=3)
            shifted = q_cc - offset_cc[:3].reshape(3, 1)
            res_3d = mat @ shifted
            if ndim == 4:
                res = np.vstack([res_3d, dE - self._offset[3]])
            else:
                res = res_3d

        if is_row:
            res = res.T
            if np.asarray(pix_input).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_pix(self, img_data: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform image coordinates into pixel coordinates (Crystal Cartesian)."""
        q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(img_data)
        use_4d = ndim == 4 and not self._do_3D_transformation
        if use_4d:
            mat, offset_cc, _ = self.get_pix_img_transformation(4, *args, nargout=3)
            pts = np.vstack([q_img, dE])
            res = np.linalg.solve(mat, pts) + offset_cc.reshape(4, 1)
        else:
            mat, offset_cc, _ = self.get_pix_img_transformation(3, *args, nargout=3)
            res_3d = np.linalg.solve(mat, q_img) + offset_cc[:3].reshape(3, 1)
            if ndim == 4:
                res = np.vstack([res_3d, dE + self._offset[3]])
            else:
                res = res_3d

        if is_row:
            res = res.T
            if np.asarray(img_data).ndim == 1:
                return res.ravel()
        return res

    def transform_img_to_hkl(self, pix_img: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform image coordinates into hkl coordinates."""
        q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_img)
        u_to_rlu = self.u_to_rlu
        if ndim == 4 and not self._do_3D_transformation:
            pts = np.vstack([q_img, dE])
            res = u_to_rlu @ pts + self._offset.reshape(4, 1)
        else:
            res_3d = u_to_rlu[:3, :3] @ q_img + self._offset[:3].reshape(3, 1)
            if ndim == 4:
                res = np.vstack([res_3d, dE + self._offset[3]])
            else:
                res = res_3d

        if is_row:
            res = res.T
            if np.asarray(pix_img).ndim == 1:
                return res.ravel()
        return res

    def transform_hkl_to_img(self, pix_hkl: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform hkl coordinates into image coordinates."""
        q_hkl, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_hkl)
        u_to_rlu = self.u_to_rlu
        if ndim == 4 and not self._do_3D_transformation:
            pts = np.vstack([q_hkl, dE]) - self._offset.reshape(4, 1)
            res = np.linalg.solve(u_to_rlu, pts)
        else:
            pts_3d = q_hkl - self._offset[:3].reshape(3, 1)
            res_3d = np.linalg.solve(u_to_rlu[:3, :3], pts_3d)
            if ndim == 4:
                res = np.vstack([res_3d, dE - self._offset[3]])
            else:
                res = res_3d

        if is_row:
            res = res.T
            if np.asarray(pix_hkl).ndim == 1:
                return res.ravel()
        return res

    def from_this_to_targ_coord(self, pix_origin: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Transform coordinates from this projection's image space to targ_proj's image space."""
        if self._targ_proj is None:
            raise HoraceError(
                "HORACE:aProjectionBase:runtime_error",
                "Target projection (targ_proj) is not set",
            )
        if self._ortho_ortho_transf_mat is not None and self._ortho_ortho_offset is not None:
            q_img, dE, is_row, ndim = self.check_and_preprocess_input_coords(pix_origin)
            if ndim == 3:
                res = self._ortho_ortho_transf_mat[:3, :3] @ q_img + self._ortho_ortho_offset[:3]
            else:
                pts = np.vstack([q_img, dE])
                res = self._ortho_ortho_transf_mat @ pts + self._ortho_ortho_offset
            if is_row:
                res = res.T
                if np.asarray(pix_origin).ndim == 1:
                    return res.ravel()
            return res
        return super().from_this_to_targ_coord(pix_origin, *args, **kwargs)

    def copy_proj_defined_properties_to_axes(self, axes_bl: AxesBlockBase) -> AxesBlockBase:
        """Copy projection properties including hkle_axes_directions to axes_bl."""
        axes_bl = super().copy_proj_defined_properties_to_axes(axes_bl)
        if self.lattice_defined:
            axes_bl.hkle_axes_directions = self.u_to_rlu
        return axes_bl

    def bin_pixels(
        self,
        axes: AxesBlockBase,
        in_pix: Any,
        npix: Any = None,
        s: Any = None,
        e: Any = None,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Project and bin pixels into axes using MEX if configured, else Python."""
        nomex = "-nomex" in args
        force_mex = "-force_mex" in args
        if nomex and force_mex:
            raise HoraceError(
                "HORACE:LineProjBase:invalid_argument",
                "invalid arguments to bin_pixels: '-nomex' and '-force_mex' cannot be used together",
            )
        if nomex:
            use_mex = False
        elif force_mex:
            use_mex = True
        else:
            use_mex = bool(config_store.instance().get_value("hor_config", "use_mex"))

        if use_mex and not isinstance(in_pix, (list, tuple)):
            return self._project_and_bin_with_mex_code(
                axes, in_pix, npix, s, e, *args, nargout=nargout
            )
        return super().bin_pixels(axes, in_pix, npix, s, e, *args, nargout=nargout)

    def _project_and_bin_with_mex_code(
        self,
        axes: AxesBlockBase,
        in_pix: Any,
        npix: Any = None,
        s: Any = None,
        e: Any = None,
        *args: Any,
        nargout: Optional[int] = None,
    ) -> Any:
        """Combined transformation and binning via C++/MEX routine ``bin_pixels_c``."""
        if nargout is None:
            nargout = _infer_nargout(default=5)
        mode_to_bin = 4 if nargout >= 4 else nargout
        mex_flags = [
            a
            for a in args
            if isinstance(a, str)
            and a in ("-force_double", "-selected_only", "-test_mex_inputs", "-nosort")
        ]
        # TODO(port-mex): Requires C++ MEX routine bin_pixels_c
        return axes.bin_pixels_with_mex_code(
            in_pix,
            mode_to_bin,
            npix,
            s,
            e,
            in_pix,
            *mex_flags,
            nargout=nargout,
        )

    # ------------------------------------------------------------------
    # Legacy Conversion Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def u_to_rlu_legacy_from_uvw(
        b_mat: np.ndarray,
        alatt: np.ndarray,
        angdeg: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: Optional[np.ndarray],
        proj_type: str,
        nonorthogonal: bool,
        nargout: Optional[int] = None,
    ) -> Any:
        """Compute legacy u_to_rlu and ulen from (u, v, w, type, nonorthogonal)."""
        if nargout is None:
            nargout = _infer_nargout(default=3)

        b, arlu, _ = bmatrix(alatt, angdeg, nargout=3)
        u_vec = np.asarray(u, dtype=float).ravel()
        v_vec = np.asarray(v, dtype=float).ravel()
        w_vec = None if (w is None or np.asarray(w).size == 0) else np.asarray(w, dtype=float).ravel()

        _, ub = ubmatrix(u_vec, v_vec, b, nargout=2)

        if not nonorthogonal:
            # Orthogonal frame defined by u, v (and w rotates around u if provided)
            if w_vec is not None and np.linalg.norm(w_vec) > 1e-12:
                # Rotate around u so w lies in (u, w_ortho) plane
                w_c = b @ w_vec
                # Project w_c onto plane perpendicular to e1 (first row of ub)
                e1 = ub[0, :]
                e2_init = ub[1, :]
                e3_init = ub[2, :]
                w_perp = w_c - np.dot(w_c, e1) * e1
                wn = float(np.linalg.norm(w_perp))
                if wn < 1e-12:
                    raise HoraceError(
                        "HORACE:line_proj:invalid_argument",
                        "u and w cannot be collinear",
                    )
                e3_new = w_perp / wn
                e2_new = np.cross(e3_new, e1)
                # Keep handedness
                if np.dot(np.cross(e1, e2_new), e3_new) < 0:
                    e2_new = -e2_new
                ub = np.vstack([e1, e2_new, e3_new])

            # Direction vectors in rlu along orthogonal axes
            rlu_dirs = np.linalg.solve(b, ub.T)  # columns are hkl vectors of unit steps in CC
            # Wait: for each axis i in 0..2, what is the unscaled vector in rlu?
            # Axis 0 is along u_vec; Axis 1 is along v_ortho; Axis 2 is along w_ortho
            # Specifically:
            # u_axis_rlu: along u_vec
            # v_axis_rlu: in (u, v) plane perpendicular to u, scaled so projection of v onto v_axis is 1?
            # Let's check exact Horace formula in projaxes_to_rlu_legacy / projaxes_to_rlu!
            u_c = b @ u_vec
            v_c = b @ v_vec
            if w_vec is None:
                # w_rlu is cross(u_c, v_c) in reciprocal space:
                # In Horace: w = inv(B) * cross(B*u, B*v) / |...|
                w_c = np.cross(u_c, v_c)
                w_rlu_dir = np.linalg.solve(b, w_c)
            else:
                w_rlu_dir = w_vec

            # Orthogonal unit vectors in CC: e_i = ub[i, :]
            # Unnormalized axis vectors in rlu:
            # u_ortho_c = (u_c . e1) * e1 = |u_c| * e1 -> rlu is u_vec
            # v_ortho_c = (v_c . e2) * e2 -> rlu is inv(B) * ((v_c . e2) * e2)
            # w_ortho_c = (w_c . e3) * e3 -> if w was not given, w_rlu_dir = inv(B) * cross(u_c, v_c)
            u_ax_c = np.dot(u_c, ub[0, :]) * ub[0, :]
            v_ax_c = np.dot(v_c, ub[1, :]) * ub[1, :]
            if w_vec is None:
                w_ax_c = np.cross(u_c, v_c)
            else:
                w_c_in = b @ w_vec
                w_ax_c = np.dot(w_c_in, ub[2, :]) * ub[2, :]

            ax_c_mat = np.column_stack([u_ax_c, v_ax_c, w_ax_c])
        else:
            # Non-orthogonal axes: u, v, w directly
            u_c = b @ u_vec
            v_c = b @ v_vec
            if w_vec is None:
                w_c = np.cross(u_c, v_c)
            else:
                w_c = b @ w_vec
            ax_c_mat = np.column_stack([u_c, v_c, w_c])

        if abs(np.linalg.det(ax_c_mat)) < 1e-12:
            raise HoraceError(
                "HORACE:line_proj:invalid_argument",
                "Projection vectors u, v, w are coplanar or collinear",
            )

        ax_rlu_mat = np.linalg.solve(b, ax_c_mat)
        ulen = np.ones(3, dtype=float)
        u_to_rlu = np.zeros((3, 3), dtype=float)

        t_str = proj_type.lower()
        for i in range(3):
            norm_c = float(np.linalg.norm(ax_c_mat[:, i]))
            ch = t_str[i]
            if ch == "p":
                # rlu normalized so max|h,k,l| or projection is 1:
                # For 'p', step of 1 is the vector itself (u_ax, v_ax, w_ax) except for default w when w is None:
                if i == 2 and w_vec is None and not nonorthogonal:
                    # When w is not given and type[2]=='p', normalize w_rlu so max(abs(w_rlu)) == 1
                    m_val = float(np.max(np.abs(ax_rlu_mat[:, i])))
                    u_to_rlu[:, i] = ax_rlu_mat[:, i] / m_val
                    ulen[i] = norm_c / m_val
                else:
                    u_to_rlu[:, i] = ax_rlu_mat[:, i]
                    ulen[i] = norm_c
            elif ch == "r":
                # Normalized so max(|h|, |k|, |l|) == 1
                m_val = float(np.max(np.abs(ax_rlu_mat[:, i])))
                u_to_rlu[:, i] = ax_rlu_mat[:, i] / m_val
                ulen[i] = norm_c / m_val
            elif ch == "a":
                # Inverse Angstroms: ulen = 1, u_to_rlu[:, i] has length 1 A^-1 in CC
                u_to_rlu[:, i] = ax_rlu_mat[:, i] / norm_c
                ulen[i] = 1.0
            else:
                raise HoraceError(
                    "HORACE:line_proj:invalid_argument",
                    f"Unknown projection type character '{ch}'",
                )

        # Convert u_to_rlu + ulen into pix_to_img_mat (3x3):
        # q_rlu = u_to_rlu @ q_img => q_cc = B @ u_to_rlu @ q_img => q_img = inv(B @ u_to_rlu) @ q_cc
        u_to_cc = b @ u_to_rlu
        pix_to_img = np.linalg.inv(u_to_cc)

        if nargout == 1:
            return pix_to_img
        if nargout == 2:
            return pix_to_img, u_to_rlu
        return pix_to_img, u_to_rlu, ulen

    @staticmethod
    def uv_from_u_to_rlu_legacy(
        u_to_rlu: np.ndarray,
        ulen: np.ndarray,
        alatt: np.ndarray,
        angdeg: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, bool]:
        """Extract (u, v, w, type, nonorthogonal) from legacy u_to_rlu and ulen.
        Ported from @LineProjBase/private/uv_from_u_to_rlu_.m.
        """
        b = bmatrix(alatt, angdeg, nargout=1)
        u_rlu_3 = np.asarray(u_to_rlu, dtype=float)[:3, :3]
        ulen_3 = np.asarray(ulen, dtype=float).ravel()[:3]

        ub_inv = u_rlu_3 / ulen_3.reshape(1, 3)
        umat = np.linalg.inv(b @ ub_inv)
        cross_proj = np.array(
            [
                float(np.dot(umat[0, :], umat[1, :])),
                float(np.dot(umat[0, :], umat[2, :])),
                float(np.dot(umat[1, :], umat[2, :])),
            ]
        )
        ortho = not bool(np.any(np.abs(cross_proj) > 1e-8))
        nonortho = not ortho

        if ortho:
            ubmat = umat @ b
            ubmatinv = np.linalg.inv(ubmat)
            uvw_orth_hkl = np.linalg.solve(b, umat.T)
            lt = []
            for i in range(3):
                ulen_i_r = 1.0 / float(np.max(np.abs(ubmatinv[:, i])))
                if ulen_3[i] == 1.0:
                    lt.append("a")
                elif abs(ulen_3[i] - ulen_i_r) < 1e-7:
                    lt.append("r")
                else:
                    lt.append("p")
                    uvw_orth_hkl[:, i] = uvw_orth_hkl[:, i] * ulen_3[i]
            u = uvw_orth_hkl[:, 0].copy()
            v = uvw_orth_hkl[:, 1].copy()
            w = uvw_orth_hkl[:, 2].copy()
            ptype = "".join(lt)
        else:
            transf_cc = np.linalg.inv(umat)
            uvw = np.linalg.solve(b, transf_cc)
            uvw_orth = transf_cc
            lt = []
            eps_s = 4.0 * float(np.finfo(np.float32).eps)
            for i in range(3):
                veclen = float(np.linalg.norm(uvw_orth[:, i]))
                vec = uvw[:, i]
                scale = 1.0
                if abs(veclen - ulen_3[i]) < eps_s:
                    t_ch = "p"
                elif ulen_3[i] == 1.0:
                    t_ch = "a"
                else:
                    scale_p = ulen_3[i] / veclen
                    scale_r = float(np.sqrt(ulen_3[i] / veclen / np.max(np.abs(vec))))
                    if abs(veclen * scale_r - ulen_3[i]) < eps_s:
                        scale = scale_r
                        t_ch = "r"
                    else:
                        scale = scale_p
                        t_ch = "p"
                if scale != 1.0:
                    uvw[:, i] = scale * uvw[:, i]
                lt.append(t_ch)
            u = uvw[:, 0].copy()
            v = uvw[:, 1].copy()
            w = uvw[:, 2].copy()
            ptype = "".join(lt)

        return u, v, w, ptype, nonortho

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """Compare two LineProjBase projections via their transformation matrices.
        Ported from @LineProjBase/private/eq_to_tol_single_.m.
        """
        from pyhorace.core.serializable import compare_values

        if not isinstance(other, LineProjBase):
            return (
                False,
                f"Objects are of different classes: '{type(self).__name__}' and '{type(other).__name__}'",
            )
        obj1_def = self.alatt_defined and self.angdeg_defined
        obj2_def = other.alatt_defined and other.angdeg_defined
        if not obj1_def and not obj2_def:
            return super().equal_to_tol(
                other,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=name_a,
                name_b=name_b,
            )
        if obj1_def != obj2_def:
            return (
                False,
                f"Object {name_a} defined={obj1_def} and Object {name_b} defined={obj2_def}",
            )
        q_to_img_1, shift_1, ulen_1 = self.get_pix_img_transformation(4, nargout=3)
        q_to_img_2, shift_2, ulen_2 = other.get_pix_img_transformation(4, nargout=3)
        mat_eq, mess1 = compare_values(
            q_to_img_1,
            q_to_img_2,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            name_a=f"{name_a}.q_to_img",
            name_b=f"{name_b}.q_to_img",
        )
        if not mat_eq:
            return False, f"Q_to_img: {mess1}"
        shift_eq, mess2 = compare_values(
            shift_1,
            shift_2,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            name_a=f"{name_a}.shift",
            name_b=f"{name_b}.shift",
        )
        if not shift_eq:
            return False, f"shift(s): {mess2}"
        len_eq, mess3 = compare_values(
            ulen_1,
            ulen_2,
            tol=tol,
            nan_equal=nan_equal,
            ignore_str=ignore_str,
            name_a=f"{name_a}.ulen",
            name_b=f"{name_b}.ulen",
        )
        if not len_eq:
            return False, f"ulen(s): {mess3}"
        return True, ""

    @staticmethod
    def get_from_old_data(data_struct: dict[str, Any]) -> "LineProjBase":
        """Construct line_proj or ubmat_proj from legacy Horace data structure."""
        from pyhorace.coord_transf.line_proj import line_proj
        from pyhorace.coord_transf.ubmat_proj import ubmat_proj

        alatt = data_struct.get("alatt", [2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi])
        angdeg = data_struct.get("angdeg", [90.0, 90.0, 90.0])
        uoffset = data_struct.get("uoffset", data_struct.get("offset", [0.0, 0.0, 0.0, 0.0]))
        raw_label = data_struct.get("ulabel", data_struct.get("label", ["\\zeta", "\\xi", "\\eta", "E"]))
        ulabel = [str(x).strip() for x in np.asarray(raw_label).ravel()] if raw_label is not None else ["\\zeta", "\\xi", "\\eta", "E"]
        u_to_rlu = np.asarray(
            data_struct.get("u_to_rlu", data_struct.get("u_to_rlu_legacy", np.eye(4))),
            dtype=float,
        )
        ulen = np.asarray(data_struct.get("ulen", np.ones(4)), dtype=float).ravel()

        try:
            u, v, w, ptype, nonortho = LineProjBase.uv_from_u_to_rlu_legacy(u_to_rlu, ulen, alatt, angdeg)
            proj = line_proj(
                u,
                v,
                w,
                nonorthogonal=nonortho,
                type=ptype,
                alatt=alatt,
                angdeg=angdeg,
                offset=uoffset,
                label=ulabel,
            )
            proj.u_to_rlu_legacy = u_to_rlu
            proj.ulen = ulen
            return proj
        except Exception:
            return ubmat_proj(
                u_to_rlu,
                ulen,
                alatt=alatt,
                angdeg=angdeg,
                offset=uoffset,
                label=ulabel,
            )

    def copy_proj_defined_properties_to_axes(self, axes_bl: Any) -> Any:
        """Copy line_proj properties including hkle_axes_directions to axes_bl."""
        axes_bl = super().copy_proj_defined_properties_to_axes(axes_bl)
        if hasattr(axes_bl, "hkle_axes_directions") and self.lattice_defined:
            axes_bl.hkle_axes_directions = self.u_to_rlu
        return axes_bl
