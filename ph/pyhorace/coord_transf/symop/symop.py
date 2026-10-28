"""Base class for symmetry operators (Symop) and factory function symop.

Translated from MATLAB ``horace_core/symop/@Symop/Symop.m``,
``@Symop/validate_and_generate_sym.m``, ``@Symop/private/short_name_.m``,
``@Symop/private/extract_common_group_offset_.m``, and ``horace_core/symop/symop.m``.
"""

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np

from pyhorace.coord_transf.a_projection_base import aProjectionBase
from pyhorace.coord_transf.bmatrix import _infer_nargout
from pyhorace.core.exceptions import HoraceError
from pyhorace.core.serializable import SERIALIZABLE_REGISTRY, Serializable


def _short_name(op_name: Any) -> str:
    """Identify short name defining the symmetry operation type.

    Ported from ``@Symop/private/short_name_.m``.
    """
    short_names = ("RE", "RO", "I", "G")
    full_names = ("reflection", "rotation", "identity", "general")
    if not isinstance(op_name, str) or len(op_name) == 0:
        raise HoraceError(
            "HORACE:Symop:invalid_argument",
            f"Type of operation has to be defined by string, describing the operation. In fact its type is {type(op_name).__name__}",
        )
    if op_name.lower().startswith("symop"):
        op_name = op_name[5:]
        if len(op_name) == 0:
            raise HoraceError(
                "HORACE:Symop:invalid_argument",
                "Short name of input operation: 'symop' can not be uniquely identified as symmetry operation",
            )
    if len(op_name) == 1:
        test_name = op_name[0].upper()
    else:
        test_name = op_name[:2].upper()

    matches = [
        sn for sn in short_names if sn.startswith(test_name) or test_name.startswith(sn)
    ]
    if len(matches) != 1:
        raise HoraceError(
            "HORACE:Symop:invalid_argument",
            f"Short name of input operation: {op_name} can not be uniquely identified as symmetry operation",
        )
    idx = short_names.index(matches[0])
    full_match = full_names[idx]
    if not full_match.startswith(op_name.lower()):
        raise HoraceError(
            "HORACE:Symop:invalid_argument",
            f"Short name of input operation: {op_name} can not be uniquely identified as symmetry operation",
        )
    return matches[0]


def _extract_common_group_offset(
    symmetries: Sequence["Symop"],
) -> Tuple[np.ndarray, list["Symop"]]:
    """Check if operations in a symmetry group share a common offset and propagate it.

    Ported from ``@Symop/private/extract_common_group_offset_.m``.
    """
    ops = list(symmetries)
    eps4 = 4.0 * float(np.finfo(float).eps)
    has_offset = [bool(np.any(np.abs(op.offset) > eps4)) for op in ops]
    if any(has_offset):
        non_zero_ops = [op for op, h in zip(ops, has_offset) if h]
        sym_offset = non_zero_ops[0].offset
        for op in non_zero_ops[1:]:
            if np.any(np.abs(op.offset - sym_offset) > eps4):
                raise HoraceError(
                    "HORACE:Symop:not_implemented",
                    "Combining symmetries with different non-zero offsets is not currently supported",
                )
        if not all(has_offset):
            for op in ops:
                op.offset = sym_offset
    else:
        sym_offset = np.zeros(3, dtype=float)
    return sym_offset, ops


def _is_same_offset(sym: Sequence["Symop"]) -> bool:
    """Return True if all operations in ``sym`` have the same offset within ``Symop.tol_``."""
    if len(sym) <= 1:
        return True
    ref_offset = sym[0].offset
    for op in sym[1:]:
        if np.any(np.abs(op.offset - ref_offset) > Symop.tol_):
            return False
    return True


def _get_rot_angle(op: "Symop") -> float:
    """Return rotation angle of ``op`` (0 for ``SymopIdentity``)."""
    if isinstance(op, _get_symop_identity_cls()):
        return 0.0
    return float(getattr(op, "theta_deg", 0.0))


def _get_symop_identity_cls() -> type:
    from pyhorace.coord_transf.symop.symop_identity import SymopIdentity

    return SymopIdentity


def _get_symop_general_cls() -> type:
    from pyhorace.coord_transf.symop.symop_general import SymopGeneral

    return SymopGeneral


def _get_symop_reflection_cls() -> type:
    from pyhorace.coord_transf.symop.symop_reflection import SymopReflection

    return SymopReflection


def _get_symop_rotation_cls() -> type:
    from pyhorace.coord_transf.symop.symop_rotation import SymopRotation

    return SymopRotation


def _resolve_symop_cls(class_name: str) -> type:
    target = SERIALIZABLE_REGISTRY.get(class_name)
    if target is not None:
        return target
    for getter in (
        _get_symop_identity_cls,
        _get_symop_general_cls,
        _get_symop_reflection_cls,
        _get_symop_rotation_cls,
    ):
        cls = getter()
        if cls.__name__ == class_name:
            return cls
    raise HoraceError("HORACE:Symop:invalid_argument", f"Unknown Symop subclass: {class_name}")


class Symop(Serializable, ABC):
    """Abstract parent class for symmetry operations in Horace."""

    tol_: ClassVar[float] = 1e-12

    def class_version(self) -> int:
        return 2

    classVersion = class_version

    def __init__(self) -> None:
        super().__init__()
        self._offset: np.ndarray = np.zeros(3, dtype=float)
        self._u_offset: np.ndarray = np.zeros(3, dtype=float)
        self._offset_specified_uoffset_not: bool = False
        self._b_matrix: Optional[np.ndarray] = None
        self._R: np.ndarray = np.zeros((3, 3), dtype=float)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def R(self) -> np.ndarray:
        """[3x3] symmetry operator transformation matrix in Crystal Cartesian coordinates."""
        return self._R.copy()

    @R.setter
    def R(self, val: Any) -> None:
        self._set_R(val)
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def offset(self) -> np.ndarray:
        """3-element vector of symmetry operator application point in rlu."""
        return self._offset.copy()

    @offset.setter
    def offset(self, val: Any) -> None:
        is_valid, brushed = self.check_and_brush_3vector(val)
        if not is_valid:
            raise HoraceError(
                "HORACE:symop:invalid_argument",
                f"Offset should be 3-element numeric vector. In fact it is: {val!r}",
            )
        self._offset = brushed
        self._offset_specified_uoffset_not = True
        if self._do_check_combo_arg:
            self.check_combo_arg()

    @property
    def b_matrix(self) -> np.ndarray:
        """Busing and Levy B-matrix (3x3) or empty array if not set."""
        if self._b_matrix is None:
            return np.array([], dtype=float)
        return self._b_matrix.copy()

    @b_matrix.setter
    def b_matrix(self, val: Any) -> None:
        if val is None or (
            isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0
        ):
            self._b_matrix = None
        else:
            if not self.is_3x3matrix(val):
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"B-matrix needs to be a 3x3 square matrix. In fact it is: {val!r}",
                )
            self._b_matrix = np.asarray(val, dtype=float).copy()
        if self._do_check_combo_arg:
            self.check_combo_arg()

    def check_offset_b_matrix_consistency(self) -> "Symop":
        """Synchronize ``_u_offset`` and ``_offset_specified_uoffset_not`` with ``_offset`` and ``_b_matrix``."""
        if np.all(self._offset == 0.0):
            self._u_offset = np.zeros(3, dtype=float)
            self._offset_specified_uoffset_not = False
        elif self._b_matrix is None:
            self._offset_specified_uoffset_not = True
        else:
            self._u_offset = self._b_matrix @ self._offset
            self._offset_specified_uoffset_not = False
        return self

    # ------------------------------------------------------------------
    # Factory & Static Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def create(op_name: Any, *args: Any, **kwargs: Any) -> Any:
        """Create a symmetry operator by name ('reflection', 'rotation', 'identity', 'general')."""
        SymopGeneral = _get_symop_general_cls()
        SymopIdentity = _get_symop_identity_cls()
        SymopReflection = _get_symop_reflection_cls()
        SymopRotation = _get_symop_rotation_cls()

        short_type = _short_name(op_name)
        if short_type == "RE":
            return SymopReflection(*args, **kwargs)
        if short_type == "RO":
            if len(args) > 0 and isinstance(args[0], str) and args[0].lower().startswith("f"):
                return SymopRotation.fold(*args[1:], **kwargs)
            return SymopRotation(*args, **kwargs)
        if short_type == "I":
            return SymopIdentity(*args, **kwargs)
        if short_type == "G":
            return SymopGeneral(*args, **kwargs)
        raise HoraceError(
            "HORACE:Symop:invalid_argument",
            f"Unknown operation type: {op_name}",
        )

    @staticmethod
    def check_and_brush_3vector(elem: Any) -> Tuple[bool, np.ndarray]:
        """Validate that ``elem`` is a 3-element numeric vector and zero small entries."""
        if isinstance(elem, (bool, np.bool_, str, bytes, dict)) or elem is None:
            return False, np.array([], dtype=float)
        try:
            arr = np.asarray(elem, dtype=float)
        except (ValueError, TypeError):
            return False, np.array([], dtype=float)
        if arr.size != 3 or arr.ndim > 2 or (arr.ndim == 2 and 1 not in arr.shape):
            return False, np.array([], dtype=float)
        out = arr.ravel().astype(float).copy()
        out[np.abs(out) < Symop.tol_] = 0.0
        return True, out

    @staticmethod
    def is_3x3matrix(elem: Any) -> bool:
        """Return True if ``elem`` is a numeric 3x3 matrix."""
        if isinstance(elem, (bool, np.bool_, str, bytes, dict)) or elem is None:
            return False
        try:
            arr = np.asarray(elem, dtype=float)
        except (ValueError, TypeError):
            return False
        return arr.ndim == 2 and arr.shape == (3, 3)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def transform_vec(self, vec: Any, use_rlu_offset: bool = False) -> np.ndarray:
        """Apply symmetry operator transformation to [3xN] (or 1D 3-element) vector(s)."""
        return SymopArray([self]).transform_vec(vec, use_rlu_offset=use_rlu_offset)

    def transform_pix(
        self,
        pix: Any,
        abs_tol: float = 0.0,
        use_rlu_offset: bool = False,
    ) -> Any:
        """Transform pixel coordinates into the irreducible zone of this symmetry operator."""
        return SymopArray([self]).transform_pix(
            pix, abs_tol=abs_tol, use_rlu_offset=use_rlu_offset
        )

    def transform_proj(
        self,
        proj: aProjectionBase,
        nargout: Optional[int] = None,
    ) -> Any:
        """Transform a projection object according to this symmetry operation."""
        nargout = _infer_nargout(default=1, explicit_nargout=nargout)
        new_proj, out_obj = SymopArray([self]).transform_proj(proj, nargout=2)
        if nargout == 1:
            return new_proj
        return new_proj, out_obj

    @staticmethod
    def validate_and_generate_sym(
        sym: Any,
        proj: aProjectionBase,
        nargout: Optional[int] = None,
    ) -> Any:
        """Validate symmetry descriptors and generate symmetry operators for ``symmetrise_sqw``.

        Ported from ``@Symop/validate_and_generate_sym.m``.
        """
        SymopIdentity = _get_symop_identity_cls()
        SymopReflection = _get_symop_reflection_cls()
        SymopRotation = _get_symop_rotation_cls()

        nargout = _infer_nargout(default=2, explicit_nargout=nargout)

        is_cell_input = isinstance(sym, (list, tuple)) and not isinstance(sym, SymopArray)

        if isinstance(sym, SymopReflection) or (
            isinstance(sym, SymopArray)
            and len(sym) > 0
            and all(isinstance(x, SymopReflection) for x in sym)
        ):
            if isinstance(sym, SymopReflection):
                fold = 1
                sym_out: Any = copy.deepcopy(sym)
            else:
                fold = len(sym)
                if not _is_same_offset(sym):
                    sym_out = [copy.deepcopy(x) for x in sym]
                else:
                    sym_out = SymopArray([copy.deepcopy(x) for x in sym])
        elif isinstance(sym, SymopRotation) or (
            isinstance(sym, SymopArray)
            and len(sym) > 0
            and all(isinstance(x, SymopRotation) for x in sym)
        ):
            if isinstance(sym, SymopArray):
                if len(sym) > 1:
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        "Cannot symetrise rotating around multiples axes unless the rotation operations are provided in cell array",
                    )
                rot_op = sym[0]
            else:
                rot_op = sym
            fold_val = 360.0 / rot_op.theta_deg - 1.0
            if np.floor(fold_val) != fold_val:
                raise HoraceError(
                    "HORACE:symmetrise_sqw:invalid_argument",
                    "fold dispersion requires a fraction of 360 degree syntax.",
                )
            fold = int(fold_val)
            sym_out = [copy.deepcopy(rot_op) for _ in range(fold)]
        elif isinstance(sym, SymopIdentity):
            fold = 1
            sym_out = copy.deepcopy(sym)
        elif is_cell_input:
            if len(sym) == 0:
                raise HoraceError(
                    "HORACE:symmetrise_sqw:invalid_argument",
                    "Unrecognised empty cell array for symmetry expansion",
                )
            is_rot = [isinstance(x, (SymopRotation, SymopIdentity)) for x in sym]
            is_refl = [isinstance(x, (SymopReflection, SymopIdentity)) for x in sym]
            if all(is_rot):
                first = sym[0]
                if not (isinstance(first, SymopIdentity) or getattr(first, "theta_deg", None) == 0):
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        "Calling rotations in cell array has to follow rotation fold rules and first symmetry operation has to be identity",
                    )
                if len(sym) < 2 or not hasattr(sym[1], "theta_deg") or sym[1].theta_deg == 0:
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        "Calling rotations in cell array has to follow rotation fold rules.",
                    )
                fold_val = 360.0 / sym[1].theta_deg
                if np.floor(fold_val) != fold_val or len(sym) != int(fold_val):
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        f"Calling rotations in cell array has to follow rotation fold rules. Rotation angle {sym[1].theta_deg:g} gives non-integer fold {fold_val:g}",
                    )
                rot_ang = np.array([_get_rot_angle(x) for x in sym], dtype=float)
                ang_diff = np.diff(rot_ang)
                the_same_ang = np.abs(ang_diff - sym[1].theta_deg) < Symop.tol_
                if len(the_same_ang) > 0:
                    the_same_ang[0] = True
                if not (
                    bool(np.all(the_same_ang))
                    or bool(np.all(np.abs(ang_diff[1:]) < Symop.tol_))
                ):
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        "Calling rotations in cell array has to follow rotation fold rules. Some angles do not belong to geometric progression of angles",
                    )
                if not _is_same_offset(sym):
                    raise HoraceError(
                        "HORACE:symmetrise_sqw:invalid_argument",
                        "Calling rotations in cell array has to follow rotation fold rules. Offset of all components must be the same",
                    )
                fold = int(fold_val) - 1
                sym_out = [copy.deepcopy(sym[1]) for _ in range(fold)]
            elif all(is_refl):
                fold = len(sym)
                sym_out = [copy.deepcopy(x) for x in sym]
            else:
                raise HoraceError(
                    "HORACE:symmetrise_sqw:not_implemented",
                    "Cell array of symmetries may only contain symmetries of the same kind, i.e. rotations or reflections",
                )
        else:
            raise HoraceError(
                "HORACE:symmetrise_sqw:invalid_argument",
                f"Unrecognised type {type(sym).__name__} for symmetry expansion",
            )

        bm = proj.bmatrix(3, nargout=1)

        def _set_bmat(elem: Any) -> Any:
            if isinstance(elem, SymopArray):
                for op in elem:
                    op.b_matrix = bm
                return elem
            if isinstance(elem, list):
                for op in elem:
                    _set_bmat(op)
                return elem
            elem.b_matrix = bm
            return elem

        sym_out = _set_bmat(sym_out)
        if nargout == 1:
            return sym_out
        return sym_out, fold

    # ------------------------------------------------------------------
    # Serialization overrides for heterogeneous Symop arrays
    # ------------------------------------------------------------------
    def from_bare_struct(self, S: Any) -> Any:
        if isinstance(S, dict) and "class" in S and "data" in S:
            classes = S["class"]
            datas = S["data"]
            if isinstance(classes, (list, tuple, np.ndarray)):
                ops = []
                for c_name, d_vals in zip(classes, datas):
                    cls = _resolve_symop_cls(str(c_name))
                    ops.append(cls(*d_vals))
                return SymopArray(ops)
            cls = _resolve_symop_cls(str(classes))
            return cls(*datas)
        if isinstance(S, (list, tuple, np.ndarray)):
            if len(S) == 0:
                return self
            ops = []
            for item in S:
                if isinstance(item, dict) and "class" in item and "data" in item:
                    cls = _resolve_symop_cls(str(item["class"]))
                    ops.append(cls(*item["data"]))
                else:
                    obj_item = self.__class__()
                    obj_item = obj_item.from_bare_struct(item)
                    ops.append(obj_item)
            return SymopArray(ops)
        return super().from_bare_struct(S)

    # ------------------------------------------------------------------
    # Abstract & Protected Interface
    # ------------------------------------------------------------------
    @abstractmethod
    def calculate_transform(self, b_mat: np.ndarray) -> np.ndarray:
        """Calculate the 3x3 transformation matrix in Crystal Cartesian coordinates."""

    @abstractmethod
    def in_irreducible(self, coords: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
        """Return boolean array indicating which [3xN] coordinates lie in the irreducible zone."""

    def _set_R(self, val: Any) -> None:
        """Internal setter for R (prohibited on subclasses except SymopGeneral)."""
        raise HoraceError(
            "HORACE:symop:invalid_argument",
            f"Setting matrix R directly is prohibited for {type(self).__name__}",
        )


def _transform_proj_single(
    proj: aProjectionBase,
    sym_transf_mat: np.ndarray,
    sym_offset: np.ndarray,
    bm: np.ndarray,
    is_identity_only: bool = False,
) -> aProjectionBase:
    """Apply ``sym_transf_mat`` and ``sym_offset`` to a single projection."""
    from pyhorace.coord_transf.line_proj import line_proj
    from pyhorace.coord_transf.ubmat_proj import ubmat_proj

    if isinstance(proj, line_proj):
        lp = copy.deepcopy(proj)
    elif isinstance(proj, ubmat_proj):
        lp = proj.get_line_proj()
    else:
        if is_identity_only:
            return copy.deepcopy(proj)
        raise HoraceError(
            "HORACE:Symop:not_implemented",
            f"transform_proj is not implemented for projection of class {type(proj).__name__}",
        )

    eps4 = 4.0 * float(np.finfo(float).eps)
    if np.any(np.abs(sym_transf_mat - np.eye(3, dtype=float)) > eps4):
        lp.sym_transf = sym_transf_mat
    else:
        lp.sym_transf = np.array([], dtype=float)
        return lp

    lp_offset = np.asarray(lp.offset, dtype=float).ravel().copy()
    offset_old_cc = bm @ lp_offset[:3]
    sym_offset_cc = bm @ sym_offset
    offset_cc = sym_transf_mat @ sym_offset_cc - sym_offset_cc + offset_old_cc
    offset_new = np.linalg.solve(bm, offset_cc)
    lp.offset = np.array(
        [offset_new[0], offset_new[1], offset_new[2], lp_offset[3]], dtype=float
    )
    return lp


class SymopArray(list):
    """Sequence of ``Symop`` objects supporting composite MATLAB array methods.

    In MATLAB, a row vector of ``Symop`` objects ``[op1, op2, ...]`` supports
    calling ``.transform_vec(vec)``, ``.transform_pix(pix)``, and
    ``.transform_proj(proj)`` directly on the array, applying operations in
    reverse order (from ``end`` down to ``1``).
    """

    def __init__(self, ops: Sequence[Symop] = ()) -> None:
        super().__init__(ops)

    @property
    def b_matrix(self) -> Any:
        if len(self) == 0:
            return np.array([], dtype=float)
        if len(self) == 1:
            return self[0].b_matrix
        return [op.b_matrix for op in self]

    @b_matrix.setter
    def b_matrix(self, val: Any) -> None:
        for op in self:
            op.b_matrix = val

    @property
    def offset(self) -> Any:
        if len(self) == 0:
            return np.zeros(3, dtype=float)
        if len(self) == 1:
            return self[0].offset
        return [op.offset for op in self]

    @offset.setter
    def offset(self, val: Any) -> None:
        for op in self:
            op.offset = val

    def transform_vec(self, vec: Any, use_rlu_offset: bool = False) -> np.ndarray:
        """Apply composite symmetry transformation(s) to [3xN] (or 1D 3-element) vector(s)."""
        if isinstance(vec, (bool, np.bool_, str, bytes, dict)) or vec is None:
            raise HoraceError(
                "HORACE:symop:invalid_argument",
                f"Input vectors set has to be array of [3xN] vectors. Provided: {type(vec).__name__}",
            )
        try:
            arr = np.asarray(vec, dtype=float)
        except (ValueError, TypeError) as exc:
            raise HoraceError(
                "HORACE:symop:invalid_argument",
                f"Input vectors set has to be array of [3xN] vectors. Provided: {type(vec).__name__}",
            ) from exc

        was_1d = arr.ndim == 1
        if was_1d:
            if arr.size != 3:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"Input vectors set has to be array of [3xN] vectors. Provided array of size: {arr.shape}",
                )
            v = arr.reshape(3, 1).copy()
        else:
            if arr.ndim != 2 or arr.shape[0] != 3:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"Input vectors set has to be array of [3xN] vectors. Provided array of size: {arr.shape}",
                )
            v = arr.copy()

        if len(self) == 0:
            return v.ravel() if was_1d else v

        first_op: Symop = self[0]
        if use_rlu_offset:
            shift = first_op._offset.reshape(3, 1)
        else:
            if first_op._offset_specified_uoffset_not:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    "Offset is specified in rlu, and its transformation in Crystal Cartesian was not defined before calling transform_vec",
                )
            shift = first_op._u_offset.reshape(3, 1)

        v = v - shift
        for op in reversed(self):
            v = op._R @ v
        v = v + shift
        return v.ravel() if was_1d else v

    def transform_pix(
        self,
        pix: Any,
        abs_tol: float = 0.0,
        use_rlu_offset: bool = False,
    ) -> Any:
        """Transform pixels into the irreducible zone of the symmetry operator(s)."""
        is_pix_obj = hasattr(pix, "q_coordinates")
        if is_pix_obj:
            q_coordinates = np.asarray(pix.q_coordinates, dtype=float).copy()
        else:
            if isinstance(pix, (bool, np.bool_, str, bytes, dict)) or pix is None:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"First argument of transform_pix may be either PixelData or 3xN array of coordinates. Provided: {type(pix).__name__}",
                )
            try:
                arr = np.asarray(pix, dtype=float)
            except (ValueError, TypeError) as exc:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"First argument of transform_pix may be either PixelData or 3xN array of coordinates. Provided: {type(pix).__name__}",
                ) from exc
            if arr.ndim != 2 or arr.shape[0] != 3:
                raise HoraceError(
                    "HORACE:symop:invalid_argument",
                    f"First argument of transform_pix may be either PixelData or 3xN array of coordinates. Provided: {type(pix).__name__}",
                )
            q_coordinates = arr.copy()

        for op in reversed(self):
            in_irr = np.asarray(op.in_irreducible(q_coordinates, abs_tol), dtype=bool).ravel()
            transform = ~in_irr
            if np.any(transform):
                q_coordinates[:, transform] = op.transform_vec(
                    q_coordinates[:, transform], use_rlu_offset=use_rlu_offset
                )

        if is_pix_obj:
            pix.q_coordinates = q_coordinates
            return pix
        return q_coordinates

    def transform_proj(
        self,
        proj: aProjectionBase,
        nargout: Optional[int] = None,
    ) -> Any:
        """Transform a projection object according to the symmetry operation(s)."""
        nargout = _infer_nargout(default=1, explicit_nargout=nargout)

        if not isinstance(proj, aProjectionBase):
            raise HoraceError(
                "HORACE:Symop:invalid_argument",
                f"Input transformation has to be instance of aProjectionBase class. It is: {type(proj).__name__}",
            )

        ops = [copy.deepcopy(op) for op in self]
        sym_offset, ops = _extract_common_group_offset(ops)

        cur_sym_transf = getattr(proj, "sym_transf", np.array([], dtype=float))
        if cur_sym_transf is None or np.asarray(cur_sym_transf).size == 0:
            sym_transf_mat = np.eye(3, dtype=float)
        else:
            sym_transf_mat = np.asarray(cur_sym_transf, dtype=float).copy()

        bm = proj.bmatrix(3, nargout=1)
        SymopIdentity = _get_symop_identity_cls()
        SymopGeneral = _get_symop_general_cls()

        for i in range(len(ops) - 1, -1, -1):
            ops[i].b_matrix = bm
            if not isinstance(ops[i], SymopIdentity):
                sym_transf_mat = ops[i].R @ sym_transf_mat
                ops[i].offset = sym_offset

        if len(ops) > 1:
            out_obj: Symop = SymopGeneral(sym_transf_mat, sym_offset)
            out_obj.b_matrix = bm
        elif len(ops) == 1:
            out_obj = ops[0]
        else:
            out_obj = SymopIdentity()

        is_identity_only = len(ops) == 1 and isinstance(ops[0], SymopIdentity)
        new_proj = _transform_proj_single(
            proj, sym_transf_mat, sym_offset, bm, is_identity_only=is_identity_only
        )

        if nargout == 1:
            return new_proj
        return new_proj, out_obj

    def to_bare_struct(self, recursive_bare: bool = False) -> Any:
        """Convert array of Symop objects to bare struct (supporting heterogeneous Symop arrays)."""
        if len(self) == 1:
            return self[0].to_bare_struct(recursive_bare=recursive_bare)
        return {
            "class": [type(op).__name__ for op in self],
            "data": [
                [getattr(op, fld) for fld in op.saveable_fields()] for op in self
            ],
        }

    def to_struct(self) -> dict[str, Any]:
        """Serialize this array of Symop objects following MATLAB Symop.to_struct."""
        if len(self) == 1:
            return self[0].to_struct()
        bare = self.to_bare_struct(recursive_bare=False)
        res: dict[str, Any] = {
            "serial_name": "SymopIdentity",
            "version": self[0].class_version() if len(self) > 0 else 2,
        }
        res.update(bare)
        return res

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """Compare two sequences of Symop objects up to tolerance."""
        if not isinstance(other, (list, tuple, np.ndarray, SymopArray)):
            return False, f"Cannot compare SymopArray with {type(other).__name__}"
        if len(self) != len(other):
            return False, f"Length mismatch: {len(self)} vs {len(other)}"
        for idx, (a, b) in enumerate(zip(self, other)):
            is_eq, mess = a.equal_to_tol(
                b,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}[{idx}]",
                name_b=f"{name_b}[{idx}]",
            )
            if not is_eq:
                return False, mess
        return True, ""


def symop(*args: Any, **kwargs: Any) -> Any:
    """Deprecated wrapper around ``Symop.create`` (emits ``HORACE:symop:deprecated``)."""
    warnings.warn(
        "HORACE:symop:deprecated: "
        "Current syntax for generating Symmetry operations is deprecated. "
        "Replace it by direct call to the symmetry class constructors or "
        "Symop.create(operation_name,operation_argument) syntax",
        UserWarning,
        stacklevel=2,
    )
    return Symop.create(*args, **kwargs)

