"""Signal and variance container class (`sigvar`).

Ported from Herbert/Horace MATLAB @sigvar class
(herbert_core/utilities/classes/@sigvar).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from ..core.data_op_interface import (
    DataOpArray,
    _is_numeric,
    _is_object_or_object_array,
    data_op_interface,
    matlab_size,
    sigvar_size,
)
from ..core.serializable import Serializable, compare_values, register_serializable


def _format_size(sz: Sequence[int]) -> str:
    return "[" + " ".join(str(x) for x in sz) + "]"


@register_serializable
class sigvar(data_op_interface, Serializable):
    """Signal array and associated variances (standard errors squared).

    Create sigvar object:
        >> w = sigvar()           # Empty object
        >> w = sigvar(s)          # Signal but no variance
        >> w = sigvar(s, e)       # Signal and variance
        >> w = sigvar(s, e, msk)  # Signal, variance, and mask

    Parameters
    ----------
    s : array-like or float, optional
        Signal (numeric array).
    e : array-like or float, optional
        Variance array (same size as signal array).
        All elements must be greater or equal to zero.
        If an empty array, then variances are assumed to all be zero.
        If scalar (and signal non-empty) then all elements assumed to be same.
    msk : array-like or bool, optional
        Mask array (logical array same size as signal array).
        Elements are 0/False (mask) or 1/True (keep).
        If an empty array, then assumed all points are retained.
        If scalar (and signal non-empty) then all elements assumed to be same.
    """

    _signal: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=float))
    _variance: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=float))
    _mask: np.ndarray = PrivateAttr(default_factory=lambda: np.empty((0, 0), dtype=bool))
    _initialized: bool = PrivateAttr(default=False)

    def __new__(cls, *args: Any, **kwargs: Any) -> "sigvar":
        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, sigvar):
                return arg0.copy()
            if (
                isinstance(arg0, data_op_interface)
                and hasattr(arg0, "sigvar")
                and callable(arg0.sigvar)
            ):
                return arg0.sigvar()
            if isinstance(arg0, dict):
                return cls.loadobj(arg0)
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, sigvar) and self is not arg0:
                super().__init__()
                self._signal = arg0._signal.copy()
                self._variance = arg0._variance.copy()
                self._mask = arg0._mask.copy()
                self._initialized = arg0._initialized
                return
            if isinstance(arg0, (data_op_interface, dict)) and self._initialized:
                return

        super().__init__()
        narg = len(args) + len(kwargs)
        if narg == 0:
            return

        if len(args) == 1 and not kwargs:
            self.s = args[0]
        else:
            flds = ["s", "e", "msk"]
            _, remains = self.set_positional_and_key_val_arguments(
                flds, False, *args, **kwargs
            )
            if remains:
                raise ValueError(
                    f"HERBERT:sigvar:invalid_argument: "
                    f"Sigval constructor inputs: {remains} have not been recognized"
                )

    # ------------------------------------------------------------------
    # Dependent properties (s, e, msk, n_elements, sig_var, is_mask_defined)
    # ------------------------------------------------------------------
    @property
    def n_elements(self) -> int:
        """Number of elements in signal, error and mask arrays."""
        if self._signal.size == 0:
            return 0
        return int(self._signal.size)

    @property
    def s(self) -> np.ndarray:
        """Return signal array (numeric array)."""
        return self._signal

    @s.setter
    def s(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            if val is not None and not _is_numeric(val):
                raise ValueError("HERBERT:sigvar:invalid_argument: signal has to be numeric")
            self._signal = np.empty((0, 0), dtype=float)
        elif not _is_numeric(val):
            raise ValueError("HERBERT:sigvar:invalid_argument: signal has to be numeric")
        else:
            self._signal = np.asarray(val, dtype=float)
        self._initialized = True
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def e(self) -> np.ndarray:
        """Return variance array, expanding [] as zeros(size(s)) as required."""
        if self._variance.size == 0:
            return np.zeros(self._signal.shape, dtype=float)
        return self._variance

    @e.setter
    def e(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            if val is not None and not _is_numeric(val):
                raise ValueError(
                    "HERBERT:sigvar:invalid_argument: variance has to be numeric non-negative array"
                )
            self._variance = np.empty((0, 0), dtype=float)
        elif not _is_numeric(val):
            raise ValueError(
                "HERBERT:sigvar:invalid_argument: variance has to be numeric non-negative array"
            )
        else:
            arr = np.asarray(val, dtype=float)
            with np.errstate(invalid="ignore"):
                has_neg = bool(np.any(arr < 0))
            if has_neg:
                raise ValueError(
                    "HERBERT:sigvar:invalid_argument: variance has to be numeric non-negative array"
                )
            self._variance = arr
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def msk(self) -> np.ndarray:
        """Return mask array, expanding [] as true(size(s)) as required."""
        if self._mask.size == 0:
            return np.ones(self._signal.shape, dtype=bool)
        return self._mask

    @msk.setter
    def msk(self, val: Any) -> None:
        if val is None or (isinstance(val, (list, tuple, np.ndarray)) and np.asarray(val).size == 0):
            arr = np.empty((0, 0), dtype=bool)
        elif _is_numeric(val):
            num_arr = np.asarray(val, dtype=float).copy()
            if np.any(np.isnan(num_arr)):
                num_arr[np.isnan(num_arr)] = 0.0
            arr = num_arr.astype(bool)
            if np.all(arr):
                arr = np.empty((0, 0), dtype=bool)
            elif arr.size == 1 and not bool(arr.item()):
                arr = np.array(False, dtype=bool)
        else:
            raw = np.asarray(val)
            if raw.dtype == bool:
                arr = raw.copy()
                if np.all(arr):
                    arr = np.empty((0, 0), dtype=bool)
            else:
                raise ValueError(
                    "HERBERT:sigvar:invalid_argument: mask must be numeric or logical array"
                )

        if arr.size > 0 and np.all(arr):
            arr = np.empty((0, 0), dtype=bool)
        self._mask = arr
        if self.do_check_combo_arg:
            self.check_combo_arg()

    @property
    def sig_var(self) -> np.ndarray:
        """Return 2xN array of [signal; variance] row vectors."""
        s_row = self._signal.ravel(order="F")
        e_row = self._variance.ravel(order="F")
        if s_row.size == 0 and e_row.size == 0:
            return np.empty((2, 0), dtype=float)
        if e_row.size == 0:
            return s_row.reshape((1, -1))
        return np.vstack([s_row, e_row])

    @sig_var.setter
    def sig_var(self, val: Any) -> None:
        if not _is_numeric(val):
            raise ValueError(
                f"HERBERT:sigvar:invalid_argument: "
                f"sig_var property accepts only 2xN or Nx2 numeric array. "
                f"Assigning {type(val).__name__}"
            )
        arr = np.asarray(val, dtype=float)
        sz = matlab_size(arr)
        if not (sz[0] == 2 or sz[1] == 2) or len(sz) != 2:
            raise ValueError(
                f"HERBERT:sigvar:invalid_argument: "
                f"sig_var property accepts only 2xN or Nx2 numeric array. "
                f"Assigning {type(val).__name__} data with size {_format_size(sz)}"
            )
        if arr.ndim == 1:
            arr = arr.reshape(sz)
        if arr.shape[0] == 2:
            self._signal = arr[0, :].copy()
            self._variance = arr[1, :].copy()
        else:
            self._signal = arr[:, 0].copy()
            self._variance = arr[:, 1].copy()
        self._initialized = True

    @property
    def is_mask_defined(self) -> bool:
        """True if mask has been explicitly set (not all true / empty)."""
        return bool(self._mask.size > 0)

    # ------------------------------------------------------------------
    # Abstract / Protected methods from data_op_interface
    # ------------------------------------------------------------------
    def get_NUM_DIMS(self) -> int:
        """Return number of object's image dimensions (non-unit dimensions in signal array)."""
        sz = [d for d in matlab_size(self._signal) if d != 1]
        if len(sz) == 0:
            return 1
        return len(sz)

    def get_img_size(self) -> Tuple[int, ...]:
        """Get MATLAB-style size of dataset's image (size of signal array)."""
        return matlab_size(self._signal)

    # ------------------------------------------------------------------
    # Public sigvar methods
    # ------------------------------------------------------------------
    def copy(self) -> "sigvar":
        """Return a value copy of this sigvar object."""
        out = sigvar()
        out._signal = self._signal.copy()
        out._variance = self._variance.copy()
        out._mask = self._mask.copy()
        out._initialized = self._initialized
        return out

    def sigvar(self) -> "sigvar":
        """Return a sigvar copy of self."""
        return self.copy()

    def sigvar_size(self) -> Tuple[int, ...]:
        """MATLAB size of public signal and variance arrays (`>> sz = sigvar_size(w)`)."""
        return matlab_size(self._signal)

    def sigvar_set(self, sigvarobj: "sigvar") -> "sigvar":
        """Set output object signal and variance fields from input sigvar object.

        >> w = sigvar_set(w, sigvarobj)

        Parameters
        ----------
        sigvarobj : sigvar
            Object of class sigvar which provides the new signal and variance arrays.

        Returns
        -------
        sigvar
            Updated sigvar object.
        """
        if not isinstance(sigvarobj, sigvar):
            raise ValueError(
                "SIGVAR:binary_op_manager: Source of new signal and variance data must be a sigvar object"
            )
        if self.sigvar_size() != sigvarobj.sigvar_size():
            raise ValueError(
                "SIGVAR:binary_op_manager: sigvar object and sigvar object have inconsistent sizes"
            )
        self._signal = sigvarobj._signal.copy()
        self._variance = sigvarobj._variance.copy()
        self._mask = sigvarobj._mask.copy()
        self._initialized = sigvarobj._initialized
        return self

    def dimensions(self) -> Tuple[Union[int, Tuple[int, ...]], Tuple[int, ...]]:
        """Number and extent of the dimensions of the public signal array.

        >> [nd, sz] = dimensions(w)

        The convention for sigvar objects is:
        - if w.s empty:         nd=(), sz=()
        - if w.s scalar:        nd=0,  sz=()
        - if w.s column vector: nd=1,  sz=(length(w.s),)
        - if w.s row vector:    nd=2,  sz=size(w.s)
        - All other cases:      nd=len(size(w.s)), sz=size(w.s)

        Returns
        -------
        nd : int or tuple
            Number of dimensions of the object (or ``()`` if empty).
        sz : Tuple[int, ...]
            Tuple with the extent along each dimension.
        """
        if self._signal.size == 0:
            return (), ()
        if self._signal.size == 1:
            return 0, ()
        sz = matlab_size(self._signal)
        if sz[1] > 1:
            return len(sz), sz
        return 1, (int(self._signal.size),)

    def get(self, field: Optional[str] = None) -> Any:
        """Get a named property or struct of properties."""
        if field is None:
            return {"s": self.s, "e": self.e, "msk": self.msk}
        if not hasattr(self, field):
            raise AttributeError(f"sigvar has no property '{field}'")
        return getattr(self, field)

    def set(self, field: str, val: Any) -> "sigvar":
        """Set a named property and return self."""
        setattr(self, field, val)
        return self

    # ------------------------------------------------------------------
    # Binary operation manager for sigvar
    # ------------------------------------------------------------------
    @staticmethod
    def binary_op_manager_single(
        w1: Any,
        w2: Any,
        binary_op: Callable[..., Any],
        ignore_cell_sorting: bool = False,
    ) -> "sigvar":
        """Implement a binary operation between two scalar sigvar objects or numeric arrays.

        Parameters
        ----------
        w1, w2 : sigvar or numeric or data_op_interface
            Scalar objects or numeric arrays.
        binary_op : Callable
            Function handle to a binary operation on two sigvar objects.
        ignore_cell_sorting : bool, optional
            Unused, provided to adhere to common ``binary_op_manager`` interface.

        Returns
        -------
        sigvar
            Result of the binary operation.
        """
        sz1 = sigvar_size(w1)
        sz2 = sigvar_size(w2)
        if not (sz1 == sz2 or sz1 == (1, 1) or sz2 == (1, 1)):
            raise ValueError(
                f"HERBERT:sigvar:invalid_argument: "
                f"Sizes of signal arrays for operand1({_format_size(sz1)}) "
                f"and operand 2({_format_size(sz2)}) are inconsistent."
            )

        sv1 = w1.copy() if isinstance(w1, sigvar) else sigvar(w1)
        sv2 = w2.copy() if isinstance(w2, sigvar) else sigvar(w2)

        # If both operands have the same MATLAB size, align NumPy shapes (e.g. (N,) vs (1, N))
        if sz1 == sz2 and sv1._signal.shape != sv2._signal.shape and sv1._signal.size == sv2._signal.size:
            target_shape = sv1._signal.shape if isinstance(w1, sigvar) else sv2._signal.shape
            if sv1._signal.shape != target_shape:
                sv1._signal = sv1._signal.reshape(target_shape, order="F")
                if sv1._variance.size > 0:
                    sv1._variance = sv1._variance.reshape(target_shape, order="F")
            if sv2._signal.shape != target_shape:
                sv2._signal = sv2._signal.reshape(target_shape, order="F")
                if sv2._variance.size > 0:
                    sv2._variance = sv2._variance.reshape(target_shape, order="F")

        return binary_op(sv1, sv2)

    # ------------------------------------------------------------------
    # Binary operations (6 methods)
    # ------------------------------------------------------------------
    def plus(self, w2: Any) -> Any:
        """Implement w1 + w2 for sigvar objects (`>> w = w1 + w2`)."""
        return self.binary_op_manager(self, w2, _plus_single)

    def minus(self, w2: Any) -> Any:
        """Implement w1 - w2 for sigvar objects (`>> w = w1 - w2`)."""
        return self.binary_op_manager(self, w2, _minus_single)

    def mtimes(self, w2: Any) -> Any:
        """Implement w1 * w2 for sigvar objects (`>> w = w1 * w2`)."""
        return self.binary_op_manager(self, w2, _mtimes_single)

    def mrdivide(self, w2: Any) -> Any:
        """Implement w1 / w2 for sigvar objects (`>> w = w1 / w2`)."""
        return self.binary_op_manager(self, w2, _mrdivide_single)

    def mldivide(self, w2: Any) -> Any:
        """Implement w1 \\ w2 for sigvar objects (`>> w = w1 \\ w2`)."""
        return self.binary_op_manager(self, w2, _mldivide_single)

    def mpower(self, w2: Any) -> Any:
        """Implement w1 ^ w2 for sigvar objects (`>> w = w1 ^ w2`)."""
        return self.binary_op_manager(self, w2, _mpower_single)

    # ------------------------------------------------------------------
    # Unary operations (30 methods)
    # ------------------------------------------------------------------
    def acos(self) -> "sigvar":
        """Implement acos(w1) for a sigvar object (`>> w = acos(w1)`)."""
        s = np.arccos(self._signal)
        e = self._variance / np.abs(1.0 - s**2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def acosh(self) -> "sigvar":
        """Implement acosh(w1) for a sigvar object (`>> w = acosh(w1)`)."""
        s = np.arccosh(self._signal)
        e = self._variance / np.abs(s**2 - 1.0) if self._variance.size > 0 else None
        return sigvar(s, e)

    def acot(self) -> "sigvar":
        """Implement acot(w1) for a sigvar object (`>> w = acot(w1)`)."""
        s = np.arctan(1.0 / self._signal)
        e = self._variance / ((1.0 + s**2) ** 2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def acoth(self) -> "sigvar":
        """Implement acoth(w1) for a sigvar object (`>> w = acoth(w1)`)."""
        s = np.arctanh(1.0 / self._signal)
        e = self._variance / ((1.0 - s**2) ** 2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def acsc(self) -> "sigvar":
        """Implement acsc(w1) for a sigvar object (`>> w = acsc(w1)`)."""
        s = np.arcsin(1.0 / self._signal)
        e = self._variance / (np.abs(s**2 - 1.0) * (s**2)) if self._variance.size > 0 else None
        return sigvar(s, e)

    def acsch(self) -> "sigvar":
        """Implement acsch(w1) for a sigvar object (`>> w = acsch(w1)`)."""
        s = np.arcsinh(1.0 / self._signal)
        e = self._variance / ((s**2 + 1.0) * (s**2)) if self._variance.size > 0 else None
        return sigvar(s, e)

    def asec(self) -> "sigvar":
        """Implement asec(w1) for a sigvar object (`>> w = asec(w1)`)."""
        s = np.arccos(1.0 / self._signal)
        e = self._variance / (np.abs(s**2 - 1.0) * (s**2)) if self._variance.size > 0 else None
        return sigvar(s, e)

    def asech(self) -> "sigvar":
        """Implement asech(w1) for a sigvar object (`>> w = asech(w1)`)."""
        s = np.arccosh(1.0 / self._signal)
        e = self._variance / (np.abs(1.0 - s**2) * (s**2)) if self._variance.size > 0 else None
        return sigvar(s, e)

    def asin(self) -> "sigvar":
        """Implement asin(w1) for a sigvar object (`>> w = asin(w1)`)."""
        s = np.arcsin(self._signal)
        e = self._variance / np.abs(1.0 - s**2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def asinh(self) -> "sigvar":
        """Implement asinh(w1) for a sigvar object (`>> w = asinh(w1)`)."""
        s = np.arcsinh(self._signal)
        e = self._variance / (1.0 + s**2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def atan(self) -> "sigvar":
        """Implement atan(w1) for a sigvar object (`>> w = atan(w1)`)."""
        s = np.arctan(self._signal)
        e = self._variance / ((1.0 + s**2) ** 2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def atanh(self) -> "sigvar":
        """Implement atanh(w1) for a sigvar object (`>> w = atanh(w1)`)."""
        s = np.arctanh(self._signal)
        e = self._variance / ((1.0 - s**2) ** 2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def cos(self) -> "sigvar":
        """Implement cos(w1) for a sigvar object (`>> w = cos(w1)`)."""
        s = np.cos(self._signal)
        e = np.abs(1.0 - s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def cosh(self) -> "sigvar":
        """Implement cosh(w1) for a sigvar object (`>> w = cosh(w1)`)."""
        s = np.cosh(self._signal)
        e = np.abs(s**2 - 1.0) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def cot(self) -> "sigvar":
        """Implement cot(w1) for a sigvar object (`>> w = cot(w1)`)."""
        s = 1.0 / np.tan(self._signal)
        e = ((1.0 + s**2) ** 2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def coth(self) -> "sigvar":
        """Implement coth(w1) for a sigvar object (`>> w = coth(w1)`)."""
        s = 1.0 / np.tanh(self._signal)
        e = ((1.0 - s**2) ** 2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def csc(self) -> "sigvar":
        """Implement csc(w1) for a sigvar object (`>> w = csc(w1)`)."""
        s = 1.0 / np.sin(self._signal)
        e = np.abs(s**2 - 1.0) * (s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def csch(self) -> "sigvar":
        """Implement csch(w1) for a sigvar object (`>> w = csch(w1)`)."""
        s = 1.0 / np.sinh(self._signal)
        e = (s**2 + 1.0) * (s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def exp(self) -> "sigvar":
        """Implement exp(w1) for a sigvar object (`>> w = exp(w1)`)."""
        s = np.exp(self._signal)
        e = (s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def log(self) -> "sigvar":
        """Implement log(w1) for a sigvar object (`>> w = log(w1)`)."""
        s = np.log(self._signal)
        e = self._variance / (self._signal**2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def log10(self) -> "sigvar":
        """Implement log10(w1) for a sigvar object (`>> w = log10(w1)`)."""
        s = np.log10(self._signal)
        e = self._variance / ((self._signal * np.log(10.0)) ** 2) if self._variance.size > 0 else None
        return sigvar(s, e)

    def sec(self) -> "sigvar":
        """Implement sec(w1) for a sigvar object (`>> w = sec(w1)`)."""
        s = 1.0 / np.cos(self._signal)
        e = np.abs(s**2 - 1.0) * (s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def sech(self) -> "sigvar":
        """Implement sech(w1) for a sigvar object (`>> w = sech(w1)`)."""
        s = 1.0 / np.cosh(self._signal)
        e = np.abs(1.0 - s**2) * (s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def sin(self) -> "sigvar":
        """Implement sin(w1) for a sigvar object (`>> w = sin(w1)`)."""
        s = np.sin(self._signal)
        e = np.abs(1.0 - s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def sinh(self) -> "sigvar":
        """Implement sinh(w1) for a sigvar object (`>> w = sinh(w1)`)."""
        s = np.sinh(self._signal)
        e = (1.0 + s**2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def sqrt(self) -> "sigvar":
        """Implement sqrt(w1) for a sigvar object (`>> w = sqrt(w1)`)."""
        s = np.sqrt(self._signal)
        e = self._variance / (4.0 * self._signal) if self._variance.size > 0 else None
        return sigvar(s, e)

    def tan(self) -> "sigvar":
        """Implement tan(w1) for a sigvar object (`>> w = tan(w1)`)."""
        s = np.tan(self._signal)
        e = ((1.0 + s**2) ** 2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def tanh(self) -> "sigvar":
        """Implement tanh(w1) for a sigvar object (`>> w = tanh(w1)`)."""
        s = np.tanh(self._signal)
        e = ((1.0 - s**2) ** 2) * self._variance if self._variance.size > 0 else None
        return sigvar(s, e)

    def uminus(self) -> "sigvar":
        """Implement -w1 for a sigvar object (`>> w = -w1`)."""
        s = -self._signal
        e = self._variance.copy() if self._variance.size > 0 else None
        return sigvar(s, e)

    def uplus(self) -> "sigvar":
        """Implement +w1 for a sigvar object (`>> w = +w1`)."""
        return self.copy()

    # ------------------------------------------------------------------
    # Serializable interface
    # ------------------------------------------------------------------
    def check_combo_arg(self) -> "sigvar":
        """Validate signal, variance, and mask consistency."""
        if self._signal.size == 0:
            self._variance = np.empty((0, 0), dtype=float)
            self._mask = np.empty((0, 0), dtype=bool)
            return self

        sig_sz = matlab_size(self._signal)

        if self._variance.size == 1 and self._signal.size != 1:
            self._variance = float(self._variance.item()) * np.ones(self._signal.shape, dtype=float)
        if self._variance.size > 0 and matlab_size(self._variance) != sig_sz:
            if self._variance.size == self._signal.size:
                self._variance = self._variance.reshape(self._signal.shape, order="F")
            else:
                raise ValueError(
                    f"HERBERT:sigvar:invalid_argument: "
                    f"Error array should contain single element or its size to be equal "
                    f"the size signal array: {_format_size(sig_sz)}. "
                    f"It is: {_format_size(matlab_size(self._variance))}"
                )
        elif self._variance.size > 0 and self._variance.shape != self._signal.shape:
            self._variance = self._variance.reshape(self._signal.shape, order="F")

        if self._mask.size > 0:
            if self._mask.size == 1 and self._signal.size != 1:
                self._mask = np.zeros(self._signal.shape, dtype=bool)
            elif matlab_size(self._mask) != sig_sz:
                if self._mask.size == self._signal.size:
                    self._mask = self._mask.reshape(self._signal.shape, order="F")
                else:
                    raise ValueError(
                        f"HERBERT:sigvar:invalid_argument: "
                        f"Mask array should contain single element or its size be equal "
                        f"to the size of signal array: {_format_size(sig_sz)}. "
                        f"It is: {_format_size(matlab_size(self._mask))}"
                    )
            elif self._mask.shape != self._signal.shape:
                self._mask = self._mask.reshape(self._signal.shape, order="F")
        return self

    def class_version(self) -> int:
        """Define version of the class to store in mat-files and nxsqw data format."""
        return 2

    def saveable_fields(self) -> Sequence[str]:
        """Return list of independent properties saved by Serializable."""
        if not self._initialized:
            return ["s", "e", "msk"]
        flds = ["s"]
        if self._variance.size > 0:
            flds.append("e")
        if self._mask.size > 0:
            flds.append("msk")
        return flds

    def from_old_struct(
        self, inputs: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ) -> Union["sigvar", List["sigvar"]]:
        """Restore object from an old structure describing previous version(s) of sigvar."""
        if isinstance(inputs, dict):
            s_in = dict(inputs)
            if "signal_" in s_in:
                s_in["s"] = s_in.pop("signal_")
                s_in["e"] = s_in.pop("variance_", [])
                s_in["msk"] = s_in.pop("mask_", [])
            if "array_dat" in s_in:
                return self.from_bare_struct(s_in["array_dat"])
            return self.from_bare_struct(s_in)
        return super().from_old_struct(inputs)

    def equal_to_tol(
        self,
        other: Any,
        tol: Union[float, Sequence[float]] = (0.0, 0.0),
        nan_equal: bool = True,
        ignore_str: bool = False,
        name_a: str = "obj1",
        name_b: str = "obj2",
    ) -> Tuple[bool, str]:
        """Compare two sigvar objects within tolerance across s, e, and msk."""
        if type(self) is not type(other):
            return (
                False,
                f"Objects are of different classes: '{type(self).__name__}' and '{type(other).__name__}'",
            )

        # Ensure variance/mask definedness and values match across union of saveable fields
        flds = list(dict.fromkeys(list(self.saveable_fields()) + list(other.saveable_fields())))
        for fld in flds:
            val1 = getattr(self, fld)
            val2 = getattr(other, fld)
            sz1 = matlab_size(val1)
            sz2 = matlab_size(val2)
            if sz1 != sz2:
                return (
                    False,
                    f"{name_a}.{fld} and {name_b}.{fld}: MATLAB array sizes differ: {sz1} vs {sz2}",
                )
            arr1 = np.asarray(val1)
            arr2 = np.asarray(val2)
            if arr1.shape != arr2.shape and arr1.size == arr2.size:
                arr2 = arr2.reshape(arr1.shape, order="F")
            iseq, mess = compare_values(
                arr1,
                arr2,
                tol=tol,
                nan_equal=nan_equal,
                ignore_str=ignore_str,
                name_a=f"{name_a}.{fld}",
                name_b=f"{name_b}.{fld}",
            )
            if not iseq:
                return False, mess
        return True, ""


# ----------------------------------------------------------------------
# Private single-object binary operations (@sigvar/private/*_single.m)
# ----------------------------------------------------------------------
def _plus_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 + w2 for scalar sigvar objects (`>> w = plus_single(w1, w2)`)."""
    s = w1._signal + w2._signal
    if w1._variance.size > 0 and w2._variance.size > 0:
        e: Any = w1._variance + w2._variance
    elif w1._variance.size > 0:
        e = w1._variance
    elif w2._variance.size > 0:
        e = w2._variance
    else:
        e = None
    return sigvar(s, e)


def _minus_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 - w2 for scalar sigvar objects (`>> w = minus_single(w1, w2)`)."""
    s = w1._signal - w2._signal
    if w1._variance.size > 0 and w2._variance.size > 0:
        e: Any = w1._variance + w2._variance
    elif w1._variance.size > 0:
        e = w1._variance
    elif w2._variance.size > 0:
        e = w2._variance
    else:
        e = None
    return sigvar(s, e)


def _mtimes_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 * w2 for scalar sigvar objects (`>> w = mtimes_single(w1, w2)`)."""
    s = w1._signal * w2._signal
    if w1._variance.size > 0 and w2._variance.size > 0:
        e: Any = (w2._signal**2) * w1._variance + (w1._signal**2) * w2._variance
    elif w1._variance.size > 0:
        e = (w2._signal**2) * w1._variance
    elif w2._variance.size > 0:
        e = (w1._signal**2) * w2._variance
    else:
        e = None
    return sigvar(s, e)


def _mrdivide_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 / w2 for scalar sigvar objects (`>> w = mrdivide_single(w1, w2)`)."""
    s = w1._signal / w2._signal
    if w1._variance.size > 0 and w2._variance.size > 0:
        e: Any = w1._variance / (w2._signal**2) + w2._variance * ((s / w2._signal) ** 2)
    elif w1._variance.size > 0:
        e = w1._variance / (w2._signal**2)
    elif w2._variance.size > 0:
        e = w2._variance * ((s / w2._signal) ** 2)
    else:
        e = None
    return sigvar(s, e)


def _mldivide_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 \\ w2 for scalar sigvar objects (`>> w = mldivide_single(w1, w2)`)."""
    s = w2._signal / w1._signal
    if w2._variance.size > 0 and w1._variance.size > 0:
        e: Any = w2._variance / (w1._signal**2) + w1._variance * ((s / w1._signal) ** 2)
    elif w2._variance.size > 0:
        e = w2._variance / (w1._signal**2)
    elif w1._variance.size > 0:
        e = w1._variance * ((s / w1._signal) ** 2)
    else:
        e = None
    return sigvar(s, e)


def _mpower_single(w1: sigvar, w2: sigvar) -> sigvar:
    """Implement w1 ^ w2 for scalar sigvar objects (`>> w = mpower_single(w1, w2)`)."""
    tmp = w1._signal ** (w2._signal - 1.0)
    s = tmp * w1._signal
    if w1._variance.size > 0 and w2._variance.size > 0:
        e: Any = ((w2._signal * tmp) ** 2) * w1._variance + ((s * np.log(w1._signal)) ** 2) * w2._variance
    elif w1._variance.size > 0:
        e = (w2._signal**2) * w1._variance
    elif w2._variance.size > 0:
        e = (w1._signal**2) * w2._variance
    else:
        e = None
    return sigvar(s, e)


# PascalCase alias
SigVar = sigvar
