"""Abstract detector base class and shared detector helper utilities."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from pyhorace.core.hashable import HashableObj
from pyhorace.core.serializable import Serializable


def _trim_matlab_size(sz: tuple[int, ...]) -> tuple[int, ...]:
    """Normalize a shape tuple to MATLAB size conventions (at least 2D, no trailing 1s beyond 2D)."""
    sz_list = list(sz)
    while len(sz_list) > 2 and sz_list[-1] == 1:
        sz_list.pop()
    while len(sz_list) < 2:
        sz_list.append(1)
    return tuple(sz_list)


def matlab_size(val: Any) -> tuple[int, ...]:
    """Return MATLAB-equivalent size tuple for a scalar or array."""
    if val is None:
        return (0, 0)
    arr = np.asarray(val)
    if arr.size == 0:
        if arr.ndim == 2:
            return tuple(arr.shape)
        return (0, 0)
    if arr.ndim == 0 or arr.size == 1:
        return (1, 1)
    if arr.ndim == 1:
        return (1, int(arr.shape[0]))
    return _trim_matlab_size(tuple(int(d) for d in arr.shape))


def is_matlab_scalar(val: Any) -> bool:
    """Return True if val is a scalar or single-element array."""
    if val is None:
        return False
    return np.asarray(val).size == 1


def size_array_stack(sz1: tuple[int, ...], sz2: tuple[int, ...]) -> tuple[int, ...]:
    """Size of an array formed by stacking arrays of size sz1 according to an outer array of size sz2.

    Equivalent to MATLAB ``size_array_stack(sz1, sz2)`` combined with MATLAB's
    automatic stripping of trailing singleton dimensions beyond 2D.
    """
    sz1 = _trim_matlab_size(sz1)
    sz2 = _trim_matlab_size(sz2)
    if sz1 == (1, 1):
        s = list(sz2)
    elif sz2 == (1, 1):
        s = list(sz1)
    else:
        end1 = 1 if (len(sz1) == 2 and sz1[1] == 1) else len(sz1)
        beg2 = 1 if (len(sz2) == 2 and sz2[0] == 1) else 0
        s = list(sz1[:end1]) + list(sz2[beg2:])
    if len(s) == 1:
        s.append(1)
    return _trim_matlab_size(tuple(s))


def size_array_split(sz_stack: tuple[int, ...], sz2: tuple[int, ...]) -> tuple[int, ...]:
    """Find the size of the stacked sub-arrays when given the size of the outer stacking array.

    Equivalent to MATLAB ``size_array_split(sz_stack, sz2)``.
    """
    sz_stack = _trim_matlab_size(sz_stack)
    sz2 = _trim_matlab_size(sz2)
    if sz2 == (1, 1):
        return sz_stack

    if len(sz2) == 2 and sz2[0] == 1:
        n = 1
        if len(sz_stack) >= n and sz_stack[-1] == sz2[1]:
            if len(sz_stack) == n:
                sz1 = (1, 1)
            else:
                rem = list(sz_stack[:-n])
                if len(rem) == 1:
                    rem.append(1)
                sz1 = _trim_matlab_size(tuple(rem))
            if size_array_stack(sz1, sz2) == sz_stack:
                return sz1
    elif len(sz2) == 2 and sz2[1] == 1:
        if sz_stack == sz2:
            sz1 = (1, 1)
            if size_array_stack(sz1, sz2) == sz_stack:
                return sz1
        n = 1
        if len(sz_stack) >= n and sz_stack[-1] == sz2[0]:
            if len(sz_stack) == n:
                sz1 = (1, 1)
            else:
                rem = list(sz_stack[:-n])
                if len(rem) == 1:
                    rem.append(1)
                sz1 = _trim_matlab_size(tuple(rem))
            if size_array_stack(sz1, sz2) == sz_stack:
                return sz1
    else:
        n = len(sz2)
        if len(sz_stack) >= n and sz_stack[-n:] == sz2:
            if len(sz_stack) == n:
                sz1 = (1, 1)
            else:
                rem = list(sz_stack[:-n])
                if len(rem) == 1:
                    rem.append(1)
                sz1 = _trim_matlab_size(tuple(rem))
            if size_array_stack(sz1, sz2) == sz_stack:
                return sz1

    raise ValueError(
        "HERBERT:size_array_split:invalid_argument: The array to be split does not "
        "have dimensions of a stack ending with the second input array"
    )


def expand_args_by_ref(aref: Any, *args: Any) -> list[np.ndarray]:
    """Expand scalar or array arguments to match the size of reference argument aref.

    Equivalent to MATLAB ``expand_args_by_ref(Aref, varargin{:})``.
    """
    ref_arr = np.asarray(aref)
    nref = ref_arr.size
    sz_ref = matlab_size(ref_arr)
    out: list[np.ndarray] = []

    if nref == 0:
        for arg in args:
            a = np.asarray(arg)
            if a.size != 0:
                raise ValueError(
                    "HERBERT:expand_args_by_ref:invalid_argument: "
                    "Arguments must all be empty if the reference object is empty"
                )
            out.append(a)
        return out

    if nref == 1:
        for arg in args:
            a = np.asarray(arg)
            if a.size != 1:
                raise ValueError(
                    "HERBERT:expand_args_by_ref:invalid_argument: "
                    "Arguments must all be scalar if the reference object is scalar"
                )
            out.append(a.reshape(sz_ref, order="F"))
        return out

    for arg in args:
        a = np.asarray(arg)
        if a.size == 1:
            out.append(np.full(sz_ref, a.item(), dtype=a.dtype))
        elif a.size == nref:
            out.append(a.reshape(sz_ref, order="F"))
        else:
            raise ValueError(
                "HERBERT:expand_args_by_ref:invalid_argument: "
                "Arguments must all be scalar or non-empty arrays with the "
                "same number of elements as reference object"
            )
    return out


def is_integer_id(id_in: Any) -> tuple[bool, str, np.ndarray, bool, np.ndarray]:
    """Check that identifiers are unique positive integers and return sorting index.

    Equivalent to MATLAB ``[ok, mess, ix, perm, id] = is_integer_id(id_in)``.
    Note that ``ix`` contains 1-based indices (or an empty array if already sorted).
    """
    if id_in is None or isinstance(id_in, (str, bytes, bool)):
        return False, "identifiers empty or invalid type", np.array([], dtype=int), False, np.array([], dtype=int)

    try:
        arr = np.asarray(id_in)
    except Exception:
        return False, "identifiers empty or invalid type", np.array([], dtype=int), False, np.array([], dtype=int)

    if arr.size == 0 or arr.dtype.kind not in ("i", "u", "f"):
        return False, "identifiers empty or invalid type", np.array([], dtype=int), False, np.array([], dtype=int)

    flat_f = arr.ravel(order="F")
    if not np.all(np.isfinite(flat_f)) or not np.all(flat_f > 0) or not np.all(np.rem(flat_f, 1) == 0 if hasattr(np, "rem") else (flat_f % 1 == 0)):
        return (
            False,
            "identifiers must have integer values and be greater than zero",
            np.array([], dtype=int),
            False,
            flat_f.reshape(-1, 1),
        )

    id_col = flat_f.astype(np.int64)
    n = id_col.size
    if n == 1:
        perm = bool(id_col[0] == 1)
        return True, "", np.array([], dtype=int), perm, id_col.reshape(-1, 1)

    diffs = np.diff(id_col)
    if np.all(diffs >= 0):
        ix = np.array([], dtype=int)
        perm = bool(id_col[0] == 1 and id_col[-1] == n)
        sorted_id = id_col
    elif np.all(diffs <= 0):
        ix = np.arange(n, 0, -1, dtype=int).reshape(-1, 1)
        perm = bool(id_col[-1] == 1 and id_col[0] == n)
        sorted_id = id_col[::-1]
    else:
        order = np.argsort(id_col, kind="mergesort")
        ix = (order + 1).astype(int).reshape(-1, 1)
        sorted_id = id_col[order]
        perm = bool(sorted_id[0] == 1 and sorted_id[-1] == n)

    if np.any(np.diff(sorted_id) == 0):
        return (
            False,
            "identifiers must be unique",
            np.array([], dtype=int),
            False,
            sorted_id.reshape(-1, 1),
        )

    return True, "", ix, perm, sorted_id.reshape(-1, 1)


def format_scalar_result(val: Any, sz: tuple[int, ...]) -> float | np.ndarray:
    """Format scalar-per-detector result into float (if sz == (1, 1)) or ndarray of shape sz."""
    arr = np.asarray(val, dtype=float)
    if sz == (1, 1):
        return float(arr.item())
    return arr.reshape(sz, order="F")


def _rand_truncexp2_small(x0: np.ndarray) -> np.ndarray:
    """Generate truncated exponential random numbers in the small x0 (< 0.01) regime."""
    sz = x0.shape
    x = x0 * np.random.rand(*sz)
    y = np.random.rand(*sz)
    reject = y > np.exp(-x)
    while np.any(reject):
        n_rej = int(np.sum(reject))
        x_sub = x0[reject] * np.random.rand(n_rej)
        y_sub = np.random.rand(n_rej)
        x[reject] = x_sub
        reject_sub = y_sub > np.exp(-x_sub)
        new_reject = np.zeros_like(reject)
        new_reject[reject] = reject_sub
        reject = new_reject
    return x


def rand_truncexp2(x0_in: Any) -> np.ndarray:
    """Generate random numbers from a truncated exponential distribution on (0, x0).

    Equivalent to MATLAB ``rand_truncexp2(x0)``.
    """
    x0 = np.asarray(x0_in, dtype=float)
    if np.any(x0 < 0) or np.any(np.isnan(x0)):
        raise ValueError("The limit(s) of the truncated exponential(s) must be in range 0 to +Inf")

    isx0inf = np.isinf(x0)
    isx0small = x0 < 0.01

    ninf = int(np.sum(isx0inf))
    nsmall = int(np.sum(isx0small))
    nmed = x0.size - ninf - nsmall

    if nmed == x0.size:
        return -np.log(1.0 - (1.0 - np.exp(-x0)) * np.random.rand(*x0.shape))
    if ninf == x0.size:
        return -np.log(np.random.rand(*x0.shape))
    if nsmall == x0.size:
        return _rand_truncexp2_small(x0)

    out = np.full(x0.shape, np.nan, dtype=float)
    if nmed > 0:
        isx0med = ~(isx0inf | isx0small)
        out[isx0med] = -np.log(
            1.0 - (1.0 - np.exp(-x0[isx0med])) * np.random.rand(nmed)
        )
    if ninf > 0:
        out[isx0inf] = -np.log(np.random.rand(ninf))
    if nsmall > 0:
        out[isx0small] = _rand_truncexp2_small(x0[isx0small])
    return out



class IX_det_abstractType(HashableObj):
    """Abstract base class for detector types (He3 tube, slab, TobyfitClassic)."""

    _tobyfit_type: ClassVar[Any] = None
    _tobyfit_parameters: ClassVar[Any] = None

    @property
    def ndet(self) -> int:
        return self.get_ndet()

    @staticmethod
    def tobyfit_defaults(val: Any = None, parms: Any = None) -> tuple[Any, Any]:
        """Store and retrieve defaults for detector types."""
        if val is not None:
            IX_det_abstractType._tobyfit_type = val
            if isinstance(parms, dict):
                IX_det_abstractType._tobyfit_parameters = parms
        return IX_det_abstractType._tobyfit_type, IX_det_abstractType._tobyfit_parameters

    def expand_internal_properties_to_max_length(self, flds: list[str]) -> IX_det_abstractType:
        """Expand all listed properties to the maximal length among them."""
        old_state = self.do_check_combo_arg
        self.do_check_combo_arg = False
        try:
            vals = [getattr(self, f) for f in flds]
            num_elements = [np.asarray(v).size for v in vals]
            max_len = max(num_elements) if num_elements else 0
            if not all(n == max_len for n in num_elements):
                max_ind = next(i for i, n in enumerate(num_elements) if n == max_len)
                ref_val = vals[max_ind]
                for i, f in enumerate(flds):
                    if num_elements[i] != max_len:
                        expanded = expand_args_by_ref(ref_val, vals[i])[0]
                        setattr(self, f, expanded)
        finally:
            self.do_check_combo_arg = old_state
        return self

    @abstractmethod
    def get_ndet(self) -> int:
        """Return number of detectors."""

    @abstractmethod
    def reorder(self, ix: Any) -> IX_det_abstractType:
        """Reorder the detector elements according to indexing array ix."""

    @abstractmethod
    def replicate(self, n: int) -> IX_det_abstractType:
        """Replicate the detector elements n times."""

    @abstractmethod
    def effic(self, *args: Any) -> float | np.ndarray:
        """Efficiency of detector(s)."""

    @abstractmethod
    def mean_d(self, *args: Any) -> float | np.ndarray:
        """Mean depth of absorption along the neutron path."""

    @abstractmethod
    def mean_x(self, *args: Any) -> float | np.ndarray:
        """Mean x-coordinate of absorption in detector frame."""

    @abstractmethod
    def mean_y(self, *args: Any) -> float | np.ndarray:
        """Mean y-coordinate of absorption in detector frame."""

    @abstractmethod
    def mean_z(self, *args: Any) -> float | np.ndarray:
        """Mean z-coordinate of absorption in detector frame."""

    @abstractmethod
    def mean(self, *args: Any) -> np.ndarray:
        """Mean position [x, y, z]^T of absorption in detector frame."""

    @abstractmethod
    def var_d(self, *args: Any) -> float | np.ndarray:
        """Variance of depth of absorption along the neutron path."""

    @abstractmethod
    def var_x(self, *args: Any) -> float | np.ndarray:
        """Variance of x-coordinate of absorption in detector frame."""

    @abstractmethod
    def var_y(self, *args: Any) -> float | np.ndarray:
        """Variance of y-coordinate of absorption in detector frame."""

    @abstractmethod
    def var_z(self, *args: Any) -> float | np.ndarray:
        """Variance of z-coordinate of absorption in detector frame."""

    @abstractmethod
    def covariance(self, *args: Any) -> np.ndarray:
        """Covariance matrix (3x3) of absorption position in detector frame."""

    @abstractmethod
    def rand(self, *args: Any) -> np.ndarray:
        """Random points [x, y, z]^T in detector(s)."""

    # ------------------------------------------------------------------
    # Argument parsing helpers
    # ------------------------------------------------------------------
    def parse_ind_wvec_(self, *args: Any) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Check that the detector indices and wavevectors are consistently sized arrays.

        Equivalent to MATLAB ``[sz, ind, wvec] = parse_ind_wvec_(obj, varargin{:})``.
        """
        if len(args) == 1:
            wvec_in = args[0]
            if wvec_in is None or np.asarray(wvec_in).size == 0:
                raise ValueError(
                    "HERBERT:parse_ind_wvec_:invalid_argument: Input argument(s) must not be empty"
                )
            wvec = np.asarray(wvec_in, dtype=float)
            ind = np.arange(1, self.ndet + 1, dtype=int).reshape(1, -1)
            if ind.size != 1 and wvec.size != 1 and ind.size != wvec.size:
                raise ValueError(
                    "HERBERT:parse_ind_wvec_:invalid_argument: "
                    "If 'wvec' only is provided, and it is an array, then "
                    "the number of elements must equal the number of detectors "
                    "if there is more than one"
                )
            sz = matlab_size(wvec_in) if wvec.size != 1 else matlab_size(ind)
            return sz, ind, wvec

        if len(args) == 2:
            ind_in, wvec_in = args
            if (
                ind_in is None
                or wvec_in is None
                or np.asarray(ind_in).size == 0
                or np.asarray(wvec_in).size == 0
            ):
                raise ValueError(
                    "HERBERT:parse_ind_wvec_:invalid_argument: Input argument(s) must not be empty"
                )
            ind = np.asarray(ind_in, dtype=int)
            wvec = np.asarray(wvec_in, dtype=float)
            if ind.size != 1 and wvec.size != 1 and ind.size != wvec.size:
                raise ValueError(
                    "HERBERT:parse_ind_wvec_:invalid_argument: "
                    "If 'ind' and 'wvec' are both arrays, they must "
                    "have the same number of elements"
                )
            sz = matlab_size(wvec_in) if wvec.size != 1 else matlab_size(ind_in)
            return sz, ind, wvec

        raise ValueError(
            "HERBERT:parse_ind_wvec_:invalid_argument: Check the number of input arguments"
        )

    _parse_ind_wvec_ = parse_ind_wvec_

    def parse_ind_npath_wvec_(
        self, *args: Any
    ) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
        """Check detector indices, neutron path vectors, and wavevectors.

        Equivalent to MATLAB ``[sz, ind, npath, wvec] = parse_ind_npath_wvec_(obj, varargin{:})``.
        """
        narg = len(args)
        if narg == 2:
            sz, ind, wvec = self.parse_ind_wvec_(args[1])
            npath_in = args[0]
        elif narg == 3:
            sz, ind, wvec = self.parse_ind_wvec_(args[0], args[2])
            npath_in = args[1]
        else:
            raise ValueError(
                "HERBERT:parse_ind_npath_wvec_:invalid_argument: "
                "Check the size and shape of 'npath': must a vector length 3 or array size [3,n]"
            )

        if npath_in is None:
            raise ValueError(
                "HERBERT:parse_ind_npath_wvec_:invalid_argument: "
                "Check the size and shape of 'npath': must a vector length 3 or array size [3,n]"
            )
        npath = np.asarray(npath_in, dtype=float)
        if npath.size == 3 and (ind.size != 3 or npath.shape != (3, 3)):
            npath = npath.ravel(order="F").reshape(3, 1)
        elif npath.ndim != 2 or npath.shape != (3, ind.size):
            raise ValueError(
                "HERBERT:parse_ind_npath_wvec_:invalid_argument: "
                "Check the size and shape of 'npath': must a vector length 3 or array size [3,n]"
            )

        nlen = np.linalg.norm(npath, ord=2, axis=0, keepdims=True)
        if np.all(nlen > 1e-12):
            npath = npath / nlen
        else:
            raise ValueError(
                "HERBERT:parse_ind_npath_wvec_:invalid_argument: "
                "One or more neutron path direction vectors has (near) zero length"
            )

        return sz, ind, npath, wvec

    _parse_ind_npath_wvec_ = parse_ind_npath_wvec_

