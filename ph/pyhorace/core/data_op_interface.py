"""Abstract interface for unary and binary data operations.

Ported from Herbert/Horace MATLAB @data_op_interface class
(herbert_core/utilities/classes/@data_op_interface) and size_array_split
(herbert_core/utilities/general/size_array_split.m).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np


def matlab_size(x: Any) -> Tuple[int, ...]:
    """Return the MATLAB-style size tuple (at least 2 dimensions) of an array or scalar.

    Rules matching MATLAB's ``size(x)``:
    - ``None`` or empty 0D/1D array -> ``(0, 0)``
    - Scalar number or 0D/1-element 1D array -> ``(1, 1)``
    - 1D array of length ``N`` -> ``(1, N)`` (row vector)
    - 2D+ array -> ``x.shape`` with trailing singleton dimensions beyond dimension 2 stripped
    """
    if x is None:
        return (0, 0)
    if isinstance(x, data_op_interface):
        return (1, 1)
    if isinstance(x, (int, float, complex, np.number)) and not isinstance(x, bool):
        return (1, 1)
    arr = np.asarray(x)
    if arr.size == 0:
        if arr.ndim < 2:
            return (0, 0)
        shape = list(arr.shape)
        while len(shape) > 2 and shape[-1] == 1:
            shape.pop()
        return tuple(shape)
    if arr.ndim == 0:
        return (1, 1)
    if arr.ndim == 1:
        return (1, int(arr.shape[0]))
    shape = list(arr.shape)
    while len(shape) > 2 and shape[-1] == 1:
        shape.pop()
    return tuple(shape)


def _is_numeric(x: Any) -> bool:
    """Return True if x is a numeric scalar or numeric array (not bool or object)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, complex, np.number)):
        return True
    if isinstance(x, DataOpArray) or isinstance(x, data_op_interface):
        return False
    if isinstance(x, np.ndarray):
        return bool(np.issubdtype(x.dtype, np.number))
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return True
        try:
            arr = np.asarray(x)
            return bool(np.issubdtype(arr.dtype, np.number)) and arr.dtype != bool
        except Exception:
            return False
    return False


def _is_object_or_object_array(x: Any) -> bool:
    """Return True if x is a data_op_interface object or an array/sequence of them."""
    if isinstance(x, data_op_interface):
        return True
    if isinstance(x, DataOpArray):
        return True
    if isinstance(x, np.ndarray) and x.dtype == object and x.size > 0:
        return all(isinstance(el, data_op_interface) for el in x.flat)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        try:
            arr = np.asarray(x, dtype=object)
            return arr.size > 0 and all(isinstance(el, data_op_interface) for el in arr.flat)
        except Exception:
            return False
    return False


def sigvar_size(w: Any) -> Tuple[int, ...]:
    """Find size of signal array in an object or numeric array.

    >> sz = sigvar_size(w)

    Parameters
    ----------
    w : Any
        Object implementing ``sigvar_size`` or a numeric scalar/array.

    Returns
    -------
    Tuple[int, ...]
        MATLAB-style size tuple (at least 2 dimensions) of the signal array.
    """
    if isinstance(w, data_op_interface) and hasattr(w, "sigvar_size") and callable(w.sigvar_size):
        return w.sigvar_size()
    if _is_numeric(w):
        return matlab_size(w)
    raise ValueError(
        "HERBERT:sigvar_size:invalid_argument: "
        f"Numeric types or derived from data_op_interface class expected; got {type(w).__name__}"
    )


def _is_a_size(sz: Any) -> bool:
    """Check that sz is a valid MATLAB size vector (1D/row of >= 2 non-negative integers)."""
    if not isinstance(sz, (tuple, list, np.ndarray)):
        return False
    arr = np.asarray(sz)
    if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[0] != 1):
        return False
    arr = arr.ravel()
    if arr.size < 2:
        return False
    if arr.dtype == bool or not np.issubdtype(arr.dtype, np.number):
        return False
    return bool(np.all(arr >= 0) and np.all(arr % 1 == 0))


def _true_size(sz: Sequence[int]) -> List[int]:
    """Return true size once irrelevant trailing singletons have been stripped."""
    sz_list = [int(x) for x in sz]
    while sz_list and sz_list[-1] == 1:
        sz_list.pop()
    return sz_list


def size_array_split(sz_full: Sequence[int], sz_stack: Sequence[int]) -> Tuple[int, ...]:
    """Return the size of the root array in a stack of those arrays.

    >> sz_root = size_array_split(sz_full, sz_stack)

    Parameters
    ----------
    sz_full : Sequence[int]
        Size of the stacked array.
    sz_stack : Sequence[int]
        Size of the array by which to stack arrays of size ``sz_root``,
        as obtained by using the MATLAB intrinsic function ``size``.

    Returns
    -------
    Tuple[int, ...]
        Size of the arrays that were stacked, as obtained by using
        the MATLAB intrinsic function ``size``.

    Notes
    -----
    This is the inverse of the function ``size_array_stack``.
    Trailing singleton dimensions are irrelevant to the dimensionality of an array
    (e.g. ``[2, 1]`` has dimensions 1, ``[1, 1]`` has dimensions 0 (a scalar),
    ``[1, 0]`` has dimensions 2 because dimension extent 0 is significant).
    """
    if not _is_a_size(sz_full):
        raise ValueError(
            "HERBERT:size_array_split:invalid_argument: full array size has invalid form"
        )
    if not _is_a_size(sz_stack):
        raise ValueError(
            "HERBERT:size_array_split:invalid_argument: stack array size has invalid form"
        )

    sz_full_t = _true_size(np.asarray(sz_full).ravel())
    sz_stack_t = _true_size(np.asarray(sz_stack).ravel())

    if len(sz_full_t) < len(sz_stack_t):
        raise ValueError(
            "HERBERT:size_array_split:invalid_argument: "
            "The number of dimensions of the stacking array is larger than that of the full array"
        )

    ind = next((i for i, v in enumerate(sz_stack_t) if v != 1), None)

    if ind is not None:
        # The stack array has at least one non-singleton dimension
        n = len(sz_stack_t) - ind

        # The dimensions trailing leading singletons of the stack array must
        # match the trailing dimensions of the full array
        if sz_full_t[len(sz_full_t) - n :] != sz_stack_t[len(sz_stack_t) - n :]:
            raise ValueError(
                "HERBERT:size_array_split:invalid_argument: "
                "The full array cannot be resolved into a stack of arrays with the given stack size"
            )

        # Strip away the stack dimensions
        sz_root = sz_full_t[: len(sz_full_t) - n]

        if len(sz_stack_t) == len(sz_full_t):
            sz_root = _true_size(sz_root)
        elif len(sz_root) > 0 and sz_root[-1] == 1:
            raise ValueError(
                "HERBERT:size_array_split:invalid_argument: "
                "The full array cannot be resolved into a stack of arrays with the given stack size"
            )
    else:
        # sz_stack is empty i.e. was originally [1, 1] i.e. a scalar
        sz_root = list(sz_full_t)

    # Add trailing singletons to the root array to give MATLAB size of at least two
    if len(sz_root) < 2:
        sz_root = sz_root + [1] * (2 - len(sz_root))

    return tuple(sz_root)


class DataOpArray(np.ndarray):
    """2D+ array container of ``data_op_interface`` objects supporting MATLAB-style array operations.

    A 1D sequence ``[w1, w2, w3]`` is stored with shape ``(1, 3)`` (row vector),
    while ``[[w1], [w2], [w3]]`` has shape ``(3, 1)`` (column vector), matching
    MATLAB's object array semantics.
    """

    def __new__(cls, input_array: Any) -> "DataOpArray":
        if isinstance(input_array, DataOpArray):
            return input_array
        if isinstance(input_array, np.ndarray) and input_array.dtype == object:
            arr = input_array
        elif isinstance(input_array, (list, tuple)):
            if len(input_array) > 0 and all(
                isinstance(el, (list, tuple, np.ndarray)) and not isinstance(el, data_op_interface)
                for el in input_array
            ):
                rows = len(input_array)
                cols = len(input_array[0])
                arr = np.empty((rows, cols), dtype=object)
                for r in range(rows):
                    for c in range(cols):
                        arr[r, c] = input_array[r][c]
            else:
                arr = np.empty((1, len(input_array)), dtype=object)
                for i, item in enumerate(input_array):
                    arr[0, i] = item
        else:
            arr = np.empty((1, 1), dtype=object)
            arr[0, 0] = input_array

        if arr.ndim == 1:
            arr = arr.reshape((1, arr.shape[0]))
        return arr.view(cls)

    @classmethod
    def from_list(
        cls, items: Any, shape: Tuple[int, ...] | None = None
    ) -> "DataOpArray":
        """Construct a ``DataOpArray`` from a flat sequence and optional shape."""
        seq = list(items)
        arr = np.empty(len(seq), dtype=object)
        for i, item in enumerate(seq):
            arr[i] = item
        if shape is not None:
            arr = arr.reshape(shape, order="F")
        else:
            arr = arr.reshape((1, len(seq)))
        return arr.view(cls)

    def __len__(self) -> int:
        if self.ndim == 2 and (self.shape[0] == 1 or self.shape[1] == 1):
            return self.size
        return super().__len__()

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            if self.ndim == 2 and (self.shape[0] == 1 or self.shape[1] == 1):
                return self.ravel(order="F")[idx]
        return super().__getitem__(idx)

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        return _dispatch_ufunc(ufunc, *inputs)

    def plus(self, other: Any) -> Any:
        return plus(self, other)

    def minus(self, other: Any) -> Any:
        return minus(self, other)

    def mtimes(self, other: Any) -> Any:
        return mtimes(self, other)

    def mrdivide(self, other: Any) -> Any:
        return mrdivide(self, other)

    def mldivide(self, other: Any) -> Any:
        return mldivide(self, other)

    def mpower(self, other: Any) -> Any:
        return mpower(self, other)

    def __add__(self, other: Any) -> Any:
        return plus(self, other)

    def __radd__(self, other: Any) -> Any:
        return plus(other, self)

    def __sub__(self, other: Any) -> Any:
        return minus(self, other)

    def __rsub__(self, other: Any) -> Any:
        return minus(other, self)

    def __mul__(self, other: Any) -> Any:
        return mtimes(self, other)

    def __rmul__(self, other: Any) -> Any:
        return mtimes(other, self)

    def __truediv__(self, other: Any) -> Any:
        return mrdivide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return mrdivide(other, self)

    def __pow__(self, other: Any) -> Any:
        return mpower(self, other)

    def __rpow__(self, other: Any) -> Any:
        return mpower(other, self)

    def __neg__(self) -> Any:
        return uminus(self)

    def __pos__(self) -> Any:
        return uplus(self)

    def copy(self) -> "DataOpArray":
        return copy.deepcopy(self)

    def equal_to_tol(
        self,
        other: Any,
        *args: Any,
        return_msg: bool = False,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(other, (DataOpArray, np.ndarray, list, tuple)):
            return (False, "Not an array") if return_msg else False
        other_arr = np.asarray(other, dtype=object)
        if self.size != other_arr.size:
            return (False, "Array sizes differ") if return_msg else False
        for a, b in zip(self.ravel(order="F"), other_arr.ravel(order="F")):
            res = a.equal_to_tol(b, *args, return_msg=True, **kwargs)
            ok, mess = res if isinstance(res, tuple) else (bool(res), "")
            if not ok:
                return (False, mess) if return_msg else False
        return (True, "") if return_msg else True


class data_op_interface(ABC):
    """DATA_OP_INTERFACE defines the operations which are available on
    numeric, sigvar, IX_data, DnD and SQW objects.

    The operations are implemented using unary and binary operation
    managers.

    Attributes
    ----------
    NUM_DIMS : int
        Number of object dimensions (dependent property calling ``get_NUM_DIMS``).
    img_size : Tuple[int, ...]
        Size of ND image stored in object or image object itself (dependent property calling ``get_img_size``).
    """

    # List of classes which have binary operations redefined.
    base_classes: Tuple[str, ...] = (
        "sqw",
        "PixelDataBase",
        "DnDBase",
        "IX_dataset",
        "sigvar",
        "numeric",
    )

    # Base priorities of the base_classes used to determine the type of
    # result of binary operations. The actual priority is calculated
    # from the base priority by adding points for presence of pixels
    # (+100) and image (+10) giving final score (priority) for an object.
    # If binary operation is performed between operands with different
    # priorities, the result has the type of the higher priority object.
    bc_priority: Tuple[int, ...] = (5, 4, 3, 2, 1, 0)

    # ------------------------------------------------------------------
    # Dependent properties
    # ------------------------------------------------------------------
    @property
    def NUM_DIMS(self) -> int:
        """Number of object dimensions."""
        return self.get_NUM_DIMS()

    @property
    def img_size(self) -> Tuple[int, ...]:
        """Size of ND image stored in object or image object itself."""
        return self.get_img_size()

    # ------------------------------------------------------------------
    # Abstract protected methods
    # ------------------------------------------------------------------
    @abstractmethod
    def get_NUM_DIMS(self) -> int:
        """Return number of object dimensions."""

    @abstractmethod
    def get_img_size(self) -> Tuple[int, ...]:
        """Return size of the ND image object represents or object itself."""

    # ------------------------------------------------------------------
    # Static priority & operation order methods
    # ------------------------------------------------------------------
    @staticmethod
    def _matches_base_class(obj: Any, base_name: str) -> bool:
        if base_name == "numeric":
            return _is_numeric(obj)
        mro_names = {cls.__name__ for cls in type(obj).__mro__}
        if base_name == "sqw":
            return bool(mro_names.intersection({"sqw", "SQW"}))
        if base_name == "PixelDataBase":
            return "PixelDataBase" in mro_names
        if base_name == "DnDBase":
            return bool(
                mro_names.intersection(
                    {"DnDBase", "d0d", "d1d", "d2d", "d3d", "d4d", "D0d", "D1d", "D2d", "D3d", "D4d"}
                )
            )
        if base_name == "IX_dataset":
            return bool(
                mro_names.intersection(
                    {
                        "IX_dataset",
                        "IXDataset",
                        "IX_data_1d",
                        "IX_data_2d",
                        "IX_data_3d",
                        "IX_dataset_1d",
                        "IX_dataset_2d",
                        "IX_dataset_3d",
                        "IXDataset1D",
                        "IXDataset2D",
                        "IXDataset3D",
                    }
                )
            )
        if base_name == "sigvar":
            return bool(mro_names.intersection({"sigvar", "SigVar"}))
        return base_name in mro_names

    @staticmethod
    def get_priority(obj: Any) -> Tuple[int, Tuple[int, ...], bool, bool]:
        """Return a class priority which defines operations order.

        Parameters
        ----------
        obj : Any
            The object to check.

        Returns
        -------
        priority : int
            The number which defines the priority of this object operation
            within the list of all operations.
        sv_size : Tuple[int, ...]
            sigvar size of the object (size of its image).
        has_pix : bool
            True if object contains pixels.
        has_img : bool
            True if the object is not a scalar (has image).
        """
        matched_priorities = [
            prio
            for cls_name, prio in zip(
                data_op_interface.base_classes, data_op_interface.bc_priority
            )
            if data_op_interface._matches_base_class(obj, cls_name)
        ]
        if not matched_priorities:
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                f"Class {type(obj).__name__} does not have Horace binary operation defined for it."
            )

        priority = max(matched_priorities)
        sv_size = sigvar_size(obj)
        if sv_size != (1, 1):
            priority += 10
            has_img = True
        else:
            has_img = False

        is_sqw = data_op_interface._matches_base_class(obj, "sqw")
        is_pix = data_op_interface._matches_base_class(obj, "PixelDataBase")
        has_pixels_method = getattr(obj, "has_pixels", None)
        if (is_sqw and callable(has_pixels_method) and has_pixels_method()) or is_pix:
            has_pix = True
            priority += 100
        else:
            has_pix = False

        return priority, sv_size, has_pix, has_img

    @staticmethod
    def get_operation_kind(
        op1_has_pix: bool, op1_has_img: bool, op2_has_pix: bool, op2_has_img: bool
    ) -> int:
        """Determine what kind of operation should be applied between two operands given operand features.

        There are 4 types of operations:
        - 1 -> object<->scalar
        - 2 -> object<->image
        - 3 -> object<->object
        - 0 -> operations can be performed by converting to sigvar

        Parameters
        ----------
        op1_has_pix : bool
            True if operand 1 has pixels.
        op1_has_img : bool
            True if operand 1 has image.
        op2_has_pix : bool
            True if operand 2 has pixels.
        op2_has_img : bool
            True if operand 2 has image.

        Returns
        -------
        int
            Number from 0 to 3 describing type of operation performed over pixels.
        """
        if op1_has_pix:
            if op2_has_pix:
                return 3
            elif op2_has_img:
                return 2
            else:
                return 1
        elif op1_has_img:
            if op2_has_pix:
                raise RuntimeError(
                    "HERBERT:data_op_interface:runtime_error: "
                    "Invalid order of operations identified. Obj1 does not have pixels and obj2 has them"
                )
            return 0
        else:
            return 0

    @staticmethod
    def get_operation_order(
        obj1: Any, obj2: Any, op_name: str = "an_operation"
    ) -> Tuple[bool, int]:
        """Helper function to establish order of operands in binary operations.

        Parameters
        ----------
        obj1 : Any
            The object provided as first term of the operation.
        obj2 : Any
            The object provided as second term of the operation.
        op_name : str, optional
            String containing the operation name, used in error reporting.

        Returns
        -------
        flip : bool
            True if class 2 is superior over class 1 and binary operations
            should return class 2 as the result of operation instead of class 1.
        page_op_kind : int
            Depending on operands, 4 types of page op are defined (0 to 3).
            See ``get_operation_kind`` for details.
        """
        priority1, sv_size1, has_pix1, has_img1 = data_op_interface.get_priority(obj1)
        priority2, sv_size2, has_pix2, has_img2 = data_op_interface.get_priority(obj2)

        if not (sv_size1 == sv_size2 or sv_size1 == (1, 1) or sv_size2 == (1, 1)):
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                f"Image size {sv_size1} of operand 1 is inconsistent image size {sv_size2} "
                f"of operand 2 in operation {op_name}"
            )

        if priority2 > priority1:
            flip = True
            op1_has_pix = has_pix2
            op2_has_pix = has_pix1
            op1_has_img = has_img2
            op2_has_img = has_img1
        else:
            flip = False
            op1_has_pix = has_pix1
            op2_has_pix = has_pix2
            op1_has_img = has_img1
            op2_has_img = has_img2

        page_op_kind = data_op_interface.get_operation_kind(
            op1_has_pix, op1_has_img, op2_has_pix, op2_has_img
        )
        return flip, page_op_kind

    # ------------------------------------------------------------------
    # Operation managers
    # ------------------------------------------------------------------
    @staticmethod
    def binary_op_manager(
        w1: Any,
        w2: Any,
        binary_op: Callable[..., Any],
        ignore_cell_sorting: bool = False,
    ) -> Any:
        """Implement a binary operation for objects with a signal and a variance array.

        >> w = binary_op_manager(w1, w2, binary_op)
        >> w = binary_op_manager(w1, w2, binary_op, ignore_cell_sorting)

        All binary operations on MATLAB double arrays are permitted
        (+, -, *, /, \\, ^) and are applied element by element to the signal and
        variance arrays.

        Parameters
        ----------
        w1, w2 : Any
            Objects on which the binary operation is to be performed.
            One of these can be a numeric scalar or array, in which case the
            variance array is taken to be zero.

            If w1, w2 are scalar objects with the same signal array sizes:
            - The operation is performed element-by-element.

            If one of w1 or w2 is a numeric array (and the other is a scalar object):
            - If a scalar, apply to each element of the object signal.
            - If it is an array of the same size as the object signal array,
              apply the operation element by element.

            If one or both of w1 and w2 are arrays of objects:
            - If objects have same array sizes, the binary operation is applied
              object element-by-object element.
            - If one of the objects is scalar (i.e. only one object), then it is
              applied by the binary operation to each object in the other array.

            If one of w1, w2 is an array of objects and the other is a numeric array:
            - If the numeric is a scalar, it is applied to every object in the array.
            - If the numeric is an array with the same size as the object array,
              then each element is applied as a scalar to the corresponding object
              in the object array.
            - If the numeric is an array with larger size than the object array,
              then the array is resolved into a stack of arrays, where the stack
              has the same size as the object array, and each array in the stack
              is applied to the corresponding object in the object array.
        binary_op : Callable
            Function handle to a binary operation.
        ignore_cell_sorting : bool, optional
            By default False. When True, skips pixel sorting within bins during
            pixel binary operations.

        Returns
        -------
        Any
            Output object or ``DataOpArray`` of objects.
        """
        w1_is_obj = _is_object_or_object_array(w1)
        w2_is_obj = _is_object_or_object_array(w2)
        w1_is_num = _is_numeric(w1)
        w2_is_num = _is_numeric(w2)

        if w1_is_obj and not isinstance(w1, data_op_interface):
            w1 = DataOpArray(w1)
        if w2_is_obj and not isinstance(w2, data_op_interface):
            w2 = DataOpArray(w2)

        w1_size_tuple = (1, 1) if isinstance(w1, data_op_interface) else matlab_size(w1)
        w2_size_tuple = (1, 1) if isinstance(w2, data_op_interface) else matlab_size(w2)

        if w1_is_obj:
            size_stack1 = w1_size_tuple
            size_root1 = (1, 1)
        elif w1_is_num:
            w1_arr = np.asarray(w1, dtype=float)
            if w1_arr.size != 1:
                size_stack1 = w2_size_tuple
                size_root1 = size_array_split(w1_size_tuple, w2_size_tuple)
            else:
                size_stack1 = (1, 1)
                size_root1 = (1, 1)
        else:
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                "Invalid first argument to binary operation - "
                f"it must be supported object or a numeric value. It is {type(w1).__name__}"
            )

        if w2_is_obj:
            size_stack2 = w2_size_tuple
            size_root2 = (1, 1)
        elif w2_is_num:
            w2_arr = np.asarray(w2, dtype=float)
            if w2_arr.size != 1:
                size_stack2 = w1_size_tuple
                size_root2 = size_array_split(w2_size_tuple, w1_size_tuple)
            else:
                size_stack2 = (1, 1)
                size_root2 = (1, 1)
        else:
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                "Invalid second argument to binary operation - "
                f"it must be a supported object or a numeric value. It is: {type(w2).__name__}"
            )

        nroot1 = int(np.prod(size_root1))
        nroot2 = int(np.prod(size_root2))
        nobj1 = int(np.prod(size_stack1))
        nobj2 = int(np.prod(size_stack2))

        def _extract_item(w_in: Any, nroot: int, nobj: int, size_root: Tuple[int, ...], idx: int) -> Any:
            if isinstance(w_in, data_op_interface):
                return w_in
            if isinstance(w_in, DataOpArray):
                return w_in.ravel(order="F")[idx]
            arr = np.asarray(w_in, dtype=float)
            if arr.size == 1 and nroot == 1 and nobj == 1:
                return arr
            arr_2d = arr.reshape((nroot, nobj), order="F")
            col = arr_2d[:, idx]
            if size_root == (1, 1):
                return float(col[0])
            return col.reshape(size_root, order="F")

        if nobj1 == 1 and nobj2 == 1:
            elem1 = w1.ravel(order="F")[0] if isinstance(w1, DataOpArray) else w1
            elem2 = w2.ravel(order="F")[0] if isinstance(w2, DataOpArray) else w2
            return _invoke_binary_op_manager_single(
                elem1, elem2, binary_op, ignore_cell_sorting
            )

        elif size_stack1 == size_stack2:
            results = []
            for i in range(nobj1):
                obj1 = _extract_item(w1, nroot1, nobj1, size_root1, i)
                obj2 = _extract_item(w2, nroot2, nobj2, size_root2, i)
                results.append(
                    _invoke_binary_op_manager_single(obj1, obj2, binary_op, ignore_cell_sorting)
                )
            out = np.empty(size_stack1, dtype=object)
            out_flat = out.ravel(order="F")
            for i, r in enumerate(results):
                out_flat[i] = r
            return out.reshape(size_stack1, order="F").view(DataOpArray)

        elif nobj1 == 1 and nobj2 > 1:
            elem1 = w1.ravel(order="F")[0] if isinstance(w1, DataOpArray) else w1
            results = []
            for i in range(nobj2):
                obj2 = _extract_item(w2, nroot2, nobj2, size_root2, i)
                results.append(
                    _invoke_binary_op_manager_single(
                        elem1, obj2, binary_op, ignore_cell_sorting
                    )
                )
            out = np.empty(size_stack2, dtype=object)
            out_flat = out.ravel(order="F")
            for i, r in enumerate(results):
                out_flat[i] = r
            return out.reshape(size_stack2, order="F").view(DataOpArray)

        elif nobj1 > 1 and nobj2 == 1:
            elem2 = w2.ravel(order="F")[0] if isinstance(w2, DataOpArray) else w2
            results = []
            for i in range(nobj1):
                obj1 = _extract_item(w1, nroot1, nobj1, size_root1, i)
                results.append(
                    _invoke_binary_op_manager_single(
                        obj1, elem2, binary_op, ignore_cell_sorting
                    )
                )
            out = np.empty(size_stack1, dtype=object)
            out_flat = out.ravel(order="F")
            for i, r in enumerate(results):
                out_flat[i] = r
            return out.reshape(size_stack1, order="F").view(DataOpArray)

        else:
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                "Array lengths are incompatible.\n"
                "Both arrays must have an equal number of elements or the "
                "number of elements in one of the arrays must be 1.\n"
                f"Arrays have number of elements '{nobj1}' & '{nobj2}'."
            )

    @staticmethod
    def binary_op_manager_single(
        w1: Any,
        w2: Any,
        binary_op: Callable[..., Any],
        ignore_cell_sorting: bool = False,
    ) -> Any:
        """Implement a binary operation between two scalar objects (or object and numeric array).

        Parameters
        ----------
        w1, w2 : Any
            Scalar objects or numeric array operands.
        binary_op : Callable
            Function handle to a binary operation.
        ignore_cell_sorting : bool, optional
            By default False.

        Returns
        -------
        Any
            Result of the binary operation.
        """
        from pyhorace.data.sigvar import sigvar

        op_name = getattr(binary_op, "__name__", str(binary_op))
        flip, page_op_kind = data_op_interface.get_operation_order(w1, w2, op_name)
        if flip:
            wout = w2.copy() if hasattr(w2, "copy") and callable(w2.copy) else copy.deepcopy(w2)
            operand = w1
        else:
            wout = w1.copy() if hasattr(w1, "copy") and callable(w1.copy) else copy.deepcopy(w1)
            operand = w2

        if page_op_kind == 0:
            sv1 = sigvar(w1)
            sv2 = sigvar(w2)
            result = binary_op(sv1, sv2)
            if hasattr(wout, "sigvar_set") and callable(wout.sigvar_set):
                return wout.sigvar_set(result)
            return result
        elif page_op_kind in (1, 2, 3):
            if (
                isinstance(wout, data_op_interface)
                and type(wout).binary_op_manager_single
                is not data_op_interface.binary_op_manager_single
            ):
                return wout.binary_op_manager_single(
                    w1, w2, binary_op, ignore_cell_sorting
                )
            if hasattr(wout, "_pix") and hasattr(wout, "has_pixels") and wout.has_pixels():
                npix_arr = getattr(getattr(wout, "data", wout), "npix", None)
                op_pix = (
                    operand._pix
                    if (page_op_kind == 3 and hasattr(operand, "_pix"))
                    else operand
                )
                wout._pix = wout._pix.do_binary_op(
                    op_pix,
                    binary_op,
                    flip=flip,
                    npix=npix_arr,
                )
            sv1 = sigvar(w1)
            sv2 = sigvar(w2)
            result = binary_op(sv1, sv2)
            if hasattr(wout, "sigvar_set") and callable(wout.sigvar_set):
                return wout.sigvar_set(result)
            return result
        else:
            raise ValueError(
                "HERBERT:data_op_interface:invalid_argument: "
                f"unsupported type of operation {page_op_kind} for operation {op_name} "
                f"between objects of class {type(w1).__name__} and class {type(w2).__name__}"
            )

    @staticmethod
    def unary_op_manager(w1: Any, unary_op: Callable[..., Any]) -> Any:
        """Implement a unary operation for an object or array of objects.

        >> w = unary_op_manager(w1, unary_op)

        Parameters
        ----------
        w1 : Any
            Input object or array of objects on which to apply the unary operator.
        unary_op : Callable
            Function handle to a unary operation on a ``sigvar`` object.

        Returns
        -------
        Any
            Output object or ``DataOpArray`` of objects.
        """
        from pyhorace.data.sigvar import sigvar

        if isinstance(w1, DataOpArray) or (
            isinstance(w1, (np.ndarray, list, tuple)) and _is_object_or_object_array(w1)
        ):
            arr = DataOpArray(w1)
            out = np.empty(arr.shape, dtype=object)
            out_flat = out.ravel(order="F")
            for i, elem in enumerate(arr.ravel(order="F")):
                if hasattr(elem, "do_unary_op") and callable(elem.do_unary_op):
                    out_flat[i] = elem.do_unary_op(unary_op)
                else:
                    res = unary_op(sigvar(elem))
                    elem_copy = elem.copy() if hasattr(elem, "copy") and callable(elem.copy) else copy.deepcopy(elem)
                    out_flat[i] = elem_copy.sigvar_set(res) if hasattr(elem_copy, "sigvar_set") else res
            return out.reshape(arr.shape, order="F").view(DataOpArray)

        if hasattr(w1, "do_unary_op") and callable(w1.do_unary_op):
            return w1.do_unary_op(unary_op)

        res = unary_op(sigvar(w1))
        w_copy = w1.copy() if hasattr(w1, "copy") and callable(w1.copy) else copy.deepcopy(w1)
        if hasattr(w_copy, "sigvar_set") and callable(w_copy.sigvar_set):
            return w_copy.sigvar_set(res)
        return res

    # ------------------------------------------------------------------
    # Unary operations (30 methods)
    # ------------------------------------------------------------------
    def acos(self) -> Any:
        """Implement acos(w1) for objects (`>> w = acos(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acos)

    def acosh(self) -> Any:
        """Implement acosh(w1) for objects (`>> w = acosh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acosh)

    def acot(self) -> Any:
        """Implement acot(w1) for objects (`>> w = acot(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acot)

    def acoth(self) -> Any:
        """Implement acoth(w1) for objects (`>> w = acoth(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acoth)

    def acsc(self) -> Any:
        """Implement acsc(w1) for objects (`>> w = acsc(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acsc)

    def acsch(self) -> Any:
        """Implement acsch(w1) for objects (`>> w = acsch(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.acsch)

    def asec(self) -> Any:
        """Implement asec(w1) for objects (`>> w = asec(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.asec)

    def asech(self) -> Any:
        """Implement asech(w1) for objects (`>> w = asech(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.asech)

    def asin(self) -> Any:
        """Implement asin(w1) for objects (`>> w = asin(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.asin)

    def asinh(self) -> Any:
        """Implement asinh(w1) for objects (`>> w = asinh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.asinh)

    def atan(self) -> Any:
        """Implement atan(w1) for objects (`>> w = atan(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.atan)

    def atanh(self) -> Any:
        """Implement atanh(w1) for objects (`>> w = atanh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.atanh)

    def cos(self) -> Any:
        """Implement cos(w1) for objects (`>> w = cos(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.cos)

    def cosh(self) -> Any:
        """Implement cosh(w1) for objects (`>> w = cosh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.cosh)

    def cot(self) -> Any:
        """Implement cot(w1) for objects (`>> w = cot(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.cot)

    def coth(self) -> Any:
        """Implement coth(w1) for objects (`>> w = coth(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.coth)

    def csc(self) -> Any:
        """Implement csc(w1) for objects (`>> w = csc(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.csc)

    def csch(self) -> Any:
        """Implement csch(w1) for objects (`>> w = csch(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.csch)

    def exp(self) -> Any:
        """Implement exp(w1) for objects (`>> w = exp(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.exp)

    def log(self) -> Any:
        """Implement log(w1) for objects (`>> w = log(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.log)

    def log10(self) -> Any:
        """Implement log10(w1) for objects (`>> w = log10(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.log10)

    def sec(self) -> Any:
        """Implement sec(w1) for objects (`>> w = sec(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.sec)

    def sech(self) -> Any:
        """Implement sech(w1) for objects (`>> w = sech(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.sech)

    def sin(self) -> Any:
        """Implement sin(w1) for objects (`>> w = sin(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.sin)

    def sinh(self) -> Any:
        """Implement sinh(w1) for objects (`>> w = sinh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.sinh)

    def sqrt(self) -> Any:
        """Implement sqrt(w1) for objects (`>> w = sqrt(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.sqrt)

    def tan(self) -> Any:
        """Implement tan(w1) for objects (`>> w = tan(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.tan)

    def tanh(self) -> Any:
        """Implement tanh(w1) for objects (`>> w = tanh(w1)`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.tanh)

    def uminus(self) -> Any:
        """Implement -w1 for objects (`>> w = -w1`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.uminus)

    def uplus(self) -> Any:
        """Implement +w1 for objects (`>> w = +w1`)."""
        from pyhorace.data.sigvar import sigvar
        return self.unary_op_manager(self, sigvar.uplus)

    # ------------------------------------------------------------------
    # Binary operations (6 methods)
    # ------------------------------------------------------------------
    def plus(self, w2: Any) -> Any:
        """Implement w1 + w2 for objects (`>> w = w1 + w2`)."""
        return self.binary_op_manager(self, w2, plus)

    def minus(self, w2: Any) -> Any:
        """Implement w1 - w2 for objects (`>> w = w1 - w2`)."""
        return self.binary_op_manager(self, w2, minus)

    def mtimes(self, w2: Any) -> Any:
        """Implement w1 * w2 for objects (`>> w = w1 * w2`)."""
        return self.binary_op_manager(self, w2, mtimes)

    def mrdivide(self, w2: Any) -> Any:
        """Implement w1 / w2 for objects (`>> w = w1 / w2`)."""
        return self.binary_op_manager(self, w2, mrdivide)

    def mldivide(self, w2: Any) -> Any:
        """Implement w1 \\ w2 for objects (`>> w = w1 \\ w2`)."""
        return self.binary_op_manager(self, w2, mldivide)

    def mpower(self, w2: Any) -> Any:
        """Implement w1 ^ w2 for objects (`>> w = w1 ^ w2`)."""
        return self.binary_op_manager(self, w2, mpower)

    # ------------------------------------------------------------------
    # Python operator overloads & NumPy ufunc dispatch
    # ------------------------------------------------------------------
    __array_priority__ = 1000.0

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method != "__call__":
            return NotImplemented
        return _dispatch_ufunc(ufunc, *inputs)

    def __add__(self, other: Any) -> Any:
        return plus(self, other)

    def __radd__(self, other: Any) -> Any:
        return plus(other, self)

    def __sub__(self, other: Any) -> Any:
        return minus(self, other)

    def __rsub__(self, other: Any) -> Any:
        return minus(other, self)

    def __mul__(self, other: Any) -> Any:
        return mtimes(self, other)

    def __rmul__(self, other: Any) -> Any:
        return mtimes(other, self)

    def __matmul__(self, other: Any) -> Any:
        return mtimes(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        return mtimes(other, self)

    def __truediv__(self, other: Any) -> Any:
        return mrdivide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return mrdivide(other, self)

    def __pow__(self, other: Any) -> Any:
        return mpower(self, other)

    def __rpow__(self, other: Any) -> Any:
        return mpower(other, self)

    def __neg__(self) -> Any:
        return self.uminus()

    def __pos__(self) -> Any:
        return self.uplus()


def _invoke_binary_op_manager_single(
    w1: Any, w2: Any, binary_op: Callable[..., Any], ignore_cell_sorting: bool = False
) -> Any:
    """Dispatch binary_op_manager_single to the leading object operand's class."""
    from pyhorace.data.sigvar import sigvar

    lead = w1 if isinstance(w1, data_op_interface) else (w2 if isinstance(w2, data_op_interface) else None)
    if isinstance(lead, sigvar):
        return sigvar.binary_op_manager_single(w1, w2, binary_op, ignore_cell_sorting)
    if lead is not None:
        return lead.binary_op_manager_single(w1, w2, binary_op, ignore_cell_sorting)
    return data_op_interface.binary_op_manager_single(w1, w2, binary_op, ignore_cell_sorting)


def _first_object(w1: Any, w2: Any) -> Any:
    """Return the first non-numeric object from (w1, w2) for method dispatch."""
    for w in (w1, w2):
        if isinstance(w, data_op_interface):
            return w
        if isinstance(w, DataOpArray) and w.size > 0:
            return w.ravel(order="F")[0]
        if isinstance(w, (np.ndarray, list, tuple)) and _is_object_or_object_array(w):
            return DataOpArray(w).ravel(order="F")[0]
    return None


# ----------------------------------------------------------------------
# Module-level binary and unary functions matching MATLAB function calls
# ----------------------------------------------------------------------
def plus(w1: Any, w2: Any) -> Any:
    """Implement w1 + w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _plus_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _plus_single)
    return data_op_interface.binary_op_manager(w1, w2, plus)


def minus(w1: Any, w2: Any) -> Any:
    """Implement w1 - w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _minus_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _minus_single)
    return data_op_interface.binary_op_manager(w1, w2, minus)


def mtimes(w1: Any, w2: Any) -> Any:
    """Implement w1 * w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _mtimes_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _mtimes_single)
    return data_op_interface.binary_op_manager(w1, w2, mtimes)


def mrdivide(w1: Any, w2: Any) -> Any:
    """Implement w1 / w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _mrdivide_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _mrdivide_single)
    return data_op_interface.binary_op_manager(w1, w2, mrdivide)


def mldivide(w1: Any, w2: Any) -> Any:
    """Implement w1 \\ w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _mldivide_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _mldivide_single)
    return data_op_interface.binary_op_manager(w1, w2, mldivide)


def mpower(w1: Any, w2: Any) -> Any:
    """Implement w1 ^ w2 for objects and arrays of objects."""
    from pyhorace.data.sigvar import _mpower_single, sigvar

    lead = _first_object(w1, w2)
    if isinstance(lead, sigvar):
        return data_op_interface.binary_op_manager(w1, w2, _mpower_single)
    return data_op_interface.binary_op_manager(w1, w2, mpower)


def acos(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acos)


def acosh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acosh)


def acot(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acot)


def acoth(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acoth)


def acsc(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acsc)


def acsch(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.acsch)


def asec(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.asec)


def asech(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.asech)


def asin(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.asin)


def asinh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.asinh)


def atan(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.atan)


def atanh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.atanh)


def cos(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.cos)


def cosh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.cosh)


def cot(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.cot)


def coth(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.coth)


def csc(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.csc)


def csch(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.csch)


def exp(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.exp)


def log(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.log)


def log10(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.log10)


def sec(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.sec)


def sech(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.sech)


def sin(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.sin)


def sinh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.sinh)


def sqrt(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.sqrt)


def tan(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.tan)


def tanh(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.tanh)


def uminus(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.uminus)


def uplus(w1: Any) -> Any:
    from pyhorace.data.sigvar import sigvar
    return data_op_interface.unary_op_manager(w1, sigvar.uplus)


def _dispatch_ufunc(ufunc: Any, *inputs: Any) -> Any:
    if len(inputs) == 2:
        w1, w2 = inputs
        if ufunc is np.add:
            return plus(w1, w2)
        if ufunc is np.subtract:
            return minus(w1, w2)
        if ufunc is np.multiply or ufunc is np.matmul:
            return mtimes(w1, w2)
        if ufunc is np.true_divide:
            return mrdivide(w1, w2)
        if ufunc is np.power:
            return mpower(w1, w2)
    elif len(inputs) == 1:
        w1 = inputs[0]
        unary_map = {
            np.negative: uminus,
            np.positive: uplus,
            np.arccos: acos,
            np.arccosh: acosh,
            np.arcsin: asin,
            np.arcsinh: asinh,
            np.arctan: atan,
            np.arctanh: atanh,
            np.cos: cos,
            np.cosh: cosh,
            np.exp: exp,
            np.log: log,
            np.log10: log10,
            np.sin: sin,
            np.sinh: sinh,
            np.sqrt: sqrt,
            np.tan: tan,
            np.tanh: tanh,
        }
        if ufunc in unary_map:
            return unary_map[ufunc](w1)
    return NotImplemented


# PascalCase alias for backward compatibility
DataOpInterface = data_op_interface
binary_op_manager = data_op_interface.binary_op_manager
binary_op_manager_single = data_op_interface.binary_op_manager_single
unary_op_manager = data_op_interface.unary_op_manager
