"""IX_mosaic class definition and mosaic helper functions.

Ported from Herbert MATLAB @IX_mosaic class and scientific utilities:
- herbert_core/classes/instrument_classes/instrument/@IX_mosaic/IX_mosaic.m
- herbert_core/classes/instrument_classes/instrument/@IX_mosaic/rand_rot_vect.m
- herbert_core/classes/instrument_classes/instrument/@IX_mosaic/private/convert_old_struct_.m
- herbert_core/utilities/scientific/check_mosaic_matrix.m
- herbert_core/utilities/scientific/rand_mosaic_gaussian.m
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import PrivateAttr

from pyhorace.core.hashable import hashable


def _is_numeric_array_or_scalar(val: Any) -> bool:
    if isinstance(val, (bool, np.bool_, str, bytes)):
        return False
    if isinstance(val, (int, float, np.number)):
        return not (np.isnan(val) or np.isinf(val))
    if isinstance(val, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(val)
        except Exception:
            return False
        if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.number):
            return False
        if arr.size > 0 and (np.any(np.isnan(arr)) or np.any(np.isinf(arr))):
            return False
        return True
    return False


def check_mosaic_matrix(
    val: Any,
) -> Tuple[
    Optional[Union[float, np.ndarray]],
    Optional[Union[float, np.ndarray]],
    Optional[np.ndarray],
]:
    """
    Check if the input is a valid mosaic parameter / matrix and return its
    eigensystem representation.

    Ported from `herbert_core/utilities/scientific/check_mosaic_matrix.m`.

    Parameters
    ----------
    val : float or array_like
        Scalar FWHH (>= 0), 3-vector of principal FWHH values (all >= 0),
        or 3x3 symmetric positive-definite matrix of FWHH^2 values.

    Returns
    -------
    val_out : float, np.ndarray, or None
        Validated mosaic parameter (float scalar, (3,) vector, or (3, 3) symmetric matrix),
        or None if invalid.
    S : float, np.ndarray, or None
        Principal FWHH values (scalar or (3,) vector), or None if invalid.
    V : np.ndarray or None
        3x3 orthonormal eigenvector matrix, or None if invalid.
    """
    if not _is_numeric_array_or_scalar(val):
        return None, None, None

    arr = np.asarray(val, dtype=float)
    if arr.size == 0:
        return None, None, None

    if arr.size == 1:
        scalar_val = float(arr.flat[0])
        if scalar_val >= 0.0:
            return scalar_val, scalar_val, np.eye(3, dtype=float)
        return None, None, None

    if arr.ndim == 1 or (arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1)):
        vec = arr.ravel()
        if vec.size == 3 and np.all(vec >= 0.0):
            return vec.copy(), vec.copy(), np.eye(3, dtype=float)
        return None, None, None

    if arr.shape == (3, 3):
        max_diag = float(np.max(np.abs(np.diag(arr))))
        max_all = float(np.max(np.abs(arr)))
        if max_diag <= 0.0 or not np.isclose(max_all / max_diag, 1.0, rtol=1e-12, atol=1e-12):
            return None, None, None
        if np.max(np.abs(arr - arr.T)) / max_all > 1e-12:
            return None, None, None
        val_sym = 0.5 * (arr + arr.T)
        eigvals, V = np.linalg.eigh(val_sym)
        if np.all(eigvals >= 0.0):
            S = np.sqrt(eigvals)
            return val_sym, S, V
        return None, None, None

    return None, None, None


def rand_mosaic_gaussian(
    sz: Any,
    eta: Any,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate random rotation vectors with a Gaussian mosaic distribution.

    Ported from `herbert_core/utilities/scientific/rand_mosaic_gaussian.m`.

    Parameters
    ----------
    sz : int or Sequence[int]
        Size of the random sample array. If scalar `n`, treated as `(n, n)`.
        A leading singleton dimension `sz[0] == 1` is suppressed in the output
        shape `(3, *sz[1:])`.
    eta : float or np.ndarray
        Mosaic spread FWHH (deg): scalar, 3-vector, or 3x3 covariance matrix of FWHH^2.
    rng : np.random.Generator, optional
        Optional NumPy random generator.

    Returns
    -------
    R : np.ndarray
        Random rotation vectors (rad) with leading dimension 3.
    """
    fwhh_to_sig = float(np.sqrt(np.log(256.0)))

    sz_arr = np.asarray(sz, dtype=int).ravel()
    if sz_arr.size == 0:
        sz_tuple: Tuple[int, ...] = (1, 1)
    elif sz_arr.size == 1:
        n = int(sz_arr[0])
        sz_tuple = (n, n)
    else:
        sz_tuple = tuple(int(x) for x in sz_arr)

    if sz_tuple[0] == 1:
        rem = list(sz_tuple[1:])
    else:
        rem = list(sz_tuple)
    while len(rem) > 0 and rem[-1] == 1:
        rem.pop()
    sz_r: Tuple[int, ...] = (3, *rem)

    eta_val, sig, V = check_mosaic_matrix(eta)
    if eta_val is None or sig is None or V is None:
        raise ValueError(
            "HERBERT:rand_mosaic_gaussian:invalid_argument: "
            "Check form of mosaic spread definitions"
        )

    sig_arr = np.asarray(sig, dtype=float) / fwhh_to_sig

    if np.all(sig_arr == 0.0):
        return np.zeros(sz_r, dtype=float)

    if sig_arr.size == 1:
        s_scalar = float(sig_arr.flat[0])
        if rng is not None:
            draws = rng.standard_normal(sz_r)
        else:
            draws = np.random.randn(*sz_r)
        return s_scalar * draws

    n_pts = int(np.prod(rem)) if len(rem) > 0 else 1
    if rng is not None:
        draws_2d = rng.standard_normal((3, n_pts))
    else:
        draws_2d = np.random.randn(3, n_pts)
    R_2d = sig_arr.reshape(3, 1) * draws_2d
    if not np.array_equal(V, np.eye(3)):
        R_2d = V @ R_2d
    return R_2d.reshape(sz_r, order="F")


def _resolve_mosaic_pdf(func_or_name: Any) -> Callable[..., Any]:
    """Resolve a function handle or function name string to a callable."""
    if callable(func_or_name):
        return func_or_name
    if isinstance(func_or_name, str):
        clean_name = func_or_name.strip().lstrip("@")
        if not clean_name:
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "mosaic distribution function name must be non-empty character string"
            )
        if clean_name == "rand_mosaic_gaussian":
            return rand_mosaic_gaussian

        def _named_pdf(*args: Any, **kwargs: Any) -> Any:
            return rand_mosaic_gaussian(*args, **kwargs)

        _named_pdf.__name__ = clean_name
        return _named_pdf
    raise ValueError(
        "HERBERT:IX_mosaic:invalid_argument: "
        "mosaic distribution must be a function handle or non-empty function name"
    )


class IX_mosaic(hashable):
    """
    Mosaic crystal spread description.
    Ported from Herbert MATLAB `@IX_mosaic`.
    """

    _xaxis: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=float)
    )
    _yaxis: np.ndarray = PrivateAttr(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=float)
    )
    _mosaic_pdf: Any = PrivateAttr(default=None)
    _parameters: Union[float, np.ndarray] = PrivateAttr(default=0.0)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._xaxis = np.array([1.0, 0.0, 0.0], dtype=float)
        self._yaxis = np.array([0.0, 1.0, 0.0], dtype=float)
        self._mosaic_pdf = rand_mosaic_gaussian
        self._parameters = 0.0
        self._do_check_combo_arg = True

        if len(args) == 0 and len(kwargs) == 0:
            return

        if len(args) == 1 and not kwargs:
            arg0 = args[0]
            if isinstance(arg0, dict):
                loaded = self.loadobj(arg0)
                if isinstance(loaded, IX_mosaic):
                    self._copy_state_from(loaded)
                return
            if isinstance(arg0, IX_mosaic):
                self._copy_state_from(arg0)
                return
            self.parameters = arg0
            return

        if (
            len(args) == 2
            and not kwargs
            and (callable(args[0]) or (isinstance(args[0], str) and args[0] not in self.saveable_fields()))
        ):
            self._do_check_combo_arg = False
            try:
                self.mosaic_pdf = args[0]
                self.parameters = args[1]
            finally:
                self._do_check_combo_arg = True
            self.check_combo_arg()
            return

        if (
            len(args) >= 3
            and _is_numeric_array_or_scalar(args[2])
            and not callable(args[2])
            and not isinstance(args[2], str)
        ):
            flds: Sequence[str] = ("xaxis", "yaxis", "parameters")
        elif len(args) >= 3 and callable(args[2]):
            flds = ("xaxis", "yaxis", "mosaic_pdf", "parameters")
        else:
            flds = self.saveable_fields()

        _, remains = self.set_positional_and_key_val_arguments(
            flds, True, *args, **kwargs
        )
        if remains:
            raise ValueError(
                f"HERBERT:IX_mosaic:invalid_argument: Unrecognized extra parameters provided "
                f"as input to IX_mosaic constructor: {remains}"
            )

    def _copy_state_from(self, other: "IX_mosaic") -> None:
        self._xaxis = other._xaxis.copy()
        self._yaxis = other._yaxis.copy()
        self._mosaic_pdf = other._mosaic_pdf
        if isinstance(other._parameters, np.ndarray):
            self._parameters = other._parameters.copy()
        else:
            self._parameters = other._parameters
        self._hash_value = other._hash_value

    def __copy__(self) -> "IX_mosaic":
        cloned = IX_mosaic()
        cloned._copy_state_from(self)
        cloned._do_check_combo_arg = self._do_check_combo_arg
        return cloned

    def _clone_for_hash(self) -> "IX_mosaic":
        return self.__copy__()

    # -------------------------------------------------------------------------
    # Dependent Properties
    # -------------------------------------------------------------------------

    @property
    def xaxis(self) -> np.ndarray:
        return self._xaxis

    @xaxis.setter
    def xaxis(self, val: Any) -> None:
        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "xaxis must be a non-zero numeric 3-vector"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.all(arr == 0.0):
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "xaxis must be a non-zero numeric 3-vector"
            )
        self._xaxis = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def yaxis(self) -> np.ndarray:
        return self._yaxis

    @yaxis.setter
    def yaxis(self, val: Any) -> None:
        if not _is_numeric_array_or_scalar(val):
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "yaxis must be a non-zero numeric 3-vector"
            )
        arr = np.asarray(val, dtype=float).ravel()
        if arr.size != 3 or np.all(arr == 0.0):
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "yaxis must be a non-zero numeric 3-vector"
            )
        self._yaxis = arr
        if self._do_check_combo_arg:
            self.check_combo_arg()
        else:
            self.clear_hash()

    @property
    def mosaic_pdf(self) -> Callable[..., Any]:
        return self._mosaic_pdf

    @mosaic_pdf.setter
    def mosaic_pdf(self, val: Any) -> None:
        if callable(val):
            self._mosaic_pdf = val
        elif isinstance(val, str) and val.strip():
            self._mosaic_pdf = _resolve_mosaic_pdf(val)
        else:
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "mosaic distribution must be a function handle"
            )
        self.clear_hash()

    @property
    def mosaic_pdf_string(self) -> str:
        return getattr(self._mosaic_pdf, "__name__", str(self._mosaic_pdf))

    @mosaic_pdf_string.setter
    def mosaic_pdf_string(self, val: Any) -> None:
        if callable(val):
            self._mosaic_pdf = val
            self.clear_hash()
            return
        if not isinstance(val, str) or not val.strip():
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "mosaic distribution function name must be non-empty character string"
            )
        self._mosaic_pdf = _resolve_mosaic_pdf(val)
        self.clear_hash()

    @property
    def parameters(self) -> Union[float, np.ndarray]:
        return self._parameters

    @parameters.setter
    def parameters(self, val: Any) -> None:
        par, _, _ = check_mosaic_matrix(val)
        if par is None:
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "Invalid mosaic parameter or matrix"
            )
        self._parameters = par
        self.clear_hash()

    # Private property aliases
    @property
    def xaxis_(self) -> np.ndarray:
        return self._xaxis

    @property
    def yaxis_(self) -> np.ndarray:
        return self._yaxis

    @property
    def mosaic_pdf_(self) -> Callable[..., Any]:
        return self._mosaic_pdf

    @property
    def parameters_(self) -> Union[float, np.ndarray]:
        return self._parameters

    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------

    def rand_rot_vect(
        self,
        dis_size: Any = (1, 1),
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Generate an array of random rotation vectors according to the mosaic PDF.
        Ported from `@IX_mosaic/rand_rot_vect.m`.
        """
        try:
            return self._mosaic_pdf(dis_size, self._parameters, rng=rng)
        except TypeError:
            return self._mosaic_pdf(dis_size, self._parameters)

    def mosaic_crystal(self) -> bool:
        """Return True if the mosaic spread is non-zero."""
        return not bool(np.all(np.asarray(self._parameters) == 0.0))

    # -------------------------------------------------------------------------
    # Serializable Interface
    # -------------------------------------------------------------------------

    def saveable_fields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return ("xaxis", "yaxis", "mosaic_pdf_string", "parameters")

    def saveableFields(self, mandatory: Optional[Any] = None) -> Sequence[str]:
        return self.saveable_fields(mandatory=mandatory)

    def class_version(self) -> int:
        return 2

    def check_combo_arg(self) -> "IX_mosaic":
        norm_x = float(np.linalg.norm(self._xaxis))
        norm_y = float(np.linalg.norm(self._yaxis))
        cross_norm = float(np.linalg.norm(np.cross(self._xaxis, self._yaxis)))
        if norm_x == 0.0 or norm_y == 0.0 or (cross_norm / (norm_x * norm_y)) < 1e-5:
            raise ValueError(
                "HERBERT:IX_mosaic:invalid_argument: "
                "Function properties are inconsistent, xaxis and yaxis are colinear or almost colinear"
            )
        self.clear_hash()
        return self

    def convert_old_struct(
        self, S: Dict[str, Any], ver: float
    ) -> Tuple[Dict[str, Any], "IX_mosaic"]:
        """
        Convert old structure versions (v0, v1) to current v2 format.
        Ported from `@IX_mosaic/private/convert_old_struct_.m`.
        """
        inputs = dict(S)
        if inputs.get("class_version_") == 1 or "xaxis_" in inputs:
            new_inputs: Dict[str, Any] = {}
            for k, v in inputs.items():
                new_k = k[:-1] if k.endswith("_") else k
                new_inputs[new_k] = v
            inputs = new_inputs
            inputs.pop("class_version", None)
            inputs.pop("class_version_", None)

        if "mosaic_pdf" in inputs:
            mpdf = inputs.pop("mosaic_pdf")
            if isinstance(mpdf, str) and mpdf.strip():
                inputs["mosaic_pdf_string"] = mpdf.strip().lstrip("@")
            elif callable(mpdf):
                inputs["mosaic_pdf_string"] = getattr(
                    mpdf, "__name__", "rand_mosaic_gaussian"
                )
            else:
                inputs["mosaic_pdf_string"] = "rand_mosaic_gaussian"

        return inputs, self


__all__ = ["IX_mosaic", "check_mosaic_matrix", "rand_mosaic_gaussian"]
