"""Shared test helpers for instrument classes unit tests.

Ported from:
- _test/common_functions/hashable_obj_tester.m
- _test/common_functions/check_matfile_IO.m
- _test/test_instrument_classes/validate_det_rand.m
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import scipy.io as sio

from pyhorace.core.hashable import HashableArray, build_hash, hashable
from pyhorace.core.serializable import Serializable, compare_values
from pyhorace.data.instr_classes.pdf_table import sampling_statistics

MATLAB_TEST_DIR = (
    Path(__file__).resolve().parents[4] / "_test" / "test_instrument_classes"
)


def hashable_obj_tester(
    hobj: hashable, names: Optional[Sequence[str]] = None
) -> str:
    """
    Test generic hashable object behaviour: building hash, verifying cache,
    and checking that setting any hashable/dependent property invalidates the hash.
    Ported from `_test/common_functions/hashable_obj_tester.m`.
    """
    hobj, hash_val, _ = build_hash(hobj)
    assert isinstance(hash_val, str) and len(hash_val) == 32
    assert hobj.hash_defined
    assert hobj.hash_value == hash_val

    if names is None:
        flds = list(hobj.hashable_fields())
    else:
        flds = list(names)

    values = [getattr(hobj, f) for f in flds]
    for fld, val in zip(flds, values):
        hobj, hash1, _ = build_hash(hobj)
        assert hobj.hash_defined
        assert hash1 == hash_val

        setattr(hobj, fld, val)
        assert not hobj.hash_defined, f"Setting '{fld}' did not invalidate cached hash"
        assert hobj.hash_value is None

    return hash_val


def check_matfile_IO(
    ver: int, save_variables: bool, *objs: Any
) -> Any:
    """
    Emulate saving and loading Serializable / hashable objects or object arrays
    and verify their `class_version() == ver`.
    Ported from `_test/common_functions/check_matfile_IO.m`.
    """
    results = []
    for obj in objs:
        if isinstance(obj, (list, tuple, HashableArray)):
            assert len(obj) > 0
            first = obj[0]
            if isinstance(first, hashable):
                S = hashable.to_struct_array(obj)
            else:
                S = Serializable.to_struct_array(obj)
            assert S["version"] == ver
            restored = type(first).from_struct(S)
        elif isinstance(obj, Serializable):
            S = obj.to_struct()
            assert S["version"] == ver
            restored = type(obj).from_struct(S)
        else:
            raise TypeError(f"Unsupported object type for check_matfile_IO: {type(obj)}")
        results.append(restored)

    if len(results) == 1:
        return results[0]
    return tuple(results)


def validate_det_rand(
    det: Any,
    nsamples: int,
    ntol: float,
    npath: Union[Sequence[float], np.ndarray],
    wvec: float,
) -> Tuple[bool, str]:
    """
    Test statistics of random sampling of points of absorption in a detector
    against deterministic mean and covariance calculations.
    Ported from `_test/test_instrument_classes/validate_det_rand.m`.
    """
    npath_arr = np.asarray(npath, dtype=float).ravel()
    npath_arr = npath_arr / np.linalg.norm(npath_arr)
    X = det.rand(np.ones((nsamples, 1), dtype=int), npath_arr, wvec)

    mean_val, sig_mean, sig_val, sig_sig, cov_val, sig_cov, _, _ = sampling_statistics(
        X.T
    )

    mean_pos = np.asarray(det.mean(npath_arr, wvec), dtype=float).ravel()
    cov_pos = np.asarray(det.covariance(npath_arr, wvec), dtype=float)
    sig_pos = np.sqrt(np.diag(cov_pos))

    ok = True
    mess = ""
    if not np.all(np.abs(mean_val - mean_pos) < ntol * sig_mean):
        ok = False
        mess += "mean point of absorption from random sampling is outside tolerance\n"
    if not np.all(np.abs(sig_val - sig_pos) < ntol * sig_sig):
        ok = False
        mess += "std point of absorption from random sampling is outside tolerance\n"
    if not np.all(np.abs(cov_val - cov_pos) < ntol * sig_cov):
        ok = False
        mess += "covariance of absorption from random sampling is outside tolerance\n"

    return ok, mess


def _convert_mat_value(val: Any) -> Any:
    """Recursively convert scipy.io.loadmat objects into Python dicts/lists/scalars/ndarrays."""
    if isinstance(val, sio.matlab.mat_struct):
        d: Dict[str, Any] = {}
        for fn in val._fieldnames:
            d[fn] = _convert_mat_value(getattr(val, fn))
        return d
    if isinstance(val, sio.matlab.MatlabObject):
        # Old MATLAB class object wraps a structured ndarray
        arr = val.view(np.ndarray)
        res = _convert_mat_value(arr)
        return res
    if isinstance(val, np.void) and val.dtype.names:
        return {name: _convert_mat_value(val[name]) for name in val.dtype.names}
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            if val.size == 1:
                elem = val.flat[0]
                return {name: _convert_mat_value(elem[name]) for name in val.dtype.names}
            return [
                {name: _convert_mat_value(elem[name]) for name in val.dtype.names}
                for elem in val.flat
            ]
        if val.dtype.kind in ("U", "S"):
            if val.size == 0:
                return ""
            if val.size == 1:
                return str(val.flat[0])
            return "".join(str(x) for x in val.flat)
        if val.dtype == object:
            if val.size == 0:
                return []
            if val.size == 1:
                inner = _convert_mat_value(val.flat[0])
                return inner
            items = [_convert_mat_value(x) for x in val.flat]
            return items
        if val.size == 1 and np.issubdtype(val.dtype, np.number):
            v = val.flat[0]
            if np.issubdtype(val.dtype, np.integer):
                return int(v)
            return float(v)
        if np.issubdtype(val.dtype, np.bool_) and val.size == 1:
            return bool(val.flat[0])
        return val
    return val


def load_mat_reference(filename: str) -> Dict[str, Any]:
    """
    Load a `.mat` file from `_test/test_instrument_classes/` and convert its variables
    to Python dictionaries/primitives suitable for `Serializable.loadobj` or comparison.
    """
    path = MATLAB_TEST_DIR / filename
    raw = sio.loadmat(str(path), squeeze_me=False, struct_as_record=False)
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k.startswith("__"):
            continue
        out[k] = _convert_mat_value(v)
    return out


def restore_from_mat_dict(cls: Type[Serializable], mat_val: Any) -> Any:
    """
    Restore a `Serializable` instance or `HashableArray` from a converted `.mat` variable.
    Handles v2 `to_struct` dicts (`serial_name`, `version`, `array_dat`), v1 structs
    (`class_version_`), and v0 bare struct lists/dicts.
    """
    if isinstance(mat_val, dict):
        if "array_dat" in mat_val and isinstance(mat_val["array_dat"], dict):
            # Single-element array_dat unwrapped by _convert_mat_value
            mat_val = dict(mat_val)
            mat_val["array_dat"] = [mat_val["array_dat"]]
        return cls.loadobj(mat_val)
    elif isinstance(mat_val, list):
        return cls.loadobj(mat_val)
    return mat_val


__all__ = [
    "MATLAB_TEST_DIR",
    "check_matfile_IO",
    "hashable_obj_tester",
    "load_mat_reference",
    "restore_from_mat_dict",
    "sampling_statistics",
    "validate_det_rand",
]

