# MATLAB to Pytest Mapping Reference

This document provides a translation mapping between MATLAB test patterns (both `matlab.unittest.TestCase` and script-based tests) and Python `pytest` equivalents.

---

## 1. Test Assertions Mapping

| MATLAB Assertion | Python / Pytest Equivalent | Notes |
|---|---|---|
| `tc.assertEqual(a, b)` | `assert a == b` | Exact equality (integers, strings, booleans). |
| `tc.assertEqual(a, b, 'AbsTol', tol)` | `np.testing.assert_allclose(a, b, atol=tol, rtol=0)` | Floating-point comparison with absolute tolerance. For scalars: `assert a == pytest.approx(b, abs=tol)`. |
| `tc.assertEqual(a, b, 'RelTol', tol)` | `np.testing.assert_allclose(a, b, rtol=tol, atol=0)` | Floating-point comparison with relative tolerance. For scalars: `assert a == pytest.approx(b, rel=tol)`. |
| `tc.assertEqual(a, b, 'AbsTol', atol, 'RelTol', rtol)` | `np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)` | Combined absolute and relative tolerance. |
| `tc.verifyTrue(cond)` / `assert(cond)` | `assert cond` | Boolean condition evaluation. |
| `tc.verifyFalse(cond)` | `assert not cond` | Negated boolean condition. |
| `tc.verifyEmpty(x)` | `assert len(x) == 0` or `assert x.size == 0` | For lists/strings use `len()`, for NumPy arrays use `x.size == 0`. |
| `tc.verifyNotEmpty(x)` | `assert len(x) > 0` or `assert x.size > 0` | Non-empty check. |
| `tc.verifySize(arr, [m, n])` | `assert arr.shape == (m, n)` | Note: 1D array in MATLAB is often `(1, N)` or `(N, 1)`. In Python prefer `(N,)`. |
| `tc.verifyClass(obj, 'Type')` | `assert isinstance(obj, Type)` | Type verification. |
| `tc.verifyError(@() f(), 'ID:name')` | `with pytest.raises(ExpectedError, match="..."): f()` | Exception assertion with regex message matching. |
| `tc.verifyWarning(@() f(), 'ID:name')` | `with pytest.warns(ExpectedWarning, match="..."): f()` | Warning capture. |
| `tc.verifyGreaterThan(a, b)` | `assert a > b` | Scalar or array inequality (`np.all(a > b)`). |
| `tc.verifyLessThan(a, b)` | `assert a < b` | Scalar or array inequality (`np.all(a < b)`). |
| `tc.verifyNaN(x)` | `assert np.isnan(x).all()` | Check if element(s) are NaN. |
| `assertEqual(a, b)` with NaNs | `np.testing.assert_allclose(a, b, equal_nan=True)` | Allows NaNs to match in the same positions. |

---

## 2. Test Structure and Lifecycle

### MATLAB `TestCase` Class Structure

```matlab
classdef TestMyClass < matlab.unittest.TestCase
    properties
        fixture_data
    end

    methods (TestClassSetup)
        function init_class(tc)
            % Runs once per test class
            tc.fixture_data = load('reference_data.mat');
        end
    end

    methods (TestMethodSetup)
        function setup_method(tc)
            % Runs before each test method
        end
    end

    methods (Test)
        function test_calculation(tc)
            obj = MyClass(1.0, 2.0);
            res = obj.calculate();
            tc.assertEqual(res, 3.0, 'AbsTol', 1e-10);
        end
    end
end
```

### Pytest Equivalent

```python
import pytest
import numpy as np
import scipy.io
from my_module import MyClass


@pytest.fixture(scope="module")
def fixture_data():
    """Runs once per test module (TestClassSetup equivalent)."""
    return scipy.io.loadmat("reference_data.mat", squeeze_me=True)


@pytest.fixture(autouse=True)
def setup_method():
    """Runs before each test function (TestMethodSetup equivalent)."""
    yield


def test_calculation(fixture_data):
    obj = MyClass(1.0, 2.0)
    res = obj.calculate()
    assert res == pytest.approx(3.0, abs=1e-10)
```

---

## 3. Handling Test Fixtures (`.mat` files)

When translating tests that rely on pre-computed `.mat` data:

1. **Use `scipy.io.loadmat`**:
   ```python
   from scipy.io import loadmat

   # squeeze_me=True eliminates single-dimensional axes, simplifying array access
   data = loadmat("test_data.mat", squeeze_me=True, struct_as_record=False)
   ref_array = data["expected_output"]
   ```

2. **Inline Constants**:
   If the `.mat` file contains small matrices or arrays, inline them directly into the test file using `np.array([...])` to eliminate external file dependencies.

---

## 4. Key Traps in Numerical Parity

1. **1-Based vs. 0-Based Indices**:
   - In MATLAB: `indices = 1:5; arr(indices)`
   - In Python: `indices = np.arange(5); arr[indices]`
2. **Column-Major (Fortran) vs. Row-Major (C) Order**:
   - Reshaping a 2D matrix in MATLAB reads down columns first.
   - When flattening or reshaping in Python, pass `order='F'` if exact column-major memory layout is expected:
     ```python
     # MATLAB: reshape(A, 4, 3)
     np.reshape(A, (4, 3), order="F")
     ```
3. **Empty Arrays**:
   - MATLAB: `size([])` is `[0, 0]` and `isempty([])` is `true`.
   - NumPy: Empty arrays are created with `np.empty((0,))` or `np.array([])`. Check with `arr.size == 0`.
4. **Division**:
   - MATLAB `/` always performs floating-point division even on integer operands.
   - Python `/` produces `float`; `//` produces integer floor division. Match MATLAB using `/`.
5. **Tolerance Defaults**:
   - When converting MATLAB tests that use unspecified default tolerances, use standard floating-point tolerances:
     - `float64` / `double`: `atol=1e-12`, `rtol=1e-7`
     - `float32` / `single`: `atol=1e-5`, `rtol=1e-5`
