---
name: matlab-class-to-python
description: >-
  Translate a single MATLAB class into an equivalent Python class, guided and validated
  by converting its MATLAB unit test into a pytest suite to ensure identical numerical results.
  Use when given a MATLAB class and its corresponding unit test to convert to Python.
---

# MATLAB Class to Python Translation Skill

Translate a single MATLAB class into an idiomatic, high-performance Python class by using its MATLAB unit test as the executable specification of behavior and numerical correctness.

## Core Principle: Test-Driven Translation (TDT)

Do not translate the class in isolation. The MATLAB unit test is the **source of truth** for expected behaviors, API signatures, edge cases, error conditions, and numerical tolerances.

The translation workflow follows a strict cycle:
1. **Analyze** both the MATLAB class and unit test.
2. **Translate the unit test to `pytest` first** to establish an automated referee.
3. **Translate the MATLAB class to Python**.
4. **Execute `pytest`** and iterate until all tests pass with numerical parity.
5. **Refactor & document** for idiomatic Python style.

---

## Step-by-Step Procedure

### Step 1: Input Analysis & Specification Extraction

Read the MATLAB class file (`<ClassName>.m`) and the corresponding unit test (`test_<ClassName>.m` or `<ClassName>_test.m`):

1. **Extract Public Interface**:
   - Class constructor and input argument handling (`nargin`, default parameters).
   - Public properties (stored vs. `Dependent` computed properties).
   - Public methods and their signatures (`function [out1, out2] = method(obj, arg1, ...)`).
   - Static methods (`methods (Static)`).
   - Overloaded operators (`plus`, `minus`, `mtimes`, `eq`, `disp`, etc.).
2. **Extract Behavior from Unit Tests**:
   - List every test method (`methods (Test)` or standalone test functions).
   - Identify test fixtures, setup (`TestClassSetup`, `TestMethodSetup`), and teardown.
   - Note required numerical tolerances: absolute (`AbsTol`) and relative (`RelTol`).
   - Identify expected error conditions (`verifyError`, `assertError`).
   - Identify external fixture files (e.g. `.mat` reference files).

---

### Step 2: Translate the Unit Test to `pytest`

Create `test_<class_name>.py` before or alongside the Python class.

1. **Map Test Fixtures & Setup**:
   - `TestMethodSetup` -> `pytest.fixture(autouse=False)` or fixture functions.
   - Loading test `.mat` data -> `scipy.io.loadmat(..., squeeze_me=True)` or inline test data.
2. **Map Assertions**:
   - Scalar equality: `assert actual == expected`
   - Floating-point scalars: `assert actual == pytest.approx(expected, abs=tol, rel=tol)`
   - Floating-point arrays: `np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)`
   - Exact array equality / shapes: `np.testing.assert_array_equal(actual, expected)`
   - Exception verification:
     ```python
     with pytest.raises(ExpectedException, match="error pattern"):
         obj.invalid_method_call()
     ```
3. See [MATLAB to Pytest Mapping Reference](./references/matlab_to_pytest_mapping.md) for full syntax comparison.

---

### Step 3: Translate the MATLAB Class to Python

Create `<class_name>.py` adhering to idiomatic Python and NumPy best practices:

#### 1. Class Structure & Initialization
- Map `classdef ClassName < SuperClass` to `class ClassName(SuperClass):` or standalone `class ClassName:`.
- Use `@dataclass` where appropriate for simple data containers.
- Map the constructor to `def __init__(self, ...):`:
  - Replace MATLAB `nargin` checks with keyword arguments and default values (`None` or typed defaults).
  - Explicitly initialize all instance attributes.

#### 2. Properties & Encapsulation
- **Public stored properties**: standard instance attributes `self.prop = value`.
- **Private properties (`Access = private/protected`)**: use single leading underscore `self._prop`.
- **Dependent properties (`Dependent`)**: use `@property` decorator:
  ```python
  @property
  def dependent_prop(self) -> float:
      return self._calc_value()

  @dependent_prop.setter
  def dependent_prop(self, value: float) -> None:
      self._set_value(value)
  ```
- **Constant properties (`Constant`)**: class-level variables or `typing.Final`.

#### 3. Method Translation
- Include explicit `self` as the first argument in instance methods.
- Mark static methods with `@staticmethod` (omit `self`).
- Map MATLAB overloaded methods to Python dunder methods:
  - `plus(a, b)` -> `__add__(self, other)`
  - `minus(a, b)` -> `__sub__(self, other)`
  - `times(a, b)` -> `__mul__(self, other)` (elementwise)
  - `mtimes(a, b)` -> `__matmul__(self, other)` (matrix product `@`)
  - `rdivide(a, b)` -> `__truediv__(self, other)`
  - `eq(a, b)` -> `__eq__(self, other)`
  - `disp(obj)` / `display(obj)` -> `__repr__(self)` and `__str__(self)`

#### 4. Array Semantics & Numerical Precision
- **0-based indexing**: Convert all 1-based MATLAB indices: `x(1)` -> `x[0]`, `x(1:end)` -> `x[:]`.
- **Vector shapes**: MATLAB 1D vectors are 2D \((1 \times N)\) or \((N \times 1)\). Prefer 1D NumPy arrays `(N,)` unless matrix operations strictly require 2D column vectors.
- **Matrix operations**:
  - Elementwise multiply: `A .* B` -> `A * B`
  - Matrix multiply: `A * B` -> `A @ B`
  - Transpose: `A.'` -> `A.T`
  - Conjugate transpose: `A'` -> `A.conj().T`
- **Memory ordering**: When reshaping or flattening multi-dimensional arrays, MATLAB uses Fortran (column-major) order. Use `order='F'` when necessary:
  ```python
  arr.flatten(order='F')
  arr.reshape((rows, cols), order='F')
  ```
- **Precision**: Default to `np.float64` to match MATLAB's native precision.

#### 5. Error Handling
- Replace MATLAB `error('ERR:id', 'message')` with standard Python exceptions (`ValueError`, `TypeError`, `IndexError`, `RuntimeError`) or specific domain exceptions.

---

### Step 4: Validate and Debug with `pytest`

Run the translated test suite using the terminal:

```bash
pytest test_<class_name>.py -v
```

1. **If tests fail on numerical comparison**:
   - Check for 1-based vs 0-based indexing discrepancies.
   - Check array shapes (row vector `(1, N)` vs column vector `(N, 1)` vs 1D array `(N,)`).
   - Check integer division: MATLAB `/` on integers does floating-point division; Python `//` is floor division, `/` is float.
   - Check Fortran vs C memory order during flatten/reshape operations.
2. **If tests fail on exceptions**:
   - Check whether the exception type or regex message matches what the test expects.
3. **Iterate** until 100% of the translated tests pass.

---

### Step 5: Code Quality & Typing Review

Once tests pass:
1. Add PEP 484 type annotations for method signatures and properties.
2. Port MATLAB function documentation blocks to Python docstrings (Google or NumPy style).
3. Ensure no MATLAB-isms remain (e.g. unnecessary `zeros()`, unused temp variables, manual loops that can be vectorised).

---

## Supporting Resources

- [MATLAB to Pytest Mapping Reference](./references/matlab_to_pytest_mapping.md): Detailed assertion, exception, and syntax conversion table.
- [End-to-End Class Translation Example](./examples/class_translation_example.md): Complete walkthrough of translating a MATLAB class and its unit test to Python.
