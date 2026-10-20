---
name: matlab-class-to-python
description: >-
  Translate a MATLAB class or standalone function (including @ClassName folders and private/
  helpers) into an equivalent Python module, guided and validated by converting its MATLAB
  unit test into a pytest suite to ensure identical numerical results. Use when given a
  MATLAB class, function, or unit test to convert to Python.
---

# MATLAB Class & Function to Python Translation Skill

Translate a MATLAB class or standalone function into an idiomatic, high-performance Python implementation by using its MATLAB unit test as the executable specification of behavior and numerical correctness.

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
   - Class inheritance (`< serializable`, `< handle`, or value class).
   - Class constructor and input argument handling (`nargin`, `parse_args`, `set_positional_and_key_val_arguments`, default parameters).
   - Public properties (stored vs. genuinely `Dependent` computed properties vs. `Dependent` wrappers around private `prop_` fields used only for validation).
   - Public methods, signatures, and `nargout`-dependent return behaviors (`function [out1, out2] = method(obj, arg1, ...)`).
   - Static methods (`methods (Static)`) and `persistent` variables.
   - Overloaded operators (`plus`, `minus`, `times`, `mtimes`, `rdivide`, `uminus`, `mpower`, `eq`, `ne`, `disp`, etc.).
2. **Extract Behavior from Unit Tests**:
   - List every test method (`methods (Test)` or standalone test functions).
   - Identify test fixtures, setup (`TestClassSetup`, `TestMethodSetup`), teardown, and `persistent` state resets.
   - Note required numerical tolerances: absolute (`AbsTol`), relative (`RelTol`), or `[abs_tol, rel_tol]` pairs used in `equal_to_tol`.
   - Identify expected error conditions (`assertExceptionThrown`, `verifyError`, `assertError`) and warning checks (`verifyWarning`, `lastwarn`).
   - Identify external fixture files (e.g. `.mat` reference files).

---

### Step 2: Translate the Unit Test to `pytest`

Create `test_<class_name>.py` before or alongside the Python class.

1. **Map Test Fixtures & Setup**:
   - `TestMethodSetup` -> `pytest.fixture(autouse=True)` or explicit fixture functions.
   - Resetting MATLAB `persistent` variables -> `pytest.fixture(autouse=True)` with `yield` to restore state before and after each test.
   - Loading test `.mat` data -> `scipy.io.loadmat(..., squeeze_me=True)` or inline test data.
   - Copying template objects -> In MATLAB value classes, `obj2 = obj1` or `repmat(obj, ...)` copies by value. In Python, explicitly use `copy.copy(obj)` or `copy.deepcopy(obj)` when populating object lists or mutating copies in tests.
2. **Map Assertions**:
   - Scalar equality: `assert actual == expected`
   - `Serializable` equality within tolerance: `iseq, mess = obj1.equal_to_tol(obj2, tol=[atol, rtol]); assert iseq, mess`
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

#### 1. Class Structure, `Serializable`, & Initialization
- Map `classdef ClassName < SuperClass` to `class ClassName(SuperClass):` or standalone `class ClassName:`.
- For simple data containers without serialization or complex validation needs, use `@dataclass`.
- **`Serializable` Classes and Data Models**: When translating classes that inherit from MATLAB `@serializable`, require serialization/deserialization, or define structured data schemas:
  - Inherit from `pyhorace.core.serializable.Serializable` (which inherits from `pydantic.BaseModel` with `validate_assignment=True` and automatically registers subclasses in `SERIALIZABLE_REGISTRY`).
  - **Core Overrides**:
    - `class_version(self) -> int`: Return the current integer class version (matching `classVersion` in MATLAB).
    - `saveable_fields(self) -> Sequence[str]`: Return the ordered sequence of property names that define the serialized state (matching `saveableFields` in MATLAB).
    - `check_combo_arg(self) -> Self`: Validate multi-property invariants and interdependent constraints (e.g. `dia >= 2 * wall`). `Serializable` already hooks `check_combo_arg` into Pydantic's `@model_validator(mode="after")` and assignment lifecycle when `self.do_check_combo_arg` is `True`—subclasses only need to override `check_combo_arg(self)`.
    - `convert_old_struct(self, S: Dict[str, Any], ver: float) -> Tuple[Dict[str, Any], Serializable]` (or `from_old_struct`): Override when migrating legacy struct versions (`ver != self.class_version()`).
  - **Standard Types & Single-Field Validation**: Declare stored properties directly as Pydantic fields with type annotations, `Field(...)` constraints, or `@field_validator("prop_name")`. Do **not** recreate MATLAB's boilerplate `Dependent` getter/setter + private `prop_` field when the MATLAB setter only performs validation and calls `check_combo_arg`.
  - **Staged Updates & Flexible Constructors**:
    - Suppress interdependent checks during multi-field updates by setting `self.do_check_combo_arg = False`, assigning fields, setting `self.do_check_combo_arg = True`, and calling `self.check_combo_arg()`.
    - For MATLAB classes using `serializable.set_positional_and_key_val_arguments` (supporting positional args, key-value pairs, 4-character prefix matching, and `-dash` options), call `self.set_positional_and_key_val_arguments(pos_names, support_dash, *args, **kwargs)` inside `__init__`. Since Python `__init__` cannot return `(self, remains)`, store `self.remains` on the instance (and provide a helper function if tests unpack `[obj, remains]`).

#### 2. Value vs. Handle Semantics & Object Arrays
- **Value vs. Reference Semantics**:
  - Most Horace/Herbert MATLAB classes are value classes (not `< handle`), where mutating methods return the modified instance (`obj = obj.mutate()`). In Python, mutate `self` in-place **and** `return self` so both `obj.mutate()` and `obj = obj.mutate()` work seamlessly.
- **MATLAB Object Arrays**:
  - MATLAB natively treats `[obj1, obj2]` or `repmat(obj, m, n)` as an object array. In Python, represent sequences of objects as `list[MyClass]` (or `np.ndarray` with `dtype=object`).
  - Use `Serializable.to_struct_array(objs)`, `Serializable.serialize_array(objs)`, and `obj.from_bare_struct(seq_of_dicts)` when converting or serializing sequences of `Serializable` objects.

#### 3. Properties, Encapsulation, & `persistent` State
- **Public stored properties**: Standard instance attributes or Pydantic fields `field_name: Type = default`. Keep property names matching `saveable_fields()` if MAT-file / struct compatibility is required.
- **Private properties (`Access = private/protected`)**: Use single leading underscore `_prop` or Pydantic `PrivateAttr()`.
- **Dependent properties (`Dependent`)**: Use `@property` only for genuinely computed properties (derived on the fly from other fields):
  ```python
  @property
  def dependent_prop(self) -> float:
      return self._calc_value()

  @dependent_prop.setter
  def dependent_prop(self, value: float) -> None:
      self._set_value(value)
  ```
- **Constant properties (`Constant`)**: Class-level variables annotated with `ClassVar[Type]` or `typing.Final`.
- **`persistent` variables**: Map MATLAB `persistent` function/method variables to `ClassVar` attributes on the class with `@classmethod` getter/setter helpers.

#### 4. Method & Function Translation, `nargout`, & Operators
- Include explicit `self` as the first argument in instance methods; mark static methods with `@staticmethod` or `@classmethod`.
- **Standalone Functions, Local Subfunctions, & `private/` Helpers**:
  - Map local subfunctions at the bottom of a `.m` file and helper functions inside a sibling `private/` directory to module-private functions prefixed with a single underscore (`def _helper_name(...):`) in the same Python module (or a private submodule if shared across multiple files).
  - **Function-level `persistent` variables**: In standalone functions, map `persistent` state to `@functools.lru_cache`, function attributes, or a module-level `_state` variable (with a reset helper invoked by an `autouse=True` `pytest.fixture`).
  - **`varargin` & Flag Parsing (`parse_char_options` / `parse_keywords`)**: When a MATLAB function accepts trailing character flags (e.g. `'-nocell'`, `'-force'`, `'- legacy'`), provide idiomatic keyword-only boolean parameters (`*, nocell: bool = False, force: bool = False`) while optionally supporting legacy string flags in `*args` if existing callers/tests pass positional flag strings.
  - **Function Handles & Cell Arrays**: Map MATLAB function handles (`@func`, `@(x) ...`) to `Callable` parameters, and cell arrays (`{a, b}`) to Python `list` or `tuple`.
- **Handling `nargout` Dispatch**:
  - Python has no `nargout`. When a MATLAB method or function behaves differently based on output count (e.g. `[ok, mess] = check(obj)` vs `check(obj)` throwing an error), either:
    - Split into two clear functions/methods (e.g., `equal_to_tol(other) -> Tuple[bool, str]` for detailed tuple return vs. `__eq__(other) -> bool` for boolean check), or
    - Return a consistent tuple (or `NamedTuple`) and/or accept an explicit keyword flag (e.g., `raise_on_error: bool = True`, `return_details: bool = False`).
- **Map MATLAB Overloaded Operators to Python Dunder Methods** (always implement reflected `__r*__` operators when scalars or NumPy arrays can appear on the left-hand side, e.g., `2 * obj`):
  - `plus(a, b)` -> `__add__(self, other)` and `__radd__(self, other)`
  - `minus(a, b)` -> `__sub__(self, other)` and `__rsub__(self, other)`
  - `times(a, b)` -> `__mul__(self, other)` and `__rmul__(self, other)` (elementwise `.*`)
  - `mtimes(a, b)` -> `__matmul__(self, other)` and `__rmatmul__(self, other)` (matrix product `*` / `@`)
  - `rdivide(a, b)` -> `__truediv__(self, other)` and `__rtruediv__(self, other)`
  - `power(a, b)` / `mpower(a, b)` -> `__pow__(self, power)`
  - `uminus(a)` / `uplus(a)` -> `__neg__(self)` / `__pos__(self)`
  - `eq(a, b)` / `ne(a, b)` -> `__eq__(self, other)` / `__ne__(self, other)`
  - `disp(obj)` / `display(obj)` -> `__repr__(self)` and `__str__(self)`

#### 5. Array Semantics & Numerical Precision
- **0-based indexing**: Convert all 1-based MATLAB indices: `x(1)` -> `x[0]`, `x(1:end)` -> `x[:]`.
- **Vector shapes**: MATLAB 1D vectors are 2D `(1, N)` or `(N, 1)`. Prefer 1D NumPy arrays `(N,)` unless matrix operations strictly require 2D row/column vectors.
- **Matrix operations**:
  - Elementwise multiply: `A .* B` -> `A * B`
  - Matrix multiply: `A * B` -> `A @ B`
  - Transpose: `A.'` -> `A.T`
  - Conjugate transpose: `A'` -> `A.conj().T`
  - Left division (`A \ b`): `np.linalg.solve(A, b)` or `np.linalg.lstsq(A, b, rcond=None)[0]`
- **Degree-based trigonometry (`sind`, `cosd`, `tand`)**:
  - Avoid `math.cos(math.radians(90.0))` which yields `6.12e-17` instead of `0.0` and fails on arrays. Use `scipy.special.cosdg`, `sindg`, `tandg` (which preserve exact zeroes at multiples of $90^\circ$ and support NumPy arrays) or `np.cos(np.deg2rad(x))`.
- **Memory ordering**: When reshaping or flattening multi-dimensional arrays, MATLAB uses Fortran (column-major) order. Use `order='F'` when layout parity matters:
  ```python
  arr.flatten(order='F')
  arr.reshape((rows, cols), order='F')
  ```
- **Precision**: Default to `np.float64` to match MATLAB's native `double` precision.

#### 6. Error & Warning Handling
- Replace MATLAB `error('HERBERT:ClassName:invalid_argument', 'message')` or `error('HORACE:ClassName:invalid_argument', 'message')` with standard Python exceptions (`ValueError`, `TypeError`, `IndexError`, `RuntimeError`, `NotImplementedError`) preserving key diagnostic substrings checked by unit tests.
- Replace MATLAB `warning('ID:name', 'message')` with `warnings.warn("message", UserWarning)` (or `DeprecationWarning`).

---

### Step 4: Validate and Debug with `pytest`

Run the translated test suite using the terminal:

```bash
pytest test_<class_name>.py -v
```

1. **If tests fail on numerical comparison**:
   - Check for 1-based vs 0-based indexing discrepancies.
   - Check array shapes (row vector `(1, N)` vs column vector `(N, 1)` vs 1D array `(N,)`).
   - Check degree trigonometry rounding (`cosd(90)` vs `math.cos(math.radians(90))`).
   - Check integer division: MATLAB `/` on integers does floating-point division (with round-half-away-from-zero in some contexts); Python `//` is floor division, `/` is float.
   - Check Fortran vs C memory order during flatten/reshape/linear-indexing operations.
   - Check unintended reference aliasing (missing `copy.copy()` / `copy.deepcopy()`).
2. **If tests fail on exceptions or warnings**:
   - Check whether the exception type or regex message matches what the test expects.
   - If a Pydantic `@field_validator` raises `ValidationError` (which subclasses `ValueError`), ensure `pytest.raises(ValueError, match=...)` matches the underlying message.
3. **Iterate** until 100% of the translated tests pass.

---

### Step 5: Code Quality & Typing Review

Once tests pass:
1. Add PEP 484 type annotations for all method signatures and properties.
2. Port MATLAB function documentation blocks (`%` comments) to Python docstrings (Google or NumPy style).
3. Ensure no MATLAB-isms remain (e.g. unnecessary pre-allocation with `zeros()` before list comprehensions, manual index loops that can be vectorized).

---

## Supporting Resources

- [MATLAB to Pytest Mapping Reference](./references/matlab_to_pytest_mapping.md): Detailed assertion, exception, and syntax conversion table.
- [End-to-End Class Translation Example](./examples/class_translation_example.md): Complete walkthrough of translating a MATLAB `@serializable` class and its unit test to Python.
