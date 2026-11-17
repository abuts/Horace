---
name: matlab-horace-refactor
description: >-
  Refactor MATLAB functions, scripts, classes (@ClassName), and xUnit tests in Horace
  and Herbert according to project coding conventions, ADRs (argument parsing, error
  identifiers, paged operations, serializable patterns), and MathWorks static-analysis
  best practices. Use when asked to clean up, modernize, modularize, or standardize
  MATLAB (.m) code in this repository.
---

# MATLAB Refactoring & Conventions Skill (Horace / Herbert)

Refactor MATLAB (`.m`) functions, `@ClassName` folders, `private/` helpers, and `_test/` suites in `Horace` and `Herbert` for clarity, modularity, performance, and adherence to project conventions without breaking public APIs or Python parity (`pyhorace`).

> **Note on generic MathWorks rules**: Generic MATLAB style guides recommend `lowerCamelCase` everywhere, `arguments` blocks, and `matlab.unittest`. In **Horace/Herbert**, follow the project conventions below wherever they intentionally differ from generic MathWorks defaults.

---

## 1. Refactoring Workflow

1. **Locate targets, callers, and tests**:
   - Inspect the target `.m` file(s), sibling methods in `@ClassName/`, `private/` helpers, and callers across `herbert_core/` and `horace_core/`.
   - Locate corresponding MATLAB xUnit tests in `_test/`. Do not alter public signatures or observable behavior unless explicitly requested.
2. **Baseline check**:
   - Note any relevant Architecture Decision Records in [`documentation/adr/`](../../../documentation/adr/) (especially ADRs `0018`, `0021`, `0022`, `0023`).
   - If MATLAB is available on the machine, check baseline static-analysis warnings (`checkcode('path/to/file.m')`) and run the relevant `_test/` suite via [`admin/validate_horace.m`](../../../admin/validate_horace.m).
3. **Refactor incrementally**:
   - Extract cohesive local functions (at the bottom of `.m` files) or shared `private/` helpers to remove duplicated logic and flatten deep nesting via early `return`.
   - Standardize argument parsing, error/warning identifiers, and encapsulation per Section 2.
   - Clean up performance anti-patterns and Code Analyzer warnings per Section 4.
4. **Verify**:
   - Confirm zero new Code Analyzer (`checkcode`) warnings on modified files.
   - Re-run the affected `_test/` tests using `validate_horace('<folder/test_name>')` (testing both pure-MATLAB `-nomex` and MEX paths when applicable). If MATLAB is unavailable in the current environment, explicitly report that checks were limited to static inspection.

---

## 2. Horace & Herbert Coding Conventions

### Naming & Layout
- **Functions, methods, properties, variables**: Use `snake_case` (matching existing `horace_core/` and `herbert_core/` code).
- **Classes**: Match the `@ClassName/` directory and `.m` filename (`UpperCamelCase` or established domain prefixes such as `IX_*`, `PageOp_*`, `SQWDnDBase`, `AxesBlockBase`).
- **Protected / private members & helpers**: Use a **trailing underscore** (`prop_`, `helper_method_`) for `Access = protected` / `Access = private` properties, methods, and `private/` helper functions.
- **Formatting**: 4 spaces per indent level (no tabs), explicit `end` on all `function`, `classdef`, `properties`, `methods`, and control blocks, and keep lines readable (wrap long statements cleanly with `...`).
- **Help text**: Keep an H1 summary line immediately below `function` / `classdef` (describing what the function/class does), followed by usage syntax, input/output descriptions, and examples. Preserve existing comments and documentation tags unless updating stale API docs.

### Argument Parsing ([ADR 0022](../../../documentation/adr/0022-use-keyword-args.md))
- Order arguments consistently: **required positional -> optional positional -> `-flag` / `'keyword', value` pairs**.
- Reuse existing Herbert utilities rather than hand-rolling `varargin` loops:
  - [`parse_char_options`](../../../herbert_core/utilities/general/parse_char_options.m) for `-flag` boolean options.
  - `set_positional_and_key_val_arguments` on `@serializable` subclasses.
- **Flags & Keywords**:
  - Prefix flag arguments with `-` (e.g., `'-nopix'`, `'-nomex'`, `'-all'`); default to `false`.
  - Remove legacy `no<flag>` negated forms where updating APIs (use `'-keyword', false` or a single clear flag like `'-nopix'`).
  - Allow minimum unambiguous prefix matching for `-flag` options, **never** abbreviate keyword names.
- Keep `arguments` blocks out of shared core signatures unless the target minimum MATLAB release explicitly supports them across CI; verify version compatibility before introducing newer syntax.

### Errors & Warnings ([ADR 0021](../../../documentation/adr/0021-errors-and-warnings.md), [ADR 0023](../../../documentation/adr/0023-error-format.md))
- Prefer `error(...)` / `try ... catch ME ... rethrow(ME)` over returning `[ok, mess]` flags for error propagation.
- Use the standard 3-part error identifier format:
  ```matlab
  error('HORACE:function_name:invalid_argument', 'Descriptive message: %s', detail);
  % or
  error('HERBERT:ClassName:io_error', 'Descriptive message: %s', detail);
  ```
  - Component 1: `HORACE` or `HERBERT` (ALL CAPS).
  - Component 2: Exact `function_name` or `ClassName` casing.
  - Component 3: `lower_snake_case` identifier, choosing from canonical IDs when applicable:
    `invalid_argument`, `invalid_output`, `not_implemented`, `not_available`, `array_mismatch`, `file_not_found`, `io_error`, `runtime_error`, `parallel_error`, `system_error`.
- **Never** perform a bare `catch` or strip stack traces/identifiers with `error(ME.message)`:
  ```matlab
  try
      ...
  catch ME
      switch ME.identifier
          case 'HORACE:known:id'
              % handle expected case or enrich with addCause
          otherwise
              rethrow(ME)
      end
  end
  ```
- Give `warning('HORACE:...:...', ...)` an identifier and state both the problem and how to fix or avoid it.

---

## 3. Class Design, `serializable`, & Paged Algorithms

- **Value semantics**: Most Horace/Herbert classes are value classes (not `< handle`). Mutating methods accept `obj` and return updated `obj` (`function obj = set_foo(obj, val)`).
- **`serializable` hierarchy** ([`@serializable`](../../../herbert_core/utilities/classes/@serializable/serializable.m)):
  - Validate single-property constraints in setters backed by `prop_` fields, and interdependent invariants in `check_combo_arg(obj)`.
  - Guard multi-field internal updates cleanly:
    ```matlab
    obj.do_check_combo_arg_ = false;
    obj.prop_a = val_a;
    obj.prop_b = val_b;
    obj.do_check_combo_arg_ = true;
    obj = obj.check_combo_arg();
    ```
  - Keep `classVersion` and `saveableFields` consistent with stored state; update `convert_old_struct` / `from_old_struct` if schema changes.
- **Paged pixel operations** ([ADR 0018](../../../documentation/adr/0018-algorithm-migration-to-paged-files.md)):
  - Algorithms that modify pixel data without changing image dimensions should use [`PageOpBase`](../../../horace_core/sqw/page_operations/@PageOpBase/PageOpBase.m) subclasses rather than loading all pixels into memory at once.
- **MEX / pure-MATLAB parity**:
  - Keep `hor_config().use_mex` / `'-nomex'` switches intact so pure-MATLAB fallbacks and MEX wrappers remain aligned.

---

## 4. Code Quality, Performance, & Cleanup Checklist

- **Pre-allocation**: Pre-allocate arrays (`zeros`, `false`, `nan`, `cell`) before loops; never grow arrays inside loops (`x(end+1) = ...` or `arr(i) = ...` without sizing).
- **Vectorization & matrix ops**: Replace trivial elementwise loops with vectorized expressions (`.*`, `./`, implicit expansion) when readability and numerical equivalence are preserved.
- **Resource safety**: Use `onCleanup` to restore modified global/config state (e.g., `hor_config`, `warning` state, current directory, open file handles) even on error.
- **Eliminate anti-patterns**:
  - Remove `eval`, `evalin`, `assignin`, and new `global` variables (use `persistent` with clear/reset support or config objects where needed).
  - Use `~` for unused function inputs and unused return values.
  - Remove dead code, unreachable branches, and redundant `if cond; tf = true; else; tf = false; end` boilerplate.
  - Keep duplicated logic in one helper rather than copying across methods or `d0d`..`d4d` / `sqw` siblings.

---

## 5. Verification Checklist

Before finishing any MATLAB refactoring task:
- [ ] Error IDs match `(HORACE|HERBERT):(function_name|ClassName):<lower_snake_id>`.
- [ ] Protected/private properties and helpers use trailing `_` (e.g. `prop_`, `helper_`).
- [ ] Argument order & `-flag` / keyword handling match ADR 0022 and reuse existing parsers.
- [ ] Temporary config/file/directory changes are guarded by `onCleanup`.
- [ ] Code Analyzer (`checkcode`) reports no new warnings on touched files (if MATLAB is available).
- [ ] Relevant `_test/` suites pass via `validate_horace(...)` (or report honestly if MATLAB cannot be executed in the current environment).
