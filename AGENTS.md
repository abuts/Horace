# AGENTS.md — Horace & PyHorace Workspace Guide

## Project Overview

This repository contains **Horace** and **Herbert**—a suite for visualising and analysing large 4D $(\mathbf{Q}, \Delta E)$ datasets from time-of-flight neutron inelastic scattering spectrometers—alongside **PyHorace** (`ph/`), its ongoing idiomatic Python rewrite.

- **Architectural References**:
  - [HERBERT_PYTHON_MIGRATION_SUMMARY.md](HERBERT_PYTHON_MIGRATION_SUMMARY.md): Analysis of `herbert_core` (scientific runtime, serializers, loaders, instruments, multifit, MPI).
  - [HORACE_PYTHON_MIGRATION_SUMMARY.md](HORACE_PYTHON_MIGRATION_SUMMARY.md): Analysis of `horace_core` (`sqw`/`DnD` data models, coordinate projections, binary/HDF5 file I/O, Tobyfit, C++ kernels).

---

## Repository Structure

```text
Horace_fork/
├── herbert_core/          # MATLAB Herbert core library (classes, utilities, config, instruments, graphics)
├── horace_core/           # MATLAB Horace domain library (sqw, dnd, algorithms, file_io, Tobyfit, symop)
├── _LowLevelCode/cpp/     # C++/OpenMP MEX sources (bin_pixels_c, accumulate_cut_c, sort_pixels_by_bins, etc.)
├── _test/                 # MATLAB xUnit test suites (test_*) and golden reference fixtures (.mat, .sqw, .nxspe)
├── ph/                    # PyHorace Python package and pytest suite
│   ├── pyproject.toml     # Package metadata and pytest configuration (Python >=3.12)
│   ├── pyhorace/          # Main Python package
│   │   ├── core/          # Serializable, Hashable, DataOpInterface, unique containers, configurations/
│   │   ├── coord_transf/  # Projections (aProjectionBase, line_proj, cyl_proj, sphere_proj) & AxesBlocks
│   │   ├── crystal/       # Crystal lattice & symmetry (oriented_lattice, ub_matrix, symmetry)
│   │   ├── data/          # SigVar, IX_axis, IX_dataset_1d/2d/3d, Experiment, instr_classes/, pixels/
│   │   ├── io/            # Binary (v2/v3) & HDF5 (v4) SQW/DnD file accessors, serializers, nxspe/spe/par
│   │   ├── sqw/           # SQWDnDBase, DnDBase, d0d..d4d, sqw, data_sqw_dnd, dnd_metadata, read_dnd/sqw
│   │   ├── fitting/       # Multifit and SQW fitting modules
│   │   ├── algorithms/    # Cut, rebin, and data reduction algorithms
│   │   └── parallel/      # Parallel execution framework
│   └── tests/             # Pytest suite mirroring pyhorace/ subpackages
└── .agents/
    └── skills/
        └── matlab-class-to-python/  # Skill for test-driven MATLAB-to-Python translation
```

---

## Python Environment (`conda`: `pyhorace`)

- **Always use the `pyhorace` conda environment** (`C:\Users\wkc26243\.conda\envs\pyhorace`) whenever running Python modules, scripts, or `pytest` test suites in this project.
- Since each terminal tool call runs in a fresh shell session, either activate the environment in the command (`conda activate pyhorace; ...`), use `conda run -n pyhorace ...`, or invoke the environment's binaries directly (`C:\Users\wkc26243\.conda\envs\pyhorace\python.exe` / `C:\Users\wkc26243\.conda\envs\pyhorace\Scripts\pytest.exe`).

---

## Core Workflows & Conventions

### 1. MATLAB-to-Python Translation (`matlab-class-to-python` Skill)
Whenever translating a MATLAB class (`@ClassName`), standalone function, or unit test into Python:
- **Always read and follow** [`.agents/skills/matlab-class-to-python/SKILL.md`](.agents/skills/matlab-class-to-python/SKILL.md).
- **Test-Driven Translation (TDT)**: Treat the MATLAB unit test in `_test/` as the executable specification. Translate the unit test to `ph/tests/` first (or alongside the module) and iterate until `pytest` passes with numerical parity.
- **Reuse Existing `pyhorace` Modules**: Search `ph/pyhorace/` before implementing any class or utility. Never duplicate or re-stub classes that already exist in `pyhorace` (e.g., `Serializable`, `Hashable`, `DataOpInterface`, `SigVar`, `config_store`, `hor_config`, `AxesBlockBase`, `aProjectionBase`, `DnDBase`, `sqw`, `Experiment`).

### 2. Python Class Design & `Serializable`
- **`Serializable` Hierarchy**: Classes inheriting from MATLAB `@serializable` must inherit from `pyhorace.core.serializable.Serializable` (built on Pydantic v2 `BaseModel` with `validate_assignment=True`):
  - Override `class_version(self) -> int` and `saveable_fields(self) -> Sequence[str]`.
  - Override `check_combo_arg(self) -> Self` for interdependent multi-field validation.
  - Suppress validation during multi-field staged updates by setting `self.do_check_combo_arg = False`, updating attributes, restoring `self.do_check_combo_arg = True`, and calling `self.check_combo_arg()`.
- **Value Class Semantics**: Most MATLAB Horace/Herbert classes are value classes where mutating methods return the modified object (`obj = obj.mutate()`). In Python, mutate `self` in-place **and** `return self` so both calling styles work.
- **Protected & Private Naming (`prop_` -> `_prop`)**: MATLAB code marks protected/private properties and helper methods with a trailing underscore (`property_name_`, `helper_method_`). In Python, always convert these to a single leading underscore (`_property_name`, `_helper_method`) and remove the trailing underscore (or use Pydantic `PrivateAttr()` on `Serializable` classes). Apply the same naming in subclasses and unit tests.

### 3. Numerical & Array Parity
- **Indexing & Layout**: Convert 1-based MATLAB indices to 0-based Python indices. Preserve Fortran column-major ordering (`order="F"`) when reshaping, flattening, or linear-indexing multi-dimensional arrays to match MATLAB memory layout and binary file formats.
- **Floating-Point Precision**: Default to `np.float64` to match MATLAB `double` (using `np.float32` only where specified by binary/HDF5 on-disk pixel schemas).
- **Degree Trigonometry**: Use `scipy.special.cosdg`, `sindg`, and `tandg` instead of `np.cos(np.deg2rad(...))` so multiples of $90^\circ$ evaluate to exact zeroes.

### 4. Unconverted Dependencies & MEX Routines
- **Never silently omit methods or tests**:
  - Keep full method signatures, type annotations, and complete translated MATLAB docstrings.
  - For unconverted collaborator classes, raise `NotImplementedError` with `# TODO(port-dependency): Requires <ClassName>`.
  - For compiled C++ MEX code paths (e.g., controlled by `hor_config().use_mex` or `'-force_mex'`), preserve the config check and wrapper method, and raise `NotImplementedError` with `# TODO(port-mex): Requires <mex_name>`.
  - Decorate corresponding unit tests with `@pytest.mark.xfail(raises=NotImplementedError, reason="...", strict=True)`.
- **Reconnect Stubs**: When translating a class that was previously stubbed via `TODO(port-dependency)`, wire up the real import and remove obsolete `xfail` markers.

### 5. Documentation & Typing
- Add PEP 484 type annotations to all functions, methods, and class fields.
- Port complete MATLAB `%` help headers and property/method comments into Python docstrings (`"""..."""`) without truncation.

---

## Common Commands

All Python and `pytest` commands must be executed within the `pyhorace` conda environment:

- **Run the full Python test suite**:
  ```powershell
  conda run -n pyhorace pytest ph/tests
  ```
- **Run a specific test file**:
  ```powershell
  conda run -n pyhorace pytest ph/tests/<subpackage>/test_<name>.py -v
  ```
- **Inspect `.mat` test fixture contents**:
  ```powershell
  conda run -n pyhorace python .agents/skills/matlab-class-to-python/scripts/inspect_mat.py <path/to/fixture.mat>
  ```
- **Install `pyhorace` in editable mode**:
  ```powershell
  conda run -n pyhorace pip install -e ph/[dev]
  ```

