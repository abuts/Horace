# Herbert Core: Python Migration Summary

## Executive summary

`herbert_core` is a substantial MATLAB-native scientific runtime shared by Horace, not a small utility library. The repository contains approximately:

- 1,719 MATLAB files under `herbert_core`
- 480 class-related MATLAB files
- 860 utility files
- 262 application files
- HDF5/NeXus, `.mat`, `.sqw`, `.nxspe`, `.spe`, `.par`, and instrument-data handling
- C++ MEX extensions
- MATLAB Parallel Computing Toolbox, MPI, Slurm, and file-based worker execution
- MATLAB graphics and GUI assumptions
- A large MATLAB/xUnit/CMake-driven test suite

The safest approach is a staged reimplementation with compatibility tests, rather than a line-by-line MATLAB-to-Python translation.

## Current dependency structure

### Runtime/bootstrap

[`herbert_init.m`](herbert_core/herbert_init.m) recursively adds these directories to the MATLAB path:

- `admin`
- `configuration`
- `classes`
- `utilities`
- `graphics`
- `applications`

The dependency graph is therefore implicit. Python should replace this with explicit package imports and should not reproduce `addpath`, `which`, or recursive path discovery.

Suggested package root:

```text
python/
  herbert/
    __init__.py
    config/
    io/
    data/
    geometry/
    numerics/
    parallel/
    instruments/
    fitting/
    plotting/
    compatibility/
  tests/
```

### Coupling to Horace

Herbert is an actual Horace library dependency:

- [`horace_paths.m`](herbert_core/horace_paths.m) resolves Herbert and Horace paths.
- [`CMakeLists.txt`](CMakeLists.txt) builds and installs both `herbert_core` and `horace_core`.
- [`documentation/adr/0007-use-herbert-as-library-dependency.md`](documentation/adr/0007-use-herbert-as-library-dependency.md) records the library-dependency decision.
- [`HORACE_PYTHON_MIGRATION_SUMMARY.md`](HORACE_PYTHON_MIGRATION_SUMMARY.md) provides the corresponding migration analysis for the Horace domain.

Before porting, define the shared boundary between future Python Herbert and Horace implementations:

- data objects
- instrument objects
- serialization formats
- configuration
- parallel job contracts
- error types
- file readers and writers

## Major dependency categories

| Area | Current MATLAB dependency | Python replacement |
|---|---|---|
| Numeric arrays | MATLAB matrices, implicit broadcasting, 1-based indexing | NumPy with explicit indexing and shape contracts |
| Scientific algorithms | MATLAB numerical functions and vectorization | NumPy/SciPy |
| HDF5/NeXus | `h5read`, `h5info`, `H5F.is_hdf5` | `h5py`, optionally `nexusformat` |
| XML | Custom XML utilities and MATLAB XML APIs | `lxml` or `xml.etree.ElementTree` |
| `.mat` files | MATLAB object/struct serialization | `scipy.io.loadmat` for basic data plus custom object migration |
| `.sqw` files | Herbert/Horace-specific serialization and paging | Versioned Python format layer, preferably HDF5-backed |
| `.nxspe` files | NeXus/HDF5 loader | `h5py` plus explicit schema validation |
| `.spe`, `.par`, `.map` | Custom text/binary loaders | Dedicated typed reader modules |
| Object model | MATLAB `classdef`, handle classes, dependent properties | Python classes/dataclasses, properties, and protocols |
| MEX | C++ MEX functions | C++ behind `pybind11` or Cython |
| Local parallelism | `parpool`, `parfor` | `concurrent.futures`, `joblib`, or Dask |
| MPI | MATLAB MPI wrappers and C++ communicator | `mpi4py` |
| Slurm | MATLAB cluster wrappers and scripts | Slurm adapter using Python subprocess APIs or Dask-MPI |
| Graphics | MATLAB figures and `sliceomatic` | Matplotlib, PyVista, or a separate GUI layer |
| Configuration | MATLAB singleton/config classes and files | Typed configuration plus TOML/YAML/JSON persistence |
| Testing | MATLAB xUnit and CMake-driven MATLAB tests | `pytest`, fixtures, parametrization, coverage, and golden tests |

## Highest-risk components

### Serializable class hierarchy

[`serializable.m`](herbert_core/utilities/classes/@serializable/serializable.m) provides:

- class versioning
- saveable-field declarations
- validation after object reconstruction
- backward compatibility for old versions
- MATLAB `.mat` persistence
- byte-stream serialization
- nested object-graph support

Do not replace this with Python `pickle`. Use a versioned, explicit serialization layer:

```text
herbert.serialization
  schema versioning
  type registry
  to_dict/from_dict
  binary payload support
  explicit migrations: v1 -> v2 -> v3
```

Use HDF5 for large scientific arrays and explicit metadata for object state. Retain a read-only MATLAB importer where practical, but do not make MATLAB object reconstruction the long-term format.

The tests under [`_test/test_serializers`](_test/test_serializers) should be an early migration target.

### MEX and native C++

[`_LowLevelCode/cpp/CMakeLists.txt`](_LowLevelCode/cpp/CMakeLists.txt) lists native targets including:

- `GetMD5`
- `cpp_communicator`
- `serialiser`
- `accumulate_cut_c`
- `bin_pixels_c`
- `calc_projections_c`
- `combine_sqw`
- `compute_pix_sums`
- `mtimesx_horace`
- `sort_pixels_by_bins`
- HDF5 reader plugins

The tests distinguish MEX and non-MEX execution; for example, [`test_cpp_serialize.m`](_test/test_serializers/test_cpp_serialize.m) skips tests when MEX is unavailable.

Recommended approach:

1. Keep performance-critical algorithms in C++.
2. Extract MATLAB-independent C++ APIs.
3. Bind them with `pybind11`.
4. Compare MATLAB and Python results using fixed fixtures.
5. Provide a NumPy fallback only where performance permits.

### MPI and cluster execution

The subsystem under [`herbert_core/classes/MPIFramework`](herbert_core/classes/MPIFramework) supports:

- MATLAB `parpool`
- external `mpiexec`
- C++ MPI communication
- file-based messaging
- Slurm cluster execution
- MATLAB worker processes

The file-based protocol in [`MessagesFilebased.m`](herbert_core/classes/MPIFramework/@MessagesFilebased/MessagesFilebased.m) is a process/message protocol, not merely a local helper.

Suggested Python abstraction:

```text
herbert.parallel
  Job
  Worker
  Message
  Communicator
  Scheduler
    LocalScheduler
    MpiScheduler
    SlurmScheduler
    FileMessageScheduler
```

Use `concurrent.futures` for local development, `mpi4py` for MPI, and a Slurm adapter for HPC. Retain the file-based backend temporarily for compatibility and diagnostics.

### HDF5/NeXus and scientific file formats

[`read_nexus_groups_recursive.m`](herbert_core/utilities/hdf_nexus/read_nexus_groups_recursive.m) and [`loader_nxspe.m`](herbert_core/classes/data_loaders/@loader_nxspe/loader_nxspe.m) depend on:

- HDF5 groups and datasets
- dataset attributes
- NeXus paths
- NXSPE-specific metadata
- detector and instrument information
- version-dependent file layouts

This is a good early Python target:

```python
class NxspeReader:
    def can_read(self, path): ...
    def read_metadata(self, path): ...
    def read_data(self, path): ...
```

Add schema validation and retain existing NXSPE fixtures as compatibility tests.

### Configuration and global state

[`config_store.m`](herbert_core/configuration/@config_store/config_store.m) implements a persistent singleton configuration store. Reproducing this literally would create hidden global state and test-order dependence.

Prefer:

- explicitly scoped configuration objects
- dependency injection
- process-level defaults only as a compatibility convenience
- TOML or JSON persistence
- isolated configuration directories in tests

### Multifit

The multifit subsystem under [`herbert_core/applications/multifit`](herbert_core/applications/multifit) manages:

- datasets
- foreground/background functions
- parameters
- fixed/free state
- parameter bindings
- masks
- local/global functions
- fit options

Port it after the data model and serialization layer. A possible Python model is:

```text
FitProblem
FitDataset
Parameter
ParameterConstraint
ModelFunction
FitResult
```

Use SciPy optimizers initially, but keep a Herbert-specific fitting API.

## Test coverage and migration value

The test taxonomy is defined in [`_test/CMakeLists.txt`](_test/CMakeLists.txt). Herbert-specific suites include:

- `test_admin`
- `test_config`
- `test_data_loaders`
- `test_debug_and_test_tools`
- `test_docify`
- `test_instrument_classes`
- `test_IX_classes`
- `test_map_mask`
- `test_mpi_wrappers`
- `test_multifit_herbert`
- `test_object_containers`
- `test_serializers`
- `test_utilities_her_geometry`
- `test_utilities_her_scientific`
- `test_utilities_herbert`
- `test_xunit_framework`

System tests cover job dispatch, MPI, MATLAB `parpool`, Slurm, and parallel framework behavior.

Classify tests before porting:

### Golden compatibility tests

Use MATLAB output as authoritative for:

- file readers
- geometry transforms
- numerical utilities
- serialization round trips
- instrument calculations

### Contract tests

Redesign in Python for:

- configuration
- error handling
- object mutation
- argument parsing
- worker lifecycle
- message delivery

### MATLAB-only tests

Do not port directly:

- MATLAB path initialization
- MATLAB figure handles
- MATLAB xUnit internals
- MEX discovery mechanics
- MATLAB-specific `parpool` behavior

There are approximately 813 `.m` test files under `_test`, including shared frameworks and helpers. Porting them wholesale should not be the first step.

## Recommended migration phases

### Phase 0: Freeze the compatibility contract

1. Enumerate public Herbert APIs used by Horace.
2. Identify serialized classes and file formats.
3. Define numerical tolerances and shape conventions.
4. Generate MATLAB reference outputs for representative fixtures.
5. Record expected errors and warnings.
6. Decide which APIs are retained, renamed, or deprecated.

### Phase 1: Build the Python foundation

Start with:

- argument parsing
- string/file utilities
- geometry and rotation utilities
- random/statistical helpers
- HDF5/NeXus inspection
- text data readers
- configuration primitives

Suggested dependencies:

- `numpy`
- `scipy`
- `h5py`
- `lxml`
- `pytest`
- `hypothesis`
- `packaging`

### Phase 2: Port data and serialization models

Implement:

- instrument metadata
- detector parameters
- sample and moderator objects
- dataset containers
- masks and axes
- serialization/version migrations
- NXSPE and related loaders

Add cross-language tests:

```text
MATLAB fixture -> Python read -> normalized object comparison
Python fixture -> MATLAB read, if backward compatibility is required
```

### Phase 3: Port native kernels

Extract and bind C++ algorithms with `pybind11`. Keep the Python interface independent of whether an implementation is Python/NumPy or C++.

### Phase 4: Port parallel execution

1. Implement local process execution.
2. Implement MPI with `mpi4py`.
3. Implement Slurm submission and monitoring.
4. Retain file-based messaging temporarily.
5. Add failure, cancellation, retry, and barrier tests.

### Phase 5: Port multifit and higher-level applications

Port multifit, instrument-specific applications, documentation tooling, and plotting only after the data and execution models are stable. Keep graphics separate from the headless scientific core.

## Suggested Python package split

```text
herbert/
  errors.py
  version.py
  config/
  io/
    nxspe.py
    nexus.py
    spe.py
    par.py
    map.py
    sqw.py
    mat_compat.py
  data/
  geometry/
  numerics/
  serialization/
  parallel/
  fitting/
  plotting/
  compatibility/
```

The `compatibility` package should be temporary and should not dictate the internal Python architecture.

## What to avoid

1. Automatic translation of all `.m` files.
2. Using `pickle` as the replacement for MATLAB object persistence.
3. Porting graphics before scientific data contracts.
4. Rewriting every MEX function in Python immediately.
5. Recreating MATLAB global state in Python.
6. Treating `parpool`, MPI, and Slurm as one backend.
7. Porting all tests before classifying them.

## Recommended first milestone

Build a standalone Python package that:

- reads representative NXSPE/NeXus files
- exposes typed detector, instrument, and data objects
- implements core geometry utilities
- passes cross-language golden tests against MATLAB

Prioritize:

- [`_test/test_data_loaders`](_test/test_data_loaders)
- [`_test/test_utilities_her_geometry`](_test/test_utilities_her_geometry)
- [`_test/test_utilities_her_scientific`](_test/test_utilities_her_scientific)
- [`_test/test_serializers`](_test/test_serializers)
- [`loader_nxspe.m`](herbert_core/classes/data_loaders/@loader_nxspe/loader_nxspe.m)
- [`read_nexus_groups_recursive.m`](herbert_core/utilities/hdf_nexus/read_nexus_groups_recursive.m)

This validates the most reusable and Python-friendly part of Herbert without prematurely committing to a parallel or GUI architecture.

## Bottom line

The recommended path is:

1. Specify MATLAB behavior and file contracts.
2. Build an explicit Python package structure.
3. Port pure numerics, geometry, and I/O first.
4. Create a safe, versioned serialization layer.
5. Keep performance-critical C++ and bind it with `pybind11`.
6. Implement parallel backends behind one Python job interface.
7. Port multifit and graphics after the data model is stable.
8. Use MATLAB/Python golden tests throughout the transition.

This should be treated as a staged Python reimplementation with a compatibility period, not as a direct syntax conversion.
