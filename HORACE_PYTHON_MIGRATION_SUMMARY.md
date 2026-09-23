# Horace: Python Migration Summary

## Executive Summary

This document picks up from [`HERBERT_PYTHON_MIGRATION_SUMMARY.md`](HERBERT_PYTHON_MIGRATION_SUMMARY.md) and provides the architectural analysis and migration strategy for **Horace** (`horace_core` and associated native C++ kernels).

While `herbert_core` provides the generic scientific runtime (loaders, serializable objects, basic fitting, and HPC job dispatching), **Horace** is the domain-specific neutron scattering analysis framework. It specializes in:

- **Large 4D datasets**: \((\mathbf{Q}, \Delta E) = (q_x, q_y, q_z, \Delta E)\) neutron scattering event volumes.
- **The SQW data model**: Combining N-dimensional binned image data (`DnD`), full event-level records (`PixelData`), and comprehensive spectrometer/sample metadata (`Experiment`, `MainHeader`).
- **Reciprocal space geometry & projections**: Non-orthogonal crystallographic frames, Busing-Levy \(UB\) matrix transformations, and generalized coordinate projections (rectilinear, cylindrical, spherical).
- **Out-of-core data handling**: Historical paging across files for datasets exceeding available system RAM.
- **Core reduction & slicing algorithms**: `gen_sqw` (data reduction pipeline from `.spe`/`.nxspe` + `.par`), `cut` (multi-dimensional slicing and integration), `rebin`, and space-group symmetrisation.
- **Tobyfit**: Resolution-convoluted model fitting using Monte Carlo integration over spectrometer chopper and moderator characteristics.
- **High-performance native kernels**: C++ OpenMP-accelerated binning, projection calculation, sorting, and pixel accumulation.
- **A large MATLAB GUI & plotting layer**: Monolithic MATLAB figure interfaces (`horace.m`, `horace_fitting.m`, `horace_planner.m`).

### Repository Footprint (Horace Domain)

- **`horace_core/`**: ~180+ MATLAB files across:
  - `sqw/` (60+ files): Data models (`@sqw`, `@DnDBase`, `@d0d`..`@d4d`, `@SQWDnDBase`), `PixelData` hierarchy, `coord_transform`, `axes_blocks`, and `page_operations`.
  - `sqw/file_io/` (96 files): Multi-generation file format accessors (v1/v2 binary, v3 footer-based, v4 HDF5), serializers, and formatters.
  - `algorithms/` (55+ files): Scientific algorithms (`gen_sqw`, `accumulate_sqw`, `cut`, `rebin`, `change_crystal`, etc.).
  - `Tobyfit/` (20+ files): Spectrometer resolution modeling, Fermi/disk chopper matrices, moderator time-of-flight pulses.
  - `lattice_functions/` (8 files): Reciprocal lattice geometry, Bragg peak indexing, crystal orientation and lattice refinement.
  - `symop/` (7 files): Space-group symmetry operations and reflections.
  - `sqw_models/` (13 files): Physical models (spin waves, phonons, dispersion curves).
  - `configuration/`: Configuration singleton (`@hor_config`, `@hpc_config`, `@planner_config`).
  - `utilities/` (30+ files): Magnetic form factors (`MagneticIons`), binning helpers, validation.
  - `GUI/` (20 files): Monolithic MATLAB `.fig` and GUI event scripts (~300 KB of MATLAB code).
- **`_LowLevelCode/cpp/`**: 15 subdirectories containing ~70 C++ source and header files for performance-critical MEX routines.
- **`_test/`**: 24 Horace-specific test suites containing ~400+ MATLAB test scripts and extensive `.mat` and `.sqw` golden reference fixtures.

---

## Horace / Herbert Boundary and Layering

A clean Python migration requires strictly maintaining the boundary between Herbert and Horace:

```
+--------------------------------------------------------------------------+
|                              User Applications                           |
|      Jupyter Notebooks / Scripts / CLI / Future Web GUI (Plotly/Dash)    |
+--------------------------------------------------------------------------+
|                                  HORACE                                  |
|  - Data Model: SQW, DnD (0D-4D), PixelData, Experiment, MainHeader       |
|  - Slicing & Reduction: cut, gen_sqw, accumulate, rebin, symmetrise      |
|  - Reciprocal Space: Projections (line, plane, sphere), UB matrix, Bragg |
|  - Tobyfit: Monte Carlo resolution convolution (LET, MAPS, MERLIN)       |
|  - File I/O: v2/v3 binary SQW reader, v4 HDF5 / NeXus (.nxsqw) reader/wr |
|  - Visualization: Matplotlib 1D/2D slices, PyVista 3D reciprocal volumes |
+--------------------------------------------------------------------------+
|                                 HERBERT                                  |
|  - Loaders: NXSPE (NeXus HDF5), SPE, PAR, PHX                            |
|  - Instrument Hierarchy: IX_inst, IX_detector, IX_moderator, IX_sample    |
|  - Foundation Math: Rotations, Euler angles, coordinate transforms       |
|  - Optimization: Multi-dataset fitting core (multifit, parameters, binds)|
|  - Parallel Execution: Local multiprocessing, mpi4py, Slurm adapter      |
|  - Configuration: Typed configuration models, TOML/JSON persistence      |
+--------------------------------------------------------------------------+
|                          C++ Native Kernels                              |
|  bin_pixels, calc_projections, compute_pix_sums, sort_pixels (pybind11)  |
+--------------------------------------------------------------------------+
|                        Python Scientific Stack                           |
|       NumPy, SciPy, h5py, Numba, xarray / Dask, Spglib, Matplotlib       |
+--------------------------------------------------------------------------+
```

---

## Major Subsystem Analysis & Python Design

### 1. The SQW & DnD Object Model

Horace represents neutron data at two fundamental levels of detail:

1. **`DnD` (Dimensional Dataset, `d0d` to `d4d`)**:
   - Represents binned, multi-dimensional neutron image data without storing individual neutron events.
   - Core arrays:
     - `signal` (\(S\)): Mean intensity array, shape \((N_1, \dots, N_d)\).
     - `variance` (\(E\)): Variance \(\sigma^2\) (or standard error squared).
     - `npix`: Array containing the integer count of detector events contributing to each bin.
   - Associated with an `AxesBlock` (bin ranges, plot labels, step sizes) and a `Projection` (coordinate orientation relative to reciprocal space).
   - **Python Equivalent**:
     - Model as a typed `DnD` class or subclass of `xarray.DataArray`.
     - In Python, `xarray` naturally captures labeled coordinates, dimension names (e.g. `['Q_h', 'Q_k', 'Q_l', 'dE']`), attributes, and slicing.
     - Retain dedicated fields for `variance` and `npix`, with automatic Gaussian error propagation across arithmetic operators (`+`, `-`, `*`, `/`).

2. **`SQW` (S(Q, \omega) with full Pixel Data)**:
   - Holds both the binned image (`data` / `DnD`) and the underlying individual neutron events (`pix` / `PixelData`).
   - Composed of:
     - `main_header`: Creation date, file history, contributing runs.
     - `experiment_info`: Run files, instrument parameters, sample crystal structure, goniometer settings.
     - `detpar`: Detector group coordinates, angles, and distances.
     - `data`: Binned `DnD` image.
     - `pix`: `PixelData` array.
   - When operations (cuts, projections, scaling) are performed on an SQW object:
     - The operations act directly on the `PixelData`.
     - The binned `DnD` image is re-accumulated from the modified pixels.

3. **`PixelData`**:
   - In Horace MATLAB, pixel data is a \(9 \times N\) array containing 9 float/integer values per neutron event:
     1. \(u_1\): Coordinate along projection axis 1 (\(\text{Å}^{-1}\))
     2. \(u_2\): Coordinate along projection axis 2 (\(\text{Å}^{-1}\))
     3. \(u_3\): Coordinate along projection axis 3 (\(\text{Å}^{-1}\))
     4. \(dE\): Energy transfer \(\Delta E\) (\(\text{meV}\))
     5. `run_idx`: Contributing experimental run index (\(1 \le \text{idx} \le N_{\text{runs}}\))
     6. `detector_idx`: Detector group index
     7. `energy_idx`: Time-of-flight energy channel index
     8. `signal`: Neutron weight / intensity
     9. `variance`: Statistical variance of the event
   - Per ADR 0015 and ADR 0016:
     - On disk, coordinates and signal/variance are single-precision floats (`float32`), while run/detector/energy indices are integers.
     - In MATLAB memory, pixel data is cast to `float64` for mathematical calculations, but truncated back to `float32` on file write.
   - **Python Equivalent**:
     - Model with a structured NumPy array, a PyArrow Table, or a dedicated column-based container:
       ```python
       pixel_dtype = np.dtype([
           ('u1', np.float32),
           ('u2', np.float32),
           ('u3', np.float32),
           ('dE', np.float32),
           ('run_idx', np.uint32),
           ('detector_idx', np.uint32),
           ('energy_idx', np.uint32),
           ('signal', np.float32),
           ('variance', np.float32),
       ])
       ```
     - For mathematical transformations, calculate in float64 where necessary or vectorized in float32 for high performance and low memory footprint.

---

### 2. Large Datasets & Paging Strategy (Modernizing Out-of-Core)

In Horace MATLAB, out-of-core operations have been a known source of architectural complexity and performance bottlenecks (as documented in ADR 0018, ADR 0024, and `Filebased_sqw_pixels_operations.md`):
- Two conflicting legacy approaches existed: manual `PixelDataFileBacked` block iteration via temporary files versus whole-file direct seek access.
- In practice, `pixel_page_size` was often set to infinity to avoid the steep performance penalty of temporary file generation.

**Python Strategy**:
Do **NOT** replicate the MATLAB temporary-file paging mechanism (`TmpFileHandler`, `PixelDataFileBacked`, `PageOp_*`).
Instead, use modern Python out-of-core paradigms:
1. **HDF5 Chunking & Memory Mapping**:
   - With HDF5-backed v4 (`.nxsqw`), `h5py` supports chunked datasets and direct hyperslab reads.
   - Slicing and binning algorithms can stream chunks from disk in fixed buffer sizes (e.g., 50–200 MB chunks) using simple generators or Dask partitions.
2. **Memory-Mapped Arrays (`numpy.memmap`)**:
   - For reading large binary legacy pixel blocks without loading the entire multi-gigabyte file into memory.
3. **Explicit In-Memory vs. Streaming API**:
   - Users can explicitly load datasets in-memory (`sqw.load()`) when memory permits, or process out-of-core via streaming generators (`sqw.iter_chunks()`).

---

### 3. File Formats & Compatibility Layer

Horace data files on disk span four distinct formats:

| Format Version | Description | Python Handling Strategy |
|---|---|---|
| **v1 / Prototype** | Legacy pre-2008 format | Deprecate write; optional read-only conversion tool. |
| **v2 (Horace 1–2)** | Sequential binary format (header, detpar, data, pixels) | Implement pure Python binary parser using `struct` and `np.fromfile`. |
| **v3 (Horace 3.x)** | Binary format with constant blocks and end-of-file metadata directory | Implement pure Python reader. Allows immediate loading of existing user archives without requiring MATLAB. |
| **v4 / NeXus HDF5** | Modern HDF5-based structure (`.nxsqw`) | Primary format for Python Horace. Implemented via `h5py`. Direct interoperability with C++, Python, and NeXus tools. |

**Key Takeaway**:
Python Horace should default to **HDF5 (v4 / NeXus)** for all new writes, while providing a fast, read-only binary reader for legacy v2/v3 `.sqw` files to guarantee backwards compatibility with the neutron scattering user community's existing data archives.

---

### 4. Reciprocal Space, Projections, and Symmetrisation

1. **Crystal Coordinates and Transformations**:
   - Momentum transfer \(\mathbf{Q}\) is defined in reciprocal lattice units (r.l.u.):
     \[
     \mathbf{Q} = h\mathbf{a}^* + k\mathbf{b}^* + l\mathbf{c}^*
     \]
   - Horace transforms coordinates between:
     - **Laboratory / Spectrometer frame** (detector angles \(2\theta, \phi\) and flight path).
     - **Crystal Cartesian frame** (\(u_1, u_2, u_3\) in \(\text{Å}^{-1}\)), where crystal orientations and goniometer offsets are applied.
     - **Reciprocal Lattice Units (r.l.u.)** via the Busing-Levy \(UB\) matrix:
       \[
       \mathbf{u}_{\text{cart}} = UB \cdot \mathbf{h}
       \]
   - **Python Replacement**:
     - Pure NumPy implementation of crystallographic mathematics: metric tensors, reciprocal lattice matrices \(B\), orientation matrices \(U\), and goniometer arc rotations.
     - Eliminate MATLAB's slow cell arrays and loops with vectorized \((3 \times 3) \times (3 \times N)\) matrix multiplications.

2. **Projections (`coord_transform`)**:
   - `line_proj` / `LineProjBase`: Linear projection defining orthonormal or non-orthonormal cutting axes \(\mathbf{u}_1, \mathbf{u}_2, \mathbf{u}_3, \mathbf{u}_4\).
   - `cylinder_proj` / `sphere_proj`: Curvilinear projections mapping pixels to cylindrical \((Q_r, \theta, Q_z)\) or spherical \((|Q|, \theta, \phi)\) coordinates for powders and textured samples.
   - Python implementation: Subclasses of an abstract `Projection` class defining `forward(pixels)` and `inverse(coords)`.

3. **Crystallographic Symmetrisation & Space Groups (`symop`)**:
   - Used to improve statistics by folding equivalent symmetry regions of reciprocal space (e.g. reflecting across mirror planes or rotating by point-group operations).
   - Horace MATLAB currently includes custom `Symop` classes and optional experimental calls to the `spglib` library.
   - **Python Strategy**:
     - Directly integrate the official Python library `spglib` for space group generation, Wyckoff positions, and point-group symmetry matrices.
     - Vectorized coordinate reflection and rotation:
       \[
       \mathbf{u}_{\text{sym}} = R \cdot \mathbf{u} + \mathbf{t}
       \]

4. **Crystal Refinement (`lattice_functions`)**:
   - Refines lattice parameters \((a, b, c, \alpha, \beta, \gamma)\) and sample misorientation angles from observed Bragg reflection positions.
   - Python replacement: `scipy.optimize.least_squares` fitting against calculated Bragg peak centroids.

---

### 5. Core Scientific Algorithms

1. **`gen_sqw` (Data Reduction Pipeline)**:
   - Takes raw instrument files (`.spe` or `.nxspe`), detector definition (`.par` or NeXus), incident energies \(E_i\), lattice parameters, and run orientation angles (\(\psi, \omega, \text{gl}, \text{gs}\)).
   - Transforms time-of-flight and detector angles to \((\mathbf{Q}, \Delta E)\) in the Crystal Cartesian frame.
   - Sorts pixels into a 4D grid and outputs a consolidated `.sqw` file.
   - **Python Architecture**:
     - Pipeline composed of:
       `NxspeReader` \(\rightarrow\) `CoordinateCalculator` \(\rightarrow\) `PixelAccumulator` \(\rightarrow\) `Hdf5Writer`.
     - Parallelize across input run files using `concurrent.futures.ProcessPoolExecutor` or `mpi4py`.

2. **`cut` (Multi-Dimensional Slicing & Integration)**:
   - The primary user-facing tool in Horace:
     - Extracts 0D (point), 1D (line cut), 2D (slice), 3D (volume), or 4D sub-volumes from an SQW or DnD object.
     - Allows integrating over arbitrary coordinate ranges (e.g., integrating \([H, 0, 0]\) along \(K \in [-0.1, 0.1]\), \(L \in [-0.1, 0.1]\), \(\Delta E \in [10, 15]\)).
   - In Python:
     - User interface matches Python slicing semantics with keyword parameters (`cut(sqw, q1=[0, 0.02, 2], q2=[-0.1, 0.1], ...)`).
     - Filtering uses vector masks:
       ```python
       mask = (
           (u1 >= q1_min) & (u1 < q1_max) &
           (u2 >= q2_min) & (u2 < q2_max) &
           (u3 >= q3_min) & (u3 < q3_max) &
           (dE >= dE_min) & (dE < dE_max)
       )
       ```
     - For large datasets, evaluate filters in C++ / Numba or chunked over HDF5.

---

### 6. Tobyfit (Spectrometer Resolution Convolution)

Tobyfit is one of the most computationally demanding and scientifically valuable components in Horace:
- Fits theoretical \(S(\mathbf{q}, \omega, \mathbf{p})\) models (spin waves, phonons, magnons) to experimental SQW cuts.
- **Convolutes** the model with the instrument 4D resolution function \(\mathbf{R}(\mathbf{Q}, \Delta E)\) using Monte Carlo integration.
- Supports direct-geometry time-of-flight spectrometers:
  - Chopper systems: Fermi choppers (MAPS, MERLIN, SEQUOIA) and Disk choppers (LET).
  - Moderator pulse shape: Ikeda-Carpenter and empirical pulse distributions.
  - Detector geometries: \({}^3\text{He}\) tube arrays, mosaic spread, sample shape absorption.
- **Python Strategy**:
  1. Leverage `herbert.fitting` / `scipy.optimize` (`least_squares`) for parameter optimization.
  2. Implement the 11-dimensional Monte Carlo resolution sampling in vectorized NumPy / Numba or bind existing C++ resolution routines.
  3. Support user models written as standard Python/NumPy functions or JIT-compiled with Numba.
  4. Enable parallel evaluation of Monte Carlo trajectories across CPU cores via `joblib` or `multiprocessing`.

---

### 7. Native Kernels & Acceleration (`_LowLevelCode/cpp`)

The native kernels in Horace are already written in C++, but currently coupled to MATLAB's legacy C MEX API:

| Kernel | Purpose | Current Implementation | Python Target |
|---|---|---|---|
| `bin_pixels_c` | Bins 9-column pixel array into 4D image grid | C++ with OpenMP, coupled to `mex.h` | Decouple C++ logic; bind with `pybind11` / `nanobind`. |
| `calc_projections_c` | Projects pixel coordinates into target axes | C++ MEX | NumPy matrix multiplication + `pybind11` kernel. |
| `compute_pix_sums` | Computes bin signal sums, errors, and pixel counts | C++ OpenMP MEX | `pybind11` C++ kernel; Numba fallback. |
| `sort_pixels_by_bins` | Sorts pixel indices for fast contiguous lookups | C++ MEX (`std::sort` / index sort) | `np.argsort` or C++ parallel sort. |
| `combine_sqw` | Merges multiple run chunks / tmp files | C++ MEX with custom binary I/O | HDF5 native chunk writing / virtual datasets. |
| `mtimesx_horace` | Multi-dimensional matrix multiplication | C++ BLAS wrapper | Standard `np.matmul` / `@` operator (NumPy uses BLAS). |
| `file_parameters` | Fast ASCII `.par` / `.phx` parsing | C++ MEX | NumPy / `scipy.io` / pure Python parser. |

**Approach for C++**:
- Extract the pure C++ core logic from the MEX wrappers (removing `mxArray`, `mexErrMsgIdAndTxt`, `mexPrintf`).
- Use modern CMake to build Python C extensions using `pybind11`.
- Provide pure Python / Numba reference implementations for each kernel to enable:
  1. Seamless running on platforms without a C++ compiler.
  2. Fast golden test comparisons (`test_mex_nomex` equivalents in `pytest`).

---

### 8. Visualization & GUI Strategy

1. **Headless Core Separation**:
   - The entire scientific core (I/O, objects, slicing, fitting, resolution) must have **zero GUI dependencies**.
2. **Standard 1D / 2D Plotting**:
   - Provide a clean `horace.plotting` module built on `matplotlib`:
     - 1D cut plots with error bars (`errorbar`).
     - 2D slices with color maps, aspect ratio adjustments, and reciprocal unit labels.
     - Spaghetti plots (concatenated 1D cuts along high-symmetry reciprocal paths, e.g. \(\Gamma - X - M - \Gamma\)).
3. **Interactive 3D / Reciprocal Space Exploration**:
   - Replace MATLAB `sliceomatic` with modern 3D visualization:
     - `PyVista` (VTK-based) or `Plotly` for interactive 3D reciprocal isosurfaces, cutting planes, and volume rendering.
4. **GUI Strategy**:
   - Do **NOT** port MATLAB `.fig` files or monolithic procedural GUI code (`horace.m`, `horace_fitting.m`).
   - In modern workflows, Jupyter notebooks / JupyterLab widgets provide an interactive environment for neutron physicists.
   - If a standalone GUI is required later, build a decoupled web-based UI (e.g. using Dash, Streamlit, or PyQt/PySide) that talks strictly to the public Python Horace API.

---

## Test Taxonomy & Validation Plan

Horace contains a mature test suite in `_test/` that will form the backbone of the Python validation suite:

```
                          MATLAB Reference Suite
                 (400+ .m files, golden .sqw and .mat fixtures)
                                    |
                                    v
       +---------------------------------------------------------+
       |               Cross-Language Golden Testing             |
       |  - Load existing golden .sqw fixtures into Python       |
       |  - Compare Python slices & cuts against MATLAB cuts     |
       |  - Compare Tobyfit MC convolution against MATLAB values |
       |  - Numerical tolerance: 1e-5 (single) / 1e-12 (double)  |
       +---------------------------------------------------------+
                                    |
                                    v
       +---------------------------------------------------------+
       |                 Pytest Unit & Contract Suite            |
       |  - Parameter parsing & keyword argument validation      |
       |  - Reciprocal lattice & UB matrix algebra               |
       |  - Arithmetic operator overloads & error propagation    |
       |  - C++ pybind11 kernel vs. NumPy/Numba fallback parity  |
       +---------------------------------------------------------+
```

### Key Test Suites for Early Migration

1. **File I/O (`_test/test_sqw_file`)**:
   - Validate reading `test_sqw_file_read_write_v3.sqw`, `test_sqw_file_read_write_v3_3.sqw`, and `faccess_dnd_v4_sample.sqw`.
   - Verify byte-exact header decoding and pixel block extraction.
2. **C++ vs. Non-C++ Parity (`_test/test_mex_nomex`)**:
   - Port `test_bin_pixels_mex_nomex_all_modes.m` to verify that `pybind11` kernels and pure Python fallbacks produce identical results.
3. **Slicing & Cuts (`_test/test_algorithms`)**:
   - Run cuts on `test_cut_ref_sqw.sqw` and verify resulting signal, variance, and pixel counts against `test_cut_simple_to_complex_output.mat`.
4. **Tobyfit Components (`_test/test_TF_components`)**:
   - Validate resolution matrix calculations (`dq_matrix_DGfermi`, `dq_matrix_DGdisk`) and Monte Carlo outputs against `test_tobyfit_resfun_data.mat`.

---

## Proposed Python Package Structure

```text
python/
  horace/
    __init__.py
    version.py
    config.py                 # Scoped configuration (chunk size, threads, logging)
    
    core/                     # Data structures
      __init__.py
      sqw.py                  # Master SQW class
      dnd.py                  # DnD base, d0d - d4d classes
      pixels.py               # PixelData container (structured array / chunked)
      experiment.py           # Experiment, RunData, Sample metadata
      header.py               # MainHeader metadata
      
    projections/              # Reciprocal space coordinate transformations
      __init__.py
      base.py                 # Abstract Projection
      line.py                 # Line projection (orthonormal / affine)
      curvilinear.py          # Cylindrical, spherical projections
      axes.py                 # AxesBlock and binning grids
      
    crystallography/          # Lattice & symmetry
      __init__.py
      lattice.py              # Direct and reciprocal lattice parameters
      ub_matrix.py            # Busing-Levy UB matrix calculation
      refinement.py           # Crystal orientation & lattice refinement
      symmetry.py             # Spglib integration, space groups, Wyckoff
      
    io/                       # File reading & writing
      __init__.py
      reader.py               # Unified file reader factory
      v2_reader.py            # Legacy Horace v2 binary reader
      v3_reader.py            # Horace v3 binary reader
      hdf5_io.py              # v4 / NeXus HDF5 (.nxsqw) reader and writer
      
    algorithms/               # Core scientific algorithms
      __init__.py
      gen_sqw.py              # SPE/NXSPE -> SQW reduction pipeline
      cut.py                  # N-dimensional slicing and integration
      rebin.py                # Grid rebinning
      symmetrise.py           # Space-group symmetrisation
      eval.py                 # Function evaluation on SQW grids
      
    tobyfit/                  # Instrument resolution & convolution fitting
      __init__.py
      fitter.py               # Tobyfit fitting class (integrating with herbert)
      mc_sampler.py           # Monte Carlo resolution convolution engine
      resolution/
        fermi.py              # Fermi chopper spectrometer resolution
        disk.py               # Disk chopper spectrometer resolution
        moderator.py          # Moderator pulse shape models
      models/                 # Physical dispersion models
        spin_waves.py
        phonons.py
        dispersion.py
        
    plotting/                 # Visualization
      __init__.py
      slices.py               # 1D/2D Matplotlib plots
      spaghetti.py            # 1D dispersion trajectories
      volume.py               # 3D interactive reciprocal space (PyVista/Plotly)
      
    kernels/                  # High-performance native bindings
      __init__.py
      _binning.cpp            # pybind11 binding for bin_pixels
      _projections.cpp        # pybind11 binding for calc_projections
      _sums.cpp               # pybind11 binding for compute_pix_sums
      _sorting.cpp            # pybind11 binding for sort_pixels
      fallbacks.py            # Pure Python / Numba reference implementations
```

---

## Recommended Phased Implementation Roadmap

```
+--------------------------------------------------------------------------+
|  Phase 0: Foundations & Data Formats                                     |
|  - Implement `horace.io.v3_reader` and `horace.io.hdf5_io`               |
|  - Validate reading existing golden .sqw files from `_test/test_sqw_file`|
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 1: Crystallography & In-Memory Data Models                        |
|  - Implement `lattice`, `ub_matrix`, and `projections`                   |
|  - Implement in-memory `DnD` (0D-4D) and `PixelData` containers          |
|  - Implement arithmetic operations and error propagation                 |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 2: High-Performance C++ Kernels (pybind11)                        |
|  - Extract `bin_pixels`, `calc_projections`, `compute_pix_sums` from MEX |
|  - Build CMake/pybind11 extensions + Numba fallbacks                     |
|  - Verify parity with `test_mex_nomex` benchmarks                        |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 3: Slicing (`cut`) & Reduction (`gen_sqw`)                        |
|  - Implement multi-dimensional `cut` pipeline (in-memory & streaming)    |
|  - Implement `gen_sqw` ingestion from NXSPE and PAR files                |
|  - Validate with `_test/test_algorithms` reference cuts                  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 4: Symmetrisation & Crystal Refinement                            |
|  - Integrate `spglib` for space-group operations                         |
|  - Implement `symmetrise_sqw` and Bragg peak lattice refinement          |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 5: Tobyfit & Spectrometer Resolution Convolution                  |
|  - Implement Monte Carlo resolution sampler for LET, MAPS, MERLIN        |
|  - Integrate with `herbert.fitting` / `scipy.optimize`                   |
|  - Validate against `_test/test_TF_components` fixtures                  |
+--------------------------------------------------------------------------+
                                    |
                                    v
+--------------------------------------------------------------------------+
|  Phase 6: Visualization & Workflows                                      |
|  - Matplotlib 1D/2D cut plotting and spaghetti plots                     |
|  - Interactive 3D PyVista reciprocal viewer                              |
|  - End-to-end Jupyter notebook scientific workflows                      |
+--------------------------------------------------------------------------+
```

---

## What to Avoid

1. **Re-implementing Temporary-File Pixel Paging**: Do not port the MATLAB `TmpFileHandler` and `PageOp_*` temporary file machinery. Use chunked HDF5 datasets and streaming iterators instead.
2. **Porting Monolithic MATLAB GUIs**: Avoid translating `horace.m` or `.fig` files. Keep the Python library headless; build modern notebooks or lightweight web interfaces instead.
3. **Line-by-Line Syntax Translation**: Do not reproduce MATLAB idioms (`varargin`, `assignin`, `eval`, 1-based indexing). Design an idiomatic Python library with keyword arguments, dataclasses, and standard exceptions.
4. **Premature PICKLE Serialization**: Use HDF5/NeXus (`.nxsqw`) and JSON/TOML for metadata serialization, avoiding insecure or unversioned pickle storage.
5. **Coupling C++ Code to MATLAB**: Strip all MEX dependencies (`mex.h`, `mxArray`) from `_LowLevelCode/cpp` so the kernels are pure C++20 libraries with clean `pybind11` wrappers.

---

## Immediate Next Milestone

The recommended first deliverable to demonstrate end-to-end viability is:
1. **Build `horace.io`**:
   - Read an existing v3 `.sqw` file (e.g., `_test/test_sqw_file/test_sqw_file_read_write_v3.sqw`) into Python.
   - Verify that header metadata, detector parameters, image grids, and pixel blocks match the MATLAB reference values.
2. **Build `horace.projections` & `horace.crystallography`**:
   - Compute Busing-Levy \(UB\) matrix and transform pixel coordinates into reciprocal lattice units.
3. **Execute an In-Memory `cut`**:
   - Perform a 1D and 2D cut on the loaded SQW object in Python.
   - Verify that signal, error, and pixel counts match MATLAB's `test_cut_ref_sqw.sqw` outputs within floating-point tolerance.
