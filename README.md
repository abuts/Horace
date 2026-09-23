# Horace

Horace is a suite of programs for the visualisation and analysis of large datasets from time-of-flight neutron inelastic scattering spectrometers.

The documentation on Horace's latest development version can be found [here](https://pace-neutrons.github.io/Horace/unstable/). 

The documentation for the latest released version is available [here](https://pace-neutrons.github.io/Horace/).

If you have problem with Horace, Horace installation or found a bug, please contact [Horace Help](mailto:HoraceHelp@stfc.ac.uk) describing the issue. Somebody from our team will come back to you to deal with the problem.

Horace is licensed under GPL v3 and includes licensed libraries:

- [MSMPI](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi)
- [MPICH](https://www.mpich.org/)

## Python Migration

Comprehensive architectural analysis and phased migration roadmaps for porting Horace and Herbert to Python:
- [Herbert Python Migration Summary](HERBERT_PYTHON_MIGRATION_SUMMARY.md) - Analysis of `herbert_core`, scientific runtime, loaders, MPI framework, and multifit.
- [Horace Python Migration Summary](HORACE_PYTHON_MIGRATION_SUMMARY.md) - Analysis of `horace_core`, SQW/DnD data model, reciprocal space geometry, Tobyfit resolution convolution, out-of-core HDF5 handling, and C++ native kernels.
