# Topological Computation / TopComp

TopComp is Daxx Delucchi's CUDA/C++ research framework for hybrid discrete, topological, and quantum computation.

## What is here

The project implements a layered, header-oriented compute stack:

- **State contract:** discrete qubit states, topological phase-space states, and hybrid tensor states.
- **Algebraic foundations:** mirror-group operations, cocycles, shadow functors, Weyl operators, Clifford/Boolean algebra, Lie and representation theory, Witt dual numbers and jets.
- **Quantum operations:** discrete and topological gates, hybrid operators, measurement, POVMs/PVMs, density matrices, CPTP channels, instruments, and Naimark dilation.
- **Compiler and runtime:** typed intermediate representation, optimization/lowering, a virtual machine, platform roles, hardware topology, fibered memory, regime planning, persistent serialization, and pipeline execution.
- **Exact dictionaries:** vector/operator/density decompositions, block and Walsh/Weyl faces, circuit kernels, transport, thermodynamic factorization, and probability reconstruction.
- **CUDA execution:** GPU kernels for golden-flow orbits, cocycle-derived bits, bias/correlation reductions, gate evolution, and structure detection.
- **External interface:** a handle-based C-compatible API surface in topcomp_api.cuh.

The implementation is organized into logical layers in CMakeLists.txt (topcomp_core, topcomp_contract, topcomp_math, topcomp_doctrine, topcomp_ir, topcomp_runtime, and topcomp_api). The current layout keeps these layers in one header tree so they can later be split into libraries.

## Build

Requirements:

- CMake 3.18 or newer
- CUDA toolkit and a CUDA-capable compiler
- A GPU architecture supported by the project configuration (75;80;86;89;90)

Configure and build with:

~~~powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
~~~

The primary targets are:

- mir_compute — verification and demonstration driver
- mir_tests — comprehensive automated test suite

## Local verification

The existing Release test executable was rerun during this import and completed with:

~~~
Results: 775 passed, 0 failed, 775 total
~~~

The repository import preserves the authored source, headers, build configuration, test suite, and reference material. Local build products, Visual Studio/CMake generated directories, binaries, logs, and empty output files are intentionally excluded.
