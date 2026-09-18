# TopComp

TopComp is an experimental CUDA/C++ framework for exploring hybrid discrete, algebraic, topological, and quantum computation in one executable research stack.

The repository is best read as mathematical-computing software with a large internal verification harness, not as a claim of demonstrated quantum advantage or validated quantum hardware behavior.

## Implemented layers

### State and algebra

- discrete qubit state
- topological phase-space state
- hybrid tensor state
- mirror-group and cocycle operations
- Weyl, Clifford/Boolean, Lie, and representation-theory utilities
- Witt dual numbers and jet constructions

### Quantum operations

- gates over discrete, topological, and hybrid carriers
- density operators
- PVM/POVM measurement structures
- channels and instruments
- Naimark-style dilation experiments

### Compiler/runtime

- typed intermediate representation
- semantic passes
- virtual-machine/runtime structures
- hardware-topology descriptors
- fibered memory
- regime planning
- persistent serialization
- pipeline execution
- handle-based C-compatible API

### CUDA

CUDA kernels are included for selected orbit, gate, reduction, and structure-detection operations. The CMake project links against the CUDA runtime and cuRAND.

## Verification harness

`src/tests.cu` contains the project's internal check suite. The imported local snapshot reported 775 checks passing with zero failures.

That number is useful as a regression record, but it is not independent validation of the mathematical framework. The harness mixes exact/discrete invariants, tight floating-point identities, broad approximation sanity bounds, and runtime/API regressions. Those are deliberately separated in [VERIFICATION.md](VERIFICATION.md), including examples of the loose `0.01`–`0.3` approximation tolerances that should not be mistaken for precision validation.

## Build

Requirements:

- CMake 3.18+
- a C++17 compiler
- CUDA toolkit
- a CUDA-capable build toolchain

```bash
cmake -S . -B build \
  -DTOPCOMP_CUDA_ARCHITECTURES="75;80;86;89;90"

cmake --build build --parallel
```

To target one architecture, for example compute capability 8.6:

```bash
cmake -S . -B build -DTOPCOMP_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

## Tests

On a machine with a compatible NVIDIA GPU:

```bash
ctest --test-dir build --output-on-failure
```

The hosted workflow is designed for CUDA compilation only; GitHub-hosted runners do not provide the NVIDIA GPU runtime environment required by the executable test suite. This account also currently reports workflow startup failures before job creation, so the repository does not claim a green hosted-CI state.

## Architecture

The CMake target structure separates logical layers:

- `topcomp_core`
- `topcomp_contract`
- `topcomp_math`
- `topcomp_doctrine`
- `topcomp_ir`
- `topcomp_runtime`
- `topcomp_api`

They currently resolve into a shared header tree and are intended as architectural boundaries rather than independently packaged libraries.

## Scope and claims

TopComp does not claim:

- quantum speedup
- hardware error tolerance
- physical realization of its topological constructions
- equivalence to a production quantum SDK
- external proof of the framework's mathematical interpretations

It does provide a substantial executable environment for experimenting with those structures and checking internal algebraic/software invariants.

## License

Source is publicly viewable for portfolio and technical evaluation. See [LICENSE](LICENSE).
