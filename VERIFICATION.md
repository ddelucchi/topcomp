# Verification model

TopComp's executable test harness is intentionally broad. It contains exact
algebraic identities, tight floating-point regression checks, approximation
sanity bounds, runtime/API checks, and serialization/compilation exercises.
Those categories are **not** treated as equivalent evidence.

## Evidence tiers

### Tier A — exact/discrete invariants

These checks are expected to be exact at the representation level or to reduce
to exact discrete structure. Examples include:

- bit-vector and sector-index identities
- group/inverse bookkeeping
- program length/cost accounting
- discrete measurement outcomes in basis states
- structural dimensions and serialization fields
- selected dual-number / truncated-polynomial identities

A failure here is normally a direct software-contract failure.

### Tier B — tight numerical identities

These compare floating-point implementations of analytically specified
relations with tight tolerances, commonly around `1e-14` to `1e-8`.
Examples include:

- `phi^2 = phi + 1`
- dual-number differentiation identities
- Lie-bracket / matrix-representation consistency
- Born-probability normalization
- exact-state purity / trace checks
- Weyl composition and interference regressions

These are numerical regression tests for the implemented formulas. They are
not independent proofs of the broader framework.

### Tier C — approximation and sanity bounds

Some algorithms are intentionally approximate or discretized. Their checks use
much looser tolerances and should be read as smoke/sanity bounds, not precision
validation. Current examples include checks with tolerances on the order of:

- `0.01` for selected approximate flow magnitudes
- `0.05` for lifted hybrid probabilities
- `0.1` for selected Padé/spectral norm behavior
- `0.3` for one approximate topological evolution norm check
- broad distribution bounds such as bias or total-variation distance below
  `0.5`

Those thresholds establish only the condition written in the test. They do not
establish convergence order, asymptotic accuracy, quantum advantage, or
physical realizability.

### Tier D — runtime and interface regressions

These checks exercise API surfaces, compiler/runtime plumbing, persistent
formats, measurement/reporting structures, and non-empty execution paths.
They are valuable software regressions but are not mathematical validation.

## What the historical check count means

The imported local snapshot recorded 775 internal checks passing with zero
failures. That count is a regression record for that snapshot. It should not be
read as 775 independent mathematical theorems or 775 external validation
results.

The current source should be evaluated by running the current harness on a
compatible CUDA machine:

```bash
cmake -S . -B build -DTOPCOMP_CUDA_ARCHITECTURES=86
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

GitHub-hosted runners can compile CUDA source but do not provide the NVIDIA GPU
runtime required by the executable test suite. In addition, this account
currently reports Actions startup failures before job creation, so no hosted
green badge is claimed.

## Validation that is still missing

TopComp does not currently provide:

- independent formal verification of its mathematical interpretations
- hardware-level validation
- demonstrated quantum speedup
- systematic convergence studies for every discretized/approximate subsystem
- cross-implementation parity against a mature external quantum SDK
- uncertainty/error budgets for every approximation path

Those are separate research tasks. The purpose of the current harness is to
make the implemented software falsifiable and regression-testable without
inflating that evidence into a stronger claim.
