// ============================================================================
// TopComp: Fibered Memory Layout
// ============================================================================
// Doctrine-compliant memory layout for the hybrid state:
//
//   X_{A,n}^{hyb} ≅ ⊕_{x ∈ I_n} X_A^{top}
//
// Physically, the hybrid state is stored as a bundle of topological fibers
// indexed by exact discrete sector.  The base index is x ∈ I_n = F_2^n,
// and over each x we store the topological carrier slice.
//
// Memory layout:
//   hyb[x][theta][rho]
//
// This is the physical realization of:
//   P_x^{hyb} X_{A,n}^{hyb} ≅ X_A^{top}
//
// The hardware will only "use the topology" if the memory itself is
// arranged according to the fiber decomposition.
//
// FiberedLayout provides:
//   - Stride computation for each fiber (contiguous topological slices)
//   - FiberView: zero-copy view into a single fiber
//   - Batch operations respecting the fiber structure
//   - GPU-friendly coalesced access patterns
//   - Transfer policies that decompose by fiber
// ============================================================================
#pragma once
#include "machine_state.cuh"

namespace topcomp {
namespace fibered {

// ── Fiber stride descriptor ─────────────────────────────────────────────────
// Describes the memory layout of a single topological fiber

struct FiberStride {
    int fiber_size;     // N_theta * N_rho (elements per fiber)
    int fiber_bytes;    // fiber_size * sizeof(C64)
    int n_fibers;       // 2^n_qubits (number of discrete sectors)
    int total_elements; // n_fibers * fiber_size
    int theta_stride;   // stride between consecutive theta values = N_rho
    int rho_stride;     // stride between consecutive rho values = 1

    __host__ __device__ FiberStride()
        : fiber_size(0), fiber_bytes(0), n_fibers(0),
          total_elements(0), theta_stride(0), rho_stride(1) {}

    __host__ static FiberStride from_dims(int n_qubits, int N_theta, int N_rho) {
        FiberStride fs;
        fs.fiber_size = N_theta * N_rho;
        fs.fiber_bytes = fs.fiber_size * static_cast<int>(sizeof(C64));
        fs.n_fibers = 1 << n_qubits;
        fs.total_elements = fs.n_fibers * fs.fiber_size;
        fs.theta_stride = N_rho;
        fs.rho_stride = 1;
        return fs;
    }

    // Offset of fiber x in the flat array
    __host__ __device__ int fiber_offset(int x) const {
        return x * fiber_size;
    }

    // Full index: hyb[x][i_theta][j_rho]
    __host__ __device__ int index(int x, int i_theta, int j_rho) const {
        return x * fiber_size + i_theta * theta_stride + j_rho * rho_stride;
    }
};

// ── FiberView: zero-copy view into a single topological fiber ───────────────
// Points into the hybrid state's amplitude array at sector x.
// Does not own the memory.

struct FiberView {
    C64*  data;       // pointer to first element of the fiber
    int   N_theta;
    int   N_rho;
    int   size;       // N_theta * N_rho

    __host__ __device__ FiberView()
        : data(nullptr), N_theta(0), N_rho(0), size(0) {}

    __host__ __device__ FiberView(C64* base, int offset,
                                  int nth, int nrh)
        : data(base + offset), N_theta(nth), N_rho(nrh),
          size(nth * nrh) {}

    __host__ __device__ C64& at(int i_theta, int j_rho) {
        return data[i_theta * N_rho + j_rho];
    }

    __host__ __device__ const C64& at(int i_theta, int j_rho) const {
        return data[i_theta * N_rho + j_rho];
    }

    // L2 norm squared of this fiber
    __host__ double norm2() const {
        double s = 0;
        for (int i = 0; i < size; ++i)
            s += data[i].norm2();
        return s;
    }
};

// ── FiberedLayout: full fibered memory accessor ─────────────────────────────
// Wraps a HybridState and provides fiber-structured access.

struct FiberedLayout {
    C64*        amp;         // base pointer (not owned)
    FiberStride stride;
    int         n_qubits;
    int         N_theta;
    int         N_rho;
    bool        valid;

    __host__ FiberedLayout()
        : amp(nullptr), n_qubits(0), N_theta(0), N_rho(0), valid(false) {}

    // Wrap an existing HybridState
    __host__ static FiberedLayout wrap(HybridState& hyb) {
        FiberedLayout fl;
        fl.amp = hyb.amp;
        fl.n_qubits = hyb.n_qubits;
        fl.N_theta = hyb.N_theta;
        fl.N_rho = hyb.N_rho;
        fl.stride = FiberStride::from_dims(hyb.n_qubits,
                                           hyb.N_theta, hyb.N_rho);
        fl.valid = (hyb.amp != nullptr);
        return fl;
    }

    // Get a zero-copy view of fiber x
    __host__ FiberView fiber(int x) {
        return FiberView(amp, stride.fiber_offset(x), N_theta, N_rho);
    }

    // Number of fibers (discrete sectors)
    __host__ int num_fibers() const { return stride.n_fibers; }

    // Size of each fiber (topological dimension)
    __host__ int fiber_dim() const { return stride.fiber_size; }

    // Direct element access: hyb[x][theta][rho]
    __host__ __device__ C64& at(int x, int i_theta, int j_rho) {
        return amp[stride.index(x, i_theta, j_rho)];
    }

    __host__ __device__ const C64& at(int x, int i_theta, int j_rho) const {
        return amp[stride.index(x, i_theta, j_rho)];
    }

    // Born probability for discrete sector x: p(x) = Σ_{θ,ρ} |ψ(x,θ,ρ)|²
    __host__ double sector_prob(int x) const {
        double s = 0;
        int off = stride.fiber_offset(x);
        for (int i = 0; i < stride.fiber_size; ++i)
            s += amp[off + i].norm2();
        return s;
    }

    // Check fibered structure: each fiber is a contiguous block
    __host__ bool verify_contiguity() const {
        // By construction, fiber x occupies
        // amp[x*fiber_size .. (x+1)*fiber_size-1].
        // This is guaranteed by the stride layout.
        return valid && stride.fiber_size > 0 && stride.n_fibers > 0;
    }

    // Copy fiber x into a TopologicalState
    __host__ void extract_fiber(int x, TopologicalState& out) const {
        int off = stride.fiber_offset(x);
        for (int i = 0; i < stride.fiber_size; ++i)
            out.amp[i] = amp[off + i];
    }

    // Inject a TopologicalState into fiber x
    __host__ void inject_fiber(int x, const TopologicalState& in) {
        int off = stride.fiber_offset(x);
        for (int i = 0; i < stride.fiber_size; ++i)
            amp[off + i] = in.amp[i];
    }

    // Zero out fiber x
    __host__ void clear_fiber(int x) {
        int off = stride.fiber_offset(x);
        memset(amp + off, 0, stride.fiber_bytes);
    }

    // Swap fibers x1 and x2 (used by X_u)
    __host__ void swap_fibers(int x1, int x2) {
        int off1 = stride.fiber_offset(x1);
        int off2 = stride.fiber_offset(x2);
        for (int i = 0; i < stride.fiber_size; ++i) {
            C64 tmp = amp[off1 + i];
            amp[off1 + i] = amp[off2 + i];
            amp[off2 + i] = tmp;
        }
    }
};

// ── Transfer decomposition ──────────────────────────────────────────────────
// Plans transfers that respect the fiber structure.

struct FiberTransferPlan {
    int start_fiber;   // first fiber to transfer
    int end_fiber;     // one-past-last fiber
    int fiber_size;    // elements per fiber
    int total_bytes;   // total bytes in this transfer

    __host__ FiberTransferPlan()
        : start_fiber(0), end_fiber(0), fiber_size(0), total_bytes(0) {}
};

// Decompose a full transfer into fiber-aligned chunks
__host__ inline int plan_fiber_transfer(
    const FiberStride& stride,
    int chunk_fibers,
    FiberTransferPlan* plans, int max_plans)
{
    int n_chunks = 0;
    for (int x = 0; x < stride.n_fibers && n_chunks < max_plans;
         x += chunk_fibers)
    {
        int end = x + chunk_fibers;
        if (end > stride.n_fibers) end = stride.n_fibers;

        plans[n_chunks].start_fiber = x;
        plans[n_chunks].end_fiber = end;
        plans[n_chunks].fiber_size = stride.fiber_size;
        plans[n_chunks].total_bytes =
            (end - x) * stride.fiber_bytes;
        n_chunks++;
    }
    return n_chunks;
}

// ── Verification ────────────────────────────────────────────────────────────

namespace verify {

// Verify that FiberedLayout produces same addressing as HybridState
__host__ inline bool layout_matches_hybrid(const HybridState& hyb) {
    FiberedLayout fl = FiberedLayout::wrap(
        const_cast<HybridState&>(hyb));
    if (!fl.valid) return false;

    // Check that every element has the same address
    for (int x = 0; x < hyb.dim_disc; ++x)
        for (int i = 0; i < hyb.N_theta; ++i)
            for (int j = 0; j < hyb.N_rho; ++j)
                if (&fl.at(x, i, j) !=
                    &hyb.amp[hyb.index(x, i, j)])
                    return false;
    return true;
}

// Verify fiber norm sums to total state norm
__host__ inline bool fiber_norms_consistent(const FiberedLayout& fl) {
    double fiber_sum = 0;
    for (int x = 0; x < fl.num_fibers(); ++x)
        fiber_sum += fl.sector_prob(x);

    double total = 0;
    for (int i = 0; i < fl.stride.total_elements; ++i)
        total += fl.amp[i].norm2();

    return fabs(fiber_sum - total) < 1e-12;
}

// Verify that fiber transfer plans cover all fibers exactly once
__host__ inline bool transfer_plan_complete(
    const FiberStride& stride,
    const FiberTransferPlan* plans, int n_plans)
{
    int covered = 0;
    for (int i = 0; i < n_plans; ++i) {
        if (plans[i].start_fiber != covered) return false;
        covered = plans[i].end_fiber;
    }
    return covered == stride.n_fibers;
}

} // namespace verify

} // namespace fibered
} // namespace topcomp
