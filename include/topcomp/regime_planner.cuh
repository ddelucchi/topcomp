// ============================================================================
// TopComp: Hardware-Aware Regime Planner
// ============================================================================
// The central planning function:
//
//   Plan(S, Q, R, H) → ExecutionPlan
//
// where:
//   S = canonical structure (StructureGraph or MachineState)
//   Q = computation/query (what operations are requested)
//   R = resource budget (time, memory, transfer bandwidth)
//   H = hardware topology (CPU/GPU/RAM/VRAM/SSD capabilities)
//
// The planner decides:
//   - how much of the structure stays DISC
//   - how much lifts into TOP
//   - how much lives in HYB
//   - where each sector lives physically
//   - what transfer policy is used
//   - what measurement boundaries are active
//   - how to schedule kernels and branches
//
// Operation labeling:
//   Every operation carries an OpWeight tag:
//     DISC_HEAVY      → prefer CPU (bitwise, exact branches)
//     TOP_HEAVY       → prefer GPU (transport, flow, spectral)
//     HYB_HEAVY       → prefer GPU (tensor evolution)
//     MIXED           → prefer GPU (coupled operators)
//     MEAS_HEAVY      → GPU reduction + CPU conditioning
//     CLOSURE_CHECK   → prefer CPU (cocycle, diagnostics)
//     TRANSFER_HEAVY  → RAM staging
//
// Without this planner, the doctrine remains mathematically correct
// but physically mediocre.
// ============================================================================
#pragma once
#include "machine_state.cuh"
#include "structure.cuh"

namespace topcomp {
namespace planner {

using hardware::SubsystemKind;
using hardware::OpWeight;
using hardware::Placement;
using hardware::PlacementKind;
using hardware::TransferPolicy;
using hardware::TransferMode;
using hardware::HardwareTopology;

// ── Resource budget ─────────────────────────────────────────────────────────

struct ResourceBudget {
    int64_t max_memory_bytes;     // total memory cap
    int64_t max_vram_bytes;       // VRAM cap
    double  max_wall_time_ms;     // wall time budget
    double  max_transfer_bytes;   // transfer budget
    int     max_kernel_launches;  // kernel launch budget

    __host__ ResourceBudget()
        : max_memory_bytes(int64_t(4) << 30),   // 4 GB default
          max_vram_bytes(int64_t(8) << 30),      // 8 GB default
          max_wall_time_ms(10000.0),
          max_transfer_bytes(1e9),
          max_kernel_launches(10000) {}

    __host__ static ResourceBudget unlimited() {
        ResourceBudget r;
        r.max_memory_bytes = int64_t(1) << 40;
        r.max_vram_bytes = int64_t(1) << 40;
        r.max_wall_time_ms = 1e12;
        r.max_transfer_bytes = 1e15;
        r.max_kernel_launches = 1000000;
        return r;
    }
};

// ── Compute query ───────────────────────────────────────────────────────────
// Describes what the caller wants to do

struct ComputeQuery {
    // Gate counts by type
    int n_disc_gates;    // X, Z, H, P
    int n_top_gates;     // U, M
    int n_hyb_gates;     // W (Weyl)
    int n_mixed_gates;   // mixed operators (κ≠0)
    int n_measurements;  // measurement operations
    int n_closures;      // closure check requests

    // State dimensions
    int n_qubits;
    int N_theta;
    int N_rho;

    __host__ ComputeQuery()
        : n_disc_gates(0), n_top_gates(0), n_hyb_gates(0),
          n_mixed_gates(0), n_measurements(0), n_closures(0),
          n_qubits(1), N_theta(16), N_rho(8) {}

    // Classify overall operation weight
    __host__ OpWeight dominant_weight() const {
        int total = n_disc_gates + n_top_gates + n_hyb_gates + n_mixed_gates;
        if (total == 0) return OpWeight::MEAS_HEAVY;
        if (n_mixed_gates > 0) return OpWeight::MIXED;
        if (n_hyb_gates > total / 2) return OpWeight::HYB_HEAVY;
        if (n_top_gates > n_disc_gates) return OpWeight::TOP_HEAVY;
        return OpWeight::DISC_HEAVY;
    }

    // Total state dimension
    __host__ int64_t state_bytes() const {
        return static_cast<int64_t>(1 << n_qubits) *
               N_theta * N_rho * sizeof(C64);
    }

    // Should the state live on GPU?
    __host__ bool prefers_gpu() const {
        OpWeight w = dominant_weight();
        return w == OpWeight::TOP_HEAVY ||
               w == OpWeight::HYB_HEAVY ||
               w == OpWeight::MIXED ||
               w == OpWeight::MEAS_HEAVY;
    }
};

// ── Execution regime ────────────────────────────────────────────────────────

enum class ExecutionRegime : int {
    CPU_ONLY       = 0,  // everything on CPU (small states, all-disc)
    GPU_ONLY       = 1,  // everything on GPU (large states, all-top/hyb)
    SPLIT          = 2,  // disc ops on CPU, top/hyb on GPU
    STREAMING      = 3,  // fiber-chunked GPU with CPU staging
};

// ── Execution plan ──────────────────────────────────────────────────────────
// The output of the planner

struct ExecutionPlan {
    ExecutionRegime regime;

    // Placement decisions
    Placement disc_placement;
    Placement top_placement;
    Placement hyb_placement;
    Placement cocyc_placement;

    // Transfer policy
    TransferPolicy transfer;

    // Scheduling parameters
    int  gpu_block_size;      // CUDA block dimension
    int  fiber_chunk_size;    // fibers per transfer chunk
    bool use_pinned_memory;   // use pinned host memory
    bool use_async_transfer;  // overlap compute and transfer

    // Cost estimate
    double estimated_wall_ms;
    double estimated_transfer_bytes;
    int    estimated_kernel_launches;

    __host__ ExecutionPlan()
        : regime(ExecutionRegime::CPU_ONLY),
          gpu_block_size(256), fiber_chunk_size(1),
          use_pinned_memory(false), use_async_transfer(false),
          estimated_wall_ms(0), estimated_transfer_bytes(0),
          estimated_kernel_launches(0) {}
};

// ── The planner ─────────────────────────────────────────────────────────────
// Plan(S, Q, R, H) → ExecutionPlan

__host__ inline ExecutionPlan plan(
    const machine_state::MachineState& state,
    const ComputeQuery& query,
    const ResourceBudget& budget,
    const HardwareTopology& hw)
{
    ExecutionPlan ep;

    int64_t state_bytes = query.state_bytes();

    // Find hardware capabilities
    int vram_idx = hw.find(SubsystemKind::VRAM);
    int64_t vram_cap = (vram_idx >= 0)
        ? hw.subsystems[vram_idx].capacity_bytes : 0;

    // Decision 1: Does the state fit in VRAM?
    bool fits_vram = (state_bytes < vram_cap &&
                      state_bytes < budget.max_vram_bytes);

    // Decision 2: Is GPU beneficial?
    bool gpu_beneficial = query.prefers_gpu();

    // Decision 3: Choose regime
    if (!gpu_beneficial && state_bytes < int64_t(1) << 20) {
        // Small, disc-heavy → CPU only
        ep.regime = ExecutionRegime::CPU_ONLY;
        ep.disc_placement = Placement::host_only();
        ep.top_placement  = Placement::host_only();
        ep.hyb_placement  = Placement::host_only();
    }
    else if (fits_vram && gpu_beneficial) {
        // Fits in VRAM, GPU beneficial → GPU only
        ep.regime = ExecutionRegime::GPU_ONLY;
        ep.disc_placement = Placement::host_only();
        ep.top_placement  = Placement::device_only();
        ep.hyb_placement  = Placement::device_only();
        ep.use_async_transfer = true;
    }
    else if (gpu_beneficial) {
        // Doesn't fit VRAM → streaming with fiber chunks
        ep.regime = ExecutionRegime::STREAMING;
        ep.disc_placement = Placement::host_only();
        ep.top_placement  = Placement::host_device();
        ep.hyb_placement  = Placement::pinned();
        ep.use_pinned_memory = true;
        ep.use_async_transfer = true;

        // Compute chunk size: how many fibers fit in VRAM
        int fiber_bytes = query.N_theta * query.N_rho *
                          static_cast<int>(sizeof(C64));
        if (fiber_bytes > 0 && vram_cap > 0) {
            int max_fibers = static_cast<int>(
                (vram_cap * 3 / 4) / fiber_bytes);
            ep.fiber_chunk_size = (max_fibers > 0) ? max_fibers : 1;
        }
    }
    else {
        // Not GPU beneficial, large state → split
        ep.regime = ExecutionRegime::SPLIT;
        ep.disc_placement = Placement::host_only();
        ep.top_placement  = Placement::device_only();
        ep.hyb_placement  = Placement::host_device();
    }

    // Always: cocycle ledger on CPU
    ep.cocyc_placement = Placement::host_only();

    // Transfer policy
    ep.transfer = TransferPolicy::host_to_device(
        ep.use_async_transfer ? TransferMode::ASYNC : TransferMode::SYNC);
    ep.transfer.chunk_fibers = ep.fiber_chunk_size;
    ep.transfer.preserve_fiber = true;

    // Cost estimates
    double gate_cost = static_cast<double>(
        query.n_disc_gates + query.n_top_gates * 2 +
        query.n_hyb_gates * 4 + query.n_mixed_gates * 6);
    ep.estimated_wall_ms = gate_cost * 0.001;  // rough estimate
    ep.estimated_transfer_bytes =
        (ep.regime == ExecutionRegime::CPU_ONLY) ? 0.0 :
        static_cast<double>(state_bytes);
    ep.estimated_kernel_launches =
        (ep.regime == ExecutionRegime::CPU_ONLY) ? 0 :
        query.n_top_gates + query.n_hyb_gates + query.n_mixed_gates;

    return ep;
}

// ── Plan from StructureGraph (generic input) ────────────────────────────────

__host__ inline ExecutionPlan plan_from_structure(
    const structure::StructureGraph& sg,
    const ResourceBudget& budget,
    const HardwareTopology& hw)
{
    // Lift the structure to determine dimensions
    structure::lift::LiftResult lr = structure::lift::analyze(sg);

    ComputeQuery q;
    q.n_qubits = 1;
    // Compute log2 of disc dimension
    int dd = lr.dim_disc;
    while ((1 << q.n_qubits) < dd && q.n_qubits < MAX_QUBITS)
        q.n_qubits++;
    q.N_theta = (lr.dim_top > 1) ? lr.dim_top : 16;
    q.N_rho = 8;

    // Classify observables
    q.n_disc_gates = lr.n_disc_obs * 2;   // estimate: 2 gates per obs
    q.n_top_gates = lr.n_top_obs * 2;
    q.n_hyb_gates = lr.n_joint_obs * 2;
    q.n_measurements = sg.n_observables;

    machine_state::MachineState ms;
    ms.init(q.n_qubits, q.N_theta, q.N_rho);
    ExecutionPlan ep = plan(ms, q, budget, hw);
    ms.free();
    return ep;
}

// ── Verification ────────────────────────────────────────────────────────────

namespace verify {

// Verify that plan respects budget
__host__ inline bool plan_within_budget(
    const ExecutionPlan& ep, const ResourceBudget& budget)
{
    bool ok = true;
    ok &= (ep.estimated_transfer_bytes <= budget.max_transfer_bytes);
    ok &= (ep.estimated_kernel_launches <= budget.max_kernel_launches);
    return ok;
}

// Verify that plan assigns correct subsystem roles
__host__ inline bool plan_roles_correct(const ExecutionPlan& ep) {
    using namespace hardware::verify;
    using hardware::SubsystemRole;
    bool ok = true;
    ok &= placement_consistent(ep.disc_placement,
                               SubsystemRole::EXACT_CONTROL);
    ok &= placement_consistent(ep.cocyc_placement,
                               SubsystemRole::EXACT_CONTROL);
    return ok;
}

// Verify that fiber chunk preserves structure
__host__ inline bool plan_fiber_safe(const ExecutionPlan& ep) {
    return ep.transfer.preserve_fiber &&
           ep.fiber_chunk_size >= 1;
}

// Full plan verification
__host__ inline bool plan_valid(
    const ExecutionPlan& ep, const ResourceBudget& budget)
{
    return plan_within_budget(ep, budget) &&
           plan_roles_correct(ep) &&
           plan_fiber_safe(ep);
}

} // namespace verify

} // namespace planner
} // namespace topcomp
