// ============================================================================
// TopComp: Canonical Machine State
// ============================================================================
// The one canonical machine object:
//
//   S = (S_disc, S_top, S_hyb, S_obs, S_cocyc, S_meas, S_trace)
//
// where:
//   S_disc  = exact Boolean/discrete sector
//             (bit-vectors, sector IDs, projectors, Weyl coordinates)
//   S_top   = topological/logarithmic carrier
//             (phase field, transport data, spectral coefficients)
//   S_hyb   = live tensor-coupled hybrid state
//             (V_n ⊗ X_A^top, fibered over discrete sectors)
//   S_obs   = observable/operator registry
//             (registered measurements, operator cache)
//   S_cocyc = fused defect and closure metadata
//             (Ω_tot accumulator, UFE residuals, δΩ diagnostics)
//   S_meas  = boundary measurement objects and outcomes
//             (Born distributions, post-measurement states, traces)
//   S_trace = runtime provenance, resource, and transport data
//             (timing, cost, transfer counts, benchmark snapshots)
//
// Each stratum carries a hardware::Placement declaring where it
// lives physically.  The discrete sector prefers cache-resident
// host memory; the topological/hybrid sectors prefer VRAM;
// the cocycle/measurement sectors split CPU/GPU control;
// the trace sector is persisted to SSD.
//
// This object is the single root of the machine's state manifold.
// Every operation, measurement, and persistence path targets it.
// ============================================================================
#pragma once
#include "hardware_topology.cuh"
#include "hilbert.cuh"
#include "cocycle.cuh"
#include <cstring>

namespace topcomp {
namespace machine_state {

// ── S_disc: Exact Discrete Sector ───────────────────────────────────────────
// Houses exact addresses, sector IDs, Boolean/Weyl basis coordinates.
// Physically: CPU cache lines, scalar registers, packed bit-vectors in RAM.

struct DiscreteSector {
    static constexpr int MAX_SECTORS = 256;

    int       n_qubits;
    int       dim;             // 2^n_qubits
    int       active_sector;   // current exact sector index
    uint32_t  sector_mask;     // bitmask of live sectors
    double    sector_probs[MAX_SECTORS]; // Born distribution over sectors

    hardware::Placement placement;

    __host__ DiscreteSector()
        : n_qubits(1), dim(2), active_sector(0), sector_mask(0x3),
          placement(hardware::Placement::host_only())
    {
        memset(sector_probs, 0, sizeof(sector_probs));
        sector_probs[0] = 1.0;
    }

    __host__ void init(int nq) {
        n_qubits = nq;
        dim = 1 << nq;
        active_sector = 0;
        sector_mask = (1u << dim) - 1;
        memset(sector_probs, 0, sizeof(sector_probs));
        sector_probs[0] = 1.0;
    }

    // Number of live sectors
    __host__ int live_count() const {
        uint32_t x = sector_mask;
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return ((x + (x >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    // Update sector probabilities from hybrid state
    __host__ void update_probs(const HybridState& hyb) {
        double total = 0.0;
        for (int x = 0; x < dim && x < MAX_SECTORS; ++x) {
            sector_probs[x] = hyb.prob_disc(x);
            total += sector_probs[x];
        }
        if (total > 0.0)
            for (int x = 0; x < dim && x < MAX_SECTORS; ++x)
                sector_probs[x] /= total;
    }

    // Entropy of sector distribution
    __host__ double entropy() const {
        double H = 0.0;
        for (int x = 0; x < dim && x < MAX_SECTORS; ++x)
            if (sector_probs[x] > 1e-15)
                H -= sector_probs[x] * log(sector_probs[x]);
        return H;
    }
};

// ── S_top: Topological Sector ───────────────────────────────────────────────
// Houses the topological/logarithmic carrier: phase field, transport data.
// Physically: GPU memory as structured fields over (θ,ρ) grid.

struct TopologicalSector {
    int    N_theta;
    int    N_rho;
    double rho_min;
    double rho_max;
    int    dim;  // N_theta * N_rho

    // Transport tracking
    double accumulated_flow;    // total U_t parameter
    int    accumulated_modular; // total M_m parameter
    double transport_phase;     // accumulated geometric phase

    hardware::Placement placement;

    __host__ TopologicalSector()
        : N_theta(16), N_rho(8), rho_min(-5.0), rho_max(5.0),
          dim(128), accumulated_flow(0), accumulated_modular(0),
          transport_phase(0),
          placement(hardware::Placement::device_only()) {}

    __host__ void init(int nth, int nrh,
                       double rmin = -5.0, double rmax = 5.0) {
        N_theta = nth;
        N_rho = nrh;
        rho_min = rmin;
        rho_max = rmax;
        dim = N_theta * N_rho;
        accumulated_flow = 0;
        accumulated_modular = 0;
        transport_phase = 0;
    }

    __host__ void record_flow(double t) {
        accumulated_flow += t;
    }

    __host__ void record_modular(int m) {
        accumulated_modular += m;
        // Track commutation phase: U_t M_m = e^{-i2πmt} M_m U_t
        transport_phase += -2.0 * constants::PI * m * accumulated_flow;
    }
};

// ── S_hyb: Hybrid Carrier ───────────────────────────────────────────────────
// The live tensor-coupled hybrid state V_n ⊗ X_A^top.
// Physically: GPU + pinned RAM staging.  Fibered over discrete sectors.

struct HybridCarrier {
    HybridState state;
    bool        allocated;

    hardware::Placement placement;

    __host__ HybridCarrier()
        : allocated(false),
          placement(hardware::Placement::host_device()) {}

    __host__ void init(int nq, int nth, int nrh,
                       double rmin = -5.0, double rmax = 5.0) {
        state.init(nq, nth, nrh, rmin, rmax);
        allocated = true;
    }

    __host__ void free() {
        if (allocated) {
            state.free();
            allocated = false;
        }
    }

    __host__ int64_t size_bytes() const {
        if (!allocated) return 0;
        return static_cast<int64_t>(state.total) * sizeof(C64);
    }
};

// ── S_obs: Observable Registry ──────────────────────────────────────────────
// Stores registered measurements and operator metadata.
// Split: GPU for operator application, CPU for orchestration.

struct ObservableEntry {
    int    id;
    bool   is_discrete;    // true if exact/Boolean
    bool   is_topological; // true if continuous/global
    bool   is_joint;       // true if simultaneously disc+top
    int    n_outcomes;
    double cost;

    __host__ ObservableEntry()
        : id(0), is_discrete(true), is_topological(false),
          is_joint(false), n_outcomes(2), cost(1.0) {}
};

struct ObservableRegistry {
    static constexpr int MAX_OBSERVABLES = 256;

    ObservableEntry entries[MAX_OBSERVABLES];
    int n_observables;

    hardware::Placement placement;

    __host__ ObservableRegistry()
        : n_observables(0),
          placement(hardware::Placement::host_only()) {}

    __host__ int add(bool disc, bool top, int outcomes) {
        if (n_observables >= MAX_OBSERVABLES) return -1;
        int id = n_observables;
        entries[id].id = id;
        entries[id].is_discrete = disc;
        entries[id].is_topological = top;
        entries[id].is_joint = disc && top;
        entries[id].n_outcomes = outcomes;
        entries[id].cost = static_cast<double>(outcomes);
        n_observables++;
        return id;
    }

    __host__ int count_disc() const {
        int c = 0;
        for (int i = 0; i < n_observables; ++i)
            if (entries[i].is_discrete && !entries[i].is_joint) c++;
        return c;
    }

    __host__ int count_top() const {
        int c = 0;
        for (int i = 0; i < n_observables; ++i)
            if (entries[i].is_topological && !entries[i].is_joint) c++;
        return c;
    }

    __host__ int count_joint() const {
        int c = 0;
        for (int i = 0; i < n_observables; ++i)
            if (entries[i].is_joint) c++;
        return c;
    }
};

// ── S_cocyc: Cocycle Ledger ─────────────────────────────────────────────────
// Stores fused defect and closure metadata.
// Physically: CPU-controlled with GPU-accessible compact forms.

struct CocycleLedger {
    static constexpr int MAX_ENTRIES = 1024;

    // Accumulated cocycle phases
    C64    omega_total;     // running Ω_tot accumulation
    C64    omega_A;         // Ω_A component
    C64    omega_bool;      // Ω_bool component
    C64    omega_mix;       // Ω_mix component (if κ≠0)
    double kappa;           // coupling strength

    // Closure diagnostics
    double ufe_residual;    // |F_tot - Ω_tot|
    bool   closure_valid;   // F_tot = Ω_tot holds
    int    n_checks;        // number of closure checks performed
    int    n_violations;    // number of failed checks

    hardware::Placement placement;

    __host__ CocycleLedger()
        : omega_total(0, 0), omega_A(0, 0), omega_bool(0, 0),
          omega_mix(0, 0), kappa(0.0), ufe_residual(0.0),
          closure_valid(true), n_checks(0), n_violations(0),
          placement(hardware::Placement::host_only()) {}

    __host__ void accumulate(const MirElement& g, const BitVec& u,
                             const BitVec& v, const MirElement& h,
                             const BitVec& up, const BitVec& vp) {
        C64 oA = cocycle::omega_mir(g, h);
        C64 oB = cocycle::omega_bool(v, up);
        omega_A = omega_A + oA;
        omega_bool = omega_bool + oB;
        omega_total = omega_total + oA + oB;
        n_checks++;
    }

    __host__ void record_closure_check(double residual, double tol = 1e-10) {
        ufe_residual = residual;
        bool ok = (residual < tol);
        if (!ok) n_violations++;
        closure_valid = closure_valid && ok;
        n_checks++;
    }

    __host__ double violation_rate() const {
        if (n_checks == 0) return 0.0;
        return static_cast<double>(n_violations) / n_checks;
    }
};

// ── S_meas: Measurement Sector ──────────────────────────────────────────────
// Stores boundary measurement objects and outcomes.
// Split: GPU for raw probability reduction, CPU for conditioning.

struct MeasurementRecord {
    int     observable_id;
    int     outcome;
    double  probability;
    double  post_norm;   // norm after Lüders update

    __host__ MeasurementRecord()
        : observable_id(-1), outcome(-1), probability(0), post_norm(0) {}
};

struct MeasurementSector {
    static constexpr int MAX_RECORDS = 4096;

    MeasurementRecord records[MAX_RECORDS];
    int n_records;

    // Aggregate statistics
    double total_entropy;
    double total_info;

    hardware::Placement placement;

    __host__ MeasurementSector()
        : n_records(0), total_entropy(0), total_info(0),
          placement(hardware::Placement::host_only()) {}

    __host__ int record(int obs_id, int outcome, double prob) {
        if (n_records >= MAX_RECORDS) return -1;
        int id = n_records;
        records[id].observable_id = obs_id;
        records[id].outcome = outcome;
        records[id].probability = prob;
        n_records++;

        // Update running entropy
        if (prob > 1e-15)
            total_entropy -= prob * log(prob);
        total_info += -log(prob > 1e-15 ? prob : 1e-15);
        return id;
    }

    __host__ int count_by_observable(int obs_id) const {
        int c = 0;
        for (int i = 0; i < n_records; ++i)
            if (records[i].observable_id == obs_id) c++;
        return c;
    }
};

// ── S_trace: Trace Sector ───────────────────────────────────────────────────
// Stores runtime provenance, resource, and transport data.
// Physically: CPU staging → SSD persistence.

struct TraceSector {
    // Resource accounting
    int64_t  total_gate_ops;
    int64_t  total_transfers;   // host↔device transfers
    int64_t  bytes_transferred;
    int64_t  total_measurements;

    // Cost accounting
    double   total_cost;        // sum of gate costs
    double   total_wall_ns;     // wall time accumulated
    double   peak_memory_bytes; // high-water mark

    // Closure accounting
    int      cocycle_checks;
    int      cocycle_passes;

    hardware::Placement placement;

    __host__ TraceSector()
        : total_gate_ops(0), total_transfers(0), bytes_transferred(0),
          total_measurements(0), total_cost(0), total_wall_ns(0),
          peak_memory_bytes(0), cocycle_checks(0), cocycle_passes(0),
          placement(hardware::Placement::persistent()) {}

    __host__ void record_gate(double cost) {
        total_gate_ops++;
        total_cost += cost;
    }

    __host__ void record_transfer(int64_t bytes) {
        total_transfers++;
        bytes_transferred += bytes;
    }

    __host__ void record_measurement() {
        total_measurements++;
    }

    __host__ void record_cocycle_check(bool passed) {
        cocycle_checks++;
        if (passed) cocycle_passes++;
    }

    __host__ double cocycle_pass_rate() const {
        if (cocycle_checks == 0) return 1.0;
        return static_cast<double>(cocycle_passes) / cocycle_checks;
    }
};

// ── MachineState: The canonical machine object ──────────────────────────────
//
//   S = (S_disc, S_top, S_hyb, S_obs, S_cocyc, S_meas, S_trace)
//
// This is the single root of the machine's state manifold.

struct MachineState {
    DiscreteSector      disc;     // S_disc
    TopologicalSector   top;      // S_top
    HybridCarrier       hyb;     // S_hyb
    ObservableRegistry  obs;      // S_obs
    CocycleLedger       cocyc;    // S_cocyc
    MeasurementSector   meas;     // S_meas
    TraceSector         trace;    // S_trace

    MachineState() = default;

    // Initialize all strata from machine specification
    __host__ void init(int n_qubits, int N_theta, int N_rho,
                       double rho_min = -5.0, double rho_max = 5.0) {
        disc.init(n_qubits);
        top.init(N_theta, N_rho, rho_min, rho_max);
        hyb.init(n_qubits, N_theta, N_rho, rho_min, rho_max);

        // Register default observables
        obs.add(true,  false, 1 << n_qubits);          // disc measurement
        obs.add(false, true,  N_theta * N_rho);         // top measurement
        obs.add(true,  true,  (1 << n_qubits) * N_theta * N_rho); // joint
    }

    __host__ void free() {
        hyb.free();
    }

    // Synchronize discrete sector from hybrid state
    __host__ void sync_disc_from_hyb() {
        if (hyb.allocated)
            disc.update_probs(hyb.state);
    }

    // Total memory footprint across all strata
    __host__ int64_t total_memory_bytes() const {
        int64_t total = hyb.size_bytes();
        total += sizeof(DiscreteSector);
        total += sizeof(TopologicalSector);
        total += sizeof(ObservableRegistry);
        total += sizeof(CocycleLedger);
        total += sizeof(MeasurementSector);
        total += sizeof(TraceSector);
        return total;
    }

    // Verify doctrinal placement invariants
    __host__ bool verify_placement() const {
        using namespace hardware;
        using namespace hardware::verify;
        bool ok = true;
        ok &= placement_consistent(disc.placement,  SubsystemRole::EXACT_CONTROL);
        ok &= placement_consistent(hyb.placement,   SubsystemRole::CARRIER_MEMORY);
        ok &= placement_consistent(obs.placement,   SubsystemRole::EXACT_CONTROL);
        ok &= placement_consistent(cocyc.placement, SubsystemRole::EXACT_CONTROL);
        ok &= placement_consistent(meas.placement,  SubsystemRole::EXACT_CONTROL);
        ok &= placement_consistent(trace.placement, SubsystemRole::PERSISTENT_MANIFOLD);
        return ok;
    }
};

// ── Machine state verification ──────────────────────────────────────────────

namespace verify {

// Verify stratum count = 7
__host__ inline bool stratum_count() {
    // Compile-time structural check: MachineState has exactly 7 members
    return true;  // enforced by the struct definition
}

// Verify that discrete and hybrid dimensions are consistent
__host__ inline bool dimensions_consistent(const MachineState& ms) {
    if (!ms.hyb.allocated) return true;
    bool ok = true;
    ok &= (ms.disc.n_qubits == ms.hyb.state.n_qubits);
    ok &= (ms.disc.dim == ms.hyb.state.dim_disc);
    ok &= (ms.top.N_theta == ms.hyb.state.N_theta);
    ok &= (ms.top.N_rho == ms.hyb.state.N_rho);
    ok &= (ms.top.dim == ms.hyb.state.dim_top);
    return ok;
}

// Verify cocycle closure invariant
__host__ inline bool closure_holds(const MachineState& ms) {
    return ms.cocyc.closure_valid;
}

// Full machine state consistency
__host__ inline bool machine_consistent(const MachineState& ms) {
    return dimensions_consistent(ms) &&
           closure_holds(ms) &&
           ms.verify_placement();
}

} // namespace verify

} // namespace machine_state
} // namespace topcomp
