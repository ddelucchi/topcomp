// ============================================================================
// TopComp Layer 3: Runtime / Virtual Machine
// ============================================================================
// The execution engine for PlatformIR programs.
//
// Key capabilities:
//   - Multi-register state management (DISC / TOP / HYB slots)
//   - State-type-aware gate dispatch (fast path for pure sectors)
//   - Lifting / lowering with correct ownership semantics
//   - Born measurement engine (discrete + joint)
//   - Cocycle / UFE verification during execution
//   - Measurement recording for result extraction
//
// Execution model:
//   PlatformVM vm(spec);
//   vm.run(ir);
//   double p0 = vm.prob(0, 0);
//
// The VM steps through the PlatformIR one operation at a time,
// dispatching each to the appropriate state representation.
// Disc ops on DISC registers use the fast QubitState path;
// disc ops on HYB registers use the full hybrid_gates path.
// ============================================================================
#pragma once
#include "platform_ir.cuh"
#include "machine.cuh"

namespace topcomp {
namespace runtime {

using contract::StateType;
using contract::MachineSpec;

// ── State slot: tagged union of DISC / TOP / HYB ────────────────────────────

struct StateSlot {
    StateType type;
    bool      allocated;

    // State storage (only one is active, determined by type)
    QubitState         disc;
    TopologicalState   top;
    HybridState        hyb;

    StateSlot() : type(StateType::DISC), allocated(false) {
        disc.amp = nullptr;
        top.amp  = nullptr;
        hyb.amp  = nullptr;
    }

    __host__ void alloc_disc(int n_qubits) {
        release();
        type = StateType::DISC;
        disc.init(n_qubits);
        disc.set_basis(0);
        allocated = true;
    }

    __host__ void alloc_top(int nth, int nrh,
                            double rmin = -5.0, double rmax = 5.0) {
        release();
        type = StateType::TOP;
        top.init(nth, nrh, rmin, rmax);
        top.set_delta(PhasePoint(0.0, 0.0));
        allocated = true;
    }

    __host__ void alloc_hyb(int nq, int nth, int nrh,
                            double rmin = -5.0, double rmax = 5.0) {
        release();
        type = StateType::HYB;
        hyb.init(nq, nth, nrh, rmin, rmax);
        // Default: |0⟩ ⊗ ξ₀
        TopologicalState xi0;
        xi0.init(nth, nrh, rmin, rmax);
        xi0.set_delta(PhasePoint(0.0, 0.0));
        hyb.init_computation(0, xi0);
        xi0.free();
        allocated = true;
    }

    __host__ void release() {
        if (!allocated) return;
        switch (type) {
            case StateType::DISC: if (disc.amp) { disc.free(); disc.amp = nullptr; } break;
            case StateType::TOP:  if (top.amp)  { top.free();  top.amp  = nullptr; } break;
            case StateType::HYB:  if (hyb.amp)  { hyb.free();  hyb.amp  = nullptr; } break;
        }
        allocated = false;
    }
};

// ── Measurement record ──────────────────────────────────────────────────────

struct MeasRecord {
    static constexpr int MAX_MEASUREMENTS = 32;
    static constexpr int MAX_MEAS_DIM = 64;  // up to 6 qubits

    struct DiscMeas {
        double probs[MAX_MEAS_DIM];
        int    dim;
        bool   valid;
    };

    DiscMeas disc_results[MAX_MEASUREMENTS];
    int      count;

    // Verification results
    bool cocycle_ok;
    bool ufe_ok;

    MeasRecord() : count(0), cocycle_ok(true), ufe_ok(true) {}
};

// ── Platform VM ─────────────────────────────────────────────────────────────

struct PlatformVM {
    MachineSpec spec;
    StateSlot   regs[platform::MAX_REGS];
    MeasRecord  measurements;
    int         pc;
    int         tau;
    bool        halted;

    // Non-copyable (state slots hold raw pointers)
    PlatformVM(const PlatformVM&) = delete;
    PlatformVM& operator=(const PlatformVM&) = delete;

    __host__ explicit PlatformVM(const MachineSpec& s)
        : spec(s), pc(0), tau(0), halted(false) {}

    __host__ ~PlatformVM() {
        for (int i = 0; i < platform::MAX_REGS; ++i)
            regs[i].release();
    }

    // ── State allocation ────────────────────────────────────────────────────

    __host__ void alloc_state(int reg, StateType type) {
        switch (type) {
            case StateType::DISC:
                regs[reg].alloc_disc(spec.n_qubits);
                break;
            case StateType::TOP:
                regs[reg].alloc_top(spec.N_theta, spec.N_rho,
                                    spec.rho_min, spec.rho_max);
                break;
            case StateType::HYB:
                regs[reg].alloc_hyb(spec.n_qubits, spec.N_theta, spec.N_rho,
                                    spec.rho_min, spec.rho_max);
                break;
        }
    }

    // ── Lifting ─────────────────────────────────────────────────────────────

    // ι_disc: |ψ⟩ ↦ Σ_x α_x (|x⟩ ⊗ ξ₀)
    __host__ void exec_lift_disc_to_hyb(int dst, int src) {
        // Copy source amplitudes before potential overwrite (dst == src case)
        int dim_d = 1 << spec.n_qubits;
        C64* disc_amp = new C64[dim_d];
        memcpy(disc_amp, regs[src].disc.amp, dim_d * sizeof(C64));

        // Allocate destination as hybrid (frees old contents if dst == src)
        regs[dst].alloc_hyb(spec.n_qubits, spec.N_theta, spec.N_rho,
                            spec.rho_min, spec.rho_max);
        HybridState& h = regs[dst].hyb;

        // Build ξ₀
        TopologicalState xi0;
        xi0.init(spec.N_theta, spec.N_rho, spec.rho_min, spec.rho_max);
        xi0.set_delta(PhasePoint(0.0, 0.0));

        // Tensor: Σ_x α_x |x⟩ ⊗ ξ₀
        memset(h.amp, 0, h.total * sizeof(C64));
        for (int x = 0; x < dim_d; ++x) {
            for (int j = 0; j < h.dim_top; ++j) {
                h.amp[x * h.dim_top + j] = disc_amp[x] * xi0.amp[j];
            }
        }

        xi0.free();
        delete[] disc_amp;
    }

    // ι_top: ξ ↦ |0⟩ ⊗ ξ
    __host__ void exec_lift_top_to_hyb(int dst, int src) {
        int dim_t = spec.N_theta * spec.N_rho;
        C64* top_amp = new C64[dim_t];
        memcpy(top_amp, regs[src].top.amp, dim_t * sizeof(C64));

        regs[dst].alloc_hyb(spec.n_qubits, spec.N_theta, spec.N_rho,
                            spec.rho_min, spec.rho_max);
        HybridState& h = regs[dst].hyb;

        memset(h.amp, 0, h.total * sizeof(C64));
        for (int j = 0; j < h.dim_top; ++j) {
            h.amp[j] = top_amp[j];  // slot x=0
        }

        delete[] top_amp;
    }

    // ── Projection ──────────────────────────────────────────────────────────

    // π_disc: Ξ ↦ √{p(x)} |x⟩  (amplitude approximation of Born probs)
    __host__ void exec_project_disc(int dst, int src) {
        HybridState& h = regs[src].hyb;
        regs[dst].alloc_disc(spec.n_qubits);
        QubitState& d = regs[dst].disc;
        for (int x = 0; x < h.dim_disc; ++x) {
            d.amp[x] = C64(sqrt(h.prob_disc(x)), 0.0);
        }
    }

    // π_top: Ξ ↦ π_x(Ξ)
    __host__ void exec_project_top(int dst, int src, int x) {
        HybridState& h = regs[src].hyb;
        regs[dst].alloc_top(spec.N_theta, spec.N_rho,
                            spec.rho_min, spec.rho_max);
        h.project(x, regs[dst].top);
    }

    // ── Gate dispatch ───────────────────────────────────────────────────────

    __host__ void apply_gate(const platform::PlatformOp& op) {
        int reg = op.reg_dst;
        StateSlot& slot = regs[reg];

        if (slot.type == StateType::HYB) {
            // Full hybrid dispatch
            switch (op.opcode) {
                case platform::PlatformOpCode::APPLY_X:
                    hybrid_gates::apply_X_hyb(slot.hyb, op.bv1);
                    break;
                case platform::PlatformOpCode::APPLY_Z:
                    hybrid_gates::apply_Z_hyb(slot.hyb, op.bv1);
                    break;
                case platform::PlatformOpCode::APPLY_H:
                    hybrid_gates::apply_H_hyb(slot.hyb, op.iparam);
                    break;
                case platform::PlatformOpCode::APPLY_U:
                    hybrid_gates::apply_U_hyb(slot.hyb, op.param);
                    break;
                case platform::PlatformOpCode::APPLY_M:
                    hybrid_gates::apply_M_hyb(slot.hyb, op.iparam);
                    break;
                case platform::PlatformOpCode::APPLY_W:
                    hybrid_gates::apply_W_hyb(slot.hyb, op.bv1, op.bv2, op.mir);
                    break;
                case platform::PlatformOpCode::APPLY_P:
                    hybrid_gates::apply_P_hyb(slot.hyb, op.target);
                    break;
                default: break;
            }
        } else if (slot.type == StateType::DISC) {
            // Fast discrete-only path
            switch (op.opcode) {
                case platform::PlatformOpCode::APPLY_X:
                    discrete_gates::apply_X(slot.disc, op.bv1);
                    break;
                case platform::PlatformOpCode::APPLY_Z:
                    discrete_gates::apply_Z(slot.disc, op.bv1);
                    break;
                case platform::PlatformOpCode::APPLY_H: {
                    int qubit = op.iparam;
                    int dim = slot.disc.dim;
                    int mask = 1 << qubit;
                    double inv_sqrt2 = 1.0 / sqrt(2.0);
                    C64* tmp = new C64[dim];
                    memcpy(tmp, slot.disc.amp, dim * sizeof(C64));
                    for (int x = 0; x < dim; ++x) {
                        int partner = x ^ mask;
                        if (x < partner) {
                            C64 a0 = slot.disc.amp[x];
                            C64 a1 = slot.disc.amp[partner];
                            tmp[x]       = (a0 + a1) * inv_sqrt2;
                            tmp[partner] = (a0 - a1) * inv_sqrt2;
                        }
                    }
                    memcpy(slot.disc.amp, tmp, dim * sizeof(C64));
                    delete[] tmp;
                    break;
                }
                default: break;
            }
        } else if (slot.type == StateType::TOP) {
            // Topological-only path
            switch (op.opcode) {
                case platform::PlatformOpCode::APPLY_U:
                    topo_gates::apply_U(slot.top, op.param);
                    break;
                case platform::PlatformOpCode::APPLY_M:
                    topo_gates::apply_M(slot.top, op.iparam);
                    break;
                default: break;
            }
        }
    }

    // ── Measurement ─────────────────────────────────────────────────────────

    __host__ void exec_measure_disc(int reg) {
        StateSlot& slot = regs[reg];
        if (slot.type != StateType::HYB) return;

        int idx = measurements.count;
        if (idx >= MeasRecord::MAX_MEASUREMENTS) return;

        MeasRecord::DiscMeas& dm = measurements.disc_results[idx];
        dm.dim = slot.hyb.dim_disc;
        dm.valid = true;

        if (dm.dim > MeasRecord::MAX_MEAS_DIM) {
            dm.valid = false;
            return;
        }

        double total = 0.0;
        for (int x = 0; x < dm.dim; ++x) {
            dm.probs[x] = slot.hyb.prob_disc(x);
            total += dm.probs[x];
        }
        if (total > 0.0) {
            for (int x = 0; x < dm.dim; ++x)
                dm.probs[x] /= total;
        }
        measurements.count++;
    }

    __host__ void exec_measure_joint(int reg) {
        // Joint measurement: compute both discrete Born probs and store
        exec_measure_disc(reg);
    }

    // ── Verification ────────────────────────────────────────────────────────

    __host__ void exec_check_cocycle() {
        MirElement g1 = mir::g_phi(1.0);
        MirElement g2 = mir::g_phi(constants::PHI);
        MirElement g3 = mir::g_phi(constants::ELL);
        measurements.cocycle_ok = cocycle::verify_cocycle(g1, g2, g3);
    }

    __host__ void exec_check_ufe() {
        MirElement g1 = mir::g_phi(1.0);
        MirElement g2 = mir::g_phi(constants::PHI);
        BitVec u(1, 2), v(0, 2), up(0, 2), vp(1, 2);
        C64 ufe = cocycle::UFE_total(g1, u, v, g2, up, vp);
        measurements.ufe_ok = (ufe.norm2() < 1e-20);
    }

    // ── Execution engine ────────────────────────────────────────────────────

    __host__ void step(const platform::PlatformIR& ir) {
        if (halted || pc >= ir.count) {
            halted = true;
            return;
        }

        const platform::PlatformOp& op = ir.ops[pc];

        switch (op.opcode) {
            case platform::PlatformOpCode::ALLOC:
                alloc_state(op.reg_dst, op.output_type);
                break;

            case platform::PlatformOpCode::FREE:
                regs[op.reg_dst].release();
                break;

            case platform::PlatformOpCode::LIFT_DISC_TO_HYB:
                exec_lift_disc_to_hyb(op.reg_dst, op.reg_src);
                break;

            case platform::PlatformOpCode::LIFT_TOP_TO_HYB:
                exec_lift_top_to_hyb(op.reg_dst, op.reg_src);
                break;

            case platform::PlatformOpCode::PROJECT_DISC:
                exec_project_disc(op.reg_dst, op.reg_src);
                break;

            case platform::PlatformOpCode::PROJECT_TOP:
                exec_project_top(op.reg_dst, op.reg_src,
                                 static_cast<int>(op.target));
                break;

            case platform::PlatformOpCode::APPLY_X:
            case platform::PlatformOpCode::APPLY_Z:
            case platform::PlatformOpCode::APPLY_H:
            case platform::PlatformOpCode::APPLY_U:
            case platform::PlatformOpCode::APPLY_M:
            case platform::PlatformOpCode::APPLY_W:
            case platform::PlatformOpCode::APPLY_P:
                apply_gate(op);
                break;

            case platform::PlatformOpCode::MEASURE_DISC:
                exec_measure_disc(op.reg_dst);
                break;

            case platform::PlatformOpCode::MEASURE_JOINT:
                exec_measure_joint(op.reg_dst);
                break;

            case platform::PlatformOpCode::CHECK_COCYCLE:
                exec_check_cocycle();
                break;

            case platform::PlatformOpCode::CHECK_UFE:
                exec_check_ufe();
                break;

            // No-ops
            case platform::PlatformOpCode::BARRIER:
            case platform::PlatformOpCode::EMIT_SECTOR:
            case platform::PlatformOpCode::NOP:
            case platform::PlatformOpCode::LABEL:
            case platform::PlatformOpCode::BRANCH:
            case platform::PlatformOpCode::REPEAT:
                break;

            default:
                break;
        }

        pc++;
        tau++;
    }

    // Run the full program
    __host__ void run(const platform::PlatformIR& ir) {
        while (!halted && pc < ir.count) {
            step(ir);
        }
        halted = true;
    }

    // ── Result extraction ───────────────────────────────────────────────────

    // Get Born probability for a measurement outcome
    __host__ double prob(int measurement_idx, int outcome) const {
        if (measurement_idx >= 0 &&
            measurement_idx < measurements.count &&
            measurements.disc_results[measurement_idx].valid &&
            outcome >= 0 &&
            outcome < measurements.disc_results[measurement_idx].dim) {
            return measurements.disc_results[measurement_idx].probs[outcome];
        }
        return 0.0;
    }

    // Number of measurements recorded
    __host__ int num_measurements() const { return measurements.count; }

    // Time elapsed (number of steps executed)
    __host__ int time() const { return tau; }

    // Verification results
    __host__ bool cocycle_verified() const { return measurements.cocycle_ok; }
    __host__ bool ufe_verified()     const { return measurements.ufe_ok; }
};

} // namespace runtime
} // namespace topcomp
