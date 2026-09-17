// ============================================================================
// TopComp Layer 1: Formal Machine Specification — State Contract
// ============================================================================
// Defines the state type system for the hybrid compute platform:
//
//   Π_n^disc:       V_n = ℂ^{2^n}           — discrete qubit sector
//   Π_{ρ,A}^top:    X_A^top = C(M/Γ)        — topological phase space sector
//   Π_{ρ,A,n}^hyb:  V_n ⊗ X_A^top           — hybrid tensor sector
//
// Lifting maps (embedding lower sectors into higher):
//   ι_disc: V_n → V_n ⊗ X_A^top             |x⟩ ↦ |x⟩ ⊗ ξ₀
//   ι_top:  X_A^top → V_n ⊗ X_A^top         ξ ↦ |0⟩ ⊗ ξ
//
// Projection maps (extracting sector information):
//   π_disc: X_{A,n}^hyb → Prob(I_n)          Ξ ↦ {p(x) = ‖π_x(Ξ)‖²}
//   π_top:  X_{A,n}^hyb → X_A^top            Ξ ↦ π₀(Ξ)
//
// Cost model:
//   cost(X_u) = ‖u‖₀,  cost(Z_v) = ‖v‖₀
//   cost(U_t) = cost(M_m) = cost(P_x) = 1
//   cost(lift) = dim(target),  cost(project) = dim(top)
//
// The contract is: own the state ontology, own the lifting semantics,
// own the measurement boundaries.  Everything above targets this layer.
// ============================================================================
#pragma once
#include "hilbert.cuh"

namespace topcomp {
namespace contract {

// ── State type ontology ─────────────────────────────────────────────────────
// The three sectors of the doctrine
enum class StateType : int {
    DISC = 0,   // V_n = ℂ^{2^n}
    TOP  = 1,   // X_A^top = functions on M/Γ
    HYB  = 2,   // V_n ⊗ X_A^top
};

// ── State descriptor ────────────────────────────────────────────────────────
// Carries the dimensions and parameters needed to allocate a state
struct StateDescriptor {
    StateType type;
    int     n_qubits;   // meaningful for DISC and HYB
    int     N_theta;    // angular resolution (TOP, HYB)
    int     N_rho;      // radial resolution  (TOP, HYB)
    double  rho_min;
    double  rho_max;

    __host__ __device__ StateDescriptor()
        : type(StateType::DISC), n_qubits(1),
          N_theta(16), N_rho(8), rho_min(-5.0), rho_max(5.0) {}

    __host__ __device__ static StateDescriptor disc(int nq) {
        StateDescriptor d;
        d.type = StateType::DISC;
        d.n_qubits = nq;
        return d;
    }

    __host__ __device__ static StateDescriptor top(int nth, int nrh,
                                                    double rmin = -5.0,
                                                    double rmax = 5.0) {
        StateDescriptor d;
        d.type = StateType::TOP;
        d.N_theta = nth;
        d.N_rho   = nrh;
        d.rho_min = rmin;
        d.rho_max = rmax;
        return d;
    }

    __host__ __device__ static StateDescriptor hyb(int nq, int nth, int nrh,
                                                    double rmin = -5.0,
                                                    double rmax = 5.0) {
        StateDescriptor d;
        d.type = StateType::HYB;
        d.n_qubits = nq;
        d.N_theta  = nth;
        d.N_rho    = nrh;
        d.rho_min  = rmin;
        d.rho_max  = rmax;
        return d;
    }

    // Total dimension of the state space
    __host__ __device__ int dim() const {
        switch (type) {
            case StateType::DISC: return 1 << n_qubits;
            case StateType::TOP:  return N_theta * N_rho;
            case StateType::HYB:  return (1 << n_qubits) * N_theta * N_rho;
        }
        return 0;
    }

    // Discrete dimension only
    __host__ __device__ int dim_disc() const { return 1 << n_qubits; }

    // Topological dimension only
    __host__ __device__ int dim_top() const { return N_theta * N_rho; }
};

// ── Type compatibility predicates ───────────────────────────────────────────
// Which operations can target which state types

// Discrete ops (X_u, Z_v, H_q) can act on DISC or HYB
__host__ __device__ inline bool can_apply_disc_op(StateType t) {
    return t == StateType::DISC || t == StateType::HYB;
}

// Topological ops (U_t, M_m) can act on TOP or HYB
__host__ __device__ inline bool can_apply_top_op(StateType t) {
    return t == StateType::TOP || t == StateType::HYB;
}

// Hybrid ops (W_{u,v;g}, P_x, joint measure) require HYB
__host__ __device__ inline bool can_apply_hyb_op(StateType t) {
    return t == StateType::HYB;
}

// Can we lift from src to dst?
__host__ __device__ inline bool can_lift(StateType src, StateType dst) {
    if (src == dst) return true;
    if (dst == StateType::HYB) return true;  // anything lifts to HYB
    return false;
}

// What type results from applying a disc op to a state?
__host__ __device__ inline StateType result_of_disc_op(StateType input) {
    return input;  // disc ops preserve the state type
}

// What type results from applying a top op to a state?
__host__ __device__ inline StateType result_of_top_op(StateType input) {
    return input;
}

// ── Cost model ──────────────────────────────────────────────────────────────
struct CostModel {
    // State allocation
    __host__ __device__ static int alloc_cost(const StateDescriptor& d) {
        return d.dim();
    }

    // Lifting: disc → hyb requires tensoring with ξ₀
    __host__ __device__ static int lift_disc_cost(const StateDescriptor& target) {
        return target.N_theta * target.N_rho;
    }

    // Lifting: top → hyb requires expanding into 2^n slots
    __host__ __device__ static int lift_top_cost(const StateDescriptor& target) {
        return (1 << target.n_qubits) * target.N_theta * target.N_rho;
    }

    // Projection: hyb → disc traces over topo
    __host__ __device__ static int project_disc_cost(const StateDescriptor& src) {
        return src.N_theta * src.N_rho;
    }

    // Projection: hyb → top extracts one slice
    __host__ __device__ static int project_top_cost(const StateDescriptor& src) {
        return src.N_theta * src.N_rho;
    }

    // Joint measurement: full Born scan
    __host__ __device__ static int joint_measure_cost(const StateDescriptor& src) {
        return src.dim();
    }

    // Gate costs (from the reference)
    __host__ __device__ static int gate_cost_x(int weight) { return weight; }
    __host__ __device__ static int gate_cost_z(int weight) { return weight; }
    __host__ __device__ static int gate_cost_u() { return 1; }
    __host__ __device__ static int gate_cost_m() { return 1; }
    __host__ __device__ static int gate_cost_p() { return 1; }
    __host__ __device__ static int gate_cost_h() { return 1; }
    __host__ __device__ static int gate_cost_w(int u_wt, int v_wt) {
        return u_wt + v_wt + 1;
    }
};

// ── Machine specification ───────────────────────────────────────────────────
// Formal parameters for the HES machine:
//   M_{A,n}^HES = (X, G, P, C, δ, Init, Halt, time, space, cost)
struct MachineSpec {
    int     n_qubits;
    int     N_theta;
    int     N_rho;
    double  rho_min;
    double  rho_max;

    __host__ __device__ MachineSpec(int nq = 1, int nth = 16, int nrh = 8,
                                    double rmin = -5.0, double rmax = 5.0)
        : n_qubits(nq), N_theta(nth), N_rho(nrh),
          rho_min(rmin), rho_max(rmax) {}

    __host__ __device__ StateDescriptor disc_desc() const {
        return StateDescriptor::disc(n_qubits);
    }

    __host__ __device__ StateDescriptor top_desc() const {
        return StateDescriptor::top(N_theta, N_rho, rho_min, rho_max);
    }

    __host__ __device__ StateDescriptor hyb_desc() const {
        return StateDescriptor::hyb(n_qubits, N_theta, N_rho, rho_min, rho_max);
    }

    __host__ __device__ int disc_dim() const { return 1 << n_qubits; }
    __host__ __device__ int top_dim()  const { return N_theta * N_rho; }
    __host__ __device__ int hyb_dim()  const { return disc_dim() * top_dim(); }
};

// ── Lifting operations (pure functions on states) ───────────────────────────

// ι_disc: |ψ⟩ ↦ Σ_x ⟨x|ψ⟩ (|x⟩ ⊗ ξ₀)
// Embeds a discrete state into hybrid by tensoring with the topological
// ground state ξ₀ = δ(0,0)
__host__ inline void lift_disc_to_hyb(const QubitState& disc,
                                       HybridState& hyb,
                                       const MachineSpec& spec) {
    // Allocate ξ₀
    TopologicalState xi0;
    xi0.init(spec.N_theta, spec.N_rho, spec.rho_min, spec.rho_max);
    xi0.set_delta(PhasePoint(0.0, 0.0));

    memset(hyb.amp, 0, hyb.total * sizeof(C64));
    for (int x = 0; x < disc.dim; ++x) {
        for (int j = 0; j < hyb.dim_top; ++j) {
            hyb.amp[x * hyb.dim_top + j] = disc.amp[x] * xi0.amp[j];
        }
    }
    xi0.free();
}

// ι_top: ξ ↦ |0⟩ ⊗ ξ
// Embeds a topological state at the x=0 computational basis slot
__host__ inline void lift_top_to_hyb(const TopologicalState& top,
                                      HybridState& hyb) {
    memset(hyb.amp, 0, hyb.total * sizeof(C64));
    for (int j = 0; j < hyb.dim_top; ++j) {
        hyb.amp[j] = top.amp[j];  // slot x=0
    }
}

// π_disc: Ξ ↦ {p(x) = ‖π_x(Ξ)‖²}
// Extracts Born probabilities over the discrete sector
__host__ inline void project_hyb_to_disc_probs(const HybridState& hyb,
                                                 double* probs) {
    double total = 0.0;
    for (int x = 0; x < hyb.dim_disc; ++x) {
        probs[x] = hyb.prob_disc(x);
        total += probs[x];
    }
    if (total > 0.0) {
        for (int x = 0; x < hyb.dim_disc; ++x)
            probs[x] /= total;
    }
}

// π_top: Ξ ↦ π_x(Ξ) — extract topological slice at computational basis x
__host__ inline void project_hyb_to_top(const HybridState& hyb,
                                         TopologicalState& top,
                                         int x) {
    hyb.project(x, top);
}

} // namespace contract
} // namespace topcomp
