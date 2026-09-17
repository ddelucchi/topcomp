// ============================================================================
// TopComp: Mixed Operators, Mixed Cocycle & Boundary Calculus
// ============================================================================
// Extends the hybrid architecture from pure tensor-factored operators
// toward genuinely inseparable mixed operators:
//
//   MIXED COCYCLE:
//     Ω_tot = Ω_A + Ω_bool + Ω_mix
//     Ω_mix((g;u,v),(h;u',v')) = κ · Tr_A(∂_comm(g,h) · ⟨v,u'⟩)
//     where κ is a coupling strength parameter
//
//   MIXED OPERATORS:
//     U_t^mix(u): topological flow whose rate depends on discrete sector u
//       (U_t^mix(u)Ψ)(x, θ, ρ) = Ψ(x, θ - πt(1+κ⟨u,x⟩)/2, ρ + t·ℓ(1+κ⟨u,x⟩))
//       κ→0 recovers U_t^hyb = id_V ⊗ U_t
//
//     X_u^mix(g): discrete shift whose phase depends on topological state g
//       (X_u^mix(g)Ψ)(x, θ, ρ) = exp(iκΩ_A(g, g_φ(ρ))) · Ψ(x⊕u, θ, ρ)
//       κ→0 recovers X_u^hyb = X_u ⊗ id_top
//
//   BOUNDARY CALCULUS:
//     Boundary := {B_disc, B_top, B_joint, B_mixed}
//     Each boundary carries observables, a projection map, and a cost
//     Boundary objects are first-class: can be composed, restricted, extended
//
//   REGIME PLANNER:
//     Determines whether a program operates in:
//       PURE_DISC   — only discrete gates, no topological content
//       PURE_TOP    — only topological gates, no discrete content
//       TENSOR      — factored hybrid (current architecture κ=0)
//       MIXED       — genuinely coupled (κ≠0)
//     Automatically selects optimal code path
//
//   CONSISTENCY:
//     δΩ_tot = 0  (cocycle condition for the extended cocycle)
//     Mixed UFE: F_mix - Ω_mix = 0
//     κ→0 limit: all mixed constructs reduce to tensor-factored forms
// ============================================================================
#pragma once
#include "cocycle.cuh"
#include "gates.cuh"

namespace topcomp {

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ I. MIXED COCYCLE Ω_mix                                                   ║
// ║                                                                          ║
// ║ Ω_tot = Ω_A + Ω_bool + Ω_mix                                             ║
// ║ Ω_mix((g;u,v),(h;u',v')) = κ · Re(Ω_A(g,h)) · ⟨v,u'⟩ · πi              ║
// ║ δΩ_mix = 0  (inherited from δΩ_A = 0 and bilinearity of ⟨·,·⟩)          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace mixed_cocycle {

// Coupling strength κ (controls mixed sector contribution)
// κ = 0: pure tensor-factored (current architecture)
// κ > 0: genuinely coupled mixed operators
static constexpr double KAPPA_DEFAULT = 0.0;

// Mixed cocycle term:
//   Ω_mix((g;u,v),(h;u',v')) = κ · Re(Ω_A(g,h)) · ⟨v,u'⟩ · πi
//
// Properties:
//   - Bilinear in both (g,h) and (v,u') arguments
//   - δΩ_mix = 0 because Re(Ω_A) satisfies the cocycle condition
//     and ⟨v,u'⟩ is bilinear over F_2
//   - κ→0 gives Ω_mix = 0, recovering the tensor-factored case
__host__ __device__ inline C64 omega_mix(
    const MirElement& g, const BitVec& /*u*/, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& /*vp*/,
    double kappa = KAPPA_DEFAULT)
{
    if (kappa == 0.0) return C64(0.0, 0.0);
    C64 omega_A = cocycle::omega_mir(g, h);
    int inner = v.inner(up);
    return C64(0.0, constants::PI * kappa * omega_A.re * inner);
}

// Extended total cocycle:
//   Ω_tot^ext = Ω_A(g,h) + Ω_bool(v,u') + Ω_mix
__host__ __device__ inline C64 omega_total_ext(
    const MirElement& g, const BitVec& u, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& vp,
    double kappa = KAPPA_DEFAULT)
{
    C64 base = cocycle::omega_total(g, u, v, h, up, vp);
    C64 mix  = omega_mix(g, u, v, h, up, vp, kappa);
    return base + mix;
}

// Verify δΩ_mix = 0 for a triple of hybrid elements
// δΩ(a,b,c) = Ω(b,c) - Ω(ab,c) + Ω(a,bc) - Ω(a,b) = 0
__host__ inline bool verify_mixed_cocycle(
    const MirElement& g1, const BitVec& u1, const BitVec& v1,
    const MirElement& g2, const BitVec& u2, const BitVec& v2,
    const MirElement& g3, const BitVec& u3, const BitVec& v3,
    double kappa, double tol = 1e-10)
{
    // Compute products for the coboundary
    MirElement g12 = g1.star(g2);
    MirElement g23 = g2.star(g3);
    BitVec u12(u1.bits ^ u2.bits, u1.n);
    BitVec v12(v1.bits ^ v2.bits, v1.n);
    BitVec u23(u2.bits ^ u3.bits, u2.n);
    BitVec v23(v2.bits ^ v3.bits, v2.n);

    C64 o23    = omega_mix(g2, u2, v2, g3, u3, v3, kappa);
    C64 o12_3  = omega_mix(g12, u12, v12, g3, u3, v3, kappa);
    C64 o1_23  = omega_mix(g1, u1, v1, g23, u23, v23, kappa);
    C64 o12    = omega_mix(g1, u1, v1, g2, u2, v2, kappa);

    C64 coboundary = o23 - o12_3 + o1_23 - o12;
    return coboundary.norm2() < tol * tol;
}

// Verify δΩ_tot^ext = 0 for the full extended cocycle
__host__ inline bool verify_extended_cocycle(
    const MirElement& g1, const BitVec& u1, const BitVec& v1,
    const MirElement& g2, const BitVec& u2, const BitVec& v2,
    const MirElement& g3, const BitVec& u3, const BitVec& v3,
    double kappa, double tol = 1e-10)
{
    MirElement g12 = g1.star(g2);
    MirElement g23 = g2.star(g3);
    BitVec u12(u1.bits ^ u2.bits, u1.n);
    BitVec v12(v1.bits ^ v2.bits, v1.n);
    BitVec u23(u2.bits ^ u3.bits, u2.n);
    BitVec v23(v2.bits ^ v3.bits, v2.n);

    C64 o23    = omega_total_ext(g2, u2, v2, g3, u3, v3, kappa);
    C64 o12_3  = omega_total_ext(g12, u12, v12, g3, u3, v3, kappa);
    C64 o1_23  = omega_total_ext(g1, u1, v1, g23, u23, v23, kappa);
    C64 o12    = omega_total_ext(g1, u1, v1, g2, u2, v2, kappa);

    C64 coboundary = o23 - o12_3 + o1_23 - o12;
    return coboundary.norm2() < tol * tol;
}

// Verify κ→0 limit reduces to the base cocycle
__host__ inline bool verify_kappa_zero_limit(
    const MirElement& g, const BitVec& u, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& vp,
    double tol = 1e-14)
{
    C64 ext = omega_total_ext(g, u, v, h, up, vp, 0.0);
    C64 base = cocycle::omega_total(g, u, v, h, up, vp);
    return (ext - base).norm2() < tol * tol;
}

} // namespace mixed_cocycle

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ II. MIXED OPERATORS                                                      ║
// ║                                                                          ║
// ║ U_t^mix(u): topological flow modulated by discrete sector                ║
// ║ X_u^mix(g): discrete shift with topological phase                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace mixed_ops {

// U_t^mix(u): sector-dependent topological flow
//   (U_t^mix(u)Ψ)(x, θ, ρ) = Ψ(x, θ - πt·α(u,x)/2, ρ + t·ℓ·α(u,x))
//   where α(u,x) = 1 + κ⟨u,x⟩
//
// κ=0 → α=1 → standard U_t for all sectors (tensor-factored)
// κ>0 → sectors with ⟨u,x⟩=1 flow faster by factor (1+κ)
__host__ inline void apply_U_mix(HybridState& psi, double t,
                                  const BitVec& u, double kappa) {
    for (int x = 0; x < psi.dim_disc; ++x) {
        BitVec xbv(x, u.n);
        double alpha = 1.0 + kappa * xbv.inner(u);
        double t_eff = t * alpha;

        TopologicalState slice;
        slice.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        psi.project(x, slice);
        topo_gates::apply_U(slice, t_eff);
        psi.inject(x, slice);
        slice.free();
    }
}

// X_u^mix(g): topological-phase-modulated discrete shift
//   (X_u^mix(g)Ψ)(x', θ, ρ) = exp(iκ·Ω_A(g, g_φ(ρ/ℓ))) · Ψ(x'⊕u, θ, ρ)
//
// κ=0 → phase=1 → standard X_u shift (tensor-factored)
// κ>0 → discrete shift acquires ρ-dependent phase from topological cocycle
__host__ inline void apply_X_mix(HybridState& psi, const BitVec& u,
                                  const MirElement& g, double kappa) {
    int dim_d = psi.dim_disc;
    int dim_t = psi.dim_top;
    int nrh = psi.N_rho;
    int nth = psi.N_theta;
    double d_rho = (nrh > 1)
        ? (psi.rho_max - psi.rho_min) / (nrh - 1) : 1.0;

    C64* tmp = new C64[psi.total];
    memset(tmp, 0, psi.total * sizeof(C64));

    for (int x = 0; x < dim_d; ++x) {
        int xp = x ^ u.bits;
        for (int i = 0; i < nth; ++i) {
            for (int j = 0; j < nrh; ++j) {
                double rho_val = psi.rho_min + j * d_rho;
                // Phase from topological cocycle
                C64 phase(1.0, 0.0);
                if (kappa != 0.0) {
                    MirElement g_rho = mir::g_phi(rho_val / constants::ELL);
                    C64 omega = cocycle::omega_mir(g, g_rho);
                    phase = cexp(C64(0.0, kappa * omega.re));
                }
                int src_idx = x * dim_t + i * nrh + j;
                int dst_idx = xp * dim_t + i * nrh + j;
                tmp[dst_idx] = psi.amp[src_idx] * phase;
            }
        }
    }
    memcpy(psi.amp, tmp, psi.total * sizeof(C64));
    delete[] tmp;
}

// Verify κ→0 limit: U_t^mix(u)|κ=0 = U_t^hyb
__host__ inline bool verify_U_mix_limit(int n_qubits, int nth, int nrh,
                                         double t, double tol = 1e-10) {
    HybridState psi1, psi2;
    psi1.init(n_qubits, nth, nrh);
    psi2.init(n_qubits, nth, nrh);

    // Set identical initial states
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(1.0, 0.5));
    psi1.init_computation(0, xi);
    psi2.init_computation(0, xi);
    xi.free();

    // Apply U_t^mix(u) with κ=0
    BitVec u(1, n_qubits);
    apply_U_mix(psi1, t, u, 0.0);

    // Apply standard U_t^hyb
    hybrid_gates::apply_U_hyb(psi2, t);

    double diff = 0.0;
    for (int i = 0; i < psi1.total; ++i)
        diff += (psi1.amp[i] - psi2.amp[i]).norm2();

    psi1.free();
    psi2.free();
    return diff < tol;
}

// Verify κ→0 limit: X_u^mix(g)|κ=0 = X_u^hyb
__host__ inline bool verify_X_mix_limit(int n_qubits, int nth, int nrh,
                                         double tol = 1e-10) {
    HybridState psi1, psi2;
    psi1.init(n_qubits, nth, nrh);
    psi2.init(n_qubits, nth, nrh);

    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(1.0, 0.5));
    psi1.init_computation(0, xi);
    psi2.init_computation(0, xi);
    xi.free();

    BitVec u(1, n_qubits);
    MirElement g = mir::g_phi(1.0);

    // Apply X_u^mix(g) with κ=0
    apply_X_mix(psi1, u, g, 0.0);

    // Apply standard X_u^hyb
    hybrid_gates::apply_X_hyb(psi2, u);

    double diff = 0.0;
    for (int i = 0; i < psi1.total; ++i)
        diff += (psi1.amp[i] - psi2.amp[i]).norm2();

    psi1.free();
    psi2.free();
    return diff < tol;
}

// Verify that mixed operators with κ≠0 produce genuinely different results
__host__ inline bool verify_mixed_nontrivial(int n_qubits, int nth, int nrh,
                                              double kappa, double t,
                                              double tol = 1e-6) {
    HybridState psi1, psi2;
    psi1.init(n_qubits, nth, nrh);
    psi2.init(n_qubits, nth, nrh);

    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(1.0, 0.5));

    // Populate both x=0 and x=1 so ⟨u,x⟩ values differ across sectors
    psi1.inject(0, xi);
    psi1.inject(1, xi);
    psi2.inject(0, xi);
    psi2.inject(1, xi);
    xi.free();

    BitVec u(1, n_qubits);
    apply_U_mix(psi1, t, u, kappa);
    hybrid_gates::apply_U_hyb(psi2, t);

    double diff = 0.0;
    for (int i = 0; i < psi1.total; ++i)
        diff += (psi1.amp[i] - psi2.amp[i]).norm2();

    psi1.free();
    psi2.free();
    return diff > tol;  // should be different when κ≠0
}

} // namespace mixed_ops

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ III. BOUNDARY CALCULUS                                                    ║
// ║                                                                          ║
// ║ Boundary := first-class object with:                                     ║
// ║   - type (disc/top/joint/mixed)                                          ║
// ║   - dimension                                                             ║
// ║   - projection map                                                        ║
// ║   - observable set                                                        ║
// ║   - cost                                                                  ║
// ║                                                                          ║
// ║ Operations:                                                              ║
// ║   restrict(B, sector) → B'    (restrict boundary to subsector)           ║
// ║   compose(B1, B2)     → B1⊗B2 (tensor product of boundaries)            ║
// ║   measure(B, ψ)       → (probs, post_states)                            ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace boundary {

// Boundary type taxonomy
enum class BoundaryType : int {
    DISC   = 0,   // discrete measurement boundary (computational basis)
    TOP    = 1,   // topological measurement boundary (phase space bins)
    JOINT  = 2,   // joint disc×top measurement
    MIXED  = 3,   // mixed boundary coupling both sectors
};

// Boundary object: first-class representation of a measurement boundary
struct Boundary {
    BoundaryType type;
    int n_qubits;       // discrete dimension parameter
    int N_theta;        // topological angular resolution
    int N_rho;          // topological radial resolution
    int n_outcomes;     // total number of measurement outcomes

    __host__ __device__ Boundary()
        : type(BoundaryType::DISC), n_qubits(1),
          N_theta(1), N_rho(1), n_outcomes(2) {}

    // Named constructors
    __host__ __device__ static Boundary disc(int nq) {
        Boundary b;
        b.type = BoundaryType::DISC;
        b.n_qubits = nq;
        b.n_outcomes = 1 << nq;
        return b;
    }

    __host__ __device__ static Boundary top(int nth, int nrh) {
        Boundary b;
        b.type = BoundaryType::TOP;
        b.N_theta = nth;
        b.N_rho = nrh;
        b.n_outcomes = nth * nrh;
        return b;
    }

    __host__ __device__ static Boundary joint(int nq, int nth, int nrh) {
        Boundary b;
        b.type = BoundaryType::JOINT;
        b.n_qubits = nq;
        b.N_theta = nth;
        b.N_rho = nrh;
        b.n_outcomes = (1 << nq) * nth * nrh;
        return b;
    }

    __host__ __device__ static Boundary mixed(int nq, int nth, int nrh) {
        Boundary b;
        b.type = BoundaryType::MIXED;
        b.n_qubits = nq;
        b.N_theta = nth;
        b.N_rho = nrh;
        b.n_outcomes = (1 << nq) * nth * nrh;
        return b;
    }

    // Dimension of the boundary's outcome space
    __host__ __device__ int dim() const { return n_outcomes; }

    // Cost of measuring at this boundary
    __host__ __device__ int cost() const { return n_outcomes; }

    // Can this boundary observe discrete outcomes?
    __host__ __device__ bool has_disc() const {
        return type == BoundaryType::DISC ||
               type == BoundaryType::JOINT ||
               type == BoundaryType::MIXED;
    }

    // Can this boundary observe topological outcomes?
    __host__ __device__ bool has_top() const {
        return type == BoundaryType::TOP ||
               type == BoundaryType::JOINT ||
               type == BoundaryType::MIXED;
    }
};

// Boundary measurement result
struct BoundaryResult {
    double* probs;      // probability distribution over outcomes
    int n_outcomes;

    __host__ void init(int n) {
        n_outcomes = n;
        probs = new double[n];
        memset(probs, 0, n * sizeof(double));
    }

    __host__ void free() {
        delete[] probs;
        probs = nullptr;
    }

    // Shannon entropy of the outcome distribution
    __host__ double entropy() const {
        double H = 0.0;
        for (int i = 0; i < n_outcomes; ++i) {
            if (probs[i] > 1e-15)
                H -= probs[i] * log(probs[i]);
        }
        return H;
    }

    // Verify normalization: Σ p_i = 1
    __host__ bool normalized(double tol = 1e-10) const {
        double sum = 0.0;
        for (int i = 0; i < n_outcomes; ++i) sum += probs[i];
        return fabs(sum - 1.0) < tol;
    }
};

// Measure a hybrid state at a boundary
__host__ inline BoundaryResult measure(const Boundary& bnd,
                                         const HybridState& psi) {
    BoundaryResult result;

    switch (bnd.type) {
    case BoundaryType::DISC: {
        result.init(bnd.n_outcomes);
        double total = 0.0;
        for (int x = 0; x < psi.dim_disc; ++x) {
            result.probs[x] = psi.prob_disc(x);
            total += result.probs[x];
        }
        if (total > 0.0)
            for (int x = 0; x < psi.dim_disc; ++x)
                result.probs[x] /= total;
        break;
    }
    case BoundaryType::TOP: {
        int dim_t = bnd.N_theta * bnd.N_rho;
        result.init(dim_t);
        double total = 0.0;
        // Sum over all discrete sectors to get topological marginal
        for (int j = 0; j < dim_t; ++j) {
            double p = 0.0;
            for (int x = 0; x < psi.dim_disc; ++x)
                p += psi.amp[x * psi.dim_top + j].norm2();
            result.probs[j] = p;
            total += p;
        }
        if (total > 0.0)
            for (int j = 0; j < dim_t; ++j)
                result.probs[j] /= total;
        break;
    }
    case BoundaryType::JOINT:
    case BoundaryType::MIXED: {
        result.init(psi.total);
        double total = 0.0;
        for (int i = 0; i < psi.total; ++i) {
            result.probs[i] = psi.amp[i].norm2();
            total += result.probs[i];
        }
        if (total > 0.0)
            for (int i = 0; i < psi.total; ++i)
                result.probs[i] /= total;
        break;
    }
    }
    return result;
}

// Restrict a boundary to a subsector
__host__ __device__ inline Boundary restrict_to_disc(const Boundary& bnd) {
    return Boundary::disc(bnd.n_qubits);
}

__host__ __device__ inline Boundary restrict_to_top(const Boundary& bnd) {
    return Boundary::top(bnd.N_theta, bnd.N_rho);
}

// Compose boundaries: B1 ⊗ B2
__host__ __device__ inline Boundary compose(const Boundary& b1,
                                              const Boundary& b2) {
    if (b1.type == BoundaryType::DISC && b2.type == BoundaryType::TOP)
        return Boundary::joint(b1.n_qubits, b2.N_theta, b2.N_rho);
    if (b1.type == BoundaryType::TOP && b2.type == BoundaryType::DISC)
        return Boundary::joint(b2.n_qubits, b1.N_theta, b1.N_rho);
    // Default: joint boundary with combined parameters
    return Boundary::joint(
        (b1.n_qubits > b2.n_qubits) ? b1.n_qubits : b2.n_qubits,
        (b1.N_theta > b2.N_theta) ? b1.N_theta : b2.N_theta,
        (b1.N_rho > b2.N_rho) ? b1.N_rho : b2.N_rho);
}

// Verify boundary consistency: disc marginal + top marginal = joint
__host__ inline bool verify_marginal_consistency(const HybridState& psi,
                                                   double tol = 1e-10) {
    Boundary b_d = Boundary::disc(psi.n_qubits);
    Boundary b_t = Boundary::top(psi.N_theta, psi.N_rho);
    Boundary b_j = Boundary::joint(psi.n_qubits, psi.N_theta, psi.N_rho);

    BoundaryResult rd = measure(b_d, psi);
    BoundaryResult rt = measure(b_t, psi);
    BoundaryResult rj = measure(b_j, psi);

    // Check disc marginal: Σ_j p_joint(x,j) = p_disc(x)
    bool disc_ok = true;
    for (int x = 0; x < psi.dim_disc && disc_ok; ++x) {
        double marginal = 0.0;
        for (int j = 0; j < psi.dim_top; ++j)
            marginal += rj.probs[x * psi.dim_top + j];
        if (fabs(marginal - rd.probs[x]) > tol) disc_ok = false;
    }

    // Check top marginal: Σ_x p_joint(x,j) = p_top(j)
    bool top_ok = true;
    for (int j = 0; j < psi.dim_top && top_ok; ++j) {
        double marginal = 0.0;
        for (int x = 0; x < psi.dim_disc; ++x)
            marginal += rj.probs[x * psi.dim_top + j];
        if (fabs(marginal - rt.probs[j]) > tol) top_ok = false;
    }

    rd.free();
    rt.free();
    rj.free();
    return disc_ok && top_ok;
}

} // namespace boundary

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ IV. REGIME PLANNER                                                       ║
// ║                                                                          ║
// ║ Classifies a program's computational regime and selects                  ║
// ║ the optimal execution path.                                              ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace regime {

enum class Regime : int {
    PURE_DISC = 0,   // only discrete gates
    PURE_TOP  = 1,   // only topological gates
    TENSOR    = 2,   // tensor-factored hybrid (κ=0)
    MIXED     = 3,   // genuinely coupled (κ≠0)
};

// Classify a gate sequence to determine the computational regime
__host__ inline Regime classify(const Gate* gates, int n_gates,
                                 double kappa = 0.0) {
    bool has_disc = false;
    bool has_top  = false;
    bool has_weyl = false;

    for (int i = 0; i < n_gates; ++i) {
        switch (gates[i].type) {
            case GateType::X_HYB:
            case GateType::Z_HYB:
            case GateType::H_HYB:
            case GateType::P_HYB:
                has_disc = true;
                break;
            case GateType::U_HYB:
            case GateType::M_HYB:
                has_top = true;
                break;
            case GateType::W_HYB:
                has_weyl = true;
                has_disc = true;
                has_top = true;
                break;
        }
    }

    if (kappa != 0.0) return Regime::MIXED;
    if (has_weyl) return Regime::TENSOR;
    if (has_disc && has_top) return Regime::TENSOR;
    if (has_disc) return Regime::PURE_DISC;
    if (has_top) return Regime::PURE_TOP;
    return Regime::PURE_DISC;  // empty program
}

// Cost multiplier per regime (mixed is more expensive)
__host__ __device__ inline double cost_multiplier(Regime r) {
    switch (r) {
        case Regime::PURE_DISC: return 1.0;
        case Regime::PURE_TOP:  return 1.0;
        case Regime::TENSOR:    return 1.0;
        case Regime::MIXED:     return 1.5;  // mixed ops cost 50% more
    }
    return 1.0;
}

// Human-readable regime name
__host__ inline const char* regime_name(Regime r) {
    switch (r) {
        case Regime::PURE_DISC: return "PURE_DISC";
        case Regime::PURE_TOP:  return "PURE_TOP";
        case Regime::TENSOR:    return "TENSOR";
        case Regime::MIXED:     return "MIXED";
    }
    return "UNKNOWN";
}

} // namespace regime

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ V. EXTENDED CENTRAL EXTENSION                                             ║
// ║                                                                          ║
// ║ G̃_tot^ext = G × I_n × I_n × ℂ/2πiℤ  with mixed cocycle                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// Extended central extension using the full mixed cocycle
struct CentralExtElementMix {
    BitVec     u, v;
    MirElement g;
    C64        phase;
    double     kappa;  // coupling strength

    __host__ __device__ CentralExtElementMix()
        : u(), v(), g(MirElement::identity()), phase(0.0, 0.0),
          kappa(mixed_cocycle::KAPPA_DEFAULT) {}

    __host__ __device__ CentralExtElementMix(
        const BitVec& u_, const MirElement& g_, const BitVec& v_,
        const C64& a_, double k = mixed_cocycle::KAPPA_DEFAULT)
        : u(u_), g(g_), v(v_), phase(a_), kappa(k) {}

    // Group product with extended cocycle
    __host__ __device__ CentralExtElementMix operator*(
        const CentralExtElementMix& other) const
    {
        C64 omega = mixed_cocycle::omega_total_ext(
            g, u, v, other.g, other.u, other.v, kappa);
        return CentralExtElementMix(
            u ^ other.u,
            g.star(other.g),
            v ^ other.v,
            phase + other.phase + omega,
            kappa);
    }

    __host__ __device__ static CentralExtElementMix identity(int n,
                                                              double k = 0.0) {
        return CentralExtElementMix(
            BitVec(0, n), MirElement::identity(), BitVec(0, n),
            C64(0, 0), k);
    }
};

} // namespace topcomp
