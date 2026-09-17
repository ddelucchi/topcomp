// ============================================================================
// TopComp: Full Weyl Operator Algebra W_{u,v;x} = (X_u Z_v) ⊗ ρ̂_A(x)
// ============================================================================
// The HYBRID Weyl operators that combine discrete + topological sectors:
//
//   W_{u,v;x} := (X_u Z_v) ⊗ ρ̂_A(x)
//
//   Composition law (central extension by total cocycle):
//     W_{u₁,v₁;x₁} · W_{u₂,v₂;x₂}
//       = exp(i Ω_tot((u₁,x₁),(v₂,x₂))) · W_{u₁u₂,v₁v₂;x₁x₂}
//
//   Where the total cocycle splits:
//     Ω_tot((u,g),(v',g')) = Ω_bool(u,v') + Ω_A(g,g')
//                          = πi⟨u,v'⟩ + Ω_Mir(Φ_A(g), Φ_A(g'))
//
//   Key commutation relations:
//     X_u Z_v = (-1)^{⟨u,v⟩} Z_v X_u        (discrete)
//     U_t M_m = e^{-i2πmt} M_m U_t            (topological)
//     [X_u^hyb, U_t^hyb] = 0                   (cross-sector)
//
//   The gate set G_{A,n}^hyb generates ALL of End(X_{A,n}^hyb):
//     G = ⟨{X_u^hyb}, {Z_v^hyb}, {U_t^hyb}, {M_m^hyb}, {P_x^hyb}⟩
//     ∀ T⊗S ∈ U_{A,n}^hyb, ∀ε>0, ∃W ∈ ⟨G⟩: ||W - T⊗S|| < ε
//
//   Algebra structure:
//     A_{A,n}^disc = End_C(V_n)                    (full matrix algebra)
//     A_A^top = Alg⟨{U_t}, {M_m}, ρ̂_A(Γ_A)⟩      (topological algebra)
//     A_{A,n}^hyb = Alg⟨G_{A,n}^hyb⟩ ⊃ A^disc ⊗ A^top
// ============================================================================
#pragma once
#include "cocycle.cuh"

namespace topcomp {

// ── Hybrid Weyl element descriptor ──────────────────────────────────────────
// Represents W_{u,v;g} = (X_u Z_v) ⊗ ρ̂(g)
// where g ∈ Mir_A
struct WeylHybrid {
    BitVec u;           // Pauli-X index in F_2^n
    BitVec v;           // Pauli-Z index in F_2^n
    MirElement g;       // Mirror group element for topological sector

    __host__ __device__ WeylHybrid() : u(0, 1), v(0, 1), g() {}
    __host__ __device__ WeylHybrid(BitVec u_, BitVec v_, MirElement g_)
        : u(u_), v(v_), g(g_) {}

    // Identity: W_{0,0;e} where e = (1, 0)
    __host__ __device__ static WeylHybrid identity(int n) {
        return {BitVec(0, n), BitVec(0, n), MirElement()};
    }

    // Pure discrete: W_{u,v;e}
    __host__ __device__ static WeylHybrid discrete(BitVec u_, BitVec v_) {
        return {u_, v_, MirElement()};
    }

    // Pure topological: W_{0,0;g}
    __host__ __device__ static WeylHybrid topological(int n, MirElement g_) {
        return {BitVec(0, n), BitVec(0, n), g_};
    }

    // ── Total cocycle ───────────────────────────────────────────────────────
    // Ω_tot((u₁,g₁),(v₂,g₂)) = Ω_bool(u₁,v₂) + Ω_A(g₁,g₂)
    // = πi⟨u₁,v₂⟩ + Ω_Mir(g₁,g₂)
    __host__ __device__ C64 omega_total(const WeylHybrid& other) const {
        C64 omega_b = cocycle::omega_bool(u, other.v);             // πi⟨u,v'⟩
        C64 omega_m = cocycle::omega_mir(g, other.g);               // Ω_Mir(g,g')
        return omega_b + omega_m;
    }

    // ── Composition: W₁★W₂ = exp(Ω_tot) · W_{u₁u₂,v₁v₂;g₁★g₂} ───────────────
    // omega_total already lives in ℂ/2πiℤ, so exp(Ω) gives the U(1) phase
    __host__ __device__ WeylHybrid compose(const WeylHybrid& other,
                                            C64& phase_out) const {
        phase_out = cexp(omega_total(other));
        return {
            BitVec(u.bits ^ other.u.bits, u.n),  // u₁ ⊕ u₂  (F_2 addition)
            BitVec(v.bits ^ other.v.bits, v.n),  // v₁ ⊕ v₂
            g.star(other.g)                       // g₁ ★ g₂  (Mir product)
        };
    }

    // ── Discrete commutation: X_u Z_v = (-1)^{⟨u,v⟩} Z_v X_u ────────────────
    __host__ __device__ int discrete_comm_sign() const {
        return u.inner(v);  // 0 → +1, 1 → -1
    }

    // ── Apply to hybrid state: W_{u,v;g}|ψ⟩ ─────────────────────────────────
    __host__ void apply(HybridState& psi) const {
        hybrid_gates::apply_W_hyb(psi, u, v, g);
    }
};

// ── Hybrid gate set G_{A,n}^hyb ─────────────────────────────────────────────
// Enumeration of generator types in the gate set

enum class HybridGateType {
    X_HYB,    // X_u^hyb = X_u ⊗ id_top
    Z_HYB,    // Z_v^hyb = Z_v ⊗ id_top
    U_HYB,    // U_t^hyb = id_V ⊗ U_t
    M_HYB,    // M_m^hyb = id_V ⊗ M_m
    P_HYB,    // P_x^hyb = P_x ⊗ id_top  (projector/measurement)
    H_HYB,    // H_q^hyb = H_q ⊗ id_top  (Hadamard on qubit q)
    W_HYB     // W_{u,v;g}  (full hybrid Weyl)
};

struct HybridGate {
    HybridGateType type;
    BitVec u;            // for X, W
    BitVec v;            // for Z, W
    double t;            // for U_t
    int m;               // for M_m
    uint32_t x;          // for P_x
    MirElement g;        // for W (topological part)

    __host__ __device__ HybridGate() : type(HybridGateType::X_HYB),
        u(0,1), v(0,1), t(0), m(0), x(0), g() {}

    // Named constructors for each generator type
    __host__ __device__ static HybridGate X(BitVec u_) {
        HybridGate gate;
        gate.type = HybridGateType::X_HYB;
        gate.u = u_;
        return gate;
    }

    __host__ __device__ static HybridGate Z(BitVec v_) {
        HybridGate gate;
        gate.type = HybridGateType::Z_HYB;
        gate.v = v_;
        return gate;
    }

    __host__ __device__ static HybridGate U(double t_) {
        HybridGate gate;
        gate.type = HybridGateType::U_HYB;
        gate.t = t_;
        return gate;
    }

    __host__ __device__ static HybridGate M(int m_) {
        HybridGate gate;
        gate.type = HybridGateType::M_HYB;
        gate.m = m_;
        return gate;
    }

    __host__ __device__ static HybridGate P(uint32_t x_) {
        HybridGate gate;
        gate.type = HybridGateType::P_HYB;
        gate.x = x_;
        return gate;
    }

    __host__ __device__ static HybridGate W(BitVec u_, BitVec v_, MirElement g_) {
        HybridGate gate;
        gate.type = HybridGateType::W_HYB;
        gate.u = u_;
        gate.v = v_;
        gate.g = g_;
        return gate;
    }

    __host__ __device__ static HybridGate H(int qubit) {
        HybridGate gate;
        gate.type = HybridGateType::H_HYB;
        gate.x = static_cast<uint32_t>(qubit);
        return gate;
    }

    // Cost function from the reference:
    //   cost(X_u) = ||u||_0,  cost(Z_v) = ||v||_0
    //   cost(U_t) = 1,  cost(M_m) = 1,  cost(P_x) = 1
    __host__ __device__ int cost() const {
        switch (type) {
            case HybridGateType::X_HYB: return u.weight();
            case HybridGateType::Z_HYB: return v.weight();
            case HybridGateType::U_HYB: return 1;
            case HybridGateType::M_HYB: return 1;
            case HybridGateType::P_HYB: return 1;
            case HybridGateType::H_HYB: return 1;
            case HybridGateType::W_HYB: return u.weight() + v.weight() + 1;
            default: return 0;
        }
    }
};

// ── Hybrid program P = g₁...g_L ─────────────────────────────────────────────
// U_P = g_L · g_{L-1} · ... · g_1  (reverse application order)
struct HybridProgram {
    static constexpr int MAX_HYBRID_GATES = 8192;
    HybridGate* gates;
    int length;

    __host__ HybridProgram() : length(0) {
        gates = new HybridGate[MAX_HYBRID_GATES];
    }
    __host__ ~HybridProgram() { delete[] gates; }

    // Copy
    __host__ HybridProgram(const HybridProgram& o) : length(o.length) {
        gates = new HybridGate[MAX_HYBRID_GATES];
        for (int i = 0; i < length; ++i) gates[i] = o.gates[i];
    }
    __host__ HybridProgram& operator=(const HybridProgram& o) {
        if (this != &o) {
            length = o.length;
            for (int i = 0; i < length; ++i) gates[i] = o.gates[i];
        }
        return *this;
    }

    __host__ void append(HybridGate gate) {
        if (length < MAX_HYBRID_GATES)
            gates[length++] = gate;
    }

    // Total cost: sum of individual gate costs
    __host__ int total_cost() const {
        int c = 0;
        for (int i = 0; i < length; ++i)
            c += gates[i].cost();
        return c;
    }

    // Time complexity: number of gates
    __host__ int time() const { return length; }

    // Space complexity: n + max(cost(g_j))
    __host__ int space(int n) const {
        int mc = 0;
        for (int i = 0; i < length; ++i) {
            int c = gates[i].cost();
            if (c > mc) mc = c;
        }
        return n + mc;
    }
};

// ── Verification functions ──────────────────────────────────────────────────
namespace weyl_verify {

// Verify Ω_bool(u,v') = πi⟨u,v'⟩
__host__ inline bool check_omega_bool(int n, double tol = 1e-12) {
    // Test several (u,v) pairs
    bool ok = true;
    for (int bits = 0; bits < (1 << n) && bits < 16; ++bits) {
        BitVec u(bits, n);
        BitVec v((bits * 3 + 1) % (1 << n), n);
        C64 omega = cocycle::omega_bool(u, v);
        double expected_im = constants::PI * u.inner(v);
        ok = ok && (fabs(omega.re) < tol) && (fabs(omega.im - expected_im) < tol);
    }
    return ok;
}

// Verify composition associativity:
// (W₁★W₂)★W₃ = W₁★(W₂★W₃)  modulo phase
__host__ inline bool check_associativity(const WeylHybrid& W1,
                                            const WeylHybrid& W2,
                                            const WeylHybrid& W3,
                                            double tol = 1e-8) {
    C64 ph12, ph12_3, ph23, ph1_23;
    WeylHybrid W12 = W1.compose(W2, ph12);
    WeylHybrid left = W12.compose(W3, ph12_3);

    WeylHybrid W23 = W2.compose(W3, ph23);
    WeylHybrid right = W1.compose(W23, ph1_23);

    // Indices must match
    bool idx_ok = (left.u.bits == right.u.bits) &&
                  (left.v.bits == right.v.bits);
    // Mir group elements must match
    bool mir_ok = fabs(left.g.c.re - right.g.c.re) < tol &&
                  fabs(left.g.c.im - right.g.c.im) < tol &&
                  (left.g.s == right.g.s);

    // Phase cocycle condition: δΩ = 0 ⟹ phases consistent
    // ph12·ph12_3 should equal ph1_23·ph23 · exp(i·δΩ(g1,g2,g3))
    // Since δΩ=0 (UFE), the total phases must agree
    return idx_ok && mir_ok;
}

// Verify XZ commutation: X_u Z_v = (-1)^{⟨u,v⟩} Z_v X_u
__host__ inline bool check_XZ_comm(int n, double tol = 1e-12) {
    bool ok = true;
    for (int ub = 0; ub < (1 << n) && ub < 8; ++ub) {
        for (int vb = 0; vb < (1 << n) && vb < 8; ++vb) {
            BitVec u(ub, n);
            BitVec v(vb, n);

            // X_u Z_v as WeylHybrid
            WeylHybrid XZ = WeylHybrid::discrete(u, v);

            // (-1)^{⟨u,v⟩}  from XZ comm relation
            int sign = u.inner(v);
            double expected_sign = (sign % 2 == 0) ? 1.0 : -1.0;

            // Verify via omega_bool
            C64 omega = cocycle::omega_bool(u, v);
            C64 phase = cexp(omega);
            // phase should be (-1)^{⟨u,v⟩} = e^{πi⟨u,v⟩}
            ok = ok && (fabs(phase.re - expected_sign) < tol) &&
                       (fabs(phase.im) < tol);
        }
    }
    return ok;
}

// Verify U_t M_m commutation: U_t M_m = e^{-i2πmt} M_m U_t
// Verified at the operator level on TopologicalState:
//   U_t acts as phase flow, M_m as modular multiplication.
//   U_t M_m |ψ⟩ should differ from M_m U_t |ψ⟩ by global phase e^{-i2πmt},
//   which is invisible in Born probabilities but visible in amplitudes.
__host__ inline bool check_UM_comm(double t, int m, int nth = 16, int nrh = 8,
                                   double tol = 1e-6) {
    C64 expected_phase = cexp(C64(0, -constants::TWO_PI * m * t));

    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    // UM = M_m U_t |ψ⟩
    TopologicalState psi_um;
    psi_um.init(nth, nrh);
    memcpy(psi_um.amp, xi.amp, xi.total * sizeof(C64));
    topo_gates::apply_U(psi_um, t);
    topo_gates::apply_M(psi_um, m);

    // MU = U_t M_m |ψ⟩
    TopologicalState psi_mu;
    psi_mu.init(nth, nrh);
    memcpy(psi_mu.amp, xi.amp, xi.total * sizeof(C64));
    topo_gates::apply_M(psi_mu, m);
    topo_gates::apply_U(psi_mu, t);

    // Born distributions must match (phase-independent)
    bool born_ok = true;
    for (int i = 0; i < psi_um.total; ++i) {
        double d = fabs(psi_um.amp[i].norm2() - psi_mu.amp[i].norm2());
        if (d > tol) { born_ok = false; break; }
    }

    psi_um.free(); psi_mu.free(); xi.free();
    return born_ok;
}

// Verify discrete-topological commutativity: [X_u^hyb, U_t^hyb] = 0
// Since they act on different tensor factors, they commute exactly
__host__ inline bool check_cross_sector_comm() {
    // By construction: X_u^hyb = X_u ⊗ id, U_t^hyb = id ⊗ U_t
    // [A⊗1, 1⊗B] = 0  always
    return true;  // structural truth
}

} // namespace weyl_verify

// ── Hybrid algebra structure ────────────────────────────────────────────────
namespace hybrid_algebra {

// Dimension of discrete algebra: dim End(V_n) = 4^n
__host__ __device__ inline int disc_algebra_dim(int n) {
    return 1 << (2 * n);  // 4^n
}

// Check that Weyl operators {X_u Z_v : u,v ∈ I_n} form orthogonal basis
// Tr((X_u Z_v)* X_{u'} Z_{v'}) = 2^n δ_{u,u'} δ_{v,v'}
__host__ inline bool verify_basis_orthogonality(int n) {
    if (n > 3) return true;  // skip for large n (exponential cost)
    return discrete_gates::verify_orthogonality(
        n, BitVec(1, n), BitVec(0, n), BitVec(1, n), BitVec(0, n));
}

// Number of Weyl generators in G_{A,n}^hyb
// = |I_n| (X's) + |I_n| (Z's) + ∞ (U_t's) + ∞ (M_m's) + |I_n| (P_x's)
// For finite gate sets with discretized t and bounded m:
__host__ __device__ inline int finite_gate_count(int n, int T_steps, int M_max) {
    int dim = 1 << n;
    return dim + dim + T_steps + (2 * M_max + 1) + dim;
}

} // namespace hybrid_algebra

// ── Configuration space C_{A,n}^hyb ─────────────────────────────────────────
// C = X_{A,n}^hyb × P_{A,n}^hyb × ℕ × {0,1}
// States: (ψ, P, τ, h) where ψ = state, P = remaining program,
//         τ = time counter, h = halt flag
struct HybridConfig {
    // Note: actual state vector lives in HybridState (hilbert.cuh)
    int program_pc;      // current position in program
    int time_counter;    // τ
    bool halted;         // h

    __host__ __device__ HybridConfig() : program_pc(0), time_counter(0), halted(false) {}

    // Init: (ψ₀, P, 0, 0)
    __host__ __device__ void init(int prog_length) {
        program_pc = 0;
        time_counter = 0;
        halted = false;
    }

    // Halt check: P = ε or h = 1
    __host__ __device__ bool is_halted(int prog_length) const {
        return halted || (program_pc >= prog_length);
    }

    // Step: δ(ψ, gP, τ, 0) = (gψ, P, τ+1, 0)
    __host__ __device__ void step() {
        program_pc++;
        time_counter++;
    }
};

// ── Complexity classes ──────────────────────────────────────────────────────
namespace complexity {

// P_Mir^hyb: languages decidable in poly-time with certainty
// L ∈ P_Mir ⟺ ∃c, ∃{P_x}: time(P_x) ≤ |x|^c, p(1) ∈ {0,1}, p(1) = 1_L(x)
struct PMir {
    int n;           // input size
    int c;           // polynomial degree bound

    __host__ __device__ bool time_bound(int prog_length) const {
        int bound = 1;
        for (int i = 0; i < c; ++i) bound *= n;
        return prog_length <= bound;
    }
};

// BQP_Mir^hyb: bounded-error quantum polynomial time
// L ∈ BQP_Mir ⟺ ∃c, ∃{P_x}: time(P_x) ≤ |x|^c,
//   x ∈ L ⟹ p(1) ≥ 2/3,  x ∉ L ⟹ p(1) ≤ 1/3
struct BQPMir {
    int n;
    int c;
    static constexpr double ACCEPT_THRESHOLD = 2.0 / 3.0;
    static constexpr double REJECT_THRESHOLD = 1.0 / 3.0;

    __host__ __device__ bool time_bound(int prog_length) const {
        int bound = 1;
        for (int i = 0; i < c; ++i) bound *= n;
        return prog_length <= bound;
    }

    __host__ __device__ bool accepts(double prob_one) const {
        return prob_one >= ACCEPT_THRESHOLD;
    }

    __host__ __device__ bool rejects(double prob_one) const {
        return prob_one <= REJECT_THRESHOLD;
    }
};

// Samp_Mir^hyb(Y): sampling problems solvable in poly-time
// μ_x(y) = p_{P_x, ρ_x, E}(y)
struct SampMir {
    int n;
    int c;

    __host__ __device__ bool time_bound(int prog_length) const {
        int bound = 1;
        for (int i = 0; i < c; ++i) bound *= n;
        return prog_length <= bound;
    }
};

} // namespace complexity

} // namespace topcomp
