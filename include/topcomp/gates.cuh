// ============================================================================
// TopComp: Gate Operators — Discrete + Topological + Hybrid
// ============================================================================
// Gate set G_{A,n}^HES for the hybrid computation machine:
//
//   DISCRETE GATES (act on V_n, trivially on X_A^top):
//     X_u: Pauli-X shift by u ∈ F_2^n        X_u|x⟩ = |x⊕u⟩
//     Z_v: Pauli-Z phase by v ∈ F_2^n        Z_v|x⟩ = (-1)^{⟨v,x⟩}|x⟩
//     P_x: projector to |x⟩                  P_x = |x⟩⟨x|
//
//   TOPOLOGICAL GATES (act on X_A^top, trivially on V_n):
//     U_t: golden-ratio flow    (U_tΨ)(θ,ρ) = Ψ(θ - πt/2, ρ + t·ln φ)
//     M_m: modular multiplier   (M_mΨ)(θ,ρ) = e^{-i(2πm/ln φ)ρ} Ψ(θ,ρ)
//
//   HYBRID OPERATORS:
//     X_u^hyb = X_u ⊗ id_{X_top}
//     Z_v^hyb = Z_v ⊗ id_{X_top}
//     U_t^hyb = id_{V_n} ⊗ U_t
//     M_m^hyb = id_{V_n} ⊗ M_m
//     P_x^hyb = P_x ⊗ id_{X_top}
//
//   COMMUTATION RELATIONS:
//     X_uZ_v = (-1)^{⟨u,v⟩} Z_vX_u
//     U_tM_m = e^{-i2πmt} M_mU_t
//     [X_u^hyb, U_t^hyb] = 0  (discrete-topological commutativity)
// ============================================================================
#pragma once
#include "hilbert.cuh"

namespace topcomp {

// ── Discrete Gate Operations (on QubitState / V_n) ──────────────────────────

namespace discrete_gates {

// X_u: shift operator  X_u|x⟩ = |x ⊕ u⟩
__host__ inline void apply_X(QubitState& psi, const BitVec& u) {
    int dim = psi.dim;
    C64* tmp = new C64[dim];
    for (int x = 0; x < dim; ++x) {
        int xp = x ^ u.bits;
        tmp[xp] = psi.amp[x];
    }
    memcpy(psi.amp, tmp, dim * sizeof(C64));
    delete[] tmp;
}

// Z_v: phase operator  Z_v|x⟩ = (-1)^{⟨v,x⟩}|x⟩
__host__ inline void apply_Z(QubitState& psi, const BitVec& v) {
    int dim = psi.dim;
    for (int x = 0; x < dim; ++x) {
        BitVec xv(x, v.n);
        if (xv.inner(v) == 1) {
            psi.amp[x] = -psi.amp[x];
        }
    }
}

// P_x: computational basis projector  P_x|ψ⟩ = ⟨x|ψ⟩|x⟩
__host__ inline void apply_P(QubitState& psi, uint32_t x) {
    C64 coeff = psi.amp[x];
    memset(psi.amp, 0, psi.dim * sizeof(C64));
    psi.amp[x] = coeff;
}

// Weyl operator W_{u,v} = X_u Z_v
// W_{u,v}|x⟩ = (-1)^{⟨v,x⟩}|x⊕u⟩
__host__ inline void apply_W(QubitState& psi, const BitVec& u, const BitVec& v) {
    int dim = psi.dim;
    C64* tmp = new C64[dim];
    for (int x = 0; x < dim; ++x) {
        BitVec xv(x, v.n);
        int phase = xv.inner(v);
        int xp = x ^ u.bits;
        tmp[xp] = phase ? -psi.amp[x] : psi.amp[x];
    }
    memcpy(psi.amp, tmp, dim * sizeof(C64));
    delete[] tmp;
}

// Check orthogonality: Tr((X_uZ_v)* X_{u'}Z_{v'}) = 2^n δ_{u,u'} δ_{v,v'}
__host__ inline bool verify_orthogonality(int n, const BitVec& u, const BitVec& v,
                                    const BitVec& up, const BitVec& vp) {
    int dim = 1 << n;
    C64 trace(0, 0);
    for (int x = 0; x < dim; ++x) {
        // (X_uZ_v)* X_{u'}Z_{v'} |x⟩
        BitVec xbv(x, n);
        int phase1 = xbv.inner(vp);             // Z_{v'}|x⟩ phase
        int xp1 = x ^ up.bits;                   // X_{u'}|x⊕u'⟩
        BitVec xp1bv(xp1, n);
        int phase2 = xp1bv.inner(v);             // Z_v
        int xp2 = xp1 ^ u.bits;                  // X_u

        // ⟨x| result⟩ = δ_{x, xp2} · (-1)^{phase1 + phase2}
        if (x == xp2) {
            trace += C64(((phase1 + phase2) % 2 == 0) ? 1.0 : -1.0, 0.0);
        }
    }
    bool expected = (u == up && v == vp);
    double expected_val = expected ? dim : 0.0;
    return fabs(trace.re - expected_val) < 1e-10 && fabs(trace.im) < 1e-10;
}

} // namespace discrete_gates

// ── Topological Gate Operations (on TopologicalState / X_A^top) ─────────────

namespace topo_gates {

// U_t: golden-ratio flow  (U_tΨ)(θ,ρ) = Ψ(U_φ^t(θ,ρ))
//    = Ψ(θ - πt/2, ρ + t·ln φ)
__host__ inline void apply_U(TopologicalState& psi, double t) {
    int nth = psi.N_theta;
    int nrh = psi.N_rho;
    double d_theta = constants::TWO_PI / nth;
    double d_rho = (psi.rho_max - psi.rho_min) / (nrh - 1);

    C64* tmp = new C64[psi.total];
    memset(tmp, 0, psi.total * sizeof(C64));

    for (int i = 0; i < nth; ++i) {
        for (int j = 0; j < nrh; ++j) {
            PhasePoint p = psi.grid_point(i, j);
            // Source point: U_φ^t(θ,ρ) = (θ - πt/2, ρ + t·ln φ)
            double src_theta = p.theta - constants::HALF_PI * t;
            double src_rho = p.rho + constants::ELL * t;

            // Map back to grid indices
            double si = fmod(src_theta / constants::TWO_PI * nth, (double)nth);
            if (si < 0) si += nth;
            double sj = (src_rho - psi.rho_min) / (psi.rho_max - psi.rho_min) * (nrh - 1);

            // Bilinear interpolation
            int i0 = (int)floor(si) % nth;
            int i1 = (i0 + 1) % nth;
            int j0 = (int)floor(sj);
            int j1 = j0 + 1;
            double fi = si - floor(si);
            double fj = sj - floor(sj);

            if (j0 >= 0 && j1 < nrh) {
                tmp[psi.index(i, j)] =
                    psi.at(i0, j0) * ((1 - fi) * (1 - fj)) +
                    psi.at(i1, j0) * (fi * (1 - fj)) +
                    psi.at(i0, j1) * ((1 - fi) * fj) +
                    psi.at(i1, j1) * (fi * fj);
            }
        }
    }
    memcpy(psi.amp, tmp, psi.total * sizeof(C64));
    delete[] tmp;
}

// M_m: modular multiplier  (M_mΨ)(θ,ρ) = exp(-i·2πm·ρ/ln φ)·Ψ(θ,ρ)
__host__ inline void apply_M(TopologicalState& psi, int m) {
    double coeff = -constants::TWO_PI * m / constants::ELL;
    for (int i = 0; i < psi.N_theta; ++i) {
        for (int j = 0; j < psi.N_rho; ++j) {
            PhasePoint p = psi.grid_point(i, j);
            double phase = coeff * p.rho;
            C64 factor(cos(phase), sin(phase));
            psi.at(i, j) = psi.at(i, j) * factor;
        }
    }
}

// Verify commutation: U_t M_m = e^{-i2πmt} M_m U_t
__host__ inline C64 commutation_phase(double t, int m) {
    double phase = -constants::TWO_PI * m * t;
    return C64(cos(phase), sin(phase));
}

// Ergodic average: Π_N = (1/(2N+1)) ∑_{n=-N}^{N} U^n
__host__ inline void apply_ergodic_projector(TopologicalState& psi, int N) {
    TopologicalState result;
    result.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);

    double norm_factor = 1.0 / (2 * N + 1);
    for (int k = -N; k <= N; ++k) {
        TopologicalState tmp;
        tmp.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        memcpy(tmp.amp, psi.amp, psi.total * sizeof(C64));
        apply_U(tmp, static_cast<double>(k));
        for (int i = 0; i < psi.total; ++i) {
            result.amp[i] += tmp.amp[i] * norm_factor;
        }
        tmp.free();
    }
    memcpy(psi.amp, result.amp, psi.total * sizeof(C64));
    result.free();
}

} // namespace topo_gates

// ── Hybrid Gate Operations (on HybridState / X_{A,n}^hyb) ───────────────────

namespace hybrid_gates {

// X_u^hyb = X_u ⊗ id_{X_top}
__host__ inline void apply_X_hyb(HybridState& psi, const BitVec& u) {
    int dim_d = psi.dim_disc;
    int dim_t = psi.dim_top;
    C64* tmp = new C64[psi.total];
    for (int x = 0; x < dim_d; ++x) {
        int xp = x ^ u.bits;
        memcpy(tmp + xp * dim_t, psi.amp + x * dim_t, dim_t * sizeof(C64));
    }
    memcpy(psi.amp, tmp, psi.total * sizeof(C64));
    delete[] tmp;
}

// Z_v^hyb = Z_v ⊗ id_{X_top}
__host__ inline void apply_Z_hyb(HybridState& psi, const BitVec& v) {
    int dim_d = psi.dim_disc;
    int dim_t = psi.dim_top;
    for (int x = 0; x < dim_d; ++x) {
        BitVec xv(x, v.n);
        if (xv.inner(v) == 1) {
            for (int i = 0; i < dim_t; ++i) {
                psi.amp[x * dim_t + i] = -psi.amp[x * dim_t + i];
            }
        }
    }
}

// U_t^hyb = id_{V_n} ⊗ U_t  (topological flow on each qubit sector)
__host__ inline void apply_U_hyb(HybridState& psi, double t) {
    // Apply U_t to each topological component independently
    for (int x = 0; x < psi.dim_disc; ++x) {
        TopologicalState slice;
        slice.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        psi.project(x, slice);
        topo_gates::apply_U(slice, t);
        psi.inject(x, slice);
        slice.free();
    }
}

// M_m^hyb = id_{V_n} ⊗ M_m
__host__ inline void apply_M_hyb(HybridState& psi, int m) {
    for (int x = 0; x < psi.dim_disc; ++x) {
        TopologicalState slice;
        slice.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        psi.project(x, slice);
        topo_gates::apply_M(slice, m);
        psi.inject(x, slice);
        slice.free();
    }
}

// P_x^hyb = P_x ⊗ id_{X_top}  (projector onto qubit state x)
__host__ inline void apply_P_hyb(HybridState& psi, uint32_t x) {
    psi.project_inplace(x);
}

// H_q^hyb = H_q ⊗ id_{X_top}  (Hadamard on qubit q)
// (H_q|ψ⟩)(x,j) = (1/√2)[ψ(x₀,j) + (-1)^{x_q} ψ(x₁,j)]
// where x₀ = x with bit q=0,  x₁ = x with bit q=1
// Justified by universality: H ∈ End(V_n) ⊂ closure of gate algebra
__host__ inline void apply_H_hyb(HybridState& psi, int qubit) {
    int dim_d = psi.dim_disc;
    int dim_t = psi.dim_top;
    int mask = 1 << qubit;
    double inv_sqrt2 = 1.0 / sqrt(2.0);
    C64* tmp = new C64[psi.total];
    memcpy(tmp, psi.amp, psi.total * sizeof(C64));
    for (int x = 0; x < dim_d; ++x) {
        int partner = x ^ mask;
        if (x < partner) {
            // x has bit q=0, partner has bit q=1
            for (int j = 0; j < dim_t; ++j) {
                C64 a0 = psi.amp[x * dim_t + j];
                C64 a1 = psi.amp[partner * dim_t + j];
                tmp[x * dim_t + j]       = (a0 + a1) * inv_sqrt2;
                tmp[partner * dim_t + j] = (a0 - a1) * inv_sqrt2;
            }
        }
    }
    memcpy(psi.amp, tmp, psi.total * sizeof(C64));
    delete[] tmp;
}

// W_{u,v;g}^hyb: full hybrid Weyl operator
// W_{u,v;g}|ψ⟩(x', j) = (-1)^{⟨v, x'⊕u⟩} · ψ(x'⊕u, ρ(g⁻¹)·j)
// i.e. for each input x: phase by (-1)^{⟨v,x⟩}, translate topo by ρ(g), shift disc by u
__host__ inline void apply_W_hyb(HybridState& psi, const BitVec& u,
                                 const BitVec& v, const MirElement& g) {
    int dim_d = psi.dim_disc;
    int dim_t = psi.dim_top;
    int nth = psi.N_theta;
    int nrh = psi.N_rho;
    double d_rho = (nrh > 1) ? (psi.rho_max - psi.rho_min) / (nrh - 1) : 1.0;

    // Compute g⁻¹ for pullback: (s,c)⁻¹ = (s, -sc)
    MirElement ginv = g.inverse();

    C64* tmp = new C64[psi.total];
    memset(tmp, 0, psi.total * sizeof(C64));

    for (int x = 0; x < dim_d; ++x) {
        int xp = x ^ u.bits;  // output discrete index x' = x ⊕ u
        // Phase: (-1)^{⟨v, x⟩}
        BitVec xbv(x, v.n);
        double sign = (xbv.inner(v) == 1) ? -1.0 : 1.0;

        // Apply ρ(g) to topological slice: (ρ(g)Ψ)(θ,ρ) = Ψ(g⁻¹·(θ,ρ))
        // g⁻¹·(θ,ρ) = (s·θ - α_inv, s·ρ + β_inv) where ginv = (s, β_inv - i·α_inv)
        for (int i = 0; i < nth; ++i) {
            for (int j = 0; j < nrh; ++j) {
                // Current grid point
                double theta = constants::TWO_PI * i / nth;
                double rho_val = psi.rho_min + j * d_rho;

                // Source point via g⁻¹ action
                double src_theta, src_rho;
                ginv.act_phase(theta, rho_val, src_theta, src_rho);

                // Map to grid indices with periodic θ and clamped ρ
                double si = fmod(src_theta / constants::TWO_PI * nth, (double)nth);
                if (si < 0) si += nth;
                double sj = (src_rho - psi.rho_min) / (psi.rho_max - psi.rho_min) * (nrh - 1);

                // Bilinear interpolation (same pattern as apply_U)
                int i0 = (int)floor(si) % nth;
                int i1 = (i0 + 1) % nth;
                int j0 = (int)floor(sj);
                int j1 = j0 + 1;
                double fi = si - floor(si);
                double fj = sj - floor(sj);

                C64 val(0, 0);
                if (j0 >= 0 && j1 < nrh) {
                    int base = x * dim_t;
                    val = psi.amp[base + i0 * nrh + j0] * ((1 - fi) * (1 - fj)) +
                          psi.amp[base + i1 * nrh + j0] * (fi * (1 - fj)) +
                          psi.amp[base + i0 * nrh + j1] * ((1 - fi) * fj) +
                          psi.amp[base + i1 * nrh + j1] * (fi * fj);
                }
                tmp[xp * dim_t + i * nrh + j] = val * sign;
            }
        }
    }
    memcpy(psi.amp, tmp, psi.total * sizeof(C64));
    delete[] tmp;
}

// ── Verify commutation relations ────────────────────────────────────────────

// [X_u^hyb, U_t^hyb] = 0
// [X_u^hyb, M_m^hyb] = 0
// [Z_v^hyb, U_t^hyb] = 0
// [Z_v^hyb, M_m^hyb] = 0
// [P_x^hyb, U_t^hyb] = 0
// [P_x^hyb, M_m^hyb] = 0

// X_u Z_v = (-1)^{⟨u,v⟩} Z_v X_u  on hybrid space
__host__ inline bool verify_XZ_commutation(HybridState& psi,
                                      const BitVec& u, const BitVec& v) {
    HybridState psi1, psi2;
    psi1.init(psi.n_qubits, psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
    psi2.init(psi.n_qubits, psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
    memcpy(psi1.amp, psi.amp, psi.total * sizeof(C64));
    memcpy(psi2.amp, psi.amp, psi.total * sizeof(C64));

    // X_u Z_v
    apply_Z_hyb(psi1, v);
    apply_X_hyb(psi1, u);

    // (-1)^{⟨u,v⟩} Z_v X_u
    apply_X_hyb(psi2, u);
    apply_Z_hyb(psi2, v);
    int phase = BitVec(u.bits, u.n).inner(v) ? -1 : 1;
    for (int i = 0; i < psi.total; ++i) {
        psi2.amp[i] = psi2.amp[i] * static_cast<double>(phase);
    }

    // Compare
    double diff = 0;
    for (int i = 0; i < psi.total; ++i) {
        diff += (psi1.amp[i] - psi2.amp[i]).norm2();
    }
    psi1.free();
    psi2.free();
    return diff < 1e-20;
}

} // namespace hybrid_gates

// ── Gate type enumeration for the HES machine ───────────────────────────────
enum class GateType : int {
    X_HYB   = 0,  // X_u^hyb
    Z_HYB   = 1,  // Z_v^hyb
    U_HYB   = 2,  // U_t^hyb
    M_HYB   = 3,  // M_m^hyb
    P_HYB   = 4,  // P_x^hyb
    H_HYB   = 5,  // Hadamard on qubit q
    W_HYB   = 6   // W_{u,v;g} full hybrid Weyl
};

// Gate descriptor (for machine programs)
struct Gate {
    GateType type;
    union {
        uint32_t bits;    // for X, Z, P, H (qubit index)
        double   t;       // for U_t
        int      m;       // for M_m
    } param;
    // Extended params for W_HYB (Weyl gate)
    BitVec w_u;           // u for W_{u,v;g}
    BitVec w_v;           // v for W_{u,v;g}
    MirElement w_g;       // g for W_{u,v;g}

    Gate() : type(GateType::X_HYB), w_u(0,1), w_v(0,1), w_g() { param.bits = 0; }

    __host__ __device__ int cost(int n) const {
        switch (type) {
            case GateType::X_HYB: {
                BitVec bv(param.bits, n);
                return bv.weight();
            }
            case GateType::Z_HYB: {
                BitVec bv(param.bits, n);
                return bv.weight();
            }
            case GateType::U_HYB: return 1;
            case GateType::M_HYB: return 1;
            case GateType::P_HYB: return 1;
            case GateType::H_HYB: return 1;
            case GateType::W_HYB: return w_u.weight() + w_v.weight() + 1;
            default: return 0;
        }
    }

    // Apply gate to hybrid state
    __host__ void apply(HybridState& psi) const {
        switch (type) {
            case GateType::X_HYB:
                hybrid_gates::apply_X_hyb(psi, BitVec(param.bits, psi.n_qubits));
                break;
            case GateType::Z_HYB:
                hybrid_gates::apply_Z_hyb(psi, BitVec(param.bits, psi.n_qubits));
                break;
            case GateType::U_HYB:
                hybrid_gates::apply_U_hyb(psi, param.t);
                break;
            case GateType::M_HYB:
                hybrid_gates::apply_M_hyb(psi, param.m);
                break;
            case GateType::P_HYB:
                hybrid_gates::apply_P_hyb(psi, param.bits);
                break;
            case GateType::H_HYB:
                hybrid_gates::apply_H_hyb(psi, static_cast<int>(param.bits));
                break;
            case GateType::W_HYB:
                hybrid_gates::apply_W_hyb(psi, w_u, w_v, w_g);
                break;
        }
    }
};

// ── Gate constructors ───────────────────────────────────────────────────────
inline Gate gate_X(uint32_t u)    { Gate g; g.type = GateType::X_HYB; g.param.bits = u; return g; }
inline Gate gate_Z(uint32_t v)    { Gate g; g.type = GateType::Z_HYB; g.param.bits = v; return g; }
inline Gate gate_U(double t)      { Gate g; g.type = GateType::U_HYB; g.param.t = t;    return g; }
inline Gate gate_M(int m)         { Gate g; g.type = GateType::M_HYB; g.param.m = m;    return g; }
inline Gate gate_P(uint32_t x)    { Gate g; g.type = GateType::P_HYB; g.param.bits = x; return g; }
inline Gate gate_H(int qubit)     { Gate g; g.type = GateType::H_HYB; g.param.bits = static_cast<uint32_t>(qubit); return g; }
inline Gate gate_W(BitVec u, BitVec v, MirElement mir_g) {
    Gate g;
    g.type = GateType::W_HYB;
    g.w_u = u;
    g.w_v = v;
    g.w_g = mir_g;
    return g;
}

} // namespace topcomp
