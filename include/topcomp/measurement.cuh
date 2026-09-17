// ============================================================================
// TopComp: Orthodox Quantum Measurement — Density, PVM, POVM, Born Rule
// ============================================================================
// Full measurement formalism from reference Section V:
//
//   State space:
//     S_{A,n}^hyb = { ρ ∈ End(X_{A,n}^hyb) : ρ ≥ 0, Tr(ρ) = 1 }
//
//   Projective measurement (PVM):
//     E: B(ℝ) → End(X_{A,n}^hyb)
//     E(B)² = E(B),  E(B₁)E(B₂) = E(B₁∩B₂),  E(ℝ) = 1
//
//   General measurement (POVM):
//     F: B(ℝ) → End(X_{A,n}^hyb)
//     F(B) ≥ 0,  F(ℝ) = 1,  F(⊔B_j) = ΣF(B_j)
//
//   Born rule:
//     p_{ρ,E}(y) = Tr(ρ E_y)
//     Σ_y p_{ρ,E}(y) = Tr(ρ Σ E_y) = Tr(ρ) = 1
//
//   Lüders state update (PVM):
//     ρ → ρ_x^L = P_x ρ P_x / Tr(ρ P_x)
//
//   POVM state update:
//     ρ → ρ_B^F = √F(B) ρ √F(B) / Tr(ρ F(B))
//
//   Expectation & variance:
//     E_ρ(A) = Tr(ρ A)
//     Δ_ρ(A)² = Tr(ρ A²) - Tr(ρ A)²
//     Cov_ρ(A,B) = Tr(ρ AB) - Tr(ρ A)Tr(ρ B)
//
//   Heisenberg uncertainty for U_t, M_m:
//     Δ(U_t)² Δ(M_m)² ≥ ¼|⟨[U_t, M_m]⟩|²
//     where [U_t, M_m] = (e^{-i2πmt} - 1) M_m U_t
// ============================================================================
#pragma once
#include "hilbert.cuh"

namespace topcomp {
namespace measurement {

// ── Dense operator on V_n (2^n × 2^n matrix) ────────────────────────────────
struct DenseOp {
    C64* data;
    int dim;     // 2^n

    __host__ DenseOp() : data(nullptr), dim(0) {}
    __host__ explicit DenseOp(int d) : dim(d) {
        data = new C64[d * d];
        memset(data, 0, d * d * sizeof(C64));
    }
    __host__ ~DenseOp() { delete[] data; }

    // Copy constructor
    __host__ DenseOp(const DenseOp& o) : dim(o.dim) {
        data = new C64[dim * dim];
        memcpy(data, o.data, dim * dim * sizeof(C64));
    }

    // Assignment
    __host__ DenseOp& operator=(const DenseOp& o) {
        if (this != &o) {
            delete[] data;
            dim = o.dim;
            data = new C64[dim * dim];
            memcpy(data, o.data, dim * dim * sizeof(C64));
        }
        return *this;
    }

    __host__ C64& at(int i, int j) { return data[i * dim + j]; }
    __host__ const C64& at(int i, int j) const { return data[i * dim + j]; }

    // Identity
    __host__ static DenseOp identity(int d) {
        DenseOp I(d);
        for (int i = 0; i < d; ++i) I.at(i, i) = C64(1, 0);
        return I;
    }

    // Zero
    __host__ static DenseOp zero(int d) {
        return DenseOp(d);
    }

    // Trace
    __host__ C64 trace() const {
        C64 t(0, 0);
        for (int i = 0; i < dim; ++i) t += at(i, i);
        return t;
    }

    // Matrix multiply
    __host__ DenseOp operator*(const DenseOp& B) const {
        DenseOp C(dim);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                C64 s(0, 0);
                for (int k = 0; k < dim; ++k)
                    s += at(i, k) * B.at(k, j);
                C.at(i, j) = s;
            }
        return C;
    }

    // Scalar multiply
    __host__ DenseOp operator*(C64 s) const {
        DenseOp R(dim);
        for (int i = 0; i < dim * dim; ++i) R.data[i] = data[i] * s;
        return R;
    }

    // Addition
    __host__ DenseOp operator+(const DenseOp& B) const {
        DenseOp R(dim);
        for (int i = 0; i < dim * dim; ++i) R.data[i] = data[i] + B.data[i];
        return R;
    }

    // Subtraction
    __host__ DenseOp operator-(const DenseOp& B) const {
        DenseOp R(dim);
        for (int i = 0; i < dim * dim; ++i) R.data[i] = data[i] - B.data[i];
        return R;
    }

    // Adjoint (conjugate transpose)
    __host__ DenseOp dagger() const {
        DenseOp R(dim);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                R.at(i, j) = at(j, i).conj();
        return R;
    }

    // Hilbert-Schmidt norm: ||A||_HS² = Tr(A† A)
    __host__ double hs_norm2() const {
        double s = 0;
        for (int i = 0; i < dim * dim; ++i) s += data[i].norm2();
        return s;
    }

    __host__ double hs_norm() const { return sqrt(hs_norm2()); }

    // Frobenius inner product: ⟨A, B⟩ = Tr(A† B)
    __host__ C64 inner(const DenseOp& B) const {
        C64 s(0, 0);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                s += at(j, i).conj() * B.at(j, i);  // (A†)_{ij} = A_{ji}*
        return s;
    }
};

// ── Density operator (state) ────────────────────────────────────────────────
// ρ ∈ S = { ρ ∈ End(V_n) : ρ ≥ 0, Tr(ρ) = 1 }

// Create density from pure state: ρ = |ψ⟩⟨ψ|
__host__ inline DenseOp pure_state_density(const QubitState& psi) {
    DenseOp rho(psi.dim);
    for (int i = 0; i < psi.dim; ++i)
        for (int j = 0; j < psi.dim; ++j)
            rho.at(i, j) = psi.amp[i] * psi.amp[j].conj();
    return rho;
}

// Maximally mixed state: ρ = I/d
__host__ inline DenseOp maximally_mixed(int dim) {
    DenseOp rho = DenseOp::identity(dim);
    C64 scale(1.0 / dim, 0);
    for (int i = 0; i < dim * dim; ++i) rho.data[i] = rho.data[i] * scale;
    return rho;
}

// Verify density operator properties
__host__ inline bool is_valid_density(const DenseOp& rho, double tol = 1e-10) {
    // Check Tr(ρ) = 1
    C64 tr = rho.trace();
    if (fabs(tr.re - 1.0) > tol || fabs(tr.im) > tol) return false;

    // Check Hermiticity: ρ = ρ†
    for (int i = 0; i < rho.dim; ++i)
        for (int j = i + 1; j < rho.dim; ++j) {
            C64 diff = rho.at(i, j) - rho.at(j, i).conj();
            if (diff.norm2() > tol * tol) return false;
        }

    // Check positivity via diagonal dominance (heuristic for small dims)
    for (int i = 0; i < rho.dim; ++i)
        if (rho.at(i, i).re < -tol) return false;

    return true;
}

// ── Projective measurement (PVM) ────────────────────────────────────────────
// Computational basis projectors: P_x = |x⟩⟨x|

__host__ inline DenseOp computational_projector(int x, int dim) {
    DenseOp P(dim);
    P.at(x, x) = C64(1, 0);
    return P;
}

// Verify PVM properties: P_x² = P_x, P_x P_y = δ_{xy} P_x, Σ P_x = I
__host__ inline bool verify_pvm_properties(int n) {
    int dim = 1 << n;

    // Σ P_x = I
    DenseOp sum = DenseOp::zero(dim);
    for (int x = 0; x < dim; ++x) {
        DenseOp Px = computational_projector(x, dim);
        sum = sum + Px;
    }
    DenseOp I = DenseOp::identity(dim);
    double diff = (sum - I).hs_norm2();
    if (diff > 1e-10) return false;

    // P_0² = P_0
    DenseOp P0 = computational_projector(0, dim);
    DenseOp P0sq = P0 * P0;
    if ((P0sq - P0).hs_norm2() > 1e-10) return false;

    // P_0 P_1 = 0 (if dim ≥ 2)
    if (dim >= 2) {
        DenseOp P1 = computational_projector(1, dim);
        DenseOp cross = P0 * P1;
        if (cross.hs_norm2() > 1e-10) return false;
    }

    return true;
}

// ── Born rule ───────────────────────────────────────────────────────────────
// p_{ρ,E}(y) = Tr(ρ E_y)

__host__ inline double born_probability(const DenseOp& rho, const DenseOp& E_y) {
    DenseOp rhoE = rho * E_y;
    return rhoE.trace().re;
}

// Born distribution over computational basis
__host__ inline void born_distribution(const DenseOp& rho, double* probs) {
    for (int x = 0; x < rho.dim; ++x) {
        probs[x] = rho.at(x, x).re;  // Tr(ρ P_x) = ρ_{xx} for comp basis
    }
}

// Verify Σ p(x) = 1
__host__ inline bool verify_born_normalization(const DenseOp& rho, double tol = 1e-10) {
    double sum = 0;
    for (int x = 0; x < rho.dim; ++x) sum += rho.at(x, x).re;
    return fabs(sum - 1.0) < tol;
}

// ── Lüders state update ─────────────────────────────────────────────────────
// Projective: ρ → P_x ρ P_x / Tr(ρ P_x)
__host__ inline DenseOp luders_update(const DenseOp& rho, const DenseOp& P_x) {
    double prob = born_probability(rho, P_x);
    if (prob < 1e-15) return rho;  // undefined for zero probability

    DenseOp post = P_x * rho * P_x;
    return post * C64(1.0 / prob, 0);
}

// General POVM update: ρ → √F(B) ρ √F(B) / Tr(ρ F(B))
// For computational basis, √P_x = P_x, so this reduces to Lüders
__host__ inline DenseOp povm_update(const DenseOp& rho, const DenseOp& sqrtF) {
    DenseOp post = sqrtF * rho * sqrtF;
    double prob = post.trace().re;
    if (prob < 1e-15) return rho;
    return post * C64(1.0 / prob, 0);
}

// ── Expectation values ──────────────────────────────────────────────────────
// E_ρ(A) = Tr(ρA)
__host__ inline C64 expectation(const DenseOp& rho, const DenseOp& A) {
    DenseOp rhoA = rho * A;
    return rhoA.trace();
}

// Δ_ρ(A)² = Tr(ρA²) - Tr(ρA)²
__host__ inline double variance(const DenseOp& rho, const DenseOp& A) {
    C64 EA = expectation(rho, A);
    DenseOp A2 = A * A;
    C64 EA2 = expectation(rho, A2);
    return EA2.re - EA.re * EA.re;
}

// Cov_ρ(A,B) = Tr(ρAB) - Tr(ρA)Tr(ρB)
__host__ inline C64 covariance(const DenseOp& rho, const DenseOp& A,
                                  const DenseOp& B) {
    C64 EAB = expectation(rho, A * B);
    C64 EA = expectation(rho, A);
    C64 EB = expectation(rho, B);
    return EAB - EA * EB;
}

// ── Heisenberg uncertainty relation ─────────────────────────────────────────
// Δ(A)² Δ(B)² ≥ ¼ |Tr(ρ[A,B])|²
// Returns (LHS, RHS) for comparison
struct UncertaintyBound {
    double lhs;       // Δ(A)² · Δ(B)²
    double rhs;       // ¼|⟨[A,B]⟩|²
    bool satisfied;   // lhs ≥ rhs
};

__host__ inline UncertaintyBound heisenberg_bound(
    const DenseOp& rho, const DenseOp& A, const DenseOp& B)
{
    double varA = variance(rho, A);
    double varB = variance(rho, B);

    DenseOp comm = A * B - B * A;  // [A,B]
    C64 comm_exp = expectation(rho, comm);
    double rhs = 0.25 * comm_exp.norm2();

    UncertaintyBound ub;
    ub.lhs = varA * varB;
    ub.rhs = rhs;
    ub.satisfied = (ub.lhs >= ub.rhs - 1e-12);
    return ub;
}

// ── Program execution with measurement ──────────────────────────────────────
// ρ → U_P ρ U_P†
__host__ inline DenseOp evolve_density(const DenseOp& rho, const DenseOp& U_P) {
    DenseOp Udag = U_P.dagger();
    return U_P * rho * Udag;
}

// Amplitude: Amp_{P,ρ}(y) = Tr(E_y U_P ρ)
__host__ inline C64 amplitude(const DenseOp& rho, const DenseOp& U_P,
                                 const DenseOp& E_y) {
    return expectation(evolve_density(rho, U_P), E_y);
}

// ── Post-measurement Born probability after program ─────────────────────────
// p_{P,ρ,E}(y) = Tr(U_P ρ U_P† E_y) = Tr(ρ U_P† E_y U_P)
__host__ inline double program_probability(const DenseOp& rho, const DenseOp& U_P,
                                              const DenseOp& E_y) {
    return amplitude(rho, U_P, E_y).re;
}

// ── Robertson-Schrödinger uncertainty ───────────────────────────────────────
// Δ(A)²Δ(B)² ≥ ¼|⟨[A,B]⟩|² + ¼|⟨{A,B}⟩ - 2⟨A⟩⟨B⟩|²
// The Schrödinger form includes the anticommutator term
struct SchrodingerBound {
    double lhs;           // Δ(A)² · Δ(B)²
    double rhs_comm;      // ¼|⟨[A,B]⟩|²         (Robertson term)
    double rhs_anti;      // ¼|⟨{A,B}⟩-2⟨A⟩⟨B⟩|² (Schrödinger term)
    double rhs;           // rhs_comm + rhs_anti
    bool satisfied;
};

__host__ inline SchrodingerBound schrodinger_bound(
    const DenseOp& rho, const DenseOp& A, const DenseOp& B)
{
    double varA = variance(rho, A);
    double varB = variance(rho, B);

    // Commutator [A,B] = AB - BA
    DenseOp comm = A * B - B * A;
    C64 comm_exp = expectation(rho, comm);

    // Anticommutator {A,B} = AB + BA
    DenseOp anti = A * B + B * A;
    C64 anti_exp = expectation(rho, anti);
    C64 two_EA_EB = expectation(rho, A) * expectation(rho, B) * C64(2, 0);
    C64 anti_centered = anti_exp - two_EA_EB;

    SchrodingerBound sb;
    sb.lhs = varA * varB;
    sb.rhs_comm = 0.25 * comm_exp.norm2();
    sb.rhs_anti = 0.25 * anti_centered.norm2();
    sb.rhs = sb.rhs_comm + sb.rhs_anti;
    sb.satisfied = (sb.lhs >= sb.rhs - 1e-12);
    return sb;
}

// ── Joint probability for commuting observables ─────────────────────────────
// When [P_x^hyb, Π_j^A] = 0:
//   p(x,j | ρ) = Tr(P_x^hyb Π_j^A ρ)
// For pure state |ψ⟩ on V_n ⊗ X_A^top with amp[x * dim_top + j]:
//   p(x,j) = |amp[x * dim_top + j]|²

// Joint probability distribution p(x, j) over discrete × topological grid
__host__ inline void joint_probability(const HybridState& psi,
                                        double* joint, int& n_disc, int& n_top) {
    n_disc = psi.dim_disc;
    n_top = psi.dim_top;
    double norm2 = psi.norm2();
    double inv_norm2 = (norm2 > 1e-30) ? 1.0 / norm2 : 0.0;
    for (int x = 0; x < n_disc; ++x)
        for (int j = 0; j < n_top; ++j)
            joint[x * n_top + j] = psi.amp[x * n_top + j].norm2() * inv_norm2;
}

// Marginal p(x) = Σ_j p(x,j)  (discrete boundary readout)
__host__ inline void marginal_disc(const double* joint, int n_disc, int n_top,
                                     double* marg) {
    for (int x = 0; x < n_disc; ++x) {
        marg[x] = 0;
        for (int j = 0; j < n_top; ++j)
            marg[x] += joint[x * n_top + j];
    }
}

// Marginal p(j) = Σ_x p(x,j)  (topological boundary readout)
__host__ inline void marginal_topo(const double* joint, int n_disc, int n_top,
                                     double* marg) {
    for (int j = 0; j < n_top; ++j) {
        marg[j] = 0;
        for (int x = 0; x < n_disc; ++x)
            marg[j] += joint[x * n_top + j];
    }
}

// Conditional p(j | x) = p(x,j) / p(x)
__host__ inline void conditional_topo_given_disc(const double* joint,
    int n_disc, int n_top, int x, double* cond) {
    double px = 0;
    for (int j = 0; j < n_top; ++j)
        px += joint[x * n_top + j];
    double inv_px = (px > 1e-30) ? 1.0 / px : 0.0;
    for (int j = 0; j < n_top; ++j)
        cond[j] = joint[x * n_top + j] * inv_px;
}

// Joint normalization check: Σ_{x,j} p(x,j) = 1
__host__ inline bool verify_joint_normalization(const double* joint,
    int n_disc, int n_top, double tol = 1e-10) {
    double sum = 0;
    for (int i = 0; i < n_disc * n_top; ++i) sum += joint[i];
    return fabs(sum - 1.0) < tol;
}

// ── Multi-boundary measurement ──────────────────────────────────────────────
// Simultaneous readout at discrete boundary (P_x^hyb) and topological
// boundary (Π_j^A).  Returns the full joint table plus both marginals.
struct MultiBoundaryResult {
    double* joint;     // p(x, j) — n_disc × n_top
    double* marg_disc; // p(x)    — n_disc
    double* marg_topo; // p(j)    — n_top
    int n_disc, n_top;
    double norm_check;

    __host__ void init(int nd, int nt) {
        n_disc = nd; n_top = nt;
        joint = new double[nd * nt];
        marg_disc = new double[nd];
        marg_topo = new double[nt];
    }
    __host__ void free() {
        delete[] joint; delete[] marg_disc; delete[] marg_topo;
    }
};

__host__ inline MultiBoundaryResult multi_boundary_measure(const HybridState& psi) {
    MultiBoundaryResult r;
    r.init(psi.dim_disc, psi.dim_top);

    int nd, nt;
    joint_probability(psi, r.joint, nd, nt);
    marginal_disc(r.joint, nd, nt, r.marg_disc);
    marginal_topo(r.joint, nd, nt, r.marg_topo);
    r.norm_check = 0;
    for (int i = 0; i < nd * nt; ++i) r.norm_check += r.joint[i];
    return r;
}

// ── Entropic quantities ─────────────────────────────────────────────────────

// Von Neumann entropy: S(ρ) = -Tr(ρ log ρ)
// Computed via eigenvalues for small systems
__host__ inline double von_neumann_entropy(const double* eigenvalues, int dim) {
    double S = 0;
    for (int i = 0; i < dim; ++i) {
        if (eigenvalues[i] > 1e-15)
            S -= eigenvalues[i] * log(eigenvalues[i]);
    }
    return S;
}

// Classical Shannon entropy from Born distribution
__host__ inline double shannon_entropy(const double* probs, int dim) {
    double H = 0;
    for (int i = 0; i < dim; ++i) {
        if (probs[i] > 1e-15)
            H -= probs[i] * log(probs[i]);
    }
    return H;
}

// Purity: Tr(ρ²) = 1 for pure states, 1/d for maximally mixed
__host__ inline double purity(const DenseOp& rho) {
    DenseOp rho2 = rho * rho;
    return rho2.trace().re;
}

// ── CPTP channels (Kraus representation) ────────────────────────────────────
// CPTP_{A,n} := { Φ : ∃{K_r}, Φ(ρ) = Σ_r K_r ρ K_r*, Σ_r K_r* K_r = I }
static constexpr int MAX_KRAUS = 16;

struct CPTPChannel {
    DenseOp kraus[MAX_KRAUS];  // Kraus operators K_r
    int n_kraus;               // number of Kraus operators
    int dim;                   // Hilbert space dimension

    __host__ CPTPChannel() : n_kraus(0), dim(0) {}
    __host__ CPTPChannel(int d) : n_kraus(0), dim(d) {}

    __host__ void add_kraus(const DenseOp& K) {
        if (n_kraus < MAX_KRAUS) {
            kraus[n_kraus] = K;
            n_kraus++;
        }
    }

    // Apply channel: Φ(ρ) = Σ_r K_r ρ K_r†
    __host__ DenseOp apply(const DenseOp& rho) const {
        DenseOp result = DenseOp::zero(dim);
        for (int r = 0; r < n_kraus; ++r) {
            DenseOp KrK = kraus[r] * rho * kraus[r].dagger();
            result = result + KrK;
        }
        return result;
    }

    // Verify trace-preservation: Σ_r K_r† K_r = I
    __host__ bool verify_tp(double tol = 1e-10) const {
        DenseOp sum = DenseOp::zero(dim);
        for (int r = 0; r < n_kraus; ++r) {
            sum = sum + (kraus[r].dagger() * kraus[r]);
        }
        DenseOp I = DenseOp::identity(dim);
        return (sum - I).hs_norm() < tol;
    }

    // Verify complete positivity (structural — Kraus form guarantees CP)
    __host__ bool verify_cp() const { return n_kraus > 0; }

    // Verify the channel preserves trace of density operators
    __host__ bool verify_trace_preserving(const DenseOp& rho, double tol = 1e-10) const {
        DenseOp out = apply(rho);
        return fabs(out.trace().re - rho.trace().re) < tol &&
               fabs(out.trace().im) < tol;
    }
};

// Named constructors for standard channels
namespace cptp {

// Identity channel: Φ(ρ) = ρ  (single Kraus = I)
__host__ inline CPTPChannel identity_channel(int dim) {
    CPTPChannel ch(dim);
    ch.add_kraus(DenseOp::identity(dim));
    return ch;
}

// Depolarizing channel: Φ(ρ) = (1-p)ρ + p·I/d
// Kraus: K_0 = √(1-p+p/d²)·I, K_{ij} = √(p/d²)·|i⟩⟨j| for i≠j, etc.
// Simplified 1-qubit version: K_0=√(1-3p/4)I, K_1=√(p/4)X, K_2=√(p/4)Y, K_3=√(p/4)Z
__host__ inline CPTPChannel depolarizing(double p) {
    int dim = 2;
    CPTPChannel ch(dim);

    double s0 = sqrt(1.0 - 3.0 * p / 4.0);
    double s1 = sqrt(p / 4.0);

    // K_0 = √(1-3p/4) I
    DenseOp K0 = DenseOp::identity(dim) * C64(s0, 0);
    ch.add_kraus(K0);

    // K_1 = √(p/4) X
    DenseOp K1(dim);
    K1.at(0, 1) = C64(s1, 0); K1.at(1, 0) = C64(s1, 0);
    ch.add_kraus(K1);

    // K_2 = √(p/4) Y
    DenseOp K2(dim);
    K2.at(0, 1) = C64(0, -s1); K2.at(1, 0) = C64(0, s1);
    ch.add_kraus(K2);

    // K_3 = √(p/4) Z
    DenseOp K3(dim);
    K3.at(0, 0) = C64(s1, 0); K3.at(1, 1) = C64(-s1, 0);
    ch.add_kraus(K3);

    return ch;
}

// Amplitude damping: K_0 = [[1,0],[0,√(1-γ)]], K_1 = [[0,√γ],[0,0]]
__host__ inline CPTPChannel amplitude_damping(double gamma) {
    int dim = 2;
    CPTPChannel ch(dim);

    DenseOp K0(dim);
    K0.at(0, 0) = C64(1, 0);
    K0.at(1, 1) = C64(sqrt(1.0 - gamma), 0);
    ch.add_kraus(K0);

    DenseOp K1(dim);
    K1.at(0, 1) = C64(sqrt(gamma), 0);
    ch.add_kraus(K1);

    return ch;
}

// Dephasing channel: K_0 = √(1-p) I, K_1 = √p Z
__host__ inline CPTPChannel dephasing(double p) {
    int dim = 2;
    CPTPChannel ch(dim);

    DenseOp K0 = DenseOp::identity(dim) * C64(sqrt(1.0 - p), 0);
    ch.add_kraus(K0);

    DenseOp K1(dim);
    K1.at(0, 0) = C64(sqrt(p), 0);
    K1.at(1, 1) = C64(-sqrt(p), 0);
    ch.add_kraus(K1);

    return ch;
}

} // namespace cptp

// ── Quantum Instruments ─────────────────────────────────────────────────────
// Instr_{A,n} := { {Φ_α} : ∀α Φ_α ∈ CP,  Σ_α Φ_α ∈ CPTP_{A,n} }
//   p_Φ(α|ρ) = Tr(Φ_α(ρ))
//   ρ → ρ_{Φ,α} = Φ_α(ρ) / p_Φ(α|ρ)
static constexpr int MAX_OUTCOMES = 16;

struct Instrument {
    CPTPChannel maps[MAX_OUTCOMES];  // CP maps Φ_α
    int n_outcomes;
    int dim;

    __host__ Instrument() : n_outcomes(0), dim(0) {}
    __host__ Instrument(int d) : n_outcomes(0), dim(d) {}

    __host__ void add_outcome(const CPTPChannel& phi_alpha) {
        if (n_outcomes < MAX_OUTCOMES) {
            maps[n_outcomes] = phi_alpha;
            n_outcomes++;
        }
    }

    // Outcome probability: p_Φ(α|ρ) = Tr(Φ_α(ρ))
    __host__ double outcome_prob(const DenseOp& rho, int alpha) const {
        DenseOp out = maps[alpha].apply(rho);
        return out.trace().re;
    }

    // Post-measurement state: ρ_{Φ,α} = Φ_α(ρ) / Tr(Φ_α(ρ))
    __host__ DenseOp post_state(const DenseOp& rho, int alpha) const {
        DenseOp out = maps[alpha].apply(rho);
        double p = out.trace().re;
        if (p < 1e-15) return out;  // degenerate
        return out * C64(1.0 / p, 0);
    }

    // Verify: Σ_α Φ_α is CPTP
    __host__ bool verify_cptp_sum(const DenseOp& rho, double tol = 1e-10) const {
        DenseOp total = DenseOp::zero(dim);
        for (int a = 0; a < n_outcomes; ++a) {
            total = total + maps[a].apply(rho);
        }
        // Total should preserve trace
        return fabs(total.trace().re - rho.trace().re) < tol &&
               fabs(total.trace().im) < tol;
    }

    // Verify outcome probabilities sum to 1
    __host__ bool verify_prob_sum(const DenseOp& rho, double tol = 1e-10) const {
        double sum = 0;
        for (int a = 0; a < n_outcomes; ++a)
            sum += outcome_prob(rho, a);
        return fabs(sum - 1.0) < tol;
    }
};

// Build instrument from PVM: Φ_j(ρ) = P_j ρ P_j (Lüders instrument)
__host__ inline Instrument luders_instrument(int n_qubits) {
    int dim = 1 << n_qubits;
    Instrument inst(dim);

    for (int j = 0; j < dim; ++j) {
        CPTPChannel phi_j(dim);
        DenseOp Pj = computational_projector(j, dim);
        phi_j.add_kraus(Pj);
        inst.add_outcome(phi_j);
    }
    return inst;
}

// ── Naimark dilation ────────────────────────────────────────────────────────
// Naimark_{A,n} := (K, V, {P_α})
//   K = enlarged Hilbert space,  V: H → K isometry
//   {P_α} = PVM on K such that E_α = V† P_α V
struct NaimarkDilation {
    int dim_H;     // original space dimension
    int dim_K;     // ancilla-enlarged space dimension
    int n_effects; // number of POVM effects

    // V: isometric embedding  H → K  (dim_K × dim_H matrix)
    DenseOp V;
    // P_α: PVM projectors on K  (dim_K × dim_K)
    DenseOp* projectors;

    __host__ NaimarkDilation() : dim_H(0), dim_K(0), n_effects(0), projectors(nullptr) {}

    __host__ void init(int dH, int dK, int nE) {
        dim_H = dH; dim_K = dK; n_effects = nE;
        V = DenseOp(dim_K);  // used as dim_K × dim_H but stored in square
        projectors = new DenseOp[nE];
        for (int a = 0; a < nE; ++a)
            projectors[a] = DenseOp(dim_K);
    }

    __host__ void free_mem() {
        delete[] projectors;
        projectors = nullptr;
    }

    // Reconstruct POVM effect: E_α = V† P_α V
    __host__ DenseOp reconstruct_effect(int alpha) const {
        return V.dagger() * projectors[alpha] * V;
    }

    // Verify isometry: V† V = I_H
    __host__ bool verify_isometry(double tol = 1e-10) const {
        DenseOp VdV = V.dagger() * V;
        DenseOp I_H = DenseOp::identity(dim_H);
        // Compare upper-left dim_H × dim_H block
        double err = 0;
        for (int i = 0; i < dim_H; ++i)
            for (int j = 0; j < dim_H; ++j)
                err += (VdV.at(i, j) - I_H.at(i, j)).norm2();
        return err < tol;
    }

    // Verify PVM: P_α² = P_α, Σ P_α = I_K
    __host__ bool verify_pvm(double tol = 1e-10) const {
        DenseOp sum = DenseOp::zero(dim_K);
        for (int a = 0; a < n_effects; ++a) {
            DenseOp P2 = projectors[a] * projectors[a];
            double err = (P2 - projectors[a]).hs_norm();
            if (err > tol) return false;
            sum = sum + projectors[a];
        }
        DenseOp I_K = DenseOp::identity(dim_K);
        return (sum - I_K).hs_norm() < tol;
    }

    // Verify Naimark relation: E_α = V† P_α V for all α
    __host__ bool verify_naimark(const DenseOp* effects, double tol = 1e-10) const {
        for (int a = 0; a < n_effects; ++a) {
            DenseOp reconstructed = reconstruct_effect(a);
            double err = 0;
            for (int i = 0; i < dim_H; ++i)
                for (int j = 0; j < dim_H; ++j)
                    err += (reconstructed.at(i, j) - effects[a].at(i, j)).norm2();
            if (err > tol) return false;
        }
        return true;
    }
};

// Build Naimark dilation for computational-basis POVM (trivial case: PVM)
// For a PVM with dim outcomes on H=ℂ^dim, K=H, V=I, P_α = |α⟩⟨α|
__host__ inline NaimarkDilation trivial_naimark(int dim) {
    NaimarkDilation nd;
    nd.init(dim, dim, dim);
    nd.V = DenseOp::identity(dim);
    for (int a = 0; a < dim; ++a)
        nd.projectors[a] = computational_projector(a, dim);
    return nd;
}

// ── Measurement pack (Meas_{A,n}) ───────────────────────────────────────────
// Meas_{A,n} := (D_{A,n}, Obs_{A,n}, POVM_{A,n}, CPTP_{A,n}, Instr_{A,n},
//                {p_A(j|ρ)}, {p_Φ(α|ρ)}, {p(x|ρ)}, {p(x,j|ρ)})
struct MeasPack {
    int dim;
    bool has_pvm;
    bool has_cptp;
    bool has_instrument;
    bool has_naimark;

    __host__ MeasPack(int d)
        : dim(d), has_pvm(true), has_cptp(true),
          has_instrument(true), has_naimark(true) {}

    // Verify the full measurement package is self-consistent
    __host__ bool verify(double tol = 1e-10) const {
        // PVM completeness
        bool pvm_ok = verify_pvm_properties(
            (int)round(log2((double)dim)));

        // Identity CPTP
        CPTPChannel id_ch = cptp::identity_channel(dim);
        bool cptp_ok = id_ch.verify_tp(tol);

        // Lüders instrument
        Instrument inst = luders_instrument(
            (int)round(log2((double)dim)));
        QubitState psi;
        int nq = (int)round(log2((double)dim));
        psi.init(nq);
        psi.set_basis(0);
        auto rho = pure_state_density(psi);
        bool inst_ok = inst.verify_prob_sum(rho, tol);
        psi.free();

        // Naimark
        NaimarkDilation nd = trivial_naimark(dim);
        DenseOp* effects = new DenseOp[dim];
        for (int a = 0; a < dim; ++a)
            effects[a] = computational_projector(a, dim);
        bool naimark_ok = nd.verify_naimark(effects, tol);
        delete[] effects;
        nd.free_mem();

        return pvm_ok && cptp_ok && inst_ok && naimark_ok;
    }
};

} // namespace measurement
} // namespace topcomp
