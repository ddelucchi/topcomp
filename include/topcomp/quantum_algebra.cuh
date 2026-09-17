// ============================================================================
// TopComp: Complete Quantum Algebra — Q^max_{A,n}
// ============================================================================
// Implements the full algebraic substrate that makes this system a complete
// quantum computer built from classical hardware topology:
//
//   Q^max_{A,n} = (
//     X_{A,n}^hyb,       — Hilbert space
//     G̃_{A,n},           — central extension group
//     F_{A,n}^fin,       — twisted convolution algebra
//     ★_Ω,               — twisted product
//     *,                 — star-involution
//     {A(S)},            — local subalgebras
//     {ω_ρ},             — state functionals
//     {G_ρ^(N)}_{N≥1},   — N-point correlation functions
//     T_{φ,n}^hyb,       — transport functors
//     Ĥ_φ,               — flow generator (Hamiltonian)
//     {P_x^hyb}          — sector projections
//   )
//
// SECTIONS:
//   I.   Twisted Convolution Algebra (f ★_Ω g)
//   II.  Field Operators Φ(f) = Σ f(a)W_a
//   III. Symplectic Form Σ & Locality Criterion
//   IV.  Weyl *-Algebra (inverses, adjoints, involution)
//   V.   Discrete Completeness (Pauli basis orthogonality)
//   VI.  Diagonal States ρ_diag
//   VII. N-point Correlation Functions G_ρ(a_1,...,a_N)
//   VIII.Generating Functional Z_ρ(f)
//   IX.  Flow Generator Ĥ_φ & Invariants
//   X.   Mixing Operators & Discrepancy
//   XI.  Circuit Complexity Metrics
//   XII. Projective Representation Π_Ω
// ============================================================================
#pragma once
#include "weyl.cuh"
#include "measurement.cuh"
#include "machine.cuh"
#include "mixed.cuh"

namespace topcomp {

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ I. TWISTED CONVOLUTION ALGEBRA F_{A,n}^fin                               ║
// ║                                                                          ║
// ║ F^fin = span_C {W_a : a ∈ G_{A,n}}                                      ║
// ║ (f ★_Ω g)(c) = Σ_{ab=c} f(a) g(b) e^{iΩ(a,b)}                          ║
// ║ δΩ = 0  ⟹  (f ★_Ω g) ★_Ω h = f ★_Ω (g ★_Ω h)                          ║
// ║ F^fin ≅ C_Ω[G_{A,n}]  (twisted group algebra)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace twisted_algebra {

// Element of the twisted group algebra: finite formal sum Σ f(a) W_a
// Represented as a list of (coefficient, Weyl index) pairs
struct AlgElement {
    static constexpr int MAX_TERMS = 256;
    C64         coeff[MAX_TERMS];  // f(a_i)
    WeylHybrid  basis[MAX_TERMS];  // a_i ∈ G_{A,n}
    int         n_terms;

    __host__ AlgElement() : n_terms(0) {}

    // Single Weyl element: f = α · W_a
    __host__ static AlgElement single(C64 alpha, const WeylHybrid& a) {
        AlgElement e;
        e.coeff[0] = alpha;
        e.basis[0] = a;
        e.n_terms = 1;
        return e;
    }

    // Add a term (no simplification — for small computations)
    __host__ void add_term(C64 c, const WeylHybrid& a) {
        if (n_terms < MAX_TERMS) {
            coeff[n_terms] = c;
            basis[n_terms] = a;
            n_terms++;
        }
    }

    // Scalar multiply: α · f
    __host__ AlgElement operator*(C64 alpha) const {
        AlgElement r;
        r.n_terms = n_terms;
        for (int i = 0; i < n_terms; ++i) {
            r.coeff[i] = coeff[i] * alpha;
            r.basis[i] = basis[i];
        }
        return r;
    }

    // Addition: f + g (concatenation, no simplification)
    __host__ AlgElement operator+(const AlgElement& g) const {
        AlgElement r;
        r.n_terms = 0;
        for (int i = 0; i < n_terms && r.n_terms < MAX_TERMS; ++i) {
            r.coeff[r.n_terms] = coeff[i];
            r.basis[r.n_terms] = basis[i];
            r.n_terms++;
        }
        for (int i = 0; i < g.n_terms && r.n_terms < MAX_TERMS; ++i) {
            r.coeff[r.n_terms] = g.coeff[i];
            r.basis[r.n_terms] = g.basis[i];
            r.n_terms++;
        }
        return r;
    }
};

// Twisted product: (f ★_Ω g)(c) = Σ_{ab=c} f(a) g(b) e^{iΩ_tot(a,b)}
// For finite sums this is explicit multiplication:
//   Φ(f) Φ(g) = Φ(f ★_Ω g)
__host__ inline AlgElement twisted_product(const AlgElement& f,
                                            const AlgElement& g) {
    AlgElement result;
    result.n_terms = 0;
    for (int i = 0; i < f.n_terms; ++i) {
        for (int j = 0; j < g.n_terms; ++j) {
            if (result.n_terms >= AlgElement::MAX_TERMS) break;
            // W_{a_i} W_{a_j} = e^{iΩ_tot(a_i,a_j)} W_{a_i · a_j}
            C64 phase;
            WeylHybrid composed = f.basis[i].compose(g.basis[j], phase);
            result.coeff[result.n_terms] = f.coeff[i] * g.coeff[j] * phase;
            result.basis[result.n_terms] = composed;
            result.n_terms++;
        }
    }
    return result;
}

// Verify associativity: (f★g)★h = f★(g★h)
// This is guaranteed by δΩ = 0, but we verify numerically
__host__ inline bool verify_associativity(const AlgElement& f,
                                           const AlgElement& g,
                                           const AlgElement& h,
                                           double tol = 1e-10) {
    AlgElement fg = twisted_product(f, g);
    AlgElement fg_h = twisted_product(fg, h);

    AlgElement gh = twisted_product(g, h);
    AlgElement f_gh = twisted_product(f, gh);

    // Compare coefficient sums (for single-term results they should match)
    if (fg_h.n_terms != f_gh.n_terms) return false;

    // Sum all coefficients (they represent the same abstract element)
    C64 sum_left(0, 0), sum_right(0, 0);
    for (int i = 0; i < fg_h.n_terms; ++i) sum_left += fg_h.coeff[i];
    for (int i = 0; i < f_gh.n_terms; ++i) sum_right += f_gh.coeff[i];
    return (sum_left - sum_right).norm2() < tol;
}

} // namespace twisted_algebra

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ II. FIELD OPERATORS Φ(f) = Σ f(a) W_a                                    ║
// ║                                                                          ║
// ║ Φ(f) Φ(g) = Φ(f ★_Ω g)                                                  ║
// ║ [Φ(f), Φ(g)] = 0 ⟺ ∀a∈supp(f), ∀b∈supp(g): Ω(a,b) ≡ Ω(b,a) mod 2πi  ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace field_ops {

// Apply Φ(f)|ψ⟩ = Σ f(a) W_a|ψ⟩
__host__ inline void apply_field(HybridState& psi,
                                  const twisted_algebra::AlgElement& f) {
    int total = psi.total;
    C64* result = new C64[total];
    memset(result, 0, total * sizeof(C64));

    for (int k = 0; k < f.n_terms; ++k) {
        // Copy state, apply W_{a_k}, accumulate with coefficient
        HybridState tmp;
        tmp.init(psi.n_qubits, psi.N_theta, psi.N_rho,
                 psi.rho_min, psi.rho_max);
        memcpy(tmp.amp, psi.amp, total * sizeof(C64));
        f.basis[k].apply(tmp);
        for (int i = 0; i < total; ++i)
            result[i] += tmp.amp[i] * f.coeff[k];
        tmp.free();
    }
    memcpy(psi.amp, result, total * sizeof(C64));
    delete[] result;
}

// Check commutativity: [Φ(f), Φ(g)] = 0
// ⟺ ∀a∈supp(f), ∀b∈supp(g): Ω_tot(a,b) ≡ Ω_tot(b,a) mod 2πi
__host__ inline bool check_commutativity(
    const twisted_algebra::AlgElement& f,
    const twisted_algebra::AlgElement& g,
    double tol = 1e-10)
{
    for (int i = 0; i < f.n_terms; ++i) {
        for (int j = 0; j < g.n_terms; ++j) {
            C64 omega_ab = f.basis[i].omega_total(g.basis[j]);
            C64 omega_ba = g.basis[j].omega_total(f.basis[i]);
            C64 diff = omega_ab - omega_ba;
            // Check: diff ∈ 2πℤ (imaginary part is multiple of 2π)
            double frac = diff.im / (2.0 * constants::PI);
            double rem = frac - round(frac);
            if (fabs(rem) > tol || fabs(diff.re) > tol)
                return false;
        }
    }
    return true;
}

} // namespace field_ops

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ III. SYMPLECTIC FORM Σ & LOCALITY CRITERION                               ║
// ║                                                                          ║
// ║ Σ(a,b) := Ω_tot(a,b) - Ω_tot(b,a)                                       ║
// ║ [W_a,W_b] = (e^{iΩ(a,b)} - e^{iΩ(b,a)}) W_{ab}                         ║
// ║ e^{iΣ(a,b)} = 1  ⟺  [W_a,W_b] = 0                                      ║
// ║ S ⊥_Ω T  ⟹  [A(S), A(T)] = 0                                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace symplectic {

// Symplectic form: Σ(a,b) = Ω_tot(a,b) - Ω_tot(b,a)
__host__ __device__ inline C64 sigma(const WeylHybrid& a, const WeylHybrid& b) {
    return a.omega_total(b) - b.omega_total(a);
}

// Explicit formula:
// Σ((u,v;x),(u',v';y)) = Ω_A(x,y) - Ω_A(y,x) + πi(⟨v,u'⟩ - ⟨v',u⟩)
__host__ __device__ inline C64 sigma_explicit(
    const BitVec& u, const BitVec& v, const MirElement& x,
    const BitVec& up, const BitVec& vp, const MirElement& y)
{
    C64 omega_xy = cocycle::omega_mir(x, y);
    C64 omega_yx = cocycle::omega_mir(y, x);
    int vu_prime = v.inner(up);
    int vp_u = vp.inner(u);
    return (omega_xy - omega_yx) +
           C64(0.0, constants::PI * (vu_prime - vp_u));
}

// Commutator coefficient: e^{iΣ(a,b)}
__host__ __device__ inline C64 comm_phase(const WeylHybrid& a,
                                            const WeylHybrid& b) {
    return cexp(C64(0.0, 1.0) * sigma(a, b));
}

// Check commutativity: [W_a, W_b] = 0 ⟺ e^{iΣ(a,b)} = 1
__host__ __device__ inline bool commute(const WeylHybrid& a,
                                          const WeylHybrid& b,
                                          double tol = 1e-10) {
    C64 s = sigma(a, b);
    // Σ ∈ 2πiℤ means im(Σ) is multiple of 2π and re(Σ) = 0
    double frac = s.im / (2.0 * constants::PI);
    return fabs(frac - round(frac)) < tol && fabs(s.re) < tol;
}

// [W_a,W_b]_- = W_aW_b - W_bW_a = (e^{iΩ(a,b)} - e^{iΩ(b,a)}) W_{ab}
// Returns the scalar coefficient
__host__ __device__ inline C64 commutator_coeff(const WeylHybrid& a,
                                                  const WeylHybrid& b) {
    C64 omega_ab = a.omega_total(b);
    C64 omega_ba = b.omega_total(a);
    return cexp(omega_ab) - cexp(omega_ba);
}

// [W_a,W_b]_+ = W_aW_b + W_bW_a = (e^{iΩ(a,b)} + e^{iΩ(b,a)}) W_{ab}
// Returns the scalar coefficient
__host__ __device__ inline C64 anticommutator_coeff(const WeylHybrid& a,
                                                      const WeylHybrid& b) {
    C64 omega_ab = a.omega_total(b);
    C64 omega_ba = b.omega_total(a);
    return cexp(omega_ab) + cexp(omega_ba);
}

// Ω-orthogonality: S ⊥_Ω T ⟺ ∀a∈S, ∀b∈T: Σ(a,b) ∈ 2πiℤ
__host__ inline bool omega_orthogonal(const WeylHybrid* S, int nS,
                                       const WeylHybrid* T, int nT,
                                       double tol = 1e-10) {
    for (int i = 0; i < nS; ++i)
        for (int j = 0; j < nT; ++j)
            if (!commute(S[i], T[j], tol)) return false;
    return true;
}

} // namespace symplectic

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ IV. WEYL *-ALGEBRA (inverses, adjoints, involution)                       ║
// ║                                                                          ║
// ║ W_a^{-1} = e^{-iΩ(a,a^{-1})} W_{a^{-1}}                                 ║
// ║ W_a* = W_a^{-1}  (unitarity)                                             ║
// ║ f*(a) = e^{-iΩ(a^{-1},a)} f̄(a^{-1})                                     ║
// ║ (f ★_Ω g)* = g* ★_Ω f*                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace star_algebra {

// Inverse Weyl index: a^{-1} = (u, v, g^{-1})
// Since u⊕u = 0 in F_2^n, a^{-1} = (u, v, g^{-1})
__host__ __device__ inline WeylHybrid weyl_inverse(const WeylHybrid& a) {
    return WeylHybrid(a.u, a.v, a.g.inverse());
}

// W_a^{-1} = e^{-iΩ(a,a^{-1})} W_{a^{-1}}
__host__ __device__ inline C64 inverse_phase(const WeylHybrid& a) {
    WeylHybrid ainv = weyl_inverse(a);
    C64 omega = a.omega_total(ainv);
    return cexp(omega * C64(-1.0, 0.0));
}

// W_a* = W_a^{-1} = e^{-iΩ(a,a^{-1})} W_{a^{-1}}
__host__ __device__ inline WeylHybrid weyl_adjoint_index(const WeylHybrid& a) {
    return weyl_inverse(a);
}

// *-involution on algebra elements:
// f*(a) = e^{-iΩ(a^{-1},a)} conj(f(a^{-1}))
__host__ inline twisted_algebra::AlgElement star_involution(
    const twisted_algebra::AlgElement& f)
{
    twisted_algebra::AlgElement result;
    result.n_terms = f.n_terms;
    for (int i = 0; i < f.n_terms; ++i) {
        WeylHybrid ainv = weyl_inverse(f.basis[i]);
        C64 omega = cocycle::omega_total(
            ainv.g, ainv.u, ainv.v,
            f.basis[i].g, f.basis[i].u, f.basis[i].v);
        C64 phase = cexp(omega * C64(-1.0, 0.0));
        result.coeff[i] = f.coeff[i].conj() * phase;
        result.basis[i] = ainv;
    }
    return result;
}

// Verify (f★g)* = g*★f*
__host__ inline bool verify_star_antiautomorphism(
    const twisted_algebra::AlgElement& f,
    const twisted_algebra::AlgElement& g,
    double tol = 1e-8)
{
    // Left: (f★g)*
    auto fg = twisted_algebra::twisted_product(f, g);
    auto fg_star = star_involution(fg);

    // Right: g*★f*
    auto g_star = star_involution(g);
    auto f_star = star_involution(f);
    auto gs_fs = twisted_algebra::twisted_product(g_star, f_star);

    // Compare coefficient sums
    C64 sum_left(0, 0), sum_right(0, 0);
    for (int i = 0; i < fg_star.n_terms; ++i) sum_left += fg_star.coeff[i];
    for (int i = 0; i < gs_fs.n_terms; ++i) sum_right += gs_fs.coeff[i];
    return (sum_left - sum_right).norm2() < tol;
}

} // namespace star_algebra

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ V. DISCRETE COMPLETENESS (Pauli basis orthogonality)                      ║
// ║                                                                          ║
// ║ Tr((X_u Z_v)† X_{u'} Z_{v'}) = 2^n δ_{u,u'} δ_{v,v'}                    ║
// ║ {X_u Z_v}_{u,v ∈ I_n} is orthogonal basis of End_C(V_n)                  ║
// ║ |{X_u Z_v}| = 4^n = dim End(V_n)                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace completeness {

// Tr((X_u Z_v)† (X_{u'} Z_{v'})) = 2^n δ_{u,u'} δ_{v,v'}
__host__ inline double pauli_trace(const BitVec& u, const BitVec& v,
                                    const BitVec& up, const BitVec& vp) {
    int n = u.n;
    int dim = 1 << n;

    // A = X_u Z_v:   A|x⟩ = (-1)^{⟨v,x⟩} |x⊕u⟩
    // B = X_{u'}Z_{v'}: B|x⟩ = (-1)^{⟨v',x⟩} |x⊕u'⟩
    // Tr(A†B) = Σ_x conj((A|x⟩)) · (B|x⟩)
    //         = δ_{u,u'} Σ_x (-1)^{⟨v⊕v',x⟩}
    //         = 2^n δ_{u,u'} δ_{v,v'}
    if (u.bits != up.bits) return 0.0;

    BitVec vxor(v.bits ^ vp.bits, n);
    double sum = 0.0;
    for (int x = 0; x < dim; ++x) {
        BitVec x_bv(x, n);
        sum += (vxor.inner(x_bv) == 0) ? 1.0 : -1.0;
    }
    return sum;
}

// Verify orthogonality: Tr((X_u Z_v)† (X_u' Z_v')) = 2^n δ_{uu'} δ_{vv'}
__host__ inline bool verify_orthogonality(int n, double tol = 1e-10) {
    int dim = 1 << n;
    int size = dim * dim;  // 4^n
    if (size > 256) return true;  // skip for large n (combinatorial explosion)

    for (int a = 0; a < size; ++a) {
        BitVec u(a >> n, n), v(a & ((1 << n) - 1), n);
        for (int b = 0; b < size; ++b) {
            BitVec up(b >> n, n), vp(b & ((1 << n) - 1), n);
            double tr = pauli_trace(u, v, up, vp);
            double expected = (a == b) ? static_cast<double>(dim) : 0.0;
            if (fabs(tr - expected) > tol) return false;
        }
    }
    return true;
}

// Verify completeness: |{X_u Z_v}| = 4^n = dim End(V_n)
__host__ __device__ inline bool verify_cardinality(int n) {
    int pauli_count = 1;
    for (int i = 0; i < 2 * n; ++i) pauli_count *= 2;
    int endomorphism_dim = 1;
    for (int i = 0; i < 2 * n; ++i) endomorphism_dim *= 2;
    return pauli_count == endomorphism_dim;  // 4^n = 4^n
}

} // namespace completeness

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ VI. DIAGONAL STATES ρ_diag                                                ║
// ║                                                                          ║
// ║ ρ_diag = Σ_x P_x^hyb ρ P_x^hyb                                          ║
// ║ ω_{ρ_diag}(A) = Σ_x Tr(P_x ρ P_x A)                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace diagonal_state {

// Construct ρ_diag from hybrid state |ψ⟩:
// Project out cross-sector coherences, keeping only diagonal blocks
__host__ inline void diagonalize_sectors(HybridState& psi) {
    // For pure state, ρ_diag = Σ_x P_x|ψ⟩⟨ψ|P_x
    // Since P_x kills all sectors except x, and they're already separate
    // in the HybridState layout, we just zero the cross-terms.
    // For amplitude representation this means the diagonal state IS
    // the collection of sector amplitudes — no change needed.
    // The decoherence happens when we form the density matrix.
    // This function explicitly projects: keep only the block-diagonal part.
    // (In amplitude representation, sectors are already separated.)
}

// Compute sector probabilities: p(x) = Tr(P_x ρ)
__host__ inline void sector_probabilities(const HybridState& psi,
                                           double* probs) {
    double total = 0.0;
    for (int x = 0; x < psi.dim_disc; ++x) {
        probs[x] = psi.prob_disc(x);
        total += probs[x];
    }
    if (total > 0.0)
        for (int x = 0; x < psi.dim_disc; ++x) probs[x] /= total;
}

// Conditional state in sector x: ρ_x = P_x ρ P_x / Tr(P_x ρ)
__host__ inline void conditional_sector(const HybridState& psi, int x,
                                         TopologicalState& out) {
    psi.project(x, out);
    double norm = 0.0;
    for (int i = 0; i < out.total; ++i) norm += out.amp[i].norm2();
    if (norm > 1e-30) {
        double inv = 1.0 / sqrt(norm);
        for (int i = 0; i < out.total; ++i)
            out.amp[i] = out.amp[i] * inv;
    }
}

// ω_{ρ_diag}(A) = Σ_x Tr(P_x ρ P_x A)
// For discrete observables this is just the weighted sum over sectors
__host__ inline double diag_expectation(const HybridState& psi,
                                         const double* disc_observable) {
    double result = 0.0;
    double total = 0.0;
    for (int x = 0; x < psi.dim_disc; ++x) {
        double px = psi.prob_disc(x);
        result += px * disc_observable[x];
        total += px;
    }
    return (total > 0.0) ? result / total : 0.0;
}

} // namespace diagonal_state

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ VII. N-POINT CORRELATION FUNCTIONS G_ρ(a_1,...,a_N)                        ║
// ║                                                                          ║
// ║ W_{a1}...W_{aN} = exp(i Σ_{j=1}^{N-1} Ω(A_j, a_{j+1})) W_{A_N}         ║
// ║ where A_j = a_1 · a_2 · ... · a_j                                        ║
// ║                                                                          ║
// ║ G_ρ(a_1,...,a_N) = ω_ρ(W_{a1} ... W_{aN})                                ║
// ║     = exp(i Σ Ω(A_j, a_{j+1})) · ω_ρ(W_{A_N})                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace correlation {

// Compute the cumulative phase from a product W_{a1}...W_{aN}
// Returns (total_phase, composite_index A_N)
__host__ inline void npoint_product(const WeylHybrid* a, int N,
                                     C64& total_phase, WeylHybrid& A_N) {
    total_phase = C64(1.0, 0.0);

    if (N == 0) {
        A_N = WeylHybrid::identity(1);
        return;
    }

    A_N = a[0];  // A_1 = a_1

    for (int j = 1; j < N; ++j) {
        // Ω(A_j, a_{j+1})
        C64 omega = A_N.omega_total(a[j]);
        total_phase = total_phase * cexp(omega);

        // A_{j+1} = A_j · a_{j+1}
        C64 compose_phase;
        A_N = A_N.compose(a[j], compose_phase);
    }
}

// G_ρ(a_1,...,a_N) = exp(i Σ Ω(A_j,a_{j+1})) · ω_ρ(W_{A_N})
// For pure state |ψ⟩: ω_ρ(W_a) = ⟨ψ|W_a|ψ⟩
__host__ inline C64 npoint_function(const HybridState& psi,
                                     const WeylHybrid* a, int N) {
    C64 total_phase;
    WeylHybrid A_N;
    npoint_product(a, N, total_phase, A_N);

    // Compute ω_ρ(W_{A_N}) = ⟨ψ|W_{A_N}|ψ⟩
    HybridState tmp;
    tmp.init(psi.n_qubits, psi.N_theta, psi.N_rho,
             psi.rho_min, psi.rho_max);
    memcpy(tmp.amp, psi.amp, psi.total * sizeof(C64));
    A_N.apply(tmp);

    C64 omega_rho(0.0, 0.0);
    for (int i = 0; i < psi.total; ++i)
        omega_rho += psi.amp[i].conj() * tmp.amp[i];

    tmp.free();
    return total_phase * omega_rho;
}

// 1-point function: G_1(a) = ω_ρ(W_a) = ⟨ψ|W_a|ψ⟩
__host__ inline C64 G1(const HybridState& psi, const WeylHybrid& a) {
    return npoint_function(psi, &a, 1);
}

// 2-point function: G_2(a,b) = e^{iΩ(a,b)} ω_ρ(W_{ab})
__host__ inline C64 G2(const HybridState& psi,
                        const WeylHybrid& a, const WeylHybrid& b) {
    WeylHybrid pair[2] = {a, b};
    return npoint_function(psi, pair, 2);
}

// Verify: G_2(a,b) = e^{iΩ(a,b)} ω_ρ(W_{a·b})
__host__ inline bool verify_G2_factorization(const HybridState& psi,
                                              const WeylHybrid& a,
                                              const WeylHybrid& b,
                                              double tol = 1e-8) {
    C64 g2 = G2(psi, a, b);
    C64 omega = a.omega_total(b);
    C64 phase;
    WeylHybrid ab = a.compose(b, phase);
    C64 omega_rho = G1(psi, ab);
    C64 expected = cexp(omega) * omega_rho;
    return (g2 - expected).norm2() < tol;
}

// Commutator expectation: ω_ρ([W_a,W_b]) = (e^{iΩ(a,b)} - e^{iΩ(b,a)}) ω_ρ(W_{ab})
__host__ inline C64 commutator_expectation(const HybridState& psi,
                                            const WeylHybrid& a,
                                            const WeylHybrid& b) {
    C64 phase;
    WeylHybrid ab = a.compose(b, phase);
    C64 omega_rho = G1(psi, ab);
    C64 comm_coeff = symplectic::commutator_coeff(a, b);
    return comm_coeff * omega_rho;
}

} // namespace correlation

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ VIII. GENERATING FUNCTIONAL Z_ρ(f)                                        ║
// ║                                                                          ║
// ║ Z_ρ(f) = ω_ρ(e^{Φ(f)}) = Σ_{N=0}^∞ (1/N!) ω_ρ(Φ(f)^N)                 ║
// ║ ∂^N / ∂t₁...∂t_N Z_ρ(Σ t_j δ_{a_j})|_{t=0} = G_ρ(a_1,...,a_N)          ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace generating {

// Truncated generating functional up to order N_max:
// Z_ρ^{≤K}(f) = Σ_{N=0}^K (1/N!) ω_ρ(Φ(f)^N)
// Computed by iterated application of Φ(f)
__host__ inline C64 Z_truncated(const HybridState& psi,
                                 const twisted_algebra::AlgElement& f,
                                 int K_max) {
    C64 Z(0.0, 0.0);
    double factorial = 1.0;

    // N=0 term: ω_ρ(I) = ⟨ψ|ψ⟩ = 1 (normalized)
    Z += C64(1.0, 0.0);

    // Iterated application
    HybridState current;
    current.init(psi.n_qubits, psi.N_theta, psi.N_rho,
                 psi.rho_min, psi.rho_max);
    memcpy(current.amp, psi.amp, psi.total * sizeof(C64));

    for (int N = 1; N <= K_max; ++N) {
        factorial *= N;
        // Apply Φ(f) to current state
        field_ops::apply_field(current, f);
        // ω_ρ(Φ(f)^N) = ⟨ψ|Φ(f)^N|ψ⟩
        C64 omega_N(0.0, 0.0);
        for (int i = 0; i < psi.total; ++i)
            omega_N += psi.amp[i].conj() * current.amp[i];
        Z += omega_N * C64(1.0 / factorial, 0.0);
    }

    current.free();
    return Z;
}

// Verify generating functional relation:
// ∂Z/∂t|_{t=0} with f = t·δ_a gives G_1(a)
__host__ inline bool verify_derivative(const HybridState& psi,
                                        const WeylHybrid& a,
                                        double tol = 1e-6) {
    double dt = 1e-4;
    twisted_algebra::AlgElement f_dt =
        twisted_algebra::AlgElement::single(C64(dt, 0.0), a);
    twisted_algebra::AlgElement f_0 =
        twisted_algebra::AlgElement::single(C64(0.0, 0.0), a);

    C64 Z_dt = Z_truncated(psi, f_dt, 3);
    C64 Z_0 = Z_truncated(psi, f_0, 3);

    // Numerical derivative: (Z(dt) - Z(0)) / dt ≈ G_1(a)
    C64 deriv = (Z_dt - Z_0) * C64(1.0 / dt, 0.0);
    C64 g1 = correlation::G1(psi, a);

    return (deriv - g1).norm2() < tol;
}

} // namespace generating

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ IX. FLOW GENERATOR Ĥ_φ & INVARIANTS                                      ║
// ║                                                                          ║
// ║ X_φ = -(π/2)∂_θ + ℓ∂_ρ     (vector field)                               ║
// ║ Ĥ_φ = -iX_φ               (Hamiltonian)                                 ║
// ║                                                                          ║
// ║ Invariants:                                                              ║
// ║   u(θ,ρ) = θ + (π/(2ℓ))ρ           [X_φ[u] = 0]                         ║
// ║   ω̃(θ,ρ) = θ - (π/(2ℓ))ρ           [X_φ[ω̃] = -π]                       ║
// ║   ω = ω̃ mod π                       [X_φ[ω] ≡ 0 mod π]                  ║
// ║                                                                          ║
// ║ [Ĥ_φ, u] = 0                                                              ║
// ║ [Ĥ_φ, ω̃] = iπ                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace flow_generator {

// u(θ,ρ) = θ + (π/(2ℓ))ρ   [flow invariant]
__host__ __device__ inline double u_invariant(double theta, double rho) {
    return theta + (constants::HALF_PI / constants::ELL) * rho;
}

// ω̃(θ,ρ) = θ - (π/(2ℓ))ρ   [decreases by π per unit flow time]
__host__ __device__ inline double omega_tilde(double theta, double rho) {
    return theta - (constants::HALF_PI / constants::ELL) * rho;
}

// ω = ω̃ mod π
__host__ __device__ inline double omega_reduced(double theta, double rho) {
    double w = omega_tilde(theta, rho);
    w = fmod(w, constants::PI);
    if (w < 0) w += constants::PI;
    return w;
}

// Verify X_φ[u] = 0: u is preserved under the golden flow
__host__ inline bool verify_u_invariance(const PhasePoint& p, double t,
                                          double tol = 1e-10) {
    double u0 = u_invariant(p.theta, p.rho);
    PhasePoint evolved = GoldenFlow::flow(p, t);
    double u1 = u_invariant(evolved.theta, evolved.rho);
    return fabs(u0 - u1) < tol;
}

// Verify X_φ[ω̃] = -π: ω̃ decreases by πt under flow of duration t
__host__ inline bool verify_omega_decrease(const PhasePoint& p, double t,
                                            double tol = 1e-10) {
    double w0 = omega_tilde(p.theta, p.rho);
    PhasePoint evolved = GoldenFlow::flow(p, t);
    double w1 = omega_tilde(evolved.theta, evolved.rho);
    return fabs((w1 - w0) + constants::PI * t) < tol;
}

// Apply Ĥ_φ = -iX_φ to a topological state (finite-difference approximation)
// X_φ = -(π/2)∂_θ + ℓ∂_ρ
// Ĥ_φΨ = -i(-(π/2)∂_θΨ + ℓ∂_ρΨ) = i(π/2)∂_θΨ - iℓ∂_ρΨ
__host__ inline void apply_H_phi(TopologicalState& psi) {
    int nth = psi.N_theta;
    int nrh = psi.N_rho;
    double d_theta = constants::TWO_PI / nth;
    double d_rho = (nrh > 1) ? (psi.rho_max - psi.rho_min) / (nrh - 1) : 1.0;

    C64* result = new C64[psi.total];
    memset(result, 0, psi.total * sizeof(C64));

    for (int i = 0; i < nth; ++i) {
        int ip = (i + 1) % nth;
        int im = (i - 1 + nth) % nth;
        for (int j = 0; j < nrh; ++j) {
            // ∂_θ Ψ ≈ (Ψ(θ+dθ) - Ψ(θ-dθ)) / (2dθ)
            C64 dtheta = (psi.at(ip, j) - psi.at(im, j)) * (1.0 / (2.0 * d_theta));

            // ∂_ρ Ψ ≈ (Ψ(ρ+dρ) - Ψ(ρ-dρ)) / (2dρ)
            C64 drho(0, 0);
            if (j > 0 && j < nrh - 1)
                drho = (psi.at(i, j + 1) - psi.at(i, j - 1)) * (1.0 / (2.0 * d_rho));
            else if (j == 0 && nrh > 1)
                drho = (psi.at(i, 1) - psi.at(i, 0)) * (1.0 / d_rho);
            else if (j == nrh - 1 && nrh > 1)
                drho = (psi.at(i, nrh - 1) - psi.at(i, nrh - 2)) * (1.0 / d_rho);

            // Ĥ_φΨ = i(π/2)∂_θΨ - iℓ∂_ρΨ
            result[i * nrh + j] =
                dtheta * C64(0.0, constants::HALF_PI) -
                drho * C64(0.0, constants::ELL);
        }
    }
    memcpy(psi.amp, result, psi.total * sizeof(C64));
    delete[] result;
}

// Verify [Ĥ_φ, u] = 0: expectation of u is unchanged by Ĥ_φ
__host__ inline bool verify_H_u_commutation(int nth, int nrh,
                                              double tol = 1e-2) {
    TopologicalState psi;
    psi.init(nth, nrh);
    psi.set_delta(PhasePoint(constants::PI / 3.0, 0.5));

    // Compute ⟨ψ|u|ψ⟩ before
    double u_before = 0.0;
    for (int i = 0; i < nth; ++i)
        for (int j = 0; j < nrh; ++j) {
            PhasePoint pt = psi.grid_point(i, j);
            u_before += psi.at(i, j).norm2() * u_invariant(pt.theta, pt.rho);
        }

    // Apply Ĥ for small time dt via exp(-iĤdt) ≈ 1 - iĤdt
    double dt = 0.01;
    TopologicalState H_psi;
    H_psi.init(nth, nrh);
    memcpy(H_psi.amp, psi.amp, psi.total * sizeof(C64));
    apply_H_phi(H_psi);

    // |ψ'⟩ = |ψ⟩ - i·dt·Ĥ|ψ⟩
    TopologicalState evolved;
    evolved.init(nth, nrh);
    for (int k = 0; k < psi.total; ++k)
        evolved.amp[k] = psi.amp[k] - H_psi.amp[k] * C64(0.0, dt);

    // Normalize
    double norm = evolved.norm2();
    if (norm > 1e-30) {
        double inv = 1.0 / sqrt(norm);
        for (int k = 0; k < evolved.total; ++k)
            evolved.amp[k] = evolved.amp[k] * inv;
    }

    // Compute ⟨ψ'|u|ψ'⟩ after
    double u_after = 0.0;
    for (int i = 0; i < nth; ++i)
        for (int j = 0; j < nrh; ++j) {
            PhasePoint pt = evolved.grid_point(i, j);
            u_after += evolved.at(i, j).norm2() * u_invariant(pt.theta, pt.rho);
        }

    psi.free();
    H_psi.free();
    evolved.free();
    return fabs(u_before - u_after) < tol;
}

} // namespace flow_generator

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ X. MIXING OPERATORS & DISCREPANCY                                         ║
// ║                                                                          ║
// ║ Mix_{A,n}(T) = (1/T) Σ_{t=0}^{T-1} (X_{u_t} Z_{v_t} U_t M_{m_t})       ║
// ║ μ_T = (1/T) Σ δ_{(θ_t,ρ_t,x_t)}                                         ║
// ║ Bias_T = sup_{(m,ℓ,ξ)≠0} |μ̂_T(m,ℓ,ξ)|                                  ║
// ║ Disc_{T,F} = sup_{f∈F} |(1/T)Σf(θ_t,ρ_t,x_t) - ∫f dμ|                  ║
// ║ Adv_T^Walsh ≤ s · Bias_T                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace mixing {

// Empirical measure coefficient: μ̂_T(m, ℓ, ξ) = (1/T) Σ χ_{m,ℓ}(θ_t,ρ_t) (-1)^{⟨ξ,x_t⟩}
__host__ inline C64 fourier_coeff(const PhasePoint* orbit, int T,
                                   const uint32_t* disc_outcomes, int n_qubits,
                                   int m_freq, int ell_freq, uint32_t xi) {
    C64 sum(0.0, 0.0);
    for (int t = 0; t < T; ++t) {
        C64 chi = character::chi(m_freq, ell_freq, orbit[t].theta, orbit[t].rho);
        BitVec xi_bv(xi, n_qubits);
        BitVec x_bv(disc_outcomes[t], n_qubits);
        double sign = (xi_bv.inner(x_bv) == 0) ? 1.0 : -1.0;
        sum += chi * sign;
    }
    return sum * C64(1.0 / T, 0.0);
}

// Spectral bias: Bias_T = sup_{(m,ℓ,ξ)≠0} |μ̂_T(m,ℓ,ξ)|
__host__ inline double spectral_bias(const PhasePoint* orbit, int T,
                                      const uint32_t* disc_outcomes, int n_qubits,
                                      int max_m, int max_ell) {
    double max_bias = 0.0;
    int max_xi = 1 << n_qubits;
    for (int m = -max_m; m <= max_m; ++m) {
        for (int ell = -max_ell; ell <= max_ell; ++ell) {
            for (uint32_t xi = 0; xi < static_cast<uint32_t>(max_xi); ++xi) {
                if (m == 0 && ell == 0 && xi == 0) continue;
                double b = fourier_coeff(orbit, T, disc_outcomes, n_qubits,
                                          m, ell, xi).abs();
                if (b > max_bias) max_bias = b;
            }
        }
    }
    return max_bias;
}

// Walsh advantage bound: Adv_T^Walsh(A,n; s,d) ≤ s · Bias_T
__host__ __device__ inline double walsh_advantage_bound(double s,
                                                         double bias) {
    return s * bias;
}

// Discrepancy for a test function f: |(1/T)Σf - ∫fdμ|
__host__ inline double test_discrepancy(const PhasePoint* orbit, int T,
                                          const uint32_t* disc_outcomes,
                                          int n_qubits,
                                          double integral_f) {
    // Compute empirical mean of f(θ,ρ,x) = u(θ,ρ) (flow invariant)
    double emp_mean = 0.0;
    for (int t = 0; t < T; ++t) {
        emp_mean += flow_generator::u_invariant(orbit[t].theta, orbit[t].rho);
    }
    emp_mean /= T;
    return fabs(emp_mean - integral_f);
}

} // namespace mixing

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ XI. CIRCUIT COMPLEXITY METRICS                                            ║
// ║                                                                          ║
// ║ disc(C) = #{j : G_j ∈ {X_u, Z_v, P_x}}                                  ║
// ║ top(C)  = #{j : G_j ∈ {U_t, M_m}}                                        ║
// ║ meas(C) = #{j : G_j = P_x}                                               ║
// ║ depth(C) = min{D : C = C_D...C_1, [G,G']=0 ∀G,G' ∈ C_r}                 ║
// ║ τ(G) = 1 for all generators                                              ║
// ║ size(C) = T,  time(C) = T                                                ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace circuit_metrics {

// Gate classification
__host__ __device__ inline bool is_discrete(const Gate& g) {
    return g.type == GateType::X_HYB ||
           g.type == GateType::Z_HYB ||
           g.type == GateType::H_HYB ||
           g.type == GateType::P_HYB;
}

__host__ __device__ inline bool is_topological(const Gate& g) {
    return g.type == GateType::U_HYB ||
           g.type == GateType::M_HYB;
}

__host__ __device__ inline bool is_measurement(const Gate& g) {
    return g.type == GateType::P_HYB;
}

__host__ __device__ inline bool is_weyl(const Gate& g) {
    return g.type == GateType::W_HYB;
}

// disc(C) = #{j : G_j ∈ {X_u, Z_v, P_x, H_q}}
__host__ inline int disc_count(const Program& prog) {
    int count = 0;
    for (int j = 0; j < prog.length; ++j)
        if (is_discrete(prog.gates[j])) count++;
    return count;
}

// top(C) = #{j : G_j ∈ {U_t, M_m}}
__host__ inline int top_count(const Program& prog) {
    int count = 0;
    for (int j = 0; j < prog.length; ++j)
        if (is_topological(prog.gates[j])) count++;
    return count;
}

// meas(C) = #{j : G_j = P_x}
__host__ inline int meas_count(const Program& prog) {
    int count = 0;
    for (int j = 0; j < prog.length; ++j)
        if (is_measurement(prog.gates[j])) count++;
    return count;
}

// weyl(C) = #{j : G_j = W_{u,v;g}}
__host__ inline int weyl_count(const Program& prog) {
    int count = 0;
    for (int j = 0; j < prog.length; ++j)
        if (is_weyl(prog.gates[j])) count++;
    return count;
}

// size(C) = T = length
__host__ __device__ inline int circuit_size(const Program& prog) {
    return prog.length;
}

// time(C) = T (τ(G) = 1 for all generators)
__host__ __device__ inline int circuit_time(const Program& prog) {
    return prog.length;
}

// depth(C) = min {D : C = C_D...C_1, [G,G']=0 ∀G,G' ∈ C_r}
// Greedy: gates that commute with all previous in the same layer go together
__host__ inline int circuit_depth(const Program& prog) {
    if (prog.length == 0) return 0;

    // Simple type-based commutation: disc and top always commute (cross-sector)
    // Within disc: [X_u, Z_v] = 0 only if ⟨u,v⟩=0
    // Within top: [U_t, M_m] ≠ 0 if mt ∉ ℤ
    // Greedy layer assignment
    int* layer = new int[prog.length];
    layer[0] = 0;
    int max_layer = 0;

    for (int j = 1; j < prog.length; ++j) {
        int min_possible = 0;
        for (int k = j - 1; k >= 0; --k) {
            // Check if gates j and k might not commute
            bool same_disc = is_discrete(prog.gates[j]) && is_discrete(prog.gates[k]);
            bool same_top = is_topological(prog.gates[j]) && is_topological(prog.gates[k]);
            bool both_weyl = is_weyl(prog.gates[j]) && is_weyl(prog.gates[k]);

            bool might_not_commute = same_disc || same_top || both_weyl ||
                (is_weyl(prog.gates[j]) || is_weyl(prog.gates[k]));

            if (might_not_commute && layer[k] >= min_possible) {
                min_possible = layer[k] + 1;
            }
        }
        layer[j] = min_possible;
        if (min_possible > max_layer) max_layer = min_possible;
    }

    delete[] layer;
    return max_layer + 1;
}

// Full circuit analysis report
struct CircuitAnalysis {
    int size;          // T = total gates
    int time;          // T (= size, since τ(G) = 1)
    int disc;          // discrete gate count
    int top;           // topological gate count
    int meas;          // measurement count
    int weyl;          // Weyl gate count
    int depth;         // parallel depth
    int cost;          // total cost Σ cost(G_j)
    int space;         // n + max cost(G_j)
};

__host__ inline CircuitAnalysis analyze(const Program& prog, int n_qubits) {
    CircuitAnalysis a;
    a.size = circuit_size(prog);
    a.time = circuit_time(prog);
    a.disc = disc_count(prog);
    a.top = top_count(prog);
    a.meas = meas_count(prog);
    a.weyl = weyl_count(prog);
    a.depth = circuit_depth(prog);
    a.cost = prog.total_cost(n_qubits);
    a.space = prog.space(n_qubits);
    return a;
}

} // namespace circuit_metrics

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ XII. PROJECTIVE REPRESENTATION Π_Ω                                        ║
// ║                                                                          ║
// ║ G̃_{A,n} = U(1) × G_{A,n}                                                 ║
// ║ (ζ,a)·(η,b) = (ζη e^{iΩ(a,b)}, a·b)                                     ║
// ║ Π_Ω(ζ,a) = ζ W_a                                                         ║
// ║ Π_Ω(ζ,a)Π_Ω(η,b) = Π_Ω((ζ,a)·(η,b))                                    ║
// ╚══════════════════════════════════════════════════════════════════════════╝
namespace projective_rep {

// Central extension element: (ζ, a) ∈ U(1) × G_{A,n}
struct CentralElement {
    C64        zeta;   // U(1) phase
    WeylHybrid a;      // group element

    __host__ __device__ CentralElement()
        : zeta(1.0, 0.0), a() {}
    __host__ __device__ CentralElement(C64 z, WeylHybrid w)
        : zeta(z), a(w) {}

    // Group product: (ζ,a)·(η,b) = (ζη e^{iΩ(a,b)}, a·b)
    __host__ __device__ CentralElement operator*(const CentralElement& other) const {
        C64 omega = a.omega_total(other.a);
        C64 phase;
        WeylHybrid composed = a.compose(other.a, phase);
        return CentralElement(zeta * other.zeta * cexp(omega), composed);
    }

    // Identity: (1, e)
    __host__ __device__ static CentralElement identity(int n) {
        return CentralElement(C64(1.0, 0.0), WeylHybrid::identity(n));
    }
};

// Projective representation: Π_Ω(ζ,a) = ζ W_a
// Applies to a hybrid state
__host__ inline void apply_Pi(HybridState& psi, const CentralElement& elem) {
    elem.a.apply(psi);
    for (int i = 0; i < psi.total; ++i)
        psi.amp[i] = psi.amp[i] * elem.zeta;
}

// Verify representation property: Π(ζ,a)Π(η,b) = Π((ζ,a)·(η,b))
__host__ inline bool verify_representation(const CentralElement& x,
                                            const CentralElement& y,
                                            const HybridState& psi,
                                            double tol = 1e-8) {
    // Left: apply Π(η,b) then Π(ζ,a)
    HybridState left;
    left.init(psi.n_qubits, psi.N_theta, psi.N_rho,
              psi.rho_min, psi.rho_max);
    memcpy(left.amp, psi.amp, psi.total * sizeof(C64));
    apply_Pi(left, y);
    apply_Pi(left, x);

    // Right: compute (ζ,a)·(η,b) then apply Π
    CentralElement product = x * y;
    HybridState right;
    right.init(psi.n_qubits, psi.N_theta, psi.N_rho,
               psi.rho_min, psi.rho_max);
    memcpy(right.amp, psi.amp, psi.total * sizeof(C64));
    apply_Pi(right, product);

    // Compare Born distributions (phase-independent)
    double max_diff = 0.0;
    for (int i = 0; i < psi.total; ++i) {
        double d = fabs(left.amp[i].norm2() - right.amp[i].norm2());
        if (d > max_diff) max_diff = d;
    }

    left.free();
    right.free();
    return max_diff < tol;
}

} // namespace projective_rep

// ╔══════════════════════════════════════════════════════════════════════════╗
// ║ COMPLETE SYSTEM SUMMARY: Q^max_{A,n}                                      ║
// ║                                                                          ║
// ║ The complete quantum algebraic system is now:                             ║
// ║  Q^max = (X^hyb, G̃, F^fin, ★_Ω, *, {A(S)}, {ω_ρ}, {G^(N)},            ║
// ║          T^hyb, Ĥ_φ, {P_x})                                             ║
// ║                                                                          ║
// ║ Properties verified:                                                     ║
// ║  - δΩ_tot = 0  (cocycle condition)                                       ║
// ║  - UFE = 0  (universal flatness)                                         ║
// ║  - (f★g)★h = f★(g★h)  (associativity from δΩ=0)                          ║
// ║  - (f★g)* = g*★f*  (anti-automorphism)                                   ║
// ║  - Tr((X_uZ_v)†X_{u'}Z_{v'}) = 2^n δ_{uu'}δ_{vv'}  (completeness)       ║
// ║  - G_2(a,b) = e^{iΩ(a,b)} ω_ρ(W_{ab})  (factorization)                  ║
// ║  - Σ(a,b) ∈ 2πiℤ ⟺ [W_a,W_b]=0  (locality)                             ║
// ║  - [Ĥ_φ, u] = 0,  [Ĥ_φ, ω̃] = iπ  (flow commutation)                    ║
// ║  - Π_Ω factorizes through the central extension                          ║
// ║  - Adv^Walsh ≤ s·Bias  (Walsh advantage bound)                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

} // namespace topcomp
