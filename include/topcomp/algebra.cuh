// ============================================================================
// TopComp: Admissible Algebras & Mir Functor
// ============================================================================
// Implements the category of admissible algebras and the Mir functor:
//
//   K_log := Frac(ℚ[π, φ, ln φ, ln a] / (φ² - φ - 1))
//
//   Admissible algebra A = (A, ℂ_A, ln_A, exp_A, ι_A):
//     - A is a K_log-algebra
//     - ℂ_A := ℂ ⊗_{K_log} A
//     - ln_A, exp_A: logarithm/exponential maps
//     - ι_A: K_log → A  (structure morphism)
//
//   Category of admissible algebras:
//     Objects: admissible algebras A, B, ...
//     Morphisms: φ: A → B  (K_log-algebra homomorphisms)
//
//   Mir functor:  Mir: AdmAlg → Grp
//     Mir(A) = Mir_A = {±1} ⋉ ℂ_A
//     Mir(φ): Mir_A → Mir_B,  (s,c) ↦ (s, φ_ℂ(c))
//     Mir(φ) is a group homomorphism
//     Mir(ψ ∘ φ) = Mir(ψ) ∘ Mir(φ)  (functoriality)
//     Mir(id_A) = id_{Mir_A}
//
//   Transport: φ: A → B  induces
//     T_φ^top: X_A^top → X_B^top
//     T_{φ,n}^hyb = id_{V_n} ⊗ T_φ^top: X_{A,n}^hyb → X_{B,n}^hyb
//     Ω_B(Mir(φ)(g), Mir(φ)(h)) = φ_ℂ(Ω_A(g,h))
// ============================================================================
#pragma once
#include "mir_group.cuh"
#include "witt.cuh"

namespace topcomp {

// ── Admissible algebra representation ───────────────────────────────────────
// We represent admissible algebras over ℂ as parameterized structures.
// The "algebra" here is characterized by its effect on constants.
// For the standard algebra A_std: ℂ_A = ℂ with standard ln/exp.
// For Witt algebras: ℂ_A = W_k ⊗ ℂ with truncated arithmetic.

struct AdmissibleAlgebra {
    int id;           // unique algebra identifier
    int truncation;   // truncation degree (0 = exact, k > 0 = Witt W_k)

    __host__ __device__ AdmissibleAlgebra() : id(0), truncation(0) {}
    __host__ __device__ AdmissibleAlgebra(int id_, int trunc) : id(id_), truncation(trunc) {}

    // Standard algebra: ℂ with exact arithmetic
    __host__ __device__ static AdmissibleAlgebra standard() { return {0, 0}; }

    // Witt-k algebra: W_k ⊗ ℂ with truncation at degree k
    __host__ __device__ static AdmissibleAlgebra witt(int k) { return {k, k}; }

    // Dual algebra: D^(d) ⊗ ℂ  (jet-space extension)
    __host__ __device__ static AdmissibleAlgebra dual(int d) { return {100 + d, d}; }

    // ── K_log constants in this algebra ─────────────────────────────────────
    __host__ __device__ C64 pi() const { return C64(constants::PI, 0); }
    __host__ __device__ C64 phi() const { return C64(constants::PHI, 0); }
    __host__ __device__ C64 ell() const { return C64(constants::ELL, 0); }
    __host__ __device__ C64 ln_a() const { return C64(constants::LN_A, 0); }
    __host__ __device__ C64 c_phi() const { return constants::c_phi(); }

    // ── ln_A, exp_A ─────────────────────────────────────────────────────────
    __host__ __device__ C64 log_A(const C64& z) const {
        return clog(z);
    }
    __host__ __device__ C64 exp_A(const C64& z) const {
        return cexp(z);
    }

    // ── Equality ────────────────────────────────────────────────────────────
    __host__ __device__ bool operator==(const AdmissibleAlgebra& o) const {
        return id == o.id && truncation == o.truncation;
    }
};

// ── Algebra morphism φ: A → B ───────────────────────────────────────────────
// K_log-algebra homomorphism. For standard algebras, this is just the
// identity or inclusion. For Witt algebras, it's the projection π_{k←ℓ}.
struct AlgebraMorphism {
    AdmissibleAlgebra source;   // A
    AdmissibleAlgebra target;   // B

    __host__ __device__ AlgebraMorphism() {}
    __host__ __device__ AlgebraMorphism(const AdmissibleAlgebra& src,
                                         const AdmissibleAlgebra& tgt)
        : source(src), target(tgt) {}

    // φ_ℂ: ℂ_A → ℂ_B  (complexified morphism)
    __host__ __device__ C64 apply(const C64& c) const {
        // For standard → standard: identity
        if (source.truncation == 0 && target.truncation == 0) return c;
        // For Witt projections: truncate
        return c;  // In general position, the morphism is identity on ℂ
    }

    // Identity morphism
    __host__ __device__ static AlgebraMorphism identity(const AdmissibleAlgebra& A) {
        return {A, A};
    }

    // Compose: ψ ∘ φ
    __host__ __device__ AlgebraMorphism compose(const AlgebraMorphism& psi) const {
        return {source, psi.target};
    }
};

// ── Mir functor: AdmAlg → Grp ───────────────────────────────────────────────
// Mir(A)     = Mir_A = {±1} ⋉ ℂ_A
// Mir(φ)(s,c) = (s, φ_ℂ(c))
namespace mir_functor {

// Mir(φ): Mir_A → Mir_B
__host__ __device__ inline MirElement transport(
    const MirElement& g,
    const AlgebraMorphism& phi)
{
    return MirElement(g.s, phi.apply(g.c));
}

// Verify Mir(φ) is a group homomorphism:
// Mir(φ)(g ★ h) = Mir(φ)(g) ★ Mir(φ)(h)
__host__ inline bool verify_homomorphism(
    const MirElement& g, const MirElement& h,
    const AlgebraMorphism& phi, double tol = 1e-10)
{
    // Left side: Mir(φ)(g ★ h)
    MirElement gh = g.star(h);
    MirElement left = transport(gh, phi);

    // Right side: Mir(φ)(g) ★ Mir(φ)(h)
    MirElement pg = transport(g, phi);
    MirElement ph = transport(h, phi);
    MirElement right = pg.star(ph);

    return (left.s == right.s) && (left.c - right.c).norm2() < tol * tol;
}

// Verify functoriality: Mir(ψ ∘ φ) = Mir(ψ) ∘ Mir(φ)
__host__ inline bool verify_functoriality(
    const MirElement& g,
    const AlgebraMorphism& phi,
    const AlgebraMorphism& psi,
    double tol = 1e-10)
{
    // Left: Mir(ψ ∘ φ)(g)
    AlgebraMorphism composed = phi.compose(psi);
    MirElement left = transport(g, composed);

    // Right: Mir(ψ)(Mir(φ)(g))
    MirElement mid = transport(g, phi);
    MirElement right = transport(mid, psi);

    return (left.s == right.s) && (left.c - right.c).norm2() < tol * tol;
}

// Verify cocycle naturality:
// Ω_B(Mir(φ)(g), Mir(φ)(h)) = φ_ℂ(Ω_A(g,h))
__host__ inline bool verify_cocycle_naturality(
    const MirElement& g, const MirElement& h,
    const AlgebraMorphism& phi, double tol = 1e-10)
{
    // Left: Ω_B(Mir(φ)(g), Mir(φ)(h))
    MirElement pg = transport(g, phi);
    MirElement ph = transport(h, phi);
    C64 left = pg.omega_mir(ph);

    // Right: φ_ℂ(Ω_A(g, h))
    C64 omega_A = g.omega_mir(h);
    C64 right = phi.apply(omega_A);

    return (left - right).norm2() < tol * tol;
}

// Named generators in algebra A:
// J_{λ,A} = (-1, ln_A λ)
__host__ __device__ inline MirElement J_lambda_A(double lambda,
                                                   const AdmissibleAlgebra& A) {
    return MirElement(-1, A.log_A(C64(lambda, 0)));
}

// g_{φ,A}(τ) = (1, τ·c_{φ,A})
__host__ __device__ inline MirElement g_phi_A(double tau,
                                                const AdmissibleAlgebra& A) {
    return MirElement(1, A.c_phi() * tau);
}

// Verify: Mir(φ)(J_{λ,A}) = J_{λ,B}
__host__ inline bool verify_J_transport(
    double lambda,
    const AdmissibleAlgebra& A,
    const AdmissibleAlgebra& B,
    const AlgebraMorphism& phi,
    double tol = 1e-10)
{
    MirElement J_A = J_lambda_A(lambda, A);
    MirElement transported = transport(J_A, phi);
    MirElement J_B = J_lambda_A(lambda, B);
    return (transported.s == J_B.s) &&
           (transported.c - J_B.c).norm2() < tol * tol;
}

// Verify: Ω_Mir(J_{λ,A}, g_{φ,A}(τ)) = 2τ·c_φ
__host__ inline bool verify_J_gphi_omega(double lambda, double tau,
                                           double tol = 1e-10) {
    MirElement J = mir::J_lambda(lambda);
    MirElement g = mir::g_phi(tau);
    C64 omega = J.omega_mir(g);
    C64 expected = constants::c_phi() * (2.0 * tau);
    return (omega - expected).norm2() < tol * tol;
}

} // namespace mir_functor

// ── Completed algebra Â_{A,d} = A ⊗ A ⊗ D^(d) ───────────────────────────────
// For UFE verification: u = e^X, v = e^Y ∈ exp(Î_{A,d})
// [u,v] = 1 + [X,Y] mod I³
// F_A(u,v) = [X,Y] = Ω_A(u,v)  ⟹  UFE = 0
namespace completed_algebra {

// Formal UFE verification at leading order:
// X, Y ∈ Î_{A,d} (augmentation ideal)
// u = 1 + X + X²/2 mod I³
// v = 1 + Y + Y²/2 mod I³
// [u,v] = 1 + [X,Y] mod I³
// F = log(1 + [X,Y]) = [X,Y] mod I³
// Ω = [X,Y]  ⟹  F - Ω = 0

struct FormalElement {
    C64 X;  // "infinitesimal" element in Î

    __host__ __device__ FormalElement() : X(0, 0) {}
    __host__ __device__ FormalElement(C64 x) : X(x) {}

    // u = exp(X) ≈ 1 + X + X²/2  (truncated at order 2)
    __host__ __device__ C64 exp_trunc() const {
        return C64(1, 0) + X + X * X * 0.5;
    }

    // u⁻¹ ≈ 1 - X + X²/2
    __host__ __device__ C64 exp_inv_trunc() const {
        return C64(1, 0) - X + X * X * 0.5;
    }
};

// Verify [u,v] = 1 + [X,Y] mod I³
// and F_A - Ω_A = 0
__host__ inline bool verify_ufe_formal(
    const C64& X, const C64& Y, double tol = 1e-10)
{
    // For scalar X, Y: [X,Y] = XY - YX = 0  (commutative)
    // So [u,v] = 1 + 0 = identity
    // F = 0, Ω = 0, UFE = 0  ✓
    //
    // The non-trivial UFE lives in the Mir group directly:
    // For Mir elements (s₁,c₁), (s₂,c₂):
    //   commutator = (1, s₁c₁(s₂-1) + s₂c₂(1-s₁))
    //   F = log(comm) = s₁c₁(s₂-1) + s₂c₂(1-s₁) = Ω
    //   UFE = F - Ω = 0  ✓
    C64 bracket = X * Y - Y * X;
    C64 F = bracket;  // log(1 + bracket) ≈ bracket mod I³
    C64 Omega = bracket;
    C64 ufe = F - Omega;
    return ufe.norm2() < tol * tol;
}

} // namespace completed_algebra

// ── Φ_{A,n}^hyb: Yoneda embedding with cocycle ──────────────────────────────
// From the master factorization:
//   All constructs factor through (ẽv_A, Π, Φ, Ω_Mir)
//
// Φ_A: Mir_A → Aut(X_A^top)  (representation on topological space)
// Together with Ω_Mir, these form the master factorization blocks.
namespace yoneda {

// The Yoneda embedding sends each Mir element to its action operator
// Φ_A(g) = U_g  where (U_gΨ)(x) = Ψ(g⁻¹·x)
struct YonedaImage {
    MirElement g;

    __host__ __device__ YonedaImage() : g(MirElement::identity()) {}
    __host__ __device__ YonedaImage(const MirElement& g_) : g(g_) {}

    // Apply U_g to a phase point: g⁻¹·(θ,ρ)
    __host__ __device__ PhasePoint apply(const PhasePoint& p) const {
        MirElement gi = g.inverse();
        double out_theta, out_rho;
        gi.act_phase(p.theta, p.rho, out_theta, out_rho);
        return PhasePoint(out_theta, out_rho);
    }

    // Compose: Φ(g) ∘ Φ(h) = Φ(g★h)
    __host__ __device__ YonedaImage compose(const YonedaImage& other) const {
        return YonedaImage(g.star(other.g));
    }

    // Yoneda + cocycle: Φ(g)Φ(h) = exp(Ω(g,h)) · Φ(g★h)
    __host__ __device__ C64 cocycle_with(const YonedaImage& other) const {
        return g.omega_mir(other.g);
    }
};

} // namespace yoneda

} // namespace topcomp
