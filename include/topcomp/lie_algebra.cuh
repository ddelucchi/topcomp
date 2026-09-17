// ============================================================================
// TopComp: Lie Algebra sl₂ — Differential Operators D, E, K
// ============================================================================
// Implements the sl₂(K) Lie algebra structure from the reference:
//
//   D = d/dt,    E = t·d/dt,    K = t²·d/dt
//
//   Commutation relations:
//     [D, E] = D      (i.e. [e_, h_] = 2e_)
//     [D, K] = 2E     (standard sl₂ bracket)
//     [E, K] = K
//
//   Standard sl₂ basis (Chevalley):
//     e = D,  f = -K,  h = -2E
//     [h, e] = 2e,  [h, f] = -2f,  [e, f] = h
//
//   Exponentiated generators:
//     e^{hD} f(t) = f(t + h)              (translation)
//     e^{(ln λ)E} f(t) = f(λt)            (scaling)
//     e^{-hK} f(t) = f(t/(1+ht))          (special conformal)
//     JDJ = -K  ⟹  J e^{hD} J = e^{-hK}  (inversion conjugation)
//
//   Ad representation on sl₂:
//     Ad_A(D) = a²D + 2ac·E + c²K
//     Ad_A(E) = abD + (ad+bc)E + cdK
//     Ad_A(K) = b²D + 2bd·E + d²K
//     where A = [[a,b],[c,d]] ∈ SL₂
//
//   Matrix representation:
//     D ↦ [[0,1],[0,0]],  E ↦ [[-1/2,0],[0,1/2]],  K ↦ [[0,0],[-1,0]]
// ============================================================================
#pragma once
#include "representation.cuh"
#include "witt.cuh"

namespace topcomp {

// ── sl₂ element: v·D + u·E + w·K ────────────────────────────────────────────
// Parameterized by (v, u, w) ∈ K³
struct SL2Element {
    C64 v;  // coefficient of D
    C64 u;  // coefficient of E  (h = -2E in Chevalley basis, so h_coeff = -2u)
    C64 w;  // coefficient of K

    __host__ __device__ SL2Element() : v(0,0), u(0,0), w(0,0) {}
    __host__ __device__ SL2Element(C64 v_, C64 u_, C64 w_) : v(v_), u(u_), w(w_) {}
    __host__ __device__ SL2Element(double v_, double u_, double w_)
        : v(v_,0), u(u_,0), w(w_,0) {}

    // Named generators
    __host__ __device__ static SL2Element D() { return {1, 0, 0}; }
    __host__ __device__ static SL2Element E() { return {0, 1, 0}; }
    __host__ __device__ static SL2Element K() { return {0, 0, 1}; }

    // Chevalley basis: e = D, f = -K, h = -2E
    __host__ __device__ static SL2Element e_chev() { return D(); }
    __host__ __device__ static SL2Element f_chev() { return {0, 0, -1}; }  // -K
    __host__ __device__ static SL2Element h_chev() { return {0, -2, 0}; }  // -2E

    // Addition
    __host__ __device__ SL2Element operator+(const SL2Element& o) const {
        return {v + o.v, u + o.u, w + o.w};
    }
    __host__ __device__ SL2Element operator-(const SL2Element& o) const {
        return {v - o.v, u - o.u, w - o.w};
    }
    __host__ __device__ SL2Element operator*(C64 s) const {
        return {v * s, u * s, w * s};
    }
    __host__ __device__ SL2Element operator*(double s) const {
        return {v * s, u * s, w * s};
    }

    // ── Lie bracket [X, Y] ──────────────────────────────────────────────────
    // [vD + uE + wK, v'D + u'E + w'K]
    // Using: [D,E]=D, [D,K]=2E, [E,K]=K
    __host__ __device__ SL2Element bracket(const SL2Element& Y) const {
        // [X,Y] = Σ_{i<j} (x_i y_j - x_j y_i) [e_i, e_j]
        // [D,E] = D:   coefficient of D from (v·u' - u·v')·1
        // [D,K] = 2E:  coefficient of E from (v·w' - w·v')·2
        //              coefficient of D from 0
        // [E,K] = K:   coefficient of K from (u·w' - w·u')·1

        C64 new_v = (v * Y.u - u * Y.v);                // from [D,E]=D
        C64 new_u = (v * Y.w - w * Y.v) * 2.0;          // from [D,K]=2E
        C64 new_w = (u * Y.w - w * Y.u);                // from [E,K]=K

        return {new_v, new_u, new_w};
    }

    // ── Matrix representation ───────────────────────────────────────────────
    // vD + uE + wK  ↦  [[-u/2, v], [-w, u/2]]
    __host__ __device__ Mat2 to_matrix() const {
        return {u * (-0.5), v, -w, u * 0.5};
    }

    // Norm squared
    __host__ __device__ double norm2() const {
        return v.norm2() + u.norm2() + w.norm2();
    }

    void print(const char* label = "") const {
        printf("%s(%.4f%+.4fi)D + (%.4f%+.4fi)E + (%.4f%+.4fi)K",
               label, v.re, v.im, u.re, u.im, w.re, w.im);
    }
};

// ── Lie bracket verification ────────────────────────────────────────────────
namespace sl2_brackets {

// Verify [D,E] = D
__host__ inline bool check_DE(double tol = 1e-12) {
    SL2Element comm = SL2Element::D().bracket(SL2Element::E());
    SL2Element expected = SL2Element::D();
    return (comm - expected).norm2() < tol;
}

// Verify [D,K] = 2E
__host__ inline bool check_DK(double tol = 1e-12) {
    SL2Element comm = SL2Element::D().bracket(SL2Element::K());
    SL2Element expected = SL2Element::E() * 2.0;
    return (comm - expected).norm2() < tol;
}

// Verify [E,K] = K
__host__ inline bool check_EK(double tol = 1e-12) {
    SL2Element comm = SL2Element::E().bracket(SL2Element::K());
    SL2Element expected = SL2Element::K();
    return (comm - expected).norm2() < tol;
}

// Verify Chevalley: [h,e] = 2e, [h,f] = -2f, [e,f] = h
__host__ inline bool check_chevalley(double tol = 1e-12) {
    SL2Element e = SL2Element::e_chev();
    SL2Element f = SL2Element::f_chev();
    SL2Element h = SL2Element::h_chev();

    SL2Element he = h.bracket(e);
    SL2Element hf = h.bracket(f);
    SL2Element ef = e.bracket(f);

    bool ok1 = (he - e * 2.0).norm2() < tol;       // [h,e] = 2e
    bool ok2 = (hf - f * (-2.0)).norm2() < tol;     // [h,f] = -2f
    bool ok3 = (ef - h).norm2() < tol;              // [e,f] = h

    return ok1 && ok2 && ok3;
}

// Jacobi identity: [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
__host__ inline bool check_jacobi(const SL2Element& X, const SL2Element& Y,
                                     const SL2Element& Z, double tol = 1e-10) {
    SL2Element t1 = X.bracket(Y.bracket(Z));
    SL2Element t2 = Y.bracket(Z.bracket(X));
    SL2Element t3 = Z.bracket(X.bracket(Y));
    SL2Element sum = t1 + t2 + t3;
    return sum.norm2() < tol;
}

} // namespace sl2_brackets

// ── Adjoint representation Ad: SL₂ → Aut(sl₂) ───────────────────────────────
// Ad_A(X) = AXA⁻¹  in matrix form
// Explicit formulas from the reference:
//   Ad_A(D) = a²D + 2ac·E + c²K
//   Ad_A(E) = abD + (ad+bc)E + cdK
//   Ad_A(K) = b²D + 2bd·E + d²K
namespace adjoint {

// Ad_A(D) = a²D + 2acE + c²K
__host__ __device__ inline SL2Element Ad_D(const Mat2& A) {
    return {A.a * A.a, A.a * A.c * 2.0, A.c * A.c};
}

// Ad_A(E) = abD + (ad+bc)E + cdK
__host__ __device__ inline SL2Element Ad_E(const Mat2& A) {
    return {A.a * A.b, A.a * A.d + A.b * A.c, A.c * A.d};
}

// Ad_A(K) = b²D + 2bdE + d²K
__host__ __device__ inline SL2Element Ad_K(const Mat2& A) {
    return {A.b * A.b, A.b * A.d * 2.0, A.d * A.d};
}

// General Ad_A(X) for X = vD + uE + wK
__host__ __device__ inline SL2Element Ad(const Mat2& A, const SL2Element& X) {
    SL2Element aD = Ad_D(A) * X.v;
    SL2Element aE = Ad_E(A) * X.u;
    SL2Element aK = Ad_K(A) * X.w;
    return aD + aE + aK;
}

// Verify Ad is a Lie algebra homomorphism:
// Ad([X,Y]) = [Ad(X), Ad(Y)]
__host__ inline bool verify_Ad_homomorphism(
    const Mat2& A, const SL2Element& X, const SL2Element& Y,
    double tol = 1e-8)
{
    SL2Element bracket_XY = X.bracket(Y);
    SL2Element left = Ad(A, bracket_XY);

    SL2Element Ad_X = Ad(A, X);
    SL2Element Ad_Y = Ad(A, Y);
    SL2Element right = Ad_X.bracket(Ad_Y);

    return (left - right).norm2() < tol;
}

} // namespace adjoint

// ── Exponentiated action on functions ───────────────────────────────────────
// These are the GROUP-level operations generated by the Lie algebra.
namespace exp_action {

// e^{hD}: f(t) → f(t+h)  — translation
__host__ __device__ inline double exp_D(double h, double t) {
    return t + h;
}

// e^{(ln λ)E}: f(t) → f(λt)  — scaling
__host__ __device__ inline double exp_E(double lambda, double t) {
    return lambda * t;
}

// e^{-hK}: f(t) → f(t/(1+ht)) = G_{-h}(t)  — special conformal
__host__ __device__ inline double exp_negK(double h, double t) {
    return t / (1.0 + h * t);
}

// Verify JDJ = -K  on a test function t^m:
// (JDJ)(t^m) should equal (-K)(t^m) = -m·t^{m+1}
// JDJ: t^m → J(D(t^{-m})) → J(-m·t^{-m-1}) → -m·t^{m+1}
// -K(t^m) = -(t²·d/dt)(t^m) = -m·t^{m+1}  ✓

// Verify J e^{hD} J = e^{-hK} = G_{-h} at a point
__host__ inline bool verify_J_expD_J(double h, double t, double tol = 1e-10) {
    // Left: J ∘ e^{hD} ∘ J (t) = 1/(1/t + h) = t/(1+ht)
    double j1 = 1.0 / t;       // J(t)
    double j2 = j1 + h;         // T_h(J(t))
    double left = 1.0 / j2;     // J(T_h(J(t)))

    // Right: e^{-hK}(t) = t/(1+ht)
    double right = exp_negK(h, t);

    return fabs(left - right) < tol;
}

} // namespace exp_action

// ── Ergodic projector Π = lim Π_N ───────────────────────────────────────────
// Π_N = (1/(2N+1)) Σ_{n=-N}^{N} U^n
// From the groupoid formalism:
//   Γ_ℝ = ⟨ev(σ) : σ ∈ Σ_arr⟩ ⊂ Mir_ℝ
//   M = M ⋊ Γ_ℝ  (action groupoid)
//   (U_f Ψ)(x) = Ψ(f·x)
//   U_f ∘ U_g = U_{g★f}  (anti-homomorphism of Mir action)
//
// Properties of Π:
//   Π² = Π,  Π* = Π  (self-adjoint projector)
//   U·Π = Π = Π·U    (U-invariant)
//   →←  = Π,  ←→ = Π  (arrow-word meaning)
//   →← ≠ 1_C = →∘←    (NOT the identity!)
namespace ergodic {

// Π_N applied to a value: average over N forward and backward steps
__host__ inline double projector_N(double (*U)(double, double),
                                      double x, double param, int N) {
    double sum = 0;
    for (int n = -N; n <= N; ++n) {
        sum += U(x, static_cast<double>(n) * param);
    }
    return sum / (2 * N + 1);
}

// Verify Π_N² ≈ Π_N  (idempotent check for finite N)
// ||Π_N² - Π_N|| ≤ error bound that → 0 as N → ∞

// ── Cesàro mean convergence rate ────────────────────────────────────────────
// |U·Π_N - Π_N| = |U^{N+1} - U^{-N}| / (2N+1)
__host__ inline double convergence_bound(int N) {
    return 2.0 / (2.0 * N + 1.0);
}

} // namespace ergodic

// ── 1-form structure: du, dω̃ ───────────────────────────────────────────────
// From section 7 of the reference:
//   u = θ + ς(ρ),  ω̃ = θ - ς(ρ)  where ς = (π/2ℓ)ρ
//   du = dθ + (π/2ℓ)dρ
//   dω̃ = dθ - (π/2ℓ)dρ
//   du(X_φ) = 0     (u is constant along golden flow)
//   dω̃(X_φ) = -π    (ω̃ decreases by π per unit step)
//
// Where X_φ = ℓ·∂_ρ - (π/2)·∂_θ  is the golden flow vector field.
namespace forms {

// X_φ components: (dθ component, dρ component) = (-π/2, ℓ)
__host__ __device__ inline void X_phi(double& dtheta, double& drho) {
    dtheta = -constants::HALF_PI;
    drho = constants::ELL;
}

// du(X_φ) = d(θ + ςρ)(X_φ) = -π/2 + (π/2ℓ)·ℓ = -π/2 + π/2 = 0
__host__ __device__ inline double du_on_Xphi() {
    return -constants::HALF_PI + (constants::HALF_PI / constants::ELL) * constants::ELL;
    // = 0
}

// dω̃(X_φ) = d(θ - ςρ)(X_φ) = -π/2 - (π/2ℓ)·ℓ = -π/2 - π/2 = -π
__host__ __device__ inline double domega_on_Xphi() {
    return -constants::HALF_PI - (constants::HALF_PI / constants::ELL) * constants::ELL;
    // = -π
}

// Coordinate change verification:
// dθ = ½(du + dū),  dς = ½(du - dū)
// dθ = dς  ⟺  dū = 0  (ū constant)
// dθ = -dς  ⟺  du = 0  (u constant)

} // namespace forms

} // namespace topcomp
