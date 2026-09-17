// ============================================================================
// TopComp: Representation Theory — PGL₂, SL₂, Möbius, ℙ¹
// ============================================================================
// Four representations from the reference document:
//
//   Π_{M,A}:   Mir_A → Aff(M)     — affine action on phase space
//   Π_{ℂ×,A}:  Mir_A → Aut(ℂ×)    — multiplicative action on ℂ×
//   Π_{PGL,A}: Mir_A → PGL₂(ℂ_A)  — projective representation
//   Π_{End,k,A}: Mir_A → End_K(R_k) — endomorphism on truncated polys
//
// Projective line ℙ¹(K) = K ∪ {∞}  with Möbius transformations:
//   T_h(t) = t + h           [[1,h],[0,1]]     (translation)
//   S_λ(t) = λt              [[λ,0],[0,1]]     (scaling)
//   J(t) = 1/t               [[0,1],[1,0]]     (inversion)
//   G_κ(t) = t/(1-κt)        [[1,0],[-κ,1]]    (special conformal)
//
// Relations:
//   J ∘ T_h ∘ J = G_{-h}
//   S_λ T_h S_λ⁻¹ = T_{λh}
//   PGL₂ ⊃ ⟨T_h, J, S_λ⟩
//
// Arrow representation ρ_{ℙ¹}:
//   ρ(→) = T_h,   ρ(↔) = J,   ρ(←) = J T_h J = e^{-hK}
//   ρ(⟸) = S_σ,   ρ(⟹) = S_{σ⁻¹}
// ============================================================================
#pragma once
#include "phase_space.cuh"
#include "witt.cuh"

namespace topcomp {

// ── SL₂(K) matrix ───────────────────────────────────────────────────────────
// 2×2 matrix with determinant 1 (or more generally, elements of GL₂)
struct Mat2 {
    C64 a, b, c, d;  // [[a,b],[c,d]]

    __host__ __device__ Mat2() : a(1,0), b(0,0), c(0,0), d(1,0) {}
    __host__ __device__ Mat2(C64 a_, C64 b_, C64 c_, C64 d_)
        : a(a_), b(b_), c(c_), d(d_) {}

    // Matrix multiply
    __host__ __device__ Mat2 operator*(const Mat2& o) const {
        return {a * o.a + b * o.c, a * o.b + b * o.d,
                c * o.a + d * o.c, c * o.b + d * o.d};
    }

    // Determinant
    __host__ __device__ C64 det() const { return a * d - b * c; }

    // Inverse (for det ≠ 0): [[d,-b],[-c,a]] / det
    __host__ __device__ Mat2 inv() const {
        C64 di = C64(1, 0) / det();
        return {d * di, -b * di, -c * di, a * di};
    }

    // Transpose
    __host__ __device__ Mat2 transpose() const {
        return {a, c, b, d};
    }

    // Adjoint action: Ad_A(X) = AXA⁻¹
    __host__ __device__ Mat2 Ad(const Mat2& X) const {
        return *this * X * inv();
    }

    // Trace
    __host__ __device__ C64 trace() const { return a + d; }

    // Möbius action on ℙ¹: t ↦ (at + b)/(ct + d)
    __host__ __device__ C64 mobius(const C64& t) const {
        return (a * t + b) / (c * t + d);
    }

    // Identity
    __host__ __device__ static Mat2 identity() {
        return {C64(1,0), C64(0,0), C64(0,0), C64(1,0)};
    }

    void print(const char* label = "") const {
        printf("%s[[%.4f%+.4fi, %.4f%+.4fi], [%.4f%+.4fi, %.4f%+.4fi]]",
               label, a.re, a.im, b.re, b.im, c.re, c.im, d.re, d.im);
    }
};

// ── Möbius generators on ℙ¹(K) ──────────────────────────────────────────────
namespace mobius {

// T_h: translation  t ↦ t + h,  [[1,h],[0,1]]
__host__ __device__ inline Mat2 T(const C64& h) {
    return {C64(1,0), h, C64(0,0), C64(1,0)};
}
__host__ __device__ inline Mat2 T(double h) {
    return {C64(1,0), C64(h,0), C64(0,0), C64(1,0)};
}

// S_λ: scaling  t ↦ λt,  [[λ,0],[0,1]]
__host__ __device__ inline Mat2 S(const C64& lambda) {
    return {lambda, C64(0,0), C64(0,0), C64(1,0)};
}
__host__ __device__ inline Mat2 S(double lambda) {
    return {C64(lambda,0), C64(0,0), C64(0,0), C64(1,0)};
}

// J: inversion  t ↦ 1/t,  [[0,1],[1,0]]
__host__ __device__ inline Mat2 J() {
    return {C64(0,0), C64(1,0), C64(1,0), C64(0,0)};
}

// G_κ: special conformal  t ↦ t/(1-κt),  [[1,0],[-κ,1]]
__host__ __device__ inline Mat2 G(const C64& kappa) {
    return {C64(1,0), C64(0,0), -kappa, C64(1,0)};
}
__host__ __device__ inline Mat2 G(double kappa) {
    return {C64(1,0), C64(0,0), C64(-kappa,0), C64(1,0)};
}

// Rotation matrix R(α)
__host__ __device__ inline Mat2 R(double alpha) {
    return {C64(cos(alpha),0), C64(-sin(alpha),0),
            C64(sin(alpha),0), C64(cos(alpha),0)};
}

// ── Verify J T_h J = G_{-h} ─────────────────────────────────────────────────
__host__ inline bool verify_JTJ(double h, double tol = 1e-10) {
    Mat2 result = J() * T(h) * J();
    Mat2 expected = G(-h);
    double err = (result.a - expected.a).norm2() + (result.b - expected.b).norm2() +
                 (result.c - expected.c).norm2() + (result.d - expected.d).norm2();
    return err < tol;
}

// ── Verify S_λ T_h S_λ⁻¹ = T_{λh} ───────────────────────────────────────────
__host__ inline bool verify_STS(double lambda, double h, double tol = 1e-10) {
    Mat2 Sl = S(lambda);
    Mat2 result = Sl * T(h) * Sl.inv();
    Mat2 expected = T(lambda * h);
    double err = (result.a - expected.a).norm2() + (result.b - expected.b).norm2() +
                 (result.c - expected.c).norm2() + (result.d - expected.d).norm2();
    return err < tol;
}

} // namespace mobius

// ── Arrow representation ρ_{ℙ¹} ─────────────────────────────────────────────
// Maps arrow alphabet to PGL₂ matrices
namespace arrow_rep {

// ρ(→) = T_{c_φ}  (translation by c_φ)
__host__ __device__ inline Mat2 rho_right() {
    return mobius::T(constants::c_phi());
}

// ρ(←) = J T_{c_φ} J = G_{-c_φ}   (≡ e^{-c_φ K})
__host__ __device__ inline Mat2 rho_left() {
    return mobius::G(C64(-constants::c_phi().re, -constants::c_phi().im));
}

// ρ(↔) = J  (Möbius inversion)
__host__ __device__ inline Mat2 rho_mirror() {
    return mobius::J();
}

// ρ(⟸) = S_{ΔΣ}  (scale by ΔΣ)
__host__ __device__ inline Mat2 rho_lldir() {
    return mobius::S(constants::DELTA_SIGMA);
}

// ρ(⟹) = S_{ΔΣ⁻¹}  (scale by 1/ΔΣ)
__host__ __device__ inline Mat2 rho_rrdir() {
    return mobius::S(1.0 / constants::DELTA_SIGMA);
}

// Verify: ↔ ∘ → ∘ ↔ = ← (conjugation identity)
__host__ inline bool verify_mirror_conjugation(double tol = 1e-10) {
    Mat2 result = rho_mirror() * rho_right() * rho_mirror();
    Mat2 expected = rho_left();
    double err = (result.a - expected.a).norm2() + (result.b - expected.b).norm2() +
                 (result.c - expected.c).norm2() + (result.d - expected.d).norm2();
    return err < tol;
}

// Map any arrow to its PGL₂ matrix
__host__ __device__ inline Mat2 rho(Arrow arrow) {
    switch (arrow) {
        case Arrow::RIGHT:  return rho_right();
        case Arrow::LEFT:   return rho_left();
        case Arrow::LLDIR:  return rho_lldir();
        case Arrow::RRDIR:  return rho_rrdir();
        case Arrow::MIRROR: return rho_mirror();
        default:            return Mat2::identity();
    }
}

// Evaluate arrow word in PGL₂
__host__ __device__ inline Mat2 rho_word(const Arrow* word, int N) {
    Mat2 result = Mat2::identity();
    for (int i = 0; i < N; ++i) {
        result = result * rho(word[i]);
    }
    return result;
}

// Commutator in PGL₂: [S_λ, T_h] = T_{(1-λ⁻¹)h}
__host__ inline bool verify_scale_translate_comm(double lambda, double h,
                                                    double tol = 1e-10) {
    Mat2 Sl = mobius::S(lambda);
    Mat2 Th = mobius::T(h);
    Mat2 comm = Sl.inv() * Th.inv() * Sl * Th;
    Mat2 expected = mobius::T((1.0 - 1.0/lambda) * h);
    double err = (comm.a - expected.a).norm2() + (comm.b - expected.b).norm2() +
                 (comm.c - expected.c).norm2() + (comm.d - expected.d).norm2();
    return err < tol;
}

} // namespace arrow_rep

// ── Π_{PGL,A}: Mir_A → PGL₂(ℂ_A) ────────────────────────────────────────────
// For s = +1: Π((+1,c)) = [[e^{c/2}, 0], [0, e^{-c/2}]]   (diagonal)
// For s = -1: Π((-1,c)) = [[0, e^{c/2}], [e^{-c/2}, 0]]    (anti-diagonal)
namespace pgl_rep {

__host__ __device__ inline Mat2 Pi(const MirElement& g) {
    C64 half_c = g.c * 0.5;
    C64 e_pos = cexp(half_c);
    C64 e_neg = cexp(-half_c);
    if (g.s == +1) {
        return {e_pos, C64(0,0), C64(0,0), e_neg};
    } else {
        return {C64(0,0), e_pos, e_neg, C64(0,0)};
    }
}

// Verify Π is a homomorphism: Π(g★h) = Π(g)·Π(h)
__host__ inline bool verify_homomorphism(const MirElement& g, const MirElement& h,
                                            double tol = 1e-8) {
    Mat2 left = Pi(g.star(h));
    Mat2 right = Pi(g) * Pi(h);
    double err = (left.a - right.a).norm2() + (left.b - right.b).norm2() +
                 (left.c - right.c).norm2() + (left.d - right.d).norm2();
    return err < tol;
}

} // namespace pgl_rep

// ── Area contraction: T = J/φ ───────────────────────────────────────────────
// T = (1/φ)·[[0,-1],[1,0]]
// det(T) = 1/φ²
// Area(T(A)) = |det(T)|·Area(A) = (1/φ²)·Area(A)
// T_N = T ⊕ ··· ⊕ T (N copies):
//   det(T_N) = (1/φ²)^N = φ^{-2N}
//   Vol_{2N}(T_N(A)) = φ^{-2N}·Vol_{2N}(A)
namespace contraction {

__host__ __device__ inline Mat2 T_matrix() {
    double inv_phi = constants::PHI_INV;
    return {C64(0, 0), C64(-inv_phi, 0),
            C64(inv_phi, 0), C64(0, 0)};
}

__host__ __device__ inline double area_ratio() {
    return constants::PHI_INV * constants::PHI_INV;  // 1/φ²
}

__host__ __device__ inline double volume_ratio(int N) {
    double r = 1.0;
    double ar = area_ratio();
    for (int i = 0; i < N; ++i) r *= ar;
    return r;  // φ^{-2N}
}

// S_d = φ·Q_{π/2}:  det(S_d) = φ^d
__host__ __device__ inline double expansion_ratio(int d) {
    double r = 1.0;
    for (int i = 0; i < d; ++i) r *= constants::PHI;
    return r;  // φ^d
}

} // namespace contraction

// ── Π_{End,k,A}: action on R_k (truncated polynomial ring) ──────────────────
// Arrow-induced endomorphisms on R_k = K_log[t]/(t^k)
// Uses the sl₂ generators e, f, h:
//   T_h → exp(h·e)   on R_k
//   J → reversal     on R_k
//   S_λ → λ-scaling  on R_k
namespace end_rep {

// exp(h·D)(p)(t) = p(t+h)  →  translation on R_k
__host__ inline TruncPoly T_action(const TruncPoly& p, double h) {
    return p.translate(h);
}

// S_λ(p)(t) = p(λt)  →  scaling on R_k
__host__ inline TruncPoly S_action(const TruncPoly& p, double lambda) {
    return p.scale(lambda);
}

// J(p)(t) = p(1/t)  →  only well-defined as formal operation
// For polynomial p, this reverses the coefficients
__host__ inline TruncPoly J_action(const TruncPoly& p) {
    return p.reverse();
}

// G_{-h} = J ∘ T_h ∘ J  →  e^{-hK} on R_k
// exp(-hK)(t) = t/(1+ht) as formal power series truncated at degree k
__host__ inline TruncPoly G_action(const TruncPoly& p, double h) {
    // Apply the composition: first J, then T_h, then J
    TruncPoly step1 = J_action(p);
    TruncPoly step2 = T_action(step1, h);
    return J_action(step2);
}

// Full arrow word evaluation on R_k
__host__ inline TruncPoly eval_word(const Arrow* word, int N,
                                       const TruncPoly& p) {
    TruncPoly result = p;
    for (int i = 0; i < N; ++i) {
        switch (word[i]) {
            case Arrow::RIGHT:
                result = T_action(result, constants::ELL);
                break;
            case Arrow::LEFT:
                result = G_action(result, -constants::ELL);
                break;
            case Arrow::LLDIR:
                result = S_action(result, constants::DELTA_SIGMA);
                break;
            case Arrow::RRDIR:
                result = S_action(result, 1.0 / constants::DELTA_SIGMA);
                break;
            case Arrow::MIRROR:
                result = J_action(result);
                break;
        }
    }
    return result;
}

} // namespace end_rep

// ── Π_{M,A}: Mir_ℝ → GL(3,ℝ) — 3×3 affine matrix representation ─────────────
// Reference Section 10:
//   M(s, β-iα) := [[s, 0, -α], [0, s, β], [0, 0, 1]]
//
// This is a faithful representation: M(g₁★g₂) = M(g₁)·M(g₂)
//
// Arrow matrices:
//   M(→) = M(+1, ln φ - iπ/2) = [[1,0,-π/2],[0,1,lnφ],[0,0,1]]
//   M(←) = M(+1, -lnφ + iπ/2) = [[1,0,π/2],[0,1,-lnφ],[0,0,1]]
//   M(⟸) = M(+1, lnσ)   = [[1,0,0],[0,1,lnσ],[0,0,1]]
//   M(⟹) = M(+1, -lnσ)  = [[1,0,0],[0,1,-lnσ],[0,0,1]]
//   M(↔) = M(-1, lnλ)    = [[-1,0,0],[0,-1,lnλ],[0,0,1]]
//
// Ξ_M vector: (θ, ρ, 1)^T,  Ξ'_M = M(g)·Ξ_M = (sθ-α, sρ+β, 1)^T
//
// Dual-number linearization:
//   M(g_φ(ε)) = I₃ + ε·N_φ,   N_φ = [[0,0,-π/2],[0,0,ℓ],[0,0,0]]
//   N_φ² = 0,  exp(ε·N_φ) = I₃ + ε·N_φ,  log(I₃ + ε·N_φ) = ε·N_φ
namespace affine_rep {

// 3×3 real matrix (stored as doubles)
struct Mat3 {
    double m[3][3];

    __host__ __device__ Mat3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                m[i][j] = (i == j) ? 1.0 : 0.0;
    }

    __host__ __device__ Mat3(double m00, double m01, double m02,
                             double m10, double m11, double m12,
                             double m20, double m21, double m22) {
        m[0][0]=m00; m[0][1]=m01; m[0][2]=m02;
        m[1][0]=m10; m[1][1]=m11; m[1][2]=m12;
        m[2][0]=m20; m[2][1]=m21; m[2][2]=m22;
    }

    __host__ __device__ Mat3 operator*(const Mat3& B) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                R.m[i][j] = 0;
                for (int k = 0; k < 3; ++k)
                    R.m[i][j] += m[i][k] * B.m[k][j];
            }
        return R;
    }

    __host__ __device__ Mat3 operator+(const Mat3& B) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] + B.m[i][j];
        return R;
    }

    __host__ __device__ Mat3 operator-(const Mat3& B) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] - B.m[i][j];
        return R;
    }

    __host__ __device__ Mat3 operator*(double s) const {
        Mat3 R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R.m[i][j] = m[i][j] * s;
        return R;
    }

    __host__ __device__ double frobenius_norm2() const {
        double s = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                s += m[i][j] * m[i][j];
        return s;
    }

    __host__ __device__ double frobenius_norm() const { return sqrt(frobenius_norm2()); }

    __host__ __device__ static Mat3 identity() { return Mat3(); }

    __host__ __device__ static Mat3 zero() {
        return Mat3(0,0,0, 0,0,0, 0,0,0);
    }

    void print(const char* label = "") const {
        printf("%s[[%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f], [%.6f, %.6f, %.6f]]",
               label, m[0][0],m[0][1],m[0][2],
               m[1][0],m[1][1],m[1][2],
               m[2][0],m[2][1],m[2][2]);
    }
};

// M(s, β-iα) := [[s, 0, -α], [0, s, β], [0, 0, 1]]
// For MirElement (s, c) where c = β - iα, i.e. Re(c)=β, Im(c)=-α → α=-Im(c)
__host__ __device__ inline Mat3 M_rep(const MirElement& g) {
    double s = (double)g.s;
    double alpha = -g.c.im;  // c = β - iα, so Im(c) = -α
    double beta  =  g.c.re;  // Re(c) = β
    return Mat3(s, 0, -alpha,
                0, s, beta,
                0, 0, 1);
}

// Apply M to phase-space vector Ξ = (θ, ρ, 1)
__host__ __device__ inline PhasePoint M_apply(const Mat3& M, const PhasePoint& xi) {
    double theta_new = M.m[0][0] * xi.theta + M.m[0][2];  // sθ - α
    double rho_new   = M.m[1][1] * xi.rho   + M.m[1][2];  // sρ + β
    return PhasePoint(theta_new, rho_new);
}

// Verify M is a homomorphism: M(g★h) = M(g)·M(h)
__host__ inline bool verify_M_homomorphism(const MirElement& g, const MirElement& h,
                                           double tol = 1e-10) {
    Mat3 M_gh = M_rep(g.star(h));
    Mat3 M_g_M_h = M_rep(g) * M_rep(h);
    return (M_gh - M_g_M_h).frobenius_norm() < tol;
}

// Arrow matrices M^(M)
__host__ __device__ inline Mat3 M_right() {
    return Mat3(1, 0, -constants::HALF_PI,
                0, 1, constants::ELL,
                0, 0, 1);
}

__host__ __device__ inline Mat3 M_left() {
    return Mat3(1, 0, constants::HALF_PI,
                0, 1, -constants::ELL,
                0, 0, 1);
}

__host__ __device__ inline Mat3 M_lldir() {
    return Mat3(1, 0, 0,
                0, 1, log(constants::DELTA_SIGMA),
                0, 0, 1);
}

__host__ __device__ inline Mat3 M_rrdir() {
    return Mat3(1, 0, 0,
                0, 1, -log(constants::DELTA_SIGMA),
                0, 0, 1);
}

__host__ __device__ inline Mat3 M_mirror(double lambda) {
    return Mat3(-1, 0, 0,
                0, -1, log(lambda),
                0, 0, 1);
}

// Verify M(↔)² = I₃
__host__ inline bool verify_mirror_squared(double lambda, double tol = 1e-10) {
    Mat3 Jm = M_mirror(lambda);
    Mat3 J2 = Jm * Jm;
    return (J2 - Mat3::identity()).frobenius_norm() < tol;
}

// Verify conjugation: M(↔)·M(→)·M(↔) = M(←)
__host__ inline bool verify_mirror_conjugation_M(double lambda, double tol = 1e-10) {
    Mat3 Jm = M_mirror(lambda);
    Mat3 result = Jm * M_right() * Jm;
    return (result - M_left()).frobenius_norm() < tol;
}

// Nilpotent generator N_φ = [[0,0,-π/2],[0,0,ℓ],[0,0,0]]
__host__ __device__ inline Mat3 N_phi() {
    return Mat3(0, 0, -constants::HALF_PI,
                0, 0, constants::ELL,
                0, 0, 0);
}

// Verify N_φ² = 0
__host__ inline bool verify_N_nilpotent(double tol = 1e-15) {
    Mat3 N = N_phi();
    Mat3 N2 = N * N;
    return N2.frobenius_norm() < tol;
}

// Verify exp(ε·N_φ) = I + ε·N_φ for truncated dual numbers
__host__ inline bool verify_dual_exp(double tol = 1e-15) {
    Mat3 N = N_phi();
    // Since N²=0, exp(ε·N) = I + ε·N (exact for any ε when ε²=0)
    // For numerical check, use small ε
    double eps = 0.01;
    Mat3 exact = Mat3::identity() + N * eps;
    // exp(ε·N) = I + ε·N + ε²·N²/2 + ... = I + ε·N (since N²=0)
    Mat3 N2 = N * N;
    Mat3 series = Mat3::identity() + N * eps + N2 * (eps * eps / 2.0);
    return (exact - series).frobenius_norm() < tol;
}

// Map arrow word to 3×3 matrix product: M_M(σ) = M(ev(σ))
__host__ inline Mat3 M_word(const Arrow* word, int N) {
    Mat3 result = Mat3::identity();
    for (int i = 0; i < N; ++i) {
        switch (word[i]) {
            case Arrow::RIGHT:  result = result * M_right(); break;
            case Arrow::LEFT:   result = result * M_left(); break;
            case Arrow::LLDIR:  result = result * M_lldir(); break;
            case Arrow::RRDIR:  result = result * M_rrdir(); break;
            case Arrow::MIRROR: result = result * M_mirror(constants::PHI); break;
        }
    }
    return result;
}

// Verify M_M(σ) = M(ev(σ)) — word evaluation consistency
__host__ inline bool verify_word_consistency(const Arrow* word, int N,
                                             double tol = 1e-10) {
    // Evaluate word via arrow_rep's evaluate mechanism
    ArrowWord aw;
    aw.length = N;
    for (int i = 0; i < N; ++i) aw.letters[i] = word[i];
    MirElement ev = aw.evaluate();

    Mat3 from_mir = M_rep(ev);
    Mat3 from_word = M_word(word, N);
    return (from_mir - from_word).frobenius_norm() < tol;
}

} // namespace affine_rep

} // namespace topcomp
