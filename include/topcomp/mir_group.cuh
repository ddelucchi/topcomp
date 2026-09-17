// ============================================================================
// TopComp: Mirror Group Algebra — Mir_A = {±1} ⋉ ℂ_A
// ============================================================================
// The fundamental algebraic structure of the framework:
//   Elements: (s, c) where s ∈ {±1}, c ∈ ℂ
//   Product:  (s₁,c₁) ★ (s₂,c₂) := (s₁s₂, c₁ + s₁c₂)
//   Identity: e = (1, 0)
//   Inverse:  (s,c)⁻¹ = (s, -sc)
//   Action:   (s,c) · w := sw + c         (on ℂ)
//             (s,c) · z := e^c · z^s      (on ℂ×)
//             (s,β-iα) · (θ,ρ) := (sθ-α, sρ+β)  (on phase space M)
//
// Mir(φ): functorial transport  Mir_A → Mir_B  preserving group structure.
// ============================================================================
#pragma once
#include "complex.cuh"
#include "constants.cuh"

namespace topcomp {

// ── MirElement: (s, c) ∈ Mir_A ──────────────────────────────────────────────
struct MirElement {
    int    s;   // sign ∈ {+1, -1}
    C64    c;   // complex displacement

    __host__ __device__ constexpr MirElement() : s(1), c(0.0, 0.0) {}
    __host__ __device__ constexpr MirElement(int sign, C64 displacement)
        : s(sign), c(displacement) {}
    __host__ __device__ constexpr MirElement(int sign, double beta, double alpha)
        : s(sign), c(beta, -alpha) {}  // c = β - iα

    // ── Group product: (s₁,c₁) ★ (s₂,c₂) = (s₁s₂, c₁ + s₁c₂) ────────────────
    __host__ __device__ MirElement star(const MirElement& other) const {
        return MirElement(s * other.s, c + C64(s, 0.0) * other.c);
    }

    // ── Identity: e = (1, 0) ────────────────────────────────────────────────
    __host__ __device__ static constexpr MirElement identity() {
        return MirElement(1, C64(0.0, 0.0));
    }

    // ── Inverse: (s,c)⁻¹ = (s, -sc) ─────────────────────────────────────────
    // Since s² = 1, s⁻¹ = s
    __host__ __device__ MirElement inverse() const {
        return MirElement(s, C64(-s, 0.0) * c);
    }

    // ── Action on w ∈ ℂ: (s,c)·w = sw + c ───────────────────────────────────
    __host__ __device__ C64 act_w(const C64& w) const {
        return C64(s, 0.0) * w + c;
    }

    // ── Action on z ∈ ℂ×: (s,c)·z = e^c · z^s ───────────────────────────────
    __host__ __device__ C64 act_z(const C64& z) const {
        C64 zs = (s == 1) ? z : C64(1.0, 0.0) / z;
        return cexp(c) * zs;
    }

    // ── Action on phase space M: (s,β-iα)·(θ,ρ) = (sθ-α, sρ+β) ──────────────
    __host__ __device__ void act_phase(double theta, double rho,
                                       double& out_theta, double& out_rho) const {
        double alpha = -c.im;  // c = β - iα  →  α = -Im(c)
        double beta  =  c.re;  // β = Re(c)
        out_theta = s * theta - alpha;
        out_rho   = s * rho + beta;
    }

    // ── Commutator: [(s,c),(t,d)] := (s,c)⁻¹ ★ (t,d)⁻¹ ★ (s,c) ★ (t,d) ──────
    __host__ __device__ MirElement commutator(const MirElement& other) const {
        return inverse().star(other.inverse()).star(*this).star(other);
    }

    // ── Omega_Mir curvature: Ω_Mir((s,c),(t,d)) = sc(t-1) + td(1-s) ─────────
    // This is Log_Mir of the commutator
    __host__ __device__ C64 omega_mir(const MirElement& other) const {
        C64 sc_part = C64(s, 0.0) * c * C64(other.s - 1, 0.0);
        C64 td_part = C64(other.s, 0.0) * other.c * C64(1 - s, 0.0);
        return sc_part + td_part;
    }

    // ── Exp/Log for positive sub-semigroup Mir⁺ = {(1,c)} ───────────────────
    __host__ __device__ static MirElement exp_mir(const C64& c) {
        return MirElement(1, c);
    }
    __host__ __device__ C64 log_mir() const {
        return c;  // Only valid for s = +1
    }

    // ── Comparison ──────────────────────────────────────────────────────────
    __host__ __device__ bool operator==(const MirElement& o) const {
        return s == o.s && c == o.c;
    }

    void print(const char* label = "") const {
        printf("%s(%+d, %.6f %+.6fi)", label, s, c.re, c.im);
    }
};

// ── Operator ★ overload ─────────────────────────────────────────────────────
__host__ __device__ inline MirElement operator*(const MirElement& a, const MirElement& b) {
    return a.star(b);
}

// ── Named constructors for the arrow generators ─────────────────────────────
namespace mir {

// B_{s;α,β} = (s, β - iα)
__host__ __device__ inline MirElement B(int s, double alpha, double beta) {
    return MirElement(s, C64(beta, -alpha));
}

// A_{α,λ} = B_{+1; α, ln λ}  (pure orientation-preserving)
__host__ __device__ inline MirElement A(double alpha, double lambda) {
    return B(+1, alpha, log(lambda));
}

// inv_λ = B_{-1; 0, ln λ}  (mirror inversion)
__host__ __device__ inline MirElement inv(double lambda) {
    return B(-1, 0.0, log(lambda));
}

// g_φ(t) = (1, t·c_φ)  (one-parameter golden-ratio flow)
__host__ __device__ inline MirElement g_phi(double t) {
    return MirElement(1, constants::c_phi() * t);
}

// J_{λ,A} = (-1, ln λ)  (mirror involution)
__host__ __device__ inline MirElement J_lambda(double lambda) {
    return MirElement(-1, C64(log(lambda), 0.0));
}

// U_π = A_{2π, 1}  (full angular rotation)
__host__ __device__ inline MirElement U_pi() {
    return A(constants::TWO_PI, 1.0);
}

// U_φ = A_{π/2, φ}  (golden spiral step)
__host__ __device__ inline MirElement U_phi() {
    return A(constants::HALF_PI, constants::PHI);
}

// U_φ^{-1}
__host__ __device__ inline MirElement U_phi_inv() {
    return A(-constants::HALF_PI, constants::PHI_INV);
}

// θ_{2π}: radial scale by a = 2π
__host__ __device__ inline MirElement theta_2pi() {
    return MirElement(1, C64(constants::LN_A, 0.0));
}

// θ_φ: radial scale by b = φ
__host__ __device__ inline MirElement theta_phi() {
    return MirElement(1, C64(constants::ELL, 0.0));
}

} // namespace mir

// ── Chain evaluation: ev(σ₁···σ_N) = G_N ────────────────────────────────────
// Given an array of MirElements (generators), compute the iterated product
// G_N = (∏ sⱼ, ∑ (∏_{p<j} sₚ) cⱼ)
struct MirChain {
    // Evaluate a sequence of generators
    __host__ __device__ static MirElement evaluate(const MirElement* generators, int N) {
        MirElement G = MirElement::identity();
        for (int j = 0; j < N; ++j) {
            G = G.star(generators[j]);
        }
        return G;
    }

    // Explicit formula: S_N = ∏ sⱼ, C_N = ∑ (∏_{p<j} sₚ) cⱼ
    __host__ __device__ static MirElement evaluate_explicit(
        const int* signs, const C64* displacements, int N)
    {
        int S = 1;
        C64 C(0.0, 0.0);
        int prefix_sign = 1;
        for (int j = 0; j < N; ++j) {
            C = C + C64(prefix_sign, 0.0) * displacements[j];
            prefix_sign *= signs[j];
        }
        // S_N = product of all signs
        S = 1;
        for (int j = 0; j < N; ++j) S *= signs[j];
        return MirElement(S, C);
    }
};

// ── Mir category: C_Mir with End(•) = Σ_arr*/~ ──────────────────────────────
// The category has one object • and morphisms are equivalence classes
// of arrow words under the kernel of ev_tilde.

} // namespace topcomp
