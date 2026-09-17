// ============================================================================
// TopComp: Mathematical Constants
// ============================================================================
// All constants from the Mirror framework lattice L_{φ,2π}:
//   φ (golden ratio), a = 2π, b = φ, ℓ = ln φ,
//   c_φ = ℓ - iπ/2, c_π = -2πi, ΔΣ = ab = 2πφ
//   K_log := Frac(ℚ[π, φ, ln φ, ln a] / (φ² - φ - 1))
// ============================================================================
#pragma once
#include "complex.cuh"

namespace topcomp {

// ── Fundamental constants ───────────────────────────────────────────────────
namespace constants {

// π and 2π
static constexpr double PI      = 3.14159265358979323846;
static constexpr double TWO_PI   = 6.28318530717958647692;
static constexpr double HALF_PI  = 1.57079632679489661923;

// Golden ratio: φ = (1+√5)/2,  φ² = φ+1
static constexpr double PHI      = 1.61803398874989484820;
static constexpr double PHI_INV  = 0.61803398874989484820; // 1/φ = φ-1

// a := 2π,  b := φ  (renamed to match paper notation)
static constexpr double A_CONST  = TWO_PI;
static constexpr double B_CONST  = PHI;

// ΔΣ := ab = 2πφ
static constexpr double DELTA_SIGMA = TWO_PI * PHI; // ≈ 10.166

// ℓ := ln φ
static constexpr double ELL      = 0.48121182505960344750; // ln(φ)

// ln a = ln(2π)
static constexpr double LN_A     = 1.83787706640934548356; // ln(2π)

// ln(ΔΣ) = ln a + ℓ
static constexpr double LN_DELTA_SIGMA = LN_A + ELL;

// k = 2·ln(φ)/π  (spiral exponent)
static constexpr double SPIRAL_K = 2.0 * ELL / PI;

// ── Complex lattice constants (in ℂ) ────────────────────────────────────────
// c_φ = ℓ - iπ/2  (complex translation for →)
__host__ __device__ inline C64 c_phi() { return C64(ELL, -HALF_PI); }

// c_π = -2πi  (complex translation for full rotation)
__host__ __device__ inline C64 c_pi() { return C64(0.0, -TWO_PI); }

// ln ΔΣ as complex (purely real)
__host__ __device__ inline C64 c_delta_sigma() { return C64(LN_DELTA_SIGMA, 0.0); }

// ── Arrow displacement constants ────────────────────────────────────────────
// c_→ = c_φ = ℓ - iπ/2
__host__ __device__ inline C64 c_right()    { return c_phi(); }
// c_← = -c_φ
__host__ __device__ inline C64 c_left()     { return -c_phi(); }
// c_⟸ = ln(ΔΣ)
__host__ __device__ inline C64 c_lldir()    { return c_delta_sigma(); }
// c_⟹ = -ln(ΔΣ)
__host__ __device__ inline C64 c_rrdir()    { return -c_delta_sigma(); }

// ── Lattice L_{φ,2π}^{add} := ℤ·ln a ⊕ ℤ·ℓ ⊕ (πi/2)ℤ ────────────────────────
// Any lattice element: n₁·ln a + n₂·ℓ + n₃·(πi/2)
__host__ __device__ inline C64 lattice_element(int n1, int n2, int n3) {
    return C64(n1 * LN_A + n2 * ELL, n3 * HALF_PI);
}

// ── Mirror lattice L^{Mir}_{φ,2π} := ℤ·c_φ ⊕ ℤ·ln(ΔΣ) ───────────────────────
// m·c_φ + n·ln(ΔΣ)
__host__ __device__ inline C64 mir_lattice(int m, int n) {
    double re_part = n * LN_A + (m + n) * ELL;
    double im_part = -m * HALF_PI;
    return C64(re_part, im_part);
}

// exp(m·c_φ + n·ln(ΔΣ)) = a^n · φ^{m+n} · e^{-iπm/2}
__host__ __device__ inline C64 mir_lattice_exp(int m, int n) {
    return cexp(mir_lattice(m, n));
}

// ── Scale set Λ := {φ^{2m} · a^{2n} : m,n ∈ ℤ} ⊂ ℝ₊ ─────────────────────────
__host__ __device__ inline double lambda_scale(int m, int n) {
    return pow(PHI, 2.0 * m) * pow(A_CONST, 2.0 * n);
}

// ── Scale set S = {±s₁, ±s₂, ±s₃, ±s₄} ──────────────────────────────────────
// s₁ = ΔΣ,  s₂ = a/b = 2π/φ,  s₃ = b/a = φ/(2π),  s₄ = ΔΣ⁻¹
__host__ __device__ inline double s1() { return DELTA_SIGMA; }
__host__ __device__ inline double s2() { return A_CONST / B_CONST; }  // 2π/φ
__host__ __device__ inline double s3() { return B_CONST / A_CONST; }  // φ/(2π)
__host__ __device__ inline double s4() { return 1.0 / DELTA_SIGMA; } // ΔΣ⁻¹

// Verify s₁·s₄ = 1, s₂·s₃ = 1, s₁ = a·b, s₂·s₃ inversions
__host__ inline bool verify_scale_set(double tol = 1e-14) {
    return fabs(s1() * s4() - 1.0) < tol &&
           fabs(s2() * s3() - 1.0) < tol &&
           fabs(s1() - A_CONST * B_CONST) < tol;
}

// ── Lattice Λ̃ := ⟨a, b⟩_ℤ = { a^m b^n : m,n ∈ ℤ } ⊂ ℝ₊ ─────────────────────
// ln Λ̃ = { m·ln a + n·ln b : m,n ∈ ℤ } = { m·ln a + n·ℓ }
__host__ __device__ inline double lambda_tilde(int m, int n) {
    return pow(A_CONST, (double)m) * pow(B_CONST, (double)n);
}

// ── Period group P = (π/2)ℤ × Λ̃ ────────────────────────────────────────────
// Element of P: (α, λ) where α ∈ (π/2)ℤ, λ ∈ Λ̃
__host__ __device__ inline double period_alpha(int k) { return k * HALF_PI; }

// ── ker(q) = 2πℤ × {0} ──────────────────────────────────────────────────────
// q(θ,ρ) = (θ mod 2π, ρ) = 0 ⟺ θ ∈ 2πℤ ∧ ρ = 0
// (θ,ρ) ~ (θ',ρ') ⟺ ρ = ρ' ∧ (θ-θ') ∈ 2πℤ
// M̃/(2πℤ × {0}) ≅ M
__host__ __device__ inline bool in_kernel_q(double theta, double rho, double tol = 1e-10) {
    double t_mod = fmod(theta, TWO_PI);
    if (t_mod < 0) t_mod += TWO_PI;
    return (t_mod < tol || fabs(t_mod - TWO_PI) < tol) && fabs(rho) < tol;
}

// Equivalence class: (θ,ρ) ~ (θ',ρ') iff same image under q
__host__ __device__ inline bool q_equivalent(double theta1, double rho1,
                                              double theta2, double rho2,
                                              double tol = 1e-10) {
    if (fabs(rho1 - rho2) > tol) return false;
    double diff = fmod(theta1 - theta2, TWO_PI);
    if (diff < 0) diff += TWO_PI;
    return diff < tol || fabs(diff - TWO_PI) < tol;
}

// ── ζ = e^{iπ/5} — golden root of unity ─────────────────────────────────────
// ζ + ζ⁻¹ + 1 = e^{iπ/5} + e^{-iπ/5} + 1 = 2cos(π/5) + 1 = φ²
__host__ __device__ inline C64 zeta() { return cexp(C64(0, PI / 5.0)); }

__host__ inline bool verify_zeta_phi_relation(double tol = 1e-14) {
    C64 z = zeta();
    C64 z_inv = cexp(C64(0, -PI / 5.0));
    double lhs = (z + z_inv + C64(1, 0)).re;  // 2cos(π/5) + 1
    return fabs(lhs - PHI * PHI) < tol;
}

// ── Twist ratio: ς(ρ) = (π / 2ln φ) · ρ ─────────────────────────────────────
__host__ __device__ inline double varsigma(double rho) {
    return (HALF_PI / ELL) * rho;
}

} // namespace constants
} // namespace topcomp
