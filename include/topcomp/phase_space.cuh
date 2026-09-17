// ============================================================================
// TopComp: Phase Space & Arrow Alphabet
// ============================================================================
// Phase space M = Θ × ℝ with Θ = ℝ/(2πℤ)
//   Coordinates: (θ, ρ) ∈ M
//   Complex:     w = ρ + iθ,  z = e^w = e^ρ · e^{iθ}
//
// Arrow alphabet Σ_arr = {→, ←, ⟸, ⟹, ↔_λ : λ ∈ Λ}
// Evaluation map ev: Σ_arr → Mir_ℝ
//   ev(→) = B_{+1; π/2, ln φ} = (+1, c_φ)
//   ev(←) = B_{+1; -π/2, -ln φ} = (+1, -c_φ)
//   ev(⟸) = B_{+1; 0, ln(ΔΣ)} = (+1, ln(ΔΣ))
//   ev(⟹) = B_{+1; 0, -ln(ΔΣ)} = (+1, -ln(ΔΣ))
//   ev(↔_λ) = B_{-1; 0, ln λ} = (-1, ln λ)
//
// Invariants:  u(θ,ρ) = θ + ς(ρ)   (φ-flow invariant)
//              ω̃(θ,ρ) = θ - ς(ρ)   (descends at rate -π per step)
//              ω = ω̃ mod π         (integer-step invariant)
// ============================================================================
#pragma once
#include "mir_group.cuh"

namespace topcomp {

// ── Arrow type enumeration ──────────────────────────────────────────────────
enum class Arrow : int {
    RIGHT  = 0,   // →  forward golden-ratio step
    LEFT   = 1,   // ←  backward golden-ratio step
    LLDIR  = 2,   // ⟸  forward scale (by ΔΣ)
    RRDIR  = 3,   // ⟹  backward scale (by ΔΣ⁻¹)
    MIRROR = 4    // ↔_λ  mirror inversion (parametric)
};

// ── PhasePoint: (θ, ρ) ∈ M = Θ × ℝ ──────────────────────────────────────────
struct PhasePoint {
    double theta;  // angular coordinate, mod 2π
    double rho;    // radial (log-radius) coordinate

    __host__ __device__ constexpr PhasePoint() : theta(0), rho(0) {}
    __host__ __device__ constexpr PhasePoint(double t, double r) : theta(t), rho(r) {}

    // w = ρ + iθ
    __host__ __device__ C64 to_w() const { return C64(rho, theta); }

    // z = e^w
    __host__ __device__ C64 to_z() const { return cexp(to_w()); }

    // r = e^ρ  (actual radius)
    __host__ __device__ double radius() const { return exp(rho); }

    // Cartesian: (x, y) = (r cos θ, r sin θ)
    __host__ __device__ void to_cartesian(double& x, double& y) const {
        double r = radius();
        x = r * cos(theta);
        y = r * sin(theta);
    }

    // ── Invariants ──────────────────────────────────────────────────────────

    // ς(ρ) = (π/2ℓ)·ρ  where ℓ = ln φ
    __host__ __device__ double varsigma() const {
        return constants::varsigma(rho);
    }

    // u = θ + ς(ρ)  — conserved by U_φ flow
    __host__ __device__ double u_invariant() const {
        return theta + varsigma();
    }

    // ω̃ = θ - ς(ρ)  — decreases by π per unit step
    __host__ __device__ double omega_tilde() const {
        return theta - varsigma();
    }

    // ω = ω̃ mod π  — conserved at integer steps
    __host__ __device__ double omega() const {
        double wt = omega_tilde();
        double result = fmod(wt, constants::PI);
        if (result < 0) result += constants::PI;
        return result;
    }

    // Normalize θ to [0, 2π)
    __host__ __device__ PhasePoint normalize() const {
        double t = fmod(theta, constants::TWO_PI);
        if (t < 0) t += constants::TWO_PI;
        return PhasePoint(t, rho);
    }

    // ── Golden spiral parametrization ───────────────────────────────────────
    // On the invariant curve {p(θ) ≡ B}: ρ(θ) = B - kθ, r(θ) = r₀·φ^{-2θ/π}
    __host__ __device__ static PhasePoint from_spiral(double theta, double r0) {
        double rho = log(r0) - constants::SPIRAL_K * theta;
        return PhasePoint(theta, rho);
    }

    // Arc-length differential: ds = r·√(1 + k²) |dθ|
    __host__ __device__ double ds_dtheta() const {
        double k = constants::SPIRAL_K;
        return radius() * sqrt(1.0 + k * k);
    }
};

// ── ArrowAlphabet: evaluation map ev: Σ_arr → Mir_ℝ ─────────────────────────
struct ArrowAlphabet {
    // ev(→) = (+1, c_φ) = B_{+1; π/2, ln φ}
    __host__ __device__ static MirElement ev_right() {
        return MirElement(+1, constants::c_right());
    }
    // ev(←) = (+1, -c_φ)
    __host__ __device__ static MirElement ev_left() {
        return MirElement(+1, constants::c_left());
    }
    // ev(⟸) = (+1, ln(ΔΣ))
    __host__ __device__ static MirElement ev_lldir() {
        return MirElement(+1, constants::c_lldir());
    }
    // ev(⟹) = (+1, -ln(ΔΣ))
    __host__ __device__ static MirElement ev_rrdir() {
        return MirElement(+1, constants::c_rrdir());
    }
    // ev(↔_λ) = (-1, ln λ)
    __host__ __device__ static MirElement ev_mirror(double lambda) {
        return MirElement(-1, C64(log(lambda), 0.0));
    }

    // Evaluate any arrow
    __host__ __device__ static MirElement ev(Arrow arrow, double lambda = 1.0) {
        switch (arrow) {
            case Arrow::RIGHT:  return ev_right();
            case Arrow::LEFT:   return ev_left();
            case Arrow::LLDIR:  return ev_lldir();
            case Arrow::RRDIR:  return ev_rrdir();
            case Arrow::MIRROR: return ev_mirror(lambda);
            default:            return MirElement::identity();
        }
    }

    // ── Apply arrow to phase point ──────────────────────────────────────────

    // →(θ,ρ) = U_φ(θ,ρ) = (θ - π/2, ρ + ln φ)
    __host__ __device__ static PhasePoint apply_right(const PhasePoint& p) {
        return PhasePoint(p.theta - constants::HALF_PI, p.rho + constants::ELL);
    }

    // ←(θ,ρ) = U_φ⁻¹(θ,ρ) = (θ + π/2, ρ - ln φ)
    __host__ __device__ static PhasePoint apply_left(const PhasePoint& p) {
        return PhasePoint(p.theta + constants::HALF_PI, p.rho - constants::ELL);
    }

    // ⟸(θ,ρ) = (θ, ρ + ln(ΔΣ))
    __host__ __device__ static PhasePoint apply_lldir(const PhasePoint& p) {
        return PhasePoint(p.theta, p.rho + constants::LN_DELTA_SIGMA);
    }

    // ⟹(θ,ρ) = (θ, ρ - ln(ΔΣ))
    __host__ __device__ static PhasePoint apply_rrdir(const PhasePoint& p) {
        return PhasePoint(p.theta, p.rho - constants::LN_DELTA_SIGMA);
    }

    // ↔_λ(θ,ρ) = (-θ, -ρ + ln λ)
    __host__ __device__ static PhasePoint apply_mirror(const PhasePoint& p, double lambda) {
        return PhasePoint(-p.theta, -p.rho + log(lambda));
    }

    // General arrow application
    __host__ __device__ static PhasePoint apply(Arrow arrow, const PhasePoint& p,
                                                 double lambda = 1.0) {
        switch (arrow) {
            case Arrow::RIGHT:  return apply_right(p);
            case Arrow::LEFT:   return apply_left(p);
            case Arrow::LLDIR:  return apply_lldir(p);
            case Arrow::RRDIR:  return apply_rrdir(p);
            case Arrow::MIRROR: return apply_mirror(p, lambda);
            default:            return p;
        }
    }

    // ── Arrow word evaluation: eṽ(σ₁···σ_N) ────────────────────────────────
    __host__ __device__ static MirElement eval_word(const Arrow* word, int N,
                                                      double lambda = 1.0) {
        MirElement result = MirElement::identity();
        for (int i = 0; i < N; ++i) {
            result = result.star(ev(word[i], lambda));
        }
        return result;
    }

    // Apply a word of arrows to a phase point
    __host__ __device__ static PhasePoint apply_word(const Arrow* word, int N,
                                                       const PhasePoint& p,
                                                       double lambda = 1.0) {
        PhasePoint result = p;
        for (int i = 0; i < N; ++i) {
            result = apply(word[i], result, lambda);
        }
        return result;
    }
};

// ── U_φ^t flow: continuous golden-ratio evolution ───────────────────────────
struct GoldenFlow {
    // U_φ^t(θ,ρ) = (θ - πt/2, ρ + t·ln φ)
    __host__ __device__ static PhasePoint flow(const PhasePoint& p, double t) {
        return PhasePoint(
            p.theta - constants::HALF_PI * t,
            p.rho + constants::ELL * t
        );
    }

    // Action on w: U_φ^t · w = w + t·c_φ
    __host__ __device__ static C64 flow_w(const C64& w, double t) {
        return w + constants::c_phi() * t;
    }

    // Action on z: U_φ^t · z = e^{t·c_φ} · z = φ^t · e^{-iπt/2} · z
    __host__ __device__ static C64 flow_z(const C64& z, double t) {
        return cexp(constants::c_phi() * t) * z;
    }

    // Generate orbit: compute T steps from initial point
    __host__ __device__ static void orbit(const PhasePoint& p0, PhasePoint* out, int T) {
        for (int t = 0; t < T; ++t) {
            out[t] = flow(p0, static_cast<double>(t));
        }
    }
};

// ── Projective line representation: PGL₂ ────────────────────────────────────
// ρ_{P¹}(→) = [[e^{c_φ/2}, 0], [0, e^{-c_φ/2}]]
// ρ_{P¹}(↔_λ) = [[0, e^{ln λ/2}], [e^{-ln λ/2}, 0]]
struct PGL2Element {
    C64 a, b, c, d;  // [[a,b],[c,d]]

    __host__ __device__ PGL2Element() : a(1), b(0), c(0), d(1) {}
    __host__ __device__ PGL2Element(C64 a_, C64 b_, C64 c_, C64 d_)
        : a(a_), b(b_), c(c_), d(d_) {}

    // Möbius action: t ↦ (at + b)/(ct + d)
    __host__ __device__ C64 act(const C64& t) const {
        return (a * t + b) / (c * t + d);
    }

    // Matrix multiply
    __host__ __device__ PGL2Element operator*(const PGL2Element& o) const {
        return PGL2Element(
            a * o.a + b * o.c, a * o.b + b * o.d,
            c * o.a + d * o.c, c * o.b + d * o.d
        );
    }

    // From MirElement to PGL₂
    __host__ __device__ static PGL2Element from_mir(const MirElement& g) {
        if (g.s == +1) {
            C64 half_c = g.c * 0.5;
            return PGL2Element(cexp(half_c), C64(0), C64(0), cexp(-half_c));
        } else {
            C64 half_c = g.c * 0.5;
            return PGL2Element(C64(0), cexp(half_c), cexp(-half_c), C64(0));
        }
    }
};

// ── Lie algebra generators: D, E, K ─────────────────────────────────────────
// [D,E] = D,  [D,K] = 2E,  [E,K] = K — the sl₂ structure
namespace lie {
    // D = d/dt (translation generator)
    __host__ __device__ inline PGL2Element exp_D(double h) {
        return PGL2Element(C64(1), C64(h), C64(0), C64(1));
    }
    // E = t·d/dt (dilation generator)
    __host__ __device__ inline PGL2Element exp_E(double s) {
        return PGL2Element(C64(exp(s)), C64(0), C64(0), C64(1));
    }
    // K = t²·d/dt (special conformal generator)
    __host__ __device__ inline PGL2Element exp_K(double kappa) {
        return PGL2Element(C64(1), C64(0), C64(-kappa), C64(1));
    }
    // J = inversion: t ↦ 1/t
    __host__ __device__ inline PGL2Element J() {
        return PGL2Element(C64(0), C64(1), C64(1), C64(0));
    }
} // namespace lie

// ── Conformal metric on ℂ× ──────────────────────────────────────────────────
// Reference Section 11:
//   dz = z·dw,  ds* = |dz|/|z| = |dw|
//   dw = dρ + i·dθ,  ds*² = dρ² + dθ²
//   g* = dρ⊗dρ + dθ⊗dθ  (flat metric on M̃)
//   r = e^ρ,  dr = r·dρ,  ds² = |dz|² = dr² + r²dθ²  (Euclidean on ℂ×)
namespace conformal_metric {

// Conformal line element: ds*² = dρ² + dθ² (flat)
__host__ __device__ inline double ds_star_squared(double d_rho, double d_theta) {
    return d_rho * d_rho + d_theta * d_theta;
}

// Euclidean line element: ds² = dr² + r²dθ² where r = e^ρ
__host__ __device__ inline double ds_euclidean_squared(double rho, double d_rho,
                                                       double d_theta) {
    double r = exp(rho);
    double dr = r * d_rho;
    return dr * dr + r * r * d_theta * d_theta;
}

// Verify ds*² = ds²/|z|² (conformal equivalence)
__host__ inline bool verify_conformal_relation(double rho, double d_rho,
                                                double d_theta, double tol = 1e-10) {
    double z_sq = exp(2.0 * rho);
    double ds2 = ds_euclidean_squared(rho, d_rho, d_theta);
    double ds_star2 = ds_star_squared(d_rho, d_theta);
    return fabs(ds2 / z_sq - ds_star2) < tol;
}

// Geodesic distance in conformal metric: d*(w₁,w₂) = |w₁-w₂|
__host__ __device__ inline double conformal_distance(const PhasePoint& p1,
                                                      const PhasePoint& p2) {
    double dr = p1.rho - p2.rho;
    double dt = p1.theta - p2.theta;
    return sqrt(dr * dr + dt * dt);
}

} // namespace conformal_metric

} // namespace topcomp
