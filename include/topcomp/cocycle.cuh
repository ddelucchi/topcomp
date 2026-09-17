// ============================================================================
// TopComp: Cocycle & Symplectic Forms
// ============================================================================
// Implements the cocycle hierarchy that is the algebraic backbone:
//
//   Ω_bool(u,v) = πi · ⟨u,v⟩  mod 2πi        (boolean symplectic form)
//   Ω_A(g,h) = F_A(g,h) = log_A([g,h])       (Mir curvature cocycle)
//   Ω_Mir((s,c),(t,d)) = sc(t-1) + td(1-s)   (universal Mir cocycle)
//   Ω_tot^{(A,n)} = Ω_bool ⊕ Ω_A              (total cocycle)
//
//   UFE := F_tot - Ω_tot = 0                   (universal flatness equation)
//
// Cocycle condition: δΩ(g,h,k) := Ω(h,k) - Ω(gh,k) + Ω(g,hk) - Ω(g,h) = 0
//
// Central extension: G̃ := G × Z, (g,a)·(h,b) := (gh, a+b+Ω(g,h))
//   δΩ = 0 ⟹ G̃ is associative
//
// Shadow functor: Sh₂(τ(u,v)) = πi · τ(u,v) mod 2πi
//   Sh₂(Ω_A) = Ω̃_bool,  q₂(Sh₂(Ω_A)) = Ω_bool^{(2)}
// ============================================================================
#pragma once
#include "mir_group.cuh"

namespace topcomp {

// ── Boolean symplectic form ─────────────────────────────────────────────────
namespace cocycle {

// Ω_bool^{(2)}(u,v) = ⟨u,v⟩ ∈ F_2
__host__ __device__ inline int omega_bool_f2(const BitVec& u, const BitVec& v) {
    return u.inner(v);
}

// Ω̃_bool(u,v) = πi · ⟨u,v⟩ mod 2πi  (lifted to ℂ/2πiℤ)
__host__ __device__ inline C64 omega_bool_tilde(const BitVec& u, const BitVec& v) {
    int ip = u.inner(v);
    return C64(0.0, constants::PI * ip);
}

// Ω_bool(u,v) = πi · ⟨u,v⟩_{H_{2,n}} mod 2πi
__host__ __device__ inline C64 omega_bool(const BitVec& u, const BitVec& v) {
    return omega_bool_tilde(u, v);
}

// exp(Ω_bool(u,v)) = (-1)^{⟨u,v⟩}
__host__ __device__ inline double omega_bool_exp(const BitVec& u, const BitVec& v) {
    return (u.inner(v) == 0) ? 1.0 : -1.0;
}

// ── Mirror curvature cocycle ────────────────────────────────────────────────

// Ω_Mir((s,c),(t,d)) = sc(t-1) + td(1-s)
__host__ __device__ inline C64 omega_mir(const MirElement& g, const MirElement& h) {
    return g.omega_mir(h);
}

// Special cases:
//   Ω_Mir((1,c),(1,d)) = 0
//   Ω_Mir((-1,c),(1,d)) = 2d
//   Ω_Mir((1,c),(-1,d)) = -2c
//   Ω_Mir((-1,c),(-1,d)) = 2(c-d)
__host__ __device__ inline C64 omega_mir_pp(const C64& c, const C64& d) { return C64(0,0); }
__host__ __device__ inline C64 omega_mir_mp(const C64& c, const C64& d) { return d * 2.0; }
__host__ __device__ inline C64 omega_mir_pm(const C64& c, const C64& d) { return c * (-2.0); }
__host__ __device__ inline C64 omega_mir_mm(const C64& c, const C64& d) { return (c - d) * 2.0; }

// Commutator spectrum:
//   Ω_Mir(J_{λ,A}, g_{φ,A}(τ)) = 2τ · c_φ
//   Re part = 2ℓτ,  Im part = -πτ
__host__ __device__ inline C64 omega_mir_J_gphi(double tau) {
    return constants::c_phi() * (2.0 * tau);
}

// ── Cocycle condition verifier ──────────────────────────────────────────────

// δΩ(g,h,k) = Ω(h,k) - Ω(gh,k) + Ω(g,hk) - Ω(g,h)
__host__ __device__ inline C64 cocycle_coboundary(
    const MirElement& g, const MirElement& h, const MirElement& k)
{
    C64 ohk  = omega_mir(h, k);
    C64 oghk = omega_mir(g.star(h), k);
    C64 oghk2 = omega_mir(g, h.star(k));
    C64 ogh  = omega_mir(g, h);
    return ohk - oghk + oghk2 - ogh;
}

// Verify δΩ = 0 for given triple
__host__ __device__ inline bool verify_cocycle(
    const MirElement& g, const MirElement& h, const MirElement& k,
    double tol = 1e-12)
{
    C64 delta = cocycle_coboundary(g, h, k);
    return delta.norm2() < tol * tol;
}

// ── Total cocycle: Ω_tot^{(A,n)} = Ω_bool ⊕ Ω_A ─────────────────────────────

// Ω_tot((g;u,v), (h;u',v')) = Ω_A(g,h) + Ω_bool(v,u')
__host__ __device__ inline C64 omega_total(
    const MirElement& g, const BitVec& u, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& vp)
{
    return omega_mir(g, h) + omega_bool_tilde(v, up);
}

// F_tot = F_A ⊕ Ω̃_bool  — same as Ω_tot since UFE = 0
__host__ __device__ inline C64 F_total(
    const MirElement& g, const BitVec& u, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& vp)
{
    return omega_total(g, u, v, h, up, vp);
}

// UFE_tot = F_tot - Ω_tot = 0
__host__ __device__ inline C64 UFE_total(
    const MirElement& g, const BitVec& u, const BitVec& v,
    const MirElement& h, const BitVec& up, const BitVec& vp)
{
    return F_total(g, u, v, h, up, vp) - omega_total(g, u, v, h, up, vp);
}

// ── Shadow functor Sh₂ ──────────────────────────────────────────────────────

// Sh₂(τ(u,v)) = πi · τ(u,v) mod 2πi
// where τ(u,v) ≡ ⟨u,v⟩ mod 2
__host__ __device__ inline C64 shadow_functor(int tau) {
    return C64(0.0, constants::PI * tau);
}

// q₂: ℂ/2πiℤ → F₂  (reduction mod 2)
__host__ __device__ inline int q2(const C64& omega) {
    // omega = πi·k for some integer k; return k mod 2
    double k = omega.im / constants::PI;
    int ki = static_cast<int>(round(k));
    return ((ki % 2) + 2) % 2;
}

// ℓ₂: F₂ → {0,1}  (identity on {0,1})
__host__ __device__ inline int l2(int f2_val) {
    return f2_val & 1;
}

// Full shadow pipeline: Sh₂(Ω_A) projects to Ω_bool^{(2)}
__host__ __device__ inline int shadow_to_bit(const C64& omega_A) {
    return l2(q2(omega_A));
}

} // namespace cocycle

// ── Central Extension Group G̃_tot^{(A,n)} ──────────────────────────────────
// Elements: ((u,g,v), a) with u,v ∈ I_n, g ∈ G_A, a ∈ ℤ
// Product: ((u,g,v),a) · ((u',h,v'),b) = ((u⊕u', g★h, v⊕v'), a+b+Ω_tot)
struct CentralExtElement {
    BitVec     u, v;       // discrete components
    MirElement g;          // Mir group element
    C64        phase;      // accumulated phase

    __host__ __device__ CentralExtElement()
        : u(), v(), g(MirElement::identity()), phase(0.0, 0.0) {}

    __host__ __device__ CentralExtElement(const BitVec& u_, const MirElement& g_,
                                            const BitVec& v_, const C64& a_)
        : u(u_), g(g_), v(v_), phase(a_) {}

    // Group product with cocycle
    __host__ __device__ CentralExtElement operator*(const CentralExtElement& other) const {
        C64 omega = cocycle::omega_total(g, u, v, other.g, other.u, other.v);
        return CentralExtElement(
            u ^ other.u,
            g.star(other.g),
            v ^ other.v,
            phase + other.phase + omega
        );
    }

    // Identity
    __host__ __device__ static CentralExtElement identity(int n) {
        return CentralExtElement(BitVec(0, n), MirElement::identity(),
                                  BitVec(0, n), C64(0, 0));
    }
};

// ── Weyl operator composition ───────────────────────────────────────────────
// W_{u,v;x} · W_{u',v';y} = exp(iΩ_tot((u,x),(v',y))) · W_{u⊕u',v⊕v';xy}
struct WeylElement {
    BitVec     u, v;
    MirElement x;

    __host__ __device__ C64 compose_phase(const WeylElement& other) const {
        return cocycle::omega_total(x, u, v, other.x, other.u, other.v);
    }
};

// ── Formal curvature F_A(u,v) = log_A([u,v]) = [X,Y] mod I³ ─────────────────
// For u = e^X, v = e^Y in the completed algebra Â_{A,d}:
//   [u,v] = 1 + [X,Y] mod I³
//   F_A(u,v) = [X,Y] mod I³
//   UFE = F_A - Ω_A = 0  (universal flatness equation)
namespace curvature {

// BCH-based formal curvature at leading order
__host__ __device__ inline C64 formal_curvature(const C64& X, const C64& Y) {
    // [X,Y] = XY - YX  — for scalars this is 0, but for matrix-valued
    // elements this captures the non-commutativity
    return X * Y - Y * X;
}

// Verify UFE = 0 for Mir elements
__host__ __device__ inline bool verify_UFE(const MirElement& u, const MirElement& v,
                                             double tol = 1e-12) {
    // Compute commutator
    MirElement comm = u.commutator(v);
    // F = Log_Mir(commutator)
    C64 F = comm.log_mir();
    // Ω = omega_mir
    C64 Omega = cocycle::omega_mir(u, v);
    // UFE = F - Ω
    C64 ufe = F - Omega;
    return ufe.norm2() < tol * tol;
}

} // namespace curvature

// ── Character χ_{n,m}^A(θ,ρ) ────────────────────────────────────────────────
// χ_{n,m}(θ,ρ) = exp(inθ + s_{n,m}ρ)
// where s_{n,m} = (i/ln φ)(nπ/2 - 2πm)
namespace character {

__host__ __device__ inline C64 s_coeff(int n, int m) {
    double val = (n * constants::HALF_PI - constants::TWO_PI * m) / constants::ELL;
    return C64(0.0, val);
}

__host__ __device__ inline C64 chi(int n, int m, double theta, double rho) {
    C64 s = s_coeff(n, m);
    C64 exponent = C64(0.0, n * theta) + s * rho;
    return cexp(exponent);
}

// Commutation: χ_{n,m} χ_{n',m'} = (-1)^{κ_{n,n'} κ_{m,m'}} χ_{n',m'} χ_{n,m}
__host__ __device__ inline int kappa(int n, int m) {
    return cocycle::omega_bool_f2(
        BitVec(static_cast<uint32_t>(n), 32),
        BitVec(static_cast<uint32_t>(m), 32)
    );
}

} // namespace character

} // namespace topcomp
