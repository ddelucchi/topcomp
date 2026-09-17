// ============================================================================
// TopComp: Hybrid Fiber Structure
// ============================================================================
// The hybrid state X^hyb_{A,n} = V_n ⊗_C X_A^top is fibered over
// the discrete support.  Each fiber carries topological content:
//
//   Ψ = Σ_x e_x ⊗ ξ_x
//
// where e_x is exact discrete support and ξ_x is the topological
// content over that support.
//
// Fiber maps:
//   ι_x(ξ)  = e_x ⊗ ξ          injection into fiber x
//   π_x(Ψ)  = ξ_x               projection onto fiber x
//   π_x∘ι_y = δ_{xy} I          orthogonal fiber extraction
//   Σ_x ι_x π_x = I             fiber resolution
//
// Hybrid projectors:
//   P_x^hyb = ι_x π_x           exact fiber projector
//
// Operator lifting:
//   X_u^hyb = X_u ⊗ I           discrete operator on hybrid space
//   U_t^hyb = I ⊗ U_t           topological operator on hybrid space
//   [X_u^hyb, U_t^hyb] = 0      cross-sector commutation
//
// Joint measurement across boundaries:
//   p(x,j | ρ) = Tr(P_x^hyb Π_j^A ρ)
//   with discrete and topological marginals and conditionals.
//
// This completes the doctrine:
//   exact fiber decomposition → operator lifting → joint measurement
// ============================================================================
#pragma once
#include "sector.cuh"

namespace topcomp {
namespace fiber {

using measurement::DenseOp;
using sector::AdmissibleFamily;
using sector::SectorWeights;

// ── Fiber map ───────────────────────────────────────────────────────────────
// Injection/projection pair for the hybrid space V_n ⊗ X_A^top

struct FiberMap {
    int dim_disc;   // n  = |discrete support|
    int dim_top;    // dt = dim(X_A^top)
    int dim_hyb;    // n * dt

    __host__ FiberMap(int dd, int dt)
        : dim_disc(dd), dim_top(dt), dim_hyb(dd * dt) {}

    // ι_x(σ): inject dim_top × dim_top operator into fiber x
    __host__ DenseOp inject(int x, const DenseOp& sigma) const {
        DenseOp result(dim_hyb);
        for (int i = 0; i < dim_top; ++i)
            for (int j = 0; j < dim_top; ++j)
                result.at(x * dim_top + i, x * dim_top + j)
                    = sigma.at(i, j);
        return result;
    }

    // π_x(ρ): extract fiber x from hybrid density
    __host__ DenseOp project(int x, const DenseOp& rho) const {
        DenseOp result(dim_top);
        for (int i = 0; i < dim_top; ++i)
            for (int j = 0; j < dim_top; ++j)
                result.at(i, j)
                    = rho.at(x * dim_top + i, x * dim_top + j);
        return result;
    }

    // P_x^hyb = ι_x π_x (projector onto fiber x)
    __host__ DenseOp hybrid_projector(int x) const {
        DenseOp result(dim_hyb);
        for (int k = 0; k < dim_top; ++k)
            result.at(x * dim_top + k, x * dim_top + k) = C64(1, 0);
        return result;
    }

    // Build AdmissibleFamily of all fiber projectors
    __host__ AdmissibleFamily fiber_family() const {
        AdmissibleFamily fam(dim_hyb, dim_disc);
        for (int x = 0; x < dim_disc; ++x)
            fam.projectors[x] = hybrid_projector(x);
        return fam;
    }
};

// ── Fiber verification ──────────────────────────────────────────────────────

namespace verify {

// π_x ∘ ι_y = δ_{xy} I
__host__ inline bool injection_projection_identity(
    const FiberMap& fm, double tol = 1e-10) {
    DenseOp sigma = DenseOp::identity(fm.dim_top);
    for (int x = 0; x < fm.dim_disc; ++x) {
        for (int y = 0; y < fm.dim_disc; ++y) {
            DenseOp injected = fm.inject(y, sigma);
            DenseOp extracted = fm.project(x, injected);
            if (x == y) {
                for (int i = 0; i < fm.dim_top; ++i)
                    for (int j = 0; j < fm.dim_top; ++j) {
                        C64 expected = (i == j)
                            ? C64(1, 0) : C64(0, 0);
                        if ((extracted.at(i, j) - expected).norm2()
                                > tol * tol)
                            return false;
                    }
            } else {
                for (int i = 0; i < fm.dim_top; ++i)
                    for (int j = 0; j < fm.dim_top; ++j)
                        if (extracted.at(i, j).norm2() > tol * tol)
                            return false;
            }
        }
    }
    return true;
}

// Σ_x ι_x π_x = I (via fiber projector resolution)
__host__ inline bool fiber_resolution(const FiberMap& fm,
                                       double tol = 1e-10) {
    AdmissibleFamily fam = fm.fiber_family();
    return sector::verify::resolution_of_identity(fam, tol);
}

} // namespace verify

// ── Operator lifting ────────────────────────────────────────────────────────
// Lift boundary operators to the full hybrid space

namespace lift {

// X_u^hyb = X_u ⊗ I_top  (discrete operator on hybrid space)
__host__ inline DenseOp disc_operator(const DenseOp& X_u,
                                       int dim_top) {
    int dim_hyb = X_u.dim * dim_top;
    DenseOp result(dim_hyb);
    for (int i = 0; i < X_u.dim; ++i)
        for (int j = 0; j < X_u.dim; ++j)
            for (int k = 0; k < dim_top; ++k)
                result.at(i * dim_top + k, j * dim_top + k)
                    = X_u.at(i, j);
    return result;
}

// U_t^hyb = I_disc ⊗ U_t  (topological operator on hybrid space)
__host__ inline DenseOp top_operator(const DenseOp& U_t,
                                      int dim_disc) {
    int dim_hyb = dim_disc * U_t.dim;
    DenseOp result(dim_hyb);
    for (int x = 0; x < dim_disc; ++x)
        for (int k = 0; k < U_t.dim; ++k)
            for (int l = 0; l < U_t.dim; ++l)
                result.at(x * U_t.dim + k, x * U_t.dim + l)
                    = U_t.at(k, l);
    return result;
}

// Verify [X^hyb, U^hyb] = 0 (cross-sector commutation)
__host__ inline bool cross_sector_commutation(
    const DenseOp& X_hyb, const DenseOp& U_hyb,
    double tol = 1e-10) {
    DenseOp XU = X_hyb * U_hyb;
    DenseOp UX = U_hyb * X_hyb;
    for (int i = 0; i < X_hyb.dim; ++i)
        for (int j = 0; j < X_hyb.dim; ++j)
            if ((XU.at(i, j) - UX.at(i, j)).norm2() > tol * tol)
                return false;
    return true;
}

} // namespace lift

// ── Joint measurement ───────────────────────────────────────────────────────
// p(x,j | ρ) = Tr(P_x^hyb Π_j^A ρ)
// Simultaneous multi-boundary exact sectorization

namespace joint {

struct JointWeights {
    static constexpr int MAX_DISC = 64;
    static constexpr int MAX_TOP  = 64;
    double weights[MAX_DISC][MAX_TOP];
    int n_disc, n_top;

    JointWeights() : n_disc(0), n_top(0) {
        for (int i = 0; i < MAX_DISC; ++i)
            for (int j = 0; j < MAX_TOP; ++j)
                weights[i][j] = 0.0;
    }

    // Discrete marginal: p(x) = Σ_j p(x,j)
    __host__ double disc_marginal(int x) const {
        double s = 0;
        for (int j = 0; j < n_top; ++j) s += weights[x][j];
        return s;
    }

    // Topological marginal: p(j) = Σ_x p(x,j)
    __host__ double top_marginal(int j) const {
        double s = 0;
        for (int x = 0; x < n_disc; ++x) s += weights[x][j];
        return s;
    }

    // Conditional: p(j|x) = p(x,j) / p(x)
    __host__ double conditional_top(int x, int j) const {
        double px = disc_marginal(x);
        return (px > 1e-15) ? weights[x][j] / px : 0.0;
    }
};

// Joint Born weights: p(x,j|ρ) = Tr(P_x Π_j ρ)
__host__ inline JointWeights joint_born_weights(
    const AdmissibleFamily& disc_fam,
    const AdmissibleFamily& top_fam,
    const DenseOp& rho) {
    JointWeights jw;
    jw.n_disc = disc_fam.n_sectors;
    jw.n_top  = top_fam.n_sectors;
    for (int x = 0; x < jw.n_disc; ++x)
        for (int j = 0; j < jw.n_top; ++j) {
            DenseOp PxQj = disc_fam.projectors[x]
                           * top_fam.projectors[j];
            jw.weights[x][j] = (PxQj * rho).trace().re;
        }
    return jw;
}

// Verify Σ_{x,j} p(x,j) = 1
__host__ inline bool verify_normalization(const JointWeights& jw,
                                           double tol = 1e-10) {
    double total = 0;
    for (int x = 0; x < jw.n_disc; ++x)
        for (int j = 0; j < jw.n_top; ++j)
            total += jw.weights[x][j];
    return fabs(total - 1.0) < tol;
}

// Verify marginal consistency: both marginals sum to 1
__host__ inline bool verify_marginal_consistency(
    const JointWeights& jw, double tol = 1e-10) {
    double disc_sum = 0;
    for (int x = 0; x < jw.n_disc; ++x)
        disc_sum += jw.disc_marginal(x);
    if (fabs(disc_sum - 1.0) > tol) return false;
    double top_sum = 0;
    for (int j = 0; j < jw.n_top; ++j)
        top_sum += jw.top_marginal(j);
    if (fabs(top_sum - 1.0) > tol) return false;
    return true;
}

} // namespace joint

} // namespace fiber
} // namespace topcomp
