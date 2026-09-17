// ============================================================================
// TopComp: Admissible Sector Decomposition
// ============================================================================
// Core doctrinal structure:
//
//   Any admissible object — discrete, continuous, symbolic, topological,
//   operational, semantic, structural, local, global, or joint — is
//   represented not as a vague probability cloud awaiting exact collapse,
//   but as an exact decomposition of the hybrid state into lawful
//   measurable sectors.  The probability law gives the weight of each
//   exact sector within the total object.
//
//   Probability is not a substitute for exactness.
//   It is the measure placed on an already exact decomposition.
//   The exactness is not created by measurement.
//   It is already present in the sector architecture.
//
// Information becomes:
//   exact sector architecture + probability measure over that architecture
//
// Mathematical form:
//
//   Discrete resolution of the identity (PVM):
//     P_a P_b = δ_{ab} P_a,   Σ_{a∈A} P_a = I
//     p(a | ρ) = Tr(P_a ρ)
//     ρ = Σ_a P_a ρ P_a               (exact sector decomposition)
//
//   Spectral (continuous) measure:
//     E(Δ₁)E(Δ₂) = E(Δ₁∩Δ₂),   E(A) = I
//     p(Δ | ρ) = Tr(E(Δ) ρ)
//
//   Compound admissible labels:
//     a ∈ A_disc × A_top × A_op × A_struct
//     P_{(a,b,...)} = P_a ⊗ Q_b ⊗ ...
//
// Key objects:
//   AdmissibleFamily  — orthogonal projector resolution {P_a}_{a∈A}
//   SectorWeights     — Born weights {p(a|ρ)} over exact sectors
//   compound          — tensor product of families (joint sectors)
//   spectral          — discretized spectral measure
// ============================================================================
#pragma once
#include "measurement.cuh"

namespace topcomp {
namespace sector {

using measurement::DenseOp;

// ── Sector weight vector ────────────────────────────────────────────────────
// Born weights p(a|ρ) = Tr(P_a ρ) for each sector a ∈ A

struct SectorWeights {
    static constexpr int MAX_SECTORS = 256;
    double weights[MAX_SECTORS];
    int    n_sectors;

    SectorWeights() : n_sectors(0) {
        for (int i = 0; i < MAX_SECTORS; ++i) weights[i] = 0.0;
    }
};

// ── Admissible family ───────────────────────────────────────────────────────
// Orthogonal resolution of the identity: {P_a}_{a ∈ A}
//   P_a P_b = δ_{ab} P_a    (orthogonal idempotents)
//   Σ_{a ∈ A} P_a = I       (completeness)

struct AdmissibleFamily {
    DenseOp* projectors;      // owned array of n_sectors projectors
    int      n_sectors;       // |A|
    int      dim;             // Hilbert space dimension

    __host__ AdmissibleFamily()
        : projectors(nullptr), n_sectors(0), dim(0) {}

    __host__ AdmissibleFamily(int d, int n)
        : n_sectors(n), dim(d) {
        projectors = new DenseOp[n_sectors];
    }

    __host__ ~AdmissibleFamily() { delete[] projectors; }

    __host__ AdmissibleFamily(const AdmissibleFamily& o)
        : n_sectors(o.n_sectors), dim(o.dim) {
        projectors = new DenseOp[n_sectors];
        for (int a = 0; a < n_sectors; ++a)
            projectors[a] = o.projectors[a];
    }

    __host__ AdmissibleFamily& operator=(const AdmissibleFamily& o) {
        if (this != &o) {
            delete[] projectors;
            n_sectors = o.n_sectors;
            dim = o.dim;
            projectors = new DenseOp[n_sectors];
            for (int a = 0; a < n_sectors; ++a)
                projectors[a] = o.projectors[a];
        }
        return *this;
    }

    // ── Named constructors ──────────────────────────────────────────────────

    // Computational basis: P_x = |x⟩⟨x| for x ∈ {0,...,d-1}
    __host__ static AdmissibleFamily computational_basis(int d) {
        AdmissibleFamily fam(d, d);
        for (int x = 0; x < d; ++x) {
            fam.projectors[x] = DenseOp(d);
            fam.projectors[x].at(x, x) = C64(1, 0);
        }
        return fam;
    }

    // Discrete sectors of a hybrid space: P_x = |x⟩⟨x| ⊗ I_top
    // Decomposes X^hyb = V_n ⊗ X^top into its discrete slices
    __host__ static AdmissibleFamily disc_sectors(int dim_disc,
                                                   int dim_top) {
        int total = dim_disc * dim_top;
        AdmissibleFamily fam(total, dim_disc);
        for (int x = 0; x < dim_disc; ++x) {
            fam.projectors[x] = DenseOp(total);
            for (int j = 0; j < dim_top; ++j) {
                int idx = x * dim_top + j;
                fam.projectors[x].at(idx, idx) = C64(1, 0);
            }
        }
        return fam;
    }

    // From an explicit array of projectors (caller provides)
    __host__ static AdmissibleFamily from_projectors(
        const DenseOp* projs, int n, int d) {
        AdmissibleFamily fam(d, n);
        for (int a = 0; a < n; ++a)
            fam.projectors[a] = projs[a];
        return fam;
    }
};

// ── Sector algebra verification ─────────────────────────────────────────────

namespace verify {

// P_a² = P_a and P_a P_b = 0 for a ≠ b
__host__ inline bool orthogonality(const AdmissibleFamily& fam,
                                    double tol = 1e-10) {
    for (int a = 0; a < fam.n_sectors; ++a) {
        DenseOp Pa2 = fam.projectors[a] * fam.projectors[a];
        for (int i = 0; i < fam.dim; ++i)
            for (int j = 0; j < fam.dim; ++j)
                if ((Pa2.at(i,j) - fam.projectors[a].at(i,j)).norm2()
                        > tol * tol)
                    return false;
        for (int b = a + 1; b < fam.n_sectors; ++b) {
            DenseOp PaPb = fam.projectors[a] * fam.projectors[b];
            for (int i = 0; i < fam.dim; ++i)
                for (int j = 0; j < fam.dim; ++j)
                    if (PaPb.at(i,j).norm2() > tol * tol)
                        return false;
        }
    }
    return true;
}

// Σ_a P_a = I
__host__ inline bool completeness(const AdmissibleFamily& fam,
                                   double tol = 1e-10) {
    DenseOp sum(fam.dim);
    for (int a = 0; a < fam.n_sectors; ++a)
        sum = sum + fam.projectors[a];
    DenseOp I = DenseOp::identity(fam.dim);
    for (int i = 0; i < fam.dim; ++i)
        for (int j = 0; j < fam.dim; ++j)
            if ((sum.at(i,j) - I.at(i,j)).norm2() > tol * tol)
                return false;
    return true;
}

// Full check: orthogonality + completeness
__host__ inline bool resolution_of_identity(const AdmissibleFamily& fam,
                                             double tol = 1e-10) {
    return orthogonality(fam, tol) && completeness(fam, tol);
}

} // namespace verify

// ── Exact sector decomposition ──────────────────────────────────────────────
// The doctrinal core: ρ decomposes exactly into admissible sectors,
// and Born weights are the measure on that exact decomposition.

namespace decompose {

// Born weights: p(a|ρ) = Tr(P_a ρ)
__host__ inline SectorWeights born_weights(
    const AdmissibleFamily& fam, const DenseOp& rho) {
    SectorWeights sw;
    sw.n_sectors = fam.n_sectors;
    for (int a = 0; a < fam.n_sectors; ++a) {
        DenseOp PaRho = fam.projectors[a] * rho;
        sw.weights[a] = PaRho.trace().re;
    }
    return sw;
}

// Conditional sector state: ρ_a = P_a ρ P_a / p(a)
__host__ inline DenseOp sector_state(const AdmissibleFamily& fam,
                                      const DenseOp& rho, int a) {
    DenseOp Pa = fam.projectors[a];
    DenseOp PaRhoPa = Pa * rho * Pa;
    double p = PaRhoPa.trace().re;
    if (p > 1e-15)
        return PaRhoPa * C64(1.0 / p, 0);
    return PaRhoPa;
}

// Verify Σ_a P_a ρ P_a = ρ (exact decomposition)
__host__ inline bool verify_exact_decomposition(
    const AdmissibleFamily& fam, const DenseOp& rho,
    double tol = 1e-10) {
    DenseOp reconstructed(fam.dim);
    for (int a = 0; a < fam.n_sectors; ++a) {
        DenseOp Pa = fam.projectors[a];
        reconstructed = reconstructed + Pa * rho * Pa;
    }
    for (int i = 0; i < fam.dim; ++i)
        for (int j = 0; j < fam.dim; ++j)
            if ((reconstructed.at(i,j) - rho.at(i,j)).norm2() > tol * tol)
                return false;
    return true;
}

// Verify Σ_a p(a|ρ) = 1 (normalization)
__host__ inline bool verify_normalization(const SectorWeights& sw,
                                           double tol = 1e-10) {
    double total = 0;
    for (int a = 0; a < sw.n_sectors; ++a)
        total += sw.weights[a];
    return fabs(total - 1.0) < tol;
}

// Verify p(a|ρ) ≥ 0 for all a (positivity)
__host__ inline bool verify_positivity(const SectorWeights& sw,
                                        double tol = 1e-10) {
    for (int a = 0; a < sw.n_sectors; ++a)
        if (sw.weights[a] < -tol)
            return false;
    return true;
}

} // namespace decompose

// ── Compound sector families ────────────────────────────────────────────────
// Tensor product of admissible families for joint sector labels:
//   a ∈ A₁ × A₂  →  P_{(a,b)} = P_a ⊗ Q_b

namespace compound {

// Build the compound family {P_a ⊗ Q_b} from two families
__host__ inline AdmissibleFamily tensor_product(
    const AdmissibleFamily& fam1,
    const AdmissibleFamily& fam2) {
    int d1 = fam1.dim, d2 = fam2.dim;
    int total_dim = d1 * d2;
    int total_sectors = fam1.n_sectors * fam2.n_sectors;
    AdmissibleFamily cf(total_dim, total_sectors);

    int idx = 0;
    for (int a = 0; a < fam1.n_sectors; ++a) {
        for (int b = 0; b < fam2.n_sectors; ++b) {
            cf.projectors[idx] = DenseOp(total_dim);
            for (int i1 = 0; i1 < d1; ++i1)
                for (int j1 = 0; j1 < d1; ++j1)
                    for (int i2 = 0; i2 < d2; ++i2)
                        for (int j2 = 0; j2 < d2; ++j2)
                            cf.projectors[idx].at(
                                i1 * d2 + i2,
                                j1 * d2 + j2)
                                = fam1.projectors[a].at(i1, j1)
                                  * fam2.projectors[b].at(i2, j2);
            idx++;
        }
    }
    return cf;
}

// Decode compound index → (a, b) pair
__host__ __device__ inline void decode(int compound_idx,
                                        int n2, int& a, int& b) {
    a = compound_idx / n2;
    b = compound_idx % n2;
}

} // namespace compound

// ── Spectral binning ────────────────────────────────────────────────────────
// For continuous observables: discretize spectrum into bins.
// Given eigenvalues λ_k of a diagonal observable, group them into bins
// to obtain an admissible family E(Δ_bin).

namespace spectral {

__host__ inline AdmissibleFamily from_diagonal(
    const double* eigenvalues, int dim,
    int n_bins, double lambda_min, double lambda_max) {
    double bin_width = (lambda_max - lambda_min) / n_bins;
    AdmissibleFamily fam(dim, n_bins);
    for (int b = 0; b < n_bins; ++b)
        fam.projectors[b] = DenseOp(dim);

    for (int k = 0; k < dim; ++k) {
        int bin = static_cast<int>(
            (eigenvalues[k] - lambda_min) / bin_width);
        if (bin < 0) bin = 0;
        if (bin >= n_bins) bin = n_bins - 1;
        fam.projectors[bin].at(k, k) = C64(1, 0);
    }
    return fam;
}

} // namespace spectral

} // namespace sector
} // namespace topcomp
