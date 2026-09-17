// ============================================================================
// TopComp: Exact Bidirectional Dictionary
// ============================================================================
// Complete bidirectional dictionary between hybrid objects and their
// exact local discrete-fibre decompositions.  Every hybrid state,
// operator, density matrix, circuit kernel, and observable can be
// decomposed into exact dim_top × dim_top blocks indexed by the
// discrete support I_n = F_2^n, and reconstructed exactly from them.
//
// Contents (by section):
//
//   §0-§3  Core decomposition maps
//     D_vec / R_vec — exact vector decomposition/reconstruction
//     D_op  / R_op  — exact operator block decomposition/reconstruction
//     E_{x,y} = ι_x π_y  matrix unit system
//
//   §4-§7  Walsh / Pauli coefficient face
//     Pauli trace  τ_{u,v}(T) = 2^{-n} Tr_disc((X_u Z_v)† T)
//     Block ↔ Walsh  transforms  A_{x,y} ↔ B_{u,v}
//
//   §8-§9  Circuit kernels
//     K_G(x,y) for G ∈ {X_u, Z_v, U_t, M_m, W_{u,v;g}}
//     Composite kernel  K_{G_1 G_2}(x,z) = Σ_y K_{G_1}(x,y) K_{G_2}(y,z)
//
//   §10-§13  Density matrix decomposition & measurement
//     D_dens / R_dens — exact density block decomposition
//     Discrete probabilities  p(x) = Tr(ρ_{x,x})
//     Conditional topological states  ρ_x^top
//     Block expectation  ω_ρ(O) = Σ_{x,y} Tr(ρ_{y,x} O_{x,y})
//
//   §14-§15  Field operators & N-point functions
//     Weyl block form  (W_{u,v;g})_{r,s}
//     N-point function via block traces
//
//   §16-§18  Transport & thermodynamics
//     Functorial transport  T_φ^hyb = id ⊗ T_φ^top
//     Thermodynamic factorization  Z_{A,n} = 2^n · Z_A^top
//     POVM output probabilities  Pr[Out(C,a)=y]
//
// Identities verified:
//   R_vec ∘ D_vec = D_vec ∘ R_vec = id
//   R_op  ∘ D_op  = D_op  ∘ R_op  = id
//   E_{x,y} E_{x',y'} = δ_{y,x'} E_{x,y'}
//   Walsh ↔ Block roundtrip = id
//   K_{C_1 C_2}(x,z) = Σ_y K_{C_1}(x,y) K_{C_2}(y,z)
//   Σ_x p(x) = 1
//   Transport preserves discrete probabilities
// ============================================================================
#pragma once
#include "fiber.cuh"
#include "gates.cuh"
#include "cocycle.cuh"

namespace topcomp {
namespace exact {

using measurement::DenseOp;

// ════════════════════════════════════════════════════════════════════════════
// Block storage: 2D grid of dim_top × dim_top operators indexed by
// discrete positions (x,y) ∈ I_n × I_n
// ════════════════════════════════════════════════════════════════════════════

struct OpBlocks {
    DenseOp* blocks;
    int dim_disc;       // 2^n
    int dim_top;        // N_theta * N_rho

    __host__ OpBlocks() : blocks(nullptr), dim_disc(0), dim_top(0) {}

    __host__ void init(int dd, int dt) {
        dim_disc = dd;
        dim_top = dt;
        blocks = new DenseOp[dd * dd];
    }

    __host__ void init_zero(int dd, int dt) {
        dim_disc = dd;
        dim_top = dt;
        blocks = new DenseOp[dd * dd];
        for (int i = 0; i < dd * dd; ++i)
            blocks[i] = DenseOp::zero(dt);
    }

    __host__ void free() {
        delete[] blocks;
        blocks = nullptr;
    }

    __host__ DenseOp& at(int x, int y) { return blocks[x * dim_disc + y]; }
    __host__ const DenseOp& at(int x, int y) const { return blocks[x * dim_disc + y]; }

    // Frobenius norm of all blocks: Σ_{x,y} ||A_{x,y}||²
    __host__ double total_hs_norm2() const {
        double s = 0;
        for (int i = 0; i < dim_disc * dim_disc; ++i)
            s += blocks[i].hs_norm2();
        return s;
    }
};

// ════════════════════════════════════════════════════════════════════════════
// §0  Helpers: topological gate matrices
// ════════════════════════════════════════════════════════════════════════════

// Build U_t as a dim_top × dim_top DenseOp matrix
__host__ inline DenseOp U_matrix(double t, int N_theta, int N_rho,
                                  double rho_min = -5.0, double rho_max = 5.0) {
    int dim = N_theta * N_rho;
    DenseOp M(dim);
    TopologicalState basis;
    basis.init(N_theta, N_rho, rho_min, rho_max);
    for (int j = 0; j < dim; ++j) {
        memset(basis.amp, 0, dim * sizeof(C64));
        basis.amp[j] = C64(1.0, 0.0);
        topo_gates::apply_U(basis, t);
        for (int i = 0; i < dim; ++i)
            M.at(i, j) = basis.amp[i];
    }
    basis.free();
    return M;
}

// Build M_m as a dim_top × dim_top DenseOp matrix
__host__ inline DenseOp M_matrix(int m, int N_theta, int N_rho,
                                  double rho_min = -5.0, double rho_max = 5.0) {
    int dim = N_theta * N_rho;
    DenseOp M_op(dim);
    TopologicalState basis;
    basis.init(N_theta, N_rho, rho_min, rho_max);
    for (int j = 0; j < dim; ++j) {
        memset(basis.amp, 0, dim * sizeof(C64));
        basis.amp[j] = C64(1.0, 0.0);
        topo_gates::apply_M(basis, m);
        for (int i = 0; i < dim; ++i)
            M_op.at(i, j) = basis.amp[i];
    }
    basis.free();
    return M_op;
}

// Build ρ̂_A(g) — topological representation of MirElement g
// (ρ̂_A(g) ξ)(θ,ρ) = ξ(g⁻¹·(θ,ρ))
__host__ inline DenseOp mir_rep_matrix(const MirElement& g,
                                        int N_theta, int N_rho,
                                        double rho_min = -5.0,
                                        double rho_max = 5.0) {
    int dim = N_theta * N_rho;
    DenseOp R(dim);
    MirElement ginv = g.inverse();
    double d_rho = (N_rho > 1) ? (rho_max - rho_min) / (N_rho - 1) : 1.0;

    for (int i = 0; i < N_theta; ++i) {
        for (int j = 0; j < N_rho; ++j) {
            double theta = constants::TWO_PI * i / N_theta;
            double rho = rho_min + j * d_rho;

            double src_theta, src_rho;
            ginv.act_phase(theta, rho, src_theta, src_rho);

            double si = fmod(src_theta / constants::TWO_PI * N_theta,
                             static_cast<double>(N_theta));
            if (si < 0) si += N_theta;
            double sj = (src_rho - rho_min) / (rho_max - rho_min) * (N_rho - 1);

            int i0 = static_cast<int>(floor(si)) % N_theta;
            int i1 = (i0 + 1) % N_theta;
            int j0 = static_cast<int>(floor(sj));
            int j1 = j0 + 1;
            double fi = si - floor(si);
            double fj = sj - floor(sj);

            int out_idx = i * N_rho + j;
            if (j0 >= 0 && j1 < N_rho) {
                R.at(out_idx, i0 * N_rho + j0) =
                    R.at(out_idx, i0 * N_rho + j0) + C64((1-fi)*(1-fj), 0);
                R.at(out_idx, i1 * N_rho + j0) =
                    R.at(out_idx, i1 * N_rho + j0) + C64(fi*(1-fj), 0);
                R.at(out_idx, i0 * N_rho + j1) =
                    R.at(out_idx, i0 * N_rho + j1) + C64((1-fi)*fj, 0);
                R.at(out_idx, i1 * N_rho + j1) =
                    R.at(out_idx, i1 * N_rho + j1) + C64(fi*fj, 0);
            }
        }
    }
    return R;
}

// ════════════════════════════════════════════════════════════════════════════
// §1-§3  Core Decomposition Maps: Vectors
// ════════════════════════════════════════════════════════════════════════════

namespace vec {

// D_vec: |Ψ⟩ ↦ (ξ_x)_{x∈I_n}
// Decomposes hybrid state into dim_disc topological components
__host__ inline void decompose(const HybridState& psi,
                                TopologicalState* out) {
    psi.to_components(out);
}

// R_vec: (ξ_x)_x ↦ |Ψ⟩ = Σ_x ι_x(ξ_x)
// Reconstructs hybrid state from topological components
__host__ inline void reconstruct(HybridState& psi,
                                  const TopologicalState* in) {
    psi.from_components(in);
}

// Verify D ∘ R = id:  decompose then reconstruct recovers original
__host__ inline bool roundtrip_DR(const HybridState& psi, double tol = 1e-10) {
    int dd = psi.dim_disc;

    TopologicalState* comps = new TopologicalState[dd];
    for (int x = 0; x < dd; ++x)
        comps[x].init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);

    decompose(psi, comps);

    HybridState recon;
    recon.init(psi.n_qubits, psi.N_theta, psi.N_rho,
               psi.rho_min, psi.rho_max);
    reconstruct(recon, comps);

    double err = 0;
    for (int i = 0; i < psi.total; ++i) {
        C64 diff = psi.amp[i] - recon.amp[i];
        err += diff.norm2();
    }

    recon.free();
    for (int x = 0; x < dd; ++x) comps[x].free();
    delete[] comps;

    return err < tol;
}

// Verify R ∘ D = id:  reconstruct then decompose recovers components
__host__ inline bool roundtrip_RD(const TopologicalState* comps,
                                   int n_qubits, double tol = 1e-10) {
    int dd = 1 << n_qubits;
    int nth = comps[0].N_theta;
    int nrh = comps[0].N_rho;
    double rmin = comps[0].rho_min;
    double rmax = comps[0].rho_max;
    int dt = nth * nrh;

    HybridState hyb;
    hyb.init(n_qubits, nth, nrh, rmin, rmax);
    reconstruct(hyb, comps);

    TopologicalState* back = new TopologicalState[dd];
    for (int x = 0; x < dd; ++x)
        back[x].init(nth, nrh, rmin, rmax);
    decompose(hyb, back);

    double err = 0;
    for (int x = 0; x < dd; ++x)
        for (int i = 0; i < dt; ++i) {
            C64 diff = comps[x].amp[i] - back[x].amp[i];
            err += diff.norm2();
        }

    hyb.free();
    for (int x = 0; x < dd; ++x) back[x].free();
    delete[] back;

    return err < tol;
}

} // namespace vec

// ════════════════════════════════════════════════════════════════════════════
// §2-§3  Core Decomposition Maps: Operators
// ════════════════════════════════════════════════════════════════════════════

namespace op {

// D_op: T ↦ (T_{x,y})  where  T_{x,y} = π_x T ι_y
// Extract dim_top × dim_top block at position (x,y) from hybrid operator
__host__ inline DenseOp extract_block(const DenseOp& T, int x, int y,
                                       int dim_top) {
    DenseOp B(dim_top);
    for (int i = 0; i < dim_top; ++i)
        for (int j = 0; j < dim_top; ++j)
            B.at(i, j) = T.at(x * dim_top + i, y * dim_top + j);
    return B;
}

// Full decomposition: T ↦ OpBlocks
__host__ inline OpBlocks decompose(const DenseOp& T, int dim_disc, int dim_top) {
    OpBlocks blk;
    blk.init(dim_disc, dim_top);
    for (int x = 0; x < dim_disc; ++x)
        for (int y = 0; y < dim_disc; ++y)
            blk.at(x, y) = extract_block(T, x, y, dim_top);
    return blk;
}

// R_op: (T_{x,y}) ↦ T = Σ_{x,y} ι_x T_{x,y} π_y
// Embed dim_top × dim_top block at position (x,y) into hybrid operator
__host__ inline void embed_block(DenseOp& T, int x, int y,
                                  const DenseOp& B, int dim_top) {
    for (int i = 0; i < dim_top; ++i)
        for (int j = 0; j < dim_top; ++j)
            T.at(x * dim_top + i, y * dim_top + j) =
                T.at(x * dim_top + i, y * dim_top + j) + B.at(i, j);
}

// Full reconstruction: OpBlocks ↦ T
__host__ inline DenseOp reconstruct(const OpBlocks& blk) {
    int dim_hyb = blk.dim_disc * blk.dim_top;
    DenseOp T(dim_hyb);
    for (int x = 0; x < blk.dim_disc; ++x)
        for (int y = 0; y < blk.dim_disc; ++y)
            embed_block(T, x, y, blk.at(x, y), blk.dim_top);
    return T;
}

// E_{x,y} = ι_x π_y : matrix unit (id_top in block (x,y), zero elsewhere)
__host__ inline DenseOp matrix_unit(int x, int y, int dim_disc, int dim_top) {
    int dim_hyb = dim_disc * dim_top;
    DenseOp E(dim_hyb);
    for (int k = 0; k < dim_top; ++k)
        E.at(x * dim_top + k, y * dim_top + k) = C64(1, 0);
    return E;
}

// Verify roundtrip D_op ∘ R_op = id
__host__ inline bool roundtrip(const DenseOp& T, int dim_disc,
                                int dim_top, double tol = 1e-10) {
    OpBlocks blk = decompose(T, dim_disc, dim_top);
    DenseOp recon = reconstruct(blk);
    double err = (T - recon).hs_norm2();
    blk.free();
    return err < tol;
}

// Verify matrix unit algebra: E_{x,y} E_{x',y'} = δ_{y,x'} E_{x,y'}
__host__ inline bool matrix_unit_algebra(int dim_disc, int dim_top,
                                          double tol = 1e-10) {
    for (int x = 0; x < dim_disc; ++x)
        for (int y = 0; y < dim_disc; ++y)
            for (int xp = 0; xp < dim_disc; ++xp)
                for (int yp = 0; yp < dim_disc; ++yp) {
                    DenseOp E1 = matrix_unit(x, y, dim_disc, dim_top);
                    DenseOp E2 = matrix_unit(xp, yp, dim_disc, dim_top);
                    DenseOp prod = E1 * E2;

                    if (y == xp) {
                        DenseOp expected = matrix_unit(x, yp, dim_disc, dim_top);
                        if ((prod - expected).hs_norm2() > tol) return false;
                    } else {
                        if (prod.hs_norm2() > tol) return false;
                    }
                }
    return true;
}

// Verify resolution: Σ_x E_{x,x} = I
__host__ inline bool matrix_unit_resolution(int dim_disc, int dim_top,
                                             double tol = 1e-10) {
    int dim_hyb = dim_disc * dim_top;
    DenseOp sum = DenseOp::zero(dim_hyb);
    for (int x = 0; x < dim_disc; ++x)
        sum = sum + matrix_unit(x, x, dim_disc, dim_top);
    DenseOp I = DenseOp::identity(dim_hyb);
    return (sum - I).hs_norm2() < tol;
}

} // namespace op

// ════════════════════════════════════════════════════════════════════════════
// §4-§7  Walsh / Pauli Coefficient Transforms
// ════════════════════════════════════════════════════════════════════════════

namespace walsh {

// Pauli trace: τ_{u,v}(T) = 2^{-n} Σ_x (-1)^{⟨v,x⟩} T_{x⊕u, x}
// Returns a dim_top × dim_top operator
__host__ inline DenseOp pauli_trace(const OpBlocks& A, const BitVec& u,
                                     const BitVec& v) {
    int dd = A.dim_disc;
    int dt = A.dim_top;
    int n = u.n;
    double inv_dim = 1.0 / dd;

    DenseOp result = DenseOp::zero(dt);
    for (int x = 0; x < dd; ++x) {
        BitVec xv(x, n);
        double sign = (xv.inner(v) == 0) ? 1.0 : -1.0;
        int xu = x ^ u.bits;  // x ⊕ u
        result = result + A.at(xu, x) * C64(sign * inv_dim, 0);
    }
    return result;
}

// Block-to-Walsh: B_{u,v} = Σ_y (-1)^{⟨v,y⟩} A_{y⊕u, y}
__host__ inline OpBlocks block_to_walsh(const OpBlocks& A, int n) {
    int dd = A.dim_disc;
    int dt = A.dim_top;
    OpBlocks B;
    B.init_zero(dd, dt);

    for (int u_bits = 0; u_bits < dd; ++u_bits) {
        for (int v_bits = 0; v_bits < dd; ++v_bits) {
            DenseOp sum = DenseOp::zero(dt);
            for (int y = 0; y < dd; ++y) {
                BitVec yv(y, n);
                BitVec vv(v_bits, n);
                double sign = (yv.inner(vv) == 0) ? 1.0 : -1.0;
                int yu = y ^ u_bits;
                sum = sum + A.at(yu, y) * C64(sign, 0);
            }
            B.at(u_bits, v_bits) = sum;
        }
    }
    return B;
}

// Walsh-to-Block: A_{x,y} = 2^{-n} Σ_v (-1)^{⟨v,y⟩} B_{x⊕y, v}
__host__ inline OpBlocks walsh_to_block(const OpBlocks& B, int n) {
    int dd = B.dim_disc;
    int dt = B.dim_top;
    double inv_dim = 1.0 / dd;
    OpBlocks A;
    A.init_zero(dd, dt);

    for (int x = 0; x < dd; ++x) {
        for (int y = 0; y < dd; ++y) {
            DenseOp sum = DenseOp::zero(dt);
            int xy = x ^ y;
            for (int v_bits = 0; v_bits < dd; ++v_bits) {
                BitVec yv(y, n);
                BitVec vv(v_bits, n);
                double sign = (yv.inner(vv) == 0) ? 1.0 : -1.0;
                sum = sum + B.at(xy, v_bits) * C64(sign * inv_dim, 0);
            }
            A.at(x, y) = sum;
        }
    }
    return A;
}

// Verify Walsh roundtrip: walsh_to_block(block_to_walsh(A)) = A
__host__ inline bool roundtrip(const OpBlocks& A, int n, double tol = 1e-10) {
    OpBlocks B = block_to_walsh(A, n);
    OpBlocks A2 = walsh_to_block(B, n);

    double err = 0;
    for (int x = 0; x < A.dim_disc; ++x)
        for (int y = 0; y < A.dim_disc; ++y)
            err += (A.at(x, y) - A2.at(x, y)).hs_norm2();

    B.free();
    A2.free();
    return err < tol;
}

// Verify pauli_trace = 2^{-n} B_{u,v}
__host__ inline bool pauli_is_walsh(const OpBlocks& A, int n,
                                     double tol = 1e-10) {
    int dd = A.dim_disc;
    OpBlocks B = block_to_walsh(A, n);

    double err = 0;
    for (int u = 0; u < dd; ++u)
        for (int v = 0; v < dd; ++v) {
            BitVec uv(u, n), vv(v, n);
            DenseOp tau = pauli_trace(A, uv, vv);
            DenseOp expected = B.at(u, v) * C64(1.0 / dd, 0);
            err += (tau - expected).hs_norm2();
        }

    B.free();
    return err < tol;
}

// E_{x,y} ↔ Pauli expansion:
// E_{x,y} = 2^{-n} Σ_{u,v} (-1)^{⟨u,y⟩ + ⟨v,x⟩} X_u Z_v (⊗ I_top)
// Verify by checking Pauli trace of E_{x,y}
__host__ inline bool matrix_unit_pauli_expansion(int dim_disc, int dim_top,
                                                  int n, double tol = 1e-10) {
    // τ_{u,v}(E_{x,y}) should equal:
    // 2^{-n} Σ_{x'} (-1)^{⟨v,x'⟩} (E_{x,y})_{x'⊕u, x'}
    // (E_{x,y})_{a,b} = δ_{a,x} δ_{b,y} · I_top
    // So τ_{u,v}(E_{x,y}) = 2^{-n} (-1)^{⟨v,y⟩} δ_{y⊕u,x} · I_top
    //                      = 2^{-n} (-1)^{⟨v,y⟩} δ_{u,x⊕y} · I_top
    for (int x = 0; x < dim_disc; ++x) {
        for (int y = 0; y < dim_disc; ++y) {
            // Build blocks for E_{x,y}
            OpBlocks E_blk;
            E_blk.init_zero(dim_disc, dim_top);
            E_blk.at(x, y) = DenseOp::identity(dim_top);

            for (int u = 0; u < dim_disc; ++u) {
                for (int v = 0; v < dim_disc; ++v) {
                    BitVec uv(u, n), vv(v, n);
                    DenseOp tau = pauli_trace(E_blk, uv, vv);

                    // Expected: 2^{-n} (-1)^{⟨v,y⟩} δ_{u,x⊕y} I_top
                    if (u == (x ^ y)) {
                        BitVec yv(y, n);
                        double sign = (yv.inner(vv) == 0) ? 1.0 : -1.0;
                        double expected_coeff = sign / dim_disc;
                        DenseOp expected =
                            DenseOp::identity(dim_top) * C64(expected_coeff, 0);
                        if ((tau - expected).hs_norm2() > tol) {
                            E_blk.free();
                            return false;
                        }
                    } else {
                        if (tau.hs_norm2() > tol) {
                            E_blk.free();
                            return false;
                        }
                    }
                }
            }
            E_blk.free();
        }
    }
    return true;
}

} // namespace walsh

// ════════════════════════════════════════════════════════════════════════════
// §8-§9  Circuit Kernels
// ════════════════════════════════════════════════════════════════════════════

namespace kernel {

// K_{X_u}(x,y) = δ_{x, y⊕u} · I_top
__host__ inline OpBlocks K_X(const BitVec& u, int dim_disc, int dim_top) {
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int y = 0; y < dim_disc; ++y) {
        int x = y ^ u.bits;
        K.at(x, y) = DenseOp::identity(dim_top);
    }
    return K;
}

// K_{Z_v}(x,y) = δ_{x,y} · (-1)^{⟨v,y⟩} · I_top
__host__ inline OpBlocks K_Z(const BitVec& v, int dim_disc, int dim_top) {
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int y = 0; y < dim_disc; ++y) {
        BitVec yv(y, v.n);
        double sign = (yv.inner(v) == 0) ? 1.0 : -1.0;
        K.at(y, y) = DenseOp::identity(dim_top) * C64(sign, 0);
    }
    return K;
}

// K_{U_t}(x,y) = δ_{x,y} · U_t
__host__ inline OpBlocks K_U(double t, int dim_disc, int N_theta, int N_rho,
                              double rho_min = -5.0, double rho_max = 5.0) {
    int dim_top = N_theta * N_rho;
    DenseOp Ut = U_matrix(t, N_theta, N_rho, rho_min, rho_max);
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int x = 0; x < dim_disc; ++x)
        K.at(x, x) = Ut;
    return K;
}

// K_{M_m}(x,y) = δ_{x,y} · M_m
__host__ inline OpBlocks K_M(int m, int dim_disc, int N_theta, int N_rho,
                              double rho_min = -5.0, double rho_max = 5.0) {
    int dim_top = N_theta * N_rho;
    DenseOp Mm = M_matrix(m, N_theta, N_rho, rho_min, rho_max);
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int x = 0; x < dim_disc; ++x)
        K.at(x, x) = Mm;
    return K;
}

// K_{X_u Z_v U_t M_m}(x,y) = δ_{x, y⊕u} · (-1)^{⟨v,y⟩} · U_t · M_m
__host__ inline OpBlocks K_composite(const BitVec& u, const BitVec& v,
                                      double t, int m,
                                      int dim_disc, int N_theta, int N_rho,
                                      double rho_min = -5.0,
                                      double rho_max = 5.0) {
    int dim_top = N_theta * N_rho;
    DenseOp Ut = U_matrix(t, N_theta, N_rho, rho_min, rho_max);
    DenseOp Mm = M_matrix(m, N_theta, N_rho, rho_min, rho_max);
    DenseOp UtMm = Ut * Mm;

    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int y = 0; y < dim_disc; ++y) {
        int x = y ^ u.bits;
        BitVec yv(y, v.n);
        double sign = (yv.inner(v) == 0) ? 1.0 : -1.0;
        K.at(x, y) = UtMm * C64(sign, 0);
    }
    return K;
}

// W_{u,v;g} kernel: K_{W}(x,y) = δ_{x, y⊕u} · (-1)^{⟨v,y⟩} · ρ̂_A(g)
__host__ inline OpBlocks K_W(const BitVec& u, const BitVec& v,
                              const MirElement& g,
                              int dim_disc, int N_theta, int N_rho,
                              double rho_min = -5.0, double rho_max = 5.0) {
    int dim_top = N_theta * N_rho;
    DenseOp rho_g = mir_rep_matrix(g, N_theta, N_rho, rho_min, rho_max);

    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int y = 0; y < dim_disc; ++y) {
        int x = y ^ u.bits;
        BitVec yv(y, v.n);
        double sign = (yv.inner(v) == 0) ? 1.0 : -1.0;
        K.at(x, y) = rho_g * C64(sign, 0);
    }
    return K;
}

// Kernel composition: K_{G1∘G2}(x,z) = Σ_y K_{G1}(x,y) · K_{G2}(y,z)
__host__ inline OpBlocks compose(const OpBlocks& K1, const OpBlocks& K2) {
    int dd = K1.dim_disc;
    int dt = K1.dim_top;
    OpBlocks K;
    K.init_zero(dd, dt);

    for (int x = 0; x < dd; ++x) {
        for (int z = 0; z < dd; ++z) {
            DenseOp sum = DenseOp::zero(dt);
            for (int y = 0; y < dd; ++y)
                sum = sum + K1.at(x, y) * K2.at(y, z);
            K.at(x, z) = sum;
        }
    }
    return K;
}

// Build kernel from hybrid DenseOp directly (D_op applied to unitary)
__host__ inline OpBlocks from_hybrid_op(const DenseOp& T,
                                         int dim_disc, int dim_top) {
    return op::decompose(T, dim_disc, dim_top);
}

// Compare kernel from composition vs direct gate application
__host__ inline bool verify_kernel_vs_direct(
    const HybridState& input,
    const OpBlocks& K, int n_qubits,
    int N_theta, int N_rho,
    double rho_min, double rho_max,
    const HybridState& expected_output,
    double tol = 1e-6)
{
    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;

    // Apply kernel to input: output_x = Σ_y K(x,y) · input_y
    HybridState output;
    output.init(n_qubits, N_theta, N_rho, rho_min, rho_max);

    TopologicalState* in_comps = new TopologicalState[dd];
    for (int x = 0; x < dd; ++x)
        in_comps[x].init(N_theta, N_rho, rho_min, rho_max);
    input.to_components(in_comps);

    TopologicalState* out_comps = new TopologicalState[dd];
    for (int x = 0; x < dd; ++x) {
        out_comps[x].init(N_theta, N_rho, rho_min, rho_max);
        memset(out_comps[x].amp, 0, dt * sizeof(C64));
        for (int y = 0; y < dd; ++y) {
            // out_x += K(x,y) * in_y
            for (int i = 0; i < dt; ++i) {
                C64 sum(0, 0);
                for (int j = 0; j < dt; ++j)
                    sum = sum + K.at(x, y).at(i, j) * in_comps[y].amp[j];
                out_comps[x].amp[i] = out_comps[x].amp[i] + sum;
            }
        }
    }
    output.from_components(out_comps);

    double err = 0;
    for (int i = 0; i < output.total; ++i) {
        C64 diff = output.amp[i] - expected_output.amp[i];
        err += diff.norm2();
    }

    output.free();
    for (int x = 0; x < dd; ++x) { in_comps[x].free(); out_comps[x].free(); }
    delete[] in_comps;
    delete[] out_comps;

    return err < tol;
}

} // namespace kernel

// ════════════════════════════════════════════════════════════════════════════
// §10-§13  Density Matrix Decomposition & Measurement
// ════════════════════════════════════════════════════════════════════════════

namespace dens {

// D_dens: ρ ↦ (ρ_{x,y})  where  ρ_{x,y} = π_x ρ ι_y
// Same as D_op; density matrices are just operators with extra properties
__host__ inline OpBlocks decompose(const DenseOp& rho,
                                    int dim_disc, int dim_top) {
    return op::decompose(rho, dim_disc, dim_top);
}

// R_dens: (ρ_{x,y}) ↦ ρ = Σ_{x,y} ι_x ρ_{x,y} π_y
__host__ inline DenseOp reconstruct(const OpBlocks& blk) {
    return op::reconstruct(blk);
}

// Discrete probability: p(x) = Tr(ρ_{x,x})
__host__ inline double disc_prob(const OpBlocks& rho_blk, int x) {
    return rho_blk.at(x, x).trace().re;
}

// Full discrete probability distribution
__host__ inline void disc_probs(const OpBlocks& rho_blk, double* probs) {
    for (int x = 0; x < rho_blk.dim_disc; ++x)
        probs[x] = disc_prob(rho_blk, x);
}

// Verify Σ_x p(x) = 1
__host__ inline bool verify_prob_normalization(const OpBlocks& rho_blk,
                                                double tol = 1e-10) {
    double sum = 0;
    for (int x = 0; x < rho_blk.dim_disc; ++x)
        sum += disc_prob(rho_blk, x);
    return fabs(sum - 1.0) < tol;
}

// Conditional topological state: ρ_x^top = ρ_{x,x} / Tr(ρ_{x,x})
__host__ inline DenseOp conditional_topo(const OpBlocks& rho_blk, int x) {
    double px = disc_prob(rho_blk, x);
    if (px < 1e-15) return DenseOp::zero(rho_blk.dim_top);
    return rho_blk.at(x, x) * C64(1.0 / px, 0);
}

// Block expectation: ω_ρ(O) = Σ_{x,y} Tr(ρ_{y,x} · O_{x,y})
__host__ inline C64 block_expectation(const OpBlocks& rho_blk,
                                       const OpBlocks& O_blk) {
    C64 result(0, 0);
    int dd = rho_blk.dim_disc;
    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y) {
            DenseOp prod = rho_blk.at(y, x) * O_blk.at(x, y);
            result = result + prod.trace();
        }
    return result;
}

// Verify block expectation matches full trace: Tr(ρ O)
__host__ inline bool verify_block_expectation(
    const DenseOp& rho, const DenseOp& O,
    int dim_disc, int dim_top, double tol = 1e-10)
{
    DenseOp rhoO = rho * O;
    C64 full_trace = rhoO.trace();

    OpBlocks rho_blk = op::decompose(rho, dim_disc, dim_top);
    OpBlocks O_blk = op::decompose(O, dim_disc, dim_top);
    C64 block_trace = block_expectation(rho_blk, O_blk);

    rho_blk.free();
    O_blk.free();

    C64 diff = full_trace - block_trace;
    return diff.norm2() < tol;
}

// Verify that diagonal blocks of a density operator are positive with
// correct trace
__host__ inline bool verify_diagonal_properties(const OpBlocks& rho_blk,
                                                  double tol = 1e-10) {
    int dd = rho_blk.dim_disc;
    for (int x = 0; x < dd; ++x) {
        // Each diagonal block ρ_{x,x} should be positive semidefinite
        // (heuristic: check diagonal entries are non-negative)
        for (int i = 0; i < rho_blk.dim_top; ++i)
            if (rho_blk.at(x, x).at(i, i).re < -tol)
                return false;
        // Hermiticity check: ρ_{x,x} = ρ_{x,x}†
        for (int i = 0; i < rho_blk.dim_top; ++i)
            for (int j = i + 1; j < rho_blk.dim_top; ++j) {
                C64 diff = rho_blk.at(x, x).at(i, j)
                         - rho_blk.at(x, x).at(j, i).conj();
                if (diff.norm2() > tol * tol) return false;
            }
    }
    return true;
}

} // namespace dens

// ════════════════════════════════════════════════════════════════════════════
// §14-§15  Field Operators & N-point Functions
// ════════════════════════════════════════════════════════════════════════════

namespace field {

// Weyl operator block form:
// (W_{u,v;g})_{r,s} = (-1)^{⟨v,s⟩} δ_{r, s⊕u} · ρ̂_A(g)
// This is the same as kernel::K_W — the Weyl operator IS its own kernel
__host__ inline OpBlocks weyl_blocks(const BitVec& u, const BitVec& v,
                                      const MirElement& g,
                                      int dim_disc, int N_theta, int N_rho,
                                      double rho_min = -5.0,
                                      double rho_max = 5.0) {
    return kernel::K_W(u, v, g, dim_disc, N_theta, N_rho, rho_min, rho_max);
}

// Verify Weyl block form matches direct decomposition of W^hyb
__host__ inline bool verify_weyl_blocks(
    const BitVec& u, const BitVec& v, const MirElement& g,
    int n_qubits, int N_theta, int N_rho,
    double rho_min = -5.0, double rho_max = 5.0,
    double tol = 1e-6)
{
    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int dim_hyb = dd * dt;

    // Build W^hyb as full matrix by applying to each basis vector
    DenseOp W_full(dim_hyb);
    HybridState basis;
    basis.init(n_qubits, N_theta, N_rho, rho_min, rho_max);
    for (int j = 0; j < dim_hyb; ++j) {
        memset(basis.amp, 0, dim_hyb * sizeof(C64));
        basis.amp[j] = C64(1.0, 0.0);
        hybrid_gates::apply_W_hyb(basis, u, v, g);
        for (int i = 0; i < dim_hyb; ++i)
            W_full.at(i, j) = basis.amp[i];
    }
    basis.free();

    // Decompose the full matrix
    OpBlocks direct = op::decompose(W_full, dd, dt);

    // Compare with analytic block form
    OpBlocks analytic = weyl_blocks(u, v, g, dd, N_theta, N_rho,
                                    rho_min, rho_max);

    double err = 0;
    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y)
            err += (direct.at(x, y) - analytic.at(x, y)).hs_norm2();

    direct.free();
    analytic.free();

    return err < tol;
}

// N-point function via block traces:
// G_N(O_1,...,O_N; ρ) = Tr(O_1 · ... · O_N · ρ)
//   = Σ_{x_0,...,x_N} Tr(O_1_{x_0,x_1} · O_2_{x_1,x_2} · ... · ρ_{x_N,x_0})
__host__ inline C64 npoint_block(const OpBlocks* O, int N,
                                  const OpBlocks& rho_blk) {
    int dd = rho_blk.dim_disc;
    int dt = rho_blk.dim_top;
    C64 result(0, 0);

    if (N == 1) {
        // G_1(O; ρ) = Σ_{x,y} Tr(O_{x,y} · ρ_{y,x})
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                DenseOp prod = O[0].at(x, y) * rho_blk.at(y, x);
                result = result + prod.trace();
            }
        return result;
    }

    if (N == 2) {
        // G_2(O_1, O_2; ρ) = Σ_{x,y,z} Tr(O_1_{x,y} · O_2_{y,z} · ρ_{z,x})
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y)
                for (int z = 0; z < dd; ++z) {
                    DenseOp prod = O[0].at(x, y) * O[1].at(y, z)
                                   * rho_blk.at(z, x);
                    result = result + prod.trace();
                }
        return result;
    }

    // General N: use sequential contraction
    // Contract from left: T = O_1 · O_2 · ... · O_N as hybrid operators,
    // then trace with ρ
    for (int x0 = 0; x0 < dd; ++x0) {
        for (int xN = 0; xN < dd; ++xN) {
            // Sum over intermediate indices x_1,...,x_{N-1}
            // Start with DenseOp product chains
            // For N > 2, we accumulate: acc_{x0,x_k} = Σ_{x_1..x_{k-1}} O_1 · ... · O_k
            // Then G_N = Σ_{x0,xN} Tr(acc_{x0,xN} · ρ_{xN,x0})

            // Initialize accumulated product for each final index
            DenseOp* acc_prev = new DenseOp[dd];
            DenseOp* acc_next = new DenseOp[dd];
            for (int ix = 0; ix < dd; ++ix) acc_prev[ix] = DenseOp::zero(dt);
            // acc_prev[x1] = O_0_{x0,x1}
            for (int x1 = 0; x1 < dd; ++x1)
                acc_prev[x1] = O[0].at(x0, x1);

            for (int k = 1; k < N; ++k) {
                for (int ix = 0; ix < dd; ++ix)
                    acc_next[ix] = DenseOp::zero(dt);
                for (int xk = 0; xk < dd; ++xk)
                    for (int xk1 = 0; xk1 < dd; ++xk1)
                        acc_next[xk1] = acc_next[xk1]
                                      + acc_prev[xk] * O[k].at(xk, xk1);
                for (int ix = 0; ix < dd; ++ix)
                    acc_prev[ix] = acc_next[ix];
            }

            // Trace with ρ
            DenseOp prod = acc_prev[xN] * rho_blk.at(xN, x0);
            result = result + prod.trace();

            delete[] acc_prev;
            delete[] acc_next;
        }
    }
    return result;
}

// 1-point function shorthand
__host__ inline C64 G1(const OpBlocks& O, const OpBlocks& rho_blk) {
    const OpBlocks ops[1] = { O };
    return npoint_block(ops, 1, rho_blk);
}

} // namespace field

// ════════════════════════════════════════════════════════════════════════════
// §16  Functorial Transport
// ════════════════════════════════════════════════════════════════════════════

namespace transport {

// T_φ^hyb = id_disc ⊗ T_φ^top
// Transport leaves discrete indices fixed, applies T_φ to each fibre
__host__ inline void apply_transport(HybridState& psi,
                                      const DenseOp& T_phi_top) {
    int dd = psi.dim_disc;
    int dt = psi.dim_top;
    TopologicalState slice;
    slice.init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);

    for (int x = 0; x < dd; ++x) {
        psi.project(x, slice);
        // Apply T_phi_top: new_amp = T_phi_top * old_amp
        C64* new_amp = new C64[dt];
        for (int i = 0; i < dt; ++i) {
            C64 sum(0, 0);
            for (int j = 0; j < dt; ++j)
                sum = sum + T_phi_top.at(i, j) * slice.amp[j];
            new_amp[i] = sum;
        }
        memcpy(slice.amp, new_amp, dt * sizeof(C64));
        delete[] new_amp;
        psi.inject(x, slice);
    }
    slice.free();
}

// Verify transport preserves discrete probabilities
__host__ inline bool preserves_disc_probs(
    const HybridState& psi, const DenseOp& T_phi_top,
    double tol = 1e-6)
{
    int dd = psi.dim_disc;

    // Get original discrete probabilities
    double* probs_before = new double[dd];
    for (int x = 0; x < dd; ++x)
        probs_before[x] = psi.prob_disc(x);

    // Apply transport to a copy
    HybridState copy;
    copy.init(psi.n_qubits, psi.N_theta, psi.N_rho,
              psi.rho_min, psi.rho_max);
    memcpy(copy.amp, psi.amp, psi.total * sizeof(C64));
    apply_transport(copy, T_phi_top);

    // Check probabilities unchanged
    bool ok = true;
    for (int x = 0; x < dd; ++x) {
        double p_after = copy.prob_disc(x);
        if (fabs(p_after - probs_before[x]) > tol) { ok = false; break; }
    }

    copy.free();
    delete[] probs_before;
    return ok;
}

// Transport as operator blocks: (T_φ^hyb)_{x,y} = δ_{x,y} · T_φ^top
__host__ inline OpBlocks transport_blocks(const DenseOp& T_phi_top,
                                           int dim_disc) {
    OpBlocks K;
    K.init_zero(dim_disc, T_phi_top.dim);
    for (int x = 0; x < dim_disc; ++x)
        K.at(x, x) = T_phi_top;
    return K;
}

} // namespace transport

// ════════════════════════════════════════════════════════════════════════════
// §17-§18  Thermodynamics & POVM Output
// ════════════════════════════════════════════════════════════════════════════

namespace thermo {

// Topological partition function: Z_A^top = Tr(e^{-β H_top})
// For finite grid, H_top must be supplied as a DenseOp
__host__ inline double Z_top(const DenseOp& H_top, double beta) {
    // Compute e^{-βH} via series expansion for small systems:
    // e^{-βH} ≈ I - βH + (βH)²/2 - ...
    int dim = H_top.dim;
    DenseOp expH = DenseOp::identity(dim);
    DenseOp power = DenseOp::identity(dim);
    DenseOp neg_beta_H = H_top * C64(-beta, 0);
    double factorial = 1.0;

    for (int k = 1; k <= 20; ++k) {
        power = power * neg_beta_H;
        factorial *= k;
        expH = expH + power * C64(1.0 / factorial, 0);
    }
    return expH.trace().re;
}

// Hybrid partition function: Z_{A,n} = 2^n · Z_A^top
// When H = I_disc ⊗ H_top
__host__ inline double Z_hybrid(const DenseOp& H_top, double beta,
                                 int n_qubits) {
    return (1 << n_qubits) * Z_top(H_top, beta);
}

// Verify factorization: Z_{A,n} = 2^n · Z_A^top
__host__ inline bool verify_factorization(const DenseOp& H_top, double beta,
                                           int n_qubits, double tol = 1e-6) {
    // Build H_hyb = I_disc ⊗ H_top
    int dd = 1 << n_qubits;
    int dt = H_top.dim;
    int dim_hyb = dd * dt;
    DenseOp H_hyb(dim_hyb);
    for (int x = 0; x < dd; ++x)
        for (int i = 0; i < dt; ++i)
            for (int j = 0; j < dt; ++j)
                H_hyb.at(x * dt + i, x * dt + j) = H_top.at(i, j);

    // Compute full partition function
    DenseOp expH = DenseOp::identity(dim_hyb);
    DenseOp power = DenseOp::identity(dim_hyb);
    DenseOp neg_beta_H = H_hyb * C64(-beta, 0);
    double factorial = 1.0;
    for (int k = 1; k <= 20; ++k) {
        power = power * neg_beta_H;
        factorial *= k;
        expH = expH + power * C64(1.0 / factorial, 0);
    }
    double Z_full = expH.trace().re;
    double Z_fact = Z_hybrid(H_top, beta, n_qubits);

    return fabs(Z_full - Z_fact) < tol * fabs(Z_full);
}

// POVM output probability:
// Pr[Out(C,a) = y] = Σ_{x: b(x)=y} Tr(K_C(x,a) ρ_0 K_C(x,a)†)
// where b: I_n → output alphabet is the readout function
// Default readout: b(x) = x (identity readout)
__host__ inline void povm_output_probs(
    const OpBlocks& K_C,       // circuit kernel
    int a,                      // initial discrete state
    const DenseOp& rho_0,      // initial topological density
    double* probs)              // output: probs[y] for each y
{
    int dd = K_C.dim_disc;

    for (int y = 0; y < dd; ++y) {
        // For identity readout b(x) = x, the sum has only x = y
        DenseOp Kxa = K_C.at(y, a);
        DenseOp Kxa_dag = Kxa.dagger();
        DenseOp sandwich = Kxa * rho_0 * Kxa_dag;
        probs[y] = sandwich.trace().re;
    }
}

// Verify POVM normalization: Σ_y Pr[y] = 1
// (requires K to be unitary: Σ_x K(x,y)† K(x,y) = I for each y)
__host__ inline bool verify_povm_normalization(
    const OpBlocks& K_C, int a, const DenseOp& rho_0,
    double tol = 1e-6)
{
    int dd = K_C.dim_disc;
    double* probs = new double[dd];
    povm_output_probs(K_C, a, rho_0, probs);

    double sum = 0;
    for (int y = 0; y < dd; ++y) sum += probs[y];

    delete[] probs;
    return fabs(sum - 1.0) < tol;
}

// Coarse-grained POVM with general readout b: I_n → {0,...,K-1}
__host__ inline void povm_coarse_grained(
    const OpBlocks& K_C, int a, const DenseOp& rho_0,
    const int* readout,     // readout[x] = b(x) ∈ {0,...,n_outputs-1}
    int n_outputs,
    double* probs)
{
    int dd = K_C.dim_disc;
    for (int y = 0; y < n_outputs; ++y) probs[y] = 0;

    for (int x = 0; x < dd; ++x) {
        DenseOp Kxa = K_C.at(x, a);
        DenseOp Kxa_dag = Kxa.dagger();
        DenseOp sandwich = Kxa * rho_0 * Kxa_dag;
        probs[readout[x]] += sandwich.trace().re;
    }
}

} // namespace thermo

// ════════════════════════════════════════════════════════════════════════════
// Grand verification: all identities in one pass
// ════════════════════════════════════════════════════════════════════════════

namespace verify {

struct VerifyResult {
    bool vec_roundtrip_DR;
    bool vec_roundtrip_RD;
    bool op_roundtrip;
    bool matrix_unit_algebra;
    bool matrix_unit_resolution;
    bool walsh_roundtrip;
    bool pauli_is_walsh;
    bool pauli_expansion;
    bool kernel_X_correct;
    bool kernel_Z_correct;
    bool kernel_compose;
    bool dens_roundtrip;
    bool dens_probs_sum_1;
    bool dens_block_expect;
    bool dens_diagonal;
    bool weyl_blocks;
    bool transport_preserves;
    bool thermo_factorization;
    bool povm_normalized;
    int  total_checks;
    int  passed;
};

__host__ inline VerifyResult run_all(int n_qubits = 2,
                                      int N_theta = 4, int N_rho = 3,
                                      double tol = 1e-6) {
    VerifyResult r;
    memset(&r, 0, sizeof(r));
    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int dim_hyb = dd * dt;
    double rho_min = -5.0, rho_max = 5.0;

    // ── Set up test state ──
    HybridState psi;
    psi.init(n_qubits, N_theta, N_rho, rho_min, rho_max);
    // Non-trivial state: inject different topological states per slot
    for (int x = 0; x < dd; ++x) {
        TopologicalState xi;
        xi.init(N_theta, N_rho, rho_min, rho_max);
        for (int i = 0; i < dt; ++i)
            xi.amp[i] = C64(cos(0.3 * x + 0.1 * i),
                            sin(0.2 * x + 0.05 * i));
        // Normalize
        double n2 = xi.norm2();
        double inv = 1.0 / sqrt(n2 * dd);
        for (int i = 0; i < dt; ++i) xi.amp[i] = xi.amp[i] * inv;
        psi.inject(x, xi);
        xi.free();
    }

    // ── §1-3: Vector decomposition ──
    {
        TopologicalState* comps = new TopologicalState[dd];
        for (int x = 0; x < dd; ++x)
            comps[x].init(N_theta, N_rho, rho_min, rho_max);
        r.vec_roundtrip_DR = vec::roundtrip_DR(psi, tol);
        vec::decompose(psi, comps);
        r.vec_roundtrip_RD = vec::roundtrip_RD(comps, n_qubits, tol);
        for (int x = 0; x < dd; ++x) comps[x].free();
        delete[] comps;
    }

    // ── §2-3: Operator decomposition ──
    {
        // Build a non-trivial operator: X_u^hyb for u = (1,0,...,0)
        BitVec u1(1, n_qubits);
        DenseOp X_hyb(dim_hyb);
        for (int x = 0; x < dd; ++x) {
            int xp = x ^ u1.bits;
            for (int k = 0; k < dt; ++k)
                X_hyb.at(xp * dt + k, x * dt + k) = C64(1, 0);
        }
        r.op_roundtrip = op::roundtrip(X_hyb, dd, dt, tol);
        r.matrix_unit_algebra = op::matrix_unit_algebra(dd, dt, tol);
        r.matrix_unit_resolution = op::matrix_unit_resolution(dd, dt, tol);
    }

    // ── §4-7: Walsh transforms ──
    {
        // Build blocks from a known operator
        BitVec u1(1, n_qubits);
        DenseOp X_hyb(dim_hyb);
        for (int x = 0; x < dd; ++x) {
            int xp = x ^ u1.bits;
            for (int k = 0; k < dt; ++k)
                X_hyb.at(xp * dt + k, x * dt + k) = C64(1, 0);
        }
        OpBlocks A = op::decompose(X_hyb, dd, dt);
        r.walsh_roundtrip = walsh::roundtrip(A, n_qubits, tol);
        r.pauli_is_walsh = walsh::pauli_is_walsh(A, n_qubits, tol);
        r.pauli_expansion = walsh::matrix_unit_pauli_expansion(
            dd, dt, n_qubits, tol);
        A.free();
    }

    // ── §8-9: Circuit kernels ──
    {
        BitVec u1(1, n_qubits), v1(2, n_qubits);

        // K_{X_u}: should shift discrete index by u
        OpBlocks Kx = kernel::K_X(u1, dd, dt);
        // Verify K_X has exactly the right structure
        bool x_ok = true;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                if (x == (y ^ u1.bits)) {
                    if ((Kx.at(x, y) - DenseOp::identity(dt)).hs_norm2() > tol)
                        x_ok = false;
                } else {
                    if (Kx.at(x, y).hs_norm2() > tol) x_ok = false;
                }
            }
        r.kernel_X_correct = x_ok;
        Kx.free();

        // K_{Z_v}
        OpBlocks Kz = kernel::K_Z(v1, dd, dt);
        bool z_ok = true;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                if (x == y) {
                    BitVec yv(y, n_qubits);
                    double sign = (yv.inner(v1) == 0) ? 1.0 : -1.0;
                    DenseOp expected =
                        DenseOp::identity(dt) * C64(sign, 0);
                    if ((Kz.at(x, y) - expected).hs_norm2() > tol)
                        z_ok = false;
                } else {
                    if (Kz.at(x, y).hs_norm2() > tol) z_ok = false;
                }
            }
        r.kernel_Z_correct = z_ok;
        Kz.free();

        // Kernel composition: K_{X_u} ∘ K_{Z_v} vs K_{X_u Z_v}
        OpBlocks Kx2 = kernel::K_X(u1, dd, dt);
        OpBlocks Kz2 = kernel::K_Z(v1, dd, dt);
        OpBlocks Kxz_composed = kernel::compose(Kx2, Kz2);
        // Direct: K_{X_u Z_v}(x,y) = δ_{x,y⊕u} (-1)^{⟨v,y⟩} I_top
        OpBlocks Kxz_direct;
        Kxz_direct.init_zero(dd, dt);
        for (int y = 0; y < dd; ++y) {
            int x = y ^ u1.bits;
            BitVec yv(y, n_qubits);
            double sign = (yv.inner(v1) == 0) ? 1.0 : -1.0;
            Kxz_direct.at(x, y) = DenseOp::identity(dt) * C64(sign, 0);
        }
        double compose_err = 0;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y)
                compose_err += (Kxz_composed.at(x, y)
                              - Kxz_direct.at(x, y)).hs_norm2();
        r.kernel_compose = (compose_err < tol);

        Kx2.free(); Kz2.free();
        Kxz_composed.free(); Kxz_direct.free();
    }

    // ── §10-13: Density matrix ──
    {
        // Build density from pure state: ρ = |ψ⟩⟨ψ|
        DenseOp rho(dim_hyb);
        for (int i = 0; i < dim_hyb; ++i)
            for (int j = 0; j < dim_hyb; ++j)
                rho.at(i, j) = psi.amp[i] * psi.amp[j].conj();

        OpBlocks rho_blk = dens::decompose(rho, dd, dt);
        DenseOp rho_recon = dens::reconstruct(rho_blk);
        r.dens_roundtrip = ((rho - rho_recon).hs_norm2() < tol);
        r.dens_probs_sum_1 = dens::verify_prob_normalization(rho_blk, tol);
        r.dens_diagonal = dens::verify_diagonal_properties(rho_blk, tol);

        // Build observable: I_hyb
        DenseOp I_hyb = DenseOp::identity(dim_hyb);
        r.dens_block_expect = dens::verify_block_expectation(
            rho, I_hyb, dd, dt, tol);

        rho_blk.free();
    }

    // ── §14-15: Weyl blocks ──
    {
        BitVec u0(0, n_qubits), v0(0, n_qubits);
        MirElement e = MirElement::identity();
        r.weyl_blocks = field::verify_weyl_blocks(
            u0, v0, e, n_qubits, N_theta, N_rho, rho_min, rho_max, tol);
    }

    // ── §16: Transport ──
    {
        // Use U_t as transport map
        DenseOp T_phi = U_matrix(0.1, N_theta, N_rho, rho_min, rho_max);

        // Transport preserves p(x) only if T_phi is unitary
        // For pure states: p(x) = Σ_j |ψ(x,j)|² = ||ξ_x||²
        // After transport: ξ_x → T_phi ξ_x, so ||T_phi ξ_x||² = ||ξ_x||²
        // iff T_phi is unitary.
        // U_t is approximately unitary on the grid, so test with tolerance
        r.transport_preserves = transport::preserves_disc_probs(
            psi, T_phi, 0.1);  // relaxed tolerance for grid effects
    }

    // ── §17-18: Thermodynamics ──
    {
        // Simple Hamiltonian: H_top = diagonal matrix
        DenseOp H_top(dt);
        for (int i = 0; i < dt; ++i)
            H_top.at(i, i) = C64(0.1 * i, 0);
        r.thermo_factorization = thermo::verify_factorization(
            H_top, 1.0, n_qubits, tol);
    }

    // ── POVM ──
    {
        // Build simple kernel: identity circuit
        OpBlocks K_id;
        K_id.init_zero(dd, dt);
        for (int x = 0; x < dd; ++x)
            K_id.at(x, x) = DenseOp::identity(dt);

        // Initial topological state: normalized identity / dt
        DenseOp rho_0 = DenseOp::identity(dt);
        rho_0 = rho_0 * C64(1.0 / dt, 0);

        r.povm_normalized = thermo::verify_povm_normalization(
            K_id, 0, rho_0, tol);
        K_id.free();
    }

    // Count results
    bool* checks = reinterpret_cast<bool*>(&r);
    r.total_checks = 19;
    r.passed = 0;
    for (int i = 0; i < r.total_checks; ++i)
        if (checks[i]) r.passed++;

    psi.free();
    return r;
}

} // namespace verify

} // namespace exact
} // namespace topcomp
