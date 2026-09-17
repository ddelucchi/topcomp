// ============================================================================
// TopComp: Kernelized Compute Doctrine
// ============================================================================
// The hybrid computer is now exactly a classical register carrying
// topological operator payloads along its transitions.  Every hybrid
// circuit is equivalently:
//
//   (1) a hybrid operator  C ∈ End(X_{A,m}^{hyb}),
//   (2) an operator-valued kernel  {K_C(x,a)}_{x,a ∈ I_m},
//   (3) a Weyl/Walsh synthesis  {B_C(u,v)}_{u,v ∈ I_m},
//
// with exact bijective *-algebra isomorphism
//   K_{A,m}: End(X_{A,m}^{hyb}) ≅ Mat_{I_m}(End(X_A^{top})).
//
// Contents (by section):
//
//   §1  *-algebra verification
//       K(CD) = K(C)K(D),  K(C*) = K(C)*,  unitarity conditions
//
//   §2  Projector kernel K_{P_b}
//
//   §3  Path-sum expansion for gate sequences
//
//   §4  Classical permutation lift  P̂_f
//       K_{P̂_f}(x,a) = δ_{x,f(a)} I
//
//   §5  Controlled operator  Ĉ_{f,{U_x}}
//       K(r,s) = δ_{r,f(s)} U_s
//       Composition law: Ĉ_{f,{U_x}} Ĉ_{g,{V_x}} = Ĉ_{f∘g, {U_{g(x)}V_x}}
//
//   §6  Induced discrete quantum channel
//       E_{C,ρ₀}(σ) = Tr_top(C(σ⊗ρ₀)C*)
//
//   §7  Quantum unitary lift  Ũ_{A,m}
//       K(x,a) = U_{x,a} · I_top
//
//   §8  Stochastic kernel channel
//       E_K(ρ) = Σ_{x,y} K(y|x) E_{y,y} ⊗ ρ_{x,x}
//
//   §9  Decomposition maps Dec_prob, Dec_full, Dec_class
//       Transformation laws under controlled operators
//
//   §10 Compilation maps Comp_{cl→hyb}, Comp_{cl→ctrl}
//       Sequential gate-sequence compilation with fiber payloads
//
//   §11 Grand verification
//
// Identities verified:
//   K(CD) = K(C)K(D)                          (*-multiplicative)
//   K(C*) = K(C)*                              (*-preserving)
//   C*C=I ⟺ Σ_x K(x,a)* K(x,b) = δ_{ab}I    (column unitarity)
//   P̂_{f∘g} = P̂_f · P̂_g                      (permutation homomorphism)
//   p_{P̂_f ρ P̂_f*}(f(x)) = p_ρ(x)            (probability permutation)
//   Ĉ_{f,{U}} Ĉ_{g,{V}} = Ĉ_{f∘g, {U_{g(x)}V_x}}
//   Dec_prob ∘ Ad_{Ĉ} = Perm_f ∘ Dec_prob
//   E_{C,ρ₀}(|a⟩⟨a|)_{xx} = p_{C,a}^disc(x)
//   K_{Ũ}(x,a) = U_{x,a} I_top
// ============================================================================
#pragma once
#include "exact_decomposition.cuh"

namespace topcomp {
namespace kc {

using measurement::DenseOp;
using exact::OpBlocks;

// ════════════════════════════════════════════════════════════════════════════
// §1  *-Algebra Isomorphism Verification
// ════════════════════════════════════════════════════════════════════════════

namespace star_alg {

// Verify K(CD) = K(C)·K(D)  (matrix multiplication of kernels)
__host__ inline bool verify_multiplicative(
    const DenseOp& C, const DenseOp& D,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp CD = C * D;
    OpBlocks K_C  = exact::op::decompose(C, dim_disc, dim_top);
    OpBlocks K_D  = exact::op::decompose(D, dim_disc, dim_top);
    OpBlocks K_CD = exact::op::decompose(CD, dim_disc, dim_top);
    OpBlocks K_prod = exact::kernel::compose(K_C, K_D);

    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a)
            err += (K_CD.at(x, a) - K_prod.at(x, a)).hs_norm2();

    K_C.free(); K_D.free(); K_CD.free(); K_prod.free();
    return err < tol;
}

// Verify K(C*) = K(C)*  where (K(C)*)_{x,a} = K_C(a,x)†
__host__ inline bool verify_star(
    const DenseOp& C, int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp Cstar = C.dagger();
    OpBlocks K_C     = exact::op::decompose(C, dim_disc, dim_top);
    OpBlocks K_Cstar = exact::op::decompose(Cstar, dim_disc, dim_top);

    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a) {
            DenseOp lhs = K_Cstar.at(x, a);
            DenseOp rhs = K_C.at(a, x).dagger();
            err += (lhs - rhs).hs_norm2();
        }

    K_C.free(); K_Cstar.free();
    return err < tol;
}

// Verify column unitarity: C*C = I ⟺ Σ_x K(x,a)* K(x,b) = δ_{a,b} I
__host__ inline bool verify_column_unitarity(
    const OpBlocks& K, int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp I_top = DenseOp::identity(dim_top);
    for (int a = 0; a < dim_disc; ++a)
        for (int b = 0; b < dim_disc; ++b) {
            DenseOp sum = DenseOp::zero(dim_top);
            for (int x = 0; x < dim_disc; ++x)
                sum = sum + K.at(x, a).dagger() * K.at(x, b);
            DenseOp expected = (a == b) ? I_top : DenseOp::zero(dim_top);
            if ((sum - expected).hs_norm2() > tol) return false;
        }
    return true;
}

// Verify row unitarity: CC* = I ⟺ Σ_a K(x,a) K(y,a)* = δ_{x,y} I
__host__ inline bool verify_row_unitarity(
    const OpBlocks& K, int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp I_top = DenseOp::identity(dim_top);
    for (int x = 0; x < dim_disc; ++x)
        for (int y = 0; y < dim_disc; ++y) {
            DenseOp sum = DenseOp::zero(dim_top);
            for (int a = 0; a < dim_disc; ++a)
                sum = sum + K.at(x, a) * K.at(y, a).dagger();
            DenseOp expected = (x == y) ? I_top : DenseOp::zero(dim_top);
            if ((sum - expected).hs_norm2() > tol) return false;
        }
    return true;
}

} // namespace star_alg

// ════════════════════════════════════════════════════════════════════════════
// §2  Projector Kernel
// ════════════════════════════════════════════════════════════════════════════

namespace proj {

// K_{P_b^hyb}(x,a) = δ_{x,b} δ_{a,b} I_top
__host__ inline OpBlocks K_P(int b, int dim_disc, int dim_top) {
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    K.at(b, b) = DenseOp::identity(dim_top);
    return K;
}

// Verify projector kernel is idempotent: K_P^2 = K_P
__host__ inline bool verify_idempotent(int b, int dim_disc, int dim_top,
                                        double tol = 1e-10) {
    OpBlocks Kp = K_P(b, dim_disc, dim_top);
    OpBlocks Kp2 = exact::kernel::compose(Kp, Kp);
    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a)
            err += (Kp2.at(x, a) - Kp.at(x, a)).hs_norm2();
    Kp.free(); Kp2.free();
    return err < tol;
}

// Verify projector resolution: Σ_b K_{P_b} applied sequentially = I kernel
__host__ inline bool verify_resolution(int dim_disc, int dim_top,
                                        double tol = 1e-10) {
    OpBlocks sum;
    sum.init_zero(dim_disc, dim_top);
    for (int b = 0; b < dim_disc; ++b)
        for (int x = 0; x < dim_disc; ++x)
            for (int a = 0; a < dim_disc; ++a) {
                OpBlocks Kp = K_P(b, dim_disc, dim_top);
                sum.at(x, a) = sum.at(x, a) + Kp.at(x, a);
                Kp.free();
            }
    // Identity kernel: K_I(x,a) = δ_{x,a} I_top
    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a) {
            DenseOp expected = (x == a) ? DenseOp::identity(dim_top)
                                        : DenseOp::zero(dim_top);
            err += (sum.at(x, a) - expected).hs_norm2();
        }
    sum.free();
    return err < tol;
}

} // namespace proj

// ════════════════════════════════════════════════════════════════════════════
// §3  Path-Sum Expansion for Gate Sequences
// ════════════════════════════════════════════════════════════════════════════

namespace path_sum {

// K_C(x,a) = Σ_{x_1,...,x_{T-1}} K_{G_T}(x,x_{T-1}) ··· K_{G_1}(x_1,a)
// Given an array of gate kernels, compute the composed circuit kernel
__host__ inline OpBlocks from_gate_sequence(
    const OpBlocks* gate_kernels, int T)
{
    if (T == 0) {
        // Identity kernel
        int dd = gate_kernels[0].dim_disc;
        int dt = gate_kernels[0].dim_top;
        OpBlocks K;
        K.init_zero(dd, dt);
        for (int x = 0; x < dd; ++x)
            K.at(x, x) = DenseOp::identity(dt);
        return K;
    }
    if (T == 1) {
        // Copy single kernel
        int dd = gate_kernels[0].dim_disc;
        int dt = gate_kernels[0].dim_top;
        OpBlocks K;
        K.init(dd, dt);
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a)
                K.at(x, a) = gate_kernels[0].at(x, a);
        return K;
    }

    // Sequential composition: G_1, G_2, ..., G_T
    // K_C = K_{G_T} ∘ ... ∘ K_{G_1}
    OpBlocks acc = exact::kernel::compose(gate_kernels[1], gate_kernels[0]);
    for (int j = 2; j < T; ++j) {
        OpBlocks next = exact::kernel::compose(gate_kernels[j], acc);
        acc.free();
        acc = next;
    }
    return acc;
}

// Verify path-sum matches direct hybrid operator application
__host__ inline bool verify_vs_direct(
    const DenseOp& C_direct,
    const OpBlocks* gate_kernels, int T,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    OpBlocks K_path = from_gate_sequence(gate_kernels, T);
    OpBlocks K_direct = exact::op::decompose(C_direct, dim_disc, dim_top);

    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a)
            err += (K_path.at(x, a) - K_direct.at(x, a)).hs_norm2();

    K_path.free(); K_direct.free();
    return err < tol;
}

} // namespace path_sum

// ════════════════════════════════════════════════════════════════════════════
// §4  Classical Permutation Lift  P̂_f
// ════════════════════════════════════════════════════════════════════════════

namespace perm_lift {

// Build kernel for P̂_f: K_{P̂_f}(x,a) = δ_{x,f(a)} I_top
__host__ inline OpBlocks build_kernel(const int* f, int dim_disc,
                                       int dim_top) {
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int a = 0; a < dim_disc; ++a)
        K.at(f[a], a) = DenseOp::identity(dim_top);
    return K;
}

// Build full hybrid operator: P̂_f = Σ_x E_{f(x),x} ⊗ I_top
__host__ inline DenseOp build_hybrid_op(const int* f, int dim_disc,
                                         int dim_top) {
    int dim_hyb = dim_disc * dim_top;
    DenseOp P(dim_hyb);
    for (int x = 0; x < dim_disc; ++x) {
        int fx = f[x];
        for (int k = 0; k < dim_top; ++k)
            P.at(fx * dim_top + k, x * dim_top + k) = C64(1, 0);
    }
    return P;
}

// Verify kernel matches direct decomposition
__host__ inline bool verify_kernel(const int* f, int dim_disc, int dim_top,
                                    double tol = 1e-10) {
    OpBlocks K_analytic = build_kernel(f, dim_disc, dim_top);
    DenseOp P = build_hybrid_op(f, dim_disc, dim_top);
    OpBlocks K_direct = exact::op::decompose(P, dim_disc, dim_top);

    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a)
            err += (K_analytic.at(x, a) - K_direct.at(x, a)).hs_norm2();

    K_analytic.free(); K_direct.free();
    return err < tol;
}

// Verify composition: P̂_{f∘g} = P̂_f · P̂_g
__host__ inline bool verify_composition(const int* f, const int* g,
                                         int dim_disc, int dim_top,
                                         double tol = 1e-10) {
    // Compute f∘g
    int* fg = new int[dim_disc];
    for (int x = 0; x < dim_disc; ++x) fg[x] = f[g[x]];

    DenseOp P_f = build_hybrid_op(f, dim_disc, dim_top);
    DenseOp P_g = build_hybrid_op(g, dim_disc, dim_top);
    DenseOp P_fg = build_hybrid_op(fg, dim_disc, dim_top);
    DenseOp prod = P_f * P_g;

    double err = (prod - P_fg).hs_norm2();
    delete[] fg;
    return err < tol;
}

// Verify probability permutation: p_{P̂_f ρ P̂_f*}(f(x)) = p_ρ(x)
__host__ inline bool verify_prob_permutation(
    const int* f, const DenseOp& rho,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp P = build_hybrid_op(f, dim_disc, dim_top);
    DenseOp Pdag = P.dagger();
    DenseOp rho_new = P * rho * Pdag;

    OpBlocks rho_blk = exact::op::decompose(rho, dim_disc, dim_top);
    OpBlocks rho_new_blk = exact::op::decompose(rho_new, dim_disc, dim_top);

    bool ok = true;
    for (int x = 0; x < dim_disc; ++x) {
        double p_old = rho_blk.at(x, x).trace().re;
        double p_new = rho_new_blk.at(f[x], f[x]).trace().re;
        if (fabs(p_old - p_new) > tol) { ok = false; break; }
    }

    rho_blk.free(); rho_new_blk.free();
    return ok;
}

// Verify unitarity: P̂_f is unitary when f is a bijection
__host__ inline bool verify_unitary(const int* f, int dim_disc, int dim_top,
                                     double tol = 1e-10) {
    DenseOp P = build_hybrid_op(f, dim_disc, dim_top);
    DenseOp Pdag = P.dagger();
    DenseOp PP = P * Pdag;
    DenseOp I = DenseOp::identity(dim_disc * dim_top);
    return (PP - I).hs_norm2() < tol;
}

// Weyl expansion coefficients:
// P_f = 2^{-n} Σ_{x,v} (-1)^{⟨v,x⟩} X_{f(x)+x} Z_v (⊗ I_top)
// α_{u,v}(f) = 2^{-n} Σ_x (-1)^{⟨v,x⟩} 𝟙[f(x) = x+u]
__host__ inline double weyl_coeff(const int* f, int dim_disc, int n,
                                   const BitVec& u, const BitVec& v) {
    double sum = 0;
    for (int x = 0; x < dim_disc; ++x) {
        if (f[x] == ((x ^ u.bits) & (dim_disc - 1))) {
            // f(x) = x ⊕ u  (addition mod 2^n = XOR)
            BitVec xv(x, n);
            double sign = (xv.inner(v) == 0) ? 1.0 : -1.0;
            sum += sign;
        }
    }
    return sum / dim_disc;
}

} // namespace perm_lift

// ════════════════════════════════════════════════════════════════════════════
// §5  Controlled Operator  Ĉ_{f,{U_x}}
// ════════════════════════════════════════════════════════════════════════════

namespace ctrl_op {

// Build kernel: K_{Ĉ}(r,s) = δ_{r,f(s)} · U_s
__host__ inline OpBlocks build_kernel(const int* f,
                                       const DenseOp* U,  // U[x] for x ∈ I_n
                                       int dim_disc, int dim_top) {
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int s = 0; s < dim_disc; ++s)
        K.at(f[s], s) = U[s];
    return K;
}

// Build full hybrid operator: Ĉ = Σ_x E_{f(x),x} ⊗ U_x
__host__ inline DenseOp build_hybrid_op(const int* f,
                                         const DenseOp* U,
                                         int dim_disc, int dim_top) {
    int dim_hyb = dim_disc * dim_top;
    DenseOp C(dim_hyb);
    for (int x = 0; x < dim_disc; ++x) {
        int fx = f[x];
        for (int i = 0; i < dim_top; ++i)
            for (int j = 0; j < dim_top; ++j)
                C.at(fx * dim_top + i, x * dim_top + j) = U[x].at(i, j);
    }
    return C;
}

// Verify kernel matches direct decomposition
__host__ inline bool verify_kernel(const int* f, const DenseOp* U,
                                    int dim_disc, int dim_top,
                                    double tol = 1e-6) {
    OpBlocks K_analytic = build_kernel(f, U, dim_disc, dim_top);
    DenseOp C = build_hybrid_op(f, U, dim_disc, dim_top);
    OpBlocks K_direct = exact::op::decompose(C, dim_disc, dim_top);

    double err = 0;
    for (int x = 0; x < dim_disc; ++x)
        for (int a = 0; a < dim_disc; ++a)
            err += (K_analytic.at(x, a) - K_direct.at(x, a)).hs_norm2();

    K_analytic.free(); K_direct.free();
    return err < tol;
}

// Composition law: Ĉ_{f,{U_x}} · Ĉ_{g,{V_x}} = Ĉ_{f∘g, {U_{g(x)}·V_x}}
__host__ inline bool verify_composition(
    const int* f, const DenseOp* U,
    const int* g, const DenseOp* V,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    // Build product directly
    DenseOp C_f = build_hybrid_op(f, U, dim_disc, dim_top);
    DenseOp C_g = build_hybrid_op(g, V, dim_disc, dim_top);
    DenseOp prod = C_f * C_g;

    // Build composed controlled operator
    int* fg = new int[dim_disc];
    DenseOp* W = new DenseOp[dim_disc];
    for (int x = 0; x < dim_disc; ++x) {
        fg[x] = f[g[x]];
        W[x] = U[g[x]] * V[x];    // U_{g(x)} · V_x
    }
    DenseOp C_fg = build_hybrid_op(fg, W, dim_disc, dim_top);

    double err = (prod - C_fg).hs_norm2();

    delete[] fg;
    delete[] W;
    return err < tol;
}

// Verify unitarity: f ∈ Sym(I_n), ∀x: U_x unitary ⟹ Ĉ unitary
__host__ inline bool verify_unitary(const int* f, const DenseOp* U,
                                     int dim_disc, int dim_top,
                                     double tol = 1e-6) {
    DenseOp C = build_hybrid_op(f, U, dim_disc, dim_top);
    DenseOp Cdag = C.dagger();
    DenseOp I = DenseOp::identity(dim_disc * dim_top);
    double err1 = (C * Cdag - I).hs_norm2();
    double err2 = (Cdag * C - I).hs_norm2();
    return err1 < tol && err2 < tol;
}

// Special case: f = id ⟹ Ĉ_{id,{U_x}} = Σ_x P_x^hyb (id ⊗ U_x)
// Action: (Ĉ Ψ)_x = U_x Ψ_x
__host__ inline DenseOp build_diagonal(const DenseOp* U,
                                        int dim_disc, int dim_top) {
    int* id_map = new int[dim_disc];
    for (int x = 0; x < dim_disc; ++x) id_map[x] = x;
    DenseOp C = build_hybrid_op(id_map, U, dim_disc, dim_top);
    delete[] id_map;
    return C;
}

// Factorization: Ĉ_{f,{U_x}} = P̂_f · Ĉ_{id,{U_x}}
__host__ inline bool verify_factorization(
    const int* f, const DenseOp* U,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    DenseOp C = build_hybrid_op(f, U, dim_disc, dim_top);
    DenseOp Pf = perm_lift::build_hybrid_op(f, dim_disc, dim_top);
    DenseOp C_diag = build_diagonal(U, dim_disc, dim_top);
    DenseOp prod = Pf * C_diag;
    return (C - prod).hs_norm2() < tol;
}

// Weyl expansion of controlled operator:
// Ĉ_{f,{U_x}} = Σ_{u,v} (X_u Z_v)^hyb ⊗ (2^{-n} Σ_{y: f(y)=y⊕u} (-1)^{⟨v,y⟩} U_y)
__host__ inline OpBlocks weyl_expansion(const int* f, const DenseOp* U,
                                         int dim_disc, int dim_top, int n) {
    OpBlocks B;
    B.init_zero(dim_disc, dim_top);

    for (int u = 0; u < dim_disc; ++u) {
        for (int v = 0; v < dim_disc; ++v) {
            DenseOp coeff = DenseOp::zero(dim_top);
            for (int y = 0; y < dim_disc; ++y) {
                if (f[y] == (y ^ u)) {   // f(y) = y ⊕ u
                    BitVec yv(y, n), vv(v, n);
                    double sign = (yv.inner(vv) == 0) ? 1.0 : -1.0;
                    coeff = coeff + U[y] * C64(sign, 0);
                }
            }
            B.at(u, v) = coeff;
        }
    }
    return B;
}

// Verify Weyl expansion matches Walsh transform of block decomposition
__host__ inline bool verify_weyl(const int* f, const DenseOp* U,
                                  int dim_disc, int dim_top, int n,
                                  double tol = 1e-6) {
    OpBlocks K = build_kernel(f, U, dim_disc, dim_top);
    OpBlocks B_walsh = exact::walsh::block_to_walsh(K, n);
    OpBlocks B_weyl = weyl_expansion(f, U, dim_disc, dim_top, n);

    double err = 0;
    for (int u = 0; u < dim_disc; ++u)
        for (int v = 0; v < dim_disc; ++v)
            err += (B_walsh.at(u, v) - B_weyl.at(u, v)).hs_norm2();

    K.free(); B_walsh.free(); B_weyl.free();
    return err < tol;
}

} // namespace ctrl_op

// ════════════════════════════════════════════════════════════════════════════
// §6  Induced Discrete Quantum Channel
// ════════════════════════════════════════════════════════════════════════════

namespace channel {

// E_{C,ρ₀}(σ) = Tr_top(C(σ ⊗ ρ₀)C*)
// Given circuit kernel K_C and initial topological state ρ₀,
// compute the induced channel on the discrete register.
//
// E_{C,ρ₀}(σ)_{x,y} = Σ_{a,b} σ_{a,b} Tr(K_C(x,a) ρ₀ K_C(y,b)*)
//
// Returns channel as dim_disc² × dim_disc² superoperator matrix:
//   (E)_{(x,y),(a,b)} = Tr(K_C(x,a) ρ₀ K_C(y,b)*)
__host__ inline DenseOp channel_matrix(const OpBlocks& K_C,
                                        const DenseOp& rho_0) {
    int dd = K_C.dim_disc;
    int dd2 = dd * dd;
    DenseOp E(dd2);  // superoperator: maps (a,b)-indexed to (x,y)-indexed

    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y)
            for (int a = 0; a < dd; ++a)
                for (int b = 0; b < dd; ++b) {
                    DenseOp prod = K_C.at(x, a) * rho_0
                                   * K_C.at(y, b).dagger();
                    E.at(x * dd + y, a * dd + b) = prod.trace();
                }
    return E;
}

// Apply the induced channel to a discrete density matrix σ
// E_{C,ρ₀}(σ)_{x,y} = Σ_{a,b} σ_{a,b} Tr(K_C(x,a) ρ₀ K_C(y,b)*)
__host__ inline DenseOp apply(const OpBlocks& K_C,
                               const DenseOp& rho_0,
                               const DenseOp& sigma) {
    int dd = K_C.dim_disc;
    DenseOp result(dd);

    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y) {
            C64 sum(0, 0);
            for (int a = 0; a < dd; ++a)
                for (int b = 0; b < dd; ++b) {
                    DenseOp prod = K_C.at(x, a) * rho_0
                                   * K_C.at(y, b).dagger();
                    sum = sum + sigma.at(a, b) * prod.trace();
                }
            result.at(x, y) = sum;
        }
    return result;
}

// Verify: E_{C,ρ₀}(|a⟩⟨a|)_{x,x} = p_{C,a}^disc(x)
__host__ inline bool verify_disc_prob_consistency(
    const OpBlocks& K_C, const DenseOp& rho_0,
    double tol = 1e-6)
{
    int dd = K_C.dim_disc;
    for (int a = 0; a < dd; ++a) {
        // Build |a⟩⟨a|
        DenseOp sigma(dd);
        sigma.at(a, a) = C64(1, 0);

        DenseOp Esigma = apply(K_C, rho_0, sigma);

        for (int x = 0; x < dd; ++x) {
            // p_{C,a}^disc(x) = Tr(K_C(x,a) ρ₀ K_C(x,a)*)
            DenseOp prod = K_C.at(x, a) * rho_0
                           * K_C.at(x, a).dagger();
            double p_direct = prod.trace().re;
            double p_channel = Esigma.at(x, x).re;
            if (fabs(p_direct - p_channel) > tol) return false;
        }
    }
    return true;
}

// Verify: for quantum unitary lift K(x,a) = U_{x,a}I with Tr(ρ₀)=1,
// the induced channel is E(σ) = UσU*
__host__ inline bool verify_unitary_case(
    const DenseOp& U, const DenseOp& rho_0,
    int dim_top, double tol = 1e-6)
{
    int dd = U.dim;
    OpBlocks K;
    K.init_zero(dd, dim_top);
    DenseOp I_top = DenseOp::identity(dim_top);
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a)
            K.at(x, a) = I_top * U.at(x, a);

    // Build test σ
    DenseOp sigma(dd);
    for (int i = 0; i < dd; ++i)
        for (int j = 0; j < dd; ++j)
            sigma.at(i, j) = C64(1.0 / dd, 0);  // maximally mixed

    DenseOp Esigma = apply(K, rho_0, sigma);
    DenseOp expected = U * sigma * U.dagger();

    // Scale by Tr(ρ₀) = 1
    double err = (Esigma - expected).hs_norm2();
    K.free();
    return err < tol;
}

// Verify: deterministic kernel K(x,a) = δ_{x,f(a)} T_a gives classical channel
__host__ inline bool verify_deterministic_case(
    const int* f, const DenseOp& rho_0,
    int dim_disc, int dim_top, double tol = 1e-6)
{
    // Build deterministic kernel
    DenseOp I_top = DenseOp::identity(dim_top);
    OpBlocks K;
    K.init_zero(dim_disc, dim_top);
    for (int a = 0; a < dim_disc; ++a)
        K.at(f[a], a) = I_top;

    for (int a = 0; a < dim_disc; ++a) {
        DenseOp sigma(dim_disc);
        sigma.at(a, a) = C64(1, 0);
        DenseOp Esigma = apply(K, rho_0, sigma);

        // Should give |f(a)⟩⟨f(a)|
        for (int x = 0; x < dim_disc; ++x) {
            double expected = (x == f[a]) ? 1.0 : 0.0;
            if (fabs(Esigma.at(x, x).re - expected) > tol) {
                K.free();
                return false;
            }
        }
    }
    K.free();
    return true;
}

} // namespace channel

// ════════════════════════════════════════════════════════════════════════════
// §7  Quantum Unitary Lift
// ════════════════════════════════════════════════════════════════════════════

namespace unitary_lift {

// Build kernel for Ũ_{A,m} = Q(U⊗I)Q^{-1}
// K_{Ũ}(x,a) = U_{x,a} · I_top
__host__ inline OpBlocks build_kernel(const DenseOp& U, int dim_top) {
    int dd = U.dim;
    OpBlocks K;
    K.init_zero(dd, dim_top);
    DenseOp I_top = DenseOp::identity(dim_top);
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a)
            K.at(x, a) = I_top * U.at(x, a);
    return K;
}

// Build full hybrid operator: Ũ = Σ_{x,a} E_{x,a} ⊗ U_{x,a} I_top
__host__ inline DenseOp build_hybrid_op(const DenseOp& U, int dim_top) {
    int dd = U.dim;
    int dim_hyb = dd * dim_top;
    DenseOp T(dim_hyb);
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a)
            for (int k = 0; k < dim_top; ++k)
                T.at(x * dim_top + k, a * dim_top + k) = U.at(x, a);
    return T;
}

// Verify kernel matches direct decomposition
__host__ inline bool verify_kernel(const DenseOp& U, int dim_top,
                                    double tol = 1e-6) {
    OpBlocks K_analytic = build_kernel(U, dim_top);
    DenseOp T = build_hybrid_op(U, dim_top);
    OpBlocks K_direct = exact::op::decompose(T, U.dim, dim_top);

    double err = 0;
    for (int x = 0; x < U.dim; ++x)
        for (int a = 0; a < U.dim; ++a)
            err += (K_analytic.at(x, a) - K_direct.at(x, a)).hs_norm2();

    K_analytic.free(); K_direct.free();
    return err < tol;
}

// Verify Eval(Ũ) = U ⊗ I:
// Q^{-1} Ũ Q = U ⊗ I is trivially true by construction
// But verify the kernel gives correct discrete prob:
// for |a⟩ input → p(x) = |U_{x,a}|²
__host__ inline bool verify_disc_probs(const DenseOp& U, int dim_top,
                                        double tol = 1e-6) {
    OpBlocks K = build_kernel(U, dim_top);
    DenseOp rho_0 = DenseOp::identity(dim_top);
    rho_0 = rho_0 * C64(1.0 / dim_top, 0);

    for (int a = 0; a < U.dim; ++a) {
        double sum = 0;
        for (int x = 0; x < U.dim; ++x) {
            DenseOp prod = K.at(x, a) * rho_0 * K.at(x, a).dagger();
            double p = prod.trace().re;
            double expected = U.at(x, a).norm2();
            if (fabs(p - expected) > tol) { K.free(); return false; }
            sum += p;
        }
        if (fabs(sum - 1.0) > tol) { K.free(); return false; }
    }
    K.free();
    return true;
}

// Verify unitarity of the lift: U unitary ⟹ Ũ unitary
__host__ inline bool verify_unitarity(const DenseOp& U, int dim_top,
                                       double tol = 1e-6) {
    OpBlocks K = build_kernel(U, dim_top);
    bool col = star_alg::verify_column_unitarity(K, U.dim, dim_top, tol);
    bool row = star_alg::verify_row_unitarity(K, U.dim, dim_top, tol);
    K.free();
    return col && row;
}

// Weyl coefficients: B_{Ũ}(u,v) = Σ_a (-1)^{⟨v,a⟩} U_{a⊕u,a} · I_top
__host__ inline OpBlocks weyl_coefficients(const DenseOp& U,
                                            int dim_top, int n) {
    int dd = U.dim;
    OpBlocks B;
    B.init_zero(dd, dim_top);
    DenseOp I_top = DenseOp::identity(dim_top);

    for (int u = 0; u < dd; ++u)
        for (int v = 0; v < dd; ++v) {
            C64 sum(0, 0);
            for (int a = 0; a < dd; ++a) {
                int au = a ^ u;
                BitVec av(a, n), vv(v, n);
                double sign = (av.inner(vv) == 0) ? 1.0 : -1.0;
                sum = sum + U.at(au, a) * sign;
            }
            B.at(u, v) = I_top * sum;
        }
    return B;
}

} // namespace unitary_lift

// ════════════════════════════════════════════════════════════════════════════
// §8  Stochastic Kernel Channel
// ════════════════════════════════════════════════════════════════════════════

namespace stochastic {

// E_K(ρ) = Σ_{x,y} K(y|x) E_{y,y} ⊗ ρ_{x,x}
// where K: I_n × I_n → [0,1] is a stochastic transition matrix
// K[y * dim_disc + x] = K(y|x), normalized: Σ_y K(y|x) = 1 for each x
__host__ inline OpBlocks apply_channel(const double* K_stoch,
                                        const OpBlocks& rho_blk) {
    int dd = rho_blk.dim_disc;
    int dt = rho_blk.dim_top;
    OpBlocks result;
    result.init_zero(dd, dt);

    for (int r = 0; r < dd; ++r)
        for (int s = 0; s < dd; ++s) {
            if (r != s) continue;   // only diagonal survives
            DenseOp sum = DenseOp::zero(dt);
            for (int x = 0; x < dd; ++x)
                sum = sum + rho_blk.at(x, x) * C64(K_stoch[r * dd + x], 0);
            result.at(r, r) = sum;
        }
    return result;
}

// Verify normalization: Σ_y K(y|x) = 1 for all x
__host__ inline bool verify_normalization(const double* K_stoch,
                                           int dim_disc, double tol = 1e-10) {
    for (int x = 0; x < dim_disc; ++x) {
        double sum = 0;
        for (int y = 0; y < dim_disc; ++y)
            sum += K_stoch[y * dim_disc + x];
        if (fabs(sum - 1.0) > tol) return false;
    }
    return true;
}

// Verify discrete prob update: p_{E_K(ρ)}(r) = Σ_x K(r|x) p_ρ(x)
__host__ inline bool verify_prob_update(const double* K_stoch,
                                         const OpBlocks& rho_blk,
                                         double tol = 1e-6) {
    int dd = rho_blk.dim_disc;
    OpBlocks result = apply_channel(K_stoch, rho_blk);

    for (int r = 0; r < dd; ++r) {
        double p_result = result.at(r, r).trace().re;
        double p_expected = 0;
        for (int x = 0; x < dd; ++x)
            p_expected += K_stoch[r * dd + x] * rho_blk.at(x, x).trace().re;
        if (fabs(p_result - p_expected) > tol) { result.free(); return false; }
    }
    result.free();
    return true;
}

// Verify CPTP: output is trace-normalized if input is
__host__ inline bool verify_trace_preservation(const double* K_stoch,
                                                const OpBlocks& rho_blk,
                                                double tol = 1e-6) {
    int dd = rho_blk.dim_disc;
    OpBlocks result = apply_channel(K_stoch, rho_blk);

    double tr_in = 0, tr_out = 0;
    for (int x = 0; x < dd; ++x) {
        tr_in += rho_blk.at(x, x).trace().re;
        tr_out += result.at(x, x).trace().re;
    }
    result.free();
    return fabs(tr_in - tr_out) < tol;
}

} // namespace stochastic

// ════════════════════════════════════════════════════════════════════════════
// §9  Decomposition Maps and Transformation Laws
// ════════════════════════════════════════════════════════════════════════════

namespace dec {

// Dec_prob: ρ ↦ (p_ρ(x))_{x ∈ I_n}
__host__ inline void dec_prob(const OpBlocks& rho_blk, double* probs) {
    for (int x = 0; x < rho_blk.dim_disc; ++x)
        probs[x] = rho_blk.at(x, x).trace().re;
}

// Dec_full: ρ ↦ (ρ_{x,y})_{x,y ∈ I_n}
// Already given by OpBlocks = op::decompose

// Dec_class: O ↦ (B_{u,v})_{u,v}
// Already given by walsh::block_to_walsh

// Verify Dec_prob ∘ Ad_{Ĉ_{f,{U_x}}} = Perm_f ∘ Dec_prob
// i.e., p_{Ĉ ρ Ĉ*}(r) = p_ρ(f^{-1}(r))
__host__ inline bool verify_prob_transform(
    const int* f, const DenseOp* U,
    const DenseOp& rho, int dim_disc, int dim_top,
    double tol = 1e-6)
{
    DenseOp C = ctrl_op::build_hybrid_op(f, U, dim_disc, dim_top);
    DenseOp Cdag = C.dagger();
    DenseOp rho_new = C * rho * Cdag;

    OpBlocks old_blk = exact::op::decompose(rho, dim_disc, dim_top);
    OpBlocks new_blk = exact::op::decompose(rho_new, dim_disc, dim_top);

    // Build f^{-1}
    int* finv = new int[dim_disc];
    for (int x = 0; x < dim_disc; ++x) finv[f[x]] = x;

    bool ok = true;
    for (int r = 0; r < dim_disc; ++r) {
        double p_new = new_blk.at(r, r).trace().re;
        double p_old = old_blk.at(finv[r], finv[r]).trace().re;
        if (fabs(p_new - p_old) > tol) { ok = false; break; }
    }

    delete[] finv;
    old_blk.free(); new_blk.free();
    return ok;
}

// Verify Dec_full ∘ Ad_{Ĉ_{f,{U_x}}} = CtrlAd_{f,{U_x}} ∘ Dec_full
// i.e., (Ĉ ρ Ĉ*)_{r,s} = U_{f^{-1}(r)} · ρ_{f^{-1}(r), f^{-1}(s)} · U_{f^{-1}(s)}*
__host__ inline bool verify_full_transform(
    const int* f, const DenseOp* U,
    const DenseOp& rho, int dim_disc, int dim_top,
    double tol = 1e-6)
{
    DenseOp C = ctrl_op::build_hybrid_op(f, U, dim_disc, dim_top);
    DenseOp Cdag = C.dagger();
    DenseOp rho_new = C * rho * Cdag;

    OpBlocks old_blk = exact::op::decompose(rho, dim_disc, dim_top);
    OpBlocks new_blk = exact::op::decompose(rho_new, dim_disc, dim_top);

    // Build f^{-1}
    int* finv = new int[dim_disc];
    for (int x = 0; x < dim_disc; ++x) finv[f[x]] = x;

    double err = 0;
    for (int r = 0; r < dim_disc; ++r)
        for (int s = 0; s < dim_disc; ++s) {
            DenseOp lhs = new_blk.at(r, s);
            DenseOp rhs = U[finv[r]] * old_blk.at(finv[r], finv[s])
                          * U[finv[s]].dagger();
            err += (lhs - rhs).hs_norm2();
        }

    delete[] finv;
    old_blk.free(); new_blk.free();
    return err < tol;
}

} // namespace dec

// ════════════════════════════════════════════════════════════════════════════
// §10  Compilation Maps
// ════════════════════════════════════════════════════════════════════════════

namespace compilation {

// Comp_{cl→hyb}(f) = P̂_f
// Just wraps perm_lift::build_hybrid_op

// Comp_{cl→ctrl}(f_1, ..., f_T ; {U_{j,x}})
// = Ĉ_{f_T,{U_{T,x}}} ··· Ĉ_{f_1,{U_{1,x}}}
// = Ĉ_{F, {W_x}}
// where F = f_T ∘ ··· ∘ f_1
// and   W_x = U_{T, f_{T-1}···f_1(x)} ··· U_{2, f_1(x)} · U_{1,x}
__host__ inline DenseOp comp_cl_to_ctrl(
    const int* const* f_steps,      // f_steps[j] = f_j
    const DenseOp* const* U_steps,  // U_steps[j][x] = U_{j,x}
    int T, int dim_disc, int dim_top)
{
    if (T == 0)
        return DenseOp::identity(dim_disc * dim_top);

    DenseOp acc = ctrl_op::build_hybrid_op(
        f_steps[0], U_steps[0], dim_disc, dim_top);

    for (int j = 1; j < T; ++j) {
        DenseOp step = ctrl_op::build_hybrid_op(
            f_steps[j], U_steps[j], dim_disc, dim_top);
        DenseOp next = step * acc;
        acc = next;
    }
    return acc;
}

// Verify compiled result matches Ĉ_{F, {W_x}} formula
__host__ inline bool verify_compilation(
    const int* const* f_steps,
    const DenseOp* const* U_steps,
    int T, int dim_disc, int dim_top,
    double tol = 1e-6)
{
    DenseOp C_seq = comp_cl_to_ctrl(f_steps, U_steps, T, dim_disc, dim_top);

    // Compute F = f_T ∘ ··· ∘ f_1
    int* F = new int[dim_disc];
    for (int x = 0; x < dim_disc; ++x) F[x] = x;
    for (int j = 0; j < T; ++j) {
        int* F_next = new int[dim_disc];
        for (int x = 0; x < dim_disc; ++x) F_next[x] = f_steps[j][F[x]];
        delete[] F;
        F = F_next;
    }

    // Compute W_x = U_{T-1, ...} ··· U_0[x]
    DenseOp* W = new DenseOp[dim_disc];
    for (int x = 0; x < dim_disc; ++x) {
        // Track position through the steps
        int pos = x;
        W[x] = U_steps[0][pos];
        pos = f_steps[0][pos];
        for (int j = 1; j < T; ++j) {
            W[x] = U_steps[j][pos] * W[x];
            pos = f_steps[j][pos];
        }
    }

    DenseOp C_formula = ctrl_op::build_hybrid_op(F, W, dim_disc, dim_top);
    double err = (C_seq - C_formula).hs_norm2();

    delete[] F;
    delete[] W;
    return err < tol;
}

// Embed irreversible f: I_r → I_k via reversible extension
// f̃(a,b) = (a, b ⊕ f(a))
// Returns the permutation on I_{r+k} = I_r × I_k
__host__ inline void irreversible_extension(
    const int* f,    // f: I_r → I_k, size dim_r
    int dim_r,       // 2^r
    int dim_k,       // 2^k
    int* f_tilde)    // output: bijection on I_{r+k}, size dim_r * dim_k
{
    for (int a = 0; a < dim_r; ++a)
        for (int b = 0; b < dim_k; ++b) {
            int idx_in = a * dim_k + b;
            int idx_out = a * dim_k + (b ^ f[a]);
            f_tilde[idx_in] = idx_out;
        }
}

} // namespace compilation

// ════════════════════════════════════════════════════════════════════════════
// §11  Grand Verification
// ════════════════════════════════════════════════════════════════════════════

namespace verify {

struct KCVerifyResult {
    bool star_multiplicative;     // K(CD) = K(C)K(D)
    bool star_adjoint;            // K(C*) = K(C)*
    bool column_unitarity;        // C*C=I kernel condition
    bool row_unitarity;           // CC*=I kernel condition
    bool proj_kernel;             // K_{P_b} correct
    bool proj_idempotent;         // K_P² = K_P
    bool proj_resolution;         // Σ_b K_{P_b} = K_I
    bool path_sum;                // gate sequence path sum
    bool perm_kernel;             // P̂_f kernel correct
    bool perm_composition;        // P̂_{f∘g} = P̂_f P̂_g
    bool perm_unitary;            // P̂_f unitary
    bool perm_prob;               // probability permutation
    bool ctrl_kernel;             // Ĉ kernel correct
    bool ctrl_composition;        // Ĉ composition law
    bool ctrl_unitary;            // Ĉ unitarity
    bool ctrl_factorization;      // Ĉ = P̂ · Ĉ_diag
    bool ctrl_weyl;               // Weyl expansion
    bool channel_prob;            // channel ↔ disc prob
    bool channel_deterministic;   // deterministic kernel → classical
    bool unitary_kernel;          // Ũ kernel correct
    bool unitary_probs;           // Ũ discrete probs
    bool unitary_unitarity;       // Ũ kernel unitary
    bool stoch_prob_update;       // stochastic prob update
    bool stoch_trace;             // stochastic trace preservation
    bool dec_prob_transform;      // Dec_prob transform law
    bool dec_full_transform;      // Dec_full transform law
    bool compilation_correct;     // compilation formula
    bool irrev_extension;         // irreversible → reversible
    int  total_checks;
    int  passed;
};

__host__ inline KCVerifyResult run_all(int n_qubits = 1,
                                        int N_theta = 4, int N_rho = 3,
                                        double tol = 1e-4) {
    KCVerifyResult r;
    memset(&r, 0, sizeof(r));
    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int dim_hyb = dd * dt;
    (void)dim_hyb;

    // ── Build test operators ──

    // Permutation f: cyclic shift
    int* f = new int[dd];
    for (int x = 0; x < dd; ++x) f[x] = (x + 1) % dd;

    // Second permutation g: reverse
    int* g = new int[dd];
    for (int x = 0; x < dd; ++x) g[x] = (dd - 1 - x);

    // Fiber operators: U_x = rotation-ish diagonal
    DenseOp* U_ops = new DenseOp[dd];
    for (int x = 0; x < dd; ++x) {
        U_ops[x] = DenseOp::identity(dt);
        for (int k = 0; k < dt; ++k) {
            double angle = 0.1 * (x + 1) * (k + 1);
            U_ops[x].at(k, k) = C64(cos(angle), sin(angle));
        }
    }

    // Build a unitary hybrid operator for *-algebra tests
    DenseOp C_hyb = ctrl_op::build_hybrid_op(f, U_ops, dd, dt);
    DenseOp D_hyb = perm_lift::build_hybrid_op(g, dd, dt);

    // Build density matrix for state tests
    DenseOp rho(dim_hyb);
    for (int i = 0; i < dim_hyb; ++i) {
        double val = cos(0.3 * i + 0.1);
        rho.at(i, i) = C64(fabs(val), 0);
    }
    // Normalize to trace 1
    double tr = rho.trace().re;
    rho = rho * C64(1.0 / tr, 0);

    // Initial topological density
    DenseOp rho_0 = DenseOp::identity(dt);
    rho_0 = rho_0 * C64(1.0 / dt, 0);

    // ── §1: *-Algebra ──
    r.star_multiplicative = star_alg::verify_multiplicative(
        C_hyb, D_hyb, dd, dt, tol);
    r.star_adjoint = star_alg::verify_star(C_hyb, dd, dt, tol);
    {
        OpBlocks K_C = exact::op::decompose(C_hyb, dd, dt);
        r.column_unitarity = star_alg::verify_column_unitarity(
            K_C, dd, dt, tol);
        r.row_unitarity = star_alg::verify_row_unitarity(
            K_C, dd, dt, tol);
        K_C.free();
    }

    // ── §2: Projector ──
    r.proj_kernel = true;
    for (int b = 0; b < dd; ++b) {
        OpBlocks Kp = proj::K_P(b, dd, dt);
        // Verify: only (b,b) is non-zero, equals I_top
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a) {
                if (x == b && a == b) {
                    if ((Kp.at(x, a) - DenseOp::identity(dt)).hs_norm2() > tol)
                        r.proj_kernel = false;
                } else {
                    if (Kp.at(x, a).hs_norm2() > tol)
                        r.proj_kernel = false;
                }
            }
        Kp.free();
    }
    r.proj_idempotent = proj::verify_idempotent(0, dd, dt, tol);
    r.proj_resolution = proj::verify_resolution(dd, dt, tol);

    // ── §3: Path sum ──
    {
        BitVec u1(1, n_qubits), v1(0, n_qubits);
        OpBlocks K1 = exact::kernel::K_X(u1, dd, dt);
        OpBlocks K2 = exact::kernel::K_Z(v1, dd, dt);
        OpBlocks gates[2] = { K1, K2 };  // not owning
        OpBlocks K_path = path_sum::from_gate_sequence(gates, 2);
        OpBlocks K_composed = exact::kernel::compose(K2, K1);
        double err = 0;
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a)
                err += (K_path.at(x, a) - K_composed.at(x, a)).hs_norm2();
        r.path_sum = (err < tol);
        K1.free(); K2.free(); K_path.free(); K_composed.free();
    }

    // ── §4: Permutation lift ──
    r.perm_kernel = perm_lift::verify_kernel(f, dd, dt, tol);
    r.perm_composition = perm_lift::verify_composition(f, g, dd, dt, tol);
    r.perm_unitary = perm_lift::verify_unitary(f, dd, dt, tol);
    r.perm_prob = perm_lift::verify_prob_permutation(f, rho, dd, dt, tol);

    // ── §5: Controlled operator ──
    r.ctrl_kernel = ctrl_op::verify_kernel(f, U_ops, dd, dt, tol);
    r.ctrl_composition = ctrl_op::verify_composition(
        f, U_ops, g, U_ops, dd, dt, tol);
    r.ctrl_unitary = ctrl_op::verify_unitary(f, U_ops, dd, dt, tol);
    r.ctrl_factorization = ctrl_op::verify_factorization(
        f, U_ops, dd, dt, tol);
    r.ctrl_weyl = ctrl_op::verify_weyl(f, U_ops, dd, dt, n_qubits, tol);

    // ── §6: Induced channel ──
    {
        OpBlocks K_f = perm_lift::build_kernel(f, dd, dt);
        r.channel_prob = channel::verify_disc_prob_consistency(
            K_f, rho_0, tol);
        r.channel_deterministic = channel::verify_deterministic_case(
            f, rho_0, dd, dt, tol);
        K_f.free();
    }

    // ── §7: Unitary lift ──
    {
        // Build a small unitary on V_n
        DenseOp U_disc(dd);
        // Hadamard-like
        double inv = 1.0 / sqrt(dd);
        for (int i = 0; i < dd; ++i)
            for (int j = 0; j < dd; ++j) {
                BitVec iv(i, n_qubits), jv(j, n_qubits);
                double sign = (iv.inner(jv) == 0) ? 1.0 : -1.0;
                U_disc.at(i, j) = C64(sign * inv, 0);
            }
        r.unitary_kernel = unitary_lift::verify_kernel(U_disc, dt, tol);
        r.unitary_probs = unitary_lift::verify_disc_probs(U_disc, dt, tol);
        r.unitary_unitarity = unitary_lift::verify_unitarity(
            U_disc, dt, tol);
    }

    // ── §8: Stochastic channel ──
    {
        // Uniform stochastic kernel
        double* K_stoch = new double[dd * dd];
        for (int y = 0; y < dd; ++y)
            for (int x = 0; x < dd; ++x)
                K_stoch[y * dd + x] = 1.0 / dd;

        OpBlocks rho_blk = exact::op::decompose(rho, dd, dt);
        r.stoch_prob_update = stochastic::verify_prob_update(
            K_stoch, rho_blk, tol);
        r.stoch_trace = stochastic::verify_trace_preservation(
            K_stoch, rho_blk, tol);
        rho_blk.free();
        delete[] K_stoch;
    }

    // ── §9: Decomposition transforms ──
    r.dec_prob_transform = dec::verify_prob_transform(
        f, U_ops, rho, dd, dt, tol);
    r.dec_full_transform = dec::verify_full_transform(
        f, U_ops, rho, dd, dt, tol);

    // ── §10: Compilation ──
    {
        const int* f_steps[2] = { f, g };
        const DenseOp* U_steps[2] = { U_ops, U_ops };
        r.compilation_correct = compilation::verify_compilation(
            f_steps, U_steps, 2, dd, dt, tol);
    }

    // ── Irreversible extension ──
    {
        // f: I_2 → I_2, identity
        int f_irrev[2] = { 0, 1 };
        int dim_r = 2, dim_k = 2;
        int* f_tilde = new int[dim_r * dim_k];
        compilation::irreversible_extension(f_irrev, dim_r, dim_k, f_tilde);

        // f̃ on I_4: f̃(a,b) = (a, b ⊕ f(a)) = (a, b ⊕ a)
        // f̃(0,0)=(0,0), f̃(0,1)=(0,1), f̃(1,0)=(1,1), f̃(1,1)=(1,0)
        bool ok = (f_tilde[0] == 0 && f_tilde[1] == 1 &&
                   f_tilde[2] == 3 && f_tilde[3] == 2);
        // Verify it's a bijection
        bool seen[4] = { false };
        for (int i = 0; i < 4; ++i) {
            if (f_tilde[i] < 0 || f_tilde[i] >= 4) { ok = false; break; }
            if (seen[f_tilde[i]]) { ok = false; break; }
            seen[f_tilde[i]] = true;
        }
        r.irrev_extension = ok;
        delete[] f_tilde;
    }

    // ── Count ──
    r.total_checks = 28;
    r.passed = 0;
    bool* checks = reinterpret_cast<bool*>(&r);
    for (int i = 0; i < r.total_checks; ++i)
        if (checks[i]) r.passed++;

    // ── Cleanup ──
    delete[] f;
    delete[] g;
    delete[] U_ops;

    return r;
}

} // namespace verify

} // namespace kc
} // namespace topcomp
