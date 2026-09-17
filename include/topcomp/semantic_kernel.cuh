// ============================================================================
// TopComp: Semantic Kernel — The Frozen Compute Doctrine
// ============================================================================
//
// This header freezes the SINGLE SOURCE OF TRUTH for the entire framework.
//
// Every backend, optimizer, serializer, visualizer, IR, and language binding
// must obey the contracts defined here.  If anything contradicts this file,
// this file wins.
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  The Exact Framework Triple                                        │
// │                                                                    │
// │   F_exact = ( ẽv_A,  Q_{A,n},  Ω_tot^{(A,n)} )                    │
// │                                                                    │
// │  ẽv_A     : evaluation–representation anchor                      │
// │  Q_{A,n}  : intertwiner  X_{A,n}^hyb ≅ V_n ⊗ X_A^top             │
// │  Ω_tot    : total cocycle (Ω_bool ⊕ Ω_A)                          │
// └─────────────────────────────────────────────────────────────────────┘
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │  The Exact Dictionaries                                            │
// │                                                                    │
// │  State:       Ψ  ↔  {Ψ_x}                                         │
// │  Observable:  O  ↔  {O_{x,y}}  ↔  {B_{u,v}}                       │
// │  Circuit:     C  ↔  {K_C(x,a)}  ↔  {B_C(u,v)}                     │
// │  Density:     ρ  ↔  {ρ_{x,y}}                                     │
// │                                                                    │
// │  These are EXACT BIJECTIONS — not approximations.                  │
// └─────────────────────────────────────────────────────────────────────┘
//
// Canonical objects — THE authoritative definition:
//
//   HYBRID STATE   : Ψ ∈ X_{A,n}^hyb = V_n ⊗ X_A^top
//                    Components: Ψ_x = π_x(Ψ) ∈ X_A^top
//                    Born rule:  p(x) = ‖Ψ_x‖²
//
//   HYBRID OPERATOR: O ∈ End(X_{A,n}^hyb)
//                    Blocks:  O_{x,y} = π_x O ι_y ∈ End(X_A^top)
//                    Walsh:   B_{u,v} = 2^{-n} Σ_y (-1)^{⟨v,y⟩} O_{y⊕u, y}
//
//   HYBRID CIRCUIT : C ∈ End(X_{A,n}^hyb),  unitary
//                    Kernel:  K_C(x,a) = π_x C ι_a ∈ End(X_A^top)
//                    Walsh:   B_C(u,v) = Σ_y (-1)^{⟨v,y⟩} K_C(y⊕u, y)
//                    *-algebra: K(CD) = K(C)K(D),  K(C*) = K(C)*
//
//   HYBRID DENSITY : ρ ∈ End(X_{A,n}^hyb),  ρ ≥ 0, Tr(ρ)=1
//                    Blocks:  ρ_{x,y} = π_x ρ ι_y
//                    Probs:   p(x) = Tr(ρ_{x,x})
//
//   CONTROLLED OP  : Ĉ_{f,{U_x}} = Σ_x E_{f(x),x} ⊗ U_x
//                    Kernel:  K(r,s) = δ_{r,f(s)} U_s
//                    Factorization: Ĉ_{f,{U}} = P̂_f · Ĉ_{id,{U}}
//                    Composition:   Ĉ_{f,U} · Ĉ_{g,V} = Ĉ_{f∘g, {U_{g(x)}V_x}}
//
//   PERM LIFT      : P̂_f = Σ_x E_{f(x),x} ⊗ I_top
//                    Kernel:  K(x,a) = δ_{x,f(a)} I
//
//   UNITARY LIFT   : Ũ = Q(U⊗I)Q^{-1}
//                    Kernel:  K(x,a) = U_{x,a} I_top
//
//   CHANNEL        : E_{C,ρ₀}(σ)_{x,y} = Σ_{a,b} σ_{a,b} Tr(K(x,a) ρ₀ K(y,b)*)
//
//   STOCH CHANNEL  : E_K(ρ)_{r,r} = Σ_x K(r|x) ρ_{x,x}
//
//   MEASUREMENT    : POVM {F_y}, Born: p(y) = Tr(ρ F_y)
//
//   EVALUATION     : C_word → DenseOp : gate sequence → full operator
//
// Convention normalization (authoritative):
//
//   block_to_walsh: B_{u,v} = Σ_y (-1)^{⟨v,y⟩} A_{y⊕u, y}   (no 2^{-n})
//   walsh_to_block: A_{x,y} = 2^{-n} Σ_v (-1)^{⟨v,y⟩} B_{x⊕y, v}
//   The 2^{-n} lives in reconstruction, NOT decomposition.
//
// ============================================================================
#pragma once

#include "kernel_calculus.cuh"

namespace topcomp {
namespace doctrine {

using measurement::DenseOp;
using exact::OpBlocks;

// ────────────────────────────────────────────────────────────────────────────
// §1  Operator Face Tags — Every operator exists in one of these forms
// ────────────────────────────────────────────────────────────────────────────

enum class OpFace : int {
    DENSE       = 0,   // Full dim_hyb × dim_hyb matrix
    BLOCK       = 1,   // {O_{x,y}} indexed by I_n × I_n, each dim_top × dim_top
    WEYL        = 2,   // {B_{u,v}} Walsh/Weyl coefficient blocks
    KERNEL      = 3,   // {K_C(x,a)} operator-valued kernel (circuits)
    CONTROLLED  = 4,   // Ĉ_{f,{U_x}} — permutation + fiber payloads
    PERMUTATION = 5,   // P̂_f — pure classical permutation lift
    CHANNEL     = 6,   // E_{C,ρ₀} — induced discrete channel (superoperator)
    SCALAR_ID   = 7,   // c·I — scalar multiple of identity
};

// ────────────────────────────────────────────────────────────────────────────
// §2  ControlledOp — First-class primitive type for Ĉ_{f,{U_x}}
// ────────────────────────────────────────────────────────────────────────────

struct ControlledOp {
    int*     f;           // permutation f: I_n → I_n
    DenseOp* U;           // fiber payloads U[x] ∈ End(X_A^top)
    int      dim_disc;    // |I_n| = 2^n
    int      dim_top;     // dim(X_A^top) = N_θ · N_ρ
    bool     f_is_id;     // true if f = identity (diagonal case)

    __host__ ControlledOp()
        : f(nullptr), U(nullptr), dim_disc(0), dim_top(0), f_is_id(false) {}

    __host__ void init(int dd, int dt) {
        dim_disc = dd;
        dim_top = dt;
        f = new int[dd];
        U = new DenseOp[dd];
        f_is_id = false;
        for (int x = 0; x < dd; ++x) {
            f[x] = x;
            U[x] = DenseOp::identity(dt);
        }
        f_is_id = true;
    }

    __host__ void free() {
        delete[] f; f = nullptr;
        delete[] U; U = nullptr;
    }

    // Build from permutation + payloads
    __host__ static ControlledOp from(const int* perm, const DenseOp* ops,
                                       int dd, int dt) {
        ControlledOp c;
        c.dim_disc = dd;
        c.dim_top = dt;
        c.f = new int[dd];
        c.U = new DenseOp[dd];
        c.f_is_id = true;
        for (int x = 0; x < dd; ++x) {
            c.f[x] = perm[x];
            c.U[x] = ops[x];
            if (perm[x] != x) c.f_is_id = false;
        }
        return c;
    }

    // Pure permutation lift (all U_x = I)
    __host__ static ControlledOp perm(const int* perm, int dd, int dt) {
        ControlledOp c;
        c.dim_disc = dd;
        c.dim_top = dt;
        c.f = new int[dd];
        c.U = new DenseOp[dd];
        c.f_is_id = true;
        for (int x = 0; x < dd; ++x) {
            c.f[x] = perm[x];
            c.U[x] = DenseOp::identity(dt);
            if (perm[x] != x) c.f_is_id = false;
        }
        return c;
    }

    // Diagonal case: f = id, branch-local payloads
    __host__ static ControlledOp diagonal(const DenseOp* ops,
                                           int dd, int dt) {
        ControlledOp c;
        c.dim_disc = dd;
        c.dim_top = dt;
        c.f = new int[dd];
        c.U = new DenseOp[dd];
        c.f_is_id = true;
        for (int x = 0; x < dd; ++x) {
            c.f[x] = x;
            c.U[x] = ops[x];
        }
        return c;
    }

    // ── Algebraic operations ────────────────────────────────────────────

    // Composition: Ĉ_{f,U} · Ĉ_{g,V} = Ĉ_{f∘g, {U_{g(x)}V_x}}
    __host__ ControlledOp compose(const ControlledOp& rhs) const {
        ControlledOp c;
        c.dim_disc = dim_disc;
        c.dim_top = dim_top;
        c.f = new int[dim_disc];
        c.U = new DenseOp[dim_disc];
        c.f_is_id = true;
        for (int x = 0; x < dim_disc; ++x) {
            c.f[x] = f[rhs.f[x]];
            c.U[x] = U[rhs.f[x]] * rhs.U[x];
            if (c.f[x] != x) c.f_is_id = false;
        }
        return c;
    }

    // Inverse: Ĉ^{-1} = Ĉ_{f^{-1}, {U_{f^{-1}(x)}^{-1}}}
    // (only valid if f is bijective and all U_x are unitary)
    __host__ ControlledOp inverse() const {
        ControlledOp c;
        c.dim_disc = dim_disc;
        c.dim_top = dim_top;
        c.f = new int[dim_disc];
        c.U = new DenseOp[dim_disc];
        c.f_is_id = f_is_id;
        for (int x = 0; x < dim_disc; ++x) c.f[f[x]] = x;    // f^{-1}
        for (int x = 0; x < dim_disc; ++x) c.U[x] = U[c.f[x]].dagger();
        return c;
    }

    // ── Face conversions ────────────────────────────────────────────────

    // → Kernel face: K(r,s) = δ_{r,f(s)} U_s
    __host__ OpBlocks to_kernel() const {
        return kc::ctrl_op::build_kernel(f, U, dim_disc, dim_top);
    }

    // → Dense face: full hybrid matrix
    __host__ DenseOp to_dense() const {
        return kc::ctrl_op::build_hybrid_op(f, U, dim_disc, dim_top);
    }

    // → Block face: decompose the dense form
    __host__ OpBlocks to_blocks() const {
        DenseOp D = to_dense();
        OpBlocks B = exact::op::decompose(D, dim_disc, dim_top);
        return B;
    }

    // → Weyl face: Walsh transform of block decomposition
    __host__ OpBlocks to_weyl(int n) const {
        OpBlocks K = to_kernel();
        OpBlocks W = exact::walsh::block_to_walsh(K, n);
        K.free();
        return W;
    }

    // ── Properties ──────────────────────────────────────────────────────

    __host__ bool is_bijective() const {
        bool* seen = new bool[dim_disc]();
        for (int x = 0; x < dim_disc; ++x) {
            if (f[x] < 0 || f[x] >= dim_disc || seen[f[x]]) {
                delete[] seen;
                return false;
            }
            seen[f[x]] = true;
        }
        delete[] seen;
        return true;
    }

    __host__ bool is_pure_perm() const {
        for (int x = 0; x < dim_disc; ++x)
            if ((U[x] - DenseOp::identity(dim_top)).hs_norm2() > 1e-10)
                return false;
        return true;
    }

    __host__ bool is_unitary(double tol = 1e-6) const {
        if (!is_bijective()) return false;
        for (int x = 0; x < dim_disc; ++x) {
            DenseOp I = DenseOp::identity(dim_top);
            if ((U[x] * U[x].dagger() - I).hs_norm2() > tol) return false;
        }
        return true;
    }

    __host__ int nonzero_fiber_count() const {
        int c = 0;
        for (int x = 0; x < dim_disc; ++x)
            if ((U[x] - DenseOp::identity(dim_top)).hs_norm2() > 1e-10)
                ++c;
        return c;
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §3  OpRepr — Tagged Runtime Operator Representation
// ────────────────────────────────────────────────────────────────────────────
//
// Every operator can exist natively as any face.
// The runtime picks the cheapest face for the operation at hand.
// DO NOT reduce all faces to one giant dense matrix.
//
// Conversions are exact bijections, not lossy transforms.
//

struct OpRepr {
    OpFace    face;        // which face is currently materialized
    int       dim_disc;    // 2^n
    int       dim_top;     // N_θ · N_ρ
    int       n_qubits;    // n

    // Storage — only the active face's pointer is non-null
    DenseOp*     dense;      // DENSE: dim_hyb × dim_hyb
    OpBlocks*    blocks;     // BLOCK: {O_{x,y}}
    OpBlocks*    weyl;       // WEYL:  {B_{u,v}}
    OpBlocks*    kernel;     // KERNEL: {K_C(x,a)}
    ControlledOp* ctrl;      // CONTROLLED: Ĉ_{f,{U_x}}
    C64           scalar;    // SCALAR_ID: c·I

    __host__ OpRepr()
        : face(OpFace::DENSE), dim_disc(0), dim_top(0), n_qubits(0),
          dense(nullptr), blocks(nullptr), weyl(nullptr),
          kernel(nullptr), ctrl(nullptr), scalar(0, 0) {}

    __host__ void free() {
        if (dense)  { delete dense;  dense = nullptr; }
        if (blocks) { blocks->free(); delete blocks; blocks = nullptr; }
        if (weyl)   { weyl->free();   delete weyl;   weyl = nullptr; }
        if (kernel) { kernel->free(); delete kernel;  kernel = nullptr; }
        if (ctrl)   { ctrl->free();   delete ctrl;    ctrl = nullptr; }
    }

    // ── Factory methods ─────────────────────────────────────────────────

    __host__ static OpRepr from_dense(const DenseOp& D, int dd, int dt, int n) {
        OpRepr r;
        r.face = OpFace::DENSE;
        r.dim_disc = dd; r.dim_top = dt; r.n_qubits = n;
        r.dense = new DenseOp(D);
        return r;
    }

    __host__ static OpRepr from_blocks(const OpBlocks& B, int n) {
        OpRepr r;
        r.face = OpFace::BLOCK;
        r.dim_disc = B.dim_disc; r.dim_top = B.dim_top; r.n_qubits = n;
        r.blocks = new OpBlocks();
        r.blocks->init(B.dim_disc, B.dim_top);
        for (int x = 0; x < B.dim_disc; ++x)
            for (int y = 0; y < B.dim_disc; ++y)
                r.blocks->at(x, y) = B.at(x, y);
        return r;
    }

    __host__ static OpRepr from_kernel(const OpBlocks& K, int n) {
        OpRepr r;
        r.face = OpFace::KERNEL;
        r.dim_disc = K.dim_disc; r.dim_top = K.dim_top; r.n_qubits = n;
        r.kernel = new OpBlocks();
        r.kernel->init(K.dim_disc, K.dim_top);
        for (int x = 0; x < K.dim_disc; ++x)
            for (int a = 0; a < K.dim_disc; ++a)
                r.kernel->at(x, a) = K.at(x, a);
        return r;
    }

    __host__ static OpRepr from_controlled(const ControlledOp& C, int n) {
        OpRepr r;
        r.face = OpFace::CONTROLLED;
        r.dim_disc = C.dim_disc; r.dim_top = C.dim_top; r.n_qubits = n;
        r.ctrl = new ControlledOp();
        *r.ctrl = ControlledOp::from(C.f, C.U, C.dim_disc, C.dim_top);
        return r;
    }

    __host__ static OpRepr from_perm(const int* f, int dd, int dt, int n) {
        OpRepr r;
        r.face = OpFace::PERMUTATION;
        r.dim_disc = dd; r.dim_top = dt; r.n_qubits = n;
        r.ctrl = new ControlledOp();
        *r.ctrl = ControlledOp::perm(f, dd, dt);
        return r;
    }

    __host__ static OpRepr from_scalar(C64 c, int dd, int dt, int n) {
        OpRepr r;
        r.face = OpFace::SCALAR_ID;
        r.dim_disc = dd; r.dim_top = dt; r.n_qubits = n;
        r.scalar = c;
        return r;
    }

    // ── Face conversions (exact bijections) ─────────────────────────────

    // Ensure DENSE face is available
    __host__ void materialize_dense() {
        if (dense) return;
        dense = new DenseOp(dim_disc * dim_top);
        switch (face) {
            case OpFace::BLOCK: {
                DenseOp D = exact::op::reconstruct(*blocks);
                *dense = D;
                break;
            }
            case OpFace::KERNEL:
            case OpFace::WEYL: {
                // Kernel IS a block decomposition for circuits
                OpBlocks* src = (face == OpFace::KERNEL) ? kernel : weyl;
                DenseOp D = exact::op::reconstruct(*src);
                *dense = D;
                break;
            }
            case OpFace::CONTROLLED:
            case OpFace::PERMUTATION:
                *dense = ctrl->to_dense();
                break;
            case OpFace::SCALAR_ID:
                *dense = DenseOp::identity(dim_disc * dim_top) * scalar;
                break;
            case OpFace::DENSE:
                break;
            case OpFace::CHANNEL:
                break;  // channel is a superoperator, not an operator
        }
    }

    // Ensure BLOCK face is available
    __host__ void materialize_blocks() {
        if (blocks) return;
        blocks = new OpBlocks();
        switch (face) {
            case OpFace::DENSE:
                *blocks = exact::op::decompose(*dense, dim_disc, dim_top);
                break;
            case OpFace::KERNEL:
                // Kernel IS the block decomposition
                blocks->init(dim_disc, dim_top);
                for (int x = 0; x < dim_disc; ++x)
                    for (int a = 0; a < dim_disc; ++a)
                        blocks->at(x, a) = kernel->at(x, a);
                break;
            case OpFace::WEYL:
                *blocks = exact::walsh::walsh_to_block(*weyl, n_qubits);
                break;
            case OpFace::CONTROLLED:
            case OpFace::PERMUTATION:
                *blocks = ctrl->to_blocks();
                break;
            case OpFace::SCALAR_ID: {
                blocks->init_zero(dim_disc, dim_top);
                DenseOp I_top = DenseOp::identity(dim_top);
                for (int x = 0; x < dim_disc; ++x)
                    blocks->at(x, x) = I_top * scalar;
                break;
            }
            default: break;
        }
    }

    // Ensure KERNEL face is available
    __host__ void materialize_kernel() {
        if (kernel) return;
        kernel = new OpBlocks();
        switch (face) {
            case OpFace::CONTROLLED:
            case OpFace::PERMUTATION:
                *kernel = ctrl->to_kernel();
                break;
            case OpFace::BLOCK:
                // Block decomposition IS the kernel decomposition
                kernel->init(dim_disc, dim_top);
                for (int x = 0; x < dim_disc; ++x)
                    for (int a = 0; a < dim_disc; ++a)
                        kernel->at(x, a) = blocks->at(x, a);
                break;
            case OpFace::WEYL:
                *kernel = exact::walsh::walsh_to_block(*weyl, n_qubits);
                break;
            default:
                materialize_dense();
                *kernel = exact::op::decompose(*dense, dim_disc, dim_top);
                break;
        }
    }

    // Ensure WEYL face is available
    __host__ void materialize_weyl() {
        if (weyl) return;
        materialize_kernel();
        weyl = new OpBlocks();
        *weyl = exact::walsh::block_to_walsh(*kernel, n_qubits);
    }

    // ── Structural queries ──────────────────────────────────────────────

    // Count non-zero blocks in the BLOCK face
    __host__ int block_sparsity() {
        materialize_blocks();
        int nonzero = 0;
        for (int x = 0; x < dim_disc; ++x)
            for (int y = 0; y < dim_disc; ++y)
                if (blocks->at(x, y).hs_norm2() > 1e-10) ++nonzero;
        return nonzero;
    }

    // Count non-zero Weyl coefficients
    __host__ int weyl_sparsity() {
        materialize_weyl();
        int nonzero = 0;
        for (int u = 0; u < dim_disc; ++u)
            for (int v = 0; v < dim_disc; ++v)
                if (weyl->at(u, v).hs_norm2() > 1e-10) ++nonzero;
        return nonzero;
    }

    // Is this a controlled operator?
    __host__ bool is_controlled() {
        if (face == OpFace::CONTROLLED || face == OpFace::PERMUTATION)
            return true;
        materialize_kernel();
        // Check if kernel has at most one non-zero entry per column
        for (int a = 0; a < dim_disc; ++a) {
            int count = 0;
            for (int x = 0; x < dim_disc; ++x)
                if (kernel->at(x, a).hs_norm2() > 1e-10) ++count;
            if (count > 1) return false;
        }
        return true;
    }

    // Extract ControlledOp if possible
    __host__ bool extract_controlled(ControlledOp& out) {
        if (face == OpFace::CONTROLLED || face == OpFace::PERMUTATION) {
            out = ControlledOp::from(ctrl->f, ctrl->U, dim_disc, dim_top);
            return true;
        }
        materialize_kernel();
        // Try to extract f and U from kernel
        int* perm = new int[dim_disc];
        DenseOp* ops = new DenseOp[dim_disc];
        for (int a = 0; a < dim_disc; ++a) {
            int found = -1;
            for (int x = 0; x < dim_disc; ++x) {
                if (kernel->at(x, a).hs_norm2() > 1e-10) {
                    if (found >= 0) {
                        delete[] perm; delete[] ops;
                        return false;  // not controlled
                    }
                    found = x;
                }
            }
            if (found < 0) {
                delete[] perm; delete[] ops;
                return false;
            }
            perm[a] = found;
            ops[a] = kernel->at(found, a);
        }
        out = ControlledOp::from(perm, ops, dim_disc, dim_top);
        delete[] perm; delete[] ops;
        return true;
    }

    // ── Cheapest face for common operations ─────────────────────────────

    // Multiplication cost heuristic
    __host__ int multiply_cost() const {
        int d = dim_disc * dim_top;
        switch (face) {
            case OpFace::DENSE:       return d * d * d;
            case OpFace::BLOCK:       return dim_disc * dim_disc * dim_disc
                                             * dim_top * dim_top * dim_top;
            case OpFace::KERNEL:      return dim_disc * dim_disc * dim_disc
                                             * dim_top * dim_top * dim_top;
            case OpFace::CONTROLLED:
            case OpFace::PERMUTATION: return dim_disc * dim_top * dim_top;
            case OpFace::SCALAR_ID:   return 0;
            default: return d * d * d;
        }
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §4  Semantic Contracts — What every subsystem must obey
// ────────────────────────────────────────────────────────────────────────────

// Contract: exact round-trips
// R_vec(D_vec(Ψ)) = Ψ
// R_op(D_op(O)) = O
// walsh_to_block(block_to_walsh(O)) = O
// K(CD) = K(C)K(D)
// K(C*) = K(C)*

struct SemanticContract {
    int dim_disc;
    int dim_top;
    int n_qubits;

    __host__ SemanticContract(int dd, int dt, int n)
        : dim_disc(dd), dim_top(dt), n_qubits(n) {}

    // Verify all exact dictionary round-trips
    __host__ bool verify_vec_roundtrip(const HybridState& psi) const {
        TopologicalState* comp = new TopologicalState[dim_disc];
        for (int x = 0; x < dim_disc; ++x)
            comp[x].init(psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        psi.to_components(comp);
        HybridState recon;
        recon.init(n_qubits, psi.N_theta, psi.N_rho, psi.rho_min, psi.rho_max);
        recon.from_components(comp);
        double err = 0;
        for (int i = 0; i < psi.total; ++i)
            err += (psi.amp[i] - recon.amp[i]).norm2();
        recon.free();
        for (int x = 0; x < dim_disc; ++x) comp[x].free();
        delete[] comp;
        return err < 1e-10;
    }

    // Verify operator block round-trip
    __host__ bool verify_op_roundtrip(const DenseOp& O) const {
        OpBlocks B = exact::op::decompose(O, dim_disc, dim_top);
        DenseOp R = exact::op::reconstruct(B);
        double err = (O - R).hs_norm2();
        B.free();
        return err < 1e-10;
    }

    // Verify Walsh round-trip
    __host__ bool verify_walsh_roundtrip(const DenseOp& O) const {
        OpBlocks B = exact::op::decompose(O, dim_disc, dim_top);
        OpBlocks W = exact::walsh::block_to_walsh(B, n_qubits);
        OpBlocks R = exact::walsh::walsh_to_block(W, n_qubits);
        double err = 0;
        for (int x = 0; x < dim_disc; ++x)
            for (int y = 0; y < dim_disc; ++y)
                err += (B.at(x, y) - R.at(x, y)).hs_norm2();
        B.free(); W.free(); R.free();
        return err < 1e-6;
    }

    // Verify *-algebra isomorphism
    __host__ bool verify_star_algebra(const DenseOp& C, const DenseOp& D,
                                      double tol = 1e-4) const {
        return kc::star_alg::verify_multiplicative(C, D, dim_disc, dim_top, tol)
            && kc::star_alg::verify_star(C, dim_disc, dim_top, tol);
    }

    // Verify channel consistency
    __host__ bool verify_channel(const OpBlocks& K_C, const DenseOp& rho_0,
                                  double tol = 1e-4) const {
        return kc::channel::verify_disc_prob_consistency(K_C, rho_0, tol);
    }
};

// ────────────────────────────────────────────────────────────────────────────
// §5  Face Selection — Runtime picks the cheapest face
// ────────────────────────────────────────────────────────────────────────────

namespace face_select {

// Given two operators, determine best face for multiplication
__host__ inline OpFace best_for_multiply(const OpRepr& A, const OpRepr& B) {
    // SCALAR_ID is always cheapest
    if (A.face == OpFace::SCALAR_ID || B.face == OpFace::SCALAR_ID)
        return OpFace::SCALAR_ID;

    // CONTROLLED composition is O(dd * dt^2) vs O(dd^3 * dt^3) for dense
    if (A.face == OpFace::CONTROLLED && B.face == OpFace::CONTROLLED)
        return OpFace::CONTROLLED;
    if (A.face == OpFace::PERMUTATION && B.face == OpFace::PERMUTATION)
        return OpFace::PERMUTATION;

    // KERNEL composition is O(dd^3 * dt^3) — same as dense
    // but preserves structure better
    if (A.face == OpFace::KERNEL || B.face == OpFace::KERNEL)
        return OpFace::KERNEL;

    return OpFace::DENSE;
}

// Given an operator, determine best face for state application
__host__ inline OpFace best_for_apply(const OpRepr& O) {
    switch (O.face) {
        case OpFace::SCALAR_ID:   return OpFace::SCALAR_ID;
        case OpFace::PERMUTATION: return OpFace::PERMUTATION;
        case OpFace::CONTROLLED:  return OpFace::CONTROLLED;
        default: return OpFace::DENSE;
    }
}

// Determine best face for decomposition/analysis
__host__ inline OpFace best_for_analysis(const OpRepr& O) {
    // Weyl is best for spectral analysis
    // Block is best for probability analysis
    // Kernel is best for circuit analysis
    return OpFace::KERNEL;
}

} // namespace face_select

// ────────────────────────────────────────────────────────────────────────────
// §6  Grand Semantic Verification
// ────────────────────────────────────────────────────────────────────────────

namespace verify {

struct DoctrineVerifyResult {
    // Exact dictionary round-trips
    bool vec_roundtrip;
    bool op_roundtrip;
    bool walsh_roundtrip;

    // *-algebra
    bool star_multiplicative;
    bool star_adjoint;

    // OpRepr face conversions
    bool ctrl_to_dense;
    bool ctrl_to_kernel;
    bool ctrl_to_blocks;
    bool ctrl_to_weyl;
    bool dense_to_blocks_to_dense;
    bool kernel_to_weyl_to_kernel;

    // ControlledOp algebra
    bool ctrl_compose;
    bool ctrl_inverse;
    bool ctrl_factorization;
    bool ctrl_extract;

    // Face selection
    bool face_select_scalar;
    bool face_select_ctrl;
    bool face_select_perm;

    // Structural queries
    bool sparsity_detection;
    bool controlled_detection;

    int total_checks;
    int passed;
};

__host__ inline DoctrineVerifyResult run_all(int n_qubits = 1,
                                              int N_theta = 4,
                                              int N_rho = 3,
                                              double tol = 1e-4) {
    DoctrineVerifyResult r;
    memset(&r, 0, sizeof(r));

    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int dim_hyb = dd * dt;

    SemanticContract contract(dd, dt, n_qubits);

    // ── Build test data ──
    int* f = new int[dd];
    for (int x = 0; x < dd; ++x) f[x] = (x + 1) % dd;

    DenseOp* U_ops = new DenseOp[dd];
    for (int x = 0; x < dd; ++x) {
        U_ops[x] = DenseOp::identity(dt);
        for (int k = 0; k < dt; ++k) {
            double angle = 0.1 * (x + 1) * (k + 1);
            U_ops[x].at(k, k) = C64(cos(angle), sin(angle));
        }
    }

    ControlledOp cop = ControlledOp::from(f, U_ops, dd, dt);

    // Build hybrid state
    HybridState psi;
    psi.init(n_qubits, N_theta, N_rho);
    for (int i = 0; i < psi.total; ++i)
        psi.amp[i] = C64(cos(0.3 * i), sin(0.2 * i));
    double norm = sqrt(psi.norm2());
    for (int i = 0; i < psi.total; ++i)
        psi.amp[i] = psi.amp[i] * C64(1.0 / norm, 0);

    // Build operator
    DenseOp C_hyb = cop.to_dense();
    int* g = new int[dd];
    for (int x = 0; x < dd; ++x) g[x] = (dd - 1 - x);
    DenseOp D_hyb = kc::perm_lift::build_hybrid_op(g, dd, dt);

    // ── Verify dictionary round-trips ──
    r.vec_roundtrip = contract.verify_vec_roundtrip(psi);
    r.op_roundtrip = contract.verify_op_roundtrip(C_hyb);
    r.walsh_roundtrip = contract.verify_walsh_roundtrip(C_hyb);

    // ── Verify *-algebra ──
    r.star_multiplicative = kc::star_alg::verify_multiplicative(
        C_hyb, D_hyb, dd, dt, tol);
    r.star_adjoint = kc::star_alg::verify_star(C_hyb, dd, dt, tol);

    // ── Verify OpRepr face conversions ──
    {
        OpRepr rep = OpRepr::from_controlled(cop, n_qubits);

        // ctrl → dense
        rep.materialize_dense();
        r.ctrl_to_dense = ((*rep.dense) - C_hyb).hs_norm2() < tol;

        // ctrl → kernel
        rep.materialize_kernel();
        OpBlocks K_direct = kc::ctrl_op::build_kernel(f, U_ops, dd, dt);
        double err = 0;
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a)
                err += (rep.kernel->at(x, a) - K_direct.at(x, a)).hs_norm2();
        r.ctrl_to_kernel = err < tol;
        K_direct.free();

        // ctrl → blocks
        rep.materialize_blocks();
        OpBlocks B_direct = exact::op::decompose(C_hyb, dd, dt);
        err = 0;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y)
                err += (rep.blocks->at(x, y) - B_direct.at(x, y)).hs_norm2();
        r.ctrl_to_blocks = err < tol;
        B_direct.free();

        // ctrl → weyl
        rep.materialize_weyl();
        r.ctrl_to_weyl = (rep.weyl != nullptr
                           && rep.weyl->dim_disc == dd);

        rep.free();
    }

    // ── Dense ↔ blocks round-trip ──
    {
        OpRepr rep = OpRepr::from_dense(C_hyb, dd, dt, n_qubits);
        rep.materialize_blocks();
        DenseOp recon = exact::op::reconstruct(*rep.blocks);
        r.dense_to_blocks_to_dense = (C_hyb - recon).hs_norm2() < tol;
        rep.free();
    }

    // ── Kernel ↔ weyl round-trip ──
    {
        OpBlocks K = kc::ctrl_op::build_kernel(f, U_ops, dd, dt);
        OpRepr rep = OpRepr::from_kernel(K, n_qubits);
        rep.materialize_weyl();
        OpBlocks K2 = exact::walsh::walsh_to_block(*rep.weyl, n_qubits);
        double err = 0;
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a)
                err += (K.at(x, a) - K2.at(x, a)).hs_norm2();
        r.kernel_to_weyl_to_kernel = err < tol;
        K.free(); K2.free(); rep.free();
    }

    // ── ControlledOp algebra ──
    {
        int* g2 = new int[dd];
        for (int x = 0; x < dd; ++x) g2[x] = (dd - 1 - x);
        ControlledOp cop2 = ControlledOp::from(g2, U_ops, dd, dt);

        // Composition
        ControlledOp prod = cop.compose(cop2);
        DenseOp lhs = cop.to_dense() * cop2.to_dense();
        DenseOp rhs = prod.to_dense();
        r.ctrl_compose = (lhs - rhs).hs_norm2() < tol;

        // Inverse
        ControlledOp inv = cop.inverse();
        DenseOp Cinv = inv.to_dense();
        DenseOp I = DenseOp::identity(dim_hyb);
        r.ctrl_inverse = (C_hyb * Cinv - I).hs_norm2() < tol;

        // Factorization: Ĉ = P̂_f · Ĉ_{id,U}
        r.ctrl_factorization = kc::ctrl_op::verify_factorization(
            f, U_ops, dd, dt, tol);

        // Extraction
        OpRepr rep = OpRepr::from_dense(C_hyb, dd, dt, n_qubits);
        ControlledOp extracted;
        r.ctrl_extract = rep.extract_controlled(extracted);
        if (r.ctrl_extract) {
            DenseOp Dext = extracted.to_dense();
            r.ctrl_extract = (C_hyb - Dext).hs_norm2() < tol;
            extracted.free();
        }
        rep.free();

        prod.free(); inv.free(); cop2.free();
        delete[] g2;
    }

    // ── Face selection ──
    {
        OpRepr s = OpRepr::from_scalar(C64(2.0, 0), dd, dt, n_qubits);
        OpRepr c = OpRepr::from_controlled(cop, n_qubits);
        OpRepr p = OpRepr::from_perm(f, dd, dt, n_qubits);

        r.face_select_scalar = (face_select::best_for_multiply(s, c) == OpFace::SCALAR_ID);
        r.face_select_ctrl = (face_select::best_for_multiply(c, c) == OpFace::CONTROLLED);
        r.face_select_perm = (face_select::best_for_multiply(p, p) == OpFace::PERMUTATION);

        s.free(); c.free(); p.free();
    }

    // ── Structural queries ──
    {
        // Pure permutation should be sparse in blocks
        OpRepr perm_rep = OpRepr::from_perm(f, dd, dt, n_qubits);
        // Permutation: only dd non-zero blocks (diagonal after reindexing)
        int bs = perm_rep.block_sparsity();
        r.sparsity_detection = (bs <= dd);  // at most dd non-zero blocks
        r.controlled_detection = perm_rep.is_controlled();
        perm_rep.free();
    }

    // ── Count ──
    r.total_checks = 20;
    r.passed = 0;
    bool* checks = reinterpret_cast<bool*>(&r);
    for (int i = 0; i < r.total_checks; ++i)
        if (checks[i]) r.passed++;

    // ── Cleanup ──
    delete[] f;
    delete[] g;
    delete[] U_ops;
    cop.free();
    psi.free();

    return r;
}

} // namespace verify

} // namespace doctrine
} // namespace topcomp
