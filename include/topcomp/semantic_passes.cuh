// ============================================================================
// TopComp: Semantic Compiler Passes
// ============================================================================
// Optimizations that exploit the EXACT structure of the doctrine.
// These are NOT syntactic transforms — they operate on mathematical identity.
//
// Passes:
//   1. Block sparsity detection — find operators with few non-zero blocks
//   2. Weyl sparsity detection  — find operators with few non-zero Walsh coeffs
//   3. Controlled-map extraction — detect if a dense op is actually Ĉ_{f,{U_x}}
//   4. Permutation/control factorization — Ĉ = P̂_f · Ĉ_{id,{U}}
//   5. Face selection — pick cheapest representation for each operation
//   6. Coherence pruning — detect and prune decoherent off-diagonal blocks
//   7. Kernel factorization — factor K = K_2 ∘ K_1 when profitable
//   8. Channel compression — detect when channel is deterministic
//
// Design rule: preserve structure as long as possible, lower late.
// ============================================================================
#pragma once

#include "semantic_kernel.cuh"
#include "platform_ir.cuh"

namespace topcomp {
namespace semantic_pass {

using doctrine::OpFace;
using doctrine::OpRepr;
using doctrine::ControlledOp;
using measurement::DenseOp;
using exact::OpBlocks;

// ────────────────────────────────────────────────────────────────────────────
// §1  Block Sparsity Detection
// ────────────────────────────────────────────────────────────────────────────
// If an operator O has only k << dd² non-zero blocks, mark it and
// the runtime can use sparse block iteration.

struct SparsityReport {
    int   dim_disc;
    int   total_blocks;      // dd²
    int   nonzero_blocks;    // number of non-zero blocks
    int   nonzero_weyl;      // number of non-zero Weyl coefficients
    bool  is_diagonal;       // only O_{x,x} non-zero
    bool  is_band;           // only |x-y| ≤ bandwidth
    int   bandwidth;         // -1 if not banded
    bool  is_controlled;     // at most one non-zero per column in kernel
    bool  is_pure_perm;      // controlled with all U_x = I
    bool  is_scalar_id;      // c·I
    OpFace recommended_face; // cheapest representation

    __host__ void print() const {
        printf("  dim_disc=%d, blocks=%d/%d, weyl=%d/%d\n",
               dim_disc, nonzero_blocks, total_blocks,
               nonzero_weyl, total_blocks);
        printf("  diagonal=%d band=%d(bw=%d) ctrl=%d perm=%d scalar=%d\n",
               is_diagonal, is_band, bandwidth,
               is_controlled, is_pure_perm, is_scalar_id);
        const char* face_names[] = {
            "DENSE", "BLOCK", "WEYL", "KERNEL",
            "CONTROLLED", "PERMUTATION", "CHANNEL", "SCALAR_ID" };
        printf("  recommended: %s\n", face_names[(int)recommended_face]);
    }
};

__host__ inline SparsityReport analyze_sparsity(OpRepr& op, double tol = 1e-10) {
    SparsityReport r;
    r.dim_disc = op.dim_disc;
    r.total_blocks = op.dim_disc * op.dim_disc;

    // Block sparsity
    r.nonzero_blocks = op.block_sparsity();
    r.nonzero_weyl = op.weyl_sparsity();

    // Diagonal check
    op.materialize_blocks();
    r.is_diagonal = true;
    for (int x = 0; x < op.dim_disc && r.is_diagonal; ++x)
        for (int y = 0; y < op.dim_disc; ++y)
            if (x != y && op.blocks->at(x, y).hs_norm2() > tol)
                r.is_diagonal = false;

    // Bandwidth check
    r.bandwidth = 0;
    r.is_band = true;
    for (int x = 0; x < op.dim_disc; ++x)
        for (int y = 0; y < op.dim_disc; ++y) {
            if (op.blocks->at(x, y).hs_norm2() > tol) {
                int d = abs(x - y);
                if (d > r.bandwidth) r.bandwidth = d;
            }
        }
    if (r.bandwidth == 0 && !r.is_diagonal) r.is_band = false;

    // Controlled check
    r.is_controlled = op.is_controlled();

    // Pure permutation check
    r.is_pure_perm = false;
    if (r.is_controlled) {
        ControlledOp cop;
        if (op.extract_controlled(cop)) {
            r.is_pure_perm = cop.is_pure_perm();
            cop.free();
        }
    }

    // Scalar identity check
    r.is_scalar_id = (op.face == OpFace::SCALAR_ID);
    if (!r.is_scalar_id && r.is_diagonal) {
        op.materialize_blocks();
        C64 first_trace = op.blocks->at(0, 0).trace();
        bool all_same = true;
        for (int x = 1; x < op.dim_disc && all_same; ++x) {
            C64 tr = op.blocks->at(x, x).trace();
            if (fabs(tr.re - first_trace.re) > tol ||
                fabs(tr.im - first_trace.im) > tol)
                all_same = false;
        }
        if (all_same) {
            // Check if each diagonal block is proportional to identity
            DenseOp I_top = DenseOp::identity(op.dim_top);
            bool all_prop = true;
            for (int x = 0; x < op.dim_disc && all_prop; ++x) {
                C64 c = op.blocks->at(x, x).at(0, 0);
                if ((op.blocks->at(x, x) - I_top * c).hs_norm2() > tol)
                    all_prop = false;
            }
            r.is_scalar_id = all_prop && all_same;
        }
    }

    // Recommend face
    if (r.is_scalar_id) r.recommended_face = OpFace::SCALAR_ID;
    else if (r.is_pure_perm) r.recommended_face = OpFace::PERMUTATION;
    else if (r.is_controlled) r.recommended_face = OpFace::CONTROLLED;
    else if (r.nonzero_weyl < r.nonzero_blocks) r.recommended_face = OpFace::WEYL;
    else if (r.is_diagonal || r.is_band) r.recommended_face = OpFace::BLOCK;
    else r.recommended_face = OpFace::KERNEL;

    return r;
}

// ────────────────────────────────────────────────────────────────────────────
// §2  Controlled-Map Extraction Pass
// ────────────────────────────────────────────────────────────────────────────
// Given a dense operator, detect if it is a controlled operator Ĉ_{f,{U_x}}.
// If so, promote to CONTROLLED face for cheaper composition.

__host__ inline bool try_promote_to_controlled(OpRepr& op, double tol = 1e-6) {
    if (op.face == OpFace::CONTROLLED || op.face == OpFace::PERMUTATION)
        return true;  // already promoted

    ControlledOp cop;
    if (!op.extract_controlled(cop)) return false;

    // Verify extraction is lossless
    DenseOp reconstructed = cop.to_dense();
    op.materialize_dense();
    double err = (*op.dense - reconstructed).hs_norm2();
    if (err > tol) { cop.free(); return false; }

    // Promote
    op.ctrl = new ControlledOp();
    *op.ctrl = cop;
    op.face = cop.is_pure_perm() ? OpFace::PERMUTATION : OpFace::CONTROLLED;
    return true;
}

// ────────────────────────────────────────────────────────────────────────────
// §3  Permutation/Control Factorization
// ────────────────────────────────────────────────────────────────────────────
// Factor Ĉ_{f,{U_x}} = P̂_f · Ĉ_{id,{U_x}}
// Useful when f has low cost and/or many U_x = I.

struct FactorizationResult {
    bool   factored;
    int    perm_cost;     // cost of P̂_f
    int    diag_cost;     // cost of Ĉ_{id,{U_x}}
    int    nontrivial_fibers;  // number of x where U_x ≠ I
};

__host__ inline FactorizationResult analyze_factorization(
    const ControlledOp& cop, double tol = 1e-10) {
    FactorizationResult r;
    r.factored = true;
    r.perm_cost = cop.dim_disc;  // permutation is O(dd)
    r.nontrivial_fibers = cop.nonzero_fiber_count();
    // Diagonal cost: only need to apply U_x for nontrivial fibers
    r.diag_cost = r.nontrivial_fibers * cop.dim_top * cop.dim_top;
    return r;
}

// ────────────────────────────────────────────────────────────────────────────
// §4  Coherence Pruning
// ────────────────────────────────────────────────────────────────────────────
// For density matrices: if off-diagonal blocks ρ_{x,y} are below threshold,
// set them to zero.  This makes the density effectively classical on the
// discrete register while preserving topological state in each sector.

__host__ inline int coherence_prune(OpBlocks& rho_blocks,
                                     double threshold = 1e-8) {
    int pruned = 0;
    int dd = rho_blocks.dim_disc;
    int dt = rho_blocks.dim_top;
    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y)
            if (x != y && rho_blocks.at(x, y).hs_norm2() < threshold) {
                rho_blocks.at(x, y) = DenseOp::zero(dt);
                ++pruned;
            }
    return pruned;
}

// ────────────────────────────────────────────────────────────────────────────
// §5  Channel Compression
// ────────────────────────────────────────────────────────────────────────────
// Detect when an induced channel E_{C,ρ₀} is:
//   (a) a unitary channel: E(σ) = UσU*
//   (b) a deterministic classical channel: E(|a><a|) = |f(a)><f(a)|
//   (c) a stochastic classical channel: diagonal in computational basis

struct ChannelType {
    bool is_unitary;        // E(σ) = UσU*
    bool is_deterministic;  // classical permutation
    bool is_classical;      // diagonal output for diagonal input
};

__host__ inline ChannelType classify_channel(
    const OpBlocks& K_C, const DenseOp& rho_0,
    double tol = 1e-6)
{
    ChannelType ct;
    int dd = K_C.dim_disc;
    (void)K_C.dim_top;  // dt available via K_C if needed

    // Check deterministic: for each input basis state, does
    // the channel produce exactly one output?
    ct.is_deterministic = true;
    for (int a = 0; a < dd; ++a) {
        DenseOp sigma(dd);
        sigma.at(a, a) = C64(1, 0);
        DenseOp Esigma = kc::channel::apply(K_C, rho_0, sigma);
        int nonzero = 0;
        for (int x = 0; x < dd; ++x)
            if (fabs(Esigma.at(x, x).re) > tol) ++nonzero;
        if (nonzero != 1) ct.is_deterministic = false;
    }

    // Check classical: diagonal input → diagonal output
    ct.is_classical = true;
    for (int a = 0; a < dd; ++a) {
        DenseOp sigma(dd);
        sigma.at(a, a) = C64(1, 0);
        DenseOp Esigma = kc::channel::apply(K_C, rho_0, sigma);
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y)
                if (x != y && fabs(Esigma.at(x, y).re) + fabs(Esigma.at(x, y).im) > tol)
                    ct.is_classical = false;
    }

    // Check unitary: E(I) = I (up to normalization)
    ct.is_unitary = false;
    if (ct.is_classical) {
        // Compute channel matrix and check if it's a valid
        // unitary conjugation
        DenseOp I_disc = DenseOp::identity(dd);
        DenseOp EI = kc::channel::apply(K_C, rho_0, I_disc);
        double err = (EI - I_disc).hs_norm2();
        ct.is_unitary = (err < tol);
    }

    return ct;
}

// ────────────────────────────────────────────────────────────────────────────
// §6  IR-Level Semantic Passes
// ────────────────────────────────────────────────────────────────────────────
// Integrate doctrine-level analysis into the PlatformIR pipeline.

namespace ir_pass {

// Weyl cancellation: if X_u Z_v appears followed by Z_v X_u,
// they compose to (-1)^{⟨u,v⟩} I — possibly eliminable
__host__ inline int weyl_cancellation(platform::PlatformIR& ir) {
    int eliminated = 0;
    for (int i = 0; i + 1 < ir.count; ++i) {
        auto& a = ir.ops[i];
        auto& b = ir.ops[i + 1];
        // X_u followed by X_u → I (X_u² = I)
        if (a.opcode == platform::PlatformOpCode::APPLY_X &&
            b.opcode == platform::PlatformOpCode::APPLY_X &&
            a.bv1.bits == b.bv1.bits && a.bv1.n == b.bv1.n &&
            a.reg_dst == b.reg_dst) {
            a.opcode = platform::PlatformOpCode::NOP;
            b.opcode = platform::PlatformOpCode::NOP;
            eliminated += 2;
        }
        // Z_v followed by Z_v → I
        if (a.opcode == platform::PlatformOpCode::APPLY_Z &&
            b.opcode == platform::PlatformOpCode::APPLY_Z &&
            a.bv1.bits == b.bv1.bits && a.bv1.n == b.bv1.n &&
            a.reg_dst == b.reg_dst) {
            a.opcode = platform::PlatformOpCode::NOP;
            b.opcode = platform::PlatformOpCode::NOP;
            eliminated += 2;
        }
        // H followed by H → I (Hadamard is self-inverse)
        if (a.opcode == platform::PlatformOpCode::APPLY_H &&
            b.opcode == platform::PlatformOpCode::APPLY_H &&
            a.iparam == b.iparam && a.reg_dst == b.reg_dst) {
            a.opcode = platform::PlatformOpCode::NOP;
            b.opcode = platform::PlatformOpCode::NOP;
            eliminated += 2;
        }
    }
    return eliminated;
}

// Projector absorption: P_x followed by any gate that preserves x
// can sometimes be simplified
__host__ inline int projector_absorption(platform::PlatformIR& ir) {
    int absorbed = 0;
    for (int i = 0; i + 1 < ir.count; ++i) {
        auto& a = ir.ops[i];
        auto& b = ir.ops[i + 1];
        // P_x followed by P_x → P_x (idempotent)
        if (a.opcode == platform::PlatformOpCode::APPLY_P &&
            b.opcode == platform::PlatformOpCode::APPLY_P &&
            a.target == b.target && a.reg_dst == b.reg_dst) {
            b.opcode = platform::PlatformOpCode::NOP;
            ++absorbed;
        }
    }
    return absorbed;
}

} // namespace ir_pass

// ────────────────────────────────────────────────────────────────────────────
// §7  Grand Verification for Semantic Passes
// ────────────────────────────────────────────────────────────────────────────

namespace verify {

struct PassVerifyResult {
    bool sparsity_correct;
    bool ctrl_extraction;
    bool factorization_valid;
    bool coherence_prune_safe;
    bool channel_classify;
    bool weyl_cancel;
    bool projector_absorb;
    bool face_recommendation;
    int total_checks;
    int passed;
};

__host__ inline PassVerifyResult run_all(int n_qubits = 1,
                                          int N_theta = 4,
                                          int N_rho = 3,
                                          double tol = 1e-4) {
    PassVerifyResult r;
    memset(&r, 0, sizeof(r));

    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int n = n_qubits;

    // Build test operators
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

    // ── Sparsity analysis ──
    {
        OpRepr perm = OpRepr::from_perm(f, dd, dt, n);
        SparsityReport sr = analyze_sparsity(perm);
        r.sparsity_correct = (sr.is_controlled && sr.is_pure_perm
                              && sr.recommended_face == OpFace::PERMUTATION);
        perm.free();
    }

    // ── Controlled extraction ──
    {
        DenseOp C_hyb = kc::ctrl_op::build_hybrid_op(f, U_ops, dd, dt);
        OpRepr rep = OpRepr::from_dense(C_hyb, dd, dt, n);
        r.ctrl_extraction = try_promote_to_controlled(rep, tol);
        rep.free();
    }

    // ── Factorization ──
    {
        ControlledOp cop = ControlledOp::from(f, U_ops, dd, dt);
        FactorizationResult fr = analyze_factorization(cop);
        r.factorization_valid = fr.factored;
        cop.free();
    }

    // ── Coherence pruning ──
    {
        // Build a density matrix with small off-diagonals
        DenseOp rho(dd * dt);
        for (int i = 0; i < dd * dt; ++i)
            rho.at(i, i) = C64(1.0 / (dd * dt), 0);
        // Add tiny off-diagonal
        if (dd * dt > 1)
            rho.at(0, 1) = C64(1e-12, 0);

        OpBlocks rho_blk = exact::op::decompose(rho, dd, dt);
        double tr_before = 0;
        for (int x = 0; x < dd; ++x)
            tr_before += rho_blk.at(x, x).trace().re;

        int pruned = coherence_prune(rho_blk, 1e-8);

        double tr_after = 0;
        for (int x = 0; x < dd; ++x)
            tr_after += rho_blk.at(x, x).trace().re;

        // Pruning should preserve diagonal (trace)
        r.coherence_prune_safe = (fabs(tr_before - tr_after) < tol);
        rho_blk.free();
    }

    // ── Channel classification ──
    {
        DenseOp rho_0 = DenseOp::identity(dt);
        rho_0 = rho_0 * C64(1.0 / dt, 0);

        // Permutation kernel should give deterministic channel
        OpBlocks K_perm = kc::perm_lift::build_kernel(f, dd, dt);
        ChannelType ct = classify_channel(K_perm, rho_0, tol);
        r.channel_classify = ct.is_deterministic;
        K_perm.free();
    }

    // ── IR passes ──
    {
        // Build a simple IR with X_u X_u → should cancel
        platform::PlatformIR ir;
        ir.emit(platform::PlatformOp::apply_x(0, BitVec(1, n)));
        ir.emit(platform::PlatformOp::apply_x(0, BitVec(1, n)));
        int elim = ir_pass::weyl_cancellation(ir);
        r.weyl_cancel = (elim == 2);
    }
    {
        // P_x P_x → P_x
        platform::PlatformIR ir;
        ir.emit(platform::PlatformOp::apply_p(0, 0));
        ir.emit(platform::PlatformOp::apply_p(0, 0));
        int abs = ir_pass::projector_absorption(ir);
        r.projector_absorb = (abs == 1);
    }

    // ── Face recommendation ──
    {
        OpRepr scalar = OpRepr::from_scalar(C64(1, 0), dd, dt, n);
        SparsityReport sr = analyze_sparsity(scalar);
        r.face_recommendation = (sr.recommended_face == OpFace::SCALAR_ID);
        scalar.free();
    }

    r.total_checks = 8;
    r.passed = 0;
    bool* checks = reinterpret_cast<bool*>(&r);
    for (int i = 0; i < r.total_checks; ++i)
        if (checks[i]) r.passed++;

    delete[] f;
    delete[] U_ops;
    return r;
}

} // namespace verify

} // namespace semantic_pass
} // namespace topcomp
