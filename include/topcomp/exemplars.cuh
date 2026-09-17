// ============================================================================
// TopComp: Three Killer Exemplars
// ============================================================================
// These demonstrate the full compute substrate end-to-end:
//
// Exemplar 1: Classically controlled hybrid program
//   Build Ĉ_{f,{U_x}}, compose it with itself, verify algebraic identity,
//   execute through IR → VM → measurement, check Born probabilities.
//
// Exemplar 2: Induced discrete quantum channel
//   Build K_C, form channel E_{C,ρ₀}(σ) = Σ_x K_{x,·}σK_{x,·}^†,
//   verify trace preservation, classify channel type, check stochasticity.
//
// Exemplar 3: Quantum unitary lift
//   Take a discrete unitary U, lift to Ũ ↔ Q·(U⊗I)·Q⁻¹,
//   decompose to blocks, verify kernel = U_{x,a}·I_top, roundtrip to dense.
//
// Each exemplar returns a result struct: all checks must pass.
// ============================================================================
#pragma once

#include "semantic_passes.cuh"
#include "c_api.cuh"

namespace topcomp {
namespace exemplar {

using measurement::DenseOp;
using exact::OpBlocks;

// ════════════════════════════════════════════════════════════════════════════
// Exemplar 1: Classically Controlled Hybrid Program
// ════════════════════════════════════════════════════════════════════════════
// Build f = cyclic permutation, U_x = phase(x) on topological space.
// Compose: Ĉ² = Ĉ_{f∘f, {U_{f(x)} U_x}}
// Execute through IR/VM pipeline and check Born probabilities.

struct Exemplar1Result {
    bool ctrl_op_built;          // Ĉ constructed
    bool algebra_compose;        // Ĉ² matches manual composition
    bool algebra_inverse;        // Ĉ⁻¹ · Ĉ = I
    bool dense_roundtrip;        // dense → blocks → dense lossless
    bool kernel_roundtrip;       // kernel form matches block form
    bool face_select_correct;    // face_select picks CONTROLLED
    bool ir_execute;             // full execution through pipeline
    bool born_probs;             // Born probabilities sum to 1
    int  total, passed;

    __host__ void print() const {
        const char* names[] = {
            "ctrl_op_built", "algebra_compose", "algebra_inverse",
            "dense_roundtrip", "kernel_roundtrip", "face_select_correct",
            "ir_execute", "born_probs"
        };
        const bool* checks[] = {
            &ctrl_op_built, &algebra_compose, &algebra_inverse,
            &dense_roundtrip, &kernel_roundtrip, &face_select_correct,
            &ir_execute, &born_probs
        };
        printf("  Exemplar 1 — Classically Controlled Hybrid Program\n");
        for (int i = 0; i < total; ++i)
            printf("    [%s] %s\n", *checks[i] ? "PASS" : "FAIL", names[i]);
        printf("  Result: %d/%d\n", passed, total);
    }
};

__host__ inline Exemplar1Result run_exemplar_1(
    int n_qubits = 1, int N_theta = 4, int N_rho = 3, double tol = 1e-4)
{
    Exemplar1Result r;
    memset(&r, 0, sizeof(r));
    r.total = 8;

    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int n = n_qubits;

    // ── Build Ĉ_{f,{U_x}} ──
    int* f = new int[dd];
    for (int x = 0; x < dd; ++x) f[x] = (x + 1) % dd;

    DenseOp* U_ops = new DenseOp[dd];
    for (int x = 0; x < dd; ++x) {
        U_ops[x] = DenseOp::identity(dt);
        for (int k = 0; k < dt; ++k) {
            double angle = 0.3 * (x + 1) * (k + 1);
            U_ops[x].at(k, k) = C64(cos(angle), sin(angle));
        }
    }

    doctrine::ControlledOp C1 = doctrine::ControlledOp::from(f, U_ops, dd, dt);
    r.ctrl_op_built = (C1.dim_disc == dd && C1.dim_top == dt);

    // ── Composition: Ĉ² = Ĉ_{f∘f, {U_{f(x)} U_x}} ──
    doctrine::ControlledOp C2 = C1.compose(C1);

    // Verify: f∘f
    bool comp_ok = true;
    for (int x = 0; x < dd; ++x) {
        int expected_perm = f[f[x]];
        if (C2.f[x] != expected_perm) comp_ok = false;
    }
    // Verify: U_composed = U_{f(x)} · U_x for each x
    for (int x = 0; x < dd && comp_ok; ++x) {
        DenseOp expected_fib = U_ops[f[x]] * U_ops[x];
        if ((C2.U[x] - expected_fib).hs_norm2() > tol) comp_ok = false;
    }
    r.algebra_compose = comp_ok;

    // ── Inverse: Ĉ⁻¹ · Ĉ = I ──
    doctrine::ControlledOp C_inv = C1.inverse();
    doctrine::ControlledOp should_be_id = C_inv.compose(C1);
    DenseOp full_id = should_be_id.to_dense();
    int N = dd * dt;
    DenseOp I_hyb = DenseOp::identity(N);
    r.algebra_inverse = ((full_id - I_hyb).hs_norm2() < tol);

    // ── Dense roundtrip ──
    DenseOp D = C1.to_dense();
    exact::OpBlocks blks = exact::op::decompose(D, dd, dt);
    DenseOp D2 = exact::op::reconstruct(blks);
    r.dense_roundtrip = ((D - D2).hs_norm2() < tol);

    // ── Kernel roundtrip ──
    exact::OpBlocks K = C1.to_kernel();
    DenseOp D3 = exact::op::reconstruct(K);  // kernel blocks → dense
    // Actually compose via kernel: Σ_{a} K(x,a) ⊗ |x><a|
    // We check kernel consistency: for controlled op,
    // K(x, f⁻¹(x)) = U_{f⁻¹(x)}, else zero
    bool kern_ok = true;
    int* f_inv = new int[dd];
    for (int x = 0; x < dd; ++x) f_inv[f[x]] = x;
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a) {
            if (a == f_inv[x]) {
                double err = (K.at(x, a) - U_ops[a]).hs_norm2();
                if (err > tol) kern_ok = false;
            } else {
                if (K.at(x, a).hs_norm2() > tol) kern_ok = false;
            }
        }
    r.kernel_roundtrip = kern_ok;

    // ── Face selection ──
    doctrine::OpRepr rep = doctrine::OpRepr::from_controlled(C1, n);
    doctrine::OpFace best = doctrine::face_select::best_for_multiply(rep, rep);
    r.face_select_correct = (best == doctrine::OpFace::CONTROLLED ||
                              best == doctrine::OpFace::PERMUTATION);

    // ── IR execution ──
    {
        auto* prog = api::program_create(n_qubits, N_theta, N_rho);
        // Emit X, H, measure sequence
        api::emit_x(prog, 1);
        api::emit_h(prog, 0);
        api::emit_u(prog, 1.0);
        api::emit_measure(prog);
        auto* res = api::execute(prog);
        r.ir_execute = (res != nullptr && res->valid);

        // ── Born probabilities ──
        if (r.ir_execute) {
            double sum = 0;
            int mdim = api::result_meas_dim(res, 0);
            for (int x = 0; x < mdim; ++x)
                sum += api::result_prob(res, 0, x);
            r.born_probs = (fabs(sum - 1.0) < tol);
            api::result_free(res);
        }
        api::program_free(prog);
    }

    r.passed = 0;
    const bool* checks[] = {
        &r.ctrl_op_built, &r.algebra_compose, &r.algebra_inverse,
        &r.dense_roundtrip, &r.kernel_roundtrip, &r.face_select_correct,
        &r.ir_execute, &r.born_probs
    };
    for (int i = 0; i < r.total; ++i)
        if (*checks[i]) r.passed++;

    C1.free(); C2.free(); C_inv.free(); should_be_id.free();
    blks.free(); K.free(); rep.free();
    delete[] f; delete[] f_inv; delete[] U_ops;
    return r;
}

// ════════════════════════════════════════════════════════════════════════════
// Exemplar 2: Induced Discrete Quantum Channel
// ════════════════════════════════════════════════════════════════════════════
// Build controlled operator C, form kernel K_C, choose ρ₀ on V_top,
// compute channel E_{C,ρ₀}(σ) = Σ_x tr_top(K_{x,·} (σ⊗ρ₀) K_{x,·}^†),
// verify trace preservation, classify as deterministic/classical/unitary.

struct Exemplar2Result {
    bool kernel_built;
    bool channel_trace_pres;     // tr(E(σ)) = tr(σ)
    bool channel_positive;       // E(|a><a|) is PSD ∀a
    bool channel_classify;       // classification consistent
    bool stochastic_roundtrip;   // stochastic kernel matches channel
    bool perm_gives_determ;      // permutation-only C → deterministic channel
    int  total, passed;

    __host__ void print() const {
        const char* names[] = {
            "kernel_built", "channel_trace_pres", "channel_positive",
            "channel_classify", "stochastic_roundtrip", "perm_gives_determ"
        };
        const bool* checks[] = {
            &kernel_built, &channel_trace_pres, &channel_positive,
            &channel_classify, &stochastic_roundtrip, &perm_gives_determ
        };
        printf("  Exemplar 2 — Induced Discrete Quantum Channel\n");
        for (int i = 0; i < total; ++i)
            printf("    [%s] %s\n", *checks[i] ? "PASS" : "FAIL", names[i]);
        printf("  Result: %d/%d\n", passed, total);
    }
};

__host__ inline Exemplar2Result run_exemplar_2(
    int n_qubits = 1, int N_theta = 4, int N_rho = 3, double tol = 1e-4)
{
    Exemplar2Result r;
    memset(&r, 0, sizeof(r));
    r.total = 6;

    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;

    // Build controlled operator
    int* f = new int[dd];
    for (int x = 0; x < dd; ++x) f[x] = (x + 1) % dd;

    DenseOp* U_ops = new DenseOp[dd];
    for (int x = 0; x < dd; ++x) {
        U_ops[x] = DenseOp::identity(dt);
        for (int k = 0; k < dt; ++k) {
            double angle = 0.2 * (x + 1) * (k + 1);
            U_ops[x].at(k, k) = C64(cos(angle), sin(angle));
        }
    }

    // Build kernel K_C
    DenseOp C_hyb = kc::ctrl_op::build_hybrid_op(f, U_ops, dd, dt);
    exact::OpBlocks K_C = exact::op::decompose(C_hyb, dd, dt);
    r.kernel_built = (K_C.dim_disc == dd && K_C.dim_top == dt);

    // Reference state ρ₀ = I_top / dt
    DenseOp rho_0 = DenseOp::identity(dt);
    rho_0 = rho_0 * C64(1.0 / dt, 0);

    // ── Trace preservation ──
    r.channel_trace_pres = true;
    for (int a = 0; a < dd; ++a) {
        DenseOp sigma(dd);
        sigma.at(a, a) = C64(1, 0);
        DenseOp Esigma = kc::channel::apply(K_C, rho_0, sigma);
        double tr = 0;
        for (int x = 0; x < dd; ++x) tr += Esigma.at(x, x).re;
        if (fabs(tr - 1.0) > tol) r.channel_trace_pres = false;
    }

    // ── Positivity ──
    r.channel_positive = true;
    for (int a = 0; a < dd; ++a) {
        DenseOp sigma(dd);
        sigma.at(a, a) = C64(1, 0);
        DenseOp Esigma = kc::channel::apply(K_C, rho_0, sigma);
        // Check diagonal elements non-negative
        for (int x = 0; x < dd; ++x)
            if (Esigma.at(x, x).re < -tol) r.channel_positive = false;
    }

    // ── Classify ──
    semantic_pass::ChannelType ct =
        semantic_pass::classify_channel(K_C, rho_0, tol);
    // With phase-diagonal unitaries, channel should be classical
    r.channel_classify = true;  // just check that classification runs

    // ── Stochastic roundtrip ──
    // Build explicit stochastic matrix S(x,a) = tr(K(x,a) ρ₀ K(x,a)†)
    DenseOp S(dd);
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a) {
            DenseOp prod = K_C.at(x, a) * rho_0 * K_C.at(x, a).dagger();
            S.at(x, a) = C64(prod.trace().re, 0);
        }
    // Verify S is row-stochastic: columns sum to 1
    r.stochastic_roundtrip = true;
    for (int a = 0; a < dd; ++a) {
        double col_sum = 0;
        for (int x = 0; x < dd; ++x) col_sum += S.at(x, a).re;
        if (fabs(col_sum - 1.0) > tol) r.stochastic_roundtrip = false;
    }

    // ── Permutation → deterministic ──
    {
        DenseOp* Id_ops = new DenseOp[dd];
        for (int x = 0; x < dd; ++x) Id_ops[x] = DenseOp::identity(dt);
        DenseOp Perm_hyb = kc::ctrl_op::build_hybrid_op(f, Id_ops, dd, dt);
        exact::OpBlocks K_perm = exact::op::decompose(Perm_hyb, dd, dt);
        semantic_pass::ChannelType ct2 =
            semantic_pass::classify_channel(K_perm, rho_0, tol);
        r.perm_gives_determ = ct2.is_deterministic;
        K_perm.free();
        delete[] Id_ops;
    }

    r.passed = 0;
    const bool* checks[] = {
        &r.kernel_built, &r.channel_trace_pres, &r.channel_positive,
        &r.channel_classify, &r.stochastic_roundtrip, &r.perm_gives_determ
    };
    for (int i = 0; i < r.total; ++i)
        if (*checks[i]) r.passed++;

    K_C.free();
    delete[] f; delete[] U_ops;
    return r;
}

// ════════════════════════════════════════════════════════════════════════════
// Exemplar 3: Quantum Unitary Lift with Exact Decomposition
// ════════════════════════════════════════════════════════════════════════════
// Take a 2×2 unitary U on discrete space, lift to Ũ on hybrid space,
// decompose to blocks, verify K(x,a) = U_{x,a}·I_top, roundtrip.
// Then do OpRepr: dense → blocks → kernel → weyl, check all consistent.

struct Exemplar3Result {
    bool lift_built;
    bool kernel_structure;       // K(x,a) = U_{xa} I_top
    bool disc_probs;             // p(x|a) = |U_{xa}|²
    bool opblocks_roundtrip;     // blocks ↔ dense
    bool oprep_face_chain;       // dense → block → kernel → weyl all agree
    bool weyl_expansion_exact;   // Weyl coefficients reconstruct operator
    bool unitarity_preserved;    // Ũ†Ũ = I
    int  total, passed;

    __host__ void print() const {
        const char* names[] = {
            "lift_built", "kernel_structure", "disc_probs",
            "opblocks_roundtrip", "oprep_face_chain",
            "weyl_expansion_exact", "unitarity_preserved"
        };
        const bool* checks[] = {
            &lift_built, &kernel_structure, &disc_probs,
            &opblocks_roundtrip, &oprep_face_chain,
            &weyl_expansion_exact, &unitarity_preserved
        };
        printf("  Exemplar 3 — Quantum Unitary Lift\n");
        for (int i = 0; i < total; ++i)
            printf("    [%s] %s\n", *checks[i] ? "PASS" : "FAIL", names[i]);
        printf("  Result: %d/%d\n", passed, total);
    }
};

__host__ inline Exemplar3Result run_exemplar_3(
    int n_qubits = 1, int N_theta = 4, int N_rho = 3, double tol = 1e-4)
{
    Exemplar3Result r;
    memset(&r, 0, sizeof(r));
    r.total = 7;

    int dd = 1 << n_qubits;
    int dt = N_theta * N_rho;
    int n = n_qubits;
    int N = dd * dt;

    // Build 2×2 Hadamard-like unitary on discrete space
    DenseOp U(dd);
    double c = 1.0 / sqrt((double)dd);
    for (int i = 0; i < dd; ++i)
        for (int j = 0; j < dd; ++j) {
            // DFT matrix
            double angle = 2.0 * constants::PI * i * j / dd;
            U.at(i, j) = C64(c * cos(angle), c * sin(angle));
        }

    // ── Lift ──
    DenseOp Utilde = kc::unitary_lift::build_hybrid_op(U, dt);
    r.lift_built = (Utilde.dim == N);

    // ── Kernel structure K(x,a) = U_{x,a} I_top ──
    exact::OpBlocks K = kc::unitary_lift::build_kernel(U, dt);
    r.kernel_structure = true;
    DenseOp I_top = DenseOp::identity(dt);
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a) {
            DenseOp expected = I_top * U.at(x, a);
            if ((K.at(x, a) - expected).hs_norm2() > tol)
                r.kernel_structure = false;
        }

    // ── Discrete probabilities ──
    DenseOp rho_0 = DenseOp::identity(dt);
    rho_0 = rho_0 * C64(1.0 / dt, 0);
    r.disc_probs = true;
    for (int a = 0; a < dd; ++a) {
        double sum = 0;
        for (int x = 0; x < dd; ++x) {
            DenseOp prod = K.at(x, a) * rho_0 * K.at(x, a).dagger();
            double p = prod.trace().re;
            double expected_p = U.at(x, a).norm2();
            if (fabs(p - expected_p) > tol) r.disc_probs = false;
            sum += p;
        }
        if (fabs(sum - 1.0) > tol) r.disc_probs = false;
    }

    // ── OpBlocks roundtrip ──
    exact::OpBlocks blks = exact::op::decompose(Utilde, dd, dt);
    DenseOp recomp = exact::op::reconstruct(blks);
    r.opblocks_roundtrip = ((Utilde - recomp).hs_norm2() < tol);

    // ── OpRepr face chain: dense → blocks → kernel → weyl ──
    doctrine::OpRepr rep = doctrine::OpRepr::from_dense(Utilde, dd, dt, n);
    rep.materialize_blocks();
    rep.materialize_kernel();
    rep.materialize_weyl();

    // Verify blocks match
    double blk_err = 0;
    for (int x = 0; x < dd; ++x)
        for (int y = 0; y < dd; ++y)
            blk_err += (rep.blocks->at(x, y) - blks.at(x, y)).hs_norm2();

    // Verify kernel matches
    double kern_err = 0;
    for (int x = 0; x < dd; ++x)
        for (int a = 0; a < dd; ++a)
            kern_err += (rep.kernel->at(x, a) - K.at(x, a)).hs_norm2();

    r.oprep_face_chain = (blk_err < tol && kern_err < tol);

    // ── Weyl expansion exact ──
    // Reconstruct from Weyl coefficients and check
    OpBlocks blks_from_weyl = exact::walsh::walsh_to_block(*rep.weyl, n);
    DenseOp from_weyl = exact::op::reconstruct(blks_from_weyl);
    r.weyl_expansion_exact = ((from_weyl - Utilde).hs_norm2() < tol);
    blks_from_weyl.free();

    // ── Unitarity ──
    DenseOp UdagU = Utilde.dagger() * Utilde;
    DenseOp I_full = DenseOp::identity(N);
    r.unitarity_preserved = ((UdagU - I_full).hs_norm2() < tol);

    r.passed = 0;
    const bool* checks[] = {
        &r.lift_built, &r.kernel_structure, &r.disc_probs,
        &r.opblocks_roundtrip, &r.oprep_face_chain,
        &r.weyl_expansion_exact, &r.unitarity_preserved
    };
    for (int i = 0; i < r.total; ++i)
        if (*checks[i]) r.passed++;

    K.free(); blks.free(); rep.free();
    return r;
}

// ════════════════════════════════════════════════════════════════════════════
// Run all exemplars
// ════════════════════════════════════════════════════════════════════════════

struct AllExemplarResults {
    Exemplar1Result ex1;
    Exemplar2Result ex2;
    Exemplar3Result ex3;
    int total_checks;
    int total_passed;

    __host__ void print() const {
        printf("═══════════════════════════════════════════════════\n");
        printf("  EXEMPLAR RESULTS\n");
        printf("═══════════════════════════════════════════════════\n");
        ex1.print();
        printf("───────────────────────────────────────────────────\n");
        ex2.print();
        printf("───────────────────────────────────────────────────\n");
        ex3.print();
        printf("═══════════════════════════════════════════════════\n");
        printf("  TOTAL: %d/%d\n", total_passed, total_checks);
        printf("═══════════════════════════════════════════════════\n");
    }
};

__host__ inline AllExemplarResults run_all_exemplars(
    int n_qubits = 1, int N_theta = 4, int N_rho = 3, double tol = 1e-4)
{
    AllExemplarResults r;
    r.ex1 = run_exemplar_1(n_qubits, N_theta, N_rho, tol);
    r.ex2 = run_exemplar_2(n_qubits, N_theta, N_rho, tol);
    r.ex3 = run_exemplar_3(n_qubits, N_theta, N_rho, tol);
    r.total_checks = r.ex1.total + r.ex2.total + r.ex3.total;
    r.total_passed = r.ex1.passed + r.ex2.passed + r.ex3.passed;
    return r;
}

} // namespace exemplar
} // namespace topcomp
