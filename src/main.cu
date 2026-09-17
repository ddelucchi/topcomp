// ============================================================================
// TopComp: Verification & Demonstration Driver
// ============================================================================
// Exercises the full compute stack end-to-end:
//   - Mathematical constants, mirror group, phase space
//   - Hilbert space construction, gate application
//   - Cocycle/UFE verification, statistical PRNG
//   - HES machine, quantum/topological/Weyl interference
//   - Deutsch-Jozsa, structure detection, GPU kernels
//   - Witt dual numbers, Lie/representation theory
//   - Compiler IR, structure finder, noncommutative dynamics
//   - Multi-boundary states, hybrid information analytics
// ============================================================================

#include "topcomp/benchmark.cuh"
#include "topcomp/structure_finder.cuh"
#include "topcomp/measurement.cuh"
#include "topcomp/transport.cuh"
#include "topcomp/kernels.cuh"
#include "topcomp/hybrid_info.cuh"
#include "topcomp/mixed.cuh"
#include "topcomp/quantum_algebra.cuh"
#include "topcomp/fibered_memory.cuh"
#include "topcomp/regime_planner.cuh"
#include "topcomp/spectral_transport.cuh"
#include "topcomp/persistent_format.cuh"
#include "topcomp/exact_decomposition.cuh"
#include "topcomp/kernel_calculus.cuh"
#include "topcomp/exemplars.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace topcomp;
using constants::PHI;
using constants::ELL;
using constants::PI;

static void verify_constants() {
    double phi_err = fabs(PHI * PHI - (PHI + 1.0));
    double ell_err = fabs(exp(ELL) - PHI);
    C64 cp = constants::c_phi();
    double cp_re_err = fabs(cp.re - ELL);
    double cp_im_err = fabs(cp.im + PI / 2.0);
    printf("constants: phi^2-(phi+1)=%.2e  e^ell-phi=%.2e  c_phi_err=(%.2e,%.2e)\n",
           phi_err, ell_err, cp_re_err, cp_im_err);
}

static void verify_mir_group() {
    MirElement g = mir::g_phi(1.0);
    MirElement h = mir::U_phi();
    MirElement k = mir::U_pi();

    MirElement gi = g.inverse();
    MirElement ggi = g.star(gi);
    double inv_err = ggi.c.abs();

    MirElement lhs = g.star(h).star(k);
    MirElement rhs = g.star(h.star(k));
    double assoc_err = (lhs.c - rhs.c).abs();

    MirElement mirror = mir::J_lambda(1.0);
    MirElement mirror2 = mirror.star(mirror);
    double J2_err = mirror2.c.abs();
    int J2_s = mirror2.s;

    printf("mir_group: inv_err=%.2e  assoc_err=%.2e  J^2=(s=%d,|c|=%.2e)\n",
           inv_err, assoc_err, J2_s, J2_err);
}

static void verify_arrows() {
    MirElement r = ArrowAlphabet::ev_right();
    MirElement l = ArrowAlphabet::ev_left();
    MirElement rl = r.star(l);
    double rl_err = rl.c.abs();  // should be ~0 (identity)
    int rl_s = rl.s;             // should be 1

    MirElement result = MirElement::identity();
    result = result.star(r).star(l).star(r);
    printf("arrows: ev(->)*ev(<-)=(s=%d,|c|=%.2e)  ev(->.<-.->)=(s=%d,c=(%.6f,%.6f))\n",
           rl_s, rl_err, result.s, result.c.re, result.c.im);
}

static void verify_phase_space() {
    PhasePoint p0(1.0, 0.5);
    double norm_w = p0.to_w().abs();
    double norm_z = p0.to_z().abs();

    int orbit_len = 1000;
    PhasePoint* orb = new PhasePoint[orbit_len];
    GoldenFlow::orbit(p0, orb, orbit_len);

    double mean_th = 0, mean_rho = 0;
    for (int i = 0; i < orbit_len; ++i) {
        mean_th += orb[i].theta;
        mean_rho += orb[i].rho;
    }
    mean_th /= orbit_len;
    mean_rho /= orbit_len;
    double var_th = 0, var_rho = 0;
    for (int i = 0; i < orbit_len; ++i) {
        var_th += (orb[i].theta - mean_th) * (orb[i].theta - mean_th);
        var_rho += (orb[i].rho - mean_rho) * (orb[i].rho - mean_rho);
    }
    var_th /= orbit_len;
    var_rho /= orbit_len;
    delete[] orb;

    printf("phase_space: |w|=%.6f  |z|=%.6f  orbit_stats: mean_th=%.6f var_th=%.6f mean_rho=%.6f var_rho=%.6f\n",
           norm_w, norm_z, mean_th, var_th, mean_rho, var_rho);
}

static void verify_hilbert() {
    int nq = 3;
    QubitState psi;
    psi.init(nq);
    psi.set_uniform();
    double uniform_norm = psi.norm2();

    QubitState basis5;
    basis5.init(nq);
    basis5.set_basis(5);
    C64 ip = psi.inner(basis5);
    psi.free();
    basis5.free();

    int nth = 8, nrh = 8;
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(constants::PI, 0.0));
    double topo_norm = xi.norm2();

    HybridState hyb;
    hyb.init(nq, nth, nrh);
    hyb.inject(5, xi);
    double hyb_norm = hyb.norm2();
    int hyb_dim = (1 << nq) * nth * nrh;

    xi.free();
    hyb.free();

    printf("hilbert: uniform_norm=%.10f  <psi|5>=%.6f  topo_norm=%.10f  hyb_dim=%d  hyb_norm=%.10f\n",
           uniform_norm, ip.abs(), topo_norm, hyb_dim, hyb_norm);
}

static void verify_gates() {
    int nq = 2, nth = 8, nrh = 4;
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    HybridState psi;
    psi.init(nq, nth, nrh);
    psi.inject(0, xi);
    double init_norm = psi.norm2();

    BitVec u1(1, nq), v1(1, nq);

    HybridState psi_x;
    psi_x.init(nq, nth, nrh);
    memcpy(psi_x.amp, psi.amp, psi.total * sizeof(C64));
    hybrid_gates::apply_X_hyb(psi_x, u1);
    double px1 = psi_x.prob_disc(1);

    HybridState psi_z;
    psi_z.init(nq, nth, nrh);
    memcpy(psi_z.amp, psi.amp, psi.total * sizeof(C64));
    hybrid_gates::apply_Z_hyb(psi_z, v1);
    double pz0 = psi_z.prob_disc(0);

    bool xz_comm = hybrid_gates::verify_XZ_commutation(psi, u1, v1);

    HybridState psi_u;
    psi_u.init(nq, nth, nrh);
    memcpy(psi_u.amp, psi.amp, psi.total * sizeof(C64));
    hybrid_gates::apply_U_hyb(psi_u, 0.5);
    double u_norm = psi_u.norm2();

    Gate gx = gate_X(u1.bits), gz = gate_Z(v1.bits), gu = gate_U(0.5);

    printf("gates: init_norm=%.10f  X1_p1=%.6f  Z1_p0=%.6f  XZ_comm=%d  U_norm=%.10f  cost_X=%d  cost_Z=%d  cost_U=%d\n",
           init_norm, px1, pz0, xz_comm ? 1 : 0, u_norm, gx.cost(nq), gz.cost(nq), gu.cost(nq));

    xi.free(); psi.free(); psi_x.free(); psi_z.free(); psi_u.free();
}

static void verify_cocycle() {
    BitVec u(0b101, 3), v(0b110, 3);
    int ob = cocycle::omega_bool_f2(u, v);

    MirElement g0 = mir::g_phi(0.0), g1 = mir::g_phi(1.0), g2 = mir::g_phi(2.0);
    C64 dOmega = cocycle::cocycle_coboundary(g0, g1, g2);

    bool cocycle_ok = true;
    double max_coboundary = 0.0;
    for (int t = 0; t < 100; ++t) {
        MirElement ga = mir::g_phi(0.1 * t);
        MirElement gb = mir::g_phi(0.1 * t + 0.3);
        MirElement gc = mir::g_phi(0.1 * t + 0.7);
        C64 cb = cocycle::cocycle_coboundary(ga, gb, gc);
        if (cb.abs() > max_coboundary) max_coboundary = cb.abs();
        if (!cocycle::verify_cocycle(ga, gb, gc, 1e-8)) cocycle_ok = false;
    }

    BitVec up(0b010, 3), vp(0b011, 3);
    C64 omega_tot = cocycle::omega_total(g1, u, v, g2, up, vp);
    C64 F_tot = cocycle::F_total(g1, u, v, g2, up, vp);
    C64 ufe = cocycle::UFE_total(g1, u, v, g2, up, vp);

    C64 chi_val = character::chi(1, 2, 0.5, 0.3);

    CentralExtElement ce1(u, g1, v, C64(1, 0));
    CentralExtElement ce2(up, g2, vp, C64(0, 1));
    CentralExtElement ce12 = ce1 * ce2;

    printf("cocycle: omega_bool=%d  |dOmega|=%.2e  max_coboundary=%.2e  cocycle_100=%d  |UFE|=%.2e  |omega_tot|=%.6f  |F_tot|=%.6f  chi_12=%.6f  ext_phase=%.6f\n",
           ob, dOmega.abs(), max_coboundary, cocycle_ok ? 1 : 0, ufe.abs(),
           omega_tot.abs(), F_tot.abs(), chi_val.abs(), ce12.phase.abs());
}

static void verify_prng() {
    MirElement seed = mir::J_lambda(constants::PHI);
    MirPRNG prng(seed, BitVec(), 4);
    int N = 10000;
    int* bits = new int[N];
    prng.generate(bits, N);

    double beta = prng.beta(N);
    stats::StatReport r = stats::run_full_tests(bits, N, beta);

    int lags3[] = {1, 3, 7};
    double gamma3 = prng.gamma_corr(N, lags3, 3);

    printf("prng: N=%d  bias=%.10f  bias_z=%.4f  bias_p=%.6e  tv=%.10f  disc=%.10f  "
           "chi2=%.4f  chi2_p=%.6e  runs_z=%.4f  runs_p=%.6e  "
           "serial=%.6e  block_p=%.6e  apen=%.6f  "
           "spectral_peak=%.4f  spectral_p=%.6e  "
           "cusum_p=%.6e  ks=%.6f  "
           "min_ent=%.6f  shannon=%.6f  perm_ent=%.6f  "
           "lin_complex=%d  maurer=%.6f  max_autocorr=%.6f  "
           "cramer=%.6f  hoeffding=%.2e  "
           "fisher_combined=%.6e  gamma3=%.10f  "
           "tail_001=%.2e  tail_005=%.2e\n",
           N, r.bias, r.bias_zscore, r.bias_pvalue, r.tv_distance, r.discrepancy_1d,
           r.chi2_1bit, r.chi2_pvalue_1bit,
           r.runs.z_score, r.runs.p_value,
           r.serial_2, r.block_freq_pvalue, r.approximate_entropy_2,
           r.spectral.peak_ratio, r.spectral.p_value,
           r.cusum_pvalue_val, r.ks,
           r.min_entropy_1bit, r.shannon_1bit, r.permutation_entropy_3,
           r.linear_complexity_val, r.maurer_stat, r.max_correlation,
           r.cramer_rate_001, r.hoeffding_001,
           r.fisher_pvalue, gamma3,
           stats::tail_bound(N, 0.01), stats::tail_bound(N, 0.05));

    delete[] bits;
}

static void verify_machine() {
    int nq = 2, nth = 8, nrh = 4;
    HESMachine machine(nq, nth, nrh);

    // Quantum interference circuit: H(0) → Z(1) → H(0)
    // Starting from |0⟩⊗ξ₀:
    //   H(0)|0⟩ = (|0⟩+|1⟩)/√2
    //   Z(1)(|0⟩+|1⟩)/√2 = (|0⟩-|1⟩)/√2   (since ⟨v=1, x=1⟩=1)
    //   H(0)(|0⟩-|1⟩)/√2 = |1⟩
    // Expected Born: p(0)≈0, p(1)≈1, p(2)=0, p(3)=0
    Program prog;
    prog.append(gate_H(0));
    prog.append(gate_Z(1));
    prog.append(gate_H(0));

    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    HybridState psi0;
    psi0.init(nq, nth, nrh);
    psi0.inject(0, xi);

    Configuration config = machine.init(psi0);
    machine.run(config, prog);

    double probs[4];
    machine.born_distribution(config.psi, probs);
    bool is_p = complexity::check_P_Mir(prog, psi0, 4, 2, machine);

    printf("machine: time=%d  cost=%d  space=%d  halt=%d  tau=%d  born=[%.6f,%.6f,%.6f,%.6f]  P_Mir=%d\n",
           machine.program_time(prog), machine.program_cost(prog),
           machine.program_space(prog), config.halt, config.tau,
           probs[0], probs[1], probs[2], probs[3], is_p ? 1 : 0);

    config.psi.free(); psi0.free(); xi.free();
}

static void verify_quantum_interference() {
    // H-Z-H circuit: the simplest quantum interference demo
    // |0⟩⊗ξ₀ → H(0) → Z(1) → H(0) → Measure
    // H|0⟩ = (|0⟩+|1⟩)/√2,  Z(|0⟩+|1⟩)/√2 = (|0⟩-|1⟩)/√2,  H(|0⟩-|1⟩)/√2 = |1⟩
    // Born: p(0)≈0, p(1)≈1  (destructive at |0⟩, constructive at |1⟩)
    int nq = 1, nth = 8, nrh = 4;
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    HybridState psi;
    psi.init(nq, nth, nrh);
    psi.inject(0, xi);
    double init_norm = psi.norm2();

    hybrid_gates::apply_H_hyb(psi, 0);
    double p0_after_H = psi.prob_disc(0);
    double p1_after_H = psi.prob_disc(1);

    hybrid_gates::apply_Z_hyb(psi, BitVec(1, nq));
    hybrid_gates::apply_H_hyb(psi, 0);
    double final_norm = psi.norm2();

    double p0 = psi.prob_disc(0);
    double p1 = psi.prob_disc(1);
    double total = p0 + p1;
    if (total > 0) { p0 /= total; p1 /= total; }

    printf("quantum_interference: init_norm=%.10f  after_H: p0=%.6f p1=%.6f  "
           "final: p0=%.6f p1=%.6f  norm=%.10f  "
           "destructive=%d  constructive=%d\n",
           init_norm, p0_after_H, p1_after_H,
           p0, p1, final_norm,
           (p0 < 0.01) ? 1 : 0, (p1 > 0.99) ? 1 : 0);

    psi.free(); xi.free();
}

static void verify_topological_interference() {
    // Inject DIFFERENT topological states into sectors 0 and 1, then apply H
    // Interference depends on the overlap of the two topological states:
    //   - Same states: p(0)=1, p(1)=0  (perfectly constructive at |0⟩)
    //   - Orthogonal states: p(0)=1/2, p(1)=1/2  (no interference)
    //   - Different but overlapping: intermediate values (interference fringes)
    int nq = 1, nth = 16, nrh = 8;
    printf("topological_interference:");

    // Test 1: Same topological state → perfect constructive
    {
        TopologicalState xi_a, xi_b;
        xi_a.init(nth, nrh); xi_b.init(nth, nrh);
        xi_a.set_delta(PhasePoint(0.0, 0.0));
        xi_b.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi;
        psi.init(nq, nth, nrh);
        double inv_sqrt2 = 1.0 / sqrt(2.0);
        // Manually create superposition: (|0⟩⊗ξ_a + |1⟩⊗ξ_b)/√2
        for (int j = 0; j < psi.dim_top; ++j) {
            psi.amp[0 * psi.dim_top + j] = xi_a.amp[j] * inv_sqrt2;
            psi.amp[1 * psi.dim_top + j] = xi_b.amp[j] * inv_sqrt2;
        }
        hybrid_gates::apply_H_hyb(psi, 0);
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        printf("  same: p0=%.6f p1=%.6f", p0/tot, p1/tot);
        psi.free(); xi_a.free(); xi_b.free();
    }

    // Test 2: Different topological states → less constructive
    {
        TopologicalState xi_a, xi_b;
        xi_a.init(nth, nrh); xi_b.init(nth, nrh);
        xi_a.set_delta(PhasePoint(0.0, 0.0));
        xi_b.set_delta(PhasePoint(PI, 0.0));  // different θ

        HybridState psi;
        psi.init(nq, nth, nrh);
        double inv_sqrt2 = 1.0 / sqrt(2.0);
        for (int j = 0; j < psi.dim_top; ++j) {
            psi.amp[0 * psi.dim_top + j] = xi_a.amp[j] * inv_sqrt2;
            psi.amp[1 * psi.dim_top + j] = xi_b.amp[j] * inv_sqrt2;
        }
        hybrid_gates::apply_H_hyb(psi, 0);
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        printf("  diff: p0=%.6f p1=%.6f", p0/tot, p1/tot);
        psi.free(); xi_a.free(); xi_b.free();
    }

    // Test 3: Same with relative phase → destructive at |0⟩
    {
        TopologicalState xi_a, xi_b;
        xi_a.init(nth, nrh); xi_b.init(nth, nrh);
        xi_a.set_delta(PhasePoint(0.0, 0.0));
        // ξ_b = -ξ_a (relative phase flip)
        xi_b.set_delta(PhasePoint(0.0, 0.0));
        for (int j = 0; j < xi_b.total; ++j)
            xi_b.amp[j] = -xi_b.amp[j];

        HybridState psi;
        psi.init(nq, nth, nrh);
        double inv_sqrt2 = 1.0 / sqrt(2.0);
        for (int j = 0; j < psi.dim_top; ++j) {
            psi.amp[0 * psi.dim_top + j] = xi_a.amp[j] * inv_sqrt2;
            psi.amp[1 * psi.dim_top + j] = xi_b.amp[j] * inv_sqrt2;
        }
        hybrid_gates::apply_H_hyb(psi, 0);
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        printf("  phase: p0=%.6f p1=%.6f\n", p0/tot, p1/tot);
        psi.free(); xi_a.free(); xi_b.free();
    }
}

static void verify_weyl_interference() {
    // Compose two Weyl operators: verify cocycle phase produces same result
    // W₁ · W₂ applied individually vs. composed W₃ = exp(iΩ_tot) · W_{u₁⊕u₂,v₁⊕v₂;g₁★g₂}
    int nq = 2, nth = 8, nrh = 4;

    // W1 = W_{01, 00; g_phi(0.3)}
    BitVec u1(0b01, nq), v1(0b00, nq);
    MirElement g1 = mir::g_phi(0.3);

    // W2 = W_{10, 01; g_phi(0.5)}
    BitVec u2(0b10, nq), v2(0b01, nq);
    MirElement g2 = mir::g_phi(0.5);

    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.5, 0.3));

    // Method 1: Apply W1 then W2 individually
    HybridState psi1;
    psi1.init(nq, nth, nrh);
    psi1.inject(0, xi);
    hybrid_gates::apply_W_hyb(psi1, u1, v1, g1);
    hybrid_gates::apply_W_hyb(psi1, u2, v2, g2);

    // Method 2: Compose algebraically via WeylHybrid, then apply
    WeylHybrid W1(u1, v1, g1);
    WeylHybrid W2(u2, v2, g2);
    C64 cocycle_phase;
    WeylHybrid W12 = W1.compose(W2, cocycle_phase);

    HybridState psi2;
    psi2.init(nq, nth, nrh);
    psi2.inject(0, xi);
    hybrid_gates::apply_W_hyb(psi2, W12.u, W12.v, W12.g);
    // Multiply by cocycle phase
    for (int i = 0; i < psi2.total; ++i) {
        psi2.amp[i] = psi2.amp[i] * cocycle_phase;
    }

    // Compare Born distributions
    double born1[4], born2[4];
    double tot1 = 0, tot2 = 0;
    for (int x = 0; x < 4; ++x) {
        born1[x] = psi1.prob_disc(x); tot1 += born1[x];
        born2[x] = psi2.prob_disc(x); tot2 += born2[x];
    }
    if (tot1 > 0) for (int x = 0; x < 4; ++x) born1[x] /= tot1;
    if (tot2 > 0) for (int x = 0; x < 4; ++x) born2[x] /= tot2;

    double max_diff = 0;
    for (int x = 0; x < 4; ++x) {
        double d = fabs(born1[x] - born2[x]);
        if (d > max_diff) max_diff = d;
    }

    printf("weyl_interference: |omega_tot|=%.6f  phase=(%.6f,%.6f)  "
           "born_individual=[%.6f,%.6f,%.6f,%.6f]  "
           "born_composed=[%.6f,%.6f,%.6f,%.6f]  "
           "max_born_diff=%.2e  match=%d\n",
           W1.omega_total(W2).abs(),
           cocycle_phase.re, cocycle_phase.im,
           born1[0], born1[1], born1[2], born1[3],
           born2[0], born2[1], born2[2], born2[3],
           max_diff, (max_diff < 1e-6) ? 1 : 0);

    psi1.free(); psi2.free(); xi.free();
}

static void verify_deutsch_jozsa() {
    // Deutsch-Jozsa for n=1: determines constant vs balanced in 1 query
    // Constant f(x)=0: p(0)=1
    // Balanced f(x)=x: p(0)=0
    int nth = 8, nrh = 4;

    auto run_dj = [&](int oracle_type) -> double {
        HESMachine machine(2, nth, nrh);
        Program prog;

        // |0⟩|0⟩ → X on qubit 1 → |0⟩|1⟩
        prog.append(gate_X(0b10));
        // H on both qubits
        prog.append(gate_H(0));
        prog.append(gate_H(1));

        // Oracle
        if (oracle_type == 1) {
            // balanced: Z on qubit 0 gives (-1)^x phase kickback
            prog.append(gate_Z(0b01));
        }
        // constant: identity (no gate)

        // H on qubit 0
        prog.append(gate_H(0));

        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi0;
        psi0.init(2, nth, nrh);
        psi0.inject(0, xi);

        Configuration config = machine.init(psi0);
        machine.run(config, prog);

        // p(qubit0=0) = p(|00⟩) + p(|01⟩)  (marginal over qubit 1)
        double p_q0_is_0 = config.psi.prob_disc(0b00) + config.psi.prob_disc(0b10);
        double total = 0;
        for (int x = 0; x < 4; ++x) total += config.psi.prob_disc(x);
        if (total > 0) p_q0_is_0 /= total;

        config.psi.free(); psi0.free(); xi.free();
        return p_q0_is_0;
    };

    double p_constant = run_dj(0);
    double p_balanced = run_dj(1);

    printf("deutsch_jozsa: constant_p0=%.6f  balanced_p0=%.6f  "
           "constant_correct=%d  balanced_correct=%d\n",
           p_constant, p_balanced,
           (p_constant > 0.99) ? 1 : 0,
           (p_balanced < 0.01) ? 1 : 0);
}

static void verify_structure_detection() {
    MirElement seed = mir::g_phi(1.23);
    int N = 5000;

    double noise_levels[] = {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49};
    int num_levels = 8;

    printf("structure_detection: N=%d\n", N);
    for (int i = 0; i < num_levels; ++i) {
        signal::DetectionResult result =
            signal::detect_structure(seed, N, noise_levels[i]);
        printf("  noise=%.2f  bias=%.8f  score=%.6f  found=%d\n",
               noise_levels[i], result.report.bias,
               result.structure_score, result.structure_found ? 1 : 0);
    }
}

static void verify_gpu() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        printf("gpu: no_device\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int N = 1000000;
    int* d_bits;
    cudaMalloc(&d_bits, N * sizeof(int));
    MirElement seed = mir::J_lambda(constants::PHI);
    topcomp::cuda::launch_prng_bits(d_bits, seed, N);
    cudaDeviceSynchronize();

    double gpu_bias = topcomp::cuda::launch_bias(d_bits, N);
    double gpu_corr1 = topcomp::cuda::launch_correlation(d_bits, N, 1);
    double gpu_corr2 = topcomp::cuda::launch_correlation(d_bits, N, 2);

    int num_seeds = 256, N_per_seed = 1000;
    double* h_seeds_r = new double[num_seeds];
    double* h_seeds_i = new double[num_seeds];
    for (int i = 0; i < num_seeds; ++i) {
        h_seeds_r[i] = 0.1 * i;
        h_seeds_i[i] = 0.05 * i;
    }

    double *d_seeds_r, *d_seeds_i, *d_beta;
    cudaMalloc(&d_seeds_r, num_seeds * sizeof(double));
    cudaMalloc(&d_seeds_i, num_seeds * sizeof(double));
    cudaMalloc(&d_beta, num_seeds * sizeof(double));
    cudaMemcpy(d_seeds_r, h_seeds_r, num_seeds * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seeds_i, h_seeds_i, num_seeds * sizeof(double), cudaMemcpyHostToDevice);

    topcomp::cuda::launch_multi_seed_beta(d_beta, d_seeds_r, d_seeds_i, N_per_seed, num_seeds);
    cudaDeviceSynchronize();

    double* h_beta = new double[num_seeds];
    cudaMemcpy(h_beta, d_beta, num_seeds * sizeof(double), cudaMemcpyDeviceToHost);

    double avg_beta = 0, max_beta = 0;
    for (int i = 0; i < num_seeds; ++i) {
        avg_beta += h_beta[i];
        if (h_beta[i] > max_beta) max_beta = h_beta[i];
    }
    avg_beta /= num_seeds;

    int num_noise = 10;
    double h_noise[10] = {0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45};
    double *d_noise, *d_noise_bias;
    cudaMalloc(&d_noise, num_noise * sizeof(double));
    cudaMalloc(&d_noise_bias, num_noise * sizeof(double));
    cudaMemcpy(d_noise, h_noise, num_noise * sizeof(double), cudaMemcpyHostToDevice);

    topcomp::cuda::launch_noisy_structure(d_noise_bias, 0.7 * constants::ELL,
                                  -0.7 * constants::HALF_PI,
                                  10000, d_noise, num_noise);
    cudaDeviceSynchronize();

    double h_noise_bias[10];
    cudaMemcpy(h_noise_bias, d_noise_bias, num_noise * sizeof(double), cudaMemcpyDeviceToHost);

    printf("gpu: device=%s  SMs=%d  N=%d  bias=%.10f  corr1=%.10f  corr2=%.10f  "
           "seeds=%d  avg_beta=%.10f  max_beta=%.10f\n",
           prop.name, prop.multiProcessorCount, N,
           gpu_bias, gpu_corr1, gpu_corr2,
           num_seeds, avg_beta, max_beta);
    printf("  noise_resilience:");
    for (int i = 0; i < num_noise; ++i)
        printf(" %.2f:%.8f", h_noise[i], h_noise_bias[i]);
    printf("\n");

    cudaFree(d_bits); cudaFree(d_seeds_r); cudaFree(d_seeds_i);
    cudaFree(d_beta); cudaFree(d_noise); cudaFree(d_noise_bias);
    delete[] h_seeds_r; delete[] h_seeds_i; delete[] h_beta;
}

static void verify_benchmark() {
    int N = 100, nq = 2;
    int* predictions = new int[N];
    double* probs = new double[N];
    int* labels = new int[N];
    Program* programs = new Program[N];

    unsigned int rng = 42;
    for (int i = 0; i < N; ++i) {
        labels[i] = i % 2;
        rng = rng * 1103515245u + 12345u;
        double noise = static_cast<double>(rng & 0xFFFF) / 65535.0;
        probs[i] = (labels[i] == 1) ? 0.7 + 0.2 * noise : 0.1 + 0.2 * noise;
        predictions[i] = (probs[i] > 0.5) ? 1 : 0;
        programs[i].append(gate_X(i % 4));
        programs[i].append(gate_Z((i + 1) % 4));
        programs[i].append(gate_U(0.1 * i));
    }

    Metrics m = benchmark::run_benchmark(predictions, probs, labels, programs, N, nq, 0.5, 300.0);
    double var_acc = m.accuracy * (1.0 - m.accuracy);
    benchmark::ConfidenceInterval ci = benchmark::compute_ci95(m.accuracy, var_acc, N);

    ParetoPoint candidates[10];
    for (int i = 0; i < 10; ++i) {
        candidates[i].nll = 0.5 + 0.1 * i;
        candidates[i].wall = 100.0 + 10 * (9 - i);
        candidates[i].mem = 5.0 + i;
        candidates[i].energy = 50.0 + 5 * i;
        candidates[i].config_id = i;
    }
    ParetoPoint front[10];
    int psize = benchmark::compute_pareto_front(candidates, 10, front, 10);

    printf("benchmark: accuracy=%.6f  nll=%.6f  ci95=[%.4f,%.4f]  pareto_size=%d  "
           "mse=%.6f  brier=%.6f  tv=%.6f  wall=%.2f  mem=%.2f\n",
           m.accuracy, m.nll, ci.lower, ci.upper, psize,
           m.mse, m.brier, m.tv, m.wall_time, m.mem);

    delete[] predictions; delete[] probs; delete[] labels; delete[] programs;
}

static void verify_witt_sl2() {
    Dual<double> x(2.0, 1.0);
    Dual<double> f = dexp(x);
    double exp_err = fabs(f.val - exp(2.0));
    double dexp_err = fabs(f.eps - exp(2.0));

    Dual<double> g = dsin(Dual<double>(PI / 4.0, 1.0));
    double sin_err = fabs(g.val - sin(PI / 4.0));
    double cos_err = fabs(g.eps - cos(PI / 4.0));

    TruncPoly test_poly;
    test_poly.coeffs[0] = 1.0; test_poly.coeffs[1] = 0.5; test_poly.coeffs[2] = 0.25;
    bool he_ok = sl2_verify::check_he(test_poly);
    bool hf_ok = sl2_verify::check_hf(test_poly);
    bool ef_ok = sl2_verify::check_ef(test_poly);

    printf("witt_sl2: dual_exp_err=%.2e  dual_dexp_err=%.2e  dual_sin_err=%.2e  dual_cos_err=%.2e  [h,e]=2e:%d  [h,f]=-2f:%d  [e,f]=h:%d\n",
           exp_err, dexp_err, sin_err, cos_err,
           he_ok ? 1 : 0, hf_ok ? 1 : 0, ef_ok ? 1 : 0);
}

static void verify_lie_rep() {
    SL2Element D_el = SL2Element::D();
    SL2Element E_el = SL2Element::E();
    SL2Element K_el = SL2Element::K();

    SL2Element DE = D_el.bracket(E_el);
    SL2Element DK = D_el.bracket(K_el);

    double t0 = 1.5;
    double exp_D_err = fabs(exp_action::exp_D(0.3, t0) - (t0 + 0.3));
    double exp_E_err = fabs(exp_action::exp_E(2.0, t0) - 2.0 * t0);
    double exp_K_err = fabs(exp_action::exp_negK(0.1, t0) - t0 / (1.0 + 0.1 * t0));

    double area_ratio = contraction::area_ratio();
    double vol3 = contraction::volume_ratio(3);

    double du_val = forms::du_on_Xphi();
    double domega_val = forms::domega_on_Xphi();
    double domega_err = fabs(domega_val - (-PI));

    printf("lie_rep: [D,E]=(%.2f,%.2f,%.2f)  [D,K]=(%.2f,%.2f,%.2f)  "
           "exp_D_err=%.2e  exp_E_err=%.2e  exp_K_err=%.2e  "
           "area_ratio=%.6f  vol3=%.6f  du=%.6f  domega_err=%.2e\n",
           DE.v.re, DE.u.re, DE.w.re,
           DK.v.re, DK.u.re, DK.w.re,
           exp_D_err, exp_E_err, exp_K_err,
           area_ratio, vol3, du_val, domega_err);
}

static void verify_compiler() {
    using namespace compiler;

    MirIR ir(3);
    ir.emit(MirOp::pauli_x(BitVec(1, 3)));
    ir.emit(MirOp::flow(0.5));
    ir.emit(MirOp::flow(1.5));
    ir.emit(MirOp::modular(2));
    ir.emit(MirOp::modular(3));
    ir.emit(MirOp::pauli_x(BitVec(0, 3)));
    ir.emit(MirOp::mirror(mir::g_phi(0.7)));
    ir.emit(MirOp::mirror(mir::g_phi(0.3)));
    ir.emit(MirOp::measure());

    int pre_count = ir.count, pre_cost = ir.total_cost();
    auto st = analysis::count_ops(ir);
    optimizer::optimize(ir);
    int post_count = ir.count, post_cost = ir.total_cost();

    HybridProgram prog = lowering::lower_to_gates(ir);

    MirIR sl2_ir(2);
    sl2_ir.emit(MirOp::sl2_translate(1.0));
    sl2_ir.emit(MirOp::sl2_scale(PHI));
    sl2_ir.emit(MirOp::sl2_conformal(0.5));
    int sl2_pre = sl2_ir.count;
    optimizer::optimize(sl2_ir);
    int sl2_post = sl2_ir.count;

    MirIR prng_ir = frontend::build_prng_circuit(3, 20, mir::J_lambda(PHI));
    MirIR detector = frontend::build_structure_detector(2, 5, mir::g_phi(1.0));
    MirIR ergodic = frontend::build_ergodic_projector(2, 10);

    printf("compiler: pre=%d/%d  post=%d/%d  lowered=%d/%d  "
           "sl2_pre=%d  sl2_post=%d  "
           "prng_ops=%d  detector_ops=%d  ergodic_ops=%d  "
           "ops_X=%d  Z=%d  U=%d  M=%d  Mir=%d  Meas=%d\n",
           pre_count, pre_cost, post_count, post_cost, prog.length, prog.total_cost(),
           sl2_pre, sl2_post,
           prng_ir.count, detector.count, ergodic.count,
           st.n_pauli_x, st.n_pauli_z, st.n_flow, st.n_modular,
           st.n_mirror, st.n_measure);
}

static void verify_structure_finder() {
    MirElement seed = mir::J_lambda(PHI);
    MirPRNG prng(seed, BitVec(), 4);
    int N = 5000;
    int* bits = new int[N];
    prng.generate(bits, N);

    auto sig = structure::find_structure(bits, N);

    int orbit_len = 200;
    PhasePoint* orbit = new PhasePoint[orbit_len];
    GoldenFlow gf;
    orbit[0] = PhasePoint(0.5, 1.0);
    for (int i = 1; i < orbit_len; ++i)
        orbit[i] = gf.flow(orbit[i-1], 1.0);

    double char_bound = structure::character_analysis::max_character_sum(orbit, orbit_len, 5, 5);
    bool eigen_ok = structure::character_analysis::verify_eigencharacter(1, 0, orbit[0]);
    double ratio = structure::contraction_detect::estimate_contraction(orbit, orbit_len);
    bool golden = structure::contraction_detect::is_golden_contraction(ratio);

    printf("structure_finder: N=%d  structured=%d  fisher_p=%.6e  "
           "bias_p=%.6e  corr_p=%.6e  chi2_p=%.6e  runs_p=%.6e  spectral_p=%.6e  "
           "entropy_deficit=%.6f  sig_tests=%d/%d  "
           "char_bound=%.6f  eigen=%d  contraction=%.6f  golden=%d\n",
           N, sig.is_structured ? 1 : 0, sig.fisher_combined_pvalue,
           sig.bias_pvalue, sig.correlation_pvalue, sig.chi2_pvalue,
           sig.runs_pvalue, sig.spectral_pvalue,
           sig.entropy_deficit, sig.num_tests_significant, sig.num_tests_total,
           char_bound, eigen_ok ? 1 : 0, ratio, golden ? 1 : 0);

    printf("  large_deviation:");
    for (double eps = 0.01; eps <= 0.2; eps += 0.05) {
        double bound = structure::large_deviation::bias_tail_bound(N, eps, bits);
        printf(" %.2f:%.6e", eps, bound);
    }
    printf("\n");

    delete[] bits;
    delete[] orbit;
}

// ════════════════════════════════════════════════════════════════════════════
// Verify: Noncommutative dynamics — [U_t, M_m] and uncertainty
// ════════════════════════════════════════════════════════════════════════════
static void verify_noncommutative_dynamics() {
    int nth = 16, nrh = 8;

    // === Part 1: U_t M_m commutation on actual states ===
    // Relation: U_t M_m = e^{-i2πmt} M_m U_t
    // psi_um = M_m U_t |ψ⟩,  psi_mu = U_t M_m |ψ⟩ = e^{-i2πmt} · psi_um
    double t = 0.3;
    int m = 2;
    TopologicalState psi_um, psi_mu;
    psi_um.init(nth, nrh);
    psi_mu.init(nth, nrh);

    TopologicalState xi0;
    xi0.init(nth, nrh);
    xi0.set_delta(PhasePoint(PI / 3.0, 0.5));

    // psi_um = M_m U_t |ψ⟩
    memcpy(psi_um.amp, xi0.amp, xi0.total * sizeof(C64));
    topo_gates::apply_U(psi_um, t);
    topo_gates::apply_M(psi_um, m);

    // psi_mu = U_t M_m |ψ⟩, then divide by phase to compare
    memcpy(psi_mu.amp, xi0.amp, xi0.total * sizeof(C64));
    topo_gates::apply_M(psi_mu, m);
    topo_gates::apply_U(psi_mu, t);
    C64 inv_phase = topo_gates::commutation_phase(t, m).conj();  // e^{+i2πmt}
    for (int i = 0; i < psi_mu.total; ++i)
        psi_mu.amp[i] = psi_mu.amp[i] * inv_phase;

    double max_diff = 0;
    for (int i = 0; i < psi_um.total; ++i) {
        double d = (psi_um.amp[i] - psi_mu.amp[i]).abs();
        if (d > max_diff) max_diff = d;
    }

    // Born distribution comparison (phase-independent, exact)
    double born_diff = 0;
    for (int i = 0; i < psi_um.total; ++i) {
        double d = fabs(psi_um.amp[i].norm2() - psi_mu.amp[i].norm2());
        if (d > born_diff) born_diff = d;
    }

    // === Part 2: Uncertainty bound (Robertson + Schrödinger) ===
    // Pauli X and Z on 1 qubit: [X,Z] = -2iY
    measurement::DenseOp X_op(2), Z_op(2);
    X_op.at(0, 1) = C64(1, 0); X_op.at(1, 0) = C64(1, 0);
    Z_op.at(0, 0) = C64(1, 0); Z_op.at(1, 1) = C64(-1, 0);

    // Pure state |+⟩ = (|0⟩+|1⟩)/√2
    QubitState plus;
    plus.init(1);
    plus.amp[0] = C64(1.0 / sqrt(2.0), 0);
    plus.amp[1] = C64(1.0 / sqrt(2.0), 0);
    auto rho = measurement::pure_state_density(plus);

    auto ub = measurement::heisenberg_bound(rho, X_op, Z_op);
    auto sb = measurement::schrodinger_bound(rho, X_op, Z_op);

    C64 phase = topo_gates::commutation_phase(t, m);
    printf("noncommutative_dynamics: UM_amp_diff=%.2e  UM_born_diff=%.2e  "
           "heis={%.6f>=%.6f,%d}  schrod={%.6f>=%.6f,%d}  "
           "comm_phase=(%.4f,%.4f)\n",
           max_diff, born_diff,
           ub.lhs, ub.rhs, ub.satisfied ? 1 : 0,
           sb.lhs, sb.rhs, sb.satisfied ? 1 : 0,
           phase.re, phase.im);

    psi_um.free(); psi_mu.free(); xi0.free();
    plus.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Verify: Multi-boundary measurement — joint p(x,j), marginals, conditionals
// ════════════════════════════════════════════════════════════════════════════
static void verify_multi_boundary() {
    int nq = 1, nth = 8, nrh = 4;
    int dim_d = 1 << nq;
    int dim_t = nth * nrh;

    // Build state: H(0)|0⟩⊗δ(π/4, 0.3)
    // After H: (1/√2)(|0⟩+|1⟩) ⊗ δ  → superposition over discrete,
    // all topological weight at one grid point
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(PI / 4.0, 0.3));

    HybridState psi;
    psi.init(nq, nth, nrh);
    psi.inject(0, xi);
    hybrid_gates::apply_H_hyb(psi, 0);

    // Joint measurement: p(x, j) over discrete × topological grid
    auto result = measurement::multi_boundary_measure(psi);

    // Compute: max discrete probability, topological concentration,
    // marginal uniformity
    double max_disc = 0;
    for (int x = 0; x < dim_d; ++x)
        if (result.marg_disc[x] > max_disc) max_disc = result.marg_disc[x];

    double max_topo = 0;
    int max_topo_j = 0;
    for (int j = 0; j < dim_t; ++j)
        if (result.marg_topo[j] > max_topo) { max_topo = result.marg_topo[j]; max_topo_j = j; }

    // After H: p(0) = p(1) = 0.5 uniformly, all topo weight at one grid point
    bool disc_uniform = fabs(result.marg_disc[0] - 0.5) < 1e-6 &&
                         fabs(result.marg_disc[1] - 0.5) < 1e-6;
    bool topo_concentrated = max_topo > 0.99;
    bool normalized = fabs(result.norm_check - 1.0) < 1e-10;

    // Conditional: p(j|x=0) should be a delta at the same topo grid point
    double* cond = new double[dim_t];
    measurement::conditional_topo_given_disc(result.joint, dim_d, dim_t, 0, cond);
    double cond_max = 0;
    for (int j = 0; j < dim_t; ++j)
        if (cond[j] > cond_max) cond_max = cond[j];

    // Now apply Weyl gate and re-measure to show how multi-boundary changes
    HybridState psi2;
    psi2.init(nq, nth, nrh);
    psi2.inject(0, xi);
    hybrid_gates::apply_H_hyb(psi2, 0);
    hybrid_gates::apply_W_hyb(psi2, BitVec(0b1, nq), BitVec(0b0, nq),
                               mir::g_phi(0.5));

    auto result2 = measurement::multi_boundary_measure(psi2);

    // After W_{1,0;g}: discrete sector shifted, topo transported
    printf("multi_boundary: disc_uniform=%d  topo_concentrated=%d  norm=%.10f  "
           "marg_disc=[%.4f,%.4f]  max_topo=%.4f  cond_max=%.4f  "
           "post_W_disc=[%.4f,%.4f]  post_W_topo_spread=%.4f\n",
           disc_uniform ? 1 : 0, topo_concentrated ? 1 : 0, result.norm_check,
           result.marg_disc[0], result.marg_disc[1],
           max_topo, cond_max,
           result2.marg_disc[0], result2.marg_disc[1],
           1.0 - result2.marg_topo[max_topo_j]);

    delete[] cond;
    result.free(); result2.free();
    xi.free(); psi.free(); psi2.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Simultaneous Information Retainment Analytics
// ════════════════════════════════════════════════════════════════════════════
static void verify_hybrid_info() {
    printf("\n=== Simultaneous Information Retainment Analytics ===\n\n");

    // --- Heisenberg Group ---
    {
        using namespace heisenberg;
        HeisElement a(BitVec(1, 2), BitVec(2, 2), C64(1, 0));
        HeisElement b(BitVec(2, 2), BitVec(1, 2), C64(1, 0));
        HeisElement ab = a * b;
        HeisElement ai = a.inverse();
        HeisElement aai = a * ai;
        printf("Heis(V₂): a=(01,10,1)  b=(10,01,1)\n");
        printf("  a·b = (%u,%u, %.4f+%.4fi)\n",
               ab.u.bits, ab.v.bits, ab.zeta.re, ab.zeta.im);
        printf("  a·a⁻¹ = (%u,%u, %.4f+%.4fi)  [identity=%d]\n",
               aai.u.bits, aai.v.bits, aai.zeta.re, aai.zeta.im,
               verify_identity(a) ? 1 : 0);
    }

    // --- Clifford Algebra ---
    {
        using namespace clifford;
        CliffElement g0 = CliffElement::gamma(3, 0);
        CliffElement g1 = CliffElement::gamma(3, 1);
        CliffElement g01 = g0 * g1;
        printf("\nCl_{0,3}: γ₀²=%.1f  γ₁²=%.1f  {γ₀,γ₁}=%.1f\n",
               (g0 * g0).coeffs[0], (g1 * g1).coeffs[0],
               (g01 + g1 * g0).coeffs[0]);
    }

    // --- Boolean Algebra ---
    {
        using namespace boolean_algebra;
        BoolFunc xor2(2);
        xor2.values[0] = 0; xor2.values[1] = 1;
        xor2.values[2] = 1; xor2.values[3] = 0;
        printf("\nB₂ XOR: ANF = c∅=%d c₁=%d c₂=%d c₁₂=%d  wt=%d  Möbius=%d\n",
               xor2.anf_coeff(0), xor2.anf_coeff(1),
               xor2.anf_coeff(2), xor2.anf_coeff(3),
               xor2.weight(), verify_mobius_inversion(xor2) ? 1 : 0);
    }

    // --- Shadow Functor ---
    {
        using namespace shadow;
        ShadowData sd(2);
        BitVec u(1, 2), v(1, 2);
        ShadowFunctorResult r = compute_shadow(u, v, 2);
        printf("\nShadow S_Mir(01,01): ⟨u,v⟩_sh=%d  σ_sh=%.1f  pipeline=%d\n",
               r.inner_product, r.sigma_sh, r.pipeline_valid ? 1 : 0);
    }

    // --- Fibonacci/Spiral ---
    {
        using namespace fibonacci;
        printf("\nFibonacci: F₁₀=%d  Binet=%.1f  φ¹⁰=%.6f≈F₁₀·φ+F₉=%.6f\n",
               fib_matrix(10), fib_binet(10),
               pow(PHI, 10), fib_binet(10) * PHI + fib_binet(9));
        printf("  Geodesic: |c_φ|=%.8f  dω̃/ds*=%.8f\n",
               geodesic_speed(), domega_tilde_ds());
    }

    // --- Connection/Index ---
    {
        using namespace connection;
        Connection1Form A = Connection1Form::golden_flow();
        printf("\nConnection: A_θ=(%.4f,%.4f)  flat=%d  Â=%.1f  Arf(2)=%.4f\n",
               A.A_theta.re, A.A_theta.im, A.is_flat() ? 1 : 0,
               A_hat_flat_dim2(), arf_invariant(2));
    }

    // --- Translation Matrices ---
    {
        using namespace translation;
        TranslationMatrix mp = M_Pi(1.0);
        TranslationMatrix mf = M_Phi(1.0);
        TranslationMatrix comp = mp * mf;
        printf("\nT-matrices: M_Π=(%.4f,%.4f)  M_Φ=(%.4f,%.4f)\n",
               mp.alpha, mp.lambda(), mf.alpha, mf.lambda());
        printf("  M_Π·M_Φ: α=%.4f  λ=%.6f\n", comp.alpha, comp.lambda());
    }

    // --- Witt Commutator ---
    {
        using namespace witt_commutator;
        Witt2Element g = group_element(1.5);
        Witt2Element h = group_element(2.3);
        Witt2Element comm = witt2_commutator(g, h);
        Witt2Element F = F_group(g, h);
        printf("\nWitt₂ [g,h]: comm=(%.4f,%.4f,%.4f)  F=(%.4f,%.4f,%.6f)\n",
               comm.x0, comm.x1, comm.x2, F.x0, F.x1, F.x2);
    }

    // --- Disc-Top Information ---
    {
        using namespace disc_top_info;
        int nq = 2;
        int dim = 1 << nq;
        auto rho = measurement::maximally_mixed(dim);
        MirElement g = mir::g_phi(1.0);
        MirElement h = mir::g_phi(2.0);
        BitVec u(1, 2), v(2, 2), up(3, 2), vp(0, 2);

        InfoRetainmentAnalytics a = compute_full_analytics(rho, g, h, u, v, up, vp);
        printf("\nDisc-Top Info (n=%d):\n", a.n_qubits);
        printf("  S_total=%.6f  S_disc=%.6f  S_top=%.6f  I(d:t)=%.6f\n",
               a.entropy_total, a.entropy_disc, a.entropy_top, a.mutual_info);
        printf("  cocycle_splits=%d  ufe_chain=%d  cross_sector=%d\n",
               a.cocycle_splits ? 1 : 0, a.ufe_chain_valid ? 1 : 0,
               a.cross_sector_commutes ? 1 : 0);
        printf("  entropy_preserved=%d  born_normalized=%d\n",
               a.entropy_preserved ? 1 : 0, a.born_normalized ? 1 : 0);
    }

    // --- Grand Chain ---
    {
        int steps = grand_chain::verify_grand_chain();
        printf("\nGrand Derivation Chain: %d/8 steps pass\n", steps);
    }

    // --- Isometry ---
    {
        using namespace isometry;
        MirElement g1 = mir::g_phi(1.0);
        MirElement g2 = mir::U_pi();
        printf("\nIsometry: g_φ(1)→%d  U_π→%d  compatible=%d\n",
               verify_isometry(g1) ? 1 : 0,
               verify_isometry(g2) ? 1 : 0,
               verify_action_compatibility(g1, g2) ? 1 : 0);
    }

    printf("\n=== End Information Retainment Analytics ===\n");
}

static void verify_mixed_operators() {
    printf("\n=== Mixed Operators & Boundary Calculus ===\n");
    using namespace mixed_cocycle;
    using namespace mixed_ops;
    using namespace boundary;

    int n = 2;

    // Mixed cocycle κ=0 limit
    {
        MirElement g = mir::g_phi(1.0);
        MirElement h = mir::g_phi(2.0);
        BitVec u(1, n), v(2, n), up(3, n), vp(0, n);
        bool ok = verify_kappa_zero_limit(g, u, v, h, up, vp);
        printf("mixed cocycle κ=0 limit:   %s\n", ok ? "PASS" : "FAIL");
    }

    // Mixed cocycle condition δΩ_mix = 0
    {
        MirElement g1 = mir::g_phi(1.0), g2 = mir::g_phi(2.0), g3 = mir::g_phi(0.5);
        BitVec u1(1, n), v1(2, n), u2(3, n), v2(1, n), u3(2, n), v3(3, n);
        bool ok0 = verify_mixed_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.0);
        bool ok1 = verify_mixed_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.5);
        printf("δΩ_mix = 0  (κ=0): %s  (κ=0.5): %s\n",
               ok0 ? "PASS" : "FAIL", ok1 ? "PASS" : "FAIL");
    }

    // U_t^mix κ=0 limit
    {
        bool ok = verify_U_mix_limit(1, 8, 4, 1.0);
        printf("U_t^mix|κ=0 = U_t^hyb:    %s\n", ok ? "PASS" : "FAIL");
    }

    // X_u^mix κ=0 limit
    {
        bool ok = verify_X_mix_limit(1, 8, 4);
        printf("X_u^mix|κ=0 = X_u^hyb:    %s\n", ok ? "PASS" : "FAIL");
    }

    // Mixed nontrivial when κ≠0
    {
        bool ok = verify_mixed_nontrivial(1, 8, 4, 0.5, 1.0);
        printf("U_t^mix|κ=0.5 ≠ U_t^hyb:  %s\n", ok ? "PASS" : "FAIL");
    }

    // Boundary marginal consistency
    {
        HybridState psi;
        psi.init(1, 8, 4);
        TopologicalState xi;
        xi.init(8, 4);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        bool ok = verify_marginal_consistency(psi);
        printf("boundary marginal consistency: %s\n", ok ? "PASS" : "FAIL");
        psi.free();
    }

    // Regime classification
    {
        using namespace regime;
        Gate dg[2] = {gate_X(1), gate_Z(2)};
        Gate tg[1] = {gate_U(1.0)};
        Gate hg[2] = {gate_X(1), gate_U(1.0)};
        printf("regime: disc=%s  top=%s  hyb=%s  mix=%s\n",
               regime_name(classify(dg, 2)),
               regime_name(classify(tg, 1)),
               regime_name(classify(hg, 2)),
               regime_name(classify(dg, 2, 0.5)));
    }

    printf("=== End Mixed Operators ===\n");
}

static void verify_quantum_algebra() {
    printf("\n=== Quantum Algebra Q^max_{A,n} ===\n");

    int n = 2;
    int nth = 8, nrh = 4;

    // --- I. Twisted Convolution Algebra ---
    {
        using namespace twisted_algebra;
        MirElement ge = MirElement::identity();
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);
        WeylHybrid wc(BitVec(1, n), BitVec(1, n), ge);

        AlgElement f = AlgElement::single(C64(1.0, 0.0), wa);
        AlgElement g = AlgElement::single(C64(1.0, 0.0), wb);
        AlgElement h = AlgElement::single(C64(1.0, 0.0), wc);

        AlgElement fg = twisted_product(f, g);
        bool prod_ok = (fg.n_terms == 1);
        printf("twisted product f★g:         %s\n", prod_ok ? "PASS" : "FAIL");

        bool assoc = verify_associativity(f, g, h);
        printf("(f★g)★h = f★(g★h):           %s\n", assoc ? "PASS" : "FAIL");
    }

    // --- III. Symplectic Form & Commutation ---
    {
        using namespace symplectic;
        MirElement ge = MirElement::identity();
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);

        // X_1 and Z_1 should NOT commute: sigma = πi ≠ 0 mod 2πi
        bool xz_commute = commute(wa, wb);
        printf("X₁, Z₁ don't commute:       %s\n", !xz_commute ? "PASS" : "FAIL");

        // Same-type should commute: X_1, X_2
        WeylHybrid wc(BitVec(2, n), BitVec(0, n), ge);
        bool xx_commute = commute(wa, wc);
        printf("X₁, X₂ commute:             %s\n", xx_commute ? "PASS" : "FAIL");

        // Commutator coefficient [W_a,W_b]:
        C64 cc = commutator_coeff(wa, wb);
        bool cc_nonzero = cc.norm2() > 0.1;
        printf("[X₁,Z₁] ≠ 0:                %s\n", cc_nonzero ? "PASS" : "FAIL");
    }

    // --- IV. *-Algebra ---
    {
        using namespace star_algebra;
        MirElement ge = MirElement::identity();
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);

        twisted_algebra::AlgElement f = twisted_algebra::AlgElement::single(C64(1.0, 0.5), wa);
        twisted_algebra::AlgElement g = twisted_algebra::AlgElement::single(C64(0.7, -0.3), wb);

        bool anti = verify_star_antiautomorphism(f, g);
        printf("(f★g)* = g*★f*:              %s\n", anti ? "PASS" : "FAIL");
    }

    // --- V. Discrete Completeness ---
    {
        using namespace completeness;
        bool orth1 = verify_orthogonality(1);
        bool orth2 = verify_orthogonality(2);
        bool card1 = verify_cardinality(1);
        bool card2 = verify_cardinality(2);
        printf("Pauli basis orthogonal n=1:  %s\n", orth1 ? "PASS" : "FAIL");
        printf("Pauli basis orthogonal n=2:  %s\n", orth2 ? "PASS" : "FAIL");
        printf("|{X_uZ_v}| = 4^n  (n=1,2):  %s\n",
               (card1 && card2) ? "PASS" : "FAIL");
    }

    // --- VI. Diagonal States ---
    {
        using namespace diagonal_state;
        HybridState psi;
        psi.init(n, nth, nrh);
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        double probs[4];
        sector_probabilities(psi, probs);
        bool diag_ok = fabs(probs[0] - 1.0) < 1e-10;
        printf("sector probs (basis state):  %s\n", diag_ok ? "PASS" : "FAIL");

        psi.free();
    }

    // --- VII. N-point Correlation Functions ---
    {
        using namespace correlation;
        HybridState psi;
        psi.init(n, nth, nrh);
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        MirElement ge = MirElement::identity();
        WeylHybrid wa = WeylHybrid::identity(n);
        WeylHybrid wb(BitVec(1, n), BitVec(0, n), ge);

        // G_1 of identity should be 1 (for normalized state)
        C64 g1_id = G1(psi, wa);
        bool g1_ok = fabs(g1_id.re - 1.0) < 1e-8 && fabs(g1_id.im) < 1e-8;
        printf("G₁(identity) = 1:           %s\n", g1_ok ? "PASS" : "FAIL");

        // G_2 factorization
        bool g2_ok = verify_G2_factorization(psi, wa, wb);
        printf("G₂ factorization:            %s\n", g2_ok ? "PASS" : "FAIL");

        psi.free();
    }

    // --- IX. Flow Generator ---
    {
        using namespace flow_generator;
        PhasePoint p(PI / 3.0, 0.5);
        bool u_inv = verify_u_invariance(p, 3.0);
        printf("u(θ,ρ) flow-invariant:       %s\n", u_inv ? "PASS" : "FAIL");

        bool omega_dec = verify_omega_decrease(p, 2.0);
        printf("ω̃ decrease rate = -π:         %s\n", omega_dec ? "PASS" : "FAIL");
    }

    // --- XI. Circuit Complexity Metrics ---
    {
        using namespace circuit_metrics;
        Program prog;
        prog.append(gate_X(1));
        prog.append(gate_Z(2));
        prog.append(gate_U(1.0));
        prog.append(gate_M(3));
        prog.append(gate_P(0));
        CircuitAnalysis a = analyze(prog, n);
        printf("circuit: size=%d disc=%d top=%d meas=%d depth=%d\n",
               a.size, a.disc, a.top, a.meas, a.depth);
        bool metrics_ok = (a.size == 5 && a.disc == 3 && a.top == 2 && a.meas == 1);
        printf("circuit metrics correct:     %s\n", metrics_ok ? "PASS" : "FAIL");
    }

    // --- XII. Projective Representation ---
    {
        using namespace projective_rep;
        HybridState psi;
        psi.init(n, nth, nrh);
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        MirElement ge = MirElement::identity();
        CentralElement x_ce(C64(1.0, 0.0), WeylHybrid(BitVec(1, n), BitVec(0, n), ge));
        CentralElement y_ce(C64(1.0, 0.0), WeylHybrid(BitVec(0, n), BitVec(1, n), ge));

        bool rep_ok = verify_representation(x_ce, y_ce, psi);
        printf("Π_Ω representation:          %s\n", rep_ok ? "PASS" : "FAIL");
        psi.free();
    }

    printf("=== End Quantum Algebra ===\n");
}

// ════════════════════════════════════════════════════════════════════════════
// Hardware Embedding Verification
// ════════════════════════════════════════════════════════════════════════════
void verify_hardware_embedding() {
    printf("=== Hardware Embedding ===\n");
    using namespace hardware;

    HardwareTopology hw = HardwareTopology::default_workstation();
    bool complete = verify::topology_complete(hw);
    printf("topology complete:           %s\n", complete ? "PASS" : "FAIL");

    bool roles_ok = true;
    roles_ok &= canonical_role(SubsystemKind::CPU) == SubsystemRole::EXACT_CONTROL;
    roles_ok &= canonical_role(SubsystemKind::GPU) == SubsystemRole::OPERATOR_ENGINE;
    roles_ok &= canonical_role(SubsystemKind::VRAM) == SubsystemRole::CARRIER_MEMORY;
    printf("canonical role assignment:   %s\n", roles_ok ? "PASS" : "FAIL");

    // Machine state placement
    machine_state::MachineState ms;
    ms.init(2, 8, 4);
    bool placement_ok = ms.verify_placement();
    printf("doctrinal placement:         %s\n", placement_ok ? "PASS" : "FAIL");

    bool consistent = machine_state::verify::machine_consistent(ms);
    printf("machine state consistent:    %s\n", consistent ? "PASS" : "FAIL");

    printf("total memory footprint:      %lld bytes\n",
           static_cast<long long>(ms.total_memory_bytes()));

    // Regime planner
    planner::ComputeQuery q;
    q.n_qubits = 2; q.N_theta = 8; q.N_rho = 4;
    q.n_disc_gates = 10; q.n_top_gates = 20; q.n_hyb_gates = 5;
    planner::ResourceBudget budget = planner::ResourceBudget::unlimited();
    planner::ExecutionPlan ep = planner::plan(ms, q, budget, hw);

    const char* regime_names[] = { "CPU_ONLY", "GPU_ONLY", "SPLIT", "STREAMING" };
    printf("execution regime:            %s\n",
           regime_names[static_cast<int>(ep.regime)]);
    bool plan_ok = planner::verify::plan_valid(ep, budget);
    printf("plan valid:                  %s\n", plan_ok ? "PASS" : "FAIL");

    // Fibered memory
    fibered::FiberedLayout fl = fibered::FiberedLayout::wrap(ms.hyb.state);
    bool fiber_ok = fibered::verify::fiber_norms_consistent(fl);
    printf("fibered layout consistent:   %s\n", fiber_ok ? "PASS" : "FAIL");

    // Spectral transport
    TopologicalState psi;
    psi.init(8, 4);
    psi.set_delta(PhasePoint(0.0, 0.0));
    double n0 = psi.norm2();
    spec_transport::evolve_spectral(psi, 1.0);
    double n1 = psi.norm2();
    bool norm_ok = fabs(n1 - n0) < 1e-10;
    printf("spectral transport unitary:  %s\n", norm_ok ? "PASS" : "FAIL");
    psi.free();

    // Persistent format roundtrip
    bool roundtrip = persistent::verify::hybrid_roundtrip(ms.hyb.state);
    printf("persistent roundtrip:        %s\n", roundtrip ? "PASS" : "FAIL");
    bool checksum = persistent::verify::checksum_detects_corruption();
    printf("checksum integrity:          %s\n", checksum ? "PASS" : "FAIL");

    ms.free();
    printf("=== End Hardware Embedding ===\n");
}

static void verify_exact_decomposition() {
    printf("\n=== Exact Bidirectional Dictionary ===\n");
    using namespace exact;
    using measurement::DenseOp;

    int nq = 2, nth = 4, nrh = 3;
    int dd = 1 << nq;
    int dt = nth * nrh;
    int dim_hyb = dd * dt;
    double rmin = -5.0, rmax = 5.0;

    // Build test state
    HybridState psi;
    psi.init(nq, nth, nrh, rmin, rmax);
    for (int x = 0; x < dd; ++x) {
        TopologicalState xi;
        xi.init(nth, nrh, rmin, rmax);
        for (int i = 0; i < dt; ++i)
            xi.amp[i] = C64(cos(0.3 * x + 0.1 * i),
                            sin(0.2 * x + 0.05 * i));
        double n2 = xi.norm2();
        double inv = 1.0 / sqrt(n2 * dd);
        for (int i = 0; i < dt; ++i) xi.amp[i] = xi.amp[i] * inv;
        psi.inject(x, xi);
        xi.free();
    }

    // Vector roundtrips
    bool dr = vec::roundtrip_DR(psi);
    printf("D_vec ∘ R_vec = id:          %s\n", dr ? "PASS" : "FAIL");

    // Operator roundtrip
    DenseOp I_hyb = DenseOp::identity(dim_hyb);
    bool opr = op::roundtrip(I_hyb, dd, dt);
    printf("D_op ∘ R_op = id:            %s\n", opr ? "PASS" : "FAIL");

    // Matrix unit algebra
    bool mua = op::matrix_unit_algebra(dd, dt);
    printf("E matrix unit algebra:       %s\n", mua ? "PASS" : "FAIL");

    // Matrix unit resolution
    bool mur = op::matrix_unit_resolution(dd, dt);
    printf("Σ E_{xx} = I resolution:     %s\n", mur ? "PASS" : "FAIL");

    // Walsh roundtrip
    OpBlocks A = op::decompose(I_hyb, dd, dt);
    bool wr = walsh::roundtrip(A, nq);
    printf("Walsh ↔ Block roundtrip:     %s\n", wr ? "PASS" : "FAIL");
    A.free();

    // Density probability normalization
    DenseOp rho(dim_hyb);
    for (int i = 0; i < dim_hyb; ++i)
        for (int j = 0; j < dim_hyb; ++j)
            rho.at(i, j) = psi.amp[i] * psi.amp[j].conj();
    OpBlocks rho_blk = dens::decompose(rho, dd, dt);
    bool pn = dens::verify_prob_normalization(rho_blk);
    printf("Σ_x p(x) = 1:               %s\n", pn ? "PASS" : "FAIL");

    // Block expectation
    bool be = dens::verify_block_expectation(rho, I_hyb, dd, dt);
    printf("block expectation = Tr(ρO):  %s\n", be ? "PASS" : "FAIL");
    rho_blk.free();

    // Thermodynamic factorization
    DenseOp H_top(dt);
    for (int i = 0; i < dt; ++i)
        H_top.at(i, i) = C64(0.1 * i, 0);
    bool tf = thermo::verify_factorization(H_top, 1.0, nq);
    printf("Z_{A,n} = 2^n Z_A^top:       %s\n", tf ? "PASS" : "FAIL");

    // Grand verification
    verify::VerifyResult vr = verify::run_all(nq, nth, nrh);
    printf("grand verify: %d/%d passed\n", vr.passed, vr.total_checks);

    psi.free();
    printf("=== End Exact Decomposition ===\n");
}

static void verify_kernel_calculus() {
    printf("=== Kernel Calculus ===");
    printf(" \"The hybrid computer is now exactly a classical register");
    printf(" carrying topological operator payloads along its transitions.\"\n");
    using namespace topcomp::kc;

    const int nq = 1, nth = 4, nrh = 3;

    verify::KCVerifyResult vr = verify::run_all(nq, nth, nrh);

    printf("K(CD) = K(C)K(D):             %s\n", vr.star_multiplicative ? "PASS" : "FAIL");
    printf("K(C*) = K(C)*:                %s\n", vr.star_adjoint ? "PASS" : "FAIL");
    printf("column unitarity:             %s\n", vr.column_unitarity ? "PASS" : "FAIL");
    printf("row unitarity:                %s\n", vr.row_unitarity ? "PASS" : "FAIL");
    printf("projector kernel:             %s\n", vr.proj_kernel ? "PASS" : "FAIL");
    printf("projector idempotent:         %s\n", vr.proj_idempotent ? "PASS" : "FAIL");
    printf("projector resolution:         %s\n", vr.proj_resolution ? "PASS" : "FAIL");
    printf("path-sum expansion:           %s\n", vr.path_sum ? "PASS" : "FAIL");
    printf("perm lift kernel:             %s\n", vr.perm_kernel ? "PASS" : "FAIL");
    printf("perm composition:             %s\n", vr.perm_composition ? "PASS" : "FAIL");
    printf("perm unitary:                 %s\n", vr.perm_unitary ? "PASS" : "FAIL");
    printf("perm prob permutation:        %s\n", vr.perm_prob ? "PASS" : "FAIL");
    printf("ctrl kernel:                  %s\n", vr.ctrl_kernel ? "PASS" : "FAIL");
    printf("ctrl composition law:         %s\n", vr.ctrl_composition ? "PASS" : "FAIL");
    printf("ctrl unitary:                 %s\n", vr.ctrl_unitary ? "PASS" : "FAIL");
    printf("ctrl factorization:           %s\n", vr.ctrl_factorization ? "PASS" : "FAIL");
    printf("ctrl Weyl expansion:          %s\n", vr.ctrl_weyl ? "PASS" : "FAIL");
    printf("channel prob consistency:     %s\n", vr.channel_prob ? "PASS" : "FAIL");
    printf("channel deterministic case:   %s\n", vr.channel_deterministic ? "PASS" : "FAIL");
    printf("unitary lift kernel:          %s\n", vr.unitary_kernel ? "PASS" : "FAIL");
    printf("unitary lift probs:           %s\n", vr.unitary_probs ? "PASS" : "FAIL");
    printf("unitary lift unitarity:       %s\n", vr.unitary_unitarity ? "PASS" : "FAIL");
    printf("stochastic prob update:       %s\n", vr.stoch_prob_update ? "PASS" : "FAIL");
    printf("stochastic trace preservation:%s\n", vr.stoch_trace ? "PASS" : "FAIL");
    printf("Dec_prob transform law:       %s\n", vr.dec_prob_transform ? "PASS" : "FAIL");
    printf("Dec_full transform law:       %s\n", vr.dec_full_transform ? "PASS" : "FAIL");
    printf("compilation formula:          %s\n", vr.compilation_correct ? "PASS" : "FAIL");
    printf("irreversible extension:       %s\n", vr.irrev_extension ? "PASS" : "FAIL");
    printf("grand verify: %d/%d passed\n", vr.passed, vr.total_checks);
    printf("=== End Kernel Calculus ===\n");
}

void verify_semantic_kernel() {
    printf("\n=== Semantic Kernel (Doctrine) ===\n");
    int n = 1;
    double tol = 1e-4;

    // Doctrine grand verify
    doctrine::verify::DoctrineVerifyResult dvr =
        doctrine::verify::run_all(n, 4, 3, tol);
    printf("  vec roundtrip:            %s\n", dvr.vec_roundtrip ? "PASS" : "FAIL");
    printf("  op roundtrip:            %s\n", dvr.op_roundtrip ? "PASS" : "FAIL");
    printf("  walsh roundtrip:         %s\n", dvr.walsh_roundtrip ? "PASS" : "FAIL");
    printf("  star multiplicative:     %s\n", dvr.star_multiplicative ? "PASS" : "FAIL");
    printf("  star adjoint:            %s\n", dvr.star_adjoint ? "PASS" : "FAIL");
    printf("  ctrl to dense:           %s\n", dvr.ctrl_to_dense ? "PASS" : "FAIL");
    printf("  ctrl to kernel:          %s\n", dvr.ctrl_to_kernel ? "PASS" : "FAIL");
    printf("  ctrl to blocks:          %s\n", dvr.ctrl_to_blocks ? "PASS" : "FAIL");
    printf("  ctrl to weyl:            %s\n", dvr.ctrl_to_weyl ? "PASS" : "FAIL");
    printf("  dense→blocks→dense:      %s\n", dvr.dense_to_blocks_to_dense ? "PASS" : "FAIL");
    printf("  kernel→weyl→kernel:      %s\n", dvr.kernel_to_weyl_to_kernel ? "PASS" : "FAIL");
    printf("  ctrl compose:            %s\n", dvr.ctrl_compose ? "PASS" : "FAIL");
    printf("  ctrl inverse:            %s\n", dvr.ctrl_inverse ? "PASS" : "FAIL");
    printf("  ctrl factorization:      %s\n", dvr.ctrl_factorization ? "PASS" : "FAIL");
    printf("  ctrl extract:            %s\n", dvr.ctrl_extract ? "PASS" : "FAIL");
    printf("  face select scalar:      %s\n", dvr.face_select_scalar ? "PASS" : "FAIL");
    printf("  face select ctrl:        %s\n", dvr.face_select_ctrl ? "PASS" : "FAIL");
    printf("  face select perm:        %s\n", dvr.face_select_perm ? "PASS" : "FAIL");
    printf("  sparsity detection:      %s\n", dvr.sparsity_detection ? "PASS" : "FAIL");
    printf("  controlled detection:    %s\n", dvr.controlled_detection ? "PASS" : "FAIL");
    printf("doctrine: %d/%d passed\n", dvr.passed, dvr.total_checks);

    // Semantic passes verify
    semantic_pass::verify::PassVerifyResult pvr =
        semantic_pass::verify::run_all(n, 4, 3, tol);
    printf("\n  sparsity correct:        %s\n", pvr.sparsity_correct ? "PASS" : "FAIL");
    printf("  ctrl extraction:         %s\n", pvr.ctrl_extraction ? "PASS" : "FAIL");
    printf("  factorization valid:     %s\n", pvr.factorization_valid ? "PASS" : "FAIL");
    printf("  coherence prune safe:    %s\n", pvr.coherence_prune_safe ? "PASS" : "FAIL");
    printf("  channel classify:        %s\n", pvr.channel_classify ? "PASS" : "FAIL");
    printf("  weyl cancel:             %s\n", pvr.weyl_cancel ? "PASS" : "FAIL");
    printf("  projector absorb:        %s\n", pvr.projector_absorb ? "PASS" : "FAIL");
    printf("  face recommendation:     %s\n", pvr.face_recommendation ? "PASS" : "FAIL");
    printf("passes: %d/%d passed\n", pvr.passed, pvr.total_checks);

    printf("=== End Semantic Kernel ===\n");
}

void verify_exemplars() {
    printf("\n=== Exemplars ===\n");

    exemplar::AllExemplarResults er =
        exemplar::run_all_exemplars(1, 4, 3, 1e-4);
    er.print();

    printf("=== End Exemplars ===\n");
}

int main() {
    verify_constants();
    verify_mir_group();
    verify_arrows();
    verify_phase_space();
    verify_hilbert();
    verify_gates();
    verify_cocycle();
    verify_prng();
    verify_machine();
    verify_quantum_interference();
    verify_topological_interference();
    verify_weyl_interference();
    verify_deutsch_jozsa();
    verify_structure_detection();
    verify_gpu();
    verify_benchmark();
    verify_witt_sl2();
    verify_lie_rep();
    verify_compiler();
    verify_structure_finder();
    verify_noncommutative_dynamics();
    verify_multi_boundary();
    verify_hybrid_info();
    verify_mixed_operators();
    verify_quantum_algebra();
    verify_hardware_embedding();
    verify_exact_decomposition();
    verify_kernel_calculus();
    verify_semantic_kernel();
    verify_exemplars();
    return 0;
}
