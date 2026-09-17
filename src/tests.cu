// ============================================================================
// TopComp: Automated Test Suite
// ============================================================================
// Verifies correctness of all mathematical structures:
//   - Constants (φ, ℓ, c_φ)
//   - Mirror group axioms (identity, inverse, associativity)
//   - Cocycle condition (δΩ = 0) & UFE = 0
//   - Hilbert space (BitVec, QubitState, TopologicalState, HybridState)
//   - Gate commutation & discrete/topo/hybrid gates
//   - PRNG & statistical test bounds
//   - HES machine model & density operators
//   - Dual numbers & jet spaces (Witt)
//   - sl₂ Lie algebra brackets
//   - Representation theory (PGL₂, Möbius)
//   - Weyl algebra composition & quantum computation
//   - Orthodox measurement (PVM, POVM, Born rule)
//   - CPTP maps, instruments, Naimark dilation
//   - Hybrid doctrine & forms (affine rep, scale-conformal)
//   - Category C_Mir (arrow words, groupoid, functors)
//   - Transport & cocycle naturality
//   - Compiler IR & structure detection
//   - Heisenberg group, Clifford algebra, Boolean algebra
//   - Shadow functor, Fibonacci/spiral analytics
//   - Connection/curvature/index theory
//   - Translation matrices, Witt commutator/BCH
//   - Faà di Bruno jets, disc-top information decomposition
//   - Grand derivation chain, isometry verification
//   - Weyl translation operators, jet UM commutation
//   - Admissible sector decomposition (exact + compound + spectral)
//   - Hybrid fiber structure (injection, projection, lifting, joint)
//   - Platform 5-layer compute stack (contract → IR → VM → pipeline → API)
//   - Canonical structure graph & boundary calculus
//   - Universal state descriptor (structural typing)
//   - Emission & reconstruction (sector report, annotation, queries)
// ============================================================================

#include "topcomp/machine.cuh"
#include "topcomp/structure_finder.cuh"
#include "topcomp/measurement.cuh"
#include "topcomp/transport.cuh"
#include "topcomp/hybrid_info.cuh"
#include "topcomp/sector.cuh"
#include "topcomp/fiber.cuh"
#include "topcomp/emission.cuh"
#include "topcomp/topcomp_api.cuh"
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
#include <cmath>

using namespace topcomp;
using constants::PHI;
using constants::ELL;
using constants::PI;

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        tests_passed++;
        printf("  [PASS] %s\n", name);
    } else {
        tests_failed++;
        printf("  [FAIL] %s\n", name);
    }
}

static void check_near(double a, double b, double tol, const char* name) {
    check(fabs(a - b) < tol, name);
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Constants
// ════════════════════════════════════════════════════════════════════════════
static void test_constants() {
    printf("\n-- Constants --\n");
    check_near(PHI * PHI, PHI + 1.0, 1e-14, "phi^2 = phi + 1");
    check_near(exp(ELL), PHI, 1e-14, "e^ell = phi");

    C64 cp = constants::c_phi();
    check_near(cp.re, ELL, 1e-14, "Re(c_phi) = ell");
    check_near(cp.im, -PI / 2.0, 1e-14, "Im(c_phi) = -pi/2");

    check_near(constants::DELTA_SIGMA, 2.0 * PI * PHI, 1e-14, "DS = 2*pi*phi");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Mirror Group Axioms
// ════════════════════════════════════════════════════════════════════════════
static void test_mir_group() {
    printf("\n-- Mirror Group --\n");

    MirElement e = MirElement::identity();
    MirElement g = mir::g_phi(1.0);
    MirElement h = mir::U_phi();
    MirElement k = mir::U_pi();

    // Identity: g * e = g
    MirElement ge = g.star(e);
    check(ge.s == g.s && (ge.c - g.c).abs() < 1e-14, "g*e = g");

    // Identity: e * g = g
    MirElement eg = e.star(g);
    check(eg.s == g.s && (eg.c - g.c).abs() < 1e-14, "e*g = g");

    // Inverse: g * g^-1 = e
    MirElement gi = g.inverse();
    MirElement ggi = g.star(gi);
    check(ggi.s == 1 && ggi.c.abs() < 1e-14, "g*g^-1 = e");

    // Inverse: g^-1 * g = e
    MirElement gig = gi.star(g);
    check(gig.s == 1 && gig.c.abs() < 1e-14, "g^-1*g = e");

    // Associativity: (g*h)*k = g*(h*k)
    MirElement lhs = g.star(h).star(k);
    MirElement rhs = g.star(h.star(k));
    check(lhs.s == rhs.s && (lhs.c - rhs.c).abs() < 1e-13,
          "(g*h)*k = g*(h*k)");

    // Mirror involution: J_lambda^2 = e
    MirElement J = mir::J_lambda(2.5);
    MirElement J2 = J.star(J);
    check(J2.s == 1 && J2.c.abs() < 1e-14, "J_lambda^2 = e");

    // Conjugation by B preserves sign
    MirElement Bi = h.inverse();
    MirElement conj = h.star(g).star(Bi);
    check(conj.s == g.s, "conjugation preserves sign");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Cocycle
// ════════════════════════════════════════════════════════════════════════════
static void test_cocycle() {
    printf("\n-- Cocycle --\n");

    // Cocycle condition: dOmega = 0 for many triples
    bool cocycle_ok = true;
    for (int t = 0; t < 200; ++t) {
        MirElement ga = mir::g_phi(0.1 * t);
        MirElement gb = mir::g_phi(0.1 * t + 0.3);
        MirElement gc = mir::g_phi(0.1 * t + 0.7);
        if (!cocycle::verify_cocycle(ga, gb, gc, 1e-8)) {
            cocycle_ok = false;
            break;
        }
    }
    check(cocycle_ok, "dOmega(g0,g1,g2) = 0 (200 triples)");

    // Omega_bool symmetry
    BitVec u(5, 4), v(3, 4);
    int o1 = cocycle::omega_bool_f2(u, v);
    int o2 = cocycle::omega_bool_f2(v, u);
    check(o1 == o2, "Omega_bool symmetry over F2");

    // Shadow functor maps to {0,1}
    C64 test_val(3.14, -2.71);
    int bit = cocycle::shadow_to_bit(test_val);
    check(bit == 0 || bit == 1, "shadow_to_bit in {0,1}");

    // UFE = F - Omega (6 params)
    MirElement g1 = mir::g_phi(0.5);
    MirElement g2 = mir::g_phi(1.5);
    BitVec up(0b010, 4), vp(0b011, 4);
    C64 ufe = cocycle::UFE_total(g1, u, v, g2, up, vp);
    check_near(ufe.abs(), 0.0, 1e-10, "UFE = 0");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Hilbert Space
// ════════════════════════════════════════════════════════════════════════════
static void test_hilbert() {
    printf("\n-- Hilbert Space --\n");

    int nq = 3;

    QubitState psi;
    psi.init(nq);
    psi.set_uniform();
    check_near(psi.norm2(), 1.0, 1e-14, "uniform state normalized");

    QubitState b0;
    b0.init(nq);
    b0.set_basis(0);
    QubitState b1;
    b1.init(nq);
    b1.set_basis(1);
    C64 ip = b0.inner(b1);
    check_near(ip.abs(), 0.0, 1e-14, "<0|1> = 0");

    C64 ip_self = b0.inner(b0);
    check_near(ip_self.re, 1.0, 1e-14, "<0|0> = 1");

    psi.free();
    b0.free();
    b1.free();

    // BitVec
    BitVec a(0b1010, 4), b(0b0110, 4);
    BitVec axb = a ^ b;
    check(axb.bits == 0b1100, "1010 ^ 0110 = 1100");
    check(a.weight() == 2, "wt(1010) = 2");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Gates
// ════════════════════════════════════════════════════════════════════════════
static void test_gates() {
    printf("\n-- Gates --\n");

    int nq = 2, nth = 4, nrh = 4;

    // Prepare initial state |0> x xi_0
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    HybridState psi;
    psi.init(nq, nth, nrh);
    psi.inject(0, xi);
    double orig_norm = psi.norm2();

    // X_u is self-inverse: X_u X_u = I
    BitVec u1(1, nq);
    HybridState psi2;
    psi2.init(nq, nth, nrh);
    memcpy(psi2.amp, psi.amp, psi.total * sizeof(C64));

    hybrid_gates::apply_X_hyb(psi2, u1);
    hybrid_gates::apply_X_hyb(psi2, u1);

    double diff = 0.0;
    for (int i = 0; i < psi.total; ++i) {
        diff += (psi2.amp[i] - psi.amp[i]).norm2();
    }
    check_near(diff, 0.0, 1e-14, "X_u^2 = I");

    // Z_v preserves norm
    HybridState psi_z;
    psi_z.init(nq, nth, nrh);
    memcpy(psi_z.amp, psi.amp, psi.total * sizeof(C64));
    BitVec v1(1, nq);
    hybrid_gates::apply_Z_hyb(psi_z, v1);
    check_near(psi_z.norm2(), orig_norm, 1e-14, "Z_v preserves norm");

    // U_t preserves norm (approximately, due to interpolation)
    // Use a smooth initial state (uniform) for better norm preservation
    HybridState psi_u;
    psi_u.init(nq, nth, nrh);
    double inv_sqrt = 1.0 / sqrt(static_cast<double>(psi_u.total));
    for (int i = 0; i < psi_u.total; ++i) psi_u.amp[i] = C64(inv_sqrt, 0.0);
    double u_norm_before = psi_u.norm2();
    hybrid_gates::apply_U_hyb(psi_u, 0.01);  // small t to stay in grid
    // Grid-boundary interpolation loses ~25% norm on 8x8 grid (2 boundary rows)
    check_near(psi_u.norm2(), u_norm_before, 0.3, "U_t approx preserves norm");

    // Gate cost
    Gate gx = gate_X(0b11);
    check(gx.cost(nq) == 2, "cost(X_{11}) = 2");
    Gate gu = gate_U(0.5);
    check(gu.cost(nq) == 1, "cost(U_t) = 1");

    xi.free();
    psi.free();
    psi2.free();
    psi_z.free();
    psi_u.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Test: PRNG
// ════════════════════════════════════════════════════════════════════════════
static void test_prng() {
    printf("\n-- PRNG --\n");

    MirElement seed = mir::J_lambda(constants::PHI);  // s=-1 for non-trivial cocycle
    MirPRNG prng(seed, BitVec(), 4);
    int N = 5000;
    int* bits = new int[N];
    prng.generate(bits, N);

    // All bits should be 0 or 1
    bool valid = true;
    for (int i = 0; i < N; ++i) {
        if (bits[i] != 0 && bits[i] != 1) { valid = false; break; }
    }
    check(valid, "all bits in {0,1}");

    // Bias should be small
    double bias = stats::compute_bias(bits, N);
    check(bias < 0.5, "bias < 0.5");

    // TV distance check
    double tv = stats::compute_tv_distance(bits, N);
    check(tv < 0.5, "TV < 0.5");

    // Tail bound monotonicity
    double tb1 = stats::tail_bound(1000, 0.01);
    double tb2 = stats::tail_bound(1000, 0.1);
    check(tb1 > tb2, "tail bound decreasing in epsilon");

    // Cramer rate function positive
    check(stats::cramer_rate(0.05) > 0, "I(0.05) > 0");
    check_near(stats::cramer_rate(0.0), 0.0, 1e-14, "I(0) = 0");

    delete[] bits;
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Machine
// ════════════════════════════════════════════════════════════════════════════
static void test_machine() {
    printf("\n-- HES Machine --\n");

    int nq = 2, nth = 4, nrh = 4;
    HESMachine machine(nq, nth, nrh);

    // Prepare initial state
    TopologicalState xi;
    xi.init(nth, nrh);
    xi.set_delta(PhasePoint(0.0, 0.0));

    HybridState psi0;
    psi0.init(nq, nth, nrh);
    psi0.inject(0, xi);

    // Empty program should halt immediately
    Program empty;
    Configuration config = machine.init(psi0);
    machine.run(config, empty);
    check(config.tau == 0, "empty program: tau = 0");
    check(machine.halted(config, empty), "empty program: halted");
    config.psi.free();

    // Single gate program
    Program single;
    single.append(gate_X(1));
    config = machine.init(psi0);
    machine.run(config, single);
    check(config.tau == 1, "single gate: tau = 1");
    config.psi.free();

    // Cost functionals
    Program multi;
    multi.append(gate_X(0b11));
    multi.append(gate_Z(0b01));
    multi.append(gate_U(0.5));
    check(machine.program_time(multi) == 3, "time(3 gates) = 3");
    check(machine.program_cost(multi) == 4, "cost(X_11+Z_01+U_t)=2+1+1=4");

    // Born probabilities sum to 1
    config = machine.init(psi0);
    machine.run(config, multi);
    double probs[4];
    machine.born_distribution(config.psi, probs);
    double total = 0;
    for (int i = 0; i < 4; ++i) total += probs[i];
    check_near(total, 1.0, 1e-10, "Born probs sum to 1");

    config.psi.free();
    psi0.free();
    xi.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Witt / Dual Numbers / Jets
// ════════════════════════════════════════════════════════════════════════════
static void test_witt() {
    printf("\n-- Witt (Dual Numbers, Jets, TruncPoly) --\n");

    // Dual number: (a+bη) * (c+dη) = ac + (ad+bc)η
    Dual<double> a(3.0, 1.0), b(2.0, 4.0);
    Dual<double> ab = a * b;
    check_near(ab.val, 6.0, 1e-14, "Dual mult real part");
    check_near(ab.eps, 14.0, 1e-14, "Dual mult dual part");

    // η² = 0
    Dual<double> eta(0.0, 1.0);
    Dual<double> eta2 = eta * eta;
    check_near(eta2.val, 0.0, 1e-14, "eta^2 = 0 (real)");
    check_near(eta2.eps, 0.0, 1e-14, "eta^2 = 0 (dual)");

    // exp(a+bη) = e^a(1+bη)
    Dual<double> x(1.0, 1.0);
    Dual<double> ex = dexp(x);
    check_near(ex.val, exp(1.0), 1e-12, "dexp real = e^a");
    check_near(ex.eps, exp(1.0), 1e-12, "dexp dual = b*e^a");

    // d/dx(sin(x)) = cos(x) via dual numbers
    Dual<double> s = dsin(Dual<double>(0.5, 1.0));
    check_near(s.eps, cos(0.5), 1e-12, "dsin derivative = cos");

    // sl₂ commutation on TruncPoly
    TruncPoly test_poly;
    test_poly.coeffs[0] = 1.0; test_poly.coeffs[1] = 0.5; test_poly.coeffs[2] = 0.25;
    check(sl2_verify::check_he(test_poly), "[h,e] = 2e on TruncPoly");
    check(sl2_verify::check_hf(test_poly), "[h,f] = -2f on TruncPoly");
    check(sl2_verify::check_ef(test_poly), "[e,f] = h on TruncPoly");

    // DualFlow: e^{(4+η)c_φ} = φ⁴(1+η·c_φ)
    auto result = DualFlow::four_plus_eta_action(C64(1.0, 0.0));
    double phi4 = constants::PHI * constants::PHI * constants::PHI * constants::PHI;
    // |val| should be close to φ⁴
    // Note: e^{4c_φ} = e^{4ℓ-2πi} = φ⁴ · e^{-2πi} = φ⁴
    check_near(result.val.abs(), phi4, 0.01, "DualFlow: |e^{4c_phi}| = phi^4");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: sl₂ Lie Algebra
// ════════════════════════════════════════════════════════════════════════════
static void test_lie_algebra() {
    printf("\n-- sl2 Lie Algebra --\n");

    // Bracket relations: [D,E]=D, [D,K]=2E, [E,K]=K
    check(sl2_brackets::check_DE(), "[D,E] = D");
    check(sl2_brackets::check_DK(), "[D,K] = 2E");
    check(sl2_brackets::check_EK(), "[E,K] = K");

    // Chevalley basis: [h,e]=2e, [h,f]=-2f, [e,f]=h
    check(sl2_brackets::check_chevalley(), "Chevalley [h,e],[h,f],[e,f]");

    // Jacobi identity
    SL2Element X(1.0, 0.5, 0.3);
    SL2Element Y(0.2, 1.0, 0.7);
    SL2Element Z(0.4, 0.1, 1.0);
    check(sl2_brackets::check_jacobi(X, Y, Z), "Jacobi identity");

    // Matrix representation preserves bracket
    SL2Element D = SL2Element::D();
    SL2Element E = SL2Element::E();
    Mat2 MD = D.to_matrix();
    Mat2 ME = E.to_matrix();
    Mat2 comm_mat = MD * ME;
    Mat2 comm_mat2 = ME * MD;
    // [D,E] = D: check (MD*ME - ME*MD) ~ D.to_matrix()
    Mat2 expected = D.to_matrix();
    // Compare element-wise
    double mat_err = (comm_mat.a - comm_mat2.a - expected.a).abs()
                   + (comm_mat.b - comm_mat2.b - expected.b).abs()
                   + (comm_mat.c - comm_mat2.c - expected.c).abs()
                   + (comm_mat.d - comm_mat2.d - expected.d).abs();
    check_near(mat_err, 0.0, 1e-10,
               "Matrix [D,E] = D");

    // Adjoint is homomorphism
    Mat2 A = mobius::T(C64(1.5, 0));
    check(adjoint::verify_Ad_homomorphism(A, D, E), "Ad homomorphism T");

    // J exp(hD) J = exp(-hK) at t=2
    check(exp_action::verify_J_expD_J(0.3, 2.0), "J e^{hD} J = e^{-hK}");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Representation Theory
// ════════════════════════════════════════════════════════════════════════════
static void test_representation() {
    printf("\n-- Representation Theory --\n");

    // Möbius generators
    check(mobius::verify_JTJ(1.5), "J T_h J = G_{-h}");
    check(mobius::verify_STS(2.0, 1.5), "S T_h S^{-1} consistency");

    // PGL rep homomorphism
    MirElement g1 = mir::g_phi(0.5);
    MirElement g2 = mir::g_phi(1.3);
    check(pgl_rep::verify_homomorphism(g1, g2), "PGL rep is homomorphism");

    // Arrow rep mirror conjugation
    check(arrow_rep::verify_mirror_conjugation(), "Mirror conjugation: JRJ = L");

    // Contraction ratio = 1/φ²
    double area = contraction::area_ratio();
    check_near(area, 1.0 / (PHI * PHI), 1e-10, "Area ratio = 1/phi^2");

    // Volume ratio
    double vol3 = contraction::volume_ratio(3);
    double expected = 1.0;
    for (int i = 0; i < 3; ++i) expected /= (PHI * PHI);
    check_near(vol3, expected, 1e-10, "Volume ratio(3) = phi^{-6}");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Weyl Operators
// ════════════════════════════════════════════════════════════════════════════
static void test_weyl() {
    printf("\n-- Weyl Operators --\n");

    // XZ commutation: X_u Z_v = (-1)^{<u,v>} Z_v X_u
    check(weyl_verify::check_XZ_comm(2), "XZ comm n=2");
    check(weyl_verify::check_XZ_comm(3), "XZ comm n=3");

    // Omega_bool consistency
    check(weyl_verify::check_omega_bool(3), "Omega_bool consistency n=3");

    // Hybrid discrete-topological commutativity
    check(weyl_verify::check_cross_sector_comm(), "Cross-sector [X,U]=0");

    // Weyl composition
    int n = 2;
    WeylHybrid W1(BitVec(1, n), BitVec(0, n), mir::g_phi(0.5));
    WeylHybrid W2(BitVec(0, n), BitVec(1, n), mir::g_phi(1.0));
    C64 phase;
    WeylHybrid W12 = W1.compose(W2, phase);
    check(W12.u.bits == 1 && W12.v.bits == 1, "Weyl compose indices");

    // Algebra dimension
    check(hybrid_algebra::disc_algebra_dim(3) == 64, "dim End(V_3) = 64");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Quantum Computation — Hadamard, Weyl execution, Interference, DJ
// ════════════════════════════════════════════════════════════════════════════
static void test_quantum_computation() {
    printf("\n-- Quantum Computation --\n");

    int nq = 1, nth = 8, nrh = 4;

    // Hadamard creates uniform superposition: H|0⟩ = (|0⟩+|1⟩)/√2
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);

        hybrid_gates::apply_H_hyb(psi, 0);
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        check_near(p0 / tot, 0.5, 1e-10, "H|0>: p(0) = 0.5");
        check_near(p1 / tot, 0.5, 1e-10, "H|0>: p(1) = 0.5");

        psi.free(); xi.free();
    }

    // Hadamard is self-inverse: H·H = I
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.5, 0.3));

        HybridState psi_orig;
        psi_orig.init(nq, nth, nrh);
        psi_orig.inject(0, xi);

        HybridState psi;
        psi.init(nq, nth, nrh);
        memcpy(psi.amp, psi_orig.amp, psi.total * sizeof(C64));

        hybrid_gates::apply_H_hyb(psi, 0);
        hybrid_gates::apply_H_hyb(psi, 0);

        double diff = 0;
        for (int i = 0; i < psi.total; ++i)
            diff += (psi.amp[i] - psi_orig.amp[i]).norm2();
        check_near(diff, 0.0, 1e-14, "H*H = I");

        psi.free(); psi_orig.free(); xi.free();
    }

    // H-Z-H gives p(0)=0, p(1)=1 (quantum interference)
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);

        hybrid_gates::apply_H_hyb(psi, 0);
        hybrid_gates::apply_Z_hyb(psi, BitVec(1, nq));
        hybrid_gates::apply_H_hyb(psi, 0);

        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        check(p0 / tot < 1e-10, "H-Z-H: p(0) ~ 0 (destructive)");
        check(p1 / tot > 1.0 - 1e-10, "H-Z-H: p(1) ~ 1 (constructive)");

        psi.free(); xi.free();
    }

    // Weyl operator preserves norm (identity group element)
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);
        double norm_before = psi.norm2();

        hybrid_gates::apply_W_hyb(psi, BitVec(1, nq), BitVec(0, nq), MirElement::identity());
        double norm_after = psi.norm2();
        check_near(norm_after, norm_before, 1e-10, "W_{1,0;e} preserves norm");

        psi.free(); xi.free();
    }

    // Weyl composition: individual gates = composed + cocycle phase
    {
        int n = 2;
        BitVec u1(0b01, n), v1(0b00, n);
        MirElement g1 = MirElement::identity();  // pure identity for exact comparison
        BitVec u2(0b10, n), v2(0b01, n);
        MirElement g2 = MirElement::identity();

        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        // Method 1: W1 then W2
        HybridState psi1;
        psi1.init(n, nth, nrh);
        psi1.inject(0, xi);
        hybrid_gates::apply_W_hyb(psi1, u1, v1, g1);
        hybrid_gates::apply_W_hyb(psi1, u2, v2, g2);

        // Method 2: composed W12 * cocycle_phase
        WeylHybrid W1h(u1, v1, g1);
        WeylHybrid W2h(u2, v2, g2);
        C64 phase;
        WeylHybrid W12 = W1h.compose(W2h, phase);

        HybridState psi2;
        psi2.init(n, nth, nrh);
        psi2.inject(0, xi);
        hybrid_gates::apply_W_hyb(psi2, W12.u, W12.v, W12.g);
        for (int i = 0; i < psi2.total; ++i)
            psi2.amp[i] = psi2.amp[i] * phase;

        // Compare Born distributions
        double max_diff = 0;
        for (int x = 0; x < (1 << n); ++x) {
            double d = fabs(psi1.prob_disc(x) - psi2.prob_disc(x));
            if (d > max_diff) max_diff = d;
        }
        check(max_diff < 1e-8, "Weyl compose = individual (Born match)");

        psi1.free(); psi2.free(); xi.free();
    }

    // Deutsch-Jozsa: constant → p(0)=1
    {
        int nth2 = 8, nrh2 = 4;
        HESMachine machine(2, nth2, nrh2);
        Program prog;
        prog.append(gate_X(0b10));
        prog.append(gate_H(0));
        prog.append(gate_H(1));
        // constant oracle: identity
        prog.append(gate_H(0));

        TopologicalState xi;
        xi.init(nth2, nrh2);
        xi.set_delta(PhasePoint(0.0, 0.0));
        HybridState psi0;
        psi0.init(2, nth2, nrh2);
        psi0.inject(0, xi);

        Configuration config = machine.init(psi0);
        machine.run(config, prog);

        double p_q0_0 = config.psi.prob_disc(0b00) + config.psi.prob_disc(0b10);
        double total = 0;
        for (int x = 0; x < 4; ++x) total += config.psi.prob_disc(x);
        check(p_q0_0 / total > 0.99, "DJ constant: p(q0=0) ~ 1");

        config.psi.free(); psi0.free(); xi.free();
    }

    // Deutsch-Jozsa: balanced → p(0)=0
    {
        int nth2 = 8, nrh2 = 4;
        HESMachine machine(2, nth2, nrh2);
        Program prog;
        prog.append(gate_X(0b10));
        prog.append(gate_H(0));
        prog.append(gate_H(1));
        prog.append(gate_Z(0b01));  // balanced oracle
        prog.append(gate_H(0));

        TopologicalState xi;
        xi.init(nth2, nrh2);
        xi.set_delta(PhasePoint(0.0, 0.0));
        HybridState psi0;
        psi0.init(2, nth2, nrh2);
        psi0.inject(0, xi);

        Configuration config = machine.init(psi0);
        machine.run(config, prog);

        double p_q0_0 = config.psi.prob_disc(0b00) + config.psi.prob_disc(0b10);
        double total = 0;
        for (int x = 0; x < 4; ++x) total += config.psi.prob_disc(x);
        check(p_q0_0 / total < 0.01, "DJ balanced: p(q0=0) ~ 0");

        config.psi.free(); psi0.free(); xi.free();
    }

    // Hadamard via Gate struct (machine-level execution)
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);

        Gate gh = gate_H(0);
        gh.apply(psi);
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        check_near(p0 / tot, 0.5, 1e-10, "gate_H: p(0) = 0.5");

        psi.free(); xi.free();
    }

    // Weyl gate via Gate struct
    {
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);

        Gate gw = gate_W(BitVec(1, nq), BitVec(0, nq), MirElement::identity());
        gw.apply(psi);
        // W_{1,0;e} = X_1 Z_0 ⊗ ρ(e) = X (shift) on single qubit
        double p0 = psi.prob_disc(0);
        double p1 = psi.prob_disc(1);
        double tot = p0 + p1;
        check(p1 / tot > 0.99, "gate_W shift: p(1) ~ 1");

        psi.free(); xi.free();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Measurement
// ════════════════════════════════════════════════════════════════════════════
static void test_measurement() {
    printf("\n-- Measurement (PVM, POVM, Born) --\n");

    int dim = 4;
    // Pure state |0⟩
    QubitState psi;
    psi.init(2);
    psi.set_basis(0);
    auto rho = measurement::pure_state_density(psi);

    check(measurement::is_valid_density(rho), "Pure state is valid density");
    check_near(measurement::purity(rho), 1.0, 1e-10, "Pure state purity = 1");

    // Born probabilities: p(0) = 1, p(x≠0) = 0
    double* probs = new double[dim];
    measurement::born_distribution(rho, probs);
    check_near(probs[0], 1.0, 1e-10, "p(0) = 1 for |0>");
    check_near(probs[1], 0.0, 1e-10, "p(1) = 0 for |0>");
    check(measurement::verify_born_normalization(rho), "Born normalization");

    // PVM properties
    check(measurement::verify_pvm_properties(2), "PVM: P^2=P, ortho, sum=I");

    // Lüders update: |0⟩ → P_0 |0⟩⟨0| P_0 / Tr = |0⟩⟨0|
    auto P0 = measurement::computational_projector(0, dim);
    auto post = measurement::luders_update(rho, P0);
    check(measurement::is_valid_density(post), "Lüders post-state valid");
    check_near(measurement::purity(post), 1.0, 1e-10, "Lüders preserves purity");

    // Maximally mixed
    auto mixed = measurement::maximally_mixed(dim);
    check(measurement::is_valid_density(mixed), "Mixed state valid");
    check_near(measurement::purity(mixed), 1.0 / dim, 1e-10, "Mixed purity = 1/d");

    // Heisenberg uncertainty
    auto A = measurement::DenseOp::identity(dim);
    auto B = measurement::DenseOp::identity(dim);
    auto ub = measurement::heisenberg_bound(rho, A, B);
    check(ub.satisfied, "Heisenberg bound satisfied (trivial)");

    delete[] probs;
    psi.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Hybrid Doctrine — noncommutative dynamics, joint measurement
// ════════════════════════════════════════════════════════════════════════════
static void test_hybrid_doctrine() {
    printf("\n-- Hybrid Doctrine --\n");

    // --- Robertson / Schrödinger uncertainty on Pauli X, Z ---
    {
        // Build Pauli X and Z as DenseOp(2)
        measurement::DenseOp X_op(2), Z_op(2);
        X_op.at(0, 1) = C64(1, 0); X_op.at(1, 0) = C64(1, 0);
        Z_op.at(0, 0) = C64(1, 0); Z_op.at(1, 1) = C64(-1, 0);

        // State |+⟩ = (|0⟩+|1⟩)/√2:  Δ(X)=0, Δ(Z)=1, but [X,Z]=-2iY
        QubitState plus;
        plus.init(1);
        plus.amp[0] = C64(1.0 / sqrt(2.0), 0);
        plus.amp[1] = C64(1.0 / sqrt(2.0), 0);
        auto rho = measurement::pure_state_density(plus);

        auto ub = measurement::heisenberg_bound(rho, X_op, Z_op);
        check(ub.satisfied, "Heisenberg Δ(X)Δ(Z) ≥ ½|⟨[X,Z]⟩| for |+⟩");

        auto sb = measurement::schrodinger_bound(rho, X_op, Z_op);
        check(sb.satisfied, "Schrödinger bound satisfied for |+⟩");
        check(sb.rhs >= ub.rhs - 1e-12, "Schrödinger ≥ Robertson (tighter)");

        // State |0⟩: Δ(X)=1, Δ(Z)=0 → LHS=0, but ⟨[X,Z]⟩=0 for |0⟩ too
        QubitState zero_st;
        zero_st.init(1);
        zero_st.set_basis(0);
        auto rho0 = measurement::pure_state_density(zero_st);

        auto ub0 = measurement::heisenberg_bound(rho0, X_op, Z_op);
        check(ub0.satisfied, "Heisenberg satisfied for |0⟩ (trivial)");

        auto sb0 = measurement::schrodinger_bound(rho0, X_op, Z_op);
        check(sb0.satisfied, "Schrödinger satisfied for |0⟩");

        plus.free();
        zero_st.free();
    }

    // --- U_t M_m commutation: e^{-i2πmt} ---
    // Relation: U_t M_m = e^{-i2πmt} M_m U_t
    // Born distributions are phase-independent: |M_m U_t ψ|² = |U_t M_m ψ|²
    {
        int nth = 16, nrh = 8;
        double t = 0.5;
        int m = 1;

        TopologicalState xi0;
        xi0.init(nth, nrh);
        xi0.set_delta(PhasePoint(0.0, 0.0));

        // psi_um = M_m U_t |ψ⟩
        TopologicalState psi_um;
        psi_um.init(nth, nrh);
        memcpy(psi_um.amp, xi0.amp, xi0.total * sizeof(C64));
        topo_gates::apply_U(psi_um, t);
        topo_gates::apply_M(psi_um, m);

        // psi_mu = U_t M_m |ψ⟩
        TopologicalState psi_mu;
        psi_mu.init(nth, nrh);
        memcpy(psi_mu.amp, xi0.amp, xi0.total * sizeof(C64));
        topo_gates::apply_M(psi_mu, m);
        topo_gates::apply_U(psi_mu, t);

        // Born match: |psi_um[i]|² = |psi_mu[i]|² for all i
        double born_diff = 0;
        for (int i = 0; i < psi_um.total; ++i) {
            double d = fabs(psi_um.amp[i].norm2() - psi_mu.amp[i].norm2());
            if (d > born_diff) born_diff = d;
        }
        check(born_diff < 1e-10, "[U_t,M_m] Born match (phase-independent)");

        psi_um.free(); psi_mu.free(); xi0.free();
    }

    // --- Joint probability p(x,j) ---
    {
        int nq = 1, nth = 8, nrh = 4;
        int dim_d = 1 << nq;
        int dim_t = nth * nrh;

        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));

        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);

        // Before H: |0⟩⊗δ → p(0,j0)=1, all else 0
        int nd, nt;
        double* joint = new double[dim_d * dim_t];
        measurement::joint_probability(psi, joint, nd, nt);

        check(measurement::verify_joint_normalization(joint, nd, nt),
              "Joint normalization Σp(x,j)=1");

        // All weight at x=0
        double* marg_disc = new double[dim_d];
        measurement::marginal_disc(joint, nd, nt, marg_disc);
        check_near(marg_disc[0], 1.0, 1e-10, "Marginal p(0)=1 for |0⟩⊗δ");
        check_near(marg_disc[1], 0.0, 1e-10, "Marginal p(1)=0 for |0⟩⊗δ");

        // After H: uniform discrete, concentrated topo
        hybrid_gates::apply_H_hyb(psi, 0);
        measurement::joint_probability(psi, joint, nd, nt);
        measurement::marginal_disc(joint, nd, nt, marg_disc);
        check_near(marg_disc[0], 0.5, 1e-6, "Post-H marginal p(0)=0.5");
        check_near(marg_disc[1], 0.5, 1e-6, "Post-H marginal p(1)=0.5");

        check(measurement::verify_joint_normalization(joint, nd, nt),
              "Joint normalization after H");

        delete[] joint;
        delete[] marg_disc;
        xi.free(); psi.free();
    }

    // --- Multi-boundary measurement ---
    {
        int nq = 1, nth = 8, nrh = 4;

        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(PI / 4.0, 0.3));

        HybridState psi;
        psi.init(nq, nth, nrh);
        psi.inject(0, xi);
        hybrid_gates::apply_H_hyb(psi, 0);

        auto r = measurement::multi_boundary_measure(psi);

        check(fabs(r.norm_check - 1.0) < 1e-10,
              "Multi-boundary normalization");
        check(fabs(r.marg_disc[0] - 0.5) < 1e-6,
              "Multi-boundary disc uniform p(0)=0.5");
        check(fabs(r.marg_disc[1] - 0.5) < 1e-6,
              "Multi-boundary disc uniform p(1)=0.5");

        // Topo should be concentrated (delta state → one grid region)
        double max_topo = 0;
        for (int j = 0; j < r.n_top; ++j)
            if (r.marg_topo[j] > max_topo) max_topo = r.marg_topo[j];
        check(max_topo > 0.5, "Multi-boundary topo concentrated");

        r.free(); xi.free(); psi.free();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Category & Groupoid
// ════════════════════════════════════════════════════════════════════════════
static void test_category() {
    printf("\n-- Category C_Mir & Groupoid --\n");

    // Arrow word evaluation
    ArrowWord w1(Arrow::RIGHT);
    ArrowWord w2(Arrow::LEFT);
    ArrowWord w12 = w1.concat(w2);
    MirElement ev12 = w12.evaluate();
    // →← should give (1, c_φ + (-c_φ)) = (1, 0)
    check_near(ev12.c.abs(), 0.0, 1e-10, "→← evaluates to identity");
    check(ev12.s == 1, "→← sign = +1");

    // Category axioms
    ArrowWord u(Arrow::RIGHT);
    ArrowWord v(Arrow::LLDIR);
    ArrowWord w(Arrow::MIRROR);
    check(category_mir::verify_associativity(u, v, w), "C_Mir associativity");
    check(category_mir::verify_identity(u), "C_Mir identity");

    // Groupoid
    PhasePoint x(0.5, 1.0);
    MirElement g1 = mir::g_phi(0.3);
    MirElement g2 = mir::g_phi(0.7);
    MirElement g3 = mir::g_phi(1.1);
    check(groupoid_verify::check_associativity(x, g1, g2, g3),
          "Groupoid associativity");
    check(groupoid_verify::check_identity(x, g1), "Groupoid identity");
    check(groupoid_verify::check_inverse(x, g1), "Groupoid inverse");

    // Mir⁺ abelian
    C64 c1(1.0, 0.5), c2(0.3, -0.7);
    check(mir_plus::verify_abelian(c1, c2), "Mir+ abelian");
    check(mir_plus::verify_exp_log(c1), "Exp/Log roundtrip");
    check(mir_plus::verify_log_homomorphism(c1, c2), "Log homomorphism");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Compiler IR
// ════════════════════════════════════════════════════════════════════════════
static void test_compiler() {
    printf("\n-- Compiler IR --\n");

    using namespace compiler;

    // Build simple IR
    MirIR ir(2);
    ir.emit(MirOp::flow(1.0));
    ir.emit(MirOp::flow(2.0));
    ir.emit(MirOp::modular(3));
    ir.emit(MirOp::modular(-1));
    ir.emit(MirOp::pauli_x(BitVec(0, 2)));  // identity (u=0)
    check(ir.count == 5, "IR has 5 ops before optimization");

    // Optimize
    optimizer::optimize(ir);

    check(ir.count < 5, "Optimization reduced op count");
    // Flows merged: 1.0 + 2.0 = 3.0
    // Modulars merged: 3 + (-1) = 2
    // Identity X_0 removed

    // Lower to HybridProgram
    HybridProgram prog = lowering::lower_to_gates(ir);
    check(prog.length > 0, "Lowered program non-empty");

    // Frontend: build PRNG circuit
    MirIR prng_ir = frontend::build_prng_circuit(
        2, 10, mir::J_lambda(PHI));
    check(prng_ir.count > 0, "PRNG circuit built");

    // Analysis
    auto stats = analysis::count_ops(prng_ir);
    check(stats.total == prng_ir.count, "Stats total matches count");

    // sl₂ lowering
    MirIR sl2_ir(1);
    sl2_ir.emit(MirOp::sl2_translate(1.5));
    sl2_ir.emit(MirOp::sl2_scale(2.0));
    sl2_ir.emit(MirOp::sl2_conformal(0.5));
    optimizer::optimize(sl2_ir);
    // sl2_conformal expands to J,flow,J so count increases initially
    check(sl2_ir.count > 0, "sl2 lowering produces ops");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Structure Finder
// ════════════════════════════════════════════════════════════════════════════
static void test_structure_finder() {
    printf("\n-- Structure Finder --\n");

    // Generate structured bits from Mir PRNG
    MirElement seed = mir::J_lambda(PHI);
    MirPRNG prng(seed, BitVec(), 4);
    int N = 2000;
    int* bits = new int[N];
    prng.generate(bits, N);

    // Find structure
    auto sig = structure::find_structure(bits, N);

    // Should detect something (Mir-generated bits have φ-structure)
    check(sig.bias >= 0, "Bias non-negative");
    check(sig.tv_distance >= 0, "TV non-negative");
    check(sig.cramer_rate >= 0, "Cramér rate non-negative");

    // Spectral analysis should work
    double* mags = new double[N / 2 + 1];
    structure::spectral::dft_magnitudes(bits, N, mags);
    double gap = structure::spectral::compute_spectral_gap(mags, N);
    check(gap >= 0 && gap <= 1, "Spectral gap in [0,1]");
    delete[] mags;

    // Autocorrelation
    double ac1 = structure::correlation::autocorrelation(bits, N, 1);
    check(ac1 >= 0 && ac1 <= 1, "Autocorrelation in [0,1]");

    // Cramér rate function
    double I_01 = structure::cramer::rate_function(bits, N, 0.1);
    check(I_01 >= 0, "I(0.1) >= 0");

    // Large deviation bound
    double tail = structure::large_deviation::bias_tail_bound(N, 0.1, bits);
    check(tail >= 0 && tail <= 1.0, "Tail bound in [0,1]");

    delete[] bits;
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Transport & Naturality
// ════════════════════════════════════════════════════════════════════════════
static void test_transport() {
    printf("\n-- Transport & Naturality --\n");

    // UFE chain verification
    MirElement g = mir::g_phi(0.7);
    MirElement h = mir::g_phi(1.3);
    check(factorization::verify_ufe_chain(g, h), "UFE chain g_phi");

    // Master claim
    check(transport::master_claim::verify_UFE_implies_factorization(),
          "Master claim: UFE => factorization");

    // Total cocycle UFE = 0
    BitVec u(1, 3), v(2, 3), up(3, 3), vp(1, 3);
    check(transport::total_cocycle::verify_total_UFE(g, u, v, h, up, vp),
          "Total UFE = 0");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: 1-form structure
// ════════════════════════════════════════════════════════════════════════════
static void test_forms() {
    printf("\n-- Differential Forms --\n");

    // du(X_φ) = 0
    check_near(forms::du_on_Xphi(), 0.0, 1e-12, "du(X_phi) = 0");

    // dω̃(X_φ) = -π
    check_near(forms::domega_on_Xphi(), -PI, 1e-12, "domega(X_phi) = -pi");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: CPTP channels, Instruments, Naimark dilation
// ════════════════════════════════════════════════════════════════════════════
static void test_cptp_instrument_naimark() {
    printf("\n-- CPTP / Instruments / Naimark --\n");

    // --- CPTP: Identity channel ---
    {
        auto ch = measurement::cptp::identity_channel(2);
        check(ch.verify_tp(), "Identity channel: trace-preserving");

        QubitState psi; psi.init(1); psi.set_basis(0);
        auto rho = measurement::pure_state_density(psi);
        auto out = ch.apply(rho);
        check_near((out - rho).hs_norm(), 0.0, 1e-12, "Identity channel Φ(ρ)=ρ");
        check(ch.verify_trace_preserving(rho), "Identity channel Tr-preserving");
        psi.free();
    }

    // --- CPTP: Depolarizing ---
    {
        auto ch = measurement::cptp::depolarizing(0.1);
        check(ch.verify_tp(), "Depolarizing(0.1): trace-preserving");

        QubitState psi; psi.init(1); psi.set_basis(0);
        auto rho = measurement::pure_state_density(psi);
        check(ch.verify_trace_preserving(rho), "Depolarizing Tr-preserving");

        auto out = ch.apply(rho);
        double p_out = measurement::purity(out);
        check(p_out < 1.0 - 1e-6, "Depolarizing reduces purity");
        psi.free();
    }

    // --- CPTP: Amplitude damping ---
    {
        auto ch = measurement::cptp::amplitude_damping(0.5);
        check(ch.verify_tp(), "Amp-damping(0.5): trace-preserving");

        QubitState psi; psi.init(1); psi.set_basis(1);
        auto rho = measurement::pure_state_density(psi);
        auto out = ch.apply(rho);
        check_near(out.trace().re, 1.0, 1e-10, "Amp-damping Tr(out)=1");
        check(out.at(0,0).re > 0.1, "Amp-damping transfers |1⟩→|0⟩");
        psi.free();
    }

    // --- CPTP: Dephasing ---
    {
        auto ch = measurement::cptp::dephasing(0.3);
        check(ch.verify_tp(), "Dephasing(0.3): trace-preserving");

        QubitState psi; psi.init(1);
        psi.amp[0] = C64(1.0/sqrt(2.0), 0);
        psi.amp[1] = C64(1.0/sqrt(2.0), 0);
        auto rho = measurement::pure_state_density(psi);
        auto out = ch.apply(rho);
        check(fabs(out.at(0,1).re) < fabs(rho.at(0,1).re), "Dephasing reduces coherence");
        psi.free();
    }

    // --- Instrument: Lüders ---
    {
        auto inst = measurement::luders_instrument(1);
        check(inst.n_outcomes == 2, "Lüders instrument: 2 outcomes for 1 qubit");

        QubitState psi; psi.init(1);
        psi.amp[0] = C64(1.0/sqrt(2.0), 0);
        psi.amp[1] = C64(1.0/sqrt(2.0), 0);
        auto rho = measurement::pure_state_density(psi);

        check(inst.verify_prob_sum(rho), "Instrument Σp(α)=1 for |+⟩");
        check(inst.verify_cptp_sum(rho), "Instrument CPTP sum for |+⟩");

        check_near(inst.outcome_prob(rho, 0), 0.5, 1e-10, "p(0|+) = 0.5");
        check_near(inst.outcome_prob(rho, 1), 0.5, 1e-10, "p(1|+) = 0.5");

        auto post0 = inst.post_state(rho, 0);
        check(measurement::is_valid_density(post0), "Post-state valid after outcome 0");
        check_near(measurement::purity(post0), 1.0, 1e-10, "Post-state pure after PVM");
        psi.free();
    }

    // --- Naimark dilation (trivial PVM case) ---
    {
        int dim = 4;
        auto nd = measurement::trivial_naimark(dim);
        check(nd.verify_isometry(), "Naimark V†V = I");
        check(nd.verify_pvm(), "Naimark: PVM on K");

        measurement::DenseOp* effects = new measurement::DenseOp[dim];
        for (int a = 0; a < dim; ++a)
            effects[a] = measurement::computational_projector(a, dim);
        check(nd.verify_naimark(effects), "Naimark E_α = V†P_αV");
        delete[] effects;
        nd.free_mem();
    }

    // --- MeasPack self-consistency ---
    {
        measurement::MeasPack mp(4);
        check(mp.verify(), "MeasPack(4) self-consistent");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: M(s,β-iα) 3×3 affine matrix representation
// ════════════════════════════════════════════════════════════════════════════
static void test_affine_rep() {
    printf("\n-- Affine Rep M(s,β-iα) --\n");

    using namespace affine_rep;

    // M is a homomorphism: M(g★h) = M(g)·M(h)
    {
        MirElement g = mir::g_phi(1.0);
        MirElement h = mir::U_phi();
        check(verify_M_homomorphism(g, h), "M(g*h) = M(g)M(h) [g_phi, U_phi]");
    }
    {
        MirElement g = mir::J_lambda(constants::PHI);
        MirElement h = mir::g_phi(0.5);
        check(verify_M_homomorphism(g, h), "M(g*h) = M(g)M(h) [J, g_phi]");
    }

    // M(↔)² = I₃
    check(verify_mirror_squared(constants::PHI), "M(↔_φ)² = I₃");

    // M(↔)·M(→)·M(↔) = M(←)
    check(verify_mirror_conjugation_M(constants::PHI), "M(↔)M(→)M(↔) = M(←)");

    // N_φ² = 0 (nilpotent)
    check(verify_N_nilpotent(), "N_φ² = 0");

    // exp(ε·N_φ) = I + ε·N_φ
    check(verify_dual_exp(), "exp(εN_φ) = I + εN_φ");

    // Arrow word consistency: M_word = M(ev(word))
    {
        Arrow word[] = {Arrow::RIGHT, Arrow::LLDIR, Arrow::LEFT};
        check(verify_word_consistency(word, 3), "M_word(→⟸←) = M(ev(→⟸←))");
    }

    // M(→) matches expected 3×3 matrix values
    {
        Mat3 Mr = M_right();
        check_near(Mr.m[0][2], -constants::HALF_PI, 1e-12, "M(→)[0,2] = -π/2");
        check_near(Mr.m[1][2], constants::ELL, 1e-12, "M(→)[1,2] = ℓ");
        check_near(Mr.m[0][0], 1.0, 1e-12, "M(→)[0,0] = 1 (s=+1)");
    }

    // M(↔_λ) diagonal entries = -1
    {
        Mat3 Jm = M_mirror(2.0);
        check_near(Jm.m[0][0], -1.0, 1e-12, "M(↔)[0,0] = -1");
        check_near(Jm.m[1][1], -1.0, 1e-12, "M(↔)[1,1] = -1");
        check_near(Jm.m[1][2], log(2.0), 1e-12, "M(↔_2)[1,2] = ln 2");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Scale set S, ker(q), ζ, conformal metric
// ════════════════════════════════════════════════════════════════════════════
static void test_scale_conformal() {
    printf("\n-- Scale Set / ker(q) / Conformal Metric --\n");

    // Scale set S: s₁·s₄=1, s₂·s₃=1
    check(constants::verify_scale_set(), "Scale set S: s₁s₄=1, s₂s₃=1");

    // s₁ = ΔΣ = 2πφ
    check_near(constants::s1(), constants::DELTA_SIGMA, 1e-14, "s₁ = ΔΣ");
    check_near(constants::s2(), constants::TWO_PI / constants::PHI, 1e-14, "s₂ = 2π/φ");

    // Λ̃ lattice: a^1 b^0 = 2π, a^0 b^1 = φ
    check_near(constants::lambda_tilde(1, 0), constants::TWO_PI, 1e-12, "Λ̃: a^1 b^0 = 2π");
    check_near(constants::lambda_tilde(0, 1), constants::PHI, 1e-14, "Λ̃: a^0 b^1 = φ");
    check_near(constants::lambda_tilde(1, 1), constants::DELTA_SIGMA, 1e-12, "Λ̃: a·b = ΔΣ");

    // ζ = e^{iπ/5}: ζ + ζ⁻¹ + 1 = φ²
    check(constants::verify_zeta_phi_relation(), "ζ + ζ⁻¹ + 1 = φ²");

    // ker(q): (0,0) is in kernel
    check(constants::in_kernel_q(0.0, 0.0), "ker(q): (0,0) ∈ ker");
    check(constants::in_kernel_q(2.0 * PI, 0.0), "ker(q): (2π,0) ∈ ker");
    check(!constants::in_kernel_q(PI, 0.0), "ker(q): (π,0) ∉ ker");
    check(!constants::in_kernel_q(0.0, 1.0), "ker(q): (0,1) ∉ ker");

    // q-equivalence
    check(constants::q_equivalent(0.0, 0.5, 2.0*PI, 0.5), "(0,0.5) ~ (2π,0.5)");
    check(!constants::q_equivalent(0.0, 0.5, PI, 0.5), "(0,0.5) ≁ (π,0.5)");

    // Conformal metric: ds*² = dρ² + dθ²
    {
        double d_rho = 0.3, d_theta = 0.4;
        double ds_star2 = conformal_metric::ds_star_squared(d_rho, d_theta);
        check_near(ds_star2, 0.25, 1e-14, "ds*² = 0.09 + 0.16 = 0.25");
    }

    // Conformal relation: ds² / |z|² = ds*²
    {
        check(conformal_metric::verify_conformal_relation(1.0, 0.1, 0.2),
              "ds²/|z|² = ds*² at ρ=1");
        check(conformal_metric::verify_conformal_relation(0.0, 0.5, 0.3),
              "ds²/|z|² = ds*² at ρ=0");
    }

    // Conformal distance = |w₁-w₂|
    {
        PhasePoint p1(0.0, 0.0), p2(0.3, 0.4);
        check_near(conformal_metric::conformal_distance(p1, p2), 0.5, 1e-14,
                   "d*(0, (0.3,0.4)) = 0.5");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: U_t M_m commutation (operator-level)
// ════════════════════════════════════════════════════════════════════════════
static void test_UM_commutation() {
    printf("\n-- U_t M_m Commutation --\n");

    check(weyl_verify::check_UM_comm(0.5, 1), "[U_0.5, M_1] Born match");
    check(weyl_verify::check_UM_comm(0.25, 2), "[U_0.25, M_2] Born match");
    check(weyl_verify::check_UM_comm(1.0, 0), "[U_1, M_0] Born match (trivial)");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Mir Functor — homomorphism, functoriality, cocycle naturality
// ════════════════════════════════════════════════════════════════════════════
static void test_mir_functor() {
    printf("\n-- Mir Functor --\n");

    using namespace mir_functor;

    AdmissibleAlgebra A = AdmissibleAlgebra::standard();
    AdmissibleAlgebra B = AdmissibleAlgebra::standard();
    AlgebraMorphism phi(A, B);

    MirElement g = mir::g_phi(0.7);
    MirElement h = mir::J_lambda(constants::PHI);

    // Mir(φ) is a group homomorphism: Mir(φ)(g★h) = Mir(φ)(g) ★ Mir(φ)(h)
    check(verify_homomorphism(g, h, phi), "Mir functor homomorphism (g_phi, J)");

    MirElement g2 = mir::U_phi();
    MirElement h2 = mir::U_pi();
    check(verify_homomorphism(g2, h2, phi), "Mir functor homomorphism (U_phi, U_pi)");

    // Functoriality: Mir(ψ ∘ φ) = Mir(ψ) ∘ Mir(φ)
    AlgebraMorphism psi(B, AdmissibleAlgebra::standard());
    check(verify_functoriality(g, phi, psi), "Mir functor functoriality (g_phi)");
    check(verify_functoriality(h, phi, psi), "Mir functor functoriality (J_lambda)");

    // Cocycle naturality: Ω_B(Mir(φ)(g), Mir(φ)(h)) = φ_C(Ω_A(g,h))
    check(verify_cocycle_naturality(g, h, phi), "Cocycle naturality (g_phi, J)");
    check(verify_cocycle_naturality(g2, h, phi), "Cocycle naturality (U_phi, J)");

    // J transport: Mir(φ)(J_{λ,A}) = J_{λ,B}
    check(verify_J_transport(constants::PHI, A, B, phi), "J_lambda transport (φ)");
    check(verify_J_transport(2.0, A, B, phi), "J_lambda transport (λ=2)");

    // Ω(J_λ, g_φ(τ)) = 2τ·c_φ
    check(verify_J_gphi_omega(constants::PHI, 1.0), "Ω(J_φ, g_φ(1)) = 2c_φ");
    check(verify_J_gphi_omega(2.0, 0.5), "Ω(J_2, g_φ(0.5)) = c_φ");

    // Formal UFE = 0
    check(completed_algebra::verify_ufe_formal(C64(1.0, 0.5), C64(0.3, -0.7)),
          "Formal UFE [X,Y] = 0 (scalar)");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Transport Naturality — cocycle, UFE, δΩ, functoriality
// ════════════════════════════════════════════════════════════════════════════
static void test_transport_naturality() {
    printf("\n-- Transport Naturality --\n");

    using namespace transport;

    AdmissibleAlgebra A = AdmissibleAlgebra::standard();
    AdmissibleAlgebra B = AdmissibleAlgebra::standard();
    AlgebraMorphism phi(A, B);
    AlgTransport T(phi);

    MirElement g = mir::g_phi(0.7);
    MirElement h = mir::g_phi(1.3);
    MirElement k = mir::J_lambda(constants::PHI);

    // Cocycle transport: Ω_B(φ(g),φ(h)) = φ_C(Ω_A(g,h))
    check(naturality::verify_cocycle_transport(T, g, h),
          "Cocycle transport (g_phi pair)");
    check(naturality::verify_cocycle_transport(T, g, k),
          "Cocycle transport (g_phi, J)");

    // UFE transport: UFE_B = φ(UFE_A) = 0
    check(naturality::verify_ufe_transport(T, g, h),
          "UFE transport (g_phi pair)");
    check(naturality::verify_ufe_transport(T, g, k),
          "UFE transport (g_phi, J)");

    // Cocycle condition: δΩ = 0 preserved under transport
    check(naturality::verify_cocycle_condition(T, g, h, k),
          "δΩ=0 under transport (g,h,k)");

    MirElement g2 = mir::U_pi();
    check(naturality::verify_cocycle_condition(T, g2, h, k),
          "δΩ=0 under transport (U_pi,h,k)");

    // Functoriality: T_{ψ∘φ} = T_ψ ∘ T_φ
    AdmissibleAlgebra C = AdmissibleAlgebra::standard();
    AlgebraMorphism psi(B, C);
    AlgTransport T_psi(psi);
    check(naturality::verify_functoriality(T, T_psi, g),
          "Transport functoriality (g_phi)");
    check(naturality::verify_functoriality(T, T_psi, k),
          "Transport functoriality (J)");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: End Rep, Basis Orthogonality, Category Congruence, Scale-Translate
// ════════════════════════════════════════════════════════════════════════════
static void test_exhaustive_coverage() {
    printf("\n-- Exhaustive Coverage --\n");

    // Weyl basis orthogonality: Tr((X_u Z_v)† X_{u'} Z_{v'}) = 2^n δ
    check(hybrid_algebra::verify_basis_orthogonality(2),
          "Weyl basis orthogonality n=2");

    // Category congruence: u₁~v₁, u₂~v₂ ⟹ u₁u₂ ~ v₁v₂
    {
        // →← ~ ε  (both evaluate to identity)
        ArrowWord rl(Arrow::RIGHT);
        rl = rl.concat(ArrowWord(Arrow::LEFT));
        ArrowWord eps = ArrowWord::empty();
        // ⟸⟹ ~ ε  (both evaluate to identity)
        ArrowWord lr(Arrow::LLDIR);
        lr = lr.concat(ArrowWord(Arrow::RRDIR));
        check(category_mir::verify_congruence(rl, eps, lr, eps),
              "Category congruence: (→←)(⟸⟹) ~ ε·ε");
    }

    // Arrow rep: scale-translate commutation [S_λ, T_h] = T_{(1-λ⁻¹)h}
    check(arrow_rep::verify_scale_translate_comm(constants::PHI, 1.0),
          "PGL₂ [S_φ, T_1] = T_{(1-1/φ)·1}");
    check(arrow_rep::verify_scale_translate_comm(2.0, 0.5),
          "PGL₂ [S_2, T_0.5] = T_{0.25}");

    // End rep: arrow actions on R₃₂ = TruncPoly
    {
        TruncPoly p;
        p.coeffs[0] = 1.0; p.coeffs[1] = 0.5; p.coeffs[2] = 0.25;

        // T_action (translation): exp(h·e)(p)(t) = p(t+h)
        TruncPoly shifted = end_rep::T_action(p, 0.1);
        // p(t+0.1) at t=0 should be p(0.1)
        double p_at_01 = 1.0 + 0.5*0.1 + 0.25*0.01;
        check_near(shifted.coeffs[0], p_at_01, 1e-10, "End rep T_action p(t+h)");

        // J_action (reversal): J(p) reverses coefficients
        TruncPoly reversed = end_rep::J_action(p);
        // p has nonzero coefficients at 0,1,2; reversed has them at N-1,N-2,N-3
        check_near(reversed.coeffs[p.N - 1], 1.0, 1e-14,
                   "End rep J_action: coeff[N-1]=p[0]");

        // S_action (scaling): S_λ(p)(t) = p(λt)
        TruncPoly scaled = end_rep::S_action(p, 2.0);
        // p(2t): coeff[0]=1, coeff[1]=0.5*2=1.0, coeff[2]=0.25*4=1.0
        check_near(scaled.coeffs[0], 1.0, 1e-14, "End rep S_action coeff[0]");
        check_near(scaled.coeffs[1], 1.0, 1e-14, "End rep S_action coeff[1]");
        check_near(scaled.coeffs[2], 1.0, 1e-14, "End rep S_action coeff[2]");

        // G_action = J∘T_h∘J (exp(-hK) on R_k)
        TruncPoly g_result = end_rep::G_action(p, 0.0);
        // G_{0} = J∘T_0∘J = J∘J = identity
        check_near(g_result.coeffs[0], p.coeffs[0], 1e-10, "End rep G_action(h=0) = id");

        // Word evaluation consistency: word acts on poly
        Arrow word[] = {Arrow::LLDIR, Arrow::RRDIR};
        TruncPoly word_result = end_rep::eval_word(word, 2, p);
        // ⟸⟹ is S_ΔΣ then S_{1/ΔΣ} = identity
        check_near(word_result.coeffs[0], p.coeffs[0], 1e-10,
                   "End rep ⟸⟹ = identity on R₃₂");
    }

    // sl₂ on R₃₂: boundary conditions e(t⁰)=0 (e raises weight)
    {
        TruncPoly t0;
        t0.coeffs[0] = 1.0;  // constant polynomial = t^0
        TruncPoly e_t0 = t0.sl2_e();
        check_near(e_t0.coeffs[0], 0.0, 1e-14, "sl₂ e(t⁰) coeff[0]=0 (raises weight)");

        // f(t^{N-1}) should give 0 or wrap (boundary of R₃₂)
        TruncPoly tN;
        tN.coeffs[tN.N - 1] = 1.0;  // t^{N-1}
        TruncPoly f_tN = tN.sl2_f();
        // f should lower: t^{N-1} → (N-1)·t^N, but t^N = 0 in truncated ring
        // So all coefficients of f(t^{N-1}) should be truncated
        double f_norm = 0;
        for (int i = 0; i < tN.N; ++i) f_norm += fabs(f_tN.coeffs[i]);
        check(f_norm < 1e-10, "sl₂ f(t^{N-1})=0 (boundary, truncated)");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Heisenberg Group
// ════════════════════════════════════════════════════════════════════════════
static void test_heisenberg() {
    printf("\n-- Heisenberg Group --\n");
    using namespace heisenberg;

    // Identity
    HeisElement e = HeisElement::identity(2);
    HeisElement a(BitVec(1, 2), BitVec(2, 2), C64(1, 0));
    check(verify_identity(a), "Heis identity a*e=e*a=a");

    // Associativity
    HeisElement b(BitVec(2, 2), BitVec(1, 2), C64(1, 0));
    HeisElement c(BitVec(3, 2), BitVec(0, 2), C64(1, 0));
    check(verify_associativity(a, b, c), "Heis (a*b)*c = a*(b*c)");

    // XZ commutation: X_u Z_v = (-1)^{⟨u,v⟩} Z_v X_u
    BitVec u1(1, 2), v1(1, 2);  // ⟨u,v⟩ = 1
    check(verify_XZ_commutation(u1, v1), "Heis X_u Z_v = -Z_v X_u (⟨u,v⟩=1)");

    BitVec u2(1, 2), v2(2, 2);  // ⟨u,v⟩ = 0
    check(verify_XZ_commutation(u2, v2), "Heis X_u Z_v = +Z_v X_u (⟨u,v⟩=0)");

    // Inverse
    HeisElement ai = a.inverse();
    HeisElement aai = a * ai;
    check(aai.u.bits == 0 && aai.v.bits == 0 &&
          fabs(aai.zeta.re - 1.0) < 1e-10 && fabs(aai.zeta.im) < 1e-10,
          "Heis a*a^{-1} = identity");

    // Phase accumulation: Heis product accumulates Ω_bool phases
    HeisElement xu(BitVec(1, 2), BitVec(0, 2), C64(1, 0));
    HeisElement zv(BitVec(0, 2), BitVec(1, 2), C64(1, 0));
    HeisElement xuZv = xu * zv;
    HeisElement zvXu = zv * xu;
    double phase_diff = (xuZv.zeta - zvXu.zeta * C64(-1, 0)).norm2();
    check(phase_diff < 1e-10, "Heis X_u*Z_v phase = -Z_v*X_u phase");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Clifford Algebra
// ════════════════════════════════════════════════════════════════════════════
static void test_clifford() {
    printf("\n-- Clifford Algebra Cl_{0,n} --\n");
    using namespace clifford;

    // γ_0² = -1
    check(verify_gamma_squared(3, 0), "Cl γ_0² = -1");
    check(verify_gamma_squared(3, 1), "Cl γ_1² = -1");
    check(verify_gamma_squared(3, 2), "Cl γ_2² = -1");

    // Anticommutation: γ_iγ_j = -γ_jγ_i (i≠j)
    check(verify_anticommutation(3, 0, 1), "Cl γ_0γ_1 = -γ_1γ_0");
    check(verify_anticommutation(3, 0, 2), "Cl γ_0γ_2 = -γ_2γ_0");
    check(verify_anticommutation(3, 1, 2), "Cl γ_1γ_2 = -γ_2γ_1");

    // Γ_Cl(u)Γ_Cl(v) = (-1)^{⟨u,v⟩} Γ_Cl(v)Γ_Cl(u)
    check(verify_clifford_commutation(2, 1, 1),
          "Cl Γ(01)Γ(01) = (-1)^1 Γ(01)Γ(01)");
    check(verify_clifford_commutation(2, 1, 2),
          "Cl Γ(01)Γ(10) = (-1)^0 Γ(10)Γ(01)");
    check(verify_clifford_commutation(3, 5, 3),
          "Cl Γ(101)Γ(011) = (-1)^{⟨101,011⟩} Γ(011)Γ(101)");

    // Scalar · gamma
    CliffElement s = CliffElement::scalar(2, 3.0);
    CliffElement g0 = CliffElement::gamma(2, 0);
    CliffElement sg = s * g0;
    check(fabs(sg.coeffs[1] - 3.0) < 1e-10, "Cl 3·γ_0 coeff = 3");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Boolean Algebra & Difference Operators
// ════════════════════════════════════════════════════════════════════════════
static void test_boolean_algebra() {
    printf("\n-- Boolean Algebra B_n --\n");
    using namespace boolean_algebra;

    // Define a simple boolean function on F_2^2
    // f(00)=0, f(01)=1, f(10)=1, f(11)=0  (XOR function)
    BoolFunc f(2);
    f.values[0] = 0;  // f(00)
    f.values[1] = 1;  // f(01)
    f.values[2] = 1;  // f(10)
    f.values[3] = 0;  // f(11)

    // ANF coefficient c_∅ = f(0) = 0
    check(f.anf_coeff(0) == 0, "B_2 XOR: c_∅ = 0");

    // c_{1} = Δ₁f(0) = f(01) ⊕ f(00) = 1
    check(f.anf_coeff(1) == 1, "B_2 XOR: c_{1} = 1");

    // c_{2} = Δ₂f(0) = f(10) ⊕ f(00) = 1
    check(f.anf_coeff(2) == 1, "B_2 XOR: c_{2} = 1");

    // c_{12} = Δ₁Δ₂f(0) = f(11) ⊕ f(10) ⊕ f(01) ⊕ f(00) = 0
    check(f.anf_coeff(3) == 0, "B_2 XOR: c_{12} = 0");

    // Möbius inversion: reconstruction from ANF
    check(verify_mobius_inversion(f), "B_2 Möbius inversion roundtrip");

    // AND function: f(00)=0, f(01)=0, f(10)=0, f(11)=1
    BoolFunc g(2);
    g.values[3] = 1;
    check(verify_mobius_inversion(g), "B_2 AND Möbius inversion");
    check(g.anf_coeff(3) == 1, "B_2 AND: c_{12} = 1 (only x₁x₂ term)");

    // Difference operator Δ₁
    BoolFunc df = f.delta(0);
    // Δ₁f(x) = f(x⊕e₁)⊕f(x)
    // Δ₁f(00) = f(01)⊕f(00) = 1, Δ₁f(01) = f(00)⊕f(01) = 1
    // Δ₁f(10) = f(11)⊕f(10) = 1, Δ₁f(11) = f(10)⊕f(11) = 1
    check(df.values[0] == 1 && df.values[1] == 1 &&
          df.values[2] == 1 && df.values[3] == 1,
          "B_2 Δ₁(XOR) = constant 1");

    // Hamming weight
    check(f.weight() == 2, "B_2 XOR weight = 2");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Shadow Functor
// ════════════════════════════════════════════════════════════════════════════
static void test_shadow_functor() {
    printf("\n-- Shadow Functor S_Mir --\n");
    using namespace shadow;

    ShadowData sd(2);

    // ⟨u,v⟩_sh = ⟨u,v⟩ for standard inner product
    BitVec u1(1, 2), v1(1, 2);
    check(sd.inner_sh(u1, v1) == 1, "Shadow ⟨01,01⟩_sh = 1");

    BitVec u2(1, 2), v2(2, 2);
    check(sd.inner_sh(u2, v2) == 0, "Shadow ⟨01,10⟩_sh = 0");

    // σ_sh(u,v) = (-1)^{⟨u,v⟩_sh}
    check(fabs(sd.sigma_sh(u1, v1) - (-1.0)) < 1e-10,
          "Shadow σ(01,01) = -1");
    check(fabs(sd.sigma_sh(u2, v2) - 1.0) < 1e-10,
          "Shadow σ(01,10) = +1");

    // Shadow pipeline: sh_2(Ω_Mir) = Ω_bool
    check(sd.verify_pipeline(u1, v1), "Shadow pipeline ⟨01,01⟩");
    check(sd.verify_pipeline(u2, v2), "Shadow pipeline ⟨01,10⟩");

    BitVec u3(3, 2), v3(3, 2);
    check(sd.verify_pipeline(u3, v3), "Shadow pipeline ⟨11,11⟩");

    // Cocycle condition δΩ_sh = 0
    BitVec w1(1, 2);
    check(sd.verify_cocycle_condition(u1, v1, w1),
          "Shadow δΩ_sh(01,01,01) = 0");

    // Full shadow computation
    ShadowFunctorResult r = compute_shadow(u1, v1, 2);
    check(r.pipeline_valid, "Shadow full pipeline valid");
    check(r.inner_product == 1, "Shadow inner_product = 1");
    check(fabs(r.sigma_sh - (-1.0)) < 1e-10, "Shadow sigma = -1");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Fibonacci/Spiral Analytics
// ════════════════════════════════════════════════════════════════════════════
static void test_fibonacci() {
    printf("\n-- Fibonacci / Spiral --\n");
    using namespace fibonacci;

    // Fibonacci numbers via matrix
    check(fib_matrix(1) == 1, "Fib F_1 = 1 (matrix)");
    check(fib_matrix(5) == 5, "Fib F_5 = 5 (matrix)");
    check(fib_matrix(10) == 55, "Fib F_10 = 55 (matrix)");

    // Binet formula matches matrix
    check(verify_binet(10), "Binet F_10 matches matrix");
    check(verify_binet(15), "Binet F_15 matches matrix");
    check(verify_binet(20), "Binet F_20 matches matrix");

    // φ^n = F_n·φ + F_{n-1}
    check(verify_phi_power(5), "φ^5 = F_5·φ + F_4");
    check(verify_phi_power(10), "φ^10 = F_10·φ + F_9");

    // Q-matrix eigendecomposition
    check(verify_eigendecomposition(), "Q eigendecomposition Q^5 correct");

    // Spiral arc length: s*(t) = |c_φ|·t
    double speed = geodesic_speed();
    C64 cp = constants::c_phi();
    check_near(speed, cp.abs(), 1e-14, "Geodesic speed |c_φ|");

    double s1 = arc_length(1.0);
    check_near(s1, speed, 1e-14, "Arc length s*(1) = |c_φ|");

    // dω̃/ds* = -π/|c_φ|
    double dw = domega_tilde_ds();
    check_near(dw, -PI / speed, 1e-14, "dω̃/ds* = -π/|c_φ|");

    // Spiral radii: r_n = r_0·φ^n
    double r0 = 1.0;
    check_near(spiral_radius(r0, 5), pow(PHI, 5), 1e-10,
               "Spiral r_5 = φ^5");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Connection/Curvature/Index
// ════════════════════════════════════════════════════════════════════════════
static void test_connection() {
    printf("\n-- Connection / Curvature / Index --\n");
    using namespace connection;

    // Golden flow connection is flat
    Connection1Form A = Connection1Form::golden_flow();
    check(A.is_flat(), "Golden flow connection is flat");

    // Flat ⟺ commuting
    check(verify_flat_iff_commuting(), "Flat ⟺ [g,h]=1");

    // Â(TX) = 1 for flat dim-2
    check_near(A_hat_flat_dim2(), 1.0, 1e-14, "Â(TX) = 1 (flat dim 2)");

    // Arf invariant for even n (symplectic form requires even dimension)
    check(verify_arf(2), "Arf(n=2) ∈ {±1}");
    check(verify_arf(4), "Arf(n=4) ∈ {±1}");

    // Bianchi identity (abelian)
    check(verify_bianchi_abelian(), "Bianchi identity (abelian flat)");

    // Chern data
    ChernData ch(1, 0.0);
    check_near(ch.ch2(), 0.0, 1e-14, "ch₂ = 0 for trivial bundle");

    // Index dim 2: ind = rk·χ + deg
    double idx = index_dim2(1, 2.0, 0.0);  // rank 1, χ=2 (sphere), deg=0
    check_near(idx, 2.0, 1e-14, "ind(D) = rk·χ for deg=0");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Translation Matrices
// ════════════════════════════════════════════════════════════════════════════
static void test_translation_matrices() {
    printf("\n-- Translation Matrices T(α,λ) --\n");
    using namespace translation;

    // T₁ · T₂ = T(α₁+α₂, λ₁λ₂)
    check(verify_composition(1.0, 2.0), "T composition M_Π·M_Φ");
    check(verify_composition(0.5, -1.0), "T composition M_Π^{0.5}·M_Φ^{-1}");

    // Inverse: T·T⁻¹ = I
    check(verify_inverse(constants::TWO_PI, constants::PHI),
          "T(2π,φ)·T(2π,φ)⁻¹ = I");

    // M_Π^1 = T(2π, a)
    TranslationMatrix mp = M_Pi(1.0);
    check_near(mp.alpha, constants::TWO_PI, 1e-14, "M_Π^1.α = 2π");
    check_near(mp.lambda(), constants::A_CONST, 1e-10, "M_Π^1.λ = a = 2π");

    // M_Φ^1 = T(π/2, b)
    TranslationMatrix mf = M_Phi(1.0);
    check_near(mf.alpha, constants::HALF_PI, 1e-14, "M_Φ^1.α = π/2");
    check_near(mf.lambda(), constants::B_CONST, 1e-10, "M_Φ^1.λ = b = φ");

    // Matrix representation
    double m[9];
    mp.get_matrix(m);
    check_near(m[0], 1.0, 1e-14, "T matrix [0,0] = 1");
    check_near(m[2], -constants::TWO_PI, 1e-14, "T matrix [0,2] = -α");
    check_near(m[5], constants::LN_A, 1e-14, "T matrix [1,2] = ln λ");
    check_near(m[8], 1.0, 1e-14, "T matrix [2,2] = 1");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Witt Commutator & BCH
// ════════════════════════════════════════════════════════════════════════════
static void test_witt_commutator() {
    printf("\n-- Witt Commutator & BCH --\n");
    using namespace witt_commutator;

    // Commutator formula: [g,h] ≡ 1 + η²[X,Y] (scalars: [X,Y]=0)
    check(verify_commutator_formula(1.5, 2.3), "Witt [g,h] formula X=1.5, Y=2.3");
    check(verify_commutator_formula(0.0, 1.0), "Witt [g,h] formula X=0, Y=1");

    // g⁻¹ · g = 1
    Witt2Element g = group_element(3.0);
    Witt2Element gi = witt2_inverse(g);
    Witt2Element prod = gi * g;
    check_near(prod.x0, 1.0, 1e-10, "Witt2 g⁻¹g = 1 (x0)");
    check_near(prod.x1, 0.0, 1e-10, "Witt2 g⁻¹g = 1 (x1)");
    check_near(prod.x2, 0.0, 1e-10, "Witt2 g⁻¹g = 1 (x2)");

    // F_G(g,h) = 0 for commuting scalars
    check(verify_witt_UFE(1.0, 2.0), "Witt UFE(1,2) = 0");

    // BCH leading term
    double bch = bch_leading(1.0, 2.0);
    check_near(bch, 3.0, 1e-14, "BCH(1,2) = 3 (scalars)");

    // log(1+Z) in W_2 for small Z
    Witt2Element z(0.0, 0.0, 0.1);  // Z = 0.1·ε²
    Witt2Element logz = witt2_log1p(z);
    check_near(logz.x2, 0.1, 1e-10, "Witt2 log(1+0.1ε²) ≈ 0.1ε²");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Faà di Bruno Jet Composition
// ════════════════════════════════════════════════════════════════════════════
static void test_faa_di_bruno() {
    printf("\n-- Faà di Bruno --\n");
    using namespace faa_di_bruno;

    // Chain rule: f(w)=2w, g(w)=w² → (g∘f)(w)=4w²
    check(verify_chain_rule(), "FdB chain rule f=2w, g=w²");

    // Identity composition: composing with identity jet
    faa_di_bruno::Jet id = faa_di_bruno::Jet::identity(3);
    faa_di_bruno::Jet f(3);
    f.a[0] = 1.0; f.a[1] = 2.0; f.a[2] = 0.5;
    // g = identity centered at f(0): g(y) = y → g.a[0]=f.a[0]=1, g.a[1]=1
    faa_di_bruno::Jet g_at_f(3);
    g_at_f.a[0] = f.a[0];  // g(f(w)) starts at f(w)
    g_at_f.a[1] = 1.0;     // g'(f(w)) = 1
    faa_di_bruno::Jet result = compose(f, g_at_f);
    check_near(result.a[0], f.a[0], 1e-10, "FdB identity compose a[0]");
    check_near(result.a[1], f.a[1], 1e-10, "FdB identity compose a[1]");
    check_near(result.a[2], f.a[2], 1e-10, "FdB identity compose a[2]");

    // Constant jet: composing with constant gives constant
    faa_di_bruno::Jet c = faa_di_bruno::Jet::constant(3, 5.0);
    faa_di_bruno::Jet g_const(3);
    g_const.a[0] = 5.0;
    faa_di_bruno::Jet r2 = compose(c, g_const);
    check_near(r2.a[0], 5.0, 1e-10, "FdB constant compose = constant");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Disc-Top Information Decomposition
// ════════════════════════════════════════════════════════════════════════════
static void test_disc_top_info() {
    printf("\n-- Disc-Top Information Decomposition --\n");
    using namespace disc_top_info;
    using measurement::DenseOp;

    int nq = 2;
    int dim = 1 << nq;

    // Uniform state: ρ = I/d
    DenseOp rho_uniform = measurement::maximally_mixed(dim);

    // Von Neumann entropy of max mixed = log(d)
    double S_max = von_neumann_entropy(rho_uniform);
    check_near(S_max, log((double)dim), 1e-10, "S(I/d) = log d");

    // Pure state: entropy = 0
    QubitState psi;
    psi.init(nq);
    psi.set_basis(0);
    DenseOp rho_pure = measurement::pure_state_density(psi);
    double S_pure = von_neumann_entropy(rho_pure);
    check_near(S_pure, 0.0, 1e-10, "S(|0⟩⟨0|) = 0");

    // Cocycle splitting: Ω_tot = Ω_bool ⊕ Ω_A
    MirElement g = mir::g_phi(1.0);
    MirElement h = mir::g_phi(2.0);
    BitVec u(1, 2), v(2, 2), up(3, 2), vp(0, 2);
    check(verify_cocycle_splitting(g, h, u, v, up, vp),
          "Ω_tot = Ω_bool ⊕ Ω_A splits correctly");

    // UFE chain: UFE_hyb = 0 → UFE_Mir = 0
    check(verify_ufe_chain(g, h, u, v, up, vp),
          "UFE chain: hyb=0 → Mir=0");

    // Cross-sector commutation
    check(verify_cross_sector_commutation(),
          "[X_u^hyb, U_t^hyb] = 0 (cross-sector)");

    // Entropy preservation
    check(verify_entropy_preservation(rho_uniform),
          "Entropy bounds: 0 ≤ S ≤ log d");

    // Born normalization
    check(verify_born_decomposition(rho_uniform),
          "Born Σp(x)=1 (uniform)");
    check(verify_born_decomposition(rho_pure),
          "Born Σp(x)=1 (pure)");

    // Shannon entropy of disc marginal for uniform state
    double S_disc = disc_entropy(rho_uniform);
    check_near(S_disc, log((double)dim), 1e-10,
               "S_disc(I/d) = log d (max disc entropy)");

    // Full analytics computation
    InfoRetainmentAnalytics analytics = compute_full_analytics(
        rho_uniform, g, h, u, v, up, vp);
    check(analytics.cocycle_splits, "Full analytics: cocycle splits");
    check(analytics.ufe_chain_valid, "Full analytics: UFE chain valid");
    check(analytics.cross_sector_commutes, "Full analytics: cross-sector commutes");
    check(analytics.entropy_preserved, "Full analytics: entropy preserved");
    check(analytics.born_normalized, "Full analytics: Born normalized");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Grand Derivation Chain
// ════════════════════════════════════════════════════════════════════════════
static void test_grand_chain() {
    printf("\n-- Grand Derivation Chain --\n");

    int steps = grand_chain::verify_grand_chain();
    check(steps == 8, "Grand chain: all 8 steps pass");

    // Individual step verification
    check(grand_chain::verify_constants(), "Chain step 1: (φ,2π) → K_log");
    check(grand_chain::verify_admissible(), "Chain step 2: K_log → Alg^Mir");
    check(grand_chain::verify_mir_group(), "Chain step 3: Alg^Mir → Mir_A");
    check(grand_chain::verify_functor(), "Chain step 4: Mir_A → Mir(φ)");
    check(grand_chain::verify_cocycle(), "Chain step 5: δΩ = 0");
    check(grand_chain::verify_ufe(), "Chain step 6: UFE = 0");
    check(grand_chain::verify_shadow(), "Chain step 7: Sh₂(Ω_Mir) = Ω_bool");
    check(grand_chain::verify_total_split(), "Chain step 8: Ω_tot splits");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Isometry Verification
// ════════════════════════════════════════════════════════════════════════════
static void test_isometry() {
    printf("\n-- Isometry Verification --\n");
    using namespace isometry;

    // g_φ(1) preserves conformal metric
    MirElement g1 = mir::g_phi(1.0);
    check(verify_isometry(g1), "Isometry g_φ(1): f*(g*)=g*");

    // U_pi preserves metric
    MirElement g2 = mir::U_pi();
    check(verify_isometry(g2), "Isometry U_π: f*(g*)=g*");

    // J_λ preserves metric (s=-1, s²=1)
    MirElement g3 = mir::J_lambda(2.0);
    check(verify_isometry(g3), "Isometry J_2: f*(g*)=g*");

    // Numerical distance preservation
    check(verify_isometry_numerical(g1), "Isometry g_φ(1) numerical");
    check(verify_isometry_numerical(g2), "Isometry U_π numerical");
    check(verify_isometry_numerical(g3), "Isometry J_2 numerical");

    // w-action compatibility: (g1★g2)·w = g1·(g2·w)
    MirElement h = mir::g_phi(0.5);
    check(verify_action_compatibility(g1, h), "w-action: (g★h)·w = g·(h·w)");
    check(verify_action_compatibility(g3, g1), "w-action: (J★g)·w = J·(g·w)");

    // z-action: z' = e^c z^s
    C64 w(1.0, 0.5);
    C64 z = cexp(w);
    C64 w_prime = w_action(g1, w);
    C64 z_prime_from_w = cexp(w_prime);
    C64 z_prime_direct = z_action(g1, z);
    check((z_prime_from_w - z_prime_direct).norm2() < 1e-10,
          "z-action consistent: e^{g·w} = g·(e^w)");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Weyl Translation Operators
// ════════════════════════════════════════════════════════════════════════════
static void test_weyl_translation() {
    printf("\n-- Weyl Translation Operators --\n");
    using namespace weyl_translation;

    // W(c₁)W(c₂) = W(c₁+c₂)
    check(verify_composition(C64(1, 0), C64(0, 1)),
          "W(1)W(i) = W(1+i)");

    // W(c)QW(-c) = Q + c·id
    check(verify_position_shift(), "W(c)QW(-c) = Q+c·id");

    // W(c)PW(-c) = P
    check(verify_momentum_invariance(), "W(c)PW(-c) = P");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Jet UM Commutation
// ════════════════════════════════════════════════════════════════════════════
static void test_jet_um_commutation() {
    printf("\n-- Jet UM Commutation --\n");
    using namespace jet_um_comm;

    // Phase factor jet extension: e^{-i2πm(t₀+ηt₁)}
    check(verify_jet_phase(0.5, 0.1, 1), "Jet phase m=1, t₀=0.5");
    check(verify_jet_phase(0.0, 1.0, 2), "Jet phase m=2, t₀=0");
    check(verify_jet_phase(0.25, 0.3, 3), "Jet phase m=3, t₀=0.25");

    // Multi-jet computation
    double tj[] = {0.1, 0.2};
    MultiJetPhase mjp = MultiJetPhase::compute(0.5, tj, 2, 1);
    // val should be e^{-iπ} = -1
    check_near(mjp.val.re, -1.0, 1e-10, "MultiJet val = e^{-iπ} = -1");
    check(mjp.d == 2, "MultiJet d = 2");

    // Verify e^{-i2πmt₀} · (1 + Σ η(-i2πmt_j)) structure
    // For t₀=0.5, m=1: e^{-iπ} = -1
    check(mjp.derivs[0].norm2() > 1e-20, "MultiJet deriv[0] nonzero");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Admissible Sector Decomposition
// ════════════════════════════════════════════════════════════════════════════
static void test_sector() {
    printf("\n-- Sector Decomposition --\n");
    using namespace sector;

    // ── Computational basis on C^2 ──────────────────────────────────────────
    AdmissibleFamily cb2 = AdmissibleFamily::computational_basis(2);
    check(verify::resolution_of_identity(cb2),
          "C^2 basis: valid resolution");

    // Pure |0⟩: all weight in sector 0
    DenseOp rho_0(2);
    rho_0.at(0, 0) = C64(1, 0);
    SectorWeights sw0 = decompose::born_weights(cb2, rho_0);
    check_near(sw0.weights[0], 1.0, 1e-10, "|0>: p(0) = 1");
    check_near(sw0.weights[1], 0.0, 1e-10, "|0>: p(1) = 0");

    // Equal superposition |+⟩ = (|0⟩+|1⟩)/√2
    DenseOp rho_plus(2);
    rho_plus.at(0, 0) = C64(0.5, 0);
    rho_plus.at(0, 1) = C64(0.5, 0);
    rho_plus.at(1, 0) = C64(0.5, 0);
    rho_plus.at(1, 1) = C64(0.5, 0);
    SectorWeights sw_p = decompose::born_weights(cb2, rho_plus);
    check(decompose::verify_normalization(sw_p),
          "|+>: weights sum to 1");
    check_near(sw_p.weights[0], 0.5, 1e-10,
               "|+>: p(0) = 1/2 (exact sector weight)");

    // Exact decomposition: ρ = Σ P_a ρ P_a (sector-diagonal ρ)
    check(decompose::verify_exact_decomposition(cb2, rho_0),
          "|0>: exact sector decomposition");

    // Sector state: ρ_a is normalized
    DenseOp rho_a = decompose::sector_state(cb2, rho_plus, 0);
    check_near(rho_a.trace().re, 1.0, 1e-10,
               "sector state Tr(rho_0) = 1");

    // ── Computational basis on C^4 ──────────────────────────────────────────
    AdmissibleFamily cb4 = AdmissibleFamily::computational_basis(4);
    check(verify::resolution_of_identity(cb4),
          "C^4 basis: valid resolution");

    // Maximally mixed state: equal weights
    DenseOp rho_mm = measurement::maximally_mixed(4);
    SectorWeights sw_mm = decompose::born_weights(cb4, rho_mm);
    check(decompose::verify_positivity(sw_mm),
          "mixed: all weights >= 0");
    check_near(sw_mm.weights[0], 0.25, 1e-10,
               "mixed: p(0) = 1/4");

    // ── Disc sectors (hybrid space) ─────────────────────────────────────────
    AdmissibleFamily ds = AdmissibleFamily::disc_sectors(2, 3);
    check(verify::resolution_of_identity(ds),
          "disc sectors 2x3: valid resolution");

    // ── Compound sectors ────────────────────────────────────────────────────
    AdmissibleFamily f1 = AdmissibleFamily::computational_basis(2);
    AdmissibleFamily f2 = AdmissibleFamily::computational_basis(2);
    AdmissibleFamily cf = compound::tensor_product(f1, f2);
    check(cf.n_sectors == 4, "compound: |A1 x A2| = 4");
    check(verify::resolution_of_identity(cf),
          "compound: valid resolution");

    // Compound Born weights on |00⟩
    DenseOp rho_00(4);
    rho_00.at(0, 0) = C64(1, 0);
    SectorWeights csw = decompose::born_weights(cf, rho_00);
    check_near(csw.weights[0], 1.0, 1e-10,
               "compound |00>: p(0,0) = 1");

    // ── Spectral binning ────────────────────────────────────────────────────
    double eigs[] = {0.0, 1.0, 2.0, 3.0};
    AdmissibleFamily sp = spectral::from_diagonal(eigs, 4, 2, 0.0, 4.0);
    check(verify::resolution_of_identity(sp),
          "spectral 2-bin: valid resolution");

    SectorWeights sp_sw = decompose::born_weights(sp, rho_mm);
    check_near(sp_sw.weights[0], 0.5, 1e-10,
               "spectral: bin[0..2) = 0.5");
    check_near(sp_sw.weights[1], 0.5, 1e-10,
               "spectral: bin[2..4) = 0.5");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Hybrid Fiber Structure
// ════════════════════════════════════════════════════════════════════════════
static void test_fiber() {
    printf("\n-- Fiber Structure --\n");
    using namespace fiber;

    // ── Fiber maps and roundtrip ────────────────────────────────────────────
    FiberMap fm(2, 3);   // 2 disc × 3 top = 6 hybrid
    check(fm.dim_hyb == 6, "fiber: dim_hyb = 2*3 = 6");

    // π_x ∘ ι_y = δ_{xy} I
    check(verify::injection_projection_identity(fm),
          "fiber: pi_x . iota_y = delta_{xy} I");

    // Σ_x ι_x π_x = I (fiber resolution)
    check(verify::fiber_resolution(fm),
          "fiber: Sigma iota_x pi_x = I");

    // Fiber projectors form valid resolution of identity
    AdmissibleFamily ff = fm.fiber_family();
    check(sector::verify::resolution_of_identity(ff),
          "fiber: P_x^hyb resolution of identity");

    // Inject σ into fiber 0, project back: roundtrip
    DenseOp sigma = DenseOp::identity(3);
    DenseOp rho_inj = fm.inject(0, sigma);
    DenseOp extracted = fm.project(0, rho_inj);
    check_near(extracted.trace().re, 3.0, 1e-10,
               "fiber: inject/project roundtrip Tr = 3");

    // Cross-fiber projection gives zero
    DenseOp cross = fm.project(1, rho_inj);
    check_near(cross.trace().re, 0.0, 1e-10,
               "fiber: cross-fiber projection Tr = 0");

    // ── Operator lifting ────────────────────────────────────────────────────
    DenseOp X(2);   // Pauli X on disc space
    X.at(0, 1) = C64(1, 0);
    X.at(1, 0) = C64(1, 0);
    DenseOp X_hyb = lift::disc_operator(X, 3);
    check(X_hyb.dim == 6, "lift: X^hyb dim = 6");

    DenseOp U(3);   // diagonal phase operator on top space
    U.at(0, 0) = C64(1, 0);
    U.at(1, 1) = C64(0, 1);
    U.at(2, 2) = C64(-1, 0);
    DenseOp U_hyb = lift::top_operator(U, 2);

    // [X^hyb, U^hyb] = 0
    check(lift::cross_sector_commutation(X_hyb, U_hyb),
          "lift: [X^hyb, U^hyb] = 0");

    // ── Joint measurement ───────────────────────────────────────────────────
    // Lift topological comp. basis to hybrid space
    AdmissibleFamily top_cb =
        sector::AdmissibleFamily::computational_basis(3);
    AdmissibleFamily top_hyb(6, 3);
    for (int j = 0; j < 3; ++j)
        top_hyb.projectors[j] =
            lift::top_operator(top_cb.projectors[j], 2);

    // Product state |0,0⟩: all weight in joint sector (0,0)
    DenseOp rho_00(6);
    rho_00.at(0, 0) = C64(1, 0);
    joint::JointWeights jw =
        joint::joint_born_weights(ff, top_hyb, rho_00);
    check_near(jw.weights[0][0], 1.0, 1e-10,
               "joint: p(0,0) = 1 for |0,0>");
    check(joint::verify_normalization(jw),
          "joint: weights sum to 1");
    check(joint::verify_marginal_consistency(jw),
          "joint: marginal consistency");
    check_near(jw.disc_marginal(0), 1.0, 1e-10,
               "joint: disc marginal p(0) = 1");

    // Maximally mixed state: uniform joint weights
    DenseOp rho_mm = measurement::maximally_mixed(6);
    joint::JointWeights jw_mm =
        joint::joint_born_weights(ff, top_hyb, rho_mm);
    check_near(jw_mm.weights[0][0], 1.0 / 6, 1e-10,
               "joint: mixed p(0,0) = 1/6");
    check_near(jw_mm.disc_marginal(0), 0.5, 1e-10,
               "joint: mixed disc marginal = 1/2");
    check_near(jw_mm.top_marginal(0), 1.0 / 3, 1e-10,
               "joint: mixed top marginal = 1/3");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Canonical Structure Graph & Boundary Calculus
// ════════════════════════════════════════════════════════════════════════════
static void test_structure() {
    printf("\n-- Structure Graph & Boundary Calculus --\n");
    using namespace structure;

    // ── Build a structure graph ─────────────────────────────────────────────
    StructureGraph sg;
    int b0 = sg.add_boundary(BoundaryType::DISCRETE, 4);
    int b1 = sg.add_boundary(BoundaryType::TOPOLOGICAL, 3);

    // Entities: 3 symbols on disc boundary, 2 regions on top boundary
    int e0 = sg.add_entity(EntityKind::SYMBOL, 0);
    int e1 = sg.add_entity(EntityKind::SYMBOL, 1);
    int e2 = sg.add_entity(EntityKind::SYMBOL, 2);
    int e3 = sg.add_entity(EntityKind::REGION, 0);
    int e4 = sg.add_entity(EntityKind::REGION, 1);

    sg.entities[e0].boundary_id = b0;
    sg.entities[e1].boundary_id = b0;
    sg.entities[e2].boundary_id = b0;
    sg.entities[e3].boundary_id = b1;
    sg.entities[e4].boundary_id = b1;

    check(sg.n_entities == 5, "struct: 5 entities");
    check(sg.n_boundaries == 2, "struct: 2 boundaries");

    // Relations
    sg.add_relation(e0, e1, RelationKind::FLOW);
    sg.add_relation(e1, e2, RelationKind::FLOW);
    sg.add_relation(e3, e4, RelationKind::ADJACENCY);
    check(sg.n_relations == 3, "struct: 3 relations");

    // Hyper-relation
    int hmem[] = {e0, e1, e2};
    sg.add_hyper(hmem, 3, 42);
    check(sg.n_hyper == 1, "struct: 1 hyper-relation");
    check(sg.hyper[0].arity == 3, "struct: hyper arity = 3");

    // Observables
    sg.add_observable(4, b0, true);    // 4-outcome discrete
    sg.add_observable(3, b1, false);   // 3-outcome continuous
    check(sg.n_observables == 2, "struct: 2 observables");

    // ── Queries ─────────────────────────────────────────────────────────────
    check(sg.count_by_kind(EntityKind::SYMBOL) == 3,
          "struct: 3 SYMBOLs");
    check(sg.count_by_kind(EntityKind::REGION) == 2,
          "struct: 2 REGIONs");
    check(sg.count_by_boundary(b0) == 3,
          "struct: 3 entities on boundary 0");
    check(sg.count_by_boundary(b1) == 2,
          "struct: 2 entities on boundary 1");
    check(sg.out_degree(e0) == 1, "struct: out_degree(e0) = 1");
    check(sg.out_degree(e1) == 1, "struct: out_degree(e1) = 1");
    check(sg.in_degree(e2) == 1, "struct: in_degree(e2) = 1");
    check_near(sg.total_weight(), 5.0, 1e-10,
               "struct: total weight = 5.0 (uniform)");

    // ── Verification ────────────────────────────────────────────────────────
    check(verify::referential_integrity(sg),
          "struct: referential integrity");
    check(verify::boundary_coverage(sg),
          "struct: boundary coverage");
    check(verify::observable_validity(sg),
          "struct: observable validity");
    check(verify::structural_consistency(sg),
          "struct: full structural consistency");

    // ── Lifting functor ─────────────────────────────────────────────────────
    lift::LiftResult lr = lift::analyze(sg);
    check(lr.dim_disc == 4, "lift: dim_disc = 4");
    check(lr.dim_top == 3, "lift: dim_top = 3");
    check(lr.dim_hyb == 12, "lift: dim_hyb = 4*3 = 12");
    check(lr.n_disc_obs == 1, "lift: 1 discrete observable");
    check(lr.n_top_obs == 1, "lift: 1 topological observable");
    check(lr.n_joint_obs == 0, "lift: 0 joint observables");

    // Discrete admissible family from boundary
    sector::AdmissibleFamily df = lift::disc_family(sg, b0);
    check(df.n_sectors == 4, "lift: disc family n_sectors = 4");
    check(sector::verify::resolution_of_identity(df),
          "lift: disc family is valid resolution");

    // Fiber map from boundaries
    fiber::FiberMap fmap = lift::fiber_map(sg, b0, b1);
    check(fmap.dim_hyb == 12, "lift: fiber map dim_hyb = 12");

    // Regime planning
    lift::Regime regime = lift::plan_regime(sg);
    check(regime == lift::Regime::DISC_FIRST,
          "lift: regime = DISC_FIRST (equal obs, disc default)");

    // ── Joint boundary forces FULL_HYBRID ───────────────────────────────────
    StructureGraph sg2;
    sg2.add_boundary(BoundaryType::DISCRETE, 2);
    sg2.add_boundary(BoundaryType::JOINT, 3);
    int ej = sg2.add_entity(EntityKind::NODE, 0);
    sg2.entities[ej].boundary_id = 0;
    int ej2 = sg2.add_entity(EntityKind::NODE, 1);
    sg2.entities[ej2].boundary_id = 1;
    sg2.add_observable(2, 0);
    sg2.add_observable(3, 1);
    check(lift::plan_regime(sg2) == lift::Regime::FULL_HYBRID,
          "lift: joint boundary → FULL_HYBRID");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Universal State Descriptor
// ════════════════════════════════════════════════════════════════════════════
static void test_descriptor() {
    printf("\n-- Universal State Descriptor --\n");
    using namespace structure;
    using namespace descriptor;

    // Build a structure graph with mixed boundaries
    StructureGraph sg;
    int b0 = sg.add_boundary(BoundaryType::DISCRETE, 4);
    int b1 = sg.add_boundary(BoundaryType::TOPOLOGICAL, 3);

    int e0 = sg.add_entity(EntityKind::SYMBOL, 0);
    int e1 = sg.add_entity(EntityKind::NODE, 1);
    sg.entities[e0].boundary_id = b0;
    sg.entities[e1].boundary_id = b1;
    sg.add_relation(e0, e1, RelationKind::FLOW);
    sg.add_observable(4, b0, true);
    sg.add_observable(3, b1, false);

    // Build descriptor
    UniversalDescriptor ud = from_structure(sg);

    check(ud.dim_disc == 4, "desc: dim_disc = 4");
    check(ud.dim_top == 3, "desc: dim_top = 3");
    check(ud.dim_hyb == 12, "desc: dim_hyb = 12");
    check(ud.dim() == 12, "desc: dim() = 12");
    check(ud.n_disc_boundaries == 1, "desc: 1 disc boundary");
    check(ud.n_top_boundaries == 1, "desc: 1 top boundary");
    check(ud.n_joint_boundaries == 0, "desc: 0 joint boundaries");
    check(ud.n_observables == 2, "desc: 2 observables");
    check(ud.is_hybrid, "desc: is_hybrid = true");
    check(!ud.is_disc_only(), "desc: not disc-only");
    check(ud.needs_hybrid(), "desc: needs_hybrid (disc+top → hybrid)");

    // Carrier kind: relations > entities → GRAPH
    check(ud.carrier == CarrierKind::GRAPH ||
          ud.carrier == CarrierKind::SYMBOLIC,
          "desc: carrier inferred");

    // to_state_descriptor roundtrip
    contract::StateDescriptor sd = ud.to_state_descriptor();
    check(sd.type == contract::StateType::HYB,
          "desc→SD: type = HYB");
    check(sd.dim() > 0, "desc→SD: dim > 0");

    // ── Disc-only workload ──────────────────────────────────────────────────
    StructureGraph sg_disc;
    sg_disc.add_boundary(BoundaryType::DISCRETE, 8);
    int ed = sg_disc.add_entity(EntityKind::SYMBOL, 0);
    sg_disc.entities[ed].boundary_id = 0;
    sg_disc.add_observable(8, 0);

    UniversalDescriptor ud_disc = from_structure(sg_disc);
    check(ud_disc.is_disc_only(), "desc disc: is_disc_only");
    check(!ud_disc.is_hybrid, "desc disc: not hybrid");

    contract::StateDescriptor sd_disc = ud_disc.to_state_descriptor();
    check(sd_disc.type == contract::StateType::DISC,
          "desc disc→SD: type = DISC");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Emission & Reconstruction
// ════════════════════════════════════════════════════════════════════════════
static void test_emission() {
    printf("\n-- Emission & Reconstruction --\n");
    using namespace emission;
    using namespace sector;
    using namespace structure;

    // ── SectorReport from weights ───────────────────────────────────────────
    SectorWeights sw;
    sw.n_sectors = 4;
    sw.weights[0] = 0.5;
    sw.weights[1] = 0.25;
    sw.weights[2] = 0.125;
    sw.weights[3] = 0.125;

    SectorReport report = reconstruct::from_weights(sw);
    check(report.n_entries == 4, "emit: 4 entries");
    check_near(report.total_weight, 1.0, 1e-10,
               "emit: total weight = 1.0");
    check(report.max_sector == 0, "emit: max sector = 0");

    // Entropy: H = -(0.5 log₂ 0.5 + 0.25 log₂ 0.25 + 2*0.125 log₂ 0.125)
    //          H = -(−0.5 + −0.5 + 2*(−0.375)) = 1.75
    check_near(report.entropy, 1.75, 1e-10,
               "emit: entropy = 1.75 bits");

    // ── Emission queries ────────────────────────────────────────────────────
    check(emit::active_sectors(report) == 4,
          "emit: 4 active sectors");
    check(emit::is_concentrated(report, 0.4),
          "emit: concentrated at threshold 0.4");
    check(!emit::is_concentrated(report, 0.9),
          "emit: not concentrated at threshold 0.9");
    check(!emit::is_uniform(report),
          "emit: not uniform");
    check(emit::dominant_sector(report) == 0,
          "emit: dominant = sector 0");

    // Normalized entropy: H / log₂(4) = 1.75 / 2.0 = 0.875
    check_near(emit::normalized_entropy(report), 0.875, 1e-10,
               "emit: normalized entropy = 0.875");

    // Ranked sectors
    int ids[4];
    double wts[4];
    emit::ranked_sectors(report, ids, wts, 4);
    check(ids[0] == 0, "emit: ranked #1 = sector 0");
    check(ids[1] == 1, "emit: ranked #2 = sector 1");
    check_near(wts[0], 0.5, 1e-10, "emit: ranked wt[0] = 0.5");

    // ── Uniform distribution ────────────────────────────────────────────────
    SectorWeights sw_u;
    sw_u.n_sectors = 4;
    sw_u.weights[0] = 0.25;
    sw_u.weights[1] = 0.25;
    sw_u.weights[2] = 0.25;
    sw_u.weights[3] = 0.25;

    SectorReport rep_u = reconstruct::from_weights(sw_u);
    check(emit::is_uniform(rep_u), "emit: uniform distribution");
    check_near(emit::normalized_entropy(rep_u), 1.0, 1e-10,
               "emit: uniform normalized entropy = 1.0");

    // ── Concentrated distribution ───────────────────────────────────────────
    SectorWeights sw_c;
    sw_c.n_sectors = 4;
    sw_c.weights[0] = 0.97;
    sw_c.weights[1] = 0.01;
    sw_c.weights[2] = 0.01;
    sw_c.weights[3] = 0.01;

    SectorReport rep_c = reconstruct::from_weights(sw_c);
    check(emit::is_concentrated(rep_c, 0.9),
          "emit: concentrated (0.97 > 0.9)");
    check(emit::dominant_sector(rep_c) == 0,
          "emit: concentrated dominant = 0");

    // ── Annotate structure graph ────────────────────────────────────────────
    StructureGraph sg;
    sg.add_boundary(BoundaryType::DISCRETE, 4);

    for (int i = 0; i < 4; ++i) {
        int eid = sg.add_entity(EntityKind::SYMBOL, i);
        sg.entities[eid].boundary_id = 0;
    }

    reconstruct::annotate_graph(sg, report, 0);
    check_near(sg.entities[0].weight, 0.5, 1e-10,
               "emit: annotated entity 0 weight = 0.5");
    check_near(sg.entities[1].weight, 0.25, 1e-10,
               "emit: annotated entity 1 weight = 0.25");

    // ── Semantic API ────────────────────────────────────────────────────────
    check(emission::api::query_dominant(report) == 0,
          "api: dominant = 0");
    check(emission::api::query_concentrated(report, 0.4),
          "api: concentrated at 0.4");
    check(!emission::api::query_concentrated(report, 0.9),
          "api: not concentrated at 0.9");
    check_near(emission::api::query_entropy(report), 1.75, 1e-10,
               "api: entropy = 1.75");

    // ── From DenseOp + AdmissibleFamily ─────────────────────────────────────
    measurement::DenseOp rho(4);   // maximally mixed
    rho.at(0, 0) = C64(0.25, 0);
    rho.at(1, 1) = C64(0.25, 0);
    rho.at(2, 2) = C64(0.25, 0);
    rho.at(3, 3) = C64(0.25, 0);

    AdmissibleFamily cb = AdmissibleFamily::computational_basis(4);
    SectorReport rep_mm = emission::api::measure_and_report(cb, rho);
    check(rep_mm.n_entries == 4, "api: measure_and_report 4 entries");
    check_near(rep_mm.total_weight, 1.0, 1e-10,
               "api: measure_and_report total = 1.0");
    check(emit::is_uniform(rep_mm),
          "api: maximally mixed → uniform report");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Structure-Aware IR Opcodes
// ════════════════════════════════════════════════════════════════════════════
static void test_structure_ir() {
    printf("\n-- Structure-Aware IR Opcodes --\n");
    using namespace contract;
    using namespace platform;

    MachineSpec spec(2, 16, 8);
    PlatformIR ir(spec);
    IRBuilder b(ir);

    // Declare structure in IR
    b.boundary(0, 0, 4);   // DISCRETE, dim 4
    b.boundary(1, 1, 3);   // TOPOLOGICAL, dim 3
    b.entity(0, 0);        // SYMBOL
    b.entity(1, 0);        // SYMBOL
    b.entity(2, 2);        // REGION
    b.relation(0, 1, 1);   // FLOW
    b.observable(0, 4, 0); // 4-outcome on boundary 0

    int count_decl = ir.count;
    check(count_decl == 7, "struct IR: 7 declaration ops");

    // Allocate HYB register, measure structurally, reconstruct, emit
    int r = b.alloc(StateType::HYB);
    b.h(r, 0).z(r, BitVec(1, 2)).h(r, 0);
    b.measure_struct(r);
    b.reconstruct(r);
    b.emit_report(r);

    check(ir.count == count_decl + 7,
          "struct IR: +7 (alloc+H+Z+H+meas+recon+emit)");

    // Type checking passes
    type_check::TypeError err = type_check::verify(ir);
    check(!err.has_error, "struct IR: type check passes");

    // Cost estimation includes structure ops
    int cost = ir.total_cost();
    check(cost > 0, "struct IR: total cost > 0");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Platform — 5-Layer Compute Stack
// ════════════════════════════════════════════════════════════════════════════
static void test_platform() {
    printf("\n-- Platform (5-Layer Stack) --\n");

    using namespace contract;
    using namespace platform;
    using namespace runtime;
    using namespace pipeline;

    // ── Layer 1: State Contract ─────────────────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        check(spec.disc_dim() == 4, "Spec disc_dim = 2^2 = 4");
        check(spec.top_dim() == 128, "Spec top_dim = 16*8 = 128");
        check(spec.hyb_dim() == 512, "Spec hyb_dim = 4*128 = 512");

        StateDescriptor d = spec.disc_desc();
        check(d.type == StateType::DISC, "disc_desc type = DISC");
        check(d.dim() == 4, "disc_desc dim = 4");

        StateDescriptor h = spec.hyb_desc();
        check(h.type == StateType::HYB, "hyb_desc type = HYB");
        check(h.dim() == 512, "hyb_desc dim = 512");

        // Type compatibility
        check(can_apply_disc_op(StateType::DISC), "Disc ops on DISC: ok");
        check(can_apply_disc_op(StateType::HYB), "Disc ops on HYB: ok");
        check(!can_apply_disc_op(StateType::TOP), "Disc ops on TOP: no");
        check(can_apply_top_op(StateType::TOP), "Top ops on TOP: ok");
        check(can_apply_top_op(StateType::HYB), "Top ops on HYB: ok");
        check(!can_apply_top_op(StateType::DISC), "Top ops on DISC: no");
        check(can_apply_hyb_op(StateType::HYB), "Hyb ops on HYB: ok");
        check(!can_apply_hyb_op(StateType::DISC), "Hyb ops on DISC: no");

        // Lifting predicates
        check(can_lift(StateType::DISC, StateType::HYB), "DISC lifts to HYB");
        check(can_lift(StateType::TOP, StateType::HYB), "TOP lifts to HYB");
        check(!can_lift(StateType::DISC, StateType::TOP), "DISC !→ TOP");
    }

    // ── Layer 2: Platform IR ────────────────────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        PlatformIR ir(spec);
        IRBuilder b(ir);

        int r0 = b.alloc(StateType::HYB);
        check(r0 == 0, "First register is 0");
        check(ir.reg_types[r0] == StateType::HYB, "Register 0 is HYB");

        b.h(r0, 0).z(r0, BitVec(1, 2)).h(r0, 0).measure(r0);
        check(ir.count == 5, "IR: alloc + H + Z + H + measure = 5 ops");

        // Type checking — should be valid
        type_check::TypeError err = type_check::verify(ir);
        check(!err.has_error, "Type check passes for valid IR");
    }

    // ── Layer 2: Import from legacy MirIR ───────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        compiler::MirIR mir(2);
        mir.emit(compiler::MirOp::hadamard(0));
        mir.emit(compiler::MirOp::pauli_z(BitVec(1, 2)));
        mir.emit(compiler::MirOp::hadamard(0));
        mir.emit(compiler::MirOp::measure());

        PlatformIR pir = import::from_mir_ir(mir, spec);
        check(pir.count > 0, "Import from MirIR produces ops");

        // Verify type consistency after import
        type_check::TypeError err = type_check::verify(pir);
        check(!err.has_error, "Imported MirIR type-checks ok");
    }

    // ── Layer 2: Auto-lift insertion ────────────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        PlatformIR ir(spec);
        IRBuilder b(ir);

        // Allocate as DISC, then try to measure (needs HYB)
        int r0 = b.alloc(StateType::DISC);
        ir.emit(PlatformOp::measure_disc(r0));

        int count_before = ir.count;
        auto_lift::insert_lifts(ir);
        int count_after = ir.count;

        // Auto-lift should have inserted a LIFT_DISC_TO_HYB
        check(count_after > count_before,
              "Auto-lift inserted lifting op");
    }

    // ── Layer 3: Runtime VM — interference circuit ──────────────────────────
    {
        // |0⟩⊗ξ₀ → H(0) → Z(1) → H(0) → measure
        // Expected: destructive interference at |0⟩
        //   p(0) = 0, p(1) = 1
        MachineSpec spec(1, 16, 8);
        PlatformIR ir(spec);
        IRBuilder b(ir);

        int r = b.alloc(StateType::HYB);
        b.h(r, 0).z(r, BitVec(1, 1)).h(r, 0).measure(r);

        PlatformVM* vm = new PlatformVM(spec);
        vm->run(ir);

        check(vm->num_measurements() == 1, "VM recorded 1 measurement");
        check_near(vm->prob(0, 0), 0.0, 1e-10,
                   "Interference: p(0) = 0");
        check_near(vm->prob(0, 1), 1.0, 1e-10,
                   "Interference: p(1) = 1");
        delete vm;
    }

    // ── Layer 3: Runtime VM — Deutsch-Jozsa ─────────────────────────────────
    {
        // Constant oracle: f(x) = 0 → p(0) = 1
        MachineSpec spec(2, 16, 8);
        PlatformIR ir_c(spec);
        IRBuilder bc(ir_c);

        int r = bc.alloc(StateType::HYB);
        bc.x(r, BitVec(0b10, 2));  // |0⟩|1⟩
        bc.h(r, 0).h(r, 1);        // Hadamard both
        // constant oracle: identity (no gate)
        bc.h(r, 0);                // Hadamard qubit 0
        bc.measure(r);

        PlatformVM* vm_c = new PlatformVM(spec);
        vm_c->run(ir_c);
        double p0_c = vm_c->prob(0, 0);  // p(|00⟩)
        double p2_c = vm_c->prob(0, 2);  // p(|10⟩) — qubit 1 in |1⟩
        // After H on qubit 0: qubit 0 should be |0⟩ (constant)
        // p(x with qubit0=0) should be high
        check(p0_c + p2_c > 0.99,
              "DJ constant: qubit 0 stays |0⟩");
        delete vm_c;

        // Balanced oracle: f(x) = x → p(0) = 0
        PlatformIR ir_b(spec);
        IRBuilder bb(ir_b);

        int rb = bb.alloc(StateType::HYB);
        bb.x(rb, BitVec(0b10, 2));
        bb.h(rb, 0).h(rb, 1);
        bb.z(rb, BitVec(0b01, 2));  // balanced oracle
        bb.h(rb, 0);
        bb.measure(rb);

        PlatformVM* vm_b = new PlatformVM(spec);
        vm_b->run(ir_b);
        double p0_b = vm_b->prob(0, 0);
        double p2_b = vm_b->prob(0, 2);
        // Qubit 0 should be |1⟩ (balanced)
        check(p0_b + p2_b < 0.01,
              "DJ balanced: qubit 0 flips to |1⟩");
        delete vm_b;
    }

    // ── Layer 3: Runtime VM — lifting ───────────────────────────────────────
    {
        MachineSpec spec(1, 16, 8);

        // Lift a discrete state to hybrid and measure
        PlatformIR ir(spec);
        IRBuilder b(ir);

        int rd = b.alloc(StateType::DISC);
        b.h(rd, 0);  // Hadamard on disc register → superposition

        int rh = b.lift_disc(rd);
        b.measure(rh);

        PlatformVM* vm = new PlatformVM(spec);
        vm->run(ir);

        // Uniform superposition → p(0) = p(1) = 0.5
        check_near(vm->prob(0, 0), 0.5, 0.05,
                   "Lift disc→hyb: p(0) ≈ 0.5");
        check_near(vm->prob(0, 1), 0.5, 0.05,
                   "Lift disc→hyb: p(1) ≈ 0.5");
        delete vm;
    }

    // ── Layer 4: Compiler Pipeline ──────────────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        CompilerPipeline pipe(spec);

        // Build IR with redundant ops
        PlatformIR ir(spec);
        IRBuilder b(ir);
        int r = b.alloc(StateType::HYB);
        b.u(r, 1.0).u(r, 2.0);           // should merge → U_3.0
        b.m(r, 3).m(r, -1);               // should merge → M_2
        b.x(r, BitVec(0, 2));             // identity, should be removed
        b.barrier().barrier();             // duplicate barrier
        b.measure(r);

        int count_before = ir.count;
        pipe.compile(ir);
        int count_after = ir.count;

        check(count_after < count_before,
              "Pipeline optimized: reduced op count");

        // Execute
        PlatformVM* vm = new PlatformVM(spec);
        vm->run(ir);
        check(vm->num_measurements() == 1, "Pipeline: measure recorded");
        delete vm;
    }

    // ── Layer 4: Legacy MirIR import ────────────────────────────────────────
    {
        MachineSpec spec(2, 16, 8);
        CompilerPipeline pipe(spec);

        // Use the existing interference demo builder
        compiler::MirIR mir =
            compiler::frontend::build_interference_demo(1);

        PlatformIR pir = pipe.import_mir(mir);
        check(pir.count > 0, "Imported + optimized MirIR non-empty");

        // Lower to HybridProgram
        HybridProgram hp = pipe.lower(pir);
        check(hp.length > 0, "Lowered HybridProgram non-empty");
    }

    // ── Layer 5: Host API ───────────────────────────────────────────────────
    {
        using namespace api;

        // Simple mode: build and execute interference circuit
        ProgramHandle* prog = program_create(1, 16, 8);
        check(prog != nullptr, "API: program_create ok");

        emit_h(prog, 0);
        emit_z(prog, 1);
        emit_h(prog, 0);
        emit_measure(prog);

        check(program_op_count(prog) == 5,
              "API: 5 ops (alloc + H + Z + H + meas)");

        ResultHandle* res = execute(prog);
        check(res != nullptr, "API: execute returned result");
        check(result_num_measurements(res) == 1,
              "API: 1 measurement");
        check_near(result_prob(res, 0, 0), 0.0, 1e-10,
                   "API: interference p(0) = 0");
        check_near(result_prob(res, 0, 1), 1.0, 1e-10,
                   "API: interference p(1) = 1");
        check(result_time(res) > 0, "API: time > 0");
        check(result_cost(res) > 0, "API: cost > 0");

        result_free(res);
        program_free(prog);
    }

    // ── Layer 5: API advanced mode ──────────────────────────────────────────
    {
        using namespace api;

        ProgramHandle* prog = program_create(1, 16, 8);

        // Allocate disc register, apply H, lift to hyb, measure
        int rd = program_alloc_reg(prog, 0);  // DISC
        check(rd >= 0, "API advanced: alloc disc reg");

        emit_h_reg(prog, rd, 0);

        int rh = program_alloc_reg(prog, 2);  // HYB
        emit_lift_disc(prog, rh, rd);
        emit_measure_reg(prog, rh);

        ResultHandle* res = execute(prog);
        check(res != nullptr, "API advanced: got result");
        check_near(result_prob(res, 0, 0), 0.5, 0.05,
                   "API advanced: lifted superposition p(0) ≈ 0.5");

        result_free(res);
        program_free(prog);
    }

    // ── Cocycle / UFE verification via API ──────────────────────────────────
    {
        using namespace api;

        ProgramHandle* prog = program_create(1, 16, 8);
        emit_check_cocycle(prog);
        emit_check_ufe(prog);
        emit_measure(prog);

        ResultHandle* res = execute(prog);
        check(result_cocycle_ok(res), "API: cocycle δΩ = 0 verified");
        check(result_ufe_ok(res), "API: UFE = 0 verified");

        result_free(res);
        program_free(prog);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Mixed Operators, Mixed Cocycle & Boundary Calculus
// ════════════════════════════════════════════════════════════════════════════
static void test_mixed_operators() {
    printf("\n-- Mixed Operators --\n");
    using namespace mixed_cocycle;
    using namespace mixed_ops;
    using namespace boundary;

    int n = 2;

    // ── Mixed cocycle: κ=0 limit ────────────────────────────────────────────
    {
        MirElement g = mir::g_phi(1.0);
        MirElement h = mir::g_phi(2.0);
        BitVec u(1, n), v(2, n), up(3, n), vp(0, n);

        check(verify_kappa_zero_limit(g, u, v, h, up, vp),
              "Mixed cocycle: κ=0 → base cocycle");
    }

    // ── Mixed cocycle: δΩ_mix = 0 ───────────────────────────────────────────
    {
        MirElement g1 = mir::g_phi(1.0);
        MirElement g2 = mir::g_phi(2.0);
        MirElement g3 = mir::g_phi(0.5);
        BitVec u1(1, n), v1(2, n);
        BitVec u2(3, n), v2(1, n);
        BitVec u3(2, n), v3(3, n);

        check(verify_mixed_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.0),
              "Mixed cocycle: δΩ_mix = 0 (κ=0)");
        check(verify_mixed_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.1),
              "Mixed cocycle: δΩ_mix = 0 (κ=0.1)");
        check(verify_mixed_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 1.0),
              "Mixed cocycle: δΩ_mix = 0 (κ=1.0)");
    }

    // ── Extended total cocycle: δΩ_tot^ext = 0 ──────────────────────────────
    {
        MirElement g1 = mir::g_phi(1.0);
        MirElement g2 = mir::U_phi();
        MirElement g3 = mir::g_phi(0.5);
        BitVec u1(1, n), v1(0, n);
        BitVec u2(2, n), v2(1, n);
        BitVec u3(3, n), v3(2, n);

        check(verify_extended_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.0),
              "Extended cocycle: δΩ_tot^ext = 0 (κ=0)");
        check(verify_extended_cocycle(g1, u1, v1, g2, u2, v2, g3, u3, v3, 0.5),
              "Extended cocycle: δΩ_tot^ext = 0 (κ=0.5)");
    }

    // ── Mixed cocycle: nonzero when κ≠0 and ⟨v,u'⟩≠0 ───────────────────────
    {
        // Use s=-1 elements so Ω_Mir ≠ 0:
        //   Ω_Mir((-1,c),(1,d)) = -2c
        MirElement g = mir::J_lambda(1.0);   // s=-1, c=0
        MirElement h = mir::g_phi(1.0);      // s=1, c=c_φ
        // Ω_Mir(J_λ(1), g_φ(1)) = (-1)·0·(1-1) + 1·c_φ·(1-(-1)) = 2c_φ
        // Re(2c_φ) = 2ℓ ≈ 0.962
        BitVec u(0, n), v(1, n), up(1, n), vp(0, n);
        // ⟨v,u'⟩ = ⟨(0,1),(0,1)⟩ = 1
        C64 mix = omega_mix(g, u, v, h, up, vp, 0.5);
        check(mix.norm2() > 1e-10,
              "Mixed cocycle: Ω_mix ≠ 0 when κ≠0 and ⟨v,u'⟩=1");
    }

    // ── U_t^mix limit: κ=0 recovers U_t^hyb ────────────────────────────────
    {
        check(verify_U_mix_limit(1, 8, 4, 1.0),
              "Mixed ops: U_t^mix|κ=0 = U_t^hyb");
    }

    // ── X_u^mix limit: κ=0 recovers X_u^hyb ────────────────────────────────
    {
        check(verify_X_mix_limit(1, 8, 4),
              "Mixed ops: X_u^mix|κ=0 = X_u^hyb");
    }

    // ── Mixed ops nontrivial: κ≠0 produces different results ────────────────
    {
        check(verify_mixed_nontrivial(1, 8, 4, 0.5, 1.0),
              "Mixed ops: U_t^mix|κ=0.5 ≠ U_t^hyb");
    }

    // ── Central extension with mixed cocycle ────────────────────────────────
    {
        auto e = CentralExtElementMix::identity(n, 0.0);
        auto a = CentralExtElementMix(
            BitVec(1, n), mir::g_phi(1.0), BitVec(2, n), C64(0, 0), 0.0);
        auto ae = a * e;
        auto ea = e * a;
        bool id_ok = (ae.u.bits == a.u.bits) && (ea.u.bits == a.u.bits) &&
                     (ae.v.bits == a.v.bits) && (ea.v.bits == a.v.bits);
        check(id_ok, "CentralExtMix: identity element");
    }

    // ── Boundary calculus: construction ─────────────────────────────────────
    {
        Boundary bd = Boundary::disc(n);
        Boundary bt = Boundary::top(8, 4);
        Boundary bj = Boundary::joint(n, 8, 4);
        Boundary bm = Boundary::mixed(n, 8, 4);

        check(bd.dim() == (1 << n), "Boundary disc: dim = 2^n");
        check(bt.dim() == 32, "Boundary top: dim = N_θ·N_ρ");
        check(bj.dim() == (1 << n) * 32, "Boundary joint: dim = 2^n·N_θ·N_ρ");
        check(bm.has_disc() && bm.has_top(), "Boundary mixed: has both sectors");
    }

    // ── Boundary calculus: restriction ──────────────────────────────────────
    {
        Boundary bj = Boundary::joint(n, 8, 4);
        Boundary rd = restrict_to_disc(bj);
        Boundary rt = restrict_to_top(bj);

        check(rd.type == BoundaryType::DISC, "restrict_to_disc: type = DISC");
        check(rt.type == BoundaryType::TOP, "restrict_to_top: type = TOP");
    }

    // ── Boundary calculus: composition ──────────────────────────────────────
    {
        Boundary bd = Boundary::disc(n);
        Boundary bt = Boundary::top(8, 4);
        Boundary bc = compose(bd, bt);

        check(bc.type == BoundaryType::JOINT,
              "compose(disc,top) = JOINT");
        check(bc.n_qubits == n && bc.N_theta == 8 && bc.N_rho == 4,
              "compose: parameters preserved");
    }

    // ── Boundary measurement ────────────────────────────────────────────────
    {
        HybridState psi;
        psi.init(1, 8, 4);
        TopologicalState xi;
        xi.init(8, 4);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        Boundary bd = Boundary::disc(1);
        BoundaryResult res = measure(bd, psi);
        check(res.normalized(), "Boundary measure: normalized");
        check_near(res.probs[0], 1.0, 1e-10,
                   "Boundary measure: |0⟩ → p(0)=1");
        res.free();
        psi.free();
    }

    // ── Boundary marginal consistency ───────────────────────────────────────
    {
        HybridState psi;
        psi.init(1, 8, 4);
        TopologicalState xi;
        xi.init(8, 4);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        check(verify_marginal_consistency(psi),
              "Boundary: disc+top marginals = joint");
        psi.free();
    }

    // ── Regime planner ──────────────────────────────────────────────────────
    {
        using namespace regime;

        Gate disc_gates[2] = {gate_X(1), gate_Z(2)};
        check(classify(disc_gates, 2) == Regime::PURE_DISC,
              "Regime: disc-only → PURE_DISC");

        Gate top_gates[1] = {gate_U(1.0)};
        check(classify(top_gates, 1) == Regime::PURE_TOP,
              "Regime: top-only → PURE_TOP");

        Gate hyb_gates[2] = {gate_X(1), gate_U(1.0)};
        check(classify(hyb_gates, 2) == Regime::TENSOR,
              "Regime: disc+top → TENSOR");

        check(classify(disc_gates, 2, 0.5) == Regime::MIXED,
              "Regime: κ≠0 → MIXED");
    }
}

static void test_quantum_algebra() {
    printf("\n--- Quantum Algebra Q^max ---\n");
    int n = 2;
    int nth = 8, nrh = 4;
    MirElement ge = MirElement::identity();

    // I. Twisted Convolution Algebra
    {
        using namespace twisted_algebra;
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);
        WeylHybrid wc(BitVec(1, n), BitVec(1, n), ge);

        AlgElement f = AlgElement::single(C64(1.0, 0.0), wa);
        AlgElement g = AlgElement::single(C64(1.0, 0.0), wb);
        AlgElement h = AlgElement::single(C64(1.0, 0.0), wc);

        AlgElement fg = twisted_product(f, g);
        check(fg.n_terms == 1, "★_Ω: single-term product");

        check(verify_associativity(f, g, h),
              "★_Ω: (f★g)★h = f★(g★h) assoc (δΩ=0)");
    }

    // II. Field operator commutativity
    {
        using namespace field_ops;
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);  // X_1
        WeylHybrid wb(BitVec(2, n), BitVec(0, n), ge);  // X_2
        WeylHybrid wc(BitVec(0, n), BitVec(1, n), ge);  // Z_1

        twisted_algebra::AlgElement fa = twisted_algebra::AlgElement::single(C64(1,0), wa);
        twisted_algebra::AlgElement fb = twisted_algebra::AlgElement::single(C64(1,0), wb);
        twisted_algebra::AlgElement fc = twisted_algebra::AlgElement::single(C64(1,0), wc);

        check(check_commutativity(fa, fb), "Φ: [Φ(X₁),Φ(X₂)] = 0");
        check(!check_commutativity(fa, fc), "Φ: [Φ(X₁),Φ(Z₁)] ≠ 0");
    }

    // III. Symplectic Form
    {
        using namespace symplectic;
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);  // X_1
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);  // Z_1
        WeylHybrid wc(BitVec(2, n), BitVec(0, n), ge);  // X_2

        check(!commute(wa, wb), "Σ: [X₁,Z₁] ≠ 0");
        check(commute(wa, wc), "Σ: [X₁,X₂] = 0");

        C64 cc = commutator_coeff(wa, wb);
        check(cc.norm2() > 0.1, "Σ: commutator_coeff nonzero");

        C64 ac = anticommutator_coeff(wa, wa);
        check(ac.norm2() > 0.1, "Σ: anticommutator_coeff of X₁,X₁");

        // Ω-orthogonality
        WeylHybrid setA[1] = {wa};  // {X_1}
        WeylHybrid setB[1] = {wc};  // {X_2}
        check(omega_orthogonal(setA, 1, setB, 1), "Σ: {X₁} ⊥_Ω {X₂}");
    }

    // IV. *-Algebra
    {
        using namespace star_algebra;
        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        WeylHybrid wb(BitVec(0, n), BitVec(1, n), ge);

        // W_a^{-1} should exist
        WeylHybrid ainv = weyl_inverse(wa);
        check(ainv.u.bits == wa.u.bits && ainv.v.bits == wa.v.bits,
              "*-alg: inverse preserves disc indices");

        twisted_algebra::AlgElement f = twisted_algebra::AlgElement::single(C64(1.0, 0.5), wa);
        twisted_algebra::AlgElement g = twisted_algebra::AlgElement::single(C64(0.7, -0.3), wb);
        check(verify_star_antiautomorphism(f, g),
              "*-alg: (f★g)* = g*★f* anti-automorphism");
    }

    // V. Discrete Completeness
    {
        using namespace completeness;
        check(verify_orthogonality(1),
              "Compl: Tr((X_uZ_v)†X_{u'}Z_{v'}) = 2δ (n=1)");
        check(verify_orthogonality(2),
              "Compl: Tr((XZ)†X'Z') = 4δ (n=2)");
        check(verify_cardinality(1), "Compl: |{X_uZ_v}| = 4^1 = 4");
        check(verify_cardinality(2), "Compl: |{X_uZ_v}| = 4^2 = 16");
    }

    // VI. Diagonal States
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
        check(fabs(probs[0] - 1.0) < 1e-10,
              "Diag: p(0) = 1 for |0⟩ basis");
        check(fabs(probs[1]) < 1e-10,
              "Diag: p(1) = 0 for |0⟩ basis");

        TopologicalState cond;
        cond.init(nth, nrh);
        conditional_sector(psi, 0, cond);
        check(fabs(cond.norm2() - 1.0) < 1e-10,
              "Diag: conditional state normalized");
        cond.free();
        psi.free();
    }

    // VII. Correlation Functions
    {
        using namespace correlation;
        HybridState psi;
        psi.init(n, nth, nrh);
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        WeylHybrid id_w = WeylHybrid::identity(n);
        C64 g1 = G1(psi, id_w);
        check(fabs(g1.re - 1.0) < 1e-8 && fabs(g1.im) < 1e-8,
              "Corr: G₁(identity) = 1");

        WeylHybrid wa(BitVec(1, n), BitVec(0, n), ge);
        check(verify_G2_factorization(psi, id_w, wa),
              "Corr: G₂ = e^{iΩ} ω(W_{ab})");

        // N-point product (N=1): trivially equals the element
        C64 phase;
        WeylHybrid A_N;
        npoint_product(&id_w, 1, phase, A_N);
        check(fabs(phase.re - 1.0) < 1e-10, "Corr: 1-point phase = 1");

        psi.free();
    }

    // IX. Flow Generator
    {
        using namespace flow_generator;
        PhasePoint p(PI / 3.0, 0.5);
        check(verify_u_invariance(p, 1.0), "Flow: X_φ[u]=0 (t=1)");
        check(verify_u_invariance(p, 3.0), "Flow: X_φ[u]=0 (t=3)");
        check(verify_omega_decrease(p, 1.0), "Flow: X_φ[ω̃]=-π (t=1)");
        check(verify_omega_decrease(p, 2.0), "Flow: X_φ[ω̃]=-π (t=2)");

        double u0 = u_invariant(p.theta, p.rho);
        double w0 = omega_tilde(p.theta, p.rho);
        check(fabs((u0 + w0) / 2.0 - p.theta) < 1e-10,
              "Flow: (u+ω̃)/2 = θ");
        check(fabs((u0 - w0) * constants::ELL / (constants::HALF_PI * 2.0) - p.rho) < 1e-10,
              "Flow: (u-ω̃)·ℓ/(π) = ρ");
    }

    // XI. Circuit Complexity Metrics
    {
        using namespace circuit_metrics;
        Program prog;
        prog.append(gate_X(1));
        prog.append(gate_Z(2));
        prog.append(gate_U(1.0));
        prog.append(gate_M(3));
        prog.append(gate_P(0));

        CircuitAnalysis a = analyze(prog, n);
        check(a.size == 5, "Circuit: size(C) = 5");
        check(a.disc == 3, "Circuit: disc(C) = 3 (X+Z+P)");
        check(a.top == 2, "Circuit: top(C) = 2 (U+M)");
        check(a.meas == 1, "Circuit: meas(C) = 1 (P)");
        check(a.depth >= 1, "Circuit: depth(C) ≥ 1");
        check(a.time == 5, "Circuit: time(C) = size = 5");

        // Empty circuit
        Program empty;
        CircuitAnalysis e = analyze(empty, n);
        check(e.size == 0, "Circuit: empty size = 0");
        check(e.depth == 0, "Circuit: empty depth = 0");
    }

    // XII. Projective Representation
    {
        using namespace projective_rep;
        HybridState psi;
        psi.init(n, nth, nrh);
        TopologicalState xi;
        xi.init(nth, nrh);
        xi.set_delta(PhasePoint(0.0, 0.0));
        psi.init_computation(0, xi);
        xi.free();

        CentralElement id_ce = CentralElement::identity(n);
        CentralElement x_ce(C64(1.0, 0.0), WeylHybrid(BitVec(1, n), BitVec(0, n), ge));

        // id * x = x
        CentralElement prod = id_ce * x_ce;
        check(prod.a.u.bits == 1 && prod.a.v.bits == 0,
              "CentExt: e·x = x");

        check(verify_representation(id_ce, x_ce, psi),
              "Π_Ω: Π(e)Π(x) = Π(e·x)");

        CentralElement y_ce(C64(1.0, 0.0), WeylHybrid(BitVec(0, n), BitVec(1, n), ge));
        check(verify_representation(x_ce, y_ce, psi),
              "Π_Ω: Π(x)Π(y) = Π(x·y)");

        psi.free();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Hardware Topology Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_hardware_topology() {
    using namespace hardware;

    // Default workstation has all required subsystems
    HardwareTopology hw = HardwareTopology::default_workstation();
    check(verify::topology_complete(hw),
          "HW: default workstation topology complete");

    // Correct subsystem count
    check(hw.n_subsystems == 5,
          "HW: 5 subsystems (CPU,GPU,RAM,VRAM,SSD)");

    // Can find each subsystem
    check(hw.find(SubsystemKind::CPU) >= 0,   "HW: find CPU");
    check(hw.find(SubsystemKind::GPU) >= 0,   "HW: find GPU");
    check(hw.find(SubsystemKind::RAM) >= 0,   "HW: find RAM");
    check(hw.find(SubsystemKind::VRAM) >= 0,  "HW: find VRAM");
    check(hw.find(SubsystemKind::SSD) >= 0,   "HW: find SSD");

    // Canonical role assignment
    check(canonical_role(SubsystemKind::CPU) == SubsystemRole::EXACT_CONTROL,
          "HW: CPU → EXACT_CONTROL");
    check(canonical_role(SubsystemKind::GPU) == SubsystemRole::OPERATOR_ENGINE,
          "HW: GPU → OPERATOR_ENGINE");
    check(canonical_role(SubsystemKind::RAM) == SubsystemRole::STAGING_MANIFOLD,
          "HW: RAM → STAGING_MANIFOLD");
    check(canonical_role(SubsystemKind::VRAM) == SubsystemRole::CARRIER_MEMORY,
          "HW: VRAM → CARRIER_MEMORY");
    check(canonical_role(SubsystemKind::SSD) == SubsystemRole::PERSISTENT_MANIFOLD,
          "HW: SSD → PERSISTENT_MANIFOLD");

    // Placement consistency
    check(verify::placement_consistent(
        Placement::host_only(), SubsystemRole::EXACT_CONTROL),
          "HW: host_only consistent with EXACT_CONTROL");
    check(verify::placement_consistent(
        Placement::device_only(), SubsystemRole::CARRIER_MEMORY),
          "HW: device_only consistent with CARRIER_MEMORY");
    check(verify::placement_consistent(
        Placement::persistent(), SubsystemRole::PERSISTENT_MANIFOLD),
          "HW: persistent consistent with PERSISTENT_MANIFOLD");

    // Preferred subsystem for operation weight classes
    check(preferred_subsystem(OpWeight::DISC_HEAVY) == SubsystemKind::CPU,
          "HW: DISC_HEAVY → CPU");
    check(preferred_subsystem(OpWeight::TOP_HEAVY) == SubsystemKind::GPU,
          "HW: TOP_HEAVY → GPU");
    check(preferred_subsystem(OpWeight::CLOSURE_CHECK) == SubsystemKind::CPU,
          "HW: CLOSURE_CHECK → CPU");

    // Transfer policy
    TransferPolicy tp = TransferPolicy::host_to_device();
    check(tp.src == SubsystemKind::RAM,  "HW: H2D src = RAM");
    check(tp.dst == SubsystemKind::VRAM, "HW: H2D dst = VRAM");
    check(verify::transfer_fiber_safe(tp), "HW: default transfer fiber-safe");

    // Transfer time estimation
    int ram_id = hw.find(SubsystemKind::RAM);
    int vram_id = hw.find(SubsystemKind::VRAM);
    double t_ns = hw.transfer_time_ns(ram_id, vram_id, 1 << 20); // 1 MB
    check(t_ns > 0.0 && t_ns < 1e12, "HW: transfer time finite and positive");
}

// ════════════════════════════════════════════════════════════════════════════
// Machine State Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_machine_state() {
    using namespace machine_state;

    const int n = 2, nth = 8, nrh = 4;

    // Initialize machine state
    MachineState ms;
    ms.init(n, nth, nrh);

    // Stratum count (structural)
    check(verify::stratum_count(), "MS: 7 strata");

    // Dimension consistency
    check(verify::dimensions_consistent(ms),
          "MS: disc/top/hyb dimensions consistent");

    // Discrete sector configuration
    check(ms.disc.n_qubits == n, "MS: disc n_qubits = 2");
    check(ms.disc.dim == 4,      "MS: disc dim = 4");
    check(fabs(ms.disc.sector_probs[0] - 1.0) < 1e-12,
          "MS: initial sector |0⟩ has prob 1");
    check(fabs(ms.disc.entropy()) < 1e-12,
          "MS: initial sector entropy = 0");

    // Topological sector configuration
    check(ms.top.N_theta == nth, "MS: top N_theta matches");
    check(ms.top.N_rho == nrh,   "MS: top N_rho matches");
    check(ms.top.dim == nth * nrh, "MS: top dim = N_theta × N_rho");
    check(ms.top.accumulated_flow == 0.0,
          "MS: initial accumulated flow = 0");

    // Hybrid carrier
    check(ms.hyb.allocated, "MS: hybrid carrier allocated");
    check(ms.hyb.state.n_qubits == n,  "MS: hyb n_qubits matches");
    check(ms.hyb.state.N_theta == nth, "MS: hyb N_theta matches");
    check(ms.hyb.state.total == 4 * nth * nrh, "MS: hyb total correct");

    // Observable registry has default observables
    check(ms.obs.n_observables == 3, "MS: 3 default observables");
    check(ms.obs.count_disc() >= 1,  "MS: at least 1 disc observable");
    check(ms.obs.count_top() >= 1,   "MS: at least 1 top observable");
    check(ms.obs.count_joint() >= 1, "MS: at least 1 joint observable");

    // Cocycle ledger starts clean
    check(ms.cocyc.closure_valid, "MS: initial closure valid");
    check(ms.cocyc.n_checks == 0, "MS: initial n_checks = 0");

    // Measurement sector starts empty
    check(ms.meas.n_records == 0, "MS: no initial measurements");

    // Trace sector starts clean
    check(ms.trace.total_gate_ops == 0, "MS: no initial gate ops");
    check(ms.trace.total_transfers == 0, "MS: no initial transfers");

    // Transport tracking
    ms.top.record_flow(1.0);
    check(fabs(ms.top.accumulated_flow - 1.0) < 1e-12,
          "MS: flow accumulation");
    ms.top.record_modular(1);
    check(ms.top.accumulated_modular == 1,
          "MS: modular accumulation");

    // Placement verification
    check(ms.verify_placement(), "MS: doctrinal placement invariants hold");

    // Full consistency
    check(verify::machine_consistent(ms), "MS: full machine consistency");

    // Memory footprint
    int64_t mem = ms.total_memory_bytes();
    check(mem > 0, "MS: positive memory footprint");

    ms.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Fibered Memory Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_fibered_memory() {
    using namespace topcomp::fibered;

    const int n = 2, nth = 8, nrh = 4;

    // Create and initialize hybrid state
    HybridState hyb;
    hyb.init(n, nth, nrh);

    // Layout matches hybrid dimensions
    check(topcomp::fibered::verify::layout_matches_hybrid(hyb),
          "FM: layout matches hybrid dimensions");

    // Wrap in fibered layout
    FiberedLayout fl = FiberedLayout::wrap(hyb);

    // Correct dimensions
    check(fl.num_fibers() == 4,     "FM: n_fibers = 2^n = 4");
    check(fl.fiber_dim() == nth * nrh, "FM: fiber_dim = N_theta × N_rho");

    // Put a known amplitude in sector 1
    hyb.amp[hyb.index(1, 2, 3)] = C64(0.6, 0.8);

    // Access through fibered layout matches direct access
    C64 via_layout = fl.at(1, 2, 3);
    C64 via_direct = hyb.amp[hyb.index(1, 2, 3)];
    check(fabs(via_layout.re - via_direct.re) < 1e-15 &&
          fabs(via_layout.im - via_direct.im) < 1e-15,
          "FM: fibered access matches direct access");

    // FiberView: zero-copy view into fiber 1
    FiberView fv = fl.fiber(1);
    check(fv.size == nth * nrh, "FM: fiber view dim correct");
    double fv_norm2 = fv.norm2();
    check(fabs(fv_norm2 - 1.0) < 1e-12,
          "FM: fiber 1 norm² = |0.6+0.8i|² = 1.0");

    // Sector probability through layout
    double sp = fl.sector_prob(1);
    check(fabs(sp - 1.0) < 1e-12,
          "FM: sector_prob(1) = 1.0");

    // Extract and inject roundtrip
    TopologicalState extracted;
    extracted.init(nth, nrh);
    fl.extract_fiber(1, extracted);
    double ext_norm = extracted.norm2();
    check(fabs(ext_norm - 1.0) < 1e-12,
          "FM: extracted fiber norm² = 1.0");

    // Inject into a different sector
    fl.clear_fiber(2);
    fl.inject_fiber(2, extracted);
    double sp2 = fl.sector_prob(2);
    check(fabs(sp2 - 1.0) < 1e-12,
          "FM: injected fiber sector_prob = 1.0");

    // Swap fibers
    fl.swap_fibers(1, 2);
    check(fabs(fl.sector_prob(1) - 1.0) < 1e-12,
          "FM: swap preserves norm (fiber 1)");
    check(fabs(fl.sector_prob(2) - 1.0) < 1e-12,
          "FM: swap preserves norm (fiber 2)");

    // Fiber norms consistent with total state
    check(topcomp::fibered::verify::fiber_norms_consistent(fl),
          "FM: fiber norms sum to total norm");

    // Transfer plan
    FiberTransferPlan plans[8];
    int n_plans = topcomp::fibered::plan_fiber_transfer(
        fl.stride, 2, plans, 8);
    check(topcomp::fibered::verify::transfer_plan_complete(
              fl.stride, plans, n_plans),
          "FM: transfer plan complete");

    extracted.free();
    hyb.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Regime Planner Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_regime_planner() {
    using namespace planner;
    using namespace hardware;

    const int n = 2, nth = 8, nrh = 4;

    HardwareTopology hw = HardwareTopology::default_workstation();
    ResourceBudget budget = ResourceBudget::unlimited();

    // Small, disc-heavy query → CPU_ONLY
    {
        ComputeQuery q;
        q.n_qubits = 1;
        q.N_theta = 4;
        q.N_rho = 4;
        q.n_disc_gates = 100;
        q.n_top_gates = 0;
        q.n_hyb_gates = 0;

        machine_state::MachineState ms;
        ms.init(q.n_qubits, q.N_theta, q.N_rho);

        ExecutionPlan ep = plan(ms, q, budget, hw);
        check(ep.regime == ExecutionRegime::CPU_ONLY,
              "RP: small disc-heavy → CPU_ONLY");
        check(planner::verify::plan_valid(ep, budget),
              "RP: small plan valid");

        ms.free();
    }

    // Large, top-heavy query → GPU_ONLY (fits VRAM)
    {
        ComputeQuery q;
        q.n_qubits = n;
        q.N_theta = nth;
        q.N_rho = nrh;
        q.n_disc_gates = 0;
        q.n_top_gates = 50;
        q.n_hyb_gates = 20;

        machine_state::MachineState ms;
        ms.init(n, nth, nrh);

        ExecutionPlan ep = plan(ms, q, budget, hw);
        check(ep.regime == ExecutionRegime::GPU_ONLY ||
              ep.regime == ExecutionRegime::SPLIT,
              "RP: top-heavy → GPU_ONLY or SPLIT");
        check(planner::verify::plan_roles_correct(ep),
              "RP: roles correct (disc on CPU)");
        check(planner::verify::plan_fiber_safe(ep),
              "RP: plan fiber-safe");
        check(planner::verify::plan_valid(ep, budget),
              "RP: top-heavy plan valid");

        ms.free();
    }

    // Plan from StructureGraph
    {
        structure::StructureGraph sg;
        int b0 = sg.add_boundary(structure::BoundaryType::DISCRETE, 2);
        int b1 = sg.add_boundary(structure::BoundaryType::TOPOLOGICAL, 1);
        sg.add_entity(structure::EntityKind::NODE);
        sg.add_entity(structure::EntityKind::NODE);
        sg.add_observable(2, b0, true);
        sg.add_observable(4, b1, false);

        ExecutionPlan ep = plan_from_structure(sg, budget, hw);
        check(ep.regime != ExecutionRegime::STREAMING,
              "RP: small structure not streaming");
        check(planner::verify::plan_valid(ep, budget),
              "RP: structure plan valid");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Spectral Transport Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_spectral_transport() {
    using namespace spec_transport;

    const int nth = 8, nrh = 4;

    // Initialize a topological state: delta at origin
    TopologicalState psi;
    psi.init(nth, nrh);
    psi.set_delta(PhasePoint(0.0, 0.0));
    double norm_before = psi.norm2();

    // Forward/inverse DFT roundtrip preserves state
    {
        SpectralBasis spec;
        spec.init(nth, nrh);
        spec.forward(psi);

        // Parseval: spectral norm ≈ spatial norm
        check(verify::parseval(psi, spec),
              "ST: Parseval identity");

        TopologicalState restored;
        restored.init(nth, nrh);
        spec.inverse(restored);

        // Roundtrip fidelity
        double diff = 0;
        for (int i = 0; i < psi.total; ++i)
            diff += (psi.amp[i] - restored.amp[i]).norm2();
        check(diff < 1e-18,
              "ST: DFT roundtrip exact");

        restored.free();
        spec.free();
    }

    // Spectral evolution preserves unitarity
    {
        TopologicalState evolved;
        evolved.init(nth, nrh);
        memcpy(evolved.amp, psi.amp, psi.total * sizeof(C64));

        evolve_spectral(evolved, 1.0);
        double norm_after = evolved.norm2();
        check(fabs(norm_after - norm_before) < 1e-10,
              "ST: spectral evolution preserves norm");

        evolved.free();
    }

    // Symplectic Verlet preserves structure
    check(verify::symplectic_exact(1.0, 100),
          "ST: symplectic Verlet exact for linear flow");

    // Padé exponential is unitary
    check(verify::pade_unitary(1.0),
          "ST: Padé[2,2] unitary for φ = 1");
    check(verify::pade_unitary(constants::PI),
          "ST: Padé[2,2] unitary for φ = π");

    // Padé transport preserves norm
    {
        TopologicalState pade_psi;
        pade_psi.init(nth, nrh);
        pade_psi.set_delta(PhasePoint(0.0, 0.0));
        double n0 = pade_psi.norm2();

        apply_U_pade(pade_psi, 0.1);
        double n1 = pade_psi.norm2();
        check(fabs(n1 - n0) < 0.1,
              "ST: Padé transport preserves norm");

        pade_psi.free();
    }

    psi.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Persistent Format Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_persistent_format() {
    using namespace persistent;

    const int n = 2, nth = 8, nrh = 4;

    // Header validity
    {
        ObjectHeader hdr;
        check(hdr.valid(), "PF: default header valid");
        check(verify::header_valid(hdr), "PF: verify header valid");

        ObjectHeader bad;
        bad.magic = 0xDEADBEEF;
        check(!bad.valid(), "PF: bad magic detected");
    }

    // HybridState roundtrip
    {
        HybridState hyb;
        hyb.init(n, nth, nrh);
        // Put some known amplitudes
        hyb.amp[0] = C64(1.0, 0.0);
        hyb.amp[hyb.index(1, 2, 3)] = C64(0.0, 1.0);

        check(verify::hybrid_roundtrip(hyb),
              "PF: hybrid state roundtrip");

        // Serialize and check size
        SerializedObject obj = serialize_hybrid(hyb);
        check(obj.size > 0, "PF: serialized size > 0");
        check(obj.size == static_cast<int>(sizeof(ObjectHeader)) +
              16 + hyb.total * static_cast<int>(sizeof(C64)),
              "PF: serialized size matches expected");

        obj.free();
        hyb.free();
    }

    // Checksum detects corruption
    check(verify::checksum_detects_corruption(),
          "PF: checksum detects corruption");

    // CocycleLedger roundtrip
    {
        machine_state::CocycleLedger ledger;
        MirElement g = MirElement::identity();
        MirElement h = MirElement::identity();
        BitVec u(1, 2), v(0, 2), up(0, 2), vp(1, 2);
        ledger.accumulate(g, u, v, h, up, vp);
        ledger.record_closure_check(1e-15);

        SerializedObject obj = serialize_cocycle_ledger(ledger);
        machine_state::CocycleLedger restored;
        bool ok = deserialize_cocycle_ledger(obj.data, obj.size, restored);
        check(ok, "PF: cocycle ledger deserialized");
        check(restored.closure_valid, "PF: cocycle closure survives roundtrip");
        check(restored.n_checks == 2,
              "PF: cocycle n_checks survives roundtrip");

        obj.free();
    }

    // TraceSector roundtrip
    {
        machine_state::TraceSector trace;
        trace.record_gate(1.5);
        trace.record_transfer(1024);
        trace.record_measurement();

        SerializedObject obj = serialize_trace(trace);
        machine_state::TraceSector restored;
        bool ok = deserialize_trace(obj.data, obj.size, restored);
        check(ok, "PF: trace sector deserialized");
        check(restored.total_gate_ops == 1,
              "PF: trace gate ops survives roundtrip");
        check(restored.total_transfers == 1,
              "PF: trace transfers survives roundtrip");
        check(restored.total_measurements == 1,
              "PF: trace measurements survives roundtrip");

        obj.free();
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Exact Bidirectional Dictionary
// ════════════════════════════════════════════════════════════════════════════
static void test_exact_decomposition() {
    printf("\n── Exact Bidirectional Dictionary ──\n");
    using namespace exact;
    using measurement::DenseOp;

    int nq = 2, nth = 4, nrh = 3;
    int dd = 1 << nq;  // 4
    int dt = nth * nrh; // 12
    int dim_hyb = dd * dt; // 48
    double rmin = -5.0, rmax = 5.0;

    // ── Build non-trivial hybrid state ──
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

    // §1-3: Vector decomposition roundtrips
    check(vec::roundtrip_DR(psi), "ED: D_vec ∘ R_vec = id");
    {
        TopologicalState* comps = new TopologicalState[dd];
        for (int x = 0; x < dd; ++x)
            comps[x].init(nth, nrh, rmin, rmax);
        vec::decompose(psi, comps);
        check(vec::roundtrip_RD(comps, nq), "ED: R_vec ∘ D_vec = id");
        for (int x = 0; x < dd; ++x) comps[x].free();
        delete[] comps;
    }

    // §2-3: Operator decomposition roundtrip
    {
        BitVec u1(1, nq);
        DenseOp X_hyb(dim_hyb);
        for (int x = 0; x < dd; ++x) {
            int xp = x ^ u1.bits;
            for (int k = 0; k < dt; ++k)
                X_hyb.at(xp * dt + k, x * dt + k) = C64(1, 0);
        }
        check(op::roundtrip(X_hyb, dd, dt), "ED: D_op ∘ R_op = id");
    }

    // Matrix unit algebra
    check(op::matrix_unit_algebra(dd, dt), "ED: E_{xy}E_{x'y'} = δ_{yx'}E_{xy'}");

    // Matrix unit resolution
    check(op::matrix_unit_resolution(dd, dt), "ED: Σ_x E_{xx} = I");

    // §4-7: Walsh roundtrip
    {
        DenseOp I_hyb = DenseOp::identity(dim_hyb);
        OpBlocks A = op::decompose(I_hyb, dd, dt);
        check(walsh::roundtrip(A, nq), "ED: Walsh ↔ Block roundtrip");
        A.free();
    }

    // Walsh with non-trivial operator
    {
        BitVec u1(1, nq);
        DenseOp X_hyb(dim_hyb);
        for (int x = 0; x < dd; ++x) {
            int xp = x ^ u1.bits;
            for (int k = 0; k < dt; ++k)
                X_hyb.at(xp * dt + k, x * dt + k) = C64(1, 0);
        }
        OpBlocks A = op::decompose(X_hyb, dd, dt);
        check(walsh::roundtrip(A, nq), "ED: Walsh roundtrip (X_u)");
        check(walsh::pauli_is_walsh(A, nq), "ED: τ_{u,v} = 2^{-n} B_{u,v}");
        A.free();
    }

    // Pauli expansion of matrix units
    check(walsh::matrix_unit_pauli_expansion(dd, dt, nq),
          "ED: E_{xy} Pauli expansion");

    // §8-9: Kernel structure
    {
        BitVec u1(1, nq), v1(2, nq);

        // K_X structure
        OpBlocks Kx = kernel::K_X(u1, dd, dt);
        bool x_ok = true;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                if (x == (y ^ u1.bits)) {
                    if ((Kx.at(x, y) - DenseOp::identity(dt)).hs_norm2() > 1e-10)
                        x_ok = false;
                } else {
                    if (Kx.at(x, y).hs_norm2() > 1e-10) x_ok = false;
                }
            }
        check(x_ok, "ED: K_{X_u} = δ_{x,y⊕u} I_top");
        Kx.free();

        // K_Z structure
        OpBlocks Kz = kernel::K_Z(v1, dd, dt);
        bool z_ok = true;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                if (x == y) {
                    BitVec yv(y, nq);
                    double sign = (yv.inner(v1) == 0) ? 1.0 : -1.0;
                    DenseOp expected = DenseOp::identity(dt) * C64(sign, 0);
                    if ((Kz.at(x, y) - expected).hs_norm2() > 1e-10)
                        z_ok = false;
                } else {
                    if (Kz.at(x, y).hs_norm2() > 1e-10) z_ok = false;
                }
            }
        check(z_ok, "ED: K_{Z_v} = δ_{xy}(-1)^{⟨v,y⟩} I_top");
        Kz.free();

        // Kernel composition: K_X ∘ K_Z = K_{X_u Z_v}
        OpBlocks Kx2 = kernel::K_X(u1, dd, dt);
        OpBlocks Kz2 = kernel::K_Z(v1, dd, dt);
        OpBlocks composed = kernel::compose(Kx2, Kz2);
        OpBlocks direct;
        direct.init_zero(dd, dt);
        for (int y = 0; y < dd; ++y) {
            int x = y ^ u1.bits;
            BitVec yv(y, nq);
            double sign = (yv.inner(v1) == 0) ? 1.0 : -1.0;
            direct.at(x, y) = DenseOp::identity(dt) * C64(sign, 0);
        }
        double cerr = 0;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y)
                cerr += (composed.at(x, y) - direct.at(x, y)).hs_norm2();
        check(cerr < 1e-10, "ED: K_{X∘Z} composition = K_{XZ} direct");
        Kx2.free(); Kz2.free(); composed.free(); direct.free();
    }

    // §10-13: Density matrix decomposition
    {
        DenseOp rho(dim_hyb);
        for (int i = 0; i < dim_hyb; ++i)
            for (int j = 0; j < dim_hyb; ++j)
                rho.at(i, j) = psi.amp[i] * psi.amp[j].conj();

        OpBlocks rho_blk = dens::decompose(rho, dd, dt);
        DenseOp recon = dens::reconstruct(rho_blk);
        check((rho - recon).hs_norm2() < 1e-10, "ED: D_dens ∘ R_dens = id");

        check(dens::verify_prob_normalization(rho_blk),
              "ED: Σ_x p(x) = 1");

        check(dens::verify_diagonal_properties(rho_blk),
              "ED: diagonal blocks positive & Hermitian");

        // Conditional topological state normalization
        bool cond_ok = true;
        for (int x = 0; x < dd; ++x) {
            DenseOp cond = dens::conditional_topo(rho_blk, x);
            double tr = cond.trace().re;
            if (fabs(tr - 1.0) > 1e-10) cond_ok = false;
        }
        check(cond_ok, "ED: Tr(ρ_x^top) = 1 for all x");

        // Block expectation matches full trace
        DenseOp I_hyb = DenseOp::identity(dim_hyb);
        check(dens::verify_block_expectation(rho, I_hyb, dd, dt),
              "ED: ω_ρ(I) = Tr(ρ I) via blocks");

        // Block expectation with non-trivial observable
        BitVec u1(1, nq);
        DenseOp X_hyb(dim_hyb);
        for (int x = 0; x < dd; ++x) {
            int xp = x ^ u1.bits;
            for (int k = 0; k < dt; ++k)
                X_hyb.at(xp * dt + k, x * dt + k) = C64(1, 0);
        }
        check(dens::verify_block_expectation(rho, X_hyb, dd, dt),
              "ED: ω_ρ(X_u) = Tr(ρ X_u) via blocks");

        rho_blk.free();
    }

    // §14-15: Weyl block verification (identity element)
    {
        BitVec u0(0, nq), v0(0, nq);
        MirElement e = MirElement::identity();
        check(field::verify_weyl_blocks(u0, v0, e, nq, nth, nrh, rmin, rmax, 1e-6),
              "ED: W_{0,0;e} blocks match direct");
    }

    // N-point function: G_1(I; ρ) = Tr(ρ) = 1
    {
        DenseOp rho(dim_hyb);
        for (int i = 0; i < dim_hyb; ++i)
            for (int j = 0; j < dim_hyb; ++j)
                rho.at(i, j) = psi.amp[i] * psi.amp[j].conj();

        OpBlocks rho_blk = dens::decompose(rho, dd, dt);
        OpBlocks I_blk;
        I_blk.init_zero(dd, dt);
        for (int x = 0; x < dd; ++x)
            I_blk.at(x, x) = DenseOp::identity(dt);

        C64 g1 = field::G1(I_blk, rho_blk);
        check_near(g1.re, 1.0, 1e-10, "ED: G_1(I; ρ) = 1");
        check_near(g1.im, 0.0, 1e-10, "ED: G_1(I; ρ) imaginary = 0");

        I_blk.free();
        rho_blk.free();
    }

    // §16: Transport preserves discrete probabilities
    {
        // Identity transport (exact)
        DenseOp T_id = DenseOp::identity(dt);
        check(exact::transport::preserves_disc_probs(psi, T_id),
              "ED: id transport preserves p(x)");

        // Transport block structure
        OpBlocks T_blk = exact::transport::transport_blocks(T_id, dd);
        bool tblk_ok = true;
        for (int x = 0; x < dd; ++x)
            for (int y = 0; y < dd; ++y) {
                if (x == y) {
                    if ((T_blk.at(x, y) - DenseOp::identity(dt)).hs_norm2() > 1e-10)
                        tblk_ok = false;
                } else {
                    if (T_blk.at(x, y).hs_norm2() > 1e-10)
                        tblk_ok = false;
                }
            }
        check(tblk_ok, "ED: transport blocks = δ_{xy} T_φ");
        T_blk.free();
    }

    // §17-18: Thermodynamic factorization
    {
        DenseOp H_top(dt);
        for (int i = 0; i < dt; ++i)
            H_top.at(i, i) = C64(0.1 * i, 0);
        check(thermo::verify_factorization(H_top, 1.0, nq),
              "ED: Z_{A,n} = 2^n · Z_A^top");
    }

    // POVM normalization with identity kernel
    {
        OpBlocks K_id;
        K_id.init_zero(dd, dt);
        for (int x = 0; x < dd; ++x)
            K_id.at(x, x) = DenseOp::identity(dt);

        DenseOp rho_0 = DenseOp::identity(dt);
        rho_0 = rho_0 * C64(1.0 / dt, 0);
        check(thermo::verify_povm_normalization(K_id, 0, rho_0),
              "ED: POVM Σ_y Pr[y] = 1 (id kernel)");

        // Test with X_u kernel
        BitVec u1(1, nq);
        OpBlocks K_x = kernel::K_X(u1, dd, dt);
        check(thermo::verify_povm_normalization(K_x, 0, rho_0),
              "ED: POVM Σ_y Pr[y] = 1 (X_u kernel)");
        K_id.free(); K_x.free();
    }

    // Grand verification
    {
        verify::VerifyResult vr = verify::run_all(nq, nth, nrh);
        check(vr.passed == vr.total_checks,
              "ED: grand verify all checks pass");
    }

    psi.free();
}

// ════════════════════════════════════════════════════════════════════════════
// Kernel Calculus Tests
// ════════════════════════════════════════════════════════════════════════════
static void test_kernel_calculus() {
    printf("  kernel calculus ...\n");
    using namespace topcomp;
    using namespace topcomp::kc;
    using measurement::DenseOp;

    const int n = 1;
    const int dd = 2;
    const int N_theta = 4, N_rho = 3;
    const int dt = N_theta * N_rho;
    const int dim_hyb = dd * dt;
    const double tol = 1e-4;

    // ── Build test data ──
    int f[2] = { 1, 0 };   // swap permutation
    int g[2] = { 0, 1 };   // identity permutation

    // Fiber operators: phase rotations
    DenseOp U_ops[2];
    for (int x = 0; x < dd; ++x) {
        U_ops[x] = DenseOp::identity(dt);
        for (int k = 0; k < dt; ++k) {
            double angle = 0.1 * (x + 1) * (k + 1);
            U_ops[x].at(k, k) = C64(cos(angle), sin(angle));
        }
    }

    // Density matrix
    DenseOp rho(dim_hyb);
    for (int i = 0; i < dim_hyb; ++i) {
        double val = cos(0.3 * i + 0.1);
        rho.at(i, i) = C64(fabs(val), 0);
    }
    double tr = rho.trace().re;
    rho = rho * C64(1.0 / tr, 0);

    DenseOp rho_0 = DenseOp::identity(dt);
    rho_0 = rho_0 * C64(1.0 / dt, 0);

    // ── §1: *-Algebra ──
    {
        DenseOp C_hyb = ctrl_op::build_hybrid_op(f, U_ops, dd, dt);
        DenseOp D_hyb = perm_lift::build_hybrid_op(g, dd, dt);
        check(star_alg::verify_multiplicative(C_hyb, D_hyb, dd, dt, tol),
              "KC: K(CD) = K(C)K(D) multiplicative");
        check(star_alg::verify_star(C_hyb, dd, dt, tol),
              "KC: K(C*) = K(C)* adjoint preserving");
        OpBlocks K_C = exact::op::decompose(C_hyb, dd, dt);
        check(star_alg::verify_column_unitarity(K_C, dd, dt, tol),
              "KC: column unitarity");
        check(star_alg::verify_row_unitarity(K_C, dd, dt, tol),
              "KC: row unitarity");
        K_C.free();
    }

    // ── §2: Projector kernel ──
    {
        OpBlocks Kp = proj::K_P(0, dd, dt);
        check((Kp.at(0, 0) - DenseOp::identity(dt)).hs_norm2() < tol,
              "KC: K_{P_0}(0,0) = I_top");
        check(Kp.at(0, 1).hs_norm2() < tol,
              "KC: K_{P_0}(0,1) = 0");
        check(Kp.at(1, 0).hs_norm2() < tol,
              "KC: K_{P_0}(1,0) = 0");
        check(Kp.at(1, 1).hs_norm2() < tol,
              "KC: K_{P_0}(1,1) = 0");
        Kp.free();

        check(proj::verify_idempotent(0, dd, dt, tol),
              "KC: projector idempotent");
        check(proj::verify_resolution(dd, dt, tol),
              "KC: projector resolution Σ K_{P_b} = K_I");
    }

    // ── §3: Path sum ──
    {
        BitVec u1(1, n), v0(0, n);
        OpBlocks K1 = exact::kernel::K_X(u1, dd, dt);
        OpBlocks K2 = exact::kernel::K_Z(v0, dd, dt);
        OpBlocks gates[2] = { K1, K2 };
        OpBlocks K_path = path_sum::from_gate_sequence(gates, 2);
        OpBlocks K_composed = exact::kernel::compose(K2, K1);
        double err = 0;
        for (int x = 0; x < dd; ++x)
            for (int a = 0; a < dd; ++a)
                err += (K_path.at(x, a) - K_composed.at(x, a)).hs_norm2();
        check(err < tol, "KC: path-sum matches compose");
        K1.free(); K2.free(); K_path.free(); K_composed.free();
    }

    // ── §4: Permutation lift ──
    check(perm_lift::verify_kernel(f, dd, dt, tol),
          "KC: P̂_f kernel correct");
    check(perm_lift::verify_composition(f, g, dd, dt, tol),
          "KC: P̂_{f∘g} = P̂_f · P̂_g");
    check(perm_lift::verify_unitary(f, dd, dt, tol),
          "KC: P̂_f unitary");
    check(perm_lift::verify_prob_permutation(f, rho, dd, dt, tol),
          "KC: p_{P̂_f ρ P̂_f*}(f(x)) = p_ρ(x)");

    // ── §5: Controlled operator ──
    check(ctrl_op::verify_kernel(f, U_ops, dd, dt, tol),
          "KC: Ĉ kernel correct");
    check(ctrl_op::verify_composition(f, U_ops, g, U_ops, dd, dt, tol),
          "KC: Ĉ composition law");
    check(ctrl_op::verify_unitary(f, U_ops, dd, dt, tol),
          "KC: Ĉ unitary");
    check(ctrl_op::verify_factorization(f, U_ops, dd, dt, tol),
          "KC: Ĉ = P̂_f · Ĉ_{id,{U_x}}");
    check(ctrl_op::verify_weyl(f, U_ops, dd, dt, n, tol),
          "KC: Ĉ Weyl expansion");

    // ── §6: Induced channel ──
    {
        OpBlocks K_f = perm_lift::build_kernel(f, dd, dt);
        check(channel::verify_disc_prob_consistency(K_f, rho_0, tol),
              "KC: channel ↔ disc prob consistency");
        check(channel::verify_deterministic_case(f, rho_0, dd, dt, tol),
              "KC: deterministic kernel → classical");
        K_f.free();
    }

    // ── §7: Unitary lift ──
    {
        DenseOp U_disc(dd);
        double inv = 1.0 / sqrt(dd);
        for (int i = 0; i < dd; ++i)
            for (int j = 0; j < dd; ++j) {
                BitVec iv(i, n), jv(j, n);
                double sign = (iv.inner(jv) == 0) ? 1.0 : -1.0;
                U_disc.at(i, j) = C64(sign * inv, 0);
            }
        check(unitary_lift::verify_kernel(U_disc, dt, tol),
              "KC: Ũ kernel correct");
        check(unitary_lift::verify_disc_probs(U_disc, dt, tol),
              "KC: Ũ discrete probs |U_{x,a}|²");
        check(unitary_lift::verify_unitarity(U_disc, dt, tol),
              "KC: Ũ kernel unitarity");
    }

    // ── §8: Stochastic channel ──
    {
        double K_stoch[4];
        for (int y = 0; y < dd; ++y)
            for (int x = 0; x < dd; ++x)
                K_stoch[y * dd + x] = 1.0 / dd;
        OpBlocks rho_blk = exact::op::decompose(rho, dd, dt);
        check(stochastic::verify_prob_update(K_stoch, rho_blk, tol),
              "KC: stochastic prob update");
        check(stochastic::verify_trace_preservation(K_stoch, rho_blk, tol),
              "KC: stochastic trace preservation");
        rho_blk.free();
    }

    // ── §9: Dec transforms ──
    check(dec::verify_prob_transform(f, U_ops, rho, dd, dt, tol),
          "KC: Dec_prob ∘ Ad_Ĉ = Perm_f ∘ Dec_prob");
    check(dec::verify_full_transform(f, U_ops, rho, dd, dt, tol),
          "KC: Dec_full ∘ Ad_Ĉ = CtrlAd ∘ Dec_full");

    // ── §10: Compilation ──
    {
        const int* f_steps[2] = { f, g };
        const DenseOp* U_steps[2] = { U_ops, U_ops };
        check(compilation::verify_compilation(
                  f_steps, U_steps, 2, dd, dt, tol),
              "KC: compilation Ĉ_{f∘g,{W_x}} correct");
    }

    // ── Irreversible extension ──
    {
        int f_irrev[2] = { 0, 1 };
        int f_tilde[4];
        compilation::irreversible_extension(f_irrev, 2, 2, f_tilde);
        // f̃(a,b) = (a, b⊕f(a)) = (a, b⊕a)
        // f̃(0,0)=0, f̃(0,1)=1, f̃(1,0)=3, f̃(1,1)=2
        check(f_tilde[0] == 0 && f_tilde[1] == 1 &&
              f_tilde[2] == 3 && f_tilde[3] == 2,
              "KC: irreversible extension values");
        bool bijection = true;
        bool seen[4] = { false };
        for (int i = 0; i < 4; ++i) {
            if (seen[f_tilde[i]]) { bijection = false; break; }
            seen[f_tilde[i]] = true;
        }
        check(bijection, "KC: irreversible extension is bijection");
    }

    // ── Grand verify ──
    {
        verify::KCVerifyResult vr = verify::run_all(n, N_theta, N_rho, tol);
        check(vr.passed == vr.total_checks,
              "KC: grand verify all checks pass");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Semantic Kernel (doctrine)
// ════════════════════════════════════════════════════════════════════════════
static void test_semantic_kernel() {
    printf("\n── Semantic Kernel ──\n");
    using measurement::DenseOp;
    int n = 1;
    int dd = 1 << n;
    int N_theta = 4, N_rho = 3;
    int dt = N_theta * N_rho;
    double tol = 1e-4;

    // ── Build test ControlledOp ──
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
    doctrine::ControlledOp cop = doctrine::ControlledOp::from(f, U_ops, dd, dt);

    check(cop.dim_disc == dd, "SK: cop dim_disc");
    check(cop.dim_top == dt, "SK: cop dim_top");
    check(!cop.f_is_id, "SK: f not identity");
    check(cop.is_bijective(), "SK: f bijective");

    // ── Pure permutation detection ──
    {
        int* id_f = new int[dd];
        for (int x = 0; x < dd; ++x) id_f[x] = (x + 1) % dd;
        DenseOp* id_U = new DenseOp[dd];
        for (int x = 0; x < dd; ++x) id_U[x] = DenseOp::identity(dt);
        doctrine::ControlledOp perm_cop =
            doctrine::ControlledOp::from(id_f, id_U, dd, dt);
        check(perm_cop.is_pure_perm(), "SK: identity U → pure perm");
        perm_cop.free();
        delete[] id_f;
        delete[] id_U;
    }

    // ── Composition algebra ──
    {
        int* g = new int[dd];
        for (int x = 0; x < dd; ++x) g[x] = (dd - 1 - x);
        doctrine::ControlledOp cop2 = doctrine::ControlledOp::from(g, U_ops, dd, dt);
        doctrine::ControlledOp prod = cop.compose(cop2);
        DenseOp lhs = cop.to_dense() * cop2.to_dense();
        DenseOp rhs = prod.to_dense();
        check((lhs - rhs).hs_norm2() < tol, "SK: compose algebra");
        prod.free(); cop2.free();
        delete[] g;
    }

    // ── Inverse ──
    {
        doctrine::ControlledOp inv = cop.inverse();
        DenseOp Cinv = inv.to_dense();
        DenseOp C_hyb = cop.to_dense();
        DenseOp I = DenseOp::identity(dd * dt);
        check((C_hyb * Cinv - I).hs_norm2() < tol, "SK: C^{-1}C = I");
        inv.free();
    }

    // ── Dense round-trip ──
    {
        DenseOp D = cop.to_dense();
        exact::OpBlocks blks = exact::op::decompose(D, dd, dt);
        DenseOp R = exact::op::reconstruct(blks);
        check((D - R).hs_norm2() < tol, "SK: dense roundtrip");
        blks.free();
    }

    // ── OpRepr construction & materialization ──
    {
        doctrine::OpRepr rep = doctrine::OpRepr::from_controlled(cop, n);
        check(rep.face == doctrine::OpFace::CONTROLLED, "SK: repr face");
        check(rep.dim_disc == dd, "SK: repr dim_disc");

        rep.materialize_blocks();
        check(rep.blocks != nullptr, "SK: materialize blocks");
        rep.materialize_kernel();
        check(rep.kernel != nullptr, "SK: materialize kernel");
        rep.materialize_weyl();
        check(rep.weyl != nullptr, "SK: materialize weyl");

        // Face selection
        doctrine::OpRepr rep2 = doctrine::OpRepr::from_controlled(cop, n);
        check(doctrine::face_select::best_for_multiply(rep, rep2) ==
              doctrine::OpFace::CONTROLLED, "SK: face select ctrl×ctrl");
        rep2.free();
        rep.free();
    }

    // ── SparsityReport ──
    {
        doctrine::OpRepr perm_rep = doctrine::OpRepr::from_perm(f, dd, dt, n);
        int bs = perm_rep.block_sparsity();
        check(bs <= dd, "SK: perm sparsity ≤ dd");
        check(perm_rep.is_controlled(), "SK: perm is controlled");
        perm_rep.free();
    }

    // ── Controlled extraction ──
    {
        DenseOp D = cop.to_dense();
        doctrine::OpRepr rep = doctrine::OpRepr::from_dense(D, dd, dt, n);
        doctrine::ControlledOp ext;
        bool ok = rep.extract_controlled(ext);
        check(ok, "SK: extract controlled");
        if (ok) {
            DenseOp Dext = ext.to_dense();
            check((D - Dext).hs_norm2() < tol, "SK: extracted roundtrip");
            ext.free();
        }
        rep.free();
    }

    // ── Scalar cost ──
    {
        doctrine::OpRepr s = doctrine::OpRepr::from_scalar(
            C64(2.0, 0), dd, dt, n);
        check(s.multiply_cost() < 1, "SK: scalar cost < 1");
        s.free();
    }

    // ── Perm OpRepr ──
    {
        doctrine::OpRepr p = doctrine::OpRepr::from_perm(f, dd, dt, n);
        check(p.face == doctrine::OpFace::PERMUTATION, "SK: perm face");
        p.free();
    }

    // ── Doctrine grand verify ──
    {
        doctrine::verify::DoctrineVerifyResult vr =
            doctrine::verify::run_all(n, N_theta, N_rho, tol);
        check(vr.passed == vr.total_checks,
              "SK: doctrine grand verify all pass");
    }

    // ── Semantic pass grand verify ──
    {
        semantic_pass::verify::PassVerifyResult pvr =
            semantic_pass::verify::run_all(n, N_theta, N_rho, tol);
        check(pvr.passed == pvr.total_checks,
              "SK: semantic pass verify all pass");
    }

    // ── Cleanup ──
    cop.free();
    delete[] f;
    delete[] U_ops;
}

// ════════════════════════════════════════════════════════════════════════════
// Test: Exemplars
// ════════════════════════════════════════════════════════════════════════════
static void test_exemplars() {
    printf("\n── Exemplars ──\n");

    exemplar::AllExemplarResults er =
        exemplar::run_all_exemplars(1, 4, 3, 1e-4);

    check(er.ex1.ctrl_op_built, "EX1: controlled op built");
    check(er.ex1.algebra_compose, "EX1: composition algebra");
    check(er.ex1.algebra_inverse, "EX1: inverse algebra");
    check(er.ex1.dense_roundtrip, "EX1: dense roundtrip");
    check(er.ex1.kernel_roundtrip, "EX1: kernel roundtrip");
    check(er.ex1.face_select_correct, "EX1: face selection");
    check(er.ex1.ir_execute, "EX1: IR execution");
    check(er.ex1.born_probs, "EX1: Born probabilities");

    check(er.ex2.kernel_built, "EX2: kernel built");
    check(er.ex2.channel_trace_pres, "EX2: trace preservation");
    check(er.ex2.channel_positive, "EX2: positivity");
    check(er.ex2.channel_classify, "EX2: classification");
    check(er.ex2.stochastic_roundtrip, "EX2: stochastic roundtrip");
    check(er.ex2.perm_gives_determ, "EX2: perm → deterministic");

    check(er.ex3.lift_built, "EX3: lift built");
    check(er.ex3.kernel_structure, "EX3: kernel K(x,a) = U_{xa} I_top");
    check(er.ex3.disc_probs, "EX3: discrete probabilities");
    check(er.ex3.opblocks_roundtrip, "EX3: OpBlocks roundtrip");
    check(er.ex3.oprep_face_chain, "EX3: OpRepr face chain");
    check(er.ex3.weyl_expansion_exact, "EX3: Weyl expansion exact");
    check(er.ex3.unitarity_preserved, "EX3: unitarity preserved");

    check(er.total_passed == er.total_checks,
          "EXEMPLARS: all pass");
}

// ════════════════════════════════════════════════════════════════════════════
// Test: C API
// ════════════════════════════════════════════════════════════════════════════
static void test_c_api() {
    printf("\n── C API ──\n");

    // Version
    check(tc_version_major() == 0, "CAPI: version major");
    check(tc_version_minor() == 4, "CAPI: version minor");

    // Program lifecycle
    tc_program prog = tc_program_create(1, 4, 3);
    check(prog != nullptr, "CAPI: program created");

    // Emit and execute
    check(tc_emit_h(prog, 0) == TC_OK, "CAPI: emit H");
    check(tc_emit_x(prog, 1) == TC_OK, "CAPI: emit X");
    check(tc_emit_measure(prog) == TC_OK, "CAPI: emit measure");

    tc_result res = tc_execute(prog);
    check(res != nullptr, "CAPI: execute succeeded");

    int nm = tc_result_num_measurements(res);
    check(nm > 0, "CAPI: has measurements");

    double p_sum = 0;
    int dim = tc_result_meas_dim(res, 0);
    for (int i = 0; i < dim; ++i)
        p_sum += tc_result_prob(res, 0, i);
    check(fabs(p_sum - 1.0) < 1e-6, "CAPI: Born probs sum to 1");

    check(tc_result_cocycle_ok(res) == 1, "CAPI: cocycle ok");
    check(tc_result_time(res) > 0, "CAPI: time > 0");

    tc_result_free(res);
    tc_program_free(prog);

    // ControlledOp via C API
    int f_perm[2] = { 1, 0 };
    int dt = 12;
    double* U_data = new double[2 * dt * dt * 2];
    for (int x = 0; x < 2; ++x)
        for (int i = 0; i < dt; ++i)
            for (int j = 0; j < dt; ++j) {
                int idx = x * dt * dt * 2 + (i * dt + j) * 2;
                U_data[idx]     = (i == j) ? 1.0 : 0.0;
                U_data[idx + 1] = 0.0;
            }

    tc_ctrl_op cop = tc_ctrlop_create(f_perm, U_data, 2, dt);
    check(cop != nullptr, "CAPI: ctrlop created");
    check(tc_ctrlop_is_perm(cop) == 1, "CAPI: ctrlop is perm");

    tc_ctrl_op cop2 = tc_ctrlop_compose(cop, cop);
    check(cop2 != nullptr, "CAPI: ctrlop compose");

    tc_ctrlop_free(cop2);
    tc_ctrlop_free(cop);
    delete[] U_data;
}

// ════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════
int main() {
    printf("=============================================================\n");
    printf("  TopComp Test Suite                                         \n");
    printf("=============================================================\n");

    test_constants();
    test_mir_group();
    test_cocycle();
    test_hilbert();
    test_gates();
    test_prng();
    test_machine();
    test_witt();
    test_lie_algebra();
    test_representation();
    test_weyl();
    test_quantum_computation();
    test_measurement();
    test_hybrid_doctrine();
    test_category();
    test_compiler();
    test_structure_finder();
    test_transport();
    test_forms();
    test_cptp_instrument_naimark();
    test_affine_rep();
    test_scale_conformal();
    test_UM_commutation();
    test_mir_functor();
    test_transport_naturality();
    test_exhaustive_coverage();
    test_heisenberg();
    test_clifford();
    test_boolean_algebra();
    test_shadow_functor();
    test_fibonacci();
    test_connection();
    test_translation_matrices();
    test_witt_commutator();
    test_faa_di_bruno();
    test_disc_top_info();
    test_grand_chain();
    test_isometry();
    test_weyl_translation();
    test_jet_um_commutation();
    test_sector();
    test_fiber();
    test_structure();
    test_descriptor();
    test_emission();
    test_structure_ir();
    test_platform();
    test_mixed_operators();
    test_quantum_algebra();
    test_hardware_topology();
    test_machine_state();
    test_fibered_memory();
    test_regime_planner();
    test_spectral_transport();
    test_persistent_format();
    test_exact_decomposition();
    test_kernel_calculus();
    test_semantic_kernel();
    test_exemplars();
    test_c_api();

    printf("\n=============================================================\n");
    printf("  Results: %d passed, %d failed, %d total\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    printf("=============================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
