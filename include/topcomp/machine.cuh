// ============================================================================
// TopComp: HES Machine Model
// ============================================================================
// Implements the Hybrid-Ergodic-Symplectic (HES) computation machine:
//
//   M_{A,n}^HES = (X, G, P, C, δ, Init, Halt, time, space, cost)
//
//   State space:  X_{A,n}^hyb
//   Gate set:     G_{A,n}^hyb = ⟨{X_u}, {Z_v}, {U_t}, {M_m}, {P_x}⟩
//   Programs:     P_{A,n}^hyb = (G_{A,n}^hyb)*
//   Config:       C_{A,n}^hyb = X × P × N × {0,1}
//   Transition:   δ(ψ, gP, τ, 0) = (gψ, P, τ+1, 0)
//                 δ(ψ, ε, τ, h)  = (ψ, ε,  τ,   1)
//
//   Cost:         cost(X_u) = ||u||_0, cost(Z_v) = ||v||_0,
//                 cost(U_t) = cost(M_m) = cost(P_x) = 1
//   Time:         time(g₁…g_L) = L
//   Space:        space(g₁…g_L) = n + max_j cost(g_j)
//
//   Complexity classes: P_Mir, BQP_Mir, Samp_Mir
//
//   Density operators: D(X) = {ρ ≥ 0 : Tr(ρ)=1}
//   POVM:              E_{A,n}(Y) = {(E_y) : E_y≥0, Σ E_y = I}
//   Born rule:         p_{ρ,E}(y) = Tr(ρ E_y)
// ============================================================================
#pragma once
#include "gates.cuh"
#include "cocycle.cuh"
#include "prng.cuh"
#include "structure_finder.cuh"

namespace topcomp {

// ── Program: sequence of gates ──────────────────────────────────────────────
struct Program {
    static constexpr int MAX_GATES = 4096;
    Gate gates[MAX_GATES];
    int  length;

    __host__ __device__ Program() : length(0) {}

    __host__ __device__ void append(const Gate& g) {
        if (length < MAX_GATES) gates[length++] = g;
    }

    // U_P := g_L ⋯ g_1  (reverse order application)
    __host__ void apply(HybridState& psi) const {
        for (int j = 0; j < length; ++j) {
            gates[j].apply(psi);
        }
    }

    // cost(g_1 ⋯ g_L) = Σ cost(g_j)
    __host__ __device__ int total_cost(int n_qubits) const {
        int c = 0;
        for (int j = 0; j < length; ++j) c += gates[j].cost(n_qubits);
        return c;
    }

    // time(g_1 ⋯ g_L) = L
    __host__ __device__ int time() const { return length; }

    // space(g_1 ⋯ g_L) = n + max_j cost(g_j)
    __host__ __device__ int space(int n_qubits) const {
        int mx = 0;
        for (int j = 0; j < length; ++j) {
            int c = gates[j].cost(n_qubits);
            if (c > mx) mx = c;
        }
        return n_qubits + mx;
    }
};

// ── Configuration: (ψ, P, τ, h) ─────────────────────────────────────────────
struct Configuration {
    HybridState psi;
    int pc;      // program counter (index into program)
    int tau;     // elapsed time steps
    int halt;    // halt flag

    Configuration() : pc(0), tau(0), halt(0) { psi.amp = nullptr; }
};

// ── HES Machine ─────────────────────────────────────────────────────────────
struct HESMachine {
    int n_qubits;
    int N_theta;
    int N_rho;

    HESMachine(int nq, int nth, int nrh)
        : n_qubits(nq), N_theta(nth), N_rho(nrh) {}

    // Init(ψ₀, P) = (ψ₀, P, 0, 0)
    __host__ Configuration init(const HybridState& psi0) const {
        Configuration config;
        config.psi.init(psi0.n_qubits, psi0.N_theta, psi0.N_rho,
                        psi0.rho_min, psi0.rho_max);
        memcpy(config.psi.amp, psi0.amp, psi0.total * sizeof(C64));
        config.pc = 0;
        config.tau = 0;
        config.halt = 0;
        return config;
    }

    // Halt(ψ, P, τ, h) ⟺ (pc ≥ len) ∨ (h = 1)
    __host__ bool halted(const Configuration& config, const Program& prog) const {
        return (config.pc >= prog.length) || (config.halt != 0);
    }

    // δ(ψ, gP, τ, 0) = (gψ, P, τ+1, 0)
    __host__ void step(Configuration& config, const Program& prog) const {
        if (halted(config, prog)) {
            config.halt = 1;
            return;
        }
        prog.gates[config.pc].apply(config.psi);
        config.pc++;
        config.tau++;
    }

    // Run the full program
    // δ(ψ, ε, τ, h) = (ψ, ε, τ, 1)  when program exhausted
    __host__ void run(Configuration& config, const Program& prog) const {
        while (!halted(config, prog)) {
            step(config, prog);
        }
        config.halt = 1;
    }

    // ── Measurement (Born rule) ─────────────────────────────────────────────

    // p_ρ^disc(x) = Tr(ρ P_x^hyb)
    // For pure state |ψ⟩: p(x) = Σ_grid |ψ(x,j)|²
    __host__ double measure_proj(const HybridState& state, int x) const {
        return state.prob_disc(x);
    }

    // Full Born probability distribution over discrete outcomes
    __host__ void born_distribution(const HybridState& state,
                                      double* probs) const {
        int dim = state.dim_disc;
        double total = 0.0;
        for (int x = 0; x < dim; ++x) {
            probs[x] = state.prob_disc(x);
            total += probs[x];
        }
        if (total > 0) {
            for (int x = 0; x < dim; ++x) probs[x] /= total;
        }
    }

    // ── Cost functionals from the paper ─────────────────────────────────────

    // time_{A,n}(P)
    __host__ int program_time(const Program& prog) const {
        return prog.time();
    }

    // space_{A,n}(P)
    __host__ int program_space(const Program& prog) const {
        return prog.space(n_qubits);
    }

    // cost_{A,n}(P)
    __host__ int program_cost(const Program& prog) const {
        return prog.total_cost(n_qubits);
    }
};

// ── Density operator operations ─────────────────────────────────────────────

namespace density {

// p_{ρ,E}(y) = Tr(ρ E_y)
// For diagonal projectors: p(y) = ρ[y,y].re
__host__ inline double born_prob(const DensityOp& rho, int y) {
    if (y < rho.total) return rho.mat[y * rho.total + y].re;
    return 0.0;
}

} // namespace density

// ── Complexity class verification ───────────────────────────────────────────

namespace complexity {

// Check if a language-deciding program family satisfies P_Mir constraints:
//   time(P_x) ≤ |x|^c, p(1) ∈ {0,1}
__host__ inline bool check_P_Mir(const Program& prog,
                                    const HybridState& initial,
                                    int input_size, int c,
                                    const HESMachine& machine) {
    int time_bound = 1;
    for (int i = 0; i < c; ++i) time_bound *= input_size;
    if (prog.time() > time_bound) return false;

    Configuration config = machine.init(initial);
    machine.run(config, prog);

    double p1 = machine.measure_proj(config.psi, 1);
    config.psi.free();
    // p(1) must be exactly 0 or 1 (within tolerance)
    return (p1 < 1e-10) || (p1 > 1.0 - 1e-10);
}

// Check BQP_Mir: time ≤ |x|^c, p(1) ≥ 2/3 or p(1) ≤ 1/3
__host__ inline bool check_BQP_Mir(const Program& prog,
                                      const HybridState& initial,
                                      int input_size, int c,
                                      const HESMachine& machine,
                                      bool in_language) {
    int time_bound = 1;
    for (int i = 0; i < c; ++i) time_bound *= input_size;
    if (prog.time() > time_bound) return false;

    Configuration config = machine.init(initial);
    machine.run(config, prog);

    double p1 = machine.measure_proj(config.psi, 1);
    config.psi.free();
    if (in_language) return p1 >= 2.0 / 3.0;
    else            return p1 <= 1.0 / 3.0;
}

} // namespace complexity

// ── Structure-in-noise detection ────────────────────────────────────────────

namespace signal {

// Detection pipeline: find topological structure hidden behind noise
//
// 1. Generate orbit via golden flow: (θ_t, ρ_t) = Φ_t(θ_0, ρ_0)
// 2. Evaluate cocycle along orbit: Ω_A(g_t, g_{t+1})
// 3. Apply shadow functor: Sh_2(Ω)
// 4. Extract bits: b_t = ℓ_2(q_2(Sh_2(Ω)))
// 5. Run statistical tests to detect non-random structure
// 6. If bias/discrepancy < ε: structure found
//
// The key insight: the cocycle Ω encodes topological invariants
// which persist even through noise, because UFE = 0 guarantees
// the symplectic structure is flat (no curvature = no information loss).

struct DetectionResult {
    stats::StatReport report;
    double structure_score;  // confidence = 1 - Fisher p-value
    bool   structure_found;
};

// Run the full detection pipeline
__host__ inline DetectionResult detect_structure(
    const MirElement& seed,
    int N,
    double noise_stddev = 0.0,
    double alpha = 0.01)
{
    MirPRNG prng(seed, BitVec(), 4);

    int* bits = new int[N];
    prng.generate(bits, N);

    // Add noise if requested (flip bits with probability proportional to noise)
    if (noise_stddev > 0.0) {
        unsigned int noise_state = 42;
        for (int t = 0; t < N; ++t) {
            noise_state = noise_state * 1103515245u + 12345u;
            double u = static_cast<double>(noise_state & 0x7FFFFFFF) / 2147483647.0;
            if (u < noise_stddev) {
                bits[t] ^= 1;
            }
        }
    }

    // Use the rigorous Fisher combined p-value test
    structure::StructureSignature sig = structure::find_structure(bits, N, nullptr, 0, alpha);

    DetectionResult result;
    result.report = stats::run_full_tests(bits, N, sig.bias);
    result.structure_score = sig.confidence;
    result.structure_found = sig.is_structured;

    delete[] bits;
    return result;
}

} // namespace signal

} // namespace topcomp
