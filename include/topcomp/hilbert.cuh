// ============================================================================
// TopComp: Hilbert Space Structures
// ============================================================================
// Three tiers of state spaces:
//
//   Π_n^disc: Discrete (qubit) layer
//     I_n = F_2^n,  V_n = C^{2^n},  dim V_n = 2^n
//     X_u, Z_v: Pauli-X shift and phase operators
//     P_x: computational basis projectors
//     Ω_bool(u,v) = πi·⟨u,v⟩ mod 2πi
//
//   Π_{ρ,A}^top: Topological layer
//     X_A^top: topological state space (functions on M/Γ)
//     U_t: golden-ratio flow operator
//     M_m: modular multiplier operator
//
//   Π_{ρ,A,n}^hyb: Hybrid layer
//     X_{A,n}^hyb = V_n ⊗ X_A^top
//     Gates: {X_u^hyb, Z_v^hyb, U_t^hyb, M_m^hyb, P_x^hyb}
//     Q_{A,n}: isomorphism ⊕_x X_A^top → X_{A,n}^hyb
// ============================================================================
#pragma once
#include "phase_space.cuh"
#include <cstring>

namespace topcomp {

// ── Maximum qubit count (compile-time bound) ────────────────────────────────
static constexpr int MAX_QUBITS = 16;
static constexpr int MAX_DIM = (1 << MAX_QUBITS);  // 2^16 = 65536

// ── F_2^n: binary vector (bit representation) ───────────────────────────────
struct BitVec {
    uint32_t bits;  // up to 32 qubits
    int      n;     // number of bits

    __host__ __device__ constexpr BitVec() : bits(0), n(0) {}
    __host__ __device__ constexpr BitVec(uint32_t b, int nn) : bits(b & ((1u << nn) - 1)), n(nn) {}

    // Inner product ⟨u,v⟩ = u·v mod 2
    __host__ __device__ int inner(const BitVec& v) const {
        uint32_t x = bits & v.bits;
        // popcount parity
        x ^= x >> 16; x ^= x >> 8; x ^= x >> 4;
        x ^= x >> 2; x ^= x >> 1;
        return x & 1;
    }

    // XOR: u ⊕ v
    __host__ __device__ BitVec operator^(const BitVec& v) const {
        return BitVec(bits ^ v.bits, n);
    }

    // Hamming weight ||u||_0
    __host__ __device__ int weight() const {
        uint32_t x = bits;
        x = x - ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        return ((x + (x >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    // Get bit i
    __host__ __device__ int get(int i) const { return (bits >> i) & 1; }

    // Standard basis vector e_i
    __host__ __device__ static BitVec basis(int i, int nn) {
        return BitVec(1u << i, nn);
    }

    __host__ __device__ bool operator==(const BitVec& o) const { return bits == o.bits && n == o.n; }
};

// ── QubitState: V_n = ℂ^{2^n} ───────────────────────────────────────────────
// Amplitude vector for n qubits
struct QubitState {
    C64*    amp;    // amplitude array, length 2^n
    int     n;      // number of qubits
    int     dim;    // 2^n

    __host__ void init(int num_qubits) {
        n = num_qubits;
        dim = 1 << n;
        amp = new C64[dim];
        memset(amp, 0, dim * sizeof(C64));
    }

    __host__ void free() {
        delete[] amp;
        amp = nullptr;
    }

    // Computational basis state |x⟩
    __host__ void set_basis(uint32_t x) {
        memset(amp, 0, dim * sizeof(C64));
        amp[x] = C64(1.0, 0.0);
    }

    // Uniform superposition
    __host__ void set_uniform() {
        double a = 1.0 / sqrt(static_cast<double>(dim));
        for (int i = 0; i < dim; ++i) amp[i] = C64(a, 0.0);
    }

    // Inner product ⟨ψ|φ⟩
    __host__ C64 inner(const QubitState& other) const {
        C64 result(0, 0);
        for (int i = 0; i < dim; ++i) {
            result += amp[i].conj() * other.amp[i];
        }
        return result;
    }

    // Norm squared
    __host__ double norm2() const {
        double s = 0;
        for (int i = 0; i < dim; ++i) s += amp[i].norm2();
        return s;
    }

    // Probability of outcome x
    __host__ double prob(uint32_t x) const {
        return amp[x].norm2();
    }
};

// ── TopologicalState: discretized representation of X_A^top ─────────────────
// Represents a function on the phase space M = Θ × ℝ, discretized on a
// grid of resolution (N_θ × N_ρ)
struct TopologicalState {
    C64*    amp;     // amplitude grid [N_theta * N_rho]
    int     N_theta; // angular resolution
    int     N_rho;   // radial resolution
    double  rho_min; // minimum ρ
    double  rho_max; // maximum ρ
    int     total;   // N_theta * N_rho

    __host__ void init(int nth, int nrh, double rmin = -5.0, double rmax = 5.0) {
        N_theta = nth;
        N_rho = nrh;
        rho_min = rmin;
        rho_max = rmax;
        total = N_theta * N_rho;
        amp = new C64[total];
        memset(amp, 0, total * sizeof(C64));
    }

    __host__ void free() {
        delete[] amp;
        amp = nullptr;
    }

    // Grid point (i,j) → (θ, ρ)
    __host__ __device__ PhasePoint grid_point(int i_theta, int j_rho) const {
        double theta = constants::TWO_PI * i_theta / N_theta;
        double rho = rho_min + (rho_max - rho_min) * j_rho / (N_rho - 1);
        return PhasePoint(theta, rho);
    }

    // Linear index
    __host__ __device__ int index(int i_theta, int j_rho) const {
        return i_theta * N_rho + j_rho;
    }

    // Access amplitude at grid point
    __host__ __device__ C64& at(int i_theta, int j_rho) {
        return amp[index(i_theta, j_rho)];
    }
    __host__ __device__ const C64& at(int i_theta, int j_rho) const {
        return amp[index(i_theta, j_rho)];
    }

    // Initialize to delta at a particular phase point
    __host__ void set_delta(const PhasePoint& p) {
        memset(amp, 0, total * sizeof(C64));
        int i = static_cast<int>(p.theta / constants::TWO_PI * N_theta) % N_theta;
        if (i < 0) i += N_theta;
        int j = static_cast<int>((p.rho - rho_min) / (rho_max - rho_min) * (N_rho - 1));
        if (j >= 0 && j < N_rho) {
            amp[index(i, j)] = C64(1.0, 0.0);
        }
    }

    // Norm squared ⟨ψ|ψ⟩
    __host__ double norm2() const {
        double s = 0;
        for (int i = 0; i < total; ++i) s += amp[i].norm2();
        return s;
    }
};

// ── HybridState: X_{A,n}^hyb = V_n ⊗ X_A^top ────────────────────────────────
// Stores 2^n copies of TopologicalState (one per computational basis state)
// Implements Q_{A,n}: ⊕_x X_A^top → X_{A,n}^hyb
struct HybridState {
    C64*    amp;       // flat array: [2^n * N_theta * N_rho]
    int     n_qubits;
    int     dim_disc;  // 2^n
    int     N_theta;
    int     N_rho;
    double  rho_min, rho_max;
    int     dim_top;   // N_theta * N_rho
    int     total;     // dim_disc * dim_top

    __host__ void init(int nq, int nth, int nrh,
                       double rmin = -5.0, double rmax = 5.0) {
        n_qubits = nq;
        dim_disc = 1 << nq;
        N_theta = nth;
        N_rho = nrh;
        rho_min = rmin;
        rho_max = rmax;
        dim_top = N_theta * N_rho;
        total = dim_disc * dim_top;
        amp = new C64[total];
        memset(amp, 0, total * sizeof(C64));
    }

    __host__ void free() {
        delete[] amp;
        amp = nullptr;
    }

    // Index: [x, i_theta, j_rho]
    __host__ __device__ int index(int x, int i_theta, int j_rho) const {
        return x * dim_top + i_theta * N_rho + j_rho;
    }

    // ι_x(ξ) = e_x ⊗ ξ  — inject topological state into slot x
    __host__ void inject(int x, const TopologicalState& xi) {
        for (int i = 0; i < dim_top; ++i) {
            amp[x * dim_top + i] = xi.amp[i];
        }
    }

    // π_x(Ξ) = ξ_x  — project out the x-th topological component
    __host__ void project(int x, TopologicalState& out) const {
        for (int i = 0; i < dim_top; ++i) {
            out.amp[i] = amp[x * dim_top + i];
        }
    }

    // P_x^hyb = ι_x ∘ π_x  (projector onto slot x)
    __host__ void project_inplace(int x) {
        for (int xp = 0; xp < dim_disc; ++xp) {
            if (xp != x) {
                memset(amp + xp * dim_top, 0, dim_top * sizeof(C64));
            }
        }
    }

    // Q_{A,n}^{-1}(Ξ) = (π_x(Ξ))_{x ∈ I_n}
    __host__ void to_components(TopologicalState* out) const {
        for (int x = 0; x < dim_disc; ++x) {
            project(x, out[x]);
        }
    }

    // Q_{A,n}((ξ_x)_x) = ∑_x e_x ⊗ ξ_x
    __host__ void from_components(const TopologicalState* in) {
        for (int x = 0; x < dim_disc; ++x) {
            inject(x, in[x]);
        }
    }

    // Init(x) = e_x ⊗ ξ₀  (initial state for computation)
    __host__ void init_computation(int x, const TopologicalState& xi0) {
        memset(amp, 0, total * sizeof(C64));
        inject(x, xi0);
    }

    // Probability of measuring qubit outcome x: p(x) = Tr(P_x^hyb · ρ)
    __host__ double prob_disc(int x) const {
        double s = 0;
        for (int i = 0; i < dim_top; ++i) {
            s += amp[x * dim_top + i].norm2();
        }
        return s;
    }

    // Total norm squared
    __host__ double norm2() const {
        double s = 0;
        for (int i = 0; i < total; ++i) s += amp[i].norm2();
        return s;
    }
};

// ── Density operator (hybrid space) ─────────────────────────────────────────
// ρ ∈ End(X_{A,n}^hyb): ρ ≥ 0, Tr(ρ) = 1
struct DensityOp {
    C64*    mat;    // dense matrix [total × total]
    int     total;

    __host__ void init(int dim) {
        total = dim;
        mat = new C64[total * total];
        memset(mat, 0, total * total * sizeof(C64));
    }

    __host__ void free() {
        delete[] mat;
        mat = nullptr;
    }

    // From pure state |ψ⟩⟨ψ|
    __host__ void from_pure(const C64* psi, int dim) {
        total = dim;
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j)
                mat[i * dim + j] = psi[i] * psi[j].conj();
    }

    // Trace
    __host__ C64 trace() const {
        C64 tr(0, 0);
        for (int i = 0; i < total; ++i)
            tr += mat[i * total + i];
        return tr;
    }

    // Tr(ρ · A) for operator A
    __host__ C64 expectation(const C64* A) const {
        C64 result(0, 0);
        for (int i = 0; i < total; ++i)
            for (int j = 0; j < total; ++j)
                result += mat[i * total + j] * A[j * total + i];
        return result;
    }
};

} // namespace topcomp
