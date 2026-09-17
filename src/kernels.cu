// ============================================================================
// TopComp: CUDA Parallel Kernels
// ============================================================================
// GPU-parallelized implementations of the core computational primitives:
//
//   1. Orbit generation:     parallel golden-flow trajectories
//   2. PRG bit generation:   parallel cocycle-shadow pipeline
//   3. Bias computation:     parallel reduction for Fourier coefficients
//   4. Correlation:          parallel lag-correlation computation
//   5. Gate application:     parallel hybrid state evolution
//   6. Cocycle evaluation:   parallel cocycle verification
//   7. Structure detection:  parallel noise analysis
// ============================================================================

#include "topcomp/benchmark.cuh"

namespace topcomp {
namespace cuda {

// ── Kernel 1: Golden-flow orbit generation ──────────────────────────────────
// Each thread generates one orbit point: (θ_t, ρ_t) = Φ_t(θ_0, ρ_0)
__global__ void kernel_golden_orbit(
    double* theta_out,   // [N] output angles
    double* rho_out,     // [N] output log-radii
    double theta0,       // initial angle
    double rho0,         // initial log-radius
    double dt,           // time step
    int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    PhasePoint pt(theta0, rho0);
    PhasePoint evolved = GoldenFlow::flow(pt, dt * t);
    theta_out[t] = evolved.theta;
    rho_out[t]   = evolved.rho;
}

// ── Kernel 2: Mirror PRG bit generation ─────────────────────────────────────
// Each thread generates one bit: b_t = ℓ₂(q₂(Sh₂(Ω_A(g₀, g_t))))
__global__ void kernel_prng_bits(
    int* bits_out,       // [N] output bits
    double g0_s,         // seed Mir element sign
    double g0_c_real,    // seed Mir element c (real part)
    double g0_c_imag,    // seed Mir element c (imag part)
    int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    MirElement g0;
    g0.s = static_cast<int>(g0_s);
    g0.c = C64(g0_c_real, g0_c_imag);

    // g_t = g_φ(t) ★ g₀
    MirElement g_phi_t = mir::g_phi(static_cast<double>(t));
    MirElement gt = g_phi_t.star(g0);

    // Ω_A(g₀, g_t)
    C64 omega = cocycle::omega_mir(g0, gt);

    // Sh₂ → q₂ → ℓ₂
    bits_out[t] = cocycle::shadow_to_bit(omega);
}

// ── Kernel 3: Parallel bias reduction ───────────────────────────────────────
// Phase 1: each thread computes partial sum of (-1)^{b_t}
__global__ void kernel_bias_partial(
    const int* bits,     // [N] input bits
    double* partial,     // [num_blocks] partial sums
    int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < N) ? ((bits[gid] == 0) ? 1.0 : -1.0) : 0.0;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// ── Kernel 4: Parallel correlation ──────────────────────────────────────────
// Corr(h) = |(1/T) Σ (-1)^{b_t ⊕ b_{t+h}}|
__global__ void kernel_correlation(
    const int* bits,     // [N] input bits
    double* partial,     // [num_blocks] partial sums
    int lag,             // correlation lag h
    int N)
{
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (gid < N && gid + lag < N) {
        int parity = bits[gid] ^ bits[gid + lag];
        val = (parity == 0) ? 1.0 : -1.0;
    }

    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

// ── Kernel 5: Hybrid gate application (X_u on full state) ───────────────────
// X_u^hyb|ψ⟩ = swap amplitudes at x and x⊕u
__global__ void kernel_apply_X_hybrid(
    double* amp_real,    // [disc_dim * grid_size]
    double* amp_imag,    // [disc_dim * grid_size]
    int disc_dim,
    int grid_size,
    unsigned int u)      // bit-flip vector
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = disc_dim * grid_size;
    if (idx >= total) return;

    int x = idx / grid_size;
    int j = idx % grid_size;
    int x_flip = x ^ u;

    if (x < x_flip) {  // only swap once
        int idx_flip = x_flip * grid_size + j;
        double tr = amp_real[idx], ti = amp_imag[idx];
        amp_real[idx] = amp_real[idx_flip];
        amp_imag[idx] = amp_imag[idx_flip];
        amp_real[idx_flip] = tr;
        amp_imag[idx_flip] = ti;
    }
}

// ── Kernel 6: Hybrid gate application (Z_v phase kick) ──────────────────────
// Z_v^hyb|x,j⟩ = (-1)^{⟨v,x⟩} |x,j⟩
__global__ void kernel_apply_Z_hybrid(
    double* amp_real,
    double* amp_imag,
    int disc_dim,
    int grid_size,
    unsigned int v)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = disc_dim * grid_size;
    if (idx >= total) return;

    int x = idx / grid_size;

    // Compute ⟨v, x⟩ mod 2
    int inner = __popc(v & x) & 1;
    if (inner) {
        amp_real[idx] = -amp_real[idx];
        amp_imag[idx] = -amp_imag[idx];
    }
}

// ── Kernel 7: Topological evolution U_t on hybrid state ─────────────────────
// U_t^hyb = id_Vn ⊗ U_t^A : translates the topological component by t
__global__ void kernel_apply_U_hybrid(
    const double* amp_real_in,
    const double* amp_imag_in,
    double* amp_real_out,
    double* amp_imag_out,
    int disc_dim,
    int grid_size,
    double t_shift)       // shift parameter
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = disc_dim * grid_size;
    if (idx >= total) return;

    int x = idx / grid_size;
    int j = idx % grid_size;

    // Compute source index via bilinear interpolation
    double src_j = j - t_shift * grid_size / (2.0 * constants::PI);
    // Periodic wrap
    int j0 = ((int)floor(src_j) % grid_size + grid_size) % grid_size;
    int j1 = (j0 + 1) % grid_size;
    double frac = src_j - floor(src_j);

    int idx0 = x * grid_size + j0;
    int idx1 = x * grid_size + j1;

    amp_real_out[idx] = (1.0 - frac) * amp_real_in[idx0] + frac * amp_real_in[idx1];
    amp_imag_out[idx] = (1.0 - frac) * amp_imag_in[idx0] + frac * amp_imag_in[idx1];
}

// ── Kernel 8: Cocycle verification ──────────────────────────────────────────
// Verify dΩ(g₀, g₁, g₂) = 0 for a set of Mir triples
__global__ void kernel_verify_cocycle(
    const double* g_s,         // [N] signs
    const double* g_c_real,    // [N] c real parts
    const double* g_c_imag,    // [N] c imag parts
    double* coboundary_out,    // [N-2] coboundary values
    int N)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t + 2 >= N) return;

    MirElement g0, g1, g2;
    g0.s = static_cast<int>(g_s[t]);
    g0.c = C64(g_c_real[t], g_c_imag[t]);
    g1.s = static_cast<int>(g_s[t + 1]);
    g1.c = C64(g_c_real[t + 1], g_c_imag[t + 1]);
    g2.s = static_cast<int>(g_s[t + 2]);
    g2.c = C64(g_c_real[t + 2], g_c_imag[t + 2]);

    C64 cb = cocycle::cocycle_coboundary(g0, g1, g2);
    coboundary_out[t] = cb.abs();
}

// ── Kernel 9: Parallel structure detection ──────────────────────────────────
// Each thread processes one seed and computes β_N for that seed
__global__ void kernel_multi_seed_beta(
    double* beta_out,          // [num_seeds] β_N values
    const double* seeds_real,  // [num_seeds] seed c real parts
    const double* seeds_imag,  // [num_seeds] seed c imag parts
    int N,                     // bits per seed
    int num_seeds)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;

    MirElement g0;
    g0.s = 1;
    g0.c = C64(seeds_real[sid], seeds_imag[sid]);

    double sum = 0.0;
    for (int t = 0; t < N; ++t) {
        MirElement gt = mir::g_phi(static_cast<double>(t)).star(g0);
        C64 omega = cocycle::omega_mir(g0, gt);
        int bit = cocycle::shadow_to_bit(omega);
        sum += (bit == 0) ? 1.0 : -1.0;
    }
    beta_out[sid] = fabs(sum / N);
}

// ── Kernel 10: Noisy structure detection ────────────────────────────────────
// Generate bits with noise, compute bias to test if structure persists
__global__ void kernel_noisy_structure(
    double* bias_out,          // [num_noise_levels] bias at each noise level
    double g0_c_real,
    double g0_c_imag,
    int N,
    const double* noise_levels,  // [num_noise_levels]
    int num_noise_levels)
{
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid >= num_noise_levels) return;

    MirElement g0;
    g0.s = 1;
    g0.c = C64(g0_c_real, g0_c_imag);

    double noise = noise_levels[nid];
    double sum = 0.0;

    // Simple LCG for noise (per-thread state)
    unsigned int rng_state = 12345u + nid * 67890u;

    for (int t = 0; t < N; ++t) {
        MirElement gt = mir::g_phi(static_cast<double>(t)).star(g0);
        C64 omega = cocycle::omega_mir(g0, gt);
        int bit = cocycle::shadow_to_bit(omega);

        // Apply noise
        rng_state = rng_state * 1103515245u + 12345u;
        double u = (rng_state & 0x7FFFFFFF) / 2147483647.0;
        if (u < noise) bit ^= 1;

        sum += (bit == 0) ? 1.0 : -1.0;
    }
    bias_out[nid] = fabs(sum / N);
}

// ── Host-side kernel launcher functions ─────────────────────────────────────

constexpr int BLOCK_SIZE = 256;

inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Launch orbit generation on GPU
void launch_golden_orbit(
    double* d_theta, double* d_rho,
    double theta0, double rho0, double dt, int N)
{
    kernel_golden_orbit<<<grid_size(N), BLOCK_SIZE>>>(
        d_theta, d_rho, theta0, rho0, dt, N);
}

// Launch PRG bit generation on GPU
void launch_prng_bits(
    int* d_bits, const MirElement& g0, int N)
{
    kernel_prng_bits<<<grid_size(N), BLOCK_SIZE>>>(
        d_bits, g0.s, g0.c.re, g0.c.im, N);
}

// Launch bias computation (two-phase reduction)
double launch_bias(const int* d_bits, int N) {
    int num_blocks = grid_size(N);
    double* d_partial;
    cudaMalloc(&d_partial, num_blocks * sizeof(double));

    kernel_bias_partial<<<num_blocks, BLOCK_SIZE,
                          BLOCK_SIZE * sizeof(double)>>>(
        d_bits, d_partial, N);

    // Copy partial sums to host and reduce
    double* h_partial = new double[num_blocks];
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) sum += h_partial[i];

    delete[] h_partial;
    cudaFree(d_partial);

    return fabs(sum / N);
}

// Launch correlation computation
double launch_correlation(const int* d_bits, int N, int lag) {
    int effective_N = N - lag;
    if (effective_N <= 0) return 0.0;
    int num_blocks = grid_size(effective_N);
    double* d_partial;
    cudaMalloc(&d_partial, num_blocks * sizeof(double));

    kernel_correlation<<<num_blocks, BLOCK_SIZE,
                         BLOCK_SIZE * sizeof(double)>>>(
        d_bits, d_partial, lag, N);

    double* h_partial = new double[num_blocks];
    cudaMemcpy(h_partial, d_partial, num_blocks * sizeof(double),
               cudaMemcpyDeviceToHost);

    double sum = 0.0;
    for (int i = 0; i < num_blocks; ++i) sum += h_partial[i];

    delete[] h_partial;
    cudaFree(d_partial);

    return fabs(sum / effective_N);
}

// Launch multi-seed beta computation
void launch_multi_seed_beta(
    double* d_beta, const double* d_seeds_real, const double* d_seeds_imag,
    int N, int num_seeds)
{
    kernel_multi_seed_beta<<<grid_size(num_seeds), BLOCK_SIZE>>>(
        d_beta, d_seeds_real, d_seeds_imag, N, num_seeds);
}

// Launch noisy structure detection
void launch_noisy_structure(
    double* d_bias, double g0_c_real, double g0_c_imag,
    int N, const double* d_noise_levels, int num_levels)
{
    kernel_noisy_structure<<<grid_size(num_levels), BLOCK_SIZE>>>(
        d_bias, g0_c_real, g0_c_imag, N, d_noise_levels, num_levels);
}

} // namespace cuda
} // namespace topcomp
