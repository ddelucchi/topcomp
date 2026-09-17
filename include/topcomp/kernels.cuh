// ============================================================================
// TopComp: CUDA Kernel Launch Function Declarations
// ============================================================================
#pragma once
#include "mir_group.cuh"

namespace topcomp {
namespace cuda {

void launch_golden_orbit(double* d_theta, double* d_rho,
                         double theta0, double rho0, double dt, int N);

void launch_prng_bits(int* d_bits, const MirElement& g0, int N);

double launch_bias(const int* d_bits, int N);

double launch_correlation(const int* d_bits, int N, int lag);

void launch_multi_seed_beta(double* d_beta, const double* d_seeds_real,
                            const double* d_seeds_imag, int N, int num_seeds);

void launch_noisy_structure(double* d_bias, double g0_c_real,
                            double g0_c_imag, int N,
                            const double* d_noise_levels, int num_levels);

} // namespace cuda
} // namespace topcomp
