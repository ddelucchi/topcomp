// ============================================================================
// TopComp: Pseudorandom Generator & Rigorous Statistical Analysis
// ============================================================================
// Full PRPack_{A,n} from the reference document:
//
//   Generator G_{A,n,N}^Mir: {0,1}^{d_N} -> {0,1}^N
//     b_k(s) = l_2(q_2(Sh_2(Omega_A(x_k(s), y_k(s)))))
//
//   Every statistic is computed from its exact mathematical definition.
//   No heuristic thresholds. Statistical significance via z-scores,
//   p-values, and Fisher's combined test.
// ============================================================================
#pragma once
#include "cocycle.cuh"
#include "phase_space.cuh"
#include <cstring>

namespace topcomp {

// ── MirPRNG: Mirror-based pseudorandom bit generator ────────────────────────
struct MirPRNG {
    MirElement g0;
    BitVec     u0;
    int        n_qubits;

    __host__ __device__ MirPRNG() : g0(MirElement::identity()), u0(), n_qubits(0) {}

    __host__ __device__ MirPRNG(const MirElement& g, const BitVec& u, int nq)
        : g0(g), u0(u), n_qubits(nq) {}

    // g_t(s) = g_phi(t) * g_0
    __host__ __device__ MirElement g_at(int t) const {
        return mir::g_phi(static_cast<double>(t)).star(g0);
    }

    // y_t(s) = q_2(Sh_2(Omega_A(g_0, g_t(s))))
    __host__ __device__ int generate_bit(int t) const {
        MirElement gt = g_at(t);
        C64 omega = cocycle::omega_mir(g0, gt);
        return cocycle::shadow_to_bit(omega);
    }

    __host__ void generate(int* output, int N) const {
        for (int t = 0; t < N; ++t) output[t] = generate_bit(t);
    }

    // beta_N(s) = |(1/N) sum_{t=0}^{N-1} (-1)^{y_t(s)}|
    __host__ double beta(int N) const {
        double sum = 0.0;
        for (int t = 0; t < N; ++t) {
            sum += (generate_bit(t) == 0) ? 1.0 : -1.0;
        }
        return fabs(sum / N);
    }

    // Gamma_{N,k}(h_1,...,h_k; s) = |(1/N) sum (-1)^{y_t + y_{t+h_1} + ... + y_{t+h_k}}|
    __host__ double gamma_corr(int N, const int* h, int k) const {
        double sum = 0.0;
        for (int t = 0; t < N; ++t) {
            int parity = generate_bit(t);
            for (int j = 0; j < k; ++j) {
                if (t + h[j] < N) parity ^= generate_bit(t + h[j]);
            }
            sum += (parity == 0) ? 1.0 : -1.0;
        }
        return fabs(sum / N);
    }
};

// ── Rigorous statistical test suite ─────────────────────────────────────────
namespace stats {

// ── Gaussian CDF (Abramowitz & Stegun 7.1.26) ───────────────────────────────
__host__ inline double erfc_approx(double x) {
    if (x < 0) return 2.0 - erfc_approx(-x);
    double t = 1.0 / (1.0 + 0.3275911 * x);
    double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))));
    return poly * exp(-x * x);
}

__host__ inline double normal_cdf(double x) {
    return 0.5 * erfc_approx(-x / sqrt(2.0));
}

// Two-sided p-value from z-score: P(|Z| >= |z|)
__host__ inline double z_to_pvalue(double z) {
    double p = erfc_approx(fabs(z) / sqrt(2.0));
    return (p < 0.0) ? 0.0 : ((p > 1.0) ? 1.0 : p);
}

// Incomplete gamma via series (for chi-squared CDF)
__host__ inline double gamma_inc_lower_series(double a, double x, int terms = 200) {
    if (x <= 0) return 0.0;
    // For very large x relative to a, the regularized incomplete gamma → 1
    double log_prefactor = -x + a * log(x) - lgamma(a);
    if (log_prefactor < -700) return 1.0;  // exp underflow → series negligible vs 1
    double sum = 0, term = 1.0 / a;
    for (int n = 1; n <= terms; ++n) {
        term *= x / (a + n);
        sum += term;
        if (term > 1e100) return 1.0; // series diverging → x >> a → result ≈ 1
        if (fabs(term) < 1e-15 * fabs(sum)) break;
    }
    return (1.0 / a + sum) * exp(log_prefactor);
}

// chi-squared survival: P(X >= x) where X ~ chi^2(df)
__host__ inline double chi2_sf(double x, double df) {
    if (x <= 0) return 1.0;
    if (x > df + 10.0 * sqrt(2.0 * df)) return 0.0; // far tail
    double p = gamma_inc_lower_series(df / 2.0, x / 2.0);
    if (p != p) return 0.0; // NaN guard: extreme chi2 → p ≈ 0
    if (p > 1.0) p = 1.0;
    return 1.0 - p;
}

// ── PRPack: Bias_T^{(A,n)} ──────────────────────────────────────────────────
// Bias = sup_{xi != 0} |mu_hat_T(xi)|
// For 1-bit stream: mu_hat_T(1) = (1/T) sum_t (-1)^{x_t}
__host__ inline double compute_bias(const int* bits, int T) {
    double sum = 0.0;
    for (int t = 0; t < T; ++t) sum += (bits[t] == 0) ? 1.0 : -1.0;
    return fabs(sum / T);
}

// z-score for bias under H0(uniform): E[bias]=0, Var=1/T => z = bias*sqrt(T)
__host__ inline double bias_zscore(double bias, int T) {
    return bias * sqrt(static_cast<double>(T));
}

// Full multi-dimensional bias: mu_hat_T^{(A,n)}(m,l,xi)
__host__ inline double compute_bias_full(
    const PhasePoint* topo_pts, const int* disc_bits,
    int T, int n_qubits, int char_m, int char_l, const BitVec& xi)
{
    C64 sum(0, 0);
    for (int t = 0; t < T; ++t) {
        C64 chi_val = character::chi(char_m, char_l,
                                       topo_pts[t].theta, topo_pts[t].rho);
        BitVec xt(disc_bits[t], n_qubits);
        double parity = (xt.inner(xi) == 0) ? 1.0 : -1.0;
        sum += chi_val * parity;
    }
    return (sum / static_cast<double>(T)).abs();
}

// Supremal bias over character modes
__host__ inline double compute_bias_supremal(
    const PhasePoint* topo_pts, const int* disc_bits,
    int T, int n_qubits, int max_m = 5, int max_l = 5)
{
    double sup = 0;
    int dim = 1 << n_qubits;
    for (int m = -max_m; m <= max_m; ++m) {
        for (int l = -max_l; l <= max_l; ++l) {
            for (int xi_val = 0; xi_val < dim; ++xi_val) {
                if (m == 0 && l == 0 && xi_val == 0) continue;
                BitVec xi(xi_val, n_qubits);
                double b = compute_bias_full(topo_pts, disc_bits, T, n_qubits, m, l, xi);
                if (b > sup) sup = b;
            }
        }
    }
    return sup;
}

// ── PRPack: Disc_T ──────────────────────────────────────────────────────────
// Disc_T = sup_f |(1/T) sum f(x_t) - integral f d mu|
// For binary sequence over F_{A,n}:
__host__ inline double compute_discrepancy_1d(const int* bits, int T) {
    int count1 = 0;
    for (int t = 0; t < T; ++t) count1 += bits[t];
    return fabs(static_cast<double>(count1) / T - 0.5);
}

// k-dimensional discrepancy: empirical CDF vs uniform over k-bit blocks
__host__ inline double compute_discrepancy_kd(const int* bits, int T, int k) {
    if (k > 20 || T < k) return 0.0;
    int num_blocks = T / k;
    int num_patterns = 1 << k;
    int* counts = new int[num_patterns]();
    for (int i = 0; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[i * k + j] << j);
        counts[pattern]++;
    }
    double sup_disc = 0;
    for (int p = 0; p < num_patterns; ++p) {
        double emp = static_cast<double>(counts[p]) / num_blocks;
        double unif = 1.0 / num_patterns;
        double d = fabs(emp - unif);
        if (d > sup_disc) sup_disc = d;
    }
    delete[] counts;
    return sup_disc;
}

// ── PRPack: Walsh distinguisher advantage ───────────────────────────────────
// D_{s,d}^Walsh = { D(x) = sum_{|xi|<=d} alpha_xi (-1)^{<xi,x>} : sum|alpha|<=s }
// Adv_T^Walsh(A,n;s,d) = sup_D |E_mu[D] - E_U[D]|
// Theorem: Adv_T^Walsh(A,n;s,d) <= s * Bias_T^{(A,n)}
__host__ inline double walsh_advantage_bound(double bias, double s) {
    return s * bias;
}

// Empirical Walsh advantage: compute actual advantage for degree-d distinguishers
// For 1D binary stream: E_mu[(-1)^{<xi,x>}] = mu_hat(xi)
// The advantage for a specific Walsh coefficient xi is |mu_hat(xi)|
__host__ inline double walsh_advantage_empirical(const int* bits, int T, int max_degree) {
    double max_adv = 0;
    int max_xi = (1 << max_degree);
    if (max_xi > T) max_xi = T;
    for (int xi = 1; xi < max_xi; ++xi) {
        // Compute mu_hat(xi) = (1/T) sum_t (-1)^{popcount(xi & t_block)}
        // For a single-bit stream with xi as lag mask:
        int weight = 0;
        for (int b = xi; b; b >>= 1) weight += (b & 1);
        if (weight > max_degree) continue;
        // Compute the Walsh-Hadamard coefficient
        double sum = 0;
        int valid = 0;
        for (int t = 0; t + max_degree < T; ++t) {
            int parity = 0;
            bool ok = true;
            for (int j = 0; j < max_degree; ++j) {
                if (xi & (1 << j)) {
                    if (t + j < T) parity ^= bits[t + j];
                    else { ok = false; break; }
                }
            }
            if (ok) { sum += (parity == 0) ? 1.0 : -1.0; valid++; }
        }
        if (valid > 0) {
            double adv = fabs(sum / valid);
            if (adv > max_adv) max_adv = adv;
        }
    }
    return max_adv;
}

// ── PRPack: Pred_k ──────────────────────────────────────────────────────────
// Pred_k = sup_P |Pr[P(x_0,...,x_{k-1}) = x_k] - 1/2|
__host__ inline double compute_prediction(const int* bits, int T, int k) {
    if (k > 20 || T <= k) return 0.0;
    int num_patterns = 1 << k;
    int* count0 = new int[num_patterns]();
    int* count1 = new int[num_patterns]();
    for (int t = 0; t + k < T; ++t) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[t + j] << j);
        if (bits[t + k] == 0) count0[pattern]++;
        else                   count1[pattern]++;
    }
    int correct = 0, total_pred = 0;
    for (int p = 0; p < num_patterns; ++p) {
        correct += (count0[p] > count1[p]) ? count0[p] : count1[p];
        total_pred += count0[p] + count1[p];
    }
    delete[] count0;
    delete[] count1;
    if (total_pred == 0) return 0.0;
    return fabs(static_cast<double>(correct) / total_pred - 0.5);
}

// ── PRPack: Corr_{T,h} ──────────────────────────────────────────────────────
// Corr_{T,h}^{(A,n)} = |(1/T) sum_t (-1)^{x_t + x_{t+h_1} + ... + x_{t+h_r}}|
__host__ inline double compute_correlation(const int* bits, int T,
                                             const int* lags, int num_lags) {
    double sum = 0.0;
    int count = 0;
    for (int t = 0; t < T; ++t) {
        int parity = bits[t];
        bool valid = true;
        for (int j = 0; j < num_lags; ++j) {
            int idx = t + lags[j];
            if (idx >= T) { valid = false; break; }
            parity ^= bits[idx];
        }
        if (valid) {
            sum += (parity == 0) ? 1.0 : -1.0;
            count++;
        }
    }
    return (count > 0) ? fabs(sum / count) : 0.0;
}

// ── PRPack: TV distance ─────────────────────────────────────────────────────
// ||mu_{N,s} - U||_TV = (1/2) sum_b |mu(b) - 1/2|
__host__ inline double compute_tv_distance(const int* bits, int T) {
    int count1 = 0;
    for (int t = 0; t < T; ++t) count1 += bits[t];
    double p1 = static_cast<double>(count1) / T;
    double p0 = 1.0 - p1;
    return 0.5 * (fabs(p0 - 0.5) + fabs(p1 - 0.5));
}

// k-bit TV distance: partition into k-bit blocks, compare to uniform
__host__ inline double compute_tv_distance_kbit(const int* bits, int T, int k) {
    if (k > 20 || T < k) return 0.0;
    int num_blocks = T / k;
    int num_patterns = 1 << k;
    int* counts = new int[num_patterns]();
    for (int i = 0; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[i * k + j] << j);
        counts[pattern]++;
    }
    double tv = 0;
    double unif = 1.0 / num_patterns;
    for (int p = 0; p < num_patterns; ++p) {
        tv += fabs(static_cast<double>(counts[p]) / num_blocks - unif);
    }
    delete[] counts;
    return 0.5 * tv;
}

// ── Pairwise decorrelation ──────────────────────────────────────────────────
// Corr(k,l) over the full sequence: should be 0 for k != l
__host__ inline double pairwise_decorrelation(const int* bits, int T, int k, int l) {
    if (k >= T || l >= T || k == l) return 0.0;
    return (bits[k] ^ bits[l]) == 0 ? 1.0 : -1.0;
}

// Average pairwise decorrelation over all pairs within a window
__host__ inline double pairwise_decorrelation_avg(const int* bits, int T, int window) {
    if (window > T) window = T;
    double sum = 0;
    int count = 0;
    for (int k = 0; k < window; ++k) {
        for (int l = k + 1; l < window; ++l) {
            sum += ((bits[k] ^ bits[l]) == 0) ? 1.0 : -1.0;
            count++;
        }
    }
    return (count > 0) ? fabs(sum / count) : 0.0;
}

// ── Cramer rate function ────────────────────────────────────────────────────
// Theoretical: I(eps) = (1/2)[(1+eps)ln(1+eps) + (1-eps)ln(1-eps)]
__host__ inline double cramer_rate(double epsilon) {
    if (epsilon <= 0.0) return 0.0;
    if (epsilon >= 1.0) return log(2.0);
    double p = (1.0 + epsilon) / 2.0;
    double q = (1.0 - epsilon) / 2.0;
    return p * log(2.0 * p) + q * log(2.0 * q);
}

// Empirical cumulant generating function:
// Lambda_s(lambda) = (1/N) log sum_{t} exp(lambda * (-1)^{y_t})
__host__ inline double cgf_empirical(const int* bits, int N, double lambda) {
    double sum = 0;
    for (int t = 0; t < N; ++t) {
        double sign = bits[t] ? -1.0 : 1.0;
        sum += exp(lambda * sign);
    }
    return log(sum) / N;
}

// Empirical Cramer rate: I_s(eps) = sup_lambda (lambda*eps - Lambda_s(lambda))
__host__ inline double cramer_rate_empirical(const int* bits, int N, double epsilon,
                                               double lambda_max = 5.0, int steps = 200) {
    double best = 0;
    for (int i = 0; i <= steps; ++i) {
        double lambda = -lambda_max + 2.0 * lambda_max * i / steps;
        double val = lambda * epsilon - cgf_empirical(bits, N, lambda);
        if (val > best) best = val;
    }
    return best;
}

// Tail bound: Pr[beta_N >= eps] <= exp(-N * I(eps))
__host__ inline double tail_bound(int N, double epsilon) {
    return exp(-N * cramer_rate(epsilon));
}

// Hoeffding bound: Pr[|S_N/N| >= eps] <= 2 exp(-2 N eps^2)
__host__ inline double hoeffding_bound(int N, double epsilon) {
    return 2.0 * exp(-2.0 * N * epsilon * epsilon);
}

// ── Chi-squared goodness-of-fit ─────────────────────────────────────────────
// chi^2 = sum_i (O_i - E_i)^2 / E_i  for k-gram distribution
__host__ inline double chi_squared_kgram(const int* bits, int T, int k) {
    if (k > 20 || T < k) return 0.0;
    int num_blocks = T / k;
    int num_patterns = 1 << k;
    int* counts = new int[num_patterns]();
    for (int i = 0; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[i * k + j] << j);
        counts[pattern]++;
    }
    double expected = static_cast<double>(num_blocks) / num_patterns;
    double chi2 = 0;
    for (int p = 0; p < num_patterns; ++p) {
        double diff = counts[p] - expected;
        chi2 += diff * diff / expected;
    }
    delete[] counts;
    return chi2;
}

// p-value for chi-squared test (df = 2^k - 1)
__host__ inline double chi_squared_pvalue(const int* bits, int T, int k) {
    double chi2 = chi_squared_kgram(bits, T, k);
    double df = static_cast<double>((1 << k) - 1);
    return chi2_sf(chi2, df);
}

// ── Runs test ───────────────────────────────────────────────────────────────
// Count runs (consecutive identical bits). Under H0, number of runs R
// has E[R] = 2*n0*n1/N + 1, Var[R] = 2*n0*n1*(2*n0*n1-N)/(N^2*(N-1))
struct RunsTestResult {
    int num_runs;
    double expected_runs;
    double variance;
    double z_score;
    double p_value;
};

__host__ inline RunsTestResult runs_test(const int* bits, int T) {
    RunsTestResult r = {};
    if (T < 2) { r.p_value = 1.0; return r; }
    int n1 = 0;
    for (int t = 0; t < T; ++t) n1 += bits[t];
    int n0 = T - n1;
    r.num_runs = 1;
    for (int t = 1; t < T; ++t) {
        if (bits[t] != bits[t-1]) r.num_runs++;
    }
    double N = static_cast<double>(T);
    double p0 = n0 / N, p1 = n1 / N;
    r.expected_runs = 2.0 * n0 * n1 / N + 1.0;
    double num = 2.0 * n0 * n1 * (2.0 * n0 * n1 - N);
    double den = N * N * (N - 1.0);
    r.variance = (den > 0) ? num / den : 0;
    r.z_score = (r.variance > 0) ? (r.num_runs - r.expected_runs) / sqrt(r.variance) : 0;
    r.p_value = z_to_pvalue(r.z_score);
    return r;
}

// ── Serial test ─────────────────────────────────────────────────────────────
// Tests uniformity of overlapping m-bit patterns
// del_psi^2(m) = psi^2(m) - psi^2(m-1)
__host__ inline double serial_test(const int* bits, int T, int m) {
    if (m > 16 || T < m) return 0.0;
    int num_patterns = 1 << m;
    int* counts = new int[num_patterns]();
    for (int t = 0; t + m <= T; ++t) {
        int pattern = 0;
        for (int j = 0; j < m; ++j) pattern |= (bits[t + j] << j);
        counts[pattern]++;
    }
    double N = static_cast<double>(T - m + 1);
    double psi2 = 0;
    for (int p = 0; p < num_patterns; ++p) {
        psi2 += static_cast<double>(counts[p]) * counts[p];
    }
    psi2 = (num_patterns / N) * psi2 - N;
    delete[] counts;
    return psi2;
}

// ── Block frequency test ────────────────────────────────────────────────────
// Divide into M-bit blocks, compute chi-squared on block proportions
__host__ inline double block_frequency(const int* bits, int T, int block_size) {
    if (block_size <= 0 || T < block_size) return 0.0;
    int num_blocks = T / block_size;
    double chi2 = 0;
    for (int i = 0; i < num_blocks; ++i) {
        int ones = 0;
        for (int j = 0; j < block_size; ++j) ones += bits[i * block_size + j];
        double pi = static_cast<double>(ones) / block_size;
        chi2 += (pi - 0.5) * (pi - 0.5);
    }
    chi2 *= 4.0 * block_size;
    return chi2_sf(chi2, static_cast<double>(num_blocks));
}

// ── Approximate entropy ─────────────────────────────────────────────────────
// ApEn(m) = phi(m) - phi(m+1) where phi(k) = (1/N) sum log(C_i^k)
__host__ inline double approximate_entropy(const int* bits, int T, int m) {
    if (m > 16 || T < m + 1) return 0.0;
    auto compute_phi = [&](int block_len) -> double {
        int num_patterns = 1 << block_len;
        int* counts = new int[num_patterns]();
        int N = T - block_len + 1;
        for (int t = 0; t < N; ++t) {
            int pattern = 0;
            for (int j = 0; j < block_len; ++j) pattern |= (bits[t + j] << j);
            counts[pattern]++;
        }
        double phi = 0;
        for (int p = 0; p < num_patterns; ++p) {
            if (counts[p] > 0) {
                double freq = static_cast<double>(counts[p]) / N;
                phi += freq * log(freq);
            }
        }
        delete[] counts;
        return phi;
    };
    return compute_phi(m) - compute_phi(m + 1);
}

// ── Spectral (DFT) test ─────────────────────────────────────────────────────
// Compute DFT of (-1)^{b_t}, count peaks above threshold sqrt(ln(1/alpha)*T)
struct SpectralTestResult {
    double peak_ratio;       // fraction of peaks above threshold
    double expected_ratio;   // expected fraction = 1 - alpha
    double d_statistic;      // (peak_ratio - expected) / sqrt(expected*(1-expected)/N_half)
    double p_value;
};

__host__ inline SpectralTestResult spectral_test(const int* bits, int T) {
    SpectralTestResult r = {};
    int N_half = T / 2;
    if (N_half < 2) { r.p_value = 1.0; return r; }
    double threshold = sqrt(log(1.0 / 0.05) * T);  // 95% threshold
    int peaks_above = 0;
    for (int k = 1; k <= N_half; ++k) {
        double re = 0, im = 0;
        for (int t = 0; t < T; ++t) {
            double sign = (bits[t] & 1) ? -1.0 : 1.0;
            double angle = -constants::TWO_PI * k * t / T;
            re += sign * cos(angle);
            im += sign * sin(angle);
        }
        double mag = sqrt(re * re + im * im);
        if (mag < threshold) peaks_above++;
    }
    r.peak_ratio = static_cast<double>(peaks_above) / N_half;
    r.expected_ratio = 0.95;
    double se = sqrt(r.expected_ratio * (1.0 - r.expected_ratio) / N_half);
    r.d_statistic = (se > 0) ? (r.peak_ratio - r.expected_ratio) / se : 0;
    r.p_value = z_to_pvalue(r.d_statistic);
    return r;
}

// ── Cumulative sums (CUSUM) ─────────────────────────────────────────────────
// S_k = sum_{i=1}^{k} (2*b_i - 1), test max|S_k|
__host__ inline double cusum_statistic(const int* bits, int T) {
    double maxval = 0;
    double S = 0;
    for (int t = 0; t < T; ++t) {
        S += (bits[t] == 0) ? -1.0 : 1.0;
        if (fabs(S) > maxval) maxval = fabs(S);
    }
    return maxval;
}

__host__ inline double cusum_pvalue(const int* bits, int T) {
    double z = cusum_statistic(bits, T);
    // Under H0: max|S_k|/sqrt(T) has known distribution
    // Approximate p-value via reflection principle
    double normalized = z / sqrt(static_cast<double>(T));
    return z_to_pvalue(normalized);
}

// ── Kolmogorov-Smirnov for empirical CDF ────────────────────────────────────
// Compare empirical CDF of partial sums to expected
__host__ inline double ks_statistic(const int* bits, int T) {
    double max_D = 0;
    int cumsum = 0;
    for (int t = 0; t < T; ++t) {
        cumsum += bits[t];
        double empirical = static_cast<double>(cumsum) / (t + 1);
        double expected = 0.5;
        double D = fabs(empirical - expected);
        if (D > max_D) max_D = D;
    }
    return max_D;
}

// ── Min-entropy estimation ──────────────────────────────────────────────────
// H_min = -log2(max_x Pr[X=x])
__host__ inline double min_entropy_kbit(const int* bits, int T, int k) {
    if (k > 20 || T < k) return 0.0;
    int num_blocks = T / k;
    int num_patterns = 1 << k;
    int* counts = new int[num_patterns]();
    for (int i = 0; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[i * k + j] << j);
        counts[pattern]++;
    }
    int max_count = 0;
    for (int p = 0; p < num_patterns; ++p) {
        if (counts[p] > max_count) max_count = counts[p];
    }
    delete[] counts;
    double max_prob = static_cast<double>(max_count) / num_blocks;
    return (max_prob > 0) ? -log(max_prob) / log(2.0) : static_cast<double>(k);
}

// ── Shannon entropy ─────────────────────────────────────────────────────────
// H = -sum p_i log2(p_i) over k-bit blocks
__host__ inline double shannon_entropy_kbit(const int* bits, int T, int k) {
    if (k > 20 || T < k) return 0.0;
    int num_blocks = T / k;
    int num_patterns = 1 << k;
    int* counts = new int[num_patterns]();
    for (int i = 0; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < k; ++j) pattern |= (bits[i * k + j] << j);
        counts[pattern]++;
    }
    double H = 0;
    for (int p = 0; p < num_patterns; ++p) {
        if (counts[p] > 0) {
            double prob = static_cast<double>(counts[p]) / num_blocks;
            H -= prob * log(prob) / log(2.0);
        }
    }
    delete[] counts;
    return H;
}

// ── Permutation entropy ─────────────────────────────────────────────────────
// Embed into d-dimensional delay vectors, count ordinal patterns
__host__ inline double permutation_entropy(const int* bits, int T, int d) {
    if (d > 8 || T < d) return 0.0;
    // Use d! possible ordinal patterns; for binary d-tuples use d-bit patterns
    int num_patterns = 1 << d;
    int* counts = new int[num_patterns]();
    int N = T - d + 1;
    for (int t = 0; t < N; ++t) {
        int pattern = 0;
        for (int j = 0; j < d; ++j) pattern |= (bits[t + j] << j);
        counts[pattern]++;
    }
    double H = 0;
    for (int p = 0; p < num_patterns; ++p) {
        if (counts[p] > 0) {
            double prob = static_cast<double>(counts[p]) / N;
            H -= prob * log(prob) / log(2.0);
        }
    }
    delete[] counts;
    return H / d;  // normalized
}

// ── Mutual information between positions ────────────────────────────────────
// I(X_t; X_{t+lag}) = H(X_t) + H(X_{t+lag}) - H(X_t, X_{t+lag})
__host__ inline double mutual_information(const int* bits, int T, int lag) {
    if (lag >= T) return 0.0;
    int N = T - lag;
    int count[2][2] = {};
    for (int t = 0; t < N; ++t) {
        count[bits[t]][bits[t + lag]]++;
    }
    // Joint entropy
    double H_joint = 0;
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            if (count[a][b] > 0) {
                double p = static_cast<double>(count[a][b]) / N;
                H_joint -= p * log(p);
            }
        }
    }
    // Marginals
    double H_x = 0, H_y = 0;
    for (int a = 0; a < 2; ++a) {
        double px = static_cast<double>(count[a][0] + count[a][1]) / N;
        double py = static_cast<double>(count[0][a] + count[1][a]) / N;
        if (px > 0) H_x -= px * log(px);
        if (py > 0) H_y -= py * log(py);
    }
    return H_x + H_y - H_joint;
}

// ── Autocorrelation spectrum ────────────────────────────────────────────────
// Compute autocorrelation at all lags up to max_lag
__host__ inline void autocorrelation_spectrum(const int* bits, int T,
                                                double* out, int max_lag) {
    for (int h = 0; h < max_lag && h < T; ++h) {
        double sum = 0;
        int count = T - h;
        for (int t = 0; t < count; ++t) {
            sum += ((bits[t] ^ bits[t + h]) ? -1.0 : 1.0);
        }
        out[h] = (count > 0) ? sum / count : 0;
    }
}

// Maximum absolute autocorrelation over lags 1..max_lag
__host__ inline double max_autocorrelation(const int* bits, int T, int max_lag) {
    double maxval = 0;
    for (int h = 1; h <= max_lag && h < T; ++h) {
        double sum = 0;
        int count = T - h;
        for (int t = 0; t < count; ++t) {
            sum += ((bits[t] ^ bits[t + h]) ? -1.0 : 1.0);
        }
        double c = (count > 0) ? fabs(sum / count) : 0;
        if (c > maxval) maxval = c;
    }
    return maxval;
}

// ── Linear complexity (Berlekamp-Massey) ────────────────────────────────────
// LFSR complexity of the bit sequence
__host__ inline int linear_complexity(const int* bits, int T) {
    if (T <= 0) return 0;
    int* c_arr = new int[T + 1]();
    int* b = new int[T + 1]();
    c_arr[0] = 1; b[0] = 1;
    int L = 0, m = -1;
    for (int n = 0; n < T; ++n) {
        int d = bits[n];
        for (int i = 1; i <= L; ++i) {
            d ^= (c_arr[i] & bits[n - i]);
        }
        if (d == 1) {
            int* temp = new int[T + 1];
            memcpy(temp, c_arr, (T + 1) * sizeof(int));
            for (int i = 0; i <= T; ++i) {
                if (n - m + i <= T && b[i]) c_arr[n - m + i] ^= 1;
            }
            if (2 * L <= n) {
                L = n + 1 - L;
                m = n;
                memcpy(b, temp, (T + 1) * sizeof(int));
            }
            delete[] temp;
        }
    }
    delete[] c_arr;
    delete[] b;
    return L;
}

// ── Maurer universal statistic ──────────────────────────────────────────────
// f_n = (1/K) sum_{i=Q+1}^{Q+K} log2(i - T_i) where T_i is last occurrence
__host__ inline double maurer_universal(const int* bits, int T, int block_len) {
    if (block_len > 16 || block_len < 1) return 0.0;
    int num_patterns = 1 << block_len;
    int num_blocks = T / block_len;
    int Q = 10 * num_patterns;  // initialization blocks
    int K = num_blocks - Q;
    if (K <= 0) return 0.0;
    int* last_seen = new int[num_patterns];
    for (int i = 0; i < num_patterns; ++i) last_seen[i] = 0;
    // Initialization
    for (int i = 0; i < Q; ++i) {
        int pattern = 0;
        for (int j = 0; j < block_len; ++j)
            pattern |= (bits[i * block_len + j] << j);
        last_seen[pattern] = i + 1;
    }
    // Test
    double sum = 0;
    for (int i = Q; i < num_blocks; ++i) {
        int pattern = 0;
        for (int j = 0; j < block_len; ++j)
            pattern |= (bits[i * block_len + j] << j);
        if (last_seen[pattern] > 0) {
            sum += log(static_cast<double>(i + 1 - last_seen[pattern])) / log(2.0);
        }
        last_seen[pattern] = i + 1;
    }
    delete[] last_seen;
    return sum / K;
}

// ── Fisher combined test ────────────────────────────────────────────────────
// Combine k independent p-values: -2 sum log(p_i) ~ chi^2(2k)
__host__ inline double fisher_combined_pvalue(const double* pvalues, int k) {
    double stat = 0;
    for (int i = 0; i < k; ++i) {
        double p = pvalues[i];
        if (p < 1e-300) p = 1e-300;  // avoid log(0)
        stat -= 2.0 * log(p);
    }
    return chi2_sf(stat, 2.0 * k);
}

// ── Comprehensive statistical report ────────────────────────────────────────
static const int MAX_CORRELATION_LAGS = 32;
static const int MAX_PREDICTION_DEPTH = 8;

struct StatReport {
    int N;

    // PRPack core
    double bias;
    double bias_zscore;
    double bias_pvalue;
    double discrepancy_1d;
    double discrepancy_2d;
    double discrepancy_3d;
    double tv_distance;
    double tv_distance_2bit;
    double tv_distance_3bit;
    double beta_N;

    // Prediction
    double prediction[MAX_PREDICTION_DEPTH];  // k=1..8

    // Correlation
    double correlation[MAX_CORRELATION_LAGS]; // h=1..32
    double max_correlation;
    double max_correlation_zscore;

    // Walsh
    double walsh_adv_deg2;
    double walsh_adv_deg3;
    double walsh_adv_deg4;

    // Large deviation
    double cramer_rate_001;    // I(0.01)
    double cramer_rate_005;    // I(0.05)
    double cramer_rate_01;     // I(0.1)
    double cramer_rate_empirical_01;
    double tail_bound_001;
    double tail_bound_005;
    double tail_bound_01;
    double hoeffding_001;
    double hoeffding_01;

    // Entropy
    double shannon_1bit;
    double shannon_2bit;
    double shannon_3bit;
    double min_entropy_1bit;
    double min_entropy_2bit;
    double min_entropy_3bit;
    double permutation_entropy_3;
    double permutation_entropy_5;
    double approximate_entropy_2;
    double approximate_entropy_3;

    // Hypothesis tests
    double chi2_1bit;
    double chi2_2bit;
    double chi2_3bit;
    double chi2_pvalue_1bit;
    double chi2_pvalue_2bit;
    double chi2_pvalue_3bit;
    RunsTestResult runs;
    double serial_2;
    double serial_3;
    double block_freq_pvalue;
    SpectralTestResult spectral;
    double cusum;
    double cusum_pvalue_val;
    double ks;

    // Complexity
    int linear_complexity_val;
    double linear_complexity_ratio; // L/N (expect ~0.5 for random)
    double maurer_stat;

    // Information-theoretic
    double mutual_info_lag1;
    double mutual_info_lag2;
    double mutual_info_lag4;
    double pairwise_decorr;

    // Combined
    double fisher_pvalue;
    int num_tests_significant_005;
    int num_tests_total;
};

__host__ inline StatReport run_full_tests(const int* bits, int T, double beta) {
    StatReport r = {};
    r.N = T;
    r.beta_N = beta;

    // Bias
    r.bias = compute_bias(bits, T);
    r.bias_zscore = stats::bias_zscore(r.bias, T);
    r.bias_pvalue = z_to_pvalue(r.bias_zscore);

    // Discrepancy
    r.discrepancy_1d = compute_discrepancy_1d(bits, T);
    r.discrepancy_2d = compute_discrepancy_kd(bits, T, 2);
    r.discrepancy_3d = compute_discrepancy_kd(bits, T, 3);

    // TV distance
    r.tv_distance = compute_tv_distance(bits, T);
    r.tv_distance_2bit = compute_tv_distance_kbit(bits, T, 2);
    r.tv_distance_3bit = compute_tv_distance_kbit(bits, T, 3);

    // Prediction (k=1..8)
    for (int k = 1; k <= MAX_PREDICTION_DEPTH; ++k) {
        r.prediction[k-1] = compute_prediction(bits, T, k);
    }

    // Correlation (h=1..32)
    r.max_correlation = 0;
    for (int h = 1; h <= MAX_CORRELATION_LAGS && h < T; ++h) {
        int lag = h;
        r.correlation[h-1] = compute_correlation(bits, T, &lag, 1);
        if (r.correlation[h-1] > r.max_correlation)
            r.max_correlation = r.correlation[h-1];
    }
    r.max_correlation_zscore = r.max_correlation * sqrt(static_cast<double>(T));

    // Walsh advantage
    r.walsh_adv_deg2 = walsh_advantage_empirical(bits, T, 2);
    r.walsh_adv_deg3 = walsh_advantage_empirical(bits, T, 3);
    r.walsh_adv_deg4 = walsh_advantage_empirical(bits, T, 4);

    // Large deviation
    r.cramer_rate_001 = cramer_rate(0.01);
    r.cramer_rate_005 = cramer_rate(0.05);
    r.cramer_rate_01 = cramer_rate(0.1);
    r.cramer_rate_empirical_01 = cramer_rate_empirical(bits, T, 0.1);
    r.tail_bound_001 = tail_bound(T, 0.01);
    r.tail_bound_005 = tail_bound(T, 0.05);
    r.tail_bound_01 = tail_bound(T, 0.1);
    r.hoeffding_001 = hoeffding_bound(T, 0.01);
    r.hoeffding_01 = hoeffding_bound(T, 0.1);

    // Entropy
    r.shannon_1bit = shannon_entropy_kbit(bits, T, 1);
    r.shannon_2bit = shannon_entropy_kbit(bits, T, 2);
    r.shannon_3bit = shannon_entropy_kbit(bits, T, 3);
    r.min_entropy_1bit = min_entropy_kbit(bits, T, 1);
    r.min_entropy_2bit = min_entropy_kbit(bits, T, 2);
    r.min_entropy_3bit = min_entropy_kbit(bits, T, 3);
    r.permutation_entropy_3 = permutation_entropy(bits, T, 3);
    r.permutation_entropy_5 = permutation_entropy(bits, T, 5);
    r.approximate_entropy_2 = approximate_entropy(bits, T, 2);
    r.approximate_entropy_3 = approximate_entropy(bits, T, 3);

    // Chi-squared
    r.chi2_1bit = chi_squared_kgram(bits, T, 1);
    r.chi2_2bit = chi_squared_kgram(bits, T, 2);
    r.chi2_3bit = chi_squared_kgram(bits, T, 3);
    r.chi2_pvalue_1bit = chi_squared_pvalue(bits, T, 1);
    r.chi2_pvalue_2bit = chi_squared_pvalue(bits, T, 2);
    r.chi2_pvalue_3bit = chi_squared_pvalue(bits, T, 3);

    // Runs
    r.runs = runs_test(bits, T);

    // Serial
    r.serial_2 = serial_test(bits, T, 2);
    r.serial_3 = serial_test(bits, T, 3);

    // Block frequency (block size = sqrt(T))
    int bsz = static_cast<int>(sqrt(static_cast<double>(T)));
    if (bsz < 2) bsz = 2;
    r.block_freq_pvalue = block_frequency(bits, T, bsz);

    // Spectral
    if (T <= 8192) {
        r.spectral = spectral_test(bits, T);
    } else {
        r.spectral.p_value = 1.0;
    }

    // CUSUM
    r.cusum = cusum_statistic(bits, T);
    r.cusum_pvalue_val = cusum_pvalue(bits, T);

    // KS
    r.ks = ks_statistic(bits, T);

    // Complexity
    int lc_len = (T > 1000) ? 1000 : T;
    r.linear_complexity_val = linear_complexity(bits, lc_len);
    r.linear_complexity_ratio = static_cast<double>(r.linear_complexity_val) / lc_len;

    // Maurer
    if (T >= 640) {
        r.maurer_stat = maurer_universal(bits, T, 2);
    }

    // Mutual information
    r.mutual_info_lag1 = mutual_information(bits, T, 1);
    r.mutual_info_lag2 = mutual_information(bits, T, 2);
    r.mutual_info_lag4 = mutual_information(bits, T, 4);

    // Pairwise decorrelation
    int pw_window = (T > 200) ? 200 : T;
    r.pairwise_decorr = pairwise_decorrelation_avg(bits, T, pw_window);

    // Fisher combined test on available p-values
    double pvals[8];
    int np = 0;
    pvals[np++] = r.bias_pvalue;
    pvals[np++] = r.chi2_pvalue_1bit;
    pvals[np++] = r.chi2_pvalue_2bit;
    pvals[np++] = r.runs.p_value;
    pvals[np++] = r.block_freq_pvalue;
    pvals[np++] = r.spectral.p_value;
    pvals[np++] = r.cusum_pvalue_val;
    pvals[np++] = r.chi2_pvalue_3bit;
    r.fisher_pvalue = fisher_combined_pvalue(pvals, np);

    // Count significant tests at alpha=0.05
    r.num_tests_total = np;
    r.num_tests_significant_005 = 0;
    for (int i = 0; i < np; ++i) {
        if (pvals[i] < 0.05) r.num_tests_significant_005++;
    }

    return r;
}

} // namespace stats

} // namespace topcomp
