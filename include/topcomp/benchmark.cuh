// ============================================================================
// TopComp: Benchmark Framework
// ============================================================================
// Implements the full benchmark formalism from the paper:
//
//   B_{A,n} = (D_train, D_test, M, Π, L, A, C)
//
//   Metrics:
//     Acc(Θ) = (1/N) Σ 1[argmax f_Θ(x_j) = y_j]
//     NLL(Θ) = -(1/N) Σ [y log f + (1-y) log(1-f)]         (cross-entropy)
//     MSE(Θ) = (1/N) Σ ||f_Θ(x) - y||²
//     Brier   = (1/N) Σ_c (p(c) - 1[c=y])²
//     TV(Θ)  = (1/2) Σ_b |p̂(b) - p_data(b)|
//
//   Cost:
//     wall(Θ) = Σ time(P_{Θ,j})
//     mem(Θ)  = max space(P_{Θ,j})
//     thr(Θ)  = N / wall(Θ)
//     ene(Θ)  = Σ E(P_{Θ,j})
//
//   Gain:
//     Gain_acc  = (A_new - A_base) / (1 - A_base)
//     Gain_cost = (C_base - C_new) / C_base
//     Approx_qc = ||K_hyb - K_qc||_HS / ||K_qc||_HS
//
//   Pareto: Min{(NLL, wall, mem, ene)}
//   Optimal: Θ* = argmin L(Θ)
// ============================================================================
#pragma once
#include "machine.cuh"
#include <cmath>

namespace topcomp {

// ── Data point ──────────────────────────────────────────────────────────────
struct DataPoint {
    double* features;  // input features (mapped to hybrid state)
    int     label;     // ground truth class
    int     n_features;

    __host__ DataPoint() : features(nullptr), label(0), n_features(0) {}
    __host__ ~DataPoint() { /* caller manages memory */ }
};

// ── Dataset ─────────────────────────────────────────────────────────────────
struct Dataset {
    DataPoint* points;
    int size;

    __host__ Dataset() : points(nullptr), size(0) {}
};

// ── Metric results ──────────────────────────────────────────────────────────
struct Metrics {
    double accuracy;
    double nll;             // negative log-likelihood (cross-entropy)
    double mse;             // mean squared error
    double brier;           // Brier score
    double tv;              // total variation distance
    double wall_time;       // total wall time (gate count)
    double mem;             // max space
    double throughput;      // N / wall_time
    double energy;          // total energy proxy
    double gain_acc;        // relative accuracy gain
    double gain_cost;       // relative cost gain

    void print() const {
        printf("═══════════════════════════════════════════════════\n");
        printf("  Benchmark Metrics\n");
        printf("═══════════════════════════════════════════════════\n");
        printf("  Accuracy:        %.6f\n", accuracy);
        printf("  NLL (CE loss):   %.6f\n", nll);
        printf("  MSE:             %.6f\n", mse);
        printf("  Brier score:     %.6f\n", brier);
        printf("  TV distance:     %.6f\n", tv);
        printf("  Wall time:       %.2f\n", wall_time);
        printf("  Memory (space):  %.2f\n", mem);
        printf("  Throughput:      %.2f\n", throughput);
        printf("  Energy:          %.2f\n", energy);
        printf("  Gain (acc):      %.4f\n", gain_acc);
        printf("  Gain (cost):     %.4f\n", gain_cost);
        printf("═══════════════════════════════════════════════════\n");
    }
};

// ── Pareto point ────────────────────────────────────────────────────────────
struct ParetoPoint {
    double nll;
    double wall;
    double mem;
    double energy;
    int    config_id;

    __host__ bool dominates(const ParetoPoint& other) const {
        return (nll <= other.nll && wall <= other.wall &&
                mem <= other.mem && energy <= other.energy) &&
               (nll < other.nll || wall < other.wall ||
                mem < other.mem || energy < other.energy);
    }
};

// ── Benchmark engine ────────────────────────────────────────────────────────

namespace benchmark {

// Acc(Θ) = (1/N) Σ 1[ŷ = y]
__host__ inline double compute_accuracy(const int* predictions,
                                           const int* labels, int N) {
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        if (predictions[i] == labels[i]) correct++;
    }
    return static_cast<double>(correct) / N;
}

// NLL(Θ) = -(1/N) Σ [y log(p) + (1-y) log(1-p)]
__host__ inline double compute_nll(const double* probs,
                                      const int* labels, int N) {
    double nll = 0.0;
    for (int i = 0; i < N; ++i) {
        double p = probs[i];
        if (p < 1e-15) p = 1e-15;
        if (p > 1.0 - 1e-15) p = 1.0 - 1e-15;
        if (labels[i] == 1) {
            nll -= log(p);
        } else {
            nll -= log(1.0 - p);
        }
    }
    return nll / N;
}

// MSE(Θ) = (1/N) Σ (f(x) - y)²
__host__ inline double compute_mse(const double* probs,
                                      const int* labels, int N) {
    double mse = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = probs[i] - labels[i];
        mse += diff * diff;
    }
    return mse / N;
}

// Brier = (1/N) Σ_c (p(c) - 1[c=y])²
// For binary: same as MSE
__host__ inline double compute_brier(const double* probs,
                                        const int* labels, int N) {
    return compute_mse(probs, labels, N);
}

// TV(Θ) = (1/2) Σ_b |p̂(b) - p_data(b)|
__host__ inline double compute_tv(const double* probs,
                                     const int* labels, int N) {
    // Estimate predicted and empirical distributions
    double pred_1 = 0.0, data_1 = 0.0;
    for (int i = 0; i < N; ++i) {
        pred_1 += probs[i];
        data_1 += labels[i];
    }
    pred_1 /= N;
    data_1 /= N;
    return 0.5 * (fabs(pred_1 - data_1) + fabs((1.0 - pred_1) - (1.0 - data_1)));
}

// wall(Θ) = Σ time(P_j)
__host__ inline double compute_wall(const Program* programs, int N) {
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        total += programs[i].time();
    }
    return total;
}

// mem(Θ) = max space(P_j)
__host__ inline double compute_mem(const Program* programs, int N,
                                      int n_qubits) {
    int mx = 0;
    for (int i = 0; i < N; ++i) {
        int s = programs[i].space(n_qubits);
        if (s > mx) mx = s;
    }
    return static_cast<double>(mx);
}

// thr(Θ) = N / wall(Θ)
__host__ inline double compute_throughput(int N, double wall) {
    return (wall > 0) ? N / wall : 0.0;
}

// Gain_acc = (A_new - A_base) / (1 - A_base)
__host__ inline double compute_gain_acc(double acc_new, double acc_base) {
    if (acc_base >= 1.0 - 1e-15) return 0.0;
    return (acc_new - acc_base) / (1.0 - acc_base);
}

// Gain_cost = (C_base - C_new) / C_base
__host__ inline double compute_gain_cost(double cost_new, double cost_base) {
    if (cost_base <= 1e-15) return 0.0;
    return (cost_base - cost_new) / cost_base;
}

// Approx_qc = ||K_hyb - K_qc||_HS / ||K_qc||_HS
__host__ inline double compute_approx_qc(const double* K_hyb,
                                            const double* K_qc,
                                            int dim) {
    double diff_sq = 0.0, norm_sq = 0.0;
    for (int i = 0; i < dim * dim; ++i) {
        double d = K_hyb[i] - K_qc[i];
        diff_sq += d * d;
        norm_sq += K_qc[i] * K_qc[i];
    }
    if (norm_sq < 1e-30) return 0.0;
    return sqrt(diff_sq / norm_sq);
}

// ── Full benchmark run ──────────────────────────────────────────────────────

// Run the complete benchmark suite for a set of programs
__host__ inline Metrics run_benchmark(
    const int* predictions,
    const double* probs,
    const int* labels,
    const Program* programs,
    int N, int n_qubits,
    double acc_base = 0.5,
    double cost_base = 100.0)
{
    Metrics m;
    m.accuracy   = compute_accuracy(predictions, labels, N);
    m.nll        = compute_nll(probs, labels, N);
    m.mse        = compute_mse(probs, labels, N);
    m.brier      = compute_brier(probs, labels, N);
    m.tv         = compute_tv(probs, labels, N);
    m.wall_time  = compute_wall(programs, N);
    m.mem        = compute_mem(programs, N, n_qubits);
    m.throughput = compute_throughput(N, m.wall_time);
    m.energy     = m.wall_time;  // energy proxy ≈ total gate count
    m.gain_acc   = compute_gain_acc(m.accuracy, acc_base);
    m.gain_cost  = compute_gain_cost(m.wall_time, cost_base);
    return m;
}

// ── Pareto front computation ────────────────────────────────────────────────

__host__ inline int compute_pareto_front(
    const ParetoPoint* candidates, int N,
    ParetoPoint* front, int max_front)
{
    int front_size = 0;
    for (int i = 0; i < N && front_size < max_front; ++i) {
        bool dominated = false;
        for (int j = 0; j < N; ++j) {
            if (i != j && candidates[j].dominates(candidates[i])) {
                dominated = true;
                break;
            }
        }
        if (!dominated) {
            front[front_size++] = candidates[i];
        }
    }
    return front_size;
}

// ── Confidence interval ─────────────────────────────────────────────────────

// CI_95 = metric ± 1.96 * σ/√N  (normal approximation)
struct ConfidenceInterval {
    double lower;
    double upper;
    double center;
};

__host__ inline ConfidenceInterval compute_ci95(double metric,
                                                   double variance,
                                                   int N) {
    double se = sqrt(variance / N);
    ConfidenceInterval ci;
    ci.center = metric;
    ci.lower = metric - 1.96 * se;
    ci.upper = metric + 1.96 * se;
    return ci;
}

// Ablation: difference in metric when removing component
__host__ inline double ablation_delta(double metric_full,
                                         double metric_ablated) {
    return metric_full - metric_ablated;
}

} // namespace benchmark

// ── Result record R_{A,n} ───────────────────────────────────────────────────
struct BenchmarkResult {
    Metrics metrics;
    int     optimal_config;
    int     pareto_size;

    void print() const {
        metrics.print();
        printf("  Optimal config:  %d\n", optimal_config);
        printf("  Pareto size:     %d\n", pareto_size);
    }
};

} // namespace topcomp
