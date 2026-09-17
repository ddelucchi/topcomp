// ============================================================================
// TopComp: Witt Vectors & Dual Numbers (Jet Spaces)
// ============================================================================
// Implements the algebraic tower from the reference:
//
//   Dual numbers D^(d):
//     D = K[η]/(η²)          — single infinitesimal, automatic differentiation
//     D^(d) = K[η₁,...,η_d]/(η_i η_j = 0 for all i,j)
//     f(t + η) = f(t) + f'(t)·η       (forward-mode AD)
//     f(t + Σ ηⱼ uⱼ) = f(t) + f'(t) Σ ηⱼ uⱼ  (multi-directional)
//
//   Jet space J_w^k:
//     J_w^k(f) = Σ_{n=0}^{k} f^{(n)}(w)/n! · ε^n,   ε^{k+1} = 0
//     J_w^k(fg) ≡ J_w^k(f) · J_w^k(g) mod ε^{k+1}
//
//   Truncated polynomial ring R_N:
//     R_N = K_log[t]/(t^N) = ⊕_{m=0}^{N-1} K_log · t^m
//     sl₂ action: e(t^m) = m·t^{m-1},  f(t^m) = -m·t^{m+1},  h(t^m) = -2m·t^m
//
//   Witt vectors W_k:
//     W_k = truncated polynomial algebra over K_log
//     Projection maps π_{k←ℓ}: W_ℓ → W_k  for k ≤ ℓ
//     W_∞ = lim←_k W_k  (pro-finite limit)
//
//   Completed algebra Â_{A,d}:
//     Â_{A,d} = A ⊗_{K_log} A ⊗_{K_log} D^(d)
//     Î_{A,d} = ker(Â_{A,d} → A ⊗ A)
//     UFE verified: for u=e^X, v=e^Y in exp(Î), [u,v]=1+[X,Y] mod I³
// ============================================================================
#pragma once
#include "complex.cuh"
#include "constants.cuh"
#include <cstring>
#include <cmath>

namespace topcomp {

// ── Dual number: D = K[η]/(η²) ──────────────────────────────────────────────
// Elements: a + b·η  with η² = 0
// Encodes f(t) and f'(t) simultaneously for automatic differentiation
template <typename T = double>
struct Dual {
    T val;    // f(t)          — the real part
    T eps;    // f'(t)         — the infinitesimal part

    __host__ __device__ constexpr Dual() : val(0), eps(0) {}
    __host__ __device__ constexpr Dual(T v) : val(v), eps(0) {}
    __host__ __device__ constexpr Dual(T v, T e) : val(v), eps(e) {}

    // (a + bη) + (c + dη) = (a+c) + (b+d)η
    __host__ __device__ constexpr Dual operator+(const Dual& o) const {
        return {val + o.val, eps + o.eps};
    }

    // (a + bη) - (c + dη) = (a-c) + (b-d)η
    __host__ __device__ constexpr Dual operator-(const Dual& o) const {
        return {val - o.val, eps - o.eps};
    }

    // (a + bη)(c + dη) = ac + (ad + bc)η    [η² = 0]
    __host__ __device__ constexpr Dual operator*(const Dual& o) const {
        return {val * o.val, val * o.eps + eps * o.val};
    }

    // (a + bη)/(c + dη) = a/c + (bc - ad)/c² · η
    __host__ __device__ Dual operator/(const Dual& o) const {
        T inv = T(1) / o.val;
        return {val * inv, (eps * o.val - val * o.eps) * inv * inv};
    }

    __host__ __device__ Dual operator-() const { return {-val, -eps}; }

    __host__ __device__ Dual& operator+=(const Dual& o) {
        val += o.val; eps += o.eps; return *this;
    }
    __host__ __device__ Dual& operator*=(const Dual& o) {
        *this = *this * o; return *this;
    }

    // Scalar operations
    __host__ __device__ constexpr Dual operator*(T s) const { return {val * s, eps * s}; }
    __host__ __device__ constexpr Dual operator/(T s) const { return {val / s, eps / s}; }

    void print(const char* label = "") const {
        printf("%s%.8f + %.8f·η", label, val, eps);
    }
};

// ── Dual number transcendentals (forward-mode AD) ───────────────────────────
// exp(a + bη) = e^a(1 + bη) = e^a + e^a·b·η
template <typename T>
__host__ __device__ Dual<T> dexp(const Dual<T>& z) {
    T ev = exp(z.val);
    return {ev, ev * z.eps};
}

// log(a + bη) = ln(a) + (b/a)η
template <typename T>
__host__ __device__ Dual<T> dlog(const Dual<T>& z) {
    return {log(z.val), z.eps / z.val};
}

// sin(a + bη) = sin(a) + cos(a)·b·η
template <typename T>
__host__ __device__ Dual<T> dsin(const Dual<T>& z) {
    return {sin(z.val), cos(z.val) * z.eps};
}

// cos(a + bη) = cos(a) - sin(a)·b·η
template <typename T>
__host__ __device__ Dual<T> dcos(const Dual<T>& z) {
    return {cos(z.val), -sin(z.val) * z.eps};
}

// sqrt(a + bη) = √a + b/(2√a)·η
template <typename T>
__host__ __device__ Dual<T> dsqrt(const Dual<T>& z) {
    T sv = sqrt(z.val);
    return {sv, z.eps / (T(2) * sv)};
}

// pow(a + bη, n) = a^n + n·a^{n-1}·b·η
template <typename T>
__host__ __device__ Dual<T> dpow(const Dual<T>& z, T n) {
    T pv = pow(z.val, n);
    return {pv, n * pow(z.val, n - T(1)) * z.eps};
}

// ── Complex Dual: D_ℂ = ℂ[η]/(η²) ───────────────────────────────────────────
// For the Mirror framework: c_φ extended to dual numbers
// (t₀ + η·t₁)·c_φ  →  t₀·c_φ + η·(t₁·c_φ)
struct ComplexDual {
    C64 val;    // complex value
    C64 eps;    // complex infinitesimal

    __host__ __device__ ComplexDual() : val(0, 0), eps(0, 0) {}
    __host__ __device__ ComplexDual(C64 v) : val(v), eps(0, 0) {}
    __host__ __device__ ComplexDual(C64 v, C64 e) : val(v), eps(e) {}

    __host__ __device__ ComplexDual operator+(const ComplexDual& o) const {
        return {val + o.val, eps + o.eps};
    }
    __host__ __device__ ComplexDual operator-(const ComplexDual& o) const {
        return {val - o.val, eps - o.eps};
    }
    __host__ __device__ ComplexDual operator*(const ComplexDual& o) const {
        return {val * o.val, val * o.eps + eps * o.val};
    }

    // Scalar
    __host__ __device__ ComplexDual operator*(double s) const {
        return {val * s, eps * s};
    }
    __host__ __device__ ComplexDual operator*(C64 s) const {
        return {val * s, eps * s};
    }
};

// exp(z₀ + η·ż) = e^{z₀}(1 + η·ż)
__host__ __device__ inline ComplexDual cdexp(const ComplexDual& z) {
    C64 ev = cexp(z.val);
    return {ev, ev * z.eps};
}

// ── Multi-variable dual: D^(d) ──────────────────────────────────────────────
// Elements: u₀ + Σⱼ ηⱼ uⱼ  with ηᵢηⱼ = 0 for all i,j
// Used for d-directional automatic differentiation
static constexpr int MAX_DUAL_VARS = 8;

struct MultiDual {
    double val;                       // u₀
    double eps[MAX_DUAL_VARS];        // u₁,...,u_d
    int    d;                         // number of infinitesimal directions

    __host__ __device__ MultiDual() : val(0), d(0) {
        for (int i = 0; i < MAX_DUAL_VARS; ++i) eps[i] = 0;
    }
    __host__ __device__ MultiDual(double v, int dim) : val(v), d(dim) {
        for (int i = 0; i < MAX_DUAL_VARS; ++i) eps[i] = 0;
    }

    // Set i-th infinitesimal component
    __host__ __device__ void set_dir(int i, double e) {
        if (i >= 0 && i < d) eps[i] = e;
    }

    // (a + Σ aⱼηⱼ)(b + Σ bⱼηⱼ) = ab + Σ(a·bⱼ + aⱼ·b)ηⱼ
    __host__ __device__ MultiDual operator*(const MultiDual& o) const {
        MultiDual result(val * o.val, d);
        for (int j = 0; j < d; ++j) {
            result.eps[j] = val * o.eps[j] + eps[j] * o.val;
        }
        return result;
    }

    __host__ __device__ MultiDual operator+(const MultiDual& o) const {
        MultiDual result(val + o.val, d);
        for (int j = 0; j < d; ++j) result.eps[j] = eps[j] + o.eps[j];
        return result;
    }

    __host__ __device__ MultiDual operator-(const MultiDual& o) const {
        MultiDual result(val - o.val, d);
        for (int j = 0; j < d; ++j) result.eps[j] = eps[j] - o.eps[j];
        return result;
    }
};

// exp on MultiDual: e^{u₀}(1 + Σ uⱼηⱼ)
__host__ __device__ inline MultiDual mdexp(const MultiDual& z) {
    double ev = exp(z.val);
    MultiDual result(ev, z.d);
    for (int j = 0; j < z.d; ++j) result.eps[j] = ev * z.eps[j];
    return result;
}

// ── Jet space J_w^k ─────────────────────────────────────────────────────────
// J_w^k(f) = Σ_{n=0}^{k} (f^(n)(w)/n!) ε^n,  with ε^{k+1} = 0
// This is the k-th order Taylor jet at w.
// Key property: J_w^k(fg) ≡ J_w^k(f)·J_w^k(g) mod ε^{k+1}
static constexpr int MAX_JET_ORDER = 32;

struct Jet {
    double coeffs[MAX_JET_ORDER];  // coeffs[n] = f^(n)(w) / n!
    int    order;                  // k
    double w;                      // base point

    __host__ __device__ Jet() : order(0), w(0) {
        for (int i = 0; i < MAX_JET_ORDER; ++i) coeffs[i] = 0;
    }
    __host__ __device__ Jet(int k, double base) : order(k), w(base) {
        for (int i = 0; i < MAX_JET_ORDER; ++i) coeffs[i] = 0;
    }

    // Set coefficient: f^(n)(w)/n!
    __host__ __device__ void set(int n, double val) {
        if (n >= 0 && n <= order) coeffs[n] = val;
    }

    // Jet multiply: (Σ aₙ εⁿ)(Σ bₘ εᵐ) = Σ_{n=0}^{k} (Σ_{j=0}^{n} aⱼ b_{n-j}) εⁿ
    __host__ __device__ Jet operator*(const Jet& o) const {
        Jet result(order, w);
        for (int n = 0; n <= order; ++n) {
            double sum = 0;
            for (int j = 0; j <= n; ++j) {
                sum += coeffs[j] * o.coeffs[n - j];
            }
            result.coeffs[n] = sum;
        }
        return result;
    }

    __host__ __device__ Jet operator+(const Jet& o) const {
        Jet result(order, w);
        for (int n = 0; n <= order; ++n)
            result.coeffs[n] = coeffs[n] + o.coeffs[n];
        return result;
    }

    __host__ __device__ Jet operator-(const Jet& o) const {
        Jet result(order, w);
        for (int n = 0; n <= order; ++n)
            result.coeffs[n] = coeffs[n] - o.coeffs[n];
        return result;
    }
};

// Build jet from polynomial: f(t) = Σ aₘ tᵐ  →  J_w^k(f)
__host__ __device__ inline Jet jet_from_poly(const double* poly_coeffs, int degree,
                                               double w, int jet_order) {
    Jet J(jet_order, w);
    // f^(n)(w) = Σ_{m≥n} m!/(m-n)! · aₘ · w^{m-n}
    for (int n = 0; n <= jet_order && n <= degree; ++n) {
        double sum = 0;
        double w_power = 1.0;
        for (int m = n; m <= degree; ++m) {
            // m!/(m-n)! / n! = C(m,n)
            double binom = 1.0;
            for (int i = 0; i < n; ++i) {
                binom *= (m - i);
                binom /= (i + 1);
            }
            sum += binom * poly_coeffs[m] * w_power;
            w_power *= w;
        }
        J.coeffs[n] = sum;
    }
    return J;
}

// ── Truncated polynomial ring R_N = K_log[t]/(t^N) ──────────────────────────
// Weight decomposition: R_N = ⊕_{m=0}^{N-1} K_log · t^m
// sl₂ representation on R_N:
//   e = D = d/dt:    e(t^m) = m·t^{m-1}
//   f = -K = -t²d/dt: f(t^m) = -m·t^{m+1}  (or 0 if m+1 ≥ N)
//   h = -2E = -2t·d/dt: h(t^m) = -2m·t^m
struct TruncPoly {
    static constexpr int MAX_N = 64;
    double coeffs[MAX_N];  // coeffs[m] = coefficient of t^m
    int    N;              // truncation: t^N = 0

    __host__ __device__ TruncPoly() : N(32) {
        for (int i = 0; i < MAX_N; ++i) coeffs[i] = 0;
    }
    __host__ __device__ TruncPoly(int n) : N(n) {
        for (int i = 0; i < MAX_N; ++i) coeffs[i] = 0;
    }

    // Basis monomial t^m
    __host__ __device__ static TruncPoly monomial(int m, int n) {
        TruncPoly p(n);
        if (m >= 0 && m < n) p.coeffs[m] = 1.0;
        return p;
    }

    // p(t) + q(t) mod t^N
    __host__ __device__ TruncPoly operator+(const TruncPoly& o) const {
        TruncPoly r(N);
        for (int i = 0; i < N; ++i) r.coeffs[i] = coeffs[i] + o.coeffs[i];
        return r;
    }

    __host__ __device__ TruncPoly operator-(const TruncPoly& o) const {
        TruncPoly r(N);
        for (int i = 0; i < N; ++i) r.coeffs[i] = coeffs[i] - o.coeffs[i];
        return r;
    }

    // p(t) · q(t) mod t^N  (truncated convolution)
    __host__ __device__ TruncPoly operator*(const TruncPoly& o) const {
        TruncPoly r(N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i + j < N) {
                    r.coeffs[i + j] += coeffs[i] * o.coeffs[j];
                }
            }
        }
        return r;
    }

    __host__ __device__ TruncPoly operator*(double s) const {
        TruncPoly r(N);
        for (int i = 0; i < N; ++i) r.coeffs[i] = coeffs[i] * s;
        return r;
    }

    // ── sl₂ action ──────────────────────────────────────────────────────────

    // e(p) = D(p) = d/dt p:  e(t^m) = m·t^{m-1}
    __host__ __device__ TruncPoly sl2_e() const {
        TruncPoly r(N);
        for (int m = 1; m < N; ++m) {
            r.coeffs[m - 1] = m * coeffs[m];
        }
        return r;
    }

    // f(p) = -K(p) = -t²·d/dt p:  f(t^m) = -m·t^{m+1}
    __host__ __device__ TruncPoly sl2_f() const {
        TruncPoly r(N);
        for (int m = 0; m < N; ++m) {
            if (m + 1 < N) {
                r.coeffs[m + 1] = -static_cast<double>(m) * coeffs[m];
            }
        }
        return r;
    }

    // h(p) = -2E(p) = -2t·d/dt p:  h(t^m) = -2m·t^m
    __host__ __device__ TruncPoly sl2_h() const {
        TruncPoly r(N);
        for (int m = 0; m < N; ++m) {
            r.coeffs[m] = -2.0 * m * coeffs[m];
        }
        return r;
    }

    // Higher powers: e^r(t^m) = m!/(m-r)! · t^{m-r}
    __host__ __device__ TruncPoly sl2_e_power(int r) const {
        TruncPoly result = *this;
        for (int i = 0; i < r; ++i) {
            result = result.sl2_e();
        }
        return result;
    }

    // exp(h·D)(p)(t) = p(t+h)  — translation operator
    __host__ __device__ TruncPoly translate(double h) const {
        TruncPoly r(N);
        // p(t+h) = Σ_m a_m (t+h)^m  — use binomial expansion
        for (int m = 0; m < N; ++m) {
            if (coeffs[m] == 0) continue;
            // (t+h)^m = Σ_{k=0}^{m} C(m,k) h^k t^{m-k}
            double h_pow = 1.0;
            double binom = 1.0;
            for (int k = 0; k <= m; ++k) {
                if (m - k < N) {
                    r.coeffs[m - k] += coeffs[m] * binom * h_pow;
                }
                h_pow *= h;
                if (k + 1 <= m) {
                    binom *= static_cast<double>(m - k) / static_cast<double>(k + 1);
                }
            }
        }
        return r;
    }

    // Evaluate p(t) at a value
    __host__ __device__ double eval(double t) const {
        double result = 0;
        double t_pow = 1.0;
        for (int m = 0; m < N; ++m) {
            result += coeffs[m] * t_pow;
            t_pow *= t;
        }
        return result;
    }

    // Scale: S_λ(p)(t) = p(λt)
    __host__ __device__ TruncPoly scale(double lambda) const {
        TruncPoly r(N);
        double lam_pow = 1.0;
        for (int m = 0; m < N; ++m) {
            r.coeffs[m] = coeffs[m] * lam_pow;
            lam_pow *= lambda;
        }
        return r;
    }

    // Inversion: J(p)(t) = p(1/t)  — only meaningful for Laurent-type
    // For polynomials: reverses coefficient order
    __host__ __device__ TruncPoly reverse() const {
        TruncPoly r(N);
        for (int m = 0; m < N; ++m) {
            r.coeffs[N - 1 - m] = coeffs[m];
        }
        return r;
    }

    // Commutator [p, q] in the polynomial algebra (only non-trivial for matrix-valued)
    // For scalar polynomials: always 0
    __host__ __device__ double commutator_norm(const TruncPoly& o) const {
        TruncPoly pq = *this * o;
        TruncPoly qp = o * (*this);
        TruncPoly comm = pq - qp;
        double n = 0;
        for (int i = 0; i < N; ++i) n += comm.coeffs[i] * comm.coeffs[i];
        return sqrt(n);
    }

    // Weight space decomposition check: h(t^m) = -2m·t^m
    __host__ __device__ int weight(int m) const {
        return -2 * m;
    }

    void print(const char* label = "") const {
        printf("%s[", label);
        bool first = true;
        for (int m = 0; m < N; ++m) {
            if (fabs(coeffs[m]) > 1e-15) {
                if (!first) printf(" + ");
                printf("%.4f·t^%d", coeffs[m], m);
                first = false;
            }
        }
        printf("]");
    }
};

// ── Commutation verification for R_N sl₂ ────────────────────────────────────
// [h, e] = 2e,  [h, f] = -2f,  [e, f] = h
namespace sl2_verify {

// Apply [h, e](p) = h(e(p)) - e(h(p)) and check = 2·e(p)
__host__ inline bool check_he(const TruncPoly& p, double tol = 1e-10) {
    TruncPoly ep = p.sl2_e();
    TruncPoly hp = p.sl2_h();
    TruncPoly hep = ep.sl2_h();
    TruncPoly ehp = hp.sl2_e();
    TruncPoly comm = hep - ehp;
    TruncPoly expected = ep * 2.0;
    double err = 0;
    for (int i = 0; i < p.N; ++i) {
        double d = comm.coeffs[i] - expected.coeffs[i];
        err += d * d;
    }
    return sqrt(err) < tol;
}

// [h, f](p) = -2f(p)
__host__ inline bool check_hf(const TruncPoly& p, double tol = 1e-10) {
    TruncPoly fp = p.sl2_f();
    TruncPoly hp = p.sl2_h();
    TruncPoly hfp = fp.sl2_h();
    TruncPoly fhp = hp.sl2_f();
    TruncPoly comm = hfp - fhp;
    TruncPoly expected = fp * (-2.0);
    double err = 0;
    for (int i = 0; i < p.N; ++i) {
        double d = comm.coeffs[i] - expected.coeffs[i];
        err += d * d;
    }
    return sqrt(err) < tol;
}

// [e, f](p) = h(p)
__host__ inline bool check_ef(const TruncPoly& p, double tol = 1e-10) {
    TruncPoly ep = p.sl2_e();
    TruncPoly fp = p.sl2_f();
    TruncPoly efp = fp.sl2_e();
    TruncPoly fep = ep.sl2_f();
    TruncPoly comm = efp - fep;
    TruncPoly expected = p.sl2_h();
    double err = 0;
    for (int i = 0; i < p.N; ++i) {
        double d = comm.coeffs[i] - expected.coeffs[i];
        err += d * d;
    }
    return sqrt(err) < tol;
}

} // namespace sl2_verify

// ── Golden-flow dual number extension ───────────────────────────────────────
// t_η = t₀ + η·t₁  →  A_{t_η}·w = w + t_η·c_φ = (w + t₀·c_φ) + η·(t₁·c_φ)
// e^{t_η·c_φ} = e^{t₀·c_φ}(1 + η·t₁·c_φ)
// This captures the tangent vector to the golden flow.
struct DualFlow {
    double t0;   // base flow parameter
    double t1;   // infinitesimal perturbation

    __host__ __device__ DualFlow() : t0(0), t1(0) {}
    __host__ __device__ DualFlow(double base, double pert) : t0(base), t1(pert) {}

    // A_{t_η} · w = (w + t₀·c_φ) + η·(t₁·c_φ)
    __host__ __device__ ComplexDual act_w(const C64& w) const {
        C64 cp = constants::c_phi();
        return ComplexDual(w + cp * t0, cp * t1);
    }

    // e^{t_η·c_φ} · z = e^{t₀·c_φ}(1 + η·t₁·c_φ) · z
    __host__ __device__ ComplexDual act_z(const C64& z) const {
        C64 cp = constants::c_phi();
        C64 base = cexp(cp * t0) * z;
        C64 deriv = cexp(cp * t0) * (cp * t1) * z;
        return ComplexDual(base, deriv);
    }

    // Compose flows: (t₀ + η·t₁) + (s₀ + η·s₁)
    __host__ __device__ DualFlow compose(const DualFlow& o) const {
        return {t0 + o.t0, t1 + o.t1};
    }

    // Special case from the paper: e^{(4+η)c_φ} = φ⁴(1 + η·c_φ)
    __host__ __device__ static ComplexDual four_plus_eta_action(const C64& z) {
        C64 cp = constants::c_phi();
        double phi4 = constants::PHI * constants::PHI * constants::PHI * constants::PHI;
        return ComplexDual(z * phi4, z * phi4 * cp);
    }
};

} // namespace topcomp
