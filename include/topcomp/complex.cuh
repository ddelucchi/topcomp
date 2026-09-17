// ============================================================================
// TopComp: Complex Number Type — Device/Host Dual-Use
// ============================================================================
// Implements ℂ_A := ℂ_{K_log} ⊗ A for the Mirror computation framework.
// All operations are __host__ __device__ constexpr for GPU kernel usage.
// ============================================================================
#pragma once
#include <cmath>
#include <cstdio>

namespace topcomp {

// ── Complex<T>: GPU-compatible complex number ───────────────────────────────
template <typename T = double>
struct Complex {
    T re, im;

    __host__ __device__ constexpr Complex() : re(0), im(0) {}
    __host__ __device__ constexpr Complex(T r) : re(r), im(0) {}
    __host__ __device__ constexpr Complex(T r, T i) : re(r), im(i) {}

    // Arithmetic
    __host__ __device__ constexpr Complex operator+(const Complex& o) const {
        return {re + o.re, im + o.im};
    }
    __host__ __device__ constexpr Complex operator-(const Complex& o) const {
        return {re - o.re, im - o.im};
    }
    __host__ __device__ constexpr Complex operator*(const Complex& o) const {
        return {re * o.re - im * o.im, re * o.im + im * o.re};
    }
    __host__ __device__ Complex operator/(const Complex& o) const {
        T d = o.re * o.re + o.im * o.im;
        return {(re * o.re + im * o.im) / d, (im * o.re - re * o.im) / d};
    }

    // Unary
    __host__ __device__ constexpr Complex operator-() const { return {-re, -im}; }
    __host__ __device__ constexpr Complex conj() const { return {re, -im}; }

    // In-place
    __host__ __device__ Complex& operator+=(const Complex& o) { re += o.re; im += o.im; return *this; }
    __host__ __device__ Complex& operator-=(const Complex& o) { re -= o.re; im -= o.im; return *this; }
    __host__ __device__ Complex& operator*=(const Complex& o) { *this = *this * o; return *this; }

    // Scalar
    __host__ __device__ constexpr Complex operator*(T s) const { return {re * s, im * s}; }
    __host__ __device__ constexpr Complex operator/(T s) const { return {re / s, im / s}; }
    friend __host__ __device__ constexpr Complex operator*(T s, const Complex& z) { return {s * z.re, s * z.im}; }

    // Comparison
    __host__ __device__ constexpr bool operator==(const Complex& o) const {
        return re == o.re && im == o.im;
    }
    __host__ __device__ constexpr bool operator!=(const Complex& o) const { return !(*this == o); }

    // Norm / Magnitude
    __host__ __device__ T norm2() const { return re * re + im * im; }
    __host__ __device__ T abs() const { return sqrt(norm2()); }
    __host__ __device__ T arg() const { return atan2(im, re); }

    // Print
    void print(const char* label = "") const {
        printf("%s(%.8f + %.8fi)", label, re, im);
    }
};

// ── Transcendental functions on Complex ─────────────────────────────────────

template <typename T>
__host__ __device__ Complex<T> cexp(const Complex<T>& z) {
    T e = exp(z.re);
    return {e * cos(z.im), e * sin(z.im)};
}

template <typename T>
__host__ __device__ Complex<T> clog(const Complex<T>& z) {
    return {log(z.abs()), z.arg()};
}

template <typename T>
__host__ __device__ Complex<T> cpow(const Complex<T>& z, T p) {
    if (z.re == 0 && z.im == 0) return {0, 0};
    return cexp(clog(z) * p);
}

template <typename T>
__host__ __device__ Complex<T> cpow(const Complex<T>& z, const Complex<T>& w) {
    if (z.re == 0 && z.im == 0) return {0, 0};
    return cexp(clog(z) * w);
}

// ── Type alias ──────────────────────────────────────────────────────────────
using C64 = Complex<double>;
using C32 = Complex<float>;

// ── Imaginary unit ──────────────────────────────────────────────────────────
__host__ __device__ inline constexpr C64 I() { return C64(0.0, 1.0); }

} // namespace topcomp
