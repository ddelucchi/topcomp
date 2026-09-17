// ============================================================================
// TopComp: C ABI Surface
// ============================================================================
// Stable extern "C" interface for language bindings (Python, Rust, Julia, ...).
// All state behind opaque handles.  All functions return int error codes.
// All types are C-compatible: int, double, uint32_t, void*.
//
// Naming: tc_<module>_<verb>
//   tc_program_create, tc_program_emit_x, tc_result_prob, ...
//
// Thread safety: one handle = one thread.  No global state.
//
// Build as shared library:
//   nvcc -shared -o libtopcomp.so c_api.cu -I include/
//   (Windows: nvcc -shared -o topcomp.dll c_api.cu -I include/)
// ============================================================================
#pragma once

#include "topcomp_api.cuh"
#include "semantic_kernel.cuh"
#include "semantic_passes.cuh"

// ── Opaque handles ──────────────────────────────────────────────────────────
typedef void* tc_program;
typedef void* tc_result;
typedef void* tc_op_repr;
typedef void* tc_ctrl_op;

// ── Error codes ─────────────────────────────────────────────────────────────
#define TC_OK         0
#define TC_ERR_NULL  -1
#define TC_ERR_INDEX -2
#define TC_ERR_TYPE  -3
#define TC_ERR_FULL  -4
#define TC_ERR_FAIL  -5

#ifdef __cplusplus
extern "C" {
#endif

// ════════════════════════════════════════════════════════════════════════════
// §1  Program lifecycle
// ════════════════════════════════════════════════════════════════════════════

inline tc_program tc_program_create(int n_qubits, int N_theta, int N_rho) {
    return static_cast<tc_program>(
        topcomp::api::program_create(n_qubits, N_theta, N_rho));
}

inline void tc_program_free(tc_program prog) {
    topcomp::api::program_free(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

// ════════════════════════════════════════════════════════════════════════════
// §2  Emit operations
// ════════════════════════════════════════════════════════════════════════════

inline int tc_emit_x(tc_program prog, uint32_t u) {
    return topcomp::api::emit_x(
        static_cast<topcomp::api::ProgramHandle*>(prog), u);
}

inline int tc_emit_z(tc_program prog, uint32_t v) {
    return topcomp::api::emit_z(
        static_cast<topcomp::api::ProgramHandle*>(prog), v);
}

inline int tc_emit_h(tc_program prog, int qubit) {
    return topcomp::api::emit_h(
        static_cast<topcomp::api::ProgramHandle*>(prog), qubit);
}

inline int tc_emit_u(tc_program prog, double t) {
    return topcomp::api::emit_u(
        static_cast<topcomp::api::ProgramHandle*>(prog), t);
}

inline int tc_emit_m(tc_program prog, int m) {
    return topcomp::api::emit_m(
        static_cast<topcomp::api::ProgramHandle*>(prog), m);
}

inline int tc_emit_p(tc_program prog, uint32_t x) {
    return topcomp::api::emit_p(
        static_cast<topcomp::api::ProgramHandle*>(prog), x);
}

inline int tc_emit_measure(tc_program prog) {
    return topcomp::api::emit_measure(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

inline int tc_emit_barrier(tc_program prog) {
    return topcomp::api::emit_barrier(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

// ════════════════════════════════════════════════════════════════════════════
// §3  Register-level operations
// ════════════════════════════════════════════════════════════════════════════

inline int tc_alloc_reg(tc_program prog, int state_type) {
    return topcomp::api::program_alloc_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), state_type);
}

inline int tc_free_reg(tc_program prog, int reg) {
    return topcomp::api::program_free_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg);
}

inline int tc_emit_x_reg(tc_program prog, int reg, uint32_t u) {
    return topcomp::api::emit_x_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg, u);
}

inline int tc_emit_z_reg(tc_program prog, int reg, uint32_t v) {
    return topcomp::api::emit_z_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg, v);
}

inline int tc_emit_h_reg(tc_program prog, int reg, int qubit) {
    return topcomp::api::emit_h_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg, qubit);
}

inline int tc_emit_u_reg(tc_program prog, int reg, double t) {
    return topcomp::api::emit_u_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg, t);
}

inline int tc_emit_lift_disc(tc_program prog, int dst, int src) {
    return topcomp::api::emit_lift_disc(
        static_cast<topcomp::api::ProgramHandle*>(prog), dst, src);
}

inline int tc_emit_lift_top(tc_program prog, int dst, int src) {
    return topcomp::api::emit_lift_top(
        static_cast<topcomp::api::ProgramHandle*>(prog), dst, src);
}

inline int tc_emit_project_disc(tc_program prog, int dst, int src) {
    return topcomp::api::emit_project_disc(
        static_cast<topcomp::api::ProgramHandle*>(prog), dst, src);
}

inline int tc_emit_measure_reg(tc_program prog, int reg) {
    return topcomp::api::emit_measure_reg(
        static_cast<topcomp::api::ProgramHandle*>(prog), reg);
}

// ════════════════════════════════════════════════════════════════════════════
// §4  Compilation and execution
// ════════════════════════════════════════════════════════════════════════════

inline int tc_compile(tc_program prog) {
    return topcomp::api::compile(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

inline tc_result tc_execute(tc_program prog) {
    return static_cast<tc_result>(topcomp::api::execute(
        static_cast<topcomp::api::ProgramHandle*>(prog)));
}

// ════════════════════════════════════════════════════════════════════════════
// §5  Result extraction
// ════════════════════════════════════════════════════════════════════════════

inline double tc_result_prob(tc_result res, int meas_idx, int outcome) {
    return topcomp::api::result_prob(
        static_cast<topcomp::api::ResultHandle*>(res), meas_idx, outcome);
}

inline int tc_result_num_measurements(tc_result res) {
    return topcomp::api::result_num_measurements(
        static_cast<topcomp::api::ResultHandle*>(res));
}

inline int tc_result_meas_dim(tc_result res, int meas_idx) {
    return topcomp::api::result_meas_dim(
        static_cast<topcomp::api::ResultHandle*>(res), meas_idx);
}

inline int tc_result_time(tc_result res) {
    return topcomp::api::result_time(
        static_cast<topcomp::api::ResultHandle*>(res));
}

inline int tc_result_cost(tc_result res) {
    return topcomp::api::result_cost(
        static_cast<topcomp::api::ResultHandle*>(res));
}

inline int tc_result_cocycle_ok(tc_result res) {
    return topcomp::api::result_cocycle_ok(
        static_cast<topcomp::api::ResultHandle*>(res)) ? 1 : 0;
}

inline int tc_result_ufe_ok(tc_result res) {
    return topcomp::api::result_ufe_ok(
        static_cast<topcomp::api::ResultHandle*>(res)) ? 1 : 0;
}

inline void tc_result_free(tc_result res) {
    topcomp::api::result_free(
        static_cast<topcomp::api::ResultHandle*>(res));
}

// ════════════════════════════════════════════════════════════════════════════
// §6  IR inspection
// ════════════════════════════════════════════════════════════════════════════

inline int tc_program_op_count(tc_program prog) {
    return topcomp::api::program_op_count(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

inline int tc_program_total_cost(tc_program prog) {
    return topcomp::api::program_total_cost(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

inline int tc_program_reg_count(tc_program prog) {
    return topcomp::api::program_reg_count(
        static_cast<topcomp::api::ProgramHandle*>(prog));
}

// ════════════════════════════════════════════════════════════════════════════
// §7  Doctrine-level: ControlledOp
// ════════════════════════════════════════════════════════════════════════════

// Create a controlled operator Ĉ_{f,{U_x}}
// f_perm: array of dd ints defining the permutation
// U_data: array of dd * dt * dt * 2 doubles (real, imag pairs row-major)
// Returns opaque handle
inline tc_ctrl_op tc_ctrlop_create(const int* f_perm, const double* U_data,
                                    int dd, int dt) {
    using topcomp::doctrine::ControlledOp;
    using topcomp::measurement::DenseOp;
    using topcomp::C64;

    ControlledOp* cop = new ControlledOp();
    cop->init(dd, dt);

    cop->f_is_id = true;
    for (int x = 0; x < dd; ++x) {
        cop->f[x] = f_perm[x];
        if (cop->f[x] != x) cop->f_is_id = false;
    }

    for (int x = 0; x < dd; ++x) {
        cop->U[x] = DenseOp(dt);
        int base = x * dt * dt * 2;
        for (int i = 0; i < dt; ++i)
            for (int j = 0; j < dt; ++j) {
                int idx = base + (i * dt + j) * 2;
                cop->U[x].at(i, j) = C64(U_data[idx], U_data[idx + 1]);
            }
    }

    return static_cast<tc_ctrl_op>(cop);
}

inline void tc_ctrlop_free(tc_ctrl_op h) {
    auto* cop = static_cast<topcomp::doctrine::ControlledOp*>(h);
    if (cop) { cop->free(); delete cop; }
}

// Compose two controlled operators: result = a ∘ b
inline tc_ctrl_op tc_ctrlop_compose(tc_ctrl_op a, tc_ctrl_op b) {
    auto* ca = static_cast<topcomp::doctrine::ControlledOp*>(a);
    auto* cb = static_cast<topcomp::doctrine::ControlledOp*>(b);
    if (!ca || !cb) return nullptr;

    auto* result = new topcomp::doctrine::ControlledOp();
    *result = ca->compose(*cb);
    return static_cast<tc_ctrl_op>(result);
}

// Convert to dense matrix: out_data must have (dd*dt)² * 2 doubles
inline int tc_ctrlop_to_dense(tc_ctrl_op h, double* out_data) {
    auto* cop = static_cast<topcomp::doctrine::ControlledOp*>(h);
    if (!cop || !out_data) return TC_ERR_NULL;

    topcomp::measurement::DenseOp D = cop->to_dense();
    int N = cop->dim_disc * cop->dim_top;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int idx = (i * N + j) * 2;
            out_data[idx]     = D.at(i, j).re;
            out_data[idx + 1] = D.at(i, j).im;
        }
    return TC_OK;
}

// Is it a pure permutation?
inline int tc_ctrlop_is_perm(tc_ctrl_op h) {
    auto* cop = static_cast<topcomp::doctrine::ControlledOp*>(h);
    return (cop && cop->is_pure_perm()) ? 1 : 0;
}

// ════════════════════════════════════════════════════════════════════════════
// §8  Doctrine-level: OpRepr
// ════════════════════════════════════════════════════════════════════════════

// Create OpRepr from dense matrix (data: N*N*2 doubles)
inline tc_op_repr tc_oprepr_from_dense(const double* data,
                                        int dd, int dt, int n) {
    using topcomp::measurement::DenseOp;
    using topcomp::doctrine::OpRepr;
    using topcomp::C64;

    int N = dd * dt;
    DenseOp dense(N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int idx = (i * N + j) * 2;
            dense.at(i, j) = C64(data[idx], data[idx + 1]);
        }

    auto* rep = new OpRepr();
    *rep = OpRepr::from_dense(dense, dd, dt, n);
    return static_cast<tc_op_repr>(rep);
}

inline void tc_oprepr_free(tc_op_repr h) {
    auto* rep = static_cast<topcomp::doctrine::OpRepr*>(h);
    if (rep) { rep->free(); delete rep; }
}

// Get the current face tag
inline int tc_oprepr_face(tc_op_repr h) {
    auto* rep = static_cast<topcomp::doctrine::OpRepr*>(h);
    if (!rep) return -1;
    return static_cast<int>(rep->face);
}

// Materialize to dense: out_data must have (dd*dt)² * 2 doubles
inline int tc_oprepr_to_dense(tc_op_repr h, double* out_data) {
    auto* rep = static_cast<topcomp::doctrine::OpRepr*>(h);
    if (!rep || !out_data) return TC_ERR_NULL;

    rep->materialize_dense();
    int N = rep->dim_disc * rep->dim_top;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            int idx = (i * N + j) * 2;
            out_data[idx]     = rep->dense->at(i, j).re;
            out_data[idx + 1] = rep->dense->at(i, j).im;
        }
    return TC_OK;
}

// Multiply cost estimate
inline double tc_oprepr_multiply_cost(tc_op_repr h) {
    auto* rep = static_cast<topcomp::doctrine::OpRepr*>(h);
    if (!rep) return -1.0;
    return rep->multiply_cost();
}

// Try to promote to controlled face
inline int tc_oprepr_try_promote(tc_op_repr h) {
    auto* rep = static_cast<topcomp::doctrine::OpRepr*>(h);
    if (!rep) return TC_ERR_NULL;
    return topcomp::semantic_pass::try_promote_to_controlled(*rep) ? TC_OK : TC_ERR_FAIL;
}

// ════════════════════════════════════════════════════════════════════════════
// §9  Version / info
// ════════════════════════════════════════════════════════════════════════════

inline int tc_version_major() { return 0; }
inline int tc_version_minor() { return 4; }
inline int tc_version_patch() { return 0; }

inline const char* tc_version_string() { return "0.4.0-doctrine"; }

#ifdef __cplusplus
} // extern "C"
#endif
