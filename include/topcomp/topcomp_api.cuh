// ============================================================================
// TopComp Layer 5: Host Language Interface
// ============================================================================
// Stable API surface for external language bindings.
//
// Design principles:
//   - Handle-based: all state is opaque to the caller
//   - Error-coded: all functions return TopcompError
//   - Two usage modes:
//       Simple:   single register, auto-allocated hybrid state
//       Advanced: multi-register with manual sector management
//   - C-compatible types: int, double, uint32_t, void*
//
// Usage (simple mode):
//   auto* prog = topcomp::api::program_create(2, 16, 8);
//   topcomp::api::emit_h(prog, 0);
//   topcomp::api::emit_z(prog, BitVec(1, 2));
//   topcomp::api::emit_h(prog, 0);
//   topcomp::api::emit_measure(prog);
//   auto* res = topcomp::api::execute(prog);
//   double p0 = topcomp::api::result_prob(res, 0, 0);
//   topcomp::api::result_free(res);
//   topcomp::api::program_free(prog);
//
// Usage (advanced mode):
//   auto* prog = topcomp::api::program_create(2, 16, 8);
//   int r_disc = topcomp::api::program_alloc_reg(prog, 0);  // DISC
//   int r_hyb  = topcomp::api::program_alloc_reg(prog, 2);  // HYB
//   topcomp::api::emit_x_reg(prog, r_disc, 1);
//   topcomp::api::emit_lift_disc(prog, r_hyb, r_disc);
//   topcomp::api::emit_u_reg(prog, r_hyb, 1.0);
//   topcomp::api::emit_measure_reg(prog, r_hyb);
//   auto* res = topcomp::api::execute(prog);
//   ...
//
// Future: compile to extern "C" in a .cu file for shared library export
// ============================================================================
#pragma once
#include "pipeline.cuh"

namespace topcomp {
namespace api {

using contract::StateType;
using contract::MachineSpec;

// ── Error codes ─────────────────────────────────────────────────────────────

enum TopcompError : int {
    TOPCOMP_OK         =  0,
    TOPCOMP_ERR_NULL   = -1,   // null handle
    TOPCOMP_ERR_INDEX  = -2,   // index out of range
    TOPCOMP_ERR_TYPE   = -3,   // type mismatch
    TOPCOMP_ERR_FULL   = -4,   // capacity exceeded
};

// ── Program handle ──────────────────────────────────────────────────────────

struct ProgramHandle {
    platform::PlatformIR ir;
    MachineSpec          spec;
    int                  primary_reg;  // auto-allocated hybrid register
    bool                 compiled;

    ProgramHandle(int nq, int nth, int nrh)
        : ir(MachineSpec(nq, nth, nrh)),
          spec(nq, nth, nrh),
          primary_reg(-1),
          compiled(false) {
        // Auto-allocate primary hybrid register
        primary_reg = ir.alloc_reg(StateType::HYB);
        ir.emit(platform::PlatformOp::alloc(primary_reg, StateType::HYB));
    }
};

// ── Result handle ───────────────────────────────────────────────────────────

struct ResultHandle {
    runtime::MeasRecord measurements;
    int    time_steps;
    int    total_cost;
    bool   cocycle_ok;
    bool   ufe_ok;
    bool   valid;

    ResultHandle() : time_steps(0), total_cost(0),
                     cocycle_ok(true), ufe_ok(true), valid(false) {}
};

// ── Program lifecycle ───────────────────────────────────────────────────────

inline ProgramHandle* program_create(int n_qubits,
                                      int N_theta = 16,
                                      int N_rho = 8) {
    return new ProgramHandle(n_qubits, N_theta, N_rho);
}

inline void program_free(ProgramHandle* prog) {
    delete prog;
}

// ── Register management (advanced mode) ─────────────────────────────────────

// state_type: 0=DISC, 1=TOP, 2=HYB
inline int program_alloc_reg(ProgramHandle* prog, int state_type) {
    if (!prog) return -1;
    StateType t = static_cast<StateType>(state_type);
    int reg = prog->ir.alloc_reg(t);
    prog->ir.emit(platform::PlatformOp::alloc(reg, t));
    return reg;
}

inline TopcompError program_free_reg(ProgramHandle* prog, int reg) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::free_reg(reg));
    prog->ir.free_reg(reg);
    return TOPCOMP_OK;
}

// ── Simple mode: emit ops on primary register ───────────────────────────────

inline TopcompError emit_x(ProgramHandle* prog, uint32_t u) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_x(
        prog->primary_reg, BitVec(u, prog->spec.n_qubits)));
    return TOPCOMP_OK;
}

inline TopcompError emit_z(ProgramHandle* prog, uint32_t v) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_z(
        prog->primary_reg, BitVec(v, prog->spec.n_qubits)));
    return TOPCOMP_OK;
}

inline TopcompError emit_h(ProgramHandle* prog, int qubit) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_h(
        prog->primary_reg, qubit));
    return TOPCOMP_OK;
}

inline TopcompError emit_u(ProgramHandle* prog, double t) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_u(
        prog->primary_reg, t));
    return TOPCOMP_OK;
}

inline TopcompError emit_m(ProgramHandle* prog, int m) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_m(
        prog->primary_reg, m));
    return TOPCOMP_OK;
}

inline TopcompError emit_p(ProgramHandle* prog, uint32_t x) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_p(
        prog->primary_reg, x));
    return TOPCOMP_OK;
}

inline TopcompError emit_w(ProgramHandle* prog,
                            uint32_t u, uint32_t v,
                            MirElement g) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_w(
        prog->primary_reg,
        BitVec(u, prog->spec.n_qubits),
        BitVec(v, prog->spec.n_qubits), g));
    return TOPCOMP_OK;
}

inline TopcompError emit_measure(ProgramHandle* prog) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::measure_disc(prog->primary_reg));
    return TOPCOMP_OK;
}

inline TopcompError emit_measure_joint(ProgramHandle* prog) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::measure_joint(prog->primary_reg));
    return TOPCOMP_OK;
}

inline TopcompError emit_check_cocycle(ProgramHandle* prog) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::check_cocycle());
    return TOPCOMP_OK;
}

inline TopcompError emit_check_ufe(ProgramHandle* prog) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::check_ufe());
    return TOPCOMP_OK;
}

// ── Advanced mode: emit ops on specified register ───────────────────────────

inline TopcompError emit_x_reg(ProgramHandle* prog, int reg, uint32_t u) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_x(
        reg, BitVec(u, prog->spec.n_qubits)));
    return TOPCOMP_OK;
}

inline TopcompError emit_z_reg(ProgramHandle* prog, int reg, uint32_t v) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_z(
        reg, BitVec(v, prog->spec.n_qubits)));
    return TOPCOMP_OK;
}

inline TopcompError emit_h_reg(ProgramHandle* prog, int reg, int qubit) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_h(reg, qubit));
    return TOPCOMP_OK;
}

inline TopcompError emit_u_reg(ProgramHandle* prog, int reg, double t) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_u(reg, t));
    return TOPCOMP_OK;
}

inline TopcompError emit_m_reg(ProgramHandle* prog, int reg, int m) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::apply_m(reg, m));
    return TOPCOMP_OK;
}

inline TopcompError emit_measure_reg(ProgramHandle* prog, int reg) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::measure_disc(reg));
    return TOPCOMP_OK;
}

// ── Lifting / projection (advanced mode) ────────────────────────────────────

inline TopcompError emit_lift_disc(ProgramHandle* prog, int dst, int src) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::lift_disc_to_hyb(dst, src));
    return TOPCOMP_OK;
}

inline TopcompError emit_lift_top(ProgramHandle* prog, int dst, int src) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::lift_top_to_hyb(dst, src));
    return TOPCOMP_OK;
}

inline TopcompError emit_project_disc(ProgramHandle* prog, int dst, int src) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::project_disc(dst, src));
    return TOPCOMP_OK;
}

inline TopcompError emit_project_top(ProgramHandle* prog,
                                      int dst, int src, uint32_t x = 0) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::project_top(dst, src, x));
    return TOPCOMP_OK;
}

// ── Sector boundaries ───────────────────────────────────────────────────────

inline TopcompError emit_sector(ProgramHandle* prog, int sector_id) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::emit_sector(sector_id));
    return TOPCOMP_OK;
}

inline TopcompError emit_barrier(ProgramHandle* prog) {
    if (!prog) return TOPCOMP_ERR_NULL;
    prog->ir.emit(platform::PlatformOp::barrier());
    return TOPCOMP_OK;
}

// ── Compilation and execution ───────────────────────────────────────────────

// Compile the program (type check, auto-lift, optimize)
inline TopcompError compile(ProgramHandle* prog, bool auto_lift = true) {
    if (!prog) return TOPCOMP_ERR_NULL;
    pipeline::CompilerPipeline pipe(prog->spec, auto_lift);
    pipe.compile(prog->ir);
    prog->compiled = true;
    return TOPCOMP_OK;
}

// Execute a compiled program, returning a result handle
inline ResultHandle* execute(ProgramHandle* prog, bool auto_compile = true) {
    if (!prog) return nullptr;

    if (!prog->compiled && auto_compile) {
        compile(prog);
    }

    runtime::PlatformVM vm(prog->spec);
    vm.run(prog->ir);

    ResultHandle* res = new ResultHandle();
    res->measurements = vm.measurements;
    res->time_steps = vm.time();
    res->total_cost = prog->ir.total_cost();
    res->cocycle_ok = vm.cocycle_verified();
    res->ufe_ok = vm.ufe_verified();
    res->valid = true;
    return res;
}

// ── Result extraction ───────────────────────────────────────────────────────

// Get Born probability for measurement_idx, outcome
inline double result_prob(const ResultHandle* res,
                           int measurement_idx, int outcome) {
    if (!res || !res->valid) return 0.0;
    if (measurement_idx < 0 ||
        measurement_idx >= res->measurements.count) return 0.0;
    const runtime::MeasRecord::DiscMeas& dm =
        res->measurements.disc_results[measurement_idx];
    if (!dm.valid || outcome < 0 || outcome >= dm.dim) return 0.0;
    return dm.probs[outcome];
}

// Number of measurements in the result
inline int result_num_measurements(const ResultHandle* res) {
    if (!res || !res->valid) return 0;
    return res->measurements.count;
}

// Dimension of a particular measurement
inline int result_meas_dim(const ResultHandle* res, int measurement_idx) {
    if (!res || !res->valid) return 0;
    if (measurement_idx < 0 ||
        measurement_idx >= res->measurements.count) return 0;
    return res->measurements.disc_results[measurement_idx].dim;
}

// Time steps executed
inline int result_time(const ResultHandle* res) {
    if (!res) return 0;
    return res->time_steps;
}

// Total cost
inline int result_cost(const ResultHandle* res) {
    if (!res) return 0;
    return res->total_cost;
}

// Cocycle verification result
inline bool result_cocycle_ok(const ResultHandle* res) {
    if (!res) return false;
    return res->cocycle_ok;
}

// UFE verification result
inline bool result_ufe_ok(const ResultHandle* res) {
    if (!res) return false;
    return res->ufe_ok;
}

// Free a result handle
inline void result_free(ResultHandle* res) {
    delete res;
}

// ── IR inspection (for debugging / tooling) ─────────────────────────────────

inline int program_op_count(const ProgramHandle* prog) {
    if (!prog) return 0;
    return prog->ir.count;
}

inline int program_total_cost(const ProgramHandle* prog) {
    if (!prog) return 0;
    return prog->ir.total_cost();
}

inline int program_reg_count(const ProgramHandle* prog) {
    if (!prog) return 0;
    return prog->ir.n_regs;
}

} // namespace api
} // namespace topcomp
