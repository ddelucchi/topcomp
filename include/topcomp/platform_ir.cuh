// ============================================================================
// TopComp Layer 2: Platform IR — Hybrid State-Aware Intermediate Representation
// ============================================================================
// Extends the existing MirIR with state-type awareness:
//
//   Every operation carries its input/output sector type:
//     DISC, TOP, or HYB
//
//   Operation classes:
//     LIFT_DISC_TO_HYB   — embed discrete into hybrid (ι_disc)
//     LIFT_TOP_TO_HYB    — embed topological into hybrid (ι_top)
//     PROJECT_DISC       — extract discrete probabilities (π_disc)
//     PROJECT_TOP        — extract topological slice (π_top)
//     MEASURE_DISC       — Born measurement over I_n
//     MEASURE_JOINT      — joint disc+top measurement
//     CHECK_COCYCLE      — verify δΩ = 0
//     CHECK_UFE          — verify F = Ω (UFE = 0)
//     EMIT_SECTOR        — mark sector boundary
//
//   Structure-aware opcodes:
//     DECLARE_ENTITY     — add entity to structure graph
//     DECLARE_RELATION   — add relation between entities
//     DECLARE_BOUNDARY   — add boundary to structure graph
//     DECLARE_OBSERVABLE — add observable on a boundary
//     MEASURE_STRUCT     — structural measurement → sector report
//     RECONSTRUCT        — sector weights → annotated structure
//     EMIT_REPORT        — emit sector report as output
//
//   Virtual register file:
//     Each register has a tracked StateType
//     Type checker verifies sector compatibility
//     Auto-lift pass inserts lifts in implicit mode
//
//   Two modes:
//     Explicit: expert writes doctrine directly (manual lifts)
//     Implicit: compiler auto-lifts when sector ops cross boundaries
//
//   Import path: legacy MirIR → PlatformIR (preserves semantics)
//   Lower path:  PlatformIR → HybridProgram (gate-level)
// ============================================================================
#pragma once
#include "state_contract.cuh"
#include "compiler.cuh"

namespace topcomp {
namespace platform {

using contract::StateType;
using contract::StateDescriptor;
using contract::MachineSpec;
using contract::CostModel;

// ── Platform opcode set ─────────────────────────────────────────────────────

enum class PlatformOpCode : int {
    // State management
    ALLOC,              // Allocate state register of given type
    FREE,               // Free state register

    // Lifting / Lowering
    LIFT_DISC_TO_HYB,   // ι_disc: embed discrete into hybrid
    LIFT_TOP_TO_HYB,    // ι_top: embed topological into hybrid
    PROJECT_DISC,       // π_disc: project hybrid → disc probabilities
    PROJECT_TOP,        // π_top: project hybrid → topo slice at x

    // Discrete sector ops
    APPLY_X,            // X_u shift
    APPLY_Z,            // Z_v phase
    APPLY_H,            // H_q Hadamard on qubit q

    // Topological sector ops
    APPLY_U,            // U_t golden-ratio flow
    APPLY_M,            // M_m modular multiplier

    // Hybrid ops
    APPLY_W,            // W_{u,v;g} full hybrid Weyl
    APPLY_P,            // P_x projector

    // Measurement
    MEASURE_DISC,       // Born measurement over discrete sector
    MEASURE_JOINT,      // Joint disc+top measurement

    // Verification
    CHECK_COCYCLE,      // Verify δΩ = 0
    CHECK_UFE,          // Verify F = Ω (UFE = 0)

    // Sector boundary
    EMIT_SECTOR,        // Mark sector boundary (metadata)
    BARRIER,            // Synchronization point

    // Control flow
    LABEL,
    BRANCH,
    REPEAT,

    // Structure-aware ops (universal structure machine)
    DECLARE_ENTITY,     // Add entity to structure graph
    DECLARE_RELATION,   // Add relation between entities
    DECLARE_BOUNDARY,   // Add boundary to structure graph
    DECLARE_OBSERVABLE, // Add observable on a boundary
    MEASURE_STRUCT,     // Structural measurement → sector report
    RECONSTRUCT,        // Sector weights → annotated structure
    EMIT_REPORT,        // Emit sector report as output

    // No-op
    NOP,
};

// ── Platform operation ──────────────────────────────────────────────────────

struct PlatformOp {
    PlatformOpCode opcode;

    // Sector type annotations
    StateType input_type;     // expected state type of operand register
    StateType output_type;    // produced state type

    // Register operands
    int reg_dst;              // destination register
    int reg_src;              // source register (for lifts/projects)

    // Gate operands
    BitVec     bv1;           // u for X, W
    BitVec     bv2;           // v for Z, W
    MirElement mir;           // g for W, mirror actions
    double     param;         // t for U_t, etc.
    int        iparam;        // m for M_m, qubit for H, sector id
    uint32_t   target;        // x for P_x, branch target

    int source_line;

    __host__ __device__ PlatformOp()
        : opcode(PlatformOpCode::NOP),
          input_type(StateType::HYB), output_type(StateType::HYB),
          reg_dst(0), reg_src(0),
          bv1(0,1), bv2(0,1), mir(),
          param(0), iparam(0), target(0), source_line(0) {}

    // ── Named constructors ──────────────────────────────────────────────────

    __host__ __device__ static PlatformOp alloc(int reg, StateType type) {
        PlatformOp op;
        op.opcode = PlatformOpCode::ALLOC;
        op.reg_dst = reg;
        op.output_type = type;
        return op;
    }

    __host__ __device__ static PlatformOp free_reg(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::FREE;
        op.reg_dst = reg;
        return op;
    }

    __host__ __device__ static PlatformOp lift_disc_to_hyb(int dst, int src) {
        PlatformOp op;
        op.opcode = PlatformOpCode::LIFT_DISC_TO_HYB;
        op.reg_dst = dst;
        op.reg_src = src;
        op.input_type = StateType::DISC;
        op.output_type = StateType::HYB;
        return op;
    }

    __host__ __device__ static PlatformOp lift_top_to_hyb(int dst, int src) {
        PlatformOp op;
        op.opcode = PlatformOpCode::LIFT_TOP_TO_HYB;
        op.reg_dst = dst;
        op.reg_src = src;
        op.input_type = StateType::TOP;
        op.output_type = StateType::HYB;
        return op;
    }

    __host__ __device__ static PlatformOp project_disc(int dst, int src) {
        PlatformOp op;
        op.opcode = PlatformOpCode::PROJECT_DISC;
        op.reg_dst = dst;
        op.reg_src = src;
        op.input_type = StateType::HYB;
        op.output_type = StateType::DISC;
        return op;
    }

    __host__ __device__ static PlatformOp project_top(int dst, int src,
                                                       uint32_t x = 0) {
        PlatformOp op;
        op.opcode = PlatformOpCode::PROJECT_TOP;
        op.reg_dst = dst;
        op.reg_src = src;
        op.target = x;
        op.input_type = StateType::HYB;
        op.output_type = StateType::TOP;
        return op;
    }

    __host__ __device__ static PlatformOp apply_x(int reg, BitVec u) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_X;
        op.reg_dst = reg;
        op.bv1 = u;
        return op;
    }

    __host__ __device__ static PlatformOp apply_z(int reg, BitVec v) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_Z;
        op.reg_dst = reg;
        op.bv1 = v;
        return op;
    }

    __host__ __device__ static PlatformOp apply_h(int reg, int qubit) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_H;
        op.reg_dst = reg;
        op.iparam = qubit;
        return op;
    }

    __host__ __device__ static PlatformOp apply_u(int reg, double t) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_U;
        op.reg_dst = reg;
        op.param = t;
        return op;
    }

    __host__ __device__ static PlatformOp apply_m(int reg, int m) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_M;
        op.reg_dst = reg;
        op.iparam = m;
        return op;
    }

    __host__ __device__ static PlatformOp apply_w(int reg, BitVec u,
                                                    BitVec v, MirElement g) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_W;
        op.reg_dst = reg;
        op.bv1 = u;
        op.bv2 = v;
        op.mir = g;
        return op;
    }

    __host__ __device__ static PlatformOp apply_p(int reg, uint32_t x) {
        PlatformOp op;
        op.opcode = PlatformOpCode::APPLY_P;
        op.reg_dst = reg;
        op.target = x;
        return op;
    }

    __host__ __device__ static PlatformOp measure_disc(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::MEASURE_DISC;
        op.reg_dst = reg;
        return op;
    }

    __host__ __device__ static PlatformOp measure_joint(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::MEASURE_JOINT;
        op.reg_dst = reg;
        return op;
    }

    __host__ __device__ static PlatformOp check_cocycle() {
        PlatformOp op;
        op.opcode = PlatformOpCode::CHECK_COCYCLE;
        return op;
    }

    __host__ __device__ static PlatformOp check_ufe() {
        PlatformOp op;
        op.opcode = PlatformOpCode::CHECK_UFE;
        return op;
    }

    __host__ __device__ static PlatformOp emit_sector(int sector_id) {
        PlatformOp op;
        op.opcode = PlatformOpCode::EMIT_SECTOR;
        op.iparam = sector_id;
        return op;
    }

    __host__ __device__ static PlatformOp barrier() {
        PlatformOp op;
        op.opcode = PlatformOpCode::BARRIER;
        return op;
    }

    // ── Structure-aware constructors ────────────────────────────────────────

    __host__ __device__ static PlatformOp declare_entity(int id,
                                                          int kind) {
        PlatformOp op;
        op.opcode = PlatformOpCode::DECLARE_ENTITY;
        op.iparam = id;
        op.target = static_cast<uint32_t>(kind);
        return op;
    }

    __host__ __device__ static PlatformOp declare_relation(int src,
                                                            int dst,
                                                            int kind) {
        PlatformOp op;
        op.opcode = PlatformOpCode::DECLARE_RELATION;
        op.reg_src = src;
        op.reg_dst = dst;
        op.iparam = kind;
        return op;
    }

    __host__ __device__ static PlatformOp declare_boundary(int id,
                                                            int btype,
                                                            int dim) {
        PlatformOp op;
        op.opcode = PlatformOpCode::DECLARE_BOUNDARY;
        op.iparam = id;
        op.target = static_cast<uint32_t>(btype);
        op.reg_src = dim;
        return op;
    }

    __host__ __device__ static PlatformOp declare_observable(
        int id, int n_outcomes, int boundary_id) {
        PlatformOp op;
        op.opcode = PlatformOpCode::DECLARE_OBSERVABLE;
        op.iparam = id;
        op.target = static_cast<uint32_t>(n_outcomes);
        op.reg_src = boundary_id;
        return op;
    }

    __host__ __device__ static PlatformOp measure_struct(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::MEASURE_STRUCT;
        op.reg_dst = reg;
        return op;
    }

    __host__ __device__ static PlatformOp reconstruct(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::RECONSTRUCT;
        op.reg_dst = reg;
        return op;
    }

    __host__ __device__ static PlatformOp emit_report(int reg) {
        PlatformOp op;
        op.opcode = PlatformOpCode::EMIT_REPORT;
        op.reg_dst = reg;
        return op;
    }

    // Cost estimation
    __host__ __device__ int estimated_cost(const MachineSpec& spec) const {
        switch (opcode) {
            case PlatformOpCode::ALLOC:
                return CostModel::alloc_cost(spec.hyb_desc());
            case PlatformOpCode::FREE:
                return 0;
            case PlatformOpCode::LIFT_DISC_TO_HYB:
                return CostModel::lift_disc_cost(spec.hyb_desc());
            case PlatformOpCode::LIFT_TOP_TO_HYB:
                return CostModel::lift_top_cost(spec.hyb_desc());
            case PlatformOpCode::PROJECT_DISC:
                return CostModel::project_disc_cost(spec.hyb_desc());
            case PlatformOpCode::PROJECT_TOP:
                return CostModel::project_top_cost(spec.hyb_desc());
            case PlatformOpCode::APPLY_X:
                return CostModel::gate_cost_x(bv1.weight());
            case PlatformOpCode::APPLY_Z:
                return CostModel::gate_cost_z(bv1.weight());
            case PlatformOpCode::APPLY_H:
                return CostModel::gate_cost_h();
            case PlatformOpCode::APPLY_U:
                return CostModel::gate_cost_u();
            case PlatformOpCode::APPLY_M:
                return CostModel::gate_cost_m();
            case PlatformOpCode::APPLY_W:
                return CostModel::gate_cost_w(bv1.weight(), bv2.weight());
            case PlatformOpCode::APPLY_P:
                return CostModel::gate_cost_p();
            case PlatformOpCode::MEASURE_DISC:
                return CostModel::project_disc_cost(spec.hyb_desc());
            case PlatformOpCode::MEASURE_JOINT:
                return CostModel::joint_measure_cost(spec.hyb_desc());
            case PlatformOpCode::CHECK_COCYCLE:
            case PlatformOpCode::CHECK_UFE:
                return 1;
            case PlatformOpCode::DECLARE_ENTITY:
            case PlatformOpCode::DECLARE_RELATION:
            case PlatformOpCode::DECLARE_BOUNDARY:
            case PlatformOpCode::DECLARE_OBSERVABLE:
                return 0;
            case PlatformOpCode::MEASURE_STRUCT:
                return CostModel::joint_measure_cost(spec.hyb_desc());
            case PlatformOpCode::RECONSTRUCT:
            case PlatformOpCode::EMIT_REPORT:
                return 1;
            default:
                return 0;
        }
    }
};

// ── Constants ───────────────────────────────────────────────────────────────
static constexpr int MAX_PLATFORM_OPS = 32768;
static constexpr int MAX_REGS = 64;

// ── Platform IR ─────────────────────────────────────────────────────────────
// State-type-aware program representation with virtual register file

struct PlatformIR {
    PlatformOp* ops;
    int count;

    // Register file: type tracking
    StateType reg_types[MAX_REGS];
    bool      reg_allocated[MAX_REGS];
    int       n_regs;           // high-water mark

    // Machine specification
    MachineSpec spec;

    __host__ PlatformIR() : count(0), n_regs(0) {
        ops = new PlatformOp[MAX_PLATFORM_OPS];
        for (int i = 0; i < MAX_REGS; ++i) {
            reg_types[i] = StateType::DISC;
            reg_allocated[i] = false;
        }
    }

    __host__ explicit PlatformIR(const MachineSpec& s)
        : count(0), n_regs(0), spec(s) {
        ops = new PlatformOp[MAX_PLATFORM_OPS];
        for (int i = 0; i < MAX_REGS; ++i) {
            reg_types[i] = StateType::DISC;
            reg_allocated[i] = false;
        }
    }

    __host__ ~PlatformIR() { delete[] ops; }

    // Copy
    __host__ PlatformIR(const PlatformIR& o)
        : count(o.count), n_regs(o.n_regs), spec(o.spec) {
        ops = new PlatformOp[MAX_PLATFORM_OPS];
        for (int i = 0; i < count; ++i) ops[i] = o.ops[i];
        for (int i = 0; i < MAX_REGS; ++i) {
            reg_types[i] = o.reg_types[i];
            reg_allocated[i] = o.reg_allocated[i];
        }
    }
    __host__ PlatformIR& operator=(const PlatformIR& o) {
        if (this != &o) {
            count = o.count;
            n_regs = o.n_regs;
            spec = o.spec;
            for (int i = 0; i < count; ++i) ops[i] = o.ops[i];
            for (int i = 0; i < MAX_REGS; ++i) {
                reg_types[i] = o.reg_types[i];
                reg_allocated[i] = o.reg_allocated[i];
            }
        }
        return *this;
    }

    // Allocate a new virtual register
    __host__ int alloc_reg(StateType type) {
        for (int i = 0; i < MAX_REGS; ++i) {
            if (!reg_allocated[i]) {
                reg_allocated[i] = true;
                reg_types[i] = type;
                if (i >= n_regs) n_regs = i + 1;
                return i;
            }
        }
        return -1;
    }

    __host__ void free_reg(int reg) {
        if (reg >= 0 && reg < MAX_REGS)
            reg_allocated[reg] = false;
    }

    __host__ void emit(PlatformOp op) {
        if (count < MAX_PLATFORM_OPS) {
            if (op.opcode == PlatformOpCode::ALLOC) {
                reg_allocated[op.reg_dst] = true;
                reg_types[op.reg_dst] = op.output_type;
                if (op.reg_dst >= n_regs) n_regs = op.reg_dst + 1;
            }
            ops[count++] = op;
        }
    }

    __host__ int total_cost() const {
        int c = 0;
        for (int i = 0; i < count; ++i)
            c += ops[i].estimated_cost(spec);
        return c;
    }

    __host__ int depth() const { return count; }
};

// ── IR Builder ──────────────────────────────────────────────────────────────
// Fluent API for constructing platform IR programs

struct IRBuilder {
    PlatformIR& ir;

    __host__ explicit IRBuilder(PlatformIR& ir_) : ir(ir_) {}

    // Allocate a state register
    __host__ int alloc(StateType type) {
        int reg = ir.alloc_reg(type);
        ir.emit(PlatformOp::alloc(reg, type));
        return reg;
    }

    // Free a register
    __host__ IRBuilder& free(int reg) {
        ir.emit(PlatformOp::free_reg(reg));
        ir.free_reg(reg);
        return *this;
    }

    // Lifting: disc → hyb (allocates new register)
    __host__ int lift_disc(int src_reg) {
        int dst = ir.alloc_reg(StateType::HYB);
        ir.emit(PlatformOp::lift_disc_to_hyb(dst, src_reg));
        return dst;
    }

    // Lifting: top → hyb (allocates new register)
    __host__ int lift_top(int src_reg) {
        int dst = ir.alloc_reg(StateType::HYB);
        ir.emit(PlatformOp::lift_top_to_hyb(dst, src_reg));
        return dst;
    }

    // Projection: hyb → disc probs (allocates new register)
    __host__ int project_disc(int src_reg) {
        int dst = ir.alloc_reg(StateType::DISC);
        ir.emit(PlatformOp::project_disc(dst, src_reg));
        return dst;
    }

    // Projection: hyb → top slice (allocates new register)
    __host__ int project_top(int src_reg, uint32_t x = 0) {
        int dst = ir.alloc_reg(StateType::TOP);
        ir.emit(PlatformOp::project_top(dst, src_reg, x));
        return dst;
    }

    // Gate operations (in-place on register)
    __host__ IRBuilder& x(int reg, BitVec u) {
        ir.emit(PlatformOp::apply_x(reg, u));
        return *this;
    }

    __host__ IRBuilder& z(int reg, BitVec v) {
        ir.emit(PlatformOp::apply_z(reg, v));
        return *this;
    }

    __host__ IRBuilder& h(int reg, int qubit) {
        ir.emit(PlatformOp::apply_h(reg, qubit));
        return *this;
    }

    __host__ IRBuilder& u(int reg, double t) {
        ir.emit(PlatformOp::apply_u(reg, t));
        return *this;
    }

    __host__ IRBuilder& m(int reg, int mod) {
        ir.emit(PlatformOp::apply_m(reg, mod));
        return *this;
    }

    __host__ IRBuilder& w(int reg, BitVec u_, BitVec v_, MirElement g) {
        ir.emit(PlatformOp::apply_w(reg, u_, v_, g));
        return *this;
    }

    __host__ IRBuilder& p(int reg, uint32_t x) {
        ir.emit(PlatformOp::apply_p(reg, x));
        return *this;
    }

    // Measurement
    __host__ IRBuilder& measure(int reg) {
        ir.emit(PlatformOp::measure_disc(reg));
        return *this;
    }

    __host__ IRBuilder& measure_joint(int reg) {
        ir.emit(PlatformOp::measure_joint(reg));
        return *this;
    }

    // Verification
    __host__ IRBuilder& check_cocycle() {
        ir.emit(PlatformOp::check_cocycle());
        return *this;
    }

    __host__ IRBuilder& check_ufe() {
        ir.emit(PlatformOp::check_ufe());
        return *this;
    }

    // Sector boundary
    __host__ IRBuilder& sector(int id) {
        ir.emit(PlatformOp::emit_sector(id));
        return *this;
    }

    __host__ IRBuilder& barrier() {
        ir.emit(PlatformOp::barrier());
        return *this;
    }

    // Structure-aware builders
    __host__ IRBuilder& entity(int id, int kind) {
        ir.emit(PlatformOp::declare_entity(id, kind));
        return *this;
    }

    __host__ IRBuilder& relation(int src, int dst, int kind) {
        ir.emit(PlatformOp::declare_relation(src, dst, kind));
        return *this;
    }

    __host__ IRBuilder& boundary(int id, int btype, int dim) {
        ir.emit(PlatformOp::declare_boundary(id, btype, dim));
        return *this;
    }

    __host__ IRBuilder& observable(int id, int n_outcomes, int bnd) {
        ir.emit(PlatformOp::declare_observable(id, n_outcomes, bnd));
        return *this;
    }

    __host__ IRBuilder& measure_struct(int reg) {
        ir.emit(PlatformOp::measure_struct(reg));
        return *this;
    }

    __host__ IRBuilder& reconstruct(int reg) {
        ir.emit(PlatformOp::reconstruct(reg));
        return *this;
    }

    __host__ IRBuilder& emit_report(int reg) {
        ir.emit(PlatformOp::emit_report(reg));
        return *this;
    }
};

// ── Import: legacy MirIR → PlatformIR ───────────────────────────────────────

namespace import {

// Convert an existing MirIR program into the new PlatformIR representation.
// The MirIR implicitly operates on a single hybrid state; we make this
// explicit by allocating register 0 as HYB.
__host__ inline PlatformIR from_mir_ir(const compiler::MirIR& mir,
                                        const MachineSpec& spec) {
    PlatformIR pir(spec);
    IRBuilder b(pir);

    int r0 = b.alloc(StateType::HYB);

    for (int i = 0; i < mir.count; ++i) {
        const compiler::MirOp& op = mir.ops[i];
        switch (op.opcode) {
            case compiler::MirOpCode::PAULI_X:
                b.x(r0, op.bv1);
                break;
            case compiler::MirOpCode::PAULI_Z:
                b.z(r0, op.bv1);
                break;
            case compiler::MirOpCode::PROJ:
                b.p(r0, op.target);
                break;
            case compiler::MirOpCode::FLOW:
                b.u(r0, op.param);
                break;
            case compiler::MirOpCode::MODULAR:
                b.m(r0, op.iparam);
                break;
            case compiler::MirOpCode::MIRROR_ACT:
                b.w(r0, BitVec(0, mir.n_qubits),
                     BitVec(0, mir.n_qubits), op.mir);
                break;
            case compiler::MirOpCode::WEYL:
                b.w(r0, op.bv1, op.bv2, op.mir);
                break;
            case compiler::MirOpCode::HADAMARD:
                b.h(r0, op.iparam);
                break;
            case compiler::MirOpCode::MEASURE:
                b.measure(r0);
                break;
            case compiler::MirOpCode::BARRIER:
                b.barrier();
                break;
            default:
                break;
        }
    }

    return pir;
}

} // namespace import

// ── Type checking ───────────────────────────────────────────────────────────

namespace type_check {

struct TypeError {
    int       op_index;
    StateType expected;
    StateType actual;
    bool      has_error;
};

// Verify all operations have compatible input/output types
__host__ inline TypeError verify(const PlatformIR& ir) {
    TypeError err = {0, StateType::DISC, StateType::DISC, false};

    StateType reg_types[MAX_REGS];
    bool      reg_live[MAX_REGS];
    for (int i = 0; i < MAX_REGS; ++i) {
        reg_types[i] = StateType::DISC;
        reg_live[i] = false;
    }

    for (int i = 0; i < ir.count; ++i) {
        const PlatformOp& op = ir.ops[i];

        switch (op.opcode) {
            case PlatformOpCode::ALLOC:
                reg_types[op.reg_dst] = op.output_type;
                reg_live[op.reg_dst] = true;
                break;

            case PlatformOpCode::FREE:
                reg_live[op.reg_dst] = false;
                break;

            case PlatformOpCode::LIFT_DISC_TO_HYB:
                if (reg_live[op.reg_src] &&
                    reg_types[op.reg_src] != StateType::DISC) {
                    return {i, StateType::DISC, reg_types[op.reg_src], true};
                }
                reg_types[op.reg_dst] = StateType::HYB;
                reg_live[op.reg_dst] = true;
                break;

            case PlatformOpCode::LIFT_TOP_TO_HYB:
                if (reg_live[op.reg_src] &&
                    reg_types[op.reg_src] != StateType::TOP) {
                    return {i, StateType::TOP, reg_types[op.reg_src], true};
                }
                reg_types[op.reg_dst] = StateType::HYB;
                reg_live[op.reg_dst] = true;
                break;

            case PlatformOpCode::PROJECT_DISC:
            case PlatformOpCode::PROJECT_TOP:
                if (reg_live[op.reg_src] &&
                    reg_types[op.reg_src] != StateType::HYB) {
                    return {i, StateType::HYB, reg_types[op.reg_src], true};
                }
                reg_types[op.reg_dst] = op.output_type;
                reg_live[op.reg_dst] = true;
                break;

            // Discrete ops: need DISC or HYB
            case PlatformOpCode::APPLY_X:
            case PlatformOpCode::APPLY_Z:
            case PlatformOpCode::APPLY_H:
                if (reg_live[op.reg_dst] &&
                    !contract::can_apply_disc_op(reg_types[op.reg_dst])) {
                    return {i, StateType::DISC, reg_types[op.reg_dst], true};
                }
                break;

            // Topological ops: need TOP or HYB
            case PlatformOpCode::APPLY_U:
            case PlatformOpCode::APPLY_M:
                if (reg_live[op.reg_dst] &&
                    !contract::can_apply_top_op(reg_types[op.reg_dst])) {
                    return {i, StateType::TOP, reg_types[op.reg_dst], true};
                }
                break;

            // Hybrid ops: need HYB
            case PlatformOpCode::APPLY_W:
            case PlatformOpCode::APPLY_P:
            case PlatformOpCode::MEASURE_DISC:
            case PlatformOpCode::MEASURE_JOINT:
            case PlatformOpCode::MEASURE_STRUCT:
                if (reg_live[op.reg_dst] &&
                    !contract::can_apply_hyb_op(reg_types[op.reg_dst])) {
                    return {i, StateType::HYB, reg_types[op.reg_dst], true};
                }
                break;

            // Structure declaration ops: no register constraints
            case PlatformOpCode::DECLARE_ENTITY:
            case PlatformOpCode::DECLARE_RELATION:
            case PlatformOpCode::DECLARE_BOUNDARY:
            case PlatformOpCode::DECLARE_OBSERVABLE:
            case PlatformOpCode::RECONSTRUCT:
            case PlatformOpCode::EMIT_REPORT:
                break;

            default:
                break;
        }
    }

    return err;
}

} // namespace type_check

// ── Auto-lift insertion (implicit mode) ─────────────────────────────────────

namespace auto_lift {

// In implicit mode, the compiler automatically inserts lift operations
// when a gate operation targets a register whose current sector type
// is incompatible.  For example, if a discrete register is targeted
// by a MEASURE_DISC (which requires HYB), the pass inserts
// LIFT_DISC_TO_HYB before the measure.

__host__ inline void insert_lifts(PlatformIR& ir) {
    StateType reg_types[MAX_REGS];
    for (int i = 0; i < MAX_REGS; ++i)
        reg_types[i] = StateType::DISC;

    PlatformOp* new_ops = new PlatformOp[MAX_PLATFORM_OPS];
    int new_count = 0;

    for (int i = 0; i < ir.count; ++i) {
        PlatformOp& op = ir.ops[i];

        // Update type tracking for alloc
        if (op.opcode == PlatformOpCode::ALLOC)
            reg_types[op.reg_dst] = op.output_type;

        // Determine if this op needs the register to be lifted
        bool needs_hyb = false;
        int target_reg = op.reg_dst;

        switch (op.opcode) {
            // These ops strictly require HYB
            case PlatformOpCode::APPLY_W:
            case PlatformOpCode::APPLY_P:
            case PlatformOpCode::MEASURE_DISC:
            case PlatformOpCode::MEASURE_JOINT:
            case PlatformOpCode::MEASURE_STRUCT:
                needs_hyb = true;
                break;

            // Disc ops on a TOP register → need lift to HYB
            case PlatformOpCode::APPLY_X:
            case PlatformOpCode::APPLY_Z:
            case PlatformOpCode::APPLY_H:
                if (reg_types[target_reg] == StateType::TOP)
                    needs_hyb = true;
                break;

            // Top ops on a DISC register → need lift to HYB
            case PlatformOpCode::APPLY_U:
            case PlatformOpCode::APPLY_M:
                if (reg_types[target_reg] == StateType::DISC)
                    needs_hyb = true;
                break;

            default:
                break;
        }

        // Insert lift if needed
        if (needs_hyb && reg_types[target_reg] != StateType::HYB) {
            if (reg_types[target_reg] == StateType::DISC) {
                if (new_count < MAX_PLATFORM_OPS)
                    new_ops[new_count++] =
                        PlatformOp::lift_disc_to_hyb(target_reg, target_reg);
            } else if (reg_types[target_reg] == StateType::TOP) {
                if (new_count < MAX_PLATFORM_OPS)
                    new_ops[new_count++] =
                        PlatformOp::lift_top_to_hyb(target_reg, target_reg);
            }
            reg_types[target_reg] = StateType::HYB;
        }

        // Track type changes from lift/project ops
        switch (op.opcode) {
            case PlatformOpCode::LIFT_DISC_TO_HYB:
            case PlatformOpCode::LIFT_TOP_TO_HYB:
                reg_types[op.reg_dst] = StateType::HYB;
                break;
            case PlatformOpCode::PROJECT_DISC:
                reg_types[op.reg_dst] = StateType::DISC;
                break;
            case PlatformOpCode::PROJECT_TOP:
                reg_types[op.reg_dst] = StateType::TOP;
                break;
            default:
                break;
        }

        if (new_count < MAX_PLATFORM_OPS)
            new_ops[new_count++] = op;
    }

    // Replace ops
    for (int i = 0; i < new_count; ++i)
        ir.ops[i] = new_ops[i];
    ir.count = new_count;

    delete[] new_ops;
}

} // namespace auto_lift

// ── Lower PlatformIR → HybridProgram ────────────────────────────────────────

namespace platform_lower {

__host__ inline HybridProgram lower_to_hybrid(const PlatformIR& ir) {
    HybridProgram prog;
    for (int i = 0; i < ir.count; ++i) {
        const PlatformOp& op = ir.ops[i];
        switch (op.opcode) {
            case PlatformOpCode::APPLY_X:
                prog.append(HybridGate::X(op.bv1));
                break;
            case PlatformOpCode::APPLY_Z:
                prog.append(HybridGate::Z(op.bv1));
                break;
            case PlatformOpCode::APPLY_H:
                prog.append(HybridGate::H(op.iparam));
                break;
            case PlatformOpCode::APPLY_U:
                prog.append(HybridGate::U(op.param));
                break;
            case PlatformOpCode::APPLY_M:
                prog.append(HybridGate::M(op.iparam));
                break;
            case PlatformOpCode::APPLY_W:
                prog.append(HybridGate::W(op.bv1, op.bv2, op.mir));
                break;
            case PlatformOpCode::APPLY_P:
                prog.append(HybridGate::P(op.target));
                break;
            case PlatformOpCode::MEASURE_DISC:
                prog.append(HybridGate::P(0));
                break;
            default:
                break;  // lifts, barriers, etc. consumed
        }
    }
    return prog;
}

} // namespace platform_lower

} // namespace platform
} // namespace topcomp
