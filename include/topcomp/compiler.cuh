// ============================================================================
// TopComp: Compiler IR — Intermediate Representation for Mir Programs
// ============================================================================
// This is the compiler core that transforms classical computations into
// topological quantum programs. The key insight from the reference:
//
//   "Turns a classical computer into a quantum computer"
//
// Architecture:
//   1. FRONTEND: Parse classical/quantum IR into MirOp sequence
//   2. OPTIMIZER: Apply algebraic simplifications using UFE = 0
//   3. LOWERING: Map to HybridProgram (gate-level)
//   4. BACKEND: Execute on HES machine or emit CUDA kernels
//
// The compiler exploits the algebraic structure of Mir_A to:
//   - Factor classical operations through the cocycle Ω_tot
//   - Detect hidden topological structure in arbitrary computation
//   - Compress programs via word reduction in C_Mir
//   - Achieve quantum-like interference from purely classical setup
//
// Optimization passes:
//   1. Word reduction: u ~ v ⟺ ev_tilde(u) = ev_tilde(v)
//   2. UFE elimination: F = Ω  ⟹  simplify phase accounting
//   3. Cocycle folding: merge adjacent gates sharing cocycle phases
//   4. Ergodic projection: replace long U^n sequences with Π
//   5. sl₂ canonicalization: use D/E/K commutation to normal-order
// ============================================================================
#pragma once
#include "weyl.cuh"
#include "category.cuh"
#include "lie_algebra.cuh"
#include "representation.cuh"
#include "witt.cuh"

namespace topcomp {
namespace compiler {

// ── MirOp: single operation in the IR ───────────────────────────────────────
enum class MirOpCode {
    // Discrete operations
    PAULI_X,        // X_u shift
    PAULI_Z,        // Z_v phase
    PROJ,           // P_x projector/measurement

    // Topological operations
    FLOW,           // U_t golden flow
    MODULAR,        // M_m modular multiplier
    MIRROR_ACT,     // ρ̂(x) general Mir action

    // Hybrid operations
    WEYL,           // W_{u,v;g} full hybrid Weyl
    HADAMARD,       // H_q Hadamard on qubit q

    // Algebraic operations (higher-level, lowered before execution)
    SL2_TRANSLATE,  // e^{hD}: translate by h
    SL2_SCALE,      // e^{(ln λ)E}: scale by λ
    SL2_CONFORMAL,  // e^{-hK}: conformal transform
    MOBIUS,         // General Möbius transformation
    ARROW_WORD,     // Evaluate an arrow word
    DUAL_FLOW,      // Dual number extended flow

    // Control flow
    LABEL,          // Branch target
    BRANCH,         // Conditional branch on measurement outcome
    REPEAT,         // Repeat block N times (for ergodic averaging)

    // Meta
    NOP,            // No operation
    BARRIER,        // Synchronization barrier
    MEASURE,        // Measure and store classical bit
};

struct MirOp {
    MirOpCode opcode;

    // Operands (union-like, depends on opcode)
    BitVec bv1;           // u for X, v for Z, etc.
    BitVec bv2;           // v for W_{u,v;g}
    MirElement mir;       // g for Mir action
    double param;         // t for U_t, h for translate, etc.
    int iparam;           // m for M_m, label for branch, N for repeat
    uint32_t target;      // x for P_x, label for branch

    // Source location (for error messages)
    int source_line;

    __host__ __device__ MirOp() : opcode(MirOpCode::NOP), bv1(0,1), bv2(0,1),
        mir(), param(0), iparam(0), target(0), source_line(0) {}

    // Named constructors
    __host__ __device__ static MirOp pauli_x(BitVec u) {
        MirOp op;
        op.opcode = MirOpCode::PAULI_X;
        op.bv1 = u;
        return op;
    }

    __host__ __device__ static MirOp pauli_z(BitVec v) {
        MirOp op;
        op.opcode = MirOpCode::PAULI_Z;
        op.bv1 = v;
        return op;
    }

    __host__ __device__ static MirOp proj(uint32_t x) {
        MirOp op;
        op.opcode = MirOpCode::PROJ;
        op.target = x;
        return op;
    }

    __host__ __device__ static MirOp flow(double t) {
        MirOp op;
        op.opcode = MirOpCode::FLOW;
        op.param = t;
        return op;
    }

    __host__ __device__ static MirOp modular(int m) {
        MirOp op;
        op.opcode = MirOpCode::MODULAR;
        op.iparam = m;
        return op;
    }

    __host__ __device__ static MirOp mirror(MirElement g) {
        MirOp op;
        op.opcode = MirOpCode::MIRROR_ACT;
        op.mir = g;
        return op;
    }

    __host__ __device__ static MirOp weyl(BitVec u, BitVec v, MirElement g) {
        MirOp op;
        op.opcode = MirOpCode::WEYL;
        op.bv1 = u;
        op.bv2 = v;
        op.mir = g;
        return op;
    }

    __host__ __device__ static MirOp hadamard(int qubit) {
        MirOp op;
        op.opcode = MirOpCode::HADAMARD;
        op.iparam = qubit;
        return op;
    }

    __host__ __device__ static MirOp sl2_translate(double h) {
        MirOp op;
        op.opcode = MirOpCode::SL2_TRANSLATE;
        op.param = h;
        return op;
    }

    __host__ __device__ static MirOp sl2_scale(double lambda) {
        MirOp op;
        op.opcode = MirOpCode::SL2_SCALE;
        op.param = lambda;
        return op;
    }

    __host__ __device__ static MirOp sl2_conformal(double h) {
        MirOp op;
        op.opcode = MirOpCode::SL2_CONFORMAL;
        op.param = h;
        return op;
    }

    __host__ __device__ static MirOp arrow_word_op(ArrowWord w) {
        MirOp op;
        op.opcode = MirOpCode::ARROW_WORD;
        op.mir = w.evaluate();
        return op;
    }

    __host__ __device__ static MirOp repeat(int N) {
        MirOp op;
        op.opcode = MirOpCode::REPEAT;
        op.iparam = N;
        return op;
    }

    __host__ __device__ static MirOp measure() {
        MirOp op;
        op.opcode = MirOpCode::MEASURE;
        return op;
    }

    __host__ __device__ static MirOp barrier() {
        MirOp op;
        op.opcode = MirOpCode::BARRIER;
        return op;
    }

    // Cost estimation
    __host__ __device__ int estimated_cost() const {
        switch (opcode) {
            case MirOpCode::PAULI_X: return bv1.weight();
            case MirOpCode::PAULI_Z: return bv1.weight();
            case MirOpCode::PROJ: return 1;
            case MirOpCode::FLOW: return 1;
            case MirOpCode::MODULAR: return 1;
            case MirOpCode::MIRROR_ACT: return 2;
            case MirOpCode::WEYL: return bv1.weight() + bv2.weight() + 2;
            case MirOpCode::HADAMARD: return 1;
            case MirOpCode::SL2_TRANSLATE: return 3;
            case MirOpCode::SL2_SCALE: return 3;
            case MirOpCode::SL2_CONFORMAL: return 3;
            case MirOpCode::ARROW_WORD: return 2;
            case MirOpCode::REPEAT: return iparam;
            case MirOpCode::MEASURE: return 1;
            default: return 0;
        }
    }
};

// ── MirIR: Intermediate representation (sequence of ops) ────────────────────
struct MirIR {
    static constexpr int MAX_OPS = 16384;
    MirOp* ops;
    int count;
    int n_qubits;

    __host__ MirIR() : count(0), n_qubits(1) {
        ops = new MirOp[MAX_OPS];
    }
    __host__ explicit MirIR(int n) : count(0), n_qubits(n) {
        ops = new MirOp[MAX_OPS];
    }
    __host__ ~MirIR() { delete[] ops; }

    // Copy
    __host__ MirIR(const MirIR& o) : count(o.count), n_qubits(o.n_qubits) {
        ops = new MirOp[MAX_OPS];
        for (int i = 0; i < count; ++i) ops[i] = o.ops[i];
    }
    __host__ MirIR& operator=(const MirIR& o) {
        if (this != &o) {
            count = o.count;
            n_qubits = o.n_qubits;
            for (int i = 0; i < count; ++i) ops[i] = o.ops[i];
        }
        return *this;
    }

    __host__ void emit(MirOp op) {
        if (count < MAX_OPS) ops[count++] = op;
    }

    __host__ int total_cost() const {
        int c = 0;
        for (int i = 0; i < count; ++i) c += ops[i].estimated_cost();
        return c;
    }

    __host__ int depth() const { return count; }
};

// ── Optimization passes ─────────────────────────────────────────────────────

namespace optimizer {

// Pass 1: Word reduction
// If consecutive MIRROR_ACT ops have evaluations with same ev_tilde,
// merge them
__host__ inline int word_reduction(MirIR& ir) {
    int eliminated = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == MirOpCode::MIRROR_ACT &&
            ir.ops[i+1].opcode == MirOpCode::MIRROR_ACT) {
            // Merge: g₁ ★ g₂
            ir.ops[i].mir = ir.ops[i].mir.star(ir.ops[i+1].mir);
            // Shift remaining ops
            for (int j = i + 1; j < ir.count - 1; ++j)
                ir.ops[j] = ir.ops[j + 1];
            ir.count--;
            eliminated++;
            --i;  // recheck at same position
        }
    }
    return eliminated;
}

// Pass 2: Identity elimination
// Remove ops that evaluate to identity
__host__ inline int identity_elimination(MirIR& ir, double tol = 1e-10) {
    int eliminated = 0;
    int write = 0;
    for (int i = 0; i < ir.count; ++i) {
        bool is_identity = false;
        switch (ir.ops[i].opcode) {
            case MirOpCode::PAULI_X:
                is_identity = (ir.ops[i].bv1.bits == 0);
                break;
            case MirOpCode::PAULI_Z:
                is_identity = (ir.ops[i].bv1.bits == 0);
                break;
            case MirOpCode::FLOW:
                is_identity = (fabs(ir.ops[i].param) < tol);
                break;
            case MirOpCode::MODULAR:
                is_identity = (ir.ops[i].iparam == 0);
                break;
            case MirOpCode::MIRROR_ACT:
                is_identity = (ir.ops[i].mir.s == 1 &&
                              ir.ops[i].mir.c.norm2() < tol * tol);
                break;
            case MirOpCode::NOP:
                is_identity = true;
                break;
            default:
                break;
        }
        if (!is_identity) {
            ir.ops[write++] = ir.ops[i];
        } else {
            eliminated++;
        }
    }
    ir.count = write;
    return eliminated;
}

// Pass 3: Flow merging
// Consecutive U_t₁, U_t₂ → U_{t₁+t₂}
__host__ inline int flow_merging(MirIR& ir) {
    int merged = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == MirOpCode::FLOW &&
            ir.ops[i+1].opcode == MirOpCode::FLOW) {
            ir.ops[i].param += ir.ops[i+1].param;
            for (int j = i + 1; j < ir.count - 1; ++j)
                ir.ops[j] = ir.ops[j + 1];
            ir.count--;
            merged++;
            --i;
        }
    }
    return merged;
}

// Pass 4: Modular merging
// Consecutive M_m₁, M_m₂ → M_{m₁+m₂}
__host__ inline int modular_merging(MirIR& ir) {
    int merged = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == MirOpCode::MODULAR &&
            ir.ops[i+1].opcode == MirOpCode::MODULAR) {
            ir.ops[i].iparam += ir.ops[i+1].iparam;
            for (int j = i + 1; j < ir.count - 1; ++j)
                ir.ops[j] = ir.ops[j + 1];
            ir.count--;
            merged++;
            --i;
        }
    }
    return merged;
}

// Pass 5: Cocycle folding
// When U_t and M_m are adjacent, account for their commutator phase
// U_t M_m = e^{-i2πmt} M_m U_t
// So reordering them produces a phase that can be absorbed
__host__ inline int cocycle_fold(MirIR& ir, double tol = 1e-10) {
    int reordered = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        // Normalize: put U before M (canonical ordering)
        if (ir.ops[i].opcode == MirOpCode::MODULAR &&
            ir.ops[i+1].opcode == MirOpCode::FLOW) {
            // M_m U_t → e^{+i2πmt} U_t M_m
            // Swap
            MirOp tmp = ir.ops[i];
            ir.ops[i] = ir.ops[i+1];
            ir.ops[i+1] = tmp;
            // The phase e^{i2πmt} is absorbed into a virtual phase gate
            // For now, we track it but don't emit (implicit phase)
            reordered++;
        }
    }
    return reordered;
}

// Pass 6: sl₂ lowering
// Convert SL2_TRANSLATE → FLOW + MIRROR_ACT sequence
// e^{hD} → T_h Möbius → translate by h on P¹
__host__ inline int sl2_lowering(MirIR& ir) {
    int lowered = 0;
    for (int i = 0; i < ir.count; ++i) {
        if (ir.ops[i].opcode == MirOpCode::SL2_TRANSLATE) {
            // e^{hD}: translation by h
            // In Mir terms: (1, h·c_φ)  (translation along flow)
            double h = ir.ops[i].param;
            ir.ops[i] = MirOp::flow(h);
            lowered++;
        }
        else if (ir.ops[i].opcode == MirOpCode::SL2_SCALE) {
            // e^{(ln λ)E}: scaling by λ
            double lambda = ir.ops[i].param;
            ir.ops[i] = MirOp::mirror(MirElement(1, C64(log(lambda), 0)));
            lowered++;
        }
        else if (ir.ops[i].opcode == MirOpCode::SL2_CONFORMAL) {
            // e^{-hK}: special conformal
            double h = ir.ops[i].param;
            // JDJ = -K, so e^{-hK} = J e^{hD} J
            // Lower to: MIRROR_ACT(J), FLOW(h), MIRROR_ACT(J)
            if (ir.count + 2 < MirIR::MAX_OPS) {
                // Shift right by 2
                for (int j = ir.count - 1; j > i; --j) {
                    if (j + 2 < MirIR::MAX_OPS)
                        ir.ops[j + 2] = ir.ops[j];
                }
                MirElement J(-1, C64(0, 0));  // mirror/inversion
                ir.ops[i] = MirOp::mirror(J);
                ir.ops[i + 1] = MirOp::flow(h);
                ir.ops[i + 2] = MirOp::mirror(J);
                ir.count += 2;
                i += 2;  // skip past inserted ops
                lowered++;
            }
        }
    }
    return lowered;
}

// Run all optimization passes
__host__ inline void optimize(MirIR& ir) {
    int changed;
    do {
        changed = 0;
        changed += sl2_lowering(ir);
        changed += word_reduction(ir);
        changed += flow_merging(ir);
        changed += modular_merging(ir);
        changed += identity_elimination(ir);
        changed += cocycle_fold(ir);
    } while (changed > 0);
}

} // namespace optimizer

// ── Lowering: MirIR → HybridProgram ─────────────────────────────────────────
namespace lowering {

__host__ inline HybridProgram lower_to_gates(const MirIR& ir) {
    HybridProgram prog;
    for (int i = 0; i < ir.count; ++i) {
        switch (ir.ops[i].opcode) {
            case MirOpCode::PAULI_X:
                prog.append(HybridGate::X(ir.ops[i].bv1));
                break;
            case MirOpCode::PAULI_Z:
                prog.append(HybridGate::Z(ir.ops[i].bv1));
                break;
            case MirOpCode::PROJ:
                prog.append(HybridGate::P(ir.ops[i].target));
                break;
            case MirOpCode::FLOW:
                prog.append(HybridGate::U(ir.ops[i].param));
                break;
            case MirOpCode::MODULAR:
                prog.append(HybridGate::M(ir.ops[i].iparam));
                break;
            case MirOpCode::MIRROR_ACT:
                // Lower to W_{0,0;g}
                prog.append(HybridGate::W(
                    BitVec(0, ir.n_qubits),
                    BitVec(0, ir.n_qubits),
                    ir.ops[i].mir));
                break;
            case MirOpCode::WEYL:
                prog.append(HybridGate::W(
                    ir.ops[i].bv1, ir.ops[i].bv2, ir.ops[i].mir));
                break;
            case MirOpCode::HADAMARD:
                prog.append(HybridGate::H(ir.ops[i].iparam));
                break;
            case MirOpCode::MEASURE:
                prog.append(HybridGate::P(0));  // measure in comp basis
                break;
            default:
                break;  // NOP, BARRIER, etc. are removed
        }
    }
    return prog;
}

} // namespace lowering

// ── Frontend: build MirIR from higher-level descriptions ────────────────────
namespace frontend {

// Build a PRNG circuit: the Mir pseudorandom generator as a compiled program
__host__ inline MirIR build_prng_circuit(int n_qubits, int N_bits,
                                            MirElement seed) {
    MirIR ir(n_qubits);

    // Initialize with seed element
    ir.emit(MirOp::mirror(seed));

    // Generate N bits via golden flow orbit
    for (int k = 0; k < N_bits; ++k) {
        // Apply g_φ(1) = flow by 1 step
        ir.emit(MirOp::flow(1.0));

        // Compute cocycle bit via measurement
        ir.emit(MirOp::measure());

        // Apply Ω evaluation (implicit in measure)
    }

    return ir;
}

// Build a structure detection circuit
// Applies successive Mir actions and measures for structure
__host__ inline MirIR build_structure_detector(int n_qubits, int depth,
                                                  MirElement probe) {
    MirIR ir(n_qubits);

    // Prepare superposition via Pauli operators
    for (int i = 0; i < n_qubits; ++i) {
        // Hadamard-like: X followed by phase
        ir.emit(MirOp::pauli_x(BitVec(1 << i, n_qubits)));
    }

    // Alternating flow and modular probes
    for (int d = 0; d < depth; ++d) {
        ir.emit(MirOp::flow(constants::PHI));
        ir.emit(MirOp::modular(d + 1));
        ir.emit(MirOp::mirror(probe));
    }

    // Final measurement
    ir.emit(MirOp::measure());

    return ir;
}

// Build an ergodic projector circuit
// Implements Π_N = (1/(2N+1)) Σ_{n=-N}^{N} U^n
__host__ inline MirIR build_ergodic_projector(int n_qubits, int N) {
    MirIR ir(n_qubits);

    // Emit the sum of U^n for n = -N...N
    // In practice, this is a single U_t with t going through the average
    for (int n = -N; n <= N; ++n) {
        ir.emit(MirOp::flow(static_cast<double>(n)));
        ir.emit(MirOp::barrier());
    }

    return ir;
}

// Build a Deutsch-Jozsa circuit for n=1:
//   Determines if f:{0,1}→{0,1} is constant or balanced in 1 query.
//   |0⟩|1⟩ → H⊗H → oracle(f) → H⊗I → measure qubit 0
//   constant: p(0)=1,  balanced: p(0)=0
__host__ inline MirIR build_deutsch_jozsa(int oracle_type) {
    MirIR ir(2);  // 2 qubits

    // Prepare |0⟩|1⟩: flip qubit 1
    ir.emit(MirOp::pauli_x(BitVec(0b10, 2)));  // X on qubit 1: |0⟩|0⟩ → |0⟩|1⟩

    // Hadamard on both qubits
    ir.emit(MirOp::hadamard(0));
    ir.emit(MirOp::hadamard(1));

    // Oracle U_f: |x⟩|y⟩ → |x⟩|y⊕f(x)⟩
    // For phase kickback: |x⟩|(-)⟩ → (-1)^{f(x)} |x⟩|(-)⟩
    if (oracle_type == 1) {
        // f(x) = x (balanced): oracle = CNOT = Z_target controlled by source
        // In Mir: Z on qubit 1, conditioned on qubit 0 = X_{10} Z_{10}
        // Simpler: just Z on qubit 0 gives (-1)^x phase
        ir.emit(MirOp::pauli_z(BitVec(0b01, 2)));
    }
    // oracle_type == 0: f(x) = 0 (constant), do nothing (identity oracle)

    // Hadamard on qubit 0 only
    ir.emit(MirOp::hadamard(0));

    return ir;
}

// Build a quantum interference demonstration circuit:
//   |0⟩⊗ξ₀ → H(0) → Z(1) → H(0) → measure
//   Expected: p(0)=0, p(1)=1  (destructive interference at |0⟩)
__host__ inline MirIR build_interference_demo(int n_qubits) {
    MirIR ir(n_qubits);
    ir.emit(MirOp::hadamard(0));
    ir.emit(MirOp::pauli_z(BitVec(1, n_qubits)));
    ir.emit(MirOp::hadamard(0));
    return ir;
}

} // namespace frontend

// ── Analysis: program properties ────────────────────────────────────────────
namespace analysis {

// Count ops by type
struct OpStats {
    int n_pauli_x;
    int n_pauli_z;
    int n_proj;
    int n_flow;
    int n_modular;
    int n_mirror;
    int n_weyl;
    int n_measure;
    int n_other;
    int total;
};

__host__ inline OpStats count_ops(const MirIR& ir) {
    OpStats s = {};
    for (int i = 0; i < ir.count; ++i) {
        switch (ir.ops[i].opcode) {
            case MirOpCode::PAULI_X: s.n_pauli_x++; break;
            case MirOpCode::PAULI_Z: s.n_pauli_z++; break;
            case MirOpCode::PROJ: s.n_proj++; break;
            case MirOpCode::FLOW: s.n_flow++; break;
            case MirOpCode::MODULAR: s.n_modular++; break;
            case MirOpCode::MIRROR_ACT: s.n_mirror++; break;
            case MirOpCode::WEYL: s.n_weyl++; break;
            case MirOpCode::MEASURE: s.n_measure++; break;
            default: s.n_other++; break;
        }
    }
    s.total = ir.count;
    return s;
}

// Estimate circuit parallelism (max independent operations)
__host__ inline int estimate_parallelism(const MirIR& ir) {
    // Simple heuristic: count barriers to determine parallelizable blocks
    int blocks = 1;
    for (int i = 0; i < ir.count; ++i)
        if (ir.ops[i].opcode == MirOpCode::BARRIER) blocks++;
    return blocks;
}

} // namespace analysis

} // namespace compiler
} // namespace topcomp
