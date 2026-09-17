// ============================================================================
// TopComp Layer 4: Compiler Pipeline
// ============================================================================
// Full compilation pipeline:
//
//   Source → PlatformIR → [type check] → [auto-lift] → [optimize] → [lower]
//
// Optimization passes (extends existing compiler.cuh passes):
//   1. Identity elimination:    remove X_0, Z_0, U_0, M_0 etc.
//   2. Gate merging:            U_t₁ ; U_t₂ → U_{t₁+t₂}, M_m₁ ; M_m₂ → M_{m₁+m₂}
//   3. Cocycle folding:         canonical U-before-M ordering with phase tracking
//   4. Dead register removal:   remove alloc/free pairs with no intervening ops
//   5. Barrier elimination:     remove consecutive barriers
//
// Sector-preserving transforms:
//   All optimizations preserve the state type annotations.
//   The pipeline verifies type correctness after optimization.
//
// Import from legacy MirIR:
//   - Existing MirIR programs can be imported and optimized
//   - Legacy optimizer passes from compiler.cuh are available
//
// Lowering:
//   PlatformIR → HybridProgram (for HESMachine execution)
//   PlatformIR → Runtime VM    (for direct execution)
// ============================================================================
#pragma once
#include "runtime.cuh"

namespace topcomp {
namespace pipeline {

using contract::StateType;
using contract::MachineSpec;

// ── Optimization passes for PlatformIR ──────────────────────────────────────

namespace optimizer {

// Pass 1: Identity elimination
// Remove operations that evaluate to identity:
//   X_0, Z_0, U_{t≈0}, M_0
__host__ inline int identity_elimination(platform::PlatformIR& ir,
                                          double tol = 1e-10) {
    int eliminated = 0;
    int write = 0;
    for (int i = 0; i < ir.count; ++i) {
        bool is_id = false;
        switch (ir.ops[i].opcode) {
            case platform::PlatformOpCode::APPLY_X:
                is_id = (ir.ops[i].bv1.bits == 0);
                break;
            case platform::PlatformOpCode::APPLY_Z:
                is_id = (ir.ops[i].bv1.bits == 0);
                break;
            case platform::PlatformOpCode::APPLY_U:
                is_id = (fabs(ir.ops[i].param) < tol);
                break;
            case platform::PlatformOpCode::APPLY_M:
                is_id = (ir.ops[i].iparam == 0);
                break;
            case platform::PlatformOpCode::NOP:
                is_id = true;
                break;
            default:
                break;
        }
        if (!is_id) {
            ir.ops[write++] = ir.ops[i];
        } else {
            eliminated++;
        }
    }
    ir.count = write;
    return eliminated;
}

// Pass 2: Flow merging
// Consecutive U_t₁, U_t₂ on the same register → U_{t₁+t₂}
__host__ inline int flow_merging(platform::PlatformIR& ir) {
    int merged = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == platform::PlatformOpCode::APPLY_U &&
            ir.ops[i+1].opcode == platform::PlatformOpCode::APPLY_U &&
            ir.ops[i].reg_dst == ir.ops[i+1].reg_dst) {
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

// Pass 3: Modular merging
// Consecutive M_m₁, M_m₂ on the same register → M_{m₁+m₂}
__host__ inline int modular_merging(platform::PlatformIR& ir) {
    int merged = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == platform::PlatformOpCode::APPLY_M &&
            ir.ops[i+1].opcode == platform::PlatformOpCode::APPLY_M &&
            ir.ops[i].reg_dst == ir.ops[i+1].reg_dst) {
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

// Pass 4: Cocycle folding
// Canonical ordering: U before M on the same register
// U_t M_m = e^{-i2πmt} M_m U_t  (phase is tracked implicitly)
__host__ inline int cocycle_fold(platform::PlatformIR& ir) {
    int reordered = 0;
    for (int i = 0; i < ir.count - 1; ++i) {
        if (ir.ops[i].opcode == platform::PlatformOpCode::APPLY_M &&
            ir.ops[i+1].opcode == platform::PlatformOpCode::APPLY_U &&
            ir.ops[i].reg_dst == ir.ops[i+1].reg_dst) {
            // Swap: M_m U_t → U_t M_m (absorb phase)
            platform::PlatformOp tmp = ir.ops[i];
            ir.ops[i] = ir.ops[i+1];
            ir.ops[i+1] = tmp;
            reordered++;
        }
    }
    return reordered;
}

// Pass 5: Barrier elimination
// Remove consecutive barriers (keep only one)
__host__ inline int barrier_elimination(platform::PlatformIR& ir) {
    int eliminated = 0;
    int write = 0;
    bool prev_barrier = false;
    for (int i = 0; i < ir.count; ++i) {
        bool is_barrier =
            (ir.ops[i].opcode == platform::PlatformOpCode::BARRIER);
        if (is_barrier && prev_barrier) {
            eliminated++;
        } else {
            ir.ops[write++] = ir.ops[i];
        }
        prev_barrier = is_barrier;
    }
    ir.count = write;
    return eliminated;
}

// Pass 6: Dead register elimination
// Remove alloc/free pairs with no operations on the register between them
__host__ inline int dead_register_elimination(platform::PlatformIR& ir) {
    int eliminated = 0;
    for (int i = 0; i < ir.count; ++i) {
        if (ir.ops[i].opcode != platform::PlatformOpCode::ALLOC) continue;
        int reg = ir.ops[i].reg_dst;

        // Find the matching FREE
        bool used = false;
        int free_idx = -1;
        for (int j = i + 1; j < ir.count; ++j) {
            if (ir.ops[j].opcode == platform::PlatformOpCode::FREE &&
                ir.ops[j].reg_dst == reg) {
                free_idx = j;
                break;
            }
            // Check if register is used by any op between alloc and free
            if (ir.ops[j].reg_dst == reg || ir.ops[j].reg_src == reg) {
                if (ir.ops[j].opcode != platform::PlatformOpCode::FREE) {
                    used = true;
                    break;
                }
            }
        }

        if (!used && free_idx >= 0) {
            // Remove both alloc and free
            ir.ops[i].opcode = platform::PlatformOpCode::NOP;
            ir.ops[free_idx].opcode = platform::PlatformOpCode::NOP;
            eliminated += 2;
        }
    }

    // Clean up NOPs
    if (eliminated > 0) {
        int write = 0;
        for (int i = 0; i < ir.count; ++i) {
            if (ir.ops[i].opcode != platform::PlatformOpCode::NOP)
                ir.ops[write++] = ir.ops[i];
        }
        ir.count = write;
    }
    return eliminated;
}

// Run all optimization passes until fixed point
__host__ inline void optimize(platform::PlatformIR& ir) {
    int changed;
    do {
        changed = 0;
        changed += identity_elimination(ir);
        changed += flow_merging(ir);
        changed += modular_merging(ir);
        changed += cocycle_fold(ir);
        changed += barrier_elimination(ir);
        changed += dead_register_elimination(ir);
    } while (changed > 0);
}

} // namespace optimizer

// ── Compiler pipeline ───────────────────────────────────────────────────────

struct CompilerPipeline {
    MachineSpec spec;
    bool        auto_lift_enabled;

    __host__ CompilerPipeline(const MachineSpec& s, bool auto_lift = true)
        : spec(s), auto_lift_enabled(auto_lift) {}

    // ── Full pipeline stages ────────────────────────────────────────────────

    // Stage 1: Type check (returns true if valid)
    __host__ bool type_check(const platform::PlatformIR& ir) const {
        platform::type_check::TypeError err =
            platform::type_check::verify(ir);
        return !err.has_error;
    }

    // Stage 2: Auto-lift insertion (implicit mode)
    __host__ void insert_lifts(platform::PlatformIR& ir) const {
        if (auto_lift_enabled)
            platform::auto_lift::insert_lifts(ir);
    }

    // Stage 3: Optimize
    __host__ void optimize(platform::PlatformIR& ir) const {
        optimizer::optimize(ir);
    }

    // Stage 4: Lower to HybridProgram
    __host__ HybridProgram lower(const platform::PlatformIR& ir) const {
        return platform::platform_lower::lower_to_hybrid(ir);
    }

    // ── Convenience: full pipeline ──────────────────────────────────────────

    // Compile: type check → auto-lift → optimize
    __host__ bool compile(platform::PlatformIR& ir) const {
        insert_lifts(ir);
        optimize(ir);
        return type_check(ir);
    }

    // Compile and execute on VM
    __host__ void compile_and_run(platform::PlatformIR& ir,
                                   runtime::PlatformVM& vm) const {
        compile(ir);
        vm.run(ir);
    }

    // ── Legacy import ───────────────────────────────────────────────────────

    // Import from MirIR, optimize with legacy passes, then convert
    __host__ platform::PlatformIR import_mir(
        const compiler::MirIR& mir) const {
        // First apply legacy optimization
        compiler::MirIR optimized = mir;
        compiler::optimizer::optimize(optimized);

        // Convert to PlatformIR
        platform::PlatformIR pir =
            platform::import::from_mir_ir(optimized, spec);

        // Apply platform-level optimization
        compile(pir);
        return pir;
    }
};

} // namespace pipeline
} // namespace topcomp
