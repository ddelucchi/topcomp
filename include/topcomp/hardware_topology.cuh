// ============================================================================
// TopComp: Hardware Topology Model
// ============================================================================
// Models the physical hardware subsystems and assigns each its doctrinal role:
//
//   CPU  = exact symbolic/discrete orchestration manifold
//   GPU  = topological and hybrid operator field engine
//   RAM  = staging manifold for lifted and measured state
//   VRAM = live topological and hybrid carrier memory
//   SSD  = persistent hybrid state algebra
//
// Each subsystem plays one exact role in maintaining the doctrine:
//
//   X_{A,n}^{hyb} = V_n ⊗_ℂ X_A^{top}
//
// Placement policies determine where each state stratum lives physically
// and how data transfers between subsystems respect the fiber decomposition.
//
// Operation weight classes label every computation by its doctrinal character:
//   DISC_HEAVY     → prefer CPU (bitwise algebra, exact branches)
//   TOP_HEAVY      → prefer GPU (transport, modulation, spectral)
//   HYB_HEAVY      → prefer GPU (tensor evolution, fiber ops)
//   MIXED          → prefer GPU (coupled disc×top operators)
//   MEAS_HEAVY     → prefer GPU (Born reductions, probability scans)
//   CLOSURE_CHECK  → prefer CPU (cocycle verification, diagnostics)
//   TRANSFER_HEAVY → prefer RAM (cross-subsystem staging)
//
// The hardware topology is not a generic resource pool.
// It is the physical carrier of the binary-topological computer.
// ============================================================================
#pragma once
#include "constants.cuh"
#include <cstring>

namespace topcomp {
namespace hardware {

// ── Physical subsystem classification ───────────────────────────────────────

enum class SubsystemKind : int {
    CPU   = 0,  // host processor: caches, registers, scalar ALU
    GPU   = 1,  // device processor: SMs, warps, shared mem, register file
    RAM   = 2,  // host main memory: DRAM channels, NUMA nodes
    VRAM  = 3,  // device memory: HBM/GDDR, L2 cache
    SSD   = 4,  // persistent storage: NVMe queues, page cache
};

// ── Doctrinal role assigned to each subsystem ───────────────────────────────

enum class SubsystemRole : int {
    EXACT_CONTROL       = 0,  // discrete orchestration, branching, projectors
    OPERATOR_ENGINE     = 1,  // topological flow, hybrid evolution, reductions
    STAGING_MANIFOLD    = 2,  // lift/project staging, sector mirrors, pinned bufs
    CARRIER_MEMORY      = 3,  // live hybrid/topological state fields
    PERSISTENT_MANIFOLD = 4,  // checkpoints, traces, compiled IR, structure cache
};

// Canonical role assignment
__host__ __device__ inline SubsystemRole canonical_role(SubsystemKind kind) {
    switch (kind) {
        case SubsystemKind::CPU:  return SubsystemRole::EXACT_CONTROL;
        case SubsystemKind::GPU:  return SubsystemRole::OPERATOR_ENGINE;
        case SubsystemKind::RAM:  return SubsystemRole::STAGING_MANIFOLD;
        case SubsystemKind::VRAM: return SubsystemRole::CARRIER_MEMORY;
        case SubsystemKind::SSD:  return SubsystemRole::PERSISTENT_MANIFOLD;
    }
    return SubsystemRole::EXACT_CONTROL;
}

// ── Operation weight class ──────────────────────────────────────────────────

enum class OpWeight : int {
    DISC_HEAVY      = 0,  // Boolean/Weyl basis, bitwise algebra
    TOP_HEAVY       = 1,  // transport, modulation, flow, spectral
    HYB_HEAVY       = 2,  // tensor evolution, fiber operations
    MIXED           = 3,  // genuinely coupled disc×top operations
    MEAS_HEAVY      = 4,  // probability reductions, Born rule
    CLOSURE_CHECK   = 5,  // cocycle verification, defect diagnostics
    TRANSFER_HEAVY  = 6,  // cross-subsystem data movement
};

// Preferred subsystem for a given operation weight
__host__ __device__ inline SubsystemKind preferred_subsystem(OpWeight w) {
    switch (w) {
        case OpWeight::DISC_HEAVY:     return SubsystemKind::CPU;
        case OpWeight::TOP_HEAVY:      return SubsystemKind::GPU;
        case OpWeight::HYB_HEAVY:      return SubsystemKind::GPU;
        case OpWeight::MIXED:          return SubsystemKind::GPU;
        case OpWeight::MEAS_HEAVY:     return SubsystemKind::GPU;
        case OpWeight::CLOSURE_CHECK:  return SubsystemKind::CPU;
        case OpWeight::TRANSFER_HEAVY: return SubsystemKind::RAM;
    }
    return SubsystemKind::CPU;
}

// ── Placement policy ────────────────────────────────────────────────────────

enum class PlacementKind : int {
    HOST_ONLY     = 0,  // CPU + RAM only (exact discrete sector)
    DEVICE_ONLY   = 1,  // GPU + VRAM only (topological carrier)
    HOST_DEVICE   = 2,  // mirrored across CPU/GPU (hybrid state)
    PINNED        = 3,  // pinned host memory for fast transfer
    PERSISTENT    = 4,  // backed to SSD
};

struct Placement {
    PlacementKind kind;
    int           device_id;     // GPU device index
    int           numa_node;     // NUMA node preference (-1 = any)
    int           stream_id;     // CUDA stream (-1 = default)
    bool          cache_aligned; // force cache-line alignment

    __host__ Placement()
        : kind(PlacementKind::HOST_ONLY), device_id(0),
          numa_node(-1), stream_id(-1), cache_aligned(true) {}

    __host__ static Placement host_only() {
        Placement p; p.kind = PlacementKind::HOST_ONLY; return p;
    }
    __host__ static Placement device_only(int dev = 0) {
        Placement p; p.kind = PlacementKind::DEVICE_ONLY;
        p.device_id = dev; return p;
    }
    __host__ static Placement host_device(int dev = 0) {
        Placement p; p.kind = PlacementKind::HOST_DEVICE;
        p.device_id = dev; return p;
    }
    __host__ static Placement pinned(int dev = 0) {
        Placement p; p.kind = PlacementKind::PINNED;
        p.device_id = dev; return p;
    }
    __host__ static Placement persistent() {
        Placement p; p.kind = PlacementKind::PERSISTENT; return p;
    }
};

// ── Transfer policy ─────────────────────────────────────────────────────────

enum class TransferMode : int {
    SYNC       = 0,  // synchronous copy
    ASYNC      = 1,  // asynchronous with stream-ordered
    ON_DEMAND  = 2,  // lazy transfer when accessed
    MIRRORED   = 3,  // always-synchronized mirror
};

struct TransferPolicy {
    TransferMode  mode;
    SubsystemKind src;
    SubsystemKind dst;
    int           chunk_fibers;   // transfer granularity in fibers
    bool          preserve_fiber; // maintain fiber structure during transfer

    __host__ TransferPolicy()
        : mode(TransferMode::SYNC), src(SubsystemKind::RAM),
          dst(SubsystemKind::VRAM), chunk_fibers(1), preserve_fiber(true) {}

    __host__ static TransferPolicy host_to_device(
        TransferMode m = TransferMode::ASYNC)
    {
        TransferPolicy tp;
        tp.mode = m;
        tp.src = SubsystemKind::RAM;
        tp.dst = SubsystemKind::VRAM;
        return tp;
    }
    __host__ static TransferPolicy device_to_host(
        TransferMode m = TransferMode::ASYNC)
    {
        TransferPolicy tp;
        tp.mode = m;
        tp.src = SubsystemKind::VRAM;
        tp.dst = SubsystemKind::RAM;
        return tp;
    }
};

// ── Hardware capability descriptor ──────────────────────────────────────────

struct SubsystemCapability {
    SubsystemKind kind;
    int64_t       capacity_bytes;  // total memory/storage capacity
    double        bandwidth_gbps;  // peak bandwidth in GB/s
    double        latency_ns;      // typical access latency in ns
    int           parallelism;     // concurrency width (cores, SMs, channels)
    bool          available;       // whether subsystem is present

    __host__ SubsystemCapability()
        : kind(SubsystemKind::CPU), capacity_bytes(0), bandwidth_gbps(0),
          latency_ns(0), parallelism(1), available(true) {}
};

// ── Hardware topology descriptor ────────────────────────────────────────────

struct HardwareTopology {
    static constexpr int MAX_SUBSYSTEMS = 8;

    SubsystemCapability subsystems[MAX_SUBSYSTEMS];
    int n_subsystems;

    // Transfer bandwidth matrix between subsystems (GB/s)
    double transfer_bw[MAX_SUBSYSTEMS][MAX_SUBSYSTEMS];

    __host__ HardwareTopology() : n_subsystems(0) {
        memset(transfer_bw, 0, sizeof(transfer_bw));
    }

    __host__ int add_subsystem(const SubsystemCapability& cap) {
        if (n_subsystems >= MAX_SUBSYSTEMS) return -1;
        int id = n_subsystems;
        subsystems[n_subsystems++] = cap;
        return id;
    }

    __host__ int find(SubsystemKind kind) const {
        for (int i = 0; i < n_subsystems; ++i)
            if (subsystems[i].kind == kind) return i;
        return -1;
    }

    __host__ void set_transfer_bw(int src, int dst, double bw_gbps) {
        if (src >= 0 && src < n_subsystems &&
            dst >= 0 && dst < n_subsystems) {
            transfer_bw[src][dst] = bw_gbps;
            transfer_bw[dst][src] = bw_gbps;
        }
    }

    // Estimate transfer time in nanoseconds
    __host__ double transfer_time_ns(int src, int dst, int64_t bytes) const {
        if (src < 0 || dst < 0 || src >= n_subsystems || dst >= n_subsystems)
            return 1e18;
        double bw = transfer_bw[src][dst];
        if (bw <= 0.0) return 1e18;
        return static_cast<double>(bytes) / (bw * 1e9) * 1e9;
    }

    // Build a default topology for a single-GPU workstation
    __host__ static HardwareTopology default_workstation() {
        HardwareTopology hw;

        SubsystemCapability cpu;
        cpu.kind = SubsystemKind::CPU;
        cpu.bandwidth_gbps = 50.0;
        cpu.latency_ns = 1.0;
        cpu.parallelism = 16;
        int cpu_id = hw.add_subsystem(cpu);

        SubsystemCapability gpu;
        gpu.kind = SubsystemKind::GPU;
        gpu.bandwidth_gbps = 500.0;
        gpu.latency_ns = 200.0;
        gpu.parallelism = 68;  // SM count
        int gpu_id = hw.add_subsystem(gpu);

        SubsystemCapability ram;
        ram.kind = SubsystemKind::RAM;
        ram.capacity_bytes = int64_t(32) << 30;
        ram.bandwidth_gbps = 50.0;
        ram.latency_ns = 80.0;
        ram.parallelism = 2;
        int ram_id = hw.add_subsystem(ram);

        SubsystemCapability vram;
        vram.kind = SubsystemKind::VRAM;
        vram.capacity_bytes = int64_t(11) << 30;
        vram.bandwidth_gbps = 616.0;
        vram.latency_ns = 300.0;
        vram.parallelism = 1;
        int vram_id = hw.add_subsystem(vram);

        SubsystemCapability ssd;
        ssd.kind = SubsystemKind::SSD;
        ssd.capacity_bytes = int64_t(1) << 40;
        ssd.bandwidth_gbps = 3.5;
        ssd.latency_ns = 10000.0;
        ssd.parallelism = 4;
        hw.add_subsystem(ssd);

        hw.set_transfer_bw(cpu_id, ram_id, 50.0);
        hw.set_transfer_bw(gpu_id, vram_id, 616.0);
        hw.set_transfer_bw(ram_id, vram_id, 15.8);  // PCIe 3.0 x16
        return hw;
    }
};

// ── Placement verification ──────────────────────────────────────────────────

namespace verify {

// Verify that a placement is consistent with its doctrinal role
__host__ inline bool placement_consistent(Placement p, SubsystemRole role) {
    switch (role) {
        case SubsystemRole::EXACT_CONTROL:
            return p.kind == PlacementKind::HOST_ONLY ||
                   p.kind == PlacementKind::PINNED;
        case SubsystemRole::OPERATOR_ENGINE:
            return p.kind == PlacementKind::DEVICE_ONLY ||
                   p.kind == PlacementKind::HOST_DEVICE;
        case SubsystemRole::STAGING_MANIFOLD:
            return p.kind == PlacementKind::HOST_DEVICE ||
                   p.kind == PlacementKind::PINNED;
        case SubsystemRole::CARRIER_MEMORY:
            return p.kind == PlacementKind::DEVICE_ONLY ||
                   p.kind == PlacementKind::HOST_DEVICE;
        case SubsystemRole::PERSISTENT_MANIFOLD:
            return p.kind == PlacementKind::PERSISTENT;
    }
    return false;
}

// Verify that a hardware topology has all required subsystems
__host__ inline bool topology_complete(const HardwareTopology& hw) {
    bool has_cpu = false, has_gpu = false, has_ram = false;
    for (int i = 0; i < hw.n_subsystems; ++i) {
        if (hw.subsystems[i].kind == SubsystemKind::CPU) has_cpu = true;
        if (hw.subsystems[i].kind == SubsystemKind::GPU) has_gpu = true;
        if (hw.subsystems[i].kind == SubsystemKind::RAM) has_ram = true;
    }
    return has_cpu && has_gpu && has_ram;
}

// Verify that a transfer policy respects fiber structure
__host__ inline bool transfer_fiber_safe(const TransferPolicy& tp) {
    return tp.preserve_fiber && tp.chunk_fibers >= 1;
}

} // namespace verify

} // namespace hardware
} // namespace topcomp
