// ============================================================================
// TopComp: Persistent Machine-Native Storage Format
// ============================================================================
// Defines the persistent object format for the machine's state manifold.
// SSD becomes part of the machine's formal topology rather than a file dump.
//
// Persistent objects:
//   - Canonical structure graphs
//   - DISC/TOP/HYB state packages
//   - Compiled IR (PlatformIR snapshots)
//   - Cocycle traces and closure ledgers
//   - Sector decompositions
//   - Measurement logs
//   - Benchmark reports
//   - Transport maps
//   - Learned structure templates
//
// Object format:
//   [ObjectHeader][payload bytes]
//
// ObjectHeader carries:
//   - magic number (format identification)
//   - version (forward compatibility)
//   - object tag (type discriminator)
//   - payload size
//   - sector metadata (n_qubits, N_theta, N_rho)
//   - integrity hash
//
// This is how the SSD becomes the machine's persistent topological memory.
// ============================================================================
#pragma once
#include "machine_state.cuh"
#include <cstring>

namespace topcomp {
namespace persistent {

// ── Format constants ────────────────────────────────────────────────────────

static constexpr uint32_t FORMAT_MAGIC   = 0x544F5043;  // "TOPC"
static constexpr uint16_t FORMAT_VERSION = 1;

// ── Object type tag ─────────────────────────────────────────────────────────

enum class ObjectTag : uint16_t {
    HYBRID_STATE       = 0x0001,  // full V_n ⊗ X_A^top state
    DISCRETE_STATE     = 0x0002,  // V_n qubit state only
    TOPOLOGICAL_STATE  = 0x0003,  // X_A^top field only
    MACHINE_STATE      = 0x0004,  // full 7-stratum machine state
    COCYCLE_LEDGER     = 0x0010,  // cocycle accumulation trace
    MEASUREMENT_LOG    = 0x0011,  // measurement outcome history
    TRACE_REPORT       = 0x0012,  // runtime provenance data
    STRUCTURE_GRAPH    = 0x0020,  // canonical structure graph
    SECTOR_DECOMP      = 0x0021,  // sector/boundary decomposition
    BENCHMARK_REPORT   = 0x0030,  // benchmark metrics
    EXECUTION_PLAN     = 0x0040,  // cached execution plan
};

// ── Object header ───────────────────────────────────────────────────────────

struct ObjectHeader {
    uint32_t magic;          // FORMAT_MAGIC
    uint16_t version;        // FORMAT_VERSION
    ObjectTag tag;           // what type of object follows
    uint32_t payload_size;   // size of payload in bytes

    // Sector metadata
    int16_t  n_qubits;
    int16_t  N_theta;
    int16_t  N_rho;
    int16_t  reserved;       // alignment padding

    // Integrity
    uint32_t checksum;       // CRC32 of payload

    __host__ ObjectHeader()
        : magic(FORMAT_MAGIC), version(FORMAT_VERSION),
          tag(ObjectTag::HYBRID_STATE), payload_size(0),
          n_qubits(0), N_theta(0), N_rho(0), reserved(0),
          checksum(0) {}

    __host__ bool valid() const {
        return magic == FORMAT_MAGIC && version <= FORMAT_VERSION;
    }

    __host__ int total_size() const {
        return static_cast<int>(sizeof(ObjectHeader)) + payload_size;
    }
};

// ── CRC32 computation ───────────────────────────────────────────────────────

__host__ inline uint32_t crc32(const void* data, int len) {
    const uint8_t* buf = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFF;
    for (int i = 0; i < len; ++i) {
        crc ^= buf[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
    }
    return ~crc;
}

// ── Serialization: HybridState → byte buffer ───────────────────────────────

struct SerializedObject {
    uint8_t* data;
    int      size;

    __host__ SerializedObject() : data(nullptr), size(0) {}

    __host__ void free() {
        delete[] data;
        data = nullptr;
        size = 0;
    }
};

// Serialize a HybridState to a persistent byte buffer
__host__ inline SerializedObject serialize_hybrid(const HybridState& hyb) {
    ObjectHeader hdr;
    hdr.tag = ObjectTag::HYBRID_STATE;
    hdr.n_qubits = static_cast<int16_t>(hyb.n_qubits);
    hdr.N_theta = static_cast<int16_t>(hyb.N_theta);
    hdr.N_rho = static_cast<int16_t>(hyb.N_rho);

    int amp_bytes = hyb.total * static_cast<int>(sizeof(C64));

    // Payload: [rho_min:8][rho_max:8][amp_data:amp_bytes]
    hdr.payload_size = 8 + 8 + amp_bytes;

    SerializedObject obj;
    obj.size = hdr.total_size();
    obj.data = new uint8_t[obj.size];

    // Write header
    memcpy(obj.data, &hdr, sizeof(ObjectHeader));

    // Write payload
    uint8_t* payload = obj.data + sizeof(ObjectHeader);
    memcpy(payload, &hyb.rho_min, 8);
    memcpy(payload + 8, &hyb.rho_max, 8);
    memcpy(payload + 16, hyb.amp, amp_bytes);

    // Compute checksum over payload
    ObjectHeader* phdr = reinterpret_cast<ObjectHeader*>(obj.data);
    phdr->checksum = crc32(payload, hdr.payload_size);

    return obj;
}

// Deserialize a HybridState from a persistent byte buffer
__host__ inline bool deserialize_hybrid(const uint8_t* buf, int buf_size,
                                         HybridState& hyb) {
    if (buf_size < static_cast<int>(sizeof(ObjectHeader))) return false;

    ObjectHeader hdr;
    memcpy(&hdr, buf, sizeof(ObjectHeader));

    if (!hdr.valid()) return false;
    if (hdr.tag != ObjectTag::HYBRID_STATE) return false;
    if (buf_size < hdr.total_size()) return false;

    // Verify checksum
    const uint8_t* payload = buf + sizeof(ObjectHeader);
    uint32_t computed = crc32(payload, hdr.payload_size);
    if (computed != hdr.checksum) return false;

    // Read dimensions and allocate
    int nq  = hdr.n_qubits;
    int nth = hdr.N_theta;
    int nrh = hdr.N_rho;

    double rho_min, rho_max;
    memcpy(&rho_min, payload, 8);
    memcpy(&rho_max, payload + 8, 8);

    hyb.init(nq, nth, nrh, rho_min, rho_max);

    int amp_bytes = hyb.total * static_cast<int>(sizeof(C64));
    if (hdr.payload_size < static_cast<uint32_t>(16 + amp_bytes)) {
        hyb.free();
        return false;
    }

    memcpy(hyb.amp, payload + 16, amp_bytes);
    return true;
}

// ── Serialization: CocycleLedger ────────────────────────────────────────────

__host__ inline SerializedObject serialize_cocycle_ledger(
    const machine_state::CocycleLedger& ledger)
{
    ObjectHeader hdr;
    hdr.tag = ObjectTag::COCYCLE_LEDGER;
    hdr.payload_size = sizeof(machine_state::CocycleLedger);

    SerializedObject obj;
    obj.size = hdr.total_size();
    obj.data = new uint8_t[obj.size];

    memcpy(obj.data, &hdr, sizeof(ObjectHeader));
    uint8_t* payload = obj.data + sizeof(ObjectHeader);
    memcpy(payload, &ledger, hdr.payload_size);

    ObjectHeader* phdr = reinterpret_cast<ObjectHeader*>(obj.data);
    phdr->checksum = crc32(payload, hdr.payload_size);

    return obj;
}

__host__ inline bool deserialize_cocycle_ledger(
    const uint8_t* buf, int buf_size,
    machine_state::CocycleLedger& ledger)
{
    if (buf_size < static_cast<int>(sizeof(ObjectHeader))) return false;

    ObjectHeader hdr;
    memcpy(&hdr, buf, sizeof(ObjectHeader));

    if (!hdr.valid()) return false;
    if (hdr.tag != ObjectTag::COCYCLE_LEDGER) return false;
    if (buf_size < hdr.total_size()) return false;

    const uint8_t* payload = buf + sizeof(ObjectHeader);
    uint32_t computed = crc32(payload, hdr.payload_size);
    if (computed != hdr.checksum) return false;

    memcpy(&ledger, payload, sizeof(machine_state::CocycleLedger));
    return true;
}

// ── Serialization: TraceSector ──────────────────────────────────────────────

__host__ inline SerializedObject serialize_trace(
    const machine_state::TraceSector& trace)
{
    ObjectHeader hdr;
    hdr.tag = ObjectTag::TRACE_REPORT;
    hdr.payload_size = sizeof(machine_state::TraceSector);

    SerializedObject obj;
    obj.size = hdr.total_size();
    obj.data = new uint8_t[obj.size];

    memcpy(obj.data, &hdr, sizeof(ObjectHeader));
    uint8_t* payload = obj.data + sizeof(ObjectHeader);
    memcpy(payload, &trace, hdr.payload_size);

    ObjectHeader* phdr = reinterpret_cast<ObjectHeader*>(obj.data);
    phdr->checksum = crc32(payload, hdr.payload_size);

    return obj;
}

__host__ inline bool deserialize_trace(
    const uint8_t* buf, int buf_size,
    machine_state::TraceSector& trace)
{
    if (buf_size < static_cast<int>(sizeof(ObjectHeader))) return false;

    ObjectHeader hdr;
    memcpy(&hdr, buf, sizeof(ObjectHeader));

    if (!hdr.valid()) return false;
    if (hdr.tag != ObjectTag::TRACE_REPORT) return false;
    if (buf_size < hdr.total_size()) return false;

    const uint8_t* payload = buf + sizeof(ObjectHeader);
    uint32_t computed = crc32(payload, hdr.payload_size);
    if (computed != hdr.checksum) return false;

    memcpy(&trace, payload, sizeof(machine_state::TraceSector));
    return true;
}

// ── Serialization: MeasurementSector ────────────────────────────────────────

__host__ inline SerializedObject serialize_measurements(
    const machine_state::MeasurementSector& meas)
{
    ObjectHeader hdr;
    hdr.tag = ObjectTag::MEASUREMENT_LOG;

    // Only serialize actual records, not the full MAX_RECORDS array
    int record_bytes = meas.n_records *
        static_cast<int>(sizeof(machine_state::MeasurementRecord));
    // Payload: [n_records:4][total_entropy:8][total_info:8][records]
    hdr.payload_size = 4 + 8 + 8 + record_bytes;

    SerializedObject obj;
    obj.size = hdr.total_size();
    obj.data = new uint8_t[obj.size];

    memcpy(obj.data, &hdr, sizeof(ObjectHeader));
    uint8_t* payload = obj.data + sizeof(ObjectHeader);
    memcpy(payload, &meas.n_records, 4);
    memcpy(payload + 4, &meas.total_entropy, 8);
    memcpy(payload + 12, &meas.total_info, 8);
    if (record_bytes > 0)
        memcpy(payload + 20, meas.records, record_bytes);

    ObjectHeader* phdr = reinterpret_cast<ObjectHeader*>(obj.data);
    phdr->checksum = crc32(payload, hdr.payload_size);

    return obj;
}

// ── Verification ────────────────────────────────────────────────────────────

namespace verify {

// Verify header integrity
__host__ inline bool header_valid(const ObjectHeader& hdr) {
    return hdr.magic == FORMAT_MAGIC &&
           hdr.version <= FORMAT_VERSION;
}

// Verify round-trip: serialize then deserialize produces identical state
__host__ inline bool hybrid_roundtrip(const HybridState& original) {
    SerializedObject obj = serialize_hybrid(original);
    HybridState restored;
    bool ok = deserialize_hybrid(obj.data, obj.size, restored);

    if (!ok) {
        obj.free();
        return false;
    }

    // Compare dimensions
    ok &= (restored.n_qubits == original.n_qubits);
    ok &= (restored.N_theta == original.N_theta);
    ok &= (restored.N_rho == original.N_rho);
    ok &= (restored.total == original.total);

    // Compare amplitudes
    if (ok) {
        double diff = 0;
        for (int i = 0; i < original.total; ++i)
            diff += (restored.amp[i] - original.amp[i]).norm2();
        ok &= (diff < 1e-20);
    }

    restored.free();
    obj.free();
    return ok;
}

// Verify checksum detection of corruption
__host__ inline bool checksum_detects_corruption() {
    HybridState hyb;
    hyb.init(1, 4, 4);
    hyb.amp[0] = C64(1, 0);

    SerializedObject obj = serialize_hybrid(hyb);

    // Corrupt one byte in the payload
    obj.data[sizeof(ObjectHeader) + 20] ^= 0xFF;

    HybridState bad;
    bool ok = deserialize_hybrid(obj.data, obj.size, bad);
    // Should fail due to checksum mismatch
    bool result = !ok;

    hyb.free();
    obj.free();
    return result;
}

} // namespace verify

} // namespace persistent
} // namespace topcomp
