// ============================================================================
// TopComp: Emission & Reconstruction
// ============================================================================
// The output path of the universal structure pipeline.
//
// The full lifecycle is:
//   anything in → canonical structure → hybrid state → computation
//   → measurement → reconstruction → anything out
//
// This file provides the final two stages:
//   1. Reconstruction: measurement outcomes → sector-labeled structure
//   2. Emission: sector-labeled structure → output form
//
// Emission targets:
//   SECTOR_REPORT  — table of (sector, weight, label)
//   GRAPH_REPORT   — annotated structure graph with measured weights
//   SEQUENCE_EMIT  — ordered sequence of sector labels
//   SUMMARY_EMIT   — compact scalar summary (entropy, max sector, ...)
//
// The key design principle:
//   The system is both a target and a source.
//   TopComp can ingest structure AND emit structure.
//
// Semantic API verbs realized here:
//   reconstruct(measurement, structure) → annotated structure
//   emit(annotated structure, format) → output
// ============================================================================
#pragma once
#include "descriptor.cuh"

namespace topcomp {
namespace emission {

using measurement::DenseOp;
using sector::SectorWeights;
using sector::AdmissibleFamily;
using structure::StructureGraph;

// ── Emission format ─────────────────────────────────────────────────────────

enum class EmitFormat : int {
    SECTOR_REPORT  = 0,   // sector → weight table
    GRAPH_REPORT   = 1,   // annotated structure graph
    SEQUENCE_EMIT  = 2,   // ordered label stream
    SUMMARY_EMIT   = 3,   // scalar statistics
};

// ── Sector report ───────────────────────────────────────────────────────────
// Table of (sector_id, weight, label, boundary_id)

struct SectorEntry {
    int    sector_id;
    double weight;
    int    label;
    int    boundary_id;

    __host__ SectorEntry()
        : sector_id(0), weight(0.0), label(0), boundary_id(0) {}
};

struct SectorReport {
    static constexpr int MAX_ENTRIES = 256;
    SectorEntry entries[MAX_ENTRIES];
    int         n_entries;
    double      total_weight;
    double      entropy;
    int         max_sector;

    __host__ SectorReport()
        : n_entries(0), total_weight(0.0), entropy(0.0), max_sector(0) {}
};

// ── Reconstruction ──────────────────────────────────────────────────────────
// Reconstruct: measurement outcomes → sector-labeled report

namespace reconstruct {

// From SectorWeights → SectorReport
__host__ inline SectorReport from_weights(const SectorWeights& sw) {
    SectorReport report;
    report.n_entries = sw.n_sectors;
    report.total_weight = 0.0;

    double max_w = -1.0;
    for (int a = 0; a < sw.n_sectors; ++a) {
        report.entries[a].sector_id = a;
        report.entries[a].weight = sw.weights[a];
        report.entries[a].label = a;
        report.total_weight += sw.weights[a];
        if (sw.weights[a] > max_w) {
            max_w = sw.weights[a];
            report.max_sector = a;
        }
    }

    // Shannon entropy: H = -Σ p log₂ p
    report.entropy = 0.0;
    for (int a = 0; a < sw.n_sectors; ++a) {
        double p = sw.weights[a];
        if (p > 1e-15)
            report.entropy -= p * log2(p);
    }

    return report;
}

// From DenseOp + AdmissibleFamily → SectorReport
__host__ inline SectorReport from_state(
    const AdmissibleFamily& fam, const DenseOp& rho) {
    SectorWeights sw = sector::decompose::born_weights(fam, rho);
    return from_weights(sw);
}

// Annotate structure graph entities with measured sector weights
__host__ inline void annotate_graph(StructureGraph& sg,
                                     const SectorReport& report,
                                     int boundary_id) {
    for (int i = 0; i < sg.n_entities; ++i) {
        if (sg.entities[i].boundary_id == boundary_id) {
            int lbl = sg.entities[i].label;
            if (lbl >= 0 && lbl < report.n_entries) {
                sg.entities[i].weight = report.entries[lbl].weight;
            }
        }
    }
}

} // namespace reconstruct

// ── Emission ────────────────────────────────────────────────────────────────
// Emit: SectorReport → output form

namespace emit {

// Count sectors with weight above threshold
__host__ inline int active_sectors(const SectorReport& report,
                                    double threshold = 1e-10) {
    int count = 0;
    for (int i = 0; i < report.n_entries; ++i)
        if (report.entries[i].weight > threshold) count++;
    return count;
}

// Check if the distribution is concentrated (max weight > threshold)
__host__ inline bool is_concentrated(const SectorReport& report,
                                      double threshold = 0.9) {
    if (report.n_entries == 0) return false;
    return report.entries[report.max_sector].weight > threshold;
}

// Check if the distribution is uniform (max deviation < tol)
__host__ inline bool is_uniform(const SectorReport& report,
                                 double tol = 0.01) {
    if (report.n_entries == 0) return false;
    double expected = 1.0 / report.n_entries;
    for (int i = 0; i < report.n_entries; ++i)
        if (fabs(report.entries[i].weight - expected) > tol)
            return false;
    return true;
}

// Normalized entropy: H / log₂(n), in [0, 1]
__host__ inline double normalized_entropy(const SectorReport& report) {
    if (report.n_entries <= 1) return 0.0;
    double max_ent = log2(static_cast<double>(report.n_entries));
    return report.entropy / max_ent;
}

// Emit the dominant sector label (argmax)
__host__ inline int dominant_sector(const SectorReport& report) {
    return report.max_sector;
}

// Emit a sequence of sector labels ordered by weight (descending)
__host__ inline void ranked_sectors(const SectorReport& report,
                                     int* out_ids, double* out_weights,
                                     int max_out) {
    // Copy and sort by weight (selection sort, fine for small n)
    int n = report.n_entries;
    if (n > max_out) n = max_out;

    bool used[SectorReport::MAX_ENTRIES] = {};
    for (int r = 0; r < n; ++r) {
        double best = -1.0;
        int best_idx = 0;
        for (int i = 0; i < report.n_entries; ++i) {
            if (!used[i] && report.entries[i].weight > best) {
                best = report.entries[i].weight;
                best_idx = i;
            }
        }
        used[best_idx] = true;
        out_ids[r] = report.entries[best_idx].sector_id;
        out_weights[r] = report.entries[best_idx].weight;
    }
}

} // namespace emit

// ── Semantic API verbs ──────────────────────────────────────────────────────
// Top-level operations matching the universal pipeline design:
//   ingest → canonicalize → lift → compile → execute
//   → measure → reconstruct → emit

namespace api {

// Measure on an admissible family and produce a sector report
__host__ inline SectorReport measure_and_report(
    const AdmissibleFamily& fam, const DenseOp& rho) {
    return reconstruct::from_state(fam, rho);
}

// Annotate a structure graph from a sector report
__host__ inline void annotate(StructureGraph& sg,
                               const SectorReport& report,
                               int boundary_id) {
    reconstruct::annotate_graph(sg, report, boundary_id);
}

// Query: what is the dominant structure?
__host__ inline int query_dominant(const SectorReport& report) {
    return emit::dominant_sector(report);
}

// Query: is the structure concentrated or diffuse?
__host__ inline bool query_concentrated(const SectorReport& report,
                                         double threshold = 0.9) {
    return emit::is_concentrated(report, threshold);
}

// Query: how much information does the structure carry?
__host__ inline double query_entropy(const SectorReport& report) {
    return report.entropy;
}

} // namespace api

} // namespace emission
} // namespace topcomp
