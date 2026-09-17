// ============================================================================
// TopComp: Universal State Descriptor
// ============================================================================
// Extends the sector-only StateDescriptor (Layer 1) into a full
// structural type system for universal workloads.
//
// Every state descriptor now carries:
//   (carrier type, boundary type, structure type,
//    measurement type, resource type)
//
// This is the constitutional layer: it tells every downstream layer
// what kind of object it is dealing with, which lifts are legal,
// which projections are legal, which measurements are legal, and
// which invariants must hold.
//
// CarrierKind:
//   SYMBOLIC, GRAPH, SEQUENCE, SPATIAL, OPERATOR, PROBABILISTIC,
//   COMPOSITE
//
// The descriptor sits alongside the existing StateDescriptor; it does
// not replace it.  Instead, it adds the structural metadata that the
// universal pipeline needs to route workloads correctly.
//
// Integration:
//   StructureGraph → UniversalDescriptor → StateDescriptor + metadata
//   The universal descriptor is computed from a structure graph by
//   inspecting its entities, boundaries, and observables.
// ============================================================================
#pragma once
#include "structure.cuh"
#include "state_contract.cuh"

namespace topcomp {
namespace descriptor {

using structure::StructureGraph;
using structure::BoundaryType;
using structure::EntityKind;

// ── Carrier kind ────────────────────────────────────────────────────────────
// What the object fundamentally is

enum class CarrierKind : int {
    SYMBOLIC      = 0,  // named atoms, labels, tokens
    GRAPH         = 1,  // relational structure with edges
    SEQUENCE      = 2,  // ordered stream of entities
    SPATIAL       = 3,  // geometric/spatial arrangement
    OPERATOR      = 4,  // algebraic operator / transformation
    PROBABILISTIC = 5,  // stochastic / ensemble
    COMPOSITE     = 6,  // multi-carrier compound
};

// ── Measurement kind ────────────────────────────────────────────────────────
// What class of measurement is primary

enum class MeasurementKind : int {
    EXACT_SECTOR   = 0,  // projective, exact outcome
    STATISTICAL    = 1,  // Born/POVM, probabilistic outcome
    JOINT_BOUNDARY = 2,  // multi-boundary joint measurement
    EXPECTATION    = 3,  // continuous expectation value
};

// ── Resource class ──────────────────────────────────────────────────────────
// Cost regime hint for the runtime planner

enum class ResourceClass : int {
    SMALL    = 0,  // dim < 64, single CPU pass
    MEDIUM   = 1,  // dim < 4096, parallel-friendly
    LARGE    = 2,  // dim >= 4096, GPU-essential
};

// ── Universal descriptor ────────────────────────────────────────────────────

struct UniversalDescriptor {
    // Structural typing
    CarrierKind       carrier;
    MeasurementKind   measurement;
    ResourceClass     resource;

    // Dimensions (derived from structure graph)
    int dim_disc;
    int dim_top;
    int dim_hyb;

    // Boundary inventory
    int n_disc_boundaries;
    int n_top_boundaries;
    int n_struct_boundaries;
    int n_joint_boundaries;

    // Observable inventory
    int n_observables;
    int n_joint_observables;

    // Regime recommendation
    structure::lift::Regime regime;

    // Invariant flags
    bool requires_cocycle;
    bool requires_ufe;
    bool requires_transport;
    bool is_hybrid;

    __host__ UniversalDescriptor()
        : carrier(CarrierKind::SYMBOLIC),
          measurement(MeasurementKind::EXACT_SECTOR),
          resource(ResourceClass::SMALL),
          dim_disc(1), dim_top(1), dim_hyb(1),
          n_disc_boundaries(0), n_top_boundaries(0),
          n_struct_boundaries(0), n_joint_boundaries(0),
          n_observables(0), n_joint_observables(0),
          regime(structure::lift::Regime::DISC_FIRST),
          requires_cocycle(false), requires_ufe(false),
          requires_transport(false), is_hybrid(false) {}

    // Total dimension of the hybrid state space
    __host__ int dim() const { return dim_hyb; }

    // Can this workload stay purely discrete?
    __host__ bool is_disc_only() const {
        return n_top_boundaries == 0 && n_joint_boundaries == 0;
    }

    // Does this workload need the full hybrid substrate?
    __host__ bool needs_hybrid() const {
        return n_joint_boundaries > 0 || is_hybrid;
    }

    // Generate a StateDescriptor for Layer 1 compatibility
    __host__ contract::StateDescriptor to_state_descriptor() const {
        if (is_disc_only()) {
            contract::StateDescriptor sd;
            sd.type = contract::StateType::DISC;
            // Find nearest power of 2 >= dim_disc for qubit count
            int nq = 0;
            int d = dim_disc;
            while ((1 << nq) < d) nq++;
            sd.n_qubits = nq;
            return sd;
        }
        if (n_disc_boundaries == 0 && n_joint_boundaries == 0) {
            return contract::StateDescriptor::top(dim_top, 1);
        }
        int nq = 0;
        int d = dim_disc;
        while ((1 << nq) < d) nq++;
        return contract::StateDescriptor::hyb(nq, dim_top, 1);
    }
};

// ── Build from StructureGraph ───────────────────────────────────────────────

__host__ inline UniversalDescriptor
from_structure(const StructureGraph& sg) {
    UniversalDescriptor ud;

    // Count boundary types
    for (int i = 0; i < sg.n_boundaries; ++i) {
        switch (sg.boundaries[i].type) {
            case BoundaryType::DISCRETE:
                ud.n_disc_boundaries++;
                ud.dim_disc *= sg.boundaries[i].dim;
                break;
            case BoundaryType::TOPOLOGICAL:
                ud.n_top_boundaries++;
                ud.dim_top *= sg.boundaries[i].dim;
                break;
            case BoundaryType::STRUCTURAL:
                ud.n_struct_boundaries++;
                ud.dim_disc *= sg.boundaries[i].dim;
                break;
            case BoundaryType::JOINT:
                ud.n_joint_boundaries++;
                ud.dim_disc *= sg.boundaries[i].dim;
                break;
        }
    }
    ud.dim_hyb = ud.dim_disc * ud.dim_top;
    ud.is_hybrid = (ud.n_disc_boundaries > 0 && ud.n_top_boundaries > 0)
                   || ud.n_joint_boundaries > 0;

    // Classify observables
    ud.n_observables = sg.n_observables;
    for (int i = 0; i < sg.n_observables; ++i) {
        int bid = sg.observables[i].boundary_id;
        if (sg.boundaries[bid].type == BoundaryType::JOINT)
            ud.n_joint_observables++;
    }

    // Infer carrier kind from dominant entity type
    int sym = sg.count_by_kind(EntityKind::SYMBOL);
    int nod = sg.count_by_kind(EntityKind::NODE);
    int reg = sg.count_by_kind(EntityKind::REGION);
    int mod = sg.count_by_kind(EntityKind::MODE);

    if (sg.n_relations > sg.n_entities)
        ud.carrier = CarrierKind::GRAPH;
    else if (reg > 0 || mod > 0)
        ud.carrier = CarrierKind::SPATIAL;
    else if (nod > sym)
        ud.carrier = CarrierKind::GRAPH;
    else
        ud.carrier = CarrierKind::SYMBOLIC;

    // Infer measurement kind
    if (ud.n_joint_observables > 0)
        ud.measurement = MeasurementKind::JOINT_BOUNDARY;
    else if (ud.n_top_boundaries > 0)
        ud.measurement = MeasurementKind::STATISTICAL;
    else
        ud.measurement = MeasurementKind::EXACT_SECTOR;

    // Resource class from dimension
    if (ud.dim_hyb >= 4096)
        ud.resource = ResourceClass::LARGE;
    else if (ud.dim_hyb >= 64)
        ud.resource = ResourceClass::MEDIUM;
    else
        ud.resource = ResourceClass::SMALL;

    // Regime
    ud.regime = structure::lift::plan_regime(sg);

    // Invariant requirements
    ud.requires_cocycle = ud.n_top_boundaries > 0;
    ud.requires_transport = ud.n_top_boundaries > 0;
    ud.requires_ufe = ud.is_hybrid;

    return ud;
}

} // namespace descriptor
} // namespace topcomp
