// ============================================================================
// TopComp: Canonical Structure Graph & Boundary Calculus
// ============================================================================
// Universal input representation for any formalizable structure.
//
// The canonical structure graph is:
//   S = (V, E, H, B, O, Ω, M)
//
// where:
//   V   = entities           (nodes: tokens, symbols, instructions, ...)
//   E   = binary relations   (edges: containment, flow, reference, ...)
//   H   = hyper-relations    (higher-arity: composition, joint, ...)
//   B   = boundaries         (view laws: what is visible/exact/measured)
//   O   = observables        (sector-valued measurements on the graph)
//   Ω   = defect/cocycle     (obstruction metadata)
//   M   = metadata/measure   (weights, labels, annotations)
//
// Boundary calculus:
//   A boundary is not merely a projection operator.
//   It is a view law: (projection, observable set, reconstruction,
//                      measurement law, cost law).
//
// Every input — text, code, graph, signal, tensor — must lower into
// this canonical form before entering the hybrid pipeline.
//
// This file defines the three primitive abstractions:
//   1. StructureGraph  — typed attributed hypergraph
//   2. Boundary        — first-class programmable view law
//   3. Lifting functor — structure graph → DISC/TOP/HYB package
// ============================================================================
#pragma once
#include "fiber.cuh"
#include <cstring>

namespace topcomp {
namespace structure {

// ── Entity / relation / hyper-relation types ────────────────────────────────

enum class EntityKind : int {
    SYMBOL   = 0,   // atomic label, token, instruction
    NODE     = 1,   // graph vertex, AST node, state
    REGION   = 2,   // spatial/temporal region, block, scope
    MODE     = 3,   // frequency, channel, phase sector
    COMPOSITE = 4,  // joint/compound entity
};

enum class RelationKind : int {
    CONTAINMENT  = 0,  // parent → child
    FLOW         = 1,  // data/control flow
    REFERENCE    = 2,  // symbolic reference, binding
    ADJACENCY    = 3,  // spatial/graph adjacency
    DEPENDENCY   = 4,  // causal, temporal, logical dependency
    SYMMETRY     = 5,  // equivalence, symmetry relation
};

// ── Entity ──────────────────────────────────────────────────────────────────

struct Entity {
    int         id;
    EntityKind  kind;
    double      weight;         // measure/probability weight
    int         label;          // integer label for sector assignment
    int         boundary_id;    // which boundary this entity belongs to

    __host__ Entity()
        : id(0), kind(EntityKind::SYMBOL), weight(1.0),
          label(0), boundary_id(0) {}

    __host__ Entity(int id_, EntityKind k, int lbl = 0)
        : id(id_), kind(k), weight(1.0), label(lbl), boundary_id(0) {}
};

// ── Relation (binary edge) ──────────────────────────────────────────────────

struct Relation {
    int          src;
    int          dst;
    RelationKind kind;
    double       weight;

    __host__ Relation()
        : src(0), dst(0), kind(RelationKind::FLOW), weight(1.0) {}

    __host__ Relation(int s, int d, RelationKind k)
        : src(s), dst(d), kind(k), weight(1.0) {}
};

// ── HyperRelation (n-ary) ───────────────────────────────────────────────────

struct HyperRelation {
    static constexpr int MAX_ARITY = 16;
    int  members[MAX_ARITY];
    int  arity;
    int  label;
    double weight;

    __host__ HyperRelation() : arity(0), label(0), weight(1.0) {
        memset(members, 0, sizeof(members));
    }

    __host__ void add(int entity_id) {
        if (arity < MAX_ARITY)
            members[arity++] = entity_id;
    }
};

// ── Observable ──────────────────────────────────────────────────────────────
// A sector-valued measurement on the structure graph

struct Observable {
    int   id;
    int   n_outcomes;     // number of admissible sectors
    int   boundary_id;    // which boundary this observable reads
    bool  is_discrete;    // true = exact/finite, false = continuous/binned

    __host__ Observable()
        : id(0), n_outcomes(2), boundary_id(0), is_discrete(true) {}

    __host__ Observable(int id_, int outcomes, int bnd, bool disc = true)
        : id(id_), n_outcomes(outcomes), boundary_id(bnd),
          is_discrete(disc) {}
};

// ── Boundary ────────────────────────────────────────────────────────────────
// A view law on the structure: what is visible, what is measured,
// what can be reconstructed.
//
// Formally: B = (projection, observable set, reconstruction law,
//                measurement law, cost law)

enum class BoundaryType : int {
    DISCRETE     = 0,   // exact, addressable, Boolean, local
    TOPOLOGICAL  = 1,   // smooth, global, transport-bearing
    STRUCTURAL   = 2,   // compositional, relational, role-based
    JOINT        = 3,   // simultaneous multi-boundary view
};

struct Boundary {
    int            id;
    BoundaryType   type;
    int            dim;              // dimension of the boundary view
    int            n_observables;    // number of observables on boundary
    bool           is_exact;         // whether the boundary gives exact info
    bool           is_reconstructible;

    __host__ Boundary()
        : id(0), type(BoundaryType::DISCRETE), dim(1),
          n_observables(0), is_exact(true), is_reconstructible(true) {}

    __host__ Boundary(int id_, BoundaryType t, int d)
        : id(id_), type(t), dim(d), n_observables(0),
          is_exact(t == BoundaryType::DISCRETE),
          is_reconstructible(true) {}
};

// ── Structure Graph ─────────────────────────────────────────────────────────
// S = (V, E, H, B, O, Ω, M)
//
// The canonical representation for any formalizable input.
// Every source — text, code, graph, signal — must lower into this form.

struct StructureGraph {
    static constexpr int MAX_ENTITIES   = 4096;
    static constexpr int MAX_RELATIONS  = 8192;
    static constexpr int MAX_HYPER      = 1024;
    static constexpr int MAX_BOUNDARIES = 64;
    static constexpr int MAX_OBSERVABLES = 256;

    Entity        entities[MAX_ENTITIES];
    Relation      relations[MAX_RELATIONS];
    HyperRelation hyper[MAX_HYPER];
    Boundary      boundaries[MAX_BOUNDARIES];
    Observable    observables[MAX_OBSERVABLES];

    int n_entities;
    int n_relations;
    int n_hyper;
    int n_boundaries;
    int n_observables;

    __host__ StructureGraph()
        : n_entities(0), n_relations(0), n_hyper(0),
          n_boundaries(0), n_observables(0) {}

    // ── Builders ────────────────────────────────────────────────────────────

    __host__ int add_entity(EntityKind kind, int label = 0) {
        if (n_entities >= MAX_ENTITIES) return -1;
        int id = n_entities;
        entities[n_entities] = Entity(id, kind, label);
        n_entities++;
        return id;
    }

    __host__ int add_relation(int src, int dst, RelationKind kind) {
        if (n_relations >= MAX_RELATIONS) return -1;
        int id = n_relations;
        relations[n_relations] = Relation(src, dst, kind);
        n_relations++;
        return id;
    }

    __host__ int add_hyper(const int* member_ids, int arity, int label = 0) {
        if (n_hyper >= MAX_HYPER) return -1;
        int id = n_hyper;
        hyper[n_hyper].arity = 0;
        hyper[n_hyper].label = label;
        for (int i = 0; i < arity && i < HyperRelation::MAX_ARITY; ++i)
            hyper[n_hyper].add(member_ids[i]);
        n_hyper++;
        return id;
    }

    __host__ int add_boundary(BoundaryType type, int dim) {
        if (n_boundaries >= MAX_BOUNDARIES) return -1;
        int id = n_boundaries;
        boundaries[n_boundaries] = Boundary(id, type, dim);
        n_boundaries++;
        return id;
    }

    __host__ int add_observable(int n_outcomes, int boundary_id,
                                 bool is_discrete = true) {
        if (n_observables >= MAX_OBSERVABLES) return -1;
        int id = n_observables;
        observables[n_observables] =
            Observable(id, n_outcomes, boundary_id, is_discrete);
        boundaries[boundary_id].n_observables++;
        n_observables++;
        return id;
    }

    // ── Queries ─────────────────────────────────────────────────────────────

    __host__ int count_by_kind(EntityKind kind) const {
        int c = 0;
        for (int i = 0; i < n_entities; ++i)
            if (entities[i].kind == kind) c++;
        return c;
    }

    __host__ int count_by_boundary(int boundary_id) const {
        int c = 0;
        for (int i = 0; i < n_entities; ++i)
            if (entities[i].boundary_id == boundary_id) c++;
        return c;
    }

    __host__ int out_degree(int entity_id) const {
        int d = 0;
        for (int i = 0; i < n_relations; ++i)
            if (relations[i].src == entity_id) d++;
        return d;
    }

    __host__ int in_degree(int entity_id) const {
        int d = 0;
        for (int i = 0; i < n_relations; ++i)
            if (relations[i].dst == entity_id) d++;
        return d;
    }

    __host__ double total_weight() const {
        double w = 0;
        for (int i = 0; i < n_entities; ++i)
            w += entities[i].weight;
        return w;
    }
};

// ── Structure verification ──────────────────────────────────────────────────

namespace verify {

// All relation endpoints must reference valid entities
__host__ inline bool referential_integrity(const StructureGraph& sg) {
    for (int i = 0; i < sg.n_relations; ++i) {
        if (sg.relations[i].src < 0 ||
            sg.relations[i].src >= sg.n_entities) return false;
        if (sg.relations[i].dst < 0 ||
            sg.relations[i].dst >= sg.n_entities) return false;
    }
    for (int i = 0; i < sg.n_hyper; ++i)
        for (int j = 0; j < sg.hyper[i].arity; ++j)
            if (sg.hyper[i].members[j] < 0 ||
                sg.hyper[i].members[j] >= sg.n_entities)
                return false;
    return true;
}

// Every entity must have a valid boundary assignment
__host__ inline bool boundary_coverage(const StructureGraph& sg) {
    for (int i = 0; i < sg.n_entities; ++i)
        if (sg.entities[i].boundary_id < 0 ||
            sg.entities[i].boundary_id >= sg.n_boundaries)
            return false;
    return true;
}

// Every observable must reference a valid boundary
__host__ inline bool observable_validity(const StructureGraph& sg) {
    for (int i = 0; i < sg.n_observables; ++i)
        if (sg.observables[i].boundary_id < 0 ||
            sg.observables[i].boundary_id >= sg.n_boundaries)
            return false;
    return true;
}

// Full structural consistency check
__host__ inline bool structural_consistency(const StructureGraph& sg) {
    return referential_integrity(sg)
        && boundary_coverage(sg)
        && observable_validity(sg);
}

} // namespace verify

// ── Lifting functor: StructureGraph → DISC/TOP/HYB ──────────────────────────
// The generic lift: determines what should be discrete, what should
// be topological, and what should be jointly hybrid.
//
// Rule:
//   SYMBOL/NODE entities with discrete observable → DISC sector
//   REGION/MODE entities with continuous/global observable → TOP sector
//   Cross-boundary joint observables → HYB sector

namespace lift {

struct LiftResult {
    int dim_disc;     // total discrete sector dimension
    int dim_top;      // total topological sector dimension
    int dim_hyb;      // dim_disc * dim_top
    int n_disc_obs;   // observables assigned to DISC
    int n_top_obs;    // observables assigned to TOP
    int n_joint_obs;  // observables requiring HYB

    __host__ LiftResult()
        : dim_disc(1), dim_top(1), dim_hyb(1),
          n_disc_obs(0), n_top_obs(0), n_joint_obs(0) {}
};

// Analyze the structure graph and compute the lift decomposition
__host__ inline LiftResult analyze(const StructureGraph& sg) {
    LiftResult lr;

    // Count discrete and topological dimensions from boundaries
    int disc_total = 1;
    int top_total = 1;
    for (int i = 0; i < sg.n_boundaries; ++i) {
        switch (sg.boundaries[i].type) {
            case BoundaryType::DISCRETE:
                disc_total *= sg.boundaries[i].dim;
                break;
            case BoundaryType::TOPOLOGICAL:
                top_total *= sg.boundaries[i].dim;
                break;
            case BoundaryType::STRUCTURAL:
                disc_total *= sg.boundaries[i].dim;
                break;
            case BoundaryType::JOINT:
                disc_total *= sg.boundaries[i].dim;
                break;
        }
    }

    lr.dim_disc = disc_total;
    lr.dim_top  = top_total;
    lr.dim_hyb  = disc_total * top_total;

    // Classify observables
    for (int i = 0; i < sg.n_observables; ++i) {
        int bid = sg.observables[i].boundary_id;
        switch (sg.boundaries[bid].type) {
            case BoundaryType::DISCRETE:
            case BoundaryType::STRUCTURAL:
                lr.n_disc_obs++;
                break;
            case BoundaryType::TOPOLOGICAL:
                lr.n_top_obs++;
                break;
            case BoundaryType::JOINT:
                lr.n_joint_obs++;
                break;
        }
    }

    return lr;
}

// Build AdmissibleFamily from a DISCRETE boundary
__host__ inline sector::AdmissibleFamily
disc_family(const StructureGraph& sg, int boundary_id) {
    int dim = sg.boundaries[boundary_id].dim;
    return sector::AdmissibleFamily::computational_basis(dim);
}

// Build fiber map from two boundaries (one disc, one top)
__host__ inline fiber::FiberMap
fiber_map(const StructureGraph& sg,
          int disc_boundary_id, int top_boundary_id) {
    int dd = sg.boundaries[disc_boundary_id].dim;
    int dt = sg.boundaries[top_boundary_id].dim;
    return fiber::FiberMap(dd, dt);
}

// Determine the execution regime for a structure graph
enum class Regime : int {
    DISC_FIRST  = 0,  // mostly classical, brief lifts
    TOP_FIRST   = 1,  // mostly topological, occasional localize
    FULL_HYBRID = 2,  // lives in HYB throughout
};

__host__ inline Regime plan_regime(const StructureGraph& sg) {
    LiftResult lr = analyze(sg);
    if (lr.n_joint_obs > 0) return Regime::FULL_HYBRID;
    if (lr.n_top_obs > lr.n_disc_obs) return Regime::TOP_FIRST;
    return Regime::DISC_FIRST;
}

} // namespace lift

} // namespace structure
} // namespace topcomp
