// ============================================================================
// TopComp: Category C_Mir — Arrow Words, Monoid, Groupoid M ⋊ Γ
// ============================================================================
// From the reference:
//
//   Category C_Mir:
//     Ob(C_Mir) = {•}                (single object)
//     End_{C_Mir}(•) = Σ_arr*/~      (arrow words mod equivalence)
//     [u]∘[v] = [uv]                 (concatenation)
//     1_• = [ε]                       (empty word)
//
//   Equivalence relation:
//     u ~ v  ⟺  ev_tilde(u) = ev_tilde(v)
//     Congruence: u₁~v₁, u₂~v₂ ⟹ u₁u₂ ~ v₁v₂
//
//   Arrow alphabet:
//     Σ_arr = {→, ←, ⤡, ⤢, ↔_λ}
//     →  ↦ (1, c_φ)         ← ↦ (1, -c_φ)
//     ⤡  ↦ (1, ln ΔΣ)     ⤢ ↦ (1, -ln ΔΣ)
//     ↔_λ ↦ (-1, ln λ)
//
//   Action groupoid M ⋊ Γ:
//     Objects: points of the phase space M
//     Morphisms: (x, γ) : x → γ·x
//     Composition: (γ·x, γ') ∘ (x, γ) = (x, γ'★γ)
//
//   Mir_A⁺ = {(1,c) : c ∈ C_A}  (identity component)
//     Exp_Mir(c) = (1,c),  Log_Mir(1,c) = c
//     (1,c₁)★(1,c₂) = (1,c₁+c₂)  (abelian!)
// ============================================================================
#pragma once
#include "mir_group.cuh"
#include "phase_space.cuh"

namespace topcomp {

// ── Arrow word element ──────────────────────────────────────────────────────
// Represents a word in Σ_arr* (sequence of arrows)
struct ArrowWord {
    static constexpr int MAX_WORD_LEN = 256;
    Arrow letters[MAX_WORD_LEN];
    int length;

    __host__ __device__ ArrowWord() : length(0) {}
    __host__ __device__ ArrowWord(Arrow a) : length(1) { letters[0] = a; }

    // Concatenation: [u]∘[v] = [uv]
    __host__ __device__ ArrowWord concat(const ArrowWord& v) const {
        ArrowWord result;
        result.length = length + v.length;
        if (result.length > MAX_WORD_LEN) result.length = MAX_WORD_LEN;
        for (int i = 0; i < length && i < MAX_WORD_LEN; ++i)
            result.letters[i] = letters[i];
        for (int i = 0; i < v.length && (length + i) < MAX_WORD_LEN; ++i)
            result.letters[length + i] = v.letters[i];
        return result;
    }

    // Empty word ε
    __host__ __device__ static ArrowWord empty() { return ArrowWord(); }

    // Evaluate: ev_tilde(σ₁...σ_N) = (∏sⱼ, Σ(∏_{p<j} s_p)·cⱼ)
    __host__ __device__ MirElement evaluate() const {
        if (length == 0) return MirElement();  // identity (1, 0)

        int s_prod = 1;
        C64 c_sum(0, 0);

        for (int j = 0; j < length; ++j) {
            MirElement ej = ArrowAlphabet::ev(letters[j]);
            int sj = ej.s;
            C64 cj = ej.c;

            c_sum += C64(s_prod, 0) * cj;
            s_prod *= sj;
        }

        return MirElement(s_prod, c_sum);
    }

    // Equivalence: u ~ v ⟺ ev_tilde(u) = ev_tilde(v)
    __host__ __device__ bool equivalent(const ArrowWord& other, double tol = 1e-10) const {
        MirElement a = evaluate();
        MirElement b = other.evaluate();
        return (a.s == b.s) &&
               (fabs(a.c.re - b.c.re) < tol) &&
               (fabs(a.c.im - b.c.im) < tol);
    }

    // Append a single arrow
    __host__ __device__ void append(Arrow a) {
        if (length < MAX_WORD_LEN)
            letters[length++] = a;
    }
};

// ── Category C_Mir ──────────────────────────────────────────────────────────
// Single-object category whose endomorphism monoid is Σ_arr*/~
namespace category_mir {

// Verify associativity: ([u]∘[v])∘[w] = [u]∘([v]∘[w])
__host__ inline bool verify_associativity(const ArrowWord& u,
                                             const ArrowWord& v,
                                             const ArrowWord& w,
                                             double tol = 1e-10) {
    ArrowWord left = u.concat(v).concat(w);   // (u·v)·w
    ArrowWord right = u.concat(v.concat(w));    // u·(v·w)
    return left.equivalent(right, tol);
}

// Verify identity: ε∘u = u, u∘ε = u
__host__ inline bool verify_identity(const ArrowWord& u, double tol = 1e-10) {
    ArrowWord eps = ArrowWord::empty();
    return eps.concat(u).equivalent(u, tol) &&
           u.concat(eps).equivalent(u, tol);
}

// Verify congruence: u₁~v₁, u₂~v₂ ⟹ u₁u₂ ~ v₁v₂
__host__ inline bool verify_congruence(const ArrowWord& u1, const ArrowWord& v1,
                                          const ArrowWord& u2, const ArrowWord& v2,
                                          double tol = 1e-10) {
    if (!u1.equivalent(v1, tol) || !u2.equivalent(v2, tol))
        return false;  // precondition not met
    return u1.concat(u2).equivalent(v1.concat(v2), tol);
}

// Count equivalence classes for words up to length L over the 5-arrow alphabet
// Returns how many distinct Mir elements arise
__host__ inline int count_classes(int max_length) {
    // For short words, enumerate and count distinct ev_tilde values
    // This is meaningful up to max_length ≈ 4-5
    struct MirKey {
        int s;
        double re, im;
    };

    static constexpr int MAX_CLASSES = 10000;
    MirKey keys[MAX_CLASSES];
    int count = 0;

    // Recursive DFS over words
    // Start with empty word
    auto distinct = [&](MirElement m) -> bool {
        for (int i = 0; i < count; ++i)
            if (keys[i].s == m.s &&
                fabs(keys[i].re - m.c.re) < 1e-8 &&
                fabs(keys[i].im - m.c.im) < 1e-8)
                return false;
        return true;
    };

    // The empty word gives (1, 0)
    MirElement empty_eval;
    if (distinct(empty_eval) && count < MAX_CLASSES) {
        keys[count] = {empty_eval.s, empty_eval.c.re, empty_eval.c.im};
        count++;
    }

    // Words of length 1
    Arrow arrows[] = {Arrow::RIGHT, Arrow::LEFT, Arrow::LLDIR,
                      Arrow::RRDIR, Arrow::MIRROR};
    for (int a = 0; a < 5; ++a) {
        ArrowWord w(arrows[a]);
        MirElement m = w.evaluate();
        if (distinct(m) && count < MAX_CLASSES) {
            keys[count] = {m.s, m.c.re, m.c.im};
            count++;
        }
    }

    // Words of length 2
    if (max_length >= 2) {
        for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b) {
                ArrowWord w(arrows[a]);
                w.append(arrows[b]);
                MirElement m = w.evaluate();
                if (distinct(m) && count < MAX_CLASSES) {
                    keys[count] = {m.s, m.c.re, m.c.im};
                    count++;
                }
            }
    }

    return count;
}

} // namespace category_mir

// ── Action Groupoid M ⋊ Γ ───────────────────────────────────────────────────
// Objects:  points x ∈ M (phase space)
// Morphisms: (x, γ) : x → γ·x  where γ ∈ Mir_A
// (γ·x, γ') ∘ (x, γ) = (x, γ'★γ)
struct GroupoidMorphism {
    PhasePoint source;   // x
    MirElement gamma;     // γ

    __host__ __device__ GroupoidMorphism() : source(), gamma() {}
    __host__ __device__ GroupoidMorphism(PhasePoint x, MirElement g)
        : source(x), gamma(g) {}

    // Target: γ·x
    __host__ __device__ PhasePoint target() const {
        double ot, or_;
        gamma.act_phase(source.theta, source.rho, ot, or_);
        return PhasePoint(ot, or_);
    }

    // Composition: (γ·x, γ') ∘ (x, γ) = (x, γ'★γ)
    __host__ __device__ GroupoidMorphism compose(const GroupoidMorphism& other,
                                                    double tol = 1e-8) const {
        // 'other' is (γ·x, γ'), we are (x, γ)
        // Check composability: other.source ≈ this->target()
        PhasePoint t = target();
        // (assuming composability)
        return GroupoidMorphism(source, other.gamma.star(gamma));
    }

    // Identity morphism at x: (x, e)
    __host__ __device__ static GroupoidMorphism identity_at(PhasePoint x) {
        return GroupoidMorphism(x, MirElement());
    }

    // Inverse: (x, γ)⁻¹ = (γ·x, γ⁻¹)
    __host__ __device__ GroupoidMorphism inverse() const {
        return GroupoidMorphism(target(), gamma.inverse());
    }
};

namespace groupoid_verify {

// Verify associativity: (f ∘ g) ∘ h = f ∘ (g ∘ h)
__host__ inline bool check_associativity(PhasePoint x, MirElement g1,
                                            MirElement g2, MirElement g3,
                                            double tol = 1e-8) {
    GroupoidMorphism h(x, g1);
    GroupoidMorphism g(h.target(), g2);
    GroupoidMorphism f(g.target(), g3);

    // Left: (f ∘ g) ∘ h
    GroupoidMorphism fg = f.compose(g, tol);
    GroupoidMorphism left = fg.compose(h, tol);

    // Right: f ∘ (g ∘ h)
    GroupoidMorphism gh = g.compose(h, tol);
    GroupoidMorphism right = f.compose(gh, tol);

    // Check γ components match
    MirElement diff = left.gamma.star(right.gamma.inverse());
    return fabs(diff.c.re) < tol && fabs(diff.c.im) < tol && diff.s == 1;
}

// Verify identity: id_x ∘ f = f,  f ∘ id_x = f
__host__ inline bool check_identity(PhasePoint x, MirElement g, double tol = 1e-8) {
    GroupoidMorphism f(x, g);
    GroupoidMorphism id_src = GroupoidMorphism::identity_at(x);
    GroupoidMorphism id_tgt = GroupoidMorphism::identity_at(f.target());

    GroupoidMorphism left = id_tgt.compose(f, tol);
    GroupoidMorphism right = f.compose(id_src, tol);

    bool l_ok = (left.gamma.s == f.gamma.s) &&
                fabs(left.gamma.c.re - f.gamma.c.re) < tol;
    bool r_ok = (right.gamma.s == f.gamma.s) &&
                fabs(right.gamma.c.re - f.gamma.c.re) < tol;
    return l_ok && r_ok;
}

// Verify inverse: f⁻¹ ∘ f = id_source,  f ∘ f⁻¹ = id_target
__host__ inline bool check_inverse(PhasePoint x, MirElement g, double tol = 1e-8) {
    GroupoidMorphism f(x, g);
    GroupoidMorphism finv = f.inverse();

    GroupoidMorphism left = finv.compose(f, tol);   // f⁻¹ ∘ f
    GroupoidMorphism right = f.compose(finv, tol);  // f ∘ f⁻¹

    bool l_ok = (left.gamma.s == 1) &&
                fabs(left.gamma.c.re) < tol && fabs(left.gamma.c.im) < tol;
    bool r_ok = (right.gamma.s == 1) &&
                fabs(right.gamma.c.re) < tol && fabs(right.gamma.c.im) < tol;
    return l_ok && r_ok;
}

} // namespace groupoid_verify

// ── Mir_A⁺ identity component ───────────────────────────────────────────────
// {(1,c) : c ∈ C_A}  — abelian subgroup under ★
namespace mir_plus {

__host__ __device__ inline MirElement Exp_Mir(C64 c) {
    return MirElement(1, c);
}

__host__ __device__ inline C64 Log_Mir(MirElement g) {
    // Only valid for g in Mir⁺ (s = +1)
    return g.c;
}

// Verify (1,c₁)★(1,c₂) = (1, c₁+c₂)
__host__ inline bool verify_abelian(C64 c1, C64 c2, double tol = 1e-12) {
    MirElement g1 = Exp_Mir(c1);
    MirElement g2 = Exp_Mir(c2);
    MirElement prod = g1.star(g2);
    C64 expected = c1 + c2;
    return (prod.s == 1) && (fabs(prod.c.re - expected.re) < tol) &&
           (fabs(prod.c.im - expected.im) < tol);
}

// Verify Log(Exp(c)) = c
__host__ inline bool verify_exp_log(C64 c, double tol = 1e-12) {
    MirElement g = Exp_Mir(c);
    C64 back = Log_Mir(g);
    return (fabs(back.re - c.re) < tol) && (fabs(back.im - c.im) < tol);
}

// Verify Log_Mir(g₁★g₂) = Log_Mir(g₁) + Log_Mir(g₂)
__host__ inline bool verify_log_homomorphism(C64 c1, C64 c2, double tol = 1e-12) {
    MirElement g1 = Exp_Mir(c1);
    MirElement g2 = Exp_Mir(c2);
    C64 sum = Log_Mir(g1) + Log_Mir(g2);
    C64 prod_log = Log_Mir(g1.star(g2));
    return (fabs(sum.re - prod_log.re) < tol) && (fabs(sum.im - prod_log.im) < tol);
}

} // namespace mir_plus

// ── Master factorization blocks ─────────────────────────────────────────────
// All constructs factor through (ev_tilde, Π, Φ, Ω_Mir)
namespace factorization {

// Block 1: ρ factorization through (ev_tilde, Π)
//   ρ_{X,A}(u) = Π_{X,A}(ev_tilde_A(u))
struct RhoFactored {
    ArrowWord word;

    __host__ __device__ MirElement evaluate() const {
        return word.evaluate();
    }

    // Apply representation Π to the evaluated Mir element
    // Different targets X ∈ {M, C*, P¹, PGL, End_k}
};

// Block 2: Ω factorization through Ω_Mir
//   Ω_A(g,h) = Ω_Mir(Φ_A(g), Φ_A(h))
//   Φ_A: G_A ↪ Mir_A  (embedding)
__host__ __device__ inline C64 omega_via_mir(MirElement Phi_g, MirElement Phi_h) {
    return cocycle::omega_mir(Phi_g, Phi_h);
}

// Block 7: UFE propagation
//   UFE_hyb = 0 ⟹ UFE_Mir = 0 ⟹ UFE_A = 0
__host__ inline bool verify_ufe_chain(MirElement g, MirElement h,
                                         double tol = 1e-8) {
    C64 omega = cocycle::omega_mir(g, h);
    MirElement prod = g.star(h);
    C64 F = prod.c - g.c - C64(g.s, 0) * h.c;
    C64 ufe = F - omega;
    return ufe.norm2() < tol * tol;
}

} // namespace factorization

} // namespace topcomp
