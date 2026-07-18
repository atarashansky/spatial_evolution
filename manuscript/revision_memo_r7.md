# Revision memo — round 7

Governing document: `preregistration/prereg_r7.md`, deposited in the public release before any round-7 data existed
(a-priori competing-population identification; test protocols; pass criteria; decision rules for every outcome).
Round-6 panel: 48 major / 2 minor, zero fatal; dominant charges T3 (census "derivation" is identification-dependent),
T2 (headline 0.44 vs pooled constant), then T5, T4, T9, T7. This memo maps each charge to what was done.

## The result of round 7, stated first

**All three pre-registered tests came out against the paper's most recent mechanistic claim, and one exposed a systematic in our own
normalisation.** They are reported as the results they are; four claims are revised (correction register in the Discussion).

1. **Census across effect size** (prereg §B; 3,712 runs; new Fig. S24). The opposite-clone contact identification is verified s_d-free
   (2.909 / 2.902 / 2.890), but the census scale θ₀ is not (0.393 / 0.404 / 0.500; Cochran Q = 943). The fixed-calibration inversion predicts a
   flat ridge (0.427) against a measured U-shaped one (rise +0.235 [+0.122, +0.321] from s_d = 0.01 to 0.04); own-θ₀ recovers 43% of the rise.
2. **Von Neumann census stencil test** (prereg §A; 1,708 runs; kernel port bit-exact, KS p = 0.53; new Fig. S26). With identification, θ₀ and c
   all fixed in advance, the inversion predicts a vN ridge of 0.193 [0.190, 0.196] against the independently measured 0.302 [0.265, 0.400]
   (replicated by this ensemble's own dynamics, 0.316 [0.256, 0.387]) — FAIL, and the miss is inversion-vs-dynamics, not estimator noise.
   θ₀ is stencil-dependent (0.298 vs 0.404). Post-hoc populations that would pass were computed and declined.
3. **Ridge profile / headline constant** (prereg §C; 4,700 new runs incl. a 2,000-run neutral anchor; new Fig. S25). Homogeneity of the five
   fixed-s_d ridges is REJECTED (Q = 41.6, p = 2e-8; τ = 0.24; I² = 0.90) once the ladder is re-anchored on a 2,856-run pooled neutral ensemble
   (five ensembles indistinguishable, Kruskal-Wallis p = 0.91). The deposited five-point law's apparent constancy and its wide 0.0025 interval
   were common-mode noise from a 200-run legacy anchor (0.9 s.e. low; elasticity ≈ −1.1, −3.7 on the shelf). Re-anchored ladder:
   0.63 / 0.36 / 0.37 / 0.53 / 0.62 — a trough (min ≈ 0.38 near s_d ≈ 0.005–0.007). RE-pooled constant 0.47 [0.38, 0.60], fair only for
   0.003 ≲ s_d ≲ 0.02. Positive detection: slope 0.25 [0.16, 0.36] vs minimum detectable 0.13 per e-fold. **The constant-ratio law is retired.**

Consequence for the paper's spine: the phenomenon (load fluidizes, 5–21× acceleration, size law), its mechanism (random-slope gap on a zero mode,
with its out-of-sample exponent test), and the *existence* of an order-unity, area-intensive ratchet criterion are untouched. What changed is
the criterion's status: it is a **measured, weakly s_d- and stencil-dependent quantity**, which the front census reproduces at one calibration
point and does not derive. The Discussion says so, and treats the fired falsifier as evidence the remaining claims meet the same standard.

## Charge-by-charge

**T3 — census derivation is identification-dependent (28 seats).** Answered by the two pre-registered tests above, on the terms the seats
asked for: (a) census repeated at s_d = 0.0025 and 0.04 (§B); (b) the identification derived a priori from the death-Birth refill rule and
written into Methods *before* comparison; (c) the von Neumann intervention with 1.63 opposite / 2.38 same contacts (§A). Outcome: the
identification's *population* survives (contact count is a lattice constant on both stencils, at every s_d), but the inversion does not
transfer, so the derivation reading is withdrawn and the section is rewritten around what the census does and does not establish
(it excludes θ₀ = 1 and class-extinction; it closes at one point). Ledger: ridge prefactor now "measured, not derived".

**T2 — headline 0.44 vs pooled constant (30 seats).** Adopted the pre-registered rule: the headline is the random-effects pool
0.47 [0.38, 0.60], quoted as a summary of a measured profile; s_d = 0.01 (0.37 [0.32, 0.43], re-anchored) is the most precise rung near
the trough, and the historical 0.43–0.44 is identified as the same rung under the legacy anchor. Homogeneity test, crossing-level
sensitivity (0.76→0.23 at s_d = 0.01 across levels 0.3–0.7; shape level-invariant), interpolation-scheme sensitivity (0.475/0.485/0.493),
minimum-detectable-slope power analysis, and a 95% prediction interval for a new s_d (0.21–1.09) are all in Fig. S25 / `ridge_profile_summary.json`.

**T5 — tissue placements under both corrections jointly (27 seats).** Recomputed against the *profile* at each tissue's own s_d and,
separately, the joint worst case (each tissue's lowest-load corner vs the criterion's upper bound × the 2-fold shape correction). Central:
colon +1.80, epidermis +2.50, blood +2.57, hypermutator +2.44 decades. Joint worst case:
+0.13 / +0.92 / +1.07 / +1.05 — all four fluid, the colon marginal (+0.13 dec). An intermediate draft had the colon crossing;
that was traced to a stray factor of two in the load axis (U_d = μ_d, not μ_d/2 — the deposited convention) and corrected before this memo;
`tissue_placement_joint.csv`, Fig. 6c and the Discussion state the thin colon margin explicitly rather than an average.

**T4 — χ / random-slope recast (21 seats).** Fig. 3 caption now states the late-window exponent χ_late ≤ 1.31 and χ_R ≤ 1.33 with the
whole-run 1.47–2.12 identified as the launch transient; "random slope" is stated as the fixation-timescale idealization, and every
downstream number is confirmed unchanged (the exponent enters no fit).

**T9 — pre-registered experimental test (15 seats).** The falsifier is named and the decision rule sharpened; the paper now points to a
falsifier that *has* fired in this revision as the demonstration that the pre-registration is real.

**T7 — statistics / provenance (12 seats).** Global seed audit: cross-ensemble RNG-prefix sharing (distinct realisations) and one
within-ensemble reuse (S23 ladders) disclosed; the anti-tracking slope survives a seed-cluster bootstrap (0.70 [0.41, 0.79]) and
seed-independent subsets (`onset_seed_dependence_check.json`); per-arm disjointness adopted for round 7 and stated in SEEDS.md.
Fitness-precision check: min living-cell fitness 9.7e-5, ~10³⁴ above float32 subnormal; local selection weights within 16-fold
(`fitness_underflow_check.json`). Run/ensemble totals updated everywhere: **118,363 runs, 37 ensembles**, S1–S26 all cited and defined.

## New display items
S24 (census across s_d), S25 (ridge profile re-anchored; retirement of the constant-ratio law), S26 (von Neumann census test).
Data: `census_sd_*`, `ridge_profile_*`, `vn_census_*`, `tissue_placement_joint.csv`, `onset_seed_dependence_check.json`, `fitness_underflow_check.json`.
