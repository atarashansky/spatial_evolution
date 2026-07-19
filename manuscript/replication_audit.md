# Track B — Run-level / hierarchical replication audit (Round 4)

**Charge (13 seats).** Several supporting mechanistic claims pool many correlated within-run
observations and analyse them as if independent, producing implausibly extreme p-values.

**Method.** Every quoted p-value / CI in manuscript v18 was traced to its underlying data in
the artifact store and its unit of replication classified. Statistics whose unit is a
within-run observation (site-passage, pre-launch event, frame increment) were recomputed with the
**run** (or the smallest independent block) as the unit — one summary per run and aggregation
across runs, or a run/block-clustered bootstrap. Statistics already computed on run-level
aggregates were verified by exact reproduction. Where the per-run data are not persisted, this
is stated and the CI is flagged rather than recomputed. Full machine-readable table:
`replication_audit_table.csv`.

## Verdict summary

| # | Claim / statistic | Old (as printed) | Corrected (run-level) | Unit old → new | Verdict |
|---|---|---|---|---|---|
| 1 | Wake fitter than replaced tissue (paired site-passage) | Δ = +0.95 %, **p < 10⁻³⁰⁰**, n = 28,066 iid | Δ = **+0.951 % (95 % CI 0.91–0.99 %)**; serial correlation lag‑1 ρ = 0.59 (corr. length ≈ 33 sites) → n_eff ≈ 3,300; worst-case 150 run-scale blocks: t = 47.8, p ≈ 3×10⁻⁹² | site-passage (28,066) → run-scale block (~150) | **Supported, decisively.** The 10⁻³⁰⁰ is pseudo-precision from treating serially correlated passages as iid; drop the p-value, report effect + block-bootstrap CI. |
| 2 | Excursion launch-window bias (52.7 % of 7,182 events vs area-corrected 50.4 %) | binomial **p = 5×10⁻⁵** (naive-null 2.5×10⁻⁶) | events in one pre-launch window share that excursion's direction → unit = excursion: 1,959 excursions carry ≥ 1 event, z = 2.04, **p = 0.021**; with run clustering (design effect 1.16, n_eff ≈ 1,690) **p = 0.029** | pre-launch event (7,182) → excursion (1,959) / run-clustered | **Weakened.** Direction survives at p ≈ 0.02–0.03, not p ≈ 10⁻⁵. Both manuscript p-values reproduced exactly, so the correction is a clean reweighting. Restate as a modest, marginally significant enrichment. |
| 3 | Excursion median displacement 31.5 rows | 31.5 rows (2,010 excursions pooled) | run-clustering weak (ICC = 0.04, design effect 1.5, n_eff ≈ 1,318); run-clustered bootstrap 95 % CI **[30.4, 33.0]** vs naive [30.4, 32.7] | excursion → run-clustered | **Unchanged.** Descriptive median essentially unaffected; quote with the run-clustered CI. |
| 4 | Arrest fraction under load not higher (9.5 ± 0.8 % vs 10.7 ± 1.9 %, n.s.) | already per-run means | per-run: loaded 9.5 ± 0.4 % (SEM, n = 150 runs), neutral 10.7 ± 1.0 % (SEM, n = 20); Welch t = −1.13, **p = 0.27**; Mann–Whitney p = 0.29 | run → run | **Correctly replicated, still n.s.** Manuscript ± values are ≈ 2× the SEM (SD-like) — conservative, not anti-conservative. State the caveat that n = 20 neutral runs bounds the detectable difference at ~3 points. |
| 5 | Neutral win probability 0.481, 95 % CI [0.444, 0.518] | Wilson CI over 700 run outcomes | reproduced: **[0.445, 0.518]** | run → run | **Correctly replicated.** |
| 6 | Roughening exponent β = 0.384 [0.371, 0.396] (fixation ensembles); width controls 0.417–0.418 with no CI | ensemble-mean fit | run-level bootstrap on the persisted width-control trajectories (400 runs/condition): L⊥ = 256, **β = 0.415 [0.406, 0.423]**; L⊥ = 128, β = 0.387 [0.376, 0.398]. Naive OLS-on-ensemble-mean CI would be ~4× too narrow ([0.414, 0.416]) — the pseudo-replication trap. The published fixation-ensemble CI [0.371, 0.396] has run-bootstrap width, so it appears correctly computed. | time point on ensemble mean (naive alternative) → run | **Supported.** Attach the run-level CI to the control value: β = 0.415 ± 0.008 — still far above KPZ 1/3 and EW 1/4. |
| 7 | Gap variance Var[g] ∝ t^α, α = 1.47–2.12; σ_ΔR per geometry with CIs (“N = 1,536–24,576”) | α without CIs; σ_ΔR CIs from N gap-sample increments | Var[g](t) is an *across-run* variance at each t, so the α point estimates are run-replicated; an α CI would require refitting under a run bootstrap, which needs per-run g(t) — **not persisted**. The independent A_load replicate ensemble (120 runs) gives α = 1.83, corroborating α ≈ 2 vs 1. The σ_ΔR CIs and their “N” labels count within-run frame increments, so those CIs are **pseudo-replicated and too narrow**; run-level CIs cannot be recomputed from the store. | gap-sample increment → run (**not recomputable** from persisted data) | **Point estimates supported; σ_ΔR CIs flagged.** The qualitative claim (α ≈ 2, random slope) is robust. The anisotropic exponent fit (σ_ΔR ∝ L∥⁻⁰·⁷⁴L⊥⁻⁰·²⁶) inherits over-narrow input CIs; its own CIs and the ±14 % leave-one-out figure are optimistic and should be labelled as such or refit once per-run gap statistics are persisted. |
| 8 | Freeze/flow ridge law s* ∝ Λ^0.48 and nested F-tests (p = 0.48, 1.4×10⁻³, 1.8×10⁻⁶) | OLS across ridge points; F-tests on residual SS | each ridge point is itself a run-level bootstrap crossing (150 seeds/rung); ladders use disjoint seeds, so the 8–10 points are independent aggregates. No within-run pooling. | ridge point → ridge point | **Correctly replicated.** Not a pseudo-replication case. |
| 10 | Excursion waiting times “exponential, with no aging across a run” | descriptor on pooled inter-launch intervals | run-level: within-run waiting-time CV = **0.68 ± 0.02** (mean over 136 runs with ≥ 5 waits; one-sample t vs the exponential value 1: p < 10⁻⁶) — waits are **more regular than exponential** (sub-Poissonian, consistent with the search phase); per-run KS-vs-exponential rejects in only 9/101 runs, so the departure is a systematic CV deficit rather than a shape catastrophe. No aging confirmed: within-run Spearman ρ(waiting time, elapsed time) = −0.01 ± 0.03 (n = 141 runs, p = 0.67). | pooled intervals → run | **Split verdict.** “No aging” **supported** run-by-run. “Exponential” is **not** supported at run level (CV 0.68 < 1); rewrite as “memoryless in the sense of no aging, though more regular than Poisson (within-run CV 0.68 ± 0.02).” |
| 9 | Intensivity of U_d/s_d (slope −0.02/decade [−0.22, +0.19], χ² p = 0.60) | “intensive” | each crossing μ_d* is a run-bootstrapped aggregate; reproduced exactly. The defect is **framing** (underpowered null), not replication — see B3. | crossing → crossing | **Correctly replicated, misframed.** Replace “intensive” with the bounded statement (B3). |

## What changes in the text

- **Item 1 (paired-passage p-value).** Delete “p < 10⁻³⁰⁰”. State the paired increment with its
  block-bootstrap CI (+0.951 %, 95 % CI 0.91–0.99 %). Optionally note that the increment is positive in
  every run-scale block and that even the most conservative aggregation (150 run-scale blocks) gives
  p ≈ 10⁻⁹². The direction and magnitude of the wake-fitness effect are unaffected.
- **Item 2 (launch-window bias).** This is the one statistic that materially weakens. The 7,182
  pre-launch boundary events are not independent: they are grouped 3.7 per excursion, and every
  event in a given pre-launch window shares that excursion's eventual launch direction — the very
  quantity the test conditions on. The excursion is therefore the largest defensible unit
  (1,959 excursions carry ≥ 1 boundary event), and the runs contribute a further design effect of
  1.16. The corrected one-sided p is ≈ 0.02–0.03. Recommended rewrite: “A modest excess of pre-launch
  beneficial events on the launching side (52.7 % versus an area-corrected null of 50.4 %) is
  present but statistically marginal once events are clustered by excursion and run (excursion-level
  z = 2.0, p ≈ 0.02); the manuscript's supporting observation — that nearly every excursion window contains
  *some* boundary event — is confirmed at the excursion level: 97.5 % of the 2,010 excursions carry
  ≥ 1 boundary event and 87.5 % carry ≥ 2 (run-clustered 95 % CI [85.9, 88.9]); these are
  excursion-level proportions, properly replicated (unit = excursion; run design effect ≈ 1.16).
  The directional test is therefore the load-bearing statistic, and it is the one that weakens.”
- **Item 10 (waiting times).** “No aging” is confirmed run-by-run. “Exponential” should be
  softened: the within-run coefficient of variation of inter-launch waits is 0.68 ± 0.02 (136 runs),
  significantly below the exponential value of 1 — the search process is more regular than
  Poisson, plausibly reflecting a refractory period after each excursion. This does not affect any
  downstream claim.
- **Item 6 (roughening).** Add the run-level CI to the width-control exponent: 0.415 ± 0.008.
- **Item 7 (gap statistics).** Add one sentence: “Per-geometry rate-scatter confidence intervals
  are computed over frame increments and are lower bounds on the true (run-level) uncertainty; the
  central values, and the α ≈ 2 conclusion, are unaffected.” The ±14 % leave-one-out spread of the
  collapse constant should likewise be labelled as computed on point estimates.
- **Items 3, 4, 5, 8, 9.** Already run-replicated; no numerical change. Item 9's *framing* is
  corrected in B3.

## Method notes and limits

- The paired-passage arrays (`deposition_pre/post`) carry no run identifier, but they are ordered
  and strongly serially correlated (Δ lag-1 ρ = 0.59, decaying to zero by lag ≈ 33; block means at
  the ~187-passage run scale are 2.9× more dispersed than iid predicts). A stationary block bootstrap
  with block length far exceeding the correlation length (results stable across L = 50–1,000) and
  the 150-block worst case bracket the run-level uncertainty. If the true run boundaries were
  available the CI would fall between these; both leave the claim overwhelming.
- The launch-window bias correction is bracketed rather than exact because the store persists
  per-excursion event *counts*, not per-event side labels: the excursion-as-unit result (p = 0.021)
  is the correct bound under the (physically forced) assumption that events in one pre-launch
  window share that excursion's direction; partial within-window correlation (ρ = 0.25–0.5) gives
  p = 1×10⁻³ – 5×10⁻³. The manuscript's own p-values were reproduced to two significant figures
  before correction, so the reweighting is exact given the assumed clustering.
- Per-run gap trajectories g(t) are not persisted (only ensemble curves and per-geometry σ_ΔR
  point/CI summaries), so run-level CIs for σ_ΔR and for the anisotropic-scaling exponents cannot be
  recomputed here. This is a genuine gap in the archive, flagged as a caveat rather than papered
  over; the fix is to persist per-run gap increments in the next production pass (no new lattice
  campaign is required — the runs exist).
