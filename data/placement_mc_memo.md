# Round 8 — Monte-Carlo tissue-placement propagation and anchor-pool leave-one-out

**Verdict: FOR the tested prediction, on both computations, and I could not break either.** (A) The five-rung ridge profile and the random-effects (RE) pooled ridge are insensitive to the composition of the pooled neutral anchor: dropping any one of the five contributing anchor sub-ensembles moves the RE pooled ridge by at most 0.007 (1.5%; from 0.474 to 0.467, when the 2,000-run round-7 anchor is dropped), moves no rung by more than 0.018, and never rescues homogeneity (worst leave-one-out Cochran Q_p = 2.1×10⁻⁶; trough at s_d = 0.005 under every leave-one-out and under both crossing schemes). (B) Propagating the full stated input boxes together with a heterogeneity-resampled ridge (log-ridge ~ N(log 0.475, τ = 0.235)) and a log-uniform 1.3–3× shape correction, 100,000 draws per tissue, puts the colonic crypt on the fluid side with probability **0.9999** (in-range subset 0.9997; median margin +1.43 decades, 5th–95th percentile +0.58 to +2.29 dec) and every other tissue at **1.0000** (epidermis, HSC, hypermutator tumour). The prediction's own thresholds (colon ≥ 0.9, others ≥ 0.97) are cleared by three to four nines. The colon 'marginal but fluid' single-corner joint worst case (+0.13 dec) is not the right summary — it is the intersection of three simultaneous 2.5-percentile events with joint probability ≈ 10⁻⁴ — and should be replaced by the propagated probability, which is not marginal at all: the freeze barrier (ridge × shape) would have to be **5.6× larger than measured** to bring the colon P(fluid) down to 0.9, and **11.7× larger** to reach the AGAINST threshold of 0.75.

## What was computed

**Track A — anchor leave-one-out** (`anchor_loo_table.csv`, script `run_anchor_loo.py`, estimator `ridge_estimator.py`). The pooled neutral anchor in `data/ridge_profile_analysis_pool.parquet` (kind = neutral, n = 2,856, median 572,584 frames) is the union of five sub-ensembles identified by the `pool` column and their seed blocks:

| anchor sub-ensemble | n | median T_fix | seeds |
|---|---:|---:|---|
| ladderN32 (legacy round-1 anchor) | 200 | 540,332 | 936,000–936,199 |
| census (round-7 census neutrals) | 256 | 575,481 | 12,643,456–12,643,711 |
| tfixexp1 | 150 | 585,868 | 200,150–200,299 |
| tfixexp2 | 250 | 601,622 | 202,750–202,999 |
| new (round-7 anchor arm) | 2,000 | 569,566 | 12,900,000–12,901,999 |

Kruskal-Wallis across the five: H = 0.97, p = 0.915 — the sub-ensembles are statistically indistinguishable, so no leave-one-out was ever expected to matter, and none does. The estimator was replicated exactly against the deposit before use: the 'log-linear' scheme is the median-T_fix/anchor ratio interpolated linearly against log θ at the first descending crossing of 1/2 (this reproduces the deposited five-point profile to 10⁻⁶; interpolating log-ratio instead shifts the s_d = 0.01 rung to 0.370 and is not the deposited scheme); the isotonic scheme is weighted PAVA on the rung ratios before the same crossing (4-decimal agreement). The bootstrap is run-level, B = 3,000, every rung and every remaining anchor sub-ensemble resampled, one shared anchor draw across the five s_d per replicate (the deposit's common-mode anchor treatment); the RE pool is DerSimonian-Laird on log θ*. Full-pool replication: RE pooled 0.474 [0.378, 0.595], τ = 0.236, Q = 42.7 (deposit: 0.475 [0.379, 0.595], τ = 0.235, Q = 41.6).

Leave-one-out results (log-linear scheme; isotonic in `anchor_loo_table.csv` behaves identically, RE 0.477–0.489):

| dropped | anchor median | θ*(0.005) | θ*(0.01) | RE pooled [95% CI] | τ | Q_p |
|---|---:|---:|---:|---|---:|---|
| none (full) | 572,584 | 0.358 | 0.373 | 0.474 [0.378, 0.595] | 0.236 | 1.2×10⁻⁸ |
| new (2,000) | 577,782 | 0.355 | 0.369 | 0.467 [0.373, 0.585] | 0.229 | 2.1×10⁻⁶ |
| ladderN32 (200) | 574,979 | 0.357 | 0.371 | 0.471 [0.375, 0.590] | 0.233 | 4.0×10⁻⁸ |
| census (256) | 571,612 | 0.359 | 0.373 | 0.476 [0.380, 0.597] | 0.235 | 2.0×10⁻⁸ |
| tfixexp1 (150) | 571,612 | 0.359 | 0.373 | 0.476 [0.380, 0.597] | 0.235 | 2.3×10⁻⁸ |
| tfixexp2 (250) | 568,860 | 0.361 | 0.375 | 0.479 [0.382, 0.600] | 0.235 | 8.7×10⁻⁹ |

Maximum |ΔRE| = 0.007 (drop 'new'); maximum per-rung |Δθ*| = 0.018 (the 0.0025 rung, whose own interval is [0.29, 0.79]); the profile minimum sits at s_d = 0.005 in all twelve variants; homogeneity is rejected in all twelve (max Q_p = 2.1×10⁻⁶, isotonic max 6.6×10⁻⁹). The reason the anchor cannot move the result: the ridge scales with the anchor median through the anchor elasticity (deposited −0.5 to −1.3 at the mid rungs, −3.7 at the 0.0025 rung), and the largest anchor perturbation any leave-one-out produces is the +0.9% median shift from dropping the 2,000-run arm; the trough (0.36–0.37 at s_d = 0.005–0.01 versus 0.62–0.63 at the two ends, a factor 1.7) is a factor-of-two structure that a 1% common-mode shift cannot flatten.

**Track B — Monte-Carlo placement propagation** (`placement_mc_summary.json`, `placement_mc_draws.parquet`, script `run_placement_mc.py`; RNG seeds 13,560,000–13,560,003 for the four tissues, 13,565,000–13,567,000 for the adversarial variants, 13,568,000–13,568,105 for the Track-A bootstraps — all inside the assigned block 13,560,000–13,569,999). Per draw: mud and s_d log-uniform in the tissue's stated box; ridge ~ exp(N(ln 0.4748, τ = 0.2345)) (the RE prediction-interval logic, 95% span 0.30–0.75 for the mean draw, the stated 0.21–1.09 being the interval on a new condition); shape multiplier log-uniform on [1.3, 3]; margin = log₁₀[(mud/s_d) / (ridge × shape)], fluid = margin > 0. Convention asserted throughout: θ = U_d/s_d = mud/s_d with U_d = mud (no factor 2).

| tissue | P(fluid) | median margin (dec) | 5th–95th pct (dec) | P(fluid), s_d in-range | P(fluid), s_d < 0.0025 (extrapolated) | fraction of draws with s_d < 0.0025 |
|---|---:|---:|---:|---:|---:|---:|
| Colonic crypt | 0.9999 | +1.43 | +0.58 … +2.29 | 0.9997 | 1.0000 | 0.538 |
| Epidermis (sun-exposed) | 1.0000 | +2.13 | +1.28 … +2.98 | — | — | 0.538 |
| HSC (blood) | 1.0000 | +2.20 | +1.35 … +3.05 | — | — | 0.538 |
| Hypermutator tumour | 1.0000 | +2.05 | +1.30 … +2.79 | — | — | 0.267 |

54% of colon draws fall below the sampled s_d range (s_d < 0.0025), as flagged in the brief — but the extrapolated left arm is the *most* fluid subset (P = 1.0000, in-range 0.9997), not the vulnerable one: within a log-uniform box θ = mud/s_d is maximal exactly where s_d is minimal. The 18 non-fluid colon draws out of 200,000 (adversarial run, seed 13,566,000) sit at the opposite corner — s_d ≈ 0.009–0.010, mud ≈ 0.016–0.020, shape ≈ 2.5–3.0, ridge ≈ 0.64–0.82 — i.e. three simultaneous tail events, which is precisely why the single-corner joint worst case (+0.13 dec at that corner) reads as 'marginal' while the propagated probability does not.

## Adversarial checks (`placement_mc_adversarial.json`)

I tried to break the colon claim four ways; none succeeded.

1. **Dependence between mud and s_d.** The independent log-uniform box could hide a corner-populating correlation. Perfectly *anti*-monotone coupling (low mud always paired with high s_d — the worst case for θ) still gives colon P(fluid) = 0.9989 (median +1.43, 5th pct +0.33); comonotone coupling gives 1.0000.
2. **s_d-dependent ridge instead of the s_d-independent RE draw.** Interpolating the measured (rising) profile in log s_d with flat extrapolation, its 95% upper curve, the legacy-protocol profile (0.756 at 0.0025), and the fitted quadratic-in-log(s_d) extrapolated to s_d = 5×10⁻⁴ (ridge 0.78 there) all leave colon P(fluid) at ≥ 0.998. The rising left arm cannot catch the colon because the same small s_d that raises the ridge raises θ = mud/s_d faster.
3. **Wider shape correction.** Extending the multiplier to log-uniform [1.3, 5] gives colon 0.998 (median +1.32).
4. **Break-point.** Multiplying the entire ridge × shape barrier by k, colon P(fluid) reaches 0.9 only at k = 5.6 and 0.75 at k = 11.7; the other tissues need k = 27–39 to reach 0.9. No measured or extrapolated ridge in any deposit is within a factor of five of this.

The result is also insensitive to the uncertainty model chosen for the primary ridge draw: adding the SE of the pooled log-mean in quadrature to τ (prediction interval on a new condition) changes colon from 0.9999 to 0.9998; fixing the ridge at the point estimate, or fixing the shape at 2, changes nothing beyond the fourth decimal (six variants in `placement_mc_summary.json`).

## Caveats and one deposit defect

- **Deposited factor-2 artefact.** `data/tissue_placement_joint.csv` in the round-8 bundle uses ratio = 0.5·mud/s_d and therefore reports the colon frozen at the joint worst case (−0.24 dec, `fluid_at_worst = False`); every one of its numbers is reproduced with 2× larger ratios under the canonical U_d = mud convention, which puts the colon joint worst case at +0.13 dec (fluid). Any 'colon crosses to frozen' phrasing derived from that file is this artefact and must not enter the manuscript.
- **The result is only as good as the tissue input boxes.** The propagation treats the stated mud/s_d ranges as the full support of log-uniform priors. The colon claim would only be in danger if colonic-crypt U_d/s_d could plausibly reach ≈ 0.9–1.6 (i.e. the freeze barrier's neighbourhood): with s_d capped at 0.01 that requires mud ≤ 0.016 — the very floor of the stated range — *and* the largest shape correction *and* an upper-tail ridge simultaneously. If a reviewer supplies a lower colon mud than 0.016 SNV·division⁻¹·(deleterious fraction), the analysis should be rerun with that box; the machinery is a 0.4-second script.
- **The RE draw is s_d-agnostic by construction**; check 2 above shows the s_d dependence of the ridge does not create a hidden failure mode for these tissues, because the small-s_d rise of the ridge is dominated by the θ ∝ 1/s_d rise of the placement.
- **No new simulation runs** were needed or executed (runs_executed = 0); this is a pure analysis track over the deposited pool. All RNG seeds (Monte-Carlo draws and bootstrap replicates) are recorded and lie inside the assigned block 13,560,000–13,569,999. Disjointness verified against SEEDS.md: the contiguous deposited blocks nearest ours are 12.90–12.92M and 12.64M; two SEEDS.md rows (`bd_results`, `tail_ensemble`) have min–max spans that nominally cover 13.56M, but they are hashed seed sets and their actual integers (2,900 and 1,600 seeds, read from the deposited parquets) contain zero values in our block; the analysis pool itself contains no seed in our block (max 12,921,299).

## Recommendation for the manuscript

Report the placements as probabilities, not as worst-case corners: "propagating the full input ranges, the resampled ridge (τ = 0.23) and the 1.3–3× shape correction, all four tissues lie on the fluid side with probability > 0.999 (colonic crypt 0.9999, median margin 1.4 decades, 5th percentile +0.6)". Retire both the '+0.13 dec joint worst case' framing (it is a 10⁻⁴-probability corner, and the deposited CSV that carries it is on the stale factor-2 convention) and the word 'marginal' for the colon.
