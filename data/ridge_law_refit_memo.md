# Round 8 — ridge-law refit and criterion characterisation (memo)

**Verdict.** (A) FOR the traced load-law with a caveat: after normalising every point by the pooled ≥600-run neutral anchor (3,056 runs, median 573,328) instead of the two small legacy anchors, the Haigh model k = 1 (s* ∝ Λ^{1/2}, i.e. U_d/s_d = const) **remains the selected model** — ΔAIC vs the free fit is −0.14 (F = 1.57, p = 0.26) — while k = 2 (γ = 1/3, ΔAIC +10.1), the constant-effect-size claim γ = 0 (ΔAIC +28.5) and the pure-rate threshold k = 0 (γ = 1, ΔAIC +30.8) all remain rejected. The re-anchored free exponent is **γ = 0.464 ± 0.028, i.e. k = 1/γ − 1 = 1.15 ± 0.13 (OLS 95% CI 0.87–1.53)**, versus the deposited γ = 0.480, k = 1.08. The exponent is **first-order insensitive to a common anchor level** (dγ/d ln a = 0.033: a 5.6% common shift moves γ by 0.002, k by 0.01) but is **sensitive to anchor consistency between ladder families**: the deposited fit's two legacy anchors were mutually inconsistent (fixed-Λ ladders anchored 5.8% low, regime grid 1.2% high), and that *differential* shift rotated the fit by Δk = +0.07 — nine times the common-mode sensitivity. (B) The freeze/flow "constant" is a **level-convention-dependent** number: at s_d = 0.01 the ridge is 1.21 / 0.37 / 0.20 for crossing levels 0.25 / 0.5 / 0.75 (0.76 / 0.37 / 0.23 for 0.3 / 0.5 / 0.7 — the reviewer's 0.23–0.76 envelope), and the RE-pooled constant runs 2.05 / 0.48 / 0.17; the level-0.25 pooled value is shelf-dominated and not data-supported. (C) The new s_d = 0.00125 rung (864 loaded runs, fresh 600-run anchor, zero censored at the 4×10⁷ cap) shows the trough's left arm **continues rising**: no half-neutral-median crossing exists within the sampled θ ≤ 1.6 under any anchor or estimator (median ratio bottoms out at 0.52–0.55), the crossing extrapolates to θ ≈ 2.3–3.5 (own-anchor bootstrap 2.0–10.3), and P(θ\*(0.00125) > θ\*(0.0025) = 0.63) = 1.00. Against the paper's implicit "shelf → constant" reading, the low-s_d arm does not flatten.

## What was tested and how (like-for-like with the deposit)

The deposited traced ridge law was first reproduced bit-for-bit before anything was changed. Its 8 points are the union of two ladder families on the 192×32 Moore geometry (dt = 2400, μ̄ = 1.16×10⁻⁵): (i) three fixed-Λ ladders (Λ = 10⁻⁵, 3×10⁻⁵, 10⁻⁴; ridge_law.parquet group T1, 5 s_d rungs × 150 runs) normalised by the legacy N32 anchor (200 runs, median 540,332), and (ii) five regime-grid fixed-Λ rows (Λ = 2.5×10⁻⁴ … 4×10⁻³, 4 s_d cells each) normalised by the grid's own legacy anchor (200 runs, median 580,331.5). The estimator is: s\* = the s_d at which the median-ratio crosses ½ scanning up in s_d, log-x / linear-y interpolation. With those legacy anchors I recover γ = 0.4799 ± 0.0263, R² = 0.982, k = 1.084 ± 0.114 and the deposited ΔAIC table (γ = ½: −1.26; γ = ⅓: +12.54; γ = 0: +30.24; γ = 1: +31.51) to two decimals. This confirms the k models on the s\*–Λ plane map as γ = 1/(k+1): **k = 0 ↔ γ = 1** (U_d = const, pure rate threshold), **k = 1 ↔ γ = ½** (Haigh U_d/s_d = const), **k = 2 ↔ γ = ⅓** (U_d/s_d² = const).

**The pooled anchor.** All 8 points share one geometry and protocol, so the pooled-anchor protocol assigns them **one** common anchor: the deposited round-7 2,856-run 192×32 pool (median 572,583.5; the legacy N32 200-run set is one of its five components), extended with the regime grid's own 200-run neutral set — which was not in that pool but is fully consistent with it (Kruskal–Wallis across all six components p = 0.965; grid vs pool Mann–Whitney p = 0.962) — giving 3,056 runs, median 573,328. Relative to this pooled anchor the T1 legacy anchor was **5.76% low** and the grid legacy anchor **1.22% high**.

## A. k-model refit

| model | γ | ΔAIC vs free (legacy anchors) | ΔAIC vs free (pooled anchor) | verdict |
|---|---|---|---|---|
| free fit | 0.480 ± 0.026 → **0.464 ± 0.028** | 0 | 0 | — |
| **k = 1 (Haigh U_d/s_d)** | ½ | −1.26 | **−0.14** (F 1.57, p 0.26) | **selected** |
| k = 2 (U_d/s_d²) | ⅓ | +12.54 | +10.10 (F 21.2, p 0.004) | rejected |
| s\* = const (paper) | 0 | +30.24 | +28.52 | decisively rejected |
| k = 0 pure rate (U_d const) | 1 | +31.51 | +30.76 | decisively rejected |

Re-anchoring moved the low-Λ (T1) points **up** by ×1.05–1.08 and the grid points **down** by ×0.98–0.99 (their anchors were offset in opposite directions), rotating the fit from γ = 0.480 to 0.464 (k 1.08 → 1.15). The ordering of the four fixed models is unchanged, and k = 1 keeps the best AIC; the margin between k = 1 and the free fit shrinks slightly (−1.26 → −0.14), i.e. the free fit sits marginally further below γ = ½ than before, but not detectably (F-test p = 0.26; OLS 95% CI on k is 0.87–1.53).

**Error-currency caveat (report both).** A run-level bootstrap (B = 3000, resampling every rung of every ladder and drawing one shared anchor per replicate) gives γ = 0.463 [0.439, 0.486], k = 1.16 [1.06, 1.28], with k = 1 preferred over k = 2 in 99.97% of replicates and P(γ < ½) = 0.997. This interval **excludes** k = 1 whereas the OLS interval includes it. The bootstrap propagates only simulation noise (median T_fix scatter), which is tiny here; the OLS residual scatter of the 8 points about the line is the model uncertainty, and it dominates. The bootstrap therefore establishes that the *point estimate* γ ≈ 0.46 is sharply determined, not that k = 1 is excluded; the model-comparison currency is the OLS/AIC comparison, under which k = 1 stands. The honest one-line statement: *k = 1 is the selected single power law (ΔAIC −0.1; OLS k CI 0.87–1.53), with a point estimate k ≈ 1.15 that the simulation noise resolves as slightly above 1.*

**A structural finding the single power law hides.** The two ladder families carry *different* local exponents — T1 (Λ ≤ 10⁻⁴): γ = 0.418 ± 0.004; grid (Λ ≥ 2.5×10⁻⁴): γ = 0.588 ± 0.063 — under **both** anchor conventions (legacy: 0.419 vs 0.587). So the split is a property of the two Λ regimes, not of re-anchoring. With n = 8 it is not significant (common- vs separate-slope F-test p = 0.15; log-quadratic curvature p = 0.087), though ΔAIC mildly prefers two slopes (−3.6). The k = 1 selection is a compromise line between two internally-consistent subsets; a reviewer probing for scale-dependence of k would find this and it should be pre-empted (report it as "no significant curvature at n = 8; low-Λ arm k ≈ 1.4, high-Λ arm k ≈ 0.7").

**Anchor sensitivity of the exponent — quantified.** (i) *Analytic*: under a common anchor multiplier a, each point moves by its crossing elasticity ∂ln s\*/∂ln a (0.78–1.86 across points, mean 1.14); the induced ∂γ/∂ln a is the OLS projection of those elasticities onto log Λ = **0.036**. (ii) *Numeric* sweep of a common multiplier a ∈ [0.90, 1.10] on the pooled anchor: **∂γ/∂ln a = 0.033**; a full 5.6% common shift moves γ by 0.0018 and k by ≈ 0.01 — the exponent is first-order anchor-**level** insensitive (it changes the prefactor by ~+6%, sliding points along the fit). (iii) *Actual re-anchoring*: Δγ = −0.016, Δk = +0.07 — ~9× the common-mode figure — because the deposited anchors were **inconsistent across families** (+5.8% / −1.2%). Statement for the paper: *the traced-ridge exponent is insensitive to the neutral-anchor level (∂k/∂ln a ≈ 0.15, i.e. |Δk| ≤ 0.01 for a 5.6% anchor error) but requires the same anchor convention for every ladder; mixing per-family legacy anchors biased k low by 0.07.*

## B. Crossing-level envelope

Recomputed on the deposited 12,756-run loaded pool with the 2,856-run anchor (loglin scheme, run-level bootstrap with a shared anchor draw); level 0.5 reproduces the deposited per-s_d table exactly (0.6325 / 0.3582 / 0.3725 / 0.5288 / 0.6215) and the RE-pooled 0.475.

| crossing level | s_d = 0.01 ridge [95% CI] | RE-pooled constant, 5 rungs [95% CI] | RE-pooled, 4 rungs (shelf excl.) |
|---|---|---|---|
| 0.25 | 1.21 [0.92, 1.51] | 2.05 [0.77, 5.46] (I² = 1.00) | 1.22 [0.87, 1.71] |
| 0.30 | 0.76 [0.69, 0.86] | 1.24 [0.89, 1.73] | 0.79 [0.71, 0.87] |
| 0.40 | 0.54 [0.46, 0.57] | 0.74 [0.60, 0.92] | 0.62 [0.54, 0.71] |
| **0.50 (canonical)** | **0.37 [0.33, 0.43]** | **0.48 [0.38, 0.60]** (I² = 0.91) | 0.46 [0.36, 0.58] |
| 0.60 | 0.30 [0.25, 0.34] | 0.34 [0.27, 0.45] | 0.35 [0.26, 0.48] |
| 0.70 | 0.23 [0.19, 0.26] | 0.22 [0.19, 0.25] (I² = 0.00) | 0.22 [0.19, 0.26] |
| 0.75 | 0.20 [0.17, 0.24] | 0.17 [0.13, 0.22] | 0.18 [0.15, 0.22] |

**Envelope to quote.** At the best-measured rung (s_d = 0.01): **0.20 (level 0.75) – 0.37 (canonical) – 1.21 (level 0.25)**; on the reviewer's 0.3–0.7 convention, **0.23–0.76** (exactly the cited envelope: it is levels 0.7 and 0.3 at s_d = 0.01, not an inter-study spread). The RE-pooled constant spans 0.17–0.48–2.05 across levels 0.75–0.5–0.25, but the level-0.25 pooled value (and to a lesser extent 0.30) is **not data-supported**: it is driven by the s_d = 0.0025 rung, whose median-ratio curve reaches 0.24 only at its last rung (θ = 16, crossing 14.5, only 71% of bootstraps defined). Excluding that shelf rung, the RE constant at 0.25 is 1.22. Recommendation: quote the ridge as a level-indexed scale (θ\*(L) at the reference rung), state the canonical L = ½ number, and give the 0.3–0.7 (or 0.25–0.75) envelope **at s_d = 0.01** rather than the pooled constant, whose low-level end diverges. Softness (crossing_softness.csv): at L = ½ the mid-s_d curves cross sharply (in-band width ≤ 0.07 decades in θ), while the s_d = 0.0025 shelf grazes ½ over 0.3 decades — the shelf, not the level convention, is the fragile element.

## C. Trough left-arm top-up at s_d = 0.00125

New ensemble: 600 fresh neutral anchor runs (seeds 13,570,000–13,570,599) + 9 θ rungs {0.1 … 1.6} × 96 runs (seeds 13,575,000 + 200j + [0..95]), 192×32 Moore, identical trough protocol, frame cap 4×10⁷. **1,464/1,464 completed, 0 errors, 0 censored** (max T_fix 3.33×10⁶ = 8.3% of cap). Seeds are inside the track block, disjoint per arm, and disjoint from every deposited seed set (checked against exact seed lists including the wide-span bd_results and tail_ensemble tables). The production kernel was verified bit-for-bit against two deposited runs before launch.

| θ | median T_fix | ratio (own 600-run anchor) | ratio (pooled 3,056-run anchor) |
|---|---|---|---|
| 0.10 | 487,819 | 0.910 | 0.852 |
| 0.14 | 552,202 | 1.030 | 0.964 |
| 0.20 | 545,516 | 1.017 | 0.953 |
| 0.28 | 465,328 | 0.868 | 0.813 |
| 0.40 | 398,763 | 0.744 | 0.696 |
| 0.57 | 422,275 | 0.787 | 0.738 |
| 0.80 | 365,167 | 0.681 | 0.638 |
| 1.13 | 320,058 | 0.597 | 0.559 |
| 1.60 | 296,979 | **0.554** | **0.519** |

The ratio declines monotonically (Spearman ρ = −0.93, p = 2×10⁻⁴; d ln(ratio)/d ln θ = −0.22 ± 0.03) but **never reaches ½ within the sampled ladder** — under the own anchor, the pooled anchor, and all three estimators (loglin, isotonic: undefined; logistic: 2.5 by extrapolation). Bootstrap (B = 4000, resampling rungs and anchor): P(no crossing within θ ≤ 1.6) = 0.965 (own anchor) / 0.79 (pooled); log-linear extrapolation of the crossing = **3.5 [2.0, 10.3]** (own anchor) / 2.6 [1.7, 6.3] (pooled); **P(θ\*(0.00125) > 0.63) = 1.000** under both. **Direction of the extended left arm: it continues rising** — θ\* goes 0.36–0.37 (s_d = 0.005–0.01) → 0.63 (s_d = 0.0025) → >1.6 (s_d = 0.00125). The s_d = 0.0025 shelf is not a floor; at s_d = 0.00125 the freeze/flow criterion leaves the θ ≤ 1.6 window. (To place the crossing itself, a future ladder would need rungs to θ ≈ 5–10 at this s_d.)

**Anchor-scatter observation (relevant to the ≥600-run rule).** The fresh 600-run anchor came out at median 536,336 — **6.3% below** the 2,856-run pool (572,583.5), statistically consistent with it (MWU p = 0.068; KW across all seven neutral ensembles p = 0.63; median 95% CI [493.5k, 577.7k]) yet the same magnitude and sign as the legacy N32 offset that motivated this re-anchoring. Because neutral T_fix is heavy-tailed (mean/median = 1.27), a 600-run anchor still carries ~±4–7% median scatter. The ≥600-run floor is therefore necessary but not sufficient to remove ~5% anchor systematics; **the pooled multi-thousand-run anchor is what does**, and the s_d = 0.00125 conclusion is anchor-robust (holds under both).

## Data provenance and caveats
- `ridge_law.parquet` and `ridge_law_summary.json` are **not inside the round-8 kernel .tgz**; the deposited artifact-store copies (ids `136daa15`, `fdc51a8d`) were used — an exact substitution, disclosed here. The production kernel's dependency `evolution_fast4.py` (also absent from the .tgz) was likewise taken from the artifact store; kernel provenance was verified by bit-for-bit reproduction of deposited runs (seed 12,900,000 → T_fix 1,129,978; seed 12,910,000 → 335,447).
- The 8-point law spans two ladder families with different local exponents (k ≈ 1.4 vs 0.7, n.s. at n = 8); a single power law is a summary, and Λ-range extension would decide whether k drifts with Λ.
- The run-level bootstrap CI on k [1.06, 1.28] excludes 1 while the OLS CI [0.87, 1.53] includes it — a difference in error currency (simulation noise vs point scatter), stated above; both are reported.
- The new-rung crossing is unbracketed within θ ≤ 1.6; its location (2.3–3.5) is an extrapolation and is reported as such, with the in-sample statement (min ratio 0.52–0.55 at θ = 1.6, ρ = −0.93) as the primary result.

## Files
`ridge_law_refit_summary.json` (all numbers, both anchors, all levels, bootstrap detail), `crossing_level_envelope.csv` (θ\* by level × s_d rung and RE-pooled, with CIs), `crossing_softness.csv` / `crossing_level_bracketing.csv` (level-band definedness), `low_sd_extension_runs.parquet` (1,464 new per-run rows, `seed` column, `censored` flag), `fig_ridge_law_refit.png` (4 panels), `ridge_law_refit.py` (estimators/fits), `sim/run_lowsd_extension.py` (driver).
