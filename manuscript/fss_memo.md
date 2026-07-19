**Verdict: FOR the paper's claim: at s_d = 0.01 the crossover load θ* plateaus rather than drifting with lattice length (half-drop θ* across L_long = 96: 0.70, 192: 0.58, 384: 0.52, 768: 0.60, 1536: 0.61; largest-minus-smallest-L difference -0.087 [-0.251, +0.083], no monotone trend; constant fit preferred over power-law+plateau, extrapolated large-L θ* = 0.550 [0.530, 0.574]).**

# Finite-size / boundary convergence of the s_d = 0.01 crossover

## Question
Does the crossover load $\theta^* = U_d^*/s_d$ (freeze/flow ridge at $s_d = 0.01$) drift with lattice size $L$, or plateau? Does a walled transverse boundary move it outside the periodic interval? All numbers below are fields of `fss_summary.json`; the design was pre-registered in `fss_design.json` before any production run.

## Data
* 20,323 new runs across 8 cells (own neutral anchor per cell), seeds 14,000,000–14,031,399 (20,323 unique), 0 worker errors, 4 censored (frame-cap) runs.
* Protocol identical to the deposited 192x32 s_d = 0.01 pool: Moore stencil, dt = 2400, avg = 1.16e-5, sb = 0.1, mub = 0, sampling 1e4, exact-fixation early stop; theta ladder {0.05 … 10} (10 rungs 0.2 … 10 on the largest lattice). Kernel bit-exactly replicated four deposited pool seeds (`kernel_check.json`) before running.

## Per-geometry crossover (run-level bootstrap, B = 2000)

| cell | geometry | boundary | n_loaded | n_anchor | half-drop θ* [95% CI] | ratio=1/2 θ* [95% CI] | ratio=0.4 | ratio=0.6 |
|---|---|---|---:|---:|---|---|---|---|
| G96 | 96x32 | periodic | 2,800 | 400 | 0.700 [0.624, 0.809] | 0.611 [0.375, 0.705] | 0.973 | 0.334 |
| G192 | 192x32 | periodic | 2,800 | 400 | 0.583 [0.492, 0.643] | 0.424 [0.276, 0.471] | 0.499 | 0.268 |
| G384 | 384x32 | periodic | 2,800 | 400 | 0.516 [0.491, 0.543] | 0.445 [0.408, 0.472] | 0.488 | 0.406 |
| G768 | 768x32 | periodic | 840 | 100 | 0.598 [0.574, 0.652] | 0.519 [0.464, 0.562] | 0.558 | 0.482 |
| G1536 | 1536x32 | periodic | 122 | 61 | 0.613 [0.478, 0.757] | 0.543 [0.275, 0.617] | 0.571 | 0.516 |
| S16 | 192x16 | periodic | 2,800 | 400 | 0.398 [0.365, 0.628] | 0.348 [0.283, 0.453] | 0.725 | 0.275 |
| S64 | 192x64 | periodic | 2,800 | 400 | 0.706 [0.624, 0.763] | 0.468 [0.391, 0.524] | 0.580 | 0.365 |
| W32 | 192x32 | **walled** | 2,800 | 400 | 0.498 [0.464, 0.582] | 0.381 [0.294, 0.442] | 0.474 | 0.286 |

## Convergence in L_long (L_short = 32)

**Half-drop (convention-free primary).** θ* by L_long: 96: 0.700 [0.624, 0.809], 192: 0.583 [0.492, 0.643], 384: 0.516 [0.491, 0.543], 768: 0.598 [0.574, 0.652], 1536: 0.613 [0.478, 0.757].
  Constant fit θ*_∞ = 0.562 [0.542, 0.584] (AICc 8.516); power-law-plus-plateau θ*_∞ = 0.550 [0.530, 0.574], exponent b = 4.293 [2.810, 5.000] (AICc 23.242). ΔAICc(power-law − constant) = 14.726; nested F-test p = 0.523. Bootstrap sign test: fraction of replicates with all four successive differences negative 0.000, positive 0.000; log–log slope -0.035 [-0.115, 0.046].
  Largest-minus-smallest-L difference (paired bootstrap): -0.087 [-0.251, 0.083], P(diff ≤ 0) = 0.888.

**Ratio = 1/2 crossing (legacy).** θ* by L_long: 96: 0.611 [0.375, 0.705], 192: 0.424 [0.276, 0.471], 384: 0.445 [0.408, 0.472], 768: 0.519 [0.464, 0.562], 1536: 0.543 [0.275, 0.617].
  Constant fit θ*_∞ = 0.474 [0.440, 0.494] (AICc -0.996); power-law-plus-plateau θ*_∞ = 0.471 [0.439, 5.057], exponent b = 5.000 [0.020, 5.000] (AICc 23.828). ΔAICc(power-law − constant) = 24.824; nested F-test p = 0.791. Bootstrap sign test: fraction of replicates with all four successive differences negative 0.000, positive 0.029; log–log slope -0.005 [-0.205, 0.170].
  Largest-minus-smallest-L difference (paired bootstrap): -0.068 [-0.358, 0.199], P(diff ≤ 0) = 0.564.

## L_short axis at fixed L_long = 192

* Half-drop: θ* at L_short 16/32/64 = 0.398 [0.365, 0.628], 0.583 [0.492, 0.643], 0.706 [0.624, 0.763]; slope d log θ*/d log L_short = 0.343 [0.103, 0.494]; θ*(64) − θ*(16) = 0.308 [0.049, 0.368] (P(diff ≤ 0) = 0.011).
* Ratio = 1/2: θ* at L_short 16/32/64 = 0.348 [0.283, 0.453], 0.424 [0.276, 0.471], 0.468 [0.391, 0.524]; slope d log θ*/d log L_short = 0.210 [-0.003, 0.383]; θ*(64) − θ*(16) = 0.120 [-0.008, 0.204] (P(diff ≤ 0) = 0.030).

## Walled transverse boundary at 192x32

* half-drop: walled θ* = 0.498 [0.464, 0.582] vs periodic θ* = 0.583 [0.492, 0.643]; walled − periodic (paired bootstrap) = -0.085 [-0.150, 0.045], P(diff ≤ 0) = 0.878; walled point estimate outside the periodic 95% CI: False.
* ratio = 1/2: walled θ* = 0.381 [0.294, 0.442] vs periodic θ* = 0.424 [0.276, 0.471]; walled − periodic (paired bootstrap) = -0.043 [-0.140, 0.141], P(diff ≤ 0) = 0.599; walled point estimate outside the periodic 95% CI: False.
* Neutral anchor median, walled/periodic = 1.013 (Mann-Whitney p = 0.292).

## Replication of the deposited 192x32 pool (fresh seeds)

* half-drop: this campaign's G192 θ* = 0.583 [0.492, 0.643] vs deposited pool θ* = 0.555 [0.477, 0.627]; difference = 0.028 [-0.094, 0.121] (P(diff ≤ 0) = 0.427).
* ratio = 1/2: this campaign's G192 θ* = 0.424 [0.276, 0.471] vs deposited pool θ* = 0.373 [0.324, 0.430]; difference = 0.051 [-0.116, 0.117] (P(diff ≤ 0) = 0.354).
* Rung-wise Mann-Whitney across the 14 shared rungs plus the anchor: Fisher-combined p = 0.356.

## Verdict fields

```json
{
 "half_drop": "PLATEAU",
 "level_half": "PLATEAU",
 "extrapolated_large_L_limit_half_drop": [
  0.5500595071116252,
  0.5302834759857111,
  0.573864209187521
 ],
 "extrapolated_large_L_limit_level_half": [
  0.4710234496130577,
  0.4391996905949408,
  5.05709916890059
 ],
 "constant_fit_half_drop": [
  0.5618101467974473,
  0.5424541495016157,
  0.5842345970999417
 ],
 "constant_fit_level_half": [
  0.4741605252183401,
  0.4400516361689219,
  0.4942122770523648
 ],
 "walled_shifts_outside_periodic_CI": false
}
```

## Caveats
1. The two largest lattices ran at reduced replication under the design's cost clause (768x32: 60 loaded runs per rung and 100 neutral anchors; 1536x32: 12 loaded runs per rung and 61 anchors, 4 of which are right-censored at the 1e8-frame cap); their bootstrap intervals are correspondingly wide, and the 1536x32 point carries the least weight (inverse-variance) in the convergence fits. The reductions are prefix-truncations of the pre-registered seed blocks, recorded in fss_design.json (deviations_executed_during_run) before any large-lattice loaded result existed.
2. The half-drop estimator shows a real dependence on the SHORT axis at fixed L_long = 192 (θ*: L_short 16 → 32 → 64 = 0.398 → 0.583 → 0.706; S64 − S16 = 0.308 [0.049, 0.368], P(≤0) = 0.011); the plateau claim is about L_long at fixed L_short = 32 and should be stated as such. The level-1/2 estimator's short-axis dependence is weaker (S64 − S16 = 0.120 [-0.008, 0.204], P(≤0) = 0.030).
3. The level-1/2 (legacy) estimator's power-law+plateau extrapolation is unconstrained on this ladder (θ*_inf = 0.471 [0.439, 5.057]); use its constant fit (0.474 [0.440, 0.494]) as the operative large-L value for that estimator.
4. Host contention from concurrent campaigns inflated wall-clock times but not results: every run is single-threaded with a pre-registered per-run seed, and the fresh 192x32 cell replicates the deposited pool (Fisher-combined rung-wise Mann-Whitney p = 0.36; anchor p = 0.61).

## Runtime
20,323 runs total (0 errors, 4 censored), summed per-run wall time 322.0 core-hours executed on the shared 64-core host over ~9.2 h wall (2026-07-18 18:30 to 2026-07-19 03:40 UTC), at 1.8-2.4x per-run slowdown from concurrent sibling campaigns; reduction (2,000-resample run-level bootstrap over 8 cells) took 34 s.
