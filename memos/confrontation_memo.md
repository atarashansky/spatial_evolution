# Track D — Data confrontation on honest statistics

## Bottom line (against the manuscript's current claim)

**The two-clone strip kernel does not reproduce control mouse-oesophagus clone-size dynamics, and the loaded-interface model cannot be distinguished from the neutral null in any published clone-size arm.** The panel's fact check is confirmed and worse than stated: all 24 published data-vs-kernel comparisons formally reject (KS 2.7–14× the α = 0.05 critical value; asymptotic p ≤ 3×10⁻³⁴). An explicit measurement layer, fitted on the 1-month control arm and predicting out of sample, closes the 1-month and part of the 3-month gap but exposes a structural failure at 6–12 months: the kernel coarsens as ⟨area⟩ ∝ t^0.80 while the tissue's median clone area saturates after 3 months (∝ t^0.31). Load never rescues the fit. The manuscript's kernel-vs-kernel "load signal" (KS 0.04–0.09) and its 600–3,200-clone power prescription are a mean-rescaling discretization artifact and must be withdrawn. On a correct likelihood-ratio test, clone-size shape reaches 80% discrimination power only at 12 months and s_d ≥ 5×10⁻³ (≈730 clones/arm at s_d = 10⁻², ≈18,000 at 5×10⁻³); everywhere else power stays below 80% up to 51,200 clones/arm (a weak sub-threshold signal exists — max 0.52 at s_d = 10⁻²/6 mo and 0.23 at s_d ≤ 2×10⁻³/3 mo — but never a designable experiment). The discriminating observable in these data is the DEN-vs-control clone enlargement (8.6× at 12 months), which is a driver-selection phenotype the kernel does not produce.

## 1. Formal tests on the manuscript's own ensembles (Task 1)

Reproduced from the archived `lineage_sims.pkl` (48 runs per condition, the ensembles behind the caption's "KS = 0.16 both"), mean-rescaled clone sizes as in the manuscript, α = 0.05 asymptotic two-sample critical value c(α)·√((n+m)/nm), c = 1.358.

| control arm | n_data | n_sim (48 runs, neutral) | D (neutral) | D (loaded s_d=10⁻²) | critical | D/critical | p (neutral) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 month | 15,865 | 15,419 | 0.217 | 0.216 | 0.0154 | 14.1× | < 10⁻³⁰⁰ |
| 3 months | 4,152 | 6,243 | **0.164** | **0.178** | **0.027** | **6.0–6.6×** | **6×10⁻⁵⁹** |
| 6 months | 2,474 | 3,462 | 0.174 | 0.163 | 0.036 | 4.6–4.9× | 2×10⁻³⁸ |
| 12 months | 3,485 | 1,904 | 0.331 | 0.281 | 0.039 | 7.0–8.5× | 2×10⁻¹²⁰ |

- Every one of the 24 comparisons (control and DEN × 4 ages × 3 kernels) rejects; D/critical ranges 2.7–14.2×, and the *largest* p-value in the control set is 3.4×10⁻³⁴. The sentence "both kernels fit the observed 3-month distribution equally well (KS = 0.16 both)" is false as a goodness-of-fit statement: at n = 4,152 the critical value is 0.027 and both kernels are rejected at p < 10⁻⁵⁸. What is true is only the second clause — neither kernel fits *better* than the other.
- **Arm accounting corrected.** `colom2020_clones.parquet` has 83,519 rows but 61,737 unique clones. The Suppl. Table 14 "YFP 30/90/180/360-day" arms (21,782 rows) are value-for-value identical to the Suppl. Table 6 YFP-DEN arms — a double count in the parquet, not additional data (the memo's "93,968 clones" and the Methods' "59,310" both need correcting). The confrontation uses Suppl. Table 6 only: control n = 15,865 / 4,152 / 2,474 / 3,485 and DEN n = 5,682 / 539 / 281 / 188 at 1/3/6/12 months (10-day arms: control 11,552, DEN 15,092, not modelled). DN-Maml1 (1,862) and confetti (565) rows are not size-distribution arms and are excluded.

## 2. Measurement-layer forward model (Task 2)

New seeded ensembles (this track): 2,000 runs per condition, 12,000 runs total, 512 × 256 strip, snapshots at frames 296/896/1,796/3,592 = 1/3/6/12 months, neutral + loaded (μ_d = 0.4; s_d ∈ {5×10⁻⁴,10⁻³,2×10⁻³,5×10⁻³,10⁻²}, Λ = 2×10⁻⁴–4×10⁻³). Checkpointed as `ensemble_D_all.pkl.gz`.

Layer (per simulated clone): induction thinning to a fitted per-cell density with fusion of 8-adjacent labelled clones (union of the retained-clone adjacency graph); projected footprint A = c·F cells → µm², F ~ LogNormal(ln μ_F, σ_F²) per clone (YFP-membrane readout marking a variable cortical fraction; the smallest recorded clone, 50.27 µm², is age-invariant and sub-cellular); 1.29-µm² pixel rounding; drop below the 50.27-µm² detection floor. Free parameters (p_ind, μ_F, σ_F) fitted by differential evolution on the **1-month control arm only**, absolute µm², KS objective; then **frozen**.

- **Fit (1 mo, n = 15,865):** μ_F = 59.0 µm² cell⁻¹ [bootstrap 95% CI 56.4–60.9], σ_F = 1.085 [1.06–1.13], p_ind = 0.010 [0.0009–0.018, unidentified], residual KS = 0.030 [0.028–0.038] (raw kernel: 0.216). The 59-µm² footprint independently matches mouse-oesophagus basal-cell area (~50–70 µm²) — the layer is anatomically sensible, not a free-form warp.
- **Ablation (which components carry the fit; 1-mo KS):** full 0.031; fusion off 0.030; pixel rounding off 0.031; detection floor off 0.083; **footprint dispersion off 0.161**. The fit is carried by per-clone footprint dispersion and the detection floor; fusion and pixel discretization are inert (≤ 5% of clones touch a neighbour at any plausible density), which is *why* p_ind is unidentified — no conclusion below depends on it.
- **Out-of-sample prediction (layer frozen; residual KS with 95% bootstrap CI; n_sim = 81k–598k measured clones):**

| age | n_data | KS neutral+layer | KS loaded(10⁻²)+layer | critical | D/critical |
|---|---:|---:|---:|---:|---:|
| 1 mo (in-sample) | 15,865 | 0.030 [0.028–0.038] | 0.029 | 0.011 | 2.8× |
| 3 mo | 4,152 | 0.057 [0.053–0.072] | 0.056 | 0.021 | 2.7× |
| 6 mo | 2,474 | 0.171 [0.159–0.191] | 0.172 | 0.028 | 6.2× |
| 12 mo | 3,485 | 0.276 [0.261–0.293] | 0.259 | 0.023 | 11.8× |

- **Verdict for Task 2: the strip kernel does not reproduce mouse oesophagus.** The residual reappears and grows with age because the failure is dynamical, not observational: predicted median clone area 419 → 916 → 1,560 → 2,785 µm² (deep-pool medians) at 1/3/6/12 mo vs observed 424 → 968 → 853 → 957 µm². The kernel's mean area grows ∝ t^0.80 (2D neutral drift coarsening); the tissue's median saturates (∝ t^0.31) — clones are lost (density falls 4.4×, 40.8 → 9.2 mm⁻² from 1 to 12 mo) *without the survivors enlarging*, which drift coarsening cannot do. A per-age refit (an unphysical, maximally flexible layer) still rejects at 12 mo (D = 0.076, 2.8× critical) and would require the per-cell footprint to shrink 60 → 20 µm² with age. Load does not help: loaded and neutral residuals coincide within CI at every age.

## 3. Power surface (Task 3) — and the artifact that produced the old prescription

**The old signal was an artifact.** On mean-rescaled clone sizes the neutral-vs-loaded KS is 0.0918 at 1 mo for *every* s_d from 5×10⁻⁴ to 10⁻² — identical to the split-half self-null of the *same neutral ensemble* (0.0915) — and 0.038–0.040 at 3 mo vs a self-null of 0.040. Small integer clone sizes divided by two means differing by ~0.5% shift the two step-CDF lattices by one atom, so the rescaled KS equals the mass of the largest atom regardless of s_d. The unscaled 1-mo KS is 0.004. The genuine shape signal exists only at 12 mo and s_d ≥ 5×10⁻³ (unscaled KS 0.021–0.057). The manuscript's 600–3,200-clone power figure was computed on the artifact and is void.

**New surface** — Neyman–Pearson likelihood-ratio test on the layered clone-area densities (log-area kernel densities from a 30% fit split of 71k–600k-clone pools, Monte Carlo on the held-out 70%; empirical size 0.040–0.055 at nominal 0.05, verified). Required clones per arm for 80% power:

| s_d | 1 mo | 3 mo | 6 mo | 12 mo |
|---:|---:|---:|---:|---:|
| 5×10⁻⁴ | none (power ≤0.06 at 51k) | none (≤0.06) | none (≤0.10) | none (≤0.05) |
| 10⁻³ | none (≤0.05) | none (≤0.23) | none (≤0.05) | none (≤0.05) |
| 2×10⁻³ | none (≤0.10) | none (≤0.23) | none (≤0.05) | none (≤0.11) |
| 5×10⁻³ | none (≤0.04) | none (≤0.07) | none (≤0.06) | **≈18,000** |
| 10⁻² | none (≤0.06) | none (≤0.17) | none (≤0.52 at 51k) | **≈730** |

- **s_d floor for a designable experiment: 5×10⁻³.** Below it, 80% power is not reached at any age up to 51,200 clones per arm; power stays at or near the test's size in most cells but not identically at it (max at 51,200/arm: 0.23 at s_d = 10⁻³ and 2×10⁻³ at 3 mo; 0.10–0.11 in a few cells). At s_d = 5×10⁻³ 80% power is 12-month-only and needs ≈18,000 clones/arm; only s_d = 10⁻² is feasible (≈730/arm at 12 mo). The 1–6 month arms never reach 80% power at any s_d ≤ 10⁻²: the 3-mo cells peak at 0.17–0.23 and s_d = 10⁻²/6 mo at 0.52 with 51,200 clones/arm, so any residual signal there is far below a usable test. Load acts through late lineage takeover (12-mo mean area +14%, surviving clones −13% at s_d = 10⁻²), not through early distribution shape.
- The LR test is the correct instrument and is ~5× more sample-efficient than KS on this contrast (KS needs ≈4,000/arm at s_d = 10⁻²/12 mo where LR needs ≈730); KS also mishandles the integer-lattice issue that produced the old artifact. Justification for reporting LR: the two hypotheses are simple and fully simulated, so Neyman–Pearson is most powerful; the KS numbers are retained only as a cross-check.
- **Caveat that governs the whole surface:** this power is conditional on a kernel that is itself rejected by the data (§2). It answers "how many clones would separate the two kernel members if the tissue obeyed the model" — a design number for a system this model describes, not for mouse oesophagus.

## 4. The test the data already permit (Task 4)

Applied the LR test (loaded vs neutral, layered) and each model's goodness of fit to the arms named by the panel.

| arm | n | LR z vs neutral null | LR z vs loaded model | KS to neutral | KS to loaded (10⁻²) | reading |
|---|---:|---:|---:|---:|---:|---|
| DEN 1 mo | 5,682 | +8.6 | +9.4 | 0.092 | 0.091 | outside **both** members; equal misfit — model inadequacy, not discrimination |
| control 3 mo | 4,152 | 1.4 (s_d=10⁻³, p=0.087) / 4.05 (s_d=10⁻²) | 0.76 / 3.9 | 0.057 | 0.056 | cannot discriminate at s_d ≤ 10⁻³; the s_d=10⁻² "signal" is dwarfed by the 2.6× critical misfit |
| control 12 mo | 3,485 | +19.4 to +27.8 | +18.5 to +21.3 | 0.276 | 0.259 | pure misfit to both |
| DEN 12 mo | 188 | +2.2 (10⁻³) / +15.8 (10⁻²) | +2.1 / +13.7 | 0.331 | 0.314 | outside both |

- **Outcome: "cannot discriminate," and the reason is model inadequacy.** At n in the thousands, an LR statistic between two members of a misspecified family reads out the misfit, not the effect: the DEN 1-month arm rejects the neutral null at 8.6 s.d. yet sits even further (9.4 s.d.) from the loaded member's own prediction, and the two members fit it equally badly (KS 0.092 vs 0.091). Reporting z = 8.6 as "load detected" would be indefensible. The 3-month control arm (n = 4,152 > the prescribed 3,200) does not discriminate at s_d = 10⁻³ (z = 1.4), exactly as the power surface predicts.
- **The observable that does discriminate — enormously — is not the model's.** DEN over control median clone area is 0.84× (1 mo), 1.42× (3 mo), 3.40× (6 mo), **8.55× (12 mo)** [KS 0.43, p = 8×10⁻³¹, D/critical 4.3]; the DEN 12-month 90th percentile is 84,457 µm² vs 8,533 µm² control (9.9×). The kernel's deleterious-load channel produces at most 1.14× (12 mo, s_d = 10⁻²): the wrong magnitude by ~50×. The mutagenised-tissue phenotype is clone *enlargement* by positively selected mutants (Notch1/Trp53/DN-Maml1-type competitive expansion, as Colom et al. report), i.e. beneficial/driver dynamics that this purely-deleterious-load kernel excludes by construction.

## 5. Honest headline (Task 5)

**Recommended (option b, scoped down):**

Abstract sentence — *"Confronting the model with the largest published in vivo clone-size dataset (Colom et al. 2020; 25,976 control and 6,690 mutagenised clones across 1–12 months), we find that the two-clone strip kernel, with or without deleterious load and after fitting an explicit measurement layer, is formally rejected by every out-of-sample control arm (two-sample KS 2.7–12× the α = 0.05 critical value) because tissue clone areas saturate after 3 months whereas the kernel coarsens without bound; clone-size distributions therefore cannot at present discriminate the loaded-interface model from the neutral null in any published dataset, and we specify the observable and design that could."*

Results paragraph — *"We tested whether the loaded-interface kernel could be validated against, or discriminated from the neutral null by, published clone-size data. An explicit measurement layer (per-cell projected YFP footprint 59 µm² [56–61], σ_F = 1.09, 1.3-µm² pixel quantisation, 50-µm² detection floor), fitted on the 1-month control arm alone (n = 15,865; residual KS = 0.030) and frozen, reproduces the 1- and 3-month control distributions to KS = 0.030 and 0.057 but fails progressively at 6 and 12 months (KS = 0.171 [0.159–0.191] and 0.276 [0.261–0.293]; α = 0.05 critical 0.028 and 0.023): the kernel coarsens as ⟨A⟩ ∝ t^0.80, whereas the tissue's median clone area is flat after 3 months (957 µm² observed vs 2,785 µm² predicted at 12 months). Load does not repair the discrepancy (loaded and neutral residuals coincide within CI at every age), and every published control and DEN arm formally rejects both kernels (D 2.7–12× critical). Second, because clone sizes are small integers, mean-rescaled KS distances between the neutral and loaded kernels are dominated by a lattice-offset artifact; a calibrated likelihood-ratio test on the layered distributions shows that clone-size shape reaches 80% power to separate the loaded from the neutral kernel only at 12 months and only for s_d ≥ 5×10⁻³ (≈18,000 clones per arm at s_d = 5×10⁻³, ≈730 at s_d = 10⁻²); at s_d ≤ 2×10⁻³ power never exceeds 0.23 and at 1-6 months never exceeds 0.52 with 51,200 clones per arm. Applied to the arms these data provide — the DEN 1-month arm (n = 5,682) and the 3-month control arm (n = 4,152) — the test does not discriminate: the DEN arm lies outside both model members (LR z = +8.6 vs the neutral null and +9.4 vs the loaded prediction, with identical goodness of fit KS = 0.092/0.091), and the control arm gives z = 1.4 (p = 0.087) at s_d = 10⁻³. The dominant feature of the mutagenised arms — a monotonically growing clone enlargement reaching 8.6× the control median at 12 months (p < 10⁻³⁰) — is a positive-selection phenotype outside the deleterious-load model."*

What the manuscript must change: (i) delete "both kernels fit ... equally well (KS = 0.16 both)" and the 600–3,200-clone prescription; (ii) replace the power point with the conditional surface and its s_d ≥ 5×10⁻³ floor, stated as conditional on a model the data reject; (iii) correct the clone accounting (Suppl. Table 14 YFP arms duplicate Table 6; 25,976 control + 6,690 DEN unique clones enter the confrontation); (iv) state that the discriminating observable would have to be one that survives the growth-law mismatch — e.g., a spatial statistic in a tissue where a flat clonal interface can be imaged, or an early-time (≤ 3-month), footprint-calibrated area distribution at s_d ≥ 5×10⁻³ with ≳ 2×10⁴ clones — not the shape of pooled 1–12-month clone-size histograms.

## Deviations from the task specification

1. Parallel pool capped at 12 workers (parent's resource directive; five sibling tracks share the machine). No seeds were cut — 2,000 seeds per condition were run (wall 45 min under contention).
2. The power test is a likelihood-ratio test rather than a KS-based test, as the task's own preference stated; KS retained only as a cross-check (Task 3). Justification is in §3: the rescaled KS is invalid on integer clone sizes and the LR is 5× more efficient.
3. The measurement-layer pools used for the power surface apply the layer without induction thinning and without fusion (unthinned survivors); §2's ablation shows fusion contributes ≤ 0.001 KS and thinning-plus-fusion vs unthinned-no-fusion agree to KS ≤ 0.010, so this is a validated simplification that buys 40× more clones per run for the density estimates.
4. Snapshot frames are the campaign convention 296/896/1,796/3,592 (stride-4 storage; ≤ 1-frame error at the odd targets, < 0.03% of the horizon).
