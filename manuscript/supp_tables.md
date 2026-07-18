# Supplementary Tables

## Supplementary Table 1 | Run provenance by figure (current numbering)

All simulations use the spatial Moran strip model (Numba kernel `evolution_strip.py`, validated against
the 2018 reference implementation; Birth–death variant `evolution_strip_bd.py`; k-clone variant
`evolution_kclone.py`; composition kernel `evolution_strip_dfe.py`; frequency-dependent kernel
`evolution_strip_fd.py`; lineage-tracing kernel `evolution_lineage.py`; rule-switch kernel
`evolution_strip_mixed.py`; instrumented contact-census kernel `census2.py`, bit-for-bit identical to `strip_census.py` at equal seed). Ensembles were run on a 64-core driver (`ensemble_driver.py`); zero unexplained failures across the campaign. **Itemized total: 108,237 runs** across the 33 ensembles itemized below (excluding exemplar renders); the ensembles added or extended in revision are marked ‡ (11 table rows; some logical additions span more than one row). A further 9,160-run rule-switch ensemble (mixed dB/Bd 'pusher' clones), not part of the present analysis, is described separately.
"Censored" = runs that reached the frame cap without fixation; censored runs enter as lower bounds where noted.

| Figure | Ensemble (track) | Runs | Geometry | Cap (frames) | Censoring | Data artifacts | Notes |
|---|---|---|---|---|---|---|---|
| S1, S8 | Neutral controls: width growth + tilt null + plateau ext | 7,900 | 512x{32-256} | 100k-800k | n/a (width/trajectory tracks) | `controls_neutral.parquet, w_trajectories(_ext).npz, theta_trajectories_v2.npz, controls_summary.json` |  |
| S8, S2 | Tension spectra (dB) | 479 | 768x256 | 250k-400k | spectra use non-fixed runs (n_ok 331 across 10 conditions) | `tension_results.parquet, tension_spectra.npz` |  |
| S2 | Collapse replication arms | 288 | 768x256 | 400k | matched protocol, t>=200k window | `collapse_replication.parquet` | n_new 124+119 usable of 288 |
| 2a–c, 1c | T_fix scaling (neutral / load / load+jackpot x L_short) | 4,250 | 192x{16-256} | 20M | 11 censored (neutral; enter as lower bounds) | `tfix_results.parquet` |  |
| 3a–d | Mechanism: arrest, deposition, excursions | 320 | 512x64 | - | - | `scar_data.npz, excursion_stats.parquet (2,010 excursions), jackpot_data.npz` |  |
| 4a–b, S3 | Effect-size (s_d) sweep | 1,580 | 192x{32,128} | 20M | 0 | `sdsweep_results.parquet` |  |
| 4c | Tail ensemble (Hill alpha vs s_d) | 1,600 | 192x32 | 20M | 0 | `tail_ensemble.parquet` |  |
| S6 | alpha(s_d, L_short) map | 9,200 | 192x{16,32,64} | 30M | 0 (max obs 5.2M = 17% cap) | `alpha_map_runs.parquet, alpha_map_results.parquet` |  |
| S3 | Bd vs dB battery | 2,900 | 192x32; 768x256 | 20M / 400k | 0 | `bd_results.parquet` |  |
| S4 | k-clone tournaments (k=2-16) | 960 | 256x32 | - | 0 | `kclone_runs.parquet, kclone_events.parquet, kclone_summary.json` |  |
| S5 | Beneficial-supply arms | 1,500 | 192x32 | 20M | 0 | `tfix_results.parquet (exp 2)` |  |
| 5 | DFE composition: ladder + anchors + sigma-sensitivity + tissue DFEs | 12,300 | 192x128 | 30M | 0 | `production_dfe.parquet` | groups: ladder 6,000; sens 2,000; tissue 4,000; anchor 300 |
| 5 (pilot) | DFE pilot + point-mass validation | 1,700 | 192x{32,128} | - | 0 | `dfe pilot/validation parquets` |  |
| S16 | Frequency-dependent selection (phases 1-3) | 5,600 | 128x24 (+ pilot 192x32) | 1.2M-30M | neutral arms censor at cap by design (0/1,200 fixed) | `fd_p1/2/3.parquet` |  |
| S16 (pilot), S7 | FD pilot + width-invariance supplement | 3,336 | 128x24 | - | - | `fd_pilot_results.parquet (3,000), fd_width_summary.parquet (336)` |  |
| 6 | Regime-map (Lambda, s_d) grid + neutral anchor | 2,300 | 192x32 | 20M | 0 | `regime_map_grid.parquet, regime_map_summary.json` | 14 new cells × 150 runs + neutral anchor 200 = 2,300 (this row); 5 further cells reuse `tfix_results.parquet` / `tail_ensemble.parquet` runs itemized in their own rows (n = 250–400); one cell (mud = 1.6) not run by design (mud > 0.8); per-cell sources in `regime_map_summary.json` |
| 7 (lineage kernels) | Lineage-tracing kernels (neutral, loaded x2) | 144 | clone-labelled | - | - | `lineage_sims.pkl, evolution_lineage.py` |  |
| 2d–e, S9, S14 ‡ | Finite-size L_long sweep (neutral + loaded, gap statistics) | 1,870 | {96–768}x32; {192,384}x128 | 40M | 0 | `finite_size.parquet, finite_size_summary.json` | zeta_neutral 1.73, zeta_loaded 1.17, Var[g]~t^alpha, collapse C=1.64 |
| 3e–f, S10 ‡ | Reduced-model inputs: drift response v(g), zero-mode D0, MSD crossover | 1,560 | 192x{16–256} | - | 0 | `reduced_model_runs.parquet, reduced_model_series.npz, reduced_model_summary.json` | plus 3,000 SDE realizations per cell |
| S15, S8 (methods) ‡ | Stiffness / noise / Laplace / tilt / controls (tension tests) | 2,520 | 384x128; 512x128 walled; enclave R=20 | - | 0 | `tension_test.parquet` | 720 mode, 800 growth, 320 droplet, 480 tilt, 200 controls |
| S15 (audit) ‡ | Independent stiffness re-measurement (k=8, 48 seeds x 4 loads + checks) | 312 | 384x128 | - | 0 | `audit_runs.tar.gz, tension_audit_results.csv` | Independent kernel implementation |
| 4d–f, S11 ‡ | Ridge tracing: low-Lambda ladders, fixed-s_d rate sweeps, N-crossing, neutral anchors | 7,400 | 192x{32,64,128} | 20M | 0 | `ridge_law.parquet, ridge_law_summary.json` | groups T1, T2a/b, T3, N32/64/128 |
| 7 ‡ | Confrontation kernel ensembles (age-matched, layered; power surface) | 12,000 | 192x32 clone-labelled | - | - | `confrontation_stats.parquet, confrontation_summary.json` | 2,000 runs/condition x 6 conditions |
| 4f ‡ | Intensivity extension: 192x256 & 192x512 rate ladders + neutral anchors; 384x128 aspect-ratio control | 2,159 | 192x{256,512}, 384x128 | 40M/60M | 0 | `intensivity_ext.parquet, intensivity_ext_summary.json` | five-point U_d/s_d; slope vs log10N -0.02 [-0.22,+0.19] |
| 4g, 4h | Least-loaded-class census (instrumented kernel, per-cell hit counts) | 306 | 192x32, 192x128 | to fixation | none (all runs to fixation) | `n0_census.parquet, n0_census_raw.parquet` | 192×32: nine U_d/s_d rungs (0.05–5) × 24 runs = 216; 192×128: six rungs (0.1–1) at 13, 14, 15, 16, 16, 16 runs = 90; s_d = 0.01 |
| S17 | Ratchet-click census and rate-spread test (instrumented kernel) | 1,984 | 192x32 | 4x10^7 | none (all runs to fixation) | `sigma_dR_ladder.parquet, sigma_dR_gridseries.parquet, sigma_dR_neutral_com.parquet` | 12 U_d/s_d rungs × 128 + 96 at U_d/s_d = 10; neutral arm 192; imposed-gap arm 160 (5 g0 × 32); s_d = 0.01 |
| S18 | Loaded-ridge neighbourhood test (von Neumann stencil arm) | 2,906 | 192x32, 192x128 | 2x10^7 | none | `neighbourhood_ridge.parquet` | vN 192×32: 13 rungs × 150 + 200 neutral; vN 192×128: 8 rungs × 60–100 + 100 neutral; Moore replication 96; s_d = 0.01 |
| S19 | Aspect-ratio replication ladder (new runs only) | 1,029 | 96x512, 768x64, 96x256, 384x64 | to fixation | none | `aspect_ladder.parquet` | 933 production + 96 pilot; pools with 3,509 reused runs booked above; 768x64 arm deadline-curtailed (n = 8–50/rung); s_d = 0.01 |
| S20 (a–c) | Pusher-switch mixed-rule ensemble (kinetic neutrality, plateau shares, incidence law) | 9,160 | 192x64 | 7.5x10^5 (plateau); 5x10^5–1.5x10^6 (b* grids) | none | `data/pusher_switch/*` (`calibration_runs.parquet`, `ensemble_main.parquet`, validation/refine tables) | b* logistic grid 4,440 runs; b/b* × load × 120 seeds plateau grid; absolute-clock contrast arm; μ_c × load incidence grid; kernel `evolution_strip_mixed.py`, validated bit-exact at μ_c = 0 |
| S20 (d, e) | Front-morphology ensembles (pusher vs fitness vs neutral fronts) | 258 | 768x64, 384x64 | 2.4x10^4 (fine); 10^5 (coarse) | none | `rough_out/`, `rough_out2/`, `pusher_morphology_summary.json` | 40 seeds/arm × 4 arms (coarse) + 24 seeds/arm × 4 arms (fine) + 2 representative snapshots; b = 0.075 = 1.8 b*, s_d = 0.04 |

## Supplementary Table 2 | Mapping model load parameters to human tissues

Conversion of the model's per-division deleterious mutation parameters to
literature-derived per-division SNV rates for renewing human tissues. μ_d is the
per-division probability of acquiring a deleterious mutation, μ_d = (SNV per
division) × f_del, with the deleterious-target fraction f_del = 2–10% of arising SNVs
(functional-genome footprint / target-size reasoning), capped at μ_d ≤ 0.8; s_d the
per-mutation fitness cost; Λ = μ_d·s_d the load rate (per division), the model's control parameter.
Ranges span the cited burden/division-rate estimates.

| Tissue | SNV per division | μ_d (per division) | s_d | Λ = μ_d·s_d | Basis |
|---|---|---|---|---|---|
| Colonic crypt | 0.8–2 | 0.016–0.2 | 5×10^-4–0.01 | 8×10^-6–0.002 | ~50 SBS/yr / 25-50 div/yr |
| Epidermis (sun-exp.) | 5–30 | 0.1–0.8 | 5×10^-4–0.01 | 5×10^-5–0.008 | 2-6 mut/Mb burden / ~2.6k divisions |
| HSC (blood) | 7–16 | 0.14–0.8 | 5×10^-4–0.01 | 7×10^-5–0.008 | ~14 SBS/yr / ~0.8-1.3 div/yr |
| Hypermutator tumor | 20–150 | 0.4–0.8 | 0.001–0.03 | 4×10^-4–0.024 | 4-100x normal per-division rate |

The probed simulation range Λ = 2.5×10^-4 – 4×10^-3 (tension sweep) and 10^-3 (fixed-Λ
effect-size sweeps) sits inside the estimated ranges for sun-exposed epidermis, blood
(HSC), and hypermutator tumors, and at the upper end of the colonic-crypt range —
i.e., the drift-dominated ("fluid") regime characterized in the main text is the
regime real renewing tissues occupy (main-text Fig. 5).

## Supplementary Table 3 | Per-rung run counts for the ensembles reported with ranges

Exact number of runs per U_d/s_d rung for every arm that a caption or the main text reports with a count range. Zero runs = rung not sampled in that arm (by design; the aspect arms sample the transition band densely and add sparse rungs elsewhere). All runs zero-censored unless noted. Deadline-curtailed rungs are marked ‡ and are reported, not filled.

**Aspect and area ladder (Fig. 4f, Supplementary Fig. S19; s_d = 0.01):**

| geometry (L_long×L_short) | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1 | 1.2 | 1.4 | 2 | 5 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 96×256 | – | 30 | 40 | 8 | 40 | – | – | 38 | – | 24 | – | – | – | 180 |
| 96×512 | – | 30 | 30 | 8 | 23 | – | – | 32 | – | 20 | – | – | – | 143 |
| 192×128 | 150 | 150 | – | 150 | – | 150 | – | 150 | – | 150 | – | 150 | 150 | 1200 |
| 192×256 | – | 150 | – | 150 | – | 150 | – | 150 | – | 150 | – | – | – | 750 |
| 192×512 | – | 60 | – | 60 | – | 60 | – | 60 | – | 60 | – | – | – | 300 |
| 384×64 | – | – | – | 8 | 40 | – | 60 | 8 | 60 | – | 20 | 24 | – | 220 |
| 384×128 | – | 150 | – | 150 | – | 150 | – | 150 | – | 150 | – | – | – | 750 |
| 768×64 ‡ | – | – | – | 8 | 39 | – | 46 | 8 | 50 | – | 40 | 30 | – | 221 |

‡ 768×64 arm curtailed by the compute deadline (n = 8–50 per rung); disclosed as partial, not powered up.

**Neighbourhood/stencil test (Supplementary Fig. S18; 192×L_short, s_d = 0.01):**

| L_short | stencil | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.6 | 0.8 | 1 | 1.2 | 2 | 5 | 10 | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | von Neumann | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 150 | 1950 |
| 32 | Moore | – | – | – | 24 | – | 24 | – | 24 | – | – | – | – | – | 72 |
| 128 | von Neumann | – | 100 | – | 100 | 60 | 60 | 60 | 60 | – | 60 | 60 | – | – | 560 |

**Least-loaded-class census (Fig. 4g,h; s_d = 0.01):**

| geometry | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.6 | 1 | 2 | 5 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 192×32 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 216 |
| 192×128 | – | 13 | 14 | 15 | 16 | 16 | 16 | – | – | 90 |
| Fig. 4h, Supplementary Fig. S21 ‡ | Front census functional form and contact structure | 956 (14-rung θ ladder 192×32 at 64/rung + 5-rung 192×128 at 12/rung, s_d = 0.01) | `census2.py` | 0 | `contact_census_runs.parquet` |
| Supplementary Fig. S22 ‡ | Fixed-s_d bracketing supplement for estimator robustness | 2,100 (14 rungs × 150 runs at s_d = 0.0025, 0.005, 0.02) | `evolution_strip.py` | 0 | `estimator_supplement_runs.parquet` |
| Supplementary Fig. S23 ‡ | Click-onset effect-size sweep (round-6 test, against hypothesis) | 5,360 (s_d = 0.0025, 0.01, 0.04 ladders at 96/rung, densified 288/rung; 288 neutral; 464 pilot) | `sdr_kernel.py` (instrumented, unmodified) | 0 | `onset_runs.parquet` |
