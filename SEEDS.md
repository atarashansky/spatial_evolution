# SEEDS.md — per-ensemble RNG seed conventions

Every run is seeded deterministically. For per-run tables the integer `seed` column IS the RNG seed passed to the kernel (`np.random.seed(seed)` / `numba`-side seeding inside the kernel). A single run is therefore reproduced by (data file, row) or (ensemble, seed). Ensembles that use consecutive indices seed run *i* with `seed = base + i`; the bases are the minima below. Files without a `seed` column are aggregates whose per-run parents are named in the README figure map.

| data file | rows | seed min | seed max | unique seeds |
|---|---|---|---|---|
| `data/aspect_ladder.parquet` | 4,538 | 946000 | 1069807 | 4499 |
| `data/bd_results.parquet` | 2,900 | 200150 | 2146924700 | 2900 |
| `data/excursion_stats.parquet` | 2,010 | 2000 | 2149 | 150 |
| `data/finite_size.parquet` | 1,870 | 11000 | 26199 | 1870 |
| `data/intensivity_ext.parquet` | 2,159 | 960000 | 985149 | 2159 |
| `data/n0_census_raw.parquet` | 79,950 | 0 | 23 | 24 |
| `data/neighbourhood_ridge.parquet` | 2,906 | 5400000 | 8660059 | 2906 |
| `data/production_dfe.parquet` | 12,300 | 30000 | 54299 | 12300 |
| `data/pusher_morphology.parquet` | 17,181 | 0 | 39 | 40 |
| `data/pusher_switch/ensemble_main.parquet` | 4,320 | 100000 | 104319 | 4320 |
| `data/pusher_switch/refine2_runs.parquet` | 840 | 9000 | 9119 | 120 |
| `data/pusher_switch/validation_runs.parquet` | 400 | 0 | 199 | 200 |
| `data/reduced_model_runs.parquet` | 1,560 | 1210000 | 1639039 | 1560 |
| `data/ridge_law.parquet` | 7,400 | 930000 | 949349 | 7400 |
| `data/sdsweep_results.parquet` | 1,580 | 200000 | 410039 | 1580 |
| `data/sigma_dR_gridseries.parquet` | 476,160 | 100000 | 404031 | 1984 |
| `data/sigma_dR_ladder.parquet` | 1,984 | 100000 | 404031 | 1984 |
| `data/sigma_dR_neutral_com.parquet` | 1,150,028 | 300000 | 300191 | 192 |
| `data/tail_ensemble.parquet` | 1,600 | 1256982 | 2146924700 | 1600 |
| `data/tfix_results.parquet` | 4,250 | 200000 | 204249 | 4250 |

Kernel-level RNG: each kernel draws from a NumPy `RandomState`/Generator seeded once per run; the njit inner loops receive pre-drawn variates or the seed as an argument (see the `seed` parameter of each kernel's public `evolve_*` function).
The two representative snapshots in Supplementary Fig. S20 (e, f) use `seed = 3` at the parameters given in the caption.

### Round-6 ensembles

- census/contact ensemble (`census2.py`): seeds recorded per run in `data/contact_census_runs.parquet` (column `seed`).
- estimator bracketing supplement: seed base 9,400,000, disjoint from all prior campaigns; per-run seeds in `data/estimator_supplement_runs.parquet`.
- click-onset sweep: per-run seeds in `data/onset_runs.parquet`; disjoint from the round-5 click-census seeds, so the s_d = 0.01 point is an independent replication.

### Round-7 ensembles

- census-across-s_d: seed base 12,640,000, disjoint per run and per arm; per-run seeds in `data/census_sd_runs.parquet`.
- von Neumann census: seed base 12,900,000, disjoint per arm (loaded / neutral anchors / validation); per-run seeds in `data/vn_census_runs.parquet`.
- fitness-precision probes: seeds 12,000,001–12,000,002.

**Seed audit (round 7).** A global audit of all deposited per-run tables found (i) seed integers reused across ensembles run at different parameters (distinct realisations sharing an RNG prefix, weakly outcome-correlated) and (ii) within the click-onset ensemble, identical per-rung seed lists reused between its s_d = 0.0025 and s_d = 0.04 ladders. Case (ii) is the only within-analysis reuse; the affected paired-bootstrap slope was recomputed with a seed-cluster bootstrap and on seed-independent subsets and is unchanged (0.70 [0.41, 0.79]; `data/onset_seed_dependence_check.json`). Round-7 ensembles were intended to allocate disjoint ranges per arm; the global audit (`data/seed_audit_global.json`) shows the von Neumann census and the ridge re-anchoring arms both took base 12,900,000 and share 1,556 seed integers (distinct realisations on different geometries, never compared or pooled in any statistic — disclosed, not corrected). Per-arm disjointness, verified by the audit before deposit, is the campaign convention going forward.


## Round 8 seed blocks

- Topology control (`topo_strip.py`, S29): {'block': [13500000, 13519999], 'used_min': 13500000, 'used_max': 13504015, 'n_unique': 3072, 'per_arm': {'hex_anchor': {'start': 13500000, 'n': 600}, 'hex_rung0_theta0.05': {'start': 13501000, 'n': 96}, 'hex_rung1_theta0.0744': {'start': 13501100, 'n': 96}, 'hex_rung2_theta0.1107': {'start': 13501200, 'n': 96}, 'hex_rung3_theta0.1646': {'start': 13501300, 'n': 96}, 'hex_rung4_theta0.2449': {'start': 13501400, 'n': 96}, 'hex_rung5_theta0.3644': {'start': 13501500, 'n': 96}, 'hex_rung6_theta0.5422': {'start': 13501600, 'n': 96}, 'hex_rung7_theta0.8066': {'start': 13501700, 'n': 96}, 'hex_rung8_theta1.2': {'start': 13501800, 'n': 96}, 'hex_census': {'start': 13501900, 'n': 64}, 'rand_anchor': {'start': 13502000, 'n': 600}, 'rand_rung0_theta0.05': {'start': 13503000, 'n': 96}, 'rand_rung1_theta0.0744': {'start': 13503100, 'n': 96}, 'rand_rung2_theta0.1107': {'start': 13503200, 'n': 96}, 'rand_rung3_theta0.1646': {'start': 13503300, 'n': 96}, 'rand_rung4_theta0.2449': {'start': 13503400, 'n': 96}, 'rand_rung5_theta0.3644': {'start': 13503500, 'n': 96}, 'rand_rung6_theta0.5422': {'start': 13503600, 'n': 96}, 'rand_rung7_theta0.8066': {'start': 13503700, 'n': 96}, 'rand_rung8_theta1.2': {'start': 13503800, 'n': 96}, 'rand_census': {'start': 13503900, 'n': 64}, 'moore_census_ctrl': {'start': 13504000, 'n': 16}, 'smoke_probes': {'start': 13519000, 'n': 300}}, 'disjoint_from_SEEDS_md': True, 'note': 'All seeds in [13,500,000, 13,504,015]; SEEDS.md tables span at most base 12,900,000 + rounds<=7 (max used 12,900,000-ledger and legacy campaigns < 13.5M). Smoke/timing probes at 13,519,000-13,519,299 are NOT deposited.'}
- Ridge-law refit low-s_d rung + fresh anchor (S28): ladder 13,575,000–13,576,695; anchor 13,570,000–13,570,599 (`ridge_law_refit_summary.json`)
- Placement Monte-Carlo bootstrap draws (S27, analysis-only): 13,560,000–13,569,999; boot 13,568,000–13,568,105
- Open system (S30) production: {"CS_halfdomain": [13546000, 13547499], "E1_lineage": [13520000, 13521799], "E2a_single": [13530000, 13537999], "E2b_disc3": [13540000, 13541799]}; diagnostic pilots (no statistic) up to 13,618,999 — see manuscript Methods for the disclosed integer overlap with the ridge-refit block
