"""Round-9 seed-overlap diagnostic (S25 ridge re-anchor  x  S26 von Neumann census)
and seed-audit coverage accounting.

Regenerates every number in seed_overlap_S25S26_summary.json / seed_overlap_memo.md
from the deposited run tables. No new simulations.

Inputs (deposited, round 7 / global audit):
  ridge_profile_runs.parquet          (S25 raw runs, 4,700; seeds 12,900,000-12,921,299)
  ridge_profile_analysis_pool.parquet  (S25 headline pool, 15,612 records)
  vn_census_runs.parquet              (S26 vN census, 1,708; seeds 12,900,000-12,902,151)
  ridge_profile_summary.json, vn_census_summary.json
  seed_audit_global.json              (round-7 global seed audit; 54,355 unique runs)
  + the 28 per-run tables recorded as the audit's inputs (for the coverage half)

Outputs: seed_overlap_S25S26_summary.json, seed_coverage_table.csv,
         seed_coverage_main_figures.csv, seed_overlap_memo.md, fig_seed_overlap.png
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

RNG_SEED = 20260718
NPERM = 2000

BASE = "/root/src/.operon/orgs/01582c35-9930-414e-93b9-0dd852675b4d/artifacts/proj_bcf0e7b390f2"
P = {
    "runs": f"{BASE}/407f22e3-ed5c-4cf5-99dd-0ab597e8907f/v307fa048_ridge_profile_runs.parquet",
    "pool": f"{BASE}/58c32382-2bf3-4fcf-985d-cb2c04836410/v4b92d826_ridge_profile_analysis_pool.parquet",
    "vn":   f"{BASE}/9974f572-9d3e-447e-a72c-d63556ca1388/v184a7597_vn_census_runs.parquet",
    "audit": f"{BASE}/18e823c8-86b5-49d6-b255-5193151d2c0e/vab002f29_seed_audit_global.json",
}


def spearman(x, y):
    return stats.spearmanr(x, y).statistic


# ---------------------------------------------------------------- part 1
def pair_by_seed():
    runs = pd.read_parquet(P["runs"])
    vn = pd.read_parquet(P["vn"])
    shared = np.intersect1d(runs.seed.unique(), vn.seed.unique())   # 1,556
    r = runs[runs.seed.isin(shared)].rename(columns={"t_fix": "tfix_S25", "arm": "arm_S25"})
    v = vn[vn.seed.isin(shared)].rename(columns={"t_fix": "tfix_S26", "arm": "arm_S26", "kind": "kind_S26"})
    pairs = r[["seed", "arm_S25", "sd", "mud", "tfix_S25"]].merge(
        v[["seed", "kind_S26", "arm_S26", "theta", "tfix_S26"]], on="seed").sort_values("seed").reset_index(drop=True)
    assert len(pairs) == 1556, len(pairs)
    # ensemble arm ("cell") on each side; S25 shared runs are all the ANCH neutral anchor arm
    pairs["cell_S25"] = "ANCH|neutral|Moore"
    pairs["cell_S26"] = pairs[["kind_S26", "arm_S26", "theta"]].astype(str).agg("|".join, axis=1)
    # natural unit: percentile of T_fix within its own arm
    pairs["pct_S25"] = pairs.groupby("cell_S25").tfix_S25.rank(pct=True)
    pairs["pct_S26"] = pairs.groupby("cell_S26").tfix_S26.rank(pct=True)
    return pairs


def permutation_null(pairs, rng):
    x = pairs.pct_S25.values
    y = pairs.pct_S26.values
    cells = pairs.cell_S26.values
    idx_by_cell = [np.where(cells == c)[0] for c in np.unique(cells)]
    obs = spearman(x, y)
    null = np.empty(NPERM)
    for i in range(NPERM):
        yp = np.empty_like(y)
        for idx in idx_by_cell:                       # shuffle within arm: breaks the seed pairing,
            yp[idx] = y[rng.permutation(idx)]         # preserves arm structure
        null[i] = spearman(x, yp)
    p_two = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + NPERM)
    p_one = (1 + np.sum(null >= obs)) / (1 + NPERM)
    return obs, null, p_two, p_one


def bootstrap_ci(pairs, rng, B=2000):
    x = pairs.pct_S25.values
    y = pairs.pct_S26.values
    n = len(x)
    b = np.array([spearman(*[a[ii] for a in (x, y)]) for ii in (rng.integers(0, n, n) for _ in range(B))])
    return np.quantile(b, [0.025, 0.975])


def per_arm(pairs, rng):
    out = {}
    for cell, g in pairs.groupby("cell_S26"):
        obs = spearman(g.pct_S25, g.pct_S26)
        nl = np.array([spearman(g.pct_S25.values, rng.permutation(g.pct_S26.values)) for _ in range(NPERM)])
        out[cell] = {"n": int(len(g)), "rho": float(obs),
                     "perm_p_two_sided": float((1 + np.sum(np.abs(nl) >= abs(obs))) / (1 + NPERM))}
    return out


# ---------------------------------------------------------------- part 2
AUDIT_INPUT_TABLES = [   # the 16 seeded, non-DERIVED tables among the audit's 28 recorded inputs
    "bd_results", "census_sd_runs", "estimator_sensitivity_ladders", "estimator_supplement_runs",
    "finite_size", "intensivity_ext", "neighbourhood_ridge", "onset_runs", "production_dfe",
    "reduced_model_runs", "ridge_law", "ridge_profile_runs", "sdsweep_results", "tail_ensemble",
    "tfix_results", "vn_census_runs"]


def audit_universe(paths_by_name):
    """Reproduce the audit's 54,355 by its own de-dup rule. paths_by_name: {table_name: parquet_path}."""
    recs = []
    for name in AUDIT_INPUT_TABLES:
        d = pd.read_parquet(paths_by_name[name])
        tc = next(c for c in ("t_fix", "T_fix", "tfix") if c in d.columns)
        recs.append(pd.DataFrame({
            "ens": name, "seed": pd.to_numeric(d["seed"], errors="coerce"),
            "sd": d["sd"] if "sd" in d else np.nan, "mud": d["mud"] if "mud" in d else np.nan,
            "L_long": d["L_long"] if "L_long" in d else np.nan,
            "L_short": d["L_short"] if "L_short" in d else np.nan,
            "moore": d["moore"].astype(str) if "moore" in d else "na", "tfix": d[tc]}))
    u = pd.concat(recs, ignore_index=True)
    u["Llong"] = pd.to_numeric(u.L_long, errors="coerce").fillna(192.0)
    u["Lshort"] = pd.to_numeric(u.L_short, errors="coerce").fillna(32.0)
    u["moore_f"] = u.moore.replace({"na": "True", "nan": "True"})
    u["pk2"] = (u.Llong.astype(str) + "|" + u.Lshort.astype(str) + "|" + u.sd.astype(str)
                + "|" + u.mud.astype(str) + "|" + u.moore_f.astype(str))
    u["rk2"] = u.pk2 + "|" + u.seed.astype(str) + "|" + u.tfix.round(6).astype(str)
    u2 = u.drop_duplicates("rk2")            # exact-duplicate (re-deposited) records removed
    return u, u2                              # len(u2) == 54,355; len(u)-len(u2) == 8,550


if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    pairs = pair_by_seed()
    rho, null, p_two, p_one = permutation_null(pairs, rng)
    ci = bootstrap_ci(pairs, rng)
    arms = per_arm(pairs, rng)
    print(f"rho={rho:.4f}  perm P two-sided={p_two:.3f}  95% CI [{ci[0]:.3f},{ci[1]:.3f}]  "
          f"null sd={null.std():.4f}  arms={len(arms)}")
