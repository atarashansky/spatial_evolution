"""Track A: anchor-pool leave-one-out for the five-rung ridge profile.

Recompute the five-rung ridge profile (s_d = 0.0025, 0.005, 0.01, 0.02, 0.04) and
the DerSimonian-Laird random-effects pool from data/ridge_profile_analysis_pool.parquet
under (i) the full 2,856-run pooled anchor and (ii) each leave-one-out anchor that
drops ONE contributing anchor sub-ensemble.  Anchor sub-ensembles are identified by
the `pool` column (kind == 'neutral').  Bootstrap: run-level, B replicates, every rung
and every remaining anchor sub-ensemble resampled, one shared anchor draw across all
five sd per replicate (common-mode anchor treatment, as deposited).

Outputs (written into ./out_loo/): anchor_loo_results.json, anchor_loo_boot_<variant>.npy
"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_estimator import crossing_loglin, crossing_isotonic, dl_random_effects

B = int(os.environ.get("LOO_B", "3000"))
SEED_BASE = 13_568_000        # bootstrap RNG seeds live inside this track's block
SDS = [0.0025, 0.005, 0.01, 0.02, 0.04]
OUT = "out_loo"
os.makedirs(OUT, exist_ok=True)

pool = pd.read_parquet("bundle/data/ridge_profile_analysis_pool.parquet")
neu = pool[pool["kind"] == "neutral"]
lod = pool[pool["kind"] == "loaded"]
neu_by_pool = {p: g["t_fix"].to_numpy() for p, g in neu.groupby("pool")}
anchor_names = sorted(neu_by_pool)  # census, ladderN32, new, tfixexp1, tfixexp2

def rungs_for(sd):
    d = lod[lod["sd"] == sd]
    return {float(t): g["t_fix"].to_numpy() for t, g in d.groupby("theta")}

rung_arrays = {sd: rungs_for(sd) for sd in SDS}


def profile(anchor_med, scheme, resample=False, rng=None):
    out = {}
    for sd in SDS:
        R = rung_arrays[sd]
        RR = {t: a[rng.integers(0, len(a), len(a))] for t, a in R.items()} if resample else R
        thetas = np.array(sorted(RR.keys()))
        med = np.array([np.median(RR[t]) for t in thetas])
        ratio = med / anchor_med
        if scheme == "loglin":
            out[sd] = crossing_loglin(thetas, ratio)
        else:
            wts = np.array([len(RR[t]) for t in thetas])
            out[sd] = crossing_isotonic(thetas, ratio, weights=wts)
    return out


def run_variant(name, sub_names, scheme, seed):
    print(f"[seed] {name} {scheme} boot_seed={seed}", flush=True)
    """sub_names: which anchor sub-ensembles compose the anchor for this variant."""
    parts_full = [neu_by_pool[p] for p in sub_names]
    anchor_all = np.concatenate(parts_full)
    a_med = float(np.median(anchor_all))
    # point profile
    pts = profile(a_med, scheme)
    # joint bootstrap
    rng = np.random.default_rng(seed)
    L = np.full((B, len(SDS)), np.nan)
    A = np.full(B, np.nan)
    for b in range(B):
        parts = [a[rng.integers(0, len(a), len(a))] for a in parts_full]
        am = float(np.median(np.concatenate(parts)))
        A[b] = am
        p = profile(am, scheme, resample=True, rng=rng)
        L[b] = [p[sd] for sd in SDS]
    np.save(f"{OUT}/anchor_loo_boot_{name}_{scheme}.npy", L)
    logL = np.log(L)
    frac_valid = np.isfinite(L).mean(0)
    se_log = np.array([np.nanstd(logL[:, j], ddof=1) for j in range(len(SDS))])
    ci = np.array([np.nanpercentile(L[:, j], [2.5, 97.5]) for j in range(len(SDS))])
    y = np.log(np.array([pts[sd] for sd in SDS]))
    re = dl_random_effects(y, se_log)
    # anchor CI from bootstrap medians
    a_ci = np.percentile(A, [2.5, 97.5])
    prof_min_sd = SDS[int(np.nanargmin([pts[sd] for sd in SDS]))]
    return dict(
        variant=name, scheme=scheme, dropped=None if name == "full" else name.replace("drop_", ""),
        n_anchor=int(len(anchor_all)), anchor_median=a_med,
        anchor_median_ci=[float(a_ci[0]), float(a_ci[1])],
        points={str(sd): float(pts[sd]) for sd in SDS},
        points_ci={str(sd): [float(ci[j, 0]), float(ci[j, 1])] for j, sd in enumerate(SDS)},
        se_log={str(sd): float(se_log[j]) for j, sd in enumerate(SDS)},
        boot_frac_valid={str(sd): float(frac_valid[j]) for j, sd in enumerate(SDS)},
        re_pooled=float(np.exp(re["mu"])),
        re_ci=[float(np.exp(re["ci"][0])), float(np.exp(re["ci"][1]))],
        re_pooled_log=re, tau=float(re["tau"]), tau2=float(re["tau2"]), I2=float(re["I2"]),
        Q=float(re["Q"]), Q_df=int(re["Q_df"]), Q_p=float(re["Q_p"]),
        re_prediction_interval=[float(np.exp(re["mu"] - 1.96 * np.sqrt(re["tau2"] + re["se"] ** 2))),
                                float(np.exp(re["mu"] + 1.96 * np.sqrt(re["tau2"] + re["se"] ** 2)))],
        profile_min_sd=float(prof_min_sd), profile_min_theta=float(pts[prof_min_sd]),
        B=B, boot_seed=int(seed),
    )


t0 = time.time()
results = []
for scheme in ("loglin", "isotonic"):
    # full anchor
    r = run_variant("full", anchor_names, scheme, seed=SEED_BASE + (0 if scheme=="loglin" else 100))
    r["kruskal_note"] = "all five sub-ensembles"
    results.append(r)
    print(f"[{scheme}] full: RE={r['re_pooled']:.4f} [{r['re_ci'][0]:.3f},{r['re_ci'][1]:.3f}] Q={r['Q']:.1f} p={r['Q_p']:.1e} anchor={r['anchor_median']:.0f}", flush=True)
    for i, drop in enumerate(anchor_names):
        subs = [p for p in anchor_names if p != drop]
        r = run_variant(f"drop_{drop}", subs, scheme, seed=SEED_BASE + (0 if scheme=="loglin" else 100) + 1 + i)
        results.append(r)
        print(f"[{scheme}] drop {drop:10s}: RE={r['re_pooled']:.4f} [{r['re_ci'][0]:.3f},{r['re_ci'][1]:.3f}] Q={r['Q']:.1f} p={r['Q_p']:.1e} anchor={r['anchor_median']:.0f}", flush=True)

# Kruskal-Wallis across the five anchor sub-ensembles + per-sub medians
from scipy import stats
kw = stats.kruskal(*[neu_by_pool[p] for p in anchor_names])
anchor_meta = {p: dict(n=int(len(neu_by_pool[p])), median=float(np.median(neu_by_pool[p])),
                       seed_min=int(neu[neu.pool == p].seed.min()), seed_max=int(neu[neu.pool == p].seed.max()))
               for p in anchor_names}
json.dump(dict(track="Round 8: anchor-pool leave-one-out for the five-rung ridge profile",
               convention="theta = U_d/s_d = mud/sd (U_d = mud, no factor of 2); ridge = theta at median T_fix ratio = 1/2 vs pooled neutral anchor",
               B=B, sds=SDS, anchor_subensembles=anchor_meta,
               kruskal_wallis_anchor=dict(H=float(kw.statistic), p=float(kw.pvalue)),
               results=results, wall_s=time.time() - t0),
          open(f"{OUT}/anchor_loo_results.json", "w"), indent=1)
print(f"[done] wall {time.time()-t0:.0f}s", flush=True)
