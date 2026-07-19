"""Reduce fss_runs.parquet -> fss_summary.json (all reported numbers).

Every estimate is stored as [value, ci_lo, ci_hi] from B >= 2000 run-level
bootstrap resamples (rungs + own anchor resampled per replicate; the same
replicate index is used across cells so cross-cell contrasts are paired).
"""
import os, sys, json, time
import numpy as np
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import fss_analysis as fa
from scipy import stats

B = int(os.environ.get("FSS_B", "2000"))
CELL_META = {  # name -> (L_long, L_short, periodic, role)
    "G96":   (96,   32, True,  "L_long_ladder"),
    "G192":  (192,  32, True,  "L_long_ladder"),
    "G384":  (384,  32, True,  "L_long_ladder"),
    "G768":  (768,  32, True,  "L_long_ladder"),
    "G1536": (1536, 32, True,  "L_long_ladder"),
    "S16":   (192,  16, True,  "L_short_axis"),
    "S64":   (192,  64, True,  "L_short_axis"),
    "W32":   (192,  32, False, "walled_arm"),
}
KEYS = ("half_drop", "level_0.5", "level_0.4", "level_0.6", "level_0.5_isotonic")


def per_cell(df):
    cells = {}
    boots = {}
    arrays = {}
    for i, name in enumerate([c for c in CELL_META if c in set(df["cell"])]):
        thetas, rung, anch = fa.cell_arrays(df, name)
        arrays[name] = (thetas, rung, anch)
        pt = fa.cell_point(thetas, rung, anch)
        bt = fa.cell_bootstrap(thetas, rung, anch, B=B, seed=101 + i, keys=KEYS)
        boots[name] = bt
        d = df[df["cell"] == name]
        L_long, L_short, per, role = CELL_META[name]
        entry = dict(L_long=L_long, L_short=L_short, N=L_long * L_short,
                     periodic_y=per, role=role,
                     n_runs=int(len(d)), n_loaded=int((d["kind"] == "loaded").sum()),
                     n_anchor=int((d["kind"] == "neutral").sum()),
                     n_error=int((d["error"] != "").sum()),
                     n_censored=int((d["t_fix"] == -1).sum()),
                     max_wall_time_s=float(np.nanmax(d["wall_time_s"].to_numpy(float))),
                     mean_wall_time_s=float(np.nanmean(d["wall_time_s"].to_numpy(float))),
                     thetas=[float(t) for t in thetas])
        entry.update(fa.summarize_boot(pt, bt))
        entry["anchor_median"] = pt["anchor_median"]
        entry["rung_medians"] = pt["rung_medians"]
        entry["rung_ratio"] = pt["rung_ratio"]
        entry["rung_n"] = pt["rung_n"]
        entry["plateau_up_logmed"] = pt["plateau_up_logmed"]
        entry["plateau_lo_logmed"] = pt["plateau_lo_logmed"]
        cells[name] = entry
    return cells, boots, arrays


def paired_diff(bootA, bootB, key, valA, valB):
    a = bootA[key]; b = bootB[key]
    d = a - b
    ok = np.isfinite(d)
    if ok.sum() < 20:
        return dict(diff=[float(valA - valB), np.nan, np.nan], p_le_0=np.nan, p_ge_0=np.nan)
    lo, hi = np.percentile(d[ok], [2.5, 97.5])
    return dict(diff=[float(valA - valB), float(lo), float(hi)],
                p_le_0=float(np.mean(d[ok] <= 0)), p_ge_0=float(np.mean(d[ok] >= 0)),
                n_valid=int(ok.sum()))


def main(runs_path="fss_runs.parquet", pool_path=None, out_path="fss_summary.json",
         design_path="fss_design.json"):
    t0 = time.time()
    df = pd.read_parquet(runs_path)
    design = json.load(open(design_path)) if os.path.exists(design_path) else {}
    S = dict(track="finite-size / boundary-condition convergence of the s_d=0.01 crossover",
             sd=0.01, bootstrap_resamples=B, bootstrap_scheme=(
                 "run-level: within each replicate every loaded rung and the cell's own "
                 "neutral anchor are resampled with replacement; the same replicate stream "
                 "(same RNG index) is used for every cell so cross-cell contrasts are paired"),
             estimators=dict(
                 half_drop=("convention-free primary: theta at which log median T_fix crosses the "
                            "midpoint of the upper (two smallest thetas) and lower (two largest "
                            "thetas) plateau log-medians; first descending crossing, log-linear "
                            "in log(theta)"),
                 level_half=("legacy: ratio R=median T_fix/median own-anchor T_fix crossing 1/2, "
                             "linear interp of R vs log(theta), first descending crossing "
                             "(deposited 'loglin' scheme); isotonic (weighted PAVA) as sensitivity")))
    S["n_runs_total"] = int(len(df))
    S["n_errors_total"] = int((df["error"] != "").sum())
    S["n_censored_total"] = int((df["t_fix"] == -1).sum())
    S["seed_span"] = [int(df["seed"].min()), int(df["seed"].max())]
    S["n_unique_seeds"] = int(df["seed"].nunique())
    cells, boots, arrays = per_cell(df)
    S["cells"] = cells

    # ---------------- L_long ladder convergence
    ladder = [c for c in ["G96", "G192", "G384", "G768", "G1536"] if c in cells]
    Ls = np.array([CELL_META[c][0] for c in ladder], float)
    conv = {}
    for key in ("half_drop", "level_0.5"):
        vals = np.array([cells[c][key][0] for c in ladder], float)
        reps = np.column_stack([boots[c][key] for c in ladder])
        if np.all(np.isfinite(vals)) and len(ladder) >= 3:
            fit = fa.convergence_fit(Ls, vals, reps)
        else:
            fit = dict(note="undefined point estimate in at least one rung; fit skipped",
                       values=[float(v) for v in vals])
        # endpoint contrast: largest minus smallest L (paired)
        end = paired_diff(boots[ladder[-1]], boots[ladder[0]], key,
                          cells[ladder[-1]][key][0], cells[ladder[0]][key][0])
        conv[key] = dict(L_long=[float(x) for x in Ls], theta_star=[cells[c][key] for c in ladder],
                         fits=fit, endpoint_diff_largest_minus_smallest_L=end)
    S["convergence_L_long"] = conv

    # ---------------- L_short axis at fixed L_long = 192
    if all(c in cells for c in ("S16", "G192", "S64")):
        for key in ("half_drop", "level_0.5"):
            Lss = np.array([16, 32, 64], float)
            vals = np.array([cells["S16"][key][0], cells["G192"][key][0], cells["S64"][key][0]], float)
            reps = np.column_stack([boots["S16"][key], boots["G192"][key], boots["S64"][key]])
            ok = np.all(np.isfinite(np.log(reps)), axis=1)
            lr = np.log(reps[ok])
            v = np.nanvar(lr, axis=0); w = 1.0 / v
            slope_pt = float(np.polyfit(np.log(Lss), np.log(vals), 1, w=np.sqrt(w))[0])
            slopes = np.array([np.polyfit(np.log(Lss), r, 1, w=np.sqrt(w))[0] for r in lr])
            lo, hi = np.percentile(slopes, [2.5, 97.5])
            d64_16 = paired_diff(boots["S64"], boots["S16"], key, cells["S64"][key][0], cells["S16"][key][0])
            S.setdefault("L_short_axis", {})[key] = dict(
                L_short=[16, 32, 64], theta_star=[cells[c][key] for c in ("S16", "G192", "S64")],
                dlogtheta_dlogLshort=[slope_pt, float(lo), float(hi)],
                diff_S64_minus_S16=d64_16)

    # ---------------- walled arm
    if all(c in cells for c in ("W32", "G192")):
        w = {}
        for key in ("half_drop", "level_0.5", "level_0.4", "level_0.6"):
            w[key] = dict(walled=cells["W32"][key], periodic=cells["G192"][key],
                          diff_walled_minus_periodic=paired_diff(boots["W32"], boots["G192"], key,
                                                                 cells["W32"][key][0], cells["G192"][key][0]),
                          walled_outside_periodic_CI=bool(
                              (cells["W32"][key][0] < cells["G192"][key][1]) or
                              (cells["W32"][key][0] > cells["G192"][key][2])))
        # anchor comparison walled vs periodic
        _, _, anchW = arrays["W32"]; _, _, anchP = arrays["G192"]
        w["anchor_median_ratio_walled_over_periodic"] = float(np.median(anchW) / np.median(anchP))
        w["anchor_mwu_p"] = float(stats.mannwhitneyu(anchW, anchP, alternative="two-sided").pvalue)
        S["walled_arm"] = w

    # ---------------- pool replication check (G192 vs deposited 192x32 pool)
    if pool_path and "G192" in cells:
        pool = pd.read_parquet(pool_path)
        p01 = pool[(pool["sd"] == 0.01) | (pool["sd"] == 0.0)]
        dfp = pd.DataFrame({"cell": "POOL",
                            "kind": np.where(p01["sd"] == 0.0, "neutral", "loaded"),
                            "theta": p01["theta"].to_numpy(), "t_fix": p01["t_fix"].to_numpy()})
        thetas, rung, anch = fa.cell_arrays(dfp, "POOL")
        pt = fa.cell_point(thetas, rung, anch)
        bt = fa.cell_bootstrap(thetas, rung, anch, B=B, seed=999, keys=KEYS)
        poolsum = fa.summarize_boot(pt, bt)
        poolsum["anchor_median"] = pt["anchor_median"]
        poolsum["n_loaded"] = int((dfp["kind"] == "loaded").sum())
        poolsum["n_anchor"] = int((dfp["kind"] == "neutral").sum())
        # rung-wise MWU G192 vs pool at shared thetas, Fisher combine
        tg, rg, ag = arrays["G192"]
        pvals = []
        for t, arr in zip(tg, rg):
            j = np.where(np.isclose(thetas, t))[0]
            if j.size:
                pvals.append(float(stats.mannwhitneyu(arr, rung[j[0]], alternative="two-sided").pvalue))
        anchor_p = float(stats.mannwhitneyu(ag, anch, alternative="two-sided").pvalue)
        chi = -2 * np.sum(np.log(np.array(pvals + [anchor_p])))
        fisher_p = float(stats.chi2.sf(chi, 2 * (len(pvals) + 1)))
        rep = {}
        for key in ("half_drop", "level_0.5"):
            d = boots["G192"][key] - bt[key][: B]
            ok = np.isfinite(d)
            rep[key] = dict(G192=cells["G192"][key], pool=poolsum[key],
                            diff_G192_minus_pool=[float(cells["G192"][key][0] - poolsum[key][0])] +
                            [float(x) for x in np.percentile(d[ok], [2.5, 97.5])] if ok.sum() > 20
                            else [np.nan, np.nan],
                            p_le_0=float(np.mean(d[ok] <= 0)) if ok.sum() > 20 else np.nan)
        S["pool_replication_G192_vs_deposited"] = dict(
            deposited_pool=poolsum, rung_mwu_p=pvals, anchor_mwu_p=anchor_p,
            fisher_combined_p=fisher_p, crossings=rep)

    # ---------------- verdict fields
    hd = S["convergence_L_long"]["half_drop"]["fits"]
    lv = S["convergence_L_long"]["level_0.5"]["fits"]
    def verdict(fit, key):
        if "constant" not in fit:
            return "INCONCLUSIVE (fit skipped)"
        dA = fit["dAICc_powerlaw_minus_constant"]
        sgn = fit["drift_sign_test"]
        mono = max(sgn["frac_reps_all_successive_diffs_negative"],
                   sgn["frac_reps_all_successive_diffs_positive"])
        if dA >= -2 and mono < 0.90:
            return "PLATEAU"
        if dA < -2 and mono >= 0.90:
            return "DRIFT"
        return "MIXED/INCONCLUSIVE"
    S["verdict"] = dict(
        half_drop=verdict(hd, "half_drop"), level_half=verdict(lv, "level_0.5"),
        extrapolated_large_L_limit_half_drop=hd.get("powerlaw_plus_plateau", {}).get("theta_inf")
            if isinstance(hd, dict) else None,
        extrapolated_large_L_limit_level_half=lv.get("powerlaw_plus_plateau", {}).get("theta_inf")
            if isinstance(lv, dict) else None,
        constant_fit_half_drop=hd.get("constant", {}).get("theta_inf") if isinstance(hd, dict) else None,
        constant_fit_level_half=lv.get("constant", {}).get("theta_inf") if isinstance(lv, dict) else None,
        walled_shifts_outside_periodic_CI=S.get("walled_arm", {}).get("level_0.5", {}).get(
            "walled_outside_periodic_CI") if "walled_arm" in S else None)
    S["wall_clock_reduce_s"] = round(time.time() - t0, 1)
    json.dump(S, open(out_path, "w"), indent=1, default=float)
    print(json.dumps(S["verdict"], indent=1))
    return S


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="fss_runs.parquet")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default="fss_summary.json")
    a = ap.parse_args()
    main(a.runs, a.pool, a.out)
