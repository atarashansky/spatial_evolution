"""Analysis for the round-8 open-system track.

Reads results/E1_runs.parquet (+ per-run npz), E2a/E2b marker tables, and the
same-geometry closed-system reference CS_runs.parquet.  Writes:
  open_system_runs.parquet      (all per-run rows, one file, `arm` column)
  open_system_summary.json      (headline numbers + bootstrap CIs)
  fig_open_system.png           (4-panel deliverable)
The memo is written by hand from the summary (see run of make_memo.py).
"""
import glob
import json
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20250718)
N_BOOT = 2000
N_SITES = 192 * 64


def boot_median_ratio(x, y, n=N_BOOT):
    """median(x)/median(y) with independent case-resampling bootstrap."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    vals = np.empty(n)
    for b in range(n):
        vals[b] = np.median(RNG.choice(x, x.size)) / np.median(RNG.choice(y, y.size))
    est = np.median(x) / np.median(y)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def boot_median(x, n=N_BOOT):
    x = np.asarray(x, float)
    vals = np.array([np.median(RNG.choice(x, x.size)) for _ in range(n)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.median(x)), float(lo), float(hi)


def boot_mean(x, n=N_BOOT):
    x = np.asarray(x, float)
    vals = np.array([np.mean(RNG.choice(x, x.size)) for _ in range(n)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(x)), float(lo), float(hi)


def boot_prob(k, n_tot, n=N_BOOT):
    p = k / n_tot
    vals = RNG.binomial(n_tot, p, size=n) / n_tot
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(p), float(lo), float(hi)


def boot_prob_ratio(k1, n1, k2, n2, n=N_BOOT):
    p1, p2 = k1 / n1, k2 / n2
    v1 = RNG.binomial(n1, p1, size=n) / n1
    v2 = RNG.binomial(n2, p2, size=n) / n2
    ok = v2 > 0
    r = np.full(n, np.nan)
    r[ok] = v1[ok] / v2[ok]
    est = p1 / p2 if p2 > 0 else np.nan
    lo, hi = np.nanpercentile(r, [2.5, 97.5])
    return float(est), float(lo), float(hi), float((~ok).mean())


# ------------------------------------------------------------------ E1
def load_E1():
    df = pd.read_parquet("results/E1_runs.parquet")
    assert (df.error == "").all(), df.error[df.error != ""].iloc[0]
    ts_by_seed = {}
    for f in glob.glob("results/E1/lin_*.npz"):
        d = np.load(f)
        ts_by_seed[int(d["seed"])] = d["ts"]
    return df, ts_by_seed


def analyse_E1(df, ts_by_seed):
    out = {"protocol": dict(L="192x64", sd=0.01, nu=5e-4, T_burn=300_000,
                             T_obs=2_000_000,
                             runs_per_rung={"neutral": 600, "loaded": 400},
                             thetas=[0, 0.2, 0.4, 0.8],
                             marking="every live cell a unique lineage at t=0 "
                                     "(post burn-in); births tagged as new "
                                     "lineages with prob nu per birth")}
    thetas = sorted(df.theta.unique())
    per = {}
    # stationary lineage number: mean over last decade of samples (t>=1e6)
    def statK(seed):
        ts = ts_by_seed[seed]
        m = ts[:, 0] >= 1_000_000
        return float(ts[m, 1].mean())
    df["K_stat"] = [statK(s) for s in df.seed]
    df["thalf"] = df["t_half_cohort"]
    # cohort extinction time: first sample with zero t=0-cohort survivors
    def text(seed):
        ts = ts_by_seed[seed]
        z = np.where(ts[:, 2] == 0)[0]
        return float(ts[z[0], 0]) if z.size else float("inf")
    df["t_cohort_ext"] = [text(s) for s in df.seed]
    neu = df[df.theta == 0]
    kn = neu.K_stat.values
    thn = neu.thalf.values
    ten = neu.t_cohort_ext.values
    for th in thetas:
        d = df[df.theta == th]
        e = dict(theta=th, n_runs=int(len(d)),
                 mud=float(d.mud.iloc[0]))
        e["K_stat_mean"], e["K_stat_lo"], e["K_stat_hi"] = boot_mean(d.K_stat)
        e["thalf_median"], e["thalf_lo"], e["thalf_hi"] = boot_median(d.thalf)
        e["text_median"], e["text_lo"], e["text_hi"] = boot_median(d.t_cohort_ext)
        e["frac_cohort_alive_at_end"] = float(np.mean(np.isinf(d.t_cohort_ext)))
        if th > 0:
            e["K_ratio_to_neutral"], e["Kr_lo"], e["Kr_hi"] = boot_median_ratio(d.K_stat.values, kn)
            e["thalf_ratio_to_neutral"], e["tr_lo"], e["tr_hi"] = boot_median_ratio(d.thalf.values, thn)
            e["text_ratio_to_neutral"], e["ter_lo"], e["ter_hi"] = boot_median_ratio(d.t_cohort_ext.values, ten)
        # cohort survivor curve, lineage-number curve: median over runs at each grid time
        M = np.stack([ts_by_seed[s] for s in d.seed])   # (runs, T, cols)
        t = M[0, :, 0]
        e["t_grid"] = t.tolist()
        e["ncoh_median"] = np.median(M[:, :, 2], axis=0).tolist()
        e["nlin_median"] = np.median(M[:, :, 1], axis=0).tolist()
        e["maxsz_median"] = np.median(M[:, :, 3], axis=0).tolist()
        e["meanfit_median"] = np.median(M[:, :, 6], axis=0).tolist()
        e["hetbond_median"] = np.median(M[:, :, 8] / np.maximum(M[:, :, 9], 1), axis=0).tolist()
        per[str(th)] = e
    out["per_theta"] = per
    return out, df


# ------------------------------------------------------------------ E2
def analyse_marker(df, thr_list, tag, cond_thr):
    """Survival prob to each size threshold vs theta (ratio to neutral),
    plus conditional (max>=cond_thr) time-to-fixation vs neutral."""
    out = {"thresholds": thr_list, "cond_thr": cond_thr}
    thetas = sorted(df.theta.unique())
    per = {}
    neu = df[df.theta == 0]
    for th in thetas:
        d = df[df.theta == th]
        e = dict(theta=th, n_runs=int(len(d)), mud=float(d.mud.iloc[0]),
                 n_censored=int((d.abs_type == -1).sum()),
                 n_fixed=int((d.abs_type == 1).sum()),
                 n_extinct=int((d.abs_type == 0).sum()))
        surv = {}
        for m in thr_list:
            k = int((d.max_marker >= m).sum())
            kn = int((neu.max_marker >= m).sum())
            p, lo, hi = boot_prob(k, len(d))
            s = dict(k=k, n=int(len(d)), p=p, p_lo=lo, p_hi=hi)
            if th > 0:
                r, rlo, rhi, undef = boot_prob_ratio(k, len(d), kn, len(neu))
                s.update(ratio_to_neutral=r, r_lo=rlo, r_hi=rhi, r_undef_frac=undef)
            surv[str(m)] = s
        e["survival"] = surv
        # conditional takeover kinetics: runs reaching cond_thr and fixing
        sel = d[(d.max_marker >= cond_thr) & (d.abs_type == 1)]
        e["n_takeover"] = int(len(sel))
        if len(sel) >= 5:
            e["t_fix_median"], e["t_fix_lo"], e["t_fix_hi"] = boot_median(sel.t_abs)
            fp = sel[f"fp_{cond_thr}"].astype(float)
            e["t_condsize_to_fix_median"], e["tc_lo"], e["tc_hi"] = boot_median(
                (sel.t_abs - fp).values)
        per[str(th)] = e
    # ratios of conditional times to neutral
    neu_sel = neu[(neu.max_marker >= cond_thr) & (neu.abs_type == 1)]
    for th in thetas:
        if th == 0:
            continue
        d = df[df.theta == th]
        sel = d[(d.max_marker >= cond_thr) & (d.abs_type == 1)]
        if len(sel) >= 5 and len(neu_sel) >= 5:
            r, lo, hi = boot_median_ratio(sel.t_abs.values, neu_sel.t_abs.values)
            per[str(th)]["tfix_ratio_to_neutral"] = r
            per[str(th)]["tfr_lo"] = lo
            per[str(th)]["tfr_hi"] = hi
            fpc = f"fp_{cond_thr}"
            r2, lo2, hi2 = boot_median_ratio(
                (sel.t_abs - sel[fpc]).values.astype(float),
                (neu_sel.t_abs - neu_sel[fpc]).values.astype(float))
            per[str(th)]["tcond_ratio_to_neutral"] = r2
            per[str(th)]["tcr_lo"] = lo2
            per[str(th)]["tcr_hi"] = hi2
    out["per_theta"] = per
    out["neutral_n_takeover"] = int(len(neu_sel))
    return out


# ------------------------------------------------------------------ CS
def analyse_CS(df):
    assert (df.error == "").all()
    neu = df[df.theta == 0]
    out = {"protocol": "closed two-clone half-domain, same 192x64 lattice",
           "neutral_n": int(len(neu)),
           "neutral_median_tfix": float(neu.t_fix.median()),
           "n_censored": int((df.censored == 1).sum()),
           "per_theta": {}}
    for th in sorted(df.theta.unique()):
        if th == 0:
            continue
        d = df[df.theta == th]
        r, lo, hi = boot_median_ratio(d.t_fix.values, neu.t_fix.values)
        med, mlo, mhi = boot_median(d.t_fix.values)
        out["per_theta"][str(th)] = dict(theta=th, n=int(len(d)),
                                          median_tfix=med, med_lo=mlo, med_hi=mhi,
                                          ratio_to_neutral=r, r_lo=lo, r_hi=hi,
                                          n_censored=int((d.censored == 1).sum()))
    return out


# ------------------------------------------------------------------ main
def main():
    summ = {"track": "round-8 open-system / ongoing-nucleation variant",
            "convention": "theta = U_d/s_d = mud/sd, U_d = mud (no factor 2)",
            "seed_block": [13_520_000, 13_559_999],
            "kernel": "evolution_open.py (bit-identical to evolution_strip.py "
                      "in two-clone mode; neutral single-cell fixation "
                      "31/60000 vs 1/N=4.883e-4 on 64x32, z=0.31)"}
    parts = []

    # E1
    dfE1, tsE1 = load_E1()
    s1, dfE1 = analyse_E1(dfE1, tsE1)
    summ["E1_lineage_nucleation"] = s1
    parts.append(dfE1.assign(arm="E1_lineage"))

    # E2a / E2b
    dfa = pd.read_parquet("results/E2a_runs.parquet")
    assert (dfa.error == "").all(), dfa.error[dfa.error != ""].iloc[0]
    summ["E2a_single_cell"] = analyse_marker(
        dfa, [8, 32, 128, 256, 512, 2048, 6144, 12288], "E2a", cond_thr=2048)
    parts.append(dfa)
    dfb = pd.read_parquet("results/E2b_runs.parquet")
    assert (dfb.error == "").all(), dfb.error[dfb.error != ""].iloc[0]
    summ["E2b_disc_r3"] = analyse_marker(
        dfb, [64, 128, 256, 512, 2048, 6144, 12288], "E2b", cond_thr=256)
    parts.append(dfb)

    # CS
    dfc = pd.read_parquet("results/CS_runs.parquet")
    summ["CS_closed_same_geometry"] = analyse_CS(dfc)
    parts.append(dfc)

    # pilots (recorded, not analysed further)
    for pf, arm in [("results/pilotA_runs.parquet", "pilotA_neutral_fix_64x32"),
                    ("results/pilotB_runs.parquet", "pilotB_nu_probe")]:
        if os.path.exists(pf):
            parts.append(pd.read_parquet(pf).assign(arm=arm))

    all_runs = pd.concat(parts, ignore_index=True, sort=False)
    # drop bulky list cols if any leaked
    for c in list(all_runs.columns):
        if all_runs[c].dtype == object and all_runs[c].map(
                lambda v: isinstance(v, (list, np.ndarray))).any():
            all_runs = all_runs.drop(columns=c)
    summ["runs_total"] = int(len(all_runs))
    summ["runs_by_arm"] = {k: int(v) for k, v in all_runs.arm.value_counts().items()}
    # seed audit
    seeds = all_runs.seed.values
    summ["seed_audit"] = dict(
        n_rows=int(len(seeds)), n_unique=int(len(np.unique(seeds))),
        min=int(seeds.min()), max=int(seeds.max()),
        all_in_block=bool(((seeds >= 13_520_000) & (seeds <= 13_559_999)).all()),
        overlaps_prior_max_seed_12900000_range=bool((seeds < 13_500_000).any()))
    all_runs.to_parquet("open_system_runs.parquet")
    with open("open_system_summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print("wrote open_system_runs.parquet", all_runs.shape)
    print("wrote open_system_summary.json")
    return summ


if __name__ == "__main__":
    main()
