"""Round-9 nu-sweep analysis: load-driven coarsening acceleration vs nu.

Reads all per-run JSONL shards, builds one row per run (parquet), and
computes per-(arm, nu, theta) statistics with percentile-bootstrap 95% CIs:

  * coarsening clock: median first-passage frame of the surviving t=0-cohort
    lineage count N_s to each level k in NS0_THR (exact, uncensored where
    the level is reached); acceleration A(k) = med_neutral / med_loaded;
  * cohort-area half-time (cohort cell count first <= N/2), from the 24/dec
    time-series interpolated on log-time;
  * cohort extinction time: Kaplan-Meier median with right-censoring;
  * compartment statistics time-averaged over the last decade of each run:
    largest-lineage fraction f1, second fraction f2, K, b12/bdiff;
  * closed-system T_fix medians and load ratios (CB burnt-in, CF fresh);
  * collapse test: acceleration vs largest-lineage fraction across (nu,theta),
    Spearman + isotonic-regression R^2, and comparison to the closed points.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

WS = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(WS, "nusweep_runs")
N_CELLS = 192 * 64
RNG = np.random.default_rng(20260716)
B_BOOT = 4000


# ------------------------------------------------------------------ load
def load_runs():
    rows = []
    for fn in sorted(os.listdir(OUTDIR)):
        if not fn.startswith("shard_"):
            continue
        with open(os.path.join(OUTDIR, fn)) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


# ------------------------------------------------------------------ per-run derived quantities
def interp_first_passage_down(t, y, level):
    """First time the (non-increasing on average, sampled) series y falls to
    <= level, log-time interpolated between the bracketing samples.
    Returns np.nan if never reached (censored)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    idx = np.where(y <= level)[0]
    if idx.size == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return t[0]
    y0, y1 = y[j - 1], y[j]
    t0, t1 = t[j - 1], t[j]
    if y0 == y1:
        return t1
    # linear in log t
    frac = (y0 - level) / (y0 - y1)
    lt = np.log(t0) + frac * (np.log(t1) - np.log(t0))
    return float(np.exp(lt))


def derive_ns_row(r):
    """Flatten one NS-arm run into a dict of scalars."""
    ts_f = np.asarray(r["ts_frame"], float)
    K = np.asarray(r["ts_K"], float)
    mx = np.asarray(r["ts_mx"], float)
    mx2 = np.asarray(r["ts_mx2"], float)
    b12f = np.asarray(r["ts_b12frac"], float)
    hetf = np.asarray(r["ts_hetfrac"], float)
    coh = np.asarray(r["ts_coh"], float)
    live = np.asarray(r["ts_live"], float)
    d = dict(arm=r["arm"], nu=r["nu"], theta=r["theta"], seed=r["seed"],
             wall=r["wall"], t_abs=r["t_abs"], abs_type=r["abs_type"],
             frames_run=r["frames_run"], n_nuc=r["n_nuc"],
             n_capped=r["n_capped"], K0=r["K0"], ns0_end=r["ns0_end"],
             coh_live_end=r["coh_live_end"])
    # exact first passages
    thr = np.asarray(r["ns0_thr"], int)
    fp = np.asarray(r["ns0_fp"], float)
    cohat = np.asarray(r["ns0_coh"], float)
    fp[fp < 0] = np.nan
    for k, f, c in zip(thr, fp, cohat):
        d["fp_%d" % k] = f
        d["cohat_%d" % k] = c if np.isfinite(f) else np.nan
    # cohort extinction: level 0
    d["t_ext"] = d.get("fp_0", np.nan)
    d["ext_observed"] = np.isfinite(d["t_ext"])
    # cohort-area half time (coh <= N/2) from the sampled series
    d["t_cohalf"] = interp_first_passage_down(ts_f, coh, 0.5 * live.max())
    # stationary compartment statistics: last-decade window of the run
    if ts_f.size >= 3:
        tmax = ts_f.max()
        w = ts_f >= 0.1 * tmax
        if w.sum() < 3:
            w = np.zeros_like(ts_f, bool)
            w[-3:] = True
        d["K_stat"] = float(np.mean(K[w]))
        d["f1_stat"] = float(np.mean(mx[w] / np.maximum(live[w], 1)))
        d["f2_stat"] = float(np.mean(mx2[w] / np.maximum(live[w], 1)))
        d["b12_stat"] = float(np.mean(b12f[w]))
        d["het_stat"] = float(np.mean(hetf[w]))
        d["f1_max"] = float(np.max(mx / np.maximum(live, 1)))
    else:
        for k in ["K_stat", "f1_stat", "f2_stat", "b12_stat", "het_stat", "f1_max"]:
            d[k] = np.nan
    return d


def derive_closed_row(r):
    return dict(arm=r["arm"], nu=0.0, theta=r["theta"], seed=r["seed"],
                wall=r["wall"], t_abs=r["t_abs"], abs_type=r["abs_type"],
                frames_run=r["frames_run"], winner=r.get("winner", -1),
                field_meanfit_at_mark=r.get("field_meanfit_at_mark", np.nan),
                field_sdfit_at_mark=r.get("field_sdfit_at_mark", np.nan))


# ------------------------------------------------------------------ statistics
def boot_median_ci(x, B=B_BOOT):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    idx = RNG.integers(0, n, size=(B, n))
    meds = np.median(x[idx], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)), n


def boot_ratio_of_medians(x_num, x_den, B=B_BOOT):
    """Ratio median(x_num)/median(x_den) with independent bootstrap of both."""
    a = np.asarray(x_num, float); a = a[np.isfinite(a)]
    b = np.asarray(x_den, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan
    ia = RNG.integers(0, a.size, size=(B, a.size))
    ib = RNG.integers(0, b.size, size=(B, b.size))
    ra = np.median(a[ia], axis=1)
    rb = np.median(b[ib], axis=1)
    r = ra / rb
    return float(np.median(a) / np.median(b)), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_mean_ci(x, B=B_BOOT):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan, 0
    idx = RNG.integers(0, x.size, size=(B, x.size))
    m = np.mean(x[idx], axis=1)
    return float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(x.size)


def km_median(times, observed):
    """Kaplan-Meier median with right censoring (times = event or censor time).
    Returns (median, S at censor horizon).  median = nan if S never <= 0.5."""
    t = np.asarray(times, float)
    e = np.asarray(observed, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    n = t.size
    at_risk = n
    S = 1.0
    med = np.nan
    for i in range(n):
        if e[i]:
            S *= (at_risk - 1) / at_risk
            if S <= 0.5 and np.isnan(med):
                med = t[i]
        at_risk -= 1
    return med, S


def boot_km_median_ratio(t_num, e_num, t_den, e_den, B=1500):
    a = np.asarray(t_num, float); ea = np.asarray(e_num, bool)
    b = np.asarray(t_den, float); eb = np.asarray(e_den, bool)
    m_a, _ = km_median(a, ea)
    m_b, _ = km_median(b, eb)
    if not (np.isfinite(m_a) and np.isfinite(m_b)):
        return np.nan, np.nan, np.nan
    rs = []
    na, nb = a.size, b.size
    for _ in range(B):
        ia = RNG.integers(0, na, na)
        ib = RNG.integers(0, nb, nb)
        ma, _ = km_median(a[ia], ea[ia])
        mb, _ = km_median(b[ib], eb[ib])
        if np.isfinite(ma) and np.isfinite(mb) and mb > 0:
            rs.append(ma / mb)
    rs = np.asarray(rs)
    if rs.size < 100:
        return float(m_a / m_b), np.nan, np.nan
    return float(m_a / m_b), float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def isotonic_r2(x, y, increasing=True):
    """PAVA isotonic regression; returns fitted values and R^2 (1 - SSE/SST)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return x, y, np.nan
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    if not increasing:
        ys = -ys
    # PAVA
    n = ys.size
    w = np.ones(n)
    v = ys.copy().astype(float)
    lvl = list(range(n))
    i = 0
    vals = list(v); wts = list(w); cnt = [1] * n
    j = 0
    while j < len(vals) - 1:
        if vals[j] > vals[j + 1]:
            nw = wts[j] + wts[j + 1]
            nv = (vals[j] * wts[j] + vals[j + 1] * wts[j + 1]) / nw
            nc = cnt[j] + cnt[j + 1]
            vals[j:j + 2] = [nv]; wts[j:j + 2] = [nw]; cnt[j:j + 2] = [nc]
            if j > 0:
                j -= 1
        else:
            j += 1
    fit = np.concatenate([np.full(c, val) for val, c in zip(vals, cnt)])
    if not increasing:
        fit = -fit
        ys = -ys
    sst = float(np.sum((ys - ys.mean()) ** 2))
    sse = float(np.sum((ys - fit) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    xf = xs; yf = fit
    return xf, yf, r2


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return np.nan
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ main
NS_LEVELS_REPORT = [8192, 6144, 3072, 1024, 256, 64, 16, 8, 4, 2, 1, 0]


def analyse(rows=None, out_prefix="nu_sweep"):
    if rows is None:
        rows = load_runs()
    ns = [derive_ns_row(r) for r in rows if r["arm"] == "NS"]
    cl = [derive_closed_row(r) for r in rows if r["arm"] in ("CB", "CF")]
    dfn = pd.DataFrame(ns)
    dfc = pd.DataFrame(cl)

    summary = dict(convention="theta = U_d/s_d = mud/sd; sd = 0.01; L = 192x64",
                   levels_reported=NS_LEVELS_REPORT, cells={}, closed={},
                   collapse={}, run_counts={})

    # ---------------- closed-system reference (CB burnt-in, CF fresh)
    for arm in ["CB", "CF"]:
        sub = dfc[dfc.arm == arm]
        if sub.empty:
            continue
        armd = {}
        # censoring: abs_type 1 = fixed; -1 = cap hit
        neut = sub[sub.theta == 0.0]
        t_n = neut.t_abs.where(neut.abs_type == 1, neut.frames_run).values
        e_n = (neut.abs_type == 1).values
        m_n, S_n = km_median(t_n, e_n)
        m_n_boot = boot_median_ci(neut.t_abs[neut.abs_type == 1].values)
        armd["theta_0.0"] = dict(n=int(len(neut)), n_censored=int((~e_n).sum()),
                                km_median_tfix=m_n,
                                median_tfix_obs=m_n_boot[0], med_lo=m_n_boot[1],
                                med_hi=m_n_boot[2])
        for th in sorted(sub.theta.unique()):
            if th == 0.0:
                continue
            s2 = sub[sub.theta == th]
            t_l = s2.t_abs.where(s2.abs_type == 1, s2.frames_run).values
            e_l = (s2.abs_type == 1).values
            m_l, _ = km_median(t_l, e_l)
            r, rlo, rhi = boot_km_median_ratio(t_l, e_l, t_n, e_n)
            armd["theta_%.1f" % th] = dict(n=int(len(s2)), n_censored=int((~e_l).sum()),
                                          km_median_tfix=m_l,
                                          ratio_to_neutral=r, r_lo=rlo, r_hi=rhi)
        summary["closed"][arm] = armd

    # ---------------- open-system cells
    per_cell = {}
    for nu in sorted(dfn.nu.unique()):
        neut = dfn[(dfn.nu == nu) & (dfn.theta == 0.0)]
        for th in sorted(dfn[dfn.nu == nu].theta.unique()):
            sub = dfn[(dfn.nu == nu) & (dfn.theta == th)]
            key = "nu=%g,theta=%.1f" % (nu, th)
            c = dict(nu=nu, theta=th, n=int(len(sub)),
                     n_capped_runs=int((sub.n_capped > 0).sum()),
                     frac_cohort_extinct=float(sub.ext_observed.mean()))
            # stationary compartment statistics
            for col in ["K_stat", "f1_stat", "f2_stat", "b12_stat", "het_stat", "f1_max"]:
                m, lo, hi, n = boot_mean_ci(sub[col].values)
                c[col] = m; c[col + "_lo"] = lo; c[col + "_hi"] = hi
            # coarsening clock medians + accelerations at each level
            c["clock"] = {}
            for k in NS_LEVELS_REPORT:
                col = "fp_%d" % k
                if col not in sub:
                    continue
                med, lo, hi, n = boot_median_ci(sub[col].values)
                ent = dict(median=med, lo=lo, hi=hi, n_reached=int(n),
                           frac_reached=float(np.isfinite(sub[col]).mean()))
                if th > 0.0 and col in neut:
                    # acceleration = neutral median / loaded median
                    a, alo, ahi = boot_ratio_of_medians(neut[col].values, sub[col].values)
                    ent.update(accel=a, accel_lo=alo, accel_hi=ahi)
                c["clock"][str(k)] = ent
            # cohort-area half time
            med, lo, hi, n = boot_median_ci(sub.t_cohalf.values)
            c["t_cohalf"] = dict(median=med, lo=lo, hi=hi, n_reached=int(n),
                                 frac_reached=float(np.isfinite(sub.t_cohalf).mean()))
            if th > 0.0:
                a, alo, ahi = boot_ratio_of_medians(neut.t_cohalf.values, sub.t_cohalf.values)
                c["t_cohalf"].update(accel=a, accel_lo=alo, accel_hi=ahi)
            # cohort extinction (KM)
            t_l = np.where(sub.ext_observed, sub.t_ext, sub.frames_run).astype(float)
            e_l = sub.ext_observed.values.astype(bool)
            m_l, S_end = km_median(t_l, e_l)
            c["t_ext_km"] = dict(median=float(m_l) if np.isfinite(m_l) else None,
                                 n_censored=int((~e_l).sum()))
            if th > 0.0:
                t_n = np.where(neut.ext_observed, neut.t_ext, neut.frames_run).astype(float)
                e_n = neut.ext_observed.values.astype(bool)
                r, rlo, rhi = boot_km_median_ratio(t_n, e_n, t_l, e_l)  # neutral/loaded
                c["t_ext_km"].update(accel=r, accel_lo=rlo, accel_hi=rhi)
            per_cell[key] = c
    summary["cells"] = per_cell

    # ---------------- collapse test: acceleration (level 2 clock) vs f1_stat
    for th in [0.4, 0.8]:
        pts = []
        for key, c in per_cell.items():
            if c["theta"] != th:
                continue
            a = c["clock"].get("2", {}).get("accel", np.nan)
            f1 = c["f1_stat"]
            pts.append((c["nu"], f1, a, c["clock"].get("2", {}).get("accel_lo", np.nan),
                        c["clock"].get("2", {}).get("accel_hi", np.nan)))
        pts.sort()
        f1s = np.array([p[1] for p in pts], float)
        accs = np.array([p[2] for p in pts], float)
        _, _, r2 = isotonic_r2(f1s, accs, increasing=True)
        rho_f1 = spearman(f1s, accs)
        rho_nu = spearman(-np.log10([p[0] for p in pts]), accs)
        # closed-system reference acceleration at this theta (CB burnt-in)
        cb = summary["closed"].get("CB", {}).get("theta_%.1f" % th, {})
        cf = summary["closed"].get("CF", {}).get("theta_%.1f" % th, {})
        summary["collapse"]["theta_%.1f" % th] = dict(
            points=[dict(nu=p[0], f1=p[1], accel_level2=p[2], lo=p[3], hi=p[4]) for p in pts],
            isotonic_r2_accel_vs_f1=r2, spearman_accel_vs_f1=rho_f1,
            spearman_accel_vs_inv_nu=rho_nu,
            closed_burnin_accel=(1.0 / cb["ratio_to_neutral"]) if cb.get("ratio_to_neutral") else None,
            closed_fresh_accel=(1.0 / cf["ratio_to_neutral"]) if cf.get("ratio_to_neutral") else None,
        )

    # run counts
    summary["run_counts"] = dict(
        NS=int((dfn.arm == "NS").sum()) if len(dfn) else 0,
        CB=int((dfc.arm == "CB").sum()) if len(dfc) else 0,
        CF=int((dfc.arm == "CF").sum()) if len(dfc) else 0,
        total=int(len(rows)))
    return dfn, dfc, summary


if __name__ == "__main__":
    dfn, dfc, summary = analyse()
    dfn.to_parquet(os.path.join(WS, "nu_sweep_ns_rows.parquet"))
    dfc.to_parquet(os.path.join(WS, "nu_sweep_closed_rows.parquet"))
    json.dump(summary, open(os.path.join(WS, "nu_sweep_summary.json"), "w"),
              indent=1, default=float)
    print("rows:", len(dfn), len(dfc))
