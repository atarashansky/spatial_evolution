"""Finite-size / boundary-condition analysis of the s_d = 0.01 crossover.

Estimators (pre-registered in fss_design.json):
  * half_drop : convention-free. Upper plateau = mean of log median T_fix over
    the two smallest thetas in the cell; lower plateau = mean over the two
    largest; theta_half = load at which log median T_fix crosses the midpoint
    (linear interpolation of log-median against log-theta, first descending
    crossing).
  * level : deposited scheme. R(theta) = median T_fix / median own-anchor
    T_fix; crossing of R = level (default 1/2) by LINEAR interpolation of R
    against log(theta) at the first descending crossing ('loglin'); isotonic
    (weighted PAVA) variant as sensitivity.
Bootstrap: run-level, every rung and the cell's own anchor resampled with
replacement within a replicate; B >= 2000; CIs = 2.5/97.5 percentiles.
Convergence: constant vs power-law-plus-plateau in L, fitted on the estimator
values with weights 1/var; fit-family bootstrap over the run-level replicates.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats


# ------------------------------------------------------------------ crossings
def loglin_cross(theta, ratio, level=0.5):
    o = np.argsort(theta)
    th = np.asarray(theta, float)[o]
    r = np.asarray(ratio, float)[o]
    for i in range(len(th) - 1):
        if (r[i] > level) and (r[i + 1] <= level):
            x0, x1 = np.log(th[i]), np.log(th[i + 1])
            y0, y1 = r[i], r[i + 1]
            return float(np.exp(x0 + (level - y0) * (x1 - x0) / (y1 - y0)))
    return np.nan


def _pava_nonincreasing(y, w):
    y = np.asarray(y, float); w = np.asarray(w, float)
    out_l, out_w, out_c = [], [], []
    for i in range(len(y)):
        out_l.append(y[i]); out_w.append(w[i]); out_c.append(1)
        while len(out_l) > 1 and out_l[-1] > out_l[-2]:
            wsum = out_w[-1] + out_w[-2]
            merged = (out_l[-1] * out_w[-1] + out_l[-2] * out_w[-2]) / wsum
            csum = out_c[-1] + out_c[-2]
            out_l = out_l[:-2] + [merged]; out_w = out_w[:-2] + [wsum]
            out_c = out_c[:-2] + [csum]
    return np.repeat(out_l, out_c)


def isotonic_cross(theta, ratio, weights=None, level=0.5):
    o = np.argsort(theta)
    th = np.asarray(theta, float)[o]
    r = np.asarray(ratio, float)[o]
    w = np.ones_like(r) if weights is None else np.asarray(weights, float)[o]
    return loglin_cross(th, _pava_nonincreasing(r, w), level=level)


def half_drop_cross(theta, logmed, n_plateau=2):
    """Convention-free crossover: crossing of log-median T_fix through the
    midpoint of the upper and lower plateau levels (means over the n_plateau
    smallest / largest thetas), first descending crossing, log-log interp."""
    o = np.argsort(theta)
    th = np.asarray(theta, float)[o]
    y = np.asarray(logmed, float)[o]
    up = float(np.mean(y[:n_plateau]))
    lo = float(np.mean(y[-n_plateau:]))
    mid = 0.5 * (up + lo)
    for i in range(len(th) - 1):
        if (y[i] > mid) and (y[i + 1] <= mid):
            x0, x1 = np.log(th[i]), np.log(th[i + 1])
            y0, y1 = y[i], y[i + 1]
            return float(np.exp(x0 + (mid - y0) * (x1 - x0) / (y1 - y0))), up, lo
    return np.nan, up, lo


# ------------------------------------------------------------------ per cell
def cell_arrays(df, cell):
    """Per-cell arrays for the estimators. Right-censored runs (t_fix == -1,
    i.e. no fixation by the frame cap `numgen`) enter as the LOWER BOUND
    t_fix := numgen. For a median-based estimator this is exact whenever the
    censored fraction of the arm is below 1/2 (all censored values lie above
    the median); the caller records the censored count per arm so the memo
    can state it. Errored rows (error != '') are excluded."""
    d = df[df["cell"] == cell].copy()
    if "error" in d.columns:
        d = d[d["error"] == ""]
    cens = d["t_fix"] <= 0
    if cens.any():
        if "numgen" in d.columns:
            d.loc[cens, "t_fix"] = d.loc[cens, "numgen"]
        else:
            raise ValueError(f"censored runs in {cell} but no numgen column to bound them")
    thetas = np.array(sorted(d.loc[d["kind"] == "loaded", "theta"].unique()))
    rung = [d.loc[(d["kind"] == "loaded") & np.isclose(d["theta"], t), "t_fix"].to_numpy(float)
            for t in thetas]
    anch = d.loc[d["kind"] == "neutral", "t_fix"].to_numpy(float)
    return thetas, rung, anch


def cell_point(thetas, rung, anch, levels=(0.4, 0.5, 0.6)):
    meds = np.array([np.median(a) for a in rung])
    ns = np.array([len(a) for a in rung])
    anch_med = float(np.median(anch)) if anch.size else np.nan
    ratio = meds / anch_med
    out = {}
    hd, up, lo = half_drop_cross(thetas, np.log(meds))
    out["half_drop"] = hd
    out["plateau_up_logmed"] = up
    out["plateau_lo_logmed"] = lo
    for lv in levels:
        out[f"level_{lv:g}"] = loglin_cross(thetas, ratio, lv)
    out["level_0.5_isotonic"] = isotonic_cross(thetas, ratio, ns, 0.5)
    out["anchor_median"] = anch_med
    out["rung_medians"] = {f"{t:g}": float(m) for t, m in zip(thetas, meds)}
    out["rung_ratio"] = {f"{t:g}": float(r) for t, r in zip(thetas, ratio)}
    out["rung_n"] = {f"{t:g}": int(n) for t, n in zip(thetas, ns)}
    out["anchor_n"] = int(anch.size)
    return out


def cell_bootstrap(thetas, rung, anch, B=2000, seed=1,
                   keys=("half_drop", "level_0.5", "level_0.4", "level_0.6",
                         "level_0.5_isotonic")):
    """Run-level bootstrap: resample every rung and the anchor with replacement.
    Returns dict key -> array of B replicate estimates (nan when undefined)."""
    rng = np.random.default_rng(seed)
    ns = np.array([len(a) for a in rung])
    out = {k: np.full(B, np.nan) for k in keys}
    for b in range(B):
        meds = np.array([np.median(a[rng.integers(0, a.size, a.size)]) for a in rung])
        an = np.median(anch[rng.integers(0, anch.size, anch.size)]) if anch.size else np.nan
        ratio = meds / an
        if "half_drop" in out:
            out["half_drop"][b] = half_drop_cross(thetas, np.log(meds))[0]
        for lv in (0.4, 0.5, 0.6):
            k = f"level_{lv:g}"
            if k in out:
                out[k][b] = loglin_cross(thetas, ratio, lv)
        if "level_0.5_isotonic" in out:
            out["level_0.5_isotonic"][b] = isotonic_cross(thetas, ratio, ns, 0.5)
    return out


def summarize_boot(point, boots):
    """[value, lo, hi, frac_valid] per key."""
    res = {}
    for k, v in point.items():
        if k in boots:
            arr = boots[k]
            ok = np.isfinite(arr)
            if ok.sum() >= 20:
                lo, hi = np.percentile(arr[ok], [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            res[k] = [float(v), float(lo), float(hi)]
            res[k + "_boot_valid_frac"] = float(ok.mean())
    return res


# ------------------------------------------------------------------ convergence
def fit_constant(logy, w):
    mu = float(np.sum(w * logy) / np.sum(w))
    rss = float(np.sum(w * (logy - mu) ** 2))
    return mu, rss


def _plpl(L, li, la, b):
    """log theta*(L) for theta* = exp(li) + exp(la) * L^-b (positive amplitude)."""
    return np.log(np.exp(li) + np.exp(la) * L ** (-b))


def fit_powerlaw_plateau(L, logy, w, b0=(0.3, 1.0, 2.0)):
    """Weighted NLS of log theta* = log(theta_inf + A L^-b), A > 0, b in (0, 5].
    Also tries the negative-amplitude branch A < 0 (rising toward plateau).
    Returns dict with params of best fit and rss."""
    best = None
    y0 = float(np.average(logy, weights=w))
    for sign in (+1.0, -1.0):
        def model(L, li, la, b, sign=sign):
            base = np.exp(li) + sign * np.exp(la) * L ** (-b)
            base = np.where(base <= 1e-12, 1e-12, base)
            return np.log(base)
        for b in b0:
            p0 = [y0, np.log(0.1 * np.exp(y0) + 1e-9) + b * np.log(L.min()), b]
            try:
                popt, _ = optimize.curve_fit(model, L, logy, p0=p0, sigma=1.0 / np.sqrt(w),
                                            bounds=([-20, -30, 0.02], [10, 30, 5.0]),
                                            maxfev=20000)
            except Exception:
                continue
            resid = logy - model(L, *popt)
            rss = float(np.sum(w * resid ** 2))
            if best is None or rss < best["rss"]:
                best = dict(theta_inf=float(np.exp(popt[0])), sign=sign,
                            logA=float(popt[1]), b=float(popt[2]), rss=rss,
                            popt=[float(x) for x in popt])
    return best


def aicc_wls(rss, w, n, p):
    """AICc for weighted least squares with known variances 1/w (Gaussian)."""
    ll = -0.5 * (np.sum(np.log(2 * np.pi / w)) + rss)
    aic = 2 * p - 2 * ll
    aicc = aic + (2 * p * (p + 1)) / (n - p - 1) if n - p - 1 > 0 else np.inf
    return float(aic), float(aicc), float(ll)


def convergence_fit(L, val, boot_reps):
    """L: array of sizes; val: point estimates; boot_reps: (B, n) replicate matrix
    (columns aligned with L). Fits constant and power-law-plus-plateau; bootstraps
    both by refitting on each replicate row."""
    L = np.asarray(L, float); val = np.asarray(val, float)
    logy = np.log(val)
    lb = np.log(boot_reps)
    v = np.nanvar(lb, axis=0)
    w = 1.0 / v
    n = len(L)
    mu, rss_c = fit_constant(logy, w)
    aic_c, aicc_c, _ = aicc_wls(rss_c, w, n, 1)
    pp = fit_powerlaw_plateau(L, logy, w)
    aic_p, aicc_p, _ = aicc_wls(pp["rss"], w, n, 3)
    # F-test constant vs 3-parameter model
    df1, df2 = 2, n - 3
    if df2 > 0 and pp["rss"] > 0:
        F = ((rss_c - pp["rss"]) / df1) / (pp["rss"] / df2)
        Fp = float(stats.f.sf(F, df1, df2))
    else:
        F, Fp = np.nan, np.nan
    # bootstrap the fits
    Bc, Binf, Bb, BA = [], [], [], []
    ok_rows = np.all(np.isfinite(lb), axis=1)
    lbo = lb[ok_rows]
    for row in lbo:
        Bc.append(fit_constant(row, w)[0])
        p2 = fit_powerlaw_plateau(L, row, w, b0=(pp["b"],))
        if p2 is not None:
            Binf.append(np.log(max(p2["theta_inf"], 1e-12)))
            Bb.append(p2["b"])
    Bc = np.array(Bc); Binf = np.array(Binf); Bb = np.array(Bb)
    # monotone-drift sign test over successive differences (replicate-wise)
    diffs = np.diff(lbo, axis=1)
    frac_all_neg = float(np.mean(np.all(diffs < 0, axis=1)))
    frac_all_pos = float(np.mean(np.all(diffs > 0, axis=1)))
    slope = np.array([np.polyfit(np.log(L), row, 1)[0] for row in lbo])
    return dict(
        n_points=int(n), weights=[float(x) for x in w],
        constant=dict(theta_inf=[float(np.exp(mu))] + [float(x) for x in
                      np.exp(np.percentile(Bc, [2.5, 97.5]))],
                      rss=rss_c, aic=aic_c, aicc=aicc_c),
        powerlaw_plus_plateau=dict(
            theta_inf=[pp["theta_inf"]] + [float(x) for x in
                       np.exp(np.percentile(Binf, [2.5, 97.5]))] if Binf.size else [pp["theta_inf"], np.nan, np.nan],
            b=[pp["b"]] + [float(x) for x in np.percentile(Bb, [2.5, 97.5])] if Bb.size else [pp["b"], np.nan, np.nan],
            amplitude_sign=pp["sign"], rss=pp["rss"], aic=aic_p, aicc=aicc_p,
            popt=pp["popt"]),
        dAICc_powerlaw_minus_constant=float(aicc_p - aicc_c),
        dAIC_powerlaw_minus_constant=float(aic_p - aic_c),
        F_test=dict(F=float(F), p=Fp, df=[df1, df2]),
        drift_sign_test=dict(frac_reps_all_successive_diffs_negative=frac_all_neg,
                             frac_reps_all_successive_diffs_positive=frac_all_pos),
        loglog_slope=[float(np.polyfit(np.log(L), logy, 1)[0])] +
                     [float(x) for x in np.percentile(slope, [2.5, 97.5])],
        n_boot_rows_used=int(ok_rows.sum()),
    )
