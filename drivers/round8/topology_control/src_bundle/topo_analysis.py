"""Round-8 topology-control analysis: ridge (T_fix crossing) per stencil,
run-level bootstrap CIs, N_eff (opposite-clone contact) per stencil, and the
verdict against the pre-stated Moore/vN envelope.

Estimator = the deposited round-5/6/7 ridge convention verbatim
(drivers/onset_analysis.py): median loaded T_fix / median NEUTRAL anchor
T_fix per theta rung; ridge = theta at which the median ratio first crosses
one-half, log-linear (log theta, linear ratio) interpolation between the
bracketing rungs; run-level bootstrap resampling runs within every rung AND
within the neutral anchor (percentile CI). Isotonic (PAVA on log-theta) and
log-log crossings are reported as scheme sensitivity, as in
ridge_profile_summary.json.
"""
import numpy as np
import pandas as pd


def cross_loglin(x, y, level=0.5):
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            x0, x1 = np.log(x[i]), np.log(x[i + 1])
            return float(np.exp(x0 + (level - y[i]) * (x1 - x0)
                                / (y[i + 1] - y[i])))
    return np.nan


def cross_loglog(x, y, level=0.5):
    for i in range(len(x) - 1):
        if y[i] >= level > y[i + 1] and y[i] > 0 and y[i + 1] > 0:
            x0, x1 = np.log(x[i]), np.log(x[i + 1])
            y0, y1 = np.log(y[i]), np.log(y[i + 1])
            return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0)
                                / (y1 - y0)))
    return np.nan


def _pava_decreasing(y, w=None):
    """Weighted isotonic regression enforcing a NON-INCREASING sequence."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    # decreasing fit == increasing fit on reversed order
    yy = y[::-1].copy(); ww = w[::-1].copy()
    n = len(yy)
    lv = list(yy); wt = list(ww); cnt = [1] * n
    i = 0
    while i < len(lv) - 1:
        if lv[i] > lv[i + 1]:
            tot = wt[i] + wt[i + 1]
            lv[i] = (lv[i] * wt[i] + lv[i + 1] * wt[i + 1]) / tot
            wt[i] = tot; cnt[i] += cnt[i + 1]
            del lv[i + 1], wt[i + 1], cnt[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.repeat(lv, cnt)
    return out[::-1]


def cross_isotonic(x, y, level=0.5, n_grid=400):
    yi = _pava_decreasing(y)
    return cross_loglin(x, yi, level)


def ratio_ladder(lad, T0, thetas=None):
    thetas = np.sort(lad.theta.unique()) if thetas is None else np.asarray(thetas, float)
    groups = [lad.t_fix.values[lad.theta.values == t].astype(np.float64)
              for t in thetas]
    med0 = float(np.median(T0))
    ratio = np.array([np.median(g) for g in groups]) / med0
    return thetas, groups, ratio, med0


def ridge_analysis(lad, T0, n_boot=3000, seed=1, thetas=None):
    """lad: DataFrame(theta, t_fix) loaded runs; T0: neutral anchor T_fix array.
    Returns point estimates under three schemes + run-level bootstrap of the
    loglin (primary) and isotonic crossings, plus the rung ratio table with
    per-rung bootstrap ratio CIs."""
    rng = np.random.default_rng(seed)
    thetas, groups, ratio, med0 = ratio_ladder(lad, T0, thetas)
    pt = dict(loglin=cross_loglin(thetas, ratio, 0.5),
              isotonic=cross_isotonic(thetas, ratio, 0.5),
              loglog=cross_loglog(thetas, ratio, 0.5))
    rb = np.empty(n_boot); ib = np.empty(n_boot)
    ratio_b = np.empty((n_boot, len(thetas)))
    for b in range(n_boot):
        m0 = np.median(T0[rng.integers(0, T0.size, T0.size)])
        rt = np.array([np.median(g[rng.integers(0, g.size, g.size)])
                       for g in groups]) / m0
        ratio_b[b] = rt
        rb[b] = cross_loglin(thetas, rt, 0.5)
        ib[b] = cross_isotonic(thetas, rt, 0.5)

    def ci(v):
        v = np.asarray(v, float)
        f = float(np.mean(np.isfinite(v)))
        if f == 0:
            return [np.nan, np.nan], f, np.nan
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        return [float(lo), float(hi)], f, float(np.nanmedian(v))

    ci_l, fv_l, med_l = ci(rb)
    ci_i, fv_i, med_i = ci(ib)
    rlo, rhi = np.nanpercentile(ratio_b, [2.5, 97.5], axis=0)
    return dict(
        theta_star=pt["loglin"], ci=ci_l, boot_frac_valid=fv_l,
        boot_median=med_l, var_log=float(np.nanvar(np.log(rb))),
        theta_star_isotonic=pt["isotonic"], ci_isotonic=ci_i,
        boot_frac_valid_isotonic=fv_i, boot_median_isotonic=med_i,
        theta_star_loglog=pt["loglog"],
        anchor_median=med0, n_anchor=int(T0.size),
        rungs=[dict(theta=float(t), n=int(g.size),
                    median_tfix=float(np.median(g)),
                    ratio=float(r), ratio_ci=[float(lo), float(hi)])
               for t, g, r, lo, hi in zip(thetas, groups, ratio, rlo, rhi)],
        boots=rb, boots_isotonic=ib,
    )


def bootstrap_diff(rb_a, rb_b):
    """Paired-by-index bootstrap of ratio and difference (independent runs)."""
    ok = np.isfinite(rb_a) & np.isfinite(rb_b)
    a, b = rb_a[ok], rb_b[ok]
    lr = np.log(a / b)
    return dict(
        ratio_median=float(np.median(a / b)),
        ratio_ci=[float(np.percentile(a / b, 2.5)), float(np.percentile(a / b, 97.5))],
        diff_median=float(np.median(a - b)),
        diff_ci=[float(np.percentile(a - b, 2.5)), float(np.percentile(a - b, 97.5))],
        p_ratio_gt_2=float(np.mean(a / b > 2.0)),
        p_ratio_lt_half=float(np.mean(a / b < 0.5)),
        n_valid=int(ok.sum()),
    )


def median_ci(x, n_boot=3000, seed=2):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([np.median(x[rng.integers(0, x.size, x.size)])
                  for _ in range(n_boot)])
    return float(np.median(x)), [float(np.percentile(m, 2.5)),
                                 float(np.percentile(m, 97.5))]


def mean_ci(x, n_boot=3000, seed=3):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = np.array([np.mean(x[rng.integers(0, x.size, x.size)])
                  for _ in range(n_boot)])
    return float(np.mean(x)), [float(np.percentile(m, 2.5)),
                               float(np.percentile(m, 97.5))]
