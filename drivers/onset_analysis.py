"""Analysis for the click-onset-tracks-ridge test.

Definitions (following round 5 exactly):
  ridge(s_d)  = U_d/s_d at which the loaded/neutral MEDIAN fixation-time ratio
                crosses one half; log-linear (log x, log y) interpolation between
                adjacent ladder rungs; run-level bootstrap (resample runs within
                each rung AND the neutral anchor runs).
  onset(s_d)  = U_d/s_d at which the fraction of runs whose two clones'
                cumulative strict-ratchet click counts DIFFER (dkmin_A != dkmin_B
                over the round-5 own-life window [0.05, 0.85] T_end) crosses one
                half; log-linear (log x, linear y) interpolation; run-level
                bootstrap resampling runs within each rung.
The onset/ridge RATIO is bootstrapped jointly: within a replicate the SAME
resampled ladder runs feed both the ridge and the onset (paired), so the CI on
the ratio accounts for their shared run population.
"""
import numpy as np
import pandas as pd


def cross_loglog(x, y, level=0.5):
    """First downward crossing of `level` by y(x); interpolation in (log x, log y)."""
    for i in range(len(x) - 1):
        if y[i] >= level > y[i + 1] and y[i] > 0 and y[i + 1] > 0:
            x0, x1 = np.log(x[i]), np.log(x[i + 1])
            y0, y1 = np.log(y[i]), np.log(y[i + 1])
            return float(np.exp(x0 + (np.log(level) - y0) * (x1 - x0) / (y1 - y0)))
    return np.nan


def cross_loglin(x, y, level=0.5):
    """First crossing of `level` by y(x); interpolation in (log x, linear y)."""
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            x0, x1 = np.log(x[i]), np.log(x[i + 1])
            return float(np.exp(x0 + (level - y[i]) * (x1 - x0) / (y[i + 1] - y[i])))
    return np.nan


def _ridge_from_groups(ratios, tf_groups, T0):
    med0 = np.median(T0)
    meds = np.array([np.median(g) for g in tf_groups]) / med0
    return meds, cross_loglog(ratios, meds, 0.5)


def _onset_from_groups(ratios, nz_groups, level=0.5):
    fnz = np.array([np.mean(g) for g in nz_groups])
    return fnz, cross_loglin(ratios, fnz, level)


def analyze_sd(lad, T0, ratios=None, onset_key="nz_win", n_boot=2000, seed=1,
               onset_level=0.5):
    """lad: DataFrame of loaded ladder runs at ONE s_d with columns
    'ratio', 't_fix', and boolean column `onset_key` (nonzero click difference).
    T0: array of neutral fixation times.
    Returns dict with ridge, onset, ratio and their bootstrap CIs (paired)."""
    rng = np.random.default_rng(seed)
    if ratios is None:
        ratios = np.sort(lad.ratio.unique())
    ratios = np.asarray(ratios, np.float64)
    tf_groups = [lad.t_fix.values[lad.ratio.values == r].astype(np.float64) for r in ratios]
    nz_groups = [lad[onset_key].values[lad.ratio.values == r].astype(np.float64) for r in ratios]
    meds, ridge = _ridge_from_groups(ratios, tf_groups, T0)
    fnz, onset = _onset_from_groups(ratios, nz_groups, onset_level)
    ratio_pt = onset / ridge if np.isfinite(onset) and np.isfinite(ridge) and ridge > 0 else np.nan
    rb = np.empty(n_boot); ob = np.empty(n_boot); qb = np.empty(n_boot)
    n0 = T0.size
    for b in range(n_boot):
        T0b = T0[rng.integers(0, n0, n0)]
        tfg = []; nzg = []
        for tf, nz in zip(tf_groups, nz_groups):
            k = tf.size
            idx = rng.integers(0, k, k)          # SAME resample for ridge & onset (paired)
            tfg.append(tf[idx]); nzg.append(nz[idx])
        _, rb[b] = _ridge_from_groups(ratios, tfg, T0b)
        _, ob[b] = _onset_from_groups(ratios, nzg, onset_level)
        qb[b] = ob[b] / rb[b] if (np.isfinite(ob[b]) and np.isfinite(rb[b]) and rb[b] > 0) else np.nan

    def ci(v):
        v = np.asarray(v, np.float64)
        f = np.mean(np.isfinite(v))
        lo, hi = (np.nanpercentile(v, [2.5, 97.5]) if f > 0 else (np.nan, np.nan))
        return float(lo), float(hi), float(f)

    r_lo, r_hi, r_def = ci(rb)
    o_lo, o_hi, o_def = ci(ob)
    q_lo, q_hi, q_def = ci(qb)
    return dict(
        ratios=ratios.tolist(),
        n_runs_per_rung=[int(g.size) for g in tf_groups],
        median_tfix_ratio=meds.tolist(),
        frac_nonzero_diff=fnz.tolist(),
        ridge=float(ridge), ridge_lo=r_lo, ridge_hi=r_hi, ridge_boot_defined=r_def,
        onset=float(onset), onset_lo=o_lo, onset_hi=o_hi, onset_boot_defined=o_def,
        onset_over_ridge=float(ratio_pt), oor_lo=q_lo, oor_hi=q_hi, oor_boot_defined=q_def,
        neutral_median=float(np.median(T0)), n_neutral=int(n0), n_boot=n_boot,
        boots=dict(ridge=rb, onset=ob, oor=qb),
    )
