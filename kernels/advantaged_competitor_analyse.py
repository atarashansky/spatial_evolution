"""Analysis for the round-9 advantaged-competitor track.

Inputs : out/prod.parquet (one row per run) [+ optional extra parquets]
Outputs: advantaged_competitor_summary.json, fig_advantaged_competitor.png,
         numbers used by advantaged_competitor_memo.md

Definitions
-----------
survival to k     : fp_k >= 0  (marker reached size >= k at some frame; the
                    first-passage record, robust to later collapse)
neutral anchor    : theta=0 (mud=0) runs of the SAME marker rule; at mud=0 the
                    burnt-in field is exactly all-ones, so 'clean' shares the
                    matched (s_b=0) anchor and 'drcleanXX' shares 'driverXX'.
survival ratio    : P_loaded(>=k) / P_anchor(>=k), bootstrap CI (resample the
                    binomial numerator and denominator independently).
conditional speed : among runs reaching k_hi, front-advance time
                    dt = fp_{k_hi} - fp_{k_lo}; effective linear front speed
                    v = (sqrt(k_hi) - sqrt(k_lo)) * L_short_eff / dt is not
                    needed - we report the RATIO of median dt (loaded/anchor),
                    whose inverse is the sweep-speed ratio.
"""
import json
import numpy as np
import pandas as pd

SD = 0.01
RNG = np.random.default_rng(20260718)
BOOT = 4000
KS = [8, 32, 128, 256, 512, 2048, 6144]          # survival thresholds reported
KMAIN = 512                                       # headline "established" size
# arm -> anchor arm sharing its theta=0 rows (verified bit-identical)
ANCHOR_OF = {
    "matched": "matched", "clean": "matched",
    "driver02": "driver02", "drclean02": "driver02",
    "driver05": "driver05", "drclean05": "driver05",
    "driver10": "driver10", "drclean10": "driver10",
}
ARM_LABEL = {
    "matched": "matched (prior arm)", "clean": "A clean-birth",
    "driver02": "B driver s_b=0.02", "driver05": "B driver s_b=0.05",
    "driver10": "B driver s_b=0.10",
    "drclean02": "C driver+clean 0.02", "drclean05": "C driver+clean 0.05",
    "drclean10": "C driver+clean 0.10",
}


def load(paths):
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["theta"] = (df["mud"] / SD).round(3)
    return df


def surv_p(sub, k):
    x = (sub[f"fp_{k}"].to_numpy() >= 0)
    return int(x.sum()), int(x.size)


def boot_ratio(k1, n1, k0, n0, boot=BOOT):
    """ratio p1/p0 with bootstrap CI; NaN if p0 == 0."""
    if n1 == 0 or n0 == 0 or k0 == 0:
        return dict(ratio=float("nan"), lo=float("nan"), hi=float("nan"))
    p1 = k1 / n1
    p0 = k0 / n0
    b1 = RNG.binomial(n1, p1, boot) / n1
    b0 = RNG.binomial(n0, p0, boot) / n0
    ok = b0 > 0
    r = b1[ok] / b0[ok]
    return dict(ratio=p1 / p0, lo=float(np.percentile(r, 2.5)),
                hi=float(np.percentile(r, 97.5)))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def survival_table(df):
    """per (arm, radius, theta): P(>=k) for k in KS, with Wilson CI, and the
    ratio to the arm's shared theta=0 anchor (same radius) with bootstrap CI."""
    rows = []
    for (arm, rad, th), g in df.groupby(["arm", "radius", "theta"]):
        anch_arm = ANCHOR_OF[arm]
        A = df[(df.arm == anch_arm) & (df.radius == rad) & (df.theta == 0.0)]
        for k in KS:
            k1, n1 = surv_p(g, k)
            k0, n0 = surv_p(A, k)
            lo, hi = wilson(k1, n1)
            br = boot_ratio(k1, n1, k0, n0)
            rows.append(dict(arm=arm, radius=int(rad), theta=float(th), k=int(k),
                             surv_k=k1, n=n1, p=(k1 / n1 if n1 else np.nan),
                             p_lo=lo, p_hi=hi, anchor_arm=anch_arm,
                             anchor_k=k0, anchor_n=n0,
                             p_anchor=(k0 / n0 if n0 else np.nan),
                             ratio=br["ratio"], ratio_lo=br["lo"],
                             ratio_hi=br["hi"]))
    return pd.DataFrame(rows)


def speed_table(df, k_lo=32, k_hi=2048):
    """conditional front-advance time from k_lo to k_hi among runs reaching
    k_hi; ratio of medians loaded/anchor (loaded FASTER -> ratio < 1) with a
    bootstrap CI on the ratio of medians."""
    rows = []

    def dt(sub):
        m = (sub[f"fp_{k_hi}"] >= 0) & (sub[f"fp_{k_lo}"] >= 0)
        d = (sub.loc[m, f"fp_{k_hi}"] - sub.loc[m, f"fp_{k_lo}"]).to_numpy()
        return d[d > 0]

    for (arm, rad, th), g in df.groupby(["arm", "radius", "theta"]):
        anch_arm = ANCHOR_OF[arm]
        A = df[(df.arm == anch_arm) & (df.radius == rad) & (df.theta == 0.0)]
        d1 = dt(g)
        d0 = dt(A)
        rec = dict(arm=arm, radius=int(rad), theta=float(th), k_lo=k_lo,
                   k_hi=k_hi, n_cond=int(d1.size), n_cond_anchor=int(d0.size),
                   dt_med=(float(np.median(d1)) if d1.size else np.nan),
                   dt_med_anchor=(float(np.median(d0)) if d0.size else np.nan))
        if d1.size >= 5 and d0.size >= 5:
            r_obs = np.median(d1) / np.median(d0)
            bs = np.empty(BOOT)
            for b in range(BOOT):
                r1 = RNG.choice(d1, d1.size)
                r0 = RNG.choice(d0, d0.size)
                bs[b] = np.median(r1) / np.median(r0)
            rec.update(dt_ratio=float(r_obs),
                       dt_ratio_lo=float(np.percentile(bs, 2.5)),
                       dt_ratio_hi=float(np.percentile(bs, 97.5)),
                       # sweep-speed ratio = inverse of the time ratio
                       speed_ratio=float(1.0 / r_obs),
                       speed_ratio_lo=float(1.0 / np.percentile(bs, 97.5)),
                       speed_ratio_hi=float(1.0 / np.percentile(bs, 2.5)))
        else:
            rec.update(dt_ratio=np.nan, dt_ratio_lo=np.nan, dt_ratio_hi=np.nan,
                       speed_ratio=np.nan, speed_ratio_lo=np.nan,
                       speed_ratio_hi=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def fixation_table(df):
    rows = []
    for (arm, rad, th), g in df.groupby(["arm", "radius", "theta"]):
        n = len(g)
        nf = int((g.abs_type == 1).sum())
        lo, hi = wilson(nf, n)
        rows.append(dict(arm=arm, radius=int(rad), theta=float(th), n=n,
                         n_fixed=nf, p_fix=nf / n, p_fix_lo=lo, p_fix_hi=hi,
                         n_censored=int((g.censored == 1).sum())))
    return pd.DataFrame(rows)
