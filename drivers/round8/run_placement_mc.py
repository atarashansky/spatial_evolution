"""Track B: Monte-Carlo tissue-placement propagation.

For each of four tissues, propagate the FULL stated input box (log-uniform mud and
sd within the stated ranges), a resampled ridge drawn from the measured
heterogeneity (log ridge ~ Normal(log 0.47, tau); default tau = 0.2345 from the
RE pooled fit; the SE of the pooled mean is optionally added in quadrature as a
sensitivity variant), and the 2x elongation shape correction with its own
uncertainty (multiplier log-uniform on [1.3, 3]).  100,000 draws per tissue.

margin (decades) = log10( theta_placement / (ridge_draw * shape_draw) ),
theta_placement = mud / sd (U_d = mud, NO factor of 2).  Fluid side = margin > 0.

RNG seeds from the assigned block 13,560,000-13,569,999 (MC use only).
Outputs -> out_mc/: placement_mc_summary.json, placement_mc_draws.parquet (downsampled)
"""
import os, json, time
import numpy as np
import pandas as pd

SEED_BASE = 13_560_000            # this track's block: 13,560,000 - 13,569,999
N_DRAWS = 100_000
OUT = "out_mc"
os.makedirs(OUT, exist_ok=True)

TISSUES = {
    "Colonic crypt":        dict(mud=(0.016, 0.2),  sd=(5e-4, 0.01), seed=SEED_BASE + 0),
    "Epidermis (sun-exp.)": dict(mud=(0.1,   0.8),  sd=(5e-4, 0.01), seed=SEED_BASE + 1),
    "HSC (blood)":          dict(mud=(0.14,  0.8),  sd=(5e-4, 0.01), seed=SEED_BASE + 2),
    "Hypermutator tumour":  dict(mud=(0.4,   0.8),  sd=(0.001, 0.03), seed=SEED_BASE + 3),
}
SD_MIN_SAMPLED = 0.0025           # lowest s_d rung actually measured in the ladder

# ridge and shape uncertainty (from data/ridge_profile_summary.json, RE pool)
RIDGE_MU_LOG = np.log(0.4747811432038846)   # RE pooled point 0.4748
RIDGE_MU_SE  = 0.11531145187808832          # SE of pooled log-mean
RIDGE_TAU    = 0.23450199946945133          # between-condition SD (task: tau=0.23)
SHAPE_LO, SHAPE_HI = 1.3, 3.0

# deposited single-corner joint worst case (canonical convention, +0.13 dec colon)
RIDGE_HI_DET = 0.5951803044845069           # RE pooled 95% upper bound
SHAPE_DET = 2.0


def loguniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def run_tissue(name, spec, ridge_mode, shape_mode, n=N_DRAWS):
    rng = np.random.default_rng(spec["seed"] + {"pred": 0, "predse": 10, "fixed": 20}[ridge_mode]
                                * 100 + {"lu": 0, "fixed": 1}[shape_mode] * 1000)
    mud = loguniform(rng, *spec["mud"], n)
    sd = loguniform(rng, *spec["sd"], n)
    theta = mud / sd
    if ridge_mode == "pred":          # prediction interval: heterogeneity only
        ridge = np.exp(rng.normal(RIDGE_MU_LOG, RIDGE_TAU, n))
    elif ridge_mode == "predse":      # heterogeneity + uncertainty of the pooled mean
        ridge = np.exp(rng.normal(RIDGE_MU_LOG, np.hypot(RIDGE_TAU, RIDGE_MU_SE), n))
    else:                             # fixed at the pooled point
        ridge = np.full(n, np.exp(RIDGE_MU_LOG))
    if shape_mode == "lu":
        shape = loguniform(rng, SHAPE_LO, SHAPE_HI, n)
    else:
        shape = np.full(n, SHAPE_DET)
    margin = np.log10(theta / (ridge * shape))
    return pd.DataFrame(dict(tissue=name, mud=mud, sd=sd, theta=theta, ridge=ridge,
                             shape=shape, margin_dec=margin, fluid=margin > 0,
                             extrapolated=sd < SD_MIN_SAMPLED))


def summarise(df, name, spec):
    inr = df[~df.extrapolated]
    ext = df[df.extrapolated]
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")
    # single-corner deterministic joint worst case (canonical convention)
    theta_lo = spec["mud"][0] / spec["sd"][1]
    theta_c = np.sqrt(spec["mud"][0] * spec["mud"][1]) / np.sqrt(spec["sd"][0] * spec["sd"][1])
    theta_hi = spec["mud"][1] / spec["sd"][0]
    m_worst = float(np.log10(theta_lo / (RIDGE_HI_DET * SHAPE_DET)))
    m_centre_pooled = float(np.log10(theta_c / (np.exp(RIDGE_MU_LOG) * SHAPE_DET)))
    return dict(
        tissue=name, n=int(len(df)),
        p_fluid=float(df.fluid.mean()),
        margin_median_dec=q(df.margin_dec, 50),
        margin_p5_dec=q(df.margin_dec, 5), margin_p95_dec=q(df.margin_dec, 95),
        margin_p2_5_dec=q(df.margin_dec, 2.5), margin_p97_5_dec=q(df.margin_dec, 97.5),
        margin_mean_dec=float(df.margin_dec.mean()),
        frac_extrapolated_sd=float(df.extrapolated.mean()),
        p_fluid_in_range=float(inr.fluid.mean()) if len(inr) else float("nan"),
        p_fluid_extrapolated=float(ext.fluid.mean()) if len(ext) else float("nan"),
        n_in_range=int(len(inr)), n_extrapolated=int(len(ext)),
        margin_median_in_range=q(inr.margin_dec, 50) if len(inr) else float("nan"),
        margin_median_extrapolated=q(ext.margin_dec, 50) if len(ext) else float("nan"),
        deterministic=dict(theta_lo=float(theta_lo), theta_c=float(theta_c), theta_hi=float(theta_hi),
                           margin_centre_pooled_dec=m_centre_pooled,
                           margin_jointworst_dec=m_worst, fluid_at_jointworst=bool(m_worst > 0)),
    )


t0 = time.time()
summary = dict(track="Round 8: Monte-Carlo tissue-placement propagation",
               convention="theta = U_d/s_d = mud/sd, U_d = mud (NO factor of 2); fluid = margin>0; margin = log10(theta/(ridge*shape))",
               inputs=dict(ridge=dict(mu=float(np.exp(RIDGE_MU_LOG)), tau=RIDGE_TAU, mu_se=RIDGE_MU_SE,
                                      pred_interval=[float(np.exp(RIDGE_MU_LOG - 1.96 * RIDGE_TAU)),
                                                     float(np.exp(RIDGE_MU_LOG + 1.96 * RIDGE_TAU))]),
                           shape=dict(lo=SHAPE_LO, hi=SHAPE_HI, det=SHAPE_DET, dist="log-uniform"),
                           n_draws=N_DRAWS, seed_block=[SEED_BASE, SEED_BASE + 9999],
                           sd_min_sampled=SD_MIN_SAMPLED),
               variants={})
draws_keep = []
for ridge_mode in ("pred", "predse", "fixed"):
    for shape_mode in ("lu", "fixed"):
        key = f"ridge_{ridge_mode}__shape_{shape_mode}"
        rows = []
        for name, spec in TISSUES.items():
            df = run_tissue(name, spec, ridge_mode, shape_mode)
            rows.append(summarise(df, name, spec))
            if ridge_mode == "pred" and shape_mode == "lu":   # PRIMARY: keep downsampled draws
                keep = df.sample(20_000, random_state=spec["seed"])
                draws_keep.append(keep)
        summary["variants"][key] = rows
        prim = " (PRIMARY)" if key == "ridge_pred__shape_lu" else ""
        print(f"{key}{prim}: " + "; ".join(f"{r['tissue']} pfluid={r['p_fluid']:.4f} med={r['margin_median_dec']:+.2f}"
                                             for r in rows), flush=True)
summary["primary"] = "ridge_pred__shape_lu"
summary["wall_s"] = time.time() - t0
json.dump(summary, open(f"{OUT}/placement_mc_summary.json", "w"), indent=1)
pd.concat(draws_keep, ignore_index=True).to_parquet(f"{OUT}/placement_mc_draws.parquet", index=False)
print(f"[done] wall {time.time()-t0:.1f}s draws kept {sum(len(d) for d in draws_keep)}", flush=True)
