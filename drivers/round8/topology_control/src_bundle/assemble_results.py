"""Assemble the round-8 topology-control deliverables from the campaign
outputs: runs parquet, summary JSON, figure. (Memo written separately once
numbers are in hand.)

Inputs (work/out):
  tfix_runs_raw.parquet     T_fix campaign (hex, rand: anchors + ladders)
  census_runs_raw.parquet   contact census (hex, rand, moore control)
Reference (bundle/data):
  ridge_profile_analysis_pool.parquet  square-Moore ladder + pooled anchor
Deposited envelope numbers (task brief): Moore 0.37 [0.32,0.43],
vN 0.302-0.316 at s_d = 0.01; deposited N_eff Moore 2.900, vN 1.625.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import topo_analysis as ta  # noqa: E402

OUT = os.path.join(WORK, "out")
BUNDLE = os.path.join(os.path.dirname(WORK), "bundle", "data")

MOORE_ENV = dict(theta_star=0.37, ci=[0.32, 0.43])   # brief: Moore at s_d=0.01
VN_ENV = dict(theta_star_range=[0.302, 0.316])      # brief: vN at s_d=0.01
NEFF_DEP = dict(moore=2.900, vn=1.625)              # deposited census N_eff
N_BOOT = 4000


def load_tfix():
    df = pd.read_parquet(os.path.join(OUT, "tfix_runs_raw.parquet"))
    return df


def stencil_ridge(df, st, n_boot=N_BOOT, seed=11):
    an = df[(df.stencil == st) & (df.arm == f"{st}_anchor")]
    lad = df[(df.stencil == st) & (df.arm == f"{st}_ladder")]
    an_c = int((an.t_fix < 0).sum())
    lad_c = int((lad.t_fix < 0).sum())
    T0 = an.t_fix.values.astype(float)          # censored anchor runs would
    T0 = np.where(T0 < 0, an.numgen.values, T0)  # be right-censored at cap;
    ladv = lad.copy()                           # (none occurred; guard only)
    ladv["t_fix"] = np.where(lad.t_fix < 0, lad.numgen, lad.t_fix)
    r = ta.ridge_analysis(ladv[["theta", "t_fix"]], T0, n_boot=n_boot,
                          seed=seed)
    r["n_anchor_censored"] = an_c
    r["n_ladder_censored"] = lad_c
    r["n_ladder"] = int(len(lad))
    r["n_errors"] = int((df[df.stencil == st].error != "").sum())
    r["anchor_seed_range"] = [int(an.seed.min()), int(an.seed.max())]
    r["ladder_seed_range"] = [int(lad.seed.min()), int(lad.seed.max())]
    r["anchor_median_ci"] = ta.median_ci(T0)[1]
    return r


def moore_reference():
    ap = pd.read_parquet(os.path.join(BUNDLE,
                                      "ridge_profile_analysis_pool.parquet"))
    T0 = ap[ap.kind == "neutral"].t_fix.values.astype(float)
    lad = ap[(ap.kind == "loaded") & (ap.sd == 0.01)][["theta", "t_fix"]]
    r = ta.ridge_analysis(lad, T0, n_boot=N_BOOT, seed=12)
    r["n_ladder"] = int(len(lad))
    return r
