"""Assemble per-run records (pickles from run_onset_sweep.py) into tidy tables."""
import pickle
import numpy as np
import pandas as pd
from onset_driver import FREL, FABS

SCALAR_KEYS = ["t_fix", "censored", "T_end", "winner", "n_samp", "p_die", "M", "ok",
               "w_i0_frame", "w_i1_frame", "h0", "h_end", "n_frames_series", "wall_s",
               "arm", "stage", "sd", "mud", "ratio", "seed", "L_long", "L_short",
               "sampling", "numgen", "rung"]
CLOCK_KEYS = ["dkmin_win_A", "dkmin_win_B", "dkmin_full_A", "dkmin_full_B",
              "kmin_first_A", "kmin_first_B", "kmin_lastalive_A", "kmin_lastalive_B",
              "dkminI_win_A", "dkminI_win_B", "dmeank_win_A", "dmeank_win_B"]


def load_records(paths):
    recs = []
    for p in paths:
        with open(p, "rb") as fh:
            recs.extend(pickle.load(fh))
    return recs


def flat_table(recs):
    rows = []
    for r in recs:
        row = {k: r.get(k, np.nan) for k in SCALAR_KEYS + CLOCK_KEYS}
        rows.append(row)
    df = pd.DataFrame(rows)
    # derived per-run click statistics
    dA, dB = df["dkmin_win_A"], df["dkmin_win_B"]
    fA, fB = df["dkmin_full_A"], df["dkmin_full_B"]
    df["dkmin_diff_win"] = dA - dB
    df["nz_win"] = (dA != dB) & np.isfinite(dA) & np.isfinite(dB)      # onset criterion (round 5)
    df["click_any_win"] = ((dA > 0) | (dB > 0)) & np.isfinite(dA) & np.isfinite(dB)
    df["dkmin_diff_full"] = fA - fB
    df["nz_full"] = (fA != fB) & np.isfinite(fA) & np.isfinite(fB)
    df["click_any_full"] = ((fA > 0) | (fB > 0)) & np.isfinite(fA) & np.isfinite(fB)
    df["mean_clicks_win"] = 0.5 * (dA + dB)
    df["mean_clicks_full"] = 0.5 * (fA + fB)
    return df


def snapshot_table(recs, kind="rel"):
    """Long-format kmin snapshots. kind='rel' -> columns for FREL fractions of
    T_end; kind='abs' -> columns for the absolute frame grid FABS (NaN after
    fixation). One row per run; wide arrays kept as object columns is heavy, so
    we return a dict of 2D arrays plus the aligned scalar frame instead."""
    key = "kmin_rel" if kind == "rel" else "kmin_abs"
    grid = FREL if kind == "rel" else FABS
    A = np.full((len(recs), grid.size), np.nan, np.float32)
    B = np.full((len(recs), grid.size), np.nan, np.float32)
    for i, r in enumerate(recs):
        a = r.get(f"{key}_A"); b = r.get(f"{key}_B")
        if a is not None and np.size(a) == grid.size:
            A[i] = a
        if b is not None and np.size(b) == grid.size:
            B[i] = b
    return grid, A, B
