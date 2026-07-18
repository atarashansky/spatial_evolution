"""Driver for the click-onset-tracks-ridge test (round 6).

One task = one two-clone strip run of sdr_kernel.run_sdr (192x32, dB Moran,
Moore, mu_b = 0), reduced to per-run click statistics.  The strict
least-loaded-class clock kmin_c(t) (minimum deleterious-hit count over all
live cells of clone c) is monotone non-decreasing per clone, so cumulative
click counts over any window are exact endpoint differences of the census
series.  We record, per clone c in {A, B}:

  * dkmin over the round-5 window [0.05 T_end, 0.85 T_end] (own-life window)
  * dkmin over the full run [first census, last census with the clone alive]
  * kmin_c evaluated on relative-life fractions FREL of T_end
  * kmin_c evaluated on a fixed absolute-frame grid FABS (NaN once fixed)

plus t_fix, winner, and mean-load / interface-height diagnostics.  Times in
frames; tau (divisions per site) = frame * p_die is applied in analysis.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdr_kernel  # noqa: E402

# relative-life fractions and absolute-frame grid used for the kmin snapshots
FREL = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9,
                 0.95, 1.0])
FABS = np.unique(np.round(np.geomspace(2_000, 40_000_000, 160)).astype(np.int64))
BIG = 1 << 40


def _at_frames(frame, series, targets):
    """Value of a monotone census series at the last census frame <= target
    (frame array sorted ascending).  target below the first census frame -> the
    initial value 0 (both clones start with zero hits)."""
    idx = np.searchsorted(frame, targets, side="right") - 1
    out = np.zeros(targets.shape, np.float64)
    ok = idx >= 0
    out[ok] = series[idx[ok]]
    return out


def summarize(r, win_lo=0.05, win_hi=0.85):
    p = r["params"]
    frame = r["frame"].astype(np.int64)
    n = r["n"].astype(np.int64)          # (T, 2) live cells per clone
    kmin = r["kmin"].astype(np.int64)    # (T, 2) strict least-loaded class
    kminI = r["kminI"].astype(np.int64)  # interfacial version
    sk = r["sk"].astype(np.float64)
    t_fix = int(r["t_fix"])
    censored = int(t_fix < 0)
    T_end = int(t_fix if t_fix > 0 else (frame[-1] if frame.size else 0))
    out = dict(t_fix=t_fix, censored=censored, T_end=T_end, winner=int(r["winner"]),
               n_samp=int(frame.size), p_die=float(p["p_die"]), M=int(p["M"]))
    if frame.size < 4:
        out["ok"] = 0
        return out
    out["ok"] = 1
    # sanitize kmin where a clone has zero live cells (last census frame at
    # fixation can be an empty-clone sentinel)
    kk = kmin.astype(np.float64).copy()
    alive = n > 0
    kk[~alive] = np.nan
    # last census frame at which each clone is alive
    last_alive = [int(np.max(np.where(alive[:, c])[0])) if alive[:, c].any() else -1
                  for c in (0, 1)]
    # ---- round-5 own-life window [0.05, 0.85] T_end
    lo, hi = win_lo * T_end, win_hi * T_end
    w = (frame >= lo) & (frame <= hi)
    if w.sum() < 2:
        w = np.ones(frame.size, bool)
    idx = np.where(w)[0]
    i0, i1 = idx[0], idx[-1]
    out["w_i0_frame"] = int(frame[i0])
    out["w_i1_frame"] = int(frame[i1])
    for c, tag in ((0, "A"), (1, "B")):
        k0 = kk[i0, c]
        k1 = kk[min(i1, last_alive[c]), c] if last_alive[c] >= 0 else np.nan
        out[f"dkmin_win_{tag}"] = float(k1 - k0)
        # full-run clicks: from first census (kmin ~ 0) to last alive census
        kend = kk[last_alive[c], c] if last_alive[c] >= 0 else np.nan
        out[f"dkmin_full_{tag}"] = float(kend - kk[0, c])
        out[f"kmin_first_{tag}"] = float(kk[0, c])
        out[f"kmin_lastalive_{tag}"] = float(kend)
        # interfacial least-loaded class increment over the same window (diagnostic)
        kI = kminI[:, c].astype(np.float64)
        kIa = np.where((r["nI"][:, c] > 0), kI, np.nan)
        out[f"dkminI_win_{tag}"] = float(kIa[min(i1, last_alive[c])] - kIa[i0]) \
            if last_alive[c] >= 0 else np.nan
        # mean load increment over the window
        mk = sk[:, c] / np.maximum(n[:, c], 1)
        out[f"dmeank_win_{tag}"] = float(mk[i1] - mk[i0])
        # kmin snapshots on relative-life fractions (last census frame <= f*T_end)
        rel_targets = np.floor(FREL * T_end).astype(np.int64)
        vals_rel = _at_frames(frame, kk[:, c], rel_targets)
        # clamp to last-alive census value for fractions at/near fixation
        if last_alive[c] >= 0:
            la = frame[last_alive[c]]
            vals_rel = np.where(rel_targets > la, kk[last_alive[c], c], vals_rel)
        out[f"kmin_rel_{tag}"] = vals_rel.astype(np.float32)
        # kmin snapshots on the fixed absolute grid: NaN after fixation
        vals_abs = _at_frames(frame, kk[:, c], FABS)
        vals_abs = np.where(FABS <= T_end, vals_abs, np.nan)
        if last_alive[c] >= 0:
            la = frame[last_alive[c]]
            vals_abs = np.where((FABS > la) & (FABS <= T_end), kk[last_alive[c], c],
                                vals_abs)
        out[f"kmin_abs_{tag}"] = vals_abs.astype(np.float32)
    hbar = r["hbar"].astype(np.float64)
    out["h0"] = float(hbar[0])
    out["h_end"] = float(hbar[-1])
    out["n_frames_series"] = int(frame.size)
    return out


def one_run(task):
    """task: dict with run_sdr kwargs + bookkeeping (arm, ratio, sd_tag ...)."""
    kw = {k: v for k, v in task.items()
          if k in ("L_long", "L_short", "numgen", "dt", "sd", "mud", "sampling",
                   "avg", "seed", "iface_w", "periodic_y", "moore", "com_stride",
                   "init_lnfB")}
    import time
    t0 = time.time()
    r = sdr_kernel.run_sdr(**kw)
    s = summarize(r)
    s["wall_s"] = float(time.time() - t0)
    for k in ("arm", "stage", "sd", "mud", "ratio", "seed", "L_long", "L_short",
              "sampling", "numgen", "rung"):
        if k in task:
            s[k] = task[k]
    return s
