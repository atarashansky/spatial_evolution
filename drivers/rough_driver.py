
import sys, json, numpy as np, time
sys.path.insert(0, ".")
import evolution_strip_mixed as esm

def front_stats(field, kind):
    """field: (H,W) snapshot int8. kind 'push': boundary between pusher (>=2) and dB/vacancy;
    kind 'label': boundary between label 0 and label 1 (dB two-clone interface, pushers absent)."""
    H, W = field.shape
    if kind == "push":
        occ = (field >= 2)
    else:  # clone A = labels {0,2}, clone B = {1,3}; vacancy -1 counted with left region by nearest-column rule
        occ = ((field == 0) | (field == 2))
    # per-row: rightmost occupied column of the region growing from the left, and count of "islands"/holes
    hpos = np.full(H, np.nan); overhang = 0; total = 0
    for y in range(H):
        row = occ[y]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        right = idx.max(); left_fill = row[: right + 1]
        hpos[y] = right + 0.5
        holes = np.count_nonzero(~left_fill)          # non-region sites left of the front tip
        beyond = np.count_nonzero(row[right + 1:])     # (0 by construction) 
        overhang += holes; total += (right + 1)
    valid = ~np.isnan(hpos)
    if valid.sum() < H * 0.8:
        return None
    h = hpos[valid]
    w2 = np.var(h)
    porosity = overhang / max(total, 1)
    # capillary spectrum of the single-valued front proxy
    hh = h - h.mean()
    hq = np.fft.rfft(hh) / len(hh)
    Sq = (np.abs(hq) ** 2)
    return dict(w2=float(w2), mean=float(h.mean()), porosity=float(porosity),
                Sq=[float(v) for v in Sq[1:9]])

def run(arm, seed, L_long=384, L_short=64, numgen=600_000, snap_stride=20_000):
    common = dict(L_long=L_long, L_short=L_short, numgen=numgen, snap_stride=snap_stride,
                  seed=seed, stop_fix=False, stop_push_frac=0.995, sampling=5000)
    if arm == "pusher_loaded":
        r = esm.evolve_strip_mixed(sd=0.04, mud=0.1, mub=0.0, mu_c=0.0, b_rate=0.075,
                                   pusher_cols=(1, 12), split=L_long // 2, **common)
        kind = "push"
    elif arm == "pusher_neutral":
        r = esm.evolve_strip_mixed(sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=0.075,
                                   pusher_cols=(1, 12), split=L_long // 2, **common)
        kind = "push"
    elif arm == "fitness_loaded":
        r = esm.evolve_strip_mixed(sd=0.04, mud=0.1, mub=0.0, mu_c=0.0, b_rate=0.0,
                                   pusher_cols=None, split=L_long // 2, **common)
        kind = "label"
    elif arm == "neutral":
        r = esm.evolve_strip_mixed(sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=0.0,
                                   pusher_cols=None, split=L_long // 2, **common)
        kind = "label"
    else:
        raise ValueError(arm)
    out = dict(arm=arm, seed=seed, t_fix=int(r["t_fix"]), frames=[], stats=[])
    if r["snap"] is not None:
        for k in range(r["snap"].shape[0]):
            st = front_stats(r["snap"][k], kind)
            out["frames"].append(int(r["snap_frame"][k])); out["stats"].append(st)
    out["n_push_final"] = int(r["n_push_final"])
    return out

if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    t0 = time.time()
    res = run(arm, seed)
    res["wall_s"] = round(time.time() - t0, 1)
    with open(f"rough_out/{arm}_{seed}.json", "w") as f:
        json.dump(res, f)
