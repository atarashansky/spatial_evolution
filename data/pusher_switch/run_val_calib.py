"""Validation ensemble (all-pusher vs Bd, KS on turnover clock) +
kinetic-neutrality calibration sweep (b*), parallel over 60 workers."""
import sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np
from concurrent.futures import ProcessPoolExecutor

P_DIE = 1.0 - np.exp(-1.16e-5 * 2400.0)


def val_bd(seed):
    from evolution_strip_bd import evolve_strip_bd
    r = evolve_strip_bd(L_long=96, L_short=32, numgen=6_000_000,
                        sd=0.0, mud=0.0, mub=0.0, sampling=10_000, seed=seed)
    return dict(kind="bd", seed=seed, t_fix=r["t_fix"], winner=r["winner"])


def val_mixed(seed):
    from evolution_strip_mixed import evolve_strip_mixed
    bb = 20 * P_DIE
    r = evolve_strip_mixed(L_long=96, L_short=32, numgen=6_000_000,
                           sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=bb,
                           pusher_all=True, sampling=10_000, seed=seed)
    occ = r["live"][-1] / r["params"]["M"] if len(r["live"]) else 1.0
    return dict(kind="mixed_allpush", seed=seed, t_fix=r["t_fix"],
                winner=r["winner"], b=bb, occ=float(occ))


def calib(args):
    b, seed = args
    from evolution_strip_mixed import evolve_strip_mixed
    # 4-column pusher patch at the right wall, labels aligned (B = patch);
    # neutral fitness; velocity from n_push series + absorption outcome.
    r = evolve_strip_mixed(L_long=192, L_short=64, numgen=1_500_000,
                           sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=b,
                           split=188, pusher_cols=(189, 192),
                           sampling=5_000, com_stride=500,
                           stop_push_frac=0.999, seed=seed)
    pc = r["push_com"]          # n_push every 500 frames
    fixed = 1 if (r["t_stop_push"] > 0 or r["winner"] == 1) else \
            (0 if r["winner"] == 0 else -1)
    # early-window slope (frames 0..100k) incl. absorption padding
    n_early = 200
    y = np.zeros(n_early)
    m = min(len(pc), n_early)
    y[:m] = pc[:m]
    if m < n_early:                       # absorbed: pad with final state
        y[m:] = r["params"]["M"] if fixed == 1 else 0
    return dict(kind="calib", b=b, seed=seed, fixed=fixed,
                t_end=int(r["t_stop_push"] if r["t_stop_push"] > 0
                          else r["t_fix"]),
                slope=float(np.polyfit(np.arange(n_early) * 500, y, 1)[0]),
                y100k=float(y[-1]))


if __name__ == "__main__":
    jobs = []
    for s in range(200):
        jobs.append(("bd", s))
        jobs.append(("mx", s))
    bs = [0.02, 0.04, 0.06, 0.08, 0.11, 0.15]
    for b in bs:
        for s in range(400):
            jobs.append(("cal", (b, 1000 + s)))

    out = []
    with ProcessPoolExecutor(max_workers=60) as ex:
        futs = []
        for kind, arg in jobs:
            fn = {"bd": val_bd, "mx": val_mixed, "cal": calib}[kind]
            futs.append(ex.submit(fn, arg))
        for i, f in enumerate(futs):
            out.append(f.result())
            if (i + 1) % 400 == 0:
                print(f"{i+1}/{len(futs)}", flush=True)

    with open("val_calib_results.json", "w") as fh:
        json.dump(out, fh)
    print("DONE", len(out))
