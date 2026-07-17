"""Frequency-dependent selection: production ensemble (P1-P3).

P1 loaded transition hunt : b0 in {0.1,0.15,0.2,0.3,0.5}, Lambda=1e-3, n=400, cap 30M, 192x32
P2 neutral escape scaling : b0 in {0, 0.002,0.005,0.01}, n=300, cap 60M, 128x24
P3 Lambda x b0 phase spine: Lambda in {2.5e-4,1e-3,4e-3} x b0 in {0.02,0.05,0.1,0.2}, n=200, cap 30M, 192x32

Incremental checkpointing: partial parquet rewritten atomically every 100
completions; per-phase parquet written when a phase completes; progress.json
for cheap polling.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

OUT = "fd_prod"
os.makedirs(OUT, exist_ok=True)

BASE = dict(sb=0.1, mub=0.0, sampling=2500, com_stride=0, prof_halfwidth=8,
            ben_cap=1024)


def one(idx, phase, kw):
    from evolution_strip_fd import evolve_strip_fd
    t0 = time.time()
    r = evolve_strip_fd(**kw)
    p = r["params"]
    return dict(idx=idx, phase=phase, t_fix=r["t_fix"], winner=r["winner"],
                seed=p["seed"], b0=p["b0"], bonus_mode=p["bonus_mode"],
                mud=p["mud"], sd=p["sd"], mub=p["mub"], sb=p["sb"],
                L_long=p["L_long"], L_short=p["L_short"], numgen=p["numgen"],
                fitA_end=float(r["fitA"][-1]) if len(r["fitA"]) else 1.0,
                fitB_end=float(r["fitB"][-1]) if len(r["fitB"]) else 1.0,
                core_s=time.time() - t0)


def cfg(phase, seed, **kw):
    d = dict(BASE)
    d.update(kw)
    d["seed"] = seed
    return (phase, d)


def build_tasks():
    tasks = []
    sd_seed = 700000

    # --- P1: loaded transition hunt (Lambda = 1e-3) -----------------------
    for b0 in [0.1, 0.15, 0.2, 0.3, 0.5]:
        for i in range(400):
            sd_seed += 1
            tasks.append(cfg("P1", sd_seed, L_long=192, L_short=32,
                             numgen=30_000_000, mud=0.1, sd=0.01,
                             b0=b0, bonus_mode="decay"))

    # --- P2: neutral escape scaling (smaller M) ---------------------------
    # b0=0 arm added beyond the memo spec: same-geometry neutral anchor for
    # the escape-scaling fit (T depends on M, can't borrow the 192x32 pilot).
    for b0 in [0.0, 0.002, 0.005, 0.01]:
        for i in range(300):
            sd_seed += 1
            tasks.append(cfg("P2", sd_seed, L_long=128, L_short=24,
                             numgen=60_000_000, mud=0.0, sd=0.0, b0=b0,
                             bonus_mode=("off" if b0 == 0.0 else "decay")))

    # --- P3: Lambda x b0 phase-diagram spine ------------------------------
    for lam_sd in [0.0025, 0.01, 0.04]:          # Lambda = 0.1 * sd
        for b0 in [0.02, 0.05, 0.1, 0.2]:
            for i in range(200):
                sd_seed += 1
                tasks.append(cfg("P3", sd_seed, L_long=192, L_short=32,
                                 numgen=30_000_000, mud=0.1, sd=lam_sd,
                                 b0=b0, bonus_mode="decay"))
    return tasks


def main():
    tasks = build_tasks()

    # LPT-ish ordering: long caps and high b0 (likely censored) first
    def prio(i):
        phase, kw = tasks[i]
        return -(kw["numgen"] * (1.0 + 10.0 * kw["b0"]) *
                 (2.0 if kw["mud"] == 0.0 else 1.0))
    order = sorted(range(len(tasks)), key=prio)
    PH_TOTAL = {}
    for ph, _ in tasks:
        PH_TOTAL[ph] = PH_TOTAL.get(ph, 0) + 1
    print(f"{len(tasks)} tasks: {PH_TOTAL}", flush=True)

    rows, t_start = [], time.time()
    done_phase = {ph: 0 for ph in PH_TOTAL}
    written_phase = set()

    def checkpoint(final=False):
        df = pd.DataFrame(rows)
        df["load"] = df["mud"] > 0
        df["censored"] = df["t_fix"] < 0
        df["t_eff"] = np.where(df["censored"], df["numgen"], df["t_fix"])
        df["Lambda"] = df["mud"] * df["sd"]
        tmp = os.path.join(OUT, "_tmp.parquet")
        df.to_parquet(tmp)
        os.replace(tmp, os.path.join(OUT, "fd_production_partial.parquet"))
        for ph in PH_TOTAL:
            if final or (done_phase[ph] == PH_TOTAL[ph]
                         and ph not in written_phase):
                sub = df[df["phase"] == ph]
                if len(sub):
                    sub.to_parquet(tmp)
                    os.replace(tmp, os.path.join(OUT, f"fd_{ph.lower()}.parquet"))
                    written_phase.add(ph)
        prog = dict(done=len(rows), total=len(tasks),
                    per_phase={ph: [done_phase[ph], PH_TOTAL[ph]]
                               for ph in PH_TOTAL},
                    elapsed_h=round((time.time() - t_start) / 3600, 3),
                    core_h=round(df["core_s"].sum() / 3600, 2))
        with open(os.path.join(OUT, "_progress_tmp.json"), "w") as f:
            json.dump(prog, f)
        os.replace(os.path.join(OUT, "_progress_tmp.json"),
                   os.path.join(OUT, "progress.json"))

    gen = Parallel(n_jobs=60, return_as="generator_unordered",
                   prefer="processes", batch_size=1)(
        delayed(one)(i, tasks[i][0], tasks[i][1]) for i in order)

    for row in gen:
        rows.append(row)
        done_phase[row["phase"]] += 1
        if len(rows) % 100 == 0:
            checkpoint()
            print(f"{len(rows)}/{len(tasks)} "
                  f"{ {ph: done_phase[ph] for ph in sorted(PH_TOTAL)} } "
                  f"{(time.time()-t_start)/3600:.2f}h", flush=True)

    checkpoint(final=True)
    with open(os.path.join(OUT, "DONE"), "w") as f:
        f.write("ok\n")
    print(f"ALL DONE {len(rows)} rows in {(time.time()-t_start)/3600:.2f}h",
          flush=True)


if __name__ == "__main__":
    main()
