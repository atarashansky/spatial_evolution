"""Round-9 nu-sweep production driver.

Arms
----
NS   open system, mode 0 (every live cell a lineage at t=0 after a 300k-frame
     unlabelled burn-in; new lineage on each birth with prob nu).
     nu in {5e-6, 2e-5, 5e-5, 1.5e-4, 5e-4, 1.5e-3}; theta in {0, 0.4, 0.8}
     (mud = theta * sd); sd = 0.01; 192x64 strip; per (nu, theta) N runs.
CB   closed two-clone half-domain WITH the same 300k-frame burn-in (mode 2,
     T_burn = 300000): the like-for-like nu -> 0 end point (both compartments
     start at the load steady state, exactly as every open lineage does).
CF   closed two-clone half-domain, FRESH field (mode 2, T_burn = 0):
     replication of the deposited round-8 closed-system reference.

Per-run outputs (one JSON line per run, appended atomically to a shard file):
    exact cohort-survivor first-passage frames (thresholds), cohort area at
    each crossing, stationary K / max-size / second-max / heterolabel bond
    stats time-averaged over the stationary window, coarsening time series
    on a 24/decade grid, t_abs / abs_type, wall time.

Usage
-----
    python nu_sweep_driver.py plan   > handoff/tasks.json     (writes task list)
    python nu_sweep_driver.py run --workers 60 [--tasks handoff/tasks.json]

Seed block 13,700,000-13,737,999 (production); pilots 13,738,000-13,739,999.
"""
import argparse
import json
import os
import sys
import time
import concurrent.futures as cf

import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WS)

SD = 0.01
L_LONG, L_SHORT = 192, 64
T_BURN = 300_000
NU_GRID = [5e-6, 2e-5, 5e-5, 1.5e-4, 5e-4, 1.5e-3]
THETAS = [0.0, 0.4, 0.8]
OUTDIR = os.path.join(WS, "nusweep_runs")

# cohort-survivor thresholds (descending); K0 = number of live cells at
# marking (12288 unless deaths coincide); we record the frame at which the
# number of surviving t=0 lineages first drops to <= each threshold.
NS0_THR = np.array([12000, 8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512,
                    384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 3,
                    2, 1, 0], np.int64)


def geom_grid(t0, t1, per_dec=24):
    n = int(np.round(np.log10(t1 / t0) * per_dec)) + 1
    return np.unique(np.round(np.logspace(np.log10(t0), np.log10(t1), n)).astype(np.int64))


# ------------------------------------------------------------------ tasks
def build_tasks():
    """Deterministic task list.  Each task: arm, nu, theta, seed, T_obs, cap.

    Run counts per (nu, theta): 100 for nu >= 5e-5, 80 for nu = 2e-5, 60 for
    nu = 5e-6 (the expensive ones).  Neutral anchors get +50%.
    Frame caps chosen from the pilot: cohort extinction ~ (0.3-1.5) x 1e6
    frames at the smallest nu; cap 8e6 keeps censoring below a few percent.
    """
    tasks = []
    seed = 13_700_000
    # loaded runs / neutral anchor runs per (nu) cell.  Neutral anchors get
    # +50% (every loaded ratio is to the arm's own theta=0 anchor).
    n_load = {5e-6: 64, 2e-5: 80, 5e-5: 100, 1.5e-4: 100, 5e-4: 100, 1.5e-3: 100}
    n_neut = {5e-6: 96, 2e-5: 120, 5e-5: 150, 1.5e-4: 150, 5e-4: 150, 1.5e-3: 150}
    # frame caps sized from the round-9 pilot (13,738,000-13,738,035): neutral
    # cohort extinction ~ 2N generations ~ 2.5e7 site-updates x ... exceeds
    # 6e6 frames at nu <= 2e-5, so those neutral arms get long caps; the
    # primary (uncensored) statistics are first-passage clocks, not medians.
    cap_load = {5e-6: 12_000_000, 2e-5: 8_000_000, 5e-5: 6_000_000,
                1.5e-4: 5_000_000, 5e-4: 3_000_000, 1.5e-3: 2_000_000}
    cap_neut = {5e-6: 25_000_000, 2e-5: 16_000_000, 5e-5: 8_000_000,
                1.5e-4: 5_000_000, 5e-4: 3_000_000, 1.5e-3: 2_000_000}
    cap = None
    for nu in NU_GRID:
        for th in THETAS:
            n = n_neut[nu] if th == 0.0 else n_load[nu]
            T = cap_neut[nu] if th == 0.0 else cap_load[nu]
            for k in range(n):
                tasks.append(dict(arm="NS", nu=nu, theta=th, seed=seed,
                                  T_obs=T, T_burn=T_BURN))
                seed += 1
    # closed, burnt-in (nu -> 0 endpoint):  theta 0/0.4/0.8, 200/120/120 runs
    for th, n in [(0.0, 200), (0.4, 120), (0.8, 120)]:
        for k in range(n):
            tasks.append(dict(arm="CB", nu=0.0, theta=th, seed=seed,
                              T_obs=12_000_000, T_burn=T_BURN))
            seed += 1
    # closed, fresh field (replication of the deposited reference)
    for th, n in [(0.0, 150), (0.4, 80), (0.8, 80)]:
        for k in range(n):
            tasks.append(dict(arm="CF", nu=0.0, theta=th, seed=seed,
                              T_obs=12_000_000, T_burn=0))
            seed += 1
    for i, t in enumerate(tasks):
        t["task_id"] = i
    return tasks


# ------------------------------------------------------------------ single run
def run_task(t):
    import evolution_open_ns as en
    nu = t["nu"]
    theta = t["theta"]
    mud = theta * SD
    seed = t["seed"]
    T_obs = t["T_obs"]
    t0 = time.time()
    if t["arm"] == "NS":
        grid = geom_grid(300, T_obs, 24)
        r = en.evolve_open(0, L_long=L_LONG, L_short=L_SHORT, T_burn=t["T_burn"],
                           T_obs=T_obs, sd=SD, mud=mud, nu=nu, seed=seed,
                           samp_frames=grid, ns0_thr=NS0_THR, T_stop_after=1)
        ts = r["ts"]
        wall = time.time() - t0
        out = dict(t)
        out.update(dict(
            wall=round(wall, 1), t_abs=r["t_abs"], abs_type=r["abs_type"],
            frames_run=int(ts[-1, 0]) if ts.shape[0] else 0,
            n_nuc=r["n_nucleated"], n_capped=r["n_capped"],
            ns0_end=r["ns0_end"], coh_live_end=r["coh_live_end"],
            live_end=r["live_end"], K0=int(r["ts"][0, 2]) if ts.shape[0] else -1,
            ns0_thr=NS0_THR.tolist(), ns0_fp=r["ns0_fp"].tolist(),
            ns0_coh=r["ns0_coh"].tolist(),
            # time series (24/decade grid): frame, K, ns0, mx, sq, live, hetbond
            # frac, mx2, b12/bdiff, coh_live, largest_is_cohort
            ts_frame=ts[:, 0].astype(np.int64).tolist(),
            ts_K=ts[:, 1].astype(np.int64).tolist(),
            ts_ns0=ts[:, 2].astype(np.int64).tolist(),
            ts_mx=ts[:, 3].astype(np.int64).tolist(),
            ts_mx2=ts[:, 10].astype(np.int64).tolist(),
            ts_sq=[round(float(x), 1) for x in ts[:, 4]],
            ts_live=ts[:, 5].astype(np.int64).tolist(),
            ts_hetfrac=[round(float(x), 5) for x in (ts[:, 8] / np.maximum(ts[:, 9], 1))],
            ts_b12frac=[round(float(x), 5) for x in (ts[:, 11] / np.maximum(ts[:, 8], 1))],
            ts_coh=ts[:, 12].astype(np.int64).tolist(),
            ts_maxiscoh=ts[:, 13].astype(np.int8).tolist(),
        ))
    else:  # closed two-clone half-domain
        grid = geom_grid(1000, T_obs, 12)
        r = en.evolve_open(2, L_long=L_LONG, L_short=L_SHORT, T_burn=t["T_burn"],
                           T_obs=T_obs, sd=SD, mud=mud, nu=0.0, seed=seed,
                           samp_frames=grid)
        ts = r["ts"]
        wall = time.time() - t0
        out = dict(t)
        out.update(dict(
            wall=round(wall, 1), t_abs=r["t_abs"], abs_type=r["abs_type"],
            winner=r["scal7"], frames_run=int(ts[-1, 0]) if ts.shape[0] else 0,
            live_end=r["live_end"],
            field_meanfit_at_mark=r["field_meanfit_at_mark"],
            field_sdfit_at_mark=r["field_sdfit_at_mark"],
            ts_frame=ts[:, 0].astype(np.int64).tolist(),
            ts_c1=ts[:, 1].astype(np.int64).tolist(),
            ts_c0=ts[:, 2].astype(np.int64).tolist(),
        ))
    return out


def run_and_write(t):
    out = run_task(t)
    shard = os.path.join(OUTDIR, "shard_%03d.jsonl" % (t["task_id"] % 128))
    line = json.dumps(out, separators=(",", ":")) + "\n"
    with open(shard, "a") as fh:
        fh.write(line)
    return dict(task_id=t["task_id"], arm=t["arm"], nu=t["nu"], theta=t["theta"],
                seed=t["seed"], wall=out["wall"], t_abs=out["t_abs"],
                abs_type=out["abs_type"], frames_run=out["frames_run"])


def done_ids():
    ids = set()
    if not os.path.isdir(OUTDIR):
        return ids
    for fn in os.listdir(OUTDIR):
        if not fn.startswith("shard_"):
            continue
        with open(os.path.join(OUTDIR, fn)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["task_id"])
                except Exception:
                    pass
    return ids


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--tasks", default=os.path.join(WS, "handoff", "tasks.json"))
    ap.add_argument("--only", default="", help="optional arm filter, e.g. NS or CB,CF")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.tasks), exist_ok=True)

    if args.cmd == "plan":
        tasks = build_tasks()
        json.dump(tasks, open(args.tasks, "w"))
        from collections import Counter
        c = Counter((t["arm"], t["nu"], t["theta"]) for t in tasks)
        for k in sorted(c):
            print(k, c[k])
        print("total tasks", len(tasks), "seeds", tasks[0]["seed"], "-", tasks[-1]["seed"])
        sys.exit(0)

    tasks = json.load(open(args.tasks))
    if args.only:
        keep = set(args.only.split(","))
        tasks = [t for t in tasks if t["arm"] in keep]
    have = done_ids()
    todo = [t for t in tasks if t["task_id"] not in have]
    # schedule the most expensive (lowest nu, closed arms) first for load balance
    def cost_key(t):
        if t["arm"] != "NS":
            return -2.0
        return t["nu"]
    todo.sort(key=cost_key)
    print("tasks total %d, already done %d, todo %d" % (len(tasks), len(have), len(todo)), flush=True)
    t0 = time.time()
    n_done = 0
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_and_write, t): t["task_id"] for t in todo}
        for fut in cf.as_completed(futs):
            n_done += 1
            try:
                r = fut.result()
            except Exception as e:
                r = dict(task_id=futs[fut], error=repr(e))
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                print(json.dumps(dict(done=n_done, of=len(todo), elapsed_s=round(el),
                                      last=r)), flush=True)
    print(json.dumps(dict(finished=True, elapsed_s=round(time.time() - t0))), flush=True)
