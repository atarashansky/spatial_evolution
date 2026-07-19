"""Finite-size / boundary-condition sweep driver for the crossover ridge.

Question (T8 cluster 1 + walled-boundary cluster): does the crossover load
theta* = U_d*/s_d at s_d = 0.01 drift with lattice size L, or plateau?

Design (pre-registered in fss_design.json, written BEFORE any production run):
  * L_long ladder at fixed L_short = 32 (the pool's reference short axis):
        96, 192 (reference geometry, independent replication of the pool),
        384, 768, 1536 (reduced ladder + replication, cost-capped)
  * L_short axis at fixed L_long = 192: 16 and 64 (reference is 32)
  * walled (non-periodic) transverse boundary arm at the reference 192x32
  * every geometry carries its OWN neutral (mud = 0) anchor arm
  * protocol identical to the deposited s_d = 0.01 pool: Moore, dt = 2400,
    avg = 1.16e-5, sb = 0.1, mub = 0, numgen (cap) = 2e7, sampling = 1e4,
    theta = mud / sd on the 14-rung deposited ladder
    {0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2,2.0,5.0,10.0}

Seeds: one contiguous fresh block per cell (never reused between cells);
task list, shard files and the done-set make the campaign resume-safe.
Workers are single-threaded by construction (spawn context + thread pinning
in the pool initializer).

Usage:
    python fss_driver.py plan                # write handoff/fss_tasks.json
    python fss_driver.py run --workers 56   # execute pending tasks
    python fss_driver.py collect             # merge shards -> fss_runs.parquet
"""
import os
import sys
import json
import time
import glob
import argparse
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --------------------------------------------------------------- protocol
BASE = dict(numgen=20_000_000, dt=2400.0, sb=0.1, mub=0.0, sampling=10_000,
            avg=1.16e-5, moore=True, com_stride=0, prof_halfwidth=64,
            ben_cap=1024)
SD = 0.01
THETAS_FULL = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2,
               2.0, 5.0, 10.0]
THETAS_1536 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]

# name, L_long, L_short, periodic_y, thetas, n_per_rung, n_anchor, seed_base, numgen
# numgen (frame cap) is a censoring bound only; it is raised on the two
# largest lattices, where the pool's 2e7 cap would censor neutral anchors,
# and kept at the pool's 2e7 everywhere else (pre-registered deviation).
CELLS = [
    ("G96",   96,   32, True,  THETAS_FULL, 200, 400, 14_000_000, 20_000_000),
    ("G192",  192,  32, True,  THETAS_FULL, 200, 400, 14_004_000, 20_000_000),
    ("G384",  384,  32, True,  THETAS_FULL, 200, 400, 14_008_000, 20_000_000),
    ("G768",  768,  32, True,  THETAS_FULL, 150, 250, 14_012_000, 40_000_000),
    ("G1536", 1536, 32, True,  THETAS_1536,  48, 100, 14_016_000, 100_000_000),
    ("S16",   192,  16, True,  THETAS_FULL, 200, 400, 14_020_000, 20_000_000),
    ("S64",   192,  64, True,  THETAS_FULL, 200, 400, 14_024_000, 20_000_000),
    ("W32",   192,  32, False, THETAS_FULL, 200, 400, 14_028_000, 20_000_000),
]
ANCHOR_OFFSET = 3_000          # anchors: base + ANCHOR_OFFSET + i
RUNG_STRIDE = 200              # loaded rung r: base + r*RUNG_STRIDE + i
SEED_FLOOR = 14_000_000
SEED_CEILING = 14_031_999      # production block; 14,032,000-14,049,999 reserved spillover
                               # (never allocated here), 14,090,000+ scratch (never deposited)


# --------------------------------------------------------------- planning
def build_tasks():
    tasks = []
    for (name, L, S, per, ths, nr, na, base, cap) in CELLS:
        assert nr <= RUNG_STRIDE, name
        assert len(ths) * RUNG_STRIDE <= ANCHOR_OFFSET, name
        assert base + ANCHOR_OFFSET + na - 1 < base + 4000, name
        # loaded rungs
        for r, th in enumerate(ths):
            for i in range(nr):
                seed = base + r * RUNG_STRIDE + i
                tasks.append(dict(cell=name, L_long=L, L_short=S,
                                  periodic_y=per, sd=SD, mud=SD * th,
                                  theta=th, kind="loaded", seed=seed,
                                  numgen=cap))
        # own neutral anchor
        for i in range(na):
            seed = base + ANCHOR_OFFSET + i
            tasks.append(dict(cell=name, L_long=L, L_short=S,
                              periodic_y=per, sd=0.0, mud=0.0,
                              theta=float("nan"), kind="neutral", seed=seed,
                              numgen=cap))
    seeds = [t["seed"] for t in tasks]
    assert len(seeds) == len(set(seeds)), "seed collision inside campaign"
    assert max(seeds) <= SEED_CEILING and min(seeds) >= SEED_FLOOR
    for j, t in enumerate(tasks):
        t["task_id"] = j
    return tasks


def cost_order(tasks):
    """Longest expected runs first (neutral, big lattices, small theta)."""
    def key(t):
        area = t["L_long"] * t["L_short"]
        load = 0.0 if t["kind"] == "neutral" else t["theta"]
        return (-area, load)
    return sorted(tasks, key=key)


# --------------------------------------------------------------- worker
def _pin_threads():
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _worker(task):
    import numpy as np
    try:
        from evolution_strip import evolve_strip
        p = dict(BASE)
        p.update(L_long=int(task["L_long"]), L_short=int(task["L_short"]),
                 periodic_y=bool(task["periodic_y"]), sd=float(task["sd"]),
                 mud=float(task["mud"]), seed=int(task["seed"]),
                 numgen=int(task.get("numgen", BASE["numgen"])))
        t0 = time.perf_counter()
        r = evolve_strip(**p)
        wall = time.perf_counter() - t0
        h = r["h_t"]
        row = dict(task)
        row.update(
            t_fix=int(r["t_fix"]), winner=int(r["winner"]),
            censored=int(r["t_fix"] < 0),
            final_width=float(h[-1].astype(np.float64).std()) if h.shape[0] else float("nan"),
            n_samples=int(h.shape[0]),
            fitA_end=float(r["fitA"][-1]) if r["fitA"].size else float("nan"),
            fitB_end=float(r["fitB"][-1]) if r["fitB"].size else float("nan"),
            numgen=int(p["numgen"]), dt=BASE["dt"], sb=BASE["sb"], mub=BASE["mub"],
            sampling=BASE["sampling"], avg=BASE["avg"], moore=BASE["moore"],
            p_die=float(1.0 - np.exp(-BASE["avg"] * BASE["dt"])),
            wall_time_s=round(wall, 3), error="")
        return row
    except Exception:
        row = dict(task)
        row.update(t_fix=-2, winner=-2, censored=-2, error=traceback.format_exc(limit=6),
                   wall_time_s=float("nan"))
        return row


# --------------------------------------------------------------- run
def load_done(shard_dir):
    done = set()
    for f in glob.glob(os.path.join(shard_dir, "shard_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(int(json.loads(line)["task_id"]))
                except Exception:
                    pass
    return done


def run(workers, shard_dir, tasks_path, cells=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    tasks = json.load(open(tasks_path))
    os.makedirs(shard_dir, exist_ok=True)
    done = load_done(shard_dir)
    pending = [t for t in tasks if t["task_id"] not in done]
    if cells:
        pending = [t for t in pending if t["cell"] in cells]
    pending = cost_order(pending)
    if os.environ.get("FSS_REVERSE_ORDER") == "1":
        pending = pending[::-1]
    print(f"[fss] tasks {len(tasks)}; done {len(done)}; pending {len(pending)}; "
          f"workers {workers}; cells {cells if cells else 'all'}", flush=True)
    if not pending:
        return
    tag = os.environ.get("FSS_SHARD_TAG", str(os.getpid()))
    shard = os.path.join(shard_dir, f"shard_{tag}.jsonl")
    t0 = time.perf_counter()
    n = len(pending)
    ctx = mp.get_context("spawn")
    ndone = 0
    with open(shard, "a", buffering=1) as fh, \
            ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                initializer=_pin_threads) as ex:
        futs = [ex.submit(_worker, t) for t in pending]
        for fut in as_completed(futs):
            row = fut.result()
            fh.write(json.dumps(row) + "\n")
            ndone += 1
            if ndone % 50 == 0 or ndone == n:
                el = time.perf_counter() - t0
                rate = ndone / el
                eta = (n - ndone) / rate if rate > 0 else float("inf")
                print(f"[fss] {ndone}/{n} done  {el:.0f}s elapsed  "
                      f"eta {eta:.0f}s", flush=True)
    print(f"[fss] wave complete: {ndone} runs in {time.perf_counter()-t0:.0f}s",
          flush=True)


def collect(shard_dir, out_parquet):
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.sort_values("task_id").drop_duplicates("task_id", keep="first")
    df = df.reset_index(drop=True)
    df["source"] = "fss_finite_size_track"
    df.to_parquet(out_parquet)
    print(f"[fss] collected {len(df)} rows -> {out_parquet}; "
          f"errors {(df['error'] != '').sum()}, censored {(df['t_fix'] == -1).sum()}",
          flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "run", "collect"])
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--shards", default="fss_shards")
    ap.add_argument("--tasks", default="fss_tasks.json")
    ap.add_argument("--out", default="fss_runs.parquet")
    ap.add_argument("--cells", default="",
                    help="comma-separated cell names to run (default: all)")
    a = ap.parse_args()
    cell_filter = [c for c in a.cells.split(",") if c] or None
    if a.mode == "plan":
        tasks = build_tasks()
        json.dump(tasks, open(a.tasks, "w"))
        seeds = [t["seed"] for t in tasks]
        print(f"[fss] planned {len(tasks)} runs; seeds {min(seeds)}..{max(seeds)} "
              f"({len(set(seeds))} unique)", flush=True)
        by = {}
        for t in tasks:
            by[t["cell"]] = by.get(t["cell"], 0) + 1
        print(json.dumps(by), flush=True)
    elif a.mode == "run":
        run(a.workers, a.shards, a.tasks, cells=cell_filter)
    else:
        collect(a.shards, a.out)
