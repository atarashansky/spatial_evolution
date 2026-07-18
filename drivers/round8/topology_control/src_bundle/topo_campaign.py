"""Round-8 topology-control campaign: worker functions + parallel farm-out.

Mirrors drivers/ensemble_driver.py (one run per task, ProcessPoolExecutor,
per-run row with all params + summary + wall time + captured error), but the
task carries a `kind` selecting the kernel entry point:

  kind='prod'      -> evolution_strip.evolve_strip (production, moore=/periodic_y=)
  kind='tfix'      -> topo_strip.evolve_topo(stencil=...)  T_fix run
  kind='census'    -> topo_strip.census_topo(stencil=...) contact census run
"""
import os
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# production protocol constants (identical to the deposited ridge ladders:
# ridge_profile_runs.parquet / neighbourhood_ridge.parquet)
PROD_TFIX = dict(L_long=192, L_short=32, numgen=20_000_000, dt=2400.0,
                 sb=0.1, mub=0.0, sampling=10_000, avg=1.16e-5,
                 com_stride=0, prof_halfwidth=64, ben_cap=1024)
# census protocol constants (round-6/7 census: n0_functional_form, vn_census)
PROD_CENSUS = dict(L_long=192, L_short=32, numgen=1_200_000, dt=2400.0,
                   sampling=2000, avg=1.16e-5, iface_w=3)


def worker(task):
    """task: dict with 'kind' plus kernel kwargs (incl. unique 'seed').
    Returns a flat row dict; never raises."""
    task = dict(task)
    kind = task.pop("kind")
    meta = {k: task.pop(k) for k in ("arm", "theta") if k in task}
    t0 = time.perf_counter()
    try:
        if kind == "prod":
            from evolution_strip import evolve_strip
            res = evolve_strip(**task)
            row = dict(task)
            row["t_fix"] = int(res["t_fix"])
            row["winner"] = int(res["winner"])
            h = res["h_t"]
            row["final_width"] = float(h[-1].astype(np.float64).std()) if h.shape[0] else np.nan
            row["n_samples"] = int(h.shape[0])
            row["fitA_end"] = float(res["fitA"][-1]) if res["fitA"].size else np.nan
            row["fitB_end"] = float(res["fitB"][-1]) if res["fitB"].size else np.nan
        elif kind == "tfix":
            from topo_strip import evolve_topo
            res = evolve_topo(**task)
            row = dict(task)
            row.update({k: res[k] for k in ("t_fix", "winner", "n_ben",
                                             "final_width", "n_samples",
                                             "last_samp_frame", "fitA_end",
                                             "fitB_end", "M", "p_die")})
        elif kind == "census":
            from topo_strip import census_topo, census_row
            res = census_topo(**task)
            row = census_row(res, task)
            row["M"] = res["M"]
            row["p_die"] = res["p_die"]
        else:
            raise ValueError(kind)
        row.update(meta)
        row["kind"] = kind
        row["wall_time_s"] = time.perf_counter() - t0
        row["error"] = ""
        return row
    except Exception:
        row = dict(task)
        row.update(meta)
        row["kind"] = kind
        row["t_fix"] = -2
        row["winner"] = -2
        row["wall_time_s"] = time.perf_counter() - t0
        row["error"] = traceback.format_exc(limit=8)
        return row


def _init_worker():
    """Pin BLAS/OpenMP runtimes to one thread per worker (the njit kernels are
    serial; the oversubscription source is threaded numpy/BLAS inside 60
    concurrent processes on a shared host). NUMBA_NUM_THREADS is deliberately
    NOT touched: setting it after numba's pool has launched raises."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def run_ensemble(tasks, n_workers=60, tag="", checkpoint=None, every=200,
                 done_rows=None):
    keys = [(t.get("kind"), t.get("stencil"), t.get("seed")) for t in tasks]
    if len(set(keys)) != len(keys) or any(k[2] is None for k in keys):
        raise ValueError("every task must carry a unique (kind,stencil,seed)")
    rows = [] if done_rows is None else [dict(r) for r in done_rows]
    n = len(tasks) + len(rows)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker) as ex:
        futs = [ex.submit(worker, t) for t in tasks]
        done = len(rows)
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % max(1, n // 20) == 0 or done == n:
                el = time.perf_counter() - t0
                print(f"[{tag}] {done}/{n} done, {el:.0f}s elapsed",
                      flush=True)
            if checkpoint and done % every == 0:
                pd.DataFrame(rows).to_parquet(checkpoint)
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.attrs["total_wall_s"] = time.perf_counter() - t0
    if checkpoint:
        df.to_parquet(checkpoint)
    return df
