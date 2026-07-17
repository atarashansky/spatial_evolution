"""Ensemble driver for two-clone strip simulations (evolution_strip.evolve_strip).

run_ensemble(param_list, n_workers=60, summary_only=True) -> pandas.DataFrame

Each element of param_list is a dict of evolve_strip kwargs (must include a
unique `seed`). Runs execute across a ProcessPoolExecutor, one evolve_strip
call per task; numba compiles once per worker process (~2-3 s on first call).

Summary row per run: all params + t_fix, winner, n_ben, final interface width
(h_t.std over rows at the last available sample), end-of-run mean fitness per
side, wall_time_s, and an `error` column ('' on success). Worker exceptions
are captured into the row instead of killing the pool.

summary_only=False additionally gzip-pickles each full result dict to
./ens_data/run_<seed>.pkl.gz.
"""
import gzip
import os
import pickle
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_ENS_DIR = "./ens_data"


def _summarize(params, res, wall):
    """Build a flat summary row from an evolve_strip result dict."""
    row = dict(params)
    row["t_fix"] = res["t_fix"]
    row["winner"] = res["winner"]
    row["n_ben"] = int(res["ben"]["frame"].size)
    h = res["h_t"]
    if h.shape[0] > 0:
        row["final_width"] = float(h[-1].astype(np.float64).std())
        row["n_samples"] = int(h.shape[0])
        row["last_samp_frame"] = int(res["samp_frame"][-1])
    else:
        row["final_width"] = np.nan
        row["n_samples"] = 0
        row["last_samp_frame"] = -1
    fA, fB = res["fitA"], res["fitB"]
    row["fitA_end"] = float(fA[-1]) if fA.size else np.nan
    row["fitB_end"] = float(fB[-1]) if fB.size else np.nan
    row["wall_time_s"] = wall
    row["error"] = ""
    return row


def _worker(params, summary_only):
    """Top-level worker: run one strip, return (row, seed). Never raises."""
    try:
        from evolution_strip import evolve_strip  # import inside worker
        t0 = time.perf_counter()
        res = evolve_strip(**params)
        wall = time.perf_counter() - t0
        row = _summarize(params, res, wall)
        if not summary_only:
            os.makedirs(_ENS_DIR, exist_ok=True)
            path = os.path.join(_ENS_DIR, f"run_{params['seed']}.pkl.gz")
            with gzip.open(path, "wb", compresslevel=2) as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            row["pickle_path"] = path
        return row
    except Exception:
        row = dict(params)
        row["t_fix"] = -2
        row["winner"] = -2
        row["error"] = traceback.format_exc(limit=8)
        row["wall_time_s"] = np.nan
        return row


def run_ensemble(param_list, n_workers=60, summary_only=True, progress=True):
    """Run evolve_strip over param_list in parallel; return summary DataFrame.

    param_list : list of dicts of evolve_strip kwargs, each with a unique seed.
    n_workers  : process count (numba compiles once per process).
    summary_only : if False, also pickle full result dicts to ./ens_data/.
    """
    seeds = [p.get("seed") for p in param_list]
    if len(set(seeds)) != len(seeds) or any(s is None for s in seeds):
        raise ValueError("every param dict must carry a unique 'seed'")

    rows = []
    t0 = time.perf_counter()
    n = len(param_list)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, p, summary_only): p["seed"]
                for p in param_list}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if progress and (done % max(1, n // 10) == 0 or done == n):
                print(f"[ensemble] {done}/{n} done, "
                      f"{time.perf_counter() - t0:.1f}s elapsed", flush=True)
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    df.attrs["total_wall_s"] = time.perf_counter() - t0
    return df
