"""Parallel runner for the click-onset sweep.

Usage: python run_onset_sweep.py <tasks.json> <out_prefix> [nproc]

Reads a JSON list of task dicts, runs them with a fork pool, and writes:
  <out_prefix>_records.pkl   list of per-run summary dicts (arrays included)
  <out_prefix>_progress.log   progress lines
Longest-first ordering (by expected fixation time) is applied by the task
builder; here we just imap_unordered with chunksize 1.
"""
import os
import sys
import json
import time
import pickle
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onset_driver  # noqa: E402


def main():
    tasks_path, prefix = sys.argv[1], sys.argv[2]
    nproc = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    tasks = json.load(open(tasks_path))
    log = open(f"{prefix}_progress.log", "w")
    print(f"launching {len(tasks)} runs on {nproc} workers", file=log, flush=True)
    t0 = time.time()
    results = []
    with get_context("fork").Pool(nproc) as pool:
        for k, r in enumerate(pool.imap_unordered(onset_driver.one_run, tasks,
                                                 chunksize=1)):
            results.append(r)
            if (k + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{len(tasks)} done  {el/60:.2f} min", file=log,
                      flush=True)
                # periodic checkpoint so partial progress survives
                if (k + 1) % 500 == 0:
                    with open(f"{prefix}_records.pkl", "wb") as fh:
                        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    el = time.time() - t0
    print(f"ALL DONE {len(results)} runs in {el/60:.2f} min", file=log, flush=True)
    with open(f"{prefix}_records.pkl", "wb") as fh:
        pickle.dump(results, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.close()


if __name__ == "__main__":
    main()
