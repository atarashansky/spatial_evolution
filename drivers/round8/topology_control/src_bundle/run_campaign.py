"""Standalone driver: round-8 topology-control T_fix + census campaign.

Run from work/ as:  python run_campaign.py {tfix|census|both} > out/campaign.log 2>&1
Resumes from out/tfix_checkpoint.parquet / out/census_checkpoint.parquet
(rows with error=='' are kept; errored rows are re-run).
"""
import os
import sys
import json
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "kernels"))
os.environ["PYTHONPATH"] = os.path.join(HERE, "kernels")

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import topo_campaign as tc  # noqa: E402

SD = 0.01
THETAS = [float(x) for x in np.round(np.geomspace(0.05, 1.2, 9), 4)]
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
PLAN = json.load(open(os.path.join(OUT, "seed_plan.json")))
N_WORKERS = int(os.environ.get("N_WORKERS", "60"))


def tfix_tasks():
    PT = dict(tc.PROD_TFIX)
    ts = []
    for st in ["hex", "rand"]:
        s0, n = PLAN[f"{st}_anchor"]["start"], PLAN[f"{st}_anchor"]["n"]
        ts += [dict(kind="tfix", stencil=st, seed=s0 + i, sd=0.0, mud=0.0,
                    arm=f"{st}_anchor", theta=np.nan, **PT) for i in range(n)]
        for k, th in enumerate(THETAS):
            key = f"{st}_rung{k}_theta{th}"
            s0, n = PLAN[key]["start"], PLAN[key]["n"]
            ts += [dict(kind="tfix", stencil=st, seed=s0 + i, sd=SD,
                        mud=th * SD, arm=f"{st}_ladder", theta=th, **PT)
                   for i in range(n)]
    return ts


def census_tasks():
    PC = dict(tc.PROD_CENSUS)
    cths = [0.05, 0.2, 0.5, 1.0]
    ts = []
    for st in ["hex", "rand"]:
        s0, n = PLAN[f"{st}_census"]["start"], PLAN[f"{st}_census"]["n"]
        per = n // len(cths)
        i = 0
        for th in cths:
            for _ in range(per):
                ts.append(dict(kind="census", stencil=st, seed=s0 + i,
                               sd=SD, mud=th * SD, arm=f"{st}_census",
                               theta=th, **PC))
                i += 1
    # matched-protocol Moore control (same kernel path via topo_strip)
    s0, n = PLAN["moore_census_ctrl"]["start"], PLAN["moore_census_ctrl"]["n"]
    per = n // len(cths)
    i = 0
    for th in cths:
        for _ in range(per):
            ts.append(dict(kind="census", stencil="moore", seed=s0 + i,
                           sd=SD, mud=th * SD, arm="moore_census_ctrl",
                           theta=th, **PC))
            i += 1
    return ts


def load_done(path):
    if not os.path.exists(path):
        return None, set()
    d = pd.read_parquet(path)
    d = d[d.error == ""].copy()
    keys = set(zip(d.kind, d.stencil, d.seed))
    return d, keys


def run(tag, tasks, ckpt):
    done_df, done_keys = load_done(ckpt)
    todo = [t for t in tasks
            if (t["kind"], t["stencil"], t["seed"]) not in done_keys]
    print(f"[{tag}] {len(tasks)} tasks, {len(done_keys)} done, "
          f"{len(todo)} remaining, workers={N_WORKERS}", flush=True)
    random.Random(7).shuffle(todo)
    t0 = time.perf_counter()
    df = tc.run_ensemble(todo, n_workers=N_WORKERS, tag=tag, checkpoint=ckpt,
                         every=100,
                         done_rows=None if done_df is None
                         else done_df.to_dict("records"))
    out = os.path.join(OUT, f"{tag}_runs_raw.parquet")
    df.to_parquet(out)
    print(f"[{tag}] DONE: {len(df)} rows -> {out}, "
          f"{time.perf_counter()-t0:.0f}s wall, errors={(df.error!='').sum()},"
          f" censored={(df.t_fix<0).sum()}", flush=True)
    if (df.error != "").any():
        print(df.error[df.error != ""].iloc[0], flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("tfix", "both"):
        run("tfix", tfix_tasks(), os.path.join(OUT, "tfix_checkpoint.parquet"))
    if which in ("census", "both"):
        run("census", census_tasks(),
            os.path.join(OUT, "census_checkpoint.parquet"))
