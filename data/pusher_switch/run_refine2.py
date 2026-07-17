"""Fine b* transition sweep: 7 b-values inside/around the bracket, 500k-frame horizon,
same 4-column patch protocol as calib/refine."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def one(args):
    b, seed = args
    from evolution_strip_mixed import evolve_strip_mixed
    r = evolve_strip_mixed(L_long=192, L_short=64, numgen=500_000,
                           sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=b,
                           split=188, pusher_cols=(189, 192),
                           sampling=100_000, com_stride=0, stop_fix=False,
                           stop_push_frac=0.0, seed=seed)
    M = r["params"]["M"]
    npf = r["n_push_final"]
    return dict(kind="refine2", b=b, seed=seed, surv=int(npf > 0),
                share=npf / M)

if __name__ == "__main__":
    B = [0.0405, 0.041, 0.0415, 0.042, 0.0425, 0.043, 0.0435]
    jobs = [(b, 9000 + s) for b in B for s in range(120)]
    out = []
    with ProcessPoolExecutor(max_workers=60) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=2)):
            out.append(r)
            if (i + 1) % 120 == 0:
                print(f"{i+1}/{len(jobs)}", flush=True)
    with open("refine2_results.json", "w") as fh:
        json.dump(out, fh)
    print("DONE", len(out))
