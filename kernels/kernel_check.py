"""Bit-exact replication check of the deposited protocol with the staged
kernel: rerun 4 seeds from ridge_profile_runs.parquet (2 neutral anchors,
2 loaded rungs) at 192x32 and require identical t_fix."""
import os, sys, json, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir(_HERE)
for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
          "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
from evolution_strip import evolve_strip

BASE = dict(L_long=192, L_short=32, numgen=20_000_000, dt=2400.0, sb=0.1,
            mub=0.0, sampling=10_000, avg=1.16e-5, periodic_y=True,
            moore=True, com_stride=0, prof_halfwidth=64, ben_cap=1024)
CASES = [
    dict(seed=12900000, sd=0.0, mud=0.0, expect=1129978),
    dict(seed=12900001, sd=0.0, mud=0.0, expect=1325918),
    dict(seed=12921119, sd=0.02, mud=0.01, expect=62462),
    dict(seed=12912152, sd=0.0025, mud=0.00175, expect=65070),
]
out = []
for c in CASES:
    p = dict(BASE); p.update(sd=c["sd"], mud=c["mud"], seed=c["seed"])
    t0 = time.perf_counter()
    r = evolve_strip(**p)
    wall = time.perf_counter() - t0
    ok = int(r["t_fix"]) == c["expect"]
    out.append(dict(seed=c["seed"], sd=c["sd"], mud=c["mud"], t_fix=int(r["t_fix"]),
                    expect=c["expect"], match=bool(ok), wall_s=round(wall, 2)))
    print(json.dumps(out[-1]), flush=True)
json.dump(dict(cases=out, all_match=all(x["match"] for x in out)),
          open("kernel_check.json", "w"), indent=1)
print("ALL_MATCH", all(x["match"] for x in out))
