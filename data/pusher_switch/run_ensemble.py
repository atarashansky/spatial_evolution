"""Pusher-switch mixed-tissue ensemble v2 (three-regime protocol).

Jobs:
  refine : b* bracket sharpening (patch protocol, neutral fitness)
  main   : mixed tissue from all-dB init, switches via mu_c; fixed window,
           plateau-share observable, rel/abs clock couplings.
Output: ensemble_results.json
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import numpy as np
from concurrent.futures import ProcessPoolExecutor

P_DIE = 1.0 - np.exp(-1.16e-5 * 2400.0)
WIN = 750_000
LOADS = {"neutral": (0.0, 0.0), "L1e-3": (0.1, 0.01), "L4e-3": (0.1, 0.04)}


def refine(args):
    b, seed = args
    from evolution_strip_mixed import evolve_strip_mixed
    r = evolve_strip_mixed(L_long=192, L_short=64, numgen=500_000,
                           sd=0.0, mud=0.0, mub=0.0, mu_c=0.0, b_rate=b,
                           split=188, pusher_cols=(189, 192),
                           sampling=100_000, com_stride=1000,
                           stop_push_frac=0.999, stop_fix=False, seed=seed)
    pc = r["push_com"]
    surv = 1 if (r["t_stop_push"] > 0 or (len(pc) and pc[-1] > 0)) else 0
    share = float(pc[-1]) / r["params"]["M"] if len(pc) else \
        (1.0 if r["t_stop_push"] > 0 else 0.0)
    if r["t_stop_push"] > 0:
        share = 1.0
    return dict(kind="refine", b=b, seed=seed, surv=surv, share=share,
                t_ext=int(np.argmax(pc == 0) * 1000) if (surv == 0 and
                len(pc)) else -1)


def main_run(args):
    mu_c, b, load_name, clock, seed = args
    from evolution_strip_mixed import evolve_strip_mixed
    mud, sd = LOADS[load_name]
    r = evolve_strip_mixed(L_long=192, L_short=64, numgen=WIN,
                           sd=sd, mud=mud, mub=0.0, mu_c=mu_c, b_rate=b,
                           sampling=50_000, com_stride=2500,
                           stop_push_frac=0.999, stop_fix=False,
                           abs_clock=(clock == "abs"), seed=seed)
    M = r["params"]["M"]
    stride = 2500
    pc = r["push_com"].astype(np.float64)
    blow = r["t_stop_push"] > 0
    # establishment: 1% of tissue
    est_mask = pc >= 0.01 * M
    t_est = int((np.argmax(est_mask)) * stride) if est_mask.any() else \
        (int(r["t_stop_push"]) if blow else -1)
    if blow and not est_mask.any():
        t_est = int(r["t_stop_push"])
    # half-tissue crossing
    half_mask = pc >= 0.5 * M
    t_half = int(np.argmax(half_mask) * stride) if half_mask.any() else \
        (int(r["t_stop_push"]) if blow else -1)
    # plateau share: mean over last 20% of window (1.0 if blow-through)
    if blow:
        share = 1.0
    elif len(pc):
        share = float(pc[int(0.8 * len(pc)):].mean() / M)
    else:
        share = 0.0
    # expansion velocity: pusher-area slope between 5% and 40% of M
    v_exp = np.nan
    seg = np.where((pc >= 0.05 * M) & (pc <= 0.40 * M))[0]
    if len(seg) >= 3 and (seg[-1] - seg[0]) > 0:
        v_exp = float(np.polyfit(seg * stride, pc[seg] / 64.0, 1)[0])
    swi = r["swi"]
    t_first_swi = int(swi["frame"][0]) if len(swi["frame"]) else -1
    return dict(kind="main", mu_c=mu_c, b=b, load=load_name, clock=clock,
                seed=seed, M=M,
                t_fix=int(r["t_fix"]), winner=int(r["winner"]),
                blow=int(blow), t_blow=int(r["t_stop_push"]),
                t_first_switch=t_first_swi,
                n_switch_total=int(r["n_switch_total"]),
                established=int(t_est >= 0), t_establish=t_est,
                t_half=t_half, plateau_share=share, v_expand=v_exp,
                push_frac_end=float(r["n_push_final"] / M),
                fitP_last=float(r["fitP"][-1]) if len(r["fitP"]) else np.nan,
                fitA_last=float(r["fitA"][-1]) if len(r["fitA"]) else np.nan,
                fitB_last=float(r["fitB"][-1]) if len(r["fitB"]) else np.nan)


if __name__ == "__main__":
    jobs = []
    s0 = 100_000
    B_GRID = [0.05, 0.0625, 0.075, 0.10, 0.15]
    for clock in ["rel", "abs"]:
        for b in B_GRID:
            for load in LOADS:
                for s in range(120):
                    jobs.append(("main", (1e-5, b, load, clock, s0)))
                    s0 += 1
    for mu_c in [1e-7, 1e-6]:
        for load in LOADS:
            for s in range(120):
                jobs.append(("main", (mu_c, 0.075, load, "rel", s0)))
                s0 += 1
    print("jobs:", len(jobs), flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=60) as ex:
        futs = [ex.submit(refine if k == "refine" else main_run, a)
                for k, a in jobs]
        for i, f in enumerate(futs):
            out.append(f.result())
            if (i + 1) % 300 == 0:
                print(f"{i+1}/{len(futs)}", flush=True)
                with open("ensemble_partial.json", "w") as fh:
                    json.dump(out, fh)
    with open("ensemble_results.json", "w") as fh:
        json.dump(out, fh)
    print("DONE", len(out))
