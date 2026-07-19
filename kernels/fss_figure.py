"""Figure: crossover load theta* vs lattice size (fss_summary.json only).

Panel a: theta*(L_long) at fixed L_short = 32 for both estimators (half-drop
         primary, level-1/2 legacy) with run-level bootstrap CIs, plus the
         L_short arm (L_long = 192) and the walled arm at 192x32; the constant
         fits are drawn as horizontal bands. n per point is in the caption.
Panel b: the per-cell median-T_fix / own-anchor ratio profiles vs theta,
         showing the crossover the estimators read.
All plotted values are read from fss_summary.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"half_drop": "#1f4e79", "level_0.5": "#c05a00",   # blue / orange (CVD-safe)
       "walled": "#7b3294", "lshort": "#4c9a2a", "pool": "#7f7f7f"}
LADDER = ["G96", "G192", "G384", "G768", "G1536"]


def make_figure(summary_path="fss_summary.json", out_png="fig_fss_crossover.png"):
    S = json.load(open(summary_path))
    C = S["cells"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw=dict(wspace=0.32))
    axA, axB = axes

    # ---------------- panel a: theta* vs L_long
    for est, dx in (("half_drop", -0.02), ("level_0.5", +0.02)):
        Ls, v, lo, hi = [], [], [], []
        for c in LADDER:
            if c in C and np.isfinite(C[c][est][0]):
                Ls.append(C[c]["L_long"]); v.append(C[c][est][0])
                lo.append(C[c][est][1]); hi.append(C[c][est][2])
        Ls = np.array(Ls, float); v = np.array(v); lo = np.array(lo); hi = np.array(hi)
        x = Ls * (10 ** dx)
        label = ("half-drop (convention-free)" if est == "half_drop"
                 else "ratio = 1/2 crossing (legacy)")
        axA.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o-", color=COL[est], ms=4.5,
                     lw=1.2, capsize=2.5, label=label, zorder=3)
        cf = S["convergence_L_long"].get(est, {}).get("fits", {}).get("constant", {})
        if cf and "theta_inf" in cf:
            ti = cf["theta_inf"]
            axA.axhspan(ti[1], ti[2], color=COL[est], alpha=0.10, lw=0, zorder=1)
            axA.axhline(ti[0], color=COL[est], lw=0.8, ls="--", alpha=0.7, zorder=2)
    # walled arm and L_short arm at their L_long, plotted for the half-drop estimator
    if "W32" in C and np.isfinite(C["W32"]["half_drop"][0]):
        w = C["W32"]["half_drop"]
        axA.errorbar([192 * 1.12], [w[0]], yerr=[[w[0] - w[1]], [w[2] - w[0]]], fmt="s",
                     color=COL["walled"], ms=5, capsize=2.5, label="walled 192x32", zorder=4)
    for c, mk in (("S16", "v"), ("S64", "^")):
        if c in C and np.isfinite(C[c]["half_drop"][0]):
            w = C[c]["half_drop"]; f = 0.89 if c == "S16" else 0.79
            axA.errorbar([192 * f], [w[0]], yerr=[[w[0] - w[1]], [w[2] - w[0]]], fmt=mk,
                         color=COL["lshort"], ms=5, capsize=2.5,
                         label=f"L_short = {C[c]['L_short']} (L_long = 192)", zorder=4)
    # deposited pool reference
    pr = S.get("pool_replication_G192_vs_deposited", {})
    if pr:
        ph = pr["deposited_pool"].get("half_drop")
        if ph and np.isfinite(ph[0]):
            axA.errorbar([192 / 1.12], [ph[0]], yerr=[[ph[0] - ph[1]], [ph[2] - ph[0]]],
                         fmt="D", color=COL["pool"], ms=4.5, capsize=2.5,
                         label="deposited 192x32 pool", zorder=4)
    axA.set_xscale("log")
    axA.set_xticks([96, 192, 384, 768, 1536])
    axA.set_xticklabels(["96", "192", "384", "768", "1536"])
    axA.minorticks_off()
    axA.set_xlabel("L_long (sites), at L_short = 32")
    axA.set_ylabel(r"crossover load $\theta^{*} = U_d^{*}/s_d$")
    axA.set_title(S.get("figure_title_a", "Crossover load vs lattice size, s_d = 0.01"),
                  loc="left")
    axA.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=2,
               handlelength=1.6, columnspacing=1.2)
    axA.margins(x=0.06)

    # ---------------- panel b: ratio profiles per cell
    cmap = plt.get_cmap("viridis")
    prof_cells = [c for c in LADDER if c in C] + [c for c in ("W32",) if c in C]
    for i, c in enumerate(prof_cells):
        rr = C[c]["rung_ratio"]
        th = np.array(sorted(float(k) for k in rr))
        r = np.array([rr[f"{t:g}"] for t in th])
        if c == "W32":
            axB.plot(th, r, "s--", color=COL["walled"], ms=3.2, lw=1.0,
                     label="walled 192x32")
        else:
            col = cmap(i / max(1, len([x for x in prof_cells if x != "W32"]) - 1))
            axB.plot(th, r, "o-", color=col, ms=3.2, lw=1.1,
                     label=f"{C[c]['L_long']}x{C[c]['L_short']}")
    axB.axhline(0.5, color="0.3", lw=0.8, ls=":")
    axB.text(9.5, 0.52, "ratio = 1/2", ha="right", va="bottom", fontsize=7, color="0.3")
    axB.set_xscale("log")
    axB.set_xlabel(r"load $\theta = U_d/s_d$")
    axB.set_ylabel(r"median $T_{fix}(\theta)$ / median own-anchor $T_{fix}$")
    axB.set_title("Fixation-time ratio profiles per lattice", loc="left")
    axB.legend(frameon=False, loc="best", handlelength=1.6, ncol=2)
    axB.margins(x=0.04)
    for lab, ax in (("a", axA), ("b", axB)):
        ax.text(-0.13, 1.05, lab, transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="left")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import sys
    make_figure(*(sys.argv[1:3] if len(sys.argv) > 1 else []))
