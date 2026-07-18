"""Build ridge_profile_summary.json + figure from a results dict.
The memo is written in the notebook (needs prose + judgement), not here."""
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SDS = [0.0025, 0.005, 0.01, 0.02, 0.04]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        v = float(o); return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_summary(R, path, extra=None):
    D = dict(R)
    if extra:
        D.update(extra)
    with open(path, "w") as f:
        json.dump(_clean(D), f, indent=1)
    return path


def make_figure(R, cross_tbl, path, style_helpers=None, R_alt=None, alt_label=None,
                iso_points=None, y2_max=1.6):
    """Three panels: (a) ridge vs s_d with CIs, RE band + quadratic; (b) crossing-level
    sensitivity; (c) forest plot of per-s_d estimates with RE pooled diamond."""
    if style_helpers:
        style_helpers["apply"]()
    fig = plt.figure(figsize=(11.0, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
    per = R["per_sd"]
    sds = np.array(SDS)
    y = np.array([per[str(s)]["theta_star"] for s in SDS])
    lo = np.array([per[str(s)]["ci"][0] for s in SDS]); hi = np.array([per[str(s)]["ci"][1] for s in SDS])
    focal = "#1f4e79"; grey = "#8c8c8c"; accent = "#c44e52"

    # ---- (a) profile
    mu, ci = R["re_pooled"], R["re_ci"]
    xg = np.geomspace(sds.min() * 0.75, sds.max() * 1.35, 300)
    ax.fill_between(xg, ci[0], ci[1], color=focal, alpha=0.12, lw=0,
                    label="RE-pooled constant, 95% CI")
    ax.axhline(mu, color=focal, lw=1.0, ls="--", alpha=0.9)
    # quadratic profile
    q = R["profile_fits"]["quadratic"]
    lx = np.log(xg); yq = np.exp(q["a"] * lx ** 2 + q["b"] * lx + q["c"])
    ax.plot(xg, yq, color=accent, lw=1.4, label="quadratic in log $s_d$ (loglin weights)")
    if R_alt is not None:  # ladder-only / legacy-anchor overlay, hollow markers
        ya = np.array([R_alt["per_sd"][str(s)]["theta_star"] for s in SDS])
        ax.plot(sds * 1.08, ya, "o", mfc="none", mec=grey, ms=5, mew=1.0, ls="none",
                label=alt_label or "deposited anchor (200 runs)")
    if iso_points is not None:  # shelf-robust monotone crossing at the low-s_d rung
        p, l, h = iso_points  # value, lo, hi at sd=0.0025
        ax.errorbar([sds[0] * 0.90], [p], yerr=[[p - l], [h - p]], fmt="^", mfc="white", mec=focal,
                    ecolor=focal, ms=5, lw=1.0, capsize=2, elinewidth=0.9, alpha=0.8,
                    label="monotone-fit crossing (0.0025 only)")
    ax.errorbar(sds, y, yerr=[y - lo, hi - y], fmt="o", color=focal, ms=5.5, capsize=2.5,
                lw=1.2, label="measured ridge, joint bootstrap 95% CI", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sds); ax.set_xticklabels([f"{s:g}" for s in sds])
    ytk = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    ax.set_yticks(ytk); ax.set_yticklabels([f"{t:g}" for t in ytk]); ax.minorticks_off()
    ax.set_ylim(0.24, 1.25)
    ax.set_xlabel("per-mutation effect size $s_d$")
    ax.set_ylabel(r"freeze/flow ridge $U_d^*/s_d$")
    ax.set_title("Measured ridge is a trough, not a constant;\none number is fair only for 0.003<$s_d$<0.02",
                 loc="left", fontsize=8)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.10, 1.0), fontsize=5.8)

    # ---- (b) crossing-level sensitivity
    levels = sorted(float(k) for k in cross_tbl)
    cmap = matplotlib.colormaps["viridis"]
    for i, lev in enumerate(levels):
        vals = [cross_tbl[str(lev) if str(lev) in cross_tbl else f"{lev}"][str(s)] for s in SDS]
        vals = np.array([np.nan if v is None else v for v in vals], float)
        col = cmap(i / (len(levels) - 1))
        # clip off-scale values (shelf tail) and mark them with an arrow at the panel top
        clipped = vals > y2_max
        vplot = np.where(clipped, np.nan, vals)
        ax2.plot(sds, vplot, "-o", color=col, ms=3.8, lw=1.1, label=f"level {lev:g}")
        for xc, v in zip(sds[clipped], vals[clipped]):
            ax2.annotate(f"{v:.1f}", xy=(xc, y2_max * 0.95), xytext=(xc * (1.0 + 0.55 * i), y2_max * (0.62 - 0.10 * i)),
                         ha="center", fontsize=5.5, color=col,
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=0.7, mutation_scale=6))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xticks(sds); ax2.set_xticklabels([f"{s:g}" for s in sds])
    yt2 = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    ax2.set_yticks(yt2); ax2.set_yticklabels([f"{t:g}" for t in yt2]); ax2.minorticks_off()
    ax2.set_ylim(0.15, y2_max)
    ax2.set_xlabel("effect size $s_d$")
    ax2.set_ylabel(r"$U_d^*/s_d$ at crossing level")
    ax2.set_title("Right-arm rise at every crossing level;\nlevel 0.3 exits the ladder on the low-$s_d$ shelf", loc="left", fontsize=8)
    ax2.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, title="ratio level", title_fontsize=6)

    # ---- (c) forest
    ypos = np.arange(len(SDS))[::-1]
    for j, s in enumerate(SDS):
        ax3.plot([lo[j], hi[j]], [ypos[j]] * 2, color=focal, lw=1.4)
        ax3.plot(y[j], ypos[j], "s", color=focal, ms=5)
        ax3.text(1.32, ypos[j], f"{y[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]", va="center", ha="right", fontsize=6.2)
    yd = -1.2
    ax3.plot([ci[0], ci[1]], [yd, yd], color=accent, lw=2.2)
    ax3.plot([mu], [yd], "D", color=accent, ms=7)
    ax3.text(1.32, yd, f"{mu:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", va="center", ha="right", fontsize=6.2)
    pi = R.get("re_prediction_interval")
    if pi:
        ax3.plot([pi[0], pi[1]], [yd - 0.55] * 2, color=accent, lw=1.0, ls=":")
        ax3.text(1.32, yd - 0.55, f"pred. {pi[0]:.2f}–{pi[1]:.2f}", va="center", ha="right", fontsize=6.2)
    ax3.axvline(mu, color=accent, lw=0.8, ls="--", alpha=0.6)
    ax3.set_yticks(list(ypos) + [yd])
    ax3.set_yticklabels([f"$s_d$ = {s:g}" for s in SDS] + ["RE pooled"])
    ax3.set_xlim(0.25, 1.35); ax3.set_ylim(yd - 1.1, len(SDS) - 0.5)
    ax3.set_xlabel(r"ridge $U_d^*/s_d$ (95% CI)")
    ax3.set_title("Forest plot: heterogeneity beyond\nrun-level noise (I$^2$, $\\tau$ in text)", loc="left", fontsize=8)
    if style_helpers:
        for a, L in zip([ax, ax2, ax3], "abc"):
            style_helpers["letter"](a, L)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig
