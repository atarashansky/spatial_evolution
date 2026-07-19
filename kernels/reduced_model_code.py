import numpy as np
import pandas as pd
import pickle
import scipy.stats as st
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text

META_GREY = "#888888"


def apply_figure_style(*, frame="open", font=None, sizes=(8, 7, 6), grid=False):
    import matplotlib as mpl
    if frame not in ("open", "boxed", "none"):
        raise ValueError(f"frame must be 'open'|'boxed'|'none', got {frame!r}")
    try:
        import os, sys, glob, matplotlib.font_manager as fm
        fdir = os.path.join(os.environ.get("CONDA_PREFIX") or sys.prefix, "fonts")
        if os.path.isdir(fdir):
            known = {f.fname for f in fm.fontManager.ttflist}
            for f in glob.glob(os.path.join(fdir, "*.ttf")):
                if f not in known:
                    fm.fontManager.addfont(f)
    except Exception:
        pass
    base, secondary, tick = sizes
    boxed = (frame == "boxed")
    rc = {
        "font.family": "sans-serif",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": boxed, "axes.spines.right": boxed,
        "axes.spines.left": frame != "none", "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.labelweight": "normal",
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    if font:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
    mpl.rcParams.update(rc)


def panel_letter(ax, letter, dx=-0.18, dy=1.02, case="lower", fontsize=None):
    import matplotlib.pyplot as plt
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 8) + 1
    s = letter.lower() if case == "lower" else letter.upper()
    ax.text(dx, dy, s, transform=ax.transAxes,
            fontweight="bold", fontsize=fontsize, va="bottom", ha="left")


# Load data
tf = pd.read_parquet("/root/src/.operon/orgs/01582c35-9930-414e-93b9-0dd852675b4d/artifacts/proj_bcf0e7b390f2/13c7f761-ec3d-4981-844b-74e4fb1b3098/v9cac3e79_tfix_results.parquet")
runs = pd.read_parquet("/root/src/.operon/orgs/01582c35-9930-414e-93b9-0dd852675b4d/artifacts/proj_bcf0e7b390f2/e41172e1-7917-4f85-9f25-bcfd445a95df/v6ef8e441_reduced_model_runs_raw.parquet")
ser = pickle.load(open("/root/src/.operon/orgs/01582c35-9930-414e-93b9-0dd852675b4d/artifacts/proj_bcf0e7b390f2/21efa53c-2b21-4774-afb6-de0bba7ab920/v5282c0fa_reduced_model_series.pkl", "rb"))

p_die = float(runs["p_die"].iloc[0])

rng = np.random.default_rng(0)

def cond_runs(tag, exp="A_load"):
    return runs[(runs.exp == exp) & (runs.tag == tag)]

# Compute D_g estimates
lam_tags = ["Lam2.5e-4", "Lam1e-3", "Lam4e-3"]
area_tags = ["Lam1e-3_L16", "Lam1e-3", "Lam1e-3_L64", "Lam1e-3_L128", "Lam1e-3_L256"]
Ls_of = {"Lam1e-3": 32, "Lam2.5e-4": 32, "Lam4e-3": 32, "Lam1e-3_L16": 16, "Lam1e-3_L64": 64, "Lam1e-3_L128": 128, "Lam1e-3_L256": 256}
Lam_of = {"Lam1e-3": 1e-3, "Lam2.5e-4": 2.5e-4, "Lam4e-3": 4e-3, "Lam1e-3_L16": 1e-3, "Lam1e-3_L64": 1e-3, "Lam1e-3_L128": 1e-3, "Lam1e-3_L256": 1e-3}
mud_of = {"Lam1e-3": 0.1, "Lam2.5e-4": 0.025, "Lam4e-3": 0.4, "Lam1e-3_L16": 0.1, "Lam1e-3_L64": 0.1, "Lam1e-3_L128": 0.1, "Lam1e-3_L256": 0.1}
sd = 0.01

def dg_estimate(tag, exp="A_load", frac=0.5, use="g", nboot=1000):
    rr = cond_runs(tag, exp)
    series = [ser[r] for r in rr["run"]]
    tf_med = rr["t_fix"].median()
    fgrid = np.arange(200, int(frac * tf_med) + 1, 200)
    G2 = np.full((len(series), fgrid.size), np.nan)
    for i, s in enumerate(series):
        gv = s[use]; m = min(fgrid.size, gv.size); G2[i, :m] = gv[:m] ** 2
    def slope_of(idx):
        m2 = np.nanmean(G2[idx], axis=0)
        okk = ~np.isnan(m2)
        return np.sum(fgrid[okk] * m2[okk]) / np.sum(fgrid[okk] ** 2)
    n = len(series); idx0 = np.arange(n)
    s0 = slope_of(idx0)
    boots = np.array([slope_of(rng.choice(idx0, n, replace=True)) for _ in range(nboot)])
    return dict(Dg=0.5 * s0, ci=[0.5 * np.percentile(boots, 2.5), 0.5 * np.percentile(boots, 97.5)], n=n, tf_med=float(tf_med))

DG = {}
for tag in Ls_of:
    d = dg_estimate(tag)
    Ls = Ls_of[tag]; N = 192 * Ls / 2
    Dg_th = 0.5 * mud_of[tag] * sd ** 2 * p_die * (2.0 / N)
    d.update(dict(tag=tag, L_short=Ls, Lambda=Lam_of[tag], N=N, Dg_theory=Dg_th, ratio=d["Dg"] / Dg_th))
    DG[tag] = d

# Power law fits for D_g
xL = [Lam_of[t] for t in lam_tags]; yL = [DG[t]["Dg"] for t in lam_tags]
eL = np.array([[DG[t]["Dg"] - DG[t]["ci"][0], DG[t]["ci"][1] - DG[t]["Dg"]] for t in lam_tags]).T
xx = np.array([2e-4, 5e-3])
k = np.exp(np.polyfit(np.log(xL), np.log(yL), 1)[1])
ex = np.polyfit(np.log(xL), np.log(yL), 1)[0]

Ns = np.array([192 * Ls_of[t] / 2 for t in area_tags]); Dgs = np.array([DG[t]["Dg"] for t in area_tags])
bfit = np.polyfit(np.log(Ns), np.log(Dgs), 1)
Dg_of_N = lambda N: np.exp(bfit[1]) * N ** bfit[0]

Da = [DG[t]["Dg"] for t in area_tags]
eA = np.array([[DG[t]["Dg"] - DG[t]["ci"][0], DG[t]["ci"][1] - DG[t]["Dg"]] for t in area_tags]).T
nn = np.array([1400, 26000])

# MSD alive curves
def msd_alive(tag, exp, stride, minfrac_alive=0.9):
    rr = cond_runs(tag, exp); series = [ser[r] for r in rr["run"]]
    tf_vals = rr.t_fix.values.astype(float); tf_med = np.median(tf_vals)
    Tuse = int(np.percentile(tf_vals, 10))
    fgrid = np.arange(stride, Tuse + 1, stride)
    H = np.full((len(series), fgrid.size), np.nan)
    for i, s in enumerate(series):
        h = s["h_bar"]; m = min(fgrid.size, h.size); H[i, :m] = h[:m]
    msd = np.nanmean((H - 96.0) ** 2, axis=0); nal = np.sum(~np.isnan(H), axis=0)
    ok = nal >= minfrac_alive * len(series)
    return fgrid[ok], msd[ok], nal[ok], tf_med, len(series)

curves = {}
for tag, exp, stride in [("Lam2.5e-4", "A_load", 20), ("Lam1e-3", "A_load", 20), ("Lam4e-3", "A_load", 20),
                          ("Lam1e-3_L16", "A_load", 20), ("Lam1e-3_L64", "A_load", 20), ("Lam1e-3_L128", "A_load", 20), ("Lam1e-3_L256", "A_load", 20),
                          ("neu_L16", "D_neutral", 100), ("neu_L32", "D_neutral", 100), ("neu_L64", "D_neutral", 100),
                          ("neu_L128", "D_neutral", 100), ("neu_L256", "D_neutral", 100)]:
    f, m, na, tfm, n = msd_alive(tag, exp, stride)
    curves[tag] = dict(f=f, msd=m, tf_med=tfm, n=n)

fN, mN = curves["neu_L32"]["f"], curves["neu_L32"]["msd"]
fL, mL = curves["Lam1e-3"]["f"], curves["Lam1e-3"]["msd"]

# D0 estimates from neutral runs
D0res = {}
for tag in sorted(runs[runs.exp == "D_neutral"]["tag"].unique(), key=lambda t: int(t.split("L")[-1])):
    rr = cond_runs(tag, "D_neutral"); series = [ser[r] for r in rr["run"]]
    Ls = int(rr.L_short.iloc[0]); tf_med = float(rr.t_fix.median())
    stride = 100; Tuse = int(0.5 * tf_med)
    fgrid = np.arange(stride, Tuse + 1, stride)
    H = np.full((len(series), fgrid.size), np.nan)
    for i, s in enumerate(series):
        h = s["h_bar"]; m = min(fgrid.size, h.size); H[i, :m] = h[:m]
    msd = np.nanmean((H - 96.0) ** 2, axis=0); nal = np.sum(~np.isnan(H), axis=0)
    ok = nal >= 0.8 * len(series)
    we = fgrid <= 0.05 * tf_med
    D0_early = np.sum(fgrid[we & ok] * msd[we & ok]) / np.sum(fgrid[we & ok] ** 2) / 2
    ll = st.linregress(np.log(fgrid[ok]), np.log(msd[ok]))
    D0res[tag] = dict(L_short=Ls, tf_med=tf_med, D0_early=D0_early, msd_exp=ll.slope, fgrid=fgrid[ok], msd=msd[ok], n=len(series))

# Drift response (panel c)
crows = []
for tag in sorted(runs[runs.exp == "B_drift"]["tag"].unique()):
    rr = cond_runs(tag, "B_drift"); series = [ser[r] for r in rr["run"]]
    Ls = int(rr.L_short.iloc[0]); g0 = -np.log(float(rr.init_ratio.iloc[0]))
    vs = []
    for s in series:
        t_s = s["h_frame"]; h = s["h_bar"]
        i0 = int(0.15 * t_s.size); i1 = int(0.85 * t_s.size)
        if i1 - i0 > 10:
            b = np.polyfit(t_s[i0:i1], h[i0:i1], 1)
            vs.append(b[0])
    vs = np.array(vs)
    win = rr.winner.values
    crows.append(dict(tag=tag, L_short=Ls, g0=g0, v_med=np.median(vs), v_mean=vs.mean(), v_sem=vs.std(ddof=1) / np.sqrt(vs.size),
                      frac_winner1=float((win == 1).mean()), tf_med=float(rr.t_fix.median()), n=vs.size))
cdf = pd.DataFrame(crows).sort_values(["L_short", "g0"])

# Fit drift response
a_v, al_v = 0.5 * (0.0307 + 0.0286), 0.5 * (0.696 + 0.658)
v_of_g = lambda g: np.sign(g) * a_v * np.abs(g) ** al_v

# D0 power law
Lsn = np.array([D0res[t]["L_short"] for t in D0res]); D0s = np.array([D0res[t]["D0_early"] for t in D0res])
d0fit = np.polyfit(np.log(Lsn), np.log(D0s), 1)
D0_of_L = lambda L: np.exp(d0fit[1]) * L ** d0fit[0]

# SDE predictions
rng2 = np.random.default_rng(42)

def sde_tfix(Ls, loaded, n=3000, dt=20.0, tmax=6_000_000):
    L_long = 192; Delta = L_long / 2.0; N = L_long * Ls / 2.0
    D0 = D0_of_L(Ls); Dg = Dg_of_N(N) if loaded else 0.0
    h = np.zeros(n); g = np.zeros(n); tfp = np.full(n, np.nan); alive = np.ones(n, bool); t = 0.0
    sdh = np.sqrt(2 * D0 * dt); sdg = np.sqrt(2 * Dg * dt)
    while t < tmax and alive.any():
        idx = np.where(alive)[0]
        if loaded:
            g[idx] += sdg * rng2.standard_normal(idx.size)
            h[idx] += v_of_g(g[idx]) * dt
        h[idx] += sdh * rng2.standard_normal(idx.size)
        t += dt
        cr = idx[np.abs(h[idx]) >= Delta]; tfp[cr] = t; alive[cr] = False
    return tfp

pred = {}
for Ls in [16, 32, 64, 128, 256]:
    tl = sde_tfix(Ls, True); tn = sde_tfix(Ls, False, dt=200.0, tmax=40_000_000)
    pred[Ls] = dict(load_med=float(np.nanmedian(tl)), load_ci=list(np.nanpercentile(tl, [25, 75])),
                    neu_med=float(np.nanmedian(tn)), neu_cens=int(np.isnan(tn).sum()), load_cens=int(np.isnan(tl).sum()))

# Time-dependent neutral SDE
neufit = {}
for tag, d in D0res.items():
    f = d["fgrid"]; m = d["msd"]
    ll = st.linregress(np.log(f), np.log(m))
    neufit[d["L_short"]] = dict(A=np.exp(ll.intercept), zeta=ll.slope)
Lsn2 = np.array(sorted(neufit)); zet = np.array([neufit[L]["zeta"] for L in Lsn2]); Aa = np.array([neufit[L]["A"] for L in Lsn2])
zbar = zet.mean()
Afit = np.polyfit(np.log(Lsn2), np.log(Aa), 1)

def sde_tfix_neu(Ls, n=3000, dt=200.0, tmax=60_000_000):
    A = np.exp(Afit[1]) * Ls ** Afit[0]; zeta = zbar; Delta = 96.0
    h = np.zeros(n); tfp = np.full(n, np.nan); alive = np.ones(n, bool); t = 1e-9
    while t < tmax and alive.any():
        idx = np.where(alive)[0]
        var_inc = A * ((t + dt) ** zeta - t ** zeta)
        h[idx] += np.sqrt(var_inc) * rng2.standard_normal(idx.size)
        t += dt
        cr = idx[np.abs(h[idx]) >= Delta]; tfp[cr] = t; alive[cr] = False
    return tfp

for Ls in [16, 32, 64, 128, 256]:
    tn = sde_tfix_neu(Ls)
    pred[Ls]["neu_med_td"] = float(np.nanmedian(tn)); pred[Ls]["neu_td_cens"] = int(np.isnan(tn).sum())

# Lattice T_fix table
lat = tf[tf.exp == 1].groupby(["regime", "L_short"])["t_fix"].agg(["median", "size"]).unstack("regime")
rowsT = []
for Ls in [16, 32, 64, 128, 256]:
    lat_load = lat.loc[Ls, ("median", "load")]; lat_neu = lat.loc[Ls, ("median", "neutral")]
    p = pred[Ls]
    rowsT.append(dict(L_short=Ls, lattice_load=lat_load, pred_load=p["load_med"], lattice_neutral=lat_neu, pred_neutral=p["neu_med_td"],
                      lat_ratio=lat_neu / lat_load, pred_ratio=p["neu_med_td"] / p["load_med"]))
T = pd.DataFrame(rowsT)

def expo(y): return np.polyfit(np.log([16, 32, 64, 128, 256]), np.log(y), 1)[0]

# Crossover time
gstar = 0.03; ceff = v_of_g(gstar) / gstar
D0e = D0res["neu_L32"]["D0_early"]; Dg = DG["Lam1e-3"]["Dg"]
fL2, mL2 = curves["Lam1e-3"]["f"], curves["Lam1e-3"]["msd"]
mNi = np.interp(fL2, fN, mN)
ratio = mL2 / mNi
j = np.argmax(ratio >= 2.0)
t2 = fL2[j] if ratio[j] >= 2.0 else np.nan
txv = t2 * p_die

tt = np.logspace(np.log10(fL[0]), np.log10(fL[-1]), 100)
cLOAD = "#c0392b"

# Final figure
apply_figure_style(sizes=(8, 7, 6))
fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw=dict(hspace=0.55, wspace=0.36))
(axA, axB), (axC, axD) = axes

# a
axA.loglog(fN * p_die, mN, color="0.35", lw=1.6, label="neutral (μd = 0)")
axA.loglog(fL * p_die, mL, color=cLOAD, lw=1.6, label="load Λ = 10⁻³")
axA.loglog(tt * p_die, 2 * D0e * tt, ls=":", color="k", lw=1.0, label="diffusive 2D₀t (measured D₀)")
axA.loglog(tt * p_die, (2 / 3) * (ceff ** 2) * Dg * tt ** 3, ls="--", color=cLOAD, lw=1.0, label="reduced model (2/3)c²D_g t³")
axA.axvline(txv, color="0.6", lw=0.8)
axA.text(txv * 1.2, 4e-2, f"t_x ≈ {txv:.0f}\ndivisions/site", fontsize=6, color="0.4")
axA.set_ylim(1e-3, 3e4)
axA.set_xlabel("time (divisions per site)"); axA.set_ylabel("MSD of h̄ (cells²)")
axA.set_title("Load makes boundary motion super-diffusive", loc="left", fontsize=7)
axA.legend(frameon=False, loc="upper left", fontsize=5.5)

# b
axB.errorbar(xL, yL, yerr=eL, fmt="o", color=cLOAD, ms=4, lw=1)
axB.loglog(xx, k * xx ** ex, color=cLOAD, lw=0.9, ls="--")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel("load rate Λ = μd·sd  (192×32)"); axB.set_ylabel("gap diffusivity D_g (frame⁻¹)")
axB.set_title(f"Gap random-walk rate: D_g ∝ Λ^{ex:.2f}, ∝ N^{bfit[0]:.2f}", loc="left", fontsize=7)
axB.text(2.2e-4, 3.5e-8, f"D_g ∝ Λ^{ex:.2f}  [0.96, 1.18]", fontsize=6, color=cLOAD)
ins = axB.inset_axes([0.57, 0.13, 0.4, 0.42])
ins.errorbar(Ns, Da, yerr=eA, fmt="s", color="0.25", ms=3, lw=0.8)
ins.loglog(nn, np.exp(bfit[1]) * nn ** bfit[0], color="0.25", lw=0.8, ls="--")
ins.set_xscale("log"); ins.set_yscale("log"); ins.tick_params(labelsize=5)
ins.set_title("vs clone area N", fontsize=5, pad=1.5)
ins.text(1500, 6.0e-9, f"N^{bfit[0]:.2f}\n(mean-field N⁻¹)", fontsize=5)

# c
gg = np.logspace(np.log10(0.004), np.log10(0.09), 50)
for Ls, mk, col in [(32, "o", cLOAD), (128, "^", "#7f1d1d")]:
    sub = cdf[cdf.L_short == Ls].sort_values("g0")
    axC.errorbar(sub.g0, sub.v_mean, yerr=1.96 * sub.v_sem, fmt=mk, ms=4, color=col, lw=1, label=f"192×{Ls}")
axC.loglog(gg, a_v * gg ** al_v, "--", color="0.3", lw=0.9)
axC.text(0.0043, 2.2e-3, f"fit: v = {a_v:.3f} g^{al_v:.2f}", fontsize=6)
axC.loglog(gg, 0.15 * gg, ":", color="0.6", lw=0.9); axC.text(0.033, 0.15 * 0.033 * 0.45, "linear\n(small-g slope)", fontsize=6, color="0.45")
axC.set_xlabel("imposed log-fitness gap g₀ (μd = 0)"); axC.set_ylabel("interface velocity v (cells/frame)")
axC.set_title("Drift response is sublinear, geometry-independent", loc="left", fontsize=7)
axC.legend(frameon=False, loc="lower right")

# d
Lsv = [16, 32, 64, 128, 256]
axD.loglog(Lsv, T.lattice_load, "o", color=cLOAD, ms=5, label="lattice, load Λ = 10⁻³")
axD.loglog(Lsv, T.pred_load, "o", mfc="white", mec=cLOAD, ms=5, label="reduced model, load")
axD.loglog(Lsv, T.lattice_neutral, "s", color="0.35", ms=5, label="lattice, neutral")
axD.loglog(Lsv, T.pred_neutral, "s", mfc="white", mec="0.35", ms=5, label="reduced model, neutral")
axD.set_ylim(2.2e4, 2.5e7)
axD.set_xlabel("interface length L_short (L_long = 192)"); axD.set_ylabel("median T_fix (frames)")
axD.set_title("No-free-parameter test of T_fix scaling", loc="left", fontsize=7)
axD.legend(frameon=False, loc="upper left", fontsize=5.5, ncol=1)

for ax, let in zip([axA, axB, axC, axD], "abcd"):
    panel_letter(ax, let)

fig.savefig("reduced_model.png", dpi=220, bbox_inches="tight")
print("saved final figure")