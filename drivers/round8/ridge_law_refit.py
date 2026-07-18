"""Round-8 ridge-law refit and criterion characterisation (analysis module).

Load-axis convention (fixed): theta = U_d/s_d = mud/sd, U_d = mud (no factor of 2).
Freeze/flow ridge = the load at which the LOADED median T_fix crosses `level` (canonical 1/2)
times the matched NEUTRAL (mud=0) median, every geometry normalised by ITS OWN pooled anchor.

Two ridge parametrisations coexist in the deposit and are both refit here:
  * traced Lambda-ridge law: at FIXED Lambda = mud*sd, sweep sd; s* = the sd at which the median
    ratio crosses `level` scanning UP in sd (log-x / linear-y interpolation). k enters via the
    power law s* ~ Lambda^gamma with gamma = 1/(k+1): k=0 <-> gamma=1 (U_d = const, pure rate),
    k=1 <-> gamma=1/2 (Haigh U_d/s_d = const), k=2 <-> gamma=1/3 (U_d/s_d^2 = const).
  * ridge profile: at FIXED sd, sweep theta = mud/sd; theta* = crossing scanning up in theta
    (ridge_analysis.loglin_cross, deposited round-7 estimator).
"""
import numpy as np
import pandas as pd
from scipy import stats

GAMMA_MODELS = {"k0_pure_rate": 1.0, "k1_Haigh": 0.5, "k2": 1.0 / 3.0, "sstar_const": 0.0}


def cross_up_in_x_lin(x, ratio, level=0.5):
    """First upward crossing of ratio through `level` scanning up in x; log-x / linear-y interp.
    (This reproduces the deposited traced-ridge s* values to ~1e-7.)"""
    o = np.argsort(x); x = np.asarray(x, float)[o]; r = np.asarray(ratio, float)[o]
    for i in range(len(x) - 1):
        if (r[i] < level) and (r[i + 1] >= level):
            lx0, lx1 = np.log(x[i]), np.log(x[i + 1]); y0, y1 = r[i], r[i + 1]
            return float(np.exp(lx0 + (level - y0) * (lx1 - lx0) / (y1 - y0)))
    return np.nan


def ols_slope(x, y):
    n = len(x); X = np.vstack([np.ones(n), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; rss = float(resid @ resid); dof = n - 2
    cov = (rss / dof) * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), rss, dof


def fit_ridge_law(Lambda, s_star):
    """Free power-law fit + constrained-exponent comparison (F-test and dAIC vs free)."""
    L = np.asarray(Lambda, float); s = np.asarray(s_star, float)
    ok = np.isfinite(s) & (s > 0); L, s = L[ok], s[ok]
    x, y = np.log(L), np.log(s); n = len(x)
    beta, se, rss_f, dof = ols_slope(x, y)
    gamma, gse = float(beta[1]), float(se[1])
    tcrit = float(stats.t.ppf(0.975, dof))
    out = dict(n=int(n), gamma=gamma, gamma_se=gse, gamma_ci95=[gamma - tcrit * gse, gamma + tcrit * gse],
               k=1.0 / gamma - 1.0, k_se=gse / gamma ** 2,
               k_ci95=[1.0 / (gamma + tcrit * gse) - 1.0, 1.0 / (gamma - tcrit * gse) - 1.0],
               prefactor=float(np.exp(beta[0])), r2=float(1 - rss_f / np.sum((y - y.mean()) ** 2)),
               rss_free=rss_f, dof=int(dof))
    aic_free = n * np.log(rss_f / n) + 2 * 2
    mc = {}
    for name, g0 in GAMMA_MODELS.items():
        a = np.mean(y - g0 * x); rss_c = float(np.sum((y - a - g0 * x) ** 2))
        F = ((rss_c - rss_f) / 1) / (rss_f / dof); p = float(stats.f.sf(F, 1, dof))
        aic_c = n * np.log(rss_c / n) + 2 * 1
        mc[name] = dict(gamma0=g0, dAIC=float(aic_c - aic_free), F=float(F), p=p)
    out["models_vs_free"] = mc
    return out


def family_slope_test(Lambda, s_star, family):
    """Do the two ladder families (0 = fixed-Lambda T1, 1 = regime grid) share one exponent?
    Common-slope vs separate-slope F-test, plus a quadratic-in-log-Lambda curvature test."""
    L = np.asarray(Lambda, float); s = np.asarray(s_star, float); f = np.asarray(family, int)
    x, y = np.log(L), np.log(s); n = len(x)
    X1 = np.vstack([np.ones(n), x]).T
    X2 = np.vstack([np.ones(n), x, f, f * x]).T
    X3 = np.vstack([np.ones(n), x, x ** 2]).T

    def rss(X):
        b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; return float(r @ r), b
    rss1, b1 = rss(X1); rss2, b2 = rss(X2); rss3, b3 = rss(X3)
    F = ((rss1 - rss2) / 2) / (rss2 / (n - 4)); pF = float(stats.f.sf(F, 2, n - 4))
    F3 = ((rss1 - rss3) / 1) / (rss3 / (n - 3)); p3 = float(stats.f.sf(F3, 1, n - 3))
    aic = lambda r, p: n * np.log(r / n) + 2 * p
    return dict(slope_T1=float(b2[1]), slope_grid=float(b2[1] + b2[3]), family_F=float(F), family_p=pF,
                dAIC_separate_minus_common=float(aic(rss2, 4) - aic(rss1, 2)),
                curvature=float(b3[2]), curvature_F=float(F3), curvature_p=p3,
                dAIC_quad_minus_lin=float(aic(rss3, 3) - aic(rss1, 2)))


def crossing_elasticity(x, ratio, level=0.5):
    """d ln s* / d ln a for a COMMON anchor multiplier a (ratio = median/(a*A0)); log-x/lin-y interp."""
    o = np.argsort(x); x = np.asarray(x, float)[o]; r = np.asarray(ratio, float)[o]
    for i in range(len(x) - 1):
        if (r[i] < level) and (r[i + 1] >= level):
            return float(level * (np.log(x[i + 1]) - np.log(x[i])) / (r[i + 1] - r[i]))
    return np.nan


def dersimonian_laird(y, v):
    y = np.asarray(y, float); v = np.asarray(v, float); w = 1.0 / v
    ybar = np.sum(w * y) / np.sum(w); Q = float(np.sum(w * (y - ybar) ** 2))
    k = len(y); df = k - 1; C = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / C); ws = 1.0 / (v + tau2)
    mu = float(np.sum(ws * y) / np.sum(ws)); se = float(np.sqrt(1.0 / np.sum(ws)))
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return dict(mu=mu, se=se, ci=(mu - 1.96 * se, mu + 1.96 * se), tau2=float(tau2),
                I2=float(I2), Q=Q, Q_p=float(stats.chi2.sf(Q, df)))
