"""Two-clone strip simulation with a distribution of fitness effects (DFE).

Extends evolution_strip.py: each deleterious mutation draws its own effect
s ~ DFE instead of the point mass s_d. Bookkeeping stays Lambda = mu_d*E[s]
per division. Effects multiply fitness as (1-s) per hit (clipped s <= 0.95);
per-cell fitness remains a single scalar (the product), exactly as the base
kernel stores it.

DFE families (dfe_kind):
  0  point(s0)                 - degenerate control; consumes NO extra RNG,
                                 so trajectories are bit-identical to the
                                 unmodified kernel at equal seed.
  1  exponential(mean)         - s = -mean * ln(1-u)
  2  lognormal(mean, sigma_ln) - s = exp(mu_ln + sigma_ln*z),
                                 mu_ln = ln(mean) - sigma_ln^2/2  (E[s]=mean)

Composition instrumentation (the purging-filter question):
  Each applied mutation is binned into one of 10 effect-size bins (edges =
  theoretical deciles of the DFE, passed in). Per-cell int32 histograms are
  inherited on refill from the frozen post-death source state (identical
  semantics to the fitness scalar copy). At fixation, the number of
  mutations FIXED in bin b ~= min over surviving cells of hist[cell, b];
  arising counts per bin are tallied globally.

Seeded, serial, bit-reproducible. Self-contained (alias helpers inlined
from evolution_fast4) so numba disk caching works in worker processes.
"""
import numpy as np
from numba import njit


# --------------------------------------------------------- alias sampling
@njit(cache=True)
def _gap(prob, alias, tail_start, log1m):
    u = np.random.rand() * prob.shape[0]
    i = np.int64(u)
    if u - i >= prob[i]:
        i = alias[i]
    if i == prob.shape[0] - 1:
        return tail_start + 1 + np.int64(np.log(1.0 - np.random.rand())
                                         / log1m)
    return i + 1


def build_geom_alias(p, table_n=2048):
    G = table_n - 1
    g = np.arange(1, G + 1, dtype=np.float64)
    pmf = p * (1.0 - p) ** (g - 1.0)
    tail = (1.0 - p) ** G
    w = np.concatenate([pmf, [tail]])
    w /= w.sum()
    n = table_n
    prob = np.zeros(n, np.float64)
    alias = np.zeros(n, np.int64)
    scaled = w * n
    small = [i for i in range(n) if scaled[i] < 1.0]
    large = [i for i in range(n) if scaled[i] >= 1.0]
    while small and large:
        s = small.pop(); l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] -= (1.0 - scaled[s])
        (small if scaled[l] < 1.0 else large).append(l)
    for i in large + small:
        prob[i] = 1.0
        alias[i] = i
    return prob, alias


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def _run_strip_dfe(fitness, label, nbr, n_nbr, sites, x_of, y_of, rowA,
                   numgen, sampling, com_stride, sb, sd, mud, mub, seed,
                   dfe_kind, dfe_p1, dfe_p2, edges,
                   d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
                   victims, src, cp_fit, cp_lab, cp_hist, mut_hist,
                   arise_cnt,
                   h_t, prof_sum, prof_cnt, fitA_t, fitB_t, live_t,
                   samp_frame, com_t, ben_frame, ben_x, ben_y, ben_side, hw):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = h_t.shape[0]
    n_com = com_t.shape[0]
    ben_cap = ben_frame.shape[0]
    L_short = rowA.shape[0]
    kcap = victims.size
    n_bins = arise_cnt.shape[0]
    n_edges = edges.shape[0]

    log1m_mub = np.log(1.0 - 0.5 * mub)
    if mub > 0.0:
        ben_ctr = 1 + np.int64(np.log(1.0 - np.random.rand()) / log1m_mub)
    else:
        ben_ctr = np.int64(1 << 60)
    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)

    # initial tallies
    cntA = np.int64(0)
    cntB = np.int64(0)
    for j in range(M):
        i = sites[j]
        if label[i] == 0:
            cntA += 1
        elif label[i] == 1:
            cntB += 1

    n_ben = np.int64(0)
    n_mut = np.int64(0)
    samp_i = 0
    com_i = 0
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    for f in range(1, numgen + 1):
        # ---- victims of this frame (sorted, exact Bernoulli process)
        k = 0
        while pos < M:
            if k < kcap:
                victims[k] = sites[pos]
                k += 1
            pos += _gap(d_prob, d_alias, d_tail, d_log1m)
        pos -= M

        if k > 0:
            # ---- kill
            for j in range(k):
                i = victims[j]
                if fitness[i] > 0.0:
                    lb = label[i]
                    if lb == 0:
                        cntA -= 1
                        rowA[y_of[i]] -= 1
                    else:
                        cntB -= 1
                    fitness[i] = np.float32(0.0)
                    label[i] = -1

            # ---- source draw on frozen state
            for j in range(k):
                cell = victims[j]
                tot = 0.0
                for o in range(n_nbr):
                    tot += np.float64(fitness[nbr[cell, o]])
                if tot <= 0.0:
                    src[j] = -1
                    continue
                r = np.random.rand() * tot
                acc = 0.0
                chosen = np.int64(-1)
                for o in range(n_nbr):
                    nb = nbr[cell, o]
                    acc += np.float64(fitness[nb])
                    if r < acc:
                        chosen = nb
                        break
                if chosen == -1:
                    for o in range(n_nbr - 1, -1, -1):
                        nb = nbr[cell, o]
                        if fitness[nb] > 0.0:
                            chosen = nb
                            break
                src[j] = chosen
                cp_fit[j] = fitness[chosen]
                cp_lab[j] = label[chosen]
                for b in range(n_bins):
                    cp_hist[j, b] = mut_hist[chosen, b]

            # ---- refill + mutations
            for j in range(k):
                sc = src[j]
                if sc < 0:
                    continue
                cell = victims[j]
                fitness[cell] = cp_fit[j]
                lb = cp_lab[j]
                label[cell] = lb
                for b in range(n_bins):
                    mut_hist[cell, b] = cp_hist[j, b]
                if lb == 0:
                    cntA += 1
                    rowA[y_of[cell]] += 1
                else:
                    cntB += 1

                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else sc
                    # ---- draw effect s ~ DFE
                    if dfe_kind == 0:
                        sv = sd                       # no RNG consumed
                    elif dfe_kind == 1:
                        sv = -dfe_p1 * np.log(1.0 - np.random.rand())
                    else:
                        sv = np.exp(dfe_p1
                                    + dfe_p2 * np.random.standard_normal())
                    if sv > 0.95:
                        sv = 0.95
                    fitness[tgt] -= fitness[tgt] * np.float32(sv)
                    b = 0
                    for e in range(n_edges):
                        if sv > edges[e]:
                            b = e + 1
                    mut_hist[tgt, b] += 1
                    arise_cnt[b] += 1
                    n_mut += 1
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)
                ben_ctr -= 2
                while ben_ctr <= 0:
                    tgt = cell if ben_ctr == -1 else sc
                    fitness[tgt] += fitness[tgt] * np.float32(sb)
                    if n_ben < ben_cap:
                        ben_frame[n_ben] = f
                        ben_x[n_ben] = x_of[tgt]
                        ben_y[n_ben] = y_of[tgt]
                        ben_side[n_ben] = label[tgt]
                        n_ben += 1
                    ben_ctr += 1 + np.int64(
                        np.log(1.0 - np.random.rand()) / log1m_mub)

        # ---- high-rate COM series
        if com_stride > 0 and f % com_stride == 0 and com_i < n_com:
            com_t[com_i] = cntA
            com_i += 1

        # ---- full sampling
        if f % sampling == 0 and samp_i < n_samp:
            for y in range(L_short):
                h_t[samp_i, y] = rowA[y]
            sfA = 0.0
            sfB = 0.0
            nA = np.int64(0)
            nB = np.int64(0)
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    continue
                fi = np.float64(fitness[i])
                d = np.int64(x_of[i]) - np.int64(rowA[y_of[i]])
                b2 = d + hw
                if lb == 0:
                    sfA += fi
                    nA += 1
                    if 0 <= b2 < 2 * hw:
                        prof_sum[samp_i, 0, b2] += fi
                        prof_cnt[samp_i, 0, b2] += 1
                else:
                    sfB += fi
                    nB += 1
                    if 0 <= b2 < 2 * hw:
                        prof_sum[samp_i, 1, b2] += fi
                        prof_cnt[samp_i, 1, b2] += 1
            fitA_t[samp_i] = sfA / max(nA, 1)
            fitB_t[samp_i] = sfB / max(nB, 1)
            live_t[samp_i] = nA + nB
            samp_frame[samp_i] = f
            samp_i += 1

        # ---- fixation early stop
        if cntA == 0 or cntB == 0:
            t_fix = f
            winner = 0 if cntB == 0 else 1
            break

    return t_fix, winner, n_ben, samp_i, com_i, n_mut


# ------------------------------------------------------------------ DFE cfg
def dfe_params(dfe_kind, mean, sigma_ln=0.0, n_bins=10):
    """Return (p1, p2, edges) for the kernel + theoretical decile edges.

    edges: n_bins-1 interior decile boundaries of the (unclipped) DFE, so
    arising mutations land ~uniformly across bins and the fixed-mutation
    histogram reads directly as the purging filter. Point mass: all edges
    at mean (everything lands in bin 0)."""
    q = np.arange(1, n_bins) / n_bins
    if dfe_kind == 0:                       # point(s0)
        p1, p2 = float(mean), 0.0
        edges = np.full(n_bins - 1, float(mean))
    elif dfe_kind == 1:                     # exponential(mean)
        p1, p2 = float(mean), 0.0
        edges = -mean * np.log(1.0 - q)
    elif dfe_kind == 2:                     # lognormal(mean, sigma_ln)
        from scipy.special import ndtri
        p2 = float(sigma_ln)
        p1 = float(np.log(mean) - 0.5 * sigma_ln ** 2)   # mu_ln
        edges = np.exp(p1 + p2 * ndtri(q))
    else:
        raise ValueError("dfe_kind must be 0/1/2")
    return p1, p2, np.ascontiguousarray(edges, np.float64)


# ------------------------------------------------------------------ driver
def _build_nbr(L_long, L_short, periodic_y, moore):
    """Neighbor index table. Grid W = L_long + 2 (x walls), H = L_short
    (+2 if walled y). Dummy site index N has fitness 0 forever."""
    W = L_long + 2
    H = L_short if periodic_y else L_short + 2
    N = W * H
    xs = np.arange(N) % W
    ys = np.arange(N) // W
    if moore:
        d = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        d = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nbr = np.full((N, len(d)), N, np.int32)  # default: dummy
    for o, (dy, dx) in enumerate(d):
        nx = xs + dx
        ny = ys + dy
        ok_x = (nx >= 1) & (nx <= L_long)
        if periodic_y:
            ny = ny % H
            ok = ok_x
        else:
            ok = ok_x & (ny >= 1) & (ny <= L_short)
        idx = ny * W + nx
        nbr[ok, o] = idx[ok]
    return nbr, W, H, N


def evolve_strip_dfe(L_long=512, L_short=64, numgen=2_000_000, dt=2400.0,
                     sb=0.1, sd=0.01, mud=0.1, mub=1e-5, sampling=250,
                     avg=1.16e-5, seed=0, split=None, tilt=0.0,
                     periodic_y=True, moore=True, com_stride=0,
                     prof_halfwidth=64, ben_cap=1 << 20,
                     dfe_kind=0, dfe_mean=None, sigma_ln=0.0, n_bins=10):
    """Two-clone strip run with DFE-drawn deleterious effects.

    dfe_mean defaults to sd (so Lambda = mud*E[s] = mud*sd bookkeeping is
    unchanged). With dfe_kind=0 and dfe_mean=sd the trajectory is
    bit-identical to evolution_strip.evolve_strip at equal seed.

    Extra returns vs base kernel:
      n_mut       total deleterious mutations applied
      arise_cnt   int64[n_bins]  arising mutations per DFE-decile bin
      fixed_cnt   int64[n_bins]  mutations fixed in the surviving population
                  (min over alive cells of the per-cell bin histogram)
      dfe         dict of the DFE configuration incl. bin edges
    """
    if tilt != 0.0:
        assert not periodic_y, "tilted interface requires walled y"
    dfe_mean = sd if dfe_mean is None else float(dfe_mean)
    p1, p2, edges = dfe_params(dfe_kind, dfe_mean, sigma_ln, n_bins)

    nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)

    fitness = np.zeros(N + 1, np.float32)   # +1 dummy
    label = np.full(N + 1, -1, np.int8)
    x_of = (np.arange(N, dtype=np.int32) % W).astype(np.int32)
    y_of = (np.arange(N, dtype=np.int32) // W).astype(np.int32)

    y0 = 0 if periodic_y else 1
    y1 = H if periodic_y else H - 1
    xs = x_of[:N]
    ys = y_of[:N]
    interior = (xs >= 1) & (xs <= L_long) & (ys >= y0) & (ys < y1)
    split = L_long // 2 if split is None else split
    ymid = 0.5 * (y0 + y1 - 1)
    thresh = split + np.tan(np.radians(tilt)) * (ys - ymid)
    isA = interior & (xs <= thresh)
    fitness[:N][interior] = 1.0
    label[:N][isA] = 0
    label[:N][interior & ~isA] = 1

    sites = np.where(interior)[0].astype(np.int32)
    M = sites.size
    rowA = np.zeros(H, np.int32)
    for i in sites[label[sites] == 0]:
        rowA[y_of[i]] += 1

    p_die = 1.0 - np.exp(-avg * dt)
    d_prob, d_alias = build_geom_alias(p_die)
    m_prob, m_alias = build_geom_alias(0.5 * mud if mud > 0 else 0.5)
    d_log1m = np.log(1.0 - p_die)
    m_log1m = np.log(1.0 - (0.5 * mud if mud > 0 else 0.5))

    kmean = M * p_die
    kcap = int(kmean + 12 * np.sqrt(max(kmean, 1.0)) + 64)
    victims = np.empty(kcap, np.int32)
    src = np.empty(kcap, np.int64)
    cp_fit = np.empty(kcap, np.float32)
    cp_lab = np.empty(kcap, np.int8)
    cp_hist = np.zeros((kcap, n_bins), np.int32)
    mut_hist = np.zeros((N + 1, n_bins), np.int32)
    arise_cnt = np.zeros(n_bins, np.int64)

    T = numgen // sampling
    hw = int(prof_halfwidth)
    h_t = np.zeros((T, H), np.int32)
    prof_sum = np.zeros((T, 2, 2 * hw), np.float32)
    prof_cnt = np.zeros((T, 2, 2 * hw), np.int32)
    fitA_t = np.zeros(T, np.float64)
    fitB_t = np.zeros(T, np.float64)
    live_t = np.zeros(T, np.int64)
    samp_frame = np.zeros(T, np.int64)
    n_com = numgen // com_stride if com_stride > 0 else 0
    com_t = np.zeros(n_com, np.int64)
    ben_frame = np.zeros(ben_cap, np.int64)
    ben_x = np.zeros(ben_cap, np.int32)
    ben_y = np.zeros(ben_cap, np.int32)
    ben_side = np.zeros(ben_cap, np.int8)

    t_fix, winner, n_ben, n_s, n_c, n_mut = _run_strip_dfe(
        fitness, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of, rowA,
        numgen, sampling, com_stride, sb, sd, mud, mub, seed,
        np.int64(dfe_kind), np.float64(p1), np.float64(p2), edges,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, cp_fit, cp_lab, cp_hist, mut_hist, arise_cnt,
        h_t, prof_sum, prof_cnt, fitA_t, fitB_t, live_t, samp_frame,
        com_t, ben_frame, ben_x, ben_y, ben_side, hw)

    alive = label[:N] >= 0
    fixed_cnt = (mut_hist[:N][alive].min(axis=0).astype(np.int64)
                 if alive.any() else np.zeros(n_bins, np.int64))

    return {
        "t_fix": int(t_fix), "winner": int(winner),
        "h_t": h_t[:n_s], "samp_frame": samp_frame[:n_s],
        "prof_sum": prof_sum[:n_s], "prof_cnt": prof_cnt[:n_s],
        "fitA": fitA_t[:n_s], "fitB": fitB_t[:n_s], "live": live_t[:n_s],
        "com": com_t[:n_c],
        "ben": dict(frame=ben_frame[:n_ben].copy(), x=ben_x[:n_ben].copy(),
                    y=ben_y[:n_ben].copy(), side=ben_side[:n_ben].copy()),
        "final_fit": fitness[:N].reshape(H, W).copy(),
        "final_label": label[:N].reshape(H, W).copy(),
        "n_mut": int(n_mut),
        "arise_cnt": arise_cnt.copy(),
        "fixed_cnt": fixed_cnt,
        "mut_hist_alive_mean": (mut_hist[:N][alive].mean(axis=0)
                                if alive.any() else np.zeros(n_bins)),
        "dfe": dict(kind=int(dfe_kind), mean=dfe_mean, sigma_ln=sigma_ln,
                    p1=p1, p2=p2, edges=edges.copy(), n_bins=n_bins),
        "params": dict(L_long=L_long, L_short=L_short, numgen=numgen, dt=dt,
                       sb=sb, sd=sd, mud=mud, mub=mub, sampling=sampling,
                       avg=avg, seed=seed, split=split, tilt=tilt,
                       periodic_y=periodic_y, moore=moore,
                       com_stride=com_stride, p_die=p_die, M=int(M)),
    }
