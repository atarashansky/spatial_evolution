"""k-clone strip simulation: multi-clone coarsening under mutational load.

Same spatial Moran dynamics as evolution_strip (validated), but initialized
with k vertical clone stripes instead of two. Tracks per-clone abundances,
per-clone mean fitness, elimination cascade, and global fixation time.
Interface-resolved observables (h_t, profiles) are dropped — with k clones
there are k-1+ interfaces and the two-clone height function is undefined.

Seeded, serial, bit-reproducible.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr


@njit
def _run_kclone(fitness, label, nbr, n_nbr, sites, numgen, sampling,
                sb, sd, mud, mub, seed,
                d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
                victims, src, cp_fit, cp_lab,
                counts, counts_t, fit_t, samp_frame, elim_frame,
                spat, spat_frame, spat_stride):
    np.random.seed(seed)
    M = sites.size
    k = counts.shape[0]
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = counts_t.shape[0]
    kcap = victims.size

    log1m_mub = np.log(1.0 - 0.5 * mub)
    if mub > 0.0:
        ben_ctr = 1 + np.int64(np.log(1.0 - np.random.rand()) / log1m_mub)
    else:
        ben_ctr = np.int64(1 << 60)
    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)

    n_ben = np.int64(0)
    samp_i = 0
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    for f in range(1, numgen + 1):
        # ---- victims (sorted, exact per-site-frame Bernoulli)
        kk = 0
        while pos < M:
            if kk < kcap:
                victims[kk] = sites[pos]
                kk += 1
            pos += _gap(d_prob, d_alias, d_tail, d_log1m)
        pos -= M

        if kk > 0:
            for j in range(kk):
                i = victims[j]
                if fitness[i] > 0.0:
                    counts[label[i]] -= 1
                    fitness[i] = np.float32(0.0)
                    label[i] = -1

            for j in range(kk):
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

            for j in range(kk):
                s = src[j]
                if s < 0:
                    continue
                cell = victims[j]
                fitness[cell] = cp_fit[j]
                lb = cp_lab[j]
                label[cell] = lb
                counts[lb] += 1

                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else s
                    fitness[tgt] -= fitness[tgt] * np.float32(sd)
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)
                ben_ctr -= 2
                while ben_ctr <= 0:
                    tgt = cell if ben_ctr == -1 else s
                    fitness[tgt] += fitness[tgt] * np.float32(sb)
                    n_ben += 1
                    ben_ctr += 1 + np.int64(
                        np.log(1.0 - np.random.rand()) / log1m_mub)

        # ---- eliminations + fixation check (O(k))
        live = np.int64(0)
        mx = np.int64(-1)
        mxc = np.int64(-1)
        for c in range(k):
            cc = counts[c]
            live += cc
            if cc > mx:
                mx = cc
                mxc = c
            if cc == 0 and elim_frame[c] < 0:
                elim_frame[c] = f

        # ---- sampling
        if f % sampling == 0 and samp_i < n_samp:
            # spatial snapshot every spat_stride samples
            if spat_stride > 0 and (samp_i % spat_stride) == 0:
                si = samp_i // spat_stride
                if si < spat.shape[0]:
                    for j in range(label.size - 1):
                        spat[si, j] = label[j]
                    spat_frame[si] = f
            for c in range(k):
                counts_t[samp_i, c] = counts[c]
            # per-clone fitness means
            for c in range(k):
                fit_t[samp_i, c] = 0.0
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb >= 0:
                    fit_t[samp_i, lb] += fitness[i]
            for c in range(k):
                if counts[c] > 0:
                    fit_t[samp_i, c] /= counts[c]
                else:
                    fit_t[samp_i, c] = np.nan
            samp_frame[samp_i] = f
            samp_i += 1

        if mx == live and live > 0:
            t_fix = f
            winner = mxc
            break

    return t_fix, winner, n_ben, samp_i


def evolve_kclone(L_long=256, L_short=64, k=8, numgen=5_000_000, dt=2400.0,
                  sb=0.1, sd=0.01, mud=0.1, mub=0.0, sampling=1000,
                  avg=1.16e-5, seed=0, periodic_y=True, moore=True,
                  spatial_stride=0):
    """k-clone strip run. Returns dict with t_fix (-1 censored), winner,
    counts_t (n_samp, k), fit_t (n_samp, k), elim_frame (k), n_ben, params.
    spatial_stride>0: also return spat (n, H, W) int8 label snapshots taken
    every sampling*spatial_stride frames, plus spat_frame (n,)."""
    nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)

    fitness = np.zeros(N + 1, np.float32)
    label = np.full(N + 1, -1, np.int8)
    x_of = (np.arange(N, dtype=np.int32) % W).astype(np.int32)
    y_of = (np.arange(N, dtype=np.int32) // W).astype(np.int32)

    y0 = 0 if periodic_y else 1
    y1 = H if periodic_y else H - 1
    xs = x_of[:N]
    ys = y_of[:N]
    interior = (xs >= 1) & (xs <= L_long) & (ys >= y0) & (ys < y1)
    # k vertical stripes of equal width
    stripe = np.minimum(((xs - 1) * k) // L_long, k - 1).astype(np.int8)
    fitness[:N][interior] = 1.0
    label[:N][interior] = stripe[interior]

    sites = np.where(interior)[0].astype(np.int32)
    M = sites.size
    counts = np.zeros(k, np.int64)
    for i in sites:
        counts[label[i]] += 1

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

    T = numgen // sampling
    counts_t = np.zeros((T, k), np.int32)
    fit_t = np.zeros((T, k), np.float32)
    samp_frame = np.zeros(T, np.int64)
    elim_frame = np.full(k, -1, np.int64)

    if spatial_stride > 0:
        n_spat = T // spatial_stride + 1
        spat = np.zeros((n_spat, N), np.int8)
        spat_frame = np.full(n_spat, -1, np.int64)
    else:
        spat = np.zeros((1, 1), np.int8)
        spat_frame = np.zeros(1, np.int64)

    t_fix, winner, n_ben, n_s = _run_kclone(
        fitness, label, nbr, np.int64(nbr.shape[1]), sites, numgen, sampling,
        sb, sd, mud, mub, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, cp_fit, cp_lab,
        counts, counts_t, fit_t, samp_frame, elim_frame,
        spat, spat_frame, np.int64(spatial_stride))

    out_spat = None
    out_spat_frame = None
    if spatial_stride > 0:
        n_sp = int(np.sum(spat_frame >= 0))
        out_spat = spat[:n_sp].reshape(n_sp, H, W)
        out_spat_frame = spat_frame[:n_sp]

    return {
        "spat": out_spat, "spat_frame": out_spat_frame,
        "t_fix": int(t_fix), "winner": int(winner), "n_ben": int(n_ben),
        "counts_t": counts_t[:n_s], "fit_t": fit_t[:n_s],
        "samp_frame": samp_frame[:n_s], "elim_frame": elim_frame.copy(),
        "final_label": label[:N].reshape(H, W).copy(),
        "params": dict(L_long=L_long, L_short=L_short, k=k, numgen=numgen,
                       dt=dt, sb=sb, sd=sd, mud=mud, mub=mub,
                       sampling=sampling, avg=avg, seed=seed,
                       periodic_y=periodic_y, moore=moore, p_die=p_die,
                       M=int(M)),
    }
