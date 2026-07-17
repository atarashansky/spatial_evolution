"""Two-clone strip simulation for clonal-interface physics.

Geometry: L_long (x) by L_short (y) strip. Clone A occupies x < split,
clone B the rest. Walls at both x ends; y is periodic (default) or walled
(required for tilted interfaces). Moore (8) or von Neumann (4) neighborhood.

Model per cell = the validated spatial Moran process (see evolution_fast3):
Bernoulli(p_die) deaths per site-frame sampled by alias-geometric gaps,
fitness-weighted neighbor refill from the frozen post-death state,
independent deleterious (0.5*mud, x(1-sd)) and beneficial (0.5*mub, x(1+sb))
mutation of source and daughter after each division.

Instrumentation (the point of this module):
  h(y,t)    interface height fn = per-row A-cell count, O(1)/event upkeep
  profiles  fitness vs signed distance from interface, per side, per sample
  com       high-rate interface center-of-mass series (jackpot ballistics)
  ben log   (frame, x, y, side) of every beneficial mutation
  T_fix     exact fixation frame (early stop), winner id, censoring flag

Seeded, serial, bit-reproducible.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap


# ------------------------------------------------------------------ kernel
@njit
def _run_strip(fitness, label, nbr, n_nbr, sites, x_of, y_of, rowA,
               numgen, sampling, com_stride, sb, sd, mud, mub, seed,
               d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
               victims, src, cp_fit, cp_lab,
               h_t, prof_sum, prof_cnt, fitA_t, fitB_t, live_t, samp_frame,
               com_t, ben_frame, ben_x, ben_y, ben_side, hw):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = h_t.shape[0]
    n_com = com_t.shape[0]
    ben_cap = ben_frame.shape[0]
    L_short = rowA.shape[0]
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

            # ---- refill + mutations
            for j in range(k):
                s = src[j]
                if s < 0:
                    continue
                cell = victims[j]
                fitness[cell] = cp_fit[j]
                lb = cp_lab[j]
                label[cell] = lb
                if lb == 0:
                    cntA += 1
                    rowA[y_of[cell]] += 1
                else:
                    cntB += 1

                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else s
                    fitness[tgt] -= fitness[tgt] * np.float32(sd)
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)
                ben_ctr -= 2
                while ben_ctr <= 0:
                    tgt = cell if ben_ctr == -1 else s
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
                b = d + hw
                if lb == 0:
                    sfA += fi
                    nA += 1
                    if 0 <= b < 2 * hw:
                        prof_sum[samp_i, 0, b] += fi
                        prof_cnt[samp_i, 0, b] += 1
                else:
                    sfB += fi
                    nB += 1
                    if 0 <= b < 2 * hw:
                        prof_sum[samp_i, 1, b] += fi
                        prof_cnt[samp_i, 1, b] += 1
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

    return t_fix, winner, n_ben, samp_i, com_i


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


def evolve_strip(L_long=512, L_short=64, numgen=2_000_000, dt=2400.0,
                 sb=0.1, sd=0.01, mud=0.1, mub=1e-5, sampling=250,
                 avg=1.16e-5, seed=0, split=None, tilt=0.0,
                 periodic_y=True, moore=True, com_stride=0,
                 prof_halfwidth=64, ben_cap=1 << 20):
    """Two-clone strip run. Returns dict with T_fix (-1 = censored), winner,
    h_t, profiles, per-side fitness series, COM series, beneficial log."""
    if tilt != 0.0:
        assert not periodic_y, "tilted interface requires walled y"
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
    # interface line: x = split + tan(tilt) * (y - y_mid)
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

    t_fix, winner, n_ben, n_s, n_c = _run_strip(
        fitness, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of, rowA,
        numgen, sampling, com_stride, sb, sd, mud, mub, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, cp_fit, cp_lab,
        h_t, prof_sum, prof_cnt, fitA_t, fitB_t, live_t, samp_frame,
        com_t, ben_frame, ben_x, ben_y, ben_side, hw)

    return {
        "t_fix": int(t_fix), "winner": int(winner),
        "h_t": h_t[:n_s], "samp_frame": samp_frame[:n_s],
        "prof_sum": prof_sum[:n_s], "prof_cnt": prof_cnt[:n_s],
        "fitA": fitA_t[:n_s], "fitB": fitB_t[:n_s], "live": live_t[:n_s],
        "com": com_t[:n_c],  # A-area; interface COM x = com / L_short
        "ben": dict(frame=ben_frame[:n_ben].copy(), x=ben_x[:n_ben].copy(),
                    y=ben_y[:n_ben].copy(), side=ben_side[:n_ben].copy()),
        "final_fit": fitness[:N].reshape(H, W).copy(),
        "final_label": label[:N].reshape(H, W).copy(),
        "params": dict(L_long=L_long, L_short=L_short, numgen=numgen, dt=dt,
                       sb=sb, sd=sd, mud=mud, mub=mub, sampling=sampling,
                       avg=avg, seed=seed, split=split, tilt=tilt,
                       periodic_y=periodic_y, moore=moore,
                       com_stride=com_stride, p_die=p_die, M=int(M)),
    }
