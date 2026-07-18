"""Least-loaded-class census kernel v2 -- adds boundary-contact instrumentation.

Dynamics identical to strip_census.py (dB Moran, Moore, alias-geometric deaths,
mutation of source AND daughter after every division, integer hit count per cell,
fitness = (1-sd)^count). Beneficials off.

NEW (v2), at every sampling frame, for each live cell that has >= 1 live
opposite-label Moore neighbour ("boundary cell"), per side (A=0, B=1):
  nbd      number of boundary cells
  ssame    sum over boundary cells of #same-label live Moore neighbours
  sopp     sum of #opposite-label live Moore neighbours
  swall    sum of #wall/dead neighbours (dummy site or label<0)
  sself0   #boundary cells whose own load equals the global minimum (least-loaded)
  sn0s     sum of #same-label neighbours in the global least-loaded class
  sn0o     sum of #opposite-label neighbours in the least-loaded class
  sabs_nbr #boundary cells with NO least-loaded cell among their 8 neighbours
  sabs_pat #boundary cells with NO least-loaded cell among the 9-cell patch (self+8)
  sabs_opp #boundary cells with NO least-loaded cell among their opposite-label nbrs
  sabs_sam #boundary cells with NO least-loaded cell among their same-label nbrs
  sexload  sum of excess load (kcnt - kmin) of boundary cells
The original interface-band (iface_w) histograms are kept unchanged so that
f0 / n0_col of the deposited census are reproduced by construction.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr

KMAXH = 100  # histogram cap on excess-over-minimum load


@njit(cache=True)
def _run_census2(fitness, kcnt, label, nbr, n_nbr, sites, x_of, y_of, rowA,
                 numgen, sampling, sd, mud, seed,
                 d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
                 victims, src,
                 hI, hB, kmin_t, meanA_t, meanB_t, nAI_t, nBI_t, nA_t, nB_t,
                 fA_t, fB_t, samp_frame, hbar_t, L_long, iface_w, ct):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = hI.shape[0]
    L_short = rowA.shape[0]
    kcap = victims.size

    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)

    cntA = np.int64(0)
    cntB = np.int64(0)
    for j in range(M):
        i = sites[j]
        if label[i] == 0:
            cntA += 1
        elif label[i] == 1:
            cntB += 1

    samp_i = 0
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)
    onemsd = np.float32(1.0 - sd)

    for f in range(1, numgen + 1):
        k = 0
        while pos < M:
            if k < kcap:
                victims[k] = sites[pos]
                k += 1
            pos += _gap(d_prob, d_alias, d_tail, d_log1m)
        pos -= M

        if k > 0:
            for j in range(k):
                i = victims[j]
                if fitness[i] > 0.0:
                    if label[i] == 0:
                        cntA -= 1
                        rowA[y_of[i]] -= 1
                    else:
                        cntB -= 1
                    fitness[i] = np.float32(0.0)
                    label[i] = -1
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
            for j in range(k):
                s = src[j]
                if s < 0:
                    continue
                cell = victims[j]
                fitness[cell] = fitness[s]
                kcnt[cell] = kcnt[s]
                lb = label[s]
                label[cell] = lb
                if lb == 0:
                    cntA += 1
                    rowA[y_of[cell]] += 1
                else:
                    cntB += 1
                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else s
                    fitness[tgt] = fitness[tgt] * onemsd
                    kcnt[tgt] += 1
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)

        if f % sampling == 0 and samp_i < n_samp:
            hbar = 0.0
            for y in range(L_short):
                hbar += rowA[y]
            hbar = hbar / L_short
            hbar_t[samp_i] = hbar
            kmin = np.int64(1 << 40)
            for j in range(M):
                i = sites[j]
                if label[i] >= 0 and kcnt[i] < kmin:
                    kmin = kcnt[i]
            sA = 0.0
            sB = 0.0
            nA = np.int64(0)
            nB = np.int64(0)
            nAI = np.int64(0)
            nBI = np.int64(0)
            sAtot = 0.0
            sBtot = 0.0
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    continue
                d = np.int64(x_of[i]) - np.int64(rowA[y_of[i]])
                near = False
                if lb == 0:
                    if d > -iface_w:
                        near = True
                else:
                    if d < iface_w + 1:
                        near = True
                ex = kcnt[i] - kmin
                if ex >= KMAXH:
                    ex = KMAXH - 1
                if lb == 0:
                    sAtot += np.float64(fitness[i])
                    nA += 1
                    hB[samp_i, 0, ex] += 1
                    if near:
                        hI[samp_i, 0, ex] += 1
                        sA += np.float64(kcnt[i])
                        nAI += 1
                else:
                    sBtot += np.float64(fitness[i])
                    nB += 1
                    hB[samp_i, 1, ex] += 1
                    if near:
                        hI[samp_i, 1, ex] += 1
                        sB += np.float64(kcnt[i])
                        nBI += 1
            kmin_t[samp_i] = kmin
            meanA_t[samp_i] = sA / max(nAI, 1)
            meanB_t[samp_i] = sB / max(nBI, 1)
            nAI_t[samp_i] = nAI
            nBI_t[samp_i] = nBI
            nA_t[samp_i] = nA
            nB_t[samp_i] = nB
            fA_t[samp_i] = sAtot / max(nA, 1)
            fB_t[samp_i] = sBtot / max(nB, 1)
            samp_frame[samp_i] = f

            # ---- v2: boundary-cell Moore contact + local least-loaded census
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    continue
                ns = 0
                no = 0
                nw = 0
                n0s = 0
                n0o = 0
                for o in range(n_nbr):
                    nb = nbr[i, o]
                    l2 = label[nb]
                    if l2 < 0:
                        nw += 1
                        continue
                    if l2 == lb:
                        ns += 1
                        if kcnt[nb] == kmin:
                            n0s += 1
                    else:
                        no += 1
                        if kcnt[nb] == kmin:
                            n0o += 1
                if no > 0:
                    sd_ = lb
                    ct[samp_i, sd_, 0] += 1          # nbd
                    ct[samp_i, sd_, 1] += ns         # ssame
                    ct[samp_i, sd_, 2] += no         # sopp
                    ct[samp_i, sd_, 3] += nw         # swall
                    self0 = 1 if kcnt[i] == kmin else 0
                    ct[samp_i, sd_, 4] += self0      # sself0
                    ct[samp_i, sd_, 5] += n0s        # sn0s
                    ct[samp_i, sd_, 6] += n0o        # sn0o
                    if n0s + n0o == 0:
                        ct[samp_i, sd_, 7] += 1      # sabs_nbr
                    if n0s + n0o + self0 == 0:
                        ct[samp_i, sd_, 8] += 1      # sabs_pat
                    if n0o == 0:
                        ct[samp_i, sd_, 9] += 1      # sabs_opp
                    if n0s == 0:
                        ct[samp_i, sd_, 10] += 1     # sabs_sam
                    ct[samp_i, sd_, 11] += kcnt[i] - kmin   # sexload
            samp_i += 1

        if cntA == 0 or cntB == 0:
            t_fix = f
            winner = 0 if cntB == 0 else 1
            break

    return t_fix, winner, samp_i


NCT = 12  # number of contact-tally channels


def census_strip2(L_long=192, L_short=32, numgen=1_200_000, dt=2400.0,
                  sd=0.01, mud=0.1, sampling=2000, avg=1.16e-5, seed=0,
                  iface_w=3, periodic_y=True, moore=True):
    nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)
    fitness = np.zeros(N + 1, np.float32)
    kcnt = np.zeros(N + 1, np.int64)
    label = np.full(N + 1, -1, np.int8)
    x_of = (np.arange(N, dtype=np.int32) % W).astype(np.int32)
    y_of = (np.arange(N, dtype=np.int32) // W).astype(np.int32)
    xs = x_of[:N]
    ys = y_of[:N]
    interior = (xs >= 1) & (xs <= L_long) & (ys >= 0) & (ys < H)
    split = L_long // 2
    isA = interior & (xs <= split)
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
    T = numgen // sampling
    hI = np.zeros((T, 2, KMAXH), np.int64)
    hB = np.zeros((T, 2, KMAXH), np.int64)
    kmin_t = np.zeros(T, np.int64)
    meanA_t = np.zeros(T, np.float64)
    meanB_t = np.zeros(T, np.float64)
    nAI_t = np.zeros(T, np.int64)
    nBI_t = np.zeros(T, np.int64)
    nA_t = np.zeros(T, np.int64)
    nB_t = np.zeros(T, np.int64)
    fA_t = np.zeros(T, np.float64)
    fB_t = np.zeros(T, np.float64)
    samp_frame = np.zeros(T, np.int64)
    hbar_t = np.zeros(T, np.float64)
    ct = np.zeros((T, 2, NCT), np.int64)
    t_fix, winner, n_s = _run_census2(
        fitness, kcnt, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of,
        rowA, numgen, sampling, sd, mud, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, hI, hB, kmin_t, meanA_t, meanB_t, nAI_t, nBI_t,
        nA_t, nB_t, fA_t, fB_t, samp_frame, hbar_t, np.int64(L_long),
        np.int64(iface_w), ct)
    return dict(t_fix=int(t_fix), winner=int(winner),
                hI=hI[:n_s], hB=hB[:n_s], kmin=kmin_t[:n_s],
                meanA=meanA_t[:n_s], meanB=meanB_t[:n_s],
                nAI=nAI_t[:n_s], nBI=nBI_t[:n_s], nA=nA_t[:n_s], nB=nB_t[:n_s],
                fA=fA_t[:n_s], fB=fB_t[:n_s], frame=samp_frame[:n_s],
                hbar=hbar_t[:n_s], ct=ct[:n_s], p_die=p_die, M=int(M))
