"""sigma_DeltaR kernel: per-clone ratchet click rates on the two-clone strip.

Dynamics are IDENTICAL to strip_census.py / evolution_strip.py: dB Moran, Moore
neighbourhood, alias-geometric per-site Bernoulli deaths (p_die = 1-exp(-avg*dt)),
fitness-proportional refill drawn on the frozen post-death state, deleterious
mutation of source AND daughter after every division (geometric waiting-time
sampler decremented twice per division, rate mud/2 per cell per division),
multiplicative fitness (1-sd)^k with an integer hit count k per cell. mud = 0
gives the neutral arm. Beneficials are off. Seeded, serial, bit-reproducible.

Instrumentation (per sampling frame), for each clone c in {A,B}:
  n_c        cell count (bulk = all live cells of the clone)
  sk_c       sum of hit counts k over bulk cells        -> mean load <k>_c
  kmin_c     minimum hit count over bulk cells          -> strict Muller click clock
  sf_c       sum of fitness over bulk cells             -> mean fitness fbar_c
  nI_c,skI_c,kminI_c  same, restricted to interfacial cells (within iface_w
             columns of the local boundary, resolved column by column)
plus the global minimum kmin_g, the interface height hbar = cntA/L_short at
sampling cadence, and a high-rate interface centre-of-mass series cntA every
com_stride frames (interface COM x-position = cntA/L_short).
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr


@njit(cache=True)
def _run_sdr(fitness, kcnt, label, nbr, n_nbr, sites, x_of, y_of, rowA,
             numgen, sampling, sd, mud, seed,
             d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
             victims, src,
             out_n, out_sk, out_kmin, out_sf, out_nI, out_skI, out_kminI,
             kmin_g_t, hbar_t, samp_frame, com_stride, com_t,
             L_long, iface_w):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = out_n.shape[0]
    n_com = com_t.shape[0]
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
    com_i = 0
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)
    onemsd = np.float32(1.0 - sd)
    BIG = np.int64(1 << 40)

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

        # ---- high-rate interface COM series (cntA; COM x = cntA / L_short)
        if com_stride > 0 and f % com_stride == 0 and com_i < n_com:
            com_t[com_i] = cntA
            com_i += 1

        # ---- full census
        if f % sampling == 0 and samp_i < n_samp:
            hbar = 0.0
            for y in range(L_short):
                hbar += rowA[y]
            hbar_t[samp_i] = hbar / L_short
            nA = np.int64(0)
            nB = np.int64(0)
            skA = np.int64(0)
            skB = np.int64(0)
            sfA = 0.0
            sfB = 0.0
            kmA = BIG
            kmB = BIG
            nAI = np.int64(0)
            nBI = np.int64(0)
            skAI = np.int64(0)
            skBI = np.int64(0)
            kmAI = BIG
            kmBI = BIG
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    continue
                kk = kcnt[i]
                # signed distance to the local interface: A occupies x = 1..rowA[y]
                d = np.int64(x_of[i]) - np.int64(rowA[y_of[i]])
                if lb == 0:
                    nA += 1
                    skA += kk
                    sfA += np.float64(fitness[i])
                    if kk < kmA:
                        kmA = kk
                    if d > -iface_w:
                        nAI += 1
                        skAI += kk
                        if kk < kmAI:
                            kmAI = kk
                else:
                    nB += 1
                    skB += kk
                    sfB += np.float64(fitness[i])
                    if kk < kmB:
                        kmB = kk
                    if d < iface_w + 1:
                        nBI += 1
                        skBI += kk
                        if kk < kmBI:
                            kmBI = kk
            out_n[samp_i, 0] = nA
            out_n[samp_i, 1] = nB
            out_sk[samp_i, 0] = skA
            out_sk[samp_i, 1] = skB
            out_kmin[samp_i, 0] = kmA
            out_kmin[samp_i, 1] = kmB
            out_sf[samp_i, 0] = sfA
            out_sf[samp_i, 1] = sfB
            out_nI[samp_i, 0] = nAI
            out_nI[samp_i, 1] = nBI
            out_skI[samp_i, 0] = skAI
            out_skI[samp_i, 1] = skBI
            out_kminI[samp_i, 0] = kmAI
            out_kminI[samp_i, 1] = kmBI
            kmin_g_t[samp_i] = kmA if kmA < kmB else kmB
            samp_frame[samp_i] = f
            samp_i += 1

        if cntA == 0 or cntB == 0:
            t_fix = f
            winner = 0 if cntB == 0 else 1
            break

    return t_fix, winner, samp_i, com_i


def run_sdr(L_long=192, L_short=32, numgen=6_000_000, dt=2400.0,
            sd=0.01, mud=0.004, sampling=250, avg=1.16e-5, seed=0,
            iface_w=3, periodic_y=True, moore=True, com_stride=0,
            init_lnfB=0.0):
    """One two-clone strip run with per-clone load instrumentation.

    init_lnfB imposes a static log-fitness offset on clone B at t = 0
    (fitness_B(0) = exp(init_lnfB) < 1 for init_lnfB < 0); with mud = 0 the
    imposed gap g = -init_lnfB is conserved exactly (drift-response calibration).

    Returns dict of time series (trimmed to the recorded length), t_fix (-1 =
    censored at the numgen cap), winner, and params.
    """
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
    if init_lnfB != 0.0:
        fB = np.float32(np.exp(init_lnfB))
        idxB = np.where(interior & ~isA)[0]
        fitness[idxB] = fB
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
    out_n = np.zeros((T, 2), np.int64)
    out_sk = np.zeros((T, 2), np.int64)
    out_kmin = np.zeros((T, 2), np.int64)
    out_sf = np.zeros((T, 2), np.float64)
    out_nI = np.zeros((T, 2), np.int64)
    out_skI = np.zeros((T, 2), np.int64)
    out_kminI = np.zeros((T, 2), np.int64)
    kmin_g_t = np.zeros(T, np.int64)
    hbar_t = np.zeros(T, np.float64)
    samp_frame = np.zeros(T, np.int64)
    n_com = numgen // com_stride if com_stride > 0 else 0
    com_t = np.zeros(n_com, np.int64)
    t_fix, winner, n_s, n_c = _run_sdr(
        fitness, kcnt, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of,
        rowA, numgen, sampling, sd, mud, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src,
        out_n, out_sk, out_kmin, out_sf, out_nI, out_skI, out_kminI,
        kmin_g_t, hbar_t, samp_frame, np.int64(com_stride), com_t,
        np.int64(L_long), np.int64(iface_w))
    return dict(
        t_fix=int(t_fix), winner=int(winner),
        n=out_n[:n_s], sk=out_sk[:n_s], kmin=out_kmin[:n_s], sf=out_sf[:n_s],
        nI=out_nI[:n_s], skI=out_skI[:n_s], kminI=out_kminI[:n_s],
        kmin_g=kmin_g_t[:n_s], hbar=hbar_t[:n_s], frame=samp_frame[:n_s],
        com=com_t[:n_c].copy(), com_stride=int(com_stride),
        params=dict(L_long=L_long, L_short=L_short, numgen=numgen, dt=dt,
                    sd=sd, mud=mud, sampling=sampling, avg=avg, seed=seed,
                    iface_w=iface_w, periodic_y=periodic_y, moore=moore,
                    p_die=p_die, M=int(M), init_lnfB=float(init_lnfB)),
    )
