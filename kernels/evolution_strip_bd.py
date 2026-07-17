"""Two-clone strip simulation under Birth-death (Bd) updating.

Counterpart to evolution_strip.py (death-Birth): there, a uniformly random
death opens a vacancy and neighbors compete fitness-proportionally to fill
it (selection acts locally, after drift). Here, a divider is chosen
fitness-proportionally from the WHOLE tissue and its offspring overwrites a
uniformly random neighbor (selection acts globally, at birth; drift only in
the overwrite direction). Same event rate as the dB kernel: ~Binomial(M,
p_die) division events per frame, so frames are directly comparable.

Implementation: divider drawn by rejection sampling (uniform candidate,
accept w.p. fitness/f_bound; f_bound refreshed by O(M) scan every 256
frames and raised immediately on any beneficial mutation). Offspring
overwrites a uniform random VALID neighbor (walls excluded by redraw).
Events are sequential within a frame (standard Moran; no vacancies exist,
so there is no frozen-state phase). Same alias-geometric mutation streams
and trial-parity semantics as the dB kernel (2 trials/division: parity 0 ->
daughter, 1 -> source). Same instrumentation: h(y,t) via rowA, fitness
profiles vs signed interface distance, per-side fitness, COM series,
beneficial-event log, exact fixation stop.

Seeded, serial, bit-reproducible.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr


@njit
def _run_strip_bd(fitness, label, nbr, n_nbr, sites, x_of, y_of, rowA,
                  numgen, sampling, com_stride, sb, sd, mud, mub, seed,
                  d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
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

    log1m_mub = np.log(1.0 - 0.5 * mub)
    if mub > 0.0:
        ben_ctr = 1 + np.int64(np.log(1.0 - np.random.rand()) / log1m_mub)
    else:
        ben_ctr = np.int64(1 << 60)
    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)

    cntA = np.int64(0)
    cntB = np.int64(0)
    f_bound = np.float32(0.0)
    for j in range(M):
        i = sites[j]
        if label[i] == 0:
            cntA += 1
        elif label[i] == 1:
            cntB += 1
        if fitness[i] > f_bound:
            f_bound = fitness[i]

    n_ben = np.int64(0)
    samp_i = 0
    com_i = 0
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    for f in range(1, numgen + 1):
        # ---- number of division events this frame (same stream as dB deaths)
        k = 0
        while pos < M:
            k += 1
            pos += _gap(d_prob, d_alias, d_tail, d_log1m)
        pos -= M

        # ---- refresh fitness bound periodically (mub=0: max only decays)
        if f % 256 == 0:
            fb = np.float32(0.0)
            for j in range(M):
                fi = fitness[sites[j]]
                if fi > fb:
                    fb = fi
            f_bound = fb

        for _ in range(k):
            # ---- divider: global fitness-proportional (rejection)
            while True:
                j = np.int64(np.random.rand() * M)
                s_idx = sites[j]
                if np.random.rand() * f_bound < fitness[s_idx]:
                    break
            # ---- target: uniform random valid neighbor (redraw walls)
            while True:
                o = np.int64(np.random.rand() * n_nbr)
                cell = nbr[s_idx, o]
                if fitness[cell] > 0.0:
                    break
            # ---- overwrite
            old_lb = label[cell]
            new_lb = label[s_idx]
            if old_lb != new_lb:
                if old_lb == 0:
                    cntA -= 1
                    rowA[y_of[cell]] -= 1
                else:
                    cntB -= 1
                if new_lb == 0:
                    cntA += 1
                    rowA[y_of[cell]] += 1
                else:
                    cntB += 1
                label[cell] = new_lb
            fitness[cell] = fitness[s_idx]

            # ---- mutations: 2 trials/division, parity 0=daughter 1=source
            del_ctr -= 2
            while del_ctr <= 0:
                tgt = cell if del_ctr == -1 else s_idx
                fitness[tgt] -= fitness[tgt] * np.float32(sd)
                del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)
            ben_ctr -= 2
            while ben_ctr <= 0:
                tgt = cell if ben_ctr == -1 else s_idx
                fitness[tgt] += fitness[tgt] * np.float32(sb)
                if fitness[tgt] > f_bound:
                    f_bound = fitness[tgt]
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


def evolve_strip_bd(L_long=512, L_short=64, numgen=2_000_000, dt=2400.0,
                    sb=0.1, sd=0.01, mud=0.1, mub=1e-5, sampling=250,
                    avg=1.16e-5, seed=0, split=None, tilt=0.0,
                    periodic_y=True, moore=True, com_stride=0,
                    prof_halfwidth=64, ben_cap=1 << 20):
    """Bd-updating two-clone strip run. Same return contract as
    evolve_strip: dict with t_fix (-1 censored), winner, h_t, samp_frame,
    prof_sum/prof_cnt, fitA/fitB, live, com, ben, final fields, params."""
    if tilt != 0.0:
        assert not periodic_y, "tilted interface requires walled y"
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

    t_fix, winner, n_ben, n_s, n_c = _run_strip_bd(
        fitness, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of, rowA,
        numgen, sampling, com_stride, sb, sd, mud, mub, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        h_t, prof_sum, prof_cnt, fitA_t, fitB_t, live_t, samp_frame,
        com_t, ben_frame, ben_x, ben_y, ben_side, hw)

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
        "params": dict(L_long=L_long, L_short=L_short, numgen=numgen, dt=dt,
                       sb=sb, sd=sd, mud=mud, mub=mub, sampling=sampling,
                       avg=avg, seed=seed, split=split, tilt=tilt,
                       periodic_y=periodic_y, moore=moore,
                       com_stride=com_stride, p_die=p_die, M=int(M),
                       update_rule="Bd"),
    }
