"""Mixed-rule two-clone strip: dB tissue with heritable dB->Bd rule-switch.

Third mutation channel: at each division, with prob mu_c per trial
(2 trials/division, parity 0 = daughter, 1 = source, same convention as
mud/mub) the target cell heritably switches update phenotype dB -> Bd
("pusher"). Phenotypes:
  dB  (phen=0): reproduces only into vacancies (fitness-proportional refill
      among dB neighbors of a death site, frozen-state draw, as
      evolution_strip.py).
  Bd  (phen=1, pusher): divides on its own clock at rate b*f_i per
      cell-frame (fitness weighs the divider, as the Bd kernel's global
      fitness-proportional draw does); offspring overwrites a uniform
      random valid neighbor (vacancy or occupied, either phenotype).
      Pushers do NOT compete for vacancy refill: their kinetic advantage
      (overwrite) is bought by losing the refill channel, which is what
      makes the kinetic-neutral rate b* emergent and positive.
Both phenotypes die by the same uniform Bernoulli death channel and carry
identical fitness bookkeeping (mud/sd, mub/sb streams shared across both
division types, 2 trials per division).

mu_c=0 with all-dB init reproduces evolution_strip.py BIT-EXACTLY (all
pusher/switch RNG is guarded behind n_push>0 / mu_c>0). All-pusher init
approaches the Bd kernel as b/p_die grows (vacancy fraction ~ p_die/b;
per-division selection identical by construction).

Seeded, serial, bit-reproducible.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr


@njit
def _run_strip_mixed(fitness, label, phen, nbr, n_nbr, n_dummy, sites,
                     x_of, y_of, rowA, push_list, push_where, n_push0,
                     numgen, sampling, com_stride, sb, sd, mud, mub,
                     mu_c, b_rate, stop_push_frac, stop_fix, abs_clock, seed,
                     d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
                     victims, src, cp_fit, cp_lab,
                     h_t, prof_sum, prof_cnt, fitA_t, fitB_t, fitP_t,
                     pushN_t, live_t, samp_frame,
                     com_t, push_com_t, snap_t, snap_frame, snap_stride,
                     ben_frame, ben_x, ben_y, ben_side,
                     swi_frame, swi_x, swi_y, swi_side, hw):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_samp = h_t.shape[0]
    n_com = com_t.shape[0]
    ben_cap = ben_frame.shape[0]
    swi_cap = swi_frame.shape[0]
    L_short = rowA.shape[0]
    kcap = victims.size
    n_push = n_push0

    log1m_mub = np.log(1.0 - 0.5 * mub)
    if mub > 0.0:
        ben_ctr = 1 + np.int64(np.log(1.0 - np.random.rand()) / log1m_mub)
    else:
        ben_ctr = np.int64(1 << 60)
    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)
    log1m_muc = np.log(1.0 - mu_c) if mu_c > 0.0 else 0.0
    if mu_c > 0.0:
        swc_ctr = 1 + np.int64(np.log(1.0 - np.random.rand()) / log1m_muc)
    else:
        swc_ctr = np.int64(1 << 60)

    f_bound = np.float32(0.0)
    cntA = np.int64(0)
    cntB = np.int64(0)
    for j in range(M):
        i = sites[j]
        if label[i] == 0:
            cntA += 1
        elif label[i] == 1:
            cntB += 1
        if fitness[i] > f_bound:
            f_bound = fitness[i]

    n_ben = np.int64(0)
    n_swi = np.int64(0)
    n_swi_tot = np.int64(0)
    samp_i = 0
    com_i = 0
    snap_i = 0
    n_snap = snap_t.shape[0]
    t_fix = np.int64(-1)
    winner = np.int64(-1)
    t_stop_push = np.int64(-1)
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    for f in range(1, numgen + 1):
        # ---- victims of this frame (identical stream to dB kernel)
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
                    if phen[i] == 1:            # dying pusher: delist
                        pj = push_where[i]
                        last = push_list[n_push - 1]
                        push_list[pj] = last
                        push_where[last] = pj
                        push_where[i] = -1
                        phen[i] = 0
                        n_push -= 1

            # ---- source draw on frozen state (dB neighbors only)
            for j in range(k):
                cell = victims[j]
                tot = 0.0
                for o in range(n_nbr):
                    nb = nbr[cell, o]
                    if phen[nb] == 0:
                        tot += np.float64(fitness[nb])
                if tot <= 0.0:
                    src[j] = -1
                    continue
                r = np.random.rand() * tot
                acc = 0.0
                chosen = np.int64(-1)
                for o in range(n_nbr):
                    nb = nbr[cell, o]
                    if phen[nb] != 0:
                        continue
                    acc += np.float64(fitness[nb])
                    if r < acc:
                        chosen = nb
                        break
                if chosen == -1:
                    for o in range(n_nbr - 1, -1, -1):
                        nb = nbr[cell, o]
                        if fitness[nb] > 0.0 and phen[nb] == 0:
                            chosen = nb
                            break
                src[j] = chosen
                cp_fit[j] = fitness[chosen]
                cp_lab[j] = label[chosen]
            # ---- refill + mutations (three channels)
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
                    if phen[tgt] == 1 and fitness[tgt] > f_bound:
                        f_bound = fitness[tgt]
                    if n_ben < ben_cap:
                        ben_frame[n_ben] = f
                        ben_x[n_ben] = x_of[tgt]
                        ben_y[n_ben] = y_of[tgt]
                        ben_side[n_ben] = label[tgt]
                        n_ben += 1
                    ben_ctr += 1 + np.int64(
                        np.log(1.0 - np.random.rand()) / log1m_mub)
                swc_ctr -= 2
                while swc_ctr <= 0:
                    tgt = cell if swc_ctr == -1 else s
                    if phen[tgt] == 0 and fitness[tgt] > 0.0:
                        phen[tgt] = 1
                        push_list[n_push] = tgt
                        push_where[tgt] = n_push
                        n_push += 1
                        n_swi_tot += 1
                        if n_push == 1:
                            # lone founder: bound = own fitness (a stale
                            # tissue-wide max would throttle its clock for
                            # up to 256 frames, differentially by ratchet
                            # depth -> establishment bias)
                            f_bound = fitness[tgt]
                        elif fitness[tgt] > f_bound:
                            f_bound = fitness[tgt]
                        if n_swi < swi_cap:
                            swi_frame[n_swi] = f
                            swi_x[n_swi] = x_of[tgt]
                            swi_y[n_swi] = y_of[tgt]
                            swi_side[n_swi] = label[tgt]
                            n_swi += 1
                    swc_ctr += 1 + np.int64(
                        np.log(1.0 - np.random.rand()) / log1m_muc)

        # ---- pusher divisions (own clock, Poisson thinning on fitness)
        if n_push > 0 and b_rate > 0.0:
            if f % 256 == 0:
                fb = np.float32(0.0)
                for j in range(n_push):
                    fi = fitness[push_list[j]]
                    if fi > fb:
                        fb = fi
                f_bound = fb
            if abs_clock == 1:
                # absolute clock: rate b*f_i per pusher (aggregate rate
                # tracks mean pusher fitness -> load can slow the clock)
                lam = b_rate * np.float64(f_bound) * np.float64(n_push)
            else:
                # relative (Bd-convention): aggregate rate b*n_push,
                # fitness only picks WHICH pusher divides
                lam = b_rate * np.float64(n_push)
            kp = np.random.poisson(lam)
            for _ in range(kp):
                if abs_clock == 1:
                    # thinning: event realized w.p. f_i/f_bound
                    j = np.int64(np.random.rand() * n_push)
                    s_idx = push_list[j]
                    if np.random.rand() * f_bound >= fitness[s_idx]:
                        continue
                else:
                    # rejection redraw: fitness-proportional divider,
                    # event always realized
                    while True:
                        j = np.int64(np.random.rand() * n_push)
                        s_idx = push_list[j]
                        if np.random.rand() * f_bound < fitness[s_idx]:
                            break
                # target: uniform valid neighbor (walls redraw)
                while True:
                    o = np.int64(np.random.rand() * n_nbr)
                    cell = nbr[s_idx, o]
                    if cell != n_dummy:
                        break
                # overwrite (vacancy or occupied, either phenotype)
                old_lb = label[cell]
                new_lb = label[s_idx]
                if old_lb >= 0:
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
                if phen[cell] == 0:              # offspring inherits Bd
                    phen[cell] = 1
                    push_list[n_push] = cell
                    push_where[cell] = n_push
                    n_push += 1

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
                swc_ctr -= 2
                while swc_ctr <= 0:
                    swc_ctr += 1 + np.int64(
                        np.log(1.0 - np.random.rand()) / log1m_muc)
        # ---- high-rate COM series (A-area + pusher count)
        if com_stride > 0 and f % com_stride == 0 and com_i < n_com:
            com_t[com_i] = cntA
            push_com_t[com_i] = n_push
            com_i += 1

        # ---- state snapshots (composite: -1 vacant/wall, 0 A-dB, 1 B-dB,
        #      2 A-pusher, 3 B-pusher)
        if snap_stride > 0 and f % snap_stride == 0 and snap_i < n_snap:
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    snap_t[snap_i, i] = -1
                else:
                    snap_t[snap_i, i] = lb + 2 * phen[i]
            snap_frame[snap_i] = f
            snap_i += 1

        # ---- full sampling
        if f % sampling == 0 and samp_i < n_samp:
            for y in range(L_short):
                h_t[samp_i, y] = rowA[y]
            sfA = 0.0
            sfB = 0.0
            sfP = 0.0
            nA = np.int64(0)
            nB = np.int64(0)
            for j in range(M):
                i = sites[j]
                lb = label[i]
                if lb < 0:
                    continue
                fi = np.float64(fitness[i])
                if phen[i] == 1:
                    sfP += fi
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
            fitP_t[samp_i] = sfP / max(n_push, 1)
            pushN_t[samp_i] = n_push
            live_t[samp_i] = nA + nB
            samp_frame[samp_i] = f
            samp_i += 1

        # ---- pusher takeover stop
        if stop_push_frac > 0.0 and t_stop_push < 0:
            if np.float64(n_push) >= stop_push_frac * np.float64(M):
                t_stop_push = f
                break

        # ---- fixation early stop
        if cntA == 0 or cntB == 0:
            if t_fix < 0:
                t_fix = f
                winner = 0 if cntB == 0 else 1
            if stop_fix == 1:
                break

    return t_fix, winner, n_ben, n_swi, n_swi_tot, n_push, t_stop_push, \
        samp_i, com_i, snap_i


# ------------------------------------------------------------------ driver
def evolve_strip_mixed(L_long=512, L_short=64, numgen=2_000_000, dt=2400.0,
                       sb=0.1, sd=0.01, mud=0.1, mub=1e-5, mu_c=0.0,
                       b_rate=0.0, sampling=250, avg=1.16e-5, seed=0,
                       split=None, tilt=0.0, periodic_y=True, moore=True,
                       com_stride=0, prof_halfwidth=64, ben_cap=1 << 20,
                       swi_cap=1 << 16, pusher_cols=None, pusher_all=False,
                       stop_push_frac=0.0, stop_fix=True, abs_clock=False,
                       snap_stride=0):
    """Mixed dB/Bd strip run.

    mu_c: per-trial dB->Bd switch prob (2 trials/division).
    b_rate: pusher division rate per unit fitness per cell-frame.
    pusher_cols: (x0, x1) 1-based inclusive column range initialized as
        pushers; pusher_all=True makes every cell a pusher at t=0.
    stop_push_frac: stop when pushers reach this fraction of M (0 = off).
    Returns evolve_strip dict + phen fields, switch log, pusher series,
    t_stop_push, n_push_final, n_switch_total.
    """
    if tilt != 0.0:
        assert not periodic_y, "tilted interface requires walled y"
    nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)

    fitness = np.zeros(N + 1, np.float32)
    label = np.full(N + 1, -1, np.int8)
    phen = np.zeros(N + 1, np.int8)
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
    if pusher_all:
        phen[:N][interior] = 1
    elif pusher_cols is not None:
        x0, x1 = pusher_cols
        phen[:N][interior & (xs >= x0) & (xs <= x1)] = 1

    sites = np.where(interior)[0].astype(np.int32)
    M = sites.size
    rowA = np.zeros(H, np.int32)
    for i in sites[label[sites] == 0]:
        rowA[y_of[i]] += 1

    push_list = np.full(M, -1, np.int32)
    push_where = np.full(N + 1, -1, np.int32)
    n_push0 = 0
    for i in sites:
        if phen[i] == 1:
            push_list[n_push0] = i
            push_where[i] = n_push0
            n_push0 += 1

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
    fitP_t = np.zeros(T, np.float64)
    pushN_t = np.zeros(T, np.int64)
    live_t = np.zeros(T, np.int64)
    samp_frame = np.zeros(T, np.int64)
    n_com = numgen // com_stride if com_stride > 0 else 0
    com_t = np.zeros(n_com, np.int64)
    push_com_t = np.zeros(n_com, np.int64)
    n_snap = numgen // snap_stride if snap_stride > 0 else 0
    snap_t = np.full((n_snap, N), np.int8(-1), np.int8)
    snap_frame = np.zeros(n_snap, np.int64)
    ben_frame = np.zeros(ben_cap, np.int64)
    ben_x = np.zeros(ben_cap, np.int32)
    ben_y = np.zeros(ben_cap, np.int32)
    ben_side = np.zeros(ben_cap, np.int8)
    swi_frame = np.zeros(swi_cap, np.int64)
    swi_x = np.zeros(swi_cap, np.int32)
    swi_y = np.zeros(swi_cap, np.int32)
    swi_side = np.zeros(swi_cap, np.int8)

    (t_fix, winner, n_ben, n_swi, n_swi_tot, n_push_final, t_stop_push,
     n_s, n_c, n_sn) = _run_strip_mixed(
        fitness, label, phen, nbr, np.int64(nbr.shape[1]), np.int64(N),
        sites, x_of, y_of, rowA, push_list, push_where, np.int64(n_push0),
        numgen, sampling, com_stride, sb, sd, mud, mub,
        mu_c, b_rate, stop_push_frac, np.int64(1 if stop_fix else 0),
        np.int64(1 if abs_clock else 0), seed, d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, cp_fit, cp_lab,
        h_t, prof_sum, prof_cnt, fitA_t, fitB_t, fitP_t,
        pushN_t, live_t, samp_frame, com_t, push_com_t,
        snap_t, snap_frame, np.int64(snap_stride),
        ben_frame, ben_x, ben_y, ben_side,
        swi_frame, swi_x, swi_y, swi_side, hw)

    return {
        "t_fix": int(t_fix), "winner": int(winner),
        "t_stop_push": int(t_stop_push),
        "n_push_final": int(n_push_final), "n_switch_total": int(n_swi_tot),
        "h_t": h_t[:n_s], "samp_frame": samp_frame[:n_s],
        "prof_sum": prof_sum[:n_s], "prof_cnt": prof_cnt[:n_s],
        "fitA": fitA_t[:n_s], "fitB": fitB_t[:n_s], "fitP": fitP_t[:n_s],
        "pushN": pushN_t[:n_s], "live": live_t[:n_s],
        "com": com_t[:n_c], "push_com": push_com_t[:n_c],
        "snap": snap_t[:n_sn].reshape(n_sn, H, W).copy() if n_sn else None,
        "snap_frame": snap_frame[:n_sn].copy(),
        "ben": dict(frame=ben_frame[:n_ben].copy(), x=ben_x[:n_ben].copy(),
                    y=ben_y[:n_ben].copy(), side=ben_side[:n_ben].copy()),
        "swi": dict(frame=swi_frame[:n_swi].copy(), x=swi_x[:n_swi].copy(),
                    y=swi_y[:n_swi].copy(), side=swi_side[:n_swi].copy()),
        "final_fit": fitness[:N].reshape(H, W).copy(),
        "final_label": label[:N].reshape(H, W).copy(),
        "final_phen": phen[:N].reshape(H, W).copy(),
        "params": dict(L_long=L_long, L_short=L_short, numgen=numgen, dt=dt,
                       sb=sb, sd=sd, mud=mud, mub=mub, mu_c=mu_c,
                       b_rate=b_rate, sampling=sampling, avg=avg, seed=seed,
                       split=split, tilt=tilt, periodic_y=periodic_y,
                       moore=moore, com_stride=com_stride, p_die=p_die,
                       M=int(M), n_push0=int(n_push0),
                       stop_push_frac=stop_push_frac, stop_fix=stop_fix,
                       abs_clock=abs_clock,
                       update_rule="mixed"),
    }
