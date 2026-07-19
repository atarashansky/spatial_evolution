"""Open-system spatial Moran kernel (round 8, ongoing-nucleation track).

Same validated per-cell dynamics as evolution_strip / evolution_strip_dfe:
Bernoulli(p_die) deaths per site-frame via alias-geometric gaps, fitness-
weighted neighbour refill drawn from the frozen post-death state, deleterious
mutation counter (rate 0.5*mud per copy, 2 copies per division -> U_d = mud
per division; multiplicative effect x(1-sd)).  Beneficials are off (mub=0).

Extensions for the open-system track:
  * an unlabelled BURN-IN phase (T_burn frames) that lets the resident
    field's fitness heterogeneity develop before any lineage is marked;
  * mode 0  LINEAGE : after burn-in every live cell gets a unique int32
            lineage id; each subsequent birth is tagged as a NEW lineage id
            with probability nu (per birth event); new lineages inherit the
            parent's fitness.  Tracks the coarsening of the lineage-size
            distribution: number of distinct lineages present, survivors of
            the t=0 cohort, max size, sum of squared sizes, heterolabel bond
            density, and log2 size histograms at checkpoint samples;
  * mode 1  MARKER  : after burn-in, all cells are label 0 (resident) and a
            given set of sites (single centre cell or a small disc) is
            relabelled 1 with UNCHANGED fitness; run to absorption
            (marker extinct or fixed) or the frame cap; records the marker
            size on a geometric time grid and first-passage frames to size
            thresholds;
  * mode 2  TWOCLONE: half-domain labels 0/1 (validation against the closed-
            system kernel; with T_burn=0, nu=0 the RNG stream is identical
            to evolution_strip / evolution_strip_dfe(dfe_kind=0)).

Load-axis convention: theta = U_d/s_d = mud/sd (no factor of 2).

ROUND-9 EXTENSION (advantaged nucleated competitor, mode 1 only): the marker
sites may be given a fitness DIFFERENT from the resident value they inherit:
  clean_birth=1 : marker fitness reset to 1.0 (deleterious hit count zero);
  s_b>0         : intrinsic multiplicative driver advantage x(1+s_b), applied
                  to the (inherited or reset) marker fitness; heritable since
                  offspring copy parental fitness.
With clean_birth=0 and s_b=0.0 no new statement executes on the RNG/arith-
metic path, so the kernel is bit-identical to the deposited evolution_open.py.
Seeded, serial, bit-reproducible.  Self-contained (alias helpers inlined) so
numba disk caching works in worker processes.
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


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def _run_open(fitness, label, nbr, n_nbr, sites, x_of, y_of, W, H, L_long,
              T_burn, T_obs, sd, mud, seed,
              d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
              victims, src, cp_fit, cp_lab,
              mode, nu, marker_sites, split, s_b, clean_birth,
              cnt, samp_frames, ts, hist, ck_idx, fp_thr, fp_frame,
              scal, scalf):
    """See module docstring.  Returns nothing; results in output arrays.

    scal (int64): 0 t_abs (frame relative to phase start; -1 none)
                  1 abs_type (0 extinct, 1 fixed, -1 running/none)
                  2 max marker size (mode 1) / -
                  3 n_nucleated (mode 0)
                  4 n_capped   (nucleations refused: id capacity exhausted)
                  5 n_samples written
                  6 live cells at end
                  7 winner label (mode 2) / marker initial size (mode 1)
    scalf (float64): 0 fitness of first marker site at marking
                     1 field mean fitness at marking
                     2 field fitness sd at marking
                     3 max fitness at marking
                     4 assigned fitness of first marker site (post-rule)
                     5 mean inherited (resident) fitness over marker sites
                     6 mean assigned fitness over marker sites
    """
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    kcap = victims.size
    n_samp = samp_frames.size
    id_cap = cnt.size
    n_thr = fp_thr.size
    n_ck = ck_idx.size
    n_hbins = hist.shape[1]

    if mud > 0.0:
        del_ctr = np.int64(_gap(m_prob, m_alias, m_tail, m_log1m))
    else:
        del_ctr = np.int64(1 << 60)

    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    # ---------------------------------------------------------- burn-in
    for f in range(1, T_burn + 1):
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
                cp_fit[j] = fitness[chosen]
                cp_lab[j] = label[chosen]
            for j in range(k):
                s = src[j]
                if s < 0:
                    continue
                cell = victims[j]
                fitness[cell] = cp_fit[j]
                label[cell] = cp_lab[j]
                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else s
                    fitness[tgt] -= fitness[tgt] * np.float32(sd)
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)

    # ------------------------------------------------ phase setup: labels
    for c in range(id_cap):
        cnt[c] = 0
    live = np.int64(0)
    sfit = 0.0
    sfit2 = 0.0
    fmax = 0.0
    for j in range(M):
        i = sites[j]
        if fitness[i] > 0.0:
            live += 1
            fv = np.float64(fitness[i])
            sfit += fv
            sfit2 += fv * fv
            if fv > fmax:
                fmax = fv
    if live > 0:
        mf = sfit / live
        vf = sfit2 / live - mf * mf
        if vf < 0.0:
            vf = 0.0
    else:
        mf = 0.0
        vf = 0.0
    scalf[1] = mf
    scalf[2] = np.sqrt(vf)
    scalf[3] = fmax

    next_id = np.int64(0)
    m0 = np.int64(0)
    if mode == 0:
        # every live cell its own lineage
        for j in range(M):
            i = sites[j]
            if fitness[i] > 0.0:
                label[i] = np.int32(next_id)
                cnt[next_id] = 1
                next_id += 1
            else:
                label[i] = -1
        m0 = next_id                       # size of the initial cohort
    elif mode == 1:
        for j in range(M):
            i = sites[j]
            if fitness[i] > 0.0:
                label[i] = 0
                cnt[0] += 1
            else:
                label[i] = -1
        scalf[0] = np.float64(fitness[marker_sites[0]])
        s_inh = 0.0
        s_asg = 0.0
        n_mk = np.int64(0)
        for j in range(marker_sites.size):
            i = marker_sites[j]
            if fitness[i] > 0.0:
                label[i] = 1
                cnt[0] -= 1
                cnt[1] += 1
                s_inh += np.float64(fitness[i])
                if clean_birth != 0:
                    fitness[i] = np.float32(1.0)
                if s_b != 0.0:
                    fitness[i] = np.float32(fitness[i] * (1.0 + s_b))
                s_asg += np.float64(fitness[i])
                n_mk += 1
        scalf[4] = np.float64(fitness[marker_sites[0]])
        if n_mk > 0:
            scalf[5] = s_inh / n_mk
            scalf[6] = s_asg / n_mk
        next_id = 2
        m0 = cnt[1]
        scal[7] = m0
    else:
        for j in range(M):
            i = sites[j]
            if fitness[i] > 0.0:
                if x_of[i] <= split:
                    label[i] = 0
                    cnt[0] += 1
                else:
                    label[i] = 1
                    cnt[1] += 1
            else:
                label[i] = -1
        next_id = 2
        m0 = 0

    n_nuc = np.int64(0)
    n_cap = np.int64(0)
    samp_i = 0
    ck_i = 0
    thr_i = 0
    max_mark = np.int64(0)
    if mode == 1:
        max_mark = cnt[1]
        # first-passage thresholds already satisfied by the initial size
        while thr_i < n_thr and cnt[1] >= fp_thr[thr_i]:
            fp_frame[thr_i] = 0
            thr_i += 1
    t_abs = np.int64(-1)
    abs_type = np.int64(-1)

    # -------------------------------------------------------- observation
    for f in range(1, T_obs + 1):
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
                    lb = label[i]
                    cnt[lb] -= 1
                    fitness[i] = np.float32(0.0)
                    label[i] = -1
                    live -= 1
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
            for j in range(k):
                s = src[j]
                if s < 0:
                    continue
                cell = victims[j]
                fitness[cell] = cp_fit[j]
                lb = cp_lab[j]
                if nu > 0.0:
                    if np.random.rand() < nu:
                        if next_id < id_cap:
                            lb = np.int32(next_id)
                            next_id += 1
                            n_nuc += 1
                        else:
                            n_cap += 1
                label[cell] = lb
                cnt[lb] += 1
                live += 1
                del_ctr -= 2
                while del_ctr <= 0:
                    tgt = cell if del_ctr == -1 else s
                    fitness[tgt] -= fitness[tgt] * np.float32(sd)
                    del_ctr += _gap(m_prob, m_alias, m_tail, m_log1m)

        # ---- marker bookkeeping (mode 1): max size + first passages
        if mode == 1:
            c1 = cnt[1]
            if c1 > max_mark:
                max_mark = c1
            while thr_i < n_thr and c1 >= fp_thr[thr_i]:
                fp_frame[thr_i] = f
                thr_i += 1

        # ---- sampling
        if samp_i < n_samp and f == samp_frames[samp_i]:
            if mode == 0:
                nal = np.int64(0)
                ns0 = np.int64(0)
                mx = np.int64(0)
                sq = 0.0
                for c in range(next_id):
                    cc = cnt[c]
                    if cc > 0:
                        nal += 1
                        sq += np.float64(cc) * np.float64(cc)
                        if cc > mx:
                            mx = cc
                        if c < m0:
                            ns0 += 1
                # heterolabel bond density (right + down bonds, alive pairs)
                bdiff = np.int64(0)
                btot = np.int64(0)
                for j in range(M):
                    i = sites[j]
                    li = label[i]
                    if li < 0:
                        continue
                    xr = x_of[i] + 1
                    if xr <= L_long:
                        i2 = i + 1
                        l2 = label[i2]
                        if l2 >= 0:
                            btot += 1
                            if l2 != li:
                                bdiff += 1
                    i3 = ((y_of[i] + 1) % H) * W + x_of[i]
                    l3 = label[i3]
                    if l3 >= 0:
                        btot += 1
                        if l3 != li:
                            bdiff += 1
                sfit = 0.0
                sfit2 = 0.0
                nl = np.int64(0)
                for j in range(M):
                    i = sites[j]
                    if fitness[i] > 0.0:
                        fv = np.float64(fitness[i])
                        sfit += fv
                        sfit2 += fv * fv
                        nl += 1
                ts[samp_i, 0] = np.float64(f)
                ts[samp_i, 1] = np.float64(nal)
                ts[samp_i, 2] = np.float64(ns0)
                ts[samp_i, 3] = np.float64(mx)
                ts[samp_i, 4] = sq
                ts[samp_i, 5] = np.float64(nl)
                ts[samp_i, 6] = sfit / max(nl, 1)
                ts[samp_i, 7] = sfit2 / max(nl, 1)
                ts[samp_i, 8] = np.float64(bdiff)
                ts[samp_i, 9] = np.float64(btot)
                # size histogram at checkpoint samples
                if ck_i < n_ck and samp_i == ck_idx[ck_i]:
                    for c in range(next_id):
                        cc = cnt[c]
                        if cc > 0:
                            b = 0
                            v = cc
                            while v > 1 and b < n_hbins - 1:
                                v = v >> 1
                                b += 1
                            hist[ck_i, b] += 1
                    ck_i += 1
            else:
                nl = np.int64(0)
                for j in range(M):
                    i = sites[j]
                    if fitness[i] > 0.0:
                        nl += 1
                ts[samp_i, 0] = np.float64(f)
                ts[samp_i, 1] = np.float64(cnt[1])
                ts[samp_i, 2] = np.float64(cnt[0])
                ts[samp_i, 3] = np.float64(nl)
            samp_i += 1

        # ---- absorption checks
        if mode == 1:
            if cnt[1] == 0:
                t_abs = f
                abs_type = 0
                break
            if cnt[0] == 0:
                t_abs = f
                abs_type = 1
                break
        elif mode == 2:
            if cnt[0] == 0 or cnt[1] == 0:
                t_abs = f
                abs_type = 1
                scal[7] = 0 if cnt[1] == 0 else 1
                break

    scal[0] = t_abs
    scal[1] = abs_type
    scal[2] = max_mark
    scal[3] = n_nuc
    scal[4] = n_cap
    scal[5] = samp_i
    scal[6] = live
    if mode == 0:
        scal[7] = next_id


# ------------------------------------------------------------------ driver
_GEOM_CACHE = {}


def _geom(L_long, L_short, periodic_y, moore):
    key = (L_long, L_short, periodic_y, moore)
    g = _GEOM_CACHE.get(key)
    if g is None:
        nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)
        x_of = (np.arange(N, dtype=np.int32) % W).astype(np.int32)
        y_of = (np.arange(N, dtype=np.int32) // W).astype(np.int32)
        g = (nbr, W, H, N, x_of, y_of)
        _GEOM_CACHE[key] = g
    return g


def evolve_open(mode, L_long=192, L_short=64, T_burn=0, T_obs=1_000_000,
                sd=0.01, mud=0.0, nu=0.0, seed=0, dt=2400.0, avg=1.16e-5,
                periodic_y=True, moore=True, samp_frames=None,
                marker_radius=0, marker_sites=None, split=None,
                ck_idx=None, n_hbins=24, fp_thr=None,
                s_b=0.0, clean_birth=False):
    """Run one open-system realisation.  mode: 0 lineage-nucleation,
    1 marker (single cell / disc), 2 two-clone half-domain (validation).

    Returns a dict of scalars + arrays (see keys below)."""
    nbr, W, H, N, x_of, y_of = _geom(L_long, L_short, periodic_y, moore)

    fitness = np.zeros(N + 1, np.float32)          # +1 dummy site
    label = np.full(N + 1, -1, np.int32)
    y0 = 0 if periodic_y else 1
    y1 = H if periodic_y else H - 1
    xs = x_of[:N]
    ys = y_of[:N]
    interior = (xs >= 1) & (xs <= L_long) & (ys >= y0) & (ys < y1)
    fitness[:N][interior] = 1.0
    label[:N][interior] = 0
    sites = np.where(interior)[0].astype(np.int32)
    M = int(sites.size)

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
    cp_lab = np.empty(kcap, np.int32)

    if samp_frames is None:
        samp_frames = np.zeros(0, np.int64)
    samp_frames = np.ascontiguousarray(samp_frames, np.int64)
    n_samp = samp_frames.size
    n_cols = 10 if mode == 0 else 4
    ts = np.zeros((max(n_samp, 1), n_cols), np.float64)

    if mode == 0:
        # id capacity: initial cohort + generous bound on nucleations
        exp_nuc = T_obs * M * p_die * nu
        id_cap = int(M + exp_nuc + 20.0 * np.sqrt(max(exp_nuc, 1.0)) + 5000)
    else:
        id_cap = 2
    cnt = np.zeros(id_cap, np.int64)

    if marker_sites is None:
        if mode == 1:
            cx = 1 + L_long // 2
            cy = (y0 + y1) // 2
            r = int(marker_radius)
            if r <= 0:
                msites = np.array([cy * W + cx], np.int32)
            else:
                lst = []
                for yy in range(y0, y1):
                    for xx in range(1, L_long + 1):
                        dy = yy - cy
                        dx = xx - cx
                        if dx * dx + dy * dy <= r * r:
                            lst.append(yy * W + xx)
                msites = np.array(lst, np.int32)
        else:
            msites = np.zeros(1, np.int32)
    else:
        msites = np.ascontiguousarray(marker_sites, np.int32)

    if split is None:
        split = L_long // 2

    if ck_idx is None:
        ck_idx = np.full(1, -1, np.int64)
    ck_idx = np.ascontiguousarray(ck_idx, np.int64)
    n_ck = ck_idx.size
    hist = np.zeros((max(n_ck, 1), n_hbins), np.int64)

    if fp_thr is None:
        fp_thr = np.zeros(1, np.int64)
    fp_thr = np.ascontiguousarray(fp_thr, np.int64)
    fp_frame = np.full(max(fp_thr.size, 1), -1, np.int64)

    scal = np.full(8, -1, np.int64)
    scalf = np.full(8, np.nan, np.float64)

    _run_open(fitness, label, nbr, np.int64(nbr.shape[1]), sites, x_of, y_of,
              np.int64(W), np.int64(H), np.int64(L_long),
              np.int64(T_burn), np.int64(T_obs), np.float64(sd),
              np.float64(mud), np.int64(seed),
              d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
              victims, src, cp_fit, cp_lab,
              np.int64(mode), np.float64(nu), msites, np.int64(split),
              np.float64(s_b), np.int64(1 if clean_birth else 0),
              cnt, samp_frames, ts, hist, ck_idx, fp_thr, fp_frame,
              scal, scalf)

    n_s = int(scal[5])
    out = {
        "t_abs": int(scal[0]), "abs_type": int(scal[1]),
        "max_marker": int(scal[2]), "n_nucleated": int(scal[3]),
        "n_capped": int(scal[4]), "n_samples": n_s,
        "live_end": int(scal[6]), "scal7": int(scal[7]),
        "f_center": float(scalf[0]), "field_meanfit_at_mark": float(scalf[1]),
        "field_sdfit_at_mark": float(scalf[2]), "field_maxfit_at_mark": float(scalf[3]),
        "marker_fit_assigned": float(scalf[4]), "marker_fit_inherit_mean": float(scalf[5]),
        "marker_fit_assigned_mean": float(scalf[6]),
        "ts": ts[:n_s].copy(),
        "hist": hist.copy(), "fp_frame": fp_frame.copy(),
        "final_label": label[:N].reshape(H, W).copy(),
        "final_fitness": fitness[:N].reshape(H, W).copy(),
        "id_cap": int(id_cap),
        "params": dict(mode=int(mode), L_long=L_long, L_short=L_short,
                       T_burn=int(T_burn), T_obs=int(T_obs), sd=sd, mud=mud,
                       nu=nu, seed=int(seed), dt=dt, avg=avg,
                       periodic_y=periodic_y, moore=moore, p_die=p_die,
                       M=M, split=int(split), n_marker0=int(msites.size),
                       s_b=float(s_b), clean_birth=bool(clean_birth)),
    }
    return out
