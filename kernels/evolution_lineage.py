"""Lineage-tracing strip simulation: clone-size distributions under mutational load.

Same spatial Moran dynamics as evolution_strip/evolution_kclone (validated kernels
from the clonal-interface paper), but initialized as a lineage-tracing experiment:
every cell in a sparse random subset (induction fraction p_ind) gets a unique
int32 label, all other cells are unlabeled background (label -1 but alive).
Tracks per-label clone sizes over time via periodic label-count snapshots.

Neutral null: mud=0 (pure spatial drift -> Klein/Simons exponential scaling).
Loaded run:  mud>0, sd>0 (deleterious load; fitness heterogeneity self-generates
             interface effects).

Seeded, serial, bit-reproducible.
"""
import numpy as np
from numba import njit
from evolution_fast4 import build_geom_alias, _gap
from evolution_strip import _build_nbr


@njit
def _run_lineage(fitness, label, nbr, n_nbr, sites, numgen, sampling,
                 sb, sd, mud, mub, seed,
                 d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
                 victims, src, cp_fit, cp_lab,
                 label_snap, snap_frame):
    np.random.seed(seed)
    M = sites.size
    d_tail = d_prob.shape[0] - 1
    m_tail = m_prob.shape[0] - 1
    n_snap = label_snap.shape[0]

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
    snap_i = 0
    pos = np.int64(_gap(d_prob, d_alias, d_tail, d_log1m) - 1)

    for f in range(1, numgen + 1):
        kk = 0
        while pos < M:
            if kk < victims.size:
                victims[kk] = sites[pos]
                kk += 1
            pos += _gap(d_prob, d_alias, d_tail, d_log1m)
        pos -= M

        if kk > 0:
            for j in range(kk):
                i = victims[j]
                if fitness[i] > 0.0:
                    fitness[i] = np.float32(0.0)
                    label[i] = -2  # dead marker (background alive = -1)

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
                label[cell] = cp_lab[j]

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

        if f % sampling == 0 and snap_i < n_snap:
            for j in range(label.size - 1):
                label_snap[snap_i, j] = label[j]
            snap_frame[snap_i] = f
            snap_i += 1

    return n_ben, snap_i


def evolve_lineage(L_long=512, L_short=256, p_ind=0.02, numgen=2_000_000,
                   dt=2400.0, sb=0.0, sd=0.01, mud=0.1, mub=0.0,
                   sampling=50_000, avg=1.16e-5, seed=0,
                   periodic_y=True, moore=True):
    """Lineage-tracing run. Every induced cell gets a unique int32 label at t=0.
    Returns dict with label snapshots (n_snap, H, W), snapshot frames, params."""
    nbr, W, H, N = _build_nbr(L_long, L_short, periodic_y, moore)

    fitness = np.zeros(N + 1, np.float32)
    label = np.full(N + 1, -2, np.int32)   # -2 = dead/border
    x_of = (np.arange(N, dtype=np.int32) % W).astype(np.int32)
    y_of = (np.arange(N, dtype=np.int32) // W).astype(np.int32)

    y0 = 0 if periodic_y else 1
    y1 = H if periodic_y else H - 1
    xs = x_of[:N]
    ys = y_of[:N]
    interior = (xs >= 1) & (xs <= L_long) & (ys >= y0) & (ys < y1)
    fitness[:N][interior] = 1.0
    label[:N][interior] = -1               # alive, unlabeled background

    sites = np.where(interior)[0].astype(np.int32)
    M = sites.size

    rng = np.random.default_rng(seed + 777)
    ind_mask = rng.random(M) < p_ind
    ind_sites = sites[ind_mask]
    for c, i in enumerate(ind_sites):
        label[i] = c
    k = ind_sites.size

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

    n_snap = numgen // sampling
    label_snap = np.full((n_snap, N), -2, np.int32)
    snap_frame = np.zeros(n_snap, np.int64)

    n_ben, n_s = _run_lineage(
        fitness, label, nbr, np.int64(nbr.shape[1]), sites, numgen, sampling,
        sb, sd, mud, mub, seed,
        d_prob, d_alias, d_log1m, m_prob, m_alias, m_log1m,
        victims, src, cp_fit, cp_lab,
        label_snap, snap_frame)

    return {
        "label_snap": label_snap[:n_s].reshape(n_s, H, W),
        "snap_frame": snap_frame[:n_s],
        "k_induced": int(k), "n_ben": int(n_ben),
        "final_fitness": fitness[:N].reshape(H, W).copy(),
        "params": dict(L_long=L_long, L_short=L_short, p_ind=p_ind,
                       numgen=numgen, dt=dt, sb=sb, sd=sd, mud=mud, mub=mub,
                       sampling=sampling, avg=avg, seed=seed,
                       periodic_y=periodic_y, moore=moore,
                       p_die=float(p_die), M=int(M)),
    }


def clone_sizes_from_snaps(res):
    """Per-snapshot clone-size arrays (surviving labeled clones only)."""
    out = []
    k = res["k_induced"]
    for s in range(res["label_snap"].shape[0]):
        lab = res["label_snap"][s].ravel()
        lab = lab[lab >= 0]
        sizes = np.bincount(lab, minlength=k)
        out.append(sizes[sizes > 0])
    return out
