"""Round-8 topology-control kernels: hexagonal and random-coordination
stencils on the 192x32 strip, reusing the PRODUCTION Moran dynamics.

Only the neighbour table changes. The death/birth/mutation event loop is the
unmodified numba kernel `evolution_strip._run_strip` (T_fix runs) and the
unmodified `census2._run_census2` (boundary-contact census); both are generic
in `nbr` (they iterate `for o in range(n_nbr)`), so a stencil is fully
specified by its (N, n_nbr) index table with the dummy site N as filler.

Stencils (all on the standard W = L_long + 2, H = L_short grid, x-walls,
y-periodic; index i <-> (x = i % W, y = i // W); dummy site index N,
fitness 0 forever, exactly as evolution_strip._build_nbr):

  square Moore / vN : evolution_strip._build_nbr (reference, unchanged)

  hexagonal         : triangular lattice by row offsetting ("offset rows"),
                      6 neighbours: (x-1,y), (x+1,y) plus, for even y,
                      (x-1,y-1),(x,y-1),(x-1,y+1),(x,y+1) and for odd y
                      (x,y-1),(x+1,y-1),(x,y+1),(x+1,y+1). L_short must be
                      even so the y-periodic seam alternates parity correctly.
                      Site count 192x32 = 6144, identical to the square strip;
                      the honeycomb site density differs from the square
                      lattice by the sqrt(3)/2 row-height factor but the model
                      is coordination-graph, not metric, so what matters is
                      the 6-neighbour connectivity.

  randcoord         : per-site random subset of the 8 Moore neighbours, size
                      k_i drawn i.i.d. uniform on {5,6,7} (topology-seeded,
                      FIXED for the whole run). The graph is NOT symmetric:
                      a site may draw neighbour j while j did not draw i; the
                      Moran refill needs no symmetry. Wall sites keep their
                      valid-Moore set clipped by the wall (walls remove
                      neighbours before subsetting; if fewer than k valid
                      Moore neighbours exist, all valid ones are kept).

Interface convention for hex/rand runs matches the production protocol:
clone A occupies x <= L_long//2 (isA = interior & x <= split with split =
L_long//2), i.e. A holds columns 1..96, B holds 97..192.
"""
import numpy as np

from evolution_fast4 import build_geom_alias
import evolution_strip as _es
import census2 as _c2

_run_strip = _es._run_strip
_run_census2 = _c2._run_census2
KMAXH = _c2.KMAXH
NCT = _c2.NCT

MOORE_D = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1)]


# --------------------------------------------------------------- stencils
def _grid(L_long, L_short):
    W = L_long + 2
    H = L_short
    N = W * H
    xs = np.arange(N) % W
    ys = np.arange(N) // W
    return W, H, N, xs, ys


def build_nbr_moore(L_long, L_short):
    """Reference Moore (identical to evolution_strip._build_nbr(moore=True))."""
    return _es._build_nbr(L_long, L_short, True, True)


def build_nbr_vn(L_long, L_short):
    return _es._build_nbr(L_long, L_short, True, False)


def build_nbr_hex(L_long, L_short):
    """Hexagonal (triangular-lattice) 6-neighbour stencil via offset rows."""
    assert L_short % 2 == 0, "hex offset-row lattice needs even L_short"
    W, H, N, xs, ys = _grid(L_long, L_short)
    even = (ys % 2) == 0
    nbr = np.full((N, 6), N, np.int32)
    # (dy, dx_even_row, dx_odd_row)
    d = [(0, -1, -1), (0, 1, 1),
         (-1, -1, 0), (-1, 0, 1),
         (1, -1, 0), (1, 0, 1)]
    for o, (dy, dxe, dxo) in enumerate(d):
        dx = np.where(even, dxe, dxo)
        nx = xs + dx
        ny = (ys + dy) % H
        ok = (nx >= 1) & (nx <= L_long)
        idx = ny * W + nx
        nbr[ok, o] = idx[ok]
    return nbr, W, H, N


def build_nbr_rand(L_long, L_short, topo_seed, kmin=5, kmax=7):
    """Randomised-coordination stencil: each site a random subset (size ~U{5..7})
    of its wall-clipped Moore neighbourhood, fixed for the run (topo_seed)."""
    W, H, N, xs, ys = _grid(L_long, L_short)
    rng = np.random.default_rng(topo_seed)
    nbr = np.full((N, kmax), N, np.int32)
    kvec = np.zeros(N, np.int32)
    interior = (xs >= 1) & (xs <= L_long)
    for i in np.where(interior)[0]:
        x, y = xs[i], ys[i]
        cand = []
        for (dy, dx) in MOORE_D:
            nx = x + dx
            ny = (y + dy) % H
            if 1 <= nx <= L_long:
                cand.append(ny * W + nx)
        k = int(rng.integers(kmin, kmax + 1))
        k = min(k, len(cand))
        pick = rng.choice(len(cand), size=k, replace=False)
        for o, p in enumerate(pick):
            nbr[i, o] = cand[p]
        kvec[i] = k
    return nbr, W, H, N, kvec


def get_stencil(name, L_long, L_short, topo_seed=None):
    if name == "moore":
        return build_nbr_moore(L_long, L_short)[0]
    if name == "vn":
        return build_nbr_vn(L_long, L_short)[0]
    if name == "hex":
        return build_nbr_hex(L_long, L_short)[0]
    if name == "rand":
        assert topo_seed is not None
        return build_nbr_rand(L_long, L_short, topo_seed)[0]
    raise ValueError(name)


# --------------------------------------------------------------- T_fix run
def evolve_topo(stencil="hex", L_long=192, L_short=32, numgen=20_000_000,
                dt=2400.0, sb=0.1, sd=0.01, mud=0.1, mub=0.0, sampling=10_000,
                avg=1.16e-5, seed=0, com_stride=0, prof_halfwidth=8,
                ben_cap=1024, topo_seed=None):
    """Two-clone strip run on the given stencil; production dynamics.

    Mirrors evolution_strip.evolve_strip line-for-line except that the
    neighbour table comes from get_stencil(). Returns the same summary
    quantities the production ensemble driver records. Trajectory-level RNG
    identical to evolve_strip: for stencil='moore'/'vn' the T_fix is bit
    identical to evolve_strip at equal seed (acceptance test).
    """
    if stencil == "rand" and topo_seed is None:
        topo_seed = seed + 1_000_000_007  # topology RNG disjoint from run RNG
    nbr = get_stencil(stencil, L_long, L_short, topo_seed)
    W = L_long + 2
    H = L_short
    N = W * H
    fitness = np.zeros(N + 1, np.float32)
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
    h = h_t[:n_s]
    return dict(
        t_fix=int(t_fix), winner=int(winner), n_ben=int(n_ben),
        final_width=float(h[-1].astype(np.float64).std()) if n_s > 0 else np.nan,
        n_samples=int(n_s),
        last_samp_frame=int(samp_frame[n_s - 1]) if n_s > 0 else -1,
        fitA_end=float(fitA_t[n_s - 1]) if n_s > 0 else np.nan,
        fitB_end=float(fitB_t[n_s - 1]) if n_s > 0 else np.nan,
        M=int(M), p_die=float(p_die),
    )


# --------------------------------------------------------------- census run
def census_topo(stencil="hex", L_long=192, L_short=32, numgen=1_200_000,
                dt=2400.0, sd=0.01, mud=0.1, sampling=2000, avg=1.16e-5,
                seed=0, iface_w=3, topo_seed=None):
    """Boundary-contact census on the given stencil (census2 kernel unchanged).

    Boundary cell = live cell with >=1 live opposite-label neighbour IN THE
    RUN'S STENCIL; opp_pc/same_pc/wall_pc are averaged over boundary cells and
    sampling frames, then over runs -> N_eff (opposite-clone contacts) per
    stencil, the interface coordination the ridge law is written in.
    """
    if stencil == "rand" and topo_seed is None:
        topo_seed = seed + 1_000_000_007
    nbr = get_stencil(stencil, L_long, L_short, topo_seed)
    W = L_long + 2
    H = L_short
    N = W * H
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
    return dict(t_fix=int(t_fix), winner=int(winner), ct=ct[:n_s].copy(),
                frame=samp_frame[:n_s].copy(), hbar=hbar_t[:n_s].copy(),
                p_die=float(p_die), M=int(M))


def census_row(res, params, tlo=0.10, thi=0.60):
    """Reduce a census result to per-run boundary-contact means over the
    round-6 time window 0.10 < t/T_fix < 0.60 (per side, then averaged)."""
    ct = res["ct"]
    fr = res["frame"]
    T = res["t_fix"] if res["t_fix"] > 0 else (fr[-1] if fr.size else 1)
    row = dict(params)
    row["t_fix"] = res["t_fix"]
    row["winner"] = res["winner"]
    row["nsamp_total"] = int(fr.size)
    if fr.size == 0:
        row.update(opp_pc=np.nan, same_pc=np.nan, wall_pc=np.nan,
                   nbd_cells=np.nan, nsamp=0)
        return row
    sel = (fr >= tlo * T) & (fr <= thi * T)
    if sel.sum() < 2:
        sel = np.ones_like(fr, dtype=bool)
    c = ct[sel].astype(np.float64)          # (n, 2, NCT)
    nbd = c[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        opp = c[:, :, 2] / np.maximum(nbd, 1)
        same = c[:, :, 1] / np.maximum(nbd, 1)
        wall = c[:, :, 3] / np.maximum(nbd, 1)
    row["opp_pc"] = float(np.nanmean(opp))
    row["same_pc"] = float(np.nanmean(same))
    row["wall_pc"] = float(np.nanmean(wall))
    row["opp_pc_A"] = float(np.nanmean(opp[:, 0]))
    row["opp_pc_B"] = float(np.nanmean(opp[:, 1]))
    row["nbd_cells"] = float(np.nanmean(nbd))
    row["nsamp"] = int(sel.sum())
    return row
