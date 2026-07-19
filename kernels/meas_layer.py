"""Measurement-layer forward model for lineage-traced clone areas.

Maps a simulated clone-size configuration (integer cell counts, spatial
adjacency, patch decomposition on the strip) to a distribution of MEASURED
projected YFP clone areas in um^2, as the Colom et al. 2020 pipeline reports
(whole-mount confocal projected area of contiguous YFP patches).

Layer components (each with an explicit, fittable or fixed parameter):

  L1  induction thinning + FUSION.
      The strip was simulated at induction p_sim = 0.02.  The layer thins the
      induced clones to a target per-cell density p_ind by retaining each
      induced label independently with probability p_ind / p_sim, then MERGES
      retained clones that are 8-adjacent on the lattice (a confocal segmenter
      cannot separate touching same-colour clones).  Fusion is the union of
      connected components of the retained-clone adjacency graph; a fused
      "measured clone" is the sum of its members' cells.  p_ind is FIT
      (the labelled fraction / clone density is a per-mouse quantity that
      varies 3-10x between mice in the source data).

  L2  areal projection of a cell-count clone.
      A clone of c basal cells projects to an area  A = c * F  where F is a
      per-clone footprint (um^2 per cell), lognormally distributed
      F ~ LogNormal(ln mu_F, sigma_F^2) to represent (i) the YFP-membrane
      readout marking a variable cortical fraction of a cell, (ii) suprabasal
      /differentiating cell overlay, (iii) clone-to-clone packing variation.
      mu_F, sigma_F are FIT.  (Independent per clone; a shared per-clone
      draw is the natural choice since cells within a clone share phenotype.)

  L3  pixel discretization: areas are rounded to the pixel quantum
      Q = 1.29 um^2 (the observed area granularity: distinct values step by
      1.29 um^2, e.g. 50.27, 51.56, 52.85 ...).  Fixed from data.

  L4  detection floor: measured clones with A < A_min = 50.27 um^2 (the
      hard minimum in every arm at every age) are DROPPED (not censored to
      the floor: the data show a continuous approach to the floor with no
      pile-up, frac at the minimum 0.1-0.7%).  Fixed from data.

FIT protocol: (p_ind, mu_F, sigma_F) are fitted on the 1-month CONTROL arm
only, by minimising the two-sample KS distance between the layer-transformed
neutral-kernel 1-month clone-area distribution and the observed 1-month
areas (in ABSOLUTE um^2, no mean rescaling).  The fitted layer is then applied
UNCHANGED to the 3/6/12-month snapshots for the out-of-sample test.
"""
import numpy as np

PIX = 1.29          # um^2, pixel quantum (from data granularity)
A_MIN = 50.27       # um^2, hard detection floor (min value in every arm)
P_SIM = 0.02        # induction density used in the strip simulations


def _union_find_sizes(cells_kept, keep_mask, adj_i, adj_j, k):
    """Fuse kept clones connected by adjacency; return array of fused sizes
    (cell counts), one per fused patch, for surviving kept clones only."""
    parent = np.arange(k, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if adj_i.size:
        m = keep_mask[adj_i] & keep_mask[adj_j] & (cells_kept[adj_i] > 0) & \
            (cells_kept[adj_j] > 0)
        for a, b in zip(adj_i[m], adj_j[m]):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[rb] = ra
    alive = np.where(keep_mask & (cells_kept > 0))[0]
    if alive.size == 0:
        return np.zeros(0, dtype=np.int64)
    roots = np.array([find(int(x)) for x in alive])
    # sum cells over each root
    order = np.argsort(roots)
    rs = roots[order]
    cs = cells_kept[alive][order].astype(np.int64)
    bnd = np.r_[0, np.where(np.diff(rs) != 0)[0] + 1, rs.size]
    fused = np.add.reduceat(cs, bnd[:-1])
    return fused


def measure(run_tp, k, p_ind, mu_F, sigma_F, rng, floor=A_MIN,
            pix=PIX, fuse=True):
    """Apply the full layer to one run's time-point summary.

    run_tp : dict with keys cells (int32[k]), adj_i, adj_j
    Returns measured areas (um^2) that pass the detection floor.
    """
    cells = run_tp["cells"]
    keep_prob = min(1.0, p_ind / P_SIM)
    keep_mask = rng.random(k) < keep_prob
    if fuse:
        sizes = _union_find_sizes(cells, keep_mask, run_tp["adj_i"],
                                  run_tp["adj_j"], k)
    else:
        alive = np.where(keep_mask & (cells > 0))[0]
        sizes = cells[alive].astype(np.int64)
    if sizes.size == 0:
        return np.zeros(0)
    # L2: lognormal footprint per (fused) clone
    F = rng.lognormal(mean=np.log(mu_F), sigma=sigma_F, size=sizes.size)
    A = sizes * F
    # L3: pixel discretization
    A = np.round(A / pix) * pix
    # L4: detection floor
    return A[A >= floor]


def measure_ensemble(runs, tp, p_ind, mu_F, sigma_F, seed=0, floor=A_MIN,
                     pix=PIX, fuse=True, max_runs=None):
    """Apply the layer to every run in `runs` at time point `tp`; returns the
    pooled measured-area sample (um^2)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, r in enumerate(runs):
        if max_runs is not None and i >= max_runs:
            break
        k = r["k_induced"]
        out.append(measure(r["tp"][tp], k, p_ind, mu_F, sigma_F, rng,
                           floor=floor, pix=pix, fuse=fuse))
    return np.concatenate(out) if out else np.zeros(0)
