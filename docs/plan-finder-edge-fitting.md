# Plan — Finder Edge Fitting from Per-Ray m Estimates

## Goal

Given the 3.5m boundary points from `ray-profile.py` cell [7], estimate the 4
edges of a finder pattern, then intersect them to get the pattern's 4 corners.

## Inputs

- `center_xy = (c_col, c_row)` — ROI-local centre of the cluster.
- `m_pos[i]`, `m_neg[i]` — fitted module pitch for each half-ray direction
  (`i = 0 … NUM_RAYS − 1`).  NaN where the fit failed.
- `theta_rad[i]` — ray direction angles, CCW from +x, equally spaced.

For each valid half-ray, compute a boundary point:

    p = center_xy + 3.5 * m * (cos θ, sin θ)

Points are stored in **cyclic order**: walk ray 0 → 35, positive half first,
then negative half (or vice-versa — the key is that the list follows the
circle).  Each point also records its ray index and which half it came from.

## Implementation notes (from first attempt, July 2026)

These notes record what was tried, what worked, and what didn't when the plan
was first implemented.  Use them to skip known dead ends.

### Boundary-point ordering

The plan's original 2N circuit (positive half of each ray, then negative half)
causes a wraparound discontinuity at the junction — two physically adjacent
points on the circle end up far apart in the array.  Instead, use a **1N
circuit**: for each of the N ray directions, pick the single best measurement
(preferring m_pos over m_neg) to get one boundary point per ray direction, in
cyclic order.  This avoids the artificial discontinuity.

### Tangent-angle computation: adaptive `h` shrink

Neighbours can be NaN (failed half-ray fits).  The implemention shrinks the
neighbour offset `h` adaptively to the next valid neighbour in each direction,
up to `h_max` steps.  If none found, the point is skipped (NaN angle).  This
was validated on a perfect square test case.

### DP segmentation

Cost is sum-of-squared-deviations from the segment mean direction vector,
computed on `(cos(2θ), sin(2θ))` lifted to a circle of circumference π (so
parallel but opposite edges map to the same direction).  The linear DP runs
for every possible circular break; the best overall cost wins.  This found
correct 4-segment splits on perfect square data.

### TLS line fit

Standard unweighted TLS (SVD on centred points) worked.  The Huber-weighted
variants (`fit_tls_line_weighted`, `refine_segments`) were implemented but
never validated on real data — the m estimates were too noisy at the time.

### Key bug discovered during work

**`finder_soft_template` transitions** (in `ray-profile.py` cell [5]) were
wrong: transitions at 0.5, 1.5, 2.5 module units instead of 1.5, 2.5, 3.5.
The finder pattern has a **3-module-wide dark centre** (not 1 module).  This
caused the MSE-based per-ray `m` fits to match the template against the wrong
parts of the intensity profile, producing systematically bad m estimates for
all rays.

Fix: transitions at `u=1.5 → 2.5 → 3.5`.

### Why not to merge the first implementation

The edge-fitting module itself (Phases 1–5) looked correct on synthetic
perfect-square data (all tests passed).  But the upstream `m` estimates were
unreliable because of the `finder_soft_template` bug above.  The first
attempt was thrown away to instead validate the `m` fits independently before
re-attempting edge fitting.

### Related files

- `src/qr_reader/detector/edge_fitting.py` — the first implementation, deleted.
- `src/qr_reader/tests/detector/test_edge_fitting.py` — 26 passing tests, deleted.
- `src/qr_reader/scripts/ray-profile.py` cells [8]–[12] — notebook cells that
  ran the edge-fitting pipeline per cluster, deleted.

## Second attempt (July 2026) — manual agglomerative with live TLS

Phases 1–2 (tangent angles + DP) were replaced by agglomerative clustering
directly on (x, y) boundary points.  Each cluster carried a live TLS line
recomputed every merge step.  Only cyclically adjacent clusters could merge.
The distance metric was symmetric mean perpendicular distance.

### Why replaced

The live-TLS approach required recomputing lines at every step and the
adjacency constraint was artificial.  Also, the initial "degeneracy cull" via
σ₂/σ₁ was unnecessary — corner points self-identify through pairwise distance,
not through a fixed window.

## Third attempt (July 2026) — precomputed pairwise σ₂/σ₁ + sklearn

No culling, no live TLS recomputation.  Simpler and at least as effective.

### Core insight

Two boundary points on the same edge of a finder pattern are nearly colinear
with their neighbours — TLS on the 4-point union of two adjacent pairs has
low σ₂/σ₁.  Two points on different edges have high σ₂/σ₁.  Corner points sit
between two edges, so their pairwise σ₂/σ₁ to points on *either* edge is
reasonably low — they can merge into either edge cluster.  Opposite edges
yield very high σ₂/σ₁ and are never merged.

### Algorithm

#### Phase 0 — Pairwise distance matrix

```
Input: N boundary points P[0..N-1] in cyclic (x,y) order.
Output: (N, N) distance matrix.

Initial clusters: for each i, cluster i = {(i, i+1 mod N)} with 2 points.

For each pair (i, j) of initial clusters:
    if cyclic_gap(i, j) > MAX_GAP:
        d[i,j] = 1.0
    else:
        union_pts = P[i] ∪ P[i+1] ∪ P[j] ∪ P[j+1]   (3 or 4 unique points)
        σ₂, σ₁ = singular values of TLS on union_pts
        d[i,j] = σ₂ / σ₁
```

`cyclic_gap(i,j)` is the number of original boundary points between the two
cluster's indices, modulo N.  `MAX_GAP = 1` means we allow one "jumped"
point (e.g. clusters {0,1} and {3,4} are separable by one missing point 2,
so gap = 1 → allowed; {0,1} and {4,5} has gap of 2 → not allowed).

#### Phase 1 — Sklearn agglomerative clustering

```python
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(
    n_clusters=None,
    metric="precomputed",
    linkage="single",
    distance_threshold=DISTANCE_THRESHOLD,  # default 0.1
)
labels = ac.fit_predict(distance_matrix)   # shape (N,), values 0..K-1
```

- **`n_clusters=None`** — let the threshold determine K.
- **`metric="precomputed"`** — use our frozen σ₂/σ₁ matrix.
- **`linkage="single"`** — cluster A merges with B if *any* member-pair of A
  has d ≤ threshold with *any* member-pair of B.  Necessary because some
  pairs within a cluster will have d=1.0 (far apart cyclically — see below).

**Why single linkage is required.**  Consider cluster A that has grown to
span {0,1,2,3} and cluster B that spans {4,5,6,7}.  The pair (3,4) has
low σ₂/σ₁ (both on the same edge).  But pairs (0,6), (1,7), etc. have
d=1.0 because the cyclic gap between them far exceeds MAX_GAP.  Single
linkage picks the *minimum* distance (3,4) and merges.  Complete linkage
would pick the *maximum* (1.0) and never merge.

#### Phase 2 — Extract 4 largest clusters + assign TLS lines

```python
cluster_sizes = [(label, np.sum(labels == label)) for label in set(labels)]
top4 = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)[:4]
top4_labels = [t[0] for t in top4]

# For each of the 4 clusters, fit TLS on its full support set
# (union of all member-pair points)

# Tie-breaking: if a point appears in multiple clusters' support sets,
# assign it to the cluster with lowest σ₂/σ₁ on that cluster's full support.
```

### Output

4 lines (normal, rho, dir), one per cluster, with their support-set points.
These pass into Phase 4 (refinement) unchanged.

### Diagnostic plots

| Plot | Content |
|------|---------|
| 0 | σ₂/σ₁ distance matrix as a heatmap (N×N, cyclic colormap).  Corner points are visible as vertical/horizontal bright bands. |
| 1 | ROI with boundary points, coloured by cluster label after sklearn.  Overlaid TLS lines for the 4 largest clusters. |
| 2 | ROI with tie-broken point-to-edge assignment. |

## Parameters (per-cluster, v3)

```python
MAX_GAP = 1             # Max cyclic index gap between mergeable pairs
DISTANCE_THRESHOLD = 0.1  # σ₂/σ₁ above which pairs are considered incomparable
N_REFINE = 5            # Refinement iterations (Phase 4)
HUBER_DELTA = None      # None → auto-scale as 1.345 * median(|d|)
```

## Remaining phases (4–5, unchanged from plan v1)

Phase 4 (iterative refinement: cut-point reassignment + Huber IRLS) and
Phase 5 (corner intersection) are unchanged from the original plan.  They
will be implemented as future notebook cells after Phase 2.

## Notebook cell structure (v3)

| Cell | Content |
|------|---------|
| [8] | Pairwise `σ₂/σ₁` distance matrix heatmap + initial clusters |
| [9] | Sklearn agglomerative clustering + labelled-points plot |
| [10] | Top-4 extraction + TLS line fits + tie-broken plot |
| (future) | IRLS/Huber refinement loop |
| (future) | Corner intersection |

## Open questions / future

- What to do when fewer than 4 clusters emerge from sklearn?
- How to determine which corner is TL/TR/BR/BL from the raw quadrilateral?
  (Orientation post-processing — not in this plan.)
