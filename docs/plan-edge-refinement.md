# Plan — Segment Refinement via Template-Matching LM

## Goal

Given the 4 initial segment lines from Phase 0–2 of the v3 edge-fitting
pipeline, refine each segment's parameters by minimising the MSE between the
radial intensity profiles and the theoretical finder-pattern template.

## Files involved

| File | Role |
|------|------|
| `src/qr_reader/scripts/ray-profile.py` | Notebook; all visualisation and per-cluster orchestration |
| `src/qr_reader/detector/edge_fitting.py` | Module; `compute_boundary_points`, new refinement functions |
| `src/qr_reader/tests/detector/test_edge_fitting.py` | Tests for the module |

All Jacobian validation, the LM loop, and diagnostic plots live in new notebook
cells in `ray-profile.py`.  The refinement-specific functions (residual,
Jacobian, half-ray assignment, `validate_jacobian`) live in `edge_fitting.py`.

## Status

| Step | Status | Deliverable |
|------|--------|-------------|
| 0 — Bug fix | ✅ done | Notebook runs cleanly on all presets; 36 half-rays; Phase 0–2 produces correct 4 edges |
| 1 — Profile visualisation | ✅ done | Two-panel heatmap (actual vs theoretical) per cluster; templates visually aligned |
| 2 — Residual + Jacobian + check | ✅ done | All 4×N_clusters segments ≤ 9.1e‑4 (well under 1e‑3 threshold); central-difference FD with eps=5e‑6 |
| 3 — LM refinement + plot | ✅ done | Combined plot per cluster: initial (dashed) + refined (solid) lines over ROI; cost 2–5× lower

---
 
## Step 0 — Prerequisite: half-ray model everywhere

### Problem

The codebase historically distinguished "full rays" (diameter lines) from
"half-rays" (centre-to-boundary rays), with separate `m_pos` / `m_neg`
variables, splitting/reversing profiles, and two-part boundary-point
construction.  This duality adds conceptual noise to every function, plot,
and test.

### Fix

A single flat `m` array of length `NUM_RAYS`, one value per half-ray
direction in `[0, 2π)`.  `sample_ray_profiles` returns `(num_rays, num_samples)`
directly — each row is one half-ray sampled outward from the centre.  No
sign change, no splitting, no pos/neg.

### File-by-file changes

#### `ray-profile.py`

| Location | Change |
|----------|--------|
| `NUM_RAYS` (line 19) | `18` → `36`; comment `"# Half-ray count (36 directions in [0, 2π))"` |
| `sample_ray_profiles` | `np.linspace(0, 2*np.pi, ...)`; each profile samples `[0, max_dist]` (not `[-max_dist, max_dist]`); shape `(num_rays, num_samples)` |
| `fit_all_rays` | Returns flat `(m, mse, success)` — no m_pos/m_neg |
| Cell [4] plot | Heatmap: 36 half-ray rows, x-axis = distance (px).  Left panel: half-rays drawn from centre outward. |
| Cell [6] plot | Heatmap: 36 half-ray rows (same shape).  Overlay at 1.5m, 2.5m, 3.5m (all `+ve`).  Fix vertical offset of overlay dots. |
| Cell [7], [8] | `theta_rad = np.linspace(0, 2*np.pi, NUM_RAYS)`.  `compute_boundary_points(center_xy, m, theta_rad)`.  `n_total = NUM_RAYS`. |

#### `edge_fitting.py` — `compute_boundary_points`

```python
def compute_boundary_points(
    center_xy: np.ndarray,
    m: np.ndarray,
    theta_rad: np.ndarray,
    pitch_constant: float = PITCH_CONSTANT,
) -> np.ndarray:
    k = len(theta_rad)
    points = np.full((k, 2), np.nan, dtype=np.float64)
    for i in range(k):
        if np.isfinite(m[i]):
            d = np.array([np.cos(theta_rad[i]), np.sin(theta_rad[i])])
            points[i] = center_xy + pitch_constant * m[i] * d
    return points
```

#### `test_edge_fitting.py` — `TestComputeBoundaryPoints`

```python
class TestComputeBoundaryPoints:
    def test_output_shape(self):
        k = 4
        theta = np.linspace(0, 2*np.pi, k, endpoint=False)
        center = np.array([5.0, 7.0])
        m = np.full(k, 2.0)
        pts = compute_boundary_points(center, m, theta, pitch_constant=3.5)
        assert pts.shape == (k, 2)
        for i in range(k):
            d = np.array([np.cos(theta[i]), np.sin(theta[i])])
            np.testing.assert_allclose(pts[i], center + 3.5 * 2.0 * d, atol=1e-12)

    def test_nan_rows(self):
        k = 4
        theta = np.linspace(0, 2*np.pi, k, endpoint=False)
        m = np.array([1.0, np.nan, 3.0, np.nan])
        pts = compute_boundary_points(np.zeros(2), m, theta)
        assert pts.shape == (4, 2)
        assert np.isfinite(pts[0]).all(); assert np.isnan(pts[1]).all()
        assert np.isfinite(pts[2]).all(); assert np.isnan(pts[3]).all()

    def test_all_nan_when_no_fits(self):
        k = 6
        theta = np.linspace(0, 2*np.pi, k, endpoint=False)
        pts = compute_boundary_points(
            np.zeros(2), np.full(k, np.nan), theta
        )
        assert pts.shape == (6, 2)
        assert np.isnan(pts).all()
```

### Verification

- `pytest src/qr_reader/tests/detector/test_edge_fitting.py -v` — 18 tests pass.
- Notebook easy/medium/hard:
  - Heatmap: 36 rows in cells [4] and [6].
  - Cell [6]: `m_fitted median` printed with `(N/36)` successes.
  - Cells [8–10]: Phase 0–2 works — 4 edges on valid clusters, symmetric
    36×36 distance matrix.
  - No exceptions.

### Deliverable

User visually verifies notebook output for all three presets.

---

## Step 1 — Profile visualisation (actual vs theoretical)

### Purpose

Before writing the LM pipeline, visually confirm that the template computed
from the intersection of each half-ray with its assigned segment matches the
actual intensity profiles.  This validates the projection code and the
assignment logic.

### New notebook cell

Place a new cell **after** cell [10] (end of v3 pipeline).

For each cluster whose `edge_data` entry has `"top4"` (the 4 `EdgeCluster`
objects) and a valid ROI:

1.  Sample the normalised radial profiles (re-use the same sampling code as
    cell [8]: `sample_ray_profiles` → `normalize_roi_intensities` →
    `fit_all_rays`).  The profiles are `profiles_norm` of shape `(18, 239)`.

2.  Construct a **36-element half-ray profile array** and a **36-element
    direction array**.

    The full profile `profiles_norm[i, :]` has 239 samples (index 0..238),
    centre at index 119.  The positive half-ray is `profiles_norm[i, 119:]`
    (120 samples from 0 to `max_dist`).  The negative half-ray is
    `profiles_norm[i, 0:120][::-1]` (120 samples from 0 to `max_dist` going
    the opposite direction).

    ```python
    centre_idx = profiles_norm.shape[1] // 2  # 119
    # Half-ray profiles: interleave m_pos and m_neg halves
    half_profiles = np.zeros((2 * NUM_RAYS, NUM_SAMPLES), dtype=np.float64)
    for i in range(NUM_RAYS):
        half_profiles[i] = profiles_norm[i, centre_idx:]            # direction θ_i
        half_profiles[NUM_RAYS + i] = profiles_norm[i, :centre_idx + 1][::-1]  # θ_i+π

    # Half-ray directions (unit vectors)
    half_dirs = np.zeros((2 * NUM_RAYS, 2), dtype=np.float64)
    for i in range(NUM_RAYS):
        d_pos = np.array([np.cos(theta_rad[i]), np.sin(theta_rad[i])])
        half_dirs[i] = d_pos
        half_dirs[NUM_RAYS + i] = -d_pos  # cos(θ+π) = -cos θ, sin(θ+π) = -sin θ
    ```

3.  Compute the t-samples for the template:

    ```python
    diag_half = 0.5 * np.hypot(W_roi, H_roi)
    max_dist = RAY_LENGTH * diag_half
    t_samples = np.linspace(0, max_dist, NUM_SAMPLES)
    ```

4.  For each of the 4 segments `s`, for each half-ray `i` from 0..35:

    **Assign** half-ray `i` to segment `s` iff `s` has the smallest positive
    intersection distance `t`:

    ```python
    t_s = (seg.rho - seg.normal @ center_xy) / (seg.normal @ half_dirs[i])
    if t_s > 0 and t_s < best_positive_t:
        best_segment = s
        best_t = t_s
    ```

    (Implementation: compute all four t values at once, pick argmin among
    positive ones.)

    **Compute theoretical template** for assigned half-rays:

    ```python
    m = best_t / 3.5
    theoretical = finder_soft_template(t_samples, m, sigma=1.0)
    ```

    Unassigned half-rays (if any) get an all-zero theoretical profile.

5.  Plot two side-by-side heatmaps:

    | Left: Actual | Right: Theoretical |
    |--------------|-------------------|
    | `half_profiles` (36 rows × 120 cols, grayscale) | Theoretical profiles for each half-ray using its assigned segment's m |

    Use `plt.imshow(..., cmap="gray", aspect="auto")` for both.  Annotate
    with the cluster index and number of half-rays assigned to each segment.

### Verification

The left panel shows the familiar 36-row radial profile heatmap (the 18 pos
and 18 neg half-rays interleaved in the display — or stacked in two bands).
The right panel shows the expected template: dark→light→dark→light transitions
at ±1.5m, ±2.5m, ±3.5m from centre.

If the templates look correct (transitions align with the actual profiles),
the projection code is working.

### Deliverable

User visually confirms left/right heatmaps are aligned.  Proceed to Step 2.

---

## Step 2 — Residual function + Jacobian + verification

### Functions to add to `edge_fitting.py`

Add three functions after the existing code (before or after `fit_finder_edges`,
at the end of the file, keeping a section-comment boundary).

All functions use `np.sqrt(2.0 * np.pi)` and `scipy.special.erfc` (which is
already imported via `finder_soft_template` — but `finder_soft_template` is
in `ray-profile.py`, not `edge_fitting.py`.  The new functions must import
`from scipy.special import erfc` locally).

#### 1. `assign_half_rays_to_segments`

```python
def assign_half_rays_to_segments(
    center_xy: np.ndarray,          # (2,) centre in ROI-local coords
    half_dirs: np.ndarray,          # (36, 2) unit direction per half-ray
    segments: list[EdgeCluster],    # 4 segments with .normal, .rho
) -> tuple[np.ndarray, np.ndarray]:
    """Return (segment_idx, intersection_distance) per half-ray.

    segment_idx : ndarray (36,) int
        Index into ``segments`` (0..3); -1 if no segment has positive t.
    t_int : ndarray (36,) float
        Intersection distance from centre; NaN for unassigned half-rays.
    """
```

For each half-ray `i`, compute `t = (seg.rho - seg.normal @ center_xy) / (seg.normal @ d)`
for each segment; pick the smallest positive t.

#### 2. `segment_refinement_residuals`

```python
def segment_refinement_residuals(
    x: np.ndarray,                    # [θ, ρ] for one segment
    center_xy: np.ndarray,            # (2,) centre point
    half_profiles: np.ndarray,        # (36, num_samples) normalised profiles
    half_dirs: np.ndarray,            # (36, 2) unit directions
    t_samples: np.ndarray,            # (num_samples,) pixel distances from centre
    segment_mask: np.ndarray,         # (36,) bool: half-rays assigned to this segment
    pitch_constant: float = 3.5,
    mask_boundary: float = 4.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Residual vector for one segment.

    Returns a 1-D float64 array: for each assigned half-ray, for each
    unmasked sample, ``template(t_j, m) - profile[i, j]``.  Masked samples
    (``|t_j| > mask_boundary * m``) contribute 0.0.
    """
```

Algorithm:

```
θ, ρ = x
n = [cos θ, sin θ]

residuals = []
for i where segment_mask[i]:
    t_int = (ρ - n @ center_xy) / (n @ half_dirs[i])
    m = t_int / pitch_constant
    template = finder_soft_template(t_samples, m, sigma)
    mask = np.abs(t_samples) <= mask_boundary * m
    for j where mask[j]:
        residuals.append(template[j] - half_profiles[i, j])
    for j where not mask[j]:
        residuals.append(0.0)
return np.array(residuals)
```

The residual vector length is `N_assigned * N_samples` where `N_assigned =
np.sum(segment_mask)` and `N_samples = len(t_samples)`.  The unmasked
samples contribute `template - profile`; the masked samples contribute `0.0`.
This fixed-length output is required by `least_squares`.

#### 3. `segment_refinement_jacobian`

```python
def segment_refinement_jacobian(
    x: np.ndarray,                    # [θ, ρ]
    center_xy: np.ndarray,            # (2,)
    half_profiles: np.ndarray,        # (36, NS)
    half_dirs: np.ndarray,            # (36, 2)
    t_samples: np.ndarray,            # (NS,)
    segment_mask: np.ndarray,         # (36,) bool
    pitch_constant: float = 3.5,
    mask_boundary: float = 4.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Jacobian of segment_refinement_residuals w.r.t. x = [θ, ρ].

    Returns an (R, 2) float64 array where R = N_assigned * NS.
    Column 0 = ∂r/∂θ, column 1 = ∂r/∂ρ.
    """
```

Derivation (per half-ray `i`, per sample `j`):

```
θ, ρ = x
n = [cos θ, sin θ]
n_perp = [-sin θ, cos θ]

t_int = (ρ - n@C) / (n@d)
a = pitch_constant * (n @ d)    # denominator of m

m = (ρ - n@C) / a

∂m/∂ρ = 1 / a

∂m/∂θ = [-(n_perp@C)(n@d) - (ρ - n@C)(n_perp@d)] / (pitch_constant * (n@d)²)

template derivative w.r.t. m (per sample j):
    z₁ = -( |t_j| - 1.5*m ) / (σ * √2)
    z₂ = -( |t_j| - 2.5*m ) / (σ * √2)
    z₃ = -( |t_j| - 3.5*m ) / (σ * √2)

    ∂template/∂m = -σ⁻¹ * (1/√(2π)) * [ 1.5·exp(-z₁²) - 2.5·exp(-z₂²) + 3.5·exp(-z₃²) ]

Chain rule:
    ∂r_ij/∂ρ = (∂template/∂m) · (∂m/∂ρ)
    ∂r_ij/∂θ = (∂template/∂m) · (∂m/∂θ)

Masked samples: ∂r_ij/∂(ρ,θ) = 0
```

The Jacobian has the **same row layout** as the residual vector:
row `i*NS + j` corresponds to sample `j` of half-ray `i`.  Both masked and
unmasked rows must appear; masked rows are all zeros.

#### 4. `check_segment_jacobian`

```python
def check_segment_jacobian(
    x0: np.ndarray,                  # [θ, ρ] at which to check
    center_xy: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    t_samples: np.ndarray,
    segment_mask: np.ndarray,
    pitch_constant: float = 3.5,
    sigma: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compare analytical Jacobian to finite-difference approximation.

    Returns (J_analytical, J_fd, max_rel_error).
    """
```

Uses a manual finite-difference loop:

```python
f0 = segment_refinement_residuals(x0, ...)
J = segment_refinement_jacobian(x0, ...)
J_fd = np.zeros_like(J)
for k in range(2):
    h = np.zeros(2)
    h[k] = eps
    f1 = segment_refinement_residuals(x0 + h, ...)
    J_fd[:, k] = (f1 - f0) / eps
```

Then compute relative error per element: `|J[a,b] - J_fd[a,b]| / max(|J[a,b]|, 1e-12)`.
Return the max relative error across all elements.

### New notebook cell

Place after the Step 1 cell.  For each cluster:

1. Reconstruct the same data as Step 1 (`center_xy`, `half_profiles`, `half_dirs`,
   `t_samples`, `top4`).
2. For each of the 4 segments, compute `segment_mask` via
   `assign_half_rays_to_segments`.
3. Form initial `x0 = [atan2(normal[1], normal[0]), rho]` for the segment.
4. Call `check_segment_jacobian` and print:

   ```
   Cluster {ci} segment {si}: max relative Jacobian error = {err:.2e}
   ```

### Verification

All 4 segments per cluster must show `max relative error ≤ 1e-3`.  If any
segment exceeds this, the analytical Jacobian or the residual function is
wrong — fix before proceeding.

### Deliverable

Notebook output shows 4×N_clusters lines of "max relative Jacobian error",
all ≤ 1e-3.  Proceed to Step 3.

---

## Step 3 — LM refinement + combined diagnostic plot

### New notebook cell

Place after the Step 2 cell.  For each cluster:

1. Reconstruct data as in Steps 1–2.
2. For each segment `k = 0..3`:

   ```python
   x0 = np.array([atan2(ec.normal[1], ec.normal[0]), ec.rho])
   result = least_squares(
       fun = segment_refinement_residuals,
       x0 = x0,
       jac = segment_refinement_jacobian,
       method = 'lm',
       args = (center_xy, half_profiles, half_dirs, t_samples, mask_k,
               PITCH_CONSTANT, MASK_BOUNDARY, 1.0),
   )
   theta_opt, rho_opt = result.x
   n_opt = np.array([cos(theta_opt), sin(theta_opt)])
   ```

   Store the refined normal and rho per segment.

   Print: `Segment {k}: (θ={x0[0]:.4f}→{theta_opt:.4f}, ρ={x0[1]:.2f}→{rho_opt:.2f})  cost={result.cost:.4f}`

3. Produce the **combined diagnostic plot** (one figure per cluster):

   - `ax.imshow(roi, cmap="gray")` — ROI background.
   - `ax.plot(center_xy[0], center_xy[1], "r+", markersize=12)` — centre marker.
   - Boundary points from `edge_data[ci]["points"]`, coloured by the new
     half-ray → segment assignment (4 colours, one per segment).

   - **Initial segment lines** (dashed, semi-transparent):
     For each segment `k`, draw the line using
     `ec.normal, ec.rho, ec.direction` from `edge_data[ci]["top4"][k]`.
     Extend the line ±20% beyond the span of the assigned points.
     Style: `"--", color=seg_colors[k], linewidth=1.5, alpha=0.5`.

   - **Refined segment lines** (solid, full colour):
     For each segment `k`, draw the line using the refined `n_opt, rho_opt`.
     Extend the same way.
     Style: `"-", color=seg_colors[k], linewidth=2.5, alpha=0.9`.

   - Legend: "Initial" (one entry, dashed grey) and "Refined" (one entry,
     solid black), plus one entry per segment colour.

   - Title: `"Cluster {ci} — LM refinement"`.

### No re-assignment

The half-ray-to-segment assignment is computed once (from the initial
segments) and held fixed.  The optimisation uses the corresponding
`segment_mask`.

### Verification

For each cluster:
- Refined lines should be visibly better aligned with the finder-pattern
  edges in the ROI than the initial (dashed) lines.
- The shift from initial to refined should be **small** (a few pixels at
  most — this is a refinement, not a global search).
- If any segment's line jumps drastically or goes behind the centre, the
  optimisation diverged — investigate the Jacobian or the initial estimate.
- The cost printed by `least_squares` should be lower than the cost at x0
  (compute initial cost manually with `segment_refinement_residuals` at x0
  and compare `np.sum(residuals**2)` to `result.cost`).

### Deliverable

A combined plot per cluster showing initial (dashed) and refined (solid)
segment lines overlaid on the ROI, with points coloured by segment assignment.
User visually verifies that the refined lines are more accurate.

Once satisfied, the `ray-profile.py` notebook has a complete diagnostic
pipeline from raw image through refined finder-pattern edges.

---

## Parameters (tunable)

```python
NUM_RAYS = 36                # Half-ray count (36 directions in [0, 2π))
TEMPLATE_SIGMA = 1.0         # Smoothing sigma for finder_soft_template
MASK_BOUNDARY = 4.5          # Ignore |t| > 4.5·m in residuals
PITCH_CONSTANT = 3.5         # Finder outer boundary in module units
JACOBIAN_CHECK_EPS = 1e-6    # FD step size for Jacobian verification
JACOBIAN_CHECK_TOL = 1e-3    # Max acceptable rel error in Jacobian
```
