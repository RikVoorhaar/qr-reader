# Plan — Joint Finder-Pattern Refinement via Projective 4-Line Model

## Goal

Replace the per-segment LM refinement (Step 3 in `plan-edge-refinement.md`) with
a joint optimisation over the 8 line parameters `(θ_i, ρ_i)` for i=0..3.  The
refinement uses a geometric template that correctly models the offset between the
ray origin (the [centerpoint]) and the [projective center] of the finder pattern.

Terms in [brackets] are defined in `CONTEXT.md`.

The optimisation jointly refines all 4 boundary lines against the intensity data
from all half-rays, producing finder-pattern edges that form a geometrically
valid quadrilateral while also matching the image intensity.

## Files

| File | Role |
|------|------|
| `src/qr_reader/detector/edge_fitting.py` | All new functions (projective math, template, residual, Jacobian, LM wrapper) |
| `src/qr_reader/tests/detector/test_edge_fitting.py` | Tests for all new functions |
| `src/qr_reader/scripts/ray-profile.py` | Notebook; Phase 4 diagnostic plot in a new cell |

## Status

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 1 — Projective geometry | pending | Functions + tests: homogeneous line constructors, corner computation, projective center, κ factors, line interpolation, ray↔line intersection |
| 2 — Template synthesis | pending | Functions + tests: ray side-determination, per-ray transition distances, smooth template synthesis, pre-computed mask |
| 3 — Residual, Jacobian, FD check | pending | `joint_refinement_residuals`, `joint_refinement_jacobian`, test verifying FD ≤ 1e-3 |
| 4 — LM loop + diagnostic plot | pending | `refine_finder_edges_joint` wrapper, notebook cell with per-cluster initial-vs-refined line plot |

---

## Phase 1 — Projective geometry

### Coordinate reminder

`edge_fitting.py` works in ROI-local **(x, y)** coordinates: x = column, y = row.
The `EdgeCluster` stores ``.normal`` as `(nx, ny)` and ``.rho`` as signed
distance from ROI origin `(0,0)`.

A homogeneous line is `ℓ = [a, b, e]ᵀ` where `a·x + b·y + e = 0`.
From `(θ, ρ)`: `a = cos(θ),  b = sin(θ),  e = -ρ`.

Users of the line system call new functions with the appropriate (×4) data
per finder pattern.

### Line ordering convention

The 4 edge clusters come from `find_four_edges_tls` in an arbitrary order.
The refinement *assumes* a specific canonical assignment:   `lines[0] = LEFT (ℓ_L)``lines[1] = RIGHT (ℓ_R)`, `lines[2] = TOP (ℓ_T)`, `lines[3] = BOTTOM (ℓ_B)`.  Before calling the refinement, the caller must reorder the 4 lines into this convention.

The **caller** (not the refinement functions) is responsible for reordering.
Helper: compute the 4 corners from the 4 lines, pick the corner with the
smallest (x+y) as TL, then assign LEFT/RIGHT/TOP/BOTTOM accordingly.  This
helper lives in the notebook cell of Phase 4.

### 1.1: Homogeneous line constructors

```python
def thetarho_to_homogeneous_line(theta: float, rho: float) -> np.ndarray:
    """Return (3,) homogeneous line vector [a, b, e] such that a·x + b·y + e = 0."""
    return np.array([np.cos(theta), np.sin(theta), -rho], dtype=np.float64)

def homogeneous_line_to_thetarho(ell: np.ndarray) -> tuple[float, float]:
    """Inverse of thetarho_to_homogeneous_line. Returns (theta, rho)."""
    a, b, e = ell
    norm = np.hypot(a, b)
    a, b, e = a / norm, b / norm, e / norm
    theta = np.arctan2(b, a)
    rho = -e
    return theta, rho
```

### 1.2: Corners from 4 lines

Given `ℓ_L, ℓ_R, ℓ_T, ℓ_B` as homogeneous vectors:

```
p_LT = cross(ℓ_L, ℓ_T)
p_RT = cross(ℓ_R, ℓ_T)
p_RB = cross(ℓ_R, ℓ_B)
p_LB = cross(ℓ_L, ℓ_B)
```

Each is a homogeneous point `(x, y, w)`.  Divide by `w` for Euclidean coordinates.

```python
def compute_corners(ell_L, ell_R, ell_T, ell_B) -> tuple[np.ndarray, ...]:
    """Return (p_LT, p_RT, p_RB, p_LB) in Euclidean (x, y) coords."""
```

### 1.3: Projective center

```
c = cross( cross(p_LT, p_RB), cross(p_RT, p_LB) )
```

All cross-products on homogeneous coordinates.  Divide by `w`.

```python
def compute_projective_center(p_LT, p_RT, p_RB, p_LB) -> np.ndarray:
    """Return (2,) Euclidean (x, y) projective center."""
```

### 1.4: Scale factors κ

```
κ_u = - (ℓ_L · c) / (ℓ_R · c)
κ_v = - (ℓ_T · c) / (ℓ_B · c)
```

Where `·` is the dot product: `a*px + b*py + e*1`.

```python
def compute_kappa(ell_L, ell_R, ell_T, ell_B, c) -> tuple[float, float]:
    """Return (kappa_u, kappa_v)."""
```

### 1.5: Projective line interpolation

```
ℓ(α) = (1-α) · ℓ_inner + α · κ · ℓ_outer
```

Valid for any real α (extrapolation is fine — it's a linear combination of homogeneous vectors).

```python
def interpolate_line(alpha: float, ell_inner: np.ndarray, ell_outer: np.ndarray,
                     kappa: float) -> np.ndarray:
    """Return (3,) homogeneous line vector ℓ(α)."""
    return (1.0 - alpha) * ell_inner + alpha * kappa * ell_outer
```

### 1.6: Ray↔line intersection

For a ray `x(s) = origin + s·direction` and a homogeneous line `ℓ = [a, b, e]ᵀ`:

```
s = -(a·origin_x + b·origin_y + e) / (a·dir_x + b·dir_y)
```

If the denominator is zero, the ray is parallel (return NaN or inf).

```python
def ray_line_intersection(origin: np.ndarray, direction: np.ndarray,
                          line: np.ndarray) -> float:
    """Return s ≥ 0 or NaN if parallel/negative.  Works with both
    homogeneous and Euclidean inputs via the formulas above."""
```

### 1.7: Canonical coordinate recovery (inverse mapping)

Given a Euclidean point `x` and the four calibrated lines:

```
u(x) = (ℓ_L · x) / (ℓ_L · x - κ_u · ℓ_R · x)
v(x) = (ℓ_T · x) / (ℓ_T · x - κ_v · ℓ_B · x)
```

This maps any image point to canonical `(u, v)` ∈ [0,1] based on the filepattern quadrilateral.

```python
def canonical_uv(point: np.ndarray, ell_L, ell_R, ell_T, ell_B,
                 kappa_u: float, kappa_v: float) -> tuple[float, float]:
    """Return (u, v) canonical coordinates of *point*."""
```

### Phase 1 tests

- `test_homogeneous_line_roundtrip`: random (θ, ρ) → ℓ → (θ, ρ), assert close.
- `test_corners_square`: perfect axis-aligned square → correct 4 corners.
- `test_projective_center_square`: for a square, center = centroid of corners.
- `test_kappa_square`: for a square centered at origin, κ_u = κ_v = 1.
- `test_line_interpolation_endpoints`: ℓ(0)=ℓ_inner, ℓ(1)~ℓ_outer (up to scale).
- `test_ray_line_intersection`: known geometry → correct s.
- `test_canonical_uv_corners`: corners map to (0,0), (1,0), (1,1), (0,1).

---

## Phase 2 — Template synthesis per ray

### The finder pattern along a ray

The finder pattern is a 7-module square with a 3-module light inner square
(the "ring"), surrounded by a 1-module quiet zone (the QR spec requirement,
which is light).  From the projective center outward, the transitions are:

| Boundary | Canonical α | Transition | Modules from center |
|----------|-------------|------------|---------------------|
| Inner dark→light | α = 3/7 | +1 | 1.5m |
| Inner light→dark | α = 4/7 | -1 | 2.5m |
| Finder edge (dark→quiet zone) | α = 1 | +1 | 3.5m |
| Quiet zone edge (light→outside) | α = 8/7 | -1 | 4.5m |

These module-unit distances are valid *only from the projective center*.  From
the centerpoint (which is offset from the projective center), the distances
are computed geometrically: intersect each boundary line with the half-ray.

### Handling arbitrary centerpoint offset

A ray starts at `centerpoint` (NOT the projective center) in direction `d`.
The projective center `c` is somewhere else.

If the centerpoint is inside the light square (the common case), the ray may
pass through the dark region first (if the ray direction is toward the offset)
or immediately be in the light region (if directed opposite the offset).

The solution: compute the canonical `(u, v)` of the *centerpoint* itself
using `canonical_uv`.  This tells us which region the origin is in.

Then for each side the ray exits, compute the sorted intersection distances
for α ∈ {3/7, 4/7, 1, 8/7}.  Retain only those with α > the origin's canonical
coordinate (if exiting through the x-side) or α > the origin's canonical
coordinate (if exiting through the y-side).

Actually, the robust approach: intersect the ray with ALL 4·4 = 16 candidate
boundaries (4 sides × 4 α values), keep the 4 with smallest positive s, and
build the template from them.  This handles any centerpoint position.

### Implemented approach (robust, Phase 2)

For each half-ray:
1. Intersect the ray from `centerpoint` with direction `d` against all 8 lines:
   ℓ_u(3/7), ℓ_u(4/7), ℓ_u(1.0), ℓ_u(8/7),
   ℓ_v(3/7), ℓ_v(4/7), ℓ_v(1.0), ℓ_v(8/7).
2. Sort the resulting positive intersection distances `s`.
3. Keep the smallest 4 (corresponding to the 4 transitions the ray actually
   encounters).
4. Assign alternating signs: `Δ_j = (+1, -1, +1, -1)` (this assumes the
   pattern dark→light→dark→light, which holds if the centerpoint is within
   the finder pattern — always true in practice).

### Smooth template

For a ray with transition distances `s₁ < s₂ < s₃ < s₄`:

```
T(s) = 0.5·erfc(-(s-s₁)/σ) - 0.5·erfc(-(s-s₂)/σ)
       + 0.5·erfc(-(s-s₃)/σ) - 0.5·erfc(-(s-s₄)/σ)
```

σ is a fixed pixel width (1.0 px default), NOT scaled by module size.

Mask: `s > s₄ + 2·σ` (beyond quiet zone).  This mask is pre-computed from
the initial line estimates and held fixed during optimisation.

### Functions

```python
def compute_transition_distances(
    centerpoint: np.ndarray,
    direction: np.ndarray,
    ell_L, ell_R, ell_T, ell_B,
    kappa_u: float, kappa_v: float,
) -> np.ndarray:
    """Return (4,) float array of sorted transition distances s₁..s₄ for one half-ray."""
```

```python
def synthesize_template(
    s_samples: np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return template values at sample positions s_samples.
    
    s_junctions is (4,) from compute_transition_distances.
    s_samples is (N_S,) distances from centre to each profile sample.
    """
```

```python
def precompute_mask(
    s_samples: np.ndarray,
    s_junctions: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return bool mask: True for samples inside quiet zone, False beyond."""
```

### Phase 2 tests

- `test_transition_distances_square`: axis-aligned square, center = centerpoint → distances match 1.5m, 2.5m, 3.5m, 4.5m.
- `test_transition_distances_offset`: centerpoint offset from center → distances differ from fixed module-unit positions.
- `test_transition_distances_4_transitions`: always returns exactly 4 valid distances.
- `test_template_matches_erfc`: synthesised template at junction points ≈ 0.5.
- `test_mask_beyond_quiet_zone`: mask is False for s > s₄ + buffer.

---

## Phase 3 — Residual, Jacobian, FD verification

### State vector

```
x = [φ₀/R, φ₁/R, φ₂/R, φ₃/R, ρ₀, ρ₁, ρ₂, ρ₃]
```

Where `R` = mean distance from projective center to the 4 corners (pre-computed
once from the initial lines, held fixed).  `θ_i = θ_i^{(0)} + φ_i / R`.

This scaling makes all 8 parameters have units of ∼pixel displacement,
improving LM conditioning.

### 3.1: Residual function

```python
def joint_refinement_residuals(
    x: np.ndarray,               # (8,) [φ₀/R, φ₁/R, φ₂/R, φ₃/R, ρ₀, ρ₁, ρ₂, ρ₃]
    centerpoint: np.ndarray,     # (2,) ray origin in ROI (x, y)
    R: float,                    # quadrilateral radius for parameter scaling
    theta0: np.ndarray,          # (4,) initial θ from EdgeCluster
    half_profiles: np.ndarray,   # (36, N_S) normalised intensity profiles
    half_dirs: np.ndarray,       # (36, 2) unit direction per half-ray
    s_samples: np.ndarray,       # (N_S,) sample distances from centre
    pre_masks: np.ndarray,       # (36, N_S) bool, pre-computed quiet-zone masks
    sigma: float = 1.0,
) -> np.ndarray:
    """Return (36 * N_S,) residual vector.
    
    Algorithm per evaluation:
    1. De-scale x: θ_i = theta0[i] + x[i] * R, ρ_i = x[4+i] for i=0..3.
    2. Build ℓ_L, ℓ_R, ℓ_T, ℓ_B from (θ_i, ρ_i).
    3. Compute corners, projective center c, κ_u, κ_v.
    4. For each half-ray k=0..35:
        a. Compute s₁..s₄ via compute_transition_distances.
        b. Synthesise template T(s).
        c. raw = T(s) - half_profiles[k, :].
        d. mask = pre_masks[k, :]  (fixed, pre-computed).
        e. Fill residual block [k*N_S : (k+1)*N_S]: masked entries = 0,
           unmasked entries = raw.
    5. Solve min ||r_raw - a·r_raw - b||² analytically for global (a, b):
        Actually: let T be the template vector, let P be the profile vector.
        residual = T - P.  Then we want to model observed = a*T + b.
        Equivalent to: minimise ||(a*T + b) - P||².
        Let r = a*T + b - P = a*T + b - profile.
        
        OLS: [ ΣT²    ΣT  ] [a] = [ Σ(T·P) ]
             [ ΣT     N   ] [b]   [ ΣP     ]
        
        Then r = a*T + b - P.
        
        Wait — the sign needs care.  The template T ranges 0→1 (dark→light).
        The actual profile intensity P also ranges 0→1 after normalisation.
        The residual should be template - (a*adjusted + b) where we fit
        a*P + b ≈ T.
        
        Actually: we want to remove brightness/contrast so that only shape
        matters.  Fit: T ≈ a*P + b  (OLS).  Then residual = (a*P + b) - T.
        
        But we already work with normalised profiles (0→1).  So a≈1, b≈0,
        and the OLS just removes any residual bias.
        
        Residual = a·profile + b - template.
        
    6. Return flattened residual.
    """
```

Note on OLS: We solve once per residual evaluation, globally across all
unmasked samples from all rays.  This adds O(1) cost per evaluation.

### 3.2: Jacobian

```python
def joint_refinement_jacobian(
    x: np.ndarray,
    centerpoint: np.ndarray,
    R: float,
    theta0: np.ndarray,
    half_profiles: np.ndarray,
    half_dirs: np.ndarray,
    s_samples: np.ndarray,
    pre_masks: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """Return (R_total, 8) Jacobian matrix.
    
    R_total = 36 * N_S (same layout as residual).

    Chain rule (per sample, per ray, per parameter p):
        ∂r/∂p = ∂template/∂s × ∂s/∂p  (where s is transition distance)

    ∂template/∂s_j for the j-th junction:
        = ±0.5 * (1/σ) * φ((s - s_j)/σ) * (-1)
          where φ is std-normal PDF, and sign depends on the Δ_j
        Actually:
        T(s) = Σ Δ_j * 0.5 * erfc(-(s - s_j)/σ)
        d/ds_j [Δ_j * 0.5 * erfc(-(s - s_j)/σ)]
        = Δ_j * 0.5 * [2/√π * exp(-(s-s_j)²/σ²)] * (1/σ)
        = Δ_j * (1/(σ√π)) * exp(-(s-s_j)²/σ²)

    ∂s_j/∂p requires differentiating ray↔line intersection through:
        p → θ_k,ρ_k → ℓ → (c, κ) → ℓ(α) → s = ray_intx(ℓ(α))

    Derivation for ∂s/∂θ_k and ∂s/∂ρ_k:

    For a line ℓ = [a, b, e] where a=cos(θ), b=sin(θ), e=-ρ:
        ∂a/∂θ = -sin(θ),  ∂b/∂θ = cos(θ),  ∂e/∂θ = 0
        ∂a/∂ρ = 0,        ∂b/∂ρ = 0,        ∂e/∂ρ = -1

    For ℓ_L, ℓ_R, ℓ_T, ℓ_B, parameters map directly:
        ∂ℓ_L/∂x[0] (where x[0] = φ₀/R, θ = theta0[0] + φ₀):
            = R * [-sin(θ₀), cos(θ₀), 0]
        ∂ℓ_L/∂x[4] (where x[4] = ρ₀):
            = [0, 0, -1]

    For an interpolated line ℓ(α) = (1-α)·ℓ_inner + α·κ·ℓ_outer:
        ∂ℓ(α)/∂param = (1-α)·∂ℓ_inner/∂param + α·(∂κ/∂param·ℓ_outer + κ·∂ℓ_outer/∂param)

    ∂κ_u/∂param (where κ_u = -(ℓ_L·c)/(ℓ_R·c)):
        Let dot_L = ℓ_L·c, dot_R = ℓ_R·c.
        ∂κ_u/∂param = -(∂dot_L/∂param · dot_R - dot_L · ∂dot_R/∂param) / dot_R²
        where ∂dot/∂param = ∂ℓ/∂param · c + ℓ · ∂c/∂param

    ∂c/∂param: c = (p_LT × p_RB) × (p_RT × p_LB)
        Each corner p = ℓ_a × ℓ_b.
        ∂p/∂param = ∂ℓ_a/∂param × ℓ_b + ℓ_a × ∂ℓ_b/∂param

    Ray intersection: s = -(ℓ·origin) / (ℓ·direction)
        where ℓ·origin = a·ox + b·oy + e·1   (for origin as (x,y))
        and ℓ·direction = a·dx + b·dy       (direction has no homogeneous component)

        ∂s/∂ℓ_component:
            denom = ℓ·direction
            numer = -(ℓ·origin)
            ∂s/∂a = -ox/denom - numer·dx/denom²
            ∂s/∂b = -oy/denom - numer·dy/denom²
            ∂s/∂e = -1/denom

    Full chain (per junction j, per param p):
        ∂s_j/∂p = Σ_k (∂s_j/∂ℓ_component_k · ∂ℓ_j/∂p)
    
    Then:
        ∂residual_i/∂p = Σ_j (∂T_i/∂s_j · ∂s_j/∂p)
```

This is a significant derivation — the implementation should build it step by
step with intermediary functions, each individually testable:

```python
def _line_deriv_wrt_thetarho(theta: float) -> np.ndarray:
    """Return (3,) ∂ℓ/∂θ."""
    return np.array([-np.sin(theta), np.cos(theta), 0.0])

def _line_deriv_wrt_rho() -> np.ndarray:
    """Return (3,) ∂ℓ/∂ρ."""
    return np.array([0.0, 0.0, -1.0])

def _corner_deriv(p_a: np.ndarray, p_b: np.ndarray,
                  dp_a_dp: np.ndarray, dp_b_dp: np.ndarray) -> np.ndarray:
    """∂(ℓ_a × ℓ_b)/∂param given ∂ℓ_a/∂param and ∂ℓ_b/∂param."""

def _projective_center_deriv(pt_crosses, derivs) -> np.ndarray:
    """Chain through the double-cross for c."""

def _kappa_deriv(...) -> np.ndarray:
    """∂κ/∂param."""

def _interp_line_deriv(alpha, d_ell_inner_dp, d_ell_outer_dp, d_kappa_dp, ...):
    """∂ℓ(α)/∂param."""

def _ray_intx_deriv(d_ell_dp, origin, direction):
    """∂s/∂param given ∂ℓ/∂param."""

def _template_deriv_wrt_junctions(s_sample, s_junctions, sigma):
    """(4,) array: ∂T/∂s_j at each sample position."""
```

The top-level `joint_refinement_jacobian` assembles these.

### 3.3: FD verification (Phase 3 deliverable)

```python
def test_joint_jacobian_vs_fd():
    """FD verification of joint_refinement_jacobian against central differences.
    
    Uses generated test data: synthetic square lines, synthetic profiles.
    Assert max relative error ≤ 1e-3 on entries with |J| > 1e-8.
    """
```

Test function — NOT a notebook cell.  Uses synthetic data so it's fast and
deterministic (no QR image needed).  The test generates:
- 4 lines forming a perfect axis-aligned square at the ROI center
- Half-ray directions, profile samples as constant 0.5 (or a simple ramp)
- Checks the Jacobian against central-difference FD with eps=5e-6

### Phase 3 tests

- `test_joint_jacobian_vs_fd`: FD verification ≤ 1e-3.
- `test_residual_shape`: returns correct-length vector.
- `test_residual_zero_for_perfect_match`: if template = profile, residual ≈ 0.
- `test_jacobian_shape`: returns (R_total, 8).

---

## Phase 4 — LM loop + diagnostic plot

### 4.1: LM wrapper

```python
def refine_finder_edges_joint(
    segments: list[EdgeCluster],     # 4 EdgeCluster objects
    centerpoint: np.ndarray,         # (2,) ray origin
    half_profiles: np.ndarray,       # (36, N_S)
    half_dirs: np.ndarray,           # (36, 2)
    s_samples: np.ndarray,           # (N_S,)
    sigma: float = 1.0,
) -> tuple[list[EdgeCluster], scipy.optimize.OptimizeResult]:
    """Jointly refine 4 finder-pattern edge lines.
    
    Returns (refined_segments, opt_result).
    refined_segments are new EdgeCluster objects with updated .normal and .rho.
    """
```

Algorithm:
1. Extract `theta0, rho0` from the 4 segments.
2. Compute projective center `c`, corners, and `R` = mean(corner_distances).
3. Compute `pre_masks` for each half-ray (frozen per the initial lines).
4. Build initial `x0 = [0,0,0,0, rho0[0], rho0[1], rho0[2], rho0[3]]`.
5. Run:
   ```python
   result = least_squares(
       fun=joint_refinement_residuals,
       x0=x0,
       jac=joint_refinement_jacobian,
       method='lm',
       args=(centerpoint, R, theta0, half_profiles, half_dirs,
             s_samples, pre_masks, sigma),
       xtol=1e-6,
       ftol=1e-6,
       max_nfev=200,
   )
   ```
6. De-scale `result.x` → new `(θ_i, ρ_i)`.
7. Build new `EdgeCluster` objects with updated normals and offsets (copy
   other fields from originals).
8. Return.

### 4.2: Notebook cell (Phase 4 deliverable)

Place a new cell **after the phase 0–2 v3 pipeline cells** in `ray-profile.py`.
This cell **replaces** the per-segment LM refinement cell (cell [13]).

For each cluster with valid `edge_data`:

1. Reorder the 4 `EdgeCluster` objects into LEFT/RIGHT/TOP/BOTTOM order:
   - Compute 4 corners from line intersections.
   - Corner with smallest x+y → TL.  Corner with largest x+y → BR.
   - Use this to determine which pair is the left/right pair (nearly vertical
     normals) and which is top/bottom (nearly horizontal normals).
   - Within each pair, the one with negative x-normal is LEFT, positive is RIGHT;
     the one with negative y-normal is TOP, positive is BOTTOM.

2. Build `half_profiles, half_dirs, s_samples, centerpoint` (same as current
   cells [10]/[11]).

3. Call `refine_finder_edges_joint`.

4. Print:
   ```
   Cluster {ci}: LM converged={success}, cost={cost:.4f}, nfev={nfev}
     L: (θ={θ_L:.4f}, ρ={ρ_L:.2f}) → (θ={θ_L_opt:.4f}, ρ={ρ_L_opt:.2f})
     R: (θ={θ_R:.4f}, ρ={ρ_R:.2f}) → ...
     T: ...
     B: ...
   ```

5. Produce **combined diagnostic plot** (one figure per cluster):
   - `ax.imshow(roi, cmap="gray")` — ROI background.
   - Red `+` markers: centerpoint (large `r+`) and projective center (small `rx`).
   - Boundary points coloured by half-ray assignment (as in cell [13]).
   - **Initial lines** (dashed, semi-transparent, colour-coded by side).
   - **Refined lines** (solid, full colour, colour-coded by side).
   - Legend: L/R/T/B colours, dashed=initial, solid=refined.
   - Title: `"Cluster {ci} — Joint refinement"`.

### Phase 4 verification

- Notebook runs cleanly on easy/medium/hard presets.
- LM converges (not necessarily to 0, but cost ≤ initial cost).
- Refined lines are visibly better aligned than initial lines.
- The shift is small (a few pixels).

---

## Parameters

```python
SIGMA = 1.0               # Template edge softness in pixels (fixed, not scaled)
FD_EPS = 5e-6             # Central-difference step for Jacobian verification
JACOBIAN_TOL = 1e-3       # Max acceptable relative Jacobian error
LM_XTOL = 1e-6            # LM termination on parameter change
LM_FTOL = 1e-6            # LM termination on cost change
LM_MAX_NFEV = 200         # Max LM function evaluations
```
