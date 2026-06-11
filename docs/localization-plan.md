# QR Code Localization & Decoding — Implementation Plan

This plan turns the algorithm description into a concrete, ordered implementation
that plugs into the existing pipeline. The end state is: `dev3.py` runs the full
chain, prints the **inferred version**, draws the **4 detected QR corners**, and
hands those corners to **OpenCV** to read the payload.

---

## 0. Context: what already exists

Existing modules (`src/qr_reader/`):

- `qr_gen.py` — synthetic image generation + `binarize_image`. Default test image is **version 1 (N=21)**.
- `alignment.py`, `clustering.py`, `region.py`, `corner.py` — finder-pattern *detection* (scanlines → clusters → flood fill → boundary → angular-NMS corners).
- `geometry.py` — `angular_distance`, `point_line_distance`, `max_offset`, `segments_intersect`, `polygon_area`.
- `finder_pattern.py` — `FinderPattern` (only `outer_corners` today), `extract_finder_patterns`, `check_association`, `find_all_associations`, `find_triplets` → `Triplet(top_left_idx, top_right_idx, bottom_left_idx)`.
- `dev3.py` — the driver script (cell-based, `# %%`), currently ends right after `find_triplets`.

### Conventions that the new code MUST respect

1. **Corner storage is `(row, col)` = `(y, x)`.** Every corner array produced by
   the detection pipeline (`comp_arr[idx]`) is in `(y, x)` order. `dev3.py` plots
   `corners[:, 1]` (x) vs `corners[:, 0]` (y). New code keeps `(row, col)`
   internally and only converts to `(x, y)` at the OpenCV boundary.
2. **Indices vs cluster ids.** `Triplet` and `Association` reference
   `cluster_idx` (the `FinderPattern.cluster_idx`), not list positions. Look
   patterns up by `cluster_idx`.
3. **Angular-NMS corners are unordered** (sorted by angle around the centroid).
   They are NOT yet in canonical `0=TL,1=BL,2=BR,3=TR` order — Step 2 fixes this.

### Two squares per finder pattern

The per-cluster flood fill already produces **both** squares we need, and both are
returned in `all_corners`:

- the **outer square** = largest-area quad → the 7×7 outer edge, canonical module coords `0` and `7` (the algorithm's A/C/E),
- the **inner square** = second-largest-area quad → the **white-ring boundary at offset 1**, canonical module coords `1` and `6` (the algorithm's B/D/F).

Important: the flood fill gives the white ring at **offset 1** (coords `1/6`), not
the black center square at offset 2 (coords `2/5`). So the canonical inner-square
coordinates are `1` and `6`, and the expected inner cross-ratio is computed from
positions `1, 6, N-6, N-1`.

> Optional (not required): a second flood fill of the black center block would
> yield 4 more landmarks at offset 2 (coords `2/5`). Nice to have for extra
> redundancy, but the plan does not rely on it.

---

## Module plan (what goes where)

| Module | Responsibility | New / Modified |
|---|---|---|
| `finder_pattern.py` | Add `inner_corners` to `FinderPattern`; extend `extract_finder_patterns` to keep top-2 quads by area. | Modified |
| `landmarks.py` | Corner ordering, named image landmarks (A0..F3), canonical grid coords, the 8 colinear quadruples, the 24 correspondences. | New |
| `version.py` | SVD line-fit cross-ratio, expected cross-ratio, constraint filtering, version estimation. | New |
| `homography.py` | Normalization, DLT, projection, RANSAC, LM refinement, QR corner projection. | New |
| `decode.py` | Thin OpenCV wrapper: corners → decoded string. | New |
| `dev3.py` | Orchestrate Steps 1–13 with prints + plots. | Modified |

Tests (mirroring existing `tests/` style, `pytest`, synthetic ground truth):
`test_landmarks.py`, `test_version.py`, `test_homography.py`, plus an additions
to `test_finder_pattern.py`.

---

## Implementation steps (in order)

### Step A — Carry inner corners on `FinderPattern`  *(modify `finder_pattern.py`)*

The inner quad is already computed and sitting in `all_corners` (it's the
second-largest quad per cluster); we just need to retain it on the dataclass
instead of dropping it.

- Add field `inner_corners: Optional[np.ndarray] = None` (shape `(4,2)`, `(row,col)`).
- Extend `extract_finder_patterns`: per cluster, sort the candidate quads by
  `polygon_area` descending; assign largest → `outer_corners`, second → `inner_corners`.
  If only one quad exists, leave `inner_corners=None`.
- Keep backward compatibility: existing callers/tests that only read
  `outer_corners` still pass.

**Unit tests** (extend `test_finder_pattern.py`):
- A cluster with 2 quads → outer is the larger, inner is the smaller.
- A cluster with 1 quad → `inner_corners is None`.

---

### Step B — Order square corners + build named landmarks  *(new `landmarks.py`)*

Functions:

```text
local_basis(triplet, fps) -> (right, down)        # unit (row,col) vectors
order_square_corners(points4, right, down) -> ndarray(4,2)   # [TL,BL,BR,TR]
```

- `local_basis`: look up TL/TR/BL `FinderPattern`s by `cluster_idx`; compute
  centroids; `right = normalize(center_TR - center_TL)`,
  `down = normalize(center_BL - center_TL)`. Vectors stay in `(row,col)` space;
  the dot products in `order_square_corners` are basis-agnostic as long as
  `right`/`down` and `points` share the same frame.
- `order_square_corners`: subtract centroid; project each point onto `right` (→`r`)
  and `down` (→`d`); assign by sign quadrant:
  `(r<0,d<0)→TL(0)`, `(r<0,d>0)→BL(1)`, `(r>0,d>0)→BR(2)`, `(r>0,d<0)→TR(3)`.
  Validate that exactly one point lands in each quadrant; otherwise fall back to
  the `atan2(d, r)` sort and raise/flag if still degenerate.

Then a builder:

```text
@dataclass NamedLandmarks:
    # each is ndarray(4,2) in (row,col), ordered [TL,BL,BR,TR]
    A, B, C, D, E, F

build_named_landmarks(triplet, fps) -> NamedLandmarks
```

- A/B from TL pattern (outer/inner), C/D from TR, E/F from BL, each passed
  through `order_square_corners` with the shared `right/down` basis.

**Unit tests** (`test_landmarks.py`):
- Hand-built axis-aligned squares with known corners → ordering returns
  `[TL,BL,BR,TR]` exactly. Use `(row,col)` so `down=(1,0)`, `right=(0,1)`.
- A rotated square (apply a known rotation to the 4 corners and the basis) →
  ordering still correct, proving basis-relative robustness.
- `build_named_landmarks` on a synthetic triplet returns 6 properly-ordered squares.

---

### Step C — Canonical grid coordinates + colinear quadruples  *(`landmarks.py`)*

```text
canonical_grid_landmarks(N) -> NamedLandmarks-of-grid-coords
```

Grid coords are `(x, y)` (x right, y down), matching the algorithm. Outer span is
`0..7`; the inner square is the white ring at **offset 1**, so it spans `1..6`:

```
A: (0,0)(0,7)(7,7)(7,0)            order [TL,BL,BR,TR]
C: (N-7,0)(N-7,7)(N,7)(N,0)
E: (0,N-7)(0,N)(7,N)(7,N-7)
B: (1,1)(1,6)(6,6)(6,1)            # white-ring square inside A
D: (N-6,1)(N-6,6)(N-1,6)(N-1,1)
F: (1,N-6)(1,N-1)(6,N-1)(6,N-6)
```

Expected inner cross-ratio therefore comes from positions `1, 6, N-6, N-1`
(outer from `0, 7, N-7, N`), both via the generic `expected_cross_ratio` helper.

Quadruple definitions (return index/name tuples, each is 4 points in a fixed order):

```text
outer: (A0,A1,E0,E1) (A3,A2,E3,E2) (A0,A3,C0,C3) (A1,A2,C1,C2)
inner: (B0,B1,F0,F1) (B3,B2,F3,F2) (B0,B3,D0,D3) (B1,B2,D1,D2)
```

Provide a helper that, given a `NamedLandmarks` (image or grid), returns the 8
ordered point-quadruples tagged `type ∈ {"outer","inner"}`.

**Unit tests:**
- For known `N` (e.g. 21), assert a couple of canonical coordinates (e.g.
  `C2 == (N,7)`, `B == [(1,1),(1,6),(6,6),(6,1)]`).
- Each canonical outer/inner quadruple is genuinely colinear in grid space
  (lateral spread ≈ 0).

---

### Step D — Cross-ratio + version estimation  *(new `version.py`)*

```text
measured_cross_ratio(points4) -> (r, line_error, span)
expected_cross_ratio(x0,x1,x2,x3) -> float
build_constraints(image_named_landmarks) -> list[Constraint]   # uses Step C quadruples
filter_constraints(constraints, k=3, eps=1e-2, min_span=?, cap=0.05) -> list[Constraint]
estimate_version(usable_constraints, v_range=range(1,41)) -> (V_best, score_by_V)
```

- `measured_cross_ratio`: center the 4 points, SVD of the `4×2` centered matrix;
  `sigma1,sigma2 = S`; `line_error = sigma2/sigma1`; `direction = Vt[0]`; project
  to 1D `u_i = dot(p_i - center, direction)`; flip sign if `u[3] < u[0]`; optional
  monotonicity sanity check (`u0<u1<u2<u3`, else flag/reject); compute
  `r = ((u2-u0)(u3-u1))/((u3-u0)(u2-u1))`. `span = sigma1`.
- `Constraint` dataclass: `type, r_measured, line_error, span`.
- `filter_constraints`: drop `span < min_span`; sort by `line_error`; take best `k`;
  `reference_error = max(line_error among best k)`; keep constraints with
  `line_error <= min(reference_error + eps, cap)`.
- `estimate_version`: for each `V`, `N=4V+17`; for each usable constraint compute
  `r_expected` from the canonical positions for that `type` (via
  `expected_cross_ratio`: outer `0,7,N-7,N`; inner `1,6,N-6,N-1`); per-constraint error
  `|log(r_measured / r_expected)|`; `score[V] = median(errors)`; return `argmin`.
- `min_span` choice: tie to expected finder size in pixels; expose as a parameter
  with a sensible default (e.g. a few px). Document the rationale (cross-ratio is
  unstable when points nearly coincide).

**Unit tests** (`test_version.py`):
- `expected_cross_ratio(0,7,N-7,N)` equals `(N-7)**2/(N*(N-14))` for several N.
- `measured_cross_ratio` on perfectly colinear synthetic points returns
  `line_error≈0` and `r` equal to the expected value (orientation-flip handled).
- `measured_cross_ratio` is **projective-invariant**: take grid quadruples, apply a
  known homography `H`, and confirm `r_measured ≈ expected_cross_ratio(grid positions)`.
- **End-to-end version recovery** (the key test): for several versions
  `V ∈ {1,2,5,10,25,40}`, synthesize the 24 grid landmarks, map them through a
  random-but-nondegenerate `H`, build/filter constraints, and assert
  `estimate_version(...) == V`. (Small V like 1 is least discriminative — confirm it
  still wins; if not, document the resolution floor.)

---

### Step E — Homography (DLT + RANSAC + LM)  *(new `homography.py`)*

All functions take/produce `(x, y)` for image points and `(X, Y)` for grid points.
Provide a small adapter to convert `(row,col)` landmark arrays → `(x,y)`.

```text
normalization_transform(points) -> T(3x3)
estimate_homography_dlt(src_xy, dst_xy) -> H(3x3)        # normalized DLT
project_points(H, src_xy) -> dst_xy
ransac_homography(src_xy, dst_xy, threshold=2.0, iters=..., min_inliers=12) -> (H, inlier_mask)
refine_homography_lm(H_init, src_xy, dst_xy, loss="huber") -> H
compute_qr_corners(H, N) -> ndarray(4,2)  # project (0,0),(N,0),(N,N),(0,N) -> image (x,y)
```

- `estimate_homography_dlt`: normalize src+dst (the `sqrt(2)` mean-distance
  transform), assemble the `2n×9` matrix, SVD, take last row of `Vt`, reshape,
  denormalize `H = inv(T_dst) @ H_norm @ T_src`, scale so `H[2,2]=1`.
- `ransac_homography`: sample 4 correspondences, skip degenerate samples (collinear
  / near-duplicate), fit DLT, count inliers by reprojection distance `< threshold`,
  keep best; require `>= min_inliers` (12 of 24, the "throw out up to half" rule);
  refit DLT on all inliers.
- `refine_homography_lm`: 8-param (`h33=1`) least squares on reprojection residuals
  via `scipy.optimize.least_squares` (Huber/Cauchy loss).
- `compute_qr_corners`: returns corners in `(x,y)`; order them as `[TL,TR,BR,BL]`
  for OpenCV (note: project order is `(0,0)=TL,(N,0)=TR,(N,N)=BR,(0,N)=BL`).

**Unit tests** (`test_homography.py`):
- `normalization_transform`: transformed centroid ≈ 0, mean distance ≈ `sqrt(2)`.
- DLT round-trip: random `H_true`, map 24 grid pts, recover `H` with DLT, assert
  reprojection error ≈ 0 (up to scale).
- `project_points` matches manual homogeneous projection.
- RANSAC robustness: corrupt 4–8 of 24 correspondences with large offsets; assert
  the clean ones are inliers and recovered `H` ≈ `H_true`.
- LM refinement: add small Gaussian noise to targets; assert refined reprojection
  error ≤ DLT reprojection error.
- `compute_qr_corners`: with identity-ish `H` and known `N`, corners match
  `(0,0),(N,0),(N,N),(0,N)` mapped through `H`.

---

### Step F — OpenCV decode wrapper  *(new `decode.py`)*

```text
decode_qr(image, corners_xy) -> (text: str, ok: bool)
```

- `corners_xy` shape `(4,2)` float32 in `[TL,TR,BR,BL]` order (`(x,y)`).
- Use `cv2.QRCodeDetector().decode(image, points)` (points reshaped to
  `(1,4,2)`/`(4,1,2)` as the API expects). The input image should be the original
  grayscale (or its uint8 form), not the boolean binary.
- Return decoded string + success flag.

**Unit test** (lightweight integration, may live in `test_homography.py` or a new
`test_decode.py`): generate a *clean* QR (no warp) with known content via
`make_qr_image`, feed the true corners, assert decoded text matches.

---

### Step G — Wire it into `dev3.py`  *(modify `dev3.py`)*

Append new `# %%` cells after the existing `find_triplets` block. Replace the
trailing triple-quoted design note with working code:

1. **Inner corners**: rely on the Step A change so each `FinderPattern` now has
   `inner_corners`. (The existing per-cluster loop already produces both ring
   edges; confirm they survive into `extract_finder_patterns`.)
2. **Pick the triplet** (`triplets[0]` for the demo) and build named image
   landmarks via `build_named_landmarks`.
3. **Version**: `build_constraints` → `filter_constraints` → `estimate_version`.
   `print(f"Inferred version: {V}  (N={N})")`. Plot the 8 quadruples colored by
   `line_error` / which were kept, for visual sanity.
4. **Correspondences**: `canonical_grid_landmarks(N)` + image landmarks →
   24 `(grid_xy, image_xy)` pairs (convert `(row,col)→(x,y)`).
5. **Homography**: `ransac_homography` → `refine_homography_lm`. Print inlier count.
6. **Corners**: `compute_qr_corners(H, N)`; overlay the 4 corners + quad on the
   image (remember plot x=col, y=row). `print` the 4 corners.
7. **Decode**: `decode_qr(img_gray, corners)`; `print` the decoded payload and
   compare to the generator's `content` ("Some data").

Add a short assertion/printout that the inferred version equals the generator's
version (1) and decoded text matches, as a smoke check while developing.

---

## Suggested build & verification order

1. **Step A** + its tests (`pytest tests/test_finder_pattern.py`).
2. **Step B** (`landmarks.py` ordering) + `test_landmarks.py`.
3. **Step C** (canonical coords/quadruples) + tests.
4. **Step D** (`version.py`) + `test_version.py` — this is the heart of "find the
   version"; get the end-to-end synthetic version-recovery test green for many V.
5. **Step E** (`homography.py`) + `test_homography.py`.
6. **Step F** (`decode.py`) + decode test.
7. **Step G** — run `dev3.py` end to end on the version-1 test image; confirm:
   inferred version = 1, 4 plausible corners drawn, OpenCV decodes "Some data".

## Risks / things to watch

- **Coordinate frame mistakes** `(row,col)` vs `(x,y)` are the most likely bug
  source — keep landmarks `(row,col)` until the homography adapter, convert once.
- **Inner-square coords**: the detected inner quad is the white ring at offset 1
  (coords `1/6`), so canonical inner positions are `1, 6, N-6, N-1`. Don't confuse
  it with the offset-2 center square (`2/5`).
- **Low-version discrimination**: cross-ratios for V=1 are close across nearby
  versions; the median-of-log-ratio score plus filtering should still resolve it,
  but validate explicitly. This is the main correctness risk — everything else is
  straightforward linear algebra.
