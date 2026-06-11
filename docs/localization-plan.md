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
- `finder_pattern.py` — `FinderPattern` (`outer_corners` + `inner_corners`), `extract_finder_patterns`, `check_association`, `find_all_associations`, `find_triplets` → `Triplet(top_left_idx, top_right_idx, bottom_left_idx)`.
- `landmarks.py` — **NEW**: `local_basis`, `order_square_corners`, `NamedLandmarks`, `build_named_landmarks`, `canonical_grid_landmarks`, `Quadruple`, `get_colinear_quadruples`.
- `version.py` — **NEW**: `measured_cross_ratio`, `expected_cross_ratio`, `Constraint`, `build_constraints`, `filter_constraints`, `estimate_version`, `expected_cross_ratio_by_N`.
- `dev3.py` — the driver script (cell-based, `# %%`), now runs Steps A–D end-to-end, infers V=1 correctly, and asserts it matches the generator.

### Conventions that the new code MUST respect

1. **Corner storage is `(row, col)` = `(y, x)`.** Every corner array produced by
   the detection pipeline (`comp_arr[idx]`) is in `(y, x)` order. `dev3.py` plots
   `corners[:, 1]` (x) vs `corners[:, 0]` (y). New code keeps `(row, col)`
   internally and only converts to `(x, y)` at the OpenCV boundary.
2. **Indices vs cluster ids.** `Triplet` and `Association` reference
   `cluster_idx` (the `FinderPattern.cluster_idx`), not list positions. Look
   patterns up by `cluster_idx`.
3. **Angular-NMS corners are unordered** (sorted by angle around the centroid).
   They are NOT yet in canonical `0=TL,1=BL,2=BR,3=TR` order — `order_square_corners` fixes this.

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

| Module | Responsibility | Status |
|---|---|---|
| `finder_pattern.py` | Add `inner_corners` to `FinderPattern`; extend `extract_finder_patterns` to keep top-2 quads by area. | ✅ DONE |
| `landmarks.py` | Corner ordering, named image landmarks (A0..F3), canonical grid coords, the 8 colinear quadruples, the 24 correspondences. | ✅ DONE |
| `version.py` | SVD line-fit cross-ratio, expected cross-ratio, constraint filtering, version estimation. | ✅ DONE |
| `homography.py` | Normalization, DLT, projection, RANSAC, LM refinement, QR corner projection. | ⬜ TODO |
| `decode.py` | Thin OpenCV wrapper: corners → decoded string. | ⬜ TODO |
| `dev3.py` | Orchestrate Steps 1–13 with prints + plots. | ✅ Steps A–D wired |

Tests (mirroring existing `tests/` style, `pytest`, synthetic ground truth):
`test_landmarks.py`, `test_version.py`, `test_homography.py` (not yet), `test_decode.py` (not yet).

---

## ✅ Steps A–D: Implementation complete (handover notes)

### What was built

#### Step A — `finder_pattern.py`

`FinderPattern.inner_corners` added as `np.ndarray | None = None`. `extract_finder_patterns` sorts candidate quads by `polygon_area` descending; largest → `outer_corners`, second → `inner_corners` (or `None`). Backward compatible — existing callers and all 3 original tests pass unchanged.

#### Step B — `landmarks.py` (corner ordering + named landmarks)

- `local_basis(triplet, fps) → (right, down)` — unit vectors in `(row, col)` from triplet centroids: `right = norm(center_TR - center_TL)`, `down = norm(center_BL - center_TL)`.
- `order_square_corners(points4, right, down) → ndarray(4,2)` — quadrant assignment via basis projections `(r<0,d<0)→TL=0, (r<0,d>0)→BL=1, (r>0,d>0)→BR=2, (r>0,d<0)→TR=3`. Falls back to `atan2(d,r)` sort anchored to the most-negative-angle quadrant if quadrant assignment is degenerate.
- `NamedLandmarks` dataclass with fields `A, B, C, D, E, F` (each `(4,2)` in `[TL,BL,BR,TR]` order; `B/D/F` may be `None`).
- `build_named_landmarks(triplet, fps) → NamedLandmarks` — orchestrates basis + ordering for all 6 squares.

#### Step C — `landmarks.py` (canonical coords + quadruples)

- `canonical_grid_landmarks(N) → NamedLandmarks` — grid coords in `(row, col)` = `(y, x)` storage. Outer squares: `A(0..7, 0..7)`, `C(0..7, N-7..N)`, `E(N-7..N, 0..7)`. Inner squares (white ring offset 1): `B(1..6, 1..6)`, `D(1..6, N-6..N-1)`, `F(N-6..N-1, 1..6)`. All ordered `[TL,BL,BR,TR]`.
- `Quadruple` dataclass: `points (4,2)`, `type ("outer"/"inner")`, `label`.
- `get_colinear_quadruples(landmarks) → list[Quadruple]` — returns up to 8 quadruples. Outer: `(A0,A1,E0,E1)`, `(A3,A2,E3,E2)`, `(A0,A3,C0,C3)`, `(A1,A2,C1,C2)`. Inner (if all inner corners present): `(B0,B1,F0,F1)`, etc.

#### Step D — `version.py` (cross-ratio + version estimation)

- `measured_cross_ratio(points4) → (r, line_error, span)` — SVD of `4×2` centered matrix; `line_error = σ2/σ1`; projects onto principal axis; flips sign if `u[3] < u[0]`; computes `r = (u2-u0)(u3-u1)/((u3-u0)(u2-u1))`.
- `expected_cross_ratio(x0, x1, x2, x3) → float` — closed-form.
- `Constraint` dataclass: `type, label, r_measured, line_error, span`.
- `build_constraints(landmarks) → list[Constraint]` — computes measured cross-ratios for all colinear quadruples.
- `filter_constraints(constraints, k=3, eps=1e-2, min_span=1.0, max_error_cap=0.05)` — span threshold → sort by line_error → keep best by reference error with cap.
- `estimate_version(constraints, v_range=range(1,41)) → (V_best, scores)` — for each V, `N=4V+17`; per-constraint error `|log(r_measured / r_expected(N))|`; score = median error; argmin wins.
- `expected_cross_ratio_by_N(N) → (outer, inner)` — convenience helper.

### Test status: 31/31 passing

| Test file | Tests | Coverage |
|---|---|---|
| `test_finder_pattern.py` | 3 | Unchanged from original (still pass) |
| `test_geometry.py` | 5 | Unchanged from original |
| `test_landmarks.py` | 8 | `order_square_corners` (axis-aligned, rotated 30°), `local_basis`, `canonical_grid_landmarks` (N=21 coords, offset-1), `get_colinear_quadruples` (colinearity of outer/inner sets, count with/without inners) |
| `test_version.py` | 11 | `expected_cross_ratio` formulas, `measured_cross_ratio` (perfect colinear, orient flip, noisy, **projective invariance**), constraint build/filter (span drop, best-k, cap), end-to-end version recovery: canonical V=1/V=5, **homography-warped V∈{1,2,5,10}** |
| **Total** | **31** | |

Run with: `uv run pytest tests/ -v`

### How to run the demo

```sh
uv run python -c "
import matplotlib; matplotlib.use('Agg')
import runpy; runpy.run_path('src/qr_reader/dev3.py')
"
```

Or individually in an interactive Python / Jupyter cell-by-cell runner.

### End-to-end result (Steps A–D, version-1 test image)

```
Inferred version: V=1  (N=21)
Top 5 version scores: V=1: 0.006, V=2: 0.080, V=3: 0.122, V=4: 0.145, V=5: 0.160
✓ Version check passed: inferred V=1 matches generator's V=1
```

The measured cross-ratios closely match expected: outer ≈1.30–1.32 (expected 1.333), inner ≈1.14–1.15 (expected 1.146). Version discrimination is sharp — V=2 score is 12× worse.

---

## ⬜ Next: Steps E–G (not yet started)

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

### Step G — Wire it into `dev3.py`  *(modify `dev3.py`)*

Append new `# %%` cells after the existing Step D block. The script currently
ends with the smoke-check assertion for V=1 and a printout of expected
cross-ratios. Append:

1. **Correspondences**: `canonical_grid_landmarks(N_best)` + image landmarks →
   24 `(grid_xy, image_xy)` pairs. Convert `(row,col)→(x,y)` at this boundary.
   The 24 pairs are: all 4 corners of all 6 squares (A–F). If any inner squares
   are `None`, skip those 8 points for 16 total.
2. **Homography**: `ransac_homography` → `refine_homography_lm`. Print inlier count.
3. **Corners**: `compute_qr_corners(H, N_best)`. Overlay the 4 QR corners +
   boundary quad on the image (remember: plot `x=col`, `y=row`). `print` the
   4 corner coordinates.
4. **Decode**: `decode_qr(img_gray, corners)` — use the original grayscale
   `img_gray` (uint8 from `generate_test_image()`), NOT the binary. `print` the
   decoded payload and compare to the known content "Some data".

Add a final assertion: `decoded_text == "Some data"`.

---

## Suggested build & verification order

1. ~~**Step A** + its tests (`pytest tests/test_finder_pattern.py`).~~ ✅
2. ~~**Step B** (`landmarks.py` ordering) + `test_landmarks.py`.~~ ✅
3. ~~**Step C** (canonical coords/quadruples) + tests.~~ ✅
4. ~~**Step D** (`version.py`) + `test_version.py`~~ ✅
5. **Step E** (`homography.py`) + `test_homography.py`.
6. **Step F** (`decode.py`) + decode test.
7. **Step G** — run `dev3.py` end to end on the version-1 test image; confirm:
   inferred version = 1, 4 plausible corners drawn, OpenCV decodes "Some data".

---

## Risks / things to watch

- **Coordinate frame mistakes** `(row,col)` vs `(x,y)` are the most likely bug
  source — keep landmarks `(row,col)` until the homography adapter, convert once.
  The conversion is simply `(x, y) = (col, row)`.
- **Inner-square coords**: the detected inner quad is the white ring at offset 1
  (coords `1/6`), so canonical inner positions are `1, 6, N-6, N-1`. Don't confuse
  it with the offset-2 center square (`2/5`).
- **Low-version discrimination**: cross-ratios for V=1 are close across nearby
  versions; the median-of-log-ratio score plus filtering should still resolve it,
  but validate explicitly. ✅ Validated — V=1 score is 0.006 vs 0.080 for V=2.

---

## Handover notes for the next agent (Steps E–G)

### Where to start

The codebase is at `src/qr_reader/`. The two new modules to create are:
- `src/qr_reader/homography.py`
- `src/qr_reader/decode.py`

And two new test files:
- `tests/test_homography.py`
- `tests/test_decode.py`

Then append to `src/qr_reader/dev3.py` (new `# %%` cells after the existing Step D block, line ~475).

### Key APIs already available for Step E

**From `landmarks.py`** — use `canonical_grid_landmarks(N_best)` to get the 24 grid correspondences and `build_named_landmarks(triplet, fps)` for the image points:

```python
from qr_reader.landmarks import canonical_grid_landmarks, build_named_landmarks

grid_lm = canonical_grid_landmarks(N_best)   # where N_best = 4*V_best + 17
image_lm = build_named_landmarks(triplet, fps)

# Convert (row,col) → (x,y) for homography:
def rc_to_xy(pts):   # pts shape (4,2) in (row,col)
    return pts[:, ::-1]  # swap col<->row: (row,col) → (col,row) = (x,y)

# Build 24 correspondences:
src_xy = []  # grid
dst_xy = []  # image
for attr in ["A", "B", "C", "D", "E", "F"]:
    g = getattr(grid_lm, attr)
    i = getattr(image_lm, attr)
    if g is not None and i is not None:
        src_xy.append(rc_to_xy(g))
        dst_xy.append(rc_to_xy(i))
src_xy = np.vstack(src_xy)
dst_xy = np.vstack(dst_xy)
```

**From `version.py`**: `V_best` and `N_best` are already computed in the Step D cell.

### OpenCV corner order

`compute_qr_corners` must return `[TL, TR, BR, BL]` in `(x,y)` because that's what `cv2.QRCodeDetector().decode()` expects. The natural projection order `(0,0)=TL, (N,0)=TR, (N,N)=BR, (0,N)=BL` already gives this order.

### Important: use the original grayscale for decode

`generate_test_image()` returns `img_gray` (uint8). Do NOT pass the boolean `img_binary` to `decode_qr`. The OpenCV decoder needs proper 0–255 grayscale.

### Running tests

```sh
uv run pytest tests/ -v
```

The project uses `uv` for dependency management. `pytest` is already installed in the venv.

### Verified invariants (don't regress)

- All 31 existing tests must keep passing.
- `dev3.py` must run end-to-end without errors and print `✓ Version check passed`.
- The smoke-check assertion at line ~471 expects `V_best == 1`. If you change the test image or version, update this.

### Coordinate frame cheat sheet

| Context | Storage | Example |
|---|---|---|
| Detection pipeline | `(row, col)` = `(y, x)` | `comp_arr[idx]` is `(y, x)` |
| `NamedLandmarks` (image + grid) | `(row, col)` = `(y, x)` | `A[0] = [0, 0]` means row=0, col=0 |
| Plotting (matplotlib) | `(x, y)` | `ax.plot(corners[:, 1], corners[:, 0])` |
| Homography functions | `(x, y)` | input/output to `homography.py` |
| OpenCV decode | `(x, y)` | `cv2.QRCodeDetector().decode()` |
