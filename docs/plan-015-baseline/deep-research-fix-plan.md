# Fix Plan — Clean V=1 Detector Failures (Deep-Research-Backed)

## Current state

3 tests fail on clean axis-aligned V=1 QR images (`box_size=10`, `border=4`):

| Test | Symptom |
|------|---------|
| `test_corner_order` | TL.x = 127 (should be ~40) — TR/BL roles are swapped |
| `test_identity_like_for_clean_image` | H maps grid origin to (74, 103) instead of (40, 40) |
| `test_clean_image_has_finder_patterns` | Sampled TL finder has 23 dark modules instead of >30 — garbled |

Diagnostic (run 2026-07-03): 3 clusters found — TL=(74.5,74.5), TR=(74.5,214.5), BL=(214.5,74.5) — but `find_valid_triplets` outputs `TL=1, TR=0, BL=2` instead of `TL=1, TR=2, BL=0`. TR and BL are swapped.

**Root cause (confirmed by literature):** The cross product in `find_valid_triplets` (`finder_pattern.py:403`) is computed in `(row, col)` coordinates but the sign convention assumes `(x, y) = (col, row)` image coordinates. The deep-research report states unequivocally: cross products for TL/TR/BL classification must be done in `(x=col, y=row)` order, where `cross(v_TR, v_BL) > 0` for a normal QR.

---

## Phase 1 — Fix cross-product sign convention

### Goal
Zero test failures. All 3 failing tests pass.

### Instructions

**File:** `src/qr_reader/detector/finder_pattern.py`

In `find_valid_triplets`, locate the cross-product role-assignment block (approximately lines 403–409):

```python
cross = float(vec_ba[0] * vec_bc[1] - vec_ba[1] * vec_bc[0])
if cross > 0:
    top_right = a_idx
    bottom_left = c_idx
else:
    top_right = c_idx
    bottom_left = a_idx
```

The vectors `vec_ba` and `vec_bc` are in `(row, col)` because `centers` is computed as `fp.outer_corners.mean(axis=0)` which yields (row, col). The cross product must use `(x=col, y=row)` image-coordinate convention.

Apply the fix — two equivalent approaches, pick either:

**Approach A (convert vectors to xy before cross):**
```python
v_ba_xy = vec_ba[::-1]
v_bc_xy = vec_bc[::-1]
cross = float(v_ba_xy[0] * v_bc_xy[1] - v_ba_xy[1] * v_bc_xy[0])
if cross > 0:
    top_right = a_idx
    bottom_left = c_idx
else:
    top_right = c_idx
    bottom_left = a_idx
```

**Approach B (negate the cross product branches):**
```python
cross = float(vec_ba[0] * vec_bc[1] - vec_ba[1] * vec_bc[0])
if cross > 0:
    top_right = c_idx
    bottom_left = a_idx
else:
    top_right = a_idx
    bottom_left = c_idx
```

Approach A is preferred — it makes the (x,y) conversion explicit and self-documenting.

**Also fix the old `find_triplets` function** (approximately line 243) — same cross-product logic, same bug. Apply the same fix. Although `find_triplets` is no longer called by the production path (we use `find_valid_triplets`), fixing it keeps the code consistent and avoids confusion if it's ever revived.

### Verification

Run:

```bash
pytest src/qr_reader/tests/test_detector.py -v
```

**Expected:** 26 passed, 0 failed (3 currently-failing tests + 23 currently-passing).

Then run the full suite to confirm no regressions:

```bash
pytest -q
```

**Expected:** 707 passed, 0 failed, 5 skipped (was 704 passed, 3 failed).

### Commit

```
fix: correct cross-product sign convention in triplet TR/BL classification

find_valid_triplets and find_triplets computed the cross product in
(row, col) coordinates but used the sign convention for (col, row)
image coordinates, causing TR and BL roles to be swapped.

Convert inter-centre vectors to (x=col, y=row) before computing the
cross product, matching ZXing's convention where cross(v_TR, v_BL) > 0
for a normal non-mirrored QR.

Fixes test_corner_order, test_identity_like_for_clean_image, and
test_clean_image_has_finder_patterns on clean V=1 images.
```

---

## Phase 2 — Canonicalize per-finder corner order to global frame

### Goal
Prevent a latent bug: the three finders may have different local corner orders because the 4-fold symmetry of the gradient histogram can produce a 90° rotation in `(e1, e2)`. Currently this doesn't fail the tests (all three finders happen to agree on a clean V=1 image), but it will cause failures as soon as one finder rotates.

### Background

The diagnostic output shows:
```
Cluster 0 (BL): e1=( 0, 1), e2=(-1, 0)  ← rotated 90° vs others
Cluster 1 (TL): e1=( 1, 0), e2=( 0, 1)
Cluster 2 (TR): e1=( 1, 0), e2=( 0, 1)
```

The DLT homography in `_run_detection` (`detector.py:142-161`) builds 12 point correspondences using `grid_offsets = [(0,0), (7,0), (7,7), (0,7)]` which assumes corner 0 = top-left of each finder in the global frame. If one finder's corners are cyclically shifted by 90°, DLT solves the wrong problem.

### Instructions

**File:** `src/qr_reader/detector/detector.py`

After the triplet is selected and TL/TR/BL roles are resolved (after line 130), add a corner-canonicalization step before the homography loop (before line 142):

1. **Define global frame directions** from the triplet centers:
   - `global_u = (c_tr - c_tl) / |c_tr - c_tl|` — the TR direction (unit vector)
   - `global_v = (c_bl - c_tl) / |c_bl - c_tl|` — the BL direction (unit vector)
   - Both are in (x, y) pixel coordinates.

2. **For each of the three finders (TL, TR, BL):**
   - Get its 4 outer corners in global (x, y) coordinates.
   - For each corner, compute its signed coordinate along `global_u` and `global_v`.
   - The corner with **most negative** u-coordinate AND **most negative** v-coordinate is the "top-left" corner → rotate to position 0.
   - The corner with **most positive** u-coordinate AND **most negative** v-coordinate is the "top-right" corner → rotate to position 1.
   - Etc. for positions 2 and 3, matching the `[(0,0), (7,0), (7,7), (0,7)]` ordering.

3. **Pseudocode for one finder's corners** `corns` (shape 4×2 in xy):
   ```python
   center = corns.mean(axis=0)
   u_proj = (corns - center) @ global_u
   v_proj = (corns - center) @ global_v
   # Find which corner index is which role
   idx_tl = argmin(u_proj + v_proj)  # or: (most negative u) AND (most negative v)
   # ... etc for idx_tr, idx_br, idx_bl
   # Reorder: [idx_tl, idx_tr, idx_br, idx_bl]
   corns_reordered = corns[[idx_tl, idx_tr, idx_br, idx_bl]]
   ```

   Alternative: label each corner by its signed u/v quadrant and reorder to match `[−u−v, +u−v, +u+v, −u+v]`.

4. **Use the reordered corners** in the `global_corners_xy` dict (or a new dict for canonicalized corners) so the DLT correspondence points use the correct ordering.

   Replace lines 142–161: instead of using `tl_c = global_corners_xy[tl_idx]` directly, use the canonicalized corners.

### Verification

The existing tests should still pass. The canonicalization should be a no-op when all finders already agree (as in the clean V=1 case, where Cluster 0 happens to produce corners in the correct physical order despite having rotated e1/e2). To verify the fix works, add a test that synthesizes a QR with 90°-rotated finder orientations:

```python
def test_finder_90deg_rotation_robustness(self):
    """Corners are correct even when one finder's orientation is 90° off."""
    img = make_qr_image(content="rot90", version=1, box_size=10, border=4)
    corners, version = detect_corners(img)
    # Same assertions as test_corner_order — TL.x < TR.x, etc.
    tl, tr, br, bl = corners
    assert tl[0] < tr[0] + 5.0
    assert tl[1] < bl[1] - 200.0
```

(This test may not reliably trigger the 90° rotation unless we control the orientation estimator's output. A simpler verification: assert that after canonicalization, the 12 DLT correspondences have the expected (x,y) ordering for each finder.)

### Commit

```
fix: canonicalize per-finder corner order to global triplet frame before DLT

Finder patterns can have local (e1,e2) frames rotated 90° relative to
each other because the 4-fold gradient histogram can pick either the
horizontal or vertical axis as e1. Before, DLT assumed corner 0 means
the same physical corner on all three finders.

Now, after triplet roles are resolved, each finder's 4 corners are
reordered to match the global TL→TR/→BL directions. This matches
quirc's approach of rotating capstone corners after the global grid
is established.
```

---

## Phase 3 — Add timing-pattern score to homography selection

### Goal
Better discrimination between candidate homographies. A TR/BL swap or a 90° corner-cycle error may still produce a numerically finite warp, but the timing pattern (row 6 and column 6 of the QR grid) will have wrong alternations. This is what quirc's `fitness_all` does: score timing-pattern consistency during grid fitting.

### Instructions

**File:** `src/qr_reader/detector/detector.py` (or a new `src/qr_reader/detector/timing.py`)

1. **Function: `score_timing_pattern(bits, N)`**
   - Given a sampled `bits` matrix (N×N bool), examine row 6 and column 6.
   - The timing pattern is alternating dark/light: `[D, L, D, L, ...]`.
   - Score = number of transitions that match the expected alternating pattern.
   - For a V=1 (N=21) QR: row 6 has cols 0..20, but cols 0–7 overlap with the TL finder and cols 13–20 overlap with TR/TL finders (or quiet zone). Focus on the central segment: cols 8..12 modulo finders. Actually, the timing pattern in row 6 runs from col 8 to col N-9 = col 12 for V=1 (very short). For higher versions, the span is longer.
   - Score both row 6 and column 6. Sum transitions that match expected polarity.

2. **Integrate into `_run_detection` homography loop** (lines 148–175):
   - For each candidate N, after computing reprojection error `err`, also compute a timing score.
   - Combine: `combined_err = err - w_timing * timing_score` where `w_timing` is a small weight (e.g. 0.1×module_pitch).
   - Select the candidate with the best combined score.

3. **Alternatively**, use the timing score as a final verification: if the best homography's timing score is below a threshold, reject the candidate or try the next-best triplet.

### Expected result

No regression on existing tests. The timing score acts as a tiebreaker or verification step. For clean V=1 images with correct homography, the timing score should be high. For incorrect homographies (e.g. if a TR/BL swap survives Phase 1), the timing score should be low and cause rejection.

### Commit

```
feat: add timing-pattern score to homography selection

Scores row 6 and column 6 of the sampled bit matrix for the expected
alternating dark/light timing pattern. Integrates into the candidate-N
search loop in _run_detection as a weighted term alongside reprojection
error. Follows quirc's grid-fitness approach for V=1 (no alignment
pattern available).
```

---

## Phase 4 — Two-stage dimension estimation with hypotenuse check

### Goal
More stable version estimation using all three inter-finder distances, not just the two TL→TR and TL→BL legs. The deep-research report recommends a simple right-triangle geometry model:

\[
\hat{s} = \frac{d(TL,TR) + d(TL,BL) + d(TR,BL)/\sqrt{2}}{3 \cdot m_{avg}}
\]

followed by snapping \(N = \hat{s} + 7\) to the legal QR dimension class (\(N \equiv 1 \pmod{4}\), with \(N \in \{21, 25, ..., 177\}\)).

### Instructions

**File:** `src/qr_reader/detector/detector.py`

1. **In `_run_detection`**, after computing `dx` and `dy` (lines 137–139), also compute the hypotenuse:

   ```python
   dh = float(np.linalg.norm(c_tr - c_bl))
   ```

2. **Replace the single-point estimate** (line 139):
   ```python
   N_est = int(round((dx + dy) / (2.0 * m_avg) + 7))
   ```
   with:
   ```python
   s_hat = (dx + dy + dh / np.sqrt(2)) / (3.0 * m_avg)
   N_est = int(round(s_hat + 7))
   ```

3. **Snap to legal QR dimensions** after the initial estimate:
   ```python
   # Snap to nearest legal QR dimension (N ≡ 1 mod 4, 21 ≤ N ≤ 177)
   N_legal = ((N_est - 17) // 4) * 4 + 21
   N_legal = max(21, min(177, N_legal))
   N_est = N_legal
   ```

   This replaces/adjusts the search range `[N_est-2, N_est+2]`. Since we snap to legal dimensions, the candidate search should iterate over adjacent legal dimensions:
   ```python
   for N_cand in range(max(21, N_legal - 4), min(181, N_legal + 5), 4):
   ```

4. **Post-homography verification** (Phase 5 companion): Once a homography is selected, verify N using the rectified timing pattern. This is deferred until the timing-pattern scoring from Phase 3 is in place.

### Expected result

No regression. The hypotenuse term stabilizes the estimate when one leg is biased. For clean V=1 images, `dx ≈ dy ≈ 140`, `dh ≈ 198`, `m_avg ≈ 10`, so:
- Old: N = round((140+140)/(20) + 7) = round(14+7) = 21 ✓
- New: s = (140+140+198/1.414)/(30) = (140+140+140)/30 = 420/30 = 14, N = 14+7 = 21 ✓

Both give the same answer for clean data; the improvement is in robustness.

### Commit

```
fix: use all three inter-finder distances for version estimation

Adds the hypotenuse TR→BL distance (divided by √2) to the two TL legs
when estimating the symbol dimension, following ZXing's right-triangle
geometry model. Snaps the initial N_est to the nearest legal QR
dimension and iterates candidates over adjacent legal sizes.
```

---

## Summary of expected outcomes

| Phase | Before | After | Key metric |
|-------|--------|-------|------------|
| **P1** | 704 passed, 3 failed | **707 passed, 0 failed** | Fixes all 3 failing tests |
| **P2** | 707 passed | 707 passed + latent bug fixed | No test regression; 90° rotation handled |
| **P3** | 707 passed | 707 passed + timing tiebreaker | Better homography discrimination |
| **P4** | 707 passed | 707 passed + stable N estimate | Hypotenuse-robust version detection |

### Files touched per phase

| File | P1 | P2 | P3 | P4 |
|------|:--:|:--:|:--:|:--:|
| `finder_pattern.py` | ✓ | | | |
| `detector.py` | | ✓ | ✓ | ✓ |
| `test_detector.py` | | ✓ (optional) | | |

### Commit order

1. `fix: correct cross-product sign convention...`  (P1)
2. `fix: canonicalize per-finder corner order...`  (P2)
3. `feat: add timing-pattern score...`  (P3)
4. `fix: use all three inter-finder distances...`  (P4)
