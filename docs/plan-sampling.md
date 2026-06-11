# QR Sampling & Thresholding — Plan

## Goal

Before decoding, we must convert the perspective-distorted grayscale image of a QR code into a clean boolean grid (`N`×`N`) of "module = white / black". This replaces the current approach of just feeding the warped image directly to OpenCV's `QRCodeDetector`.

We do this by **forward-projecting each QR cell through the homography** into the original grayscale image, supersampling a 3×3 neighborhood per cell, computing an adaptive threshold from known finder-pattern cells, and voting.

## Architecture overview

```
img_gray (uint8)
   │
   ├── finder_pattern_known_cells(N)  → list of (r,c) for black / white cells
   │                                         │
   │     compute_adaptive_threshold(image, H, N)  ──── float threshold
   │
   └── sample_qr_bits(image, H, N, [threshold])
           │
           │  for each (r,c) in 0..N-1:
           │       supersample_cell(image, H, r, c)  → 9 floats
           │       weighted majority vote vs. threshold
           │
           └── (N,N) bool array
```

---

## New file: `src/qr_reader/sample.py`

### 1. `supersample_cell(image, H, row, col)` → `np.ndarray` shape `(3,3)`, dtype `float64`

**Purpose**: Sample a 3×3 neighborhood in the **grayscale image** for a single QR-grid cell at `(row, col)`.

**Inputs**:
| Arg | Type | Meaning |
|---|---|---|
| `image` | `np.ndarray` | Grayscale `uint8` image, shape `(H_img, W_img)` |
| `H` | `np.ndarray` | 3×3 homography: QR-grid `(x, y)` → image `(x, y)` |
| `row` | `int` | QR module row (0..N−1) |
| `col` | `int` | QR module col (0..N−1) |

**Algorithm**:

For each of the 9 sub-cell positions (offset `dx` in col-direction, `dy` in row-direction):
```
dx ∈ [0.25, 0.5, 0.75]
dy ∈ [0.25, 0.5, 0.75]
```

Build a `(9, 2)` array of QR-grid points in `(x, y)` convention:
```
x = col + dx
y = row + dy
```

Project through `H` using `homography.project_points(H, grid_xy)` → `(9, 2)` image coordinates in `(x, y)` order.

Sample `image` at these sub-pixel coordinates using `scipy.ndimage.map_coordinates`:
- `order=1` (bilinear interpolation)
- `mode='nearest'` (safe for marginally out-of-bounds points)
- Input to `map_coordinates` expects coordinates in `(row, col)` order, so we pass `(y_coords, x_coords)`.

Return the 9 values reshaped to `(3, 3)` where index `[1,1]` is the exact cell center `(col+0.5, row+0.5)`.

---

### 2. `finder_pattern_known_cells(N)` → `tuple[list, list]`
Returns `(black_cells, white_cells)` where each element is a `list[tuple[int, int]]` of `(row, col)` positions.

**Purpose**: Enumerate cells whose value is known a priori from the three finder patterns (TL, TR, BL), so we can fit an adaptive threshold.

**Finder pattern structure** (each is a 7×7 square):

| Ring | Coordinates (relative to pattern origin) | Value |
|---|---|---|
| Outer border | row∈{0,6} for all 7 cols **or** col∈{0,6} for rows 1..5 | Black |
| White ring (offset 1) | rows 1-5, cols 1-5, *excluding* inner 3×3 | White |
| Inner 3×3 (offset 2) | rows 2-4, cols 2-4 | Black |

The three patterns are located at:
| Pattern | Row range | Col range |
|---|---|---|
| **TL** | 0..6 | 0..6 |
| **TR** | 0..6 | N−7..N−1 |
| **BL** | N−7..N−1 | 0..6 |

Black cell count per pattern: 24 (outer border) + 9 (inner 3×3) = **33**.
White cell count per pattern: 25 (5×5 square) − 9 (inner 3×3) = **16**.

Across 3 patterns: 99 black cells, 48 white cells.

Cells in the **overlap** region (the BL finder's outer border and the TL finder's outer border share a corner at `(6,6)` — but the inner 3×3's don't overlap) are harmless duplicates. We don't deduplicate; more samples only improve the median estimate.

**Edge case**: the separator column/row (distinct black-white-black-white-black vertical/horizontal lines at column/row 6 between the finder patterns) is not a finder pattern. We do not sample from those. For the adaptive threshold, the 99+48 samples from the finder patterns are more than adequate.

---

### 3. `compute_adaptive_threshold(image, H, N)` → `float`

**Purpose**: Compute a single global threshold that cleanly separates black from white modules for *this specific image*, using the known finder-pattern cells as ground truth.

**Algorithm**:
1. Call `finder_pattern_known_cells(N)` → `(black_cells, white_cells)`.
2. For each black cell `(r, c)`, call `supersample_cell(image, H, r, c)` and extract only the **center** value (index `[1,1]`, i.e., the pixel at the exact cell center `(r+0.5, c+0.5)`).
3. Same for white cells.
4. Compute: `threshold = (median(black_vals) + median(white_vals)) / 2.0`.

**Why center-only for threshold calibration**: The finder pattern cells are large and well-separated — a single high-quality center sample is sufficient and avoids introducing noise from cell boundaries. The full 3×3 is used later for decoding actual content cells.

**Why median**: Outlier resistance. If a few finder-pattern cells fall on shadows, noise, or partial occlusion, the median shrugs it off.

---

### 4. `sample_qr_bits(image, H, N, threshold=None)` → `np.ndarray` shape `(N,N)`, dtype `bool`

**Purpose**: Sample every QR module and produce the final boolean grid.

**Algorithm**:
1. If `threshold` is not provided, compute `threshold = compute_adaptive_threshold(image, H, N)`.
2. Initialize `bits = np.empty((N, N), dtype=bool)`.
3. For each `(r, c)` in `0..N-1`:
   - `vals = supersample_cell(image, H, r, c)` → 9 floats
   - Binary decisions: each `val > threshold` → white vote
   - Weighted sum:
     - Center `vals[1,1]` contributes **weight 2**.
     - The other 8 values contribute **weight 1** each.
     - Total weight = 10. Majority threshold = ≥ 5 white votes.
   - `bits[r, c] = (white_votes >= 5)`.

**Center-weight rationale**: The true module value is best represented by the exact cell center. Off-center samples are more likely to bleed into neighboring modules. Double-weighting the center makes the vote more robust while still benefiting from the spatial averaging of the 3×3 neighborhood.

`(False = black, True = white)`, matching the convention in `binarize_image` (where `True` = white).

---

## Integration into `dev3.py` — Step G (new cell)

After the existing homography cell (which produces `H_refined` and `N_best`), insert a new cell:

```python
# %%
# Step G — Supersample QR bits from grayscale & decode via OpenCV
from qr_reader.sample import sample_qr_bits

bits = sample_qr_bits(img_gray, H_refined, N_best)
print(f"Sampled grid shape: {bits.shape}, "
      f"white fraction: {bits.mean():.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(bits, cmap="gray", interpolation="nearest")
ax.set_title(f"Sampled QR bits (V={V_best}, N={N_best})")
plt.show()

# Build a clean uint8 image for OpenCV
# Up-scale with box_size=10 and add a white quiet-zone border (4 modules)
box_size = 10
border = 4
img_clean = np.full(
    ((N_best + 2 * border) * box_size, (N_best + 2 * border) * box_size),
    255, dtype=np.uint8,
)
for r in range(N_best):
    for c in range(N_best):
        val = 255 if bits[r, c] else 0
        img_clean[
            (r + border) * box_size : (r + border + 1) * box_size,
            (c + border) * box_size : (c + border + 1) * box_size,
        ] = val

from qr_reader.decode import decode_qr
decoded_text, ok = decode_qr(img_clean, corners_xy=None)  # let OpenCV find corners

if ok:
    print(f'✓ Decoded from sampled bits: "{decoded_text}"')
else:
    print("✗ Decode failed from sampled bits")

assert ok, f"Decode failed for V={V_best}"
assert decoded_text == QR_CONTENT, (
    f"Content mismatch: expected '{QR_CONTENT}', got '{decoded_text}'"
)
print(f"✓ Content check passed: '{decoded_text}' == '{QR_CONTENT}'")
```

**OpenCV integration notes**:
- We **upsample** the `N`×`N` boolean grid by `box_size=10` to get decent-resolution pixels for OpenCV's internal scanning.
- We add a **quiet zone** (white border of 4 modules) matching the QR spec.
- We call `decode_qr(img_clean, corners_xy=None)`. OpenCV's `QRCodeDetector.detectAndDecode` with no corners will find the code itself in this clean, rectified image.
- This fully replaces the previous decode step (old Step F that passes `corners_xy`). The old cell can remain for comparison but a new assertion uses the sampled path.

**Dependency**: The `decode_qr` wrapper needs a small update to handle `corners_xy=None` — see below.

---

## Minor update to `decode.py`

`decode_qr` currently always reshapes `corners_xy` into `(1, 4, 2)`. Update to handle `None`:

```python
def decode_qr(image: np.ndarray, corners_xy: np.ndarray | None = None) -> tuple[str, bool]:
    detector = cv2.QRCodeDetector()
    if corners_xy is None:
        text, points, straight_qrcode = detector.detectAndDecode(image)
    else:
        points = corners_xy.astype(np.float32).reshape(1, 4, 2)
        text, straight_qrcode = detector.decode(image, points)
    return text, text != ""
```

---

## Testing strategy

### Test dependency

We use the `qrcode` library (already a project dependency in `qr_gen.py`) to generate **ground-truth bit grids** for known QR codes. OpenCV's `QRCodeDetector` is the "test-time dependency" per the spec — it validates that our sampled bits are correct by independently decoding them.

### Test file: `tests/test_sample.py`

#### Test 1: `test_supersample_cell_identity`

**Setup**: Create a synthetic grayscale image (e.g., 100×100) with a known value per pixel (e.g., `image[y, x] = y * 100 + x`). Set `H = np.eye(3)` (identity homography).

**Exercise**: Call `supersample_cell(image, H, row=10, col=20)`.

**Verify**: The 9 returned values match what bilinear interpolation would produce at `(y, x) = (10.25, 20.25), (10.25, 20.5), ...`. The center value `[1,1]` matches the pixel at `(10.5, 20.5)`.

#### Test 2: `test_supersample_cell_with_homography`

**Setup**: Same synthetic image. Set `H` to a translation (`[[1,0,5],[0,1,10],[0,0,1]]`).

**Exercise**: `supersample_cell(image, H, row=5, col=5)`.

**Verify**: Values match sampling at `(y, x) = (10+dy, 10+dx)` — i.e., the homography shift is correctly applied.

#### Test 3: `test_finder_pattern_known_cells_counts`

**Exercise**: `black, white = finder_pattern_known_cells(21)` (version 1).

**Verify**:
- `len(black) == 99` (3 × 33)
- `len(white) == 48` (3 × 16)
- No duplicates between black and white sets.
- Spot-check: `(0, 0)` is in black (TL outer corner).
- Spot-check: `(1, 1)` is in white (TL white ring corner).
- Spot-check: `(0, 14)` is in black (TR outer corner).
- Spot-check: `(14, 0)` is in black (BL outer corner).

#### Test 4: `test_finder_pattern_known_cells_spot_check_version_1`

**Verify**: For N=21 (version 1), the TL inner 3×3 cells `(2,2), (3,3), (4,4)` are black. The TL white ring cells `(1,3), (3,5)` are white.

#### Test 5: `test_compute_adaptive_threshold`

**Setup**: Create a rectified QR image. Generate a clean QR with `qrcode.QRCode`, convert to `uint8` (0 / 255). No distortion (so `H = eye(3)`). The black pixels are exactly 0, white are exactly 255.

**Exercise**: `threshold = compute_adaptive_threshold(image, eye(3), N)`.

**Verify**: `threshold` is roughly 127.5 (the midpoint). Since there's no noise, both medians are exact, so `threshold == 127.5`.

#### Test 6: `test_compute_adaptive_threshold_with_noise`

**Setup**: Same as Test 5 but add Gaussian noise (`std=10`) to the image.

**Verify**: Threshold falls in `[100, 155]` — robust in the presence of moderate noise.

#### Test 7: `test_sample_qr_bits_round_trip`  (the critical end-to-end unit test)

**Setup**: For each version `V ∈ {1, 2, 3}` and several content strings:
1. Generate a clean QR with `qrcode.QRCode` → `img_clean` (uint8, 0/255).
2. Set `N = 4*V + 17`.
3. `H = np.eye(3)`.  (rectified — no perspective distortion)
4. The image dimensions equal the QR module dimensions (since `box_size=1, border=0` internally).

**Exercise**: `bits = sample_qr_bits(img_clean, H, N)` with `threshold=None`.

**Verify**:
- `bits.shape == (N, N)`.
- Ground truth boolean grid: extract from the `qrcode` library's internal module matrix (`qr.modules` — a `list[list[bool|None]]` where `True` = black, `None` = white).  Convert to `bool` (True=white) to match our convention.
- `np.all(bits == expected_bits)` is `True`.
- OR: at least `bits.mean()` matches `expected_bits.mean()` within 0.02 (allowing minor boundary differences from bilinear interpolation).

#### Test 8: `test_sample_qr_bits_with_warp`

**Setup**: Generate a clean QR image. Apply a known perspective warp (using `cv2.getPerspectiveTransform` with known source/destination points). Compute the **inverse** of this warp as our ground-truth "homography" (grid→image).

**Exercise**: `bits = sample_qr_bits(warped_img, H_inv, N)`.

**Verify**: `bits` matches the ground-truth boolean grid (within a tolerance of 1-2% for boundary cells due to bilinear interpolation blur).

**Note**: This test may need a slightly relaxed tolerance because warping + bilinear interpolation of a sharp binary image can bleed edge pixels. We accept 98%+ bit accuracy.

#### Test 9: `test_decode_round_trip_via_opencv`  (integration)

**Setup**: For versions 1–3 and several content strings:
1. `generate_test_image(version=V, content=content)` → `img_gray` (distorted, noisy, blurry).
2. Run the full pipeline from `dev3.py`: find landmarks, estimate version, compute homography → `H_refined`, `N_best`.
3. Call `sample_qr_bits(img_gray, H_refined, N_best)` → `bits`.
4. Convert `bits` to an upscaled clean image with quiet zone (see Step G code).
5. Call `decode_qr(clean_img)` (OpenCV).

**Verify**: `decoded_text == content` for all test cases.

**Variations**: Test with rotation-only (no perspective), perspective-only, and both combined. Test with different noise levels (`noise_std` in `{30, 50, 80}`).

#### Test 10: `test_voting_weights`

**Setup**: Create a 3×3 array of grayscale values where:
- The center value is 100 (would classify as black if threshold=128)
- All 8 surrounding values are 200 (would classify as white if threshold=128)
- With equal weights: 8 white vs. 1 black → white (correct, since the cell is actually white — the center is a noise artifact)
- With center weight 2: 8 white × 1 vs. 1 black × 2 + 0 = 8 vs. 2 → white

**Exercise**: Isolate the voting logic and test it directly with a helper.

**Verify**: The center-weighted vote recovers the correct value even when the center pixel is corrupted.

---

### Running tests

```bash
cd /home/rik/git/qr-reader
python -m pytest tests/test_sample.py -v
```

Tests 1–8 and 10 are fast unit tests (no heavy pipeline). Test 9 is an integration test that exercises the full `dev3.py` flow and may take a few seconds.

---

## Summary of files touched

| File | Action | Purpose |
|---|---|---|
| `src/qr_reader/sample.py` | **Create** | 4 new functions |
| `src/qr_reader/decode.py` | **Edit** | Support `corners_xy=None` |
| `src/qr_reader/dev3.py` | **Append** | New Step G cell + replace old decode assertion |
| `tests/test_sample.py` | **Create** | 10 tests |
| `tests/__init__.py` | Possibly create | Make `tests` a package |

---

## Open questions / design notes

1. **Should we also sample the separator column and timing patterns?** — No. The adaptive threshold is computed from the finder patterns only. The timing patterns (row 6 alternating black/white) and separator are sampled during `sample_qr_bits` like any other cell. The 3×3 voting handles any alignment uncertainty there.

2. **What if the image is color?** — Currently `generate_test_image` produces grayscale. If a color image is passed, `map_coordinates` on a 3D array would need per-channel handling. For now: assume grayscale; document the constraint.

3. **Performance**: Sampling `N×N` cells with 9 projections each = `9N²` calls to `project_points`. For V=40 (N=177), that's ~282k projections. `project_points` with a `(9, 2)` input is vectorized and fast. The outer loop over all cells can be batched: project all `9N²` points in 1–2 calls instead of N² calls. The plan describes the per-cell API for clarity; the implementation can batch internally.
