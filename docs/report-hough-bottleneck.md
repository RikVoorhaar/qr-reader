# QR Finder Edge Detection — Pipeline, Experiments, and Bottleneck

> A detailed technical report for estimating literature relevance to the
> current bottleneck.  Describes the full Hough-based finder-pattern edge
> detection pipeline, parameter sweeps to improve recall, and strong evidence
> that vote-quantization is the limiting factor.

---

## 1. What we are trying to do

Given an RGB image of a QR code (version 12, 640×640 px, with realistic
synthetic augmentations: random rotation, sub-module-pitch jitter, aspect
distortion, feathering, Gaussian noise, blur, and JPEG compression), we want
to detect **lexact finder-pattern edge segments**.  A QR finder pattern is a
7×7 module square with a 1∶1∶3∶1∶1 black-white-black-white-black concentric
structure.  Each finder has 4 cardinal sides (top, bottom, left, right), and
each side has 3 module-boundary edges (the outer boundary at module offset
k=0, the dark-light ring transition at k=1, the light-dark ring transition at
k=2 — and symmetrically at k=5,6,7).  That gives **12 edge segments per
finder pattern** (36 total across the three finder patterns TL, TR, BL).

The task: detect the **infinite line** (θ, ρ) and the **clipped segment
endpoints** for each of these 36 GT edge segments, using gradient-guided Hough
voting followed by weighted TLS refinement on a per-finder ROI cutout.

---

## 2. Full pipeline

### 2.1 Image synthesis

QR images are synthesized with `qrcode` + OpenCV augmentation:

1. Generate clean QR modules → place on sub-pixel grid at target PPM
2. Apply random rotation (0–360°)
3. Perspective warp via corner jitter (±15% module-pitch fraction)
4. Bilinear downscale to target image resolution
5. Poisson-disk feathering to blend QR onto a smooth gradient background
6. Add per-pixel Gaussian noise (σ = 1–5 intensity units)
7. Gaussian blur (σ = 0.2–1.0 px)
8. JPEG compression (quality 65–95)

Metadata includes the **4 image-space QR corners** `(x, y)` and the module
count `N = 17 + 4·version`.

### 2.2 ROI extraction

For each candidate cluster (detected via 1D alignment-pattern scanning along
rows and columns of the binarized image, then clustered into `CandidateCluster`
groups), we compute a padded bounding box:

```
center_row = cluster.row
center_col = mean(cluster.cols[2], cluster.cols[3])
half_extent = 1.5 × max(width, height) / 2
bbox = (center_row ± half_extent, center_col ± half_extent)
```

The bounding box is clamped to image bounds and the grayscale cutout is
extracted as the ROI.  Each ROI is typically 100–200 px wide/tall.

### 2.3 Edge extraction

On each ROI we run `extract_thin_edges`:

```
Gaussian blur (σ = 1.0) → Sobel 3×3 → L2 magnitude →
interpolated non-maximum suppression (NMS)
```

The output is a `(H, W)` pair `(nms, angle)`:
- `nms[y,x]` = float64 edge magnitude after NMS (0 for non-edges)
- `angle[y,x]` = `atan2(gy, gx)` in [-π, π] (edge-normal direction)

Border pixels (1 px margin) are zeroed.

### 2.4 Gradient-guided Hough voting

For each edge pixel `(x, y)` with magnitude `s = nms[y,x] > 0`:

1. Compute edge-normal angle `θ = angle[y,x] mod π` in [0, π)
2. Quantize to theta bin `t_idx = round(θ / θ_step) mod n_theta`
   (`θ_step = 0.5°`, `n_theta = 360`)
3. Compute ρ = x·cos(θ) + y·sin(θ)
4. Quantize to rho bin `r_idx = round(ρ / ρ_step)`
   (`ρ_step = 0.5 px`, `n_ρ ≈ ceil(hypot(W,H) / ρ_step)`)
5. Cast vote: `acc[t_idx, r_idx] += s`

Voting is `onebin` (each pixel votes into exactly one (θ, ρ) bin).

### 2.5 Peak extraction

Iterative peak NMS on the accumulator:

```
threshold = threshold_rel × acc.max()    (threshold_rel = 0.15)
for _ in range(max_peaks):               (max_peaks = 20)
    idx = argmax(acc)
    if acc[idx] < threshold: break
    record peak (θ, ρ, score)
    zero out a (2×nms_radius_θ+1) × (2×nms_radius_ρ+1) patch around the peak
    (nms_radius = 5, i.e., ±5 bins = ±2.5° × ±2.5 px)
```

Output: up to 20 `(normal, rho, score)` triplets sorted by score descending.

### 2.6 Line refinement (weighted TLS + support run)

For each Hough peak `(n, ρ, score)`:

1. Collect edge pixels within `distance_thresh = 1.5 px` of the candidate
   line and within `angle_gate_deg` of the peak normal.
2. Fit a weighted total-least-squares line to these inlier pixels.
3. Project all inliers onto the refined line direction.
4. Find the longest contiguous 1D support run with `gap_tolerance = 3 px`
   gap-bridging.
5. The endpoints of this support run (projected back to 2D) become the
   detected segment `endpoints`.

Output: `LineSegment(normal, rho, endpoints, vote_score)`.

### 2.7 GT edge geometry

GT edges are computed via a module-grid homography:

1. Estimate `H` via normalized DLT from QR-grid corners `(0,0), (N,0),
   (N,N), (0,N)` → image-space corners.
2. For each finder at grid position `TL=(0,0)`, `TR=(0,N-7)`, `BL=(N-7,0)`:
   - "top"/"bot" sides (horizontal in grid): for k ∈ {0,1,2,5,6,7}, with
     `k_vis = min(k, 7-k)`, segment from `(r₀+k, c₀+k_vis)` to
     `(r₀+k, c₀+7-k_vis)` in grid coords.
   - "left"/"right" sides (vertical): similar logic with rows and cols
     swapped.
3. Project each grid endpoint through `H` → image-space (x,y) coords.
4. Translate to ROI-local coords and clip via Cohen-Sutherland to the ROI
   rectangle.  If the line does not intersect the ROI, the segment is `None`
   (the edge is not evaluated).

Inner segments are clipped to the visible feature span:
- Outer boundary (k=0, k=7): full 7-module width
- Ring transition (k=1, k=6): 5 modules (skip corners)
- Centre square transition (k=2, k=5): 3 modules

### 2.8 Evaluation metrics

**Peak-level** (per GT edge): Does a Hough peak exist within 5° angle × 5 px
ρ of the GT `(normal, rho)`?  TP/FN/FP counts, linear precision/recall/F1.

**Segment-level** (per GT edge with peak match): Refine the matched peak to a
`LineSegment`, then compute **1-D interval IoU** by projecting both the GT
segment and the detected segment endpoints onto the GT line direction:
```
t_gt = sort(endpoints_gt · direction_gt)
t_det = sort(endpoints_det · direction_gt)
inter = max(0, min(t_gt[1], t_det[1]) - max(t_gt[0], t_det[0]))
union = max(t_gt[1], t_det[1]) - min(t_gt[0], t_det[0])
iou_1d = inter / union
```
A detected segment is a TP if `iou_1d ≥ 0.3`.

Additional per-TP metrics: coverage_gt (recall = inter / gt_span),
coverage_seg (precision = inter / det_span), endpoint error (max distance
from each detected endpoint to the nearest GT endpoint), lateral error
(perpendicular distance from detected line to GT line).

**Per-finder/side/k-group breakdowns** are also computed.

---

## 3. Baselines and improvements

### 3.1 Starting point (E6 best from prior sweeps)

Config: `θ_step=0.5°`, `ρ_step=1.0`, `NMS=2×2`, `threshold=0.25`

| Case | Peak recall | Seg recall |
|------|------------|------------|
| v12-clean | 1.000 | 1.000 |
| v12-default | 0.806 | 0.811 |
| v5-default | 0.833 | 0.833 |

All 34 missed edges across 5 seeds of v12-default were **"left" edges** —
the inner vertical edges of finder patterns that face toward the QR centre.

### 3.2 Fine-grained rho quantisation (ρ_step sweep)

Swept `ρ_step ∈ {0.5, 1.0, 2.0}` and `acc_smooth ∈ {none, 1×3 triangular, 1×5 triangular}`.

**Result**: `ρ_step=0.5` improved recall to 0.833 (+2.7pp).  Finer ρ
quantisation aligns edge-pixel ρ projections more precisely with GT ρ bins.
`acc_smooth` was strictly harmful at all settings — it polls legitimate
near-by votes into the GT bin but also raises the noise floor, lowering all
SNRs.

### 3.3 Angular quantisation (θ_step sweep)

Swept `θ_step ∈ {0.25°, 0.5°, 1.0°, 2.0°}`.

**Result**: `0.5°` confirmed optimal.  `0.25°` caused ~1pp regression — edge
direction jitter from image noise scatters votes across 2+ theta bins, so
finer bins don't help.

### 3.4 NMS radius

Swept `nms_radius_rho × nms_radius_theta ∈ {1,2,3}²`.

**Result**: Larger radius **improved** recall.  `3×3` reached 0.856 (+2.3pp
over 2×2).  This is counter-intuitive (larger suppression should lose more
weak peaks), but in practice larger NMS suppresses near-miss phantom peaks
that would otherwise fill the `max_peaks=20` budget, leaving room for genuine
peaks.

### 3.5 Peak threshold

Swept `threshold_rel ∈ {0.10, 0.15, 0.20, 0.25}`.

**Result**: `0.15` reached 0.917 (+6.1pp over 0.25).  This was the single
biggest gain.  Many weak left-edge peaks have accumulator values near 10–15%
of the global `acc_max`.  At 0.10 the same recall is achieved but with ~40
more phantom peaks, so 0.15 is the precision/recall sweet spot.

### 3.6 Edge preprocessing (blur, hysteresis)

Swept `blur_sigma ∈ {0.5, 1.0, 1.5, 2.0, 2.5}`; tested hysteresis
thresholding (`_hysteresis_link` with various high/low percentiles).

**Result**: Zero effect from blur sigma (identical 0.917 recall at all
settings).  Hysteresis drops 54–82% of NMS edge pixels (only strong edges +
weak ones 8-connected to strong ones survive), so it was not tested further.

### 3.7 Final combinatorial push

Best config from individual sweeps tried with NMS 4×4, 5×5, 6×6, combined
with threshold 0.10–0.15 and max_peaks 20–30.

**Result**: NMS=5×5 + threshold=0.15 + max_peaks=20 reached **0.928**
(+1.1pp over NMS=3×3).  Further increases (NMS=6, threshold=0.10, max_peaks=30)
all gave **exactly the same 0.928 recall** — this is a hard ceiling.  The
remaining 13 FP across all 15 combinations were identical 13 edge instances
(not parameter-dependent).

### 3.8 Summary of gains

| Phase | Change | Peak recall |
|-------|--------|-------------|
| E6 baseline | — | 0.806 |
| P1: ρ_step | 1.0 → 0.5 | 0.833 |
| P3: NMS | 2×2 → 3×3 | 0.856 |
| P4: threshold | 0.25 → 0.15 | 0.917 |
| P7: NMS + combos | 3×3 → 5×5 | **0.928** |

**Total improvement: +12.2 percentage points.**

---

## 4. Evidence for vote-quantisation as the remaining bottleneck

### 4.1 What the 13 remaining misses look like

Across v12-default's 5 seeds (180 total GT edges), the 13 misses are **not
systematic** — each missed edge label appears in exactly 1 out of 5 seeds.
No single label consistently fails.  The misses span all finders (TL, TR, BL)
and all sides (top, bot, left, right).

**Crucially**, every missed edge has a **well-formed GT segment** (length
10–46 px, properly clipped in the ROI).  The edge pixels DO exist:

```
Example missed edges (seed=42):
  TR_left0: 14 edge pixels within 2px of GT line, 2 with correct angle
  TR_left2: 14 edge pixels within 2px of GT line, 2 with correct angle
  TL_left0: 19 edge pixels within 2px of GT line, 3 with correct angle
```

But the **GT rho bin is empty** in the accumulator:

```
  TR_left0: acc[θ_gt, ρ_gt] = 0
            BUT acc[θ_gt, *] has 1188 total votes (acc_max_theta = 1188)
            → votes go into bins adjacent to GT ρ, not the GT bin itself
```

Edge pixels are detected at the correct angle (±5°) but their `(x,y)`
positions jitter by ±0.5–1.0 px from noise, causing their ρ projections to
shift by ±0.5–1.0 px.  With `ρ_step=0.5`, this means votes land in bins ±1
or ±2 away from the GT bin, which itself receives **zero votes**.

### 4.2 Why lowering the threshold doesn't help

Lowering `threshold_rel` from 0.15 to 0.08 lets ALL accumulator bins above
8% of `acc_max` through as peaks.  But the GT bin value is **0**, so no peak
is detected there regardless of threshold.

### 4.3 Why accumulator smoothing doesn't help

`acc_smooth="1×3_triangular"` convolves each rho row with a (1,2,1)/4 kernel.
This pools votes from adjacent bins into the GT bin.  But it also pools
noise, reducing the peak-to-background ratio across the entire accumulator.
Empirically, smoothing reduced recall from 0.833 to 0.811.

### 4.4 Why larger NMS doesn't help

Larger NMS (5×5 or 6×6) zeroes out a wider neighborhood around each detected
peak.  If the GT peak's votes are spread across ±2 bins in rho, a larger NMS
radius could theoretically suck them into the zeroed region.  But
empirically, NMS=5×5 gave the same 0.928 recall as NMS=4×4 and 6×6 — the
bin-zeroing region already covers the vote spread, so further enlargement
does nothing.

### 4.5 The fundamental problem

The Hough transform is a **discrete**, **bin-quantised** voting scheme.  Each
edge pixel `(x, y, θ)` maps to exactly one `(θ_idx, ρ_idx)` bin via
`round()`.  For a clean, noise-free image, all edge pixels along a straight
line vote into the same bin.  For a noisy image with `σ_noise ≈ 2–5
intensity units`, edge positions jitter by ≈0.3–0.7 px (estimated from the
Gaussian-noise-to-edge-position propagation through the Sobel gradient), and
edge-normal angles jitter by ≈0.3–0.5°.

This jitter means that edge pixels nominally belonging to a single line cast
their votes into **different** `(θ, ρ)` bins.  The vote for the true line is
**fragmented** across a small cluster of bins.  The strongest single bin
might contain only 60–80% of the total vote for that line.  The GT bin (the
theoretical bin for the unfragmented vote) might contain **zero** votes if
the mean position offset doesn't align with any integer ρ_step.

**The Hough peak detector as implemented assumes each line maps to a single
dominant accumulator bin.**  When votes are fragmented, the single-bin peak
is weaker than the true line's total vote, and a different (noisier) bin in
the same neighbourhood might have the maximal accumulator value — or no bin
might exceed the threshold at all.

---

## 5. Questions for literature

We're looking for prior work on:

1. **Sub-bin peak localisation in Hough accumulators**:  How to detect a
   peak when the votes for a single line are spread across multiple adjacent
   bins.  This could involve:
   - Parabolic / quadratic interpolation of the accumulator surface around
     local maxima (the standard sub-pixel peak refinement)
   - Detecting peaks as the centroid of a vote cluster rather than a single
     bin maximum
   - Multi-bin thresholding that sums votes across a small window before
     applying a threshold

2. **Vote-weighting schemes** that make the accumulator robust to edge-pixel
   position noise:
   - Soft voting (e.g., bilinear interpolation of the vote across the 4
     nearest bins in (θ, ρ) space)
   - Edge-strength-weighted voting where the weight depends on the local edge
     confidence
   - Gradient-direction-uncertainty-aware voting that spreads the vote
     proportionally to the expected angular error

3. **Alternative line representations** that don't suffer from the
   fragmentation problem:
   - Probabilistic Hough transforms that model the vote as a distribution
     rather than a point
   - Kernel-based Hough transforms with explicit rho/theta smoothing built
     into the vote function
   - Direct line fitting to edge pixels (RANSAC, LSD) as an alternative to
     Hough voting

4. **Edge detection under noise** specifically for man-made straight-edge
   structures:
   - Methods that recover weak edge pixels by exploiting the known rectilinear
     structure of the target
   - Edge-preserving denoising before gradient computation

We are not looking for deep-learning approaches (the project is a
from-scratch algorithmic QR reader).  Classical computer vision / geometric
methods preferred.

---

## 6. Key numbers for context

- **Image size**: 640 × 640 px RGB
- **QR version**: 5, 12 (tested; 12 is primary target)
- **Module count N**: 21 (v5), 45 (v12)
- **Typical ROI size**: 100–200 px per side (1.5× cluster padding)
- **Total GT edges**: 36 per image (12 per finder × 3 finders)
- **Edge pixels per finder ROI**: 500–1000 (after NMS)
- **Hough accumulator**: 360 θ bins × ~200–400 ρ bins = ~72k–144k cells
- **Runtime**: ~5 ms per cluster (Hough vote + peak + refine)

## 7. Parameter space searched

| Parameter | Values tried | Best |
|-----------|-------------|------|
| `theta_step_deg` | 0.25, 0.5, 1.0, 2.0 | 0.5° |
| `rho_step` | 0.5, 1.0, 2.0 | 0.5 px |
| `nms_radius_theta` | 1–6 | 5 bins |
| `nms_radius_rho` | 1–6 | 5 bins |
| `threshold_rel` | 0.08–0.25 | 0.15 |
| `max_peaks` | 20, 30 | 20 |
| `acc_smooth` | none, 1×3, 1×5 | none |
| `blur_sigma` | 0.5–2.5 | 1.0 |
| `vote_scheme` | onebin, gaussian, dot | onebin |
| `theta_window_deg` | 0, 1, 2, 5 | 0 |

Total: ~70 unique configurations across 6 phases of sweeps.
