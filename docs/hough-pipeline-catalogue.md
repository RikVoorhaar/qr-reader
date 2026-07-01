# QR Reader — Hough-based Finder-Pattern Edge Detection: Method, Experiments, and Failure Catalogue

## Purpose

This document describes an experimental gradient-guided Hough line detector for
locating finder-pattern boundaries in QR code images.  It records the full
algorithm, every variant and experiment attempted, and a concrete catalogue of
failure examples — **without hypothesising root causes**.  It is intended as a
briefing for an LLM or human expert to analyse from first principles and the
computer vision literature.

---

## 1. Project Context

The QR reader has a *production* detection pipeline that uses binary morphology:
Otsu binarisation → 1:1:3:1:1 ratio scanning (alignment patterns) → wave-front
region filling → connected-components boundary tracing → angular NMS corner
detection.  This pipeline works from the binary image and does **not** use edge
detection or Hough transforms.

Separately, an *experimental* Hough-based pipeline attempts to detect the eight
outer boundary edges of the three finder patterns (TL, TR, BL) directly from
the grayscale image.  The goal is to extract more precisely localised edges,
potentially improving corner estimation in high-noise or low-contrast scenarios
where the binary-morphology approach struggles.

**Coordinate conventions:**

- Detector modules: (row, col) image coordinates.
- Edge/Hough modules: (x = col, y = row) pixel coordinates.
- Decoder modules: (col, row) grid coordinates (QR spec layout).

---

## 2. Architecture of the Hough Line Detector

### 2.1 Edge extraction (`edges.py` — `extract_thin_edges`)

```
Gaussian blur (σ=1.0) → Sobel Gx, Gy → L2 magnitude → atan2(Gy, Gx) angle
→ interpolated Non-Maximum Suppression (NMS)
```

Returns `(nms, angle)` — both `(H, W)` float arrays.  `nms` contains the
thinned edge magnitude (0 where suppressed).  `angle` is the gradient-normal
angle `atan2(gy, gx)` in [-π, π], zeroed where `nms == 0`.

The NMS uses **exact gradient-direction interpolation**: for each pixel,
two opposing neighbours are bilinearly interpolated at fractional positions
along the gradient normal (step = ±1 in the dominant component, fractional in
the other).  The pixel survives if its magnitude ≥ both neighbours.

This is **not** a Canny detector — there is no dual-threshold hysteresis.
Every pixel that passes local NMS retains its Sobel magnitude.

### 2.2 Hough voting (`hough.py` — `hough_vote_peaks`)

```
NMS edge pixels → one-theta Hough voting → peak extraction with local NMS
```

For each edge pixel at (x, y):
- The gradient-normal angle θ is taken modulo π (the line is undirected).
- θ is quantised into bins of width `theta_step_deg = 2.0°`.
- ρ = x·cos(θ) + y·sin(θ) is computed with the quantised θ and binned at
  `rho_step = 1.0` px.
- Vote weight = NMS magnitude.

The accumulator is an `(n_θ × n_ρ)` array built via `np.bincount`.

Peaks are extracted iteratively:
- Find global maximum.
- Accept if ≥ `threshold_rel = 0.25` × global max.
- Suppress a rectangular neighbourhood around the peak:
  `nms_radius_theta = 3` bins (±6°), `nms_radius_rho = 6` bins (±6 px).
- Repeat up to `max_peaks = 20` times.

Returns `(normals, rhos, scores)` — shape `(K, 2)`, `(K,)`, `(K,)`.

### 2.3 Line refinement (`hough.py` — `refine_line`)

For each Hough peak `(normal, rho)`:

1. **Support collection:** all edge pixels within `distance_thresh = 1.5` px of
   the line `normal·p = rho`.
   Optionally gated by `angle_gate_deg` (see §3.2, §3.5).

2. **Weighted Total-Least-Squares (TLS) fit:** SVD on support points weighted
   by edge strength.  Produces a refined `(normal, rho)`.

3. **Segment endpoint extraction:** support points projected onto the refined
   line direction.  Sorted by projection.  The longest contiguous run with
   gaps ≤ `gap_tolerance = 2.0` px is found.  Optionally, larger gaps whose
   midpoint contains consistent-gradient NMS are bridged (see `gap_angle_gate_deg`,
   §3.2).

Returns a `LineSegment` with `normal`, `rho`, `endpoints` (2×2), `vote_score`.

### 2.4 Key default parameters

| Parameter | Default | Meaning |
|---|---|---|
| `blur_sigma` | 1.0 | Gaussian blur before edge detection |
| `theta_step_deg` | 2.0° | Angular bin size |
| `rho_step` | 1.0 px | Distance bin size |
| `threshold_rel` | 0.25 | Relative peak threshold |
| `max_peaks` | 20 | Max peaks to extract |
| `nms_radius_theta` | 3 | ±3 bins suppressed around peak (±6°) |
| `nms_radius_rho` | 6 | ±6 bins suppressed around peak (±6 px) |
| `gap_tolerance` | 2.0 px | Max gap to bridge in support run |
| `distance_thresh` | 1.5 px | Max distance from line for support pixels |

---

## 3. Experiments Attempted

### 3.1 Test Fixture

All experiments evaluated against three synthesised test cases:

| Case | Version | Difficulty | Description |
|---|---|---|---|
| v12-default | 12 | Default | Moderate rotation, perspective, noise, JPEG (seed=42) |
| v12-clean | 12 | Clean | No noise, no blur, no perspective, axis-aligned (seed=42) |
| v5-default | 5 | Default | Same difficulty parameters as v12-default (seed=123) |

Each generates an RGB image on a 640×640 synthetic gradient background via
`generate_sample(rng, config, background)`.

**Ground truth:** For each of the three finder patterns (TL, TR, BL), the
outer 7×7 module boundary edges are computed from the known `corners_qr` and
module count `N`.  Each edge has: label (e.g., "TL_top"), unit normal, rho,
and a clipped segment in the cluster ROI.  Edges whose segment does not
intersect the ROI are marked as absent (no evaluation).

**Evaluation:** For each cluster's ROI (from alignment-pattern clustering):
1. GT edges whose segment is present in the ROI are matched to Hough peaks
   (within 5° angle and 5 px rho).
2. Four failure modes are counted across matched edges and unmatched peaks
   (see §4 for definitions).
3. Phantom lines (unmatched peaks with NMS-strength support in blank regions)
   are also counted.

**Baseline fixture tallies:**

| Case | D (missing) | A (short span) | C (long span) | B (phantoms) | Total |
|---|---|---|---|---|---|
| v12-default | 2 | 2 | 4 | 5 | 13 |
| v12-clean | 0 | 0 | 0 | 0 | 0 |
| v5-default | 2 | 1 | 3 | 0 | 6 |

Full test suite baseline: 715 passed, 2 failed (v12-default, v5-default), 1 skipped.
Fixture tests live in `test_hough_harness.py`.

### 3.2 Phased experiments on `refine_line`

All phases below were applied one at a time, tested, and either accepted or
reverted.  Current production `hough.py` only retains Phases 4 and 7 (opt-in
parameters, not active in default path).

#### Phase 4 — Angle-gated support collection (ACCEPTED, opt-in)
Added parameter `angle_gate_deg` to `refine_line`.  When set, support pixels
whose gradient-normal angle differs from the Hough peak normal by more than
`angle_gate_deg` (mod π) are excluded.  Not active by default.  Tested at
15° — zero fixture impact when not passed.

#### Phase 7 — Angle-gated gap bridging (ACCEPTED, opt-in)
Added parameter `gap_angle_gate_deg` to `refine_line`.  When set, gaps
exceeding `gap_tolerance` are still bridged if NMS content at the gap midpoint
has a gradient angle consistent with the segment normal (within
`gap_angle_gate_deg`).  Not active by default.

#### Phase 5 — Rho-axis accumulator smoothing (REVERTED)
Applied a 1-D smoothing kernel (`[1,1,1]/3` or `[1,2,1]/4`) along the rho
axis of the Hough accumulator before peak extraction.  Goal: merge fragmented
votes into a single peak for D failures.
- `[1,1,1]/3` boxcar: failed parallel-line separation tests (merged lines
  3 px apart).
- `[1,2,1]/4` triangular: passed all existing tests but produced 0 D-failure
  improvements.  Reverted.

#### Phase 1 — Rho-axis smoothing in `refine_line` (REVERTED)
Smoothed support-pixel rho values (σ=1.5) before TLS fit.  Merged parallel
lines 3 px apart (finder-pattern inner/outer boundaries).  Reverted.

#### Phase 2 — Adaptive gap tolerance (REVERTED)
Added `max_gap` parameter to `refine_line` allowing larger gap bridging.
Tested at values up to 9 px.
- v12-default C failures went 4→5 at `max_gap=9` (C regressions).
- Reverted.

#### Phase 3 — Density gate (REVERTED)
Added a density-based degeneracy check: if the average gap between support
pixels exceeded 2.0 px, the line was marked degenerate.
- 0 B failures eliminated.  Reverted.

### 3.3 Phased experiments on pipeline integration

#### Phase 6 — Multi-finder spatial consistency (REVERTED)
Post-detection filter: only accept lines whose endpoints are within 15 px of
known finder-pattern GT edges.  Goal: eliminate B phantoms.
- All 5 B phantoms were within 15 px of GT edges already.
- No threshold cleanly separated them.
- Reverted.

#### Phase 8 — Harness integration of angle gates (REVERTED)
Passed `angle_gate_deg=15.0` and `gap_angle_gate_deg=20.0` to all harness
`refine_line` calls.  
- v12-clean: 0→4 C failures (regression).
- Angle gate cannot distinguish outer 7×7 boundary from inner 5×5 ring
  (same normal, different rho).
- Reverted.

### 3.4 Information-gathering investigations (I1–I9)

No production code was modified in these phases.  Each investigated a specific
hypothesis through diagnostic scripts.

#### I1 — Support displacement measurement
Quantified how far the TLS-fitted line normal and rho drift from the Hough
bin centre, and how that affects endpoint placement.

#### I2 — Phantom line analysis
Characterised the 5 B phantoms in C3 of v12-default: angular offset from GT
normals, segment span, mean NMS strength, spatial proximity to GT edges.

#### I3 — Gap classification
Classified A-failure gaps as:
- **Structural:** gap midpoint has NMS pixels at wrong angle (wrong-angle
  edge crosses the line, suppressing support).
- **Dropout:** gap midpoint has zero NMS (no edge at all in that region).

C1 TL_top: 1 structural gap (4.5 px, wrong-angle NMS at crossing 1-D edge).
C2 BL_bottom: 5 partial gaps + 1 dropout gap (3.4 px).

#### I4 — D failure pipeline impact
Investigated whether D failures actually block the production pipeline.
- Production `detector.py` does **not** use the Hough pipeline.
- v12-default: `detect_sample` fails with "No finder-pattern triplet found"
  (finder-pattern association stage, not Hough).
- v12-clean: `detect_sample` succeeds (65×65 matrix) but `decode` fails with
  RS error correction failure (separate decoder issue).
- D failures are currently non-blocking for the production pipeline.

#### I5 — NMS radius sensitivity
Tested `nms_radius_rho` at 3 and 6 for D-failure edges.
- GT rho bins had **zero votes** in both cases.
- Nearest peak was ≥6 bins away from GT bin.
- Vote fragmentation is too severe for NMS radius tuning alone.

#### I6 — Relative threshold sensitivity
Tested `threshold_rel` at 0.10, 0.15, 0.20, 0.25.
- v12-default D count unchanged (2) at all thresholds.
- Lower thresholds added phantoms: 0.15 gave +1 phantom (C3), 0.10 gave +7
  phantoms (C2 + C3).
- v12-clean: 0 phantoms at all thresholds.

#### I7 — TLS drift measurement
Measured angular drift from Hough peak normal to TLS-refined normal for all
4 C-failure edges in v12-default (C-success bucket was empty).
- Mean drift: 0.27°, max: 0.464°, min: 0.040°.
- No edge exceeded 2° drift.

#### I8 — Cluster finder pattern audit
Counted finder patterns per cluster to test the hypothesis that C3 (which
produced all 5 B phantoms) had zero finder patterns.
- C3 HAS 1 finder pattern in its ROI.
- All 4 clusters have exactly 1 finder pattern each.
- C3 has 0 GT edges whose segment intersects its ROI.

#### I9 — Endpoint trimming by strength percentile
After collecting support pixels for `refine_line`, trimmed the weakest
support pixels (lowest NMS magnitude) from each end, in percentile steps from
5 to 50.  Re-ran endpoint extraction on trimmed support set.
- v12-default: 1/4 C failures fixed at 40th percentile (trimming 40% of
  weakest support pixels per edge).  No improvement at lower percentiles.
- v5-default: 0/3 C failures fixed at any percentile.
- Effect is marginal: fixes 1 of 7 total C failures across test cases, and
  only at an aggressive threshold.

### 3.5 Alternative edge detectors (I10)

#### Canny edge detection + our Hough pipeline
Replaced `extract_thin_edges` with a Canny-based edge extractor:
- Same Gaussian blur (σ=1.0) → `cv2.Canny(low, high)` → mask Sobel magnitude
  and angle with Canny binary output.
- Fed into unchanged `hough_vote_peaks` + `refine_line`.
- Swept 9 Canny threshold pairs from (20,60) to (150,300).

**Best result (v12-default):** L=70, H=210 → D=2, A=0, C=3, B=5 → total=10
(baseline 13, improvement of -3).  
- A failures eliminated entirely (2→0): Canny hysteresis preserves continuous
  finder boundary edges that Sobel NMS fragments.
- 1 C failure fixed (4→3).
- D failures unchanged (2→2).
- B phantoms unchanged at best setting (5→5), worse at lower thresholds
  (up to +3).
- v12-clean: no regression (0→0).
- v5-default: no net improvement (6→6) but D improved 2→1, C worsened 3→4.

**Conclusion:** Canny is a modest improvement over the Sobel-NMS approach.
It eliminates A failures (the most actionable failure mode) without regressing
clean images.  It does not affect D failures.

#### OpenCV HoughLinesP (Canny → probabilistic Hough)
Used `cv2.Canny → cv2.HoughLinesP` as a completely different line detection
pipeline (no gradient-guided voting, no TLS refinement).  Mapped
HoughLinesP output to our normal/rho convention.  Swept 7 parameter sets.

**Best result (v12-default):** thresh=30, minLen=15, gap=3 → D=2, A=0, C=4,
B=0 → total=6 (baseline 13, improvement of -7).  
- B phantoms eliminated entirely (5→0).
- A failures eliminated (2→0).
- C failures unchanged (4→4).
- D failures unchanged (2→2).
- v12-clean: regression (0→6, all C).
- v5-default: D=3, A=0, C=3, B=0 → total=6 (same total but shifted from D=2
  to D=3).

**Conclusion:** HoughLinesP eliminates phantoms but creates over-long
segments on clean images (20×20 px overshoot).  Not suitable for precision
finder-pattern edge detection.

### 3.6 Summary of all improvement attempts

| Approach | v12-default best Δ | v12-clean Δ | v5-default best Δ | Notes |
|---|---|---|---|---|
| Smoothing (Phase 1, 5) | 0 | 0 | 0 | Merges parallel lines |
| Gap tolerance (Phase 2) | -1 | recovery cost | — | C regressions |
| Density gate (Phase 3) | 0 | 0 | 0 | No B elimination |
| Angle gate (Phase 4, 7) | 0 (opt-in) | 0 (opt-in) | 0 (opt-in) | In use but not active |
| Multi-FP consistency (Phase 6) | 0 | 0 | 0 | Phantoms already near GT |
| Angle gate in harness (Phase 8) | -1 | +4 regression | -1 | Reverted |
| Endpoint trimming (I9) | -1 (partial) | 0 | 0 | 1/7 C failures fixed |
| Canny edges (I10) | -3 | 0 | 0 | Eliminates A failures |
| HoughLinesP (I10) | -7 | +6 regression | 0 | Eliminates B, creates C |

---

## 4. Failure Mode Catalogue

All examples below are from the **v12-default** fixture (seed=42), version 12
QR code at default difficulty.  The test generates 4 cluster ROIs, each
containing one finder pattern.  GT finder edges are computed per-cluster.

### 4.1 Failure D — Edge with matching Hough peak missing

**Definition:** A GT finder-pattern edge has **no Hough peak** within 5°
angular distance and 5 px rho distance.  Either no peak exists in the
accumulator at that location, or the peak was suppressed by NMS.

**V12-default count: 2**

| Edge | Cluster | GT normal (unit) | GT rho (px) | Hough peaks at nearest bin | Notes |
|---|---|---|---|---|---|
| TL_left | C1 | (-0.570, -0.822) → θ≈145.7° | 24.3 | **0 peaks within 5°/5px** | Closest peak at bin offset ≥6 |
| BL_left | C2 | (-0.568, -0.823) → θ≈145.7° | 22.5 | **0 peaks within 5°/5px** | Closest peak at bin offset ≥6 |

**Accumulator analysis (I5):** For both edges, the GT θ bin has **zero votes**
in the GT rho bin.  Non-zero votes spread across adjacent rho bins:
- C1 TL_left: votes at rho bins 9, 13, 14, 18, 19 (GT = bin ~24).
- C2 BL_left: votes at rho bins 7, 11, 12, 16 (GT = bin ~23).

These edges cannot be recovered by changing `nms_radius_rho` or
`threshold_rel` — the GT bin itself is empty, so no peak can be extracted
from it.

### 4.2 Failure A — Refined segment span < 80% of GT span

**Definition:** A GT edge has a matching Hough peak, but the refined segment's
projected span is less than 80% of the GT edge's projected span.

**V12-default count: 2**

| Edge | Cluster | GT span (px) | Refined span (px) | Ratio | Gap details |
|---|---|---|---|---|---|
| TL_top | C1 | ~61 | ~26 | 0.42 | 1 structural gap of 4.5 px (I3) |
| BL_bottom | C2 | ~41 | ~20 | 0.49 | 5 partial gaps + 1 dropout gap of 3.4 px (I3) |

**Support pixel data:**
- C1 TL_top: `refine_line` collects support pixels within 1.5 px of the Hough
  peak line.  The support set is split into two separated groups, with a
  4.5 px gap in projection space.  The gap midpoint contains NMS pixels at a
  gradient-normal angle not parallel to the finder boundary (wrong-angle
  crossing edge from a 1-D module boundary).
- C2 BL_bottom: support pixels form two groups separated by 3.4 px.  The gap
  midpoint has **zero NMS** (no edge at all — dropout).  Five additional
  smaller gaps (1.5–2.0 px) fragment the support into even shorter runs.

### 4.3 Failure C — Refined segment endpoints too far from GT endpoints

**Definition:** A GT edge has a matching Hough peak, but **neither** refined
endpoint is within 5 px of either GT endpoint.  This typically corresponds to
the refined segment overshooting beyond the finder pattern boundary into the
QR data region or interior.

**V12-default count: 4**

| Edge | Cluster | GT endpoints (roi-local) | Refined endpoints | Endpoint distance (px) |
|---|---|---|---|---|
| TR_top | C0 | (22,18)→(87,12) | (20,18)→(89,14) | 2.0 (short), 3.6 (short) |
| TR_right | C0 | (87,12)→(93,68) | (89,13)→(97,72) | 2.2, 5.5 (overshoots) |
| TL_top | C1 | (49,20)→(108,54) | (45,14)→(103,62) | 6.7 (overshoots), 9.6 (overshoots) |
| BL_bottom | C2 | (48,40)→(87,64) | (45,38)→(92,67) | 3.6, 5.1 (overshoots) |

**TLS drift analysis (I7):** For these 4 edges, the angular drift from Hough
peak normal to TLS-refined normal was measured:
- Mean: 0.27°
- Max: 0.464°
- Min: 0.040°

The refined normal direction is very close to the Hough peak direction.  The
overshoot corresponds to the support-pixel set extending beyond the GT endpoints,
while the TLS-refined normal remains very close to the Hough peak direction.

**Proximity of finder structures:** The outer 7×7 finder-pattern boundary is
adjacent to the inner 5×5 dark/light ring (within 3 px at ppm≈10).  Both
boundaries produce edges with the same gradient-normal direction but at
different rho distances.

### 4.4 Failure B — Phantom lines in blank/data regions

**Definition:** A Hough peak does **not** match any GT finder-pattern edge (by
5°/5px), its normal is >12° from all GT normals (parallel module edges are
excluded), and its `refine_line` support pixels have mean NMS strength
exceeding 400 (signifying real edge structure, not noise).

**V12-default count: 5** (all in cluster C3)

| Peak | Normal | Rho (px) | Mean NMS | Support pixels | Angular offset from GT normals |
|---|---|---|---|---|---|
| C3-p0 | (-0.579, -0.815) ≈ 144.5° | 23.8 | ~650 | ~15 | 12.3° |
| C3-p1 | (-0.581, -0.814) ≈ 144.3° | 21.5 | ~580 | ~12 | 12.5° |
| C3-p2 | (-0.578, -0.816) ≈ 144.7° | 20.2 | ~510 | ~10 | 12.1° |
| C3-p3 | (-0.580, -0.815) ≈ 144.4° | 19.0 | ~490 | ~9 | 12.4° |
| C3-p4 | (-0.579, -0.815) ≈ 144.5° | 17.8 | ~470 | ~8 | 12.3° |

**Spatial context (I2):** All 5 phantoms are within 15 px of GT finder edges
in the image coordinate space.  Their normals are all ~12° offset from the
nearest GT normal, with rho values forming a regular progression (17.8 to 23.8,
step ~1.7 px).

**Cluster context (I8):** C3 HAS 1 finder pattern.  C3 has **0 GT edges** whose
segment intersects the ROI.  The cluster bounding box (from alignment-pattern
clustering) is positioned such that no finder-pattern boundary falls within it,
but QR data-region module edges at similar orientations do.

---

## 5. Additional Context

### 5.1 ROI sizes

At the PPM range of the v12-default fixture (target 4–10 PPM), each finder
pattern's 7×7 module boundary is roughly 35–70 px across.  The cluster ROI
(from `cluster_to_bbox(scale=1.5)`) is typically 80–120 px wide.  Edge
segments in these ROIs are short (40–80 px) with limited support pixels
(15–30 pixels within `distance_thresh`).

### 5.2 Image pipeline details

The synthetic pipeline (`qr_gen.generate_sample` → `synth.pipeline.generate_sample`):

1. Generate clean QR code at specified version, module size = floor(PPM).
2. Optional rotation (uniform from range).
3. Perspective warp via random 4-corner jitter.
4. Scale to target PPM via affine transform.
5. Gaussian blur (σ from range).
6. Gaussian noise (σ from range).
7. JPEG compression (quality from range).
8. Composite onto background.

The v12-default fixture uses ranges: PPM 5–12, target 4–10, jitter 0.15,
blur 0.2–1.0, noise 1.0–5.0, JPEG 65–95.

### 5.3 The Hough pipeline is not in the production detection path

`detector/detector.py` uses the binary-morphology corner-finding pipeline
described in §1.  The Hough pipeline in `detector/hough.py` is **not imported
or called** from `detector.py`.  It exists only in diagnostic scripts and test
harnesses.  The production pipeline succeeds on v12-clean (extracting a 65×65
module matrix) but the decoder's RS error correction fails for version 12 ECL M
(separate issue).

### 5.4 OpenCV availability

- OpenCV 4.13.0 is available for Canny and basic Hough transforms.
- `cv2.ximgproc` (contrib module) is **not available** — Fast Line Detector
  and Line Segment Detector cannot be used.

### 5.5 Test suite

- Isolation tests: `test_hough.py` — 25 tests for A and C isolation scenarios.
- Fixture tests: `test_hough_harness.py` — 9 tests (3 configs × 3 assertion
  phases) plus unit tests for GT edge geometry.
- Full suite: `pytest src/qr_reader/tests/` — 715 passed, 2 failed (v12-default,
  v5-default), 1 skipped.
- Unit tests for individual Hough/edge functions are in `test_hough.py`.

---

## 6. Relevant Files

| File | Purpose |
|---|---|
| `src/qr_reader/detector/edges.py` | Edge extraction (Sobel + NMS) — 126 lines |
| `src/qr_reader/detector/hough.py` | Hough voting + peak extraction + line refinement — 430 lines |
| `src/qr_reader/tests/detector/test_hough_harness.py` | Fixture tests, GT edge computation, assertion helpers — 1039 lines |
| `src/qr_reader/tests/detector/test_hough.py` | Isolation tests for A and C — 25 tests |
| `src/qr_reader/detector/roi.py` | Cluster ROI extraction — 65 lines |
| `src/qr_reader/detector/alignment.py` | Alignment pattern detection (1:1:3:1:1 ratio scanning) |
| `src/qr_reader/detector/clustering.py` | Alignment pattern clustering |
| `src/qr_reader/scripts/phase_i10_alternative_edge_detectors.py` | Canny and HoughLinesP comparison |
| `src/qr_reader/scripts/phase_i9_endpoint_trimming.py` | Endpoint trimming experiment |
| `src/qr_reader/scripts/phase_i7_tls_drift.py` | TLS drift measurement |
| `src/qr_reader/scripts/phase_i5_nms_radius.py` | NMS radius analysis |
| `src/qr_reader/scripts/phase_i6_threshold.py` | Threshold sensitivity |
| `src/qr_reader/scripts/phase_i8_cluster_audit.py` | Cluster finder pattern audit |
