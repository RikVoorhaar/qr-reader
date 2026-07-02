# Plan 013 — Integrate `finder_fit` into the Detection Pipeline

> **Goal:** Replace the Region→Boundary→NMS corner extraction stage with
> `finder_fit`, producing `FinderPattern` objects with higher accuracy, then
> simplify the downstream version-estimation and homography pipeline to take
> advantage of the richer fitted geometry.

## Motivation

`finder_fit` (Plan 012) produces 4 outer corners per finder pattern at
~3–4 px mean error on v12-default and ~2.8 px on v12-clean. The current
boundary-NMS corner extraction is noisier (boundaries drift on blurred/noisy
images) and produces no module-pitch or orientation estimate.

Once we have 3 fitted finders with accurate corners AND module pitch, we
don't need the full landmark → cross-ratio → version → RANSAC chain.
We can:

1. Estimate the version **directly** from inter-finder distances
2. Fit a homography **directly** from 12 corner-to-grid correspondences (4
   corners × 3 finders)
3. Eliminate the alignment-pattern dependency for version estimation

The plan proceeds in 3 phases, each independently testable.

## Out of scope

- Removing `find_alignment_patterns_2d` / `cluster_candidates` — we still
  need them for finder candidate localisation (where to run edge extraction).
- Multi-QR detection.
- Perspective-skewed finder handling (e1 ⊥ e2 is assumed orthogonal; the
  product code already handles skew downstream).
- Phase 4 template fitting (still regresses vs Phase 3, Plan 012 result).

## Architecture

### Phase 1: Minimal integration (`fit_finder_full` → `FinderPattern`)

Replace only the corner-extraction step (steps 4–5 of `_run_detection`) while
keeping the associator → triplet → landmark → version → RANSAC chain intact.

```
Old:  cluster → region_fill → boundary → NMS corners → FinderPattern → ...
New:  cluster → thin_edges → fit_finder_full → corners_rc → FinderPattern → ...
```

**Changes to `detector.py`:**

1. Add imports for `edges.extract_thin_edges`, `roi.cluster_to_bbox`,
   `roi.cutout`, `finder_fit.fit_finder_full`.
2. In `_run_detection`, after `cluster_candidates`:
   ```python
   fps = []
   for ci, cluster in enumerate(clusters):
       bbox = cluster_to_bbox(cluster, scale=1.5)
       roi = cutout(img_gray, bbox)
       if roi.size == 0:
           continue
       nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
       r0_off, c0_off = bbox[0], bbox[2]
       c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0_off
       c_row = float(cluster.row) - r0_off
       center_xy = np.array([c_col, c_row])
       m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

       fit = fit_finder_full(nms, angle, roi, center_xy, m_est)

       corners_xy = fit.corners + np.array([c0_off, r0_off])  # global (x, y)
       corners_rc = corners_xy[:, ::-1]                        # (row, col)
       fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
   ```
3. Skip the old `all_corners` + `extract_finder_patterns` path.
4. Feed `fps` into `find_all_associations(fps)` → `find_triplets(fps, associations)`.
5. Rest of the pipeline unchanged.

**What stays the same:**
- `find_all_associations`, `find_triplets`, `build_named_landmarks`,
  `build_constraints`, `filter_constraints`, `estimate_version`,
  `ransac_homography`, `refine_homography_lm`, `compute_qr_corners`.

**What changes:**
- `extract_finder_patterns(all_corners)` is replaced by direct
  `FinderPattern` construction from `finder_fit` results.
- `inner_corners` is `None` (we only produce outer corners; the inner ring
  is implicit in the 1:1:3:1:1 model).
- `NamedLandmarks.B`, `.D`, `.F` will be `None` → colinear quadruples
  will have only outer-edge constraints (4 quads instead of 8). Version
  estimation should still work — outer-only quadruples span the full QR
  code.

**Print:** pipeline trace: finder count, association count, triplet found?,
V_est, GT V, corner reprojection error.

**Success criteria:**
- Pipeline runs end-to-end with correct version on v12-clean and
  v12-default.
- Homography corner error ≤ current-pipeline error (qualitative check).

### Phase 2: Direct version + homography from fitted finders

Once Phase 1 proves `finder_fit` corners work in the pipeline, simplify
the downstream path.  We have:

```
fps → find_associations → find_triplets → triplet
```

From the triplet, directly compute:

**Version estimation:**
```python
c_tl = fps[tl_idx].outer_corners.mean(axis=0)  # (row, col) center
c_tr = fps[tr_idx].outer_corners.mean(axis=0)
c_bl = fps[bl_idx].outer_corners.mean(axis=0)

# Module pitch from finder fit (average across 3 finders)
m = mean([fit_results[i].m for i in (tl_idx, tr_idx, bl_idx)])

N_est_x = |c_tr[1] - c_tl[1]| / m + 7  # cols → x, horizontal
N_est_y = |c_bl[0] - c_tl[0]| / m + 7  # rows → y, vertical
N_est = int(round((N_est_x + N_est_y) / 2.0))
V_est = (N_est - 17) // 4
```
Validate via format-info decode if downstream decoder is available.

**Homography:**
```python
src_xy = []  # grid coordinates
dst_xy = []  # image coordinates

for finder, (r0, c0) in [("TL", (0, 0)), ("TR", (0, N-7)), ("BL", (N-7, 0))]:
    corners_image = fit_results[finder].corners  # (4, 2) global (x, y)
    corners_grid = [
        (c0,     r0),     # top-left of finder
        (c0 + 7, r0),     # top-right
        (c0 + 7, r0 + 7), # bottom-right
        (c0,     r0 + 7), # bottom-left
    ]
    src_xy.extend(corners_grid)
    dst_xy.extend(corners_image)

H = estimate_homography_dlt(np.array(src_xy), np.array(dst_xy))
# LM refinement with 12 correspondences
H = refine_homography_lm(H, src_xy, dst_xy)
```

12 point correspondences → well-conditioned DLT, no RANSAC needed (RANSAC
was compensating for boundary-noise outlier corners, which `finder_fit`
doesn't produce).

**What is eliminated:**
- `build_named_landmarks` + `canonical_grid_landmarks`
- `build_constraints` + `filter_constraints` + `estimate_version`
- `ransac_homography` (replaced by single DLT on 12 clean correspondences)
- `rc_to_xy` manual coordinate juggling (corners are in (x,y) from fitter)

**What remains:**
- `compute_qr_corners` (QR corners from H)
- `sample_qr_bits` (bit sampling unchanged)
- `refine_homography_lm` (LM refinement on all 12 points)

**Risk — version estimation from geometry:**  `N_est` is sensitive to `m`
accuracy. If `m_fit` is wrong by 15% and inter-finder distance is ~200 px
(v12), N_est could be off by 2–3.  **Mitigation:** try N_est, N_est ± 1,
N_est ± 2; sample modules from each homography; the correct N produces
cleaner samples (lower variance per sample bin).

**Success criteria:**
- `V_est == GT_version` on v12-clean and v12-default.
- Homography corner error ≤ Phase 1 error.

### Phase 3: End-to-end decode

Wire the Phase 2 pipeline through `detect_homography`, `detect_sample`,
and `decode()`:

```python
def _run_detection(image):
    # ... binarize, alignment scan, cluster ...
    fps, fit_results = _fit_finders(clusters, img_gray)
    associations = find_all_associations(fps)
    triplet = find_triplets(fps, associations)[0]
    H, version = _direct_homography_and_version(triplet, fit_results)
    return H, version
```

Runtime consideration: `finder_fit` per cluster involves edge extraction
(blur + Sobel + NMS) which is O(W×H) per ROI. For a 640×640 image with
4–6 clusters of ~80×80, this adds ~20 ms of edge extraction + ~5 ms of
fitting. Acceptable; the old boundary/NMS path is roughly the same cost.

**Benchmark:**
Run `detect_sample` → `decode.decode()` on a sample of versions 1–6 and
7–12 (clean and default augmentations).  Count decode successes (correct
content string).

**Success criteria:**  Decode success rate ≥ current pipeline on the same
test set.

## Implementation order

1. **Phase 1** — `_run_detection` with `finder_fit` → `FinderPattern` → existing pipeline tail.  Validate on full-pipeline-profile.py.
2. **Phase 2** — New `_direct_homography_and_version()` replacing landmarks/version/RANSAC.  Validate corner accuracy vs Phase 1.
3. **Phase 3** — End-to-end decode benchmark.  Compare to current pipeline.

Each phase commits to a branch; Phase 3 PR is the merge candidate.

## Reused components

| Component | Source | Usage |
|-----------|--------|-------|
| `extract_thin_edges` | `edges.py` | ROI edge extraction |
| `cluster_to_bbox`, `cutout` | `roi.py` | ROI from cluster |
| `fit_finder_full` | `finder_fit.py` | Per-cluster finder fitting |
| `find_all_associations`, `find_triplets` | `finder_pattern.py` | Triplet finding (unchanged) |
| `estimate_homography_dlt`, `refine_homography_lm` | `homography.py` | DLT + LM (unchanged) |
| `compute_qr_corners` | `homography.py` | QR corner projection (unchanged) |
| `sample_qr_bits` | `sample.py` | Bit sampling (unchanged) |

## Risks and mitigations

1. **Version from N_est ± 2 is wrong (m_fit drifts > 15%):**  The only
   reliable fallback is decode-and-check.  If 5 candidate homographies all
   fail to decode, fall through to the old pipeline.  This should be rare
   if Phase 2 corner accuracy is in the 3–4 px range.

2. **Corner-ordering ambiguity in `fit_finder`:**  `_corners_from_rho`
   returns `[(-,-), (+,-), (+,+), (-,+)]` in local (e1, e2) axes.  The
   `order_square_corners` function in `landmarks.py` sorts them via
   projection onto `(right, down)` basis.  If e1/e2 axes don't align with
   the QR's right/down (the 4-fold symmetry means they're equivalent mod
   90°), the order will be a permutation that `order_square_corners`'s
   quadrant-based heuristics must handle.  **Mitigation:** test on varying
   rotations.

3. **Association failure (2 finders too far apart):**  `check_association`
   uses angular and offset tolerances designed for boundary-based corners.
   `finder_fit` corners may have different segment-centre distances.
   **Mitigation:** tunable tolerances or a bypass that directly constructs
   `Triplet` from cluster geometry when associations fail.

4. **Cluster-to-finder mismatch:**  The same cluster may match different GT
   finders across seeds (observed in Plan 012 benchmarking).  The
   associator handles this — it pairs `fp1_idx` with `fp2_idx` regardless
   of which GT finder they represent.  The triplet finding then identifies
   the L-shape.
