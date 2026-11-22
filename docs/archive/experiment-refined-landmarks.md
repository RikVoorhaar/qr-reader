# Experiment: refined landmark extraction for a better homography

**Status:** archived experiment. The method was integrated into `decode_image`,
using 12 outer corners plus interior alignment patterns (DLT + LM). The
measurements below describe the experiments, not the current pipeline.
**Module:** `src/qr_reader/detector/landmarks_refined.py`

## Question

Is homography estimation ever the limiting factor for decoding, and can we
measurably improve correspondence quality by extracting **more landmarks** after
cross-ratio version estimation?

The pipeline originally fitted the homography from only the **12 outer finder
corners** (three finder patterns × four corners).  The idea under test: because
the version is known, template-match the interior **alignment patterns** (the
5×5 patterns) for extra, spread-out correspondences and refit.  Inner finder
corners were also tried as a candidate source of extra corners, and rejected
(see below).

## What was measured

For synthetic images (`generate_test_image`, easy/medium/hard presets ×
versions 1–20), the exact ground-truth homography `H_gt` is reconstructed by
replaying the deterministic rotation + perspective transform chain.  The
quality metric is the **dense reprojection error**: for every module centre
``(col+0.5, row+0.5)``, ``||H(grid) − H_gt(grid)||`` in image pixels.  Only
images whose top-ranked triplet was correct (outer-corner correspondence error
< 5 px) are aggregated.

## Results

| Preset | baseline mean / max | refined mean / max | mean Δ | max Δ |
|--------|--------------------:|--------------------:|-------:|------:|
| easy   | 0.82 / 1.28 | 0.76 / 1.04 | +7% | +19% |
| medium | 0.91 / 1.84 | 0.78 / 1.28 | +15% | +30% |
| hard   | 0.89 / 2.15 | 0.73 / 1.20 | +18% | +44% |

("refined" = 12 outer + 12 inner + all alignment patterns, RANSAC + LM.)

Per-landmark correspondence error vs ground truth (all presets):

| Type | mean | p90 | max |
|------|-----:|----:|----:|
| outer | 0.86 | 1.34 | 1.48 |
| inner | 0.79 | 1.16 | 1.29 |
| align | 1.03 | 1.12 | 1.15 |

## Findings

1. **More correspondences measurably improve the homography.**  Alignment
   patterns are the real win: because they are spread across the interior, they
   constrain the homography where the 12 corner points do not reach.  Worst-case
   dense error drops 19–44 %, and the improvement grows with difficulty.

2. **Inner finder corners alone barely help.**  They are geometrically redundant
   with the outer corners (colinear, co-located), so they add little spatial
   constraint.  They are slightly *more* accurately localised (0.79 px vs
   0.86 px) but that does not move the homography.

3. **Confidence weighting does not help in this setting.**  Template-correlation
   confidences are compressed into a narrow band (~0.45–0.61) and are not
   discriminative, so weighted LM matches unweighted LM.  RANSAC adds a small
   additional worst-case gain by rejecting occasional outlier alignment matches.

4. **This does not improve *synthetic* benchmark decode scores.**  On the hard
   preset the baseline homography's worst-case module-centre error reaches
   5.79 px (≈ half a module) in a few images, yet those images still decode —
   the 3×3 supersampling, adaptive threshold, and Reed-Solomon correction absorb
   the remaining sub-module misalignment.  The refined homography caps
   worst-case error at ≤ 2.48 px, but the extra margin is already in the "safe"
   region.

## Captured (real-photo) benchmark

An A/B ablation was run on the 100-photo captured benchmark, fitting each
homography variant per candidate and decoding with each (**no RANSAC**):

| Variant | decode accuracy | vs baseline |
|---------|----------------:|-------------|
| baseline (12 outer corners) | 52 / 100 | — |
| inner (outer + inner corners) | 51 / 100 | −1 |
| **align (outer + alignment patterns)** | **56 / 100** | **+4, −0** |
| both (outer + inner + alignment) | 56 / 100 | +4, −0 |

The alignment patterns alone **recover 4 images the baseline misses and lose
none**.  Adding inner corners on top changes nothing (both == align).  Inner
corners *alone* are neutral-to-slightly-negative (they are colinear with the
outer corners and add no spatial spread, only a little extra noise).

**Result:** `decode_image` now fits the homography from outer corners +
alignment patterns, and `benchmark_captured.py` reports **56 / 100** (up from
52), with no image lost.  The added runtime is ~1–2 ms/image (alignment
template matching), vs ~23 ms/image for a naive re-fit that also recomputed the
inner-corner ray profiles — so the integrated version is >10× cheaper.

## Conclusion

The "cleaner story" the team wanted is the correct one: **the alignment patterns
are what make the homography estimation better** — inner corners are not needed.
On the real-photo captured benchmark, alignment patterns lift decode accuracy
52 → 56 (4 images recovered, 0 lost) at ~1–2 ms/image.  On synthetic images the
improvement is geometric only (Reed-Solomon absorbs the residual sub-module
error, so decode accuracy is unchanged).  Inner finder corners are a red
herring for homography: colinear and co-located with the outer corners, they add
no spatial constraint and even cost a decode when used alone.

## Terminology note

The codebase historically calls the 1:1:3:1:1 ratio scan "alignment patterns"
(`detector/alignment.py`), but that ratio is actually the **finder pattern**.
The true **alignment patterns** are the interior 5×5 patterns (ratio
1:1:1:1:1), which this experiment detects for the first time via template
matching.  This naming drift predates this experiment and is left unchanged.

## Implementation note (no OpenCV)

The alignment-pattern centre is found by **normalized template matching**, but
the cross-correlation is implemented directly with numpy/scipy
(`scipy.signal.fftconvolve` + integral images) rather than `cv2.matchTemplate`,
so the pipeline core has no OpenCV dependency for this step.  The result is
bit-for-bit equivalent to `TM_CCOEFF_NORMED` (max difference ~1e-9).

A **ray-profile** variant (reusing the finder pattern's per-ray module-pitch fit
+ 4-edge clustering, with the 5×5 cross-section and boundary points at 2.5
modules) was also tried.  It finds the centre perfectly on clean synthetic
patterns but fails on real photos (decode 56 → 38) and is ~70× slower
(~35 ms/pattern vs ~1 ms), because the ray + edge-clustering machinery is the
right tool for the three large finder patterns but overkill and fragile for many
small alignment patterns.  It was therefore rejected.
