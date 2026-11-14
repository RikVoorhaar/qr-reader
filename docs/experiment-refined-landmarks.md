# Experiment: refined landmark extraction for a better homography

**Status:** integrated (spike) — inner corners emitted in `fit_finder_ray`;
alignment-pattern refinement via `build_refined_homography`.  Not yet wired into
`decode_image`.
**Script:** `src/qr_reader/scripts/benchmark_captured_refined.py` (A/B +
ablation); `src/qr_reader/scripts/benchmark_homography_quality.py` (geometry)
**Module:** `src/qr_reader/detector/landmarks_refined.py`

## Question

Is homography estimation ever the limiting factor for decoding, and can we
measurably improve correspondence quality by extracting **more landmarks** after
cross-ratio version estimation?

The current pipeline fits the homography from only the **12 outer finder
corners** (three finder patterns × four corners).  The idea under test:

1. Reuse the ray-profile machinery to also find the **inner** finder corners
   (the white-ring square at grid offset 1..6).
2. Because the version is known, template-match the interior **alignment
   patterns** (the 5×5 patterns) for extra, spread-out correspondences.
3. Optionally weight each correspondence by a confidence score and refit.

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

The A/B + ablation was run on the 100-photo captured benchmark
(`scripts/benchmark_captured_refined.py`), fitting each homography variant per
candidate and decoding with each (**no RANSAC**):

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

### Runtime (whole benchmark, 100 images; integrated inner corners)

| Variant | homography fit | per image |
|---------|---------------:|----------:|
| baseline | 0.14 s | 1.4 ms |
| inner | 0.17 s | 1.7 ms |
| align | 0.31 s | 3.1 ms |
| both | 0.34 s | 3.4 ms |

After the integration, the extra cost of the useful (alignment) step is
**~1.7 ms/image** vs the naive re-fit's 23 ms/image — a >10× speedup.  Emitting
the inner corners inside `fit_finder_ray` reuses the per-ray module-pitch
estimates already computed for the outer corners, so it adds only ~0.4 ms/image
of homography work (plus a small always-on detection cost of ~2–3 ms/image
because inner corners are fitted for every candidate cluster).

## Conclusion

The "cleaner story" the team wanted is the correct one: **the alignment patterns
are what make the homography estimation better** — inner corners are not needed.
On the real-photo captured benchmark, alignment patterns lift decode accuracy
52 → 56 (4 images recovered, 0 lost) at ~1.7 ms/image.  On synthetic images the
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
