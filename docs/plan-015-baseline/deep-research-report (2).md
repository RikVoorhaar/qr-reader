# Deep Research on Finder-Only QR Detection for Clean Version One Images

## Bottom line

For clean, axis-aligned Version 1 QR images, the geometry available from the three finder patterns is normally sufficient for detection, dimension estimation, and a usable projective warp. Standard detector families such as ZXing and quirc both build the QR geometry primarily from the three finder patterns; Version 1 is a special case precisely because there is no alignment pattern, so these implementations either synthesize the bottom-right geometry or refine it from timing-pattern consistency rather than from a fourth fiducial. That means your three failing detector-level tests are more likely to come from a discrete geometry bug than from a lack of information: most commonly a sign convention bug in the TL/TR/BL ordering, a 90° corner-order mismatch on one finder, or a biased corner-localization model that is only exposed when the bottom-right corner is extrapolated instead of observed. citeturn25search0turn22view0turn34view0turn7search0

Your current design choices are broadly in line with the literature and with production decoders: a 1:1:3:1:1 scan for finder candidates, finder grouping by center geometry and module-size agreement, dimension estimation from the two TL-to-neighbor distances, and a homography refined against more than four point correspondences are all standard or closely related to standard practice. What is *not* standard is relying on an independently estimated per-finder 90° orientation to decide TR versus BL; reference implementations instead determine TL from triangle geometry and then use an orientation test on the three finder centers themselves. citeturn3view6turn33view0turn35view1turn5view0turn22view0

## Cross-product conventions in image coordinates

In image geometry, the safest convention is to do every orientation test in **Cartesian-like point order** `(x, y) = (col, row)`, even if the image is stored as `(row, col)`. OpenCV-style image coordinates place the origin at the top-left, with `x` increasing to the right and `y` increasing downward; that flips the usual “positive means counterclockwise” interpretation from textbook Cartesian geometry. In other words, the standard 2D signed-area expression

```python
def cross_img(v1_rc, v2_rc):
    x1, y1 = v1_rc[1], v1_rc[0]   # col, row
    x2, y2 = v2_rc[1], v2_rc[0]
    return x1 * y2 - y1 * x2
```

still computes the correct signed area, but in image coordinates `cross_img(u, v) > 0` corresponds to a **clockwise** turn on screen, not a counterclockwise one. If you instead apply the determinant directly to `(row, col)` vectors without swapping axes, the sign flips. citeturn26search1turn26search6turn27search1turn5view0

That is also how reference QR code implementations behave. ZXing first identifies the point closest to the other two as the top-left finder and then uses a cross-product sign test to decide whether the other two are in the correct order, swapping them if the sign is wrong. Its `crossProductZ(pointA, pointB, pointC)` function computes the signed area of `(C-B) × (A-B)` and expects a positive sign for the desired ordering in image coordinates. quirc does the same thing in a slightly different form: it constructs the hypotenuse line between the two non-TL capstones and then enforces a clockwise `A-B-C` ordering before it canonicalizes the capstone corners. citeturn5view0turn22view0

For your exact question, if `TL` is already known and you define the two candidate vectors as `v1 = P1 - TL` and `v2 = P2 - TL` in `(col,row)` order, then for a normal, non-mirrored QR image the ordering `(TR, BL)` satisfies

\[
\operatorname{cross}(v_{TR}, v_{BL}) > 0
\]

in image coordinates. On a clean axis-aligned symbol this reduces to the obvious case `v_TR ≈ (+d, 0)` and `v_BL ≈ (0, +d)`, so the determinant is positive. If you are storing vectors as `(row,col)`, the same test will appear with the opposite sign unless you convert to `(x=col, y=row)` first. citeturn26search6turn5view0turn22view0

The 90° ambiguity from a per-finder orientation estimator is a separate issue. The finder pattern is designed to be detectable “from any direction,” and the 1:1:3:1:1 structure is rotationally symmetric at the detector level. So a 4-fold gradient histogram can recover only a local axis pair modulo 90°, not a globally consistent TR-versus-BL label. That ambiguity should **not** affect TR/BL classification if you classify TR/BL from the triangle of finder centers, but it absolutely can affect any downstream step that assumes a fixed local corner order for each finder. quirc explicitly waits until the global grid topology is known and only then rotates each capstone so that corner 0 is “top-left with respect to the grid,” which is exactly the canonicalization step your pipeline needs before 12-corner homography fitting. citeturn34view0turn22view0

## Homography from finder corners

A planar homography has 8 degrees of freedom up to scale, so four non-collinear point correspondences are the minimal case; anything above that is an overdetermined fit. OpenCV’s homography documentation states that robust methods sample subsets of four point pairs and then refine the final estimate with Levenberg–Marquardt to reduce reprojection error, while standard DLT tutorials likewise note that a minimum of four non-collinear points is required and that larger point sets are solved in least squares. On that basis alone, **twelve** corresponding outer-corner points from three finders are more than enough in principle. citeturn30view0turn31view0

The important qualification is *where* those twelve points live. In a Version 1 QR code, all geometric evidence is concentrated in the three finder neighborhoods, leaving the bottom-right corner effectively extrapolated rather than directly constrained. That is not a true projective degeneracy: an “L-shaped” layout of the three finder regions is actually the nominal QR topology, because the three finder patterns are always placed at top-left, top-right, and bottom-left. The real issue is conditioning of the *extrapolated* region. If your point labeling is correct and corner localization is subpixel to low-single-pixel accurate, a clean synthetic 21×21 symbol rendered at `box_size=10` gives about 210 px across the active code body, so a 5 px error corresponds to about half a module. That is a loose target for an overdetermined homography fit on clean data; if you miss it consistently, the usual cause is not insufficient correspondences but a wrong permutation, wrong sign convention, or a systematic corner-bias model. This is an inference from the QR dimensions and from standard homography conditioning, not a formal spec guarantee. citeturn7search0turn15search0turn30view0turn31view0

Normalization is still worth keeping. Hartley-style normalized DLT exists precisely because unnormalized DLT is sensitive to the origin and scale of the coordinate system; the normalized version recenters points and scales them to a stable average distance before solving. If your current solver is already normalized DLT plus LM refinement, that is the right family of estimator for this problem. citeturn37view0turn30view0

There is no special degeneracy caused by the three finders lying on two orthogonal lines; that is the expected geometry. What *does* improve conditioning is adding evidence away from the three finder clusters. Production libraries do exactly that for higher versions. ZXing explicitly says that anything above Version 1 has an alignment pattern and searches for it to improve the transform; OpenCV exposes `setUseAlignmentMarkers()` with the documentation that alignment markers are used “to improve the position of the corners”; and quirc uses alignment patterns only on grids larger than 21 modules, while also scoring the timing pattern in the fitted grid. Those are strong signs that your instinct to add alignment-pattern samples for `V ≥ 2` or timing-pattern crossings for any version is correct. citeturn25search0turn3view0turn22view0

For Version 1 specifically, timing-pattern evidence is the most natural extra constraint because it exists even when alignment patterns do not. quirc’s `fitness_all()` explicitly scores the timing pattern along row 6 and column 6 of the grid, in addition to the capstones and any alignment patterns. That makes timing crossings or timing-line phase consistency a better detector-stage verification signal than format information areas, which are usually checked only after grid sampling. citeturn23view2turn34view0

## Version estimation and the ninety-degree ambiguity

Your estimator

\[
N \approx \frac{d_{TR} + d_{BL}}{2\,m_{avg}} + 7
\]

is extremely close to what ZXing does. In ZXing’s detector, the provisional dimension is computed from the two TL-to-neighbor center distances divided by module size, averaged, and then shifted by `+7`; the result is then snapped to the legal QR dimension class using the `mod 4` rule. It also computes module size as an average derived from the three finder patterns, using black-white-black runs along the axes joining finder centers. So the overall structure of your estimator is not unusual at all. citeturn35view1turn35view2turn35view3

Because that formula depends only on **center distances** and **module pitch**, it is not directly sensitive to your per-finder 90° orientation estimate. Orientation error only enters indirectly if your axis estimate corrupts either the finder center or the module-pitch estimate. So if your finder centers are stable and `m_avg` is measured from robust scanline fits, dimension estimation should be much less fragile than corner ordering or homography fitting. In practice, when this estimator fails on clean data, the biggest culprits are usually a biased module-size estimate, a triplet mislabeling, or a dimension-snapping bug rather than a 90° local-axis flip by itself. citeturn35view1turn35view2turn35view3

If you want to use all three inter-finder distances simultaneously, the natural improvement is a small least-squares fit over the ideal right-triangle geometry rather than a plain two-edge average. ZXing’s multi-finder grouping already uses the same right-triangle constraint explicitly: it expects the two TL legs to be similar in length and the third side to agree with Pythagoras within a 10% tolerance. That gives you a principled geometry model:

\[
d(TL,TR)\approx s\,m,\quad
d(TL,BL)\approx s\,m,\quad
d(TR,BL)\approx \sqrt{2}\,s\,m
\]

with \( s = N-7 \). A simple unweighted estimator would therefore be

\[
\hat s \;=\; \frac{d(TL,TR)+d(TL,BL)+d(TR,BL)/\sqrt2}{3\,m_{avg}}
\]

followed by snapping \( \hat N=\hat s+7 \) to the legal `1 mod 4` dimension class. That formula is an engineering extrapolation from ZXing’s two-edge dimension rule plus its explicit right-triangle checks, not a standardized QR formula, but it is geometrically sound and often more stable when one leg is slightly biased. citeturn35view1turn33view0

A still better approach is to make dimension estimation *post-homography* rather than *pre-homography*: use the three finders to get a provisional warp, then estimate the symbol size from timing-pattern periodicity or from timing-pattern transition counts in the rectified grid. That lines up well with how timing patterns are described in QR references—as the structure that determines module coordinates—and with quirc’s use of timing-pattern consistency as a grid fitness term. citeturn34view0turn23view2

The 90° ambiguity in your gradient histogram is a known and expected consequence of the finder pattern symmetry. A finder pattern is intentionally readable from any direction, and quirc’s source shows that it does not trust an independently fixed corner order for each capstone: it rotates each capstone only after the global grid reference has been established. So yes, you should canonicalize all finders to a unified coordinate frame **before** any step that depends on ordered local corners. The `max(dot1,dot2)` idea is fine for axis-compatibility checks that are supposed to be invariant under 90° local rotations, but it is not sufficient if the next stage assumes that “corner 0” means the same physical corner on all three finders. citeturn34view0turn22view0

## What normally breaks on Version One symbols

The literature and production decoders point to a consistent set of detector-level failure modes. Low resolution and near-Nyquist sampling reduce QR recognition sharply because module edges blur and phase shifts make the black/white structure unstable at the pixel level; the same paper also highlights dependence on orientation, scale, blur, surrounding image content, and preprocessing. Another common failure source is quiet-zone violation, since the QR standard expects a four-module blank margin and reference material explicitly calls it out as part of the usable symbol area. A third source is damage or corruption in a finder pattern, which is why some specialized detectors focus on handling partially damaged finders. citeturn16view0turn15search0turn18search9

For clean synthetic Version 1 images, though, the dominant extra vulnerability is simply the absence of an alignment pattern. Standard decoders do rely solely on the three finder patterns at this version. ZXing explicitly handles the “no alignment pattern” case by fabricating the bottom-right point from the three known finders, and quirc similarly uses the three capstones to establish the grid, adding an alignment pattern only when `grid_size > 21`. So relying on the three finders alone is standard for Version 1; what is also standard is to use the **timing pattern** as a verification signal because it helps stabilize the grid when the fourth-corner geometry is only inferred. citeturn25search0turn22view0turn23view2

The format-information area is less attractive as a detector-stage geometric check. Keyence’s QR structure reference describes format information as the field containing error-correction level and mask pattern, and says it is read first **when the code is decoded**. That makes it very useful after you have a sampled grid, but not the first thing to add if your failure is still in triplet classification or homography formation. Timing-pattern checks are usually the earlier and cheaper detector-stage verification. citeturn34view0

On tolerances, there is no one universal “axis alignment tolerance” in the standard, but reference implementations do show what practical detectors accept. ZXing’s `MultiFinderPatternFinder` uses three permissive geometric checks when deciding whether three finder candidates form a QR triplet: module-size agreement within about 5% or 0.5 px/module, equality of the two TL legs within 10%, and Pythagorean agreement of the diagonal within 10%. That corresponds roughly to allowing on the order of ten degrees of deviation from a perfect right angle when the two legs are similar. Those are multi-code, real-image tolerances; for clean synthetic unit tests you can usually tighten them materially. OpenCV, separately, exposes `setEpsX()` and `setEpsY()` for the horizontal and vertical 1:1:3:1:1 scan tolerances, which is a reminder that scan-ratio tolerance and triplet-geometry tolerance are distinct knobs. citeturn33view0turn3view1

## What I would change first in your pipeline

The first change I would make is to **separate center geometry from local finder orientation completely**. Use only the triangle of finder centers to decide `[TL, TR, BL]`, with the sign test performed in `(x=col, y=row)` image coordinates. Do not let a 4-fold local orientation estimate participate in TR-versus-BL classification. Local orientation should only be used later, after triplet formation, to canonicalize local axes or to propose corner permutations. That is the closest match to how ZXing and quirc structure the problem. citeturn5view0turn22view0

The second change is to **canonicalize the four outer corners of every finder in a global frame before DLT**. Your observed case—two finders with `(e1,e2)` in one orientation and the third rotated by 90°—is exactly the situation where a 12-point homography can fail catastrophically even though every finder was localized well. If one finder's corners are cyclically permuted relative to the other two, DLT is solving the wrong correspondence problem. The clean fix is to set the global frame from the selected triplet and then rotate each finder’s corner cycle to match that frame, just as quirc rotates capstones after the global grid has been chosen. A brute-force fallback is also reasonable here: 4 possible corner rotations per finder means only \(4^3=64\) corner-order combinations, which is cheap enough to test with a reprojection-plus-timing score on detector-level unit tests. The recommendation to enumerate permutations is an engineering suggestion; the need for global canonicalization is supported directly by quirc’s design. citeturn22view0turn23view2

The third change is to add a **timing-pattern score** to both version estimation and homography acceptance. quirc’s grid fitness explicitly checks timing cells, and timing patterns are documented as the structure that determines module coordinates. In practice, this gives you a powerful discriminator for exactly the bugs you are chasing: a wrong TR/BL swap, a 90° corner-cycle error on one finder, or a biased homography can all still produce a numerically finite warp, but they usually produce the wrong timing phase or the wrong number of alternations along row 6 and column 6. Timing consistency is therefore the best extra verification signal for Version 1. citeturn23view2turn34view0

The fourth change is to make dimension estimation a **two-stage estimate**: use your current inter-finder-distance formula only for a provisional `N`, snap it to the legal QR dimension class, and then re-estimate or verify `N` from the rectified timing pattern. That follows the spirit of ZXing’s provisional dimension rule and quirc’s timing-pattern validation, while reducing the impact of any one biased module-pitch estimate. citeturn35view1turn23view2

If I had to guess the single most likely root cause behind *clean axis-aligned Version 1 failures* in your current implementation, it would be this combination: your center-level triplet logic is probably close to correct, but one of the following is wrong in a discrete way—row/col cross-product sign, TR/BL swap under image-coordinate orientation, or per-finder corner-cycle canonicalization. Those are exactly the kinds of bugs that survive on many real images yet fail deterministically on clean synthetic unit tests where the expected geometry is uncompromising. citeturn5view0turn22view0turn25search0