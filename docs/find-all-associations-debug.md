# `find_all_associations` high-version failure investigation

## Summary

A debug script was added at:

- `src/qr_reader/scripts/debug_find_all_associations.py`

It follows the same pipeline as `src/qr_reader/scripts/full-pipeline.py` up to:

```python
fps = extract_finder_patterns(all_corners)
associations = find_all_associations(fps)
```

Then it recreates the important logic from `check_association()` with extra diagnostics for every finder-pattern pair and every segment pair.

The current high-version default reproduction (`version=12`, content `https://www.rikvoorhaar.com`, `border=15`, `seed=0`) confirms the issue:

```text
Clusters: 4
Extracted finder patterns: 4
find_all_associations returned 0 association(s):
  CONFIRMED: no associations found for this reproduction.
```

## How to run

From the repository root:

```sh
uv run python src/qr_reader/scripts/debug_find_all_associations.py
```

Useful variants:

```sh
uv run python src/qr_reader/scripts/debug_find_all_associations.py --version 12
uv run python src/qr_reader/scripts/debug_find_all_associations.py --version 4
```

The script defaults to the current `full-pipeline.py` high-version case:

- `version=12`
- `content="https://www.rikvoorhaar.com"`
- `border=15`
- `seed=0`
- QR generator defaults for perspective/noise/blur
- `angle_tol=0.1`
- `offset_tol=0.15`

## Observed reproduction details

For `--version 12`, the script finds 4 clusters:

```text
cluster 0: center=(row=886.00, col=530.35)
cluster 1: center=(row=469.50, col=887.00)
cluster 2: center=(row=86.43, col=460.86)
cluster 3: center=(row=568.00, col=370.00)
```

The first three are the real finder patterns. Cluster 3 is the middle-image false positive described in the issue. Its selected outer quad is extremely large and clearly not a finder pattern:

```text
FP cluster 3: outer_area=200874.00
outer corners(row,col)=[[509.  50.]
                        [573. 109.]
                        [525. 884.]
                        [ 99. 511.]]
```

However, this false positive is not the immediate cause of the no-association result. The real adjacent finder-pattern pairs are also rejected.

## Relevant production logic

`check_association()` currently does:

1. Build 4 outer segments from each finder pattern.
2. Reject immediately if any segment intersects.
3. For every segment pair:
   - compute acute `angular_distance`
   - if `angle < angle_tol`, compute `max_offset`
   - if `offset < offset_tol`, record the pair as colinear
4. Accept only when there are exactly 2 colinear segment pairs.

```python
if len(colinear_pairs) == 2:
    return Association(...)
return None
```

`max_offset()` computes absolute line-distance error divided by the distance between the two segment midpoints:

```python
return max_d / L
```

where `L` is the segment-midpoint distance.

## Main finding

The high-version failure appears to be caused by the `offset_tol` normalization interacting with the strict `len(colinear_pairs) == 2` requirement.

At `version=12`, the finder patterns are much farther apart, but each finder pattern remains approximately the same pixel size because `box_size=10`. Since `max_offset()` divides by the midpoint distance between segment pairs, the offset score for “wrong but parallel” segment combinations becomes smaller as QR version increases.

That makes extra segment pairs pass the offset test. The real adjacent finder pairs therefore produce 4 colinear pairs instead of exactly 2, so `check_association()` rejects them.

Example for real adjacent pair `FP 0 <-> FP 1` at `version=12`:

```text
Pair FP 0 <-> FP 1: REJECT: expected exactly 2 colinear pairs, got 4
colinear_pairs passing both predicates: [(0, 0), (0, 2), (2, 0), (2, 2)]
```

The two expected pairs pass:

```text
s0-s0: angle=0.009925 rad, offset=0.020018
s2-s2: angle=0.011842 rad, offset=0.021027
```

But the cross-pairs also pass because the normalized offset is below `0.15`:

```text
s2-s0: angle=0.009925 rad, offset=0.109606
s0-s2: angle=0.011842 rad, offset=0.139119
```

The same pattern happens for the other real adjacent pair `FP 1 <-> FP 2`:

```text
Pair FP 1 <-> FP 2: REJECT: expected exactly 2 colinear pairs, got 4
colinear_pairs passing both predicates: [(1, 1), (1, 3), (3, 1), (3, 3)]
```

Again, this is not because the finder patterns are badly detected. Their corners and segment directions look coherent. The rejection comes from having too many passing segment pairs.

## Version 4 comparison

Running the same debug script with `--version 4` reproduces the user observation that the issue disappears.

For `version=4`, real adjacent pairs produce exactly 2 colinear pairs:

```text
Pair FP 0 <-> FP 1: ACCEPT
colinear_pairs passing both predicates: [(0, 0), (2, 2)]
```

The corresponding cross-pairs fail the offset threshold:

```text
s2-s0: offset=0.252992 (fail)
s0-s2: offset=0.296563 (fail)
```

This supports the hypothesis: at lower version numbers, the distance between finder patterns is smaller, so the same kind of absolute cross-pair line-distance normalizes to a larger offset and fails the `0.15` threshold. At higher version numbers, the denominator is larger, so those cross-pairs fall below the threshold and create 4 colinear pairs.

## Current root-cause hypothesis

Most likely root cause:

> `check_association()` assumes a true association creates exactly 2 colinear segment-pairs, but `max_offset()` normalizes by inter-segment distance. At high QR versions, this distance grows while finder-pattern size stays fixed, causing extra opposite-edge cross-pairs to pass `offset_tol`. The true adjacent finder pairs then have 4 passing pairs and are rejected.

The false-positive cluster is still a separate quality issue and can create additional noise, but it is not necessary to explain why the three valid finder patterns fail to associate in this reproduction.

## Follow-up experiments: local normalization + best-two selection

The debug script now also includes an experimental association strategy. This is not wired into production code.

Experimental strategy:

1. Keep the existing intersection rejection.
2. Replace production `max_offset()` normalization with a local-scale offset:

   ```python
   local_offset = max_abs_line_distance / mean(segment_length_1, segment_length_2)
   ```

3. Do not require exactly two total passing segment pairs.
4. Instead, try both opposite-side axes in each finder pattern: `(0, 2)` and `(1, 3)`.
5. For each axis combination, try both possible one-to-one pairings and select the lowest-scoring valid pair set.
6. Accept when the selected two pairs both pass angle and local-offset thresholds.

That last point matters: seed 7 showed a valid adjacent pair whose best pairing was `(0, 1), (2, 3)`, not same-index pairs such as `(0, 0), (2, 2)`. So a robust best-two selection cannot assume segment indices line up across finder patterns.

### Version sweep

Command:

```sh
uv run python src/qr_reader/scripts/debug_find_all_associations.py --sweep-versions 1:12 --local-offset-tol 0.30
```

Result summary for `seed=0`:

- Production association count drops below the needed two associations at version 8 and reaches zero from versions 9 through 12.
- Experimental local-scale best-two association finds two associations for every version from 1 through 12.
- For version 12, the experimental associations are:

```text
0-1:[(0, 0), (2, 2)]:off=0.177,ang=0.68
1-2:[(1, 1), (3, 3)]:off=0.197,ang=1.05
```

### Tolerance sensitivity

I tested `local_offset_tol` values `0.15`, `0.20`, `0.25`, `0.30`, and the original default experiment value `0.35`.

Observed behavior:

- `0.15` is too strict for versions 8+.
- `0.20` recovers version 12 seed 0 but misses versions 10 and 11.
- `0.25` recovers most versions but still misses version 11 seed 0.
- `0.30` recovers versions 1–12 for seed 0.
- `0.35` also recovers versions 1–12 for seed 0, but `0.30` is tighter and therefore preferable based on these experiments.

The highest observed selected-pair local offset in the seed-0 version sweep was about `0.269` at version 11, so `0.30` gives some margin without admitting the obvious cross-pairs in the inspected cases.

### High-version seed sweep

Command:

```sh
uv run python src/qr_reader/scripts/debug_find_all_associations.py --version 12 --sweep-seeds 0:8 --local-offset-tol 0.30
```

For seeds `0..8` at version 12:

- Production found zero associations for most seeds.
- Production occasionally found one association, but the examples were suspicious/degenerate, e.g. duplicated segment indices like `[0, 2]/[0, 0]`, not a useful finder-pattern triplet basis.
- Experimental local-scale best-two selection found two associations for every sampled seed.
- The experimental method remained stable even when there were many false-positive clusters/finder patterns, e.g. seeds with 7, 8, or 10 extracted FPs.

Example seed 7 after generalizing axis matching:

```text
12,7,... experimental_count=2,
0-1:[(1, 1), (3, 3)]:off=0.185,ang=1.80;
1-2:[(0, 1), (2, 3)]:off=0.231,ang=1.85
```

This confirms the need to select best compatible pairs rather than requiring exactly two total passing pairs or assuming equal segment indices.

## Updated root-cause assessment

The experiments support both proposed directions:

1. **Better normalization:** normalizing offset by local finder-pattern segment length separates true edge matches from cross-pairs much better than normalizing by inter-FP distance.
   - In the original failing version-12 seed-0 case, true selected pairs had local offsets around `0.16–0.20`.
   - Cross-pairs that incorrectly passed production offset had local offsets around `0.8–1.2`, so they are cleanly rejected by a local threshold near `0.30`.

2. **Return/select the best pairs:** the production requirement `len(colinear_pairs) == 2` is brittle.
   - At high versions, true adjacent pairs can produce four production-colinear pairs because cross-pairs pass the distance-normalized offset.
   - Selecting the best two compatible opposite-side pairs recovers the expected association.
   - The selection should consider all opposite-axis combinations and both pairings because segment numbering is not consistent enough to rely on same-index matches.

Remaining caution:

- These experiments still depend on upstream extraction producing reasonable finder-pattern quads. False-positive filtering is still valuable, but it is a separate issue from the high-version association predicate.
- Before implementing a production fix, this experimental logic should be converted into unit tests covering at least:
  - version 12 seed 0 reproduction
  - version 12 seed 7 axis-mismatch case
  - a lower-version case such as version 4
  - a case with false-positive clusters where no false triplet should be selected

## Suggested production fix

The production fix should start with more aggressive unit tests around the association step, before changing `check_association()` itself. The main risk is that a more permissive association rule could recover true high-version pairs while also admitting false-positive pairs. Tests should lock down both sides.

> **Status update (2026-06-12):** The unit tests described below have been implemented in `src/qr_reader/tests/detector/test_finder_pattern_association.py`.  Three tests currently **pass** (guarding existing low-version behaviour) and three tests intentionally **fail** (documenting the high-version bug).  The next agent should implement the production fix and then verify that all six tests pass.

### Step 1: add association-focused unit tests

Add tests close to the existing finder-pattern tests, e.g. in `src/qr_reader/tests/detector/test_finder_pattern.py` or a new dedicated file such as `src/qr_reader/tests/detector/test_finder_pattern_association.py`.

The most important tests here should be true **unit tests** for association logic. They should not depend on binarization, alignment-pattern detection, clustering, boundary extraction, or corner extraction. Image-based tests are still useful, but they are integration tests and should come after the synthetic unit coverage.

#### Synthetic finder-pattern generator

Create a small test helper that directly generates `FinderPattern` instances.

Suggested API shape:

```python
def generate_synthetic_finder_patterns(
    *,
    version: int,
    seed: int = 0,
    module_size: float = 10.0,
    origin: tuple[float, float] = (0.0, 0.0),
    rotation_rad: float = 0.0,
    scale: float = 1.0,
    perspective_amount: float = 0.0,
    jitter_std: float = 0.0,
    include_inner: bool = True,
) -> list[FinderPattern]:
    ...
```

The helper should:

1. Generate the three correct QR finder-pattern locations for the given version.
   - QR grid size is `N = 4 * version + 17`.
   - Finder-pattern outer squares are 7 modules wide.
   - In QR grid coordinates, the three outer finder squares can be represented near:
     - top-left: rows/cols `0..7`
     - top-right: rows `0..7`, cols `N-7..N`
     - bottom-left: rows `N-7..N`, cols `0..7`
   - Use the same `(row, col)` convention as `FinderPattern.outer_corners`.

2. Convert module coordinates to image-like coordinates using `module_size` and `origin`.

3. Apply an affine transform:
   - uniform scale
   - rotation
   - translation chosen so all coordinates remain positive

4. Apply a slight perspective transform when `perspective_amount > 0`.
   - This can be modeled by perturbing the four QR-code corner positions and applying a homography to all finder-pattern corners.
   - Keep it seeded and deterministic.

5. Apply corner jitter when `jitter_std > 0`.
   - Jitter should be small relative to `module_size`.
   - Use a seeded `np.random.default_rng(seed)`.

6. Optionally generate `inner_corners` as a smaller square inset from the outer square by roughly one module.

This helper should make it easy to generate low-version and high-version finder-pattern sets that differ mainly in finder-pattern separation, not finder-pattern size. That is exactly the condition that exposes the current production normalization problem.

#### Bogus finder-pattern generator

Add a second helper for false positives:

```python
def generate_bogus_finder_patterns(
    *,
    count: int,
    seed: int,
    image_extent: tuple[float, float],
    module_size: float = 10.0,
    jitter_std: float = 0.0,
) -> list[FinderPattern]:
    ...
```

Bogus patterns should be plausible enough to challenge association logic but not arranged at correct QR finder locations. Useful variants:

- isolated square-like patterns in the middle of the QR extent
- patterns roughly parallel to a true finder-pattern edge but at the wrong offset
- very large or very small quads
- slightly skewed/non-square quads
- patterns close enough to create tempting one-edge matches but not a valid two-edge association

Keep these deterministic by seed.

#### Unit test cases to add before the fix

These tests should intentionally document the current failure, then be flipped/updated as part of the fix.

Recommended cases:

1. **Low version, no false positives**
   - Generate version 4 synthetic finder patterns.
   - Current production should find the two expected associations.
   - This guards existing behavior.

2. **High version, no false positives**
   - Generate version 12 synthetic finder patterns with the same module size.
   - Current production should fail or return fewer than two associations because extra cross-pairs pass production offset normalization.
   - After the fix, this should pass with two expected associations.

3. **Low version with bogus finder patterns**
   - Generate version 4 true FPs plus seeded bogus FPs.
   - Association/triplet logic should still find the true QR triplet and not prefer bogus associations.

4. **High version with bogus finder patterns**
   - Generate version 12 true FPs plus seeded bogus FPs.
   - Current production should fail or be unstable.
   - After the fix, it should find the true associations/triplet.

5. **Perspective + jitter stress cases**
   - Repeat high-version tests with moderate `perspective_amount` and `jitter_std`.
   - These should verify that the replacement logic is not overfit to perfect squares.

6. **Axis-mismatch case**
   - Create a synthetic case where the correct pairing is not same-index, e.g. selected pairs equivalent to `(0, 1), (2, 3)`.
   - This is important because the debug experiments showed real extracted FPs can require cross-index pairing.

Example association assertions:

```python
associations = find_all_associations(fps)
pairs = {frozenset((a.fp1_idx, a.fp2_idx)) for a in associations}
assert frozenset((top_left_idx, top_right_idx)) in pairs
assert frozenset((top_left_idx, bottom_left_idx)) in pairs

triplets = find_triplets(fps, associations)
assert any(t.top_left_idx == top_left_idx for t in triplets)
```

For tests with bogus FPs, assert not only that enough associations exist, but that the expected true triplet can be recovered.

#### Integration tests based on actual images

After the unit tests exist, add image-based regression tests using known deterministic reproductions. These are integration tests because they depend on all prior pipeline stages being correct.

Suggested integration cases:

- Version 12 seed 0 should produce the two real associations.
- Version 12 seed 7 should produce two real associations, including the axis-mismatch case observed in the debug script:

  ```text
  1-2:[(0, 1), (2, 3)]
  ```

- Version 4 seed 0 should continue to produce the expected associations.

The integration assertions should focus on association/triplet behavior, not necessarily full decoding.

### Step 2: introduce local-offset scoring

Replace or supplement `max_offset()` inside association checking with a local-scale offset:

```python
local_offset = max_abs_line_distance(segment_a, segment_b) / mean_segment_length
```

where:

```python
mean_segment_length = (length(segment_a) + length(segment_b)) / 2
```

Based on experiments, an initial `local_offset_tol` around `0.30` is reasonable:

- `0.15` was too strict for high versions.
- `0.20–0.25` recovered some high-version cases but missed others.
- `0.30` recovered versions 1–12 for seed 0 and version 12 seeds 0–8 in the debug experiments.

This tolerance should be treated as empirical and covered by tests.

### Step 3: select the best compatible two-pair association

Change the acceptance rule from:

```python
len(colinear_pairs) == 2
```

to a scoring/selection rule:

1. Consider opposite-side axes `(0, 2)` and `(1, 3)` in each finder pattern.
2. Try all axis combinations between the two finder patterns.
3. Try both one-to-one pairings for each axis combination.
4. Keep candidates where both segment pairs pass:
   - angle tolerance
   - local-offset tolerance
5. Return the candidate with the best score, e.g. lowest `max_local_offset`, then lowest total score.

This avoids rejecting true associations just because additional weaker/cross-pairs pass a broad predicate.

The returned `Association` can still store two segment indices per finder pattern, but the implementation should preserve the actual selected pairing internally while constructing it. If downstream code ever needs exact pair correspondence, the `Association` dataclass may need to evolve from parallel segment-index lists to explicit pair tuples:

```python
colinear_segment_pairs: list[tuple[int, int]]
```

For the current code, check `find_triplets()` compatibility carefully. It mostly uses the set of segments from the center finder pattern, so parallel lists may continue to work, but the axis-mismatch case should be covered by tests.

### Step 4: keep false-positive filtering separate

The false-positive middle cluster is real and should eventually be filtered, but it is not the root cause of this high-version association failure. Association changes should be tested independently first. After that, add finder-pattern quality filters such as:

- outer/inner area ratio sanity checks
- minimum/maximum side-length consistency
- requirement for plausible inner corners
- rejection of highly non-square or self-intersecting selected outer quads

## Suggested next investigation steps

The unit-test scaffolding is now in place.  The next useful steps are:

1. **Implement the production fix** in `src/qr_reader/detector/finder_pattern.py`:
   - Introduce a local-scale offset metric (offset divided by mean segment length) inside `check_association()`.
   - Replace the brittle `len(colinear_pairs) == 2` rule with a best-compatible-two-pair selection that considers all opposite-side axis combinations and both one-to-one pairings.
   - Preserve compatibility with `find_triplets()`; the `Association` dataclass may need to evolve from parallel segment-index lists to explicit pair tuples.
2. **Run the new unit tests** (`uv run pytest src/qr_reader/tests/detector/test_finder_pattern_association.py -v`) and confirm all six cases pass.
3. **Add image-based integration tests** using the deterministic reproductions from `debug_find_all_associations.py` (version 12 seed 0, version 12 seed 7, version 4 seed 0).
4. **Separately add a filter for false-positive finder patterns** (area / inner-corner sanity checks), but treat that as a separate issue from the high-version association failure.
