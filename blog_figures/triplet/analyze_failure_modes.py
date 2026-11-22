#!/usr/bin/env python3
"""Failure-mode analysis of the *repaired* Attempt 1 triplet association.

Attempt 1 associates two finder patterns when two pairs of their outer edges are
nearly parallel and nearly collinear.  The repaired rule (commit ``e335de8``):

* normalizes the collinearity error by the mean length of the two segments
  (``local_offset``) instead of by the distance between their midpoints;
* uses ``angle_tol = 0.1`` rad and ``offset_tol = 0.30``;
* tries both opposite-side axes and both one-to-one pairings;
* keeps the valid pairing with the smallest ``max(offset)``, then the smallest
  total ``angle + offset``;
* builds the association graph and extracts L-shaped triplets.

This script runs that rule on the deduplicated ray-fit finder candidates from
the current detector front end, compares it with the current centre-geometry
triplet finder (used as reference truth) on the captured benchmark, and
classifies every failure.

Reference subset: images where the current pipeline detects successfully *and*
sampling + decoding returns ``rikvoorhaar.com``.  For those images the first
triplet from ``find_valid_triplets`` is the reference; decode success confirms
its TL/TR/BL roles are correct.

Usage (from the repository root):

    .venv/bin/python blog_figures/triplet/analyze_failure_modes.py
    .venv/bin/python blog_figures/triplet/analyze_failure_modes.py --refresh

The first run decodes all 100 photos and caches the front-end candidates in
``/tmp/opencode/triplet_failure_modes/cache.pkl``; later runs reuse the cache.
``--refresh`` re-runs the pipeline.  Diagnostic PNGs for the shortlisted cases
and a markdown report are written to ``--out-dir``.

The script is deterministic: no random component, Agg rendering, finder
candidates sorted by ``cluster_idx``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2

# Each worker process must stay single-threaded: OpenCV otherwise spawns one
# thread per core per process, and a pool of workers then thrashes.
cv2.setNumThreads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.detector.finder_pattern import (  # noqa: E402
    Association,
    FinderPattern,
    Triplet,
    find_valid_triplets,
)
from qr_reader.detector.geometry import (  # noqa: E402
    angular_distance,
    local_offset,
    point_line_distance,
    segments_intersect,
)
from qr_reader.detector.sample import sample_qr_bits  # noqa: E402

from make_figures import classify_edges  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT_DIR = Path("/tmp/opencode/triplet_failure_modes")
DEFAULT_CACHE = DEFAULT_OUT_DIR / "cache.pkl"
EXPECTED_PAYLOAD = "rikvoorhaar.com"

ANGLE_TOL = 0.1
MAIN_OFFSET_TOL = 0.30
SWEEP = (0.15, 0.20, 0.25, 0.30, 0.40, 0.60, 1.20)


# ---------------------------------------------------------------------------
# Repaired Attempt 1 (e335de8)
# ---------------------------------------------------------------------------


@dataclass
class Pairing:
    """One of the eight axis/pairing combinations tried for a finder pair."""

    pairs: tuple[tuple[int, int], tuple[int, int]]  # ((seg1,seg2), (seg1,seg2))
    angles: tuple[float, float]
    offsets: tuple[float, float]
    passed: bool
    max_offset: float = np.inf
    total: float = np.inf


@dataclass
class AssociationCheck:
    """Result of applying the repaired rule to one finder pair."""

    fp1: int
    fp2: int
    intersects: bool
    pairings: list[Pairing]
    selected: Association | None

    @property
    def valid_pairings(self) -> list[Pairing]:
        return [p for p in self.pairings if p.passed]

    def pairing_for(self, pairs) -> Pairing | None:
        """The candidate whose two segment pairs equal ``pairs``.

        Members are returned in the same order as ``pairs`` so per-member
        angles and offsets can be zipped with the requested segment pairs.
        """
        wanted = [tuple(p) for p in pairs]
        for p in self.pairings:
            got = [tuple(q) for q in p.pairs]
            if sorted(got) != sorted(wanted):
                continue
            order = []
            remaining = list(range(len(got)))
            for w in wanted:
                k = next(i for i in remaining if got[i] == w)
                order.append(k)
                remaining.remove(k)
            return Pairing(
                pairs=tuple(p.pairs[i] for i in order),
                angles=tuple(p.angles[i] for i in order),
                offsets=tuple(p.offsets[i] for i in order),
                passed=p.passed,
                max_offset=p.max_offset,
                total=p.total,
            )
        return None


def check_association_repaired(
    fp1: FinderPattern, fp2: FinderPattern, angle_tol=ANGLE_TOL, offset_tol=MAIN_OFFSET_TOL
) -> AssociationCheck:
    """Repaired Attempt 1 association, plus every tried pairing for diagnostics."""
    segs1 = fp1.segments()
    segs2 = fp2.segments()

    for s1 in segs1:
        for s2 in segs2:
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                return AssociationCheck(fp1.cluster_idx, fp2.cluster_idx, True, [], None)

    axes = ((0, 2), (1, 3))
    pairings: list[Pairing] = []
    seen: set = set()

    for axis1 in axes:
        for axis2 in axes:
            for pairing in (
                ((axis1[0], axis2[0]), (axis1[1], axis2[1])),
                ((axis1[0], axis2[1]), (axis1[1], axis2[0])),
            ):
                # Keep each (fp1 segment, fp2 segment) ordered: swapping the
                # two is a different pairing, not a duplicate.
                key = tuple(sorted(pairing))
                if key in seen:
                    continue
                seen.add(key)

                angles = []
                offsets = []
                ok = True
                for i, j in pairing:
                    s1 = segs1[i]
                    s2 = segs2[j]
                    ang = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
                    off = float(local_offset(s1[0], s1[1], s2[0], s2[1]))
                    angles.append(ang)
                    offsets.append(off)
                    if ang >= angle_tol or off >= offset_tol:
                        ok = False

                score = angles[0] + offsets[0] + angles[1] + offsets[1]
                pairings.append(
                    Pairing(
                        pairs=(pairing[0], pairing[1]),
                        angles=(angles[0], angles[1]),
                        offsets=(offsets[0], offsets[1]),
                        passed=ok,
                        max_offset=max(offsets),
                        total=score,
                    )
                )

    valid = [p for p in pairings if p.passed]
    if not valid:
        return AssociationCheck(fp1.cluster_idx, fp2.cluster_idx, False, pairings, None)

    best = min(valid, key=lambda p: (p.max_offset, p.total))
    assoc = Association(
        fp1_idx=fp1.cluster_idx,
        fp2_idx=fp2.cluster_idx,
        colinear_segments_1=[p[0] for p in best.pairs],
        colinear_segments_2=[p[1] for p in best.pairs],
    )
    return AssociationCheck(fp1.cluster_idx, fp2.cluster_idx, False, pairings, assoc)


def find_all_associations_repaired(
    fps: list[FinderPattern], angle_tol=ANGLE_TOL, offset_tol=MAIN_OFFSET_TOL
) -> tuple[list[Association], list[AssociationCheck]]:
    associations: list[Association] = []
    checks: list[AssociationCheck] = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            check = check_association_repaired(fps[i], fps[j], angle_tol, offset_tol)
            checks.append(check)
            if check.selected is not None:
                associations.append(check.selected)
    return associations, checks


def find_triplets_repaired(
    fps: list[FinderPattern], associations: list[Association]
) -> list[Triplet]:
    """Triplet discovery from the repaired association graph (e335de8).

    The cross product is evaluated in (x, y) = (col, row) coordinates, the
    convention fixed in ``7f91833`` so TR/BL are not swapped.
    """
    adj: dict[int, list[tuple[int, Association]]] = defaultdict(list)
    for a in associations:
        adj[a.fp1_idx].append((a.fp2_idx, a))
        adj[a.fp2_idx].append((a.fp1_idx, a))

    centers = {fp.cluster_idx: fp.outer_corners.mean(axis=0) for fp in fps}
    assoc_pairs = {(a.fp1_idx, a.fp2_idx) for a in associations} | {
        (a.fp2_idx, a.fp1_idx) for a in associations
    }

    triplets: list[Triplet] = []
    for b_idx in adj:
        neighbors = adj[b_idx]
        if len(neighbors) < 2:
            continue
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a_idx, assoc_ab = neighbors[i]
                c_idx, assoc_bc = neighbors[j]

                if (a_idx, c_idx) in assoc_pairs:
                    continue

                if assoc_ab.fp1_idx == b_idx:
                    b_segs_ab = set(assoc_ab.colinear_segments_1)
                else:
                    b_segs_ab = set(assoc_ab.colinear_segments_2)
                if assoc_bc.fp1_idx == b_idx:
                    b_segs_bc = set(assoc_bc.colinear_segments_1)
                else:
                    b_segs_bc = set(assoc_bc.colinear_segments_2)

                if b_segs_ab & b_segs_bc:
                    continue

                vec_ba = centers[a_idx] - centers[b_idx]
                vec_bc = centers[c_idx] - centers[b_idx]
                v_ba_xy = vec_ba[::-1]
                v_bc_xy = vec_bc[::-1]
                cross = float(v_ba_xy[0] * v_bc_xy[1] - v_ba_xy[1] * v_bc_xy[0])

                if cross > 0:
                    top_right, bottom_left = a_idx, c_idx
                else:
                    top_right, bottom_left = c_idx, a_idx

                triplets.append(Triplet(b_idx, top_right, bottom_left))

    return triplets


# ---------------------------------------------------------------------------
# Per-image reference data
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    image: str
    version: int
    detection_ok: bool
    decode_ok: bool
    payload_ok: bool
    text: str
    error: str
    fps: list[FinderPattern]
    score_map: dict[int, float]
    current_triplets: list[Triplet]

    @property
    def reference_triplet(self) -> Triplet | None:
        if self.payload_ok and self.current_triplets:
            return self.current_triplets[0]
        return None


def evaluate_image(path: Path) -> ImageRecord:
    """Run the current pipeline once, capturing the deduplicated finder set.

    ``find_valid_triplets`` is spied on rather than recomputing the front end,
    so the captured candidates are exactly the ones the production triplet
    finder saw.
    """
    import qr_reader.detector.detector as detmod

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"cannot read {path}")

    captured: dict = {}
    original = detmod.find_valid_triplets

    def spy(fps, score_map, *args, **kwargs):
        captured["fps"] = [
            FinderPattern(cluster_idx=fp.cluster_idx, outer_corners=fp.outer_corners.copy())
            for fp in fps
        ]
        captured["score_map"] = dict(score_map)
        return original(fps, score_map, *args, **kwargs)

    detmod.find_valid_triplets = spy
    try:
        det = detmod._run_detection(gray)
    finally:
        detmod.find_valid_triplets = original

    fps = captured.get("fps", [])
    score_map = captured.get("score_map", {})
    fps.sort(key=lambda fp: fp.cluster_idx)
    current_triplets = find_valid_triplets(fps, score_map) if len(fps) >= 3 else []

    text = ""
    decode_ok = False
    payload_ok = False
    if det.success:
        assert det.H is not None and det.version is not None
        N = 4 * det.version + 17
        try:
            bits = sample_qr_bits(gray, det.H, N)
            text = decode_qr(bits)
            decode_ok = True
            payload_ok = text == EXPECTED_PAYLOAD
        except Exception:
            pass

    return ImageRecord(
        image=path.name,
        version=det.version or -1,
        detection_ok=det.success,
        decode_ok=decode_ok,
        payload_ok=payload_ok,
        text=text,
        error="" if det.success else f"{type(det.error).__name__}: {det.error}",
        fps=fps,
        score_map=score_map,
        current_triplets=current_triplets,
    )


def collect_records(
    data_dir: Path, cache: Path, refresh: bool, workers: int = 1
) -> list[ImageRecord]:
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            records = pickle.load(f)
        print(f"loaded {len(records)} cached image records from {cache}")
        return records

    paths = sorted(data_dir.glob("*.jpeg"), key=lambda p: natural_key(p.stem))
    records: list[ImageRecord] = []
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cache.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(evaluate_image, p): p for p in paths}
            for k, fut in enumerate(as_completed(futures), start=1):
                records.append(fut.result())
                if k % 10 == 0 or k == len(paths):
                    print(f"  {k}/{len(paths)} images...")
                    partial = cache.with_suffix(".partial.pkl")
                    with open(partial, "wb") as f:
                        pickle.dump(records, f)
    else:
        for k, path in enumerate(paths):
            records.append(evaluate_image(path))
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(paths)} images...")

    order = {p.name: i for i, p in enumerate(paths)}
    records.sort(key=lambda r: order[r.image])

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(records, f)
    print(f"wrote cache {cache}")
    return records


def natural_key(stem: str) -> tuple[int, ...]:
    import re

    return tuple(int(n) for n in re.findall(r"\d+", stem))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class PairAnalysis:
    fp_a: int
    fp_b: int
    true_pairs: tuple[tuple[int, int], tuple[int, int]]
    true_pairing: Pairing | None
    selected: Pairing | None
    all_pairings: list[Pairing]
    offset_tol: float

    def true_failure(self) -> str | None:
        """Describe the member(s) of the true pairing that fail the thresholds."""
        if self.true_pairing is None:
            return "true pairing not among the tried pairings"
        if self.true_pairing.passed:
            return None
        parts = []
        for (sa, sb), ang, off in zip(
            self.true_pairing.pairs, self.true_pairing.angles, self.true_pairing.offsets
        ):
            if ang >= ANGLE_TOL or off >= self.offset_tol:
                parts.append(f"seg {sa}-{sb} angle={ang:.3f} offset={off:.3f}")
        return "; ".join(parts) if parts else "unknown"

    @property
    def associated(self) -> bool:
        return self.selected is not None

    @property
    def true_pairing_passes(self) -> bool:
        return bool(self.true_pairing and self.true_pairing.passed)

    @property
    def selected_is_true(self) -> bool:
        if self.selected is None or self.true_pairing is None:
            return False
        return {tuple(p) for p in self.selected.pairs} == {
            tuple(p) for p in self.true_pairing.pairs
        }

    def best_false(self) -> Pairing | None:
        false = [
            p
            for p in self.all_pairings
            if p.passed
            and self.true_pairing is not None
            and {tuple(q) for q in p.pairs}
            != {tuple(q) for q in self.true_pairing.pairs}
        ]
        if not false:
            return None
        return min(false, key=lambda p: (p.max_offset, p.total))


@dataclass
class ImageAnalysis:
    image: str
    version: int
    n_finders: int
    reference: Triplet
    current_n_triplets: int
    offset_tol: float
    edges: dict[int, dict[str, int]]
    true_pair_analyses: dict[tuple[int, int], PairAnalysis]
    associations: list[Association]
    checks: list[AssociationCheck]
    triplets: list[Triplet]
    label: str  # root cause
    outcome: str  # success / wrong triplet / no triplet
    detail: str
    extra_false_edges: list[tuple[int, int]] = field(default_factory=list)

    @property
    def first_triplet(self) -> Triplet | None:
        return self.triplets[0] if self.triplets else None


def repaired_triplet_count(rec: ImageRecord, offset_tol: float) -> int:
    associations, _ = find_all_associations_repaired(
        rec.fps, angle_tol=ANGLE_TOL, offset_tol=offset_tol
    )
    return len(find_triplets_repaired(rec.fps, associations))


def analyse_image(rec: ImageRecord, offset_tol: float) -> ImageAnalysis:
    ref = rec.reference_triplet
    assert ref is not None
    fps = rec.fps
    fp_map = {fp.cluster_idx: fp for fp in fps}

    tl, tr, bl = ref.top_left_idx, ref.top_right_idx, ref.bottom_left_idx
    centres = {i: fp_map[i].outer_corners.mean(axis=0) for i in (tl, tr, bl)}
    u = centres[tr] - centres[tl]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = centres[bl] - centres[tl]
    v = v / (np.linalg.norm(v) + 1e-12)

    edges = {i: classify_edges(fp_map[i], u, v) for i in (tl, tr, bl)}
    true_specs = {
        (tl, tr): (
            (edges[tl]["top"], edges[tr]["top"]),
            (edges[tl]["bottom"], edges[tr]["bottom"]),
        ),
        (tl, bl): (
            (edges[tl]["left"], edges[bl]["left"]),
            (edges[tl]["right"], edges[bl]["right"]),
        ),
    }

    associations, checks = find_all_associations_repaired(
        fps, angle_tol=ANGLE_TOL, offset_tol=offset_tol
    )
    triplets = find_triplets_repaired(fps, associations)
    check_map = {(c.fp1, c.fp2): c for c in checks}

    pair_analyses: dict[tuple[int, int], PairAnalysis] = {}
    for (a, b), pairs in true_specs.items():
        if (a, b) in check_map:
            check = check_map[(a, b)]
            flip = False
        else:
            check = check_map[(b, a)]
            # The check stores fp1=b, fp2=a; express everything in (a, b)
            # order so per-member metrics line up with the drawn segments.
            flip = True

        def reorient(pairing: Pairing | None) -> Pairing | None:
            if pairing is None or not flip:
                return pairing
            return Pairing(
                pairs=tuple((j, i) for i, j in pairing.pairs),
                angles=pairing.angles,
                offsets=pairing.offsets,
                passed=pairing.passed,
                max_offset=pairing.max_offset,
                total=pairing.total,
            )

        oriented_pairs = (
            tuple((j, i) for i, j in pairs) if flip else pairs
        )
        true_pairing = reorient(check.pairing_for(oriented_pairs))
        selected_pairing = None
        if check.selected is not None:
            sel_pairs = tuple(
                zip(check.selected.colinear_segments_1, check.selected.colinear_segments_2)
            )
            selected_pairing = reorient(check.pairing_for(sel_pairs))
        pair_analyses[(a, b)] = PairAnalysis(
            fp_a=a,
            fp_b=b,
            true_pairs=pairs,
            true_pairing=true_pairing,
            selected=selected_pairing,
            all_pairings=[reorient(p) for p in check.pairings],
            offset_tol=offset_tol,
        )

    true_edge_set = {frozenset((tl, tr)), frozenset((tl, bl))}
    all_edge_set = {frozenset((a.fp1_idx, a.fp2_idx)) for a in associations}
    extra_false_edges = sorted(
        {tuple(sorted(e)) for e in all_edge_set - true_edge_set},
        key=lambda e: (e[0], e[1]),
    )

    label, outcome, detail = classify(
        rec, ref, pair_analyses, triplets, associations, extra_false_edges
    )

    return ImageAnalysis(
        image=rec.image,
        version=rec.version,
        n_finders=len(fps),
        reference=ref,
        current_n_triplets=len(rec.current_triplets),
        offset_tol=offset_tol,
        edges=edges,
        true_pair_analyses=pair_analyses,
        associations=associations,
        checks=checks,
        triplets=triplets,
        label=label,
        outcome=outcome,
        detail=detail,
        extra_false_edges=extra_false_edges,
    )


def triplet_key(t: Triplet) -> tuple[int, int, int]:
    return (t.top_left_idx, t.top_right_idx, t.bottom_left_idx)


def classify(rec, ref, pair_analyses, triplets, associations, extra_false_edges):
    """Assign a root cause and an outcome.

    Root cause follows the analysis taxonomy (``true pair missed`` etc.).
    Outcome is what the triplet stage actually returned: ``success`` (first
    triplet matches the reference roles), ``wrong triplet`` or ``no triplet``.
    The two overlap by construction: every no-triplet case here is caused by a
    missed true pair, so ``no triplet`` is reported separately rather than
    used as a root cause.
    """
    tl, tr, bl = ref.top_left_idx, ref.top_right_idx, ref.bottom_left_idx
    true_edges = {frozenset((tl, tr)), frozenset((tl, bl))}

    if triplets and triplet_key(triplets[0]) == triplet_key(ref):
        outcome = "success"
    elif triplets:
        outcome = "wrong triplet"
    else:
        outcome = "no triplet"

    if outcome == "success":
        if len(triplets) > 1:
            return "success", outcome, f"correct first triplet, {len(triplets) - 1} extra"
        return "success", outcome, "correct first triplet"

    missed = [p for p in pair_analyses.values() if not p.associated]
    outranked = [
        p
        for p in pair_analyses.values()
        if p.associated and p.true_pairing_passes and not p.selected_is_true
    ]

    if missed:
        parts = []
        for p in missed:
            reason = p.true_failure() or "no valid pairing"
            cross = p.selected is not None
            suffix = " (a cross pairing passes)" if cross else ""
            parts.append(f"{p.fp_a}-{p.fp_b}: {reason}{suffix}")
        return "true pair missed", outcome, "; ".join(parts)

    if outranked:
        parts = []
        for p in outranked:
            parts.append(
                f"{p.fp_a}-{p.fp_b}: selected {p.selected.pairs} "
                f"max={p.selected.max_offset:.3f} over true "
                f"max={p.true_pairing.max_offset:.3f}"
            )
        return "true pair outranked", outcome, "; ".join(parts)

    # Both true associations were found with the geometric true pairing.
    if not triplets:
        if extra_false_edges:
            return "false association", outcome, "extra edges prevent the L-shape"
        return "other", outcome, "true associations present but no triplet"

    first = triplets[0]
    first_edges = {frozenset((first.top_left_idx, first.top_right_idx)),
                   frozenset((first.top_left_idx, first.bottom_left_idx))}
    if not first_edges <= true_edges:
        return "false association", outcome, "first triplet uses a false edge"
    if len(triplets) > 1:
        return "triplet selection", outcome, f"{len(triplets)} triplets, first is wrong"
    return "triplet selection", outcome, "single triplet but wrong roles"


# ---------------------------------------------------------------------------
# Aggregates and report
# ---------------------------------------------------------------------------


def sweep_label_counts(records, offset_tol):
    labels = Counter()
    outcomes = Counter()
    per_image = []
    for rec in records:
        if rec.reference_triplet is None:
            continue
        a = analyse_image(rec, offset_tol)
        labels[a.label] += 1
        outcomes[a.outcome] += 1
        per_image.append(a)
    return labels, outcomes, per_image


def aggregate(records):
    refs = [r for r in records if r.reference_triplet is not None]
    print(f"\n{'=' * 72}")
    print(f"Captured dataset: {len(records)} images")
    print(f"Current detection success: {sum(r.detection_ok for r in records)}")
    print(f"Decode success:            {sum(r.decode_ok for r in records)}")
    print(f"Payload == {EXPECTED_PAYLOAD!r}: {sum(r.payload_ok for r in records)}")
    print(f"Reference subset:          {len(refs)}")
    with_finders = [r for r in records if len(r.fps) >= 3]
    print(f"Images with >= 3 finders:  {len(with_finders)}")

    labels, outcomes, analyses = sweep_label_counts(records, MAIN_OFFSET_TOL)
    n = sum(labels.values())
    print(f"\nOutcome at offset_tol={MAIN_OFFSET_TOL} (n={n}):")
    for label, count in outcomes.most_common():
        print(f"  {label:<22} {count:3d}  {100.0 * count / n:5.1f}%")
    print(f"\nRoot cause at offset_tol={MAIN_OFFSET_TOL} (n={n}):")
    for label, count in labels.most_common():
        print(f"  {label:<22} {count:3d}  {100.0 * count / n:5.1f}%")

    print("\nThreshold sweep (root causes):")
    print(f"  {'tol':>5} {'success':>8} {'no triplet':>11} {'missed':>8} "
          f"{'outrank':>8} {'false':>7} {'select':>7} {'other':>6}")
    for tol in SWEEP:
        counts, _, _ = sweep_label_counts(records, tol)
        print(
            f"  {tol:5.2f} {counts['success']:8d} {counts['no triplet']:11d} "
            f"{counts['true pair missed']:8d} {counts['true pair outranked']:8d} "
            f"{counts['false association']:7d} {counts['triplet selection']:7d} "
            f"{counts['other']:6d}"
        )

    print("\nThreshold sweep (outcomes):")
    print(f"  {'tol':>5} {'success':>8} {'wrong triplet':>14} {'no triplet':>11} "
          f"{'repaired>=1':>11} {'repaired total':>14}")
    for tol in SWEEP:
        _, outcome_counts, per_image = sweep_label_counts(records, tol)
        print(
            f"  {tol:5.2f} {outcome_counts['success']:8d} "
            f"{outcome_counts['wrong triplet']:14d} {outcome_counts['no triplet']:11d} "
            f"{sum(bool(a.triplets) for a in per_image):11d} "
            f"{sum(len(a.triplets) for a in per_image):14d}"
        )

    # Repaired vs current triplet counts on images with >= 3 finders.
    print("\nTriplets per image (>= 3 finders):")
    repaired_counts = [repaired_triplet_count(r, MAIN_OFFSET_TOL) for r in with_finders]
    current_counts = [len(r.current_triplets) for r in with_finders]
    both = sum(1 for a, b in zip(repaired_counts, current_counts) if a > 0 and b > 0)
    only_rep = sum(1 for a, b in zip(repaired_counts, current_counts) if a > 0 and b == 0)
    only_cur = sum(1 for a, b in zip(repaired_counts, current_counts) if a == 0 and b > 0)
    neither = sum(1 for a, b in zip(repaired_counts, current_counts) if a == 0 and b == 0)
    print(f"  images with repaired >= 1: {sum(c > 0 for c in repaired_counts)} / {len(with_finders)}")
    print(f"  images with current  >= 1: {sum(c > 0 for c in current_counts)} / {len(with_finders)}")
    print(f"  total repaired triplets:   {sum(repaired_counts)}")
    print(f"  total current triplets:    {sum(current_counts)}")
    print(f"  both: {both}  repaired-only: {only_rep}  current-only: {only_cur}  neither: {neither}")

    print("\nVersion breakdown (reference subset):")
    by_version = defaultdict(Counter)
    for a in analyses:
        by_version[a.version][a.label] += 1
    for version in sorted(by_version):
        counts = by_version[version]
        total = sum(counts.values())
        parts = ", ".join(f"{k} {v}" for k, v in counts.most_common())
        print(f"  v{version:<2} n={total:<2}  {parts}")

    return analyses, labels, outcomes


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def fmt_score(p: Pairing | None) -> str:
    if p is None:
        return "n/a"
    return f"max={p.max_offset:.3f} total={p.total:.3f}"


def shortlist(analyses: list[ImageAnalysis], k=4) -> list[ImageAnalysis]:
    """Prefer low versions, clean candidate sets, legible failures."""
    failures = [a for a in analyses if a.label != "success"]
    failures.sort(
        key=lambda a: (
            a.version if a.version > 0 else 99,
            abs(a.n_finders - 3),
            a.n_finders,
            a.label,
            a.image,
        )
    )
    out: list[ImageAnalysis] = []
    seen_labels = set()
    for a in failures:
        if a.label in seen_labels and len(out) >= 2:
            continue
        out.append(a)
        seen_labels.add(a.label)
        if len(out) == k:
            break
    # Fill up if fewer distinct labels.
    for a in failures:
        if a not in out:
            out.append(a)
        if len(out) == k:
            break
    out.sort(key=lambda a: (a.version, a.image))
    return out


def threshold_sensitivity_case(records) -> ImageAnalysis | None:
    """A low-version, clean case that misses at 0.25 but succeeds at 0.30."""
    best: ImageAnalysis | None = None
    best_key = None
    for rec in records:
        if rec.reference_triplet is None or rec.version > 10 or len(rec.fps) != 3:
            continue
        low = analyse_image(rec, 0.25)
        if low.label == "success":
            continue
        if analyse_image(rec, MAIN_OFFSET_TOL).label != "success":
            continue
        key = (rec.version, abs(len(rec.fps) - 3), rec.image)
        if best_key is None or key < best_key:
            best, best_key = low, key
    return best


def crop_bounds(fps, shape, pad=90):
    corners = np.vstack([fp.outer_corners for fp in fps])
    r0 = max(0, int(corners[:, 0].min()) - pad)
    r1 = min(shape[0], int(corners[:, 0].max()) + pad)
    c0 = max(0, int(corners[:, 1].min()) - pad)
    c1 = min(shape[1], int(corners[:, 1].max()) + pad)
    return r0, r1, c0, c1


def draw_finder(ax, fp, color, lw=2.6, label=None):
    c = fp.outer_corners[:, ::-1]
    q = np.vstack([c, c[0]])
    ax.plot(q[:, 0], q[:, 1], color="#101010", lw=lw + 2, alpha=0.6, zorder=3)
    ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, zorder=4)
    ax.scatter(c[:, 0], c[:, 1], s=26, facecolors="none", edgecolors=color,
               linewidths=1.6, zorder=5)
    if label is not None:
        m = fp.outer_corners.mean(axis=0)
        ax.text(m[1], m[0], label, color=color, fontsize=15, fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="#101010", ec="none", alpha=0.8))


def connect_edges(ax, fp_a, sa, fp_b, sb, color, lw=3.2, ls="-", zorder=6):
    p = np.asarray(fp_a.segments()[sa], dtype=float).mean(axis=0)[::-1]
    q = np.asarray(fp_b.segments()[sb], dtype=float).mean(axis=0)[::-1]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=zorder,
            solid_capstyle="round")
    ax.scatter([p[0], q[0]], [p[1], q[1]], s=26, color=color, zorder=zorder + 1)


PASS_COLOR = "#39ff88"
FAIL_COLOR = "#ff3b3b"
FALSE_COLOR = "#ffb000"


def transform_points(matrix: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts_xy = np.asarray(pts_xy, dtype=np.float64)
    return pts_xy @ matrix[:, :2].T + matrix[:, 2]


def rotate_for_pair(gray: np.ndarray, pts_xy: np.ndarray, axis_a, axis_b):
    """Rotate the image so the ``axis_a → axis_b`` direction is horizontal.

    Returns the rotated image, the affine matrix, and the transformed points.
    """
    h, w = gray.shape
    d = np.asarray(axis_b, dtype=float) - np.asarray(axis_a, dtype=float)
    angle = float(np.degrees(np.arctan2(d[1], d[0])))
    best = None
    for sign in (1.0, -1.0):
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), sign * angle, 1.0)
        ca = transform_points(M, np.asarray([axis_a]))[0]
        cb = transform_points(M, np.asarray([axis_b]))[0]
        dy = abs(cb[1] - ca[1])
        if best is None or dy < best[0]:
            best = (dy, M)
    M = best[1]

    image_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)
    all_pts = np.vstack([pts_xy, image_corners])
    tp = transform_points(M, all_pts)
    x0, y0 = np.floor(tp.min(axis=0)).astype(int)
    x1, y1 = np.ceil(tp.max(axis=0)).astype(int)
    M = M.copy()
    M[0, 2] -= x0
    M[1, 2] -= y0
    out = cv2.warpAffine(
        gray, M, (int(x1 - x0) + 2, int(y1 - y0) + 2),
        flags=cv2.INTER_LINEAR, borderValue=255,
    )
    return out, M, transform_points(M, pts_xy)


def render_pair_strip(ax, gray, fp_a, fp_b, pa: PairAnalysis, title: str):
    """Wide strip through two finders, rotated so the pair axis is horizontal.

    The two members of the true geometry are drawn as thick segments on the
    finder edges plus thin extensions along their infinite lines, so a
    non-collinear pair shows two visibly separated parallel lines.  The
    perpendicular gap of a failing member is annotated with its normalized
    offset and the threshold.
    """
    corners_a = fp_a.outer_corners[:, ::-1]
    corners_b = fp_b.outer_corners[:, ::-1]
    ca = corners_a.mean(axis=0)
    cb = corners_b.mean(axis=0)
    pts = np.vstack([corners_a, corners_b])

    side = float(
        np.mean([np.linalg.norm(corners_a[(k + 1) % 4] - corners_a[k]) for k in range(4)])
    )
    pad = 0.75 * side
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    # generous padding before rotation so interpolation at the border is clean
    px0 = max(0, x0 - int(2 * pad))
    py0 = max(0, y0 - int(2 * pad))
    px1 = min(gray.shape[1], x1 + int(2 * pad))
    py1 = min(gray.shape[0], y1 + int(2 * pad))
    sub = gray[py0:py1, px0:px1]
    pts_shift = pts - np.array([px0, py0])
    ca_s, cb_s = ca - np.array([px0, py0]), cb - np.array([px0, py0])

    rotated, M, tpts = rotate_for_pair(sub, pts_shift, ca_s, cb_s)
    tca, tcb = tpts[:4].mean(axis=0), tpts[4:].mean(axis=0)

    # Crop after rotation.  The vertical padding keeps the strip close to the
    # natural aspect of the pair, so the image fills the panel and the edge
    # lines stay legible.
    pad_y = 1.35 * side
    pad_x = 0.35 * side
    rx0 = max(0, int(tpts[:, 0].min() - pad_x))
    rx1 = min(rotated.shape[1], int(tpts[:, 0].max() + pad_x))
    ry0 = max(0, int(min(tpts[:, 1].min(), tca[1], tcb[1]) - pad_y))
    ry1 = min(rotated.shape[0], int(max(tpts[:, 1].max(), tca[1], tcb[1]) + pad_y))
    crop_w, crop_h_px = rx1 - rx0, ry1 - ry0
    ax.imshow(rotated[ry0:ry1, rx0:rx1], cmap="gray",
              extent=[0, crop_w, crop_h_px, 0])

    tpts = tpts - np.array([rx0, ry0])
    tca = tca - np.array([rx0, ry0])
    tcb = tcb - np.array([rx0, ry0])
    crop_h = ry1 - ry0

    shift = np.array([px0, py0])

    def edge_xy(idx_fp, seg):
        p, q = fp_a.segments()[seg] if idx_fp == 0 else fp_b.segments()[seg]
        return np.vstack([p, q])[:, ::-1] - shift

    def centre_xy(idx_fp):
        return tca if idx_fp == 0 else tcb

    summary = [
        f"true pairing: {fmt_score(pa.true_pairing)}  "
        f"({'PASS' if pa.true_pairing_passes else 'MISS'})",
        f"threshold: offset < {pa.offset_tol:.2f}, angle < {ANGLE_TOL:.2f} rad",
    ]
    fail_labels: list[str] = []

    # True edge pairs, one member at a time.
    for (sa, sb), ang, off, passed in zip(
        pa.true_pairs, pa.true_pairing.angles, pa.true_pairing.offsets,
        [a_ < ANGLE_TOL and o_ < pa.offset_tol
         for a_, o_ in zip(pa.true_pairing.angles, pa.true_pairing.offsets)],
    ):
        color = PASS_COLOR if passed else FAIL_COLOR
        seg_a = transform_points(M, edge_xy(0, sa)) - np.array([rx0, ry0])
        seg_b = transform_points(M, edge_xy(1, sb)) - np.array([rx0, ry0])
        for seg in (seg_a, seg_b):
            ax.plot(seg[:, 0], seg[:, 1], color="#101010", lw=8.5, alpha=0.55,
                    solid_capstyle="round", zorder=5)
            ax.plot(seg[:, 0], seg[:, 1], color=color, lw=4.8,
                    solid_capstyle="round", zorder=6)
            ax.axline((float(seg[0, 0]), float(seg[0, 1])),
                      (float(seg[1, 0]), float(seg[1, 1])),
                      color=color, lw=1.8, alpha=0.9, zorder=4)
        ma, mb = seg_a.mean(axis=0), seg_b.mean(axis=0)

        # Segment tags placed inside each finder (towards its centre).
        for m, ctr, tag in ((ma, tca, f"{sa}"), (mb, tcb, f"{sb}")):
            v = ctr - m
            n = np.linalg.norm(v)
            if n > 1e-9:
                pos = m + 0.45 * side * v / n
            else:
                pos = m
            ax.text(pos[0], pos[1], tag, color=color, fontsize=11.5, fontweight="bold",
                    ha="center", va="center", zorder=8, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.12", fc="#101010", ec="none", alpha=0.8))

        summary.append(
            f"seg {sa}–{sb}: offset {off:.2f}, angle {ang:.2f} rad "
            f"{'ok' if passed else 'FAIL'}"
        )

        if not passed:
            # local_offset is the largest endpoint-to-opposite-line distance
            # divided by the mean segment length.  Show exactly that endpoint.
            candidates = [
                (seg_a[0], seg_b),
                (seg_a[1], seg_b),
                (seg_b[0], seg_a),
                (seg_b[1], seg_a),
            ]
            best_e, best_line = max(
                candidates,
                key=lambda c: point_line_distance(
                    c[0], c[1][0], c[1][1]
                ),
            )
            dev_px = float(point_line_distance(best_e, best_line[0], best_line[1]))

            d = best_line[1] - best_line[0]
            denom = float(np.dot(d, d))
            if denom > 1e-12:
                t = float(np.dot(best_e - best_line[0], d) / denom)
                proj = best_line[0] + t * d
            else:
                proj = best_line[0]

            ax.plot([best_e[0]], [best_e[1]], marker="o", ms=7, mfc="none",
                    mec=color, mew=2.0, zorder=10)
            ax.annotate(
                "",
                xy=(float(best_e[0]), float(best_e[1])),
                xytext=(float(proj[0]), float(proj[1])),
                arrowprops=dict(arrowstyle="<->", color=color, lw=2.4,
                                shrinkA=0, shrinkB=0),
                zorder=9,
            )
            mid = 0.5 * (best_e + proj)
            ax.text(
                float(np.clip(mid[0] + 0.18 * side, 0.01 * crop_w, 0.72 * crop_w)),
                float(np.clip(mid[1] + (0.35 * side if mid[1] < crop_h / 2 else -0.35 * side),
                              0.04 * crop_h, 0.96 * crop_h)),
                f"{dev_px:.0f} px",
                color=color, fontsize=10, ha="left", va="center", zorder=10,
                clip_on=True,
                bbox=dict(boxstyle="round,pad=0.16", fc="#101010", ec="none", alpha=0.86),
            )
            fail_labels.append(
                f"seg {sa}–{sb}: circled endpoint is {dev_px:.0f} px = "
                f"{off:.2f} × side from the other edge (threshold {pa.offset_tol:.2f})"
            )

    # Best false (cross-side) pairing, if any passes.
    false_pair = pa.best_false()
    if false_pair is not None:
        for (sa, sb) in false_pair.pairs:
            seg_a = transform_points(M, edge_xy(0, sa)) - np.array([rx0, ry0])
            seg_b = transform_points(M, edge_xy(1, sb)) - np.array([rx0, ry0])
            for seg in (seg_a, seg_b):
                ax.plot(seg[:, 0], seg[:, 1], color=FALSE_COLOR, lw=3.6, ls="--",
                        zorder=7)
                ax.axline((float(seg[0, 0]), float(seg[0, 1])),
                          (float(seg[1, 0]), float(seg[1, 1])),
                          color=FALSE_COLOR, lw=1.5, ls=(0, (2, 4)), alpha=0.8,
                          zorder=6)
        summary.append(f"best false pairing: {fmt_score(false_pair)}")
    else:
        summary.append("best false pairing: none below threshold")

    ax.text(0.012, 0.975, "\n".join(summary), transform=ax.transAxes, ha="left", va="top",
            fontsize=10, color="white", linespacing=1.35, zorder=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="#101010", ec="none", alpha=0.87))
    if fail_labels:
        ax.text(0.012, 0.03, "\n".join(fail_labels), transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, color="#ff8a8a",
                linespacing=1.35, zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="#101010", ec="none", alpha=0.87))

    ax.set_xlim(0, crop_w)
    ax.set_ylim(crop_h_px, 0)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, pad=6)


def render_case(a: ImageAnalysis, rec: ImageRecord, out_path: Path, data_dir: Path = DATA_DIR):
    gray = cv2.imread(str(data_dir / a.image), cv2.IMREAD_GRAYSCALE)
    fp_map = {fp.cluster_idx: fp for fp in rec.fps}
    r0, r1, c0, c1 = crop_bounds(rec.fps, gray.shape, pad=60)

    interesting = [
        pa for pa in a.true_pair_analyses.values()
        if not pa.true_pairing_passes or (pa.selected is not None and not pa.selected_is_true)
    ]
    if not interesting:
        interesting = list(a.true_pair_analyses.values())
    interesting.sort(key=lambda p: -(p.true_pairing.max_offset if p.true_pairing else 0))

    n = len(interesting)
    fig = plt.figure(figsize=(11.5, 4.6 + 3.9 * n), dpi=170)
    gs = fig.add_gridspec(
        n + 1, 2, height_ratios=[1.15] + [1.0] * n, width_ratios=[1.0, 1.35],
        hspace=0.2, wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gray[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])

    colors = {fp.cluster_idx: ["#ffd23f", "#4cc9f0", "#c77dff", "#ff6b6b", "#7bed9f"][k % 5]
              for k, fp in enumerate(rec.fps)}
    roles = {a.reference.top_left_idx: "TL", a.reference.top_right_idx: "TR",
             a.reference.bottom_left_idx: "BL"}
    for fp in rec.fps:
        draw_finder(ax, fp, colors[fp.cluster_idx],
                    label=roles.get(fp.cluster_idx, str(fp.cluster_idx)))

    for (pa, pb), pa_an in a.true_pair_analyses.items():
        for (sa, sb) in pa_an.true_pairs:
            connect_edges(ax, fp_map[pa], sa, fp_map[pb], sb,
                          PASS_COLOR if pa_an.true_pairing_passes else FAIL_COLOR, lw=2.6)

    true_edges = {frozenset(k) for k in a.true_pair_analyses}
    for (ea, eb) in a.extra_false_edges:
        if frozenset((ea, eb)) not in true_edges:
            p1 = fp_map[ea].outer_corners.mean(axis=0)[::-1]
            p2 = fp_map[eb].outer_corners.mean(axis=0)[::-1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=FAIL_COLOR, lw=2.0, ls=":", zorder=6)

    summary = [
        f"{a.label.upper()} — {a.outcome}",
        f"v{a.version}, {a.n_finders} finder candidates, offset_tol={a.offset_tol:.2f}",
        f"repaired triplets: {len(a.triplets)}",
        f"reference: TL={a.reference.top_left_idx} TR={a.reference.top_right_idx} "
        f"BL={a.reference.bottom_left_idx}",
    ]
    if a.triplets:
        t = a.triplets[0]
        summary.append(f"first triplet: TL={t.top_left_idx} TR={t.top_right_idx} "
                       f"BL={t.bottom_left_idx}")
    summary.append("")
    for (pa, pb), pa_an in a.true_pair_analyses.items():
        summary.append(
            f"true pair {pa}–{pb}: {fmt_score(pa_an.true_pairing)} "
            f"{'PASS' if pa_an.true_pairing_passes else 'MISS'}"
        )
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    ax.set_title(f"{a.image} — repaired Attempt 1", loc="left", fontsize=15, pad=8)

    info_ax = fig.add_subplot(gs[0, 1])
    info_ax.axis("off")
    info_ax.text(0.0, 1.0, "\n".join(summary), transform=info_ax.transAxes,
                 ha="left", va="top", fontsize=12, color="#101010", linespacing=1.5)

    for k, pa in enumerate(interesting, start=1):
        strip_ax = fig.add_subplot(gs[k, :])
        role_a = roles.get(pa.fp_a, str(pa.fp_a))
        role_b = roles.get(pa.fp_b, str(pa.fp_b))
        render_pair_strip(
            strip_ax, gray, fp_map[pa.fp_a], fp_map[pa.fp_b], pa,
            f"{role_a} ↔ {role_b} (finders {pa.fp_a}–{pa.fp_b})"
            + ("  — true pair missed" if not pa.true_pairing_passes else ""),
        )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def version_breakdown(per_image):
    by_version: dict[int, Counter] = defaultdict(Counter)
    for a in per_image:
        by_version[a.version][a.label] += 1
    return by_version


def extra_association_stats(records, tol):
    extra_edges = images_with_extra = extra_triplets = 0
    for rec in records:
        if rec.reference_triplet is None:
            continue
        a = analyse_image(rec, tol)
        tl, tr, bl = (
            a.reference.top_left_idx, a.reference.top_right_idx, a.reference.bottom_left_idx
        )
        true = {frozenset((tl, tr)), frozenset((tl, bl))}
        extra = [e for e in a.extra_false_edges if frozenset(e) not in true]
        if extra:
            images_with_extra += 1
        extra_edges += len(extra)
        extra_triplets += max(0, len(a.triplets) - 1)
    return extra_edges, images_with_extra, extra_triplets


def triplet_comparison(records):
    with_finders = [r for r in records if len(r.fps) >= 3]
    repaired = [repaired_triplet_count(r, MAIN_OFFSET_TOL) for r in with_finders]
    current = [len(r.current_triplets) for r in with_finders]
    return {
        "images": len(with_finders),
        "repaired_images": sum(c > 0 for c in repaired),
        "current_images": sum(c > 0 for c in current),
        "repaired_total": sum(repaired),
        "current_total": sum(current),
        "both": sum(1 for a, b in zip(repaired, current) if a > 0 and b > 0),
        "repaired_only": sum(1 for a, b in zip(repaired, current) if a > 0 and b == 0),
        "current_only": sum(1 for a, b in zip(repaired, current) if a == 0 and b > 0),
        "neither": sum(1 for a, b in zip(repaired, current) if a == 0 and b == 0),
    }


def write_report(
    analyses, records, labels, outcomes, out_dir: Path, shortlisted, png_paths,
    extra_case=None, extra_png=None, data_dir: Path = DATA_DIR,
):
    n = sum(labels.values())
    lines = ["# Repaired Attempt 1 triplet association — failure modes", ""]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Dataset: `{data_dir}` ({len(records)} photos, payload `{EXPECTED_PAYLOAD}`)")
    lines.append(
        f"- Current pipeline: detection success {sum(r.detection_ok for r in records)}/{len(records)}, "
        f"decode success {sum(r.decode_ok for r in records)}/{len(records)}, "
        f"payload match {sum(r.payload_ok for r in records)}/{len(records)}"
    )
    lines.append(f"- Reference subset (detection success **and** payload match): {n}")
    lines.append(
        "- Reference triplet: the first `find_valid_triplets` result on the same finder set; "
        "decode success confirms its TL/TR/BL roles."
    )
    lines.append(
        f"- Repaired rule: `angle_tol = {ANGLE_TOL}` rad, main `offset_tol = {MAIN_OFFSET_TOL}`, "
        "`local_offset` normalized by the mean of the two segment lengths."
    )
    lines.append(
        "- The checked-in `benchmark_captured.json` runs on the downscaled `captured/` copy: "
        "49 detection, 40 payload, 64 with >= 3 finders, 49 with a current triplet."
    )
    lines.append("")
    lines.append("## Key findings")
    lines.append("")
    lines.append(
        f"1. On the {n} reference images the repaired rule at `offset_tol = {MAIN_OFFSET_TOL}` "
        f"returns the correct first triplet in {sum(a.label == 'success' for a in analyses)} "
        f"({100.0 * sum(a.label == 'success' for a in analyses) / n:.1f}%)."
    )
    lines.append(
        "2. Every failure is a **missed true pair**: no valid pairing at all. No true pair is "
        "outranked, no false association changes the first triplet, and no wrong triplet is "
        "returned in the subset. The rule fails closed."
    )
    low_counts, _, low_per = sweep_label_counts(records, SWEEP[0])
    low_fail = sum(c for k, c in low_counts.items() if k != "success")
    low_versions = sorted({a.version for a in low_per if a.label != "success"})
    tc_report = triplet_comparison(records)
    lines.append(
        f"3. At the original `offset_tol = {SWEEP[0]:.2f}`, {low_fail}/{n} images fail "
        f"(versions {', '.join('v' + str(v) for v in low_versions)}). "
        f"The local normalization plus `{MAIN_OFFSET_TOL:.2f}` recovers all of those; only "
        "v20 corners remain above threshold."
    )
    lines.append(
        f"4. Across all {tc_report['images']} images with >= 3 finders the repaired rule "
        f"produces a triplet on {tc_report['repaired_images']} images "
        f"({tc_report['repaired_total']} triplets) versus {tc_report['current_images']} images "
        f"({tc_report['current_total']} triplets) for centre geometry: lower triplet recall "
        "and at most one triplet per image."
    )
    lines.append(
        "5. Implementation check: on all 100 cached finder sets the association graph is "
        "identical to `e335de8:finder_pattern.py`; the only triplet differences are the "
        "intentional TR/BL swap from the `7f91833` `(x, y) = (col, row)` cross-product fix."
    )
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append(
        "- The reference subset is conditioned on the current detector succeeding and decoding. "
        "It is therefore the easier half of the dataset; it cannot say whether the repaired "
        "rule would pick the right triplet on the 9 detection-success/decode-failure images "
        "(no verified truth there)."
    )
    lines.append(
        "- On this captured set the repaired rule is not broadly brittle once the threshold is "
        "0.30. The blog's abandonment evidence — 16 failed / 2 passed synthetic detector tests "
        "after the corner fitter changed — is a different (synthetic, older corner-geometry) "
        "regime and is not reproduced by this analysis."
    )
    lines.append(
        "- All three failures are version 20, where the 7-module finder is only roughly "
        "50–100 px across in a ~3000 px code. There are no version <= 10 failures at the main "
        "threshold, so no such representative failure exists to shortlist."
    )
    e_main = extra_association_stats(records, MAIN_OFFSET_TOL)
    e_loose = extra_association_stats(records, SWEEP[-1])
    lines.append(
        f"- Extra false associations do appear as the tolerance loosens: {e_main[0]} extra "
        f"edges in {e_main[1]}/{n} images at `{MAIN_OFFSET_TOL:.2f}`, and {e_loose[0]} in "
        f"{e_loose[1]}/{n} at `{SWEEP[-1]:.2f}`. They never displace the first correct triplet "
        f"on this subset, but the loosest settings create {e_loose[2]} extra (ambiguous) "
        "triplets. The rule becomes less discriminative, not immediately wrong."
    )
    lines.append("")
    lines.append(f"## Outcomes at `offset_tol = {MAIN_OFFSET_TOL}`")
    lines.append("")
    lines.append("| outcome | count | share |")
    lines.append("|---|---:|---:|")
    for label, count in outcomes.most_common():
        lines.append(f"| {label} | {count} | {100.0 * count / n:.1f}% |")
    lines.append("")
    lines.append(f"## Root causes at `offset_tol = {MAIN_OFFSET_TOL}`")
    lines.append("")
    lines.append("| mode | count | share |")
    lines.append("|---|---:|---:|")
    for label, count in labels.most_common():
        lines.append(f"| {label} | {count} | {100.0 * count / n:.1f}% |")
    lines.append("")
    lines.append("## Threshold sweep (root causes)")
    lines.append("")
    lines.append("| offset_tol | success | no triplet | true pair missed | true pair outranked | false association | triplet selection | other |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tol in SWEEP:
        counts, _, _ = sweep_label_counts(records, tol)
        lines.append(
            f"| {tol:.2f} | {counts['success']} | {counts['no triplet']} | "
            f"{counts['true pair missed']} | {counts['true pair outranked']} | "
            f"{counts['false association']} | {counts['triplet selection']} | {counts['other']} |"
        )
    lines.append("")
    lines.append("## Threshold sweep (outcomes)")
    lines.append("")
    lines.append("| offset_tol | success | wrong triplet | no triplet |")
    lines.append("|---:|---:|---:|---:|")
    for tol in SWEEP:
        _, oc, _ = sweep_label_counts(records, tol)
        lines.append(
            f"| {tol:.2f} | {oc['success']} | {oc['wrong triplet']} | {oc['no triplet']} |"
        )
    lines.append("")
    lines.append("## Triplet counts on all images with >= 3 finders")
    lines.append("")
    tc = triplet_comparison(records)
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| images with >= 3 finders | {tc['images']} |")
    lines.append(f"| images with repaired >= 1 triplet | {tc['repaired_images']} |")
    lines.append(f"| images with current >= 1 triplet | {tc['current_images']} |")
    lines.append(f"| total repaired triplets | {tc['repaired_total']} |")
    lines.append(f"| total current triplets | {tc['current_total']} |")
    lines.append(f"| both produce a triplet | {tc['both']} |")
    lines.append(f"| repaired only | {tc['repaired_only']} |")
    lines.append(f"| current only | {tc['current_only']} |")
    lines.append(f"| neither | {tc['neither']} |")
    lines.append("")
    lines.append("## Version breakdown")
    lines.append("")
    for tol in (MAIN_OFFSET_TOL, SWEEP[0]):
        _, _, per = sweep_label_counts(records, tol)
        lines.append(f"### `offset_tol = {tol:.2f}`")
        lines.append("")
        lines.append("| version | n | success | missed | other |")
        lines.append("|---:|---:|---:|---:|---:|")
        for version, counts in sorted(version_breakdown(per).items()):
            total = sum(counts.values())
            missed = counts["true pair missed"]
            lines.append(
                f"| {version} | {total} | {counts['success']} | {missed} | "
                f"{total - counts['success'] - missed} |"
            )
        lines.append("")
    lines.append(f"## Failure scores at `offset_tol = {MAIN_OFFSET_TOL}`")
    lines.append("")
    lines.append(
        "| image | v | pair | true max offset | true total | true segs | selected pairing | "
        "best false |"
    )
    lines.append("|---|---:|---|---:|---:|---|---|---|")
    for a in analyses:
        if a.label == "success":
            continue
        for (pa, pb), p in a.true_pair_analyses.items():
            segs = ", ".join(f"{x}–{y}" for x, y in p.true_pairs)
            sel = (
                "none"
                if p.selected is None
                else ("true" if p.selected_is_true else f"{p.selected.pairs}")
            )
            false = p.best_false()
            lines.append(
                f"| {a.image} | {a.version} | {pa}–{pb} | "
                f"{p.true_pairing.max_offset:.3f} | {p.true_pairing.total:.3f} | {segs} | "
                f"{sel} | {fmt_score(false) if false else 'none'} |"
            )
    lines.append("")
    lines.append("## Per-image threshold matrix")
    lines.append("")
    lines.append("`.` = success, `M` = true pair missed, `O` = outranked, "
                 "`F` = false association, `S` = triplet selection, `?` = other")
    lines.append("")
    header = "| image | v | " + " | ".join(f"{t:.2f}" for t in SWEEP) + " |"
    lines.append(header)
    lines.append("|---|---:|" + "---:|" * len(SWEEP))
    by_image = {a.image: a for a in analyses}
    refs = [r for r in records if r.reference_triplet is not None]
    for rec in sorted(refs, key=lambda r: (r.version, r.image)):
        cells = []
        for tol in SWEEP:
            a = analyse_image(rec, tol)
            cells.append(
                {
                    "success": ".",
                    "true pair missed": "M",
                    "true pair outranked": "O",
                    "false association": "F",
                    "triplet selection": "S",
                }.get(a.label, "?")
            )
        lines.append(f"| {rec.image} | {rec.version} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-image detail")
    lines.append("")
    lines.append("| image | v | finders | outcome | root cause | detail |")
    lines.append("|---|---:|---:|---|---|---|")
    for a in analyses:
        lines.append(
            f"| {a.image} | {a.version} | {a.n_finders} | {a.outcome} | {a.label} | {a.detail} |"
        )
    lines.append("")
    lines.append("## Shortlisted cases")
    lines.append("")
    for a, png in zip(shortlisted, png_paths):
        lines.append(f"### {a.image} (v{a.version}, {a.label})")
        lines.append("")
        lines.append(f"![{a.image}]({png.name})")
        lines.append("")
        lines.append(a.detail)
        for (pa, pb), pa_an in a.true_pair_analyses.items():
            lines.append(f"- true pair {pa}-{pb}: {fmt_score(pa_an.true_pairing)} "
                         f"(pairs {pa_an.true_pairs}); "
                         f"best false: {fmt_score(pa_an.best_false())}")
        lines.append("")
    if extra_case is not None and extra_png is not None:
        lines.append("## Threshold-sensitivity case")
        lines.append("")
        lines.append(
            f"### {extra_case.image} (v{extra_case.version}, {extra_case.label} "
            f"at offset_tol={extra_case.offset_tol:.2f})"
        )
        lines.append("")
        lines.append(f"![{extra_case.image}]({extra_png.name})")
        lines.append("")
        lines.append(
            f"This is a clean three-finder version-{extra_case.version} image. It fails at "
            f"`offset_tol = {extra_case.offset_tol:.2f}` and succeeds at "
            f"`{MAIN_OFFSET_TOL:.2f}`; the failing true pair is only slightly above the lower "
            "threshold. It is included because no version <= 10 image fails at the main "
            "threshold."
        )
        lines.append("")
        for (pa, pb), pa_an in extra_case.true_pair_analyses.items():
            lines.append(
                f"- true pair {pa}-{pb}: {fmt_score(pa_an.true_pairing)} "
                f"(pairs {pa_an.true_pairs}); best false: {fmt_score(pa_an.best_false())}"
            )
        lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/rik/git/qr-reader")
    lines.append("# first run: decodes all 100 photos and writes the cache")
    lines.append(".venv/bin/python blog_figures/triplet/analyze_failure_modes.py --refresh")
    lines.append("# later runs: reuse /tmp/opencode/triplet_failure_modes/cache.pkl")
    lines.append(".venv/bin/python blog_figures/triplet/analyze_failure_modes.py")
    lines.append("```")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--refresh", action="store_true", help="ignore the cached pipeline results")
    ap.add_argument("--workers", type=int, default=12, help="parallel image workers")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_records(
        args.data_dir, args.cache, args.refresh, workers=args.workers
    )
    analyses, labels, outcomes = aggregate(records)

    chosen = shortlist(analyses)
    png_paths = []
    rec_map = {r.image: r for r in records}
    print(f"\nShortlisted cases ({len(chosen)}):")
    for a in chosen:
        png = args.out_dir / f"case_{Path(a.image).stem.replace(' ', '_')}.png"
        render_case(a, rec_map[a.image], png, args.data_dir)
        png_paths.append(png)
        print(f"  {a.image}: {a.label} (v{a.version}, {a.n_finders} finders) -> {png}")

    extra = threshold_sensitivity_case(records)
    extra_png = None
    if extra is not None:
        extra_png = args.out_dir / (
            f"case_{Path(extra.image).stem.replace(' ', '_')}"
            f"_tol{extra.offset_tol:.2f}.png"
        )
        render_case(extra, rec_map[extra.image], extra_png, args.data_dir)
        print(
            f"  threshold case: {extra.image}: {extra.label} at "
            f"{extra.offset_tol:.2f} (v{extra.version}) -> {extra_png}"
        )

    write_report(
        analyses, records, labels, outcomes, args.out_dir, chosen, png_paths,
        extra_case=extra, extra_png=extra_png, data_dir=args.data_dir,
    )
    print(f"\nreport: {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()