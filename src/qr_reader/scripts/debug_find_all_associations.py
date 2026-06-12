"""Debug finder-pattern association failures for high-version QR images.

This script mirrors ``scripts/full-pipeline.py`` up to ``find_all_associations(fps)``.
It then re-runs ``check_association()``-style predicates with verbose diagnostics
and includes experimental association strategies for investigation only.

Run from the repository root:

    uv run python src/qr_reader/scripts/debug_find_all_associations.py
    uv run python src/qr_reader/scripts/debug_find_all_associations.py --version 4
    uv run python src/qr_reader/scripts/debug_find_all_associations.py --sweep-versions 1:12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.corner import angular_nms_top_radial_indices
from qr_reader.detector.finder_pattern import (
    FinderPattern,
    check_association,
    extract_finder_patterns,
    find_all_associations,
)
from qr_reader.detector.geometry import (
    angular_distance,
    max_offset,
    point_line_distance,
    polygon_area,
    segments_intersect,
)
from qr_reader.detector.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.qr_gen import binarize_image, generate_test_image

DEFAULT_CONTENT = "https://www.rikvoorhaar.com"
DEFAULT_VERSION = 12
DEFAULT_BORDER = 15
ANGLE_TOL = 0.1
OFFSET_TOL = 0.15
LOCAL_OFFSET_TOL = 0.35


@dataclass(frozen=True)
class SegmentDebug:
    seg1_idx: int
    seg2_idx: int
    angle_rad: float
    angle_deg: float
    offset: float
    local_offset: float
    midpoint_distance: float
    max_abs_line_distance: float
    angle_ok: bool
    offset_ok: bool
    local_offset_ok: bool


@dataclass(frozen=True)
class PairDebug:
    fp1_idx: int
    fp2_idx: int
    intersections: list[tuple[int, int]]
    segment_rows: list[SegmentDebug]
    colinear_pairs: list[tuple[int, int]]
    production_result: object | None


@dataclass(frozen=True)
class ExperimentalAssociation:
    fp1_idx: int
    fp2_idx: int
    selected_pairs: list[tuple[int, int]]
    scores: list[float]
    max_local_offset: float
    max_angle_rad: float
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug why find_all_associations() rejects finder-pattern pairs."
    )
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--content", default=DEFAULT_CONTENT)
    parser.add_argument("--border", type=int, default=DEFAULT_BORDER)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--box-size", type=int, default=10)
    parser.add_argument("--perspective-max-shift", type=float, default=50.0)
    parser.add_argument("--noise-std", type=float, default=50.0)
    parser.add_argument("--noise-blur-kernel", type=int, default=3)
    parser.add_argument("--intensity-scale", type=float, default=0.8)
    parser.add_argument("--final-blur-kernel", type=int, default=5)
    parser.add_argument("--angle-tol", type=float, default=ANGLE_TOL)
    parser.add_argument("--offset-tol", type=float, default=OFFSET_TOL)
    parser.add_argument(
        "--local-offset-tol",
        type=float,
        default=LOCAL_OFFSET_TOL,
        help="Experimental offset tolerance when normalizing by local segment scale.",
    )
    parser.add_argument(
        "--sweep-versions",
        default=None,
        help="Optional inclusive version range, e.g. '1:12', for compact experiment output.",
    )
    parser.add_argument(
        "--sweep-seeds",
        default=None,
        help="Optional inclusive seed range, e.g. '0:8', for compact experiment output.",
    )
    return parser.parse_args()


def area_from_full_pipeline(corners: np.ndarray) -> float:
    """Same diagnostic area helper used in full-pipeline.py, without the 0.5."""
    diag1 = corners[0] - corners[2]
    diag2 = corners[1] - corners[3]
    return float(np.abs(np.linalg.det(np.vstack([diag1, diag2]))))


def segment_midpoint(seg: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return (
        np.asarray(seg[0], dtype=np.float64) + np.asarray(seg[1], dtype=np.float64)
    ) / 2


def segment_length(seg: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.linalg.norm(np.asarray(seg[1]) - np.asarray(seg[0])))


def max_abs_line_distance(
    s1: tuple[np.ndarray, np.ndarray], s2: tuple[np.ndarray, np.ndarray]
) -> float:
    return float(
        max(
            point_line_distance(s2[0], s1[0], s1[1]),
            point_line_distance(s2[1], s1[0], s1[1]),
            point_line_distance(s1[0], s2[0], s2[1]),
            point_line_distance(s1[1], s2[0], s2[1]),
        )
    )


def local_offset(
    s1: tuple[np.ndarray, np.ndarray], s2: tuple[np.ndarray, np.ndarray]
) -> float:
    """Offset normalized by local finder-pattern segment size, not inter-FP distance."""
    denom = (segment_length(s1) + segment_length(s2)) / 2
    if denom == 0:
        return float("inf")
    return max_abs_line_distance(s1, s2) / denom


def cluster_center(cluster: CandidateCluster) -> tuple[float, float]:
    return float(cluster.row), float((cluster.cols[2] + cluster.cols[3]) / 2)


def build_finder_patterns(
    args: argparse.Namespace,
) -> tuple[list[CandidateCluster], list[tuple[int, np.ndarray]], list[FinderPattern]]:
    """Follow full-pipeline.py until finder-pattern extraction."""
    img_gray = generate_test_image(
        seed=args.seed,
        content=args.content,
        version=args.version,
        box_size=args.box_size,
        border=args.border,
        perspective_max_shift=args.perspective_max_shift,
        noise_std=args.noise_std,
        noise_blur_kernel=args.noise_blur_kernel,
        intensity_scale=args.intensity_scale,
        final_blur_kernel=args.final_blur_kernel,
    )
    img_binary = binarize_image(img_gray)

    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    clusters = cluster_candidates(rows_valid, cols_valid_all)

    angular_distance_nms = 10 * 2 * np.pi / 360
    all_corners: list[tuple[int, np.ndarray]] = []
    for ci, cluster in enumerate(clusters):
        seed_row = int(cluster.row)
        seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2)
        region_mask = region_fill_wave_front(np.asarray(img_binary), seed_row, seed_col)
        boundary = region_boundary_8(region_mask)
        components = boundary_connected_components_ndimage(np.asarray(boundary))

        for comp in components:
            comp_arr = np.asarray(comp, dtype=np.float64)
            if comp_arr.shape[0] < 4:
                continue
            centroid_i = comp_arr.mean(axis=0)
            rd = np.linalg.norm(comp_arr - centroid_i, axis=1)
            ang = np.arctan2(
                comp_arr[:, 1] - centroid_i[1], comp_arr[:, 0] - centroid_i[0]
            )
            try:
                idx = angular_nms_top_radial_indices(
                    rd,
                    ang,
                    angular_nms_rad=angular_distance_nms,
                    k=4,
                )
            except ValueError:
                continue
            all_corners.append((ci, comp_arr[idx]))

    fps = extract_finder_patterns(all_corners)
    return clusters, all_corners, fps


def debug_pair(
    fp1: FinderPattern,
    fp2: FinderPattern,
    angle_tol: float,
    offset_tol: float,
    local_offset_tol: float,
) -> PairDebug:
    segs1 = fp1.segments()
    segs2 = fp2.segments()

    intersections = []
    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                intersections.append((i, j))

    rows = []
    colinear_pairs = []
    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            angle_rad = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
            offset = float(max_offset(s1[0], s1[1], s2[0], s2[1]))
            loc_offset = float(local_offset(s1, s2))
            angle_ok = angle_rad < angle_tol
            offset_ok = offset < offset_tol
            local_offset_ok = loc_offset < local_offset_tol
            if angle_ok and offset_ok:
                colinear_pairs.append((i, j))
            rows.append(
                SegmentDebug(
                    seg1_idx=i,
                    seg2_idx=j,
                    angle_rad=angle_rad,
                    angle_deg=float(np.rad2deg(angle_rad)),
                    offset=offset,
                    local_offset=loc_offset,
                    midpoint_distance=float(
                        np.linalg.norm(segment_midpoint(s2) - segment_midpoint(s1))
                    ),
                    max_abs_line_distance=max_abs_line_distance(s1, s2),
                    angle_ok=angle_ok,
                    offset_ok=offset_ok,
                    local_offset_ok=local_offset_ok,
                )
            )

    return PairDebug(
        fp1_idx=fp1.cluster_idx,
        fp2_idx=fp2.cluster_idx,
        intersections=intersections,
        segment_rows=rows,
        colinear_pairs=colinear_pairs,
        production_result=check_association(fp1, fp2, angle_tol, offset_tol),
    )


def iter_sorted_pair_debug(
    fps: Iterable[FinderPattern],
    angle_tol: float,
    offset_tol: float,
    local_offset_tol: float,
) -> Iterable[PairDebug]:
    for fp1, fp2 in combinations(fps, 2):
        yield debug_pair(fp1, fp2, angle_tol, offset_tol, local_offset_tol)


def experimental_best_two_association(
    fp1: FinderPattern,
    fp2: FinderPattern,
    *,
    angle_tol: float,
    local_offset_tol: float,
) -> ExperimentalAssociation | None:
    """Experimental association: local-scale offset plus best two opposite-edge pairs.

    This does not modify production code. It tests whether a more local
    normalization and selecting the best pair per opposite-side axis would recover
    valid high-version associations.
    """
    segs1 = fp1.segments()
    segs2 = fp2.segments()

    for s1 in segs1:
        for s2 in segs2:
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                return None

    candidates: list[ExperimentalAssociation] = []
    axes = ((0, 2), (1, 3))
    for axis1 in axes:
        for axis2 in axes:
            # Try both one-to-one pairings between the opposite sides. The corner
            # ordering is not guaranteed to make axis indices line up across FPs.
            pairings = (
                ((axis1[0], axis2[0]), (axis1[1], axis2[1])),
                ((axis1[0], axis2[1]), (axis1[1], axis2[0])),
            )
            for pairing in pairings:
                selected = []
                scores = []
                angles = []
                offsets = []
                pairing_ok = True

                for i, j in pairing:
                    s1 = segs1[i]
                    s2 = segs2[j]
                    angle = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
                    offset = float(local_offset(s1, s2))
                    score = offset + angle
                    if angle >= angle_tol or offset >= local_offset_tol:
                        pairing_ok = False
                        break
                    selected.append((i, j))
                    scores.append(float(score))
                    angles.append(angle)
                    offsets.append(offset)

                if pairing_ok and len(selected) == 2:
                    candidates.append(
                        ExperimentalAssociation(
                            fp1_idx=fp1.cluster_idx,
                            fp2_idx=fp2.cluster_idx,
                            selected_pairs=selected,
                            scores=scores,
                            max_local_offset=max(offsets),
                            max_angle_rad=max(angles),
                            method="local_offset_best_two_by_axis",
                        )
                    )

    if not candidates:
        return None
    return min(
        candidates, key=lambda assoc: (assoc.max_local_offset, sum(assoc.scores))
    )


def experimental_associations(
    fps: list[FinderPattern], *, angle_tol: float, local_offset_tol: float
) -> list[ExperimentalAssociation]:
    assocs = []
    for fp1, fp2 in combinations(fps, 2):
        assoc = experimental_best_two_association(
            fp1,
            fp2,
            angle_tol=angle_tol,
            local_offset_tol=local_offset_tol,
        )
        if assoc is not None:
            assocs.append(assoc)
    return assocs


def parse_inclusive_range(range_text: str) -> range:
    start_text, end_text = range_text.split(":", maxsplit=1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError("sweep range end must be >= start")
    return range(start, end + 1)


def compact_association_summary(
    args: argparse.Namespace,
) -> tuple[int, int, int, str, int, str]:
    clusters, _all_corners, fps = build_finder_patterns(args)
    production = find_all_associations(fps, args.angle_tol, args.offset_tol)
    experimental = experimental_associations(
        fps,
        angle_tol=args.angle_tol,
        local_offset_tol=args.local_offset_tol,
    )
    production_pairs = ";".join(
        f"{a.fp1_idx}-{a.fp2_idx}:{a.colinear_segments_1}/{a.colinear_segments_2}"
        for a in production
    )
    experimental_pairs = ";".join(
        f"{a.fp1_idx}-{a.fp2_idx}:{a.selected_pairs}:off={a.max_local_offset:.3f},ang={np.rad2deg(a.max_angle_rad):.2f}"
        for a in experimental
    )
    return (
        len(clusters),
        len(fps),
        len(production),
        production_pairs,
        len(experimental),
        experimental_pairs,
    )


def run_sweep(args: argparse.Namespace) -> int:
    versions = (
        parse_inclusive_range(args.sweep_versions)
        if args.sweep_versions
        else range(args.version, args.version + 1)
    )
    seeds = (
        parse_inclusive_range(args.sweep_seeds)
        if args.sweep_seeds
        else range(args.seed, args.seed + 1)
    )
    print(
        "version,seed,clusters,fps,production_count,production_pairs,experimental_count,experimental_pairs"
    )
    for version in versions:
        for seed in seeds:
            args_for_run = argparse.Namespace(**vars(args))
            args_for_run.version = version
            args_for_run.seed = seed
            clusters_count, fps_count, prod_count, prod_pairs, exp_count, exp_pairs = (
                compact_association_summary(args_for_run)
            )
            print(
                f"{version},{seed},{clusters_count},{fps_count},{prod_count},{prod_pairs},"
                f"{exp_count},{exp_pairs}"
            )
    return 0


def print_clusters(clusters: list[CandidateCluster]) -> None:
    print(f"\nClusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        center = cluster_center(cluster)
        print(
            f"  cluster {i}: row={cluster.row:.2f}, center=(row={center[0]:.2f}, col={center[1]:.2f}), "
            f"height={cluster.height:.2f}, candidates={cluster.num_candidates}, "
            f"cols={np.array2string(cluster.cols, precision=2)}"
        )


def print_all_corners(all_corners: list[tuple[int, np.ndarray]]) -> None:
    print(f"\nCorner quads from boundary components: {len(all_corners)}")
    for ci, corners in all_corners:
        print(
            f"  cluster {ci}: polygon_area={polygon_area(corners):.2f}, "
            f"full_pipeline_area={area_from_full_pipeline(corners):.2f}, "
            f"corners(row,col)={np.array2string(corners, precision=1)}"
        )


def print_finder_patterns(fps: list[FinderPattern]) -> None:
    print(f"\nExtracted finder patterns: {len(fps)}")
    for fp in fps:
        center = fp.outer_corners.mean(axis=0)
        print(
            f"  FP cluster {fp.cluster_idx}: center(row,col)=({center[0]:.2f}, {center[1]:.2f}), "
            f"outer_area={polygon_area(fp.outer_corners):.2f}, "
            f"has_inner={fp.inner_corners is not None}"
        )
        print(
            f"    outer corners(row,col)={np.array2string(fp.outer_corners, precision=1)}"
        )
        for si, (p1, p2) in enumerate(fp.segments()):
            v = np.asarray(p2) - np.asarray(p1)
            print(
                f"    seg {si}: p1={np.array2string(p1, precision=1)}, "
                f"p2={np.array2string(p2, precision=1)}, vector={np.array2string(v, precision=1)}, "
                f"length={np.linalg.norm(v):.2f}"
            )


def print_pair_debug(
    pair: PairDebug, angle_tol: float, offset_tol: float, local_offset_tol: float
) -> None:
    if pair.intersections:
        reason = f"REJECT: segment intersections {pair.intersections}"
    elif len(pair.colinear_pairs) != 2:
        reason = (
            f"REJECT: expected exactly 2 colinear pairs, got {len(pair.colinear_pairs)}"
        )
    else:
        reason = "ACCEPT"

    print(f"\nPair FP {pair.fp1_idx} <-> FP {pair.fp2_idx}: {reason}")
    print(f"  production check_association returned: {pair.production_result}")
    print(
        f"  angle_tol={angle_tol:.6f} rad ({np.rad2deg(angle_tol):.3f} deg), "
        f"offset_tol={offset_tol:.6f}, local_offset_tol={local_offset_tol:.3f}"
    )
    print(f"  colinear_pairs passing production predicates: {pair.colinear_pairs}")

    interesting = [
        r for r in pair.segment_rows if r.angle_ok or r.offset_ok or r.local_offset_ok
    ]
    if not interesting:
        interesting = sorted(
            pair.segment_rows, key=lambda r: (r.angle_rad, r.local_offset)
        )[:4]
        print(
            "  no segment pairs passed any predicate; showing four smallest angle/local-offset candidates"
        )
    else:
        interesting = sorted(
            interesting,
            key=lambda r: (
                not (r.angle_ok and r.local_offset_ok),
                r.local_offset,
                r.angle_rad,
            ),
        )
        print(
            "  segment pairs passing angle, production-offset, and/or local-offset predicates:"
        )

    for r in interesting:
        print(
            f"    s{r.seg1_idx}-s{r.seg2_idx}: "
            f"angle={r.angle_rad:.6f} rad/{r.angle_deg:.3f} deg "
            f"({'ok' if r.angle_ok else 'fail'}), "
            f"prod_offset={r.offset:.6f} ({'ok' if r.offset_ok else 'fail'}), "
            f"local_offset={r.local_offset:.3f} ({'ok' if r.local_offset_ok else 'fail'}), "
            f"mid_dist={r.midpoint_distance:.2f}, max_line_dist={r.max_abs_line_distance:.2f}"
        )


def print_experimental_details(
    fps: list[FinderPattern], *, angle_tol: float, local_offset_tol: float
) -> None:
    print("\nExperimental local-scale best-two associations:")
    assocs = experimental_associations(
        fps,
        angle_tol=angle_tol,
        local_offset_tol=local_offset_tol,
    )
    print(
        f"  returned {len(assocs)} association(s) with local_offset_tol={local_offset_tol:.3f}"
    )
    for assoc in assocs:
        print(
            f"  FP {assoc.fp1_idx} <-> FP {assoc.fp2_idx}: pairs={assoc.selected_pairs}, "
            f"max_local_offset={assoc.max_local_offset:.3f}, "
            f"max_angle={np.rad2deg(assoc.max_angle_rad):.3f} deg, "
            f"scores={[round(s, 4) for s in assoc.scores]}"
        )

    for fp1, fp2 in combinations(fps, 2):
        print(
            f"\n  Pair FP {fp1.cluster_idx} <-> FP {fp2.cluster_idx} local-offset candidates:"
        )
        segs1 = fp1.segments()
        segs2 = fp2.segments()
        rows = []
        for i, s1 in enumerate(segs1):
            for j, s2 in enumerate(segs2):
                angle = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
                offset = float(local_offset(s1, s2))
                if angle < angle_tol or offset < local_offset_tol:
                    rows.append((offset + angle, i, j, angle, offset))
        for _score, i, j, angle, offset in sorted(rows)[:8]:
            print(
                f"    s{i}-s{j}: local_offset={offset:.3f} "
                f"({'ok' if offset < local_offset_tol else 'fail'}), "
                f"angle={np.rad2deg(angle):.3f} deg ({'ok' if angle < angle_tol else 'fail'})"
            )


def main() -> int:
    args = parse_args()
    if args.sweep_versions is not None or args.sweep_seeds is not None:
        return run_sweep(args)

    print("Debug find_all_associations reproduction")
    print(
        f"  version={args.version}, content={args.content!r}, border={args.border}, seed={args.seed}, "
        f"noise_std={args.noise_std}, perspective_max_shift={args.perspective_max_shift}"
    )

    clusters, all_corners, fps = build_finder_patterns(args)
    print_clusters(clusters)
    print_all_corners(all_corners)
    print_finder_patterns(fps)

    associations = find_all_associations(fps, args.angle_tol, args.offset_tol)
    print(f"\nfind_all_associations returned {len(associations)} association(s):")
    for assoc in associations:
        print(f"  {assoc}")
    if len(associations) == 0:
        print("  CONFIRMED: no associations found for this reproduction.")

    print("\nDetailed check_association diagnostics:")
    for pair in iter_sorted_pair_debug(
        fps, args.angle_tol, args.offset_tol, args.local_offset_tol
    ):
        print_pair_debug(pair, args.angle_tol, args.offset_tol, args.local_offset_tol)

    print_experimental_details(
        fps,
        angle_tol=args.angle_tol,
        local_offset_tol=args.local_offset_tol,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
