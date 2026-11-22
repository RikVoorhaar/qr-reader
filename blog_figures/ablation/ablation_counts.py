#!/usr/bin/env python3
"""Per-image stage attribution for the 100-image captured benchmark.

Instruments the first-pass candidate loop (up to 5 triplets x 2 versions) and
cross-checks it against the actual ``decode_image`` trace. Fallback recoveries
are attributed using the successful attempt's provenance, not a simulated
copy of the retry logic.

The unit is the image: an image "reaches" a stage if at least one candidate
gets a usable output from that stage.  An image that never decodes is lost at
the furthest stage it reached; the recorded failure reason is why it could not
get past that stage.

Writes ``ablation_counts.json`` next to this script and prints a compact table.

Run from the repository root:

    .venv/bin/python blog_figures/ablation/ablation_counts.py
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from qr_reader.decoder.decoder import decode as decode_bits
from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.detector import (
    DetectionContext,
    DetectionResult,
    DetectionTimings,
    _fit_candidate_homography,
    _rank_triplets,
)
from qr_reader.detector.edge_fitting import DISTANCE_THRESHOLD, MAX_GAP
from qr_reader.detector.finder_pattern import (
    FinderPattern, find_all_associations, find_triplets,
)
from qr_reader.detector.landmarks_refined import build_refined_homography
from qr_reader.detector.ray_fit import fit_finder_ray
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.detector.sample import sample_qr_bits
from qr_reader.preprocess import binarize_image_illumination_corrected, to_gray
from qr_reader.pipeline import decode_image

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "bench" / "captured"
OUT_PATH = Path(__file__).resolve().parent / "ablation_counts.json"
PAYLOAD = "rikvoorhaar.com"

MAX_TRIPLETS = 5
MAX_VERSIONS_PER_TRIPLET = 2
TIME_BUDGET_S = 15.0

STAGES = [
    "binarize",
    "find_candidates",
    "cluster",
    "fit_finders",
    "find_triplets",
    "rank_versions",
    "fit_homography",
    "refine_alignment",
    "sample",
    "decode",
]


def _natural_sort_key(path: Path) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", path.stem))


def _image_paths(data_dir: Path) -> list[Path]:
    return sorted(
        list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.jpg")),
        key=_natural_sort_key,
    )


def _prep(
    gray: np.ndarray,
) -> tuple[dict[str, bool], dict[str, list[str]], DetectionContext | None, dict]:
    """Run the preprocessing stages, recording which ones succeeded.

    Mirrors ``detector._prepare_detection`` exactly, but keeps going far enough
    to attribute a failure to a specific stage.
    """
    reached = {s: False for s in STAGES}
    reasons: dict[str, list[str]] = {s: [] for s in STAGES}
    info = {"n_candidates": 0, "n_finder_patterns": 0, "n_triplets": 0}

    timings = DetectionTimings()
    img_gray = to_gray(gray)

    try:
        img_binary = binarize_image_illumination_corrected(img_gray)
    except Exception as exc:  # noqa: BLE001
        reasons["binarize"].append(f"{type(exc).__name__}: {exc}")
        return reached, reasons, None, info
    reached["binarize"] = True

    try:
        rows_valid, cols_valid_all = find_alignment_patterns_2d(
            img_binary, np.log(1.3)
        )
    except Exception as exc:  # noqa: BLE001
        reasons["find_candidates"].append(f"{type(exc).__name__}: {exc}")
        return reached, reasons, None, info
    if len(rows_valid) == 0:
        reasons["find_candidates"].append("no alignment pattern candidates")
        return reached, reasons, None, info
    reached["find_candidates"] = True

    try:
        clusters = cluster_candidates(rows_valid, cols_valid_all)
    except Exception as exc:  # noqa: BLE001
        reasons["cluster"].append(f"{type(exc).__name__}: {exc}")
        return reached, reasons, None, info
    info["n_candidates"] = len(clusters)
    if not clusters:
        reasons["cluster"].append("no candidate clusters")
        return reached, reasons, None, info
    reached["cluster"] = True

    # Per-cluster finder fitting (identical to _prepare_detection).
    fps: list[FinderPattern] = []
    score_map: dict[int, float] = {}
    global_corners_xy: dict[int, np.ndarray] = {}
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(img_gray, bbox)
        if roi.size == 0:
            continue

        r0 = max(0, int(bbox[0]))
        c0 = max(0, int(bbox[2]))

        c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
        c_row = float(cluster.row) - r0
        center_xy = np.array([c_col, c_row], dtype=np.float64)
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

        fr = fit_finder_ray(
            roi,
            center_xy,
            m_est,
            max_gap=MAX_GAP,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        if not fr.valid:
            continue

        corners_xy_global = fr.corners + np.array([c0, r0], dtype=np.float64)
        wh = np.ptp(corners_xy_global, axis=0)
        if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est:
            continue

        corners_rc = corners_xy_global[:, ::-1]
        fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners_rc))
        score_map[ci] = fr.score
        global_corners_xy[ci] = corners_xy_global

    if not fps:
        reasons["fit_finders"].append("no finder patterns fitted")
        return reached, reasons, None, info

    # Deduplicate overlapping finder candidates (identical to _prepare_detection).
    keep_mask = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep_mask[i]:
            continue
        ci = fps[i].outer_corners.mean(axis=0)
        seg_i = float(
            np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1])
        )
        for j in range(i + 1, len(fps)):
            if not keep_mask[j]:
                continue
            cj = fps[j].outer_corners.mean(axis=0)
            seg_j = float(
                np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1])
            )
            if float(np.linalg.norm(ci - cj)) < 1.0 * min(seg_i, seg_j):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    fps = [fp for fp, keep in zip(fps, keep_mask) if keep]
    info["n_finder_patterns"] = len(fps)
    if not fps:
        reasons["fit_finders"].append("no finder patterns after deduplication")
        return reached, reasons, None, info
    reached["fit_finders"] = True

    try:
        triplets = find_triplets(fps, find_all_associations(fps, offset_tol=0.62))
    except Exception as exc:  # noqa: BLE001
        reasons["find_triplets"].append(f"{type(exc).__name__}: {exc}")
        return reached, reasons, None, info
    info["n_triplets"] = len(triplets)
    if not triplets:
        reasons["find_triplets"].append("no finder-pattern triplet found")
        return reached, reasons, None, info
    reached["find_triplets"] = True

    context = DetectionContext(
        img_gray=img_gray,
        fps=fps,
        score_map=score_map,
        global_corners_xy=global_corners_xy,
        triplets=triplets,
        timings=timings,
        n_candidates=info["n_candidates"],
        n_finder_patterns=info["n_finder_patterns"],
    )
    return reached, reasons, context, info


def evaluate_image(img_path: Path) -> dict:
    """Run the instrumented pipeline on one image and return its record."""
    t_total = perf_counter()
    reached = {s: False for s in STAGES}
    reasons: dict[str, list[str]] = {s: [] for s in STAGES}

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        reasons["binarize"].append("cv2.imread returned None")
        return _record(img_path, reached, reasons, 0, None, 0, 0, 0)

    reached, reasons, context, info = _prep(gray)

    ranked = []
    if reached["find_triplets"] and context is not None:
        try:
            ranked = _rank_triplets(context)
        except Exception as exc:  # noqa: BLE001
            reasons["rank_versions"].append(f"{type(exc).__name__}: {exc}")
        else:
            if ranked:
                reached["rank_versions"] = True
            else:
                reasons["rank_versions"].append("no ranked triplets")

    n_attempts = 0
    decoded_version: int | None = None
    first_homography_version: int | None = None
    budget_exhausted = False

    if reached["rank_versions"]:
        for candidate in ranked[:MAX_TRIPLETS]:
            for version in candidate.version_candidates[:MAX_VERSIONS_PER_TRIPLET]:
                if perf_counter() - t_total >= TIME_BUDGET_S:
                    budget_exhausted = True
                    break
                n_attempts += 1
                N = 4 * version + 17

                # Stage 7: 12-corner DLT + LM homography.
                try:
                    H, _err = _fit_candidate_homography(
                        context, candidate.triplet, N
                    )
                except Exception as exc:  # noqa: BLE001
                    reasons["fit_homography"].append(
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                reached["fit_homography"] = True
                if first_homography_version is None:
                    first_homography_version = version

                # Stage 8: alignment-pattern-refined homography.  A failure
                # here fails this candidate (no silent fallback).
                try:
                    H, _stats = build_refined_homography(
                        context.img_gray,
                        H,
                        candidate.triplet,
                        context.fps,
                        version,
                    )
                except Exception as exc:  # noqa: BLE001
                    reasons["refine_alignment"].append(
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                reached["refine_alignment"] = True

                # Stage 9: sample the module matrix.
                try:
                    bits = sample_qr_bits(context.img_gray, H, N)
                except Exception as exc:  # noqa: BLE001
                    reasons["sample"].append(f"{type(exc).__name__}: {exc}")
                    continue
                reached["sample"] = True

                # Stage 10: decode to the exact payload.
                try:
                    text = decode_bits(bits)
                except Exception as exc:  # noqa: BLE001
                    reasons["decode"].append(f"{type(exc).__name__}: {exc}")
                    continue
                if text != PAYLOAD:
                    reasons["decode"].append(
                        f"payload mismatch: decoded {text!r}"
                    )
                    continue

                reached["decode"] = True
                decoded_version = version
                break
            if reached["decode"] or budget_exhausted:
                break

    if budget_exhausted:
        reasons["decode"].append("time budget exhausted")

    version = decoded_version if decoded_version is not None else first_homography_version
    record = _record(
        img_path,
        reached,
        reasons,
        n_attempts,
        version,
        info["n_candidates"],
        info["n_finder_patterns"],
        info["n_triplets"],
    )
    # Run the actual cascade rather than duplicating its retry logic. The
    # instrumented first pass above supplies stage-level diagnostics; verify it
    # agrees with the production trace before attributing a recovery.
    result = decode_image(gray, expected_payload=PAYLOAD)
    first_pass = any(a.success and a.provenance == "production" for a in result.attempts)
    if first_pass != reached["decode"]:
        raise RuntimeError(f"{img_path.name}: instrumented and production first passes disagree")
    if (result.n_candidates, result.n_finder_patterns, result.n_triplets) != (
        info["n_candidates"], info["n_finder_patterns"], info["n_triplets"]
    ):
        raise RuntimeError(f"{img_path.name}: instrumented and production stage counts disagree")
    successes = [a.provenance for a in result.attempts if a.success]
    record["final_decoded"] = result.success
    record["success_provenance"] = successes[0] if successes else None
    return record


def _record(
    img_path: Path,
    reached: dict[str, bool],
    reasons: dict[str, list[str]],
    n_attempts: int,
    version: int | None,
    n_candidates: int,
    n_finder_patterns: int,
    n_triplets: int,
) -> dict:
    decoded = reached["decode"]
    furthest = "binarize"
    for stage in STAGES:
        if reached[stage]:
            furthest = stage
        else:
            break

    if decoded:
        failure_reason = ""
    else:
        # The image is lost at the stage after the furthest one it reached.
        idx = STAGES.index(furthest)
        next_stage = STAGES[idx + 1] if idx + 1 < len(STAGES) else None
        if next_stage is None:
            failure_reason = ""
        else:
            candidates = reasons.get(next_stage, [])
            failure_reason = (
                Counter(candidates).most_common(1)[0][0] if candidates else ""
            )

    return {
        "image": img_path.name,
        "decoded": decoded,
        "payload": PAYLOAD if decoded else "",
        "furthest_stage": furthest,
        "failure_reason": failure_reason,
        "attempts": n_attempts,
        "version": version,
        "n_candidates": n_candidates,
        "n_finder_patterns": n_finder_patterns,
        "n_triplets": n_triplets,
    }


def _commit_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    commit = _commit_hash()
    paths = _image_paths(DATA_DIR)
    if not paths:
        raise SystemExit(f"no JPEGs found in {DATA_DIR}")

    print(f"Attributing {len(paths)} images from {DATA_DIR} ...\n")
    images: list[dict] = []
    for i, path in enumerate(paths):
        images.append(evaluate_image(path))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)}...")

    counts = {stage: sum(1 for r in images if _reaches(r, stage)) for stage in STAGES}
    losses: dict[str, int] = {}
    prev = len(images)
    for stage in STAGES:
        losses[stage] = prev - counts[stage]
        prev = counts[stage]

    output = {
        "config": {
            "commit": commit,
            "dataset": str(DATA_DIR.relative_to(REPO_ROOT)),
            "payload": PAYLOAD,
        },
        "counts": counts,
        "losses": losses,
        "recoveries": dict(Counter(
            r["success_provenance"] for r in images
            if r["success_provenance"] not in (None, "production")
        )),
        "final_decoded": sum(r["final_decoded"] for r in images),
        "images": images,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2) + "\n")

    _print_table(images, counts, losses)
    print(f"\nwrote {OUT_PATH}")


def _reaches(record: dict, stage: str) -> bool:
    """True if the image reached *stage* (all stages up to furthest_stage)."""
    return STAGES.index(record["furthest_stage"]) >= STAGES.index(stage)


def _print_table(
    images: list[dict], counts: dict[str, int], losses: dict[str, int]
) -> None:
    n = len(images)
    print(f"\n{'stage':<18} {'reached':>8} {'lost':>6}  failure reasons")
    print(f"{'-' * 18} {'-------':>8} {'----':>6}  {'-' * 40}")
    for stage in STAGES:
        # Reasons for images lost at this stage: furthest_stage is the previous
        # stage and the failure reason belongs to this stage.
        idx = STAGES.index(stage)
        if idx == 0:
            lost_here = [r for r in images if not _reaches(r, stage)]
        else:
            prev_stage = STAGES[idx - 1]
            lost_here = [
                r
                for r in images
                if r["furthest_stage"] == prev_stage and not _reaches(r, stage)
            ]
        reason_counts = Counter(
            r["failure_reason"] for r in lost_here if r["failure_reason"]
        )
        top = "; ".join(
            f"{reason} ({count})" for reason, count in reason_counts.most_common(3)
        )
        print(f"{stage:<18} {counts[stage]:>8} {losses[stage]:>6}  {top}")

    recoveries = Counter(r["success_provenance"] for r in images)
    print(f"\nFirst-pass exact payload: {counts['decode']}/{n}")
    for provenance in ("scale_retry", "base_global_retry", "refined_spatial_retry"):
        print(f"  {provenance}: +{recoveries[provenance]}")
    print(f"Final exact payload: {sum(r['final_decoded'] for r in images)}/{n}")

    few_candidates = sum(1 for r in images if r["n_candidates"] < 3)
    few_finders = sum(1 for r in images if r["n_finder_patterns"] < 3)
    print(f"Images with <3 candidate clusters: {few_candidates}/{n}")
    print(f"Images with <3 finder patterns:    {few_finders}/{n}")


if __name__ == "__main__":
    main()
