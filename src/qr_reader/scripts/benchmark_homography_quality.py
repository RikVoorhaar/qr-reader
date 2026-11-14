"""Measure homography / correspondence quality: baseline vs. refined landmarks.

For each generated image we reconstruct the exact ground-truth homography
``H_gt`` (grid → image) by replaying the deterministic transform chain in
:func:`qr_reader.qr_gen.generate_test_image`.  We then fit several homographies
from different correspondence sets and measure how well each one reproduces
``H_gt`` over a dense grid of module centres.

Variants compared:

  baseline         12 outer finder corners (current pipeline)
  inner             + 12 inner finder corners (white-ring square)
  inner_align       + interior alignment-pattern centres (all matches)
  inner_align_filt  + alignment centres with confidence >= threshold
  inner_align_wgt   + alignment centres, weighted LM by confidence

Metrics (all in image pixels, lower is better):

  corr_mean        mean |detected landmark − GT landmark|
  dense_mean       mean |H(grid) − H_gt(grid)| over module centres
  dense_p90        90th percentile of the dense error
  dense_max        worst-case dense error

Run with ``python -m qr_reader.scripts.benchmark_homography_quality``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from time import perf_counter

import cv2
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.detector import (
    _canonicalize_corners,
    _fit_candidate_homography,
    _triplet_axes,
)
from qr_reader.detector.edge_fitting import DISTANCE_THRESHOLD, MAX_GAP
from qr_reader.detector.finder_pattern import FinderPattern, find_valid_triplets
from qr_reader.detector.homography import project_points
from qr_reader.detector.landmarks import (
    build_named_landmarks,
)
from qr_reader.detector.landmarks_refined import (
    Correspondence,
    alignment_centers_grid,
    fit_finder_inner_corners,
    fit_homography_from_correspondences,
    match_alignment_pattern,
)
from qr_reader.detector.ray_fit import fit_finder_ray
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.detector.version import (
    build_constraints,
    estimate_version,
    filter_constraints,
)
from qr_reader.qr_gen import (
    binarize_image_illumination_corrected,
    generate_test_image,
    make_qr_image,
)


# ---------------------------------------------------------------------------
# Ground-truth homography
# ---------------------------------------------------------------------------


def _actual_version(content: str, requested_version: int, box_size: int, border: int) -> int:
    import qrcode
    from qrcode.constants import ERROR_CORRECT_L

    qr = qrcode.QRCode(
        version=requested_version,
        error_correction=ERROR_CORRECT_L,
        box_size=box_size,
        border=border,
    )
    qr.add_data(content)
    qr.make(fit=True)
    return int(qr.version)


def compute_gt_homography(
    version: int,
    content: str,
    box_size: int,
    border: int,
    seed: int,
    perspective_max_shift: float,
    rotation_angle_deg: float | None,
) -> tuple[np.ndarray, int]:
    """Return (H_gt, N) mapping canonical grid (x, y) → image (x, y).

    Replays the exact transform chain of ``generate_test_image`` and composes
    the rotation and perspective matrices analytically.
    """
    rng = np.random.default_rng(seed)

    clean = make_qr_image(content=content, version=version, box_size=box_size, border=border)
    h, w = clean.shape
    N = 17 + 4 * version

    # Canonical grid (module units) → clean image pixels.
    corner_px = border * box_size
    H_clean = np.array(
        [
            [box_size, 0.0, corner_px],
            [0.0, box_size, corner_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Rotation (same random draw as generate_test_image).
    if rotation_angle_deg is None:
        rotation_amount = rng.uniform(0, 2 * np.pi)
    else:
        rotation_amount = np.deg2rad(rotation_angle_deg)
    angle_deg = float(np.rad2deg(rotation_amount))
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    M_rot_23 = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M_rot = np.vstack([M_rot_23, [0.0, 0.0, 1.0]])

    # Perspective warp (same random shifts as generate_test_image).
    src_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    shifts = rng.uniform(0, perspective_max_shift, size=(4, 2)).astype(np.float32)
    dst_pts = src_pts + shifts
    dst_pts[:, 0] = np.clip(dst_pts[:, 0], 0, w - 1)
    dst_pts[:, 1] = np.clip(dst_pts[:, 1], 0, h - 1)
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts).astype(np.float64)

    H_gt = M_persp @ M_rot @ H_clean
    H_gt = H_gt / H_gt[2, 2]
    return H_gt, N


# ---------------------------------------------------------------------------
# Canonical finder landmark grid coordinates (x, y)
# ---------------------------------------------------------------------------


def finder_grid_coords(N: int) -> dict[str, np.ndarray]:
    """Canonical outer + inner finder corners in (x, y), [TL, TR, BR, BL]."""
    outer = {
        "TL": np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64),
        "TR": np.array([[N - 7, 0], [N, 0], [N, 7], [N - 7, 7]], dtype=np.float64),
        "BL": np.array([[0, N - 7], [7, N - 7], [7, N], [0, N]], dtype=np.float64),
    }
    inner = {
        "TL": np.array([[1, 1], [6, 1], [6, 6], [1, 6]], dtype=np.float64),
        "TR": np.array([[N - 6, 1], [N - 1, 1], [N - 1, 6], [N - 6, 6]], dtype=np.float64),
        "BL": np.array([[1, N - 6], [6, N - 6], [6, N - 1], [1, N - 1]], dtype=np.float64),
    }
    return {"outer": outer, "inner": inner}


# ---------------------------------------------------------------------------
# Detection (mirrors detector._prepare_detection but keeps per-cluster info)
# ---------------------------------------------------------------------------


@dataclass
class Detection:
    img_gray: np.ndarray
    fps: list[FinderPattern]
    score_map: dict[int, float]
    global_outer_xy: dict[int, np.ndarray]
    cluster_info: dict[int, dict]
    triplets: list
    best_triplet: object
    estimated_version: int


def detect(img_gray: np.ndarray) -> Detection | None:
    img_binary = binarize_image_illumination_corrected(img_gray)
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    if len(rows_valid) == 0:
        return None
    clusters = cluster_candidates(rows_valid, cols_valid_all)

    fps: list[FinderPattern] = []
    score_map: dict[int, float] = {}
    global_outer_xy: dict[int, np.ndarray] = {}
    cluster_info: dict[int, dict] = {}

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
            roi, center_xy, m_est,
            max_gap=MAX_GAP, distance_threshold=DISTANCE_THRESHOLD,
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
        global_outer_xy[ci] = corners_xy_global
        cluster_info[ci] = dict(
            roi=roi,
            center_xy=center_xy,
            offset=np.array([c0, r0], dtype=np.float64),
            m_est=m_est,
        )

    if not fps:
        return None

    # Deduplicate (same rule as detector._prepare_detection).
    keep_mask = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep_mask[i]:
            continue
        ci = fps[i].outer_corners.mean(axis=0)
        seg_i = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
        for j in range(i + 1, len(fps)):
            if not keep_mask[j]:
                continue
            cj = fps[j].outer_corners.mean(axis=0)
            seg_j = float(np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1]))
            if float(np.linalg.norm(ci - cj)) < 1.0 * min(seg_i, seg_j):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break
    fps = [fp for fp, keep in zip(fps, keep_mask) if keep]
    if not fps:
        return None

    triplets = find_valid_triplets(fps, score_map)
    if not triplets:
        return None

    # Rank by cross-ratio (mirror detector._rank_triplets) and pick the best.
    best = None
    best_score = np.inf
    best_version = None
    for triplet in triplets:
        landmarks = build_named_landmarks(triplet, fps)
        constraints = build_constraints(landmarks)
        if not constraints:
            continue
        filtered = filter_constraints(constraints)
        if not filtered:
            continue
        v, scores = estimate_version(filtered)
        if not np.isfinite(scores[v - 1]):
            continue
        if scores[v - 1] < best_score:
            best_score = scores[v - 1]
            best = triplet
            best_version = v

    if best is None or best_version is None:
        return None

    return Detection(
        img_gray=img_gray,
        fps=fps,
        score_map=score_map,
        global_outer_xy=global_outer_xy,
        cluster_info=cluster_info,
        triplets=triplets,
        best_triplet=best,
        estimated_version=best_version,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def dense_error_stats(H: np.ndarray, H_gt: np.ndarray, N: int) -> dict[str, float]:
    rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    grid = np.column_stack([cols.ravel() + 0.5, rows.ravel() + 0.5])
    gt = project_points(H_gt, grid)
    est = project_points(H, grid)
    err = np.linalg.norm(est - gt, axis=1)
    return {
        "dense_mean": float(np.mean(err)),
        "dense_median": float(np.median(err)),
        "dense_p90": float(np.percentile(err, 90)),
        "dense_max": float(np.max(err)),
    }


def correspondence_error(image_xy: np.ndarray, grid_xy: np.ndarray, H_gt: np.ndarray) -> float:
    gt = project_points(H_gt, grid_xy[None, :])[0]
    return float(np.linalg.norm(np.asarray(image_xy) - gt))


# ---------------------------------------------------------------------------
# Per-image measurement
# ---------------------------------------------------------------------------


def _local_module_px(H: np.ndarray, gx: float, gy: float) -> float:
    p0 = project_points(H, np.array([[gx, gy]]))[0]
    px = project_points(H, np.array([[gx + 1.0, gy]]))[0]
    py = project_points(H, np.array([[gx, gy + 1.0]]))[0]
    return float((np.linalg.norm(px - p0) + np.linalg.norm(py - p0)) / 2.0)


def measure_one(
    img_gray: np.ndarray,
    H_gt: np.ndarray,
    N: int,
    version: int,
    align_conf_threshold: float,
) -> dict | None:
    det = detect(img_gray)
    if det is None:
        return None

    triplet = det.best_triplet
    right, down = _triplet_axes_from(det, triplet)
    grid = finder_grid_coords(N)

    # --- Baseline: 12 outer corners ---
    outer_corr: list[Correspondence] = []
    role_order = ["TL", "TR", "BL"]
    idx = {
        "TL": triplet.top_left_idx,
        "TR": triplet.top_right_idx,
        "BL": triplet.bottom_left_idx,
    }
    for role in role_order:
        ci = idx[role]
        corners_xy = _canonicalize_corners(det.global_outer_xy[ci], right, down)
        g = grid["outer"][role]
        for k in range(4):
            outer_corr.append(
                Correspondence(
                    grid_xy=g[k], image_xy=corners_xy[k],
                    confidence=float(det.score_map[ci]), label=f"outer-{role}-{k}",
                )
            )

    H_base, _err = fit_homography_from_correspondences(outer_corr, weighted=False)

    # --- Inner corners ---
    inner_corr: list[Correspondence] = []
    for role in role_order:
        ci = idx[role]
        info = det.cluster_info[ci]
        inner_local = fit_finder_inner_corners(info["roi"], info["center_xy"], info["m_est"])
        if inner_local is None:
            continue
        inner_global = inner_local + info["offset"]
        inner_ordered = _canonicalize_corners(inner_global, right, down)
        g = grid["inner"][role]
        for k in range(4):
            inner_corr.append(
                Correspondence(
                    grid_xy=g[k], image_xy=inner_ordered[k],
                    confidence=0.9, label=f"inner-{role}-{k}",
                )
            )

    # --- Alignment patterns (predicted by H_base) ---
    align_corr: list[Correspondence] = []
    for (gx, gy) in alignment_centers_grid(version):
        center0 = project_points(H_base, np.array([[gx, gy]]))[0]
        module_px = _local_module_px(H_base, gx, gy)
        if module_px < 0.5:
            continue
        image_xy, conf = match_alignment_pattern(det.img_gray, center0, module_px)
        align_corr.append(
            Correspondence(
                grid_xy=np.array([gx, gy]), image_xy=image_xy,
                confidence=conf, label=f"align-{gx:.1f}-{gy:.1f}",
            )
        )

    # --- Variants ---
    all_inner = outer_corr + inner_corr
    all_align = all_inner + align_corr
    align_filt = [c for c in align_corr if c.confidence >= align_conf_threshold]
    all_align_filt = all_inner + align_filt

    variants: dict[str, dict] = {}
    for name, corr, weighted, robust in [
        ("baseline", outer_corr, False, False),
        ("inner", all_inner, False, False),
        ("inner_align", all_align, False, False),
        ("inner_align_filt", all_align_filt, False, False),
        ("inner_align_wgt", all_align, True, False),
        ("inner_align_ransac", all_align, False, True),
    ]:
        if len(corr) < 4:
            variants[name] = {"n_corr": len(corr), "error": None}
            continue
        H, _err = fit_homography_from_correspondences(corr, weighted=weighted, robust=robust)
        stats = dense_error_stats(H, H_gt, N)
        stats["n_corr"] = len(corr)
        stats["n_align"] = sum(1 for c in corr if c.label.startswith("align"))
        stats["n_inner"] = sum(1 for c in corr if c.label.startswith("inner"))
        variants[name] = stats

    # Correspondence errors (per variant) vs GT.
    corr_err = {}
    for name, corr in [
        ("outer", outer_corr),
        ("inner", inner_corr),
        ("align", align_corr),
    ]:
        if not corr:
            corr_err[name] = None
            continue
        errs = [
            correspondence_error(c.image_xy, c.grid_xy, H_gt) for c in corr
        ]
        corr_err[name] = {
            "mean": float(np.mean(errs)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(np.max(errs)),
            "n": len(errs),
        }

    outer_err = corr_err["outer"]["mean"] if corr_err.get("outer") else float("inf")
    triplet_ok = outer_err < 5.0

    return {
        "version_estimated": det.estimated_version,
        "version_actual": version,
        "triplet_ok": triplet_ok,
        "variants": variants,
        "corr_err": corr_err,
    }


def _triplet_axes_from(det: Detection, triplet):
    # Reuse detector's helper by building the same quantities.
    tl = det.global_outer_xy[triplet.top_left_idx].mean(axis=0)
    tr = det.global_outer_xy[triplet.top_right_idx].mean(axis=0)
    bl = det.global_outer_xy[triplet.bottom_left_idx].mean(axis=0)
    right = tr - tl
    right = right / (float(np.linalg.norm(right)) + 1e-12)
    down = bl - tl
    down = down / (float(np.linalg.norm(down)) + 1e-12)
    return right, down


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(output_json: str, presets, versions, seeds, content, align_conf_threshold) -> None:
    box_size = 10
    border = 4
    rows_out: list[dict] = []

    for preset_name in presets:
        from qr_reader.scripts.qr_benchmark import PRESETS

        preset = PRESETS[preset_name]
        for version in versions:
            actual = _actual_version(content, version, box_size, border)
            for seed in seeds:
                img = generate_test_image(
                    seed=seed,
                    content=content,
                    version=version,
                    box_size=box_size,
                    border=border,
                    noise_std=preset.noise_std,
                    perspective_max_shift=preset.perspective_max_shift,
                    final_blur_kernel=preset.final_blur_kernel,
                    intensity_scale=preset.intensity_scale,
                    noise_blur_kernel=preset.noise_blur_kernel,
                )
                H_gt, N = compute_gt_homography(
                    version=actual,
                    content=content,
                    box_size=box_size,
                    border=border,
                    seed=seed,
                    perspective_max_shift=preset.perspective_max_shift,
                    rotation_angle_deg=None,
                )
                t0 = perf_counter()
                result = measure_one(img, H_gt, N, actual, align_conf_threshold)
                elapsed = perf_counter() - t0

                rec = {
                    "preset": preset_name,
                    "version_req": version,
                    "version_actual": actual,
                    "seed": seed,
                    "time_s": elapsed,
                }
                if result is None:
                    rec["error"] = "detection failed"
                else:
                    rec.update(result)
                rows_out.append(rec)

                det_str = "✗" if result is None else "✓"
                print(f"[{preset_name:>6s} V{version:2d}({actual}) s{seed:02d}] det={det_str}  {elapsed:.2f}s")

    with open(output_json, "w") as f:
        json.dump(rows_out, f, indent=2)
    print(f"\nWrote {output_json}")
    print_summary(rows_out)


def print_summary(rows_out: list[dict]) -> None:
    from collections import defaultdict

    variant_names = [
        "baseline", "inner", "inner_align",
        "inner_align_filt", "inner_align_wgt", "inner_align_ransac",
    ]
    ok_rows = [r for r in rows_out if r.get("triplet_ok")]
    n_total = sum(1 for r in rows_out if "variants" in r)
    n_ok = len(ok_rows)

    print(f"\nCorrect-triplet images: {n_ok}/{n_total} "
          f"({100*n_ok/max(n_total,1):.0f}%)")

    agg = defaultdict(list)
    for rec in ok_rows:
        for name in variant_names:
            v = rec["variants"].get(name)
            if v and v.get("error") is None and v.get("dense_mean") is not None:
                agg[name].append(v)

    print("\n## Homography quality on correct-triplet images (dense reprojection error, px)\n")
    print("| Variant | n_corr | n_inner | n_align | dense_mean | dense_p90 | dense_max |")
    print("|---------|--------|---------|---------|------------|-----------|-----------|")
    for name in variant_names:
        vals = agg[name]
        if not vals:
            continue
        mean = np.mean([v["dense_mean"] for v in vals])
        p90 = np.mean([v["dense_p90"] for v in vals])
        mx = np.mean([v["dense_max"] for v in vals])
        nc = np.mean([v["n_corr"] for v in vals])
        ni = np.mean([v["n_inner"] for v in vals])
        na = np.mean([v["n_align"] for v in vals])
        print(f"| {name:>17s} | {nc:6.1f} | {ni:7.1f} | {na:7.1f} | {mean:10.2f} | {p90:9.2f} | {mx:9.2f} |")

    print("\n## Landmark correspondence error vs GT (px, correct-triplet images)\n")
    print("| Type | n | mean | p90 | max |")
    print("|------|---|------|-----|-----|")
    for key in ["outer", "inner", "align"]:
        errs = []
        counts = []
        for rec in ok_rows:
            ce = rec.get("corr_err", {}).get(key)
            if ce:
                errs.append(ce["mean"])
                counts.append(ce["n"])
        if errs:
            print(f"| {key:>5s} | {np.mean(counts):.1f} | {np.mean(errs):.2f} | "
                  f"{np.mean([rec['corr_err'][key]['p90'] for rec in ok_rows if rec.get('corr_err', {}).get(key)]):.2f} | "
                  f"{np.mean([rec['corr_err'][key]['max'] for rec in ok_rows if rec.get('corr_err', {}).get(key)]):.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="easy,medium,hard")
    parser.add_argument("--versions", default="1,3,5,8,12,20")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--content", default="https://www.rikvoorhaar.com")
    parser.add_argument("--align-conf", type=float, default=0.6)
    parser.add_argument("--output", default="homography_quality.json")
    args = parser.parse_args()

    presets = [p.strip() for p in args.presets.split(",")]
    versions = [int(v) for v in args.versions.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    run(args.output, presets, versions, seeds, args.content, args.align_conf)
