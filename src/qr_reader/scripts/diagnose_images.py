"""Batch visual diagnostic — saves annotated images for manual inspection.

Usage:
    python diagnose_images.py [--all] [--preset medium] [--version 1] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.detector import _run_detection
from qr_reader.detector.finder_pattern import FinderPattern, find_valid_triplets
from qr_reader.detector.homography import compute_qr_corners
from qr_reader.detector.ray_fit import fit_finder_ray
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

OUT_DIR = Path("diagnose_images")
BG_DIR = Path("data/images/train")


def diagnose_one(preset: str, version: int, seed: int, out_dir: Path) -> dict:
    bg_paths = sorted(BG_DIR.glob("*.jpg"))
    base_cfg = PRESET_MAP[preset]
    config = AugmentationConfig(**base_cfg.__dict__)
    config.version = version
    config.content = f"v{version}"
    config.error_correction = "M"
    config.global_seed = seed

    rng = np.random.default_rng(seed)
    bg_path = bg_paths[seed % len(bg_paths)]
    bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)
    img_rgb, meta = generate_sample(
        rng, config, bg, sample_index=0, background_path=str(bg_path),
    )
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    gt_corners = np.array([
        meta["corners_qr"]["TL"], meta["corners_qr"]["TR"],
        meta["corners_qr"]["BR"], meta["corners_qr"]["BL"],
    ], dtype=np.float64)

    # Pipeline stages
    img_binary = binarize_image(gray)
    rows_v, cols_v_all = find_alignment_patterns_2d(img_binary, np.log(1.3))
    clusters = cluster_candidates(rows_v, cols_v_all)

    # Per-cluster fitting
    fps = []; score_map = {}; global_corners_xy = {}
    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0: continue
        r0 = max(0, int(bbox[0])); c0 = max(0, int(bbox[2]))
        cx = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
        cy = float(cluster.row) - r0
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0
        result = fit_finder_ray(roi, np.array([cx, cy]), m_est)
        if not result.valid: continue
        cxy = result.corners + np.array([c0, r0], dtype=np.float64)
        wh = np.ptp(cxy, axis=0)
        if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est: continue
        fps.append(FinderPattern(cluster_idx=ci, outer_corners=cxy[:, ::-1]))
        score_map[ci] = result.score
        global_corners_xy[ci] = cxy

    # Dedup
    keep_mask = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep_mask[i]: continue
        ci = fps[i].outer_corners.mean(axis=0)
        seg_i = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
        for j in range(i + 1, len(fps)):
            if not keep_mask[j]: continue
            cj = fps[j].outer_corners.mean(axis=0)
            seg_j = float(np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1]))
            if float(np.linalg.norm(ci - cj)) < 1.0 * min(seg_i, seg_j):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False; break
    fps = [fp for fp, keep in zip(fps, keep_mask) if keep]

    # Triplets
    triplets = find_valid_triplets(fps, score_map)

    # Full detection
    try:
        H, dv = _run_detection(gray)
        det_ok = True
    except Exception:
        dv = version
        det_ok = False

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_rgb)

    gt_poly = np.vstack([gt_corners, gt_corners[:1]])
    ax.plot(gt_poly[:, 0], gt_poly[:, 1], "-", color="#d62728", linewidth=2, label="GT")

    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        r0, r1, c0, c1 = bbox
        rect = patches.Rectangle((c0, r0), c1 - c0, r1 - r0,
                                  fill=False, edgecolor="cyan", linewidth=0.5, alpha=0.3)
        ax.add_patch(rect)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fps), 1)))
    for i, fp in enumerate(fps):
        cxy = fp.outer_corners[:, ::-1]
        poly = np.vstack([cxy, cxy[:1]])
        ax.plot(poly[:, 0], poly[:, 1], "-", color=colors[i], linewidth=2,
                label=f"FP{i} s={score_map[fp.cluster_idx]:.2f}")
        ax.plot(cxy[0, 0], cxy[0, 1], "o", color=colors[i], markersize=6)

    if len(rows_v) > 0:
        for row, cvs in zip(rows_v, cols_v_all):
            ax.plot(cvs, [row] * len(cvs), ".", color="yellow", markersize=2, alpha=0.4)

    status = "OK" if det_ok else "FAIL"
    ax.set_title(f"{preset} V={version} s={seed} | {len(fps)} finders → {len(triplets)} triplets | DET={status}",
                 fontsize=11)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.axis("off")
    fig.tight_layout()

    fname = out_dir / f"{preset}_v{version:02d}_s{seed:02d}.png"
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "preset": preset, "version": version, "seed": seed,
        "n_finders": len(fps), "n_triplets": len(triplets),
        "detection_ok": det_ok, "detected_version": dv,
        "image": str(fname),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Generate all interesting cases from diagnose_composited.json")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        with open("diagnose_composited.json") as f:
            data = json.load(f)
        ok_list = [r for r in data["results"] if r["detection_ok"]][:2]
        fail_many = [r for r in data["results"]
                     if not r["detection_ok"] and r["n_finders"] >= 3][:3]
        fail_few = [r for r in data["results"]
                    if not r["detection_ok"] and r["n_finders"] <= 1 and len(r.get("finder_valid", [])) == 0][:2]

        cases = [{"preset": r["preset"], "version": r["version"], "seed": r["seed"]}
                 for r in ok_list + fail_many + fail_few]
    else:
        cases = [{"preset": args.preset, "version": args.version, "seed": args.seed}]

    results = []
    for c in cases:
        try:
            r = diagnose_one(c["preset"], c["version"], c["seed"], out_dir)
            results.append(r)
            print(f"  {r['preset']} V={r['version']:2d} s={r['seed']:02d} "
                  f"finders={r['n_finders']} triplets={r['n_triplets']} "
                  f"det={'OK' if r['detection_ok'] else 'FAIL'} → {r['image']}")
        except Exception as e:
            print(f"  ERROR ({c}): {e}")

    print(f"\nSaved {len(results)} images to {out_dir}/")


if __name__ == "__main__":
    main()
