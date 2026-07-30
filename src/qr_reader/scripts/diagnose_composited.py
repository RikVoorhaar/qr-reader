"""Diagnose pipeline failures on composited QR images (synth pipeline).

Sweeps presets × versions × seeds on real backgrounds, reports failure
stages (alignment scan, finder fitting, triplets) and coordinates of good
and bad examples for visual inspection.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.detector import _run_detection
from qr_reader.detector.ray_fit import fit_finder_ray, normalize_roi_intensities
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.decoder.decoder import decode as decode_qr
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP

# ── Config ────────────────────────────────────────────────────────────────────
BG_DIR = Path("data/images/train")
VERSIONS = [1, 3, 5, 8]
NUM_SEEDS_PER = 20  # seeds per preset×version combination

# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class DiagResult:
    preset: str
    version: int
    seed: int
    bg_path: str = ""

    # pipeline stages
    n_candidates: int = 0  # RLE candidates from alignment scan
    n_clusters: int = 0  # after clustering
    n_finders: int = 0  # finders fitted by ray_fit
    n_triplets_after_dedup: int = 0  # finders after dedup
    n_triplets: int = 0  # valid triplets found

    # finder details (for up to 3 clusters)
    finder_valid: list[bool] = field(default_factory=list)
    finder_score: list[float] = field(default_factory=list)
    finder_concentration: list[float] = field(default_factory=list)
    finder_center_xy: list[list[float]] = field(default_factory=list)

    # end-to-end
    detection_ok: bool = False
    detected_version: int = -1
    corner_error_px: float = -1.0
    decode_ok: bool = False
    time_ms: float = 0.0
    error: str = ""


def run_diagnostic() -> list[DiagResult]:
    bg_paths = sorted(BG_DIR.glob("*.jpg"))
    if not bg_paths:
        print("ERROR: No background images found in", BG_DIR)
        sys.exit(1)

    results: list[DiagResult] = []
    presets = ["medium"]
    total = len(presets) * len(VERSIONS) * NUM_SEEDS_PER

    n = 0
    for preset_name in presets:
        base_cfg = PRESET_MAP[preset_name]
        for version in VERSIONS:
            for si in range(NUM_SEEDS_PER):
                n += 1
                seed = si
                dr = DiagResult(preset=preset_name, version=version, seed=seed)

                config = AugmentationConfig(**base_cfg.__dict__)
                config.version = version
                config.content = f"v{version}"
                config.error_correction = "M"
                config.global_seed = seed

                rng = np.random.default_rng(seed)
                bg_path = bg_paths[seed % len(bg_paths)]
                dr.bg_path = bg_path.name
                bg = cv2.cvtColor(cv2.imread(str(bg_path)), cv2.COLOR_BGR2RGB)

                try:
                    img_rgb, meta = generate_sample(
                        rng, config, bg, sample_index=0,
                        background_path=str(bg_path),
                    )
                    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

                    gt_corners = np.array([
                        meta["corners_qr"]["TL"],
                        meta["corners_qr"]["TR"],
                        meta["corners_qr"]["BR"],
                        meta["corners_qr"]["BL"],
                    ], dtype=np.float64)

                    # ── 1. Alignment scan ──
                    img_binary = binarize_image(gray)
                    rows_v, cols_v_all = find_alignment_patterns_2d(
                        img_binary, np.log(1.3),
                    )
                    dr.n_candidates = len(rows_v)
                    if len(rows_v) == 0:
                        dr.error = "No alignment patterns found"
                        results.append(dr)
                        continue

                    # ── 2. Clustering ──
                    clusters = cluster_candidates(rows_v, cols_v_all)
                    dr.n_clusters = len(clusters)

                    # ── 3. Per-cluster finder fitting ──
                    finder_details = []
                    for ci, cluster in enumerate(clusters[:5]):
                        bbox = cluster_to_bbox(cluster, scale=1.5)
                        roi = cutout(gray, bbox)
                        if roi.size == 0:
                            finder_details.append(
                                {"valid": False, "score": 0.0,
                                 "concentration": 0.0, "center": [0, 0]})
                            continue
                        r0 = max(0, int(bbox[0]))
                        c0 = max(0, int(bbox[2]))
                        cx = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
                        cy = float(cluster.row) - r0
                        center_xy = np.array([cx, cy], dtype=np.float64)
                        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

                        result = fit_finder_ray(roi, center_xy, m_est)

                        # Compute concentration after the fact
                        _, _, _ = normalize_roi_intensities(roi, center_xy, m_est)
                        # rough concentration from the fit itself
                        conc = 1.0 if result.valid else 0.0

                        finder_details.append({
                            "valid": result.valid,
                            "score": float(result.score),
                            "concentration": conc,
                            "center": [float(center_xy[0] + c0), float(center_xy[1] + r0)],
                        })
                    dr.finder_valid = [d["valid"] for d in finder_details]
                    dr.finder_score = [d["score"] for d in finder_details]
                    dr.finder_concentration = [d["concentration"] for d in finder_details]
                    dr.finder_center_xy = [d["center"] for d in finder_details]
                    dr.n_finders = sum(dr.finder_valid)

                    if dr.n_finders == 0:
                        dr.error = "No finder patterns fitted"
                        results.append(dr)
                        continue

                    # ── 4. Full detection ──
                    t0 = perf_counter()
                    H, dv = _run_detection(gray)
                    dt_ms = (perf_counter() - t0) * 1000
                    dr.time_ms = dt_ms
                    dr.detection_ok = True
                    dr.detected_version = dv

                    # corner error
                    corners = np.array([
                        [0, 0],
                        [(4 * dv + 17), 0],
                        [(4 * dv + 17), (4 * dv + 17)],
                        [0, (4 * dv + 17)],
                    ], dtype=np.float64)
                    corners_h = np.column_stack([corners, np.ones(4)])
                    proj = (H @ corners_h.T).T
                    proj = proj[:, :2] / proj[:, 2:3]
                    # Best match to GT
                    err = np.mean([np.min([np.linalg.norm(proj[i] - gt_corners[j])
                                          for j in range(4)]) for i in range(4)])
                    dr.corner_error_px = float(err)

                    # decode
                    try:
                        from qr_reader.detector.detector import detect_sample
                        bits = detect_sample(gray)
                        decoded = decode_qr(bits)
                        dr.decode_ok = (decoded == f"v{version}")
                    except Exception:
                        dr.decode_ok = False

                except Exception as e:
                    dr.error = str(e)[:80]

                results.append(dr)

    return results


if __name__ == "__main__":
    import os
    os.environ.pop("DISPLAY", None)  # prevent matplotlib from trying to show

    results = run_diagnostic()

    # Write JSON
    out = {
        "config": {
            "presets": ["medium"],
            "versions": VERSIONS,
            "seeds_per": NUM_SEEDS_PER,
            "bg_dir": str(BG_DIR),
        },
        "results": [asdict(r) for r in results],
    }
    with open("diagnose_composited.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print summary
    print(f"\nScanned {len(results)} images (medium preset)")
    ok = [r for r in results if r.detection_ok]
    fail = [r for r in results if not r.detection_ok]
    print(f"  OK: {len(ok)}  FAIL: {len(fail)}")
    if fail:
        by_stage = {}
        for r in fail:
            stage = r.error or "unknown"
            by_stage[stage] = by_stage.get(stage, 0) + 1
        print("  Failure stages:")
        for stage, cnt in sorted(by_stage.items(), key=lambda x: -x[1]):
            print(f"    {stage}: {cnt}")

    if ok:
        print(f"\n  Success corner errors: mean={np.mean([r.corner_error_px for r in ok]):.1f}px")
        print(f"  Decode success: {sum(r.decode_ok for r in ok)}/{len(ok)}")
        print(f"  Mean time: {np.mean([r.time_ms for r in ok]):.0f}ms")

    print(f"\nDetailed results written to diagnose_composited.json")
