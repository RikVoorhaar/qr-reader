"""Benchmark pipeline against pre-generated composited benchmark images.

Usage:
    python benchmark_composited.py
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

from qr_reader.decoder.decoder import decode as decode_qr
from qr_reader.detector.detector import detect_corners, detect_sample

METADATA = Path("data/bench/synth/metadata.jsonl")


@dataclass
class BenchResult:
    image_path: str = ""
    preset: str = ""
    version: int = -1
    seed: int = -1
    gt_qr_size_px: float = 0.0

    detection_time_ms: float = 0.0
    detection_ok: bool = False
    detected_version: int = -1
    corner_error_px: float = -1.0
    gt_version_match: bool = False
    decode_ok: bool = False
    error: str = ""


def _corner_error(detected: np.ndarray, gt: np.ndarray) -> float:
    from itertools import permutations

    best = np.inf
    for perm in permutations(range(4)):
        d = sum(float(np.linalg.norm(detected[p] - gt[q]))
                for p, q in enumerate(perm))
        if d < best:
            best = d
    return best / 4.0


def run_benchmark() -> list[BenchResult]:
    if not METADATA.exists():
        print(f"ERROR: {METADATA} not found. Run generate_bench.py first.")
        sys.exit(1)

    lines = [json.loads(l) for l in open(METADATA)]
    results: list[BenchResult] = []

    for i, meta in enumerate(lines):
        br = BenchResult(
            image_path=meta["image_path"],
            preset=meta["preset"],
            version=meta["version"],
            seed=meta["seed"],
        )
        img_path = Path(meta["image_path"])
        if not img_path.exists():
            br.error = "image not found"
            results.append(br)
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            br.error = "cv2.imread returned None"
            results.append(br)
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        gt_corners = np.array([
            meta["corners_qr"]["TL"],
            meta["corners_qr"]["TR"],
            meta["corners_qr"]["BR"],
            meta["corners_qr"]["BL"],
        ], dtype=np.float64)
        br.gt_qr_size_px = float(np.mean([
            np.linalg.norm(gt_corners[0] - gt_corners[1]),
            np.linalg.norm(gt_corners[3] - gt_corners[0]),
        ]))

        try:
            t0 = perf_counter()
            corners, dv = detect_corners(gray)
            dt = (perf_counter() - t0) * 1000
            br.detection_time_ms = dt
            br.detection_ok = True
            br.detected_version = dv
            br.gt_version_match = (dv == meta["version"])
            br.corner_error_px = _corner_error(corners, gt_corners)

            try:
                bits = detect_sample(gray)
                decoded = decode_qr(bits)
                br.decode_ok = (decoded == meta["payload"])
            except Exception:
                br.decode_ok = False

        except Exception as e:
            br.error = str(e)[:100]

        results.append(br)
        pct = 100 * (i + 1) / len(lines)
        status = "✓" if br.detection_ok else "✗"
        ver = f"v={br.detected_version}" if br.detection_ok else ""
        print(f"[{i+1:3d}/{len(lines)}] {br.preset:>6s} V={br.version:2d} s={br.seed:03d}  "
              f"det={status} {ver} err={br.corner_error_px:5.1f}px dec={br.decode_ok}  "
              f"{br.detection_time_ms:6.0f}ms  {br.error}")

    return results


def print_summary(results: list[BenchResult]):
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        groups[(r.preset, r.version)].append(r)

    print("\n## Benchmark Summary (composited images)\n")
    print("| Preset | Version | Detect % | Corner Err (px) | Ver Match % | Decode % | Time (ms) |")
    print("|--------|---------|----------|-----------------|-------------|----------|-----------|")

    for preset_name in ["easy", "medium", "hard"]:
        for version in sorted(set(r.version for r in results)):
            grp = groups[(preset_name, version)]
            n = len(grp)
            det_ok = sum(1 for r in grp if r.detection_ok)
            dec_ok = sum(1 for r in grp if r.decode_ok)
            ver_match = sum(1 for r in grp if r.gt_version_match)
            corner_errs = [r.corner_error_px for r in grp if r.detection_ok]
            times = [r.detection_time_ms for r in grp if r.detection_ok]

            mean_err = np.mean(corner_errs) if corner_errs else float("nan")
            mean_time = np.mean(times) if times else float("nan")

            print(
                f"| {preset_name:>6s} | {version:7d} | "
                f"{det_ok/n*100:5.1f}% ({det_ok}/{n}) | "
                f"{mean_err:15.1f} | "
                f"{ver_match/n*100:5.1f}% | "
                f"{dec_ok/n*100:5.1f}% ({dec_ok}/{n}) | "
                f"{mean_time:9.0f} |"
            )

    print("|--------|---------|----------|-----------------|-------------|----------|-----------|")
    for preset_name in ["easy", "medium", "hard"]:
        preset_results = [r for r in results if r.preset == preset_name]
        avg_det = np.mean([r.detection_ok for r in preset_results]) * 100
        avg_dec = np.mean([r.decode_ok for r in preset_results]) * 100
        avg_ver = np.mean([r.gt_version_match for r in preset_results]) * 100
        print(
            f"| **{preset_name:>6s}** | **avg** | **{avg_det:5.1f}%** "
            f"| | **{avg_ver:5.1f}%** | **{avg_dec:5.1f}%** | |"
        )

    print()


if __name__ == "__main__":
    results = run_benchmark()
    print_summary(results)
