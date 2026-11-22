#!/usr/bin/env python3
"""Benchmark the detector/decoder against the captured QR benchmark dataset.

The dataset is 100 real-world photos of a QR code encoding
``rikvoorhaar.com`` (see the README "Datasets" section for download
instructions).  It has no per-image ground truth, so the benchmark reports a
six-stage detection funnel — the share of images reaching each pipeline stage —
plus decode accuracy and wall time:

    1. >= 3 candidate clusters from the alignment scan
    2. >= 3 refined finder patterns (after ray-fit + dedup)
    3. >= 1 valid triplet
    4. homography + version estimated
    5. decodable (sampling + Reed-Solomon produced any text)
    6. decodes to the expected payload

Usage:
    python src/qr_reader/scripts/benchmark_captured.py [--data-dir DIR] [--output JSON]

Results are written to benchmark_captured.json and a summary is printed.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qr_reader.pipeline import decode_image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "bench" / "captured"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark_captured.json"
PAYLOAD = "rikvoorhaar.com"


@dataclass
class ImageResult:
    image: str = ""
    n_candidates: int = 0
    n_finder_patterns: int = 0
    n_triplets: int = 0
    detection_ok: bool = False
    detected_version: int = -1
    detection_error: str = ""
    detection_time_s: float = 0.0
    sample_time_s: float = 0.0
    decode_time_s: float = 0.0
    attempts: int = 0
    decode_ok: bool = False
    payload_ok: bool = False
    text: str = ""
    decode_error_text: str = ""


def _natural_sort_key(path: Path) -> tuple[int, ...]:
    nums = re.findall(r"\d+", path.stem)
    return tuple(int(n) for n in nums)


def _image_paths(data_dir: Path) -> list[Path]:
    return sorted(list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.jpg")), key=_natural_sort_key)


def evaluate_image(img_path: Path) -> ImageResult:
    result = ImageResult(image=img_path.name)

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        result.detection_error = "cv2.imread returned None"
        return result
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detection, sampling, and decoding (multi-triplet / multi-version)
    dec = decode_image(gray, expected_payload=PAYLOAD)
    result.n_candidates = dec.n_candidates
    result.n_finder_patterns = dec.n_finder_patterns
    result.n_triplets = dec.n_triplets
    result.attempts = len(dec.attempts)
    result.detection_time_s = dec.prep_time_s + sum(
        a.fit_time_s for a in dec.attempts
    )
    result.sample_time_s = sum(a.sample_time_s for a in dec.attempts)
    result.decode_time_s = sum(a.decode_time_s for a in dec.attempts)
    result.detection_ok = dec.detection_ok
    result.decode_ok = any(a.decoded_text is not None for a in dec.attempts)
    result.payload_ok = dec.success
    if dec.version is not None:
        result.detected_version = dec.version
    elif dec.detected_version is not None:
        result.detected_version = dec.detected_version
    if dec.text is not None:
        result.text = dec.text
    else:
        decoded = [a.decoded_text for a in dec.attempts if a.decoded_text is not None]
        if decoded:
            result.text = decoded[-1]

    if not dec.detection_ok:
        if dec.error is not None:
            result.detection_error = f"{type(dec.error).__name__}: {dec.error}"
        elif dec.time_budget_exhausted:
            result.detection_error = "time budget exhausted before any candidate"
        else:
            result.detection_error = "no homography candidate produced"
    elif not result.payload_ok:
        result.decode_error_text = "; ".join(
            a.failure_reason for a in dec.attempts if not a.success
        )

    return result


def run_benchmark(data_dir: Path, output_json: Path) -> None:
    paths = _image_paths(data_dir)
    if not paths:
        print(
            f"ERROR: no JPEGs found in {data_dir}.\n"
            "Download the dataset first — see README.md 'Datasets' section "
            "(Captured QR Benchmark):\n"
            "  mkdir -p data/bench\n"
            "  curl -L -o /tmp/captured-qr-benchmark-v1.tar.gz "
            "https://github.com/RikVoorhaar/qr-reader/releases/download/"
            "benchmark-v1/captured-qr-benchmark-v1.tar.gz\n"
            "  tar -xzf /tmp/captured-qr-benchmark-v1.tar.gz -C data/bench/"
        )
        sys.exit(1)

    print(f"Benchmarking {len(paths)} images from {data_dir} ...\n")

    results: list[ImageResult] = []
    for i, img_path in enumerate(paths):
        results.append(evaluate_image(img_path))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(paths)}...")

    output: dict[str, Any] = {
        "config": {
            "data_dir": str(data_dir),
            "payload": PAYLOAD,
            "num_images": len(paths),
        },
        "results": [asdict(r) for r in results],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {output_json}")
    print_summary(results)


def print_summary(results: list[ImageResult]) -> None:
    n = len(results)

    # 6-stage funnel
    stage1 = sum(1 for r in results if r.n_candidates >= 3)
    stage2 = sum(1 for r in results if r.n_finder_patterns >= 3)
    stage3 = sum(1 for r in results if r.n_triplets >= 1)
    stage4 = sum(1 for r in results if r.detection_ok)
    stage5 = sum(1 for r in results if r.decode_ok)
    stage6 = sum(1 for r in results if r.payload_ok)
    funnel = [(">=3 candidates", stage1), (">=3 finder patterns", stage2),
              (">=1 triplet", stage3), ("version estimated", stage4),
              ("decodable", stage5), ("correct payload", stage6)]

    print("## Funnel (share of all images reaching each stage)\n")
    print(f"{'stage':<22} {'count':>6} {'%':>7}")
    print(f"{'-' * 22} {'------':>6} {'-------':>7}")
    for name, count in funnel:
        print(f"{name:<22} {count:>6} {100 * count / n:>6.1f}%")

    # Stage-to-stage retention
    print("\n## Stage-to-stage retention\n")
    print(f"{'':<22} {'kept':>7} {'%':>7}")
    print(f"{'-' * 22} {'-------':>7} {'-------':>7}")
    for (name, count), (_, prev) in zip(funnel, [(name, n)] + funnel[:-1]):
        keep = 100 * count / prev if prev else 0
        print(f"{name:<22} {count:>7} {keep:>6.1f}%")

    # Wall time (per-phase stats over detected images only, so zeros from
    # images that never reached the phase don't skew the mean/median)
    detected = [r for r in results if r.detection_ok]
    totals = [r.detection_time_s + r.sample_time_s + r.decode_time_s for r in results]
    detects = [r.detection_time_s for r in detected]
    samples = [r.sample_time_s for r in detected]
    decodes = [r.decode_time_s for r in detected]
    print("\n## Wall time\n")
    print(f"  total:        {sum(totals):8.2f}s  mean {1000 * np.mean(totals):7.0f}ms  "
          f"median {1000 * np.median(totals):7.0f}ms")
    if detected:
        print(f"  detect:       {sum(detects):8.2f}s  mean {1000 * np.mean(detects):7.0f}ms  "
              f"median {1000 * np.median(detects):7.0f}ms")
        print(f"  sample:       {sum(samples):8.2f}s  mean {1000 * np.mean(samples):7.0f}ms  "
              f"median {1000 * np.median(samples):7.0f}ms")
        print(f"  decode:       {sum(decodes):8.2f}s  mean {1000 * np.mean(decodes):7.0f}ms  "
              f"median {1000 * np.median(decodes):7.0f}ms")

    attempted = [r for r in results if r.attempts > 0]
    if attempted:
        attempts = [r.attempts for r in attempted]
        print("\n## Decode attempts\n")
        print(f"  images with attempts: {len(attempted)}/{n}")
        print(f"  attempts:             {sum(attempts)} total  "
              f"mean {np.mean(attempts):.2f}  median {np.median(attempts):.0f}  "
              f"max {max(attempts)}")
        for count, num in sorted(Counter(attempts).items()):
            print(f"    {num:3d} images with {count} attempt(s)")

    # Detected version distribution (diagnostic; no ground truth)
    versions = Counter(r.detected_version for r in results if r.detection_ok)
    print("\n## Detected version distribution (detected only)\n")
    for v, count in sorted(versions.items()):
        print(f"  v{v}: {count}")

    # Failure breakdown
    det_fail = Counter(r.detection_error for r in results if not r.detection_ok)
    if det_fail:
        print("\n## Detection failures\n")
        for msg, count in det_fail.most_common():
            print(f"  [{count:3d}] {msg}")
    dec_fail = Counter(r.decode_error_text for r in results if r.detection_ok and not r.decode_ok)
    if dec_fail:
        print("\n## Decode failures (detected but not decoded)\n")
        for msg, count in dec_fail.most_common(8):
            print(f"  [{count:3d}] {msg}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark the QR detector/decoder on the captured dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with the captured benchmark JPEGs "
        f"(default: {DEFAULT_DATA_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir
    run_benchmark(data_dir, args.output)
