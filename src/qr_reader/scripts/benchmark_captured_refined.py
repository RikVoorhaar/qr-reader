"""Captured-benchmark A/B + ablation: homography variants.

For each image in the captured QR benchmark this runs the same detection and
ranked (triplet, version) candidate loop as ``decode_image``, but for every
candidate it fits several homographies and decodes with each:

* **baseline** — current 12 outer-corner DLT + LM.
* **inner**     — outer + inner finder corners (no alignment patterns).
* **align**     — outer + interior alignment patterns (no inner corners).
* **both**      — outer + inner + alignment.

Inner corners now come straight from the finder fit itself
(``FinderPattern.inner_corners``, emitted inside ``fit_finder_ray`` reusing its
per-ray module-pitch estimates), so the only genuinely new per-candidate work in
``inner``/``both`` is the alignment-pattern template matching.

Usage:
    PYTHONPATH=src python -m qr_reader.scripts.benchmark_captured_refined \
        --data-dir /path/to/data/bench/captured [--output refined_captured.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from qr_reader.decoder.decoder import decode as decode_bits
from qr_reader.detector.detector import (
    DetectionResult,
    _fit_candidate_homography,
    _prepare_detection,
    _rank_triplets,
)
from qr_reader.detector.landmarks_refined import build_refined_homography
from qr_reader.detector.sample import sample_qr_bits

PAYLOAD = "rikvoorhaar.com"
MAX_TRIPLETS = 5
MAX_VERSIONS = 2


@dataclass
class Variant:
    success: bool = False
    text: str = ""
    attempts: int = 0
    homo_s: float = 0.0


@dataclass
class ImageResult:
    image: str = ""
    detection_ok: bool = False
    detection_error: str = ""
    n_candidates: int = 0
    variants: dict[str, Variant] = field(default_factory=dict)


def _natural_sort_key(path: Path) -> tuple[int, ...]:
    nums = re.findall(r"\d+", path.stem)
    return tuple(int(n) for n in nums)


def _image_paths(data_dir: Path) -> list[Path]:
    return sorted(
        list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.jpg")),
        key=_natural_sort_key,
    )


def evaluate_image(img_path: Path) -> ImageResult:
    result = ImageResult(image=img_path.name)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        result.detection_error = "cv2.imread returned None"
        return result
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detection = DetectionResult(success=False)
    try:
        context = _prepare_detection(gray, detection)
        ranked = _rank_triplets(context)
    except Exception as exc:
        result.detection_error = f"{type(exc).__name__}: {exc}"
        return result

    result.detection_ok = True
    result.n_candidates = len(ranked)
    result.variants = {
        name: Variant() for name in ("baseline", "inner", "align", "both")
    }

    for candidate in ranked[:MAX_TRIPLETS]:
        for version in candidate.version_candidates[:MAX_VERSIONS]:
            N = 4 * version + 17

            # baseline
            t0 = perf_counter()
            H_base, _err = _fit_candidate_homography(context, candidate.triplet, N)
            result.variants["baseline"].homo_s += perf_counter() - t0

            # refined variants (inner / align / both)
            for name, kwargs in (
                ("inner", dict(include_inner=True, include_align=False)),
                ("align", dict(include_inner=False, include_align=True)),
                ("both", dict(include_inner=True, include_align=True)),
            ):
                t0 = perf_counter()
                H, _stats = build_refined_homography(
                    context.img_gray, H_base, candidate.triplet,
                    context.fps, context.score_map, version,
                    robust=False, **kwargs,
                )
                result.variants[name].homo_s += perf_counter() - t0
                self_var = result.variants[name]

                if not self_var.success:
                    self_var.attempts += 1
                    try:
                        bits = sample_qr_bits(context.img_gray, H, N)
                        text = decode_bits(bits)
                        if text == PAYLOAD:
                            self_var.success = True
                            self_var.text = text
                    except Exception:
                        pass

            base_var = result.variants["baseline"]
            if not base_var.success:
                base_var.attempts += 1
                try:
                    bits = sample_qr_bits(context.img_gray, H_base, N)
                    text = decode_bits(bits)
                    if text == PAYLOAD:
                        base_var.success = True
                        base_var.text = text
                except Exception:
                    pass

    return result


def run(data_dir: Path, output_json: Path) -> None:
    paths = _image_paths(data_dir)
    if not paths:
        print(f"ERROR: no JPEGs in {data_dir}")
        sys.exit(1)

    print(f"Benchmarking {len(paths)} images ...\n")
    results = [evaluate_image(p) for p in paths]

    with open(output_json, "w") as f:
        json.dump(
            {"payload": PAYLOAD, "results": [asdict(r) for r in results]},
            f, indent=2,
        )
    print(f"\nWrote {output_json}")
    print_summary(results)


def print_summary(results: list[ImageResult]) -> None:
    n = len(results)
    det_ok = sum(1 for r in results if r.detection_ok)

    print("\n## Decode accuracy\n")
    print(f"  images:               {n}")
    print(f"  detection reached:    {det_ok} ({100*det_ok/n:.1f}%)")
    for name in ("baseline", "inner", "align", "both"):
        ok = sum(1 for r in results if r.variants.get(name) and r.variants[name].success)
        print(f"  {name:>10s} decode:    {ok:3d} ({100*ok/n:.1f}%)")

    # Which images differ from baseline?
    base = {r.image for r in results if r.variants.get("baseline") and r.variants["baseline"].success}
    print("\n## vs baseline\n")
    for name in ("inner", "align", "both"):
        s = {r.image for r in results if r.variants.get(name) and r.variants[name].success}
        gained = sorted(s - base)
        lost = sorted(base - s)
        print(f"  {name:>10s}: +{len(gained)} gained, -{len(lost)} lost", end="")
        if gained:
            print(f"   gained: {gained}", end="")
        if lost:
            print(f"   lost: {lost}", end="")
        print()

    # Wall time (homography fit only; detection happens once and is shared).
    print("\n## Homography-fit wall time (whole benchmark)\n")
    for name in ("baseline", "inner", "align", "both"):
        t = sum(r.variants[name].homo_s for r in results if r.variants.get(name))
        print(f"  {name:>10s}: {t:7.2f}s   mean {1000*t/n:6.1f}ms/img")
    extra_inner = sum(
        r.variants["inner"].homo_s - r.variants["baseline"].homo_s
        for r in results if r.variants.get("inner")
    )
    extra_both = sum(
        r.variants["both"].homo_s - r.variants["baseline"].homo_s
        for r in results if r.variants.get("both")
    )
    print(f"\n  extra for 'inner' vs baseline: {extra_inner:6.2f}s  ({1000*extra_inner/n:.1f}ms/img)")
    print(f"  extra for 'both'  vs baseline: {extra_both:6.2f}s  ({1000*extra_both/n:.1f}ms/img)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("refined_captured.json"))
    args = parser.parse_args()
    run(args.data_dir, args.output)
