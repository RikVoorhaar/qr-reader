"""I4 — Validate D is non-blocking via full detect→decode pipeline.

Runs the full pipeline (detect_sample → decode) on v12-default and
v12-clean images.  The question: despite D failures (missing finder
edges at the Hough level), does the full pipeline still produce a
correct decode?

If yes, D is non-blocking and D-fix phases are low priority.
"""
from __future__ import annotations

import sys

import numpy as np

from qr_reader.decoder.decoder import DecodeError, decode
from qr_reader.detector.detector import detect_sample
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

sys.path.insert(0, "src/qr_reader/tests/detector")
from test_hough_harness import _make_background


def _config_v12_default() -> AugmentationConfig:
    return AugmentationConfig(
        version=12,
        content="https://www.rikvoorhaar.com",
        error_correction="M",
        ppm_range=(5.0, 12.0),
        target_ppm_range=(4.0, 10.0),
        jitter_fraction=0.15,
        feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.2, 1.0),
        noise_sigma_range=(1.0, 5.0),
        jpeg_quality_range=(65, 95),
        global_seed=42,
    )


def _config_v12_clean() -> AugmentationConfig:
    return AugmentationConfig(
        version=12,
        content="https://www.rikvoorhaar.com",
        error_correction="M",
        ppm_range=(10.0, 10.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(10.0, 10.0),
        feather_sigma_range=(0.5, 0.5),
        blur_sigma_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        jpeg_quality_range=(100, 100),
        global_seed=42,
    )


def test_one(config: AugmentationConfig, name: str) -> None:
    print(f"  --- {name} ---")
    bg = _make_background(640, 640)
    rng = np.random.default_rng(42)
    image, metadata = generate_sample(rng, config, bg)

    try:
        matrix = detect_sample(image)
    except ValueError as e:
        print(f"    detect_sample FAILED: {e}")
        print(f"    → Pipeline cannot detect QR at all on {name}")
        return False

    print(f"    detect_sample OK: matrix shape={matrix.shape}")
    print(f"    metadata says version={metadata['version']}, "
          f"N={metadata['N']}")

    try:
        decoded = decode(matrix)
        print(f"    decode OK: '{decoded}'")
        if decoded == config.content:
            print(f"    ✓ Content matches expected")
        else:
            print(f"    ✗ Content mismatch: expected '{config.content}'")
        return True
    except DecodeError as e:
        print(f"    decode FAILED: {e}")
        return False


def main() -> None:
    print("=" * 70)
    print("I4 — D non-blocking validation")
    print("=" * 70)
    print()

    ok_default = test_one(_config_v12_default(), "v12-default (the failing config)")
    print()

    ok_clean = test_one(_config_v12_clean(), "v12-clean (baseline)")
    print()

    print("=" * 70)
    print("RESULT:", "PASS" if (ok_default and ok_clean) else "PARTIAL" if ok_default else "FAIL")
    print()
    if ok_default:
        print("  D is NON-BLOCKING: the full pipeline correctly decodes")
        print("  despite Hough-level D failures (missing finder edges).")
        print("  → D-fix phases (10, 11, 14) are LOW priority.")
    else:
        print("  D is BLOCKING: the full pipeline cannot decode.")
        print("  → D-fix phases (10, 11, 14) are HIGH priority.")


if __name__ == "__main__":
    main()
