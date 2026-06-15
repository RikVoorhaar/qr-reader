#!/usr/bin/env python3
"""Phase 6 Deliverable Checkpoint — Pipeline Orchestrator.

Generates small dataset samples and validates the pipeline end-to-end:
1. Generates a 10-sample dataset at easy settings
2. Generates a 20-sample dataset at mixed settings
3. Attempts to decode a subset using the existing QR decoder

Usage:
    python src/qr_reader/scripts/pipeline_checkpoint.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_dataset

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _make_synthetic_backgrounds(tmpdir: Path, count: int = 10) -> Path:
    """Create a directory of small synthetic background images."""
    bg_dir = tmpdir / "backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        # Each background is a unique colour field with a mild gradient
        r = np.linspace(80 + i * 10, 180 + i * 10, 480, dtype=np.uint8)
        g = np.linspace(60 + i * 5, 160 + i * 5, 480, dtype=np.uint8)
        b = np.linspace(100 + i * 8, 200 + i * 8, 480, dtype=np.uint8)
        r_2d = np.tile(r[:, np.newaxis], (1, 640))
        g_2d = np.tile(g[:, np.newaxis], (1, 640))
        b_2d = np.tile(b[:, np.newaxis], (1, 640))
        bg = np.stack([r_2d, g_2d, b_2d], axis=-1)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(bg_dir / f"bg_{i:03d}.jpg"), bg_bgr)

    return bg_dir


def _run_easy_check(output_dir: Path, bg_dir: Path) -> None:
    """Generate 10 easy samples and inspect metadata."""
    print("=" * 60)
    print("Check 1: 10 easy samples")
    print("=" * 60)

    easy = AugmentationConfig(
        version=1,
        content="EasyTest",
        error_correction="L",
        ppm_range=(12.0, 12.0),
        rotation_deg_range=(0.0, 0.0),
        jitter_fraction=0.0,
        aspect_scale_range=(1.0, 1.0),
        target_ppm_range=(8.0, 8.0),
        feather_sigma_range=(0.5, 0.5),
        blur_sigma_range=(0.0, 0.0),
        noise_sigma_range=(0.0, 0.0),
        jpeg_quality_range=(100, 100),
    )

    easy_dir = output_dir / "easy"
    generate_dataset(easy, bg_dir, easy_dir, num_samples=10)

    # Inspect metadata
    meta_path = easy_dir / "metadata.jsonl"
    lines = meta_path.read_text().strip().split("\n")
    print(f"  Generated {len(lines)} metadata entries")
    for line in lines[:3]:
        print(f"    {line}")


def _run_mixed_check(output_dir: Path, bg_dir: Path) -> None:
    """Generate 20 mixed-difficulty samples."""
    print("\n" + "=" * 60)
    print("Check 2: 20 mixed-difficulty samples")
    print("=" * 60)

    mixed = AugmentationConfig(
        version=3,
        content="MixedDifficulty42!",
        error_correction="M",
        ppm_range=(6.0, 14.0),
        rotation_deg_range=(0.0, 45.0),
        jitter_fraction=0.1,
        aspect_scale_range=(0.9, 1.1),
        target_ppm_range=(5.0, 10.0),
        feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.0, 0.5),
        noise_sigma_range=(0.0, 3.0),
        jpeg_quality_range=(80, 100),
    )

    mixed_dir = output_dir / "mixed"
    generate_dataset(mixed, bg_dir, mixed_dir, num_samples=20)

    meta_path = mixed_dir / "metadata.jsonl"
    lines = meta_path.read_text().strip().split("\n")
    print(f"  Generated {len(lines)} metadata entries")
    for line in lines[:3]:
        print(f"    {line}")


def _run_inspection(output_dir: Path) -> None:
    """Inspect metadata content and image dimensions."""
    print("\n" + "=" * 60)
    print("Check 3: Metadata inspection (mixed samples)")
    print("=" * 60)

    import json

    meta_path = output_dir / "mixed" / "metadata.jsonl"
    with open(meta_path) as f:
        records = [json.loads(line) for line in f]

    # Verify all required keys across all records
    required_keys = {
        "sample_index",
        "seed",
        "background_path",
        "payload",
        "version",
        "N",
        "ecl",
        "pixels_per_module",
        "corners_qr",
        "augmentations",
    }
    for rec in records:
        missing = required_keys - set(rec.keys())
        if missing:
            print(f"  WARNING: Record {rec['sample_index']} missing keys: {missing}")

    # Check image files exist
    images_dir = output_dir / "mixed" / "images"
    image_files = sorted(images_dir.iterdir())
    print(f"  Images: {len(image_files)} files")
    if image_files:
        img = cv2.imread(str(image_files[0]))
        print(f"  First image shape: {img.shape}, dtype={img.dtype}")

    # Check augmentations spread
    augs = [r["augmentations"] for r in records]
    print(
        f"  rotation_deg range: "
        f"{min(a['rotation_deg'] for a in augs):.1f} – "
        f"{max(a['rotation_deg'] for a in augs):.1f}"
    )
    print(
        f"  blur_sigma range: "
        f"{min(a['blur_sigma'] for a in augs):.2f} – "
        f"{max(a['blur_sigma'] for a in augs):.2f}"
    )
    print(
        f"  noise_sigma range: "
        f"{min(a['noise_sigma'] for a in augs):.1f} – "
        f"{max(a['noise_sigma'] for a in augs):.1f}"
    )
    print(
        f"  jpeg_quality range: "
        f"{min(a['jpeg_quality'] for a in augs)} – "
        f"{max(a['jpeg_quality'] for a in augs)}"
    )


def _run_decodability_check(output_dir: Path) -> None:
    """Attempt to decode easy samples using the existing QR decoder."""
    print("\n" + "=" * 60)
    print("Check 4: Decodability check (easy samples)")
    print("=" * 60)

    from qr_reader.decoder.decoder import decode
    from qr_reader.detector.detector import detect_sample

    images_dir = output_dir / "easy" / "images"
    meta_path = output_dir / "easy" / "metadata.jsonl"

    import json

    with open(meta_path) as f:
        records = [json.loads(line) for line in f]

    decoded_count = 0
    for rec in records:
        img_path = images_dir / f"{rec['sample_index']:06d}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            bits = detect_sample(image_rgb)
            text = decode(bits)
            success = text == rec["payload"]
            if success:
                decoded_count += 1
            else:
                print(
                    f"  Sample {rec['sample_index']}: decoded mismatch "
                    f"({text!r} vs {rec['payload']!r})"
                )
        except Exception as exc:
            print(f"  Sample {rec['sample_index']}: decode failed — {exc}")

    total = len(records)
    if total > 0:
        print(f"  Decoded {decoded_count}/{total} samples successfully")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pipeline_checkpoint_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        # Create synthetic backgrounds
        bg_dir = _make_synthetic_backgrounds(tmpdir)

        # Run checks
        _run_easy_check(tmpdir, bg_dir)
        _run_mixed_check(tmpdir, bg_dir)
        _run_inspection(tmpdir)
        _run_decodability_check(tmpdir)

    print("\n✅ Phase 6 checkpoint complete.")


if __name__ == "__main__":
    main()
