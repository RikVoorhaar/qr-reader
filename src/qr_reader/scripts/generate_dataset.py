#!/usr/bin/env python3
"""Phase 7.2 — CLI script for batch-generating synthetic QR datasets.

Usage examples
--------------
Generate 1000 medium-difficulty samples from the HomeObjects-3K dataset::

    python src/qr_reader/scripts/generate_dataset.py \\
        --background-dir data/images/train \\
        --output-dir data/synth \\
        --preset medium \\
        --num-samples 1000 \\
        --seed 42 \\
        --version-range 1 10

Override individual fields after picking a preset::

    python src/qr_reader/scripts/generate_dataset.py \\
        --background-dir data/images/train \\
        --output-dir data/synth \\
        --preset hard \\
        --num-samples 500 \\
        --noise-range 0.0 2.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_dataset
from qr_reader.synth.presets import PRESET_MAP

# ===================================================================
# CLI
# ===================================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QR code dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Overrides modify a specific preset field.  Example:\n"
            "  --noise-range 0.0 2.0   # override noise sigma range\n"
            "  --jitter-fraction 0.1    # override jitter fraction\n"
            "\n"
            "Available presets: " + ", ".join(sorted(PRESET_MAP)) + "\n"
        ),
    )

    # Required
    parser.add_argument(
        "--background-dir",
        required=True,
        type=Path,
        help="Directory containing background images (.jpg, .png).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory (images/ and metadata.jsonl will be created).",
    )

    # Preset
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_MAP),
        default="medium",
        help="Difficulty preset (default: medium).",
    )

    # Count
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000).",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed for reproducible generation (default: 42).",
    )

    # QR content
    parser.add_argument(
        "--content",
        type=str,
        default=None,
        help="Text payload to encode (default: preset value, or 'QR Reader v1').",
    )

    # Version range
    parser.add_argument(
        "--version-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="QR version range (min, max), e.g. 1 10.",
    )

    # Overrides — individual fields that can override the preset
    parser.add_argument(
        "--jitter-fraction",
        type=float,
        default=None,
        help="Override jitter_fraction (0–1).",
    )
    parser.add_argument(
        "--ppm-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override patch pixels-per-module range (e.g. 8.0 16.0).",
    )
    parser.add_argument(
        "--target-ppm-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override target pixels-per-module range (e.g. 8.0 16.0).",
    )
    parser.add_argument(
        "--feather-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override feather sigma range (px).",
    )
    parser.add_argument(
        "--blur-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override post-composite blur sigma range.",
    )
    parser.add_argument(
        "--noise-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override post-composite noise sigma range.",
    )
    parser.add_argument(
        "--jpeg-range",
        type=int,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="Override JPEG quality range (0–100).",
    )

    return parser


def _apply_overrides(
    config: AugmentationConfig,
    args: argparse.Namespace,
) -> AugmentationConfig:
    """Return a new config with CLI overrides applied.

    We create a fresh copy via ``__dict__`` update so that the original preset
    object is not mutated.
    """
    overrides: dict = {}

    if args.content is not None:
        overrides["content"] = args.content
    if args.version_range is not None:
        v_min, v_max = args.version_range
        overrides["version"] = v_min  # single version; the spec uses config.version
        # We can't easily grid-search versions here — note this in help
    if args.jitter_fraction is not None:
        overrides["jitter_fraction"] = args.jitter_fraction
    if args.ppm_range is not None:
        overrides["ppm_range"] = tuple(args.ppm_range)
    if args.target_ppm_range is not None:
        overrides["target_ppm_range"] = tuple(args.target_ppm_range)
    if args.feather_range is not None:
        overrides["feather_sigma_range"] = tuple(args.feather_range)
    if args.blur_range is not None:
        overrides["blur_sigma_range"] = tuple(args.blur_range)
    if args.noise_range is not None:
        overrides["noise_sigma_range"] = tuple(args.noise_range)
    if args.jpeg_range is not None:
        overrides["jpeg_quality_range"] = tuple(args.jpeg_range)

    if overrides:
        return AugmentationConfig(
            **{**config.__dict__, **overrides},
        )
    return config


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Resolve preset
    base_config = PRESET_MAP[args.preset]
    config = _apply_overrides(base_config, args)

    # Seed
    config.global_seed = args.seed

    # Log config summary
    print(f"Preset:      {args.preset}")
    print(f"Output:      {args.output_dir}")
    print(f"Samples:     {args.num_samples}")
    print(f"Seed:        {args.seed}")
    print(f"Version:     {config.version}")
    print(f"Content:     {config.content!r}")
    print(f"PPM range:   {config.ppm_range}")
    print(f"Target PPM:  {config.target_ppm_range}")
    print(f"Jitter:      {config.jitter_fraction}")
    print(f"Feather:     {config.feather_sigma_range}")
    print(f"Blur:        {config.blur_sigma_range}")
    print(f"Noise:       {config.noise_sigma_range}")
    print(f"JPEG qual:   {config.jpeg_quality_range}")
    print()

    generate_dataset(
        config=config,
        background_dir=args.background_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
