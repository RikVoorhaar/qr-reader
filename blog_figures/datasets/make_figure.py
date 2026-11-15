#!/usr/bin/env python3
"""Three test-set examples side by side: synthetic, composited, captured.

The three panels are cropped around the QR code and shown at the same scale.
All three use the same low version so the visual comparison is direct.

Run from the repository root:

    .venv/bin/python blog_figures/datasets/make_figure.py
    .venv/bin/python blog_figures/datasets/make_figure.py --captured "data/bench/captured/qr_code_ - 7.jpeg"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.homography import compute_qr_corners  # noqa: E402
from qr_reader.pipeline import decode_image  # noqa: E402
from qr_reader.qr_gen import generate_test_image  # noqa: E402

VERSION = 3
SEED = 1
PAYLOAD = "rikvoorhaar.com"

COMPOSITED_PATH = (
    REPO_ROOT / "data" / "bench" / "synth" / "images" / "medium_v03_s000.jpg"
)
METADATA_PATH = REPO_ROOT / "data" / "bench" / "synth" / "metadata.jsonl"
CAPTURED_PATH = REPO_ROOT / "data" / "bench" / "captured" / "qr_code_ - 23.jpeg"
OUT_PATH = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "dataset-examples.png"
)

LABELS = ("synthetic", "composited", "captured")
BORDER = "#d6d3d1"


def _corners_bbox(corners_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bbox of four QR corners in image (x, y)."""
    x0, y0 = corners_xy.min(axis=0)
    x1, y1 = corners_xy.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _square_crop(
    img: np.ndarray, corners_xy: np.ndarray, pad_fraction: float = 0.22
) -> np.ndarray:
    """Crop a square around *corners_xy*, padding outside the image with white."""
    x0, y0, x1, y1 = _corners_bbox(corners_xy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * pad_fraction)
    half = side / 2.0

    sx0 = int(np.floor(cx - half))
    sx1 = int(np.ceil(cx + half))
    sy0 = int(np.floor(cy - half))
    sy1 = int(np.ceil(cy + half))

    h, w = img.shape[:2]
    pad_l = max(0, -sx0)
    pad_t = max(0, -sy0)
    pad_r = max(0, sx1 - w)
    pad_b = max(0, sy1 - h)
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(w, sx1), min(h, sy1)

    crop = img[sy0:sy1, sx0:sx1].copy()
    if pad_l or pad_t or pad_r or pad_b:
        fill = (255, 255, 255) if crop.ndim == 3 else 255
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=fill
        )
    return crop


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _composited_corners(path: Path) -> np.ndarray:
    """Read the QR corners for a composited image from metadata.jsonl."""
    for line in METADATA_PATH.read_text().splitlines():
        row = json.loads(line)
        if Path(row["image_path"]).name == path.name:
            c = row["corners_qr"]
            return np.array(
                [c["TL"], c["TR"], c["BR"], c["BL"]], dtype=np.float64
            )
    raise SystemExit(f"{path.name} not found in {METADATA_PATH}")


def _synthetic_example() -> tuple[np.ndarray, np.ndarray]:
    img = generate_test_image(
        seed=SEED,
        content=PAYLOAD,
        version=VERSION,
        noise_std=25.0,
        perspective_max_shift=20.0,
        final_blur_kernel=5,
    )
    result = decode_image(img, expected_payload=PAYLOAD)
    if not result.success or result.H is None:
        raise SystemExit(f"could not detect the synthetic example: {result.error}")
    n = 4 * result.version + 17
    corners = compute_qr_corners(result.H, n)
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    return rgb, corners


def _captured_example(path: Path) -> tuple[np.ndarray, np.ndarray]:
    img = _read_rgb(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = decode_image(gray, expected_payload=PAYLOAD)
    if not result.success or result.H is None:
        raise SystemExit(f"could not detect {path.name}: {result.error}")
    n = 4 * result.version + 17
    corners = compute_qr_corners(result.H, n)
    return img, corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, default=CAPTURED_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    synthetic_img, synthetic_corners = _synthetic_example()
    composited_img = _read_rgb(COMPOSITED_PATH)
    composited_corners = _composited_corners(COMPOSITED_PATH)
    captured_img, captured_corners = _captured_example(args.captured)

    panels = [
        _square_crop(synthetic_img, synthetic_corners),
        _square_crop(composited_img, composited_corners),
        _square_crop(captured_img, captured_corners),
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.8), dpi=300, facecolor="white"
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.02, top=0.92, wspace=0.035)

    for ax, panel, label in zip(axes, panels, LABELS):
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BORDER)
            spine.set_linewidth(1.2)
        ax.set_title(label, fontsize=20, color="#1a1a1a", pad=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
