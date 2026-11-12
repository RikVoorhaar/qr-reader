#!/usr/bin/env python3
"""Build the illumination-correction figure for the QR-reader blog.

The photo is a real captured image (``qr_code_ - 23.jpeg``) with strong
illumination variation.  A single global Otsu threshold produces a very noisy
binary: the finder rings and data modules fragment.  Estimating the
illumination field with a 101x101 box blur, dividing it out, and applying Otsu
once to the normalized image recovers the symbol.

Layout (2x3):

    input photo          estimated illumination L       corrected R = I / L
    intensity histogram  Otsu on input                  Otsu after correction

The histogram is for the QR crop, but the Otsu thresholds are computed on the
full image, which is what the pipeline actually does.  Before correction the
threshold sits at the edge of the dark mode; after correction it falls into the
valley between the dark and light modes.

Output:

    blog/src/assets/blog/qr-reader/illumination-correction.png

Run from the repository root:

    python blog_figures/illumination/make_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "bench" / "captured"
IMAGE_NAME = "qr_code_ - 23.jpeg"
OUTPUT_PATH = (
    REPO_ROOT
    / "blog"
    / "src"
    / "assets"
    / "blog"
    / "qr-reader"
    / "illumination-correction.png"
)

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.qr_gen import (  # noqa: E402
    binarize_image_illumination_corrected,
    estimate_illumination,
    illumination_corrected_image,
)

# Binary palette matches the other blog figures.
BINARY_DARK = "#1c2136"
BINARY_LIGHT = "#f6f1da"
MUTED = "#6b7280"
WHITE = "#ffffff"
ORANGE = "#ff7800"
ORANGE_DARK = "#c2410c"

CROP_MARGIN = 0.06

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def _locate_qr(gray: np.ndarray) -> np.ndarray:
    """Return the four QR corners in full-resolution image coordinates."""
    detector = cv2.QRCodeDetector()
    found, points = detector.detect(gray)
    if found and points is not None:
        return points.reshape(-1, 2)

    # Some photos are decoded by our pipeline but not by OpenCV's detector.
    # Fall back to the corrected pipeline's homography.
    from qr_reader.detector.homography import compute_qr_corners
    from qr_reader.pipeline import decode_image

    decoded = decode_image(gray, expected_payload="rikvoorhaar.com")

    if decoded.H is None or decoded.version is None:
        raise RuntimeError("could not locate the QR code")
    n_modules = 4 * decoded.version + 17
    return compute_qr_corners(decoded.H, n_modules)


def _crop_box(
    corners: np.ndarray,
    shape: tuple[int, int],
    margin_fraction: float = CROP_MARGIN,
) -> tuple[int, int, int, int]:
    """Crop around the QR quad, with a margin proportional to its size."""
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    margin = margin_fraction * max(x1 - x0, y1 - y0)
    h, w = shape
    return (
        max(0, int(x0 - margin)),
        max(0, int(y0 - margin)),
        min(w, int(x1 + margin)),
        min(h, int(y1 + margin)),
    )


def _otsu(gray: np.ndarray) -> tuple[float, np.ndarray]:
    threshold, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return float(threshold), binary.astype(bool)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def build_figure(
    gray: np.ndarray,
    corners: np.ndarray,
    *,
    margin_fraction: float = CROP_MARGIN,
) -> plt.Figure:
    """Simple 2x2 figure used by the candidate gallery."""
    x0, y0, x1, y1 = _crop_box(corners, gray.shape, margin_fraction)

    corrected = illumination_corrected_image(gray)
    _, binary_otsu = _otsu(gray)
    binary_corrected = binarize_image_illumination_corrected(gray)

    panels = [
        (gray[y0:y1, x0:x1], "Input photo", "gray"),
        (corrected[y0:y1, x0:x1], "Illumination-corrected", "gray"),
        (binary_otsu[y0:y1, x0:x1], "Otsu on input", "binary"),
        (binary_corrected[y0:y1, x0:x1], "Otsu after correction", "binary"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 9.2))
    _draw_panels(axes.ravel(), panels, binary_cmap=ListedColormap(
        [BINARY_DARK, BINARY_LIGHT]
    ))
    fig.tight_layout(pad=1.0)
    return fig


def build_expanded_figure(
    gray: np.ndarray,
    corners: np.ndarray,
    *,
    margin_fraction: float = CROP_MARGIN,
) -> plt.Figure:
    """2x3 figure: transformation, illumination field, corrected image,
    histogram, and the two binary results."""
    x0, y0, x1, y1 = _crop_box(corners, gray.shape, margin_fraction)
    crop = gray[y0:y1, x0:x1]

    illumination = estimate_illumination(gray)
    corrected = illumination_corrected_image(gray)
    corrected_crop = corrected[y0:y1, x0:x1]

    threshold_input, binary_otsu = _otsu(gray)
    threshold_corrected, _ = _otsu(corrected)
    binary_corrected = binarize_image_illumination_corrected(gray)

    fig = plt.figure(figsize=(15.2, 9.4))
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.92], hspace=0.12, wspace=0.08
    )

    # Top row: input, illumination field, corrected image.
    ax_input = fig.add_subplot(grid[0, 0])
    ax_input.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    ax_input.set_title("Input photo", color=BINARY_DARK, pad=8)

    ax_field = fig.add_subplot(grid[0, 1])
    field = ax_field.imshow(
        illumination[y0:y1, x0:x1],
        cmap="magma",
        vmin=0,
        vmax=255,
        interpolation="antialiased",
    )
    ax_field.set_title("Estimated illumination L", color=BINARY_DARK, pad=8)
    colorbar = fig.colorbar(field, ax=ax_field, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=9)
    colorbar.set_label("gray level", fontsize=9, color=MUTED)

    ax_corrected = fig.add_subplot(grid[0, 2])
    ax_corrected.imshow(
        corrected_crop,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="antialiased",
    )
    ax_corrected.set_title("Corrected R = I / L", color=BINARY_DARK, pad=8)

    # Bottom row: histogram, Otsu on input, Otsu after correction.
    ax_hist = fig.add_subplot(grid[1, 0])
    _draw_histogram(
        ax_hist, crop, corrected_crop, threshold_input, threshold_corrected
    )

    binary_cmap = ListedColormap([BINARY_DARK, BINARY_LIGHT])
    ax_otsu = fig.add_subplot(grid[1, 1])
    ax_otsu.imshow(
        binary_otsu[y0:y1, x0:x1],
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_otsu.set_title("Otsu on input", color=BINARY_DARK, pad=8)

    ax_fixed = fig.add_subplot(grid[1, 2])
    ax_fixed.imshow(
        binary_corrected[y0:y1, x0:x1],
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax_fixed.set_title("Otsu after correction", color=BINARY_DARK, pad=8)

    for ax in (ax_input, ax_field, ax_corrected, ax_otsu, ax_fixed):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.8)
    for spine in ax_hist.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)

    fig.text(
        0.5,
        0.012,
        "L = 101×101 box blur    ·    R = I / L    ·    Otsu on R    ·    3×3 opening",
        ha="center",
        va="bottom",
        fontsize=11,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.06)
    return fig


def _draw_panels(axes, panels, *, binary_cmap: ListedColormap) -> None:
    for ax, (image, title, kind) in zip(axes, panels):
        if kind == "gray":
            ax.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255,
                interpolation="antialiased",
            )
        else:
            ax.imshow(
                image,
                cmap=binary_cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        ax.set_title(title, color=BINARY_DARK, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(0.8)


def _draw_histogram(
    ax,
    input_crop: np.ndarray,
    corrected_crop: np.ndarray,
    threshold_input: float,
    threshold_corrected: float,
) -> None:
    bins = np.linspace(1, 254, 64)
    for data, label, color in (
        (input_crop, "Input", MUTED),
        (corrected_crop, "Corrected", ORANGE),
    ):
        density, edges = np.histogram(data, bins=bins, density=True)
        ax.step(
            edges[:-1],
            density,
            where="post",
            label=label,
            color=color,
            lw=2.0,
        )

    ax.axvline(
        threshold_input,
        color=MUTED,
        ls="--",
        lw=1.4,
        label=f"Otsu input ({threshold_input:.0f})",
    )
    ax.axvline(
        threshold_corrected,
        color=ORANGE,
        ls="--",
        lw=1.4,
        label=f"Otsu corrected ({threshold_corrected:.0f})",
    )

    ax.set_title(
        "QR-region intensity histogram\n(thresholds computed on full image)",
        color=BINARY_DARK,
        pad=8,
        fontsize=13,
    )
    ax.set_xlabel("gray level", color=BINARY_DARK)
    ax.set_ylabel("density", color=BINARY_DARK)
    ax.set_xlim(0, 255)
    ax.tick_params(labelsize=10)
    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
        facecolor=WHITE,
        edgecolor="#d1d5db",
        framealpha=0.95,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--image",
        default=IMAGE_NAME,
        help=f"Captured image filename (default: {IMAGE_NAME!r}).",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Write the old 2x2 figure instead of the expanded 2x3 figure.",
    )
    args = parser.parse_args()

    path = DATA_DIR / args.image
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(
            f"{path} not found. See README.md 'Datasets' section."
        )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners = _locate_qr(gray)

    fig = (
        build_figure(gray, corners)
        if args.simple
        else build_expanded_figure(gray, corners)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.out,
        dpi=220,
        facecolor=WHITE,
        metadata={"Software": "qr-reader blog figure"},
    )
    plt.close(fig)
    print(f"image: {args.image}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
