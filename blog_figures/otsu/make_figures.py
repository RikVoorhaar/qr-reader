#!/usr/bin/env python3
"""Build the Otsu-thresholding figures for the QR-reader blog post.

Outputs:

* ``blog/src/assets/blog/qr-reader/otsu-color-gray.png``
    The same captured crop in color and in grayscale, side by side.

* ``blog/src/assets/blog/qr-reader/otsu-threshold.gif``
    The grayscale crop thresholded at a moving threshold ``t`` (left), next to
    its intensity histogram (right).  The Otsu threshold is marked with a fixed
    orange dashed line; a cyan line tracks the current ``t``.  The histogram
    area left of ``t`` is colored like the dark class and the area right of it
    like the light class, so the image and the histogram change together.  The
    threshold sweeps Otsu -> light end -> Otsu -> dark end -> Otsu and loops
    seamlessly, with no pause at any value.

It uses the same captured photo and crop as ``rle-scan.gif`` and
``candidate-clustering.gif``:

    data/bench/captured-fullsize/qr_code_ - 99.jpeg

Run from the repository root:

    .venv/bin/python blog_figures/otsu/make_figures.py

The script is deterministic: the photo, crop, Otsu threshold and animation
timeline are all fixed.  It writes only the two figures.
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
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.qr_gen import binarize_image  # noqa: E402

# ── Figure configuration ─────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
IMAGE_NAME = "qr_code_ - 99.jpeg"
OUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

QR_CROP_SCALE = 1.44  # same square crop as the candidate figures
TARGET_H = 600

# ── Palette (shared with the candidate figures) ──────────────────────────────

INK = "#1c2136"
MUTED = "#6b7280"
BINARY_DARK = "#1c2136"
BINARY_LIGHT = "#f6f1da"
ORANGE = "#ff7800"
CYAN = "#00dcff"
WHITE = "#ffffff"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)

# ── Animation timeline ───────────────────────────────────────────────────────

FPS = 20
FRAME_MS = 1000 // FPS

# The threshold sweeps Otsu -> white -> Otsu -> black -> Otsu, then loops.
# The first and last frames are both Otsu, so the loop is seamless.  The
# segment lengths keep the sweep at a roughly constant number of intensity
# levels per frame.
LOW_THRESHOLD = 64
HIGH_THRESHOLD = 255
SWEEP_DOWN_FRAMES = 37
SWEEP_UP_FRAMES = 37
SWEEP_HIGH_FRAMES = 23
SWEEP_BACK_FRAMES = 23


# ── Data ─────────────────────────────────────────────────────────────────────


def _locate_qr(image: np.ndarray) -> np.ndarray:
    """Return the four QR corners in full-resolution image coordinates."""
    shrink = 1200.0 / max(image.shape)
    small = cv2.resize(
        image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA
    )
    found, points = cv2.QRCodeDetector().detect(small)
    if not found or points is None:
        raise RuntimeError("cv2.QRCodeDetector did not find a QR code")
    return points.reshape(-1, 2) / shrink


def load_crop() -> tuple[np.ndarray, np.ndarray]:
    """Load the colour and grayscale crop used by the candidate figures."""
    path = DATA_DIR / IMAGE_NAME
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    corners = _locate_qr(gray)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * QR_CROP_SCALE / 2.0

    left = max(0, int(centre_x - half))
    right = min(bgr.shape[1], int(centre_x + half))
    top = max(0, int(centre_y - half))
    bottom = min(bgr.shape[0], int(centre_y + half))

    rgb = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    gray = gray[top:bottom, left:right]
    rgb = cv2.resize(rgb, (TARGET_H, TARGET_H), interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, (TARGET_H, TARGET_H), interpolation=cv2.INTER_AREA)
    return rgb, gray


def otsu_threshold(gray: np.ndarray) -> int:
    """The threshold the detector actually uses (OpenCV's Otsu)."""
    threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return int(threshold)


# ── Static colour / grayscale figure ─────────────────────────────────────────


def build_color_gray(rgb: np.ndarray, gray: np.ndarray) -> plt.Figure:
    fig = plt.figure(figsize=(10.0, 5.0), dpi=160)

    color_ax = fig.add_axes([0.03, 0.08, 0.42, 0.84])
    gray_ax = fig.add_axes([0.55, 0.08, 0.42, 0.84])

    color_ax.imshow(rgb, interpolation="nearest")
    color_ax.set_title("color photo", color=INK, fontsize=15, pad=10)
    color_ax.axis("off")

    gray_ax.imshow(gray, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    gray_ax.set_title("grayscale", color=INK, fontsize=15, pad=10)
    gray_ax.axis("off")

    fig.text(
        0.5,
        0.5,
        "→",
        color=MUTED,
        fontsize=26,
        ha="center",
        va="center",
    )
    return fig


# ── Threshold animation ──────────────────────────────────────────────────────


def _threshold_frame(
    gray: np.ndarray,
    hist: np.ndarray,
    threshold: int,
    otsu: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(10.0, 5.0), dpi=160)

    image_ax = fig.add_axes([0.03, 0.08, 0.42, 0.84])
    hist_ax = fig.add_axes([0.56, 0.16, 0.40, 0.72])

    # Left: the binary image at the current threshold.  True = light, matching
    # qr_reader.qr_gen.binarize_image.
    binary = gray > threshold
    image_ax.imshow(
        binary,
        cmap=ListedColormap([BINARY_DARK, BINARY_LIGHT]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    image_ax.set_title(
        f"thresholded image   t = {threshold}",
        color=INK,
        fontsize=15,
        pad=10,
    )
    image_ax.axis("off")

    # Right: intensity histogram, split into the two classes at t.
    x = np.arange(256)
    ymax = float(hist.max()) * 1.14
    hist_ax.fill_between(
        x,
        hist,
        0,
        where=x <= threshold,
        step="mid",
        color=BINARY_DARK,
        linewidth=0,
        zorder=2,
    )
    hist_ax.fill_between(
        x,
        hist,
        0,
        where=x > threshold,
        step="mid",
        color=BINARY_LIGHT,
        linewidth=0,
        zorder=2,
    )
    hist_ax.step(x, hist, where="mid", color=INK, lw=0.8, alpha=0.55, zorder=3)

    # Fixed Otsu marker, then the moving threshold on top of it.
    hist_ax.axvline(
        otsu,
        color=ORANGE,
        lw=2.2,
        ls=(0, (4, 3)),
        zorder=5,
    )
    hist_ax.axvline(threshold, color=CYAN, lw=3.2, zorder=4)

    # Current threshold label, kept beside the line so the line stays visible.
    if threshold < 225:
        label_x = min(threshold + 5.0, 248.0)
        label_ha = "left"
    else:
        label_x = threshold - 5.0
        label_ha = "right"
    hist_ax.text(
        label_x,
        ymax * 0.985,
        f"t = {threshold}",
        color=CYAN,
        fontsize=12,
        fontweight="bold",
        ha=label_ha,
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.18",
            facecolor=WHITE,
            edgecolor="none",
            alpha=0.85,
        ),
        zorder=6,
    )

    hist_ax.set_xlim(0, 255)
    hist_ax.set_ylim(0, ymax)
    hist_ax.set_xlabel("pixel intensity", color=INK)
    hist_ax.set_ylabel("pixel count", color=INK)
    hist_ax.set_title("intensity histogram", color=INK, fontsize=15, pad=28)
    hist_ax.text(
        0.5,
        1.015,
        f"orange dashed = Otsu threshold ({otsu})",
        transform=hist_ax.transAxes,
        color=ORANGE,
        fontsize=10.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    hist_ax.set_xticks([0, 64, 128, 192, 255])
    hist_ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ("top", "right"):
        hist_ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        hist_ax.spines[spine].set_color("#c9c4b4")

    return fig


def _linear_segment(start: int, end: int, count: int) -> list[int]:
    """Integer threshold values from ``start`` to ``end``, both included."""
    if count <= 1:
        return [end]
    return [
        int(round(start + (end - start) * i / (count - 1)))
        for i in range(count)
    ]


def _threshold_sequence(otsu: int) -> list[int]:
    """Threshold values for one seamless loop, starting and ending at Otsu.

    Otsu -> light end -> Otsu -> dark end -> Otsu.  The final Otsu frame is
    dropped so the GIF can wrap straight back to the first frame without a
    duplicate frame or a pause.
    """
    values = [otsu]
    values += _linear_segment(otsu, LOW_THRESHOLD, SWEEP_DOWN_FRAMES)[1:]
    values += _linear_segment(LOW_THRESHOLD, otsu, SWEEP_UP_FRAMES)[1:]
    values += _linear_segment(otsu, HIGH_THRESHOLD, SWEEP_HIGH_FRAMES)[1:]
    values += _linear_segment(HIGH_THRESHOLD, otsu, SWEEP_BACK_FRAMES)[1:]
    return values[:-1]


def _figure_to_array(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return buffer[:, :, :3].copy()


def _build_palette(frames: list[np.ndarray], colors: int = 255) -> Image.Image:
    """One adaptive palette shared by every frame, to avoid colour flicker."""
    sample = frames[:: max(1, len(frames) // 12)]
    montage = Image.new(
        "RGB",
        (sample[0].shape[1], sample[0].shape[0] * len(sample)),
    )
    for i, frame in enumerate(sample):
        montage.paste(Image.fromarray(frame), (0, i * frame.shape[0]))
    return montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def build_threshold_gif(
    gray: np.ndarray, otsu: int
) -> tuple[list[Image.Image], Image.Image]:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    thresholds = _threshold_sequence(otsu)

    arrays: list[np.ndarray] = []
    for threshold in thresholds:
        fig = _threshold_frame(gray, hist, threshold, otsu)
        arrays.append(_figure_to_array(fig))
        plt.close(fig)

    palette = _build_palette(arrays)
    frames = [
        Image.fromarray(array).quantize(
            palette=palette, dither=Image.Dither.NONE
        )
        for array in arrays
    ]
    return frames, palette


def save_gif(path: Path, frames: list[Image.Image]) -> None:
    durations = [FRAME_MS] * len(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    rgb, gray = load_crop()
    otsu = otsu_threshold(gray)
    print(f"image: {IMAGE_NAME}")
    print(f"crop: {gray.shape[1]}x{gray.shape[0]}")
    print(f"Otsu threshold: {otsu}")

    color_fig = build_color_gray(rgb, gray)
    color_path = args.out_dir / "otsu-color-gray.png"
    color_path.parent.mkdir(parents=True, exist_ok=True)
    color_fig.savefig(
        color_path,
        dpi=160,
        facecolor=WHITE,
        metadata={"Software": "qr-reader blog figure"},
    )
    plt.close(color_fig)
    print(f"wrote {color_path}")

    frames, _ = build_threshold_gif(gray, otsu)
    gif_path = args.out_dir / "otsu-threshold.gif"
    save_gif(gif_path, frames)
    seconds = len(frames) / FPS
    print(f"wrote {gif_path} ({len(frames)} frames, ~{seconds:.1f}s)")


if __name__ == "__main__":
    main()
