#!/usr/bin/env python3
"""Build the "3x3 vote vs one pixel" figure for the QR-reader blog.

One real capture (``qr_code_ - 77.jpeg``, version 10) contains a dark module
where the single centre sample lands on a small bright speck, so sampling only
that pixel reads *light* (wrong), while eight of the nine 3x3 samples are dark,
so the weighted vote reads *dark* (right).

Left panel shows the single centre sample as one white dot.  Right panel shows
all nine samples as filled white/black dots and labels the weighted result
(centre weight 2, total 10).  Both panels share the same square crop of the
photo and the same projected module grid.

Run from the repository root::

    .venv/bin/python blog_figures/sampling_vote/make_figure.py

The script is deterministic (Agg backend, no RNG) and writes

    blog/src/assets/blog/qr-reader/sampling-vote.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qrcode
from matplotlib.patches import Polygon
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.decoder.format_info import decode_format_info  # noqa: E402
from qr_reader.decoder.tables import ECL_NAMES  # noqa: E402
from qr_reader.detector.homography import project_points  # noqa: E402
from qr_reader.detector.sample import (  # noqa: E402
    compute_adaptive_threshold,
    sample_qr_bits,
)
from qr_reader.pipeline import decode_image  # noqa: E402

DATA_PATH = (
    REPO_ROOT / "data" / "bench" / "captured-fullsize" / "qr_code_ - 77.jpeg"
)
OUT_PATH = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "sampling-vote.png"
)
PAYLOAD = "rikvoorhaar.com"

MODULE_ROW = 56
MODULE_COL = 37
CROP_HALF_PX = 48.0

SAMPLE_OFFSETS = (0.25, 0.5, 0.75)
SAMPLE_WEIGHTS = np.array(
    [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 1.0]]
)

WRONG_COLOR = "#ff4d4d"
RIGHT_COLOR = "#3ddc84"
GRID_DARK = "#101010"

ECL_TO_QRCODE = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}


def reference_matrix(version: int, ecl: str, mask: int) -> np.ndarray:
    """Clean ``qrcode`` matrix for the same payload/version/ECL/mask, [row, col]."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ECL_TO_QRCODE[ecl],
        box_size=1,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(PAYLOAD)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=bool)


def centre_values(gray: np.ndarray, H: np.ndarray, N: int) -> np.ndarray:
    """Bilinear single-centre samples, indexed [row, col]."""
    rows, cols = np.mgrid[0:N, 0:N]
    grid_xy = np.column_stack([(cols + 0.5).ravel(), (rows + 0.5).ravel()])
    img_xy = project_points(H, grid_xy.astype(np.float64))
    vals = map_coordinates(
        gray.astype(np.float64),
        np.stack([img_xy[:, 1], img_xy[:, 0]]),
        order=1,
        mode="nearest",
    )
    return vals.reshape(N, N)


def analyse(row: int = MODULE_ROW, col: int = MODULE_COL):
    """Run the production pipeline on the chosen capture and check the story."""
    gray = cv2.cvtColor(cv2.imread(str(DATA_PATH)), cv2.COLOR_BGR2GRAY)

    result = decode_image(gray, expected_payload=PAYLOAD)
    if not result.success or result.H is None or result.version is None:
        raise SystemExit(f"production pipeline failed on {DATA_PATH.name}")
    H, version = result.H, result.version
    N = 4 * version + 17

    bits_3x3 = sample_qr_bits(gray, H, N)
    if decode_qr(bits_3x3) != PAYLOAD:
        raise SystemExit("3x3 vote did not decode the expected payload")

    threshold = compute_adaptive_threshold(gray, H, N)
    centre = centre_values(gray, H, N)
    bits_centre = (~(centre > threshold)).T  # [col, row]

    ecl_idx, mask = decode_format_info(bits_3x3, version)
    ref = reference_matrix(version, ECL_NAMES[ecl_idx], mask)

    nine = np.empty((3, 3))
    for i, dy in enumerate(SAMPLE_OFFSETS):
        for j, dx in enumerate(SAMPLE_OFFSETS):
            pt = project_points(H, np.array([[col + dx, row + dy]], dtype=float))
            nine[i, j] = map_coordinates(
                gray.astype(np.float64),
                np.array([[pt[0, 1]], [pt[0, 0]]]),
                order=1,
                mode="nearest",
            )[0]

    white_votes = float(np.sum((nine > threshold) * SAMPLE_WEIGHTS))
    bit_3x3 = not (white_votes >= 5.0)
    bit_centre = bool(bits_centre[col, row])

    checks = {
        "centre_bit_wrong": bit_centre != bool(ref[row, col]),
        "vote_bit_right": bit_3x3 == bool(ref[row, col]),
        "vote_is_decisive": int(np.sum(nine > threshold)) <= 3,
    }
    if not all(checks.values()):
        raise SystemExit(f"module {row, col} does not show the expected contrast: {checks}")

    print(f"image      : {DATA_PATH.name}")
    print(f"version    : {version} (N={N}), ECL {ECL_NAMES[ecl_idx]}, mask {mask}")
    print(f"module     : (row={row}, col={col}), reference bit = dark")
    print(f"threshold  : {threshold:.2f}")
    print(f"nine       : {np.round(nine, 1).tolist()}")
    print(f"centre bit : {'dark' if bit_centre else 'light'} (WRONG)")
    print(f"3x3 bit    : {'dark' if bit_3x3 else 'light'} (right), "
          f"dark weight {10.0 - white_votes:.0f}/10")

    return gray, H, N, row, col, threshold, nine


def grid_bounds(H: np.ndarray, x0, y0, x1, y1) -> tuple[float, float, float, float]:
    """Grid-coordinate bounds of an image-space rectangle."""
    H_inv = np.linalg.inv(H)
    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    grid = project_points(H_inv, corners)
    return (
        float(grid[:, 0].min()),
        float(grid[:, 1].min()),
        float(grid[:, 0].max()),
        float(grid[:, 1].max()),
    )


def draw_grid(ax, H, col_min, col_max, row_min, row_max):
    """Projected module grid, clipped to the crop."""
    for c in range(int(np.floor(col_min)), int(np.ceil(col_max)) + 1):
        pts = project_points(
            H, np.array([[c, row_min], [c, row_max]], dtype=np.float64)
        )
        ax.plot(pts[:, 0], pts[:, 1], color=GRID_DARK, lw=1.5, alpha=0.30, zorder=2)
        ax.plot(pts[:, 0], pts[:, 1], color="#f2f2f2", lw=0.7, alpha=0.75, zorder=3)
    for r in range(int(np.floor(row_min)), int(np.ceil(row_max)) + 1):
        pts = project_points(
            H, np.array([[col_min, r], [col_max, r]], dtype=np.float64)
        )
        ax.plot(pts[:, 0], pts[:, 1], color=GRID_DARK, lw=1.5, alpha=0.30, zorder=2)
        ax.plot(pts[:, 0], pts[:, 1], color="#f2f2f2", lw=0.7, alpha=0.75, zorder=3)


def module_quad(H, row, col):
    corners = np.array(
        [[col, row], [col + 1, row], [col + 1, row + 1], [col, row + 1]],
        dtype=np.float64,
    )
    return project_points(H, corners)


def sample_dot(ax, x, y, light: bool, size: float, *, zorder: int = 8) -> None:
    """Filled dot in the colour the sample votes for, with a contrasting edge."""
    face = "white" if light else GRID_DARK
    edge = GRID_DARK if light else "white"
    ax.scatter(
        [x],
        [y],
        s=size,
        facecolors=face,
        edgecolors=edge,
        linewidths=0.9,
        zorder=zorder,
    )


def draw_module_marks(ax, H, row, col, color):
    quad = module_quad(H, row, col)
    ax.add_patch(
        Polygon(
            quad,
            closed=True,
            facecolor=color,
            edgecolor="none",
            alpha=0.11,
            zorder=4,
        )
    )
    ax.add_patch(
        Polygon(
            quad,
            closed=True,
            facecolor="none",
            edgecolor=GRID_DARK,
            lw=5.8,
            alpha=0.5,
            joinstyle="round",
            zorder=5,
        )
    )
    ax.add_patch(
        Polygon(
            quad,
            closed=True,
            facecolor="none",
            edgecolor=color,
            lw=3.4,
            joinstyle="round",
            zorder=6,
        )
    )


def sample_xy(H, row, col):
    return project_points(
        H, np.array([[col + 0.5, row + 0.5]], dtype=np.float64)
    )[0]


def render(gray, H, N, row, col, threshold, nine, out_path: Path = OUT_PATH):
    x_img, y_img = sample_xy(H, row, col)
    half = CROP_HALF_PX
    x0, x1 = x_img - half, x_img + half
    y0, y1 = y_img - half, y_img + half
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        raise SystemExit("crop leaves the image")

    col_min, row_min, col_max, row_max = grid_bounds(H, x0, y0, x1, y1)

    fig, axes = plt.subplots(
        1, 2, figsize=(10.4, 5.2), dpi=300, facecolor="white"
    )
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.055, top=0.9, wspace=0.045)

    for ax, title in zip(axes, ("one pixel", "3×3 vote")):
        ax.imshow(
            gray,
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=[0, gray.shape[1], gray.shape[0], 0],
            interpolation="nearest",
            zorder=1,
        )
        draw_grid(ax, H, col_min, col_max, row_min, row_max)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_box_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="center", fontsize=22, color="#1a1a1a", pad=10)

    # Left panel: the single centre sample, in the colour it votes for.
    ax = axes[0]
    draw_module_marks(ax, H, row, col, WRONG_COLOR)
    cx, cy = sample_xy(H, row, col)
    sample_dot(ax, cx, cy, True, 300, zorder=8)

    # Right panel: all nine samples as black/white dots.  The centre is only
    # slightly larger; the exact weighting is stated by the vote label.
    ax = axes[1]
    draw_module_marks(ax, H, row, col, RIGHT_COLOR)
    light = nine > threshold
    for i, dy in enumerate(SAMPLE_OFFSETS):
        for j, dx in enumerate(SAMPLE_OFFSETS):
            if dx == 0.5 and dy == 0.5:
                continue
            pt = project_points(H, np.array([[col + dx, row + dy]], dtype=float))[0]
            sample_dot(ax, pt[0], pt[1], bool(light[i, j]), 65, zorder=8)
    sample_dot(ax, cx, cy, bool(light[1, 1]), 95, zorder=9)

    white_votes = float(np.sum(light * SAMPLE_WEIGHTS))
    dark_votes = 10.0 - white_votes
    if dark_votes >= 5.0:
        vote_label = f"{dark_votes:g}/10 dark"
    else:
        vote_label = f"{white_votes:g}/10 white"

    quad = module_quad(H, row, col)
    module_w = float(np.linalg.norm(quad[1] - quad[0]))
    ax.text(
        cx + 1.8 * module_w,
        cy,
        vote_label,
        fontsize=13.5,
        fontweight="bold",
        color=GRID_DARK,
        ha="left",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="white",
            ec=GRID_DARK,
            lw=1.0,
            alpha=0.96,
        ),
        zorder=10,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row", type=int, default=MODULE_ROW)
    ap.add_argument("--col", type=int, default=MODULE_COL)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()
    gray, H, N, row, col, threshold, nine = analyse(args.row, args.col)
    render(gray, H, N, row, col, threshold, nine, args.out)


if __name__ == "__main__":
    main()
