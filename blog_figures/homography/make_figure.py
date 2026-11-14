#!/usr/bin/env python3
"""Build the homography figure for the QR-reader blog post.

The figure shows the twelve finder-pattern corners that the detector fits, and
the mapping from those corners in a captured photo to the canonical corner
positions of a rectified QR grid.

Left panel
    A tight crop of the real photo, with the twelve detected outer finder
    corners marked (four per finder).  No grid.

Right panel
    The same photo warped into a top-down, module-aligned square with the
    inverse homography.  A module grid is overlaid and the twelve canonical
    corners are marked as open circles.  The small filled dots are the detected
    corners mapped through ``H^{-1}``: they should sit on the open circles.

Bottom inset
    A zoom on the top-left finder.  Open circles are the ideal canonical
    corners; filled dots are the mapped detected corners.  Because the true
    residual is sub-pixel at the figure scale, the error vectors are drawn
    ``EXAGGERATION`` times longer than life, and the factor is printed in the
    inset and in the statistics block.  The un-exaggerated mapped points are
    also drawn as small dots, so nothing is silently distorted.

The detector pipeline used here is the real one:

    _prepare_detection -> _rank_triplets -> _fit_candidate_homography

``H`` maps QR-grid (x, y) to image (x, y).  The twelve source points are the
same ones ``_fit_candidate_homography`` builds, in the same [TL, TR, BR, BL]
order per finder:

    TL finder  (0,0), (7,0), (7,7), (0,7)
    TR finder  (N-7,0), (N,0), (N,7), (N-7,7)
    BL finder  (0,N-7), (7,N-7), (7,N), (0,N)

Usage (from the repository root):

    .venv/bin/python blog_figures/homography/make_figure.py

The script is deterministic (Agg backend, no RNG).  It writes

    blog/src/assets/blog/qr-reader/homography.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.detector import (  # noqa: E402
    DetectionResult,
    _canonicalize_corners,
    _fit_candidate_homography,
    _prepare_detection,
    _rank_triplets,
    _triplet_axes,
)
from qr_reader.detector.homography import compute_qr_corners, project_points  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "homography.png"
)

IMAGE_NAME = "qr_code_ - 20.jpeg"
EXPECTED_VERSION = 10

# Rectified canvas: an exact integer number of pixels per module.
RECT_PX_PER_MODULE = 25

# The photo is cropped just above the hand that holds the paper; this is the
# last safe row in this specific capture.
PHOTO_BOTTOM = 2732

# Finder colours.  Three groups only, so the correspondence reads as three
# bundles instead of twelve unrelated lines.
TL_COLOR = "#f5a524"  # amber
TR_COLOR = "#4cc9f0"  # cyan
BL_COLOR = "#8b5cf6"  # violet

GRID_MINOR = "#9ca3af"
GRID_MAJOR = "#374151"
INK = "#111827"
MUTED = "#6b7280"

BOX_BG = "#0d0d0d"
BOX_EDGE = "#3f3f3f"
TEXT_BRIGHT = "#f2f2f2"

EXAGGERATION = 10.0

# Figure layout, in inches / figure fractions.
FIG_W, FIG_H = 10.0, 8.0
TOP_W_IN = 4.25
TOP_H_IN = 4.30
TOP_X = (0.032, 0.540)
TOP_Y = 0.448

INSET_IN = 3.25
INSET_XY = (0.035, 0.014)


@dataclass
class Fit:
    """Everything the figure needs from one detection run."""

    gray: np.ndarray
    version: int
    n_modules: int
    H: np.ndarray
    H_inv: np.ndarray
    src: np.ndarray  # (12, 2) canonical grid corners, (x, y)
    dst: np.ndarray  # (12, 2) detected image corners, (x, y)
    groups: dict[str, np.ndarray]  # finder -> indices into src/dst
    fit_error: float
    rect: np.ndarray
    scale: float

    @property
    def image_errors(self) -> np.ndarray:
        return np.linalg.norm(
            project_points(self.H, self.src) - self.dst, axis=1
        )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(data_dir: Path = DATA_DIR) -> Fit:
    """Run the real detector and build the twelve-point correspondence."""
    path = data_dir / IMAGE_NAME
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"could not read {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = DetectionResult(success=False)
    ctx = _prepare_detection(gray, result)
    ranked = _rank_triplets(ctx)
    if not ranked:
        raise SystemExit("no triplet candidate")
    candidate = ranked[0]
    if candidate.best_version != EXPECTED_VERSION:
        raise SystemExit(
            f"unexpected version estimate {candidate.best_version}, "
            f"expected {EXPECTED_VERSION}"
        )

    version = candidate.best_version
    n = 17 + 4 * version
    H, fit_error = _fit_candidate_homography(ctx, candidate.triplet, n)

    # Rebuild exactly the correspondence _fit_candidate_homography uses, so
    # the residuals plotted here are the ones behind ``fit_error``.
    right, down = _triplet_axes(ctx, candidate.triplet)
    corners = {
        "TL": (
            _canonicalize_corners(
                ctx.global_corners_xy[candidate.triplet.top_left_idx], right, down
            ),
            (0, 0),
        ),
        "TR": (
            _canonicalize_corners(
                ctx.global_corners_xy[candidate.triplet.top_right_idx], right, down
            ),
            (n - 7, 0),
        ),
        "BL": (
            _canonicalize_corners(
                ctx.global_corners_xy[candidate.triplet.bottom_left_idx], right, down
            ),
            (0, n - 7),
        ),
    }
    grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64)

    src: list[list[float]] = []
    dst: list[list[float]] = []
    groups: dict[str, np.ndarray] = {}
    for name, (quad, origin) in corners.items():
        groups[name] = np.arange(len(src), len(src) + 4)
        for i in range(4):
            src.append([origin[0] + grid_offsets[i, 0], origin[1] + grid_offsets[i, 1]])
            dst.append(quad[i].tolist())

    src_arr = np.array(src, dtype=np.float64)
    dst_arr = np.array(dst, dtype=np.float64)

    proj = project_points(H, src_arr)
    recomputed = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
    if abs(recomputed - fit_error) > 1e-9:
        raise SystemExit(
            f"recomputed fit error {recomputed} != returned {fit_error}"
        )

    H_inv = np.linalg.inv(H)
    scale = float(RECT_PX_PER_MODULE)
    rect_size = int(round(scale * n))
    rectified = cv2.warpPerspective(
        gray,
        np.diag([scale, scale, 1.0]) @ H_inv,
        (rect_size, rect_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return Fit(
        gray=gray,
        version=version,
        n_modules=n,
        H=H,
        H_inv=H_inv,
        src=src_arr,
        dst=dst_arr,
        groups=groups,
        fit_error=fit_error,
        rect=rectified,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def finder_color(name: str) -> str:
    return {"TL": TL_COLOR, "TR": TR_COLOR, "BL": BL_COLOR}[name]


def fig_coords(fig: plt.Figure, ax: plt.Axes, pts_xy: np.ndarray) -> np.ndarray:
    """Convert axes data coordinates to figure-fraction coordinates."""
    display = ax.transData.transform(pts_xy)
    return fig.transFigure.inverted().transform(display)


def clip_segment_to_x(p0: np.ndarray, p1: np.ndarray, x: float) -> np.ndarray:
    """Point on the p0→p1 segment at figure x-coordinate ``x``."""
    t = (x - p0[0]) / (p1[0] - p0[0])
    return p0 + t * (p1 - p0)


def rounded_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    ha: str = "left",
    va: str = "top",
    fontsize: float = 10.5,
    transform=None,
):
    ax.text(
        x,
        y,
        text,
        transform=transform if transform is not None else ax.transAxes,
        color=TEXT_BRIGHT,
        fontsize=fontsize,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.38",
            fc=BOX_BG,
            ec=BOX_EDGE,
            lw=0.9,
            alpha=0.94,
        ),
        zorder=30,
    )


def draw_grid(ax: plt.Axes, n: int, scale: float, *, minor_alpha: float, major_alpha: float):
    """Module grid aligned with the rectified canvas."""
    minor: list[list[tuple[float, float]]] = []
    major: list[list[tuple[float, float]]] = []
    for k in range(n + 1):
        x = k * scale
        seg = [[x, 0.0], [x, n * scale]]
        if k % 7 == 0:
            major.append(seg)
        else:
            minor.append(seg)
        seg = [[0.0, x], [n * scale, x]]
        if k % 7 == 0:
            major.append(seg)
        else:
            minor.append(seg)

    ax.add_collection(
        LineCollection(
            minor,
            colors=GRID_MINOR,
            linewidths=0.38,
            alpha=minor_alpha,
            zorder=4,
            antialiased=True,
        )
    )
    ax.add_collection(
        LineCollection(
            major,
            colors=GRID_MAJOR,
            linewidths=0.85,
            alpha=major_alpha,
            zorder=5,
            antialiased=True,
        )
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def add_photo_panel(fig: plt.Figure, fit: Fit, crop: tuple[int, int, int, int]) -> plt.Axes:
    c0, r0, c1, r1 = crop
    panel_h_in = TOP_W_IN  # the photo crop is square
    y = TOP_Y + (TOP_H_IN - panel_h_in) / 2.0 / FIG_H
    ax = fig.add_axes([TOP_X[0], y, TOP_W_IN / FIG_W, panel_h_in / FIG_H])
    ax.imshow(
        fit.gray[r0:r1, c0:c1],
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[c0, c1, r1, r0],
        interpolation="lanczos",
    )
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    rounded_label(ax, 0.014, 0.986, "captured photo")

    # Finder outlines: the detected quadrilateral, in the finder's colour.
    for name, idx in fit.groups.items():
        pts = fit.dst[idx]
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                fill=False,
                edgecolor=finder_color(name),
                lw=0.9,
                alpha=0.48,
                zorder=7,
            )
        )

    # The twelve detected corners.
    for name, idx in fit.groups.items():
        pts = fit.dst[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=30,
            facecolors=finder_color(name),
            edgecolors="#ffffff",
            linewidths=0.9,
            zorder=9,
        )
    return ax


def add_rectified_panel(fig: plt.Figure, fit: Fit, margins: tuple[float, float, float, float]) -> plt.Axes:
    """Top-down panel.  ``margins`` = (left, top, right, bottom) in canvas px."""
    m_l, m_t, m_r, m_b = margins
    size = fit.n_modules * fit.scale

    data_w = size + m_l + m_r
    data_h = size + m_t + m_b
    panel_h_in = TOP_W_IN * data_h / data_w
    ax = fig.add_axes(
        [
            TOP_X[1],
            TOP_Y + (TOP_H_IN - panel_h_in) / 2.0 / FIG_H,
            TOP_W_IN / FIG_W,
            panel_h_in / FIG_H,
        ]
    )
    ax.set_facecolor("white")
    ax.imshow(
        fit.rect,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[0.0, size, size, 0.0],
        interpolation="lanczos",
        zorder=1,
    )
    draw_grid(ax, fit.n_modules, fit.scale, minor_alpha=0.22, major_alpha=0.48)

    # Canonical 7x7 finder squares, in colour.
    for name, idx in fit.groups.items():
        grid_pts = fit.src[idx] * fit.scale
        ax.add_patch(
            Polygon(
                grid_pts,
                closed=True,
                fill=False,
                edgecolor=finder_color(name),
                lw=1.0,
                alpha=0.72,
                zorder=7,
            )
        )

    # Canonical corners: open circles.  Detected corners mapped through H^-1:
    # tiny filled dots.  At this scale the two nearly coincide; the inset shows
    # the separation.
    mapped = project_points(fit.H_inv, fit.dst) * fit.scale
    for name, idx in fit.groups.items():
        ideal = fit.src[idx] * fit.scale
        color = finder_color(name)
        ax.scatter(
            ideal[:, 0],
            ideal[:, 1],
            s=48,
            facecolors="white",
            edgecolors=color,
            linewidths=1.4,
            zorder=9,
        )
        ax.scatter(
            mapped[idx, 0],
            mapped[idx, 1],
            s=13,
            facecolors=color,
            edgecolors="#ffffff",
            linewidths=0.5,
            zorder=10,
        )

    ax.set_xlim(-m_l, size + m_r)
    ax.set_ylim(size + m_b, -m_t)
    ax.set_aspect("equal")
    ax.axis("off")
    rounded_label(ax, 0.014, 0.986, f"rectified grid  ·  {fit.n_modules} × {fit.n_modules}")
    return ax


def add_correspondence(
    fig: plt.Figure, fit: Fit, left_ax: plt.Axes, right_ax: plt.Axes
) -> None:
    """Thin colour-coded bridges for the zoomed finder's four corners.

    Twelve lines across two dense panels read as spaghetti, so only the
    top-left finder — the one the inset zooms — gets lines.  The other eight
    corners keep their colour-coded dots and open circles; the colour key and
    the finder outlines carry that part of the correspondence.
    """
    overlay = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=20)
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)
    overlay.axis("off")
    overlay.patch.set_alpha(0.0)

    fig.canvas.draw()
    left_fig = fig_coords(fig, left_ax, fit.dst)
    right_fig = fig_coords(fig, right_ax, fit.src * fit.scale)

    left_box = left_ax.get_position()
    right_box = right_ax.get_position()
    gutter_l = left_box.x1
    gutter_r = right_box.x0

    for i in fit.groups["TL"]:
        p0, p1 = left_fig[i], right_fig[i]
        overlay.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=TL_COLOR,
            lw=1.05,
            alpha=0.28,
            solid_capstyle="round",
            zorder=3,
        )

        a = clip_segment_to_x(p0, p1, gutter_l)
        b = clip_segment_to_x(p0, p1, gutter_r)
        overlay.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=TL_COLOR,
            lw=1.45,
            alpha=0.95,
            solid_capstyle="round",
            zorder=4,
        )


# ---------------------------------------------------------------------------
# Zoom inset
# ---------------------------------------------------------------------------


def add_inset(fig: plt.Figure, fit: Fit) -> plt.Axes:
    ax = fig.add_axes(
        [
            INSET_XY[0],
            INSET_XY[1],
            INSET_IN / FIG_W,
            INSET_IN / FIG_H,
        ]
    )
    ax.set_facecolor("white")
    size = fit.n_modules * fit.scale
    ax.imshow(
        fit.rect,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[0.0, size, size, 0.0],
        interpolation="lanczos",
        zorder=1,
    )
    draw_grid(ax, 9, fit.scale, minor_alpha=0.36, major_alpha=0.62)

    idx = fit.groups["TL"]
    ideal = fit.src[idx] * fit.scale
    mapped = project_points(fit.H_inv, fit.dst[idx]) * fit.scale
    exaggerated = ideal + EXAGGERATION * (mapped - ideal)

    # Exaggerated error vectors.  The true mapped point lies on the arrow, one
    # EXAGGERATION-th of the way from the open circle to the filled head.  A
    # white under-stroke keeps the dark arrow visible where it crosses the
    # black finder ring.
    for p, q in zip(ideal, exaggerated):
        ax.add_patch(
            FancyArrowPatch(
                (p[0], p[1]),
                (q[0], q[1]),
                arrowstyle="-|>",
                mutation_scale=14,
                color="#ffffff",
                lw=3.6,
                alpha=0.9,
                shrinkA=4.0,
                shrinkB=0.0,
                zorder=7,
            )
        )
    for p, q in zip(ideal, exaggerated):
        ax.add_patch(
            FancyArrowPatch(
                (p[0], p[1]),
                (q[0], q[1]),
                arrowstyle="-|>",
                mutation_scale=13,
                color=INK,
                lw=1.25,
                alpha=0.95,
                shrinkA=5.0,
                shrinkB=0.0,
                zorder=8,
            )
        )

    # True mapped positions, faint: visible for the corners whose residual is
    # large enough to escape the ideal marker.
    ax.scatter(
        mapped[:, 0],
        mapped[:, 1],
        s=22,
        facecolors=TL_COLOR,
        edgecolors="#ffffff",
        linewidths=0.8,
        alpha=0.9,
        zorder=9,
    )
    ax.scatter(
        exaggerated[:, 0],
        exaggerated[:, 1],
        s=46,
        facecolors=TL_COLOR,
        edgecolors="#ffffff",
        linewidths=1.3,
        zorder=10,
    )
    ax.scatter(
        ideal[:, 0],
        ideal[:, 1],
        s=60,
        facecolors="white",
        edgecolors=INK,
        linewidths=1.4,
        zorder=11,
    )

    ax.set_xlim(-1.35 * fit.scale, 8.35 * fit.scale)
    ax.set_ylim(8.35 * fit.scale, -1.35 * fit.scale)
    ax.set_aspect("equal")
    ax.axis("off")
    rounded_label(
        ax,
        0.014,
        0.986,
        f"top-left finder  ·  residual ×{EXAGGERATION:g}",
        fontsize=10.5,
    )
    return ax


# ---------------------------------------------------------------------------
# Statistics block
# ---------------------------------------------------------------------------


def add_stats(fig: plt.Figure, fit: Fit) -> None:
    ax = fig.add_axes([0.397, 0.018, 0.575, 0.385])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    image_err = fit.image_errors
    mapped = project_points(fit.H_inv, fit.dst)
    rect_err = np.linalg.norm(mapped - fit.src, axis=1) * fit.scale

    ax.text(
        0.0,
        0.95,
        "Twelve-point fit",
        color=INK,
        fontsize=14.5,
        ha="left",
        va="top",
        fontweight="medium",
    )
    ax.text(
        0.0,
        0.785,
        f"mean {image_err.mean():.2f} px   ·   max {image_err.max():.2f} px",
        color=INK,
        fontsize=13.5,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.665,
        "residual in the photo (image pixels)",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.585,
        f"after rectification: {rect_err.mean():.2f} px mean  ·  "
        f"{rect_err.max():.2f} px max",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )

    # Compact marker key.
    y_key = 0.37
    ax.scatter(
        [0.016],
        [y_key],
        s=68,
        facecolors="white",
        edgecolors=INK,
        linewidths=1.4,
        clip_on=False,
    )
    ax.text(0.052, y_key, "ideal canonical corner", color=INK, fontsize=11.0, va="center")
    ax.scatter(
        [0.405],
        [y_key],
        s=38,
        facecolors=TL_COLOR,
        edgecolors="#ffffff",
        linewidths=1.0,
        clip_on=False,
    )
    ax.text(0.435, y_key, "mapped detected corner", color=INK, fontsize=11.0, va="center")
    ax.text(
        0.0,
        0.255,
        f"inset error vectors (arrow + dot) drawn ×{EXAGGERATION:g}",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.185,
        "correspondence lines drawn for the zoomed finder; all twelve corners marked",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )

    # Finder colour key.
    y_col = 0.06
    x = 0.0
    for name, label in (("TL", "top-left"), ("TR", "top-right"), ("BL", "bottom-left")):
        ax.scatter(
            [x],
            [y_col],
            s=46,
            facecolors=finder_color(name),
            edgecolors="#ffffff",
            linewidths=0.9,
            clip_on=False,
        )
        ax.text(x + 0.017, y_col, label, color=MUTED, fontsize=10.2, va="center")
        x += 0.225


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def crop_for(fit: Fit) -> tuple[int, int, int, int]:
    """Square crop around the projected QR, with a little paper margin.

    The bottom edge is chosen just above the hand in the photo; a thin strip of
    the printed footer is allowed to stay, matching the cross-ratio figure.
    """
    corners = compute_qr_corners(fit.H, fit.n_modules)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)

    pad = 0.105 * max(x1 - x0, y1 - y0)
    side = max(x1 - x0, y1 - y0) + 2 * pad
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    c0 = int(round(cx - side / 2))
    c1 = int(round(cx + side / 2))
    r0 = int(round(cy - side / 2))
    r1 = int(round(cy + side / 2))

    # Shift the window up just enough to keep the hand out of frame.
    r1 = min(r1, PHOTO_BOTTOM)
    r0 = r1 - int(round(side))

    c0 = max(0, c0)
    c1 = min(fit.gray.shape[1], c1)
    r0 = max(0, r0)
    r1 = min(fit.gray.shape[0], r1)
    return c0, r0, c1, r1


def right_margins(fit: Fit) -> tuple[float, float, float, float]:
    """White margins around the rectified square, in canvas pixels.

    The vertical fractions roughly match the photo crop, so matching corners
    sit at similar heights and the correspondence lines stay readable.  The
    horizontal margins are deliberately smaller than the photo's: a narrower
    gutter makes the two panels read as one object.
    """
    size = fit.n_modules * fit.scale
    return (0.060 * size, 0.115 * size, 0.080 * size, 0.040 * size)


def build(out_path: Path, data_dir: Path = DATA_DIR) -> Fit:
    fit = detect(data_dir)
    crop = crop_for(fit)
    margins = right_margins(fit)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor="white")
    left_ax = add_photo_panel(fig, fit, crop)
    right_ax = add_rectified_panel(fig, fit, margins)
    add_correspondence(fig, fit, left_ax, right_ax)
    add_inset(fig, fit)
    add_stats(fig, fit)

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fit = build(args.out, args.data_dir)
    image_err = fit.image_errors
    mapped = project_points(fit.H_inv, fit.dst)
    rect_err = np.linalg.norm(mapped - fit.src, axis=1) * fit.scale

    print(f"photo           : {IMAGE_NAME}")
    print(f"version         : {fit.version}")
    print(f"N (modules)     : {fit.n_modules}")
    print(f"fit residual    : mean {image_err.mean():.3f} px, max {image_err.max():.3f} px (image)")
    print(
        f"rectified error : mean {rect_err.mean():.3f} px, max {rect_err.max():.3f} px"
    )
    print(f"exaggeration    : x{EXAGGERATION:g}")
    print(f"wrote           : {args.out}")


if __name__ == "__main__":
    main()
