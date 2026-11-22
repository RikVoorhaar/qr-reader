#!/usr/bin/env python3
"""Build the homography figure for the QR-reader blog post.

The figure shows the twelve outer finder corners that the detector fits, and
the mapping from those corners in a captured photo to the canonical corner
positions of a rectified QR grid.

Top-left panel
    A tight crop of the real photo, with the twelve detected outer finder
    corners marked (four per finder).  No grid, no correspondence lines.

Top-right panel
    The same photo warped into a top-down, module-aligned square with the
    inverse homography.  A module grid is overlaid and the twelve canonical
    corners are marked as open circles.  The small filled dots are the detected
    corners mapped through ``H^{-1}``: they should sit on the open circles.
    An amber callout marks the top-left finder.

Bottom-left panel
    A high-resolution, true-scale zoom on the top-left finder pattern.  Open
    amber circles are the ideal corners; filled amber dots are the mapped
    corners; the thin connectors are the actual residuals.  No scaling is
    applied.  An amber callout marks the deeper zoom region.

Bottom-right panel
    A deeper zoom on the bottom-left corner of that finder (grid ``(0, 7)``),
    which has the clearest ideal/mapped separation.  Only that corner is
    shown, with the same open-circle / filled-dot convention.

The image itself carries no statistics or explanatory text, only a compact
two-symbol legend.  The finder colour (amber) does the linking between the
panels, so no correspondence lines are drawn across the gutter.

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
from matplotlib.patches import Polygon, Rectangle

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

# Finder colours.  Three groups only, so the colour-coded corners read as
# three bundles instead of twelve unrelated points.
TL_COLOR = "#f5a524"  # amber
TR_COLOR = "#4cc9f0"  # cyan
BL_COLOR = "#8b5cf6"  # violet

# The bottom panels are both about the top-left finder.  Use a brighter amber
# there so that the connection to the top-left finder is unmistakable.
INSET_COLOR = "#ffb020"

GRID_MINOR = "#9ca3af"
GRID_MAJOR = "#374151"
INK = "#111827"

# The main rectified panel is at RECT_PX_PER_MODULE; the insets are separate,
# higher-resolution renders of the top-left finder and one of its corners.
INSET_PX_PER_MODULE = 100
INSET_HALF_MODULES = 5.0  # 10x10 modules: the 7x7 finder plus a quiet margin

DEEP_PX_PER_MODULE = 280
DEEP_HALF_MODULES = 1.2  # 2.4x2.4 modules around the chosen corner

# Figure layout, in inches / figure fractions.  Two square panels per row.
FIG_W, FIG_H = 10.0, 9.6
TOP_W_IN = 4.15
TOP_H_IN = 4.15
TOP_X = (0.032, 0.525)
TOP_Y = 0.56

BOTTOM_W_IN = 4.15
BOTTOM_H_IN = 4.15
BOTTOM_Y = 0.105


@dataclass
class Fit:
    """Everything the figure needs from one detection run."""

    gray: np.ndarray
    version: int
    n_modules: int
    H: np.ndarray
    H_inv: np.ndarray
    src: np.ndarray  # (12, 2) canonical grid points: 4 per finder
    dst: np.ndarray  # (12, 2) detected image points
    groups: dict[str, np.ndarray]  # finder -> indices into src/dst
    fit_error: float
    rect: np.ndarray
    scale: float
    finder_region: tuple[float, float, float, float]  # x0, y0, x1, y1
    inset_rect: np.ndarray
    inset_scale: float
    deep_region: tuple[float, float, float, float]  # x0, y0, x1, y1
    deep_rect: np.ndarray
    deep_scale: float
    deep_corner_idx: int  # index into src/dst

    @property
    def image_errors(self) -> np.ndarray:
        return np.linalg.norm(
            project_points(self.H, self.src) - self.dst, axis=1
        )


# ---------------------------------------------------------------------------
# Rectification
# ---------------------------------------------------------------------------


def warp_region(
    gray: np.ndarray,
    H_inv: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    scale: float,
) -> np.ndarray:
    """Warp one grid region into a ``scale``-px/module image.

    The grid origin of the output is ``(x0, y0)`` in module coordinates; the
    output size is ``(x1-x0, y1-y0) * scale`` pixels.
    """
    transform = np.array(
        [
            [scale, 0.0, -scale * x0],
            [0.0, scale, -scale * y0],
            [0.0, 0.0, 1.0],
        ]
    )
    width = int(round((x1 - x0) * scale))
    height = int(round((y1 - y0) * scale))
    return cv2.warpPerspective(
        gray,
        transform @ H_inv,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect(data_dir: Path = DATA_DIR) -> Fit:
    """Run the production detector and build the 12-corner correspondence."""
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

    # Rebuild exactly the 12 outer correspondences used by the fit.
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

    # The zoomed finder is the top-left one; the rectified callout covers the
    # same 9x9 module window as the bottom-left panel.
    finder_region = (
        3.5 - INSET_HALF_MODULES,
        3.5 - INSET_HALF_MODULES,
        3.5 + INSET_HALF_MODULES,
        3.5 + INSET_HALF_MODULES,
    )

    # Deeper zoom on the TL corner with the largest rectified residual.  For
    # this photo that is the bottom-left corner, grid (0, 7).
    tl_idx = groups["TL"]
    mapped = project_points(H_inv, dst_arr[tl_idx])
    tl_rect_err = np.linalg.norm(mapped - src_arr[tl_idx], axis=1)
    deep_corner_idx = int(tl_idx[int(np.argmax(tl_rect_err))])
    cx, cy = src_arr[deep_corner_idx]
    deep_region = (
        float(cx - DEEP_HALF_MODULES),
        float(cy - DEEP_HALF_MODULES),
        float(cx + DEEP_HALF_MODULES),
        float(cy + DEEP_HALF_MODULES),
    )

    scale = float(RECT_PX_PER_MODULE)
    rectified = warp_region(gray, H_inv, 0.0, 0.0, float(n), float(n), scale)

    inset_scale = float(INSET_PX_PER_MODULE)
    inset_rect = warp_region(gray, H_inv, *finder_region, inset_scale)

    deep_scale = float(DEEP_PX_PER_MODULE)
    deep_rect = warp_region(gray, H_inv, *deep_region, deep_scale)

    return Fit(
        gray=gray,
        version=version,
        n_modules=n,
        H=H,
        H_inv=H_inv,
        src=src_arr,
        dst=dst_arr,
        groups=groups,
        fit_error=float(fit_error),
        rect=rectified,
        scale=scale,
        finder_region=finder_region,
        inset_rect=inset_rect,
        inset_scale=inset_scale,
        deep_region=deep_region,
        deep_rect=deep_rect,
        deep_scale=deep_scale,
        deep_corner_idx=deep_corner_idx,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def finder_color(name: str) -> str:
    return {"TL": TL_COLOR, "TR": TR_COLOR, "BL": BL_COLOR}[name]


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


def draw_inset_grid(
    ax: plt.Axes,
    n: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    minor_alpha: float,
    major_alpha: float,
):
    """Module grid for the inset, whose data coordinates are modules."""
    minor: list[list[tuple[float, float]]] = []
    major: list[list[tuple[float, float]]] = []
    w, h = x1 - x0, y1 - y0

    for k in range(n + 1):
        if x0 <= k <= x1:
            x = k - x0
            seg = [[x, 0.0], [x, h]]
            (major if k % 7 == 0 else minor).append(seg)
        if y0 <= k <= y1:
            y = k - y0
            seg = [[0.0, y], [w, y]]
            (major if k % 7 == 0 else minor).append(seg)

    ax.add_collection(
        LineCollection(
            minor,
            colors=GRID_MINOR,
            linewidths=0.42,
            alpha=minor_alpha,
            zorder=4,
            antialiased=True,
        )
    )
    ax.add_collection(
        LineCollection(
            major,
            colors=GRID_MAJOR,
            linewidths=1.0,
            alpha=major_alpha,
            zorder=5,
            antialiased=True,
        )
    )


def draw_corner(
    ax: plt.Axes,
    ideal: np.ndarray,
    mapped: np.ndarray,
    *,
    size: float = 34.0,
    dot_size: float = 20.0,
) -> None:
    """One ideal corner (open amber circle) and one mapped dot (filled)."""
    ax.plot(
        [ideal[0], mapped[0]],
        [ideal[1], mapped[1]],
        color="#ffffff",
        lw=2.8,
        alpha=0.95,
        solid_capstyle="round",
        zorder=8,
    )
    ax.plot(
        [ideal[0], mapped[0]],
        [ideal[1], mapped[1]],
        color=INSET_COLOR,
        lw=1.5,
        alpha=1.0,
        solid_capstyle="round",
        zorder=9,
    )
    ax.scatter(
        [mapped[0]],
        [mapped[1]],
        s=dot_size,
        facecolors=INSET_COLOR,
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=10,
    )
    ax.scatter(
        [ideal[0]],
        [ideal[1]],
        s=size,
        facecolors="white",
        edgecolors=INSET_COLOR,
        linewidths=1.7,
        zorder=11,
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


def add_rectified_panel(
    fig: plt.Figure, fit: Fit, margins: tuple[float, float, float, float]
) -> plt.Axes:
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
    # tiny filled dots.  At this scale the two nearly coincide; the bottom
    # panels show the separation.
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

    # Callout around the top-left finder, matching the bottom-left panel.
    ax.add_patch(
        Rectangle(
            (fit.finder_region[0] * fit.scale, fit.finder_region[1] * fit.scale),
            (fit.finder_region[2] - fit.finder_region[0]) * fit.scale,
            (fit.finder_region[3] - fit.finder_region[1]) * fit.scale,
            fill=False,
            edgecolor=TL_COLOR,
            lw=1.25,
            alpha=0.95,
            zorder=12,
        )
    )

    ax.set_xlim(-m_l, size + m_r)
    ax.set_ylim(size + m_b, -m_t)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def add_finder_inset(fig: plt.Figure, fit: Fit) -> plt.Axes:
    """True-scale, high-resolution zoom on the top-left finder pattern."""
    ax = fig.add_axes(
        [
            TOP_X[0],
            BOTTOM_Y,
            BOTTOM_W_IN / FIG_W,
            BOTTOM_H_IN / FIG_H,
        ]
    )
    ax.set_facecolor("white")

    x0, y0, x1, y1 = fit.finder_region
    w_mod = x1 - x0
    h_mod = y1 - y0
    ax.imshow(
        fit.inset_rect,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[0.0, w_mod, h_mod, 0.0],
        interpolation="lanczos",
        zorder=1,
    )
    draw_inset_grid(
        ax,
        fit.n_modules,
        x0,
        y0,
        x1,
        y1,
        minor_alpha=0.34,
        major_alpha=0.55,
    )

    for idx in fit.groups["TL"]:
        ideal = fit.src[idx] - np.array([x0, y0])
        mapped = project_points(fit.H_inv, fit.dst[[idx]])[0] - np.array([x0, y0])
        draw_corner(ax, ideal, mapped, size=34.0, dot_size=20.0)

    # Callout around the region shown in the deeper zoom.
    ax.add_patch(
        Rectangle(
            (fit.deep_region[0] - x0, fit.deep_region[1] - y0),
            fit.deep_region[2] - fit.deep_region[0],
            fit.deep_region[3] - fit.deep_region[1],
            fill=False,
            edgecolor=INSET_COLOR,
            lw=1.5,
            alpha=1.0,
            zorder=12,
        )
    )

    ax.set_xlim(0.0, w_mod)
    ax.set_ylim(h_mod, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def add_deep_inset(fig: plt.Figure, fit: Fit) -> plt.Axes:
    """Deeper zoom on one corner of the top-left finder."""
    ax = fig.add_axes(
        [
            TOP_X[1],
            BOTTOM_Y,
            BOTTOM_W_IN / FIG_W,
            BOTTOM_H_IN / FIG_H,
        ]
    )
    ax.set_facecolor("white")

    x0, y0, x1, y1 = fit.deep_region
    w_mod = x1 - x0
    h_mod = y1 - y0
    ax.imshow(
        fit.deep_rect,
        cmap="gray",
        vmin=0,
        vmax=255,
        extent=[0.0, w_mod, h_mod, 0.0],
        interpolation="lanczos",
        zorder=1,
    )
    draw_inset_grid(
        ax,
        fit.n_modules,
        x0,
        y0,
        x1,
        y1,
        minor_alpha=0.30,
        major_alpha=0.52,
    )

    idx = fit.deep_corner_idx
    ideal = fit.src[idx] - np.array([x0, y0])
    mapped = project_points(fit.H_inv, fit.dst[[idx]])[0] - np.array([x0, y0])
    draw_corner(ax, ideal, mapped, size=46.0, dot_size=28.0)

    ax.set_xlim(0.0, w_mod)
    ax.set_ylim(h_mod, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def add_legend(fig: plt.Figure) -> None:
    """Compact symbol key, centred under the bottom panels."""
    ax = fig.add_axes([0.32, 0.018, 0.36, 0.055])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    y = 0.5
    ax.scatter(
        [0.06],
        [y],
        s=62,
        facecolors="white",
        edgecolors=INSET_COLOR,
        linewidths=1.7,
        clip_on=False,
    )
    ax.text(0.13, y, "ideal corner", color=INK, fontsize=11.5, va="center")

    ax.scatter(
        [0.57],
        [y],
        s=34,
        facecolors=INSET_COLOR,
        edgecolors="#ffffff",
        linewidths=0.9,
        clip_on=False,
    )
    ax.text(0.63, y, "mapped corner", color=INK, fontsize=11.5, va="center")


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
    """White margins around the rectified square, in canvas pixels."""
    size = fit.n_modules * fit.scale
    return (0.060 * size, 0.115 * size, 0.080 * size, 0.040 * size)


def build(out_path: Path, data_dir: Path = DATA_DIR) -> Fit:
    fit = detect(data_dir)
    crop = crop_for(fit)
    margins = right_margins(fit)

    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor="white")
    add_photo_panel(fig, fit, crop)
    add_rectified_panel(fig, fit, margins)
    add_finder_inset(fig, fit)
    add_deep_inset(fig, fit)
    add_legend(fig)

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
    print("correspondences : 12 outer")
    print(
        f"fit             : mean {image_err.mean():.3f} px, max {image_err.max():.3f} px "
        "(image)"
    )
    print(
        f"rectified error : mean {rect_err.mean():.3f} px, max {rect_err.max():.3f} px "
        "(25 px/module)"
    )
    idx = fit.deep_corner_idx
    cx, cy = fit.src[idx]
    delta = project_points(fit.H_inv, fit.dst[[idx]])[0] - fit.src[idx]
    modules = float(np.linalg.norm(delta))
    print(f"zoom corner     : TL finder, max-residual corner ({cx:g}, {cy:g})")
    print(
        f"zoom residual   : {modules:.4f} modules = "
        f"{modules * fit.deep_scale:.2f} px at {fit.deep_scale:g} px/module "
        f"({image_err[idx]:.3f} px in image)"
    )
    print(f"wrote           : {args.out}")


if __name__ == "__main__":
    main()
