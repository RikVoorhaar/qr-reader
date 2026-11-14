#!/usr/bin/env python3
"""Build the homography figure for the QR-reader blog post.

The figure shows the correspondence set that the production pipeline fits:
the twelve outer finder corners plus the six interior alignment-pattern
centres (versions >= 2), eighteen points in total.

Left panel
    A tight crop of the real photo, with the twelve outer finder corners in
    their finder colours and the six alignment centres as green diamonds.

Right panel
    The same photo warped into a top-down, module-aligned square with the
    inverse *refined* homography.  The twelve canonical outer corners are open
    circles; the six canonical alignment centres are open green diamonds.
    Tiny filled dots are the detected points mapped through ``H^{-1}``.

Correspondence lines
    Only the six alignment centres (green) and the four top-left outer corners
    (amber) are joined across the gutter.  The alignment lines are the point:
    they show the new constraints spread across the symbol.

Bottom inset
    A high-resolution, true-scale zoom on the alignment pattern with the
    largest alignment residual, centred at (28.5, 28.5).  The open circle is
    the ideal canonical centre; the filled dot is the matched image point; the
    thin connector is the actual residual.  No scaling is applied.  A green
    callout rectangle in the rectified panel marks the zoomed pattern.

The production pipeline used here is the real one:

    _prepare_detection -> _rank_triplets -> _fit_candidate_homography
        -> build_refined_homography

``H`` maps QR-grid (x, y) to image (x, y).  The outer source points are the
same ones ``_fit_candidate_homography`` builds, in [TL, TR, BR, BL] order per
finder.  The six alignment centres come from ``alignment_centers_grid(10)``
and are matched with ``match_alignment_pattern`` exactly as the refit does.

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
from qr_reader.detector.landmarks_refined import (  # noqa: E402
    _local_module_px,
    alignment_centers_grid,
    build_refined_homography,
    match_alignment_pattern,
)

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
ALIGN_COLOR = "#2f9e44"  # green

GRID_MINOR = "#9ca3af"
GRID_MAJOR = "#374151"
INK = "#111827"
MUTED = "#6b7280"

BOX_BG = "#0d0d0d"
BOX_EDGE = "#3f3f3f"
TEXT_BRIGHT = "#f2f2f2"

# The main rectified panel is at RECT_PX_PER_MODULE; the inset is a separate,
# higher-resolution render of the chosen alignment pattern (5x5) plus a
# one-module context margin.
INSET_PX_PER_MODULE = 160
INSET_HALF_MODULES = 2.5  # exactly the 5x5 alignment pattern
INSET_FOOTER_MODULES = 0.55
# Figure layout, in inches / figure fractions.
FIG_W, FIG_H = 10.0, 9.2
TOP_W_IN = 4.15
TOP_H_IN = 4.15
TOP_X = (0.032, 0.525)
TOP_Y = 0.535

INSET_W_IN = 4.0
INSET_H_IN = INSET_W_IN * (2 * INSET_HALF_MODULES + INSET_FOOTER_MODULES) / (2 * INSET_HALF_MODULES)
INSET_XY = (0.03, 0.02)


@dataclass
class Fit:
    """Everything the figure needs from one detection run."""

    gray: np.ndarray
    version: int
    n_modules: int
    H: np.ndarray
    H_inv: np.ndarray
    src: np.ndarray  # (18, 2) canonical grid points: 12 outer + 6 alignment
    dst: np.ndarray  # (18, 2) detected/matched image points
    groups: dict[str, np.ndarray]  # finder -> indices into src/dst (0..11)
    align_idx: np.ndarray  # indices 12..17 in src/dst
    align_conf: np.ndarray  # template-match confidences for the 6 centres
    fit_error: float
    base_error: float
    ref_stats: dict
    rect: np.ndarray
    scale: float
    inset_rect: np.ndarray  # high-resolution alignment-pattern zoom
    inset_scale: float
    inset_region: tuple[float, float, float, float]  # x0, y0, x1, y1
    inset_center_idx: int  # index into align_idx

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
    """Run the production detector and build the 12 + 6 correspondence."""
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
    H_base, base_error = _fit_candidate_homography(ctx, candidate.triplet, n)
    H, ref_stats = build_refined_homography(
        ctx.img_gray, H_base, candidate.triplet, ctx.fps, version
    )
    if ref_stats["n_outer"] != 12 or ref_stats["n_align"] != 6:
        raise SystemExit(f"unexpected refined correspondence counts: {ref_stats}")

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

    n_outer = len(src)

    # Match the six interior alignment centres exactly as build_refined_homography
    # does: seed with H_base, use the local module pitch from H_base.
    align_conf: list[float] = []
    for gx, gy in alignment_centers_grid(version):
        center0 = project_points(H_base, np.array([[gx, gy]]))[0]
        module_px = _local_module_px(H_base, gx, gy)
        image_xy, confidence = match_alignment_pattern(
            ctx.img_gray, center0, module_px
        )
        src.append([gx, gy])
        dst.append(image_xy.tolist())
        align_conf.append(confidence)

    src_arr = np.array(src, dtype=np.float64)
    dst_arr = np.array(dst, dtype=np.float64)
    align_idx = np.arange(n_outer, len(src_arr))

    proj = project_points(H, src_arr)
    recomputed = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
    if abs(recomputed - ref_stats["err"]) > 1e-9:
        raise SystemExit(
            f"recomputed refined error {recomputed} != returned {ref_stats['err']}"
        )

    # The alignment pattern with the largest image-plane residual becomes the
    # inset subject.  For v10 that is the centre (28.5, 28.5).
    align_err = np.linalg.norm(proj[align_idx] - dst_arr[align_idx], axis=1)
    inset_center_idx = int(align_idx[int(np.argmax(align_err))])
    cx, cy = src_arr[inset_center_idx]
    inset_region = (
        float(cx - INSET_HALF_MODULES),
        float(cy - INSET_HALF_MODULES),
        float(cx + INSET_HALF_MODULES),
        float(cy + INSET_HALF_MODULES),
    )

    H_inv = np.linalg.inv(H)
    scale = float(RECT_PX_PER_MODULE)
    rectified = warp_region(gray, H_inv, 0.0, 0.0, float(n), float(n), scale)
    inset_scale = float(INSET_PX_PER_MODULE)
    inset_rect = warp_region(
        gray,
        H_inv,
        *inset_region,
        inset_scale,
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
        align_idx=align_idx,
        align_conf=np.array(align_conf, dtype=np.float64),
        fit_error=float(ref_stats["err"]),
        base_error=float(base_error),
        ref_stats=ref_stats,
        rect=rectified,
        scale=scale,
        inset_rect=inset_rect,
        inset_scale=inset_scale,
        inset_region=inset_region,
        inset_center_idx=inset_center_idx,
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

    # The six matched alignment centres, plus a green 5x5 outline around
    # each pattern so the new landmarks are unmistakable.
    for i in fit.align_idx:
        gx, gy = fit.src[i]
        corners = np.array(
            [
                [gx - 2.5, gy - 2.5],
                [gx + 2.5, gy - 2.5],
                [gx + 2.5, gy + 2.5],
                [gx - 2.5, gy + 2.5],
            ]
        )
        image_corners = project_points(fit.H, corners)
        ax.add_patch(
            Polygon(
                image_corners,
                closed=True,
                fill=False,
                edgecolor=ALIGN_COLOR,
                lw=1.15,
                alpha=0.80,
                zorder=8,
            )
        )

    apts = fit.dst[fit.align_idx]
    ax.scatter(
        apts[:, 0],
        apts[:, 1],
        marker="D",
        s=38,
        facecolors=ALIGN_COLOR,
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=10,
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

    # Canonical alignment centres: open green diamonds; matched centres are
    # tiny green dots inside them.
    align_grid = fit.src[fit.align_idx] * fit.scale
    align_mapped = mapped[fit.align_idx]
    ax.scatter(
        align_grid[:, 0],
        align_grid[:, 1],
        marker="D",
        s=46,
        facecolors="white",
        edgecolors=ALIGN_COLOR,
        linewidths=1.5,
        zorder=9,
    )
    ax.scatter(
        align_mapped[:, 0],
        align_mapped[:, 1],
        s=13,
        facecolors=ALIGN_COLOR,
        edgecolors="#ffffff",
        linewidths=0.5,
        zorder=10,
    )

    # Green 5x5 outlines around the alignment patterns.  The chosen one gets
    # the thicker callout below instead, so skip it here to avoid a double line.
    for i in fit.align_idx:
        if i == fit.inset_center_idx:
            continue
        gx, gy = fit.src[i]
        corners = np.array(
            [
                [gx - 2.5, gy - 2.5],
                [gx + 2.5, gy - 2.5],
                [gx + 2.5, gy + 2.5],
                [gx - 2.5, gy + 2.5],
            ]
        ) * fit.scale
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=ALIGN_COLOR,
                lw=1.0,
                alpha=0.72,
                zorder=8,
            )
        )

    # Callout around the corner region shown in the bottom inset.
    ax.add_patch(
        Rectangle(
            (fit.inset_region[0] * fit.scale, fit.inset_region[1] * fit.scale),
            (fit.inset_region[2] - fit.inset_region[0]) * fit.scale,
            (fit.inset_region[3] - fit.inset_region[1]) * fit.scale,
            fill=False,
            edgecolor=ALIGN_COLOR,
            lw=1.25,
            alpha=0.95,
            zorder=12,
        )
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
    """Thin colour-coded bridges for the alignment centres and TL outer corners.

    Eighteen lines across two dense panels read as spaghetti.  The new
    alignment constraints are the point, so all six are drawn (green); the
    four top-left finder corners (amber) tie the inset colour story together.
    The other eight outer corners keep their colour-coded dots and open
    circles.
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

    lines: list[tuple[np.ndarray, str, float, float, float, float]] = []
    for i in fit.groups["TL"]:
        lines.append((np.array([i]), TL_COLOR, 1.05, 1.45, 0.28, 0.95))
    for i in fit.align_idx:
        lines.append((np.array([i]), ALIGN_COLOR, 1.15, 1.40, 0.34, 0.92))

    for indices, color, in_lw, gap_lw, in_alpha, gap_alpha in lines:
        for i in indices:
            p0, p1 = left_fig[i], right_fig[i]
            overlay.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                lw=in_lw,
                alpha=in_alpha,
                solid_capstyle="round",
                zorder=3,
            )

            a = clip_segment_to_x(p0, p1, gutter_l)
            b = clip_segment_to_x(p0, p1, gutter_r)
            overlay.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=color,
                lw=gap_lw,
                alpha=gap_alpha,
                solid_capstyle="round",
                zorder=4,
            )


# ---------------------------------------------------------------------------
# Zoom inset
# ---------------------------------------------------------------------------


def add_inset(fig: plt.Figure, fit: Fit) -> plt.Axes:
    """True-scale, high-resolution zoom on the chosen alignment pattern."""
    ax = fig.add_axes(
        [
            INSET_XY[0],
            INSET_XY[1],
            INSET_W_IN / FIG_W,
            INSET_H_IN / FIG_H,
        ]
    )
    ax.set_facecolor("white")

    x0, y0, x1, y1 = fit.inset_region
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

    idx = fit.inset_center_idx
    ideal = fit.src[idx] - np.array([x0, y0])
    mapped = project_points(fit.H_inv, fit.dst[[idx]])[0] - np.array([x0, y0])

    # True-length connector; white under-stroke keeps it visible over the
    # dark alignment-pattern centre.
    ax.plot(
        [ideal[0], mapped[0]],
        [ideal[1], mapped[1]],
        color="#ffffff",
        lw=2.4,
        alpha=0.9,
        solid_capstyle="round",
        zorder=8,
    )
    ax.plot(
        [ideal[0], mapped[0]],
        [ideal[1], mapped[1]],
        color=ALIGN_COLOR,
        lw=1.2,
        alpha=0.98,
        solid_capstyle="round",
        zorder=9,
    )

    # Matched centre: filled dot.  Ideal centre: open circle.
    ax.scatter(
        [mapped[0]],
        [mapped[1]],
        s=12,
        facecolors=ALIGN_COLOR,
        edgecolors="#ffffff",
        linewidths=0.9,
        zorder=10,
    )
    ax.scatter(
        [ideal[0]],
        [ideal[1]],
        s=20,
        facecolors="white",
        edgecolors=INK,
        linewidths=1.3,
        zorder=11,
    )

    # The scale bar and label live in a white footer band below the pattern,
    # so neither covers the 5x5 alignment pattern.
    footer_y = h_mod + 0.5 * INSET_FOOTER_MODULES
    bar_x0, bar_x1 = 3.55, 4.55
    ax.plot(
        [bar_x0, bar_x1],
        [footer_y, footer_y],
        color=INK,
        lw=1.5,
        solid_capstyle="butt",
        zorder=12,
    )
    for x in (bar_x0, bar_x1):
        ax.plot(
            [x, x],
            [footer_y - 0.12, footer_y + 0.12],
            color=INK,
            lw=1.2,
            zorder=12,
        )
    ax.text(
        0.5 * (bar_x0 + bar_x1),
        footer_y + 0.07,
        "1 module",
        color=INK,
        fontsize=8.6,
        ha="center",
        va="top",
        zorder=12,
    )
    ax.text(
        0.10,
        footer_y,
        "alignment pattern  ·  centre",
        color=INK,
        fontsize=10.5,
        ha="left",
        va="center",
        zorder=12,
    )

    ax.set_xlim(0.0, w_mod)
    ax.set_ylim(h_mod + INSET_FOOTER_MODULES, 0.0)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ---------------------------------------------------------------------------
# Statistics block
# ---------------------------------------------------------------------------


def add_stats(fig: plt.Figure, fit: Fit) -> None:
    ax = fig.add_axes([0.51, 0.03, 0.46, 0.47])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    image_err = fit.image_errors
    mapped = project_points(fit.H_inv, fit.dst)
    rect_err = np.linalg.norm(mapped - fit.src, axis=1) * fit.scale

    align_img_err = image_err[fit.align_idx]
    idx = int(fit.align_idx[int(np.argmax(align_img_err))])
    cx, cy = fit.src[idx]
    chosen_delta = project_points(fit.H_inv, fit.dst[[idx]])[0] - fit.src[idx]
    chosen_modules = float(np.linalg.norm(chosen_delta))
    chosen_inset_px = chosen_modules * fit.inset_scale
    chosen_img_px = float(image_err[idx])

    ax.text(
        0.0,
        0.95,
        "18 correspondences",
        color=INK,
        fontsize=14.5,
        ha="left",
        va="top",
        fontweight="medium",
    )
    ax.text(
        0.0,
        0.865,
        "12 outer finder corners  +  6 alignment centres",
        color=MUTED,
        fontsize=10.8,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.755,
        f"refined mean {image_err.mean():.2f} px   ·   max {image_err.max():.2f} px",
        color=INK,
        fontsize=13.5,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.655,
        "residual in the photo (image pixels) · true scale",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.59,
        f"rectified (25 px/module): {rect_err.mean():.2f} px mean  ·  "
        f"{rect_err.max():.2f} px max",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )

    # Compact marker key, two rows.
    y_key = 0.475
    ax.scatter(
        [0.016],
        [y_key],
        s=62,
        facecolors="white",
        edgecolors=INK,
        linewidths=1.4,
        clip_on=False,
    )
    ax.text(0.05, y_key, "ideal outer corner", color=INK, fontsize=10.8, va="center")
    ax.scatter(
        [0.405],
        [y_key],
        s=36,
        facecolors=TL_COLOR,
        edgecolors="#ffffff",
        linewidths=1.0,
        clip_on=False,
    )
    ax.text(0.435, y_key, "mapped detected", color=INK, fontsize=10.8, va="center")

    y_align = 0.395
    ax.scatter(
        [0.016],
        [y_align],
        marker="D",
        s=48,
        facecolors="white",
        edgecolors=ALIGN_COLOR,
        linewidths=1.5,
        clip_on=False,
    )
    ax.text(
        0.05,
        y_align,
        "alignment centre (6)",
        color=INK,
        fontsize=10.8,
        va="center",
    )
    ax.scatter(
        [0.405],
        [y_align],
        s=30,
        facecolors=ALIGN_COLOR,
        edgecolors="#ffffff",
        linewidths=0.9,
        clip_on=False,
    )
    ax.text(
        0.435,
        y_align,
        "mapped alignment",
        color=INK,
        fontsize=10.8,
        va="center",
    )

    ax.text(
        0.0,
        0.30,
        f"inset: true scale · {fit.inset_scale:g} px/module",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.235,
        f"chosen alignment centre ({cx:g}, {cy:g})",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.18,
        f"{chosen_modules:.3f} modules = {chosen_inset_px:.1f} px at this scale"
        f"  ·  {chosen_img_px:.2f} px in the photo",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )
    ax.text(
        0.0,
        0.11,
        "lines: 6 alignment centres + 4 TL outer corners; all 18 marked",
        color=MUTED,
        fontsize=10.6,
        ha="left",
        va="top",
    )

    # Finder colour key plus the alignment colour.
    y_col = 0.035
    x = 0.0
    for name, label in (
        ("TL", "top-left"),
        ("TR", "top-right"),
        ("BL", "bottom-left"),
    ):
        ax.scatter(
            [x],
            [y_col],
            s=40,
            facecolors=finder_color(name),
            edgecolors="#ffffff",
            linewidths=0.8,
            clip_on=False,
        )
        ax.text(x + 0.016, y_col, label, color=MUTED, fontsize=9.4, va="center")
        x += 0.235
    ax.scatter(
        [x],
        [y_col],
        marker="D",
        s=34,
        facecolors=ALIGN_COLOR,
        edgecolors="#ffffff",
        linewidths=0.8,
        clip_on=False,
    )
    ax.text(x + 0.016, y_col, "alignment", color=MUTED, fontsize=9.4, va="center")


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
    print(
        f"correspondences : {fit.ref_stats['n_outer']} outer + "
        f"{fit.ref_stats['n_align']} alignment = {fit.ref_stats['n_corr']}"
    )
    print(
        f"refined fit     : mean {image_err.mean():.3f} px, max {image_err.max():.3f} px "
        "(image)"
    )
    print(f"base fit        : mean {fit.base_error:.3f} px (12 outer only)")
    print(
        f"rectified error : mean {rect_err.mean():.3f} px, max {rect_err.max():.3f} px "
        "(25 px/module)"
    )
    align_err = image_err[fit.align_idx]
    print(
        "alignment errs  : "
        + ", ".join(
            f"({fit.src[i][0]:g},{fit.src[i][1]:g})={e:.3f}"
            for i, e in zip(fit.align_idx, align_err)
        )
    )
    idx = fit.inset_center_idx
    cx, cy = fit.src[idx]
    delta = project_points(fit.H_inv, fit.dst[[idx]])[0] - fit.src[idx]
    modules = float(np.linalg.norm(delta))
    print(f"zoom alignment  : centre ({cx:g}, {cy:g}), grid {fit.src[idx].tolist()}")
    print(
        f"zoom residual   : {modules:.4f} modules = "
        f"{modules * fit.inset_scale:.2f} px at {fit.inset_scale:g} px/module "
        f"({image_err[idx]:.3f} px in image)"
    )
    print(f"wrote           : {args.out}")


if __name__ == "__main__":
    main()
