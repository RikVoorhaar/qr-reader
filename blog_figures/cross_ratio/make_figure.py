#!/usr/bin/env python3
"""Build the cross-ratio version-ranking figure for the QR-reader blog post.

The figure continues the "choosing a triplet" visual the post already uses
(``triplet-colinearity.png``): the same captured photo, the same upright
orientation, the same thin annotations on a real grayscale image.  Instead of
every finder edge, it shows only the four lines that carry the cross-ratio
measurement.

Those four lines are the four sides of the top-left finder, extended:

    line 1  A0-A1-E0-E1   left  side of TL, continues into BL
    line 2  A3-A2-E3-E2   right side of TL, continues into BL
    line 3  A0-A3-C0-C3   top   side of TL, continues into TR
    line 4  A1-A2-C1-C2   bottom side of TL, continues into TR

Each line passes through four known corners — two on the TL finder and two on
the adjacent finder — so the cross-ratio

    r = (u2 - u0)(u3 - u1) / ((u3 - u0)(u2 - u1))

is a projective invariant of that line.  For version 10 (N = 57) all four
lines have the same expected value, because in grid coordinates the four
points always sit at 0, 7, N-7 and N.  Each measured ratio is printed in a
small dark box beside its line, and a box in the top-left corner gives the
value expected for version 10 -- the comparison the version ranking makes.

The landmarks come from the current pipeline (``_prepare_detection`` +
``_rank_triplets``), so the printed numbers are the ones the detector actually
uses — the same numbers as ``pipeline.decode_image`` on this photo.

Usage (from the repository root):

    .venv/bin/python blog_figures/cross_ratio/make_figure.py

The script is deterministic (no random component, Agg backend): running it
twice produces byte-identical PNGs.  It writes

    blog/src/assets/blog/qr-reader/cross-ratio.png
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.detector import (  # noqa: E402
    DetectionResult,
    _prepare_detection,
    _rank_triplets,
)
from qr_reader.detector.landmarks import (  # noqa: E402
    NamedLandmarks,
    build_named_landmarks,
    get_colinear_quadruples,
)
from qr_reader.detector.version import (  # noqa: E402
    expected_cross_ratio,
    measured_cross_ratio,
)

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "cross-ratio.png"

IMAGE_NAME = "qr_code_ - 20.jpeg"
EXPECTED_VERSION = 10

# One colour per line: two warm, two cool, all readable on gray modules.
LINE_COLORS = ["#f5a524", "#ef6a5a", "#4cc9f0", "#a98bf5"]

BOX_BG = "#0d0d0d"
TEXT_BRIGHT = "#f2f2f2"

# Padding around the three finder patterns.  The top pad leaves room for the
# expected-value box; the side and bottom pads keep the finders and a little
# paper margin in frame without dragging in the hand at the bottom.
CROP_PADS = (150, 50, 140, 100)  # top, bottom, left, right
LINE_EXTEND = 46  # px the lines run past their outermost point
# Where along each line its measured value sits (fraction from the first to
# the last point).  The two lines of a pair are staggered so the labels never
# collide.
VALUE_POSITIONS = (0.58, 0.42, 0.58, 0.42)
# Value-box geometry, in rendered pixels (the axes work in image pixels, so
# these are converted with the current figure scale at draw time).
VALUE_BOX_PX = (186.0, 52.0)  # nominal label size, used for the offset
LABEL_GAP_PX = 13.0  # gap between a line and the near edge of its value box


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def crop_bounds(
    landmarks: NamedLandmarks,
    shape: tuple[int, int],
    pads: tuple[int, int, int, int] = CROP_PADS,
):
    """Padded bounding box (r0, r1, c0, c1) around the three finder patterns."""
    pad_t, pad_b, pad_l, pad_r = pads
    pts = np.vstack([landmarks.A, landmarks.C, landmarks.E])
    r0 = max(0, int(np.floor(pts[:, 0].min())) - pad_t)
    r1 = min(shape[0], int(np.ceil(pts[:, 0].max())) + pad_b)
    c0 = max(0, int(np.floor(pts[:, 1].min())) - pad_l)
    c1 = min(shape[1], int(np.ceil(pts[:, 1].max())) + pad_r)
    return r0, r1, c0, c1


def outer_quads(landmarks: NamedLandmarks):
    """The four outer colinear quadruples, in pipeline order."""
    quads = [q for q in get_colinear_quadruples(landmarks) if q.type == "outer"]
    if len(quads) != 4:
        raise RuntimeError(f"expected 4 outer quadruples, got {len(quads)}")
    return quads


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_line(ax, pts_xy: np.ndarray, color: str, extend: float = LINE_EXTEND):
    """Draw a colored line through the four points, slightly over-long."""
    d = pts_xy[-1] - pts_xy[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    a = pts_xy[0] - d * extend
    b = pts_xy[-1] + d * extend

    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color="#000000",
        lw=6.0,
        alpha=0.30,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color=color,
        lw=2.6,
        alpha=0.97,
        solid_capstyle="round",
        zorder=5,
    )
    return d


def draw_value_label(
    ax,
    line_dir: np.ndarray,
    anchor_xy: np.ndarray,
    center_xy: np.ndarray,
    color: str,
    text: str,
    inside: bool,
    px_per_unit: float,
):
    """Put a measured value in a small dark box beside its line.

    ``inside`` lines (the second line of each parallel pair) are labelled on
    the code side, because their outward side is occupied by the other line
    of the pair.
    """
    n = np.array([-line_dir[1], line_dir[0]])
    points_outward = float(np.dot(n, anchor_xy - center_xy)) >= 0
    if inside == points_outward:
        n = -n

    # Offset by half the box extent along the normal so the near edge of the
    # box sits LABEL_GAP_PX pixels off the line.
    w, h = (px / px_per_unit for px in VALUE_BOX_PX)
    offset = 0.5 * (abs(n[0]) * w + abs(n[1]) * h) + LABEL_GAP_PX / px_per_unit
    pos = anchor_xy + n * offset
    ax.text(
        float(pos[0]),
        float(pos[1]),
        text,
        color=color,
        fontsize=12.2,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.40", fc=BOX_BG, ec=color, lw=1.5, alpha=0.94),
        zorder=9,
    )


def draw_expected_box(ax, expected: float, version: int, n_modules: int):
    """Compact reference value in the top-left corner of the photo."""
    ax.text(
        0.016,
        0.984,
        f"version {version}   ·   N = {n_modules}   ·   expected r = {expected:.6f}",
        transform=ax.transAxes,
        color=TEXT_BRIGHT,
        fontsize=11.5,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.55", fc=BOX_BG, ec="#3f3f3f", lw=1.0, alpha=0.94),
        zorder=10,
    )


def render(
    gray_rot: np.ndarray,
    landmarks: NamedLandmarks,
    out_path: Path,
    version: int,
    n_modules: int,
) -> list[float]:
    quads = outer_quads(landmarks)
    expected = expected_cross_ratio(0.0, 7.0, float(n_modules - 7), float(n_modules))

    measured = []
    for q in quads:
        r, _, _ = measured_cross_ratio(q.points)
        measured.append(float(r))

    r0, r1, c0, c1 = crop_bounds(landmarks, gray_rot.shape)
    crop_w, crop_h = c1 - c0, r1 - r0

    fig_w = 8.0
    fig_h = fig_w * crop_h / crop_w
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax.imshow(gray_rot[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")

    center_rc = np.vstack([landmarks.A, landmarks.C, landmarks.E]).mean(axis=0)
    center_xy = center_rc[::-1]

    # Rendered pixels per image pixel, for sizing the annotation boxes.
    ax_x0 = ax.transData.transform((0.0, 0.0))[0]
    ax_x1 = ax.transData.transform((1.0, 0.0))[0]
    px_per_unit = float(ax_x1 - ax_x0)

    # Lines first, so the dots sit on top of them.
    dirs = []
    for k, quad in enumerate(quads):
        pts = quad.points[:, ::-1]
        dirs.append(draw_line(ax, pts, LINE_COLORS[k]))

    # Four corners of the TL finder are shared by two lines: neutral dots.
    ax.scatter(
        landmarks.A[:, 1],
        landmarks.A[:, 0],
        s=72,
        facecolors="#f4f4f4",
        edgecolors="#101010",
        linewidths=1.3,
        zorder=7,
    )

    # The two corners on each adjacent finder carry that line's colour.
    for k, quad in enumerate(quads):
        neighbours = quad.points[2:]
        ax.scatter(
            neighbours[:, 1],
            neighbours[:, 0],
            s=64,
            facecolors=LINE_COLORS[k],
            edgecolors="#ffffff",
            linewidths=1.2,
            zorder=8,
        )

    for k, quad in enumerate(quads):
        pts = quad.points[:, ::-1]
        t = VALUE_POSITIONS[k]
        anchor = pts[0] + t * (pts[-1] - pts[0])
        draw_value_label(
            ax,
            dirs[k],
            anchor,
            center_xy,
            LINE_COLORS[k],
            f"{measured[k]:.6f}",
            inside=k in (1, 3),
            px_per_unit=px_per_unit,
        )

    draw_expected_box(ax, expected, version, n_modules)

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return measured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyse(data_dir: Path = DATA_DIR) -> tuple[np.ndarray, NamedLandmarks, int]:
    img = cv2.imread(str(data_dir / IMAGE_NAME))
    if img is None:
        raise SystemExit(f"could not read {data_dir / IMAGE_NAME}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = DetectionResult(success=False)
    ctx = _prepare_detection(gray, result)
    candidates = _rank_triplets(ctx)
    if not candidates:
        raise SystemExit("no triplet candidate")
    best = candidates[0]
    if best.best_version != EXPECTED_VERSION:
        raise SystemExit(f"unexpected version estimate {best.best_version}")
    tp = best.triplet
    print(
        f"triplet (detected orientation): TL={tp.top_left_idx} "
        f"TR={tp.top_right_idx} BL={tp.bottom_left_idx}"
    )

    # Keep the same upright orientation as triplet-colinearity.png.  The
    # pipeline labels the triplet on this image directly.
    landmarks = build_named_landmarks(tp, ctx.fps)
    return gray, landmarks, EXPECTED_VERSION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gray_rot, landmarks, version = analyse(args.data_dir)
    n_modules = 17 + 4 * version
    measured = render(gray_rot, landmarks, args.out, version, n_modules)

    expected = expected_cross_ratio(0.0, 7.0, float(n_modules - 7), float(n_modules))
    print(f"expected r (v{version}, N={n_modules}) = {expected:.9f}")
    for k, r in enumerate(measured, start=1):
        print(f"  line {k}: measured r = {r:.9f}  |log(r/r10)| = {abs(np.log(r / expected)):.6f}")
    if max(abs(r - expected) for r in measured) > 1e-3:
        raise SystemExit("cross-ratios deviate too far from the v10 expectation")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
