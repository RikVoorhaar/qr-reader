#!/usr/bin/env python3
"""Build the RLE-normalization figure for the QR-reader blog post.

The figure illustrates the paragraph that turns a measured 1:1:3:1:1 run-length
tuple into a normalized tuple and checks it against the ideal within 30%.

It uses the same captured photo as ``rle-scan.gif``:

    data/bench/captured-fullsize/qr_code_ - 99.jpeg

and measures a real full-resolution image row.  The crop is the wide 2:1 crop
around the two upper finder patterns; the scanline is row 2591, which passes
through the centre of the left finder.  The right finder sits about 126 px
higher in the photo (perspective), so at row 2591 its five-run cut is
off-centre and fails the 30% test.  The figure therefore works the left
finder's tuple and adds the right finder's own centre-row tuple (row 2467) as
a compact second row.

Output:

    blog/src/assets/blog/qr-reader/rle-normalization.png

Run from the repository root:

    .venv/bin/python blog_figures/rle_normalization/make_figure.py

The script is deterministic: the photo, crop, Otsu threshold, row, clustering
seed and rendering are all fixed.  It writes only the PNG.
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
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.detector.edge_fitting import (  # noqa: E402
    DISTANCE_THRESHOLD,
    MAX_GAP,
)
from qr_reader.detector.ray_fit import fit_finder_ray  # noqa: E402
from qr_reader.detector.roi import cluster_to_bbox, cutout  # noqa: E402
from qr_reader.qr_gen import binarize_image  # noqa: E402

# ── Figure configuration ─────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
IMAGE_NAME = "qr_code_ - 99.jpeg"
OUTPUT_PATH = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "rle-normalization.png"
)

# Full-resolution crop: 1675 x 838 px, exactly 2:1.  The two upper finder
# patterns sit symmetrically inside it: about 260 px of horizontal margin and
# about 165 px of vertical margin above and below.
CROP_X0, CROP_X1 = 677, 2352
CROP_Y0, CROP_Y1 = 2110, 2948

# The real image row used for the RLE strip.  It passes through the centre of
# the left finder; the right finder is about 126 px higher in the photo.
ROW = 2591

# The right finder's own centre row, used for the compact second tuple.
RIGHT_OWN_ROW = 2467

MAX_ERROR = float(np.log(1.3))  # detector tolerance: each run within 30%

# ── Palette ──────────────────────────────────────────────────────────────────

INK = "#1c2136"  # dark text / binary black
MUTED = "#6b7280"  # secondary text
RULE = "#e5e7eb"  # thin separators
BINARY_DARK = "#1c2136"
BINARY_LIGHT = "#f6f1da"
ORANGE = "#ff7800"  # finder tuples / candidate accent
CYAN = "#00dcff"  # scanline
GREEN = "#00a86b"  # pass mark
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


# ── Measurements ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Run:
    """One run of a binary row, in full-resolution image coordinates."""

    is_white: bool
    length: int
    x0: int
    x1: int


@dataclass(frozen=True)
class FiveTuple:
    """Five consecutive runs and their normalization against 1:1:3:1:1."""

    start_index: int
    lengths: np.ndarray
    x0: int
    x1: int
    score: float

    @property
    def total(self) -> int:
        return int(self.lengths.sum())

    @property
    def normalized(self) -> np.ndarray:
        return self.lengths / self.total * 7.0

    @property
    def errors(self) -> np.ndarray:
        return self.normalized / np.array([1.0, 1.0, 3.0, 1.0, 1.0]) - 1.0

    @property
    def passes(self) -> bool:
        return bool(self.score < MAX_ERROR)


def _rle_row(binary: np.ndarray, row: int, x0: int, x1: int) -> list[Run]:
    """Run-length encode one real image row between ``x0`` and ``x1``."""
    segment = binary[row, x0:x1]
    transitions = np.where(np.diff(segment.astype(np.int8)) != 0)[0] + 1
    bounds = np.concatenate([[0], transitions, [len(segment)]])
    lengths = np.diff(bounds)
    values = segment[bounds[:-1]]
    return [
        Run(
            is_white=bool(value),
            length=int(length),
            x0=x0 + int(start),
            x1=x0 + int(end),
        )
        for value, length, start, end in zip(values, lengths, bounds[:-1], bounds[1:])
    ]


def _best_five(runs: list[Run], center_x: float, window: float = 150.0) -> FiveTuple:
    """Best 1:1:3:1:1-scoring five-run group near a finder centre.

    This mirrors the detector's score (log-ratio distance to the ideal) but
    keeps the best group even when it fails, so the figure can show the
    off-centre right-finder cut at the chosen row.
    """
    ideal = np.array([1.0, 1.0, 3.0, 1.0, 1.0])
    log_ideal = np.log(ideal / ideal.sum())
    best: FiveTuple | None = None
    for i in range(len(runs) - 4):
        group = runs[i : i + 5]
        center = (group[0].x0 + group[-1].x1) / 2.0
        if abs(center - center_x) > window:
            continue
        lengths = np.array([run.length for run in group], dtype=float)
        score = float(
            np.max(np.abs(np.log(lengths / lengths.sum()) - log_ideal))
        )
        candidate = FiveTuple(
            start_index=i,
            lengths=lengths.astype(int),
            x0=group[0].x0,
            x1=group[-1].x1,
            score=score,
        )
        if best is None or candidate.score < best.score:
            best = candidate
    if best is None:
        raise RuntimeError(f"no five-run group near x={center_x}")
    return best


def _finder_quads(gray: np.ndarray) -> list[np.ndarray]:
    """Ray-fit the finder patterns and return their corner quads in (x, y)."""
    binary = binarize_image(gray)
    rows, cols = find_alignment_patterns_2d(binary, MAX_ERROR)
    clusters = cluster_candidates(rows, cols)

    quads: list[np.ndarray] = []
    for cluster in clusters:
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0:
            continue
        r0 = max(0, int(bbox[0]))
        c0 = max(0, int(bbox[2]))
        center = np.array(
            [
                float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0,
                float(cluster.row) - r0,
            ]
        )
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0
        result = fit_finder_ray(
            roi,
            center,
            m_est,
            max_gap=MAX_GAP,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        if result.valid:
            quads.append(result.corners + np.array([c0, r0], dtype=float))
    return quads


@dataclass(frozen=True)
class Measurements:
    rgb: np.ndarray
    left_quad: np.ndarray
    right_quad: np.ndarray
    runs: list[Run]
    left: FiveTuple
    right_at_row: FiveTuple
    right_own: FiveTuple


def measure() -> Measurements:
    path = DATA_DIR / IMAGE_NAME
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary = binarize_image(gray)

    quads = _finder_quads(gray)
    if len(quads) < 3:
        raise RuntimeError(f"expected three finder quads, found {len(quads)}")
    # The two upper finders are the two with the smallest y centres; sort them
    # left-to-right for the figure.
    upper = sorted(quads, key=lambda q: float(q[:, 1].mean()))[:2]
    upper.sort(key=lambda q: float(q[:, 0].mean()))
    left_quad, right_quad = upper

    runs = _rle_row(binary, ROW, CROP_X0, CROP_X1)
    left = _best_five(runs, float(left_quad[:, 0].mean()))
    right_at_row = _best_five(runs, float(right_quad[:, 0].mean()))

    own_runs = _rle_row(binary, RIGHT_OWN_ROW, CROP_X0, CROP_X1)
    right_own = _best_five(own_runs, float(right_quad[:, 0].mean()))

    return Measurements(
        rgb=rgb,
        left_quad=left_quad,
        right_quad=right_quad,
        runs=runs,
        left=left,
        right_at_row=right_at_row,
        right_own=right_own,
    )


# ── Drawing ──────────────────────────────────────────────────────────────────


def _draw_photo(ax, data: Measurements) -> None:
    ax.imshow(
        data.rgb[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1],
        extent=[CROP_X0, CROP_X1, CROP_Y1, CROP_Y0],
        interpolation="lanczos",
    )

    for quad in (data.left_quad, data.right_quad):
        closed = np.vstack([quad, quad[0]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#101010",
            lw=4.2,
            alpha=0.55,
            solid_joinstyle="round",
            zorder=3,
        )
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color=ORANGE,
            lw=1.8,
            solid_joinstyle="round",
            zorder=4,
        )

    # Scanline: dark halo first, then bright cyan.
    ax.plot(
        [CROP_X0, CROP_X1],
        [ROW, ROW],
        color="#101010",
        lw=5.5,
        alpha=0.55,
        zorder=5,
    )
    ax.plot(
        [CROP_X0, CROP_X1],
        [ROW, ROW],
        color=CYAN,
        lw=2.4,
        zorder=6,
    )

    ax.text(
        CROP_X0 + 14,
        ROW - 26,
        f"row {ROW}",
        color=CYAN,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.28", fc="#101010", ec="none", alpha=0.88),
    )

    ax.set_xlim(CROP_X0, CROP_X1)
    ax.set_ylim(CROP_Y1, CROP_Y0)
    ax.axis("off")


def _draw_strip(ax, data: Measurements) -> None:
    for run in data.runs:
        ax.add_patch(
            Rectangle(
                (run.x0, 0.30),
                run.x1 - run.x0,
                0.50,
                facecolor=BINARY_LIGHT if run.is_white else BINARY_DARK,
                edgecolor="#d8d2b8" if run.is_white else BINARY_DARK,
                linewidth=0.5,
                zorder=2,
            )
        )

    for group, label in (
        (data.left, "left finder"),
        (data.right_at_row, "right finder"),
    ):
        # Thin rule above the group and a bracket below it.
        ax.plot(
            [group.x0, group.x1],
            [0.84, 0.84],
            color=ORANGE,
            lw=1.4,
            zorder=4,
        )
        for x in (group.x0, group.x1):
            ax.plot([x, x], [0.84, 0.88], color=ORANGE, lw=1.4, zorder=4)
        ax.plot(
            [group.x0, group.x1],
            [0.20, 0.20],
            color=ORANGE,
            lw=1.4,
            zorder=4,
        )
        for x in (group.x0, group.x1):
            ax.plot([x, x], [0.20, 0.27], color=ORANGE, lw=1.4, zorder=4)
        ax.text(
            (group.x0 + group.x1) / 2.0,
            0.02,
            label,
            color=ORANGE,
            fontsize=9.0,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=5,
        )

    ax.set_xlim(CROP_X0, CROP_X1)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")


def _packed_line(segments: list[tuple[str, str, bool]], fontsize: float):
    """Pack coloured text segments into one horizontal line."""
    children = [
        TextArea(
            text,
            textprops=dict(
                fontfamily="DejaVu Sans Mono",
                fontsize=fontsize,
                color=color,
                fontweight="bold" if bold else "normal",
            ),
        )
        for text, color, bold in segments
    ]
    return HPacker(children=children, align="center", pad=0, sep=0)


def _draw_rle_list(ax, data: Measurements) -> None:
    runs = data.runs
    left = data.left
    right = data.right_at_row

    def number_list(start: int, stop: int) -> str:
        return ", ".join(str(run.length) for run in runs[start:stop])

    line1 = [
        (number_list(0, left.start_index) + ", ", INK, False),
        (
            "["
            + ", ".join(str(v) for v in left.lengths)
            + "]",
            ORANGE,
            True,
        ),
        (", " + number_list(left.start_index + 5, right.start_index) + ", ", INK, False),
    ]
    line2 = [
        (
            "["
            + ", ".join(str(v) for v in right.lengths)
            + "]",
            ORANGE,
            True,
        ),
        (", " + number_list(right.start_index + 5, len(runs)), INK, False),
    ]

    for y, segments in ((0.68, line1), (0.24, line2)):
        box = AnnotationBbox(
            _packed_line(segments, fontsize=10.5),
            (0.5, y),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0.5, 0.5),
        )
        ax.add_artist(box)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_normalization(ax, data: Measurements) -> None:
    left = data.left
    right = data.right_own

    label_x = 0.0
    col_x = [0.20, 0.35, 0.50, 0.65, 0.80]
    right_x = 0.995

    ax.text(
        label_x,
        0.955,
        "normalize",
        fontsize=15,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    ax.text(
        label_x,
        0.885,
        f"left finder · row {ROW}",
        fontsize=10.5,
        color=MUTED,
        ha="left",
        va="top",
    )

    # Raw lengths.
    ax.text(label_x, 0.785, "raw px", fontsize=10.5, color=MUTED, ha="left", va="center")
    for x, value in zip(col_x, left.lengths):
        ax.text(
            x,
            0.785,
            str(int(value)),
            fontsize=14,
            color=INK,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
        )
    ax.text(
        right_x,
        0.785,
        f"S = {left.total}",
        fontsize=12,
        fontweight="bold",
        color=INK,
        ha="right",
        va="center",
    )

    # Explicit sum.
    ax.text(
        label_x,
        0.705,
        "S = "
        + " + ".join(str(int(v)) for v in left.lengths)
        + f" = {left.total}",
        fontsize=10.5,
        color=INK,
        ha="left",
        va="center",
        fontfamily="DejaVu Sans Mono",
    )

    # x / S * 7, then the result.
    ax.text(
        label_x, 0.585, "x / S × 7", fontsize=10.5, color=MUTED, ha="left", va="center"
    )
    for x, value in zip(col_x, left.lengths):
        ax.text(
            x,
            0.585,
            f"{int(value)}/{left.total}×7",
            fontsize=10.5,
            color=INK,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
        )
    for x, value in zip(col_x, left.normalized):
        ax.text(
            x,
            0.505,
            f"= {value:.2f}",
            fontsize=11,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
        )

    # Ideal and error.
    ax.text(label_x, 0.385, "ideal", fontsize=10.5, color=MUTED, ha="left", va="center")
    for x, value in zip(col_x, (1, 1, 3, 1, 1)):
        ax.text(
            x,
            0.385,
            str(value),
            fontsize=13,
            color=INK,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
        )

    ax.text(label_x, 0.285, "error", fontsize=10.5, color=MUTED, ha="left", va="center")
    for x, value in zip(col_x, left.errors):
        ax.text(
            x,
            0.285,
            f"{value * 100:+.0f}%",
            fontsize=10.5,
            color=INK,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
        )
    ax.text(
        right_x,
        0.285,
        "all within 30%  ✓",
        fontsize=11,
        fontweight="bold",
        color=GREEN,
        ha="right",
        va="center",
    )

    # Separator and the second finder's own-centre-row tuple.
    ax.plot([0.0, 1.0], [0.205, 0.205], color=RULE, lw=1.0)
    ax.text(
        label_x,
        0.135,
        (
            f"right finder at its own centre row {RIGHT_OWN_ROW}:  "
            + ", ".join(str(int(v)) for v in right.lengths)
            + f"  (S = {right.total})  →  "
            + ", ".join(f"{v:.2f}" for v in right.normalized)
            + "  ✓"
        ),
        fontsize=10.0,
        color=MUTED,
        ha="left",
        va="center",
    )
    ax.text(
        label_x,
        0.065,
        f"at row {ROW} the scanline is off-centre for the right finder, so its 5-tuple is distorted",
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ── Main ─────────────────────────────────────────────────────────────────────


def build_figure(data: Measurements) -> plt.Figure:
    fig = plt.figure(figsize=(9.0, 10.0), dpi=200)

    photo_ax = fig.add_axes([0.02, 0.555, 0.96, 0.432])
    strip_ax = fig.add_axes([0.02, 0.455, 0.96, 0.070])
    list_ax = fig.add_axes([0.02, 0.385, 0.96, 0.055])
    norm_ax = fig.add_axes([0.02, 0.020, 0.96, 0.330])

    _draw_photo(photo_ax, data)
    _draw_strip(strip_ax, data)
    _draw_rle_list(list_ax, data)
    _draw_normalization(norm_ax, data)

    fig.text(
        0.02,
        0.535,
        f"run-length encoding of row {ROW}",
        fontsize=12,
        color=INK,
        ha="left",
        va="bottom",
    )
    fig.text(
        0.5,
        0.365,
        "brackets = finder 5-tuples  ·  lengths in full-resolution pixels",
        fontsize=9.5,
        color=MUTED,
        ha="center",
        va="center",
    )

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    data = measure()
    fig = build_figure(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.out,
        dpi=200,
        facecolor=WHITE,
        metadata={"Software": "qr-reader blog figure"},
    )
    plt.close(fig)

    print(f"image: {IMAGE_NAME}")
    print(f"crop: x {CROP_X0}..{CROP_X1}, y {CROP_Y0}..{CROP_Y1} "
          f"({CROP_X1 - CROP_X0} x {CROP_Y1 - CROP_Y0} px)")
    print(
        "left finder centre: "
        f"({data.left_quad[:, 0].mean():.1f}, {data.left_quad[:, 1].mean():.1f})"
    )
    print(
        "right finder centre: "
        f"({data.right_quad[:, 0].mean():.1f}, {data.right_quad[:, 1].mean():.1f})"
    )
    print(f"row: {ROW}")
    print("full RLE:")
    print("  " + ", ".join(str(run.length) for run in data.runs))
    for name, group in (
        ("left finder", data.left),
        (f"right finder @ row {ROW}", data.right_at_row),
        (f"right finder @ own row {RIGHT_OWN_ROW}", data.right_own),
    ):
        print(f"{name}:")
        print(f"  runs: {group.lengths.tolist()}  sum={group.total}")
        print(f"  normalized: {np.round(group.normalized, 4).tolist()}")
        print(f"  error: {np.round(group.errors * 100, 1).tolist()} %")
        print(f"  score={group.score:.4f}  pass={group.passes}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
