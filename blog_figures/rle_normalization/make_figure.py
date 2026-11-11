#!/usr/bin/env python3
"""Build the RLE-normalization figure for the QR-reader blog post.

The figure shows a thresholded 2:1 crop of a real QR photo, a scanline through
the two upper finder patterns, the run lengths along that row, and the
normalized ratios for the left finder's five runs.  The right finder is
annotated but not normalized: at this row it is off-centre, so its cut is
distorted.

It uses the same captured photo as ``rle-scan.gif``:

    data/bench/captured-fullsize/qr_code_ - 99.jpeg

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
from matplotlib.colors import ListedColormap
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
BINARY_DARK = "#1c2136"
BINARY_LIGHT = "#f6f1da"
ORANGE = "#ff7800"  # finder tuple / worked example
ORANGE_LIGHT = "#ffb86b"  # finder labels on dark runs
ORANGE_DARK = "#c2410c"  # finder labels on light runs
ORANGE_TEXT = "#d95f00"  # normalized ratios on white
CYAN = "#00dcff"  # scanline
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
    binary: np.ndarray
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
        binary=binary,
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
        data.binary[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1],
        cmap=ListedColormap([BINARY_DARK, BINARY_LIGHT]),
        vmin=0,
        vmax=1,
        extent=[CROP_X0, CROP_X1, CROP_Y1, CROP_Y0],
        interpolation="nearest",
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

    ax.set_xlim(CROP_X0, CROP_X1)
    ax.set_ylim(CROP_Y1, CROP_Y0)
    ax.axis("off")


def _fit_text_to_run(ax, text_artist, run: Run, padding: float = 4.0) -> None:
    """Scale one label down until it fits inside its run rectangle."""
    renderer = ax.figure.canvas.get_renderer()
    bbox = text_artist.get_window_extent(renderer=renderer)
    x0 = ax.transData.transform((run.x0, 0.0))[0]
    x1 = ax.transData.transform((run.x1, 0.0))[0]
    max_width = abs(x1 - x0) - padding
    if max_width > 0 and bbox.width > max_width:
        text_artist.set_fontsize(text_artist.get_fontsize() * max_width / bbox.width)


def _draw_strip(ax, data: Measurements) -> None:
    left = data.left
    right = data.right_at_row
    left_indices = set(range(left.start_index, left.start_index + 5))

    ax.set_xlim(CROP_X0, CROP_X1)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    labels = []

    for i, run in enumerate(data.runs):
        ax.add_patch(
            Rectangle(
                (run.x0, 0.55),
                run.x1 - run.x0,
                0.30,
                facecolor=BINARY_LIGHT if run.is_white else BINARY_DARK,
                edgecolor="#d8d2b8" if run.is_white else BINARY_DARK,
                linewidth=0.5,
                zorder=2,
            )
        )

        is_left = i in left_indices
        if is_left:
            color = ORANGE_LIGHT if not run.is_white else ORANGE_DARK
            fontsize = 10.0
            fontweight = "bold"
        else:
            color = WHITE if not run.is_white else INK
            fontsize = 9.0
            fontweight = "normal"

        text = ax.text(
            (run.x0 + run.x1) / 2.0,
            0.70,
            str(run.length),
            color=color,
            fontsize=fontsize,
            fontweight=fontweight,
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
            zorder=5,
        )
        labels.append((text, run))

    for group, label, color in (
        (left, "left finder", ORANGE),
        (right, "right finder", MUTED),
    ):
        ax.plot([group.x0, group.x1], [0.90, 0.90], color=color, lw=1.4, zorder=4)
        for x in (group.x0, group.x1):
            ax.plot([x, x], [0.85, 0.90], color=color, lw=1.4, zorder=4)
        ax.text(
            (group.x0 + group.x1) / 2.0,
            0.97,
            label,
            color=color,
            fontsize=9.0,
            fontweight="bold",
            ha="center",
            va="bottom",
            zorder=5,
        )

    for run, value in zip(
        data.runs[left.start_index : left.start_index + 5], left.normalized
    ):
        ratio_y = 0.14 if run.is_white else 0.32
        ax.text(
            (run.x0 + run.x1) / 2.0,
            ratio_y,
            f"{value:.2f}",
            color=ORANGE_TEXT,
            fontsize=9.0,
            fontweight="bold",
            ha="center",
            va="center",
            fontfamily="DejaVu Sans Mono",
            zorder=5,
        )

    # Measure after the axes limits and rectangle positions are final.
    ax.figure.canvas.draw()
    for text, run in labels:
        _fit_text_to_run(ax, text, run)


# ── Main ─────────────────────────────────────────────────────────────────────


def build_figure(data: Measurements) -> plt.Figure:
    fig = plt.figure(figsize=(9.0, 6.0), dpi=200)

    photo_ax = fig.add_axes([0.02, 0.26, 0.96, 0.72])
    strip_ax = fig.add_axes([0.02, 0.02, 0.96, 0.22])

    _draw_photo(photo_ax, data)
    _draw_strip(strip_ax, data)

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
