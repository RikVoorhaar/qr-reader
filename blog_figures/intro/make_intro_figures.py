#!/usr/bin/env python3
"""Generate the two opening figures for the QR-reader blog post.

Outputs (under ``blog/src/assets/blog/qr-reader/``):

* ``qr-anatomy.png``  — a clean QR code with its three finder patterns boxed,
  plus a zoomed crop of one finder with the 1:1:3:1:1 run lengths labelled.
* ``qr-versions.png`` — a montage of versions 1, 2, 5 and 10 with their module
  counts, plus the worked version-estimate example from the post
  (finder 70 px, gap 110 px, total 250 px -> 10 px/module -> 25 modules ->
  version 2).

The QRs are generated with the ``qrcode`` package (the same dependency used by
``qr_reader.qr_gen``).  Everything is deterministic: the payload is fixed and
the library's masking choice is deterministic for a fixed version.

Run from the repository root:

    .venv/bin/python blog_figures/intro/make_intro_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import qrcode

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

# ── Shared style ─────────────────────────────────────────────────────────────
ACCENT = "#d1495b"
INK = "#1b1b1b"
MUTED = "#6b7280"
GRID = "#b9bec7"
FINDER_COLOR = "#111111"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# Same payload and EC level for every code in both figures.
PAYLOAD = "rikvoorhaar.com"
ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_L

# The canonical 7x7 finder pattern (1 = dark, 0 = light).
FINDER_7 = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=float,
)


# ── QR generation and drawing helpers ────────────────────────────────────────


def qr_matrix(version: int, payload: str = PAYLOAD) -> np.ndarray:
    """Return the module matrix (True = dark) for one fixed QR version."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECTION,
        box_size=1,
        border=0,
    )
    qr.add_data(payload)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=bool)


def draw_qr(
    ax: plt.Axes,
    matrix: np.ndarray,
    quiet: int = 4,
    highlight_finders: bool = False,
) -> int:
    """Draw a QR module matrix with a quiet zone; return the padded side length."""
    n = matrix.shape[0]
    side = n + 2 * quiet
    image = np.ones((side, side), dtype=float)
    # Dark modules become 0 so the default gray colormap renders them black.
    image[quiet : quiet + n, quiet : quiet + n] = np.where(matrix, 0.0, 1.0)

    ax.imshow(
        image,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=(0, side, side, 0),
        interpolation="nearest",
        zorder=1,
    )

    if highlight_finders:
        finder_origins = [
            (quiet, quiet),
            (quiet, quiet + n - 7),
            (quiet + n - 7, quiet),
        ]
        for row, col in finder_origins:
            # White underlay keeps the accent outline legible over dark modules.
            ax.add_patch(
                Rectangle(
                    (col - 0.16, row - 0.16),
                    7.32,
                    7.32,
                    fill=False,
                    edgecolor="white",
                    linewidth=5.6,
                    joinstyle="round",
                    zorder=4,
                )
            )
            ax.add_patch(
                Rectangle(
                    (col - 0.16, row - 0.16),
                    7.32,
                    7.32,
                    fill=False,
                    edgecolor=ACCENT,
                    linewidth=2.8,
                    joinstyle="round",
                    zorder=5,
                )
            )

    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    return side


def draw_finder_arrays(ax: plt.Axes, origin_xy: tuple[float, float], pitch: float) -> None:
    """Draw the canonical finder pattern at a given origin and module pitch."""
    x0, y0 = origin_xy
    ax.imshow(
        FINDER_7,
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        extent=(x0, x0 + 7 * pitch, y0 + 7 * pitch, y0),
        interpolation="nearest",
        zorder=2,
    )


# ── Figure 1: QR anatomy ─────────────────────────────────────────────────────


def make_anatomy_figure(output_path: Path) -> None:
    matrix = qr_matrix(version=2)

    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.06)

    # ── Panel 1: the whole code, with the three finders boxed ──
    ax = fig.add_subplot(gs[0, 0])
    draw_qr(ax, matrix, quiet=4, highlight_finders=True)
    ax.set_ylim(35.4, -0.6)

    # Small legend below the code.
    legend_y = 33.9
    ax.add_patch(
        Rectangle(
            (10.2, legend_y),
            1.1,
            1.1,
            facecolor="none",
            edgecolor=ACCENT,
            linewidth=2.6,
            joinstyle="round",
        )
    )
    ax.text(
        12.0,
        legend_y + 0.58,
        "finder pattern",
        fontsize=13,
        color=INK,
        va="center",
        ha="left",
    )

    # ── Panel 2: zoomed finder, with the 1:1:3:1:1 runs labelled ──
    ax = fig.add_subplot(gs[0, 1])

    # One module of white padding around the 7x7 finder.
    crop = np.ones((9, 9), dtype=float)
    crop[1:8, 1:8] = np.where(matrix[:7, :7], 0.0, 1.0)
    ax.imshow(
        crop,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        extent=(0, 9, 9, 0),
        interpolation="nearest",
        zorder=1,
    )

    # Horizontal scan through the centre of the finder (row 3 of the 7x7).
    scan_y = 4.5
    ax.plot(
        [0.25, 8.75],
        [scan_y, scan_y],
        linestyle=(0, (4, 3)),
        color=ACCENT,
        linewidth=1.8,
        zorder=6,
    )

    # Run-length bands directly below the crop: black, white, black, white, black.
    runs = [(1, 2, 1, "dark"), (2, 3, 1, "light"), (3, 6, 3, "dark"),
            (6, 7, 1, "light"), (7, 8, 1, "dark")]
    band_top = 9.45
    band_height = 0.7
    for start, stop, length, shade in runs:
        face = FINDER_COLOR if shade == "dark" else "white"
        edge = FINDER_COLOR if shade == "light" else "none"
        ax.add_patch(
            Rectangle(
                (start, band_top),
                stop - start,
                band_height,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.9,
                zorder=2,
            )
        )
        ax.text(
            (start + stop) / 2,
            band_top + band_height / 2,
            str(length),
            fontsize=13,
            fontweight="bold",
            color="white" if shade == "dark" else INK,
            ha="center",
            va="center",
            zorder=3,
        )

    # Bracket under the bands: the five runs together span seven modules.
    bracket_y = 10.55
    ax.plot([1, 8], [bracket_y, bracket_y], color=ACCENT, linewidth=1.8)
    for x in (1, 8):
        ax.plot(
            [x, x],
            [bracket_y - 0.22, bracket_y + 0.22],
            color=ACCENT,
            linewidth=1.8,
        )
    ax.text(
        4.5,
        bracket_y + 0.5,
        "7 modules",
        fontsize=13,
        fontweight="bold",
        color=ACCENT,
        ha="center",
        va="center",
    )

    ax.set_xlim(0, 9)
    ax.set_ylim(11.45, -0.35)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")


# ── Figure 2: versions and version estimation ────────────────────────────────


def draw_version_panel(ax: plt.Axes, version: int) -> None:
    matrix = qr_matrix(version=version)
    n = matrix.shape[0]
    draw_qr(ax, matrix, quiet=2)
    ax.set_title(
        f"v{version}  ·  {n}×{n} modules",
        fontsize=13.5,
        color=INK,
        pad=7,
    )


def draw_dimension(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    label: str,
    *,
    color: str = MUTED,
    fontsize: float = 12.0,
    label_offset: float = 9.0,
    weight: str = "normal",
) -> None:
    """Horizontal dimension arrow with an end tick at each side and a label above."""
    for x in (x0, x1):
        ax.plot([x, x], [y - 5.0, y], color=color, linewidth=1.1, zorder=3)
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="<|-|>",
            mutation_scale=9,
            linewidth=1.2,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )
    ax.text(
        (x0 + x1) / 2,
        y - label_offset,
        label,
        fontsize=fontsize,
        color=color,
        fontweight=weight,
        ha="center",
        va="center",
    )


def make_versions_figure(output_path: Path) -> None:
    versions = (1, 2, 5, 10)

    fig = plt.figure(figsize=(12.4, 6.2))
    outer = fig.add_gridspec(
        1, 2, width_ratios=[1.02, 1.0], wspace=0.10,
        left=0.015, right=0.985, bottom=0.03, top=0.93,
    )
    montage = outer[0, 0].subgridspec(2, 2, wspace=0.10, hspace=0.34)

    for index, version in enumerate(versions):
        ax = fig.add_subplot(montage[index // 2, index % 2])
        draw_version_panel(ax, version)

    # ── Spacing diagram (right panel) ──
    ax = fig.add_subplot(outer[0, 1])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-24, 274)
    ax.set_ylim(202, -44)

    # The top edge of the QR: finder (70 px), gap (110 px), finder (70 px).
    draw_finder_arrays(ax, (0, 0), pitch=10)
    draw_finder_arrays(ax, (180, 0), pitch=10)

    # Faint module grid in the gap makes the 11 modules countable.
    for x in np.arange(70, 180.1, 10):
        ax.plot([x, x], [0, 70], color=GRID, linewidth=1.0, zorder=1)
    ax.plot([0, 250], [70, 70], color=GRID, linewidth=1.1, zorder=1)

    # Dimension arrows above the strip.
    draw_dimension(ax, 0, 70, -17, "70 px", fontsize=13.0)
    draw_dimension(ax, 70, 180, -17, "110 px", fontsize=13.0)
    draw_dimension(ax, 180, 250, -17, "70 px", fontsize=13.0)

    # Module counts under each part.
    ax.text(35, 80, "7 modules", fontsize=12.5, color=MUTED, ha="center", va="top")
    ax.text(125, 80, "11 modules", fontsize=12.5, color=MUTED, ha="center", va="top")
    ax.text(215, 80, "7 modules", fontsize=12.5, color=MUTED, ha="center", va="top")

    # Total side length.
    draw_dimension(
        ax, 0, 250, 116, "250 px = 25 modules",
        color=ACCENT, fontsize=14.0, label_offset=10.5, weight="bold",
    )

    # The arithmetic from the post.
    ax.text(
        0, 154,
        "one module = 70 px / 7 = 10 px",
        fontsize=13.5, color=INK, ha="left", va="center",
    )
    ax.text(
        0, 172,
        "side = 250 px / 10 px = 25 modules",
        fontsize=13.5, color=INK, ha="left", va="center",
    )
    ax.text(
        0, 190,
        "25 = 17 + 4 × 2   →   version 2",
        fontsize=13.5, color=ACCENT, fontweight="bold", ha="left", va="center",
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_anatomy_figure(OUTPUT_DIR / "qr-anatomy.png")
    make_versions_figure(OUTPUT_DIR / "qr-versions.png")


if __name__ == "__main__":
    main()
