#!/usr/bin/env python3
"""Render the captured-image pipeline funnel as a compact two-column SVG."""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "blog_figures" / "ablation" / "ablation_counts.json"
OUT_PATH = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "pipeline-ablation.svg"
)

# Original pipeline stages and observed counts reaching each stage.
STAGES = [
    ("binarize", "Binarize"),
    ("find_candidates", "Find candidates"),
    ("cluster", "Cluster candidates"),
    ("fit_finders", "Fit finder corners"),
    ("find_triplets", "Find triplets"),
    ("rank_versions", "Rank versions"),
    ("fit_homography", "Fit homography"),
    ("refine_alignment", "Refine with alignment patterns"),
    ("sample", "Sample modules"),
    ("decode", "Decode payload"),
]

# Secondary annotations: viability thresholds are explicitly described as
# diagnostics, not as gates that the pipeline actually enforced.

# Quiet, unified palette drawn from the QR-reader blog figures.
INK = "#1c2136"
CARD = "#f6f1da"
CARD_STROKE = "#d8d1b8"
ARROW = "#ff7800"
ACCENT = "#c2410c"
ACCENT_RULE = "#ff7800"

WIDTH, HEIGHT = 900, 558
BOX_W, BOX_H = 400, 68
X = (20, 480)
Y0, ROW_GAP = 16, 88


def _stage_positions() -> list[tuple[int, int]]:
    positions = []
    for i in range(len(STAGES)):
        row, position = divmod(i, 2)
        col = position if row % 2 == 0 else 1 - position
        positions.append((X[col], Y0 + row * ROW_GAP))
    return positions


def render(counts: dict[str, int], notes: dict, scale_recoveries: int) -> str:
    positions = _stage_positions()
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img">',
        (
            "<desc>Pipeline viability counts without resizing, followed separately by "
            "the additional decodes from retrying on an enlarged image.</desc>"
        ),
        '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" '
        'refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 Z" '
        f'fill="{ARROW}"/></marker></defs>',
    ]

    # Draw the serpentine connectors first, behind the stage cards.
    for row in range(4):
        right_to_left = row % 2 == 1
        edge_x = X[0] + BOX_W / 2 if right_to_left else X[1] + BOX_W / 2
        y = Y0 + row * ROW_GAP + BOX_H + 2
        next_y = Y0 + (row + 1) * ROW_GAP - 5
        lines.append(
            f'<path d="M {edge_x} {y} V {next_y}" fill="none" '
            f'stroke="{ARROW}" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )
    for row in range(5):
        i = row * 2
        x1, x2 = (
            (X[0] + BOX_W + 5, X[1] - 6)
            if row % 2 == 0
            else (X[1] - 5, X[0] + BOX_W + 6)
        )
        y = Y0 + row * ROW_GAP + BOX_H / 2
        lines.append(
            f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
            f'stroke="{ARROW}" stroke-width="2.5" marker-end="url(#arrow)"/>'
        )

    for i, ((key, label), (x, y)) in enumerate(zip(STAGES, positions)):
        note = notes.get(key)
        lines.append(
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
            f'fill="{CARD}" stroke="{CARD_STROKE}" stroke-width="1.4"/>'
        )
        # A consistent slim accent, not a separate failure-status block.
        lines.append(
            f'<path d="M {x + 2} {y + 9} V {y + BOX_H - 9}" '
            f'stroke="{ACCENT_RULE}" stroke-width="3" stroke-linecap="round"/>'
        )
        label_size = 17 if len(label) < 25 else 15
        lines.append(
            f'<text x="{x + 16}" y="{y + 28}" '
            'font-family="ui-sans-serif, sans-serif" '
            f'font-size="{label_size}" font-weight="700" fill="{INK}">'
            f"{html.escape(label)}</text>"
        )
        lines.append(
            f'<text x="{x + BOX_W - 16}" y="{y + 30}" text-anchor="end" '
            'font-family="ui-sans-serif, sans-serif" '
            f'font-size="20" font-weight="700" fill="{INK}">{int(counts[key])}</text>'
        )
        if note:
            number, explanation = note
            lines.append(
                f'<text x="{x + 16}" y="{y + 53}" xml:space="preserve" '
                'font-family="ui-sans-serif, sans-serif" '
                f'font-size="13" fill="{ACCENT}">'
                f'<tspan font-weight="700">{html.escape(number)}</tspan>'
                f'<tspan font-weight="500">{html.escape(explanation)}</tspan></text>'
            )

    # This is a separate retry, not the next step for images that decoded.
    x, y = X[1], 474
    total = counts["decode"] + scale_recoveries
    lines.append(
        f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="8" '
        f'fill="{CARD}" stroke="{CARD_STROKE}" stroke-width="1.4"/>'
    )
    lines.append(
        f'<path d="M {x + 2} {y + 9} V {y + BOX_H - 9}" '
        f'stroke="{ACCENT_RULE}" stroke-width="3" stroke-linecap="round"/>'
    )
    lines.append(
        f'<text x="{x + 16}" y="{y + 28}" font-family="ui-sans-serif, sans-serif" '
        f'font-size="17" font-weight="700" fill="{INK}">With scale retry</text>'
    )
    lines.append(
        f'<text x="{x + BOX_W - 16}" y="{y + 30}" text-anchor="end" '
        f'font-family="ui-sans-serif, sans-serif" font-size="20" font-weight="700" '
        f'fill="{INK}">{total}</text>'
    )
    lines.append(
        f'<text x="{x + 16}" y="{y + 53}" xml:space="preserve" font-family="ui-sans-serif, sans-serif" '
        f'font-size="13" fill="{ACCENT}"><tspan font-weight="700">+{scale_recoveries}</tspan>'
        '<tspan font-weight="500"> after enlarging the image</tspan></text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    data = json.loads(COUNTS_PATH.read_text())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    images = data["images"]
    ordered_stages = [
        "binarize",
        "find_candidates",
        "cluster",
        "fit_finders",
        "find_triplets",
        "rank_versions",
        "fit_homography",
        "refine_alignment",
        "sample",
        "decode",
    ]
    stage_index = {stage: i for i, stage in enumerate(ordered_stages)}

    def reached(image: dict, stage: str) -> bool:
        return stage_index[image["furthest_stage"]] >= stage_index[stage]

    # Keep the real stage names, but show the viability funnel through the
    # minimum-cluster and minimum-finder checks before actual triplet discovery.
    funnel_counts = {
        "binarize": len(images),
        "find_candidates": sum(image["n_candidates"] > 0 for image in images),
        "cluster": sum(image["n_candidates"] >= 3 for image in images),
        "fit_finders": sum(image["n_finder_patterns"] >= 3 for image in images),
        "find_triplets": sum(image["n_triplets"] >= 1 for image in images),
        **{
            stage: sum(reached(image, stage) for image in images)
            for stage in ordered_stages[5:]
        },
    }
    # Count both sampling variants in the decode stage on the original image.
    scale_recoveries = data["recoveries"].get("scale_retry", 0)
    funnel_counts["decode"] = data["final_decoded"] - scale_recoveries
    notes = {
        "find_candidates": (str(len(images) - funnel_counts["find_candidates"]), " had no candidate clusters"),
        "cluster": (str(funnel_counts["find_candidates"] - funnel_counts["cluster"]), " below the 3-cluster viability threshold"),
        "fit_finders": (str(funnel_counts["cluster"] - funnel_counts["fit_finders"]), " below the 3-finder threshold after dedup"),
        "find_triplets": (str(funnel_counts["fit_finders"] - funnel_counts["find_triplets"]), " had ≥3 finders, but no valid triplet"),
        "decode": (str(funnel_counts["sample"] - funnel_counts["decode"]), " failed to decode after sampling"),
    }
    OUT_PATH.write_text(render(funnel_counts, notes, scale_recoveries))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
