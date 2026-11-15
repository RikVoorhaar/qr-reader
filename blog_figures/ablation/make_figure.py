#!/usr/bin/env python3
"""Render the pipeline ablation as a vertical SVG flow.

Reads ``ablation_counts.json`` produced by ``ablation_counts.py`` and writes
``pipeline-ablation.svg`` for the blog post.

Each box shows a pipeline stage and the number of captured photos that reach
it. The label on the arrow below a box is the number of photos lost at that
stage. Grey stages cannot fail, amber stages can fail but never did on the
100 photos, and red stages did fail.

Run from the repository root:

    .venv/bin/python blog_figures/ablation/make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COUNTS_PATH = REPO_ROOT / "blog_figures" / "ablation" / "ablation_counts.json"
OUT_PATH = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "pipeline-ablation.svg"
)

# (json key, display label, can_fail)
STAGES = [
    ("binarize", "Binarize", False),
    ("find_candidates", "Find candidates", True),
    ("cluster", "Cluster candidates", False),
    ("fit_finders", "Fit finder corners", True),
    ("find_triplets", "Find triplets", True),
    ("rank_versions", "Rank versions", True),
    ("fit_homography", "Fit homography", True),
    ("refine_alignment", "Refine with alignment patterns", True),
    ("sample", "Sample modules", True),
    ("decode", "Decode payload", True),
]

# Palette matches the stone/amber/red tones used elsewhere on the blog.
GREY = {
    "fill": "#e7e5e4",
    "stroke": "#a8a29e",
    "text": "#292524",
}
AMBER = {
    "fill": "#fef3c7",
    "stroke": "#f59e0b",
    "text": "#78350f",
}
RED = {
    "fill": "#fee2e2",
    "stroke": "#ef4444",
    "text": "#7f1d1d",
}

WIDTH = 760
BOX_X = 70
BOX_W = 620
BOX_H = 58
GAP = 28
TOP = 28
BOTTOM = 104


def _style(stage_key: str, can_fail: bool, loss: int) -> dict:
    if not can_fail:
        return GREY
    if loss > 0:
        return RED
    return AMBER


def render(counts: dict[str, int]) -> str:
    n_stages = len(STAGES)
    height = TOP + n_stages * BOX_H + (n_stages - 1) * GAP + BOTTOM
    lines: list[str] = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="{WIDTH}" height="{height}" role="img">'
    )
    lines.append("<title>QR pipeline ablation</title>")
    lines.append(f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>')

    prev_count = 100
    last_loss = 0
    last_y = TOP
    centers: list[tuple[float, float]] = []
    for i, (key, label, can_fail) in enumerate(STAGES):
        count = int(counts.get(key, 0))
        loss = max(0, prev_count - count)
        style = _style(key, can_fail, loss)
        y = TOP + i * (BOX_H + GAP)
        centers.append((BOX_X + BOX_W / 2, y + BOX_H / 2))

        lines.append(
            f'<rect x="{BOX_X}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="10" '
            f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{BOX_X + 24}" y="{y + BOX_H / 2 + 6}" '
            f'font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif" '
            f'font-size="19" fill="{style["text"]}">{label}</text>'
        )
        lines.append(
            f'<text x="{BOX_X + BOX_W - 24}" y="{y + BOX_H / 2 + 7}" '
            f'text-anchor="end" font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif" '
            f'font-size="22" font-weight="700" fill="{style["text"]}">{count}</text>'
        )

        if i < n_stages - 1:
            y0 = y + BOX_H
            y1 = y + BOX_H + GAP
            mx = BOX_X + BOX_W / 2
            lines.append(
                f'<line x1="{mx}" y1="{y0 + 2}" x2="{mx}" y2="{y1 - 8}" '
                f'stroke="#78716c" stroke-width="2"/>'
            )
            lines.append(
                f'<path d="M {mx - 5} {y1 - 9} L {mx} {y1 - 1} L {mx + 5} {y1 - 9}" '
                f'fill="none" stroke="#78716c" stroke-width="2" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
            if loss > 0:
                lines.append(
                    f'<text x="{mx + 14}" y="{y0 + GAP / 2 + 5}" '
                    f'font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif" '
                    f'font-size="15" font-weight="600" fill="#b91c1c">−{loss}</text>'
                )

        prev_count = count
        last_loss = loss
        last_y = y

    # The final stage has no arrow below it, so show its loss under the count.
    if last_loss > 0:
        lines.append(
            f'<text x="{BOX_X + BOX_W - 24}" y="{last_y + BOX_H + 18}" '
            f'text-anchor="end" font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif" '
            f'font-size="15" font-weight="600" fill="#b91c1c">−{last_loss}</text>'
        )

    # Legend.
    ly = height - 58
    lx = BOX_X
    items = [
        (GREY, "cannot fail"),
        (AMBER, "can fail, never did"),
        (RED, "did fail"),
    ]
    for style, label in items:
        lines.append(
            f'<rect x="{lx}" y="{ly}" width="18" height="18" rx="4" '
            f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="2"/>'
        )
        lines.append(
            f'<text x="{lx + 26}" y="{ly + 14}" '
            f'font-family="ui-sans-serif, -apple-system, Segoe UI, sans-serif" '
            f'font-size="14" fill="#44403c">{label}</text>'
        )
        lx += 190

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    data = json.loads(COUNTS_PATH.read_text())
    counts = data["counts"]
    svg = render(counts)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(svg)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
