#!/usr/bin/env python3
"""Animate rough affine sampling of a real QR photo.

Left panel : the affine-rectified crop with the module grid and the modules
             sampled in the current frame highlighted.
Right panel: the sampled bit matrix, filling in row by row.  During the final
             hold, bits that disagree with the clean reference are marked red.

Run from the repository root:

    python blog_figures/sampling/make_animation.py
    python blog_figures/sampling/make_animation.py --image "data/bench/captured/qr_code_ - 50.jpeg"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from sampling_compare import Comparison, compare_image  # noqa: E402

REPO_ROOT = HERE.parents[1]
DEFAULT_IMAGE = REPO_ROOT / "data" / "bench" / "captured" / "qr_code_ - 50.jpeg"
DEFAULT_OUTPUT = (
    REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader" / "affine-sampling.gif"
)

# ── Layout / style ───────────────────────────────────────────────────────────

LEFT_SIZE = 420  # px, square rectified crop
RIGHT_PITCH = 15  # px per module in the sampled matrix
PAD = 20
LABEL_H = 26
FPS = 10
MODULES_PER_FRAME = 10
HOLD_SECONDS = 1.2
# A fixed palette keeps the thin red/orange/cyan marks from being quantised
# away by the mostly grey QR image.
GIF_PALETTE = [
    (255, 255, 255),
    (250, 250, 250),
    (226, 226, 226),
    (205, 205, 205),
    (170, 170, 170),
    (130, 130, 130),
    (90, 90, 90),
    (55, 55, 55),
    (24, 24, 24),
    (0, 0, 0),
    (0, 170, 220),
    (0, 110, 150),
    (255, 127, 14),
    (255, 0, 0),
    (150, 0, 0),
    (35, 35, 35),
]

BG = (255, 255, 255)
UNSAMPLED = (226, 226, 226)
SAMPLED_LIGHT = (250, 250, 250)
SAMPLED_DARK = (24, 24, 24)
GRID = (0, 170, 220)
GRID_ALPHA = 105
CURRENT = (255, 127, 14)
WRONG = (255, 0, 0)
TEXT = (35, 35, 35)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def _overlay_grid(rectified: np.ndarray, N: int, size: int) -> Image.Image:
    """Rectified grayscale crop with a translucent module grid."""
    base = Image.fromarray(rectified).convert("RGBA")
    base = base.resize((size, size), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    step = base.size[0] / N
    for k in range(N + 1):
        p = round(k * step)
        draw.line([(p, 0), (p, base.size[1])], fill=(*GRID, GRID_ALPHA), width=1)
        draw.line([(0, p), (base.size[0], p)], fill=(*GRID, GRID_ALPHA), width=1)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _draw_current(
    panel: Image.Image, N: int, current: list[tuple[int, int]]
) -> None:
    """Orange outline around the modules sampled in this frame."""
    draw = ImageDraw.Draw(panel)
    step = panel.size[0] / N
    for r, c in current:
        x0, y0 = c * step, r * step
        x1, y1 = (c + 1) * step, (r + 1) * step
        draw.rectangle([x0, y0, x1, y1], outline=CURRENT, width=3)


def _draw_matrix(
    panel: Image.Image,
    comp: Comparison,
    sampled_rc: np.ndarray,
    n_sampled: int,
    show_wrong: bool,
) -> None:
    """Draw the sampled matrix; ``sampled_rc`` is indexed [row, col]."""
    N = comp.N
    draw = ImageDraw.Draw(panel)
    wrong = comp.wrong_mask

    for r in range(N):
        for c in range(N):
            idx = r * N + c
            x0, y0 = c * RIGHT_PITCH, r * RIGHT_PITCH
            x1, y1 = x0 + RIGHT_PITCH - 1, y0 + RIGHT_PITCH - 1
            if idx >= n_sampled:
                fill = UNSAMPLED
            elif sampled_rc[r, c]:
                fill = SAMPLED_DARK
            else:
                fill = SAMPLED_LIGHT
            draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Subtle cell separators so the matrix reads as a grid.
    for k in range(N + 1):
        p = k * RIGHT_PITCH
        draw.line([(p, 0), (p, N * RIGHT_PITCH)], fill=(205, 205, 205), width=1)
        draw.line([(0, p), (N * RIGHT_PITCH, p)], fill=(205, 205, 205), width=1)

    if show_wrong:
        overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for r, c in comp.wrong_locations:
            x0, y0 = c * RIGHT_PITCH, r * RIGHT_PITCH
            x1, y1 = x0 + RIGHT_PITCH - 1, y0 + RIGHT_PITCH - 1
            od.rectangle([x0, y0, x1, y1], outline=(*WRONG, 255), width=2)
            od.line([(x0 + 3, y0 + 3), (x1 - 3, y1 - 3)], fill=(*WRONG, 255), width=2)
            od.line([(x0 + 3, y1 - 3), (x1 - 3, y0 + 3)], fill=(*WRONG, 255), width=2)
        panel.alpha_composite(overlay)


def make_animation(comp: Comparison, output: Path) -> None:
    N = comp.N
    right_size = N * RIGHT_PITCH
    width = PAD + LEFT_SIZE + PAD + right_size + PAD
    height = PAD + LABEL_H + max(LEFT_SIZE, right_size) + PAD

    left_base = _overlay_grid(comp.rectified, N, LEFT_SIZE)
    sampled_rc = comp.sampled_bits.T  # [row, col]

    total = N * N
    groups = [
        comp.scan_order[i : i + MODULES_PER_FRAME]
        for i in range(0, total, MODULES_PER_FRAME)
    ]

    frames: list[Image.Image] = []
    label_font = _font(18)

    def render(n_sampled: int, current: list[tuple[int, int]], show_wrong: bool) -> Image.Image:
        canvas = Image.new("RGB", (width, height), BG)
        draw = ImageDraw.Draw(canvas)

        left_x, top_y = PAD, PAD + LABEL_H
        right_x = PAD + LEFT_SIZE + PAD

        draw.text((left_x, PAD - 4), "rectified image + grid", fill=TEXT, font=label_font)
        draw.text((right_x, PAD - 4), "sampled matrix", fill=TEXT, font=label_font)

        left = left_base.copy()
        if current:
            _draw_current(left, N, current)
        canvas.paste(left, (left_x, top_y))

        right = Image.new("RGBA", (right_size, right_size), (*BG, 255))
        _draw_matrix(right, comp, sampled_rc, n_sampled, show_wrong)
        canvas.paste(right.convert("RGB"), (right_x, top_y))
        return canvas

    # Frame 0: rectified crop, blank matrix.
    frames.append(render(0, [], False))

    n_sampled = 0
    for group in groups:
        n_sampled += len(group)
        frames.append(render(n_sampled, group, False))

    # Final hold: all bits sampled, wrong bits marked.
    hold_frames = max(1, round(HOLD_SECONDS * FPS))
    final = render(total, [], True)
    frames.extend([final] * hold_frames)

    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = round(1000 / FPS)
    palette = Image.new("P", (1, 1))
    flat = [channel for colour in GIF_PALETTE for channel in colour]
    palette.putpalette(flat + [0] * (768 - len(flat)))
    palette_frames = [
        f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    palette_frames[0].save(
        output,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(
        f"Wrote {output} ({len(frames)} frames, {width}x{height}, "
        f"{duration_ms} ms/frame, {comp.num_wrong} wrong bits)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    comp = compare_image(args.image)
    print(
        f"{comp.image_name}: v{comp.version}, {comp.ecl}/mask {comp.mask}, "
        f"{comp.num_wrong} wrong bits"
    )
    make_animation(comp, args.output)


if __name__ == "__main__":
    main()
