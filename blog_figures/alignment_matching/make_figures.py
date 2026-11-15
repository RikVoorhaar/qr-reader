#!/usr/bin/env python3
"""Build the alignment-pattern matching figures for the QR-reader blog.

Two assets are produced from one real capture:

``alignment-matching.gif``
    Two panels.  The left panel is a tight crop of the photo with the
    synthetic alignment-pattern template drawn as a moving outline (its
    dark-ring boundaries) plus a centre crosshair and a live score.  The
    right panel is the normalized-cross-correlation surface produced by the
    *real* ``_ncc_match`` call, filling in as the template slides, with a
    "Pearson correlation" colour bar, the current position, and the peak
    marked at the end.  No captions: panels, axis labels, colour bar, score
    and peak label only.

``alignment-subpixel.png``
    A compact 1 x 2 figure with just the two 1-D parabolas (horizontal and
    vertical) that give the sub-pixel centre.  Each panel shows the three
    correlation samples at 4 d.p., the fitted parabola, its vertex (diamond)
    and a dotted line at the interpolated centre coordinate.  The diamond sits
    at the parabola's apex -- ``(vertex abscissa, vertex height)``, where the
    height is the parabola evaluated at the vertex -- not at the integer
    sample's score; the dotted line and the leader end at that same apex.

Subject image
    ``data/bench/captured/qr_code_ - 55.jpeg`` (real capture, QR v3,
    ~8.95 px/module, exactly one alignment pattern, match score 0.571).  The
    ``data/`` tree lives in the main checkout, so the absolute path is the
    default here; override with ``--image``.

Honest notes on the algorithm
    *  Production does **not** slide the template pixel by pixel.  ``_ncc_match``
       evaluates *every* integer shift at once with ``scipy.signal.fftconvolve``
       plus integral images.  The animation walks a coarse serpentine subset of
       the integer positions purely for legibility, and reveals the (already
       fully computed) surface along that path.  The values are untouched.
    *  Sub-pixel refinement is **two separable 1-D parabolas** through the
       horizontal and vertical triples (5 of the 9 neighbours), not a 2-D
       quadratic fit over 3 x 3.
    *  ``res[i, j]`` scores the template whose **top-left** sits at ``(i, j)``;
       the template **centre** is at ``(i + ctr, j + ctr)``.  All axes are
       labelled by centre position in image pixels.
    *  The GIF is written with ``disposal=1`` and PIL's delta rectangles.  Every
       frame is a complete, opaque canvas, so the composite is exact; with
       ``disposal=2`` the same frames would encode as full frames.

Usage (from the repository root)::

    python blog_figures/alignment_matching/make_figures.py
    python blog_figures/alignment_matching/make_figures.py \\
        --image "data/bench/captured/qr_code_ - 31.jpeg" --outdir /tmp/figs

The script is deterministic (Agg backend, fixed palette, no RNG).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for _p in (str(REPO_ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qr_reader.detector.detector import (  # noqa: E402
    DetectionResult,
    _fit_candidate_homography,
    _prepare_detection,
    _rank_triplets,
)
from qr_reader.detector.homography import project_points  # noqa: E402
from qr_reader.detector.landmarks_refined import (  # noqa: E402
    _local_module_px,
    _ncc_match,
    _parabolic_peak,
    alignment_centers_grid,
    alignment_pattern_template,
    match_alignment_pattern,
)

# The data tree is not part of this worktree; the absolute path is deliberate.
DEFAULT_IMAGE = Path(
    "/home/rik/git/qr-reader/data/bench/captured/qr_code_ - 55.jpeg"
)
DEFAULT_OUTDIR = HERE

# ── Shared style ─────────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

INK = (17, 24, 39)
MUTED = (107, 114, 128)
PAPER = (255, 255, 255)
UNSAMPLED = (236, 236, 236)
GRID_CYAN = (0, 170, 220)
CYAN_DARK = (0, 110, 150)
ACCENT = (255, 127, 14)
ACCENT_SOFT = (255, 190, 140)
PEAK_RED = (200, 30, 30)
HAIRLINE = (205, 205, 205)

INK_HEX = "#111827"
MUTED_HEX = "#6b7280"
CYAN_HEX = "#00aadc"
ACCENT_HEX = "#ff7f0e"
PEAK_HEX = "#c81e1e"

CMAP_NAME = "magma"
RAMP_STEPS = 48
GRAY_STEPS = 64

# ── Animation layout ─────────────────────────────────────────────────────────
PANEL = 300
PAD = 16
LABEL_H = 26
FOOTER_H = 38  # axis tick labels + axis title built into the right panel
GUTTER = 72
CBAR_W = 14
CBAR_H = 220
CBAR_GAP = 12
FPS = 12
HOLD_FRAMES = 13
STEP_TARGET = 80  # animated template positions (before the hold)
CROP_HALF_MODULES = 5.5


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


# ---------------------------------------------------------------------------
# Real pipeline + faithful search-window reconstruction
# ---------------------------------------------------------------------------


@dataclass
class MatchRun:
    """Everything the figures need from one real alignment-pattern match."""

    image_path: Path
    gray: np.ndarray
    version: int
    module_px: float
    grid_xy: tuple[float, float]
    seed_xy: np.ndarray
    centre_xy: np.ndarray
    score: float
    integer_xy: np.ndarray
    dx: float
    dy: float
    template: np.ndarray
    window: np.ndarray
    res: np.ndarray
    peak_ij: tuple[int, int]
    x_of_centre: np.ndarray
    y_of_centre: np.ndarray
    vmin: float
    vmax: float
    crop: tuple[int, int, int, int]

    @property
    def offset_px(self) -> float:
        return float(math.hypot(self.dx, self.dy))

    @property
    def integer_distance_px(self) -> float:
        return float(np.linalg.norm(self.integer_xy - self.centre_xy))


def _search_window(
    gray: np.ndarray, center_xy: np.ndarray, module_px: float
) -> tuple[np.ndarray, np.ndarray, float, float, int, int, int, int]:
    """Rebuild the exact window + crop offsets used by ``match_alignment_pattern``.

    This is a line-for-line copy of the production code (search radius of two
    modules, template padding, edge padding); the assertion in :func:`run_pipeline`
    proves it agrees with the real function.
    """
    search_radius_px = 2.0 * module_px
    template = alignment_pattern_template(module_px)
    th, tw = template.shape
    ctr_x = (tw - 1) / 2.0
    ctr_y = (th - 1) / 2.0

    half = int(np.ceil(search_radius_px)) + int(max(ctr_x, ctr_y)) + 1
    cx, cy = float(center_xy[0]), float(center_xy[1])

    h, w = gray.shape
    x0 = int(round(cx - half))
    x1 = int(round(cx + half)) + 1
    y0 = int(round(cy - half))
    y1 = int(round(cy + half)) + 1

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    window = gray[sy0:sy1, sx0:sx1].astype(np.float32)
    if pad_l or pad_t or pad_r or pad_b:
        window = np.pad(window, ((pad_t, pad_b), (pad_l, pad_r)), mode="edge")
    return window, template, ctr_x, ctr_y, x0, y0, pad_l, pad_t


def run_pipeline(image_path: Path) -> MatchRun:
    """Locate the single alignment pattern exactly as production does."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = DetectionResult(success=False)
    ctx = _prepare_detection(gray, result)
    ranked = _rank_triplets(ctx)
    if not ranked:
        raise SystemExit("no triplet candidate")

    candidate = ranked[0]
    version = candidate.best_version
    n_modules = 17 + 4 * version
    H_base, _base_err = _fit_candidate_homography(
        ctx, candidate.triplet, n_modules
    )

    centres = alignment_centers_grid(version)
    if len(centres) != 1:
        raise SystemExit(
            f"expected exactly one alignment pattern, got {len(centres)} "
            f"(version {version})"
        )
    gx, gy = centres[0]
    seed_xy = project_points(H_base, np.array([[gx, gy]]))[0]
    module_px = _local_module_px(H_base, gx, gy)

    centre_xy, score = match_alignment_pattern(ctx.img_gray, seed_xy, module_px)

    # Rebuild the surface from the same window the production call used.
    window, template, ctr_x, ctr_y, x0, y0, pad_l, pad_t = _search_window(
        ctx.img_gray, seed_xy, module_px
    )
    res = _ncc_match(window, template)
    my, mx = (int(v) for v in np.unravel_index(int(np.argmax(res)), res.shape))
    max_val = float(res[my, mx])
    dx, dy = _parabolic_peak(res, mx, my)

    origin_x = (x0 - pad_l) + ctr_x
    origin_y = (y0 - pad_t) + ctr_y
    rebuilt = np.array([origin_x + mx + dx, origin_y + my + dy])
    integer_xy = np.array([origin_x + mx, origin_y + my])

    # The heatmap must literally be what the production call computed.
    if not np.allclose(rebuilt, centre_xy, atol=1e-9):
        raise SystemExit(
            f"window reconstruction disagrees with match_alignment_pattern: "
            f"{rebuilt} vs {centre_xy}"
        )
    if abs(max_val - score) > 1e-9:
        raise SystemExit(f"peak score {max_val} != returned {score}")
    if np.linalg.norm(integer_xy - centre_xy) >= 0.6:
        raise SystemExit(
            "integer argmax is too far from the returned sub-pixel centre: "
            f"{np.linalg.norm(integer_xy - centre_xy):.3f} px"
        )

    nh, nw = res.shape
    x_of_centre = origin_x + np.arange(nw, dtype=np.float64)
    y_of_centre = origin_y + np.arange(nh, dtype=np.float64)

    # Tight crop for the left panel / photo panel (square, centred on the match).
    half = 0.5 * int(round(2.0 * CROP_HALF_MODULES * module_px))
    c0 = int(round(centre_xy[0] - half))
    r0 = int(round(centre_xy[1] - half))
    size = int(round(2.0 * half))
    c1, r1 = c0 + size, r0 + size
    if c0 < 0 or r0 < 0 or c1 > gray.shape[1] or r1 > gray.shape[0]:
        raise SystemExit("subject image is too close to the border for the crop")

    return MatchRun(
        image_path=image_path,
        gray=ctx.img_gray,
        version=version,
        module_px=float(module_px),
        grid_xy=(gx, gy),
        seed_xy=np.asarray(seed_xy, dtype=np.float64),
        centre_xy=np.asarray(centre_xy, dtype=np.float64),
        score=float(score),
        integer_xy=integer_xy,
        dx=float(dx),
        dy=float(dy),
        template=template,
        window=window,
        res=res,
        peak_ij=(my, mx),
        x_of_centre=x_of_centre,
        y_of_centre=y_of_centre,
        vmin=float(res.min()),
        vmax=float(res.max()),
        crop=(c0, r0, c1, r1),
    )


# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------


def _ramp() -> np.ndarray:
    cmap = matplotlib.colormaps[CMAP_NAME]
    return (cmap(np.linspace(0.0, 1.0, RAMP_STEPS))[:, :3] * 255.0).round().astype(
        np.uint8
    )


def _norm(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)


def _rgb_heatmap(res: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    ramp = _ramp()
    idx = np.round(_norm(res, vmin, vmax) * (RAMP_STEPS - 1)).astype(int)
    return ramp[idx]


def _palette(ramp: np.ndarray) -> Image.Image:
    colours: list[tuple[int, int, int]] = [tuple(int(v) for v in c) for c in ramp]
    for v in np.linspace(0, 255, GRAY_STEPS).round().astype(int):
        colours.append((int(v), int(v), int(v)))
    colours += [
        PAPER,
        (0, 0, 0),
        INK,
        MUTED,
        UNSAMPLED,
        GRID_CYAN,
        CYAN_DARK,
        ACCENT,
        ACCENT_SOFT,
        PEAK_RED,
        HAIRLINE,
    ]
    palette = Image.new("P", (1, 1))
    flat = [channel for colour in colours for channel in colour]
    palette.putpalette(flat + [0] * (768 - len(flat)))
    return palette


# ---------------------------------------------------------------------------
# PIL drawing helpers
# ---------------------------------------------------------------------------


def _rotated(text: str, font: ImageFont.FreeTypeFont, fill, pad: int = 3) -> Image.Image:
    """Render *text* rotated by 90 degrees (counter-clockwise)."""
    probe = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text(
        (pad - bbox[0], pad - bbox[1]), text, font=font, fill=(*fill, 255)
    )
    return layer.rotate(90, expand=True)


def _halo_square(draw, cx, cy, half, colour, width):
    box = [cx - half, cy - half, cx + half, cy + half]
    draw.rectangle(box, outline=PAPER, width=width + 2)
    draw.rectangle(box, outline=colour, width=width)


def _halo_cross(draw, cx, cy, arm, colour, width):
    for seg in (
        [(cx - arm, cy), (cx + arm, cy)],
        [(cx, cy - arm), (cx, cy + arm)],
    ):
        draw.line(seg, fill=PAPER, width=width + 2)
        draw.line(seg, fill=colour, width=width)


def _ring(draw, cx, cy, radius, colour, width=2):
    box = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(box, outline=PAPER, width=width + 2)
    draw.ellipse(box, outline=colour, width=width)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def _serpentine(nh: int, nw: int) -> np.ndarray:
    """Raster order: entry ``[i, j]`` is the visit index of cell ``(i, j)``."""
    order = np.empty((nh, nw), dtype=int)
    k = 0
    for i in range(nh):
        js = range(nw) if i % 2 == 0 else range(nw - 1, -1, -1)
        for j in js:
            order[i, j] = k
            k += 1
    return order


def _revealed_rows(order: np.ndarray, i: int, j: int) -> np.ndarray:
    """Cells already swept: every row above, plus the current row's prefix.

    Row-based rather than strictly order-based so the fill edge reads as a
    clean raster instead of a ragged band at the direction reversals.
    """
    mask = np.zeros_like(order, dtype=bool)
    mask[:i, :] = True
    if i % 2 == 0:
        mask[i, : j + 1] = True
    else:
        mask[i, j:] = True
    return mask


def _tick_values(lo: float, hi: float, step: float = 20.0) -> list[float]:
    start = math.ceil(lo / step) * step
    vals = []
    v = start
    while v <= hi + 1e-9:
        vals.append(float(v))
        v += step
    return vals


def _draw_left_panel(
    canvas: Image.Image,
    base: Image.Image,
    run: MatchRun,
    tx_img: float,
    ty_img: float,
    score: float,
    final: bool,
    fonts: dict,
) -> None:
    """Crop with the template outline + crosshair + score readout."""
    c0, r0, c1, _r1 = run.crop
    scale = PANEL / float(c1 - c0)
    px = (tx_img - c0) * scale
    py = (ty_img - r0) * scale
    m = run.module_px * scale  # one module, in panel pixels

    panel = base.copy()
    draw = ImageDraw.Draw(panel)

    # Template footprint (7 x 7 modules) then the three dark-ring boundaries.
    _halo_square(draw, px, py, 3.5 * m, (225, 225, 225), 1)
    _halo_square(draw, px, py, 2.5 * m, ACCENT, 2)
    _halo_square(draw, px, py, 1.5 * m, ACCENT_SOFT, 1)
    _halo_square(draw, px, py, 0.5 * m, ACCENT, 2)
    _halo_cross(draw, px, py, 0.85 * m, GRID_CYAN, 2)

    label = f"score = {score:.3f}" + ("  ·  peak" if final else "")
    tw = draw.textlength(label, font=fonts["score"])
    bx0, by0 = 10, PANEL - 44
    draw.rounded_rectangle(
        [bx0, by0, bx0 + tw + 18, by0 + 33],
        radius=7,
        fill=PAPER,
        outline=HAIRLINE,
    )
    draw.text((bx0 + 9, by0 + 6), label, font=fonts["score"], fill=INK)

    canvas.paste(panel, (PAD, PAD + LABEL_H))


def _draw_right_panel(
    canvas: Image.Image,
    run: MatchRun,
    revealed: np.ndarray,
    cur_ij: tuple[int, int],
    show_peak: bool,
    fonts: dict,
) -> None:
    nh, nw = run.res.shape
    rgb = _rgb_heatmap(run.res, run.vmin, run.vmax)
    rgb = np.where(revealed[:, :, None], rgb, np.array(UNSAMPLED, dtype=np.uint8))

    heat = Image.fromarray(rgb, mode="RGB").resize((PANEL, PANEL), Image.NEAREST)
    x0 = PAD + PANEL + GUTTER
    y0 = PAD + LABEL_H
    canvas.paste(heat, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x0, y0, x0 + PANEL, y0 + PANEL], outline=HAIRLINE, width=1)

    # Axes: centre positions in image pixels, not array indices.
    cell_x = PANEL / nw
    cell_y = PANEL / nh
    tick_y = y0 + PANEL + 5
    for v in _tick_values(
        float(run.x_of_centre[0]), float(run.x_of_centre[-1]), step=10.0
    ):
        cx = x0 + (v - run.x_of_centre[0] + 0.5) * cell_x
        draw.line([cx, y0 + PANEL, cx, y0 + PANEL + 3], fill=MUTED, width=1)
        draw.text(
            (cx, tick_y), f"{v:.0f}", font=fonts["tick"], fill=MUTED, anchor="ma"
        )
    for v in _tick_values(
        float(run.y_of_centre[0]), float(run.y_of_centre[-1]), step=10.0
    ):
        cy = y0 + (v - run.y_of_centre[0] + 0.5) * cell_y
        draw.line([x0 - 3, cy, x0, cy], fill=MUTED, width=1)
        draw.text(
            (x0 - 6, cy), f"{v:.0f}", font=fonts["tick"], fill=MUTED, anchor="rm"
        )

    draw.text(
        (x0 + PANEL / 2, tick_y + 17),
        "template centre · image x (px)",
        font=fonts["axis"],
        fill=MUTED,
        anchor="ma",
    )
    title = _rotated("template centre · image y (px)", fonts["axis"], MUTED)
    canvas.paste(title, (x0 - 56, y0 + int(PANEL / 2 - title.height / 2)), title)

    # Colour bar (top = vmax, i.e. the colormap's bright end).
    bar_x = x0 + PANEL + CBAR_GAP
    bar_y = y0 + (PANEL - CBAR_H) // 2
    ramp = _ramp()
    bar = Image.fromarray(ramp[::-1][:, None, :].repeat(CBAR_H // RAMP_STEPS + 1, 1))
    bar = bar.resize((CBAR_W, CBAR_H), Image.NEAREST)
    canvas.paste(bar, (bar_x, bar_y))
    draw.rectangle([bar_x, bar_y, bar_x + CBAR_W, bar_y + CBAR_H], outline=HAIRLINE)
    for v in [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6]:
        if not (run.vmin - 1e-9 <= v <= run.vmax + 1e-9):
            continue
        frac = float(_norm(np.array([v]), run.vmin, run.vmax)[0])
        ty = bar_y + (1.0 - frac) * CBAR_H
        draw.line([bar_x + CBAR_W, ty, bar_x + CBAR_W + 4, ty], fill=MUTED)
        draw.text(
            (bar_x + CBAR_W + 7, ty), f"{v:+.1f}" if v else "0.0",
            font=fonts["tick"], fill=MUTED, anchor="lm",
        )
    cbar_label = _rotated("Pearson correlation", fonts["axis"], MUTED)
    canvas.paste(
        cbar_label,
        (bar_x + CBAR_W + 42, y0 + int(PANEL / 2 - cbar_label.height / 2)),
        cbar_label,
    )

    # Current template position.
    ci, cj = cur_ij
    hx = x0 + (cj + 0.5) * cell_x
    hy = y0 + (ci + 0.5) * cell_y
    _ring(draw, hx, hy, 5, GRID_CYAN, 2)
    _ring(draw, hx, hy, 2, PAPER, 1)

    if show_peak:
        my, mx = run.peak_ij
        px = x0 + (mx + 0.5) * cell_x
        py = y0 + (my + 0.5) * cell_y
        draw.ellipse(
            [px - 8, py - 8, px + 8, py + 8], outline=PAPER, width=4
        )
        draw.ellipse(
            [px - 8, py - 8, px + 8, py + 8], outline=PEAK_RED, width=2
        )
        draw.line([px - 12, py, px + 12, py], fill=PEAK_RED, width=1)
        draw.line([px, py - 12, px, py + 12], fill=PEAK_RED, width=1)
        label = f"peak {run.score:.3f}"
        tw = draw.textlength(label, font=fonts["small_bold"])
        lx = min(px + 12, x0 + PANEL - tw - 16)
        ly = max(py - 30, y0 + 6)
        draw.rounded_rectangle(
            [lx - 5, ly - 3, lx + tw + 5, ly + 18],
            radius=5,
            fill=PAPER,
            outline=PEAK_RED,
        )
        draw.text((lx, ly), label, font=fonts["small_bold"], fill=PEAK_RED)


def make_animation(run: MatchRun, out_path: Path) -> tuple[int, int]:
    """Render the GIF; returns ``(logical frames, stored frames)``."""
    nh, nw = run.res.shape
    order = _serpentine(nh, nw)
    flat = np.argsort(order, axis=None)
    ij_of_order = np.stack(np.unravel_index(flat, order.shape), axis=1)

    step = max(1, int(round(nh * nw / STEP_TARGET)))
    # Start once the first row has been swept, so frame 0 already shows a
    # filled band rather than a blank panel.
    positions = [k for k in range(0, nh * nw, step) if k >= nw]
    peak_order = int(order[run.peak_ij])
    if positions[-1] != peak_order:
        positions.append(peak_order)

    # The surface is fully computed in one FFT call; the reveal is only a
    # reading aid, so the filled raster leads the template by one row.

    # Both panels: left crop base and fixed labels.
    c0, r0, c1, r1 = run.crop
    # The crop is upscaled (about 3x at this module pitch); a light unsharp
    # mask only restores edge contrast lost to the interpolation.
    base = (
        Image.fromarray(run.gray[r0:r1, c0:c1])
        .convert("L")
        .resize((PANEL, PANEL), Image.Resampling.LANCZOS)
        .filter(ImageFilter.UnsharpMask(radius=1.1, percent=110, threshold=3))
        .convert("RGB")
    )

    fonts = {
        "score": _font(18, bold=True),
        "tick": _font(11),
        "axis": _font(12),
        "small_bold": _font(11, bold=True),
        "label": _font(14, bold=True),
    }

    width = PAD + PANEL + GUTTER + PANEL + CBAR_GAP + CBAR_W + 66 + PAD
    height = PAD + LABEL_H + PANEL + FOOTER_H + PAD

    def render(order_index: int, show_peak: bool, full: bool) -> Image.Image:
        canvas = Image.new("RGB", (width, height), PAPER)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (PAD, PAD - 3), "image crop · template outline",
            font=fonts["label"], fill=INK,
        )
        draw.text(
            (PAD + PANEL + GUTTER, PAD - 3), "correlation surface",
            font=fonts["label"], fill=INK,
        )

        i, j = (int(v) for v in ij_of_order[order_index])
        score = float(run.res[i, j])
        _draw_left_panel(
            canvas, base, run,
            float(run.x_of_centre[j]), float(run.y_of_centre[i]),
            score, show_peak, fonts,
        )
        revealed = (
            np.ones_like(order, dtype=bool)
            if full
            else _revealed_rows(order, i, j)
        )
        _draw_right_panel(canvas, run, revealed, (i, j), show_peak, fonts)
        return canvas

    frames = [render(k, False, False) for k in positions[:-1]]
    frames.append(render(positions[-1], True, True))
    frames.extend([frames[-1]] * (HOLD_FRAMES - 1))

    palette = _palette(_ramp())
    quantised = [
        f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Every frame is a complete, opaque canvas, so disposal=1 plus the delta
    # rectangles that ``optimize`` writes is exact and keeps the file small.
    # (disposal=2 forces a full-frame encode: ~18 MB for the same animation.)
    quantised[0].save(
        out_path,
        save_all=True,
        append_images=quantised[1:],
        duration=round(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )
    with Image.open(out_path) as written:
        stored = written.n_frames
    return len(frames), stored


# ---------------------------------------------------------------------------
# Static figure
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors=MUTED_HEX, labelsize=8.5, length=3, width=0.8)


def _label_profile_samples(ax: plt.Axes, xs, ys) -> None:
    """4 d.p. value labels on the three profile samples.

    Placed outside the parabola — above-left, below-centre, above-right — so
    nothing collides with the curve, the markers, or the dx/dy annotation.
    """
    placements = (
        (-8, 9, "right", "bottom"),
        (-10, -13, "right", "top"),
        (8, 9, "left", "bottom"),
    )
    for x, y, (ox, oy, ha, va) in zip(xs, ys, placements):
        ax.annotate(
            f"{y:.4f}",
            xy=(x, y), xytext=(ox, oy), textcoords="offset points",
            ha=ha, va=va, fontsize=7.6, color=MUTED_HEX, zorder=8,
        )


def make_static(
    run: MatchRun, out_path: Path
) -> tuple[tuple[int, int], float, float, float, float]:
    """Render the two-parabola figure.

    Returns ``(size, dx labels, dy labels, vertex y horizontal, vertex y
    vertical)``.
    """
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.edgecolor": "#9ca3af",
            "text.color": INK_HEX,
            "axes.labelcolor": INK_HEX,
            "xtick.color": MUTED_HEX,
            "ytick.color": MUTED_HEX,
        }
    )

    res = run.res
    my, mx = run.peak_ij

    fig = plt.figure(figsize=(10.0, 4.2), dpi=200, facecolor="white")
    gs = fig.add_gridspec(
        1, 2, left=0.062, right=0.985, top=0.865, bottom=0.155, wspace=0.19
    )
    fig.suptitle(
        "Sub-pixel centre from two separable 1-D parabolas",
        fontsize=13.5, fontweight="bold", color=INK_HEX, y=0.972,
    )

    def parabola(f: np.ndarray) -> tuple[float, float, float]:
        c = float(f[1])
        b = 0.5 * (f[2] - f[0])
        a = 0.5 * (f[2] + f[0]) - c
        return a, b, c

    t = np.linspace(-1.2, 1.2, 161)

    # ── horizontal profile ─────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[0, 0])
    fh = res[my, mx - 1 : mx + 2].astype(np.float64)
    a, b, c = parabola(fh)
    curve_h = a * t**2 + b * t + c
    ax_h.plot(run.integer_xy[0] + t, curve_h, color=ACCENT_HEX, lw=2.0, zorder=3)
    ax_h.plot(
        [run.integer_xy[0] - 1, run.integer_xy[0], run.integer_xy[0] + 1],
        fh, "o", ms=7, mfc="white", mec=ACCENT_HEX, mew=1.8, zorder=4,
    )
    _label_profile_samples(
        ax_h, run.integer_xy[0] + np.array([-1.0, 0.0, 1.0]), fh
    )

    vertex = float(_parabolic_peak(res, mx, my)[0])
    if not np.isclose(vertex, run.dx):
        raise SystemExit(f"plotted dx {vertex} != _parabolic_peak {run.dx}")
    # The diamond marks the *vertex of the fitted parabola*, so its height is
    # the parabola evaluated at the vertex abscissa -- not the integer sample
    # ``run.score`` (which is only the centre sample, f[1]).
    vertex_y_h = float(a * run.dx**2 + b * run.dx + c)
    ax_h.plot(
        [run.integer_xy[0] + run.dx], [vertex_y_h],
        marker="D", ms=7, mfc=CYAN_HEX, mec="white", mew=0.9, zorder=5,
    )
    ax_h.set_xticks(
        [run.integer_xy[0] - 1, run.integer_xy[0], run.integer_xy[0] + 1]
    )
    ax_h.set_xlabel("template centre · x (px)")
    ax_h.set_ylabel("Pearson correlation")
    _style_axes(ax_h)

    # ── vertical profile ───────────────────────────────────────────────────
    ax_v = fig.add_subplot(gs[0, 1])
    fv = res[my - 1 : my + 2, mx].astype(np.float64)
    a, b, c = parabola(fv)
    curve_v = a * t**2 + b * t + c
    ax_v.plot(run.integer_xy[1] + t, curve_v, color=ACCENT_HEX, lw=2.0, zorder=3)
    ax_v.plot(
        [run.integer_xy[1] - 1, run.integer_xy[1], run.integer_xy[1] + 1],
        fv, "o", ms=7, mfc="white", mec=ACCENT_HEX, mew=1.8, zorder=4,
    )
    _label_profile_samples(
        ax_v, run.integer_xy[1] + np.array([-1.0, 0.0, 1.0]), fv
    )

    vertex_y = float(_parabolic_peak(res, mx, my)[1])
    if not np.isclose(vertex_y, run.dy):
        raise SystemExit(f"plotted dy {vertex_y} != _parabolic_peak {run.dy}")
    vertex_y_v = float(a * run.dy**2 + b * run.dy + c)
    ax_v.plot(
        [run.integer_xy[1] + run.dy], [vertex_y_v],
        marker="D", ms=7, mfc=CYAN_HEX, mec="white", mew=0.9, zorder=5,
    )
    ax_v.set_xticks(
        [run.integer_xy[1] - 1, run.integer_xy[1], run.integer_xy[1] + 1]
    )
    ax_v.set_xlabel("template centre · y (px)")
    ax_v.set_ylabel("Pearson correlation")
    _style_axes(ax_v)

    # Identical y-limits (union of both curves, same headroom).
    lo = float(min(curve_h.min(), curve_v.min()))
    hi = float(max(curve_h.max(), curve_v.max(), vertex_y_h, vertex_y_v))
    span = hi - lo
    for _ax in (ax_h, ax_v):
        _ax.set_ylim(lo - 0.10 * span, hi + 0.90 * span)

    # Dotted vertex lines only rise to the vertex, so they never cut through
    # the annotations; then label the interpolated coordinate (not an offset).
    lo_lim, hi_lim = ax_h.get_ylim()
    for _ax, _x, _y, _text in (
        (
            ax_h,
            run.integer_xy[0] + run.dx,
            vertex_y_h,
            f"centre x = {run.centre_xy[0]:.2f} px",
        ),
        (
            ax_v,
            run.integer_xy[1] + run.dy,
            vertex_y_v,
            f"centre y = {run.centre_xy[1]:.2f} px",
        ),
    ):
        _ax.vlines(
            _x, lo_lim, _y, color=CYAN_HEX, ls=":", lw=1.3, zorder=2
        )
        _ax.annotate(
            _text,
            xy=(_x, _y),
            xytext=(0, 30), textcoords="offset points",
            ha="center", va="bottom", fontsize=10.0, color=INK_HEX,
            arrowprops=dict(arrowstyle="-", color=MUTED_HEX, lw=0.8),
        )

    # The diamond must sit on the curve, at the apex -- not at the integer
    # sample.  The parabola opens downward, so its vertex is never below the
    # centre sample f[1] == run.score, and it must equal the curve's maximum.
    for _name, _y_v, _curve in (
        ("x", vertex_y_h, curve_h),
        ("y", vertex_y_v, curve_v),
    ):
        if _y_v < run.score - 1e-12:
            raise SystemExit(
                f"vertex {_name} {_y_v:.6f} below centre sample {run.score:.6f}"
            )
        _peak = float(_curve.max())
        if abs(_y_v - _peak) > 1e-4:
            raise SystemExit(
                f"vertex {_name} {_y_v:.6f} off the plotted curve (max {_peak:.6f})"
            )

    # The panels annotate the samples to 4 d.p.; check that those printed
    # values really do reproduce the vertices, so the figure is self-checking.
    def delta_from_labels(f: np.ndarray) -> float:
        r = np.round(f, 4)
        return float(0.5 * (r[0] - r[2]) / (r[0] - 2.0 * r[1] + r[2]))

    dx_labels = delta_from_labels(fh)
    dy_labels = delta_from_labels(fv)
    for name, approx, exact in (
        ("dx", dx_labels, run.dx),
        ("dy", dy_labels, run.dy),
    ):
        if abs(approx - exact) > 2e-3:
            raise SystemExit(
                f"{name} recomputed from the 4 d.p. labels ({approx:+.4f}) "
                f"differs from _parabolic_peak ({exact:+.4f})"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    with Image.open(out_path) as im:
        return im.size, dx_labels, dy_labels, vertex_y_h, vertex_y_v

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    run = run_pipeline(args.image)
    gif_path = args.outdir / "alignment-matching.gif"
    png_path = args.outdir / "alignment-subpixel.png"

    frames, stored = make_animation(run, gif_path)
    png_size, dx_labels, dy_labels, vertex_y_h, vertex_y_v = make_static(
        run, png_path
    )

    print(f"image            : {run.image_path.name}")
    print(f"version          : {run.version}")
    print(f"module pitch     : {run.module_px:.3f} px")
    print(f"alignment grid   : ({run.grid_xy[0]:g}, {run.grid_xy[1]:g})")
    print(
        f"integer peak     : ({run.integer_xy[0]:.3f}, {run.integer_xy[1]:.3f})"
        f"  [res index {run.peak_ij}]"
    )
    print(
        f"sub-pixel centre : ({run.centre_xy[0]:.3f}, {run.centre_xy[1]:.3f})"
    )
    print(f"dx, dy           : ({run.dx:+.4f}, {run.dy:+.4f})")
    print(
        f"labels -> vertex : dx {dx_labels:+.4f} px, dy {dy_labels:+.4f} px "
        f"(from the 4 d.p. samples)"
    )
    print(f"offset magnitude : {run.offset_px:.4f} px")
    print(f"integer->subpix  : {run.integer_distance_px:.4f} px")
    print(f"score            : {run.score:.4f}")
    print(
        f"vertex y         : x-fit {vertex_y_h:.4f}, y-fit {vertex_y_v:.4f} "
        f"(diamond heights; curve apex, not the integer sample)"
    )
    print(
        f"surface          : {run.res.shape[0]}x{run.res.shape[1]} cells, "
        f"range [{run.vmin:+.3f}, {run.vmax:+.3f}]"
    )
    print(
        f"gif              : {frames} frames ({stored} stored; identical hold "
        f"frames merged), {gif_path.stat().st_size} bytes"
    )
    print(f"png              : {png_size[0]}x{png_size[1]} px, "
          f"{png_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
