"""Shared setup for the "From pixels to candidates" blog figures.

The two figures (``rle-scan.gif`` and ``candidate-clustering.gif``) use the
same captured photo, the same crop, the same Otsu-thresholded binary image and
the same colour palette, so they read as one pair.

Data preparation mirrors the detector:

* crop the QR plus a margin, resize to :data:`TARGET_H` pixels tall,
* Otsu-threshold with :func:`qr_reader.qr_gen.binarize_image`,
* collect 1:1:3:1:1 row matches with
  :func:`qr_reader.detector.alignment.find_alignment_patterns` at the
  detector's 30% tolerance (``log(1.3)``),
* merge them with the real greedy rules from
  :mod:`qr_reader.detector.clustering`,
* reject clusters whose centre column has no vertical 1:1:3:1:1 match, using
  the same :func:`find_alignment_patterns` test as the detector.

Everything here is deterministic.  The only random choice in the pipeline is
the reference-candidate order inside ``cluster_candidates``, which seeds
``numpy.random.default_rng(0)``; the local merge animation re-runs that exact
loop so it can show memberships.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.alignment import find_alignment_patterns  # noqa: E402
from qr_reader.detector.clustering import (  # noqa: E402
    CandidateCluster,
    candidate_overlaps,
    cluster_candidates,
    merge_candidates,
)
from qr_reader.qr_gen import binarize_image  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
OUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

# The clearest photo with three strong finder clusters and two genuine
# row-scan false positives that the vertical check rejects.
IMAGE_NAME = "qr_code_ - 99.jpeg"
QR_CROP_SCALE = 1.44  # square crop side as a multiple of the QR's longest extent
TARGET_H = 600
MAX_ERROR = float(np.log(1.3))  # detector tolerance: each run within 30%

# ── Shared palette ──────────────────────────────────────────────────────────
# Binary image: two saturated, high-contrast colours, never plain black/white.
BINARY_DARK = (28, 33, 54)  # #1c2136  (binary False)
BINARY_LIGHT = (246, 241, 218)  # #f6f1da  (binary True)
SCAN = (0, 220, 255)  # #00dcff  scanline / geometry
CAND = (255, 120, 0)  # #ff7800  RLE candidate
CLUSTER = (255, 0, 128)  # #ff0080  merged candidate cluster
PASS = (0, 220, 100)  # #00dc64  vertical check passed
FAIL = (255, 40, 40)  # #ff2828  vertical check failed
INK = (18, 20, 30)  # dark outline for badges

FPS = 25
FRAME_MS = 1000 // FPS
HOLD_MS = 1000


@dataclass(frozen=True)
class ClusterGroup:
    """One merged cluster plus the raw candidates that formed it."""

    cluster: CandidateCluster
    members: tuple[int, ...]

    @property
    def centre_x(self) -> float:
        return float((self.cluster.cols[2] + self.cluster.cols[3]) / 2)

    @property
    def box(self) -> tuple[float, float, float, float]:
        c = self.cluster
        return (
            float(c.cols[0]),
            float(c.row - c.height),
            float(c.cols[5]),
            float(c.row + c.height),
        )


@dataclass(frozen=True)
class CandidateData:
    image_name: str
    crop: np.ndarray  # uint8 grayscale, TARGET_H tall
    binary: np.ndarray  # bool, True = light
    rows: np.ndarray  # (N,) candidate row indices
    cols: np.ndarray  # (N, 6) transition columns
    groups: tuple[ClusterGroup, ...]  # sorted top-to-bottom
    vertical_pass: tuple[bool, ...]  # aligned with groups


def _locate_qr(image: np.ndarray) -> np.ndarray:
    """Return the four QR corners in full-resolution image coordinates."""
    shrink = 1200.0 / max(image.shape)
    small = cv2.resize(
        image, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA
    )
    found, points = cv2.QRCodeDetector().detect(small)
    if not found or points is None:
        raise RuntimeError("cv2.QRCodeDetector did not find a QR code")
    return points.reshape(-1, 2) / shrink


def _group_candidates(rows: np.ndarray, cols: np.ndarray) -> list[ClusterGroup]:
    """Re-run the greedy merge loop, keeping track of which candidates merge.

    The loop is intentionally identical to
    :func:`qr_reader.detector.clustering.cluster_candidates` (same reference
    selection, same overlap test, same weighted merge); only membership
    bookkeeping and the final top-to-bottom sort are added.
    """
    rows_c = np.asarray(rows, dtype=np.float32)
    cols_c = np.asarray(cols, dtype=np.float32)
    heights = np.asarray(cols_c[:, 3] - cols_c[:, 2], dtype=np.float32)
    n = rows_c.shape[0]

    rng = np.random.default_rng(0)
    processed = np.zeros(n, dtype=bool)
    groups: list[ClusterGroup] = []

    while not processed.all():
        free = np.where(~processed)[0]
        ref_index = rng.choice(free)
        processed[ref_index] = True
        reference = CandidateCluster(
            rows_c[ref_index], cols_c[ref_index], heights[ref_index], 1
        )
        members = [int(ref_index)]
        for i in range(n):
            if processed[i]:
                continue
            candidate = CandidateCluster(
                rows_c[i], cols_c[i], heights[i], 1
            )
            if candidate_overlaps(reference, candidate):
                reference = merge_candidates(reference, candidate)
                processed[i] = True
                members.append(int(i))
        groups.append(ClusterGroup(reference, tuple(members)))

    expected = cluster_candidates(rows, cols)
    assert len(groups) == len(expected), "local merge diverged from detector"
    for group, cluster in zip(
        sorted(groups, key=lambda g: g.cluster.row), sorted(expected, key=lambda c: c.row)
    ):
        assert np.allclose(group.cluster.cols, cluster.cols)
        assert np.isclose(group.cluster.row, cluster.row)
        assert group.cluster.num_candidates == cluster.num_candidates

    groups.sort(key=lambda g: g.cluster.row)
    return groups


def _vertical_check(binary: np.ndarray, cluster: CandidateCluster) -> bool:
    """Detector-style vertical RLE test at the cluster's centre column."""
    x = int(round((cluster.cols[2] + cluster.cols[3]) / 2))
    x = int(np.clip(x, 0, binary.shape[1] - 1))
    _, vertical_cols = find_alignment_patterns(
        binary[:, x][None, :], MAX_ERROR
    )
    return any(bound[2] <= cluster.row <= bound[3] for bound in vertical_cols)


def load_candidate_data(
    image_name: str = IMAGE_NAME, *, crop_scale: float = QR_CROP_SCALE
) -> CandidateData:
    path = DATA_DIR / image_name
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)

    corners = _locate_qr(image)
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    centre_x, centre_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * crop_scale / 2.0

    left = max(0, int(centre_x - half))
    right = min(image.shape[1], int(centre_x + half))
    top = max(0, int(centre_y - half))
    bottom = min(image.shape[0], int(centre_y + half))
    crop = image[top:bottom, left:right]

    crop = cv2.resize(
        crop,
        (TARGET_H, TARGET_H),
        interpolation=cv2.INTER_AREA,
    )
    binary = binarize_image(crop)

    rows, cols = find_alignment_patterns(binary, MAX_ERROR)
    groups = _group_candidates(rows, cols)
    vertical_pass = tuple(_vertical_check(binary, g.cluster) for g in groups)
    return CandidateData(
        image_name=image_name,
        crop=crop,
        binary=binary,
        rows=np.asarray(rows),
        cols=np.asarray(cols),
        groups=tuple(groups),
        vertical_pass=vertical_pass,
    )


# ── Rendering helpers ───────────────────────────────────────────────────────


def binary_base(data: CandidateData) -> Image.Image:
    """Recolour the binary image with the shared two-colour palette."""
    rgb = np.where(
        data.binary[:, :, None], BINARY_LIGHT, BINARY_DARK
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def rgba(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int, int]:
    alpha = int(round(np.clip(alpha, 0.0, 1.0) * 255))
    return (color[0], color[1], color[2], alpha)


def composite(
    canvas: Image.Image, draw_calls
) -> Image.Image:
    """Composite a sequence of ``fn(draw)`` calls in order.

    Each call gets its own transparent layer.  Drawing several translucent
    shapes onto one shared RGBA layer would overwrite pixels instead of
    blending them, which makes faint fills erase earlier marks.
    """
    for call in draw_calls:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        pen = ImageDraw.Draw(overlay)
        call(pen)
        canvas = Image.alpha_composite(canvas, overlay)
    return canvas


def candidate_segment(
    cols: np.ndarray, row: float, color: tuple[int, int, int], alpha: float, width: int = 4
):
    """A horizontal segment spanning a raw candidate's six boundaries."""

    def draw(pen: ImageDraw.ImageDraw) -> None:
        pen.line(
            [(float(cols[0]), float(row)), (float(cols[5]), float(row))],
            fill=rgba(color, alpha),
            width=width,
        )

    return draw


def candidate_box(
    row: float,
    cols: np.ndarray,
    height: float,
    color: tuple[int, int, int],
    alpha: float,
    width: int = 2,
    fill_alpha: float = 0.0,
):
    """The box implied by one RLE match (same geometry as the detector)."""
    x0, x1 = float(cols[0]), float(cols[5])
    y0, y1 = float(row - height), float(row + height)

    def draw(pen: ImageDraw.ImageDraw) -> None:
        if fill_alpha > 0:
            pen.rectangle(
                (x0, y0, x1, y1), fill=rgba(color, fill_alpha)
            )
        pen.rectangle(
            (x0, y0, x1, y1),
            outline=rgba(color, alpha),
            width=width,
        )

    return draw


def cluster_box(
    box: tuple[float, float, float, float],
    color: tuple[int, int, int],
    alpha: float,
    width: int = 4,
    fill_alpha: float = 0.0,
    radius: float = 6.0,
):
    x0, y0, x1, y1 = box

    def draw(pen: ImageDraw.ImageDraw) -> None:
        if fill_alpha > 0:
            pen.rounded_rectangle(
                (x0, y0, x1, y1),
                radius=radius,
                fill=rgba(color, fill_alpha),
            )
        pen.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=radius,
            outline=rgba(color, alpha),
            width=width,
        )

    return draw


def vertical_line(
    x: float,
    y0: float,
    y1: float,
    color: tuple[int, int, int],
    alpha: float,
    width: int = 3,
):
    def draw(pen: ImageDraw.ImageDraw) -> None:
        pen.line(
            [(x, y0), (x, y1)],
            fill=rgba(color, alpha),
            width=width,
        )

    return draw


def verdict_badge(
    x: float,
    y: float,
    passed: bool,
    alpha: float,
    radius: float = 12.0,
):
    """Small green check / red cross badge, legible at mobile size."""
    color = PASS if passed else FAIL

    def draw(pen: ImageDraw.ImageDraw) -> None:
        pen.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=rgba(color, alpha),
            outline=rgba(BINARY_LIGHT, min(1.0, alpha + 0.1)),
            width=2,
        )
        if passed:
            pen.line(
                [
                    (x - 5, y),
                    (x - 1, y + 4),
                    (x + 6, y - 5),
                ],
                fill=rgba(INK, alpha),
                width=3,
            )
        else:
            pen.line(
                [(x - 5, y - 5), (x + 5, y + 5)],
                fill=rgba(INK, alpha),
                width=3,
            )
            pen.line(
                [(x + 5, y - 5), (x - 5, y + 5)],
                fill=rgba(INK, alpha),
                width=3,
            )

    return draw


# ── Animation and saving ────────────────────────────────────────────────────


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_box(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    return tuple(lerp(a, b, t) for a, b in zip(box_a, box_b))


def _blend(
    background: tuple[int, int, int],
    foreground: tuple[int, int, int],
    alpha: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(bg + (fg - bg) * alpha))
        for bg, fg in zip(background, foreground)
    )


def build_shared_palette(colors: int = 128) -> Image.Image:
    """Build one fixed palette shared by both figures.

    The figures are flat-colour drawings: the binary image has two colours and
    every overlay is one accent blended at some alpha over one of those two
    backgrounds.  Enumerating those blends directly keeps the accents vivid
    (an adaptive palette built from thin lines tends to wash them out) and
    guarantees that both GIFs use exactly the same colours.
    """
    palette_colors: list[tuple[int, int, int]] = [
        BINARY_DARK,
        BINARY_LIGHT,
        INK,
    ]
    accents = (SCAN, CAND, CLUSTER, PASS, FAIL)
    alphas = (0.08, 0.16, 0.25, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0)
    for accent in accents:
        for alpha in alphas:
            for background in (BINARY_DARK, BINARY_LIGHT):
                palette_colors.append(_blend(background, accent, alpha))
    # A few ink blends for the scanline halo and badge outlines.
    for alpha in (0.25, 0.45, 0.65):
        for background in (BINARY_DARK, BINARY_LIGHT):
            palette_colors.append(_blend(background, INK, alpha))

    if len(palette_colors) > colors:
        raise ValueError("palette has more colours than requested")
    palette_colors.extend(
        [BINARY_DARK] * (colors - len(palette_colors))
    )

    palette = Image.new("P", (16, 8))
    flat: list[int] = []
    for color in palette_colors:
        flat.extend(color)
    flat.extend([0] * (768 - len(flat)))
    palette.putpalette(flat)
    return palette


def save_gif(
    path: Path,
    frames: list[Image.Image],
    palette: Image.Image,
    *,
    last_hold_ms: int = HOLD_MS,
) -> None:
    quantized = [
        frame.convert("RGB").quantize(
            palette=palette, dither=Image.Dither.NONE
        )
        for frame in frames
    ]
    durations = [FRAME_MS] * len(quantized)
    durations[-1] = last_hold_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
