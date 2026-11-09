#!/usr/bin/env python3
"""Build the flood-fill success/failure GIFs for the QR-reader blog post.

The figure recreates the abandoned corner finder from git history
(``f5451e8^:src/qr_reader/scripts/full-pipeline.py``):

1. seed the first dark run of the 1:1:3:1:1 scan (the outer black ring),
2. grow the 8-connected dark region with a wave front,
3. trace its boundary and split it into connected components,
4. take four angular-NMS corners per boundary component.

Both examples come from the same captured photo so lighting, focus and JPEG
noise are shared; only the finder pattern changes.

Success
    ``data/bench/captured-fullsize/qr_code_ - 6.jpeg``, cluster 0.
    The mask fills exactly the outer ring, the boundary has two components
    (outer and inner edge), and four corners per component land on the ring.
Failure
    Same photo, cluster 2.
    The mask is still a clean ring.  But this finder is strongly sheared by
    perspective, and "four farthest points from the boundary centroid" collapses
    onto the two acute corners: one component picks top-left, bottom-left and
    bottom-right twice; the other picks top-left twice and bottom-right twice.
    The other two corners are never chosen.  This is the failure the old
    approach could not fix without changing the corner rule itself.

Usage (from the repository root):

    .venv/bin/python blog_figures/flood_fill/make_figures.py

The script is deterministic: it re-runs the alignment scan and clustering with
their fixed RNG seed and selects the two clusters by index.  It writes

    blog/src/assets/blog/qr-reader/flood-fill-success.gif
    blog/src/assets/blog/qr-reader/flood-fill-failure.gif
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import CandidateCluster, cluster_candidates
from qr_reader.detector.corner import angular_nms_top_radial_indices
from qr_reader.detector.region import (
    boundary_connected_components_ndimage,
    region_boundary_8,
    region_fill_wave_front,
)
from qr_reader.qr_gen import binarize_image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

IMAGE_NAME = "qr_code_ - 6.jpeg"
SUCCESS_CLUSTER = 0
FAILURE_CLUSTER = 2
CROP_SCALE = 1.9  # square crop side as a multiple of the cluster width
OUT_SIZE = 560  # final square GIF side in pixels
FPS = 12
COLORS = 64

# Shared palette (RGB); kept identical for the two GIFs.
MASK_GREEN = np.array([56, 214, 116], dtype=np.float32)
BOUNDARY_ORANGE = (255, 132, 32)  # RGB
CORNER_YELLOW = (255, 215, 0)  # RGB #ffd700
QUAD_MAGENTA = (255, 40, 214)  # RGB #ff28d6
CORNER_OUTLINE = (20, 20, 20)

ANGULAR_NMS_RAD = 10 * 2 * math.pi / 360

# Frame schedule, in frames at FPS.  Identical for both GIFs so they loop in sync.
N_GROWTH = 18
N_BOUNDARY = 10
N_CORNERS = 13
N_HOLD = 5
N_FRAMES = N_GROWTH + N_BOUNDARY + N_CORNERS + N_HOLD
GROWTH_END = N_GROWTH - 1
BOUNDARY_START = N_GROWTH
BOUNDARY_END = BOUNDARY_START + N_BOUNDARY - 1
CORNERS_START = BOUNDARY_END + 1


@dataclass
class Example:
    name: str
    image: str
    cluster: CandidateCluster
    cluster_index: int
    crop_box: tuple[int, int, int, int]
    gray: np.ndarray  # cropped grayscale photo
    region: np.ndarray  # final flood-fill mask, cropped
    boundary: np.ndarray
    components: list[np.ndarray]  # boundary components, largest first
    corners: list[np.ndarray]  # four (row, col) points per component
    depth: np.ndarray  # geodesic depth from the seed inside the region
    boundary_depth: np.ndarray
    seed: tuple[int, int]  # (row, col) in the crop


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return 3.0 * x * x - 2.0 * x * x * x


def geodesic_depth(region: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """8-connected BFS distance from ``seed`` inside ``region`` (-1 outside)."""
    h, w = region.shape
    depth = np.full((h, w), -1, dtype=np.int32)
    depth[seed] = 0
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        y, x = queue.popleft()
        nd = depth[y, x] + 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and region[ny, nx] and depth[ny, nx] < 0:
                    depth[ny, nx] = nd
                    queue.append((ny, nx))
    return depth


def angular_corners(comp: np.ndarray) -> np.ndarray | None:
    """The old per-component corner rule: 4 angular-NMS farthest points."""
    arr = np.asarray(comp, dtype=np.float64)
    if arr.shape[0] < 4:
        return None
    centroid = arr.mean(axis=0)
    radial = np.linalg.norm(arr - centroid, axis=1)
    angles = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    try:
        idx = angular_nms_top_radial_indices(
            radial, angles, angular_nms_rad=ANGULAR_NMS_RAD, k=4
        )
    except ValueError:
        return None
    return arr[idx]


def prepare_example(
    gray_full: np.ndarray,
    binary_full: np.ndarray,
    image_name: str,
    cluster_index: int,
    cluster: CandidateCluster,
    *,
    name: str,
) -> Example:
    width = float(cluster.cols[5] - cluster.cols[0])
    center_row = float(cluster.row)
    center_col = float((cluster.cols[2] + cluster.cols[3]) / 2.0)
    side = int(round(CROP_SCALE * width))
    h, w = binary_full.shape
    r0 = int(round(center_row - side / 2.0))
    c0 = int(round(center_col - side / 2.0))
    r0 = max(0, min(r0, h - side))
    c0 = max(0, min(c0, w - side))

    gray = np.ascontiguousarray(gray_full[r0 : r0 + side, c0 : c0 + side])
    binary = np.ascontiguousarray(binary_full[r0 : r0 + side, c0 : c0 + side])

    seed_row = int(cluster.row) - r0
    # Old seed: midpoint of the first dark run of the 1:1:3:1:1 scan,
    # i.e. the outer black ring (cols[0]..cols[1]).
    seed_col = int((cluster.cols[0] + cluster.cols[1]) // 2) - c0
    if binary[seed_row, seed_col]:
        raise RuntimeError(f"{name}: seed is not in a dark run")
    seed = (seed_row, seed_col)

    region = region_fill_wave_front(binary, seed_row, seed_col)
    if (
        region[0, :].any()
        or region[-1, :].any()
        or region[:, 0].any()
        or region[:, -1].any()
    ):
        raise RuntimeError(
            f"{name}: flood fill touches the crop border; the crop is too tight "
            "to reproduce the full-image connected region"
        )
    boundary = region_boundary_8(region)
    raw_components = boundary_connected_components_ndimage(np.asarray(boundary))
    components = [
        np.asarray(c, dtype=np.float64)
        for c in sorted(raw_components, key=len, reverse=True)
    ]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for comp in components:
        c = angular_corners(comp)
        if c is not None:
            pairs.append((comp, c))

    depth = geodesic_depth(region, seed)
    boundary_depth = np.where(boundary, depth, -1)

    return Example(
        name=name,
        image=image_name,
        cluster=cluster,
        cluster_index=cluster_index,
        crop_box=(r0, r0 + side, c0, c0 + side),
        gray=gray,
        region=region,
        boundary=boundary,
        components=[p[0] for p in pairs],
        corners=[p[1] for p in pairs],
        depth=depth,
        boundary_depth=boundary_depth,
        seed=seed,
    )


def blend_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    out_wh: tuple[int, int],
) -> np.ndarray:
    if alpha <= 0 or not mask.any():
        return rgb
    m = cv2.resize(
        mask.astype(np.float32), out_wh, interpolation=cv2.INTER_LINEAR
    )
    a = (alpha * m)[..., None]
    return rgb * (1.0 - a) + MASK_GREEN * a


def draw_boundary(
    rgb: np.ndarray, mask: np.ndarray, out_wh: tuple[int, int]
) -> np.ndarray:
    if not mask.any():
        return rgb
    b = cv2.resize(mask.astype(np.float32), out_wh, interpolation=cv2.INTER_LINEAR)
    b = (b > 0.35).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    core = cv2.dilate(b, kernel, iterations=1).astype(bool)
    outline = cv2.dilate(b, kernel, iterations=2).astype(bool)
    rgb[outline] = (20, 20, 20)
    rgb[core] = BOUNDARY_ORANGE
    return rgb


def draw_corners(
    rgb: np.ndarray,
    components: list[np.ndarray],
    corners: list[np.ndarray],
    n_visible: int,
    scale: float,
) -> np.ndarray:
    for comp, q in zip(components[:n_visible], corners[:n_visible]):
        pts = (q[:, ::-1] * scale).astype(np.int32)
        # Quad through the four chosen points: black outline, magenta core.
        cv2.polylines(
            rgb,
            [pts.reshape(-1, 1, 2)],
            True,
            (20, 20, 20),
            6,
            cv2.LINE_AA,
        )
        cv2.polylines(
            rgb,
            [pts.reshape(-1, 1, 2)],
            True,
            QUAD_MAGENTA,
            2,
            cv2.LINE_AA,
        )
        for x, y in pts:
            cv2.drawMarker(
                rgb,
                (int(x), int(y)),
                CORNER_OUTLINE,
                cv2.MARKER_CROSS,
                24,
                6,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                rgb,
                (int(x), int(y)),
                CORNER_YELLOW,
                cv2.MARKER_CROSS,
                24,
                3,
                cv2.LINE_AA,
            )
    return rgb


def render_frame(ex: Example, frame_index: int) -> np.ndarray:
    big = OUT_SIZE * 2
    gray = cv2.resize(ex.gray, (big, big), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    scale = big / ex.gray.shape[0]

    max_depth = int(ex.depth[ex.region].max())
    max_boundary_depth = int(ex.boundary_depth[ex.boundary].max())

    mask_alpha = 0.48
    boundary_mask = np.zeros_like(ex.boundary)
    n_visible = 0

    if frame_index <= GROWTH_END:
        u = frame_index / GROWTH_END
        threshold = max_depth * smoothstep(u)
        mask = (ex.depth >= 0) & (ex.depth <= threshold)
    else:
        mask = ex.region
        if frame_index <= BOUNDARY_END:
            u = (frame_index - BOUNDARY_START + 1) / N_BOUNDARY
            threshold = max_boundary_depth * smoothstep(u)
            boundary_mask = ex.boundary & (ex.boundary_depth <= threshold)
            mask_alpha = 0.30
        else:
            boundary_mask = ex.boundary
            mask_alpha = 0.24
            if frame_index >= CORNERS_START:
                n_corners = len(ex.components)
                if ex.name == "success":
                    n_visible = (
                        1
                        if frame_index < CORNERS_START + N_CORNERS // 2
                        else n_corners
                    )
                else:
                    u = (frame_index - CORNERS_START + 1) / N_CORNERS
                    n_visible = max(1, math.ceil(u * n_corners))
                n_visible = min(n_visible, n_corners)
            else:
                n_visible = 0

    rgb = blend_mask(rgb, mask, mask_alpha, (big, big))
    rgb = draw_boundary(rgb, boundary_mask, (big, big))
    rgb = draw_corners(rgb, ex.components, ex.corners, n_visible, scale)

    out = cv2.resize(rgb, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    return np.clip(out, 0, 255).astype(np.uint8)


def write_gif(frames: list[np.ndarray], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # One shared palette derived from a frame containing every overlay colour.
    reference = Image.fromarray(frames[-1])
    palette = reference.quantize(colors=COLORS, method=Image.MEDIANCUT)
    p_frames = [
        Image.fromarray(frame).quantize(
            palette=palette, dither=Image.Dither.NONE
        )
        for frame in frames
    ]
    p_frames[0].save(
        path,
        save_all=True,
        append_images=p_frames[1:],
        duration=int(round(1000 / FPS)),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    image_path = args.data_dir / IMAGE_NAME
    gray_full = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2GRAY)
    if gray_full is None:
        raise SystemExit(f"could not read {image_path}")
    binary_full = binarize_image(gray_full)
    rows, cols = find_alignment_patterns_2d(binary_full, np.log(1.3))
    clusters = cluster_candidates(rows, cols)

    examples = [
        prepare_example(
            gray_full,
            binary_full,
            IMAGE_NAME,
            SUCCESS_CLUSTER,
            clusters[SUCCESS_CLUSTER],
            name="success",
        ),
        prepare_example(
            gray_full,
            binary_full,
            IMAGE_NAME,
            FAILURE_CLUSTER,
            clusters[FAILURE_CLUSTER],
            name="failure",
        ),
    ]

    print(f"Image: {image_path}")
    for ex in examples:
        c = ex.cluster
        print(f"\n{ex.name}:")
        print(f"  cluster index : {ex.cluster_index}")
        print(
            f"  cluster row/col: ({c.row:.1f}, "
            f"{(c.cols[2] + c.cols[3]) / 2:.1f})  width={c.cols[5] - c.cols[0]:.1f}px  "
            f"support={c.num_candidates}"
        )
        print(f"  seed (row,col): {ex.seed[0]}, {ex.seed[1]}  (crop-local)")
        print(
            f"  crop (r0,r1,c0,c1): {ex.crop_box}  "
            f"({ex.crop_box[1] - ex.crop_box[0]}x{ex.crop_box[3] - ex.crop_box[2]})"
        )
        print(
            f"  boundary components: {len(ex.components)} "
            f"(sizes {[len(c) for c in ex.components[:6]]}"
            f"{' ...' if len(ex.components) > 6 else ''})"
        )
        print(f"  corner sets: {len(ex.corners)} (= {4 * len(ex.corners)} points)")
        print("  mask does not touch crop border -> matches full-image fill")

    for ex in examples:
        frames = [render_frame(ex, i) for i in range(N_FRAMES)]
        out_path = args.out_dir / f"flood-fill-{ex.name}.gif"
        size = write_gif(frames, out_path)
        print(f"\nWrote {out_path} ({size / 1024:.0f} KiB, {N_FRAMES} frames)")


if __name__ == "__main__":
    main()
