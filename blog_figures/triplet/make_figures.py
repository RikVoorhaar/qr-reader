#!/usr/bin/env python3
"""Build the "choosing a triplet" figures for the QR-reader blog post.

Both figures use the same real captured photo,
``data/bench/captured-fullsize/qr_code_ - 37.jpeg`` (a version-20 code already
stored upside down — its ``v20`` label reads upside down).  No rotation is
applied: the point of the figures is that the detector does not assume an
upright code.

The photo is run through the *current* detection front end: binarize →
1:1:3:1:1 alignment scan → clustering → ray-fit finder corners → deduplicate.
That yields five ``FinderPattern`` candidates, three true finders and two clear
false positives.  The current centre-geometry triplet finder succeeds and the
sampled matrix decodes to ``rikvoorhaar.com`` at version 20.

The old pairwise edge-colinearity rule from ``f3f1a1d`` is reconstructed below.
On this photo it produces **no association at all**, because each of the two
true adjacent pairs has *four* segment pairs under the 0.15 offset tolerance
instead of the exactly two the old rule demanded.  The two extra pairs connect
the near edge of one finder to the far edge of the other (the 7-module-wide
finder edges).  ``max_offset`` divides the absolute perpendicular error by the
distance between segment midpoints, and at version 20 the finders are so far
apart that those cross-pairs normalise to ~0.09–0.10, below 0.15.

Figure A (``triplet-ambiguity.png``) shows the five candidate outlines and
corners with no labels, no chosen triplet and no QR boundary: the human eye
immediately sees the three real finders, but the algorithm has to decide.

Figure B (``triplet-colinearity.png``) extends all four outer edges of every
candidate as thin infinite lines, then highlights the true collinear pairs
(solid green) and one false cross-pair (dashed red).  The inset zooms on the
pair FP 1 ↔ FP 0 and annotates the perpendicular error and the normalisation
distance that let the false pair pass.

Usage (from the repository root):

    .venv/bin/python blog_figures/triplet/make_figures.py

The script is deterministic: the only randomness in the pipeline is fixed
before this stage, and rendering uses Agg.  Running it twice produces
byte-identical PNGs.

It writes

    blog/src/assets/blog/qr-reader/triplet-ambiguity.png
    blog/src/assets/blog/qr-reader/triplet-colinearity.png
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

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.detector.edge_fitting import MAX_GAP, DISTANCE_THRESHOLD  # noqa: E402
from qr_reader.detector.finder_pattern import (  # noqa: E402
    FinderPattern,
    find_valid_triplets,
)
from qr_reader.detector.geometry import (  # noqa: E402
    angular_distance,
    max_abs_line_distance,
    max_offset,
    segments_intersect,
)
from qr_reader.detector.homography import (  # noqa: E402
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)
from qr_reader.detector.ray_fit import fit_finder_ray  # noqa: E402
from qr_reader.detector.roi import cluster_to_bbox, cutout  # noqa: E402
from qr_reader.detector.sample import sample_qr_bits  # noqa: E402
from qr_reader.detector.detector import _score_timing_pattern  # noqa: E402
from qr_reader.qr_gen import binarize_image  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

IMAGE_NAME = "qr_code_ - 37.jpeg"
EXPECTED_PAYLOAD = "rikvoorhaar.com"

# Old rule from f3f1a1d/src/qr_reader/finder_pattern.py.
OLD_ANGLE_TOL = 0.1
OLD_OFFSET_TOL = 0.15

# Crop padding around the union of all candidate corners, in pixels.
CROP_PAD = 90

# Colours.  Assigned in sorted ``cluster_idx`` order so the figure is stable.
FINDER_COLORS = ["#ffd23f", "#4cc9f0", "#c77dff", "#ff6b6b", "#7bed9f"]
TRUE_PAIR_COLOR = "#7bed9f"
FALSE_PAIR_COLOR = "#ff4d4d"


# ---------------------------------------------------------------------------
# Current pipeline front end (mirrors detector._run_detection up to dedup)
# ---------------------------------------------------------------------------


def detect_candidates(gray: np.ndarray):
    """Return ``(fps, score_map, corners_xy)`` after ray fitting and dedup."""
    binary = binarize_image(gray)
    rows, cols = find_alignment_patterns_2d(binary, np.log(1.3))
    clusters = cluster_candidates(rows, cols)

    fps: list[FinderPattern] = []
    score_map: dict[int, float] = {}
    corners_xy: dict[int, np.ndarray] = {}

    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray, bbox)
        if roi.size == 0:
            continue

        r0 = max(0, int(bbox[0]))
        c0 = max(0, int(bbox[2]))
        c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0
        c_row = float(cluster.row) - r0
        m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

        fr = fit_finder_ray(
            roi,
            np.array([c_col, c_row], dtype=np.float64),
            m_est,
            max_gap=MAX_GAP,
            distance_threshold=DISTANCE_THRESHOLD,
        )
        if not fr.valid:
            continue

        corners = fr.corners + np.array([c0, r0], dtype=np.float64)
        wh = np.ptp(corners, axis=0)
        if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est:
            continue

        fps.append(FinderPattern(cluster_idx=ci, outer_corners=corners[:, ::-1]))
        score_map[ci] = fr.score
        corners_xy[ci] = corners

    # Deduplicate overlapping candidates (same rule as detector.py).
    keep = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep[i]:
            continue
        ci = fps[i].outer_corners.mean(axis=0)
        si = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
        for j in range(i + 1, len(fps)):
            if not keep[j]:
                continue
            cj = fps[j].outer_corners.mean(axis=0)
            sj = float(np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1]))
            if float(np.linalg.norm(ci - cj)) < min(si, sj):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    fps = [fp for fp, k in zip(fps, keep) if k]
    return fps, score_map, corners_xy


def decode_triplet(gray: np.ndarray, fps, corners_xy, triplet) -> tuple[str, int]:
    """Fit the global homography for ``triplet`` and decode the payload.

    This mirrors the version-selection loop in ``detector._run_detection``.
    """
    fp_map = {fp.cluster_idx: fp for fp in fps}
    idxs = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    centers_rc = {
        idx: fp_map[idx].outer_corners.mean(axis=0).astype(np.float64) for idx in idxs
    }

    tl, tr, bl = idxs
    ctl = centers_rc[tl][::-1]
    ctr = centers_rc[tr][::-1]
    cbl = centers_rc[bl][::-1]
    global_u = (ctr - ctl) / (np.linalg.norm(ctr - ctl) + 1e-12)
    global_v = (cbl - ctl) / (np.linalg.norm(cbl - ctl) + 1e-12)

    def canonicalize(corners_xy: np.ndarray) -> np.ndarray:
        centre = corners_xy.mean(axis=0)
        uv = corners_xy - centre
        u_proj = uv @ global_u
        v_proj = uv @ global_v
        return corners_xy[
            np.array(
                [
                    int(np.argmin(u_proj + v_proj)),
                    int(np.argmax(u_proj - v_proj)),
                    int(np.argmax(u_proj + v_proj)),
                    int(np.argmin(u_proj - v_proj)),
                ]
            )
        ]

    grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=np.float64)
    tl_c = canonicalize(corners_xy[tl])
    tr_c = canonicalize(corners_xy[tr])
    bl_c = canonicalize(corners_xy[bl])

    best_err = np.inf
    best_H = None
    best_N = 0
    for N_cand in range(21, 178, 4):
        src: list[list[float]] = []
        dst: list[list[float]] = []
        for corners, origin in [
            (tl_c, (0, 0)),
            (tr_c, (N_cand - 7, 0)),
            (bl_c, (0, N_cand - 7)),
        ]:
            for i in range(4):
                src.append(
                    [origin[0] + grid_offsets[i, 0], origin[1] + grid_offsets[i, 1]]
                )
                dst.append(corners[i].tolist())
        src_arr = np.array(src, dtype=np.float64)
        dst_arr = np.array(dst, dtype=np.float64)

        H = estimate_homography_dlt(src_arr, dst_arr)
        try:
            H = refine_homography_lm(H, src_arr, dst_arr, loss="linear")
        except Exception:
            pass
        proj = project_points(H, src_arr)
        err = float(np.mean(np.linalg.norm(proj - dst_arr, axis=1)))
        timing = _score_timing_pattern(gray, H, N_cand)
        if err - timing < best_err:
            best_err = err - timing
            best_H = H
            best_N = N_cand

    if best_H is None:
        return "", 0

    bits = sample_qr_bits(gray, best_H, best_N)
    return decode_qr(bits), (best_N - 17) // 4


# ---------------------------------------------------------------------------
# Old pairwise edge-colinearity rule (f3f1a1d)
# ---------------------------------------------------------------------------


def old_check_association(fp1, fp2, angle_tol=OLD_ANGLE_TOL, offset_tol=OLD_OFFSET_TOL):
    """Return the list of passing ``(i, j, angle, offset)`` pairs, or ``None``."""
    segs1 = fp1.segments()
    segs2 = fp2.segments()

    for s1 in segs1:
        for s2 in segs2:
            if segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                return None

    passing = []
    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            ang = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
            if ang < angle_tol:
                off = float(max_offset(s1[0], s1[1], s2[0], s2[1]))
                if off < offset_tol:
                    passing.append((i, j, ang, off))

    # The old rule demanded *exactly* two colinear segment pairs.
    return passing if len(passing) == 2 else None


def old_associations(fps):
    """All pair associations found by the old rule."""
    out = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            passing = old_check_association(fps[i], fps[j])
            if passing is not None:
                out.append(
                    (
                        fps[i].cluster_idx,
                        fps[j].cluster_idx,
                        passing[0][0],
                        passing[0][1],
                        passing[1][0],
                        passing[1][1],
                    )
                )
    return out


def all_segment_matches(fp1, fp2, angle_tol=OLD_ANGLE_TOL, offset_tol=OLD_OFFSET_TOL):
    """Every parallel-and-close segment pair, regardless of the exactly-two count."""
    matches = []
    for i, s1 in enumerate(fp1.segments()):
        for j, s2 in enumerate(fp2.segments()):
            ang = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
            if ang >= angle_tol:
                continue
            off = float(max_offset(s1[0], s1[1], s2[0], s2[1]))
            if off >= offset_tol:
                continue
            matches.append(
                {
                    "i": i,
                    "j": j,
                    "angle": ang,
                    "offset": off,
                    "p1": s1,
                    "p2": s2,
                }
            )
    return matches


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def crop_bounds(fps, shape, pad=CROP_PAD):
    all_corners = np.vstack([fp.outer_corners for fp in fps])
    r0 = max(0, int(np.floor(all_corners[:, 0].min())) - pad)
    r1 = min(shape[0], int(np.ceil(all_corners[:, 0].max())) + pad)
    c0 = max(0, int(np.floor(all_corners[:, 1].min())) - pad)
    c1 = min(shape[1], int(np.ceil(all_corners[:, 1].max())) + pad)
    return r0, r1, c0, c1


def quad_xy(fp):
    """Closed outline in (x, y) = (col, row)."""
    c = fp.outer_corners[:, ::-1]
    return np.vstack([c, c[0]])


def draw_quad(ax, fp, color, lw=3.0, alpha=1.0, zorder=4):
    q = quad_xy(fp)
    ax.plot(
        q[:, 0],
        q[:, 1],
        color="#101010",
        lw=lw + 1.8,
        alpha=0.55 * alpha,
        solid_joinstyle="round",
        zorder=zorder - 1,
    )
    ax.plot(
        q[:, 0],
        q[:, 1],
        color=color,
        lw=lw,
        alpha=alpha,
        solid_joinstyle="round",
        zorder=zorder,
    )
    c = fp.outer_corners[:, ::-1]
    ax.scatter(
        c[:, 0],
        c[:, 1],
        s=42,
        facecolors="none",
        edgecolors=color,
        linewidths=2.0,
        alpha=alpha,
        zorder=zorder + 1,
    )


def axline_segment(ax, p1, p2, **kwargs):
    """Infinite line through two (x, y) points, clipped to the axes."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    ax.axline((x1, y1), (x2, y2), **kwargs)


def rc_pair_to_xy(pair):
    """Convert a ((row, col), (row, col)) segment to ((x, y), (x, y))."""
    return pair[0][::-1], pair[1][::-1]


def offset_arrow(ax, line_a, line_b, color, label, fontsize=12):
    """Draw a perpendicular double arrow between two near-parallel segments."""
    p1, p2 = np.asarray(line_a, dtype=float), np.asarray(line_b, dtype=float)
    mid_a = p1.mean(axis=0)
    direction = p2[1] - p2[0]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    # Closest point on line b to mid_a.
    proj = p2[0] + direction * float(np.dot(mid_a - p2[0], direction))
    ax.annotate(
        "",
        xy=(float(proj[0]), float(proj[1])),
        xytext=(float(mid_a[0]), float(mid_a[1])),
        arrowprops=dict(arrowstyle="<->", color=color, lw=2.0, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    mid = (mid_a + proj) / 2.0
    ax.text(
        float(mid[0]),
        float(mid[1]),
        label,
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.22", fc="#101010", ec="none", alpha=0.72),
        zorder=9,
    )


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def render_ambiguity(gray, fps, bounds, out_path: Path):
    r0, r1, c0, c1 = bounds
    colors = {fp.cluster_idx: FINDER_COLORS[k % len(FINDER_COLORS)] for k, fp in enumerate(fps)}

    fig, ax = plt.subplots(figsize=(7.4, 7.4), dpi=200)
    ax.imshow(gray[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])
    for fp in fps:
        draw_quad(ax, fp, colors[fp.cluster_idx], lw=3.4)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_colinearity(gray, fps, colors, triplet, bounds, out_path: Path):
    r0, r1, c0, c1 = bounds
    fp_map = {fp.cluster_idx: fp for fp in fps}

    tl, tr, bl = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx
    true_pairs = [(tl, tr), (tl, bl)]

    # --- Left: all candidates, all four edges extended ---------------------
    fig = plt.figure(figsize=(9.2, 11.6), dpi=170)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.04)
    ax = fig.add_subplot(gs[0])

    ax.imshow(gray[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])
    for fp in fps:
        c = fp.outer_corners[:, ::-1]
        for k in range(4):
            axline_segment(
                ax,
                c[k],
                c[(k + 1) % 4],
                color=colors[fp.cluster_idx],
                lw=1.0,
                alpha=0.42,
                zorder=2,
            )
    for fp in fps:
        draw_quad(ax, fp, colors[fp.cluster_idx], lw=2.0, alpha=0.95, zorder=4)

    # Highlight the true collinear edge pairs, and one false cross-pair.
    for a_idx, b_idx in true_pairs:
        matches = all_segment_matches(fp_map[a_idx], fp_map[b_idx])
        matches.sort(key=lambda m: m["offset"])
        for m in matches[:2]:
            for p in (m["p1"], m["p2"]):
                q1, q2 = rc_pair_to_xy(p)
                axline_segment(
                    ax,
                    q1,
                    q2,
                    color=TRUE_PAIR_COLOR,
                    lw=2.8,
                    alpha=0.92,
                    zorder=6,
                )

    first_pair = true_pairs[0]
    cross_matches = all_segment_matches(fp_map[first_pair[0]], fp_map[first_pair[1]])
    cross_matches.sort(key=lambda m: m["offset"])
    false_match = cross_matches[2]  # first cross-pair after the two true pairs
    for p in (false_match["p1"], false_match["p2"]):
        q1, q2 = rc_pair_to_xy(p)
        axline_segment(
            ax,
            q1,
            q2,
            color=FALSE_PAIR_COLOR,
            lw=2.6,
            alpha=0.95,
            ls="--",
            zorder=7,
        )

    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    ax.set_title("candidate edges", loc="left", fontsize=17, pad=6)

    # --- Right: zoom on the first true pair --------------------------------
    a_idx, b_idx = first_pair
    pair_corners = np.vstack(
        [fp_map[a_idx].outer_corners, fp_map[b_idx].outer_corners]
    )
    pad_r = 170
    pad_c = 70
    zr0 = max(0, int(pair_corners[:, 0].min()) - pad_r)
    zr1 = min(gray.shape[0], int(pair_corners[:, 0].max()) + pad_r)
    zc0 = max(0, int(pair_corners[:, 1].min()) - pad_c)
    zc1 = min(gray.shape[1], int(pair_corners[:, 1].max()) + pad_c)

    az = fig.add_subplot(gs[1])
    az.imshow(gray[zr0:zr1, zc0:zc1], cmap="gray", extent=[zc0, zc1, zr1, zr0])
    for idx in (a_idx, b_idx):
        draw_quad(az, fp_map[idx], colors[idx], lw=3.4, zorder=5)

    matches = all_segment_matches(fp_map[a_idx], fp_map[b_idx])
    matches.sort(key=lambda m: m["offset"])
    true_matches = matches[:2]
    false_matches = matches[2:]

    for m in true_matches:
        for p in (m["p1"], m["p2"]):
            q1, q2 = rc_pair_to_xy(p)
            axline_segment(
                az,
                q1,
                q2,
                color=TRUE_PAIR_COLOR,
                lw=3.0,
                alpha=0.95,
                zorder=6,
            )
    for m in false_matches[:1]:
        for p in (m["p1"], m["p2"]):
            q1, q2 = rc_pair_to_xy(p)
            axline_segment(
                az,
                q1,
                q2,
                color=FALSE_PAIR_COLOR,
                lw=2.6,
                alpha=0.95,
                ls="--",
                zorder=7,
            )

    # Perpendicular gap of the highlighted false cross-pair, in pixels/modules.
    fm = false_matches[0]
    line_a_rc = np.asarray(fm["p1"], dtype=float)
    line_b_rc = np.asarray(fm["p2"], dtype=float)
    off_px = float(
        max_abs_line_distance(
            line_a_rc[0], line_a_rc[1], line_b_rc[0], line_b_rc[1]
        )
    )
    line_a = line_a_rc[:, ::-1]
    line_b = line_b_rc[:, ::-1]
    mid_a = line_a.mean(axis=0)
    mid_b = line_b.mean(axis=0)
    norm = float(np.linalg.norm(mid_a - mid_b))
    module_px = float(
        np.mean(
            [
                np.linalg.norm(fp_map[idx].outer_corners[k] - fp_map[idx].outer_corners[(k + 1) % 4])
                for idx in (a_idx, b_idx)
                for k in range(4)
            ]
        )
        / 7.0
    )
    offset_arrow(
        az,
        line_a,
        line_b,
        FALSE_PAIR_COLOR,
        f"{off_px:.0f} px ≈ {off_px / module_px:.1f} modules",
        fontsize=12,
    )

    az.text(
        0.015,
        0.965,
        (
            f"4 edge pairs pass (offset < {OLD_OFFSET_TOL:.2f})\n"
            f"old rule needs exactly 2 → pair rejected\n"
            f"L = {norm:.0f} px → {fm['offset']:.3f}"
        ),
        transform=az.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        color="#f5f5f5",
        bbox=dict(boxstyle="round,pad=0.35", fc="#101010", ec="none", alpha=0.78),
        zorder=10,
    )

    # Mark the pair identity in the zoom only.
    for idx in (a_idx, b_idx):
        c = fp_map[idx].outer_corners.mean(axis=0)
        az.text(
            c[1],
            c[0],
            str(idx),
            color=colors[idx],
            fontsize=15,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
        )

    az.set_xlim(zc0, zc1)
    az.set_ylim(zr1, zr0)
    az.axis("off")
    az.set_title(
        f"zoom: FP {a_idx} ↔ FP {b_idx}", loc="left", fontsize=17, pad=6
    )

    # Compact legend, no caption.
    handles = [
        plt.Line2D([0], [0], color=TRUE_PAIR_COLOR, lw=3.2, label="true collinear pair"),
        plt.Line2D(
            [0], [0], color=FALSE_PAIR_COLOR, lw=2.6, ls="--", label="false cross-pair"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=11.5,
        framealpha=0.80,
        facecolor="#101010",
        edgecolor="none",
        labelcolor="white",
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.data_dir / IMAGE_NAME
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fps, score_map, corners_xy = detect_candidates(gray)
    fps.sort(key=lambda fp: fp.cluster_idx)
    print(f"image: {image_path}")
    print(f"image shape: {gray.shape}")
    print(f"finder candidates: {len(fps)}")
    for fp in fps:
        c = fp.outer_corners
        sides = [float(np.linalg.norm(c[(k + 1) % 4] - c[k])) for k in range(4)]
        print(
            f"  FP {fp.cluster_idx}: centre=({c[:, 1].mean():.1f}, {c[:, 0].mean():.1f})px "
            f"sides={np.round(sides, 1).tolist()}"
        )

    triplets = find_valid_triplets(fps, score_map)
    if not triplets:
        raise SystemExit("current triplet finder failed unexpectedly")
    triplet = triplets[0]
    text, version = decode_triplet(gray, fps, corners_xy, triplet)
    print(
        f"current triplet: TL={triplet.top_left_idx} TR={triplet.top_right_idx} "
        f"BL={triplet.bottom_left_idx}"
    )
    print(f"decoded: {text!r}  version={version}")
    if text != EXPECTED_PAYLOAD:
        raise SystemExit(f"unexpected payload {text!r}, expected {EXPECTED_PAYLOAD!r}")

    assoc = old_associations(fps)
    print(f"old associations: {assoc if assoc else 'none'}")

    fp_map = {fp.cluster_idx: fp for fp in fps}
    tl, tr, bl = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx
    print("old rule detail for the two true adjacent pairs:")
    for a_idx, b_idx in ((tl, tr), (tl, bl)):
        matches = all_segment_matches(fp_map[a_idx], fp_map[b_idx])
        matches.sort(key=lambda m: m["offset"])
        print(f"  FP {a_idx} ↔ FP {b_idx}: {len(matches)} passing pairs")
        for m in matches:
            print(
                f"    seg ({m['i']},{m['j']}): angle={m['angle']:.4f} rad, "
                f"offset={m['offset']:.4f}"
            )

    bounds = crop_bounds(fps, gray.shape)
    print(f"crop bounds (r0,r1,c0,c1): {bounds}")

    colors = {
        fp.cluster_idx: FINDER_COLORS[k % len(FINDER_COLORS)]
        for k, fp in enumerate(fps)
    }

    ambiguity_path = args.out_dir / "triplet-ambiguity.png"
    colinearity_path = args.out_dir / "triplet-colinearity.png"
    render_ambiguity(gray, fps, bounds, ambiguity_path)
    render_colinearity(gray, fps, colors, triplet, bounds, colinearity_path)
    print(f"wrote {ambiguity_path}")
    print(f"wrote {colinearity_path}")


if __name__ == "__main__":
    main()
