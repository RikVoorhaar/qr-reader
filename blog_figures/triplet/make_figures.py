#!/usr/bin/env python3
"""Build the "choosing a triplet" figures for the QR-reader blog post.

The two figures use different captured photos.  No single version-10-or-lower
photo in the benchmark had both a clean set of extra candidates *and* the
four-pair colinearity failure, so the ambiguity figure uses one photo and the
colinearity figure another (both encode ``rikvoorhaar.com``).

Figure A — ``qr_code_ - 7.jpeg`` (version 5)
    A code that is already stored upside down, with four candidates: the three
    true finders and one plausible extra square near the right edge.  No
    rotation is applied.  Candidate outlines and corners only: no labels, no
    chosen triplet, no QR boundary.

Figure B — ``qr_code_ - 20.jpeg`` (version 10)
    A code that is upright in the photo; the image and finder corners are
    rotated rigidly by 180° for the figure.  The pipeline finds five
    candidates, but the figure shows only the two adjacent true finders
    involved in the failure (TL↔TR).  The current centre-geometry triplet
    finder succeeds and the sampled matrix decodes to ``rikvoorhaar.com``.

Old pairwise rule (Figure B)
----------------------------
The rule from ``f3f1a1d`` is reconstructed below.  For the TL↔TR pair it sees
four segment pairs under the tolerances:

    top–top        offset 0.005   (true)
    bottom–bottom  offset 0.007   (true)
    top–bottom     offset 0.145   (false cross-pair)
    bottom–top     offset 0.147   (false cross-pair)

Both true adjacent pairs (TL↔TR and TL↔BL) have the same structure.  The old
rule required *exactly two* passing pairs, so it rejects every true pair and
returns no association at all.  The false cross-pairs pass because
``max_offset`` divides the absolute perpendicular error (~99 px, about 7.5
modules) by the distance between segment midpoints (~675 px), which is the
distance between the two finder patterns.

The true pairs are identified geometrically from the QR grid directions
(TL→TR and TL→BL), not by sorting offsets: top–top and bottom–bottom for the
TL↔TR pair, left–left and right–right for the TL↔BL pair.  The highlighted
false pair is the opposite-edge cross-pair (bottom edge of one finder against
the top edge of the other) that makes the count four.

Usage (from the repository root):

    .venv/bin/python blog_figures/triplet/make_figures.py

The script is deterministic: the pipeline stages used here have no random
component and rendering uses Agg.  Running it twice produces byte-identical
PNGs.

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

FIGURE_A_IMAGE = "qr_code_ - 7.jpeg"
FIGURE_B_IMAGE = "qr_code_ - 20.jpeg"
EXPECTED_PAYLOAD = "rikvoorhaar.com"

# Old rule from f3f1a1d/src/qr_reader/finder_pattern.py.
OLD_ANGLE_TOL = 0.1
OLD_OFFSET_TOL = 0.15

CROP_PAD = 110

# Colours, assigned in sorted ``cluster_idx`` order.
FINDER_COLORS = ["#ffd23f", "#4cc9f0", "#c77dff", "#ff6b6b", "#7bed9f"]
TRUE_PAIR_COLOR = "#39ff88"
FALSE_PAIR_COLOR = "#ff3b3b"


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

    Mirrors the version-selection loop in ``detector._run_detection``.
    """
    fp_map = {fp.cluster_idx: fp for fp in fps}
    idxs = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    centers = {idx: fp_map[idx].outer_corners.mean(axis=0) for idx in idxs}

    tl, tr, bl = idxs
    ctl, ctr, cbl = (centers[i][::-1] for i in idxs)
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
    corners_c = [canonicalize(corners_xy[i]) for i in idxs]

    best_err = np.inf
    best_H = None
    best_N = 0
    for N_cand in range(21, 178, 4):
        src: list[list[float]] = []
        dst: list[list[float]] = []
        for corners, origin in [
            (corners_c[0], (0, 0)),
            (corners_c[1], (N_cand - 7, 0)),
            (corners_c[2], (0, N_cand - 7)),
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
# Rigid 180-degree rotation of the candidate set
# ---------------------------------------------------------------------------


def rotate_corners_180(corners_rc: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    out = np.empty_like(corners_rc, dtype=np.float64)
    out[:, 0] = (h - 1) - corners_rc[:, 0]
    out[:, 1] = (w - 1) - corners_rc[:, 1]
    return out


def rotate_fps_180(fps, shape):
    return [
        FinderPattern(
            cluster_idx=fp.cluster_idx,
            outer_corners=rotate_corners_180(fp.outer_corners, shape),
        )
        for fp in fps
    ]


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
    out = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            passing = old_check_association(fps[i], fps[j])
            if passing is not None:
                out.append((fps[i].cluster_idx, fps[j].cluster_idx, passing))
    return out


# ---------------------------------------------------------------------------
# Geometric identification of the true edge pairs
# ---------------------------------------------------------------------------


def edge_data(fp: FinderPattern):
    """Per-edge direction, outward normal and midpoint in (row, col)."""
    c = fp.outer_corners
    centre = c.mean(axis=0)
    out = {}
    for k in range(4):
        p, q = c[k], c[(k + 1) % 4]
        d = q - p
        d = d / (np.linalg.norm(d) + 1e-12)
        mid = (p + q) / 2.0
        n = np.array([-d[1], d[0]])
        if float(np.dot(n, mid - centre)) < 0:
            n = -n
        out[k] = {"direction": d, "normal": n, "midpoint": mid}
    return out


def classify_edges(fp: FinderPattern, u: np.ndarray, v: np.ndarray):
    """Return segment indices for the top/bottom/left/right edges.

    ``u`` is TL→TR and ``v`` is TL→BL in (row, col) coordinates.  For the
    TL↔TR pair the true collinear edges are top–top and bottom–bottom; for the
    TL↔BL pair they are left–left and right–right.
    """
    edges = edge_data(fp)
    dots_u = {k: abs(float(np.dot(e["direction"], u))) for k, e in edges.items()}
    order_u = sorted(edges, key=lambda k: dots_u[k], reverse=True)
    along_u, along_v = order_u[:2], order_u[2:]

    top = [k for k in along_u if float(np.dot(edges[k]["normal"], v)) < 0]
    bottom = [k for k in along_u if float(np.dot(edges[k]["normal"], v)) > 0]
    left = [k for k in along_v if float(np.dot(edges[k]["normal"], u)) < 0]
    right = [k for k in along_v if float(np.dot(edges[k]["normal"], u)) > 0]

    if not (top and bottom and left and right):
        raise RuntimeError(f"could not classify edges of FP {fp.cluster_idx}")
    return {
        "top": top[0],
        "bottom": bottom[0],
        "left": left[0],
        "right": right[0],
    }


def pair_metrics(fp_a, seg_a: int, fp_b, seg_b: int):
    s1 = fp_a.segments()[seg_a]
    s2 = fp_b.segments()[seg_b]
    angle = float(angular_distance(s1[0], s1[1], s2[0], s2[1]))
    offset = float(max_offset(s1[0], s1[1], s2[0], s2[1]))
    deviation = float(max_abs_line_distance(s1[0], s1[1], s2[0], s2[1]))
    midpoint_distance = float(
        np.linalg.norm(np.asarray(s1).mean(axis=0) - np.asarray(s2).mean(axis=0))
    )
    return {
        "angle": angle,
        "offset": offset,
        "deviation": deviation,
        "midpoint_distance": midpoint_distance,
        "seg_a": seg_a,
        "seg_b": seg_b,
        "fp_a": fp_a.cluster_idx,
        "fp_b": fp_b.cluster_idx,
    }


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
    c = fp.outer_corners[:, ::-1]
    return np.vstack([c, c[0]])


def draw_quad(ax, fp, color, lw=3.0, alpha=1.0, zorder=4):
    q = quad_xy(fp)
    ax.plot(
        q[:, 0],
        q[:, 1],
        color="#101010",
        lw=lw + 2.0,
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
        s=38,
        facecolors="none",
        edgecolors=color,
        linewidths=2.0,
        alpha=alpha,
        zorder=zorder + 1,
    )


def axline_segment(ax, p1, p2, **kwargs):
    """Infinite line through two (x, y) points, clipped to the axes."""
    ax.axline(
        (float(p1[0]), float(p1[1])),
        (float(p2[0]), float(p2[1])),
        **kwargs,
    )


def rc_pair_to_xy(pair):
    return pair[0][::-1], pair[1][::-1]


def connect_midpoints(ax, fp_a, seg_a, fp_b, seg_b, color, lw=3.2, ls="-", zorder=6):
    s1 = np.asarray(fp_a.segments()[seg_a], dtype=float).mean(axis=0)[::-1]
    s2 = np.asarray(fp_b.segments()[seg_b], dtype=float).mean(axis=0)[::-1]
    ax.plot(
        [s1[0], s2[0]],
        [s1[1], s2[1]],
        color=color,
        lw=lw,
        ls=ls,
        alpha=0.96,
        zorder=zorder,
        solid_capstyle="round",
    )
    ax.scatter(
        [s1[0], s2[0]],
        [s1[1], s2[1]],
        s=34,
        color=color,
        edgecolors="none",
        zorder=zorder + 1,
    )


def gap_arrow(ax, line_a, line_b, x, color, label, fontsize=15):
    """Vertical double arrow between two near-horizontal (x, y) lines."""
    p1 = np.asarray(line_a, dtype=float)
    p2 = np.asarray(line_b, dtype=float)

    def y_at(line, xv):
        a, b = line
        if abs(b[0] - a[0]) < 1e-9:
            raise ValueError("vertical line")
        return a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0])

    ya = y_at(p1, x)
    yb = y_at(p2, x)
    ax.annotate(
        "",
        xy=(float(x), float(ya)),
        xytext=(float(x), float(yb)),
        arrowprops=dict(arrowstyle="<->", color=color, lw=2.6, shrinkA=0, shrinkB=0),
        zorder=8,
    )
    ax.text(
        float(x + 26),
        float((ya + yb) / 2.0),
        label,
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="#101010", ec="none", alpha=0.85),
        zorder=9,
    )


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------


def render_ambiguity(gray_rot, fps, bounds, out_path: Path):
    r0, r1, c0, c1 = bounds
    # One accent colour for every candidate: the figure asks *which three*,
    # so colour must not hint at an answer.
    accent = "#ff4fa3"

    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=200)
    ax.imshow(gray_rot[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])
    for fp in fps:
        draw_quad(ax, fp, accent, lw=3.6)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_colinearity(
    gray_rot,
    fps,
    colors,
    triplet,
    true_matches,
    false_match,
    out_path: Path,
):
    fp_map = {fp.cluster_idx: fp for fp in fps}
    tl, tr, bl = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx
    a_idx, b_idx = tl, tr

    pair_corners = np.vstack(
        [fp_map[a_idx].outer_corners, fp_map[b_idx].outer_corners]
    )
    main_pad_r, main_pad_c = 360, 120
    mr0 = max(0, int(pair_corners[:, 0].min()) - main_pad_r)
    mr1 = min(gray_rot.shape[0], int(pair_corners[:, 0].max()) + main_pad_r)
    mc0 = max(0, int(pair_corners[:, 1].min()) - main_pad_c)
    mc1 = min(gray_rot.shape[1], int(pair_corners[:, 1].max()) + main_pad_c)

    fig = plt.figure(figsize=(10.2, 12.4), dpi=170)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])

    # --- main panel: only the two adjacent true finders --------------------
    ax.imshow(gray_rot[mr0:mr1, mc0:mc1], cmap="gray", extent=[mc0, mc1, mr1, mr0])
    for idx in (a_idx, b_idx):
        fp = fp_map[idx]
        c = fp.outer_corners[:, ::-1]
        for k in range(4):
            axline_segment(
                ax,
                c[k],
                c[(k + 1) % 4],
                color=colors[idx],
                lw=1.25,
                alpha=0.55,
                zorder=2,
            )
        draw_quad(ax, fp, colors[idx], lw=2.6, alpha=1.0, zorder=4)

    # True associations for this pair only: top–top and bottom–bottom.
    for m in true_matches:
        if {m["fp_a"], m["fp_b"]} != {a_idx, b_idx}:
            continue
        connect_midpoints(
            ax,
            fp_map[m["fp_a"]],
            m["seg_a"],
            fp_map[m["fp_b"]],
            m["seg_b"],
            TRUE_PAIR_COLOR,
            lw=4.0,
            zorder=6,
        )

    # The false opposite-edge cross-pair that makes it four instead of two.
    connect_midpoints(
        ax,
        fp_map[false_match["fp_a"]],
        false_match["seg_a"],
        fp_map[false_match["fp_b"]],
        false_match["seg_b"],
        FALSE_PAIR_COLOR,
        lw=3.8,
        ls="--",
        zorder=7,
    )

    ax.set_xlim(mc0, mc1)
    ax.set_ylim(mr1, mr0)
    ax.axis("off")
    ax.set_title("TL ↔ TR", loc="left", fontsize=19, pad=8)

    handles = [
        plt.Line2D([0], [0], color=TRUE_PAIR_COLOR, lw=3.4, label="true collinear pair"),
        plt.Line2D(
            [0], [0], color=FALSE_PAIR_COLOR, lw=3.2, ls="--", label="false cross-pair"
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=13,
        framealpha=0.88,
        facecolor="#101010",
        edgecolor="none",
        labelcolor="white",
    )

    # --- inset: the TL↔TR pair and the false cross-pair --------------------
    pad_r, pad_c = 150, 70
    zr0 = max(0, int(pair_corners[:, 0].min()) - pad_r)
    zr1 = min(gray_rot.shape[0], int(pair_corners[:, 0].max()) + pad_r)
    zc0 = max(0, int(pair_corners[:, 1].min()) - pad_c)
    zc1 = min(gray_rot.shape[1], int(pair_corners[:, 1].max()) + pad_c)

    az = fig.add_subplot(gs[1])
    az.imshow(gray_rot[zr0:zr1, zc0:zc1], cmap="gray", extent=[zc0, zc1, zr1, zr0])
    for idx in (a_idx, b_idx):
        draw_quad(az, fp_map[idx], colors[idx], lw=3.6, zorder=5)

    # Draw the two parallel edge lines involved in the false cross-pair.
    for fp, seg, col in (
        (fp_map[false_match["fp_a"]], false_match["seg_a"], colors[false_match["fp_a"]]),
        (fp_map[false_match["fp_b"]], false_match["seg_b"], colors[false_match["fp_b"]]),
    ):
        p = fp.segments()[seg]
        q1, q2 = rc_pair_to_xy(p)
        axline_segment(az, q1, q2, color=col, lw=1.5, alpha=0.85, zorder=3)

    for m in true_matches:
        if {m["fp_a"], m["fp_b"]} != {a_idx, b_idx}:
            continue
        connect_midpoints(
            az,
            fp_map[m["fp_a"]],
            m["seg_a"],
            fp_map[m["fp_b"]],
            m["seg_b"],
            TRUE_PAIR_COLOR,
            lw=4.2,
            zorder=6,
        )
    connect_midpoints(
        az,
        fp_map[false_match["fp_a"]],
        false_match["seg_a"],
        fp_map[false_match["fp_b"]],
        false_match["seg_b"],
        FALSE_PAIR_COLOR,
        lw=3.8,
        ls="--",
        zorder=7,
    )

    # The gap between the two parallel edge lines in the false cross-pair.
    line_a_rc = np.asarray(
        fp_map[false_match["fp_a"]].segments()[false_match["seg_a"]], dtype=float
    )
    line_b_rc = np.asarray(
        fp_map[false_match["fp_b"]].segments()[false_match["seg_b"]], dtype=float
    )
    line_a = line_a_rc[:, ::-1]
    line_b = line_b_rc[:, ::-1]
    x_mid = float(
        (
            fp_map[a_idx].outer_corners[:, 1].mean()
            + fp_map[b_idx].outer_corners[:, 1].mean()
        )
        / 2.0
    )
    module_px = float(
        np.mean(
            [
                np.linalg.norm(
                    fp_map[idx].outer_corners[k] - fp_map[idx].outer_corners[(k + 1) % 4]
                )
                for idx in (a_idx, b_idx)
                for k in range(4)
            ]
        )
        / 7.0
    )
    gap_arrow(
        az,
        line_a,
        line_b,
        x_mid,
        FALSE_PAIR_COLOR,
        (
            f"{false_match['deviation']:.0f} px ≈ "
            f"{false_match['deviation'] / module_px:.1f} modules"
        ),
        fontsize=15,
    )

    az.text(
        0.015,
        0.97,
        (
            "4 edge pairs pass\n"
            "rule needs exactly 2\n"
            f"{false_match['deviation']:.0f} / "
            f"{false_match['midpoint_distance']:.0f} px = "
            f"{false_match['offset']:.3f}"
        ),
        transform=az.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        color="#ffffff",
        linespacing=1.25,
        bbox=dict(boxstyle="round,pad=0.38", fc="#101010", ec="none", alpha=0.85),
        zorder=10,
    )

    for idx in (a_idx, b_idx):
        c = fp_map[idx].outer_corners.mean(axis=0)
        az.text(
            c[1],
            c[0],
            str(idx),
            color=colors[idx],
            fontsize=17,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=11,
        )

    az.set_xlim(zc0, zc1)
    az.set_ylim(zr1, zr0)
    az.axis("off")
    az.set_title(
        f"zoom: FP {a_idx} ↔ FP {b_idx}", loc="left", fontsize=19, pad=8
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.975, bottom=0.01)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyse_image(data_dir: Path, image_name: str, rotate_display: bool):
    """Run the current front end and the old rule on one captured photo."""
    image_path = data_dir / image_name
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    fps_orig, score_map, corners_xy = detect_candidates(gray)
    fps_orig.sort(key=lambda fp: fp.cluster_idx)

    # Verify the current detector on the original orientation.
    triplets_orig = find_valid_triplets(fps_orig, score_map)
    if not triplets_orig:
        raise SystemExit(f"{image_name}: current triplet finder failed")
    text, version = decode_triplet(gray, fps_orig, corners_xy, triplets_orig[0])
    if text != EXPECTED_PAYLOAD:
        raise SystemExit(
            f"{image_name}: unexpected payload {text!r}, "
            f"expected {EXPECTED_PAYLOAD!r}"
        )

    if rotate_display:
        gray_used = np.ascontiguousarray(np.rot90(gray, 2))
        fps = rotate_fps_180(fps_orig, gray.shape)
    else:
        gray_used = gray
        fps = fps_orig
    fps.sort(key=lambda fp: fp.cluster_idx)

    triplets = find_valid_triplets(fps, score_map)
    if not triplets:
        raise SystemExit(f"{image_name}: current triplet finder failed after transform")
    triplet = triplets[0]
    tl, tr, bl = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx

    fp_map = {fp.cluster_idx: fp for fp in fps}
    centers = {i: fp_map[i].outer_corners.mean(axis=0) for i in (tl, tr, bl)}
    u = centers[tr] - centers[tl]
    u = u / np.linalg.norm(u)
    v = centers[bl] - centers[tl]
    v = v / np.linalg.norm(v)
    edge_class = {i: classify_edges(fp_map[i], u, v) for i in (tl, tr, bl)}

    true_specs = [
        (tl, edge_class[tl]["top"], tr, edge_class[tr]["top"]),
        (tl, edge_class[tl]["bottom"], tr, edge_class[tr]["bottom"]),
        (tl, edge_class[tl]["left"], bl, edge_class[bl]["left"]),
        (tl, edge_class[tl]["right"], bl, edge_class[bl]["right"]),
    ]
    true_matches = [
        pair_metrics(fp_map[a], ka, fp_map[b], kb) for a, ka, b, kb in true_specs
    ]
    cross_candidates = [
        pair_metrics(
            fp_map[tl], edge_class[tl]["top"], fp_map[tr], edge_class[tr]["bottom"]
        ),
        pair_metrics(
            fp_map[tl], edge_class[tl]["bottom"], fp_map[tr], edge_class[tr]["top"]
        ),
    ]
    false_match = max(cross_candidates, key=lambda m: m["offset"])

    print(f"\n=== {image_name}")
    print(f"image shape: {gray.shape}")
    print(f"decoded: {text!r}  version={version}")
    print(f"finder candidates: {len(fps)}")
    for fp in fps:
        c = fp.outer_corners
        sides = [float(np.linalg.norm(c[(k + 1) % 4] - c[k])) for k in range(4)]
        print(
            f"  FP {fp.cluster_idx}: centre=({c[:, 1].mean():.1f}, {c[:, 0].mean():.1f})px "
            f"sides={np.round(sides, 1).tolist()}"
        )
    print(f"current triplet: TL={tl} TR={tr} BL={bl}")
    print("true collinear pairs:")
    for m in true_matches:
        print(
            f"  FP {m['fp_a']} seg {m['seg_a']} ↔ FP {m['fp_b']} seg {m['seg_b']}: "
            f"angle={m['angle']:.4f} offset={m['offset']:.4f} "
            f"dev={m['deviation']:.1f}px L={m['midpoint_distance']:.1f}px"
        )
    print("opposite-edge cross-pairs for TL↔TR:")
    for m in cross_candidates:
        print(
            f"  FP {m['fp_a']} seg {m['seg_a']} ↔ FP {m['fp_b']} seg {m['seg_b']}: "
            f"angle={m['angle']:.4f} offset={m['offset']:.4f} "
            f"dev={m['deviation']:.1f}px L={m['midpoint_distance']:.1f}px"
        )
    assoc = old_associations(fps)
    print(f"old associations: {assoc if assoc else 'none'}")

    bounds = crop_bounds(fps, gray_used.shape)
    print(f"crop bounds (r0,r1,c0,c1): {bounds}")
    colors = {
        fp.cluster_idx: FINDER_COLORS[k % len(FINDER_COLORS)]
        for k, fp in enumerate(fps)
    }
    return {
        "gray": gray_used,
        "fps": fps,
        "colors": colors,
        "triplet": triplet,
        "true_matches": true_matches,
        "false_match": false_match,
        "bounds": bounds,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig_a = analyse_image(args.data_dir, FIGURE_A_IMAGE, rotate_display=False)
    fig_b = analyse_image(args.data_dir, FIGURE_B_IMAGE, rotate_display=True)

    ambiguity_path = args.out_dir / "triplet-ambiguity.png"
    colinearity_path = args.out_dir / "triplet-colinearity.png"

    render_ambiguity(
        fig_a["gray"], fig_a["fps"], fig_a["bounds"], ambiguity_path
    )
    render_colinearity(
        fig_b["gray"],
        fig_b["fps"],
        fig_b["colors"],
        fig_b["triplet"],
        fig_b["true_matches"],
        fig_b["false_match"],
        colinearity_path,
    )
    print(f"\nwrote {ambiguity_path}")
    print(f"wrote {colinearity_path}")


if __name__ == "__main__":
    main()
