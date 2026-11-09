#!/usr/bin/env python3
"""Build the Hough success/failure figures for the QR-reader blog post.

Both figures are static four-panel rows rendered from real captured photos in
``data/bench/captured-fullsize``:

1. Original grayscale crop around one finder-pattern candidate.
2. Thin Sobel + non-maximum-suppression edge map.
3. Gradient-guided Hough accumulator, with all extracted peaks faint and the
   four selected peaks highlighted.
4. The original crop with the four refined segments, their infinite-line
   extensions, and the four line intersections marked as corners.

The detector is the abandoned experimental one, frozen in
``blog_figures/hough/hough_legacy.py`` (copied from ``f5451e8^``).  Settings
match the blog post: Gaussian blur sigma 1, Sobel gradients, NMS along the
gradient, Hough bins of 0.5 degree and 0.5 pixel, votes weighted by edge
strength, iterative peak extraction with local non-maximum suppression.

Success case
    ``qr_code_ - 16.jpeg``, cluster 2.  The four outer edges give one strong,
    well-separated peak each, the refined segments span the whole edge, and
    the four intersections land on the finder corners.

Failure case
    ``qr_code_ - 97.jpeg``, cluster 0.  One outer edge is split into six
    separate Hough peaks and another has no matching peak at all; the longest
    support runs cover only ~28% of the true edge length on average, while
    surrounding QR-module edges contribute 12 further strong peaks.  The four
    selected lines take inner/parallel intercepts, so the intersection quad is
    too narrow: its two right-hand corners land inside the finder.  This is
    the "gaps + background clutter" failure the post describes, not an
    invented one.

Usage (from the repository root):

    .venv/bin/python blog_figures/hough/make_figures.py

The script is deterministic: alignment scanning, clustering, NMS, voting and
peak selection have no random component.  It writes

    blog/src/assets/blog/qr-reader/hough-success.png
    blog/src/assets/blog/qr-reader/hough-failure.png
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
from matplotlib.colors import PowerNorm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "src"))

from hough_legacy import extract_thin_edges, hough_vote_peaks, refine_line  # noqa: E402
from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.qr_gen import binarize_image  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

# --- Detector settings (match the post) -------------------------------------
CROP_SCALE = 1.9        # crop side as a multiple of the 1:1:3:1:1 cluster width
BLUR_SIGMA = 1.0
THETA_STEP_DEG = 0.5
RHO_STEP = 0.5
THRESHOLD_REL = 0.25
MAX_PEAKS = 20
NMS_RADIUS_THETA = 2    # +/- 1.0 deg
NMS_RADIUS_RHO = 4      # +/- 2.0 px
GAP_TOLERANCE = 2.0
DISTANCE_THRESH = 1.5

# --- Figure style ------------------------------------------------------------
FIG_SIZE = (16.0, 4.8)
DPI = 125
LINE_COLORS = ["#ffd23f", "#ff6b6b", "#4cc9f0", "#c77dff"]
CORNER_COLOR = "#ff3fa4"

CASES = {
    "success": {"image": "qr_code_ - 16.jpeg", "cluster": 2},
    "failure": {"image": "qr_code_ - 97.jpeg", "cluster": 0},
}


def angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected line angles, modulo pi."""
    d = abs((a - b) % np.pi)
    return min(d, np.pi - d)


def order_quad(corners: np.ndarray) -> np.ndarray:
    c = corners.mean(axis=0)
    ang = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    return corners[np.argsort(ang)]


def crop_candidate(gray: np.ndarray, image_name: str, cluster_idx: int):
    """Run the alignment scan, then crop the requested candidate cluster."""
    binary = binarize_image(gray)
    rows, cols_all = find_alignment_patterns_2d(binary, np.log(1.3))
    clusters = cluster_candidates(rows, cols_all)
    if cluster_idx >= len(clusters):
        raise ValueError(
            f"{image_name}: cluster {cluster_idx} requested, "
            f"only {len(clusters)} found"
        )
    cluster = clusters[cluster_idx]

    width = float(cluster.cols[5] - cluster.cols[0])
    side = max(8, int(round(CROP_SCALE * width)))
    center_row = float(cluster.row)
    center_col = float((cluster.cols[2] + cluster.cols[3]) / 2.0)

    h, w = gray.shape
    side = min(side, h, w)
    r0 = max(0, min(int(round(center_row - side / 2)), h - side))
    c0 = max(0, min(int(round(center_col - side / 2)), w - side))
    roi = np.ascontiguousarray(gray[r0:r0 + side, c0:c0 + side])

    center = np.array([center_col - c0, center_row - r0], dtype=np.float64)
    return roi, center, width


def select_four(segments, normals, rhos, scores, center, s):
    """Select the four Hough lines that best form the expected outer square.

    The scan that produced the crop already tells us the finder's centre and
    its 7-module width, so ``s`` (half the width) is known without knowing the
    corners.  The four outer edges are the strongest lines on either side of
    the centre in two near-orthogonal orientation families.

    Returns ``(segment_indices, corner_points)`` or ``None``.
    """
    if len(segments) < 4:
        return None

    thetas = np.arctan2(normals[:, 1], normals[:, 0]) % np.pi
    sides = normals @ center - rhos
    principal = thetas[int(np.argmax(scores))]

    best = None
    for theta0 in (principal, (principal + np.pi / 2) % np.pi):
        families = []
        ok = True
        for theta in (theta0, (theta0 + np.pi / 2) % np.pi):
            family = []
            for sign in (+1, -1):
                candidates = [
                    i
                    for i in range(len(segments))
                    if angle_diff(thetas[i], theta) < np.deg2rad(14.0)
                    and sign * sides[i] > 0.20 * s
                    and abs(sides[i]) < 1.05 * s
                ]
                if not candidates:
                    ok = False
                    break
                family.append(
                    max(
                        candidates,
                        key=lambda i: scores[i]
                        * np.exp(-0.5 * ((abs(sides[i]) - s) / (0.30 * s)) ** 2),
                    )
                )
            if not ok:
                break
            families.append(family)
        if not ok:
            continue

        quad_idx = families[0] + families[1]
        corners = []
        for a in families[0]:
            for b in families[1]:
                M = np.array([normals[a], normals[b]])
                if abs(np.linalg.det(M)) < 1e-6:
                    ok = False
                    break
                corners.append(np.linalg.solve(M, np.array([rhos[a], rhos[b]])))
            if not ok:
                break
        if not ok or len(corners) != 4:
            continue

        corners = order_quad(np.asarray(corners))
        side_lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if side_lengths.min() < 0.70 * s or side_lengths.max() > 2.60 * s:
            continue
        if np.linalg.norm(corners.mean(axis=0) - center) > 0.60 * s:
            continue

        value = float(sum(scores[quad_idx]))
        if best is None or value > best[0]:
            best = (value, quad_idx, corners)

    return None if best is None else (best[1], best[2])


def analyse(image_name: str, cluster_idx: int):
    img_bgr = cv2.imread(str(DATA_DIR / image_name))
    if img_bgr is None:
        raise FileNotFoundError(DATA_DIR / image_name)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    roi, center, width = crop_candidate(gray, image_name, cluster_idx)
    nms, angle = extract_thin_edges(roi, blur_sigma=BLUR_SIGMA)

    normals, rhos, scores, acc_data = hough_vote_peaks(
        nms,
        angle,
        theta_step_deg=THETA_STEP_DEG,
        rho_step=RHO_STEP,
        threshold_rel=THRESHOLD_REL,
        max_peaks=MAX_PEAKS,
        nms_radius_theta=NMS_RADIUS_THETA,
        nms_radius_rho=NMS_RADIUS_RHO,
        return_acc=True,
    )

    segments = []
    seg_normals = []
    seg_rhos = []
    seg_scores = []
    for normal, rho, score in zip(normals, rhos, scores):
        seg = refine_line(
            normal,
            float(rho),
            float(score),
            nms,
            angle,
            gap_tolerance=GAP_TOLERANCE,
            distance_thresh=DISTANCE_THRESH,
        )
        if not np.all(seg.endpoints == 0):
            segments.append(seg)
            seg_normals.append(normal)
            seg_rhos.append(float(rho))
            seg_scores.append(float(score))

    selection = select_four(
        segments,
        np.asarray(seg_normals),
        np.asarray(seg_rhos),
        np.asarray(seg_scores),
        center,
        width / 2.0,
    )
    if selection is None:
        raise RuntimeError(f"{image_name}: no four outer lines selected")
    sel_idx, corners = selection

    return dict(
        image=image_name,
        cluster=cluster_idx,
        roi=roi,
        nms=nms,
        acc=acc_data["acc"],
        normals=normals,
        rhos=rhos,
        scores=scores,
        segments=segments,
        sel_idx=sel_idx,
        corners=corners,
        center=center,
    )


def draw(case, cand, out_path: Path):
    roi = cand["roi"]
    nms = cand["nms"]
    acc = cand["acc"]
    normals = cand["normals"]
    rhos = cand["rhos"]
    segments = cand["segments"]
    sel_idx = cand["sel_idx"]
    corners = cand["corners"]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=FIG_SIZE,
        dpi=DPI,
        gridspec_kw={"width_ratios": [1, 1, 1.55, 1]},
    )
    ax0, ax1, ax2, ax3 = axes

    # 1 -- original crop.
    ax0.imshow(roi, cmap="gray", vmin=0, vmax=255)
    ax0.set_title("1  Original crop", fontsize=14, loc="left", pad=6)
    ax0.axis("off")

    # 2 -- thin NMS edge map, white on black.
    vmax = np.percentile(nms[nms > 0], 99) if np.any(nms > 0) else 1.0
    ax1.imshow(nms, cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax1.set_title("2  Sobel + NMS edges", fontsize=14, loc="left", pad=6)
    ax1.axis("off")

    # 3 -- Hough accumulator.
    n_theta, n_rho = acc.shape
    ax2.imshow(
        acc.T,
        origin="lower",
        aspect="auto",
        extent=[0, n_theta * THETA_STEP_DEG, 0, n_rho * RHO_STEP],
        cmap="magma",
        norm=PowerNorm(gamma=0.35, vmin=0, vmax=acc.max()),
        interpolation="nearest",
    )
    ax2.set_xlim(-2, 182)
    ax2.set_xlabel("angle θ (deg)", fontsize=13)
    ax2.set_ylabel("intercept ρ (px)", fontsize=13)
    ax2.set_title("3  Hough accumulator", fontsize=14, loc="left", pad=6)
    ax2.tick_params(labelsize=11)

    # All extracted peaks faint, the four selected peaks highlighted.
    for normal, rho in zip(normals, rhos):
        theta = np.rad2deg(np.arctan2(normal[1], normal[0])) % 180
        ax2.plot(theta, rho, "o", ms=3.2, mfc="none", mec="white", mew=0.6, alpha=0.5)
    for k, si in enumerate(sel_idx):
        seg = segments[si]
        theta = np.rad2deg(np.arctan2(seg.normal[1], seg.normal[0])) % 180
        ax2.plot(theta, seg.rho, "o", ms=11, mfc="none", mec=LINE_COLORS[k], mew=2.4)
        ax2.plot(theta, seg.rho, "o", ms=2.0, color=LINE_COLORS[k])

    # 4 -- selected segments and their intersections on the crop.
    ax3.imshow(roi, cmap="gray", vmin=0, vmax=255)
    h, w = roi.shape
    for k, si in enumerate(sel_idx):
        seg = segments[si]
        ep = seg.endpoints
        ax3.plot(
            [ep[0, 0], ep[1, 0]],
            [ep[0, 1], ep[1, 1]],
            color=LINE_COLORS[k],
            lw=3.0,
            solid_capstyle="round",
        )
        # Faint infinite-line extension, for seeing where the corners come from.
        base_pt = seg.rho * seg.normal
        direction = np.array([-seg.normal[1], seg.normal[0]])
        t = np.linspace(-1.2 * max(h, w), 1.2 * max(h, w), 2)
        ax3.plot(
            base_pt[0] + t * direction[0],
            base_pt[1] + t * direction[1],
            color=LINE_COLORS[k],
            lw=0.8,
            alpha=0.35,
            ls="--",
        )
    for point in corners:
        if 0 <= point[0] <= w and 0 <= point[1] <= h:
            ax3.plot(
                point[0],
                point[1],
                "o",
                ms=11,
                mfc="none",
                mec=CORNER_COLOR,
                mew=2.6,
            )
    ax3.set_title("4  Four lines → corners", fontsize=14, loc="left", pad=6)
    ax3.set_xlim(0, w)
    ax3.set_ylim(h, 0)
    ax3.axis("off")

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)

    print(f"{case}: {cand['image']} cluster {cand['cluster']}")
    print(
        f"  selected peak indices {list(sel_idx)} "
        f"at ρ = {[round(segments[i].rho, 2) for i in sel_idx]}"
    )
    print(f"  corners (x, y) = {np.round(corners, 1).tolist()}")
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory for the two PNGs (default: blog asset directory)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for case, spec in CASES.items():
        cand = analyse(spec["image"], spec["cluster"])
        draw(case, cand, args.out_dir / f"hough-{case}.png")


if __name__ == "__main__":
    main()
