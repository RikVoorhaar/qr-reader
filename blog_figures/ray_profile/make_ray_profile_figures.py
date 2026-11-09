#!/usr/bin/env python3
"""Generate the ray-profile figures for the QR-reader blog post.

Produces two static PNGs under ``blog/src/assets/blog/qr-reader/``:

* ``ray-profile-signal.png``    — a finder crop with one 20-degree ray, and the
  intensity profile along that ray with the fitted soft template overlaid.
* ``ray-profile-clustering.png`` — the same finder crop with all 36 half-rays,
  the fitted 3.5-module boundary points, and those points clustered into the
  four finder sides with a TLS segment per side.

The crop and the fitting steps mirror ``detector.detector._run_detection`` and
``detector.ray_fit.fit_finder_ray``.  Run from the repository root:

    python blog_figures/ray_profile/make_ray_profile_figures.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.detector.edge_fitting import (  # noqa: E402
    PITCH_CONSTANT,
    EdgeFitResult,
    _reorder_to_standard,
    compute_boundary_points,
    fit_finder_edges,
)
from qr_reader.detector.ray_fit import (  # noqa: E402
    finder_soft_template,
    fit_all_rays,
    normalize_roi_intensities,
    sample_ray_profiles,
)
from qr_reader.detector.roi import cluster_to_bbox, cutout  # noqa: E402
from qr_reader.qr_gen import binarize_image  # noqa: E402

# ── Figure configuration ─────────────────────────────────────────────────────

IMAGE_PATH = REPO_ROOT / "data" / "bench" / "captured" / "qr_code_ - 100.jpeg"
CLUSTER_INDEX = 1
OUTPUT_DIR = REPO_ROOT / "blog" / "src" / "assets" / "blog" / "qr-reader"

NUM_RAYS = 36
NUM_SAMPLES = 120
RAY_ANGLE_DEG = 20.0
SIGMA = 1.0

# Same cluster colours as the reference diagnostic script (scripts/ray-profile.py),
# ordered by side: left, right, top, bottom.
EDGE_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
RAY_COLOR = "#f28e2b"
PROFILE_COLOR = "#4c566a"
TEMPLATE_COLOR = "#d1495b"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


@dataclass
class FinderCrop:
    """One finder-pattern crop and the centre/pitch produced by the detector."""

    image_name: str
    cluster_index: int
    roi: np.ndarray
    center_xy: np.ndarray
    m_est: float
    bbox: np.ndarray


@dataclass
class RayFit:
    """Ray-fit intermediates for one crop."""

    theta_rad: np.ndarray
    profiles_norm: np.ndarray
    max_dist: float
    m: np.ndarray
    mse: np.ndarray
    success: np.ndarray
    edge_result: EdgeFitResult


def load_finder_crop(image_path: Path, cluster_index: int) -> FinderCrop:
    """Replicate the detector's crop/centre/pitch extraction for one cluster."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    img_binary = binarize_image(image_gray)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(
        img_binary, np.log(1.3)
    )
    clusters = cluster_candidates(rows_valid, cols_valid_all)
    if cluster_index >= len(clusters):
        raise IndexError(
            f"Cluster {cluster_index} requested, but only {len(clusters)} found"
        )

    cluster = clusters[cluster_index]
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(image_gray, bbox)
    if roi.size == 0:
        raise ValueError(f"Empty crop for cluster {cluster_index}")

    r0 = max(0, int(bbox[0]))
    c0 = max(0, int(bbox[2]))
    center_xy = np.array(
        [
            float(cluster.cols[2] + cluster.cols[3]) / 2.0 - c0,
            float(cluster.row) - r0,
        ],
        dtype=np.float64,
    )
    m_est = float(cluster.cols[5] - cluster.cols[0]) / 7.0

    return FinderCrop(
        image_name=image_path.name,
        cluster_index=cluster_index,
        roi=roi,
        center_xy=center_xy,
        m_est=m_est,
        bbox=bbox,
    )


def fit_rays(crop: FinderCrop) -> RayFit:
    """Run the same normalise → sample → per-ray m → boundary → cluster steps."""
    roi_norm, dark, bright = normalize_roi_intensities(
        crop.roi, crop.center_xy, crop.m_est
    )
    profiles, max_dist = sample_ray_profiles(
        crop.roi,
        crop.center_xy[0],
        crop.center_xy[1],
        num_rays=NUM_RAYS,
        num_samples=NUM_SAMPLES,
        ray_length=1.0,
    )
    span = max(bright - dark, 1.0)
    profiles_norm = np.clip((profiles - dark) / span, 0.0, 1.0)

    m, mse, success = fit_all_rays(profiles_norm, crop.m_est, max_dist, sigma=SIGMA)
    theta_rad = np.linspace(0.0, 2.0 * np.pi, NUM_RAYS, endpoint=False)
    boundary_points = compute_boundary_points(
        crop.center_xy, m, theta_rad, PITCH_CONSTANT
    )
    edge_result = fit_finder_edges(boundary_points)

    return RayFit(
        theta_rad=theta_rad,
        profiles_norm=profiles_norm,
        max_dist=max_dist,
        m=m,
        mse=mse,
        success=success,
        edge_result=edge_result,
    )


def _show_crop(ax: plt.Axes, crop: FinderCrop) -> None:
    """Draw the crop with image coordinates and no axis furniture."""
    h, w = crop.roi.shape
    ax.imshow(crop.roi, cmap="gray", extent=[0, w, h, 0], interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_centre(ax: plt.Axes, center_xy: np.ndarray) -> None:
    ax.plot(
        center_xy[0],
        center_xy[1],
        "o",
        markersize=9,
        markerfacecolor=RAY_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=10,
    )


def make_signal_figure(crop: FinderCrop, fit: RayFit, output_path: Path) -> None:
    """Two panels: one ray on the crop, and its 1-D intensity profile."""
    ray_index = int(round(RAY_ANGLE_DEG / 360.0 * NUM_RAYS)) % NUM_RAYS
    theta = fit.theta_rad[ray_index]
    m = fit.m[ray_index]
    profile = fit.profiles_norm[ray_index]
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    t_samples = np.linspace(0.0, fit.max_dist, NUM_SAMPLES)

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.24)

    # ── Panel 1: crop + the chosen ray ──
    ax_img = fig.add_subplot(gs[0, 0])
    _show_crop(ax_img, crop)

    ray_end = crop.center_xy + 4.0 * m * direction
    arrow = FancyArrowPatch(
        posA=tuple(crop.center_xy),
        posB=tuple(ray_end),
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.6,
        color=RAY_COLOR,
        shrinkA=0,
        shrinkB=0,
        zorder=8,
    )
    ax_img.add_patch(arrow)
    _draw_centre(ax_img, crop.center_xy)

    label_pos = ray_end + np.array([3.0, -15.0])
    ax_img.text(
        label_pos[0],
        label_pos[1],
        f"{RAY_ANGLE_DEG:.0f}\N{DEGREE SIGN}",
        color=RAY_COLOR,
        fontsize=15,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        zorder=11,
    )

    # ── Panel 2: intensity profile + fitted template ──
    ax_prof = fig.add_subplot(gs[0, 1])

    # Show the samples all the way to the edge of the crop, so the profile
    # runs into the background instead of stopping at the template mask.
    h, w = crop.roi.shape
    cx, cy = crop.center_xy
    dx, dy = direction
    t_exit = min(
        (w - 1 - cx) / dx if dx > 0 else (0 - cx) / dx,
        (h - 1 - cy) / dy if dy > 0 else (0 - cy) / dy,
    )
    x_max = min(fit.max_dist, t_exit)
    valid = t_samples <= x_max
    t_plot = t_samples[valid]
    profile_plot = profile[valid]

    # The template is only meaningful inside the fitted mask.
    t_dense = np.linspace(0.0, min(4.5 * m, x_max), 1000)

    ax_prof.plot(
        t_plot,
        profile_plot,
        "-",
        color=PROFILE_COLOR,
        linewidth=1.0,
        alpha=0.55,
        zorder=2,
    )
    ax_prof.plot(
        t_plot[::3],
        profile_plot[::3],
        "o",
        color=PROFILE_COLOR,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.9,
        label="raw samples",
        zorder=3,
    )
    ax_prof.plot(
        t_dense,
        finder_soft_template(t_dense, m, sigma=SIGMA),
        color=TEMPLATE_COLOR,
        linewidth=2.6,
        label="fitted template",
        zorder=4,
    )

    for multiple in (1.5, 2.5, 3.5):
        t_j = multiple * m
        ax_prof.axvline(
            t_j,
            color=TEMPLATE_COLOR,
            linestyle=":",
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )
        ax_prof.plot(
            [t_j],
            [float(finder_soft_template(np.array([t_j]), m, sigma=SIGMA)[0])],
            "o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=TEMPLATE_COLOR,
            markeredgewidth=1.7,
            zorder=6,
        )

    ax_prof.set_xlim(0.0, x_max)
    ax_prof.set_ylim(-0.04, 1.10)
    ax_prof.set_xlabel("distance from centre (px)")
    ax_prof.set_ylabel("intensity (0 = black, 1 = white)")
    ax_prof.grid(alpha=0.16, linewidth=0.7)
    ax_prof.legend(loc="lower right", frameon=True, framealpha=0.95)
    ax_prof.spines[["top", "right"]].set_visible(False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")


def _side_order(edge_result: EdgeFitResult) -> list[int]:
    """Return the four cluster indices ordered left, right, top, bottom."""
    try:
        return list(_reorder_to_standard(edge_result.clusters))
    except Exception:
        return list(range(len(edge_result.clusters)))


def make_clustering_figure(crop: FinderCrop, fit: RayFit, output_path: Path) -> None:
    """Three panels: rays, boundary points, clustered points + TLS segments."""
    edge_result = fit.edge_result
    valid = np.all(np.isfinite(edge_result.points), axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.02)

    # ── Panel 1: all 36 half-rays ──
    ax = axes[0]
    _show_crop(ax, crop)
    ray_end_distance = 4.0 * crop.m_est
    for theta in fit.theta_rad:
        direction = np.array([np.cos(theta), np.sin(theta)])
        end = crop.center_xy + ray_end_distance * direction
        ax.plot(
            [crop.center_xy[0], end[0]],
            [crop.center_xy[1], end[1]],
            color="#2f80b7",
            linewidth=0.9,
            alpha=0.65,
            zorder=5,
        )
    _draw_centre(ax, crop.center_xy)

    # ── Panel 2: fitted 3.5-module boundary points ──
    ax = axes[1]
    _show_crop(ax, crop)
    ax.scatter(
        edge_result.points[valid, 0],
        edge_result.points[valid, 1],
        s=55,
        facecolor=RAY_COLOR,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    _draw_centre(ax, crop.center_xy)

    # ── Panel 3: clustered points + TLS segment per side ──
    ax = axes[2]
    _show_crop(ax, crop)
    order = _side_order(edge_result)
    for side, cluster_idx in enumerate(order):
        if cluster_idx >= len(edge_result.clusters):
            continue
        color = EDGE_COLORS[side % len(EDGE_COLORS)]
        assigned = edge_result.assignment == cluster_idx
        if np.any(assigned):
            ax.scatter(
                edge_result.points[assigned, 0],
                edge_result.points[assigned, 1],
                s=55,
                facecolor=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=7,
            )

        edge = edge_result.clusters[cluster_idx]
        support = edge_result.points[edge.support]
        projection = support @ edge.direction
        lo, hi = float(projection.min()), float(projection.max())
        extension = 0.18 * (hi - lo)
        t_line = np.array([lo - extension, hi + extension])
        line_points = edge.rho * edge.normal + t_line[:, None] * edge.direction

        # White underlay keeps the fitted segment legible on dark modules.
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            color="white",
            linewidth=5.2,
            solid_capstyle="round",
            alpha=0.9,
            zorder=4,
        )
        ax.plot(
            line_points[:, 0],
            line_points[:, 1],
            color=color,
            linewidth=2.6,
            solid_capstyle="round",
            zorder=5,
        )

    outliers = edge_result.assignment < 0
    if np.any(outliers):
        ax.scatter(
            edge_result.points[outliers, 0],
            edge_result.points[outliers, 1],
            s=55,
            facecolor="#9aa0a6",
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )

    _draw_centre(ax, crop.center_xy)

    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crop = load_finder_crop(IMAGE_PATH, CLUSTER_INDEX)
    fit = fit_rays(crop)

    print(
        f"crop: {crop.image_name}, cluster {crop.cluster_index}, "
        f"ROI {crop.roi.shape[1]}x{crop.roi.shape[0]}, "
        f"centre=({crop.center_xy[0]:.1f}, {crop.center_xy[1]:.1f}), "
        f"m_est={crop.m_est:.2f}px"
    )
    print(
        f"rays: {int(np.isfinite(fit.m).sum())}/{NUM_RAYS} valid; "
        f"edge clusters: {len(fit.edge_result.clusters)}; "
        f"unassigned points: {int(np.sum(fit.edge_result.assignment < 0))}"
    )

    make_signal_figure(
        crop, fit, OUTPUT_DIR / "ray-profile-signal.png"
    )
    make_clustering_figure(
        crop, fit, OUTPUT_DIR / "ray-profile-clustering.png"
    )


if __name__ == "__main__":
    main()
