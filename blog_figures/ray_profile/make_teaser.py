#!/usr/bin/env python3
"""Make the QR-reader card image from the ray-profile figure's source data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from make_ray_profile_figures import (
    CLUSTER_INDEX,
    IMAGE_PATH,
    NUM_RAYS,
    NUM_SAMPLES,
    RAY_ANGLE_DEG,
    REPO_ROOT,
    SIGMA,
    finder_soft_template,
    fit_rays,
    load_finder_crop,
)


def main() -> None:
    crop = load_finder_crop(IMAGE_PATH, CLUSTER_INDEX)
    fit = fit_rays(crop)
    index = round(RAY_ANGLE_DEG / 360 * NUM_RAYS) % NUM_RAYS
    theta, pitch = fit.theta_rad[index], fit.m[index]
    direction = np.array([np.cos(theta), np.sin(theta)])
    h, w = crop.roi.shape
    cx, cy = crop.center_xy
    dx, dy = direction
    x_max = min(fit.max_dist, (w - 1 - cx) / dx, (h - 1 - cy) / dy)
    samples = np.linspace(0, fit.max_dist, NUM_SAMPLES)
    valid = samples <= x_max
    dense = np.linspace(0, min(4.5 * pitch, x_max), 1000)

    fig = plt.figure(figsize=(12.8, 8), facecolor="none")
    image = fig.add_axes([0.0, 0.10, 0.50, 0.80])
    image.imshow(crop.roi, cmap="gray", interpolation="nearest")
    image.axis("off")
    end = crop.center_xy + 4.0 * pitch * direction
    image.add_patch(
        FancyArrowPatch(
            tuple(crop.center_xy),
            tuple(end),
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=5,
            color="#f28e2b",
        )
    )
    image.plot(cx, cy, "o", markersize=13, color="#f28e2b",
               markeredgecolor="white", markeredgewidth=2.5)

    graph = fig.add_axes([0.51, 0.13, 0.49, 0.74], facecolor="none")
    graph.plot(samples[valid], fit.profiles_norm[index][valid],
               color="#56819d", linewidth=4, solid_capstyle="round")
    graph.plot(dense, finder_soft_template(dense, pitch, sigma=SIGMA),
               color="#df4961", linewidth=7, solid_capstyle="round")
    graph.set_xlim(0, x_max)
    graph.set_ylim(-0.07, 1.07)
    graph.axis("off")

    output = REPO_ROOT / "blog" / "src" / "assets" / "teasers" / "qr-reader.png"
    fig.savefig(output, dpi=100, transparent=True)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
