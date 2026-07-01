"""Deep Hough diagnostic: why do strong NMS edges fail to produce detected lines?

For each cluster ROI, generates:
  1. Edge extraction: L2 gradient, NMS, angle histogram
  2. Hough accumulator heatmap with peaks + GT edges marked
  3. Rho-vs-theta scatter of per-pixel votes
  4. Per-peak support pixel maps
  5. ROI overlay with detected lines + GT edges

Usage::

    .venv/bin/python -m qr_reader.scripts.debug_hough_fn
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.hough import build_hough_accumulator, hough_vote_peaks, refine_line
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.config import AugmentationConfig

OUT = Path("out/debug_hough_fn")


def _normal_to_theta(normal):
    rad = float(np.arctan2(normal[1], normal[0]))
    if rad < 0:
        rad += np.pi
    return rad


def _theta_to_idx(theta, theta_step_rad, n_theta):
    return int(np.round(theta / theta_step_rad)) % n_theta


def _edge_normal_from_points(a, b):
    d = b - a
    length = np.linalg.norm(d)
    if length < 1e-12:
        return np.array([1.0, 0.0]), 0.0
    direction = d / length
    normal = np.array([direction[1], -direction[0]])
    rho = float(normal @ a)
    if rho < 0:
        normal = -normal
        rho = -rho
    return normal, rho


def _clip_segment(p0, p1, xmin, xmax, ymin, ymax):
    # Cohen-Sutherland
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _code(x, y):
        c = INSIDE
        if x < xmin: c |= LEFT
        elif x > xmax: c |= RIGHT
        if y < ymin: c |= TOP
        elif y > ymax: c |= BOTTOM
        return c

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    c0 = _code(x0, y0)
    c1 = _code(x1, y1)
    while True:
        if (c0 | c1) == 0:
            return np.array([[x0, y0], [x1, y1]])
        if (c0 & c1) != 0:
            return None
        oc = c0 if c0 != 0 else c1
        x = y = 0.0
        if oc & TOP:
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymin
        elif oc & BOTTOM:
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0) if y1 != y0 else x0
            y = ymax
        elif oc & RIGHT:
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmax
        elif oc & LEFT:
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0) if x1 != x0 else y0
            x = xmin
        if oc == c0:
            x0, y0 = x, y
            c0 = _code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = _code(x1, y1)


def _compute_gt_edges(metadata, roi_offset, roi_shape):
    corners = metadata["corners_qr"]
    N = metadata["N"]
    frac = 7.0 / N
    TL = np.array(corners["TL"])
    TR = np.array(corners["TR"])
    BR = np.array(corners["BR"])
    BL = np.array(corners["BL"])
    specs = [
        (TL, TR, "TL_top"), (TL, BL, "TL_left"),
        (TR, TL, "TR_top"), (TR, BR, "TR_right"),
        (BL, TL, "BL_left"), (BL, BR, "BL_bottom"),
        (BR, TR, "BR_right"), (BR, BL, "BR_bottom"),
    ]
    r0, c0 = int(roi_offset[0]), int(roi_offset[1])
    H, W = int(roi_shape[0]), int(roi_shape[1])
    offset_xy = np.array([c0, r0], dtype=np.float64)
    results = []
    for start, toward, label in specs:
        a = start
        b = start + frac * (toward - start)
        normal, rho = _edge_normal_from_points(a, b)
        rho_local = float(rho - normal @ offset_xy)
        if rho_local < 0:
            rho_local = -rho_local
            normal_local = -normal
        else:
            normal_local = normal.copy()
        clipped = _clip_segment(a - offset_xy, b - offset_xy, 0.0, W - 1, 0.0, H - 1)
        results.append({
            "label": label, "normal": normal_local, "rho": rho_local,
            "segment": clipped,
        })
    return results


def _draw_line_on_ax(ax, normal, rho, H, W, **kwargs):
    nx, ny = normal
    eps = 1e-9
    pts = []
    if abs(ny) > eps:
        y0 = rho / ny
        if 0 <= y0 < H: pts.append((0.0, y0))
        yw = (rho - nx * (W - 1)) / ny
        if 0 <= yw < H: pts.append((W - 1, yw))
    if abs(nx) > eps:
        x0 = rho / nx
        if 0 <= x0 < W: pts.append((x0, 0.0))
        xh = (rho - ny * (H - 1)) / nx
        if 0 <= xh < W: pts.append((xh, H - 1))
    if len(pts) < 2:
        return
    p = np.array(pts[:2])
    ax.plot([p[0, 0], p[1, 0]], [p[0, 1], p[1, 1]], **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Generate sample with metadata
    rng = np.random.default_rng(42)
    config = AugmentationConfig(
        version=12, content="https://www.rikvoorhaar.com",
        error_correction="M", ppm_range=(5.0, 12.0), target_ppm_range=(4.0, 10.0),
        jitter_fraction=0.15, feather_sigma_range=(0.5, 2.0),
        blur_sigma_range=(0.2, 1.0), noise_sigma_range=(1.0, 5.0),
        jpeg_quality_range=(65, 95), global_seed=42, rotation_deg_range=(-180, 180),
    )
    bg_h, bg_w = 640, 640
    xx = np.linspace(0, 1, bg_w, dtype=np.float32).reshape(1, -1)
    yy = np.linspace(0, 1, bg_h, dtype=np.float32).reshape(-1, 1)
    bg_val = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
    background = np.stack([bg_val] * 3, axis=-1)

    image, metadata = generate_sample(rng, config, background)
    gray = np.asarray(image[:, :, 0], dtype=float) if image.ndim == 3 else image

    img_binary = binarize_image(gray.astype(np.uint8))
    max_error = np.log(1.3)
    rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)
    clusters = cluster_candidates(rows_valid, cols_valid_all)
    print(f"Found {len(clusters)} clusters")

    for ci, cluster in enumerate(clusters):
        bbox = cluster_to_bbox(cluster, scale=1.5)
        roi = cutout(gray.astype(np.uint8), bbox)
        if roi.size == 0:
            print(f"  Cluster {ci}: empty ROI, skip")
            continue

        nms, angle = extract_thin_edges(roi, blur_sigma=1.0)
        H_roi, W_roi = roi.shape

        # Hough params
        THETA_STEP_DEG = 0.5
        RHO_STEP = 1.0
        NMS_RHO = 2
        NMS_THETA = 2

        # GT edges
        gt_edges = _compute_gt_edges(metadata, (bbox[0], bbox[2]), roi.shape)
        gt_labels = [e["label"] for e in gt_edges if e["segment"] is not None]

        # Hough
        normals, rhos, scores, acc_data = hough_vote_peaks(
            nms, angle, theta_step_deg=THETA_STEP_DEG, rho_step=RHO_STEP,
            nms_radius_rho=NMS_RHO, nms_radius_theta=NMS_THETA, return_acc=True,
            theta_window_deg=0.0, vote_scheme="onebin",
        )
        acc = acc_data["acc"]
        n_theta = acc_data["n_theta"]
        n_rho = acc_data["n_rho"]
        theta_step = acc_data["theta_step_rad"]
        theta_step_deg = np.rad2deg(theta_step)

        print(f"  Cluster {ci}: {len(normals)} peaks, {len(gt_labels)} GT edges")

        # --- refine each peak ---
        segments = []
        for n, r, s in zip(normals, rhos, scores):
            seg = refine_line(n, float(r), float(s), nms, angle,
                              gap_tolerance=3.0, distance_thresh=1.5,
                              support_dilate=0)
            segments.append(seg)

        # =============================================================
        # FIGURE: 3-row diagnostic
        # Row 1: Edge extraction (grayscale, NMS, angle hist)
        # Row 2: Accumulator heatmap + rho-vs-theta scatter
        # Row 3: Per-peak support maps
        # =============================================================
        n_peaks = len(normals)
        fig = plt.figure(figsize=(16, 4 + 4 + 3 * max(1, (n_peaks + 1) // 2)))
        gs = fig.add_gridspec(
            3 + (n_peaks + 1) // 2, max(4, n_peaks),
            hspace=0.5, wspace=0.3,
        )

        # --- Row 1: edge extraction ---
        ax_gray = fig.add_subplot(gs[0, 0])
        ax_gray.imshow(roi, cmap="gray")
        ax_gray.set_title(f"C{ci} — Grayscale")
        ax_gray.axis("off")

        ax_nms = fig.add_subplot(gs[0, 1])
        ax_nms.imshow(nms, cmap="gray")
        ax_nms.set_title(f"NMS ({np.count_nonzero(nms)} px, max={nms.max():.0f})")
        ax_nms.axis("off")

        # L2 gradient (raw sobel before NMS)
        from scipy import ndimage
        roi_f = roi.astype(np.float64)
        blurred = ndimage.gaussian_filter(roi_f, sigma=1.0, mode="reflect")
        gx = ndimage.sobel(blurred, axis=1, mode="constant")
        gy = ndimage.sobel(blurred, axis=0, mode="constant")
        mag = np.hypot(gx, gy)
        ax_mag = fig.add_subplot(gs[0, 2])
        ax_mag.imshow(mag, cmap="gray")
        ax_mag.set_title("Sobel L2")
        ax_mag.axis("off")

        # Angle histogram with GT markers
        ax_ah = fig.add_subplot(gs[0, 3])
        ys_nz, xs_nz = np.nonzero(nms)
        thetas = np.fmod(angle[ys_nz, xs_nz], np.pi)
        thetas = np.where(thetas < 0, thetas + np.pi, thetas)
        strengths = nms[ys_nz, xs_nz]
        ax_ah.hist(np.rad2deg(thetas), bins=90, range=(0, 180),
                    weights=strengths, color="steelblue", edgecolor="white", alpha=0.8)
        colors = plt.cm.tab10.colors
        for i, gt in enumerate(gt_edges):
            if gt["segment"] is None:
                continue
            gt_th = np.rad2deg(_normal_to_theta(gt["normal"]))
            ax_ah.axvline(gt_th, color=colors[i % 10], linestyle="--",
                          linewidth=1.5, label=gt["label"])
            # Also mark peak angles
        for i in range(len(normals)):
            p_th = np.rad2deg(_normal_to_theta(normals[i]))
            ax_ah.axvline(p_th, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
        ax_ah.set_xlabel("Angle (deg)")
        ax_ah.set_title("NMS angle histogram + GT edges")
        if len(gt_labels) <= 10:
            ax_ah.legend(fontsize=6, loc="upper right")

        # --- Row 2: accumulator + rho-vs-theta ---
        ax_acc = fig.add_subplot(gs[1, :2])
        extent_acc = [0, min(n_theta * theta_step_deg, 180), 0, n_rho * RHO_STEP]
        im = ax_acc.imshow(acc.T, origin="lower", aspect="auto",
                           extent=extent_acc,
                           cmap="inferno", vmax=acc.max() * 0.3)
        plt.colorbar(im, ax=ax_acc, label="Votes", fraction=0.046, pad=0.04)
        ax_acc.set_title(f"C{ci} — Hough Accumulator (θ-step={theta_step_deg}°, ρ-step={RHO_STEP}px)")
        ax_acc.set_xlabel("θ (deg)"); ax_acc.set_ylabel("ρ (px)")

        # GT edges
        for i, gt in enumerate(gt_edges):
            if gt["segment"] is None:
                continue
            gt_th = np.rad2deg(_normal_to_theta(gt["normal"]))
            ax_acc.plot(gt_th, gt["rho"], "o", color=colors[i % 10],
                        markersize=10, markeredgewidth=2, markeredgecolor="white",
                        label=gt["label"])
        # Detected peaks
        for i in range(len(normals)):
            p_th = np.rad2deg(_normal_to_theta(normals[i]))
            ax_acc.plot(p_th, rhos[i], "x", color="cyan", markersize=8, markeredgewidth=2)
        if len(gt_labels) <= 10:
            ax_acc.legend(fontsize=6, loc="upper right")

        # Rho-vs-theta vote scatter (zoom window)
        ax_rvt = fig.add_subplot(gs[1, 2:])
        theta_idx = acc_data["theta_idx"]
        rho_idx = acc_data["rho_idx"]
        v_strengths = acc_data["strengths"]
        valid = (theta_idx >= 0) & (rho_idx >= 0)
        if valid.any():
            tv = theta_idx[valid].astype(float) * theta_step_deg
            rv = rho_idx[valid].astype(float)
            sv = v_strengths[valid]
            sc = ax_rvt.scatter(tv, rv, c=sv, s=2 * np.sqrt(sv / (sv.max() + 0.1)),
                                cmap="inferno", alpha=0.4)
            plt.colorbar(sc, ax=ax_rvt, label="Strength", fraction=0.046, pad=0.04)
        ax_rvt.set_xlim(0, 180)
        ax_rvt.set_ylim(0, n_rho * RHO_STEP)
        ax_rvt.set_title("ρ-vs-θ vote cloud")
        ax_rvt.set_xlabel("θ (deg)"); ax_rvt.set_ylabel("ρ (px)")

        # Mark GT edges with circles
        for i, gt in enumerate(gt_edges):
            if gt["segment"] is None:
                continue
            gt_th = np.rad2deg(_normal_to_theta(gt["normal"]))
            ax_rvt.plot(gt_th, gt["rho"], "o", color=colors[i % 10],
                        markersize=6, markeredgewidth=1.5, markeredgecolor="white",
                        label=gt["label"])
        # Detected peaks
        for i in range(len(normals)):
            p_th = np.rad2deg(_normal_to_theta(normals[i]))
            ax_rvt.plot(p_th, rhos[i], "x", color="cyan", markersize=6, markeredgewidth=2)
        if len(gt_labels) <= 10:
            ax_rvt.legend(fontsize=5, loc="upper left")

        # --- Rows 3+: per-peak support maps ---
        for pi in range(min(n_peaks, 12)):
            row = 2 + pi // 4
            col = pi % 4
            seg = segments[pi]
            ax_s = fig.add_subplot(gs[row, col])
            ax_s.imshow(roi, cmap="gray", alpha=0.4)
            ax_s.set_xlim(0, W_roi)
            ax_s.set_ylim(H_roi, 0)

            # Show NMS pixels within distance_thresh of this line
            ys_s, xs_s = np.nonzero(nms)
            pts = np.column_stack([xs_s.astype(float), ys_s.astype(float)])
            dists = np.abs(pts @ seg.normal - seg.rho)
            inlier = dists < 1.5
            ax_s.scatter(xs_s[inlier], ys_s[inlier], c="lime", s=3, alpha=0.7, label="support")
            ax_s.scatter(xs_s[~inlier], ys_s[~inlier], c="gray", s=1, alpha=0.15, label="other")

            # Refined segment
            if not np.all(seg.endpoints == 0):
                ax_s.plot([seg.endpoints[0, 0], seg.endpoints[1, 0]],
                          [seg.endpoints[0, 1], seg.endpoints[1, 1]],
                          "r-", linewidth=3, label="segment")
            # Infinite line
            _draw_line_on_ax(ax_s, normals[pi], rhos[pi], H_roi, W_roi,
                             linestyle="--", linewidth=1, color="yellow", alpha=0.5)

            n_support = inlier.sum()
            peak_theta = np.rad2deg(_normal_to_theta(normals[pi]))
            status = "OK" if not np.all(seg.endpoints == 0) else "DEGEN"
            ax_s.set_title(f"P{pi}: θ={peak_theta:.0f}° ρ={rhos[pi]:.0f} "
                           f"supp={n_support} [{status}]", fontsize=7)
            ax_s.axis("off")

        fig.suptitle(f"Cluster {ci} — Deep Hough Diagnostics (v12, seed=42)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_png = OUT / f"cluster_{ci:02d}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  Saved {out_png}")

        # -------------------------------------------------
        # Per-GT-edge accumulation zoom
        # -------------------------------------------------
        n_gt_with_seg = len([e for e in gt_edges if e["segment"] is not None])
        if n_gt_with_seg == 0:
            continue

        n_r = int(np.ceil(n_gt_with_seg / 3))
        fig2, axes2 = plt.subplots(
            n_r, 3, figsize=(14, 5 * n_r), squeeze=False,
        )
        fig2.suptitle(f"Cluster {ci} — Per-GT-Edge Accumulator Zoom (θ-step={theta_step_deg}°)",
                      fontsize=11, fontweight="bold")
        zoom_deg = 12
        zoom_rho = 15
        dtheta_bins = max(1, int(np.ceil(zoom_deg / theta_step_deg)))
        d_rho_bins = max(1, int(np.ceil(zoom_rho / RHO_STEP)))

        plot_i = 0
        for ei, gt in enumerate(gt_edges):
            if gt["segment"] is None:
                continue
            ax_gt = axes2[plot_i // 3, plot_i % 3]
            gt_theta = _normal_to_theta(gt["normal"])
            gt_ti = _theta_to_idx(gt_theta, theta_step, n_theta)
            gt_ri = int(np.round(gt["rho"] / RHO_STEP))
            gt_ri = max(0, min(n_rho - 1, gt_ri))

            t0 = max(0, gt_ti - dtheta_bins)
            t1 = min(n_theta, gt_ti + dtheta_bins + 1)
            r0 = max(0, gt_ri - d_rho_bins)
            r1 = min(n_rho, gt_ri + d_rho_bins + 1)

            zoom_acc = acc[t0:t1, r0:r1]
            if zoom_acc.size == 0:
                ax_gt.text(0.5, 0.5, "out of bounds", transform=ax_gt.transAxes, ha="center")
                ax_gt.set_title(gt["label"])
                plot_i += 1
                continue

            im = ax_gt.imshow(zoom_acc.T, origin="lower", aspect="auto",
                              extent=[t0*theta_step_deg, t1*theta_step_deg,
                                      r0*RHO_STEP, r1*RHO_STEP],
                              cmap="inferno")
            plt.colorbar(im, ax=ax_gt, fraction=0.046, pad=0.04)

            # GT position
            ax_gt.plot(gt_ti * theta_step_deg, gt["rho"], "o", color="lime",
                       markersize=10, markeredgewidth=2, markeredgecolor="white",
                       label=f"GT ({gt['label']})")

            # Detected peaks near GT
            for pi in range(len(normals)):
                p_th = np.rad2deg(_normal_to_theta(normals[pi]))
                p_ri = int(np.round(rhos[pi] / RHO_STEP))
                if r0 <= p_ri <= r1:
                    ax_gt.plot(p_th, rhos[pi], "x", color="cyan", markersize=10,
                               markeredgewidth=2)

            # Peak bin value
            gt_val = acc[gt_ti, gt_ri]
            peak_max = zoom_acc.max()
            ax_gt.set_title(
                f"{gt['label']}: GT-bin={gt_val:.0f}, max-in-zoom={peak_max:.0f}, "
                f"ratio={gt_val/max(peak_max,0.1):.3f}",
                fontsize=8,
            )
            ax_gt.set_xlabel("θ (deg)"); ax_gt.set_ylabel("ρ (px)")
            ax_gt.legend(fontsize=6)
            plot_i += 1

        # Hide unused subplots
        for ei in range(plot_i, axes2.size):
            axes2[ei // 3, ei % 3].set_visible(False)

        plt.tight_layout()
        out_png2 = OUT / f"cluster_{ci:02d}_per_gt_zoom.png"
        fig2.savefig(out_png2, dpi=150)
        plt.close(fig2)
        print(f"  Saved {out_png2}")

        # -------------------------------------------------
        # Summary analysis: which GT edges are missed and why
        # -------------------------------------------------
        print(f"\n  --- Cluster {ci} GT-edge analysis ---")
        threshold_rel = 0.25
        acc_max = acc.max()
        threshold = threshold_rel * acc_max

        angle_tol_deg = 5.0
        rho_tol = 5.0
        angle_tol_rad = np.deg2rad(angle_tol_deg)

        for gt in gt_edges:
            if gt["segment"] is None:
                print(f"    {gt['label']}: no ROI overlap")
                continue
            gt_theta = _normal_to_theta(gt["normal"])
            gt_ti = _theta_to_idx(gt_theta, theta_step, n_theta)
            gt_ri = int(np.round(gt["rho"] / RHO_STEP))
            gt_ri = max(0, min(n_rho - 1, gt_ri))
            gt_val = float(acc[gt_ti, gt_ri])

            # Check if any detected peak matches
            matched = False
            best_dist = float("inf")
            for pi in range(len(normals)):
                p_th = _normal_to_theta(normals[pi])
                ang_dist = min(abs(p_th - gt_theta) % np.pi,
                               np.pi - abs(p_th - gt_theta) % np.pi)
                rho_dist = abs(rhos[pi] - gt["rho"])
                if ang_dist <= angle_tol_rad and rho_dist <= rho_tol:
                    matched = True
                    best_dist = min(best_dist, ang_dist + rho_dist)

            # Window sum check
            window_sum = 0.0
            dth = max(1, int(np.ceil(angle_tol_deg / theta_step_deg)))
            dr = max(1, int(np.ceil(rho_tol / RHO_STEP)))
            for dt in range(-dth, dth + 1):
                tt = (gt_ti + dt) % n_theta
                window_sum += float(acc[tt, max(0, gt_ri - dr):min(n_rho, gt_ri + dr + 1)].sum())

            # Peak in theta band at GT rho region
            r_rho_band = max(1, int(np.ceil(5.0 / RHO_STEP)))
            rho_band = acc[:, max(0, gt_ri - r_rho_band):min(n_rho, gt_ri + r_rho_band + 1)].sum(axis=1)
            peak_theta_idx = int(np.argmax(rho_band))
            peak_theta_deg = peak_theta_idx * theta_step_deg

            if matched:
                print(
                    f"    {gt['label']}: HIT — GT-bin={gt_val:.0f}, "
                    f"θ={np.rad2deg(gt_theta):.1f}°, ρ={gt['rho']:.1f}, "
                    f"window_sum={window_sum:.0f}"
                )
            else:
                # Classify
                if window_sum < 10:
                    cls = "empty"
                    reason = f"vote sum in ±{angle_tol_deg}° × ±{rho_tol}px = {window_sum:.1f}"
                elif gt_val > 0 and gt_val < threshold:
                    cls = "vote_dilution"
                    reason = f"GT bin={gt_val:.0f} < threshold={threshold:.0f} (rel=0.25 × max={acc_max:.0f})"
                elif gt_val < 0.1 * acc_max:
                    cls = "theta_spread"
                    reason = f"votes at θ={peak_theta_deg:.1f}° not GT θ={np.rad2deg(gt_theta):.1f}° (diff={abs(peak_theta_deg - np.rad2deg(gt_theta)):.1f}°)"
                else:
                    cls = "nms_suppress"
                    reason = f"GT bin={gt_val:.0f} > threshold but no peak matched (NMS suppressed?)"
                print(
                    f"    {gt['label']}: MISS [{cls}] — {reason}, "
                    f"window_sum={window_sum:.0f}, GT-bin={gt_val:.0f}"
                )

        plt.close("all")
        print()

    plt.close("all")
    print(f"Done. Output in {OUT}/")


if __name__ == "__main__":
    main()
