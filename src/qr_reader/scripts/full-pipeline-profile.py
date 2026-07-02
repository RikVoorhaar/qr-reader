# %%
"""Diagnostic script — pipeline up to clusters, then finder-profile edge fitting per ROI.

Four figures per cluster:
  1. Orientation estimation: angle histogram mod π/2 + axes overlay
  2. 1D projection profiles + transition fitting
  3. TLS-refined outer lines + corner extraction
  4. Template fitting with polarity + contrast scoring
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates, CandidateCluster
from qr_reader.detector.edges import extract_thin_edges
from qr_reader.detector.finder_fit import (
    FinderFit,
    estimate_orientation,
    build_projection_profile,
    fit_finder_1d,
    refine_outer_line,
    intersect_lines,
    fit_finder_template,
)
from qr_reader.detector.homography import estimate_homography_dlt, project_points
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image


def _draw_infinite_line(normal, rho, H, W):
    nx, ny = normal
    eps = 1e-9
    points = []
    if abs(ny) > eps:
        y0 = rho / ny
        if 0 <= y0 < H:
            points.append((0.0, y0))
    if abs(ny) > eps:
        yw = (rho - nx * (W - 1)) / ny
        if 0 <= yw < H:
            points.append((float(W - 1), yw))
    if abs(nx) > eps:
        x0 = rho / nx
        if 0 <= x0 < W:
            points.append((x0, 0.0))
    if abs(nx) > eps:
        xh = (rho - ny * (H - 1)) / nx
        if 0 <= xh < W:
            points.append((xh, float(H - 1)))
    if len(points) < 2:
        return np.array([[0, 0], [0, 0]])
    return np.array(points[:2])


def _normal_theta_deg(normal):
    rad = float(np.arctan2(normal[1], normal[0]))
    if rad < 0:
        rad += np.pi
    return np.rad2deg(rad)


def _match_corners_min_dist(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """Compute per-corner errors by optimal permutation (min L2 distance).

    Returns errors in the order of *corners_a*.
    """
    n = corners_a.shape[0]
    dists = np.linalg.norm(
        corners_a[:, None, :] - corners_b[None, :, :], axis=2
    )
    # Greedy match: for each a, pick the closest b not yet assigned
    assigned = set()
    errors = np.full(n, np.nan)
    for i in range(n):
        order = np.argsort(dists[i])
        for j in order:
            if j not in assigned:
                errors[i] = dists[i, j]
                assigned.add(int(j))
                break
    return errors


def _compute_gt_geometry(metadata, roi_offset, roi_shape):
    """Compute GT finder geometry: center, module pitch, orientation, corners."""
    corners = metadata["corners_qr"]
    N = metadata["N"]

    src_xy = np.array(
        [[0.0, 0.0], [float(N), 0.0], [float(N), float(N)], [0.0, float(N)]],
        dtype=np.float64,
    )
    dst_xy = np.array(
        [
            [float(corners["TL"][0]), float(corners["TL"][1])],
            [float(corners["TR"][0]), float(corners["TR"][1])],
            [float(corners["BR"][0]), float(corners["BR"][1])],
            [float(corners["BL"][0]), float(corners["BL"][1])],
        ],
        dtype=np.float64,
    )
    H = estimate_homography_dlt(src_xy, dst_xy)

    def _grid_to_image(row: float, col: float) -> np.ndarray:
        pt = np.array([[col, row]], dtype=np.float64)
        return project_points(H, pt)[0]

    finder_positions: dict[str, tuple[int, int]] = {
        "TL": (0, 0),
        "TR": (0, N - 7),
        "BL": (N - 7, 0),
    }

    r0_off, c0_off = int(roi_offset[0]), int(roi_offset[1])
    offset_xy = np.array([c0_off, r0_off], dtype=np.float64)

    gt_data: dict[str, dict] = {}

    for finder_name, (r0, c0) in finder_positions.items():
        center_gt = 0.5 * (
            _grid_to_image(float(r0 + 3), float(c0 + 3))
            + _grid_to_image(float(r0 + 4), float(c0 + 4))
        )
        p_origin = _grid_to_image(float(r0), float(c0))
        p_one_x = _grid_to_image(float(r0), float(c0 + 1))
        m_gt_x = float(np.linalg.norm(p_one_x - p_origin))
        p_one_y = _grid_to_image(float(r0 + 1), float(c0))
        m_gt_y = float(np.linalg.norm(p_one_y - p_origin))
        m_gt = (m_gt_x + m_gt_y) / 2.0

        corners_gt = np.array([
            _grid_to_image(float(r0), float(c0)),
            _grid_to_image(float(r0), float(c0 + 7)),
            _grid_to_image(float(r0 + 7), float(c0 + 7)),
            _grid_to_image(float(r0 + 7), float(c0)),
        ])

        top_a = corners_gt[0]
        top_b = corners_gt[1]
        top_dir = top_b - top_a
        top_len = np.linalg.norm(top_dir)
        if top_len > 1e-12:
            top_dir = top_dir / top_len
        else:
            top_dir = np.array([1.0, 0.0])
        top_normal = np.array([top_dir[1], -top_dir[0]])
        phi_gt = float(np.arctan2(top_normal[1], top_normal[0])) % (np.pi / 2)

        center_local = center_gt - offset_xy
        corners_local = corners_gt - offset_xy

        gt_data[finder_name] = {
            "center": center_local,
            "m": m_gt,
            "phi": phi_gt,
            "corners": corners_local,
        }

    return gt_data, H


def _find_cluster_finder(
    cluster: CandidateCluster, clusters: list, gt_centers: dict,
    roi_offset: np.ndarray,
) -> str | None:
    """Match a cluster to the nearest GT finder center (TL/TR/BL)."""
    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - roi_offset[1]
    c_row = float(cluster.row) - roi_offset[0]
    cluster_xy = np.array([c_col, c_row])

    best = None
    best_dist = float("inf")
    for name, gd in gt_centers.items():
        d = float(np.linalg.norm(cluster_xy - gd["center"]))
        if d < best_dist:
            best_dist = d
            best = name
    return best


# %%
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample

rng = np.random.default_rng(42)
config = AugmentationConfig(
    version=12,
    content="https://www.rikvoorhaar.com",
    error_correction="M",
    ppm_range=(5.0, 12.0),
    target_ppm_range=(4.0, 10.0),
    jitter_fraction=0.15,
    feather_sigma_range=(0.5, 2.0),
    blur_sigma_range=(0.2, 1.0),
    noise_sigma_range=(1.0, 5.0),
    jpeg_quality_range=(65, 95),
    global_seed=42,
)
bg_h, bg_w = 640, 640
xx = np.linspace(0, 1, bg_w, dtype=np.float32).reshape(1, -1)
yy = np.linspace(0, 1, bg_h, dtype=np.float32).reshape(-1, 1)
bg_val = (200 + 55 * (xx + yy) / 2).clip(0, 255).astype(np.uint8)
background = np.stack([bg_val] * 3, axis=-1)

image, metadata = generate_sample(rng, config, background)
img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)
QR_VERSION = metadata["version"]
QR_CONTENT = metadata["payload"]
print(f"v{QR_VERSION} — {QR_CONTENT}")

# %%
img_binary = binarize_image(img_gray)

# %%
max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)

# %%
clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"Found {len(clusters)} clusters.")

# %%
for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)

    if roi.size == 0:
        print(f"  Cluster {ci}: empty ROI, skipping")
        continue

    nms, angle_npy = extract_thin_edges(roi, blur_sigma=1.0)
    H_roi, W_roi = roi.shape

    roi_offset = np.array([bbox[0], bbox[2]], dtype=np.float64)
    gt_geo, _H_gt = _compute_gt_geometry(metadata, roi_offset, roi.shape)
    finder_name = _find_cluster_finder(cluster, clusters, gt_geo, roi_offset)
    gt_info = gt_geo.get(finder_name) if finder_name else None

    c_col = float(cluster.cols[2] + cluster.cols[3]) / 2.0 - roi_offset[1]
    c_row = float(cluster.row) - roi_offset[0]
    center_xy = np.array([c_col, c_row], dtype=np.float64)

    cluster_width = float(cluster.cols[5] - cluster.cols[0])
    m_est = cluster_width / 7.0

    print(f"\n  Cluster {ci} ({finder_name or '?'}): center=({c_col:.1f}, {c_row:.1f}) "
          f"m_est={m_est:.2f}px  roi={W_roi}×{H_roi}")

    if finder_name and gt_info is not None:
        gt_c = gt_info["center"]
        print(f"    GT center=({gt_c[0]:.1f}, {gt_c[1]:.1f})  m_gt={gt_info['m']:.2f}px"
              f"  φ_gt={np.rad2deg(gt_info['phi']):.1f}°")

    # ===== Phase 1: Orientation =====
    phi, e1, e2 = estimate_orientation(nms, angle_npy, center_xy)
    phi_deg = np.rad2deg(phi)

    if gt_info is not None:
        phi_gt_deg = np.rad2deg(gt_info["phi"])
        axis_error = min(abs(phi_deg - phi_gt_deg),
                         90 - abs(phi_deg - phi_gt_deg))
        print(f"    φ_est={phi_deg:.1f}°  φ_gt={phi_gt_deg:.1f}°  "
              f"axis_error={axis_error:.1f}°")
    else:
        axis_error = None
        print(f"    φ_est={phi_deg:.1f}°  (no GT)")

    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f"Cluster {ci} ({finder_name or '?'}) — Phase 1")

    ax = axes[0, 0]
    ax.imshow(roi, cmap="gray")
    ax.set_title("Grayscale cutout")
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(nms, cmap="gray")
    ax.set_title(f"NMS edges (nonzero={np.count_nonzero(nms)})")
    ax.axis("off")

    ax = axes[1, 0]
    ys_nz, xs_nz = np.nonzero(nms)
    strengths = nms[ys_nz, xs_nz]
    alpha_mod = np.fmod(angle_npy[ys_nz, xs_nz], np.pi)
    alpha_mod = np.where(alpha_mod < 0, alpha_mod + np.pi, alpha_mod)
    alpha_mod = np.fmod(alpha_mod, np.pi / 2.0)
    ax.hist(np.rad2deg(alpha_mod), bins=45, range=(0, 90),
            weights=strengths, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(phi_deg, color="green", linestyle="-", linewidth=2, label="est φ")
    if gt_info is not None:
        ax.axvline(phi_gt_deg, color="red", linestyle="--", linewidth=2, label="GT φ")
    ax.set_xlabel("Angle mod π/2 (deg)")
    ax.set_title("Angle histogram (4-fold symmetry)")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.imshow(roi, cmap="gray")
    arrow_len = 3.5 * m_est
    ax.arrow(center_xy[0], center_xy[1],
             arrow_len * e1[0], arrow_len * e1[1],
             color="lime", width=2, head_width=4, label="e1")
    ax.arrow(center_xy[0], center_xy[1],
             arrow_len * e2[0], arrow_len * e2[1],
             color="cyan", width=2, head_width=4, label="e2")
    if gt_info is not None:
        gt_e1 = np.array([np.cos(gt_info["phi"]), np.sin(gt_info["phi"])])
        gt_e2 = np.array([-np.sin(gt_info["phi"]), np.cos(gt_info["phi"])])
        ax.arrow(gt_info["center"][0], gt_info["center"][1],
                 arrow_len * gt_e1[0], arrow_len * gt_e1[1],
                 color="red", width=1, head_width=3, linestyle="--", alpha=0.7)
        ax.arrow(gt_info["center"][0], gt_info["center"][1],
                 arrow_len * gt_e2[0], arrow_len * gt_e2[1],
                 color="orange", width=1, head_width=3, linestyle="--", alpha=0.7)
    ax.set_title("Estimated axes + GT")
    ax.axis("off")
    plt.tight_layout()

    # ===== Phase 2: 1D projection profiles =====
    pos_u, prof_u = build_projection_profile(nms, angle_npy, center_xy, e1, m_est)
    pos_v, prof_v = build_projection_profile(nms, angle_npy, center_xy, e2, m_est)

    fit_u = fit_finder_1d(prof_u, pos_u, m_est)
    fit_v = fit_finder_1d(prof_v, pos_v, m_est)

    m_fit = (fit_u["m_fitted"] + fit_v["m_fitted"]) / 2.0
    du_fit = fit_u["center_offset"]
    dv_fit = fit_v["center_offset"]
    fitted_center = center_xy + du_fit * e1 + dv_fit * e2

    if gt_info is not None:
        m_err_pct = 100.0 * (m_fit - gt_info["m"]) / gt_info["m"]
        centre_err = np.linalg.norm(fitted_center - gt_info["center"])
        print(f"    Phase 2: m_fit={m_fit:.2f} (err {m_err_pct:+.1f}%)  "
              f"centre_err={centre_err:.2f}px")
    else:
        print(f"    Phase 2: m_fit={m_fit:.2f}")

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f"Cluster {ci} — Phase 2")

    for idx, (pos, prof, fit, axis_label) in enumerate([
        (pos_u, prof_u, fit_u, "e1 (u)"),
        (pos_v, prof_v, fit_v, "e2 (v)"),
    ]):
        ax = axes[0, idx]
        if len(pos) > 0:
            bw = (pos[1] - pos[0]) if len(pos) > 1 else m_est / 4
            ax.bar(pos, prof, width=bw, color="steelblue", alpha=0.7)
        for p in fit["peak_positions"]:
            ax.axvline(p, color="green", linestyle="-", linewidth=2, alpha=0.8)
        if gt_info is not None:
            gt_offsets = np.array([-3.5, -2.5, -1.5, 1.5, 2.5, 3.5])
            gt_peaks = gt_offsets * gt_info["m"]
            for p in gt_peaks:
                ax.axvline(p, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(f"Position along {axis_label} (px)")
        ax.set_ylabel("Edge strength")
        ax.set_title(f"Profile along {axis_label}")

    ax = axes[1, 0]
    ax.imshow(roi, cmap="gray")
    for sign in [1, -1]:
        for axis in [e1, e2]:
            pos = sign * 3.5 * m_fit
            n = axis
            r = float(axis @ (fitted_center + pos * axis))
            pts = _draw_infinite_line(n, r, H_roi, W_roi)
            ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]],
                    color="green", linewidth=2, alpha=0.7)
    if gt_info is not None:
        for sign in [1, -1]:
            for axis in [e1, e2]:
                pos = sign * 3.5 * gt_info["m"]
                n = axis
                r = float(axis @ (gt_info["center"] + pos * axis))
                pts = _draw_infinite_line(n, r, H_roi, W_roi)
                ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]],
                        color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_title("Phase 2 outer lines (green) vs GT (red)")
    ax.axis("off")

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.1, 0.5, f"m_est={m_est:.2f}  m_fit={m_fit:.2f}\n"
            f"du={du_fit:.2f}  dv={dv_fit:.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="center")
    if gt_info is not None:
        ax.text(0.1, 0.2, f"m_gt={gt_info['m']:.2f}  centre_err={centre_err:.2f}px",
                transform=ax.transAxes, fontsize=9, verticalalignment="center",
                color="red")
    plt.tight_layout()

    # ===== Phase 3: Refinement + corners =====
    um = refine_outer_line(nms, angle_npy, fitted_center, e1, -3.5 * m_fit)[1]
    up = refine_outer_line(nms, angle_npy, fitted_center, e1, +3.5 * m_fit)[1]
    vm = refine_outer_line(nms, angle_npy, fitted_center, e2, -3.5 * m_fit)[1]
    vp = refine_outer_line(nms, angle_npy, fitted_center, e2, +3.5 * m_fit)[1]

    outer_lines = {
        "u+": (e1.copy(), up),
        "u-": (e1.copy(), um),
        "v+": (e2.copy(), vp),
        "v-": (e2.copy(), vm),
    }

    c00 = um * e1 + vm * e2
    c10 = up * e1 + vm * e2
    c11 = up * e1 + vp * e2
    c01 = um * e1 + vp * e2
    corners_phase3 = np.array([c00, c10, c11, c01])

    if gt_info is not None:
        errs = _match_corners_min_dist(corners_phase3, gt_info["corners"])
        print(f"    Phase 3: corners: mean_err={errs.mean():.2f}px  "
              f"max={errs.max():.2f}px  "
              + " ".join(f"{e:.1f}" for e in errs))
    else:
        print(f"    Phase 3: corners computed (no GT)")

    fig3, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig3.suptitle(f"Cluster {ci} — Phase 3")

    ax = axes[0]
    ax.imshow(roi, cmap="gray")
    colors = {"u+": "lime", "u-": "lime", "v+": "cyan", "v-": "cyan"}
    for label, (n, r) in outer_lines.items():
        pts = _draw_infinite_line(n, r, H_roi, W_roi)
        ax.plot([pts[0, 0], pts[1, 0]], [pts[0, 1], pts[1, 1]],
                color=colors[label], linewidth=2, alpha=0.8)

    idx = [0, 1, 2, 3, 0]
    ax.plot(corners_phase3[idx, 0], corners_phase3[idx, 1],
            "g-", linewidth=2, label="Phase 3")
    if gt_info is not None:
        ax.plot(gt_info["corners"][idx, 0], gt_info["corners"][idx, 1],
                "r--", linewidth=2, label="GT")
        for k in range(4):
            ax.plot(gt_info["corners"][k, 0], gt_info["corners"][k, 1],
                    "ro", markersize=4)
    for k in range(4):
        ax.plot(corners_phase3[k, 0], corners_phase3[k, 1], "go", markersize=4)
    ax.legend(fontsize=7)
    ax.set_title("Quadrilateral vs GT")
    ax.axis("off")

    ax = axes[1]
    ax.imshow(roi, cmap="gray")
    for label, (n, r) in outer_lines.items():
        ys, xs = np.nonzero(nms)
        points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        dists = np.abs(points @ n - r)
        mask = dists < 3.0
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1],
                       s=2, alpha=0.5, label=f"support {label}")
    ax.legend(fontsize=5, loc="upper right")
    ax.set_title("Support pixels")
    ax.axis("off")
    plt.tight_layout()

    # ===== Phase 4: Template fitting =====
    fit_tmpl = fit_finder_template(
        roi, nms, angle_npy, center_xy, e1, e2, m_fit,
        w_edge=0.25, w_polarity=0.35, w_contrast=0.25, w_quiet=0.15,
    )

    if gt_info is not None and fit_tmpl.corners is not None:
        errs_p4 = _match_corners_min_dist(fit_tmpl.corners, gt_info["corners"])
        errs_p3 = _match_corners_min_dist(corners_phase3, gt_info["corners"])
        print(f"    Phase 4: score={fit_tmpl.score:.2f}  m={fit_tmpl.m:.2f}  "
              f"corners: mean_err={errs_p4.mean():.2f}px  "
              f"max={errs_p4.max():.2f}px  "
              f"(P3 mean={errs_p3.mean():.2f}px)")
    else:
        print(f"    Phase 4: score={fit_tmpl.score:.2f}  m={fit_tmpl.m:.2f}")

    fig4, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig4.suptitle(f"Cluster {ci} — Phase 4")

    from qr_reader.detector.finder_fit import _sample_1d_cross_section

    for idx, (axis, label) in enumerate([(e1, "e1 (u)"), (e2, "e2 (v)")]):
        ax = axes[0, idx]
        samples = _sample_1d_cross_section(roi, fit_tmpl.center, axis,
                                           4.5 * fit_tmpl.m, 90)
        ts = np.linspace(-4.5 * fit_tmpl.m, 4.5 * fit_tmpl.m, len(samples))
        ax.plot(ts, samples, color="black", linewidth=1.5)
        ideal = np.array([255, 0, 255, 0, 0, 0, 255, 0, 255])
        n_seg = len(ideal)
        seg_ts = np.linspace(-4.5 * fit_tmpl.m, 4.5 * fit_tmpl.m, n_seg + 1)
        for i in range(n_seg):
            ax.axvspan(seg_ts[i], seg_ts[i + 1],
                       alpha=0.15, color="gray" if ideal[i] < 128 else "white")
        ax.set_xlabel(f"Position along {label} (px)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Cross-section along {label}")

    ax = axes[1, 0]
    ax.imshow(roi, cmap="gray")
    for sign, axis in [(1, e1), (-1, e1), (1, e2), (-1, e2)]:
        for k in [1, 2, 3]:
            pos = sign * k * fit_tmpl.m
            pt = fit_tmpl.center + pos * axis
            expected_sign = sign * ((-1) ** (k + 1))
            qi = int(round(pt[1]))
            qj = int(round(pt[0]))
            color = "green"
            if 0 <= qi < H_roi and 0 <= qj < W_roi and nms[qi, qj] > 0:
                ax_grad = angle_npy[qi, qj]
                ax_norm = float(np.arctan2(axis[1], axis[0])) % np.pi
                edge_grad = np.fmod(ax_grad, np.pi)
                edge_grad = edge_grad if edge_grad >= 0 else edge_grad + np.pi
                diff = abs(edge_grad - ax_norm)
                diff = min(diff, np.pi - diff)
                actual_sign = 1 if diff < np.pi / 2 else -1
                color = "green" if actual_sign == expected_sign else "red"
            ax.plot(pt[0], pt[1], "o", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title("Polarity: green=correct, red=wrong")
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(roi, cmap="gray")
    ax.plot(corners_phase3[[0, 1, 2, 3, 0], 0],
            corners_phase3[[0, 1, 2, 3, 0], 1],
            "b-", linewidth=1.5, label="Phase 3")
    ax.plot(fit_tmpl.corners[[0, 1, 2, 3, 0], 0],
            fit_tmpl.corners[[0, 1, 2, 3, 0], 1],
            "g-", linewidth=1.5, label="Phase 4")
    if gt_info is not None:
        ax.plot(gt_info["corners"][[0, 1, 2, 3, 0], 0],
                gt_info["corners"][[0, 1, 2, 3, 0], 1],
                "r--", linewidth=1.5, label="GT")
    ax.legend(fontsize=7)
    ax.set_title("P3 vs P4 vs GT quadrilaterals")
    ax.axis("off")
    plt.tight_layout()

    plt.show()
