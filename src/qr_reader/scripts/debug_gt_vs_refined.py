"""Debug — measure GT error before & after joint refinement.
"""
from pathlib import Path
import cv2
import numpy as np
from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.edge_fitting import (
    _reorder_to_standard, refine_finder_edges_joint,
    PITCH_CONSTANT, MAX_GAP, DISTANCE_THRESHOLD,
    build_pair_distance_matrix, cluster_pairs, compute_boundary_points,
    extract_top_clusters, assign_points,
)
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image
from qr_reader.synth.config import AugmentationConfig
from qr_reader.synth.pipeline import generate_sample
from qr_reader.synth.presets import PRESET_MAP
from scipy.special import erfc
from scipy.optimize import minimize_scalar

PRESET = "medium"
VERSION = 8
SAMPLE_SEED = 44
NUM_RAYS = 36
NUM_SAMPLES = 120
RAY_LENGTH = 1.0

# ── Helpers ─────────────────────────────────────────────────────────────────

def sample_ray_profiles(roi, cx, cy, nr=36, ns=120, rl=1.0):
    H, W = roi.shape
    dd = 0.5 * np.hypot(W, H)
    md = rl * dd
    th = np.linspace(0, 2 * np.pi, nr, endpoint=False)
    dx = np.cos(th)
    dy = np.sin(th)
    xs = cx + np.linspace(0, md, ns)[None, :] * dx[:, None]
    ys = cy + np.linspace(0, md, ns)[None, :] * dy[:, None]
    p = np.zeros((nr, ns), dtype=np.float64)
    for i in range(nr):
        ix = np.clip(xs[i].astype(int), 0, W - 1)
        iy = np.clip(ys[i].astype(int), 0, H - 1)
        p[i] = roi[iy, ix]
    return p, md, th


def normalize_roi_intensities(roi, cxy, mest):
    H, W = roi.shape
    ys, xs = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xs.astype(np.float64) - cxy[0]) ** 2
                   + (ys.astype(np.float64) - cxy[1]) ** 2)
    sig = 3.5 * mest
    w = np.exp(-0.5 * (dist / sig) ** 2)
    vals = roi.ravel().astype(np.float64)
    wf = w.ravel()
    o = np.argsort(vals)
    vs, ws = vals[o], wf[o]
    cw = np.cumsum(ws)
    tw = cw[-1]
    def wp(pct):
        t = pct / 100 * tw
        i = int(np.searchsorted(cw, t))
        return float(vs[max(0, min(i, len(vs) - 1))])
    dk = wp(10)
    br = wp(90)
    sp = br - dk
    if sp < 1:
        sp = 1
    return np.clip((roi.astype(np.float64) - dk) / sp, 0, 1), dk, br


def finder_soft_template(t, m, sigma=1.0):
    u = np.abs(np.asarray(t, np.float64)) / m
    s = sigma / m
    sq = np.sqrt(2.0)
    r = 0.5 * erfc(-(u - 1.5) / (s * sq))
    r -= 0.5 * erfc(-(u - 2.5) / (s * sq))
    r += 0.5 * erfc(-(u - 3.5) / (s * sq))
    return r


def _masked_mse(tv, pv, m_val, mb, sig):
    inside = np.abs(tv) <= mb * m_val
    if np.sum(inside) < 3:
        return np.inf
    return float(np.mean((finder_soft_template(tv[inside], m_val, sig)
                          - pv[inside]) ** 2))


def fit_m_half_ray(ts, pr, mest, mb=4.5, ng=50, gw=2.0, sig=1.0):
    msk = np.isfinite(pr)
    if np.sum(msk) < 10:
        return mest
    tv, pv = ts[msk], pr[msk]
    ml, mh = mest / gw, mest * gw
    mg = np.linspace(ml, mh, ng)
    losses = [_masked_mse(tv, pv, m, mb, sig) for m in mg]
    bi = np.argmin(losses)
    mb2 = mg[bi]
    if not np.isfinite(losses[bi]):
        return mb2
    inside = np.abs(tv) <= mb * mb2
    if np.sum(inside) < 3:
        return mb2
    tr, pr2 = tv[inside], pv[inside]
    step = mg[1] - mg[0] if ng > 1 else mest * 0.05
    def cost(mv):
        return float(np.mean((finder_soft_template(tr, mv, sig) - pr2) ** 2))
    res = minimize_scalar(cost, bounds=(mb2 - step, mb2 + step),
                          method="bounded")
    return float(res.x)


def fit_all_rays(profiles, mest, max_dist):
    tp = np.linspace(0, max_dist, profiles.shape[1])
    out = np.array([fit_m_half_ray(tp, profiles[i], mest)
                    for i in range(profiles.shape[0])])
    return out


def build_edge_data(img_gray, clusters):
    ed = {}
    for ci, cl in enumerate(clusters):
        bb = cluster_to_bbox(cl, scale=1.5)
        roi = cutout(img_gray, bb)
        if roi.size == 0:
            continue
        r0 = max(0, int(bb[0]))
        c0 = max(0, int(bb[2]))
        H, W = roi.shape
        cx = float(cl.cols[2] + cl.cols[3]) / 2 - c0
        cy = float(cl.row) - r0
        cxy = np.array([cx, cy], np.float64)
        me = float(cl.cols[5] - cl.cols[0]) / 7.0
        rn, dk, br = normalize_roi_intensities(roi, cxy, me)
        prof, md, thr = sample_ray_profiles(roi, cx, cy, NUM_RAYS,
                                            NUM_SAMPLES, RAY_LENGTH)
        sp = br - dk
        if sp < 1:
            sp = 1
        pn = np.clip((prof - dk) / sp, 0, 1)
        m_arr = fit_all_rays(pn, me, md)
        bp = compute_boundary_points(cxy, m_arr, thr, PITCH_CONSTANT)
        vld = np.all(np.isfinite(bp), axis=1)
        pts = bp[vld]
        vidx = np.flatnonzero(vld)
        if len(pts) < 4:
            continue
        D, prs = build_pair_distance_matrix(pts, vidx, NUM_RAYS,
                                            max_gap=MAX_GAP)
        lbs = cluster_pairs(D, distance_threshold=DISTANCE_THRESHOLD)
        t4 = extract_top_clusters(lbs, prs, pts, k=4)
        asg = assign_points(t4, len(pts))
        ed[ci] = {"roi": roi, "cxy": cxy, "t4": t4, "asg": asg,
                  "pts": pts, "pn": pn, "thr": thr, "md": md, "me": me,
                  "r0": r0, "c0": c0}
    return ed


def _line_from_corners(p1, p2):
    """(normal, rho) for line p1→p2. Normal points toward center of segment."""
    dx = p2[1] - p1[1]
    dy = p2[0] - p1[0]
    n = np.array([dx, dy], dtype=np.float64)
    n /= np.linalg.norm(n)
    rho = float(n @ p1)
    cent = (p1 + p2) / 2
    if n @ cent < rho - 1e-9:
        n = -n
        rho = -rho
    return n, rho


def side_error(est_normal, est_rho, gt_corners):
    """Mean |distance| from two GT corners to estimated line."""
    d1 = abs(float(est_normal @ np.asarray(gt_corners[0]).ravel() - est_rho))
    d2 = abs(float(est_normal @ np.asarray(gt_corners[1]).ravel() - est_rho))
    return (d1 + d2) / 2


def main():
    rng = np.random.default_rng(SAMPLE_SEED)
    config = AugmentationConfig(
        version=VERSION, content=f"QR v{VERSION}", error_correction="M",
        global_seed=SAMPLE_SEED,
        ppm_range=PRESET_MAP["medium"].ppm_range,
        target_ppm_range=PRESET_MAP["medium"].target_ppm_range,
        jitter_fraction=PRESET_MAP["medium"].jitter_fraction,
        feather_sigma_range=PRESET_MAP["medium"].feather_sigma_range,
        blur_sigma_range=PRESET_MAP["medium"].blur_sigma_range,
        noise_sigma_range=PRESET_MAP["medium"].noise_sigma_range,
        jpeg_quality_range=PRESET_MAP["medium"].jpeg_quality_range,
    )
    bgd = Path("data/images/train")
    bgp = sorted(bgd.glob("*.jpg"))
    bg = cv2.cvtColor(cv2.imread(str(bgp[SAMPLE_SEED % len(bgp)])),
                      cv2.COLOR_BGR2RGB)
    image, metadata = generate_sample(rng, config, bg)
    img_gray = np.asarray(image[:, :, 0], dtype=np.uint8)

    gt_corners_img = np.array([
        metadata["corners_qr"]["TL"], metadata["corners_qr"]["TR"],
        metadata["corners_qr"]["BR"], metadata["corners_qr"]["BL"],
    ], dtype=np.float64)

    # GT sides: (TL,TR)=T, (TR,BR)=R, (BR,BL)=B, (BL,TL)=L
    gt_side_labels = ["T", "R", "B", "L"]
    # We'll build GT sides lazily per cluster after converting to ROI coords

    img_binary = binarize_image(img_gray)
    rows, cols = find_alignment_patterns_2d(img_binary, np.log(1.3))
    clusters = cluster_candidates(rows, cols)
    edge_data = build_edge_data(img_gray, clusters)

    for ci, data in edge_data.items():
        t4 = data["t4"]
        try:
            li, ri, ti, bi = _reorder_to_standard(t4)
        except ValueError:
            continue
        ordered = [t4[li], t4[ri], t4[ti], t4[bi]]
        cxy = data["cxy"]
        pn = data["pn"]
        thr = data["thr"]
        md = data["md"]
        me = data["me"]
        r0 = data["r0"]
        c0 = data["c0"]
        hd = np.column_stack([np.cos(thr), np.sin(thr)])
        ss = np.linspace(0, md, pn.shape[1])

        # Convert GT to ROI-local coords
        gt_local = gt_corners_img - np.array([c0, r0], dtype=np.float64)
        gt_sides = {}
        for a, b, label in [(0, 1, "T"), (1, 2, "R"), (2, 3, "B"), (3, 0, "L")]:
            n, rh = _line_from_corners(gt_local[a], gt_local[b])
            gt_sides[label] = (n, rh, (gt_local[a], gt_local[b]))

        print(f"\n=== Cluster {ci} (m_est={me:.2f}px) ===")

        refined, result = refine_finder_edges_joint(t4, cxy, pn, hd, ss)

        print(f"  LM: converged={result.success}, cost={result.cost:.4f}, "
              f"nfev={result.nfev}")
        print(f"  {'Side':>4} {'init_err':>9} {'ref_err':>9} {'Δ_err':>8} "
              f"{'better?':>8}  {'init_ρ':>8} {'ref_ρ':>8} {'GT_ρ':>8}")
        total_init = 0.0
        total_ref = 0.0
        for k, name in enumerate(["L", "R", "T", "B"]):
            gt_n, gt_rh, gt_pair = gt_sides[name]
            ec_init = ordered[k]
            ec_ref = refined[k]
            e_init = side_error(ec_init.normal, ec_init.rho, gt_pair)
            e_ref = side_error(ec_ref.normal, ec_ref.rho, gt_pair)
            total_init += e_init
            total_ref += e_ref
            de = e_ref - e_init
            if de < -0.01:
                b = "BETTER"
            elif de > 0.01:
                b = "WORSE"
            else:
                b = "~same"
            print(f"    {name}: {e_init:9.3f} {e_ref:9.3f} {de:+8.3f} "
                  f"{b:>8}  {ec_init.rho:8.2f} {ec_ref.rho:8.2f} {gt_rh:8.2f}")
        print(f"    TOTAL: init={total_init:.3f} ref={total_ref:.3f} "
              f"({'BETTER' if total_ref < total_init - 0.02 else 'WORSE'})")

        # Also compute per-ray residual breakdown
        print(f"\n  Per-ray residual analysis (initial lines):")
        from qr_reader.detector.edge_fitting import (
            thetarho_to_homogeneous_line, compute_corners,
            compute_projective_center, compute_kappa,
            _assign_rays_to_sides, compute_transition_distances,
            _fit_ols_params, precompute_mask,
            joint_refinement_residuals,
        )
        theta_i = np.array([float(np.arctan2(s.normal[1], s.normal[0]))
                            for s in ordered], dtype=np.float64)
        rho_i = np.array([float(s.rho) for s in ordered], dtype=np.float64)
        ell_L_i = thetarho_to_homogeneous_line(float(theta_i[0]),
                                                float(rho_i[0]))
        ell_R_i = thetarho_to_homogeneous_line(float(theta_i[1]),
                                                float(rho_i[1]))
        ell_T_i = thetarho_to_homogeneous_line(float(theta_i[2]),
                                                float(rho_i[2]))
        ell_B_i = thetarho_to_homogeneous_line(float(theta_i[3]),
                                                float(rho_i[3]))
        corners_i = compute_corners(ell_L_i, ell_R_i, ell_T_i, ell_B_i)
        c = compute_projective_center(*corners_i)
        ku, kv = compute_kappa(ell_L_i, ell_R_i, ell_T_i, ell_B_i, c)
        R_val = float(np.mean([np.linalg.norm(cr - c) for cr in corners_i]))
        per_ray = _assign_rays_to_sides(cxy, hd, ell_L_i, ell_R_i,
                                         ell_T_i, ell_B_i)
        n_rays = len(hd)
        pre_masks = np.zeros((n_rays, pn.shape[1]), dtype=bool)
        for k in range(n_rays):
            si = int(per_ray[k]) if per_ray[k] >= 0 else None
            s_j = compute_transition_distances(cxy, hd[k], ell_L_i, ell_R_i,
                                               ell_T_i, ell_B_i, ku, kv,
                                               side_idx=si)
            pre_masks[k] = precompute_mask(ss, s_j, 1.0)

        x0 = np.zeros(8, dtype=np.float64)
        x0[4:8] = rho_i
        ab = _fit_ols_params(cxy, pn, hd, ss, pre_masks, ell_L_i, ell_R_i,
                             ell_T_i, ell_B_i, ku, kv, 1.0,
                             per_ray_side=per_ray)
        r0 = joint_refinement_residuals(x0, cxy, R_val, theta_i, pn, hd, ss,
                                        pre_masks, 1.0, ab_fixed=ab,
                                        per_ray_side=per_ray)

        # Per-ray residual cost
        n_s = pn.shape[1]
        ray_costs = np.zeros(n_rays)
        for k in range(n_rays):
            chunk = r0[k * n_s:(k + 1) * n_s]
            active = chunk[pre_masks[k]]
            if len(active) > 0:
                ray_costs[k] = 0.5 * np.sum(active ** 2)

        # Group by side
        side_ray_costs = {}
        for si, sname in enumerate(["L", "R", "T", "B"]):
            mask = per_ray == si
            if np.any(mask):
                rcost = float(np.sum(ray_costs[mask]))
                n_active = int(np.sum(pre_masks[mask]))
                pct = rcost / float(np.sum(ray_costs)) * 100
                print(f"    {sname}: {int(np.sum(mask))} rays, "
                      f"Σcost={rcost:.3f} ({pct:.1f}%), "
                      f"masked={n_active} samples")
                for k_ray in np.flatnonzero(mask):
                    side_ray_costs[k_ray] = ray_costs[k_ray]
        # Print top 3 highest-cost rays
        sorted_rays = sorted(side_ray_costs.items(), key=lambda x: -x[1])
        print(f"    Top 3 highest-cost rays:")
        for k_ray, rc in sorted_rays[:3]:
            si = int(per_ray[k_ray])
            sname = ["L","R","T","B"][si]
            angle = np.rad2deg(thr[k_ray])
            print(f"      Ray {k_ray} ({angle:.0f}°, side={sname}): cost={rc:.3f}")


if __name__ == "__main__":
    main()
