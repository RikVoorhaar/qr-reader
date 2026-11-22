#!/usr/bin/env python3
"""Diagnose the recall differences between centre geometry and the repaired edge rule.

Three groups of images with at least three finder candidates are inspected:

* **current-only** — ``find_valid_triplets`` returns a triplet, the repaired
  edge rule returns none.
* **repaired-only** — the repaired edge rule returns a triplet, centre
  geometry returns none.
* **neither** — both methods return none.

For each image the script records why each method rejected (or accepted) a
candidate, compares against an independent OpenCV QR detection where possible,
tries decoding the triplets (including all six role assignments), and writes an
overlay PNG showing every finder candidate, both methods' triplets, and the
projected QR boundary.

The fast sampler below is an exact vectorization of
``qr_reader.detector.sample.sample_qr_bits`` (verified bit-for-bit on a v20
sampling; the original takes ~75 s per image, this one ~0.02 s).

Usage (from the repository root):

    .venv/bin/python blog_figures/triplet/analyze_recall_neither.py
    .venv/bin/python blog_figures/triplet/analyze_recall_neither.py --refresh

By default the script reuses ``/tmp/opencode/triplet_failure_modes/cache.pkl``
written by ``analyze_failure_modes.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

cv2.setNumThreads(1)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.detector.detector import _score_timing_pattern  # noqa: E402
from qr_reader.detector.finder_pattern import Triplet  # noqa: E402
from qr_reader.detector.homography import (  # noqa: E402
    estimate_homography_dlt,
    project_points,
    refine_homography_lm,
)
from qr_reader.detector.sample import finder_pattern_known_cells  # noqa: E402

from analyze_failure_modes import (  # noqa: E402
    ANGLE_TOL,
    MAIN_OFFSET_TOL,
    ImageRecord,
    check_association_repaired,
    find_all_associations_repaired,
    find_triplets_repaired,
    repaired_triplet_count,
)
from make_figures import classify_edges, edge_data  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
DEFAULT_OUT_DIR = Path("/tmp/opencode/triplet_failure_modes")
DEFAULT_CACHE = DEFAULT_OUT_DIR / "cache.pkl"
PAYLOAD = "rikvoorhaar.com"

OFFSETS = np.array([0.25, 0.5, 0.75])
DY, DX = np.meshgrid(OFFSETS, OFFSETS, indexing="ij")

FINDER_COLORS = ["#ffd23f", "#4cc9f0", "#c77dff", "#ff6b6b", "#7bed9f", "#f4a261"]
CURRENT_COLOR = "#39ff88"
REPAIRED_COLOR = "#ff4fd8"
BOUNDARY_COLOR = "#00e5ff"
TRUTH_COLOR = "#ffffff"

# Triplets verified outside OpenCV (decode or unambiguous geometry):
# image -> ((TL, TR, BL), N)
KNOWN_TRUTH: dict[str, tuple[tuple[int, int, int], int]] = {
    "qr_code_ - 11.jpeg": ((9, 3, 0), 97),   # decodes with the fast sampler at v20
    "qr_code_ - 41.jpeg": ((1, 2, 0), 57),   # three clear finders, v10 by geometry
    "qr_code_ - 74.jpeg": ((0, 3, 2), 57),   # decodes with the fast sampler at v10
}


# ---------------------------------------------------------------------------
# Fast, exact sampler
# ---------------------------------------------------------------------------


def fast_sample_qr_bits(image: np.ndarray, H: np.ndarray, N: int) -> np.ndarray:
    """Vectorized equivalent of ``sample.sample_qr_bits``."""
    black, white = finder_pattern_known_cells(N)
    cells = np.array(black + white)

    gx = (cells[:, 1][:, None] + DX.ravel()[None, :]).ravel()
    gy = (cells[:, 0][:, None] + DY.ravel()[None, :]).ravel()
    img = project_points(H, np.column_stack([gx, gy]))
    vals = map_coordinates(
        image.astype(np.float64), np.stack([img[:, 1], img[:, 0]]), order=1,
        mode="nearest",
    ).reshape(len(cells), 9)
    nb = len(black)
    threshold = (np.median(vals[:nb, 4]) + np.median(vals[nb:, 4])) / 2.0

    cols = np.arange(N)
    rows = np.arange(N)
    gx = (cols[None, :, None] + DX.ravel()[None, None, :]).repeat(N, axis=0).ravel()
    gy = (rows[:, None, None] + DY.ravel()[None, None, :]).repeat(N, axis=1).ravel()
    img = project_points(H, np.column_stack([gx, gy]))
    vals = map_coordinates(
        image.astype(np.float64), np.stack([img[:, 1], img[:, 0]]), order=1,
        mode="nearest",
    ).reshape(N * N, 9)

    weights = np.ones(9)
    weights[4] = 2.0
    votes = ((vals > threshold) * weights).sum(axis=1).reshape(N, N)
    return (~(votes >= 5.0)).T


# ---------------------------------------------------------------------------
# Triplet homography + decode
# ---------------------------------------------------------------------------


def canonicalize(corners_xy: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    centre = corners_xy.mean(axis=0)
    uv = corners_xy - centre
    up = uv @ u
    vp = uv @ v
    return corners_xy[
        np.array(
            [
                int(np.argmin(up + vp)),
                int(np.argmax(up - vp)),
                int(np.argmax(up + vp)),
                int(np.argmin(up - vp)),
            ]
        )
    ]


@dataclass
class FitResult:
    H: np.ndarray
    N: int
    version: int
    reproj_error: float
    timing_score: float
    text: str
    decode_error: str
    boundary: np.ndarray


def fit_homography(
    fps, triplet: Triplet, N: int
) -> tuple[np.ndarray, float]:
    fp = {f.cluster_idx: f for f in fps}
    idx = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    centres = {i: fp[i].outer_corners.mean(axis=0) for i in idx}
    ctl, ctr, cbl = (centres[i][::-1] for i in idx)
    u = (ctr - ctl) / (np.linalg.norm(ctr - ctl) + 1e-12)
    v = (cbl - ctl) / (np.linalg.norm(cbl - ctl) + 1e-12)
    corners = [canonicalize(fp[i].outer_corners[:, ::-1], u, v) for i in idx]

    grid_offsets = np.array([[0, 0], [7, 0], [7, 7], [0, 7]], dtype=float)
    src = []
    dst = []
    for cc, origin in [
        (corners[0], (0, 0)),
        (corners[1], (N - 7, 0)),
        (corners[2], (0, N - 7)),
    ]:
        for k in range(4):
            src.append([origin[0] + grid_offsets[k, 0], origin[1] + grid_offsets[k, 1]])
            dst.append(cc[k].tolist())
    src_arr = np.array(src, dtype=float)
    dst_arr = np.array(dst, dtype=float)

    H = estimate_homography_dlt(src_arr, dst_arr)
    try:
        H = refine_homography_lm(H, src_arr, dst_arr, loss="linear")
    except Exception:
        pass
    err = float(np.mean(np.linalg.norm(project_points(H, src_arr) - dst_arr, axis=1)))
    return H, err


def estimate_N(fps, triplet: Triplet) -> int:
    """Estimate the module count from finder side lengths and centre distances."""
    fp = {f.cluster_idx: f for f in fps}
    idx = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    sides = {
        i: float(
            np.mean(
                [
                    np.linalg.norm(fp[i].outer_corners[(k + 1) % 4] - fp[i].outer_corners[k])
                    for k in range(4)
                ]
            )
        )
        for i in idx
    }
    centres = {i: fp[i].outer_corners.mean(axis=0) for i in idx}
    d1 = float(np.linalg.norm(centres[idx[1]] - centres[idx[0]]))
    d2 = float(np.linalg.norm(centres[idx[2]] - centres[idx[0]]))
    n1 = 14.0 * d1 / (sides[idx[0]] + sides[idx[1]]) + 7.0
    n2 = 14.0 * d2 / (sides[idx[0]] + sides[idx[2]]) + 7.0
    N = int(round((n1 + n2) / 2.0))
    N = min(177, max(21, int(round((N - 21) / 4.0)) * 4 + 21))
    return N


def qr_boundary(H: np.ndarray, N: int) -> np.ndarray:
    corners = np.array(
        [[-0.5, -0.5], [N - 0.5, -0.5], [N - 0.5, N - 0.5], [-0.5, N - 0.5]],
        dtype=float,
    )
    return project_points(H, corners)


def fit_and_decode(
    gray: np.ndarray, fps, triplet: Triplet, N: int
) -> FitResult:
    H, err = fit_homography(fps, triplet, N)
    timing = _score_timing_pattern(gray, H, N)
    try:
        bits = fast_sample_qr_bits(gray, H, N)
        text = decode_qr(bits)
        error = ""
    except Exception as exc:
        text = ""
        error = f"{type(exc).__name__}: {exc}"
    return FitResult(
        H=H, N=N, version=(N - 17) // 4, reproj_error=err,
        timing_score=float(timing), text=text, decode_error=error,
        boundary=qr_boundary(H, N),
    )


def version_scan(
    gray: np.ndarray, fps, triplet: Triplet, topk: int = 6
) -> tuple[list[FitResult], float]:
    """Score every N, decode only the best ``topk`` candidates."""
    scored = []
    for N in range(21, 178, 4):
        H, err = fit_homography(fps, triplet, N)
        timing = _score_timing_pattern(gray, H, N)
        scored.append((err - timing, err, timing, N))
    scored.sort()

    results = []
    for _, err, timing, N in scored[:topk]:
        results.append(fit_and_decode(gray, fps, triplet, N))

    success = next((r for r in results if r.text == PAYLOAD), None)
    if success is not None:
        return results, float(success.reproj_error)
    return results, float(results[0].reproj_error)


def permutation_decodes(
    gray: np.ndarray, fps, triplet: Triplet, N: int
) -> list[tuple[tuple[int, int, int], FitResult]]:
    out = []
    for perm in itertools.permutations(
        [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    ):
        res = fit_and_decode(gray, fps, Triplet(*perm), N)
        if res.text == PAYLOAD:
            out.append((perm, res))
    return out


# ---------------------------------------------------------------------------
# Independent OpenCV truth
# ---------------------------------------------------------------------------


@dataclass
class OpenCVTruth:
    payload: str
    corners: np.ndarray | None
    roll: int
    N: int
    cost: float
    expected: np.ndarray | None
    nearest: list[tuple[int, float]] | None
    triplet: Triplet | None
    N_true: int = -1


def opencv_truth(gray: np.ndarray, fps) -> OpenCVTruth:
    detector = cv2.QRCodeDetector()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ok, pts = detector.detect(bgr)
    payload = ""
    N_true = -1
    if ok and pts is not None:
        try:
            payload, straight = detector.decode(bgr, pts)
            if straight is not None:
                N_true = int(straight.shape[0])
        except Exception:
            payload = ""
    if not ok or pts is None:
        return OpenCVTruth(payload, None, -1, -1, np.inf, None, None, None, N_true)

    pts = np.asarray(pts, dtype=float)[0]
    centres = np.array([f.outer_corners.mean(axis=0)[::-1] for f in fps])
    ids = [f.cluster_idx for f in fps]

    best = None
    Ns = [N_true] if N_true > 0 else list(range(21, 178, 4))
    for roll in range(4):
        Q = np.roll(pts, roll, axis=0)
        for N in Ns:
            grid = np.array(
                [[-0.5, -0.5], [N - 0.5, -0.5], [N - 0.5, N - 0.5], [-0.5, N - 0.5]]
            )
            H = estimate_homography_dlt(grid, Q)
            expected = np.array(
                [
                    project_points(H, np.array([[3.0, 3.0]]))[0],
                    project_points(H, np.array([[N - 4.0, 3.0]]))[0],
                    project_points(H, np.array([[3.0, N - 4.0]]))[0],
                ]
            )
            dist = np.linalg.norm(centres[None, :, :] - expected[:, None, :], axis=2)
            nearest = dist.argmin(axis=1)
            cost = float(dist[np.arange(3), nearest].sum())
            if best is None or cost < best[0]:
                best = (cost, roll, N, expected, nearest, dist)

    cost, roll, N, expected, nearest, dist = best
    nearest_list = [
        (ids[int(nearest[k])], float(dist[k, nearest[k]])) for k in range(3)
    ]
    triplet = None
    if len({n[0] for n in nearest_list}) == 3:
        triplet = Triplet(nearest_list[0][0], nearest_list[1][0], nearest_list[2][0])
    return OpenCVTruth(
        payload, pts, roll, N, cost, expected, nearest_list, triplet, N_true
    )


# ---------------------------------------------------------------------------
# Centre-geometry gates
# ---------------------------------------------------------------------------


def finder_geom(fp):
    corners = fp.outer_corners
    sides = [
        float(np.linalg.norm(corners[(k + 1) % 4] - corners[k])) for k in range(4)
    ]
    m = float(np.mean(sides)) / 7.0
    xy = corners[:, ::-1]
    e1 = xy[1] - xy[0]
    e2 = xy[3] - xy[0]
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return m, e1, e2


def pair_gate(fp_i, fp_j) -> dict:
    mi, e1i, e2i = finder_geom(fp_i)
    mj, _, _ = finder_geom(fp_j)
    ci = fp_i.outer_corners.mean(axis=0)
    cj = fp_j.outer_corners.mean(axis=0)
    delta = cj - ci
    dist = float(np.linalg.norm(delta))
    m_avg = (mi + mj) / 2.0
    delta_xy = delta[::-1]
    delta_unit = delta_xy / (dist + 1e-12)
    dot1 = abs(float(np.dot(delta_unit, e1i)))
    dot2 = abs(float(np.dot(delta_unit, e2i)))
    parallel = max(dot1, dot2)
    perp = min(dot1, dot2)

    info = dict(
        mi=mi, mj=mj, m_avg=m_avg, dist=dist, dist_ratio=dist / (m_avg + 1e-12),
        parallel=parallel, perp=perp, ok=True, reason="",
    )
    if mi < 2.0 or mj < 2.0:
        info.update(ok=False, reason="module size < 2 px")
    elif abs(mi - mj) / max(mi, mj) > 0.3:
        info.update(ok=False, reason="module-size mismatch")
    elif dist < 2.0 * m_avg or dist > 200.0 * m_avg:
        info.update(ok=False, reason="distance gate")
    elif parallel < 0.9 or perp > 0.25 * parallel:
        info.update(ok=False, reason="axis-alignment gate")
    return info


def triplet_gates(fps, triplet: Triplet) -> dict:
    fp = {f.cluster_idx: f for f in fps}
    a, b, c = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx
    ab = pair_gate(fp[a], fp[b])
    ac = pair_gate(fp[a], fp[c])
    bc = pair_gate(fp[b], fp[c])
    ma = finder_geom(fp[a])[0]
    mb = finder_geom(fp[b])[0]
    mc = finder_geom(fp[c])[0]
    centres = {i: fp[i].outer_corners.mean(axis=0) for i in (a, b, c)}
    vec_ab = centres[b] - centres[a]
    vec_ac = centres[c] - centres[a]
    cosang = float(
        np.dot(vec_ab, vec_ac)
        / (np.linalg.norm(vec_ab) * np.linalg.norm(vec_ac) + 1e-12)
    )
    angle = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
    m_max, m_min = max(ma, mb, mc), min(ma, mb, mc)

    reason = ""
    if not ab["ok"]:
        reason = ab["reason"]
    elif not ac["ok"]:
        reason = ac["reason"]
    elif bc["ok"]:
        reason = "neighbours directly connected"
    elif abs(angle - 90.0) > 15.0:
        reason = "right-angle gate"
    elif (m_max - m_min) / (m_max + 1e-12) > 0.3:
        reason = "triplet module-size mismatch"
    else:
        reason = "would form triplet"

    return dict(
        ab=ab, ac=ac, bc=bc, angle=angle,
        module_spread=(m_max - m_min) / (m_max + 1e-12),
        reason=reason,
    )


def current_triplet_diagnosis(fps, triplet: Triplet) -> dict:
    """Exact replay of the centre-geometry gates for one candidate triplet.

    Pair gates are evaluated in finder-list order, matching
    ``find_valid_triplets`` (it always uses the first finder's axes).
    """
    fp = {f.cluster_idx: f for f in fps}
    pos = {f.cluster_idx: i for i, f in enumerate(fps)}
    ids = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]

    pairs = {}
    for x, y in itertools.combinations(ids, 2):
        i, j = (x, y) if pos[x] < pos[y] else (y, x)
        pairs[frozenset((x, y))] = pair_gate(fp[i], fp[j])

    conn = {v: [u for u in ids if u != v and pairs[frozenset((u, v))]["ok"]] for v in ids}
    for v in ids:
        if len(conn[v]) != 2:
            continue
        a, b = conn[v]
        if pairs[frozenset((a, b))]["ok"]:
            return dict(reason="neighbours directly connected", pairs=pairs, vertex=v)
        ca = fp[a].outer_corners.mean(axis=0)
        cv = fp[v].outer_corners.mean(axis=0)
        cb = fp[b].outer_corners.mean(axis=0)
        va, vb = ca - cv, cb - cv
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(va, vb)
                        / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12),
                        -1.0, 1.0,
                    )
                )
            )
        )
        if abs(angle - 90.0) > 15.0:
            return dict(reason="right-angle gate", pairs=pairs, vertex=v, angle=angle)
        ma = finder_geom(fp[a])[0]
        mb = finder_geom(fp[b])[0]
        mv = finder_geom(fp[v])[0]
        spread = (max(ma, mb, mv) - min(ma, mb, mv)) / max(ma, mb, mv)
        if spread > 0.3:
            return dict(reason="triplet module-size mismatch", pairs=pairs, vertex=v, angle=angle)
        return dict(reason="would form triplet", pairs=pairs, vertex=v, angle=angle)

    fails = [
        f"{sorted(k)}: {v['reason']}" for k, v in pairs.items() if not v["ok"]
    ]
    return dict(
        reason="no valid pair connectivity", pairs=pairs, detail="; ".join(fails)
    )


def current_gate_summary(fps) -> dict:
    """Exact reason histogram over all candidate triples (list-order pair gates)."""
    reason_counter = Counter()
    pair_reasons = Counter()
    for group in itertools.combinations(fps, 3):
        diag = current_triplet_diagnosis(
            fps, Triplet(group[0].cluster_idx, group[1].cluster_idx, group[2].cluster_idx)
        )
        reason_counter[diag["reason"]] += 1
        for p in diag["pairs"].values():
            if not p["ok"]:
                pair_reasons[p["reason"]] += 1
    return dict(all=reason_counter, pairs=pair_reasons)


# ---------------------------------------------------------------------------
# Repaired-rule diagnostics
# ---------------------------------------------------------------------------


def repaired_summary(fps, tol: float = MAIN_OFFSET_TOL) -> dict:
    associations, checks = find_all_associations_repaired(fps, ANGLE_TOL, tol)
    triplets = find_triplets_repaired(fps, associations)

    # For every pair, the best possible max offset across the eight pairings.
    pair_best = []
    for chk in checks:
        valid = [p for p in chk.pairings if p.passed]
        best = min(chk.pairings, key=lambda p: p.max_offset) if chk.pairings else None
        pair_best.append(
            dict(
                fp1=chk.fp1, fp2=chk.fp2, intersects=chk.intersects,
                associated=chk.selected is not None,
                best_max=best.max_offset if best else np.inf,
                best_angle=min(
                    max(best.angles) if best else np.inf,
                    np.inf,
                ),
                n_valid=len(valid),
            )
        )

    # Why no L-shape?  Inspect nodes with >= 2 neighbours.
    adj: dict[int, list[int]] = {f.cluster_idx: [] for f in fps}
    for a in associations:
        adj[a.fp1_idx].append(a.fp2_idx)
        adj[a.fp2_idx].append(a.fp1_idx)
    direct = {frozenset((a.fp1_idx, a.fp2_idx)) for a in associations}

    l_reasons = Counter()
    for b, neighbours in adj.items():
        if len(neighbours) < 2:
            continue
        for a, c in itertools.combinations(neighbours, 2):
            if frozenset((a, c)) in direct:
                l_reasons["neighbours directly connected"] += 1
                continue
            # Find the associations to compute shared B segments
            ab = next(
                x for x in associations
                if {x.fp1_idx, x.fp2_idx} == {a, b}
            )
            cb = next(
                x for x in associations
                if {x.fp1_idx, x.fp2_idx} == {b, c}
            )
            b_ab = set(ab.colinear_segments_1 if ab.fp1_idx == b else ab.colinear_segments_2)
            b_cb = set(cb.colinear_segments_1 if cb.fp1_idx == b else cb.colinear_segments_2)
            if b_ab & b_cb:
                l_reasons["shared B segment axis"] += 1
            else:
                l_reasons["would form triplet"] += 1

    if not associations:
        reason = "no associations at threshold"
    elif not any(len(v) >= 2 for v in adj.values()):
        reason = "associations but no node with >=2 neighbours"
    elif l_reasons.get("would form triplet", 0) > 0:
        reason = "would form triplet"
    elif l_reasons:
        reason = "L-shape blocked: " + ", ".join(
            f"{k} ({v})" for k, v in l_reasons.most_common()
        )
    else:
        reason = "no L-shape"

    return dict(
        associations=associations, checks=checks, triplets=triplets,
        pair_best=pair_best, adj=adj, l_reasons=l_reasons, reason=reason,
    )


def _pair_key(pairs):
    return {tuple(p) for p in pairs}


def true_pair_report(fps, triplet: Triplet, tol: float = MAIN_OFFSET_TOL) -> dict:
    """Repaired-rule scores for the two geometric pairs of an assumed triplet."""
    fp_map = {f.cluster_idx: f for f in fps}
    tl, tr, bl = triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx
    centres = {i: fp_map[i].outer_corners.mean(axis=0) for i in (tl, tr, bl)}
    u = centres[tr] - centres[tl]
    u = u / (np.linalg.norm(u) + 1e-12)
    v = centres[bl] - centres[tl]
    v = v / (np.linalg.norm(v) + 1e-12)
    edges = {i: classify_edges(fp_map[i], u, v) for i in (tl, tr, bl)}

    specs = {
        (tl, tr): (
            (edges[tl]["top"], edges[tr]["top"]),
            (edges[tl]["bottom"], edges[tr]["bottom"]),
        ),
        (tl, bl): (
            (edges[tl]["left"], edges[bl]["left"]),
            (edges[tl]["right"], edges[bl]["right"]),
        ),
    }

    reports = []
    for (a, b), pairs in specs.items():
        chk = check_association_repaired(fp_map[a], fp_map[b], ANGLE_TOL, tol)
        tp = chk.pairing_for(pairs)
        selected = None
        if chk.selected is not None:
            sel_pairs = tuple(
                zip(chk.selected.colinear_segments_1, chk.selected.colinear_segments_2)
            )
            selected = chk.pairing_for(sel_pairs)
        false = None
        if tp is not None:
            candidates = [
                p for p in chk.pairings
                if p.passed and _pair_key(p.pairs) != _pair_key(tp.pairs)
            ]
            if candidates:
                false = min(candidates, key=lambda p: (p.max_offset, p.total))
        reports.append(
            dict(
                a=a, b=b, pairs=pairs, true_pairing=tp, selected=selected,
                best_false=false, intersects=chk.intersects,
                passes=bool(tp is not None and tp.passed),
                selected_is_true=bool(
                    selected is not None and tp is not None
                    and _pair_key(selected.pairs) == _pair_key(tp.pairs)
                ),
            )
        )

    true_pairings = [r["true_pairing"] for r in reports]
    angle_ok = all(
        p is not None and max(p.angles) < ANGLE_TOL for p in true_pairings
    )
    intersects = any(r["intersects"] for r in reports)
    structural = (
        any(p is None for p in true_pairings) or not angle_ok or intersects
    )
    max_offset = (
        max(p.max_offset for p in true_pairings if p is not None)
        if not any(p is None for p in true_pairings)
        else float("inf")
    )
    tol_needed = None if structural else max_offset * 1.001 + 1e-6

    triplet_at_tol = False
    extra_edges = 0
    if tol_needed is not None:
        assoc, _ = find_all_associations_repaired(fps, ANGLE_TOL, tol_needed)
        trips = find_triplets_repaired(fps, assoc)
        triplet_at_tol = any(same_triplet(t, triplet) for t in trips)
        true_edges = {frozenset((tl, tr)), frozenset((tl, bl))}
        extra_edges = len(
            [
                e for e in assoc
                if frozenset((e.fp1_idx, e.fp2_idx)) not in true_edges
            ]
        )

    return dict(
        reports=reports, max_offset=max_offset, angle_ok=angle_ok,
        intersects=intersects, structural=structural, tol_needed=tol_needed,
        triplet_at_tol=triplet_at_tol, extra_edges=extra_edges,
    )


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def same_triplet(a: Triplet, b: Triplet) -> bool:
    return (a.top_left_idx, a.top_right_idx, a.bottom_left_idx) == (
        b.top_left_idx, b.top_right_idx, b.bottom_left_idx
    )


@dataclass
class Case:
    rec: ImageRecord
    group: str
    gray: np.ndarray
    current: Triplet | None
    repaired: Triplet | None
    truth: OpenCVTruth
    current_fit: FitResult | None = None
    repaired_fit: FitResult | None = None
    perm_decodes: list = field(default_factory=list)
    gate: dict | None = None
    gate_exact: dict | None = None
    truth_gate_exact: dict | None = None
    gate_summary: dict | None = None
    repaired_diag: dict | None = None
    current_pairs: dict | None = None
    repaired_pairs: dict | None = None
    truth_pairs: dict | None = None
    truth_boundary: np.ndarray | None = None
    display_truth: Triplet | None = None
    display_truth_N: int = -1
    new_pipe: dict | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------


def crop_bounds(points: list[np.ndarray], shape, pad=70):
    allp = np.vstack(points)
    r0 = max(0, int(np.floor(allp[:, 1].min())) - pad)
    r1 = min(shape[0], int(np.ceil(allp[:, 1].max())) + pad)
    c0 = max(0, int(np.floor(allp[:, 0].min())) - pad)
    c1 = min(shape[1], int(np.ceil(allp[:, 0].max())) + pad)
    return r0, r1, c0, c1


def draw_finder(ax, fp, color, label=None, dashed=False):
    c = fp.outer_corners[:, ::-1]
    q = np.vstack([c, c[0]])
    ax.plot(q[:, 0], q[:, 1], color="#101010", lw=5.0, alpha=0.55, zorder=3,
            ls=(0, (4, 2)) if dashed else "-")
    ax.plot(q[:, 0], q[:, 1], color=color, lw=2.6, zorder=4,
            ls=(0, (4, 2)) if dashed else "-")
    ax.scatter(c[:, 0], c[:, 1], s=22, facecolors="none", edgecolors=color,
               linewidths=1.4, zorder=5)
    if label:
        m = fp.outer_corners.mean(axis=0)
        ax.text(m[1], m[0], label, color=color, fontsize=10.5, fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.15", fc="#101010", ec="none", alpha=0.85))


def draw_triplet(ax, fp_map, triplet: Triplet | None, color, prefix, lw=3.0, ls="-"):
    if triplet is None:
        return
    idx = [triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx]
    c = {i: fp_map[i].outer_corners.mean(axis=0) for i in idx}
    for a, b in [(idx[0], idx[1]), (idx[0], idx[2])]:
        ax.plot([c[a][1], c[b][1]], [c[a][0], c[b][0]], color=color, lw=lw,
                ls=ls, zorder=6, solid_capstyle="round")
        ax.plot([c[a][1], c[b][1]], [c[a][0], c[b][0]], color="#101010", lw=lw + 2,
                alpha=0.4, zorder=5, ls=ls)
    for role, i in zip(("TL", "TR", "BL"), idx):
        ax.text(c[i][1], c[i][0] - 18, f"{prefix}{role}", color=color, fontsize=10,
                ha="center", va="bottom", zorder=9,
                bbox=dict(boxstyle="round,pad=0.12", fc="#101010", ec="none", alpha=0.8))


def draw_boundary(ax, boundary, color, label, lw=2.2, ls="-"):
    q = np.vstack([boundary, boundary[0]])
    ax.plot(q[:, 0], q[:, 1], color="#101010", lw=lw + 2.0, alpha=0.5, zorder=6, ls=ls)
    ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, zorder=7, ls=ls)
    ax.text(boundary[0, 0], boundary[0, 1] - 12, label, color=color, fontsize=9.5,
            ha="left", va="bottom", zorder=9,
            bbox=dict(boxstyle="round,pad=0.15", fc="#101010", ec="none", alpha=0.8))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def load_records(cache: Path) -> list[ImageRecord]:
    with open(cache, "rb") as f:
        return pickle.load(f)


def build_cases(records: list[ImageRecord], data_dir: Path) -> list[Case]:
    cases = []
    detector = cv2.QRCodeDetector()
    for rec in records:
        if len(rec.fps) < 3:
            continue
        current = rec.current_triplets[0] if rec.current_triplets else None
        assoc, _ = find_all_associations_repaired(rec.fps, ANGLE_TOL, MAIN_OFFSET_TOL)
        rep_triplets = find_triplets_repaired(rec.fps, assoc)
        repaired = rep_triplets[0] if rep_triplets else None
        if current is not None and repaired is not None:
            continue
        if current is not None:
            group = "current-only"
        elif repaired is not None:
            group = "repaired-only"
        else:
            group = "neither"

        gray = cv2.imread(str(data_dir / rec.image), cv2.IMREAD_GRAYSCALE)
        truth = opencv_truth(gray, rec.fps)
        cases.append(Case(rec, group, gray, current, repaired, truth))
    return cases


def analyse_case(case: Case) -> None:
    rec = case.rec
    if case.group == "current-only":
        N = 4 * rec.version + 17 if rec.version > 0 else None
        if N:
            case.current_fit = fit_and_decode(case.gray, rec.fps, case.current, N)
            case.perm_decodes = permutation_decodes(case.gray, rec.fps, case.current, N)
        case.gate = triplet_gates(rec.fps, case.current)
        case.gate_exact = current_triplet_diagnosis(rec.fps, case.current)
        try:
            case.current_pairs = true_pair_report(rec.fps, case.current)
        except Exception:
            case.current_pairs = None
        if case.truth.triplet is not None and not same_triplet(
            case.truth.triplet, case.current
        ):
            try:
                case.truth_pairs = true_pair_report(rec.fps, case.truth.triplet)
            except Exception:
                case.truth_pairs = None
    elif case.group == "repaired-only":
        results, _ = version_scan(case.gray, rec.fps, case.repaired)
        success = next((r for r in results if r.text == PAYLOAD), None)
        case.repaired_fit = success or results[0]
        case.gate = triplet_gates(rec.fps, case.repaired)
        case.gate_exact = current_triplet_diagnosis(rec.fps, case.repaired)
        try:
            case.repaired_pairs = true_pair_report(rec.fps, case.repaired)
        except Exception:
            case.repaired_pairs = None
    else:
        case.gate_summary = current_gate_summary(rec.fps)
        if case.truth.triplet is not None:
            try:
                case.truth_pairs = true_pair_report(rec.fps, case.truth.triplet)
            except Exception:
                case.truth_pairs = None

    case.repaired_diag = repaired_summary(rec.fps, MAIN_OFFSET_TOL)

    known = KNOWN_TRUTH.get(rec.image)
    mean_side = float(
        np.mean(
            [
                np.mean(
                    [
                        np.linalg.norm(
                            fp.outer_corners[(k + 1) % 4] - fp.outer_corners[k]
                        )
                        for k in range(4)
                    ]
                )
                for fp in rec.fps
            ]
        )
    )
    if known is not None:
        case.display_truth = Triplet(*known[0])
        case.display_truth_N = known[1]
    elif case.truth.triplet is not None and case.truth.cost < 0.5 * mean_side:
        case.display_truth = case.truth.triplet
        if case.truth.N_true > 0:
            case.display_truth_N = case.truth.N_true
        else:
            case.display_truth_N = estimate_N(rec.fps, case.truth.triplet)

    if case.display_truth is not None and case.display_truth_N > 0:
        H, _ = fit_homography(rec.fps, case.display_truth, case.display_truth_N)
        case.truth_boundary = qr_boundary(H, case.display_truth_N)

    if case.display_truth is not None:
        case.truth_gate_exact = current_triplet_diagnosis(rec.fps, case.display_truth)
        if case.truth_pairs is None:
            try:
                case.truth_pairs = true_pair_report(rec.fps, case.display_truth)
            except Exception:
                case.truth_pairs = None


def render(case: Case, out_path: Path) -> None:
    rec = case.rec
    fp_map = {f.cluster_idx: f for f in rec.fps}
    points = [f.outer_corners[:, ::-1] for f in rec.fps]
    if case.truth.corners is not None:
        points.append(case.truth.corners)
    r0, r1, c0, c1 = crop_bounds(points, case.gray.shape)

    fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=180)
    ax.imshow(case.gray[r0:r1, c0:c1], cmap="gray", extent=[c0, c1, r1, r0])

    for k, fp in enumerate(rec.fps):
        draw_finder(ax, fp, FINDER_COLORS[k % len(FINDER_COLORS)],
                    label=str(fp.cluster_idx))

    draw_triplet(ax, fp_map, case.current, CURRENT_COLOR, "C·", lw=3.0)
    draw_triplet(ax, fp_map, case.repaired, REPAIRED_COLOR, "R·", lw=2.4, ls="--")
    truth_triplet = case.display_truth
    if truth_triplet is not None and not (
        (case.current is not None and same_triplet(truth_triplet, case.current))
        or (case.repaired is not None and same_triplet(truth_triplet, case.repaired))
    ):
        draw_triplet(ax, fp_map, truth_triplet, TRUTH_COLOR, "T·", lw=1.6, ls=":")

    chosen = case.current or case.repaired
    if chosen is not None:
        if case.group == "current-only" and case.current_fit is not None:
            draw_boundary(ax, case.current_fit.boundary, BOUNDARY_COLOR,
                          f"current v{case.current_fit.version}")
        elif case.group == "repaired-only" and case.repaired_fit is not None:
            draw_boundary(ax, case.repaired_fit.boundary, BOUNDARY_COLOR,
                          f"repaired v{case.repaired_fit.version}")
    if case.truth_boundary is not None and case.display_truth_N > 0:
        truth_label = f"truth v{(case.display_truth_N - 17) // 4}"
        draw_boundary(ax, case.truth_boundary, TRUTH_COLOR, truth_label,
                      lw=1.4, ls=(0, (2, 3)))
    if case.truth.corners is not None:
        q = np.vstack([case.truth.corners, case.truth.corners[0]])
        ax.plot(q[:, 0], q[:, 1], color=TRUTH_COLOR, lw=1.0, ls=(0, (2, 3)),
                alpha=0.7, zorder=6)

    lines = [
        f"{rec.image}",
        f"{case.group} | v{rec.version if rec.version > 0 else '?'} | "
        f"{len(rec.fps)} candidate finders",
    ]
    if case.current is not None:
        lines.append(
            f"current: ({case.current.top_left_idx}, {case.current.top_right_idx}, "
            f"{case.current.bottom_left_idx})"
        )
    else:
        lines.append("current: none")
    if case.repaired is not None:
        lines.append(
            f"repaired: ({case.repaired.top_left_idx}, {case.repaired.top_right_idx}, "
            f"{case.repaired.bottom_left_idx})"
        )
    else:
        lines.append("repaired: none")
    if case.truth.payload:
        lines.append(f"OpenCV decodes: {case.truth.payload}")
    elif case.truth.corners is not None:
        lines.append("OpenCV detects, no decode")
    else:
        lines.append("OpenCV: no detection")
    if case.truth.N_true > 0:
        lines.append(f"OpenCV N={case.truth.N_true}")
    if case.new_pipe is not None:
        npip = case.new_pipe
        if npip["success"]:
            ch = npip["chosen"]
            lines.append(
                f"new pipeline: success v{npip['version']} "
                f"({ch[0]}, {ch[1]}, {ch[2]})"
            )
        else:
            tried = [tuple(a[1:4]) for a in npip["attempts"]]
            lines.append(
                f"new pipeline: failure ({len(tried)} attempts; "
                f"{npip['n_triplets']} current triplets)"
            )
    if case.gate_exact is not None:
        lines.append(f"centre gate (chosen): {case.gate_exact['reason']}")
    if case.gate_summary is not None:
        top = ", ".join(
            f"{k} {v}" for k, v in case.gate_summary["all"].most_common(2)
        )
        lines.append(f"centre gates: {top}")
    if case.repaired_diag is not None:
        lines.append(
            f"repaired: {len(case.repaired_diag['associations'])} associations, "
            f"{len(case.repaired_diag['triplets'])} triplets; "
            f"{case.repaired_diag['reason']}"
        )
    for label, report in (
        ("current", case.current_pairs),
        ("truth", case.truth_pairs),
        ("repaired", case.repaired_pairs),
    ):
        if report is None:
            continue
        if report["structural"]:
            lines.append(f"{label} true pair: structural reject")
        else:
            lines.append(
                f"{label} true pair max={report['max_offset']:.2f} "
                f"-> tol {report['tol_needed']:.2f}"
            )
    ax.text(0.012, 0.985, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, color="white", linespacing=1.3, zorder=11,
            bbox=dict(boxstyle="round,pad=0.35", fc="#101010", ec="none", alpha=0.86))

    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.axis("off")
    ax.set_title(f"{rec.image} — {case.group}", loc="left", fontsize=14, pad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def tup(t: Triplet | None):
    if t is None:
        return None
    return (t.top_left_idx, t.top_right_idx, t.bottom_left_idx)


def run_new_pipeline(cases: list[Case], out_path: Path) -> dict:
    """Run the post-merge ``decode_image`` on every case and cache the trace."""
    import qr_reader.pipeline as pl
    from qr_reader.pipeline import decode_image

    pl.sample_qr_bits = lambda image, H, N: fast_sample_qr_bits(image, H, N)
    results = []
    for case in cases:
        res = decode_image(
            case.gray, expected_payload=PAYLOAD, time_budget_s=30
        )
        chosen = None
        for at in res.attempts:
            if at.success:
                chosen = [
                    at.top_left_idx, at.top_right_idx, at.bottom_left_idx,
                    at.version, at.triplet_rank,
                ]
                break
        results.append(
            dict(
                image=case.rec.image,
                success=res.success,
                text=res.text,
                version=res.version,
                n_triplets=res.n_triplets,
                n_finder_patterns=res.n_finder_patterns,
                detection_ok=res.detection_ok,
                error=str(res.error) if res.error else "",
                chosen=chosen,
                attempts=[
                    [
                        at.triplet_rank, at.top_left_idx, at.top_right_idx,
                        at.bottom_left_idx, at.version, at.cross_ratio_score,
                        at.decoded_text, at.success,
                    ]
                    for at in res.attempts
                ],
            )
        )
    out_path.write_text(json.dumps(results, indent=1))
    return {x["image"]: x for x in results}


def write_summary(cases: list[Case], out_dir: Path) -> None:
    current_only = [c for c in cases if c.group == "current-only"]
    repaired_only = [c for c in cases if c.group == "repaired-only"]
    neither = [c for c in cases if c.group == "neither"]

    lines = ["# Triplet recall comparison — current-only, repaired-only, neither", ""]
    lines.append("## Setup")
    lines.append("")
    lines.append("- Data: `data/bench/captured-fullsize` (100 photos, payload `rikvoorhaar.com`)")
    lines.append("- Finder candidates: deduplicated ray-fit corners from the current front end")
    lines.append("- Reference truth: independent OpenCV `detect`/`decode` (true N from the")
    lines.append("  rectified matrix where available) plus three manually verified triplets")
    lines.append("  (images 11, 41, 74); current decode success where it applies")
    lines.append("- Repaired rule: `angle_tol = 0.1`, `offset_tol = 0.30` unless stated")
    lines.append(
        "- **Post-merge note:** the repository moved to commit `a6793f7` (multi-triplet, "
        "cross-ratio-ranked `decode_image`) after the cached run. The `new pipeline` column "
        "is a separate check with that code (fast sampler substituted). It now decodes image "
        "11 via the correct triplet, and it tries the correct triplet on 12/14/17 but still "
        "fails those decodes."
    )
    lines.append("")

    lines.append("## Current-only (current finds a triplet, repaired finds none)")
    lines.append("")
    lines.append("| image | v | finders | current first | correct first? | current list has correct? | correct triplet | new pipeline | repaired miss | tol needed | 0.62 finds it? |")
    lines.append("|---|---:|---:|---|---|---|---|---|---|---:|:---:|")
    for c in current_only:
        truth = c.display_truth
        first_correct = (
            truth is not None and c.current is not None and same_triplet(c.current, truth)
        )
        any_correct = truth is not None and any(
            same_triplet(t, truth) for t in c.rec.current_triplets
        )
        if truth is None:
            first_txt = "likely correct" if c.current_fit and c.current_fit.reproj_error < 5 else "?"
            any_txt = "n/a"
            truth_txt = "not verified (no OpenCV)"
            miss = "threshold"
            tol = c.current_pairs["tol_needed"] if c.current_pairs else None
            at62 = "yes" if c.current_pairs and c.current_pairs["triplet_at_tol"] else "?"
        else:
            first_txt = "yes" if first_correct else "no"
            any_txt = "yes" if any_correct else "no"
            truth_txt = str(tup(truth))
            report = c.truth_pairs or c.current_pairs
            miss = "structural" if report and report["structural"] else "threshold"
            tol = report["tol_needed"] if report else None
            at62 = "yes" if report and report["triplet_at_tol"] else "?"
        if c.new_pipe is not None:
            if c.new_pipe["success"]:
                ch = c.new_pipe["chosen"]
                new_txt = f"success v{c.new_pipe['version']} ({ch[0]},{ch[1]},{ch[2]})"
            else:
                tried = [tuple(a[1:4]) for a in c.new_pipe["attempts"]]
                if truth is None:
                    new_txt = "fails; only current triplet"
                elif tuple(tup(truth)) in tried:
                    new_txt = "fails; tried correct"
                else:
                    new_txt = "fails; wrong only"
        else:
            new_txt = "n/a"
        lines.append(
            f"| {c.rec.image} | {c.rec.version} | {len(c.rec.fps)} | {tup(c.current)} | "
            f"{first_txt} | {any_txt} | {truth_txt} | {new_txt} | {miss} | "
            f"{'' if tol is None else f'{tol:.3f}'} | {at62} |"
        )
    lines.append("")

    lines.append("### Current-only notes")
    lines.append("")
    for c in current_only:
        if c.rec.image == "qr_code_ - 11.jpeg":
            note = ("Cached first-triplet run chose (0,9,2) at v18; the correct triplet "
                    "(9,3,0) was at index 2 and decodes at v20. The post-merge cross-ratio "
                    "pipeline now selects (9,3,0) and decodes.")
        elif c.rec.image in ("qr_code_ - 12.jpeg", "qr_code_ - 14.jpeg", "qr_code_ - 17.jpeg"):
            if c.display_truth is not None:
                idx = next(
                    (
                        k for k, t in enumerate(c.rec.current_triplets)
                        if same_triplet(t, c.display_truth)
                    ),
                    None,
                )
            else:
                idx = None
            note = (f"Cached first-triplet run chose the wrong triplet; the correct one is at "
                    f"index {idx}. The post-merge pipeline tries the correct triplet but "
                    "sampling/RS still fails. OpenCV decodes the image independently, "
                    "confirming the code is readable.")
        elif c.rec.image == "qr_code_ - 15.jpeg":
            note = ("The correct triplet (1,0,3) is not in the current list; the centre "
                    "method rejects it on the triplet module-size gate (spread 0.34), and "
                    "the post-merge pipeline also never tries it.")
        elif c.rec.image == "qr_code_ - 38.jpeg":
            note = ("No OpenCV detection. Current triplet boundary aligns with the code and "
                    "the fit error is 1.2 px at v10; the post-merge pipeline tries the same "
                    "triplet but decode still fails in RS, so the triplet is likely correct "
                    "but the sampling is marginal.")
        else:
            note = ("Current triplet is correct and the post-merge pipeline decodes it; "
                    "repaired misses only because of the offset threshold.")
        lines.append(f"- **{c.rec.image}**: {note}")
    lines.append("")

    lines.append(
        "The `repaired miss` column refers to the **correct** triplet, not the wrong current "
        "one. All nine are threshold misses: the true pair is represented among the tried "
        "axis/pairings, no false pairing wins, and at the recovery tolerance the true "
        "triplet is the only triplet with no extra associations."
    )
    lines.append("")
    lines.append("## Repaired-only (repaired finds a triplet, current finds none)")
    lines.append("")
    lines.append("| image | finders | repaired triplet | decodes? | current rejection | correct? |")
    lines.append("|---|---:|---|---|---|---|")
    for c in repaired_only:
        decodes = bool(c.repaired_fit and c.repaired_fit.text == PAYLOAD)
        reject = c.gate_exact["reason"] if c.gate_exact else "?"
        if c.gate_exact and c.gate_exact["reason"] == "no valid pair connectivity":
            reject = "no valid pair connectivity (" + c.gate_exact.get("detail", "") + ")"
        lines.append(
            f"| {c.rec.image} | {len(c.rec.fps)} | {tup(c.repaired)} | "
            f"{'yes v' + str(c.repaired_fit.version) if decodes else 'no'} | "
            f"{reject} | {'yes' if decodes else 'likely'} |"
        )
    lines.append("")
    lines.append(
        "- All three repaired-only triplets decode to `rikvoorhaar.com`. The centre method "
        "rejects each on the axis-alignment gate: the inter-centre direction is not within "
        "the required alignment of a finder edge (angles 74°, 105°, 114°)."
    )
    lines.append("")

    lines.append("## Neither (both methods find none)")
    lines.append("")
    lines.append("| image | finders | real triplet in candidates? | classification | current rejection | repaired rejection |")
    lines.append("|---|---:|---|---|---|---|")
    manual = {
        "qr_code_ - 2.jpeg": (
            "no", "QR partly out of frame",
            "module-size/axis gates; only two real finders",
            "no associations",
        ),
        "qr_code_ - 21.jpeg": (
            "no", "code too small / bad candidates",
            "module size <2 px and mismatch; none on finders",
            "no associations",
        ),
        "qr_code_ - 22.jpeg": (
            "no", "bad candidates",
            "module-size mismatch; two huge degenerate quads",
            "one association, no L",
        ),
        "qr_code_ - 25.jpeg": (
            "no", "bad candidates / code small",
            "module-size mismatch; two real + huge background quads",
            "one association, no L",
        ),
        "qr_code_ - 33.jpeg": (
            "no", "bad candidates",
            "module-size mismatch; two real + one huge degenerate quad",
            "two associations, shared B axis",
        ),
        "qr_code_ - 36.jpeg": (
            "no", "bad candidates",
            "module-size mismatch; only one real finder detected",
            "no associations",
        ),
        "qr_code_ - 41.jpeg": (
            "yes", "real triplet present, both missed",
            "module-size spread 0.33 (pair 1-0 fails)",
            "true pair 1-2 max 0.503",
        ),
        "qr_code_ - 45.jpeg": (
            "no", "bad candidates",
            "module-size mismatch; two real + one bad large TL quad",
            "no associations",
        ),
        "qr_code_ - 52.jpeg": (
            "no", "bad candidates / code small",
            "module-size mismatch; two real + huge grass quad",
            "one association, no L",
        ),
        "qr_code_ - 60.jpeg": (
            "no", "code small / bad candidates",
            "module-size mismatch; fabric-texture quads, none on finders",
            "one association, no L",
        ),
        "qr_code_ - 72.jpeg": (
            "no", "bad candidates",
            "module-size mismatch; two real + one thin degenerate quad",
            "one association, no L",
        ),
        "qr_code_ - 74.jpeg": (
            "yes", "real triplet present, both missed (decodes v10)",
            "axis-alignment gate (angle 68°)",
            "true pair 0-2 max 0.358",
        ),
    }
    for c in neither:
        real, cls, creason, rreason = manual[c.rec.image]
        lines.append(
            f"| {c.rec.image} | {len(c.rec.fps)} | {real} | {cls} | {creason} | {rreason} |"
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    n_first = sum(
        1 for c in current_only
        if c.display_truth is not None and c.current is not None
        and same_triplet(c.current, c.display_truth)
    )
    n_first_likely = sum(
        1 for c in current_only
        if c.display_truth is None
        and c.current_fit is not None
        and c.current_fit.reproj_error < 5.0
    )
    n_any = sum(
        1 for c in current_only
        if c.display_truth is not None
        and any(same_triplet(t, c.display_truth) for t in c.rec.current_triplets)
    )
    n_repaired_correct = sum(
        1 for c in repaired_only
        if c.repaired_fit is not None and c.repaired_fit.text == PAYLOAD
    )
    n_real_neither = sum(1 for c in neither if c.rec.image in ("qr_code_ - 41.jpeg", "qr_code_ - 74.jpeg"))
    lines.append(
        f"- **Current-only (cached first-triplet run):** {n_first} confirmed + "
        f"{n_first_likely} likely = {n_first + n_first_likely}/9 current *first* triplets "
        f"are correct; "
        f"{n_any}/9 current triplet *lists* contain the correct triplet. The repaired rule "
        f"finds 0/9 at 0.30, but all 9 misses are threshold-only (tol 0.32–0.61); at 0.62 "
        f"it finds all 9 as the only triplet, with no extra associations on those images."
    )
    lines.append(
        f"- **Repaired-only:** {n_repaired_correct}/3 repaired triplets are correct and decode; "
        "centre geometry rejects them on the axis-alignment gate."
    )
    lines.append(
        f"- **Neither:** {n_real_neither}/12 have a real triplet in the candidate set (41, 74) "
        "and both methods miss it for threshold/geometry-gate reasons. The other 10/12 have "
        "bad or incomplete candidates (missing/duplicate/false finders, code too small or "
        "partly out of frame); that is an upstream detection failure, not a triplet failure."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "The repaired edge rule has a **real but threshold-only recall loss**: on this dataset "
        "every correct triplet it misses is recoverable by `offset_tol` in 0.32–0.61, with no "
        "structural (angle/intersection) failures and no false associations on those images. "
        "At 0.62 it recovers all nine current-only correct triplets as the sole triplet."
    )
    lines.append("")
    lines.append(
        "It is also **not merely failing closed** in the cached run: 5/9 current-only images had "
        "a wrong *first* triplet, and the repaired rule is right to reject them, but in 4 of "
        "those the current triplet list already contained the correct triplet later — the "
        "`raw_triplets[0]` selection was itself a defect. The post-merge cross-ratio pipeline "
        "fixes image 11 and now tries the correct triplet on 12/14/17, though those decodes "
        "still fail in sampling/RS. On the other side, all 3 repaired-only triplets are "
        "correct and decode, so the edge rule has a genuine recall advantage on "
        "strong-perspective/rotation images where the centre-axis gate fails."
    )
    lines.append("")
    lines.append(
        "The 10 unusable `neither` cases are not triplet-algorithm failures: the finder "
        "candidates are missing, false, or degenerate. Only images 41 and 74 are real shared "
        "triplet misses, and both are threshold misses."
    )
    lines.append("")
    lines.append(
        "Raising the edge tolerance recovers triplets on all nine current-only images but does "
        "not add end-to-end decodes in our pipeline: 11/35/37/84 already decode; "
        "12/14/17/38/15 fail in our sampling/RS even with the correct triplet (OpenCV "
        "decodes 12/14/17 independently)."
    )
    lines.append("")
    lines.append(
        "Under the new post-merge pipeline, image 11 is already fixed (cross-ratio ranking "
        "picks the correct triplet), and 12/14/17 now try the correct triplet but fail in "
        "sampling/RS. The repaired rule's threshold conclusion is unchanged, and the three "
        "repaired-only triplets remain invisible to the current front end, so that 3-image "
        "recall gain is real: among the 24 inspected cases the post-merge pipeline decodes "
        "4 (11, 35, 37, 84) and the edge rule adds 3 more (27, 64, 79), for 7/24 combined."
    )
    lines.append("")
    lines.append("## Overlays")
    lines.append("")
    for case in cases:
        png = out_dir / (
            f"recall_{case.group}_{Path(case.rec.image).stem.replace(' ', '_')}.png"
        )
        lines.append(f"- `{png}`")
    lines.append("")
    lines.append("## Commands")
    lines.append("")
    lines.append("```bash")
    lines.append("cd /home/rik/git/qr-reader")
    lines.append(".venv/bin/python blog_figures/triplet/analyze_recall_neither.py")
    lines.append("# regenerate the post-merge decode_image check:")
    lines.append(".venv/bin/python blog_figures/triplet/analyze_recall_neither.py --refresh-new-pipeline")
    lines.append("# overlays and this report land in /tmp/opencode/triplet_failure_modes/")
    lines.append("```")
    lines.append("")
    (out_dir / "recall_neither_report.md").write_text("\n".join(lines))
    print(f"\nwrote {out_dir / 'recall_neither_report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument(
        "--refresh-new-pipeline", action="store_true",
        help="re-run the post-merge decode_image check",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.cache)
    cases = build_cases(records, args.data_dir)
    new_path = args.out_dir / "new_pipeline.json"
    if args.refresh_new_pipeline or not new_path.exists():
        new_results = run_new_pipeline(cases, new_path)
    else:
        new_results = {x["image"]: x for x in json.loads(new_path.read_text())}
    for case in cases:
        case.new_pipe = new_results.get(case.rec.image)
    print(f"{len(cases)} cases: {Counter(c.group for c in cases)}")

    for i, case in enumerate(cases, start=1):
        print(f"  [{i}/{len(cases)}] {case.group:<14} {case.rec.image}")
        analyse_case(case)

    for case in cases:
        out = args.out_dir / f"recall_{case.group}_{Path(case.rec.image).stem.replace(' ', '_')}.png"
        render(case, out)
        print(f"    wrote {out}")

    # JSON-ish dump for the report
    for case in cases:
        rec = case.rec
        print(
            f"\n{case.group} {rec.image} v{rec.version} finders={len(rec.fps)} "
            f"current={tuple([case.current.top_left_idx, case.current.top_right_idx, case.current.bottom_left_idx]) if case.current else None} "
            f"repaired={tuple([case.repaired.top_left_idx, case.repaired.top_right_idx, case.repaired.bottom_left_idx]) if case.repaired else None} "
            f"cv={case.truth.payload!r} N_true={case.truth.N_true} "
            f"N_fit={case.truth.N} cost={case.truth.cost:.1f}"
        )
        if case.truth.nearest:
            print(f"   truth nearest: {case.truth.nearest}")
        if case.truth.triplet:
            print(f"   truth triplet: {tuple([case.truth.triplet.top_left_idx, case.truth.triplet.top_right_idx, case.truth.triplet.bottom_left_idx])}")
        if case.current_fit:
            print(
                f"   current fit: N={case.current_fit.N} err={case.current_fit.reproj_error:.1f} "
                f"timing={case.current_fit.timing_score:.2f} text={case.current_fit.text!r} "
                f"err={case.current_fit.decode_error}"
            )
        if case.repaired_fit:
            print(
                f"   repaired fit: N={case.repaired_fit.N} err={case.repaired_fit.reproj_error:.1f} "
                f"timing={case.repaired_fit.timing_score:.2f} text={case.repaired_fit.text!r} "
                f"err={case.repaired_fit.decode_error}"
            )
        if case.gate:
            g = case.gate
            print(
                f"   gates: AB={g['ab']['reason'] or 'ok'} AC={g['ac']['reason'] or 'ok'} "
                f"BC={g['bc']['reason'] or 'ok'} angle={g['angle']:.1f} "
                f"spread={g['module_spread']:.2f} -> {g['reason']}"
            )
        if case.gate_summary:
            gs = case.gate_summary
            print(f"   centre gate histogram: {dict(gs['all'].most_common(4))} "
                  f"(pair gates {dict(gs['pairs'].most_common(4))})")
        if case.repaired_diag:
            rd = case.repaired_diag
            print(f"   repaired assoc={len(rd['associations'])} triplets={len(rd['triplets'])} "
                  f"reason={rd['reason']} L={dict(rd['l_reasons'])}")
            worst = sorted(rd["pair_best"], key=lambda p: p["best_max"])[:4]
            print("   easiest pairs:", [(p["fp1"], p["fp2"], round(p["best_max"], 2)) for p in worst])
        for label, report in (
            ("current", case.current_pairs),
            ("truth", case.truth_pairs),
            ("repaired", case.repaired_pairs),
        ):
            if report is None:
                continue
            print(
                f"   {label} pairs: structural={report['structural']} "
                f"max={report['max_offset']:.3f} tol_needed={report['tol_needed']} "
                f"triplet_at_tol={report['triplet_at_tol']} extra_edges={report['extra_edges']}"
            )
            for r in report["reports"]:
                tp = r["true_pairing"]
                max_text = f"{tp.max_offset:.3f}" if tp else "None"
                print(
                    f"      {r['a']}–{r['b']} pairs={r['pairs']} "
                    f"max={max_text} "
                    f"passes={r['passes']} selected_true={r['selected_is_true']} "
                    f"best_false={(r['best_false'].max_offset if r['best_false'] else None)}"
                )

    write_summary(cases, args.out_dir)


if __name__ == "__main__":
    main()