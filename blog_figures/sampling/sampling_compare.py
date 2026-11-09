#!/usr/bin/env python3
"""Rough affine-alignment comparison for one captured QR photo.

This is a standalone blog-figure helper (deliberately outside the package).
It reproduces the *rough* part of the pipeline:

1. Run the RLE finder-pattern scan (``alignment.py``) and cluster the
   candidates (``clustering.py``).
2. Pick the three finder centres that form a valid triplet.
3. Fit an **affine** transform (not a homography) from the three known
   finder-centre grid positions: TL (3.5, 3.5), TR (N-3.5, 3.5) and
   BL (3.5, N-3.5).
4. Warp/crop the photo through the inverse affine into an N-pitch square.
5. Sample every module centre, threshold, and compare with a clean reference
   matrix generated with ``qrcode`` from the decoded payload/version/ECL/mask.

The dataset metadata (version, payload, ECL, mask) comes from the regular
detector + decoder so the reference is unambiguous; the sampling being
*evaluated* is purely the rough affine one.

Run from the repository root:

    python blog_figures/sampling/sampling_compare.py              # scan v1 candidates
    python blog_figures/sampling/sampling_compare.py IMAGE_PATH   # one image
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import qrcode
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.decoder.format_info import decode_format_info  # noqa: E402
from qr_reader.detector.alignment import find_alignment_patterns_2d  # noqa: E402
from qr_reader.detector.clustering import cluster_candidates  # noqa: E402
from qr_reader.detector.detector import _run_detection  # noqa: E402
from qr_reader.detector.edge_fitting import DISTANCE_THRESHOLD, MAX_GAP  # noqa: E402
from qr_reader.detector.finder_pattern import (  # noqa: E402
    FinderPattern,
    find_valid_triplets,
)
from qr_reader.detector.ray_fit import fit_finder_ray  # noqa: E402
from qr_reader.detector.roi import cluster_to_bbox, cutout  # noqa: E402
from qr_reader.qr_gen import binarize_image  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured"
EXPECTED_PAYLOAD = "rikvoorhaar.com"

ECL_TO_QRCODE = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}

@dataclass
class RleFinderCentres:
    """Three RLE-cluster finder centres, in (row, col), TL / TR / BL order."""

    centres_rc: np.ndarray  # (3, 2)
    module_pitch: float  # rough px per module from the cluster width / 7

    @property
    def centres_xy(self) -> np.ndarray:
        return self.centres_rc[:, ::-1].copy()


@dataclass
class Comparison:
    image_path: Path
    image_name: str
    version: int
    N: int
    payload: str
    ecl: str
    mask: int
    finders: RleFinderCentres
    affine_grid_to_image: np.ndarray  # 3x3, (x, y) grid -> (x, y) image
    affine_image_to_rect: np.ndarray  # 3x3, image -> rectified pixel (x, y)
    rectified: np.ndarray  # uint8 grayscale square
    rectified_size: int
    sampled_bits: np.ndarray  # (N, N) bool, indexed [col, row] like decoder
    reference_bits: np.ndarray  # (N, N) bool, indexed [col, row]
    threshold: float
    sampled_values: np.ndarray  # (N, N) float, indexed [row, col]
    scan_order: list[tuple[int, int]]  # (row, col) in scan order

    @property
    def wrong_mask(self) -> np.ndarray:
        """(N, N) bool, indexed [row, col]."""
        return (self.sampled_bits.T != self.reference_bits.T)

    @property
    def num_wrong(self) -> int:
        return int(self.wrong_mask.sum())

    @property
    def wrong_locations(self) -> list[tuple[int, int]]:
        rr, cc = np.where(self.wrong_mask)
        return list(zip(rr.tolist(), cc.tolist()))


# ── RLE finder centres ───────────────────────────────────────────────────────


def _rle_finder_centres(gray: np.ndarray) -> RleFinderCentres:
    """Run the RLE candidate scan and return the triplet's cluster centres.

    The centres are *not* ray-fitted: they are the weighted row / midpoint of
    the six RLE boundaries stored in each merged ``CandidateCluster``.  The
    ray-fit step is only used to decide which clusters form the triplet,
    mirroring ``detector._run_detection``.
    """
    img_binary = binarize_image(gray)
    rows, cols_all = find_alignment_patterns_2d(img_binary, np.log(1.3))
    if len(rows) == 0:
        raise RuntimeError("No RLE candidates found")
    clusters = cluster_candidates(rows, cols_all)

    fps: list[FinderPattern] = []
    score_map: dict[int, float] = {}
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
        corners_xy = fr.corners + np.array([c0, r0], dtype=np.float64)
        wh = np.ptp(corners_xy, axis=0)
        if wh[0] < 2.0 * m_est or wh[1] < 2.0 * m_est:
            continue
        fps.append(
            FinderPattern(cluster_idx=ci, outer_corners=corners_xy[:, ::-1])
        )
        score_map[ci] = fr.score

    # Same dedup rule as detector._run_detection.
    keep = np.ones(len(fps), dtype=bool)
    for i in range(len(fps)):
        if not keep[i]:
            continue
        ci = fps[i].outer_corners.mean(axis=0)
        seg_i = float(np.linalg.norm(fps[i].outer_corners[0] - fps[i].outer_corners[1]))
        for j in range(i + 1, len(fps)):
            if not keep[j]:
                continue
            cj = fps[j].outer_corners.mean(axis=0)
            seg_j = float(
                np.linalg.norm(fps[j].outer_corners[0] - fps[j].outer_corners[1])
            )
            if float(np.linalg.norm(ci - cj)) < min(seg_i, seg_j):
                if score_map[fps[i].cluster_idx] >= score_map[fps[j].cluster_idx]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    fps = [fp for fp, k in zip(fps, keep) if k]

    triplets = find_valid_triplets(fps, score_map)
    if not triplets:
        raise RuntimeError("No finder triplet found")

    # Prefer the triplet with the largest finder-centre span (stable choice).
    def span(t):
        idx = {f.cluster_idx: f for f in fps}
        pts = np.array(
            [
                idx[t.top_left_idx].outer_corners.mean(axis=0),
                idx[t.top_right_idx].outer_corners.mean(axis=0),
                idx[t.bottom_left_idx].outer_corners.mean(axis=0),
            ]
        )
        return float(
            np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[0])
        )

    triplet = max(triplets, key=span)
    centres, pitches = [], []
    for ci in (triplet.top_left_idx, triplet.top_right_idx, triplet.bottom_left_idx):
        cluster = clusters[ci]
        centres.append(
            [
                float(cluster.row),
                float(cluster.cols[2] + cluster.cols[3]) / 2.0,
            ]
        )
        pitches.append(float(cluster.cols[5] - cluster.cols[0]) / 7.0)
    return RleFinderCentres(
        centres_rc=np.array(centres, dtype=np.float64),
        module_pitch=float(np.median(pitches)),
    )


# ── Affine rectification and sampling ────────────────────────────────────────


def _affine_image_to_grid(
    finders: RleFinderCentres, N: int
) -> np.ndarray:
    """Affine image (x, y) → QR grid (x, y) from the three finder centres."""
    src = np.array(
        [[3.5, 3.5], [N - 3.5, 3.5], [3.5, N - 3.5]], dtype=np.float64
    )
    dst = finders.centres_xy
    M = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
    A_grid_to_image = np.eye(3, dtype=np.float64)
    A_grid_to_image[:2, :] = M
    return np.linalg.inv(A_grid_to_image)


def _warp_rectified(
    gray: np.ndarray, A_image_to_grid: np.ndarray, size: int, N: int
) -> np.ndarray:
    """Warp image → square rectified image with ``size`` px and ``size/N`` pitch."""
    pitch = size / N
    # image (x, y) -> grid -> rectified pixel = pitch * grid
    S = np.array([[pitch, 0.0, 0.0], [0.0, pitch, 0.0], [0.0, 0.0, 1.0]])
    M = (S @ A_image_to_grid)[:2, :]
    return cv2.warpAffine(
        gray,
        M,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _sample_centres(
    gray: np.ndarray, A_grid_to_image: np.ndarray, N: int
) -> np.ndarray:
    """Bilinear sample of every module centre. Returns (N, N) indexed [row, col]."""
    rows, cols = np.mgrid[0:N, 0:N]
    grid_xy = np.stack([cols + 0.5, rows + 0.5], axis=-1).reshape(-1, 2)
    img_xy = (
        A_grid_to_image
        @ np.column_stack([grid_xy, np.ones(len(grid_xy))]).T
    ).T[:, :2]
    vals = map_coordinates(
        gray.astype(np.float64),
        np.stack([img_xy[:, 1], img_xy[:, 0]]),
        order=1,
        mode="nearest",
    )
    return vals.reshape(N, N)


def _threshold_from_finders(
    gray: np.ndarray, A_inv: np.ndarray, N: int
) -> float:
    """Midpoint of black/white centre medians over the three finder patterns."""
    values = _sample_centres(gray, A_inv, N)
    black, white = [], []
    origins = [(0, 0), (0, N - 7), (N - 7, 0)]
    for r0, c0 in origins:
        for dr in range(7):
            for dc in range(7):
                v = values[r0 + dr, c0 + dc]
                inner3 = 2 <= dr <= 4 and 2 <= dc <= 4
                border = dr in (0, 6) or dc in (0, 6)
                if inner3 or border:
                    black.append(v)
                else:
                    white.append(v)
    return float((np.median(black) + np.median(white)) / 2.0)


def _reference_matrix(
    payload: str, version: int, ecl: str, mask: int
) -> np.ndarray:
    """Clean QR matrix from the same payload/version/ECL/mask, [col, row]."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ECL_TO_QRCODE[ecl],
        box_size=1,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(payload)
    qr.make(fit=False)
    mat = np.array(qr.get_matrix(), dtype=bool)  # [row, col]
    return mat.T  # decoder convention: [col, row]


def compare_image(image_path: Path) -> Comparison:
    image_path = Path(image_path)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Metadata from the regular detector + decoder.  This is only used to
    # name the symbol; the sampling below uses the rough affine transform.
    det = _run_detection(gray)
    if not det.success or det.H is None or det.version is None:
        raise RuntimeError(f"Detection failed: {det.error}")
    N = 4 * det.version + 17
    from qr_reader.detector.sample import sample_qr_bits

    det_bits = sample_qr_bits(gray, det.H, N)
    payload = decode_qr(det_bits)
    ecl_idx, mask = decode_format_info(det_bits, det.version)
    from qr_reader.decoder.tables import ECL_NAMES

    ecl = ECL_NAMES[ecl_idx]

    finders = _rle_finder_centres(gray)
    A_image_to_grid = _affine_image_to_grid(finders, N)
    A_grid_to_image = np.linalg.inv(A_image_to_grid)

    rect_size = N * 24
    rectified = _warp_rectified(gray, A_image_to_grid, rect_size, N)

    sampled_values = _sample_centres(gray, A_grid_to_image, N)
    threshold = _threshold_from_finders(gray, A_grid_to_image, N)
    sampled_rc = sampled_values < threshold  # True = dark, indexed [row, col]
    sampled_bits = sampled_rc.T  # [col, row]

    reference_bits = _reference_matrix(payload, det.version, ecl, mask)

    # Row-major scan order; (row, col) coordinates.
    scan_order = [(r, c) for r in range(N) for c in range(N)]

    return Comparison(
        image_path=image_path,
        image_name=image_path.name,
        version=det.version,
        N=N,
        payload=payload,
        ecl=ecl,
        mask=mask,
        finders=finders,
        affine_grid_to_image=A_grid_to_image,
        affine_image_to_rect=np.array(
            [
                [rect_size / N, 0.0, 0.0],
                [0.0, rect_size / N, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        @ A_image_to_grid,
        rectified=rectified,
        rectified_size=rect_size,
        sampled_bits=sampled_bits,
        reference_bits=reference_bits,
        threshold=threshold,
        sampled_values=sampled_values,
        scan_order=scan_order,
    )


def _natural_sort_key(path: Path) -> tuple[int, ...]:
    import re

    return tuple(int(n) for n in re.findall(r"\d+", path.stem))


def scan_candidates(data_dir: Path, version: int = 1) -> list[Comparison]:
    paths = sorted(data_dir.glob("*.jpeg"), key=_natural_sort_key)
    results: list[Comparison] = []
    for path in paths:
        try:
            comp = compare_image(path)
        except Exception as exc:  # noqa: BLE001 - scan should continue
            print(f"  skip {path.name}: {type(exc).__name__}: {exc}")
            continue
        if comp.version != version:
            continue
        if comp.payload != EXPECTED_PAYLOAD:
            continue
        results.append(comp)
        print(
            f"  {path.name:<24} v{comp.version:<2} N={comp.N:<3} "
            f"{comp.ecl} mask={comp.mask}  wrong={comp.num_wrong:>3}  "
            f"wrong@={comp.wrong_locations[:8]}"
        )
    return results


def _print_summary(comp: Comparison) -> None:
    print(f"image      : {comp.image_path}")
    print(f"version    : {comp.version} (N={comp.N})")
    print(f"payload    : {comp.payload!r}")
    print(f"ECL / mask : {comp.ecl} / {comp.mask}")
    print(f"threshold  : {comp.threshold:.1f}")
    print(
        "finders rc : "
        + ", ".join(f"({r:.1f}, {c:.1f})" for r, c in comp.finders.centres_rc)
    )
    print(f"wrong bits : {comp.num_wrong} / {comp.N * comp.N}")
    print(f"locations  : {comp.wrong_locations}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image", nargs="?", type=Path, help="captured JPEG to compare"
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="scan all v1 images in the captured dataset",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="captured dataset directory"
    )
    parser.add_argument(
        "--version", type=int, default=1, help="version to filter when scanning"
    )
    args = parser.parse_args()

    if args.scan or args.image is None:
        print(f"Scanning v{args.version} candidates in {args.data_dir} ...")
        results = scan_candidates(args.data_dir, args.version)
        if not results:
            print(f"No v{args.version} images found.")
            return
        best = min(results, key=lambda c: (c.num_wrong, c.image_name))
        print("\nBest candidate:")
        _print_summary(best)
        return

    _print_summary(compare_image(args.image))


if __name__ == "__main__":
    main()
