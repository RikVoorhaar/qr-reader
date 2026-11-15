#!/usr/bin/env python3
"""Scan the captured dataset for 3x3-vote vs single-centre disagreements.

For every real capture in ``data/bench/captured-fullsize``:

1. Run the production pipeline (``pipeline.decode_image``) to get ``H`` and
   ``version``.
2. Sample with the production 3x3 weighted-vote sampler and decode it.
3. Sample every module with a single centre pixel through the same ``H`` and
   the same adaptive threshold, then decode that.
4. Keep images where (2) decodes ``rikvoorhaar.com`` and (3) does not.
5. For each kept image, list modules where the single-centre bit is wrong
   (against the ``qrcode`` reference matrix) while the 3x3 vote is right.

Run from the repository root::

    .venv/bin/python blog_figures/sampling_vote/scan_candidates.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode
from scipy.ndimage import map_coordinates

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from qr_reader.decoder.decoder import decode as decode_qr  # noqa: E402
from qr_reader.decoder.format_info import decode_format_info  # noqa: E402
from qr_reader.decoder.tables import ECL_NAMES  # noqa: E402
from qr_reader.detector.homography import project_points  # noqa: E402
from qr_reader.detector.sample import (  # noqa: E402
    compute_adaptive_threshold,
    sample_qr_bits,
)
from qr_reader.pipeline import decode_image  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "bench" / "captured-fullsize"
PAYLOAD = "rikvoorhaar.com"

ECL_TO_QRCODE = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}

OFFSETS = np.array([0.25, 0.5, 0.75])


def natural_key(path: Path) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", path.stem))


def reference_matrix(version: int, ecl: str, mask: int) -> np.ndarray:
    qr = qrcode.QRCode(
        version=version,
        error_correction=ECL_TO_QRCODE[ecl],
        box_size=1,
        border=0,
        mask_pattern=mask,
    )
    qr.add_data(PAYLOAD)
    qr.make(fit=False)
    return np.array(qr.get_matrix(), dtype=bool).T  # [col, row]


def single_centre_values(gray: np.ndarray, H: np.ndarray, N: int) -> np.ndarray:
    rows, cols = np.mgrid[0:N, 0:N]
    grid_xy = np.column_stack(
        [(cols + 0.5).ravel(), (rows + 0.5).ravel()]
    ).astype(np.float64)
    img_xy = project_points(H, grid_xy)
    vals = map_coordinates(
        gray.astype(np.float64),
        np.stack([img_xy[:, 1], img_xy[:, 0]]),
        order=1,
        mode="nearest",
    )
    return vals.reshape(N, N)  # [row, col]


def nine_values(gray: np.ndarray, H: np.ndarray, row: int, col: int) -> np.ndarray:
    dy, dx = np.meshgrid(OFFSETS, OFFSETS, indexing="ij")
    grid_xy = np.column_stack(
        [(col + dx).ravel(), (row + dy).ravel()]
    ).astype(np.float64)
    img_xy = project_points(H, grid_xy)
    vals = map_coordinates(
        gray.astype(np.float64),
        np.stack([img_xy[:, 1], img_xy[:, 0]]),
        order=1,
        mode="nearest",
    )
    return vals.reshape(3, 3)


def safe_decode(bits: np.ndarray) -> tuple[str | None, str]:
    """Return ``(text, error)``; ``text`` is None when decoding raises."""
    try:
        return decode_qr(bits), ""
    except Exception as exc:  # noqa: BLE001 - a decode failure is data here
        return None, f"{type(exc).__name__}: {exc}"


def scan_one(path: Path) -> dict | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = decode_image(gray, expected_payload=PAYLOAD)
    if not result.success or result.H is None or result.version is None:
        return {"image": path.name, "detect_ok": False}
    H, version = result.H, result.version
    N = 4 * version + 17

    bits_3x3 = sample_qr_bits(gray, H, N)
    text_3x3, err_3x3 = safe_decode(bits_3x3)
    threshold = compute_adaptive_threshold(gray, H, N)

    centre = single_centre_values(gray, H, N)  # [row, col]
    light_centre = centre > threshold
    bits_centre = (~light_centre).T  # [col, row], True = dark

    text_centre, err_centre = safe_decode(bits_centre)

    ecl_idx, mask = decode_format_info(bits_3x3, version)
    ecl = ECL_NAMES[ecl_idx]
    ref = reference_matrix(version, ecl, mask)
    dark_3x3 = bits_3x3.T  # [row, col]
    dark_centre = bits_centre.T
    dark_ref = ref.T

    disagree = dark_centre != dark_3x3
    wrong_centre = dark_centre != dark_ref
    right_3x3 = dark_3x3 == dark_ref

    candidate_modules = []
    for r, c in zip(*np.where(disagree & wrong_centre & right_3x3)):
        vals = nine_values(gray, H, int(r), int(c))
        candidate_modules.append(
            {
                "row": int(r),
                "col": int(c),
                "threshold": float(threshold),
                "centre": float(centre[r, c]),
                "nine": vals.tolist(),
                "bit_centre": bool(dark_centre[r, c]),
                "bit_3x3": bool(dark_3x3[r, c]),
                "bit_ref": bool(dark_ref[r, c]),
                "centre_margin": float(abs(centre[r, c] - threshold)),
                "min_sample_margin": float(np.min(np.abs(vals - threshold))),
            }
        )

    return {
        "image": path.name,
        "detect_ok": True,
        "version": int(version),
        "N": int(N),
        "ecl": ecl,
        "mask": int(mask),
        "threshold": float(threshold),
        "text_3x3": text_3x3,
        "text_centre": text_centre,
        "err_3x3": err_3x3,
        "err_centre": err_centre,
        "n_bits_wrong_3x3": int((dark_3x3 != dark_ref).sum()),
        "n_bits_wrong_centre": int(wrong_centre.sum()),
        "modules": candidate_modules,
    }


def main() -> None:
    paths = sorted(DATA_DIR.glob("*.jpeg"), key=natural_key)
    print(f"scanning {len(paths)} images ...")
    rows = []
    all_info = []
    for k, path in enumerate(paths):
        info = scan_one(path)
        if info is None:
            continue
        all_info.append(info)
        if not info["detect_ok"]:
            print(f"[{k + 1:>3}/{len(paths)}] {path.name:<24} detect failed")
            continue
        condition = (
            info["text_3x3"] == PAYLOAD and info["text_centre"] != PAYLOAD
        )
        print(
            f"[{k + 1:>3}/{len(paths)}] {path.name:<24} v{info['version']:<2} "
            f"3x3={str(info['text_3x3']):<18} centre={str(info['text_centre']):<18} "
            f"wrong_3x3={info['n_bits_wrong_3x3']:<3} "
            f"wrong_centre={info['n_bits_wrong_centre']:<3} "
            f"cand={len(info['modules']):<3} {'<== KEEP' if condition else ''}"
        )
        if info["err_centre"]:
            print(f"      centre decode error: {info['err_centre']}")
        if info["err_3x3"]:
            print(f"      3x3 decode error: {info['err_3x3']}")
        if condition:
            rows.append(info)

    out = REPO_ROOT / "blog_figures" / "sampling_vote" / "scan_results.json"
    out.write_text(json.dumps(rows, indent=2))
    all_out = REPO_ROOT / "blog_figures" / "sampling_vote" / "scan_all.json"
    all_out.write_text(json.dumps(all_info, indent=2))
    n_detect = sum(1 for i in all_info if i["detect_ok"])
    print(
        f"\n{len(rows)} images met the decode condition; "
        f"{n_detect} detected, {len(all_info) - n_detect} failed detection."
    )

    for info in rows:
        print(f"\n=== {info['image']}  v{info['version']} N={info['N']} "
              f"{info['ecl']} mask={info['mask']} thr={info['threshold']:.1f}")
        for m in info["modules"]:
            print(
                f"  ({m['row']:>3},{m['col']:>3}) centre={m['centre']:>6.1f} "
                f"margin={m['centre_margin']:>5.1f} "
                f"min_sample_margin={m['min_sample_margin']:>5.1f} "
                f"bits c/3/ref={int(m['bit_centre'])}/{int(m['bit_3x3'])}/"
                f"{int(m['bit_ref'])}"
            )
            print("      nine=", np.round(np.array(m["nine"]), 1).tolist())


if __name__ == "__main__":
    main()
