#!/usr/bin/env python3
"""Build the two "From pixels to candidates" GIFs for the QR-reader post.

Outputs (both 600 px tall, same crop, same palette):

* ``blog/src/assets/blog/qr-reader/rle-scan.gif``
    A full-width horizontal scan sweeps the Otsu-thresholded crop.  Every row
    is run-length encoded and tested against 1:1:3:1:1 at the detector's 30%
    tolerance.  Matches leave persistent candidate boxes behind.

* ``blog/src/assets/blog/qr-reader/candidate-clustering.gif``
    The raw row matches fade into merged candidate clusters, then each cluster
    gets the detector's vertical 1:1:3:1:1 check.  Real finder patterns pass
    (green); two genuine row-scan false positives fail (red) and fade away.

Run from the repository root:

    .venv/bin/python blog_figures/candidates/make_figures.py

The script is deterministic: the photo, crop, Otsu threshold, RLE scan and
greedy merge order are all fixed.  It writes only the two GIFs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from common import (
    CAND,
    CLUSTER,
    FAIL,
    FPS,
    INK,
    OUT_DIR,
    PASS,
    SCAN,
    CandidateData,
    binary_base,
    build_shared_palette,
    candidate_box,
    candidate_segment,
    cluster_box,
    composite,
    lerp_box,
    load_candidate_data,
    rgba,
    save_gif,
    smoothstep,
    verdict_badge,
    vertical_line,
)

# ── RLE scan ────────────────────────────────────────────────────────────────

SCAN_STEP = 3  # pixels per frame
SCAN_INTRO_FRAMES = 6
TRAIL_ALPHAS = (0.55, 0.42, 0.31, 0.22, 0.14, 0.08, 0.04)


def _rle_scan_frame(
    data: CandidateData, y: int, *, show_scanline: bool = True
) -> Image.Image:
    canvas = binary_base(data).convert("RGBA")
    height, width = data.binary.shape
    calls = []

    # Persistent candidate boxes for every match the scan has already passed.
    for row, cols in zip(data.rows, data.cols):
        if row > y:
            continue
        run_height = float(cols[3] - cols[2])
        calls.append(
            candidate_box(
                float(row),
                cols,
                run_height,
                CAND,
                alpha=0.90,
                width=2,
                fill_alpha=0.0,
            )
        )

    if show_scanline:
        # Fading trail of recent scanlines.
        for k, alpha in enumerate(TRAIL_ALPHAS, start=1):
            trail_y = y - k * SCAN_STEP
            if trail_y < 0:
                continue
            calls.append(
                lambda pen, yy=trail_y, a=alpha: pen.line(
                    [(0, yy), (width - 1, yy)],
                    fill=rgba(SCAN, a),
                    width=2,
                )
            )

        # Current scanline: dark halo first, then bright cyan.
        calls.append(
            lambda pen: pen.line(
                [(0, y), (width - 1, y)],
                fill=rgba(INK, 0.45),
                width=5,
            )
        )
        calls.append(
            lambda pen: pen.line(
                [(0, y), (width - 1, y)],
                fill=rgba(SCAN, 1.0),
                width=3,
            )
        )

        # On this pass, draw the five runs of every match the scanline just
        # crossed, with the 1:1:3:1:1 widths made visible (the middle run is
        # drawn thicker).  A 3 px step can skip a match row, so use the rows
        # in the last step rather than only the exact scanline row.
        for row, cols in zip(data.rows, data.cols):
            if not (y - SCAN_STEP < row <= y):
                continue
            for j in range(5):
                xa, xb = float(cols[j]), float(cols[j + 1])
                run_width = 10 if j == 2 else 4
                calls.append(
                    lambda pen, a=xa, b=xb, w=run_width, r=float(row): (
                        pen.line(
                            [(a, r), (b, r)],
                            fill=rgba(CAND, 1.0),
                            width=w,
                        )
                    )
                )
            for boundary in cols:
                calls.append(
                    lambda pen, x=float(boundary), r=float(row): pen.line(
                        [(x, r - 14), (x, r + 14)],
                        fill=rgba(CAND, 1.0),
                        width=2,
                    )
                )

    return composite(canvas, calls).convert("RGB")


def build_rle_scan(data: CandidateData) -> list[Image.Image]:
    height = data.binary.shape[0]
    frames: list[Image.Image] = []

    # A brief look at the thresholded image before the sweep starts.
    for _ in range(SCAN_INTRO_FRAMES):
        frames.append(_rle_scan_frame(data, -1, show_scanline=False))

    for y in range(0, height, SCAN_STEP):
        frames.append(_rle_scan_frame(data, y))

    # Final clean frame: all accumulated candidates, no scanline.
    frames.append(_rle_scan_frame(data, height, show_scanline=False))
    return frames


# ── Candidate clustering + vertical check ───────────────────────────────────

CLUSTER_INTRO_FRAMES = 12
CLUSTER_PHASE_FRAMES = 90
CLUSTER_SETTLE_FRAMES = 12
LINE_DRAW_FRAMES = 5
VERDICT_HOLD_FRAMES = 8
FAIL_FADE_FRAMES = 9
PASS_PAUSE_FRAMES = 2


def _member_union_box(data: CandidateData, group) -> tuple[float, float, float, float]:
    boxes = []
    for index in group.members:
        cols = data.cols[index]
        row = float(data.rows[index])
        run_height = float(cols[3] - cols[2])
        boxes.append(
            (
                float(cols[0]),
                row - run_height,
                float(cols[5]),
                row + run_height,
            )
        )
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _cluster_frame(
    data: CandidateData,
    *,
    phase1_t: float | None = None,
    verdicts: dict[int, dict] | None = None,
    drawing: tuple[int, float] | None = None,
) -> Image.Image:
    """Render one clustering/vertical-check frame.

    ``phase1_t`` in [0, 1] animates raw segments into cluster boxes.  When it
    is ``None`` the clusters are fully settled and ``verdicts`` / ``drawing``
    drive the vertical-check phase.
    """
    canvas = binary_base(data).convert("RGBA")
    calls = []
    verdicts = verdicts or {}

    for order, group in enumerate(data.groups):
        final_box = group.box
        union_box = _member_union_box(data, group)

        if phase1_t is not None:
            delay = 0.10 * order
            u = smoothstep((phase1_t - delay) / 0.55)
            for index in group.members:
                cols = data.cols[index]
                row = float(data.rows[index])
                calls.append(
                    candidate_segment(
                        cols, row, CAND, alpha=1.0 * (1.0 - u), width=5
                    )
                )
            if u > 0.01:
                box = lerp_box(union_box, final_box, u)
                calls.append(
                    cluster_box(
                        box,
                        CLUSTER,
                        alpha=0.95 * min(1.0, u * 3.0),
                        width=4,
                        fill_alpha=0.10 * u,
                    )
                )
            continue

        # Settled clusters: raw segments are gone.
        verdict = verdicts.get(order)
        if verdict is not None:
            passed = verdict["passed"]
            alpha = verdict["alpha"]
            if alpha <= 0.01:
                continue
            color = PASS if passed else FAIL
            calls.append(
                cluster_box(
                    final_box,
                    color,
                    alpha=alpha,
                    width=4,
                    fill_alpha=0.10 * alpha,
                )
            )
            x = group.centre_x
            y0, y1 = final_box[1], final_box[3]
            calls.append(
                vertical_line(x, y0 - 6, y1 + 6, color, alpha=alpha, width=3)
            )
            calls.append(
                verdict_badge(
                    final_box[2], final_box[1], passed, alpha=alpha
                )
            )
        else:
            calls.append(
                cluster_box(
                    final_box,
                    CLUSTER,
                    alpha=0.95,
                    width=4,
                    fill_alpha=0.10,
                )
            )

        if drawing is not None and drawing[0] == order:
            progress = drawing[1]
            x = group.centre_x
            y0, y1 = final_box[1], final_box[3]
            calls.append(
                vertical_line(
                    x,
                    y0 - 6,
                    y0 - 6 + (y1 - y0 + 12) * progress,
                    CLUSTER,
                    alpha=1.0,
                    width=3,
                )
            )

    return composite(canvas, calls).convert("RGB")


def build_candidate_clustering(data: CandidateData) -> list[Image.Image]:
    frames: list[Image.Image] = []

    # Raw row matches, before any merging.
    for _ in range(CLUSTER_INTRO_FRAMES):
        frames.append(_cluster_frame(data, phase1_t=0.0))

    # Phase 1: segments fade into growing cluster boxes.
    for i in range(CLUSTER_PHASE_FRAMES):
        t = i / (CLUSTER_PHASE_FRAMES - 1)
        frames.append(_cluster_frame(data, phase1_t=t))

    # Brief pause on the settled clusters.
    for _ in range(CLUSTER_SETTLE_FRAMES):
        frames.append(_cluster_frame(data))

    # Phase 2: vertical check, one cluster at a time, top to bottom.
    verdicts: dict[int, dict] = {}
    for order, group in enumerate(data.groups):
        for k in range(1, LINE_DRAW_FRAMES + 1):
            frames.append(
                _cluster_frame(
                    data,
                    verdicts=verdicts,
                    drawing=(order, k / LINE_DRAW_FRAMES),
                )
            )

        passed = data.vertical_pass[order]
        verdicts[order] = {"passed": passed, "alpha": 1.0}
        for _ in range(VERDICT_HOLD_FRAMES):
            frames.append(_cluster_frame(data, verdicts=verdicts))

        if not passed:
            for k in range(1, FAIL_FADE_FRAMES + 1):
                verdicts[order]["alpha"] = 1.0 - k / FAIL_FADE_FRAMES
                frames.append(_cluster_frame(data, verdicts=verdicts))
            verdicts[order]["alpha"] = 0.0
        else:
            for _ in range(PASS_PAUSE_FRAMES):
                frames.append(_cluster_frame(data, verdicts=verdicts))

    # Final frame: only the kept clusters.
    frames.append(_cluster_frame(data, verdicts=verdicts))
    return frames


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args()

    data = load_candidate_data()
    n_removed = sum(not passed for passed in data.vertical_pass)
    print(f"image: {data.image_name}")
    print(f"crop: {data.crop.shape[1]}x{data.crop.shape[0]}")
    print(f"row matches: {len(data.rows)}")
    print(f"raw candidates: {len(data.rows)}")
    print(f"clusters: {len(data.groups)}")
    print(f"removed by vertical check: {n_removed}")

    rle_frames = build_rle_scan(data)
    cluster_frames = build_candidate_clustering(data)
    palette = build_shared_palette()

    save_gif(args.out_dir / "rle-scan.gif", rle_frames, palette)
    save_gif(
        args.out_dir / "candidate-clustering.gif", cluster_frames, palette
    )

    for name, frames in (
        ("rle-scan.gif", rle_frames),
        ("candidate-clustering.gif", cluster_frames),
    ):
        path = args.out_dir / name
        seconds = (len(frames) - 1) / FPS + 1.0
        print(f"wrote {path} ({len(frames)} frames, ~{seconds:.1f}s)")


if __name__ == "__main__":
    main()
