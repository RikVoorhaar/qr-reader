#!/usr/bin/env python3
"""Fair, reproducible timing + accuracy comparison on the 100-image captured set.

Compares:

1. this project's ``qr_reader.pipeline.decode_image``
2. ``cv2.QRCodeDetector`` (OpenCV)
3. zxing-cpp (built out-of-tree from the pinned ``third_party/zxing-cpp`` submodule)

Protocol
--------
* Dataset: ``data/bench/captured/`` — 100 JPEGs, payload ``rikvoorhaar.com``.
* Images are preloaded into memory as uint8 grayscale arrays *before* any
  timing, so no implementation pays for disk I/O or JPEG decoding.  All three
  implementations therefore time detection + decode only.
* One warm-up pass over ``WARMUP_IMAGES`` images per implementation
  (discarded), then ``N_PASSES`` timed passes over all 100 images.
* Single threaded: ``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
  ``MKL_NUM_THREADS``, ``NUMEXPR_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS`` are
  forced to 1 *before* numpy/cv2 are imported, and ``cv2.setNumThreads(1)`` is
  called.  zxing-cpp's core has no thread pool; this is verified at runtime by
  counting entries in ``/proc/self/task``.
* Accuracy: exact payload match after normalising (strip whitespace / NULs),
  counted per unique image.

Run with the repo venv, pointing ``PYTHONPATH`` at the out-of-tree zxing-cpp
python module build::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=/tmp/opencode/zxing-cpp-build/wrappers/python \
    .venv/bin/python blog_figures/results/benchmark_comparison.py
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time

# --- threading must be pinned before numpy / cv2 are imported -----------------
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)

import zxingcpp  # noqa: E402

import qr_reader  # noqa: E402
from qr_reader.pipeline import decode_image  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = os.path.join(REPO_ROOT, "data", "bench", "captured")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_comparison.json")

EXPECTED_PAYLOAD = "rikvoorhaar.com"
N_PASSES = 3
WARMUP_IMAGES = 5


# ----------------------------------------------------------------------------- helpers
def normalize(payload: str | None) -> str:
    """Strip surrounding whitespace and NUL bytes before comparing payloads."""
    if payload is None:
        return ""
    return payload.replace("\x00", "").strip()


def load_images() -> tuple[list[str], list[np.ndarray]]:
    names = sorted(n for n in os.listdir(DATASET) if n.lower().endswith((".jpg", ".jpeg", ".png")))
    images = []
    for name in names:
        img = cv2.imread(os.path.join(DATASET, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit(f"could not read {name}")
        images.append(img)
    return names, images


def lscpu_info() -> dict:
    try:
        raw = subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    keep = ("Model name", "Architecture", "CPU(s)", "Thread(s) per core", "Core(s) per socket", "CPU max MHz")
    info = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip().startswith(keep):
                info[k.strip()] = v.strip()
    info["lscpu_raw"] = raw.strip().splitlines()
    return info


def git(*args: str, cwd: str = REPO_ROOT) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def thread_count() -> int:
    try:
        return len(os.listdir("/proc/self/task"))
    except OSError:  # pragma: no cover
        return -1


# ----------------------------------------------------------------------------- decoders
class Ours:
    name = "qr_reader (this project)"

    def decode(self, gray: np.ndarray) -> tuple[str, str | None]:
        """Return (payload, error_text)."""
        res = decode_image(gray, expected_payload=EXPECTED_PAYLOAD)
        payload = res.text or ""
        err = None
        if not payload and res.error is not None:
            err = repr(res.error)
        return payload, err


class OpenCVDetectAndDecode:
    name = "OpenCV detectAndDecode"

    def __init__(self) -> None:
        # The detector object is constructed once, outside the timed region.
        self._det = cv2.QRCodeDetector()

    def decode(self, gray: np.ndarray) -> tuple[str, str | None]:
        text, _points, _straight = self._det.detectAndDecode(gray)
        return text or "", None


class OpenCVMulti:
    name = "OpenCV detectAndDecodeMulti"

    def __init__(self) -> None:
        self._det = cv2.QRCodeDetector()

    def decode(self, gray: np.ndarray) -> tuple[str, str | None]:
        ok, texts, _points, _straights = self._det.detectAndDecodeMulti(gray)
        if not ok or not texts:
            return "", None
        # Single-QR assumption: take the first non-empty payload.
        for t in texts:
            if t:
                return t, None
        return "", None


class ZXingCPP:
    name = "zxing-cpp"

    def decode(self, gray: np.ndarray) -> tuple[str, str | None]:
        # formats=QRCode restricts to QR (no 1D barcodes); try_rotate / try_downscale /
        # try_invert stay at the library defaults.
        res = zxingcpp.read_barcode(gray, formats=zxingcpp.BarcodeFormat.QRCode)
        if res is None:
            return "", None
        return res.text or "", None


# ----------------------------------------------------------------------------- runner
def run_impl(impl, images: list[np.ndarray], names: list[str]) -> dict:
    # Warm-up (discarded).
    for img in images[:WARMUP_IMAGES]:
        try:
            impl.decode(img)
        except Exception:
            pass

    payloads: list[str] = [""] * len(images)
    errors: list[str | None] = [None] * len(images)
    per_pass_times: list[list[float]] = []
    threads_before = thread_count()

    for _pass in range(N_PASSES):
        pass_times: list[float] = []
        for i, img in enumerate(images):
            t0 = time.perf_counter()
            try:
                payload, err = impl.decode(img)
            except Exception as exc:  # recorded as a failure, never fatal
                payload, err = "", f"{type(exc).__name__}: {exc}"
            dt = time.perf_counter() - t0
            pass_times.append(dt)
            if _pass == 0:
                payloads[i] = payload
                errors[i] = err
        per_pass_times.append(pass_times)

    threads_after = thread_count()

    payloads_norm = [normalize(p) for p in payloads]
    correct = [p == EXPECTED_PAYLOAD for p in payloads_norm]
    all_times = [t for times in per_pass_times for t in times]

    def ms(times: list[float]) -> float:
        return 1000.0 * statistics.fmean(times)

    def med(times: list[float]) -> float:
        return 1000.0 * statistics.median(times)

    return {
        "name": impl.name,
        "payloads": payloads,
        "payloads_normalized": payloads_norm,
        "errors": errors,
        "correct_per_image": correct,
        "n_correct": sum(correct),
        "accuracy_pct": 100.0 * sum(correct) / len(images),
        "per_pass_times_s": per_pass_times,
        "per_pass_mean_ms": [ms(t) for t in per_pass_times],
        "per_pass_median_ms": [med(t) for t in per_pass_times],
        "per_pass_total_s": [sum(t) for t in per_pass_times],
        "mean_ms": ms(all_times),
        "median_ms": med(all_times),
        "total_s": sum(all_times),
        "mean_total_s_per_pass": statistics.fmean([sum(t) for t in per_pass_times]),
        "min_ms": 1000.0 * min(all_times),
        "max_ms": 1000.0 * max(all_times),
        "threads_before": threads_before,
        "threads_after": threads_after,
        "failed_images": [names[i] for i, c in enumerate(correct) if not c],
    }


def repo_state() -> dict:
    """State of the tree that actually provides ``qr_reader`` (may be a snapshot worktree)."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(qr_reader.__file__)))
    return {
        "path": pkg_dir,
        "commit": git("rev-parse", "HEAD", cwd=pkg_dir),
        "dirty": bool(git("status", "--porcelain", "--", "src", cwd=pkg_dir)),
    }


def main() -> int:
    repo_at_start = repo_state()
    names, images = load_images()
    print(f"loaded {len(images)} images, shape {images[0].shape}, dtype {images[0].dtype}", flush=True)

    impls = [Ours(), OpenCVDetectAndDecode(), OpenCVMulti(), ZXingCPP()]
    results = []
    for impl in impls:
        print(f"running {impl.name} ...", flush=True)
        results.append(run_impl(impl, images, names))

    n = len(images)
    config = {
        "dataset": "data/bench/captured",
        "n_images": n,
        "image_shape": list(images[0].shape),
        "image_dtype": str(images[0].dtype),
        "expected_payload": EXPECTED_PAYLOAD,
        "n_passes": N_PASSES,
        "warmup_images": WARMUP_IMAGES,
        "image_loading": "preloaded once into memory (uint8 grayscale, cv2.IMREAD_GRAYSCALE) before timing; excluded from all timers",
        "timed_region": "detection + decode of a preloaded grayscale array; no disk I/O, no import, no decoder construction",
        "opencv_call": "cv2.QRCodeDetector().detectAndDecode(gray) [primary]; detectAndDecodeMulti(gray) also measured",
        "zxingcpp_call": "zxingcpp.read_barcode(gray, formats=BarcodeFormat.QRCode) with library defaults (try_rotate=True, try_downscale=True, try_invert=True)",
        "ours_call": "qr_reader.pipeline.decode_image(gray, expected_payload='rikvoorhaar.com')",
        "threading": {
            "env": {v: os.environ[v] for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "cv2_getNumThreads": cv2.getNumThreads(),
            "notes": "zxing-cpp core is single threaded (no thread pool); /proc/self/task counts recorded per implementation",
        },
        "accuracy_rule": "exact match to 'rikvoorhaar.com' after remove NULs + strip whitespace",
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "qr_reader_module": qr_reader.__file__,
        "qr_reader_commit": git("rev-parse", "HEAD", cwd=os.path.dirname(os.path.dirname(os.path.abspath(qr_reader.__file__)))),
    }

    env_info = {
        "cpu": lscpu_info(),
        "os": platform.platform(),
        "os_release": {k: v for k, v in (line.split("=", 1) for line in open("/etc/os-release").read().splitlines() if "=" in line and not line.startswith("#"))},
        "kernel": platform.release(),
        "nproc": os.cpu_count(),
    }

    zxing_commit = git("rev-parse", "HEAD", cwd=os.path.join(REPO_ROOT, "third_party", "zxing-cpp"))
    zxing_info = {
        "submodule_commit": zxing_commit,
        "submodule_describe": git("describe", "--tags", "--always", cwd=os.path.join(REPO_ROOT, "third_party", "zxing-cpp")),
        "build_dir": "/tmp/opencode/zxing-cpp-build",
        "cmake_options": [
            "-DCMAKE_BUILD_TYPE=Release",
            "-DZXING_EXAMPLES=ON",
            "-DZXING_UNIT_TESTS=OFF",
            "-DZXING_BLACKBOX_TESTS=OFF",
            "-DZXING_EXAMPLES_QT=OFF",
            "-DZXING_PYTHON_MODULE=ON",
        ],
        "python_module": "/tmp/opencode/zxing-cpp-build/wrappers/python/zxingcpp.abi3.so (nanobind, abi3/stable ABI)",
        "python_headers_used": "CPython 3.12.12 (uv-managed standalone); module is abi3 so it imports in the 3.12.3 venv",
    }

    repo_info = {
        "measured_tree": repo_at_start,
        "measured_tree_at_end": repo_state(),
        "main_repo": {
            "path": REPO_ROOT,
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "note": "measured_tree is the tree that provided qr_reader; it is a pinned git worktree so concurrent edits in the main checkout cannot contaminate the run",
    }

    # images where the implementations disagree on the (normalised) payload
    key_impls = [r for r in results if r["name"] in (Ours.name, ZXingCPP.name, OpenCVDetectAndDecode.name)]
    disagreements = []
    for i, name in enumerate(names):
        payloads = {r["name"]: r["payloads_normalized"][i] for r in key_impls}
        verdicts = {r["name"]: r["correct_per_image"][i] for r in key_impls}
        if len(set(verdicts.values())) > 1:
            disagreements.append({"image": name, "payloads": payloads, "correct": verdicts})

    # Is zxing-cpp's miss rate a configuration artefact?  Re-run only its failures
    # under a few alternative settings (not part of the headline table).
    zxing_res = next(r for r in results if r["name"] == ZXingCPP.name)
    zxing_fail_idx = [i for i, c in enumerate(zxing_res["correct_per_image"]) if not c]
    zxing_sensitivity = {}
    for label, kwargs in {
        "default": {},
        "binarizer=FixedThreshold": {"binarizer": zxingcpp.Binarizer.FixedThreshold},
        "binarizer=GlobalHistogram": {"binarizer": zxingcpp.Binarizer.GlobalHistogram},
        "try_downscale=False": {"try_downscale": False},
        "try_invert=False": {"try_invert": False},
    }.items():
        recovered = 0
        for i in zxing_fail_idx:
            res = zxingcpp.read_barcode(images[i], formats=zxingcpp.BarcodeFormat.QRCode, **kwargs)
            if res is not None and normalize(res.text) == EXPECTED_PAYLOAD:
                recovered += 1
        zxing_sensitivity[label] = {"recovered": recovered, "of": len(zxing_fail_idx)}

    doc = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config,
        "environment": env_info,
        "versions": {
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "zxingcpp": zxing_info,
        },
        "repo": repo_info,
        "images": names,
        "per_image": [
            {
                "image": names[i],
                "payloads": {r["name"]: r["payloads"][i] for r in results},
                "correct": {r["name"]: bool(r["correct_per_image"][i]) for r in results},
                "times_ms": {r["name"]: [1000.0 * t for t in [r["per_pass_times_s"][p][i] for p in range(N_PASSES)]] for r in results},
                "errors": {r["name"]: r["errors"][i] for r in results if r["errors"][i]},
            }
            for i in range(n)
        ],
        "results": results,
        "disagreements": disagreements,
        "zxing_sensitivity": zxing_sensitivity,
    }

    with open(OUT_JSON, "w") as fh:
        json.dump(doc, fh, indent=2)

    # ------------------------------------------------------------------- report
    def fmt_row(r: dict) -> str:
        return (
            f"{r['name']:<30} | {r['n_correct']:>4}/{n} | {r['accuracy_pct']:>9.1f}% | "
            f"{r['mean_ms']:>13.2f} | {r['median_ms']:>15.2f} | {r['mean_total_s_per_pass']:>10.3f}"
        )

    print()
    print(f"{'implementation':<30} | {'ok':>7} | {'accuracy':>10} | {'mean ms/image':>13} | {'median ms/image':>15} | {'total s/100':>10}")
    print("-" * 106)
    for r in results:
        print(fmt_row(r))

    print()
    print("per-pass means (ms/image):")
    for r in results:
        means = ", ".join(f"{m:.2f}" for m in r["per_pass_mean_ms"])
        totals = ", ".join(f"{t:.3f}s" for t in r["per_pass_total_s"])
        print(f"  {r['name']:<30} mean=[{means}]  total=[{totals}]")

    print()
    print(f"disagreeing images (ours vs zxing-cpp vs OpenCV detectAndDecode): {len(disagreements)}")
    for d in disagreements[:20]:
        print(f"  {d['image']:<24} correct={d['correct']} payloads={d['payloads']}")

    print()
    print("failed images per implementation:")
    for r in results:
        print(f"  {r['name']:<30} {len(r['failed_images'])} -> {r['failed_images'][:8]}{' ...' if len(r['failed_images']) > 8 else ''}")

    print()
    print("zxing-cpp sensitivity on its own failures (not part of the headline table):")
    for label, v in zxing_sensitivity.items():
        print(f"  {label:<28} recovers {v['recovered']}/{v['of']}")

    print()
    print(f"wrote {OUT_JSON}")
    if repo_at_start != repo_state():
        print("WARNING: measured tree changed during the run; results are not trustworthy", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
