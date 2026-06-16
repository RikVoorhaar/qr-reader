"""full-pipeline-canny.py — Pipeline up to clustered alignment patterns with Canny ROI visualization."""

# %%
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from qr_reader.detector.alignment import find_alignment_patterns_2d
from qr_reader.detector.clustering import cluster_candidates
from qr_reader.detector.roi import cluster_to_bbox, cutout
from qr_reader.qr_gen import binarize_image

# %%
# Load synthetic image (or generate one if data/synth is unavailable)

metadata_path = Path("data/synth/metadata.jsonl")
if metadata_path.exists():
    import json

    with open(metadata_path, "r") as f:
        metadata = [json.loads(line.strip()) for line in f]
    sample = metadata[0]
    QR_VERSION = sample["version"]
    QR_CONTENT = sample["payload"]
    img_path = Path("data/synth/images") / f"{sample['sample_index']:06d}.jpg"
    img = cv2.imread(str(img_path))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    from qr_reader.qr_gen import generate_test_image

    QR_VERSION = 12
    QR_CONTENT = "https://www.rikvoorhaar.com"
    img_gray = generate_test_image(version=QR_VERSION, content=QR_CONTENT, border=15)

# %%
# Binarize (Otsu)

img_binary = binarize_image(img_gray)

# %%
# Find alignment patterns (2-D: horizontal + vertical cross-validation)

max_error = np.log(1.3)
rows_valid, cols_valid_all = find_alignment_patterns_2d(img_binary, max_error)

# %%
# Cluster candidates

clusters = cluster_candidates(rows_valid, cols_valid_all)
print(f"Found {len(clusters)} clusters.")

# %%
# Per-cluster ROI → cutout (grayscale) → OpenCV Canny → display
# One figure per cluster, two subplots: cutout (left) + Canny edges (right)

canny_low = 50
canny_high = 150

for ci, cluster in enumerate(clusters):
    bbox = cluster_to_bbox(cluster, scale=1.5)
    roi = cutout(img_gray, bbox)
    edges = cv2.Canny(roi, canny_low, canny_high)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"Cluster {ci} — cutout + Canny edges")

    ax_left.imshow(roi, cmap="gray")
    ax_left.set_title("Grayscale cutout")
    ax_left.axis("off")

    ax_right.imshow(edges, cmap="gray")
    ax_right.set_title("Canny edges")
    ax_right.axis("off")

    plt.tight_layout()
    plt.show()
