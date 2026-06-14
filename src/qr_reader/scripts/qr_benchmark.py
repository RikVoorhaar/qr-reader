# %%
from time import perf_counter
import numpy as np

from qr_reader.detector.detector import detect_corners, detect_sample, detect_homography, compute_qr_corners, \
    sample_qr_bits
from qr_reader.qr_gen import generate_test_image, make_qr_image
from qr_reader.decoder.decoder import decode
from matplotlib import pyplot as plt


def time_format(t):
    if t < 1:
        return f"{t * 1000:.2f}ms"
    return f"{t:.2f}s"

content = "https://www.rikvoorhaar.com"
version = 8
image = generate_test_image(content=content, version=version, border=20, noise_std=10, seed=2)
img = make_qr_image(content=content, version=version, border=0)
bits_correct = ~(img[5::10, 5::10].astype(bool))

t0 = perf_counter()
homography_det, version_det = detect_homography(image)
N = 4 * version_det + 17

corners_det = compute_qr_corners(homography_det, N)
t1 = perf_counter()
bits_det = sample_qr_bits(image, homography_det, N).T
t2 = perf_counter()

all_correct = np.all(bits_det == bits_correct)
print(f"all correct? {all_correct}")
if not all_correct:
    pct_incorrect = 100 * np.sum(bits_det != bits_correct) / bits_det.size
    print(f"pct incorrect: {pct_incorrect:.2f}%")

    
text_det = decode(bits_correct)
t3 = perf_counter()
print(f"text_det = {text_det}")
corners_aug = np.concat([corners_det, corners_det[:1]])
# plt.plot(corners_aug[:, 0], corners_aug[:, 1], "ro-")
# plt.imshow(image, cmap="gray")
# plt.axis("off")
# plt.show()
print(f"total time: {time_format(t3 - t0)}")
print(f"detect time: {time_format(t1 - t0)}")
print(f"sample time: {time_format(t2 - t1)}")
print(f"decode time: {time_format(t3 - t2)}")

# %%