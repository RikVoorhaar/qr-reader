# %%
import numpy as np

from qr_reader.detector.detector import detect_corners, detect_sample, detect_homography, compute_qr_corners, \
    sample_qr_bits
from qr_reader.qr_gen import generate_test_image, make_qr_image
from qr_reader.decoder.decoder import decode
from matplotlib import pyplot as plt

content = "https://www.rikvoorhaar.com"
version = 12
image = generate_test_image(content=content, version=version, border=10, noise_std=100, seed=None)
img = make_qr_image(content=content, version=version, border=0)
bits_correct = ~(img[5::10, 5::10].astype(bool))

homography_det, version_det = detect_homography(image)
N = 4 * version_det + 17
corners_det = compute_qr_corners(homography_det, N)
bits_det = sample_qr_bits(image, homography_det, N).T
assert np.all(bits_det == bits_correct)
text_det = decode(bits_correct)
# %%
plt.matshow(~bits_det.T, cmap="gray")
plt.show()
# %%
import numpy as np

corners_aug = np.concat([corners_det, corners_det[:1]])
plt.plot(corners_aug[:, 0], corners_aug[:, 1], "ro-")
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
# %%
