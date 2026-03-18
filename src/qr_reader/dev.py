# %%
import qrcode
import matplotlib.pyplot as plt
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data('Some data')
qr.make(fit=True)

img = qr.make_image()
import numpy as np
img = np.array(img).astype(np.uint8)*255

plt.imshow(img, cmap='gray')

# %%
import cv2
rows,cols = img.shape
 
# cols-1 and rows-1 are the coordinate limits.
M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),20,1)
img_rotated = cv2.warpAffine(img,M,(cols,rows), borderValue=(255,255,255))

plt.imshow(img_rotated, cmap='gray')

# %%

# Define source points (corners of the original QR code image)
src_pts = np.float32([
    [0, 0],
    [cols-1, 0],
    [cols-1, rows-1],
    [0, rows-1]
])

# Define destination points to apply a perspective transformation
dst_pts = np.float32([
    [20, 50],         # top-left is shifted to the right and down
    [cols-25, 0],     # top-right a bit left and unshifted vertically
    [cols-10, rows-30], # bottom-right up and in
    [40, rows-20]     # bottom-left in and up
])

# Compute the perspective transform matrix
M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the perspective warp
img_persp = cv2.warpPerspective(img, M_persp, (cols, rows), borderValue=(255,255,255))

# Visualize
plt.imshow(img_persp, cmap='gray')
plt.title('Perspective Transformed QR Code')
plt.show()

# %%
noise = np.random.normal(0, 50, img.shape)
spatial_noise = cv2.GaussianBlur(noise, (3, 3), 0)
img_noisy = np.clip(img_persp*0.8 + spatial_noise, 0, 255).astype(np.uint8)
plt.imshow(img_noisy, cmap='gray')
plt.title('Noisy QR Code')
plt.show()

# %%
img_noisy = cv2.GaussianBlur(img_noisy, (5, 5), 0)
plt.imshow(img_noisy, cmap='gray')
plt.title('Blurred Noisy QR Code')
plt.show()

# %%
threshold = 128
img_binary = cv2.threshold(img_noisy, threshold, 255, cv2.THRESH_BINARY)[1].astype(bool)
plt.imshow(img_binary, cmap='gray')
plt.title('Binary Noisy QR Code')
plt.show()

# %%
first_row = img_binary[0, :]

def run_length_encoding(row):
    run_lengths = []
    current_run = 0
    current_value = row[0]
    for i in range(len(row)):
        if row[i] == current_value:
            current_run += 1
        else:
            run_lengths.append((current_value, current_run))
            current_value = row[i]
            current_run = 1
    return run_lengths

run_length_encoding(first_row)

# True is white, False is black
# We want to find a ratio of 1:1:3:1:1 for white:black:white:black:white, so True:False:True:False:True
# but within a tolerance of ~10% for each value.
# so for a window of 5 RLE values, if they start with True, we compute the total length, and then check if each member is in the expected range. 

row_num = 100
run_lengths = run_length_encoding(img_binary[row_num, :])
run_lengths_smart = np.diff(np.where(np.diff(img_binary[row_num, :]) != 0)[0], prepend=0)

offset = 1 
seq = run_lengths_smart[offset:5+offset]/sum(run_lengths_smart[offset:5+offset])
expected = np.array([1, 1, 3, 1, 1])
expected = expected/sum(expected)
log_expected = np.log(expected)

score = np.abs(np.max(np.log(seq)-log_expected))

# %%
from numpy.lib.stride_tricks import sliding_window_view

windows = sliding_window_view(run_lengths_smart, window_shape=5)
np.max(np.abs(np.log(windows / np.sum(windows, axis=1, keepdims=True)) - log_expected), axis=1)

# %%
max_error = np.log(1.3)    # 30% error
run_lengths_smart = np.diff(np.where(np.diff(img_binary) != 0)[0], prepend=0)
rows, columns = np.where(np.diff(img_binary))

row_changes = np.diff(rows)>0
run_lengths_smart = np.diff(columns)
run_lengths_smart[row_changes] = -1
run_lengths_smart

seqs = sliding_window_view(run_lengths_smart, window_shape=5)
scores = np.max(np.abs(np.log(seqs / np.sum(seqs, axis=1, keepdims=True)) - log_expected), axis=1)

(candidate_indices,) = np.where(scores < max_error)
candidate_rows = rows[candidate_indices]
candidate_column_starts = columns[candidate_indices]
candidate_column_ends = columns[candidate_indices + 5]


img_plot = img_binary.copy().astype(np.uint8)*255
img_plot = cv2.cvtColor(img_plot, cv2.COLOR_GRAY2BGR)
for row, start, end in zip(candidate_rows, candidate_column_starts, candidate_column_ends):
    img_plot[row, start:end] = (255, 0, 0)

plt.imshow(img_plot)
plt.title('Candidate QR Codes')
plt.show()

# %%

# next: vertical test (with  higher tolerance)
# space filling to find boundaries of alignment pattern