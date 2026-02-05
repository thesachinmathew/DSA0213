# Common Imports & Image Upload

import cv2, numpy as np, matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
img_name = list(uploaded.keys())[0]
img = cv2.imread(img_name, 0)  # grayscale


# Question 1: Image Smoothing – Gaussian & Median Filtering

noisy = img.copy()
p = 0.02
r = np.random.rand(*img.shape)
noisy[r < p/2] = 0
noisy[r > 1 - p/2] = 255

gaussian = cv2.GaussianBlur(noisy, (5,5), 1)
median = cv2.medianBlur(noisy, 5)

plt.figure(figsize=(8,6))
titles = ["Original", "Noisy", "Gaussian", "Median"]
images = [img, noisy, gaussian, median]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# Question 2: Contrast Enhancement – Hist & Stretch

hist_eq = cv2.equalizeHist(img)
stretch = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

plt.figure(figsize=(9,3))
titles = ["Original", "Histogram EQ", "Contrast Stretch"]
images = [img, hist_eq, stretch]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# Question 3: Edge Detection – Sobel & Canny

blur = cv2.GaussianBlur(img, (3,3), 0)

sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
canny = cv2.Canny(blur, 100, 200)

plt.figure(figsize=(9,3))
titles = ["Original", "Sobel", "Canny"]
images = [img, sobel, canny]

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.show()


# Question 4: Image Restoration – Deblurring & Denoising

blurred = cv2.GaussianBlur(img, (7,7), 1.5)
noise = np.random.normal(0, 10, img.shape)
noisy_blur = np.clip(blurred + noise, 0, 255).astype(np.uint8)

kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
deblur = cv2.filter2D(noisy_blur, -1, kernel)
denoise = cv2.fastNlMeansDenoising(noisy_blur, None, 15, 7, 21)

plt.figure(figsize=(8,6))
titles = ["Original", "Noisy+Blur", "Deblurred", "Denoised"]
images = [img, noisy_blur, deblur, denoise]

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.show()

# =========================================================
# Question 5: Morphological Operations – Erosion to Closing
# =========================================================
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

erode = cv2.erode(binary, kernel, 1)
dilate = cv2.dilate(binary, kernel, 1)
open_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
close_img = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10,6))
titles = ["Gray", "Binary", "Erosion", "Dilation", "Opening", "Closing"]
images = [img, binary, erode, dilate, open_img, close_img]

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")
plt.show()
