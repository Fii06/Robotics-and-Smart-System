# Install library yang diperlukan
!pip install opencv-python-headless matplotlib scikit-learn

# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import urllib.request

# Fungsi untuk menampilkan gambar
def show_image(title, img, cmap=None):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Fungsi untuk mengunduh gambar dari URL
def download_image(url, filename):
    urllib.request.urlretrieve(url, filename)
    print(f"Gambar berhasil diunduh: {filename}")

# URL gambar baru
image_url = "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"
template_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/JPEG_example_flower.jpg/800px-JPEG_example_flower.jpg"

# Unduh gambar dan template
download_image(image_url, "image.jpg")
download_image(template_url, "template.jpg")

image_path = "image.jpg"
template_path = "template.jpg"

# 1. Ekstraksi garis dengan Hough Transform
def extract_lines(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    show_image("Lines Detected", img)

# 2. Template matching untuk deteksi objek
def template_matching(image_path, template_path):
    img = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = template.shape[::-1]
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    show_image("Template Matching", img)

# 3. Pembuatan pyramid gambar
def image_pyramid(image_path):
    img = cv2.imread(image_path)
    layer = img.copy()
    for i in range(3):  # Buat 3 level
        layer = cv2.pyrDown(layer)
        show_image(f"Pyramid Level {i+1}", layer)

# 4. Deteksi lingkaran menggunakan Hough Transform
def detect_circles(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=10, maxRadius=50
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    show_image("Circles Detected", img)

# 5. Ekstraksi warna dominan pada gambar
def extract_dominant_color(image_path, k=3):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    reshaped_img = img.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(reshaped_img)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    dominant_color = colors[np.argmax(np.bincount(labels))]
    print("Dominant Color (RGB):", dominant_color)

    plt.imshow([[dominant_color]])
    plt.axis('off')
    plt.title("Dominant Color")
    plt.show()

# 6. Deteksi kontur pada gambar
def detect_contours(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    show_image("Contours Detected", img)

# Eksekusi fungsi-fungsi
extract_lines(image_path)
template_matching(image_path, template_path)
image_pyramid(image_path)
detect_circles(image_path)
extract_dominant_color(image_path)
detect_contours(image_path)
