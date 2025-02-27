import cv2
import numpy as np

# Đọc ảnh dưới dạng grayscale (1 kênh)
def load_grayscale_image(image_array):
    # Đọc ảnh trực tiếp dưới dạng grayscale (1 kênh)
    image_array = np.array(image_array)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return image

# Resize ảnh, đảm bảo giữ đúng 1 kênh
def resize_image(image, target_size=(640, 640)):
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return image_resized

# Bộ lọc Gaussian Blur
def gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

# Cân bằng histogram (CLAHE)
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)  # Áp dụng CLAHE trực tiếp lên ảnh grayscale
    return enhanced_image

# Hàm tiền xử lý ảnh (resize, khử nhiễu và cân bằng histogram)
def preprocess_image(image_array, target_size=(640, 640)):
    # Đọc ảnh ở định dạng grayscale (1 kênh)
    image = load_grayscale_image(image_array)

    # Resize ảnh và đảm bảo giữ đúng 1 kênh
    image_resized = resize_image(image, target_size)

    # Áp dụng Gaussian Blur (khử nhiễu)
    image_blurred = gaussian_blur(image_resized, kernel_size=(5, 5))

    # Áp dụng CLAHE (cân bằng histogram)
    enhanced_image = apply_clahe(image_blurred)

    # Đảm bảo ảnh có 1 kênh (grayscale) sau tất cả các bước tiền xử lý
    if len(enhanced_image.shape) == 3 and enhanced_image.shape[2] == 3:
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    return enhanced_image