import numpy as np
import cv2
import base64

# Chuyển đổi hình ảnh OpenCV thành Base64
def convert_image_to_base64(image):
    # Đảm bảo hình ảnh là mảng NumPy
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Xử lý ảnh đơn kênh (grayscale) thành BGR
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Chuyển hình ảnh thành Base64
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64