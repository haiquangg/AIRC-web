# Hàm chuẩn hóa tọa độ
def norm_coordinates(box, original_size, cropped_size):
    x1, y1, x2, y2 = box
    scale_x = cropped_size[0] / original_size[0]
    scale_y = cropped_size[1] / original_size[1]
    return int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)