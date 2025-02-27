from ultralytics import YOLO
import cv2
import numpy as np
from groq import Groq
import config
from models import processing_img, normalize_coordinates

# Hàm chính để chẩn đoán thoái hóa khớp gối
def knee_predict(image):
    model = YOLO(config.MODEL_YOLO)
    client = Groq(api_key=config.CLIENT_CHAT)

    if image is not None:
        processed_image = processing_img.preprocess_image(image)
        processed_width, processed_height = processed_image.shape[1], processed_image.shape[0]
        cropped_width, cropped_height = image.size

        try:
            processed_bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
            results = model.predict(processed_bgr_image, imgsz=640, conf=0.5)

            # Chuyển ảnh crop sang dạng numpy array để xử lý
            image_array = np.array(image)
            if len(image_array.shape) == 2 or image_array.shape[2] == 1:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

            # Bộ đếm cho các cấp độ bệnh
            level_count = {str(i): 0 for i in range(5)}

            # Kiểm tra kết quả và vẽ bounding boxes
            for result in results:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2, conf, cls = box
                        x1, y1, x2, y2 = normalize_coordinates.norm_coordinates((x1, y1, x2, y2),
                                                                                (processed_width, processed_height),
                                                                                (cropped_width, cropped_height))
                        level_count[str(int(cls))] += 1
                        color = (0, 255, 0)
                        cv2.rectangle(image_array, (x1, y1), (x2, y2), color, 2)
                        names = result.names
                        cv2.putText(image_array, names[int(cls)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

            # Tổng hợp kết quả
            if sum(level_count.values()) == 0:
                ket_qua_benh = "Không phát hiện vùng bệnh nào."
            else:
                ket_qua_benh = ", ".join([f"{count} vùng cấp độ {level}" for level, count in level_count.items() if count > 0])

            # Lời khuyên từ chuyên gia
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""Bạn là bác sĩ, chuyên gia chuẩn đoán bệnh thoái hóa khớp gối.
                                Bệnh có 5 mức độ là "0", "1", "2", "3", "4".
                                Hiện bệnh nhân đang có các vùng bệnh: {ket_qua_benh}.
                                Từ các vùng bệnh trên hãy đưa ra mức độ chung của bệnh nhân.
                                Hãy đưa ra lời khuyên dành cho bệnh nhân.
                                Trả lời 100% bằng Tiếng Việt""",
                    },
                    {"role": "user", "content": "Bệnh nhân cần tư vấn từ bác sĩ"},
                ],
                model="llama3-8b-8192",
            )

            expert_advice = chat_completion.choices[0].message.content

            # Trả về kết quả
            return {
                "expert_advice": expert_advice,
                "annotated_image": image_array
            }

        except Exception as e:
            raise RuntimeError(f"Lỗi khi dự đoán hoặc gọi API: {e}")
    else:
        raise ValueError("Không có ảnh đầu vào để chẩn đoán.")
