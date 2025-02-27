from flask import Flask, request, jsonify, render_template, redirect, send_from_directory
import cv2
import numpy as np
import torch
import time
import os
from ultralytics import YOLO
from keras.models import load_model

# Khởi tạo ứng dụng Flask và cấu hình thư mục lưu trữ
app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = "./Data/input/"
RESULT_FOLDER = "./Data/output/"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Load model YOLO và model ConvLSTM
model_Yolo = YOLO("./weight_model/best.pt")
model_ConvLSTM = load_model("./weight_model/best_model_ConvLSTM_v3.hdf5")


# =======================
# Xoá toàn bộ file trong thư mục
# =======================
def clear_folder(folder_path):
    """Xoá tất cả file trong folder_path."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


# =======================
# Các route giao diện
# =======================
@app.route("/")
def index():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username == "admin" and password == "admin123":
        return redirect("/index")
    else:
        return render_template("login.html", error="Tên người dùng hoặc mật khẩu không đúng")


@app.route("/index")
def secure_index():
    return render_template("index.html")


@app.route("/intro")
def intro():
    return render_template("intro.html")


@app.route("/tool")
def tool():
    return render_template("tool.html")


# =======================
# Hàm xử lý ảnh và segmentation
# =======================
def preprocess_masks(masks):
    masks = np.array(masks, dtype=np.float32) / 255.0
    return np.expand_dims(masks, axis=(0, -1))


def get_segmentation_mask(img):
    results = model_Yolo.predict(source=img, save=False, save_txt=False)
    mask_np = None

    if results and len(results) > 0:
        result = results[0]
        if result.masks is not None:
            boxes = result.boxes.data
            clss = boxes[:, 5]
            if clss.numel() > 0:
                people_indices = torch.where(clss == 0)
                if people_indices[0].numel() > 0:
                    people_masks = result.masks.data[people_indices]
                    people_mask = torch.any(people_masks, dim=0).int() * 255
                    mask_np = people_mask.cpu().numpy()

    # Nếu mask_np là None hoặc không hợp lệ, tạo mask toàn 0 dựa trên kích thước ảnh gốc
    if (mask_np is None or mask_np.size == 0 or
        (mask_np.ndim >= 2 and (mask_np.shape[0] == 0 or mask_np.shape[1] == 0))):
        mask_np = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # Cố gắng resize mask; nếu có lỗi, tạo mask 0 có kích thước 320x320
    try:
        mask_resized = cv2.resize(mask_np, (320, 320), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        print("Error in cv2.resize:", e)
        mask_resized = np.zeros((320, 320), dtype=np.uint8)
    return mask_resized


def predict_next_mask(segmentation_masks):
    """
    Dự đoán ảnh mask tiếp theo bằng ConvLSTM, đầu ra được scale về [0,255] và trả về dạng uint8.
    """
    preprocessed_masks = preprocess_masks(segmentation_masks)
    predicted = model_ConvLSTM.predict(preprocessed_masks)[0]
    predicted = np.squeeze(predicted)  # Giả sử kết quả có shape (320, 320)
    predicted = (predicted * 255).clip(0, 255).astype(np.uint8)
    predicted = np.ascontiguousarray(predicted)
    return predicted


# =======================
# Route xử lý segmentation và dự đoán
# =======================
@app.route("/segment", methods=["POST"])
def segment_image():
    # Mỗi lần thực hiện, xoá sạch nội dung trong thư mục UPLOAD_FOLDER và RESULT_FOLDER
    clear_folder(UPLOAD_FOLDER)
    clear_folder(RESULT_FOLDER)

    files = request.files.getlist("image")
    if len(files) < 1 or len(files) > 20:
        return jsonify({"error": "Please upload between 1 and 20 images"}), 400

    original_filenames = []
    images = []
    segmentation_masks = []
    segmentation_filenames = []

    # Đọc và lưu ảnh gốc, thực hiện segmentation trên từng ảnh
    start_time = time.time()

    for idx, file in enumerate(files):
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Lưu ảnh gốc với tên đơn giản orig1.png, orig2.png, ...
        orig_filename = f"orig_{idx+1}.png"
        orig_filepath = os.path.join(app.config["UPLOAD_FOLDER"], orig_filename)
        cv2.imwrite(orig_filepath, img)
        original_filenames.append(orig_filename)
        images.append(img)

        # Thực hiện segmentation cho ảnh hiện tại
        mask = get_segmentation_mask(img)
        segmentation_masks.append(mask)

        # Lưu mask phân đoạn với tên đơn giản seg1.png, seg2.png, ...
        seg_filename = f"seg_{idx+1}.png"
        seg_filepath = os.path.join(app.config["RESULT_FOLDER"], seg_filename)
        cv2.imwrite(seg_filepath, mask)
        segmentation_filenames.append(seg_filename)

    if not images:
        return jsonify({"error": "No valid image selected"}), 400

    # Dự đoán segmentation mask tiếp theo từ chuỗi các segmentation mask
    predicted_mask = predict_next_mask(segmentation_masks)

    # Lưu ảnh dự đoán (mask) với tên cố định predict.png
    pred_filename = "predict.png"
    pred_filepath = os.path.join(app.config["RESULT_FOLDER"], pred_filename)
    cv2.imwrite(pred_filepath, predicted_mask)

    # Tính diện tích dựa trên ảnh dự đoán
    sum_pixel = cv2.countNonZero(predicted_mask)
    area = "{:,}".format(int(sum_pixel * 10 * 10))  # Quy đổi tuỳ ý

    execution_time = round(time.time() - start_time, 2)

    return jsonify({
        "original_images": original_filenames,
        "segmented_images": segmentation_filenames,
        "predicted_image": pred_filename,
        "area": area,
        "execution_time": execution_time,
    })
    


# =======================
# Các route truy xuất file
# =======================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/result")
def result():
    return render_template("result.html")


# =======================
# Chạy ứng dụng
# =======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
