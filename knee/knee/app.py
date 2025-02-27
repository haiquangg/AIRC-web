from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from ultralytics import YOLO
from groq import Groq
import config
import base64
from PIL import Image
import io
from models import model_detection
import cv2

app = Flask(__name__)

# Thiết lập secret_key để sử dụng session an toàn
app.secret_key = 'ductrunglyn'

# Khởi tạo mô hình YOLO và client Groq (chỉ cần làm một lần)
model = YOLO(config.MODEL_YOLO)
client = Groq(api_key=config.CLIENT_CHAT)

# Reset lịch sử sau khi đăng xuất
@app.after_request
def add_header(response):
    """
    Thêm header HTTP để ngăn trình duyệt lưu cache, đảm bảo không thể quay lại trang sau khi đăng xuất.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Các trang chủ cơ bản
@app.route('/')
def home():
    return render_template('pages_base/index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Kiểm tra thông tin người dùng
        if username == 'admin' and password == '123':  # Ví dụ kiểm tra tài khoản
            session['user'] = username  # Lưu thông tin người dùng vào session
            flash('Đăng nhập thành công!', 'success')
            return redirect(url_for('menu'))
        
        flash('Đăng nhập thất bại. Vui lòng thử lại.', 'danger')
        return redirect(url_for('login'))
    
    return render_template('pages_base/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Logic xử lý đăng ký tài khoản
        return redirect(url_for('login'))  # Chuyển hướng đến trang đăng nhập sau khi đăng ký thành công

    return render_template('pages_base/register.html')

@app.route('/logout')
def logout():
    session.clear()  # Xóa toàn bộ session của người dùng
    flash('Bạn đã đăng xuất thành công.', 'info')
    return redirect(url_for('home'))

@app.route('/menu')
def menu():
    if 'user' not in session:  # Kiểm tra session
        flash('Vui lòng đăng nhập trước khi truy cập menu.', 'warning')
        return redirect(url_for('login'))
    return render_template('pages_base/menu.html')

# Trang giới thiệu bệnh
@app.route('/information')
def information():
    if 'user' not in session:  # Kiểm tra session
        flash('Vui lòng đăng nhập trước khi truy cập.', 'warning')
        return redirect(url_for('login'))
    return render_template('pages_base/information.html')

# Trang chẩn đoán bệnh
@app.route('/knee_detection')
def knee_detection():
    if 'user' not in session:  # Kiểm tra session
        flash('Vui lòng đăng nhập trước khi truy cập.', 'warning')
        return redirect(url_for('login'))
    return render_template('pages_ai/knee_detection.html')

# Trang hỗ trợ chẩn đoán
@app.route('/diagnose_knee', methods=['POST'])
def diagnose_knee():   
    # Lấy dữ liệu JSON từ request
    data = request.json
    if not data or 'cropped_image' not in data:
        return jsonify({"error": "No cropped_image provided"}), 400

    # Decode ảnh Base64
    try:
        img_data = base64.b64decode(data['cropped_image'].split(',')[1])
        image = Image.open(io.BytesIO(img_data))
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {str(e)}"}), 400

    # Xử lý ảnh qua hàm `knee_predict`
    try:
        prediction_result = model_detection.knee_predict(image)

        # Trích xuất kết quả từ hàm `knee_predict`
        expert_advice = prediction_result.get("expert_advice", "Không có lời khuyên.")
        expert_advice1 = prediction_result.get("expert_advice1", "Không có lời khuyên.")
        expert_advice2 = prediction_result.get("expert_advice2", "Không có lời khuyên.")
        annotated_image = prediction_result.get("annotated_image", None)

        # Encode ảnh kết quả về Base64 để gửi lại frontend
        if annotated_image is not None:
            annotated_pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            output_buffer = io.BytesIO()
            annotated_pil_image.save(output_buffer, format='JPEG')
            result_image_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        else:
            result_image_base64 = None

        # Trả kết quả
        return jsonify({
            "result_image": f"data:image/jpeg;base64,{result_image_base64}" if result_image_base64 else None,
            "advice": expert_advice,
            "advice1": expert_advice1,
            "advice2": expert_advice2,
        })

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

# Chạy ứng dụng Flask với host và port tùy chỉnh
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
