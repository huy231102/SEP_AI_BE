import asyncio
import base64
import cv2
import numpy as np
import pickle
import mediapipe as mp
import os
import ssl
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import Counter
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# --- Cấu hình ---
# Nên thay bằng địa chỉ của frontend khi triển khai thực tế
CORS_ALLOWED_ORIGINS = ["*"]
HOST = "0.0.0.0"
PORT = 8001
# Ngưỡng tin cậy tối thiểu để một dự đoán được xem xét
CONFIDENCE_THRESHOLD = 0.35
# Số lượng dự đoán gần nhất trong lịch sử để xem xét tính ổn định
PREDICTION_HISTORY_SIZE = 5
# Số lần một ký tự phải xuất hiện trong lịch sử để được coi là "ổn định"
STABILITY_THRESHOLD = 3


# Cấu hình CORS để cho phép client từ React kết nối vào
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Xác định đường dẫn tuyệt đối ---
# Lấy đường dẫn của thư mục chứa file script này
script_dir = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn đến thư mục chứa model và dữ liệu training
model_data_path = os.path.join(script_dir, 'model_data')

model_path = os.path.join(model_data_path, 'svm_model.pkl')
training_data_path = os.path.join(model_data_path, 'training_data')


class ConnectionManager:
    """Quản lý các kết nối WebSocket."""
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- Tải mô hình AI và các thành phần cần thiết ---
try:
    # Tải mô hình SVM đã được huấn luyện
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Lấy danh sách các nhãn (tên các ký tự) từ thư mục dữ liệu huấn luyện
    labels = sorted([d for d in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, d))])

    # Khởi tạo MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Dùng static_image_mode=False để detector chạy 1 lần rồi tracking → nhanh hơn
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
except Exception as e:
    print(f"Lỗi khi khởi tạo AI: {e}")
    model = None
    labels = []
    hands = None

def eulidean_distance(landmarkA, landmarkB):
    """Tính khoảng cách Euclidean giữa hai điểm mốc."""
    A = np.array([landmarkA.x, landmarkA.y])
    B = np.array([landmarkB.x, landmarkB.y])
    distance = np.linalg.norm(A-B)
    return distance

def extract_features(landmarks):
    """Trích xuất vector đặc trưng từ các điểm mốc bàn tay."""
    data = []
    # Logic này được sao chép từ notebook svm_training.ipynb
    # Đảm bảo thứ tự các khoảng cách là giống hệt
    data.append(eulidean_distance(landmarks[4], landmarks[0]))
    data.append(eulidean_distance(landmarks[8], landmarks[0]))
    data.append(eulidean_distance(landmarks[12], landmarks[0]))
    data.append(eulidean_distance(landmarks[16], landmarks[0]))
    data.append(eulidean_distance(landmarks[20], landmarks[0]))
    data.append(eulidean_distance(landmarks[4], landmarks[8]))
    data.append(eulidean_distance(landmarks[4], landmarks[12]))
    data.append(eulidean_distance(landmarks[8], landmarks[12]))
    data.append(eulidean_distance(landmarks[12], landmarks[16]))
    data.append(eulidean_distance(landmarks[20], landmarks[16]))
    data.append(eulidean_distance(landmarks[8], landmarks[16]))
    data.append(eulidean_distance(landmarks[8], landmarks[20]))
    data.append(eulidean_distance(landmarks[12], landmarks[20]))
    data.append(eulidean_distance(landmarks[4], landmarks[16]))
    data.append(eulidean_distance(landmarks[4], landmarks[20]))
    data.append(eulidean_distance(landmarks[5], landmarks[9]))
    
    return np.array(data)

def predict_sign_language(image_np, hands_instance=None):
    """
    Hàm xử lý ảnh và dịch ngôn ngữ ký hiệu.
    Trả về một tuple (ký tự dự đoán, độ tin cậy) nếu hợp lệ,
    ngược lại trả về None.
    """
    hands_to_use = hands_instance or hands
    if model is None or hands_to_use is None or not labels:
        print("Lỗi: Mô hình AI chưa được tải.")
        return None

    # Chuyển ảnh sang RGB vì MediaPipe yêu cầu
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    results = hands_to_use.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Trích xuất đặc trưng
            features = extract_features(hand_landmarks.landmark)
            
            # Đưa ra dự đoán
            prediction_idx = model.predict([features])[0]
            predicted_char = labels[prediction_idx]
            
            # Lấy xác suất của dự đoán
            prediction_proba = model.predict_proba([features])[0]
            confidence = np.max(prediction_proba)

            # In thông tin gỡ lỗi ra terminal
            print(f"Phát hiện: {predicted_char}, Độ tin cậy: {confidence:.2f}")

            # Chỉ trả về kết quả nếu độ tin cậy cao
            if confidence > CONFIDENCE_THRESHOLD:
                return predicted_char, confidence
    else:
        # Thêm log khi không phát hiện thấy tay
        print("Không phát hiện thấy tay trong khung hình.")
    
    return None # Không phát hiện thấy tay hoặc không đủ tự tin

# ------------------------------------

@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Khởi tạo Hands riêng cho từng client
    hands_instance = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    prediction_history = []
    last_sent_char = None
    import time
    MIN_INTERVAL = 0.15  # giãn cách tối thiểu giữa 2 lần infer (≈6-7fps)
    last_infer = 0
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith('data:image/jpeg;base64,'):
                # Loại bỏ phần tiền tố "data:image/jpeg;base64,"
                base64_data = data.split(',')[1]
                
                # Giải mã base64 thành dữ liệu nhị phân
                image_bytes = base64.b64decode(base64_data)
                
                # Chuyển dữ liệu nhị phân thành mảng numpy
                np_arr = np.frombuffer(image_bytes, np.uint8)
                
                # Đọc ảnh từ mảng numpy bằng OpenCV
                img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Giới hạn tốc độ suy luận
                now = time.time()
                if now - last_infer < MIN_INTERVAL:
                    continue
                last_infer = now

                # --- Gọi hàm xử lý AI ---
                prediction = predict_sign_language(img_np, hands_instance)
                # -------------------------

                if prediction:
                    predicted_char, confidence = prediction
                    prediction_history.append(predicted_char)

                    # Giới hạn kích thước của lịch sử dự đoán
                    if len(prediction_history) > PREDICTION_HISTORY_SIZE:
                        prediction_history.pop(0)

                    # Tìm ký tự xuất hiện nhiều nhất trong lịch sử
                    if prediction_history:
                        most_common_char = Counter(prediction_history).most_common(1)[0][0]
                        
                        # Kiểm tra xem ký tự đó có đủ "ổn định" không
                        if prediction_history.count(most_common_char) >= STABILITY_THRESHOLD:
                            # Chỉ gửi nếu ký tự ổn định khác với ký tự đã gửi lần cuối
                            if most_common_char != last_sent_char:
                                last_sent_char = most_common_char
                                response = f"{last_sent_char} ({int(confidence*100)}%)"
                                print(f"--- GỬI VỀ FRONTEND: '{response}' ---")
                                await websocket.send_text(response)
                else:
                    # Nếu không nhận diện được (tay biến mất), gửi tín hiệu xóa
                    # và reset trạng thái để có thể nhận diện ký tự mới ngay lập tức
                    if last_sent_char is not None:
                        print("--- GỬI VỀ FRONTEND: '' (Tín hiệu xóa) ---")
                        await websocket.send_text("")
                        prediction_history.clear()
                        last_sent_char = None

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)
    finally:
        # Đảm bảo giải phóng tài nguyên MediaPipe
        hands_instance.close()

if __name__ == "__main__":
    import uvicorn
    # Đọc đường dẫn file SSL từ biến môi trường hoặc default vào thư mục certs
    certfile = os.environ.get("SSL_CRT_FILE") or os.environ.get("SSL_CERT_PATH") or "certs/cert.pem"
    keyfile = os.environ.get("SSL_KEY_FILE") or os.environ.get("SSL_KEY_PATH") or "certs/key.pem"
    ssl_args = {}
    # Nếu cả hai file tồn tại thì tạo SSLContext và hạ SecLevel
    if os.path.exists(certfile) and os.path.exists(keyfile):
        print(f"Running with HTTPS using certs at {certfile} and {keyfile}")
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            ssl_certfile=certfile,
            ssl_keyfile=keyfile,
            ssl_ciphers="DEFAULT:@SECLEVEL=1",
        )
    else:
        print(f"Warning: SSL files not found at {certfile} and {keyfile}, running without SSL")
        uvicorn.run(app, host=HOST, port=PORT) 