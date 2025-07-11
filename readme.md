# SEP_AI_BE

Dự án backend cho SEP AI, cung cấp các API phục vụ mô hình trí tuệ nhân tạo.

## Mô tả
Repository này chứa mã nguồn cho máy chủ AI được viết bằng Python. Tệp chính `ai_server.py` khởi tạo và chạy server để phục vụ mô hình cũng như các endpoint liên quan.

## Yêu cầu hệ thống
- Python >= 3.10
- pip >= 20

## Cài đặt
```bash
# Tạo virtualenv (tuỳ chọn)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt các phụ thuộc
pip install -r requirements.txt
```

## Chạy ứng dụng
```bash
python ai_server.py
```
Mặc định server sẽ chạy ở `http://localhost:8000` (hoặc cổng được chỉ định trong mã).

## Cấu trúc thư mục
- `ai_server.py`: Tệp chính khởi chạy server.
- `model_data/`: Chứa các tệp mô hình đã huấn luyện.
- `requirements.txt`: Danh sách phụ thuộc Python.

## Đóng góp
Pull request được hoan nghênh. Vui lòng đảm bảo code tuân thủ PEP8 và đã được kiểm thử.

## Giấy phép
Phân phối theo giấy phép MIT. 