# Smart Traffic & Urban Incident Classifier (ST-NLP)

Ứng dụng NLP tiếng Việt và học máy để phân loại phản ánh giao thông / hạ tầng đô thị thành 11 nhóm.

> **Trạng thái:** mã nguồn nền có thể chạy và mở rộng. Mô hình thực nghiệm chỉ có ý nghĩa sau khi nạp bộ dữ liệu thật đã được gán nhãn, kiểm tra và cấp quyền sử dụng. File ví dụ trong `data/samples/` là dữ liệu tổng hợp để minh họa định dạng, **không phải dữ liệu nghiên cứu và không dùng để báo cáo độ chính xác**.

## 1. Phạm vi phiên bản
- Pipeline tiền xử lý tiếng Việt: làm sạch HTML/URL, chuẩn hóa Unicode NFC, chuẩn hóa một số viết tắt, tách từ bằng Underthesea (ưu tiên) hoặc PyVi, lọc stopwords có giữ từ phủ định.
- TF-IDF cho Multinomial Naive Bayes / SVM; Tokenizer + Embedding cho 1D-CNN / LSTM.
- Huấn luyện bốn mô hình trên cùng cách chia Train / Validation / Test; xuất metrics, classification report, confusion matrix và artifacts.
- FastAPI: health, predict, tạo phản ánh, danh sách/chi tiết quản trị, kiểm duyệt và thống kê.
- SQLAlchemy: SQLite mặc định; MySQL qua biến môi trường / Docker Compose.
- Website demo tiếng Việt: gửi phản ánh, xem dự đoán, Admin xem danh sách và hiệu chỉnh nhãn bằng API key.

Các phần mở rộng như Vietmap/Leaflet, tải ảnh, chatbot, tài khoản quản trị đầy đủ, heatmap và quy trình tái huấn luyện có kiểm soát được ghi trong roadmap; không giả định là đã cấu hình dịch vụ bên ngoài.

## 2. Công nghệ
Python 3.11; Pandas; NumPy; Scikit-learn; TensorFlow/Keras; Underthesea / PyVi; FastAPI; SQLAlchemy; MySQL; HTML/CSS/JavaScript.

## 3. Cấu trúc chính
```text
src/stnlp/
  config.py, labels.py
  preprocessing/       # chuẩn hóa và tách từ tiếng Việt
  training/            # chia tập, huấn luyện 4 mô hình, đánh giá
  inference/           # nạp artifact và dự đoán
  database/            # SQLAlchemy, models, seed danh mục
  api/                  # FastAPI routes, schemas, dependencies
frontend/               # website demo
data/                    # raw, interim, processed, samples, templates
artifacts/               # mô hình đã huấn luyện (không commit mặc định)
tests/ docs/
```

## 4. Cài đặt local (Windows PowerShell)
```powershell
git clone https://github.com/khanh0102102/smart-traffic-urban-nlp.git
cd smart-traffic-urban-nlp
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```
Nếu PowerShell chặn kích hoạt, dùng Command Prompt với `.venv\Scripts\activate.bat`. Dùng Python 3.11 64-bit.

## 5. Chuẩn bị dữ liệu thật
Tạo CSV theo `data/templates/traffic_reports_template.csv` với hai cột:
- `text`: nội dung phản ánh tiếng Việt.
- `label`: mã `R01` ... `R11` (hoặc tên nhãn chuẩn trong `src/stnlp/labels.py`).

Mỗi dòng cần có nhãn được con người kiểm tra. Xem `docs/DATASET_GUIDE.md` để thu thập hợp pháp, gán nhãn, loại trùng và tránh rò rỉ dữ liệu. Mục tiêu dữ liệu cần quyết định từ thống kê và learning curve; không dùng số mẫu mẫu minh họa làm kết quả.

## 6. Huấn luyện bốn mô hình
```powershell
pip install -e .
python -m stnlp.training.train --data data/processed/traffic_reports.csv --artifacts-dir artifacts --epochs 12
```
Lệnh tạo artifact cho NB, SVM, CNN, LSTM và báo cáo ở `artifacts/evaluation/`. Chọn mô hình dựa trên Validation (ưu tiên Macro F1 và recall từng lớp); Test chỉ dùng cho đánh giá cuối.

## 7. Chạy API
```powershell
uvicorn stnlp.api.main:app --reload
```
Website: http://127.0.0.1:8000/ · Swagger: http://127.0.0.1:8000/docs · Health: http://127.0.0.1:8000/api/v1/health

Nếu chưa có artifact, health vẫn hoạt động nhưng dự đoán trả HTTP 503.

## 8. Cấu hình
Trong `.env`: `MODEL_NAME=svm` (hoặc naive_bayes/cnn/lstm), `MODEL_ARTIFACT_DIR=artifacts`, `DATABASE_URL=sqlite:///./stnlp.db`, `API_ADMIN_KEY=...`. API key là bảo vệ demo; không dùng secret mẫu cho triển khai công khai.

## 9. Docker Compose
```powershell
Copy-Item .env.example .env
# Sửa MYSQL_ROOT_PASSWORD, MYSQL_PASSWORD và API_ADMIN_KEY trong .env
docker compose up --build
```
Gắn thư mục artifacts đã huấn luyện; chưa có artifact thì chức năng dự đoán chưa sẵn sàng.

## 10. Kiểm thử
```powershell
pytest -q
```

## 11. Giới hạn
- Không commit dữ liệu cá nhân, ảnh, secret hoặc artifact lớn.
- API key demo chưa thay thế IAM/JWT production.
- Confidence không đồng nghĩa xác suất đúng đã được hiệu chỉnh; cần đánh giá calibration.
- Ngưỡng kiểm duyệt mặc định 0.60 theo mô tả đề tài và có thể cấu hình.
- Không tự động retrain từ nhãn Admin khi chưa kiểm tra chất lượng và đánh giá độc lập.

## 12. Tài liệu
Xem `docs/PROJECT_SCOPE.md`, `docs/DATASET_GUIDE.md`, `docs/API_CONTRACT.md`, `docs/MODEL_CARD_TEMPLATE.md`, `docs/ROADMAP.md`.
