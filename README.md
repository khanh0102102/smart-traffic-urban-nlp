# Smart Traffic & Urban Incident Classifier (ST-NLP)

Ứng dụng NLP tiếng Việt và học máy để phân loại phản ánh giao thông / hạ tầng đô thị thành 11 nhóm.

> **Trạng thái:** mã nguồn nền có thể chạy và mở rộng. Mô hình thực nghiệm chỉ có ý nghĩa sau khi nạp bộ dữ liệu thật đã được gán nhãn, kiểm tra và cấp quyền sử dụng. File ví dụ trong `data/samples/` là dữ liệu tổng hợp để minh họa định dạng, **không phải dữ liệu nghiên cứu và không dùng để báo cáo độ chính xác**.

## 1. Phạm vi phiên bản
- Pipeline tiền xử lý tiếng Việt: làm sạch HTML/URL, chuẩn hóa Unicode NFC, chuẩn hóa một số viết tắt, tách từ bằng Underthesea (ưu tiên) hoặc PyVi, lọc stopwords có giữ từ phủ định.
- TF-IDF cho Multinomial Naive Bayes / SVM; Tokenizer + Embedding cho 1D-CNN / LSTM.
- Huấn luyện và đánh giá bốn mô hình trên cùng cách chia Train / Validation / Test; xuất metrics, classification report, confusion matrix và artifacts.
- FastAPI: health, predict, tạo phản ánh, danh sách/chi tiết quản trị, kiểm duyệt, thống kê và upload ảnh.
- SQLAlchemy: SQLite mặc định; MySQL qua biến môi trường / Docker Compose.
- Website demo tiếng Việt: gửi phản ánh, xem dự đoán, chọn tọa độ bằng Leaflet/GPS, đính kèm ảnh; Admin đăng nhập bằng mật khẩu đã băm và JWT để xem danh sách / chỉnh nhãn. API key chỉ được hỗ trợ như fallback khi phát triển local.

Bản đồ mặc định dùng Leaflet + OpenStreetMap nên cần Internet trong trình duyệt; có thể thay tile layer bằng Vietmap khi nhóm có API key. Tải ảnh hiện lưu local trong `uploads/`; khi triển khai thật nên chuyển sang object storage. Chatbot, tài khoản quản trị đầy đủ, heatmap và workflow tái huấn luyện có kiểm soát vẫn là phần mở rộng.

## 2. Công nghệ
Python 3.11; Pandas; NumPy; Scikit-learn; TensorFlow/Keras; Underthesea / PyVi; FastAPI; SQLAlchemy; MySQL; HTML/CSS/JavaScript; Leaflet.

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
uploads/                 # ảnh demo (không commit ảnh người dùng)
tests/ docs/ scripts/
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
pip install -e .
Copy-Item .env.example .env
```
Có thể chạy nhanh `Set-ExecutionPolicy -Scope Process Bypass` cho cửa sổ PowerShell hiện tại nếu bị chặn script kích hoạt. Dùng Python 3.11 64-bit.

## 5. Chuẩn bị dữ liệu thật
Tạo CSV theo `data/templates/traffic_reports_template.csv` với hai cột:
- `text`: nội dung phản ánh tiếng Việt.
- `label`: mã `R01` ... `R11` (hoặc tên nhãn chuẩn trong `src/stnlp/labels.py`).

Mỗi dòng cần có nhãn được con người kiểm tra. Xem `docs/DATASET_GUIDE.md` để thu thập hợp pháp, gán nhãn, loại trùng và tránh rò rỉ dữ liệu. Mục tiêu dữ liệu cần quyết định từ thống kê và learning curve; không dùng số mẫu minh họa làm kết quả.

## 5.1. Gợi ý nhãn bằng từ khóa để người gán nhãn kiểm tra

```powershell
python scripts/suggest_labels.py --input data/raw/collected_reports.csv --output data/interim/label_suggestions.csv
```

Công cụ chỉ tạo `suggested_label`, từ khóa khớp và trạng thái `pending`; tất cả dòng đều cần con người duyệt. Dòng không có từ khóa hoặc mơ hồ không tự động biến thành R11. Chỉ sau khi hoàn tất gán nhãn, đưa CSV có cột `text,label` đã được xác nhận vào `data/processed/`. Quy tắc hiện tại là baseline, cần rà soát theo dữ liệu thực tế và trao đổi với giảng viên.

## 6. Huấn luyện bốn mô hình
```powershell
python -m stnlp.training.train --data data/processed/traffic_reports.csv --artifacts-dir artifacts --epochs 12
```
Lệnh tạo artifact cho NB, SVM, CNN, LSTM và báo cáo ở `artifacts/evaluation/`: phân bố 11 lớp, metrics, classification report, Confusion Matrix, các câu dự đoán sai và các cặp nhãn nhầm lẫn. Chọn mô hình dựa trên Validation (ưu tiên Macro F1 và recall từng lớp); Test chỉ dùng cho đánh giá cuối. Sau khi huấn luyện, đặt `MODEL_NAME` trong `.env` theo mô hình được chọn và khởi động lại API.

## 7. Chạy API và Website
```powershell
uvicorn stnlp.api.main:app --reload
```
Website: http://127.0.0.1:8000/ · Swagger: http://127.0.0.1:8000/docs · Health: http://127.0.0.1:8000/api/v1/health

Nếu chưa có artifact, health vẫn hoạt động nhưng dự đoán trả HTTP 503. Leaflet/OpenStreetMap cần Internet; GPS cần quyền định vị của trình duyệt.

## 8. Cấu hình
Trong `.env`: `MODEL_NAME=svm` (hoặc naive_bayes/cnn/lstm), `MODEL_ARTIFACT_DIR=artifacts`, `DATABASE_URL=sqlite:///./stnlp.db`. Để tạo tài khoản Admin, chạy `python scripts/hash_password.py`, chép chuỗi hash vào `ADMIN_PASSWORD_HASH`, đặt `ADMIN_USERNAME` và tạo `JWT_SECRET_KEY` ngẫu nhiên bằng `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Khởi động lại API để seed tài khoản ban đầu. JWT mặc định hết hạn sau 120 phút. API key `X-Admin-Key` chỉ được chấp nhận khi `APP_ENV=development`; đặt `APP_ENV=production` khi triển khai và thay mọi secret mẫu.

## 9. Docker Compose
```powershell
Copy-Item .env.example .env
# Sửa MYSQL_ROOT_PASSWORD, MYSQL_PASSWORD, JWT_SECRET_KEY và ADMIN_PASSWORD_HASH trong .env
docker compose up --build
```
Compose chạy MySQL + API và mount `artifacts/`, `uploads/`; chưa có artifact thì dự đoán chưa sẵn sàng.

## 10. Kiểm thử
```powershell
pytest -q
```
Các script PowerShell: `scripts/setup.ps1`, `scripts/train.ps1 -Dataset <path>`, `scripts/run-api.ps1 -Reload`.

## 11. Giới hạn
- Không commit dữ liệu cá nhân, ảnh, secret hoặc artifact lớn.
- Tài khoản Admin dùng PBKDF2 password hash và JWT; cần HTTPS, secret riêng, rate-limit và quy trình đổi/khóa tài khoản trước production.
- Confidence không đồng nghĩa xác suất đúng đã được hiệu chỉnh; cần đánh giá calibration.
- Ngưỡng kiểm duyệt mặc định 0.60 theo mô tả đề tài và có thể cấu hình.
- Upload local cần thay bằng object storage, kiểm soát truy cập và quét nội dung khi production.
- Không tự động retrain từ nhãn Admin khi chưa kiểm tra chất lượng và đánh giá độc lập.

## 12. Tài liệu
Xem `docs/PROJECT_SCOPE.md`, `docs/DATASET_GUIDE.md`, `docs/LABELING_GUIDELINES.md`, `docs/API_CONTRACT.md`, `docs/MODEL_CARD_TEMPLATE.md`, `docs/TESTING_GUIDE.md`, `docs/TEAM_WORKFLOW.md`, `docs/ROADMAP.md`.
