# API Contract (v1)

Base URL: `/api/v1`

## Health
`GET /health` — không yêu cầu xác thực. Trả trạng thái API, database và việc artifact có được nạp hay không.

## Predict
`POST /predict`
```json
{
  "text": "Hố ga trước nhà bị mất nắp",
  "include_probabilities": true
}
```
Trả `predicted_category` (code/name), `confidence` trong [0,1], `needs_review`, `model_name`, `model_version` và tùy chọn xác suất từng lớp.

## Gợi ý nhãn khi chuẩn bị dữ liệu
Tiện ích offline `python scripts/suggest_labels.py --input <csv> --output <csv>` tạo gợi ý từ khóa cho annotator; không phải endpoint công khai và không được coi là nhãn ground truth.

## Upload ảnh
`POST /uploads/image` — multipart/form-data với field `file`. Chấp nhận JPEG, PNG, WebP tối đa 5 MB; endpoint kiểm tra MIME và magic bytes, trả `image_url` dạng `/media/{filename}`. Bản demo lưu local trong `uploads/`; cần object storage và kiểm soát truy cập khi triển khai thật.

## Create report
`POST /reports`
```json
{
  "text": "Hố ga trước nhà bị mất nắp",
  "latitude": 16.0471,
  "longitude": 108.2068,
  "image_url": "/media/example.jpg"
}
```
Lưu report + prediction và trả mã tiếp nhận cùng kết quả. Tọa độ phải hợp lệ và latitude/longitude phải được gửi cùng nhau. Frontend có thể lấy GPS hoặc cho người dùng click bản đồ Leaflet. Upload ảnh là tùy chọn.

## Admin
Các endpoint dưới `/admin` yêu cầu header `X-Admin-Key` khớp `API_ADMIN_KEY`.
- `GET /admin/reports?skip=0&limit=50&needs_review=true`
- `GET /admin/reports/{report_id}`
- `PATCH /admin/reports/{report_id}/moderation` với body `{"category_code":"R03","note":"..." }`
- `GET /admin/stats`

Prediction ban đầu không bị ghi đè; thay đổi được lưu moderation log, nhãn xác nhận lưu riêng trên report.

## HTTP errors
422 request không hợp lệ; 401 thiếu/sai admin key; 404 không tìm thấy; 413 ảnh quá lớn; 415 ảnh sai định dạng; 503 chưa có model; 500 lỗi nội bộ (không trả stack trace).

API key chỉ bảo vệ demo, không phải IAM production. Triển khai công khai cần HTTPS, secret manager, rate-limit và xác thực/phân quyền phù hợp.
