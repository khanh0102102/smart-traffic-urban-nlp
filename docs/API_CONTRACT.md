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

## Create report
`POST /reports`
```json
{
  "text": "Hố ga trước nhà bị mất nắp",
  "latitude": 16.0471,
  "longitude": 108.2068,
  "image_url": null
}
```
Lưu report + prediction và trả mã tiếp nhận cùng kết quả. Tọa độ phải hợp lệ. API hiện nhận image_url có sẵn; chưa triển khai upload binary.

## Admin
Các endpoint dưới `/admin` yêu cầu header `X-Admin-Key` khớp `API_ADMIN_KEY`.
- `GET /admin/reports?skip=0&limit=50&needs_review=true`
- `GET /admin/reports/{report_id}`
- `PATCH /admin/reports/{report_id}/moderation` với body `{"category_code":"R03","note":"..." }`
- `GET /admin/stats`

Prediction ban đầu không bị ghi đè; thay đổi được lưu moderation log, nhãn xác nhận lưu riêng trên report.

## HTTP errors
422 request không hợp lệ; 401 thiếu/sai admin key; 404 không tìm thấy; 503 chưa có model; 500 lỗi nội bộ (không trả stack trace).

API key chỉ bảo vệ demo, không phải IAM production. Triển khai công khai cần HTTPS, secret manager, rate-limit và xác thực/phân quyền phù hợp.
