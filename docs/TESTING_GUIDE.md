# Hướng dẫn kiểm thử ST-NLP

## Trước khi có dataset
1. Cài dependency theo README.
2. Chạy `pytest -q`.
3. Khởi động API và mở `/api/v1/health`; database phải ready. Nếu chưa huấn luyện, model_ready=false là trạng thái có chủ đích.
4. Mở `/docs`; kiểm tra schema các endpoint.
5. Gọi `POST /api/v1/predict` khi chưa có artifact và xác nhận API trả 503, không trả nhãn giả.
6. Kiểm thử upload bằng JPEG/PNG/WebP hợp lệ, tệp sai MIME và tệp vượt 5 MB.
7. Kiểm tra Admin endpoint trả 401 nếu thiếu/sai X-Admin-Key.

## Sau khi có dataset thật
1. Kiểm tra schema, nhãn và số lượng từng lớp.
2. Chạy training CLI; lưu nguyên bản training_summary.json và comparison.csv.
3. Kiểm tra confusion matrix và classification report của từng mô hình.
4. Chọn MODEL_NAME theo Validation; khởi động lại API.
5. Dùng Postman gửi những câu chưa dùng trong train/validation/test.
6. Tạo phản ánh, kiểm tra record reports/predictions trong DB.
7. Gọi Admin moderation và xác nhận prediction ban đầu vẫn còn, moderation_logs có bản ghi.
8. Đo latency nhiều lần sau warm-up, ghi môi trường đo và p50/p95.

## Kiểm thử ngoài tập
Tạo external test riêng, có ground truth do người gán nhãn. Không dùng tập này để chọn model hoặc tinh chỉnh ngưỡng. Khi đã dùng để phát triển, phải tạo một external test mới nếu muốn báo cáo đánh giá độc lập.
