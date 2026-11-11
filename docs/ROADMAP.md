# Roadmap phát triển

## Baseline trong repository
- [x] Danh mục 11 lớp và định dạng dữ liệu.
- [x] Pipeline làm sạch / chuẩn hóa / tách từ tiếng Việt.
- [x] Script huấn luyện NB, SVM, CNN, LSTM và báo cáo đánh giá.
- [x] FastAPI, SQLAlchemy, lưu phản ánh / prediction / moderation log.
- [x] Website demo gửi phản ánh và Admin dashboard cơ bản.
- [x] Docker Compose, test và CI.

## Việc nhóm nghiên cứu cần thực hiện
- [ ] Thu thập dữ liệu thật có quyền sử dụng.
- [ ] Hoàn thiện guideline gán nhãn và adjudication.
- [ ] Kiểm tra thủ công, phân tích trùng và mất cân bằng.
- [ ] Chạy thực nghiệm và ghi kết quả thật.
- [ ] Đánh giá calibration và ngưỡng confidence.
- [ ] Kiểm thử dữ liệu ngoài tập và đo latency mục tiêu.

## Mở rộng ứng dụng
- [ ] Xác thực và phân quyền Admin thật (JWT/OAuth, audit).
- [ ] Upload ảnh và object storage.
- [ ] Vietmap hoặc Leaflet: chọn vị trí và bản đồ sự cố.
- [ ] Heatmap theo vùng và bộ lọc thời gian.
- [ ] Chatbot giao tiếp.
- [ ] Human-in-the-loop, review queue và export.
- [ ] Dataset versioning, model registry, retraining có phê duyệt và rollback.

Không xem checkbox roadmap là tính năng đã nghiệm thu cho tới khi code, test và cấu hình dịch vụ tương ứng được bổ sung.
