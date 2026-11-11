# Model Card — điền sau thực nghiệm

## Định danh
- Tên / phiên bản mô hình:
- Ngày huấn luyện:
- Git commit:
- Dataset version/hash:
- Người thực hiện:

## Mục đích và phạm vi
Phân loại phản ánh giao thông / đô thị tiếng Việt thành R01–R11. Không dùng để tự động ra quyết định xử phạt hoặc thay thế cán bộ tiếp nhận.

## Dữ liệu
- Nguồn và quyền sử dụng:
- Số mẫu tổng và từng lớp:
- Quy tắc gán nhãn:
- Quy trình ẩn danh:
- Cách loại trùng:
- Chia Train/Validation/Test:

## Tiền xử lý và mô hình
- Unicode/cleaning:
- Từ điển viết tắt:
- Word segmenter/version:
- Stopwords:
- Vectorizer hoặc tokenizer:
- Kiến trúc và hyperparameters:
- Cách xử lý mất cân bằng:

## Kết quả
- Accuracy:
- Macro Precision / Recall / F1:
- Weighted F1:
- Per-class precision/recall/F1:
- Confusion matrix:
- Calibration (nếu có):
- Latency và môi trường đo:

## Hạn chế, rủi ro, hướng dẫn sử dụng
Mô tả lớp ít dữ liệu, miền dữ liệu, lỗi chính tả/không dấu, phản ánh nhiều vấn đề, confidence chưa hiệu chỉnh và drift. Ghi rõ preprocessing, threshold, artifact, phiên bản code và cách rollback.
