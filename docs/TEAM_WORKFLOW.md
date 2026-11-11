# Quy trình làm việc nhóm với Git

## Nhánh
- `main`: phiên bản ổn định có thể demo.
- `develop`: nhánh tích hợp công việc.
- `feature/data-pipeline`, `feature/nlp-models`, `feature/backend-api`, `feature/frontend`: nhánh tính năng.

## Mỗi tính năng
1. Pull latest develop.
2. Tạo feature branch.
3. Commit nhỏ, mô tả rõ.
4. Chạy pytest và kiểm tra thủ công module.
5. Tạo Pull Request vào develop; mô tả thay đổi, cách test, ảnh chụp nếu có.
6. Chỉ merge khi CI xanh và không đưa dữ liệu/secret vào commit.

## Dữ liệu và artifacts
- Dataset thật và ảnh người dùng không commit vào repo nếu chưa có quyền.
- Ghi dataset version/hash, số lượng từng lớp và guideline gán nhãn trong báo cáo nghiên cứu.
- Artifacts được tạo từ training script; chia sẻ qua kho lưu trữ nội bộ có kiểm soát hoặc release phù hợp, không ghi đè model không có version.
- Không sửa tập Test sau khi đã dùng báo cáo kết quả.
