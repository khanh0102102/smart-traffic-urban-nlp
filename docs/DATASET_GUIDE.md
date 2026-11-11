# Hướng dẫn xây dựng Dataset

## Định dạng
CSV UTF-8 có tối thiểu `text` và `label`; label thuộc R01..R11. Khuyến nghị thêm source_id, collected_at, annotator, review_status, privacy_processed. Không đưa thông tin nhận dạng cá nhân không cần thiết lên repository.

## Thu thập
Chỉ thu thập nguồn có quyền truy cập và quyền sử dụng phù hợp; ghi nhận nguồn, ngày thu thập và điều kiện sử dụng. Không vượt qua đăng nhập, CAPTCHA, giới hạn kỹ thuật hoặc điều khoản sử dụng. Dữ liệu thật lưu riêng, không commit nếu chưa được phép.

## Gán nhãn
1. Viết định nghĩa bao gồm/loại trừ cho từng nhãn.
2. Tạo ví dụ dương tính, ví dụ gần ranh giới và quy tắc cho câu có nhiều sự cố.
3. Keyword chỉ dùng gợi ý, không tự động coi là ground truth.
4. Hai người gán nhãn độc lập một phần dữ liệu; thảo luận bất đồng và lưu guideline phiên bản.
5. R11 chỉ dùng khi không thuộc 10 nhóm còn lại; không dùng làm lớp chứa mọi trường hợp khó.

## Chất lượng và loại trùng
- Chuẩn hóa Unicode để phát hiện trùng; kiểm tra trùng trước khi chia tập.
- Không gộp các báo cáo khác nhau chỉ vì cùng một câu mẫu.
- Loại dòng rỗng, sai định dạng, spam và bản sao; lưu số lượng và lý do loại.
- Kiểm tra lớp thiếu mẫu và nhãn sai bằng con người.

## Chia tập và mất cân bằng
- Chia Train/Validation/Test phân tầng (mặc định 70/15/15).
- Chia sau khi loại trùng và trước mọi oversampling.
- Chỉ resample / class weight trên Train; không resample Validation/Test.
- Nếu cùng sự cố được đăng lại hoặc cùng nguồn có nhiều bản gần trùng, cân nhắc group split.
- Kiểm tra số mẫu từng lớp; cần đủ mẫu để stratify. Dùng learning curve để quyết định mức dữ liệu cần thiết.

## Tệp mẫu
`data/templates/traffic_reports_template.csv` là header trống. `data/samples/synthetic_examples.csv` là dữ liệu tổng hợp minh họa schema, không dùng cho kết luận nghiên cứu.
