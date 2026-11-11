# Phạm vi và kiến trúc ST-NLP

## Mục tiêu
Phân loại phản ánh hiện trường bằng tiếng Việt thành 11 danh mục giao thông / hạ tầng đô thị. Trọng tâm nghiên cứu là pipeline NLP và so sánh NB, SVM, CNN, LSTM; API và Website là lớp minh họa ứng dụng.

## 11 nhãn chuẩn
R01 Đường/cầu/hạ tầng hư hỏng  
R02 Điện lực/chiếu sáng  
R03 Cống/hố ga/thoát nước  
R04 Vỉa hè/lòng đường  
R05 Cây xanh/vật cản  
R06 Biển báo  
R07 Tổ chức giao thông  
R08 Đèn tín hiệu  
R09 Rác thải/vệ sinh môi trường  
R10 Đỗ xe cản trở  
R11 Khác

Định nghĩa chính thức và quy tắc phân biệt trường hợp giao thoa phải được nhóm nghiên cứu thống nhất trong guideline gán nhãn trước khi gán nhãn hàng loạt.

## Luồng huấn luyện ngoại tuyến
Nguồn dữ liệu được phép → kiểm tra/ẩn danh → gán nhãn con người → loại trùng → chia Train/Validation/Test → preprocessing → (TF-IDF + NB/SVM) và (Tokenizer + Embedding + CNN/LSTM) → đánh giá → lưu artifacts và báo cáo.

## Luồng dự đoán trực tuyến
Người dùng gửi văn bản → FastAPI kiểm tra request → preprocessing giống lúc huấn luyện → mô hình đã nạp → nhãn/confidence → kiểm tra threshold → lưu phản ánh/prediction → trả response. Quản trị viên dùng endpoint được bảo vệ bằng API key demo để xem và kiểm duyệt.

## Ranh giới phiên bản hiện tại
Đã thiết kế baseline cho văn bản và tọa độ số. Tích hợp Vietmap/Leaflet, tải ảnh thật, chatbot, tài khoản/quyền nhiều vai trò, heatmap và workflow retraining có phê duyệt là các phần mở rộng cần cấu hình và kiểm thử riêng; không coi là đã hoàn thành chỉ vì có trường dữ liệu hoặc roadmap.

## Nguyên tắc nghiên cứu
- Tập Test độc lập, không resample và không dùng để chọn siêu tham số.
- Fit vectorizer/tokenizer trên Train; tuyệt đối không fit trên Test.
- Cùng một cách chia dữ liệu và thứ tự nhãn khi so sánh bốn mô hình.
- Lưu preprocessing, tokenizer/vectorizer, mapping nhãn, cấu hình và phiên bản dataset cùng model.
- Macro F1 và metric từng lớp quan trọng khi dữ liệu mất cân bằng.
- Ghi metric thực tế và giới hạn dữ liệu; không biến kết quả dự kiến thành kết quả thực nghiệm.
