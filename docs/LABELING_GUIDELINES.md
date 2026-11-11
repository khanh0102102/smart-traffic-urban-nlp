# Quy tắc gán nhãn 11 nhóm (bản khởi đầu cần nhóm nghiên cứu duyệt)

## Nguyên tắc chung
- Gán theo **sự cố chính** mà người dân yêu cầu xử lý, không chỉ theo một từ khóa xuất hiện.
- Một phản ánh có nhiều sự cố độc lập: nếu hệ thống chỉ hỗ trợ một nhãn, chọn sự cố được mô tả là nguy hiểm/chính/yêu cầu xử lý trọng tâm; ghi chú các sự cố phụ. Nếu không xác định được, đưa vào hàng đợi adjudication, không đoán.
- Không gán R11 chỉ vì câu mơ hồ, ngắn, sai chính tả hoặc thiếu từ khóa; phải có người xem xét.
- Gợi ý từ `scripts/suggest_labels.py` không phải ground truth và luôn phải được người kiểm duyệt xác nhận.
- Hai annotator nên gán độc lập một phần dữ liệu; bất đồng phải được thảo luận và ghi lại quyết định.

## Phạm vi từng nhãn

| Mã | Nhóm | Quy tắc bao gồm chính |
|---|---|---|
| R01 | Đường/cầu/hạ tầng hư hỏng | Hư hỏng kết cấu mặt đường, cầu và hạ tầng giao thông dạng đường/cầu như ổ gà, nứt, sụt lún |
| R02 | Điện lực/chiếu sáng | Hệ thống cấp điện, cột/dây điện và chiếu sáng đường phố; phân biệt với đèn tín hiệu giao thông |
| R03 | Cống/hố ga/thoát nước | Cống, hố ga, nắp cống, hệ thống thoát nước và vấn đề ngập liên quan thoát nước |
| R04 | Vỉa hè/lòng đường | Hư hỏng/chiếm dụng/sử dụng sai công năng vỉa hè hoặc lòng đường, trừ khi vấn đề chính là hành vi đỗ xe |
| R05 | Cây xanh/vật cản | Cây xanh, cành cây, vật cản không thuộc nhóm công trình/biển báo |
| R06 | Biển báo | Biển báo, biển chỉ dẫn, biển báo hiệu giao thông bị mất/hỏng/sai vị trí |
| R07 | Tổ chức giao thông | Phân luồng, phân làn, tổ chức/cấm đường, điều tiết và bất cập tổ chức giao thông |
| R08 | Đèn tín hiệu | Đèn tín hiệu giao thông tại nút giao, chu kỳ đèn và thiết bị tín hiệu |
| R09 | Rác thải/vệ sinh môi trường | Rác, vệ sinh công cộng, ô nhiễm và vệ sinh môi trường đô thị |
| R10 | Đỗ xe cản trở | Xe dừng/đỗ gây cản trở, chiếm lối đi hoặc đỗ sai quy định; ưu tiên nhãn này khi hành vi đỗ xe là sự cố chính |
| R11 | Khác | Phản ánh thuộc lĩnh vực đô thị/giao thông nhưng không thuộc R01–R10 sau khi đã được người có trách nhiệm xem xét |

## Các cặp cần quy tắc rõ
- R02 / R08: đèn chiếu sáng đường phố thuộc R02; đèn điều khiển giao thông thuộc R08.
- R04 / R10: nếu nguyên nhân chính là xe dừng/đỗ thì R10; nếu nội dung chính là hạ tầng hoặc chiếm dụng vỉa hè bởi hoạt động khác thì R04.
- R01 / R03: kết cấu mặt đường/cầu thuộc R01; cống, hố ga, thoát nước thuộc R03.
- R07 / R08: tổ chức/phân luồng thuộc R07; thiết bị đèn tín hiệu thuộc R08.

Đây là phiên bản khởi đầu dựa trên 11 danh mục của đề tài. Nhóm cần trao đổi với giảng viên và cập nhật ví dụ thực tế trước khi gán nhãn hàng loạt.
