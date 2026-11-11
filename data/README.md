# Data directory

- `raw/`: dữ liệu nguồn; giữ quyền truy cập hạn chế, không commit dữ liệu thật.
- `interim/`: dữ liệu trung gian.
- `processed/`: CSV đã làm sạch và được con người kiểm tra nhãn.
- `templates/`: mẫu định dạng.
- `samples/`: ví dụ tổng hợp phục vụ minh họa schema / smoke test.

Training CLI yêu cầu CSV có cột `text` và `label`; label thuộc R01..R11. Tham khảo `docs/DATASET_GUIDE.md`.
