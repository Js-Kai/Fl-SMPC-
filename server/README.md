# Thư mục `server/` — Phía Server

Phần này tương ứng với **"robust server-side aggregator"** ở nửa dưới Hình 3.

## File

| File | Nhiệm vụ |
|------|----------|
| `server.py` | Lớp `Server`: lọc client (3 cửa) + tổng hợp SMPC + cập nhật mô hình |

## Những gì server làm mỗi vòng

Xem phương thức `Server.tong_hop_mot_vong()`:

1. **Cửa 1 — Xác minh ZKP**: chứng minh Schnorr có hợp lệ với commitment không? Bắt kẻ bịa commitment mà không huấn luyện.
2. **Cửa 2 — Đối chiếu vân tay**: bản gửi đi có khớp commitment không? Bắt kẻ "niêm bản tốt, gửi bản xấu".
3. **Cửa 3 — Lọc validation**: điểm S có đạt ngưỡng θ không? Bắt kẻ đảo nhãn (điểm thấp).
4. **Kiểm quorum**: có đủ client hợp lệ để tổng hợp an toàn không?
5. **Tổng hợp SMPC**: cộng các bản đã che — nhiễu tự triệt tiêu, ra đúng tổng.
6. **Cập nhật** mô hình toàn cục.

## Lưu ý quan trọng về Cửa 2

Trong bản demo này, mỗi client che Δw bằng mặt nạ SMPC, nên "vân tay" của bản
**đã che** khác vân tay của bản **gốc** đã cam kết. Vì vậy Cửa 2 (đối chiếu vân
tay trực tiếp) không dùng được nguyên trạng khi có mask.

Đây chính là một **khoảng hở thật** trong thiết kế của bài báo: ZKP chứng minh
"biết cách mở commitment", nhưng không chứng minh "bản đã che chính là bản đã
cam kết". Muốn vá đúng cần ZKP chứng minh cả phép che (verifiable masking) —
bài báo không có. Trong bản demo, Cửa 1 và Cửa 3 gánh phần lọc chính.

## Muốn đổi cách tổng hợp?

Sửa `server.py`. Ví dụ muốn thử tổng hợp có trọng số theo lượng dữ liệu, hoặc
dùng ngưỡng thích nghi thay ngưỡng cố định — chỉnh trong `tong_hop_mot_vong()`.
