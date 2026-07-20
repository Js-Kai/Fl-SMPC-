# Thư mục `client/` — Phía Client

Phần này tương ứng với **"client-side module"** ở nửa trên Hình 3.

## File

| File | Nhiệm vụ | Bước trong Hình 3 |
|------|----------|-------------------|
| `mo_hinh.py` | Mô hình softmax + huấn luyện cục bộ → Δw | Bước 1 |
| `smpc.py` | Che bản cập nhật bằng mặt nạ nhiễu triệt tiêu | Bước 6 (SMPC mask) |
| `client.py` | Lớp `Client` ghép đủ 6 bước, tạo gói gửi server | Toàn bộ phía client |

## Sáu bước một client làm mỗi vòng

Xem phương thức `Client.chay_mot_vong()`:

1. **Huấn luyện cục bộ** trên dữ liệu riêng → Δw
2. **(Nếu độc hại)** bóp méo Δw (ví dụ đảo nhãn)
3. **Niêm phong** Δw bằng Pedersen commitment → C
4. **Chấm điểm** mô hình trên tập validation chung → S
5. **Sinh ZKP** (Schnorr) chứng minh biết cách mở C mà không lộ Δw
6. **Che** Δw bằng SMPC → Δw̃

Cuối cùng gửi gói `(S, C, ZKP, Δw̃)` lên server.

## Muốn đổi mô hình sang CNN?

Sửa `mo_hinh.py`. Chỉ cần giữ nguyên "giao diện": hàm `huan_luyen_cuc_bo(...)`
phải trả về một véc-tơ phẳng Δw. Phần mật mã và SMPC không cần đổi gì.

## Muốn thêm kiểu tấn công mới?

Sửa `client.py`, trong Bước 1-2 của `chay_mot_vong()`, thêm nhánh xử lý cho
`self.kieu_tan_cong` mới của bạn.
