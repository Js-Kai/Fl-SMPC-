"""
cau_hinh.py — Tất cả tham số của hệ thống nằm ở MỘT chỗ duy nhất.

Muốn thử nghiệm gì, chỉ cần sửa file này rồi chạy lại `chay.py`.
Không cần đụng tới code trong 3 thư mục dataset/ client/ server/.
"""


class CauHinh:
    # ---- Liên bang (federation) ----
    SO_CLIENT = 10               # tổng số client tham gia
    SO_VONG = 15                 # số vòng huấn luyện

    # ---- Client độc hại ----
    CLIENT_DOC_HAI = [0, 3, 7]   # id của các client sẽ tấn công (30%)
    KIEU_TAN_CONG = "dao_nhan"   # "dao_nhan" | "khong_tan_cong"

    # ---- Huấn luyện cục bộ ----
    SO_EPOCH = 5                 # số vòng lặp huấn luyện trên máy client
    BATCH = 32
    LEARNING_RATE = 0.1

    # ---- Cơ chế phòng thủ của server ----
    NGUONG_VALIDATION = 0.55     # θ: điểm dưới ngưỡng này thì bị loại
    QUORUM = 4                   # cần ít nhất bao nhiêu client hợp lệ mới tổng hợp

    # ---- Warm-up (Section 4.1) ----
    DUNG_WARMUP = True           # bật/tắt giai đoạn khởi động
    WARMUP_EPOCH = 40

    # ---- SMPC ----
    DO_LON_NHIEU = 0.01          # biên độ nhiễu che (mask)

    # ---- Dữ liệu ----
    NON_IID = True               # True = dữ liệu không đồng đều (Dirichlet)
    ALPHA = 0.3                  # độ lệch non-IID (càng nhỏ càng lệch)
    SEED = 42                    # hạt giống ngẫu nhiên (để tái lập kết quả)
