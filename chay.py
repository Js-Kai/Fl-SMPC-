"""
chay.py — Bộ điều phối: ghép dataset + client + server thành vòng lặp Hình 3.

Chạy:  python chay.py

File này KHÔNG chứa logic của hệ thống. Nó chỉ:
  1. nạp dữ liệu   (gọi thư mục dataset/)
  2. tạo các client (gọi thư mục client/)
  3. tạo server     (gọi thư mục server/)
  4. lặp qua từng vòng, in ra luồng để bạn thấy Hình 3 chạy

Muốn đổi tham số -> sửa file cau_hinh.py, KHÔNG sửa file này.
"""

import numpy as np

from cau_hinh import CauHinh as cfg
from dataset import nap_va_chia_du_lieu
from client import Client
from client.mo_hinh import so_tham_so, huan_luyen_cuc_bo, do_chinh_xac
from server import Server


def main():
    print("=" * 70)
    print("FL-SMPC++  —  Tái dựng theo Hình 3 (Section 4)")
    print("=" * 70)

    rng = np.random.default_rng(cfg.SEED)

    # ---------- (1) NẠP DỮ LIỆU (thư mục dataset/) ----------
    du_lieu = nap_va_chia_du_lieu(cfg.SO_CLIENT, cfg.NON_IID, cfg.ALPHA, cfg.SEED)
    tap_val = du_lieu["validation"]
    tap_test = du_lieu["test"]
    so_lop = du_lieu["so_lop"]
    so_dac_trung = du_lieu["so_dac_trung"]

    print(f"\nDữ liệu: {cfg.SO_CLIENT} client | "
          f"{'non-IID' if cfg.NON_IID else 'IID'} | "
          f"validation chung {len(tap_val[1])} mẫu | test {len(tap_test[1])} mẫu")
    print(f"Client độc hại (đảo nhãn): id {cfg.CLIENT_DOC_HAI}\n")

    # ---------- (2) TẠO CÁC CLIENT (thư mục client/) ----------
    cac_client = []
    for i, (X, y) in enumerate(du_lieu["clients"]):
        cac_client.append(Client(
            client_id=i, X=X, y=y, so_lop=so_lop,
            doc_hai=(i in cfg.CLIENT_DOC_HAI),
            kieu_tan_cong=cfg.KIEU_TAN_CONG,
        ))
    danh_sach_id = list(range(cfg.SO_CLIENT))

    # ---------- Khởi tạo mô hình toàn cục ----------
    mo_hinh = np.zeros(so_tham_so(so_dac_trung, so_lop))

    # ---------- Warm-up (Section 4.1) ----------
    if cfg.DUNG_WARMUP:
        Xw, yw = du_lieu["warmup"]
        delta_warm = huan_luyen_cuc_bo(mo_hinh, Xw, yw, so_lop,
                                       cfg.WARMUP_EPOCH, cfg.BATCH, cfg.LEARNING_RATE, rng)
        mo_hinh = mo_hinh + delta_warm
        print(f"[Warm-up] khởi động mô hình trên {len(yw)} mẫu công khai "
              f"-> acc ban đầu = {do_chinh_xac(mo_hinh, *tap_test, so_lop):.1%}\n")

    # ---------- (3) TẠO SERVER (thư mục server/) ----------
    server = Server(mo_hinh, cfg.NGUONG_VALIDATION, cfg.QUORUM)

    # ---------- (4) VÒNG LẶP HUẤN LUYỆN ----------
    print("-" * 70)
    for vong in range(1, cfg.SO_VONG + 1):
        seed_vong = cfg.SEED * 10_000 + vong

        # --- PHÍA CLIENT: mỗi client chạy 6 bước, tạo gói gửi lên ---
        cac_goi = []
        for c in cac_client:
            goi = c.chay_mot_vong(server.mo_hinh, tap_val, danh_sach_id,
                                  seed_vong, cfg, rng)
            cac_goi.append(goi)

        # --- PHÍA SERVER: lọc + tổng hợp + cập nhật ---
        ket_qua = server.tong_hop_mot_vong(cac_goi, learning_rate=1.0)

        # --- In luồng để thấy Hình 3 chạy ---
        acc = do_chinh_xac(server.mo_hinh, *tap_test, so_lop)
        if ket_qua["thanh_cong"]:
            doc_hai_bi_loai = [i for i, _ in ket_qua["bi_loai"] if i in cfg.CLIENT_DOC_HAI]
            print(f"Vòng {vong:2d} | nhận {ket_qua['so_nhan']}/{cfg.SO_CLIENT} client "
                  f"| loại {len(doc_hai_bi_loai)}/{len(cfg.CLIENT_DOC_HAI)} kẻ độc hại "
                  f"| test acc = {acc:.1%}")
        else:
            print(f"Vòng {vong:2d} | {ket_qua['ly_do']} -> bỏ vòng")

    print("-" * 70)
    print(f"\nKẾT QUẢ CUỐI: test accuracy = {do_chinh_xac(server.mo_hinh, *tap_test, so_lop):.1%}")
    print("\nMở HUONG_DAN.html để xem giải thích cấu trúc và cách chỉnh sửa.")


if __name__ == "__main__":
    main()
