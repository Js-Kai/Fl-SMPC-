"""
xem_du_lieu_server.py — Chạy hệ thống và in chi tiết dữ liệu server nhận được mỗi vòng.

Khác với chay.py (chỉ in 1 dòng tóm tắt/vòng), file này in ra:
  - từng gói (S, C, proof, Δw̃) mà mỗi client gửi lên server
  - kết quả qua từng cửa lọc (ZKP, validation) cho từng client
  - client nào được nhận / bị loại và lý do
  - trạng thái mô hình toàn cục trước và sau khi tổng hợp

Chạy:  python xem_du_lieu_server.py
"""

import numpy as np

from cau_hinh import CauHinh as cfg
from dataset import nap_va_chia_du_lieu
from client import Client
from client.mo_hinh import so_tham_so, huan_luyen_cuc_bo, do_chinh_xac
from server import Server
from nen_mat_ma import SchnorrZKP


def rut_gon(vec, n=4):
    """In gọn n phần tử đầu của một véc-tơ dài."""
    return "[" + ", ".join(f"{x:+.4f}" for x in vec[:n]) + ", ...]"


def main():
    print("=" * 78)
    print("XEM DỮ LIỆU TRÊN SERVER — từng gói client gửi lên, từng cửa lọc")
    print("=" * 78)

    rng = np.random.default_rng(cfg.SEED)

    du_lieu = nap_va_chia_du_lieu(cfg.SO_CLIENT, cfg.NON_IID, cfg.ALPHA, cfg.SEED)
    tap_val = du_lieu["validation"]
    tap_test = du_lieu["test"]
    so_lop = du_lieu["so_lop"]
    so_dac_trung = du_lieu["so_dac_trung"]

    cac_client = []
    for i, (X, y) in enumerate(du_lieu["clients"]):
        cac_client.append(Client(
            client_id=i, X=X, y=y, so_lop=so_lop,
            doc_hai=(i in cfg.CLIENT_DOC_HAI),
            kieu_tan_cong=cfg.KIEU_TAN_CONG,
        ))
    danh_sach_id = list(range(cfg.SO_CLIENT))

    mo_hinh = np.zeros(so_tham_so(so_dac_trung, so_lop))
    if cfg.DUNG_WARMUP:
        Xw, yw = du_lieu["warmup"]
        delta_warm = huan_luyen_cuc_bo(mo_hinh, Xw, yw, so_lop,
                                       cfg.WARMUP_EPOCH, cfg.BATCH, cfg.LEARNING_RATE, rng)
        mo_hinh = mo_hinh + delta_warm

    server = Server(mo_hinh, cfg.NGUONG_VALIDATION, cfg.QUORUM)

    print(f"\nSố client: {cfg.SO_CLIENT} | ngưỡng validation θ = {cfg.NGUONG_VALIDATION} "
          f"| quorum = {cfg.QUORUM}")
    print(f"Client độc hại thật sự (server KHÔNG được biết trước): {cfg.CLIENT_DOC_HAI}\n")

    SO_VONG_XEM = min(3, cfg.SO_VONG)   # chỉ in chi tiết N vòng đầu cho dễ đọc
    for vong in range(1, SO_VONG_XEM + 1):
        seed_vong = cfg.SEED * 10_000 + vong
        print("-" * 78)
        print(f"VÒNG {vong} — mô hình toàn cục hiện tại: norm = {np.linalg.norm(server.mo_hinh):.4f}, "
              f"{rut_gon(server.mo_hinh)}")

        cac_goi = []
        for c in cac_client:
            goi = c.chay_mot_vong(server.mo_hinh, tap_val, danh_sach_id, seed_vong, cfg, rng)
            cac_goi.append(goi)

            # ---- dữ liệu THỰC TẾ mà server nhận được từ client này ----
            print(f"\n  Gói từ client {goi['id']:2d}  (thực tế {'ĐỘC HẠI' if goi['doc_hai'] else 'sạch'}):")
            print(f"    điểm validation S      = {goi['diem']:.4f}")
            print(f"    commitment C           = {goi['commitment']}")
            print(f"    proof (T,s1,s2) size   = {SchnorrZKP.size_bytes()} bytes")
            print(f"    vân tay (m)            = {goi['van_tay']}")
            print(f"    Δw̃ đã che (4 phần tử) = {rut_gon(goi['delta_da_che'])}")

            # ---- kiểm tra qua từng cửa, in kết quả riêng từng cửa ----
            qua_zkp = SchnorrZKP.verify(goi["commitment"], goi["proof"])
            qua_diem = goi["diem"] >= cfg.NGUONG_VALIDATION
            print(f"    Cửa 1 (ZKP hợp lệ?)        -> {'ĐẠT' if qua_zkp else 'TRƯỢT'}")
            print(f"    Cửa 3 (điểm >= {cfg.NGUONG_VALIDATION})       -> "
                  f"{'ĐẠT' if qua_diem else 'TRƯỢT'}")

        ket_qua = server.tong_hop_mot_vong(cac_goi, learning_rate=1.0)

        print(f"\n  >> KẾT QUẢ LỌC CỦA SERVER VÒNG {vong}:")
        if ket_qua["thanh_cong"]:
            print(f"     nhận {ket_qua['so_nhan']}/{cfg.SO_CLIENT} client -> "
                  f"id được nhận: {ket_qua['id_duoc_nhan']}")
            print(f"     bị loại: {ket_qua['bi_loai']}")
            print(f"     mô hình sau tổng hợp: norm = {np.linalg.norm(server.mo_hinh):.4f}, "
                  f"{rut_gon(server.mo_hinh)}")
        else:
            print(f"     KHÔNG đủ quorum -> {ket_qua['ly_do']} -> bỏ vòng, giữ nguyên mô hình cũ")

        acc = do_chinh_xac(server.mo_hinh, *tap_test, so_lop)
        print(f"     test accuracy hiện tại: {acc:.1%}")

    if cfg.SO_VONG > SO_VONG_XEM:
        print("-" * 78)
        print(f"(chỉ in chi tiết {SO_VONG_XEM} vòng đầu; sửa SO_VONG_XEM trong file này "
              f"nếu muốn xem thêm — chạy python chay.py để xem tóm tắt đủ {cfg.SO_VONG} vòng)")


if __name__ == "__main__":
    main()
