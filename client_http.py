"""
client_http.py — Client THẬT chạy qua HTTP, dùng trên MÁY/MÁY ẢO riêng biệt.

Cài đặt (trên mỗi máy client):
    pip install requests numpy

Chạy (thay <client_id> = 0..4 và <dia_chi_server> = IP máy server):
    python client_http.py <client_id> <dia_chi_server>

Ví dụ (client số 2, server ở 192.168.1.10 cổng 5000):
    python client_http.py 2 http://192.168.1.10:5000

Máy này KHÔNG cần dataset/ hay server/ — chỉ cần client/, nen_mat_ma.py,
giao_tiep.py và file này. Dữ liệu riêng + tập validation được lấy từ server
lúc khởi động (mô phỏng: dữ liệu vốn đã có sẵn cục bộ ở máy client).
"""

import sys

import numpy as np
import requests

from client import Client
from giao_tiep import goi_to_json, mo_hinh_from_json


class _Cfg:
    """Gói lại vài tham số huấn luyện lấy từ server thành object giống cau_hinh.CauHinh,
    vì Client.chay_mot_vong(...) cần cfg.SO_EPOCH, cfg.BATCH, cfg.LEARNING_RATE, cfg.DO_LON_NHIEU."""

    def __init__(self, cfg_json):
        self.SO_EPOCH = cfg_json["so_epoch"]
        self.BATCH = cfg_json["batch"]
        self.LEARNING_RATE = cfg_json["learning_rate"]
        self.DO_LON_NHIEU = cfg_json["do_lon_nhieu"]


def main():
    if len(sys.argv) < 3:
        print("Dùng: python client_http.py <client_id> <dia_chi_server>")
        print("Ví dụ: python client_http.py 0 http://192.168.1.10:5000")
        sys.exit(1)

    client_id = int(sys.argv[1])
    dia_chi = sys.argv[2].rstrip("/")

    # --- Lấy cấu hình + dữ liệu RIÊNG của mình + tập validation CHUNG ---
    cfg_json = requests.get(f"{dia_chi}/cau-hinh", timeout=30).json()
    if not (0 <= client_id < cfg_json["so_client"]):
        print(f"client_id phải trong khoảng 0..{cfg_json['so_client'] - 1}")
        sys.exit(1)

    du_lieu = requests.get(f"{dia_chi}/du-lieu/{client_id}", timeout=30).json()
    X = np.array(du_lieu["X"], dtype=np.float64)
    y = np.array(du_lieu["y"], dtype=np.int64)

    validation = requests.get(f"{dia_chi}/du-lieu-validation", timeout=30).json()
    X_val = np.array(validation["X"], dtype=np.float64)
    y_val = np.array(validation["y"], dtype=np.int64)

    doc_hai = client_id in cfg_json["client_doc_hai"]
    client = Client(client_id=client_id, X=X, y=y, so_lop=cfg_json["so_lop"],
                    doc_hai=doc_hai, kieu_tan_cong=cfg_json["kieu_tan_cong"])

    danh_sach_id = list(range(cfg_json["so_client"]))
    rng = np.random.default_rng(cfg_json["seed"] * 1009 + client_id)
    cfg_train = _Cfg(cfg_json)

    print(f"Client {client_id} ({'ĐỘC HẠI' if doc_hai else 'sạch'}) đã kết nối tới {dia_chi} "
          f"— {len(y)} mẫu dữ liệu riêng")

    trang_thai = requests.get(f"{dia_chi}/mo-hinh", timeout=30).json()

    while not trang_thai.get("ket_thuc", False):
        vong = trang_thai["vong"]
        mo_hinh = mo_hinh_from_json(trang_thai["mo_hinh"])
        seed_vong = cfg_json["seed"] * 10_000 + vong

        goi = client.chay_mot_vong(mo_hinh, (X_val, y_val), danh_sach_id,
                                   seed_vong, cfg_train, rng)
        print(f"[Client {client_id}] vòng {vong}: điểm validation = {goi['diem']:.4f} "
              f"-> đã gửi lên server, đang chờ các client khác...")

        dap_an = requests.post(
            f"{dia_chi}/goi",
            json={"vong": vong, "goi": goi_to_json(goi)},
            timeout=200,
        ).json()

        if dap_an.get("ket_qua_vong_truoc"):
            kq = dap_an["ket_qua_vong_truoc"]
            if kq["vong"] == vong:
                print(f"[Client {client_id}] server đã tổng hợp vòng {vong} "
                      f"-> test acc = {kq.get('test_acc', 0):.1%}")

        trang_thai = dap_an

    print(f"\n[Client {client_id}] HỆ THỐNG ĐÃ CHẠY XONG toàn bộ {cfg_json['so_vong']} vòng.")


if __name__ == "__main__":
    main()
