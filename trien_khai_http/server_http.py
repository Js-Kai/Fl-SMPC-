"""
trien_khai_http/server_http.py — Server THẬT chạy qua HTTP (Flask), để nhiều máy
client thật kết nối vào.

Dùng khi có nhiều MÁY khác nhau (ví dụ 5 máy ảo đóng vai client, 1 máy thật
làm server) thay vì chạy mô phỏng trong 1 process như mo_phong/chay.py.

Server KHÔNG còn giữ dữ liệu riêng của client — chỉ giữ tập warm-up/test
(dùng nội bộ để khởi động + đánh giá mô hình) và tập validation CHUNG
(công khai, mọi client cùng thấy). Dữ liệu riêng của từng client nằm
trên máy client, đọc từ file .npz cục bộ (xem client_http.py).

Cài đặt (chỉ trên máy server):
    pip install flask numpy scikit-learn

Chạy (TỪ THƯ MỤC GỐC dự án — cần chuan_bi_du_lieu.py chạy trước để tạo .npz):
    python trien_khai_http/chuan_bi_du_lieu.py
    python trien_khai_http/server_http.py

Server lắng nghe ở 0.0.0.0:5000 — các máy client trỏ tới:
    http://<địa-chỉ-IP-máy-server>:5000

Luồng hoạt động mỗi vòng:
  1. Mỗi client GET /mo-hinh để biết vòng hiện tại + mô hình toàn cục.
  2. Mỗi client tự huấn luyện trên dữ liệu CỤC BỘ của nó (file .npz trên
     chính máy client, không tải từ server), tạo gói (S, C, proof, Δw̃),
     rồi POST /goi.
  3. Request POST /goi sẽ CHỜ (blocking) cho tới khi đủ cfg.SO_CLIENT gói
     (hoặc hết TIMEOUT_VONG giây) — lúc đó server tự lọc + tổng hợp + cập
     nhật mô hình, rồi trả về mô hình mới cho MỌI client đang chờ.
  4. Client nhận mô hình mới, lặp lại cho vòng kế tiếp.
"""

import os
import sys
import threading

import numpy as np
from flask import Flask, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cau_hinh import CauHinh as cfg
from client.mo_hinh import do_chinh_xac, huan_luyen_cuc_bo, so_tham_so
from giao_tiep import goi_from_json, mo_hinh_to_json
from server import Server

CONG = 5000               # cổng lắng nghe
TIMEOUT_VONG = 120        # giây chờ tối đa 1 vòng trước khi tổng hợp với ai đã có mặt

app = Flask(__name__)


def _doi_hoi_file(ten_file):
    if not os.path.exists(ten_file):
        raise SystemExit(
            f"Thiếu file {ten_file} — chạy `python trien_khai_http/chuan_bi_du_lieu.py` một lần trước "
            f"để tạo dữ liệu warm-up/test/validation cho server."
        )
    return np.load(ten_file)


# ---------------------------------------------------------------- khởi tạo
# Server CHỈ nạp warm-up + test (dùng nội bộ) và validation chung (công khai).
# KHÔNG nạp dữ liệu riêng của bất kỳ client nào — dữ liệu đó chỉ nằm trên máy client.
rng = np.random.default_rng(cfg.SEED)

_meta = _doi_hoi_file("du_lieu_server.npz")
X_warmup, y_warmup = _meta["X_warmup"], _meta["y_warmup"]
X_test, y_test = _meta["X_test"], _meta["y_test"]
so_lop = int(_meta["so_lop"])
so_dac_trung = int(_meta["so_dac_trung"])
tap_test = (X_test, y_test)

_val = _doi_hoi_file("du_lieu_validation.npz")
tap_val = (_val["X"], _val["y"])

mo_hinh_ban_dau = np.zeros(so_tham_so(so_dac_trung, so_lop))
if cfg.DUNG_WARMUP:
    delta_warm = huan_luyen_cuc_bo(mo_hinh_ban_dau, X_warmup, y_warmup, so_lop,
                                   cfg.WARMUP_EPOCH, cfg.BATCH, cfg.LEARNING_RATE, rng)
    mo_hinh_ban_dau = mo_hinh_ban_dau + delta_warm
    print(f"[Warm-up] acc ban đầu = {do_chinh_xac(mo_hinh_ban_dau, *tap_test, so_lop):.1%}")

server = Server(mo_hinh_ban_dau, cfg.NGUONG_VALIDATION, cfg.QUORUM)

# ---------------------------------------------------- trạng thái dùng chung
khoa = threading.Lock()
dieu_kien = threading.Condition(khoa)
vong_hien_tai = 1
goi_da_nhan = {}          # client_id -> gói, chỉ chứa gói của VÒNG HIỆN TẠI
ket_qua_vong_truoc = None
da_ket_thuc = False


def _tong_hop_va_sang_vong_moi(vong_vua_xong):
    """Lọc + tổng hợp các gói đã nhận, cập nhật mô hình, chuyển sang vòng kế tiếp.

    PHẢI được gọi trong lúc đang giữ `khoa` (không tự lock ở đây).
    """
    global vong_hien_tai, goi_da_nhan, ket_qua_vong_truoc, da_ket_thuc

    cac_goi = list(goi_da_nhan.values())
    ket_qua = server.tong_hop_mot_vong(cac_goi, learning_rate=1.0)
    acc = do_chinh_xac(server.mo_hinh, *tap_test, so_lop)
    ket_qua_vong_truoc = {**ket_qua, "vong": vong_vua_xong, "test_acc": acc}

    if ket_qua["thanh_cong"]:
        print(f"Vòng {vong_vua_xong:2d} | nhận {ket_qua['so_nhan']}/{cfg.SO_CLIENT} client "
              f"| bị loại {[i for i, _ in ket_qua['bi_loai']]} | test acc = {acc:.1%}")
    else:
        print(f"Vòng {vong_vua_xong:2d} | {ket_qua['ly_do']} -> bỏ vòng")

    goi_da_nhan = {}
    vong_hien_tai = vong_vua_xong + 1
    if vong_hien_tai > cfg.SO_VONG:
        da_ket_thuc = True
        print(f"\nHOÀN TẤT {cfg.SO_VONG} vòng — test accuracy cuối = {acc:.1%}")
    dieu_kien.notify_all()


# --------------------------------------------------------------- endpoint
@app.get("/cau-hinh")
def lay_cau_hinh():
    """Client gọi 1 lần lúc khởi động để biết tham số + có phải client độc hại không."""
    return jsonify({
        "so_client": cfg.SO_CLIENT,
        "so_vong": cfg.SO_VONG,
        "so_epoch": cfg.SO_EPOCH,
        "batch": cfg.BATCH,
        "learning_rate": cfg.LEARNING_RATE,
        "do_lon_nhieu": cfg.DO_LON_NHIEU,
        "seed": cfg.SEED,
        "so_lop": so_lop,
        "so_dac_trung": so_dac_trung,
        "client_doc_hai": cfg.CLIENT_DOC_HAI,
        "kieu_tan_cong": cfg.KIEU_TAN_CONG,
    })


@app.get("/du-lieu-validation")
def lay_validation():
    """Tập validation CHUNG — công khai, mọi client đều thấy giống nhau."""
    X_val, y_val = tap_val
    return jsonify({"X": X_val.tolist(), "y": y_val.tolist()})


@app.get("/mo-hinh")
def lay_mo_hinh():
    with khoa:
        return jsonify({
            "vong": vong_hien_tai,
            "mo_hinh": mo_hinh_to_json(server.mo_hinh),
            "ket_thuc": da_ket_thuc,
        })


@app.post("/goi")
def nhan_goi():
    body = request.get_json()
    vong_client = body["vong"]
    goi = goi_from_json(body["goi"])

    with dieu_kien:
        if da_ket_thuc:
            return jsonify({"loi": "hệ thống đã chạy xong toàn bộ vòng",
                             "ket_thuc": True, "vong": vong_hien_tai}), 409

        if vong_client != vong_hien_tai:
            # client bị lệch vòng (VD gửi trễ) -> báo để client đồng bộ lại
            return jsonify({
                "loi": f"sai vòng — server đang ở vòng {vong_hien_tai}",
                "vong": vong_hien_tai,
                "mo_hinh": mo_hinh_to_json(server.mo_hinh),
                "ket_thuc": da_ket_thuc,
            }), 409

        goi_da_nhan[goi["id"]] = goi
        print(f"[Vòng {vong_hien_tai}] nhận gói từ client {goi['id']} "
              f"({len(goi_da_nhan)}/{cfg.SO_CLIENT})")

        if len(goi_da_nhan) >= cfg.SO_CLIENT:
            _tong_hop_va_sang_vong_moi(vong_client)
        else:
            da_sang_vong_moi = dieu_kien.wait_for(
                lambda: vong_hien_tai != vong_client or da_ket_thuc,
                timeout=TIMEOUT_VONG,
            )
            if not da_sang_vong_moi and vong_hien_tai == vong_client:
                # hết giờ chờ -> tổng hợp với những client đã có mặt (nếu đủ quorum)
                _tong_hop_va_sang_vong_moi(vong_client)

        return jsonify({
            "vong": vong_hien_tai,
            "mo_hinh": mo_hinh_to_json(server.mo_hinh),
            "ket_thuc": da_ket_thuc,
            "ket_qua_vong_truoc": ket_qua_vong_truoc,
        })


if __name__ == "__main__":
    print("=" * 70)
    print(f"SERVER HTTP — cổng {CONG} | {cfg.SO_CLIENT} client | {cfg.SO_VONG} vòng")
    print("Máy client trỏ tới: http://<địa-chỉ-IP-máy-này>:" + str(CONG))
    print("=" * 70)
    app.run(host="0.0.0.0", port=CONG, threaded=True)
