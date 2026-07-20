"""
client/client.py — Lớp Client: toàn bộ luồng PHÍA CLIENT trong Hình 3.

Mỗi vòng, một client thực hiện đúng 6 bước (xem phương thức `chay_mot_vong`):

    Bước 1: huấn luyện cục bộ trên dữ liệu riêng      -> Δw
    Bước 2: (nếu độc hại) bóp méo bản cập nhật
    Bước 3: niêm phong Δw bằng Pedersen commitment     -> C
    Bước 4: chấm điểm mô hình trên tập validation chung -> S
    Bước 5: sinh chứng minh ZKP (Schnorr)              -> proof
    Bước 6: che Δw bằng SMPC                            -> Δw̃

Cuối cùng client gửi gói (S, C, proof, Δw̃) lên server.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nen_mat_ma import Pedersen, SchnorrZKP, vector_to_scalar
from client.mo_hinh import huan_luyen_cuc_bo, do_chinh_xac
from client.smpc import tao_mat_na


class Client:
    def __init__(self, client_id, X, y, so_lop, doc_hai=False, kieu_tan_cong="dao_nhan"):
        self.id = client_id
        self.X = X
        self.y = y
        self.so_lop = so_lop
        self.doc_hai = doc_hai
        self.kieu_tan_cong = kieu_tan_cong

    def chay_mot_vong(self, mo_hinh_toan_cuc, tap_validation, danh_sach_id,
                      seed_vong, cfg, rng):
        """Thực hiện đủ 6 bước phía client, trả về gói dữ liệu gửi lên server."""
        X_val, y_val = tap_validation

        # --- Bước 1: huấn luyện cục bộ -> Δw ---
        y_train = self.y
        if self.doc_hai and self.kieu_tan_cong == "dao_nhan":
            # tấn công đảo nhãn: lật nhãn (0<->9, 1<->8, ...) rồi mới huấn luyện
            y_train = (self.so_lop - 1 - self.y)
        delta = huan_luyen_cuc_bo(mo_hinh_toan_cuc, self.X, y_train, self.so_lop,
                                  cfg.SO_EPOCH, cfg.BATCH, cfg.LEARNING_RATE, rng)

        # --- Bước 4 (chấm điểm trên bản mô hình sẽ được niêm phong) ---
        mo_hinh_cuc_bo = mo_hinh_toan_cuc + delta
        diem = do_chinh_xac(mo_hinh_cuc_bo, X_val, y_val, self.so_lop)

        # --- Bước 3: Pedersen commitment (niêm phong Δw) ---
        m = vector_to_scalar(delta)             # nén Δw thành 1 "vân tay" số
        C, r = Pedersen.commit(m)               # C = m·G + r·H

        # --- Bước 5: sinh ZKP chứng minh biết cách mở C, không lộ Δw ---
        proof = SchnorrZKP.prove(m, r, C)

        # --- Bước 6: che Δw bằng SMPC ---
        mat_na = tao_mat_na(self.id, danh_sach_id, seed_vong, len(delta), cfg.DO_LON_NHIEU)
        delta_da_che = delta + mat_na

        # gói gửi lên server: (điểm, commitment, chứng minh, bản đã che, + vân tay m để đối chiếu)
        return {
            "id": self.id,
            "diem": diem,
            "commitment": C,
            "proof": proof,
            "delta_da_che": delta_da_che,
            "van_tay": m,               # server dùng để kiểm tra tính nhất quán
            "doc_hai": self.doc_hai,    # chỉ dùng để thống kê, server KHÔNG được nhìn khi lọc
        }
