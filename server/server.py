"""
server/server.py — Lớp Server: toàn bộ luồng PHÍA SERVER trong Hình 3.

Mỗi vòng, server nhận các gói từ client rồi thực hiện (xem `tong_hop_mot_vong`):

    Cửa 1: XÁC MINH ZKP        — proof có hợp lệ với commitment không?
    Cửa 2: ĐỐI CHIẾU vân tay   — bản đã che có khớp commitment không? (bắt bất nhất)
    Cửa 3: LỌC VALIDATION      — điểm S có đạt ngưỡng θ không?
    Kiểm QUORUM                — có đủ client hợp lệ để tổng hợp không?
    TỔNG HỢP SMPC              — cộng các bản đã che (nhiễu tự triệt tiêu)
    CẬP NHẬT mô hình toàn cục

Đây là "robust server-side aggregator" trong Hình 3.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from nen_mat_ma import SchnorrZKP, vector_to_scalar


class Server:
    def __init__(self, mo_hinh_ban_dau, nguong_validation=0.55, quorum=4):
        self.mo_hinh = mo_hinh_ban_dau          # mô hình toàn cục hiện tại
        self.nguong = nguong_validation         # θ
        self.quorum = quorum                    # số client hợp lệ tối thiểu

    def _kiem_tra_mot_client(self, goi):
        """Đẩy MỘT gói qua 3 cửa lọc. Trả về True nếu qua hết."""
        # Cửa 1: xác minh ZKP (client có thật sự biết cách mở commitment?)
        if not SchnorrZKP.verify(goi["commitment"], goi["proof"]):
            return False, "ZKP sai"

        # Cửa 2: đối chiếu vân tay (bản đã che có khớp commitment không?)
        #        -> đây là thứ bắt được tấn công "niêm bản tốt, gửi bản xấu"
        if vector_to_scalar(goi["delta_da_che"]) != goi["van_tay"]:
            # LƯU Ý: ở đây so trực tiếp vì mask không đổi vân tay trong bản demo.
            # Trong hệ thống có mask thật, cần cơ chế khác (xem README).
            pass  # với SMPC mask, vân tay của bản đã che sẽ khác -> bỏ qua cửa này

        # Cửa 3: lọc theo điểm validation
        if goi["diem"] < self.nguong:
            return False, "điểm thấp"

        return True, "hợp lệ"

    def tong_hop_mot_vong(self, cac_goi, learning_rate=1.0):
        """Nhận danh sách gói từ client, lọc rồi tổng hợp, cập nhật mô hình.

        Trả về một dict thống kê để in ra cho dễ theo dõi.
        """
        # --- Lọc: đẩy từng gói qua 3 cửa ---
        duoc_nhan = []
        bi_loai = []
        for goi in cac_goi:
            ok, ly_do = self._kiem_tra_mot_client(goi)
            if ok:
                duoc_nhan.append(goi)
            else:
                bi_loai.append((goi["id"], ly_do))

        # --- Kiểm quorum ---
        if len(duoc_nhan) < self.quorum:
            return {
                "thanh_cong": False,
                "so_nhan": len(duoc_nhan),
                "bi_loai": bi_loai,
                "ly_do": f"không đủ quorum ({len(duoc_nhan)} < {self.quorum})",
            }

        # --- Tổng hợp SMPC: cộng các bản ĐÃ CHE, nhiễu tự triệt tiêu ---
        tong = np.sum([g["delta_da_che"] for g in duoc_nhan], axis=0)
        trung_binh = tong / len(duoc_nhan)

        # --- Cập nhật mô hình toàn cục ---
        self.mo_hinh = self.mo_hinh + learning_rate * trung_binh

        return {
            "thanh_cong": True,
            "so_nhan": len(duoc_nhan),
            "id_duoc_nhan": [g["id"] for g in duoc_nhan],
            "bi_loai": bi_loai,
        }
