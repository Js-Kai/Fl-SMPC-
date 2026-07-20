"""
client/smpc.py — Che bản cập nhật bằng mặt nạ nhiễu triệt tiêu (SMPC).

Đây là bước SMPC MASK trong Hình 3. Ý tưởng (Section 4.4.1):

  Mỗi cặp client (i, j) dùng chung một "hạt giống" bí mật. Từ hạt giống đó,
  cả hai sinh ra CÙNG một véc-tơ nhiễu. Client nhỏ hơn CỘNG nó, client lớn hơn
  TRỪ nó. Khi server cộng tất cả các bản đã che, các nhiễu tự triệt tiêu nhau,
  nên tổng vẫn ĐÚNG — nhưng server không thấy được bản cập nhật gốc của ai.

GHI CHÚ: Trong hệ thống thật, hạt giống chung được tạo bằng trao khoá
Diffie-Hellman để server không biết. Ở bản demo một máy này, ta suy hạt giống
từ một seed chung của vòng — đủ để minh hoạ tính triệt tiêu.
"""

import numpy as np


def tao_mat_na(id_cua_toi, danh_sach_id, seed_vong, so_chieu, do_lon=0.01):
    """Sinh mặt nạ nhiễu cho client `id_cua_toi`.

    Điểm mấu chốt: mỗi client tự tính mặt nạ của MÌNH một cách độc lập, nhưng
    khi cộng tất cả lại thì bằng 0. Không cần ai điều phối.

    Trả về: một véc-tơ nhiễu cùng số chiều với bản cập nhật.
    """
    mat_na = np.zeros(so_chieu)
    for id_khac in danh_sach_id:
        if id_khac == id_cua_toi:
            continue
        # hạt giống chung cho cặp (nhỏ, lớn) -> cả hai bên tính ra giống nhau
        nho, lon = min(id_cua_toi, id_khac), max(id_cua_toi, id_khac)
        hat_giong = seed_vong * 1_000_003 + nho * 1009 + lon
        nhieu = np.random.default_rng(hat_giong).normal(0, do_lon, so_chieu)
        # client nhỏ hơn CỘNG, client lớn hơn TRỪ -> triệt tiêu khi ghép cặp
        mat_na += nhieu if id_cua_toi < id_khac else -nhieu
    return mat_na
