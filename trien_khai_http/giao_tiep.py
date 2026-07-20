"""
giao_tiep.py — Chuyển gói dữ liệu sang JSON và ngược lại để gửi qua mạng thật (HTTP).

Gói client gửi lên có 3 loại dữ liệu JSON không hiểu ngay được:
  - điểm trên đường cong EC (tuple 2 số nguyên rất lớn, hoặc None = điểm vô cực)
  - số nguyên lớn (van_tay, s1, s2 trong proof) — vượt quá độ chính xác an toàn
    của kiểu number trong JSON ở nhiều ngôn ngữ khác -> ta chuyển sang string
  - numpy array (Δw̃) -> list số thực

File này CHỈ làm nhiệm vụ chuyển đổi, không chứa logic mật mã hay huấn luyện.
"""

import numpy as np


def _diem_to_json(pt):
    if pt is None:
        return None
    return [str(pt[0]), str(pt[1])]


def _diem_from_json(data):
    if data is None:
        return None
    return (int(data[0]), int(data[1]))


def goi_to_json(goi):
    """Chuyển 1 gói (dict) mà Client.chay_mot_vong() trả về sang dạng JSON gửi được qua mạng."""
    T, s1, s2 = goi["proof"]
    return {
        "id": goi["id"],
        "diem": goi["diem"],
        "commitment": _diem_to_json(goi["commitment"]),
        "proof": [_diem_to_json(T), str(s1), str(s2)],
        "delta_da_che": np.asarray(goi["delta_da_che"]).tolist(),
        "van_tay": str(goi["van_tay"]),
        "doc_hai": goi["doc_hai"],
    }


def goi_from_json(data):
    """Chiều ngược lại: JSON nhận qua mạng -> dict gói giống hệt bản gốc phía client."""
    T, s1, s2 = data["proof"]
    return {
        "id": data["id"],
        "diem": data["diem"],
        "commitment": _diem_from_json(data["commitment"]),
        "proof": (_diem_from_json(T), int(s1), int(s2)),
        "delta_da_che": np.array(data["delta_da_che"], dtype=np.float64),
        "van_tay": int(data["van_tay"]),
        "doc_hai": data.get("doc_hai", False),
    }


def mo_hinh_to_json(mo_hinh):
    return np.asarray(mo_hinh).tolist()


def mo_hinh_from_json(data):
    return np.array(data, dtype=np.float64)
