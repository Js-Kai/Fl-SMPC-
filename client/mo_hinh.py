"""
client/mo_hinh.py — Mô hình + huấn luyện cục bộ.

Đây là bước ĐẦU TIÊN của client trong Hình 3: nhận mô hình toàn cục,
huấn luyện trên dữ liệu riêng, tạo ra bản cập nhật Δw.

GHI CHÚ: Bài báo dùng CNN. Ở đây dùng softmax tuyến tính (numpy) cho gọn và
chạy nhanh không cần GPU. Mọi cơ chế bảo mật đều thao tác trên véc-tơ tham số
phẳng Δw nên KHÔNG phụ thuộc vào kiến trúc mô hình — đổi sang CNN cũng được.
"""

import numpy as np


def so_tham_so(so_dac_trung, so_lop):
    """Tổng số tham số của mô hình = trọng số W (phẳng) + độ chệch b."""
    return so_dac_trung * so_lop + so_lop


def _tach(theta, d, k):
    """Tách véc-tơ tham số phẳng thành ma trận trọng số W và véc-tơ chệch b."""
    W = theta[:d * k].reshape(d, k)
    b = theta[d * k:]
    return W, b


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def du_doan(theta, X, so_lop):
    """Dự đoán nhãn cho dữ liệu X."""
    W, b = _tach(theta, X.shape[1], so_lop)
    return _softmax(X @ W + b).argmax(axis=1)


def do_chinh_xac(theta, X, y, so_lop):
    """Tính độ chính xác (tỉ lệ đoán đúng)."""
    if len(y) == 0:
        return 0.0
    return float((du_doan(theta, X, so_lop) == y).mean())


def huan_luyen_cuc_bo(theta_toan_cuc, X, y, so_lop,
                      so_epoch=5, batch=32, lr=0.1, rng=None):
    """Huấn luyện mô hình trên dữ liệu client bằng gradient descent.

    Trả về Δw = (mô hình sau huấn luyện) − (mô hình toàn cục nhận được).
    Đây chính là "bản cập nhật" mà client sẽ niêm phong và gửi đi.
    """
    if rng is None:
        rng = np.random.default_rng()
    d = X.shape[1]
    theta = theta_toan_cuc.copy()
    if len(y) == 0:
        return np.zeros_like(theta_toan_cuc)

    Y = np.eye(so_lop)[y]                       # nhãn dạng one-hot
    for _ in range(so_epoch):
        thu_tu = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            bi = thu_tu[i:i + batch]
            W, b = _tach(theta, d, so_lop)
            xac_suat = _softmax(X[bi] @ W + b)
            gW = X[bi].T @ (xac_suat - Y[bi]) / len(bi)
            gb = (xac_suat - Y[bi]).mean(axis=0)
            grad = np.concatenate([gW.ravel(), gb])
            theta -= lr * grad

    return theta - theta_toan_cuc
