"""
dataset/du_lieu.py — Nạp dữ liệu và chia cho các client.

Đây là phần "SHARED VALIDATION PIPELINE" ở đáy Hình 3: nó tạo ra
  - dữ liệu riêng cho từng client (mỗi client giữ, không chia sẻ)
  - tập validation CHUNG (server dùng để chấm điểm client)
  - tập warm-up công khai (dùng khởi động mô hình, Section 4.1)

GHI CHÚ VỀ DỮ LIỆU
------------------
Bài báo dùng MNIST, CIFAR-10, FEMNIST, UCI Heart Disease. Ở đây dùng bộ
'digits' (ảnh chữ số 8x8) có sẵn trong scikit-learn để chạy được ngay,
không cần tải mạng. Muốn đổi sang dữ liệu thật, chỉ cần sửa hàm `_nap_goc()`.
"""

import numpy as np
from sklearn.datasets import load_digits


SO_LOP = 10          # digits có 10 lớp (chữ số 0-9)


def _nap_goc():
    """Nạp dữ liệu gốc. ĐỔI DỮ LIỆU THẬT Ở ĐÂY nếu muốn.

    Ví dụ đổi sang MNIST thật (cần torchvision):
        from torchvision.datasets import MNIST
        ...
    """
    X, y = load_digits(return_X_y=True)
    return X / 16.0, y            # chuẩn hoá pixel về [0, 1]


def _chia_non_iid(y, so_client, alpha, rng):
    """Chia không đồng đều bằng phân phối Dirichlet (giống bài báo, α=0.3).

    Mỗi client sẽ lệch về một vài lớp -> mô phỏng thực tế (bệnh viện A nhiều
    ca loại này, bệnh viện B nhiều ca loại khác).
    """
    phan = [[] for _ in range(so_client)]
    for lop in range(SO_LOP):
        idx = np.where(y == lop)[0]
        rng.shuffle(idx)
        ti_le = rng.dirichlet(np.repeat(alpha, so_client))
        cat = (np.cumsum(ti_le) * len(idx)).astype(int)[:-1]
        for i, manh in enumerate(np.split(idx, cat)):
            phan[i].extend(manh.tolist())
    return phan


def _chia_deu(y, so_client, rng):
    """Chia đều (IID): mỗi client nhận lượng dữ liệu và tỉ lệ lớp như nhau."""
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return [manh.tolist() for manh in np.array_split(idx, so_client)]


def nap_va_chia_du_lieu(so_client, non_iid=True, alpha=0.3, seed=42):
    """Hàm chính. Trả về một dict gồm mọi thứ hệ thống cần.

    Trả về:
        {
          'clients':  [(X_0,y_0), (X_1,y_1), ...]   # dữ liệu riêng mỗi client
          'validation': (X_val, y_val)              # tập chung để chấm điểm
          'warmup':     (X_warm, y_warm)            # tập công khai để khởi động
          'test':       (X_test, y_test)            # tập kiểm tra cuối
          'so_lop':     10
          'so_dac_trung': 64
        }
    """
    rng = np.random.default_rng(seed)
    X, y = _nap_goc()

    # trộn rồi cắt: 22% test, 12% validation, 6% warm-up, còn lại chia cho client
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    n = len(y)
    n_test = int(0.22 * n)
    n_val = int(0.12 * n)
    n_warm = int(0.06 * n)

    X_test, y_test = X[:n_test], y[:n_test]
    X_val, y_val = X[n_test:n_test + n_val], y[n_test:n_test + n_val]
    X_warm, y_warm = X[n_test + n_val:n_test + n_val + n_warm], y[n_test + n_val:n_test + n_val + n_warm]
    X_tr, y_tr = X[n_test + n_val + n_warm:], y[n_test + n_val + n_warm:]

    # chia phần huấn luyện cho các client
    if non_iid:
        phan = _chia_non_iid(y_tr, so_client, alpha, rng)
    else:
        phan = _chia_deu(y_tr, so_client, rng)

    clients = []
    for i in range(so_client):
        idx = np.array(phan[i], dtype=int)
        if len(idx) < 5:                       # đảm bảo client nào cũng có dữ liệu
            idx = rng.choice(len(y_tr), size=10, replace=False)
        clients.append((X_tr[idx].copy(), y_tr[idx].copy()))

    return {
        "clients": clients,
        "validation": (X_val, y_val),
        "warmup": (X_warm, y_warm),
        "test": (X_test, y_test),
        "so_lop": SO_LOP,
        "so_dac_trung": X.shape[1],
    }
