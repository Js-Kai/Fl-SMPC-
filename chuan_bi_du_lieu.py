"""
chuan_bi_du_lieu.py — Chạy MỘT LẦN để tách dữ liệu thành file .npz riêng cho từng máy.

Đây là bước "chuẩn bị hạ tầng" mô phỏng thực tế: mỗi client (VD mỗi bệnh viện)
vốn đã có sẵn dữ liệu riêng của họ trên máy của họ, không ai gửi cho ai.

Chạy (1 lần, ở đâu cũng được, miễn có sẵn dataset/):
    python chuan_bi_du_lieu.py

Sinh ra:
    du_lieu_server.npz       -> giữ lại ở máy SERVER (warm-up + test + metadata)
    du_lieu_validation.npz   -> giữ lại ở máy SERVER (tập validation chung)
    du_lieu_client_0.npz
    du_lieu_client_1.npz
    ...
    du_lieu_client_{N-1}.npz -> mỗi file COPY sang ĐÚNG 1 máy client tương ứng

Sau khi copy xong file của client nào sang máy đó, hãy XOÁ các file
du_lieu_client_*.npz khỏi máy server — đúng tinh thần "dữ liệu không rời
khỏi máy sở hữu nó", server chỉ nên giữ du_lieu_server.npz + du_lieu_validation.npz.
"""

import numpy as np

from cau_hinh import CauHinh as cfg
from dataset import nap_va_chia_du_lieu


def main():
    du_lieu = nap_va_chia_du_lieu(cfg.SO_CLIENT, cfg.NON_IID, cfg.ALPHA, cfg.SEED)

    Xw, yw = du_lieu["warmup"]
    Xt, yt = du_lieu["test"]
    np.savez("du_lieu_server.npz",
             X_warmup=Xw, y_warmup=yw, X_test=Xt, y_test=yt,
             so_lop=du_lieu["so_lop"], so_dac_trung=du_lieu["so_dac_trung"])
    print("Đã tạo du_lieu_server.npz (warm-up + test)      -> giữ ở máy SERVER")

    Xv, yv = du_lieu["validation"]
    np.savez("du_lieu_validation.npz", X=Xv, y=yv)
    print("Đã tạo du_lieu_validation.npz (validation chung) -> giữ ở máy SERVER")

    print()
    for i, (X, y) in enumerate(du_lieu["clients"]):
        ten_file = f"du_lieu_client_{i}.npz"
        np.savez(ten_file, X=X, y=y)
        print(f"Đã tạo {ten_file} ({len(y)} mẫu)  -> COPY sang máy ẢO client {i}")

    print(f"\nXong. Copy {cfg.SO_CLIENT} file du_lieu_client_*.npz sang đúng máy ảo tương ứng,")
    print("sau đó XOÁ chúng khỏi máy chuẩn bị/máy server (server không được giữ dữ liệu riêng của client).")


if __name__ == "__main__":
    main()
