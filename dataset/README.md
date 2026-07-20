# Thư mục `dataset/` — Dữ liệu

Phần này tương ứng với **"shared validation pipeline"** ở đáy Hình 3.

## File

| File | Nhiệm vụ |
|------|----------|
| `du_lieu.py` | Nạp dữ liệu, chia cho các client (IID / non-IID), tạo tập validation chung và tập warm-up |

## Hàm chính

`nap_va_chia_du_lieu(so_client, non_iid, alpha, seed)` trả về một dict gồm:
- `clients` — danh sách dữ liệu riêng của từng client (mỗi client giữ, không chia sẻ)
- `validation` — tập validation **chung** mà server dùng để chấm điểm client
- `warmup` — tập công khai nhỏ để khởi động mô hình (Section 4.1)
- `test` — tập kiểm tra cuối cùng

## Muốn đổi sang dữ liệu thật?

Mở `du_lieu.py`, sửa **duy nhất** hàm `_nap_goc()`. Ví dụ dùng MNIST thật:

```python
from torchvision.datasets import MNIST
import numpy as np

def _nap_goc():
    ds = MNIST(root="./data", train=True, download=True)
    X = ds.data.numpy().reshape(-1, 784) / 255.0
    y = ds.targets.numpy()
    return X, y
```

Nhớ sửa `SO_LOP` nếu số lớp khác 10, và các phần khác của hệ thống tự thích ứng theo `so_dac_trung`.
