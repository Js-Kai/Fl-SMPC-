"""
crypto.py — Pedersen commitment + Schnorr ZKP trên đường cong secp256k1.

Bài báo (Table 3, mục Cryptographic Parameters) ghi rõ:
    Group: secp256k1,  key size 256 bits,  commitment 32 bytes,
    proof: Schnorr-style ZKP (~1.2 KB),  verification < 5 ms.

Ở đây ta hiện thực ĐÚNG secp256k1 bằng Python thuần (không cần thư viện ngoài).
Toạ độ Jacobian để tránh nghịch đảo modulo trong vòng lặp -> nhanh hơn nhiều.
"""

import hashlib
import secrets

# ---------------------------------------------------------------- secp256k1
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F  # trường
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # bậc nhóm
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

G = (Gx, Gy)          # generator chuẩn
INF = None            # điểm vô cực (phần tử trung hoà)


# ------------------------------------------------------- số học trên đường cong
def _inv(a: int, m: int = P) -> int:
    return pow(a, m - 2, m)


def _jac_double(Pt):
    if Pt is None:
        return None
    X, Y, Z = Pt
    if Y == 0:
        return None
    A = (Y * Y) % P
    B = (4 * X * A) % P
    C = (8 * A * A) % P
    D = (3 * X * X) % P            # a = 0 với secp256k1
    X3 = (D * D - 2 * B) % P
    Y3 = (D * (B - X3) - C) % P
    Z3 = (2 * Y * Z) % P
    return (X3, Y3, Z3)


def _jac_add(Pt, Qt):
    if Pt is None:
        return Qt
    if Qt is None:
        return Pt
    X1, Y1, Z1 = Pt
    X2, Y2, Z2 = Qt
    Z1Z1 = (Z1 * Z1) % P
    Z2Z2 = (Z2 * Z2) % P
    U1 = (X1 * Z2Z2) % P
    U2 = (X2 * Z1Z1) % P
    S1 = (Y1 * Z2 * Z2Z2) % P
    S2 = (Y2 * Z1 * Z1Z1) % P
    if U1 == U2:
        if S1 != S2:
            return None
        return _jac_double(Pt)
    H = (U2 - U1) % P
    Rr = (S2 - S1) % P
    HH = (H * H) % P
    HHH = (H * HH) % P
    U1HH = (U1 * HH) % P
    X3 = (Rr * Rr - HHH - 2 * U1HH) % P
    Y3 = (Rr * (U1HH - X3) - S1 * HHH) % P
    Z3 = (H * Z1 * Z2) % P
    return (X3, Y3, Z3)


def _to_jac(pt):
    return None if pt is None else (pt[0], pt[1], 1)


def _from_jac(Pt):
    if Pt is None:
        return None
    X, Y, Z = Pt
    zi = _inv(Z)
    zi2 = (zi * zi) % P
    return ((X * zi2) % P, (Y * zi2 * zi) % P)


def mul(k: int, pt=G):
    """Nhân vô hướng k·pt (thuật toán double-and-add trên toạ độ Jacobian)."""
    k %= N
    if k == 0 or pt is None:
        return None
    R = None
    Q = _to_jac(pt)
    while k:
        if k & 1:
            R = _jac_add(R, Q)
        Q = _jac_double(Q)
        k >>= 1
    return _from_jac(R)


def add(p1, p2):
    return _from_jac(_jac_add(_to_jac(p1), _to_jac(p2)))


def encode(pt) -> bytes:
    """Nén điểm thành 33 byte (1 byte parity + 32 byte x) — đúng 32-33 byte như bài báo."""
    if pt is None:
        return b"\x00" * 33
    x, y = pt
    return bytes([2 + (y & 1)]) + x.to_bytes(32, "big")


# Generator thứ hai H, dẫn xuất bằng hash-to-curve đơn giản (nothing-up-my-sleeve).
# Không ai biết log_G(H) -> điều kiện BẮT BUỘC để Pedersen có tính binding.
def _derive_H():
    seed = b"FL-SMPC++ second generator H"
    ctr = 0
    while True:
        x = int.from_bytes(hashlib.sha256(seed + ctr.to_bytes(4, "big")).digest(), "big") % P
        y2 = (pow(x, 3, P) + 7) % P
        y = pow(y2, (P + 1) // 4, P)          # p ≡ 3 mod 4 -> căn bậc hai nhanh
        if (y * y) % P == y2:
            return (x, y if y % 2 == 0 else P - y)
        ctr += 1


H = _derive_H()


# ------------------------------- tăng tốc: bảng tiền tính cho hai base cố định
# G và H không đổi trong suốt hệ thống -> tiền tính 15 bội số cho mỗi nibble.
# Nhân vô hướng khi đó chỉ còn <=64 phép cộng thay vì 256 double + add.
_W = 4                      # độ rộng cửa sổ (nibble)
_NIB = 256 // _W            # 64 nibble


def _build_table(base):
    tbl = []
    cur = _to_jac(base)
    for _ in range(_NIB):
        row = [None]
        acc = None
        for _ in range(15):
            acc = _jac_add(acc, cur)
            row.append(_from_jac(acc))
        tbl.append(row)
        for _ in range(_W):
            cur = _jac_double(cur)
    return tbl


_TBL_G = _build_table(G)
_TBL_H = _build_table(H)


def _mul_fixed(k: int, tbl):
    k %= N
    if k == 0:
        return None
    R = None
    i = 0
    while k:
        d = k & 0xF
        if d:
            R = _jac_add(R, _to_jac(tbl[i][d]))
        k >>= _W
        i += 1
    return _from_jac(R)


def mul_g(k: int):
    """k·G dùng bảng tiền tính."""
    return _mul_fixed(k, _TBL_G)


def mul_h(k: int):
    """k·H dùng bảng tiền tính."""
    return _mul_fixed(k, _TBL_H)


# ------------------------------------------------------------------ tiện ích
def hash_to_scalar(*chunks: bytes) -> int:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return int.from_bytes(h.digest(), "big") % N


def vector_to_scalar(vec) -> int:
    """Nén vector cập nhật (chiều cao) -> 1 scalar trong Z_n.

    Bài báo: 'committing to a Merkle root or a cryptographic hash digest of the
    update vector' — ta dùng SHA-256 của biểu diễn đã lượng tử hoá.
    """
    import numpy as np
    q = np.round(np.asarray(vec, dtype=np.float64), 6) + 0.0   # +0.0 xoá -0.0
    return hash_to_scalar(q.tobytes())


# ------------------------------------------------------- Pedersen commitment
class Pedersen:
    """C = m·G + r·H   (viết cộng tính vì đang ở nhóm đường cong elliptic).

    - hiding: r ngẫu nhiên -> C không lộ m
    - binding: mở sang m' ≠ m đòi hỏi giải DLog -> bất khả thi
    """

    @staticmethod
    def commit(m: int, r: int | None = None):
        if r is None:
            r = secrets.randbelow(N)
        C = add(mul_g(m % N), mul_h(r))
        return C, r

    @staticmethod
    def verify_opening(C, m: int, r: int) -> bool:
        return C == add(mul_g(m % N), mul_h(r))

    @staticmethod
    def size_bytes() -> int:
        return 33


# ---------------------------------------------------------------- Schnorr ZKP
class SchnorrZKP:
    """Chứng minh biết (m, r) sao cho C = m·G + r·H, không tiết lộ m, r.

    Prover:  a,b ngẫu nhiên;  T = a·G + b·H
             e = H(G, H, C, T)                    (Fiat-Shamir)
             s1 = a + e·m,   s2 = b + e·r         (mod n)
    Verifier: s1·G + s2·H  ==  T + e·C
    """

    @staticmethod
    def prove(m: int, r: int, C):
        a = secrets.randbelow(N)
        b = secrets.randbelow(N)
        T = add(mul_g(a), mul_h(b))
        e = hash_to_scalar(encode(G), encode(H), encode(C), encode(T))
        s1 = (a + e * (m % N)) % N
        s2 = (b + e * r) % N
        return (T, s1, s2)

    @staticmethod
    def verify(C, proof) -> bool:
        T, s1, s2 = proof
        e = hash_to_scalar(encode(G), encode(H), encode(C), encode(T))
        lhs = add(mul_g(s1), mul_h(s2))
        rhs = add(T, mul(e, C))
        return lhs == rhs

    @staticmethod
    def size_bytes() -> int:
        return 33 + 32 + 32          # T nén + s1 + s2
