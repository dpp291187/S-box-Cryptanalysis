import math
import numpy as np

# =========================================================
# Chỉ cần sửa SBOX ở đây
# =========================================================
SBOX = [1, 9, 15, 13, 14, 11, 10, 5, 6, 12, 4, 0, 2, 8, 3, 7]


# =========================================================
# Hàm cơ bản
# =========================================================
def is_power_of_two(x):
    return x > 0 and (x & (x - 1)) == 0


def infer_n(sbox):
    L = len(sbox)
    if not is_power_of_two(L):
        raise ValueError("Độ dài SBOX phải là 2^n.")
    n = int(math.log2(L))
    if min(sbox) < 0 or max(sbox) >= (1 << n):
        raise ValueError("Giá trị trong SBOX không nằm trong [0, 2^n - 1].")
    return n


def format_sbox_hex(sbox):
    lines = []
    for i in range(0, len(sbox), 16):
        lines.append("   " + ", ".join(f"0x{x:02X}" for x in sbox[i:i+16]))
    return "\n".join(lines)


def boolean_functions_from_sbox(sbox, n):
    """
    f0 là bit LSB, f1 là bit tiếp theo, ...
    Mỗi hàm là chuỗi bit ứng với x = 0..2^n-1
    """
    funcs = []
    for bit in range(n):
        bits = ''.join(str((sbox[x] >> bit) & 1) for x in range(1 << n))
        funcs.append(bits)
    return funcs


# =========================================================
# Nonlinearity
# =========================================================
def fwht(arr):
    """Fast Walsh-Hadamard Transform"""
    a = arr.astype(np.int32).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x = a[i:i+h].copy()
            y = a[i+h:i+2*h].copy()
            a[i:i+h] = x + y
            a[i+h:i+2*h] = x - y
        h <<= 1
    return a


def nonlinearity_bool(bits):
    n = int(math.log2(len(bits)))
    seq = np.array([1 - 2 * int(b) for b in bits], dtype=np.int32)  # 0->1, 1->-1
    W = fwht(seq)
    return (1 << (n - 1)) - (np.max(np.abs(W)) // 2)


# =========================================================
# SAC
# =========================================================
def sac_vector(bits, n):
    """
    SAC theo từng bit vào:
    SAC[i] = P(f(x) != f(x xor e_i))
    """
    f = np.array([int(b) for b in bits], dtype=np.int8)
    out = []
    for i in range(n):
        e = 1 << i
        cnt = 0
        for x in range(1 << n):
            cnt += f[x] ^ f[x ^ e]
        out.append(cnt / (1 << n))
    return out


# =========================================================
# BIC-SAC
# =========================================================
def bic_functions(funcs):
    out = []
    m = len(funcs)
    for i in range(m):
        for j in range(i + 1, m):
            bits = ''.join(str(int(funcs[i][k]) ^ int(funcs[j][k])) for k in range(len(funcs[i])))
            out.append(((i, j), bits))
    return out


# =========================================================
# DDT / DP
# =========================================================
def ddt(sbox, n):
    D = [[0] * (1 << n) for _ in range(1 << n)]
    for a in range(1 << n):
        for x in range(1 << n):
            b = sbox[x] ^ sbox[x ^ a]
            D[a][b] += 1
    return D


def differential_probability(sbox, n):
    """
    DP = max_{a!=0, b} DDT[a][b] / 2^n
    """
    D = ddt(sbox, n)
    best = 0
    best_pair = None
    for a in range(1, 1 << n):
        for b in range(1 << n):
            if D[a][b] > best:
                best = D[a][b]
                best_pair = (a, b)
    return best / (1 << n), best, best_pair


# =========================================================
# LAT / LP
# =========================================================
def parity(x):
    return x.bit_count() & 1


def lat(sbox, n):
    """
    LAT dạng bias:
    LAT[a][b] = #{x | a·x xor b·S(x)=0} - 2^(n-1)
    """
    L = [[0] * (1 << n) for _ in range(1 << n)]
    for a in range(1 << n):
        for b in range(1 << n):
            e = 0
            for x in range(1 << n):
                if parity(a & x) ^ parity(b & sbox[x]) == 0:
                    e += 1
            L[a][b] = e - (1 << (n - 1))
    return L


def linear_probability(sbox, n):
    """
    LP = max_{a!=0, b!=0} |LAT[a][b]| / 2^n
    """
    L = lat(sbox, n)
    best = 0
    best_pair = None
    for a in range(1, 1 << n):
        for b in range(1, 1 << n):
            v = abs(L[a][b])
            if v > best:
                best = v
                best_pair = (a, b)
    return best / (1 << n), best, best_pair


# =========================================================
# FP / OFP
# =========================================================
def fixed_and_opposite_fixed_points(sbox, n):
    fp = 0
    ofp = 0
    mask = (1 << n) - 1
    for x, y in enumerate(sbox):
        if y == x:
            fp += 1
        if y == (mask ^ x):
            ofp += 1
    return fp, ofp


# =========================================================
# ANF bằng Möbius transform
# =========================================================
def anf_of_bits(bits, n):
    """
    Trả về ANF của 1 hàm Boolean.
    Biến in ra theo thứ tự x(n-1), ..., x0
    """
    a = np.array([int(b) for b in bits], dtype=np.int8)

    # Möbius transform
    for i in range(n):
        step = 1 << i
        for mask in range(1 << n):
            if mask & step:
                a[mask] ^= a[mask ^ step]

    vars_name = [f"x{i}" for i in range(n - 1, -1, -1)]
    terms = []

    for mask, coef in enumerate(a):
        if coef == 0:
            continue
        if mask == 0:
            terms.append("1")
        else:
            monomial = []
            for i in range(n):
                if (mask >> i) & 1:
                    monomial.append(f"x{i}")
            terms.append("(" + " & ".join(monomial) + ")")

    return " ^ ".join(terms) if terms else "0"


# =========================================================
# MAIN
# =========================================================
def main():
    n = infer_n(SBOX)
    funcs = boolean_functions_from_sbox(SBOX, n)

    print("====================================================")
    print("        SIMPLE S-BOX ANALYSIS")
    print("====================================================")
    print(f"n = {n}")
    print("SBOX (hex):")
    print(format_sbox_hex(SBOX))
    print()

    # FP / OFP
    fp, ofp = fixed_and_opposite_fixed_points(SBOX, n)
    print("FP  =", fp)
    print("OFP =", ofp)
    print()

    # NL + SAC + ANF cho từng hàm tọa độ
    print("=============== COORDINATE FUNCTIONS ===============")
    nl_list = []
    sac_avg_list = []

    for i, bits in enumerate(funcs):
        nl = int(nonlinearity_bool(bits))
        sac_vec = sac_vector(bits, n)
        sac_avg = sum(sac_vec) / n
        anf = anf_of_bits(bits, n)

        nl_list.append(nl)
        sac_avg_list.append(sac_avg)

        print(f"f{i}(x) = {bits}")
        print(f"  NL      = {nl}")
        print(f"  SAC     = {[round(v, 4) for v in sac_vec]}")
        print(f"  SAC_avg = {sac_avg:.4f}")
        print(f"  ANF     = {anf}")
        print()

    print("Average NL      =", sum(nl_list) / len(nl_list))
    print("Average SAC     =", sum(sac_avg_list) / len(sac_avg_list))
    print()

    # BIC-SAC
    print("=================== BIC - SAC ======================")
    bic = bic_functions(funcs)
    bic_avg_all = []

    for (i, j), bits in bic:
        sac_vec = sac_vector(bits, n)
        sac_avg = sum(sac_vec) / n
        bic_avg_all.append(sac_avg)
        print(f"f{i} ^ f{j}: SAC = {[round(v, 4) for v in sac_vec]}, SAC_avg = {sac_avg:.4f}")

    print("BIC-SAC average =", sum(bic_avg_all) / len(bic_avg_all))
    print()

    # DP
    dp, dp_count, dp_pair = differential_probability(SBOX, n)
    print("====================== DP ==========================")
    print(f"DP = {dp:.6f}   (max DDT = {dp_count}, at a={dp_pair[0]}, b={dp_pair[1]})")
    print()

    # LP
    lp, lp_bias, lp_pair = linear_probability(SBOX, n)
    print("====================== LP ==========================")
    print(f"LP = {lp:.6f}   (max |LAT| = {lp_bias}, at a={lp_pair[0]}, b={lp_pair[1]})")
    print()


if __name__ == "__main__":
    main()
