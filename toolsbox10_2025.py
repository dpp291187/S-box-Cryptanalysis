import math
import numpy as np

# =========================================================
# Only edit SBOX here
# =========================================================
SBOX = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]

# =========================================================
# Basic helpers
# =========================================================
def is_power_of_two(x):
    return x > 0 and (x & (x - 1)) == 0


def infer_n(sbox):
    length = len(sbox)
    if not is_power_of_two(length):
        raise ValueError("SBOX length must be 2^n.")
    n = int(math.log2(length))
    if min(sbox) < 0 or max(sbox) >= (1 << n):
        raise ValueError("SBOX values must be in [0, 2^n - 1].")
    if len(set(sbox)) != len(sbox):
        print("Warning: the S-box is not bijective.")
    return n


def format_sbox_hex(sbox):
    lines = []
    for i in range(0, len(sbox), 16):
        lines.append("   " + ", ".join(f"0x{x:02X}" for x in sbox[i:i + 16]))
    return "\n".join(lines)


# =========================================================
# Coordinate Boolean functions
# Convention:
# - f0 is the MSB output bit
# - f(n-1) is the LSB output bit
# =========================================================
def boolean_functions_from_sbox(sbox, n):
    funcs = []
    for bit in range(n - 1, -1, -1):
        bits = ''.join(str((sbox[x] >> bit) & 1) for x in range(1 << n))
        funcs.append(bits)
    return funcs


# =========================================================
# Nonlinearity
# =========================================================
def fwht(arr):
    a = arr.astype(np.int32).copy()
    h = 1
    while h < len(a):
        for i in range(0, len(a), 2 * h):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h <<= 1
    return a


def nonlinearity_bool(bits):
    n = int(math.log2(len(bits)))
    seq = np.array([1 - 2 * int(b) for b in bits], dtype=np.int32)
    walsh = fwht(seq)
    return (1 << (n - 1)) - (np.max(np.abs(walsh)) // 2)


# =========================================================
# SAC
# =========================================================
def sac_vector(bits, n):
    f = np.array([int(b) for b in bits], dtype=np.int8)
    out = []
    for i in range(n):
        diff_count = 0
        mask = 1 << i
        for x in range(1 << n):
            diff_count += f[x] ^ f[x ^ mask]
        out.append(diff_count / (1 << n))
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
# Fixed points and opposite fixed points
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
    L = [[0] * (1 << n) for _ in range(1 << n)]
    half = 1 << (n - 1)
    for a in range(1 << n):
        for b in range(1 << n):
            equal_count = 0
            for x in range(1 << n):
                if parity(a & x) ^ parity(b & sbox[x]) == 0:
                    equal_count += 1
            L[a][b] = equal_count - half
    return L


def linear_probability(sbox, n):
    L = lat(sbox, n)
    best = 0
    best_pair = None
    for a in range(1, 1 << n):
        for b in range(1, 1 << n):
            value = abs(L[a][b])
            if value > best:
                best = value
                best_pair = (a, b)
    return best / (1 << n), best, best_pair


# =========================================================
# ANF via Möbius transform
# Convention:
# - x0 is the MSB input bit
# - x(n-1) is the LSB input bit
# =========================================================
def anf_of_bits(bits, n):
    coeffs = np.array([int(b) for b in bits], dtype=np.int8)

    for i in range(n):
        step = 1 << i
        for mask in range(1 << n):
            if mask & step:
                coeffs[mask] ^= coeffs[mask ^ step]

    terms = []
    for mask, coef in enumerate(coeffs):
        if coef == 0:
            continue

        if mask == 0:
            terms.append("1")
        else:
            monomial = []
            for i in range(n):
                if (mask >> i) & 1:
                    monomial.append(f"x{n - 1 - i}")
            terms.append("".join(monomial))

    return " ⊕ ".join(terms) if terms else "0"


# =========================================================
# Main
# =========================================================
def main():
    n = infer_n(SBOX)
    funcs = boolean_functions_from_sbox(SBOX, n)

    print("====================================================")
    print("              SIMPLE S-BOX ANALYSIS")
    print("====================================================")
    print(f"Input size n = {n}")
    print("Bit convention: f0 = MSB output bit, x0 = MSB input bit")
    print()
    print("S-box (hex):")
    print(format_sbox_hex(SBOX))
    print()

    fp, ofp = fixed_and_opposite_fixed_points(SBOX, n)
    print("Fixed points and opposite fixed points")
    print(f"FP  = {fp}")
    print(f"OFP = {ofp}")
    print()

    print("Coordinate Boolean functions")
    nl_list = []
    sac_avg_list = []

    for i, bits in enumerate(funcs):
        nl = int(nonlinearity_bool(bits))
        sac = sac_vector(bits, n)
        sac_avg = sum(sac) / n

        nl_list.append(nl)
        sac_avg_list.append(sac_avg)

        print(f"f{i}(x) = {bits}")
        print(f"  NL      = {nl}")
        print(f"  SAC     = {[round(v, 4) for v in sac]}")
        print(f"  SAC avg = {sac_avg:.4f}")

    print()
    print("ANF of coordinate Boolean functions")
    for i, bits in enumerate(funcs):
        print(f"f{i}(x) = {anf_of_bits(bits, n)}")

    print()
    print(f"Average NL      = {sum(nl_list) / len(nl_list):.4f}")
    print(f"Average SAC     = {sum(sac_avg_list) / len(sac_avg_list):.4f}")
    print()

    print("BIC-SAC")
    bic = bic_functions(funcs)
    bic_avg_values = []

    for (i, j), bits in bic:
        sac = sac_vector(bits, n)
        sac_avg = sum(sac) / n
        bic_avg_values.append(sac_avg)
        print(f"f{i} ⊕ f{j}: SAC = {[round(v, 4) for v in sac]}, SAC avg = {sac_avg:.4f}")

    print(f"BIC-SAC average = {sum(bic_avg_values) / len(bic_avg_values):.4f}")
    print()

    dp, dp_count, dp_pair = differential_probability(SBOX, n)
    print("Differential probability")
    print(f"DP = {dp:.6f}   (max DDT entry = {dp_count}, at a = {dp_pair[0]}, b = {dp_pair[1]})")
    print()

    lp, lp_bias, lp_pair = linear_probability(SBOX, n)
    print("Linear probability")
    print(f"LP = {lp:.6f}   (max |LAT| = {lp_bias}, at a = {lp_pair[0]}, b = {lp_pair[1]})")
    print()


if __name__ == "__main__":
    main()
