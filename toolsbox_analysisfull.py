
import argparse
import math
import numpy as np
import re
from typing import List, Tuple, Dict

MDPISBOX =[
   0x01, 0x11, 0x91, 0xE1, 0xD1, 0xB1, 0x71, 0x61, 0xF1, 0x21, 0xC1, 0x51, 0xA1, 0x41, 0x31, 0x81,
   0x00, 0x10, 0x93, 0xE2, 0xD5, 0xB4, 0x77, 0x66, 0xF9, 0x28, 0xCB, 0x5A, 0xAD, 0x4C, 0x3F, 0x8E,
   0x08, 0x2C, 0x18, 0xF5, 0x90, 0x5D, 0xE9, 0xC4, 0xD3, 0x4E, 0xBA, 0xA7, 0x72, 0x8F, 0x6B, 0x36,
   0x0F, 0x3A, 0x84, 0x1F, 0x4B, 0xE0, 0x9E, 0xA5, 0x26, 0x6D, 0x73, 0xF8, 0xDC, 0xC7, 0x59, 0xB2,
   0x0C, 0x4F, 0x2E, 0xD0, 0x1C, 0xA2, 0xF3, 0xBD, 0x98, 0x86, 0x57, 0x79, 0xE5, 0x3B, 0xCA, 0x64,
   0x0A, 0x58, 0xB0, 0x39, 0xC3, 0x1A, 0x82, 0xDB, 0x65, 0xAC, 0x94, 0x2D, 0x47, 0x7E, 0xF6, 0xEF,
   0x06, 0x67, 0x3D, 0x2B, 0x8A, 0xFC, 0x16, 0x70, 0x44, 0xC2, 0xE8, 0xDE, 0x9F, 0xB9, 0xA3, 0x55,
   0x07, 0x76, 0xAF, 0xC8, 0x5E, 0x49, 0x60, 0x17, 0xBC, 0xEB, 0x22, 0x85, 0x33, 0xF4, 0x9D, 0xDA,
   0x0E, 0x8B, 0x46, 0x9C, 0x2F, 0x75, 0xD8, 0x52, 0x1E, 0x34, 0xA9, 0xE3, 0xF0, 0x6A, 0xB7, 0xCD,
   0x03, 0x95, 0xD9, 0x7D, 0xF2, 0xC6, 0xAA, 0x3E, 0xE7, 0x13, 0x6F, 0xBB, 0x54, 0x20, 0x8C, 0x48,
   0x0D, 0xAE, 0x5C, 0x63, 0xB8, 0x27, 0x35, 0x9A, 0xC0, 0x7F, 0x1D, 0x42, 0x89, 0xE6, 0xD4, 0xFB,
   0x04, 0xB3, 0xC5, 0x87, 0x69, 0x9B, 0x4D, 0xFF, 0x32, 0x50, 0xD6, 0x14, 0x2A, 0xA8, 0xEE, 0x7C,
   0x0B, 0xC9, 0x62, 0x4A, 0x37, 0xDF, 0x24, 0xEC, 0x8D, 0xB5, 0xFE, 0x96, 0x1B, 0x53, 0x78, 0xA0,
   0x05, 0xD2, 0xF7, 0xA4, 0xED, 0x6E, 0x5B, 0x88, 0x7A, 0x99, 0x3C, 0xCF, 0xB6, 0x15, 0x40, 0x23,
   0x02, 0xE4, 0x7B, 0xBE, 0xA6, 0x83, 0xCC, 0x29, 0x5F, 0xFA, 0x45, 0x30, 0x68, 0xDD, 0x12, 0x97,
   0x09, 0xFD, 0xEA, 0x56, 0x74, 0x38, 0xBF, 0x43, 0xAB, 0xD7, 0x80, 0x6C, 0xCE, 0x92, 0x25, 0x19]

MDPISBOX2 = [
    0x01, 0x11, 0x91, 0xE1, 0xD1, 0xB1, 0x71, 0x61, 0xF1, 0x21, 0xC1, 0x51, 0xA1, 0x41, 0x31, 0x81,
    0x00, 0x10, 0xE4, 0x95, 0xB3, 0xD2, 0x76, 0x67, 0x8B, 0x3A, 0xAE, 0x4F, 0xC9, 0x58, 0x2C, 0xFD,
    0x08, 0x2F, 0xF2, 0x1C, 0x5E, 0x90, 0xED, 0xC3, 0x37, 0x69, 0x74, 0x8A, 0xB8, 0xA6, 0x4B, 0xD5,
    0x0F, 0x38, 0x1A, 0x83, 0xE0, 0x49, 0x9B, 0xA2, 0xB4, 0x5D, 0xDF, 0xC6, 0x75, 0xFC, 0x6E, 0x27,
    0x0C, 0x4A, 0xD0, 0x2B, 0xA4, 0x1F, 0xF5, 0xBE, 0x63, 0xC8, 0xE2, 0x39, 0x56, 0x7D, 0x87, 0x9C,
    0x0A, 0x5C, 0x3D, 0xB0, 0x18, 0xC5, 0x84, 0xD9, 0xEA, 0xF7, 0x46, 0x7B, 0x93, 0x2E, 0xAF, 0x62,
    0x06, 0x66, 0x29, 0x3E, 0xFF, 0x88, 0x17, 0x70, 0x52, 0xA5, 0x9A, 0xBD, 0xEC, 0xDB, 0xC4, 0x43,
    0x07, 0x77, 0xCC, 0xAA, 0x4D, 0x5B, 0x60, 0x16, 0xD8, 0x9E, 0x35, 0xF3, 0x24, 0x82, 0xE9, 0xBF,
    0x0E, 0x89, 0x9F, 0x47, 0x72, 0x2A, 0xDC, 0x54, 0xCE, 0xB6, 0xF0, 0x68, 0xAD, 0xE5, 0x33, 0x1B,
    0x03, 0x92, 0x7E, 0xDD, 0xC7, 0xF4, 0xA8, 0x3B, 0x4C, 0x8F, 0x53, 0x20, 0x6A, 0xB9, 0x15, 0xE6,
    0x0D, 0xAB, 0x65, 0x5F, 0x26, 0xBC, 0x32, 0x98, 0xF9, 0xD3, 0x8D, 0xE7, 0x1E, 0x44, 0x7A, 0xC0,
    0x04, 0xB5, 0x86, 0xC2, 0x99, 0x6D, 0x4E, 0xFA, 0x7F, 0xEB, 0x28, 0xAC, 0xD7, 0x13, 0x50, 0x34,
    0x0B, 0xCD, 0x48, 0x64, 0xDA, 0x36, 0x23, 0xEF, 0xA0, 0x7C, 0x19, 0x55, 0xFB, 0x97, 0xB2, 0x8E,
    0x05, 0xD4, 0xA3, 0xF6, 0x6B, 0xEE, 0x59, 0x8C, 0x25, 0x40, 0xB7, 0x12, 0x3F, 0xCA, 0x9D, 0x78,
    0x02, 0xE3, 0xBB, 0x79, 0x85, 0xA7, 0xCF, 0x2D, 0x96, 0x14, 0x6C, 0xDE, 0x42, 0x30, 0xF8, 0x5A,
    0x09, 0xFE, 0x57, 0xE8, 0x3C, 0x73, 0xBA, 0x45, 0x1D, 0x22, 0xCB, 0x94, 0x80, 0x6F, 0xD6, 0xA9,
]

AES = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

def is_power_of_two(x: int) -> bool:
    """Return True iff x is a power of two (x=2^k, k>=0)."""
    return x > 0 and (x & (x - 1)) == 0


def bitstring_to_hex(bits: str) -> str:
    """Convert a bitstring (MSB-first) to '0x...' with zero padding to a full nibble."""
    if not bits:
        return "0x0"
    pad = (-len(bits)) % 4
    if pad:
        bits = '0' * pad + bits
    return f"0x{int(bits, 2):0{len(bits)//4}X}"


def format_sbox_hex(sbox: List[int]) -> str:
    """Pretty-print S-box as hex, 16 values per row."""
    lines = []
    for i in range(0, len(sbox), 16):
        line = ", ".join(f"0x{v:02X}" for v in sbox[i:i+16])
        lines.append("   " + line)
    return "\n".join(lines)


# =========================
# ===== CORE ROUTINES =====
# =========================
def invert_sbox(sbox: List[int]) -> List[int]:
    """Return inverse permutation of a bijective S-box (no bijective check here)."""
    inv = [0] * len(sbox)
    for i, v in enumerate(sbox):
        inv[v] = i
    return inv


def check_bijective(sbox: List[int], n: int) -> bool:
    """Check if S-box is a permutation on {0..2^n-1}."""
    return set(sbox) == set(range(1 << n))


def count_fixed_and_opposite_fixed_points(sbox: List[int], n: int) -> Tuple[int, int]:
    """Count fixed points S(x)=x and opposite fixed points S(x)=(~x mod 2^n)."""
    fp = ofp = 0
    mask = (1 << n) - 1
    for x in range(1 << n):
        y = sbox[x]
        if y == x:
            fp += 1
        elif y == (mask ^ x):
            ofp += 1
    return fp, ofp


def boolean_functions_from_sbox(sbox: List[int], n: int) -> List[str]:
    """
    Extract n coordinate Boolean functions f_i as bitstrings of length 2^n.
    Convention: bit i is LSB=0. The bitstring enumerates x = 0..2^n-1.
    """
    funcs = []
    for i in range(n):
        bits = ''.join(str((sbox[x] >> i) & 1) for x in range(1 << n))
        funcs.append(bits)
    return funcs


def bic_boolean_functions(funcs: List[str]) -> List[Tuple[Tuple[int, int], str]]:
    """
    Build XOR-combination (BIC) functions: g_{i,j} = f_i XOR f_j for i<j.
    Return list of ((i,j), bitstring).
    """
    out = []
    m = len(funcs)
    for i in range(m):
        for j in range(i + 1, m):
            xored = ''.join(str(int(funcs[i][k]) ^ int(funcs[j][k])) for k in range(len(funcs[i])))
            out.append(((i, j), xored))
    return out


def walsh_spectrum(bits: str, n: int) -> np.ndarray:
    """
    Compute Walsh spectrum W_f(w) for Boolean function 'bits' (length 2^n):
      W_f(w) = sum_x (-1)^{ f(x) XOR (x·w) }.
    Direct method (fine for n<=8). For larger n, use FWHT.
    """
    f_vals = np.fromiter((int(b) for b in bits), dtype=np.int8, count=1 << n)
    W = np.zeros(1 << n, dtype=np.int32)
    for w in range(1 << n):
        acc = 0
        for x in range(1 << n):
            dot = (x & w).bit_count() & 1
            acc += 1 if (f_vals[x] ^ dot) == 0 else -1
        W[w] = acc
    return W


def nonlinearity_from_walsh(W: np.ndarray, n: int) -> int:
    """
    Nonlinearity of Boolean function:
      NL(f) = 2^{n-1} - (max_w |W_f(w)|)/2
    Return integer.
    """
    return int((1 << (n - 1)) - (int(np.max(np.abs(W))) // 2))


def sac_matrix(bits: str, n: int) -> np.ndarray:
    """
    Strict Avalanche Criterion (SAC) matrix of shape (2^n) x n:
      SAC[x,i] = f(x) XOR f(x XOR e_i)
    """
    f_vals = np.fromiter((int(b) for b in bits), dtype=np.int8, count=1 << n)
    sac = np.zeros((1 << n, n), dtype=np.int8)
    for x in range(1 << n):
        fx = f_vals[x]
        for i in range(n):
            sac[x, i] = fx ^ f_vals[x ^ (1 << i)]
    return sac


def lat(S: List[int], n: int, m: int) -> List[List[int]]:
    """
    Linear Approximation Table
      L(alpha, beta) = e - 2^(n-1),
      where e = #{ x : alpha·x XOR beta·S(x) == 0 }.
    This is the 'bias' table (Walsh/2), not the raw Walsh sum.
    """
    def dot(u: int, v: int) -> int:
        # parity of bitwise dot product
        return (u & v).bit_count() & 1

    L = [[0] * (1 << m) for _ in range(1 << n)]
    for alpha in range(1 << n):
        for beta in range(1 << m):
            e = 0
            for x in range(1 << n):
                if dot(alpha, x) ^ dot(beta, S[x]) == 0:
                    e += 1
            L[alpha][beta] = e - (1 << (n - 1))   # <-- bias, matches your code
    return L


def ddtsbox(S: List[int], n: int, m: int) -> List[List[int]]:
    """
    Differential Distribution Table:
      D(alpha,beta) = # { x | S(x) XOR S(x XOR alpha) = beta }.
    """
    D = [[0] * (1 << m) for _ in range(1 << n)]
    for a in range(1 << n):
        for x in range(1 << n):
            b = S[x] ^ S[x ^ a]
            D[a][b] += 1
    return D


def lap_user_def(S: List[int], n: int, m: int) -> Tuple[float, List[List[int]], int]:
    """
    EXACT user's LAP:
      - Build LAT
      - Take all abs(L[a][b]), sort descending
      - Take the second-largest value
      - Divide by 2^n
    Returns (lap_ratio, LAT, second_largest_abs_value).
    """
    L = lat(S, n, m)
    all_values = [abs(L[a][b]) for a in range(1 << n) for b in range(1 << m)]
    all_values.sort(reverse=True)
    second_largest_abs_value = all_values[1] if len(all_values) >= 2 else all_values[0]
    lap_ratio = second_largest_abs_value / float(1 << n)
    return lap_ratio, L, second_largest_abs_value


def dap_user_def(S: List[int], n: int, m: int) -> Tuple[float, List[List[int],], int]:
    """
    EXACT user's DAP (named cal_ddt in your original code):
      - Build DDT
      - Take unique values, sort descending
      - Choose the first value that is NOT equal to 2^n
      - Divide by 2^n
    Returns (dap_ratio, DDT, chosen_value).
    """
    D = ddtsbox(S, n, m)
    vals = []
    for a in range(1 << n):
        for b in range(1 << m):
            vals.append(D[a][b])
    vals = sorted(set(vals), reverse=True)
    chosen = 0
    for v in vals:
        if v != (1 << n):
            chosen = v
            break
    dap_ratio = chosen / float(1 << n)
    return dap_ratio, D, chosen


def compute_bct_fbct_second_largest(S: List[int], n: int) -> Tuple[int, int]:
    """
    Compute BCT and FBCT and return their second-largest values (global).
    """
    inv = invert_sbox(S)

    bct = [[0] * (1 << n) for _ in range(1 << n)]
    for a in range(1 << n):
        for b in range(1 << n):
            c = 0
            for x in range(1 << n):
                if (inv[S[x] ^ b] ^ inv[S[x ^ a] ^ b]) == a:
                    c += 1
            bct[a][b] = c

    fbct = [[0] * (1 << n) for _ in range(1 << n)]
    for a in range(1 << n):
        for b in range(1 << n):
            c = 0
            for x in range(1 << n):
                if (S[x] ^ S[x ^ a] ^ S[x ^ b] ^ S[x ^ a ^ b]) == 0:
                    c += 1
            fbct[a][b] = c

    def second_largest(table: List[List[int]]) -> int:
        largest = second = -10**9
        for row in table:
            for v in row:
                if v > largest:
                    second = largest
                    largest = v
                elif largest > v > second:
                    second = v
        return second

    return second_largest(bct), second_largest(fbct)


def confusion_metrics(S: List[int], n: int) -> Tuple[float, float]:
    """
    Confusion-like metrics per your older code:
      E_k = (1/2^n) Σ_x ((HW(S(x)) - HW(S(x⊕k)))^2)/4
      MCC = 2nd smallest of {E_k}
      CCV = Var over E'_k = (1/2^n) Σ_x ((HW(S(x)) - HW(S(x⊕k)))^2)
    """
    def hw(x: int) -> int:
        return x.bit_count()

    E, E1 = [], []
    for k in range(1, 1 << n):
        s = s1 = 0.0
        for x in range(1 << n):
            d = hw(S[x]) - hw(S[x ^ k])
            s += (d * d) / 4.0
            s1 += (d * d)
        E.append(s / (1 << n))
        E1.append(s1 / (1 << n))
    E.sort()
    mcc = E[1] if len(E) >= 2 else (E[0] if E else 0.0)
    mu = sum(E1) / len(E1) if E1 else 0.0
    ccv = sum((v - mu) ** 2 for v in E1) / len(E1) if E1 else 0.0
    return mcc, ccv


# =========================
# ===== ANF (MOEBIUS) =====
# =========================
def index_to_term_with_parentheses(index: int, variables: List[str]) -> str:
    """Convert monomial mask to AND-term, e.g., (x3 & x1). 0 -> '1'."""
    if index == 0:
        return "1"
    parts = [v for i, v in enumerate(variables) if (index >> i) & 1]
    return '(' + ' & '.join(parts) + ')'


def generate_all_anf_expressions(coefs: np.ndarray, variables: List[str]) -> Tuple[List[str], int, int]:
    """Build ANF strings and count AND/XOR gates."""
    exprs = []
    total_and = total_xor = 0
    for row in coefs:
        terms = [index_to_term_with_parentheses(i, variables) for i, c in enumerate(row) if c]
        expr = ' ^ '.join(terms) if terms else '0'
        exprs.append(expr)
        total_and += expr.count('&')
        total_xor += expr.count('^')
    return exprs, total_and, total_xor


def max_variables_in_terms(exprs: List[str]) -> int:
    """Algebraic degree = max number of distinct variables in any AND-term."""
    pat = re.compile(r'x\d+')
    best = 0
    for e in exprs:
        for term in e.split('^'):
            best = max(best, len(set(pat.findall(term))))
    return best


def anf_via_moebius(S: List[int], n: int) -> Tuple[List[str], int, int, int]:
    """
    Compute ANF of each coordinate with in-place Möbius transform over truth tables.
    Variables named x{n-1}..x0 (MSB..LSB for visual consistency).
    """
    T = np.zeros((n, 1 << n), dtype=np.int8)
    for x in range(1 << n):
        y = S[x]
        for row in range(n):                      # row 0 is MSB
            T[row, x] = (y >> (n - 1 - row)) & 1

    # Möbius transform per row
    C = T.copy()
    for r in range(n):
        for i in range(n):
            step = 1 << i
            for m in range(1 << n):
                if m & step:
                    C[r, m] ^= C[r, m ^ step]

    vars_names = [f'x{i}' for i in range(n - 1, -1, -1)]
    exprs, t_and, t_xor = generate_all_anf_expressions(C, vars_names)
    ad = max_variables_in_terms(exprs)
    return exprs, t_and, t_xor, ad


# =========================
# ===== MAIN METRICS ======
# =========================
def per_function_metrics(func_bits: str, n: int) -> Dict[str, object]:
    """
    Compute NL and SAC (vector & average) for a single Boolean function.
    """
    W = walsh_spectrum(func_bits, n)
    NL = nonlinearity_from_walsh(W, n)
    SAC = sac_matrix(func_bits, n)
    sac_per_bit = SAC.sum(axis=0) / float(1 << n)  # length n, each ideally ~0.5
    sac_avg = sac_per_bit.mean()
    return {"NL": int(NL), "SAC_vector": sac_per_bit.tolist(), "SAC_avg": float(sac_avg)}


def aggregate_metrics(S: List[int], n: int) -> Dict[str, object]:
    """Compute and return all requested metrics in a single structure."""
    # Basic checks
    bij = check_bijective(S, n)
    fp, ofp = count_fixed_and_opposite_fixed_points(S, n)

    # Coordinate & BIC Boolean functions
    coords = boolean_functions_from_sbox(S, n)
    bic = bic_boolean_functions(coords)  # list of ((i,j), bits)

    # Per-function NL & SAC (coords)
    per_coord = [per_function_metrics(bits, n) for bits in coords]
    NL_coords = [pc["NL"] for pc in per_coord]
    SAC_coords_avg = [pc["SAC_avg"] for pc in per_coord]

    # Per-function NL & SAC (BIC)
    per_bic = []
    for (i, j), bits in bic:
        m = per_function_metrics(bits, n)
        per_bic.append({"pair": (i, j), **m})
    NL_bic_vals = [pb["NL"] for pb in per_bic]
    SAC_bic_avg = [pb["SAC_avg"] for pb in per_bic]

    # LAP & DAP by user's definitions
    LAP_ratio, LAT, LAP_abs2 = lap_user_def(S, n, n)
    DAP_ratio, DDT, DAP_val = dap_user_def(S, n, n)

    # BCT/FBCT (second-largest)
    BCT2, FBCT2 = compute_bct_fbct_second_largest(S, n)

    # Overlap family
    TO = calculate_TO(S, n)
    MTO = calculate_MTO(S, n)
    RTO = calculate_RTO(S, n)

    MCC, CCV = confusion_metrics(S, n)

    # ANF
    anf_exprs, total_and, total_xor, ad = anf_via_moebius(S, n)

    return {
        "Bijective": bij,
        "FP": fp, "OFP": ofp,
        "PerCoord": per_coord,                     # list of dicts: NL, SAC_vector, SAC_avg
        "PerBIC": per_bic,                         # list of dicts: pair,(i,j), NL, SAC_vector, SAC_avg
        "Sbox_NL_avg": float(np.mean(NL_coords)) if NL_coords else 0.0,
        "BIC_NL_avg": float(np.mean(NL_bic_vals)) if NL_bic_vals else 0.0,
        "SAC_avg_over_coords": float(np.mean(SAC_coords_avg)) if SAC_coords_avg else 0.0,
        "SAC_avg_over_bic": float(np.mean(SAC_bic_avg)) if SAC_bic_avg else 0.0,

        "LAP": LAP_ratio,          # per user's definition
        "LAP_abs2": LAP_abs2,      # the second-largest |LAT|
        "LAT": LAT,

        "DAP": DAP_ratio,          # per user's definition
        "DAP_val": DAP_val,        # the chosen DDT value (< 2^n)
        "DDT": DDT,

        "BCT2": BCT2, "FBCT2": FBCT2,
        "TO": TO, "MTO": MTO, "RTO": RTO,
        "MCC": MCC, "CCV": CCV,

        "ANF": {"exprs": anf_exprs, "total_and": total_and, "total_xor": total_xor, "AD": ad},
        "BooleanFunctions": {"coords": coords, "bic": [bits for _, bits in bic]},
    }


# ===== Overlap-family kept faithful to your original math =====
def calculate_TO(sbox: List[int], n: int) -> float:
    """TO = n - (sum_{u!=0} | Σ_i A_fi(u) |) / ((2^n)^2 - 2^n)."""
    funcs = boolean_functions_from_sbox(sbox, n)
    Af = np.zeros((1 << n, n), dtype=np.int32)
    for i, bits in enumerate(funcs):
        for u in range(1 << n):
            acc = 0
            for x in range(1 << n):
                acc += 1 if (int(bits[x]) ^ int(bits[x ^ u])) == 0 else -1
            Af[u, i] = acc
    comb_abs = [abs(np.sum(Af[u, :])) for u in range(1 << n)]
    num = sum(comb_abs[1:])  # exclude u=0
    denom = (1 << (2 * n)) - (1 << n)
    return n - (num / denom)


def calculate_MTO(sbox: List[int], n: int) -> float:
    """MTO as in your original code."""
    funcs = boolean_functions_from_sbox(sbox, n)
    Af = np.zeros((n, n, 1 << n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            bi = funcs[i]
            bj = funcs[j]
            for u in range(1 << n):
                acc = 0
                for x in range(1 << n):
                    acc += 1 if (int(bi[x]) ^ int(bj[x ^ u])) == 0 else -1
                Af[i, j, u] = acc

    sums = []
    for u in range(1, 1 << n):
        for j in range(n):
            sums.append(int(np.sum(Af[:, j, u])))
    final_sum = sum(abs(v) for v in sums)
    denom = (1 << (2 * n)) - (1 << n)
    return n - (final_sum / denom)


def calculate_RTO(sbox: List[int], n: int) -> float:
    """RTO as in your original code."""
    funcs = boolean_functions_from_sbox(sbox, n)
    Af = np.zeros((n, n, 1 << n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            bi = funcs[i]
            bj = funcs[j]
            for u in range(1 << n):
                acc = 0
                for x in range(1 << n):
                    acc += 1 if (int(bi[x]) ^ int(bj[x ^ u])) == 0 else -1
                Af[i, j, u] = acc

    sums_u = []
    for u in range(1, 1 << n):
        sums_u.append(int(np.sum(Af[:, :, u])))
    final_sum = sum(abs(v) for v in sums_u)
    denom = (1 << (2 * n)) - (1 << n)
    return n - (final_sum / denom)


# =========================
# ====== PRINT HELPERS ====
# =========================
def print_boolean_functions(n: int, coords: List[str], bic_bits: List[str]) -> None:
    """Print Boolean functions in both BINARY and HEX for coords and BIC set."""
    print("================ BOOLEAN FUNCTIONS (BINARY & HEX) =========")
    print(f"Coordinate Boolean functions (n={n}, bit0=LSB):")
    for i, bits in enumerate(coords):
        print(f"  f{i}(x)  BINARY: {bits}")
        print(f"  f{i}(x)     HEX: {bitstring_to_hex(bits)}")
    print("\nBIC XOR-combination Boolean functions (f_i XOR f_j, i<j):")
    for k, bits in enumerate(bic_bits, 1):
        print(f"  g{k}(x)  BINARY: {bits}")
        print(f"  g{k}(x)     HEX: {bitstring_to_hex(bits)}")
    print("")


def print_per_function_reports(per_coord: List[Dict], per_bic: List[Dict]) -> None:
    """Print per-function NL and SAC (vector + average) for coords and BIC functions."""
    print("================ PER-FUNCTION NL & SAC (COORDS) ===========")
    for i, pc in enumerate(per_coord):
        sac_vec = ", ".join(f"{v:.4f}" for v in pc["SAC_vector"])
        print(f"f{i}: NL={pc['NL']:d} | SAC_vector=[{sac_vec}] | SAC_avg={pc['SAC_avg']:.4f}")
    print("\n================ PER-FUNCTION NL & SAC (BIC) ==============")
    for idx, pb in enumerate(per_bic, 1):
        i, j = pb["pair"]
        sac_vec = ", ".join(f"{v:.4f}" for v in pb["SAC_vector"])
        print(f"g{idx}=f{i}^f{j}: NL={pb['NL']:d} | SAC_vector=[{sac_vec}] | SAC_avg={pb['SAC_avg']:.4f}")
    print("")


def print_matrix_with_headers(M: List[List[int]], row_bits: int, col_bits: int, title: str) -> None:
    """Pretty ASCII matrix with hex headers (row index = alpha, col index = beta)."""
    rows, cols = len(M), (len(M[0]) if M else 0)
    print(f"================ {title} ================")
    hdr_w = max(1, (col_bits + 3) // 4)
    row_w = max(1, (row_bits + 3) // 4)
    header = [" " * (row_w + 2)] + [f"{c:0{hdr_w}X}" for c in range(cols)]
    print(" ".join(header))
    for r in range(rows):
        row_label = f"{r:0{row_w}X}:"
        values = " ".join(f"{M[r][c]:>4d}" for c in range(cols))
        print(f"{row_label} {values}")
    print("")


# =========================
# ========= DRIVER =========
# =========================
def main():
    parser = argparse.ArgumentParser(description="VLSILAB — S-Box Cryptanalysis (Console, LAP & DAP per user's defs)")
    parser.add_argument("--n", type=int, default=None, help="Override n (bits). If omitted, inferred from len(SBOX).")
    parser.add_argument("--no-lat", action="store_true", help="Do not print LAT to console.")
    parser.add_argument("--no-ddt", action="store_true", help="Do not print DDT to console.")
    args = parser.parse_args()
    SBOX=SBOX20
    # Validate SBOX
    if not SBOX:
        raise ValueError("SBOX is empty. Paste your S-box values into the SBOX list at the top of this file.")
    L = len(SBOX)
    if not is_power_of_two(L):
        raise ValueError(f"SBOX length must be a power of two; got {L}.")
    n_auto = int(round(math.log2(L)))
    if (1 << n_auto) != L:
        raise ValueError("SBOX length is not exactly a power of two.")
    n = args.n if args.n is not None else n_auto
    if (1 << n) != L:
        raise ValueError(f"Inconsistent n: len(SBOX)={L} but n={n} => 2^n={1<<n}.")

    print("============================================================")
    print(" VLSILAB — SBOX CRYPTANALYSIS")
    print("============================================================")
    print(f"n = {n}  (S-box size = {1<<n} entries)\n")
    print("S-box (hex):")
    print(format_sbox_hex(SBOX))
    print("")

    res = aggregate_metrics(SBOX, n)

    print("============== CRYPTANALYSIS PARAMETERS ====================")
    print(f"0. Bijective: {'YES' if res['Bijective'] else 'NO'}")
    print(f"1. Fixed points: {res['FP']}")
    print(f"2. Opposite fixed points: {res['OFP']}\n")

    print("============== MAIN CRYPTOGRAPHIC CRITERIA =================")
    print(f"3.  S-box Nonlinearity (avg over coords) : {res['Sbox_NL_avg']:.6f}")
    print(f"4.  SAC average       (coords)           : {res['SAC_avg_over_coords']:.6f}")
    print(f"5.  BIC Nonlinearity  (avg over XORs)    : {res['BIC_NL_avg']:.6f}")
    print(f"6.  BIC SAC average   (XORs)             : {res['SAC_avg_over_bic']:.6f}")
    print(f"7.  LAP                                  : {res['LAP']:.6f}   | second |LAT| = {res['LAP_abs2']}")
    print(f"8.  DAP                                  : {res['DAP']:.6f}   | chosen DDT = {res['DAP_val']}\n")

    print("================ BOOMERANG ATTACK RESISTANCE ===============")
    print(f"9.1  BCT value : {res['BCT2']}")
    print(f"9.2  FBCT value: {res['FBCT2']}\n")

    print("============= SIDE-CHANNEL ATTACK RESISTANCE ==============")
    print(f"10. TO : {res['TO']:.6f}")
    print(f"11. MTO: {res['MTO']:.6f}")
    print(f"12. RTO: {res['RTO']:.6f}")
    print(f"13. MCC: {res['MCC']:.6f}")
    print(f"14. CCV: {res['CCV']:.6f}")

    print("================ ANF (ALGEBRAIC NORMAL FORM) ===============")
    print(f"Algebraic Degree (max over coords): {res['ANF']['AD']}")
    print(f"Total AND gates across coords     : {res['ANF']['total_and']}")
    print(f"Total XOR gates across coords     : {res['ANF']['total_xor']}\n")
    # for i, expr in enumerate(res["ANF"]["exprs"]):
    #     print(f"  f{i}(x) = {expr}")
    # print("")

    # Boolean functions (binary + hex)
    print_boolean_functions(n, res["BooleanFunctions"]["coords"], res["BooleanFunctions"]["bic"])

    # Detailed per-function NL & SAC
    print_per_function_reports(res["PerCoord"], res["PerBIC"])

    # # LAT / DDT printing
    # if not args.no_lat:
    #     print_matrix_with_headers(res["LAT"], row_bits=n, col_bits=n, title="LINEAR APPROXIMATION TABLE (LAT)")
    # if not args.no_ddt:
    #     print_matrix_with_headers(res["DDT"], row_bits=n, col_bits=n, title="DIFFERENTIAL DISTRIBUTION TABLE (DDT)")

    print("Done.")


if __name__ == "__main__":
    main()

