# defined_sboxes1: S-boxes for f1(x) = x^4 + x + 1
defined_sboxes1 = [
 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # Identity mapping (no XOR)
 [0, 1, 4, 5, 3, 2, 7, 6, 12, 13, 8, 9, 15, 14, 11, 10], # 2 XOR gates
 [0, 1, 3, 2, 5, 4, 6, 7, 15, 14, 12, 13, 10, 11, 9, 8], # 5 XOR gates
 [0, 1, 11, 13, 9, 14, 6, 7, 12, 5, 8, 3, 15, 2, 4, 10], # Complexity (27,24)
 [0, 1, 5, 4, 2, 3, 7, 6, 10, 11, 15, 14, 8, 9, 13, 12], # 2 XOR gates
 [0, 1, 14, 9, 11, 13, 7, 6, 8, 3, 10, 4, 12, 5, 2, 15], # Complexity (30,26)
 [0, 1, 13, 11, 14, 9, 6, 7, 10, 4, 15, 2, 8, 3, 5, 12], # Complexity (28,26)
 [0, 1, 9, 14, 13, 11, 7, 6, 15, 2, 12, 5, 10, 4, 3, 8], # Complexity (22,23)
]

# defined_sboxes2: S-boxes for f2(x) = x^4 + x^3 + 1
defined_sboxes2 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # Identity mapping
    [0, 1, 4, 5, 9, 8, 13, 12, 15, 14, 11, 10, 6, 7, 2, 3], # 4 XOR gates
    [0, 1, 9, 8, 14, 15, 7, 6, 3, 2, 10, 11, 13, 12, 4, 5], # 4 XOR gates
    [0, 1, 7, 5, 12, 8, 2, 9, 15, 6, 10, 11, 14, 4, 13, 3], # Complexity (33,28)
    [0, 1, 14, 15, 2, 3, 12, 13, 5, 4, 11, 10, 7, 6, 9, 8], # 3 XOR gates
    [0, 1, 13, 3, 7, 5, 14, 4, 8, 12, 11, 10, 9, 2, 6, 15], # Complexity (30,26)
    [0, 1, 6, 15, 13, 3, 9, 2, 5, 7, 10, 11, 4, 14, 12, 8], # Complexity (34,27)
    [0, 1, 12, 8, 6, 15, 4, 14, 3, 13, 11, 10, 2, 9, 7, 5], # Complexity (32,26)
]

# defined_sboxes3: S-boxes for f3(x) = x^4 + x^3 + x^2 + x + 1
defined_sboxes3 = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # Identity mapping
    [0, 1, 4, 5, 15, 14, 11, 10, 2, 3, 6, 7, 13, 12, 9, 8],
    [0, 1, 15, 14, 8, 9, 7, 6, 4, 5, 11, 10, 12, 13, 3, 2],
    [0, 1, 4, 7, 15, 10, 3, 14, 2, 11, 9, 5, 12, 13, 6, 8],
    [0, 1, 8, 9, 2, 3, 10, 11, 15, 14, 7, 6, 13, 12, 5, 4],
    [0, 1, 2, 11, 4, 7, 9, 5, 8, 6, 14, 3, 13, 12, 10, 15],
    [0, 1, 8, 6, 2, 11, 14, 3, 15, 10, 5, 9, 12, 13, 7, 4],
    [0, 1, 15, 10, 8, 6, 5, 9, 4, 7, 3, 14, 13, 12, 11, 2],
]

# Select S-boxes for the construction
S1 = defined_sboxes1[5]
S2 = defined_sboxes1[7]
S3 = defined_sboxes1[6]
S4 = defined_sboxes1[7]

# Multiplication in GF(16) with irreducible polynomial x^4 + x + 1 (0b10011)
def gf16_mul(a, b):
    """Multiply two 4-bit numbers in GF(2^4)."""
    result = 0
    for _ in range(4):
        if b & 1:
            result ^= a  # XOR if the lowest bit of b is 1
        b >>= 1  # Shift b to the right
        a <<= 1  # Shift a to the left
        # If overflow occurs (5th bit set), reduce with the polynomial
        if a & 0b10000:
            a ^= 0b10011
    return result & 0b1111  # Keep only 4 bits

def calculate_sbox():
    """Construct an 8×8 S-box from the selected 4-bit S-boxes."""
    sbox = []
    for x in range(256):
        # Split input x into two 4-bit parts
        x_i = (x >> 4) & 0xF  # High 4 bits
        y_i = x & 0xF         # Low 4 bits

        # Compute x_o
        if x_i != 0:
            x_o = gf16_mul(S1[y_i], x_i)  # Multiplication in GF(16)
        else:
            x_o = S2[y_i]

        # Compute y_o
        if x_o != 0:
            y_o = S3[gf16_mul(x_i, x_o)]
        else:
            y_o = S4[x_i]

        # Combine x_o and y_o into one 8-bit output
        y = (x_o << 4) | y_o
        y ^= 1  # Final XOR with 1 for additional transformation
        sbox.append(y)

    return sbox

# Compute the final S-box
SBOX = calculate_sbox()

# Print the S-box in hex format
print("MDPISBOX = [")
for i in range(0, len(SBOX), 16):
    row = ", ".join(f"0x{val:02X}" for val in SBOX[i:i+16])
    print(f"  {row},")
print("]")
