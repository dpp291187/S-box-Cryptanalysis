# DDT = Differential Distribution Table
# S is a list of integers giving the values of the S-box
# n = number of input bits
# m = number of output bits

def ddt(S, n, m):
    D = [[0] * (2 ** m) for _ in range(2 ** n)]
    for alpha in range(2 ** n):
        for x in range(2 ** n):
            beta = S[x] ^ S[x ^ alpha]
            D[alpha][beta] += 1
    return D
