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

def count_values_in_ddt(D):
    counts = [0] * 8  # count values 0, 2, 4, 6 , 16.....

    for alpha in range(len(D)):
        for beta in range(len(D[0])):
            value = D[alpha][beta]
            if value == 0:
                counts[0] += 1
            elif value == 2:
                counts[1] += 1
            elif value == 4:
                counts[2] += 1
            elif value == 6:
                counts[3] += 1
            elif value == 8:
                counts[4] += 1
            elif value == 10:
                counts[5] += 1
            elif value == 12:
                counts[6] += 1
            elif value == 16:
                counts[7] += 1

    return counts

def print_ddt_to_file(S, n, m):
    D = ddt(S, n, m)
    ddt_text_file = open("ddt_44.txt", "w")
    first_row = '|'.rjust(4)
    for i in range(2 ** m):
        first_row += str(hex(i))[2:].rjust(4)
    print(first_row, file=ddt_text_file)
    horiziontal_divider = '-' * len(first_row)
    print(horiziontal_divider, file=ddt_text_file)
    for alpha in range(2 ** n):
        row_string = ''
        row_string += str(hex(alpha))[2:].rjust(3) + '| '
        for beta in range(2 ** m):
            d = D[alpha][beta]
            row_string += str(d).rjust(3) + ' '
        print(row_string, file=ddt_text_file)
    ddt_text_file.close()

#  S-Box

sbox =[9, 15, 0, 8, 10, 11, 2, 12, 3, 4, 7, 6, 5, 1, 13, 14]

# Print the DDT matrix to a file
#print_ddt_to_file(R, 5, 5)
print_ddt_to_file(sbox, 4, 4)
# Count the number of values 0, 2, 4, 6, and 16 in the DDT matrix
#D = ddt(R, 5, 5)
D = ddt(sbox, 4, 4)
value_counts = count_values_in_ddt(D)

# Print the results
print("Count of values 0 in the DDT matrix:", value_counts[0])
print("Count of values 2 in the DDT matrix:", value_counts[1])
print("Count of values 4 in the DDT matrix:", value_counts[2])
print("Count of values 6 in the DDT matrix:", value_counts[3])
print("Count of values 8 in the DDT matrix:", value_counts[4])
print("Count of values 10 in the DDT matrix:", value_counts[5])
print("Count of values 12 in the DDT matrix:", value_counts[6])
print("Count of values 16 in the DDT matrix:", value_counts[7])
