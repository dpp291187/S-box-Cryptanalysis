# Example S-box as a list

s_box = [24,14,31,20,30,9,1,19,29,11,8,4,17,0,6,22,12,2,21,23,3,16,28,27,18,7,25,15,13,5,26,10] #IRFAN
s_box = [10, 8, 18, 16, 17, 2, 1, 21, 11, 6, 24, 9, 26, 12, 4, 0, 13, 23, 22, 25, 15, 29, 5, 20, 27, 14, 31, 30, 7, 28, 19, 3]

s_box =[10,3,11,22,17,4,1,8,12,28,23,18,26,6,31,20,15,24,29,13,14,19,30,5,25,27,7,0,16,21,2,9] #thakor
s_box = [10, 8, 18, 16, 17, 2, 1, 21, 11, 6, 24, 9, 26, 12, 4, 0, 13, 23, 22, 25, 15, 29, 5, 20, 27, 14, 31, 30, 7, 28, 19, 3] #proposed
s_box = [4, 11, 31,20,26,21,9,2,27,5,8,18,29,3,6,28,30,19,7,14,0,13,17,24,16,12,1,25,22,10,15,23] #asocn
s_box = [1,0,25,26,17,29,21,27,20,5,4,23,14,18,2,28,15,8,6,3,13,7,24,16,30,9,31,10,22,12,11,19]  #fides
s_box =[31, 9, 18, 11, 5, 12, 22, 15, 10, 3, 24, 1, 13, 4, 30, 7,20, 21, 6, 23, 17, 16, 2, 19, 26, 27, 8, 25, 29, 28, 14, 0] #ICEPOLE

s_box = [4, 11, 31,20,26,21,9,2,27,5,8,18,29,3,6,28,30,19,7,14,0,13,17,24,16,12,1,25,22,10,15,23] #asocn

s_box = [9, 15, 0, 8, 10, 11, 2, 12, 3, 4, 7, 6, 5, 1, 13, 14]
s_box = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]
s_box = [9, 6, 7, 1, 12, 0, 15, 5, 8, 3, 13, 14, 2, 11, 10, 4]
s_box = [10, 6, 14, 7, 13, 4, 9, 12, 2, 15, 5, 1, 3, 11, 8, 0]
s_box =[0, 15, 14, 5, 13, 3, 6, 12, 11, 9, 10, 8, 7, 4, 2, 1]
s_box =[9, 2, 12, 13, 10, 5, 3, 14, 15, 8, 11, 6, 4, 7, 0, 1]
s_box =[14, 13, 11, 0, 2, 1, 4, 15, 7, 10, 8, 5, 9, 12, 3, 6] #elephant
s_box =[1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14] #gifft
s_box =[4, 0, 10, 7, 11, 14, 1, 13, 9, 15, 6, 8, 5, 2, 12, 3] #KNOT
s_box =[2, 13, 3, 9, 7, 11, 10, 6, 14, 0, 15, 4, 8, 5, 1, 12] #Pyjamask (2019)
s_box =[0, 6, 14, 1, 15, 4, 7, 13, 9, 8, 12, 5, 2, 10, 3, 11] #satunin
s_box =[7, 4, 10, 9, 1, 15, 11, 0, 12, 3, 2, 6, 8, 14, 13, 5] #Klein
s_box =[0, 8, 1, 15, 2, 10, 7, 9, 4, 13, 5, 6, 14, 3, 11, 12] #spook
s_box =[6, 5, 12, 10, 1, 14, 7, 9, 11, 0, 3, 13, 8, 15, 4, 2] #RECtangle
s_box =[0, 4, 8, 15, 1, 5, 14, 9, 2, 7, 10, 12, 11, 13, 6, 3] #PRIDE
s_box =[12, 10, 13, 3, 14, 11, 15, 7, 8, 9, 1, 5, 0, 2, 4, 6] #CRAFT

n=4

s_box = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]
s_box = [9, 15, 0, 8, 10, 11, 2, 12, 3, 4, 7, 6, 5, 1, 13, 14]

def invert_sbox(s_box):
    inverted_s_box = [0] * len(s_box)
    for i, value in enumerate(s_box):
        inverted_s_box[value] = i
    return inverted_s_box

def calculate_bct(s_box,n):
    inverted_s_box = invert_sbox(s_box)

    bct_table = [[0 for _ in range(2**n)] for _ in range(2**n)]

    for a in range(2**n):
        for b in range(2**n):
            bct_value = 0
            for x in range(2**n):
                if inverted_s_box[s_box[x] ^ b] ^ inverted_s_box[s_box[x ^ a] ^ b] == a:
                    bct_value += 1
            bct_table[a][b] = bct_value

    return bct_table

def calculate_fbct(s_box,n):
    inverted_s_box = invert_sbox(s_box)

    fbct_table = [[0 for _ in range(2**n)] for _ in range(2**n)]

    for a in range(2**n):
        for b in range(2**n):
            fbct_value = 0
            for x in range(2**n):
                if s_box[x] ^ s_box[x ^ a] ^ s_box[x ^ b] ^ s_box[x ^ a ^ b] == 0:
                    fbct_value += 1
            fbct_table[a][b] = fbct_value

    return fbct_table

# Calculate BCT for the example S-box
bct_table = calculate_bct(s_box,n)
fbct_table = calculate_fbct(s_box,n)



def print_table(table):
    for row in table:
        formatted_row = [f"{value:2}" for value in row]  #
        print(" ".join(formatted_row))

# Print the BCT Table
print("BCT Table:")
print_table(bct_table)

# Print the FBCT Table
print("FBCT Table:")
print_table(fbct_table)

def count_even_values(table):
    even_count = [0] * 17  #
    for row in table:
        for value in row:
            if value % 2 == 0:
                even_count[value // 2] += 1  # Increase the count of even value at the corresponding index in the array
    return even_count

# Print the count of even values in the BCT Table
even_count_bct = count_even_values(bct_table)
print("Count of even values in BCT Table:")
for value, count in enumerate(even_count_bct):
    print(f"Count of value {value * 2}: {count}")

# Print the count of even values in the FBCT Table
even_count_fbct = count_even_values(fbct_table)
print("Count of even values in FBCT Table:")
for value, count in enumerate(even_count_fbct):
    print(f"Count of value {value * 2}: {count}")
