# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:31:48 2023

@author: gaiya
"""

from PIL import Image
import hashlib
import numpy as np
import ECC
import matplotlib.pyplot as plt



img = Image.open(r"C:\Users\gaiya\Desktop\測試圖片\lena_gray_512.tif")
#img.show()
M, N = img.size

original_picture= np.array(img)

#256
hash_object = hashlib.sha256(original_picture)
sha_img = int(hash_object.hexdigest(),16)
print(sha_img)


#ECC
eccBlock = ECC.ECC()
enc = eccBlock.encrypt(sha_img)
rec = eccBlock.decrypt(enc)
print("原文：0x%x" % sha_img)
print("密文：X1: (0x%x, \n      0x%x)" % (enc[0], enc[1]))
print("      C: 0x%x" % enc[2])
print("解密：0x%x" % rec)


#shuffle_row
initial_key5 = 0.3333
row_number = []
while len(row_number) < N:
    initial_key5 = 4 * initial_key5 * (1 - initial_key5)
    l = np.floor(np.mod(initial_key5 * 10**14, N)).astype(int)
    if l in row_number:
        continue
    else:
        row_number.append(l)

# Sort rows
row = np.zeros((M, N), dtype=np.uint8)

for i in range(N):
    row[i] = original_picture[row_number[i]]



#shuffle_column
initial_key6 = 0.4444
column_number = []
while len(column_number) < M:
    initial_key6 = 4 * initial_key6 * (1 - initial_key6)
    e = np.floor(np.mod(initial_key6 * 10**14, M)).astype(int)
    if e in column_number:
        continue
    else:
        column_number.append(e)

# Sort rows
column = np.zeros((M, N), dtype=np.uint8)
for i in range(N):
    column[:,i] = row[:,column_number[i]]


image_shuffled = Image.fromarray(column.astype(np.uint8))




# 定义Henon Map超混沌系统的参数
a = 1.4
b = 0.3

# 定义Henon Map超混沌系统的初始状态
x0 = 0.1151654456486489
y0 = 0.146545645645645

# 定义Henon Map超混沌系统的迭代函数
def henon_map(state, a, b):
    x, y, z = state
    xn = 1 - a * x**2 + y
    yn = b * x
    zn = 0.1 * x + z
    return [xn, yn, zn]

# 计算Henon Map超混沌系统的状态随时间的变化
num_iter = 400000
states = np.zeros((num_iter, 3))
states[0, 0] = x0
states[0, 1] = y0
sequence=[]
for i in range(1, num_iter):
    states[i, :] = henon_map(states[i-1, :], a, b)
    sequence += states[i, :].tolist()


sequence = np.floor(np.mod((np.abs(sequence) - np.floor(np.abs(sequence))) * (10 ** 14), 256))
sequence=sequence[1000:M*N*8+1000]


keys = []
for i in range(1,9):
    keys.append(sequence[N*int(M/2)*i-N*int(M/2):int(M/2)*N*i])

keys = np.array(keys,dtype=int)



def gray_to_DNA(gray_value):
    binary_value = bin(gray_value)[2:].zfill(8)  # 将像素值转换成 8 位二进制数
    DNA = ''
    for i in range(0, 8, 2):
        if binary_value[i:i+2] == '00':
            DNA += 'A'
        elif binary_value[i:i+2] == '11':
            DNA += 'T'
        elif binary_value[i:i+2] == '01':
            DNA += 'G'
        elif binary_value[i:i+2] == '10':
            DNA += 'C'
    return DNA

def DNA_addition(DNA1, DNA2):
    result = ''
    for i in range(4):
        if DNA1[i] == 'A':
            if   DNA2[i] == 'A' :
                 result += 'A'
            elif DNA2[i] == 'G':
                 result += 'G'
            elif DNA2[i] == 'C':   
                 result += 'C'
            elif DNA2[i] == 'T':
                 result += 'T'
        if DNA1[i] == 'G':
           if   DNA2[i] == 'A' :
                result += 'G'
           elif DNA2[i] == 'G':
                result += 'C'
           elif DNA2[i] == 'C':   
                result += 'T'
           elif DNA2[i] == 'T':
                result += 'A'
        if DNA1[i] == 'C':
           if   DNA2[i] == 'A' :
                result += 'C'
           elif DNA2[i] == 'G':
                result += 'T'
           elif DNA2[i] == 'C':   
                result += 'A'
           elif DNA2[i] == 'T':
                result += 'G'
        if DNA1[i] == 'T':
           if   DNA2[i] == 'A' :
                result += 'T'
           elif DNA2[i] == 'G':
                result += 'A'
           elif DNA2[i] == 'C':   
                result += 'G'
           elif DNA2[i] == 'T':
                result += 'C'         
    return result

def DNA_to_gray(DNA):
    binary_value = ''
    for i in range(4):
        if DNA[i] == 'A':
            binary_value += '00'
        elif DNA[i] == 'T':
            binary_value += '11'
        elif DNA[i] == 'G':
            binary_value += '01'
        elif DNA[i] == 'C':
            binary_value += '10'
    gray_value = int(binary_value, 2)
    return gray_value




Limage=(column[:int(M/2)])
Limage=[int(i) for item in Limage for i in item]
Limage = np.array(Limage,dtype=int)

Rimage=(column[int(M/2):])
Rimage=[int(i) for item in Rimage for i in item]
Rimage = np.array(Rimage,dtype=int)



for i in range(1,5):
    NextL = Rimage.copy()  # 下一轮的L是上一轮的R
    now_key = keys[i-1]  # 这一轮加密的密钥
    Rtext_E = np.bitwise_xor(Rimage, now_key)  # 与密钥异或处理
    Rimage = np.bitwise_xor(Rtext_E, Limage)  # 与上一轮L异或处理
    Limage = NextL
    
DNA_KEY= np.zeros(M*int(N/2), dtype=np.int32())
Rimage= np.zeros(M*int(N/2), dtype=np.int32())

for i in range(5,9):
    NextL = Rimage.copy()  # 下一轮的L是上一轮的R
    now_key = keys[i-1]  # 这一轮加密的密钥
    for j in range(1,M*int(N/2)):
        gray_value_1 = Rimage[j]
        gray_value_2 = now_key[j]
        DNA_1 = gray_to_DNA(gray_value_1)
        DNA_2 = gray_to_DNA(gray_value_2)
        DNA_sum = DNA_addition(DNA_1, DNA_2)
        RE=DNA_to_gray(DNA_sum)
        DNA_KEY[j]=RE
    for K in range(1,M*int(N/2)):
        gray_value_1 =  DNA_KEY[K]
        gray_value_2 = Limage[K]
        DNA_1 = gray_to_DNA(gray_value_1)
        DNA_2 = gray_to_DNA(gray_value_2)
        DNA_sum = DNA_addition(DNA_1, DNA_2)
        RE=DNA_to_gray(DNA_sum)
        Rimage[K]=RE
    Limage = NextL
    
Cipherimage = np.concatenate([Rimage, Limage]).reshape(N, M).astype(np.uint8)
cipher_img = Image.fromarray(Cipherimage)

# Show and save image
cipher_img.show()
image_shuffled.show()

