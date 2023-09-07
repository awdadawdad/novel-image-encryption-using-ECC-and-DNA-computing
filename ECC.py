# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:54:25 2023

@author: gaiya
"""

from random import randint

# function for extended Euclidean Algorithm  
def gcdExtended(a, b):  
    # Base Case  
    if a == 0 :   
        return b,0,1
             
    gcd,x1,y1 = gcdExtended(b%a, a)  
     
    # Update x and y using results of recursive  
    # call  
    x = y1 - (b//a) * x1  
    y = x1  
     
    return gcd,x,y 

def inverse(a, n):
    _, x, _ = gcdExtended(a%n, n)
    return x % n

class ECC:
    def __init__(self):
        self.p = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_FFFFFC2F
        self.a = 0
        self.b = 7
        self.n = 0xFFFFFFFF_FFFFFFFF_FFFFFFFF_FFFFFFFE_BAAEDCE6_AF48A03B_BFD25E8C_D0364141
        self.Gx = 0x79BE667E_F9DCBBAC_55A06295_CE870B07_029BFCDB_2DCE28D9_59F2815B_16F81798
        self.Gy = 0x483ADA77_26A3C465_5DA4FBFC_0E1108A8_FD17B448_A6855419_9C47D08F_FB10D4B8
        self.d = 0x483ADA77_26A3C465_5DA4FBFC
        self.Qx, self.Qy = self.multiply(self.d, self.Gx, self.Gy)
    
    def encrypt(self, m):
        while True:
            k = randint(1, self.n - 1)
            x1, y1 = self.multiply(k, self.Gx, self.Gy)
            x2, y2 = self.multiply(k, self.Qx, self.Qy)
            if x2 != 0:
                break
        C = m * x2 % self.n
        return x1, y1, C
    
    def decrypt(self, enc):
        x1, y1, C = enc
        x2, y2 = self.multiply(self.d, x1, y1)
        m = C * inverse(x2, self.n) % self.n
        return m

    def dup(self, x, y):
        la = ((3*x*x + self.a) * inverse(2*y, self.p)) % self.p
        x3 = (la*la - 2*x) % self.p
        y3 = (la * (x-x3) - y) % self.p
        return x3, y3
    
    def add(self, x1, y1, x2, y2):
        if x1 == x2 and y1 == y2:
            return self.dup(x1, y1)
        elif x1 == x2 and (y1 + y2) % self.p == 0:
            return float("inf"), float("inf")
        elif x1 == float("inf") and y1 == float("inf"):
            return x2, y2
        elif x2 == float("inf") and y2 == float("inf"):
            return x1, y1
        la = ((y2-y1) * inverse(x2-x1, self.p)) % self.p
        x3 = (la*la - x1 - x2) % self.p
        y3 = (la * (x1-x3) - y1) % self.p
        return x3, y3
    
    def multiply(self, k, x, y):
        resultX = resultY = float("inf")
        while k > 0:
            if k & 1:
                resultX, resultY = self.add(resultX, resultY, x, y)
            k = k >> 1
            x, y = self.dup(x, y)
        return resultX, resultY

if __name__ == "__main__":
    msg = 0x32996064ffee934eb48f20b83679567a5492e6a804d0835e964efd302328dc2b
    eccBlock = ECC()
    enc = eccBlock.encrypt(msg)
    rec = eccBlock.decrypt(enc)
    print("原文：0x%x" % msg)
    print("密文：X1: (0x%x, \n      0x%x)" % (enc[0], enc[1]))
    print("      C: 0x%x" % enc[2])
    print("解密：0x%x" % rec)