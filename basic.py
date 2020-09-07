# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       basic
   Description :
   Author :         amilyxy
   date：           2020/3/14
-------------------------------------------------
"""
'''
python实现组合数求解
C(A,B) = C(A,B)/C(A,A) = jiec(a,b)/jiec(a,a)
'''
def jiec(a, b):
    res = 1
    for i in range(b, b-a, -1):
        res*=i
    return res

'''
Python实现求两个数的最大公约数
知道怎么写就成，所以只写两种方法：辗转相除法和辗转相减法
'''
class gcdsolution1():
    def gcd(self, p, q):
        tmp = p%q
        while tmp!=0:
            p = q
            q = tmp
            tmp = p%q
        return q

class gcdsolution2():
    def gcd(self, p, q):
        while p!=q:
            if p>q: p-=q
            else: q-=p
        return p

'''
python求解两个数的最小公倍数
方法一：a*b/gcd(a,b) 方法二：老方法（明确一点最小公倍数肯定比最大数大）
'''
def lcm(a, b):
    tmp = max(a, b)
    while tmp%a or tmp%b:
        tmp+=1
    return tmp

'''
python判断一个数是否为质数
质数：除了1和该数自身外，无法被其他自然数整除的数
方法一：时间复杂度为O(sqrt(n))
'''
import math
def is_prime(a):
    if a<=1: return False
    for i in range(2, math.ceil(math.sqrt(a))+1):
        if not a%i:
            return False
    return True

# 扩展 求n以内的所有质数 使用厄拉多塞筛法
# 见math204

'''
python对一个数因式分解
方法1：总感觉第一个比较耗时？？？
方法2：改进思路：①从最小的质数开始一直除，②除2之外所有的质数都是奇数 ③被3无限整除之后，不可能再被9整除，起到了筛选作用
'''
def factorization1(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    return factor

def factorization2(num):
    factor = []
    k = 0
    while num>1:
        k+=2
        while num>1 and num%k == 0:
            factor.append(k)
            num = num/k
        if k == 2: k = 1
    return factor

'''
# 可测时间
import time
st = time.perf_counter()
print(factorization1(707829217))
et = time.perf_counter()
print("用时:", et - st)
'''

'''
python 快速幂
'''
def powxn(x, n):
    neg = 0
    if n<0:
        neg = 1
        n = abs(n)
    res = 1
    while n>0:
        if n&0x01:
            res *= x
        x *= x
        n>>=1
    if neg: return 1/res
    return res

'''
python求平方根
① 二分法 ②牛顿法
'''
def sqrt_binary(num):
  y=num/2.0
  low=0.0
  up=num*1.0
  while abs(y*y-num)>0.00000001:
    if (y*y>num):
      up=y
      y=low+(y-low)/2
    else:
      low=y
      y=up-(up-y)/2
  return y

def sqrt_newton(num):
  y=num/2.0
  while abs(y*y-num)>0.00000001:
    y=((y*1.0)+(1.0*num)/y)/2.0000
  return y













